"""Recover a quality-qualified IVF-PQ policy on the fixed 120M universe."""
from __future__ import annotations

import os
import resource
import time
from functools import partial
from typing import Any, Mapping

import numpy as np

from basemap.artifact_identity import (
    expected_input_signature,
    sha256_bytes,
)
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0049_program import (
    DIMENSION,
    K,
    SOURCE_ROWS,
    compact_to_global,
    global_to_compact,
)
from basemap.round0065_substrates import (
    subset_spec,
    validate_scale_substrate,
)
from basemap.round0081_quality import (
    MEAN_RECALL_FLOOR,
    POLICY_GRID,
    QUALIFICATION_SCHEMA,
    ROUND_ID,
    Round0081Error,
    _selected_cell,
    cell_key,
    seal,
)
from experiments.round0049_nodes import (
    _clean_search,
    _exact_representative_truth,
    _exact_rerank_shortlist,
    _membership,
    _sample_retained_rows,
)
from experiments.round0059_nodes import (
    _GpuSearchAdapter,
    _project_full_graph_hours,
    _runtime_stamp,
)


TIER = "120m"
SPEC = subset_spec(TIER)
ROW_COUNT = int(SPEC["row_count"])
INTERVALS = tuple(
    (int(start), int(stop))
    for start, stop in SPEC["intervals"]
)
ELIGIBILITY_SUMMARY = dict(SPEC["eligibility_summary"])
QUALITY_SAMPLE_ROWS = 4_096
QUALITY_SEED = 81
BENCHMARK_ROWS = 10_000
BENCHMARK_REPEATS = 3
BENCHMARK_WARMUP_ROWS = 512
RUNTIME_SPEC = os.path.join(
    os.path.dirname(__file__),
    "round0060_runtime.json",
)


def _queries(
    encoded: np.ndarray,
    scales: np.ndarray,
    rows: np.ndarray,
) -> np.ndarray:
    values = (
        np.asarray(encoded[rows], dtype=np.float32)
        * np.asarray(scales[rows], dtype=np.float32)[:, None]
    )
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (
        not np.isfinite(values).all()
        or not np.isfinite(norms).all()
        or np.any(norms <= 0)
    ):
        raise Round0081Error(f"balanced-{TIER} policy queries are invalid")
    values /= norms
    return np.ascontiguousarray(values)


def _search_and_rerank(
    *,
    index: Any,
    nprobe: int,
    shortlist_width: int,
    queries: np.ndarray,
    compact_sources: np.ndarray,
    encoded: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    compact_to_global_fn = partial(
        compact_to_global,
        intervals=INTERVALS,
    )
    global_to_compact_fn = partial(
        global_to_compact,
        intervals=INTERVALS,
    )
    started = time.monotonic()
    _distances, raw = index.search(queries, shortlist_width + 1)
    search_seconds = time.monotonic() - started
    shortlist, self_seen = _clean_search(
        raw,
        global_sources=compact_to_global_fn(compact_sources),
        candidate_count=shortlist_width,
        source_rows=SOURCE_ROWS,
        global_to_compact_fn=global_to_compact_fn,
    )
    selected, rerank = _exact_rerank_shortlist(
        queries=queries,
        shortlist=shortlist,
        encoded=encoded,
        scales=scales,
    )
    return selected, {
        "nprobe": nprobe,
        "shortlist_width": shortlist_width,
        "search_seconds": search_seconds,
        "queries": len(queries),
        "queries_per_second": len(queries) / search_seconds,
        "self_returned": self_seen,
        "all_shortlist_rows_in_range": bool(
            np.all((shortlist >= 0) & (shortlist < len(encoded)))
        ),
        "exact_rerank": rerank,
    }


def _benchmark_cell(
    *,
    gpu: Any,
    nprobe: int,
    shortlist_width: int,
    queries: np.ndarray,
    sources: np.ndarray,
    encoded: np.ndarray,
    scales: np.ndarray,
) -> dict[str, Any]:
    gpu.nprobe = nprobe
    adapter = _GpuSearchAdapter(gpu, nprobe)
    warm_rows = min(BENCHMARK_WARMUP_ROWS, len(sources))
    _search_and_rerank(
        index=adapter,
        nprobe=nprobe,
        shortlist_width=shortlist_width,
        queries=queries[:warm_rows],
        compact_sources=sources[:warm_rows],
        encoded=encoded,
        scales=scales,
    )
    repeats: list[dict[str, Any]] = []
    for _repeat in range(BENCHMARK_REPEATS):
        _selected, performance = _search_and_rerank(
            index=adapter,
            nprobe=nprobe,
            shortlist_width=shortlist_width,
            queries=queries,
            compact_sources=sources,
            encoded=encoded,
            scales=scales,
        )
        search = float(performance["search_seconds"])
        rerank = float(
            performance["exact_rerank"]["wall_seconds"]
        )
        repeats.append({
            "search_seconds": search,
            "rerank_seconds": rerank,
            "total_seconds": search + rerank,
            "self_returned": performance["self_returned"],
        })
    median_search = float(np.median([
        row["search_seconds"] for row in repeats
    ]))
    median_rerank = float(np.median([
        row["rerank_seconds"] for row in repeats
    ]))
    median_total = float(np.median([
        row["total_seconds"] for row in repeats
    ]))
    return {
        "rows": len(sources),
        "warmup_rows": warm_rows,
        "repeats": repeats,
        "median_search_seconds": median_search,
        "median_rerank_seconds": median_rerank,
        "median_total_seconds": median_total,
        "median_wall_seconds_per_query": median_total / len(sources),
    }


def run_qualification(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    import faiss

    output = create_fresh_directory(
        str(job["outputs"][0]),
        label=f"Round {ROUND_ID} balanced-{TIER} policy qualification",
    )
    substrate = validate_scale_substrate(
        str(job["substrate_manifest"]),
        tier=TIER,
        expected_sha256=str(job["substrate_manifest_sha256"]),
    )
    outputs = substrate["manifest"]["outputs"]
    eligibility = substrate["eligibility"]
    excluded = np.asarray(eligibility["excluded_rows"], dtype=np.int64)
    expected_excluded = int(
        ELIGIBILITY_SUMMARY["excluded_row_count"]
    )
    expected_retained = int(
        ELIGIBILITY_SUMMARY["retained_row_count"]
    )
    if (
        len(excluded) != expected_excluded
        or ROW_COUNT - len(excluded) != expected_retained
    ):
        raise Round0081Error(
            f"balanced-{TIER} eligibility accounting changed"
        )
    encoded = np.memmap(
        outputs["int8"]["canonical_path"],
        dtype=np.int8,
        mode="r",
        shape=(ROW_COUNT, DIMENSION),
    )
    scales = np.memmap(
        outputs["scales"]["canonical_path"],
        dtype="<f2",
        mode="r",
        shape=(ROW_COUNT,),
    )
    runtime = _runtime_stamp(
        str(job["runtime_spec"]),
        str(job["runtime_spec_sha256"]),
    )
    filtered_signature = expected_input_signature(
        str(job["filtered_index"])
    )
    if filtered_signature["sha256"] != str(
        job["filtered_index_sha256"]
    ):
        raise Round0081Error("reviewed R0077 filtered index changed")

    sample = _sample_retained_rows(
        excluded,
        count=QUALITY_SAMPLE_ROWS,
        seed=QUALITY_SEED,
        row_count=ROW_COUNT,
    )
    exact, ties, exact_performance = _exact_representative_truth(
        encoded=encoded,
        scales=scales,
        excluded=excluded,
        sample=sample,
        row_count=ROW_COUNT,
    )
    unambiguous = ~ties
    sample_queries = _queries(encoded, scales, sample)

    first = np.arange(min(100_000, ROW_COUNT), dtype=np.int64)
    retained = first[~_membership(excluded, first)]
    benchmark_sources = retained[:BENCHMARK_ROWS]
    if len(benchmark_sources) != BENCHMARK_ROWS:
        raise Round0081Error("fixed policy benchmark rows changed")
    benchmark_queries = _queries(
        encoded,
        scales,
        benchmark_sources,
    )

    filtered = faiss.read_index(filtered_signature["canonical_path"])
    if (
        type(filtered).__name__ != "IndexIVFPQ"
        or int(filtered.ntotal) != expected_retained
        or int(filtered.d) != DIMENSION
        or int(filtered.nlist) != 8_192
        or int(filtered.code_size) != 48
        or int(filtered.pq.M) != 48
        or int(filtered.pq.nbits) != 8
    ):
        raise Round0081Error(
            f"reviewed {TIER} filtered index geometry changed"
        )

    resources = faiss.StandardGpuResources()
    resources.setTempMemory(1 << 30)
    options = faiss.GpuClonerOptions()
    options.indicesOptions = faiss.INDICES_64_BIT
    options.useFloat16 = False
    options.usePrecomputed = True
    clone_started = time.monotonic()
    gpu = faiss.index_cpu_to_gpu(resources, 0, filtered, options)
    clone_seconds = time.monotonic() - clone_started

    cells: dict[str, Any] = {}
    for nprobe, shortlist_width in POLICY_GRID:
        gpu.nprobe = nprobe
        adapter = _GpuSearchAdapter(gpu, nprobe)
        selected, performance = _search_and_rerank(
            index=adapter,
            nprobe=nprobe,
            shortlist_width=shortlist_width,
            queries=sample_queries,
            compact_sources=sample,
            encoded=encoded,
            scales=scales,
        )
        overlap = (
            selected[:, :, None] == exact[:, None, :]
        ).any(axis=2).sum(axis=1) / K
        clear = overlap[unambiguous]
        clear_mean = float(clear.mean()) if len(clear) else 0.0
        cell = {
            "nprobe": nprobe,
            "shortlist_width": shortlist_width,
            "mean_recall_at_15": float(overlap.mean()),
            "p10_recall_at_15": float(np.percentile(overlap, 10)),
            "mean_recall_at_15_unambiguous": clear_mean,
            "p10_recall_at_15_unambiguous": (
                float(np.percentile(clear, 10))
                if len(clear)
                else 0.0
            ),
            "passes_mean_floor": clear_mean >= MEAN_RECALL_FLOOR,
            "quality_performance": performance,
            "benchmark": None,
            "projected_full_graph_hours": None,
        }
        if cell["passes_mean_floor"]:
            benchmark = _benchmark_cell(
                gpu=gpu,
                nprobe=nprobe,
                shortlist_width=shortlist_width,
                queries=benchmark_queries,
                sources=benchmark_sources,
                encoded=encoded,
                scales=scales,
            )
            cell["benchmark"] = benchmark
            cell["projected_full_graph_hours"] = (
                _project_full_graph_hours(
                    row_count=ROW_COUNT,
                    benchmark_rows=BENCHMARK_ROWS,
                    gpu_search_seconds=float(
                        benchmark["median_search_seconds"]
                    ),
                    gpu_rerank_seconds=float(
                        benchmark["median_rerank_seconds"]
                    ),
                    clone_seconds=clone_seconds,
                )
            )
        cells[cell_key(nprobe, shortlist_width)] = cell

    selected_cell = _selected_cell({"cells": cells})
    checks = {
        "runtime_matches": all(
            value is True for value in runtime["checks"].values()
        ),
        f"fixed_registered_{TIER}_universe": (
            substrate["manifest"]["tier"] == TIER
            and substrate["manifest"]["row_count"] == ROW_COUNT
        ),
        "filtered_candidate_count": (
            int(filtered.ntotal) == expected_retained
        ),
        "all_registered_cells_present": (
            set(cells)
            == {
                cell_key(nprobe, width)
                for nprobe, width in POLICY_GRID
            }
        ),
        "unambiguous_fraction_at_least_0_90": (
            float(unambiguous.mean()) >= 0.90
        ),
        "passing_policy_selected": selected_cell is not None,
        "minimum_measured_wall_policy_selected": (
            selected_cell is not None
            and selected_cell == _selected_cell({"cells": cells})
        ),
        "no_training_performed": True,
        "no_scale_decision_made": True,
    }
    passed = all(value is True for value in checks.values())
    body = {
        "schema": QUALIFICATION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "validity_passed": passed,
        "failed_checks": sorted(
            key
            for key, value in checks.items()
            if value is not True
        ),
        "training_performed": False,
        "optimizer_updates": 0,
        "scale_decision_made": False,
        "tier": TIER,
        "substrate": substrate["signature"],
        "eligibility": outputs["eligibility"],
        "filtered_index": filtered_signature,
        "runtime": runtime,
        "policy_grid": [
            {"nprobe": nprobe, "shortlist_width": width}
            for nprobe, width in POLICY_GRID
        ],
        "policy_selector": (
            "lowest median three-repeat 10000-query search-plus-rerank "
            "wall among cells with mean unambiguous exact-reranked "
            f"recall@15 at least {MEAN_RECALL_FLOOR:.2f}; "
            "ties by shortlist width then nprobe"
        ),
        "selected": selected_cell,
        "cells": cells,
        "quality": {
            "sample_rows": len(sample),
            "sample_seed": QUALITY_SEED,
            "sample_sha256": sha256_bytes(sample.tobytes()),
            "boundary_ties": int(ties.sum()),
            "unambiguous_fraction": float(unambiguous.mean()),
            "floor": MEAN_RECALL_FLOOR,
        },
        "gpu_index": {
            "implementation": "faiss-classic-GpuIndexIVFPQ",
            "indices_options": "INDICES_64_BIT",
            "use_float16": False,
            "use_precomputed": True,
            "temporary_memory_bytes": 1 << 30,
            "clone_seconds": clone_seconds,
        },
        "performance": {
            "exact_truth": exact_performance,
            "peak_rss_gib": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024 ** 2)
            ),
        },
        "checks": checks,
    }
    receipt = seal(body)
    path = os.path.join(
        output,
        "gpu-ivfpq-policy-qualification-v1.json",
    )
    atomic_write_new_json(path, receipt, immutable=True)
    del resources, gpu, filtered
    if not passed:
        raise Round0081Error(
            f"balanced-{TIER} policy qualification failed: "
            + ", ".join(receipt["failed_checks"])
        )
    return {
        **receipt,
        "receipt": expected_input_signature(path),
    }


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0081Error("R0081 handler received another queue")
    selected = job if job is not None else active.get("job") or {}
    if selected.get("action") != "qualify_balanced_120m_gpu_ivfpq_policy":
        raise Round0081Error("R0081 accepts only policy qualification")
    return run_qualification(active, selected)
