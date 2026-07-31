"""Qualify GPU IVF-PQ candidate search on the fixed balanced-120M rung."""
from __future__ import annotations

import gc
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
    INDEX_PATH,
    INDEX_SHA256,
    K,
    SOURCE_ROWS,
    compact_to_global,
    global_to_compact,
)
from basemap.round0065_substrates import (
    subset_spec,
    validate_scale_substrate,
)
from basemap.round0077_quality import (
    MEAN_RECALL_FLOOR,
    NPROBE_GRID,
    QUALIFICATION_SCHEMA,
    ROUND_ID,
    Round0077Error,
    seal,
)
from experiments.round0049_nodes import (
    INDEX_SEARCH_WIDTH,
    SEARCH_WIDTH,
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
QUALITY_SEED = 77
BENCHMARK_ROWS = 10_000
RUNTIME_SPEC = os.path.join(
    os.path.dirname(__file__),
    "round0060_runtime.json",
)


def _build_filtered_index(
    *,
    faiss: Any,
    destination_path: str,
    excluded_global: np.ndarray,
) -> tuple[Any, dict[str, Any]]:
    started = time.monotonic()
    source = faiss.read_index(INDEX_PATH)
    if (
        type(source).__name__ != "IndexIVFPQ"
        or int(source.ntotal) != SOURCE_ROWS
        or int(source.d) != DIMENSION
        or int(source.nlist) != 8_192
        or int(source.code_size) != 48
        or int(source.pq.M) != 48
        or int(source.pq.nbits) != 8
    ):
        raise Round0077Error("registered source IVF-PQ geometry changed")
    destination = faiss.clone_index(source)
    destination.reset()
    for start, stop in INTERVALS:
        source.copy_subset_to(
            destination,
            faiss.InvertedLists.SUBSET_TYPE_ID_RANGE,
            start,
            stop,
        )
    copied = int(destination.ntotal)
    if copied != ROW_COUNT:
        raise Round0077Error(
            f"balanced range copy produced {copied}, wanted {ROW_COUNT}"
        )
    selector = faiss.IDSelectorBatch(
        np.ascontiguousarray(excluded_global, dtype=np.int64)
    )
    removed = int(destination.remove_ids(selector))
    expected_excluded = int(
        ELIGIBILITY_SUMMARY["excluded_row_count"]
    )
    expected_retained = int(
        ELIGIBILITY_SUMMARY["retained_row_count"]
    )
    if (
        removed != expected_excluded
        or int(destination.ntotal) != expected_retained
    ):
        raise Round0077Error("physical 120M eligibility filtering changed")
    temporary = destination_path + ".partial"
    if os.path.exists(temporary) or os.path.exists(destination_path):
        raise Round0077Error("filtered 120M index output exists")
    faiss.write_index(destination, temporary)
    os.replace(temporary, destination_path)
    os.chmod(destination_path, 0o444)
    signature = expected_input_signature(destination_path)
    performance = {
        "wall_seconds": time.monotonic() - started,
        "source_ntotal": int(source.ntotal),
        "balanced_range_rows": copied,
        "physically_removed_rows": removed,
        "filtered_ntotal": int(destination.ntotal),
        "nlist": int(destination.nlist),
        "code_size": int(destination.code_size),
        "pq_m": int(destination.pq.M),
        "pq_nbits": int(destination.pq.nbits),
        "index": signature,
    }
    del source, selector
    gc.collect()
    return destination, performance


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
        raise Round0077Error("balanced-120M quality queries are invalid")
    values /= norms
    return np.ascontiguousarray(values)


def _search_and_rerank(
    *,
    index: Any,
    nprobe: int,
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
    _distances, raw = index.search(queries, INDEX_SEARCH_WIDTH)
    search_seconds = time.monotonic() - started
    shortlist, self_seen = _clean_search(
        raw,
        global_sources=compact_to_global_fn(compact_sources),
        candidate_count=SEARCH_WIDTH,
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
        "search_seconds": search_seconds,
        "queries": len(queries),
        "queries_per_second": len(queries) / search_seconds,
        "self_returned": self_seen,
        "all_shortlist_rows_in_range": bool(
            np.all((shortlist >= 0) & (shortlist < len(encoded)))
        ),
        "exact_rerank": rerank,
    }


def run_qualification(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    import faiss

    output = create_fresh_directory(
        str(job["outputs"][0]),
        label="Round 0077 balanced-120M GPU qualification",
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
        raise Round0077Error("balanced-120M eligibility accounting changed")
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
    index_signature = expected_input_signature(INDEX_PATH)
    if index_signature["sha256"] != INDEX_SHA256:
        raise Round0077Error("registered 150M IVF-PQ bytes changed")
    runtime = _runtime_stamp(
        str(job["runtime_spec"]),
        str(job["runtime_spec_sha256"]),
    )
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

    filtered_path = os.path.join(
        output,
        "balanced-120m-retained.ivfpq",
    )
    excluded_global = compact_to_global(
        excluded,
        intervals=INTERVALS,
    )
    filtered, filtering = _build_filtered_index(
        faiss=faiss,
        destination_path=filtered_path,
        excluded_global=excluded_global,
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

    rows_by_nprobe: dict[str, Any] = {}
    selected_nprobe: int | None = None
    for nprobe in NPROBE_GRID:
        gpu.nprobe = nprobe
        adapter = _GpuSearchAdapter(gpu, nprobe)
        selected, performance = _search_and_rerank(
            index=adapter,
            nprobe=nprobe,
            queries=sample_queries,
            compact_sources=sample,
            encoded=encoded,
            scales=scales,
        )
        overlap = (
            selected[:, :, None] == exact[:, None, :]
        ).any(axis=2).sum(axis=1) / K
        clear_mean = (
            float(overlap[unambiguous].mean())
            if np.any(unambiguous)
            else 0.0
        )
        row = {
            "nprobe": nprobe,
            "mean_recall_at_15": float(overlap.mean()),
            "p10_recall_at_15": float(np.percentile(overlap, 10)),
            "mean_recall_at_15_unambiguous": clear_mean,
            "p10_recall_at_15_unambiguous": (
                float(np.percentile(overlap[unambiguous], 10))
                if np.any(unambiguous)
                else 0.0
            ),
            "passes_mean_floor": clear_mean >= MEAN_RECALL_FLOOR,
            "performance": performance,
        }
        rows_by_nprobe[str(nprobe)] = row
        if row["passes_mean_floor"] and selected_nprobe is None:
            selected_nprobe = nprobe

    projection: dict[str, Any] | None = None
    benchmark: dict[str, Any] | None = None
    if selected_nprobe is not None:
        first = np.arange(min(100_000, ROW_COUNT), dtype=np.int64)
        retained = first[~_membership(excluded, first)]
        benchmark_rows = retained[:BENCHMARK_ROWS]
        benchmark_queries = _queries(encoded, scales, benchmark_rows)
        gpu.nprobe = selected_nprobe
        adapter = _GpuSearchAdapter(gpu, selected_nprobe)
        _selected, benchmark = _search_and_rerank(
            index=adapter,
            nprobe=selected_nprobe,
            queries=benchmark_queries,
            compact_sources=benchmark_rows,
            encoded=encoded,
            scales=scales,
        )
        projection = _project_full_graph_hours(
            row_count=ROW_COUNT,
            benchmark_rows=len(benchmark_rows),
            gpu_search_seconds=float(benchmark["search_seconds"]),
            gpu_rerank_seconds=float(
                benchmark["exact_rerank"]["wall_seconds"]
            ),
            clone_seconds=clone_seconds,
        )

    checks = {
        "runtime_matches": all(
            value is True for value in runtime["checks"].values()
        ),
        "fixed_registered_120m_universe": (
            substrate["manifest"]["tier"] == TIER
            and substrate["manifest"]["row_count"] == ROW_COUNT
        ),
        "filtered_candidate_count": (
            filtering["filtered_ntotal"] == expected_retained
        ),
        "unambiguous_fraction_at_least_0_90": (
            float(unambiguous.mean()) >= 0.90
        ),
        "passing_nprobe_selected": selected_nprobe is not None,
        "smallest_passing_nprobe_selected": (
            selected_nprobe
            == next(
                (
                    value
                    for value in NPROBE_GRID
                    if rows_by_nprobe[str(value)][
                        "passes_mean_floor"
                    ]
                ),
                None,
            )
        ),
        "no_training_performed": True,
        "no_scale_decision_made": True,
    }
    passed = all(value is True for value in checks.values())
    selected_row = (
        rows_by_nprobe[str(selected_nprobe)]
        if selected_nprobe is not None
        else None
    )
    body = {
        "schema": QUALIFICATION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "validity_passed": passed,
        "failed_checks": sorted(
            key for key, value in checks.items()
            if value is not True
        ),
        "training_performed": False,
        "optimizer_updates": 0,
        "scale_decision_made": False,
        "tier": TIER,
        "substrate": substrate["signature"],
        "eligibility": outputs["eligibility"],
        "source_index": index_signature,
        "runtime": runtime,
        "nprobe_grid": list(NPROBE_GRID),
        "selected_nprobe": selected_nprobe,
        "rows_by_nprobe": rows_by_nprobe,
        "candidate_universe": {
            "balanced_intervals": [list(value) for value in INTERVALS],
            "physical_exclusions": expected_excluded,
            "retained_rows": expected_retained,
            "filtered_index": filtering["index"],
        },
        "gpu_index": {
            "implementation": "faiss-classic-GpuIndexIVFPQ",
            "indices_options": "INDICES_64_BIT",
            "use_float16": False,
            "use_precomputed": True,
            "temporary_memory_bytes": 1 << 30,
            "clone_seconds": clone_seconds,
        },
        "quality": {
            "sample_rows": len(sample),
            "sample_seed": QUALITY_SEED,
            "sample_sha256": sha256_bytes(sample.tobytes()),
            "boundary_ties": int(ties.sum()),
            "unambiguous_fraction": float(unambiguous.mean()),
            "selected": selected_row,
            "floor": MEAN_RECALL_FLOOR,
        },
        "benchmark": {
            "selected_nprobe": selected_nprobe,
            "measurement": benchmark,
            "projected_full_graph_hours": projection,
            "projection_is_planning_not_acceptance": True,
        },
        "performance": {
            "filtered_index_build": {
                key: value
                for key, value in filtering.items()
                if key != "index"
            },
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
        "gpu-ivfpq-qualification-v1.json",
    )
    atomic_write_new_json(path, receipt, immutable=True)
    del resources, gpu, filtered
    if not passed:
        raise Round0077Error(
            "balanced-120M GPU qualification failed: "
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
        raise Round0077Error("R0077 handler received another queue")
    selected = job if job is not None else active.get("job") or {}
    if selected.get("action") != "qualify_balanced_120m_gpu_ivfpq":
        raise Round0077Error("R0077 accepts only GPU qualification")
    return run_qualification(active, selected)
