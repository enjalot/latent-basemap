"""Split and qualify sharded candidate search on the balanced 150M universe."""
from __future__ import annotations

import json
import os
import resource
import time
from functools import partial
from typing import Any, Mapping

import numpy as np

from basemap.artifact_identity import expected_input_signature, sha256_bytes
from basemap.output_safety import atomic_write_new_json, create_fresh_directory
from basemap.round0049_program import (
    DIMENSION,
    K,
    SOURCE_ROWS,
    compact_to_global,
    global_to_compact,
)
from basemap.round0086_program import validate_substrate
from basemap.round0093_policy import load_decision as load_r0093_decision
from basemap.round0094_sharded_search import (
    DECISION_SCHEMA,
    MAX_MEDIAN_SECONDS_PER_QUERY,
    MEAN_RECALL_FLOOR,
    POLICY_GRID,
    QUALIFICATION_SCHEMA,
    RETAINED_ROWS,
    ROUND_ID,
    ROW_COUNT,
    SHARD_SPECS,
    SPLIT_SCHEMA,
    Round0094Error,
    cell_key,
    load_split_receipt,
    seal,
    select_cell,
)
from experiments.round0049_nodes import (
    _clean_search,
    _exact_representative_truth,
    _exact_rerank_shortlist,
    _membership,
    _sample_retained_rows,
)
from experiments.round0059_nodes import _GpuSearchAdapter, _runtime_stamp


INTERVALS = ((0, ROW_COUNT),)
QUALITY_ROWS = 4_096
QUALITY_SEED = 86
BENCHMARK_ROWS = 10_000
BENCHMARK_WARMUP_ROWS = 512
BENCHMARK_REPEATS = 3


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
        raise Round0094Error("R0094 queries are invalid")
    values /= norms
    return np.ascontiguousarray(values)


def run_split(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    import faiss

    output = create_fresh_directory(
        str(job["outputs"][0]),
        label="R0094 corpus-sharded retained index",
    )
    source_signature = expected_input_signature(str(job["filtered_index"]))
    if source_signature["sha256"] != str(job["filtered_index_sha256"]):
        raise Round0094Error("R0093-reviewed filtered-index bytes changed")
    substrate = validate_substrate(
        str(job["substrate_manifest"]),
        expected_sha256=str(job["substrate_manifest_sha256"]),
    )
    r0093 = load_r0093_decision(
        str(job["r0093_decision"]),
        expected_sha256=str(job["r0093_decision_sha256"]),
    )
    if (
        r0093["receipt"].get("filtered_index") != source_signature
        or r0093["receipt"].get("substrate") != substrate["signature"]
    ):
        raise Round0094Error("R0093 decision does not bind split inputs")

    source = faiss.read_index(source_signature["canonical_path"])
    if (
        type(source).__name__ != "IndexIVFPQ"
        or int(source.ntotal) != RETAINED_ROWS
        or int(source.d) != DIMENSION
        or int(source.nlist) != 8_192
        or int(source.code_size) != 48
        or int(source.pq.M) != 48
        or int(source.pq.nbits) != 8
    ):
        raise Round0094Error("R0094 source-index geometry changed")

    started = time.monotonic()
    shards: dict[str, Any] = {}
    for name, spec in SHARD_SPECS.items():
        index = faiss.clone_index(source)
        index.reset()
        source.copy_subset_to(
            index,
            faiss.InvertedLists.SUBSET_TYPE_ID_RANGE,
            spec["start"],
            spec["stop"],
        )
        if int(index.ntotal) != spec["retained_rows"]:
            raise Round0094Error(f"{name} shard count changed")
        path = os.path.join(output, f"{name}.ivfpq")
        partial_path = path + ".partial"
        faiss.write_index(index, partial_path)
        os.replace(partial_path, path)
        os.chmod(path, 0o444)
        shards[name] = {
            **spec,
            "ntotal": int(index.ntotal),
            "index": expected_input_signature(path),
        }
        del index
    if sum(value["ntotal"] for value in shards.values()) != RETAINED_ROWS:
        raise Round0094Error("R0094 shard totals do not close")
    body = {
        "schema": SPLIT_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "source_index": source_signature,
        "substrate": substrate["signature"],
        "r0093_decision": r0093["signature"],
        "shards": shards,
        "global_ids_preserved": True,
        "disjoint_complete_id_ranges": True,
        "training_performed": False,
        "optimizer_updates": 0,
        "performance": {
            "wall_seconds": time.monotonic() - started,
        },
    }
    receipt = seal(body)
    path = os.path.join(output, "split-receipt.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def _search_and_rerank(
    *,
    indices: list[Any],
    nprobe: int,
    width_per_shard: int,
    queries: np.ndarray,
    sources: np.ndarray,
    encoded: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    global_sources = compact_to_global(sources, intervals=INTERVALS)
    global_to_compact_fn = partial(
        global_to_compact,
        intervals=INTERVALS,
    )
    shortlists: list[np.ndarray] = []
    shard_search_seconds: list[float] = []
    self_seen = 0
    for index in indices:
        index.index.nprobe = nprobe
        index.nprobe = nprobe
        started = time.monotonic()
        _distances, raw = index.search(queries, width_per_shard + 1)
        shard_search_seconds.append(time.monotonic() - started)
        shortlist, seen = _clean_search(
            raw,
            global_sources=global_sources,
            candidate_count=width_per_shard,
            source_rows=SOURCE_ROWS,
            global_to_compact_fn=global_to_compact_fn,
        )
        shortlists.append(shortlist)
        self_seen += seen
    combined = np.ascontiguousarray(np.concatenate(shortlists, axis=1))
    selected, rerank = _exact_rerank_shortlist(
        queries=queries,
        shortlist=combined,
        encoded=encoded,
        scales=scales,
    )
    return selected, {
        "nprobe_per_shard": nprobe,
        "width_per_shard": width_per_shard,
        "total_shortlist_width": int(combined.shape[1]),
        "shard_search_seconds": shard_search_seconds,
        "search_seconds": float(sum(shard_search_seconds)),
        "self_returned": self_seen,
        "exact_rerank": rerank,
    }


def _benchmark(
    *,
    indices: list[Any],
    nprobe: int,
    width_per_shard: int,
    queries: np.ndarray,
    sources: np.ndarray,
    encoded: np.ndarray,
    scales: np.ndarray,
) -> dict[str, Any]:
    warm = min(BENCHMARK_WARMUP_ROWS, len(sources))
    _search_and_rerank(
        indices=indices,
        nprobe=nprobe,
        width_per_shard=width_per_shard,
        queries=queries[:warm],
        sources=sources[:warm],
        encoded=encoded,
        scales=scales,
    )
    repeats = []
    for _repeat in range(BENCHMARK_REPEATS):
        _selected, performance = _search_and_rerank(
            indices=indices,
            nprobe=nprobe,
            width_per_shard=width_per_shard,
            queries=queries,
            sources=sources,
            encoded=encoded,
            scales=scales,
        )
        search = float(performance["search_seconds"])
        rerank = float(performance["exact_rerank"]["wall_seconds"])
        repeats.append({
            "search_seconds": search,
            "rerank_seconds": rerank,
            "total_seconds": search + rerank,
            "shard_search_seconds": performance["shard_search_seconds"],
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
        "warmup_rows": warm,
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
        label="R0094 sharded-search qualification",
    )
    substrate = validate_substrate(
        str(job["substrate_manifest"]),
        expected_sha256=str(job["substrate_manifest_sha256"]),
    )
    filtered = expected_input_signature(str(job["filtered_index"]))
    if filtered["sha256"] != str(job["filtered_index_sha256"]):
        raise Round0094Error("R0094 source index changed")
    r0093 = load_r0093_decision(
        str(job["r0093_decision"]),
        expected_sha256=str(job["r0093_decision_sha256"]),
    )
    runtime = _runtime_stamp(
        str(job["runtime_spec"]),
        str(job["runtime_spec_sha256"]),
    )
    split_path = os.path.join(str(job["split_root"]), "split-receipt.json")
    split = load_split_receipt(
        split_path,
        expected_source=filtered,
        expected_release_sha=active["manifest"]["release_sha"],
    )
    if (
        split["receipt"].get("substrate") != substrate["signature"]
        or split["receipt"].get("r0093_decision") != r0093["signature"]
    ):
        raise Round0094Error("R0094 split receipt lineage changed")

    outputs = substrate["manifest"]["outputs"]
    excluded = np.asarray(
        substrate["eligibility"]["excluded_rows"], dtype=np.int64
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
    sample = _sample_retained_rows(
        excluded,
        count=QUALITY_ROWS,
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
    first = np.arange(100_000, dtype=np.int64)
    benchmark_sources = first[~_membership(excluded, first)][:BENCHMARK_ROWS]
    if len(benchmark_sources) != BENCHMARK_ROWS:
        raise Round0094Error("R0094 benchmark source set changed")
    benchmark_queries = _queries(encoded, scales, benchmark_sources)

    gpu_resources = []
    gpu_indices = []
    clone_seconds = []
    options = faiss.GpuClonerOptions()
    options.indicesOptions = faiss.INDICES_64_BIT
    options.useFloat16 = False
    options.usePrecomputed = True
    for name in SHARD_SPECS:
        declared = split["receipt"]["shards"][name]["index"]
        actual = expected_input_signature(declared["canonical_path"])
        if actual != declared:
            raise Round0094Error(f"{name} shard bytes changed")
        cpu = faiss.read_index(actual["canonical_path"])
        if (
            type(cpu).__name__ != "IndexIVFPQ"
            or int(cpu.ntotal) != SHARD_SPECS[name]["retained_rows"]
            or int(cpu.nlist) != 8_192
            or int(cpu.code_size) != 48
        ):
            raise Round0094Error(f"{name} shard geometry changed")
        resource = faiss.StandardGpuResources()
        resource.setTempMemory(1 << 29)
        started = time.monotonic()
        gpu = faiss.index_cpu_to_gpu(resource, 0, cpu, options)
        clone_seconds.append(time.monotonic() - started)
        gpu_resources.append(resource)
        gpu_indices.append(_GpuSearchAdapter(gpu, 32))

    cells: dict[str, Any] = {}
    for nprobe, width in POLICY_GRID:
        selected, performance = _search_and_rerank(
            indices=gpu_indices,
            nprobe=nprobe,
            width_per_shard=width,
            queries=sample_queries,
            sources=sample,
            encoded=encoded,
            scales=scales,
        )
        overlap = (
            selected[:, :, None] == exact[:, None, :]
        ).any(axis=2).sum(axis=1) / K
        clear = overlap[unambiguous]
        clear_mean = float(clear.mean()) if len(clear) else 0.0
        cell = {
            "nprobe_per_shard": nprobe,
            "width_per_shard": width,
            "total_shortlist_width": width * len(SHARD_SPECS),
            "mean_recall_at_15": float(overlap.mean()),
            "mean_recall_at_15_unambiguous": clear_mean,
            "p10_recall_at_15_unambiguous": (
                float(np.percentile(clear, 10)) if len(clear) else 0.0
            ),
            "passes_mean_floor": clear_mean >= MEAN_RECALL_FLOOR,
            "passes_performance_ceiling": False,
            "quality_performance": performance,
            "benchmark": None,
            "projected_full_graph_hours": None,
        }
        if cell["passes_mean_floor"]:
            benchmark = _benchmark(
                indices=gpu_indices,
                nprobe=nprobe,
                width_per_shard=width,
                queries=benchmark_queries,
                sources=benchmark_sources,
                encoded=encoded,
                scales=scales,
            )
            per_query = float(
                benchmark["median_wall_seconds_per_query"]
            )
            cell["benchmark"] = benchmark
            cell["passes_performance_ceiling"] = (
                per_query <= MAX_MEDIAN_SECONDS_PER_QUERY
            )
            cell["projected_full_graph_hours"] = (
                per_query * RETAINED_ROWS / 3_600
            )
        cells[cell_key(nprobe, width)] = cell

    selected = select_cell({"cells": cells})
    checks = {
        "runtime_matches": all(
            value is True for value in runtime["checks"].values()
        ),
        "all_registered_cells_present": (
            set(cells)
            == {
                cell_key(nprobe, width)
                for nprobe, width in POLICY_GRID
            }
        ),
        "complete_disjoint_candidate_universe": True,
        "unambiguous_fraction_at_least_0_90": (
            float(unambiguous.mean()) >= 0.90
        ),
        "passing_quality_and_performance_policy_selected": (
            selected is not None
        ),
        "no_training_performed": True,
        "no_scale_decision_made": True,
    }
    passed = all(value is True for value in checks.values())
    r0093_selected = r0093["receipt"]["selected"]
    body = {
        "schema": QUALIFICATION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "validity_passed": passed,
        "failed_checks": sorted(
            key for key, value in checks.items() if value is not True
        ),
        "training_performed": False,
        "optimizer_updates": 0,
        "scale_decision_made": False,
        "substrate": substrate["signature"],
        "source_index": filtered,
        "split_receipt": split["signature"],
        "r0093_decision": r0093["signature"],
        "r0093_monolithic_selected": r0093_selected,
        "runtime": runtime,
        "policy_grid": [
            {
                "nprobe_per_shard": nprobe,
                "width_per_shard": width,
                "total_shortlist_width": width * len(SHARD_SPECS),
            }
            for nprobe, width in POLICY_GRID
        ],
        "policy_selector": (
            "lowest median three-repeat 10000-query search-plus-rerank "
            "wall among cells with mean unambiguous exact-reranked "
            "recall@15 >= 0.84 and median wall <= 0.001 seconds/query"
        ),
        "selected": selected,
        "cells": cells,
        "quality": {
            "sample_rows": len(sample),
            "sample_seed": QUALITY_SEED,
            "sample_sha256": sha256_bytes(sample.tobytes()),
            "boundary_ties": int(ties.sum()),
            "unambiguous_fraction": float(unambiguous.mean()),
            "floor": MEAN_RECALL_FLOOR,
        },
        "performance": {
            "maximum_median_seconds_per_query": (
                MAX_MEDIAN_SECONDS_PER_QUERY
            ),
            "r0093_median_seconds_per_query": r0093_selected[
                "benchmark"
            ]["median_wall_seconds_per_query"],
            "exact_truth": exact_performance,
            "gpu_clone_seconds_by_shard": clone_seconds,
            "peak_rss_gib": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024 ** 2)
            ),
        },
        "checks": checks,
    }
    receipt = seal(body)
    receipt_path = os.path.join(
        output,
        "sharded-search-qualification.json",
    )
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    if not passed or selected is None:
        raise Round0094Error(
            "R0094 sharded search failed: "
            + ", ".join(receipt["failed_checks"])
        )
    baseline = float(
        r0093_selected["benchmark"]["median_wall_seconds_per_query"]
    )
    selected_wall = float(
        selected["benchmark"]["median_wall_seconds_per_query"]
    )
    decision = seal({
        "schema": DECISION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "validity_passed": True,
        "registered_mean_recall_floor": MEAN_RECALL_FLOOR,
        "maximum_median_seconds_per_query": (
            MAX_MEDIAN_SECONDS_PER_QUERY
        ),
        "selected": selected,
        "speedup_vs_r0093_monolithic": baseline / selected_wall,
        "qualification": expected_input_signature(receipt_path),
        "split_receipt": split["signature"],
        "substrate": substrate["signature"],
        "source_index": filtered,
        "r0093_decision": r0093["signature"],
        "full_150m_map_evaluation_still_required": True,
        "training_performed": False,
        "optimizer_updates": 0,
    })
    decision_path = os.path.join(
        output,
        "sharded-search-decision.json",
    )
    atomic_write_new_json(decision_path, decision, immutable=True)
    return {**decision, "receipt": expected_input_signature(decision_path)}


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID or job is None:
        raise Round0094Error("R0094 handler requires its exact round/job")
    action = str(job.get("action"))
    if action == "split_corpus_indices":
        return run_split(active, job)
    if action == "qualify_sharded_search":
        return run_qualification(active, job)
    raise Round0094Error(f"unknown R0094 action {action!r}")
