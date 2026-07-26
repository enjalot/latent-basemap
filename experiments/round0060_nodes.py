"""Build the matched balanced-30M representative graph with GPU IVF-PQ."""
from __future__ import annotations

import gc
import math
import os
import resource
import time
from typing import Any, Mapping

import numpy as np

from basemap.artifact_identity import (
    expected_input_signature,
    sha256_bytes,
)
from basemap.int8_eligibility import load_int8_eligibility
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0034_pipeline import GRAPH_SCHEMA
from basemap.round0049_program import (
    DIMENSION,
    INDEX_PATH,
    INDEX_SHA256,
    K,
    SOURCE_ROWS as INDEX_ROWS,
    _seal,
)
from basemap.round0053_program import (
    EXPECTED_EXCLUDED_ROWS,
    EXPECTED_RETAINED_ROWS,
    GLOBAL_150M_INTERVALS,
    ROW_COUNT,
    Round0053Error,
    compact30_to_global150,
    global150_to_compact30,
    validate_control_substrate,
)
from experiments.round0049_nodes import (
    INDEX_SEARCH_WIDTH,
    MEAN_RECALL_FLOOR,
    SEARCH_BATCH_ROWS,
    SEARCH_WIDTH,
    SHARD_ROWS,
    _assemble_graph,
    _clean_search,
    _exact_representative_truth,
    _exact_rerank_shortlist,
    _initialize_graph_output,
    _membership,
    _sample_retained_rows,
    _warm_page_cache,
    _write_shard,
)
from experiments.round0053_nodes import (
    QUALITY_SAMPLE_ROWS,
    QUALITY_SEED,
)
from experiments.round0054_nodes import _validate_quality
from experiments.round0059_nodes import (
    FAISS_WHEEL,
    RECEIPT_SCHEMA as R0059_RECEIPT_SCHEMA,
    _GpuSearchAdapter,
    _load_sealed_json,
    _project_full_graph_hours,
    _runtime_stamp,
)


ROUND_ID = "0060"
QUALIFICATION_SCHEMA = "round0060-balanced-30m-gpu-index-v1"
GRAPH_RECEIPT_SCHEMA = "round0060-balanced-30m-gpu-graph-receipt-v1"
NPROBE = 64
BENCHMARK_ROWS = 10_000
MAX_PROJECTED_GRAPH_HOURS = 2.0
RUNTIME_SPEC = os.path.join(
    os.path.dirname(__file__),
    "round0060_runtime.json",
)


def _load_qualification(path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    signature = expected_input_signature(path)
    return _load_sealed_json(
        path,
        expected_sha256=signature["sha256"],
        schema=QUALIFICATION_SCHEMA,
    )


def _build_filtered_index(
    *,
    faiss: Any,
    destination_path: str,
    excluded: np.ndarray,
) -> tuple[Any, dict[str, Any]]:
    """Copy the registered 30M ranges and physically remove nonrepresentatives."""
    started = time.monotonic()
    source = faiss.read_index(INDEX_PATH)
    if (
        type(source).__name__ != "IndexIVFPQ"
        or int(source.ntotal) != INDEX_ROWS
        or int(source.d) != DIMENSION
        or int(source.nlist) != 8_192
        or int(source.code_size) != 48
        or int(source.pq.M) != 48
        or int(source.pq.nbits) != 8
    ):
        raise Round0053Error("registered IVF-PQ geometry changed")
    destination = faiss.clone_index(source)
    destination.reset()
    for start, stop in GLOBAL_150M_INTERVALS:
        source.copy_subset_to(
            destination,
            faiss.InvertedLists.SUBSET_TYPE_ID_RANGE,
            start,
            stop,
        )
    copied = int(destination.ntotal)
    if copied != ROW_COUNT:
        raise Round0053Error(
            f"balanced-30M range copy produced {copied} rows"
        )
    excluded_global = np.ascontiguousarray(
        compact30_to_global150(excluded),
        dtype=np.int64,
    )
    selector = faiss.IDSelectorBatch(excluded_global)
    removed = int(destination.remove_ids(selector))
    if (
        removed != EXPECTED_EXCLUDED_ROWS
        or int(destination.ntotal) != EXPECTED_RETAINED_ROWS
    ):
        raise Round0053Error("physical 30M eligibility filtering changed")
    temporary = destination_path + ".partial"
    if os.path.exists(temporary) or os.path.exists(destination_path):
        raise Round0053Error("filtered 30M index output already exists")
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
    del source, selector, excluded_global
    gc.collect()
    return destination, performance


def _queries(
    encoded: np.ndarray,
    scales: np.ndarray,
    rows: np.ndarray,
) -> np.ndarray:
    queries = (
        np.asarray(encoded[rows], dtype=np.float32)
        * np.asarray(scales[rows], dtype=np.float32)[:, None]
    )
    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    if (
        not np.isfinite(queries).all()
        or not np.isfinite(norms).all()
        or np.any(norms <= 0)
    ):
        raise Round0053Error("balanced-30M GPU queries are invalid")
    queries /= norms
    return np.ascontiguousarray(queries)


def _search_and_rerank(
    *,
    index: Any,
    queries: np.ndarray,
    compact_sources: np.ndarray,
    encoded: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    started = time.monotonic()
    _distances, raw = index.search(queries, INDEX_SEARCH_WIDTH)
    search_seconds = time.monotonic() - started
    shortlist, self_seen = _clean_search(
        raw,
        global_sources=compact30_to_global150(compact_sources),
        candidate_count=SEARCH_WIDTH,
        global_to_compact_fn=global150_to_compact30,
        source_rows=INDEX_ROWS,
    )
    selected, rerank = _exact_rerank_shortlist(
        queries=queries,
        shortlist=shortlist,
        encoded=encoded,
        scales=scales,
    )
    return selected, {
        "search_seconds": search_seconds,
        "queries": len(queries),
        "queries_per_second": len(queries) / search_seconds,
        "self_returned": self_seen,
        "exact_rerank": rerank,
    }


def _to_gpu(faiss: Any, index: Any) -> tuple[Any, Any, float]:
    resources = faiss.StandardGpuResources()
    resources.setTempMemory(1 << 30)
    options = faiss.GpuClonerOptions()
    options.indicesOptions = faiss.INDICES_64_BIT
    options.useFloat16 = False
    options.usePrecomputed = True
    started = time.monotonic()
    gpu = faiss.index_cpu_to_gpu(resources, 0, index, options)
    clone_seconds = time.monotonic() - started
    gpu.nprobe = NPROBE
    return resources, _GpuSearchAdapter(gpu, NPROBE), clone_seconds


def run_qualify_index(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    import faiss

    output = create_fresh_directory(
        str(job["outputs"][0]),
        label="Round 0060 balanced-30M GPU index",
    )
    runtime = _runtime_stamp(
        str(job["runtime_spec"]),
        str(job["runtime_spec_sha256"]),
    )
    substrate = validate_control_substrate(
        str(job["substrate_manifest"]),
        expected_sha256=str(job["substrate_manifest_sha256"]),
    )
    quality, quality_signature = _validate_quality(
        str(job["quality_receipt"]),
        expected_sha256=str(job["quality_receipt_sha256"]),
        nprobe=NPROBE,
    )
    r0059, r0059_signature = _load_sealed_json(
        str(job["gpu_qualification_receipt"]),
        expected_sha256=str(job["gpu_qualification_receipt_sha256"]),
        schema=R0059_RECEIPT_SCHEMA,
    )
    if r0059.get("validity_passed") is not True:
        raise Round0053Error("R0059 did not qualify GPU IVF-PQ")
    index_signature = expected_input_signature(INDEX_PATH)
    if index_signature["sha256"] != INDEX_SHA256:
        raise Round0053Error("registered 150M IVF-PQ bytes changed")
    outputs = substrate["manifest"]["outputs"]
    eligibility = load_int8_eligibility(
        outputs["eligibility"]["canonical_path"],
        expected_sha256=outputs["eligibility"]["sha256"],
        row_count=ROW_COUNT,
    )
    excluded = np.asarray(eligibility["excluded_rows"], dtype=np.int64)
    if (
        len(excluded) != EXPECTED_EXCLUDED_ROWS
        or ROW_COUNT - len(excluded) != EXPECTED_RETAINED_ROWS
    ):
        raise Round0053Error("balanced-30M eligibility accounting changed")
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
    filtered_path = os.path.join(output, "balanced-30m-retained.ivfpq")
    filtered, filtering = _build_filtered_index(
        faiss=faiss,
        destination_path=filtered_path,
        excluded=excluded,
    )
    resources, gpu, clone_seconds = _to_gpu(faiss, filtered)

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
    sample_queries = _queries(encoded, scales, sample)
    selected, quality_performance = _search_and_rerank(
        index=gpu,
        queries=sample_queries,
        compact_sources=sample,
        encoded=encoded,
        scales=scales,
    )
    overlap = (
        selected[:, :, None] == exact[:, None, :]
    ).any(axis=2).sum(axis=1) / K
    unambiguous = ~ties
    mean_quality = float(overlap[unambiguous].mean())
    p10_quality = float(np.percentile(overlap[unambiguous], 10))

    first_rows = np.arange(100_000, dtype=np.int64)
    first_retained = first_rows[~_membership(excluded, first_rows)]
    benchmark_rows = first_retained[:BENCHMARK_ROWS]
    benchmark_queries = _queries(encoded, scales, benchmark_rows)
    _benchmark_selected, benchmark = _search_and_rerank(
        index=gpu,
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
    expected_mean = float(
        quality["recall"]["mean_recall_at_15_unambiguous"]
    )
    checks = {
        "runtime_matches": all(
            value is True for value in runtime["checks"].values()
        ),
        "r0059_gpu_qualification_passed": True,
        "filtered_candidate_count": (
            filtering["filtered_ntotal"] == EXPECTED_RETAINED_ROWS
        ),
        "sample_matches_r0053": (
            sha256_bytes(sample.tobytes())
            == quality["sample"]["row_sha256"]
        ),
        "gpu_mean_reproduces_r0053": (
            abs(mean_quality - expected_mean) <= 1e-12
        ),
        "gpu_mean_recall_floor": (
            mean_quality >= MEAN_RECALL_FLOOR
        ),
        "projected_graph_wall": (
            projection["total_hours"] <= MAX_PROJECTED_GRAPH_HOURS
        ),
        "no_training_performed": True,
    }
    passed = all(value is True for value in checks.values())
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
        "runtime": runtime,
        "substrate": substrate["signature"],
        "eligibility": outputs["eligibility"],
        "source_index": index_signature,
        "r0053_quality": quality_signature,
        "r0059_gpu_qualification": r0059_signature,
        "nprobe": NPROBE,
        "filtered_index": filtering["index"],
        "filtering": {
            key: value for key, value in filtering.items()
            if key != "index"
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
            "sample_sha256": sha256_bytes(sample.tobytes()),
            "boundary_ties": int(ties.sum()),
            "mean_recall_at_15_unambiguous": mean_quality,
            "p10_recall_at_15_unambiguous": p10_quality,
            "registered_r0053_mean": expected_mean,
            "floor": MEAN_RECALL_FLOOR,
            "performance": quality_performance,
            "exact_truth": exact_performance,
        },
        "benchmark": {
            **benchmark,
            "rows": len(benchmark_rows),
            "projected_full_30m_graph_hours": (
                projection["total_hours"]
            ),
            "projected_full_30m_search_hours": (
                projection["search_hours"]
            ),
            "projected_full_30m_rerank_hours": (
                projection["rerank_hours"]
            ),
            "maximum_projected_graph_hours": (
                MAX_PROJECTED_GRAPH_HOURS
            ),
        },
        "checks": checks,
        "peak_rss_gib": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024 ** 2)
        ),
    }
    receipt = _seal(body)
    path = os.path.join(output, "gpu-index-qualification-v1.json")
    atomic_write_new_json(path, receipt, immutable=True)
    del resources, gpu, filtered
    if not passed:
        raise Round0053Error(
            "balanced-30M GPU qualification failed: "
            + ", ".join(receipt["failed_checks"])
        )
    return {
        **receipt,
        "receipt": expected_input_signature(path),
    }


def run_build_graph(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    import faiss

    started = time.monotonic()
    runtime = _runtime_stamp(
        str(job["runtime_spec"]),
        str(job["runtime_spec_sha256"]),
    )
    substrate = validate_control_substrate(
        str(job["substrate_manifest"]),
        expected_sha256=str(job["substrate_manifest_sha256"]),
    )
    quality, quality_signature = _validate_quality(
        str(job["quality_receipt"]),
        expected_sha256=str(job["quality_receipt_sha256"]),
        nprobe=NPROBE,
    )
    qualification, qualification_signature = _load_qualification(
        str(job["qualification_receipt"])
    )
    if (
        qualification.get("validity_passed") is not True
        or qualification.get("release_sha")
        != active["manifest"]["release_sha"]
        or qualification.get("substrate") != substrate["signature"]
        or int(qualification.get("nprobe", -1)) != NPROBE
    ):
        raise Round0053Error("R0060 GPU qualification identity changed")
    filtered_signature = expected_input_signature(
        str(job["filtered_index"])
    )
    if filtered_signature != qualification["filtered_index"]:
        raise Round0053Error("qualified 30M filtered index changed")
    outputs = substrate["manifest"]["outputs"]
    eligibility = load_int8_eligibility(
        outputs["eligibility"]["canonical_path"],
        expected_sha256=outputs["eligibility"]["sha256"],
        row_count=ROW_COUNT,
    )
    excluded = np.asarray(eligibility["excluded_rows"], dtype=np.int64)
    if (
        len(excluded) != EXPECTED_EXCLUDED_ROWS
        or ROW_COUNT - len(excluded) != EXPECTED_RETAINED_ROWS
    ):
        raise Round0053Error("R0060 graph eligibility changed")
    output = str(job["outputs"][0])
    contract = {
        "schema": "round0060-balanced-30m-gpu-graph-contract-v1",
        "release_sha": active["manifest"]["release_sha"],
        "substrate": substrate["signature"],
        "eligibility": outputs["eligibility"],
        "quality_validation_receipt": quality_signature,
        "gpu_qualification": qualification_signature,
        "filtered_index": filtered_signature,
        "runtime_spec": expected_input_signature(
            str(job["runtime_spec"])
        ),
        "nprobe": NPROBE,
        "search_width": SEARCH_WIDTH,
        "index_search_width": INDEX_SEARCH_WIDTH,
        "exact_rerank": True,
        "k": K,
        "shard_rows": SHARD_ROWS,
        "search_batch_rows": SEARCH_BATCH_ROWS,
        "engine": "faiss-classic-GpuIndexIVFPQ",
        "candidate_universe": (
            "first 10M rows per corpus AND NOT within-subset zero/copies"
        ),
    }
    shard_root = _initialize_graph_output(output, contract=contract)
    page_cache_warm = {
        "int8": _warm_page_cache(
            outputs["int8"]["canonical_path"]
        ),
        "scales": _warm_page_cache(
            outputs["scales"]["canonical_path"]
        ),
    }
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
    filtered = faiss.read_index(filtered_signature["canonical_path"])
    if int(filtered.ntotal) != EXPECTED_RETAINED_ROWS:
        raise Round0053Error("qualified filtered index count changed")
    resources, gpu, clone_seconds = _to_gpu(faiss, filtered)
    parameters = faiss.SearchParametersIVF()
    parameters.nprobe = NPROBE
    resumed = 0
    shard_receipts: list[dict[str, Any]] = []
    for shard, start in enumerate(range(0, ROW_COUNT, SHARD_ROWS)):
        stop = min(start + SHARD_ROWS, ROW_COUNT)
        receipt = _write_shard(
            index=gpu,
            parameters=parameters,
            encoded=encoded,
            scales=scales,
            excluded=excluded,
            shard_root=shard_root,
            shard=shard,
            start=start,
            stop=stop,
            nprobe=NPROBE,
            round_id=ROUND_ID,
            compact_to_global_fn=compact30_to_global150,
            global_to_compact_fn=global150_to_compact30,
            source_rows=INDEX_ROWS,
        )
        resumed += int(receipt["resumed"])
        shard_receipts.append(receipt)
        print(
            f"R0060 graph shard {shard + 1}/"
            f"{math.ceil(ROW_COUNT / SHARD_ROWS)} "
            f"rows[{start}:{stop}] "
            f"{receipt['retained_sources']} sources "
            f"{receipt['wall_seconds']:.2f}s"
            + (" resumed" if receipt["resumed"] else ""),
            flush=True,
        )
    target_signature, degree_signature = _assemble_graph(
        output=output,
        shard_root=shard_root,
        excluded=excluded,
        nprobe=NPROBE,
        round_id=ROUND_ID,
        row_count=ROW_COUNT,
    )
    search_seconds = sum(
        float(value["search_seconds"]) for value in shard_receipts
    )
    rerank_seconds = sum(
        float(value["rerank_seconds"]) for value in shard_receipts
    )
    self_seen = sum(
        int(value["self_returned"]) for value in shard_receipts
    )
    graph_body = {
        "schema": GRAPH_SCHEMA,
        "round_id": ROUND_ID,
        "row_count": ROW_COUNT,
        "input_k": K,
        "source_semantics": (
            "compact balanced-30M int8 row; excluded rows degree zero"
        ),
        "destination_policy": (
            "native representative-only GPU IVF-PQ search in matched "
            "30M candidate universe; remove self"
        ),
        "sampling_semantics": (
            "source-uniform and edge-uniform are identical at fixed degree 15"
        ),
        "negative_policy": "uniform-eligibility-retained-rows-nonself",
        "weight_semantics": "unweighted directed k15",
        "inputs": {
            "eligibility": outputs["eligibility"],
            "substrate": substrate["signature"],
            "quality_validation_receipt": quality_signature,
            "gpu_qualification": qualification_signature,
            "filtered_index": filtered_signature,
        },
        "outputs": {
            "targets": target_signature,
            "degrees": degree_signature,
        },
        "target_dtype": "<i4",
        "target_shape": [ROW_COUNT, K],
        "degree_dtype": "|u1",
        "degree_shape": [ROW_COUNT],
        "summary": {
            "eligibility_excluded_source_count": EXPECTED_EXCLUDED_ROWS,
            "eligibility_retained_row_count": EXPECTED_RETAINED_ROWS,
            "retained_positive_source_count": EXPECTED_RETAINED_ROWS,
            "zero_degree_retained_source_count": 0,
            "zero_degree_retained_source_fraction": 0.0,
            "valid_canonical_edge_count": EXPECTED_RETAINED_ROWS * K,
            "input_edge_count": EXPECTED_RETAINED_ROWS * K,
            "degree_histogram": {
                "0": EXPECTED_EXCLUDED_ROWS,
                str(K): EXPECTED_RETAINED_ROWS,
            },
            "self_returned_count": self_seen,
            "self_returned_fraction": (
                self_seen / EXPECTED_RETAINED_ROWS
            ),
        },
        "candidate_generator": {
            "index_type": "GpuIndexIVFPQ",
            "source_index_rows": INDEX_ROWS,
            "physically_filtered_index_rows": EXPECTED_RETAINED_ROWS,
            "nlist": 8192,
            "pq_m": 48,
            "pq_nbits": 8,
            "nprobe": NPROBE,
            "search_width": SEARCH_WIDTH,
            "index_search_width": INDEX_SEARCH_WIDTH,
            "selected_neighbors": K,
            "exact_rerank": True,
            "rerank_vector_source": (
                "balanced-subset int8-plus-fp16-scale exact cosine"
            ),
            "candidate_universe": "first 10M rows per corpus, retained",
        },
        "quality": {
            "mean_recall_at_15_unambiguous": (
                qualification["quality"][
                    "mean_recall_at_15_unambiguous"
                ]
            ),
            "floor": MEAN_RECALL_FLOOR,
        },
        "runtime": runtime,
        "timing": {
            "page_cache_warm": page_cache_warm,
            "gpu_clone_seconds": clone_seconds,
            "search_seconds": search_seconds,
            "rerank_seconds": rerank_seconds,
            "total_seconds": time.monotonic() - started,
            "shard_count": len(shard_receipts),
            "resumed_shards": resumed,
        },
    }
    graph = _seal(graph_body)
    graph_path = os.path.join(output, "canonical-graph-v1.json")
    atomic_write_new_json(graph_path, graph, immutable=True)
    receipt_body = {
        "schema": GRAPH_RECEIPT_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "training_performed": False,
        "optimizer_updates": 0,
        "graph": expected_input_signature(graph_path),
        "substrate": substrate["signature"],
        "gpu_qualification": qualification_signature,
        "search": graph["candidate_generator"],
        "summary": graph["summary"],
        "performance": {
            **graph["timing"],
            "peak_rss_gib": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024 ** 2)
            ),
        },
    }
    receipt = _seal(receipt_body)
    receipt_path = os.path.join(output, "receipt.json")
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    del resources, gpu, filtered
    return {
        **receipt,
        "receipt": expected_input_signature(receipt_path),
    }


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0053Error("R0060 handler received another queue")
    selected = job if job is not None else active.get("job") or {}
    handler = {
        "qualify_gpu_index": run_qualify_index,
        "build_gpu_graph": run_build_graph,
    }.get(selected.get("action"))
    if handler is None:
        raise Round0053Error(
            f"unknown R0060 action: {selected.get('action')!r}"
        )
    return handler(active, selected)
