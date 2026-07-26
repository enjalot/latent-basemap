"""Build the balanced-60M representative graph with qualified GPU IVF-PQ."""
from __future__ import annotations

import math
import os
import resource
import time
from typing import Any, Mapping

import numpy as np

from basemap.artifact_identity import expected_input_signature
from basemap.int8_eligibility import load_int8_eligibility
from basemap.output_safety import atomic_write_new_json
from basemap.round0034_pipeline import GRAPH_SCHEMA
from basemap.round0049_program import (
    DIMENSION,
    K,
    ROW_COUNT,
    SOURCE_ROWS,
    Round0049Error,
    _seal,
    compact_to_global,
    global_to_compact,
    validate_substrate_manifest,
)
from experiments.round0049_nodes import (
    INDEX_SEARCH_WIDTH,
    MEAN_RECALL_FLOOR,
    SEARCH_BATCH_ROWS,
    SEARCH_WIDTH,
    SHARD_ROWS,
    _assemble_graph,
    _initialize_graph_output,
    _warm_page_cache,
    _write_shard,
)
from experiments.round0059_nodes import (
    RECEIPT_SCHEMA as QUALIFICATION_SCHEMA,
    _GpuSearchAdapter,
    _load_sealed_json,
    _runtime_stamp,
)


ROUND_ID = "0062"
NPROBE = 40
EXPECTED_EXCLUDED_ROWS = 600_712
EXPECTED_RETAINED_ROWS = 59_399_288
GRAPH_RECEIPT_SCHEMA = "round0062-balanced-60m-gpu-graph-receipt-v1"
RUNTIME_SPEC = os.path.join(
    os.path.dirname(__file__),
    "round0060_runtime.json",
)


def _validate_qualification(
    value: Mapping[str, Any],
    *,
    substrate_signature: Mapping[str, Any],
    eligibility_signature: Mapping[str, Any],
) -> dict[str, Any]:
    universe = value.get("candidate_universe") or {}
    filtered = universe.get("filtered_index") or {}
    quality = value.get("quality") or {}
    checks = value.get("checks") or {}
    if (
        value.get("validity_passed") is not True
        or value.get("training_performed") is not False
        or int(value.get("optimizer_updates", -1)) != 0
        or int(value.get("selected_nprobe", -1)) != NPROBE
        or value.get("substrate") != substrate_signature
        or value.get("eligibility") != eligibility_signature
        or int(universe.get("physical_exclusions", -1))
        != EXPECTED_EXCLUDED_ROWS
        or int(universe.get("retained_rows", -1))
        != EXPECTED_RETAINED_ROWS
        or not filtered.get("sha256")
        or float(
            quality.get("gpu_mean_recall_at_15_unambiguous", -1.0)
        )
        < MEAN_RECALL_FLOOR
        or not checks
        or any(check is not True for check in checks.values())
    ):
        raise Round0049Error("R0059 GPU qualification capability changed")
    return dict(filtered)


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
    substrate = validate_substrate_manifest(
        str(job["substrate_manifest"]),
        expected_sha256=str(job["substrate_manifest_sha256"]),
    )
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
        raise Round0049Error("balanced-60M eligibility accounting changed")
    qualification, qualification_signature = _load_sealed_json(
        str(job["gpu_qualification_receipt"]),
        expected_sha256=str(job["gpu_qualification_receipt_sha256"]),
        schema=QUALIFICATION_SCHEMA,
    )
    filtered_registered = _validate_qualification(
        qualification,
        substrate_signature=substrate["signature"],
        eligibility_signature=outputs["eligibility"],
    )
    filtered_signature = expected_input_signature(
        str(job["filtered_index"])
    )
    if (
        filtered_signature != filtered_registered
        or filtered_signature["canonical_path"]
        != os.path.realpath(str(job["filtered_index"]))
    ):
        raise Round0049Error("qualified balanced-60M index changed")

    output = str(job["outputs"][0])
    contract = {
        "schema": "round0062-balanced-60m-gpu-graph-contract-v1",
        "release_sha": active["manifest"]["release_sha"],
        "substrate": substrate["signature"],
        "eligibility": outputs["eligibility"],
        "gpu_qualification": qualification_signature,
        "filtered_index": filtered_signature,
        "runtime_spec": expected_input_signature(str(job["runtime_spec"])),
        "nprobe": NPROBE,
        "search_width": SEARCH_WIDTH,
        "index_search_width": INDEX_SEARCH_WIDTH,
        "exact_rerank": True,
        "k": K,
        "shard_rows": SHARD_ROWS,
        "search_batch_rows": SEARCH_BATCH_ROWS,
        "engine": "faiss-classic-GpuIndexIVFPQ",
        "candidate_universe": (
            "balanced 20M-per-corpus intervals with all within-subset "
            "zero and duplicate copies physically removed"
        ),
    }
    shard_root = _initialize_graph_output(output, contract=contract)
    page_cache_warm = {
        "int8": _warm_page_cache(outputs["int8"]["canonical_path"]),
        "scales": _warm_page_cache(outputs["scales"]["canonical_path"]),
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
    if (
        type(filtered).__name__ != "IndexIVFPQ"
        or int(filtered.ntotal) != EXPECTED_RETAINED_ROWS
        or int(filtered.d) != DIMENSION
        or int(filtered.nlist) != 8_192
        or int(filtered.code_size) != 48
        or int(filtered.pq.M) != 48
        or int(filtered.pq.nbits) != 8
    ):
        raise Round0049Error("qualified balanced-60M IVF-PQ geometry changed")
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
            compact_to_global_fn=compact_to_global,
            global_to_compact_fn=global_to_compact,
            source_rows=SOURCE_ROWS,
        )
        resumed += int(receipt["resumed"])
        shard_receipts.append(receipt)
        print(
            f"R0062 graph shard {shard + 1}/"
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
            "compact balanced-60M int8 row; excluded rows degree zero"
        ),
        "destination_policy": (
            "native representative-only GPU IVF-PQ search in the matched "
            "balanced-60M candidate universe; remove self"
        ),
        "sampling_semantics": (
            "source-uniform and edge-uniform are identical at fixed degree 15"
        ),
        "negative_policy": "uniform-eligibility-retained-rows-nonself",
        "weight_semantics": "unweighted directed k15",
        "inputs": {
            "eligibility": outputs["eligibility"],
            "substrate": substrate["signature"],
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
            "self_returned_fraction": self_seen / EXPECTED_RETAINED_ROWS,
        },
        "candidate_generator": {
            "index_type": "GpuIndexIVFPQ",
            "source_index_rows": SOURCE_ROWS,
            "physically_filtered_index_rows": EXPECTED_RETAINED_ROWS,
            "nlist": 8_192,
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
            "candidate_universe": (
                "first 20M rows per corpus, retained representatives only"
            ),
        },
        "quality": {
            "mean_recall_at_15_unambiguous": qualification["quality"][
                "gpu_mean_recall_at_15_unambiguous"
            ],
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
        raise Round0049Error("R0062 handler received another queue")
    selected = job if job is not None else active.get("job") or {}
    if selected.get("action") != "build_gpu_graph":
        raise Round0049Error("R0062 accepts only the GPU graph action")
    return run_build_graph(active, selected)
