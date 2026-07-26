"""Build the R0064-selected next-rung graph with qualified GPU IVF-PQ."""
from __future__ import annotations

import math
import os
import resource
import time
from functools import partial
from typing import Any, Mapping

import numpy as np

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_write_new_json
from basemap.round0034_pipeline import GRAPH_SCHEMA
from basemap.round0049_program import (
    DIMENSION,
    K,
    SOURCE_ROWS,
    _seal,
    compact_to_global,
    global_to_compact,
)
from basemap.round0065_substrates import (
    SUBSETS,
    validate_scale_substrate,
)
from basemap.round0066_quality import load_scale_decision
from basemap.round0067_graph import (
    GRAPH_RECEIPT_SCHEMA,
    ROUND_ID,
    Round0067Error,
    load_gpu_qualification,
)
from experiments.round0049_nodes import (
    INDEX_SEARCH_WIDTH,
    SEARCH_BATCH_ROWS,
    SEARCH_WIDTH,
    SHARD_ROWS,
    _assemble_graph,
    _initialize_graph_output,
    _warm_page_cache,
    _write_shard,
)
from experiments.round0059_nodes import (
    _GpuSearchAdapter,
    _runtime_stamp,
)


RUNTIME_SPEC = os.path.join(
    os.path.dirname(__file__),
    "round0060_runtime.json",
)


def _to_gpu(
    faiss: Any,
    index: Any,
    *,
    nprobe: int,
) -> tuple[Any, Any, float]:
    resources = faiss.StandardGpuResources()
    resources.setTempMemory(1 << 30)
    options = faiss.GpuClonerOptions()
    options.indicesOptions = faiss.INDICES_64_BIT
    options.useFloat16 = False
    options.usePrecomputed = True
    started = time.monotonic()
    gpu = faiss.index_cpu_to_gpu(resources, 0, index, options)
    clone_seconds = time.monotonic() - started
    gpu.nprobe = nprobe
    return resources, _GpuSearchAdapter(gpu, nprobe), clone_seconds


def run_build_graph(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    import faiss

    started = time.monotonic()
    decision = load_scale_decision(
        str(job["scale_comparison"]),
        expected_sha256=str(job["scale_comparison_sha256"]),
    )
    tier = decision["tier"]
    if tier != str(job["tier"]):
        raise Round0067Error("materialized graph tier differs from R0064")
    spec = SUBSETS[tier]
    row_count = int(spec["row_count"])
    intervals = tuple(spec["intervals"])
    expected_excluded = int(
        spec["eligibility_summary"]["excluded_row_count"]
    )
    expected_retained = int(
        spec["eligibility_summary"]["retained_row_count"]
    )
    substrate = validate_scale_substrate(
        str(job["substrate_manifest"]),
        tier=tier,
        expected_sha256=str(job["substrate_manifest_sha256"]),
    )
    outputs = substrate["manifest"]["outputs"]
    qualification = load_gpu_qualification(
        str(job["gpu_qualification_receipt"]),
        expected_sha256=str(job["gpu_qualification_receipt_sha256"]),
        tier=tier,
        scale_decision_signature=decision["signature"],
        substrate_signature=substrate["signature"],
        eligibility_signature=outputs["eligibility"],
    )
    receipt = qualification["receipt"]
    nprobe = int(receipt["selected_nprobe"])
    filtered_registered = receipt["candidate_universe"]["filtered_index"]
    filtered_signature = expected_input_signature(
        str(job["filtered_index"])
    )
    if filtered_signature != filtered_registered:
        raise Round0067Error("qualified filtered index bytes changed")
    runtime = _runtime_stamp(
        str(job["runtime_spec"]),
        str(job["runtime_spec_sha256"]),
    )
    eligibility = substrate["eligibility"]
    excluded = np.asarray(eligibility["excluded_rows"], dtype=np.int64)
    if (
        len(excluded) != expected_excluded
        or row_count - len(excluded) != expected_retained
    ):
        raise Round0067Error("selected graph eligibility changed")

    output = str(job["outputs"][0])
    contract = {
        "schema": "round0067-next-rung-gpu-graph-contract-v1",
        "release_sha": active["manifest"]["release_sha"],
        "tier": tier,
        "scale_decision": decision["signature"],
        "substrate": substrate["signature"],
        "eligibility": outputs["eligibility"],
        "gpu_qualification": qualification["signature"],
        "filtered_index": filtered_signature,
        "runtime_spec": expected_input_signature(
            str(job["runtime_spec"])
        ),
        "nprobe": nprobe,
        "search_width": SEARCH_WIDTH,
        "index_search_width": INDEX_SEARCH_WIDTH,
        "exact_rerank": True,
        "k": K,
        "shard_rows": SHARD_ROWS,
        "search_batch_rows": SEARCH_BATCH_ROWS,
        "engine": "faiss-classic-GpuIndexIVFPQ",
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
        shape=(row_count, DIMENSION),
    )
    scales = np.memmap(
        outputs["scales"]["canonical_path"],
        dtype="<f2",
        mode="r",
        shape=(row_count,),
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
        raise Round0067Error("qualified filtered-index geometry changed")
    resources, gpu, clone_seconds = _to_gpu(
        faiss,
        filtered,
        nprobe=nprobe,
    )
    parameters = faiss.SearchParametersIVF()
    parameters.nprobe = nprobe
    compact_to_global_fn = partial(
        compact_to_global,
        intervals=intervals,
    )
    global_to_compact_fn = partial(
        global_to_compact,
        intervals=intervals,
    )

    resumed = 0
    shard_receipts: list[dict[str, Any]] = []
    for shard, start in enumerate(range(0, row_count, SHARD_ROWS)):
        stop = min(start + SHARD_ROWS, row_count)
        shard_receipt = _write_shard(
            index=gpu,
            parameters=parameters,
            encoded=encoded,
            scales=scales,
            excluded=excluded,
            shard_root=shard_root,
            shard=shard,
            start=start,
            stop=stop,
            nprobe=nprobe,
            round_id=ROUND_ID,
            compact_to_global_fn=compact_to_global_fn,
            global_to_compact_fn=global_to_compact_fn,
            source_rows=SOURCE_ROWS,
        )
        resumed += int(shard_receipt["resumed"])
        shard_receipts.append(shard_receipt)
        print(
            f"R0067 {tier} graph shard {shard + 1}/"
            f"{math.ceil(row_count / SHARD_ROWS)} "
            f"rows[{start}:{stop}] "
            f"{shard_receipt['retained_sources']} sources "
            f"{shard_receipt['wall_seconds']:.2f}s"
            + (" resumed" if shard_receipt["resumed"] else ""),
            flush=True,
        )

    target_signature, degree_signature = _assemble_graph(
        output=output,
        shard_root=shard_root,
        excluded=excluded,
        nprobe=nprobe,
        round_id=ROUND_ID,
        row_count=row_count,
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
        "tier": tier,
        "row_count": row_count,
        "input_k": K,
        "source_semantics": (
            f"compact balanced-{tier} int8 row; excluded rows degree zero"
        ),
        "destination_policy": (
            "native representative-only GPU IVF-PQ search in the selected "
            "next-rung universe; remove self"
        ),
        "sampling_semantics": (
            "source-uniform and edge-uniform are identical at fixed degree 15"
        ),
        "negative_policy": "uniform-eligibility-retained-rows-nonself",
        "weight_semantics": "unweighted directed k15",
        "inputs": {
            "scale_decision": decision["signature"],
            "eligibility": outputs["eligibility"],
            "substrate": substrate["signature"],
            "gpu_qualification": qualification["signature"],
            "filtered_index": filtered_signature,
        },
        "outputs": {
            "targets": target_signature,
            "degrees": degree_signature,
        },
        "target_dtype": "<i4",
        "target_shape": [row_count, K],
        "degree_dtype": "|u1",
        "degree_shape": [row_count],
        "summary": {
            "eligibility_excluded_source_count": expected_excluded,
            "eligibility_retained_row_count": expected_retained,
            "retained_positive_source_count": expected_retained,
            "zero_degree_retained_source_count": 0,
            "zero_degree_retained_source_fraction": 0.0,
            "valid_canonical_edge_count": expected_retained * K,
            "input_edge_count": expected_retained * K,
            "degree_histogram": {
                "0": expected_excluded,
                str(K): expected_retained,
            },
            "self_returned_count": self_seen,
            "self_returned_fraction": self_seen / expected_retained,
        },
        "candidate_generator": {
            "index_type": "GpuIndexIVFPQ",
            "source_index_rows": SOURCE_ROWS,
            "physically_filtered_index_rows": expected_retained,
            "nlist": 8_192,
            "pq_m": 48,
            "pq_nbits": 8,
            "nprobe": nprobe,
            "search_width": SEARCH_WIDTH,
            "index_search_width": INDEX_SEARCH_WIDTH,
            "selected_neighbors": K,
            "exact_rerank": True,
            "rerank_vector_source": (
                "selected-subset int8-plus-fp16-scale exact cosine"
            ),
            "candidate_universe": (
                f"first {spec['first_rows_per_corpus']} rows per corpus, "
                "retained representatives only"
            ),
        },
        "quality": {
            "mean_recall_at_15_unambiguous": receipt["quality"][
                "selected"
            ]["mean_recall_at_15_unambiguous"],
            "floor": 0.90,
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
        "tier": tier,
        "training_performed": False,
        "optimizer_updates": 0,
        "graph": expected_input_signature(graph_path),
        "substrate": substrate["signature"],
        "gpu_qualification": qualification["signature"],
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
    graph_receipt = _seal(receipt_body)
    receipt_path = os.path.join(output, "receipt.json")
    atomic_write_new_json(receipt_path, graph_receipt, immutable=True)
    del resources, gpu, filtered
    return {
        **graph_receipt,
        "receipt": expected_input_signature(receipt_path),
    }


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0067Error("R0067 handler received another queue")
    selected = job if job is not None else active.get("job") or {}
    if selected.get("action") != "build_selected_gpu_graph":
        raise Round0067Error("R0067 accepts only the selected graph")
    return run_build_graph(active, selected)
