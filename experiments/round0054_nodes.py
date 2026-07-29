"""CPU-only native graph builder for the matched balanced-30M control."""
from __future__ import annotations

import json
import math
import os
import resource
import time
from typing import Any, Mapping

import numpy as np

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from basemap.int8_eligibility import load_int8_eligibility
from basemap.output_safety import atomic_write_new_json
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
    _eligible_selector,
    _initialize_graph_output,
    _warm_page_cache,
    _write_shard,
)
from experiments.round0053_nodes import QUALITY_RECEIPT_SCHEMA


ROUND_ID = "0054"
GRAPH_RECEIPT_SCHEMA = "round0054-balanced-30m-graph-receipt-v1"
DEFAULT_THREADS = 24


def _validate_quality(
    path: str,
    *,
    expected_sha256: str,
    nprobe: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise Round0053Error("R0053 quality receipt bytes changed")
    with open(path, encoding="utf-8") as handle:
        quality = json.load(handle)
    body = {
        key: value for key, value in quality.items()
        if key != "identity_sha256"
    }
    if (
        quality.get("schema") != QUALITY_RECEIPT_SCHEMA
        or quality.get("identity_sha256")
        != sha256_bytes(canonical_json(body))
        or quality.get("validity_passed") is not True
        or int(
            quality.get("candidate_generator", {}).get("nprobe", -1)
        )
        != nprobe
        or int(
            quality.get("candidate_generator", {}).get(
                "search_width",
                -1,
            )
        )
        != SEARCH_WIDTH
        or quality.get("candidate_generator", {}).get(
            "exact_rerank"
        )
        is not True
        or float(
            quality.get("recall", {}).get(
                "mean_recall_at_15_unambiguous",
                -1,
            )
        )
        < MEAN_RECALL_FLOOR
    ):
        raise Round0053Error(
            "R0054 lacks a passing matched candidate-quality receipt"
        )
    return quality, signature


def run_build_graph(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    import faiss

    nprobe = int(job["nprobe"])
    threads = int(job.get("cpu_threads", DEFAULT_THREADS))
    if (
        nprobe <= 0
        or nprobe > 8192
        or threads <= 0
        or threads > (os.cpu_count() or 1)
    ):
        raise Round0053Error("R0054 search geometry is invalid")
    substrate = validate_control_substrate(
        str(job["substrate_manifest"]),
        expected_sha256=str(job["substrate_manifest_sha256"]),
    )
    outputs = substrate["manifest"]["outputs"]
    eligibility = load_int8_eligibility(
        outputs["eligibility"]["canonical_path"],
        expected_sha256=outputs["eligibility"]["sha256"],
        row_count=ROW_COUNT,
    )
    excluded = np.asarray(
        eligibility["excluded_rows"],
        dtype=np.int64,
    )
    if (
        len(excluded) != EXPECTED_EXCLUDED_ROWS
        or ROW_COUNT - len(excluded) != EXPECTED_RETAINED_ROWS
    ):
        raise Round0053Error("R0054 eligibility accounting changed")
    quality, quality_signature = _validate_quality(
        str(job["quality_validation_receipt"]),
        expected_sha256=str(job["quality_validation_receipt_sha256"]),
        nprobe=nprobe,
    )
    r0047_signature = expected_input_signature(
        str(job["candidate_quality_receipt"])
    )
    if (
        r0047_signature["sha256"]
        != str(job["candidate_quality_receipt_sha256"])
    ):
        raise Round0053Error("R0047 quality receipt changed")
    index_signature = expected_input_signature(INDEX_PATH)
    if index_signature["sha256"] != INDEX_SHA256:
        raise Round0053Error("registered 150M index changed")
    contract = {
        "schema": "round0054-balanced-30m-graph-build-contract-v1",
        "release_sha": active["manifest"]["release_sha"],
        "substrate": substrate["signature"],
        "eligibility": outputs["eligibility"],
        "index": index_signature,
        "r0047_candidate_quality_receipt": r0047_signature,
        "quality_validation_receipt": quality_signature,
        "nprobe": nprobe,
        "search_width": SEARCH_WIDTH,
        "index_search_width": INDEX_SEARCH_WIDTH,
        "exact_rerank": True,
        "k": K,
        "shard_rows": SHARD_ROWS,
        "search_batch_rows": SEARCH_BATCH_ROWS,
        "cpu_threads": threads,
        "rerank_page_cache_policy": (
            "synchronous-sequential-read-before-random-gathers"
        ),
        "candidate_universe": (
            "first 10M rows per corpus AND NOT within-subset zero/copies"
        ),
    }
    output = str(job["outputs"][0])
    shard_root = _initialize_graph_output(
        output,
        contract=contract,
    )
    started = time.monotonic()
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
    selector, selector_keepalive, excluded_global = (
        _eligible_selector(
            excluded,
            intervals=GLOBAL_150M_INTERVALS,
            compact_to_global_fn=compact30_to_global150,
        )
    )
    parameters = faiss.SearchParametersIVF()
    parameters.nprobe = nprobe
    parameters.sel = selector
    faiss.omp_set_num_threads(threads)
    index = faiss.read_index(
        INDEX_PATH,
        faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY,
    )
    if (
        type(index).__name__ != "IndexIVFPQ"
        or int(index.ntotal) != INDEX_ROWS
        or int(index.d) != DIMENSION
        or int(index.nlist) != 8192
        or int(index.pq.M) != 48
        or int(index.pq.nbits) != 8
    ):
        raise Round0053Error("registered IVF-PQ geometry changed")
    resumed = 0
    shard_receipts: list[dict[str, Any]] = []
    for shard, start in enumerate(range(0, ROW_COUNT, SHARD_ROWS)):
        stop = min(start + SHARD_ROWS, ROW_COUNT)
        receipt = _write_shard(
            index=index,
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
            compact_to_global_fn=compact30_to_global150,
            global_to_compact_fn=global150_to_compact30,
            source_rows=INDEX_ROWS,
        )
        resumed += int(receipt["resumed"])
        shard_receipts.append(receipt)
        print(
            f"R0054 graph shard {shard + 1}/"
            f"{math.ceil(ROW_COUNT / SHARD_ROWS)} "
            f"rows[{start}:{stop}] "
            f"{receipt['retained_sources']} sources "
            f"{receipt['wall_seconds']:.2f}s"
            + (" resumed" if receipt["resumed"] else ""),
            flush=True,
        )
    del selector_keepalive, excluded_global
    target_signature, degree_signature = _assemble_graph(
        output=output,
        shard_root=shard_root,
        excluded=excluded,
        nprobe=nprobe,
        round_id=ROUND_ID,
        row_count=ROW_COUNT,
    )
    search_seconds = sum(
        float(value["search_seconds"])
        for value in shard_receipts
    )
    rerank_seconds = sum(
        float(value["rerank_seconds"])
        for value in shard_receipts
    )
    self_seen = sum(
        int(value["self_returned"])
        for value in shard_receipts
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
            "native representative-only IVF-PQ search in matched 30M "
            "candidate universe; remove self"
        ),
        "sampling_semantics": (
            "source-uniform and edge-uniform are identical at fixed degree 15"
        ),
        "negative_policy": "uniform-eligibility-retained-rows-nonself",
        "weight_semantics": "unweighted directed k15",
        "inputs": {
            "eligibility": outputs["eligibility"],
            "substrate": substrate["signature"],
            "index": index_signature,
            "r0047_candidate_quality_receipt": r0047_signature,
            "quality_validation_receipt": quality_signature,
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
            "eligibility_excluded_source_count": (
                EXPECTED_EXCLUDED_ROWS
            ),
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
            "index_type": "IndexIVFPQ",
            "source_index_rows": INDEX_ROWS,
            "nlist": 8192,
            "pq_m": 48,
            "pq_nbits": 8,
            "nprobe": nprobe,
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
                quality["recall"][
                    "mean_recall_at_15_unambiguous"
                ]
            ),
            "floor": MEAN_RECALL_FLOOR,
        },
        "timing": {
            "page_cache_warm": page_cache_warm,
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
        "quality_validation_receipt": quality_signature,
        "search": graph["candidate_generator"],
        "summary": graph["summary"],
        "performance": {
            **graph["timing"],
            "cpu_threads": threads,
            "peak_rss_gib": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024 ** 2)
            ),
        },
    }
    receipt = _seal(receipt_body)
    receipt_path = os.path.join(output, "receipt.json")
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    return {
        **receipt,
        "receipt": expected_input_signature(receipt_path),
    }


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0053Error("R0054 handler received another queue")
    selected = job if job is not None else active.get("job") or {}
    if selected.get("action") != "build_graph":
        raise Round0053Error("R0054 accepts only graph build")
    return run_build_graph(active, selected)
