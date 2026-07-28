"""Build and assemble the three independently bounded 150M graph parts."""
from __future__ import annotations

import contextlib
import math
import os
import resource
import time
from collections import deque
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0049_program import (
    DIMENSION,
    K,
    SOURCE_ROWS,
    compact_to_global,
    global_to_compact,
)
from basemap.round0086_program import RETAINED_ROWS, ROW_COUNT
from basemap.round0088_graph import (
    ASSEMBLED_GRAPH_SCHEMA,
    ASSEMBLY_RECEIPT_SCHEMA,
    ASSEMBLY_ROUND_ID,
    CORPUS_BY_ROUND,
    CORPUS_SPECS,
    PART_SCHEMA,
    ROUND_BY_CORPUS,
    Round0088Error,
    corpus_spec,
    seal,
    validate_filter_receipt,
    validate_part_receipt,
    validate_qualification,
    validate_staged_substrate,
)
from basemap.round0093_policy import load_decision as load_r0093_decision
from experiments.round0049_nodes import (
    SEARCH_BATCH_ROWS,
    SHARD_ROWS,
    _clean_search,
    _exact_rerank_shortlist,
    _fresh_raw_file,
    _initialize_graph_output,
    _membership,
    _shard_paths,
    _validate_shard,
    _warm_page_cache,
)
from experiments.round0059_nodes import _GpuSearchAdapter, _runtime_stamp
from experiments.round0078_nodes import (
    MAX_PENDING_RERANKS,
    RERANK_BLAS_THREADS,
    RERANK_WORKERS,
)


INTERVALS = ((0, ROW_COUNT),)
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


def _write_part_shard(
    *,
    round_id: str,
    index: Any,
    parameters: Any,
    encoded: np.ndarray,
    scales: np.ndarray,
    excluded: np.ndarray,
    shard_root: str,
    shard: int,
    start: int,
    stop: int,
    nprobe: int,
    search_width: int,
) -> dict[str, Any]:
    """Build one globally numbered shard while GPU search overlaps reranking."""
    target_path, receipt_path = _shard_paths(shard_root, shard)
    previous = _validate_shard(
        target_path=target_path,
        receipt_path=receipt_path,
        start=start,
        stop=stop,
        nprobe=nprobe,
        search_width=search_width,
        round_id=round_id,
    )
    if previous is not None:
        return {**previous, "resumed": True}

    started = time.monotonic()
    targets = np.full((stop - start, K), -1, dtype="<i4")
    source_rows = np.arange(start, stop, dtype=np.int64)
    retained = source_rows[~_membership(excluded, source_rows)]
    self_seen = 0
    search_seconds = 0.0
    rerank_seconds = 0.0
    pending: deque[
        tuple[np.ndarray, Future[tuple[np.ndarray, dict[str, Any]]]]
    ] = deque()
    compact_to_global_fn = partial(compact_to_global, intervals=INTERVALS)
    global_to_compact_fn = partial(global_to_compact, intervals=INTERVALS)

    def resolve_oldest() -> None:
        nonlocal rerank_seconds
        batch_rows, future = pending.popleft()
        selected, rerank = future.result()
        targets[batch_rows - start] = selected
        rerank_seconds += float(rerank["wall_seconds"])

    with threadpool_limits(limits=RERANK_BLAS_THREADS, user_api="blas"):
        with ThreadPoolExecutor(
            max_workers=RERANK_WORKERS,
            thread_name_prefix=f"r{round_id}-rerank",
        ) as executor:
            for batch_start in range(0, len(retained), SEARCH_BATCH_ROWS):
                batch_rows = retained[
                    batch_start:batch_start + SEARCH_BATCH_ROWS
                ]
                query = (
                    np.asarray(encoded[batch_rows], dtype=np.float32)
                    * np.asarray(scales[batch_rows], dtype=np.float32)[:, None]
                )
                norms = np.linalg.norm(query, axis=1, keepdims=True)
                if (
                    not np.isfinite(query).all()
                    or not np.isfinite(norms).all()
                    or np.any(norms <= 0)
                ):
                    raise Round0088Error(
                        f"{round_id} 150M query block is nonfinite"
                    )
                query /= norms
                global_rows = compact_to_global_fn(batch_rows)
                search_started = time.monotonic()
                _distances, raw = index.search(
                    np.ascontiguousarray(query),
                    search_width + 1,
                    params=parameters,
                )
                search_seconds += time.monotonic() - search_started
                shortlist, seen = _clean_search(
                    raw,
                    global_sources=global_rows,
                    candidate_count=search_width,
                    source_rows=SOURCE_ROWS,
                    global_to_compact_fn=global_to_compact_fn,
                )
                self_seen += seen
                pending.append((
                    batch_rows.copy(),
                    executor.submit(
                        _exact_rerank_shortlist,
                        queries=query,
                        shortlist=shortlist,
                        encoded=encoded,
                        scales=scales,
                    ),
                ))
                if len(pending) >= MAX_PENDING_RERANKS:
                    resolve_oldest()
            while pending:
                resolve_oldest()

    retained_mask = ~_membership(excluded, source_rows)
    if (
        np.any(targets[retained_mask] < 0)
        or np.any(targets[~retained_mask] != -1)
    ):
        raise Round0088Error("150M graph shard eligibility disagrees")
    atomic_save_new_npy(target_path, targets, immutable=True)
    body = {
        "schema": "round0049-exact-rerank-graph-shard-v2",
        "round_id": round_id,
        "shard": shard,
        "start": start,
        "stop": stop,
        "retained_sources": len(retained),
        "excluded_sources": stop - start - len(retained),
        "valid_edges": len(retained) * K,
        "nprobe": nprobe,
        "search_width": search_width,
        "index_search_width": search_width + 1,
        "selected_neighbors": K,
        "exact_rerank": True,
        "rerank_workers": RERANK_WORKERS,
        "rerank_blas_threads_per_worker": RERANK_BLAS_THREADS,
        "max_pending_reranks": MAX_PENDING_RERANKS,
        "search_rerank_overlap": True,
        "self_returned": self_seen,
        "search_seconds": search_seconds,
        "rerank_seconds": rerank_seconds,
        "wall_seconds": time.monotonic() - started,
        "targets": expected_input_signature(target_path),
    }
    receipt = seal(body)
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    return {**receipt, "resumed": False}


def run_build_part(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    import faiss

    started = time.monotonic()
    round_id = str(active["manifest"]["round_id"])
    corpus = CORPUS_BY_ROUND.get(round_id)
    if corpus is None or corpus != str(job.get("corpus")):
        raise Round0088Error("150M graph part round/corpus mismatch")
    spec = corpus_spec(corpus)
    substrate = validate_staged_substrate(
        str(job["substrate_manifest"]),
        expected_sha256=str(job["substrate_manifest_sha256"]),
    )
    outputs = substrate["manifest"]["outputs"]
    filtered = expected_input_signature(str(job["filtered_index"]))
    filter_receipt = validate_filter_receipt(
        str(job["filter_receipt"]),
        expected_sha256=str(job["filter_receipt_sha256"]),
        substrate_signature=substrate["signature"],
        filtered_index_signature=filtered,
    )
    qualification = validate_qualification(
        str(job["gpu_qualification_receipt"]),
        expected_sha256=str(job["gpu_qualification_receipt_sha256"]),
        substrate_signature=substrate["signature"],
        filtered_index_signature=filtered,
    )
    selected = qualification["selected"]
    policy_decision = load_r0093_decision(
        str(job["policy_decision"]),
        expected_sha256=str(job["policy_decision_sha256"]),
    )
    decision_receipt = policy_decision["receipt"]
    if (
        decision_receipt.get("selected") != selected
        or decision_receipt.get("qualification")
        != qualification["signature"]
        or decision_receipt.get("substrate") != substrate["signature"]
        or decision_receipt.get("filtered_index") != filtered
    ):
        raise Round0088Error(
            "R0093 decision does not bind the graph-part policy"
        )
    nprobe = int(selected["nprobe"])
    search_width = int(selected["shortlist_width"])
    runtime = _runtime_stamp(
        str(job["runtime_spec"]),
        str(job["runtime_spec_sha256"]),
    )
    excluded = np.asarray(
        substrate["eligibility"]["excluded_rows"], dtype=np.int64
    )
    in_part = excluded[
        (excluded >= spec["start"]) & (excluded < spec["stop"])
    ]
    if (
        len(excluded) != ROW_COUNT - RETAINED_ROWS
        or len(in_part) != spec["excluded_rows"]
    ):
        raise Round0088Error(f"{corpus} eligibility accounting changed")

    output = str(job["outputs"][0])
    contract = {
        "schema": "round0088-split-150m-graph-part-contract-v1",
        "round_id": round_id,
        "release_sha": active["manifest"]["release_sha"],
        "corpus": corpus,
        "start": spec["start"],
        "stop": spec["stop"],
        "substrate": substrate["signature"],
        "filter_receipt": filter_receipt["signature"],
        "gpu_qualification": qualification["signature"],
        "policy_decision": policy_decision["signature"],
        "filtered_index": filtered,
        "runtime_spec": expected_input_signature(str(job["runtime_spec"])),
        "nprobe": nprobe,
        "search_width": search_width,
        "index_search_width": search_width + 1,
        "exact_rerank": True,
        "k": K,
        "shard_rows": SHARD_ROWS,
        "search_batch_rows": SEARCH_BATCH_ROWS,
        "rerank_workers": RERANK_WORKERS,
        "rerank_blas_threads_per_worker": RERANK_BLAS_THREADS,
        "max_pending_reranks": MAX_PENDING_RERANKS,
        "search_rerank_overlap": True,
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
        shape=(ROW_COUNT, DIMENSION),
    )
    scales = np.memmap(
        outputs["scales"]["canonical_path"],
        dtype="<f2",
        mode="r",
        shape=(ROW_COUNT,),
    )
    index = faiss.read_index(filtered["canonical_path"])
    if (
        type(index).__name__ != "IndexIVFPQ"
        or int(index.ntotal) != RETAINED_ROWS
        or int(index.d) != DIMENSION
        or int(index.nlist) != 8_192
        or int(index.code_size) != 48
        or int(index.pq.M) != 48
        or int(index.pq.nbits) != 8
    ):
        raise Round0088Error("qualified 150M filtered-index geometry changed")
    resources, gpu, clone_seconds = _to_gpu(
        faiss, index, nprobe=nprobe
    )
    parameters = faiss.SearchParametersIVF()
    parameters.nprobe = nprobe

    shard_receipts: list[dict[str, Any]] = []
    resumed = 0
    for start in range(spec["start"], spec["stop"], SHARD_ROWS):
        stop = min(start + SHARD_ROWS, spec["stop"])
        shard = start // SHARD_ROWS
        receipt = _write_part_shard(
            round_id=round_id,
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
            search_width=search_width,
        )
        resumed += int(receipt["resumed"])
        shard_receipts.append(receipt)
        print(
            f"R{round_id} {corpus} shard "
            f"{shard - spec['start'] // SHARD_ROWS + 1}/"
            f"{math.ceil((spec['stop'] - spec['start']) / SHARD_ROWS)} "
            f"rows[{start}:{stop}] {receipt['retained_sources']} retained "
            f"{receipt['wall_seconds']:.2f}s"
            + (" resumed" if receipt["resumed"] else ""),
            flush=True,
        )

    retained_sources = sum(
        int(value["retained_sources"]) for value in shard_receipts
    )
    excluded_sources = sum(
        int(value["excluded_sources"]) for value in shard_receipts
    )
    if (
        retained_sources != spec["retained_rows"]
        or excluded_sources != spec["excluded_rows"]
    ):
        raise Round0088Error(f"{corpus} completed counts changed")
    body = {
        "schema": PART_SCHEMA,
        "round_id": round_id,
        "release_sha": active["manifest"]["release_sha"],
        "corpus": corpus,
        "start": spec["start"],
        "stop": spec["stop"],
        "retained_sources": retained_sources,
        "excluded_sources": excluded_sources,
        "valid_edges": retained_sources * K,
        "shard_count": len(shard_receipts),
        "resumed_shards": resumed,
        "shard_receipt_identities_sha256": seal({
            "identities": [
                value["identity_sha256"] for value in shard_receipts
            ],
        })["identity_sha256"],
        "substrate": substrate["signature"],
        "filter_receipt": filter_receipt["signature"],
        "gpu_qualification": qualification["signature"],
        "policy_decision": policy_decision["signature"],
        "filtered_index": filtered,
        "nprobe": nprobe,
        "search_width": search_width,
        "selected_neighbors": K,
        "exact_rerank": True,
        "quality": {
            "mean_recall_at_15_unambiguous": selected[
                "mean_recall_at_15_unambiguous"
            ],
            "floor": qualification["mean_recall_floor"],
            "qualification_sample_rows": qualification["receipt"][
                "quality"
            ]["sample_rows"],
            "qualification_sample_seed": qualification["receipt"][
                "quality"
            ]["sample_seed"],
        },
        "training_performed": False,
        "optimizer_updates": 0,
        "performance": {
            "page_cache_warm": page_cache_warm,
            "gpu_clone_seconds": clone_seconds,
            "search_seconds": sum(
                float(value["search_seconds"]) for value in shard_receipts
            ),
            "rerank_seconds": sum(
                float(value["rerank_seconds"]) for value in shard_receipts
            ),
            "total_seconds": time.monotonic() - started,
            "peak_rss_gib": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024 ** 2)
            ),
        },
        "runtime": runtime,
    }
    part = seal(body)
    path = os.path.join(output, "part-receipt.json")
    atomic_write_new_json(path, part, immutable=True)
    del resources, gpu, index
    return {**part, "receipt": expected_input_signature(path)}


def _assemble_part_roots(
    *,
    output: str,
    roots: Mapping[str, str],
    excluded: np.ndarray,
    nprobe: int,
    search_width: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target_path = os.path.join(output, "canonical-targets.i32")
    degree_path = os.path.join(output, "valid-degrees.u8")
    target_fd, _ = _fresh_raw_file(
        target_path, bytes_count=ROW_COUNT * K * 4
    )
    degree_fd = -1
    try:
        offset = 0
        for corpus, spec in CORPUS_SPECS.items():
            root = os.path.join(str(roots[corpus]), "shards")
            round_id = ROUND_BY_CORPUS[corpus]
            for start in range(spec["start"], spec["stop"], SHARD_ROWS):
                stop = min(start + SHARD_ROWS, spec["stop"])
                shard = start // SHARD_ROWS
                shard_path, receipt_path = _shard_paths(root, shard)
                if _validate_shard(
                    target_path=shard_path,
                    receipt_path=receipt_path,
                    start=start,
                    stop=stop,
                    nprobe=nprobe,
                    search_width=search_width,
                    round_id=round_id,
                ) is None:
                    raise Round0088Error(
                        f"assembly found missing {corpus} shard {shard}"
                    )
                values = np.load(
                    shard_path, mmap_mode="r", allow_pickle=False
                )
                payload = memoryview(values).cast("B")
                written = 0
                while written < len(payload):
                    count = os.pwrite(
                        target_fd,
                        payload[written:written + (1 << 30)],
                        offset + written,
                    )
                    if count <= 0:
                        raise Round0088Error("short graph assembly write")
                    written += count
                offset += len(payload)
        if offset != ROW_COUNT * K * 4:
            raise Round0088Error("assembled 150M target bytes do not close")
        os.fsync(target_fd)
        os.fchmod(target_fd, 0o444)
        os.close(target_fd)
        target_fd = -1

        degree_fd, _ = _fresh_raw_file(
            degree_path, bytes_count=ROW_COUNT
        )
        degrees = np.memmap(
            degree_path, dtype="u1", mode="r+", shape=(ROW_COUNT,)
        )
        degrees[:] = K
        degrees[excluded] = 0
        degrees.flush()
        del degrees
        os.fsync(degree_fd)
        os.fchmod(degree_fd, 0o444)
        os.close(degree_fd)
        degree_fd = -1
    except BaseException:
        if target_fd >= 0:
            with contextlib.suppress(OSError):
                os.close(target_fd)
        if degree_fd >= 0:
            with contextlib.suppress(OSError):
                os.close(degree_fd)
        for path in (target_path, degree_path):
            with contextlib.suppress(OSError):
                os.unlink(path)
        raise
    return (
        expected_input_signature(target_path),
        expected_input_signature(degree_path),
    )


def run_assemble(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    if active["manifest"]["round_id"] != ASSEMBLY_ROUND_ID:
        raise Round0088Error("150M assembly received another round")
    substrate = validate_staged_substrate(
        str(job["substrate_manifest"]),
        expected_sha256=str(job["substrate_manifest_sha256"]),
    )
    excluded = np.asarray(
        substrate["eligibility"]["excluded_rows"], dtype=np.int64
    )
    roots = dict(job["part_roots"])
    parts = {
        corpus: validate_part_receipt(
            os.path.join(str(roots[corpus]), "part-receipt.json"),
            expected_sha256=str(job["part_receipt_sha256"][corpus]),
        )
        for corpus in CORPUS_SPECS
    }
    qualifications = {
        value["receipt"]["gpu_qualification"]["sha256"]
        for value in parts.values()
    }
    policies = {
        (
            int(value["receipt"]["nprobe"]),
            int(value["receipt"]["search_width"]),
        )
        for value in parts.values()
    }
    substrates = {
        value["receipt"]["substrate"]["sha256"]
        for value in parts.values()
    }
    qualities = [
        value["receipt"]["quality"] for value in parts.values()
    ]
    if (
        len(qualifications) != 1
        or len(policies) != 1
        or substrates != {substrate["signature"]["sha256"]}
        or any(value != qualities[0] for value in qualities[1:])
    ):
        raise Round0088Error("150M graph parts do not share one contract")
    nprobe, search_width = next(iter(policies))
    output = create_fresh_directory(
        str(job["outputs"][0]),
        label="R0091 canonical balanced-150M graph",
    )
    targets, degrees = _assemble_part_roots(
        output=output,
        roots=roots,
        excluded=excluded,
        nprobe=nprobe,
        search_width=search_width,
    )
    graph_body = {
        "schema": ASSEMBLED_GRAPH_SCHEMA,
        "round_id": ASSEMBLY_ROUND_ID,
        "row_count": ROW_COUNT,
        "input_k": K,
        "source_semantics": (
            "global balanced-150m int8 row; excluded rows degree zero"
        ),
        "destination_policy": (
            "native representative-only GPU IVF-PQ search in the fixed "
            "balanced-150m universe; remove self"
        ),
        "sampling_semantics": (
            "source-uniform and edge-uniform are identical at fixed degree 15"
        ),
        "negative_policy": "uniform-eligibility-retained-rows-nonself",
        "weight_semantics": "unweighted directed k15",
        "inputs": {
            "eligibility": substrate["manifest"]["outputs"]["eligibility"],
            "substrate": substrate["signature"],
            "parts": {
                corpus: value["signature"]
                for corpus, value in parts.items()
            },
        },
        "outputs": {"targets": targets, "degrees": degrees},
        "target_dtype": "<i4",
        "target_shape": [ROW_COUNT, K],
        "degree_dtype": "|u1",
        "degree_shape": [ROW_COUNT],
        "summary": {
            "eligibility_excluded_source_count": ROW_COUNT - RETAINED_ROWS,
            "eligibility_retained_row_count": RETAINED_ROWS,
            "retained_positive_source_count": RETAINED_ROWS,
            "zero_degree_retained_source_count": 0,
            "valid_canonical_edge_count": RETAINED_ROWS * K,
            "degree_histogram": {
                "0": ROW_COUNT - RETAINED_ROWS,
                str(K): RETAINED_ROWS,
            },
        },
        "candidate_generator": {
            "index_type": "GpuIndexIVFPQ",
            "source_index_rows": SOURCE_ROWS,
            "physically_filtered_index_rows": RETAINED_ROWS,
            "nprobe": nprobe,
            "search_width": search_width,
            "index_search_width": search_width + 1,
            "selected_neighbors": K,
            "exact_rerank": True,
            "candidate_universe": "all retained balanced-150m representatives",
        },
        "quality": qualities[0],
    }
    graph = seal(graph_body)
    graph_path = os.path.join(output, "canonical-graph-v1.json")
    atomic_write_new_json(graph_path, graph, immutable=True)
    receipt_body = {
        "schema": ASSEMBLY_RECEIPT_SCHEMA,
        "round_id": ASSEMBLY_ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "training_performed": False,
        "optimizer_updates": 0,
        "scale_decision_made": False,
        "graph": expected_input_signature(graph_path),
        "substrate": substrate["signature"],
        "parts": {
            corpus: value["signature"] for corpus, value in parts.items()
        },
        "summary": graph["summary"],
        "performance": {
            "total_wall_seconds": time.monotonic() - started,
            "peak_rss_gib": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024 ** 2)
            ),
        },
    }
    receipt = seal(receipt_body)
    receipt_path = os.path.join(output, "receipt.json")
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(receipt_path)}


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected = job if job is not None else active.get("job") or {}
    action = str(selected.get("action"))
    if action == "build_150m_graph_part":
        return run_build_part(active, selected)
    if action == "assemble_150m_graph":
        return run_assemble(active, selected)
    raise Round0088Error(f"unknown split-150M graph action {action!r}")
