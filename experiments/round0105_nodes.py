"""Build and qualify the retained-only diverse-Jina search index."""
from __future__ import annotations

import gc
import json
import os
import tempfile
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import (
    expected_input_signature,
    ordered_array_sha256,
    sha256_bytes,
)
from basemap.output_safety import (
    atomic_save_new_npz,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0104_training import validate_substrate_manifest
from basemap.round0105_search import (
    BENCHMARK_REPEATS,
    BENCHMARK_WARMUP_ROWS,
    BOUNDARY_TIE_ATOL,
    DECISION_SCHEMA,
    DIMENSION,
    ELIGIBILITY_PATH,
    ELIGIBILITY_SHA256,
    EVERY_GROUP_MEAN_FLOOR,
    GLOBAL_MEAN_FLOOR,
    GROUPS,
    INDEX_SCHEMA,
    INDEX_TRAIN_ROWS,
    INDEX_TRAIN_SAMPLE_SHA256,
    INDEX_TRAIN_SEED,
    K,
    NLIST,
    POLICY_GRID,
    PQ_BITS,
    PQ_M,
    QUALIFICATION_SCHEMA,
    QUALITY_GROUP_IDS_SHA256,
    QUALITY_ROWS,
    QUALITY_ROWS_PER_GROUP,
    QUALITY_SAMPLE_SHA256,
    QUALITY_SEED,
    RETAINED_ROWS,
    ROUND_ID,
    ROW_COUNT,
    Round0105Error,
    group_ranges,
    membership,
    sample_retained_rows,
    sample_stratified_rows,
    seal,
    select_cell,
)


ADD_BATCH_ROWS = 200_000
EXACT_BLOCK_ROWS = 131_072
RERANK_BATCH_ROWS = 128


def _peak_rss_gib() -> float:
    import resource

    # Linux reports ru_maxrss in KiB.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024 ** 2)


def _load_sealed(
    path: str,
    *,
    schema: str,
    round_id: str,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    signature = expected_input_signature(path)
    with open(signature["canonical_path"], encoding="utf-8") as handle:
        value = json.load(handle)
    body = {key: item for key, item in value.items() if key != "identity_sha256"}
    from basemap.artifact_identity import canonical_json

    if (
        value.get("schema") != schema
        or value.get("round_id") != round_id
        or value.get("identity_sha256") != sha256_bytes(canonical_json(body))
    ):
        raise Round0105Error(f"{label} seal changed")
    return value, signature


def _substrate_arrays() -> tuple[
    dict[str, Any], np.ndarray, np.memmap, np.memmap
]:
    substrate = validate_substrate_manifest(verify_payloads=False)
    eligibility = expected_input_signature(ELIGIBILITY_PATH)
    if eligibility["sha256"] != ELIGIBILITY_SHA256:
        raise Round0105Error("R0087 eligibility bytes changed")
    with np.load(eligibility["canonical_path"], allow_pickle=False) as archive:
        excluded = np.asarray(archive["excluded_rows"], dtype=np.int64)
    if (
        len(excluded) != ROW_COUNT - RETAINED_ROWS
        or np.any(excluded[1:] <= excluded[:-1])
        or excluded[0] < 0
        or excluded[-1] >= ROW_COUNT
    ):
        raise Round0105Error("R0087 retained selector changed")
    outputs = substrate["manifest"]["outputs"]
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
    return substrate, excluded, encoded, scales


def _normalized_rows(
    encoded: np.ndarray,
    scales: np.ndarray,
    rows: np.ndarray,
) -> np.ndarray:
    ids = np.asarray(rows, dtype=np.int64)
    row_scales = np.asarray(scales[ids], dtype=np.float32)
    values = np.asarray(encoded[ids], dtype=np.float32)
    if (
        values.ndim != 2
        or values.shape != (len(ids), DIMENSION)
        or not np.isfinite(row_scales).all()
        or np.any(row_scales <= 0)
    ):
        raise Round0105Error("native int8-plus-scale rows are malformed")
    values *= row_scales[:, None]
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise Round0105Error("native int8-plus-scale rows are zero/nonfinite")
    values /= norms
    return np.ascontiguousarray(values)


def _gpu_options() -> Any:
    import faiss

    options = faiss.GpuClonerOptions()
    options.indicesOptions = faiss.INDICES_64_BIT
    # For IndexIVFPQ, GpuClonerOptions.useFloat16 maps to the
    # GpuIndexIVFPQConfig.useFloat16LookupTables C++ field.  This is required
    # for PQ96x8 on the local 49,152-byte shared-memory limit; assigning a
    # Python-only useFloat16LookupTables attribute would be ignored by SWIG.
    options.useFloat16 = True
    options.usePrecomputed = True
    return options


def _write_index_new(index: Any, path: str) -> dict[str, Any]:
    import faiss

    if os.path.lexists(path):
        raise FileExistsError(f"refuse existing FAISS output: {path}")
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=os.path.dirname(path),
    )
    os.close(fd)
    try:
        faiss.write_index(index, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path, follow_symlinks=False)
        directory_fd = os.open(os.path.dirname(path), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return expected_input_signature(path)


def _index_geometry(index: Any) -> dict[str, Any]:
    return {
        "class": type(index).__name__,
        "dimension": int(index.d),
        "ntotal": int(index.ntotal),
        "nlist": int(index.nlist),
        "pq_m": int(index.pq.M),
        "pq_bits": int(index.pq.nbits),
        "code_size": int(index.code_size),
        "metric_type": int(index.metric_type),
        "is_trained": bool(index.is_trained),
    }


def _require_geometry(index: Any, *, ntotal: int) -> None:
    import faiss

    observed = _index_geometry(index)
    expected = {
        "class": "IndexIVFPQ",
        "dimension": DIMENSION,
        "ntotal": ntotal,
        "nlist": NLIST,
        "pq_m": PQ_M,
        "pq_bits": PQ_BITS,
        "code_size": PQ_M,
        "metric_type": int(faiss.METRIC_INNER_PRODUCT),
        "is_trained": True,
    }
    if observed != expected:
        raise Round0105Error(
            f"unexpected retained-only IVF-PQ geometry: {observed}"
        )


def _retained_batch(
    excluded: np.ndarray,
    *,
    start: int,
    stop: int,
) -> np.ndarray:
    rows = np.arange(start, stop, dtype=np.int64)
    left = int(np.searchsorted(excluded, start, side="left"))
    right = int(np.searchsorted(excluded, stop, side="left"))
    if right > left:
        keep = np.ones(stop - start, dtype=bool)
        keep[excluded[left:right] - start] = False
        rows = rows[keep]
    return rows


def _validate_index_ids(index: Any, excluded: np.ndarray) -> dict[str, Any]:
    import faiss

    seen = np.zeros(ROW_COUNT, dtype=bool)
    list_sizes = np.empty(NLIST, dtype=np.int64)
    for list_id in range(NLIST):
        size = int(index.invlists.list_size(list_id))
        list_sizes[list_id] = size
        if not size:
            continue
        ids = np.array(
            faiss.rev_swig_ptr(index.invlists.get_ids(list_id), size),
            dtype=np.int64,
            copy=True,
        )
        if (
            np.any(ids < 0)
            or np.any(ids >= ROW_COUNT)
            or len(np.unique(ids)) != size
            or np.any(seen[ids])
        ):
            raise Round0105Error("retained-only index IDs are invalid")
        seen[ids] = True
    if (
        int(seen.sum()) != RETAINED_ROWS
        or np.any(seen[excluded])
        or int(list_sizes.sum()) != RETAINED_ROWS
        or int((~seen).sum()) != len(excluded)
    ):
        raise Round0105Error("retained-only index coverage changed")
    return {
        "list_size_min": int(list_sizes.min()),
        "list_size_mean": float(list_sizes.mean()),
        "list_size_p90": float(np.percentile(list_sizes, 90)),
        "list_size_max": int(list_sizes.max()),
        "seen_retained_rows": int(seen.sum()),
        "excluded_rows_absent": True,
        "global_ids_unique": True,
    }


def run_build_index(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    import faiss
    import torch

    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0105 retained-only IVF-PQ index"
    )
    started = time.monotonic()
    substrate, excluded, encoded, scales = _substrate_arrays()
    sample = sample_retained_rows(
        excluded,
        count=INDEX_TRAIN_ROWS,
        seed=INDEX_TRAIN_SEED,
    )
    sample_sha = sha256_bytes(sample.tobytes())
    if sample_sha != INDEX_TRAIN_SAMPLE_SHA256:
        raise Round0105Error("registered index-training sample changed")
    vectors = _normalized_rows(encoded, scales, sample)
    cpu = faiss.IndexIVFPQ(
        faiss.IndexFlatIP(DIMENSION),
        DIMENSION,
        NLIST,
        PQ_M,
        PQ_BITS,
        faiss.METRIC_INNER_PRODUCT,
    )
    resource = faiss.StandardGpuResources()
    resource.setTempMemory(1 << 30)
    gpu = faiss.index_cpu_to_gpu(resource, 0, cpu, _gpu_options())
    gpu.cp.seed = INDEX_TRAIN_SEED
    gpu.cp.niter = 25
    gpu.cp.spherical = True
    gpu.pq.cp.seed = INDEX_TRAIN_SEED
    gpu.pq.cp.niter = 25
    train_started = time.monotonic()
    gpu.train(vectors)
    train_seconds = time.monotonic() - train_started
    del vectors

    add_started = time.monotonic()
    added = 0
    for start in range(0, ROW_COUNT, ADD_BATCH_ROWS):
        stop = min(start + ADD_BATCH_ROWS, ROW_COUNT)
        rows = _retained_batch(excluded, start=start, stop=stop)
        if len(rows):
            gpu.add_with_ids(_normalized_rows(encoded, scales, rows), rows)
            added += len(rows)
    add_seconds = time.monotonic() - add_started
    if added != RETAINED_ROWS or int(gpu.ntotal) != RETAINED_ROWS:
        raise Round0105Error("retained-only GPU index count changed")
    gpu_geometry = {
        "class": type(gpu).__name__,
        "device": 0,
        "ntotal": int(gpu.ntotal),
        "nprobe_at_clone": int(gpu.nprobe),
        "faiss_gpu_count": int(faiss.get_num_gpus()),
        "cuda_device": torch.cuda.get_device_name(0),
        "indices": "int64-global-row-id",
        "use_float16_lookup_tables": True,
        "use_float16_coarse_quantizer": False,
        "use_precomputed": True,
    }
    assembled = faiss.index_gpu_to_cpu(gpu)
    _require_geometry(assembled, ntotal=RETAINED_ROWS)
    id_validation = _validate_index_ids(assembled, excluded)
    index_path = os.path.join(output, "jina-diverse-25m-retained.ivfpq")
    index_signature = _write_index_new(assembled, index_path)
    receipt = seal({
        "schema": INDEX_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "substrate": substrate["signature"],
        "eligibility": expected_input_signature(ELIGIBILITY_PATH),
        "index": index_signature,
        "geometry": _index_geometry(assembled),
        "gpu_clone_receipt": gpu_geometry,
        "id_validation": id_validation,
        "index_training": {
            "method": "uniform retained global rows before final sort",
            "rows": len(sample),
            "seed": INDEX_TRAIN_SEED,
            "sample_sha256": sample_sha,
            "coarse_iterations": 25,
            "coarse_spherical": True,
            "pq_iterations": 25,
            "input_semantics": (
                "native signed-int8 times row fp16 scale, normalized fp32"
            ),
        },
        "performance": {
            "train_seconds": train_seconds,
            "add_seconds": add_seconds,
            "wall_seconds": time.monotonic() - started,
            "peak_rss_gib": _peak_rss_gib(),
        },
        "training_performed": False,
        "optimizer_updates": 0,
    })
    receipt_path = os.path.join(output, "index-receipt.json")
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    del gpu, assembled, encoded, scales
    gc.collect()
    return {**receipt, "receipt": expected_input_signature(receipt_path)}


def _exact_truth(
    *,
    encoded: np.ndarray,
    scales: np.ndarray,
    excluded: np.ndarray,
    sample: np.ndarray,
    k: int = K,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    import torch
    import torch.nn.functional as functional

    if not torch.cuda.is_available():
        raise Round0105Error("R0105 exact truth requires CUDA")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()
    query = torch.from_numpy(
        _normalized_rows(encoded, scales, sample)
    ).to(device)
    best_values = torch.full(
        (len(sample), k + 1),
        -torch.inf,
        dtype=torch.float32,
        device=device,
    )
    best_ids = torch.full(
        (len(sample), k + 1),
        -1,
        dtype=torch.int64,
        device=device,
    )
    blocks = 0
    try:
        with torch.inference_mode():
            for start in range(0, ROW_COUNT, EXACT_BLOCK_ROWS):
                stop = min(start + EXACT_BLOCK_ROWS, ROW_COUNT)
                ids = _retained_batch(excluded, start=start, stop=stop)
                candidates = torch.from_numpy(
                    _normalized_rows(encoded, scales, ids)
                ).to(device)
                similarity = query @ candidates.T
                positions = np.searchsorted(ids, sample)
                present = positions < len(ids)
                indices = np.flatnonzero(present)
                present[indices] = ids[positions[indices]] == sample[indices]
                if np.any(present):
                    rows = torch.from_numpy(
                        np.flatnonzero(present).astype(np.int64)
                    ).to(device)
                    columns = torch.from_numpy(
                        positions[present].astype(np.int64)
                    ).to(device)
                    similarity[rows, columns] = -torch.inf
                local_values, local_positions = torch.topk(
                    similarity,
                    min(k + 1, len(ids)),
                    dim=1,
                    largest=True,
                    sorted=True,
                )
                id_tensor = torch.from_numpy(ids).to(device)
                local_ids = id_tensor[local_positions]
                merged_values = torch.cat((best_values, local_values), dim=1)
                merged_ids = torch.cat((best_ids, local_ids), dim=1)
                best_values, order = torch.topk(
                    merged_values,
                    k + 1,
                    dim=1,
                    largest=True,
                    sorted=True,
                )
                best_ids = torch.gather(merged_ids, 1, order)
                blocks += 1
                del (
                    candidates,
                    similarity,
                    local_values,
                    local_positions,
                    id_tensor,
                    local_ids,
                    merged_values,
                    merged_ids,
                    order,
                )
        values = best_values.cpu().numpy()
        neighbors = best_ids[:, :k].cpu().numpy().astype(np.int64, copy=False)
        margins = values[:, k - 1] - values[:, k]
        ties = np.abs(margins) <= BOUNDARY_TIE_ATOL
        if (
            np.any(neighbors < 0)
            or np.any(neighbors == sample[:, None])
            or np.any(np.diff(np.sort(neighbors, axis=1), axis=1) == 0)
            or np.any(membership(excluded, neighbors.reshape(-1)))
        ):
            raise Round0105Error("exact retained-only truth is malformed")
        return neighbors, ties, margins, {
            "wall_seconds": time.monotonic() - started,
            "candidate_blocks": blocks,
            "block_rows": EXACT_BLOCK_ROWS,
            "matmul_dtype": "float32",
            "tf32_allowed": False,
            "boundary_tie_atol": BOUNDARY_TIE_ATOL,
            "peak_allocated_gib": (
                torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            ),
            "peak_reserved_gib": (
                torch.cuda.max_memory_reserved(device) / (1024 ** 3)
            ),
        }
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        del query, best_values, best_ids
        torch.cuda.empty_cache()


def _clean_search(
    raw: np.ndarray,
    *,
    sources: np.ndarray,
    width: int,
    excluded: np.ndarray,
) -> tuple[np.ndarray, int]:
    candidates = np.asarray(raw, dtype=np.int64)
    source = np.asarray(sources, dtype=np.int64)
    if (
        candidates.ndim != 2
        or candidates.shape[0] != len(source)
        or candidates.shape[1] < width
    ):
        raise Round0105Error("IVF-PQ output has wrong geometry")
    valid = (
        (candidates >= 0)
        & (candidates < ROW_COUNT)
        & (candidates != source[:, None])
    )
    counts = valid.sum(axis=1)
    if np.any(counts < width):
        row = int(np.flatnonzero(counts < width)[0])
        raise Round0105Error(
            f"query {int(source[row])} returned fewer than {width} nonself rows"
        )
    ranks = np.cumsum(valid, axis=1)
    chosen = candidates[valid & (ranks <= width)].reshape(len(source), width)
    if (
        np.any(chosen < 0)
        or np.any(chosen == source[:, None])
        or np.any(np.diff(np.sort(chosen, axis=1), axis=1) == 0)
        or np.any(membership(excluded, chosen.reshape(-1)))
    ):
        raise Round0105Error("IVF-PQ retained candidate cleanup failed")
    return chosen, int(
        np.count_nonzero(np.any(candidates == source[:, None], axis=1))
    )


def _exact_rerank(
    *,
    queries: np.ndarray,
    shortlist: np.ndarray,
    encoded: np.ndarray,
    scales: np.ndarray,
    k: int = K,
) -> tuple[np.ndarray, dict[str, Any]]:
    query = np.asarray(queries, dtype=np.float32)
    candidates = np.asarray(shortlist, dtype=np.int64)
    if (
        query.shape != (len(candidates), DIMENSION)
        or candidates.ndim != 2
        or candidates.shape[1] < k
    ):
        raise Round0105Error("exact-rerank inputs have invalid geometry")
    output = np.empty((len(query), k), dtype=np.int64)
    started = time.monotonic()
    for start in range(0, len(query), RERANK_BATCH_ROWS):
        stop = min(start + RERANK_BATCH_ROWS, len(query))
        ids = candidates[start:stop]
        candidate_scales = np.asarray(scales[ids], dtype=np.float32)
        vectors = np.asarray(encoded[ids], dtype=np.float32)
        if (
            not np.isfinite(candidate_scales).all()
            or np.any(candidate_scales <= 0)
        ):
            raise Round0105Error("exact-rerank scales are malformed")
        vectors *= candidate_scales[:, :, None]
        norms = np.linalg.norm(vectors, axis=2)
        if not np.isfinite(norms).all() or np.any(norms <= 0):
            raise Round0105Error("exact-rerank vectors are malformed")
        scores = np.einsum(
            "bd,bkd->bk",
            query[start:stop],
            vectors,
            optimize=True,
        )
        scores /= norms
        order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
        output[start:stop] = np.take_along_axis(ids, order, axis=1)
    if np.any(np.diff(np.sort(output, axis=1), axis=1) == 0):
        raise Round0105Error("exact-rerank output contains duplicates")
    return output, {
        "wall_seconds": time.monotonic() - started,
        "shortlist_width": int(candidates.shape[1]),
        "selected_neighbors": k,
        "batch_rows": RERANK_BATCH_ROWS,
        "score_dtype": "float32",
        "vector_source": "native-int8-plus-fp16-scale",
        "tie_policy": "stable-IVFPQ-shortlist-order",
    }


def _search_and_rerank(
    gpu: Any,
    *,
    nprobe: int,
    width: int,
    queries: np.ndarray,
    sample: np.ndarray,
    excluded: np.ndarray,
    encoded: np.ndarray,
    scales: np.ndarray,
    k: int = K,
) -> tuple[np.ndarray, dict[str, Any]]:
    gpu.nprobe = nprobe
    started = time.monotonic()
    _distances, raw = gpu.search(queries, width + 1)
    search_seconds = time.monotonic() - started
    shortlist, self_seen = _clean_search(
        raw, sources=sample, width=width, excluded=excluded
    )
    selected, rerank = _exact_rerank(
        queries=queries,
        shortlist=shortlist,
        encoded=encoded,
        scales=scales,
        k=k,
    )
    return selected, {
        "search_seconds": search_seconds,
        "self_returned": self_seen,
        "exact_rerank": rerank,
        "total_seconds": search_seconds + float(rerank["wall_seconds"]),
    }


def _policy_metrics(
    selected: np.ndarray,
    exact: np.ndarray,
    *,
    group_ids: np.ndarray,
    unambiguous: np.ndarray,
) -> dict[str, Any]:
    overlap = (
        selected[:, :, None] == exact[:, None, :]
    ).any(axis=2).sum(axis=1) / K
    clear = overlap[unambiguous]
    by_group: dict[str, Any] = {}
    passes: list[bool] = []
    for group_id, group in enumerate(GROUPS):
        mask = (group_ids == group_id) & unambiguous
        values = overlap[mask]
        mean = float(values.mean()) if len(values) else None
        passed = mean is not None and mean >= EVERY_GROUP_MEAN_FLOOR
        passes.append(passed)
        by_group[group] = {
            "registered_rows": int((group_ids == group_id).sum()),
            "boundary_ties_excluded": int(
                ((group_ids == group_id) & ~unambiguous).sum()
            ),
            "unambiguous_rows": int(mask.sum()),
            "mean_recall_at_15_unambiguous": mean,
            "passes_floor": passed,
        }
    mean = float(clear.mean()) if len(clear) else None
    complete = (
        selected.shape == (QUALITY_ROWS, K)
        and np.all(selected >= 0)
        and np.all(np.diff(np.sort(selected, axis=1), axis=1) != 0)
    )
    return {
        "mean_recall_at_15": float(overlap.mean()),
        "mean_recall_at_15_unambiguous": mean,
        "p10_recall_at_15_unambiguous": (
            float(np.percentile(clear, 10)) if len(clear) else None
        ),
        "passes_global_floor": mean is not None and mean >= GLOBAL_MEAN_FLOOR,
        "passes_every_group_floor": all(passes),
        "all_rows_complete": bool(complete),
        "by_group": by_group,
    }


def _benchmark(
    gpu: Any,
    *,
    nprobe: int,
    width: int,
    queries: np.ndarray,
    sample: np.ndarray,
    excluded: np.ndarray,
    encoded: np.ndarray,
    scales: np.ndarray,
) -> dict[str, Any]:
    _search_and_rerank(
        gpu,
        nprobe=nprobe,
        width=width,
        queries=queries[:BENCHMARK_WARMUP_ROWS],
        sample=sample[:BENCHMARK_WARMUP_ROWS],
        excluded=excluded,
        encoded=encoded,
        scales=scales,
    )
    repeats: list[float] = []
    for _ in range(BENCHMARK_REPEATS):
        _selected, timing = _search_and_rerank(
            gpu,
            nprobe=nprobe,
            width=width,
            queries=queries,
            sample=sample,
            excluded=excluded,
            encoded=encoded,
            scales=scales,
        )
        repeats.append(float(timing["total_seconds"]) / len(queries))
    return {
        "queries": len(queries),
        "warmup_queries": BENCHMARK_WARMUP_ROWS,
        "repeats": BENCHMARK_REPEATS,
        "repeats_seconds_per_query": repeats,
        "median_wall_seconds_per_query": float(np.median(repeats)),
    }


def run_qualify_index(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    import faiss
    import torch

    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0105 retained search qualification"
    )
    started = time.monotonic()
    substrate, excluded, encoded, scales = _substrate_arrays()
    index_receipt, index_receipt_signature = _load_sealed(
        str(job["index_receipt"]),
        schema=INDEX_SCHEMA,
        round_id=ROUND_ID,
        label="R0105 index receipt",
    )
    index_signature = expected_input_signature(str(job["index"]))
    if (
        index_receipt.get("index") != index_signature
        or index_receipt.get("substrate") != substrate["signature"]
        or index_receipt.get("release_sha") != active["manifest"]["release_sha"]
    ):
        raise Round0105Error("R0105 index lineage changed")

    ranges = group_ranges(substrate["manifest"])
    sample, group_ids = sample_stratified_rows(excluded, ranges)
    sample_sha = sha256_bytes(sample.tobytes())
    group_ids_sha = sha256_bytes(group_ids.tobytes())
    if (
        sample_sha != QUALITY_SAMPLE_SHA256
        or group_ids_sha != QUALITY_GROUP_IDS_SHA256
    ):
        raise Round0105Error("registered stratified quality sample changed")
    exact, ties, boundary_margins, exact_performance = _exact_truth(
        encoded=encoded,
        scales=scales,
        excluded=excluded,
        sample=sample,
    )
    unambiguous = ~ties
    truth_path = os.path.join(output, "exact-truth.npz")
    atomic_save_new_npz(
        truth_path,
        immutable=True,
        sample_rows=sample,
        group_ids=group_ids,
        exact_neighbors=exact,
        boundary_ties=ties,
        boundary_margins=boundary_margins.astype(np.float32),
    )
    truth_signature = expected_input_signature(truth_path)
    queries = _normalized_rows(encoded, scales, sample)

    cpu = faiss.read_index(index_signature["canonical_path"])
    _require_geometry(cpu, ntotal=RETAINED_ROWS)
    resource = faiss.StandardGpuResources()
    resource.setTempMemory(1 << 30)
    gpu = faiss.index_cpu_to_gpu(resource, 0, cpu, _gpu_options())
    cells: dict[str, Any] = {}
    for nprobe, width in POLICY_GRID:
        selected, execution = _search_and_rerank(
            gpu,
            nprobe=nprobe,
            width=width,
            queries=queries,
            sample=sample,
            excluded=excluded,
            encoded=encoded,
            scales=scales,
        )
        cells[f"nprobe-{nprobe}-width-{width}"] = {
            "nprobe": nprobe,
            "shortlist_width": width,
            **_policy_metrics(
                selected,
                exact,
                group_ids=group_ids,
                unambiguous=unambiguous,
            ),
            "execution": execution,
            "benchmark": None,
        }
    for cell in cells.values():
        if (
            cell["passes_global_floor"]
            and cell["passes_every_group_floor"]
            and cell["all_rows_complete"]
        ):
            cell["benchmark"] = _benchmark(
                gpu,
                nprobe=int(cell["nprobe"]),
                width=int(cell["shortlist_width"]),
                queries=queries,
                sample=sample,
                excluded=excluded,
                encoded=encoded,
                scales=scales,
            )
    selected = select_cell(cells)
    group_denominators = {
        group: int(((group_ids == group_id) & unambiguous).sum())
        for group_id, group in enumerate(GROUPS)
    }
    checks = {
        "quality_sample_sha_matches": sample_sha == QUALITY_SAMPLE_SHA256,
        "quality_group_ids_sha_matches": (
            group_ids_sha == QUALITY_GROUP_IDS_SHA256
        ),
        "exactly_256_rows_per_group": all(
            int((group_ids == group_id).sum()) == QUALITY_ROWS_PER_GROUP
            for group_id in range(len(GROUPS))
        ),
        "every_group_has_unambiguous_rows": all(
            value > 0 for value in group_denominators.values()
        ),
        "all_registered_cells_present": set(cells) == {
            f"nprobe-{nprobe}-width-{width}"
            for nprobe, width in POLICY_GRID
        },
        "retained_only_index": (
            int(cpu.ntotal) == RETAINED_ROWS
            and index_receipt["id_validation"]["excluded_rows_absent"] is True
        ),
        "one_local_cuda_device": (
            faiss.get_num_gpus() == 1 and torch.cuda.device_count() == 1
        ),
        "no_graph_built": True,
        "no_map_training_performed": True,
        "no_map_decision_made": True,
    }
    validity_passed = all(value is True for value in checks.values())
    qualification = seal({
        "schema": QUALIFICATION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "validity_passed": validity_passed,
        "failed_checks": sorted(
            key for key, value in checks.items() if value is not True
        ),
        "substrate": substrate["signature"],
        "eligibility": expected_input_signature(ELIGIBILITY_PATH),
        "index_receipt": index_receipt_signature,
        "index": index_signature,
        "geometry": _index_geometry(cpu),
        "truth": truth_signature,
        "truth_arrays": {
            "sample_rows_ordered_sha256": ordered_array_sha256(sample),
            "group_ids_ordered_sha256": ordered_array_sha256(group_ids),
            "exact_neighbors_ordered_sha256": ordered_array_sha256(exact),
            "boundary_ties_ordered_sha256": ordered_array_sha256(ties),
            "boundary_margins_ordered_sha256": ordered_array_sha256(
                boundary_margins.astype(np.float32)
            ),
        },
        "quality": {
            "global_mean_floor": GLOBAL_MEAN_FLOOR,
            "every_group_mean_floor": EVERY_GROUP_MEAN_FLOOR,
            "sample_seed": QUALITY_SEED,
            "rows_per_group": QUALITY_ROWS_PER_GROUP,
            "sample_rows": len(sample),
            "sample_sha256": sample_sha,
            "group_ids_sha256": group_ids_sha,
            "groups": list(GROUPS),
            "boundary_ties": int(ties.sum()),
            "boundary_tie_atol": BOUNDARY_TIE_ATOL,
            "unambiguous_rows": int(unambiguous.sum()),
            "group_denominators": group_denominators,
        },
        "cells": cells,
        "selected": selected,
        "performance": {
            "exact_truth": exact_performance,
            "wall_seconds": time.monotonic() - started,
            "peak_rss_gib": _peak_rss_gib(),
        },
        "checks": checks,
        "training_performed": False,
        "optimizer_updates": 0,
        "map_decision_made": False,
    })
    qualification_path = os.path.join(
        output, "search-qualification.json"
    )
    atomic_write_new_json(
        qualification_path, qualification, immutable=True
    )
    decision = seal({
        "schema": DECISION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "validity_passed": validity_passed,
        "qualification": expected_input_signature(qualification_path),
        "selected": selected,
        "outcome": (
            "qualified"
            if selected is not None
            else "valid-negative-no-registered-cell-passed"
        ),
        "graph_build_released": selected is not None,
        "training_performed": False,
        "optimizer_updates": 0,
        "map_decision_made": False,
    })
    decision_path = os.path.join(output, "search-policy-decision.json")
    atomic_write_new_json(decision_path, decision, immutable=True)
    if not validity_passed:
        raise Round0105Error(
            "R0105 qualification invalid: "
            + ", ".join(qualification["failed_checks"])
        )
    return {**decision, "receipt": expected_input_signature(decision_path)}


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID or job is None:
        raise Round0105Error("R0105 handler requires its exact round/job")
    action = job.get("action")
    if action == "build_index":
        return run_build_index(active, job)
    if action == "qualify_index":
        return run_qualify_index(active, job)
    raise Round0105Error(f"unknown R0105 action: {action!r}")
