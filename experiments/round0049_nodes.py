"""Fresh-process CPU handlers for the balanced-60M Round 0049 substrate."""
from __future__ import annotations

import contextlib
import json
import math
import os
import resource
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from basemap.int8_eligibility import load_int8_eligibility
from basemap.output_safety import (
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
    refuse_existing,
)
from basemap.round0034_pipeline import GRAPH_SCHEMA
from basemap.round0049_program import (
    CORPUS_INTERVALS,
    DIMENSION,
    INDEX_PATH,
    INDEX_SHA256,
    K,
    ROUND_ID,
    ROW_COUNT,
    SOURCE_ELIGIBILITY_PATH,
    SOURCE_ELIGIBILITY_SHA256,
    SOURCE_INT8_PATH,
    SOURCE_INT8_SHA256,
    SOURCE_ROWS,
    SOURCE_SCALES_PATH,
    SOURCE_SCALES_SHA256,
    Round0049Error,
    _seal,
    compact_to_global,
    global_to_compact,
    validate_substrate_manifest,
    write_subset_eligibility,
)


SUBSTRATE_SCHEMA = "round0049-balanced-60m-substrate-v1"
GRAPH_RECEIPT_SCHEMA = "round0049-balanced-60m-graph-receipt-v1"
QUALITY_RECEIPT_SCHEMA = "round0049-balanced-60m-candidate-quality-v1"
NPROBE_SWEEP_RECEIPT_SCHEMA = "round0058-balanced-60m-nprobe-sweep-v1"
DEFAULT_NPROBE = 16
# R0047 established the smallest accepted policy as an nprobe-64 IVF-PQ
# shortlist of 128 retained, nonself rows followed by exact-vector reranking.
# Request one extra index row because the query itself is eligible at search
# time and is returned for almost every query.
SEARCH_WIDTH = 128
INDEX_SEARCH_WIDTH = SEARCH_WIDTH + 1
SEARCH_BATCH_ROWS = 10_000
RERANK_BATCH_ROWS = 512
SHARD_ROWS = 100_000
DEFAULT_THREADS = 24
QUALITY_SAMPLE_ROWS = 1_024
QUALITY_SEED = 49
EXACT_BLOCK_ROWS = 262_144
MEAN_RECALL_FLOOR = 0.90


def _receipt_body(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in receipt.items()
        if key != "identity_sha256"
    }


def _quality_authority_mean_recall(
    receipt: Mapping[str, Any],
    *,
    nprobe: int,
) -> float:
    if receipt.get("identity_sha256") != sha256_bytes(
        canonical_json(_receipt_body(receipt))
    ):
        raise Round0049Error("balanced-60M quality authority seal is invalid")
    schema = receipt.get("schema")
    if schema == QUALITY_RECEIPT_SCHEMA:
        mean = float(
            receipt.get("recall", {}).get(
                "mean_recall_at_15_unambiguous",
                -1,
            )
        )
        if (
            receipt.get("validity_passed") is not True
            or int(
                receipt.get("candidate_generator", {}).get(
                    "nprobe",
                    -1,
                )
            )
            != nprobe
            or int(
                receipt.get("candidate_generator", {}).get(
                    "search_width",
                    -1,
                )
            )
            != SEARCH_WIDTH
            or receipt.get("candidate_generator", {}).get(
                "exact_rerank"
            )
            is not True
            or mean < MEAN_RECALL_FLOOR
        ):
            raise Round0049Error(
                "R0049 candidate-quality receipt is not a passing policy"
            )
        return mean
    if schema == NPROBE_SWEEP_RECEIPT_SCHEMA:
        row = (receipt.get("rows_by_nprobe") or {}).get(str(nprobe))
        generator = receipt.get("candidate_generator") or {}
        mean = float(
            (row or {}).get("mean_recall_at_15_unambiguous", -1)
        )
        if (
            receipt.get("validity_passed") is not True
            or receipt.get("training_performed") is not False
            or int(receipt.get("optimizer_updates", -1)) != 0
            or int(receipt.get("selected_nprobe", -1)) != nprobe
            or row is None
            or row.get("passes_mean_floor") is not True
            or mean < MEAN_RECALL_FLOOR
            or int(generator.get("search_width", -1)) != SEARCH_WIDTH
            or int(generator.get("index_search_width", -1))
            != INDEX_SEARCH_WIDTH
            or int(generator.get("selected_neighbors", -1)) != K
            or generator.get("native_representative_selector") is not True
            or generator.get("exact_rerank") is not True
        ):
            raise Round0049Error(
                "R0058 nprobe sweep does not authorize this graph policy"
            )
        return mean
    raise Round0049Error("unknown balanced-60M quality authority schema")


def _copy_intervals(
    source: str,
    destination: str,
    *,
    row_bytes: int,
    intervals: tuple[tuple[int, int], ...] = CORPUS_INTERVALS,
) -> dict[str, Any]:
    """Copy disjoint row intervals into one compact immutable raw file."""
    destination = refuse_existing(
        destination,
        label="Round 0049 compact substrate output",
    )
    source_fd = os.open(
        source,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
    )
    destination_fd = os.open(
        destination,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    copied = 0
    started = time.monotonic()
    try:
        for start, stop in intervals:
            os.lseek(source_fd, start * row_bytes, os.SEEK_SET)
            remaining = (stop - start) * row_bytes
            while remaining:
                requested = min(remaining, 1 << 30)
                count = os.copy_file_range(
                    source_fd,
                    destination_fd,
                    requested,
                )
                if count <= 0:
                    raise Round0049Error(
                        f"short interval copy from {source}"
                    )
                remaining -= count
                copied += count
        os.fsync(destination_fd)
        os.fchmod(destination_fd, 0o444)
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(destination_fd)
        destination_fd = -1
        with contextlib.suppress(OSError):
            os.unlink(destination)
        raise
    finally:
        os.close(source_fd)
        if destination_fd >= 0:
            os.close(destination_fd)
    return {
        "bytes": copied,
        "wall_seconds": time.monotonic() - started,
        "signature": expected_input_signature(destination),
    }


def run_build_substrate(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    output = create_fresh_directory(
        str(job["outputs"][0]),
        label="Round 0049 balanced-60M substrate",
    )
    started = time.monotonic()
    source_signatures = {
        "int8": expected_input_signature(SOURCE_INT8_PATH),
        "scales": expected_input_signature(SOURCE_SCALES_PATH),
        "eligibility": expected_input_signature(SOURCE_ELIGIBILITY_PATH),
    }
    expected = {
        "int8": SOURCE_INT8_SHA256,
        "scales": SOURCE_SCALES_SHA256,
        "eligibility": SOURCE_ELIGIBILITY_SHA256,
    }
    if {
        key: value["sha256"] for key, value in source_signatures.items()
    } != expected:
        raise Round0049Error("registered 150M substrate inputs changed")

    int8_path = os.path.join(output, "embeddings.i8")
    scales_path = os.path.join(output, "scales.f16")
    eligibility_path = os.path.join(
        output,
        "minilm-balanced-60m-row-eligibility-v1.npz",
    )
    int8_copy = _copy_intervals(
        SOURCE_INT8_PATH,
        int8_path,
        row_bytes=DIMENSION,
    )
    scales_copy = _copy_intervals(
        SOURCE_SCALES_PATH,
        scales_path,
        row_bytes=2,
    )
    if (
        int8_copy["bytes"] != ROW_COUNT * DIMENSION
        or scales_copy["bytes"] != ROW_COUNT * 2
    ):
        raise Round0049Error("balanced-60M compact copy has wrong size")
    eligibility = write_subset_eligibility(eligibility_path)
    summary = eligibility["metadata"]["summary"]
    body = {
        "schema": SUBSTRATE_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "row_count": ROW_COUNT,
        "dimension": DIMENSION,
        "corpus_order": ["fineweb", "redpajama", "pile"],
        "global_150m_intervals": [list(value) for value in CORPUS_INTERVALS],
        "compact_row_policy": (
            "concatenate first 20M global rows from each 50M corpus interval"
        ),
        "quantization": (
            "byte-identical subset of R0025 per-row symmetric int8 plus "
            "exact fp16 scale"
        ),
        "exact_family_policy": (
            "recompute representative membership after subset restriction"
        ),
        "inputs": source_signatures,
        "outputs": {
            "int8": int8_copy["signature"],
            "scales": scales_copy["signature"],
            "eligibility": eligibility["signature"],
        },
        "eligibility_summary": summary,
        "timing": {
            "int8_copy_seconds": int8_copy["wall_seconds"],
            "scales_copy_seconds": scales_copy["wall_seconds"],
            "total_seconds": time.monotonic() - started,
        },
        "peak_rss_gib": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024 ** 2)
        ),
    }
    manifest = _seal(body)
    path = os.path.join(output, "balanced-60m-substrate-v1.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return {
        **manifest,
        "manifest": expected_input_signature(path),
    }


def _membership(sorted_rows: np.ndarray, values: np.ndarray) -> np.ndarray:
    if not len(sorted_rows):
        return np.zeros(values.shape, dtype=np.bool_)
    positions = np.searchsorted(sorted_rows, values)
    bounded = positions < len(sorted_rows)
    result = np.zeros(values.shape, dtype=np.bool_)
    result[bounded] = (
        sorted_rows[positions[bounded]] == values[bounded]
    )
    return result


def _eligible_selector(
    excluded_compact: np.ndarray,
) -> tuple[Any, list[Any], np.ndarray]:
    """Build a native representative-only selector in 150M ID space."""
    import faiss

    excluded_global = np.ascontiguousarray(
        compact_to_global(excluded_compact),
        dtype=np.int64,
    )
    ranges = [
        faiss.IDSelectorRange(start, stop)
        for start, stop in CORPUS_INTERVALS
    ]
    union = ranges[0]
    keepalive: list[Any] = list(ranges)
    for selector in ranges[1:]:
        union = faiss.IDSelectorOr(union, selector)
        keepalive.append(union)
    excluded = faiss.IDSelectorBatch(excluded_global)
    not_excluded = faiss.IDSelectorNot(excluded)
    retained = faiss.IDSelectorAnd(
        union,
        not_excluded,
    )
    keepalive.extend([excluded, not_excluded, retained])
    return retained, keepalive, excluded_global


def _clean_search(
    raw: np.ndarray,
    *,
    global_sources: np.ndarray,
    candidate_count: int = K,
) -> tuple[np.ndarray, int]:
    candidates = np.asarray(raw, dtype=np.int64)
    sources = np.asarray(global_sources, dtype=np.int64)
    if candidates.ndim != 2 or candidates.shape[0] != len(sources):
        raise Round0049Error("IVF-PQ output has wrong geometry")
    valid = (
        (candidates >= 0)
        & (candidates < SOURCE_ROWS)
        & (candidates != sources[:, None])
    )
    if candidate_count < K or candidate_count > candidates.shape[1]:
        raise Round0049Error("requested clean candidate width is invalid")
    counts = valid.sum(axis=1)
    if np.any(counts < candidate_count):
        row = int(np.flatnonzero(counts < candidate_count)[0])
        raise Round0049Error(
            f"query {int(sources[row])} returned only "
            f"{int(counts[row])} eligible nonself candidates"
    )
    ranks = np.cumsum(valid, axis=1)
    selected = valid & (ranks <= candidate_count)
    cleaned = candidates[selected].reshape(
        len(sources),
        candidate_count,
    )
    compact = global_to_compact(cleaned)
    if (
        np.any(compact < 0)
        or np.any(compact == global_to_compact(sources)[:, None])
        or np.any(np.diff(np.sort(compact, axis=1), axis=1) == 0)
    ):
        raise Round0049Error(
            "native representative candidate cleanup is invalid"
        )
    self_seen = int(
        np.count_nonzero(
            np.any(candidates == sources[:, None], axis=1)
        )
    )
    return compact.astype(np.int32, copy=False), self_seen


def _exact_rerank_shortlist(
    *,
    queries: np.ndarray,
    shortlist: np.ndarray,
    encoded: np.ndarray,
    scales: np.ndarray,
    k: int = K,
    batch_rows: int = RERANK_BATCH_ROWS,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rerank compact candidate IDs by exact dequantized cosine.

    Each row's fp16 scale is a positive scalar, so it cancels exactly during
    cosine normalization.  We still validate those registered scales and
    score the underlying int8 vector in float32.  Stable sorting makes equal
    scores deterministic in the IVF-PQ shortlist order.
    """
    query = np.asarray(queries, dtype=np.float32)
    candidates = np.asarray(shortlist, dtype=np.int64)
    if (
        query.ndim != 2
        or candidates.ndim != 2
        or len(query) != len(candidates)
        or query.shape[1] != DIMENSION
        or candidates.shape[1] < k
        or k <= 0
        or batch_rows <= 0
        or np.any(candidates < 0)
        or np.any(
            np.diff(np.sort(candidates, axis=1), axis=1) == 0
        )
    ):
        raise Round0049Error("exact-rerank inputs have invalid geometry")
    query_norms = np.linalg.norm(query, axis=1)
    if (
        not np.isfinite(query).all()
        or not np.isfinite(query_norms).all()
        or np.any(np.abs(query_norms - 1.0) > 2e-4)
    ):
        raise Round0049Error("exact-rerank queries are not unit vectors")

    output = np.empty((len(query), k), dtype="<i4")
    started = time.monotonic()
    for start in range(0, len(query), batch_rows):
        stop = min(start + batch_rows, len(query))
        ids = candidates[start:stop]
        candidate_scales = np.asarray(scales[ids], dtype=np.float32)
        vectors = np.asarray(encoded[ids], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=2)
        if (
            not np.isfinite(candidate_scales).all()
            or np.any(candidate_scales <= 0)
            or not np.isfinite(vectors).all()
            or not np.isfinite(norms).all()
            or np.any(norms <= 0)
        ):
            raise Round0049Error(
                "exact-rerank candidate vectors are invalid"
            )
        scores = np.einsum(
            "bd,bkd->bk",
            query[start:stop],
            vectors,
            optimize=True,
        )
        scores /= norms
        order = np.argsort(
            -scores,
            axis=1,
            kind="stable",
        )[:, :k]
        output[start:stop] = np.take_along_axis(
            ids,
            order,
            axis=1,
        )
    if (
        np.any(output < 0)
        or np.any(np.diff(np.sort(output, axis=1), axis=1) == 0)
    ):
        raise Round0049Error("exact-rerank output is malformed")
    return output, {
        "wall_seconds": time.monotonic() - started,
        "shortlist_width": int(candidates.shape[1]),
        "selected_neighbors": int(k),
        "batch_rows": int(batch_rows),
        "score_dtype": "float32",
        "vector_source": "int8-plus-fp16-scale;scale-cancels-in-cosine",
        "tie_policy": "stable-ivfpq-shortlist-order",
    }


def _warm_page_cache(path: str) -> dict[str, Any]:
    """Synchronously populate the page cache before random rerank gathers."""
    started = time.monotonic()
    buffer = bytearray(64 * 1024 * 1024)
    observed = 0
    with open(path, "rb", buffering=0) as handle:
        while True:
            count = handle.readinto(buffer)
            if not count:
                break
            observed += count
    signature = expected_input_signature(path)
    if observed != signature["bytes"]:
        raise Round0049Error("short read while warming rerank substrate")
    return {
        "bytes": observed,
        "wall_seconds": time.monotonic() - started,
    }


def _shard_paths(root: str, shard: int) -> tuple[str, str]:
    return (
        os.path.join(root, f"targets-{shard:04d}.npy"),
        os.path.join(root, f"receipt-{shard:04d}.json"),
    )


def _validate_shard(
    *,
    target_path: str,
    receipt_path: str,
    start: int,
    stop: int,
    nprobe: int,
    round_id: str = ROUND_ID,
) -> dict[str, Any] | None:
    if not os.path.exists(target_path) and not os.path.exists(receipt_path):
        return None
    if not os.path.isfile(target_path) or not os.path.isfile(receipt_path):
        raise Round0049Error("partial graph shard pair exists")
    with open(receipt_path, encoding="utf-8") as handle:
        receipt = json.load(handle)
    body = {
        key: value for key, value in receipt.items()
        if key != "identity_sha256"
    }
    signature = expected_input_signature(target_path)
    if (
        receipt.get("schema") != "round0049-exact-rerank-graph-shard-v2"
        or receipt.get("identity_sha256")
        != sha256_bytes(canonical_json(body))
        or receipt.get("round_id") != round_id
        or receipt.get("start") != start
        or receipt.get("stop") != stop
        or receipt.get("nprobe") != nprobe
        or receipt.get("targets") != signature
    ):
        raise Round0049Error("completed graph shard identity changed")
    value = np.load(target_path, mmap_mode="r", allow_pickle=False)
    if value.shape != (stop - start, K) or value.dtype != np.dtype("<i4"):
        raise Round0049Error("completed graph shard geometry changed")
    return receipt


def _write_shard(
    *,
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
    round_id: str = ROUND_ID,
) -> dict[str, Any]:
    target_path, receipt_path = _shard_paths(shard_root, shard)
    previous = _validate_shard(
        target_path=target_path,
        receipt_path=receipt_path,
        start=start,
        stop=stop,
        nprobe=nprobe,
        round_id=round_id,
    )
    if previous is not None:
        return {**previous, "resumed": True}
    started = time.monotonic()
    targets = np.full((stop - start, K), -1, dtype="<i4")
    compact_rows = np.arange(start, stop, dtype=np.int64)
    retained = compact_rows[
        ~_membership(excluded, compact_rows)
    ]
    self_seen = 0
    search_seconds = 0.0
    rerank_seconds = 0.0
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
            raise Round0049Error("balanced-60M query block is nonfinite")
        query /= norms
        global_rows = compact_to_global(batch_rows)
        search_started = time.monotonic()
        _distances, raw = index.search(
            np.ascontiguousarray(query),
            INDEX_SEARCH_WIDTH,
            params=parameters,
        )
        search_seconds += time.monotonic() - search_started
        shortlist, seen = _clean_search(
            raw,
            global_sources=global_rows,
            candidate_count=SEARCH_WIDTH,
        )
        selected, rerank = _exact_rerank_shortlist(
            queries=query,
            shortlist=shortlist,
            encoded=encoded,
            scales=scales,
        )
        rerank_seconds += float(rerank["wall_seconds"])
        self_seen += seen
        targets[batch_rows - start] = selected
    if (
        np.any(targets[~_membership(
            excluded,
            compact_rows,
        )] < 0)
        or np.any(
            targets[_membership(excluded, compact_rows)] != -1
        )
    ):
        raise Round0049Error("graph shard retained/excluded rows disagree")
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
        "search_width": SEARCH_WIDTH,
        "index_search_width": INDEX_SEARCH_WIDTH,
        "selected_neighbors": K,
        "exact_rerank": True,
        "self_returned": self_seen,
        "search_seconds": search_seconds,
        "rerank_seconds": rerank_seconds,
        "wall_seconds": time.monotonic() - started,
        "targets": expected_input_signature(target_path),
    }
    receipt = _seal(body)
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    return {**receipt, "resumed": False}


def _sample_retained_rows(
    excluded: np.ndarray,
    *,
    count: int = QUALITY_SAMPLE_ROWS,
    seed: int = QUALITY_SEED,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    selected: list[np.ndarray] = []
    have = 0
    while have < count:
        proposed = rng.randint(
            0,
            ROW_COUNT,
            size=max(2 * (count - have), 1_024),
            dtype=np.int64,
        )
        proposed = proposed[
            ~_membership(excluded, proposed)
        ]
        selected.append(proposed)
        have += len(proposed)
    rows = np.unique(np.concatenate(selected))
    while len(rows) < count:
        proposed = rng.randint(
            0,
            ROW_COUNT,
            size=count,
            dtype=np.int64,
        )
        proposed = proposed[
            ~_membership(excluded, proposed)
        ]
        rows = np.unique(np.concatenate((rows, proposed)))
    return np.sort(rows[:count]).astype(np.int64, copy=False)


def _exact_representative_truth(
    *,
    encoded: np.ndarray,
    scales: np.ndarray,
    excluded: np.ndarray,
    sample: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import torch
    import torch.nn.functional as functional

    if not torch.cuda.is_available():
        raise Round0049Error(
            "R0049 exact balanced-60M quality sample requires CUDA"
        )
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()
    queries = (
        np.asarray(encoded[sample], dtype=np.float32)
        * np.asarray(scales[sample], dtype=np.float32)[:, None]
    )
    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    if (
        not np.isfinite(norms).all()
        or np.any(norms <= 0)
    ):
        raise Round0049Error("candidate-quality queries are invalid")
    queries /= norms
    query = functional.normalize(
        torch.from_numpy(np.ascontiguousarray(queries)).to(device),
        dim=1,
    )
    best_values = torch.full(
        (len(sample), K + 1),
        -torch.inf,
        dtype=torch.float32,
        device=device,
    )
    best_ids = torch.full(
        (len(sample), K + 1),
        -1,
        dtype=torch.int64,
        device=device,
    )
    blocks = 0
    try:
        with torch.inference_mode():
            for start in range(0, ROW_COUNT, EXACT_BLOCK_ROWS):
                stop = min(start + EXACT_BLOCK_ROWS, ROW_COUNT)
                candidate_ids = np.arange(start, stop, dtype=np.int64)
                candidate_ids = candidate_ids[
                    ~_membership(excluded, candidate_ids)
                ]
                candidates = (
                    np.asarray(encoded[candidate_ids], dtype=np.float32)
                    * np.asarray(
                        scales[candidate_ids],
                        dtype=np.float32,
                    )[:, None]
                )
                candidate = functional.normalize(
                    torch.from_numpy(
                        np.ascontiguousarray(candidates)
                    ).to(device),
                    dim=1,
                )
                similarity = query @ candidate.T
                positions = np.searchsorted(candidate_ids, sample)
                present = positions < len(candidate_ids)
                present_indices = np.flatnonzero(present)
                present[present_indices] = (
                    candidate_ids[positions[present_indices]]
                    == sample[present_indices]
                )
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
                    min(K + 1, len(candidate_ids)),
                    dim=1,
                    largest=True,
                    sorted=True,
                )
                candidate_id_tensor = torch.from_numpy(
                    candidate_ids
                ).to(device)
                local_ids = candidate_id_tensor[local_positions]
                merged_values = torch.cat(
                    (best_values, local_values),
                    dim=1,
                )
                merged_ids = torch.cat((best_ids, local_ids), dim=1)
                best_values, order = torch.topk(
                    merged_values,
                    K + 1,
                    dim=1,
                    largest=True,
                    sorted=True,
                )
                best_ids = torch.gather(merged_ids, 1, order)
                blocks += 1
                del (
                    candidates,
                    candidate,
                    similarity,
                    local_values,
                    local_positions,
                    candidate_id_tensor,
                    local_ids,
                    merged_values,
                    merged_ids,
                    order,
                )
        values = best_values.cpu().numpy()
        neighbors = best_ids[:, :K].cpu().numpy().astype(
            np.int64,
            copy=False,
        )
        ties = np.abs(values[:, K - 1] - values[:, K]) <= 1e-7
        if (
            np.any(neighbors < 0)
            or np.any(neighbors == sample[:, None])
            or np.any(
                np.diff(np.sort(neighbors, axis=1), axis=1) == 0
            )
        ):
            raise Round0049Error(
                "exact balanced-60M truth is malformed"
            )
        performance = {
            "wall_seconds": time.monotonic() - started,
            "candidate_blocks": blocks,
            "block_rows": EXACT_BLOCK_ROWS,
            "matmul_dtype": "float32",
            "tf32_allowed": False,
            "peak_allocated_gib": (
                torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            ),
            "peak_reserved_gib": (
                torch.cuda.max_memory_reserved(device) / (1024 ** 3)
            ),
        }
        return neighbors, ties, performance
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        del query, best_values, best_ids
        torch.cuda.empty_cache()


def run_validate_candidate_quality(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    import faiss

    nprobe = int(job["nprobe"])
    output = create_fresh_directory(
        str(job["outputs"][0]),
        label="Round 0049 candidate-quality output",
    )
    substrate = validate_substrate_manifest(
        str(job["substrate_manifest"]),
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
    sample = _sample_retained_rows(excluded)
    exact, ties, exact_performance = _exact_representative_truth(
        encoded=encoded,
        scales=scales,
        excluded=excluded,
        sample=sample,
    )
    selector, selector_keepalive, excluded_global = _eligible_selector(
        excluded
    )
    index_signature = expected_input_signature(INDEX_PATH)
    if index_signature["sha256"] != INDEX_SHA256:
        raise Round0049Error("registered 150M IVF-PQ index bytes changed")
    index = faiss.read_index(
        INDEX_PATH,
        faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY,
    )
    parameters = faiss.SearchParametersIVF()
    parameters.nprobe = nprobe
    parameters.sel = selector
    queries = (
        np.asarray(encoded[sample], dtype=np.float32)
        * np.asarray(scales[sample], dtype=np.float32)[:, None]
    )
    query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    if (
        not np.isfinite(query_norms).all()
        or np.any(query_norms <= 0)
    ):
        raise Round0049Error("IVF-PQ quality queries are invalid")
    queries /= query_norms
    search_started = time.monotonic()
    _distances, raw = index.search(
        np.ascontiguousarray(queries),
        INDEX_SEARCH_WIDTH,
        params=parameters,
    )
    search_seconds = time.monotonic() - search_started
    shortlist, self_seen = _clean_search(
        raw,
        global_sources=compact_to_global(sample),
        candidate_count=SEARCH_WIDTH,
    )
    selected, rerank_performance = _exact_rerank_shortlist(
        queries=queries,
        shortlist=shortlist,
        encoded=encoded,
        scales=scales,
    )
    overlap = (
        selected[:, :, None] == exact[:, None, :]
    ).any(axis=2).sum(axis=1) / K
    unambiguous = ~ties
    if not np.any(unambiguous):
        raise Round0049Error(
            "exact sample has no unambiguous boundary rows"
        )
    mean = float(overlap.mean())
    p10 = float(np.percentile(overlap, 10))
    clear_mean = float(overlap[unambiguous].mean())
    clear_p10 = float(np.percentile(overlap[unambiguous], 10))
    checks = {
        "sample_count_is_registered": len(sample) == QUALITY_SAMPLE_ROWS,
        "unambiguous_fraction_at_least_0_90": (
            float(unambiguous.mean()) >= 0.90
        ),
        "mean_recall_at_15_unambiguous_at_least_0_90": (
            clear_mean >= MEAN_RECALL_FLOOR
        ),
        "all_shortlist_candidates_are_representatives": (
            not np.any(_membership(excluded, shortlist))
        ),
        "all_selected_candidates_are_representatives": (
            not np.any(_membership(excluded, selected))
        ),
        "no_training_performed": True,
    }
    passed = all(value is True for value in checks.values())
    body = {
        "schema": QUALITY_RECEIPT_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "validity_passed": passed,
        "failed_checks": sorted(
            key for key, value in checks.items()
            if value is not True
        ),
        "training_performed": False,
        "optimizer_updates": 0,
        "substrate": substrate["signature"],
        "eligibility": outputs["eligibility"],
        "index": index_signature,
        "sample": {
            "seed": QUALITY_SEED,
            "rows": len(sample),
            "row_sha256": sha256_bytes(sample.tobytes()),
            "boundary_ties": int(ties.sum()),
            "unambiguous_fraction": float(unambiguous.mean()),
        },
        "candidate_generator": {
            "index_type": "IndexIVFPQ",
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
            "native_representative_selector": True,
        },
        "recall": {
            "mean_recall_at_15": mean,
            "p10_recall_at_15": p10,
            "mean_recall_at_15_unambiguous": clear_mean,
            "p10_recall_at_15_unambiguous": clear_p10,
            "floor": MEAN_RECALL_FLOOR,
        },
        "self_returned_count": self_seen,
        "checks": checks,
        "performance": {
            "exact_truth": exact_performance,
            "ivfpq_search_seconds": search_seconds,
            "exact_rerank": rerank_performance,
            "peak_rss_gib": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024 ** 2)
            ),
        },
    }
    receipt = _seal(body)
    path = os.path.join(
        output,
        "balanced-60m-candidate-quality-v1.json",
    )
    atomic_write_new_json(path, receipt, immutable=True)
    del selector_keepalive, excluded_global
    if not passed:
        raise Round0049Error(
            "balanced-60M candidate-quality floor failed; receipt preserved"
        )
    return {
        **receipt,
        "receipt": expected_input_signature(path),
    }


def _fresh_raw_file(path: str, *, bytes_count: int) -> tuple[int, str]:
    path = refuse_existing(path, label="Round 0049 raw graph output")
    descriptor = os.open(
        path,
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    os.ftruncate(descriptor, bytes_count)
    return descriptor, path


def _assemble_graph(
    *,
    output: str,
    shard_root: str,
    excluded: np.ndarray,
    nprobe: int,
    round_id: str = ROUND_ID,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target_path = os.path.join(output, "canonical-targets.i32")
    degree_path = os.path.join(output, "valid-degrees.u8")
    target_fd, _ = _fresh_raw_file(
        target_path,
        bytes_count=ROW_COUNT * K * 4,
    )
    degree_fd = -1
    try:
        offset = 0
        for shard, start in enumerate(range(0, ROW_COUNT, SHARD_ROWS)):
            stop = min(start + SHARD_ROWS, ROW_COUNT)
            shard_path, receipt_path = _shard_paths(
                shard_root,
                shard,
            )
            if _validate_shard(
                target_path=shard_path,
                receipt_path=receipt_path,
                start=start,
                stop=stop,
                nprobe=nprobe,
                round_id=round_id,
            ) is None:
                raise Round0049Error("graph assembly found a missing shard")
            values = np.load(
                shard_path,
                mmap_mode="r",
                allow_pickle=False,
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
                    raise Round0049Error("short graph target assembly write")
                written += count
            offset += len(payload)
        if offset != ROW_COUNT * K * 4:
            raise Round0049Error("assembled target bytes do not close")
        os.fsync(target_fd)
        os.fchmod(target_fd, 0o444)
        os.close(target_fd)
        target_fd = -1

        degree_fd, _ = _fresh_raw_file(
            degree_path,
            bytes_count=ROW_COUNT,
        )
        degrees = np.memmap(
            degree_path,
            dtype="u1",
            mode="r+",
            shape=(ROW_COUNT,),
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


def _initialize_graph_output(
    output: str,
    *,
    contract: Mapping[str, Any],
) -> str:
    contract_path = os.path.join(output, "graph-build-contract.json")
    if not os.path.exists(output):
        create_fresh_directory(
            output,
            label="Round 0049 balanced-60M graph",
        )
        atomic_write_new_json(
            contract_path,
            dict(contract),
            immutable=True,
        )
    else:
        with open(contract_path, encoding="utf-8") as handle:
            observed = json.load(handle)
        if observed != contract:
            raise Round0049Error(
                "resumed graph output has a different build contract"
            )
    return ensure_data_directory(os.path.join(output, "shards"))


def run_build_graph(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    import faiss

    round_id = str(active.get("manifest", {}).get("round_id"))
    if round_id not in {ROUND_ID, "0050"}:
        raise Round0049Error(
            "balanced-60M graph builder received an unauthorized round"
        )
    output = str(job["outputs"][0])
    nprobe = int(job["nprobe"])
    threads = int(job.get("cpu_threads", DEFAULT_THREADS))
    if (
        nprobe <= 0
        or nprobe > 8192
        or threads <= 0
        or threads > (os.cpu_count() or 1)
    ):
        raise Round0049Error("registered search resource geometry is invalid")
    substrate = validate_substrate_manifest(
        str(job["substrate_manifest"]),
        expected_sha256=(
            str(job["substrate_manifest_sha256"])
            if job.get("substrate_manifest_sha256")
            else None
        ),
    )
    manifest = substrate["manifest"]
    outputs = manifest["outputs"]
    eligibility = load_int8_eligibility(
        outputs["eligibility"]["canonical_path"],
        expected_sha256=outputs["eligibility"]["sha256"],
        row_count=ROW_COUNT,
    )
    excluded = np.asarray(
        eligibility["excluded_rows"],
        dtype=np.int64,
    )
    index_signature = expected_input_signature(INDEX_PATH)
    if index_signature["sha256"] != INDEX_SHA256:
        raise Round0049Error("registered 150M IVF-PQ index bytes changed")
    r0047_quality_receipt = expected_input_signature(
        str(job["candidate_quality_receipt"])
    )
    if (
        r0047_quality_receipt["sha256"]
        != str(job["candidate_quality_receipt_sha256"])
    ):
        raise Round0049Error("R0047 candidate-quality receipt changed")
    quality_path = str(job["quality_validation_receipt"])
    quality_signature = expected_input_signature(quality_path)
    with open(quality_path, encoding="utf-8") as handle:
        quality = json.load(handle)
    quality_mean = _quality_authority_mean_recall(
        quality,
        nprobe=nprobe,
    )
    contract = {
        "schema": "round0049-balanced-60m-graph-build-contract-v1",
        "release_sha": active["manifest"]["release_sha"],
        "substrate": substrate["signature"],
        "eligibility": outputs["eligibility"],
        "index": index_signature,
        "r0047_candidate_quality_receipt": r0047_quality_receipt,
        "quality_validation_receipt": quality_signature,
        "quality_mean_recall_at_15_unambiguous": quality_mean,
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
            "balanced intervals AND NOT within-subset zero/duplicate copies"
        ),
    }
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
    selector, selector_keepalive, excluded_global = _eligible_selector(
        excluded
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
        or int(index.ntotal) != SOURCE_ROWS
        or int(index.d) != DIMENSION
        or int(index.nlist) != 8192
        or int(index.pq.M) != 48
        or int(index.pq.nbits) != 8
    ):
        raise Round0049Error("registered 150M IVF-PQ geometry changed")
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
            round_id=round_id,
        )
        resumed += int(receipt["resumed"])
        shard_receipts.append(receipt)
        print(
            f"R0049 graph shard {shard + 1}/"
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
        round_id=round_id,
    )
    retained = ROW_COUNT - len(excluded)
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
        "round_id": round_id,
        "row_count": ROW_COUNT,
        "input_k": K,
        "source_semantics": (
            "compact balanced-60M row ID; excluded rows have degree zero"
        ),
        "destination_policy": (
            "native representative-only IVF-PQ search in the balanced "
            "candidate universe; remove self"
        ),
        "sampling_semantics": (
            "consumer must register source-uniform or edge-uniform separately"
        ),
        "negative_policy": "uniform-eligibility-retained-rows-nonself",
        "weight_semantics": (
            "unweighted directed k15; no materialized weight array"
        ),
        "inputs": {
            "eligibility": outputs["eligibility"],
            "substrate": substrate["signature"],
            "index": index_signature,
            "r0047_candidate_quality_receipt": r0047_quality_receipt,
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
            "eligibility_excluded_source_count": len(excluded),
            "eligibility_retained_row_count": retained,
            "retained_positive_source_count": retained,
            "zero_degree_retained_source_count": 0,
            "zero_degree_retained_source_fraction": 0.0,
            "valid_canonical_edge_count": retained * K,
            "input_edge_count": retained * K,
            "degree_histogram": {
                "0": len(excluded),
                str(K): retained,
            },
            "native_candidate_excluded_global_rows": len(excluded),
            "self_returned_count": self_seen,
            "self_returned_fraction": self_seen / retained,
        },
        "candidate_generator": {
            "index_type": "IndexIVFPQ",
            "source_index_rows": SOURCE_ROWS,
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
            "candidate_universe": (
                "first 20M rows per corpus, exact zero/copy exclusions applied "
                "inside IVF scanning"
            ),
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
    graph_manifest = _seal(graph_body)
    graph_path = os.path.join(output, "canonical-graph-v1.json")
    atomic_write_new_json(
        graph_path,
        graph_manifest,
        immutable=True,
    )
    receipt_body = {
        "schema": GRAPH_RECEIPT_SCHEMA,
        "round_id": round_id,
        "release_sha": active["manifest"]["release_sha"],
        "training_performed": False,
        "optimizer_updates": 0,
        "graph": expected_input_signature(graph_path),
        "substrate": substrate["signature"],
        "r0047_candidate_quality_receipt": r0047_quality_receipt,
        "quality_validation_receipt": quality_signature,
        "search": graph_manifest["candidate_generator"],
        "summary": graph_manifest["summary"],
        "performance": {
            **graph_manifest["timing"],
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
        raise Round0049Error("R0049 handler received another queue")
    selected = job if job is not None else active.get("job") or {}
    action = selected.get("action")
    if action == "build_substrate":
        return run_build_substrate(active, selected)
    if action == "validate_candidate_quality":
        return run_validate_candidate_quality(active, selected)
    if action == "build_graph":
        return run_build_graph(active, selected)
    raise Round0049Error(f"unknown R0049 action: {action!r}")
