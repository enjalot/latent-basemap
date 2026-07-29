"""Build and assemble the retained diverse-Jina fuzzy graph."""
from __future__ import annotations

import contextlib
import gc
import json
import os
import resource
import shutil
import tempfile
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    ordered_array_sha256,
    sha256_bytes,
)
from basemap.output_safety import (
    atomic_save_new_npy,
    atomic_save_new_npz,
    atomic_write_new_json,
    ensure_data_directory,
)
from basemap.round0105_search import (
    DIMENSION,
    ELIGIBILITY_PATH,
    GROUPS,
    K,
    RETAINED_ROWS,
    ROW_COUNT,
)
from basemap.round0106_graph import (
    GRAPH_SCHEMA,
    LOCAL_CONNECTIVITY,
    MINIMUM_SHARD_SOURCES_PER_SECOND,
    N_NEIGHBORS,
    PAIR_BUCKETS,
    PART_SCHEMA,
    PARTS,
    PERFORMANCE_SUBFLOOR_PATIENCE,
    PERFORMANCE_WARMUP_SHARDS,
    RERANK_BATCH_ROWS,
    ROUND_ID,
    SEARCH_BATCH_ROWS,
    SHARD_ROWS,
    SHARD_SCHEMA,
    Round0106Error,
    compact_to_global,
    global_to_compact,
    membership,
    part_spec,
    seal,
    update_performance_streak,
    validate_search_artifacts,
)
from experiments.build_weighted_graph import (
    _REC,
    _pair_bucket,
    fuzzy_directed_from_knn,
    phase_c_join,
    symmetrize_bucket,
)
from experiments.round0105_nodes import (
    _clean_search,
    _gpu_options,
    _normalized_rows,
    _require_geometry,
    _substrate_arrays,
)


FINAL_SCAN_ROWS = 25_000_000
ASSEMBLY_WORKERS = 8


def _peak_rss_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024 ** 2)


def _load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _validate_signature(signature: Mapping[str, Any], *, label: str) -> None:
    observed = expected_input_signature(str(signature["canonical_path"]))
    expected = {
        key: signature[key]
        for key in ("kind", "canonical_path", "bytes", "sha256")
    }
    if observed != expected:
        raise Round0106Error(f"{label} bytes changed")


def _load_search(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.memmap, np.memmap, dict[str, Any]]:
    substrate, excluded, encoded, scales = _substrate_arrays()
    search = validate_search_artifacts(
        index_path=str(job["index"]),
        index_sha256=str(job["index_sha256"]),
        index_receipt_path=str(job["index_receipt"]),
        index_receipt_sha256=str(job["index_receipt_sha256"]),
        qualification_path=str(job["qualification"]),
        qualification_sha256=str(job["qualification_sha256"]),
        decision_path=str(job["decision"]),
        decision_sha256=str(job["decision_sha256"]),
        substrate_signature=substrate["signature"],
    )
    if active["manifest"]["release_sha"] != str(job["release_sha"]):
        raise Round0106Error("R0106 release binding changed")
    return substrate, excluded, encoded, scales, search


def _gpu_exact_rerank(
    *,
    queries: np.ndarray,
    shortlist: np.ndarray,
    encoded: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Exact native-vector cosine rerank on GPU, returning distances too."""
    import torch

    query = np.asarray(queries, dtype=np.float32)
    candidates = np.asarray(shortlist, dtype=np.int64)
    if (
        query.ndim != 2
        or query.shape != (len(candidates), DIMENSION)
        or candidates.ndim != 2
        or candidates.shape[1] < K
    ):
        raise Round0106Error("R0106 exact-rerank inputs have invalid geometry")
    device = torch.device("cuda")
    selected = np.empty((len(query), K), dtype=np.int64)
    distances = np.empty((len(query), K), dtype=np.float32)
    started = time.monotonic()
    transfer_seconds = 0.0
    compute_seconds = 0.0
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats(device)
    try:
        with torch.inference_mode():
            for start in range(0, len(query), RERANK_BATCH_ROWS):
                stop = min(start + RERANK_BATCH_ROWS, len(query))
                ids = candidates[start:stop]
                host_values = np.asarray(encoded[ids], dtype=np.int8)
                host_scales = np.asarray(scales[ids], dtype=np.float16)
                if (
                    host_values.shape
                    != (stop - start, candidates.shape[1], DIMENSION)
                    or not np.isfinite(host_scales).all()
                    or np.any(host_scales <= 0)
                ):
                    raise Round0106Error("R0106 native candidates are malformed")
                transfer_started = time.monotonic()
                candidate_tensor = torch.from_numpy(host_values).to(device)
                scale_tensor = torch.from_numpy(host_scales).to(
                    device, dtype=torch.float32
                )
                query_tensor = torch.from_numpy(query[start:stop]).to(device)
                torch.cuda.synchronize(device)
                transfer_seconds += time.monotonic() - transfer_started
                compute_started = time.monotonic()
                candidate_tensor = (
                    candidate_tensor.to(torch.float32)
                    * scale_tensor[:, :, None]
                )
                norms = torch.linalg.vector_norm(
                    candidate_tensor, dim=2, keepdim=True
                )
                if not bool(torch.all(torch.isfinite(norms))) or bool(
                    torch.any(norms <= 0)
                ):
                    raise Round0106Error(
                        "R0106 exact-rerank candidate norms are malformed"
                    )
                candidate_tensor /= norms
                scores = torch.einsum(
                    "bd,bkd->bk", query_tensor, candidate_tensor
                )
                if not bool(torch.all(torch.isfinite(scores))):
                    raise Round0106Error("R0106 exact cosine scores are nonfinite")
                order = torch.argsort(
                    scores, dim=1, descending=True, stable=True
                )[:, :K]
                chosen_scores = torch.gather(scores, 1, order)
                chosen_ids = torch.gather(
                    torch.from_numpy(ids).to(device), 1, order
                )
                selected[start:stop] = chosen_ids.cpu().numpy()
                distances[start:stop] = (
                    1.0 - chosen_scores
                ).clamp_(min=0.0, max=2.0).cpu().numpy()
                torch.cuda.synchronize(device)
                compute_seconds += time.monotonic() - compute_started
                del (
                    candidate_tensor,
                    scale_tensor,
                    query_tensor,
                    norms,
                    scores,
                    order,
                    chosen_scores,
                    chosen_ids,
                )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    if (
        np.any(selected < 0)
        or np.any(np.diff(np.sort(selected, axis=1), axis=1) == 0)
        or not np.isfinite(distances).all()
        or np.any(distances < 0)
        or np.any(distances > 2)
    ):
        raise Round0106Error("R0106 exact-rerank output is malformed")
    return selected, distances, {
        "wall_seconds": time.monotonic() - started,
        "host_to_device_seconds": transfer_seconds,
        "gpu_compute_seconds": compute_seconds,
        "shortlist_width": int(candidates.shape[1]),
        "selected_neighbors": K,
        "batch_rows": RERANK_BATCH_ROWS,
        "score_dtype": "float32",
        "stored_vector_dtype": "int8-plus-fp16-scale",
        "tf32": False,
        "stable_tie_policy": "IVFPQ-shortlist-order",
        "peak_vram_gib": float(torch.cuda.max_memory_allocated(device)) / (1024 ** 3),
    }


def _validate_directed_memberships(
    *,
    rows: np.ndarray,
    all_targets: np.ndarray,
    sources: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
) -> dict[str, int]:
    """Validate positive memberships as a subset of the fixed kNN topology."""
    source_rows = np.asarray(rows, dtype=np.int64)
    knn_targets = np.asarray(all_targets, dtype=np.int32)
    edge_sources = np.asarray(sources)
    edge_targets = np.asarray(targets)
    edge_weights = np.asarray(weights)
    knn_edges = len(source_rows) * K
    if (
        source_rows.ndim != 1
        or knn_targets.shape != (len(source_rows), K)
        or edge_sources.ndim != 1
        or edge_targets.ndim != 1
        or edge_weights.ndim != 1
        or edge_sources.shape != edge_targets.shape
        or edge_sources.shape != edge_weights.shape
        or len(edge_sources) > knn_edges
        or edge_sources.dtype != np.int32
        or edge_targets.dtype != np.int32
        or edge_weights.dtype != np.float32
        or not np.isfinite(edge_weights).all()
        or np.any(edge_weights <= 0)
        or np.any(edge_weights > 1)
    ):
        raise Round0106Error("R0106 directed fuzzy membership is malformed")
    if len(source_rows) == 0:
        raise Round0106Error("R0106 fuzzy membership source set is empty")
    compact_start = int(source_rows[0])
    if (
        not np.array_equal(
            source_rows,
            np.arange(
                compact_start,
                compact_start + len(source_rows),
                dtype=np.int64,
            ),
        )
    ):
        raise Round0106Error("R0106 fuzzy membership source order changed")
    if (
        np.any(edge_sources < compact_start)
        or np.any(edge_sources >= compact_start + len(source_rows))
        or np.any(edge_sources[1:] < edge_sources[:-1])
    ):
        raise Round0106Error("R0106 directed fuzzy membership is malformed")
    source_offsets = edge_sources.astype(np.int64) - compact_start
    target_matches = (
        edge_targets[:, None] == knn_targets[source_offsets]
    ).sum(axis=1)
    counts = np.bincount(source_offsets, minlength=len(source_rows))
    keys = (
        edge_sources.astype(np.uint64) * np.uint64(RETAINED_ROWS)
        + edge_targets.astype(np.uint64)
    )
    if (
        np.any(counts < 1)
        or np.any(counts > K)
        or not np.all(target_matches == 1)
        or len(np.unique(keys)) != len(keys)
    ):
        raise Round0106Error("R0106 directed fuzzy membership is malformed")
    return {
        "knn_edges": knn_edges,
        "directed_edges": len(edge_weights),
        "zero_memberships_eliminated": knn_edges - len(edge_weights),
        "sources_with_eliminated_memberships": int(np.count_nonzero(counts < K)),
        "minimum_memberships_per_source": int(counts.min()),
    }


def _shard_paths(output: str, shard: int) -> tuple[str, str]:
    artifact = os.path.join(output, f"shard-{shard:05d}.npz")
    return artifact, artifact + ".receipt.json"


def _validate_shard(
    artifact: str,
    receipt_path: str,
    *,
    part: str,
    shard: int,
    compact_start: int,
    compact_stop: int,
    contract_sha256: str,
) -> dict[str, Any] | None:
    if os.path.exists(artifact) != os.path.exists(receipt_path):
        raise Round0106Error(f"incomplete R0106 shard pair: {artifact}")
    if not os.path.exists(artifact):
        return None
    receipt = _load_json(receipt_path)
    body = {
        key: value
        for key, value in receipt.items()
        if key != "identity_sha256"
    }
    expected = {
        "schema": SHARD_SCHEMA,
        "round_id": ROUND_ID,
        "part": part,
        "shard": shard,
        "compact_start": compact_start,
        "compact_stop": compact_stop,
        "contract_sha256": contract_sha256,
    }
    if (
        any(receipt.get(key) != value for key, value in expected.items())
        or receipt.get("identity_sha256") != sha256_bytes(canonical_json(body))
    ):
        raise Round0106Error(f"R0106 shard receipt changed: {receipt_path}")
    _validate_signature(receipt["artifact"], label=f"R0106 {part} shard {shard}")
    with np.load(artifact, allow_pickle=False) as archive:
        sources = archive["sources"]
        targets = archive["targets"]
        weights = archive["weights"]
        retained_sources = compact_stop - compact_start
        knn_edges = retained_sources * K
        directed_edges = int(receipt.get("directed_edges", -1))
        zero_memberships = int(
            receipt.get("zero_memberships_eliminated", -1)
        )
        if (
            int(receipt.get("retained_sources", -1)) != retained_sources
            or int(receipt.get("knn_edges", -1)) != knn_edges
            or directed_edges < retained_sources
            or directed_edges > knn_edges
            or zero_memberships != knn_edges - directed_edges
            or sources.shape != (directed_edges,)
            or targets.shape != (directed_edges,)
            or weights.shape != (directed_edges,)
            or sources.dtype != np.int32
            or targets.dtype != np.int32
            or weights.dtype != np.float32
            or not np.isfinite(weights).all()
            or np.any(weights <= 0)
            or np.any(weights > 1)
            or np.any(sources < compact_start)
            or np.any(sources >= compact_stop)
            or np.any(targets < 0)
            or np.any(targets >= RETAINED_ROWS)
            or np.any(sources == targets)
            or np.any(sources[1:] < sources[:-1])
        ):
            raise Round0106Error(f"R0106 shard arrays changed: {artifact}")
        counts = np.bincount(
            sources.astype(np.int64) - compact_start,
            minlength=retained_sources,
        )
        keys = (
            sources.astype(np.uint64) * np.uint64(RETAINED_ROWS)
            + targets.astype(np.uint64)
        )
        if (
            np.any(counts < 1)
            or np.any(counts > K)
            or len(np.unique(keys)) != len(keys)
            or int(receipt.get("minimum_memberships_per_source", -1))
            != int(counts.min())
            or int(
                receipt.get("sources_with_eliminated_memberships", -1)
            )
            != int(np.count_nonzero(counts < K))
        ):
            raise Round0106Error(f"R0106 shard arrays changed: {artifact}")
    return receipt


def _write_shard(
    *,
    gpu: Any,
    part: str,
    shard: int,
    compact_start: int,
    compact_stop: int,
    width: int,
    excluded: np.ndarray,
    encoded: np.ndarray,
    scales: np.ndarray,
    output: str,
    contract_sha256: str,
) -> dict[str, Any]:
    artifact, receipt_path = _shard_paths(output, shard)
    prior = _validate_shard(
        artifact,
        receipt_path,
        part=part,
        shard=shard,
        compact_start=compact_start,
        compact_stop=compact_stop,
        contract_sha256=contract_sha256,
    )
    if prior is not None:
        return {**prior, "resumed": True}
    started = time.monotonic()
    rows = np.arange(compact_start, compact_stop, dtype=np.int64)
    global_rows = compact_to_global(rows, excluded)
    all_targets = np.empty((len(rows), K), dtype=np.int32)
    all_distances = np.empty((len(rows), K), dtype=np.float32)
    search_seconds = 0.0
    rerank_seconds = 0.0
    transfer_seconds = 0.0
    gpu_compute_seconds = 0.0
    self_returned = 0
    peak_vram_gib = 0.0
    for offset in range(0, len(rows), SEARCH_BATCH_ROWS):
        stop = min(offset + SEARCH_BATCH_ROWS, len(rows))
        source_global = global_rows[offset:stop]
        queries = _normalized_rows(encoded, scales, source_global)
        search_started = time.monotonic()
        _approximate, raw = gpu.search(queries, width + 1)
        search_seconds += time.monotonic() - search_started
        shortlist, seen = _clean_search(
            raw,
            sources=source_global,
            width=width,
            excluded=excluded,
        )
        self_returned += seen
        selected_global, distances, rerank = _gpu_exact_rerank(
            queries=queries,
            shortlist=shortlist,
            encoded=encoded,
            scales=scales,
        )
        all_targets[offset:stop] = global_to_compact(
            selected_global, excluded
        ).astype(np.int32)
        all_distances[offset:stop] = distances
        rerank_seconds += float(rerank["wall_seconds"])
        transfer_seconds += float(rerank["host_to_device_seconds"])
        gpu_compute_seconds += float(rerank["gpu_compute_seconds"])
        peak_vram_gib = max(peak_vram_gib, float(rerank["peak_vram_gib"]))
    if (
        np.any(all_targets == rows.astype(np.int32)[:, None])
        or np.any(np.diff(np.sort(all_targets, axis=1), axis=1) == 0)
    ):
        raise Round0106Error("R0106 selected topology is not distinct/nonself")
    knn_indices = np.empty((len(rows), N_NEIGHBORS), dtype=np.int32)
    knn_indices[:, 0] = rows.astype(np.int32)
    knn_indices[:, 1:] = all_targets
    knn_distances = np.zeros((len(rows), N_NEIGHBORS), dtype=np.float32)
    knn_distances[:, 1:] = all_distances
    (
        sources,
        targets,
        weights,
        sigmas,
        rhos,
        rho_zero,
    ) = fuzzy_directed_from_knn(
        knn_indices,
        knn_distances,
        N_NEIGHBORS,
        local_connectivity=LOCAL_CONNECTIVITY,
    )
    expected_targets = all_targets.reshape(-1)
    membership_closure = _validate_directed_memberships(
        rows=rows,
        all_targets=all_targets,
        sources=sources,
        targets=targets,
        weights=weights,
    )
    audit_offsets = np.unique(
        np.linspace(0, len(rows) - 1, min(len(rows), 32)).astype(np.int64)
    )
    audit_sources_global = global_rows[audit_offsets]
    audit_targets_compact = all_targets[audit_offsets].astype(np.int64)
    audit_targets_global = compact_to_global(
        audit_targets_compact.reshape(-1), excluded
    )
    audit_source_vectors = _normalized_rows(
        encoded, scales, audit_sources_global
    )
    audit_target_vectors = _normalized_rows(
        encoded, scales, audit_targets_global
    ).reshape(len(audit_offsets), K, DIMENSION)
    audit_distances = 1.0 - np.einsum(
        "bd,bkd->bk",
        audit_source_vectors,
        audit_target_vectors,
        optimize=True,
    )
    audit_distances = np.clip(audit_distances, 0.0, 2.0).astype(np.float32)
    distance_max_abs_error = float(
        np.max(
            np.abs(
                audit_distances.astype(np.float64)
                - all_distances[audit_offsets].astype(np.float64)
            )
        )
    )
    audit_knn_indices = np.empty(
        (len(audit_offsets), N_NEIGHBORS), dtype=np.int32
    )
    audit_knn_indices[:, 0] = rows[audit_offsets].astype(np.int32)
    audit_knn_indices[:, 1:] = audit_targets_compact.astype(np.int32)
    audit_knn_distances = np.zeros_like(
        audit_knn_indices, dtype=np.float32
    )
    audit_knn_distances[:, 1:] = all_distances[audit_offsets]
    (
        audit_fuzzy_sources,
        audit_fuzzy_targets,
        audit_fuzzy_weights,
        *_audit_fuzzy_stats,
    ) = fuzzy_directed_from_knn(
        audit_knn_indices,
        audit_knn_distances,
        N_NEIGHBORS,
        local_connectivity=LOCAL_CONNECTIVITY,
    )
    audit_source_ids = rows[audit_offsets].astype(np.int32)
    reference_mask = np.isin(sources, audit_source_ids)
    reference_sources = sources[reference_mask]
    reference_targets = targets[reference_mask]
    reference_weights = weights[reference_mask]
    fuzzy_topology_matches = (
        np.array_equal(audit_fuzzy_sources, reference_sources)
        and np.array_equal(audit_fuzzy_targets, reference_targets)
    )
    weight_max_abs_error = (
        float(
            np.max(
                np.abs(
                    audit_fuzzy_weights.astype(np.float64)
                    - reference_weights.astype(np.float64)
                )
            )
        )
        if fuzzy_topology_matches and len(reference_weights)
        else float("inf")
    )
    cpu_knn_distances = np.zeros_like(
        audit_knn_indices, dtype=np.float32
    )
    cpu_knn_distances[:, 1:] = audit_distances
    (
        cpu_fuzzy_sources,
        cpu_fuzzy_targets,
        cpu_fuzzy_weights,
        *_cpu_fuzzy_stats,
    ) = fuzzy_directed_from_knn(
        audit_knn_indices,
        cpu_knn_distances,
        N_NEIGHBORS,
        local_connectivity=LOCAL_CONNECTIVITY,
    )
    cpu_fuzzy_topology_matches = (
        np.array_equal(cpu_fuzzy_sources, reference_sources)
        and np.array_equal(cpu_fuzzy_targets, reference_targets)
    )
    cpu_fuzzy_weight_max_abs_error = (
        float(
            np.max(
                np.abs(
                    cpu_fuzzy_weights.astype(np.float64)
                    - reference_weights.astype(np.float64)
                )
            )
        )
        if cpu_fuzzy_topology_matches and len(reference_weights)
        else None
    )
    if (
        not fuzzy_topology_matches
        or distance_max_abs_error > 2e-5
        or weight_max_abs_error > 2e-5
    ):
        raise Round0106Error("R0106 independent shard audit failed")
    atomic_save_new_npz(
        artifact,
        immutable=True,
        sources=sources,
        targets=targets,
        weights=weights,
    )
    body = {
        "schema": SHARD_SCHEMA,
        "round_id": ROUND_ID,
        "part": part,
        "shard": shard,
        "compact_start": compact_start,
        "compact_stop": compact_stop,
        "global_start": int(global_rows[0]),
        "global_stop_inclusive": int(global_rows[-1]),
        "contract_sha256": contract_sha256,
        "retained_sources": len(rows),
        **membership_closure,
        "k_real": K,
        "n_neighbors_including_self": N_NEIGHBORS,
        "local_connectivity": LOCAL_CONNECTIVITY,
        "exact_rerank": True,
        "distance_dtype": "float32",
        "weight_dtype": "float32",
        "source_ids_ordered_sha256": ordered_array_sha256(rows),
        "knn_target_ids_ordered_sha256": ordered_array_sha256(
            expected_targets
        ),
        "target_ids_ordered_sha256": ordered_array_sha256(targets),
        "weights_ordered_sha256": ordered_array_sha256(weights),
        "distance_summary": {
            "minimum": float(all_distances.min()),
            "mean": float(all_distances.astype(np.float64).mean()),
            "maximum": float(all_distances.max()),
        },
        "fuzzy_summary": {
            "minimum": float(weights.min()),
            "mean": float(weights.astype(np.float64).mean()),
            "maximum": float(weights.max()),
            "rho_zero_rows": int(rho_zero),
            "sigma_mean": float(sigmas.astype(np.float64).mean()),
            "rho_mean": float(rhos.astype(np.float64).mean()),
        },
        "independent_audit": {
            "rows": len(audit_offsets),
            "compact_rows_ordered_sha256": ordered_array_sha256(
                rows[audit_offsets]
            ),
            "distance_path": (
                "CPU fp32 native-int8-plus-scale exact cosine"
            ),
            "distance_max_abs_error": distance_max_abs_error,
            "fuzzy_path": (
                "exact-row replay of reviewed UMAP kernel on sealed "
                "production fp32 distances"
            ),
            "weight_max_abs_error": weight_max_abs_error,
            "tolerance": 2e-5,
            "passed": True,
            "cpu_recomputed_fuzzy_diagnostic": {
                "role": (
                    "diagnostic-only; endpoint-distance tolerance and exact "
                    "kernel replay are gated separately"
                ),
                "topology_matches": cpu_fuzzy_topology_matches,
                "weight_max_abs_error": cpu_fuzzy_weight_max_abs_error,
            },
        },
        "performance": {
            "search_seconds": search_seconds,
            "exact_rerank_wall_seconds": rerank_seconds,
            "host_to_device_seconds": transfer_seconds,
            "gpu_rerank_compute_seconds": gpu_compute_seconds,
            "wall_seconds": time.monotonic() - started,
            "sources_per_second": len(rows) / (time.monotonic() - started),
            "self_returned": self_returned,
            "peak_vram_gib": peak_vram_gib,
            "peak_rss_gib": _peak_rss_gib(),
        },
        "artifact": expected_input_signature(artifact),
    }
    receipt = seal(body)
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    return {**receipt, "resumed": False}


def _part_contract(
    *,
    part: str,
    release_sha: str,
    search: Mapping[str, Any],
    substrate_signature: Mapping[str, Any],
) -> str:
    return sha256_bytes(canonical_json({
        "schema": PART_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": release_sha,
        "part": part,
        "part_spec": part_spec(part),
        "substrate": dict(substrate_signature),
        "search": {
            key: search[key]
            for key in ("index", "index_receipt", "qualification", "decision")
        },
        "selected": search["selected"],
        "k": K,
        "n_neighbors_including_self": N_NEIGHBORS,
        "local_connectivity": LOCAL_CONNECTIVITY,
        "shard_rows": SHARD_ROWS,
    }))


def run_build_part(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    import faiss
    import torch

    part = str(job["part"])
    spec = part_spec(part)
    output = ensure_data_directory(
        str(job["outputs"][0]), label=f"R0106 {part} graph part"
    )
    completed_path = os.path.join(output, "part-receipt.json")
    substrate, excluded, encoded, scales, search = _load_search(active, job)
    contract_sha256 = _part_contract(
        part=part,
        release_sha=active["manifest"]["release_sha"],
        search=search,
        substrate_signature=substrate["signature"],
    )
    if os.path.exists(completed_path):
        completed = _load_json(completed_path)
        body = {
            key: value
            for key, value in completed.items()
            if key != "identity_sha256"
        }
        if (
            completed.get("schema") != PART_SCHEMA
            or completed.get("part") != part
            or completed.get("contract_sha256") != contract_sha256
            or completed.get("identity_sha256")
            != sha256_bytes(canonical_json(body))
        ):
            raise Round0106Error(f"R0106 completed {part} receipt changed")
        return {**completed, "receipt": expected_input_signature(completed_path)}
    started = time.monotonic()
    index = faiss.read_index(str(job["index"]))
    _require_geometry(index, ntotal=RETAINED_ROWS)
    resources = faiss.StandardGpuResources()
    resources.setTempMemory(1 << 30)
    clone_started = time.monotonic()
    gpu = faiss.index_cpu_to_gpu(resources, 0, index, _gpu_options())
    clone_seconds = time.monotonic() - clone_started
    selected = search["selected"]
    nprobe = int(selected["nprobe"])
    width = int(selected["shortlist_width"])
    gpu.nprobe = nprobe
    shard_receipts = []
    completed_new_shards = 0
    performance_subfloor_streak = 0
    compact_start = spec["compact_start"]
    compact_stop = spec["compact_stop"]
    for shard, start in enumerate(
        range(compact_start, compact_stop, SHARD_ROWS)
    ):
        stop = min(start + SHARD_ROWS, compact_stop)
        receipt = _write_shard(
            gpu=gpu,
            part=part,
            shard=shard,
            compact_start=start,
            compact_stop=stop,
            width=width,
            excluded=excluded,
            encoded=encoded,
            scales=scales,
            output=output,
            contract_sha256=contract_sha256,
        )
        shard_receipts.append(receipt)
        rate = float(receipt["performance"]["sources_per_second"])
        if receipt["resumed"] is not True:
            completed_new_shards += 1
            performance_subfloor_streak = update_performance_streak(
                performance_subfloor_streak,
                completed_new_shards=completed_new_shards,
                sources_per_second=rate,
            )
        print(
            f"R0106 {part} shard {shard + 1}: "
            f"{stop - compact_start:,}/{spec['retained_rows']:,} sources "
            f"({rate:.1f} source/s)",
            flush=True,
        )
        if (
            performance_subfloor_streak
            >= PERFORMANCE_SUBFLOOR_PATIENCE
        ):
            raise Round0106Error(
                "R0106 shard throughput stayed below "
                f"{MINIMUM_SHARD_SOURCES_PER_SECOND:.1f} source/s for "
                f"{PERFORMANCE_SUBFLOOR_PATIENCE} consecutive post-warmup "
                "shards"
            )
    retained_sources = sum(
        int(receipt["retained_sources"]) for receipt in shard_receipts
    )
    knn_edges = sum(
        int(receipt["knn_edges"]) for receipt in shard_receipts
    )
    directed_edges = sum(
        int(receipt["directed_edges"]) for receipt in shard_receipts
    )
    zero_memberships_eliminated = sum(
        int(receipt["zero_memberships_eliminated"])
        for receipt in shard_receipts
    )
    if (
        retained_sources != spec["retained_rows"]
        or knn_edges != spec["retained_rows"] * K
        or directed_edges + zero_memberships_eliminated != knn_edges
        or directed_edges < retained_sources
    ):
        raise Round0106Error(f"R0106 {part} source/edge closure failed")
    receipt = seal({
        "schema": PART_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "part": part,
        "part_spec": spec,
        "contract_sha256": contract_sha256,
        "substrate": substrate["signature"],
        "search": {
            key: search[key]
            for key in ("index", "index_receipt", "qualification", "decision")
        },
        "selected_policy": selected,
        "retained_sources": retained_sources,
        "knn_edges": knn_edges,
        "directed_edges": directed_edges,
        "zero_memberships_eliminated": zero_memberships_eliminated,
        "sources_with_eliminated_memberships": sum(
            int(receipt["sources_with_eliminated_memberships"])
            for receipt in shard_receipts
        ),
        "minimum_memberships_per_source": min(
            int(receipt["minimum_memberships_per_source"])
            for receipt in shard_receipts
        ),
        "shard_rows": SHARD_ROWS,
        "shards": [
            {
                "shard": int(value["shard"]),
                "compact_start": int(value["compact_start"]),
                "compact_stop": int(value["compact_stop"]),
                "receipt": expected_input_signature(
                    _shard_paths(output, int(value["shard"]))[1]
                ),
                "artifact": value["artifact"],
            }
            for value in shard_receipts
        ],
        "pipeline": {
            "candidate_universe": "all-24,948,663-retained-global-IDs",
            "search": "R0105-selected-GpuIndexIVFPQ",
            "exact_rerank": "native-int8-plus-fp16-scale-cosine-fp32",
            "exact_rerank_device": "cuda:0",
            "tf32": False,
            "selected_neighbors": K,
            "distance": "one-minus-exact-cosine-fp32",
            "fuzzy": (
                "umap-smooth_knn_dist-n16-local_connectivity1-directed"
            ),
        },
        "performance": {
            "clone_seconds": clone_seconds,
            "search_seconds": sum(
                float(value["performance"]["search_seconds"])
                for value in shard_receipts
            ),
            "exact_rerank_wall_seconds": sum(
                float(value["performance"]["exact_rerank_wall_seconds"])
                for value in shard_receipts
            ),
            "wall_seconds": time.monotonic() - started,
            "peak_vram_gib": max(
                float(value["performance"]["peak_vram_gib"])
                for value in shard_receipts
            ),
            "peak_rss_gib": _peak_rss_gib(),
            "early_regression_guard": {
                "warmup_new_shards": PERFORMANCE_WARMUP_SHARDS,
                "minimum_sources_per_second": (
                    MINIMUM_SHARD_SOURCES_PER_SECOND
                ),
                "subfloor_patience": PERFORMANCE_SUBFLOOR_PATIENCE,
                "ending_subfloor_streak": performance_subfloor_streak,
            },
        },
        "training_performed": False,
        "optimizer_updates": 0,
        "map_decision_made": False,
    })
    atomic_write_new_json(completed_path, receipt, immutable=True)
    del gpu, resources, index, encoded, scales
    gc.collect()
    torch.cuda.empty_cache()
    return {**receipt, "receipt": expected_input_signature(completed_path)}


def _validate_part_receipt(
    output: str,
    *,
    expected_sha256: str | None,
    part: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = os.path.join(output, "part-receipt.json")
    signature = expected_input_signature(path)
    if expected_sha256 and signature["sha256"] != expected_sha256:
        raise Round0106Error(f"R0106 {part} part receipt bytes changed")
    receipt = _load_json(path)
    body = {
        key: value
        for key, value in receipt.items()
        if key != "identity_sha256"
    }
    spec = part_spec(part)
    if (
        receipt.get("schema") != PART_SCHEMA
        or receipt.get("round_id") != ROUND_ID
        or receipt.get("part") != part
        or receipt.get("part_spec") != spec
        or receipt.get("identity_sha256") != sha256_bytes(canonical_json(body))
        or int(receipt.get("retained_sources", -1)) != spec["retained_rows"]
        or int(receipt.get("knn_edges", -1)) != spec["retained_rows"] * K
        or int(receipt.get("directed_edges", -1))
        + int(receipt.get("zero_memberships_eliminated", -1))
        != int(receipt.get("knn_edges", -1))
        or int(receipt.get("directed_edges", -1)) < spec["retained_rows"]
        or int(receipt.get("sources_with_eliminated_memberships", -1)) < 0
        or int(receipt.get("sources_with_eliminated_memberships", -1))
        > spec["retained_rows"]
        or int(receipt.get("minimum_memberships_per_source", -1)) < 1
        or int(receipt.get("minimum_memberships_per_source", -1)) > K
    ):
        raise Round0106Error(f"R0106 {part} part receipt contract changed")
    for member in receipt.get("shards") or []:
        _validate_signature(member["receipt"], label=f"R0106 {part} shard receipt")
        _validate_signature(member["artifact"], label=f"R0106 {part} shard")
    return receipt, signature


def _partition_forward_edges(
    *,
    output: str,
    parts: Mapping[str, tuple[str, Mapping[str, Any]]],
    contract_sha256: str,
) -> str:
    final = os.path.join(output, "forward-buckets")
    receipt_path = os.path.join(final, "_DONE.json")
    if os.path.exists(receipt_path):
        receipt = _load_json(receipt_path)
        if (
            receipt.get("contract_sha256") != contract_sha256
            or int(receipt.get("partitions", -1)) != PAIR_BUCKETS
        ):
            raise Round0106Error("R0106 forward-bucket receipt changed")
        for member in receipt.get("buckets") or []:
            _validate_signature(
                {
                    **member,
                    "canonical_path": os.path.join(final, member["path"]),
                    "kind": "file",
                },
                label="R0106 forward bucket",
            )
        return final
    temporary = os.path.join(output, "forward-buckets.tmp")
    if os.path.exists(temporary):
        shutil.rmtree(temporary)
    os.mkdir(temporary)
    handles = [
        open(os.path.join(temporary, f"p{bucket:04d}.bin"), "wb")
        for bucket in range(PAIR_BUCKETS)
    ]
    expected_forward_records = sum(
        int(receipt["directed_edges"]) for _root, receipt in parts.values()
    )
    shard_count = 0
    record_count = 0
    try:
        for part in PARTS:
            root, receipt = parts[part]
            for member in receipt["shards"]:
                path = str(member["artifact"]["canonical_path"])
                with np.load(path, allow_pickle=False) as archive:
                    sources = archive["sources"]
                    targets = archive["targets"]
                    weights = archive["weights"]
                    low = np.minimum(sources, targets)
                    high = np.maximum(sources, targets)
                    buckets = _pair_bucket(
                        low, high, RETAINED_ROWS, PAIR_BUCKETS
                    )
                    order = np.argsort(buckets, kind="stable")
                    sorted_buckets = buckets[order]
                    records = np.empty(len(order), dtype=_REC)
                    records["s"] = sources[order]
                    records["t"] = targets[order]
                    records["w"] = weights[order]
                    boundaries = np.flatnonzero(
                        sorted_buckets[1:] != sorted_buckets[:-1]
                    ) + 1
                    starts = np.r_[0, boundaries]
                    stops = np.r_[boundaries, len(order)]
                    for start, stop in zip(starts, stops):
                        handles[int(sorted_buckets[start])].write(
                            records[start:stop].tobytes()
                        )
                    record_count += len(records)
                shard_count += 1
                if shard_count % 25 == 0:
                    print(
                        f"R0106 assembly partitioned {shard_count} shards",
                        flush=True,
                    )
    finally:
        for handle in handles:
            handle.close()
    if record_count != expected_forward_records:
        raise Round0106Error("R0106 forward bucket edge count did not close")
    os.replace(temporary, final)
    buckets = []
    for bucket in range(PAIR_BUCKETS):
        path = os.path.join(final, f"p{bucket:04d}.bin")
        signature = expected_input_signature(path)
        buckets.append({
            "path": os.path.basename(path),
            "bytes": signature["bytes"],
            "sha256": signature["sha256"],
            "records": signature["bytes"] // _REC.itemsize,
        })
    atomic_write_new_json(
        receipt_path,
        {
            "schema": "round0106-forward-pair-buckets-v1",
            "contract_sha256": contract_sha256,
            "phase_a_closure_sha256": contract_sha256,
            "partitions": PAIR_BUCKETS,
            "forward_records": record_count,
            "expected_forward_records": expected_forward_records,
            "buckets": buckets,
        },
        immutable=True,
    )
    return final


def _publish_memmaps(
    *,
    output: str,
    joined: str,
    counts: list[int],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    destinations = {
        "sources": os.path.join(output, "sources.i32.npy"),
        "targets": os.path.join(output, "targets.i32.npy"),
        "weights": os.path.join(output, "weights.f32.npy"),
    }
    if all(os.path.exists(path) for path in destinations.values()):
        return tuple(
            expected_input_signature(destinations[key])
            for key in ("sources", "targets", "weights")
        )
    if any(os.path.exists(path) for path in destinations.values()):
        raise Round0106Error("R0106 final edge arrays are only partly present")
    total = int(sum(counts))
    temporary: dict[str, str] = {}
    arrays: dict[str, np.memmap] = {}
    try:
        for key, dtype in (
            ("sources", "<i4"),
            ("targets", "<i4"),
            ("weights", "<f4"),
        ):
            fd, path = tempfile.mkstemp(
                prefix=f".{key}.",
                suffix=".tmp.npy",
                dir=output,
            )
            os.close(fd)
            os.unlink(path)
            temporary[key] = path
            arrays[key] = np.lib.format.open_memmap(
                path, mode="w+", dtype=dtype, shape=(total,)
            )
        cursor = 0
        for bucket, count in enumerate(counts):
            path = os.path.join(joined, f"part-{bucket:04d}.npz")
            with np.load(path, allow_pickle=False) as archive:
                stop = cursor + count
                arrays["sources"][cursor:stop] = archive["sources"]
                arrays["targets"][cursor:stop] = archive["targets"]
                arrays["weights"][cursor:stop] = archive["weights"]
            cursor = stop
        if cursor != total:
            raise Round0106Error("R0106 final edge concatenation did not close")
        for key, array in arrays.items():
            array.flush()
            del array
            os.chmod(temporary[key], 0o444)
            os.link(temporary[key], destinations[key], follow_symlinks=False)
    finally:
        arrays.clear()
        for path in temporary.values():
            with contextlib.suppress(OSError):
                os.unlink(path)
    return tuple(
        expected_input_signature(destinations[key])
        for key in ("sources", "targets", "weights")
    )


def _validate_joined_reciprocity(
    *,
    joined: str,
    counts: list[int],
) -> dict[str, Any]:
    """Full bucket-local proof that every final edge has one equal reverse."""
    checked = 0
    for bucket, expected_count in enumerate(counts):
        path = os.path.join(joined, f"part-{bucket:04d}.npz")
        with np.load(path, allow_pickle=False) as archive:
            sources = np.asarray(archive["sources"], dtype=np.int64)
            targets = np.asarray(archive["targets"], dtype=np.int64)
            weights = np.asarray(archive["weights"], dtype=np.float32)
        if len(sources) != expected_count:
            raise Round0106Error(
                f"R0106 joined bucket {bucket} count changed"
            )
        keys = sources.astype(np.uint64) * np.uint64(RETAINED_ROWS)
        keys += targets.astype(np.uint64)
        if len(keys) > 1 and np.any(keys[1:] <= keys[:-1]):
            raise Round0106Error(
                f"R0106 joined bucket {bucket} keys are not unique/sorted"
            )
        reverse = targets.astype(np.uint64) * np.uint64(RETAINED_ROWS)
        reverse += sources.astype(np.uint64)
        positions = np.searchsorted(keys, reverse)
        bounded = positions < len(keys)
        if not np.all(bounded):
            raise Round0106Error(
                f"R0106 joined bucket {bucket} lacks reverse edges"
            )
        if (
            not np.array_equal(keys[positions], reverse)
            or not np.array_equal(weights[positions], weights)
        ):
            raise Round0106Error(
                f"R0106 joined bucket {bucket} reciprocal weights changed"
            )
        checked += len(keys)
    return {
        "method": "full-bucket-key-scan",
        "directed_edges_checked": checked,
        "unique_directed_keys": True,
        "every_reverse_present_once": True,
        "reciprocal_weights_equal": True,
    }


def _publish_compact_mapping(
    *,
    output: str,
    excluded: np.ndarray,
) -> dict[str, Any]:
    path = os.path.join(output, "compact-to-global.i64.npy")
    if os.path.exists(path):
        signature = expected_input_signature(path)
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            values.shape != (RETAINED_ROWS,)
            or values.dtype != np.int64
            or int(values[0]) < 0
            or int(values[-1]) >= ROW_COUNT
        ):
            raise Round0106Error("R0106 compact mapping output changed")
        return signature
    mapping = np.empty(RETAINED_ROWS, dtype=np.int64)
    for start in range(0, RETAINED_ROWS, 1_000_000):
        stop = min(start + 1_000_000, RETAINED_ROWS)
        mapping[start:stop] = compact_to_global(
            np.arange(start, stop, dtype=np.int64), excluded
        )
    atomic_save_new_npy(path, mapping, immutable=True)
    return expected_input_signature(path)


def _graph_diagnostics(
    *,
    sources_path: str,
    targets_path: str,
    weights_path: str,
    compact_mapping_path: str,
    labels_path: str,
) -> dict[str, Any]:
    sources = np.load(sources_path, mmap_mode="r", allow_pickle=False)
    targets = np.load(targets_path, mmap_mode="r", allow_pickle=False)
    weights = np.load(weights_path, mmap_mode="r", allow_pickle=False)
    mapping = np.load(compact_mapping_path, mmap_mode="r", allow_pickle=False)
    with np.load(labels_path, allow_pickle=False) as labels:
        dataset_ids = labels["dataset_id"]
        compact_groups = np.asarray(dataset_ids[mapping], dtype=np.uint8)
    if (
        sources.dtype != np.int32
        or targets.dtype != np.int32
        or weights.dtype != np.float32
        or sources.shape != targets.shape
        or sources.shape != weights.shape
    ):
        raise Round0106Error("R0106 final edge geometry changed")
    mixing = np.zeros((len(GROUPS), len(GROUPS)), dtype=np.int64)
    part_mixing = np.zeros((len(PARTS), len(PARTS)), dtype=np.int64)
    part_bounds = np.asarray(
        [PARTS[name]["compact_stop"] for name in PARTS], dtype=np.int64
    )
    minimum = None
    maximum = None
    weight_sum = 0.0
    for start in range(0, len(sources), FINAL_SCAN_ROWS):
        stop = min(start + FINAL_SCAN_ROWS, len(sources))
        source = np.asarray(sources[start:stop])
        target = np.asarray(targets[start:stop])
        weight = np.asarray(weights[start:stop])
        if (
            np.any(source < 0)
            or np.any(source >= RETAINED_ROWS)
            or np.any(target < 0)
            or np.any(target >= RETAINED_ROWS)
            or np.any(source == target)
            or not np.isfinite(weight).all()
            or np.any(weight <= 0)
            or np.any(weight > 1)
        ):
            raise Round0106Error("R0106 final edge structural scan failed")
        source_group = compact_groups[source]
        target_group = compact_groups[target]
        mixing += np.bincount(
            source_group.astype(np.int64) * len(GROUPS)
            + target_group.astype(np.int64),
            minlength=len(GROUPS) ** 2,
        ).reshape(len(GROUPS), len(GROUPS))
        source_part = np.searchsorted(part_bounds, source, side="right")
        target_part = np.searchsorted(part_bounds, target, side="right")
        part_mixing += np.bincount(
            source_part * len(PARTS) + target_part,
            minlength=len(PARTS) ** 2,
        ).reshape(len(PARTS), len(PARTS))
        value_min = float(weight.min())
        value_max = float(weight.max())
        minimum = value_min if minimum is None else min(minimum, value_min)
        maximum = value_max if maximum is None else max(maximum, value_max)
        weight_sum += float(weight.astype(np.float64).sum())
    degrees = np.bincount(sources, minlength=RETAINED_ROWS)
    if (
        int(degrees.sum()) != len(sources)
        or np.any(degrees <= 0)
        or int(mixing.sum()) != len(sources)
        or int(part_mixing.sum()) != len(sources)
    ):
        raise Round0106Error("R0106 diagnostic accounting did not close")
    group_rows = mixing.sum(axis=1)
    return {
        "groups": list(GROUPS),
        "mixing_matrix": mixing.tolist(),
        "within_group_fraction": {
            group: float(mixing[index, index] / group_rows[index])
            for index, group in enumerate(GROUPS)
        },
        "part_order": list(PARTS),
        "part_mixing_matrix": part_mixing.tolist(),
        "degree": {
            "minimum": int(degrees.min()),
            "p10": float(np.percentile(degrees, 10)),
            "median": float(np.median(degrees)),
            "p90": float(np.percentile(degrees, 90)),
            "p99": float(np.percentile(degrees, 99)),
            "maximum": int(degrees.max()),
            "maximum_hub_share": float(degrees.max() / len(sources)),
            "zero_degree_rows": int(np.count_nonzero(degrees == 0)),
        },
        "weights": {
            "minimum": minimum,
            "mean": weight_sum / len(weights),
            "maximum": maximum,
            "sum": weight_sum,
        },
        "structural_scan": {
            "directed_edges": len(sources),
            "endpoint_bounds_valid": True,
            "self_edges": 0,
            "weight_domain_valid": True,
        },
    }


def run_assemble(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    output = ensure_data_directory(
        str(job["outputs"][0]), label="R0106 graph assembly"
    )
    final_manifest_path = os.path.join(output, "graph-manifest.json")
    if os.path.exists(final_manifest_path):
        manifest = _load_json(final_manifest_path)
        body = {
            key: value
            for key, value in manifest.items()
            if key != "identity_sha256"
        }
        if (
            manifest.get("schema") != GRAPH_SCHEMA
            or manifest.get("identity_sha256")
            != sha256_bytes(canonical_json(body))
        ):
            raise Round0106Error("R0106 final graph manifest changed")
        return {
            **manifest,
            "receipt": expected_input_signature(final_manifest_path),
        }
    part_values: dict[str, tuple[str, Mapping[str, Any]]] = {}
    part_signatures = {}
    for part in PARTS:
        root = str(job["part_outputs"][part])
        expected_part_hashes = job.get("part_receipt_sha256") or {}
        receipt, signature = _validate_part_receipt(
            root,
            expected_sha256=expected_part_hashes.get(part),
            part=part,
        )
        part_values[part] = (root, receipt)
        part_signatures[part] = signature
    knn_edge_count = sum(
        int(receipt["knn_edges"]) for _root, receipt in part_values.values()
    )
    forward_membership_count = sum(
        int(receipt["directed_edges"])
        for _root, receipt in part_values.values()
    )
    zero_memberships_eliminated = sum(
        int(receipt["zero_memberships_eliminated"])
        for _root, receipt in part_values.values()
    )
    if (
        knn_edge_count != RETAINED_ROWS * K
        or forward_membership_count + zero_memberships_eliminated
        != knn_edge_count
    ):
        raise Round0106Error("R0106 pre-assembly membership accounting changed")
    substrate, excluded, _encoded, _scales = _substrate_arrays()
    labels_signature = substrate["manifest"]["outputs"]["labels"]
    _validate_signature(labels_signature, label="R0103 labels")
    assembly_contract = sha256_bytes(canonical_json({
        "schema": GRAPH_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "parts": part_signatures,
        "retained_rows": RETAINED_ROWS,
        "k": K,
        "pair_buckets": PAIR_BUCKETS,
        "symmetrization": "a+b-a*b-set-op-mix-ratio-1",
    }))
    forward_buckets = _partition_forward_edges(
        output=output,
        parts=part_values,
        contract_sha256=assembly_contract,
    )
    joined, join_stats = phase_c_join(
        forward_buckets,
        output,
        RETAINED_ROWS,
        PAIR_BUCKETS,
        contract_sha256=assembly_contract,
        workers=ASSEMBLY_WORKERS,
    )
    counts = [int(value) for value in join_stats["counts"]]
    reciprocity = _validate_joined_reciprocity(
        joined=joined, counts=counts
    )
    sources_signature, targets_signature, weights_signature = _publish_memmaps(
        output=output,
        joined=joined,
        counts=counts,
    )
    mapping_signature = _publish_compact_mapping(
        output=output, excluded=excluded
    )
    diagnostics = _graph_diagnostics(
        sources_path=sources_signature["canonical_path"],
        targets_path=targets_signature["canonical_path"],
        weights_path=weights_signature["canonical_path"],
        compact_mapping_path=mapping_signature["canonical_path"],
        labels_path=labels_signature["canonical_path"],
    )
    if int(diagnostics["structural_scan"]["directed_edges"]) != int(
        join_stats["n_edges"]
    ):
        raise Round0106Error("R0106 final edge count disagrees with join")
    manifest = seal({
        "schema": GRAPH_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "contract_sha256": assembly_contract,
        "row_count": ROW_COUNT,
        "retained_rows": RETAINED_ROWS,
        "dimension": DIMENSION,
        "k_real": K,
        "n_neighbors_including_self": N_NEIGHBORS,
        "local_connectivity": LOCAL_CONNECTIVITY,
        "part_receipts": part_signatures,
        "substrate": substrate["signature"],
        "eligibility": expected_input_signature(ELIGIBILITY_PATH),
        "labels": labels_signature,
        "compact_mapping": mapping_signature,
        "outputs": {
            "sources": sources_signature,
            "targets": targets_signature,
            "weights": weights_signature,
        },
        "knn_topology": {
            "distinct_nonself_neighbors_per_source": K,
            "knn_edge_count": knn_edge_count,
            "source_coverage_complete": True,
        },
        "forward_memberships": {
            "positive_count": forward_membership_count,
            "zero_memberships_eliminated": zero_memberships_eliminated,
            "elimination_semantics": "umap-eliminate-zeros-after-fp32-cast",
            "sources_with_eliminated_memberships": sum(
                int(receipt["sources_with_eliminated_memberships"])
                for _root, receipt in part_values.values()
            ),
            "minimum_positive_memberships_per_source": min(
                int(receipt["minimum_memberships_per_source"])
                for _root, receipt in part_values.values()
            ),
        },
        "directed_edge_count": int(join_stats["n_edges"]),
        "weight_sum": float(join_stats["weight_sum"]),
        "symmetrization": {
            "method": "probabilistic-t-conorm",
            "formula": "a+b-a*b",
            "set_op_mix_ratio": 1.0,
            "unordered_pair_partitions": PAIR_BUCKETS,
            "both_orientations_emitted": True,
        },
        "diagnostics": diagnostics,
        "reciprocity_validation": reciprocity,
        "performance": {
            "wall_seconds": time.monotonic() - started,
            "peak_rss_gib": _peak_rss_gib(),
            "assembly_workers": ASSEMBLY_WORKERS,
        },
        "training_performed": False,
        "optimizer_updates": 0,
        "map_decision_made": False,
    })
    atomic_write_new_json(
        final_manifest_path, manifest, immutable=True
    )
    return {
        **manifest,
        "receipt": expected_input_signature(final_manifest_path),
    }


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID or job is None:
        raise Round0106Error("R0106 handler requires its exact round/job")
    action = job.get("action")
    if action == "build_part":
        return run_build_part(active, job)
    if action == "assemble_graph":
        return run_assemble(active, job)
    raise Round0106Error(f"unknown R0106 action: {action!r}")
