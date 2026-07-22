#!/usr/bin/env python3
"""Slim-runner handlers for R0036's 150M evaluation and map publication."""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    ordered_array_sha256,
    sha256_bytes,
)
from basemap.output_safety import (
    atomic_build_new_file,
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0034_program import INT8_PATH, SCALES_PATH
from basemap.round0036_pipeline import (
    CoordinateStream,
    DIMENSION,
    INDEX_PATH,
    RETAINED_ROWS,
    ROW_COUNT,
    TRANSFORM_SCHEMA,
    COORDINATE_SCHEMA,
    EncodedInt8Array,
    RetainedArrayView,
    RetainedFaissIndex,
    Round0036Error,
    load_released_selector,
    load_reviewed_model,
    panel_config_identity,
    seal,
    validate_reviewed_model_bundle,
)


CENTROIDS = {
    256: "/data/latent-basemap/track1/centroids_minilm_k256.npy",
    1024: "/data/latent-basemap/track1/centroids_minilm_k1024.npy",
}
MINILM_QUERIES = "/data/latent-basemap/track1/minilm_queries.npy"
MINILM_QUERY_PROVENANCE = "/data/latent-basemap/track1/minilm_queries_prov.json"
MAP_LABEL = "r0034-150m-seed42"
SCORER_CANARY_ROWS = 2_000_000
SCORER_CANARY_ANCHORS = 512
SCORER_TOLERANCES = {
    "ffr": 0.01,
    "recall@k": 0.01,
    "density": 0.02,
    "purity_k1024": 0.05,
    "proj_ffr": 0.02,
}


def _read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _bundle(job: Mapping[str, Any]) -> dict[str, Any]:
    return validate_reviewed_model_bundle(
        model_path=job["model_path"],
        model_sha256=job["model_sha256"],
        train_receipt_path=job["train_receipt_path"],
        train_receipt_sha256=job["train_receipt_sha256"],
    )


def _load_model(job: Mapping[str, Any], *, device: str = "cuda") -> Any:
    return load_reviewed_model(_bundle(job), device=device)


def _project_float(model: Any, vectors: np.ndarray, *, batch_rows: int = 65_536) -> np.ndarray:
    import torch

    output = np.empty((len(vectors), 2), dtype="<f4")
    model.model.eval()
    with torch.no_grad():
        for start in range(0, len(vectors), batch_rows):
            block = np.asarray(vectors[start : start + batch_rows], dtype=np.float32)
            tensor = torch.from_numpy(np.array(block, copy=True)).to(model.device)
            projected = model.model(tensor).detach().cpu().numpy().astype("<f4")
            if projected.shape != (len(block), 2) or not np.isfinite(projected).all():
                raise Round0036Error("R0036 projection emitted malformed coordinates")
            output[start : start + len(block)] = projected
        if str(model.device).startswith("cuda"):
            torch.cuda.synchronize()
    return output


def _project_encoded_block(
    model: Any,
    encoded: EncodedInt8Array,
    start: int,
    stop: int,
    *,
    batch_rows: int,
) -> np.ndarray:
    """Project a global row interval with one invariant CUDA GEMM geometry.

    The production model is row-independent, but CUDA may choose a different
    GEMM kernel for the short tail of a 5M coordinate chunk.  Padding every
    short tail to ``batch_rows`` makes duplicate rows traverse the same matrix
    shape wherever they occur, while still evaluating every real row through
    the network (coordinates are never copied from a representative).
    """
    output = np.empty((stop - start, 2), dtype="<f4")
    for batch_start in range(start, stop, batch_rows):
        batch_stop = min(batch_start + batch_rows, stop)
        block = np.asarray(encoded[batch_start:batch_stop], dtype=np.float32)
        real_rows = len(block)
        if real_rows < batch_rows:
            padded = np.zeros((batch_rows, block.shape[1]), dtype=np.float32)
            padded[:real_rows] = block
            block = padded
        projected = _project_float(model, block, batch_rows=batch_rows)
        output[batch_start - start : batch_stop - start] = projected[:real_rows]
    return output


def _project_lazy_array_shape_stable(
    model: Any,
    vectors: Any,
    *,
    batch_rows: int = 65_536,
) -> np.ndarray:
    """Shape-stable projection for a lazy matrix without materialising it."""
    output = np.empty((len(vectors), 2), dtype="<f4")
    for start in range(0, len(vectors), batch_rows):
        stop = min(start + batch_rows, len(vectors))
        block = np.asarray(vectors[start:stop], dtype=np.float32)
        real_rows = len(block)
        if real_rows < batch_rows:
            padded = np.zeros((batch_rows, block.shape[1]), dtype=np.float32)
            padded[:real_rows] = block
            block = padded
        output[start:stop] = _project_float(
            model, block, batch_rows=batch_rows
        )[:real_rows]
    return output


def _gather_members(members: list[dict[str, Any]], rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    output = np.empty((len(rows), 2), dtype="<f4")
    for member in members:
        low = int(member["global_row_start"])
        high = int(member["global_row_stop"])
        selected = np.flatnonzero((rows >= low) & (rows < high))
        if len(selected):
            array = np.load(member["path"], mmap_mode="r", allow_pickle=False)
            output[selected] = array[rows[selected] - low]
    return output


def _validate_duplicate_coordinates(
    members: list[dict[str, Any]],
    duplicate_rows: np.ndarray,
    representative_rows: np.ndarray,
    *,
    block_rows: int = 250_000,
) -> dict[str, Any]:
    if duplicate_rows.shape != representative_rows.shape:
        raise Round0036Error("duplicate coordinate validation map is malformed")
    compared = 0
    for start in range(0, len(duplicate_rows), block_rows):
        left_rows = duplicate_rows[start : start + block_rows]
        right_rows = representative_rows[start : start + block_rows]
        left = _gather_members(members, left_rows)
        right = _gather_members(members, right_rows)
        if not np.array_equal(left, right):
            mismatch = int(np.flatnonzero(np.any(left != right, axis=1))[0])
            raise Round0036Error(
                "duplicate coordinate differs from representative: "
                f"copy={int(left_rows[mismatch])} rep={int(right_rows[mismatch])}"
            )
        compared += len(left)
    return {
        "policy": "all duplicate copies compared bit-for-bit to R0033 representative",
        "pairs_compared": int(compared),
        "all_equal": True,
        "duplicate_rows_sha256": ordered_array_sha256(duplicate_rows),
        "representative_rows_sha256": ordered_array_sha256(representative_rows),
    }


def _exact_retained_2d_slice_search(
    coordinates: np.ndarray,
    selector: Any,
    *,
    anchors: np.ndarray,
    k: int = 50,
) -> dict[str, Any]:
    """Exercise exact 2D candidates in the compact retained namespace."""
    if len(coordinates) != selector.retained_count or k >= len(coordinates):
        raise Round0036Error("R0036 fixed-slice 2D canary geometry is invalid")
    compact_neighbors = np.empty((len(anchors), k), dtype=np.int64)
    for position, anchor in enumerate(np.asarray(anchors, dtype=np.int64)):
        delta = coordinates - coordinates[int(anchor)]
        distance = np.einsum("ij,ij->i", delta, delta)
        distance[int(anchor)] = np.inf
        candidates = np.argpartition(distance, k - 1)[:k]
        order = np.lexsort((candidates, distance[candidates]))
        compact_neighbors[position] = candidates[order]
    global_anchors = selector.compact_to_global(anchors)
    global_neighbors = selector.compact_to_global(compact_neighbors)
    retained = bool(np.all(selector.is_retained(global_neighbors)))
    self_free = not bool(np.any(global_neighbors == global_anchors[:, None]))
    return {
        "algorithm": "exact-fp32-squared-l2-argpartition-then-distance/id-order",
        "corpus_rows": int(len(coordinates)),
        "anchor_count": int(len(anchors)),
        "k": int(k),
        "anchor_global_rows": global_anchors.tolist(),
        "neighbors_global_sha256": ordered_array_sha256(global_neighbors),
        "all_candidates_retained": retained,
        "self_excluded": self_free,
        "passed": retained and self_free,
    }


def _run_2m_scorer_parity(
    *,
    model: Any,
    encoded: EncodedInt8Array,
    selector: Any,
) -> dict[str, Any]:
    """Compare the retained-view scorer to the registered independent 2M path."""
    parity_started = time.monotonic()
    from basemap.panel_v2 import (
        PanelV2Config,
        build_hiD_reference,
        sample_anchors,
        score_panel,
    )
    from experiments.golden_validate import _exact_reference
    from experiments.score_complete_panel import projection_ffr

    local_selector = type(selector)(
        selector.excluded_rows[selector.excluded_rows < SCORER_CANARY_ROWS],
        row_count=SCORER_CANARY_ROWS,
    )
    prefix = EncodedInt8Array(
        encoded.encoded[:SCORER_CANARY_ROWS],
        encoded.scales[:SCORER_CANARY_ROWS],
        signatures=encoded.signatures,
    )
    retained = RetainedArrayView(prefix, local_selector)
    stage_started = time.monotonic()
    coordinates = _project_lazy_array_shape_stable(model, retained)
    transform_wall = time.monotonic() - stage_started
    config = PanelV2Config(
        frac=0.001,
        n_anchors=SCORER_CANARY_ANCHORS,
        anchor_seed=42,
        corpus_chunk=500_000,
    )
    centroids = {
        key: np.load(path, mmap_mode="r", allow_pickle=False)
        for key, path in CENTROIDS.items()
    }
    anchors = sample_anchors(len(retained), config).astype(np.int64)
    data_identity = {
        "kind": "round0036-production-prefix-selection",
        "source": encoded.scientific_identity(),
        "global_row_interval": [0, SCORER_CANARY_ROWS],
        "shape": [len(retained), DIMENSION],
        "dtype": "<f4",
        "selector": local_selector.identity(),
    }
    convention = {
        "row_order": "compact ascending retained rows inside global [0, 2M)",
        "distance": "squared L2 on fp32 dequantized R0025 int8 rows",
        "self_exclusion": True,
        "anchor_namespace": "compact retained-row positions",
    }
    reference_identity = {
        "data_identity": data_identity,
        "convention": convention,
    }
    stage_started = time.monotonic()
    reference = build_hiD_reference(
        retained,
        anchors,
        config,
        centroids,
        data_identity=data_identity,
        convention=convention,
    )
    reference_wall = time.monotonic() - stage_started
    stage_started = time.monotonic()
    streamed_panel = score_panel(
        retained,
        coordinates,
        config=config,
        centroids_by_k=centroids,
        hiD_reference=reference,
        reference_identity=reference_identity,
        provenance={
            "gate": "round0036-retained-adapter-2m-parity",
            "global_row_interval": [0, SCORER_CANARY_ROWS],
        },
    )
    streamed_panel_wall = time.monotonic() - stage_started
    queries = np.load(MINILM_QUERIES, mmap_mode="r", allow_pickle=False)[:128]
    query_coordinates = _project_float(model, queries)
    stage_started = time.monotonic()
    streamed_projection, _ = projection_ffr(
        retained, coordinates, queries, query_coordinates, config
    )
    streamed_projection_wall = time.monotonic() - stage_started
    stage_started = time.monotonic()
    exact = _exact_reference(
        retained,
        coordinates,
        config,
        centroids=centroids,
        proj={"Xq": queries, "Zq": query_coordinates},
    )
    independent_reference_wall = time.monotonic() - stage_started
    streamed = {
        "ffr": streamed_panel["ffr"],
        "recall@k": streamed_panel["recall@k"],
        "density": streamed_panel["density"],
        "purity_k1024": streamed_panel["purity"]["k1024"],
        "proj_ffr": streamed_projection,
    }
    deltas = {
        key: round(abs(float(streamed[key]) - float(exact[key])), 5)
        for key in SCORER_TOLERANCES
    }
    checks = {
        key: deltas[key] <= tolerance
        for key, tolerance in SCORER_TOLERANCES.items()
    }
    two_d = _exact_retained_2d_slice_search(
        coordinates,
        local_selector,
        anchors=anchors[:8],
    )
    return {
        "schema": "round0036-retained-scorer-2m-parity-v1",
        "global_row_interval": [0, SCORER_CANARY_ROWS],
        "input_rows": SCORER_CANARY_ROWS,
        "retained_rows": len(retained),
        "selector": local_selector.identity(),
        "inference": {
            "batch_rows": 65_536,
            "short_tail_policy": "zero-pad-to-fixed-batch-then-discard-padding",
            "all_real_rows_projected": True,
            "coordinates_sha256": ordered_array_sha256(coordinates),
        },
        "config": {
            "frac": config.frac,
            "n_anchors": config.n_anchors,
            "anchor_seed": config.anchor_seed,
            "corpus_chunk": config.corpus_chunk,
            "k_hit": config.k_hit,
            "k_density": config.k_density,
        },
        "registered_tolerances": dict(SCORER_TOLERANCES),
        "streamed": streamed,
        "independent_fp64_reference": {
            key: exact[key] for key in SCORER_TOLERANCES
        },
        "deltas": deltas,
        "per_metric_pass": checks,
        "retained_only_2d_search": two_d,
        "stage_wall_seconds": {
            "shape_stable_transform": transform_wall,
            "streamed_high_d_reference": reference_wall,
            "streamed_panel": streamed_panel_wall,
            "streamed_projection": streamed_projection_wall,
            "independent_fp64_reference": independent_reference_wall,
            "total": time.monotonic() - parity_started,
        },
        "passed": all(checks.values()) and two_d["passed"],
    }


def _production_canary_body(job: Mapping[str, Any], *, started: float) -> dict[str, Any]:
    bundle = _bundle(job)
    selector, eligibility = load_released_selector(
        job["eligibility_path"], eligibility_sha256=job["eligibility_sha256"]
    )
    encoded = EncodedInt8Array.from_files()
    counts = eligibility["family_counts"]
    largest = int(np.argmax(counts))
    low = int(eligibility["family_offsets"][largest])
    high = int(eligibility["family_offsets"][largest + 1])
    family = eligibility["member_rows"][low:high]
    if len(family) != 141_158 or int(family[0]) != 126_474:
        raise Round0036Error("R0033 largest duplicate family changed")
    sample = np.concatenate((family[:8], eligibility["zero_rows"][:8]))
    model = load_reviewed_model(bundle, device="cuda")
    coords = _project_float(model, encoded[sample], batch_rows=len(sample))
    if not np.array_equal(coords[:8], np.repeat(coords[:1], 8, axis=0)):
        raise Round0036Error("largest-family projection is not bit-identical")

    import faiss

    index_signature = expected_input_signature(INDEX_PATH)
    index = faiss.read_index(INDEX_PATH)
    filtered = RetainedFaissIndex(index, selector, nprobe=128)
    _distances, neighbors = filtered.search_global(encoded[family[:1]], 15)
    if np.any(np.isin(neighbors, family[1:], assume_unique=False)):
        raise Round0036Error("filtered production IVF admitted largest-family copies")
    index_class = type(index).__name__
    index_rows = int(index.ntotal)
    del filtered, index
    scorer_parity = _run_2m_scorer_parity(
        model=model,
        encoded=encoded,
        selector=selector,
    )
    return {
        "schema": "round0036-production-canary-v1",
        "round_id": "0036",
        "model": bundle["model"],
        "train_receipt": bundle["train_receipt"],
        "eligibility": eligibility["signature"],
        "selector": selector.identity(),
        "largest_family": {
            "representative_row": int(family[0]),
            "member_count": int(len(family)),
            "sample_rows": sample.tolist(),
            "sample_coordinates_sha256": ordered_array_sha256(coords),
            "sampled_duplicate_coordinates_bit_identical": True,
        },
        "retained_ivf": {
            "index": index_signature,
            "index_class": index_class,
            "ntotal": index_rows,
            "nprobe": 128,
            "selector": "FAISS-IDSelectorBitmap-exact-include-bitmap",
            "returned_neighbors": neighbors.tolist(),
            "all_returned_ids_retained": True,
            "largest_excluded_family_absent": True,
        },
        "scorer_parity": scorer_parity,
        "passed": bool(scorer_parity["passed"]),
        "wall_seconds": time.monotonic() - started,
    }


def run_production_canary(_active: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    output = create_fresh_directory(job["outputs"][0], label="R0036 canary output")
    started = time.monotonic()
    try:
        body = _production_canary_body(job, started=started)
    except Exception as exc:
        body = {
            "schema": "round0036-production-canary-v1",
            "round_id": "0036",
            "passed": False,
            "failure": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
            "wall_seconds": time.monotonic() - started,
        }
    path = os.path.join(output, "verdict.json")
    receipt = seal(body)
    atomic_write_new_json(path, receipt, immutable=True)
    if not receipt["passed"]:
        failure = receipt.get("failure") or {
            "type": "ScorerParityFailure",
            "message": "one or more registered 2M scorer tolerances failed",
        }
        raise Round0036Error(
            f"R0036 production canary failed: {failure['type']}: {failure['message']}"
        )
    return {**receipt, "verdict": expected_input_signature(path)}


def run_transform(_active: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    output = create_fresh_directory(job["outputs"][0], label="R0036 transform output")
    started = time.monotonic()
    bundle = _bundle(job)
    selector, eligibility = load_released_selector(
        job["eligibility_path"], eligibility_sha256=job["eligibility_sha256"]
    )
    encoded = EncodedInt8Array.from_files()
    model = load_reviewed_model(bundle, device="cuda")
    chunk_rows = int(job.get("coordinate_chunk_rows", 5_000_000))
    batch_rows = int(job.get("model_batch_rows", 65_536))
    if chunk_rows <= 0 or batch_rows != 65_536:
        raise ValueError(
            "R0036 transform requires positive chunks and the canary-bound "
            "65,536-row inference geometry"
        )
    # Divisibility is intentionally not required: the tail code is part of the
    # row-coverage contract and should be exercised on every chunk.
    members: list[dict[str, Any]] = []
    for index, start in enumerate(range(0, ROW_COUNT, chunk_rows)):
        stop = min(start + chunk_rows, ROW_COUNT)
        root = create_fresh_directory(
            os.path.join(output, f"chunk-{index:05d}"),
            label="R0036 coordinate chunk",
        )
        path = os.path.join(root, "coordinates.npy")
        coords = _project_encoded_block(
            model, encoded, start, stop, batch_rows=batch_rows
        )
        atomic_save_new_npy(path, coords, immutable=True)
        signature = expected_input_signature(path)
        members.append(
            {
                "chunk_index": index,
                "global_row_start": start,
                "global_row_stop": stop,
                "bytes": signature["bytes"],
                "sha256": signature["sha256"],
                "path": path,
            }
        )
        del coords

    duplicate_validation = _validate_duplicate_coordinates(
        members,
        eligibility["duplicate_excluded_rows"],
        eligibility["duplicate_representative_rows"],
    )
    queries = np.load(MINILM_QUERIES, mmap_mode="r", allow_pickle=False)
    if queries.shape != (10_002, DIMENSION) or queries.dtype.str != "<f4":
        raise Round0036Error("held-out MiniLM query artifact changed")
    query_coordinates = _project_float(model, queries)
    query_path = os.path.join(output, "heldout-query-coordinates.npy")
    atomic_save_new_npy(query_path, query_coordinates, immutable=True)
    summary = eligibility["metadata"]["summary"]
    body = {
        "schema": TRANSFORM_SCHEMA,
        "round_id": "0036",
        "map_label": MAP_LABEL,
        "model": bundle["model"],
        "train_receipt": bundle["train_receipt"],
        "production_config_sha256": bundle["production_config_sha256"],
        "input": {
            "int8": encoded.signatures["int8"],
            "scales": encoded.signatures["scales"],
            "dequantization": "fp32(int8) * fp32(exact stored fp16 row scale)",
        },
        "inference": {
            "batch_rows": batch_rows,
            "short_tail_policy": "zero-pad-to-fixed-batch-then-discard-padding",
            "all_real_rows_projected": True,
            "representative_coordinates_copied_post_hoc": False,
        },
        "eligibility": eligibility["signature"],
        "row_accounting": {
            "all_rows": ROW_COUNT,
            "retained_representatives": int(summary["retained_row_count"]),
            "excluded_duplicate_copies": int(summary["duplicate_copy_rows_excluded"]),
            "zero_invalid_rows": int(summary["zero_row_count"]),
            "scientific_claim_for_zero_invalid_rows": False,
        },
        "coordinate_stream": {
            "schema": COORDINATE_SCHEMA,
            "row_count": ROW_COUNT,
            "dimension": 2,
            "dtype": "<f4",
            "row_order": "R0025 fineweb/redpajama/pile 50M blocks",
            "ordered_chunks": [
                {key: value for key, value in member.items() if key != "path"}
                for member in members
            ],
        },
        "duplicate_coordinate_validation": duplicate_validation,
        "heldout_queries": expected_input_signature(MINILM_QUERIES),
        "heldout_query_provenance": expected_input_signature(MINILM_QUERY_PROVENANCE),
        "heldout_query_coordinates": expected_input_signature(query_path),
        "wall_seconds": time.monotonic() - started,
    }
    receipt = seal(body)
    path = os.path.join(output, "actual-transform.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def _panel_config():
    from basemap.panel_v2 import PanelV2Config

    return PanelV2Config(**{
        key: tuple(value) if key == "k_clust" else value
        for key, value in panel_config_identity().items()
        if key != "formula_version"
    })


def _retained_data_identity(
    encoded: EncodedInt8Array,
    eligibility: Mapping[str, Any],
) -> dict[str, Any]:
    base = encoded.scientific_identity()
    eligibility_signature = eligibility["signature"]
    return {
        "kind": "ordered_shards",
        "shape": [RETAINED_ROWS, DIMENSION],
        "dtype": "<f4",
        "shards": [
            *base["shards"],
            {
                "position": len(base["shards"]),
                "name": "R0033-retained-selector.npz",
                "bytes": int(eligibility_signature["bytes"]),
                "sha256": eligibility_signature["sha256"],
            },
        ],
    }


def _reference_identity(
    encoded: EncodedInt8Array,
    eligibility: Mapping[str, Any],
    selector: Any,
) -> dict[str, Any]:
    return {
        "data_identity": _retained_data_identity(encoded, eligibility),
        "convention": {
            "row_order": "compact ascending R0033-retained global row IDs",
            "selector": selector.identity(),
            "distance": "squared L2 on fp32 dequantized R0025 int8 rows",
            "self_exclusion": True,
            "anchor_namespace": "compact retained-row positions",
        },
    }


def run_high_d_reference(
    _active: dict[str, Any], job: dict[str, Any]
) -> dict[str, Any]:
    from basemap.panel_v2 import (
        _self_knn,
        build_hiD_reference,
        load_hiD_reference,
        sample_anchors,
        save_hiD_reference,
    )

    output = create_fresh_directory(
        job["outputs"][0], label="R0036 high-D reference output"
    )
    started = time.monotonic()
    selector, eligibility = load_released_selector(
        job["eligibility_path"], eligibility_sha256=job["eligibility_sha256"]
    )
    encoded = EncodedInt8Array.from_files()
    retained = RetainedArrayView(encoded, selector)
    config = _panel_config()
    centroids = {
        key: np.load(path, mmap_mode="r", allow_pickle=False)
        for key, path in CENTROIDS.items()
    }
    anchors = sample_anchors(len(retained), config).astype(np.int64)
    identity = _reference_identity(encoded, eligibility, selector)
    reference = build_hiD_reference(
        retained, anchors, config, centroids, **identity
    )
    reference_path = os.path.join(output, "reference.npz")
    save_hiD_reference(reference, reference_path)
    reopened = load_hiD_reference(
        reference_path,
        expected_key=reference["key"],
        expected_key_parts=reference["key_parts"],
    )
    hi50, _, guard50 = _self_knn(
        retained, anchors, 50, config, hi_dim=True, exact=True
    )
    hi50_path = os.path.join(output, "recall50-truth.npy")
    atomic_save_new_npy(hi50_path, hi50.astype(np.int64), immutable=True)
    global_anchors = selector.compact_to_global(anchors)
    anchor_path = os.path.join(output, "anchor-global-rows.npy")
    atomic_save_new_npy(anchor_path, global_anchors, immutable=True)
    body = {
        "schema": "round0036-high-d-reference-v1",
        "round_id": "0036",
        "eligibility": eligibility["signature"],
        "selector": selector.identity(),
        "input": encoded.scientific_identity(),
        "reference": expected_input_signature(reference_path),
        "reference_key": reopened["key"],
        "reference_content_sha256": reopened["content_sha256"],
        "reference_identity": identity,
        "anchor_compact_rows_sha256": ordered_array_sha256(anchors),
        "anchor_global_rows": expected_input_signature(anchor_path),
        "anchor_global_rows_sha256": ordered_array_sha256(global_anchors),
        "recall50_truth": expected_input_signature(hi50_path),
        "recall50_guard": guard50,
        "centroids": {
            f"k{key}": expected_input_signature(path)
            for key, path in CENTROIDS.items()
        },
        "excluded_rows_entered_reference": False,
        "wall_seconds": time.monotonic() - started,
    }
    receipt = seal(body)
    path = os.path.join(output, "reference-receipt.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def _score_untrained_floor(
    *,
    bundle: Mapping[str, Any],
    queries: np.ndarray,
    coordinates: RetainedArrayView,
    truth: np.ndarray,
    config: Any,
) -> dict[str, Any]:
    from experiments.run_round0034_node import _exact_model
    from experiments.score_complete_panel import projection_ffr
    import torch

    cells = []
    for seed_value in (0, 1, 2):
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        model = _exact_model(bundle["production_config"])
        model._init_model(DIMENSION)
        model.model.eval()
        model.is_fitted = True
        projected = _project_float(model, queries)
        ffr, recall = projection_ffr(
            None,
            coordinates,
            None,
            projected,
            config,
            hi_truth=truth[:, : config.k_hit],
        )
        cells.append(
            {
                "seed": seed_value,
                "projection_ffr": ffr,
                "projection_recall_at_k": recall,
                "coordinates_sha256": ordered_array_sha256(projected),
            }
        )
        del model, projected
        torch.cuda.empty_cache()
    floor = max(float(cell["projection_ffr"]) for cell in cells)
    return {
        "policy": "maximum of three deterministic untrained-network seeds",
        "cells": cells,
        "floor_ffr": floor,
        "training_performed": False,
    }


def run_registered_panel(
    _active: dict[str, Any], job: dict[str, Any]
) -> dict[str, Any]:
    from basemap.panel_v2 import (
        QueryTruthCache,
        _self_knn,
        load_hiD_reference,
        recall_at_k_from_neighbors,
        score_panel,
    )
    from experiments.score_complete_panel import score_query_bundle

    output = create_fresh_directory(job["outputs"][0], label="R0036 panel output")
    started = time.monotonic()
    bundle = _bundle(job)
    selector, eligibility = load_released_selector(
        job["eligibility_path"], eligibility_sha256=job["eligibility_sha256"]
    )
    encoded = EncodedInt8Array.from_files()
    retained = RetainedArrayView(encoded, selector)
    coordinates_full = CoordinateStream(job["transform_output"])
    coordinates = RetainedArrayView(coordinates_full, selector)
    config = _panel_config()
    centroids = {
        key: np.load(path, mmap_mode="r", allow_pickle=False)
        for key, path in CENTROIDS.items()
    }
    reference_path = os.path.join(job["reference_output"], "reference.npz")
    reference = load_hiD_reference(reference_path)
    reference_identity = _reference_identity(encoded, eligibility, selector)
    panel = score_panel(
        retained,
        coordinates,
        config=config,
        centroids_by_k=centroids,
        hiD_reference=reference,
        reference_identity=reference_identity,
        provenance={
            "round_id": "0036",
            "map_label": MAP_LABEL,
            "model": bundle["model"],
            "train_receipt": bundle["train_receipt"],
            "coordinate_capability": expected_input_signature(
                os.path.join(job["transform_output"], "actual-transform.json")
            ),
            "eligibility": eligibility["signature"],
            "excluded_rows_entered_scientific_universe": False,
        },
    )
    anchors = np.asarray(reference["anchor_ids"], dtype=np.int64)
    hi50 = np.load(
        os.path.join(job["reference_output"], "recall50-truth.npy"),
        mmap_mode="r",
        allow_pickle=False,
    )
    lo50, _, guard50 = _self_knn(
        coordinates, anchors, 50, config, hi_dim=False, exact=True
    )
    recall50 = round(recall_at_k_from_neighbors(hi50, lo50, 50), 5)

    queries = np.load(MINILM_QUERIES, mmap_mode="r", allow_pickle=False)
    query_coordinates = np.load(
        os.path.join(job["transform_output"], "heldout-query-coordinates.npy"),
        mmap_mode="r",
        allow_pickle=False,
    )
    cache = QueryTruthCache(
        cache_dir=os.path.join(output, "query-truth-cache"), enabled=True
    )
    truth = cache.get_or_build(
        queries,
        retained,
        cfg=config,
        corpus_identity=reference["key_parts"]["data"],
        query_identity={
            "query": expected_input_signature(MINILM_QUERIES),
            "provenance": expected_input_signature(MINILM_QUERY_PROVENANCE),
            "excluded_from_training_blocks": True,
        },
        k=15,
    )
    projection = score_query_bundle(
        X=retained,
        Z=coordinates,
        Xq=queries,
        Zq=query_coordinates,
        cfg=config,
        truth_cache=cache,
        label=MAP_LABEL,
        random_seed=123,
    )
    untrained = _score_untrained_floor(
        bundle=bundle,
        queries=queries,
        coordinates=coordinates,
        truth=truth["neighbors"],
        config=config,
    )
    guards = panel.get("guards") or {}
    purity = panel.get("purity") or {}
    checks = {
        "ffr_at_least_0_40": panel.get("ffr", -math.inf) >= 0.40,
        "density_at_least_0_60": panel.get("density", -math.inf) >= 0.60,
        "purity_k256_at_least_0_50": purity.get("k256", -math.inf) >= 0.50,
        "purity_k1024_at_least_0_50": purity.get("k1024", -math.inf) >= 0.50,
        "heldout_projection_beats_untrained_floor": (
            projection["proj_ffr"] > untrained["floor_ffr"]
        ),
        "coords_finite": guards.get("coords_finite") is True,
        "coords_not_collapsed": guards.get("coords_collapsed") is False,
        "embeddings_finite": guards.get("emb_finite") is True,
        "eligible_embeddings_nonzero": guards.get("emb_zero_rows") == 0,
    }
    report = {
        "schema": "round0036-registered-panel-v1",
        "round_id": "0036",
        "map": {
            "label": MAP_LABEL,
            "model": bundle["model"],
            "coordinate_receipt": expected_input_signature(
                os.path.join(job["transform_output"], "actual-transform.json")
            ),
        },
        "eligibility": eligibility["signature"],
        "scientific_universe": {
            "rows": len(retained),
            "row_namespace": "compact ascending R0033-retained global IDs",
            "excluded_rows_in_anchors": False,
            "excluded_rows_in_high_d_truth": False,
            "excluded_rows_in_2d_candidates": False,
            "excluded_rows_in_denominators": False,
        },
        "panel": panel,
        "recall_at_10": panel["recall@k"],
        "recall_at_50": recall50,
        "recall50_guard": guard50,
        "projection": projection,
        "untrained_projection_floor": untrained,
        "query_truth_cache": cache.telemetry(),
        "decision_checks": checks,
        "selector_passed": all(checks.values()),
        "low_ood_retention_is_non_gating_map_card_evidence": True,
        "wall_seconds": time.monotonic() - started,
    }
    receipt = seal(report)
    path = os.path.join(output, "panel.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "panel_receipt": expected_input_signature(path)}


def _configure_ood(job: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle = _bundle(job)
    coordinate_path = os.path.join(job["transform_output"], "actual-transform.json")
    coordinate_signature = expected_input_signature(coordinate_path)

    def loader() -> Any:
        return load_reviewed_model(bundle, device="cuda")

    common = {
        "round_id": "0036",
        "map_label": MAP_LABEL,
        "model_path": bundle["model"]["canonical_path"],
        "model_sha256": bundle["model"]["sha256"],
        "model_loader": loader,
    }
    universality = {
        **common,
        "coordinates_root": job["transform_output"],
        "coordinate_receipt_sha256": coordinate_signature["sha256"],
    }
    common_corpus = {
        **common,
        "coordinate_receipt": coordinate_path,
        "coordinate_receipt_sha256": coordinate_signature["sha256"],
    }
    return universality, common_corpus


def run_ood_panels(_active: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    output = create_fresh_directory(job["outputs"][0], label="R0036 OOD output")
    universality_root = os.path.join(output, "universality")
    common_root = os.path.join(output, "common-corpus")
    universality_config, common_config = _configure_ood(job)

    from experiments import universality_panel

    universality_panel.configure_map(**universality_config)
    ucanary = universality_panel.run_canary(
        output_root=os.path.join(universality_root, "canary")
    )
    upanel = universality_panel.run_panel(
        canary_path=ucanary["verdict"]["canonical_path"],
        output_root=os.path.join(universality_root, "panel"),
    )

    from experiments import common_corpus_ood_round0035 as common

    common.configure_map(**common_config)
    ccanary = common.run_canary(output_root=os.path.join(common_root, "canary"))
    cpanel = common.run_panel(
        canary_path=ccanary["verdict"]["canonical_path"],
        output_root=os.path.join(common_root, "panel"),
    )
    body = {
        "schema": "round0036-ood-bundle-v1",
        "round_id": "0036",
        "map_label": MAP_LABEL,
        "universality_canary": ucanary["verdict"],
        "universality_panel": upanel["panel"],
        "common_corpus_canary": ccanary["verdict"],
        "common_corpus_panel": cpanel["panel"],
        "probe_names": ["dadabase", "trec-covid", "code", "science", "latin"],
        "retention_is_map_card_evidence_not_same_domain_gate": True,
    }
    receipt = seal(body)
    path = os.path.join(output, "ood-bundle.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_semantic_render(
    _active: dict[str, Any], job: dict[str, Any]
) -> dict[str, Any]:
    output = create_fresh_directory(job["outputs"][0], label="R0036 render output")
    selector, eligibility = load_released_selector(
        job["eligibility_path"], eligibility_sha256=job["eligibility_sha256"]
    )
    full = CoordinateStream(job["transform_output"])
    retained = RetainedArrayView(full, selector)
    rng = np.random.RandomState(20260722)
    compact = np.sort(rng.choice(len(retained), 50_000, replace=False)).astype(np.int64)
    global_rows = selector.compact_to_global(compact)
    points = retained[compact]
    if not np.isfinite(points).all() or np.any(np.std(points, axis=0) <= 1e-8):
        raise Round0036Error("R0036 representative render is invalid/collapsed")
    ids_path = os.path.join(output, "sample-semantic-ids.npy")
    atomic_save_new_npy(ids_path, global_rows, immutable=True)
    image_path = os.path.join(output, "seed42-map.png")

    def draw(path: str) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(10, 10))
        axis.scatter(
            points[:, 0], points[:, 1], s=0.15, alpha=0.35,
            linewidths=0, rasterized=True,
        )
        axis.set_aspect("equal", adjustable="box")
        axis.set_title("R0034/R0036 seed-42 150M MiniLM map (retained representatives)")
        axis.set_xticks([])
        axis.set_yticks([])
        figure.tight_layout()
        figure.savefig(path, format="png", dpi=180, bbox_inches="tight")
        plt.close(figure)

    atomic_build_new_file(image_path, draw, immutable=True)
    body = {
        "schema": "round0036-semantic-render-v1",
        "round_id": "0036",
        "map_label": MAP_LABEL,
        "eligibility": eligibility["signature"],
        "coordinate_stream": expected_input_signature(
            os.path.join(job["transform_output"], "actual-transform.json")
        ),
        "panel": expected_input_signature(
            os.path.join(job["panel_output"], "panel.json")
        ),
        "sample_seed": 20260722,
        "sample_size": 50_000,
        "sample_semantic_ids": expected_input_signature(ids_path),
        "sample_semantic_ids_sha256": ordered_array_sha256(global_rows),
        "sample_ids_are_global_R0025_rows": True,
        "sample_contains_only_R0033_retained_representatives": bool(
            np.all(selector.is_retained(global_rows))
        ),
        "image": expected_input_signature(image_path),
        "diagnostics": {
            "finite_fraction": 1.0,
            "axis_std": points.std(axis=0).astype(float).tolist(),
            "axis_span": np.ptp(points, axis=0).astype(float).tolist(),
            "collapsed": False,
        },
    }
    receipt = seal(body)
    path = os.path.join(output, "render-manifest.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def _required_projection_page_signatures(
    entries: list[dict[str, Any]],
    *,
    site_dir: Path,
    required_probes: set[str],
) -> dict[str, dict[str, Any]]:
    """Prove publish produced a nonempty explorer for every registered probe."""
    by_probe = {
        item.get("projection", {}).get("probe"): item
        for item in entries
        if item.get("kind") == "projection-map"
    }
    signatures: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for probe in sorted(required_probes):
        entry = by_probe.get(probe)
        if not isinstance(entry, dict) or not entry.get("map_id"):
            missing.append(f"{probe}:registry-entry")
            continue
        path = site_dir / "projections" / str(entry["map_id"]) / "index.html"
        if not path.is_file() or path.stat().st_size <= 0:
            missing.append(f"{probe}:{path}")
            continue
        signatures[probe] = expected_input_signature(str(path))
    if missing:
        raise Round0036Error(
            "R0036 projection explorers were not published: " + ", ".join(missing)
        )
    return signatures


def run_registry_publication(
    _active: dict[str, Any], job: dict[str, Any]
) -> dict[str, Any]:
    output = create_fresh_directory(job["outputs"][0], label="R0036 registry receipt")
    from experiments import map_registry

    registry = map_registry.scan()
    entries = [
        item for item in registry["maps"] if item.get("round_id") == "0036"
    ]
    kinds = {item.get("kind") for item in entries}
    probes = {
        item.get("projection", {}).get("probe")
        for item in entries
        if item.get("kind") == "projection-map"
    }
    required_probes = {"dadabase", "trec-covid", "code", "science", "latin"}
    if "round-map" not in kinds or not required_probes.issubset(probes):
        raise Round0036Error(
            f"map registry cannot discover R0036 map/projections: kinds={kinds} probes={probes}"
        )
    map_registry.REGISTRY_PATH.write_text(json.dumps(registry, indent=1))
    map_registry.publish(registry)
    explorer_pages = _required_projection_page_signatures(
        entries,
        site_dir=map_registry.SITE_DIR,
        required_probes=required_probes,
    )
    body = {
        "schema": "round0036-map-registry-publication-v1",
        "round_id": "0036",
        "registry": expected_input_signature(str(map_registry.REGISTRY_PATH)),
        "map_ids": sorted(item["map_id"] for item in entries),
        "projection_probes": sorted(probes),
        "projection_explorer_pages": explorer_pages,
        "local_site_url": map_registry.SITE_URL,
    }
    receipt = seal(body)
    path = os.path.join(output, "registry-publication.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_job(active: dict[str, Any], job: dict[str, Any] | None = None) -> Any:
    if active.get("manifest", {}).get("round_id") != "0036":
        raise RuntimeError("R0036 handler received another queue")
    if job is None:
        raise RuntimeError("R0036 slim handler requires the exact job object")
    handlers = {
        "production_canary": run_production_canary,
        "transform_150m": run_transform,
        "high_d_reference": run_high_d_reference,
        "registered_panel": run_registered_panel,
        "ood_panels": run_ood_panels,
        "semantic_render": run_semantic_render,
        "registry_publication": run_registry_publication,
    }
    try:
        handler = handlers[job["id"]]
    except KeyError as exc:
        raise RuntimeError(f"unknown R0036 job {job.get('id')!r}") from exc
    return handler(active, job)
