"""Recover calibrated-style density evidence for accepted prompted maps."""
from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature, ordered_array_sha256
from basemap.output_safety import (
    atomic_save_new_npz,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.panel_v2 import (
    PanelV2Config,
    _self_knn,
    process_cuda_peak,
    reset_process_cuda_peak,
    sample_anchors,
)
from basemap.round0104_training import L2NormalizedArray
from basemap.round0108_evaluation import seal
from basemap.round0113_prompt_contrast import validate_seal as validate_prompt_seal
from basemap.round0157_prompted_density import (
    ANCHORS,
    ANCHOR_SEED,
    CAPABILITY,
    DIMENSION,
    K_DENSITY,
    RAW_UNIVERSE_CONTEXT_FLOOR,
    ROUND_ID,
    ROWS,
    Round0157Error,
    density_v2_from_radii,
    transcribe_native_prompted_score,
)


OUTPUT_SCHEMA = "round0157-native-prompted-density-v2-v1"


def _signature(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    actual = expected_input_signature(str(expected.get("canonical_path") or ""))
    if actual != dict(expected):
        raise Round0157Error(f"{label} bytes changed")
    return actual


def _read_json(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    signature = _signature(expected, label=label)
    with open(signature["canonical_path"], encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0157Error(f"{label} is not a JSON object")
    return value


def _config() -> PanelV2Config:
    return PanelV2Config(
        frac=0.001,
        k_clust=(),
        k_density=K_DENSITY,
        k_hit=10,
        n_anchors=ANCHORS,
        anchor_seed=ANCHOR_SEED,
        corpus_chunk=500_000,
        overselect=8,
        block_elems=500_000_000,
        rerank_byte_cap=2_000_000_000,
        rerank_scratch=3.0,
        peak_byte_cap=26_000_000_000,
    )


def _mean_exact_radius(
    values: Any,
    anchors: np.ndarray,
    *,
    high_dimensional: bool,
    config: PanelV2Config,
) -> tuple[np.ndarray, dict[str, Any]]:
    _indices, distances, guard = _self_knn(
        values,
        anchors,
        K_DENSITY,
        config,
        hi_dim=high_dimensional,
        want_dist=True,
        exact=True,
    )
    radius = np.asarray(distances.mean(axis=1), dtype=np.float64)
    if radius.shape != (ANCHORS,) or not np.isfinite(radius).all() or np.any(radius <= 0):
        raise Round0157Error("exact prompted radius computation failed")
    return radius, guard


def run_recovery(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0157Error("R0157 handler received another queue")
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {None, "", "-1"}:
        raise Round0157Error("R0157 exact high-dimensional scoring requires CUDA")
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0157 prompted density recovery"
    )
    started = time.monotonic()

    assembly = _read_json(job["assembly"], label="R0113 compact assembly")
    validate_prompt_seal(assembly, label="R0113 compact assembly")
    document_signature = _signature(
        job["document_compact"], label="R0113 document compact matrix"
    )
    if (
        assembly.get("schema") != "round0113-compact-prompt-arrays-v1"
        or int(assembly.get("retained_rows", -1)) != ROWS
        or int(assembly.get("dimension", -1)) != DIMENSION
        or assembly.get("outputs", {}).get("document") != document_signature
        or assembly.get("paired_row_population_identical") is not True
    ):
        raise Round0157Error("R0113 prompted population changed")

    source = np.memmap(
        document_signature["canonical_path"],
        dtype=np.dtype("<f2"),
        mode="r",
        shape=(ROWS, DIMENSION),
    )
    config = _config()
    anchors = sample_anchors(ROWS, config)
    expected_anchors = np.sort(
        np.random.RandomState(ANCHOR_SEED).choice(ROWS, ANCHORS, replace=False)
    ).astype(np.int64)
    if not np.array_equal(anchors, expected_anchors):
        raise Round0157Error("R0037-style prompted anchors changed")

    reset_process_cuda_peak()
    high_radius, high_guard = _mean_exact_radius(
        L2NormalizedArray(source),
        anchors,
        high_dimensional=True,
        config=config,
    )
    cells: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {
        "anchor_compact_rows": anchors,
        "prompted_high_radius": high_radius,
    }
    for cell in job["cells"]:
        seed = int(cell["seed"])
        coordinates_signature = _signature(
            cell["coordinates"], label=f"seed-{seed} prompted coordinates"
        )
        score = _read_json(cell["score"], label=f"seed-{seed} prompted score")
        validate_prompt_seal(score, label=f"seed-{seed} prompted score")
        transcription = transcribe_native_prompted_score(
            score,
            seed=seed,
            expected_coordinates=coordinates_signature,
        )
        coordinates = np.load(
            coordinates_signature["canonical_path"], mmap_mode="r", allow_pickle=False
        )
        if coordinates.shape != (ROWS, 2) or coordinates.dtype != np.float32:
            raise Round0157Error(f"seed-{seed} prompted coordinate geometry changed")
        low_radius, low_guard = _mean_exact_radius(
            coordinates,
            anchors,
            high_dimensional=False,
            config=config,
        )
        summary, bootstrap, null = density_v2_from_radii(high_radius, low_radius)
        key = f"seed{seed}"
        cells[key] = {
            "coordinates": coordinates_signature,
            "coordinates_ordered_sha256": ordered_array_sha256(coordinates),
            "accepted_score": dict(cell["score"]),
            "accepted_native_panel": transcription,
            "density_v2": summary,
            "low_dimensional_exact_search_guard": low_guard,
            "raw_universe_floor_context_only": {
                "floor": RAW_UNIVERSE_CONTEXT_FLOOR,
                "correlation_at_or_above": (
                    float(summary["correlation"]) >= RAW_UNIVERSE_CONTEXT_FLOOR
                ),
                "gating": False,
                "reason": (
                    "R0108 calibrated this floor on a different raw FineWeb "
                    "row and embedding universe"
                ),
            },
        }
        arrays[f"{key}__low_radius"] = low_radius
        arrays[f"{key}__bootstrap"] = bootstrap
        arrays[f"{key}__permuted_null"] = null

    arrays_path = os.path.join(output, "prompted-density-v2-arrays.npz")
    atomic_save_new_npz(arrays_path, immutable=True, **arrays)
    receipt = seal({
        "schema": OUTPUT_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capability": CAPABILITY,
        "lineage": {
            "assembly": dict(job["assembly"]),
            "document_compact": document_signature,
            "accepted_reviews": [dict(item) for item in job["accepted_reviews"]],
        },
        "population": {
            "rows": ROWS,
            "dimension": DIMENSION,
            "convention": "Document: ",
            "policy": "R0113 shared source/raw/document exact-family-union representatives",
        },
        "scorer": {
            "metric": (
                "Pearson correlation of log exact prompted high-D mean-k15 "
                "radius with log exact low-D mean-k15 radius"
            ),
            "anchor_rows": ANCHORS,
            "anchor_seed": ANCHOR_SEED,
            "anchor_compact_rows_sha256": ordered_array_sha256(anchors),
            "high_dimensional_exact_search_guard": high_guard,
            "bootstrap": {"draws": 1_000, "seed": 10_801},
            "permuted_null": {"draws": 1_000, "seed": 10_802},
        },
        "cells": cells,
        "arrays": expected_input_signature(arrays_path),
        "decision": {
            "outcome": "native-prompted-density-evidence-recovered",
            "quality_gate": None,
            "raw_floor_comparison_is_context_only": True,
            "prompt_noninferiority_verdict_changed": False,
            "floor_changed": False,
        },
        "cuda_peak": process_cuda_peak(),
        "training_performed": False,
        "graph_built": False,
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(
        os.path.join(output, "prompted-density-v2.json"), receipt, immutable=True
    )


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if str(job.get("action") or "") != "recover_native_prompted_density_v2":
        raise Round0157Error("unknown R0157 action")
    run_recovery(active, job)
