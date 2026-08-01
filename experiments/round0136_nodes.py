"""Calibrate density-v3 and freshly replay three current-recipe 25M maps."""
from __future__ import annotations

import gc
import json
import os
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import (
    expected_input_signature,
    ordered_array_sha256,
)
from basemap.output_safety import (
    atomic_save_new_npz,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0108_evaluation import seal, validate_seal
from basemap.round0136_density_v3 import (
    ATLAS_CAPABILITY,
    CALIBRATION_CELLS,
    CALIBRATION_SCHEMA,
    DECISION_SCHEMA,
    DENSITY_CAPABILITY,
    REPLAY_CELLS,
    REPLAY_SCHEMA,
    ROUND_ID,
    SOURCE_CELL_KEYS,
    Round0136Error,
    calibrate_floor,
    decide_replay,
)
from experiments.round0085_nodes import density_v2_calibration
from experiments.round0119_nodes import (
    _authenticate_model,
    _load_universe,
    _score_cell,
)


SOURCE_SPECS = {
    "r0104_fp16_seed42": ("r0122", "new_cells"),
    "r0115_raw_seed42": ("r0119", "cells"),
    "r0117_raw_seed43": ("r0119", "cells"),
    "r0107_25m_seed42": ("r0119", "cells"),
    "r0109_25m_seed43": ("r0119", "cells"),
    "r0111_25m_seed44": ("r0118", "cells"),
}


def _exact_signature(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    actual = expected_input_signature(str(value.get("canonical_path") or ""))
    if actual != dict(value):
        raise Round0136Error(f"{label} bytes changed")
    return actual


def _read_sealed(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    signature = _exact_signature(value, label=label)
    with open(signature["canonical_path"], encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise Round0136Error(f"{label} is not a JSON object")
    validate_seal(document, label=label)
    return document


def _summary_close(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    paths = (
        ("correlation",),
        ("bootstrap", "standard_deviation"),
        ("bootstrap", "central_99_percent", 0),
        ("bootstrap", "central_99_percent", 1),
        ("permuted_radius_null", "mean"),
        ("permuted_radius_null", "standard_deviation"),
        ("permuted_radius_null", "absolute_99_9_percentile"),
    )

    def get(value: Any, path: tuple[Any, ...]) -> float:
        for key in path:
            value = value[key]
        return float(value)

    return all(abs(get(left, path) - get(right, path)) <= 1.0e-12 for path in paths)


def run_calibration(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any]:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0136 density-v3 calibration"
    )
    started = time.monotonic()
    sources = job.get("calibration_sources")
    if not isinstance(sources, Mapping) or set(sources) != {"r0122", "r0119", "r0118"}:
        raise Round0136Error("density-v3 source bundle changed")

    receipts: dict[str, dict[str, Any]] = {}
    archives: dict[str, dict[str, np.ndarray]] = {}
    lineages: dict[str, Any] = {}
    for source_key in ("r0122", "r0119", "r0118"):
        source = sources[source_key]
        receipt = _read_sealed(source["receipt"], label=f"{source_key} density receipt")
        arrays_signature = _exact_signature(
            source["arrays"], label=f"{source_key} density arrays"
        )
        if receipt.get("arrays") != arrays_signature:
            raise Round0136Error(f"{source_key} density receipt/arrays binding changed")
        with np.load(arrays_signature["canonical_path"], allow_pickle=False) as archive:
            archives[source_key] = {
                key: np.asarray(archive[key]) for key in archive.files
            }
        receipts[source_key] = receipt
        lineages[source_key] = {
            "receipt": dict(source["receipt"]),
            "arrays": arrays_signature,
        }

    reference_anchors = archives["r0119"].get("anchor_compact_rows")
    reference_global = archives["r0119"].get("anchor_global_rows")
    reference_high = archives["r0119"].get("high_radius")
    if (
        reference_anchors is None
        or reference_global is None
        or reference_high is None
        or reference_anchors.shape != (10_000,)
        or reference_global.shape != reference_anchors.shape
        or reference_high.shape != reference_anchors.shape
    ):
        raise Round0136Error("matched-FineWeb density reference changed")
    for source_key in ("r0122", "r0118"):
        for name, expected in (
            ("anchor_compact_rows", reference_anchors),
            ("anchor_global_rows", reference_global),
            ("high_radius", reference_high),
        ):
            if not np.array_equal(archives[source_key].get(name), expected):
                raise Round0136Error(
                    f"{source_key} does not use the shared density reference"
                )

    cells: dict[str, Any] = {}
    consolidated: dict[str, np.ndarray] = {
        "anchor_compact_rows": reference_anchors,
        "anchor_global_rows": reference_global,
        "high_radius": reference_high,
    }
    for key in CALIBRATION_CELLS:
        source_key, section = SOURCE_SPECS[key]
        source_cell_key = SOURCE_CELL_KEYS[key]
        receipt_cells = receipts[source_key].get(section)
        if not isinstance(receipt_cells, Mapping) or source_cell_key not in receipt_cells:
            raise Round0136Error(f"source density cell missing for {key}")
        source_cell = receipt_cells[source_cell_key]
        if not isinstance(source_cell, Mapping):
            raise Round0136Error(f"source density cell malformed for {key}")
        arrays = archives[source_key]
        low = np.asarray(arrays[f"{source_cell_key}__low_radius"], dtype=np.float64)
        bootstrap = np.asarray(arrays[f"{source_cell_key}__bootstrap"], dtype=np.float64)
        null = np.asarray(arrays[f"{source_cell_key}__permuted_null"], dtype=np.float64)
        regenerated, regenerated_bootstrap, regenerated_null = density_v2_calibration(
            reference_high,
            low,
            bootstrap_draws=1_000,
            bootstrap_seed=10_801,
            null_draws=1_000,
            null_seed=10_802,
        )
        if (
            not np.array_equal(regenerated_bootstrap, bootstrap)
            or not np.array_equal(regenerated_null, null)
            or not _summary_close(regenerated, source_cell["density_v2"])
        ):
            raise Round0136Error(f"frozen density arrays do not replay for {key}")
        cells[key] = {
            "key": key,
            "source_round": source_key.removeprefix("r"),
            "source_cell": source_cell_key,
            "density_v2": regenerated,
            "source_model": source_cell.get("model"),
            "source_train_receipt": source_cell.get("train_receipt"),
            "source_production_config": source_cell.get("production_config"),
        }
        consolidated[f"{key}__low_radius"] = low
        consolidated[f"{key}__bootstrap"] = bootstrap
        consolidated[f"{key}__permuted_null"] = null

    floor = calibrate_floor(cells)
    arrays_path = os.path.join(output, "density-v3-calibration-arrays.npz")
    atomic_save_new_npz(arrays_path, immutable=True, **consolidated)
    receipt = seal({
        "schema": CALIBRATION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "lineage": lineages,
        "universe": {
            "name": "R0040 exact FineWeb representative matched universe",
            "representative_rows": 1_996_279,
            "anchors": 10_000,
            "family_size_cutoff_exclusive": 16,
            "anchor_compact_rows_sha256": ordered_array_sha256(reference_anchors),
            "anchor_global_rows_sha256": ordered_array_sha256(reference_global),
            "high_radius_sha256": ordered_array_sha256(reference_high),
        },
        "cells": cells,
        "floor_calibration": floor,
        "arrays": expected_input_signature(arrays_path),
        "scorer_replayed_from_sealed_radii": True,
        "training_performed": False,
        "wall_seconds": time.monotonic() - started,
    })
    path = os.path.join(output, "density-v3-calibration.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_replay(active: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0136 three-seed density replay"
    )
    started = time.monotonic()
    (
        source,
        representatives,
        retained_global_rows,
        anchors,
        global_rows,
        high_radius,
        lineage,
        reference,
    ) = _load_universe(job)
    specs = job.get("model_bundles")
    if not isinstance(specs, list) or tuple(spec["key"] for spec in specs) != REPLAY_CELLS:
        raise Round0136Error("three-seed replay model order changed")
    cells: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {
        "anchor_compact_rows": anchors,
        "anchor_global_rows": global_rows,
        "high_radius": high_radius,
    }
    for spec in specs:
        key = str(spec["key"])
        bundle = _authenticate_model(spec)
        cell, cell_arrays = _score_cell(
            key=key,
            bundle=bundle,
            source=source,
            representatives=representatives,
            retained_global_rows=retained_global_rows,
            anchors=anchors,
            high_radius=high_radius,
            reference=reference,
        )
        cells[key] = cell
        arrays.update(cell_arrays)
        del bundle["model"]
        gc.collect()
    arrays_path = os.path.join(output, "density-v3-replay-arrays.npz")
    atomic_save_new_npz(arrays_path, immutable=True, **arrays)
    receipt = seal({
        "schema": REPLAY_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "lineage": lineage,
        "cells": cells,
        "arrays": expected_input_signature(arrays_path),
        "fresh_model_transforms": True,
        "fresh_exact_low_dim_search": True,
        "training_performed": False,
        "wall_seconds": time.monotonic() - started,
    })
    path = os.path.join(output, "density-v3-three-seed-replay.json")
    atomic_write_new_json(path, receipt, immutable=True)
    del source, representatives, retained_global_rows
    gc.collect()
    return {**receipt, "receipt": expected_input_signature(path)}


def _quality_guards(job: Mapping[str, Any]) -> dict[str, Any]:
    r0110 = _read_sealed(job["r0110_decision"], label="R0110 two-seed decision")
    r0118 = _read_sealed(job["r0118_decision"], label="R0118 three-seed decision")
    checks10 = r0110.get("checks")
    checks18 = r0118.get("checks")
    if not isinstance(checks10, Mapping) or not isinstance(checks18, Mapping):
        raise Round0136Error("accepted non-density quality decisions changed")
    checks = {
        "seed42_native_non_density_core_passed": checks10.get(
            "seed42_native_non_density_core_passed"
        ) is True,
        "seed42_fixed_polish_ood_gate_passed": checks10.get(
            "seed42_fixed_polish_ood_gate_passed"
        ) is True,
        "seed43_native_non_density_core_passed": checks10.get(
            "seed43_native_non_density_core_passed"
        ) is True,
        "seed43_fixed_polish_ood_gate_passed": checks10.get(
            "seed43_fixed_polish_ood_gate_passed"
        ) is True,
        "seed44_native_non_density_core_passed": checks18.get(
            "seed44_native_non_density_core_passed"
        ) is True,
        "seed44_fixed_polish_ood_gate_passed": checks18.get(
            "seed44_fixed_polish_ood_gate_passed"
        ) is True,
    }
    if not all(checks.values()):
        raise Round0136Error("a frozen non-density atlas-quality guard is not positive")
    return checks


def run_decision(active: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0136 density-v3 decision"
    )
    calibration_path = os.path.join(
        str(job["calibration_output"]), "density-v3-calibration.json"
    )
    replay_path = os.path.join(
        str(job["replay_output"]), "density-v3-three-seed-replay.json"
    )
    calibration = _read_sealed(
        expected_input_signature(calibration_path), label="R0136 calibration"
    )
    replay = _read_sealed(expected_input_signature(replay_path), label="R0136 replay")
    if (
        calibration.get("schema") != CALIBRATION_SCHEMA
        or replay.get("schema") != REPLAY_SCHEMA
        or calibration.get("round_id") != ROUND_ID
        or replay.get("round_id") != ROUND_ID
    ):
        raise Round0136Error("R0136 calibration/replay schema changed")
    quality = _quality_guards(job)
    decision = decide_replay(calibration["floor_calibration"], replay["cells"])
    releases = []
    if decision["density_capability_released"]:
        releases.append(DENSITY_CAPABILITY)
    if decision["atlas_quality_capability_released"]:
        releases.append(ATLAS_CAPABILITY)
    receipt = seal({
        "schema": DECISION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "calibration": expected_input_signature(calibration_path),
        "replay": expected_input_signature(replay_path),
        "frozen_non_density_quality_checks": quality,
        **decision,
        "capabilities_ready_for_review": releases,
        "registry_promotion_authorized_after_review": (
            ATLAS_CAPABILITY in releases
        ),
        "training_performed": False,
        "map_registry_state_changed": False,
        "prompt_or_production_transfer_claimed": False,
    })
    path = os.path.join(output, "density-v3-decision.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_job(
    active: dict[str, Any], job: dict[str, Any] | None = None
) -> dict[str, Any]:
    if job is None:
        job = active["job"]
    if (active.get("manifest") or {}).get("round_id") != ROUND_ID:
        raise Round0136Error("R0136 handler received another round")
    action = str(job.get("action"))
    if action == "calibrate_density_v3":
        return run_calibration(active, job)
    if action == "replay_three_seed_density_v3":
        return run_replay(active, job)
    if action == "decide_density_v3":
        return run_decision(active, job)
    raise Round0136Error(f"unknown R0136 action: {action}")
