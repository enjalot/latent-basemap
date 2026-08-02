"""Execute the CPU-only Track-A density-v2 forensics in Round 0153."""
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
from basemap.round0040_program import (
    RepresentativeRowSelector,
    load_jina_census,
)
from basemap.round0108_evaluation import map_family_sizes, seal, validate_seal
from basemap.round0153_density_forensics import (
    ANCHORS,
    CAPABILITY,
    CURRENT_POPULATION_REFERENCES,
    HISTORICAL_ROW_CELLS,
    REGISTERED_FLOOR,
    REPRESENTATIVE_ROWS,
    ROUND_ID,
    ROWS,
    Round0153Error,
    classify_density_branch,
    density_v2_from_radii,
    diagnostic_values,
    exact_low_radius_cpu,
)


OUTPUT_SCHEMA = "round0153-track-a-density-forensics-v1"
CALIBRATION_SCHEMA = "round0108-jina-density-v2-calibration-v1"
REFERENCE_SCHEMA = "hiD_reference.v3"


def _signature(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    actual = expected_input_signature(str(expected.get("canonical_path") or ""))
    if actual != dict(expected):
        raise Round0153Error(f"{label} bytes changed")
    return actual


def _read_json(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    signature = _signature(expected, label=label)
    with open(signature["canonical_path"], encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0153Error(f"{label} is not a JSON object")
    return value


def _load_frozen_universe(
    job: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    calibration = _read_json(job["r0108_calibration"], label="R0108 calibration")
    validate_seal(calibration, label="R0108 calibration")
    floor = calibration.get("floor_calibration") or {}
    if (
        calibration.get("schema") != CALIBRATION_SCHEMA
        or calibration.get("round_id") != "0108"
        or floor.get("registered_floor") != REGISTERED_FLOOR
        or floor.get("gating_floor_registered") is not True
        or calibration.get("threshold_tuned_after_treatment") is not False
    ):
        raise Round0153Error("frozen R0108 calibration semantics changed")

    census_receipt = _read_json(
        calibration["census_receipt"], label="R0040 census receipt"
    )
    census = load_jina_census(
        str(calibration["census_receipt"]["canonical_path"])
    )
    if (
        census_receipt.get("census") != calibration.get("census")
        or census["signature"] != calibration.get("census")
    ):
        raise Round0153Error("R0040 census lineage changed")
    selector = RepresentativeRowSelector(
        census["arrays"]["excluded_rows"],
        row_count=ROWS,
        source=census["signature"],
        policy="R0040 exact nonzero fp16 family; minimum row representative",
    )
    retained = selector.compact_to_global(
        np.arange(selector.retained_count, dtype=np.int64)
    )

    reference_signature = _signature(
        calibration["representative_reference"],
        label="R0040 representative high-D reference",
    )
    with np.load(reference_signature["canonical_path"], allow_pickle=False) as archive:
        schema = str(np.asarray(archive["schema"]).item())
        key = str(np.asarray(archive["key"]).item())
        anchors = np.asarray(archive["anchor_ids"], dtype=np.int64)
        high_radius = np.asarray(archive["r_hd"], dtype=np.float64)
    anchor_contract = calibration.get("anchors") or {}
    global_rows = selector.compact_to_global(anchors)
    family_sizes = map_family_sizes(
        global_rows,
        census["arrays"]["representative_rows"],
        census["arrays"]["family_counts"],
    )
    if (
        schema != REFERENCE_SCHEMA
        or key != calibration.get("representative_reference_key")
        or selector.retained_count != REPRESENTATIVE_ROWS
        or retained.shape != (REPRESENTATIVE_ROWS,)
        or anchors.shape != (ANCHORS,)
        or high_radius.shape != (ANCHORS,)
        or ordered_array_sha256(anchors) != anchor_contract.get("compact_rows_sha256")
        or ordered_array_sha256(global_rows) != anchor_contract.get("global_rows_sha256")
        or ordered_array_sha256(family_sizes) != anchor_contract.get("family_sizes_sha256")
        or not np.all(family_sizes < 16)
    ):
        raise Round0153Error("R0040 anchor/reference universe changed")

    calibration_arrays_signature = _signature(
        calibration["arrays"], label="R0108 calibration arrays"
    )
    frozen: dict[str, np.ndarray] = {}
    with np.load(calibration_arrays_signature["canonical_path"], allow_pickle=False) as archive:
        for seed in ("seed42", "seed43"):
            for suffix in ("high_radius", "low_radius", "bootstrap", "permuted_null"):
                frozen[f"{seed}__{suffix}"] = np.asarray(archive[f"{seed}__{suffix}"])
            if not np.array_equal(frozen[f"{seed}__high_radius"], high_radius):
                raise Round0153Error("R0108 frozen high-D radii changed")
    lineage = {
        "r0108_calibration": dict(job["r0108_calibration"]),
        "r0108_calibration_arrays": calibration_arrays_signature,
        "census_receipt": dict(calibration["census_receipt"]),
        "census": dict(calibration["census"]),
        "representative_reference": reference_signature,
        "representative_reference_key": key,
        "registered_floor": REGISTERED_FLOOR,
    }
    return retained, anchors, high_radius, lineage, frozen


def _score_coordinates(
    signature: Mapping[str, Any],
    *,
    retained: np.ndarray,
    anchors: np.ndarray,
    high_radius: np.ndarray,
    workers: int,
    label: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    bound = _signature(signature, label=f"{label} coordinates")
    coordinates = np.load(
        bound["canonical_path"], mmap_mode="r", allow_pickle=False
    )
    low_radius, search = exact_low_radius_cpu(
        coordinates, retained, anchors, workers=workers
    )
    summary, bootstrap, null = density_v2_from_radii(high_radius, low_radius)
    return {
        "coordinates": bound,
        "coordinates_ordered_sha256": ordered_array_sha256(coordinates),
        "density_v2": summary,
        "clears_registered_floor": (
            float(summary["correlation"]) >= REGISTERED_FLOOR
        ),
        "exact_cpu_search": search,
    }, {
        "low_radius": low_radius,
        "bootstrap": bootstrap,
        "permuted_null": null,
    }


def run_forensics(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0153Error("R0153 handler received another queue")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise Round0153Error("R0153 must run with CUDA hidden")
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0153 density forensics"
    )
    started = time.monotonic()
    retained, anchors, high_radius, lineage, frozen = _load_frozen_universe(job)
    workers = int(job.get("cpu_workers", 4))

    # Authenticate the CPU scorer against both already-frozen controls before
    # allowing it to score any Track-A treatment.
    reproductions: dict[str, Any] = {}
    for seed, signature in job["historical_control_coordinates"].items():
        scored, arrays = _score_coordinates(
            signature,
            retained=retained,
            anchors=anchors,
            high_radius=high_radius,
            workers=workers,
            label=f"R0108 historical {seed}",
        )
        reference_low = np.asarray(frozen[f"{seed}__low_radius"], dtype=np.float64)
        reference_bootstrap = np.asarray(frozen[f"{seed}__bootstrap"])
        reference_null = np.asarray(frozen[f"{seed}__permuted_null"])
        maximum_delta = float(
            np.max(np.abs(arrays["low_radius"] - reference_low))
        )
        reproduction = {
            "coordinates": scored["coordinates"],
            "density_v2": scored["density_v2"],
            "maximum_low_radius_absolute_delta": maximum_delta,
            "low_radius_within_1e_6": bool(
                np.allclose(arrays["low_radius"], reference_low, rtol=1e-6, atol=1e-6)
            ),
            "bootstrap_within_1e_6": bool(
                np.allclose(arrays["bootstrap"], reference_bootstrap, rtol=1e-6, atol=1e-6)
            ),
            "permuted_null_within_1e_6": bool(
                np.allclose(arrays["permuted_null"], reference_null, rtol=1e-6, atol=1e-6)
            ),
        }
        reproduction["reproduces_frozen_control"] = all(
            reproduction[key]
            for key in (
                "low_radius_within_1e_6",
                "bootstrap_within_1e_6",
                "permuted_null_within_1e_6",
            )
        )
        if not reproduction["reproduces_frozen_control"]:
            raise Round0153Error(f"CPU scorer did not reproduce frozen {seed}")
        reproductions[seed] = reproduction

    cells: dict[str, Any] = {}
    output_arrays: dict[str, np.ndarray] = {
        "anchor_compact_rows": anchors,
        "anchor_global_rows": retained[anchors],
        "high_radius": high_radius,
    }
    for spec in job["cells"]:
        key = str(spec["key"])
        panel = _read_json(spec["panel"], label=f"{key} functional panel")
        validate_seal(panel, label=f"{key} functional panel")
        source_cell = panel.get("cells", {}).get(spec["panel_cell"])
        if (
            panel.get("round_id") != spec["source_round"]
            or not isinstance(source_cell, Mapping)
            or source_cell.get("coordinates") != spec["coordinates"]
        ):
            raise Round0153Error(f"{key} panel/coordinate binding changed")
        scored, arrays = _score_coordinates(
            spec["coordinates"],
            retained=retained,
            anchors=anchors,
            high_radius=high_radius,
            workers=workers,
            label=key,
        )
        cells[key] = {
            "source_round": spec["source_round"],
            "source_cell": spec["panel_cell"],
            "panel": dict(spec["panel"]),
            "role": spec["role"],
            "full_functional_diagnostics": diagnostic_values(source_cell),
            **scored,
        }
        for suffix, value in arrays.items():
            output_arrays[f"{key}__{suffix}"] = value

    r0119 = _read_json(job["r0119_density_panel"], label="R0119 density panel")
    validate_seal(r0119, label="R0119 density panel")
    if r0119.get("schema") != "round0119-jina-density-localization-panel-v1":
        raise Round0153Error("R0119 density reference semantics changed")
    current_references = {
        CURRENT_POPULATION_REFERENCES[0]: float(
            r0119["cells"]["current_2m_seed42"]["density_v2"]["correlation"]
        ),
        CURRENT_POPULATION_REFERENCES[1]: float(
            r0119["cells"]["current_2m_seed43"]["density_v2"]["correlation"]
        ),
    }
    historical_references = {
        "r0037_historical_seed42": float(
            r0119["cells"]["historical_2m_seed42"]["density_v2"]["correlation"]
        ),
        "r0038_historical_seed43": float(
            r0119["cells"]["historical_2m_seed43"]["density_v2"]["correlation"]
        ),
    }
    decision = classify_density_branch(
        {
            key: float(cells[key]["density_v2"]["correlation"])
            for key in HISTORICAL_ROW_CELLS
        },
        current_references,
    )
    arrays_path = os.path.join(output, "density-forensics-arrays.npz")
    atomic_save_new_npz(arrays_path, immutable=True, **output_arrays)
    receipt = seal({
        "schema": OUTPUT_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capability": CAPABILITY,
        "lineage": lineage,
        "scorer": {
            "metric": "Pearson correlation of log exact high-D mean-k15 radius with log exact low-D mean-k15 radius",
            "candidate_population": "R0040 retained exact-family representatives",
            "anchor_population": "R0040 frozen compact anchors; all original family sizes <16",
            "registered_floor": REGISTERED_FLOOR,
            "cpu_only": True,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "workers": workers,
        },
        "frozen_control_reproduction": reproductions,
        "historical_references": historical_references,
        "current_population_references": current_references,
        "cells": cells,
        "decision": decision,
        "arrays": expected_input_signature(arrays_path),
        "training_performed": False,
        "floor_changed": False,
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(
        os.path.join(output, "density-forensics.json"), receipt, immutable=True
    )


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if str(job.get("action") or "") != "track_a_density_forensics":
        raise Round0153Error("unknown R0153 action")
    run_forensics(active, job)
