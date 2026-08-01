"""Pure contracts for current-recipe Jina density-v3 calibration."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


ROUND_ID = "0136"
CALIBRATION_SCHEMA = "round0136-jina-density-v3-calibration-v1"
REPLAY_SCHEMA = "round0136-jina-density-v3-three-seed-replay-v1"
DECISION_SCHEMA = "round0136-jina-density-v3-decision-v1"

CALIBRATION_CELLS = (
    "r0104_fp16_seed42",
    "r0115_raw_seed42",
    "r0117_raw_seed43",
    "r0107_25m_seed42",
    "r0109_25m_seed43",
    "r0111_25m_seed44",
)
REPLAY_CELLS = (
    "r0107_25m_seed42",
    "r0109_25m_seed43",
    "r0111_25m_seed44",
)
SOURCE_CELL_KEYS = {
    "r0104_fp16_seed42": "r0104_fp16_seed42_full_transform",
    "r0115_raw_seed42": "current_2m_seed42",
    "r0117_raw_seed43": "current_2m_seed43",
    "r0107_25m_seed42": "current_25m_seed42",
    "r0109_25m_seed43": "current_25m_seed43",
    "r0111_25m_seed44": "seed44",
}

FLOOR_SIGMA_MULTIPLIER = 3.0
REPLAY_ABSOLUTE_TOLERANCE = 1.0e-6
DECISION_TOLERANCE = 1.0e-12

OUTCOME_QUALIFIED = "current-recipe-three-seed-density-qualified"
OUTCOME_REPLAY_DRIFT = "density-v3-replay-drift"
OUTCOME_FLOOR_NOT_REGISTERED = "density-v3-floor-not-registered"
OUTCOME_SEED_FAILURE = "current-recipe-seed-below-density-v3-floor"

DENSITY_CAPABILITY = "jina-density-v3-current-recipe-calibration-v1"
ATLAS_CAPABILITY = "jina-diverse-25m-atlas-quality-v1"


class Round0136Error(RuntimeError):
    """The preregistered R0136 density-v3 contract was violated."""


def _metric(cell: Mapping[str, Any], *path: str) -> float:
    value: Any = cell
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise Round0136Error(f"density cell is missing {'.'.join(path)}")
        value = value[key]
    result = float(value)
    if not math.isfinite(result):
        raise Round0136Error(f"density cell has nonfinite {'.'.join(path)}")
    return result


def calibrate_floor(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the R0085 min-minus-three-max-SD rule to the frozen family."""
    if tuple(cells) != CALIBRATION_CELLS:
        raise Round0136Error("density-v3 calibration cell order changed")
    points = {
        key: _metric(cells[key], "density_v2", "correlation")
        for key in CALIBRATION_CELLS
    }
    bootstrap_sd = {
        key: _metric(
            cells[key], "density_v2", "bootstrap", "standard_deviation"
        )
        for key in CALIBRATION_CELLS
    }
    null_bounds = {
        key: _metric(
            cells[key],
            "density_v2",
            "permuted_radius_null",
            "absolute_99_9_percentile",
        )
        for key in CALIBRATION_CELLS
    }
    if (
        any(value < -1.0 or value > 1.0 for value in points.values())
        or any(value <= 0.0 for value in bootstrap_sd.values())
        or any(value < 0.0 or value > 1.0 for value in null_bounds.values())
    ):
        raise Round0136Error("density-v3 calibration metric domain changed")

    minimum_key = min(CALIBRATION_CELLS, key=lambda key: points[key])
    maximum_sd_key = max(CALIBRATION_CELLS, key=lambda key: bootstrap_sd[key])
    maximum_null_key = max(CALIBRATION_CELLS, key=lambda key: null_bounds[key])
    proposed = (
        points[minimum_key]
        - FLOOR_SIGMA_MULTIPLIER * bootstrap_sd[maximum_sd_key]
    )
    maximum_null = null_bounds[maximum_null_key]
    registered = proposed > 0.0 and proposed > maximum_null
    return {
        "rule": (
            "minimum current-recipe 2M/25M matched-FineWeb density-v2 "
            "minus three times the maximum current-family bootstrap SD"
        ),
        "sigma_multiplier": FLOOR_SIGMA_MULTIPLIER,
        "minimum_density_cell": minimum_key,
        "minimum_density_v2": points[minimum_key],
        "maximum_bootstrap_sd_cell": maximum_sd_key,
        "maximum_bootstrap_standard_deviation": bootstrap_sd[maximum_sd_key],
        "maximum_null_cell": maximum_null_key,
        "maximum_absolute_null_99_9_percentile": maximum_null,
        "proposed_floor": proposed,
        "positive": proposed > 0.0,
        "separated_from_permuted_radius_null": proposed > maximum_null,
        "gating_floor_registered": registered,
        "registered_floor": proposed if registered else None,
        "calibration_points": points,
        "bootstrap_standard_deviations": bootstrap_sd,
        "absolute_null_99_9_percentiles": null_bounds,
    }


def decide_replay(
    calibration: Mapping[str, Any],
    replay_cells: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Reproduce the three 25M cells and apply the frozen density-v3 floor."""
    if tuple(replay_cells) != REPLAY_CELLS:
        raise Round0136Error("density-v3 replay cell order changed")
    registered = calibration.get("gating_floor_registered") is True
    floor_value = calibration.get("registered_floor")
    if registered != isinstance(floor_value, (int, float)):
        raise Round0136Error("density-v3 floor registration is malformed")
    floor = float(floor_value) if registered else math.nan
    calibration_points = calibration.get("calibration_points")
    if not isinstance(calibration_points, Mapping):
        raise Round0136Error("density-v3 calibration points are missing")

    cells: dict[str, Any] = {}
    every_reproduces = True
    every_clears = registered
    for key in REPLAY_CELLS:
        observed = _metric(replay_cells[key], "density_v2", "correlation")
        expected = float(calibration_points.get(key, math.nan))
        if not math.isfinite(expected):
            raise Round0136Error(f"calibration point is missing for {key}")
        delta = observed - expected
        reproduces = abs(delta) <= REPLAY_ABSOLUTE_TOLERANCE
        clears = registered and observed + DECISION_TOLERANCE >= floor
        cells[key] = {
            "calibration_density_v2": expected,
            "fresh_replay_density_v2": observed,
            "replay_delta": delta,
            "reproduces_within_absolute_tolerance": reproduces,
            "replay_absolute_tolerance": REPLAY_ABSOLUTE_TOLERANCE,
            "clears_density_v3_floor": clears,
        }
        every_reproduces = every_reproduces and reproduces
        every_clears = every_clears and clears

    if not registered:
        outcome = OUTCOME_FLOOR_NOT_REGISTERED
    elif not every_reproduces:
        outcome = OUTCOME_REPLAY_DRIFT
    elif not every_clears:
        outcome = OUTCOME_SEED_FAILURE
    else:
        outcome = OUTCOME_QUALIFIED
    return {
        "outcome": outcome,
        "registered_floor": floor_value,
        "cells": cells,
        "all_three_reproduce_calibration": every_reproduces,
        "all_three_clear_density_v3_floor": every_clears,
        "density_capability_released": registered and every_reproduces,
        "atlas_quality_capability_released": outcome == OUTCOME_QUALIFIED,
    }
