from __future__ import annotations

import math

import pytest

from basemap.round0136_density_v3 import (
    CALIBRATION_CELLS,
    OUTCOME_FLOOR_NOT_REGISTERED,
    OUTCOME_QUALIFIED,
    OUTCOME_REPLAY_DRIFT,
    REPLAY_CELLS,
    Round0136Error,
    calibrate_floor,
    decide_replay,
)


def _cell(point: float, sd: float = 0.01, null: float = 0.03):
    return {
        "density_v2": {
            "correlation": point,
            "bootstrap": {"standard_deviation": sd},
            "permuted_radius_null": {"absolute_99_9_percentile": null},
        }
    }


def test_floor_uses_min_minus_three_max_sd_and_null_guard() -> None:
    cells = {
        key: _cell(0.20 - index * 0.01, sd=0.008 + index * 0.001)
        for index, key in enumerate(CALIBRATION_CELLS)
    }
    floor = calibrate_floor(cells)
    assert floor["minimum_density_cell"] == CALIBRATION_CELLS[-1]
    assert floor["maximum_bootstrap_sd_cell"] == CALIBRATION_CELLS[-1]
    assert floor["registered_floor"] == pytest.approx(0.15 - 3 * 0.013)
    assert floor["gating_floor_registered"] is True


def test_floor_refuses_to_register_below_null() -> None:
    cells = {key: _cell(0.05, sd=0.01, null=0.03) for key in CALIBRATION_CELLS}
    floor = calibrate_floor(cells)
    assert floor["proposed_floor"] == pytest.approx(0.02)
    assert floor["gating_floor_registered"] is False
    replay = {key: _cell(0.05) for key in REPLAY_CELLS}
    assert decide_replay(floor, replay)["outcome"] == OUTCOME_FLOOR_NOT_REGISTERED


def test_three_seed_replay_must_reproduce_before_qualification() -> None:
    points = {
        key: 0.11 + index * 0.01 for index, key in enumerate(CALIBRATION_CELLS)
    }
    calibration = {
        "gating_floor_registered": True,
        "registered_floor": 0.08,
        "calibration_points": points,
    }
    replay = {
        key: _cell(points[key])
        for key in REPLAY_CELLS
    }
    accepted = decide_replay(calibration, replay)
    assert accepted["outcome"] == OUTCOME_QUALIFIED
    assert accepted["atlas_quality_capability_released"] is True

    replay[REPLAY_CELLS[1]] = _cell(points[REPLAY_CELLS[1]] + 2.0e-6)
    drift = decide_replay(calibration, replay)
    assert drift["outcome"] == OUTCOME_REPLAY_DRIFT
    assert drift["atlas_quality_capability_released"] is False


def test_cell_order_is_part_of_contract() -> None:
    cells = {key: _cell(0.2) for key in reversed(CALIBRATION_CELLS)}
    with pytest.raises(Round0136Error, match="cell order"):
        calibrate_floor(cells)


def test_nonfinite_metric_rejected() -> None:
    cells = {key: _cell(0.2) for key in CALIBRATION_CELLS}
    cells[CALIBRATION_CELLS[0]] = _cell(math.nan)
    with pytest.raises(Round0136Error, match="nonfinite"):
        calibrate_floor(cells)
