"""R0256 contract — the repaired independence control and the three corrections.

The test that matters most here is `test_a_builder_that_reads_a_held_out_cell_makes
_the_arms_report_false`. R0255 shipped an independence control with no positive
control at all, and review-0255 showed its arms could not fail. Every guard this
round touches plants its own defect and is verified to catch it.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from basemap import round0234_calibration as calibration
from basemap.round0234_calibrated_floors import (
    DESCRIPTIVE_METRICS,
    GATED_METRICS,
    METRICS,
    PURITY_METRICS,
    PURITY_RATIO_KEYS,
)
from basemap.round0255_gate_n29 import (
    EXACT_FAMILY_SEEDS,
    OWNER_RULING_ESTIMATOR,
    exact_cell_id,
    family_series_from_cells,
    independence_control,
)
from basemap.round0255_treatment import HELD_OUT_CELL_IDS
from basemap.round0256_gate_repair import (
    DEFINING_CELL_IDS,
    PLANTED_BUILDERS,
    REGISTERED_ESTIMATOR_NAMES,
    RULING_HELD_OUT_CELL,
    Round0256RepairError,
    assert_registry_is_readable_only_as_registered,
    independence_positive_controls,
    panel_false_alarm_rate,
    registered_criteria,
    registry_controls,
    two_sided_detection_power,
)

K_ONE = 2.6934393367626024
K_TWO = 3.147763220676551


def _family_cells():
    """A synthetic twenty-nine-cell family in the shape of the real one.

    Every value is distinct and the first cell sits ABOVE the median. Ties
    would let a rank-order estimator absorb a single-cell perturbation by
    landing the median on an equal value, and driving a cell that is already
    below the median never moves the median at all -- both would make a plant
    look uncaught for a reason that has nothing to do with the property under
    test. In the real family, `exact-seed42` sits above the median on `ffr`,
    which is why R0255's arm 3 moved that floor.
    """
    order = [((index + 20) % len(EXACT_FAMILY_SEEDS)) - 14 for index in range(29)]
    base = {
        "density_v2": 0.44,
        "ffr": 0.33,
        "purity_fidelity_k256": 0.986,
        "purity_fidelity_k1024": 0.711,
    }
    return [
        {
            "cell_id": exact_cell_id(seed),
            "family": "exact-graph",
            "values": {
                metric: base[metric] + 0.0005 * order[index]
                for metric in METRICS
            },
            "ratios": {
                "k256": 1.008 + 0.0007 * order[index],
                "k1024": 0.712 + 0.0006 * order[index],
            },
        }
        for index, seed in enumerate(EXACT_FAMILY_SEEDS)
    ]


def _held_out_cells():
    """Twelve held-out cells, deliberately far from the family."""
    return [
        {
            "cell_id": cell_id,
            "family": cell_id.rsplit("-", 1)[0],
            "values": {metric: 0.20 + 0.001 * index for metric in METRICS},
            "ratios": {"k256": 1.30, "k1024": 0.40},
        }
        for index, cell_id in enumerate(HELD_OUT_CELL_IDS)
    ]


def _control(build=None):
    kwargs = {
        "estimator": OWNER_RULING_ESTIMATOR,
        "multiplier_one_sided": K_ONE,
        "multiplier_two_sided": K_TWO,
        "family_cells": _family_cells(),
        "held_out_cells": _held_out_cells(),
        "defining_cell_ids": DEFINING_CELL_IDS,
    }
    if build is not None:
        kwargs["build"] = build
    return independence_control(**kwargs)


# --------------------------------------------------------------------------- #
# A. the repaired independence control
# --------------------------------------------------------------------------- #


def test_the_builder_selects_only_the_defining_cells():
    cells = _family_cells() + _held_out_cells()
    series, log_series = family_series_from_cells(
        cells, defining_cell_ids=DEFINING_CELL_IDS
    )
    assert all(len(values) == len(EXACT_FAMILY_SEEDS) for values in series.values())
    family_ffr = sorted(cell["values"]["ffr"] for cell in _family_cells())
    assert sorted(series["ffr"]) == family_ffr
    for metric in PURITY_METRICS:
        assert log_series[metric] == [
            math.log(cell["ratios"][PURITY_RATIO_KEYS[metric]])
            for cell in _family_cells()
        ]


def test_the_builder_refuses_a_missing_defining_cell():
    with pytest.raises(Exception):
        family_series_from_cells(
            _held_out_cells(), defining_cell_ids=DEFINING_CELL_IDS
        )


def test_every_arm_passes_its_perturbation_into_the_fit():
    control = _control()
    assert len(control["arms"]) == 4
    assert all(arm["perturbation_reaches_the_fit"] for arm in control["arms"])


def test_the_held_out_arms_are_bitwise_identical_and_the_family_arms_move():
    control = _control()
    assert control["arms"][0]["every_floor_bitwise_identical"] is True
    assert control["arms"][1]["every_floor_bitwise_identical"] is True
    assert control["the_fit_is_independent_of_every_held_out_cell"] is True
    assert control["arms"][3]["every_floor_moved"] is True
    assert control["the_fit_is_not_inert"] is True
    assert control["holds"] is True


def test_a_builder_that_reads_a_held_out_cell_makes_the_arms_report_false():
    """The positive control R0255 never shipped. This is the load-bearing test."""
    for _name, build, _why in PLANTED_BUILDERS:
        under_the_plant = _control(build=build)
        assert under_the_plant[
            "the_fit_is_independent_of_every_held_out_cell"
        ] is False, (
            f"{build.__name__} planted a held-out cell into the fit and the "
            "control still reported independence"
        )


def test_the_positive_control_harness_catches_every_plant():
    report = independence_positive_controls(
        estimator=OWNER_RULING_ESTIMATOR,
        multiplier_one_sided=K_ONE,
        multiplier_two_sided=K_TWO,
        family_cells=_family_cells(),
        held_out_cells=_held_out_cells(),
    )
    assert report["planted_defects"] == len(PLANTED_BUILDERS)
    assert report["every_planted_defect_is_caught"] is True
    assert report["the_honest_builder_still_reports_independent"] is True
    assert report["the_honest_builder_is_not_inert"] is True
    assert report["holds"] is True


def test_the_arms_still_fail_when_a_family_cell_is_the_one_driven():
    """Arm 3 must MOVE the location -- the correction to R0255's result section D3."""
    control = _control()
    arm3 = control["arms"][2]
    assert arm3["any_floor_moved"] is True
    assert arm3["is_an_assertion"] is False
    assert arm3["floors"]["ffr"] != control["baseline_floors"]["ffr"]


# --------------------------------------------------------------------------- #
# B. the restricted calibration
# --------------------------------------------------------------------------- #


def test_restricting_the_estimator_leaves_the_rng_stream_and_the_numbers_alone():
    full = calibration.calibrate(9, families=20_000)
    full.pop("_arrays")
    restricted = calibration.calibrate(
        9, families=20_000, names=REGISTERED_ESTIMATOR_NAMES
    )
    restricted.pop("_arrays")
    assert tuple(restricted["candidates"]) == REGISTERED_ESTIMATOR_NAMES
    left = full["candidates"][OWNER_RULING_ESTIMATOR]
    right = restricted["candidates"][OWNER_RULING_ESTIMATOR]
    assert left["one_sided"] == right["one_sided"]
    assert left["two_sided"] == right["two_sided"]


# --------------------------------------------------------------------------- #
# C. two-sided power
# --------------------------------------------------------------------------- #


def test_two_sided_power_is_strictly_below_one_sided_power_and_matches_its_size():
    at = calibration.calibrate(29, families=60_000, names=REGISTERED_ESTIMATOR_NAMES)
    centre, scale = at["_arrays"][OWNER_RULING_ESTIMATOR]
    entry = at["candidates"][OWNER_RULING_ESTIMATOR]
    k_two = float(entry["two_sided"]["calibrated_multiplier"])
    power = two_sided_detection_power(centre, scale, k_two)
    size = two_sided_detection_power(centre, scale, k_two, alternatives=(0.0,))
    assert size["minus_0_sigma"] == float(
        entry["two_sided"]["new_cell_false_fail_rate"]
    )
    one_sided = entry["one_sided"]["detection_power"]
    for key in ("minus_1_sigma", "minus_2_sigma", "minus_3_sigma"):
        assert power[key] < float(one_sided[key])


def test_two_sided_power_of_a_symmetric_band_is_computed_on_both_tails():
    """A one-tailed implementation would miss the upper tail; plant a shifted null."""
    centre = np.zeros(200_000)
    scale = np.ones(200_000)
    upper_only = two_sided_detection_power(centre, scale, 2.0, alternatives=(-6.0,))
    assert upper_only["minus_-6_sigma"] > 0.99


def test_the_panel_rate_compounds_above_every_per_metric_rate():
    report = panel_false_alarm_rate(
        one_sided_rate=0.01222093973890843,
        two_sided_rate=0.01112250056202322,
        two_sided_criteria=2,
    )
    rate = report["panel_false_alarm_rate_under_independence"]
    assert rate > 0.01222093973890843
    assert rate == pytest.approx(0.034072, abs=1e-6)
    assert report["this_is_an_upper_bound"] is True


# --------------------------------------------------------------------------- #
# D. the registry
# --------------------------------------------------------------------------- #


def _registry_fixture():
    floors = {metric: 0.3 for metric in METRICS}
    bands = {
        metric: {"ratio_lower": 0.9, "ratio_upper": 1.1} for metric in PURITY_METRICS
    }
    criteria = registered_criteria(floors=floors, bands=bands)
    return criteria, {"ffr": floors["ffr"]}, floors


def test_the_registry_carries_every_gated_criterion_and_no_descriptive_metric():
    criteria, scalar_floors, _floors = _registry_fixture()
    assert set(criteria) == set(GATED_METRICS)
    assert not set(criteria) & set(DESCRIPTIVE_METRICS)
    for metric in PURITY_METRICS:
        assert criteria[metric]["kind"] == "two_sided_ratio_band"
    assert criteria["ffr"]["kind"] == "one_sided_lower_floor"
    assert assert_registry_is_readable_only_as_registered(criteria, scalar_floors)[
        "holds"
    ] is True


def test_the_registry_refuses_a_descriptive_metric_left_readable_as_a_floor():
    """R0255's shipped shape: density_v2 populated, the gated purity entries null."""
    criteria, scalar_floors, floors = _registry_fixture()
    with pytest.raises(Round0256RepairError):
        assert_registry_is_readable_only_as_registered(
            criteria,
            {**scalar_floors, DESCRIPTIVE_METRICS[0]: floors[DESCRIPTIVE_METRICS[0]]},
        )


def test_the_registry_refuses_a_dropped_gated_criterion():
    criteria, scalar_floors, _floors = _registry_fixture()
    with pytest.raises(Round0256RepairError):
        assert_registry_is_readable_only_as_registered(
            {k: v for k, v in criteria.items() if k != PURITY_METRICS[0]},
            scalar_floors,
        )


def test_the_registry_controls_refuse_every_plant_and_pass_the_honest_registry():
    criteria, scalar_floors, floors = _registry_fixture()
    report = registry_controls(
        criteria=criteria,
        scalar_floors=scalar_floors,
        descriptive_values={metric: floors[metric] for metric in DESCRIPTIVE_METRICS},
    )
    assert report["the_honest_registry_passes"] is True
    assert report["every_planted_defect_is_refused"] is True
    assert report["planted_defects"] == 5
    assert report["holds"] is True


def test_the_ruling_held_out_cell_is_the_one_arm_one_drives():
    assert RULING_HELD_OUT_CELL in set(HELD_OUT_CELL_IDS)
