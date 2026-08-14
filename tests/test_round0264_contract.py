"""R0264 (DRAFT) contract — the collapse floor, the fail-closed fog ceiling, and
density_v3 registered descriptive-only.

CUDA-hidden, no ``/data`` write, no queue, no 4-million-family calibration. Every test
runs on this round's own sealed CONSTANTS (``round0264_nodes`` carries R0264's copy of
the calibration multipliers, the healthy-family statistics, the legacy fingerprint and
the md035 degeneracy) plus the deterministic metric mirror — so the whole file is
hermetic and needs neither ``/data`` nor the collapse/fog program's home repo.

The load-bearing guard of this round is the FOG FAIL-CLOSED behaviour: a degenerate /
insufficient-resolution measurement is refused (NOT_MEASURABLE), never a pass. It is
planted here two ways — a synthetic degenerate result dict, and a real degenerate map
measured through the metric mirror's ``_map_fog`` binning — and both must fire the
guard, while a naive value-only judge is shown to pass the same input.
"""
from __future__ import annotations

import math

import pytest

from basemap.round0234_calibrated_floors import identity_bound
from basemap.round0255_gate_n29 import N_EXACT
from experiments.round0264_nodes import (
    COLLAPSE_METRIC,
    DENSITY_V3_METRIC,
    DESCRIPTIVE_MARKING,
    FOG_METRIC,
    KIND_ONE_SIDED_LOWER_FLOOR,
    KIND_ONE_SIDED_UPPER_CEILING_FAIL_CLOSED,
    LEGACY_FINGERPRINT,
    HEALTHY_FAMILY,
    SEALED_COLLAPSE_FLOOR,
    SEALED_FOG_CEILING,
    SEALED_IDENTITY_BOUND_AT_N29,
    SEALED_K1_N29,
    SEALED_K7_HEALTHY_ROUNDED,
    SEALED_MD035_DEGENERATE,
    SEALED_VALIDATION_GATE_N13,
    VERDICT_NOT_MEASURABLE,
    Round0264NodeError,
    _assert_registry_gates_collapse_and_fog_only,
    _collapse_floor_controls,
    _descriptive_density_v3_diagnostics,
    _fit_provisional_bands,
    _fog_fail_closed_controls,
    _judge_collapse,
    _judge_fog,
    _judge_fog_naive,
    _map_fog,
    _measurements_cross_check,
    _planted_degenerate_map,
    _registered_criteria_collapse_fog,
    _registry_controls_collapse_fog,
    _separation_record,
)


# The provisional bands, at the precision REPORT.md §6 publishes.
COLLAPSE_FLOOR = 0.644414
FOG_CEILING = 0.732966
LEGACY_COLLAPSE_MAX = 0.1904337
HEALTHY_COLLAPSE_MIN = 1.047621099372249


# --------------------------------------------------------------------------- #
# A. the calibration multipliers — bitwise
# --------------------------------------------------------------------------- #


def test_the_calibration_multipliers_reproduce_bitwise():
    assert SEALED_K1_N29 == 2.6934393367626024
    assert SEALED_VALIDATION_GATE_N13 == 3.7363661743730416


def test_the_identity_bound_is_28_over_sqrt_29_and_k1_sits_below_it():
    assert identity_bound(N_EXACT) == SEALED_IDENTITY_BOUND_AT_N29
    assert SEALED_K1_N29 < SEALED_IDENTITY_BOUND_AT_N29


def test_k7_exceeds_the_identity_bound_at_n7_the_provisional_caveat():
    """At n = 7 the 95/95 multiplier (~6.055) exceeds the identity bound (~2.268). That
    is why the bands are provisional; it is reported, not gated."""
    assert SEALED_K7_HEALTHY_ROUNDED > identity_bound(7)


# --------------------------------------------------------------------------- #
# B. the healthy-family band fit reproduces the published provisional bounds
# --------------------------------------------------------------------------- #


def test_the_healthy_family_fit_reproduces_the_published_bands():
    bands = _fit_provisional_bands(SEALED_K7_HEALTHY_ROUNDED)
    assert bands["family_size"] == 7
    assert round(bands["collapse"]["floor"], 4) == round(SEALED_COLLAPSE_FLOOR, 4)
    assert round(bands["fog"]["ceiling"], 4) == round(SEALED_FOG_CEILING, 4)
    # the collapse median / MAD_n are the healthy family's, not the legacy family's
    assert round(bands["collapse"]["median"], 4) == 1.1379
    assert round(bands["fog"]["median"], 4) == 0.4792


def test_the_collapse_floor_separates_the_two_modes():
    """legacy max < floor < healthy min — the 5.5x gap the floor sits inside."""
    separation = _separation_record(collapse_floor=COLLAPSE_FLOOR)
    assert separation["legacy_collapse_max"] == LEGACY_COLLAPSE_MAX
    assert separation["healthy_collapse_min"] == HEALTHY_COLLAPSE_MIN
    assert separation["the_two_modes_do_not_overlap"]
    assert separation["floor_is_above_the_legacy_mode"]
    assert separation["floor_is_below_the_healthy_mode"]
    assert separation["separation_factor"] > 5.0


def test_the_legacy_family_is_never_the_fog_source():
    """The fog ceiling comes from the healthy family; the legacy family's fog is LOWER,
    so a legacy-fitted ceiling would reject the healthy maps — hence fingerprint only."""
    assert LEGACY_FINGERPRINT["fog"]["median"] < HEALTHY_FAMILY[0]["fog"]


# --------------------------------------------------------------------------- #
# C. the FOG FAIL-CLOSED guard — the load-bearing positive control
# --------------------------------------------------------------------------- #


def test_a_planted_degenerate_map_is_refused_and_a_naive_judge_would_pass_it():
    """Plant a degenerate map, measure it through the real binning, and prove the shipped
    fog judge REFUSES it (NOT_MEASURABLE) while a value-only judge PASSES it — so the
    degeneracy guard is what changes the outcome, not a coincidence."""
    planted_xy = _planted_degenerate_map()
    result = _map_fog(planted_xy)
    # the degeneracy arises from the real binning: N < 100 => peak bin < 100 => no level
    assert result["peak_bin_count"] < 100
    assert result["resolution_levels"] < 1
    assert result["degenerate"] is True
    assert result["fog"] == 0.0

    shipped = _judge_fog(result, FOG_CEILING)
    assert shipped["verdict"] == VERDICT_NOT_MEASURABLE
    assert shipped["passes"] is False
    assert shipped["measurable"] is False

    naive = _judge_fog_naive(result, FOG_CEILING)
    assert naive["passes"] is True  # 0.0 <= ceiling: the spurious pass the guard blocks


def test_the_sealed_md035_case_is_refused_never_passed():
    """The real-data degeneracy: fog 0.0000 with peak bin 99 must never PASS the ceiling."""
    md035 = {
        "fog": SEALED_MD035_DEGENERATE["fog"],
        "peak_bin_count": SEALED_MD035_DEGENERATE["peak_bin_count"],
        "resolution_levels": SEALED_MD035_DEGENERATE["resolution_levels"],
        "degenerate": SEALED_MD035_DEGENERATE["degenerate"],
    }
    shipped = _judge_fog(md035, FOG_CEILING)
    assert shipped["verdict"] == VERDICT_NOT_MEASURABLE
    assert shipped["passes"] is False
    # without the guard it would have passed (0.0 <= 0.7330)
    assert _judge_fog_naive(md035, FOG_CEILING)["passes"] is True


def test_a_resolution_levels_zero_flag_alone_fails_closed():
    """Fail-closed on `resolution_levels < 1` even if `degenerate` is not set."""
    result = {"fog": 0.0, "resolution_levels": 0, "degenerate": False}
    assert _judge_fog(result, FOG_CEILING)["verdict"] == VERDICT_NOT_MEASURABLE


def test_an_honest_hazy_map_still_fails_the_ceiling_from_above():
    """The guard is not a blanket veto: a measurable, genuinely hazy map FAILS (over)."""
    hazy = {"fog": 0.95, "resolution_levels": 14, "degenerate": False}
    judged = _judge_fog(hazy, FOG_CEILING)
    assert judged["measurable"] is True
    assert judged["verdict"] == "FAIL"
    # and a measurable, healthy map PASSES
    healthy = {"fog": HEALTHY_FAMILY[0]["fog"], "resolution_levels": 14, "degenerate": False}
    assert _judge_fog(healthy, FOG_CEILING)["verdict"] == "PASS"


def test_the_fog_fail_closed_control_aggregate_holds():
    controls = _fog_fail_closed_controls(fog_ceiling=FOG_CEILING)
    assert controls["shipped_judge_refuses_the_planted_map"]
    assert controls["naive_judge_would_have_passed_it"]
    assert controls["the_guard_changed_the_outcome"]
    assert controls["md035"]["shipped_refused"]
    assert controls["md035"]["naive_would_have_passed"]
    assert controls["an_honest_non_degenerate_map_is_measurable_and_passes"]
    assert controls["holds"]


# --------------------------------------------------------------------------- #
# D. the COLLAPSE floor guard
# --------------------------------------------------------------------------- #


def test_the_collapse_floor_fails_a_planted_bead_and_the_legacy_family_but_passes_healthy():
    controls = _collapse_floor_controls(collapse_floor=COLLAPSE_FLOOR)
    assert controls["planted_bead_fails_the_floor"]
    assert controls["legacy_fingerprint_max_fails_the_floor"]
    assert controls["every_healthy_arm_passes_the_floor"]
    assert controls["cuml_reference_passes_the_floor"]
    assert controls["holds"]


def test_the_collapse_judge_is_one_sided_lower():
    assert _judge_collapse(COLLAPSE_FLOOR, COLLAPSE_FLOOR)["verdict"] == "PASS"
    assert _judge_collapse(COLLAPSE_FLOOR - 1e-9, COLLAPSE_FLOOR)["verdict"] == "FAIL"
    assert _judge_collapse(2.0, COLLAPSE_FLOOR)["verdict"] == "PASS"


# --------------------------------------------------------------------------- #
# E. density_v3 is descriptive-only — it never appears as a gate
# --------------------------------------------------------------------------- #


def _honest_registry():
    criteria = _registered_criteria_collapse_fog(
        collapse_floor=COLLAPSE_FLOOR, fog_ceiling=FOG_CEILING
    )
    descriptive = _descriptive_density_v3_diagnostics()
    return criteria, descriptive


def test_the_honest_registry_gates_collapse_and_fog_only():
    criteria, descriptive = _honest_registry()
    report = _assert_registry_gates_collapse_and_fog_only(criteria, descriptive)
    assert report["holds"]
    assert criteria[COLLAPSE_METRIC]["kind"] == KIND_ONE_SIDED_LOWER_FLOOR
    assert criteria[FOG_METRIC]["kind"] == KIND_ONE_SIDED_UPPER_CEILING_FAIL_CLOSED
    assert criteria[FOG_METRIC]["fails_closed_on_degeneracy"] is True
    # density_v3 is NOT a criterion, and its block is marked descriptive-only.
    assert DENSITY_V3_METRIC not in criteria
    assert descriptive[DESCRIPTIVE_MARKING] is True
    assert descriptive["contributes_to_verdict"] is False


def test_registering_density_v3_as_a_gate_is_refused():
    criteria, descriptive = _honest_registry()
    with pytest.raises(Round0264NodeError):
        _assert_registry_gates_collapse_and_fog_only(
            {
                **criteria,
                DENSITY_V3_METRIC: {
                    "kind": KIND_ONE_SIDED_LOWER_FLOOR, "floor": 0.1,
                    "applicable_power": "one_sided",
                },
            },
            descriptive,
        )


def test_a_fog_gate_without_the_fail_closed_flag_is_refused():
    """The realistic defect a fog gate must not ship: a ceiling that reads the value
    before the degeneracy flag — the exact shape that passes md035 spuriously."""
    criteria, descriptive = _honest_registry()
    with pytest.raises(Round0264NodeError):
        _assert_registry_gates_collapse_and_fog_only(
            {
                **criteria,
                FOG_METRIC: {**criteria[FOG_METRIC], "fails_closed_on_degeneracy": False},
            },
            descriptive,
        )


def test_fog_registered_as_a_lower_floor_is_refused():
    """Wrong direction: haze fails from ABOVE; a lower floor cannot see it."""
    criteria, descriptive = _honest_registry()
    with pytest.raises(Round0264NodeError):
        _assert_registry_gates_collapse_and_fog_only(
            {**criteria, FOG_METRIC: {**criteria[FOG_METRIC], "kind": KIND_ONE_SIDED_LOWER_FLOOR}},
            descriptive,
        )


def test_dropping_the_collapse_floor_is_refused():
    criteria, descriptive = _honest_registry()
    with pytest.raises(Round0264NodeError):
        _assert_registry_gates_collapse_and_fog_only(
            {k: v for k, v in criteria.items() if k != COLLAPSE_METRIC}, descriptive
        )


def test_an_unmarked_descriptive_block_is_refused():
    criteria, descriptive = _honest_registry()
    with pytest.raises(Round0264NodeError):
        _assert_registry_gates_collapse_and_fog_only(
            criteria, {k: v for k, v in descriptive.items() if k != DESCRIPTIVE_MARKING}
        )


def test_every_planted_registry_defect_is_refused():
    criteria, descriptive = _honest_registry()
    controls = _registry_controls_collapse_fog(criteria=criteria, descriptive=descriptive)
    assert controls["the_honest_registry_passes"]
    assert controls["planted_defects"] == 5
    assert controls["planted_defects_refused"] == 5
    assert controls["every_planted_defect_is_refused"]
    assert controls["holds"]


# --------------------------------------------------------------------------- #
# F. bitwise reproduction of the carried band constants
# --------------------------------------------------------------------------- #


def test_the_provisional_band_constants_match_the_fitted_bands():
    bands = _fit_provisional_bands(SEALED_K7_HEALTHY_ROUNDED)
    assert round(bands["collapse"]["floor"], 6) == SEALED_COLLAPSE_FLOOR
    assert round(bands["fog"]["ceiling"], 6) == SEALED_FOG_CEILING


def test_the_measurements_cross_check_is_mandatory_and_raises_when_unbound():
    """The cross-check is not optional: with no ``collapse_fog_measurements`` bound it
    RAISES rather than returning a skip literal — the positive control for making it a
    computation that always executes. (An absent/altered file raises at ``_bound_path``.)"""
    with pytest.raises(Round0264NodeError):
        _measurements_cross_check({})
    with pytest.raises(Round0264NodeError):
        _measurements_cross_check({"collapse_fog_measurements": None})


def test_map_fog_reports_the_fail_closed_keys():
    """The metric mirror returns exactly the keys the fail-closed judge consults."""
    result = _map_fog(_planted_degenerate_map())
    for key in ("fog", "low_density_mass_fraction", "resolution_levels", "degenerate",
                "peak_bin_count"):
        assert key in result
    assert result["fog"] == result["low_density_mass_fraction"]
    assert math.isfinite(result["fog"])
