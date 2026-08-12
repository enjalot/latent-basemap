"""R0260 contract — the one-sided purity re-registration and its controls.

CUDA-hidden, no `/data` write, no queue. Everything here runs on synthetic
families or on constants, except the two tests that read the sealed twenty-nine
cells read-only to confirm the fit reproduces the retired bands.
"""
from __future__ import annotations

import json
import math
import os

import pytest

from basemap.round0234_calibrated_floors import (
    GATED_METRICS,
    PURITY_METRICS,
    band_at,
    identity_bound,
)
from basemap.round0254_dispatch import SCOPE_MODULES
from basemap.round0255_gate_n29 import EXACT_FAMILY_SEEDS
from basemap.round0260_one_sided_purity import (
    CONFORMITY_MARKING,
    KIND_RATIO_FLOOR,
    KIND_SCALAR_FLOOR,
    Round0260RegistrationError,
    assert_no_gate_reads_a_conformity_value,
    assert_registry_is_one_sided_only,
    attainability_against_the_identity_bound,
    conformity_diagnostics,
    conformity_diagnostics_never_gate_controls,
    judge_map_one_sided,
    judge_population_one_sided,
    one_sided_purity_floors,
    panel_false_alarm_rate_one_sided,
    registered_criteria_one_sided,
    registry_controls_one_sided,
    sidedness_controls,
    the_retired_band_disagrees_control,
)

ESTIMATOR = "median_minus_k_madn"
K_ONE = 2.6934393367626024
K_TWO = 3.147763220676551
N = 29
IDENTITY_BOUND = 5.199469468957452

R0255_PANEL = (
    "/data/latent-basemap/runs/round-0255/queue/artifacts/"
    "minilm-mixed-2m-seed-family-panel-n29-v1/seed-family-panel-n29.json"
)

#: The two bands R0255 registered and R0256 re-derived bitwise.
SEALED_BANDS = {
    "purity_fidelity_k256": (0.9697263687993266, 1.0550731452902518),
    "purity_fidelity_k1024": (0.6616903134148893, 0.7719564268121961),
}


def _log_series_from_sealed() -> dict[str, list[float]]:
    with open(R0255_PANEL, "rb") as handle:
        panel = json.load(handle)
    ratios = panel["raw_purity_ratios"]
    return {
        metric: [
            math.log(float(ratios[str(seed)][metric.replace("purity_fidelity_", "")]))
            for seed in EXACT_FAMILY_SEEDS
        ]
        for metric in PURITY_METRICS
    }


def _gate(log_series: dict[str, list[float]], *, ffr_floor: float = 0.3125328635126472):
    floors = one_sided_purity_floors(ESTIMATOR, log_series, K_ONE)
    criteria = registered_criteria_one_sided(
        ffr_floor=ffr_floor, purity_floors=floors
    )
    bands = {
        metric: {
            "ratio_lower": math.exp(band_at(ESTIMATOR, log_series[metric], K_TWO)[0]),
            "ratio_upper": math.exp(band_at(ESTIMATOR, log_series[metric], K_TWO)[1]),
        }
        for metric in PURITY_METRICS
    }
    conformity = conformity_diagnostics(
        two_sided_bands=bands, two_sided_multiplier=K_TWO
    )
    far = panel_false_alarm_rate_one_sided(one_sided_rate=0.01222093973890843)
    return {
        "registered_criteria": criteria,
        "registered_floors": {"ffr": ffr_floor},
        "registered_ratio_floors": {
            metric: floors[metric]["ratio_floor"] for metric in PURITY_METRICS
        },
        "descriptive_conformity_diagnostics": conformity,
        "detection_power_by_criterion": {
            metric: {
                "applicable_power": "one_sided",
                "detection_power": {"minus_2_sigma": 0.30035090040737566},
                "new_cell_false_fail_rate": 0.01222093973890843,
            }
            for metric in GATED_METRICS
        },
        "panel_false_alarm_rate": far,
    }, floors


sealed_family = pytest.mark.skipif(
    not os.path.exists(R0255_PANEL), reason="sealed n=29 family panel is not present"
)


# --------------------------------------------------------------------------- #
# A. the fit
# --------------------------------------------------------------------------- #


@sealed_family
def test_the_retired_two_sided_bands_reproduce_bitwise_from_the_sealed_family():
    """The re-registration must be the SAME fit, changed only in sidedness."""
    log_series = _log_series_from_sealed()
    for metric, (lower, upper) in SEALED_BANDS.items():
        low, high = band_at(ESTIMATOR, log_series[metric], K_TWO)
        assert math.exp(low) == lower
        assert math.exp(high) == upper


@sealed_family
def test_the_one_sided_floor_sits_strictly_inside_the_retired_band():
    """`k1 < k2`, so the one-sided floor is above the band's lower edge.

    This is the arithmetic content of the change: the gated region grows from
    `[lower, upper]` to `[floor, +inf)`, with `floor > lower`. It is a strictly
    tighter test below and no test at all above.
    """
    log_series = _log_series_from_sealed()
    floors = one_sided_purity_floors(ESTIMATOR, log_series, K_ONE)
    for metric, (lower, upper) in SEALED_BANDS.items():
        ratio_floor = floors[metric]["ratio_floor"]
        assert lower < ratio_floor < upper


@sealed_family
def test_no_defining_cell_fails_its_own_one_sided_floor_but_every_one_could():
    log_series = _log_series_from_sealed()
    floors = one_sided_purity_floors(ESTIMATOR, log_series, K_ONE)
    for metric in PURITY_METRICS:
        observed = [math.exp(value) for value in log_series[metric]]
        assert min(observed) > floors[metric]["ratio_floor"]
    identity = attainability_against_the_identity_bound(n=N, multiplier=K_ONE)
    assert identity["multiplier_is_below_the_identity_bound"]


def test_the_identity_bound_is_28_over_sqrt_29():
    assert identity_bound(N) == IDENTITY_BOUND
    identity = attainability_against_the_identity_bound(n=N, multiplier=K_ONE)
    assert identity["identity_bound"] == IDENTITY_BOUND
    assert identity["headroom"] == IDENTITY_BOUND - K_ONE


def test_a_multiplier_at_the_identity_bound_is_reported_as_unfailable():
    """The control for the attainability report: it must be able to say no."""
    identity = attainability_against_the_identity_bound(
        n=N, multiplier=IDENTITY_BOUND
    )
    assert not identity["multiplier_is_below_the_identity_bound"]
    assert identity["headroom"] == 0.0


def test_the_panel_false_alarm_rate_compounds_three_one_sided_criteria():
    """And it goes UP, not down, which is the counter-intuitive part.

    `k1 < k2`, so the one-sided floor sits ABOVE the retired band's lower edge:
    `0.01222093973890843` against `0.01112250056202322` per metric. Dropping the
    upper side removes a failure mode but tightens the side that remains, and the
    compounded panel rate rises from R0256's `0.034071887878658114`.
    """
    one_sided = 0.01222093973890843
    two_sided = 0.01112250056202322
    assert one_sided > two_sided
    far = panel_false_alarm_rate_one_sided(one_sided_rate=one_sided)
    assert far["gated_criteria_count"] == 3
    compounded = 1.0 - (1.0 - one_sided) ** 3
    assert far["panel_false_alarm_rate_under_independence"] == compounded
    assert far["panel_false_alarm_rate_under_independence"] > one_sided
    r0256_panel_rate = 1.0 - (1.0 - one_sided) * (1.0 - two_sided) ** 2
    assert r0256_panel_rate == 0.034071887878658114
    assert far["panel_false_alarm_rate_under_independence"] > r0256_panel_rate


# --------------------------------------------------------------------------- #
# B. the registry guards, and their planted defects
# --------------------------------------------------------------------------- #


@sealed_family
def test_the_honest_registry_passes_both_guards():
    gate, floors = _gate(_log_series_from_sealed())
    assert assert_registry_is_one_sided_only(
        gate["registered_criteria"],
        gate["registered_floors"],
        gate["registered_ratio_floors"],
    )["every_gated_criterion_is_one_sided"]
    assert assert_no_gate_reads_a_conformity_value(
        criteria=gate["registered_criteria"],
        scalar_floors=gate["registered_floors"],
        ratio_floors=gate["registered_ratio_floors"],
        conformity=gate["descriptive_conformity_diagnostics"],
    )["holds"]
    for metric in PURITY_METRICS:
        assert gate["registered_criteria"][metric]["kind"] == KIND_RATIO_FLOOR
        assert "ratio_upper" not in gate["registered_criteria"][metric]
        assert floors[metric]["ratio_floor"] > 0.0
    assert gate["registered_criteria"]["ffr"]["kind"] == KIND_SCALAR_FLOOR


@sealed_family
def test_every_planted_registry_defect_is_refused():
    gate, _ = _gate(_log_series_from_sealed())
    controls = registry_controls_one_sided(
        criteria=gate["registered_criteria"],
        scalar_floors=gate["registered_floors"],
        ratio_floors=gate["registered_ratio_floors"],
        conformity=gate["descriptive_conformity_diagnostics"],
        descriptive_values={"density_v2": 0.41643957035196294},
    )
    assert controls["planted_defects"] == 9
    assert controls["every_planted_defect_is_refused"]
    assert controls["the_honest_registry_passes"]
    assert controls["holds"]


@sealed_family
def test_the_retired_upper_edge_cannot_be_smuggled_in_under_a_permitted_key():
    """The value-level guard, exercised directly rather than through the harness."""
    gate, _ = _gate(_log_series_from_sealed())
    upper = gate["descriptive_conformity_diagnostics"]["two_sided_bands"][
        "purity_fidelity_k1024"
    ]["ratio_upper"]
    assert upper == 0.7719564268121961
    with pytest.raises(Round0260RegistrationError):
        assert_no_gate_reads_a_conformity_value(
            criteria=gate["registered_criteria"],
            scalar_floors=gate["registered_floors"],
            ratio_floors={
                **gate["registered_ratio_floors"],
                "purity_fidelity_k1024": upper,
            },
            conformity=gate["descriptive_conformity_diagnostics"],
        )


def test_the_conformity_block_must_declare_that_it_never_gates():
    bands = {
        metric: {"ratio_lower": 0.9, "ratio_upper": 1.1} for metric in PURITY_METRICS
    }
    conformity = conformity_diagnostics(two_sided_bands=bands, two_sided_multiplier=K_TWO)
    assert conformity[CONFORMITY_MARKING] is True
    assert conformity["contributes_to_verdict"] is False
    for metric in PURITY_METRICS:
        faithful = conformity["a_perfectly_faithful_map_against_the_retired_bands"][
            metric
        ]
        assert faithful["a_perfectly_faithful_map_ratio"] == 1.0
    with pytest.raises(Round0260RegistrationError):
        assert_no_gate_reads_a_conformity_value(
            criteria={},
            scalar_floors={},
            ratio_floors={},
            conformity={
                key: value
                for key, value in conformity.items()
                if key != CONFORMITY_MARKING
            },
        )


# --------------------------------------------------------------------------- #
# C. sidedness — the direction that gates, and the direction that does not
# --------------------------------------------------------------------------- #


@sealed_family
def test_every_sidedness_control_holds():
    gate, _ = _gate(_log_series_from_sealed())
    controls = sidedness_controls(gate)
    assert controls["planted"] == 6
    assert controls["every_control_held"]
    assert controls["at_least_one_control_fails_the_map"]
    assert controls["at_least_one_control_passes_the_map"]


@sealed_family
def test_an_over_separating_map_passes_and_an_under_separating_one_fails():
    """The ruling, stated as two synthetic maps rather than as prose."""
    gate, floors = _gate(_log_series_from_sealed())
    over = judge_map_one_sided(
        cell_id="over",
        metrics={"ffr": 0.5, "density_v2": 0.6},
        raw_ratios={"k256": 5.0, "k1024": 5.0},
        gate=gate,
    )
    assert over["verdict"] == "PASS"
    under = judge_map_one_sided(
        cell_id="under",
        metrics={"ffr": 0.5, "density_v2": 0.6},
        raw_ratios={
            "k256": floors["purity_fidelity_k256"]["ratio_floor"] * (1.0 - 1e-9),
            "k1024": 1.0,
        },
        gate=gate,
    )
    assert under["verdict"] == "FAIL"
    assert under["failing_criteria"] == ["purity_fidelity_k256"]


@sealed_family
def test_a_perfectly_faithful_map_passes_the_new_criterion_and_failed_the_old_one():
    gate, _ = _gate(_log_series_from_sealed())
    disagreement = the_retired_band_disagrees_control(gate)
    assert disagreement["the_re_registration_is_not_a_rename"]
    faithful = [
        row for row in disagreement["rows"] if row["synthetic_map"] == "perfectly_faithful"
    ][0]
    assert faithful["verdict_under_the_one_sided_criterion"] == "PASS"
    assert faithful["criteria_the_retired_band_would_fail"] == [
        "purity_fidelity_k1024"
    ]


@sealed_family
def test_the_judge_reads_the_unfolded_ratio_and_not_the_folded_fidelity():
    """A ratio of 1.10 folds to 0.908; the folded value would fail the k256 floor."""
    gate, floors = _gate(_log_series_from_sealed())
    ratio = 1.1012
    folded = 1.0 / ratio
    assert folded < floors["purity_fidelity_k256"]["ratio_floor"] < ratio
    judged = judge_map_one_sided(
        cell_id="over_separated",
        metrics={
            "ffr": 0.4142,
            "purity_fidelity_k256": folded,
            "purity_fidelity_k1024": 0.8338,
            "density_v2": 0.624,
        },
        raw_ratios={"k256": ratio, "k1024": 0.8338},
        gate=gate,
    )
    assert judged["per_metric"]["purity_fidelity_k256"]["observed"] == ratio
    assert judged["per_metric"]["purity_fidelity_k256"]["passes"]


# --------------------------------------------------------------------------- #
# D. the diagnostics are inert, and the proof is not vacuous
# --------------------------------------------------------------------------- #


@sealed_family
def test_the_conformity_diagnostics_never_reach_a_shipped_verdict():
    gate, _ = _gate(_log_series_from_sealed())
    cells = {
        "ladder-6250k-h2048-seed42": {
            "panel_metrics": {
                "ffr": 0.4142,
                "purity_fidelity_k256": 0.9081002542680713,
                "purity_fidelity_k1024": 0.8338,
                "density_v2": 0.624,
            },
            "raw_purity_ratios": {"k256": 1.1012, "k1024": 0.8338},
        }
    }
    controls = conformity_diagnostics_never_gate_controls(cells=cells, gate=gate)
    assert controls["the_shipped_judge_is_inert_to_the_diagnostics"]
    assert controls["shipped_margins_are_bitwise_identical"]
    assert controls["the_mutation_is_potent"]
    assert controls["holds"]


@sealed_family
def test_the_inertness_control_would_catch_a_judge_that_read_the_diagnostics():
    """The negative control on the control: a potent mutation must flip a reader."""
    gate, _ = _gate(_log_series_from_sealed())
    cells = {
        "reader": {
            "panel_metrics": {"ffr": 0.4142, "density_v2": 0.624},
            "raw_purity_ratios": {"k256": 1.1012, "k1024": 0.8338},
        }
    }
    controls = conformity_diagnostics_never_gate_controls(cells=cells, gate=gate)
    assert controls["planted_consumer_verdicts_before"]["reader"] == "FAIL"
    assert controls["planted_consumer_verdicts_after"]["reader"] == "PASS"
    assert controls["planted_consumer_verdicts_changed"] == ["reader"]


# --------------------------------------------------------------------------- #
# E. population judging and scope
# --------------------------------------------------------------------------- #


@sealed_family
def test_judge_population_reports_every_map_and_pools_nothing():
    gate, _ = _gate(_log_series_from_sealed())
    cells = {
        f"synthetic-{index}": {
            "panel_metrics": {"ffr": 0.4, "density_v2": 0.6},
            "raw_purity_ratios": {"k256": ratio, "k1024": ratio},
        }
        for index, ratio in enumerate((0.5, 1.0, 1.1))
    }
    judged = judge_population_one_sided(cells=cells, gate=gate)
    assert judged["maps_judged"] == 3
    assert judged["maps_failing"] == ["synthetic-0"]
    assert sorted(judged["maps_passing"]) == ["synthetic-1", "synthetic-2"]
    assert not judged["every_map_passes"]


def test_the_round0260_node_module_is_in_scope_for_the_stop_hook_guard():
    assert "experiments.round0260_nodes" in SCOPE_MODULES


def test_a_nonfinite_observation_is_refused():
    bands = {
        metric: {"ratio_lower": 0.9, "ratio_upper": 1.1} for metric in PURITY_METRICS
    }
    gate = {
        "registered_criteria": registered_criteria_one_sided(
            ffr_floor=0.3,
            purity_floors={
                metric: {"ratio_floor": 0.9, "log_floor": math.log(0.9)}
                for metric in PURITY_METRICS
            },
        ),
        "descriptive_conformity_diagnostics": conformity_diagnostics(
            two_sided_bands=bands, two_sided_multiplier=K_TWO
        ),
        "detection_power_by_criterion": {
            metric: {
                "detection_power": {"minus_2_sigma": 0.3},
                "new_cell_false_fail_rate": 0.012,
            }
            for metric in GATED_METRICS
        },
        "panel_false_alarm_rate": panel_false_alarm_rate_one_sided(
            one_sided_rate=0.012
        ),
    }
    with pytest.raises(Round0260RegistrationError):
        judge_map_one_sided(
            cell_id="nan",
            metrics={"ffr": float("nan"), "density_v2": 0.6},
            raw_ratios={"k256": 1.0, "k1024": 1.0},
            gate=gate,
        )
