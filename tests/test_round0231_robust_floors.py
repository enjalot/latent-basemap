"""Registered checks for R0231's robust floors at n = 13.

The tests that matter are the ones that would catch the failure mode this round
exists to end: a gate that cannot be failed, and a floor that gets laxer when a
cell gets worse.

* the three robust estimators are **exactly invariant** to a 3-sigma injection and
  both variance estimators are **not** — reproduced on Review 0225's own eight-cell
  numbers, so the harness is calibrated against a published measurement before it
  is trusted on thirteen cells;
* a defining cell can fail every registered family at `n = 13`, exhibited by an
  explicit witness rather than inferred from an inequality;
* the same statement is **false** at `n = 4`, at `n = 3`, and at `n = 8` under
  `k = 3.187` — the three historical registrations;
* a purity band has a lower *and* an upper bound and reports direction, so
  over-separation and under-separation are distinguishable;
* `density_v2` is not in the gated set.
"""
from __future__ import annotations

import math
import statistics

import pytest

from basemap.round0231_robust_floors import (
    CHOSEN_ESTIMATOR,
    DESCRIPTIVE_METRICS,
    EXACT_FAMILY_SEEDS,
    FLOOR_ESTIMATORS,
    GATED_METRICS,
    MAD_CONSISTENCY,
    METRICS,
    N_EXACT,
    PURITY_METRICS,
    ROBUST_ESTIMATORS,
    VARIANCE_ESTIMATORS,
    Round0231Error,
    defining_cell_can_fail,
    fit_floors,
    identity_bound,
    log_ratio_band,
    mad_n,
    mean_minus_2sd_floor,
    measure_self_loosening,
    median_fixed_margin_floor,
    median_minus_3_mad_floor,
    one_sided_tolerance_factor,
    one_sided_tolerance_floor,
    score_cells_against,
    trimmed_mean_minus_2s_floor,
    two_sided_tolerance_factor,
    witness_defining_cell_failure,
)


#: R0222's eight sealed cells, so the harness can be checked against Review
#: 0225-01's published self-loosening measurement before it is used on thirteen.
R0225_EIGHT_CELL_VALUES = {
    "density_v2": [0.4377, 0.4406, 0.4387, 0.4477, 0.4434, 0.4400, 0.4393, 0.4491],
    "ffr": [0.3369, 0.3382, 0.3258, 0.3227, 0.3312, 0.3209, 0.3344, 0.3240],
    "purity_fidelity_k256": [
        0.9788566953797964, 0.9941346058256287, 0.9954210631096955, 0.9929,
        0.9951238929246692, 0.9932, 0.9643201542912248, 0.9901970492127933,
    ],
    "purity_fidelity_k1024": [
        0.7326, 0.7229, 0.6980, 0.6936, 0.7214, 0.6842, 0.7266, 0.6991,
    ],
}
#: review-0225-2026-08-08-01.md section 3, "3-sigma injection", to seven decimals.
R0225_PUBLISHED_SHIFTS = {
    "density_v2": {"mean_minus_2sd": -0.0078509, "one_sided": -0.0115616,
                   "median_minus_3_mad": -0.0011119},
    "ffr": {"mean_minus_2sd": -0.0131613, "one_sided": -0.0194668,
            "median_minus_3_mad": 0.0},
    "purity_fidelity_k256": {"mean_minus_2sd": -0.0259379, "one_sided": -0.0388918,
                             "median_minus_3_mad": 0.0},
    "purity_fidelity_k1024": {"mean_minus_2sd": -0.0366460, "one_sided": -0.0543833,
                              "median_minus_3_mad": 0.0},
}


# --------------------------------------------------------------------------- #
# calibration against a published measurement
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("metric", sorted(R0225_EIGHT_CELL_VALUES))
def test_reproduces_review_0225s_self_loosening_measurement(metric: str) -> None:
    measured = measure_self_loosening(R0225_EIGHT_CELL_VALUES[metric])
    at_3 = measured["injections"][-1]
    published = R0225_PUBLISHED_SHIFTS[metric]
    assert at_3["mean_minus_2sd"]["shift"] == pytest.approx(
        published["mean_minus_2sd"], abs=5e-8
    )
    assert at_3["one_sided_tolerance_95_95"]["shift"] == pytest.approx(
        published["one_sided"], abs=5e-8
    )
    assert at_3["median_minus_3_mad"]["shift"] == pytest.approx(
        published["median_minus_3_mad"], abs=5e-8
    )
    assert at_3["trimmed1_mean_minus_2s"]["shift"] == 0.0
    assert at_3["median_fixed_margin"]["shift"] == 0.0


def test_the_tolerance_floor_moves_further_than_mean_minus_2sd() -> None:
    """R0225's honest negative result, re-measured: a bigger k loosens more."""
    for metric, values in R0225_EIGHT_CELL_VALUES.items():
        at_3 = measure_self_loosening(values)["injections"][-1]
        legacy = abs(at_3["mean_minus_2sd"]["shift"])
        tolerance = abs(at_3["one_sided_tolerance_95_95"]["shift"])
        assert tolerance > legacy, metric
        assert tolerance / legacy == pytest.approx(1.48, abs=0.03), metric


def test_the_derived_tolerance_factors_match_the_published_table() -> None:
    for n, k in ((3, 7.655900), (4, 5.143875), (8, 3.187294), (13, 2.670504),
                 (16, 2.523659)):
        assert one_sided_tolerance_factor(n)["k"] == pytest.approx(k, abs=1e-6)
    for n, k2 in ((8, 3.768539), (13, 3.100799), (16, 2.918147)):
        assert two_sided_tolerance_factor(n)["k2"] == pytest.approx(k2, abs=1e-6)


# --------------------------------------------------------------------------- #
# the estimators
# --------------------------------------------------------------------------- #


def test_mad_n_is_normal_consistent() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert mad_n(values) == pytest.approx(MAD_CONSISTENCY * 1.0, abs=1e-12)


def test_the_three_robust_estimators_are_invariant_to_one_extreme_cell() -> None:
    base = [0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40,
            0.41, 0.42]
    assert len(base) == N_EXACT
    injected = list(base)
    injected[0] = base[0] - 10.0 * statistics.stdev(base)
    for name in ("median_minus_3_mad", "trimmed1_mean_minus_2s",
                 "median_fixed_margin"):
        estimator = FLOOR_ESTIMATORS[name]
        assert estimator(injected)["floor"] == pytest.approx(
            estimator(base)["floor"], abs=1e-12
        ), name
    for name in VARIANCE_ESTIMATORS:
        estimator = FLOOR_ESTIMATORS[name]
        assert estimator(injected)["floor"] < estimator(base)["floor"], name


def test_the_trimmed_estimator_is_defeated_by_two_outliers_and_the_mad_is_not() -> None:
    """Why breakdown point, not just single-outlier invariance, decides the choice."""
    base = [0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40,
            0.41, 0.42]
    injected = list(base)
    injected[0] = -5.0
    injected[1] = -5.0
    assert trimmed_mean_minus_2s_floor(injected)["floor"] < trimmed_mean_minus_2s_floor(
        base
    )["floor"]
    assert median_minus_3_mad_floor(injected)["floor"] == pytest.approx(
        median_minus_3_mad_floor(base)["floor"], abs=1e-12
    )


def test_the_fixed_margin_is_not_derived_from_the_family() -> None:
    """It is invariant, but its strictness is unrelated to the family's spread."""
    tight = [1.0 + 0.0001 * index for index in range(13)]
    loose = [1.0 + 0.05 * index for index in range(13)]
    tight_margin = median_fixed_margin_floor(tight)["margin_in_sample_sd"]
    loose_margin = median_fixed_margin_floor(loose)["margin_in_sample_sd"]
    assert tight_margin > 10.0 * loose_margin


def test_the_chosen_estimator_is_robust_and_named_in_the_contract() -> None:
    assert CHOSEN_ESTIMATOR == "median_minus_3_mad"
    assert CHOSEN_ESTIMATOR in ROBUST_ESTIMATORS
    assert CHOSEN_ESTIMATOR not in VARIANCE_ESTIMATORS


def test_an_estimator_refuses_a_nonfinite_or_too_small_family() -> None:
    with pytest.raises(Round0231Error):
        mean_minus_2sd_floor([1.0, 2.0])
    with pytest.raises(Round0231Error):
        median_minus_3_mad_floor([1.0, 2.0, float("nan")])


# --------------------------------------------------------------------------- #
# can a defining cell fail?
# --------------------------------------------------------------------------- #


def test_the_identity_makes_the_three_historical_registrations_untestable() -> None:
    assert defining_cell_can_fail(
        n=3, multiplier=2.0, scale="sample_sd"
    )["defining_cell_can_fail"] is False
    assert defining_cell_can_fail(
        n=4, multiplier=2.0, scale="sample_sd"
    )["defining_cell_can_fail"] is False
    assert defining_cell_can_fail(
        n=8, multiplier=3.187293568447751, scale="sample_sd"
    )["defining_cell_can_fail"] is False
    # R0222's n=8 mean-2s gate is the one that could be failed, and was.
    assert defining_cell_can_fail(
        n=8, multiplier=2.0, scale="sample_sd"
    )["defining_cell_can_fail"] is True


def test_at_n13_a_defining_cell_can_fail_every_variance_family() -> None:
    bound = identity_bound(N_EXACT)
    assert bound == pytest.approx(3.328201177351375, abs=1e-12)
    for multiplier in (2.0, one_sided_tolerance_factor(13)["k"],
                       two_sided_tolerance_factor(13)["k2"]):
        assert multiplier < bound
        assert defining_cell_can_fail(
            n=N_EXACT, multiplier=multiplier, scale="sample_sd"
        )["defining_cell_can_fail"] is True


def test_the_identity_is_inapplicable_to_a_robust_scale_and_says_so() -> None:
    verdict = defining_cell_can_fail(n=N_EXACT, multiplier=3.0, scale="mad_n")
    assert verdict["identity_applies"] is False
    assert verdict["defining_cell_can_fail"] is True
    assert "SAMPLE" in verdict["reason"]


@pytest.mark.parametrize("estimator", sorted(FLOOR_ESTIMATORS))
def test_every_family_has_a_witness_in_which_a_defining_cell_fails(
    estimator: str,
) -> None:
    witness = witness_defining_cell_failure(estimator)
    assert witness["n"] == N_EXACT
    assert witness["lowest_cell_fails_its_own_floor"] is True


# --------------------------------------------------------------------------- #
# two-sided, unfolded purity
# --------------------------------------------------------------------------- #


def test_a_purity_band_has_both_bounds_and_a_direction() -> None:
    ratios = [1.0216, 1.0059, 1.0046, 0.9929, 1.0049, 0.9932, 1.0370, 1.0099,
              1.0080, 1.0191, 1.0345, 0.9989, 1.0063]
    band = log_ratio_band(ratios, estimator="median_minus_3_mad")
    assert band["ratio_lower"] < band["ratio_upper"]
    assert band["centre_is_above_one"] is True
    assert band["ratio_geometric_centre"] == pytest.approx(
        math.exp(statistics.median([math.log(r) for r in ratios])), abs=1e-12
    )


def test_the_fold_destroys_the_direction_the_band_preserves() -> None:
    """exp(-|log r|) sends an over- and an under-separating cell to one number."""
    over, under = 1.0370, 1.0 / 1.0370
    assert math.exp(-abs(math.log(over))) == pytest.approx(
        math.exp(-abs(math.log(under))), abs=1e-12
    )
    ratios = [1.00 + 0.001 * index for index in range(13)]
    band = log_ratio_band(ratios, estimator="median_minus_3_mad")
    cells = [
        {
            "cell_id": "over",
            "family": "probe",
            "values": {"purity_fidelity_k256": 0.5},
            "ratios": {"k256": band["ratio_upper"] * 1.10},
        },
        {
            "cell_id": "under",
            "family": "probe",
            "values": {"purity_fidelity_k256": 0.5},
            "ratios": {"k256": band["ratio_lower"] * 0.90},
        },
    ]
    scored = score_cells_against(
        floors={"purity_fidelity_k256": 0.0},
        bands={"purity_fidelity_k256": band},
        cells=cells,
        metrics=["purity_fidelity_k256"],
        defining_seed_ids=[],
        family_can_fail_a_defining_cell=True,
    )
    directions = {
        row["cell_id"]: row["metrics"]["purity_fidelity_k256"]["band_direction"]
        for row in scored["cells"]
    }
    assert directions == {"over": "above_band", "under": "below_band"}
    assert scored["failures"] == 2


def test_a_nonpositive_ratio_is_refused() -> None:
    with pytest.raises(Round0231Error):
        log_ratio_band([1.0, 0.0, 1.1], estimator="median_minus_3_mad")


# --------------------------------------------------------------------------- #
# fitting and attainability
# --------------------------------------------------------------------------- #


def _thirteen_cells():
    cells = {
        str(seed): {
            "density_v2": 0.44 + 0.0005 * index,
            "ffr": 0.33 - 0.0005 * index,
            "purity_fidelity_k256": 0.99 - 0.0005 * index,
            "purity_fidelity_k1024": 0.70 + 0.0005 * index,
        }
        for index, seed in enumerate(EXACT_FAMILY_SEEDS)
    }
    ratios = {
        str(seed): {"k256": 1.00 + 0.001 * index, "k1024": 0.70 + 0.001 * index}
        for index, seed in enumerate(EXACT_FAMILY_SEEDS)
    }
    return cells, ratios


def test_fit_floors_covers_every_metric_and_every_estimator() -> None:
    cells, ratios = _thirteen_cells()
    gate = fit_floors(cells=cells, ratios=ratios)
    assert gate["n"] == 13
    assert set(gate["gates"]) == set(METRICS)
    for metric in METRICS:
        assert set(gate["gates"][metric]["families"]) == set(FLOOR_ESTIMATORS)
        assert gate["gates"][metric]["role"] == (
            "descriptive-only" if metric in DESCRIPTIVE_METRICS else "gated"
        )
    for metric in PURITY_METRICS:
        assert "two_sided_bands" in gate["gates"][metric]
    assert "density_v2" not in GATED_METRICS
    assert gate["chosen_estimator"] == CHOSEN_ESTIMATOR


def test_fit_floors_refuses_a_family_that_is_not_seeds_42_to_54() -> None:
    cells, ratios = _thirteen_cells()
    del cells["54"]
    del ratios["54"]
    with pytest.raises(Round0231Error):
        fit_floors(cells=cells, ratios=ratios)


def test_an_all_defining_count_is_attainable_only_when_a_cell_could_fail() -> None:
    cells = [
        {
            "cell_id": "exact-seed42",
            "family": "exact-graph",
            "values": {"ffr": 0.33},
        }
    ]
    unattainable = score_cells_against(
        floors={"ffr": 0.0},
        bands={},
        cells=cells,
        metrics=["ffr"],
        defining_seed_ids=["exact-seed42"],
        family_can_fail_a_defining_cell=False,
    )
    assert unattainable["count_is_attainable"] is False
    assert unattainable["non_defining_cells_scored"] == 0
    attainable = score_cells_against(
        floors={"ffr": 0.0},
        bands={},
        cells=cells,
        metrics=["ffr"],
        defining_seed_ids=["exact-seed42"],
        family_can_fail_a_defining_cell=True,
    )
    assert attainable["count_is_attainable"] is True
    held_out = score_cells_against(
        floors={"ffr": 0.0},
        bands={},
        cells=cells,
        metrics=["ffr"],
        defining_seed_ids=[],
        family_can_fail_a_defining_cell=False,
    )
    assert held_out["count_is_attainable"] is True
    assert held_out["non_defining_cells_scored"] == 1


def test_a_failing_cell_is_reported_with_its_value_and_floor() -> None:
    scored = score_cells_against(
        floors={"ffr": 0.34},
        bands={},
        cells=[
            {"cell_id": "held-out", "family": "probe", "values": {"ffr": 0.33}}
        ],
        metrics=["ffr"],
        defining_seed_ids=[],
        family_can_fail_a_defining_cell=False,
    )
    assert scored["failures"] == 1
    assert scored["failing_cell_metrics"][0]["value"] == 0.33
    assert scored["failing_cell_metrics"][0]["floor"] == 0.34
    assert scored["cells_clearing_every_gated_metric"] == 0
