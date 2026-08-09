"""R0234 — the calibrated-robust-floor contract, exercised without any GPU.

These tests are the release CPU smoke for R0234. They check the arithmetic of
every scale estimator against hand-computed values, that the Monte-Carlo
calibrator reproduces the two closed forms it can be checked against, that
attainability is *derived* rather than asserted (including the case where the
1-trimmed family stops being attainable at its own calibrated multiplier), that
the injection test still separates variance from robust scales, and that no
published verdict can change without being enumerated.
"""
from __future__ import annotations

import math
import statistics

import pytest

from basemap.round0234_calibrated_floors import (
    CANDIDATES,
    CANDIDATE_ORDER,
    DEGENERATE_WITNESS,
    GATED_METRICS,
    IQR_CONSISTENCY,
    MAD_CONSISTENCY,
    N_EXACT,
    POWER_MATERIALITY,
    Round0234Error,
    SELECTION_RULE,
    attainability,
    band_at,
    centre_and_scale,
    degenerate_witness,
    floor_at,
    identity_bound,
    injection_ladder,
    iqr_n,
    mad_n,
    positive_scale_witness,
    qn_scale,
    rank_slack_bound,
    score_cell_metric,
    score_population,
    sn_scale,
    trimmed_centre_scale,
    verdict_changes,
)


SAMPLE = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]


# --------------------------------------------------------------------------- #
# estimators
# --------------------------------------------------------------------------- #


def test_mad_n_matches_its_definition():
    # median 7; |x - 7| = 6,5,4,3,2,1,0,1,2,3,4,5,6 -> median 3.
    assert mad_n(SAMPLE) == pytest.approx(MAD_CONSISTENCY * 3.0, rel=0, abs=1e-12)


def test_iqr_n_uses_type7_quantiles():
    # n = 13 -> Q1 is exactly x_(4) = 4, Q3 is exactly x_(10) = 10.
    assert iqr_n(SAMPLE) == pytest.approx(6.0 / IQR_CONSISTENCY, rel=0, abs=1e-12)


def test_qn_scale_matches_its_order_statistic():
    # 78 pairwise gaps of an arithmetic ladder; rank C(7,2) = 21.
    gaps = sorted(
        abs(SAMPLE[i] - SAMPLE[j])
        for i in range(len(SAMPLE))
        for j in range(i + 1, len(SAMPLE))
    )
    expected = (13 / 14.4) * 2.2219 * gaps[20]
    assert qn_scale(SAMPLE) == pytest.approx(expected, rel=0, abs=1e-12)


def test_sn_scale_matches_its_order_statistic():
    inner = sorted(
        sorted(abs(left - right) for right in SAMPLE)[len(SAMPLE) // 2]
        for left in SAMPLE
    )
    expected = (13 / 12.1) * 1.1926 * inner[(len(SAMPLE) + 1) // 2 - 1]
    assert sn_scale(SAMPLE) == pytest.approx(expected, rel=0, abs=1e-12)


def test_trimmed_drops_one_per_tail():
    centre, scale = trimmed_centre_scale(SAMPLE)
    kept = SAMPLE[1:-1]
    assert centre == pytest.approx(statistics.fmean(kept))
    assert scale == pytest.approx(statistics.stdev(kept))


def test_every_candidate_is_fittable_and_positive_on_a_spread_family():
    for name in CANDIDATE_ORDER:
        centre, scale = centre_and_scale(name, SAMPLE)
        assert math.isfinite(centre)
        assert scale > 0.0


def test_nonfinite_and_tiny_families_are_refused():
    with pytest.raises(Round0234Error):
        mad_n([1.0, float("nan"), 3.0])
    with pytest.raises(Round0234Error):
        mad_n([1.0, 2.0])


# --------------------------------------------------------------------------- #
# calibration — checked against the two closed forms
# --------------------------------------------------------------------------- #


def test_calibrator_reproduces_the_noncentral_t_factor():
    from basemap import round0234_calibration as calibration

    result = calibration.calibrate(
        N_EXACT, families=200_000, chunk=50_000,
        names=("mean_minus_k_sample_sd",),
    )
    result.pop("_arrays")
    k = result["candidates"]["mean_minus_k_sample_sd"]["one_sided"][
        "calibrated_multiplier"
    ]
    assert k == pytest.approx(calibration.nct_one_sided_factor(N_EXACT), abs=0.02)
    delivered = result["candidates"]["mean_minus_k_sample_sd"]["one_sided"][
        "delivered_coverage"
    ]
    assert delivered == pytest.approx(0.95, abs=0.005)


def test_calibrator_reproduces_howes_two_sided_factor():
    from basemap import round0234_calibration as calibration

    result = calibration.calibrate(
        N_EXACT, families=100_000, chunk=50_000,
        names=("mean_minus_k_sample_sd",),
    )
    result.pop("_arrays")
    k2 = result["candidates"]["mean_minus_k_sample_sd"]["two_sided"][
        "calibrated_multiplier"
    ]
    assert k2 == pytest.approx(calibration.howe_two_sided_factor(N_EXACT), abs=0.05)


def test_three_is_not_the_calibrated_multiplier_for_madn_at_n13():
    """The whole reason this round exists: `3` under-delivers at `n = 13`."""
    from basemap import round0234_calibration as calibration

    drawn = calibration.draw_centres_and_scales(
        N_EXACT, families=200_000, chunk=50_000, names=("median_minus_k_madn",)
    )
    at_three = calibration.summarise(
        *drawn["median_minus_k_madn"], 3.0, content=0.95, alternatives=(2.0,)
    )
    assert at_three["delivered_coverage"] < 0.93


# --------------------------------------------------------------------------- #
# attainability, derived not asserted
# --------------------------------------------------------------------------- #


def test_rank_slack_bound_is_the_actual_inequality():
    assert rank_slack_bound(terms=13, rank=7, terms_touched_by_one_cell=1)
    assert rank_slack_bound(terms=78, rank=21, terms_touched_by_one_cell=12)
    assert not rank_slack_bound(terms=13, rank=13, terms_touched_by_one_cell=1)


def test_sample_sd_attainability_is_the_identity_bound():
    entry = attainability("mean_minus_k_sample_sd", n=N_EXACT, multiplier=2.6712)
    assert entry["max_abs_z_bound"] == pytest.approx(identity_bound(N_EXACT))
    assert entry["every_defining_cell_can_fail"]
    assert not attainability(
        "mean_minus_k_sample_sd", n=N_EXACT, multiplier=3.4
    )["every_defining_cell_can_fail"]


def test_trimmed_family_loses_attainability_at_its_calibrated_multiplier():
    """Only the trimmed cells can fail once `k` exceeds `(11-1)/sqrt(11)`."""
    kept_bound = identity_bound(N_EXACT - 2)
    assert kept_bound == pytest.approx(10.0 / math.sqrt(11.0))
    entry = attainability(
        "trimmed1_mean_minus_k_trimmed_sd", n=N_EXACT, multiplier=3.72
    )
    assert not entry["every_defining_cell_can_fail"]
    assert entry["cells_that_can_fail"] == 2
    loose = attainability(
        "trimmed1_mean_minus_k_trimmed_sd", n=N_EXACT, multiplier=2.0
    )
    assert loose["every_defining_cell_can_fail"]


def test_robust_scales_are_unbounded_so_every_cell_can_fail():
    for name in (
        "median_minus_k_madn",
        "median_minus_k_sn",
        "median_minus_k_qn",
        "median_minus_k_iqrn",
    ):
        entry = attainability(name, n=N_EXACT, multiplier=99.0)
        assert entry["bound_is_finite"] is False
        assert entry["every_defining_cell_can_fail"]
        assert entry["cells_that_can_fail"] == N_EXACT


def test_a_far_outlier_cannot_move_a_robust_scale_to_infinity():
    """The empirical face of the rank-slack argument."""
    for name in ("median_minus_k_madn", "median_minus_k_sn", "median_minus_k_qn"):
        base = centre_and_scale(name, SAMPLE)[1]
        pushed = list(SAMPLE)
        pushed[0] = -1e9
        assert centre_and_scale(name, pushed)[1] < 10.0 * base
        z = (centre_and_scale(name, pushed)[0] - pushed[0]) / centre_and_scale(
            name, pushed
        )[1]
        assert z > 1e6


def test_positive_scale_witness_is_not_degenerate_and_the_r0231_one_is():
    for name in CANDIDATE_ORDER:
        witness = positive_scale_witness(name, 3.7)
        assert witness["scale_is_strictly_positive"]
        can_fail = attainability(name, n=N_EXACT, multiplier=3.7)[
            "every_defining_cell_can_fail"
        ]
        # A witness exists exactly when the derived bound says one can. At
        # k = 3.7 > (13-1)/sqrt(13) the sample-sd family has no witness, and
        # that is the correct answer rather than a defect in the construction.
        assert witness["lowest_cell_fails_its_own_floor"] == (
            can_fail or name == "trimmed1_mean_minus_k_trimmed_sd"
        )
    assert not positive_scale_witness("mean_minus_k_sample_sd", 3.7)[
        "lowest_cell_fails_its_own_floor"
    ]
    assert positive_scale_witness("mean_minus_k_sample_sd", 2.6712)[
        "lowest_cell_fails_its_own_floor"
    ]
    degenerate = degenerate_witness("median_minus_k_madn", 3.7)
    assert degenerate["scale_value"] == 0.0
    assert not degenerate["scale_is_strictly_positive"]
    assert list(DEGENERATE_WITNESS) == [1.0] * (N_EXACT - 1) + [0.0]


# --------------------------------------------------------------------------- #
# invariance
# --------------------------------------------------------------------------- #


def test_sample_sd_is_self_loosening_and_madn_is_not():
    variance = injection_ladder("mean_minus_k_sample_sd", SAMPLE, 2.6712)
    assert variance["exact_invariance_depth"] == 0
    assert variance["self_loosening_at_any_depth"]
    robust = injection_ladder("median_minus_k_madn", SAMPLE, 3.7364)
    assert robust["exact_invariance_depth"] >= 1
    assert all(
        row["shift"] == 0.0 for row in robust["rows"] if row["depth"] == 1
    )


def test_injection_tracks_the_bound_that_faces_the_contaminated_tail():
    upper = injection_ladder(
        "median_minus_k_madn", SAMPLE, 4.45, side="upper", depths=(1,)
    )
    assert all(row["shift"] == 0.0 for row in upper["rows"])
    assert upper["side"] == "upper"


def test_calibration_does_not_buy_back_invariance_for_the_trimmed_family():
    """Two contaminated cells defeat a family that trims one per tail."""
    ladder = injection_ladder(
        "trimmed1_mean_minus_k_trimmed_sd", SAMPLE, 3.7215, depths=(1, 2)
    )
    assert ladder["exact_invariance_depth"] == 1


# --------------------------------------------------------------------------- #
# scoring, bands, verdict changes
# --------------------------------------------------------------------------- #


def test_band_scoring_is_two_sided_and_keeps_the_direction():
    below = score_cell_metric(value=0.9, floor=0.5, ratio=0.97, band=(0.98, 1.02))
    assert not below["passes"] and below["band_direction"] == "below_band"
    assert below["separation"] == "under_separating"
    above = score_cell_metric(value=0.9, floor=0.5, ratio=1.04, band=(0.98, 1.02))
    assert not above["passes"] and above["band_direction"] == "above_band"
    assert above["separation"] == "over_separating"
    inside = score_cell_metric(value=0.9, floor=0.5, ratio=1.0, band=(0.98, 1.02))
    assert inside["passes"] and inside["band_direction"] == "inside_band"
    # the folded floor rides along descriptively and never decides
    assert inside["descriptive_folded_floor"] == 0.5


def test_a_one_sided_floor_still_decides_an_unbanded_metric():
    entry = score_cell_metric(value=0.30, floor=0.31, ratio=None, band=None)
    assert not entry["passes"] and entry["direction"] == "below_floor"


def _cells():
    return [
        {
            "cell_id": "a",
            "family": "f",
            "values": {metric: 1.0 for metric in GATED_METRICS},
            "ratios": {"k256": 1.0, "k1024": 0.71},
        },
        {
            "cell_id": "b",
            "family": "f",
            "values": {"ffr": 0.10, "purity_fidelity_k256": 1.0,
                       "purity_fidelity_k1024": 1.0},
            "ratios": {"k256": 1.0, "k1024": 0.71},
        },
    ]


def test_score_population_reports_attainability_and_failures():
    scored = score_population(
        cells=_cells(),
        floors={"ffr": 0.30},
        bands={"purity_fidelity_k256": (0.98, 1.02)},
        metrics=GATED_METRICS,
        defining_cell_ids=["a"],
        every_defining_cell_can_fail=True,
    )
    assert scored["cells_scored"] == 2
    assert scored["failures"] == 1
    assert scored["non_defining_cells_scored"] == 1
    assert scored["count_is_attainable"]


def test_verdict_changes_names_every_reversal():
    chosen = score_population(
        cells=_cells(),
        floors={"ffr": 0.05},
        bands={},
        metrics=("ffr",),
        defining_cell_ids=["a"],
        every_defining_cell_can_fail=True,
    )
    published = {
        "released_family": score_population(
            cells=_cells(),
            floors={"ffr": 0.30},
            bands={},
            metrics=("ffr",),
            defining_cell_ids=["a"],
            every_defining_cell_can_fail=True,
        )
    }
    changes = verdict_changes(chosen=chosen, published=published)
    assert changes["count"] == 1
    assert changes["un_failed_published_failures"][0]["cell_id"] == "b"
    assert changes["un_failed_published_failures"][0]["metric"] == "ffr"


# --------------------------------------------------------------------------- #
# the contract itself
# --------------------------------------------------------------------------- #


def test_the_selection_rule_is_registered_in_the_release_module():
    for phrase in (
        "COVERAGE", "INVARIANCE", "ATTAINABILITY", "detection power",
        "breakdown point", "register NOTHING",
    ):
        assert phrase in SELECTION_RULE
    assert POWER_MATERIALITY == 0.01


def test_breakdown_points_are_recorded_a_priori():
    assert CANDIDATES["median_minus_k_madn"]["breakdown_point"] == 0.5
    assert CANDIDATES["median_minus_k_sn"]["breakdown_point"] == 0.5
    assert CANDIDATES["median_minus_k_qn"]["breakdown_point"] == 0.5
    assert CANDIDATES["median_minus_k_iqrn"]["breakdown_point"] == 0.25
    assert CANDIDATES["trimmed1_mean_minus_k_trimmed_sd"]["breakdown_point"] == 1 / 13
    assert CANDIDATES["mean_minus_k_sample_sd"]["breakdown_point"] == 0.0


def test_floor_and_band_helpers_agree_with_centre_and_scale():
    centre, scale = centre_and_scale("median_minus_k_madn", SAMPLE)
    assert floor_at("median_minus_k_madn", SAMPLE, 3.0) == pytest.approx(
        centre - 3.0 * scale
    )
    lower, upper = band_at("median_minus_k_madn", SAMPLE, 3.0)
    assert lower == pytest.approx(centre - 3.0 * scale)
    assert upper == pytest.approx(centre + 3.0 * scale)
