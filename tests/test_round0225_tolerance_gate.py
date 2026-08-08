"""R0225 — the tolerance factor is derived, and the defect is measured.

Review 0222-01 blocked `claim:r0222-purity-fidelity-floor-is-a-calibrated-
instrument` on three grounds. These tests pin the answer to each:

1. the one-sided 95/95 factor at `n = 8` is **derived** from the noncentral t,
   and independently confirmed by simulating the coverage it delivers — not
   copied from the review;
2. the purity band is two-sided on the **unfolded** log-ratio scale, so
   over-separation and under-separation are different failures;
3. self-loosening is measured rather than asserted, and the measurement is
   reported even where it is unflattering to the new method.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basemap.round0225_tolerance_gate import (  # noqa: E402
    EXACT_FAMILY_SEEDS,
    GATE_METRICS,
    LEGACY_MULTIPLIER,
    PURITY_METRICS,
    REVIEW_0222_ONE_SIDED_FACTOR_N8,
    Round0225Error,
    evaluate_cell,
    log_ratio_band,
    mean_minus_2sd_floor,
    measure_self_loosening,
    one_sided_tolerance_factor,
    one_sided_tolerance_floor,
    registered_gate,
    score_all_cells,
    two_sided_tolerance_factor,
)

#: The sealed 8-cell exact-graph family (R0222 `pooled_panel_metric_cells`) and
#: its unfolded ratios (R0223 `exact_family_purity_ratios`). Fixtures only; the
#: node reads these from the artifacts, never from here.
EXACT_CELLS = {
    "42": {"density_v2": 0.4377, "ffr": 0.3369, "purity_fidelity_k256": 0.9788566953797964, "purity_fidelity_k1024": 0.7326},
    "43": {"density_v2": 0.4406, "ffr": 0.3382, "purity_fidelity_k256": 0.9941346058256287, "purity_fidelity_k1024": 0.7229},
    "44": {"density_v2": 0.4387, "ffr": 0.3258, "purity_fidelity_k256": 0.9954210631096955, "purity_fidelity_k1024": 0.6980},
    "45": {"density_v2": 0.4477, "ffr": 0.3227, "purity_fidelity_k256": 0.9929, "purity_fidelity_k1024": 0.6936},
    "46": {"density_v2": 0.4434, "ffr": 0.3312, "purity_fidelity_k256": 0.9951238929246692, "purity_fidelity_k1024": 0.7214},
    "47": {"density_v2": 0.4400, "ffr": 0.3209, "purity_fidelity_k256": 0.9932, "purity_fidelity_k1024": 0.6842},
    "48": {"density_v2": 0.4393, "ffr": 0.3344, "purity_fidelity_k256": 0.9643201542912248, "purity_fidelity_k1024": 0.7266},
    "49": {"density_v2": 0.4491, "ffr": 0.3240, "purity_fidelity_k256": 0.9901970492127933, "purity_fidelity_k1024": 0.6991},
}
EXACT_RATIOS = {
    "42": {"k256": 1.0216, "k1024": 0.7326},
    "43": {"k256": 1.0059, "k1024": 0.7229},
    "44": {"k256": 1.0046, "k1024": 0.6980},
    "45": {"k256": 0.9929, "k1024": 0.6936},
    "46": {"k256": 1.0049, "k1024": 0.7214},
    "47": {"k256": 0.9932, "k1024": 0.6842},
    "48": {"k256": 1.0370, "k1024": 0.7266},
    "49": {"k256": 1.0099, "k1024": 0.6991},
}


# --------------------------------------------------------------------------- #
# A. the factor is derived
# --------------------------------------------------------------------------- #


def test_one_sided_factor_at_n8_reproduces_the_reviews_value() -> None:
    factor = one_sided_tolerance_factor(8)
    assert factor["n"] == 8
    assert factor["degrees_of_freedom"] == 7
    assert factor["cross_check_passes"] is True
    assert abs(factor["k"] - REVIEW_0222_ONE_SIDED_FACTOR_N8) < 1.0e-3
    # And it is much larger than the multiplier R0222 used.
    assert factor["k"] > LEGACY_MULTIPLIER


def test_one_sided_factor_delivers_the_coverage_it_claims() -> None:
    """Independent confirmation: simulate, do not re-use the nct formula.

    A 95/95 lower tolerance bound must sit below the population's 5th
    percentile in 95% of families. This checks that directly by simulation,
    and checks that `mean - 2*sigma` does NOT — which is the calibration half
    of review-0222-01's objection.
    """
    numpy = pytest.importorskip("numpy")
    stats = pytest.importorskip("scipy.stats")

    n, families = 8, 200_000
    k = one_sided_tolerance_factor(n)["k"]
    rng = numpy.random.default_rng(20260808)
    sample = rng.standard_normal((families, n))
    mean = sample.mean(1)
    sd = sample.std(1, ddof=1)
    target = stats.norm.ppf(0.05)

    covered_tolerance = float(((mean - k * sd) <= target).mean())
    covered_legacy = float(((mean - LEGACY_MULTIPLIER * sd) <= target).mean())

    assert 0.945 < covered_tolerance < 0.955, covered_tolerance
    # The registered multiplier delivers far less than 95% confidence.
    assert covered_legacy < 0.75, covered_legacy


def test_factor_shrinks_as_n_grows() -> None:
    factors = [one_sided_tolerance_factor(n)["k"] for n in (4, 8, 16, 30)]
    assert factors == sorted(factors, reverse=True)
    # Review 0222-01 quotes 2.524 at n = 16.
    assert abs(one_sided_tolerance_factor(16)["k"] - 2.524) < 1.0e-2


def test_two_sided_factor_exceeds_the_one_sided_factor() -> None:
    assert two_sided_tolerance_factor(8)["k2"] > one_sided_tolerance_factor(8)["k"]


def test_factor_refuses_a_family_too_small_to_fit() -> None:
    with pytest.raises(Round0225Error):
        one_sided_tolerance_factor(2)


# --------------------------------------------------------------------------- #
# B. the purity band is two-sided, on the unfolded scale
# --------------------------------------------------------------------------- #


def test_log_ratio_band_is_two_sided_and_centred_on_the_family() -> None:
    ratios = [EXACT_RATIOS[str(seed)]["k256"] for seed in EXACT_FAMILY_SEEDS]
    band = log_ratio_band(ratios)
    assert band["n"] == 8
    assert band["ratio_lower"] < 1.0 < band["ratio_upper"]
    # The family centres ABOVE 1.0, which is why folding about 1.0 is wrong.
    assert band["centre_is_above_one"] is True
    assert band["ratio_geometric_mean"] > 1.0
    # Verified from the cells, not copied from R0223's published value.
    assert abs(band["log_ratio_mean"] - 0.008620608788497673) < 1e-12


def test_band_distinguishes_over_from_under_separation() -> None:
    ratios = [EXACT_RATIOS[str(seed)]["k256"] for seed in EXACT_FAMILY_SEEDS]
    band = log_ratio_band(ratios)
    over = evaluate_cell(
        value=band["ratio_upper"] * 1.05,
        floor=band["ratio_lower"],
        upper=band["ratio_upper"],
    )
    under = evaluate_cell(
        value=band["ratio_lower"] * 0.95,
        floor=band["ratio_lower"],
        upper=band["ratio_upper"],
    )
    assert over["direction"] == "above_band" and over["passes"] is False
    assert under["direction"] == "below_band" and under["passes"] is False
    # The folded metric maps both of these to the same value, which is the
    # defect: exp(-|log r|) is symmetric about r = 1.
    assert math.isclose(
        math.exp(-abs(math.log(1.05))), math.exp(-abs(math.log(1 / 1.05)))
    )


def test_k1024_family_centres_below_one() -> None:
    """Both directions really do occur in this program, so both bounds matter."""
    ratios = [EXACT_RATIOS[str(seed)]["k1024"] for seed in EXACT_FAMILY_SEEDS]
    band = log_ratio_band(ratios)
    assert band["centre_is_above_one"] is False
    assert band["ratio_upper"] < 1.0


def test_log_ratio_band_refuses_a_nonpositive_ratio() -> None:
    with pytest.raises(Round0225Error):
        log_ratio_band([1.0, 1.0, 0.0, 1.0])


# --------------------------------------------------------------------------- #
# C. self-loosening is measured, including where it is unflattering
# --------------------------------------------------------------------------- #


def test_both_floor_families_are_self_loosening() -> None:
    values = [EXACT_CELLS[str(seed)]["purity_fidelity_k256"] for seed in EXACT_FAMILY_SEEDS]
    report = measure_self_loosening(values)
    for row in report["injections"]:
        assert row["mean_minus_2sd"]["loosened"] is True
        assert row["one_sided_tolerance_95_95"]["loosened"] is True


def test_the_tolerance_floor_loosens_more_not_less() -> None:
    """The honest finding: 95/95 fixes calibration, NOT self-loosening.

    Because `k > 2`, the same inflation of `s` moves the tolerance floor
    further than it moves `mean - 2*sigma`. Reporting this is the point; a
    round that claimed the tolerance interval cured self-loosening would be
    wrong, and review-0222-01's required correction 3 (calibrate on held-out
    cells) would be left unaddressed while looking addressed.
    """
    for metric in GATE_METRICS:
        values = [EXACT_CELLS[str(seed)][metric] for seed in EXACT_FAMILY_SEEDS]
        report = measure_self_loosening(values)
        for row in report["injections"]:
            assert row["tolerance_moves_more_than_legacy"] is True
    assert "does not fix this property" in report["finding"] or "not this property" in report["finding"]


def test_self_loosening_is_monotone_in_the_injected_outlier() -> None:
    values = [EXACT_CELLS[str(seed)]["ffr"] for seed in EXACT_FAMILY_SEEDS]
    report = measure_self_loosening(values, sigma_multiples=(1.0, 2.0, 3.0, 4.0))
    shifts = [row["mean_minus_2sd"]["shift"] for row in report["injections"]]
    assert shifts == sorted(shifts, reverse=True)
    assert all(shift < 0 for shift in shifts)


# --------------------------------------------------------------------------- #
# D. the eleven cells
# --------------------------------------------------------------------------- #


def _gate():
    return registered_gate(exact_cells=EXACT_CELLS, exact_ratios=EXACT_RATIOS)


def test_gate_carries_n_beside_every_floor() -> None:
    gate = _gate()
    for metric in GATE_METRICS:
        assert gate["gates"][metric]["mean_minus_2sd"]["n"] == 8
        assert gate["gates"][metric]["one_sided_tolerance_95_95"]["n"] == 8
    for metric in PURITY_METRICS:
        assert gate["gates"][metric]["two_sided_log_ratio_95_95"]["n"] == 8


def test_tolerance_floor_is_below_the_legacy_floor_on_every_metric() -> None:
    gate = _gate()
    for metric in GATE_METRICS:
        legacy = gate["gates"][metric]["mean_minus_2sd"]["floor"]
        tolerance = gate["gates"][metric]["one_sided_tolerance_95_95"]["floor"]
        assert tolerance < legacy, metric


def test_seed_48_fails_only_the_legacy_k256_floor() -> None:
    """The headline: the cell R0222 failed clears both new floors."""
    gate = _gate()
    value = EXACT_CELLS["48"]["purity_fidelity_k256"]
    assert value < gate["gates"]["purity_fidelity_k256"]["mean_minus_2sd"]["floor"]
    assert value >= gate["gates"]["purity_fidelity_k256"][
        "one_sided_tolerance_95_95"
    ]["floor"]
    band = gate["gates"]["purity_fidelity_k256"]["two_sided_log_ratio_95_95"]
    ratio = EXACT_RATIOS["48"]["k256"]
    assert band["ratio_lower"] <= ratio <= band["ratio_upper"]


def test_all_eleven_cells_are_scored_against_every_family() -> None:
    gate = _gate()
    cells = [
        {
            "cell_id": f"exact-seed{seed}",
            "family": "exact-graph",
            "seed": seed,
            "defines_the_floors": True,
            "values": EXACT_CELLS[str(seed)],
            "ratios": EXACT_RATIOS[str(seed)],
        }
        for seed in EXACT_FAMILY_SEEDS
    ] + [
        {
            "cell_id": f"cuvs-igd48-seed{seed}",
            "family": "cuvs-igd48",
            "seed": seed,
            "defines_the_floors": False,
            "values": values,
            "ratios": ratios,
        }
        for seed, values, ratios in (
            (42, {"density_v2": 0.4360, "ffr": 0.3237, "purity_fidelity_k256": 0.9920634920634921, "purity_fidelity_k1024": 0.7076}, {"k256": 1.0080, "k1024": 0.7076}),
            (43, {"density_v2": 0.4550, "ffr": 0.3322, "purity_fidelity_k256": 0.9812579727210284, "purity_fidelity_k1024": 0.7249}, {"k256": 1.0191, "k1024": 0.7249}),
            (44, {"density_v2": 0.4349, "ffr": 0.3367, "purity_fidelity_k256": 0.9666505558240697, "purity_fidelity_k1024": 0.7279}, {"k256": 1.0345, "k1024": 0.7279}),
        )
    ]
    scored = score_all_cells(gate=gate, cells=cells)
    assert scored["cell_count"] == 11
    assert sum(1 for row in scored["cells"] if row["defines_the_floors"]) == 8

    # The one legacy failure is seed 48 on k256, and nothing else fails anywhere.
    legacy = scored["per_family"]["mean_minus_2sd"]["failing_cell_metrics"]
    assert legacy == [
        {"cell_id": "exact-seed48", "metric": "purity_fidelity_k256", "direction": None}
    ]
    assert scored["per_family"]["one_sided_tolerance_95_95"]["failures"] == 0
    assert scored["per_family"]["two_sided_log_ratio_95_95"]["failures"] == 0


def test_registered_gate_refuses_a_family_that_is_not_seeds_42_to_49() -> None:
    cells = dict(EXACT_CELLS)
    cells.pop("49")
    with pytest.raises(Round0225Error):
        registered_gate(exact_cells=cells, exact_ratios=EXACT_RATIOS)


def test_legacy_floors_match_the_sealed_r0222_values() -> None:
    """R0222's floors must reproduce, or the two families are not comparable."""
    published = {
        "density_v2": 0.4335282413076137,
        "ffr": 0.3157181021069332,
        "purity_fidelity_k256": 0.9660625420699066,
        "purity_fidelity_k1024": 0.6737066290217798,
    }
    for metric, expected in published.items():
        values = [EXACT_CELLS[str(seed)][metric] for seed in EXACT_FAMILY_SEEDS]
        assert abs(mean_minus_2sd_floor(values)["floor"] - expected) < 1e-12, metric


def test_tolerance_floors_match_r0223s_independently_sealed_values() -> None:
    """R0223 sealed the same four 95/95 floors. Agreement is a cross-check."""
    published = {
        "density_v2": 0.4284619060791439,
        "ffr": 0.30767751385346537,
        "purity_fidelity_k256": 0.9530280530310268,
        "purity_fidelity_k1024": 0.65227991540876,
    }
    for metric, expected in published.items():
        values = [EXACT_CELLS[str(seed)][metric] for seed in EXACT_FAMILY_SEEDS]
        assert abs(one_sided_tolerance_floor(values)["floor"] - expected) < 1e-12, metric
