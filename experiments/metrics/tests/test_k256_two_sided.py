"""Tests for the restored TWO-SIDED ``purity_fidelity_k256`` criterion.

Positive controls are mandatory here: a criterion proved only by "nothing
failed" is indistinguishable from a criterion that reaches nothing.  Every
directional claim is planted and observed.

CPU only.  Run with::

    CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest \\
        experiments/metrics/tests/test_k256_two_sided.py -q
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import k256_two_sided as k256  # noqa: E402


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def family():
    return k256.load_sealed_family()["ratios"]


@pytest.fixture(scope="module")
def band(family):
    return k256.fit_band(family)


# --------------------------------------------------------------------------
# The band itself
# --------------------------------------------------------------------------


def test_band_reproduces_the_r0255_sealed_band(band):
    """The restored criterion IS R0255's registered band, not a new one."""
    sealed = k256.R0255_K256_BAND
    assert band["n"] == 29
    assert band["k2"] == pytest.approx(k256.R0255_N29_K2, rel=0, abs=0)
    assert band["log_center"] == pytest.approx(sealed["log_centre"], rel=0, abs=1e-15)
    assert band["log_scale"] == pytest.approx(sealed["log_scale"], rel=0, abs=1e-15)
    assert band["lower"] == pytest.approx(sealed["ratio_lower"], rel=0, abs=1e-15)
    assert band["upper"] == pytest.approx(sealed["ratio_upper"], rel=0, abs=1e-15)
    assert band["center"] == pytest.approx(sealed["ratio_geometric_centre"], rel=0, abs=1e-12)


def test_band_straddles_the_ideal_ratio(band):
    """Ideal is 1.0 and the band must be able to see both sides of it."""
    assert band["lower"] < 1.0 < band["upper"]
    assert band["sided"] == "two_sided"
    assert band["scale"] == "unfolded log ratio"


def test_the_one_sided_floor_in_force_derives_from_the_same_fit(band):
    """R0260's floor is the same centre/scale with k1 instead of +/- k2."""
    floor = math.exp(band["log_center"] - k256.R0255_N29_K1 * band["log_scale"])
    assert floor == pytest.approx(k256.R0260_K256_ONE_SIDED_FLOOR, rel=0, abs=1e-15)


# --------------------------------------------------------------------------
# (a) planted OVER-separating cell -- the collapse direction
# --------------------------------------------------------------------------


def test_planted_over_separating_cell_fails_with_direction_over(band):
    planted = band["upper"] * 1.05  # well above the upper edge
    v = k256.judge(planted, band)
    assert v["verdict"] == "FAIL"
    assert v["direction"] == "over"
    assert v["separation"] == "over_separating"
    assert v["margin"] < 0
    assert v["z_madn"] > band["k2"]


def test_the_shipped_rung_maps_are_the_planted_over_case_in_the_wild(band):
    """R0257's three 6.25M maps: FAIL from above under the restored criterion."""
    for ratio in (1.1012, 1.1022, 1.0982):
        v = k256.judge(ratio, band)
        assert v["verdict"] == "FAIL"
        assert v["direction"] == "over"
        # ... while the criterion in force passes every one of them.
        assert k256.judge_one_sided(ratio)["verdict"] == "PASS"


def test_a_five_times_over_separated_map_still_fails(band):
    """R0260's own sidedness control planted ratio 5.0 and PASSED it."""
    assert k256.judge(5.0, band)["verdict"] == "FAIL"
    assert k256.judge(5.0, band)["direction"] == "over"
    assert k256.judge_one_sided(5.0)["verdict"] == "PASS"


# --------------------------------------------------------------------------
# (b) planted UNDER-separating cell
# --------------------------------------------------------------------------


def test_planted_under_separating_cell_fails_with_direction_under(band):
    planted = band["lower"] * 0.95
    v = k256.judge(planted, band)
    assert v["verdict"] == "FAIL"
    assert v["direction"] == "under"
    assert v["separation"] == "under_separating"
    assert v["margin"] < 0
    assert v["z_madn"] < -band["k2"]


def test_one_part_in_a_million_below_the_lower_edge_fails(band):
    v = k256.judge(band["lower"] * (1.0 - 1e-6), band)
    assert v["verdict"] == "FAIL"
    assert v["direction"] == "under"


# --------------------------------------------------------------------------
# (c) conforming cells pass
# --------------------------------------------------------------------------


def test_a_conforming_cell_passes(band):
    v = k256.judge(band["center"], band)
    assert v["verdict"] == "PASS"
    assert v["direction"] == "inside"
    assert v["margin"] >= 0


def test_a_perfectly_faithful_map_passes(band):
    """Ratio 1.0 must never fail a criterion whose ideal is 1.0."""
    v = k256.judge(1.0, band)
    assert v["verdict"] == "PASS"
    assert v["separation"] == "exactly_faithful"


def test_every_sealed_defining_cell_passes(family, band):
    for ratio in family:
        assert k256.judge(ratio, band)["verdict"] == "PASS", ratio


def test_the_edges_are_inclusive(band):
    assert k256.judge(band["lower"], band)["verdict"] == "PASS"
    assert k256.judge(band["upper"], band)["verdict"] == "PASS"


# --------------------------------------------------------------------------
# (d) the simulator validation gate
# --------------------------------------------------------------------------


def test_simulator_reproduces_r0234_n13_two_sided_multiplier():
    """Gate: reproduce R0234's published n = 13 k2 = 4.45241 within 2 %."""
    res = k256.calibrate_multipliers(13, families=1_000_000, seed=k256.R0234_CALIBRATION_SEED)
    k2 = res["two_sided_multiplier"]
    rel = abs(k2 / k256.R0234_N13_K2 - 1.0)
    assert rel < 0.02, f"k2={k2} vs published {k256.R0234_N13_K2} (rel {rel:.4%})"
    # the one-sided multiplier is calibrated by the same node and is also checked
    rel1 = abs(res["one_sided_multiplier"] / k256.R0234_N13_K1 - 1.0)
    assert rel1 < 0.02


def test_simulator_reproduces_the_sealed_n29_two_sided_multiplier():
    res = k256.calibrate_multipliers(29, families=1_000_000, seed=k256.R0234_CALIBRATION_SEED)
    rel = abs(res["two_sided_multiplier"] / k256.R0255_N29_K2 - 1.0)
    assert rel < 0.02, res["two_sided_multiplier"]


def test_two_sided_multiplier_exceeds_the_one_sided_one():
    res = k256.calibrate_multipliers(29, families=200_000, seed=1234)
    assert res["two_sided_multiplier"] > res["one_sided_multiplier"]


def test_calibration_refuses_a_degenerate_request():
    with pytest.raises(ValueError):
        k256.calibrate_multipliers(1, families=10)


# --------------------------------------------------------------------------
# (e) folding-vs-unfolded regression -- review-0222-01
# --------------------------------------------------------------------------

# review-0222-01: the n = 8 folded floor, the number that failed exact-seed48.
R0222_N8_FOLDED_FLOOR = 0.9660625420699066


def test_folded_floor_passes_a_cell_the_unfolded_band_fails(band):
    """The asked regression: fold says PASS, the two-sided band says FAIL.

    The folded floor is a lower bound on ``min(r, 1/r)``, i.e. the symmetric
    interval ``[f, 1/f]`` about 1.0.  The sealed family does not centre on 1.0
    (``r_geo = 1.0115``), so the folded interval and the unfolded band are
    genuinely different criteria and disagree in BOTH directions.  Here:
    ``[0.9620488..., 0.9697263...)`` is passed by the fold and failed, as
    under-separation, by the band.
    """
    folded_floor = k256.R0255_K256_FOLDED_FLOOR
    assert folded_floor < band["lower"], "the disagreement window must be non-empty"
    planted = 0.5 * (folded_floor + band["lower"])  # 0.96589..., inside the window

    folded = k256.judge_folded_floor(planted, folded_floor)
    unfolded = k256.judge(planted, band)

    assert folded["verdict"] == "PASS"
    assert unfolded["verdict"] == "FAIL"
    assert unfolded["direction"] == "under"


def test_folded_floor_fails_a_cell_the_unfolded_band_passes(band):
    """The other half: review-0222's actual exact-seed48 case, carried.

    seed 48 reads ``r = 1.037``; its folded fidelity is ``1/1.037 = 0.9643...``,
    below the n = 8 folded floor ``0.9660625420699066``, so the fold failed it.
    On the unfolded log-ratio scale review-0222 measured ``z = +1.921`` -- not
    an outlier -- and the restored band passes it.
    """
    seed48 = 1.037
    assert k256.fold(seed48) == pytest.approx(0.9643201542912248, rel=0, abs=1e-12)
    assert k256.judge_folded_floor(seed48, R0222_N8_FOLDED_FLOOR)["verdict"] == "FAIL"
    assert k256.judge(seed48, band)["verdict"] == "PASS"


def test_the_fold_is_never_used_inside_the_criterion(family, band):
    """A cell and its reciprocal must not judge the same. A folded criterion
    cannot tell over- from under-separation; this one must."""
    over = band["upper"] * 1.02
    under = 1.0 / over
    assert k256.fold(over) == pytest.approx(k256.fold(under), rel=0, abs=1e-15)
    assert k256.judge(over, band)["direction"] == "over"
    assert k256.judge(under, band)["direction"] == "under"


# --------------------------------------------------------------------------
# Input hygiene
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_judge_refuses_a_non_positive_or_non_finite_ratio(band, bad):
    with pytest.raises(ValueError):
        k256.judge(bad, band)


def test_fit_band_refuses_a_degenerate_family():
    with pytest.raises(ValueError):
        k256.fit_band([1.0] * 29)  # MAD_n == 0
    with pytest.raises(ValueError):
        k256.fit_band([1.0])
    with pytest.raises(ValueError):
        k256.fit_band([1.0, -1.0, 1.0])


# --------------------------------------------------------------------------
# The would-be verdict table
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.path.exists(k256.SEALED["floors_n29"]["path"]),
    reason="sealed artifacts not reachable",
)
def test_would_be_verdicts_flip_exactly_the_expected_maps():
    table = k256.would_be_verdicts()
    flipped = {row["map"]: row["direction"] for row in table["flips"]}
    assert flipped == {
        "ladder-6250k-h2048-seed42": "over",
        "ladder-6250k-h2048-seed43": "over",
        "ladder-6250k-h2048-seed44": "over",
        "cluster-spill-c8-seed43": "inside",
    }
    assert table["flip_count"] == 4
    assert table["maps_scored"] == 45
    # the three rung maps flip PASS -> FAIL, in the over-separation direction
    for name in ("seed42", "seed43", "seed44"):
        row = next(r for r in table["rows"] if r["map"] == f"ladder-6250k-h2048-{name}")
        assert (row["one_sided_verdict"], row["two_sided_verdict"]) == ("PASS", "FAIL")
    # and the one held-out 2M cell flips the other way: R0260's one-sided floor
    # (k1 = 2.693) is TIGHTER on the under side than the band (k2 = 3.148)
    row = next(r for r in table["rows"] if r["map"] == "cluster-spill-c8-seed43")
    assert (row["one_sided_verdict"], row["two_sided_verdict"]) == ("FAIL", "PASS")
