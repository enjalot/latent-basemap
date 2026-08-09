"""R0234 — a floor that is outlier-invariant AND delivers its nominal coverage.

R0231 registered `median - 3*MAD_n` at `n = 13`. Review 0231-01 confirmed every
digit, confirmed the estimator is exactly outlier-invariant, and then measured
the thing R0231 declined to measure: the registered family delivers **88.4%
confidence against a nominal 95%**, with a new-cell false-fail rate of `1.98%`
against `mean - 2s`'s `3.90%`. The multiplier `3` was inherited from a Gaussian
consistency analogy (`MAD_n` estimates sigma, so `3*MAD_n` "is" `3*sigma`) that
does no work at `n = 13`, where the *sampling variability* of `MAD_n` — not its
expectation — decides coverage.

The multiplier is a free parameter. This module calibrates it.

**What "delivers its nominal coverage" means here.** A one-sided lower `95/95`
tolerance bound `L` must satisfy `P(L <= mu - z_0.95*sigma) = 0.95` — the same
definition the noncentral-t identity encodes for `L = xbar - k*s`, and the same
one review-0225-01 and review-0231-01 simulated. For any `centre - k*scale`
family that definition rearranges to a quantile:

    P(centre - k*scale <= mu - z*sigma) = P(k >= (centre - mu + z*sigma)/scale)

so the calibrated multiplier is exactly the `gamma`-quantile of
`(centre + z)/scale` over standard-normal families of size `n`. No search, no
tuning, no reference to any cell of this program. Two-sided bands are calibrated
against the standard content definition `Phi(U) - Phi(L) >= P`, which is what
Howe's factor approximates, by taking the `gamma`-quantile of the per-family
minimal `k`.

**Why a robust scale still needs its own multiplier.** `MAD_n` is consistent for
sigma but only ~37% efficient at the Gaussian, so at `n = 13` it is far more
variable than `s`; delivering the same confidence therefore costs a larger
multiplier. That is a fact about the estimator, not about this program's data,
and it is why `3` under-delivers.

**The candidates.** Five scale estimators plus the sample sd as the reference:
`MAD_n`; a 1-trimmed sample sd; Rousseeuw-Croux `S_n` and `Q_n` (1993), which buy
Gaussian efficiency 58% and 82% against `MAD_n`'s 37% at the same 50% breakdown
point; and a normal-consistent interquartile scale. Every one of them is
calibrated, then re-subjected to review-0225-01's injection test, because an
estimator that regains power by using more of the sample also regains the
self-loosening the exercise exists to remove.

**Attainability is derived, not asserted.** R0231 said the `(n-1)/sqrt(n)`
identity was "inapplicable" to a robust scale and exhibited witnesses that all
reduced to `MAD_n = 0`. The correct statement is a rank-slack argument: a scale
built from an order statistic of rank `r` out of `m` terms, of which at most `a`
involve any single cell, is bounded as that cell goes to `-inf` whenever
`m - a >= r`. Every robust candidate here satisfies that, so `sup |z| = +inf` and
**every** defining cell can fail at **any** multiplier. The 1-trimmed family does
not: its kept cells obey `(k-1)/sqrt(k)` on the *kept* subsample, so at a
calibrated multiplier only the two trimmed cells can ever fail.

`density_v2` stays descriptive-only (R0230's own extremes widened the `n = 8`
spread from `0.0114` to `0.0214`). Purity is gated two-sidedly on the unfolded
log-ratio, lower and upper, with raw ratios and direction — never folded.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from typing import Any

ROUND_ID = "0234"

GATE_CAPABILITY = "minilm-mixed-2m-calibrated-robust-floors-n13-v1"
GATE_SCHEMA = "round0234-minilm-mixed-2m-calibrated-robust-floors-n13-v1"

EXACT_FAMILY_SEEDS: tuple[int, ...] = (
    42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
)
CUVS_FAMILY_SEEDS: tuple[int, ...] = (42, 43, 44)
CANDIDATE_CLUSTER_COUNTS: tuple[int, ...] = (4, 8, 16)
CANDIDATE_SEEDS: tuple[int, ...] = (42, 43, 44)
N_EXACT = len(EXACT_FAMILY_SEEDS)
N_HELD_OUT = len(CUVS_FAMILY_SEEDS) + len(CANDIDATE_CLUSTER_COUNTS) * len(
    CANDIDATE_SEEDS
)

METRICS: tuple[str, ...] = (
    "density_v2",
    "ffr",
    "purity_fidelity_k256",
    "purity_fidelity_k1024",
)
GATED_METRICS: tuple[str, ...] = (
    "ffr",
    "purity_fidelity_k256",
    "purity_fidelity_k1024",
)
DESCRIPTIVE_METRICS: tuple[str, ...] = ("density_v2",)
PURITY_METRICS: tuple[str, ...] = ("purity_fidelity_k256", "purity_fidelity_k1024")
PURITY_RATIO_KEYS: dict[str, str] = {
    "purity_fidelity_k256": "k256",
    "purity_fidelity_k1024": "k1024",
}

SD_DDOF = 1
MAD_CONSISTENCY = 1.4826
SN_CONSISTENCY = 1.1926
QN_CONSISTENCY = 2.2219
#: `IQR / 1.34898` is the normal-consistent interquartile scale;
#: `1.3489795003921634 = 2 * Phi^-1(0.75)`.
IQR_CONSISTENCY = 1.3489795003921634
TRIM_COUNT = 1

TOLERANCE_CONTENT = 0.95
TOLERANCE_CONFIDENCE = 0.95

DENSITY_V2_DEFECT = (
    "density_v2 is DESCRIPTIVE ONLY and may not be used to fail a map. One anchor "
    "of 4,000 -- index 2846, substrate row 1449227 -- carries 1,377 duplicate "
    "coordinates and therefore r_hd == 0, and supplies about two-thirds of the "
    "metric's value; dropping it moved the eight-cell family mean from 0.4420625 "
    "to 0.15435 and left the cell ranking uncorrelated with the as-defined metric "
    "(Pearson -0.0465). R0230's thirteenth cell widened the family spread from "
    "0.0114 to 0.0214. Floors are computed and reported for it here and are not "
    "acceptance criteria (Review 0225-01, Review 0231-01)."
)


class Round0234Error(RuntimeError):
    """The registered R0234 calibrated-floor contract changed."""


# --------------------------------------------------------------------------- #
# the pre-registered selection rule
# --------------------------------------------------------------------------- #

#: Delivered one-sided and two-sided coverage must sit inside this band. The
#: multipliers are calibrated *to* 0.95 by construction, so this is a
#: verification that the calibration quantile did what it claims.
COVERAGE_TARGET = 0.95
COVERAGE_TOLERANCE = 0.005

#: Review 0225-01's injection: make a family member worse and re-fit. A floor
#: that moves at all has let the cell it should catch defend itself. Depth 1 is
#: the hard bar; the full ladder 1..6 is measured and reported for every
#: candidate on every series, in both tail directions for a two-sided band.
INVARIANCE_SIGMA_MULTIPLES: tuple[float, ...] = (1.0, 2.0, 3.0)
INVARIANCE_DEPTH_LADDER: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
REQUIRED_INVARIANCE_DEPTH = 1

#: Detection power is measured against a new cell drawn from `N(mu - 2*sigma,
#: sigma)`. Differences below one percentage point are treated as ties: the null
#: used to calibrate is Gaussian and the real families are not, so a
#: fraction-of-a-point ordering is not a fact about this program.
POWER_ALTERNATIVE_SIGMAS: tuple[float, ...] = (1.0, 2.0, 3.0)
POWER_SELECTION_ALTERNATIVE = 2.0
POWER_MATERIALITY = 0.01

SELECTION_RULE = (
    "PRE-REGISTERED, and evaluated on a Gaussian null that references no cell of "
    "this program. A candidate QUALIFIES iff all three hold. (1) COVERAGE: at its "
    "calibrated multiplier its delivered one-sided coverage P(floor <= population "
    "5th percentile) and its delivered two-sided content-0.95 confidence both sit "
    "in 0.95 +/- 0.005 at n = 13, and the calibration harness reproduces every "
    "published external reference value listed in EXTERNAL_CALIBRATION_TARGETS. "
    "(2) INVARIANCE: at its calibrated multiplier the floor moves EXACTLY 0.0 when "
    "the family's worst cell is made 1, 2 and 3 sample-sd worse -- Review "
    "0225-01's registered injection -- on every gated metric series and on both "
    "unfolded purity log-ratio series, and for a two-sided band on both bounds "
    "under contamination of the tail each bound faces. (3) ATTAINABILITY: at its "
    "calibrated multiplier EVERY defining cell must be able to fail, judged by the "
    "estimator's derived attainability bound on max (centre - x_i)/scale, and a "
    "witness family with a strictly POSITIVE scale estimate in which a defining "
    "cell does fail must exist. Among qualifying candidates, register the one with "
    "the highest delivered detection power against a new cell from N(mu - 2*sigma, "
    "sigma); candidates within 1.0 percentage point of the best are ties, broken "
    "first by the higher asymptotic breakdown point (a property known a priori, "
    "not measured on these cells), then by the smaller calibrated multiplier. If "
    "no candidate qualifies, register NOTHING and report the smallest n at which "
    "one would."
)

#: Published numbers this harness must reproduce before it is trusted. Sources
#: are review-0225-01 and review-0231-01 (independent reviewer simulations) and
#: two closed forms. A harness that reproduces an outside party's measurement is
#: externally checked rather than self-consistent.
EXTERNAL_CALIBRATION_TARGETS: tuple[dict[str, Any], ...] = (
    {"key": "n13_mean_minus_2s_coverage", "value": 0.734, "tolerance": 0.004,
     "source": "review-0231-2026-08-09-01.md section 5"},
    {"key": "n13_mean_minus_2s_false_fail", "value": 0.0390, "tolerance": 0.0010,
     "source": "review-0231-2026-08-09-01.md section 5"},
    {"key": "n13_one_sided_95_95_coverage", "value": 0.950, "tolerance": 0.004,
     "source": "review-0231-2026-08-09-01.md section 5"},
    {"key": "n13_one_sided_95_95_false_fail", "value": 0.0122, "tolerance": 0.0010,
     "source": "review-0231-2026-08-09-01.md section 5"},
    {"key": "n13_median_minus_3_madn_coverage", "value": 0.884, "tolerance": 0.004,
     "source": "review-0231-2026-08-09-01.md section 5"},
    {"key": "n13_median_minus_3_madn_false_fail", "value": 0.0198,
     "tolerance": 0.0010, "source": "review-0231-2026-08-09-01.md section 5"},
    {"key": "n13_two_sided_mean_2s_false_fail", "value": 0.0780, "tolerance": 0.0020,
     "source": "review-0231-2026-08-09-01.md section 5"},
    {"key": "n13_two_sided_median_3madn_false_fail", "value": 0.0395,
     "tolerance": 0.0020, "source": "review-0231-2026-08-09-01.md section 5"},
    {"key": "n8_mean_minus_2s_coverage", "value": 0.665, "tolerance": 0.005,
     "source": "review-0225-2026-08-08-01.md, re-reported in review-0231-01"},
    {"key": "n8_howe_two_sided_delivered", "value": 0.9518, "tolerance": 0.004,
     "source": "review-0231-2026-08-09-01.md section 1"},
    {"key": "n13_calibrated_sample_sd_k_vs_nct", "value": 0.0, "tolerance": 0.006,
     "source": "closed form: k = t'_{n-1,z*sqrt(n)}(gamma)/sqrt(n) = 2.6705039"},
    {"key": "n13_calibrated_sample_sd_k2_vs_howe", "value": 0.0, "tolerance": 0.025,
     "source": "closed form: Howe (1969) k2 = 3.1007995 at n = 13"},
)


# --------------------------------------------------------------------------- #
# scale estimators
# --------------------------------------------------------------------------- #


def _checked(values: Sequence[float]) -> list[float]:
    numbers = [float(value) for value in values]
    if len(numbers) < 3:
        raise Round0234Error("R0234 needs at least three cells to fit a floor")
    if any(not math.isfinite(value) for value in numbers):
        raise Round0234Error("R0234 refuses a nonfinite cell value")
    return numbers


def sample_sd(values: Sequence[float]) -> float:
    return statistics.stdev(_checked(values))


def mad_n(values: Sequence[float]) -> float:
    """`1.4826 * median|x - median|` — rank `ceil(n/2)` of `n` deviations."""
    numbers = _checked(values)
    centre = statistics.median(numbers)
    return MAD_CONSISTENCY * statistics.median(
        [abs(value - centre) for value in numbers]
    )


def iqr_n(values: Sequence[float]) -> float:
    """`(Q3 - Q1) / 1.34898`, type-7 quantiles (linear on the order statistics)."""
    numbers = sorted(_checked(values))
    size = len(numbers)

    def quantile(probability: float) -> float:
        position = probability * (size - 1)
        lower = int(math.floor(position))
        upper = min(lower + 1, size - 1)
        return numbers[lower] + (position - lower) * (numbers[upper] - numbers[lower])

    return (quantile(0.75) - quantile(0.25)) / IQR_CONSISTENCY


def _sn_correction(size: int) -> float:
    """Rousseeuw & Croux (1993) finite-sample factor for `S_n`."""
    table = {2: 0.743, 3: 1.851, 4: 0.954, 5: 1.351, 6: 0.993, 7: 1.198, 8: 1.005,
             9: 1.131}
    if size in table:
        return table[size]
    return size / (size - 0.9) if size % 2 else 1.0


def _qn_correction(size: int) -> float:
    """Rousseeuw & Croux (1993) finite-sample factor for `Q_n`."""
    table = {2: 0.399, 3: 0.994, 4: 0.512, 5: 0.844, 6: 0.611, 7: 0.857, 8: 0.669,
             9: 0.872}
    if size in table:
        return table[size]
    return size / (size + 1.4) if size % 2 else size / (size + 3.8)


def sn_scale(values: Sequence[float]) -> float:
    """`S_n = c_n * 1.1926 * lomed_i himed_j |x_i - x_j|` (Rousseeuw & Croux 1993)."""
    numbers = _checked(values)
    size = len(numbers)
    inner = []
    for left in numbers:
        distances = sorted(abs(left - right) for right in numbers)
        inner.append(distances[size // 2])
    inner.sort()
    return _sn_correction(size) * SN_CONSISTENCY * inner[(size + 1) // 2 - 1]


def qn_scale(values: Sequence[float]) -> float:
    """`Q_n = d_n * 2.2219 * {|x_i - x_j|; i<j}_(k)`, `k = C(floor(n/2)+1, 2)`."""
    numbers = _checked(values)
    size = len(numbers)
    pairs = sorted(
        abs(numbers[i] - numbers[j])
        for i in range(size)
        for j in range(i + 1, size)
    )
    half = size // 2 + 1
    rank = half * (half - 1) // 2
    return _qn_correction(size) * QN_CONSISTENCY * pairs[rank - 1]


def trimmed_centre_scale(values: Sequence[float]) -> tuple[float, float]:
    numbers = sorted(_checked(values))
    if len(numbers) <= 2 * TRIM_COUNT + 2:
        raise Round0234Error("R0234 trimmed floor needs more cells than it trims")
    kept = numbers[TRIM_COUNT : len(numbers) - TRIM_COUNT]
    return statistics.fmean(kept), statistics.stdev(kept)


def _mean_sd(values: Sequence[float]) -> tuple[float, float]:
    numbers = _checked(values)
    return statistics.fmean(numbers), statistics.stdev(numbers)


def _median_with(scale: Callable[[Sequence[float]], float]):
    def inner(values: Sequence[float]) -> tuple[float, float]:
        numbers = _checked(values)
        return statistics.median(numbers), scale(numbers)

    return inner


#: name -> (centre_and_scale, asymptotic breakdown point, robust?)
CANDIDATES: dict[str, dict[str, Any]] = {
    "mean_minus_k_sample_sd": {
        "centre_scale": _mean_sd,
        "centre": "mean",
        "scale_name": "sample sd (ddof=1)",
        "breakdown_point": 0.0,
        "robust": False,
        "gaussian_efficiency": 1.0,
    },
    "median_minus_k_madn": {
        "centre_scale": _median_with(mad_n),
        "centre": "median",
        "scale_name": "MAD_n = 1.4826 * median|x - median|",
        "breakdown_point": 0.5,
        "robust": True,
        "gaussian_efficiency": 0.37,
    },
    "trimmed1_mean_minus_k_trimmed_sd": {
        "centre_scale": trimmed_centre_scale,
        "centre": "mean of the middle n-2",
        "scale_name": "sample sd of the middle n-2",
        "breakdown_point": TRIM_COUNT / N_EXACT,
        "robust": True,
        "gaussian_efficiency": None,
    },
    "median_minus_k_sn": {
        "centre_scale": _median_with(sn_scale),
        "centre": "median",
        "scale_name": "S_n (Rousseeuw & Croux 1993)",
        "breakdown_point": 0.5,
        "robust": True,
        "gaussian_efficiency": 0.58,
    },
    "median_minus_k_qn": {
        "centre_scale": _median_with(qn_scale),
        "centre": "median",
        "scale_name": "Q_n (Rousseeuw & Croux 1993)",
        "breakdown_point": 0.5,
        "robust": True,
        "gaussian_efficiency": 0.82,
    },
    "median_minus_k_iqrn": {
        "centre_scale": _median_with(iqr_n),
        "centre": "median",
        "scale_name": "IQR / 1.34898 (type-7 quantiles)",
        "breakdown_point": 0.25,
        "robust": True,
        "gaussian_efficiency": 0.37,
    },
}
CANDIDATE_ORDER: tuple[str, ...] = tuple(CANDIDATES)
ROBUST_CANDIDATES: tuple[str, ...] = tuple(
    name for name, item in CANDIDATES.items() if item["robust"]
)


def centre_and_scale(name: str, values: Sequence[float]) -> tuple[float, float]:
    if name not in CANDIDATES:
        raise Round0234Error(f"R0234 has no candidate {name!r}")
    return CANDIDATES[name]["centre_scale"](values)


def floor_at(name: str, values: Sequence[float], multiplier: float) -> float:
    centre, scale = centre_and_scale(name, values)
    return centre - float(multiplier) * scale


def band_at(
    name: str, logs: Sequence[float], multiplier: float
) -> tuple[float, float]:
    centre, scale = centre_and_scale(name, logs)
    return centre - float(multiplier) * scale, centre + float(multiplier) * scale


# --------------------------------------------------------------------------- #
# attainability, derived rather than asserted
# --------------------------------------------------------------------------- #


def identity_bound(n: int) -> float:
    """`(n-1)/sqrt(n)`: the largest `|x_i - xbar|/s` any sample point can reach."""
    if int(n) < 2:
        raise Round0234Error("R0234 identity bound needs at least two cells")
    return (int(n) - 1) / math.sqrt(int(n))


def rank_slack_bound(*, terms: int, rank: int, terms_touched_by_one_cell: int) -> bool:
    """Is an order statistic of rank `rank` out of `terms` immune to one cell?

    A scale built as the `rank`-th smallest of `terms` quantities, of which at
    most `terms_touched_by_one_cell` involve any single cell, cannot be driven to
    infinity by that cell whenever `terms - touched >= rank`: the remaining terms
    already supply `rank` values no larger than the ones that moved. If the scale
    stays bounded while `|centre - x_i|` grows without limit, then
    `max_i (centre - x_i)/scale` is unbounded and **no** multiplier can make the
    floor unfailable by a defining cell.
    """
    return int(terms) - int(terms_touched_by_one_cell) >= int(rank)


def attainability(name: str, *, n: int, multiplier: float) -> dict[str, Any]:
    """The bound that decides whether a defining cell CAN fail, per estimator."""
    n = int(n)
    multiplier = float(multiplier)
    if name == "mean_minus_k_sample_sd":
        bound = identity_bound(n)
        return {
            "estimator": name,
            "n": n,
            "multiplier": multiplier,
            "max_abs_z_bound": bound,
            "bound_is_finite": True,
            "cells_that_can_fail": n if multiplier < bound else 0,
            "every_defining_cell_can_fail": multiplier < bound,
            "derivation": (
                f"max_i |x_i - xbar| / s <= (n-1)/sqrt(n) = {bound!r} at n = {n}; "
                "every point inflates s, so the bound is attained by the sharp "
                f"family [1]*({n}-1) + [0]. The multiplier is {multiplier!r}."
            ),
        }
    if name == "trimmed1_mean_minus_k_trimmed_sd":
        kept = n - 2 * TRIM_COUNT
        bound = identity_bound(kept)
        kept_can_fail = multiplier < bound
        return {
            "estimator": name,
            "n": n,
            "multiplier": multiplier,
            "kept_subsample": kept,
            "max_abs_z_bound": bound,
            "bound_is_finite": True,
            "cells_that_can_fail": (n if kept_can_fail else 2 * TRIM_COUNT),
            "every_defining_cell_can_fail": kept_can_fail,
            "derivation": (
                "the floor is fitted to the kept subsample only, so a KEPT cell "
                f"obeys the same identity on {kept} points: "
                f"max |x - mean_kept| / s_kept <= ({kept}-1)/sqrt({kept}) = "
                f"{bound!r}. At multiplier {multiplier!r} a kept cell "
                + ("CAN" if kept_can_fail else "CANNOT")
                + f" fall below the floor, so only the {2 * TRIM_COUNT} trimmed "
                "cells remain able to fail; the trimmed cells are unbounded "
                "because they enter neither the centre nor the scale."
            ),
        }
    if name == "median_minus_k_madn":
        terms, rank, touched = n, (n + 1) // 2, 1
    elif name == "median_minus_k_iqrn":
        terms, rank, touched = n, int(math.floor(0.75 * (n - 1))) + 2, 1
    elif name == "median_minus_k_sn":
        terms, rank, touched = n, (n + 1) // 2, 1
    elif name == "median_minus_k_qn":
        half = n // 2 + 1
        terms, rank, touched = n * (n - 1) // 2, half * (half - 1) // 2, n - 1
    else:
        raise Round0234Error(f"R0234 has no attainability rule for {name!r}")
    bounded = rank_slack_bound(
        terms=terms, rank=rank, terms_touched_by_one_cell=touched
    )
    if not bounded:
        raise Round0234Error(
            f"R0234 {name} scale is not one-cell bounded at n = {n}; the "
            "attainability argument below does not hold and must be re-derived"
        )
    return {
        "estimator": name,
        "n": n,
        "multiplier": multiplier,
        "max_abs_z_bound": None,
        "max_abs_z_bound_note": "+inf — unbounded, see the derivation below",
        "bound_is_finite": False,
        "rank_slack": {
            "terms": terms,
            "rank": rank,
            "terms_touched_by_one_cell": touched,
            "unaffected_terms": terms - touched,
        },
        "cells_that_can_fail": n,
        "every_defining_cell_can_fail": True,
        "derivation": (
            f"the scale is the rank-{rank} order statistic of {terms} quantities, "
            f"of which at most {touched} involve any single cell; "
            f"{terms} - {touched} = {terms - touched} >= {rank}, so the scale stays "
            "bounded as that cell goes to -inf while |centre - x_i| grows without "
            "limit (the centre is itself a middle order statistic and is bounded "
            "for the same reason). Hence sup max_i (centre - x_i)/scale = +inf and "
            "EVERY defining cell can fall below this floor at ANY multiplier. This "
            "replaces R0231's assertion that the (n-1)/sqrt(n) identity is merely "
            "'inapplicable'."
        ),
    }


#: R0231's five witnesses were all the same vector; for the robust families it
#: works only because the scale estimate is exactly zero, which is the estimator's
#: degenerate case rather than a demonstration of failability at real dispersion.
DEGENERATE_WITNESS: tuple[float, ...] = tuple([1.0] * (N_EXACT - 1) + [0.0])


def positive_scale_witness(
    name: str, multiplier: float, *, n: int = N_EXACT
) -> dict[str, Any]:
    """A witness family with a strictly positive scale in which a cell fails."""
    spread = [1.0 + 0.01 * index for index in range(n - 1)]
    values = [-1.0e6] + spread
    centre, scale = centre_and_scale(name, values)
    floor = centre - float(multiplier) * scale
    lowest = min(values)
    return {
        "estimator": name,
        "multiplier": float(multiplier),
        "n": int(n),
        "values": values,
        "centre": centre,
        "scale_value": scale,
        "scale_is_strictly_positive": scale > 0.0,
        "floor": floor,
        "lowest_cell": lowest,
        "lowest_cell_fails_its_own_floor": lowest < floor,
    }


def degenerate_witness(name: str, multiplier: float) -> dict[str, Any]:
    """R0231's witness, reported so its degeneracy is on the record."""
    values = list(DEGENERATE_WITNESS)
    centre, scale = centre_and_scale(name, values)
    floor = centre - float(multiplier) * scale
    return {
        "estimator": name,
        "multiplier": float(multiplier),
        "values": values,
        "scale_value": scale,
        "scale_is_strictly_positive": scale > 0.0,
        "floor": floor,
        "lowest_cell": min(values),
        "lowest_cell_fails_its_own_floor": min(values) < floor,
        "note": (
            "this is the vector R0231 exhibited for all five families. For the "
            "robust ones it works only because the scale estimate is exactly 0, so "
            "it demonstrates the estimator's degenerate case, not failability at "
            "positive dispersion. Review 0231-01 blocked the claim that it did."
        ),
    }


# --------------------------------------------------------------------------- #
# invariance: Review 0225-01's injection, deepened
# --------------------------------------------------------------------------- #


def injection_ladder(
    name: str,
    values: Sequence[float],
    multiplier: float,
    *,
    side: str = "lower",
    depths: Sequence[int] = INVARIANCE_DEPTH_LADDER,
    sigma_multiples: Sequence[float] = INVARIANCE_SIGMA_MULTIPLES,
) -> dict[str, Any]:
    """Make the `d` worst cells worse by `m` sample-sd each and re-fit.

    `side = "lower"` contaminates the low tail and tracks the lower bound; the
    upper bound of a two-sided band faces the opposite tail, so `side = "upper"`
    contaminates the high tail and tracks the upper bound. A floor that moves has
    let the cell it should catch defend itself — the defect R0231 identified and
    this round must not reintroduce while chasing power.
    """
    base = _checked(values)
    sd = statistics.stdev(base)
    order = sorted(range(len(base)), key=lambda index: base[index])
    if side == "upper":
        order = list(reversed(order))
    sign = -1.0 if side == "lower" else 1.0

    def bound_of(sample: Sequence[float]) -> float:
        centre, scale = centre_and_scale(name, sample)
        return centre + sign * float(multiplier) * scale

    baseline = bound_of(base)
    rows: list[dict[str, Any]] = []
    for depth in depths:
        for multiple in sigma_multiples:
            injected = list(base)
            for index in order[: int(depth)]:
                injected[index] = base[index] + sign * float(multiple) * sd
            moved = bound_of(injected)
            shift = moved - baseline
            rows.append({
                "depth": int(depth),
                "injected_sigma_multiple": float(multiple),
                "bound_before": baseline,
                "bound_after": moved,
                "shift": shift,
                "invariant": shift == 0.0,
                "loosened": (shift < 0.0) if side == "lower" else (shift > 0.0),
            })
    exact_depth = 0
    for depth in depths:
        at_depth = [row for row in rows if row["depth"] == int(depth)]
        if all(row["invariant"] for row in at_depth):
            exact_depth = int(depth)
        else:
            break
    return {
        "estimator": name,
        "side": side,
        "n": len(base),
        "sample_sd_ddof1": sd,
        "multiplier": float(multiplier),
        "baseline_bound": baseline,
        "rows": rows,
        "exact_invariance_depth": exact_depth,
        "invariant_at_depth_1": exact_depth >= 1,
        "self_loosening_at_any_depth": any(row["loosened"] for row in rows),
    }


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def score_cell_metric(
    *,
    value: float,
    floor: float | None,
    ratio: float | None = None,
    band: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """One cell, one metric. A banded metric is decided on the unfolded ratio."""
    entry: dict[str, Any] = {"value": float(value)}
    if band is not None:
        if ratio is None:
            raise Round0234Error("R0234 banded metric needs its raw ratio")
        lower, upper = float(band[0]), float(band[1])
        inside = lower <= float(ratio) <= upper
        entry.update({
            "decided_on": "raw unfolded purity ratio, two-sided band",
            "raw_ratio": float(ratio),
            "log_ratio": math.log(float(ratio)),
            "ratio_lower": lower,
            "ratio_upper": upper,
            "descriptive_folded_floor": (float(floor) if floor is not None else None),
            "passes": inside,
            "band_direction": (
                "below_band" if float(ratio) < lower
                else "above_band" if float(ratio) > upper
                else "inside_band"
            ),
            "separation": (
                "over_separating" if float(ratio) > 1.0
                else "under_separating" if float(ratio) < 1.0
                else "exact"
            ),
        })
        return entry
    if floor is None:
        raise Round0234Error("R0234 unbanded metric needs a floor")
    entry.update({
        "decided_on": "one-sided lower floor",
        "floor": float(floor),
        "margin": float(value) - float(floor),
        "passes": float(value) >= float(floor),
        "direction": "below_floor" if float(value) < float(floor) else "at_or_above",
    })
    return entry


def score_population(
    *,
    cells: Sequence[Mapping[str, Any]],
    floors: Mapping[str, float],
    bands: Mapping[str, tuple[float, float]],
    metrics: Sequence[str],
    defining_cell_ids: Sequence[str],
    every_defining_cell_can_fail: bool,
) -> dict[str, Any]:
    defining = {str(item) for item in defining_cell_ids}
    rows: list[dict[str, Any]] = []
    for cell in cells:
        cell_id = str(cell["cell_id"])
        row: dict[str, Any] = {
            "cell_id": cell_id,
            "family": str(cell["family"]),
            "defines_the_floors": cell_id in defining,
            "metrics": {},
        }
        for metric in metrics:
            if metric not in floors and metric not in bands:
                continue
            band = bands.get(metric)
            ratio = None
            if band is not None:
                ratio = float(cell["ratios"][PURITY_RATIO_KEYS[metric]])
            row["metrics"][metric] = score_cell_metric(
                value=float(cell["values"][metric]),
                floor=floors.get(metric),
                ratio=ratio,
                band=band,
            )
        row["clears_every_gated_metric"] = all(
            entry["passes"] for entry in row["metrics"].values()
        )
        rows.append(row)
    failures = [
        {
            "cell_id": row["cell_id"],
            "metric": metric,
            "defines_the_floors": row["defines_the_floors"],
            **{
                key: entry[key]
                for key in entry
                if key not in {"passes"}
            },
        }
        for row in rows
        for metric, entry in row["metrics"].items()
        if not entry["passes"]
    ]
    non_defining = sum(1 for row in rows if not row["defines_the_floors"])
    return {
        "cells_scored": len(rows),
        "cells_clearing_every_gated_metric": sum(
            1 for row in rows if row["clears_every_gated_metric"]
        ),
        "failures": len(failures),
        "failing_cell_metrics": failures,
        "defining_cells_scored": len(rows) - non_defining,
        "non_defining_cells_scored": non_defining,
        "every_defining_cell_can_fail": bool(every_defining_cell_can_fail),
        "count_is_attainable": bool(non_defining) or bool(every_defining_cell_can_fail),
        "cells": rows,
    }


def verdict_changes(
    *,
    chosen: Mapping[str, Any],
    published: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Every cell-metric verdict this registration would change, named.

    R0231 silently reversed R0228's released `c8-seed42` `ffr` failure. Nothing
    here is allowed to be silent: for every previously published family, every
    cell-metric whose pass/fail differs is listed, and the direction that unmakes
    a published failure is called out separately.
    """
    by_cell_metric: dict[str, dict[str, bool]] = {}
    for row in chosen["cells"]:
        for metric, entry in row["metrics"].items():
            by_cell_metric[f"{row['cell_id']}::{metric}"] = bool(entry["passes"])
    changes: list[dict[str, Any]] = []
    for family_name, scoring in published.items():
        for row in scoring["cells"]:
            for metric, entry in row["metrics"].items():
                key = f"{row['cell_id']}::{metric}"
                if key not in by_cell_metric:
                    continue
                was = bool(entry["passes"])
                now = by_cell_metric[key]
                if was == now:
                    continue
                changes.append({
                    "cell_id": row["cell_id"],
                    "metric": metric,
                    "published_family": family_name,
                    "published_verdict": "pass" if was else "FAIL",
                    "verdict_under_this_registration": "pass" if now else "FAIL",
                    "direction": (
                        "un-fails a published failure" if (not was and now)
                        else "fails a cell a published family passed"
                    ),
                })
    return {
        "changes": changes,
        "count": len(changes),
        "un_failed_published_failures": [
            item for item in changes
            if item["direction"] == "un-fails a published failure"
        ],
        "newly_failed_cells": [
            item for item in changes
            if item["direction"] == "fails a cell a published family passed"
        ],
    }


__all__ = [
    "CANDIDATES",
    "CANDIDATE_CLUSTER_COUNTS",
    "CANDIDATE_ORDER",
    "CANDIDATE_SEEDS",
    "COVERAGE_TARGET",
    "COVERAGE_TOLERANCE",
    "CUVS_FAMILY_SEEDS",
    "DEGENERATE_WITNESS",
    "DENSITY_V2_DEFECT",
    "DESCRIPTIVE_METRICS",
    "EXACT_FAMILY_SEEDS",
    "EXTERNAL_CALIBRATION_TARGETS",
    "GATED_METRICS",
    "GATE_CAPABILITY",
    "GATE_SCHEMA",
    "INVARIANCE_DEPTH_LADDER",
    "INVARIANCE_SIGMA_MULTIPLES",
    "IQR_CONSISTENCY",
    "MAD_CONSISTENCY",
    "METRICS",
    "N_EXACT",
    "N_HELD_OUT",
    "POWER_ALTERNATIVE_SIGMAS",
    "POWER_MATERIALITY",
    "POWER_SELECTION_ALTERNATIVE",
    "PURITY_METRICS",
    "PURITY_RATIO_KEYS",
    "QN_CONSISTENCY",
    "REQUIRED_INVARIANCE_DEPTH",
    "ROBUST_CANDIDATES",
    "ROUND_ID",
    "Round0234Error",
    "SD_DDOF",
    "SELECTION_RULE",
    "SN_CONSISTENCY",
    "TOLERANCE_CONFIDENCE",
    "TOLERANCE_CONTENT",
    "TRIM_COUNT",
    "attainability",
    "band_at",
    "centre_and_scale",
    "degenerate_witness",
    "floor_at",
    "identity_bound",
    "injection_ladder",
    "iqr_n",
    "mad_n",
    "positive_scale_witness",
    "qn_scale",
    "rank_slack_bound",
    "sample_sd",
    "score_cell_metric",
    "score_population",
    "sn_scale",
    "trimmed_centre_scale",
    "verdict_changes",
]
