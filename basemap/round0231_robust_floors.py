"""R0231 — register floors a defining cell can actually fail, on a robust scale.

Three findings, all measured and all binding in `guides/plan-minilm-100m-v2.md`,
determine everything this module does.

**1. The identity, not the multiplier.** For any sample of size `n`,
`max_i |x_i - xbar| / s <= (n-1)/sqrt(n)`. That bound is `1.1547` at `n = 3`,
`1.5` at `n = 4`, `2.4749` at `n = 8`, `3.3282` at `n = 13`, and it first exceeds
`2` only at `n = 6`. So R0193 (`n = 3`), R0161/R0219 (`n = 4`) and R0225
(`n = 8`, `k = 3.1873`) all reported pass counts that were **theorems**: no cell
that helped define those floors could have fallen below one. At `n = 13` the
bound exceeds `2.0`, the one-sided 95/95 factor `2.6705` and Howe's two-sided
`3.1008`, so every `mean - k*s` family below is one a defining cell can fail.

**2. The scale estimator is the defect, not the multiplier.** `mean - k*s` is
self-loosening for *any* `k` — an outlier inflates `s` and therefore lowers the
floor, so the cell a gate should catch partly defends itself — and the 95/95
floor moves `1.47-1.50x` FURTHER than `mean - 2s` under the same injection,
because `k > 2`. But "no variance-estimated floor is outlier-robust" is false:
Review 0225 measured `median - 3*MAD_n`, a 1-trimmed `mean - 2s` and a fixed
margin off the median all moving **exactly 0.0** under a 3-sigma injection. This
module measures all of them again, on the `n = 13` family, and registers one.

**3. `density_v2` is not a sound acceptance criterion.** One anchor of `4,000`
(index `2846`, substrate row `1449227`, `1,377` duplicate coordinates, `r_hd == 0`)
supplies about two-thirds of its value; dropping it moves the family mean
`0.4421 -> 0.1544` and leaves the cell ranking correlated at Pearson `-0.047`. It
is carried **descriptive-only** here: floors are computed for it and explicitly
may not be used to fail a map.

**Purity is gated two-sidedly on the unfolded log-ratio.** `purity_fidelity` is
`exp(-|log r|)`, a fold about `r = 1.0` that the family is not centred on: it
manufactures `|z|` and, worse, destroys the *direction*, so over-separation and
under-separation become the same failure. R0223's cuVS cells over-separate at
`k256` while R0228's candidate arms under-separate at `k256` — a folded floor
cannot tell those apart. Every purity band here carries a lower AND an upper
bound, and every cell is reported with its raw ratio and its direction.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

ROUND_ID = "0231"

GATE_CAPABILITY = "minilm-mixed-2m-robust-floors-n13-v1"
GATE_SCHEMA = "round0231-minilm-mixed-2m-robust-floors-n13-v1"

#: The thirteen cells that define the floors.
EXACT_FAMILY_SEEDS: tuple[int, ...] = (
    42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
)
#: Held out: they define nothing, so they are the only cells that can genuinely
#: test a floor built from the exact family.
CUVS_FAMILY_SEEDS: tuple[int, ...] = (42, 43, 44)
CANDIDATE_CLUSTER_COUNTS: tuple[int, ...] = (4, 8, 16)
CANDIDATE_SEEDS: tuple[int, ...] = (42, 43, 44)

N_EXACT = len(EXACT_FAMILY_SEEDS)

#: FFR and both purity fidelities are gated. `density_v2` is computed and
#: reported and may **not** be used to fail a map (Review 0225; binding rule).
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
PURITY_METRICS: tuple[str, ...] = (
    "purity_fidelity_k256",
    "purity_fidelity_k1024",
)
PURITY_RATIO_KEYS: dict[str, str] = {
    "purity_fidelity_k256": "k256",
    "purity_fidelity_k1024": "k1024",
}

DENSITY_V2_DEFECT = (
    "density_v2 is DESCRIPTIVE ONLY. Anchor index 2846 / substrate row 1449227 "
    "has 1,377 duplicate coordinates and therefore r_hd == 0; with eps = 1e-12 it "
    "sits at log r = -27.63 in both variables and supplies about two-thirds of "
    "the metric's value. Dropping it moves the eight-cell family mean from "
    "0.4420625 to 0.15435, moves 2*sd/mean from 1.93% to 26.03%, and leaves the "
    "cell ranking uncorrelated with the as-defined metric (Pearson -0.0465, "
    "Spearman -0.1429). Floors are computed for it here and may not be used to "
    "fail a map until the anchor set excludes degenerate r_hd == 0 rows and the "
    "metric is re-registered from a fresh family (Review 0225)."
)

SD_DDOF = 1
LEGACY_MULTIPLIER = 2.0

#: Normal-consistency constant: `1.4826 * median|x - median|` estimates sigma for
#: Gaussian data, so `median - 3*MAD_n` is `mean - 3*sigma` in the Gaussian limit.
MAD_CONSISTENCY = 1.4826
MAD_MULTIPLIER = 3.0
TRIM_COUNT = 1
TRIMMED_MULTIPLIER = 2.0
FIXED_MARGIN_FRACTION = 0.02

TOLERANCE_CONTENT = 0.95
TOLERANCE_CONFIDENCE = 0.95

#: The estimator this round registers, and why. Recorded here so the choice is
#: part of the preregistered contract rather than prose written afterwards.
CHOSEN_ESTIMATOR = "median_minus_3_mad"
CHOSEN_ESTIMATOR_JUSTIFICATION = (
    "median - 3*MAD_n is the only candidate that is BOTH outlier-invariant and "
    "derived from the family's own dispersion. (a) Centre and scale each have a "
    "50% breakdown point, so the floor survives up to six contaminated cells out "
    "of thirteen; the 1-trimmed mean survives exactly one per tail and is "
    "defeated by two, which is the wrong guarantee for a gate that will judge "
    "unknown future cells. (b) A fixed margin off the median is also exactly "
    "outlier-invariant but its margin is a free parameter unrelated to the data: "
    "2% of the median is a different number of sample sd on each metric, so it "
    "is arbitrarily strict on tight metrics and arbitrarily lax on loose ones. "
    "(c) MAD_n carries the 1.4826 normal-consistency factor, so median - 3*MAD_n "
    "equals mean - 3*sigma in the Gaussian limit -- a STRICTER multiplier than "
    "R0222's 2.0 and than the one-sided 95/95 factor 2.6705 at n = 13 -- which "
    "means adopting it is not a weakening dressed up as a robustness fix. The "
    "realised ratio 3*MAD_n / s is measured and published per metric so a reader "
    "can see where the robust floor actually sits relative to mean - 2s on THIS "
    "family, including where it is stricter and fails a cell mean - 2s passed."
)

FAMILY_DEFINITIONS = (
    "mean_minus_2sd: R0219/R0222's method, floor = mean - 2*s (ddof=1). "
    "one_sided_tolerance_95_95: R0225's method, floor = mean - k*s with k the "
    "derived one-sided 95/95 normal tolerance factor at this n. "
    "median_minus_3_mad: floor = median - 3*MAD_n, MAD_n = 1.4826 * "
    "median|x - median|. "
    "trimmed1_mean_minus_2s: floor = mean(trimmed) - 2*s(trimmed) after removing "
    "the single lowest and single highest cell. "
    "median_fixed_margin: floor = median * (1 - 0.02). "
    "Every purity metric additionally carries a TWO-SIDED band on the unfolded "
    "log-ratio, so over-separation and under-separation are distinguishable."
)

ONE_SIDED_DERIVATION = (
    "A one-sided lower tolerance bound L = xbar - k*s must satisfy "
    "P(L <= mu - z_P*sigma) >= gamma. Rearranging, "
    "(xbar-mu)/(sigma/sqrt(n)) + z_P*sqrt(n) <= k*sqrt(n)*(s/sigma); the left "
    "numerator is N(delta,1) with delta = z_P*sqrt(n) and s/sigma = "
    "sqrt(chi2_{n-1}/(n-1)) independently, so the ratio is a noncentral t with "
    "df = n-1 and noncentrality delta, giving k = t'_{n-1,delta}(gamma)/sqrt(n). "
    "The identity is valid ONLY for the sample standard deviation; it is not "
    "applied to any robust scale here."
)
TWO_SIDED_DERIVATION = (
    "Howe (1969): k2 = z_{(1+P)/2} * sqrt(1+1/n) * sqrt((n-1)/chi2_{1-gamma,n-1}) "
    "* sqrt(1 + (n-3-chi2_{1-gamma,n-1})/(2*(n+1)^2)). Registered here as the "
    "comparison family. Review 0225-01 measured its delivered coverage at n = 8 "
    "as 0.9518 against a nominal 0.95, i.e. slightly conservative; the claim that "
    "R0225 confirmed it by simulation was false and R0225's artifact should not "
    "be read as carrying such a simulation."
)

INJECTION_SIGMA_MULTIPLES: tuple[float, ...] = (1.0, 2.0, 3.0)


class Round0231Error(RuntimeError):
    """The registered R0231 robust-floor contract changed."""


# --------------------------------------------------------------------------- #
# the identity, and whether it applies
# --------------------------------------------------------------------------- #


def identity_bound(n: int) -> float:
    """`(n-1)/sqrt(n)`: the largest `|x_i - xbar|/s` any sample point can reach."""
    if int(n) < 2:
        raise Round0231Error("R0231 identity bound needs at least two cells")
    return (int(n) - 1) / math.sqrt(int(n))


def defining_cell_can_fail(
    *, n: int, multiplier: float | None, scale: str
) -> dict[str, Any]:
    """Can a cell that helped define this floor fall below it? Answer, with why.

    For a `mean - k*s` floor the answer is purely arithmetic: yes iff
    `k < (n-1)/sqrt(n)`. For a robust floor the identity is **inapplicable**,
    because it bounds deviation in units of the sample sd — a statistic every
    point inflates — whereas `MAD_n` depends only on the middle of the sample and
    an extreme cell can be arbitrarily far below the median in MAD units. That is
    exactly the property that makes the robust floor falsifiable by its own
    family at any `n`, and `witness_defining_cell_failure` exhibits one.
    """
    bound = identity_bound(n)
    if scale == "sample_sd":
        if multiplier is None:
            raise Round0231Error("R0231 mean-k*s family needs its multiplier")
        can = float(multiplier) < bound
        return {
            "n": int(n),
            "scale": scale,
            "multiplier": float(multiplier),
            "identity_bound": bound,
            "identity_applies": True,
            "defining_cell_can_fail": bool(can),
            "reason": (
                f"max|x - xbar|/s <= (n-1)/sqrt(n) = {bound!r} at n = {int(n)}; "
                f"the multiplier is {float(multiplier)!r}, so a defining cell "
                + ("CAN" if can else "CANNOT")
                + " fall below this floor for any values whatsoever"
            ),
        }
    return {
        "n": int(n),
        "scale": scale,
        "multiplier": float(multiplier) if multiplier is not None else None,
        "identity_bound": bound,
        "identity_applies": False,
        "defining_cell_can_fail": True,
        "reason": (
            "the (n-1)/sqrt(n) identity bounds deviation in units of the SAMPLE "
            "SD, which every point inflates. This floor's scale is robust and "
            "depends only on the middle of the sample, so |x_i - centre| / scale "
            "is unbounded and a defining cell can fall below the floor at any n. "
            "witness_defining_cell_failure() exhibits an explicit 13-cell family "
            "in which one does."
        ),
    }


def witness_defining_cell_failure(estimator: str, n: int = N_EXACT) -> dict[str, Any]:
    """An explicit family in which a defining cell falls below its own floor.

    A bound argument is not a demonstration. This constructs `n` values, fits the
    floor to all of them, and shows the lowest one below it — so the claim
    "a defining cell can fail" is a witnessed fact for every registered family,
    not an inference from an inequality.
    """
    values = [1.0] * (int(n) - 1) + [0.0]
    floor = float(FLOOR_ESTIMATORS[estimator](values)["floor"])
    lowest = min(values)
    return {
        "estimator": estimator,
        "n": int(n),
        "values": values,
        "floor": floor,
        "lowest_cell": lowest,
        "lowest_cell_fails_its_own_floor": lowest < floor,
    }


# --------------------------------------------------------------------------- #
# scale estimators
# --------------------------------------------------------------------------- #


def _checked(values: Sequence[float]) -> list[float]:
    numbers = [float(value) for value in values]
    if len(numbers) < 3:
        raise Round0231Error("R0231 needs at least three cells to fit a floor")
    if any(not math.isfinite(value) for value in numbers):
        raise Round0231Error("R0231 refuses a nonfinite cell value")
    return numbers


def mad_n(values: Sequence[float]) -> float:
    """`1.4826 * median|x - median|`: the normal-consistent median absolute deviation."""
    numbers = _checked(values)
    centre = statistics.median(numbers)
    return MAD_CONSISTENCY * statistics.median(
        [abs(value - centre) for value in numbers]
    )


def mean_minus_2sd_floor(values: Sequence[float]) -> dict[str, Any]:
    numbers = _checked(values)
    mean = statistics.fmean(numbers)
    sd = statistics.stdev(numbers)
    return {
        "family": "mean_minus_2sd",
        "scale": "sample_sd",
        "n": len(numbers),
        "centre": mean,
        "scale_value": sd,
        "multiplier": LEGACY_MULTIPLIER,
        "floor": mean - LEGACY_MULTIPLIER * sd,
    }


def one_sided_tolerance_factor(
    n: int,
    *,
    content: float = TOLERANCE_CONTENT,
    confidence: float = TOLERANCE_CONFIDENCE,
) -> dict[str, Any]:
    """`k = t'_{n-1, z_P*sqrt(n)}(gamma) / sqrt(n)`, derived rather than quoted."""
    from scipy import stats

    n = int(n)
    if n < 3:
        raise Round0231Error("R0231 tolerance factor needs n >= 3")
    z_content = float(stats.norm.ppf(content))
    noncentrality = z_content * math.sqrt(n)
    k = float(stats.nct.ppf(confidence, n - 1, noncentrality)) / math.sqrt(n)
    return {
        "n": n,
        "content": float(content),
        "confidence": float(confidence),
        "z_content": z_content,
        "noncentrality": noncentrality,
        "degrees_of_freedom": n - 1,
        "k": k,
        "derivation": ONE_SIDED_DERIVATION,
    }


def two_sided_tolerance_factor(
    n: int,
    *,
    content: float = TOLERANCE_CONTENT,
    confidence: float = TOLERANCE_CONFIDENCE,
) -> dict[str, Any]:
    """Howe's two-sided factor `k2` for `[mean - k2*s, mean + k2*s]`."""
    from scipy import stats

    n = int(n)
    if n < 3:
        raise Round0231Error("R0231 tolerance factor needs n >= 3")
    z = float(stats.norm.ppf((1.0 + content) / 2.0))
    chi = float(stats.chi2.ppf(1.0 - confidence, n - 1))
    u = z * math.sqrt(1.0 + 1.0 / n)
    v = math.sqrt((n - 1) / chi)
    correction = math.sqrt(1.0 + (n - 3.0 - chi) / (2.0 * (n + 1.0) ** 2))
    return {
        "n": n,
        "content": float(content),
        "confidence": float(confidence),
        "z_half_content": z,
        "chi2_lower": chi,
        "k2": u * v * correction,
        "method": "Howe (1969)",
        "derivation": TWO_SIDED_DERIVATION,
    }


def one_sided_tolerance_floor(values: Sequence[float]) -> dict[str, Any]:
    numbers = _checked(values)
    mean = statistics.fmean(numbers)
    sd = statistics.stdev(numbers)
    factor = one_sided_tolerance_factor(len(numbers))
    return {
        "family": "one_sided_tolerance_95_95",
        "scale": "sample_sd",
        "n": len(numbers),
        "centre": mean,
        "scale_value": sd,
        "multiplier": factor["k"],
        "factor": factor,
        "floor": mean - factor["k"] * sd,
    }


def median_minus_3_mad_floor(values: Sequence[float]) -> dict[str, Any]:
    numbers = _checked(values)
    centre = statistics.median(numbers)
    scale = mad_n(numbers)
    sd = statistics.stdev(numbers)
    return {
        "family": "median_minus_3_mad",
        "scale": "mad_n",
        "n": len(numbers),
        "centre": centre,
        "scale_value": scale,
        "multiplier": MAD_MULTIPLIER,
        "floor": centre - MAD_MULTIPLIER * scale,
        "mad_n_over_sample_sd": (scale / sd) if sd else None,
        "effective_sigma_multiplier": (
            (statistics.fmean(numbers) - (centre - MAD_MULTIPLIER * scale)) / sd
            if sd
            else None
        ),
        "note": (
            "MAD_n carries the 1.4826 normal-consistency factor, so this floor is "
            "mean - 3*sigma in the Gaussian limit. effective_sigma_multiplier is "
            "how many sample sd below the MEAN the floor actually lands on this "
            "family, which is the honest comparison against mean - 2s."
        ),
    }


def trimmed_mean_minus_2s_floor(values: Sequence[float]) -> dict[str, Any]:
    numbers = sorted(_checked(values))
    if len(numbers) <= 2 * TRIM_COUNT + 2:
        raise Round0231Error("R0231 trimmed floor needs more cells than it trims")
    kept = numbers[TRIM_COUNT : len(numbers) - TRIM_COUNT]
    mean = statistics.fmean(kept)
    sd = statistics.stdev(kept)
    return {
        "family": "trimmed1_mean_minus_2s",
        "scale": "trimmed_sample_sd",
        "n": len(numbers),
        "trimmed_per_tail": TRIM_COUNT,
        "kept": len(kept),
        "centre": mean,
        "scale_value": sd,
        "multiplier": TRIMMED_MULTIPLIER,
        "floor": mean - TRIMMED_MULTIPLIER * sd,
        "breakdown_point_per_tail": TRIM_COUNT / len(numbers),
    }


def median_fixed_margin_floor(values: Sequence[float]) -> dict[str, Any]:
    numbers = _checked(values)
    centre = statistics.median(numbers)
    sd = statistics.stdev(numbers)
    floor = centre * (1.0 - FIXED_MARGIN_FRACTION)
    return {
        "family": "median_fixed_margin",
        "scale": "none (fixed fraction of the median)",
        "n": len(numbers),
        "centre": centre,
        "scale_value": None,
        "multiplier": None,
        "margin_fraction": FIXED_MARGIN_FRACTION,
        "floor": floor,
        "margin_in_sample_sd": ((centre - floor) / sd) if sd else None,
        "note": (
            "outlier-invariant, but the margin is a free parameter unrelated to "
            "the family's dispersion: margin_in_sample_sd differs by metric, so "
            "the same 2% is arbitrarily strict on a tight metric and arbitrarily "
            "lax on a loose one"
        ),
    }


FLOOR_ESTIMATORS = {
    "mean_minus_2sd": mean_minus_2sd_floor,
    "one_sided_tolerance_95_95": one_sided_tolerance_floor,
    "median_minus_3_mad": median_minus_3_mad_floor,
    "trimmed1_mean_minus_2s": trimmed_mean_minus_2s_floor,
    "median_fixed_margin": median_fixed_margin_floor,
}
ROBUST_ESTIMATORS: tuple[str, ...] = (
    "median_minus_3_mad",
    "trimmed1_mean_minus_2s",
    "median_fixed_margin",
)
VARIANCE_ESTIMATORS: tuple[str, ...] = (
    "mean_minus_2sd",
    "one_sided_tolerance_95_95",
)


# --------------------------------------------------------------------------- #
# two-sided bands, on the unfolded log-ratio
# --------------------------------------------------------------------------- #


def _logs(ratios: Sequence[float]) -> list[float]:
    logs = []
    for item in ratios:
        value = float(item)
        if not math.isfinite(value) or value <= 0.0:
            raise Round0231Error("R0231 purity ratio must be finite and positive")
        logs.append(math.log(value))
    return logs


def log_ratio_band(ratios: Sequence[float], *, estimator: str) -> dict[str, Any]:
    """A two-sided band on `log r` — the scale the family actually lives on.

    Lower AND upper, always. `purity_fidelity`'s fold about `r = 1.0` cannot tell
    over-separation from under-separation, and both occur in this program:
    R0223's cuVS cells over-separate at `k256` (`r > 1`) while R0228's candidate
    arms under-separate at `k256`, and every cell under-separates at `k1024`.
    """
    logs = _logs(ratios)
    if estimator == "median_minus_3_mad":
        centre = statistics.median(logs)
        scale = mad_n(logs)
        multiplier = MAD_MULTIPLIER
        factor: dict[str, Any] | None = None
    elif estimator == "two_sided_tolerance_95_95":
        centre = statistics.fmean(logs)
        scale = statistics.stdev(logs)
        factor = two_sided_tolerance_factor(len(logs))
        multiplier = factor["k2"]
    elif estimator == "mean_minus_2sd":
        centre = statistics.fmean(logs)
        scale = statistics.stdev(logs)
        multiplier = LEGACY_MULTIPLIER
        factor = None
    else:
        raise Round0231Error(f"R0231 has no two-sided band for {estimator!r}")
    lower = centre - multiplier * scale
    upper = centre + multiplier * scale
    payload: dict[str, Any] = {
        "family": f"two_sided_log_ratio_{estimator}",
        "estimator": estimator,
        "scale": "natural log of the raw purity ratio, unfolded",
        "n": len(logs),
        "log_ratio_centre": centre,
        "log_ratio_scale": scale,
        "multiplier": multiplier,
        "ratio_geometric_centre": math.exp(centre),
        "centre_is_above_one": centre > 0.0,
        "log_lower": lower,
        "log_upper": upper,
        "ratio_lower": math.exp(lower),
        "ratio_upper": math.exp(upper),
        "quantisation_note": (
            "panel_v2 rounds each purity ratio to 4 decimals inside the scorer, "
            "so this band inherits a +/- 5e-5 quantisation in r"
        ),
    }
    if factor is not None:
        payload["factor"] = factor
    return payload


# --------------------------------------------------------------------------- #
# self-loosening
# --------------------------------------------------------------------------- #


def measure_self_loosening(
    values: Sequence[float],
    *,
    sigma_multiples: Sequence[float] = INJECTION_SIGMA_MULTIPLES,
) -> dict[str, Any]:
    """Replace the family's worst cell with a worse one and re-fit every estimator.

    A floor that DROPS when a member gets worse is self-loosening: the cell the
    gate should catch partly defends itself, monotonically more so the bigger the
    outlier. Review 0225 measured `mean - 2s` moving `-0.0079..-0.0366` and the
    95/95 floor `1.47-1.50x` further, while three robust estimators moved exactly
    `0.0`. Measured again here, on `n = 13`.
    """
    base = _checked(values)
    sd = statistics.stdev(base)
    lowest = min(range(len(base)), key=lambda index: base[index])
    baseline = {
        name: float(estimator(base)["floor"])
        for name, estimator in FLOOR_ESTIMATORS.items()
    }
    rows: list[dict[str, Any]] = []
    for multiple in sigma_multiples:
        injected = list(base)
        injected[lowest] = base[lowest] - float(multiple) * sd
        moved = {
            name: float(estimator(injected)["floor"])
            for name, estimator in FLOOR_ESTIMATORS.items()
        }
        row: dict[str, Any] = {
            "injected_sigma_multiple": float(multiple),
            "injected_value": injected[lowest],
            "replaced_value": base[lowest],
        }
        for name, before in baseline.items():
            after = moved[name]
            row[name] = {
                "floor_before": before,
                "floor_after": after,
                "shift": after - before,
                "shift_in_sd": (after - before) / sd if sd else 0.0,
                "loosened": after < before,
                "invariant": after == before,
            }
        rows.append(row)
    at_3sigma = rows[-1]
    return {
        "n": len(base),
        "sample_sd_ddof1": sd,
        "lowest_cell_index": lowest,
        "baseline_floors": baseline,
        "injections": rows,
        "outlier_invariant_estimators": sorted(
            name for name in FLOOR_ESTIMATORS if at_3sigma[name]["invariant"]
        ),
        "self_loosening_estimators": sorted(
            name for name in FLOOR_ESTIMATORS if at_3sigma[name]["loosened"]
        ),
    }


# --------------------------------------------------------------------------- #
# fitting and scoring
# --------------------------------------------------------------------------- #


def fit_floors(
    *,
    cells: Mapping[str, Mapping[str, float]],
    ratios: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Every candidate estimator, on every metric, over the thirteen-cell family."""
    seeds = sorted(cells, key=int)
    if tuple(int(seed) for seed in seeds) != EXACT_FAMILY_SEEDS:
        raise Round0231Error(
            f"R0231 exact family must be exactly seeds {list(EXACT_FAMILY_SEEDS)}"
        )
    n = len(seeds)
    gates: dict[str, Any] = {}
    for metric in METRICS:
        values = [float(cells[seed][metric]) for seed in seeds]
        entry: dict[str, Any] = {
            "seed_order": [int(seed) for seed in seeds],
            "values": values,
            "n": n,
            "role": (
                "descriptive-only" if metric in DESCRIPTIVE_METRICS else "gated"
            ),
            "self_loosening": measure_self_loosening(values),
            "families": {},
        }
        for name, estimator in FLOOR_ESTIMATORS.items():
            floor = estimator(values)
            floor["defining_cell_can_fail"] = defining_cell_can_fail(
                n=n, multiplier=floor.get("multiplier"), scale=floor["scale"]
            )
            entry["families"][name] = floor
        if metric in PURITY_METRICS:
            key = PURITY_RATIO_KEYS[metric]
            raw = [float(ratios[seed][key]) for seed in seeds]
            entry["raw_ratios"] = raw
            entry["two_sided_bands"] = {
                estimator: log_ratio_band(raw, estimator=estimator)
                for estimator in (
                    "median_minus_3_mad",
                    "two_sided_tolerance_95_95",
                    "mean_minus_2sd",
                )
            }
        gates[metric] = entry
    return {
        "n": n,
        "seed_order": [int(seed) for seed in seeds],
        "metrics": list(METRICS),
        "gated_metrics": list(GATED_METRICS),
        "descriptive_metrics": list(DESCRIPTIVE_METRICS),
        "purity_metrics": list(PURITY_METRICS),
        "identity_bound_at_n": identity_bound(n),
        "family_definitions": FAMILY_DEFINITIONS,
        "sample_standard_deviation_ddof": SD_DDOF,
        "chosen_estimator": CHOSEN_ESTIMATOR,
        "chosen_estimator_justification": CHOSEN_ESTIMATOR_JUSTIFICATION,
        "density_v2_defect": DENSITY_V2_DEFECT,
        "gates": gates,
    }


def score_cells_against(
    *,
    floors: Mapping[str, float],
    bands: Mapping[str, Mapping[str, float]],
    cells: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    defining_seed_ids: Sequence[str],
    family_can_fail_a_defining_cell: bool,
) -> dict[str, Any]:
    """Score every cell against one floor family, and say whether the count meant anything.

    `attainable` is the honest half of a pass count: a count of `m/m` is evidence
    only if some cell in the set was *able* to fail. That is true when the set
    contains a cell that did not define the floors, or when the family's own
    defining cells can fail it.
    """
    defining = {str(item) for item in defining_seed_ids}
    rows: list[dict[str, Any]] = []
    for cell in cells:
        cell_id = str(cell["cell_id"])
        is_defining = cell_id in defining
        row: dict[str, Any] = {
            "cell_id": cell_id,
            "family": str(cell["family"]),
            "defines_the_floors": is_defining,
            "metrics": {},
        }
        for metric in metrics:
            if metric not in floors:
                continue
            value = float(cell["values"][metric])
            floor = float(floors[metric])
            entry: dict[str, Any] = {
                "value": value,
                "floor": floor,
                "margin": value - floor,
                "clears_floor": value >= floor,
                "passes": value >= floor,
            }
            band = bands.get(metric)
            if band is not None and cell.get("ratios"):
                ratio = float(cell["ratios"][PURITY_RATIO_KEYS[metric]])
                lower = float(band["ratio_lower"])
                upper = float(band["ratio_upper"])
                inside = lower <= ratio <= upper
                entry.update({
                    "raw_ratio": ratio,
                    "log_ratio": math.log(ratio),
                    "ratio_lower": lower,
                    "ratio_upper": upper,
                    "within_band": inside,
                    "passes": inside,
                    "band_direction": (
                        "below_band" if ratio < lower
                        else "above_band" if ratio > upper
                        else "inside_band"
                    ),
                    "separation": (
                        "over_separating" if ratio > 1.0
                        else "under_separating" if ratio < 1.0
                        else "exact"
                    ),
                })
            row["metrics"][metric] = entry
        rows.append(row)
    failures = []
    for row in rows:
        for metric, entry in row["metrics"].items():
            if entry["passes"]:
                continue
            record = {
                "cell_id": row["cell_id"],
                "metric": metric,
                "defines_the_floors": row["defines_the_floors"],
                "value": entry["value"],
                "floor": entry["floor"],
            }
            if "raw_ratio" in entry:
                # For a banded metric the decision is made on the raw unfolded
                # ratio, so the failure record must carry that rather than the
                # folded fidelity alone.
                record.update({
                    "decided_on": "raw unfolded purity ratio, two-sided band",
                    "raw_ratio": entry["raw_ratio"],
                    "ratio_lower": entry["ratio_lower"],
                    "ratio_upper": entry["ratio_upper"],
                    "direction": entry["band_direction"],
                    "separation": entry["separation"],
                })
            else:
                record["decided_on"] = "one-sided lower floor"
                record["direction"] = "below_floor"
            failures.append(record)
    non_defining = [row for row in rows if not row["defines_the_floors"]]
    clearing = sum(
        1
        for row in rows
        if all(row["metrics"][metric]["passes"] for metric in row["metrics"])
    )
    return {
        "cells_scored": len(rows),
        "cells_clearing_every_gated_metric": clearing,
        "failing_cell_metrics": failures,
        "failures": len(failures),
        "defining_cells_scored": len(rows) - len(non_defining),
        "non_defining_cells_scored": len(non_defining),
        "family_can_fail_a_defining_cell": bool(family_can_fail_a_defining_cell),
        "count_is_attainable": bool(non_defining) or bool(
            family_can_fail_a_defining_cell
        ),
        "attainability_note": (
            "a pass count is evidence only if some scored cell was able to fail. "
            "That holds when the set contains a non-defining cell, or when the "
            "family's multiplier is below (n-1)/sqrt(n) at its own n, or when the "
            "family uses a robust scale to which that identity does not apply."
        ),
        "cells": rows,
    }


__all__ = [
    "CANDIDATE_CLUSTER_COUNTS",
    "CANDIDATE_SEEDS",
    "CHOSEN_ESTIMATOR",
    "CHOSEN_ESTIMATOR_JUSTIFICATION",
    "CUVS_FAMILY_SEEDS",
    "DENSITY_V2_DEFECT",
    "DESCRIPTIVE_METRICS",
    "EXACT_FAMILY_SEEDS",
    "FAMILY_DEFINITIONS",
    "FIXED_MARGIN_FRACTION",
    "FLOOR_ESTIMATORS",
    "GATED_METRICS",
    "GATE_CAPABILITY",
    "GATE_SCHEMA",
    "INJECTION_SIGMA_MULTIPLES",
    "LEGACY_MULTIPLIER",
    "MAD_CONSISTENCY",
    "MAD_MULTIPLIER",
    "METRICS",
    "N_EXACT",
    "ONE_SIDED_DERIVATION",
    "PURITY_METRICS",
    "PURITY_RATIO_KEYS",
    "ROBUST_ESTIMATORS",
    "ROUND_ID",
    "Round0231Error",
    "SD_DDOF",
    "TOLERANCE_CONFIDENCE",
    "TOLERANCE_CONTENT",
    "TRIMMED_MULTIPLIER",
    "TRIM_COUNT",
    "TWO_SIDED_DERIVATION",
    "VARIANCE_ESTIMATORS",
    "defining_cell_can_fail",
    "fit_floors",
    "identity_bound",
    "log_ratio_band",
    "mad_n",
    "mean_minus_2sd_floor",
    "measure_self_loosening",
    "median_fixed_margin_floor",
    "median_minus_3_mad_floor",
    "one_sided_tolerance_factor",
    "one_sided_tolerance_floor",
    "score_cells_against",
    "trimmed_mean_minus_2s_floor",
    "two_sided_tolerance_factor",
    "witness_defining_cell_failure",
]
