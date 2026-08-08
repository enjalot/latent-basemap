#!/usr/bin/env python3
"""R0225 — re-register the 2M gate as a tolerance interval, on the right scale.

Review 0222-01 found the gate method structurally defective in three ways, and
this module is the arithmetic that answers each.

**1. `mean - 2*sigma` is self-loosening.** A low cell drags the mean down *and*
inflates `s`, and because the floor subtracts a multiple of `s`, the cell partly
defends itself. Seed 48 widened R0222's admissible `k256` band by 73% by failing
it. `measure_self_loosening()` turns that argument into a number for every
metric and every floor family.

**2. `purity_fidelity` is folded about the wrong centre.** The panel reports
`exp(-|log r|)`, which reflects the ratio about `r = 1.0` while the family
actually centres at `r-bar = 1.0086`. Folding also destroys the sign, so
over-separation and under-separation become the same failure — and R0223 showed
both occur (over at `k256`, under at `k1024`). `log_ratio_band()` gates the
purity metrics **two-sidedly on the unfolded log-ratio scale**, where the two
directions are distinguishable.

**3. `n = 8` is thin, and `k = 2.0` is not the right multiplier for it.** The
one-sided 95/95 normal tolerance factor at `n = 8` is `k = 3.187...`, not `2.0`.
`one_sided_tolerance_factor()` derives it rather than quoting it, and
`ONE_SIDED_DERIVATION` states the derivation in full.

Nothing here trains, scores a map, or reads a GPU. Every input is a sealed
artifact from R0218/R0222/R0223.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

ROUND_ID = "0225"

GATE_CAPABILITY = "minilm-mixed-2m-tolerance-gates-n8-v1"
GATE_SCHEMA = "round0225-minilm-mixed-2m-tolerance-gates-v1"

#: The four metrics R0222 registered. Two of the accepted six
#: (`heldout_recall_at_10`, `projection_ffr`) are *unavailable* on this universe
#: — R0216 sampled 2,000,000 rows and used all of them, so no held-out reserve
#: exists — not excluded by judgement. Review 0222-01 verified that distinction
#: and it is carried forward unchanged.
GATE_METRICS: tuple[str, ...] = (
    "density_v2",
    "ffr",
    "purity_fidelity_k256",
    "purity_fidelity_k1024",
)
#: The two purity metrics are ratios and get the two-sided log-ratio treatment.
PURITY_METRICS: tuple[str, ...] = (
    "purity_fidelity_k256",
    "purity_fidelity_k1024",
)
PURITY_RATIO_KEYS: dict[str, str] = {
    "purity_fidelity_k256": "k256",
    "purity_fidelity_k1024": "k1024",
}

EXACT_FAMILY_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46, 47, 48, 49)
CUVS_FAMILY_SEEDS: tuple[int, ...] = (42, 43, 44)

TOLERANCE_CONTENT = 0.95
TOLERANCE_CONFIDENCE = 0.95
SD_DDOF = 1
LEGACY_MULTIPLIER = 2.0

#: Published by review-0222-01 as `k = 3.187` at `n = 8`. Registered here only
#: as a CROSS-CHECK: `one_sided_tolerance_factor(8)` derives the value from the
#: noncentral t distribution and the two are compared. The round reproduces the
#: factor rather than copying it.
REVIEW_0222_ONE_SIDED_FACTOR_N8 = 3.187
ONE_SIDED_CROSS_CHECK_TOLERANCE = 1.0e-3

ONE_SIDED_DERIVATION = (
    "A one-sided lower tolerance bound L = xbar - k*s must satisfy "
    "P(L <= mu - z_P*sigma) >= gamma, i.e. with confidence gamma it sits below "
    "the P-content quantile of the population. Rearranging: "
    "xbar - k*s <= mu - z_P*sigma  <=>  (xbar-mu)/(sigma/sqrt(n)) + z_P*sqrt(n) "
    "<= k*sqrt(n)*(s/sigma). The left numerator is N(delta, 1) with "
    "delta = z_P*sqrt(n); s/sigma = sqrt(chi2_{n-1}/(n-1)) independently. Their "
    "ratio is therefore a NONCENTRAL t with df = n-1 and noncentrality delta, so "
    "k*sqrt(n) = t'_{n-1,delta}(gamma) and k = t'_{n-1,delta}(gamma)/sqrt(n)."
)
TWO_SIDED_DERIVATION = (
    "The two-sided factor has no closed form. Howe (1969) gives "
    "k2 = z_{(1+P)/2} * sqrt(1+1/n) * sqrt((n-1)/chi2_{1-gamma,n-1}) * "
    "sqrt(1 + (n-3-chi2_{1-gamma,n-1})/(2*(n+1)^2)), which is registered here "
    "and independently confirmed by simulating the coverage it delivers."
)

FAMILY_DEFINITIONS = (
    "mean_minus_2sd: the R0222 method, floor = mean - 2*s (ddof=1). "
    "one_sided_tolerance_95_95: floor = mean - k*s with k the derived 95/95 "
    "one-sided normal tolerance factor at this n. "
    "two_sided_log_ratio_95_95: for the purity metrics only, a two-sided band "
    "on log r, [mu - k2*s, mu + k2*s], computed on the UNFOLDED ratio so that "
    "over-separation and under-separation are distinguishable failures."
)


class Round0225Error(RuntimeError):
    """The registered R0225 tolerance-gate contract changed."""


# --------------------------------------------------------------------------- #
# the tolerance factors, derived
# --------------------------------------------------------------------------- #


def one_sided_tolerance_factor(
    n: int,
    *,
    content: float = TOLERANCE_CONTENT,
    confidence: float = TOLERANCE_CONFIDENCE,
) -> dict[str, Any]:
    """`k` such that `mean - k*s` is a `content`/`confidence` lower bound.

    Derived, not quoted: see `ONE_SIDED_DERIVATION`. The value is compared to
    review-0222-01's published `3.187` at `n = 8` as a cross-check, and a
    disagreement beyond `ONE_SIDED_CROSS_CHECK_TOLERANCE` is an error, not a
    note — the whole point of this round is that the factor is reproducible.
    """
    from scipy import stats

    n = int(n)
    if n < 3:
        raise Round0225Error("R0225 tolerance factor needs n >= 3")
    z_content = float(stats.norm.ppf(content))
    noncentrality = z_content * math.sqrt(n)
    k = float(stats.nct.ppf(confidence, n - 1, noncentrality)) / math.sqrt(n)
    payload: dict[str, Any] = {
        "n": n,
        "content": float(content),
        "confidence": float(confidence),
        "z_content": z_content,
        "noncentrality": noncentrality,
        "degrees_of_freedom": n - 1,
        "k": k,
        "derivation": ONE_SIDED_DERIVATION,
    }
    if n == 8 and content == TOLERANCE_CONTENT and confidence == TOLERANCE_CONFIDENCE:
        delta = abs(k - REVIEW_0222_ONE_SIDED_FACTOR_N8)
        payload["review_0222_published_factor"] = REVIEW_0222_ONE_SIDED_FACTOR_N8
        payload["cross_check_delta"] = delta
        payload["cross_check_passes"] = delta <= ONE_SIDED_CROSS_CHECK_TOLERANCE
        if not payload["cross_check_passes"]:
            raise Round0225Error(
                f"R0225 derived one-sided factor {k} disagrees with "
                f"review-0222's {REVIEW_0222_ONE_SIDED_FACTOR_N8} by {delta}"
            )
    return payload


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
        raise Round0225Error("R0225 tolerance factor needs n >= 3")
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


# --------------------------------------------------------------------------- #
# floors
# --------------------------------------------------------------------------- #


def _stats(values: Sequence[float]) -> tuple[int, float, float]:
    values = [float(item) for item in values]
    n = len(values)
    if n < 3:
        raise Round0225Error("R0225 needs at least three cells to fit a floor")
    return n, statistics.fmean(values), statistics.stdev(values)


def mean_minus_2sd_floor(values: Sequence[float]) -> dict[str, Any]:
    """R0222's method, reproduced so the two families are computed alike."""
    n, mean, sd = _stats(values)
    return {
        "family": "mean_minus_2sd",
        "n": n,
        "mean": mean,
        "sample_sd_ddof1": sd,
        "multiplier": LEGACY_MULTIPLIER,
        "floor": mean - LEGACY_MULTIPLIER * sd,
    }


def one_sided_tolerance_floor(
    values: Sequence[float],
    *,
    content: float = TOLERANCE_CONTENT,
    confidence: float = TOLERANCE_CONFIDENCE,
) -> dict[str, Any]:
    n, mean, sd = _stats(values)
    factor = one_sided_tolerance_factor(n, content=content, confidence=confidence)
    return {
        "family": "one_sided_tolerance_95_95",
        "n": n,
        "mean": mean,
        "sample_sd_ddof1": sd,
        "k": factor["k"],
        "factor": factor,
        "floor": mean - factor["k"] * sd,
    }


def log_ratio_band(
    ratios: Sequence[float],
    *,
    content: float = TOLERANCE_CONTENT,
    confidence: float = TOLERANCE_CONFIDENCE,
) -> dict[str, Any]:
    """A two-sided 95/95 band on `log r`, the scale the family actually lives on.

    The folded `purity_fidelity` cannot tell over-separation from
    under-separation, and both occur in this program: R0223's three cuVS cells
    all over-separate at `k256` (`r > 1`) while every cell under-separates at
    `k1024` (`r < 1`). On the log scale the two are opposite signs and the band
    has a lower AND an upper bound, so they are distinguishable failures.
    """
    logs = []
    for item in ratios:
        value = float(item)
        if not math.isfinite(value) or value <= 0:
            raise Round0225Error("R0225 purity ratio must be finite and positive")
        logs.append(math.log(value))
    n, mean, sd = _stats(logs)
    factor = two_sided_tolerance_factor(n, content=content, confidence=confidence)
    k2 = factor["k2"]
    lower, upper = mean - k2 * sd, mean + k2 * sd
    return {
        "family": "two_sided_log_ratio_95_95",
        "scale": "natural log of the raw purity ratio, unfolded",
        "n": n,
        "log_ratio_mean": mean,
        "log_ratio_sample_sd_ddof1": sd,
        "ratio_geometric_mean": math.exp(mean),
        "centre_is_above_one": mean > 0.0,
        "k2": k2,
        "factor": factor,
        "log_lower": lower,
        "log_upper": upper,
        "ratio_lower": math.exp(lower),
        "ratio_upper": math.exp(upper),
        "note": (
            "the fold in purity_fidelity reflects about r = 1.0, but this family "
            "centres at exp(log_ratio_mean); folding about a point the family is "
            "not centred on manufactures |z| and destroys the direction of the "
            "deviation"
        ),
    }


# --------------------------------------------------------------------------- #
# the self-loosening measurement
# --------------------------------------------------------------------------- #


def measure_self_loosening(
    values: Sequence[float], *, sigma_multiples: Sequence[float] = (1.0, 2.0, 3.0)
) -> dict[str, Any]:
    """How far does a floor move when the family's worst cell gets worse?

    Review 0222-01 made the argument for `k256` by comparing `n = 7` to
    `n = 8`. This measures it directly and for every metric: replace the lowest
    cell with one `m` sample-sd lower and re-fit. A floor that *drops* when a
    member fails it is self-loosening — the cell partly defends itself.

    Both families are measured, and the result is not the flattering one. The
    tolerance factor fixes the floor's CALIBRATION; it does not fix
    self-loosening, and because `k` is larger the tolerance floor is moved MORE
    by the same outlier. The structural fix is calibrating on held-out cells,
    which is review 0222-01's required correction 3 and is NOT claimed here.
    """
    base = [float(item) for item in values]
    n, _mean, sd = _stats(base)
    lowest = min(range(len(base)), key=lambda index: base[index])
    baseline = {
        "mean_minus_2sd": mean_minus_2sd_floor(base)["floor"],
        "one_sided_tolerance_95_95": one_sided_tolerance_floor(base)["floor"],
    }
    rows: list[dict[str, Any]] = []
    for multiple in sigma_multiples:
        injected = list(base)
        injected[lowest] = base[lowest] - float(multiple) * sd
        moved = {
            "mean_minus_2sd": mean_minus_2sd_floor(injected)["floor"],
            "one_sided_tolerance_95_95": one_sided_tolerance_floor(injected)["floor"],
        }
        row: dict[str, Any] = {
            "injected_sigma_multiple": float(multiple),
            "injected_value": injected[lowest],
            "replaced_value": base[lowest],
        }
        for family, before in baseline.items():
            after = moved[family]
            row[family] = {
                "floor_before": before,
                "floor_after": after,
                "shift": after - before,
                "shift_in_sd": (after - before) / sd if sd else 0.0,
                "loosened": after < before,
            }
        row["tolerance_moves_more_than_legacy"] = abs(
            row["one_sided_tolerance_95_95"]["shift"]
        ) > abs(row["mean_minus_2sd"]["shift"])
        rows.append(row)
    return {
        "n": n,
        "sample_sd_ddof1": sd,
        "lowest_cell_index": lowest,
        "baseline_floors": baseline,
        "injections": rows,
        "finding": (
            "both families are self-loosening: injecting a worse worst-cell "
            "lowers the floor that would judge it. The 95/95 factor corrects "
            "the floor's CALIBRATION (its false-fail rate), not this property, "
            "and moves further per unit of injected outlier because k > 2. "
            "Fixing self-loosening structurally requires calibrating on cells "
            "that are held out of the fit; this round does not do that and does "
            "not claim to."
        ),
    }


# --------------------------------------------------------------------------- #
# evaluating cells against floors
# --------------------------------------------------------------------------- #


def evaluate_cell(
    *,
    value: float,
    floor: float,
    upper: float | None = None,
) -> dict[str, Any]:
    passes = float(value) >= float(floor)
    payload: dict[str, Any] = {
        "value": float(value),
        "floor": float(floor),
        "clears_floor": passes,
        "margin": float(value) - float(floor),
    }
    if upper is not None:
        under = float(value) <= float(upper)
        payload.update({
            "upper": float(upper),
            "within_upper": under,
            "passes": passes and under,
            "direction": (
                "below_band" if not passes
                else "above_band" if not under
                else "inside_band"
            ),
        })
    else:
        payload["passes"] = passes
    return payload


def registered_gate(
    *,
    exact_cells: Mapping[str, Mapping[str, float]],
    exact_ratios: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Fit all three floor families on the 8-cell exact-graph family."""
    seeds = sorted(exact_cells, key=int)
    if len(seeds) != len(EXACT_FAMILY_SEEDS):
        raise Round0225Error(
            f"R0225 expects {len(EXACT_FAMILY_SEEDS)} exact-graph cells, "
            f"got {len(seeds)}"
        )
    if tuple(int(seed) for seed in seeds) != EXACT_FAMILY_SEEDS:
        raise Round0225Error("R0225 exact-graph family seeds changed")

    gates: dict[str, Any] = {}
    for metric in GATE_METRICS:
        values = [float(exact_cells[seed][metric]) for seed in seeds]
        gates[metric] = {
            "seed_order": [int(seed) for seed in seeds],
            "values": values,
            "n": len(values),
            "mean_minus_2sd": mean_minus_2sd_floor(values),
            "one_sided_tolerance_95_95": one_sided_tolerance_floor(values),
            "self_loosening": measure_self_loosening(values),
        }
        if metric in PURITY_METRICS:
            key = PURITY_RATIO_KEYS[metric]
            ratios = [float(exact_ratios[seed][key]) for seed in seeds]
            gates[metric]["raw_ratios"] = ratios
            gates[metric]["two_sided_log_ratio_95_95"] = log_ratio_band(ratios)
    return {
        "n": len(seeds),
        "seed_order": [int(seed) for seed in seeds],
        "metrics": list(GATE_METRICS),
        "purity_metrics": list(PURITY_METRICS),
        "family_definitions": FAMILY_DEFINITIONS,
        "sample_standard_deviation_ddof": SD_DDOF,
        "gates": gates,
    }


def score_all_cells(
    *,
    gate: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Every cell against every floor family, pass/fail per cell per metric.

    `cells` carries both populations: the 8 exact-graph cells that DEFINE the
    floors and R0223's 3 cuVS cells that do not. Defining cells are marked as
    such, because a defining cell clearing its own floor is a much weaker
    statement than an independent cell clearing it.
    """
    rows: list[dict[str, Any]] = []
    for cell in cells:
        row: dict[str, Any] = {
            "cell_id": str(cell["cell_id"]),
            "family": str(cell["family"]),
            "seed": int(cell["seed"]),
            "defines_the_floors": bool(cell.get("defines_the_floors")),
            "metrics": {},
        }
        for metric in GATE_METRICS:
            value = float(cell["values"][metric])
            entry = {
                "mean_minus_2sd": evaluate_cell(
                    value=value,
                    floor=gate["gates"][metric]["mean_minus_2sd"]["floor"],
                ),
                "one_sided_tolerance_95_95": evaluate_cell(
                    value=value,
                    floor=gate["gates"][metric]["one_sided_tolerance_95_95"]["floor"],
                ),
            }
            if metric in PURITY_METRICS and cell.get("ratios"):
                band = gate["gates"][metric]["two_sided_log_ratio_95_95"]
                ratio = float(cell["ratios"][PURITY_RATIO_KEYS[metric]])
                entry["two_sided_log_ratio_95_95"] = {
                    **evaluate_cell(
                        value=ratio,
                        floor=band["ratio_lower"],
                        upper=band["ratio_upper"],
                    ),
                    "raw_ratio": ratio,
                    "log_ratio": math.log(ratio),
                }
            row["metrics"][metric] = entry
        rows.append(row)

    summary: dict[str, Any] = {}
    for family in (
        "mean_minus_2sd",
        "one_sided_tolerance_95_95",
        "two_sided_log_ratio_95_95",
    ):
        failures = [
            {
                "cell_id": row["cell_id"],
                "metric": metric,
                "direction": row["metrics"][metric][family].get("direction"),
            }
            for row in rows
            for metric in GATE_METRICS
            if family in row["metrics"][metric]
            and not row["metrics"][metric][family]["passes"]
        ]
        summary[family] = {
            "failing_cell_metrics": failures,
            "failures": len(failures),
        }
    return {"cells": rows, "per_family": summary, "cell_count": len(rows)}
