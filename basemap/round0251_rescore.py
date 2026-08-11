"""R0251 — settle poolability on the MAP side by rescoring one archived cell.

review-0250-01 §B.6 blocked `claim:r0250-the-sixteen-cells-are-poolable-on-the-map-side`
and named the settling experiment in one line: *"re-score one archived checkpoint
— seed 42 — with this release and check it returns `1.0216`."*

The block is precise about what R0250 did and did not prove. R0250's poolability
evidence is byte-identity of R0218's high-D reference on five components plus the
`hi_D_agreement` numerators `0.3828` / `0.2385` — **all of which are
map-independent**. They constrain the reference side of every purity ratio and
say nothing about the scorer that turns a checkpoint into coordinates and
coordinates into metrics. R0230 and R0250 both declined to rescore the prior
cells, so between R0218 (seeds 42-45), R0222 (46-49), R0230 (50-54) and R0250
(55-57) the map-side scorer has changed release four times with no cell scored
twice. Meanwhile all three of R0250's new cells rank 13/14/15 of 16 on the
`k256` ratio, and **two of the three defining-cell failures ride entirely on
that shift**.

This module is the control. It rescores **seed 42** — R0218's first cell, the
one whose `k256` ratio `1.0216` the review names — from its archived `model.pt`
on this release, against the same frozen panel, the same reference bytes and the
same anchors, and compares every panel value to R0218's sealed cell.

**What a pass and a fail each mean, stated before the run.**

* Every value reproduces → the map-side scorer has not drifted across four
  releases, the sixteen cells are commensurate on the map side as well as the
  reference side, and the `k256` shift in seeds 55-57 is a property of those
  maps rather than of the instrument. The shift then stands as a real, unexplained
  systematic and the two defining-cell failures that ride on it stand with it.
* Any value moves → the pooled family is a mixed population, the `n = 16` gate
  is fitted to cells scored by different instruments, and everything downstream
  inherits that.

**What this control does NOT settle**, and the round says so rather than letting
a reviewer find it: reproducing seed 42 proves the scorer is stable on ONE map.
It cannot prove the scorer is stable in the direction the new cells moved, and it
is not a randomised test of the shift. It removes one explanation, the cheapest
and most likely one, and leaves the rest.

The shift test itself is arithmetic on the sealed sixteen-cell panel and is
computed here rather than quoted, including the exact Mann-Whitney null (the
review reports `p = 0.025` for `k256` and `p = 0.031` for `density_v2`; those are
the same statistic under the exact and the asymptotic null, and both are reported
for both metrics).
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .round0250_panel_n16 import (
    HI_D_AGREEMENT,
    PANEL_METRICS,
    POOLED_SEEDS,
    PURITY_RATIO_KEYS,
    REFERENCE_KEY,
    R0218_SEEDS,
)
from .round0250_seed_extension_n16 import SEEDS as R0250_SEEDS


ROUND_ID = "0251"

RESCORE_CAPABILITY = "round0251-seed42-map-side-rescore-v1"
RESCORE_SCHEMA = "round0251-seed42-map-side-rescore-v1"

#: The cell the review names. It is R0218's first cell, so it is the longest
#: lever available: four release generations separate its original scoring from
#: this one.
RESCORED_SEED = 42
RESCORED_SOURCE_ROUND = "0218"

#: The cells whose scoring release is being tested against, in order.
SCORING_RELEASE_GENERATIONS: tuple[str, ...] = ("0218", "0222", "0230", "0250")

#: `panel_v2` rounds every purity ratio and every panel metric to four decimals
#: inside the scorer, so one quantum is `1e-4`. Drift is declared at anything
#: strictly larger than one quantum; exact equality is reported separately and is
#: the outcome a stable scorer should produce.
SCORER_QUANTUM = 1e-4
SCORER_DRIFT_TOLERANCE = SCORER_QUANTUM

#: The prior/new split the review tested. Fixed here so the test cannot be
#: re-cut after seeing the answer.
PRIOR_SEEDS: tuple[int, ...] = tuple(
    seed for seed in POOLED_SEEDS if seed not in set(R0250_SEEDS)
)
NEW_SEEDS: tuple[int, ...] = tuple(R0250_SEEDS)

SHIFT_METRICS: tuple[str, ...] = ("density_v2", "ffr")
SHIFT_RATIO_KEYS: tuple[str, ...] = ("k256", "k1024")

POOLABILITY_NOTE = (
    "R0250 proved the MAP-INDEPENDENT half of poolability (R0218's reference "
    "bytes, its content key, its anchors, and the hi_D_agreement numerators "
    f"{dict(HI_D_AGREEMENT)}). This module tests the MAP-DEPENDENT half: the "
    "checkpoint -> coordinates -> panel path, on one archived cell, under this "
    "release. Neither half implies the other."
)


class Round0251RescoreError(RuntimeError):
    """The registered R0251 map-side rescore contract changed."""


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise Round0251RescoreError(f"R0251 {label} is not finite: {value!r}")
    return number


def sealed_seed42_cell(
    panel: Mapping[str, Any], *, pooled_panel: Mapping[str, Any]
) -> dict[str, Any]:
    """R0218's published seed-42 values, read from its sealed panel bytes.

    Nothing here is typed: the comparison targets come from the artifact the
    review points at, and the function fails closed if that artifact is not the
    one R0218 published for this reference.

    R0218's panel does not carry the UNFOLDED purity ratios — R0230 sourced all
    eight of those from R0223 and R0250 pooled them — so the `1.0216` the review
    names is read from the sealed sixteen-cell panel's `raw_purity_ratios`,
    which is the artifact the `n = 16` gate was actually fitted to. Both sources
    must agree on this cell's identity before either is used.
    """
    if (
        str(panel.get("round_id") or "") != RESCORED_SOURCE_ROUND
        or str(panel.get("high_d_reference_key") or "") != REFERENCE_KEY
        or tuple(int(seed) for seed in panel.get("seeds") or ()) != tuple(R0218_SEEDS)
    ):
        raise Round0251RescoreError(
            "R0251 rescore target is not R0218's published panel for this reference"
        )
    if (
        str(pooled_panel.get("round_id") or "") != "0250"
        or int(pooled_panel.get("n", -1)) != len(POOLED_SEEDS)
        or str(pooled_panel.get("high_d_reference_key") or "") != REFERENCE_KEY
    ):
        raise Round0251RescoreError(
            "R0251 needs R0250's sealed sixteen-cell panel for the unfolded ratios"
        )
    key = str(RESCORED_SEED)
    cell = dict(panel["cells"][key])
    if str(dict(pooled_panel["scored_in_round_by_seed"])[key]) != RESCORED_SOURCE_ROUND:
        raise Round0251RescoreError(
            "R0251 expects the pooled family to attribute seed 42 to R0218"
        )
    pooled_metrics = dict(pooled_panel["panel_metric_cells"])[key]
    for metric in PANEL_METRICS:
        if float(pooled_metrics[metric]) != float(cell["panel_metrics"][metric]):
            raise Round0251RescoreError(
                f"R0251 seed-42 {metric} differs between R0218's panel and the "
                "pooled sixteen-cell table before anything is rescored"
            )
    ratios = {
        ratio_key: _finite(
            dict(pooled_panel["raw_purity_ratios"])[key][ratio_key],
            f"sealed seed-42 ratio {ratio_key}",
        )
        for ratio_key in SHIFT_RATIO_KEYS
    }
    scored = dict(cell["panel"])
    numerators = {
        ratio_key: _finite(
            scored["purity_numerators"][ratio_key]["hi_D_agreement"],
            f"R0218 seed-42 hi_D_agreement {ratio_key}",
        )
        for ratio_key in SHIFT_RATIO_KEYS
    }
    return {
        "seed": RESCORED_SEED,
        "source_round": RESCORED_SOURCE_ROUND,
        "capability": str(cell["capability"]),
        "model": dict(cell["model"]),
        "train_receipt": dict(cell["train_receipt"]),
        "coordinates_ordered_sha256": str(cell["coordinates_ordered_sha256"]),
        "panel_metrics": {
            metric: _finite(cell["panel_metrics"][metric], f"R0218 seed-42 {metric}")
            for metric in PANEL_METRICS
        },
        "purity_ratios": ratios,
        "hi_d_agreement": numerators,
        "corpus_ffr": {
            slug: {
                "anchors": int(entry["anchors"]),
                "ffr": _finite(entry["ffr"], f"R0218 seed-42 {slug} ffr"),
            }
            for slug, entry in dict(cell["corpus_ffr"]).items()
        },
    }


def compare_rescore(
    *,
    sealed: Mapping[str, Any],
    observed_panel_metrics: Mapping[str, float],
    observed_ratios: Mapping[str, float],
    observed_hi_d_agreement: Mapping[str, float],
    observed_corpus_ffr: Mapping[str, Mapping[str, Any]],
    observed_coordinates_sha256: str,
) -> dict[str, Any]:
    """Every comparable value, with its delta. Never raises on a difference.

    A drift is this round's finding, not its error: the node publishes the
    comparison either way and the verdict field says which happened.
    """
    rows: list[dict[str, Any]] = []

    def _row(kind: str, name: str, target: float, observed: float) -> None:
        delta = float(observed) - float(target)
        rows.append({
            "kind": kind,
            "name": name,
            "r0218_sealed": float(target),
            "r0251_observed": float(observed),
            "delta": delta,
            "abs_delta": abs(delta),
            "exactly_equal": float(observed) == float(target),
            "within_one_panel_quantum": abs(delta) <= SCORER_DRIFT_TOLERANCE,
        })

    for metric in PANEL_METRICS:
        _row(
            "panel_metric",
            metric,
            sealed["panel_metrics"][metric],
            _finite(observed_panel_metrics[metric], f"observed {metric}"),
        )
    for ratio_key in SHIFT_RATIO_KEYS:
        _row(
            "raw_purity_ratio",
            ratio_key,
            sealed["purity_ratios"][ratio_key],
            _finite(observed_ratios[ratio_key], f"observed ratio {ratio_key}"),
        )
        _row(
            "hi_d_agreement",
            ratio_key,
            sealed["hi_d_agreement"][ratio_key],
            _finite(
                observed_hi_d_agreement[ratio_key],
                f"observed hi_D_agreement {ratio_key}",
            ),
        )
    for slug, entry in dict(sealed["corpus_ffr"]).items():
        _row(
            "corpus_ffr",
            slug,
            entry["ffr"],
            _finite(
                dict(observed_corpus_ffr[slug])["ffr"], f"observed {slug} ffr"
            ),
        )
    exact = [row for row in rows if row["exactly_equal"]]
    drifted = [row for row in rows if not row["within_one_panel_quantum"]]
    coordinates_identical = str(observed_coordinates_sha256) == str(
        sealed["coordinates_ordered_sha256"]
    )
    return {
        "seed": RESCORED_SEED,
        "comparisons": rows,
        "values_compared": len(rows),
        "values_exactly_equal": len(exact),
        "values_within_one_panel_quantum": len(rows) - len(drifted),
        "values_drifted": len(drifted),
        "drifted": [row["name"] for row in drifted],
        "coordinates_ordered_sha256_identical": coordinates_identical,
        "coordinates_note": (
            "the coordinate digest is the strictest possible comparison and is "
            "reported as evidence, never as a requirement: the transform runs "
            "cuBLAS kernels whose reduction order is not guaranteed across "
            "driver or library versions, so a digest mismatch with identical "
            "four-decimal panel values is a numerical detail, while a PANEL "
            "value moving is scorer drift."
        ),
        "the_map_side_scorer_reproduces": len(drifted) == 0,
        "poolability_note": POOLABILITY_NOTE,
        "what_this_does_not_settle": (
            "one cell. It removes the instrument as an explanation for the k256 "
            "shift in seeds 55-57; it does not test the scorer in the direction "
            "those cells moved, and it is not a randomised control on the shift."
        ),
    }


def shift_test(
    *,
    panel_metric_cells: Mapping[str, Mapping[str, float]],
    raw_purity_ratios: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """The prior-13 vs new-3 comparison the review ran, recomputed here.

    Both Mann-Whitney nulls are reported. The review quotes `p = 0.025` for
    `k256` and `p = 0.031` for `density_v2`; with `n = 13` against `n = 3` and no
    ties those are the SAME statistic under the exact and the normal-approximation
    nulls respectively, so reporting one of each understates how alike the two
    shifts are.
    """
    from scipy import stats

    prior = [str(seed) for seed in PRIOR_SEEDS]
    new = [str(seed) for seed in NEW_SEEDS]
    if len(prior) + len(new) != len(POOLED_SEEDS):
        raise Round0251RescoreError("R0251 shift split is not the sixteen cells")

    series: dict[str, tuple[list[float], list[float]]] = {}
    for metric in SHIFT_METRICS:
        series[metric] = (
            [_finite(panel_metric_cells[seed][metric], metric) for seed in prior],
            [_finite(panel_metric_cells[seed][metric], metric) for seed in new],
        )
    for ratio_key in SHIFT_RATIO_KEYS:
        series[f"ratio::{ratio_key}"] = (
            [_finite(raw_purity_ratios[seed][ratio_key], ratio_key) for seed in prior],
            [_finite(raw_purity_ratios[seed][ratio_key], ratio_key) for seed in new],
        )

    rows: dict[str, Any] = {}
    for name, (left, right) in series.items():
        pooled = sorted(left + right)
        ranks = [pooled.index(value) + 1 for value in right]
        exact = stats.mannwhitneyu(right, left, alternative="two-sided", method="exact")
        approx = stats.mannwhitneyu(
            right, left, alternative="two-sided", method="asymptotic"
        )
        welch = stats.ttest_ind(right, left, equal_var=False)
        mean_left = sum(left) / len(left)
        mean_right = sum(right) / len(right)
        var_left = sum((value - mean_left) ** 2 for value in left) / (len(left) - 1)
        rows[name] = {
            "prior_n": len(left),
            "new_n": len(right),
            "prior_mean": mean_left,
            "prior_sample_sd": math.sqrt(var_left),
            "new_mean": mean_right,
            "new_cell_ranks_in_the_pooled_sixteen": ranks,
            "mann_whitney_u": float(exact.statistic),
            "mann_whitney_p_exact": float(exact.pvalue),
            "mann_whitney_p_asymptotic": float(approx.pvalue),
            "welch_p": float(welch.pvalue),
            "direction": "new cells higher" if mean_right > mean_left else "new cells lower",
        }
    return {
        "prior_seeds": list(PRIOR_SEEDS),
        "new_seeds": list(NEW_SEEDS),
        "series": rows,
        "note": (
            "four metrics and two tails, so no single p is decisive. The finding "
            "is the JOINT pattern the review named: two metrics shifting in one "
            "round with the new cells occupying three consecutive extreme ranks "
            "in both. This function does not adjust for multiplicity and does "
            "not claim significance; it publishes the arithmetic."
        ),
    }


def poolability_verdict(
    *, rescore: Mapping[str, Any], shift: Mapping[str, Any]
) -> dict[str, Any]:
    """What the control settles, and what survives it. No hedging either way."""
    reproduces = bool(rescore["the_map_side_scorer_reproduces"])
    k256 = dict(shift["series"]["ratio::k256"])
    density = dict(shift["series"]["density_v2"])
    return {
        "the_map_side_scorer_reproduces_on_seed_42": reproduces,
        "scoring_release_generations_spanned": list(SCORING_RELEASE_GENERATIONS),
        "the_sixteen_cells_are_poolable_on_the_map_side": reproduces,
        "the_k256_shift_survives_the_control": bool(
            reproduces and float(k256["mann_whitney_p_exact"]) < 0.05
        ),
        "the_density_shift_survives_the_control": bool(
            reproduces and float(density["mann_whitney_p_exact"]) < 0.05
        ),
        "k256_shift_p_exact": float(k256["mann_whitney_p_exact"]),
        "density_v2_shift_p_exact": float(density["mann_whitney_p_exact"]),
        "what_it_means_for_the_sixteen_cell_family": (
            "the family is one population as far as the instrument is concerned, "
            "so the n=16 gate is not fitted across a mixed scorer. The shift is "
            "therefore a property of seeds 55-57's MAPS and remains unexplained; "
            "the two defining-cell failures that ride on the k256 upper bound "
            "stand, and stand on an unexplained systematic rather than on an "
            "instrument artefact."
            if reproduces
            else
            "the pooled family is MIXED: at least one prior cell does not "
            "reproduce under this release, so the n=16 floors are fitted to "
            "cells scored by more than one instrument and every count derived "
            "from them inherits that. The gate must be refitted on rescored "
            "cells before it judges any rung."
        ),
        "the_instrument_this_rests_on": (
            "panel_v2.score_panel and ParametricUMAP.transform. Neither is "
            "behind an AbortPollGate and neither is defeatable by the "
            "review-0249 attribute class; the comparison targets are R0218's "
            "sealed bytes, read at run time."
        ),
    }


__all__ = [
    "NEW_SEEDS",
    "POOLABILITY_NOTE",
    "PRIOR_SEEDS",
    "PURITY_RATIO_KEYS",
    "RESCORED_SEED",
    "RESCORED_SOURCE_ROUND",
    "RESCORE_CAPABILITY",
    "RESCORE_SCHEMA",
    "ROUND_ID",
    "Round0251RescoreError",
    "SCORER_DRIFT_TOLERANCE",
    "SCORER_QUANTUM",
    "SCORING_RELEASE_GENERATIONS",
    "SHIFT_METRICS",
    "SHIFT_RATIO_KEYS",
    "compare_rescore",
    "poolability_verdict",
    "sealed_seed42_cell",
    "shift_test",
]
