"""R0250 — the calibrated robust gate re-registered at `n = 16`, under JOINT criteria.

Two binding rules from `plan-minilm-100m-v2.md` drive this module.

**"Do NOT judge any rung with the n = 13 gate."** At `n = 13` the calibrated robust
family detects a genuine `2 sigma` regression `17.15%` of the time. R0234's own
power ladder — read here from its sealed artifact rather than quoted — puts the
`n = 16` rung above it and `n = 29` at parity with the `n = 13` sample-sd gate.
`n >= 16` is the standing minimum, so the multiplier is **re-derived at 16** by
R0234's calibrated method. `k = 3.187` was `n = 8` and `3.73637` was `n = 13`;
neither may be carried.

**"A later registration does not supersede an earlier released floor."** R0231
silently reversed R0228's released `c8-seed42` `ffr` failure, which is why the joint
rule exists. Every criterion any prior family released stays in force **alongside**
this one, and a map must clear all of them. This module never types a prior floor:
`joint_criteria_from_sealed` builds the joint set from the bytes of R0225's,
R0231's and R0234's sealed artifacts, and `score_joint` ANDs the verdicts.

Everything about the estimator family, the calibration, the invariance injection,
the attainability argument, the scoring and the verdict-change enumeration is
R0234's, imported and called. This module adds exactly four things:

1. the sixteen-cell population and its `n`;
2. the joint-criteria construction and its AND;
3. `n = 16` external calibration checks read from R0234's sealed ladder;
4. the falsifiability statement at the new `n`, computed rather than asserted.

`density_v2` remains **descriptive-only** in every family, including the prior ones
that gated it: one anchor of 4,000 supplies about two-thirds of its value, so it may
not fail a map. It is reported for every family and excluded from every verdict.

Purity is decided on the **unfolded log-ratio, two-sided**, in every family. A
folded one-sided floor cannot distinguish over-separation from under-separation and
manufactures significance about a centre the family does not have.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .round0234_calibrated_floors import (
    CANDIDATES,
    CANDIDATE_CLUSTER_COUNTS,
    CANDIDATE_ORDER,
    CANDIDATE_SEEDS,
    COVERAGE_TARGET,
    COVERAGE_TOLERANCE,
    CUVS_FAMILY_SEEDS,
    DENSITY_V2_DEFECT,
    DESCRIPTIVE_METRICS,
    GATED_METRICS,
    INVARIANCE_DEPTH_LADDER,
    INVARIANCE_SIGMA_MULTIPLES,
    METRICS,
    POWER_MATERIALITY,
    POWER_SELECTION_ALTERNATIVE,
    PURITY_METRICS,
    PURITY_RATIO_KEYS,
    REQUIRED_INVARIANCE_DEPTH,
    ROBUST_CANDIDATES,
    SELECTION_RULE,
    TOLERANCE_CONFIDENCE,
    TOLERANCE_CONTENT,
    Round0234Error,
    attainability,
    band_at,
    centre_and_scale,
    degenerate_witness,
    floor_at,
    identity_bound,
    injection_ladder,
    positive_scale_witness,
    score_cell_metric,
    score_population,
    verdict_changes,
)
from .round0250_seed_extension_n16 import POOLED_SEEDS, STANDING_MINIMUM_N


ROUND_ID = "0250"

GATE_CAPABILITY = "minilm-mixed-2m-calibrated-robust-floors-n16-v1"
GATE_SCHEMA = "round0250-minilm-mixed-2m-calibrated-robust-floors-n16-v1"

#: The sixteen exact-graph cells. Seeds 42-54 exist; 55-57 are this round's.
EXACT_FAMILY_SEEDS: tuple[int, ...] = tuple(POOLED_SEEDS)
N_EXACT = len(EXACT_FAMILY_SEEDS)

#: The twelve held-out cells R0231 and R0234 both scored: R0223's three cuVS-igd48
#: cells and R0228's nine cluster-spill c4/c8/c16 x seeds 42/43/44. They define
#: nothing, so they are the only cells that could ever genuinely test a floor.
N_HELD_OUT = len(CUVS_FAMILY_SEEDS) + len(CANDIDATE_CLUSTER_COUNTS) * len(
    CANDIDATE_SEEDS
)

IDENTITY_BOUND_AT_N = identity_bound(N_EXACT)


class Round0250GateError(RuntimeError):
    """The registered R0250 n=16 calibrated-floor contract changed."""


# --------------------------------------------------------------------------- #
# the joint criteria
# --------------------------------------------------------------------------- #

JOINT_CRITERIA_RULE = (
    "BINDING (plan-minilm-100m-v2.md): a map must clear the JOINT criteria -- "
    "R0234's calibrated floors AND the retained R0225/R0231 ones AND this "
    "round's n=16 registration. A later registration does not supersede an "
    "earlier released floor. Every criterion is read from the SEALED artifact of "
    "the round that registered it; none is typed here. Each family contributes "
    "its own published instrument for each gated metric: a one-sided lower floor "
    "for `ffr`, and a two-sided band on the UNFOLDED purity log-ratio for the two "
    "purity metrics -- never a folded one-sided purity floor, which cannot "
    "distinguish over- from under-separation. `density_v2` contributes NOTHING to "
    "any verdict in any family: it is descriptive-only by the same binding rule. "
    "The joint verdict for a cell is the AND over families of the AND over gated "
    "metrics; a cell that any family fails on any gated metric does not clear."
)

#: The prior families whose criteria are retained, newest last. Each entry names
#: the capability whose sealed artifact supplies the numbers and how to read it.
RETAINED_FAMILY_SOURCES: tuple[dict[str, str], ...] = (
    {
        "family": "r0225_n8_tolerance_95_95",
        "capability": "minilm-mixed-2m-tolerance-gates-n8-v1",
        "round_id": "0225",
        "reader": "gate.gates[metric].one_sided_tolerance_95_95 / .two_sided_log_ratio_95_95",
        "why_retained": (
            "R0234 §F retained this family's `ffr` floor explicitly because its "
            "own calibrated floor passes cluster-spill-c8-seed42, which R0228 "
            "published as a released failure. R0234's machine-readable "
            "`retained_stricter_criteria` watch-list covered only this family, "
            "which R0234 itself disclosed as an incomplete field; R0250 retains "
            "R0231's released band too rather than inherit that gap."
        ),
    },
    {
        "family": "r0231_n13_median_minus_3_mad",
        "capability": "minilm-mixed-2m-robust-floors-n13-v1",
        "round_id": "0231",
        "reader": "registered_floors[ffr] / registered_two_sided_bands[purity]",
        "why_retained": (
            "R0234 published `supersedes: []`, so R0231's released band remains "
            "in force alongside it; R0234 §F.2 states this in prose but its "
            "`retained_stricter_criteria` field does not record it."
        ),
    },
    {
        "family": "r0234_n13_calibrated_robust",
        "capability": "minilm-mixed-2m-calibrated-robust-floors-n13-v1",
        "round_id": "0234",
        "reader": "registered_floors[ffr] / registered_two_sided_bands[purity]",
        "why_retained": (
            "the calibrated n=13 registration. Its purity entries in "
            "`registered_floors` are FOLDED and descriptive -- the round publishes "
            "them under `registered_floors_purity_entries_are_descriptive` -- so "
            "the gated purity criterion is its two-sided unfolded band."
        ),
    },
)

THIS_FAMILY = "r0250_n16_calibrated_robust"


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise Round0250GateError(f"R0250 {label} is not finite: {value!r}")
    return number


def criteria_from_r0225(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """R0225's released one-sided `ffr` floor and its unfolded purity bands."""
    gates = dict(dict(artifact["gate"])["gates"])
    floors = {
        "ffr": _finite(
            gates["ffr"]["one_sided_tolerance_95_95"]["floor"], "R0225 ffr floor"
        )
    }
    bands: dict[str, tuple[float, float]] = {}
    for metric in PURITY_METRICS:
        entry = gates[metric]["two_sided_log_ratio_95_95"]
        bands[metric] = (
            _finite(entry["ratio_lower"], f"R0225 {metric} lower"),
            _finite(entry["ratio_upper"], f"R0225 {metric} upper"),
        )
    return {
        "family": "r0225_n8_tolerance_95_95",
        "capability": str(artifact["capability"]),
        "round_id": str(artifact["round_id"]),
        "n": int(dict(artifact["gate"])["n"]),
        "floors": floors,
        "bands": bands,
        "gate_status": str(artifact.get("gate_status") or ""),
    }


def criteria_from_registered_floors(artifact: Mapping[str, Any], *, family: str) -> dict[str, Any]:
    """R0231's and R0234's shape: `registered_floors` + `registered_two_sided_bands`."""
    registered = dict(artifact["registered_floors"])
    bands_in = dict(artifact["registered_two_sided_bands"])
    floors = {"ffr": _finite(registered["ffr"], f"{family} ffr floor")}
    bands: dict[str, tuple[float, float]] = {}
    for metric in PURITY_METRICS:
        entry = dict(bands_in[metric])
        bands[metric] = (
            _finite(entry["ratio_lower"], f"{family} {metric} lower"),
            _finite(entry["ratio_upper"], f"{family} {metric} upper"),
        )
    return {
        "family": family,
        "capability": str(artifact["capability"]),
        "round_id": str(artifact["round_id"]),
        "n": int(artifact["n"]),
        "floors": floors,
        "bands": bands,
        "gate_status": str(artifact.get("gate_status") or ""),
    }


def joint_criteria_from_sealed(
    *,
    r0225: Mapping[str, Any],
    r0231: Mapping[str, Any],
    r0234: Mapping[str, Any],
    this_round: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """The joint set, oldest first, every number read from sealed bytes."""
    families = [
        criteria_from_r0225(r0225),
        criteria_from_registered_floors(r0231, family="r0231_n13_median_minus_3_mad"),
        criteria_from_registered_floors(r0234, family="r0234_n13_calibrated_robust"),
        {
            "family": THIS_FAMILY,
            "capability": GATE_CAPABILITY,
            "round_id": ROUND_ID,
            "n": N_EXACT,
            "floors": {"ffr": _finite(this_round["floors"]["ffr"], "R0250 ffr floor")},
            "bands": {
                metric: (
                    _finite(this_round["bands"][metric][0], f"R0250 {metric} lower"),
                    _finite(this_round["bands"][metric][1], f"R0250 {metric} upper"),
                )
                for metric in PURITY_METRICS
            },
            "gate_status": str(this_round.get("gate_status") or ""),
        },
    ]
    expected = [item["family"] for item in RETAINED_FAMILY_SOURCES] + [THIS_FAMILY]
    if [item["family"] for item in families] != expected:
        raise Round0250GateError("R0250 joint-criteria family order changed")
    for item in families:
        if set(item["floors"]) != {"ffr"} or set(item["bands"]) != set(PURITY_METRICS):
            raise Round0250GateError(
                f"R0250 joint family {item['family']} does not cover every gated "
                "metric exactly once"
            )
        if set(item["floors"]) & set(DESCRIPTIVE_METRICS):
            raise Round0250GateError(
                f"R0250 joint family {item['family']} tried to gate a "
                "descriptive-only metric"
            )
    return families


def score_joint(
    *,
    cells: Sequence[Mapping[str, Any]],
    families: Sequence[Mapping[str, Any]],
    defining_cell_ids: Sequence[str],
    every_defining_cell_can_fail: bool,
) -> dict[str, Any]:
    """Score every cell against every retained family and AND the verdicts.

    Each family is scored with R0234's own `score_population`, so the per-cell,
    per-metric arithmetic is the released one; this function only conjoins.
    """
    per_family: dict[str, Any] = {}
    for item in families:
        per_family[str(item["family"])] = {
            "capability": item["capability"],
            "round_id": item["round_id"],
            "n": item["n"],
            "gate_status": item["gate_status"],
            "floors": dict(item["floors"]),
            "bands": {key: list(value) for key, value in item["bands"].items()},
            "scoring": score_population(
                cells=cells,
                floors=item["floors"],
                bands=item["bands"],
                metrics=GATED_METRICS,
                defining_cell_ids=defining_cell_ids,
                every_defining_cell_can_fail=every_defining_cell_can_fail,
            ),
        }
    defining = {str(item) for item in defining_cell_ids}
    rows: list[dict[str, Any]] = []
    for cell in cells:
        cell_id = str(cell["cell_id"])
        by_family: dict[str, bool] = {}
        binding: list[dict[str, Any]] = []
        for name, entry in per_family.items():
            row = next(
                item for item in entry["scoring"]["cells"] if item["cell_id"] == cell_id
            )
            by_family[name] = bool(row["clears_every_gated_metric"])
            for metric, verdict in row["metrics"].items():
                if not verdict["passes"]:
                    binding.append({
                        "family": name,
                        "metric": metric,
                        "capability": entry["capability"],
                    })
        rows.append({
            "cell_id": cell_id,
            "family": str(cell["family"]),
            "defines_the_floors": cell_id in defining,
            "clears_by_family": by_family,
            "clears_the_joint_criteria": all(by_family.values()),
            "binding_failures": binding,
        })
    cleared = [row for row in rows if row["clears_the_joint_criteria"]]
    return {
        "rule": JOINT_CRITERIA_RULE,
        "families": [str(item["family"]) for item in families],
        "per_family": per_family,
        "cells": rows,
        "cells_scored": len(rows),
        "cells_clearing_the_joint_criteria": len(cleared),
        "defining_cells_clearing": sum(
            1 for row in cleared if row["defines_the_floors"]
        ),
        "held_out_cells_clearing": sum(
            1 for row in cleared if not row["defines_the_floors"]
        ),
        "cells_failing_only_a_retained_family": [
            row["cell_id"]
            for row in rows
            if not row["clears_the_joint_criteria"]
            and row["clears_by_family"].get(THIS_FAMILY) is True
        ],
        "every_defining_cell_can_fail": bool(every_defining_cell_can_fail),
        "count_is_attainable": bool(every_defining_cell_can_fail) or any(
            not row["defines_the_floors"] for row in rows
        ),
    }


# --------------------------------------------------------------------------- #
# falsifiability at the new n, computed
# --------------------------------------------------------------------------- #


def falsifiability_statement(
    *, multiplier_one_sided: float, multiplier_two_sided: float, n: int = N_EXACT
) -> dict[str, Any]:
    """Can a defining cell fail its own floor at `n`? Answered, not asserted.

    Two different answers are due, because two different bounds apply. For a
    `mean - k*s` family the bound on `max|z|` is the identity `(n-1)/sqrt(n)`, so
    the family can falsify itself iff `k` is below it. For the robust
    `median - k*MAD_n` family R0234's rank-slack argument gives `+inf`, so the
    answer is yes at every multiplier and the identity is not the operative test.
    """
    bound = identity_bound(int(n))
    robust = attainability(
        "median_minus_k_madn", n=int(n), multiplier=float(multiplier_one_sided)
    )
    variance = attainability(
        "mean_minus_k_sample_sd", n=int(n), multiplier=float(multiplier_one_sided)
    )
    return {
        "n": int(n),
        "identity_bound_max_abs_z": bound,
        "registered_one_sided_multiplier": float(multiplier_one_sided),
        "registered_two_sided_multiplier": float(multiplier_two_sided),
        "one_sided_multiplier_below_the_identity_bound": (
            float(multiplier_one_sided) < bound
        ),
        "two_sided_multiplier_below_the_identity_bound": (
            float(multiplier_two_sided) < bound
        ),
        "registered_family_bound_is_finite": bool(robust["bound_is_finite"]),
        "registered_family_every_defining_cell_can_fail": bool(
            robust["every_defining_cell_can_fail"]
        ),
        "registered_family_attainability": robust,
        "reference_variance_family_attainability": variance,
        "plain_statement": (
            f"At n = {int(n)} the identity bounds max|x - xbar|/s at {bound!r}. The "
            "registered family is median - k*MAD_n, whose rank-slack bound is +inf, "
            "so EVERY defining cell can fall below its own floor at ANY multiplier "
            "and the family CAN falsify itself. The identity is reported because it "
            "is the operative test for the mean - k*s families scored beside it: "
            f"their one-sided multiplier {float(multiplier_one_sided)!r} "
            + ("is" if float(multiplier_one_sided) < bound else "is NOT")
            + f" below {bound!r}."
        ),
    }


__all__ = [
    "CANDIDATES",
    "CANDIDATE_CLUSTER_COUNTS",
    "CANDIDATE_ORDER",
    "CANDIDATE_SEEDS",
    "COVERAGE_TARGET",
    "COVERAGE_TOLERANCE",
    "CUVS_FAMILY_SEEDS",
    "DENSITY_V2_DEFECT",
    "DESCRIPTIVE_METRICS",
    "EXACT_FAMILY_SEEDS",
    "GATED_METRICS",
    "GATE_CAPABILITY",
    "GATE_SCHEMA",
    "IDENTITY_BOUND_AT_N",
    "INVARIANCE_DEPTH_LADDER",
    "INVARIANCE_SIGMA_MULTIPLES",
    "JOINT_CRITERIA_RULE",
    "METRICS",
    "N_EXACT",
    "N_HELD_OUT",
    "POWER_MATERIALITY",
    "POWER_SELECTION_ALTERNATIVE",
    "PURITY_METRICS",
    "PURITY_RATIO_KEYS",
    "REQUIRED_INVARIANCE_DEPTH",
    "RETAINED_FAMILY_SOURCES",
    "ROBUST_CANDIDATES",
    "ROUND_ID",
    "Round0234Error",
    "Round0250GateError",
    "SELECTION_RULE",
    "STANDING_MINIMUM_N",
    "THIS_FAMILY",
    "TOLERANCE_CONFIDENCE",
    "TOLERANCE_CONTENT",
    "attainability",
    "band_at",
    "centre_and_scale",
    "criteria_from_r0225",
    "criteria_from_registered_floors",
    "degenerate_witness",
    "falsifiability_statement",
    "floor_at",
    "identity_bound",
    "injection_ladder",
    "joint_criteria_from_sealed",
    "positive_scale_witness",
    "score_cell_metric",
    "score_joint",
    "score_population",
    "verdict_changes",
]
