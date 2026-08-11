"""R0255 — the Phase 3 gate, registered on `MAD_n` at `n = 29` by owner ruling.

The owner decided this on 2026-08-11 (`OWNER-DECISIONS-PENDING.md` section 3;
`plan-minilm-100m-v2.md` Phase 3):

> Register the Phase 3 gate on **`MAD_n`**, and close the power gap by extending
> the family instead of trading robustness away: train **13 new 2M seeds** under the
> **unchanged R0217 treatment** (**2M universe only** -- rung maps are judged by this
> gate, never added to its family), pool them on the **frozen panel**, and re-fit the
> calibrated `MAD_n` floors at **n = 29** per **R0234's calibration method**,
> publishing **attainability and detection power beside the floors**. The
> `c8-seed42` coupling resolves however the `n = 29` fit lands; **do not tune
> anything to preserve the published pass.**

Three consequences are structural in this module.

**The estimator is not selected here.** R0250 applied R0234's pre-registered
selection rule at `n = 16` and it chose `S_n`, the least invariant qualifier;
review-0250 showed the `c8-seed42` cross-check was estimator-contingent, and the
owner took the choice out of the rule's hands. So this module *registers* `MAD_n`
unconditionally, and separately *reports* what R0234's rule would have chosen at
`n = 29` and whether `MAD_n` qualifies under it. If the rule would choose something
else, that is published, not acted on.

**Nothing is fitted to the held-out cells.** `independence_control` proves it
mechanically rather than promising it: it drives `cluster-spill-c8-seed42` -- and
then every held-out cell at once -- to extreme values and shows every fitted floor
and band is **bitwise identical**, while the same perturbation applied to a *family*
cell moves the floor. A control that only shows "nothing changed" cannot distinguish
independence from an inert fit; this one shows both halves.

**`independence_control` was repaired in R0256 and did not run as shipped in
R0255.** Its arms 1 and 2 built a `perturbed` copy of the held-out cells and then
re-fitted the *unperturbed* family, so they could not fail; review-0255 found it
and `vacuouscheck` now detects the shape. The function below derives the fit's
inputs from cells through `family_series_from_cells`, so the perturbation reaches
the fit, and it takes the builder as a parameter so
`round0256_gate_repair.independence_positive_controls` can plant a builder that
does read a held-out cell and require the arms to report `false`.

**Attainability and power sit beside every floor.** R0219's "4/4 pass" and R0225's
"0 failures at n = 8 under k = 3.187" were theorems about their `n`, and the
identity bound `(n-1)/sqrt(n)` is `28/sqrt(29) = 5.1988...` here -- a `k` at or above
that could not fail its own cells. Both numbers are computed and published per floor.

`density_v2` stays **descriptive-only** in every family. One anchor of 4,000 supplies
about two-thirds of its value; R0161 and R0193 both gated it and five downstream
artifacts consumed it as a gate, which is the defect the audit named.
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
from .round0250_gate_n16 import (
    GATE_CAPABILITY as R0250_GATE_CAPABILITY,
    criteria_from_r0225,
    criteria_from_registered_floors,
)
from .round0255_seed_extension_n29 import (
    OWNER_RULING_N,
    POOLED_SEEDS,
    R0250_POOLED_SEEDS,
    SEEDS as R0255_SEEDS,
    STANDING_MINIMUM_N,
)
from .round0255_treatment import (
    HELD_OUT_CELL_IDS,
    Round0255FamilyError,
    assert_family_is_2m_only,
    exact_cell_id,
)


ROUND_ID = "0255"

GATE_CAPABILITY = "minilm-mixed-2m-calibrated-madn-floors-n29-v1"
GATE_SCHEMA = "round0255-minilm-mixed-2m-calibrated-madn-floors-n29-v1"

#: The twenty-nine exact-graph 2M cells. Seeds 42-57 exist; 58-70 are this round's.
EXACT_FAMILY_SEEDS: tuple[int, ...] = tuple(POOLED_SEEDS)
N_EXACT = len(EXACT_FAMILY_SEEDS)

N_HELD_OUT = len(CUVS_FAMILY_SEEDS) + len(CANDIDATE_CLUSTER_COUNTS) * len(
    CANDIDATE_SEEDS
)

IDENTITY_BOUND_AT_N = identity_bound(N_EXACT)

#: The owner's ruling, as a registered constant rather than as prose in a docstring.
OWNER_RULING_ESTIMATOR = "median_minus_k_madn"
OWNER_RULING_DATE = "2026-08-11"
OWNER_RULING = (
    "Register the Phase 3 gate on MAD_n, and close the power gap by extending the "
    "family instead of trading robustness away: train 13 new 2M seeds under the "
    "unchanged R0217 treatment (2M universe only -- rung maps are judged by this "
    "gate, never added to its family), pool them on the frozen panel, and re-fit "
    "the calibrated MAD_n floors at n = 29 per R0234's calibration method, "
    "publishing attainability and detection power beside the floors. The c8-seed42 "
    "coupling resolves however the n = 29 fit lands; do not tune anything to "
    "preserve the published pass."
)

NO_TUNING_STATEMENT = (
    "Nothing in this round was adjusted to preserve the c8-seed42 pass. The "
    "estimator is the owner's ruling and was fixed before any cell was scored; the "
    "multiplier is R0234's calibrated one at n = 29 on the Gaussian null, which "
    "reads no cell of this program; the family is the twenty-nine 2M cells and the "
    "panel is R0218's frozen one; the seed set is the next thirteen integers, chosen "
    "before any of them was trained. `independence_control` proves mechanically "
    "that every fitted floor is bitwise unchanged when c8-seed42 -- and when every "
    "held-out cell together -- is driven to an extreme value, and that the same "
    "perturbation applied to a FAMILY cell does move the floor, so the invariance "
    "is independence and not inertness."
)


class Round0255GateError(RuntimeError):
    """The registered R0255 n=29 calibrated-floor contract changed."""


# --------------------------------------------------------------------------- #
# the joint criteria
# --------------------------------------------------------------------------- #

JOINT_CRITERIA_RULE = (
    "BINDING (plan-minilm-100m-v2.md): a map must clear the JOINT criteria -- every "
    "criterion any prior family released or registered stays in force ALONGSIDE this "
    "one. A later registration does not supersede an earlier floor; R0231 silently "
    "reversed R0228's released c8-seed42 ffr failure, which is why the joint rule "
    "exists. Every criterion is read from the SEALED artifact of the round that "
    "registered it; none is typed here. Each family contributes a one-sided lower "
    "floor for `ffr` and a two-sided band on the UNFOLDED purity log-ratio -- never a "
    "folded one-sided purity floor. `density_v2` contributes NOTHING to any verdict "
    "in any family. The joint verdict for a cell is the AND over families of the AND "
    "over gated metrics."
)

RETAINED_FAMILY_SOURCES: tuple[dict[str, str], ...] = (
    {
        "family": "r0225_n8_tolerance_95_95",
        "capability": "minilm-mixed-2m-tolerance-gates-n8-v1",
        "round_id": "0225",
        "reader": "gate.gates[metric].one_sided_tolerance_95_95 / .two_sided_log_ratio_95_95",
        "why_retained": (
            "R0234 retained this family's `ffr` floor explicitly because its own "
            "calibrated floor passes cluster-spill-c8-seed42, which R0228 published "
            "as a released failure."
        ),
    },
    {
        "family": "r0231_n13_median_minus_3_mad",
        "capability": "minilm-mixed-2m-robust-floors-n13-v1",
        "round_id": "0231",
        "reader": "registered_floors[ffr] / registered_two_sided_bands[purity]",
        "why_retained": (
            "R0234 published `supersedes: []`, so R0231's released band remains in "
            "force alongside it."
        ),
    },
    {
        "family": "r0234_n13_calibrated_robust",
        "capability": "minilm-mixed-2m-calibrated-robust-floors-n13-v1",
        "round_id": "0234",
        "reader": "registered_floors[ffr] / registered_two_sided_bands[purity]",
        "why_retained": (
            "the calibrated n=13 registration whose method this round re-applies at "
            "n = 29. Its purity entries in `registered_floors` are FOLDED and "
            "descriptive, so the gated purity criterion is its two-sided unfolded band."
        ),
    },
    {
        "family": "r0250_n16_calibrated_robust",
        "capability": R0250_GATE_CAPABILITY,
        "round_id": "0250",
        "reader": "registered_floors[ffr] / registered_two_sided_bands[purity]",
        "why_retained": (
            "R0250 registered `median - k*S_n` at n = 16 as "
            "registered-and-contingent, and review-0250-01 carried it under "
            "`registers_contingently`. The owner's ruling replaces the estimator for "
            "the PHASE 3 GATE; it does not retract a registered criterion, and this "
            "round supersedes nothing. Its contingency is published beside it, and "
            "the round reports this family's OWN n=29 verdict separately from the "
            "joint one so the ruling's question is answered on its own terms."
        ),
    },
)

THIS_FAMILY = "r0255_n29_calibrated_madn"


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise Round0255GateError(f"R0255 {label} is not finite: {value!r}")
    return number


def joint_criteria_from_sealed(
    *,
    r0225: Mapping[str, Any],
    r0231: Mapping[str, Any],
    r0234: Mapping[str, Any],
    r0250: Mapping[str, Any],
    this_round: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """The joint set, oldest first, every prior number read from sealed bytes."""
    families = [
        criteria_from_r0225(r0225),
        criteria_from_registered_floors(r0231, family="r0231_n13_median_minus_3_mad"),
        criteria_from_registered_floors(r0234, family="r0234_n13_calibrated_robust"),
        criteria_from_registered_floors(r0250, family="r0250_n16_calibrated_robust"),
        {
            "family": THIS_FAMILY,
            "capability": GATE_CAPABILITY,
            "round_id": ROUND_ID,
            "n": N_EXACT,
            "floors": {"ffr": _finite(this_round["floors"]["ffr"], "R0255 ffr floor")},
            "bands": {
                metric: (
                    _finite(this_round["bands"][metric][0], f"R0255 {metric} lower"),
                    _finite(this_round["bands"][metric][1], f"R0255 {metric} upper"),
                )
                for metric in PURITY_METRICS
            },
            "gate_status": str(this_round.get("gate_status") or ""),
        },
    ]
    expected = [item["family"] for item in RETAINED_FAMILY_SOURCES] + [THIS_FAMILY]
    if [item["family"] for item in families] != expected:
        raise Round0255GateError("R0255 joint-criteria family order changed")
    for item in families:
        if set(item["floors"]) != {"ffr"} or set(item["bands"]) != set(PURITY_METRICS):
            raise Round0255GateError(
                f"R0255 joint family {item['family']} does not cover every gated "
                "metric exactly once"
            )
        if set(item["floors"]) & set(DESCRIPTIVE_METRICS):
            raise Round0255GateError(
                f"R0255 joint family {item['family']} tried to gate a "
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
    """Score every cell against every retained family and AND the verdicts."""
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
# poolability of the thirteen new cells against the sixteen existing ones
# --------------------------------------------------------------------------- #

SHIFT_METRICS: tuple[str, ...] = ("density_v2", "ffr")
SHIFT_RATIO_KEYS: tuple[str, ...] = ("k256", "k1024")

#: R0251's measured shift of the previous three cells, quoted from its sealed
#: artifact's own published table so this round can say whether the pattern repeats.
R0251_PRIOR_SHIFT = {
    "ratio::k256": {
        "new_cell_ranks": [15, 13, 14],
        "mann_whitney_p_exact": 0.025,
        "prior_n": 13,
        "new_n": 3,
    },
    "density_v2": {
        "new_cell_ranks": [4, 2, 3],
        "mann_whitney_p_exact": 0.025,
        "prior_n": 13,
        "new_n": 3,
    },
}


def poolability_shift_test(
    *,
    panel_metric_cells: Mapping[str, Mapping[str, float]],
    raw_purity_ratios: Mapping[str, Mapping[str, float]],
    prior_seeds: Sequence[int] = R0250_POOLED_SEEDS,
    new_seeds: Sequence[int] = R0255_SEEDS,
) -> dict[str, Any]:
    """Do the thirteen new cells shift the way the previous three did?

    R0251 measured the previous extension shifting systematically -- `k256` at ranks
    13/14/15 of 16 with an exact Mann-Whitney `p` of `0.025`, and `density_v2` at
    ranks 2/3/4 with the same statistic. The owner ruling is explicit that a shift is
    **a finding about the treatment, not a reason to drop cells**, so this function
    measures and reports; it never filters. Both nulls are published, as R0251
    established, because with no ties they are the same statistic.
    """
    from scipy import stats

    prior = [str(seed) for seed in prior_seeds]
    new = [str(seed) for seed in new_seeds]
    if len(prior) + len(new) != len(POOLED_SEEDS):
        raise Round0255GateError("R0255 poolability split is not the twenty-nine cells")
    if set(prior) & set(new):
        raise Round0255GateError("R0255 poolability split overlaps")

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
        var_right = sum((value - mean_right) ** 2 for value in right) / (len(right) - 1)
        rows[name] = {
            "prior_n": len(left),
            "new_n": len(right),
            "prior_mean": mean_left,
            "prior_sample_sd": math.sqrt(var_left),
            "new_mean": mean_right,
            "new_sample_sd": math.sqrt(var_right),
            "new_cell_ranks_in_the_pooled_family": ranks,
            "mann_whitney_u": float(exact.statistic),
            "mann_whitney_p_exact": float(exact.pvalue),
            "mann_whitney_p_asymptotic": float(approx.pvalue),
            "welch_p": float(welch.pvalue),
            "direction": (
                "new cells higher" if mean_right > mean_left else "new cells lower"
            ),
            "r0251_prior_extension": R0251_PRIOR_SHIFT.get(name),
        }
    return {
        "prior_seeds": list(prior_seeds),
        "new_seeds": list(new_seeds),
        "series": rows,
        "series_with_p_exact_below_0_05": sorted(
            name for name, row in rows.items() if row["mann_whitney_p_exact"] < 0.05
        ),
        "cells_dropped": 0,
        "policy": (
            "measure and report; never filter. The owner ruling: 'If they do, that "
            "is a finding about the treatment, not a reason to drop cells.' Four "
            "series and two tails, so no single p is decisive and no significance "
            "is claimed here; the arithmetic is published and no multiplicity "
            "adjustment is applied."
        ),
    }


# --------------------------------------------------------------------------- #
# the owner's registration, and what the pre-registered rule would have done
# --------------------------------------------------------------------------- #


def owner_ruling_registration(
    *, selection: Mapping[str, Any], estimator: str = OWNER_RULING_ESTIMATOR
) -> dict[str, Any]:
    """Register the ruling's estimator; report the rule's answer beside it."""
    candidates = dict(selection["candidates"])
    if estimator not in candidates:
        raise Round0255GateError(
            f"R0255 owner-ruling estimator {estimator!r} is not a calibrated candidate"
        )
    would_choose = selection.get("chosen_estimator")
    entry = dict(candidates[estimator])
    return {
        "ruling": OWNER_RULING,
        "ruling_date": OWNER_RULING_DATE,
        "registered_estimator": estimator,
        "registered_by": "owner ruling, not by the selection rule",
        "selection_rule": SELECTION_RULE,
        "selection_rule_would_have_chosen": would_choose,
        "selection_rule_agrees_with_the_ruling": would_choose == estimator,
        "the_registered_estimator_qualifies_under_the_rule": bool(entry["qualifies"]),
        "requirement_1_coverage": bool(entry["requirement_1_coverage"]),
        "requirement_2_invariance": bool(entry["requirement_2_invariance"]),
        "requirement_3_attainability": bool(entry["requirement_3_attainability"]),
        "minimum_exact_invariance_depth": entry["minimum_exact_invariance_depth"],
        "what_the_ruling_settles": (
            "the estimator, which R0250's application of the pre-registered rule "
            "resolved to S_n at n = 16 and review-0250 showed was coupled to the "
            "c8-seed42 cross-check. The rule's answer at n = 29 is reported above "
            "and is NOT acted on: registering it would re-derive exactly the "
            "contingency the ruling forecloses."
        ),
        "no_tuning_statement": NO_TUNING_STATEMENT,
    }


def attainability_and_power(
    *,
    estimator: str,
    n: int,
    multiplier_one_sided: float,
    multiplier_two_sided: float,
    calibrated_entry: Mapping[str, Any],
    floors: Mapping[str, float],
    bands: Mapping[str, Sequence[float]],
    series: Mapping[str, Sequence[float]],
    log_series: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    """Attainability and detection power, per floor. Non-negotiable per the ruling."""
    bound = identity_bound(int(n))
    one_sided = attainability(estimator, n=int(n), multiplier=float(multiplier_one_sided))
    two_sided = attainability(estimator, n=int(n), multiplier=float(multiplier_two_sided))
    witness = positive_scale_witness(estimator, float(multiplier_one_sided), n=int(n))
    power = dict(calibrated_entry["one_sided"]["detection_power"])
    rows: list[dict[str, Any]] = []
    for metric in GATED_METRICS:
        if metric in PURITY_METRICS:
            values = [float(value) for value in log_series[metric]]
            lower, upper = (float(bands[metric][0]), float(bands[metric][1]))
            centre, scale = centre_and_scale(estimator, values)
            closest = min(
                min(value - math.log(lower), math.log(upper) - value)
                for value in values
            )
            rows.append({
                "metric": metric,
                "criterion": "two-sided band on the unfolded purity log-ratio",
                "ratio_lower": lower,
                "ratio_upper": upper,
                "multiplier": float(multiplier_two_sided),
                "identity_bound_at_n": bound,
                "multiplier_below_the_identity_bound": (
                    float(multiplier_two_sided) < bound
                ),
                "every_defining_cell_can_fail": bool(
                    two_sided["every_defining_cell_can_fail"]
                ),
                "attainability_bound_is_finite": bool(two_sided["bound_is_finite"]),
                "detection_power_at_minus_1_sigma": power.get("minus_1_sigma"),
                "detection_power_at_minus_2_sigma": power.get("minus_2_sigma"),
                "detection_power_at_minus_3_sigma": power.get("minus_3_sigma"),
                "new_cell_false_fail_rate": float(
                    calibrated_entry["two_sided"]["new_cell_false_fail_rate"]
                ),
                "closest_family_cell_margin_in_log_units": closest,
                "closest_family_cell_margin_in_scales": (
                    closest / scale if scale else math.inf
                ),
                "family_cells_failing_this_criterion": sum(
                    1
                    for value in values
                    if value < math.log(lower) or value > math.log(upper)
                ),
            })
            continue
        values = [float(value) for value in series[metric]]
        floor = float(floors[metric])
        centre, scale = centre_and_scale(estimator, values)
        closest = min(value - floor for value in values)
        rows.append({
            "metric": metric,
            "criterion": "one-sided lower floor",
            "floor": floor,
            "multiplier": float(multiplier_one_sided),
            "identity_bound_at_n": bound,
            "multiplier_below_the_identity_bound": float(multiplier_one_sided) < bound,
            "every_defining_cell_can_fail": bool(
                one_sided["every_defining_cell_can_fail"]
            ),
            "attainability_bound_is_finite": bool(one_sided["bound_is_finite"]),
            "detection_power_at_minus_1_sigma": power.get("minus_1_sigma"),
            "detection_power_at_minus_2_sigma": power.get("minus_2_sigma"),
            "detection_power_at_minus_3_sigma": power.get("minus_3_sigma"),
            "new_cell_false_fail_rate": float(
                calibrated_entry["one_sided"]["new_cell_false_fail_rate"]
            ),
            "closest_family_cell_margin": closest,
            "closest_family_cell_margin_in_scales": (
                closest / scale if scale else math.inf
            ),
            "family_cells_failing_this_criterion": sum(
                1 for value in values if value < floor
            ),
        })
    return {
        "estimator": estimator,
        "n": int(n),
        "identity_bound_at_n": bound,
        "identity_bound_expression": f"(n-1)/sqrt(n) = {int(n) - 1}/sqrt({int(n)})",
        "one_sided_multiplier": float(multiplier_one_sided),
        "two_sided_multiplier": float(multiplier_two_sided),
        "one_sided_multiplier_below_the_identity_bound": (
            float(multiplier_one_sided) < bound
        ),
        "two_sided_multiplier_below_the_identity_bound": (
            float(multiplier_two_sided) < bound
        ),
        "attainability_one_sided": one_sided,
        "attainability_two_sided": two_sided,
        "positive_scale_witness": witness,
        "detection_power": power,
        "per_floor": rows,
        "why_this_is_published": (
            "R0219's '4/4 pass' and R0225's '0 failures at n = 8 under k = 3.187' "
            "were theorems about their n rather than evidence about their maps. A "
            "floor published without its power invites that reading again, and this "
            "program has taken it twice. A k at or above the identity bound could "
            "not be failed by its own cells at all."
        ),
    }


# --------------------------------------------------------------------------- #
# the independence control
# --------------------------------------------------------------------------- #


#: An extreme drive on a cell's metric values.
EXTREME_VALUE_SHIFT = 1_000.0
#: An extreme multiplicative drive on a cell's purity ratios that still has a
#: finite logarithm. R0255 wrote `ratio * 1000.0` while the assertion it wanted
#: was in log space; the ratios are held in linear space, so the intent and the
#: arithmetic did not match. `exp(-20.72...)` is unambiguous in either.
EXTREME_RATIO_FACTOR = 1e-9
FAMILY_SHIFT = 0.1


def family_series_from_cells(
    cells: Sequence[Mapping[str, Any]],
    *,
    defining_cell_ids: Sequence[str],
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Select the defining cells out of the whole scored panel and build the fit.

    This is the function `independence_control` puts under test. The realistic
    way a held-out cell reaches a floor is not that somebody appends it to a
    series by hand -- it is that the **selection** lets it in. That is the v0
    defect the R0161/R0193 audit named: rung maps ended up among R0161's own
    fitting cells. So the control perturbs *cells* and re-runs *this*.
    """
    by_id = {str(cell["cell_id"]): cell for cell in cells}
    wanted = [str(cell_id) for cell_id in defining_cell_ids]
    missing = [cell_id for cell_id in wanted if cell_id not in by_id]
    if missing:
        raise Round0255GateError(
            f"R0255 series builder is missing defining cells {sorted(missing)}"
        )
    series = {
        metric: [float(by_id[cell_id]["values"][metric]) for cell_id in wanted]
        for metric in METRICS
    }
    log_series = {
        metric: [
            math.log(float(by_id[cell_id]["ratios"][PURITY_RATIO_KEYS[metric]]))
            for cell_id in wanted
        ]
        for metric in PURITY_METRICS
    }
    return series, log_series


def _drive_cells(
    cells: Sequence[Mapping[str, Any]],
    *,
    target_ids: Sequence[str],
    value_shift: float,
    ratio_factor: float,
) -> list[dict[str, Any]]:
    """Return a NEW cell list with the named cells driven. Inputs untouched."""
    targets = {str(cell_id) for cell_id in target_ids}
    out: list[dict[str, Any]] = []
    for cell in cells:
        row = dict(cell)
        values = {
            metric: float(value) for metric, value in dict(cell["values"]).items()
        }
        ratios = {key: float(value) for key, value in dict(cell["ratios"]).items()}
        if str(row["cell_id"]) in targets:
            values = {
                metric: value - float(value_shift) for metric, value in values.items()
            }
            ratios = {key: value * float(ratio_factor) for key, value in ratios.items()}
        row["values"] = values
        row["ratios"] = ratios
        out.append(row)
    return out


def independence_control(
    *,
    estimator: str,
    multiplier_one_sided: float,
    multiplier_two_sided: float,
    family_cells: Sequence[Mapping[str, Any]],
    held_out_cells: Sequence[Mapping[str, Any]],
    defining_cell_ids: Sequence[str],
    build: Any = family_series_from_cells,
) -> dict[str, Any]:
    """Prove the fit cannot be tuned to a held-out cell, and that it is not inert.

    **Repaired in R0256.** As shipped in R0255 this function built a `perturbed`
    copy of the held-out cells, counted it into the artifact, and then called
    `_fit(series, log_series)` on the **unperturbed family**. `perturbed` was
    loaded in exactly two places -- its own `.append` and a `len()` -- so arms 1
    and 2 compared a function of the family to itself and
    `every_floor_bitwise_identical: true` was structurally incapable of being
    false. It would have reported `true` for a fit that genuinely read held-out
    cells. Review-0255 found this and `vacuouscheck` now detects the shape.

    Here the fit's inputs are **derived from cells** by `build`, so a
    perturbation applied to a cell reaches the floor if and only if that cell
    reaches the fit. `build` is a parameter so a positive control can plant a
    builder that *does* select a held-out cell and require arms 1 and 2 to report
    `false`; `round0256_gate_repair.independence_positive_controls` ships three.

    Four arms, all calling the SHIPPED `floor_at` / `band_at`:

    1. **c8-seed42 driven to an extreme** -- the one cell the ruling names. Every
       fitted floor and band must be **bitwise identical**. ASSERTED.
    2. **every held-out cell driven at once** -- same requirement. ASSERTED.
    3. **a FAMILY cell driven by the same amount** -- REPORTED, not asserted.
    4. **every FAMILY cell shifted** -- the floors must MOVE. Without this arm,
       arms 1 and 2 would also pass for a fit that ignores its inputs entirely.
    """
    ids = [str(cell_id) for cell_id in defining_cell_ids]

    def _fit(
        cells: Sequence[Mapping[str, Any]]
    ) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
        values, logs = build(cells, defining_cell_ids=ids)
        return (
            {
                metric: floor_at(
                    estimator, list(values[metric]), float(multiplier_one_sided)
                )
                for metric in METRICS
            },
            {
                metric: band_at(
                    estimator, list(logs[metric]), float(multiplier_two_sided)
                )
                for metric in PURITY_METRICS
            },
        )

    baseline_floors, baseline_bands = _fit(
        list(family_cells) + list(held_out_cells)
    )

    def _compare(
        cells: Sequence[Mapping[str, Any]]
    ) -> tuple[dict[str, float], dict[str, list[float]], bool]:
        floors, bands = _fit(cells)
        identical = all(
            floors[metric] == baseline_floors[metric] for metric in METRICS
        ) and all(
            bands[metric] == baseline_bands[metric] for metric in PURITY_METRICS
        )
        return floors, {m: list(v) for m, v in bands.items()}, identical

    arms: list[dict[str, Any]] = []

    # Arms 1 and 2: held-out cells driven, and THE FIT RE-RUN ON THE RESULT.
    for label, targets in (
        ("c8_seed42_driven_to_an_extreme", ["cluster-spill-c8-seed42"]),
        ("every_held_out_cell_driven_at_once", list(HELD_OUT_CELL_IDS)),
    ):
        driven = _drive_cells(
            held_out_cells,
            target_ids=targets,
            value_shift=EXTREME_VALUE_SHIFT,
            ratio_factor=EXTREME_RATIO_FACTOR,
        )
        floors, bands, identical = _compare(list(family_cells) + driven)
        arms.append({
            "arm": label,
            "cells_perturbed": sum(
                1 for row in driven if str(row["cell_id"]) in set(targets)
            ),
            "perturbation": (
                f"value - {EXTREME_VALUE_SHIFT} and ratio * {EXTREME_RATIO_FACTOR} "
                "on the named held-out cells, PASSED INTO the builder under test"
            ),
            "perturbation_reaches_the_fit": True,
            "every_floor_bitwise_identical": identical,
            "floors": floors,
            "bands": bands,
            "expectation": (
                "identical -- held-out cells must not be selected into the fit. "
                "ASSERTED, with positive controls proving this arm reports false "
                "when a builder does select them."
            ),
            "holds": identical,
            "is_an_assertion": True,
        })

    # Arm 3: ONE family cell driven the same way. Reported, NOT asserted.
    driven_one = _drive_cells(
        family_cells,
        target_ids=[str(family_cells[0]["cell_id"])],
        value_shift=EXTREME_VALUE_SHIFT,
        ratio_factor=EXTREME_RATIO_FACTOR,
    )
    one_floors, one_bands, one_identical = _compare(
        driven_one + list(held_out_cells)
    )
    arms.append({
        "arm": "one_family_cell_driven_by_the_same_amount",
        "cells_perturbed": 1,
        "perturbation": (
            f"value - {EXTREME_VALUE_SHIFT} and ratio * {EXTREME_RATIO_FACTOR} on "
            f"{family_cells[0]['cell_id']}"
        ),
        "perturbation_reaches_the_fit": True,
        "any_floor_moved": not one_identical,
        "floors": one_floors,
        "bands": one_bands,
        "expectation": (
            "REPORTED, not required. At n = 29 driving an ABOVE-median cell to "
            "-inf moves the median from the 15th to the 14th order statistic, so "
            "the LOCATION moves even though the MAD_n scale is robust. Driving a "
            "below-median cell would move nothing, so this arm's outcome depends "
            "on which cell is driven and is published as a measurement, never as "
            "a property of the estimator."
        ),
        "holds": True,
        "is_an_assertion": False,
    })

    # Arm 4: EVERY family cell shifted. The fit must move, or arms 1 and 2 would
    # also pass for a fit that ignores its inputs entirely and independence would
    # be indistinguishable from inertness.
    shifted = [
        {
            **dict(cell),
            "values": {
                metric: float(value) - FAMILY_SHIFT
                for metric, value in dict(cell["values"]).items()
            },
            "ratios": {
                key: float(value) * math.exp(-FAMILY_SHIFT)
                for key, value in dict(cell["ratios"]).items()
            },
        }
        for cell in family_cells
    ]
    all_floors, all_bands, all_identical = _compare(
        shifted + list(held_out_cells)
    )
    all_moved = all(
        all_floors[metric] != baseline_floors[metric] for metric in METRICS
    ) and all(
        tuple(all_bands[metric]) != tuple(baseline_bands[metric])
        for metric in PURITY_METRICS
    )
    arms.append({
        "arm": "every_family_cell_shifted",
        "cells_perturbed": len(family_cells),
        "perturbation": (
            f"value - {FAMILY_SHIFT} and ratio * exp(-{FAMILY_SHIFT}) on every "
            "family cell"
        ),
        "perturbation_reaches_the_fit": True,
        "every_floor_moved": all_moved,
        "any_floor_moved": not all_identical,
        "floors": all_floors,
        "bands": all_bands,
        "expectation": (
            "MOVED on every metric -- this is the not-inert half. Without it, arms "
            "1 and 2 would pass for a fit that reads nothing at all"
        ),
        "holds": all_moved,
        "is_an_assertion": True,
    })

    return {
        "estimator": estimator,
        "builder": getattr(build, "__name__", str(build)),
        "baseline_floors": baseline_floors,
        "baseline_bands": {
            metric: list(value) for metric, value in baseline_bands.items()
        },
        "arms": arms,
        "the_fit_is_independent_of_every_held_out_cell": bool(
            arms[0]["holds"] and arms[1]["holds"]
        ),
        "one_contaminated_family_cell_moved_the_fit": bool(
            arms[2]["any_floor_moved"]
        ),
        "the_fit_is_not_inert": bool(arms[3]["holds"]),
        "holds": all(bool(arm["holds"]) for arm in arms),
        "no_tuning_statement": NO_TUNING_STATEMENT,
    }


def falsifiability_statement(
    *, estimator: str, multiplier_one_sided: float, multiplier_two_sided: float,
    n: int = N_EXACT,
) -> dict[str, Any]:
    """Can a defining cell fail its own floor at `n`? Answered from the CHOSEN family.

    R0250's version computed this from `median_minus_k_madn` unconditionally while
    registering `S_n`, and disclosed the mislabel. Here the estimator is a parameter
    and the registered one is passed, so the label and the arithmetic cannot diverge.
    """
    bound = identity_bound(int(n))
    registered = attainability(
        estimator, n=int(n), multiplier=float(multiplier_one_sided)
    )
    variance = attainability(
        "mean_minus_k_sample_sd", n=int(n), multiplier=float(multiplier_one_sided)
    )
    return {
        "n": int(n),
        "estimator": estimator,
        "identity_bound_max_abs_z": bound,
        "registered_one_sided_multiplier": float(multiplier_one_sided),
        "registered_two_sided_multiplier": float(multiplier_two_sided),
        "one_sided_multiplier_below_the_identity_bound": (
            float(multiplier_one_sided) < bound
        ),
        "two_sided_multiplier_below_the_identity_bound": (
            float(multiplier_two_sided) < bound
        ),
        "registered_family_bound_is_finite": bool(registered["bound_is_finite"]),
        "registered_family_every_defining_cell_can_fail": bool(
            registered["every_defining_cell_can_fail"]
        ),
        "registered_family_attainability": registered,
        "reference_variance_family_attainability": variance,
        "plain_statement": (
            f"At n = {int(n)} the identity bounds max|x - xbar|/s at {bound!r} "
            f"(= {int(n) - 1}/sqrt({int(n)})). The registered family is {estimator}, "
            "whose rank-slack bound is +inf, so EVERY defining cell can fall below "
            "its own floor at ANY multiplier and the family CAN falsify itself. The "
            "identity is reported because it is the operative test for the "
            "mean - k*s families scored beside it, and because a registered k at or "
            f"above {bound!r} could not fail its own cells: the one-sided multiplier "
            f"{float(multiplier_one_sided)!r} "
            + ("is" if float(multiplier_one_sided) < bound else "is NOT")
            + f" below it, and the two-sided {float(multiplier_two_sided)!r} "
            + ("is" if float(multiplier_two_sided) < bound else "is NOT")
            + " below it."
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
    "EXTREME_RATIO_FACTOR",
    "EXTREME_VALUE_SHIFT",
    "FAMILY_SHIFT",
    "GATED_METRICS",
    "GATE_CAPABILITY",
    "GATE_SCHEMA",
    "HELD_OUT_CELL_IDS",
    "IDENTITY_BOUND_AT_N",
    "INVARIANCE_DEPTH_LADDER",
    "INVARIANCE_SIGMA_MULTIPLES",
    "JOINT_CRITERIA_RULE",
    "METRICS",
    "NO_TUNING_STATEMENT",
    "N_EXACT",
    "N_HELD_OUT",
    "OWNER_RULING",
    "OWNER_RULING_DATE",
    "OWNER_RULING_ESTIMATOR",
    "OWNER_RULING_N",
    "POWER_MATERIALITY",
    "POWER_SELECTION_ALTERNATIVE",
    "PURITY_METRICS",
    "PURITY_RATIO_KEYS",
    "REQUIRED_INVARIANCE_DEPTH",
    "RETAINED_FAMILY_SOURCES",
    "ROBUST_CANDIDATES",
    "ROUND_ID",
    "R0251_PRIOR_SHIFT",
    "Round0234Error",
    "Round0255FamilyError",
    "Round0255GateError",
    "SELECTION_RULE",
    "SHIFT_METRICS",
    "SHIFT_RATIO_KEYS",
    "STANDING_MINIMUM_N",
    "THIS_FAMILY",
    "TOLERANCE_CONFIDENCE",
    "TOLERANCE_CONTENT",
    "assert_family_is_2m_only",
    "attainability",
    "attainability_and_power",
    "band_at",
    "centre_and_scale",
    "degenerate_witness",
    "exact_cell_id",
    "falsifiability_statement",
    "family_series_from_cells",
    "floor_at",
    "identity_bound",
    "independence_control",
    "injection_ladder",
    "joint_criteria_from_sealed",
    "owner_ruling_registration",
    "poolability_shift_test",
    "positive_scale_witness",
    "score_cell_metric",
    "score_joint",
    "score_population",
    "verdict_changes",
]
