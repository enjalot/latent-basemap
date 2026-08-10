"""R0245 — the two DiD defects, fixed before the instrument costs 50 GPU-h.

review-0244-01 §C and §F1 named them:

4. **The decision map is prose, not an executable gate.** `DID_DECISION_RULE`
   is a real, correctly shaped rule sealed into a receipt at the release
   commit — but there is no `did_decision()` a future round must call, so
   nothing enforces it the way `did_registration` enforces the resolution
   floor. This program has shipped five checks that looked like gates and were
   not. `did_decision()` here is the gate, with the anti-vacuity clause — *an
   unpowered null is INDETERMINATE, never harmless* — as executable branches
   and a positive control that plants an underpowered null and requires
   `INDETERMINATE`.

5. **Two probe rows carry more tie-aware loss than strict loss**, which the
   population definitions do not tolerate: probe `21785` has `tie_aware = 1`
   with `strict = 0`, so it lands in **both** the treated `genuine` population
   (`tie_aware > 0`) and the `control` pool (`strict == 0`). A row on both
   sides of a paired design corrupts the pairing.

   The diagnosis is in this module and it is a scoring fact worth knowing:
   **strict and tie-aware are not nested estimators.** Strict recall is
   `strict_containment_rows` — a set test over truth *ids*, immune to float
   noise. Tie-aware recall is `tie_aware_rows` — a *value* test comparing each
   candidate's CPU-`float32`-recomputed cosine against R0238's independently
   computed true `k`-th cosine, accepted within `TIE_TOLERANCE = 1e-6`. The two
   cosine computations of the *same pair* disagree by ~`1e-6` on this
   substrate, so a candidate that IS a member of the exact truth set can fall
   below `kth - 1e-6` and be scored invalid while strict counts it. Tie
   forgiveness cannot forgive its way *down*; a tolerance below the estimator's
   own numerical agreement floor can.

   The fix is an arm-assignment rule, not a re-scoring: the registered
   populations are built on `tie_effective = min(tie_aware, strict)`, which is
   the same clip-at-zero convention `loss_decomposition` already uses for the
   partition/builder split ("never invent builder loss"). It is conservative,
   it makes `genuine` a subset of `strict > 0` by construction, and the arms
   are then disjoint by construction rather than by inspection.

6. **The permutation is unrestricted over paired values** — documented here
   with the cost computed, not accepted silently.

Nothing in this module trains a map, computes a displacement, or touches the
GPU. R0228's `density_matched_control`, R0229's `exact_displacement_permutation`
and `holm_bonferroni`, and R0244's `did_populations` / `DID_*` constants are
IMPORTED, never re-typed.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from basemap.round0227_low_c_contract import TIE_TOLERANCE
from basemap.round0229_quality_contract import (
    exact_displacement_permutation,
    holm_bonferroni,
)
from basemap.round0244_prereq import (
    DID_ALPHA,
    DID_DECISION_RULE,
    DID_MAPS_PER_ARM,
    DID_SD_EQUIVALENCE_BOUND,
    DID_TESTS_IN_FAMILY,
    did_populations,
)
from basemap.round0245_guard import ROUND_ID, Round0245Error

HARMFUL = "HARMFUL"
HARMLESS = "HARMLESS"
INDETERMINATE = "INDETERMINATE"

DECISION_VERDICTS = (HARMFUL, HARMLESS, INDETERMINATE)

#: The rule this function executes is R0244's, imported rather than re-typed.
#: `did_decision` is the executable form of the same three clauses.
DECISION_SOURCE_RULE = DID_DECISION_RULE

DECISION_NOTE = (
    "review-0244-01 section C: the decision map was a string constant sealed "
    "into a receipt, with no did_decision() a future round must call and no "
    "test that could plant an underpowered null into it. This is that "
    "function. The third clause is a BRANCH, not a sentence: HARMLESS requires "
    "a planted-displacement positive control that the test rejects, on the "
    "same maps; without it a null returns INDETERMINATE."
)

#: 'Both arms significant' in the registered rule is operationalised as: the
#: PLACEBO contrast (control half 2 scored against control half 1) is itself
#: significant. If two arbitrary halves of the same control separate, the gap
#: statistic is biased on these maps and no treated contrast computed from it
#: is interpretable. Stated here because the registered sentence is short and
#: this is the reading the code implements.
BOTH_ARMS_NOTE = (
    "the registered rule ends 'including both arms null without the power "
    "demonstration, and including both arms significant'. The second phrase is "
    "implemented as: if the placebo contrast inside the control pool is itself "
    "significant at alpha, the statistic is biased on these maps and the "
    "verdict is INDETERMINATE regardless of the treated contrast."
)


def did_decision(
    *,
    stratification: str,
    difference_in_differences: float,
    holm_adjusted_p: float,
    placebo_sd: float,
    placebo_contrast_p: float | None = None,
    power_control: Mapping[str, Any] | None = None,
    alpha: float = DID_ALPHA,
    equivalence_bound_in_placebo_sd: float = DID_SD_EQUIVALENCE_BOUND,
) -> dict[str, Any]:
    """The registered decision map, as code that runs and returns a verdict.

    `power_control` is the planted-displacement positive control the rule
    requires for a HARMLESS reading: a mapping carrying `planted_displacement`,
    `rejected` and `same_maps`. Absent, incomplete, or not rejected, the null is
    **INDETERMINATE, never harmless**.
    """
    did = float(difference_in_differences)
    p_value = float(holm_adjusted_p)
    sd = float(placebo_sd)
    bound = float(equivalence_bound_in_placebo_sd)
    if not math.isfinite(did) or not math.isfinite(p_value):
        raise Round0245Error(
            f"R0245 did_decision needs finite inputs, got DiD {did} p {p_value}"
        )
    if not (0.0 <= p_value <= 1.0):
        raise Round0245Error(f"R0245 did_decision needs a p in [0, 1], got {p_value}")
    if sd < 0.0 or not math.isfinite(sd):
        raise Round0245Error(
            f"R0245 did_decision needs a finite non-negative placebo sd, got {sd}"
        )

    control = dict(power_control or {})
    power_demonstrated = bool(
        control.get("planted_displacement") is not None
        and bool(control.get("rejected"))
        and bool(control.get("same_maps"))
    )
    placebo_significant = bool(
        placebo_contrast_p is not None and float(placebo_contrast_p) <= alpha
    )
    significant = bool(p_value <= float(alpha))
    positive = bool(did > 0.0)
    within_equivalence = bool(sd > 0.0 and abs(did) <= bound * sd)

    if placebo_significant:
        verdict = INDETERMINATE
        because = (
            "the placebo contrast inside the control pool is itself "
            f"significant (p {placebo_contrast_p} <= alpha {alpha}), so the gap "
            "statistic is biased on these maps and neither a rejection nor a "
            "null of the treated contrast is interpretable"
        )
    elif significant and positive:
        verdict = HARMFUL
        because = (
            f"DiD {did} > 0 with a Holm-adjusted one-sided p {p_value} <= "
            f"{alpha}: the residual displaces rows and training on this graph "
            "stops"
        )
    elif significant and not positive:
        verdict = INDETERMINATE
        because = (
            f"the one-sided test is significant (p {p_value}) with a "
            f"non-positive DiD {did}, which the registered rule does not map "
            "to either reading"
        )
    elif within_equivalence and power_demonstrated:
        verdict = HARMLESS
        because = (
            f"p {p_value} > {alpha}, |DiD| {abs(did)} <= {bound} placebo sd "
            f"({sd}), and a planted displacement of "
            f"{control.get('planted_displacement')} was rejected on the same "
            "maps, so the instrument is shown to have the power to have seen "
            "the effect at issue"
        )
    elif within_equivalence and not power_demonstrated:
        verdict = INDETERMINATE
        because = (
            f"p {p_value} > {alpha} and |DiD| {abs(did)} is inside the "
            f"equivalence bound, but the planted-displacement power control is "
            "absent or was not rejected on the same maps. A null from a design "
            "whose power was not demonstrated is INDETERMINATE, never harmless"
        )
    else:
        verdict = INDETERMINATE
        because = (
            f"p {p_value} > {alpha} but |DiD| {abs(did)} exceeds {bound} "
            f"placebo sd ({sd}): the test did not reject and the effect is not "
            "bounded either"
        )

    if verdict not in DECISION_VERDICTS:  # pragma: no cover - structural
        raise Round0245Error(f"R0245 produced a verdict outside the map: {verdict}")
    return {
        "instrument": "round0245-did-decision-gate-v1",
        "round_id": ROUND_ID,
        "stratification": str(stratification),
        "verdict": verdict,
        "because": because,
        "difference_in_differences": did,
        "holm_adjusted_p": p_value,
        "alpha": float(alpha),
        "placebo_sd": sd,
        "equivalence_bound_in_placebo_sd": bound,
        "difference_in_differences_in_placebo_sd": (
            None if sd <= 0.0 else did / sd
        ),
        "test_rejected": significant,
        "effect_within_equivalence_bound": within_equivalence,
        "power_was_demonstrated_on_the_same_maps": power_demonstrated,
        "placebo_contrast_p": (
            None if placebo_contrast_p is None else float(placebo_contrast_p)
        ),
        "placebo_contrast_is_significant": placebo_significant,
        "registered_rule": DECISION_SOURCE_RULE,
        "note": DECISION_NOTE,
        "both_arms_reading": BOTH_ARMS_NOTE,
    }


def did_decision_positive_controls(*, alpha: float = DID_ALPHA) -> dict[str, Any]:
    """Plant each branch and require the gate to route it correctly.

    The one that matters is the third: a null with no power demonstration must
    come back `INDETERMINATE`. A decision map nobody has seen refuse is the
    same kind of evidence as a guard nobody has seen fire.
    """
    powered = {
        "planted_displacement": 0.05,
        "rejected": True,
        "same_maps": True,
    }
    unpowered_absent: Mapping[str, Any] | None = None
    unpowered_not_rejected = {
        "planted_displacement": 0.05,
        "rejected": False,
        "same_maps": True,
    }
    other_maps = {
        "planted_displacement": 0.05,
        "rejected": True,
        "same_maps": False,
    }
    cases = [
        ("harmful_effect", HARMFUL, dict(
            difference_in_differences=0.04, holm_adjusted_p=0.001,
            placebo_sd=0.01, power_control=powered,
        )),
        ("powered_null", HARMLESS, dict(
            difference_in_differences=0.002, holm_adjusted_p=0.42,
            placebo_sd=0.01, power_control=powered,
        )),
        ("unpowered_null_no_control", INDETERMINATE, dict(
            difference_in_differences=0.002, holm_adjusted_p=0.42,
            placebo_sd=0.01, power_control=unpowered_absent,
        )),
        ("unpowered_null_control_did_not_reject", INDETERMINATE, dict(
            difference_in_differences=0.002, holm_adjusted_p=0.42,
            placebo_sd=0.01, power_control=unpowered_not_rejected,
        )),
        ("power_shown_on_other_maps", INDETERMINATE, dict(
            difference_in_differences=0.002, holm_adjusted_p=0.42,
            placebo_sd=0.01, power_control=other_maps,
        )),
        ("null_outside_the_equivalence_bound", INDETERMINATE, dict(
            difference_in_differences=0.05, holm_adjusted_p=0.42,
            placebo_sd=0.01, power_control=powered,
        )),
        ("placebo_contrast_significant", INDETERMINATE, dict(
            difference_in_differences=0.04, holm_adjusted_p=0.001,
            placebo_sd=0.01, power_control=powered, placebo_contrast_p=0.002,
        )),
        ("significant_but_negative", INDETERMINATE, dict(
            difference_in_differences=-0.04, holm_adjusted_p=0.001,
            placebo_sd=0.01, power_control=powered,
        )),
    ]
    outcomes = []
    failures = []
    for name, expected, kwargs in cases:
        got = did_decision(stratification=name, alpha=alpha, **kwargs)
        outcomes.append({
            "case": name,
            "expected_verdict": expected,
            "verdict": got["verdict"],
            "because": got["because"],
            "routed_as_registered": bool(got["verdict"] == expected),
        })
        if got["verdict"] != expected:
            failures.append(f"{name}: {got['verdict']} != {expected}")
    evidence = {
        "control": "round0245-did-decision-positive-control-v1",
        "cases": outcomes,
        "cases_planted": len(cases),
        "unpowered_nulls_returned_indeterminate": sum(
            1 for row in outcomes
            if row["case"].startswith(("unpowered_null", "power_shown"))
            and row["verdict"] == INDETERMINATE
        ),
        "failures": failures,
        "holds": not failures,
        "note": DECISION_NOTE,
    }
    if failures:
        raise Round0245Error(
            f"R0245 DiD DECISION GATE DID NOT ROUTE AS REGISTERED: {failures}"
        )
    return evidence


# --------------------------------------------------------------------------- #
# defect 5 — the arm assignment, and why tie-aware exceeded strict
# --------------------------------------------------------------------------- #
TIE_EFFECTIVE_RULE = (
    "the registered DiD populations are built on tie_effective = "
    "min(tie_aware_builder_missing, strict_builder_missing). Tie forgiveness "
    "can only forgive, so tie_effective is the tie-aware count with the two "
    "rows where the two estimators disagree numerically clamped back to the "
    "strict count. This is the SAME clip-at-zero convention loss_decomposition "
    "already uses for the partition/builder split - 'it never invents builder "
    "loss' - applied to the tie/strict split. Consequences, all conservative: "
    "genuine = tie_effective > 0 is a SUBSET of strict > 0, so genuine and "
    "control = strict == 0 are disjoint BY CONSTRUCTION rather than by "
    "inspection; genuine and tie_forgiven partition strict > 0 exactly; and "
    "the clamp can only move a row OUT of the treated arm, never into it."
)

TIE_SCORING_NOTE = (
    "strict and tie-aware are not nested estimators. strict_containment_rows "
    "is a set test over truth IDS and is immune to float noise. tie_aware_rows "
    "is a VALUE test: each candidate's CPU float32 recomputed cosine against "
    "R0238's independently computed true k-th cosine, accepted within "
    f"TIE_TOLERANCE = {TIE_TOLERANCE}. Two independent float32 dot products "
    "over 384 dimensions of the same pair disagree by ~1e-6 on this substrate, "
    "which is the same order as the tolerance, so a candidate that IS in the "
    "exact truth set can fall below kth - tolerance and be scored invalid "
    "while strict counts it. That is how tie-aware loss can exceed strict "
    "loss on a row whose true k-th neighbours are exactly tied."
)


def tie_effective_builder_missing(
    *, strict_builder_missing: np.ndarray, tie_aware_builder_missing: np.ndarray
) -> np.ndarray:
    """`min(tie, strict)`, per row. The clamp, in one place."""
    strict = np.asarray(strict_builder_missing, dtype=np.int64)
    tie = np.asarray(tie_aware_builder_missing, dtype=np.int64)
    if strict.shape != tie.shape or strict.ndim != 1:
        raise Round0245Error(
            f"R0245 tie clamp needs matched 1-D vectors, got {strict.shape} "
            f"and {tie.shape}"
        )
    return np.minimum(tie, strict)


def tie_monotonicity_audit(
    *, strict_builder_missing: np.ndarray, tie_aware_builder_missing: np.ndarray
) -> dict[str, Any]:
    """Which rows violate `tie <= strict`, and what the clamp costs."""
    strict = np.asarray(strict_builder_missing, dtype=np.int64)
    tie = np.asarray(tie_aware_builder_missing, dtype=np.int64)
    effective = tie_effective_builder_missing(
        strict_builder_missing=strict, tie_aware_builder_missing=tie
    )
    violating = np.flatnonzero(tie > strict)
    both_arms = np.flatnonzero((tie > 0) & (strict == 0))
    return {
        "probe_rows": int(strict.size),
        "rows_with_tie_above_strict": int(violating.size),
        "violating_rows": [int(row) for row in violating],
        "violating_strict": [int(strict[row]) for row in violating],
        "violating_tie_aware": [int(tie[row]) for row in violating],
        "rows_in_both_the_treated_and_control_arms_before_the_clamp": (
            [int(row) for row in both_arms]
        ),
        "genuine_rows_before_the_clamp": int(np.count_nonzero(tie > 0)),
        "genuine_rows_after_the_clamp": int(np.count_nonzero(effective > 0)),
        "tie_forgiven_rows_before_the_clamp": int(
            np.count_nonzero((strict > 0) & (tie == 0))
        ),
        "tie_forgiven_rows_after_the_clamp": int(
            np.count_nonzero((strict > 0) & (effective == 0))
        ),
        "control_rows": int(np.count_nonzero(strict == 0)),
        "rows_carrying_strict_loss": int(np.count_nonzero(strict > 0)),
        "treated_plus_forgiven_equals_strict_loss_rows": bool(
            int(np.count_nonzero(effective > 0))
            + int(np.count_nonzero((strict > 0) & (effective == 0)))
            == int(np.count_nonzero(strict > 0))
        ),
        "aggregate_tie_aware_edges_before_the_clamp": int(tie.sum()),
        "aggregate_tie_aware_edges_after_the_clamp": int(effective.sum()),
        "aggregate_strict_edges": int(strict.sum()),
        "rule": TIE_EFFECTIVE_RULE,
        "why_it_happens": TIE_SCORING_NOTE,
    }


def require_disjoint_arms(populations: Mapping[str, Any]) -> dict[str, Any]:
    """No row may sit in a treated population and in the control pool.

    R0244's `did_populations` returns the sampled row sets; this checks the
    thing review-0244-01 §F1 found by hand, and fails the node rather than
    footnoting it.
    """
    partition = {
        "probe_rows": int(populations["probe_rows"]),
        "genuine_rows_total": int(populations["genuine_rows_total"]),
        "tie_forgiven_rows_total": int(populations["tie_forgiven_rows_total"]),
        "intact_rows_total": int(populations["intact_rows_total"]),
    }
    partition["three_populations_sum"] = (
        partition["genuine_rows_total"]
        + partition["tie_forgiven_rows_total"]
        + partition["intact_rows_total"]
    )
    partition["populations_partition_the_probe"] = bool(
        partition["three_populations_sum"] == partition["probe_rows"]
    )
    genuine = np.asarray(populations["genuine"]["lost_sample"], dtype=np.int64)
    forgiven = np.asarray(
        populations["tie_forgiven"]["lost_sample"], dtype=np.int64
    )
    placebo_a = np.asarray(populations["placebo_a"], dtype=np.int64)
    placebo_b = np.asarray(populations["placebo_b"], dtype=np.int64)
    control_pool = np.union1d(placebo_a, placebo_b)
    overlaps = {
        "genuine_in_the_control_pool": int(
            np.intersect1d(genuine, control_pool).size
        ),
        "tie_forgiven_in_the_control_pool": int(
            np.intersect1d(forgiven, control_pool).size
        ),
        "genuine_in_tie_forgiven": int(np.intersect1d(genuine, forgiven).size),
        "placebo_halves_overlap": int(
            np.intersect1d(placebo_a, placebo_b).size
        ),
    }
    total = sum(overlaps.values())
    if not partition["populations_partition_the_probe"]:
        raise Round0245Error(
            "R0245 STOP: the registered DiD populations do not partition the "
            f"probe ({partition}). genuine + tie_forgiven + control must equal "
            "the probe row count; a surplus means a row is defined into two "
            "populations at once."
        )
    verdict = {
        "population_partition": partition,
        "overlaps": overlaps,
        "rows_on_both_sides": total,
        "arms_are_disjoint": bool(total == 0),
        "rule": TIE_EFFECTIVE_RULE,
    }
    if total:
        raise Round0245Error(
            f"R0245 STOP: the registered DiD arms overlap on {total} rows "
            f"({overlaps}). A row in both arms corrupts the pairing."
        )
    return verdict


def did_populations_v2(
    *,
    strict_builder_missing: np.ndarray,
    tie_aware_builder_missing: np.ndarray,
    kth_cosine: np.ndarray,
    **kwargs: Any,
) -> dict[str, Any]:
    """R0244's registered population builder, on the clamped tie vector.

    The builder itself is imported unchanged — this round re-specifies the
    *input*, which is where the defect is, and then proves the output is
    disjoint instead of assuming it.
    """
    strict = np.asarray(strict_builder_missing, dtype=np.int64)
    tie = np.asarray(tie_aware_builder_missing, dtype=np.int64)
    audit = tie_monotonicity_audit(
        strict_builder_missing=strict, tie_aware_builder_missing=tie
    )
    effective = tie_effective_builder_missing(
        strict_builder_missing=strict, tie_aware_builder_missing=tie
    )
    populations = did_populations(
        strict_builder_missing=strict,
        tie_aware_builder_missing=effective,
        kth_cosine=kth_cosine,
        **kwargs,
    )
    populations["tie_monotonicity_audit"] = audit
    populations["arm_disjointness"] = require_disjoint_arms(populations)
    populations["arm_assignment_rule"] = TIE_EFFECTIVE_RULE
    return populations


def tie_violation_forensics(
    *,
    probe_rows_to_explain: Sequence[int],
    substrate: np.ndarray,
    graph_ids: np.ndarray,
    probe_query_rows: np.ndarray,
    truth_ids: np.ndarray,
    truth_cosines: np.ndarray,
    tolerance: float = TIE_TOLERANCE,
) -> dict[str, Any]:
    """Recompute the two rows' candidate cosines and show the mechanism.

    For each named probe row this reproduces R0241/R0242's CPU `float32`
    candidate-cosine recompute, finds the candidates that ARE members of the
    exact truth set yet fall below `kth - tolerance`, and reports the deficit
    against the tolerance. That is the whole of the explanation: the deficit is
    the same order as the tolerance.
    """
    rows: list[dict[str, Any]] = []
    for probe in probe_rows_to_explain:
        probe = int(probe)
        query = int(probe_query_rows[probe])
        candidate_ids = np.asarray(graph_ids[query], dtype=np.int64)
        anchor = np.asarray(substrate[query], dtype=np.float32)
        gathered = np.asarray(substrate[candidate_ids], dtype=np.float32)
        recomputed = np.einsum("d,kd->k", anchor, gathered).astype(np.float64)
        truth_row = np.asarray(truth_ids[probe], dtype=np.int64)
        truth_row_cos = np.asarray(truth_cosines[probe], dtype=np.float64)
        kth = float(truth_row_cos[-1])
        in_truth = np.isin(candidate_ids, truth_row)
        passes_tie = recomputed >= (kth - float(tolerance))
        demoted = np.flatnonzero(in_truth & ~passes_tie)
        rows.append({
            "probe_row": probe,
            "graph_row": query,
            "true_kth_cosine": kth,
            "strict_containment_count": int(in_truth.sum()),
            "tie_aware_count": int(passes_tie.sum()),
            "candidates_in_truth_scored_invalid_by_the_tolerance": int(
                demoted.size
            ),
            "demoted": [
                {
                    "candidate_id": int(candidate_ids[index]),
                    "recomputed_cosine": float(recomputed[index]),
                    "deficit_below_the_kth": float(kth - recomputed[index]),
                    "deficit_over_the_tolerance": float(
                        (kth - recomputed[index]) / float(tolerance)
                    ),
                }
                for index in demoted
            ],
            "distinct_truth_cosines_at_the_kth": int(
                np.count_nonzero(truth_row_cos == kth)
            ),
        })
    deficits = [
        entry["deficit_below_the_kth"]
        for row in rows for entry in row["demoted"]
    ]
    return {
        "instrument": "round0245-tie-scoring-forensics-v1",
        "tolerance": float(tolerance),
        "rows": rows,
        "demoted_candidates": len(deficits),
        "min_deficit": float(min(deficits)) if deficits else None,
        "max_deficit": float(max(deficits)) if deficits else None,
        "every_deficit_is_within_one_order_of_the_tolerance": bool(
            deficits
            and max(deficits) < 10.0 * float(tolerance)
            and min(deficits) > 0.1 * float(tolerance)
        ),
        "why": TIE_SCORING_NOTE,
    }


def cosine_agreement_profile(
    *,
    substrate: np.ndarray,
    graph_ids: np.ndarray,
    probe_query_rows: np.ndarray,
    truth_ids: np.ndarray,
    truth_cosines: np.ndarray,
    sample_rows: int = 20_000,
    seed: int = 245_001,
    tolerance: float = TIE_TOLERANCE,
    abort_check: Any = None,
    block: int = 2_000,
) -> dict[str, Any]:
    """The numerical agreement floor of the two cosine computations, measured.

    For candidates that are members of the exact truth set, the recomputed
    `float32` cosine and R0238's sealed truth cosine are two computations of
    the same quantity. The spread between them is the noise floor that
    `TIE_TOLERANCE` has to clear, and it has never been measured in this
    program.
    """
    rng = np.random.default_rng(int(seed))
    total = int(np.asarray(probe_query_rows).size)
    take = min(int(sample_rows), total)
    picked = np.sort(rng.choice(total, size=take, replace=False))
    deltas: list[np.ndarray] = []
    beyond = 0
    matched = 0
    for start in range(0, take, int(block)):
        if abort_check is not None:
            abort_check(f"R0245 cosine agreement rows [{start}, {take})")
        chunk = picked[start:start + int(block)]
        queries = np.asarray(probe_query_rows[chunk], dtype=np.int64)
        candidate_ids = np.asarray(graph_ids[queries], dtype=np.int64)
        anchors = np.asarray(substrate[queries], dtype=np.float32)
        flat = candidate_ids.reshape(-1)
        order = np.argsort(flat, kind="stable")
        gathered = np.empty((flat.size, substrate.shape[1]), dtype=np.float32)
        gathered[order] = substrate[flat[order]]
        recomputed = np.einsum(
            "bd,bkd->bk", anchors,
            gathered.reshape(candidate_ids.shape[0], candidate_ids.shape[1], -1),
        ).astype(np.float64)
        truth_row = np.asarray(truth_ids[chunk], dtype=np.int64)
        truth_row_cos = np.asarray(truth_cosines[chunk], dtype=np.float64)
        #: Join candidate -> truth by id, per row, and difference the two
        #: computations of that pair's cosine.
        for index in range(candidate_ids.shape[0]):
            position = np.searchsorted(
                np.sort(truth_row[index]), candidate_ids[index]
            )
            sorted_truth = np.sort(truth_row[index])
            sort_order = np.argsort(truth_row[index], kind="stable")
            position = np.clip(position, 0, sorted_truth.size - 1)
            hit = sorted_truth[position] == candidate_ids[index]
            if not hit.any():
                continue
            truth_slot = sort_order[position[hit]]
            delta = recomputed[index][hit] - truth_row_cos[index][truth_slot]
            deltas.append(np.abs(delta))
            matched += int(hit.sum())
            beyond += int(np.count_nonzero(np.abs(delta) > float(tolerance)))
        del candidate_ids, anchors, gathered, recomputed
    stacked = np.concatenate(deltas) if deltas else np.zeros(0)
    if stacked.size == 0:
        raise Round0245Error(
            "R0245 cosine agreement profile matched no candidate to truth"
        )
    return {
        "instrument": "round0245-cosine-agreement-profile-v1",
        "probe_rows_sampled": take,
        "seed": int(seed),
        "matched_candidate_truth_pairs": matched,
        "tolerance": float(tolerance),
        "abs_delta_mean": float(stacked.mean()),
        "abs_delta_p50": float(np.quantile(stacked, 0.50)),
        "abs_delta_p99": float(np.quantile(stacked, 0.99)),
        "abs_delta_max": float(stacked.max()),
        "pairs_beyond_the_tolerance": beyond,
        "fraction_beyond_the_tolerance": float(beyond) / float(matched),
        "tolerance_over_the_p99_disagreement": (
            float(tolerance) / float(np.quantile(stacked, 0.99))
            if np.quantile(stacked, 0.99) > 0 else None
        ),
        "why": TIE_SCORING_NOTE,
    }


# --------------------------------------------------------------------------- #
# finding 6 — the unrestricted permutation, priced rather than accepted
# --------------------------------------------------------------------------- #
PERMUTATION_TYPEI_TRIALS = 20_000
PERMUTATION_TYPEI_SEED = 245_006

PERMUTATION_COST_NOTE = (
    "review-0244-01 section C: the placebo null lives inside the SAME maps, so "
    "the ten pooled values are five PAIRS, and C(10,5) = 252 treats two "
    "statistics from one map as exchangeable. The strictly-correct paired "
    "exact test is the sign-flip test with 2^n patterns and a floor of 1/2^n, "
    "which at n = 5 is 0.03125 and cannot clear the Holm threshold 0.005 at "
    "any n below 8. So the registered design's ABILITY TO REJECT depends on "
    "the unrestricted choice. What that costs is measured here rather than "
    "argued: the realised one-sided type-I rate of the unrestricted test on "
    "paired data with a shared per-map effect, against its nominal alpha."
)


def permutation_design_cost(
    *,
    maps_per_arm: int = DID_MAPS_PER_ARM,
    alpha: float = DID_ALPHA,
    tests_in_family: int = DID_TESTS_IN_FAMILY,
    trials: int = PERMUTATION_TYPEI_TRIALS,
    seed: int = PERMUTATION_TYPEI_SEED,
    map_sd: float = 1.0,
    within_sd: float = 1.0,
) -> dict[str, Any]:
    """What the unrestricted permutation buys and what it costs.

    The simulation is the honest part. Under the null the treated value and the
    second control half of a map are exchangeable draws around that map's own
    level, so each map contributes one value to each arm and the two are
    correlated through the shared level. The unrestricted permutation ignores
    that correlation. Whether it is conservative or anti-conservative is a
    measurement, not a stance.
    """
    n = int(maps_per_arm)
    strictest = float(alpha) / float(tests_in_family)
    unrestricted_labellings = math.comb(2 * n, n)
    paired_patterns = 2 ** n
    smallest_n_unrestricted = next(
        (size for size in range(2, 33)
         if 1.0 / math.comb(2 * size, size) <= strictest),
        None,
    )
    smallest_n_paired = next(
        (size for size in range(2, 33) if 1.0 / float(2 ** size) <= strictest),
        None,
    )

    rng = np.random.default_rng(int(seed))
    rejected = 0
    p_values: list[float] = []
    for _trial in range(int(trials)):
        level = rng.normal(0.0, float(map_sd), size=n)
        c1 = level + rng.normal(0.0, float(within_sd), size=n)
        c2 = level + rng.normal(0.0, float(within_sd), size=n)
        treated = level + rng.normal(0.0, float(within_sd), size=n)
        outcome = exact_displacement_permutation(
            candidate_gaps=(treated - c1).tolist(),
            exact_gaps=(c2 - c1).tolist(),
        )
        p_value = float(outcome["p_one_sided"])
        p_values.append(p_value)
        if p_value <= strictest:
            rejected += 1
    realised = float(rejected) / float(trials)
    return {
        "instrument": "round0245-permutation-design-cost-v1",
        "maps_per_arm": n,
        "alpha_family_wise": float(alpha),
        "tests_in_family": int(tests_in_family),
        "strictest_holm_threshold": strictest,
        "unrestricted_labellings": int(unrestricted_labellings),
        "unrestricted_floor": 1.0 / float(unrestricted_labellings),
        "paired_sign_flip_patterns": int(paired_patterns),
        "paired_sign_flip_floor": 1.0 / float(paired_patterns),
        "smallest_arm_that_can_reject_unrestricted": smallest_n_unrestricted,
        "smallest_arm_that_can_reject_paired": smallest_n_paired,
        "paired_test_can_reject_at_this_n": bool(
            1.0 / float(paired_patterns) <= strictest
        ),
        "simulation": {
            "trials": int(trials),
            "seed": int(seed),
            "map_level_sd": float(map_sd),
            "within_map_sd": float(within_sd),
            "model": (
                "per map: level ~ N(0, map_sd); control half 1, control half 2 "
                "and the treated value are independent draws around that "
                "level; gap_treated = treated - c1, gap_placebo = c2 - c1. "
                "This is the null, with the shared per-map effect the "
                "unrestricted permutation ignores."
            ),
            "nominal_one_sided_rate": strictest,
            "realised_one_sided_rate": realised,
            "realised_over_nominal": (
                realised / strictest if strictest > 0 else None
            ),
            "p_value_mean": float(np.mean(p_values)),
            "p_value_min": float(np.min(p_values)),
        },
        "unrestricted_test_is_conservative": bool(realised <= strictest),
        "restricted_permutation_is_required": bool(realised > strictest),
        "what_it_costs": PERMUTATION_COST_NOTE,
        "consequence": (
            "if the realised rate is at or below nominal, the unrestricted "
            "permutation is conservative and the 252 is a modelling choice "
            "that costs power, not validity; the design keeps its ability to "
            "reject. If it is above nominal, the design is anti-conservative "
            "and a restricted (sign-flip) permutation is REQUIRED, which at "
            f"n = {n} has a floor of {1.0 / float(paired_patterns)} and cannot "
            f"clear {strictest} - i.e. the DiD would need at least "
            f"{smallest_n_paired} maps per arm."
        ),
    }


def did_family_decision(
    *, per_stratification: Mapping[str, Mapping[str, Any]],
    alpha: float = DID_ALPHA,
) -> dict[str, Any]:
    """Holm over the family, then the registered map per stratification.

    `per_stratification` carries, per name, the raw one-sided p, the DiD, the
    placebo sd, the placebo contrast p and the power control. Holm is R0229's,
    imported.
    """
    raw = {
        name: float(entry["p_one_sided"])
        for name, entry in per_stratification.items()
    }
    holm = holm_bonferroni(raw, alpha=float(alpha))
    decisions = {}
    for name, entry in per_stratification.items():
        adjusted = holm["decisions"][name]
        decisions[name] = did_decision(
            stratification=name,
            difference_in_differences=float(
                entry["difference_in_differences"]
            ),
            holm_adjusted_p=float(adjusted["p_one_sided"]),
            placebo_sd=float(entry["placebo_sd"]),
            placebo_contrast_p=entry.get("placebo_contrast_p"),
            power_control=entry.get("power_control"),
            alpha=float(adjusted["holm_threshold"]),
        )
    verdicts = {name: value["verdict"] for name, value in decisions.items()}
    if HARMFUL in verdicts.values():
        overall = HARMFUL
    elif all(value == HARMLESS for value in verdicts.values()):
        overall = HARMLESS
    else:
        overall = INDETERMINATE
    return {
        "instrument": "round0245-did-family-decision-v1",
        "holm": holm,
        "decisions": decisions,
        "verdicts": verdicts,
        "overall_verdict": overall,
        "note": DECISION_NOTE,
    }


__all__ = [
    "BOTH_ARMS_NOTE",
    "DECISION_NOTE",
    "DECISION_SOURCE_RULE",
    "DECISION_VERDICTS",
    "HARMFUL",
    "HARMLESS",
    "INDETERMINATE",
    "PERMUTATION_COST_NOTE",
    "PERMUTATION_TYPEI_SEED",
    "PERMUTATION_TYPEI_TRIALS",
    "TIE_EFFECTIVE_RULE",
    "TIE_SCORING_NOTE",
    "cosine_agreement_profile",
    "did_decision",
    "did_decision_positive_controls",
    "did_family_decision",
    "did_populations_v2",
    "permutation_design_cost",
    "require_disjoint_arms",
    "tie_effective_builder_missing",
    "tie_monotonicity_audit",
    "tie_violation_forensics",
]
