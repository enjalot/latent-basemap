"""R0246 — decide what the tie-aware tolerance finding invalidates.

R0245 measured the numerical agreement floor of the two cosine computations the
tie-aware estimator compares, and review-0245-01 replicated it at a fresh seed:

| statistic | R0245 seed `245001` | reviewer seed `918273` |
| --- | ---: | ---: |
| p99 `\\|delta\\|` | `5.364418029785156e-07` | `5.960464477539062e-07` |
| max `\\|delta\\|` | `1.0132789611816406e-06` | `1.3113021850585938e-06` |
| `TIE_TOLERANCE / p99` | `1.864135111111111` | `1.678` |

So the tolerance sits at `1.68-1.86x` its own noise floor and the **maximum
disagreement exceeds the tolerance outright**. The reviewer's position is that
aggregate figures stand and per-row uses of tie-aware do not. This module makes
that a measurement and a rule rather than a position:

1. `tie_aware_precision_profile` recomputes each candidate's cosine in `float32`
   *and* in `float64` from the same substrate bytes at a fresh seed, and counts
   the candidates whose **tie-aware verdict flips** between the two. That flip
   rate is the per-decision error rate of the estimator, measured rather than
   inferred from a quantile.
2. `TIE_AWARE_CLAIM_LEDGER` enumerates every published claim in R0241 and R0243
   that rests on a tie-aware decision, with the number of per-candidate
   decisions behind it and the margin that would have to be crossed to change
   it. `adjudicate_tie_aware_claims` propagates the measured flip rate through
   each and returns survives / does-not-survive.
3. `require_aggregate_tie_aware_use` is the registered rule: the tie-aware
   vector may be consumed where the expected number of flipped decisions is
   small against the claim's own margin, and not otherwise.

The tolerance is **not raised**. Raising it would change R0243's published
figures for a sub-`0.1` pp effect and trade one arbitrary threshold for
another; the reviewer said so and this round agrees. The alternative this round
measures instead is the `float64` recompute, which is what removes the
threshold from the estimator rather than moving it.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from basemap.round0227_low_c_contract import TIE_TOLERANCE
from basemap.round0246_guard import Round0246Error
from basemap.round0247_registry import (
    TIE_USE_MAX_EXPECTED_FLIPS_OVER_MARGIN,
    clamp,
    clamp_int,
    override_records,
    registered_bounds,
    require_no_weakening_overrides,
    verify_registry,
)

#: A fresh seed, neither R0245's `245001` nor review-0245-01's `918273`, so the
#: replication is a third independent draw.
TIE_PRECISION_SEED = 246_001
TIE_PRECISION_ROWS = 20_000

TIE_AGGREGATE_ONLY_RULE = (
    "REGISTERED 2026-08-10 (R0246). tie_aware_rows is a float32 VALUE test at a "
    f"tolerance of {TIE_TOLERANCE} against a measured p99 disagreement of order "
    "5e-07 and a measured maximum that exceeds the tolerance. It is therefore "
    "an AGGREGATE estimator only. A published claim may consume it where the "
    "expected number of flipped per-candidate decisions behind the claim is "
    "below 1% of the margin that would have to be crossed to change the claim "
    "(TIE_USE_MAX_EXPECTED_FLIPS_OVER_MARGIN = 0.01, the criterion for "
    "ADMITTING a new consumption); a claim ALREADY PUBLISHED is adjudicated "
    "instead at TIE_CLAIM_MAX_EXPECTED_FLIPS_OVER_MARGIN = 1.0, which asks "
    "the different question of whether the published number is expected to be "
    "wrong. review-0246-01 H5 found this text and the code disagreeing by "
    "100x; the reconciliation is that the two functions were always asking "
    "different questions and only one rule text had been written. And "
    "no claim may rest on the tie-aware verdict of an individual row, or on an "
    "exact small integer count of rows selected by the tie-aware threshold "
    "test. The strict estimator is a set test over truth ids, is immune to "
    "float noise, and is the one a per-row decision must use. This rule does "
    "NOT raise TIE_TOLERANCE: raising it would move R0243's published figures "
    "by under 0.1 pp and replace one arbitrary threshold with another. The fix "
    "for a future round is to recompute both cosines in float64, which R0246 "
    "measures here."
)

#: The survival criterion, as arithmetic rather than a judgement: a claim
#: survives when `expected_flipped_decisions <= margin`, where `margin` is
#: defined per claim as the number of candidate verdicts that would have to
#: change for the claim to read differently AT THE PRECISION IT IS PUBLISHED
#: TO. The precision is already inside the margin, so the ratio's threshold is
#: `1.0` and not some second fudge factor on top of it. R0247 keeps this as the
#: RETROSPECTIVE criterion and registers the sealed rule text's 1% as the
#: PROSPECTIVE one; see `TIE_USE_MAX_EXPECTED_FLIPS_OVER_MARGIN`.
TIE_CLAIM_MAX_EXPECTED_FLIPS_OVER_MARGIN = 1.0

#: A PLANTED rate for the routing control, not a measurement. R0247 moved it
#: from `1e-06` to `1e-08` because the admission criterion moved from `1.0` to
#: the sealed `0.01`: at `1e-08` an aggregate consumption of R0243's `2,915`
#: count expects `0.075` flipped verdicts against a `29.15` margin
#: (`0.00257` of it, permitted) while a claim whose margin is one verdict
#: expects the same `0.075` (`0.075` of it, refused). The control exists to
#: show the gate routes the two differently at the SAME rate, and the rate has
#: to sit inside the window where they differ or the control tests nothing.
#: The measured rate is what the adjudication uses; this number is registered
#: as free to move for exactly that reason (round0247_registry
#: EXAMINED_NOT_SAFETY).
TIE_CONTROL_PLANTED_FLIP_RATE = 1e-08

#: The population at risk behind any claim computed from the probe's tie-aware
#: vector. A flip anywhere in the probe can move a probe-wide count, and - the
#: reason this matters - it can also move a row into or out of a small selected
#: set, which is how review-0241-01 F4 watched one row's perfection change
#: under a different float32 contraction order. A claim's decision population
#: is therefore the whole probe unless the claim is adjudicated on a quantity
#: that is not the tolerance.
PROBE_CANDIDATE_DECISIONS = 7_500_000


def tie_aware_precision_profile(
    *,
    substrate: np.ndarray,
    graph_ids: np.ndarray,
    probe_query_rows: np.ndarray,
    truth_ids: np.ndarray,
    truth_cosines: np.ndarray,
    sample_rows: int = TIE_PRECISION_ROWS,
    seed: int = TIE_PRECISION_SEED,
    tolerance: float = TIE_TOLERANCE,
    abort_check: Any = None,
    block: int = 500,
) -> dict[str, Any]:
    """Measure the estimator's per-decision error rate, two ways.

    For every candidate of every sampled probe row this computes the cosine in
    `float32` (what R0241/R0242/R0243 did) and in `float64` (the same bytes,
    the same pairs, one more mantissa), and applies the tie-aware value test
    `cos >= kth - tolerance` under both. A candidate whose verdict differs
    between the two is a decision the estimator got wrong under one of them —
    the *direct* measurement of the thing the p99 quantile only bounds.

    The `float64` column is not noise-free: R0238's truth cosines are stored as
    `float32`, so the irreducible floor is that quantisation. Both floors are
    reported.
    """
    verify_registry(label="R0246 tie precision profile")
    #: R0247 attack tie-r0247-1: raising the tolerance widens the band, so
    #: fewer verdicts differ between float32 and float64, so the MEASURED flip
    #: rate falls and more published claims survive. The measurement that
    #: adjudicates the claims was scored at a threshold the caller supplied.
    tolerance_effective, tolerance_record = clamp(
        "tie_tolerance", tolerance,
        site="tie_aware_precision_profile(tolerance=)",
        label="R0246 tie precision profile",
    )
    rows_effective, rows_record = clamp_int(
        "tie_precision_min_rows", sample_rows,
        site="tie_aware_precision_profile(sample_rows=)",
        label="R0246 tie precision profile",
        population=float(np.asarray(probe_query_rows).size),
    )
    overrides = override_records([tolerance_record, rows_record])
    require_no_weakening_overrides(
        overrides, label="R0246 tie precision profile"
    )
    sample_rows = int(rows_effective)
    rng = np.random.default_rng(int(seed))
    total = int(np.asarray(probe_query_rows).size)
    take = min(int(sample_rows), total)
    picked = np.sort(rng.choice(total, size=take, replace=False))
    tolerance = float(tolerance_effective)

    f32_deltas: list[np.ndarray] = []
    f64_deltas: list[np.ndarray] = []
    matched = 0
    candidates_scored = 0
    f32_pass = 0
    f64_pass = 0
    flips_f32_pass_f64_fail = 0
    flips_f32_fail_f64_pass = 0
    inside_the_noise_band = 0
    f32_beyond_tolerance = 0

    for start in range(0, take, int(block)):
        if abort_check is not None:
            abort_check(f"R0246 tie precision rows [{start}, {take})")
        chunk = picked[start:start + int(block)]
        queries = np.asarray(probe_query_rows[chunk], dtype=np.int64)
        candidate_ids = np.asarray(graph_ids[queries], dtype=np.int64)
        flat = candidate_ids.reshape(-1)
        order = np.argsort(flat, kind="stable")
        gathered = np.empty((flat.size, substrate.shape[1]), dtype=np.float32)
        gathered[order] = substrate[flat[order]]
        gathered = gathered.reshape(
            candidate_ids.shape[0], candidate_ids.shape[1], -1
        )
        anchors32 = np.asarray(substrate[queries], dtype=np.float32)
        recomputed32 = np.einsum("bd,bkd->bk", anchors32, gathered).astype(
            np.float64
        )
        recomputed64 = np.einsum(
            "bd,bkd->bk",
            anchors32.astype(np.float64), gathered.astype(np.float64),
        )
        truth_row = np.asarray(truth_ids[chunk], dtype=np.int64)
        truth_row_cos = np.asarray(truth_cosines[chunk], dtype=np.float64)
        kth = truth_row_cos[:, -1]

        pass32 = recomputed32 >= (kth[:, None] - tolerance)
        pass64 = recomputed64 >= (kth[:, None] - tolerance)
        candidates_scored += int(pass32.size)
        f32_pass += int(pass32.sum())
        f64_pass += int(pass64.sum())
        flips_f32_pass_f64_fail += int(np.count_nonzero(pass32 & ~pass64))
        flips_f32_fail_f64_pass += int(np.count_nonzero(~pass32 & pass64))
        #: candidates sitting within one float32 noise width of the threshold -
        #: the population any contraction-order change can move.
        band = np.abs(recomputed32 - (kth[:, None] - tolerance)) <= tolerance
        inside_the_noise_band += int(band.sum())

        for index in range(candidate_ids.shape[0]):
            sorted_truth = np.sort(truth_row[index])
            sort_order = np.argsort(truth_row[index], kind="stable")
            position = np.clip(
                np.searchsorted(sorted_truth, candidate_ids[index]),
                0, sorted_truth.size - 1,
            )
            hit = sorted_truth[position] == candidate_ids[index]
            if not hit.any():
                continue
            slot = sort_order[position[hit]]
            reference = truth_row_cos[index][slot]
            delta32 = np.abs(recomputed32[index][hit] - reference)
            delta64 = np.abs(recomputed64[index][hit] - reference)
            f32_deltas.append(delta32)
            f64_deltas.append(delta64)
            matched += int(hit.sum())
            f32_beyond_tolerance += int(
                np.count_nonzero(delta32 > tolerance)
            )
        del candidate_ids, gathered, recomputed32, recomputed64

    if not f32_deltas:
        raise Round0246Error(
            "R0246 tie precision profile matched no candidate to truth"
        )
    stacked32 = np.concatenate(f32_deltas)
    stacked64 = np.concatenate(f64_deltas)
    flips = flips_f32_pass_f64_fail + flips_f32_fail_f64_pass
    p99_32 = float(np.quantile(stacked32, 0.99))
    p99_64 = float(np.quantile(stacked64, 0.99))
    return {
        "instrument": "round0246-tie-precision-profile-v1",
        "probe_rows_sampled": take,
        "seed": int(seed),
        "tolerance": tolerance,
        "matched_candidate_truth_pairs": matched,
        "candidate_decisions_scored": candidates_scored,
        "float32": {
            "abs_delta_mean": float(stacked32.mean()),
            "abs_delta_p50": float(np.quantile(stacked32, 0.50)),
            "abs_delta_p99": p99_32,
            "abs_delta_max": float(stacked32.max()),
            "pairs_beyond_the_tolerance": f32_beyond_tolerance,
            "tolerance_over_the_p99_disagreement": (
                tolerance / p99_32 if p99_32 > 0 else None
            ),
            "tolerance_over_the_max_disagreement": (
                tolerance / float(stacked32.max())
                if stacked32.max() > 0 else None
            ),
            "candidates_passing_the_tie_test": f32_pass,
        },
        "float64": {
            "abs_delta_mean": float(stacked64.mean()),
            "abs_delta_p50": float(np.quantile(stacked64, 0.50)),
            "abs_delta_p99": p99_64,
            "abs_delta_max": float(stacked64.max()),
            "tolerance_over_the_p99_disagreement": (
                tolerance / p99_64 if p99_64 > 0 else None
            ),
            "candidates_passing_the_tie_test": f64_pass,
            "note": (
                "not noise-free: R0238's truth cosines are stored as float32, "
                "so this column measures the residual after the recompute "
                "side is exact and the reference side is still quantised."
            ),
        },
        "p99_improvement_from_float64": (
            p99_32 / p99_64 if p99_64 > 0 else None
        ),
        "verdict_flips": {
            "float32_passed_float64_failed": flips_f32_pass_f64_fail,
            "float32_failed_float64_passed": flips_f32_fail_f64_pass,
            "total": flips,
            "per_candidate_flip_rate": (
                float(flips) / float(candidates_scored)
                if candidates_scored else None
            ),
            "candidates_within_one_tolerance_of_the_threshold": (
                inside_the_noise_band
            ),
            "fraction_within_one_tolerance_of_the_threshold": (
                float(inside_the_noise_band) / float(candidates_scored)
                if candidates_scored else None
            ),
            "why_this_is_the_number": (
                "the p99 disagreement bounds how far a recomputed cosine can "
                "sit from its reference; it does not say how often that moves "
                "a VERDICT. This does: it is the fraction of tie-aware "
                "decisions that change when the same bytes are contracted in "
                "float64 instead of float32."
            ),
        },
        "safety_overrides": [dict(record) for record in overrides],
        **registered_bounds([
            "tie_tolerance", "tie_precision_min_rows",
            "tie_claim_max_expected_flips_over_margin",
            "tie_use_max_expected_flips_over_margin",
        ]),
        "rule": TIE_AGGREGATE_ONLY_RULE,
    }


# --------------------------------------------------------------------------- #
# the ledger — every published claim that rests on a tie-aware decision
# --------------------------------------------------------------------------- #
#: `decisions` is the number of per-candidate tie-aware value tests the claim
#: is computed from. `margin` is what would have to change for the claim to be
#: different, in the same units as `decisions` (candidate verdicts), and
#: `margin_note` says where it comes from. `kind` records whether the claim is
#: an aggregate over the probe or a per-row / per-cluster / small-integer use.
TIE_AWARE_CLAIM_LEDGER: tuple[dict[str, Any], ...] = (
    {
        "round": "0241",
        "claim": "tie-aware mean recall 0.9979422666666667 over the 500,000-row probe",
        "where": "result-0241-2026-08-10.md, Digest recall table; Claims 1",
        "kind": "aggregate",
        "decisions": 7_500_000,
        "margin": 3_750.0,
        "margin_note": (
            "the figure is published to 16 digits but review-0241-01 F4 capped "
            "its meaningful precision at six significant figures, i.e. "
            "+/- 5e-07 of recall = 3,750 candidate verdicts of 7,500,000"
        ),
    },
    {
        "round": "0241",
        "claim": "tie-aware recall clears the 0.90 floor by 0.0979 of slack",
        "where": "result-0241-2026-08-10.md, Digest floor verdict; Claims 1",
        "kind": "aggregate",
        "decisions": 7_500_000,
        "margin": 734_250.0,
        "margin_note": "0.0979 of recall over 7,500,000 candidate verdicts",
    },
    {
        "round": "0241",
        "claim": "tie-aware recall by density decile, ten bins of 50,000 rows",
        "where": "result-0241-2026-08-10.md, Recall by density decile",
        "kind": "aggregate",
        "decisions": 750_000,
        "margin": 375.0,
        "margin_note": (
            "the deciles are published to six decimals; the smallest reported "
            "difference between adjacent deciles that carries interpretation is "
            "5e-04 of recall over 750,000 verdicts"
        ),
    },
    {
        "round": "0241",
        "claim": "tie_aware_rows_below_one = 9,552",
        "where": "result-0241-2026-08-10.md, Recall section",
        "kind": "aggregate",
        "decisions": 7_500_000,
        "margin": 95.52,
        "margin_note": "1% of the count itself",
    },
    {
        "round": "0241",
        "claim": "tie_aware_rows_at_zero = 25, exactly",
        "where": "result-0241-2026-08-10.md, Recall section; the 25-row table",
        "kind": "small-integer selection by the tie-aware threshold",
        "decisions": PROBE_CANDIDATE_DECISIONS,
        "margin": 1.0,
        "margin_note": (
            "an exact integer: ONE flipped candidate verdict anywhere in the "
            "probe can move a row into or out of the zero-recall set, so the "
            "decision population is the whole probe and the margin is one "
            "verdict. review-0241-01 F4 already watched one row's perfection "
            "move under a different float32 contraction order"
        ),
    },
    {
        "round": "0241",
        "claim": (
            "explanation_verified: false - 24 of 25 zero-recall rows are "
            "duplicate-family artefacts and one is not"
        ),
        "where": "result-0241-2026-08-10.md, Digest; the 25-row table; Claims 3",
        "kind": "per-row verdict on a set selected by the tie-aware threshold",
        "decisions": PROBE_CANDIDATE_DECISIONS,
        "margin": 1.0,
        "margin_note": (
            "the verdict flips on ONE row of 25, and membership of the 25 is "
            "itself a tie-aware threshold decision"
        ),
    },
    {
        "round": "0241",
        "claim": (
            "row 15,753,171 is a genuine neighbour miss, not a tie artefact "
            "(shortfall 0.00911480188369751)"
        ),
        "where": "result-0241-2026-08-10.md, Recall section; Claims 3",
        "kind": "per-row verdict, decided on a shortfall not on the tolerance",
        "decisions": 15,
        "margin": 9_114.8,
        "margin_note": (
            "the shortfall is 0.0091148, i.e. 9,114.8 tolerance widths. The "
            "row's SELECTION came from the tie-aware test; its adjudication "
            "did not, and is four orders of magnitude clear of the noise"
        ),
    },
    {
        "round": "0243",
        "claim": (
            "builder-missing edges fall 18,129 -> 2,915, so 83.92% of builder "
            "loss is tie-forgiven"
        ),
        "where": "result-0243-2026-08-10.md, Digest; headline table; Claims",
        "kind": "aggregate",
        "decisions": 7_500_000,
        "margin": 29.15,
        "margin_note": (
            "1% of the 2,915 count; equivalently the forgiveness figure is "
            "published to two decimals of a percent, which is +/- 1.8 edges"
        ),
    },
    {
        "round": "0243",
        "claim": (
            "probe-wide tie-aware builder-missing rate 0.00038932608122797856; "
            "H1 does not fire against its 0.01 bar"
        ),
        "where": "result-0243-2026-08-10.md, the registered halt table",
        "kind": "aggregate",
        "decisions": 7_500_000,
        "margin": 72_000.0,
        "margin_note": (
            "the bar is 0.01 against a measured 0.000389 over 7,487,297 "
            "reachable exposure, i.e. about 72,000 edges of margin"
        ),
    },
    {
        "round": "0243",
        "claim": (
            "H2 fires on zero cells tie-aware and three cells strict, which is "
            "the discriminator that let Part B run"
        ),
        "where": "result-0243-2026-08-10.md, the arms table; Part B gating",
        "kind": "per-cell decision with a measured margin",
        "decisions": 7_500_000,
        "margin": 74.0,
        "margin_note": (
            "the nearest cell (9) runs at 0.01401018922852984 against a "
            "0.0185 bar over an exposure of about 16,500 candidate verdicts, "
            "i.e. about 74 edge verdicts of margin"
        ),
    },
    {
        "round": "0243",
        "claim": (
            "per-cluster tie-aware builder edges 149 / 231 / 10 for clusters "
            "168 / 9 / 285, and forgiveness 97.88% / 86.70% / 99.43%"
        ),
        "where": "result-0243-2026-08-10.md, the three named cells table",
        "kind": "per-cluster small counts",
        "decisions": 60_015,
        "margin": 0.1,
        "margin_note": (
            "the per-cluster forgiveness figures are published to four "
            "significant figures; on cluster 285's ten edges that is 0.1 of an "
            "edge verdict"
        ),
    },
    {
        "round": "0243",
        "claim": (
            "partition-limited loss is almost unchanged, 12,595 -> 12,518, so "
            "essentially all tie-forgiveness lives in the builder population"
        ),
        "where": "result-0243-2026-08-10.md, the two-population table",
        "kind": "aggregate with a small discriminating delta",
        "decisions": 7_500_000,
        "margin": 7.7,
        "margin_note": (
            "the discriminating quantity is the 77-edge delta; 1% of it is 0.77 "
            "edges, and the claim would survive any change under about 7.7"
        ),
    },
    {
        "round": "0243",
        "claim": (
            "the residual is 2,915 edges in 7,500,000 and its worst cell holds "
            "231 of them - 'not a hole'"
        ),
        "where": "result-0243-2026-08-10.md, the magnitudes table",
        "kind": "aggregate plus a per-cell count",
        "decisions": 7_500_000,
        "margin": 29.15,
        "margin_note": "1% of the 2,915 count",
    },
    {
        "round": "0245",
        "claim": (
            "the DiD arm assignment: probe rows 21785 and 453495 carry more "
            "tie-aware loss than strict loss"
        ),
        "where": "result-0245-2026-08-10.md, the arm assignment",
        "kind": "per-row decision",
        "decisions": PROBE_CANDIDATE_DECISIONS,
        "margin": 1.0,
        "margin_note": (
            "one candidate verdict per row decides it. Already repaired: R0245 "
            "clamps with tie_effective = min(tie, strict), so the arms no "
            "longer depend on the per-row tie-aware decision at all"
        ),
        "already_repaired": True,
    },
)


def adjudicate_tie_aware_claims(
    profile: Mapping[str, Any],
    *,
    ledger: Sequence[Mapping[str, Any]] = TIE_AWARE_CLAIM_LEDGER,
    margin_fraction: float = TIE_CLAIM_MAX_EXPECTED_FLIPS_OVER_MARGIN,
) -> dict[str, Any]:
    """Propagate the measured flip rate through every ledger entry."""
    verify_registry(label="R0246 tie claim adjudication")
    fraction_effective, fraction_record = clamp(
        "tie_claim_max_expected_flips_over_margin", margin_fraction,
        site="adjudicate_tie_aware_claims(margin_fraction=)",
        label="R0246 tie claim adjudication",
    )
    overrides = override_records([fraction_record])
    require_no_weakening_overrides(
        overrides, label="R0246 tie claim adjudication"
    )
    margin_fraction = float(fraction_effective)
    rate = profile["verdict_flips"]["per_candidate_flip_rate"]
    if rate is None:
        raise Round0246Error("R0246 cannot adjudicate without a flip rate")
    rate = float(rate)
    rows: list[dict[str, Any]] = []
    for entry in ledger:
        decisions = float(entry["decisions"])
        margin = float(entry["margin"])
        expected = decisions * rate
        ratio = expected / margin if margin > 0 else float("inf")
        survives = bool(ratio <= float(margin_fraction))
        rows.append({
            **{key: value for key, value in entry.items()},
            "measured_per_candidate_flip_rate": rate,
            "expected_flipped_decisions": expected,
            "expected_flips_over_margin": ratio,
            "survives": survives,
        })
    failing = [row for row in rows if not row["survives"]]
    return {
        "instrument": "round0246-tie-claim-adjudication-v1",
        "measured_per_candidate_flip_rate": rate,
        "margin_fraction": float(margin_fraction),
        "safety_overrides": [dict(record) for record in overrides],
        **registered_bounds(["tie_claim_max_expected_flips_over_margin"]),
        "claims": rows,
        "claims_examined": len(rows),
        "claims_that_survive": sum(1 for row in rows if row["survives"]),
        "claims_that_do_not_survive": [
            {"round": row["round"], "claim": row["claim"], "kind": row["kind"]}
            for row in failing
        ],
        "every_aggregate_claim_survives": all(
            row["survives"] for row in rows if row["kind"] == "aggregate"
        ),
        "rule": TIE_AGGREGATE_ONLY_RULE,
    }


def require_aggregate_tie_aware_use(
    *,
    decisions: int,
    margin: float,
    flip_rate: float,
    label: str,
    margin_fraction: float = TIE_USE_MAX_EXPECTED_FLIPS_OVER_MARGIN,
) -> dict[str, Any]:
    """The registered gate: refuse a tie-aware consumption that is too fine.

    R0247 reconciles this with `TIE_AGGREGATE_ONLY_RULE`: the sealed text says
    "below 1% of the margin" and R0246 applied `1.0`. The 1% is now what this
    function applies, and it is clamped at the registry so a caller cannot
    restore the looser number.
    """
    verify_registry(label=f"R0246 tie use gate {label}")
    fraction_effective, fraction_record = clamp(
        "tie_use_max_expected_flips_over_margin", margin_fraction,
        site="require_aggregate_tie_aware_use(margin_fraction=)", label=label,
    )
    overrides = override_records([fraction_record])
    require_no_weakening_overrides(overrides, label=f"tie-aware use {label}")
    margin_fraction = float(fraction_effective)
    expected = float(decisions) * float(flip_rate)
    ratio = expected / float(margin) if float(margin) > 0 else float("inf")
    verdict = {
        "label": label,
        "decisions": int(decisions),
        "margin": float(margin),
        "flip_rate": float(flip_rate),
        "expected_flipped_decisions": expected,
        "expected_flips_over_margin": ratio,
        "margin_fraction": float(margin_fraction),
        "use_is_permitted": bool(ratio <= float(margin_fraction)),
        "safety_overrides": [dict(record) for record in overrides],
        **registered_bounds(["tie_use_max_expected_flips_over_margin"]),
        "rule": TIE_AGGREGATE_ONLY_RULE,
    }
    if not verdict["use_is_permitted"]:
        raise Round0246Error(
            f"R0246 REFUSES the tie-aware consumption {label!r}: "
            f"{decisions} per-candidate decisions at a measured flip rate of "
            f"{flip_rate} give {expected} expected flipped verdicts against a "
            f"margin of {margin}, i.e. {ratio} of it, over the registered "
            f"{margin_fraction}. " + TIE_AGGREGATE_ONLY_RULE
        )
    return verdict


def tie_use_positive_control(
    *, flip_rate: float = TIE_CONTROL_PLANTED_FLIP_RATE
) -> dict[str, Any]:
    """Plant one permitted use and one forbidden one; both must route.

    The rate is a PLANT, not the measurement, so the control fires whatever the
    node goes on to measure. A control that only fires at the rate the round
    happened to observe is not a control.
    """
    permitted = require_aggregate_tie_aware_use(
        decisions=PROBE_CANDIDATE_DECISIONS, margin=29.15,
        flip_rate=float(flip_rate),
        label="R0243's 2,915 aggregate builder-missing count",
    )
    refused = False
    message = None
    try:
        require_aggregate_tie_aware_use(
            decisions=PROBE_CANDIDATE_DECISIONS, margin=1.0,
            flip_rate=float(flip_rate),
            label="R0241's exact count of 25 tie-aware-zero rows",
        )
    except Round0246Error as error:
        refused = True
        message = str(error)
    evidence = {
        "control": "round0246-tie-aggregate-only-positive-control-v1",
        "planted_flip_rate": float(flip_rate),
        "planted_not_measured": True,
        "aggregate_use_permitted": bool(permitted["use_is_permitted"]),
        "aggregate_use": permitted,
        "small_count_use_refused": refused,
        "small_count_message": message,
    }
    failures = [
        name for name, ok in (
            ("aggregate_use_permitted", evidence["aggregate_use_permitted"]),
            ("small_count_use_refused", refused),
        ) if not ok
    ]
    evidence["failures"] = failures
    evidence["holds"] = not failures
    if failures:
        raise Round0246Error(
            f"R0246 TIE-USE GATE DID NOT ROUTE AS REGISTERED: {failures}"
        )
    return evidence


__all__ = [
    "TIE_AGGREGATE_ONLY_RULE",
    "TIE_USE_MAX_EXPECTED_FLIPS_OVER_MARGIN",
    "TIE_AWARE_CLAIM_LEDGER",
    "PROBE_CANDIDATE_DECISIONS",
    "TIE_CLAIM_MAX_EXPECTED_FLIPS_OVER_MARGIN",
    "TIE_CONTROL_PLANTED_FLIP_RATE",
    "TIE_PRECISION_ROWS",
    "TIE_PRECISION_SEED",
    "adjudicate_tie_aware_claims",
    "require_aggregate_tie_aware_use",
    "tie_aware_precision_profile",
    "tie_use_positive_control",
]
