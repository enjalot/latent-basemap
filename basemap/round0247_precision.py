"""R0247 — the precision fix the reviewer showed was cheap, and the sealed bound.

review-0246-01 F found R0246 right about the `float64` surprise and wrong about
its cause, and the correction makes the fix cheap:

* R0246 said `float64` buys only `1.0052x` because R0238's truth cosines are
  **stored** as `float32`, whose half-ulp at `cos ~ 1` is `5.96e-08`.
* The measured `float64`-vs-stored p99 is `5.336e-07` — **nine times larger**.
  Storage quantisation cannot produce that. What produces it is that R0238's
  truth cosines are a `float32` **computation**: a 384-term dot product
  accumulated in `float32` carries an error of order `sqrt(384) * eps ~ 2e-6`
  worst case and a few `1e-7` typically. The residual is the truth's
  *arithmetic*, not its container.
* Therefore re-deriving the truth **cosines** does not need the exact search at
  all. The truth **ids** are already sealed. The operation is a gather of
  `500,000 x 15 = 7,500,000` substrate rows and a `float64` dot per pair — CPU
  work, no GPU, and it removes the dominant term.

This module does that (`recompute_truth_cosines_f64`), measures what it bought
(`cosine_noise_floor`, which separates storage quantisation from arithmetic by
recomputing the same pairs a second time in a different contraction order), and
states the tolerance that is defensible against the new floor
(`defensible_tolerance`).

It also fixes the ledger, in the two ways review-0246-01 E asked for:

* `poisson_upper_bound` generalises the rule of three. `3/n` is the 95% one-
  sided Poisson upper limit at **zero** observed events; this is the same limit
  at any `k`, so a non-zero measurement is bounded by the same rule instead of
  being adjudicated at a point estimate.
* `sealed_bound_adjudication` makes the bound adjudication a **receipt a
  reviewer can recompute** rather than the prose column review-0246-01 E found
  the whole corrections programme resting on. It runs the sealed ledger through
  the sealed function at the bound, at both registered margin fractions, and
  publishes the counts, the per-claim arithmetic, and the delta against R0246's
  point-estimate adjudication.

Nothing here signals anything, starts a child process, touches the GPU, or
imports a GPU array library. Every bulk input is a read-only `np.memmap`.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from basemap.round0227_low_c_contract import TIE_TOLERANCE
from basemap.round0246_tie import (
    TIE_AWARE_CLAIM_LEDGER,
    TIE_CLAIM_MAX_EXPECTED_FLIPS_OVER_MARGIN,
    adjudicate_tie_aware_claims,
)
from basemap.round0247_registry import (
    TIE_BOUND_CONFIDENCE,
    TIE_USE_MAX_EXPECTED_FLIPS_OVER_MARGIN,
    Round0247Error,
    registered_bounds,
    verify_registry,
)

#: `float32` has a 24-bit significand, so the half-ulp of a value in `[0.5, 1)`
#: is `2**-25`. This is the number R0246 attributed the whole residual to.
FLOAT32_HALF_ULP_AT_ONE = 2.0 ** -25

#: The dimension of the substrate R0238's truth cosines are dot products over.
#: `sqrt(384) * 2**-24 = 1.17e-06` is the standard worst-case bound on the
#: accumulated rounding error of a `float32` 384-term dot product, and
#: `sqrt(384) * 2**-24 / sqrt(3) ~ 6.7e-07` is the usual stochastic estimate.
#: Both are the right order for the measured `5.34e-07`; the storage half-ulp
#: is 9x too small. This is review-0246-01 F's arithmetic, restated as the
#: prediction R0247 tests.
SUBSTRATE_DIMENSION = 384

PRECISION_NOTE = (
    "review-0246-01 F. R0246 attributed the float64 residual to the float32 "
    "STORAGE of R0238's truth cosines, whose half-ulp at cos ~ 1 is "
    f"{FLOAT32_HALF_ULP_AT_ONE}. The measured p99 was 5.336e-07, nine times "
    "larger, so storage cannot be the cause: R0238's truth cosines are a "
    "float32 COMPUTATION and the residual is its accumulated arithmetic. That "
    "makes the fix a gather of 7,500,000 sealed rows and a float64 dot, not a "
    "100M-row GPU job, and R0247 runs it."
)


# --------------------------------------------------------------------------- #
# the recompute
# --------------------------------------------------------------------------- #
def recompute_truth_cosines_f64(
    *,
    substrate: np.ndarray,
    probe_query_rows: np.ndarray,
    truth_ids: np.ndarray,
    abort_check: Any = None,
    block: int = 2_000,
) -> dict[str, Any]:
    """Recompute the truth cosines in `float64` from the SEALED truth ids.

    The ids are not re-derived — re-deriving them is the exact search, which is
    the expensive job R0246 priced. Only the cosines are recomputed, from the
    same substrate bytes, in `float64`. Returns the array and its statistics.
    """
    rows = int(np.asarray(truth_ids).shape[0])
    k = int(np.asarray(truth_ids).shape[1])
    out = np.empty((rows, k), dtype=np.float64)
    gathered_rows = 0
    for start in range(0, rows, int(block)):
        if abort_check is not None:
            abort_check(
                f"R0247 truth cosine recompute rows [{start}, {rows})"
            )
        stop = min(start + int(block), rows)
        queries = np.asarray(probe_query_rows[start:stop], dtype=np.int64)
        candidate_ids = np.asarray(truth_ids[start:stop], dtype=np.int64)
        flat = candidate_ids.reshape(-1)
        #: The sorted gather R0243 priced: random reads over a 153 GB memmap
        #: are ordered so the page cache sees a forward scan.
        order = np.argsort(flat, kind="stable")
        gathered = np.empty(
            (flat.size, int(substrate.shape[1])), dtype=np.float64
        )
        gathered[order] = np.asarray(substrate[flat[order]], dtype=np.float64)
        gathered = gathered.reshape(
            candidate_ids.shape[0], candidate_ids.shape[1], -1
        )
        anchors = np.asarray(substrate[queries], dtype=np.float64)
        out[start:stop] = np.einsum("bd,bkd->bk", anchors, gathered)
        gathered_rows += int(flat.size)
        del candidate_ids, gathered, anchors, flat, order
    if abort_check is not None:
        abort_check("R0247 truth cosine recompute complete")
    return {
        "instrument": "round0247-truth-cosine-f64-recompute-v1",
        "rows": rows,
        "k": k,
        "substrate_rows_gathered": gathered_rows,
        "substrate_bytes_gathered": int(
            gathered_rows * int(substrate.shape[1]) * 4
        ),
        "cosines": out,
        "note": PRECISION_NOTE,
    }


def cosine_noise_floor(
    *,
    stored_f32: np.ndarray,
    recomputed_f64: np.ndarray,
    substrate: np.ndarray,
    probe_query_rows: np.ndarray,
    truth_ids: np.ndarray,
    abort_check: Any = None,
    control_rows: int = 20_000,
    block: int = 2_000,
) -> dict[str, Any]:
    """Separate STORAGE quantisation from ARITHMETIC, by measuring both.

    Three quantities, from the same pairs:

    * `stored_vs_recomputed` — `|float32 stored truth - float64 recompute|`.
      This is the residual R0246 measured and misattributed. If storage were
      the cause it would sit at `2**-25 = 2.98e-08`; review-0246-01 F predicts
      it sits an order of magnitude higher.
    * `storage_quantisation` — `|float64 recompute - float32(float64
      recompute)|`. The pure container effect, with no arithmetic in it at all.
    * `float64_arithmetic` — the same pairs contracted a SECOND way in
      `float64` (`(a * b).sum(-1)` instead of `einsum`), differenced against the
      first. This is the new floor: what remains when both sides are `float64`.
    """
    stored = np.asarray(stored_f32, dtype=np.float64)
    recomputed = np.asarray(recomputed_f64, dtype=np.float64)
    if stored.shape != recomputed.shape:
        raise Round0247Error(
            f"R0247 noise floor needs matching shapes, got {stored.shape} and "
            f"{recomputed.shape}"
        )
    delta_storage_and_arithmetic = np.abs(stored - recomputed).reshape(-1)
    quantised = np.abs(
        recomputed - recomputed.astype(np.float32).astype(np.float64)
    ).reshape(-1)

    rows = int(min(int(control_rows), recomputed.shape[0]))
    second_order: list[np.ndarray] = []
    for start in range(0, rows, int(block)):
        if abort_check is not None:
            abort_check(f"R0247 float64 contraction control [{start}, {rows})")
        stop = min(start + int(block), rows)
        queries = np.asarray(probe_query_rows[start:stop], dtype=np.int64)
        candidate_ids = np.asarray(truth_ids[start:stop], dtype=np.int64)
        flat = candidate_ids.reshape(-1)
        order = np.argsort(flat, kind="stable")
        gathered = np.empty(
            (flat.size, int(substrate.shape[1])), dtype=np.float64
        )
        gathered[order] = np.asarray(substrate[flat[order]], dtype=np.float64)
        gathered = gathered.reshape(
            candidate_ids.shape[0], candidate_ids.shape[1], -1
        )
        anchors = np.asarray(substrate[queries], dtype=np.float64)
        #: A different contraction order over the same bytes.
        alternate = (anchors[:, None, :] * gathered).sum(axis=-1)
        second_order.append(
            np.abs(alternate - recomputed[start:stop]).reshape(-1)
        )
        del candidate_ids, gathered, anchors, alternate, flat, order
    arithmetic = (
        np.concatenate(second_order) if second_order
        else np.zeros(1, dtype=np.float64)
    )

    def _stats(values: np.ndarray) -> dict[str, Any]:
        return {
            "pairs": int(values.size),
            "mean": float(values.mean()),
            "p50": float(np.quantile(values, 0.50)),
            "p99": float(np.quantile(values, 0.99)),
            "max": float(values.max()),
        }

    stored_stats = _stats(delta_storage_and_arithmetic)
    quantisation_stats = _stats(quantised)
    arithmetic_stats = _stats(arithmetic)
    ratio = (
        stored_stats["p99"] / quantisation_stats["p99"]
        if quantisation_stats["p99"] > 0 else None
    )
    return {
        "instrument": "round0247-cosine-noise-floor-v1",
        "stored_vs_recomputed": stored_stats,
        "storage_quantisation": quantisation_stats,
        "float64_arithmetic": arithmetic_stats,
        "float32_half_ulp_at_one": FLOAT32_HALF_ULP_AT_ONE,
        "stored_p99_over_storage_quantisation_p99": ratio,
        "float32_dot_worst_case_error_bound": float(
            math.sqrt(SUBSTRATE_DIMENSION) * (2.0 ** -24)
        ),
        "float32_dot_stochastic_error_estimate": float(
            math.sqrt(SUBSTRATE_DIMENSION) * (2.0 ** -24) / math.sqrt(3.0)
        ),
        "the_residual_is_arithmetic_not_storage": bool(
            ratio is not None and ratio > 3.0
        ),
        "improvement_over_the_stored_reference": (
            stored_stats["p99"] / arithmetic_stats["p99"]
            if arithmetic_stats["p99"] > 0 else None
        ),
        "note": PRECISION_NOTE,
    }


#: How far above the measured `float64` noise floor a tolerance has to sit to
#: be defensible. `1000x` is not a fudge: the tie-aware test is `cos >= kth -
#: tolerance`, so the tolerance must dominate the noise in BOTH cosines and in
#: their difference, and three orders of magnitude is the smallest round number
#: at which a p99 and a max cannot be confused. R0246 shipped a tolerance at
#: `1.86x` its noise floor with a MAXIMUM disagreement that exceeded it; this
#: is the number that condition failed against.
DEFENSIBLE_TOLERANCE_OVER_NOISE = 1_000.0


def defensible_tolerance(floor: Mapping[str, Any]) -> dict[str, Any]:
    """What tolerance the new floor supports. STATED, not applied.

    `TIE_TOLERANCE` is registered by R0227 and every published R0241 and R0243
    tie-aware figure consumes it. Moving it here would change those figures, so
    R0247 does not move it: it publishes what the recompute makes defensible and
    leaves the change to a round that re-derives the figures.
    """
    arithmetic_p99 = float(floor["float64_arithmetic"]["p99"])
    arithmetic_max = float(floor["float64_arithmetic"]["max"])
    stored_p99 = float(floor["stored_vs_recomputed"]["p99"])
    supported = max(arithmetic_max, arithmetic_p99) * (
        DEFENSIBLE_TOLERANCE_OVER_NOISE
    )
    #: Round down to a power of ten so the registered number is a number and
    #: not a measurement artefact.
    decade = (
        10.0 ** math.floor(math.log10(supported)) if supported > 0 else 0.0
    )
    return {
        "instrument": "round0247-defensible-tolerance-v1",
        "current_tie_tolerance": float(TIE_TOLERANCE),
        "float64_noise_floor_p99": arithmetic_p99,
        "float64_noise_floor_max": arithmetic_max,
        "stored_reference_noise_p99": stored_p99,
        "current_tolerance_over_the_stored_floor": (
            float(TIE_TOLERANCE) / stored_p99 if stored_p99 > 0 else None
        ),
        "current_tolerance_over_the_float64_floor": (
            float(TIE_TOLERANCE) / arithmetic_p99 if arithmetic_p99 > 0
            else None
        ),
        "required_multiple_over_the_noise": DEFENSIBLE_TOLERANCE_OVER_NOISE,
        "smallest_defensible_tolerance": supported,
        "smallest_defensible_tolerance_rounded_to_a_decade": decade,
        "the_tolerance_was_not_moved": True,
        "why_not": (
            "TIE_TOLERANCE is registered by R0227 and consumed by every "
            "published R0241 and R0243 tie-aware figure. Lowering it would "
            "change those figures, which is a re-derivation and not a "
            "threshold move. R0247 publishes what the recompute makes "
            "defensible and leaves the change to the round that re-derives "
            "them."
        ),
        "what_this_buys": (
            "with both cosines in float64 the estimator's own noise falls from "
            "the stored reference's arithmetic to float64 arithmetic, so a "
            "tolerance three orders of magnitude above the new floor is still "
            "orders of magnitude BELOW the current 1e-06 - i.e. the tie-aware "
            "test stops being a threshold at the edge of its own noise and "
            "becomes a test of genuine near-ties in the true cosine"
        ),
        "note": PRECISION_NOTE,
    }


# --------------------------------------------------------------------------- #
# the bound — the rule of three, generalised, and sealed
# --------------------------------------------------------------------------- #
def _regularized_lower_gamma(shape: float, x: float) -> float:
    """`P(a, x)`, by series below the transition and continued fraction above."""
    if x <= 0.0:
        return 0.0
    if x < shape + 1.0:
        term = 1.0 / shape
        total = term
        index = shape
        for _ in range(10_000):
            index += 1.0
            term *= x / index
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return total * math.exp(-x + shape * math.log(x) - math.lgamma(shape))
    tiny = 1e-300
    b = x + 1.0 - shape
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for index in range(1, 10_000):
        an = -index * (index - shape)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return 1.0 - math.exp(-x + shape * math.log(x) - math.lgamma(shape)) * h


def poisson_upper_bound(
    events: int, *, confidence: float = TIE_BOUND_CONFIDENCE
) -> float:
    """The one-sided upper limit on a Poisson mean given `events` observed.

    At `events = 0` and `confidence = 0.95` this is `-ln(0.05) = 2.9957`, i.e.
    the rule of three. R0247 registers the CONFIDENCE rather than the formula
    so that a non-zero observation is bounded by the same rule instead of being
    adjudicated at a point estimate, which is review-0246-01 E's objection to
    retracting eight precisions on `0/300,000`.
    """
    k = int(events)
    if k < 0:
        raise Round0247Error("R0247 bound needs a non-negative event count")
    alpha = 1.0 - float(confidence)
    if not 0.0 < alpha < 1.0:
        raise Round0247Error("R0247 bound needs a confidence in (0, 1)")
    if k == 0:
        return -math.log(alpha)
    #: Solve `P(X <= k; lambda) = alpha` for lambda. `P(X <= k; lambda)` is
    #: `1 - P(k+1, lambda)` with `P` the regularised lower incomplete gamma, and
    #: it is monotone decreasing in lambda, so a bisection is exact enough.
    low, high = 0.0, float(k) + 1.0
    while 1.0 - _regularized_lower_gamma(k + 1.0, high) > alpha:
        high *= 2.0
        if high > 1e12:
            raise Round0247Error("R0247 bound failed to bracket")
    for _ in range(200):
        middle = 0.5 * (low + high)
        if 1.0 - _regularized_lower_gamma(k + 1.0, middle) > alpha:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def flip_rate_bound(profile: Mapping[str, Any]) -> dict[str, Any]:
    """The measured flip rate and its upper bound, as arithmetic."""
    flips = profile["verdict_flips"]
    observed = int(flips["total"])
    decisions = int(profile["candidate_decisions_scored"])
    if decisions <= 0:
        raise Round0247Error("R0247 bound needs a positive decision count")
    limit = poisson_upper_bound(observed)
    return {
        "instrument": "round0247-flip-rate-bound-v1",
        "observed_flips": observed,
        "candidate_decisions_scored": decisions,
        "point_estimate_flip_rate": float(observed) / float(decisions),
        "poisson_upper_limit_events": limit,
        "upper_bound_flip_rate": limit / float(decisions),
        "confidence": float(TIE_BOUND_CONFIDENCE),
        "is_the_rule_of_three": bool(observed == 0),
        "rule_of_three_check": (
            3.0 / float(decisions) if observed == 0 else None
        ),
        "why": (
            "the rule of three, 3/n, is the 95% one-sided Poisson upper limit "
            "at zero observed events (-ln 0.05 = 2.9957). Registering the "
            "confidence rather than the formula means a non-zero measurement "
            "is bounded by the same rule instead of being adjudicated at a "
            "point estimate. review-0246-01 E: '0/300,000 is a weak experiment "
            "for a 1e-05 question: the bound is a property of the sample size, "
            "not of the estimator'."
        ),
        **registered_bounds(["tie_bound_confidence"]),
    }


def sealed_bound_adjudication(
    profile: Mapping[str, Any],
    *,
    ledger: Sequence[Mapping[str, Any]] = TIE_AWARE_CLAIM_LEDGER,
) -> dict[str, Any]:
    """The bound adjudication as a RECEIPT, at both registered criteria.

    review-0246-01 E: "The bound adjudication is prose, not receipt. The sealed
    `claim_adjudication` ran at the point estimate only ... The entire
    corrections programme rests on an author-computed column." This runs the
    sealed ledger through the sealed function at the sealed bound and publishes
    every intermediate, so a reviewer recomputes rather than re-derives.
    """
    verify_registry(label="R0247 sealed bound adjudication")
    bound = flip_rate_bound(profile)
    at_bound = dict(profile)
    at_bound["verdict_flips"] = {
        **dict(profile["verdict_flips"]),
        "per_candidate_flip_rate": bound["upper_bound_flip_rate"],
    }
    retrospective = adjudicate_tie_aware_claims(
        at_bound, ledger=ledger,
        margin_fraction=TIE_CLAIM_MAX_EXPECTED_FLIPS_OVER_MARGIN,
    )
    prospective = adjudicate_tie_aware_claims(
        at_bound, ledger=ledger,
        margin_fraction=TIE_USE_MAX_EXPECTED_FLIPS_OVER_MARGIN,
    )
    point = adjudicate_tie_aware_claims(profile, ledger=ledger)

    def _names(adjudication: Mapping[str, Any]) -> list[str]:
        return [
            f"{row['round']}: {row['claim']}"
            for row in adjudication["claims"] if not row["survives"]
        ]

    retrospective_failing = _names(retrospective)
    prospective_failing = _names(prospective)
    return {
        "instrument": "round0247-sealed-bound-adjudication-v1",
        "flip_rate_bound": bound,
        "ledger_size": len(ledger),
        #: review-0246-01 E: "the round's own executable ledger says EIGHT
        #: claims fail at the bound, not seven ... The digest, the outcome
        #: string in the front matter and the appended text all say 'seven'."
        #: This is that count, computed here rather than written here, and the
        #: already-repaired claim is counted and separately labelled instead of
        #: being silently dropped.
        "claims_that_survive_at_the_bound": int(
            retrospective["claims_that_survive"]
        ),
        "claims_that_do_not_survive_at_the_bound": len(retrospective_failing),
        "claims_that_do_not_survive_at_the_bound_names": retrospective_failing,
        "already_repaired_among_the_non_survivors": [
            f"{row['round']}: {row['claim']}"
            for row in retrospective["claims"]
            if not row["survives"] and row.get("already_repaired")
        ],
        "corrections_owed": [
            f"{row['round']}: {row['claim']}"
            for row in retrospective["claims"]
            if not row["survives"] and not row.get("already_repaired")
        ],
        "at_the_retrospective_criterion": retrospective,
        "at_the_prospective_criterion": prospective,
        "claims_that_do_not_survive_the_prospective_criterion": (
            prospective_failing
        ),
        "at_the_point_estimate": {
            "measured_per_candidate_flip_rate": point[
                "measured_per_candidate_flip_rate"
            ],
            "claims_that_survive": point["claims_that_survive"],
            "claims_that_do_not_survive": len(
                point["claims_that_do_not_survive"]
            ),
        },
        "why_the_bound_and_not_the_point_estimate": (
            "a point estimate of 0.0 says a claim survives because the "
            "experiment observed nothing, which is a statement about the "
            "sample size. The bound is the honest standard for a published "
            "claim and it is what R0246 chose; what R0246 did not do is seal "
            "it. This is the seal."
        ),
        **registered_bounds([
            "tie_claim_max_expected_flips_over_margin",
            "tie_use_max_expected_flips_over_margin",
            "tie_bound_confidence",
        ]),
    }


__all__ = [
    "DEFENSIBLE_TOLERANCE_OVER_NOISE",
    "FLOAT32_HALF_ULP_AT_ONE",
    "PRECISION_NOTE",
    "SUBSTRATE_DIMENSION",
    "cosine_noise_floor",
    "defensible_tolerance",
    "flip_rate_bound",
    "poisson_upper_bound",
    "recompute_truth_cosines_f64",
    "sealed_bound_adjudication",
]
