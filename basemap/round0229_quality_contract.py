"""R0229 — the frozen contract: sweep grids, the registered test, the projections.

Nothing here touches a round0215-0228 file. Everything R0226/R0227/R0228 already
measured is *imported* read-only, never restated, so a number can only disagree
with a prior round by disagreeing with its bytes.

Three registered objects live here:

1. **The structural bound.** R0227 measured the spill partition's reachability
   ceiling over all 2,000,000 rows. It bounds what *any* nn-descent setting can
   reach at a given `(c, s)`, before nn-descent runs and independently of how
   well it runs. At `c = 16, s = 2` the ceiling is `0.953101` tie-aware against a
   built `0.951162`, so the whole nn-descent headroom is `0.001939` and `c = 4`'s
   `0.988947` is `0.035846` out of reach. The sweep tests this as a falsifiable
   prediction: a cell above its own partition's ceiling means the instrument is
   wrong.

2. **The registered displacement test.** R0228 reported a difference in
   differences in units of the null arm's dispersion and declined to supply an
   inference rule. Here it is an exact one-sided permutation test over all
   `C(11, 3) = 165` relabellings of eleven map-level gaps, with a registered
   decision rule and its smallest attainable `p` enumerated beside it. R0228's
   result called the statistic un-registered; review-0228-01 blocked that — it
   WAS registered in `round-0228` section Geometry item 3, formula included,
   and only the inference rule was missing. That rule is supplied here.

3. **The per-rung `c`, from measurement.** Review-0227-01 found R0227 published
   `c = 22` at 100M off review-0226-01's imbalance *model* while its own artifact
   said so; `c = 24` is the measured answer. This module reads R0227's sealed
   measured imbalance and uses nothing else.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

from basemap.round0226_graph_builders import (
    A_GRAPH_DEGREE,
    A_INTERMEDIATE_DEGREE,
    A_MAX_ITERATIONS,
    A_METRIC,
    A_SEED,
    A_SPILL,
    GRAPH_K,
)
from basemap.round0227_low_c_contract import (
    CLUSTER_CAPACITY_ROWS,
    GUARD_BUDGET_NOTE,
    GUARD_DEVICE_BUDGET_BYTES,
    GUARD_HOST_ANON_BUDGET_BYTES,
    GUARD_SWAP_GROWTH_ABORT_BYTES,
    SAMPLE_INTERVAL_S,
    guard_decision,
)

ROUND_ID = "0229"
ROWS = 2_000_000
DIMENSION = 384

SWEEP_SCHEMA = "round0229-nnd-quality-sweep-v1"
SPILL_SCHEMA = "round0229-spill-reachability-v1"
RETRO_SCHEMA = "round0229-retrospective-displacement-v1"
BUILD_SCHEMA = "round0229-quality-build-v1"
REACHABILITY_SCHEMA = "round0229-spill-reachability-probe-v1"

SWEEP_CAPABILITY = "minilm-mixed-2m-nnd-quality-sweep-v1"
SPILL_CAPABILITY = "minilm-mixed-2m-spill-reachability-v1"
RETRO_CAPABILITY = "minilm-mixed-2m-registered-displacement-test-v1"

GATE_REGISTERABLE_HERE = False
GATE_RELEASE_CLAIMED = False
ADOPTION_CLAIMED = False
EQUIVALENCE_CLAIMED = False
TRAINING_PERFORMED = False

#: Every recall and ceiling headline is over every substrate row. Review-0227-01
#: caught a headline scored on seeds unioned with their exact neighbours, a
#: size-biased set whose mean 15th-NN cosine was 0.652 against the substrate's
#: 0.604. There is no sampled recall population anywhere in this round.
RECALL_POPULATION = "all-2000000-substrate-rows"
RECALL_POPULATION_NOTE = (
    "every recall and strict-ceiling figure in R0229 is over all 2,000,000 "
    "substrate rows; the only sampled instrument is the tie-aware ceiling in the "
    "spill grid, which is registered as a 200,000-row seeded uniform sample "
    "(SE ~2e-4) and is never the basis of a decision"
)


class Round0229Error(RuntimeError):
    """R0229 refused to proceed."""


# --------------------------------------------------------------------------- #
# 1. the structural bound — R0227's sealed ceilings, and R0228's sealed builds
# --------------------------------------------------------------------------- #
#: R0227's measured structural reachability ceilings at 2M, spill s = 2, strict
#: over all 2,000,000 rows. Released by review-0227-01. The node re-reads these
#: from R0227's sealed artifact and refuses if they disagree; the copy here
#: exists so the registered arithmetic is auditable without loading 32 kB of
#: JSON, and `verify_r0227_ceilings` is what actually binds.
R0227_STRICT_CEILING_BY_C = {
    4: 0.9915615666666667,
    8: 0.9735297333333338,
    16: 0.9532495999999998,
    24: 0.9447928666666668,
    32: 0.9356989333333326,
    48: 0.9194068333333335,
    64: 0.911356,
    100: 0.8948668666666666,
    200: 0.8676522000000004,
}
R0227_TIE_CEILING_BY_C = {
    4: 0.9915466666666667,
    8: 0.9732266666666665,
    16: 0.9531013333333335,
    24: 0.9437000000000001,
    32: 0.9348306666666668,
    48: 0.9188173333333334,
    64: 0.9096413333333334,
    100: 0.8941426666666668,
    200: 0.8675479999999999,
}
#: R0227's measured max/mean cluster imbalance at 2M, spill s = 2. This is the
#: *measured* set; review-0226-01's model is never used in this round.
R0227_MEASURED_IMBALANCE = {
    4: 1.128894,
    8: 1.215494,
    16: 1.27412,
    24: 1.4502000000000002,
    32: 1.457544,
    48: 1.896792,
    64: 1.679024,
    100: 1.7241,
    200: 2.5919,
}
#: Review-0227-01: imbalance at fixed `c` is noisy in `N` (c = 16 gives
#: 1.2742 / 1.3295 / 1.2374 / 1.4842 at 2M / 4M / 8M / 16M), so a point estimate
#: is the wrong instrument and every rung carries the upper end beside it.
R0227_IMBALANCE_BY_N_AT_C16 = {2_000_000: 1.2742, 4_000_000: 1.3295,
                              8_000_000: 1.2374, 16_000_000: 1.4842}
IMBALANCE_UPPER_MULTIPLIER = max(R0227_IMBALANCE_BY_N_AT_C16.values()) / (
    R0227_IMBALANCE_BY_N_AT_C16[2_000_000]
)

#: R0228's sealed uniform tie-aware recall over all 2,000,000 rows.
R0228_TIE_AWARE_RECALL_BY_C = {4: 0.988947, 8: 0.970770, 16: 0.951162}
#: R0228's sealed null-arm DiD, in exact-family sd. Re-derived here as p-values.
R0228_DID_IN_EXACT_SD_BY_C = {4: 0.027, 8: 3.332, 16: 3.939}
#: Review-0228-01's own exact permutation p-values on the same sealed per-map
#: gaps, at 0.0 GPU seconds, with complete separation at c = 8 and c = 16. This
#: round's retrospective node is third-codebase CONFIRMATION of these, never a
#: new result; a disagreement refuses the node.
REVIEW_0228_DISPLACEMENT_P_BY_C = {4: 0.43636, 8: 0.00606, 16: 0.00606}
REVIEW_0228_P_TOLERANCE = 5e-4
#: R0228's sealed fraction of rows carrying any loss, over all 2,000,000 rows.
#: This is the trend test's regressor: review-0228-01 established the effect is
#: monotone in missing edge mass, not in `c`.
R0228_ROWS_CARRYING_LOSS_BY_C = {4: 0.1042965, 8: 0.1974085, 16: 0.2698095}
RETROSPECTIVE_LABEL = (
    "RETROSPECTIVE CONFIRMATION of review-0228-01's own permutation test on "
    "R0228's sealed per-map gaps; not a new result of this round. Review-0228-01 "
    "blocked claim:r0228-the-displacement-statistic-was-not-pre-registered: the "
    "statistic WAS registered in round-0228 section Geometry item 3, formula "
    "included; only the inference rule was absent."
)


def verify_r0228_displacement(sealed: Mapping[str, Any]) -> dict[str, Any]:
    """Bind every R0228 number this round carries to R0228's sealed bytes.

    Review-0228-01 found R0228's *prose* misreported its own sealed clump cells
    as `26 / 26 / 26` when the artifact holds `28 / 29 / 21`. Nothing here is
    carried from prose: the per-map gaps, the DiD, the exact-null dispersion and
    the rows carrying loss are read from the artifact and compared against the
    registered constants, and a mismatch refuses.
    """
    displacement = sealed.get("displacement")
    if not isinstance(displacement, Mapping):
        raise Round0229Error("R0228 geometry artifact has no displacement block")
    checked: dict[str, Any] = {}
    for clusters in sorted(R0228_DID_IN_EXACT_SD_BY_C):
        cell = displacement.get(str(clusters))
        if not isinstance(cell, Mapping):
            raise Round0229Error(f"R0228 displacement is missing c = {clusters}")
        exact_arm = cell.get("vs_exact_family") or cell
        per_map = exact_arm.get("per_map")
        if not isinstance(per_map, Mapping):
            raise Round0229Error(
                f"R0228 displacement at c = {clusters} has no per_map gaps"
            )
        candidate_names = list(exact_arm["candidate_maps"])
        exact_names = list(exact_arm["exact_maps"])
        if len(candidate_names) != CANDIDATE_MAPS_PER_ARM:
            raise Round0229Error(
                f"R0228 c = {clusters} has {len(candidate_names)} candidate maps"
            )
        if len(exact_names) != EXACT_NULL_MAPS:
            raise Round0229Error(
                f"R0228 c = {clusters} has {len(exact_names)} exact null maps"
            )
        did_sd = float(exact_arm["difference_in_differences_in_exact_sd"])
        if abs(did_sd - R0228_DID_IN_EXACT_SD_BY_C[clusters]) > 5e-3:
            raise Round0229Error(
                f"R0229's registered DiD sd at c = {clusters} is "
                f"{R0228_DID_IN_EXACT_SD_BY_C[clusters]} against R0228's sealed "
                f"{did_sd}"
            )
        loss_fraction = float(cell["rows_carrying_loss_fraction"])
        if abs(loss_fraction - R0228_ROWS_CARRYING_LOSS_BY_C[clusters]) > 1e-6:
            raise Round0229Error(
                f"R0229's registered rows-carrying-loss at c = {clusters} is "
                f"{R0228_ROWS_CARRYING_LOSS_BY_C[clusters]} against R0228's "
                f"sealed {loss_fraction}"
            )
        checked[str(clusters)] = {
            "candidate_maps": candidate_names,
            "exact_maps": exact_names,
            "rows_carrying_loss_fraction": loss_fraction,
            "rows_carrying_loss": int(cell["rows_carrying_loss"]),
            "candidate_gaps": [
                float(per_map[name]["gap_lost_minus_control"])
                for name in candidate_names
            ],
            "exact_gaps": [
                float(per_map[name]["gap_lost_minus_control"]) for name in exact_names
            ],
            "difference_in_differences": float(
                exact_arm["difference_in_differences"]
            ),
            "difference_in_differences_in_exact_sd": did_sd,
            "density_match_exact": bool(
                (cell.get("density_match") or {}).get("matched_exactly")
            ),
        }
    return {"bound_to": "R0228 sealed geometry artifact", "cells": checked}

#: The bound the round registers before running: at c = 16, s = 2, every
#: nn-descent knob together can buy at most this much tie-aware recall, and the
#: gap to c = 4's built graph cannot be closed at all.
NND_HEADROOM_AT_C16 = R0227_TIE_CEILING_BY_C[16] - R0228_TIE_AWARE_RECALL_BY_C[16]
C4_UNREACHABLE_MARGIN_AT_C16 = (
    R0227_TIE_CEILING_BY_C[16] - R0228_TIE_AWARE_RECALL_BY_C[4]
)
STRUCTURAL_BOUND_NOTE = (
    "the spill partition's reachability ceiling bounds every within-cluster "
    "search at every nn-descent setting; at c = 16, s = 2, 2M it is 0.953101 "
    f"tie-aware against a built 0.951162, so the total nn-descent headroom is "
    f"{NND_HEADROOM_AT_C16:.6f} and c = 4's built 0.988947 is "
    f"{-C4_UNREACHABLE_MARGIN_AT_C16:.6f} out of reach at any setting"
)


def verify_r0227_ceilings(sealed: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the registered ceilings to R0227's sealed artifact, or refuse.

    The round's whole bound rests on these nine numbers, so they are read from
    the bytes review-0227-01 released rather than trusted from this module.
    """
    ceilings = sealed.get("ceilings_by_clusters")
    if not isinstance(ceilings, Mapping):
        raise Round0229Error("R0227 reachability artifact has no ceilings_by_clusters")
    checked: dict[str, Any] = {}
    for clusters, expected in sorted(R0227_STRICT_CEILING_BY_C.items()):
        cell = ceilings.get(str(clusters))
        if not isinstance(cell, Mapping):
            raise Round0229Error(f"R0227 ceilings are missing c = {clusters}")
        strict = float(cell["strict_mean_all_rows"])
        tie = float(cell["tie_mean_query_sample"])
        imbalance = float(cell["imbalance_max_over_mean"])
        if abs(strict - expected) > 1e-12:
            raise Round0229Error(
                f"R0229's registered strict ceiling at c = {clusters} is {expected} "
                f"against R0227's sealed {strict}"
            )
        if abs(tie - R0227_TIE_CEILING_BY_C[clusters]) > 1e-12:
            raise Round0229Error(
                f"R0229's registered tie ceiling at c = {clusters} disagrees with R0227"
            )
        if abs(imbalance - R0227_MEASURED_IMBALANCE[clusters]) > 1e-9:
            raise Round0229Error(
                f"R0229's registered imbalance at c = {clusters} disagrees with R0227"
            )
        checked[str(clusters)] = {
            "strict_ceiling_all_rows": strict,
            "tie_ceiling_query_sample": tie,
            "measured_imbalance_max_over_mean": imbalance,
        }
    return {
        "bound_to": "R0227 sealed reachability artifact",
        "cells": checked,
        "structural_bound_note": STRUCTURAL_BOUND_NOTE,
        "nn_descent_headroom_at_c16_tie_aware": NND_HEADROOM_AT_C16,
        "c4_margin_unreachable_at_c16_tie_aware": C4_UNREACHABLE_MARGIN_AT_C16,
    }


# --------------------------------------------------------------------------- #
# 2. the two grids
# --------------------------------------------------------------------------- #
SWEEP_CLUSTERS = 16
SWEEP_SPILL = A_SPILL

#: Ascending in cost. The ladder stops on the first refusal, abort or timeout,
#: which is the R0224 rule and the reason this box is still up.
QUALITY_SWEEP: tuple[dict[str, Any], ...] = (
    {"cell": "q0-baseline", "graph_degree": 32, "intermediate_graph_degree": 48,
     "max_iterations": 20},
    {"cell": "q1", "graph_degree": 32, "intermediate_graph_degree": 64,
     "max_iterations": 20},
    {"cell": "q2", "graph_degree": 32, "intermediate_graph_degree": 96,
     "max_iterations": 20},
    {"cell": "q3", "graph_degree": 32, "intermediate_graph_degree": 128,
     "max_iterations": 20},
    {"cell": "q4", "graph_degree": 32, "intermediate_graph_degree": 128,
     "max_iterations": 40},
    {"cell": "q5", "graph_degree": 32, "intermediate_graph_degree": 256,
     "max_iterations": 20},
    {"cell": "q6", "graph_degree": 64, "intermediate_graph_degree": 128,
     "max_iterations": 20},
    {"cell": "q7", "graph_degree": 64, "intermediate_graph_degree": 256,
     "max_iterations": 40},
)
BASELINE_CELL = "q0-baseline"

#: `(c, s)` cells. Families hold `c / s` — and therefore mean cluster rows and
#: device cost — constant, so the only thing that varies is how the same budget
#: is spent. The three 100M-feasible configurations are measured as themselves.
SPILL_GRID: tuple[dict[str, Any], ...] = (
    {"cell": "A-c8-s1", "family": "A", "clusters": 8, "spill": 1},
    {"cell": "A-c16-s2", "family": "A", "clusters": 16, "spill": 2},
    {"cell": "A-c32-s4", "family": "A", "clusters": 32, "spill": 4},
    {"cell": "A-c64-s8", "family": "A", "clusters": 64, "spill": 8},
    {"cell": "B-c2-s1", "family": "B", "clusters": 2, "spill": 1},
    {"cell": "B-c4-s2", "family": "B", "clusters": 4, "spill": 2},
    {"cell": "B-c8-s4", "family": "B", "clusters": 8, "spill": 4},
    {"cell": "B-c16-s8", "family": "B", "clusters": 16, "spill": 8},
    {"cell": "F-c24-s2", "family": "F", "clusters": 24, "spill": 2},
    {"cell": "F-c64-s4", "family": "F", "clusters": 64, "spill": 4},
    {"cell": "F-c200-s8", "family": "F", "clusters": 200, "spill": 8},
)
#: `(16, 2)` and `(4, 2)` reproduce R0227's sealed ceilings and validate this
#: round's implementation of the same instrument against its bytes.
SPILL_CONTROL_CELLS = {"A-c16-s2": 16, "B-c4-s2": 4}
CONTROL_CEILING_TOLERANCE = 5e-3

TIE_QUERY_ROWS = 200_000
TIE_QUERY_SEED = 229

PHASE2_CLUSTER_COUNTS_NOT_MEASURED_BY_R0227 = (2,)


def family_mean_cluster_rows(clusters: int, spill: int, *, rows: int = ROWS) -> float:
    """Mean rows per cluster. Device cost is set by the largest cluster, and
    `(c, s)` pairs with the same `c / s` therefore cost the same."""
    if clusters <= 0 or spill <= 0:
        raise Round0229Error(f"R0229 spill cell ({clusters}, {spill}) is malformed")
    return float(spill) * float(rows) / float(clusters)


# --------------------------------------------------------------------------- #
# 3. per-rung `c`, from R0227's MEASURED imbalance and nothing else
# --------------------------------------------------------------------------- #
PHASE2_RUNGS = (6_250_000, 12_500_000, 25_000_000, 50_000_000, 100_000_000)


def projected_max_cluster_rows(
    *, rows: int, clusters: int, spill: int, imbalance: float | None = None
) -> float:
    """`imbalance(c) x s x N / c`, the term the device law is a function of."""
    if imbalance is None:
        if clusters not in R0227_MEASURED_IMBALANCE:
            raise Round0229Error(
                f"c = {clusters} is not in R0227's measured imbalance set; this "
                "round never interpolates or models an imbalance"
            )
        imbalance = R0227_MEASURED_IMBALANCE[clusters]
    return float(imbalance) * family_mean_cluster_rows(clusters, spill, rows=rows)


def rung_is_feasible(*, rows: int, clusters: int, spill: int,
                     capacity_rows: int = CLUSTER_CAPACITY_ROWS) -> bool:
    return projected_max_cluster_rows(
        rows=rows, clusters=clusters, spill=spill
    ) <= float(capacity_rows)


def smallest_measured_clusters(
    *, rows: int, spill: int = A_SPILL,
    capacity_rows: int = CLUSTER_CAPACITY_ROWS,
) -> dict[str, Any]:
    """The smallest `c` **in R0227's measured set** that fits the capacity.

    Never interpolates and never uses review-0226-01's model. Where the true
    answer lies between measured points the next larger measured `c` is taken
    and the choice is labelled, which is exactly the correction review-0227-01
    required after R0227 published `c = 22` off a model.
    """
    for clusters in sorted(R0227_MEASURED_IMBALANCE):
        largest = projected_max_cluster_rows(
            rows=rows, clusters=clusters, spill=spill
        )
        if largest <= float(capacity_rows):
            upper = largest * IMBALANCE_UPPER_MULTIPLIER
            return {
                "rows": int(rows),
                "spill": int(spill),
                "clusters": int(clusters),
                "measured_imbalance": R0227_MEASURED_IMBALANCE[clusters],
                "imbalance_source": "R0227 sealed measured imbalance",
                "projected_max_cluster_rows": largest,
                "projected_max_cluster_rows_upper_end": upper,
                "upper_end_basis": (
                    "review-0227-01: imbalance at fixed c is noisy in N "
                    f"(c = 16 spans {sorted(R0227_IMBALANCE_BY_N_AT_C16.values())} "
                    "at 2M/4M/8M/16M); the upper end scales the point estimate by "
                    f"{IMBALANCE_UPPER_MULTIPLIER:.4f}"
                ),
                "upper_end_within_capacity": bool(upper <= float(capacity_rows)),
                "capacity_rows": int(capacity_rows),
                "structural_ceiling_strict_at_this_c": (
                    R0227_STRICT_CEILING_BY_C.get(clusters)
                ),
                "feasible": True,
            }
    return {
        "rows": int(rows),
        "spill": int(spill),
        "clusters": None,
        "feasible": False,
        "capacity_rows": int(capacity_rows),
        "note": (
            "no c in R0227's measured set fits the registered capacity at this "
            "(N, s); this round does not extrapolate one"
        ),
    }


# --------------------------------------------------------------------------- #
# 4. the registered displacement test
# --------------------------------------------------------------------------- #
CANDIDATE_MAPS_PER_ARM = 3
EXACT_NULL_MAPS = 8
PERMUTATION_LABELLINGS = math.comb(
    CANDIDATE_MAPS_PER_ARM + EXACT_NULL_MAPS, CANDIDATE_MAPS_PER_ARM
)
PERMUTATION_RESOLUTION_CEILING = 1.0 / float(PERMUTATION_LABELLINGS)
DISPLACEMENT_ALPHA = 0.05
C4_LIKE_MAX_ABS_DID_SD = 1.0

DISPLACEMENT_TEST_NOTE = (
    "exact one-sided permutation over all C(11, 3) = 165 relabellings of the "
    "eleven map-level gaps g_i = mean(s over lost) - mean(s over control); "
    "statistic T = mean(labelled 3) - mean(other 8); one-sided in the direction "
    "damage predicts. Resolution ceiling 1/165 = 0.0060606, so no p below that "
    "is representable and the test cannot resolve an effect finer than it."
)
DECISION_RULE_NOTE = (
    "DISPLACED iff p <= 0.05; c=4-LIKE iff p > 0.05 and |DiD_sd| <= 1.0; "
    "otherwise INDETERMINATE. A c=4-LIKE verdict licenses 'no displacement was "
    "detected at n = 3 against an eight-map null arm at this rung' and never "
    "equivalence: review-0223-01 puts the smallest certifiable margin for this "
    "design at 1.47-2.02 exact-family sd and a real equivalence test at n ~ 18."
)


def exact_displacement_permutation(
    *,
    candidate_gaps: Sequence[float],
    exact_gaps: Sequence[float],
) -> dict[str, Any]:
    """One-sided exact permutation over every relabelling of the pooled gaps.

    `candidate_gaps` and `exact_gaps` are per-map `s_lost - s_control` values.
    Under the null the arm label carries no information, so all
    `C(n_c + n_e, n_c)` labellings are equally likely.
    """
    candidate = [float(value) for value in candidate_gaps]
    exact = [float(value) for value in exact_gaps]
    if len(candidate) < 2 or len(exact) < 2:
        raise Round0229Error("R0229 displacement test needs >= 2 maps per arm")
    pooled = candidate + exact
    take = len(candidate)
    observed = (
        sum(candidate) / len(candidate) - sum(exact) / len(exact)
    )
    total = 0
    at_least = 0
    best = float("-inf")
    at_best = 0
    for chosen in combinations(range(len(pooled)), take):
        picked = set(chosen)
        left = [pooled[i] for i in chosen]
        right = [pooled[i] for i in range(len(pooled)) if i not in picked]
        statistic = sum(left) / len(left) - sum(right) / len(right)
        total += 1
        # `>=` includes the observed labelling, which is the conservative and
        # standard convention for an exact permutation p.
        if statistic >= observed - 1e-15:
            at_least += 1
        if statistic > best + 1e-15:
            best, at_best = statistic, 1
        elif statistic >= best - 1e-15:
            at_best += 1
    p_value = float(at_least) / float(total)
    # The smallest p this design could ever have produced, by enumeration
    # rather than by assuming a unique maximum. Review-0228-01 requires it
    # beside every published p.
    attainable = float(at_best) / float(total)
    exact_mean = sum(exact) / len(exact)
    exact_sd = (
        sum((value - exact_mean) ** 2 for value in exact) / (len(exact) - 1)
    ) ** 0.5
    did_sd = (observed / exact_sd) if exact_sd > 0 else float("nan")
    separation = min(candidate) > max(exact)
    return {
        "statistic": "mean(candidate gaps) - mean(exact-null gaps)",
        "difference_in_differences": observed,
        "difference_in_differences_in_exact_sd": did_sd,
        "exact_null_gap_mean": exact_mean,
        "exact_null_gap_sd": exact_sd,
        "candidate_gaps": candidate,
        "exact_gaps": exact,
        "complete_separation": bool(separation),
        "labellings": total,
        "labellings_at_or_above_observed": at_least,
        "p_one_sided": p_value,
        # Review-0228-01 requires this beside every published p: a test whose
        # smallest attainable p lies above its decision threshold cannot
        # reject, and reporting its null describes the design, not the data.
        "smallest_attainable_p": attainable,
        "labellings_at_the_maximum": at_best,
        "resolution_ceiling": 1.0 / float(total),
        "alpha": DISPLACEMENT_ALPHA,
        "can_reject_at_alpha": bool(attainable <= DISPLACEMENT_ALPHA),
        "note": DISPLACEMENT_TEST_NOTE,
    }


RESOLUTION_RULE_NOTE = (
    "review-0228-01: R0228's twelve per-configuration permutation tests had a "
    "smallest attainable p of 1/165 = 0.0060606 against their own 0.05/12 = "
    "0.0041667 correction, so no outcome could ever have cleared it and 'null "
    "after correction' described the design, not the maps. R0229 publishes "
    "smallest_attainable_p beside every p and refuses to publish any test as "
    "null under a threshold below its own resolution ceiling."
)


def test_can_reject(*, smallest_attainable_p: float, threshold: float) -> bool:
    """Could this test have produced a rejection at this threshold at all?"""
    return float(smallest_attainable_p) <= float(threshold)


#: The trend test's resolution: nine maps into three labelled arms of three.
TREND_ASSIGNMENTS = 1680
TREND_RESOLUTION_CEILING = 1.0 / float(TREND_ASSIGNMENTS)
TREND_TEST_NOTE = (
    "review-0228-01 recommendation #8, adopted as this round's primary map-side "
    "instrument because it has resolution the per-arm test does not. Nine "
    "per-map DiD values d_i = g_i - mean(g over the eight exact maps on that "
    "map's OWN configuration row sets), so each is centred on its own null arm "
    "and the nine are commensurate even though the lost/control row sets differ "
    "by configuration. Regressor: the arm's measured fraction of rows carrying "
    "any loss over all 2,000,000 rows, because the effect is monotone in missing "
    "edge mass rather than in c. Statistic |Pearson r| over all 9!/(3!3!3!) = "
    "1,680 assignments of the nine maps to three labelled arms; resolution "
    f"ceiling 1/1680 = {TREND_RESOLUTION_CEILING:.6e}."
)


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = float(len(xs))
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    sxx = sum((x - mean_x) ** 2 for x in xs)
    syy = sum((y - mean_y) ** 2 for y in ys)
    if sxx <= 0.0 or syy <= 0.0:
        return 0.0
    return sxy / math.sqrt(sxx * syy)


def exact_did_trend(
    *, arm_values: Mapping[str, Sequence[float]], regressor: Mapping[str, float]
) -> dict[str, Any]:
    """Exact permutation trend of per-map DiD on the arms' missing-edge mass.

    Enumerates every assignment of the pooled maps to the labelled arms, keeping
    each arm's size, which is `9!/(3!3!3!) = 1,680` for three arms of three.
    """
    names = list(arm_values)
    if len(names) < 3:
        raise Round0229Error("R0229 trend test needs at least three arms")
    sizes = [len(arm_values[name]) for name in names]
    pooled: list[float] = []
    observed_x: list[float] = []
    observed_y: list[float] = []
    for name in names:
        for value in arm_values[name]:
            pooled.append(float(value))
            observed_x.append(float(regressor[name]))
            observed_y.append(float(value))
    observed_r = _pearson(observed_x, observed_y)

    def _assign(remaining: tuple[int, ...], index: int):
        if index == len(sizes) - 1:
            yield [list(remaining)]
            return
        for chosen in combinations(range(len(remaining)), sizes[index]):
            picked = set(chosen)
            head = [remaining[i] for i in chosen]
            tail = tuple(remaining[i] for i in range(len(remaining)) if i not in picked)
            for rest in _assign(tail, index + 1):
                yield [head, *rest]

    xs_by_arm = [float(regressor[name]) for name in names]
    total = 0
    at_least = 0
    best = float("-inf")
    at_best = 0
    for grouping in _assign(tuple(range(len(pooled))), 0):
        xs: list[float] = []
        ys: list[float] = []
        for arm_index, members in enumerate(grouping):
            for member in members:
                xs.append(xs_by_arm[arm_index])
                ys.append(pooled[member])
        total += 1
        statistic = abs(_pearson(xs, ys))
        if statistic >= abs(observed_r) - 1e-12:
            at_least += 1
        if statistic > best + 1e-12:
            best, at_best = statistic, 1
        elif statistic >= best - 1e-12:
            at_best += 1
    return {
        "statistic": "|Pearson r| of per-map DiD on the arm's rows-carrying-loss",
        "arms": names,
        "arm_sizes": sizes,
        "regressor": {name: float(regressor[name]) for name in names},
        "arm_did_means": {
            name: sum(float(v) for v in arm_values[name]) / float(len(arm_values[name]))
            for name in names
        },
        "observed_pearson_r": observed_r,
        "assignments": total,
        "assignments_at_or_above_observed": at_least,
        "p_two_sided": float(at_least) / float(total),
        # |r| is symmetric under reversing the arm order, so the maximum is
        # attained at least twice and the attainable floor is 2/1680, not
        # 1/1680. Enumerated rather than assumed.
        "smallest_attainable_p": float(at_best) / float(total),
        "labellings_at_the_maximum": at_best,
        "resolution_ceiling": 1.0 / float(total),
        "alpha": DISPLACEMENT_ALPHA,
        "can_reject_at_alpha": bool(
            float(at_best) / float(total) <= DISPLACEMENT_ALPHA
        ),
        "note": TREND_TEST_NOTE,
    }


def displacement_verdict(result: Mapping[str, Any]) -> str:
    p_value = float(result["p_one_sided"])
    did_sd = float(result["difference_in_differences_in_exact_sd"])
    if p_value <= DISPLACEMENT_ALPHA:
        return "DISPLACED"
    if abs(did_sd) <= C4_LIKE_MAX_ABS_DID_SD:
        return "c4-LIKE"
    return "INDETERMINATE"


def holm_bonferroni(p_values: Mapping[str, float], *,
                    alpha: float = DISPLACEMENT_ALPHA) -> dict[str, Any]:
    """Holm-Bonferroni over the *new* arms only. Retrospective tests on already
    published bytes are reported uncorrected and labelled as such."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    total = len(ordered)
    decisions: dict[str, Any] = {}
    rejected_so_far = True
    for index, (name, p_value) in enumerate(ordered):
        threshold = alpha / float(total - index)
        reject = bool(rejected_so_far and float(p_value) <= threshold)
        if not reject:
            rejected_so_far = False
        decisions[name] = {
            "p_one_sided": float(p_value),
            "holm_threshold": threshold,
            "reject_at_holm": reject,
        }
    return {"alpha": alpha, "tests": total, "decisions": decisions}


# --------------------------------------------------------------------------- #
# 5. the registered phase-2 trigger (round addendum 2026-08-09)
# --------------------------------------------------------------------------- #
PHASE2_RECALL_TRIGGER = 0.005
PHASE2_CEILING_TRIGGER = 0.005
PHASE2_CEILING_REFERENCE = R0227_STRICT_CEILING_BY_C[16]
PHASE2_TRIGGER_NOTE = (
    "phase 2 runs iff (1) some sweep cell's uniform tie-aware recall over all "
    "2,000,000 rows exceeds q0-baseline's by >= 0.005, or (2) some 100M-feasible "
    "spill cell's strict ceiling over all 2,000,000 rows exceeds the c = 16, "
    "s = 2 strict ceiling 0.953250 by >= 0.005, or (3) some sweep cell exceeds "
    "its own partition's measured strict ceiling, i.e. the instrument is wrong. "
    "0.005 is registered as the smallest recall change worth 0.6 GPU-h of "
    "training: R0228's three arms are separated by 0.018 and 0.020."
)


def phase2_trigger(
    *,
    sweep_cells: Sequence[Mapping[str, Any]],
    spill_cells: Sequence[Mapping[str, Any]],
    partition_strict_ceiling: float,
) -> dict[str, Any]:
    """Evaluate the registered trigger mechanically from phase-1 measurements."""
    baseline = None
    for cell in sweep_cells:
        if str(cell.get("cell")) == BASELINE_CELL:
            baseline = float(cell["tie_aware_recall_all_rows"])
    if baseline is None:
        raise Round0229Error("R0229 phase-2 trigger needs the q0-baseline cell")

    best_recall_cell = max(
        sweep_cells, key=lambda cell: float(cell["tie_aware_recall_all_rows"])
    )
    best_recall = float(best_recall_cell["tie_aware_recall_all_rows"])
    tunable_gain = best_recall - baseline

    feasible = [
        cell for cell in spill_cells
        if bool(cell.get("feasible_at_100m")) and cell.get("strict_ceiling_all_rows")
        is not None
    ]
    best_feasible = (
        max(feasible, key=lambda cell: float(cell["strict_ceiling_all_rows"]))
        if feasible else None
    )
    structural_gain = (
        float(best_feasible["strict_ceiling_all_rows"]) - PHASE2_CEILING_REFERENCE
        if best_feasible is not None else None
    )

    over_ceiling = [
        str(cell["cell"]) for cell in sweep_cells
        if float(cell["tie_aware_recall_all_rows"]) > float(partition_strict_ceiling)
        + 1e-9
    ]

    triggers = {
        "tunable_gain": bool(tunable_gain >= PHASE2_RECALL_TRIGGER),
        "structural_gain": bool(
            structural_gain is not None and structural_gain >= PHASE2_CEILING_TRIGGER
        ),
        "bound_violated": bool(over_ceiling),
    }
    return {
        "note": PHASE2_TRIGGER_NOTE,
        "baseline_tie_aware_recall": baseline,
        "best_sweep_cell": str(best_recall_cell["cell"]),
        "best_sweep_tie_aware_recall": best_recall,
        "tunable_gain": tunable_gain,
        "tunable_gain_threshold": PHASE2_RECALL_TRIGGER,
        "best_feasible_spill_cell": (
            str(best_feasible["cell"]) if best_feasible is not None else None
        ),
        "best_feasible_strict_ceiling": (
            float(best_feasible["strict_ceiling_all_rows"])
            if best_feasible is not None else None
        ),
        "structural_reference_strict_ceiling": PHASE2_CEILING_REFERENCE,
        "structural_gain": structural_gain,
        "structural_gain_threshold": PHASE2_CEILING_TRIGGER,
        "partition_strict_ceiling": float(partition_strict_ceiling),
        "cells_above_their_own_ceiling": over_ceiling,
        "triggers": triggers,
        "phase2_runs": bool(any(triggers.values())),
    }


# --------------------------------------------------------------------------- #
# 6. projections — labelled, ranged, and never a projection over a projection
# --------------------------------------------------------------------------- #
#: Review-0226-01 measured gsv:/data cold sequential read at 5.53 and 6.36 GB/s.
#: Review-0227-01 found R0227's spill phase actually ran at 2.52-4.30 GB/s on
#: largely warm reads that 100M's 153.6 GB per pass cannot be, and that charging
#: 5.53 GB/s was therefore optimistic rather than conservative. Both are carried.
SPILL_IO_RATES_BYTES_PER_S = (5.53e9, 6.36e9)
SPILL_IO_MEASURED_RATES_BYTES_PER_S = (2.52e9, 4.30e9)
SPILL_IO_NOTE = (
    "spill I/O is modelled explicitly and reported as its own line, never folded "
    "into a compute fit: s x N x 384 x 4 bytes written and read back. The "
    "5.53-6.36 GB/s band is review-0226-01's cold sequential measurement; the "
    "2.52-4.30 GB/s band is what R0227's own spill phase achieved on largely "
    "warm reads, and review-0227-01 established that the warm advantage does not "
    "survive to 100M, where one pass is 153.6 GB against 123 GiB of RAM"
)


def spill_io_seconds(*, rows: int, spill: int, dimension: int = DIMENSION,
                     rate_bytes_per_s: float) -> dict[str, Any]:
    payload = float(spill) * float(rows) * float(dimension) * 4.0
    return {
        "spill_write_bytes": payload,
        "spill_read_back_bytes": payload,
        "total_bytes": 2.0 * payload,
        "rate_bytes_per_s": float(rate_bytes_per_s),
        "seconds": 2.0 * payload / float(rate_bytes_per_s),
    }


def power_fit(sizes: Sequence[float], seconds: Sequence[float]) -> dict[str, Any]:
    """`t = a * m^b` by least squares in log space, with the fitted range kept."""
    points = [
        (float(size), float(second))
        for size, second in zip(sizes, seconds)
        if float(size) > 0.0 and float(second) > 0.0
    ]
    if len(points) < 3:
        raise Round0229Error("R0229 power fit needs at least three positive points")
    xs = [math.log(size) for size, _ in points]
    ys = [math.log(second) for _, second in points]
    n = float(len(points))
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0.0:
        raise Round0229Error("R0229 power fit has no spread in cluster rows")
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    exponent = sxy / sxx
    intercept = mean_y - exponent * mean_x
    predicted = [intercept + exponent * x for x in xs]
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, predicted))
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    return {
        "model": "t = a * max_cluster_rows^b, least squares in log space",
        "coefficient_a": math.exp(intercept),
        "exponent_b": exponent,
        "r_squared": (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n_points": len(points),
        "fitted_range_cluster_rows": [
            min(size for size, _ in points), max(size for size, _ in points)
        ],
    }


def project_from_power_fit(fit: Mapping[str, Any], cluster_rows: float) -> dict[str, Any]:
    lo, hi = (float(value) for value in fit["fitted_range_cluster_rows"])
    return {
        "cluster_rows": float(cluster_rows),
        "seconds": float(fit["coefficient_a"]) * float(cluster_rows) ** float(
            fit["exponent_b"]
        ),
        "fitted_range_cluster_rows": [lo, hi],
        "extrapolation_factor_beyond_fitted_max": float(cluster_rows) / hi,
        "is_extrapolation": bool(float(cluster_rows) > hi),
        "label": "PROJECTION",
    }


def guard_for_spill(*, rows: int, clusters: int, spill: int) -> dict[str, Any]:
    """R0227's predictive guard, with the prediction scaled for `s != 2`.

    `predict_footprint` assumes `A_SPILL = 2` throughout, so a cell with more
    spill has proportionally more rows in its largest cluster and the device term
    must be scaled before the budget is applied. Scaling `rows` by `s / 2` is
    exact for this purpose because every term the guard computes is a function of
    `s x N / c`, and it keeps R0227's registered budgets and refusal semantics
    untouched.
    """
    if spill == A_SPILL:
        scaled_rows = int(rows)
    else:
        scaled_rows = int(math.ceil(float(rows) * float(spill) / float(A_SPILL)))
    decision = dict(guard_decision(rows=scaled_rows, clusters=clusters))
    decision["spill"] = int(spill)
    decision["guard_rows_actual"] = int(rows)
    decision["guard_rows_scaled_for_spill"] = scaled_rows
    decision["spill_scaling_note"] = (
        "R0227's guard assumes A_SPILL = 2; every term it computes is a function "
        f"of s x N / c, so rows are scaled by s / 2 = {spill / A_SPILL} before "
        "the registered device and host budgets are applied. Budgets, refusal "
        "semantics and refused_a_priori accounting are R0227's, unchanged."
    )
    return decision


__all__ = [
    "ADOPTION_CLAIMED",
    "BASELINE_CELL",
    "BUILD_SCHEMA",
    "C4_LIKE_MAX_ABS_DID_SD",
    "C4_UNREACHABLE_MARGIN_AT_C16",
    "CONTROL_CEILING_TOLERANCE",
    "DECISION_RULE_NOTE",
    "DIMENSION",
    "DISPLACEMENT_ALPHA",
    "DISPLACEMENT_TEST_NOTE",
    "EQUIVALENCE_CLAIMED",
    "GATE_REGISTERABLE_HERE",
    "GATE_RELEASE_CLAIMED",
    "NND_HEADROOM_AT_C16",
    "PERMUTATION_LABELLINGS",
    "PERMUTATION_RESOLUTION_CEILING",
    "PHASE2_CEILING_REFERENCE",
    "PHASE2_CEILING_TRIGGER",
    "PHASE2_RECALL_TRIGGER",
    "PHASE2_RUNGS",
    "PHASE2_TRIGGER_NOTE",
    "QUALITY_SWEEP",
    "R0227_MEASURED_IMBALANCE",
    "R0227_STRICT_CEILING_BY_C",
    "R0227_TIE_CEILING_BY_C",
    "R0228_DID_IN_EXACT_SD_BY_C",
    "R0228_ROWS_CARRYING_LOSS_BY_C",
    "R0228_TIE_AWARE_RECALL_BY_C",
    "RESOLUTION_RULE_NOTE",
    "RETROSPECTIVE_LABEL",
    "REVIEW_0228_DISPLACEMENT_P_BY_C",
    "REVIEW_0228_P_TOLERANCE",
    "TREND_ASSIGNMENTS",
    "TREND_RESOLUTION_CEILING",
    "TREND_TEST_NOTE",
    "exact_did_trend",
    "test_can_reject",
    "verify_r0228_displacement",
    "REACHABILITY_SCHEMA",
    "RECALL_POPULATION",
    "RECALL_POPULATION_NOTE",
    "RETRO_CAPABILITY",
    "RETRO_SCHEMA",
    "ROUND_ID",
    "ROWS",
    "Round0229Error",
    "SPILL_CAPABILITY",
    "SPILL_CONTROL_CELLS",
    "SPILL_GRID",
    "SPILL_IO_MEASURED_RATES_BYTES_PER_S",
    "SPILL_IO_NOTE",
    "SPILL_IO_RATES_BYTES_PER_S",
    "SPILL_SCHEMA",
    "STRUCTURAL_BOUND_NOTE",
    "SWEEP_CAPABILITY",
    "SWEEP_CLUSTERS",
    "SWEEP_SCHEMA",
    "SWEEP_SPILL",
    "TIE_QUERY_ROWS",
    "TIE_QUERY_SEED",
    "TRAINING_PERFORMED",
    "displacement_verdict",
    "exact_displacement_permutation",
    "family_mean_cluster_rows",
    "guard_for_spill",
    "holm_bonferroni",
    "phase2_trigger",
    "power_fit",
    "project_from_power_fit",
    "projected_max_cluster_rows",
    "rung_is_feasible",
    "smallest_measured_clusters",
    "spill_io_seconds",
    "project_from_power_fit",
    "verify_r0227_ceilings",
]
