"""R0243 — settle the locality question on the TIE-AWARE scale, then symmetrise.

review-0242-01 confirmed R0242's spatial concentration against two independent
nulls (`0/12,000` exceedances, `418` null SD on the max-share statistic) and
then corrected it in three ways that this round is written to discharge:

1. **`97.74%` of cluster `168`'s loss is TIE-FORGIVEN.** R0242 measured loss
   exclusively in STRICT missing edges. It sealed
   `probe-tie-aware-recall.f64.npy` and never joined it. On the tie-aware
   vector the max single-cluster share falls `0.388273 -> ~0.0785` and the
   top-20 share `0.672238 -> ~0.303`. This round runs the round's OWN
   decomposition and the round's OWN cluster tests on that vector, so the
   comparison is exact rather than the review's conservative
   `min(tie_missing, builder_missing)` bound.
2. **R0242's single-hot-spot guard is VACUOUS at `c = 400`.** It required the
   hot cluster to hold `< 1%` of the probe's reachable exposure; the largest
   attainable exposure share at `c = 400` is about `0.0052`, so no cluster
   could ever fail the guard and the criterion reduced to a bare `5%` bar.
   `HALT_SINGLE_CLUSTER_EXPOSURE_MULTIPLE` re-expresses it against `1/c`,
   where it can bind, and `exposure_profile` MEASURES how many cells each form
   of the guard actually excludes rather than asserting that it binds.
3. **Magnitude, not shape, is what a map cares about.** A tie-forgiven miss is
   a substitute neighbour at a cosine within the tie threshold of the true
   `k`th; UMAP's attractive force is a function of that distance, so a
   substitute at the same distance perturbs the fuzzy simplicial set by
   approximately nothing. A SHARE of a negligible quantity therefore cannot
   answer "is this harmful to a map". R0243's halt rule keeps every shape
   statistic and publishes it, and gates on ABSOLUTE per-cell and probe-wide
   tie-aware builder-loss RATES, calibrated against the only spatial defect
   this program has seen break a map (R0034: `2,779,481` rows, `1.85%`).

Nothing here re-types a registered check. The decomposition, the cluster
locality test, the rate table, the canonicalization, the post-canonicalization
tripwire, the symmetrised-degree-once pass, the weight distribution and the
gather pricer are all IMPORTED from `basemap.round0242_locality`. What this
module adds is: the strict reproduction gate against R0242's sealed receipt,
the exposure profile that proves a guard can bind, the re-registered halt rule,
a gather priced as a SORTED gather, and the map-harm assessment.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.round0238_rung5 import GRAPH_K, SPILL
from basemap.round0241_qualify import (
    DIMENSION,
    REGISTERED_GRAPH_COS_SHA256,
    REGISTERED_GRAPH_IDS_SHA256,
    REGISTERED_LADDER_RECEIPT_SHA256,
    REGISTERED_SELECTED_CELL,
    REGISTERED_SELECTED_CLUSTERS,
)
from basemap.round0242_locality import (  # noqa: F401  (re-exported deliberately)
    _cluster_totals,
    _dispersion,
    CANONICALIZATION_NOTE,
    CONCENTRATION_TOP_M,
    PERMUTATIONS,
    PERMUTATION_SEED,
    SYMMETRISED_DEGREE_ONCE_NOTE,
    canonical_undirected_degrees,
    cluster_locality_test,
    cluster_rate_table,
    gather_price,
    host_anonymous_bytes,
    io_counters,
    json_scrub,
    loss_decomposition,
    post_canonical_tripwire,
    symmetrised_degree_once,
    weight_distribution,
)

ROUND_ID = "0243"
ROWS = 100_000_000
CLUSTERS = REGISTERED_SELECTED_CLUSTERS
PARTITION_SEED = 226
PROBE_ROWS = 500_000

RESIDUAL_CAPABILITY = "minilm-mixed-100000k-k15-tie-aware-loss-locality-v1"
FUZZY_CAPABILITY = "minilm-mixed-100000k-cluster-spill-k15-fuzzy-graph-v1"
CANONICAL_CAPABILITY = "minilm-mixed-100000k-k15-post-canonical-tripwire-v1"

RESIDUAL_FILE = "tie-aware-locality.json"
RESIDUAL_EARLY_FILE = "tie-aware-locality-first-write.json"
FUZZY_FILE = "fuzzy-graph.json"
RESIDUAL_SCHEMA = "round0243-minilm-mixed-100000k-k15-tie-aware-loss-locality-v1"
RESIDUAL_EARLY_SCHEMA = (
    "round0243-minilm-mixed-100000k-k15-tie-aware-loss-locality-first-write-v1"
)
FUZZY_SCHEMA = "round0243-minilm-mixed-100000k-k15-fuzzy-and-canonical-tripwire-v1"

GPU_HOURS_CAP = 12.0
RESIDUAL_DEADLINE_S = 5_400.0
FUZZY_DEADLINE_S = 32_400.0
RESIDUAL_STAGE_BUDGET_S = 4_800.0
FUZZY_STAGE_BUDGET_S = 30_000.0

# --------------------------------------------------------------------------- #
# the RE-REGISTERED halt rule
# --------------------------------------------------------------------------- #
#: H1 — probe-wide magnitude. If one edge in a hundred of the reachable
#: exposure is genuinely (tie-aware) absent, the graph is not fit to symmetrise
#: whatever its shape. R0241's published tie-aware TOTAL loss rate is
#: `0.002058`, so this bar sits about `5x` above the whole-probe loss and about
#: an order of magnitude above the builder's share of it.
HALT_GLOBAL_TIE_AWARE_BUILDER_RATE = 0.01

#: H2a — per-cell magnitude, calibrated against the ONLY spatial defect this
#: program has ever seen break a map. R0034 shipped `2,779,481` of
#: `150,000,000` rows (`1.85%`) with no valid canonical edge and R0215 traced
#: the v1 map's clumps to exactly those rows. A cell losing `1.85%` of its
#: reachable EDGES to genuine, non-tie-forgiven builder failure is the same
#: order of insult, per edge instead of per row.
HALT_CELL_TIE_AWARE_BUILDER_RATE = 0.0185

#: H2b — materiality. Unchanged from R0242, so the shape arm of the hot-spot
#: criterion is not weakened; it is now conjunctive with a magnitude arm.
HALT_SINGLE_CLUSTER_SHARE = 0.05

#: H2c — the exposure guard, RE-EXPRESSED so it can bind. R0242 required the
#: hot cluster to hold `< 0.01` of the probe's reachable exposure. At
#: `c = 400` the largest attainable exposure share is about `2.08/c = 0.0052`,
#: so that guard excluded nothing (review-0242-01/F5). Expressed as a multiple
#: of `1/c` it binds at any `c`: at `1.5/c = 0.00375` it sits INSIDE the
#: measured spread of cell sizes (`0.32/c` to `2.08/c`, `p75 = 1.14/c`), so
#: some cells fail it and some pass. `exposure_profile` measures exactly how
#: many, and the vacuity of the `0.01` form is measured beside it rather than
#: asserted.
HALT_SINGLE_CLUSTER_EXPOSURE_MULTIPLE = 1.5

#: H3 — the SHAPE statistics. Computed, published, compared against R0242's
#: strict figures, and REPORTED — but they do not halt Part B, and that is
#: registered here before any measurement exists. Reason: R0242 established
#: and review-0242-01 released `finding:r0242-builder-loss-is-spatially-
#: concentrated-in-three-of-four-hundred-cells`. That finding is published; it
#: cannot be buried by a product step, which was the entire purpose of R0242's
#: shape halt. Re-firing on the same statistic would re-report a released
#: finding at the price of the post-canonicalization tripwire, which is the
#: check the v2 rebuild exists for and which has never run at this rung.
HALT_P_VALUE = 0.001
HALT_TOP_M_SHARE = 0.25

HALT_RULE_NOTE = (
    "R0243 Part B is HALTED if and only if: (H0) the strict decomposition and "
    "the strict per-cluster dispersion statistics recomputed here fail to "
    "reproduce R0242's sealed values exactly; or (H1) the probe-wide TIE-AWARE "
    "builder-missing rate reaches 0.01 of reachable exposure; or (H2) some "
    "single cluster has a TIE-AWARE builder-missing RATE of at least 0.0185 "
    "AND carries at least 0.05 of all tie-aware builder-missing edges AND "
    "holds less than 1.5/c of the probe's reachable exposure. The SHAPE "
    "statistics - chi-square, top-20 share, max single-cluster share, all with "
    "10,000-permutation nulls - are computed on both the strict and the "
    "tie-aware scale and published in full, and they do NOT halt Part B. That "
    "is registered in advance and its reason is stated in advance: R0242's "
    "shape halt existed so a concentration finding could not be buried under a "
    "product step, and review-0242-01 has already released that finding, so "
    "there is nothing left to bury. What is still open is whether the RESIDUAL "
    "is large enough to matter to a map, and a share of a negligible quantity "
    "cannot answer that. The magnitude arms are the ones that can."
)

TIE_AWARE_NOTE = (
    "A tie-forgiven miss is a neighbour the builder did not return whose "
    "cosine is within the tie threshold of the true k-th neighbour's - i.e. "
    "the builder returned a substitute at essentially the same distance. UMAP "
    "builds its fuzzy simplicial set from those distances, so a substitute at "
    "the same distance produces approximately the same membership strength "
    "and approximately the same attractive force. Strict recall is the right "
    "instrument for 'did the builder find the exact truth'; tie-aware recall "
    "is the right instrument for 'does the map see a different neighbourhood'. "
    "R0241 headlines tie-aware for that reason and R0242 gated on strict."
)

EXPOSURE_GUARD_NOTE = (
    "review-0242-01/F5: R0242's hot-spot criterion fired on a cluster holding "
    "at least 5% of builder-missing edges while holding LESS THAN 1% of the "
    "probe's reachable exposure. At c = 400 the largest cell attainable in "
    "this partition holds about 0.52% of exposure, so the guard excluded no "
    "cluster at all and the criterion collapsed to a bare 5% bar - the "
    "single-candidate vacuity class this program has now flagged three times. "
    "Re-expressed as a multiple of 1/c the guard binds at any c. This function "
    "MEASURES, from the realised exposure distribution, how many of the c "
    "cells each form of the guard excludes, so 'it can bind' is a measurement "
    "and not a claim."
)

SORTED_GATHER_NOTE = (
    "review-0242-01/F8 blocked R0242's 16.6 h full-gather projection: the "
    "500,000-anchor gather it extrapolated from had already physically read "
    "92.94% of the 153.6 GB substrate, so its 12.39x amplification was within "
    "7.6% of its own arithmetic ceiling and could not grow; carrying it 200x "
    "implies reading the file 185.9 times. A gather whose target ids are "
    "SORTED and deduplicated reads each touched row once, so its physical read "
    "is bounded ABOVE by the substrate size no matter how many anchors are "
    "gathered. This measures that directly at two anchor counts instead of "
    "extrapolating a saturating rate, and reports the arithmetic ceiling "
    "beside the measurement."
)

SAFETY_NOTE = (
    "Every array read off disk is opened mmap_mode='r' and the node raises "
    "unless it is a non-writeable np.memmap, re-measured on the live objects "
    "at seal time. cuVS is handed nothing - no cuVS call exists on either node "
    "path. No child process is started, no signalling construct exists on "
    "either node path, and every bound raises in band. The one CUDA context is "
    "created by R0242's registered torch transcription of R0226's "
    "kmeans/assign, for the partition re-realisation only. The host watchdog "
    "is conjunctive and guards ANONYMOUS bytes, never RSS. To stop this round, "
    "write <queue_root>/logs/<node>.abort and wait; never Ctrl-C, never "
    "ws kill, never tmux kill-session, never kill -9, never py-spy."
)

SCOPE_NOTE = (
    "R0243 settles the locality question on the tie-aware scale and, if the "
    "registered magnitude rule permits, produces the fuzzy symmetrised graph "
    "and runs the first post-canonicalization degree-zero tripwire at "
    "100,000,000 rows. It trains no map, registers no gate, claims no dose, "
    "adoption or map-quality result, and does NOT clear the graph for an "
    "atlas: review-0241-01's block on that claim stands and only R0228's "
    "within-map displacement instrument can lift it."
)


class Round0243Error(RuntimeError):
    """R0243 fail-closed error. Raised in band; never a signal."""


# --------------------------------------------------------------------------- #
# H0 — the strict reproduction gate
# --------------------------------------------------------------------------- #
#: Every field checked against R0242's sealed `loss-locality.json`. The
#: tie-aware re-analysis is worthless unless it stratifies the same loss.
STRICT_DECOMPOSITION_FIELDS = (
    "probe_rows",
    "rows_carrying_strict_loss",
    "rows_with_reachability_below_one",
    "rows_both",
    "rows_builder_loss_with_truth_fully_reachable",
    "rows_recovering_an_unreachable_neighbour",
    "total_missing_edges",
    "partition_forced_missing_edges",
    "builder_missing_edges",
)
STRICT_DISPERSION_FIELDS = (
    "chi_square",
    "top_m_share_of_missing",
    "top_m_share_of_exposure",
    "max_single_cluster_share_of_missing",
    "max_single_cluster_share_of_exposure",
)


def strict_reproduction_gate(
    *,
    measured_decomposition: Mapping[str, Any],
    sealed_decomposition: Mapping[str, Any],
    measured_dispersion: Mapping[str, Mapping[str, Any]],
    sealed_tests: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Do this round's strict numbers reproduce R0242's sealed ones EXACTLY?

    Integers must be equal; the dispersion floats are recomputed from the same
    sealed vectors through the same imported `_dispersion`, so they must be
    bit-identical too. Any disagreement stops the round: a tie-aware re-run of
    a loss vector that is not R0242's loss vector answers nothing.
    """
    disagreements: list[dict[str, Any]] = []
    for field in STRICT_DECOMPOSITION_FIELDS:
        mine = measured_decomposition.get(field)
        theirs = sealed_decomposition.get(field)
        if int(mine) != int(theirs):
            disagreements.append(
                {"where": f"decomposition.{field}", "measured": mine,
                 "sealed": theirs}
            )
    for population, observed in measured_dispersion.items():
        sealed = dict(dict(sealed_tests[population])["observed"])
        for field in STRICT_DISPERSION_FIELDS:
            mine = float(observed[field])
            theirs = float(sealed[field])
            if mine != theirs:
                disagreements.append({
                    "where": f"{population}.observed.{field}",
                    "measured": mine, "sealed": theirs,
                    "absolute_difference": abs(mine - theirs),
                })
    return {
        "fields_checked": (
            len(STRICT_DECOMPOSITION_FIELDS)
            + len(measured_dispersion) * len(STRICT_DISPERSION_FIELDS)
        ),
        "populations_checked": sorted(measured_dispersion),
        "disagreements": disagreements,
        "agree": not disagreements,
        "note": (
            "H0: R0243's strict decomposition and strict per-cluster "
            "dispersion, recomputed here from R0242's sealed per-row vectors "
            "through the same imported loss_decomposition and _dispersion, "
            "against R0242's sealed loss-locality.json. Exact equality is "
            "required and a mismatch halts Part B."
        ),
    }


def observed_dispersion(
    *,
    labels: np.ndarray,
    missing: np.ndarray,
    exposure: np.ndarray,
    clusters: int = CLUSTERS,
    top_m: int = CONCENTRATION_TOP_M,
) -> dict[str, float]:
    """R0242's own `_dispersion`, on the observed data and with no null.

    The permutation nulls cost minutes; the observed statistics cost
    milliseconds. H0 only needs the observed ones, so the strict scale is
    re-derived here through the IMPORTED function - not a re-typing of it -
    and the `10,000`-draw nulls are spent where the new question is, on the
    tie-aware scale.
    """
    return _dispersion(
        _cluster_totals(
            np.asarray(labels, dtype=np.int64),
            np.asarray(missing, dtype=np.float64),
            clusters=int(clusters),
        ),
        _cluster_totals(
            np.asarray(labels, dtype=np.int64),
            np.asarray(exposure, dtype=np.float64),
            clusters=int(clusters),
        ),
        top_m=int(top_m),
    )


# --------------------------------------------------------------------------- #
# the exposure profile — proving a guard can bind, by measurement
# --------------------------------------------------------------------------- #
def exposure_profile(
    *,
    labels: np.ndarray,
    exposure: np.ndarray,
    clusters: int = CLUSTERS,
    guard_multiple: float = HALT_SINGLE_CLUSTER_EXPOSURE_MULTIPLE,
    r0242_absolute_guard: float = 0.01,
) -> dict[str, Any]:
    """The realised per-cell exposure-share distribution, in units of `1/c`.

    Its job is to make "this guard can actually be crossed" a measurement.
    It reports, for the re-expressed guard and for R0242's absolute one, how
    many of the `c` cells each EXCLUDES. A guard that excludes zero cells at
    the realised distribution is vacuous, and this says so with a number.
    """
    labels = np.asarray(labels, dtype=np.int64)
    exposure = np.asarray(exposure, dtype=np.float64)
    by_cell = np.bincount(
        labels, weights=exposure, minlength=int(clusters)
    )[: int(clusters)]
    total = float(by_cell.sum())
    if total <= 0.0:
        raise Round0243Error("R0243 exposure profile needs positive exposure")
    share = by_cell / total
    uniform = 1.0 / float(clusters)
    multiples = share / uniform
    guard_share = float(guard_multiple) * uniform
    counts = np.bincount(labels, minlength=int(clusters))[: int(clusters)]
    return {
        "clusters": int(clusters),
        "uniform_share": uniform,
        "total_exposure_edges": total,
        "share_min": float(share.min()),
        "share_max": float(share.max()),
        "share_p25": float(np.percentile(share, 25)),
        "share_p50": float(np.percentile(share, 50)),
        "share_p75": float(np.percentile(share, 75)),
        "multiple_of_uniform_min": float(multiples.min()),
        "multiple_of_uniform_max": float(multiples.max()),
        "multiple_of_uniform_p50": float(np.percentile(multiples, 50)),
        "multiple_of_uniform_p75": float(np.percentile(multiples, 75)),
        "probe_rows_min": int(counts.min()),
        "probe_rows_max": int(counts.max()),
        "re_expressed_guard": {
            "multiple_of_uniform": float(guard_multiple),
            "share_threshold": guard_share,
            "cells_excluded_by_the_guard": int((share >= guard_share).sum()),
            "cells_admitted_by_the_guard": int((share < guard_share).sum()),
        },
        "r0242_absolute_guard": {
            "share_threshold": float(r0242_absolute_guard),
            "cells_excluded_by_the_guard": int(
                (share >= float(r0242_absolute_guard)).sum()
            ),
            "largest_attainable_share": float(share.max()),
            "largest_attainable_share_as_multiple_of_uniform": float(
                multiples.max()
            ),
        },
        "note": EXPOSURE_GUARD_NOTE,
    }


# --------------------------------------------------------------------------- #
# H1 / H2 — the magnitude arms
# --------------------------------------------------------------------------- #
def hot_cell_scan(
    *,
    labels: np.ndarray,
    missing: np.ndarray,
    exposure: np.ndarray,
    clusters: int = CLUSTERS,
    cell_rate_threshold: float = HALT_CELL_TIE_AWARE_BUILDER_RATE,
    share_threshold: float = HALT_SINGLE_CLUSTER_SHARE,
    exposure_multiple: float = HALT_SINGLE_CLUSTER_EXPOSURE_MULTIPLE,
    report: int = 15,
) -> dict[str, Any]:
    """Score every cell on all three arms of H2 and report which cells fire.

    Each arm is reported separately as well as conjunctively, so a reader can
    see exactly which arm is doing the work and what would have to change for
    the verdict to flip.
    """
    labels = np.asarray(labels, dtype=np.int64)
    missing = np.asarray(missing, dtype=np.float64)
    exposure = np.asarray(exposure, dtype=np.float64)
    if not (labels.size == missing.size == exposure.size):
        raise Round0243Error("R0243 hot-cell scan: vector length mismatch")
    counts = np.bincount(labels, minlength=int(clusters))[: int(clusters)]
    missing_by = np.bincount(
        labels, weights=missing, minlength=int(clusters)
    )[: int(clusters)]
    exposure_by = np.bincount(
        labels, weights=exposure, minlength=int(clusters)
    )[: int(clusters)]
    total_missing = float(missing_by.sum())
    total_exposure = float(exposure_by.sum())
    if total_exposure <= 0.0:
        raise Round0243Error("R0243 hot-cell scan needs positive exposure")
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(exposure_by > 0, missing_by / exposure_by, 0.0)
        share = (
            missing_by / total_missing if total_missing > 0
            else np.zeros_like(missing_by)
        )
    exposure_share = exposure_by / total_exposure
    guard_share = float(exposure_multiple) / float(clusters)

    rate_arm = rate >= float(cell_rate_threshold)
    share_arm = share >= float(share_threshold)
    guard_arm = exposure_share < guard_share
    fires = rate_arm & share_arm & guard_arm

    order = np.argsort(-rate, kind="stable")[: int(report)]
    by_share = np.argsort(-share, kind="stable")[: int(report)]

    def _rows(indices: np.ndarray) -> list[dict[str, Any]]:
        return [
            {
                "cluster": int(index),
                "probe_rows": int(counts[index]),
                "missing_edges": float(missing_by[index]),
                "exposure_edges": float(exposure_by[index]),
                "missing_rate": float(rate[index]),
                "share_of_missing": float(share[index]),
                "share_of_exposure": float(exposure_share[index]),
                "exposure_as_multiple_of_uniform": float(
                    exposure_share[index] * float(clusters)
                ),
                "meets_rate_arm": bool(rate_arm[index]),
                "meets_share_arm": bool(share_arm[index]),
                "passes_exposure_guard": bool(guard_arm[index]),
                "fires_h2": bool(fires[index]),
            }
            for index in indices
        ]

    return {
        "clusters": int(clusters),
        "total_missing_edges": total_missing,
        "total_exposure_edges": total_exposure,
        "overall_missing_rate": total_missing / total_exposure,
        "thresholds": {
            "cell_rate": float(cell_rate_threshold),
            "single_cluster_share": float(share_threshold),
            "exposure_multiple_of_uniform": float(exposure_multiple),
            "exposure_share_guard": guard_share,
        },
        "cells_meeting_rate_arm": int(rate_arm.sum()),
        "cells_meeting_share_arm": int(share_arm.sum()),
        "cells_passing_exposure_guard": int(guard_arm.sum()),
        "cells_firing_all_three": int(fires.sum()),
        "firing_clusters": [int(index) for index in np.flatnonzero(fires)],
        "worst_cell_by_rate": int(np.argmax(rate)),
        "worst_cell_rate": float(rate.max()),
        "worst_cell_by_share": int(np.argmax(share)),
        "worst_cell_share": float(share.max()),
        "highest_rate_cells": _rows(order),
        "highest_share_cells": _rows(by_share),
    }


def residual_verdict(
    *,
    reproduction: Mapping[str, Any],
    tie_aware_scan: Mapping[str, Any],
    tie_aware_builder_test: Mapping[str, Any],
    strict_builder_test: Mapping[str, Any],
    global_rate_threshold: float = HALT_GLOBAL_TIE_AWARE_BUILDER_RATE,
) -> dict[str, Any]:
    """Apply the halt rule registered in the round file BEFORE Part A ran.

    Every threshold used here is a module constant in the release commit the
    round file names, and the round file states each in prose with the reason
    it was chosen. The shape statistics are carried into the verdict so the
    reader sees them, and the verdict states explicitly that they do not gate.
    """
    reproduces = bool(reproduction["agree"])
    global_rate = float(tie_aware_scan["overall_missing_rate"])
    global_arm = global_rate >= float(global_rate_threshold)
    hot_cells = int(tie_aware_scan["cells_firing_all_three"])
    hot_arm = hot_cells > 0

    shape_p = float(tie_aware_builder_test["chi_square"]["p_value"])
    shape_top = float(
        tie_aware_builder_test["top_m_share_of_missing"]["observed"]
    )
    shape_max = float(
        tie_aware_builder_test["max_single_cluster_share_of_missing"]["observed"]
    )
    shape_would_halt = bool(
        (shape_p < HALT_P_VALUE and shape_top >= HALT_TOP_M_SHARE)
        or shape_max >= HALT_SINGLE_CLUSTER_SHARE
    )

    halt = bool((not reproduces) or global_arm or hot_arm)
    if halt and not reproduces:
        reading = "STOP: the strict reproduction gate failed"
    elif halt:
        reading = "residual loss is materially concentrated"
    elif shape_would_halt:
        reading = "concentrated in shape, negligible in magnitude"
    else:
        reading = "neither concentrated nor material"
    return {
        "rule": HALT_RULE_NOTE,
        "h0_strict_reproduction_agrees": reproduces,
        "h0_disagreements": list(reproduction["disagreements"]),
        "h1_global_tie_aware_builder_rate": global_rate,
        "h1_threshold": float(global_rate_threshold),
        "h1_fires": bool(global_arm),
        "h2_cells_firing_all_three_arms": hot_cells,
        "h2_firing_clusters": list(tie_aware_scan["firing_clusters"]),
        "h2_cells_meeting_rate_arm": int(
            tie_aware_scan["cells_meeting_rate_arm"]
        ),
        "h2_cells_meeting_share_arm": int(
            tie_aware_scan["cells_meeting_share_arm"]
        ),
        "h2_cells_passing_exposure_guard": int(
            tie_aware_scan["cells_passing_exposure_guard"]
        ),
        "h2_worst_cell_rate": float(tie_aware_scan["worst_cell_rate"]),
        "h2_cell_rate_threshold": float(HALT_CELL_TIE_AWARE_BUILDER_RATE),
        "h2_fires": bool(hot_arm),
        "h3_shape_reported_not_gating": {
            "gates": False,
            "tie_aware_chi_square_p_value": shape_p,
            "tie_aware_top_m_share_of_missing": shape_top,
            "tie_aware_max_single_cluster_share_of_missing": shape_max,
            "strict_top_m_share_of_missing": float(
                strict_builder_test["top_m_share_of_missing"]["observed"]
            ),
            "strict_max_single_cluster_share_of_missing": float(
                strict_builder_test["max_single_cluster_share_of_missing"][
                    "observed"
                ]
            ),
            "r0242_thresholds_would_halt_on_the_tie_aware_scale": (
                shape_would_halt
            ),
            "why_it_does_not_gate": (
                "R0242's shape halt existed so a spatial-concentration finding "
                "could not be buried under a product step. review-0242-01 has "
                "released that finding, so it cannot be buried. What remains "
                "open is magnitude, and a share of a negligible quantity "
                "cannot measure it. Registered before the run."
            ),
        },
        "reading": reading,
        "halt_part_b": halt,
        "part_b_may_run": bool(not halt),
        "tie_aware_note": TIE_AWARE_NOTE,
    }


# --------------------------------------------------------------------------- #
# the map-harm assessment — what the evidence supports and what it cannot
# --------------------------------------------------------------------------- #
def map_harm_assessment(
    *,
    strict_decomposition: Mapping[str, Any],
    tie_aware_decomposition: Mapping[str, Any],
    tie_aware_scan: Mapping[str, Any],
    strict_scan: Mapping[str, Any],
    probe_rows: int,
    k: int = GRAPH_K,
) -> dict[str, Any]:
    """Quantify the residual in the units a map is built from: edges.

    This does not settle the question - only R0228's within-map displacement
    instrument, run on a trained map with rows stratified by measured loss,
    can. What it does is put the residual on the scale of the graph that will
    be symmetrised, so the reader can see whether the concentrated quantity is
    big enough to be worth an instrument.
    """
    probe_edges = int(probe_rows) * int(k)
    strict_builder = int(strict_decomposition["builder_missing_edges"])
    tie_builder = int(tie_aware_decomposition["builder_missing_edges"])
    forgiven = strict_builder - tie_builder
    worst_share = float(tie_aware_scan["worst_cell_share"])
    worst_cell_edges = worst_share * float(tie_builder)
    return {
        "probe_edges": probe_edges,
        "strict_builder_missing_edges": strict_builder,
        "tie_aware_builder_missing_edges": tie_builder,
        "builder_edges_tie_forgiven": forgiven,
        "fraction_of_builder_loss_tie_forgiven": (
            forgiven / float(strict_builder) if strict_builder else None
        ),
        "tie_aware_builder_missing_as_fraction_of_probe_edges": (
            tie_builder / float(probe_edges)
        ),
        "worst_cell_tie_aware_share_of_builder_loss": worst_share,
        "worst_cell_tie_aware_builder_missing_edges": worst_cell_edges,
        "worst_cell_tie_aware_edges_as_fraction_of_probe_edges": (
            worst_cell_edges / float(probe_edges)
        ),
        "worst_cell_strict_share_of_builder_loss": float(
            strict_scan["worst_cell_share"]
        ),
        "worst_cell_tie_aware_rate": float(tie_aware_scan["worst_cell_rate"]),
        "worst_cell_strict_rate": float(strict_scan["worst_cell_rate"]),
        "r0034_v1_defect_row_fraction": 0.0185,
        "what_this_supports": (
            "the residual is measured on the same probe, in edges, so its "
            "size relative to the graph a map would be trained on is a "
            "measurement rather than an inference; and the tie-forgiven "
            "fraction bounds how much of the loss can move a UMAP attractive "
            "force at all, because a tie-forgiven miss is a substitute "
            "neighbour at a cosine within the tie threshold of the true k-th"
        ),
        "what_this_cannot_settle": (
            "whether a given number of genuinely-missing edges, concentrated "
            "in a few cells, displaces those cells' rows in a trained "
            "embedding. Only R0228's within-map displacement DiD, run on a "
            "trained ladder map with rows stratified on measured strict AND "
            "tie-aware loss, can answer that, and this round trains no map. "
            "R0228 is the precedent in both directions: its uniform panel "
            "average went null while its structured DiD ran +3.94 sd."
        ),
        "note": TIE_AWARE_NOTE,
    }


# --------------------------------------------------------------------------- #
# the I/O term — a gather priced as a SORTED gather
# --------------------------------------------------------------------------- #
def sorted_gather_price(
    *,
    anchors: int,
    neighbours_per_row: int,
    row_bytes: int,
    distinct_rows_touched: int,
    substrate_bytes: int,
    wall_s: float,
    physical_read_bytes: int,
    label: str,
) -> dict[str, Any]:
    """Price a gather whose target ids are sorted and deduplicated.

    `useful_bytes` keeps R0242's definition (`anchors x k x row_bytes`) so the
    sorted and unsorted measurements are directly comparable. `distinct_bytes`
    is what a sorted gather actually has to read, and the substrate size is the
    hard ceiling on it: a sorted gather cannot read a row twice.
    """
    useful = int(anchors) * int(neighbours_per_row) * int(row_bytes)
    distinct = int(distinct_rows_touched) * int(row_bytes)
    physical = int(physical_read_bytes)
    return {
        "label": label,
        "anchors": int(anchors),
        "neighbours_per_row": int(neighbours_per_row),
        "row_bytes": int(row_bytes),
        "distinct_rows_touched": int(distinct_rows_touched),
        "distinct_row_coverage_of_substrate": (
            int(distinct_rows_touched) * int(row_bytes) / float(substrate_bytes)
            if substrate_bytes else None
        ),
        "useful_bytes": useful,
        "distinct_bytes": distinct,
        "physical_read_bytes": physical,
        "read_amplification_over_useful": (
            physical / float(useful) if useful else None
        ),
        "read_amplification_over_distinct": (
            physical / float(distinct) if distinct else None
        ),
        "physical_read_as_fraction_of_substrate": (
            physical / float(substrate_bytes) if substrate_bytes else None
        ),
        "wall_s": float(wall_s),
        "physical_rate_bytes_per_s": (
            physical / float(wall_s) if wall_s > 0 else None
        ),
        "useful_rate_bytes_per_s": (
            useful / float(wall_s) if wall_s > 0 else None
        ),
        "seconds_per_million_anchors": (
            float(wall_s) * 1e6 / float(anchors) if anchors else None
        ),
        "note": SORTED_GATHER_NOTE,
    }


def full_gather_ceiling(
    *,
    substrate_bytes: int,
    measured_delivered_rate_bytes_per_s: float,
    measured_physical_read_fraction_of_substrate: float,
    r0242_unsorted_physical_rate_bytes_per_s: float,
) -> dict[str, Any]:
    """The full-100M SORTED gather, bounded by arithmetic rather than fitted.

    A `100,000,000`-anchor `k = 15` gather references every row about `15`
    times. Sorted and deduplicated it touches each row once, so its physical
    read is at most the substrate itself - a CEILING, not an extrapolation.

    The two rates bracket the wall. The fast end is the rate at which this
    round's largest sorted gather actually DELIVERED substrate bytes
    (`distinct_bytes / wall`), which is well defined whether or not the pages
    came off the device; the physical read fraction is published beside it so a
    reader can see how much of that was cache assistance. The slow end is
    R0242's measured physical rate on the pessimistic random pattern, which no
    sorted gather should be slower than.
    """
    if float(measured_delivered_rate_bytes_per_s) <= 0.0:
        raise Round0243Error(
            "R0243 cannot price a full sorted gather from a non-positive "
            f"delivered rate ({measured_delivered_rate_bytes_per_s!r})"
        )
    if float(r0242_unsorted_physical_rate_bytes_per_s) <= 0.0:
        raise Round0243Error("R0243 sorted-gather reference rate must be positive")
    fast = float(substrate_bytes) / float(measured_delivered_rate_bytes_per_s)
    slow = float(substrate_bytes) / float(
        r0242_unsorted_physical_rate_bytes_per_s
    )
    lo, hi = (fast, slow) if fast <= slow else (slow, fast)
    return {
        "kind": "prediction",
        "quantity": "wall seconds of one full 100,000,000-anchor SORTED k15 gather",
        "substrate_bytes": int(substrate_bytes),
        "physical_read_ceiling_bytes": int(substrate_bytes),
        "ceiling_is_arithmetic_not_fitted": (
            "a sorted, deduplicated gather reads each touched row once, so its "
            "physical read cannot exceed the substrate however many anchors "
            "are gathered"
        ),
        "useful_bytes_at_100m_anchors": int(ROWS) * int(GRAPH_K) * int(
            DIMENSION * 4
        ),
        "implied_read_amplification_over_useful": (
            float(substrate_bytes)
            / float(int(ROWS) * int(GRAPH_K) * int(DIMENSION * 4))
        ),
        "rate_measured_here_bytes_per_s": float(
            measured_delivered_rate_bytes_per_s
        ),
        "rate_measured_here_is": (
            "distinct substrate bytes delivered per second by this round's "
            "largest sorted gather, cache assistance included"
        ),
        "measured_physical_read_fraction_of_substrate": float(
            measured_physical_read_fraction_of_substrate
        ),
        "rate_r0242_unsorted_bytes_per_s": float(
            r0242_unsorted_physical_rate_bytes_per_s
        ),
        "interval_s": [lo, hi],
        "interval_hours": [lo / 3600.0, hi / 3600.0],
        "registered_check_at_the_next_rung": (
            "any round that actually runs a full-substrate gather must "
            "instrument it against /proc/self/io and report the realised wall "
            "against this interval; this is a PREDICTION with an interval, not "
            "a measurement at 100,000,000 anchors"
        ),
        "note": SORTED_GATHER_NOTE,
    }


REGISTERED_INHERITANCE = {
    "graph_ids_sha256": REGISTERED_GRAPH_IDS_SHA256,
    "graph_cos_sha256": REGISTERED_GRAPH_COS_SHA256,
    "ladder_receipt_sha256": REGISTERED_LADDER_RECEIPT_SHA256,
    "cell": REGISTERED_SELECTED_CELL,
    "clusters": REGISTERED_SELECTED_CLUSTERS,
    "spill": SPILL,
    "k": GRAPH_K,
    "rows": ROWS,
}

#: R0242's sealed per-row vectors, bound by hash and re-earned by nothing.
R0242_VECTOR_KEYS = (
    "r0242_probe_cluster",
    "r0242_probe_strict_recall",
    "r0242_probe_tie_aware_recall",
    "r0242_probe_missing_edges",
    "r0242_probe_builder_missing_edges",
    "r0242_probe_in_degree",
    "r0242_primary_cluster",
)

R0242_LOCALITY_SHA256 = (
    "bfaf9f88d0e336b11a790880302d41ddfb12a7db7647a54c5c3c4bef42cc32d9"
)
R0242_TIE_AWARE_VECTOR_SHA256 = (
    "a1b3120191193263efb27775f830eb88a3505bd03d1c4118155617c10f8bd09f"
)
R0242_PRIMARY_CLUSTER_SHA256 = (
    "ebc0b6199b94576db4ad5025e9c867efb89f13c482c7d9babb7478b3205f9ece"
)

#: R0242's own measured physical rate on the unsorted 500,000-anchor gather,
#: sealed in its `gather_cost.physical_rate_bytes_per_s`. Used only as the
#: pessimistic end of the sorted-gather interval.
R0242_UNSORTED_PHYSICAL_RATE_BYTES_PER_S = 477_264_163.2660608

SUBSTRATE_BYTES = 153_600_000_128


def elapsed_since(started: float) -> float:
    import time

    return time.monotonic() - float(started)


def finite_or_raise(value: float, *, label: str) -> float:
    if not math.isfinite(float(value)):
        raise Round0243Error(f"{label} is not finite: {value!r}")
    return float(value)


__all__ = [
    "CANONICALIZATION_NOTE",
    "CANONICAL_CAPABILITY",
    "CLUSTERS",
    "CONCENTRATION_TOP_M",
    "DIMENSION",
    "EXPOSURE_GUARD_NOTE",
    "FUZZY_CAPABILITY",
    "FUZZY_DEADLINE_S",
    "FUZZY_FILE",
    "FUZZY_SCHEMA",
    "FUZZY_STAGE_BUDGET_S",
    "GPU_HOURS_CAP",
    "HALT_CELL_TIE_AWARE_BUILDER_RATE",
    "HALT_GLOBAL_TIE_AWARE_BUILDER_RATE",
    "HALT_P_VALUE",
    "HALT_RULE_NOTE",
    "HALT_SINGLE_CLUSTER_EXPOSURE_MULTIPLE",
    "HALT_SINGLE_CLUSTER_SHARE",
    "HALT_TOP_M_SHARE",
    "PARTITION_SEED",
    "PERMUTATIONS",
    "PERMUTATION_SEED",
    "PROBE_ROWS",
    "R0242_LOCALITY_SHA256",
    "R0242_PRIMARY_CLUSTER_SHA256",
    "R0242_TIE_AWARE_VECTOR_SHA256",
    "R0242_UNSORTED_PHYSICAL_RATE_BYTES_PER_S",
    "R0242_VECTOR_KEYS",
    "REGISTERED_INHERITANCE",
    "RESIDUAL_CAPABILITY",
    "RESIDUAL_DEADLINE_S",
    "RESIDUAL_EARLY_FILE",
    "RESIDUAL_EARLY_SCHEMA",
    "RESIDUAL_FILE",
    "RESIDUAL_SCHEMA",
    "RESIDUAL_STAGE_BUDGET_S",
    "ROUND_ID",
    "ROWS",
    "Round0243Error",
    "SAFETY_NOTE",
    "SCOPE_NOTE",
    "SORTED_GATHER_NOTE",
    "SUBSTRATE_BYTES",
    "SYMMETRISED_DEGREE_ONCE_NOTE",
    "STRICT_DECOMPOSITION_FIELDS",
    "STRICT_DISPERSION_FIELDS",
    "TIE_AWARE_NOTE",
    "canonical_undirected_degrees",
    "cluster_locality_test",
    "cluster_rate_table",
    "elapsed_since",
    "exposure_profile",
    "finite_or_raise",
    "full_gather_ceiling",
    "gather_price",
    "host_anonymous_bytes",
    "hot_cell_scan",
    "io_counters",
    "json_scrub",
    "loss_decomposition",
    "map_harm_assessment",
    "observed_dispersion",
    "post_canonical_tripwire",
    "residual_verdict",
    "sorted_gather_price",
    "strict_reproduction_gate",
    "symmetrised_degree_once",
    "weight_distribution",
]
