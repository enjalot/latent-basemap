"""Frozen contract for R0227 — does a LOW cluster count rescue `cluster-spill-nnd`?

R0226 qualified two 100M-capable builders and review-0226-01 closed the memory
question while blocking the recommendation. Three of its findings set this round
up exactly:

1. **Memory is solved and its law is exact.** Candidate A's device peak follows
   `4.65 GiB + 1560.9 B x max_cluster_rows` at `r^2 = 1.0000` across R0226's
   whole ladder, and its RMM peak is `1048.00002 B x max_cluster_rows` at every
   rung to five significant figures. It is **not** the plateau R0226 reported;
   what costs is the largest cluster, and nothing else.
2. **A's recall ceiling is set by `c`, not by `N`.** Review-0226-01 measured A's
   structural reachability ceiling on the sealed 2M substrate as `0.985` at
   `c = 8` falling to `0.867` at `c = 200`, with a fixed-`c` control showing that
   doubling `N` moves it by `< 0.001`. Under R0226's registered law
   `c = max(8, ceil(N*s/1e6))`, 100M runs at `c = 200`.
3. **~25 GiB of the card goes unused at every rung.**

Put those together and the experiment writes itself. A's memory law says
`max_cluster_rows` is the only thing that costs device memory, so the unused
headroom buys **fewer, larger clusters**, and `c` is exactly the knob the recall
ceiling responds to. At a `24 GiB` budget the law admits roughly
`(24 - 4.65 - 1.0) GiB / 1600 B ~= 12.3M` rows in the largest cluster, which at
`N = 100M` and `s = 2` is `c ~ 20-30` rather than `200`.

This round measures whether that trade actually works, and — because
review-0226-01's central finding was that A's *missing edges are concentrated in
sparse regions* rather than spread uniformly — whether it fixes the
concentration and not merely the mean.

## What is registered here

* A **reachability-versus-`c` map** at fixed `N = 2,000,000` on R0216's sealed
  `queue-correction-3` substrate, the only population with exact truth. Strict
  and tie-aware containment ceilings are both reported (the substrate contains
  exact-duplicate clusters, one with `1,377` members, so strict understates),
  plus the zero-reachable-row tripwire, at every `c`.
* A **build ladder ascending `max_cluster_rows`**, the registered resource axis,
  which is what A's memory law charges. `N` ascends within each fixed `c`. The
  ladder stops at its first refusal, abort, timeout or failure.
* A **verification of the memory law on this round's own cells**, including one
  cell whose largest cluster is `~6x` larger than anything R0226 measured. If
  the law holds there, the `c` that fits any budget at any `N` is a calculation.
* **Loss concentration** — recall by local-density decile and the
  neighbour-loss autocorrelation — measured for every emitted graph, against
  review-0226-01's `0.9130 -> 0.9957` and `r = 0.6216` at `c = 8`.
* A **100M projection** built phase by phase from measured per-phase costs,
  with spill I/O modelled explicitly rather than ignored. No projection is
  divided by another projection.

## What is NOT registered here

No adoption, no gate, no map, no fuzzy weighting, no sealed graph for downstream
consumption. R0226 was a qualification; this is a configuration study of the same
builder. A separate round adopts.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from basemap.round0226_graph_builders import (
    A_CLUSTER_STAGE_ROWS,
    A_GRAPH_DEGREE,
    A_INTERMEDIATE_DEGREE,
    A_KMEANS_ITERATIONS,
    A_KMEANS_SUBSAMPLE_ROWS,
    A_MAX_ITERATIONS,
    A_METRIC,
    A_SEED,
    A_SPILL,
    CANDIDATE_A,
    DIMENSION,
    GRAPH_K,
    PHASE2_RUNGS,
    PROJECTION_ROWS,
    SUBSTRATE_16M_PATH,
    SUBSTRATE_16M_ROWS,
    SUBSTRATE_2M_PATH,
    SUBSTRATE_2M_ROWS,
    TRUTH_COS_PATH,
    TRUTH_IDS_PATH,
    TRUTH_RECEIPT_PATH,
    TRUTH_SCHEMA,
)


ROUND_ID = "0227"

LOW_C_CAPABILITY = "minilm-100m-low-cluster-count-graph-configuration-v1"
REACHABILITY_SCHEMA = "round0227-reachability-versus-cluster-count-v1"
LADDER_SCHEMA = "round0227-low-c-build-ladder-v1"
EVALUATION_SCHEMA = "round0227-low-c-recall-concentration-and-projection-v1"
BUILD_SCHEMA = "round0227-low-c-cluster-spill-build-v1"

#: R0226's builder, unchanged in every respect except the cluster count. Every
#: nn-descent parameter, the spill factor, the k-means recipe and the seed are
#: imported rather than restated so a drift cannot hide in a copy.
CANDIDATE = CANDIDATE_A
SPILL = A_SPILL
SEED = A_SEED

#: R0226's `graph-k15-ids.i32.npy` for `cluster-spill-nnd-n2000000`, which ran at
#: `c = 8`. Consumed read-only as the round's control: this round re-derives its
#: recall and its loss concentration with this round's own code, so the
#: comparison against review-0226-01's numbers is like-for-like.
R0226_A_2M_GRAPH_IDS = (
    "/data/latent-basemap/runs/round-0226/queue/artifacts/"
    "minilm-100m-graph-builder-qualification-v1/builds/cluster-spill-nnd-n2000000/"
    "graph-k15-ids.i32.npy"
)
R0226_A_2M_CLUSTERS = 8

# --------------------------------------------------------------------------- #
# the memory law, from review-0226-01, restated as a prediction to be tested
# --------------------------------------------------------------------------- #
#: `device_wide_peak = intercept + slope x max_cluster_rows`.
#:
#: **Unit correction, carried forward deliberately.** Review-0226-01 writes the
#: intercept as `4.65 GiB`. Read as GiB the law does not reproduce the review's
#: own numbers: `4.65 GiB + 1560.9 x 1.69e6 = 7.107 GiB`, not the `6.79 GiB` the
#: review publishes beside it. Read as `4.65e9 bytes` (decimal GB) it reproduces
#: `6.787 GiB` exactly, and it reproduces R0226's three capacity-saturated
#: `device_wide` measurements to `0.06%`. Refitting those three rungs directly
#: gives `slope 1560.87`, `intercept 4.6467e9 B`, `r^2 = 0.999989` — which is the
#: law, and which is what is registered here. The review's slope is right; its
#: unit on the intercept is not.
DEVICE_LAW_INTERCEPT_BYTES = 4.6467e9
DEVICE_LAW_BYTES_PER_MAX_CLUSTER_ROW = 1560.9
#: R0226's `device_wide` reading at its 2M cell (`2,248,146,944 B`) is the one
#: the parent's 250 ms poll is known to have missed — the result says so, and the
#: child's 5 ms sampler read `5,604,966,400 B` in the same cell. Fitting all four
#: rungs on the *child* instrument gives `slope 1546.28`, `intercept 4.6646e9`,
#: `r^2 = 0.999991`; fitting all four on `device_wide` gives `r^2 = 0.953` and a
#: negative intercept. So review-0226-01's "fits all FOUR rungs at r^2 = 1.0000"
#: holds on the child sampler, not on the budget instrument. This round reports
#: the law's agreement on both instruments rather than picking the flattering one.
DEVICE_LAW_FOUR_RUNG_CHILD_SLOPE = 1546.28
DEVICE_LAW_FOUR_RUNG_CHILD_INTERCEPT_BYTES = 4.6646e9
#: `rmm_peak_bytes / max_cluster_rows = 1048.00002` at every R0226 rung — R0224's
#: measured nn-descent floor applied to the largest cluster.
RMM_LAW_BYTES_PER_MAX_CLUSTER_ROW = 1048.0
DEVICE_LAW_NOTE = (
    "review-0226-01: A's device peak is linear in max_cluster_rows, not flat in "
    "N, and its RMM peak is 1048.00002 B x max_cluster_rows to five significant "
    "figures at every R0226 rung. R0226 called this a plateau; it is a line in "
    "the largest cluster. Two corrections are carried here: the review's "
    "intercept unit is GB and not GiB (4.6467e9 B refits its own numbers, and "
    "the GiB reading contradicts the review's own 6.79 GiB figure), and its "
    "r^2 = 1.0000 over all four rungs holds on the child 5 ms sampler rather "
    "than on the 250 ms budget instrument, whose 2M reading the round already "
    "reported as a missed peak. This round tests the law at a max cluster "
    "roughly 6x larger than anything R0226 measured, on both instruments."
)

#: Cluster imbalance `max/mean` rises with `c`. Review-0226-01 fitted R0226's
#: three measured points (`1.163` at `c=8`, `1.237` at `c=16`, `1.395` at
#: `c=32`) as linear in `log2(c)` at `+0.116` per doubling, `r^2 = 0.959`. Used
#: ONLY by the pre-launch guard; every published figure uses this round's own
#: measured imbalance.
IMBALANCE_AT_C8 = 1.163
IMBALANCE_PER_DOUBLING = 0.116
IMBALANCE_FLOOR = 1.02


def imbalance_model(clusters: int) -> float:
    """Predicted `max/mean` cluster size at this `c`. Guard use only."""
    clusters = int(clusters)
    if clusters < 1:
        raise Round0227Error("R0227 imbalance model needs clusters >= 1")
    value = IMBALANCE_AT_C8 + IMBALANCE_PER_DOUBLING * math.log2(clusters / 8.0)
    return float(max(IMBALANCE_FLOOR, value))


# --------------------------------------------------------------------------- #
# the cluster-count law this round proposes, and its guard
# --------------------------------------------------------------------------- #
#: With `s = 2`, `c = 2` puts every row in every cluster and nothing is
#: partitioned at all; `c = 4` is the smallest count that still partitions.
C_MIN = 4
#: Guard margin on the predicted largest cluster. The prediction is a model; the
#: child re-checks the REALISED largest cluster against the same budget after
#: assignment and before any per-cluster build, which is the fail-closed check.
GUARD_IMBALANCE_MARGIN = 1.15
#: Bytes per max-cluster row charged by the guard: the measured slope rounded up.
GUARD_BYTES_PER_MAX_CLUSTER_ROW = 1600.0
#: Charged on top of the law's intercept so a guard error cannot walk the card.
GUARD_SAFETY_BYTES = 1024 ** 3

DEVICE_TOTAL_BYTES = 31.37 * 1024 ** 3
GUARD_DEVICE_BUDGET_BYTES = 24 * 1024 ** 3
#: **Anonymous**, not RSS (review-0224-01, confirmed twice).
GUARD_HOST_ANON_BUDGET_BYTES = 60 * 1024 ** 3
#: Swap is judged as GROWTH over a pre-launch baseline, never as an absolute
#: level: this box holds swap at idle and an absolute threshold refuses
#: everything.
GUARD_SWAP_GROWTH_ABORT_BYTES = 1 * 1024 ** 3
WATCHDOG_POLL_S = 0.25
GUARD_SIGTERM_GRACE_S = 180.0
BUILD_TIMEOUT_S = 3_600.0
SAMPLE_INTERVAL_S = 0.005

GPU_HOURS_CAP = 3.0

#: The largest realised cluster this round will build, from the guard model at
#: the device budget. Re-checked inside the child against the realised sizes.
CLUSTER_CAPACITY_ROWS = int(
    (GUARD_DEVICE_BUDGET_BYTES - DEVICE_LAW_INTERCEPT_BYTES - GUARD_SAFETY_BYTES)
    / GUARD_BYTES_PER_MAX_CLUSTER_ROW
)

GUARD_BUDGET_NOTE = (
    "device budget 24 GiB of the card's 31.37 GiB; host ANONYMOUS budget 60 GiB "
    "of the box's 123 GiB. A cell whose predicted footprint exceeds either is "
    "refused before launch and recorded as refused_a_priori with its "
    "prediction; a cell whose REALISED largest cluster exceeds the registered "
    "capacity refuses itself after assignment and before any per-cluster build. "
    "Both refusals are data: they measure where the configuration stops being "
    "launchable on this box."
)

# --------------------------------------------------------------------------- #
# spill scratch, and the I/O the fitted range never exercised
# --------------------------------------------------------------------------- #
#: Peak spill scratch on `/data`. R0226 split clusters into `ceil(N*s*1536 /
#: budget)` equal groups, which at LOW `c` no longer bounds anything: a group of
#: two 12.5M-row clusters is 38 GB against a 24 GiB budget. This round packs
#: whole clusters into groups instead, so peak scratch is bounded by
#: construction and the number of substrate passes is an output rather than an
#: assumption. Passes are what the 100M I/O term is made of.
SCRATCH_BUDGET_BYTES = 24 * 1024 ** 3
#: Measured cold sequential read on `gsv:/data` (review-0226-01, two untouched
#: shards with `posix_fadvise(DONTNEED)`): 5.53 and 6.36 GB/s. The conservative
#: end is used for the projection and the round says so.
DATA_COLD_READ_BYTES_PER_S = 5.53e9
DATA_READ_NOTE = (
    "review-0226-01 measured gsv:/data cold sequential read at 5.53 and 6.36 "
    "GB/s with posix_fadvise(DONTNEED); the 100M spill-I/O term uses the "
    "conservative 5.53 GB/s and is reported as its own line, never folded into "
    "a compute fit"
)


def pack_clusters_into_groups(
    sizes: Sequence[int], *, budget_bytes: int = SCRATCH_BUDGET_BYTES
) -> list[list[int]]:
    """Assign clusters to spill groups so each group fits the scratch budget.

    One pass over the substrate is made per group, so the group count IS the
    number of substrate re-reads and it is the quantity the 100M I/O projection
    is built from. Clusters are packed in index order and a cluster larger than
    the whole budget gets a group of its own — the budget is then exceeded by
    construction, which the caller reports rather than hides.
    """
    row_bytes = DIMENSION * 4
    groups: list[list[int]] = []
    current: list[int] = []
    current_bytes = 0
    for index, size in enumerate(sizes):
        size = int(size)
        if size <= 0:
            continue
        cost = size * row_bytes
        if current and current_bytes + cost > int(budget_bytes):
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(index)
        current_bytes += cost
    if current:
        groups.append(current)
    return groups


def substrate_passes(rows: int, clusters: int, *, imbalance: float | None = None) -> int:
    """How many times the substrate is re-read at this `(N, c)`.

    Model only, for the projection; the builds report their realised group count.
    """
    rows = int(rows)
    clusters = int(clusters)
    factor = imbalance_model(clusters) if imbalance is None else float(imbalance)
    mean_rows = rows * SPILL / float(clusters)
    sizes = [int(round(mean_rows))] * clusters
    largest = int(round(mean_rows * factor))
    if sizes:
        sizes[0] = largest
    return len(pack_clusters_into_groups(sizes))


# --------------------------------------------------------------------------- #
# the matrix
# --------------------------------------------------------------------------- #
#: Cluster counts swept for the reachability map at fixed `N = 2,000,000`. It
#: spans this round's proposed regime (`c = 4-32`) and reaches R0226's
#: registered 100M value (`c = 200`) so the map connects to the measurement
#: review-0226-01 already made.
REACHABILITY_CLUSTERS: tuple[int, ...] = (4, 8, 16, 24, 32, 48, 64, 100, 200)
REACHABILITY_ROWS = SUBSTRATE_2M_ROWS
#: Every row's strict ceiling is computed (it is a table lookup); the tie-aware
#: ceiling needs a full similarity scan per query, so it runs on a seeded sample.
REACHABILITY_TIE_QUERY_ROWS = 50_000
REACHABILITY_QUERY_SEED = 227
TIE_TOLERANCE = 1e-6

#: The build ladder, ascending in **predicted max_cluster_rows** — the axis A's
#: memory law charges, and therefore the axis along which a cell can stop being
#: launchable. `N` ascends within each fixed `c`. A refusal, abort, timeout or
#: failure stops the ladder: a larger cluster of a configuration that already
#: failed cannot succeed.
BUILD_CELLS: tuple[dict[str, Any], ...] = (
    {"rows": 2_000_000, "clusters": 64, "emit_graph": True},
    {"rows": 2_000_000, "clusters": 32, "emit_graph": True},
    {"rows": 2_000_000, "clusters": 16, "emit_graph": True},
    {"rows": 2_000_000, "clusters": 8, "emit_graph": True},
    {"rows": 4_000_000, "clusters": 16, "emit_graph": False},
    {"rows": 8_000_000, "clusters": 16, "emit_graph": False},
    {"rows": 16_000_000, "clusters": 16, "emit_graph": True},
    {"rows": 16_000_000, "clusters": 8, "emit_graph": False},
    {"rows": 16_000_000, "clusters": 4, "emit_graph": True},
)
SUBSTRATE_BY_ROWS: dict[int, str] = {
    2_000_000: SUBSTRATE_2M_PATH,
    4_000_000: SUBSTRATE_16M_PATH,
    8_000_000: SUBSTRATE_16M_PATH,
    16_000_000: SUBSTRATE_16M_PATH,
}
RECALL_ROWS = SUBSTRATE_2M_ROWS
#: The 16M rungs have no sealed exact truth, so this round computes its own by
#: brute force for a seeded query set — review-0226-01's highest-value
#: follow-up. Seeds plus their exact neighbours are both scored so the
#: neighbour-loss autocorrelation is computable at 16M too.
LARGE_RECALL_ROWS = SUBSTRATE_16M_ROWS
LARGE_RECALL_SEED_ROWS = 5_000
LARGE_RECALL_SEED = 1227
LARGE_RECALL_SCAN_BLOCK = 262_144

#: Phase 2's rungs and the floors the plan requires of any graph builder.
RECALL_MEAN_FLOOR = 0.90
RECALL_P10_FLOOR = 0.80
DENSITY_DECILES = 10

#: review-0226-01's measurements at `c = 8`, restated so this round's tables
#: carry the comparison the mandate asks for. Not inputs to any computation.
R0226_REVIEW_BASELINE: dict[str, Any] = {
    "clusters": 8,
    "tie_aware_mean": 0.9707655906677246,
    "strict_mean": 0.9696856141090393,
    "density_decile_tie_aware": [
        0.9130, 0.9508, 0.9626, 0.9706, 0.9759,
        0.9794, 0.9828, 0.9866, 0.9902, 0.9957,
    ],
    "neighbour_loss_correlation": 0.6216,
    "rows_carrying_any_loss": 0.1957,
    "worst_1pct_share_of_loss": 0.193,
    "worst_10pct_share_of_loss": 0.782,
    "reachability_ceiling_by_c": {"8": 0.9851, "16": 0.9597, "32": 0.9418,
                                  "64": 0.9125, "200": 0.8675},
    "source": "review-0226-2026-08-08-01.md",
    "note": (
        "review-0226-01's own CPU reimplementation of k-means and spill "
        "assignment; its max/mean at c=8 was 1.92-2.14 against the release's "
        "1.216, so its ceilings are expected to sit BELOW the release's. The "
        "comparison of interest is the gradient in c, not the absolute level."
    ),
}


class Round0227Error(RuntimeError):
    """The registered R0227 low-cluster-count contract changed."""


def cluster_settings() -> tuple[dict[str, Any], ...]:
    """Every build cell, in run order, with its identity and prediction."""
    out: list[dict[str, Any]] = []
    for cell in BUILD_CELLS:
        rows = int(cell["rows"])
        clusters = int(cell["clusters"])
        if clusters < C_MIN:
            raise Round0227Error(
                f"R0227 cell c={clusters} is below the registered floor {C_MIN}"
            )
        if rows not in SUBSTRATE_BY_ROWS:
            raise Round0227Error(f"R0227 has no substrate registered for {rows} rows")
        out.append({
            "id": f"low-c-n{rows}-c{clusters}",
            "candidate": CANDIDATE,
            "rows": rows,
            "clusters": clusters,
            "spill": SPILL,
            "dimension": DIMENSION,
            "k": GRAPH_K,
            "substrate": SUBSTRATE_BY_ROWS[rows],
            "emit_graph": bool(cell["emit_graph"]),
            "predicted_max_cluster_rows": predicted_max_cluster_rows(
                rows=rows, clusters=clusters
            ),
        })
    ordered = sorted(out, key=lambda item: item["predicted_max_cluster_rows"])
    if [item["id"] for item in ordered] != [item["id"] for item in out]:
        raise Round0227Error(
            "R0227 build cells must be registered in ascending predicted "
            "max_cluster_rows, which is the resource axis the ladder ascends"
        )
    return tuple(out)


def predicted_max_cluster_rows(*, rows: int, clusters: int) -> int:
    """Guard-side prediction of the largest realised cluster."""
    rows = int(rows)
    clusters = int(clusters)
    if rows <= 0 or clusters < 1:
        raise Round0227Error("R0227 max-cluster prediction needs rows > 0, c >= 1")
    mean_rows = rows * SPILL / float(clusters)
    return int(math.ceil(mean_rows * imbalance_model(clusters) * GUARD_IMBALANCE_MARGIN))


def device_bytes_from_law(max_cluster_rows: float) -> float:
    """The measured law's prediction. Published beside every measurement."""
    return float(
        DEVICE_LAW_INTERCEPT_BYTES
        + DEVICE_LAW_BYTES_PER_MAX_CLUSTER_ROW * float(max_cluster_rows)
    )


def rmm_bytes_from_law(max_cluster_rows: float) -> float:
    return float(RMM_LAW_BYTES_PER_MAX_CLUSTER_ROW * float(max_cluster_rows))


def predict_footprint(*, rows: int, clusters: int) -> dict[str, Any]:
    """Expected device and host-anonymous bytes for a cell, BEFORE it launches."""
    rows = int(rows)
    clusters = int(clusters)
    row_bytes = DIMENSION * 4
    largest = predicted_max_cluster_rows(rows=rows, clusters=clusters)
    topk_bytes = rows * GRAPH_K * 8
    kmeans_device = A_KMEANS_SUBSAMPLE_ROWS * row_bytes + clusters * row_bytes
    cluster_device = (
        DEVICE_LAW_INTERCEPT_BYTES
        + GUARD_BYTES_PER_MAX_CLUSTER_ROW * largest
        + GUARD_SAFETY_BYTES
    )
    device_bytes = int(max(kmeans_device + GUARD_SAFETY_BYTES, cluster_device))
    assignment_bytes = rows * SPILL * 4
    stage_bytes = A_CLUSTER_STAGE_ROWS * row_bytes
    # The local graph of the largest cluster is materialised host-side as int64.
    local_graph_bytes = largest * A_GRAPH_DEGREE * 8
    anonymous_bytes = int(
        topk_bytes
        + assignment_bytes
        + stage_bytes
        + local_graph_bytes
        + 3 * 1024 ** 3
    )
    passes = substrate_passes(rows, clusters)
    return {
        "candidate": CANDIDATE,
        "rows": rows,
        "clusters": clusters,
        "spill": SPILL,
        "dimension": DIMENSION,
        "predicted_max_cluster_rows": largest,
        "predicted_mean_cluster_rows": rows * SPILL / float(clusters),
        "predicted_imbalance": imbalance_model(clusters),
        "guard_imbalance_margin": GUARD_IMBALANCE_MARGIN,
        "predicted_device_bytes": device_bytes,
        "predicted_device_gib": device_bytes / (1024 ** 3),
        "device_law_bytes": device_bytes_from_law(largest),
        "device_law_gib": device_bytes_from_law(largest) / (1024 ** 3),
        "rmm_law_bytes": rmm_bytes_from_law(largest),
        "predicted_host_anon_bytes": anonymous_bytes,
        "predicted_host_anon_gib": anonymous_bytes / (1024 ** 3),
        "terms": {
            "global_topk_accumulator_bytes": int(topk_bytes),
            "assignment_table_bytes": int(assignment_bytes),
            "cluster_device_stage_bytes": int(stage_bytes),
            "largest_local_graph_host_bytes": int(local_graph_bytes),
            "kmeans_device_bytes": int(kmeans_device),
            "cluster_capacity_rows": int(CLUSTER_CAPACITY_ROWS),
            "predicted_substrate_passes": int(passes),
            "predicted_substrate_read_bytes": int(passes * rows * row_bytes),
            "predicted_peak_scratch_bytes": int(
                min(SCRATCH_BUDGET_BYTES, rows * SPILL * row_bytes)
            ),
        },
        "device_model": (
            "review-0226-01's measured law, guarded: 4.65 GiB intercept + 1600 "
            "B per predicted max-cluster row + 1 GiB safety. Set by the largest "
            "cluster, not by N — which is the whole reason c is the free knob."
        ),
        "host_model": (
            "anonymous only. The substrate memmap and the spill files are clean "
            "file-backed page cache: evicted, not swapped (review-0224-01), so "
            "they are measured and never charged to the swappable budget."
        ),
    }


def guard_decision(
    *,
    rows: int,
    clusters: int,
    device_budget_bytes: int = GUARD_DEVICE_BUDGET_BYTES,
    host_anon_budget_bytes: int = GUARD_HOST_ANON_BUDGET_BYTES,
) -> dict[str, Any]:
    """Launch this cell, or refuse it and record the refusal as a measurement."""
    prediction = predict_footprint(rows=rows, clusters=clusters)
    device_over = prediction["predicted_device_bytes"] > int(device_budget_bytes)
    host_over = prediction["predicted_host_anon_bytes"] > int(host_anon_budget_bytes)
    capacity_over = (
        int(prediction["predicted_max_cluster_rows"]) > int(CLUSTER_CAPACITY_ROWS)
    )
    reasons: list[str] = []
    if device_over:
        reasons.append(
            f"predicted device {prediction['predicted_device_gib']:.2f} GiB "
            f"exceeds the {device_budget_bytes / 1024 ** 3:.2f} GiB budget"
        )
    if host_over:
        reasons.append(
            f"predicted host anonymous {prediction['predicted_host_anon_gib']:.2f} "
            f"GiB exceeds the {host_anon_budget_bytes / 1024 ** 3:.2f} GiB budget"
        )
    if capacity_over:
        reasons.append(
            f"predicted largest cluster "
            f"{prediction['predicted_max_cluster_rows']} exceeds the registered "
            f"capacity {CLUSTER_CAPACITY_ROWS}"
        )
    return {
        "prediction": prediction,
        "device_budget_bytes": int(device_budget_bytes),
        "host_anon_budget_bytes": int(host_anon_budget_bytes),
        "cluster_capacity_rows": int(CLUSTER_CAPACITY_ROWS),
        "device_over_budget": bool(device_over),
        "host_over_budget": bool(host_over),
        "capacity_over_budget": bool(capacity_over),
        "allowed": not (device_over or host_over or capacity_over),
        "refused_a_priori": bool(device_over or host_over or capacity_over),
        "refusal_reasons": reasons,
        "budget_note": GUARD_BUDGET_NOTE,
    }


# --------------------------------------------------------------------------- #
# the low-c cluster law, and what it recommends per rung
# --------------------------------------------------------------------------- #
def smallest_feasible_clusters(
    *,
    rows: int,
    device_budget_bytes: int = GUARD_DEVICE_BUDGET_BYTES,
    imbalance: "Mapping[int, float] | None" = None,
    c_min: int = C_MIN,
    c_max: int = 4096,
) -> dict[str, Any]:
    """The smallest `c` whose largest cluster still fits the device budget.

    Fewer clusters is strictly better for reachability, so the recommended
    configuration is the smallest `c` the memory law admits. With the law
    verified this is a calculation, not a fit: it takes a measured intercept, a
    measured slope, a measured imbalance and a budget, and returns an integer.
    """
    rows = int(rows)
    budget = float(device_budget_bytes)
    admissible_rows = (
        budget - DEVICE_LAW_INTERCEPT_BYTES - GUARD_SAFETY_BYTES
    ) / GUARD_BYTES_PER_MAX_CLUSTER_ROW
    for clusters in range(int(c_min), int(c_max) + 1):
        factor = (
            float(imbalance[clusters])
            if imbalance is not None and clusters in imbalance
            else imbalance_model(clusters)
        )
        largest = rows * SPILL * factor / float(clusters)
        if largest <= admissible_rows:
            return {
                "rows": rows,
                "clusters": int(clusters),
                "imbalance_used": factor,
                "imbalance_source": (
                    "measured in this round" if imbalance is not None
                    and clusters in imbalance else "review-0226-01 model"
                ),
                "predicted_max_cluster_rows": float(largest),
                "admissible_max_cluster_rows": float(admissible_rows),
                "predicted_device_bytes": float(
                    DEVICE_LAW_INTERCEPT_BYTES
                    + GUARD_BYTES_PER_MAX_CLUSTER_ROW * largest
                    + GUARD_SAFETY_BYTES
                ),
                "device_budget_bytes": float(budget),
                "feasible": True,
            }
    return {
        "rows": rows,
        "clusters": None,
        "feasible": False,
        "admissible_max_cluster_rows": float(admissible_rows),
        "reason": f"no c in [{c_min}, {c_max}] fits the budget at {rows} rows",
    }


def linear_fit(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    """Least squares `y = slope*x + intercept` with its own `r^2` and range."""
    xs = np.asarray(list(x), dtype=np.float64)
    ys = np.asarray(list(y), dtype=np.float64)
    if xs.size != ys.size or xs.size < 2 or not np.all(np.isfinite(ys)):
        raise Round0227Error("R0227 linear fit needs >= 2 finite matched points")
    design = np.vstack([xs, np.ones_like(xs)]).T
    coefficients, *_ = np.linalg.lstsq(design, ys, rcond=None)
    predicted = design @ coefficients
    residual = float(((ys - predicted) ** 2).sum())
    total = float(((ys - ys.mean()) ** 2).sum())
    return {
        "slope": float(coefficients[0]),
        "intercept": float(coefficients[1]),
        "r_squared": 1.0 - residual / total if total > 0 else 1.0,
        "n_points": int(xs.size),
        "fitted_range": [float(xs.min()), float(xs.max())],
        "residuals": [float(value) for value in (ys - predicted)],
    }


def power_fit(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    """Least squares `y = a * x**b` in log-log, with its own `r^2` and range."""
    xs = np.asarray(list(x), dtype=np.float64)
    ys = np.asarray(list(y), dtype=np.float64)
    if xs.size != ys.size or xs.size < 2:
        raise Round0227Error("R0227 power fit needs >= 2 matched points")
    if np.any(xs <= 0) or np.any(ys <= 0) or not np.all(np.isfinite(ys)):
        raise Round0227Error("R0227 power fit needs positive finite points")
    design = np.vstack([np.log(xs), np.ones_like(xs)]).T
    coefficients, *_ = np.linalg.lstsq(design, np.log(ys), rcond=None)
    predicted = design @ coefficients
    residual = float(((np.log(ys) - predicted) ** 2).sum())
    total = float(((np.log(ys) - np.log(ys).mean()) ** 2).sum())
    return {
        "exponent": float(coefficients[0]),
        "coefficient": float(np.exp(coefficients[1])),
        "r_squared": 1.0 - residual / total if total > 0 else 1.0,
        "n_points": int(xs.size),
        "fitted_range": [float(xs.min()), float(xs.max())],
    }


def law_agreement(
    *,
    measured_bytes: Sequence[float],
    max_cluster_rows: Sequence[int],
    child_bytes: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Does review-0226-01's memory law hold on THIS round's cells?

    Reports the law's residuals as published, the ratio the law is built from,
    and a fresh fit on this round's own points so a reader can see whether the
    published constants or the refit are the better description.
    """
    measured = np.asarray(list(measured_bytes), dtype=np.float64)
    largest = np.asarray(list(max_cluster_rows), dtype=np.float64)
    if measured.size != largest.size or measured.size < 2:
        raise Round0227Error("R0227 law agreement needs >= 2 matched points")
    predicted = np.asarray([device_bytes_from_law(value) for value in largest])
    relative = (measured - predicted) / predicted
    refit = linear_fit(largest, measured)
    child_refit = None
    if child_bytes is not None:
        child = np.asarray(list(child_bytes), dtype=np.float64)
        if child.size == largest.size and np.all(np.isfinite(child)) and child.min() > 0:
            child_refit = linear_fit(largest, child)
    return {
        "child_instrument_refit": child_refit,
        "child_instrument_note": (
            "the child's 5 ms cudaMemGetInfo sampler, refitted on the same "
            "max-cluster rows. R0226's 250 ms parent poll missed a peak in one "
            "cell; publishing both refits means the law is not resting on "
            "whichever instrument happens to flatter it."
        ),
        "n_points": int(measured.size),
        "max_cluster_rows": [int(value) for value in largest],
        "measured_bytes": [float(value) for value in measured],
        "law_predicted_bytes": [float(value) for value in predicted],
        "relative_error": [float(value) for value in relative],
        "worst_absolute_relative_error": float(np.max(np.abs(relative))),
        "max_cluster_row_range": [int(largest.min()), int(largest.max())],
        "extension_beyond_r0226_max_cluster": float(largest.max() / 1_395_191.0),
        "published_law": {
            "intercept_bytes": DEVICE_LAW_INTERCEPT_BYTES,
            "bytes_per_max_cluster_row": DEVICE_LAW_BYTES_PER_MAX_CLUSTER_ROW,
            "source": "review-0226-2026-08-08-01.md",
        },
        "refit_on_this_round": refit,
    }


def project_100m(
    *,
    clusters: int,
    per_cluster_nn_descent: Mapping[str, Any],
    cosine_per_spilled_row_s: float,
    merge_per_row_s: float,
    kmeans_assign_seconds: float,
    imbalance: float,
    rows: int = PROJECTION_ROWS,
    read_bytes_per_s: float = DATA_COLD_READ_BYTES_PER_S,
) -> dict[str, Any]:
    """A phase-by-phase 100M wall projection, every term labelled with its basis.

    Built from measured per-phase costs. Spill I/O is modelled explicitly from
    the realised group-packing rule and review-0226-01's measured `/data`
    throughput rather than left inside a whole-builder power law that never
    sampled the multi-pass regime. **No term here is a projection divided by
    another projection.**
    """
    rows = int(rows)
    clusters = int(clusters)
    mean_cluster = rows * SPILL / float(clusters)
    largest_cluster = mean_cluster * float(imbalance)
    per_cluster_seconds = float(per_cluster_nn_descent["coefficient"]) * (
        mean_cluster ** float(per_cluster_nn_descent["exponent"])
    )
    nn_descent_seconds = per_cluster_seconds * clusters
    spilled_rows = rows * SPILL
    cosine_seconds = float(cosine_per_spilled_row_s) * spilled_rows
    merge_seconds = float(merge_per_row_s) * spilled_rows
    sizes = [int(round(mean_cluster))] * clusters
    if sizes:
        sizes[0] = int(round(largest_cluster))
    passes = len(pack_clusters_into_groups(sizes))
    read_bytes = passes * rows * DIMENSION * 4
    write_bytes = spilled_rows * DIMENSION * 4
    spill_io_seconds = (read_bytes + write_bytes) / float(read_bytes_per_s)
    total = (
        float(kmeans_assign_seconds)
        + spill_io_seconds
        + nn_descent_seconds
        + cosine_seconds
        + merge_seconds
    )
    return {
        "rows": rows,
        "clusters": clusters,
        "imbalance_used": float(imbalance),
        "mean_cluster_rows": float(mean_cluster),
        "max_cluster_rows": float(largest_cluster),
        "substrate_passes": int(passes),
        "spill_read_bytes": int(read_bytes),
        "spill_write_bytes": int(write_bytes),
        "read_bytes_per_s": float(read_bytes_per_s),
        "terms_seconds": {
            "kmeans_and_assign": float(kmeans_assign_seconds),
            "spill_io": float(spill_io_seconds),
            "nn_descent": float(nn_descent_seconds),
            "exact_cosine": float(cosine_seconds),
            "merge": float(merge_seconds),
        },
        "per_cluster_nn_descent_seconds": float(per_cluster_seconds),
        "projected_seconds": float(total),
        "projected_hours": float(total) / 3600.0,
        "is_projection": True,
        "bases": {
            "nn_descent": (
                f"per-cluster power fit over {per_cluster_nn_descent['n_points']} "
                f"measured clusters spanning "
                f"{per_cluster_nn_descent['fitted_range'][0]:.0f}-"
                f"{per_cluster_nn_descent['fitted_range'][1]:.0f} rows, "
                f"exponent {per_cluster_nn_descent['exponent']:.4f}, "
                f"r^2 {per_cluster_nn_descent['r_squared']:.4f}; extrapolation "
                f"factor on cluster rows "
                f"{mean_cluster / float(per_cluster_nn_descent['fitted_range'][1]):.2f}"
            ),
            "spill_io": DATA_READ_NOTE,
            "exact_cosine": "measured seconds per spilled row, linear",
            "merge": "measured seconds per spilled row, linear",
            "kmeans_and_assign": "measured at the largest built cell",
        },
        "discipline": (
            "labelled projection. Each phase carries its own measured basis and "
            "extrapolation factor; no projection is divided by another "
            "projection and no speedup ratio is formed from two projections."
        ),
    }


__all__ = [
    "A_GRAPH_DEGREE",
    "A_INTERMEDIATE_DEGREE",
    "A_KMEANS_ITERATIONS",
    "A_KMEANS_SUBSAMPLE_ROWS",
    "A_MAX_ITERATIONS",
    "A_METRIC",
    "BUILD_CELLS",
    "BUILD_SCHEMA",
    "BUILD_TIMEOUT_S",
    "CANDIDATE",
    "CLUSTER_CAPACITY_ROWS",
    "C_MIN",
    "DATA_COLD_READ_BYTES_PER_S",
    "DATA_READ_NOTE",
    "DENSITY_DECILES",
    "DEVICE_LAW_BYTES_PER_MAX_CLUSTER_ROW",
    "DEVICE_LAW_INTERCEPT_BYTES",
    "DEVICE_LAW_NOTE",
    "DEVICE_TOTAL_BYTES",
    "DIMENSION",
    "EVALUATION_SCHEMA",
    "GPU_HOURS_CAP",
    "GRAPH_K",
    "GUARD_BUDGET_NOTE",
    "GUARD_BYTES_PER_MAX_CLUSTER_ROW",
    "GUARD_DEVICE_BUDGET_BYTES",
    "GUARD_HOST_ANON_BUDGET_BYTES",
    "GUARD_IMBALANCE_MARGIN",
    "GUARD_SAFETY_BYTES",
    "GUARD_SIGTERM_GRACE_S",
    "GUARD_SWAP_GROWTH_ABORT_BYTES",
    "IMBALANCE_AT_C8",
    "IMBALANCE_PER_DOUBLING",
    "LADDER_SCHEMA",
    "LARGE_RECALL_ROWS",
    "LARGE_RECALL_SCAN_BLOCK",
    "LARGE_RECALL_SEED",
    "LARGE_RECALL_SEED_ROWS",
    "LOW_C_CAPABILITY",
    "PHASE2_RUNGS",
    "PROJECTION_ROWS",
    "R0226_A_2M_CLUSTERS",
    "R0226_A_2M_GRAPH_IDS",
    "R0226_REVIEW_BASELINE",
    "REACHABILITY_CLUSTERS",
    "REACHABILITY_QUERY_SEED",
    "REACHABILITY_ROWS",
    "REACHABILITY_SCHEMA",
    "REACHABILITY_TIE_QUERY_ROWS",
    "RECALL_MEAN_FLOOR",
    "RECALL_P10_FLOOR",
    "RECALL_ROWS",
    "RMM_LAW_BYTES_PER_MAX_CLUSTER_ROW",
    "ROUND_ID",
    "Round0227Error",
    "SAMPLE_INTERVAL_S",
    "SCRATCH_BUDGET_BYTES",
    "SEED",
    "SPILL",
    "SUBSTRATE_16M_PATH",
    "SUBSTRATE_2M_PATH",
    "SUBSTRATE_BY_ROWS",
    "TIE_TOLERANCE",
    "TRUTH_COS_PATH",
    "TRUTH_IDS_PATH",
    "TRUTH_RECEIPT_PATH",
    "TRUTH_SCHEMA",
    "WATCHDOG_POLL_S",
    "cluster_settings",
    "device_bytes_from_law",
    "guard_decision",
    "imbalance_model",
    "law_agreement",
    "linear_fit",
    "pack_clusters_into_groups",
    "power_fit",
    "predict_footprint",
    "predicted_max_cluster_rows",
    "project_100m",
    "rmm_bytes_from_law",
    "smallest_feasible_clusters",
    "substrate_passes",
]
