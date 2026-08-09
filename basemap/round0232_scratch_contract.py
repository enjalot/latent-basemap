"""Frozen contract for R0232 — does 100M fit the disk we have?

Review-0229-01 blocked `claim:r0229-50m-and-100m-are-unblocked-inside-the-existing-
device-and-storage-budgets` on a storage argument:

    "At 100M `s = 8` the spill set is `1,229 GB`. Under the same budget that is
    ~48 spill groups and ~48 substrate passes (~7.4 TB of unmodelled reads), and
    under a raised budget it needs ~1.2 TB of free scratch. `/data` currently has
    280 GB free (92% used)."

Two quantities are being named there and only one of them is a disk requirement.
`1,229 GB` is the **total spill volume** `V = s x N x 1536 B`. **Peak scratch at
any instant** is a different number, and R0227's builder already bounds it:
`pack_clusters_into_groups` packs whole clusters against `SCRATCH_BUDGET_BYTES`,
materialises one group at a time, and removes each cluster file the instant its
local graph has been merged. R0229's sealed arm receipt records
`peak_scratch_bytes: 24576000000` in `spill_groups: 1`.

But `peak_scratch_bytes` is **modelled**: the builder computes it from `sizes`
before a byte is written and no round has ever sampled the filesystem while a
build ran. A disk decision resting on a model is exactly what review-0229-01 §5.4
penalised. This round measures it.

## What is registered here

* A **measured peak-scratch law** against `(N, c, s, bound)`, from a filesystem
  sampler running inside the build, published beside the modelled value and beside
  `/proc/<pid>/io` byte counters.
* Two **zero-scratch builders**, `stream-resident` (the mandate's design: a
  bounded host buffer replacing the spill file) and `stream-gather` (the limit of
  the same idea: one cluster at a time, taken by direct ascending index gather
  from the substrate memmap).
* A **byte-identity claim** between the three modes at a shared cached partition,
  tested on `sha256` of the merged `top_ids` and `top_cos` arrays.
* A **device-law refit at `gd 64 / igd 256`**, which R0229 registered and did not
  do, using a largest cluster about `12x` bigger than the arm's `170,504`.
* **Measured `/data` cold read and fsync'd write throughput on this box**, so the
  I/O line of the projection is a measurement and not carried prose.
* A **disk axis on the predictive guard**. A cell whose predicted peak scratch
  exceeds the round budget, or whose prediction would drive `/data` free space
  below the reserve, is refused before launch and the refusal is recorded as data.

## What is NOT registered here

No adoption, no gate, no 50M or 100M build, no equivalence claim beyond what
byte-identity of the merged arrays supports, and no claim that the map arm is
undisplaced — only whether its displacement is smaller than the design's minimum
detectable effect.
"""
from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from typing import Any

from basemap.round0226_graph_builders import (
    DIMENSION,
    GRAPH_K,
    SUBSTRATE_16M_PATH,
    SUBSTRATE_16M_ROWS,
    SUBSTRATE_2M_PATH,
    SUBSTRATE_2M_ROWS,
    TRUTH_COS_PATH,
    TRUTH_IDS_PATH,
    TRUTH_RECEIPT_PATH,
)
from basemap.round0227_low_c_contract import (
    CLUSTER_CAPACITY_ROWS,
    DEVICE_LAW_BYTES_PER_MAX_CLUSTER_ROW,
    DEVICE_LAW_INTERCEPT_BYTES,
    GUARD_DEVICE_BUDGET_BYTES,
    GUARD_HOST_ANON_BUDGET_BYTES,
    GUARD_SWAP_GROWTH_ABORT_BYTES,
    SCRATCH_BUDGET_BYTES,
    pack_clusters_into_groups,
)
from basemap.round0229_quality_contract import guard_for_spill


ROUND_ID = "0232"


class Round0232Error(RuntimeError):
    """A fail-closed R0232 contract violation."""


# --------------------------------------------------------------------------- #
# identity
# --------------------------------------------------------------------------- #
GRID_CAPABILITY = "minilm-mixed-2m-cluster-spill-scratch-law-v1"
LARGER_N_CAPABILITY = "minilm-mixed-8m-cluster-spill-scratch-and-device-calibration-v1"
GRAPH_CAPABILITY = "minilm-mixed-2m-streamed-spill-k15-fuzzy-graph-v1"
GEOMETRY_CAPABILITY = "minilm-mixed-2m-streamed-spill-map-geometry-v1"
PROJECTION_CAPABILITY = "minilm-100m-cluster-spill-scratch-and-cost-projection-v1"

GRID_SCHEMA = "round0232-cluster-spill-scratch-law-v1"
LARGER_N_SCHEMA = "round0232-larger-n-scratch-and-device-calibration-v1"
BUILD_SCHEMA = "round0232-bounded-scratch-cluster-spill-build-v1"
GRAPH_SCHEMA = "round0232-streamed-spill-fuzzy-graph-v1"
TRAIN_SCHEMA = "round0232-streamed-spill-map-train-v1"
GEOMETRY_SCHEMA = "round0232-streamed-spill-map-geometry-v1"
PROJECTION_SCHEMA = "round0232-scratch-and-cost-projection-v1"
PRODUCTION_CONFIG_SCHEMA = "round0232-streamed-spill-map-production-config-v1"

ARM_NAME = "streamed-spill"
ROWS = SUBSTRATE_2M_ROWS
SEEDS: tuple[int, ...] = (42, 43, 44)

ADOPTION_CLAIMED = False
EQUIVALENCE_CLAIMED = False
GATE_REGISTERABLE_HERE = False
GATE_RELEASE_CLAIMED = False
TRAINING_PERFORMED = False


def map_capability(seed: int) -> str:
    return f"minilm-mixed-2m-streamed-spill-map-seed{int(seed)}-v1"


# --------------------------------------------------------------------------- #
# the three designs
# --------------------------------------------------------------------------- #
MODE_MATERIALISE = "materialise"
MODE_STREAM_RESIDENT = "stream-resident"
MODE_STREAM_GATHER = "stream-gather"
MODES: tuple[str, ...] = (MODE_MATERIALISE, MODE_STREAM_RESIDENT, MODE_STREAM_GATHER)

DESIGN_NOTE = (
    "Total spill volume V = s x N x 1536 B is the same in every design; what "
    "differs is how much of V is resident at once. materialise: peak scratch "
    "<= bound, substrate reads G x N x 1536, spill I/O 2V. stream-resident: the "
    "same grouping with the spill file replaced by a host buffer of the same "
    "bound, so peak scratch is 0 and spill I/O is 0. stream-gather: no grouping "
    "at all, one cluster at a time taken by a direct ascending index gather from "
    "the substrate memmap, so peak scratch is 0, spill I/O is 0, and the bytes "
    "read are V rather than G x N x 1536."
)

#: The 1.23 TB figure that reached this round as if it were a disk requirement.
#: It is the total spill VOLUME at 100M s = 8, not peak scratch, and this
#: constant exists so the distinction is testable rather than rhetorical.
SPILL_VOLUME_100M_S8_BYTES = 100_000_000 * 8 * DIMENSION * 4


# --------------------------------------------------------------------------- #
# the disk axis of the guard — new in this round
# --------------------------------------------------------------------------- #
#: Peak scratch this round will hold on `/data` at any instant, across all cells.
ROUND_SCRATCH_BUDGET_BYTES = 100 * 10 ** 9
#: `/data` free bytes that must remain after a cell's predicted peak scratch.
#: Every other service on this box shares the volume; a round that fills it takes
#: them all down.
DISK_FREE_RESERVE_BYTES = 150 * 10 ** 9
DATA_ROOT = "/data"
#: The filesystem sampler's period. Fast enough to catch the end of a spill-write
#: phase, cheap enough that it never competes with the build.
SCRATCH_SAMPLE_INTERVAL_S = 0.05
#: The child aborts itself cooperatively if measured on-disk scratch exceeds its
#: own bound by more than this many bytes on top of the largest cluster. The
#: allowance exists because a group is packed whole-cluster, so the last cluster
#: admitted to a group can carry the group past the bound by construction, and
#: `pack_clusters_into_groups` documents that.
SCRATCH_ABORT_SLACK_BYTES = 2 * 1024 ** 3

DISK_GUARD_NOTE = (
    "predicted peak scratch <= 100 GB round-wide, and /data free bytes after the "
    "prediction >= 150 GB. Free space is re-read immediately before every cell so "
    "a concurrent consumer of the volume cannot be run over. A refusal is data: "
    "it is recorded with its prediction as refused_a_priori and the cell is not "
    "launched."
)


def data_free_bytes(path: str = DATA_ROOT) -> int:
    """Bytes available to an unprivileged writer on the volume holding `path`."""
    stat = os.statvfs(path)
    return int(stat.f_bavail) * int(stat.f_frsize)


def predicted_peak_scratch_bytes(
    *, rows: int, clusters: int, spill: int, mode: str, bound_bytes: int,
    imbalance: float,
) -> int:
    """Peak on-disk scratch this cell can hold at any instant, before it runs.

    The streamed modes never open a scratch file, so their prediction is exactly
    zero. The materialising mode is bounded by the packing, and the bound can be
    exceeded by at most one cluster because clusters are packed whole.
    """
    if mode not in MODES:
        raise Round0232Error(f"R0232 mode {mode!r} is not registered")
    if mode in (MODE_STREAM_RESIDENT, MODE_STREAM_GATHER):
        return 0
    volume = int(rows) * int(spill) * DIMENSION * 4
    mean_cluster_bytes = volume / float(clusters)
    largest_cluster_bytes = int(math.ceil(mean_cluster_bytes * float(imbalance)))
    return int(min(volume, int(bound_bytes) + largest_cluster_bytes))


def predicted_resident_host_bytes(
    *, rows: int, clusters: int, spill: int, mode: str, bound_bytes: int,
    imbalance: float,
) -> int:
    """Extra host-anonymous bytes the streamed modes hold that the spill file held.

    This is the whole cost of driving scratch to zero and it is charged, not
    hidden: `stream-resident` holds a group in RAM, `stream-gather` holds one
    cluster.
    """
    volume = int(rows) * int(spill) * DIMENSION * 4
    mean_cluster_bytes = volume / float(clusters)
    largest_cluster_bytes = int(math.ceil(mean_cluster_bytes * float(imbalance)))
    if mode == MODE_MATERIALISE:
        return 0
    if mode == MODE_STREAM_GATHER:
        return largest_cluster_bytes
    return int(min(volume, int(bound_bytes) + largest_cluster_bytes))


#: The guard's imbalance model is R0227's and is used for prediction only; every
#: published figure uses this round's own measured imbalance.
GUARD_IMBALANCE_FOR_PREDICTION = 2.2


def disk_guard(
    *, rows: int, clusters: int, spill: int, mode: str, bound_bytes: int,
    free_bytes: int | None = None,
    scratch_budget_bytes: int = ROUND_SCRATCH_BUDGET_BYTES,
    reserve_bytes: int = DISK_FREE_RESERVE_BYTES,
    imbalance: float = GUARD_IMBALANCE_FOR_PREDICTION,
) -> dict[str, Any]:
    """Refuse, before launch, any cell that could put `/data` at risk."""
    free = data_free_bytes() if free_bytes is None else int(free_bytes)
    peak = predicted_peak_scratch_bytes(
        rows=rows, clusters=clusters, spill=spill, mode=mode,
        bound_bytes=bound_bytes, imbalance=imbalance,
    )
    host_extra = predicted_resident_host_bytes(
        rows=rows, clusters=clusters, spill=spill, mode=mode,
        bound_bytes=bound_bytes, imbalance=imbalance,
    )
    over_budget = peak > int(scratch_budget_bytes)
    over_reserve = (free - peak) < int(reserve_bytes)
    reasons: list[str] = []
    if over_budget:
        reasons.append(
            f"predicted peak scratch {peak / 1e9:.2f} GB exceeds the round budget "
            f"{int(scratch_budget_bytes) / 1e9:.2f} GB"
        )
    if over_reserve:
        reasons.append(
            f"predicted peak scratch {peak / 1e9:.2f} GB would leave "
            f"{(free - peak) / 1e9:.2f} GB free on /data, below the "
            f"{int(reserve_bytes) / 1e9:.2f} GB reserve"
        )
    return {
        "mode": mode,
        "bound_bytes": int(bound_bytes),
        "predicted_peak_scratch_bytes": peak,
        "predicted_extra_host_anon_bytes": host_extra,
        "spill_volume_bytes": int(rows) * int(spill) * DIMENSION * 4,
        "data_free_bytes_at_guard": free,
        "data_free_bytes_after_prediction": free - peak,
        "round_scratch_budget_bytes": int(scratch_budget_bytes),
        "disk_free_reserve_bytes": int(reserve_bytes),
        "guard_imbalance_for_prediction": float(imbalance),
        "over_round_scratch_budget": bool(over_budget),
        "under_disk_free_reserve": bool(over_reserve),
        "allowed": not (over_budget or over_reserve),
        "refused_a_priori": bool(over_budget or over_reserve),
        "refusal_reasons": reasons,
        "note": DISK_GUARD_NOTE,
    }


def cell_guard(cell: Mapping[str, Any], *, free_bytes: int | None = None) -> dict[str, Any]:
    """Device, host and disk, all three, with a refusal on any one of them."""
    rows = int(cell["rows"])
    clusters = int(cell["clusters"])
    spill = int(cell["spill"])
    device = guard_for_spill(rows=rows, clusters=clusters, spill=spill)
    disk = disk_guard(
        rows=rows, clusters=clusters, spill=spill, mode=str(cell["mode"]),
        bound_bytes=int(cell["bound_bytes"]), free_bytes=free_bytes,
    )
    host_after = (
        int(device["prediction"]["predicted_host_anon_bytes"])
        + int(disk["predicted_extra_host_anon_bytes"])
    )
    host_over = host_after > GUARD_HOST_ANON_BUDGET_BYTES
    reasons = list(device.get("refusal_reasons") or []) + list(
        disk.get("refusal_reasons") or []
    )
    if host_over:
        reasons.append(
            f"predicted host anonymous {host_after / 1024 ** 3:.2f} GiB including "
            f"the streamed residency exceeds the "
            f"{GUARD_HOST_ANON_BUDGET_BYTES / 1024 ** 3:.2f} GiB budget"
        )
    return {
        "cell": str(cell["cell"]),
        "device_guard": device,
        "disk_guard": disk,
        "predicted_host_anon_bytes_including_residency": host_after,
        "host_over_budget_including_residency": bool(host_over),
        "device_budget_bytes": GUARD_DEVICE_BUDGET_BYTES,
        "host_anon_budget_bytes": GUARD_HOST_ANON_BUDGET_BYTES,
        "swap_growth_abort_bytes": GUARD_SWAP_GROWTH_ABORT_BYTES,
        "cluster_capacity_rows": CLUSTER_CAPACITY_ROWS,
        "allowed": bool(device.get("allowed")) and bool(disk.get("allowed"))
        and not host_over,
        "refused_a_priori": not (
            bool(device.get("allowed")) and bool(disk.get("allowed"))
            and not host_over
        ),
        "refusal_reasons": reasons,
        "axes": ["device", "host-anonymous", "disk"],
    }


# --------------------------------------------------------------------------- #
# the grids
# --------------------------------------------------------------------------- #
#: The arm's nn-descent setting, so wall is comparable across every cell and the
#: device-law refit lands at the setting R0229 recommended.
ARM_GRAPH_DEGREE = 64
ARM_INTERMEDIATE_DEGREE = 256
ARM_MAX_ITERATIONS = 40

_GIB = 1024 ** 3


def _cell(
    name: str, rows: int, clusters: int, spill: int, mode: str, bound: int,
    *, scored: bool, note: str,
) -> dict[str, Any]:
    return {
        "cell": name,
        "rows": int(rows),
        "clusters": int(clusters),
        "spill": int(spill),
        "mode": mode,
        "bound_bytes": int(bound),
        "graph_degree": ARM_GRAPH_DEGREE,
        "intermediate_graph_degree": ARM_INTERMEDIATE_DEGREE,
        "max_iterations": ARM_MAX_ITERATIONS,
        "scored_against_exact_truth": bool(scored),
        "note": note,
    }


GRID_A: tuple[dict[str, Any], ...] = (
    _cell("a1", 2_000_000, 200, 8, MODE_MATERIALISE, SCRATCH_BUDGET_BYTES,
          scored=True, note="R0229's arm reproduced: the reference cell"),
    _cell("a2", 2_000_000, 200, 8, MODE_MATERIALISE, 4 * _GIB,
          scored=True, note="the bound moved at fixed (N, c, s) — P1 is tested here"),
    _cell("a3", 2_000_000, 200, 8, MODE_MATERIALISE, 2 * _GIB,
          scored=True, note="the bound moved again; group count should roughly double"),
    _cell("a4", 2_000_000, 200, 8, MODE_STREAM_RESIDENT, 4 * _GIB,
          scored=True, note="the mandate's design: a4 against a2 at the same bound"),
    _cell("a5", 2_000_000, 200, 8, MODE_STREAM_GATHER, 0,
          scored=True, note="the arm's streamed twin; the graph the map arm trains on"),
    _cell("a6", 2_000_000, 64, 8, MODE_MATERIALISE, SCRATCH_BUDGET_BYTES,
          scored=True, note="c moved: larger clusters, coarser packing"),
    _cell("a7", 2_000_000, 64, 8, MODE_STREAM_GATHER, 0,
          scored=True, note="c moved, streamed"),
    _cell("a8", 2_000_000, 200, 2, MODE_MATERIALISE, SCRATCH_BUDGET_BYTES,
          scored=True, note="s moved: a quarter of the spill volume"),
    _cell("a9", 2_000_000, 200, 2, MODE_STREAM_GATHER, 0,
          scored=True, note="s moved, streamed"),
    _cell("a10", 1_000_000, 200, 8, MODE_MATERIALISE, SCRATCH_BUDGET_BYTES,
          scored=False, note="N moved; no exact truth below 2M, so unscored"),
    _cell("a11", 500_000, 200, 8, MODE_MATERIALISE, SCRATCH_BUDGET_BYTES,
          scored=False, note="N moved again; unscored"),
)

GRID_B: tuple[dict[str, Any], ...] = (
    _cell("b1", 8_000_000, 64, 8, MODE_MATERIALISE, SCRATCH_BUDGET_BYTES,
          scored=False,
          note="multi-group at a real N: 98.3 GB of spill volume against a 24 GiB "
               "bound, and the device-law refit point at gd 64 / igd 256"),
    _cell("b2", 8_000_000, 64, 8, MODE_STREAM_GATHER, 0,
          scored=False,
          note="the same cell streamed; the cold-cache I/O comparison"),
)

#: The cell whose graph the map arm consumes.
ARM_CELL = "a5"
#: The cell it must be byte-identical to if the restructure is recall-neutral.
ARM_REFERENCE_CELL = "a1"

#: Byte-identity is asserted only within a matched `(rows, clusters, spill)`
#: family, because a different partition is a different reachable set.
IDENTITY_FAMILIES: tuple[tuple[str, ...], ...] = (
    ("a1", "a2", "a3", "a4", "a5"),
    ("a6", "a7"),
    ("a8", "a9"),
    ("b1", "b2"),
)

SUBSTRATE_BY_ROWS: dict[int, str] = {
    500_000: SUBSTRATE_2M_PATH,
    1_000_000: SUBSTRATE_2M_PATH,
    2_000_000: SUBSTRATE_2M_PATH,
    8_000_000: SUBSTRATE_16M_PATH,
}

RECALL_POPULATION = "all-2000000-substrate-rows"
RECALL_POPULATION_NOTE = (
    "strict and tie-aware recall are computed over ALL 2,000,000 rows of R0216's "
    "sealed substrate, never over a seed union, a hub-biased sample or the "
    "builder's own accumulator (review-0216-01, review-0227-01). The substrate "
    "contains exact-duplicate clusters, one with 1,377 members, so strict "
    "understates and both figures are published."
)

#: Floors the streamed arm must clear, from R0229's sealed graph manifest.
ARM_TIE_AWARE_FLOOR = 0.998
ARM_STRICT_FLOOR = 0.997
R0229_ARM_TIE_AWARE = 0.9982163
R0229_ARM_STRICT = 0.9971441
R0229_ARM_ROWS_CARRYING_LOSS = 40_930
R0229_ARM_DIRECTED_EDGES = 48_348_096
R0229_ARM_MODELLED_PEAK_SCRATCH_BYTES = 24_576_000_000
R0229_ARM_SPILL_GROUPS = 1
R0229_ARM_MAX_CLUSTER_ROWS = 170_504


# --------------------------------------------------------------------------- #
# inference discipline, carried from review-0229-01 as a registered constant
# --------------------------------------------------------------------------- #
DISPLACEMENT_ALPHA = 0.05
#: `C(11, 3) = 165` labellings; the smallest attainable one-sided p is `1/165`.
PERMUTATION_LABELLINGS = 165
PERMUTATION_RESOLUTION_CEILING = 1.0 / 165.0
#: review-0229-01 §2, derived by binary search on the candidate gaps: the
#: smallest shift reaching `p <= 0.05` is `+0.21 sd` on top of the observed
#: `+0.766`, so the design's minimum detectable effect is `+0.98` exact-null sd.
MINIMUM_DETECTABLE_DISPLACEMENT_SD = 0.98
NON_REJECTION_NOTE = (
    "this design can only reject a displacement of +0.98 exact-null sd or "
    "larger. A non-rejection therefore licenses exactly one statement: the "
    "displacement is SMALLER THAN ABOUT ONE exact-family sd. It does not license "
    "'equivalent', 'clean', or 'like c = 4' — c = 4 sits at +0.027 sd, 2.8% of "
    "the detection threshold, and R0229's spill-lifted arm sat at +0.766 sd, 78% "
    "of it (review-0229-01 §2)."
)


def licensed_statement(*, p_value: float, alpha: float = DISPLACEMENT_ALPHA) -> str:
    if float(p_value) <= float(alpha):
        return (
            "REJECTED: the arm's displacement is detectably larger than the exact "
            "family's at alpha = 0.05"
        )
    return (
        "NOT REJECTED: the arm's displacement is smaller than about one "
        f"exact-family sd ({MINIMUM_DETECTABLE_DISPLACEMENT_SD} sd minimum "
        "detectable effect). This is NOT a claim of equivalence."
    )


# --------------------------------------------------------------------------- #
# the measured scratch law
# --------------------------------------------------------------------------- #
def scratch_law(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Bytes of on-disk scratch per resident spilled row, from measurement.

    The model is not fitted: a spilled row is `DIMENSION x 4` bytes on disk and
    the only question is how many of them are resident at once. This function
    checks that against every measured cell and reports the worst disagreement,
    so a wrong model shows up as a number rather than as an assumption.
    """
    row_bytes = DIMENSION * 4
    entries: list[dict[str, Any]] = []
    for cell in cells:
        if cell.get("measured_peak_scratch_bytes") is None:
            continue
        measured = int(cell["measured_peak_scratch_bytes"])
        modelled = int(cell.get("modelled_peak_scratch_bytes") or 0)
        resident_rows = measured / float(row_bytes)
        entries.append({
            "cell": str(cell["cell"]),
            "rows": int(cell["rows"]),
            "clusters": int(cell["clusters"]),
            "spill": int(cell["spill"]),
            "mode": str(cell["mode"]),
            "bound_bytes": int(cell["bound_bytes"]),
            "measured_peak_scratch_bytes": measured,
            "modelled_peak_scratch_bytes": modelled,
            "measured_minus_modelled_bytes": measured - modelled,
            "resident_spilled_rows_implied": resident_rows,
            "spill_volume_bytes": int(cell["rows"]) * int(cell["spill"]) * row_bytes,
            "fraction_of_volume_resident": (
                measured / float(int(cell["rows"]) * int(cell["spill"]) * row_bytes)
            ),
            "spill_groups": cell.get("spill_groups"),
            "substrate_passes": cell.get("substrate_passes"),
        })
    worst = max(
        (abs(int(entry["measured_minus_modelled_bytes"])) for entry in entries),
        default=0,
    )
    return {
        "model": (
            "peak on-disk scratch = (spilled rows resident at once) x DIMENSION x "
            "4 B, where the resident set is one spill group under `materialise` "
            "and empty under both streamed modes"
        ),
        "bytes_per_resident_spilled_row": row_bytes,
        "cells": entries,
        "worst_absolute_measured_minus_modelled_bytes": int(worst),
        "note": (
            "there is no fit here and no free parameter. The law is arithmetic; "
            "what this round contributes is that it was MEASURED against a "
            "filesystem sampler rather than computed from `sizes` and published "
            "as though it had been observed."
        ),
    }


# --------------------------------------------------------------------------- #
# the device law, refitted where R0229 did not refit it
# --------------------------------------------------------------------------- #
def linear_fit(xs: Sequence[float], ys: Sequence[float]) -> dict[str, Any]:
    points = [(float(x), float(y)) for x, y in zip(xs, ys)]
    if len(points) < 2:
        raise Round0232Error("R0232 linear fit needs at least two points")
    n = float(len(points))
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    sxx = sum((x - mean_x) ** 2 for x, _ in points)
    if sxx <= 0.0:
        raise Round0232Error("R0232 linear fit has no spread in the regressor")
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in points)
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in points)
    ss_tot = sum((y - mean_y) ** 2 for _, y in points)
    return {
        "model": "device_bytes = intercept + slope x max_cluster_rows",
        "slope_bytes_per_max_cluster_row": slope,
        "intercept_bytes": intercept,
        "r_squared": (1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
        "n_points": len(points),
        "fitted_range_max_cluster_rows": [
            min(x for x, _ in points), max(x for x, _ in points)
        ],
    }


def device_law_prediction(max_cluster_rows: float, fit: Mapping[str, Any]) -> dict[str, Any]:
    lo, hi = (float(v) for v in fit["fitted_range_max_cluster_rows"])
    value = (
        float(fit["intercept_bytes"])
        + float(fit["slope_bytes_per_max_cluster_row"]) * float(max_cluster_rows)
    )
    unrefitted = (
        DEVICE_LAW_INTERCEPT_BYTES
        + DEVICE_LAW_BYTES_PER_MAX_CLUSTER_ROW * float(max_cluster_rows)
    )
    return {
        "max_cluster_rows": float(max_cluster_rows),
        "refitted_bytes": value,
        "refitted_gib": value / 1024 ** 3,
        "r0227_unrefitted_bytes": unrefitted,
        "r0227_unrefitted_gib": unrefitted / 1024 ** 3,
        "refit_minus_unrefitted_gib": (value - unrefitted) / 1024 ** 3,
        "fitted_range_max_cluster_rows": [lo, hi],
        "extrapolation_factor_beyond_fitted_max": float(max_cluster_rows) / hi,
        "is_extrapolation": bool(float(max_cluster_rows) > hi),
        "label": "PROJECTION",
    }


def capacity_rows_at_device_budget(
    fit: Mapping[str, Any], *, budget_bytes: int = GUARD_DEVICE_BUDGET_BYTES,
    safety_bytes: int = 1024 ** 3,
) -> int:
    """The largest cluster the refitted law admits inside the device budget."""
    slope = float(fit["slope_bytes_per_max_cluster_row"])
    if slope <= 0.0:
        raise Round0232Error("R0232 refitted device slope is not positive")
    return int(
        (float(budget_bytes) - float(fit["intercept_bytes"]) - float(safety_bytes))
        / slope
    )


# --------------------------------------------------------------------------- #
# the I/O law, from this round's own measured throughput
# --------------------------------------------------------------------------- #
def io_projection(
    *, rows: int, clusters: int, spill: int, mode: str, bound_bytes: int,
    imbalance: float, read_bytes_per_s: float, write_bytes_per_s: float,
) -> dict[str, Any]:
    """Bytes moved and seconds spent moving them, per design, as its own line."""
    row_bytes = DIMENSION * 4
    volume = int(rows) * int(spill) * row_bytes
    mean_cluster_rows = int(rows) * int(spill) / float(clusters)
    sizes = [int(round(mean_cluster_rows))] * int(clusters)
    sizes[0] = int(round(mean_cluster_rows * float(imbalance)))
    substrate_bytes = int(rows) * row_bytes
    if mode == MODE_STREAM_GATHER:
        groups = 0
        read = float(volume)
        write = 0.0
        access = "scattered ascending gather, one cluster at a time"
    else:
        groups = len(pack_clusters_into_groups(sizes, budget_bytes=int(bound_bytes)))
        read = float(groups) * float(substrate_bytes)
        write = 0.0 if mode == MODE_STREAM_RESIDENT else float(volume)
        if mode == MODE_MATERIALISE:
            read += float(volume)
        access = "sequential block sweep per group"
    seconds = read / float(read_bytes_per_s) + (
        write / float(write_bytes_per_s) if write else 0.0
    )
    return {
        "mode": mode,
        "bound_bytes": int(bound_bytes),
        "spill_groups": groups,
        "substrate_passes": groups,
        "substrate_bytes_per_pass": substrate_bytes,
        "spill_volume_bytes": volume,
        "bytes_read": read,
        "bytes_written": write,
        "total_bytes_moved": read + write,
        "access_pattern": access,
        "read_bytes_per_s": float(read_bytes_per_s),
        "write_bytes_per_s": float(write_bytes_per_s),
        "seconds": seconds,
        "hours": seconds / 3600.0,
        "peak_scratch_bytes": predicted_peak_scratch_bytes(
            rows=rows, clusters=clusters, spill=spill, mode=mode,
            bound_bytes=bound_bytes, imbalance=imbalance,
        ),
        "extra_host_anon_bytes": predicted_resident_host_bytes(
            rows=rows, clusters=clusters, spill=spill, mode=mode,
            bound_bytes=bound_bytes, imbalance=imbalance,
        ),
        "label": "PROJECTION",
        "note": (
            "the gather figure is bytes REQUESTED, not bytes the block layer "
            "reads: a scattered ascending gather touches whole pages, so the "
            "realised read amplification is measured in this round rather than "
            "modelled, and the measured value is published beside this line"
        ),
    }


# --------------------------------------------------------------------------- #
# the deliverable
# --------------------------------------------------------------------------- #
def ladder_disk_requirement(
    *, rows: int, peak_scratch_bytes: int, k: int = GRAPH_K,
    dimension: int = DIMENSION,
) -> dict[str, Any]:
    """Every byte the rung needs on `/data` at once, not just the scratch."""
    substrate = int(rows) * int(dimension) * 4
    neighbour_ids = int(rows) * int(k) * 4
    neighbour_cos = int(rows) * int(k) * 4
    # The symmetrised fuzzy graph: R0229's 2M arm produced 48,348,096 directed
    # edges from 2,000,000 rows, i.e. 24.174 edges/row at (int32, int32, float32).
    fuzzy_edges = int(round(int(rows) * 24.174))
    fuzzy_bytes = fuzzy_edges * 12
    total = substrate + neighbour_ids + neighbour_cos + fuzzy_bytes + int(
        peak_scratch_bytes
    )
    return {
        "rows": int(rows),
        "substrate_bytes": substrate,
        "neighbour_ids_bytes": neighbour_ids,
        "neighbour_cosines_bytes": neighbour_cos,
        "fuzzy_edges_projected": fuzzy_edges,
        "fuzzy_edge_file_bytes": fuzzy_bytes,
        "peak_scratch_bytes": int(peak_scratch_bytes),
        "total_bytes_at_peak": total,
        "total_gb_at_peak": total / 1e9,
        "fuzzy_edges_per_row_source": (
            "R0229's sealed 2M arm manifest: 48,348,096 directed edges over "
            "2,000,000 rows = 24.174 per row; carried as a rate, labelled a "
            "projection, and not re-derived from a model"
        ),
        "label": "PROJECTION",
    }


__all__ = [
    "ADOPTION_CLAIMED",
    "ARM_CELL",
    "ARM_GRAPH_DEGREE",
    "ARM_INTERMEDIATE_DEGREE",
    "ARM_MAX_ITERATIONS",
    "ARM_NAME",
    "ARM_REFERENCE_CELL",
    "ARM_STRICT_FLOOR",
    "ARM_TIE_AWARE_FLOOR",
    "BUILD_SCHEMA",
    "DATA_ROOT",
    "DESIGN_NOTE",
    "DIMENSION",
    "DISK_FREE_RESERVE_BYTES",
    "DISK_GUARD_NOTE",
    "DISPLACEMENT_ALPHA",
    "EQUIVALENCE_CLAIMED",
    "GATE_REGISTERABLE_HERE",
    "GATE_RELEASE_CLAIMED",
    "GEOMETRY_CAPABILITY",
    "GEOMETRY_SCHEMA",
    "GRAPH_CAPABILITY",
    "GRAPH_K",
    "GRAPH_SCHEMA",
    "GRID_A",
    "GRID_B",
    "GRID_CAPABILITY",
    "GRID_SCHEMA",
    "IDENTITY_FAMILIES",
    "LARGER_N_CAPABILITY",
    "LARGER_N_SCHEMA",
    "MINIMUM_DETECTABLE_DISPLACEMENT_SD",
    "MODES",
    "MODE_MATERIALISE",
    "MODE_STREAM_GATHER",
    "MODE_STREAM_RESIDENT",
    "NON_REJECTION_NOTE",
    "PERMUTATION_LABELLINGS",
    "PERMUTATION_RESOLUTION_CEILING",
    "PRODUCTION_CONFIG_SCHEMA",
    "PROJECTION_CAPABILITY",
    "PROJECTION_SCHEMA",
    "R0229_ARM_DIRECTED_EDGES",
    "R0229_ARM_MAX_CLUSTER_ROWS",
    "R0229_ARM_MODELLED_PEAK_SCRATCH_BYTES",
    "R0229_ARM_ROWS_CARRYING_LOSS",
    "R0229_ARM_SPILL_GROUPS",
    "R0229_ARM_STRICT",
    "R0229_ARM_TIE_AWARE",
    "RECALL_POPULATION",
    "RECALL_POPULATION_NOTE",
    "ROUND_ID",
    "ROUND_SCRATCH_BUDGET_BYTES",
    "ROWS",
    "Round0232Error",
    "SCRATCH_ABORT_SLACK_BYTES",
    "SCRATCH_SAMPLE_INTERVAL_S",
    "SEEDS",
    "SPILL_VOLUME_100M_S8_BYTES",
    "SUBSTRATE_16M_PATH",
    "SUBSTRATE_16M_ROWS",
    "SUBSTRATE_2M_PATH",
    "SUBSTRATE_BY_ROWS",
    "TRAINING_PERFORMED",
    "TRAIN_SCHEMA",
    "TRUTH_COS_PATH",
    "TRUTH_IDS_PATH",
    "TRUTH_RECEIPT_PATH",
    "capacity_rows_at_device_budget",
    "cell_guard",
    "data_free_bytes",
    "device_law_prediction",
    "disk_guard",
    "io_projection",
    "ladder_disk_requirement",
    "licensed_statement",
    "linear_fit",
    "map_capability",
    "predicted_peak_scratch_bytes",
    "predicted_resident_host_bytes",
    "scratch_law",
]
