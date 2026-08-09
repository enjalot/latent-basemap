"""Frozen contract for R0233 — the 6.25M rung: substrate, reserves, k15 graph.

Phase 2's first rung of `../guides/plan-minilm-100m-v2.md`. Three things happen,
and nothing else: assemble a 6,250,000-row mixed universe with held-out reserves,
build its k15 fuzzy graph with `cluster-spill-nnd` at `s = 8`, and qualify that
graph against brute-force truth over **every** row. No map is trained here — the
standing rule from R0216 is that bundling assembly with graph construction drags
a data discovery through four queues.

Everything registered here is a reaction to a measured defect:

* **Selection spans, never prefixes.** R0216's first correction walked shards in
  order and stopped at quota, silently sampling the leading 90.87-94.16% of each
  corpus. The law here is uniform over ALL complete rows with replacement rounds
  drawn from the unpicked complement, and a `>= 99.9%` per-corpus shard-coverage
  assertion that RAISES. The assertion matters more than the fix: the defect is
  invisible in the output.
* **Damaged fineweb shard 37 is excluded whole** (24.89% exactly-zero rows), as
  are exactly-zero and non-finite rows everywhere. Imported from R0216 rather
  than restated so a drift cannot hide in a copy.
* **Held-out reserves exist.** R0222 could gate only four of six accepted
  metrics because Phase 1's substrate had no reserve, so `heldout_recall_at_10`
  and `projection_ffr` were not computable at all. Phase 0-C called for reserves
  and Phase 1 never built one. This round builds one per training corpus,
  disjoint from the training rows, with its own provenance.
* **`c` comes from MEASURED imbalance at THIS N.** Review-0227-01 found R0227
  published `c = 22` at 100M while its own artifact recorded the imbalance source
  as a model; the measured answer was `c = 24`. Every `c` in this round is chosen
  from an imbalance realised by an actual k-means + assign on this substrate.
* **`C_MIN = 2 * s`.** R0229's per-rung table reads `c = 8` at `(6.25M, s = 8)`
  with a projected largest cluster of `7,596,838` rows — larger than the whole
  rung. At `c = s` every row lands in every cluster, mean cluster rows equal `N`
  and nothing is partitioned, so the imbalance it borrowed (R0227's `1.215494`,
  measured at `s = 2`) does not describe that configuration at all. R0227's own
  floor was `C_MIN = 4` at `s = 2`, i.e. `2s`; this round carries the same rule
  forward to `s = 8` and records the correction.

## Safety preconditions, registered as contract, not as style

Both GPU wedges in this program were NVIDIA UVM page-fault deadlocks on driver
`570.211.01`. R0232's wedge had device memory at baseline and no signal sent; the
only delta against three working cells was an anonymous `ndarray` instead of a
read-only `np.memmap`. cuVS allocates through managed memory, so the GPU takes
replayable faults against whatever host buffer it is handed: file-backed pages
map cheaply, anonymous pages force fault-driven migration.

1. **Every buffer handed to cuVS is a read-only `np.memmap`, including every
   intermediate one.** `assert_memmap_for_cuvs` is called on the substrate view
   and on each per-cluster spill file immediately before the build, and it
   raises. R0232 was commissioned without that clause for intermediates.
2. **No signal is ever delivered to a process inside a cuVS build.** Review-0232
   located the deadlock in the *signal-triggered exit path*, so cooperative
   termination was itself sufficient to wedge the card. The abort path here is a
   flag file the child polls between clusters; the parent never calls
   `terminate()` or `kill()`.
3. **A wedged GPU is never probed.** A diagnostic allocation against a stuck UVM
   lock joins the deadlock.
4. **Predictive guard before every cell** — device, host anonymous, and disk —
   with `refused_a_priori` recorded as data rather than as an absence.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from basemap.round0216_minilm_2m_substrate import (
    DIMENSION,
    EXCLUDED_SHARDS,
    GRAPH_K,
    KNOWN_TRAILING_FRAGMENTS,
    RAW_FORMAT,
    ROW_POLICY,
    TRAILING_FRAGMENT_POLICY,
    ZERO_ROW_POLICY,
    resolve_shard_rows,
)
from basemap.round0227_low_c_contract import (
    DEVICE_LAW_BYTES_PER_MAX_CLUSTER_ROW,
    DEVICE_LAW_INTERCEPT_BYTES,
    GUARD_HOST_ANON_BUDGET_BYTES,
    GUARD_SAFETY_BYTES,
    GUARD_SWAP_GROWTH_ABORT_BYTES,
    SAMPLE_INTERVAL_S,
    SCRATCH_BUDGET_BYTES,
    WATCHDOG_POLL_S,
    linear_fit,
    pack_clusters_into_groups,
)


ROUND_ID = "0233"

SUBSTRATE_CAPABILITY = "minilm-mixed-6250k-substrate-and-reserves-v1"
TRUTH_CAPABILITY = "minilm-mixed-6250k-exact-k15-truth-v1"
GRAPH_CAPABILITY = "minilm-mixed-6250k-cluster-spill-k15-fuzzy-graph-v1"
LADDER_CAPABILITY = "minilm-mixed-6250k-cluster-spill-build-ladder-v1"

SUBSTRATE_SCHEMA = "round0233-minilm-mixed-6250k-substrate-v1"
TRUTH_SCHEMA = "round0233-minilm-mixed-6250k-exact-k15-truth-v1"
LADDER_SCHEMA = "round0233-minilm-mixed-6250k-build-ladder-v1"
BUILD_SCHEMA = "round0233-minilm-mixed-6250k-cluster-spill-build-v1"
GRAPH_SCHEMA = "round0233-minilm-mixed-6250k-k15-fuzzy-graph-v1"
LAW_SCHEMA = "round0233-device-law-refit-and-per-rung-derivation-v1"

ROWS = 6_250_000
SELECTION_SEED = 233

#: The owner-confirmed 40/25/25/10 shares of `plan-minilm-100m-v2.md`, at this
#: rung's exact row counts. Scaling R0216's registered law by 3.125x.
COMPOSITION: tuple[tuple[str, int], ...] = (
    ("fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2", 2_500_000),
    ("RedPajama-Data-V2-sample-10B-chunked-120-all-MiniLM-L6-v2", 1_562_500),
    ("pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2", 1_562_500),
    ("starcoderdata-code-chunked-120-all-MiniLM-L6-v2", 625_000),
)
TARGET_SHARES = {name: n / ROWS for name, n in COMPOSITION}

#: R0216's fail-closed span assertion, carried verbatim. `>= 99.9%` of each
#: corpus's non-excluded shards must be touched by the selection or the node
#: raises before anything is written.
SHARD_COVERAGE_FLOOR = 0.999
MAX_REPLACEMENT_ROUNDS = 8

# --------------------------------------------------------------------------- #
# held-out reserves — the thing Phase 1 never built
# --------------------------------------------------------------------------- #
#: R0108's registered held-out probe size, per training corpus. R0222 recorded
#: `heldout_recall_at_10` and `projection_ffr` as UNAVAILABLE rather than
#: excluded, because neither is computable without a reserve. R0108's
#: `fixed_probe_split` semantics are reused unchanged: `RESERVE_ROWS_PER_CORPUS`
#: rows per corpus, of which `RESERVE_QUERY_ROWS` are the held-out queries and
#: the remainder is the held-out corpus side.
RESERVE_ROWS_PER_CORPUS = 50_000
RESERVE_QUERY_ROWS = 500
RESERVE_CORPUS_ROWS = RESERVE_ROWS_PER_CORPUS - RESERVE_QUERY_ROWS
RESERVE_SEED = 233_000
RESERVE_ROWS = RESERVE_ROWS_PER_CORPUS * len(COMPOSITION)
RESERVE_NOTE = (
    "one reserve per training corpus, drawn uniformly from the complement of "
    "the training picks after the training quota is met, so it is disjoint by "
    "construction and the disjointness is asserted on provenance as well. Split "
    "into a 49,500-row held-out corpus side and a 500-row held-out query side "
    "per corpus, R0108's registered sizes, so heldout_recall_at_10 and "
    "projection_ffr become computable on this universe."
)

# --------------------------------------------------------------------------- #
# the builder — R0229's adopted arm, with c the only free knob
# --------------------------------------------------------------------------- #
CANDIDATE = "cluster-spill-nnd"
#: Adopted with spill raised 2 -> 8. At `(c = 200, s = 8)` on 2M this reached
#: tie-aware `0.998216` over all 2,000,000 rows with `2.0465%` of rows carrying
#: any loss and a displacement `p = 0.109` (`c4-LIKE` against R0228's registered
#: threshold). At `s = 2` the high-`c` arms displace at `p = 1/165` with complete
#: separation, so raising `s` is mandatory, not optional.
SPILL = 8
GRAPH_DEGREE = 64
INTERMEDIATE_GRAPH_DEGREE = 256
MAX_ITERATIONS = 40
NN_DESCENT_SETTING = "nnd-gd64-igd256-it40"

#: `c = s` puts every row in every cluster: mean cluster rows equal `N` and
#: nothing is partitioned. R0227 registered `C_MIN = 4` at `s = 2`; the general
#: rule is `2s`, and at `s = 8` that is 16.
C_MIN = 2 * SPILL
C_MIN_NOTE = (
    "C_MIN = 2 * spill. R0229's per-rung table selected c = 8 at (6.25M, s = 8) "
    "with a projected largest cluster of 7,596,838 rows, which exceeds N. That "
    "cell is degenerate — at c = s every row joins every cluster — and the "
    "imbalance it borrowed was measured at s = 2. Corrected here."
)

#: The ladder, in ASCENDING predicted max-cluster rows, which is the resource
#: axis the device law charges and therefore the axis along which a cell stops
#: being launchable. A refusal, abort or failure stops the ladder: a larger
#: cluster of a configuration that already failed cannot succeed.
LADDER_CLUSTERS: tuple[int, ...] = (200, 64, 32, 16)

#: Imbalance `max/mean` measured at `s = 8` on the 2M substrate by R0229's spill
#: grid — the ONLY `s = 8` imbalance measurements that exist. Used by the
#: pre-launch guard only; every published figure and the `c` selection itself use
#: this round's own imbalance, realised on THIS substrate at THIS N.
R0229_MEASURED_IMBALANCE_S8: dict[int, float] = {
    16: 1.1834,
    64: 1.5390,
    200: 2.1311,
}
#: `(32, 4)` is the nearest measured neighbour for `c = 32`; it is a different
#: `s`, and it is labelled as such wherever it is used.
R0229_MEASURED_IMBALANCE_NEAR: dict[int, float] = {32: 1.4237}
IMBALANCE_GUARD_MARGIN = 1.1648840
IMBALANCE_GUARD_NOTE = (
    "R0229's upper-end scaling. Imbalance at fixed c is noisy in N — R0227's "
    "c = 16 ladder gave 1.2742 / 1.3295 / 1.2374 / 1.4842 at 2M / 4M / 8M / 16M "
    "— so a point estimate is the wrong instrument for a guard and the upper end "
    "is carried instead."
)

# --------------------------------------------------------------------------- #
# the device law, as a PRIOR to be refitted — this round owes the refit
# --------------------------------------------------------------------------- #
#: R0227's law, fitted at `graph_degree 32 / igd 48` over max-cluster rows
#: `104,931 - 9,221,222`: `4.6467e9 B + 1560.9 B x max_cluster_rows`, worst
#: residual `0.93%` above 600k rows. It does NOT hold at `gd = 64`.
PRIOR_LAW_INTERCEPT_BYTES = DEVICE_LAW_INTERCEPT_BYTES
PRIOR_LAW_BYTES_PER_ROW = DEVICE_LAW_BYTES_PER_MAX_CLUSTER_ROW

#: Every `gd = 64` device measurement that exists, from sealed receipts. Two
#: cluster sizes only — which is exactly why review-0229-01 asked for a refit and
#: why R0232's 8M point never running left it owed.
GD64_PRIOR_POINTS: tuple[tuple[int, float], ...] = (
    (170_504, 7_469_921_270.0),   # R0229 arm, 6.957 GiB device-wide over baseline
    (318_519, 7_957_488_459.0),   # R0229 q7,  7.396-7.411 GiB, gd 64 / igd 256
)
GD64_PRIOR_NOTE = (
    "a two-point sanity bound, deliberately NOT called a law: slope ~3294 B/row "
    "with intercept ~6.91e9 B, i.e. 2.1x R0227's gd-32 slope. Review-0229-01 "
    "measured the gd-64 shortfall against R0227's law at +2.38 GiB on the arm "
    "and R0232 reproduced the arm's device peak to 0.027%, so the shortfall is "
    "stable but its dependence on cluster size is unmeasured. This round's "
    "largest cluster is roughly 12x anything ever built at gd = 64, which is the "
    "missing lever arm."
)
#: The guard charges the conservative end of that two-point bound, rounded up,
#: so a guard error cannot walk the card while the law is still unknown.
GUARD_BYTES_PER_MAX_CLUSTER_ROW = 3400.0
GUARD_INTERCEPT_BYTES = 7.0e9

DEVICE_TOTAL_BYTES = 31.37 * 1024 ** 3
GUARD_DEVICE_BUDGET_BYTES = 24 * 1024 ** 3
GPU_HOURS_CAP = 3.0
#: Per-cell deadline. R0229's 2M `(200, 8)` cell built in `225.6 s` over
#: `16,000,000` spilled rows; this rung spills `50,000,000`, so a cell is
#: expected near `700 s` and this is a `3.8x` margin. On expiry the parent sets
#: the cooperative flag and waits — it never signals.
BUILD_TIMEOUT_S = 2_700.0

#: The largest cluster the conservative gd-64 guard admits at the 24 GiB budget.
CLUSTER_CAPACITY_ROWS = int(
    (GUARD_DEVICE_BUDGET_BYTES - GUARD_INTERCEPT_BYTES - GUARD_SAFETY_BYTES)
    / GUARD_BYTES_PER_MAX_CLUSTER_ROW
)

#: Disk. Peak spill scratch is bounded structurally by `pack_clusters_into_groups`
#: at `1536 B` per resident spilled row, measured by R0232 to within one `.npy`
#: header per resident cluster file. Review-0232 corrected the peak law from
#: `bound + largest_cluster` to `max(bound, largest single cluster)`, because a
#: group is flushed before admitting a cluster that would cross the budget.
DISK_FREE_FLOOR_BYTES = 60 * 1000 ** 3
#: `/data` cold sequential read, measured by review-0232-01 with
#: `posix_fadvise(DONTNEED)`: `5.90 GB/s` on a 4-extent file against
#: `1.24 GB/s` on a 395-extent one. Budget from the fragmented rate; the
#: unfragmented rate is published beside it, never instead of it.
DATA_READ_FRAGMENTED_BYTES_PER_S = 1.24e9
DATA_READ_CONTIGUOUS_BYTES_PER_S = 5.90e9
DATA_WRITE_BYTES_PER_S = 2.52e9
IO_NOTE = (
    "I/O hours = (substrate_passes * N * 1536 + spill * N * 1536) / read_rate + "
    "spill * N * 1536 / write_rate. The read rate reported first is the "
    "FRAGMENTED 1.24 GB/s review-0232-01 measured on a 395-extent file, because "
    "a large rung artifact written on this volume reads like that file and not "
    "like a 3 GB one. The contiguous 5.90 GB/s rate is published beside it."
)

# --------------------------------------------------------------------------- #
# qualification
# --------------------------------------------------------------------------- #
#: Uniform over EVERY row. Not a seed set, not a neighbour union, not a
#: hub-biased sample — review-0227-01 measured that a seed-union population reads
#: `0.993918` where the uniform truth is `0.98897`, because being someone's exact
#: neighbour is a size-biased event.
RECALL_POPULATION = (
    "all 6,250,000 substrate rows, uniform; no seed set, no neighbour union"
)
TRUTH_METHOD = "brute-force fp32 cosine top-k over the full substrate, all rows"
RECALL_MEAN_FLOOR = 0.90
RECALL_P10_FLOOR = 0.80
#: The R0215 tripwire. v1 carried `2,779,481` edgeless rows and that is what
#: produced its clumps. Zero is the only acceptable number.
MAX_ZERO_DEGREE_ROWS = 0
DENSITY_DECILES = 10
TIE_TOLERANCE = 1e-6
FUZZY_RANDOM_STATE_SEED = 42

#: cuVS `nn_descent` is NOT run-to-run deterministic — R0220, R0223, R0229 and
#: R0232 all record it, and R0232's three cells over a bit-identical partition
#: produced three different graph hashes differing on `0.15%` of rows. No byte
#: identity is registered anywhere in this round; neutrality is scored on recall.
DETERMINISM_NOTE = (
    "no byte-identity prediction is registered for any cuVS graph in this round"
)

PHASE2_RUNGS: tuple[int, ...] = (
    6_250_000, 12_500_000, 25_000_000, 50_000_000, 100_000_000,
)


class Round0233Error(RuntimeError):
    """The registered R0233 contract changed."""


# --------------------------------------------------------------------------- #
# safety preconditions
# --------------------------------------------------------------------------- #
def assert_memmap_for_cuvs(array: Any, *, label: str) -> None:
    """Refuse to hand cuVS anything but a read-only, file-backed `np.memmap`.

    This is a machine-safety precondition, not an optimisation. It applies to
    INTERMEDIATE buffers too: R0232 wedged the box with an anonymous `ndarray`
    assembled by a streaming builder, and its round spec never forbade it.
    """
    if not isinstance(array, np.memmap):
        raise Round0233Error(
            f"{label} is a {type(array).__name__}, not an np.memmap. cuVS "
            "allocates through managed memory and takes replayable UVM faults "
            "against whatever host buffer it is given; anonymous pages force "
            "fault-driven migration and that is what wedged this box in R0232."
        )
    if array.flags.writeable:
        raise Round0233Error(f"{label} memmap is writeable; cuVS gets read-only bytes")
    if not array.flags.c_contiguous:
        raise Round0233Error(f"{label} memmap is not C-contiguous")


def assert_no_signal_policy(escalations: Sequence[str]) -> None:
    """Fail closed if any signal reached a build process."""
    forbidden = [
        item for item in escalations
        if "SIGTERM" in str(item) or "SIGKILL" in str(item)
    ]
    if forbidden:
        raise Round0233Error(
            f"R0233 forbids signalling a cuVS build; escalations={forbidden}. "
            "Review-0232 located the deadlock in the signal-triggered exit path, "
            "so cooperative termination is itself sufficient to wedge the card."
        )


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def validate_composition(counts: Mapping[str, int]) -> dict[str, Any]:
    """Fail closed unless the assembled universe is exactly the registered mix."""
    total = sum(int(value) for value in counts.values())
    if total != ROWS:
        raise Round0233Error(f"substrate has {total} rows, registered {ROWS}")
    observed: dict[str, Any] = {}
    for name, want in COMPOSITION:
        got = int(counts.get(name, 0))
        if got != want:
            raise Round0233Error(f"{name}: assembled {got} rows, registered {want}")
        observed[name] = {
            "rows": got,
            "share": got / ROWS,
            "registered_share": TARGET_SHARES[name],
        }
    return observed


def validate_shard_span(
    *, corpus: str, shards_touched: int, shards_total: int
) -> dict[str, Any]:
    """R0216's span assertion. It RAISES; the defect is invisible otherwise."""
    if shards_total <= 0:
        raise Round0233Error(f"{corpus}: no shards")
    coverage = shards_touched / float(shards_total)
    if coverage < SHARD_COVERAGE_FLOOR:
        raise Round0233Error(
            f"{corpus}: selection touched {shards_touched}/{shards_total} shards "
            f"({coverage:.4%}), below the registered {SHARD_COVERAGE_FLOOR:.1%} "
            "floor. The registered law requires the sample to SPAN the corpus; "
            "an oversample-then-stop-at-quota loop produces a leading prefix and "
            "R0216 shipped one undetected."
        )
    return {
        "shards_touched": int(shards_touched),
        "shards_total": int(shards_total),
        "coverage": float(coverage),
        "floor": SHARD_COVERAGE_FLOOR,
    }


def reserve_split(corpus_index: int) -> tuple[np.ndarray, np.ndarray]:
    """R0108's `fixed_probe_split` semantics on this corpus's reserve block.

    Returns positions WITHIN that corpus's reserve block: the held-out corpus
    side and the held-out query side, disjoint and deterministic.
    """
    rng = np.random.RandomState(RESERVE_SEED + int(corpus_index))
    selected = rng.choice(
        RESERVE_ROWS_PER_CORPUS, size=RESERVE_ROWS_PER_CORPUS, replace=False
    ).astype(np.int64)
    corpus_side = np.sort(selected[:RESERVE_CORPUS_ROWS])
    query_side = np.sort(selected[RESERVE_CORPUS_ROWS:])
    if (
        corpus_side.size != RESERVE_CORPUS_ROWS
        or query_side.size != RESERVE_QUERY_ROWS
        or np.intersect1d(corpus_side, query_side).size != 0
    ):
        raise Round0233Error("R0233 reserve split does not close")
    return corpus_side, query_side


# --------------------------------------------------------------------------- #
# c selection, from MEASURED imbalance
# --------------------------------------------------------------------------- #
def mean_cluster_rows(*, rows: int, clusters: int, spill: int = SPILL) -> float:
    return float(rows) * float(spill) / float(clusters)


def max_cluster_rows_from_imbalance(
    *, rows: int, clusters: int, imbalance: float, spill: int = SPILL
) -> float:
    return mean_cluster_rows(rows=rows, clusters=clusters, spill=spill) * float(
        imbalance
    )


def guard_admissible_max_cluster_rows(
    *, device_budget_bytes: int = GUARD_DEVICE_BUDGET_BYTES
) -> float:
    return (
        float(device_budget_bytes) - GUARD_INTERCEPT_BYTES - float(GUARD_SAFETY_BYTES)
    ) / GUARD_BYTES_PER_MAX_CLUSTER_ROW


def device_bytes_from_guard(max_cluster: float) -> float:
    return (
        GUARD_INTERCEPT_BYTES
        + GUARD_BYTES_PER_MAX_CLUSTER_ROW * float(max_cluster)
        + float(GUARD_SAFETY_BYTES)
    )


def select_clusters(
    *,
    rows: int,
    measured_imbalance: Mapping[int, float],
    candidates: Sequence[int] = LADDER_CLUSTERS,
    spill: int = SPILL,
    c_min: int = C_MIN,
    device_budget_bytes: int = GUARD_DEVICE_BUDGET_BYTES,
) -> dict[str, Any]:
    """The smallest admissible `c`, from imbalance MEASURED at this N.

    Fewer, larger clusters are strictly better for reachability, so the chosen
    configuration is the smallest `c` the device budget admits. `c` is never
    interpolated and never modelled: a candidate without a measured imbalance is
    not selectable.
    """
    rows = int(rows)
    admissible = guard_admissible_max_cluster_rows(
        device_budget_bytes=device_budget_bytes
    )
    considered: list[dict[str, Any]] = []
    chosen: dict[str, Any] | None = None
    for clusters in sorted(int(value) for value in candidates):
        entry: dict[str, Any] = {"clusters": int(clusters)}
        if clusters < int(c_min):
            entry.update({
                "admissible": False,
                "reason": (
                    f"c = {clusters} is below C_MIN = {c_min} = 2 * spill; at "
                    "c <= s every row lands in every cluster and nothing is "
                    "partitioned"
                ),
            })
            considered.append(entry)
            continue
        if int(clusters) not in measured_imbalance:
            entry.update({
                "admissible": False,
                "reason": "no imbalance measured at this N for this c",
            })
            considered.append(entry)
            continue
        imbalance = float(measured_imbalance[int(clusters)])
        largest = max_cluster_rows_from_imbalance(
            rows=rows, clusters=clusters, imbalance=imbalance, spill=spill
        )
        fits = largest <= admissible
        entry.update({
            "admissible": bool(fits),
            "measured_imbalance": imbalance,
            "imbalance_source": "measured on this substrate at this N",
            "mean_cluster_rows": mean_cluster_rows(
                rows=rows, clusters=clusters, spill=spill
            ),
            "max_cluster_rows": float(largest),
            "admissible_max_cluster_rows": float(admissible),
            "guard_device_bytes": device_bytes_from_guard(largest),
        })
        considered.append(entry)
        if fits and chosen is None:
            chosen = entry
    if chosen is None:
        raise Round0233Error(
            f"R0233 found no admissible c at {rows} rows, s = {spill}: "
            f"{considered}"
        )
    return {
        "rows": rows,
        "spill": int(spill),
        "c_min": int(c_min),
        "c_min_note": C_MIN_NOTE,
        "selected_clusters": int(chosen["clusters"]),
        "selection": chosen,
        "candidates_considered": considered,
        "admissible_max_cluster_rows": float(admissible),
        "rule": (
            "the smallest c >= 2*spill whose MEASURED largest cluster at this N "
            "fits the conservative gd-64 device guard; c is never interpolated"
        ),
    }


def guard_decision(
    *,
    rows: int,
    clusters: int,
    imbalance: float,
    imbalance_source: str,
    spill: int = SPILL,
    device_budget_bytes: int = GUARD_DEVICE_BUDGET_BYTES,
    host_anon_budget_bytes: int = GUARD_HOST_ANON_BUDGET_BYTES,
    disk_free_bytes: int | None = None,
) -> dict[str, Any]:
    """Launch this cell, or refuse it and record the refusal as a measurement."""
    rows = int(rows)
    clusters = int(clusters)
    row_bytes = DIMENSION * 4
    guarded_imbalance = float(imbalance) * IMBALANCE_GUARD_MARGIN
    largest = max_cluster_rows_from_imbalance(
        rows=rows, clusters=clusters, imbalance=guarded_imbalance, spill=spill
    )
    device_bytes = device_bytes_from_guard(largest)

    topk_bytes = rows * GRAPH_K * 8
    assignment_bytes = rows * spill * 4
    member_bytes = rows * spill * 8 * 3
    stage_bytes = 500_000 * row_bytes
    local_graph_bytes = int(largest) * GRAPH_DEGREE * 8
    local_cosine_bytes = int(largest) * GRAPH_DEGREE * 4
    anonymous_bytes = int(
        topk_bytes
        + assignment_bytes
        + member_bytes
        + stage_bytes
        + local_graph_bytes
        + local_cosine_bytes
        + 3 * 1024 ** 3
    )

    # Disk: peak scratch is max(bound, largest single cluster) — review-0232's
    # correction to `bound + largest`, because `pack_clusters_into_groups`
    # flushes a group before admitting a cluster that would cross the budget.
    peak_scratch_bytes = int(max(
        min(SCRATCH_BUDGET_BYTES, rows * spill * row_bytes),
        int(largest) * row_bytes,
    ))

    device_over = device_bytes > float(device_budget_bytes)
    host_over = anonymous_bytes > int(host_anon_budget_bytes)
    capacity_over = largest > float(CLUSTER_CAPACITY_ROWS)
    below_c_min = clusters < C_MIN
    disk_over = (
        disk_free_bytes is not None
        and (peak_scratch_bytes + DISK_FREE_FLOOR_BYTES) > int(disk_free_bytes)
    )

    reasons: list[str] = []
    if below_c_min:
        reasons.append(f"c = {clusters} is below C_MIN = {C_MIN} (= 2 * spill)")
    if device_over:
        reasons.append(
            f"predicted device {device_bytes / 1024 ** 3:.2f} GiB exceeds the "
            f"{device_budget_bytes / 1024 ** 3:.2f} GiB budget"
        )
    if host_over:
        reasons.append(
            f"predicted host anonymous {anonymous_bytes / 1024 ** 3:.2f} GiB "
            f"exceeds the {host_anon_budget_bytes / 1024 ** 3:.2f} GiB budget"
        )
    if capacity_over:
        reasons.append(
            f"predicted largest cluster {largest:.0f} exceeds the registered "
            f"capacity {CLUSTER_CAPACITY_ROWS}"
        )
    if disk_over:
        reasons.append(
            f"predicted peak scratch {peak_scratch_bytes / 1000 ** 3:.2f} GB "
            f"plus the {DISK_FREE_FLOOR_BYTES / 1000 ** 3:.0f} GB free floor "
            f"exceeds the {int(disk_free_bytes) / 1000 ** 3:.2f} GB available"
        )
    return {
        "prediction": {
            "candidate": CANDIDATE,
            "rows": rows,
            "clusters": clusters,
            "spill": int(spill),
            "measured_imbalance": float(imbalance),
            "imbalance_source": str(imbalance_source),
            "guarded_imbalance": guarded_imbalance,
            "imbalance_guard_margin": IMBALANCE_GUARD_MARGIN,
            "imbalance_guard_note": IMBALANCE_GUARD_NOTE,
            "predicted_mean_cluster_rows": mean_cluster_rows(
                rows=rows, clusters=clusters, spill=spill
            ),
            "predicted_max_cluster_rows": float(largest),
            "predicted_device_bytes": float(device_bytes),
            "predicted_device_gib": float(device_bytes) / 1024 ** 3,
            "predicted_host_anon_bytes": int(anonymous_bytes),
            "predicted_host_anon_gib": anonymous_bytes / 1024 ** 3,
            "predicted_peak_scratch_bytes": peak_scratch_bytes,
            "prior_law_device_bytes": float(
                PRIOR_LAW_INTERCEPT_BYTES + PRIOR_LAW_BYTES_PER_ROW * largest
            ),
            "gd64_prior_note": GD64_PRIOR_NOTE,
        },
        "device_budget_bytes": int(device_budget_bytes),
        "host_anon_budget_bytes": int(host_anon_budget_bytes),
        "cluster_capacity_rows": int(CLUSTER_CAPACITY_ROWS),
        "disk_free_bytes": None if disk_free_bytes is None else int(disk_free_bytes),
        "device_over_budget": bool(device_over),
        "host_over_budget": bool(host_over),
        "capacity_over_budget": bool(capacity_over),
        "disk_over_budget": bool(disk_over),
        "below_c_min": bool(below_c_min),
        "allowed": not (
            device_over or host_over or capacity_over or disk_over or below_c_min
        ),
        "refused_a_priori": bool(
            device_over or host_over or capacity_over or disk_over or below_c_min
        ),
        "refusal_reasons": reasons,
    }


# --------------------------------------------------------------------------- #
# the refit this round owes, and what it re-derives
# --------------------------------------------------------------------------- #
def refit_device_law(
    *, max_cluster_rows: Sequence[int], device_bytes: Sequence[float]
) -> dict[str, Any]:
    """Refit `device = intercept + slope x max_cluster_rows` on THIS round's cells.

    Registered by R0229 and not done; still owed after R0232's 8M refit point
    never ran. Reports the fitted range and every residual, plus the prior law's
    residuals on the same points so the reader can see which describes the data.
    """
    fit = linear_fit(max_cluster_rows, device_bytes)
    largest = np.asarray(list(max_cluster_rows), dtype=np.float64)
    measured = np.asarray(list(device_bytes), dtype=np.float64)
    prior = PRIOR_LAW_INTERCEPT_BYTES + PRIOR_LAW_BYTES_PER_ROW * largest
    guard = np.asarray([device_bytes_from_guard(value) for value in largest])
    return {
        "refit": fit,
        "fitted_range_max_cluster_rows": [
            int(largest.min()), int(largest.max())
        ],
        "fitted_span_factor": float(largest.max() / max(1.0, largest.min())),
        "residual_bytes": [float(value) for value in fit["residuals"]],
        "residual_relative": [
            float(value / measured[index]) for index, value in
            enumerate(fit["residuals"])
        ],
        "worst_absolute_relative_residual": float(
            np.max(np.abs(np.asarray(fit["residuals"]) / measured))
        ),
        "prior_gd32_law_predicted_bytes": [float(value) for value in prior],
        "prior_gd32_law_relative_error": [
            float(value) for value in (measured - prior) / prior
        ],
        "guard_predicted_bytes": [float(value) for value in guard],
        "guard_is_conservative_everywhere": bool(np.all(guard >= measured)),
        "prior_law": {
            "intercept_bytes": PRIOR_LAW_INTERCEPT_BYTES,
            "bytes_per_max_cluster_row": PRIOR_LAW_BYTES_PER_ROW,
            "calibrated_at": "graph_degree 32 / intermediate_graph_degree 48",
            "source": "round0227_low_c_contract, from review-0226-01",
        },
        "setting": NN_DESCENT_SETTING,
        "note": (
            "refitted at gd 64 / igd 256 / it 40, the setting R0229 adopted and "
            "at which R0227's published law was never calibrated. Review-0229-01 "
            "measured its shortfall at +2.38 GiB on the arm."
        ),
    }


def io_projection(
    *,
    rows: int,
    substrate_passes: int,
    spill: int = SPILL,
    read_bytes_per_s: float = DATA_READ_FRAGMENTED_BYTES_PER_S,
    write_bytes_per_s: float = DATA_WRITE_BYTES_PER_S,
) -> dict[str, Any]:
    """The I/O term, from the realised group packing and a stated read rate."""
    row_bytes = DIMENSION * 4
    substrate_read = int(substrate_passes) * int(rows) * row_bytes
    spill_volume = int(rows) * int(spill) * row_bytes
    read_bytes = substrate_read + spill_volume
    seconds = read_bytes / float(read_bytes_per_s) + spill_volume / float(
        write_bytes_per_s
    )
    return {
        "rows": int(rows),
        "spill": int(spill),
        "substrate_passes": int(substrate_passes),
        "substrate_read_bytes": substrate_read,
        "spill_read_back_bytes": spill_volume,
        "spill_write_bytes": spill_volume,
        "read_bytes_per_s": float(read_bytes_per_s),
        "write_bytes_per_s": float(write_bytes_per_s),
        "io_seconds": float(seconds),
        "io_hours": float(seconds) / 3600.0,
        "note": IO_NOTE,
    }


def rung_derivation(
    *,
    rung: int,
    imbalance_by_c: Mapping[int, float],
    imbalance_source: str,
    law_intercept_bytes: float,
    law_bytes_per_row: float,
    spill: int = SPILL,
    device_budget_bytes: int = GUARD_DEVICE_BUDGET_BYTES,
    safety_bytes: float = float(GUARD_SAFETY_BYTES),
) -> dict[str, Any]:
    """Re-derive `c` for one Phase-2 rung under a stated law and stated imbalance.

    Never interpolates: a `c` without a measured imbalance is not selectable, and
    the reason is recorded rather than silently skipped.
    """
    admissible = (
        float(device_budget_bytes) - float(law_intercept_bytes) - float(safety_bytes)
    ) / float(law_bytes_per_row)
    considered: list[dict[str, Any]] = []
    chosen: dict[str, Any] | None = None
    for clusters in sorted(int(value) for value in imbalance_by_c):
        if clusters < 2 * int(spill):
            considered.append({
                "clusters": int(clusters), "admissible": False,
                "reason": f"below C_MIN = {2 * int(spill)}",
            })
            continue
        imbalance = float(imbalance_by_c[clusters])
        largest = max_cluster_rows_from_imbalance(
            rows=rung, clusters=clusters, imbalance=imbalance, spill=spill
        )
        device = float(law_intercept_bytes) + float(law_bytes_per_row) * largest
        entry = {
            "clusters": int(clusters),
            "imbalance": imbalance,
            "mean_cluster_rows": mean_cluster_rows(
                rows=rung, clusters=clusters, spill=spill
            ),
            "max_cluster_rows": float(largest),
            "device_bytes": device,
            "device_gib": device / 1024 ** 3,
            "admissible": bool(largest <= admissible),
        }
        considered.append(entry)
        if entry["admissible"] and chosen is None:
            chosen = entry
    return {
        "rung": int(rung),
        "spill": int(spill),
        "imbalance_source": str(imbalance_source),
        "admissible_max_cluster_rows": float(admissible),
        "selected_clusters": None if chosen is None else int(chosen["clusters"]),
        "selection": chosen,
        "candidates_considered": considered,
        "feasible": chosen is not None,
    }


def substrate_pass_count(
    *, rows: int, clusters: int, imbalance: float, spill: int = SPILL
) -> int:
    """Groups (= substrate re-reads) for an idealised size vector at this `(c, s)`."""
    mean = mean_cluster_rows(rows=rows, clusters=clusters, spill=spill)
    sizes = [int(round(mean))] * int(clusters)
    if sizes:
        sizes[0] = int(round(mean * float(imbalance)))
    return len(pack_clusters_into_groups(sizes))


__all__ = [
    "BUILD_SCHEMA",
    "BUILD_TIMEOUT_S",
    "CANDIDATE",
    "CLUSTER_CAPACITY_ROWS",
    "COMPOSITION",
    "C_MIN",
    "C_MIN_NOTE",
    "DATA_READ_CONTIGUOUS_BYTES_PER_S",
    "DATA_READ_FRAGMENTED_BYTES_PER_S",
    "DATA_WRITE_BYTES_PER_S",
    "DENSITY_DECILES",
    "DETERMINISM_NOTE",
    "DEVICE_TOTAL_BYTES",
    "DIMENSION",
    "DISK_FREE_FLOOR_BYTES",
    "EXCLUDED_SHARDS",
    "FUZZY_RANDOM_STATE_SEED",
    "GD64_PRIOR_NOTE",
    "GD64_PRIOR_POINTS",
    "GPU_HOURS_CAP",
    "GRAPH_CAPABILITY",
    "GRAPH_DEGREE",
    "GRAPH_K",
    "GRAPH_SCHEMA",
    "GUARD_BYTES_PER_MAX_CLUSTER_ROW",
    "GUARD_DEVICE_BUDGET_BYTES",
    "GUARD_HOST_ANON_BUDGET_BYTES",
    "GUARD_INTERCEPT_BYTES",
    "GUARD_SAFETY_BYTES",
    "GUARD_SWAP_GROWTH_ABORT_BYTES",
    "IMBALANCE_GUARD_MARGIN",
    "IMBALANCE_GUARD_NOTE",
    "INTERMEDIATE_GRAPH_DEGREE",
    "IO_NOTE",
    "KNOWN_TRAILING_FRAGMENTS",
    "LADDER_CAPABILITY",
    "LADDER_CLUSTERS",
    "LADDER_SCHEMA",
    "LAW_SCHEMA",
    "MAX_ITERATIONS",
    "MAX_REPLACEMENT_ROUNDS",
    "MAX_ZERO_DEGREE_ROWS",
    "NN_DESCENT_SETTING",
    "PHASE2_RUNGS",
    "PRIOR_LAW_BYTES_PER_ROW",
    "PRIOR_LAW_INTERCEPT_BYTES",
    "R0229_MEASURED_IMBALANCE_NEAR",
    "R0229_MEASURED_IMBALANCE_S8",
    "RAW_FORMAT",
    "RECALL_MEAN_FLOOR",
    "RECALL_P10_FLOOR",
    "RECALL_POPULATION",
    "RESERVE_CORPUS_ROWS",
    "RESERVE_NOTE",
    "RESERVE_QUERY_ROWS",
    "RESERVE_ROWS",
    "RESERVE_ROWS_PER_CORPUS",
    "RESERVE_SEED",
    "ROUND_ID",
    "ROWS",
    "ROW_POLICY",
    "Round0233Error",
    "SAMPLE_INTERVAL_S",
    "SCRATCH_BUDGET_BYTES",
    "SELECTION_SEED",
    "SHARD_COVERAGE_FLOOR",
    "SPILL",
    "SUBSTRATE_CAPABILITY",
    "SUBSTRATE_SCHEMA",
    "TARGET_SHARES",
    "TIE_TOLERANCE",
    "TRAILING_FRAGMENT_POLICY",
    "TRUTH_CAPABILITY",
    "TRUTH_METHOD",
    "TRUTH_SCHEMA",
    "WATCHDOG_POLL_S",
    "ZERO_ROW_POLICY",
    "assert_memmap_for_cuvs",
    "assert_no_signal_policy",
    "device_bytes_from_guard",
    "guard_admissible_max_cluster_rows",
    "guard_decision",
    "io_projection",
    "max_cluster_rows_from_imbalance",
    "mean_cluster_rows",
    "pack_clusters_into_groups",
    "refit_device_law",
    "reserve_split",
    "resolve_shard_rows",
    "rung_derivation",
    "select_clusters",
    "substrate_pass_count",
    "validate_composition",
    "validate_shard_span",
]
