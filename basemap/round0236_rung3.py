"""Frozen contract for R0236 — Phase 2 rung 3 (25M), nested on R0235's rung.

Three things happen and nothing else: assemble a 25,000,000-row mixed universe
that **contains** R0235's 12,500,000 training rows, build and qualify its k15
graph, and pay the two debts `review-0235-2026-08-09-01.md` left open. No map is
trained.

Every registered rule below answers a finding of review-0235-01:

* **B1 / F1 — 100M is not safely priced.** The 100M rung carried a 12.5M
  imbalance three doublings against a `7.43%` tolerance, while the largest
  single-doubling move the program has measured is `13.42%`. This round measures
  imbalance at 25M for the 50M and 100M `c` values (`128` and `200`) as well as
  its own, and re-derives every rung from the nearest measured `N` with the
  tolerance to adverse imbalance stated per rung.
* **B3 / F3 — drift is measured but not attributed.** Every prior point is one
  realisation per `(N, c)` against R0227's `20%` spread at fixed `c`. This round
  takes **three k-means seeds** at **three nested `N`** (6.25M / 12.5M / 25M),
  all read off prefixes of one substrate, so the draw channel and the `N`
  channel are separated by construction rather than by argument.
* **B4 / F6 — substrate reads are quadratic in `N` and unpriced.** Substrate
  passes equal the spill-group count, which is set by packing whole clusters
  into a fixed scratch budget, so architectural substrate reads grow as
  `N**2 * s * 1536**2 / budget`. Nobody has measured what fraction of that ever
  reaches the disk. This round instruments the build child's `/proc/<pid>/io`
  and the block layer, registers a **two-regime prediction** (a substrate that
  fits page cache is read once physically, one that does not is read every
  pass), and prices 50M and 100M in hours at rates measured on the file layouts
  this round actually writes.
* **F4 — one cell carried two different `host_anon_peak_bytes`.** The child
  samples `RssAnon` at phase boundaries; the parent watchdog polls it
  continuously and its reading overwrote the child's in the ladder record. Here
  the two live under **different names**, and the authoritative one is named in
  the artifact.

## Nesting — the Phase 2 design constraint

    T3 = T2  U  uniform_without_replacement(P \\ T2 \\ R1, n3 - n2)

`T2` is R0235's sealed training selection and `R1` the reserve R0233 drew and
R0235 inherited. Conditional on `R1`, `T2` is uniform over size-`n2` subsets of
`P \\ R1`, so adding a uniform draw from the complement makes `T3` **exactly** a
uniform size-25,000,000 subset of `P \\ R1`. Nesting is positional: rung-2 row
`i` is rung-3 row `i` for every `i < 12,500,000`, verified by hashing the prefix
against R0235's sealed `ordered_substrate_sha256`.

Because the prefix is byte-identical all the way down, `substrate[:6,250,000]`
is R0233's rung and `substrate[:12,500,000]` is R0235's. The imbalance replicate
grid is measured on those prefixes, so every point in the drift table comes off
one file and one code path.

## Truth at this rung

Exact brute-force truth over all 25,000,000 rows costs `4 x` R0235's
`4,223.74 s`, i.e. `4.69 GPU-h` against this round's `4.0` cap: it is not
affordable and the round says so rather than pretending. Qualification therefore
scores a **uniform probe of 1,000,000 query rows drawn without replacement over
all 25,000,000 row ids at a seed registered here, before the substrate exists**,
searched against **all** 25,000,000 rows by brute force. It is not a seed set,
not a neighbour union and not hub-biased; the only difference from R0235's
population is that 4% of rows carry the estimate. The R0215 degree-zero tripwire
and the structural checks still run over **every** row, in and out.

## Safety preconditions, carried unchanged

Both GPU wedges in this program were NVIDIA UVM page-fault deadlocks on driver
`570.211.01`. The build path is not copied: `basemap/round0236_build.py` calls
`basemap/round0235_build.py`, which calls `basemap/round0233_build.py` with one
named constant rebound. Both wrappers are fail-closed and neither installs a
signal handler.

1. Every buffer handed to cuVS is a read-only, C-contiguous `np.memmap`,
   including every intermediate spill file, asserted immediately before
   `nn_descent.build` receives it.
2. No signal is ever delivered to a build process. The abort path is a flag file
   the child polls; the parent never calls `terminate()`, `kill()` or `os.kill`.
3. A wedged GPU is never probed.
4. A predictive guard runs before every cell on device, host **anonymous** bytes
   and disk, with `refused_a_priori` recorded as data, and the watchdog trips on
   swap **growth** over a pre-launch baseline.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from basemap.round0227_low_c_contract import SCRATCH_BUDGET_BYTES
from basemap.round0235_rung2 import (
    BUILD_TIMEOUT_S as R0235_BUILD_TIMEOUT_S,
    CANDIDATE,
    C_MIN,
    DATA_READ_CONTIGUOUS_BYTES_PER_S,
    DATA_READ_FRAGMENTED_BYTES_PER_S,
    DATA_WRITE_BYTES_PER_S,
    DENSITY_DECILES,
    DETERMINISM_NOTE,
    DEVICE_TOTAL_BYTES,
    DIMENSION,
    DISK_FREE_FLOOR_BYTES,
    EXCLUDED_SHARDS,
    FUZZY_RANDOM_STATE_SEED,
    GRAPH_DEGREE,
    GRAPH_K,
    GUARD_DEVICE_BUDGET_BYTES,
    GUARD_HOST_ANON_BUDGET_BYTES,
    GUARD_IMBALANCE_MARGIN,
    GUARD_NOTE,
    GUARD_SAFETY_BYTES,
    GUARD_SWAP_GROWTH_ABORT_BYTES,
    INTERMEDIATE_GRAPH_DEGREE,
    IO_NOTE,
    KNOWN_TRAILING_FRAGMENTS,
    LAW_GRAPH_DEGREE,
    LAW_HOMOGENEITY_NOTE,
    LAW_INTERMEDIATE_GRAPH_DEGREE,
    LAW_MAX_ITERATIONS,
    LAW_RESIDUAL_MARGIN,
    MAX_ITERATIONS,
    MAX_REPLACEMENT_ROUNDS,
    MAX_ZERO_DEGREE_ROWS,
    NN_DESCENT_SETTING,
    PHASE2_RUNGS,
    RAW_FORMAT,
    RECALL_MEAN_FLOOR,
    RECALL_P10_FLOOR,
    RESERVE_CORPUS_ROWS,
    RESERVE_QUERY_ROWS,
    RESERVE_ROWS,
    RESERVE_ROWS_PER_CORPUS,
    RESERVE_SEED,
    ROW_POLICY,
    SAMPLE_INTERVAL_S,
    SHARD_COVERAGE_FLOOR,
    SPILL,
    TIE_TOLERANCE,
    TRAILING_FRAGMENT_POLICY,
    WATCHDOG_POLL_S,
    ZERO_ROW_POLICY,
    admissible_max_cluster_rows,
    admit_law_point,
    assert_memmap_for_cuvs,
    assert_no_signal_policy,
    fit_device_law,
    guard_decision,
    guard_device_bytes,
    guarded_max_cluster_rows,
    io_projection,
    law_device_bytes,
    mean_cluster_rows,
    pack_clusters_into_groups,
    resolve_shard_rows,
    select_clusters,
    substrate_pass_count,
)


ROUND_ID = "0236"

SUBSTRATE_CAPABILITY = "minilm-mixed-25000k-nested-substrate-and-reserves-v1"
TRUTH_CAPABILITY = "minilm-mixed-25000k-uniform-probe-k15-truth-v1"
LADDER_CAPABILITY = "minilm-mixed-25000k-cluster-spill-build-ladder-v1"
GRAPH_CAPABILITY = "minilm-mixed-25000k-cluster-spill-k15-fuzzy-graph-v1"
IMBALANCE_CAPABILITY = "minilm-mixed-cluster-spill-s8-imbalance-replicate-grid-v1"
IO_CAPABILITY = "cluster-spill-substrate-io-scaling-v1"

SUBSTRATE_SCHEMA = "round0236-minilm-mixed-25000k-nested-substrate-v1"
TRUTH_SCHEMA = "round0236-minilm-mixed-25000k-uniform-probe-k15-truth-v1"
LADDER_SCHEMA = "round0236-minilm-mixed-25000k-build-ladder-v1"
GRAPH_SCHEMA = "round0236-minilm-mixed-25000k-k15-fuzzy-graph-v1"
LAW_SCHEMA = "round0236-guarded-device-law-io-scaling-and-rung-rederivation-v1"

#: Rung 3 of `../guides/plan-minilm-100m-v2.md` Phase 2.
ROWS = 25_000_000
#: Rung 2. Every one of these rows is contained in `ROWS`, at the same position.
PARENT_ROWS = 12_500_000
PARENT_ROUND_ID = "0235"
#: Rung 1, which is R0235's own byte-identical prefix and therefore ours too.
GRANDPARENT_ROWS = 6_250_000
GRANDPARENT_ROUND_ID = "0233"

#: The owner-confirmed 40/25/25/10 shares at this rung's exact row counts.
COMPOSITION: tuple[tuple[str, int], ...] = (
    ("fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2", 10_000_000),
    ("RedPajama-Data-V2-sample-10B-chunked-120-all-MiniLM-L6-v2", 6_250_000),
    ("pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2", 6_250_000),
    ("starcoderdata-code-chunked-120-all-MiniLM-L6-v2", 2_500_000),
)
#: R0235's composition, which this round's prefix must reproduce exactly.
PARENT_COMPOSITION: tuple[tuple[str, int], ...] = (
    ("fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2", 5_000_000),
    ("RedPajama-Data-V2-sample-10B-chunked-120-all-MiniLM-L6-v2", 3_125_000),
    ("pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2", 3_125_000),
    ("starcoderdata-code-chunked-120-all-MiniLM-L6-v2", 1_250_000),
)
TARGET_SHARES = {name: n / ROWS for name, n in COMPOSITION}
INCREMENT_BY_CORPUS = {
    name: n - dict(PARENT_COMPOSITION)[name] for name, n in COMPOSITION
}

#: The increment's own seed. The prefix is not re-drawn: it is R0235's bytes.
SELECTION_SEED = 236
SELECTION_LAW = (
    "T3 = T2 U uniform_without_replacement(P \\ T2 \\ R1, n3 - n2), per corpus, "
    "at seed 236 + corpus index, with rejected rows replaced by fresh uniform "
    "draws from the unpicked complement until the increment quota is met. T2 is "
    "R0235's sealed training selection and R1 the reserve R0233 drew and R0235 "
    "inherited. Because (T2, R1) is a uniform ordered pair of disjoint sets, T3 "
    "is EXACTLY a uniform size-25,000,000 subset of P \\ R1. Never a prefix of "
    "the corpus; always a positional prefix of the substrate."
)
NESTING_NOTE = (
    "rung-2 row i is rung-3 row i for every i < 12,500,000. Verified by hashing "
    "the substrate prefix with ordered_array_sha256 and comparing against "
    "R0235's sealed ordered_substrate_sha256, and independently by set "
    "containment on packed (corpus, shard, row) provenance keys. Because R0235's "
    "own prefix is byte-identical to R0233's, substrate[:6,250,000] is rung 1."
)
RESERVE_NOTE = (
    "R0233's reserve, inherited through R0235 verbatim - the same 200,000 rows, "
    "50,000 per training corpus, R0108's 49,500 + 500 split - copied into this "
    "round's artifact tree and verified byte-identical by sha256. Those rows are "
    "EXCLUDED from rung 3's draw pool, exactly as they were from rung 2's, so "
    "one fixed eval set stays valid for the whole ladder and a rung-to-rung "
    "comparison is a comparison of N alone."
)

# --------------------------------------------------------------------------- #
# the builder — R0229's adopted arm, unchanged since R0233
# --------------------------------------------------------------------------- #
#: The `c` values whose imbalance is MEASURED at this N, unchanged from R0235's
#: probe set so the drift table is like-for-like. `128` and `200` are the values
#: the 50M and 100M rungs select and are the reason review-0235-01 asked for this
#: measurement; `400` is the priced 100M fallback.
IMBALANCE_PROBE_CLUSTERS: tuple[int, ...] = (16, 32, 64, 128, 200, 400)

#: **The replicate grid (B3).** Three k-means seeds at three nested `N`. Seed 226
#: is `A_SEED`, the seed every prior rung used, so the 226 realisation is the
#: like-for-like point in the drift table and the other two bound its spread.
#: `_kmeans` draws both its 1,000,000-row subsample and its initial centroids
#: from this seed, so a seed change is exactly the "draw changed" channel
#: review-0235-01 could not separate from the "N changed" channel.
IMBALANCE_REPLICATE_SEEDS: tuple[int, ...] = (226, 236, 1236)
PRIMARY_IMBALANCE_SEED = 226
#: Nested prefixes of THIS substrate. `substrate[:6,250,000]` is R0233's rung and
#: `substrate[:12,500,000]` is R0235's, byte-identically, so the three `N` differ
#: in nothing but their row count.
IMBALANCE_PROBE_ROWS: tuple[int, ...] = (6_250_000, 12_500_000, 25_000_000)
REPLICATE_NOTE = (
    "three k-means seeds at each of three nested N, all measured on prefixes of "
    "one substrate by R0226's _kmeans/_assign imported unmodified. The 226 "
    "realisation is the like-for-like point every prior rung reported; the "
    "spread across seeds at fixed N is the draw channel, and the movement of the "
    "226 series across N is the N channel. review-0235-01 F3 blocked attribution "
    "because no prior round had both."
)

#: The `c` values this round's own graph may be built at. R0235's derivation
#: prices 25M at `c = 64` with `65.6%` of tolerance to adverse imbalance, and
#: review-0235-01 answered the runner's question directly: `c = 32` misses the
#: budget by `1.377%` on an imbalance value that moved `13.42%` in the one
#: doubling we have for it, so building it would be "choosing a coin flip".
#: `c = 32` is still PRICED from this round's own measurement and published; it
#: is simply not a build candidate. `c = 200` is the registered fallback if this
#: round's own measured imbalance refuses `c = 64`.
SELECTION_CANDIDATES: tuple[int, ...] = (64, 200)
C_BUILD_MIN = 64
C_BUILD_MIN_NOTE = (
    "the c-selection law is unchanged - the smallest candidate whose GUARDED "
    "largest cluster fits the device budget under the binding law - but the "
    "candidate set is registered as (64, 200) rather than (16, 32, 64, 200). "
    "c = 16 and c = 32 are priced from this round's own measurement and "
    "published beside the selection; they are not built. review-0235-01 "
    "answered this question in advance."
)
#: No control cell. R0235's `c = 64` control settled the `N`-independence of the
#: device law at a matched cluster size and twice the `N`; this round's budget
#: buys the replicate grid and the I/O instrument instead.
CONTROL_CLUSTERS: tuple[int, ...] = ()
LADDER_RULE = (
    "one cell: the c the selection law picks from imbalance measured at THIS N "
    "at the primary seed, with the round's own imbalance margin applied. A "
    "refusal, abort or failure stops the ladder and is recorded as a "
    "measurement, with its GPU time charged to the round."
)

# --------------------------------------------------------------------------- #
# truth — a registered uniform probe, because full truth is not affordable
# --------------------------------------------------------------------------- #
#: Registered BEFORE the substrate exists. Uniform without replacement over all
#: 25,000,000 row ids. Never a seed set, never a neighbour union, never
#: hub-biased: review-0227-01 measured that a seed-union population reads
#: `0.993918` where the uniform truth is `0.98897`, because being someone's exact
#: neighbour is a size-biased event.
TRUTH_PROBE_ROWS = 1_000_000
TRUTH_PROBE_SEED = 236_000
TRUTH_METHOD = (
    "exact brute-force fp32 cosine top-k of a registered uniform probe of "
    "1,000,000 query rows against ALL 25,000,000 substrate rows"
)
RECALL_POPULATION = (
    "a uniform probe of 1,000,000 of the 25,000,000 substrate rows, drawn "
    "without replacement at seed 236000 registered before the substrate "
    "existed, searched against all 25,000,000 rows; no seed set, no neighbour "
    "union, no hub bias"
)
TRUTH_AFFORDABILITY_NOTE = (
    "full exact truth over all 25,000,000 rows scales as N^2: R0235 spent "
    "4,223.74 s at 12,500,000, so 25,000,000 costs about 16,895 s = 4.69 GPU-h "
    "against this round's 4.0 GPU-h cap for the whole queue. It is not "
    "affordable and this round does not pretend otherwise. The probe costs "
    "about 676 s of matmul at the same throughput."
)
#: Structural checks - degree, self-loops, duplicates, the R0215 tripwire - run
#: over EVERY row in and out. Only recall is estimated on the probe.
STRUCTURAL_POPULATION = "all 25,000,000 rows, both in-degree and out-degree"

# --------------------------------------------------------------------------- #
# the guard
# --------------------------------------------------------------------------- #
GPU_HOURS_CAP = 4.0
#: Per-cell deadline. R0233's cells ran ~640 s over 50,000,000 spilled rows and
#: R0235's ~1,300 s over 100,000,000; the phases scale linearly in spilled rows
#: (nn-descent 363 -> 742 s, merge 218 -> 446 s across that doubling), so this
#: rung's 200,000,000 spilled rows are expected near 2,700 s and this is a
#: `2.2x` margin. On expiry the parent sets the cooperative flag and waits - it
#: never signals.
BUILD_TIMEOUT_S = 6_000.0

# --------------------------------------------------------------------------- #
# the I/O instrument — review-0235-01 F6 / B4
# --------------------------------------------------------------------------- #
#: Page cache available to the build. The box has 123 GiB of RAM and the build's
#: own anonymous footprint at this rung is ~20 GiB, so 80 GB is a deliberately
#: conservative statement of what the kernel can keep resident. It exists to make
#: the two-regime prediction below FALSIFIABLE, not to model the kernel.
PAGE_CACHE_BUDGET_BYTES = 80 * 1000 ** 3
IO_REGIME_NOTE = (
    "architectural substrate reads are substrate_passes * N * 1536 and grow "
    "QUADRATICALLY in N, because substrate_passes is the spill-group count and "
    "that is N * s * 1536 / scratch_budget. How much of it reaches the disk is a "
    "different question and nobody has measured it. Registered prediction, in "
    "two regimes: a substrate that fits page cache is read from the block layer "
    "about ONCE however many passes the builder makes, and one that does not is "
    "read about every pass. At 25,000,000 rows the substrate is 38.4 GB and fits "
    "80 GB; at 100,000,000 it is 153.6 GB and cannot. If the prediction holds, "
    "the quadratic term is free below ~52M rows and fully charged above it, "
    "which is a sharper statement than either bound alone."
)
IO_INSTRUMENT_NOTE = (
    "measured three ways for every cell: the child's own /proc/<pid>/io "
    "(read_bytes and write_bytes are block-layer bytes and do NOT count "
    "page-cache hits; rchar and wchar are syscall bytes and do not count memmap "
    "faults), the system block layer via /proc/diskstats on the device backing "
    "/data, and the architectural counts the build receipt already emits. All "
    "three are published; none is derived from another."
)

# --------------------------------------------------------------------------- #
# the host-anonymous field, review-0235-01 F4
# --------------------------------------------------------------------------- #
HOST_ANON_AUTHORITATIVE_SAMPLER = "parent-watchdog-continuous"
HOST_ANON_FIELD_NOTE = (
    "review-0235-01 F4: `host_anon_peak_bytes` carried two different values for "
    "the same cell because the child samples RssAnon only at phase boundaries "
    "while the parent watchdog polls it every 0.25 s and its reading overwrote "
    "the child's. Here the two never share a name. "
    "`host_anon_peak_bytes_parent_watchdog` is the authoritative, conservative "
    "figure and is the one every claim in this round uses; "
    "`host_anon_peak_bytes_child_phase_sampled` is the child's, published beside "
    "it and never used for a decision. `host_anon_peak_bytes` is set equal to "
    "the parent's so a consumer that reads the old name gets the safe value. The "
    "per-cell build-receipt.json is written by R0233's reviewed child and still "
    "carries the child's phase-sampled figure under the bare name - that file is "
    "not rewritten, and this field says so."
)


class Round0236Error(RuntimeError):
    """The registered R0236 contract changed."""


# --------------------------------------------------------------------------- #
# composition, span, nesting — same shapes as R0235, at this rung's counts
# --------------------------------------------------------------------------- #
def validate_composition(counts: Mapping[str, int]) -> dict[str, Any]:
    """Fail closed unless the assembled universe is exactly the registered mix."""
    total = sum(int(value) for value in counts.values())
    if total != ROWS:
        raise Round0236Error(f"substrate has {total} rows, registered {ROWS}")
    observed: dict[str, Any] = {}
    for name, want in COMPOSITION:
        got = int(counts.get(name, 0))
        if got != want:
            raise Round0236Error(f"{name}: assembled {got} rows, registered {want}")
        observed[name] = {
            "rows": got,
            "share": got / ROWS,
            "registered_share": TARGET_SHARES[name],
            "inherited_from_rung2": dict(PARENT_COMPOSITION)[name],
            "newly_drawn": INCREMENT_BY_CORPUS[name],
        }
    return observed


def validate_shard_span(
    *, corpus: str, shards_touched: int, shards_total: int, label: str
) -> dict[str, Any]:
    """R0216's span assertion. It RAISES; the defect is invisible otherwise."""
    if shards_total <= 0:
        raise Round0236Error(f"{corpus}: no shards")
    coverage = shards_touched / float(shards_total)
    if coverage < SHARD_COVERAGE_FLOOR:
        raise Round0236Error(
            f"{corpus} [{label}]: selection touched {shards_touched}/"
            f"{shards_total} shards ({coverage:.4%}), below the registered "
            f"{SHARD_COVERAGE_FLOOR:.1%} floor. The registered law requires the "
            "sample to SPAN the corpus; an oversample-then-stop-at-quota loop "
            "produces a leading prefix and R0216 shipped one undetected."
        )
    return {
        "label": str(label),
        "shards_touched": int(shards_touched),
        "shards_total": int(shards_total),
        "coverage": float(coverage),
        "floor": SHARD_COVERAGE_FLOOR,
    }


def provenance_keys(records: np.ndarray) -> np.ndarray:
    """Pack `(corpus, shard, row)` into one int64 key per row."""
    corpus = np.asarray(records["corpus"], dtype=np.int64)
    shard = np.asarray(records["shard"], dtype=np.int64)
    row = np.asarray(records["row"], dtype=np.int64)
    if corpus.size and (
        int(corpus.max()) >= 256 or int(shard.max()) >= 65_536
        or int(row.min()) < 0 or int(row.max()) >= (1 << 40)
    ):
        raise Round0236Error("R0236 provenance does not fit the registered key")
    return (corpus << 56) | (shard << 40) | row


def assert_nesting(*, parent: np.ndarray, child: np.ndarray) -> dict[str, Any]:
    """Rung 3 must CONTAIN rung 2, on row ids, and every child row is distinct."""
    parent_keys = provenance_keys(parent)
    child_keys = provenance_keys(child)
    if int(np.unique(child_keys).size) != int(child_keys.size):
        raise Round0236Error("R0236 substrate holds a duplicated source row")
    missing = int(np.setdiff1d(parent_keys, child_keys, assume_unique=False).size)
    if missing != 0:
        raise Round0236Error(
            f"R0236 is not nested on R0235: {missing} rung-2 rows are absent. "
            "Phase 2's whole design is one variable per rung; a non-nested rung "
            "confounds N with the sample."
        )
    positional = bool(
        parent_keys.size <= child_keys.size
        and np.array_equal(parent_keys, child_keys[: parent_keys.size])
    )
    if not positional:
        raise Round0236Error(
            "R0236 prefix is not R0235's rows in R0235's order; the registered "
            "nesting is positional, not merely set-theoretic"
        )
    return {
        "parent_rows": int(parent_keys.size),
        "child_rows": int(child_keys.size),
        "parent_rows_missing_from_child": missing,
        "positional_prefix": positional,
        "child_rows_distinct": True,
        "note": NESTING_NOTE,
    }


def assert_reserve_disjoint(
    *, training: np.ndarray, reserve: np.ndarray
) -> dict[str, Any]:
    """The reserve must not intersect the training rows, globally and per corpus."""
    train_keys = provenance_keys(training)
    reserve_keys = provenance_keys(reserve)
    overlap = int(np.intersect1d(train_keys, reserve_keys).size)
    if overlap != 0:
        raise Round0236Error(
            f"R0236 reserve overlaps the training selection on {overlap} rows"
        )
    per_corpus: dict[str, Any] = {}
    for index, (corpus, _rows) in enumerate(COMPOSITION):
        mask_t = np.asarray(training["corpus"], dtype=np.int64) == index
        mask_r = np.asarray(reserve["corpus"], dtype=np.int64) == index
        per_corpus[corpus] = {
            "training_rows": int(mask_t.sum()),
            "reserve_rows": int(mask_r.sum()),
            "intersection_rows": int(
                np.intersect1d(train_keys[mask_t], reserve_keys[mask_r]).size
            ),
        }
        if per_corpus[corpus]["intersection_rows"] != 0:
            raise Round0236Error(f"R0236 reserve overlaps training in {corpus}")
    return {
        "global_intersection_rows": overlap,
        "per_corpus": per_corpus,
        "reserve_rows": int(reserve_keys.size),
        "note": RESERVE_NOTE,
    }


# --------------------------------------------------------------------------- #
# the registered truth probe
# --------------------------------------------------------------------------- #
def truth_probe_query_rows(
    *, rows: int = ROWS, size: int = TRUTH_PROBE_ROWS, seed: int = TRUTH_PROBE_SEED
) -> np.ndarray:
    """The registered uniform probe: `size` distinct row ids, ascending.

    Uniform without replacement over `range(rows)` at a seed fixed in this
    module before the substrate existed. Ascending order is for memmap locality
    and carries no information: the draw is exchangeable.
    """
    rows = int(rows)
    size = int(size)
    if size <= 0 or size > rows:
        raise Round0236Error(f"R0236 probe of {size} rows is not drawable from {rows}")
    rng = np.random.RandomState(int(seed))
    return np.sort(rng.choice(rows, size=size, replace=False)).astype(np.int64)


# --------------------------------------------------------------------------- #
# replicates and the drift table — review-0235-01 F3 / B3
# --------------------------------------------------------------------------- #
def replicate_summary(values: Mapping[int, float]) -> dict[str, Any]:
    """Spread of one `(N, c)` cell across k-means seeds.

    `spread_relative` is `(max - min) / mean`, the quantity review-0235-01 needed
    and did not have: R0227 measured a `20%` spread at fixed `c` across four
    substrates, which is larger than any drift in R0235's table.
    """
    seeds = sorted(int(key) for key in values)
    if not seeds:
        raise Round0236Error("R0236 replicate cell has no realisations")
    array = np.asarray([float(values[seed]) for seed in seeds], dtype=np.float64)
    mean = float(array.mean())
    return {
        "seeds": seeds,
        "values": [float(value) for value in array],
        "n": int(array.size),
        "mean": mean,
        "min": float(array.min()),
        "max": float(array.max()),
        "spread_absolute": float(array.max() - array.min()),
        "spread_relative": float((array.max() - array.min()) / mean) if mean else None,
        "sample_sd": float(array.std(ddof=1)) if array.size > 1 else None,
        "primary_seed": PRIMARY_IMBALANCE_SEED,
        "primary": (
            float(values[PRIMARY_IMBALANCE_SEED])
            if PRIMARY_IMBALANCE_SEED in values else None
        ),
    }


def replicate_drift(
    grid: Mapping[int, Mapping[int, Mapping[int, float]]],
    *,
    inherited: Mapping[int, Mapping[int, float]] | None = None,
) -> dict[str, Any]:
    """The drift table WITH an error bar, per `c`, across the measured `N`.

    `grid` is `{rows: {clusters: {seed: imbalance}}}` measured in this round.
    `inherited` is `{rows: {clusters: imbalance}}` from sealed artifacts of other
    rounds, which have exactly one realisation each and are reported as such.

    Two drift figures are published per `c` and neither is presented as the
    other: `drift_primary` on the seed-226 series, which is the like-for-like
    comparison every prior rung reported, and `drift_replicate_mean` on the
    across-seed means. `drift_exceeds_spread` says whether the movement is larger
    than the pooled within-`N` spread that produced it - the question F3 asked.
    """
    normalised: dict[int, dict[int, dict[int, float]]] = {
        int(rows): {
            int(clusters): {int(seed): float(value) for seed, value in seeds.items()}
            for clusters, seeds in cells.items()
        }
        for rows, cells in grid.items()
    }
    other = {
        int(rows): {int(c): float(v) for c, v in cells.items()}
        for rows, cells in (inherited or {}).items()
    }
    sizes = sorted(set(normalised) | set(other))
    all_c = sorted(
        {c for cells in normalised.values() for c in cells}
        | {c for cells in other.values() for c in cells}
    )
    by_c: dict[str, Any] = {}
    for clusters in all_c:
        row: dict[str, Any] = {"clusters": int(clusters), "by_rows": {}}
        primary_series: list[tuple[int, float]] = []
        mean_series: list[tuple[int, float]] = []
        spreads: list[float] = []
        for size in sizes:
            cell = normalised.get(size, {}).get(clusters)
            if cell is not None:
                summary = replicate_summary(cell)
                row["by_rows"][str(size)] = {
                    "replicated": True,
                    "source": "measured in R0236 on a nested prefix",
                    **summary,
                }
                if summary["primary"] is not None:
                    primary_series.append((size, float(summary["primary"])))
                mean_series.append((size, float(summary["mean"])))
                if summary["spread_relative"] is not None:
                    spreads.append(float(summary["spread_relative"]))
                continue
            value = other.get(size, {}).get(clusters)
            row["by_rows"][str(size)] = None if value is None else {
                "replicated": False,
                "n": 1,
                "source": "single realisation inherited from a sealed artifact",
                "primary": float(value),
                "mean": float(value),
                "spread_relative": None,
            }
            if value is not None:
                primary_series.append((size, float(value)))
                mean_series.append((size, float(value)))

        def _drift(series: Sequence[tuple[int, float]]) -> dict[str, Any]:
            if len(series) < 2:
                return {
                    "measured_at_rows": [int(n) for n, _v in series],
                    "drift_relative": None,
                    "insufficient_points": True,
                }
            base_n, base = series[0]
            top_n, top = series[-1]
            return {
                "measured_at_rows": [int(n) for n, _v in series],
                "drift_relative": (top - base) / base,
                "drift_span_rows": [int(base_n), int(top_n)],
                "monotone_increasing": bool(
                    all(b >= a for (_n1, a), (_n2, b) in zip(series, series[1:]))
                ),
            }

        pooled = float(max(spreads)) if spreads else None
        primary = _drift(primary_series)
        row.update({
            "drift_primary": primary,
            "drift_replicate_mean": _drift(mean_series),
            "worst_within_n_spread_relative": pooled,
            "drift_exceeds_spread": (
                None if pooled is None or primary.get("drift_relative") is None
                else bool(abs(float(primary["drift_relative"])) > pooled)
            ),
        })
        by_c[str(clusters)] = row
    return {
        "rows_measured": sizes,
        "replicate_seeds": [int(seed) for seed in IMBALANCE_REPLICATE_SEEDS],
        "primary_seed": PRIMARY_IMBALANCE_SEED,
        "by_clusters": by_c,
        "note": REPLICATE_NOTE,
        "reading": (
            "drift_relative is (imbalance at the largest measured N) / "
            "(imbalance at the smallest measured N) - 1; a positive value moves "
            "toward the device budget. drift_exceeds_spread is false when the "
            "movement across N is inside the spread the draw alone produces at "
            "fixed N, in which case the movement is NOT attributable to N."
        ),
    }


# --------------------------------------------------------------------------- #
# feasibility with its tolerance stated — review-0235-01 F1 / B1
# --------------------------------------------------------------------------- #
def imbalance_tolerance(
    *,
    rung: int,
    clusters: int,
    imbalance: float,
    laws: Sequence[Mapping[str, Any]],
    spill: int = SPILL,
    margin: float = GUARD_IMBALANCE_MARGIN,
    device_budget_bytes: int = GUARD_DEVICE_BUDGET_BYTES,
    residual_margin: float = LAW_RESIDUAL_MARGIN,
) -> dict[str, Any]:
    """How far imbalance may rise, BEYOND the margin already charged, and still fit.

    This is the number review-0235-01 computed by hand to show that 25M and 50M
    were robust (`65.6%`, `59.2%`) and 100M was not (`7.43%`, carried three
    doublings against a measured one-doubling move of `13.42%`). It is published
    per rung here rather than reconstructed by a reviewer.
    """
    guarded = guarded_max_cluster_rows(
        rows=int(rung), clusters=int(clusters), imbalance=float(imbalance),
        spill=int(spill), margin=float(margin),
    )
    admissible = admissible_max_cluster_rows(
        laws, device_budget_bytes=device_budget_bytes,
        residual_margin=residual_margin,
    )
    charge = guard_device_bytes(laws, guarded, residual_margin=residual_margin)
    return {
        "rung": int(rung),
        "clusters": int(clusters),
        "imbalance": float(imbalance),
        "imbalance_margin_applied": float(margin),
        "guarded_max_cluster_rows": float(guarded),
        "admissible_max_cluster_rows": float(admissible),
        "predicted_device_bytes": float(charge["predicted_device_bytes"]),
        "predicted_device_gib": float(charge["predicted_device_gib"]),
        "binding_law": charge["binding_law"],
        "admissible": bool(guarded <= admissible),
        "tolerance_to_adverse_imbalance": (
            float(admissible / guarded - 1.0) if guarded > 0 else None
        ),
        "note": (
            "tolerance is how much the TRUE imbalance at this rung may exceed "
            "the value carried here before the cell becomes inadmissible, on top "
            "of the margin already charged"
        ),
    }


def carry_distance(*, measured_at_rows: int, rung: int) -> dict[str, Any]:
    """Doublings of `N` between where imbalance was measured and where it is used."""
    measured_at_rows = int(measured_at_rows)
    rung = int(rung)
    if measured_at_rows <= 0 or rung <= 0:
        raise Round0236Error("R0236 carry distance needs positive row counts")
    return {
        "measured_at_rows": measured_at_rows,
        "rung": rung,
        "ratio": rung / measured_at_rows,
        "doublings_carried": float(np.log2(rung / measured_at_rows)),
        "measured_at_the_rung": bool(measured_at_rows == rung),
    }


# --------------------------------------------------------------------------- #
# the I/O model — review-0235-01 F6 / B4
# --------------------------------------------------------------------------- #
def architectural_io(
    *, rows: int, substrate_passes: int, spill: int = SPILL,
    dimension: int = DIMENSION,
) -> dict[str, Any]:
    """Bytes the builder's structure implies, independent of what the disk sees."""
    row_bytes = int(dimension) * 4
    substrate_bytes = int(rows) * row_bytes
    substrate_read = int(substrate_passes) * substrate_bytes
    spill_volume = int(rows) * int(spill) * row_bytes
    return {
        "rows": int(rows),
        "spill": int(spill),
        "substrate_passes": int(substrate_passes),
        "substrate_bytes": substrate_bytes,
        "substrate_read_bytes": substrate_read,
        "spill_write_bytes": spill_volume,
        "spill_read_back_bytes": spill_volume,
        "total_read_bytes": substrate_read + spill_volume,
        "total_write_bytes": spill_volume,
        "note": (
            "substrate_passes is the spill-group count, which is "
            "ceil-packing of whole clusters into the fixed "
            f"{SCRATCH_BUDGET_BYTES} B scratch budget, so it grows LINEARLY in N "
            "and substrate_read_bytes therefore grows QUADRATICALLY in N"
        ),
    }


def predicted_substrate_passes(
    *, rows: int, clusters: int, imbalance: float, spill: int = SPILL
) -> int:
    """Spill groups for an idealised size vector at this `(c, s)` and imbalance."""
    return int(substrate_pass_count(
        rows=int(rows), clusters=int(clusters), imbalance=float(imbalance),
        spill=int(spill),
    ))


def physical_io_prediction(
    *, rows: int, substrate_passes: int, spill: int = SPILL,
    page_cache_budget_bytes: int = PAGE_CACHE_BUDGET_BYTES,
    dimension: int = DIMENSION,
) -> dict[str, Any]:
    """The registered two-regime prediction for what reaches the block layer."""
    architecture = architectural_io(
        rows=rows, substrate_passes=substrate_passes, spill=spill,
        dimension=dimension,
    )
    fits = architecture["substrate_bytes"] <= int(page_cache_budget_bytes)
    predicted_substrate = (
        architecture["substrate_bytes"] if fits
        else architecture["substrate_read_bytes"]
    )
    # A spill group is bounded by the scratch budget, which is smaller than the
    # page-cache budget at every rung, so the read-back of a group just written
    # is predicted to be served from cache in both regimes.
    return {
        **architecture,
        "page_cache_budget_bytes": int(page_cache_budget_bytes),
        "substrate_fits_page_cache": bool(fits),
        "predicted_physical_substrate_read_bytes": int(predicted_substrate),
        "predicted_physical_spill_read_back_bytes": 0,
        "predicted_physical_write_bytes": int(architecture["spill_write_bytes"]),
        "predicted_physical_read_bytes": int(predicted_substrate),
        "regime": "page-cache-resident" if fits else "page-cache-thrashing",
        "note": IO_REGIME_NOTE,
    }


def io_hours(
    *, read_bytes: float, write_bytes: float,
    read_bytes_per_s: float, write_bytes_per_s: float,
) -> dict[str, Any]:
    """Hours of pure I/O at stated rates. The rates are never inferred here."""
    if read_bytes_per_s <= 0 or write_bytes_per_s <= 0:
        raise Round0236Error("R0236 I/O rates must be positive")
    seconds = float(read_bytes) / float(read_bytes_per_s) + float(
        write_bytes
    ) / float(write_bytes_per_s)
    return {
        "read_bytes": float(read_bytes),
        "write_bytes": float(write_bytes),
        "read_bytes_per_s": float(read_bytes_per_s),
        "write_bytes_per_s": float(write_bytes_per_s),
        "io_seconds": float(seconds),
        "io_hours": float(seconds) / 3600.0,
    }


def io_scaling_fit(points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Log-log fit of architectural substrate reads against `N`.

    The structural prediction is an exponent of exactly `2`. Fitting it is a
    check on the structure, not a discovery: if the realised group packing
    tracked `N` linearly the exponent would come back near `2`, and if the
    builder ever stopped re-reading the substrate per group it would fall.
    """
    usable = [
        point for point in points
        if float(point.get("rows") or 0) > 0
        and float(point.get("substrate_read_bytes") or 0) > 0
    ]
    if len(usable) < 2:
        raise Round0236Error("R0236 I/O scaling fit needs at least two points")
    usable = sorted(usable, key=lambda point: float(point["rows"]))
    x = np.log(np.asarray([float(p["rows"]) for p in usable], dtype=np.float64))
    y = np.log(
        np.asarray([float(p["substrate_read_bytes"]) for p in usable],
                   dtype=np.float64)
    )
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    return {
        "points": [dict(point) for point in usable],
        "exponent": float(slope),
        "coefficient": float(np.exp(intercept)),
        "r_squared": float(1.0 - residual / total) if total > 0 else None,
        "n_points": len(usable),
        "structural_exponent": 2.0,
        "note": (
            "substrate_read_bytes = substrate_passes * N * 1536 with "
            "substrate_passes proportional to N, so the structural exponent is "
            "2. Integer group counts make the realised exponent land near it "
            "rather than on it."
        ),
    }


__all__ = [
    "BUILD_TIMEOUT_S",
    "CANDIDATE",
    "COMPOSITION",
    "CONTROL_CLUSTERS",
    "C_BUILD_MIN",
    "C_BUILD_MIN_NOTE",
    "C_MIN",
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
    "GPU_HOURS_CAP",
    "GRANDPARENT_ROUND_ID",
    "GRANDPARENT_ROWS",
    "GRAPH_CAPABILITY",
    "GRAPH_DEGREE",
    "GRAPH_K",
    "GRAPH_SCHEMA",
    "GUARD_DEVICE_BUDGET_BYTES",
    "GUARD_HOST_ANON_BUDGET_BYTES",
    "GUARD_IMBALANCE_MARGIN",
    "GUARD_NOTE",
    "GUARD_SAFETY_BYTES",
    "GUARD_SWAP_GROWTH_ABORT_BYTES",
    "HOST_ANON_AUTHORITATIVE_SAMPLER",
    "HOST_ANON_FIELD_NOTE",
    "IMBALANCE_CAPABILITY",
    "IMBALANCE_PROBE_CLUSTERS",
    "IMBALANCE_PROBE_ROWS",
    "IMBALANCE_REPLICATE_SEEDS",
    "INCREMENT_BY_CORPUS",
    "INTERMEDIATE_GRAPH_DEGREE",
    "IO_CAPABILITY",
    "IO_INSTRUMENT_NOTE",
    "IO_NOTE",
    "IO_REGIME_NOTE",
    "KNOWN_TRAILING_FRAGMENTS",
    "LADDER_CAPABILITY",
    "LADDER_RULE",
    "LADDER_SCHEMA",
    "LAW_GRAPH_DEGREE",
    "LAW_HOMOGENEITY_NOTE",
    "LAW_INTERMEDIATE_GRAPH_DEGREE",
    "LAW_MAX_ITERATIONS",
    "LAW_RESIDUAL_MARGIN",
    "LAW_SCHEMA",
    "MAX_ITERATIONS",
    "MAX_REPLACEMENT_ROUNDS",
    "MAX_ZERO_DEGREE_ROWS",
    "NESTING_NOTE",
    "NN_DESCENT_SETTING",
    "PAGE_CACHE_BUDGET_BYTES",
    "PARENT_COMPOSITION",
    "PARENT_ROUND_ID",
    "PARENT_ROWS",
    "PHASE2_RUNGS",
    "PRIMARY_IMBALANCE_SEED",
    "R0235_BUILD_TIMEOUT_S",
    "RAW_FORMAT",
    "RECALL_MEAN_FLOOR",
    "RECALL_P10_FLOOR",
    "RECALL_POPULATION",
    "REPLICATE_NOTE",
    "RESERVE_CORPUS_ROWS",
    "RESERVE_NOTE",
    "RESERVE_QUERY_ROWS",
    "RESERVE_ROWS",
    "RESERVE_ROWS_PER_CORPUS",
    "RESERVE_SEED",
    "ROUND_ID",
    "ROWS",
    "ROW_POLICY",
    "Round0236Error",
    "SAMPLE_INTERVAL_S",
    "SCRATCH_BUDGET_BYTES",
    "SELECTION_CANDIDATES",
    "SELECTION_LAW",
    "SELECTION_SEED",
    "SHARD_COVERAGE_FLOOR",
    "SPILL",
    "STRUCTURAL_POPULATION",
    "SUBSTRATE_CAPABILITY",
    "SUBSTRATE_SCHEMA",
    "TARGET_SHARES",
    "TIE_TOLERANCE",
    "TRAILING_FRAGMENT_POLICY",
    "TRUTH_AFFORDABILITY_NOTE",
    "TRUTH_CAPABILITY",
    "TRUTH_METHOD",
    "TRUTH_PROBE_ROWS",
    "TRUTH_PROBE_SEED",
    "TRUTH_SCHEMA",
    "WATCHDOG_POLL_S",
    "ZERO_ROW_POLICY",
    "admissible_max_cluster_rows",
    "admit_law_point",
    "architectural_io",
    "assert_memmap_for_cuvs",
    "assert_nesting",
    "assert_no_signal_policy",
    "assert_reserve_disjoint",
    "carry_distance",
    "fit_device_law",
    "guard_decision",
    "guard_device_bytes",
    "guarded_max_cluster_rows",
    "imbalance_tolerance",
    "io_hours",
    "io_projection",
    "io_scaling_fit",
    "law_device_bytes",
    "mean_cluster_rows",
    "pack_clusters_into_groups",
    "physical_io_prediction",
    "predicted_substrate_passes",
    "provenance_keys",
    "replicate_drift",
    "replicate_summary",
    "resolve_shard_rows",
    "select_clusters",
    "substrate_pass_count",
    "truth_probe_query_rows",
    "validate_composition",
    "validate_shard_span",
]
