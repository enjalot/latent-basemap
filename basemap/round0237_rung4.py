"""Frozen contract for R0237 — Phase 2 rung 4 (50M), nested on R0236's rung.

Three things happen and nothing else: assemble a 50,000,000-row mixed universe
that **contains** R0236's 25,000,000 training rows, build and qualify its k15
graph, and settle the last open question in the ladder — whether 100M is
affordable inside the card at any registered `c`. No map is trained.

## What R0236 left open, and what this round does about it

* **B1 — 100M is answered, and the answer is `c = 400`.** R0236 priced 100M at
  `c = 200` with `4.84%` tolerance and blocked it. Review-0236-01 F2 showed the
  round had listed the lever that already settles it third, as "the fallback":
  `c = 400` at 100M is `16.360 GiB` at `+96.58%` tolerance and, decisively, sits
  at `0.69x` of the fitted device law's range where `c = 200` demands a `1.30x`
  extrapolation. A higher `s` is **counterproductive** and is struck from
  consideration (`mean_cluster_rows = N*s/c` is linear in `s`; `s = 10` at
  `c = 200` gives `-16.13%`), and a larger device budget cannot reach 50M's
  safety level without consuming `31.10` of the card's `31.37 GiB`. So this
  round **confirms** rather than re-derives: it measures imbalance at
  50,000,000 rows across **five** k-means seeds, re-prices every rung from it,
  and registers 100M with the candidate set `(200, 400)` and a recommendation.
* **The one genuine gap: recall above `c = 64` has never been measured.** Every
  sealed ladder scores recall at `c = 16` or `c = 64` only. If the graph a
  `c = 400` partition can reach is poor, `c = 400`'s tolerance advantage is
  worthless — and that must be known before 100M is built, not after. R0229's
  reachability *ceiling* is the right instrument and it is nearly free: R0236's
  realised `c = 64` recall at 25M (`0.9996659`) equals the 2M `c = 64` ceiling
  (`0.9996687`) to `2.8e-6`, so the builder attains its ceiling and the ceiling
  predicts the build. This round scans it at `c = 64 / 128 / 200 / 400` on the
  **25,000,000-row substrate that already exists**, with no new assembly.
* **The margin question is CLOSED, and this round spends nothing on it.**
  R0236 flagged that `IMBALANCE_GUARD_MARGIN = 1.1648840` (`16.49%`) is smaller
  than the worst within-`N` spread it measured (`22.35%` at `c = 128`).
  Review-0236-01 F3 traced the path from spread to a device OOM and found there
  is none: the imbalance the guard consumes is *measured at the same seed on the
  same substrate the build then partitions* (realised `1.57375936` against the
  probe's `1.57359776`, `1.03e-4` relative), a hash-bound after-assignment
  capacity refusal fires `16.6%` below any plausible adverse value, and reaching
  the device budget would take an `11.6%` law under-prediction against a `2.73%`
  worst observed. **The hazard is a spurious refusal, not a wedge.** The margin
  therefore stays exactly as registered, no sensitivity table is published, and
  no cell is spent on it.
* **B4 — the page-cache flip point has never been measured, and this rung is
  where it gets tested.** R0236 measured substrate reads landing at `0.15%`
  (process) and `0.60%` (device) of their architectural volume at 25M. But
  review-0236-01 F5 showed the framing around that measurement does not hold up:
  `substrate_read_bytes` is *identically* `substrate_passes x N x 1536` in all
  seven fitted points and six of them were never measured physically, so
  `N^1.9675` is a regression on an algebraic identity describing a ceiling
  function, not a measured scaling — **this round does not inherit that
  exponent**. And the `~52M` flip point is exactly
  `page_cache_budget_bytes / 1536`, i.e. a restatement of the registered `80 GB`
  assumption, which 25M sat far from under any plausible budget. This rung's
  substrate is `76.8 GB` against that `80 GB`: a `4%` margin, the first rung with
  any power to locate the boundary. The finding reported here is the **measured**
  one — actual `read_bytes` against the architectural volume, plus a direct
  `mincore` reading of how much of the substrate was resident in page cache
  before and after the build — and the regime is stated from that, not from a fit.

## Nesting — the Phase 2 design constraint

    T4 = T3  U  uniform_without_replacement(P \\ T3 \\ R1, n4 - n3)

`T3` is R0236's sealed training selection and `R1` the reserve R0233 drew,
R0235 inherited and R0236 inherited again. Conditional on `R1`, `T3` is uniform
over size-`n3` subsets of `P \\ R1`, so adding a uniform draw from the
complement makes `T4` **exactly** a uniform size-50,000,000 subset of `P \\ R1`.
Nesting is positional: rung-3 row `i` is rung-4 row `i` for every
`i < 25,000,000`, verified by hashing the prefix against R0236's sealed
`ordered_substrate_sha256`.

Because the prefix is byte-identical all the way down, **one file now carries
the whole ladder**: `substrate[:6,250,000]` is R0233's rung,
`substrate[:12,500,000]` is R0235's and `substrate[:25,000,000]` is R0236's.
All four prefix hashes are published.

## Truth at this rung

Exact brute-force truth over all 50,000,000 rows costs `16 x` R0235's
`4,223.74 s` at 12.5M, i.e. about `18.8 GPU-h` against this round's `5.0` cap:
it is not affordable and the round says so rather than pretending. Qualification
therefore scores a **uniform probe of 1,000,000 query rows drawn without
replacement over all 50,000,000 row ids at a seed registered here, before the
substrate exists**, searched against **all** 50,000,000 rows by brute force. It
is not a seed set, not a neighbour union and not hub-biased. The R0215
degree-zero tripwire and the structural checks still run over **every** row, in
and out.

`min = 0.0` on a handful of probe rows is expected: review-0235-01 traced it at
12.5M to exact-duplicate families whose true 15th-best cosine is `~1.0` and
whose builder-returned neighbours are a *different* member of the same
near-duplicate block, short by `2.15e-06` to `4.05e-06` — above the `1e-6` tie
tolerance but at or below the fp32 noise floor at `cos ~ 1`. R0236 assumed that
explanation carried. This round **verifies it** instead, per row, with
`zero_recall_forensic`.

## Safety preconditions, carried unchanged

Both GPU wedges in this program were NVIDIA UVM page-fault deadlocks on driver
`570.211.01`. The build path is not copied: `basemap/round0237_build.py` calls
`basemap/round0236_build.py`, which calls `basemap/round0235_build.py`, which
calls `basemap/round0233_build.py` with one named constant rebound. Every
wrapper is fail-closed and none installs a signal handler.

1. Every buffer handed to cuVS is a read-only, C-contiguous `np.memmap`,
   including every intermediate spill file, asserted immediately before
   `nn_descent.build` receives it.
2. No signal is ever delivered to a build process. The abort path is a flag file
   the child polls; the parent never calls `terminate()`, `kill()` or `os.kill`.
3. A wedged GPU is never probed.
4. A predictive guard runs before every cell on device, host **anonymous** bytes
   and disk, with `refused_a_priori` recorded as data.
5. **New at this rung, because the card is shared:** free device memory is read
   immediately before the cell and the cell is refused unless the *predicted*
   device charge fits what is actually free, not what a 32 GiB card would have.
   An unrelated project intermittently holds `~6.3 GB` on this GPU.
6. **New at this rung, registered by R0236 in advance:** the swap-growth abort
   is *conjunctive* — swap growth alone no longer trips it. R0236 measured swap
   rising `2,111 -> 3,576 MB` under memmap I/O while `MemAvailable` stayed at
   `97.3%` of RAM and the largest anonymous consumer box-wide was `372 MB`; that
   is page cache filling, not memory pressure, and aborting on it would refuse a
   healthy cell. Growth must now coincide with a large anonymous footprint **or**
   a small `MemAvailable`. `MemAvailable` is sampled continuously rather than
   once. Whether the old swap-only rule *would* have fired is recorded either
   way, so the change is auditable rather than invisible.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from basemap.round0227_low_c_contract import SCRATCH_BUDGET_BYTES
from basemap.round0236_rung3 import (
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
    IO_INSTRUMENT_NOTE,
    IO_NOTE,
    IO_REGIME_NOTE,
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
    PAGE_CACHE_BUDGET_BYTES,
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
    architectural_io,
    assert_memmap_for_cuvs,
    assert_no_signal_policy,
    carry_distance,
    fit_device_law,
    guard_decision,
    guard_device_bytes,
    guarded_max_cluster_rows,
    imbalance_tolerance,
    io_hours,
    io_projection,
    io_scaling_fit,
    json_safe,
    law_device_bytes,
    mean_cluster_rows,
    pack_clusters_into_groups,
    physical_io_prediction,
    predicted_substrate_passes,
    resolve_shard_rows,
    select_clusters,
    substrate_pass_count,
)
from basemap.round0235_rung2 import rung_derivation


ROUND_ID = "0237"

SUBSTRATE_CAPABILITY = "minilm-mixed-50000k-nested-substrate-and-reserves-v1"
TRUTH_CAPABILITY = "minilm-mixed-50000k-uniform-probe-k15-truth-v1"
LADDER_CAPABILITY = "minilm-mixed-50000k-cluster-spill-build-ladder-v1"
GRAPH_CAPABILITY = "minilm-mixed-50000k-cluster-spill-k15-fuzzy-graph-v1"
IMBALANCE_CAPABILITY = "minilm-mixed-cluster-spill-s8-imbalance-50m-five-seed-v1"
IO_CAPABILITY = "cluster-spill-substrate-io-scaling-v2"
FEASIBILITY_CAPABILITY = "cluster-spill-100m-feasibility-verdict-v1"

SUBSTRATE_SCHEMA = "round0237-minilm-mixed-50000k-nested-substrate-v1"
TRUTH_SCHEMA = "round0237-minilm-mixed-50000k-uniform-probe-k15-truth-v1"
LADDER_SCHEMA = "round0237-minilm-mixed-50000k-build-ladder-v1"
GRAPH_SCHEMA = "round0237-minilm-mixed-50000k-k15-fuzzy-graph-v1"
LAW_SCHEMA = "round0237-guarded-device-law-io-scaling-and-100m-verdict-v1"

#: Rung 4 of `../guides/plan-minilm-100m-v2.md` Phase 2.
ROWS = 50_000_000
#: Rung 3. Every one of these rows is contained in `ROWS`, at the same position.
PARENT_ROWS = 25_000_000
PARENT_ROUND_ID = "0236"
#: Rung 2, which is R0236's own byte-identical prefix and therefore ours too.
GRANDPARENT_ROWS = 12_500_000
GRANDPARENT_ROUND_ID = "0235"
#: Rung 1, the same way, one level deeper.
GREAT_GRANDPARENT_ROWS = 6_250_000
GREAT_GRANDPARENT_ROUND_ID = "0233"

#: The four prefixes this round publishes. One file carries the whole ladder.
LADDER_PREFIX_ROWS: tuple[int, ...] = (
    GREAT_GRANDPARENT_ROWS, GRANDPARENT_ROWS, PARENT_ROWS, ROWS,
)

#: The owner-confirmed 40/25/25/10 shares at this rung's exact row counts.
COMPOSITION: tuple[tuple[str, int], ...] = (
    ("fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2", 20_000_000),
    ("RedPajama-Data-V2-sample-10B-chunked-120-all-MiniLM-L6-v2", 12_500_000),
    ("pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2", 12_500_000),
    ("starcoderdata-code-chunked-120-all-MiniLM-L6-v2", 5_000_000),
)
#: R0236's composition, which this round's prefix must reproduce exactly.
PARENT_COMPOSITION: tuple[tuple[str, int], ...] = (
    ("fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2", 10_000_000),
    ("RedPajama-Data-V2-sample-10B-chunked-120-all-MiniLM-L6-v2", 6_250_000),
    ("pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2", 6_250_000),
    ("starcoderdata-code-chunked-120-all-MiniLM-L6-v2", 2_500_000),
)
TARGET_SHARES = {name: n / ROWS for name, n in COMPOSITION}
INCREMENT_BY_CORPUS = {
    name: n - dict(PARENT_COMPOSITION)[name] for name, n in COMPOSITION
}

#: The increment's own seed. The prefix is not re-drawn: it is R0236's bytes.
SELECTION_SEED = 237
SELECTION_LAW = (
    "T4 = T3 U uniform_without_replacement(P \\ T3 \\ R1, n4 - n3), per corpus, "
    "at seed 237 + corpus index, with rejected rows replaced by fresh uniform "
    "draws from the unpicked complement until the increment quota is met. T3 is "
    "R0236's sealed training selection and R1 the reserve R0233 drew and R0235 "
    "and R0236 inherited. Because (T3, R1) is a uniform ordered pair of disjoint "
    "sets, T4 is EXACTLY a uniform size-50,000,000 subset of P \\ R1. Never a "
    "prefix of the corpus; always a positional prefix of the substrate."
)
NESTING_NOTE = (
    "rung-3 row i is rung-4 row i for every i < 25,000,000. Verified by hashing "
    "the substrate prefix with ordered_array_sha256 and comparing against "
    "R0236's sealed ordered_substrate_sha256, and independently by set "
    "containment on packed (corpus, shard, row) provenance keys. Because R0236's "
    "own prefix is byte-identical to R0235's and R0235's to R0233's, this one "
    "file carries rungs 1-4 and all four prefix hashes are published."
)
RESERVE_NOTE = (
    "R0233's reserve, inherited through R0235 and R0236 verbatim - the same "
    "200,000 rows, 50,000 per training corpus, R0108's 49,500 + 500 split - "
    "copied into this round's artifact tree and verified byte-identical by "
    "sha256. Those rows are EXCLUDED from rung 4's draw pool, exactly as they "
    "were from every earlier rung's, so one fixed eval set stays valid for the "
    "whole ladder and a rung-to-rung comparison is a comparison of N alone."
)

# --------------------------------------------------------------------------- #
# the builder — R0229's adopted arm, unchanged since R0233
# --------------------------------------------------------------------------- #
#: The `c` values whose imbalance is MEASURED at this N, unchanged from R0235 and
#: R0236 so the drift table stays like-for-like. `128` is the value this rung
#: selects; `200` and `400` are the two the 100M rung has to choose between and
#: are the reason review-0236 asked for this measurement.
IMBALANCE_PROBE_CLUSTERS: tuple[int, ...] = (16, 32, 64, 128, 200, 400)

#: **Five k-means seeds, up from R0236's three.** R0236's closing note asked for
#: exactly this: "run the replicate grid at c = 200 with more seeds at 25M to put
#: a confidence interval on the spread rather than a range of three". Five is
#: what fits the budget at this N (~8 s/cell at 50,000,000 rows). Seed 226 is
#: `A_SEED`, the seed every prior rung used, so the 226 realisation stays the
#: like-for-like point in the drift table; 236 and 1236 are R0236's other two, so
#: three of the five columns are directly comparable to the sealed 25M grid.
IMBALANCE_REPLICATE_SEEDS: tuple[int, ...] = (226, 236, 1236, 2236, 3236)
PRIMARY_IMBALANCE_SEED = 226
#: This round measures at its own N only. R0236 already measured the three
#: smaller nested prefixes with three seeds each and sealed them; those cells are
#: read from its hash-bound artifact and merged into the drift table rather than
#: recomputed, which is both cheaper and non-negotiable evidence.
IMBALANCE_PROBE_ROWS: tuple[int, ...] = (50_000_000,)
REPLICATE_NOTE = (
    "five k-means seeds at N = 50,000,000, by R0226's _kmeans/_assign imported "
    "unmodified, on this round's substrate. The 226 realisation is the "
    "like-for-like point every prior rung reported; the spread across seeds at "
    "fixed N is the draw channel. R0236 established that the movement across N "
    "is smaller than that spread at every c (drift_exceeds_spread false in all "
    "six rows), so this round does not re-litigate drift - it shrinks the "
    "uncertainty on the spread itself, which is the quantity that actually "
    "binds the 100M decision."
)

#: The `c` values this round's own graph may be built at. R0236's re-derivation
#: prices 50M at `c = 128` with `52.67%` of tolerance to adverse imbalance
#: against a worst measured within-N spread of `22.35%`, which is why this rung
#: is next; `c = 200` is the registered fallback if this round's own measured
#: imbalance refuses `c = 128`. `c = 16 / 32 / 64` are below the value the law
#: admits at this N and are PRICED from this round's own measurement and
#: published beside the selection; they are not built. `c = 400` is priced too -
#: it is the 100M fallback, not a 50M candidate.
SELECTION_CANDIDATES: tuple[int, ...] = (128, 200)
C_BUILD_MIN = 128
C_BUILD_MIN_NOTE = (
    "the c-selection law is unchanged - the smallest candidate whose GUARDED "
    "largest cluster fits the device budget under the binding law - but the "
    "candidate set is registered as (128, 200). Every other probed c is priced "
    "from this round's own measurement and published beside the selection "
    "without being built. R0236's derivation already shows c = 64 at 50M needs "
    "a guarded largest cluster near 11.4M rows against an admissible 9.94M."
)
#: No control cell. R0235's `c = 64` control settled the `N`-independence of the
#: device law at a matched cluster size, R0236's 9th point confirmed the fitted
#: law to `0.09%`, and this round's budget buys the five-seed grid instead.
CONTROL_CLUSTERS: tuple[int, ...] = ()
LADDER_RULE = (
    "one cell: the c the selection law picks from imbalance measured at THIS N, "
    "taken on the WORST of the five replicate realisations, with the round's own "
    "imbalance margin applied. A refusal, abort or failure stops the ladder and "
    "is recorded as a measurement, with its GPU time charged to the round."
)

# --------------------------------------------------------------------------- #
# truth — a registered uniform probe, because full truth is not affordable
# --------------------------------------------------------------------------- #
TRUTH_PROBE_ROWS = 1_000_000
TRUTH_PROBE_SEED = 237_000
TRUTH_METHOD = (
    "exact brute-force fp32 cosine top-k of a registered uniform probe of "
    "1,000,000 query rows against ALL 50,000,000 substrate rows"
)
RECALL_POPULATION = (
    "a uniform probe of 1,000,000 of the 50,000,000 substrate rows, drawn "
    "without replacement at seed 237000 registered before the substrate "
    "existed, searched against all 50,000,000 rows; no seed set, no neighbour "
    "union, no hub bias"
)
TRUTH_AFFORDABILITY_NOTE = (
    "full exact truth over all 50,000,000 rows scales as N^2: R0235 spent "
    "4,223.74 s at 12,500,000, so 50,000,000 costs about 67,580 s = 18.8 GPU-h "
    "against this round's 5.0 GPU-h cap for the whole queue. It is not "
    "affordable and this round does not pretend otherwise. The probe costs "
    "about 1,352 s of matmul at the same throughput - twice R0236's 676 s, "
    "because the database doubled while the probe did not."
)
STRUCTURAL_POPULATION = "all 50,000,000 rows, both in-degree and out-degree"

# --------------------------------------------------------------------------- #
# the min = 0.0 forensic — verified rather than assumed
# --------------------------------------------------------------------------- #
#: fp32 accumulation over 384 dimensions of unit vectors puts the noise floor of
#: a cosine near 1.0 at roughly `5e-6`; review-0235-01 measured shortfalls of
#: `2.15e-06` to `4.05e-06` on the rows that score tie-aware zero at 12.5M. A row
#: whose shortfall sits at or below this is a tie-tolerance artefact on a
#: duplicate family, not a retrieval failure. This constant is a REPORTING
#: threshold: it is published beside every zero-recall row and is never used to
#: adjust a recall number, relax a floor, or exclude a row from a mean.
FP32_TIE_NOISE_FLOOR = 5e-6
#: A row whose exact 15th-best cosine is at least this is inside a duplicate
#: family by review-0235-01's own criterion (it measured `0.99999964-1.00000072`).
DUPLICATE_FAMILY_KTH_COSINE = 0.99999
ZERO_RECALL_FORENSIC_NOTE = (
    "review-0235-01 explained tie-aware min = 0.0 at 12.5M as a tie-tolerance "
    "artefact on exact-duplicate families: the row's true 15th-best cosine is "
    "~1.0, the builder returned 15 members of a DIFFERENT near-duplicate block "
    "at cosine ~0.9999975, and the shortfall of 2.15e-06 to 4.05e-06 is above "
    "the 1e-6 tie tolerance but at or below the fp32 noise floor at cos ~ 1. "
    "R0236 saw the same figure at 25M and did not re-derive it. This round "
    "verifies it per row instead of assuming it: for every probe row scoring "
    "tie-aware 0.0 it publishes the exact kth cosine, the builder's own best "
    "and worst candidate cosine, the shortfall against truth, and whether the "
    "row meets BOTH criteria. A row that does not meet them is a different "
    "phenomenon and the result must say so."
)

# --------------------------------------------------------------------------- #
# the guard
# --------------------------------------------------------------------------- #
GPU_HOURS_CAP = 5.0
#: Per-cell deadline. R0233's cells ran ~640 s over 50,000,000 spilled rows,
#: R0235's ~1,300 s over 100,000,000 and R0236's `2,789 s` over 200,000,000; the
#: measured ratios across those doublings are `2.03x` and `2.15x`, so this rung's
#: 400,000,000 spilled rows are expected near `6,000 s` and this is a `2.0x`
#: margin. On expiry the parent sets the cooperative flag and waits - it never
#: signals.
BUILD_TIMEOUT_S = 12_000.0

#: **The card is shared.** An unrelated project intermittently runs a ~6.3 GB
#: embedding job on this GPU. A guard that sizes a cell against a nominal 31.37
#: GiB card would admit a cell that then cannot allocate. Free device memory is
#: therefore read immediately before the cell and the predicted charge must fit
#: it with this much to spare. This is an ADDITIONAL refusal condition; it never
#: admits a cell the existing budget refuses.
DEVICE_FREE_HEADROOM_BYTES = 1024 ** 3
DEVICE_FREE_NOTE = (
    "free device memory is read from nvidia-smi immediately before the cell "
    "launches and the cell is refused unless predicted_device_bytes + 1 GiB "
    "fits inside it. The 24 GiB budget is a ceiling on what this program will "
    "ask for; this is a check on what the machine can actually give, and the "
    "two are different numbers whenever another process holds the card. Any "
    "foreign compute process is recorded, never signalled and never killed."
)

#: **The conjunctive swap rule, registered by R0236 in advance for this rung.**
#: R0236 measured swap growing `2,111 -> 3,576 MB` during assembly while
#: `MemAvailable` held `125.9 GB` of `129.4 GB` and the largest anonymous
#: consumer box-wide was `372 MB`. That is page cache filling under memmap I/O
#: and evicting cold anonymous pages: healthy, and the exact behaviour a
#: memmap-everything build is supposed to produce. Aborting on it would refuse a
#: cell that is nowhere near memory pressure. Growth must now coincide with
#: EITHER a large anonymous footprint OR a small MemAvailable.
GUARD_SWAP_CONJUNCTION_ANON_BYTES = 40 * 1024 ** 3
GUARD_SWAP_CONJUNCTION_MEMAVAILABLE_BYTES = 16 * 1024 ** 3
GUARD_SWAP_CONJUNCTION_NOTE = (
    "abort on swap growth over the pre-launch baseline ONLY when it coincides "
    "with host anonymous bytes above 40 GiB or MemAvailable below 16 GiB. The "
    "standalone anonymous-budget abort at 60 GiB is unchanged and still fires "
    "on its own. Whether the old swap-only rule WOULD have fired is recorded "
    "per cell as `swap_only_rule_would_have_fired`, so relaxing it is auditable "
    "rather than invisible. MemAvailable is sampled continuously by the parent "
    "watchdog rather than once before launch, which is the other half of what "
    "R0236 registered."
)

# --------------------------------------------------------------------------- #
# the host-anonymous field, review-0235-01 F4 / review-0236-01 F6
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
    "the parent's so a consumer that reads the old name gets the safe value. "
    "review-0236-01 F6 notes the per-cell build-receipt.json, written by "
    "R0233's reviewed child, still carries the child's phase-sampled figure "
    "under the bare name - that file is not rewritten and this field says so."
)

# --------------------------------------------------------------------------- #
# the margin — closed by review-0236-01 F3, unchanged and unspent-on here
# --------------------------------------------------------------------------- #
MARGIN_NOTE = (
    "IMBALANCE_GUARD_MARGIN stays at 1.1648840, unchanged since R0233, and this "
    "round publishes no sensitivity table and spends no cell on it. R0236 "
    "flagged the margin (16.49%) as possibly undersized against the worst "
    "within-N seed spread it measured (22.35% at c = 128). Review-0236-01 F3 "
    "looked for a path from that spread to a device OOM and found none: the "
    "imbalance the guard consumes is measured at the same seed on the same "
    "substrate the build then partitions (realised 1.57375936 against the "
    "probe's 1.57359776, 1.03e-4 relative, so the seed spread never enters), a "
    "hash-bound after-assignment capacity refusal fires 16.6% below any "
    "plausible adverse value, and reaching the device budget would require an "
    "11.6% law UNDER-prediction against a 2.73% worst observed. The hazard a "
    "larger margin would address is a spurious refusal, not a wedge, so raising "
    "it would make the guard worse, not safer."
)

# --------------------------------------------------------------------------- #
# the reachability ceiling at high c — the one real gap in the 100M case
# --------------------------------------------------------------------------- #
#: Measured on R0236's SEALED 25,000,000-row substrate. No assembly, no build,
#: no new bytes: `_kmeans` and `_assign` from R0226 imported unmodified, then a
#: host-side scan of how many of each probe row's 15 exact-truth neighbours share
#: a cluster with it. That fraction is the CEILING on strict recall for the
#: partition - no builder can retrieve a neighbour it never compares against.
REACHABILITY_CAPABILITY = "minilm-mixed-25000k-cluster-spill-high-c-reachability-v1"
REACHABILITY_SCHEMA = "round0237-minilm-mixed-25000k-high-c-reachability-ceiling-v1"
REACHABILITY_ROWS = 25_000_000
#: `64` is the control: R0236 BUILT this partition at this `N` and scored
#: `0.9996659` tie-aware / `0.9981179` strict on the same probe, so the control
#: says whether the ceiling predicts the build at this rung the way it did at 2M.
#: `128` is the partition this round's own rung uses. `200` and `400` are the two
#: 100M candidates and are the reason the scan exists.
REACHABILITY_CLUSTERS: tuple[int, ...] = (64, 128, 200, 400)
#: R0226's `A_SEED`, the seed every build in this program has used.
REACHABILITY_SEED = 226
#: Below this the partition is discarding too much true structure for a graph
#: built inside it to clear the `0.90` recall floor with any margin. It is a
#: REPORTING threshold on a ceiling, not a queue-aborting floor: a low ceiling at
#: `c = 400` does not invalidate this round's 50M rung, it changes the 100M
#: recommendation, and the round says which.
REACHABILITY_CONCERN_FLOOR = 0.99
REACHABILITY_NOTE = (
    "R0229 measured this ceiling at 2,000,000 rows in ~16 s per cell and it has "
    "never been measured above that. Every sealed ladder scores realised recall "
    "at c = 16 or c = 64 only, so 'is a c = 400 graph any good' has been an open "
    "assumption behind every 100M projection. The instrument transfers because "
    "the builder ATTAINS its ceiling: R0236's realised c = 64 recall at 25M "
    "(0.9996659) equals R0229's 2M c = 64 tie-aware ceiling (0.9996687) to "
    "2.8e-6. This scan runs on R0236's sealed substrate against R0236's sealed "
    "exact truth for its registered 1,000,000-row uniform probe, so it inherits "
    "a hash-bound population rather than drawing a new one. Only the STRICT "
    "ceiling is scanned: the tie-aware ceiling needs a full N x N similarity "
    "pass, which at 25,000,000 rows costs a truth node rather than 16 s, and "
    "R0229's own cells put the two within 2e-5 of each other at every c."
)
#: R0229's sealed `s = 8` ceilings at 2,000,000 rows, for the trend. Read from
#: the hash-bound artifact at run time; the literals exist so a CPU test can
#: check the reader against a known answer.
R0229_2M_S8_CEILING_REFERENCE: dict[int, float] = {
    16: 0.9999999, 64: 0.9996886, 200: 0.9982593,
}

# --------------------------------------------------------------------------- #
# the 100M verdict — confirm, do not re-open
# --------------------------------------------------------------------------- #
#: Registered here, in advance, per review-0236-01 F2: the final rung's
#: candidate set is `(200, 400)`. `c = 16 / 32 / 64 / 128` are priced and
#: published but are not 100M candidates.
HUNDRED_M_ROWS = 100_000_000
HUNDRED_M_CANDIDATES: tuple[int, ...] = (200, 400)
#: A candidate whose guarded largest cluster exceeds this multiple of the fitted
#: law's largest observed point is an EXTRAPOLATION and is recommended against
#: while an interpolating candidate is admissible. `c = 200` at 100M sits at
#: `1.30x`; `c = 400` at `0.69x`. This is the rule that makes the recommendation
#: a derivation rather than a preference, and it is registered before the run.
LAW_RANGE_CEILING = 1.0
HUNDRED_M_RULE = (
    "among the registered 100M candidates (200, 400), recommend the one that "
    "is (a) admissible under the guard with the registered margin, (b) INSIDE "
    "the fitted device law's range (guarded max cluster <= the largest point the "
    "law was fitted on), and (c) at or above the reachability concern floor. If "
    "more than one qualifies, prefer the smaller c, because fewer larger "
    "clusters are strictly better for reachability. If none is inside the law's "
    "range, recommend the admissible candidate with the largest tolerance and "
    "say plainly that it extrapolates. The recommendation is a recommendation: "
    "this round registers, builds and qualifies nothing at 100,000,000 rows."
)


class Round0237Error(RuntimeError):
    """The registered R0237 contract changed."""


# --------------------------------------------------------------------------- #
# composition, span, nesting — same shapes as R0236, at this rung's counts
# --------------------------------------------------------------------------- #
def validate_composition(counts: Mapping[str, int]) -> dict[str, Any]:
    """Fail closed unless the assembled universe is exactly the registered mix."""
    total = sum(int(value) for value in counts.values())
    if total != ROWS:
        raise Round0237Error(f"substrate has {total} rows, registered {ROWS}")
    observed: dict[str, Any] = {}
    for name, want in COMPOSITION:
        got = int(counts.get(name, 0))
        if got != want:
            raise Round0237Error(f"{name}: assembled {got} rows, registered {want}")
        observed[name] = {
            "rows": got,
            "share": got / ROWS,
            "registered_share": TARGET_SHARES[name],
            "inherited_from_rung3": dict(PARENT_COMPOSITION)[name],
            "newly_drawn": INCREMENT_BY_CORPUS[name],
        }
    return observed


def validate_shard_span(
    *, corpus: str, shards_touched: int, shards_total: int, label: str
) -> dict[str, Any]:
    """R0216's span assertion. It RAISES; the defect is invisible otherwise."""
    if shards_total <= 0:
        raise Round0237Error(f"{corpus}: no shards")
    coverage = shards_touched / float(shards_total)
    if coverage < SHARD_COVERAGE_FLOOR:
        raise Round0237Error(
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
        raise Round0237Error("R0237 provenance does not fit the registered key")
    return (corpus << 56) | (shard << 40) | row


def assert_nesting(*, parent: np.ndarray, child: np.ndarray) -> dict[str, Any]:
    """Rung 4 must CONTAIN rung 3, on row ids, and every child row is distinct."""
    parent_keys = provenance_keys(parent)
    child_keys = provenance_keys(child)
    if int(np.unique(child_keys).size) != int(child_keys.size):
        raise Round0237Error("R0237 substrate holds a duplicated source row")
    missing = int(np.setdiff1d(parent_keys, child_keys, assume_unique=False).size)
    if missing != 0:
        raise Round0237Error(
            f"R0237 is not nested on R0236: {missing} rung-3 rows are absent. "
            "Phase 2's whole design is one variable per rung; a non-nested rung "
            "confounds N with the sample."
        )
    positional = bool(
        parent_keys.size <= child_keys.size
        and np.array_equal(parent_keys, child_keys[: parent_keys.size])
    )
    if not positional:
        raise Round0237Error(
            "R0237 prefix is not R0236's rows in R0236's order; the registered "
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
        raise Round0237Error(
            f"R0237 reserve overlaps the training selection on {overlap} rows"
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
            raise Round0237Error(f"R0237 reserve overlaps training in {corpus}")
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
        raise Round0237Error(f"R0237 probe of {size} rows is not drawable from {rows}")
    rng = np.random.RandomState(int(seed))
    return np.sort(rng.choice(rows, size=size, replace=False)).astype(np.int64)


# --------------------------------------------------------------------------- #
# replicates — five seeds at one N, merged with R0236's sealed three
# --------------------------------------------------------------------------- #
def replicate_summary(values: Mapping[int, float]) -> dict[str, Any]:
    """Spread of one `(N, c)` cell across k-means seeds.

    `spread_relative` is `(max - min) / mean`. At five seeds `sample_sd` is
    finally worth quoting: R0236's three-seed cells could report a range but not
    a scale, and the 100M decision turns on exactly that scale.
    """
    seeds = sorted(int(key) for key in values)
    if not seeds:
        raise Round0237Error("R0237 replicate cell has no realisations")
    array = np.asarray([float(values[seed]) for seed in seeds], dtype=np.float64)
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if array.size > 1 else None
    return {
        "seeds": seeds,
        "values": [float(value) for value in array],
        "n": int(array.size),
        "mean": mean,
        "min": float(array.min()),
        "max": float(array.max()),
        "spread_absolute": float(array.max() - array.min()),
        "spread_relative": float((array.max() - array.min()) / mean) if mean else None,
        "sample_sd": sd,
        "relative_sample_sd": (
            float(sd / mean) if sd is not None and mean else None
        ),
        "primary_seed": PRIMARY_IMBALANCE_SEED,
        "primary": (
            float(values[PRIMARY_IMBALANCE_SEED])
            if PRIMARY_IMBALANCE_SEED in values else None
        ),
    }


def replicate_grid_table(
    grid: Mapping[int, Mapping[int, Mapping[int, float]]],
    *,
    sources_by_rows: Mapping[int, str] | None = None,
    inherited: Mapping[int, Mapping[int, float]] | None = None,
) -> dict[str, Any]:
    """The imbalance table across every `N` and `c` that has a measurement.

    `grid` is `{rows: {clusters: {seed: imbalance}}}`; cells measured in this
    round and cells read from R0236's sealed artifact both live here and are
    distinguished by `sources_by_rows`. `inherited` carries single-realisation
    points from artifacts that have only one (R0229's 2M cells).

    R0236 settled the drift question - `drift_exceeds_spread` was false for all
    six `c` - so this round reports the movement across `N` descriptively and
    puts its weight on the spread, which is what the 100M guard is exposed to.
    """
    normalised: dict[int, dict[int, dict[int, float]]] = {
        int(rows): {
            int(clusters): {int(seed): float(value) for seed, value in seeds.items()}
            for clusters, seeds in cells.items()
        }
        for rows, cells in grid.items()
    }
    labels = {int(k): str(v) for k, v in (sources_by_rows or {}).items()}
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
        spreads: list[float] = []
        for size in sizes:
            cell = normalised.get(size, {}).get(clusters)
            if cell is not None:
                summary = replicate_summary(cell)
                row["by_rows"][str(size)] = {
                    "replicated": True,
                    "source": labels.get(size, "measured on a nested prefix"),
                    **summary,
                }
                if summary["primary"] is not None:
                    primary_series.append((size, float(summary["primary"])))
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

        movement: dict[str, Any]
        if len(primary_series) < 2:
            movement = {
                "measured_at_rows": [int(n) for n, _v in primary_series],
                "movement_relative": None,
                "insufficient_points": True,
            }
        else:
            base_n, base = primary_series[0]
            top_n, top = primary_series[-1]
            movement = {
                "measured_at_rows": [int(n) for n, _v in primary_series],
                "movement_relative": (top - base) / base,
                "movement_span_rows": [int(base_n), int(top_n)],
            }
        pooled = float(max(spreads)) if spreads else None
        row.update({
            "movement_primary_across_n": movement,
            "worst_within_n_spread_relative": pooled,
            "movement_exceeds_spread": (
                None if pooled is None or movement.get("movement_relative") is None
                else bool(abs(float(movement["movement_relative"])) > pooled)
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
            "movement_relative is (imbalance at the largest measured N) / "
            "(imbalance at the smallest measured N) - 1 on the primary-seed "
            "series; a positive value moves toward the device budget. "
            "worst_within_n_spread_relative is the largest (max - min) / mean "
            "any single N produces from the k-means draw alone, and it is the "
            "quantity the guard margin has to cover. R0236 established that the "
            "movement is inside the spread at every c."
        ),
    }


# --------------------------------------------------------------------------- #
# the 100M verdict — a derivation from a rule registered before the run
# --------------------------------------------------------------------------- #
def hundred_m_verdict(
    *,
    imbalance_by_c: Mapping[int, float],
    laws: Sequence[Mapping[str, Any]],
    reachability_by_c: Mapping[int, float] | None = None,
    candidates: Sequence[int] = HUNDRED_M_CANDIDATES,
    rung: int = HUNDRED_M_ROWS,
    spill: int = SPILL,
    margin: float = GUARD_IMBALANCE_MARGIN,
    device_budget_bytes: int = GUARD_DEVICE_BUDGET_BYTES,
    residual_margin: float = LAW_RESIDUAL_MARGIN,
    law_range_ceiling: float = LAW_RANGE_CEILING,
    reachability_floor: float = REACHABILITY_CONCERN_FLOOR,
) -> dict[str, Any]:
    """Which `c` the final rung should be registered at, and why.

    Three published quantities per candidate, none of them new: the guard's own
    admissibility, the ratio of the guarded largest cluster to the largest point
    the device law was fitted on, and the measured reachability ceiling. The
    rule combining them is `HUNDRED_M_RULE`, registered before the run.
    """
    admissible_rows = admissible_max_cluster_rows(
        laws, device_budget_bytes=device_budget_bytes,
        residual_margin=residual_margin,
    )
    fitted_max = max(
        float(point["max_cluster_rows"])
        for law in laws for point in law["points"]
    )
    reach = {
        int(key): float(value) for key, value in (reachability_by_c or {}).items()
    }
    considered: list[dict[str, Any]] = []
    for clusters in sorted(int(value) for value in candidates):
        if int(clusters) not in imbalance_by_c:
            considered.append({
                "clusters": int(clusters),
                "qualifies": False,
                "reason": "no imbalance measured for this c at the source rung",
            })
            continue
        imbalance = float(imbalance_by_c[int(clusters)])
        guarded = guarded_max_cluster_rows(
            rows=int(rung), clusters=int(clusters), imbalance=imbalance,
            spill=int(spill), margin=float(margin),
        )
        charge = guard_device_bytes(laws, guarded, residual_margin=residual_margin)
        ratio = float(guarded) / fitted_max if fitted_max > 0 else None
        ceiling = reach.get(int(clusters))
        entry = {
            "clusters": int(clusters),
            "imbalance": imbalance,
            "imbalance_margin_applied": float(margin),
            "guarded_max_cluster_rows": float(guarded),
            "predicted_device_bytes": float(charge["predicted_device_bytes"]),
            "predicted_device_gib": float(charge["predicted_device_gib"]),
            "binding_law": charge["binding_law"],
            "admissible": bool(guarded <= admissible_rows),
            "tolerance_to_adverse_imbalance": (
                float(admissible_rows / guarded - 1.0) if guarded > 0 else None
            ),
            "law_fitted_max_cluster_rows": float(fitted_max),
            "law_range_ratio": ratio,
            "inside_fitted_law_range": (
                None if ratio is None else bool(ratio <= float(law_range_ceiling))
            ),
            "reachability_strict_ceiling": ceiling,
            "reachability_measured": ceiling is not None,
            "clears_reachability_floor": (
                None if ceiling is None else bool(ceiling >= float(reachability_floor))
            ),
        }
        entry["qualifies"] = bool(
            entry["admissible"]
            and entry["inside_fitted_law_range"]
            and (entry["clears_reachability_floor"] is not False)
        )
        considered.append(entry)

    qualifying = [entry for entry in considered if entry.get("qualifies")]
    if qualifying:
        chosen = min(qualifying, key=lambda entry: int(entry["clusters"]))
        basis = (
            "admissible under the registered guard, inside the fitted device "
            "law's range, and at or above the reachability concern floor"
        )
    else:
        usable = [
            entry for entry in considered
            if entry.get("admissible") and entry.get("clears_reachability_floor")
            is not False
        ]
        chosen = (
            max(
                usable,
                key=lambda entry: float(
                    entry.get("tolerance_to_adverse_imbalance") or -1.0
                ),
            ) if usable else None
        )
        basis = (
            "no candidate is inside the fitted law's range; the admissible "
            "candidate with the largest tolerance is recommended and it "
            "EXTRAPOLATES the law"
        ) if chosen else "no candidate qualifies at all"
    return {
        "rung": int(rung),
        "candidates": [int(value) for value in candidates],
        "candidates_considered": considered,
        "recommended_clusters": None if chosen is None else int(chosen["clusters"]),
        "recommendation_basis": basis,
        "recommendation": chosen,
        "admissible_max_cluster_rows": float(admissible_rows),
        "law_range_ceiling": float(law_range_ceiling),
        "reachability_concern_floor": float(reachability_floor),
        "rule": HUNDRED_M_RULE,
        "higher_spill_note": (
            "raising s is struck from consideration: mean_cluster_rows = N*s/c "
            "is linear in s, so s = 10 at c = 200 costs -16.13% of tolerance and "
            "s = 12 costs -30.11% (review-0236-01 F2). A larger device budget is "
            "also weak: c = 200 needs 31.10 of the card's 31.37 GiB to reach the "
            "safety 50M has at c = 128."
        ),
        "scope": (
            "a recommendation for a FUTURE round. This round registers, builds "
            "and qualifies nothing at 100,000,000 rows."
        ),
    }


# --------------------------------------------------------------------------- #
# the min = 0.0 forensic — verify the rungs 2-3 explanation, do not assume it
# --------------------------------------------------------------------------- #
def zero_recall_forensic(
    *,
    zero_rows: np.ndarray,
    truth_kth_cosine: np.ndarray,
    truth_best_cosine: np.ndarray,
    candidate_best_cosine: np.ndarray,
    candidate_worst_cosine: np.ndarray,
    tie_tolerance: float = TIE_TOLERANCE,
    noise_floor: float = FP32_TIE_NOISE_FLOOR,
    duplicate_kth: float = DUPLICATE_FAMILY_KTH_COSINE,
    sample: int = 64,
) -> dict[str, Any]:
    """Does every tie-aware-zero row match review-0235-01's explanation?

    All five arrays are indexed by position within `zero_rows`. A row matches
    when (a) its exact 15th-best cosine is at or above `duplicate_kth`, i.e. it
    sits in an exact-duplicate family, and (b) the shortfall of the builder's
    BEST candidate against that 15th-best cosine is at or below the fp32 noise
    floor at `cos ~ 1`. The verdict is data: a row that fails either clause is
    reported individually and the result has to explain it.
    """
    zero_rows = np.asarray(zero_rows, dtype=np.int64)
    if zero_rows.size == 0:
        return {
            "zero_rows": 0,
            "explanation_verified": None,
            "note": (
                "no probe row scored tie-aware 0.0 at this rung, so there is "
                "nothing to explain"
            ),
            "criteria": {
                "duplicate_family_kth_cosine": float(duplicate_kth),
                "fp32_noise_floor": float(noise_floor),
                "tie_tolerance": float(tie_tolerance),
            },
            "forensic_note": ZERO_RECALL_FORENSIC_NOTE,
        }
    kth = np.asarray(truth_kth_cosine, dtype=np.float64)
    best_truth = np.asarray(truth_best_cosine, dtype=np.float64)
    best_cand = np.asarray(candidate_best_cosine, dtype=np.float64)
    worst_cand = np.asarray(candidate_worst_cosine, dtype=np.float64)
    for name, array in (
        ("truth_kth_cosine", kth), ("truth_best_cosine", best_truth),
        ("candidate_best_cosine", best_cand),
        ("candidate_worst_cosine", worst_cand),
    ):
        if array.shape != zero_rows.shape:
            raise Round0237Error(
                f"R0237 zero-recall forensic: {name} has shape {array.shape}, "
                f"expected {zero_rows.shape}"
            )
    shortfall = kth - best_cand
    in_family = kth >= float(duplicate_kth)
    within_noise = shortfall <= float(noise_floor)
    matches = in_family & within_noise
    order = np.argsort(-shortfall, kind="stable")[: int(sample)]
    return {
        "zero_rows": int(zero_rows.size),
        "rows_in_duplicate_family": int(in_family.sum()),
        "rows_within_fp32_noise_floor": int(within_noise.sum()),
        "rows_matching_both": int(matches.sum()),
        "rows_matching_neither": int((~in_family & ~within_noise).sum()),
        "explanation_verified": bool(matches.all()),
        "shortfall_against_truth": {
            "min": float(shortfall.min()),
            "median": float(np.median(shortfall)),
            "max": float(shortfall.max()),
        },
        "truth_kth_cosine": {
            "min": float(kth.min()), "max": float(kth.max()),
        },
        "truth_best_cosine": {
            "min": float(best_truth.min()), "max": float(best_truth.max()),
        },
        "candidate_best_cosine": {
            "min": float(best_cand.min()), "max": float(best_cand.max()),
        },
        "candidate_worst_cosine": {
            "min": float(worst_cand.min()), "max": float(worst_cand.max()),
        },
        "worst_rows": [
            {
                "row": int(zero_rows[index]),
                "truth_kth_cosine": float(kth[index]),
                "candidate_best_cosine": float(best_cand[index]),
                "shortfall": float(shortfall[index]),
                "in_duplicate_family": bool(in_family[index]),
                "within_fp32_noise_floor": bool(within_noise[index]),
            }
            for index in order
        ],
        "criteria": {
            "duplicate_family_kth_cosine": float(duplicate_kth),
            "fp32_noise_floor": float(noise_floor),
            "tie_tolerance": float(tie_tolerance),
        },
        "forensic_note": ZERO_RECALL_FORENSIC_NOTE,
    }


def reachability_cell_summary(
    strict: np.ndarray, *, clusters: int, spill: int = SPILL,
    floor: float = REACHABILITY_CONCERN_FLOOR,
) -> dict[str, Any]:
    """Summarise one partition's strict reachability ceiling over the probe.

    `strict[i]` is the fraction of probe row `i`'s 15 exact-truth neighbours that
    share at least one cluster with it. A builder cannot retrieve a neighbour it
    never compares against, so the mean of this is a hard ceiling on strict
    recall for the partition, and `rows_with_zero_reachable` is the ceiling
    analogue of the R0215 degree-zero tripwire.
    """
    strict = np.asarray(strict, dtype=np.float64)
    if strict.ndim != 1 or strict.size == 0:
        raise Round0237Error("R0237 reachability needs a non-empty per-row vector")
    if float(strict.min()) < 0.0 or float(strict.max()) > 1.0:
        raise Round0237Error("R0237 reachability fractions must lie in [0, 1]")
    mean = float(strict.mean())
    return {
        "clusters": int(clusters),
        "spill": int(spill),
        "rows_scored": int(strict.size),
        "strict_ceiling_mean": mean,
        "strict_ceiling_p10": float(np.percentile(strict, 10)),
        "strict_ceiling_min": float(strict.min()),
        "fraction_fully_reachable": float((strict >= 1.0).mean()),
        "rows_with_zero_reachable": int((strict <= 0.0).sum()),
        "rows_below_one": int((strict < 1.0).sum()),
        "concern_floor": float(floor),
        "clears_concern_floor": bool(mean >= float(floor)),
        "note": REACHABILITY_NOTE,
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
    "DEVICE_FREE_HEADROOM_BYTES",
    "DEVICE_FREE_NOTE",
    "DEVICE_TOTAL_BYTES",
    "DIMENSION",
    "DISK_FREE_FLOOR_BYTES",
    "DUPLICATE_FAMILY_KTH_COSINE",
    "EXCLUDED_SHARDS",
    "FEASIBILITY_CAPABILITY",
    "FP32_TIE_NOISE_FLOOR",
    "FUZZY_RANDOM_STATE_SEED",
    "GPU_HOURS_CAP",
    "GRANDPARENT_ROUND_ID",
    "GRANDPARENT_ROWS",
    "GRAPH_CAPABILITY",
    "GRAPH_DEGREE",
    "GRAPH_K",
    "GRAPH_SCHEMA",
    "GREAT_GRANDPARENT_ROUND_ID",
    "GREAT_GRANDPARENT_ROWS",
    "GUARD_DEVICE_BUDGET_BYTES",
    "GUARD_HOST_ANON_BUDGET_BYTES",
    "GUARD_IMBALANCE_MARGIN",
    "GUARD_NOTE",
    "GUARD_SAFETY_BYTES",
    "GUARD_SWAP_CONJUNCTION_ANON_BYTES",
    "GUARD_SWAP_CONJUNCTION_MEMAVAILABLE_BYTES",
    "GUARD_SWAP_CONJUNCTION_NOTE",
    "GUARD_SWAP_GROWTH_ABORT_BYTES",
    "HOST_ANON_AUTHORITATIVE_SAMPLER",
    "HOST_ANON_FIELD_NOTE",
    "HUNDRED_M_CANDIDATES",
    "HUNDRED_M_ROWS",
    "HUNDRED_M_RULE",
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
    "LADDER_PREFIX_ROWS",
    "LADDER_RULE",
    "LADDER_SCHEMA",
    "LAW_GRAPH_DEGREE",
    "LAW_HOMOGENEITY_NOTE",
    "LAW_INTERMEDIATE_GRAPH_DEGREE",
    "LAW_MAX_ITERATIONS",
    "LAW_RANGE_CEILING",
    "LAW_RESIDUAL_MARGIN",
    "LAW_SCHEMA",
    "MARGIN_NOTE",
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
    "R0229_2M_S8_CEILING_REFERENCE",
    "RAW_FORMAT",
    "REACHABILITY_CAPABILITY",
    "REACHABILITY_CLUSTERS",
    "REACHABILITY_CONCERN_FLOOR",
    "REACHABILITY_NOTE",
    "REACHABILITY_ROWS",
    "REACHABILITY_SCHEMA",
    "REACHABILITY_SEED",
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
    "Round0237Error",
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
    "ZERO_RECALL_FORENSIC_NOTE",
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
    "hundred_m_verdict",
    "imbalance_tolerance",
    "io_hours",
    "io_projection",
    "io_scaling_fit",
    "json_safe",
    "law_device_bytes",
    "mean_cluster_rows",
    "pack_clusters_into_groups",
    "physical_io_prediction",
    "predicted_substrate_passes",
    "provenance_keys",
    "replicate_grid_table",
    "replicate_summary",
    "resolve_shard_rows",
    "reachability_cell_summary",
    "rung_derivation",
    "select_clusters",
    "substrate_pass_count",
    "truth_probe_query_rows",
    "validate_composition",
    "validate_shard_span",
    "zero_recall_forensic",
]
