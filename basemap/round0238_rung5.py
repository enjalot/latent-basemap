"""Frozen contract for R0238 — Phase 2 rung 5, the **100,000,000-row rung**.

This is the top of the composition-controlled nested ladder in
`../guides/plan-minilm-100m-v2.md`. Three things happen and nothing else:
assemble a 100,000,000-row mixed universe that **contains** R0237's 50,000,000
training rows, build its k15 fuzzy graph with `cluster-spill-nnd` at `s = 8`,
`c = 400`, and qualify it. **No map is trained here.**

## What is settled before this round starts, and is therefore not re-derived

* **`c = 400`, confirmed twice.** Review-0236-01 F2 and review-0237-01 both
  reach it independently under R0237's registered rule: at 100M `c = 200` is
  admissible but sits at `1.320x` of the fitted device law's range with `3.59%`
  of tolerance, while `c = 400` sits **inside** the range at `0.787x` with
  `73.83%` and a device charge of `17.396 GiB`. The registered 100M candidate
  set was `(200, 400)`; this round registers `(400,)` and publishes `c = 200`'s
  own re-derived tolerance beside it, refused in advance under clause (b).
* **The reachability cost of `c = 400` is measured, not assumed.** R0237 scanned
  the strict partition ceiling at 25M: `0.9998664` at `c = 64`, `0.9994947` at
  `c = 128`, `0.9990732` at `c = 200`, `0.9977319` at `c = 400`. Review-0237-01
  established that the ceiling is an **upper bound**, not a predictor, above 2M
  (builder loss is `~0.002` and roughly `c`-independent, `13x` the partition's
  loss at `c = 64`), so the projected realised recall at this rung is
  `~0.9957` strict / `~0.9975` tie-aware. R0228 saw displacement only at
  `0.9708` and `0.9512` and found `0.9889` clean, so the projection sits an
  order of magnitude clear of the regime where displacement was ever observed.
  This round **measures** the ceiling at its own `N` at `c = 400` as well, which
  closes review-0237-01 F9's objection that the ceiling trend rested on two
  cross-substrate points, and it registers `rows_with_zero_reachable` as a
  reported instrument exactly as review-0237-01 asked.
* **The margin does not move.** Review-0236-01 F3 found no path from seed spread
  to a device OOM and identified the hazard as a *spurious refusal*.
  `IMBALANCE_GUARD_MARGIN` stays at `1.1648840`. No sensitivity table is
  published and no cell is spent on it.
* **The conjunctive swap rule is kept unchanged.** Review-0237-01 released it as
  the program's first hard evidence that a conjunctive memory watchdog is the
  correct formulation: R0237's swap grew `4.09 GiB` over baseline while
  `MemAvailable` never fell below `100.4 GB`, and a swap-only rule would have
  cooperatively aborted a healthy `107.6`-minute build cell.

## The adverse drift this round carries explicitly

Review-0237-01 F5: on the three seeds common to the 25M and 50M grids, `c = 400`
— the one `c` this rung uses — moved **`+3.15%`** from 25M to 50M, primary seed
**`+10.69%`**. That is the **opposite** of the `-1.749%`-per-doubling "safe
direction" review-0236-01 F1 established and the program has been quoting.
R0237 held both grids and did not publish it. This round registers it as an
adverse prior in advance: the `73.83%` tolerance is being asked to absorb a
carry of one doubling whose observed movement at this `c` is `+3.15%` mean /
`+10.69%` primary, plus a within-`N` spread of `19.01%`. That is a `3.9x`
margin on the mean movement and `2.3x` on spread-plus-primary-movement — real
headroom, but it is headroom against a hazard moving the wrong way, and this
round says so before it measures rather than after. **If this round's own
measured worst-of-five imbalance at 100M exceeds the 50M value carried into the
prediction, the result states the excess plainly and does not absorb it.**

## Nesting — the Phase 2 design constraint

    T5 = T4  U  uniform_without_replacement(P \\ T4 \\ R1, n5 - n4)

`T4` is R0237's sealed training selection and `R1` the reserve R0233 drew and
R0235, R0236 and R0237 each inherited verbatim. Nesting is positional: rung-4
row `i` is rung-5 row `i` for every `i < 50,000,000`, verified by hashing the
prefix against R0237's sealed `ordered_substrate_sha256`.

Because the prefix is byte-identical all the way down, **one file carries the
whole five-rung ladder**: `substrate[:6,250,000]` is R0233's rung,
`[:12,500,000]` R0235's, `[:25,000,000]` R0236's and `[:50,000,000]` R0237's.
All **five** prefix hashes are published, every one computed from this round's
own bytes.

## The registered selection-law change: the code corpus pool was extended

Review-0237-01 F8 established, and this round re-verified from the shard files,
that `starcoderdata` held exactly `10,000,000` rows with `50,000` permanently
withheld as the fixed eval reserve, leaving `9,950,000` drawable against the
`10,000,000` code TRAINING rows this rung needs — short by exactly `50,000`.
Of the four options the review assessed, the owner approved **extending the
corpus**, which the review named the only one preserving both `N` and the
confirmed 40/25/25/10 shares, and which it required be registered as a
**selection-law change** rather than treated as a data top-up. It is registered
here, with its consequence stated rather than glossed:

* The 20 original shards are **byte-identical** and keep their `-of-00020`
  names; one shard `data-00020-of-00021.npy` (`100,000` rows) is appended and
  **sorts last** under the assembler's own `sorted(glob(...))` enumeration, so
  every existing global row id still denotes the same row and the nesting chain
  stays valid. `CODE_POOL_EXTENSION` records the verification this round
  performs before it draws anything.
* **The consequence, stated:** `T5` is no longer *exactly* uniform over
  `P_new \\ R1`. `T4` was drawn uniformly from `P_old \\ R1` before `P` grew, and
  the increment is uniform over `P_new \\ T4 \\ R1`, so the marginal inclusion
  probability is `0.99507438...` for a pre-extension row and `0.99009901...`
  for an appended one, against `0.99502488...` under exact uniformity — the
  appended `100,000` rows are under-represented by exactly `0.5000%` relative,
  an expected shortfall of `492.59` rows in `100,000,000` (`4.9259e-06` of the
  rung). Exact uniformity over `P_new` and byte-identical
  nesting on `T4` are mutually exclusive once `P` changes after `T4` is drawn;
  the ladder is composition-FIXED and nested, so nesting wins and the deviation
  is published with its arithmetic. `POOL_EXTENSION_UNIFORMITY` carries it.

## Truth at this rung

Exact truth over all 100,000,000 rows scales as `N^2`: R0235 spent `4,223.74 s`
at 12.5M, so this rung would cost about `75 GPU-h` against a `6.0` cap.
Review-0237-01 named that explicitly ("do not spend `~75 GPU-h` on full truth at
100M"). Qualification therefore scores a **uniform probe of 500,000 query rows
drawn without replacement over all 100,000,000 row ids at a seed registered
here, in the release commit, before the substrate exists**, searched against
**all** 100,000,000 rows by brute force. It is not a seed set, not a neighbour
union and not hub-biased (review-0227-01).

`500,000` rather than R0237's `1,000,000` is a budget decision and is registered
as one: at 100M a 1,000,000-row probe costs about `2,790 s` against `1,395 s`
for 500,000, and the queue's remaining margin under a `6.0` GPU-h cap does not
support the difference. The instrument is not weakened in any way that matters:
the standard error of the mean on a recall near `0.9957` is about `4.3e-6` at
`n = 500,000`, four orders of magnitude below the `0.0957` of slack between the
projected recall and the registered `0.90` floor, and the expected number of
duplicate-family `min = 0.0` rows (`~15`) is still large enough to adjudicate
individually as R0237 did.

The R0215 degree-zero tripwire and every structural check still run over
**every one of the 100,000,000 rows, in-degree AND out-degree**.

## Host memory — the reason this round has new code at all

Review-0237-01's closing note: "at 100M this stage roughly doubles and is the
host-memory constraint to watch, not the build". R0237's qualification node
peaked at `52.6 GB` anonymous. Doubling that is `105 GB` against `123 GB` of
RAM, and the naive path is worse than double, because the symmetrised graph
carries `~2.5e9` directed edges and `scipy.sparse` must widen its index dtype to
`int64` past `2^31` nonzeros. Two stages are therefore re-implemented here with
bounded host memory and are verified against the reviewed originals by test
rather than by argument:

* `_draw_streaming` reproduces R0233's `_draw` — the same RNG calls, the same
  rejection-and-replacement accounting, the same result — while writing accepted
  vectors to a staging memmap instead of accumulating them in a Python list. The
  naive path would have held `~92 GB` of anonymous memory for the FineWeb
  increment alone at this rung.
* `_fuzzy_symmetrise_blocked` computes exactly UMAP's
  `A + A^T - A o A^T` (`set_op_mix_ratio = 1.0`) in row stripes, using
  `scipy.sparse` for the arithmetic inside each stripe so the law itself is
  unchanged, and never materialises a sparse matrix with more than `~2.5e8`
  nonzeros. `smooth_knn_dist` and `compute_membership_strengths` are UMAP's own,
  called unmodified.

Both are covered by CPU tests that assert **exact** agreement with the reviewed
original on small inputs.

## Safety preconditions, carried unchanged

Both GPU wedges in this program were NVIDIA UVM page-fault deadlocks on driver
`570.211.01`. The build path is not copied: `basemap/round0238_build.py` calls
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
5. The card is shared: free device memory is read immediately before the cell
   and the cell is refused unless the *predicted* device charge fits what is
   actually free. Any foreign compute process is recorded, never signalled.
6. The swap-growth abort is **conjunctive**, unchanged from R0237, which is the
   formulation review-0237-01 released as load-bearing.
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


ROUND_ID = "0238"

SUBSTRATE_CAPABILITY = "minilm-mixed-100000k-nested-substrate-and-reserves-v1"
TRUTH_CAPABILITY = "minilm-mixed-100000k-uniform-probe-k15-truth-v1"
LADDER_CAPABILITY = "minilm-mixed-100000k-cluster-spill-build-ladder-v1"
GRAPH_CAPABILITY = "minilm-mixed-100000k-cluster-spill-k15-fuzzy-graph-v1"
IMBALANCE_CAPABILITY = "minilm-mixed-cluster-spill-s8-imbalance-50m-five-seed-v1"
IO_CAPABILITY = "cluster-spill-substrate-io-scaling-v2"
FEASIBILITY_CAPABILITY = "cluster-spill-100m-feasibility-verdict-v1"

SUBSTRATE_SCHEMA = "round0238-minilm-mixed-100000k-nested-substrate-v1"
TRUTH_SCHEMA = "round0238-minilm-mixed-100000k-uniform-probe-k15-truth-v1"
LADDER_SCHEMA = "round0238-minilm-mixed-100000k-build-ladder-v1"
GRAPH_SCHEMA = "round0238-minilm-mixed-100000k-k15-fuzzy-graph-v1"
LAW_SCHEMA = "round0238-guarded-device-law-io-scaling-and-100m-verdict-v1"

#: Rung 5 — the top of `../guides/plan-minilm-100m-v2.md` Phase 2.
ROWS = 100_000_000
#: Rung 4. Every one of these rows is contained in `ROWS`, at the same position.
PARENT_ROWS = 50_000_000
PARENT_ROUND_ID = "0237"
#: Rung 3, which is R0237's own byte-identical prefix and therefore ours too.
GRANDPARENT_ROWS = 25_000_000
GRANDPARENT_ROUND_ID = "0236"
#: Rung 2, the same way, one level deeper.
GREAT_GRANDPARENT_ROWS = 12_500_000
GREAT_GRANDPARENT_ROUND_ID = "0235"
#: Rung 1, the bottom of the ladder.
GREAT2_GRANDPARENT_ROWS = 6_250_000
GREAT2_GRANDPARENT_ROUND_ID = "0233"
#: The round that actually DREW the 200,000-row reserve. R0235 wrote its own
#: grandparent id into `originally_drawn_by_round` and R0236 and R0237 inherited
#: the expression, so their sealed manifests name the wrong round for a set of
#: rows every rung has copied byte-identically since R0233. Fixed here as a
#: literal; the earlier manifests are not rewritten.
RESERVE_DRAWN_BY_ROUND_ID = "0233"

#: The FIVE prefixes this round publishes. One file carries the whole ladder.
LADDER_PREFIX_ROWS: tuple[int, ...] = (
    GREAT2_GRANDPARENT_ROWS, GREAT_GRANDPARENT_ROWS, GRANDPARENT_ROWS,
    PARENT_ROWS, ROWS,
)
#: The three inherited prefix hashes this round must reproduce from its own
#: bytes, sealed by the rounds that drew them. `25,000,000` and `50,000,000`
#: come from the parent manifest chain at run time and are not literals here.
INHERITED_PREFIX_SHA256: dict[int, str] = {
    6_250_000: (
        "5d976ab6d895db45095967afd5ce7dd078a6242bc62edfff941c120fec473e36"
    ),
    12_500_000: (
        "bd004db8511c9e3ea44bbc1471f739cdcc5d78adb35f7cb53c96919638ec7ad5"
    ),
    25_000_000: (
        "466ef03904f86fdd7bb5b3b491a028be7a06dcf1c0856effdf0c7757adc7bc03"
    ),
}

#: The owner-confirmed 40/25/25/10 shares at the final rung's exact row counts.
COMPOSITION: tuple[tuple[str, int], ...] = (
    ("fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2", 40_000_000),
    ("RedPajama-Data-V2-sample-10B-chunked-120-all-MiniLM-L6-v2", 25_000_000),
    ("pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2", 25_000_000),
    ("starcoderdata-code-chunked-120-all-MiniLM-L6-v2", 10_000_000),
)
#: R0237's composition, which this round's prefix must reproduce exactly.
PARENT_COMPOSITION: tuple[tuple[str, int], ...] = (
    ("fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2", 20_000_000),
    ("RedPajama-Data-V2-sample-10B-chunked-120-all-MiniLM-L6-v2", 12_500_000),
    ("pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2", 12_500_000),
    ("starcoderdata-code-chunked-120-all-MiniLM-L6-v2", 5_000_000),
)
TARGET_SHARES = {name: n / ROWS for name, n in COMPOSITION}
INCREMENT_BY_CORPUS = {
    name: n - dict(PARENT_COMPOSITION)[name] for name, n in COMPOSITION
}

# --------------------------------------------------------------------------- #
# the registered selection-law change — the code corpus pool was extended
# --------------------------------------------------------------------------- #
#: The corpus whose pool changed between rung 4 and rung 5.
CODE_CORPUS = "starcoderdata-code-chunked-120-all-MiniLM-L6-v2"
#: The pool the parent rung drew from, and the pool this rung draws from. Both
#: are asserted against the live shard enumeration before anything is drawn.
CODE_POOL_PARENT_ROWS = 10_000_000
CODE_POOL_PARENT_SHARDS = 20
CODE_POOL_ROWS = 10_100_000
CODE_POOL_SHARDS = 21
#: The shard the extension appended. It must sort LAST under the assembler's own
#: `sorted(glob(...))` enumeration, or an existing global row id would move and
#: the byte-identical nesting this rung asserts would be a lie.
CODE_POOL_APPENDED_SHARDS: tuple[tuple[str, int, int], ...] = (
    ("data-00020-of-00021.npy", 100_000, 153_600_128),
)
CODE_POOL_EXTENSION = (
    "REGISTERED SELECTION-LAW CHANGE (review-0237-01 F8, owner-approved "
    "2026-08-09). The 100M rung needs 10,000,000 code TRAINING rows; the "
    "corpus held exactly 10,000,000 with 50,000 permanently withheld as the "
    "fixed eval reserve, leaving 9,950,000 drawable - short by exactly 50,000. "
    "Of the four options review-0237-01 assessed, shrinking the reserve was "
    "ruled out outright (it is the held-out eval set and changing it "
    "invalidates every rung-to-rung comparison the ladder exists to make), a "
    "silent backfill from another corpus was rejected as the same breach with "
    "worse disclosure, and accepting 99,950,000 rows at a 9.9950% code share "
    "was kept as the zero-GPU fallback. The owner approved EXTENDING the "
    "corpus, which is the only option preserving both N and the confirmed "
    "40/25/25/10 shares. It redefines the code corpus's pool P between rungs "
    "and is therefore a change to the selection law, not a data top-up, and is "
    "registered as one here. This round VERIFIES rather than assumes every "
    "claim about it before it draws a single row: the 20 original shards are "
    "byte-identical (size against the parent's sealed size manifest, plus the "
    "parent's sealed first-shard sha256), the appended shard sorts last, and "
    "the corpus-offset prefix is therefore unchanged so every existing global "
    "row id still denotes the same row. If any of those checks fails the node "
    "raises and the round stops."
)
POOL_EXTENSION_UNIFORMITY = (
    "T5 is NOT exactly uniform over P_new \\ R1, and the deviation is published "
    "rather than glossed. T4 was drawn uniformly from P_old \\ R1 (9,950,000 "
    "rows) BEFORE the pool grew, and this rung's increment is uniform over "
    "P_new \\ T4 \\ R1 (5,050,000 rows, of which 5,000,000 are drawn). So the "
    "marginal inclusion probability in T5 is 5,000,000/9,950,000 + "
    "(4,950,000/9,950,000) x (5,000,000/5,050,000) = 0.99507438181004... for a "
    "pre-extension row and 5,000,000/5,050,000 = 0.99009900990099... for one of "
    "the 100,000 appended rows, against 0.99502487562189... under exact "
    "uniformity: the appended rows are under-represented by exactly 0.5000% "
    "relative to a pre-extension row, an expected shortfall of 492.5866 rows "
    "against exact uniformity, i.e. 4.9259e-06 of the rung. Every one of those "
    "figures is recomputed in-node and sealed rather than transcribed. "
    "Exact uniformity over P_new and "
    "byte-identical nesting on T4 are mutually exclusive once P changes after "
    "T4 is drawn; the Phase-2 ladder is composition-FIXED and nested, so "
    "nesting wins and the arithmetic above is registered in advance. The "
    "increment ITSELF is exactly uniform over its own residual pool, which is "
    "what the shard-span and per-shard-count checks test."
)

#: The increment's own seed. The prefix is not re-drawn: it is R0237's bytes.
SELECTION_SEED = 238
SELECTION_LAW = (
    "T5 = T4 U uniform_without_replacement(P \\ T4 \\ R1, n5 - n4), per corpus, "
    "at seed 238 + corpus index, with rejected rows replaced by fresh uniform "
    "draws from the unpicked complement until the increment quota is met. T4 is "
    "R0237's sealed training selection and R1 the reserve R0233 drew and R0235, "
    "R0236 and R0237 each inherited verbatim. For the three base corpora P is "
    "unchanged, so (T4, R1) is a uniform ordered pair of disjoint sets and T5 is "
    "EXACTLY a uniform size-n5 subset of P \\ R1. For the code corpus P was "
    "EXTENDED between rungs under a registered selection-law change, and T5's "
    "marginal uniformity over the extended pool is off by the amount "
    "POOL_EXTENSION_UNIFORMITY states. Never a prefix of the corpus; always a "
    "positional prefix of the substrate."
)
NESTING_NOTE = (
    "rung-4 row i is rung-5 row i for every i < 50,000,000. Verified by hashing "
    "the substrate prefix with ordered_array_sha256 and comparing against "
    "R0237's sealed ordered_substrate_sha256, and independently by set "
    "containment on packed (corpus, shard, row) provenance keys. Because "
    "R0237's own prefix is byte-identical to R0236's, R0236's to R0235's and "
    "R0235's to R0233's, this one file carries rungs 1-5 and ALL FIVE prefix "
    "hashes are published, every one computed from this round's own bytes."
)
RESERVE_NOTE = (
    "R0233's reserve, inherited through R0235, R0236 and R0237 verbatim - the "
    "same 200,000 rows, 50,000 per training corpus, R0108's 49,500 + 500 split "
    "- copied into this round's artifact tree and verified byte-identical by "
    "sha256. Those rows are EXCLUDED from rung 5's draw pool, exactly as they "
    "were from every earlier rung's, so one fixed eval set stays valid for the "
    "whole ladder and a rung-to-rung comparison is a comparison of N alone. "
    "The code corpus's 50,000 reserve rows all lie inside the 20 pre-extension "
    "shards and are untouched by the pool extension; shrinking the reserve to "
    "close the 50,000-row code shortfall was ruled out outright."
)

# --------------------------------------------------------------------------- #
# the builder — R0229's adopted arm, unchanged since R0233
# --------------------------------------------------------------------------- #
#: The `c` values whose imbalance is MEASURED at this N, unchanged from R0235,
#: R0236 and R0237 so the four-rung drift table stays like-for-like. `400` is
#: the value this rung builds at; the rest are priced and published beside it.
IMBALANCE_PROBE_CLUSTERS: tuple[int, ...] = (16, 32, 64, 128, 200, 400)

#: **The same five k-means seeds R0237 used**, so all five columns compare
#: directly against the sealed 50M grid and the 50M -> 100M doubling can be read
#: per seed rather than only in aggregate. Seed 226 is `A_SEED`, the seed every
#: prior rung used, so the 226 realisation stays the like-for-like point.
IMBALANCE_REPLICATE_SEEDS: tuple[int, ...] = (226, 236, 1236, 2236, 3236)
PRIMARY_IMBALANCE_SEED = 226
#: This round measures at its own N only. R0236 sealed three seeds at the three
#: smallest nested prefixes and R0237 five at 50,000,000; both are byte-identical
#: prefixes of this substrate, so those cells are merged from their hash-bound
#: artifacts rather than recomputed.
IMBALANCE_PROBE_ROWS: tuple[int, ...] = (100_000_000,)
REPLICATE_NOTE = (
    "five k-means seeds at N = 100,000,000, by R0226's _kmeans/_assign imported "
    "unmodified, on this round's substrate. The 226 realisation is the "
    "like-for-like point every prior rung reported; the spread across seeds at "
    "fixed N is the draw channel. This round does NOT inherit the claim that "
    "movement across N is safe: review-0237-01 F5 showed that on shared seeds "
    "c = 400 moved +3.15% (primary +10.69%) from 25M to 50M, the OPPOSITE of "
    "review-0236-01 F1's -1.749% per doubling, and c = 400 is this rung's own "
    "partition. The 50M -> 100M movement measured here is published per c and "
    "per shared seed, and if this rung's worst-of-five at c = 400 exceeds the "
    "50M value the prediction carried, the result says so plainly."
)
#: The imbalance the 100M prediction was priced from — R0237's sealed worst of
#: five at 50,000,000 — and the tolerance it implied. Held here as literals so
#: the round can state its own measured excess against the prediction instead of
#: absorbing it, and so a CPU test can check the comparison arithmetic.
PREDICTION_IMBALANCE_AT_C400 = 2.456543
PREDICTION_TOLERANCE_AT_C400 = 0.738255
ADVERSE_DRIFT_NOTE = (
    "the carry is one doubling, 50,000,000 -> 100,000,000, and it is carried "
    "against an ADVERSE prior, not a safe one. review-0237-01 F5, computed from "
    "grids R0237 held but did not publish: on the three seeds common to the 25M "
    "and 50M rungs, c = 400 moved +3.15% mean and +10.69% on the primary seed, "
    "while review-0236-01 F1's -1.749%-per-doubling result does not reproduce "
    "there. Two thirds of c = 400's tolerance fall from 96.58% to 73.83% was a "
    "real rise in worst-of-the-same-three-seeds (2.170136 -> 2.359035, +8.70%) "
    "and one third the arithmetic of taking a max over five draws rather than "
    "three (2.359035 -> 2.456543, +4.13%); only the first part carries here. "
    "73.83% of tolerance against a +3.15% observed doubling movement is a 3.9x "
    "margin, and against +10.69% primary movement plus the 19.01% within-N "
    "spread it is still about 2.3x. The round registers that arithmetic in "
    "advance and reports its own measured value against it."
)

#: The `c` this round's graph is built at, registered as a set of ONE. Both
#: review-0236-01 F2 and review-0237-01 select `c = 400` independently under
#: R0237's registered rule, from R0237's own measured 50M imbalance: `c = 200`
#: is admissible but sits at `1.320x` of the fitted device law's range with
#: `3.59%` of tolerance, `c = 400` sits INSIDE the range at `0.787x` with
#: `73.83%` at `17.396 GiB`, and `c = 400`'s partition reachability ceiling is
#: MEASURED at `0.9977319`, above the `0.99` concern floor. Every other probed
#: `c` - including `200` - is re-priced from THIS round's own measurement and
#: published beside the selection without being built.
SELECTION_CANDIDATES: tuple[int, ...] = (400,)
C_BUILD_MIN = 400
C_BUILD_MIN_NOTE = (
    "the c-selection law is unchanged - the smallest candidate whose GUARDED "
    "largest cluster fits the device budget under the binding law - but the "
    "candidate set at this rung is registered as (400,), because R0237's own "
    "registered 100M rule already selected it on measured evidence and two "
    "independent reviews concurred. c = 200 is refused IN ADVANCE under that "
    "rule's clause (b), not on this round's discretion: it demands a 1.320x "
    "extrapolation of the device law where c = 400 interpolates it at 0.787x, "
    "and its 3.59% tolerance is 20.6x smaller. Its re-derived tolerance under "
    "THIS round's measured imbalance is published beside the selection anyway, "
    "so a reader can see what the refused candidate would have cost. No "
    "unregistered fallback c exists: if c = 400 is refused under this round's "
    "own measured imbalance the ladder STOPS and the refusal is the "
    "measurement, because no partition above c = 128 has ever been built and a "
    "larger c would carry an unmeasured reachability ceiling."
)
#: No control cell. R0235's `c = 64` control settled the `N`-independence of the
#: device law at a matched cluster size and R0236's and R0237's points confirmed
#: the fit; this rung's budget is entirely committed to the one 100M build.
CONTROL_CLUSTERS: tuple[int, ...] = ()
LADDER_RULE = (
    "one cell: the c the selection law picks from imbalance measured at THIS N, "
    "taken on the WORST of the five replicate realisations, never the best and "
    "never the mean, with the round's own imbalance margin applied. A refusal, "
    "abort or failure stops the ladder and is recorded as a measurement, with "
    "its GPU time charged to the round."
)

# --------------------------------------------------------------------------- #
# truth — a registered uniform probe, because full truth is not affordable
# --------------------------------------------------------------------------- #
TRUTH_PROBE_ROWS = 500_000
TRUTH_PROBE_SEED = 238_000
TRUTH_METHOD = (
    "exact brute-force fp32 cosine top-k of a registered uniform probe of "
    "500,000 query rows against ALL 100,000,000 substrate rows"
)
RECALL_POPULATION = (
    "a uniform probe of 500,000 of the 100,000,000 substrate rows, drawn "
    "without replacement at seed 238000 registered in the release commit before "
    "the substrate existed, searched against all 100,000,000 rows; no seed set, "
    "no neighbour union, no hub bias (review-0227-01)"
)
TRUTH_AFFORDABILITY_NOTE = (
    "full exact truth over all 100,000,000 rows scales as N^2: R0235 spent "
    "4,223.74 s at 12,500,000, so 100,000,000 costs about 270,000 s = 75 GPU-h "
    "against this round's 6.0 GPU-h cap. review-0237-01 named that figure and "
    "said not to spend it. The probe size is 500,000 rather than R0237's "
    "1,000,000 and that is a BUDGET decision registered as one: at this N the "
    "database doubled while the probe did not, so 1,000,000 rows cost about "
    "2,790 s against 1,395 s, and the queue's margin under a 6.0 GPU-h cap "
    "does not support the difference. The instrument is not weakened where it "
    "matters - the standard error of a mean recall near 0.9957 is about 4.3e-6 "
    "at n = 500,000, four orders below the 0.0957 of slack to the registered "
    "0.90 floor, and about 15 duplicate-family tie-aware zeros are still "
    "expected, enough to adjudicate individually as R0237 did."
)
STRUCTURAL_POPULATION = "all 100,000,000 rows, both in-degree and out-degree"

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
GPU_HOURS_CAP = 6.0
#: Per-cell deadline. Measured build cells: R0233 `~640 s` over `50,000,000`
#: spilled rows, R0235 `~1,300 s` over `100,000,000`, R0236 `2,789 s` over
#: `200,000,000`, R0237 `5,833.9 s` over `400,000,000` — doubling ratios
#: `2.03x`, `2.15x`, `2.09x`. This rung spills `800,000,000`, so the builder
#: alone is expected near `12,200 s`. Its substrate is `153.6 GB` against
#: `~119 GB` of page cache, so unlike every earlier rung its `51` predicted
#: passes cannot be served from cache: review-0237-01 priced that I/O term at
#: `0.67-0.92 h` on rates R0237 measured, giving an expected cell near
#: `14,000 s`. The deadline is therefore `16,000 s`, a `1.14x` margin rather
#: than R0237's `2.0x` — a `2.0x` margin here would exceed the round's whole
#: GPU cap and would be a deadline that can never fire. The binding limit is
#: the cap, and the round says so rather than registering a decorative number.
#: On expiry the parent sets the cooperative flag and waits — it never signals.
BUILD_TIMEOUT_S = 16_000.0

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
# the reachability ceiling AT THIS RUNG — review-0237-01's registered instrument
# --------------------------------------------------------------------------- #
#: Measured on THIS round's own 100,000,000-row substrate, at the partition this
#: round actually builds. No new bulk bytes: `_kmeans` and `_assign` from R0226
#: imported unmodified, then a host-side scan of how many of each probe row's 15
#: exact-truth neighbours share a spill cluster with it. That fraction is the
#: CEILING on strict recall for the partition — no builder can retrieve a
#: neighbour it never compares against — and it is an upper bound, not a
#: predictor (review-0237-01 F1: builder loss is ~0.002 and roughly
#: c-independent, 13x the partition's loss at c = 64).
REACHABILITY_CAPABILITY = "minilm-mixed-100000k-cluster-spill-c400-reachability-v1"
REACHABILITY_SCHEMA = (
    "round0238-minilm-mixed-100000k-c400-reachability-ceiling-v1"
)
REACHABILITY_ROWS = 100_000_000
#: One cell, at the partition this rung builds. R0237 already scanned
#: `64 / 128 / 200 / 400` at 25M and the four-cell scan is sealed and released;
#: repeating it here would spend GPU re-measuring a released capability. What no
#: rung has done is measure the ceiling at the `N` and `c` a graph is actually
#: built at, which is exactly what review-0237-01 F9 said the two-point
#: cross-substrate trend could not license.
REACHABILITY_CLUSTERS: tuple[int, ...] = (400,)
#: R0226's `A_SEED`, the seed every build in this program has used.
REACHABILITY_SEED = 226
#: Below this the partition is discarding too much true structure for a graph
#: built inside it to clear the `0.90` recall floor with any margin. It is a
#: REPORTING threshold on a ceiling, not a queue-aborting floor.
REACHABILITY_CONCERN_FLOOR = 0.99
#: R0237's sealed strict ceilings at 25,000,000 rows, for the trend and for the
#: carry check. Literals so a CPU test can check the comparison arithmetic; the
#: run-time values come from the hash-bound artifact.
R0237_25M_S8_CEILING_REFERENCE: dict[int, float] = {
    64: 0.9998664, 128: 0.9994947, 200: 0.9990732, 400: 0.9977319,
}
REACHABILITY_NOTE = (
    "R0237 measured the strict partition ceiling at 25,000,000 rows across "
    "c = 64/128/200/400 (0.9998664 / 0.9994947 / 0.9990732 / 0.9977319) and "
    "review-0237-01 released it. Two things about that measurement are open and "
    "this cell closes both. First, review-0237-01 F9 blocked "
    "'the ceiling improves with N at fixed c' as a measured scaling: it rested "
    "on two points per c, single realisations, with the 2M point on a DIFFERENT "
    "substrate. Measuring at 100,000,000 on this ladder's own prefix chain adds "
    "a third point at a c that matters, on the same substrate family. Second, "
    "review-0237-01 asked in as many words for `rows_with_zero_reachable` to be "
    "a REPORTED INSTRUMENT at this rung: c = 200 and c = 400 showed 1 and 3 "
    "rows per million with NO reachable true neighbour at 25M, which scales to "
    "~300 rows at 100M. That is not the R0215 tripwire — that fires on realised "
    "graph degree and is checked separately over every row in both directions — "
    "but it is the mechanism R0215 traced to the v1 map's clumps, so it is "
    "counted and published rather than left to be inferred."
)

# --------------------------------------------------------------------------- #
# this IS the 100M rung — there is no verdict to issue, only a rung to build
# --------------------------------------------------------------------------- #
HUNDRED_M_ROWS = 100_000_000
#: R0237 registered the 100M candidate set as `(200, 400)` and its rule selected
#: `400`. This round builds `400` and re-prices `200` from its own measurement.
HUNDRED_M_CANDIDATES: tuple[int, ...] = (200, 400)
#: A candidate whose guarded largest cluster exceeds this multiple of the fitted
#: law's largest observed point is an EXTRAPOLATION.
LAW_RANGE_CEILING = 1.0
HUNDRED_M_RULE = (
    "R0237's registered rule, applied here to a rung this round BUILDS rather "
    "than recommends: among the candidates (200, 400), the one that is (a) "
    "admissible under the guard with the registered margin, (b) INSIDE the "
    "fitted device law's range, and (c) at or above the reachability concern "
    "floor. On R0237's sealed 50M imbalance that is c = 400, and review-0236-01 "
    "F2 and review-0237-01 concur independently. This round re-evaluates the "
    "same rule against its OWN measured imbalance at 100,000,000 rows and "
    "publishes the result; the round is registered at c = 400 and if its own "
    "measurement refused c = 400 the ladder would stop rather than substitute "
    "an unregistered c."
)
#: What the round declines to spend GPU on, registered so the omission is a
#: decision rather than an oversight.
DECLINED_DERISK_NOTE = (
    "review-0237-01 recommended spending ~0.85 GPU-h on a single c = 400 build "
    "at 25M against R0236's existing probe truth, to convert the last "
    "projection in the c = 400 case (builder loss at high c) into a "
    "measurement before the top rung. This round DECLINES it, and registers "
    "why rather than omitting it silently. The de-risking build's purpose was "
    "to avoid a surprise at 100M; this round measures realised recall at "
    "c = 400 at 100,000,000 rows directly, which is strictly more informative "
    "about this rung than a 25M proxy for it, and the budget does not hold "
    "both (~0.85 GPU-h against a 6.0 cap whose expected consumption is ~5.5). "
    "The exposure the decline accepts is stated plainly: if the c = 400 graph "
    "at 100M misses the 0.90 floor, the round will have spent its whole budget "
    "to learn it, where 0.85 GPU-h at 25M would have learned something similar "
    "sooner. The projected realised recall is ~0.9957 strict / ~0.9975 "
    "tie-aware against a 0.90 floor, so the round judges that exposure small "
    "and says so before the run rather than after."
)


class Round0238Error(RuntimeError):
    """The registered R0238 contract changed."""


# --------------------------------------------------------------------------- #
# composition, span, nesting — same shapes as R0236, at this rung's counts
# --------------------------------------------------------------------------- #
def validate_composition(counts: Mapping[str, int]) -> dict[str, Any]:
    """Fail closed unless the assembled universe is exactly the registered mix."""
    total = sum(int(value) for value in counts.values())
    if total != ROWS:
        raise Round0238Error(f"substrate has {total} rows, registered {ROWS}")
    observed: dict[str, Any] = {}
    for name, want in COMPOSITION:
        got = int(counts.get(name, 0))
        if got != want:
            raise Round0238Error(f"{name}: assembled {got} rows, registered {want}")
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
        raise Round0238Error(f"{corpus}: no shards")
    coverage = shards_touched / float(shards_total)
    if coverage < SHARD_COVERAGE_FLOOR:
        raise Round0238Error(
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
        raise Round0238Error("R0238 provenance does not fit the registered key")
    return (corpus << 56) | (shard << 40) | row


def assert_nesting(*, parent: np.ndarray, child: np.ndarray) -> dict[str, Any]:
    """Rung 4 must CONTAIN rung 3, on row ids, and every child row is distinct."""
    parent_keys = provenance_keys(parent)
    child_keys = provenance_keys(child)
    if int(np.unique(child_keys).size) != int(child_keys.size):
        raise Round0238Error("R0238 substrate holds a duplicated source row")
    missing = int(np.setdiff1d(parent_keys, child_keys, assume_unique=False).size)
    if missing != 0:
        raise Round0238Error(
            f"R0238 is not nested on R0236: {missing} rung-3 rows are absent. "
            "Phase 2's whole design is one variable per rung; a non-nested rung "
            "confounds N with the sample."
        )
    positional = bool(
        parent_keys.size <= child_keys.size
        and np.array_equal(parent_keys, child_keys[: parent_keys.size])
    )
    if not positional:
        raise Round0238Error(
            "R0238 prefix is not R0236's rows in R0236's order; the registered "
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
        raise Round0238Error(
            f"R0238 reserve overlaps the training selection on {overlap} rows"
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
            raise Round0238Error(f"R0238 reserve overlaps training in {corpus}")
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
        raise Round0238Error(f"R0238 probe of {size} rows is not drawable from {rows}")
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
        raise Round0238Error("R0238 replicate cell has no realisations")
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
            "NOT a recommendation for a future round: this round IS the "
            "100,000,000-row rung. The same rule R0237 registered is "
            "re-evaluated here against imbalance measured at 100,000,000 rows "
            "on this round's own substrate, and the field name is kept so the "
            "two evaluations are directly comparable. `recommended_clusters` "
            "is what the rule selects from THIS round's measurement; the "
            "cluster count actually built is `selected_clusters` in the "
            "ladder's own `cluster_selection` block, and the result states "
            "plainly if the two differ."
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
            raise Round0238Error(
                f"R0238 zero-recall forensic: {name} has shape {array.shape}, "
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
        raise Round0238Error("R0238 reachability needs a non-empty per-row vector")
    if float(strict.min()) < 0.0 or float(strict.max()) > 1.0:
        raise Round0238Error("R0238 reachability fractions must lie in [0, 1]")
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
    "GREAT2_GRANDPARENT_ROUND_ID",
    "RESERVE_DRAWN_BY_ROUND_ID",
    "GREAT2_GRANDPARENT_ROWS",
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
    "R0237_25M_S8_CEILING_REFERENCE",
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
    "Round0238Error",
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
    "ADVERSE_DRIFT_NOTE",
    "CODE_CORPUS",
    "CODE_POOL_APPENDED_SHARDS",
    "CODE_POOL_EXTENSION",
    "CODE_POOL_PARENT_ROWS",
    "CODE_POOL_PARENT_SHARDS",
    "CODE_POOL_ROWS",
    "CODE_POOL_SHARDS",
    "DECLINED_DERISK_NOTE",
    "INHERITED_PREFIX_SHA256",
    "POOL_EXTENSION_UNIFORMITY",
    "PREDICTION_IMBALANCE_AT_C400",
    "PREDICTION_TOLERANCE_AT_C400",
]
