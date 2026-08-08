"""Frozen contract for R0226 — what builds the 100M k15 graph inside 31.37 GiB?

R0224 (accepted in part by review-0224-01) closed one door: a single-shot
`cuvs.neighbors.nn_descent.build` cannot reach 100M on this box. Its RMM floor
is `1,048 B/row`, measured and exactly linear (`r^2 = 1`), which is
`97.60 GiB` at 100M against a `31.37 GiB` card — `3.11x` over, needing no fit.
Phase 2 of `guides/plan-minilm-100m-v2.md` cannot proceed until some builder is
shown to fit.

This round qualifies two, and **qualification is not adoption**. Nothing here
trains a map, registers a gate, or seals a graph for downstream use.

## The two candidates

**A — `cluster-spill-nnd`.** Paper 0197's out-of-core all-neighbours design,
implemented by hand because the local `cuvs 25.02.01` ships **no**
`cuvs.neighbors.all_neighbors` and **no** `cuvs.cluster` (verified at prepare
time and re-verified inside the child, recorded in the receipt). k-means on a
GPU-sized seeded subsample; every row assigned to its `s` nearest centroids
(spilling, so neighbourhoods straddling a partition boundary survive);
per-cluster local nn-descent on a **memmapped** per-cluster file (R0224's memmap
lever: at matched N = 2M, non-RMM device use is `0.63 GiB` memmap against
`3.38 GiB` materialize); exact cosines recomputed for every local edge; and an
exact incremental global top-k merge.

**B — `sharded-ivf-flat`.** The R0171 path the plan originally specified for
Phase 2: one fp32 `IndexIVFFlat/IP` coarse quantizer trained once, cloned into
row-disjoint GPU shards, **every** query searched against **every** shard at a
fixed nprobe, exact global top-k over the union.

## The recall cost, stated honestly up front

B loses nothing to sharding: searching row-disjoint shards and taking the global
top-k is the same candidate operation as searching their union (R0171's
argument, and its 50k-row equivalence smoke returned identical ids and scores).
Its recall gap versus exact truth is the ordinary IVF `nprobe` gap.

A **does** lose cross-cluster neighbours: a true neighbour `j` of row `i` is
reachable only if `j` sits in one of the `s` clusters `i` was assigned to. That
is a real recall cost and it is the failure mode that produced the v1-150M map
(R0215: edge precision `~0.47` in every population, and edgeless rows piling
into the clumps). So this round measures A's recall against exact truth rather
than assuming spilling repairs it, and it runs the R0215 degree-zero tripwire on
every graph either candidate produces. **A candidate that emits an edgeless row
is disqualified regardless of speed.**

## The instrument, and why it resolves what is being asked

The budget instrument is `device_wide_peak_bytes`: `nvidia-smi
--query-gpu=memory.used` polled from the **parent** at 250 ms against a baseline
read immediately before the child starts, under the queue's exclusive GPU lease.
It is the only device instrument available to *both* candidates, which run in
different environments (A under the RAPIDS env, B under the release venv with
FAISS); a cross-candidate comparison on any instrument only one of them has
would not be a comparison.

Review-0224-01 rejected the inference "it resolves N, therefore it resolves the
term of interest" and it was right to: `N` moves a term that is on the device,
and R0224's term of interest had an unknown location. **The valid argument is
effect size against quantization, and it is registered here in advance:**

* `nvidia-smi --query-gpu=memory.used` reports whole MiB, so its quantum is
  `1 MiB` (`1,048,576 B`); observed idle jitter is absorbed by the pre-launch
  baseline.
* The contrast under test is whether peak device memory is **flat** in `N` at
  fixed shard/cluster capacity, or **linear** in `N`. The linear alternative
  predicts `1,048 B/row` (A, R0224's measured RMM floor) or `1,536 B/row` (B, a
  resident fp32 row) times `dN >= 2,000,000` — that is `>= 1.95 GiB` between
  adjacent rungs.
* Effect-to-quantum is therefore `>= 2,000:1`. The instrument can distinguish
  flat from linear here by three orders of magnitude.
* Unlike R0224's intermediate graph, the *locus* of the term is not in
  question: in both candidates the quantity being bounded is vectors and graph
  buffers the builder itself places on the device.

Every registered instrument is published for every cell, including instruments
that come out flat and instruments that do not apply to a candidate (published
as `null` with the reason). Review-0224-01 caught R0224 dropping one of eight.

## The 100M verdict rule, registered before any measurement

If a candidate's measured device peak is **flat across the ladder** (spread
`<= FLATNESS_TOLERANCE` of its own maximum over rungs at fixed capacity), then
its 100M device footprint **is** that measured plateau: no extrapolation is
performed, and the round says so. If it is not flat, the round reports the
measured per-row slope, the fitted range and the extrapolation factor, and never
divides one projection by another (review-0220-01, review-0224-01).

N-dependent costs that are *not* on the device are reported separately and
never folded into the device verdict: the host global top-k array
(`N * k * 8 B` = `12 GiB` at 100M) and, for A only, the spill scratch on disk.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


ROUND_ID = "0226"

QUALIFICATION_CAPABILITY = "minilm-100m-graph-builder-qualification-v1"
QUALIFICATION_SCHEMA = "round0226-graph-builder-qualification-v1"
RECALL_SCHEMA = "round0226-graph-builder-recall-and-verdict-v1"
BUILD_SCHEMA = "round0226-graph-builder-build-v1"

DIMENSION = 384
GRAPH_K = 15

# --------------------------------------------------------------------------- #
# sealed inputs
# --------------------------------------------------------------------------- #
#: R0216's 2,000,000-row mixed-MiniLM substrate (queue-correction-3). The recall
#: rung runs here because this is the population whose exact k15 truth exists.
SUBSTRATE_2M_PATH = (
    "/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
    "minilm-mixed-2m-substrate-and-exact-k15-graph-v1/substrate.f32.npy"
)
SUBSTRATE_2M_ROWS = 2_000_000
#: R0224's 16,000,000-row benchmark substrate (queue-correction-2), written in a
#: seeded global row permutation so every prefix is a uniform subsample. The
#: scaling rungs read prefixes of this file.
SUBSTRATE_16M_PATH = (
    "/data/latent-basemap/runs/round-0224/queue-correction-2/artifacts/"
    "minilm-mixed-16m-benchmark-substrate-v1/substrate.f32.npy"
)
SUBSTRATE_16M_ROWS = 16_000_000
#: R0220's recomputed exact k15 truth over the 2M substrate. R0216 persisted only
#: the symmetrised fuzzy edge file, so the raw neighbour ids and cosines were
#: rebuilt in R0220 and passed a registered probe against R0216's sealed
#: adjacency (`tie_aware_mean = 0.9999704996744792`, `p10 = 1.0`, floors
#: `0.999 / 0.99`). Review-0220-01 independently reproduced it with its own torch
#: fp32 kernel. This round consumes those bytes rather than re-deriving them.
TRUTH_IDS_PATH = (
    "/data/latent-basemap/runs/round-0220/queue-correction-1/artifacts/"
    "exact-k15-truth/truth-k15-ids.i32.npy"
)
TRUTH_COS_PATH = (
    "/data/latent-basemap/runs/round-0220/queue-correction-1/artifacts/"
    "exact-k15-truth/truth-k15-cos.f32.npy"
)
TRUTH_RECEIPT_PATH = (
    "/data/latent-basemap/runs/round-0220/queue-correction-1/artifacts/"
    "exact-k15-truth/truth-rebuild.json"
)
TRUTH_SCHEMA = "round0220-exact-k15-truth-rebuild-v1"

#: Both substrates are unit-normalised (checked in-node, tolerance below), so
#: squared-euclidean ordering is identical to cosine ordering:
#: ||a-b||^2 = 2 - 2*cos. cuVS nn-descent exposes `sqeuclidean`; FAISS exposes
#: inner product. On unit rows they rank the same object R0216's cosine search
#: ranked.
NORM_TOLERANCE = 1e-3
METRIC_EQUIVALENCE = (
    "substrate rows are unit-normalised to within 1e-3, so sqeuclidean "
    "(2 - 2*cos) and inner product induce the identical ranking; candidate A "
    "builds under sqeuclidean, candidate B searches under inner product, and "
    "both are scored against R0216/R0220's fp32 cosine truth"
)

# --------------------------------------------------------------------------- #
# the ladder
# --------------------------------------------------------------------------- #
#: Ascending. A candidate's ladder stops at its first refusal, abort, timeout or
#: failure — a larger build of a configuration that already failed cannot
#: succeed, and queueing one anyway is how R0224's first attempt came to hold the
#: GPU for 1.30235 h and cost a hard reboot.
LADDER_ROWS: tuple[int, ...] = (2_000_000, 4_000_000, 8_000_000, 16_000_000)
#: Which sealed substrate each rung reads. 2M is R0216's (it has the truth);
#: everything above is a prefix of R0224's benchmark file. Both were drawn under
#: R0216's selection law at the same composition shares (40/25/25/10), so the
#: rungs differ in scale and row identity, not in composition.
SUBSTRATE_BY_ROWS: dict[int, str] = {
    2_000_000: SUBSTRATE_2M_PATH,
    4_000_000: SUBSTRATE_16M_PATH,
    8_000_000: SUBSTRATE_16M_PATH,
    16_000_000: SUBSTRATE_16M_PATH,
}
#: The rung that carries the recall measurement, because it is the only one with
#: exact ground truth.
RECALL_ROWS = 2_000_000

PROJECTION_ROWS = 100_000_000
#: The rungs Phase 2 has to serve.
PHASE2_RUNGS: tuple[int, ...] = (
    6_250_000,
    12_500_000,
    25_000_000,
    50_000_000,
    100_000_000,
)

# --------------------------------------------------------------------------- #
# candidate A — cluster-spill nn-descent
# --------------------------------------------------------------------------- #
CANDIDATE_A = "cluster-spill-nnd"
#: Spill factor. Paper 0197's default; each row joins its `s` nearest clusters so
#: a neighbourhood that straddles a boundary is still built somewhere.
A_SPILL = 2
#: Target rows per cluster. The device cost of a per-cluster build is
#: `~1048 B/row x cluster_rows` (R0224's measured RMM floor), so this constant —
#: not N — is what sets candidate A's device footprint. Held FIXED across the
#: ladder, which is the whole point: it makes flatness in N a measurement.
A_CLUSTER_TARGET_ROWS = 1_000_000
#: A floor on cluster count so the spill is not degenerate: with `c = s` every
#: row lands in every cluster and nothing is partitioned.
A_MIN_CLUSTERS = 8
#: Refuse a cell whose largest realised cluster exceeds this. Checked inside the
#: child after assignment, before any per-cluster build is launched.
A_CLUSTER_CAPACITY_ROWS = 4_000_000
A_KMEANS_SUBSAMPLE_ROWS = 1_000_000
A_KMEANS_ITERATIONS = 25
A_SEED = 226
A_ASSIGN_BLOCK = 200_000
#: Rows staged host-side per copy when a cluster is moved onto the device for the
#: exact-cosine pass. Bounds the transient anonymous copy.
A_CLUSTER_STAGE_ROWS = 500_000
#: nn-descent parameters, frozen at R0220/R0223's qualified setting
#: (`nnd-gd32-igd48-it20`), which reached tie-aware `0.994164` at 2M monolithic.
A_GRAPH_DEGREE = 32
A_INTERMEDIATE_DEGREE = 48
A_MAX_ITERATIONS = 20
A_METRIC = "sqeuclidean"
#: Spill scratch is written in groups so peak disk stays bounded: with 292 GiB
#: free on `/data`, a single-pass 100M spill (`307 GB`) would not fit, and the
#: grouped scheme is the one that generalises. Total substrate reads are
#: `groups x N x 1536 B`; peak scratch is `<= this budget`.
A_SCRATCH_BUDGET_BYTES = 24 * 1024 ** 3


def a_cluster_count(rows: int) -> int:
    """Number of k-means clusters for candidate A at this N."""
    rows = int(rows)
    if rows <= 0:
        raise Round0226Error("R0226 cluster count needs rows > 0")
    return max(
        A_MIN_CLUSTERS,
        int(math.ceil(rows * A_SPILL / float(A_CLUSTER_TARGET_ROWS))),
    )


def a_spill_groups(rows: int) -> int:
    """How many passes over the substrate the spill write is split into."""
    spill_bytes = int(rows) * A_SPILL * DIMENSION * 4
    return max(1, int(math.ceil(spill_bytes / float(A_SCRATCH_BUDGET_BYTES))))


# --------------------------------------------------------------------------- #
# candidate B — sharded fp32 IVF (the R0171 path)
# --------------------------------------------------------------------------- #
CANDIDATE_B = "sharded-ivf-flat"
#: Rows per GPU shard. Like A's cluster target, this — not N — sets B's device
#: footprint (`shard_rows x 1536 B` of resident fp32), and it is held fixed
#: across the ladder for the same reason.
B_SHARD_ROWS = 1_000_000
#: R0171's registered graph law, unchanged.
B_NLIST = 8192
B_NPROBE = 64
B_QUERY_BLOCK = 16_384
#: Search width per shard. `k + 1` so a row's own vector, which lives in exactly
#: one shard, can be dropped after the merge without costing a neighbour.
B_SEARCH_K = GRAPH_K + 1
B_TRAIN_ROWS = 1_000_000
B_SEED = 226
#: FAISS's per-GPU scratch arena. Declared so the guard can charge it.
B_TEMP_MEMORY_BYTES = 1536 * 1024 ** 2


def b_shard_count(rows: int) -> int:
    return max(1, int(math.ceil(int(rows) / float(B_SHARD_ROWS))))


CANDIDATES: tuple[str, ...] = (CANDIDATE_A, CANDIDATE_B)

# --------------------------------------------------------------------------- #
# instruments
# --------------------------------------------------------------------------- #
INSTRUMENTS: tuple[str, ...] = (
    "device_wide_peak_bytes",
    "device_wide_peak_over_baseline_bytes",
    "nvidia_smi_per_process_peak_bytes",
    "child_device_peak_sampled_bytes",
    "rmm_peak_bytes",
    "host_rss_peak_bytes",
    "host_anon_peak_bytes",
    "host_vmhwm_bytes",
    "system_swap_growth_bytes",
)
DEVICE_BUDGET_INSTRUMENT = "device_wide_peak_bytes"
INSTRUMENT_APPLICABILITY: dict[str, str] = {
    "device_wide_peak_bytes": "both",
    "device_wide_peak_over_baseline_bytes": "both",
    "nvidia_smi_per_process_peak_bytes": "both",
    "child_device_peak_sampled_bytes": "both",
    # RMM is a RAPIDS allocator. FAISS does not route through it, so it is
    # published as null for candidate B rather than omitted.
    "rmm_peak_bytes": CANDIDATE_A,
    "host_rss_peak_bytes": "both",
    "host_anon_peak_bytes": "both",
    "host_vmhwm_bytes": "both",
    "system_swap_growth_bytes": "both",
}
INSTRUMENT_NOTE = (
    "device_wide_peak_bytes is the budget instrument and the only device "
    "instrument both candidates share, because A runs under the RAPIDS env and "
    "B under the release venv with FAISS. rmm_peak_bytes is null for B by "
    "construction (FAISS does not allocate through RMM) and is published as "
    "null rather than dropped from the table."
)
#: Quantum of `nvidia-smi --query-gpu=memory.used`, which reports whole MiB.
DEVICE_INSTRUMENT_QUANTUM_BYTES = 1024 * 1024
SENSITIVITY_ARGUMENT = (
    "effect size against quantization. The contrast under test is flat-in-N "
    "versus linear-in-N device peak at fixed shard/cluster capacity. The linear "
    "alternative predicts 1048 B/row (A, R0224's measured RMM floor) or 1536 "
    "B/row (B, a resident fp32 row) times dN >= 2,000,000, i.e. >= 1.95 GiB "
    "between adjacent rungs, against a 1 MiB instrument quantum: an "
    "effect-to-quantum ratio >= 2000:1. This is NOT the invalid 'it resolves N "
    "therefore it resolves the term of interest' inference review-0224-01 "
    "rejected; the term bounded here is vectors and graph buffers the builder "
    "places on the device, so its locus is not in question, only its size."
)
#: A candidate's device peak counts as flat across the ladder when its spread is
#: within this fraction of its own maximum. Registered before measurement.
FLATNESS_TOLERANCE = 0.10

# --------------------------------------------------------------------------- #
# budgets, guard, watchdog
# --------------------------------------------------------------------------- #
#: The card, as the owner's mandate states it.
DEVICE_TOTAL_BYTES = 31.37 * 1024 ** 3
GUARD_DEVICE_BUDGET_BYTES = 24 * 1024 ** 3
#: **Anonymous**, not RSS. Review-0224-01 established that VmHWM/RSS counts clean
#: file-backed pages, which are evicted rather than swapped, so projecting them
#: as a memory requirement was an artifact. Anonymous bytes are the swappable
#: ones and swapping is what wedged the box.
GUARD_HOST_ANON_BUDGET_BYTES = 60 * 1024 ** 3
#: Swap is judged as GROWTH over a baseline read immediately before launch. An
#: absolute threshold falsely aborts everything, because idle daemons already
#: hold swap on this box.
GUARD_SWAP_GROWTH_ABORT_BYTES = 1 * 1024 ** 3
WATCHDOG_POLL_S = 0.25
#: Never SIGKILL a process holding a CUDA context: that is what deadlocked RCU,
#: put PID 1 in D state and cost a hard reboot in R0224's first attempt.
GUARD_SIGTERM_GRACE_S = 180.0
BUILD_TIMEOUT_S = 3_600.0
SAMPLE_INTERVAL_S = 0.005
GUARD_FIXED_OVERHEAD_BYTES = 3 * 1024 ** 3
GUARD_DEVICE_CONTEXT_BYTES = 1024 ** 3

GUARD_BUDGET_NOTE = (
    "device budget 24 GiB of the card's 31.37 GiB; host ANONYMOUS budget 60 GiB "
    "of the box's 123 GiB. A cell whose predicted footprint exceeds either is "
    "refused before launch and recorded as refused_a_priori with its "
    "prediction. A refusal is data: it measures where a builder stops being "
    "launchable on this box."
)

GPU_HOURS_CAP = 3.0


class Round0226Error(RuntimeError):
    """The registered R0226 qualification contract changed."""


def predict_footprint(*, candidate: str, rows: int) -> dict[str, Any]:
    """Expected device and host-anonymous bytes for a cell, BEFORE it launches.

    Predictor only. Every published number is measured; this exists so a cell
    that cannot fit is never launched.
    """
    rows = int(rows)
    if rows <= 0:
        raise Round0226Error("R0226 footprint prediction needs rows > 0")
    if candidate not in CANDIDATES:
        raise Round0226Error(f"R0226 unknown candidate {candidate!r}")

    row_bytes = DIMENSION * 4
    dataset_bytes = rows * row_bytes
    # Both candidates carry a host-resident global top-k accumulator: ids int32
    # plus cosines float32, k wide. This is the one host term that scales with N
    # and it is anonymous, so it is charged.
    topk_bytes = rows * GRAPH_K * 8
    terms: dict[str, int] = {
        "global_topk_accumulator_bytes": int(topk_bytes),
        "fixed_overhead_bytes": int(GUARD_FIXED_OVERHEAD_BYTES),
    }

    if candidate == CANDIDATE_A:
        clusters = a_cluster_count(rows)
        groups = a_spill_groups(rows)
        capacity = A_CLUSTER_CAPACITY_ROWS
        # Device: the largest admissible per-cluster nn-descent build at
        # R0224's measured RMM floor, plus the exact-cosine recompute chunk
        # (one cluster's vectors resident), plus the k-means subsample, plus
        # context. The k-means and per-cluster phases are sequential, so the
        # device peak is their maximum, not their sum.
        kmeans_device = A_KMEANS_SUBSAMPLE_ROWS * row_bytes + clusters * row_bytes
        cluster_device = int(capacity * 1048.0) + capacity * row_bytes
        device_bytes = int(
            max(kmeans_device, cluster_device) + GUARD_DEVICE_CONTEXT_BYTES
        )
        # Host anonymous: the accumulator, the per-row assignment table, one
        # spill-group's worth of write buffering, and overhead. Cluster files
        # are read back as memmaps (R0224's lever) and are file-backed, not
        # anonymous.
        assignment_bytes = rows * A_SPILL * 4
        spill_buffer_bytes = A_ASSIGN_BLOCK * A_SPILL * row_bytes
        stage_bytes = A_CLUSTER_STAGE_ROWS * row_bytes
        anonymous_bytes = int(
            topk_bytes
            + assignment_bytes
            + spill_buffer_bytes
            + stage_bytes
            + GUARD_FIXED_OVERHEAD_BYTES
        )
        terms.update({
            "cluster_device_stage_bytes": int(stage_bytes),
            "clusters": int(clusters),
            "spill_groups": int(groups),
            "cluster_capacity_rows": int(capacity),
            "kmeans_device_bytes": int(kmeans_device),
            "largest_cluster_device_bytes": int(cluster_device),
            "assignment_table_bytes": int(assignment_bytes),
            "spill_write_buffer_bytes": int(spill_buffer_bytes),
            "peak_scratch_disk_bytes": int(
                math.ceil(rows * A_SPILL / float(groups)) * row_bytes
            ),
            "substrate_read_bytes": int(groups * dataset_bytes),
        })
        device_model = (
            "max(kmeans subsample + centroids, cluster_capacity x (1048 B RMM "
            "floor + 1536 B resident vectors)) + 1 GiB context. Sequential "
            "phases, so the peak is the max and not the sum. Set by "
            "A_CLUSTER_CAPACITY_ROWS, not by N."
        )
    else:
        shards = b_shard_count(rows)
        shard_rows = min(rows, B_SHARD_ROWS)
        # Device: one resident fp32 shard, the coarse centroids, one query
        # block, FAISS's scratch arena, and context.
        shard_device = shard_rows * row_bytes
        device_bytes = int(
            shard_device
            + B_NLIST * row_bytes
            + B_QUERY_BLOCK * row_bytes
            + B_TEMP_MEMORY_BYTES
            + GUARD_DEVICE_CONTEXT_BYTES
        )
        # Host anonymous: accumulator, one shard staged for `add`, the training
        # subsample, per-block search results, and overhead.
        staging_bytes = shard_rows * row_bytes
        train_bytes = min(rows, B_TRAIN_ROWS) * row_bytes
        block_bytes = B_QUERY_BLOCK * B_SEARCH_K * 8 * 2
        anonymous_bytes = int(
            topk_bytes
            + staging_bytes
            + train_bytes
            + block_bytes
            + GUARD_FIXED_OVERHEAD_BYTES
        )
        terms.update({
            "shards": int(shards),
            "shard_rows": int(shard_rows),
            "resident_shard_bytes": int(shard_device),
            "coarse_centroid_bytes": int(B_NLIST * row_bytes),
            "faiss_temp_memory_bytes": int(B_TEMP_MEMORY_BYTES),
            "shard_staging_bytes": int(staging_bytes),
            "training_subsample_bytes": int(train_bytes),
            "peak_scratch_disk_bytes": 0,
            "substrate_read_bytes": int(shards * dataset_bytes + dataset_bytes),
        })
        device_model = (
            "one resident fp32 shard (shard_rows x 1536 B) + nlist centroids + "
            "one query block + FAISS temp arena + 1 GiB context. Set by "
            "B_SHARD_ROWS, not by N."
        )

    return {
        "candidate": str(candidate),
        "rows": rows,
        "dimension": DIMENSION,
        "dataset_bytes": int(dataset_bytes),
        "terms": terms,
        "predicted_device_bytes": int(device_bytes),
        "predicted_device_gib": device_bytes / (1024 ** 3),
        "predicted_host_anon_bytes": int(anonymous_bytes),
        "predicted_host_anon_gib": anonymous_bytes / (1024 ** 3),
        "predicted_file_backed_bytes": int(dataset_bytes),
        "device_model": device_model,
        "host_model": (
            "anonymous only. File-backed pages (the substrate memmap and, for "
            "A, the spill files) are clean page cache: evicted, not swapped, so "
            "they are reported separately and never charged to the swappable "
            "budget (review-0224-01)."
        ),
    }


def guard_decision(
    *,
    candidate: str,
    rows: int,
    device_budget_bytes: int = GUARD_DEVICE_BUDGET_BYTES,
    host_anon_budget_bytes: int = GUARD_HOST_ANON_BUDGET_BYTES,
) -> dict[str, Any]:
    """Launch this cell, or refuse it and record the refusal as a measurement."""
    prediction = predict_footprint(candidate=candidate, rows=rows)
    device_over = prediction["predicted_device_bytes"] > int(device_budget_bytes)
    host_over = prediction["predicted_host_anon_bytes"] > int(host_anon_budget_bytes)
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
    return {
        "prediction": prediction,
        "device_budget_bytes": int(device_budget_bytes),
        "host_anon_budget_bytes": int(host_anon_budget_bytes),
        "device_over_budget": bool(device_over),
        "host_over_budget": bool(host_over),
        "allowed": not (device_over or host_over),
        "refused_a_priori": bool(device_over or host_over),
        "refusal_reasons": reasons,
        "budget_note": GUARD_BUDGET_NOTE,
    }


def ladder_settings() -> tuple[dict[str, Any], ...]:
    """Every (candidate, N) cell of the registered matrix, in run order."""
    out: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for rows in LADDER_ROWS:
            out.append({
                "id": f"{candidate}-n{rows}",
                "candidate": str(candidate),
                "rows": int(rows),
                "dimension": DIMENSION,
                "k": GRAPH_K,
                "substrate": SUBSTRATE_BY_ROWS[int(rows)],
                "emit_graph": int(rows) == RECALL_ROWS,
            })
    return tuple(out)


# --------------------------------------------------------------------------- #
# the exact global top-k merge
# --------------------------------------------------------------------------- #
_ABSENT = np.iinfo(np.int64).max


def merge_into_topk(
    top_ids: np.ndarray,
    top_cos: np.ndarray,
    *,
    rows: np.ndarray,
    candidate_ids: np.ndarray,
    candidate_cos: np.ndarray,
    k: int = GRAPH_K,
) -> None:
    """Merge a candidate block into a global top-k accumulator, exactly.

    In place on `top_ids` / `top_cos` at the positions named by `rows`.

    Top-k over a union is associative, so merging block by block gives the same
    answer as sorting every candidate at once. What has to be right is what
    counts as a candidate:

    * **FAISS returns id `-1`** with a sentinel score when a query's probed
      lists hold fewer than `k` vectors *in this shard*. R0209 shipped a merge
      that ranked those slots as though they were neighbours; on a population
      whose shards are not homogeneous that silently fabricates edges. They are
      excluded here, never ranked, and a slot that never fills is re-emitted as
      `-1` so the caller's completeness guard still fails closed.
    * **Self** is excluded: a row is not its own neighbour.
    * **Duplicates** are collapsed to one occurrence, so a candidate list that
      repeats an id cannot inflate a row's degree.
    * Ties break on **lower global id**, deterministically (R0166's rule).
    """
    top_ids = np.asarray(top_ids)
    top_cos = np.asarray(top_cos)
    rows = np.asarray(rows, dtype=np.int64)
    candidate_ids = np.asarray(candidate_ids)
    candidate_cos = np.asarray(candidate_cos)
    if top_ids.shape != top_cos.shape or top_ids.shape[1] != int(k):
        raise Round0226Error("R0226 top-k accumulator has the wrong geometry")
    if candidate_ids.shape != candidate_cos.shape:
        raise Round0226Error("R0226 candidate ids and cosines disagree in shape")
    if candidate_ids.shape[0] != rows.shape[0]:
        raise Round0226Error("R0226 candidate block and row index disagree")
    if rows.size == 0:
        return

    merged_ids = np.concatenate(
        (top_ids[rows].astype(np.int64), candidate_ids.astype(np.int64)), axis=1
    )
    merged_cos = np.concatenate(
        (top_cos[rows].astype(np.float64), candidate_cos.astype(np.float64)), axis=1
    )
    valid = (
        (merged_ids >= 0)
        & (merged_ids != rows[:, None])
        & np.isfinite(merged_cos)
    )
    merged_ids = np.where(valid, merged_ids, _ABSENT)
    merged_cos = np.where(valid, merged_cos, -np.inf)

    # Collapse repeats of the same id within a row. Sorting by (id ascending,
    # cosine descending) puts equal ids adjacent and puts the best-scoring
    # occurrence first, so the survivor does not depend on which block a
    # duplicate arrived in. That is what makes the incremental merge give the
    # same answer as sorting every candidate at once.
    order = np.lexsort((-merged_cos, merged_ids), axis=1)
    ordered = np.take_along_axis(merged_ids, order, axis=1)
    fresh = np.ones(ordered.shape, dtype=bool)
    fresh[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
    first = np.empty_like(fresh)
    np.put_along_axis(first, order, fresh, axis=1)
    merged_ids = np.where(first, merged_ids, _ABSENT)
    merged_cos = np.where(first, merged_cos, -np.inf)

    # The merged table is narrow, so sort it completely rather than partition:
    # argpartition would pick an arbitrary member of a score tie at the k-th
    # boundary, and this round's substrate provably contains exact-duplicate
    # clusters (one with 1,377 members).
    keep = np.lexsort((merged_ids, -merged_cos), axis=1)[:, : int(k)]
    kept_ids = np.take_along_axis(merged_ids, keep, axis=1)
    kept_cos = np.take_along_axis(merged_cos, keep, axis=1)
    unfilled = kept_ids == _ABSENT
    kept_ids = np.where(unfilled, -1, kept_ids)
    kept_cos = np.where(unfilled, -np.inf, kept_cos)
    top_ids[rows] = kept_ids.astype(top_ids.dtype, copy=False)
    top_cos[rows] = kept_cos.astype(top_cos.dtype, copy=False)


# --------------------------------------------------------------------------- #
# verdicts
# --------------------------------------------------------------------------- #
def flatness(values: Sequence[float]) -> dict[str, Any]:
    """Is a measured series flat, by the tolerance registered in advance?"""
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise Round0226Error("R0226 flatness needs finite measurements")
    largest = float(array.max())
    spread = largest - float(array.min())
    relative = spread / largest if largest > 0 else 0.0
    return {
        "n_points": int(array.size),
        "max_bytes": largest,
        "min_bytes": float(array.min()),
        "spread_bytes": spread,
        "relative_spread": relative,
        "tolerance": FLATNESS_TOLERANCE,
        "flat": bool(relative <= FLATNESS_TOLERANCE),
        "instrument_quantum_bytes": DEVICE_INSTRUMENT_QUANTUM_BYTES,
        "spread_in_quanta": spread / float(DEVICE_INSTRUMENT_QUANTUM_BYTES),
    }


def device_verdict_at_100m(
    *, candidate: str, rows: Sequence[int], device_peaks: Sequence[float]
) -> dict[str, Any]:
    """The registered 100M device rule.

    Flat across the ladder -> the 100M footprint IS the measured plateau and no
    extrapolation happens. Not flat -> report the measured slope with its fitted
    range and extrapolation factor, and say plainly that it is an extrapolation.
    """
    sizes = np.asarray(list(rows), dtype=np.float64)
    peaks = np.asarray(list(device_peaks), dtype=np.float64)
    if sizes.size != peaks.size or sizes.size < 2:
        raise Round0226Error("R0226 device verdict needs >= 2 matched points")
    flat = flatness(peaks)
    budget = float(DEVICE_TOTAL_BYTES)
    if flat["flat"]:
        projected = float(peaks.max())
        return {
            "candidate": str(candidate),
            "method": "measured plateau, no extrapolation",
            "flatness": flat,
            "measured_rows": [int(value) for value in sizes],
            "device_bytes_at_100m": projected,
            "device_gib_at_100m": projected / (1024 ** 3),
            "card_gib": budget / (1024 ** 3),
            "headroom_gib": (budget - projected) / (1024 ** 3),
            "fits_100m": bool(projected <= budget),
            "extrapolation_factor": 1.0,
            "is_extrapolation": False,
        }
    # Least squares slope in bytes per row, reported with its own range.
    design = np.vstack([sizes, np.ones_like(sizes)]).T
    coefficients, *_ = np.linalg.lstsq(design, peaks, rcond=None)
    slope = float(coefficients[0])
    intercept = float(coefficients[1])
    predicted = design @ coefficients
    residual = float(((peaks - predicted) ** 2).sum())
    total = float(((peaks - peaks.mean()) ** 2).sum())
    r_squared = 1.0 - residual / total if total > 0 else 0.0
    projected = slope * float(PROJECTION_ROWS) + intercept
    return {
        "candidate": str(candidate),
        "method": "linear extrapolation in N (device peak was NOT flat)",
        "flatness": flat,
        "measured_rows": [int(value) for value in sizes],
        "bytes_per_row": slope,
        "intercept_bytes": intercept,
        "r_squared": r_squared,
        "fitted_range_rows": [int(sizes.min()), int(sizes.max())],
        "extrapolation_factor": float(PROJECTION_ROWS) / float(sizes.max()),
        "is_extrapolation": True,
        "device_bytes_at_100m": float(projected),
        "device_gib_at_100m": float(projected) / (1024 ** 3),
        "card_gib": budget / (1024 ** 3),
        "headroom_gib": (budget - float(projected)) / (1024 ** 3),
        "fits_100m": bool(projected <= budget),
    }


def power_law(sizes: Sequence[int], seconds: Sequence[float]) -> dict[str, Any]:
    """Least-squares log-log fit `t = a * N**b`, with its own R^2."""
    n = np.asarray(list(sizes), dtype=np.float64)
    t = np.asarray(list(seconds), dtype=np.float64)
    if n.size != t.size or n.size < 2 or not np.all(np.isfinite(t)) or t.min() <= 0:
        raise Round0226Error("R0226 power-law fit needs >= 2 positive matched points")
    design = np.vstack([np.log(n), np.ones_like(n)]).T
    coefficients, *_ = np.linalg.lstsq(design, np.log(t), rcond=None)
    predicted = design @ coefficients
    residual = float(((np.log(t) - predicted) ** 2).sum())
    total = float(((np.log(t) - np.log(t).mean()) ** 2).sum())
    return {
        "exponent": float(coefficients[0]),
        "coefficient": float(np.exp(coefficients[1])),
        "r_squared": 1.0 - residual / total if total > 0 else 0.0,
        "fitted_range_rows": [int(n.min()), int(n.max())],
        "extrapolation_factor": float(PROJECTION_ROWS) / float(n.max()),
    }


def project_wall(fit: Mapping[str, Any], *, rows: int = PROJECTION_ROWS) -> dict[str, Any]:
    """A labelled wall projection. Never divided by another projection."""
    seconds = float(fit["coefficient"]) * float(rows) ** float(fit["exponent"])
    return {
        "rows": int(rows),
        "projected_seconds": seconds,
        "projected_hours": seconds / 3600.0,
        "is_projection": True,
        "fitted_range_rows": list(fit["fitted_range_rows"]),
        "extrapolation_factor": float(fit["extrapolation_factor"]),
        "r_squared": float(fit["r_squared"]),
        "discipline": (
            "labelled projection; no projection in this round is divided by "
            "another projection or reported as a speedup ratio"
        ),
    }


def rung_recommendation(
    *, verdicts: Mapping[str, Mapping[str, Any]], recalls: Mapping[str, Any]
) -> dict[str, Any]:
    """Which builder Phase 2 should use at each rung, with headroom.

    A candidate is *eligible* when it fits the card at 100M under the registered
    device rule, produced no zero-degree row anywhere, and cleared the recall
    floor at the rung where truth exists. Among eligible candidates the cheaper
    measured wall wins. If none is eligible, the recommendation says so — that
    is a real finding and Phase 2's design has to change again.
    """
    eligible: list[str] = []
    for candidate in CANDIDATES:
        verdict = verdicts.get(candidate) or {}
        recall = recalls.get(candidate) or {}
        if (
            verdict.get("fits_100m")
            and recall.get("zero_degree_rows") == 0
            and recall.get("clears_recall_floor")
        ):
            eligible.append(candidate)
    out: dict[str, Any] = {
        "eligible_candidates": eligible,
        "eligibility_rule": (
            "fits the card at 100M under the registered device rule, zero "
            "degree-zero rows (R0215 tripwire), and clears the >=0.90 sampled "
            "exact-recall floor at the rung where exact truth exists"
        ),
        "rungs": {},
    }
    for rung in PHASE2_RUNGS:
        if not eligible:
            out["rungs"][str(rung)] = {
                "builder": None,
                "reason": "no candidate qualified; Phase 2 needs a new design",
            }
            continue
        chosen = eligible[0]
        best = None
        for candidate in eligible:
            wall = (recalls.get(candidate) or {}).get("wall_seconds_at_largest_n")
            if wall is None:
                continue
            if best is None or float(wall) < best:
                best = float(wall)
                chosen = candidate
        verdict = verdicts[chosen]
        out["rungs"][str(rung)] = {
            "builder": chosen,
            "device_gib": verdict.get("device_gib_at_100m"),
            "headroom_gib": verdict.get("headroom_gib"),
            "device_basis": verdict.get("method"),
        }
    return out


__all__ = [
    "A_CLUSTER_CAPACITY_ROWS",
    "A_CLUSTER_STAGE_ROWS",
    "A_CLUSTER_TARGET_ROWS",
    "A_ASSIGN_BLOCK",
    "A_GRAPH_DEGREE",
    "A_INTERMEDIATE_DEGREE",
    "A_KMEANS_ITERATIONS",
    "A_KMEANS_SUBSAMPLE_ROWS",
    "A_MAX_ITERATIONS",
    "A_METRIC",
    "A_MIN_CLUSTERS",
    "A_SCRATCH_BUDGET_BYTES",
    "A_SEED",
    "A_SPILL",
    "B_NLIST",
    "B_NPROBE",
    "B_QUERY_BLOCK",
    "B_SEARCH_K",
    "B_SHARD_ROWS",
    "B_TEMP_MEMORY_BYTES",
    "B_TRAIN_ROWS",
    "BUILD_SCHEMA",
    "BUILD_TIMEOUT_S",
    "CANDIDATES",
    "CANDIDATE_A",
    "CANDIDATE_B",
    "DEVICE_BUDGET_INSTRUMENT",
    "DEVICE_INSTRUMENT_QUANTUM_BYTES",
    "DEVICE_TOTAL_BYTES",
    "DIMENSION",
    "FLATNESS_TOLERANCE",
    "GPU_HOURS_CAP",
    "GRAPH_K",
    "GUARD_BUDGET_NOTE",
    "GUARD_DEVICE_BUDGET_BYTES",
    "GUARD_HOST_ANON_BUDGET_BYTES",
    "GUARD_SIGTERM_GRACE_S",
    "GUARD_SWAP_GROWTH_ABORT_BYTES",
    "INSTRUMENTS",
    "INSTRUMENT_APPLICABILITY",
    "INSTRUMENT_NOTE",
    "LADDER_ROWS",
    "METRIC_EQUIVALENCE",
    "NORM_TOLERANCE",
    "PHASE2_RUNGS",
    "PROJECTION_ROWS",
    "QUALIFICATION_CAPABILITY",
    "QUALIFICATION_SCHEMA",
    "RECALL_ROWS",
    "RECALL_SCHEMA",
    "ROUND_ID",
    "Round0226Error",
    "SAMPLE_INTERVAL_S",
    "SENSITIVITY_ARGUMENT",
    "SUBSTRATE_16M_PATH",
    "SUBSTRATE_16M_ROWS",
    "SUBSTRATE_2M_PATH",
    "SUBSTRATE_2M_ROWS",
    "SUBSTRATE_BY_ROWS",
    "TRUTH_COS_PATH",
    "TRUTH_IDS_PATH",
    "TRUTH_RECEIPT_PATH",
    "TRUTH_SCHEMA",
    "WATCHDOG_POLL_S",
    "a_cluster_count",
    "a_spill_groups",
    "b_shard_count",
    "device_verdict_at_100m",
    "flatness",
    "guard_decision",
    "ladder_settings",
    "merge_into_topk",
    "power_law",
    "predict_footprint",
    "project_wall",
    "rung_recommendation",
]
