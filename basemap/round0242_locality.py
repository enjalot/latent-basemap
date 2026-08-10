"""R0242 — is the measured neighbour-graph loss SPATIALLY CONCENTRATED?

R0241 qualified the `100,000,000`-row `k = 15` graph as a *neighbour graph*:
tie-aware recall `0.997942`, strict `0.995903`, and a clean degree-zero tripwire
over all `1.5e9` entries. review-0241-01 accepted it and blocked the atlas claim
for one reason, which this module exists to test: **the probe is spatially
blind.** In R0228 a uniform map-quality average went null over exactly the
population where a structured displacement effect ran `+3.94` sd. A uniform
average is not evidence of the absence of structure.

R0241 held both halves of a locality test and never joined them. This module is
the join, and it is free — every input already exists on disk:

* **loss vs in-degree** — R0241 measured per-row in-degree over all
  `100,000,000` rows in node A and per-row loss over `500,000` rows in node B,
  in the same queue, and never asked whether loss-carrying rows are antihubs.
* **loss vs cluster** — the `c = 400` cluster-spill partition is a *spatial*
  stratifier the probe cannot see. If loss concentrates in particular clusters
  rather than spreading uniformly, that is the R0228 signature, and it changes
  Phase 3's design *before* an atlas is trained rather than after.

The decomposition review-0241-01/F3 computed is re-derived here rather than
carried from its prose: of `14,330` strict-loss-carrying probe rows, some are
**partition-limited** (their exact truth is not reachable inside the `c = 400`
partition at all) and the rest are **builder loss inside the partition**. The two
have different causes and are analysed separately, because only the second can
carry a spatial signature that is not mechanically forced by the partition
itself.

Part B — the fuzzy symmetrisation — adds the check the v1 defect actually needs:
R0034 found `2,779,481` rows (`~1.85%`) with no valid edges **after
canonicalization**, and R0215 showed those rows produced the clumps. No round in
this program has ever run a degree-zero tripwire *post-canonicalization* at
`100,000,000` rows. `canonical_undirected_degrees` and `post_canonical_tripwire`
are that check, and `tests/test_round0242_cpu_smoke.py` plants the defect and
proves they catch it.

This module contains no signalling construct, starts no child process and
creates no CUDA context. Every registered check it needs is imported from the
release, never re-typed.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from basemap.round0238_rung5 import GRAPH_K, SPILL
from basemap.round0241_qualify import (
    REGISTERED_GRAPH_COS_SHA256,
    REGISTERED_GRAPH_IDS_SHA256,
    REGISTERED_LADDER_RECEIPT_SHA256,
    REGISTERED_SELECTED_CELL,
    REGISTERED_SELECTED_CLUSTERS,
    StageGuard,
)

ROUND_ID = "0242"
ROWS = 100_000_000
DIMENSION = 384
CLUSTERS = REGISTERED_SELECTED_CLUSTERS
PARTITION_SEED = 226
PROBE_ROWS = 500_000

LOCALITY_CAPABILITY = "minilm-mixed-100000k-k15-loss-locality-v1"
FUZZY_CAPABILITY = "minilm-mixed-100000k-cluster-spill-k15-fuzzy-graph-v1"
CANONICAL_CAPABILITY = "minilm-mixed-100000k-k15-post-canonical-tripwire-v1"

LOCALITY_FILE = "loss-locality.json"
LOCALITY_EARLY_FILE = "loss-locality-first-write.json"
FUZZY_FILE = "fuzzy-graph.json"
LOCALITY_SCHEMA = "round0242-minilm-mixed-100000k-k15-loss-locality-v1"
LOCALITY_EARLY_SCHEMA = "round0242-minilm-mixed-100000k-k15-loss-locality-first-write-v1"
FUZZY_SCHEMA = "round0242-minilm-mixed-100000k-k15-fuzzy-and-canonical-tripwire-v1"

GPU_HOURS_CAP = 12.0
LOCALITY_DEADLINE_S = 3_600.0
FUZZY_DEADLINE_S = 32_400.0
LOCALITY_STAGE_BUDGET_S = 3_000.0
FUZZY_STAGE_BUDGET_S = 30_000.0

#: The permutation budget for every locality test in this round. Registered in
#: advance so the p-value's resolution is a pre-declared property and not a
#: number chosen after seeing the statistic. The smallest p any of these tests
#: can report is `1 / (PERMUTATIONS + 1)`.
PERMUTATIONS = 10_000
PERMUTATION_SEED = 242_000
#: How many of the `400` clusters the concentration statistic looks at. Twenty
#: is `5%` of the partition, so a uniform null puts `~5%` of the loss mass there
#: (plus selection noise, which the permutation null measures exactly).
CONCENTRATION_TOP_M = 20

# --------------------------------------------------------------------------- #
# the registered halt condition — declared BEFORE Part A runs
# --------------------------------------------------------------------------- #
#: Part B (the `~4.3 h` fuzzy symmetrisation) is a PRODUCT step. Part A is a
#: SCIENCE check whose positive outcome is worth more than the product. These
#: are the numbers that stop the product, fixed in the round file before the
#: measurement exists.
HALT_P_VALUE = 0.001
HALT_TOP_M_SHARE = 0.25
HALT_SINGLE_CLUSTER_SHARE = 0.05
HALT_SINGLE_CLUSTER_EXPOSURE = 0.01

HALT_RULE_NOTE = (
    "Part B is halted if, and only if, the BUILDER-LOSS population - missing "
    "true edges on probe rows whose exact truth was fully reachable inside the "
    "c = 400 partition, i.e. the loss whose spatial shape is NOT mechanically "
    "forced by the partition - is BOTH significantly over-dispersed across the "
    "400 clusters (permutation p < 0.001 on the exposure-weighted chi-square, "
    "10,000 permutations of the cluster labels) AND materially concentrated: "
    "the top 20 of 400 clusters by builder-missing-edge count carry at least "
    "25% of all builder-missing edges, against roughly 5% under a uniform null "
    "and against the permutation null's own 99.9th percentile, which is "
    "reported beside it. A single cluster carrying at least 5% of all "
    "builder-missing edges while holding under 1% of the probe's reachable "
    "edge exposure halts Part B on its own. Significance WITHOUT material "
    "concentration is 'structured but diffuse': it is reported prominently and "
    "does NOT halt Part B, because a p-value on 500,000 rows detects "
    "heterogeneity far below the magnitude that would change Phase 3's design. "
    "The partition-limited population is measured and reported under the same "
    "statistics but CANNOT halt Part B: cluster-spill loss is partition-shaped "
    "by construction, so finding it concentrated would be a tautology, not a "
    "finding."
)

LOCALITY_NOTE = (
    "Both halves of this test already existed on disk before this round: "
    "R0241's per-row in-degree over all 100,000,000 rows, R0238's sealed "
    "500,000-row uniform probe with exact brute-force k15 truth, and R0238's "
    "sealed per-row c = 400 partition reachability vector. The only new "
    "computation is the join, plus one re-realisation of the registered "
    "c = 400, s = 8, seed 226 partition to recover the per-row cluster label "
    "that no artifact seals. Nothing is rebuilt and no graph byte is touched."
)

PARTITION_REALISATION_NOTE = (
    "The cluster labels are a re-realisation of the registered procedure - "
    "R0226's _kmeans and _assign imported unmodified at seed 226, c = 400, "
    "s = 8, exactly as R0238's sealed reachability probe called them - not the "
    "sealed labels, because no round has ever sealed a per-row cluster label. "
    "Lloyd's algorithm here accumulates on the device through cupyx.scatter_add, "
    "whose summation order is not deterministic, so this realisation is not "
    "guaranteed bit-identical to R0238's. It is therefore VALIDATED rather than "
    "assumed: the per-row partition reachability is recomputed from these "
    "labels and compared row by row against R0238's sealed strict-c400.f64.npy, "
    "and the agreement is published. review-0241-01/F3 already established that "
    "the build's partition, R0238's reachability partition and the imbalance "
    "grid's partition are three different realisations of the same procedure, "
    "so this is the honest framing rather than a defect."
)

CANONICALIZATION_NOTE = (
    "R0034's canonicalization dropped, per source row: edges from excluded "
    "sources, zero-row destinations, self destinations, and repeated canonical "
    "destinations - and left 2,779,481 rows (~1.85%) of the v1 150M map with no "
    "valid edges at all, which R0215 traced to the clumps. The v2 rung has no "
    "exclusion list and no duplicate remapping - the 100,000,000 substrate rows "
    "ARE the universe - so canonicalization here is exactly three operations on "
    "the symmetrised edge list: drop any self-loop, fold each undirected pair "
    "onto its (min, max) representative so a mutual edge is counted once, and "
    "drop any edge whose fuzzy weight did not survive scipy's eliminate_zeros. "
    "The tripwire is then the SAME question R0034 answered badly: after those "
    "three operations, how many of the 100,000,000 rows carry no edge? This is "
    "the check that has never run at this rung, and it is the one that matters, "
    "because a clean tripwire on the RAW directed graph cannot fail for a "
    "reason that only appears at canonicalization."
)

SYMMETRISED_DEGREE_ONCE_NOTE = (
    "The symmetrised adjacency is A + A^T - A o A^T at set_op_mix_ratio 1.0, "
    "which is symmetric by construction, so its in-degree and its out-degree "
    "are the SAME NUMBER for every row. review-0240-01/F1 found that identity "
    "shipping as two independent gating quantities since at least R0237. This "
    "round reports symmetrised degree ONCE and states the identity rather than "
    "reporting it twice. The identity is asserted on a sample of rows rather "
    "than assumed, and the assertion is a cross-check, not a second test."
)

GATHER_TERM_NOTE = (
    "review-0240-01/F3 blocked R0240's gather-amplification table as a reusable "
    "cost model: its four-figure rates lived only in an agent scratchpad, were "
    "hand-transcribed, and did not reproduce to better than about 2x. The "
    "51-pass model predicted R0240's BUILD wall to 1.44% but has no gatherer "
    "term at all, and a random neighbour gather runs at 2.2-8.3x read "
    "amplification. This round prices the gather from its OWN measured, sealed "
    "sample with its own variance reported, separately from the pass term, and "
    "carries no inherited constant."
)

SAFETY_NOTE = (
    "This round starts NO child process, creates NO CUDA context outside "
    "R0226's registered _kmeans / _assign, and hands cuVS NOTHING - no cuVS "
    "call exists on either node path. Every array read off disk is opened with "
    "np.load(mmap_mode='r') and the node raises unless it is a non-writeable "
    "read-only np.memmap; the fuzzy stage's own bulk intermediates are written "
    "as np.lib.format.open_memmap files under the queue's scratch, never as "
    "anonymous buffers, and none of them is handed to a GPU library. Neither "
    "file this round adds to the node path imports the child-process module, "
    "constructs a child, calls a process-signalling method, or bounds anything "
    "by a wall that signals on expiry; every bound raises in-band and the "
    "interpreter unwinds normally. experiments/check_signal_safety.py is run "
    "for INFORMATION ONLY and its exit code is cited as evidence of nothing: "
    "it is unreleased, it has a known waiver hole on a locally hoisted argument "
    "vector, 2 of 18 planted evasions still evade it, and it reports findings "
    "on string constants. To stop this round write <queue_root>/logs/<node>.abort "
    "and wait."
)

SCOPE_NOTE = (
    "R0242 answers one science question - is the measured k15 neighbour loss "
    "spatially concentrated - and, only if the answer is no, produces the fuzzy "
    "symmetrised graph and runs the first post-canonicalization degree-zero "
    "tripwire at 100,000,000 rows. It builds no neighbour graph, assembles no "
    "substrate, trains no map, registers no gate and claims no dose, adoption, "
    "map-quality or registry result. A clean locality test does NOT clear the "
    "graph for an atlas; review-0241-01 blocks that claim and only R0228's "
    "within-map displacement instrument, run in Phase 3 on rows selected by "
    "MEASURED loss, can settle it."
)


class Round0242Error(RuntimeError):
    """R0242 fail-closed error. Raised in-band; never a signal."""


# --------------------------------------------------------------------------- #
# the wall guard, with review-0241-01/F5's two corrections applied
# --------------------------------------------------------------------------- #
class StageGuard0242(StageGuard):
    """R0241's `StageGuard` with its clock and its deadline field corrected.

    review-0241-01/F5 found two defects, neither of which touched a measured
    quantity but both of which made the guard's own receipt misdescribe it:

    1. `receipt()` read `elapsed_s` at SEAL time, not at stage completion, so
       R0241's degree pass was published as `80.28 s` when it took `52.72 s`.
       `stop()` stamps the wall the moment the stage ends and `receipt()`
       reports that stamp.
    2. `receipt()["deadline_reached"]` was a hardcoded `False` - a literal in a
       checks dict, which is exactly the `vacuouscheck` `literal-check` class.
       It is a measurement here.

    The base class is imported and subclassed, never edited: R0241's file is a
    reviewed artifact and this round modifies nothing in it.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.stopped_at_s: float | None = None

    def stop(self) -> float:
        """Stamp the wall at stage completion. Idempotent."""
        if self.stopped_at_s is None:
            self.stopped_at_s = self.elapsed_s
        return float(self.stopped_at_s)

    @property
    def stage_wall_s(self) -> float:
        return float(
            self.stopped_at_s if self.stopped_at_s is not None else self.elapsed_s
        )

    def receipt(self) -> dict[str, Any]:
        wall = self.stage_wall_s
        return {
            "label": self.label,
            "units_total": self.units_total,
            "units_done": self.units_done,
            "wall_s": wall,
            "wall_is_stage_completion_stamp": self.stopped_at_s is not None,
            "budget_s": self.budget_s,
            "deadline_s": self.deadline_s,
            "deadline_reached": bool(wall > self.deadline_s),
            "calibration_prediction": self.prediction,
            "poll_points_per_unit": 1,
            "note": (
                "review-0241-01/F5: the wall is stamped when the stage ends, "
                "not when the receipt is written, and deadline_reached is a "
                "measurement rather than a literal"
            ),
        }


# --------------------------------------------------------------------------- #
# Part A — the decomposition
# --------------------------------------------------------------------------- #
def loss_decomposition(
    *,
    strict: np.ndarray,
    reachability: np.ndarray,
    k: int = GRAPH_K,
) -> dict[str, Any]:
    """Split per-row loss into partition-limited and builder-inside-partition.

    `strict` is the per-row fraction of exact truth the graph recovered;
    `reachability` is R0238's per-row fraction of exact truth that the `c = 400`
    partition even permits the builder to find. A row can only lose an edge in
    two ways: the partition never offered it (`1 - reachability`), or the
    partition offered it and the builder did not return it. The two are
    different mechanisms with different fixes, so every locality statistic in
    this round is computed on each separately as well as on the total.

    Both counts are per-EDGE, not per-row: a row that loses one of fifteen
    neighbours is not the same evidence as a row that loses all fifteen, and
    the row-level split review-0241-01/F3 published cannot see the difference.
    """
    strict = np.asarray(strict, dtype=np.float64)
    reach = np.asarray(reachability, dtype=np.float64)
    if strict.shape != reach.shape or strict.ndim != 1:
        raise Round0242Error(
            f"R0242 decomposition needs matched 1-D vectors, got "
            f"{strict.shape} and {reach.shape}"
        )
    if strict.size == 0:
        raise Round0242Error("R0242 decomposition needs a non-empty probe")
    lost = np.rint((1.0 - strict) * float(k)).astype(np.int32)
    unreachable = np.rint((1.0 - reach) * float(k)).astype(np.int32)
    #: A row may recover a neighbour the reachability realisation calls
    #: unreachable - review-0241-01/F3 measured 48 such probe rows and traced
    #: them to the two partitions being different realisations. Clipping at
    #: zero is the conservative reading: it never invents builder loss.
    builder = np.maximum(lost - unreachable, 0).astype(np.int32)
    partition = (lost - builder).astype(np.int32)
    exposure_builder = np.maximum(int(k) - unreachable, 0).astype(np.int32)
    carrying = lost > 0
    reach_below_one = reach < 1.0
    return {
        "k": int(k),
        "probe_rows": int(strict.size),
        "rows_carrying_strict_loss": int(carrying.sum()),
        "rows_with_reachability_below_one": int(reach_below_one.sum()),
        "rows_both": int((carrying & reach_below_one).sum()),
        "rows_builder_loss_with_truth_fully_reachable": int(
            (carrying & ~reach_below_one).sum()
        ),
        "rows_recovering_an_unreachable_neighbour": int(
            (lost < unreachable).sum()
        ),
        "total_missing_edges": int(lost.sum(dtype=np.int64)),
        "partition_forced_missing_edges": int(partition.sum(dtype=np.int64)),
        "builder_missing_edges": int(builder.sum(dtype=np.int64)),
        "mean_strict_loss": float((1.0 - strict).mean()),
        "mean_partition_loss": float((1.0 - reach).mean()),
        "mean_builder_loss_inside_partition": float(
            (1.0 - strict).mean() - (1.0 - reach).mean()
        ),
        "correlation_strict_recall_with_reachability": float(
            np.corrcoef(strict, reach)[0, 1]
        ),
        "vectors": {
            "lost": lost,
            "builder_lost": builder,
            "partition_lost": partition,
            "unreachable": unreachable,
            "exposure_all": np.full(strict.size, int(k), dtype=np.int32),
            "exposure_builder": exposure_builder,
        },
    }


# --------------------------------------------------------------------------- #
# Part A test 1 — loss against in-degree
# --------------------------------------------------------------------------- #
def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared. Written out because Spearman needs them."""
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ordered = values[order]
    start = 0
    for index in range(1, ordered.size + 1):
        if index == ordered.size or ordered[index] != ordered[start]:
            ranks[order[start:index]] = 0.5 * (start + index - 1) + 1.0
            start = index
    return ranks


def spearman_permutation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    permutations: int = PERMUTATIONS,
    seed: int = PERMUTATION_SEED,
    label: str = "spearman",
    poll: Callable[[str], None] | None = None,
    poll_every: int = 500,
) -> dict[str, Any]:
    """Spearman rho with a permutation p-value on the ranks.

    A Spearman rho over `500,000` rows has a normal-theory p-value so small it
    stops being informative, and the loss vector is a heavily tied count, which
    is where normal theory is least trustworthy. Permuting one rank vector
    against the other is exact under the null of independence, costs a few
    seconds, and reports its own resolution.
    """
    rank_x = _rankdata(np.asarray(x))
    rank_y = _rankdata(np.asarray(y))
    if rank_x.size != rank_y.size:
        raise Round0242Error(f"{label}: vectors differ in length")
    centred_x = rank_x - rank_x.mean()
    centred_y = rank_y - rank_y.mean()
    denominator = float(np.sqrt((centred_x**2).sum() * (centred_y**2).sum()))
    if denominator <= 0.0:
        raise Round0242Error(f"{label}: a rank vector is constant")
    observed = float((centred_x * centred_y).sum() / denominator)
    rng = np.random.default_rng(int(seed))
    at_least = 0
    null_max = 0.0
    for index in range(int(permutations)):
        if poll is not None and index % int(poll_every) == 0:
            poll(f"{label}: permutation {index}")
        shuffled = rng.permutation(centred_y)
        value = abs(float((centred_x * shuffled).sum() / denominator))
        null_max = max(null_max, value)
        if value >= abs(observed):
            at_least += 1
    return {
        "label": label,
        "n": int(rank_x.size),
        "spearman_rho": observed,
        "permutations": int(permutations),
        "permutation_seed": int(seed),
        "two_sided_p_value": float((at_least + 1) / (int(permutations) + 1)),
        "smallest_reportable_p": float(1.0 / (int(permutations) + 1)),
        "largest_absolute_rho_under_null": null_max,
        "null_note": (
            "the null permutes one rank vector against the other, which is "
            "exact under independence and makes no distributional assumption "
            "about a heavily tied count"
        ),
    }


IN_DEGREE_BIN_EDGES: tuple[int, ...] = (0, 1, 5, 10, 20, 50, 100, 1_000)


def in_degree_bin_table(
    *,
    in_degree: np.ndarray,
    lost: np.ndarray,
    builder_lost: np.ndarray,
    exposure_builder: np.ndarray,
    k: int = GRAPH_K,
) -> list[dict[str, Any]]:
    """Loss rate by raw-in-degree band, with each band's share of the loss mass.

    The bands are registered in advance (`IN_DEGREE_BIN_EDGES`) and start with
    an isolated `0` bucket, because "is loss concentrated in the `4,424,010`
    rows nobody names as a neighbour" is the specific question R0241 left open.
    """
    in_degree = np.asarray(in_degree, dtype=np.int64)
    lost = np.asarray(lost, dtype=np.int64)
    builder_lost = np.asarray(builder_lost, dtype=np.int64)
    exposure_builder = np.asarray(exposure_builder, dtype=np.int64)
    total_missing = int(lost.sum())
    total_builder = int(builder_lost.sum())
    edges = list(IN_DEGREE_BIN_EDGES)
    table: list[dict[str, Any]] = []
    for index, low in enumerate(edges):
        high = edges[index + 1] if index + 1 < len(edges) else None
        mask = (in_degree >= low) if high is None else (
            (in_degree >= low) & (in_degree < high)
        )
        count = int(mask.sum())
        if count == 0:
            continue
        band_missing = int(lost[mask].sum())
        band_builder = int(builder_lost[mask].sum())
        band_exposure = int(exposure_builder[mask].sum())
        table.append({
            "in_degree_low": int(low),
            "in_degree_high": None if high is None else int(high - 1),
            "probe_rows": count,
            "share_of_probe_rows": count / float(in_degree.size),
            "rows_carrying_loss": int((lost[mask] > 0).sum()),
            "fraction_carrying_loss": float((lost[mask] > 0).mean()),
            "missing_edges": band_missing,
            "missing_edge_rate": band_missing / float(count * int(k)),
            "share_of_all_missing_edges": (
                band_missing / float(total_missing) if total_missing else None
            ),
            "builder_missing_edges": band_builder,
            "builder_missing_edge_rate": (
                band_builder / float(band_exposure) if band_exposure else None
            ),
            "share_of_all_builder_missing_edges": (
                band_builder / float(total_builder) if total_builder else None
            ),
        })
    return table


# --------------------------------------------------------------------------- #
# Part A test 2 — loss against cluster (the spatial test)
# --------------------------------------------------------------------------- #
def _cluster_totals(
    labels: np.ndarray, values: np.ndarray, *, clusters: int
) -> np.ndarray:
    return np.bincount(
        labels, weights=values.astype(np.float64), minlength=int(clusters)
    )[: int(clusters)]


def _dispersion(
    missing: np.ndarray, exposure: np.ndarray, *, top_m: int
) -> dict[str, float]:
    """Exposure-weighted chi-square, plus two concentration statistics.

    The chi-square asks whether per-cluster loss rates differ at all. It answers
    "yes" for any real heterogeneity at this sample size, which is why it is
    never the halt criterion on its own. The concentration statistics ask the
    question that would actually change Phase 3's design: does the loss MASS
    pile up in a few places?
    """
    total_missing = float(missing.sum())
    total_exposure = float(exposure.sum())
    if total_exposure <= 0.0:
        raise Round0242Error("R0242 dispersion needs positive exposure")
    rate = total_missing / total_exposure
    expected = exposure * rate
    live = expected > 0.0
    variance = expected[live] * max(1.0 - rate, np.finfo(np.float64).tiny)
    chi_square = float((((missing[live] - expected[live]) ** 2) / variance).sum())
    order = np.argsort(-missing, kind="stable")
    top = order[: int(top_m)]
    return {
        "chi_square": chi_square,
        "top_m_share_of_missing": (
            float(missing[top].sum() / total_missing) if total_missing else 0.0
        ),
        "top_m_share_of_exposure": float(exposure[top].sum() / total_exposure),
        "max_single_cluster_share_of_missing": (
            float(missing.max() / total_missing) if total_missing else 0.0
        ),
        "max_single_cluster_share_of_exposure": float(
            exposure[int(np.argmax(missing))] / total_exposure
        ),
    }


def cluster_locality_test(
    *,
    labels: np.ndarray,
    missing: np.ndarray,
    exposure: np.ndarray,
    clusters: int = CLUSTERS,
    permutations: int = PERMUTATIONS,
    seed: int = PERMUTATION_SEED,
    top_m: int = CONCENTRATION_TOP_M,
    population: str,
    poll: Callable[[str], None] | None = None,
    poll_every: int = 500,
) -> dict[str, Any]:
    """Is per-cluster loss rate independent of cluster identity?

    The null permutes the CLUSTER LABELS, not the loss values. That keeps each
    probe row's `(missing, exposure)` pair intact and breaks only the row-to-
    cluster association, so the null preserves the cluster size distribution,
    the marginal loss distribution and the exposure/loss coupling exactly. It is
    the exchangeability null the question actually asks, and it needs no
    distributional assumption at all.
    """
    labels = np.asarray(labels, dtype=np.int64)
    missing = np.asarray(missing, dtype=np.float64)
    exposure = np.asarray(exposure, dtype=np.float64)
    if not (labels.size == missing.size == exposure.size):
        raise Round0242Error(f"{population}: label/missing/exposure length mismatch")
    if labels.size == 0:
        raise Round0242Error(f"{population}: empty probe")
    if int(labels.min()) < 0 or int(labels.max()) >= int(clusters):
        raise Round0242Error(f"{population}: cluster label out of range")

    observed = _dispersion(
        _cluster_totals(labels, missing, clusters=clusters),
        _cluster_totals(labels, exposure, clusters=clusters),
        top_m=int(top_m),
    )
    rng = np.random.default_rng(int(seed))
    keys = ("chi_square", "top_m_share_of_missing", "max_single_cluster_share_of_missing")
    null: dict[str, list[float]] = {key: [] for key in keys}
    at_least = dict.fromkeys(keys, 0)
    for index in range(int(permutations)):
        if poll is not None and index % int(poll_every) == 0:
            poll(f"{population}: permutation {index}")
        shuffled = rng.permutation(labels)
        value = _dispersion(
            _cluster_totals(shuffled, missing, clusters=clusters),
            _cluster_totals(shuffled, exposure, clusters=clusters),
            top_m=int(top_m),
        )
        for key in keys:
            null[key].append(value[key])
            if value[key] >= observed[key]:
                at_least[key] += 1
    summary: dict[str, Any] = {
        "population": population,
        "clusters": int(clusters),
        "probe_rows": int(labels.size),
        "top_m": int(top_m),
        "total_missing_edges": float(missing.sum()),
        "total_exposure_edges": float(exposure.sum()),
        "overall_missing_rate": float(missing.sum() / exposure.sum()),
        "permutations": int(permutations),
        "permutation_seed": int(seed),
        "smallest_reportable_p": float(1.0 / (int(permutations) + 1)),
        "null_note": (
            "cluster labels permuted; each row's (missing, exposure) pair is "
            "carried intact, so cluster sizes, the marginal loss distribution "
            "and the loss/exposure coupling are all preserved under the null"
        ),
        "observed": observed,
    }
    for key in keys:
        draws = np.asarray(null[key], dtype=np.float64)
        summary[key] = {
            "observed": float(observed[key]),
            "null_mean": float(draws.mean()),
            "null_sd": float(draws.std(ddof=1)),
            "null_p50": float(np.percentile(draws, 50)),
            "null_p99": float(np.percentile(draws, 99)),
            "null_p99_9": float(np.percentile(draws, 99.9)),
            "null_max": float(draws.max()),
            "p_value": float((at_least[key] + 1) / (int(permutations) + 1)),
        }
    return summary


def cluster_rate_table(
    *,
    labels: np.ndarray,
    missing: np.ndarray,
    exposure: np.ndarray,
    clusters: int = CLUSTERS,
    report: int = 15,
    minimum_rows: int = 100,
) -> dict[str, Any]:
    """Per-cluster loss rates: the whole dispersion, plus both tails by name."""
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=int(clusters))[: int(clusters)].astype(
        np.int64
    )
    missing_by_c = _cluster_totals(labels, np.asarray(missing), clusters=clusters)
    exposure_by_c = _cluster_totals(labels, np.asarray(exposure), clusters=clusters)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(exposure_by_c > 0, missing_by_c / exposure_by_c, np.nan)
    live = (counts >= int(minimum_rows)) & (exposure_by_c > 0)
    ranked = np.flatnonzero(live)[np.argsort(-rate[live], kind="stable")]

    def _rows(indices: np.ndarray) -> list[dict[str, Any]]:
        return [
            {
                "cluster": int(index),
                "probe_rows": int(counts[index]),
                "missing_edges": float(missing_by_c[index]),
                "exposure_edges": float(exposure_by_c[index]),
                "missing_rate": float(rate[index]),
            }
            for index in indices
        ]

    finite = rate[live]
    overall = float(missing_by_c.sum() / exposure_by_c.sum())
    if finite.size == 0:
        return {
            "clusters_scored": 0,
            "clusters_suppressed_below_minimum_rows": int((~live).sum()),
            "minimum_rows_to_report_a_rate": int(minimum_rows),
            "overall_missing_rate": overall,
            "cluster_probe_row_min": int(counts.min()),
            "cluster_probe_row_max": int(counts.max()),
            "note": (
                "no cluster held the minimum number of probe rows, so no "
                "per-cluster rate is reported; the permutation test is "
                "unaffected because it never suppresses a cluster"
            ),
        }
    return {
        "clusters_scored": int(live.sum()),
        "clusters_suppressed_below_minimum_rows": int((~live).sum()),
        "minimum_rows_to_report_a_rate": int(minimum_rows),
        "overall_missing_rate": overall,
        "rate_mean": float(finite.mean()),
        "rate_sd": float(finite.std(ddof=1)),
        "rate_coefficient_of_variation": float(finite.std(ddof=1) / finite.mean())
        if finite.mean() > 0 else None,
        "rate_min": float(finite.min()),
        "rate_p25": float(np.percentile(finite, 25)),
        "rate_p50": float(np.percentile(finite, 50)),
        "rate_p75": float(np.percentile(finite, 75)),
        "rate_max": float(finite.max()),
        "max_over_overall": float(finite.max() / overall) if overall > 0 else None,
        "cluster_probe_row_min": int(counts.min()),
        "cluster_probe_row_max": int(counts.max()),
        "highest_rate_clusters": _rows(ranked[: int(report)]),
        "lowest_rate_clusters": _rows(ranked[-int(report):][::-1]),
    }


def locality_verdict(
    *,
    builder_test: Mapping[str, Any],
    partition_test: Mapping[str, Any],
    total_test: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the halt rule registered in the round file BEFORE Part A ran.

    Nothing here is chosen after seeing a number: `HALT_P_VALUE`,
    `HALT_TOP_M_SHARE`, `HALT_SINGLE_CLUSTER_SHARE` and
    `HALT_SINGLE_CLUSTER_EXPOSURE` are module constants in the release commit
    the round file names, and the round file states them in prose.
    """
    p_value = float(builder_test["chi_square"]["p_value"])
    top_share = float(builder_test["top_m_share_of_missing"]["observed"])
    single_share = float(
        builder_test["max_single_cluster_share_of_missing"]["observed"]
    )
    single_exposure = float(
        builder_test["observed"]["max_single_cluster_share_of_exposure"]
    )
    significant = p_value < HALT_P_VALUE
    concentrated = top_share >= HALT_TOP_M_SHARE
    hot_spot = (
        single_share >= HALT_SINGLE_CLUSTER_SHARE
        and single_exposure < HALT_SINGLE_CLUSTER_EXPOSURE
    )
    halt = bool((significant and concentrated) or hot_spot)
    if halt:
        reading = "spatially concentrated"
    elif significant:
        reading = "structured but diffuse"
    else:
        reading = "not distinguishable from uniform"
    return {
        "rule": HALT_RULE_NOTE,
        "population_that_can_halt": (
            "builder loss inside the c = 400 partition only"
        ),
        "builder_chi_square_p_value": p_value,
        "builder_chi_square_p_value_floor": HALT_P_VALUE,
        "builder_significant": bool(significant),
        "builder_top_m_share_of_missing": top_share,
        "builder_top_m_share_threshold": HALT_TOP_M_SHARE,
        "builder_top_m_share_null_p99_9": float(
            builder_test["top_m_share_of_missing"]["null_p99_9"]
        ),
        "builder_materially_concentrated": bool(concentrated),
        "builder_max_single_cluster_share_of_missing": single_share,
        "builder_max_single_cluster_share_of_exposure": single_exposure,
        "single_cluster_hot_spot": bool(hot_spot),
        "partition_limited_chi_square_p_value": float(
            partition_test["chi_square"]["p_value"]
        ),
        "partition_limited_top_m_share_of_missing": float(
            partition_test["top_m_share_of_missing"]["observed"]
        ),
        "partition_limited_cannot_halt_note": (
            "cluster-spill loss is partition-shaped by construction, so this "
            "population's concentration is a property of the design and not a "
            "finding about the builder; it is measured, published and excluded "
            "from the halt rule"
        ),
        "total_loss_chi_square_p_value": float(
            total_test["chi_square"]["p_value"]
        ),
        "total_loss_top_m_share_of_missing": float(
            total_test["top_m_share_of_missing"]["observed"]
        ),
        "reading": reading,
        "halt_part_b": halt,
        "part_b_may_run": bool(not halt),
    }


# --------------------------------------------------------------------------- #
# Part A — validating the re-realised partition against R0238's sealed vector
# --------------------------------------------------------------------------- #
def partition_agreement(
    *,
    reproduced: np.ndarray,
    sealed: np.ndarray,
    sealed_mean: float,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    """How closely this realisation of `c = 400` reproduces R0238's sealed one.

    Not asserted equal - `_kmeans` accumulates through non-deterministic device
    atomics, so bit-identity is not a property the procedure has. What IS
    asserted is that the realisation is close enough for its cluster labels to
    be the same spatial stratifier: the mean partition reachability must land
    within `tolerance` of the sealed `0.9983062666666668`, or the cluster test
    is reported on a different partition than the one whose ceiling the round
    quotes, and this round stops rather than quietly relabelling.
    """
    reproduced = np.asarray(reproduced, dtype=np.float64)
    sealed = np.asarray(sealed, dtype=np.float64)
    if reproduced.shape != sealed.shape:
        raise Round0242Error(
            f"R0242 reachability shapes differ: {reproduced.shape} vs {sealed.shape}"
        )
    difference = np.abs(reproduced - sealed)
    mean_gap = abs(float(reproduced.mean()) - float(sealed_mean))
    holds = bool(mean_gap <= float(tolerance))
    return {
        "rows": int(reproduced.size),
        "reproduced_mean": float(reproduced.mean()),
        "sealed_mean_reported_by_r0238": float(sealed_mean),
        "sealed_vector_mean_recomputed": float(sealed.mean()),
        "mean_absolute_gap": abs(
            float(reproduced.mean()) - float(sealed.mean())
        ),
        "gap_against_registered_mean": mean_gap,
        "tolerance": float(tolerance),
        "rows_identical": int((difference == 0.0).sum()),
        "fraction_identical": float((difference == 0.0).mean()),
        "max_absolute_row_difference": float(difference.max()),
        "reproduced_rows_with_zero_reachable": int((reproduced <= 0.0).sum()),
        "sealed_rows_with_zero_reachable": int((sealed <= 0.0).sum()),
        "holds": holds,
        "note": PARTITION_REALISATION_NOTE,
    }


def partition_reachability(
    *,
    assignment: np.ndarray,
    probe_rows: np.ndarray,
    truth_ids: np.ndarray,
    block: int = 100_000,
) -> np.ndarray:
    """R0238's sealed reachability probe, recomputed from a fresh assignment.

    Reproduced from the sealed `reachability-probe.py`
    (`f3bf33ec7ab907627eb597d576d92dea3ca93b6648e3e7deddd0bae46ae7159b`), which
    is a queue artifact rather than an importable module - so this is a
    reproduction, and `partition_agreement` is what proves it reproduces.
    """
    assignment = np.asarray(assignment)
    probe_rows = np.asarray(probe_rows, dtype=np.int64)
    truth_ids = np.asarray(truth_ids, dtype=np.int64)
    if truth_ids.shape[0] != probe_rows.shape[0]:
        raise Round0242Error("R0242 reachability: probe/truth row mismatch")
    k = int(truth_ids.shape[1])
    strict = np.empty(probe_rows.size, dtype=np.float64)
    for begin in range(0, probe_rows.size, int(block)):
        end = min(begin + int(block), probe_rows.size)
        mine = assignment[probe_rows[begin:end]]
        gathered = assignment[truth_ids[begin:end]]
        shared = (
            gathered[:, :, :, None] == mine[:, None, None, :]
        ).any(axis=3).any(axis=2)
        strict[begin:end] = shared.sum(axis=1) / float(k)
        del mine, gathered, shared
    return strict


def reproduce_r0241_recall(
    *, measured: Mapping[str, Any], sealed: Mapping[str, Any]
) -> dict[str, Any]:
    """Does this round's own scoring reproduce R0241's sealed recall exactly?

    The locality test is only worth anything if the loss vector it stratifies is
    the same loss R0241 published. This is the check that says so, and it is
    fail-closed: a mismatch stops the round before any statistic is computed.
    """
    fields = {
        "strict_mean": (
            float(measured["strict"]["mean"]), float(sealed["strict"]["mean"])
        ),
        "tie_aware_mean": (
            float(measured["tie_aware"]["mean"]), float(sealed["tie_aware"]["mean"])
        ),
        "rows_carrying_any_loss": (
            int(measured["rows_carrying_any_loss"]),
            int(sealed["rows_carrying_any_loss"]),
        ),
        "missing_true_edges": (
            int(measured["missing_true_edges"]), int(sealed["missing_true_edges"])
        ),
        "tie_aware_rows_at_zero": (
            int(measured["tie_aware_rows_at_zero"]),
            int(sealed["tie_aware_rows_at_zero"]),
        ),
    }
    disagreements = [
        name for name, (mine, theirs) in fields.items() if mine != theirs
    ]
    return {
        "fields": {
            name: {"this_round": mine, "r0241_sealed": theirs, "agree": mine == theirs}
            for name, (mine, theirs) in fields.items()
        },
        "disagreements": disagreements,
        "agree": not disagreements,
        "note": (
            "the loss vector this round stratifies is bit-for-bit the loss "
            "R0241 published, recomputed here through the same reviewed "
            "strict_containment_rows / tie_aware_rows / _score_probe"
        ),
    }


def reproduce_r0241_in_degree(
    *, measured: Mapping[str, Any], sealed: Mapping[str, Any]
) -> dict[str, Any]:
    """Does this round's in-degree array reproduce R0241's sealed distribution?"""
    fields = {
        "zero_rows": (int(measured["zero_rows"]), int(sealed["zero_rows"])),
        "max": (int(measured["max"]), int(sealed["max"])),
        "mean": (float(measured["mean"]), float(sealed["mean"])),
        "rows_at_or_above_1000": (
            int(measured["rows_at_or_above_1000"]),
            int(sealed["rows_at_or_above_1000"]),
        ),
        "share_of_edges_in_top_1_percent_of_rows": (
            float(measured["share_of_edges_in_top_1_percent_of_rows"]),
            float(sealed["share_of_edges_in_top_1_percent_of_rows"]),
        ),
    }
    disagreements = [
        name for name, (mine, theirs) in fields.items() if mine != theirs
    ]
    return {
        "fields": {
            name: {"this_round": mine, "r0241_sealed": theirs, "agree": mine == theirs}
            for name, (mine, theirs) in fields.items()
        },
        "disagreements": disagreements,
        "agree": not disagreements,
        "note": (
            "no reviewed function builds a per-row in-degree ARRAY - R0241's "
            "_degree_pass returns only its aggregates - so this round "
            "accumulates its own and proves it is the same quantity by "
            "reproducing every published aggregate exactly"
        ),
    }


# --------------------------------------------------------------------------- #
# Part B — canonicalization and the tripwire that has never run
# --------------------------------------------------------------------------- #
def canonical_undirected_degrees(
    *,
    src: np.ndarray,
    dst: np.ndarray,
    weights: np.ndarray,
    rows: int,
    block: int = 1 << 26,
) -> dict[str, Any]:
    """Canonicalize the symmetrised edge list and take every row's degree.

    Three operations, in this order, exactly as `CANONICALIZATION_NOTE` states:

    1. drop any self-loop `(i, i)`;
    2. drop any edge whose weight is not strictly positive - `eliminate_zeros`
       should already have removed these, and if it has not, that is precisely
       the defect this check exists to catch;
    3. fold `(i, j)` and `(j, i)` onto the single representative `(min, max)`.

    The degree of row `i` is then the number of canonical undirected edges
    incident to `i`, which is the quantity R0034 called `valid-degrees.u8` and
    the quantity whose zeros became the v1 map's clumps.

    Duplicate canonical destinations cannot arise here - the symmetrised matrix
    is a CSR with unique `(i, j)` - but they are counted rather than assumed
    away, because "cannot arise" is what R0034 also believed.
    """
    src = np.asarray(src)
    dst = np.asarray(dst)
    weights = np.asarray(weights)
    total = int(src.size)
    if not (total == int(dst.size) == int(weights.size)):
        raise Round0242Error("R0242 canonicalization: edge arrays differ in length")
    degree = np.zeros(int(rows), dtype=np.int64)
    self_loops = 0
    non_positive = 0
    out_of_range = 0
    kept = 0
    for begin in range(0, total, int(block)):
        end = min(begin + int(block), total)
        i = np.asarray(src[begin:end], dtype=np.int64)
        j = np.asarray(dst[begin:end], dtype=np.int64)
        w = np.asarray(weights[begin:end], dtype=np.float64)
        bad = (i < 0) | (i >= int(rows)) | (j < 0) | (j >= int(rows))
        out_of_range += int(bad.sum())
        loop = (i == j) & ~bad
        self_loops += int(loop.sum())
        dead = (w <= 0.0) & ~bad & ~loop
        non_positive += int(dead.sum())
        keep = ~bad & ~loop & ~dead & (i < j)
        kept += int(keep.sum())
        if keep.any():
            degree += np.bincount(i[keep], minlength=int(rows))
            degree += np.bincount(j[keep], minlength=int(rows))
        del i, j, w, bad, loop, dead, keep
    return {
        "directed_entries_scanned": total,
        "out_of_range_entries": int(out_of_range),
        "self_loop_entries_dropped": int(self_loops),
        "non_positive_weight_entries_dropped": int(non_positive),
        "canonical_undirected_edges": int(kept),
        "degree": degree,
        "note": CANONICALIZATION_NOTE,
    }


def post_canonical_tripwire(
    *, degree: np.ndarray, rows: int, maximum_zero_rows: int = 0
) -> dict[str, Any]:
    """The R0215 degree-zero tripwire, run for the first time AFTER canonicalization.

    R0034 shipped `2,779,481` rows (`~1.85%`) with no valid canonical edges and
    R0215 showed they are where the clumps are. A clean tripwire on the raw
    directed graph - which R0241 has - cannot see this, because a raw k-NN row
    has out-degree `k` by construction. This is the check that can.
    """
    degree = np.asarray(degree, dtype=np.int64)
    if degree.size != int(rows):
        raise Round0242Error(
            f"R0242 post-canonical tripwire covers {degree.size} of {rows} rows"
        )
    zero = int((degree == 0).sum())
    return {
        "population": (
            "all 100,000,000 rows of the CANONICALIZED undirected symmetrised "
            "graph; degree is reported ONCE because the relation is undirected"
        ),
        "rows": int(rows),
        "zero_degree_rows": zero,
        "zero_degree_fraction": zero / float(rows),
        "maximum_permitted": int(maximum_zero_rows),
        "min_degree": int(degree.min()),
        "max_degree": int(degree.max()),
        "mean_degree": float(degree.mean()),
        "median_degree": float(np.median(degree)),
        "rows_below_2": int((degree < 2).sum()),
        "v1_comparison": {
            "r0034_zero_degree_rows_at_150m": 2_779_481,
            "r0034_zero_degree_fraction": 0.0185,
            "r0215_clump_canonical_degree_zero_rate": 0.0365,
            "r0215_control_canonical_degree_zero_rate": 0.0180,
        },
        "holds": bool(zero <= int(maximum_zero_rows)),
        "note": CANONICALIZATION_NOTE,
    }


def symmetrised_degree_once(
    *, src: np.ndarray, dst: np.ndarray, rows: int, sample: int = 1_000_000,
    seed: int = 242_001, block: int = 1 << 26,
) -> dict[str, Any]:
    """Symmetrised degree, reported ONCE, with the identity asserted not assumed.

    review-0240-01/F1: `A + A^T - A o A^T` is symmetric, so its in-degree and
    out-degree are the same number for every row, and reporting them as two
    gating quantities has been an algebraic identity shipping since at least
    R0237. This function reports one degree vector and CHECKS the identity on a
    seeded sample of rows as a cross-check - which is what it is - rather than
    presenting it as a second measurement.
    """
    src = np.asarray(src)
    dst = np.asarray(dst)
    total = int(src.size)
    out_degree = np.zeros(int(rows), dtype=np.int64)
    in_degree = np.zeros(int(rows), dtype=np.int64)
    for begin in range(0, total, int(block)):
        end = min(begin + int(block), total)
        out_degree += np.bincount(
            np.asarray(src[begin:end], dtype=np.int64), minlength=int(rows)
        )
        in_degree += np.bincount(
            np.asarray(dst[begin:end], dtype=np.int64), minlength=int(rows)
        )
    rng = np.random.default_rng(int(seed))
    picked = rng.choice(int(rows), size=min(int(sample), int(rows)), replace=False)
    identical = bool(np.array_equal(out_degree[picked], in_degree[picked]))
    return {
        "degree": out_degree,
        "reported": "once",
        "identity_note": SYMMETRISED_DEGREE_ONCE_NOTE,
        "identity_cross_check": {
            "rows_sampled": int(picked.size),
            "seed": int(seed),
            "in_degree_equals_out_degree_on_every_sampled_row": identical,
            "role": (
                "a cross-check that the symmetrisation produced a symmetric "
                "matrix, NOT a second independent gating direction"
            ),
        },
        "directed_edges": total,
        "zero_degree_rows": int((out_degree == 0).sum()),
        "min": int(out_degree.min()),
        "max": int(out_degree.max()),
        "mean": float(out_degree.mean()),
        "median": float(np.median(out_degree)),
    }


def weight_distribution(
    weights: np.ndarray, *, block: int = 1 << 26, quantile_sample: int = 5_000_000,
    seed: int = 242_002,
) -> dict[str, Any]:
    """Fuzzy-weight distribution: exact extremes and mass, sampled quantiles."""
    weights = np.asarray(weights)
    total = int(weights.size)
    minimum = math.inf
    maximum = -math.inf
    accumulated = 0.0
    non_finite = 0
    at_one = 0
    non_positive = 0
    for begin in range(0, total, int(block)):
        end = min(begin + int(block), total)
        piece = np.asarray(weights[begin:end], dtype=np.float64)
        non_finite += int((~np.isfinite(piece)).sum())
        finite = piece[np.isfinite(piece)]
        if finite.size:
            minimum = min(minimum, float(finite.min()))
            maximum = max(maximum, float(finite.max()))
            accumulated += float(finite.sum())
            at_one += int((finite >= 1.0).sum())
            non_positive += int((finite <= 0.0).sum())
        del piece, finite
    rng = np.random.default_rng(int(seed))
    picked = rng.choice(total, size=min(int(quantile_sample), total), replace=False)
    picked.sort()
    sampled = np.asarray(weights[picked], dtype=np.float64)
    return {
        "directed_edges": total,
        "min": None if minimum is math.inf else minimum,
        "max": None if maximum is -math.inf else maximum,
        "mean": accumulated / float(total) if total else None,
        "total_weight": accumulated,
        "non_finite_entries": int(non_finite),
        "entries_at_or_above_one": int(at_one),
        "entries_at_or_below_zero": int(non_positive),
        "quantiles_from_sample": {
            "sample_rows": int(picked.size),
            "seed": int(seed),
            **{
                str(q): float(np.percentile(sampled, q))
                for q in (1, 5, 10, 25, 50, 75, 90, 95, 99)
            },
        },
        "valid": bool(non_finite == 0 and non_positive == 0 and maximum <= 1.0),
    }


# --------------------------------------------------------------------------- #
# the I/O model — a gather term priced separately from passes
# --------------------------------------------------------------------------- #
def gather_price(
    *,
    rows_gathered: int,
    neighbours_per_row: int,
    row_bytes: int,
    wall_s: float,
    physical_read_bytes: int,
    label: str,
) -> dict[str, Any]:
    """Price a random neighbour gather, with amplification, from a MEASURED sample.

    Passes are the right unit for a streaming builder and the wrong unit for a
    gathering consumer: the 51-pass model predicted R0240's build wall to
    `1.44%` and has no term for the gatherer at all, which is what put that
    round `1.675 GPU-h` over its cap. Every number here comes from this round's
    own instrumented sample and is sealed with it.
    """
    useful = int(rows_gathered) * int(neighbours_per_row) * int(row_bytes)
    physical = int(physical_read_bytes)
    return {
        "label": label,
        "rows_gathered": int(rows_gathered),
        "neighbours_per_row": int(neighbours_per_row),
        "row_bytes": int(row_bytes),
        "useful_bytes": useful,
        "physical_read_bytes": physical,
        "read_amplification": (physical / float(useful)) if useful else None,
        "wall_s": float(wall_s),
        "physical_rate_bytes_per_s": (
            physical / float(wall_s) if wall_s > 0 else None
        ),
        "useful_rate_bytes_per_s": (
            useful / float(wall_s) if wall_s > 0 else None
        ),
        "seconds_per_million_rows_gathered": (
            float(wall_s) * 1e6 / float(rows_gathered) if rows_gathered else None
        ),
        "note": GATHER_TERM_NOTE,
    }


def io_counters() -> dict[str, int]:
    """This process's physical read bytes, from `/proc/self/io`."""
    counters: dict[str, int] = {}
    with open("/proc/self/io", encoding="utf-8") as handle:
        for line in handle:
            key, _, value = line.partition(":")
            counters[key.strip()] = int(value.strip())
    return counters


def host_anonymous_bytes() -> dict[str, int]:
    """Host ANONYMOUS memory, never RSS (R0226/R0233: RSS counts page cache)."""
    fields = {"RssAnon": 0, "VmRSS": 0, "VmHWM": 0}
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            key, _, value = line.partition(":")
            key = key.strip()
            if key in fields:
                fields[key] = int(value.strip().split()[0]) * 1024
    return {
        "anonymous_bytes": fields["RssAnon"],
        "rss_bytes": fields["VmRSS"],
        "rss_peak_bytes": fields["VmHWM"],
        "guard_note": (
            "guarded on ANONYMOUS bytes; RSS is reported beside it and never "
            "gated on, because at these rungs RSS is dominated by the "
            "memmapped page cache we deliberately want (R0226: 56.96 GB RSS "
            "against 3.43 GB anonymous)"
        ),
    }


def json_scrub(value: Any) -> Any:
    """Drop the bulk vectors before a receipt is sealed."""
    if isinstance(value, Mapping):
        return {
            key: json_scrub(item)
            for key, item in value.items()
            if not isinstance(item, np.ndarray) or item.size <= 4_096
        }
    if isinstance(value, (list, tuple)):
        return [json_scrub(item) for item in value]
    return value


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


def elapsed_since(started: float) -> float:
    return time.monotonic() - float(started)


__all__ = [
    "CANONICALIZATION_NOTE",
    "CANONICAL_CAPABILITY",
    "CLUSTERS",
    "CONCENTRATION_TOP_M",
    "DIMENSION",
    "FUZZY_CAPABILITY",
    "FUZZY_DEADLINE_S",
    "FUZZY_FILE",
    "FUZZY_SCHEMA",
    "FUZZY_STAGE_BUDGET_S",
    "GATHER_TERM_NOTE",
    "GPU_HOURS_CAP",
    "HALT_P_VALUE",
    "HALT_RULE_NOTE",
    "HALT_SINGLE_CLUSTER_EXPOSURE",
    "HALT_SINGLE_CLUSTER_SHARE",
    "HALT_TOP_M_SHARE",
    "IN_DEGREE_BIN_EDGES",
    "LOCALITY_CAPABILITY",
    "LOCALITY_DEADLINE_S",
    "LOCALITY_EARLY_FILE",
    "LOCALITY_EARLY_SCHEMA",
    "LOCALITY_FILE",
    "LOCALITY_NOTE",
    "LOCALITY_SCHEMA",
    "LOCALITY_STAGE_BUDGET_S",
    "PARTITION_REALISATION_NOTE",
    "PARTITION_SEED",
    "PERMUTATIONS",
    "PERMUTATION_SEED",
    "PROBE_ROWS",
    "REGISTERED_INHERITANCE",
    "ROUND_ID",
    "ROWS",
    "Round0242Error",
    "SAFETY_NOTE",
    "SCOPE_NOTE",
    "SYMMETRISED_DEGREE_ONCE_NOTE",
    "StageGuard0242",
    "canonical_undirected_degrees",
    "cluster_locality_test",
    "cluster_rate_table",
    "elapsed_since",
    "gather_price",
    "host_anonymous_bytes",
    "in_degree_bin_table",
    "io_counters",
    "json_scrub",
    "locality_verdict",
    "loss_decomposition",
    "partition_agreement",
    "partition_reachability",
    "post_canonical_tripwire",
    "reproduce_r0241_in_degree",
    "reproduce_r0241_recall",
    "spearman_permutation",
    "symmetrised_degree_once",
    "weight_distribution",
]
