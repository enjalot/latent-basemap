"""Frozen contract for R0235 — Phase 2 rung 2 (12.5M), nested on R0233's rung.

Two things happen and nothing else: assemble a 12,500,000-row mixed universe that
**contains** R0233's 6,250,000 training rows, build and qualify its k15 graph, and
measure cluster imbalance across a wider `c` sweep so the 25M and 50M rungs can be
re-derived from measurement rather than from extrapolation. No map is trained.

Every registered rule below is a reaction to a measured defect in R0233 or in its
independent review (`review-0233-2026-08-09-01.md`):

* **D1 — the guard margin is applied to the rung derivation too.** R0233's
  `guard_decision` multiplies measured imbalance by `IMBALANCE_GUARD_MARGIN`
  before a cell may launch, but its `rung_derivation` took the point estimate
  straight. Applying the round's own margin turns its 25M and 50M cells
  infeasible. Here **one** function computes the guarded max-cluster figure and
  both the launch guard and the rung derivation call it.
* **D2 — imbalance is measured over a wider `c` sweep, at this `N`.** "Imbalance
  at fixed `c` is stable in `N`" held at `c = 200` and failed at `c = 64`
  (`+3.90%`, 2M -> 6.25M); `c = 32` had no `s = 8` measurement at any other `N`.
  This round probes `c` in `(16, 32, 64, 128, 200, 400)` at `s = 8`, which closes
  both gaps and adds the two candidates a 100M contingency would need.
* **D3 — no device point is ever transcribed from prose.** The device law is
  fitted on points read out of **sealed artifacts** and filtered on the exact
  `(graph_degree, intermediate_graph_degree, max_iterations)` triple, so
  homogeneity is *enforced* rather than asserted. R0233 inherited a
  `igd 128 / it 20` cell into a law it called homogeneous at `igd 256 / it 40`,
  reconstructed from a rounded GiB figure in a review.
* **D5 — only fields that exist are cited.** Every safety claim this round makes
  names a field its own artifacts emit.
* **D7 — the rung's artifacts do not live inside a failed queue's tree.** R0233's
  released substrate sits under `round-0233/queue/` — the queue that terminated
  `failed`. R0235's substrate carries R0233's 6,250,000 rows byte-identically in
  its own prefix, in its own queue tree, which is a durable copy of the parent
  rung as a side effect of nesting.

## Nesting — the Phase 2 design constraint

The ladder is `6.25M -> 12.5M -> 25M -> 50M -> 100M` with **one variable per
rung**, so rung 2 must *contain* rung 1. The selection law is therefore:

    T2 = T1  U  uniform_without_replacement(P \\ T1 \\ R1, n2 - n1)

where `P` is the corpus's non-degenerate row pool, `T1` is R0233's training
selection and `R1` is R0233's held-out reserve. This is not an approximation.
`(T1, R1)` was drawn as a uniform ordered pair of disjoint sets, so conditional
on `R1`, `T1` is uniform over size-`n1` subsets of `P \\ R1`; adding a uniform
draw from the complement makes `T2` **exactly** a uniform size-`n2` subset of
`P \\ R1`. The only departure from "uniform over `P`" is the 200,000 reserve rows
that are held out on purpose.

Nesting is realised **positionally**: rung-1 row `i` is rung-2 row `i` for every
`i < 6,250,000`, verified by hashing the prefix and comparing against R0233's
sealed `ordered_substrate_sha256`. That makes rung 1's truth, graph and any
future map index-aligned with rung 2 for free.

## The reserve

R0233's reserve is inherited **verbatim** — the same 200,000 rows, 50,000 per
training corpus, R0108's `49,500 + 500` split — and copied into this round's
artifact tree so rung 2's capability is self-contained. Two consequences are
registered rather than discovered:

1. Rung-1 reserve rows are excluded from rung 2's draw pool, so the reserve stays
   disjoint from **rung 2's** training rows as well as rung 1's.
2. Holding the eval reserve fixed across rungs is what makes a rung-to-rung
   comparison a comparison of `N` alone. A reserve that grew with `N` would move
   two variables at once.

## Safety preconditions, carried unchanged from R0233

Both GPU wedges in this program were NVIDIA UVM page-fault deadlocks on driver
`570.211.01`. R0233's build path is inherited *as executable code*, not as a
copy: `basemap/round0235_build.py` calls `basemap/round0233_build.py` with one
named constant rebound (the cluster capacity, which is the very guard D1
concerns). So the memmap precondition, the signal-free cooperative abort and the
OOM-as-measurement discipline are the reviewed ones, byte for byte.

1. Every buffer handed to cuVS is a read-only, C-contiguous `np.memmap`,
   including every intermediate spill file, asserted immediately before
   `nn_descent.build` receives it.
2. No signal is ever delivered to a build process. The abort path is a flag file
   the child polls; the parent never calls `terminate()`, `kill()` or `os.kill`.
3. A wedged GPU is never probed.
4. A predictive guard runs before every cell on device, host **anonymous** bytes
   and disk, with `refused_a_priori` recorded as data.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
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
    GUARD_HOST_ANON_BUDGET_BYTES,
    GUARD_SAFETY_BYTES,
    GUARD_SWAP_GROWTH_ABORT_BYTES,
    SAMPLE_INTERVAL_S,
    SCRATCH_BUDGET_BYTES,
    WATCHDOG_POLL_S,
    linear_fit,
    pack_clusters_into_groups,
)
from basemap.round0233_substrate import (
    C_MIN as R0233_C_MIN,
    CLUSTER_CAPACITY_ROWS as R0233_CLUSTER_CAPACITY_ROWS,
    DATA_READ_CONTIGUOUS_BYTES_PER_S,
    DATA_READ_FRAGMENTED_BYTES_PER_S,
    DATA_WRITE_BYTES_PER_S,
    DENSITY_DECILES,
    DETERMINISM_NOTE,
    DISK_FREE_FLOOR_BYTES,
    FUZZY_RANDOM_STATE_SEED,
    GRAPH_DEGREE,
    IMBALANCE_GUARD_MARGIN,
    INTERMEDIATE_GRAPH_DEGREE,
    IO_NOTE,
    MAX_ITERATIONS,
    MAX_REPLACEMENT_ROUNDS,
    MAX_ZERO_DEGREE_ROWS,
    NN_DESCENT_SETTING,
    RECALL_MEAN_FLOOR,
    RECALL_P10_FLOOR,
    RESERVE_CORPUS_ROWS,
    RESERVE_QUERY_ROWS,
    RESERVE_ROWS,
    RESERVE_ROWS_PER_CORPUS,
    RESERVE_SEED,
    SHARD_COVERAGE_FLOOR,
    SPILL,
    TIE_TOLERANCE,
    assert_memmap_for_cuvs,
    assert_no_signal_policy,
    io_projection,
    mean_cluster_rows,
    reserve_split,
    substrate_pass_count,
)


ROUND_ID = "0235"

SUBSTRATE_CAPABILITY = "minilm-mixed-12500k-nested-substrate-and-reserves-v1"
TRUTH_CAPABILITY = "minilm-mixed-12500k-exact-k15-truth-v1"
LADDER_CAPABILITY = "minilm-mixed-12500k-cluster-spill-build-ladder-v1"
GRAPH_CAPABILITY = "minilm-mixed-12500k-cluster-spill-k15-fuzzy-graph-v1"
IMBALANCE_CAPABILITY = "minilm-mixed-cluster-spill-s8-imbalance-drift-v1"

SUBSTRATE_SCHEMA = "round0235-minilm-mixed-12500k-nested-substrate-v1"
TRUTH_SCHEMA = "round0235-minilm-mixed-12500k-exact-k15-truth-v1"
LADDER_SCHEMA = "round0235-minilm-mixed-12500k-build-ladder-v1"
GRAPH_SCHEMA = "round0235-minilm-mixed-12500k-k15-fuzzy-graph-v1"
LAW_SCHEMA = "round0235-guarded-device-law-and-rung-rederivation-v1"

#: Rung 2 of `../guides/plan-minilm-100m-v2.md` Phase 2.
ROWS = 12_500_000
#: Rung 1. Every one of these rows is contained in `ROWS`, at the same position.
PARENT_ROWS = 6_250_000
PARENT_ROUND_ID = "0233"

#: The owner-confirmed 40/25/25/10 shares at this rung's exact row counts.
COMPOSITION: tuple[tuple[str, int], ...] = (
    ("fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2", 5_000_000),
    ("RedPajama-Data-V2-sample-10B-chunked-120-all-MiniLM-L6-v2", 3_125_000),
    ("pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2", 3_125_000),
    ("starcoderdata-code-chunked-120-all-MiniLM-L6-v2", 1_250_000),
)
#: R0233's composition, which this round's prefix must reproduce exactly.
PARENT_COMPOSITION: tuple[tuple[str, int], ...] = (
    ("fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2", 2_500_000),
    ("RedPajama-Data-V2-sample-10B-chunked-120-all-MiniLM-L6-v2", 1_562_500),
    ("pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2", 1_562_500),
    ("starcoderdata-code-chunked-120-all-MiniLM-L6-v2", 625_000),
)
TARGET_SHARES = {name: n / ROWS for name, n in COMPOSITION}
INCREMENT_BY_CORPUS = {
    name: n - dict(PARENT_COMPOSITION)[name] for name, n in COMPOSITION
}

#: The increment's own seed. The prefix is not re-drawn: it is R0233's bytes.
SELECTION_SEED = 235
SELECTION_LAW = (
    "T2 = T1 U uniform_without_replacement(P \\ T1 \\ R1, n2 - n1), per corpus, "
    "at seed 235 + corpus index, with rejected rows replaced by fresh uniform "
    "draws from the unpicked complement until the increment quota is met. T1 is "
    "R0233's sealed training selection and R1 its sealed reserve. Because "
    "(T1, R1) was drawn as a uniform ordered pair of disjoint sets, T2 is "
    "EXACTLY a uniform size-12,500,000 subset of P \\ R1. Never a prefix of the "
    "corpus; always a positional prefix of the substrate."
)
NESTING_NOTE = (
    "rung-1 row i is rung-2 row i for every i < 6,250,000. Verified by hashing "
    "the substrate prefix with ordered_array_sha256 and comparing against "
    "R0233's sealed ordered_substrate_sha256, and independently by set "
    "containment on packed (corpus, shard, row) provenance keys."
)
RESERVE_NOTE = (
    "R0233's reserve inherited verbatim - the same 200,000 rows, 50,000 per "
    "training corpus, R0108's 49,500 + 500 split - copied into this round's "
    "artifact tree and verified byte-identical by sha256. Rung-1 reserve rows "
    "are excluded from rung 2's draw pool, so the reserve is disjoint from rung "
    "2's training rows as well as rung 1's. Holding the eval reserve fixed "
    "across rungs is what makes a rung-to-rung comparison a comparison of N "
    "alone."
)

# --------------------------------------------------------------------------- #
# the builder — R0229's adopted arm, unchanged since R0233
# --------------------------------------------------------------------------- #
CANDIDATE = "cluster-spill-nnd"
C_MIN = R0233_C_MIN  # = 2 * SPILL = 16

#: The `c` values whose imbalance is MEASURED at this N. Wider than the selection
#: ladder on purpose: `c = 32` had no `s = 8` measurement at any N but 6.25M,
#: `c = 64` was the one value observed drifting the wrong way, and `c = 128 / 400`
#: are the candidates a 25M/50M/100M contingency needs priced.
IMBALANCE_PROBE_CLUSTERS: tuple[int, ...] = (16, 32, 64, 128, 200, 400)
#: The `c` values this round's own graph may be built at, inherited from R0233.
SELECTION_CANDIDATES: tuple[int, ...] = (16, 32, 64, 200)
#: A control cell that always runs beside the selected one. At `(12.5M, c = 64)`
#: the predicted largest cluster is ~2.498M rows against R0233's `(6.25M, c = 32)`
#: cell at 2,496,850 — the same cluster size at twice the `N`. That is the only
#: clean test of whether the device law is a function of max-cluster rows alone,
#: and review-0233-01 asked for a second `N` at `gd = 64` before 50M is priced.
CONTROL_CLUSTERS: tuple[int, ...] = (64,)
LADDER_RULE = (
    "the ladder is the registered control cell c = 64 plus the c the selection "
    "law picks from measured imbalance, built in ASCENDING predicted max-cluster "
    "rows; if they coincide only one cell runs. A refusal, abort or failure "
    "stops the ladder."
)

# --------------------------------------------------------------------------- #
# the device law and the guard — D1 and D3 fixed here
# --------------------------------------------------------------------------- #
#: The setting every fitted point must share. A point that does not carry this
#: exact triple is not admissible into the law, which makes homogeneity a
#: structural property of the fit rather than a claim about it.
LAW_GRAPH_DEGREE = GRAPH_DEGREE                    # 64
LAW_INTERMEDIATE_GRAPH_DEGREE = INTERMEDIATE_GRAPH_DEGREE  # 256
LAW_MAX_ITERATIONS = MAX_ITERATIONS                # 40
LAW_HOMOGENEITY_NOTE = (
    "every point in the device law is read from a sealed artifact and admitted "
    "only if its receipt records graph_degree 64, intermediate_graph_degree 256 "
    "and max_iterations 40. R0233's six-point law contained one igd 128 / it 20 "
    "cell, mislabelled and reconstructed from a rounded GiB figure in a review "
    "(review-0233-01 D3)."
)

#: R0233's margin, carried unchanged — and now applied EVERYWHERE a max-cluster
#: figure feeds a decision, including the per-rung derivation (D1).
GUARD_IMBALANCE_MARGIN = IMBALANCE_GUARD_MARGIN
#: The device law's own worst absolute relative residual on the inherited points
#: is 4.41%; five percent covers it. This is the extrapolation charge, separate
#: from the imbalance charge, because they are different uncertainties.
LAW_RESIDUAL_MARGIN = 0.05
GUARD_NOTE = (
    "predicted device = law(imbalance * 1.1648840 * mean_cluster_rows) * 1.05 + "
    "1 GiB, taken as the MAX over the homogeneous inherited law and this round's "
    "own-cells law, against a 24 GiB budget on a 31.37 GiB card. R0233's guard "
    "used a deliberately steep two-point bound (3400 B/row) because the law was "
    "unknown; the law is now measured over a 21.44x span and verified "
    "independently, so the guard charges the law plus explicit, separately "
    "named margins instead of a shape that was wrong."
)

DEVICE_TOTAL_BYTES = 31.37 * 1024 ** 3
GUARD_DEVICE_BUDGET_BYTES = 24 * 1024 ** 3
GPU_HOURS_CAP = 3.5
#: Per-cell deadline. R0233's cells ran 634.98-643.58 s over 50,000,000 spilled
#: rows; this rung spills 100,000,000 and packs into roughly twice as many
#: groups, so a cell is expected near 1,400 s and this is a 2.9x margin. On
#: expiry the parent sets the cooperative flag and waits — it never signals.
BUILD_TIMEOUT_S = 4_000.0

# --------------------------------------------------------------------------- #
# qualification
# --------------------------------------------------------------------------- #
RECALL_POPULATION = (
    "all 12,500,000 substrate rows, uniform; no seed set, no neighbour union"
)
TRUTH_METHOD = "brute-force fp32 cosine top-k over the full substrate, all rows"

PHASE2_RUNGS: tuple[int, ...] = (
    6_250_000, 12_500_000, 25_000_000, 50_000_000, 100_000_000,
)

#: Sealed `s = 8` imbalance at 2M, from R0229's spill-reachability artifact. Read
#: from the artifact at run time; the literal here exists only so a CPU test can
#: check the reader against a known answer. `c = 32` is absent at 2M and that
#: absence is the point of D2.
R0229_2M_S8_IMBALANCE_REFERENCE: dict[int, float] = {
    16: 1.183384, 64: 1.539004, 200: 2.1311125,
}


class Round0235Error(RuntimeError):
    """The registered R0235 contract changed."""


# --------------------------------------------------------------------------- #
# composition, span, nesting
# --------------------------------------------------------------------------- #
def validate_composition(counts: Mapping[str, int]) -> dict[str, Any]:
    """Fail closed unless the assembled universe is exactly the registered mix."""
    total = sum(int(value) for value in counts.values())
    if total != ROWS:
        raise Round0235Error(f"substrate has {total} rows, registered {ROWS}")
    observed: dict[str, Any] = {}
    for name, want in COMPOSITION:
        got = int(counts.get(name, 0))
        if got != want:
            raise Round0235Error(f"{name}: assembled {got} rows, registered {want}")
        observed[name] = {
            "rows": got,
            "share": got / ROWS,
            "registered_share": TARGET_SHARES[name],
            "inherited_from_rung1": dict(PARENT_COMPOSITION)[name],
            "newly_drawn": INCREMENT_BY_CORPUS[name],
        }
    return observed


def validate_shard_span(
    *, corpus: str, shards_touched: int, shards_total: int, label: str
) -> dict[str, Any]:
    """R0216's span assertion. It RAISES; the defect is invisible otherwise.

    Applied to the union selection AND to this round's increment on its own. The
    union alone would be nearly unfalsifiable here, because rung 1 already spans
    every shard of every corpus.
    """
    if shards_total <= 0:
        raise Round0235Error(f"{corpus}: no shards")
    coverage = shards_touched / float(shards_total)
    if coverage < SHARD_COVERAGE_FLOOR:
        raise Round0235Error(
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
    """Pack `(corpus, shard, row)` into one int64 key per row.

    `shard` is bounded by 177 and `row` by 227,613,720, so 8 bits of corpus and
    16 bits of shard leave 40 bits for the row id — ample, and asserted.
    """
    corpus = np.asarray(records["corpus"], dtype=np.int64)
    shard = np.asarray(records["shard"], dtype=np.int64)
    row = np.asarray(records["row"], dtype=np.int64)
    if corpus.size and (
        int(corpus.max()) >= 256 or int(shard.max()) >= 65_536
        or int(row.min()) < 0 or int(row.max()) >= (1 << 40)
    ):
        raise Round0235Error("R0235 provenance does not fit the registered key")
    return (corpus << 56) | (shard << 40) | row


def assert_nesting(*, parent: np.ndarray, child: np.ndarray) -> dict[str, Any]:
    """Rung 2 must CONTAIN rung 1, on row ids, and every child row is distinct."""
    parent_keys = provenance_keys(parent)
    child_keys = provenance_keys(child)
    if int(np.unique(child_keys).size) != int(child_keys.size):
        raise Round0235Error("R0235 substrate holds a duplicated source row")
    missing = int(np.setdiff1d(parent_keys, child_keys, assume_unique=False).size)
    if missing != 0:
        raise Round0235Error(
            f"R0235 is not nested on R0233: {missing} rung-1 rows are absent. "
            "Phase 2's whole design is one variable per rung; a non-nested rung "
            "confounds N with the sample."
        )
    positional = bool(
        parent_keys.size <= child_keys.size
        and np.array_equal(parent_keys, child_keys[: parent_keys.size])
    )
    if not positional:
        raise Round0235Error(
            "R0235 prefix is not R0233's rows in R0233's order; the registered "
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
        raise Round0235Error(
            f"R0235 reserve overlaps the training selection on {overlap} rows"
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
            raise Round0235Error(f"R0235 reserve overlaps training in {corpus}")
    return {
        "global_intersection_rows": overlap,
        "per_corpus": per_corpus,
        "reserve_rows": int(reserve_keys.size),
        "note": RESERVE_NOTE,
    }


# --------------------------------------------------------------------------- #
# the device law, from sealed artifacts only
# --------------------------------------------------------------------------- #
def admit_law_point(point: Mapping[str, Any]) -> dict[str, Any]:
    """Admit one device measurement into the law, or refuse it with a reason.

    A point must carry the registered `(gd, igd, it)` triple and a positive
    max-cluster row count and device byte count. Anything else is refused; the
    refusal is recorded rather than silently dropped.
    """
    setting = (
        point.get("graph_degree"),
        point.get("intermediate_graph_degree"),
        point.get("max_iterations"),
    )
    registered = (
        LAW_GRAPH_DEGREE, LAW_INTERMEDIATE_GRAPH_DEGREE, LAW_MAX_ITERATIONS
    )
    rows = point.get("max_cluster_rows")
    device = point.get("device_bytes")
    reasons: list[str] = []
    if any(value is None for value in setting):
        reasons.append("receipt does not record the nn-descent setting")
    elif tuple(int(value) for value in setting) != registered:
        reasons.append(
            f"setting {tuple(int(v) for v in setting)} is not the registered "
            f"{registered}"
        )
    if not rows or int(rows) <= 0:
        reasons.append("no positive max_cluster_rows")
    if not device or float(device) <= 0.0:
        reasons.append("no positive device_wide_peak_over_baseline_bytes")
    return {
        **{key: point.get(key) for key in (
            "source", "cell", "rows", "clusters", "spill", "graph_degree",
            "intermediate_graph_degree", "max_iterations",
        )},
        "max_cluster_rows": None if not rows else int(rows),
        "device_bytes": None if not device else float(device),
        "admitted": not reasons,
        "refusal_reasons": reasons,
    }


def fit_device_law(points: Sequence[Mapping[str, Any]], *, label: str) -> dict[str, Any]:
    """Fit `device = intercept + slope * max_cluster_rows` on admitted points."""
    admitted = [admit_law_point(point) for point in points]
    usable = [entry for entry in admitted if entry["admitted"]]
    if len(usable) < 2:
        raise Round0235Error(
            f"R0235 device law '{label}' has {len(usable)} admissible points"
        )
    usable.sort(key=lambda entry: int(entry["max_cluster_rows"]))
    x = [int(entry["max_cluster_rows"]) for entry in usable]
    y = [float(entry["device_bytes"]) for entry in usable]
    fit = linear_fit(x, y)
    if float(fit["slope"]) <= 0.0:
        raise Round0235Error(
            f"R0235 device law '{label}' fitted a non-positive slope "
            f"({fit['slope']}); device cost cannot fall with cluster size and a "
            "law like that cannot bound anything"
        )
    measured = np.asarray(y, dtype=np.float64)
    residuals = np.asarray(fit["residuals"], dtype=np.float64)
    return {
        "label": str(label),
        "setting": NN_DESCENT_SETTING,
        "homogeneity_note": LAW_HOMOGENEITY_NOTE,
        "slope_bytes_per_max_cluster_row": float(fit["slope"]),
        "intercept_bytes": float(fit["intercept"]),
        "r_squared": float(fit["r_squared"]),
        "n_points": int(fit["n_points"]),
        "fitted_range_max_cluster_rows": [int(min(x)), int(max(x))],
        "fitted_span_factor": float(max(x) / max(1, min(x))),
        "residual_bytes": [float(value) for value in residuals],
        "residual_relative": [float(value) for value in (residuals / measured)],
        "worst_absolute_relative_residual": float(
            np.max(np.abs(residuals / measured))
        ),
        "points": usable,
        "points_refused": [
            entry for entry in admitted if not entry["admitted"]
        ],
    }


def law_device_bytes(law: Mapping[str, Any], max_cluster_rows: float) -> float:
    return float(law["intercept_bytes"]) + float(
        law["slope_bytes_per_max_cluster_row"]
    ) * float(max_cluster_rows)


def guarded_max_cluster_rows(
    *, rows: int, clusters: int, imbalance: float, spill: int = SPILL,
    margin: float = GUARD_IMBALANCE_MARGIN,
) -> float:
    """The max-cluster figure every decision in this round is taken on.

    R0233 had two of these — one inside `guard_decision`, with the margin, and
    one inside `rung_derivation`, without it — and the difference is what made
    its 25M and 50M cells look feasible (review-0233-01 D1). There is one here.
    """
    return mean_cluster_rows(rows=rows, clusters=clusters, spill=spill) * float(
        imbalance
    ) * float(margin)


def guard_device_bytes(
    laws: Sequence[Mapping[str, Any]],
    guarded_max_cluster: float,
    *,
    residual_margin: float = LAW_RESIDUAL_MARGIN,
) -> dict[str, Any]:
    """Charge the MAX over the supplied laws, plus the residual and safety margins."""
    per_law = [
        {
            "label": str(law["label"]),
            "law_bytes": law_device_bytes(law, guarded_max_cluster),
            "guarded_bytes": law_device_bytes(law, guarded_max_cluster)
            * (1.0 + float(residual_margin)) + float(GUARD_SAFETY_BYTES),
        }
        for law in laws
    ]
    if not per_law:
        raise Round0235Error("R0235 guard needs at least one device law")
    worst = max(per_law, key=lambda entry: entry["guarded_bytes"])
    return {
        "guarded_max_cluster_rows": float(guarded_max_cluster),
        "per_law": per_law,
        "binding_law": worst["label"],
        "predicted_device_bytes": float(worst["guarded_bytes"]),
        "predicted_device_gib": float(worst["guarded_bytes"]) / 1024 ** 3,
        "law_residual_margin": float(residual_margin),
        "safety_bytes": int(GUARD_SAFETY_BYTES),
        "note": GUARD_NOTE,
    }


def admissible_max_cluster_rows(
    laws: Sequence[Mapping[str, Any]],
    *, device_budget_bytes: int = GUARD_DEVICE_BUDGET_BYTES,
    residual_margin: float = LAW_RESIDUAL_MARGIN,
) -> float:
    """Largest guarded max-cluster the budget admits under the binding law."""
    ceiling = (
        float(device_budget_bytes) - float(GUARD_SAFETY_BYTES)
    ) / (1.0 + float(residual_margin))
    limits = [
        (ceiling - float(law["intercept_bytes"]))
        / float(law["slope_bytes_per_max_cluster_row"])
        for law in laws
    ]
    if not limits:
        raise Round0235Error("R0235 admissibility needs at least one device law")
    return float(min(limits))


# --------------------------------------------------------------------------- #
# c selection and the launch guard
# --------------------------------------------------------------------------- #
def select_clusters(
    *,
    rows: int,
    measured_imbalance: Mapping[int, float],
    laws: Sequence[Mapping[str, Any]],
    candidates: Sequence[int] = SELECTION_CANDIDATES,
    spill: int = SPILL,
    c_min: int = C_MIN,
    device_budget_bytes: int = GUARD_DEVICE_BUDGET_BYTES,
) -> dict[str, Any]:
    """The smallest admissible `c`, from imbalance MEASURED at this N.

    Fewer, larger clusters are strictly better for reachability, so the chosen
    configuration is the smallest `c` the guard admits. `c` is never interpolated
    and never modelled: a candidate with no measured imbalance is not selectable.
    """
    rows = int(rows)
    admissible = admissible_max_cluster_rows(
        laws, device_budget_bytes=device_budget_bytes
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
        point = mean_cluster_rows(
            rows=rows, clusters=clusters, spill=spill
        ) * imbalance
        guarded = guarded_max_cluster_rows(
            rows=rows, clusters=clusters, imbalance=imbalance, spill=spill
        )
        charge = guard_device_bytes(laws, guarded)
        fits = guarded <= admissible
        entry.update({
            "admissible": bool(fits),
            "measured_imbalance": imbalance,
            "imbalance_source": "measured on this substrate at this N",
            "mean_cluster_rows": mean_cluster_rows(
                rows=rows, clusters=clusters, spill=spill
            ),
            "max_cluster_rows_point_estimate": float(point),
            "guarded_max_cluster_rows": float(guarded),
            "admissible_max_cluster_rows": float(admissible),
            "guard": charge,
        })
        considered.append(entry)
        if fits and chosen is None:
            chosen = entry
    if chosen is None:
        raise Round0235Error(
            f"R0235 found no admissible c at {rows} rows, s = {spill}: {considered}"
        )
    return {
        "rows": rows,
        "spill": int(spill),
        "c_min": int(c_min),
        "selected_clusters": int(chosen["clusters"]),
        "selection": chosen,
        "candidates_considered": considered,
        "admissible_max_cluster_rows": float(admissible),
        "rule": (
            "the smallest c >= 2*spill whose GUARDED largest cluster at this N "
            "fits the device budget under the binding law; c is never "
            "interpolated and the imbalance margin is always applied"
        ),
    }


def guard_decision(
    *,
    rows: int,
    clusters: int,
    imbalance: float,
    imbalance_source: str,
    laws: Sequence[Mapping[str, Any]],
    spill: int = SPILL,
    device_budget_bytes: int = GUARD_DEVICE_BUDGET_BYTES,
    host_anon_budget_bytes: int = GUARD_HOST_ANON_BUDGET_BYTES,
    disk_free_bytes: int | None = None,
) -> dict[str, Any]:
    """Launch this cell, or refuse it and record the refusal as a measurement."""
    rows = int(rows)
    clusters = int(clusters)
    row_bytes = DIMENSION * 4
    guarded = guarded_max_cluster_rows(
        rows=rows, clusters=clusters, imbalance=imbalance, spill=spill
    )
    charge = guard_device_bytes(laws, guarded)
    device_bytes = float(charge["predicted_device_bytes"])
    capacity = admissible_max_cluster_rows(
        laws, device_budget_bytes=device_budget_bytes
    )

    topk_bytes = rows * GRAPH_K * 8
    assignment_bytes = rows * spill * 4
    member_bytes = rows * spill * 8 * 3
    stage_bytes = 500_000 * row_bytes
    local_graph_bytes = int(guarded) * GRAPH_DEGREE * 8
    local_cosine_bytes = int(guarded) * GRAPH_DEGREE * 4
    anonymous_bytes = int(
        topk_bytes
        + assignment_bytes
        + member_bytes
        + stage_bytes
        + local_graph_bytes
        + local_cosine_bytes
        + 3 * 1024 ** 3
    )

    # Peak scratch is max(bound, largest single cluster) — review-0232's
    # correction, because `pack_clusters_into_groups` flushes a group before
    # admitting a cluster that would cross the budget.
    peak_scratch_bytes = int(max(
        min(SCRATCH_BUDGET_BYTES, rows * spill * row_bytes),
        int(guarded) * row_bytes,
    ))

    device_over = device_bytes > float(device_budget_bytes)
    host_over = anonymous_bytes > int(host_anon_budget_bytes)
    capacity_over = guarded > float(capacity)
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
            f"guarded largest cluster {guarded:.0f} exceeds the admissible "
            f"{capacity:.0f} rows"
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
            "imbalance_margin": GUARD_IMBALANCE_MARGIN,
            "predicted_mean_cluster_rows": mean_cluster_rows(
                rows=rows, clusters=clusters, spill=spill
            ),
            "max_cluster_rows_point_estimate": mean_cluster_rows(
                rows=rows, clusters=clusters, spill=spill
            ) * float(imbalance),
            "guarded_max_cluster_rows": float(guarded),
            "predicted_device_bytes": device_bytes,
            "predicted_device_gib": device_bytes / 1024 ** 3,
            "predicted_host_anon_bytes": int(anonymous_bytes),
            "predicted_host_anon_gib": anonymous_bytes / 1024 ** 3,
            "predicted_peak_scratch_bytes": peak_scratch_bytes,
            "device_charge": charge,
        },
        "device_budget_bytes": int(device_budget_bytes),
        "host_anon_budget_bytes": int(host_anon_budget_bytes),
        "admissible_max_cluster_rows": float(capacity),
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
# per-rung re-derivation — D1's fix, and the answer this round owes
# --------------------------------------------------------------------------- #
def rung_derivation(
    *,
    rung: int,
    imbalance_by_c: Mapping[int, float],
    imbalance_source: str,
    laws: Sequence[Mapping[str, Any]],
    spill: int = SPILL,
    device_budget_bytes: int = GUARD_DEVICE_BUDGET_BYTES,
    apply_margin: bool = True,
) -> dict[str, Any]:
    """Re-derive `c` for one Phase-2 rung, WITH the round's own imbalance margin.

    `apply_margin=False` reproduces R0233's arithmetic so the two can be printed
    side by side; it is never the basis of a decision here.
    """
    margin = GUARD_IMBALANCE_MARGIN if apply_margin else 1.0
    residual = LAW_RESIDUAL_MARGIN if apply_margin else 0.0
    admissible = admissible_max_cluster_rows(
        laws, device_budget_bytes=device_budget_bytes, residual_margin=residual
    )
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
        point = mean_cluster_rows(
            rows=rung, clusters=clusters, spill=spill
        ) * imbalance
        guarded = guarded_max_cluster_rows(
            rows=rung, clusters=clusters, imbalance=imbalance, spill=spill,
            margin=margin,
        )
        charge = guard_device_bytes(laws, guarded, residual_margin=residual)
        entry = {
            "clusters": int(clusters),
            "imbalance": imbalance,
            "mean_cluster_rows": mean_cluster_rows(
                rows=rung, clusters=clusters, spill=spill
            ),
            "max_cluster_rows_point_estimate": float(point),
            "guarded_max_cluster_rows": float(guarded),
            "device_bytes": float(charge["predicted_device_bytes"]),
            "device_gib": float(charge["predicted_device_gib"]),
            "binding_law": charge["binding_law"],
            "admissible": bool(guarded <= admissible),
        }
        considered.append(entry)
        if entry["admissible"] and chosen is None:
            chosen = entry
    return {
        "rung": int(rung),
        "spill": int(spill),
        "imbalance_source": str(imbalance_source),
        "imbalance_margin_applied": float(margin),
        "law_residual_margin_applied": float(residual),
        "margins_note": (
            "apply_margin=False reproduces R0233's arithmetic exactly - point "
            "imbalance, no residual charge - so the two rows can be read side "
            "by side. Only the margined row is ever the basis of a decision."
        ),
        "admissible_max_cluster_rows": float(admissible),
        "device_budget_bytes": int(device_budget_bytes),
        "selected_clusters": None if chosen is None else int(chosen["clusters"]),
        "selection": chosen,
        "candidates_considered": considered,
        "feasible": chosen is not None,
        "note": (
            "imbalance here is measured at 12,500,000 rows and CARRIED to the "
            "rung, not measured at it. The drift table is the evidence for how "
            "much that carry is worth; each rung should still probe its own N "
            "before it is built."
        ),
    }


def imbalance_drift(series: Mapping[int, Mapping[int, float]]) -> dict[str, Any]:
    """Per-`c` imbalance across the measured `N`, with drift against the smallest.

    `series` maps `N` to `{c: imbalance}`. Cells absent at an `N` are reported as
    absent, never interpolated — the absence of an `s = 8, c = 32` measurement
    below 6.25M is exactly what review-0233-01 D2 identified.
    """
    normalised = {
        int(key): {int(c): float(v) for c, v in value.items()}
        for key, value in series.items()
    }
    sizes = sorted(normalised)
    by_c: dict[str, Any] = {}
    for clusters in sorted({c for value in normalised.values() for c in value}):
        row: dict[str, Any] = {"clusters": int(clusters), "by_rows": {}}
        seen: list[tuple[int, float]] = []
        for size in sizes:
            value = normalised[size].get(clusters)
            row["by_rows"][str(size)] = None if value is None else float(value)
            if value is not None:
                seen.append((size, float(value)))
        if len(seen) >= 2:
            base_n, base = seen[0]
            top_n, top = seen[-1]
            row.update({
                "measured_at_rows": [int(n) for n, _v in seen],
                "drift_relative": (top - base) / base,
                "drift_span_rows": [int(base_n), int(top_n)],
                "monotone_increasing": bool(
                    all(b >= a for (_n1, a), (_n2, b) in zip(seen, seen[1:]))
                ),
            })
        else:
            row.update({
                "measured_at_rows": [int(n) for n, _v in seen],
                "drift_relative": None,
                "insufficient_points": True,
            })
        by_c[str(clusters)] = row
    return {
        "rows_measured": sizes,
        "by_clusters": by_c,
        "note": (
            "drift_relative is (imbalance at the largest measured N) / "
            "(imbalance at the smallest measured N) - 1. A positive value moves "
            "toward the device budget."
        ),
    }


__all__ = [
    "BUILD_TIMEOUT_S",
    "CANDIDATE",
    "COMPOSITION",
    "CONTROL_CLUSTERS",
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
    "IMBALANCE_CAPABILITY",
    "IMBALANCE_PROBE_CLUSTERS",
    "INCREMENT_BY_CORPUS",
    "INTERMEDIATE_GRAPH_DEGREE",
    "IO_NOTE",
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
    "PARENT_COMPOSITION",
    "PARENT_ROUND_ID",
    "PARENT_ROWS",
    "PHASE2_RUNGS",
    "R0229_2M_S8_IMBALANCE_REFERENCE",
    "R0233_CLUSTER_CAPACITY_ROWS",
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
    "Round0235Error",
    "SAMPLE_INTERVAL_S",
    "SCRATCH_BUDGET_BYTES",
    "SELECTION_CANDIDATES",
    "SELECTION_LAW",
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
    "admissible_max_cluster_rows",
    "admit_law_point",
    "assert_memmap_for_cuvs",
    "assert_nesting",
    "assert_no_signal_policy",
    "assert_reserve_disjoint",
    "fit_device_law",
    "guard_decision",
    "guard_device_bytes",
    "guarded_max_cluster_rows",
    "imbalance_drift",
    "io_projection",
    "law_device_bytes",
    "mean_cluster_rows",
    "pack_clusters_into_groups",
    "provenance_keys",
    "reserve_split",
    "resolve_shard_rows",
    "rung_derivation",
    "select_clusters",
    "substrate_pass_count",
    "validate_composition",
    "validate_shard_span",
]
