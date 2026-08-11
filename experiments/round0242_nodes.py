"""Execute R0242 — the locality test first, the fuzzy symmetrisation only after.

Two nodes, in this order and for this reason: R0240 spent five GPU-hours on a
product step and never reached its science check, and review-0241-01 accepted
R0241's graph "for the fuzzy round; NOT for an atlas" because the round held
both halves of a locality test and never joined them.

* `locality_100000k` is that join. Every input already exists on disk, so it
  costs the GPU only one re-realisation of the registered `c = 400` partition.
  It answers two questions R0241 could have answered for free: is recall loss
  concentrated in low-in-degree rows or in hubs, and - the question the
  spatially-blind `0.5%` probe structurally cannot see - is it concentrated in
  particular clusters of the `c = 400` partition. The partition-limited and
  builder-inside-the-partition populations are analysed separately because they
  have different causes and only the second can carry a spatial signature that
  is not mechanically forced by the partition.
* `fuzzy_100000k` runs ONLY if the locality verdict permits it. It symmetrises
  the qualified graph with UMAP's own law, reports symmetrised degree ONCE, and
  then runs the check that has never run at this rung: the R0215 degree-zero
  tripwire AFTER canonicalization, which is where the v1 defect actually arose.

Every registered check is IMPORTED, never re-typed: `verify_inheritance`,
`_probe_candidate_cosines`, `_ProbeRowView`, `_score_probe`,
`strict_containment_rows`, `tie_aware_rows`, `_blocked_descending_sort`,
`_fuzzy_symmetrise_blocked`, `_check_runner_abort`, R0226's `_kmeans` and
`_assign`. What this module adds is the join, the per-row in-degree ARRAY that
no reviewed function returns, the canonicalization, and a gather term priced
from its own measured sample.

This module starts no child process, creates no CUDA context outside R0226's
registered `_kmeans`/`_assign`, hands cuVS nothing, and contains no signalling
construct of any kind.
"""
from __future__ import annotations

import gc
import os
import shutil
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap import round0113_prompt_contrast as prompt_contract
from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_save_new_npy,
    atomic_save_new_npz,
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0220_cuvs_qualification import (
    strict_containment_rows,
    tie_aware_rows,
)
from basemap.round0238_rung5 import (
    GRAPH_K,
    SPILL,
    TRUTH_PROBE_ROWS,
    TRUTH_PROBE_SEED,
    json_safe,
    truth_probe_query_rows,
)
from basemap.round0240_rung5 import REGISTERED_REACHABILITY_CEILING_C400
from basemap.round0247_registry import registered_value
from basemap.round0241_qualify import (
    DIMENSION,
    GPU_HOURS_CAP_NOTE,
    PROBE_GATHER_BLOCK,
    REGISTERED_SELECTED_CELL,
    REGISTERED_SELECTED_CLUSTERS,
)
from basemap.round0242_locality import (
    CANONICAL_CAPABILITY,
    CLUSTERS,
    CONCENTRATION_TOP_M,
    FUZZY_CAPABILITY,
    FUZZY_DEADLINE_S,
    FUZZY_FILE,
    FUZZY_SCHEMA,
    GPU_HOURS_CAP,
    LOCALITY_CAPABILITY,
    LOCALITY_DEADLINE_S,
    LOCALITY_EARLY_FILE,
    LOCALITY_EARLY_SCHEMA,
    LOCALITY_FILE,
    LOCALITY_NOTE,
    LOCALITY_SCHEMA,  # noqa: F401  (Part B asserts Part A's schema)
    MEASUREMENT_EARLY_FILE,
    MEASUREMENT_EARLY_SCHEMA,
    PARTITION_BACKEND_NOTE,
    PARTITION_REALISATION_NOTE,
    PARTITION_SEED,
    PERMUTATIONS,
    PERMUTATION_SEED,
    REGISTERED_INHERITANCE,
    ROUND_ID,
    ROWS,
    Round0242Error,
    SAFETY_NOTE,
    SCOPE_NOTE,
    StageGuard0242,
    canonical_undirected_degrees,
    cluster_locality_test,
    cluster_rate_table,
    gather_price,
    host_anonymous_bytes,
    in_degree_bin_table,
    io_counters,
    assign_torch,
    json_scrub,
    kmeans_torch,
    locality_verdict,
    loss_decomposition,
    partition_agreement,
    partition_reachability,
    post_canonical_tripwire,
    reproduce_r0241_in_degree,
    reproduce_r0241_recall,
    spearman_permutation,
    symmetrised_degree_once,
    weight_distribution,
)
from experiments.round0238_nodes import (
    FUZZY_STRIPE_ROWS,
    _blocked_descending_sort,
    _check_runner_abort,
    _fuzzy_symmetrise_blocked,
    _score_probe,
)
from experiments.round0241_nodes import (
    _ProbeRowView,
    _probe_candidate_cosines,
    _readonly_memmap,
    verify_inheritance,
)

LOCALITY_ACTION = "locality_100000k"
FUZZY_ACTION = "fuzzy_100000k"

DEGREE_BLOCK = 5_000_000
EDGE_BLOCK = 1 << 26
#: The adversarial builder-agreement sample: uniform rows plus the two
#: populations where a builder accumulator error would most plausibly hide.
AGREEMENT_SEED = 242_100
AGREEMENT_UNIFORM_ROWS = 50_000
AGREEMENT_STRATUM_ROWS = 25_000

#: Conjunctive host-memory watchdog. A swap-only rule would have destroyed
#: R0240's completed 100M graph (its own receipt records
#: `swap_only_rule_would_have_fired: true` at `+5.095 GiB` of swap while
#: `MemAvailable` never fell below `79.24 GiB`). So swap growth alone never
#: fires: it must coincide with a genuinely large anonymous footprint or with
#: real host pressure.
WATCHDOG_SWAP_GROWTH_BYTES = 4 * (1 << 30)
WATCHDOG_ANON_BYTES = 60 * (1 << 30)
WATCHDOG_MEM_AVAILABLE_BYTES = 16 * (1 << 30)


def _memmap_attestation(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """MEASURE, at receipt time, that every bulk input is still a read-only memmap.

    `vacuouscheck`'s `literal-check` class exists because a published `True` in
    a checks dict is a verification no computation performed - R0234 shipped
    four and R0241 shipped this one. So the property is re-measured on the live
    objects here rather than asserted, and the per-array evidence is published
    beside the verdict.
    """
    measured = {
        name: {
            "is_np_memmap": bool(isinstance(array, np.memmap)),
            "writeable": bool(array.flags.writeable),
            "bytes": int(array.nbytes),
            "shape": [int(size) for size in array.shape],
        }
        for name, array in arrays.items()
    }
    return {
        "arrays": measured,
        "arrays_checked": len(measured),
        "all_read_only_memmaps": bool(measured) and all(
            entry["is_np_memmap"] and not entry["writeable"]
            for entry in measured.values()
        ),
        "note": (
            "re-measured on the live array objects at seal time, not asserted. "
            "cuVS is handed nothing on either node path, so the R0232 "
            "precondition is satisfied vacuously and is checked anyway: the "
            "rule earned there is that the assertion is what makes the "
            "property checkable"
        ),
    }


def _meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            key, _, rest = line.partition(":")
            parts = rest.split()
            if parts:
                values[key.strip()] = int(parts[0]) * 1024
    return values


def _swap_used(values: Mapping[str, int]) -> int:
    return int(values.get("SwapTotal", 0)) - int(values.get("SwapFree", 0))


class _HostWatchdog:
    """Conjunctive host-memory guard, polled in band between units of work.

    It raises a `Round0242Error` and lets the interpreter unwind. It never
    signals anything, never spawns anything, and it charges ANONYMOUS bytes
    rather than RSS, because at these rungs RSS is dominated by the memmapped
    page cache the design deliberately wants.
    """

    def __init__(self) -> None:
        baseline = _meminfo()
        self.baseline_swap_bytes = _swap_used(baseline)
        self.baseline_mem_available_bytes = int(baseline.get("MemAvailable", 0))
        self.peak_anonymous_bytes = 0
        self.peak_rss_bytes = 0
        self.peak_swap_growth_bytes = 0
        self.minimum_mem_available_bytes = int(baseline.get("MemAvailable", 0))
        self.polls = 0
        self.fired = False

    def poll(self, where: str) -> None:
        self.polls += 1
        host = host_anonymous_bytes()
        values = _meminfo()
        swap_growth = _swap_used(values) - self.baseline_swap_bytes
        available = int(values.get("MemAvailable", 0))
        self.peak_anonymous_bytes = max(
            self.peak_anonymous_bytes, int(host["anonymous_bytes"])
        )
        self.peak_rss_bytes = max(self.peak_rss_bytes, int(host["rss_bytes"]))
        self.peak_swap_growth_bytes = max(self.peak_swap_growth_bytes, swap_growth)
        self.minimum_mem_available_bytes = min(
            self.minimum_mem_available_bytes, available
        )
        #: R0249, review-0248-01 §D.3. This was the SEVENTH bare registered
        #: symbol and the one the R0248 inventory could not see, because it sat
        #: one file outside a hand-written `GUARDED_MODULES` list. It is
        #: byte-for-byte the rule R0248 fixed at `round0244_guard:404`, against
        #: a second local literal copy (`WATCHDOG_ANON_BYTES = 60 * (1 << 30)`)
        #: of the registered `max_declared_anonymous_budget_bytes`. The
        #: comparison now resolves the registry at the moment of the
        #: comparison; the module constant stays as a receipt mirror only.
        registered_anon_bytes = registered_value(
            "max_declared_anonymous_budget_bytes"
        )
        pressure = (
            int(host["anonymous_bytes"]) > registered_anon_bytes
            or available < WATCHDOG_MEM_AVAILABLE_BYTES
        )
        if swap_growth > WATCHDOG_SWAP_GROWTH_BYTES and pressure:
            self.fired = True
            raise Round0242Error(
                f"R0242 conjunctive host watchdog at {where}: swap grew "
                f"{swap_growth / (1 << 30):.2f} GiB AND anonymous is "
                f"{host['anonymous_bytes'] / (1 << 30):.2f} GiB with "
                f"{available / (1 << 30):.2f} GiB MemAvailable. Unwinding "
                "in-band; no signal was delivered and no child process exists."
            )

    def receipt(self) -> dict[str, Any]:
        return {
            "rule": (
                "conjunctive only: swap growth above 4 GiB AND (anonymous "
                "above 60 GiB OR MemAvailable below 16 GiB). A swap-only rule "
                "would have destroyed R0240's completed 100M graph"
            ),
            "polls": int(self.polls),
            "fired": bool(self.fired),
            "baseline_swap_bytes": int(self.baseline_swap_bytes),
            "peak_swap_growth_bytes": int(self.peak_swap_growth_bytes),
            "peak_host_anonymous_bytes": int(self.peak_anonymous_bytes),
            "peak_host_rss_bytes": int(self.peak_rss_bytes),
            "minimum_mem_available_bytes": int(self.minimum_mem_available_bytes),
            "swap_growth_threshold_bytes": WATCHDOG_SWAP_GROWTH_BYTES,
            #: R0249: the number the `if` actually applied, read from the
            #: registry at the comparison site, beside the module mirror it
            #: used to be compared against.
            "anonymous_threshold_bytes": WATCHDOG_ANON_BYTES,
            "registered_anonymous_threshold_bytes_at_the_comparison_site": (
                registered_value("max_declared_anonymous_budget_bytes")
            ),
            "mem_available_threshold_bytes": WATCHDOG_MEM_AVAILABLE_BYTES,
            "guard_axis": "anonymous, never RSS",
        }


# --------------------------------------------------------------------------- #
# per-row in-degree, as an ARRAY (no reviewed function returns one)
# --------------------------------------------------------------------------- #
def _in_degree_array(
    ids: np.ndarray, *, rows: int, guard: StageGuard0242
) -> tuple[np.ndarray, dict[str, Any]]:
    """Raw in-degree per row, plus every aggregate R0241 sealed.

    R0241's `_degree_pass` computes this distribution and returns only its
    summary, so the join review-0241-01/F8.2 asked for needs the array itself.
    Every aggregate is recomputed here and checked field by field against
    R0241's sealed tripwire artifact, which is what proves this is the same
    quantity and not a second, differently-defined one.
    """
    in_degree = np.zeros(int(rows), dtype=np.int32)
    scanned = 0
    for begin in range(0, int(rows), DEGREE_BLOCK):
        end = min(begin + DEGREE_BLOCK, int(rows))
        stripe = np.asarray(ids[begin:end]).astype(np.int64)
        in_range = (stripe >= 0) & (stripe < int(rows))
        in_degree += np.bincount(
            stripe[in_range], minlength=int(rows)
        ).astype(np.int32)
        scanned += end - begin
        del stripe, in_range
        guard.unit_done(f"in-degree rows [{begin}, {end})")
    if scanned != int(rows):
        raise Round0242Error(
            f"R0242 in-degree pass covered {scanned} of {rows} rows"
        )
    ordered = np.sort(in_degree)
    total = int(ordered.sum(dtype=np.int64))
    top_percent = max(1, int(rows) // 100)
    summary = {
        "zero_rows": int((in_degree == 0).sum()),
        "min": int(in_degree.min()),
        "max": int(in_degree.max()),
        "mean": float(in_degree.mean()),
        "hubness_percentiles": {
            str(q): float(ordered[min(int(rows) - 1, int(q / 100.0 * int(rows)))])
            for q in (50, 75, 90, 99, 99.9, 99.99)
        },
        "rows_at_or_above_1000": int((in_degree >= 1_000).sum()),
        "share_of_edges_in_top_1_percent_of_rows": (
            float(ordered[-top_percent:].sum(dtype=np.int64)) / float(total)
            if total else None
        ),
    }
    del ordered
    return in_degree, summary


def _builder_agreement_strata(
    *,
    ids: np.ndarray,
    host: np.ndarray,
    builder_cos: np.ndarray,
    in_degree: np.ndarray,
    guard: StageGuard0242,
) -> dict[str, Any]:
    """Builder cosines against an independent CPU recompute, ADVERSARIALLY drawn.

    R0241 measured this agreement on the registered uniform probe and found
    `4.17e-07`. A uniform draw is the weakest possible test of an accumulator
    error, because an error that lives in a structured minority is exactly what
    a uniform draw dilutes. These three strata are drawn instead: uniform, the
    `4,424,010` rows nobody names as a neighbour, and the top `1%` by in-degree
    - the two populations where a builder error would most plausibly hide.

    This matters here because Part B takes the fuzzy weights from the builder's
    own sealed cosines rather than re-gathering all `100,000,000` rows, so the
    agreement is doing real work rather than being a free curiosity.
    """
    rng = np.random.default_rng(AGREEMENT_SEED)
    zero_rows = np.flatnonzero(in_degree == 0)
    threshold = int(np.percentile(in_degree, 99.0))
    hub_rows = np.flatnonzero(in_degree >= max(threshold, 1))
    strata = {
        "uniform": np.sort(rng.choice(
            int(ROWS), size=AGREEMENT_UNIFORM_ROWS, replace=False
        )).astype(np.int64),
        "zero_in_degree": np.sort(rng.choice(
            zero_rows, size=min(AGREEMENT_STRATUM_ROWS, zero_rows.size),
            replace=False,
        )).astype(np.int64),
        "top_one_percent_in_degree": np.sort(rng.choice(
            hub_rows, size=min(AGREEMENT_STRATUM_ROWS, hub_rows.size),
            replace=False,
        )).astype(np.int64),
    }
    results: dict[str, Any] = {
        "seed": AGREEMENT_SEED,
        "hub_in_degree_threshold": int(max(threshold, 1)),
        "why": (
            "Part B takes the fuzzy weights from the builder's sealed cosines "
            "rather than re-gathering the 153.6 GB substrate for all "
            "100,000,000 rows, a step R0240 measured at ~5 GPU-h and "
            "review-0240-01 priced at 4-9 h +/- 2x. That substitution is only "
            "sound if the two arithmetics agree off the uniform probe as well "
            "as on it, so this samples where an error would hide"
        ),
        "strata": {},
    }
    worst = 0.0
    for name, rows_at in strata.items():
        _ids, recomputed = _probe_candidate_cosines(
            probe_rows=rows_at, ids=ids, host=host, guard=guard,
        )
        emitted = np.asarray(builder_cos[rows_at], dtype=np.float64)
        difference = np.abs(emitted - recomputed.astype(np.float64))
        worst = max(worst, float(difference.max()))
        results["strata"][name] = {
            "rows": int(rows_at.size),
            "values_compared": int(rows_at.size) * GRAPH_K,
            "max_absolute_difference": float(difference.max()),
            "mean_absolute_difference": float(difference.mean()),
            "p99_9_absolute_difference": float(np.percentile(difference, 99.9)),
        }
        del emitted, difference, recomputed, _ids
    results["max_absolute_difference_over_all_strata"] = worst
    results["fp32_epsilon"] = float(np.finfo(np.float32).eps)
    return results


# --------------------------------------------------------------------------- #
# node A — the locality test
# --------------------------------------------------------------------------- #
def _cluster_assignment(
    *, substrate: np.ndarray, clusters: int, seed: int, spill: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """The registered `c = 400, s = 8, seed 226` partition, re-realised in torch.

    Attempt 1 of this round died here: R0226's `_kmeans`/`_assign` are `cupy`
    functions, `cupy` lives only in the cuml-env, and reaching it needs the
    child process R0238 used and this round's safety contract forbids. The
    algorithm is transcribed instead - identical registered constants, identical
    seeded `numpy` draws, identical Lloyd step - and the realisation is
    VALIDATED against R0238's sealed reachability vector rather than assumed.
    See `PARTITION_BACKEND_NOTE`.

    This is the only stage that touches the card. cuVS is not involved and
    nothing is mapped: the device sees explicit copies of contiguous host
    blocks, never a managed-memory handle to an anonymous buffer, which is the
    construct that wedged this box at R0232.
    """
    import torch

    from basemap.round0226_graph_builders import (
        A_ASSIGN_BLOCK,
        A_KMEANS_ITERATIONS,
        A_KMEANS_SUBSAMPLE_ROWS,
    )

    if not torch.cuda.is_available():
        raise Round0242Error(
            "R0242 partition re-realisation needs the card; torch reports no "
            "CUDA device"
        )
    device = torch.device("cuda")
    started = time.perf_counter()
    centroids = kmeans_torch(
        torch, substrate, clusters=int(clusters), seed=int(seed),
        subsample_rows=A_KMEANS_SUBSAMPLE_ROWS,
        iterations=A_KMEANS_ITERATIONS, device=device,
    )
    kmeans_s = time.perf_counter() - started
    started = time.perf_counter()
    assignment = assign_torch(
        torch, substrate, centroids, rows=int(substrate.shape[0]),
        spill=int(spill), block=A_ASSIGN_BLOCK, device=device,
        poll=_check_runner_abort,
    )
    assign_s = time.perf_counter() - started
    device_peak = int(torch.cuda.max_memory_allocated(device))
    del centroids
    torch.cuda.empty_cache()
    sizes = np.bincount(
        assignment.ravel(), minlength=int(clusters)
    ).astype(np.int64)
    primary = np.bincount(
        assignment[:, 0].astype(np.int64), minlength=int(clusters)
    ).astype(np.int64)
    return assignment, {
        "clusters": int(clusters),
        "spill": int(spill),
        "seed": int(seed),
        "backend": "torch-cuda",
        "backend_note": PARTITION_BACKEND_NOTE,
        "registered_constants": {
            "kmeans_subsample_rows": int(A_KMEANS_SUBSAMPLE_ROWS),
            "kmeans_iterations": int(A_KMEANS_ITERATIONS),
            "assign_block_rows": int(A_ASSIGN_BLOCK),
        },
        "device_peak_allocated_bytes": device_peak,
        "kmeans_seconds": float(kmeans_s),
        "assign_seconds": float(assign_s),
        "spill_cluster_sizes": {
            "min": int(sizes.min()), "max": int(sizes.max()),
            "mean": float(sizes.mean()), "empty": int((sizes == 0).sum()),
        },
        "imbalance_max_over_mean": float(sizes.max() / sizes.mean()),
        "primary_cluster_sizes": {
            "min": int(primary.min()), "max": int(primary.max()),
            "mean": float(primary.mean()), "empty": int((primary == 0).sum()),
            "imbalance_max_over_mean": float(primary.max() / primary.mean()),
        },
        "stratifier_note": (
            "the spatial stratifier is the row's PRIMARY cluster - its nearest "
            "centroid, assignment[:, 0] - which is a partition of all "
            "100,000,000 rows into 400 cells. The spill assignment is what the "
            "builder searches and what reachability is defined against; the "
            "primary cluster is what a row's LOCATION is"
        ),
        "note": PARTITION_REALISATION_NOTE,
    }


def run_locality(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    """Part A — the two free tests, run before any product step."""
    manifest = active["manifest"]
    if str(manifest.get("round_id")) != ROUND_ID:
        raise Round0242Error("R0242 handler received another queue")
    started = time.monotonic()
    watchdog = _HostWatchdog()
    inheritance = verify_inheritance(job)
    output = create_fresh_directory(str(job["outputs"][0]), label="R0242 locality")

    r0241_tripwire = prompt_contract.read_sealed(
        _bound_path(job, "r0241_tripwire", label="R0242 R0241 tripwire"),
        label="R0242 R0241 tripwire",
    )
    r0241_recall = prompt_contract.read_sealed(
        _bound_path(job, "r0241_qualification", label="R0242 R0241 qualification"),
        label="R0242 R0241 qualification",
    )

    truth = prompt_contract.read_sealed(
        str(inheritance["truth"]["source"]["canonical_path"]), label="R0242 truth"
    )
    probe_rows = np.load(
        prompt_contract.verify_signature(
            dict(truth["outputs"]["query_rows"]), label="R0242 probe rows"
        ),
        allow_pickle=False,
    )
    truth_ids = np.load(
        prompt_contract.verify_signature(
            dict(truth["outputs"]["ids"]), label="R0242 truth ids"
        ),
        allow_pickle=False,
    )
    truth_cos = np.load(
        prompt_contract.verify_signature(
            dict(truth["outputs"]["cosines"]), label="R0242 truth cosines"
        ),
        allow_pickle=False,
    )
    if not np.array_equal(probe_rows, truth_probe_query_rows(
        rows=ROWS, size=TRUTH_PROBE_ROWS, seed=TRUTH_PROBE_SEED
    )):
        raise Round0242Error(
            "R0242 sealed probe rows are not R0238's registered uniform draw at "
            f"seed {TRUTH_PROBE_SEED}"
        )
    kth = truth_cos[:, GRAPH_K - 1].astype(np.float64)
    truth_best = truth_cos[:, 0].astype(np.float64)
    del truth_cos

    reachability = prompt_contract.read_sealed(
        str(inheritance["reachability"]["source"]["canonical_path"]),
        label="R0242 reachability",
    )
    cell = next(iter(reachability["cells"]))
    sealed_reach = np.load(
        prompt_contract.verify_signature(
            dict(cell["strict_vector"]), label="R0242 sealed reachability vector"
        ),
        allow_pickle=False,
    )

    substrate_path = prompt_contract.verify_signature(
        dict(prompt_contract.read_sealed(
            str(inheritance["substrate"]["source"]["canonical_path"]),
            label="R0242 substrate manifest",
        )["substrate"]),
        label="R0242 substrate",
    )
    host = _readonly_memmap(
        substrate_path, label="R0242 substrate", shape=(ROWS, DIMENSION)
    )
    ids = _readonly_memmap(
        str(inheritance["graph"]["ids"]["canonical_path"]),
        label="R0242 graph ids", shape=(ROWS, GRAPH_K),
    )
    builder_cos = _readonly_memmap(
        str(inheritance["graph"]["cosines"]["canonical_path"]),
        label="R0242 builder cosines", shape=(ROWS, GRAPH_K),
    )

    # ---- stage 1: the probe gather, priced ---------------------------------- #
    budget = float(job.get("stage_budget_s") or LOCALITY_DEADLINE_S)
    gather_guard = StageGuard0242(
        label="probe gather", units_total=-(-int(probe_rows.size) // PROBE_GATHER_BLOCK),
        budget_s=budget, deadline_s=LOCALITY_DEADLINE_S,
        abort_check=_check_runner_abort,
    )
    io_before = io_counters()
    probe_ids, probe_cos = _probe_candidate_cosines(
        probe_rows=probe_rows, ids=ids, host=host, guard=gather_guard,
    )
    gather_wall = gather_guard.stop()
    io_after = io_counters()
    watchdog.poll("after the probe gather")
    gather_cost = gather_price(
        rows_gathered=int(probe_rows.size), neighbours_per_row=GRAPH_K,
        row_bytes=DIMENSION * 4, wall_s=gather_wall,
        physical_read_bytes=int(io_after["read_bytes"] - io_before["read_bytes"]),
        label="500,000-anchor k15 neighbour gather off the 153.6 GB substrate",
    )

    # ---- stage 2: recall, reproduced exactly from R0241 ---------------------- #
    measured_recall = _score_probe(
        ids=_ProbeRowView(rows=probe_rows, values=probe_ids),
        candidate_cos=_ProbeRowView(rows=probe_rows, values=probe_cos),
        probe_rows=probe_rows, truth_ids=truth_ids, kth=kth, truth_best=truth_best,
    )
    recall_reproduction = reproduce_r0241_recall(
        measured=measured_recall, sealed=r0241_recall["recall"]
    )
    if not recall_reproduction["agree"]:
        raise Round0242Error(
            "R0242 STOP: this round's recall does not reproduce R0241's sealed "
            f"values on {recall_reproduction['disagreements']}. The loss vector "
            "the locality test would stratify is not the loss R0241 published."
        )
    strict = strict_containment_rows(probe_ids, truth_ids)
    tie = tie_aware_rows(probe_cos.astype(np.float64), probe_ids, kth)
    _check_runner_abort("R0242 scored the probe")

    # ---- stage 3: the in-degree array --------------------------------------- #
    degree_guard = StageGuard0242(
        label="in-degree pass", units_total=-(-ROWS // DEGREE_BLOCK),
        budget_s=budget, deadline_s=LOCALITY_DEADLINE_S,
        abort_check=_check_runner_abort,
    )
    in_degree, in_degree_summary = _in_degree_array(
        ids, rows=ROWS, guard=degree_guard
    )
    degree_guard.stop()
    watchdog.poll("after the in-degree pass")
    in_degree_reproduction = reproduce_r0241_in_degree(
        measured=in_degree_summary,
        sealed=r0241_tripwire["own_degree_pass"]["in_degree_raw"],
    )
    if not in_degree_reproduction["agree"]:
        raise Round0242Error(
            "R0242 STOP: the in-degree array does not reproduce R0241's sealed "
            f"distribution on {in_degree_reproduction['disagreements']}"
        )

    # ---- stage 4: the adversarial builder-cosine agreement ------------------- #
    agreement_guard = StageGuard0242(
        label="adversarial agreement gather",
        units_total=(
            -(-AGREEMENT_UNIFORM_ROWS // PROBE_GATHER_BLOCK)
            + 2 * -(-AGREEMENT_STRATUM_ROWS // PROBE_GATHER_BLOCK)
        ),
        budget_s=budget, deadline_s=LOCALITY_DEADLINE_S,
        abort_check=_check_runner_abort,
    )
    agreement = _builder_agreement_strata(
        ids=ids, host=host, builder_cos=builder_cos, in_degree=in_degree,
        guard=agreement_guard,
    )
    agreement_guard.stop()
    watchdog.poll("after the adversarial agreement gather")

    # ---- EARLY WRITE 1: everything measured so far, before the card is used -- #
    # Attempt 1 of this round reached exactly this point - recall reproduced,
    # in-degree reproduced, agreement measured - and then lost all `565.89 s` of
    # it to a `ModuleNotFoundError` in the next stage. That is review-0240-01/F2
    # happening again inside the round written to avoid it. Nothing measured is
    # held in memory across a stage boundary any more.
    measurement_path = os.path.join(output, MEASUREMENT_EARLY_FILE)
    atomic_write_new_json(measurement_path, prompt_contract.seal(json_safe(json_scrub({
        "schema": MEASUREMENT_EARLY_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": str(manifest["release_sha"]),
        "rows": ROWS,
        "k": GRAPH_K,
        "cell": REGISTERED_SELECTED_CELL,
        "recall": measured_recall,
        "recall_reproduction_of_r0241": recall_reproduction,
        "in_degree_distribution": in_degree_summary,
        "in_degree_reproduction_of_r0241": in_degree_reproduction,
        "builder_cosine_agreement_adversarial": agreement,
        "gather_cost": gather_cost,
        "wall_guards": {
            "probe_gather": gather_guard.receipt(),
            "in_degree_pass": degree_guard.receipt(),
            "adversarial_agreement": agreement_guard.receipt(),
        },
        "is_complete": False,
        "completed_by": LOCALITY_FILE,
        "why_this_file_exists": (
            "every stage that produces a result persists it before the next "
            "stage can fail; attempt 1 lost 565.89 s of exactly these "
            "measurements to a failure three stages later"
        ),
    }))), immutable=True)

    # ---- stage 5: the partition, re-realised and validated ------------------- #
    assignment, partition = _cluster_assignment(
        substrate=host, clusters=CLUSTERS, seed=PARTITION_SEED, spill=SPILL
    )
    watchdog.poll("after the cluster assignment")
    reproduced_reach = partition_reachability(
        assignment=assignment, probe_rows=probe_rows, truth_ids=truth_ids
    )
    agreement_partition = partition_agreement(
        reproduced=reproduced_reach, sealed=sealed_reach,
        sealed_mean=REGISTERED_REACHABILITY_CEILING_C400,
    )
    cluster_of_probe = assignment[
        np.asarray(probe_rows, dtype=np.int64), 0
    ].astype(np.int64)
    primary_cluster = assignment[:, 0].astype(np.int16)
    del assignment
    gc.collect()
    watchdog.poll("after the reachability reproduction")
    if not agreement_partition["holds"]:
        raise Round0242Error(
            "R0242 STOP: this realisation of the c = 400 partition reproduces "
            f"a mean reachability of {agreement_partition['reproduced_mean']} "
            f"against R0238's registered "
            f"{REGISTERED_REACHABILITY_CEILING_C400}. The cluster labels are "
            "not the stratifier this round registered and the locality test "
            "would be reported against the wrong partition."
        )

    # ---- stage 6: the decomposition and the two tests ------------------------ #
    decomposition = loss_decomposition(
        strict=strict, reachability=sealed_reach, k=GRAPH_K
    )
    vectors = decomposition.pop("vectors")
    probe_in_degree = in_degree[np.asarray(probe_rows, dtype=np.int64)].astype(
        np.int64
    )

    in_degree_tests = {
        "note": (
            "R0241 computed a per-row in-degree over all 100,000,000 rows in "
            "node A and a per-row loss over 500,000 rows in node B, in the "
            "same queue, and never joined them. This is the join"
        ),
        "bins": in_degree_bin_table(
            in_degree=probe_in_degree, lost=vectors["lost"],
            builder_lost=vectors["builder_lost"],
            exposure_builder=vectors["exposure_builder"], k=GRAPH_K,
        ),
        "spearman_total_loss": spearman_permutation(
            probe_in_degree, vectors["lost"],
            label="raw in-degree vs total missing edges",
            poll=_check_runner_abort,
        ),
        "spearman_builder_loss": spearman_permutation(
            probe_in_degree, vectors["builder_lost"],
            label="raw in-degree vs builder missing edges",
            poll=_check_runner_abort,
        ),
        "spearman_partition_loss": spearman_permutation(
            probe_in_degree, vectors["partition_lost"],
            label="raw in-degree vs partition-forced missing edges",
            poll=_check_runner_abort,
        ),
        "probe_in_degree_summary": {
            "zero_rows": int((probe_in_degree == 0).sum()),
            "mean": float(probe_in_degree.mean()),
            "max": int(probe_in_degree.max()),
            "mean_in_degree_of_loss_carrying_rows": float(
                probe_in_degree[vectors["lost"] > 0].mean()
            ) if int((vectors["lost"] > 0).sum()) else None,
            "mean_in_degree_of_clean_rows": float(
                probe_in_degree[vectors["lost"] == 0].mean()
            ),
        },
    }

    populations = {
        "total_loss": (vectors["lost"], vectors["exposure_all"]),
        "partition_limited_loss": (
            vectors["partition_lost"], vectors["exposure_all"]
        ),
        "builder_loss_inside_partition": (
            vectors["builder_lost"], vectors["exposure_builder"]
        ),
    }
    cluster_tests: dict[str, Any] = {}
    cluster_tables: dict[str, Any] = {}
    for name, (missing, exposure) in populations.items():
        cluster_tests[name] = cluster_locality_test(
            labels=cluster_of_probe, missing=missing, exposure=exposure,
            clusters=CLUSTERS, permutations=PERMUTATIONS, seed=PERMUTATION_SEED,
            top_m=CONCENTRATION_TOP_M, population=name, poll=_check_runner_abort,
        )
        cluster_tables[name] = cluster_rate_table(
            labels=cluster_of_probe, missing=missing, exposure=exposure,
            clusters=CLUSTERS,
        )
        _check_runner_abort(f"R0242 cluster locality test: {name}")

    verdict = locality_verdict(
        builder_test=cluster_tests["builder_loss_inside_partition"],
        partition_test=cluster_tests["partition_limited_loss"],
        total_test=cluster_tests["total_loss"],
    )

    # ---- PERSIST THE SCIENCE THE MOMENT IT EXISTS ---------------------------- #
    # review-0240-01/F2: R0240 computed recall and lost it because nothing was
    # written until after the last long stage.
    early_path = os.path.join(output, LOCALITY_EARLY_FILE)
    atomic_write_new_json(early_path, prompt_contract.seal(json_safe(json_scrub({
        "schema": LOCALITY_EARLY_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": str(manifest["release_sha"]),
        "rows": ROWS,
        "k": GRAPH_K,
        "clusters": CLUSTERS,
        "cell": REGISTERED_SELECTED_CELL,
        "decomposition": decomposition,
        "in_degree_tests": in_degree_tests,
        "cluster_locality_tests": cluster_tests,
        "locality_verdict": verdict,
        "is_complete": False,
        "completed_by": LOCALITY_FILE,
        "why_this_file_exists": (
            "the two science results of this round are written the instant "
            "they exist, before any artifact save or receipt can fail"
        ),
    }))), immutable=True)

    vector_dir = ensure_data_directory(os.path.join(output, "vectors"))
    saved = {
        "probe_cluster_labels": atomic_save_new_npy(
            os.path.join(vector_dir, "probe-cluster-c400.i16.npy"),
            cluster_of_probe.astype(np.int16), immutable=True,
        ),
        "probe_strict_recall": atomic_save_new_npy(
            os.path.join(vector_dir, "probe-strict-recall.f64.npy"),
            strict, immutable=True,
        ),
        "probe_tie_aware_recall": atomic_save_new_npy(
            os.path.join(vector_dir, "probe-tie-aware-recall.f64.npy"),
            tie, immutable=True,
        ),
        "probe_missing_edges": atomic_save_new_npy(
            os.path.join(vector_dir, "probe-missing-edges.i16.npy"),
            vectors["lost"].astype(np.int16), immutable=True,
        ),
        "probe_builder_missing_edges": atomic_save_new_npy(
            os.path.join(vector_dir, "probe-builder-missing-edges.i16.npy"),
            vectors["builder_lost"].astype(np.int16), immutable=True,
        ),
        "probe_in_degree": atomic_save_new_npy(
            os.path.join(vector_dir, "probe-in-degree.i32.npy"),
            probe_in_degree.astype(np.int32), immutable=True,
        ),
        "primary_cluster_all_rows": atomic_save_new_npy(
            os.path.join(vector_dir, "primary-cluster-c400.i16.npy"),
            primary_cluster, immutable=True,
        ),
    }

    receipt = prompt_contract.seal(json_safe(json_scrub({
        "schema": LOCALITY_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": str(manifest["release_sha"]),
        "capability": LOCALITY_CAPABILITY,
        "rows": ROWS,
        "k": GRAPH_K,
        "spill": SPILL,
        "clusters": CLUSTERS,
        "cell": REGISTERED_SELECTED_CELL,
        "registered_inheritance": REGISTERED_INHERITANCE,
        "inheritance": inheritance,
        "locality_note": LOCALITY_NOTE,
        "recall_reproduction_of_r0241": recall_reproduction,
        "in_degree_reproduction_of_r0241": in_degree_reproduction,
        "recall": measured_recall,
        "in_degree_distribution": in_degree_summary,
        "partition": partition,
        "partition_agreement_with_r0238": agreement_partition,
        "builder_cosine_agreement_adversarial": agreement,
        "decomposition": decomposition,
        "in_degree_tests": in_degree_tests,
        "cluster_locality_tests": cluster_tests,
        "cluster_rate_tables": cluster_tables,
        "locality_verdict": verdict,
        "gather_cost": gather_cost,
        "wall_guards": {
            "probe_gather": gather_guard.receipt(),
            "in_degree_pass": degree_guard.receipt(),
            "adversarial_agreement": agreement_guard.receipt(),
        },
        "host_watchdog": watchdog.receipt(),
        "vectors_saved": {
            name: expected_input_signature(path) for name, path in saved.items()
        },
        "measurement_first_write": expected_input_signature(measurement_path),
        "locality_first_write": expected_input_signature(early_path),
        "bulk_input_memmap_attestation": _memmap_attestation({
            "substrate": host, "graph_ids": ids, "builder_cosines": builder_cos,
        }),
        "cuvs_calls": 0,
        "cuda_context_created": True,
        "cuda_context_note": (
            "one CUDA context, created by R0226's registered _kmeans/_assign "
            "for the c = 400 re-realisation and by nothing else. cuVS is not "
            "called; the device sees explicit cupy copies of contiguous host "
            "blocks, never a managed-memory handle to an anonymous buffer"
        ),
        "child_processes_launched": 0,
        "signal_delivered": False,
        "abort_policy": (
            "in-band cooperative flag only; no SIGTERM, no SIGKILL, no ptrace, "
            "and no child process exists to signal"
        ),
        "safety_note": SAFETY_NOTE,
        "scope_note": SCOPE_NOTE,
        "gpu_hours_cap": GPU_HOURS_CAP,
        "gpu_hours_cap_note": GPU_HOURS_CAP_NOTE,
        "training_performed": False,
        "gate_registered": False,
        "adoption_claimed": False,
        "map_quality_claimed": False,
        "performance": {"total_wall_s": time.monotonic() - started},
    })))
    atomic_write_new_json(
        os.path.join(output, LOCALITY_FILE), receipt, immutable=True
    )


def _bound_path(job: Mapping[str, Any], key: str, *, label: str) -> str:
    reference = dict(job[key])
    path = str(reference.get("canonical_path") or "")
    if not path or not reference.get("sha256"):
        raise Round0242Error(f"{label} must be bound at a full sha256 signature")
    if expected_input_signature(path) != reference:
        raise Round0242Error(f"{label} content changed")
    return path


# --------------------------------------------------------------------------- #
# node B — the fuzzy symmetrisation and the post-canonicalization tripwire
# --------------------------------------------------------------------------- #
def run_fuzzy(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    """Part B — symmetrise, then run the tripwire where the v1 defect arose."""
    manifest = active["manifest"]
    if str(manifest.get("round_id")) != ROUND_ID:
        raise Round0242Error("R0242 handler received another queue")
    started = time.monotonic()
    watchdog = _HostWatchdog()
    inheritance = verify_inheritance(job)

    # The Part A receipt is an INTRA-queue artifact, so it cannot be bound at a
    # hash the manifest was written before it existed. It is bound instead by
    # its own seal - `read_sealed` recomputes `identity_sha256` over the
    # content and refuses a tampered file - plus its registered schema and
    # release commit. That is the strongest binding available to a node that
    # consumes its own queue's output.
    locality_path = str(job["locality_reference"])
    if not os.path.isfile(locality_path):
        raise Round0242Error(
            f"R0242 Part B cannot run: Part A's receipt is absent at "
            f"{locality_path}"
        )
    locality = prompt_contract.read_sealed(
        locality_path, label="R0242 locality receipt"
    )
    if str(locality.get("schema")) != LOCALITY_SCHEMA or str(
        locality.get("round_id")
    ) != ROUND_ID:
        raise Round0242Error("R0242 Part B read a receipt from another round")
    if str(locality.get("release_sha")) != str(manifest["release_sha"]):
        raise Round0242Error(
            "R0242 Part B and Part A did not run from the same release commit"
        )
    verdict = dict(locality["locality_verdict"])
    if not verdict.get("part_b_may_run"):
        raise Round0242Error(
            "R0242 STOP before Part B: the registered locality rule halted it. "
            f"Builder-loss chi-square p = {verdict['builder_chi_square_p_value']}, "
            f"top-{CONCENTRATION_TOP_M} share "
            f"{verdict['builder_top_m_share_of_missing']} against a threshold "
            f"of {verdict['builder_top_m_share_threshold']}. Loss is "
            "SPATIALLY CONCENTRATED and the atlas plan needs R0228's "
            "displacement instrument before any ladder map is trained. This is "
            "the registered behaviour, not a failure."
        )

    output = create_fresh_directory(str(job["outputs"][0]), label="R0242 fuzzy")
    ids = _readonly_memmap(
        str(inheritance["graph"]["ids"]["canonical_path"]),
        label="R0242 graph ids", shape=(ROWS, GRAPH_K),
    )
    cosines = _readonly_memmap(
        str(inheritance["graph"]["cosines"]["canonical_path"]),
        label="R0242 builder cosines", shape=(ROWS, GRAPH_K),
    )

    budget = float(job.get("stage_budget_s") or FUZZY_DEADLINE_S)
    sort_guard = StageGuard0242(
        label="descending sort", units_total=1, budget_s=budget,
        deadline_s=FUZZY_DEADLINE_S, abort_check=_check_runner_abort,
    )
    io_before = io_counters()
    ids_sorted, cos_sorted = _blocked_descending_sort(
        ids=ids, cosines=cosines, rows=ROWS
    )
    sort_guard.unit_done("descending sort")
    sort_guard.stop()
    watchdog.poll("after the descending sort")

    dists = np.maximum((1.0 - cos_sorted).astype(np.float32), 0.0)
    if not np.isfinite(dists).all():
        raise Round0242Error("R0242 candidate distances are not finite")
    del cos_sorted
    gc.collect()
    watchdog.poll("after the distance transform")

    import umap.umap_ as umap_api

    ids_out = atomic_save_new_npy(
        os.path.join(output, "graph-k15-ids.i32.npy"), ids_sorted, immutable=True
    )
    fuzzy_guard = StageGuard0242(
        label="fuzzy symmetrisation", units_total=-(-ROWS // FUZZY_STRIPE_ROWS),
        budget_s=budget, deadline_s=FUZZY_DEADLINE_S,
        abort_check=_check_runner_abort,
    )
    fuzzy_started = time.monotonic()
    fuzzy = _fuzzy_symmetrise_blocked(
        knn_indices=ids_sorted, knn_dists=dists, rows=ROWS, k=GRAPH_K,
        umap_api=umap_api, out_dir=output,
    )
    fuzzy_guard.units_done = fuzzy_guard.units_total
    fuzzy_seconds = time.monotonic() - fuzzy_started
    fuzzy_guard.stop()
    del dists, ids_sorted
    gc.collect()
    watchdog.poll("after the fuzzy symmetrisation")

    src = fuzzy["src"]
    dst = fuzzy["dst"]
    wts = fuzzy["weights"]
    directed_edges = int(fuzzy["directed_edges"])

    weights = weight_distribution(wts, block=EDGE_BLOCK)
    if not weights["valid"]:
        raise Round0242Error(
            "R0242 fuzzy weights are invalid: "
            f"min {weights['min']}, max {weights['max']}, "
            f"{weights['non_finite_entries']} non-finite"
        )
    watchdog.poll("after the weight scan")

    degrees = symmetrised_degree_once(
        src=src, dst=dst, rows=ROWS, block=EDGE_BLOCK
    )
    symmetrised_degree = degrees.pop("degree")
    if not degrees["identity_cross_check"][
        "in_degree_equals_out_degree_on_every_sampled_row"
    ]:
        raise Round0242Error(
            "R0242 symmetrisation produced an asymmetric adjacency: in-degree "
            "and out-degree differ on a sampled row, which the set operation "
            "A + A^T - A o A^T makes impossible"
        )
    del symmetrised_degree
    watchdog.poll("after the symmetrised degree pass")

    canonical = canonical_undirected_degrees(
        src=src, dst=dst, weights=wts, rows=ROWS, block=EDGE_BLOCK
    )
    canonical_degree = canonical.pop("degree")
    tripwire = post_canonical_tripwire(degree=canonical_degree, rows=ROWS)
    del canonical_degree
    gc.collect()
    watchdog.poll("after canonicalization")
    io_after = io_counters()

    # The edge list is published as three streamed `.npy` arrays plus a small
    # `.npz` header, NOT as one `edges-*.npz` carrying the bulk. `zipfile`
    # cannot stream a member: `atomic_save_new_npz` writes each array into an
    # `io.BytesIO` and then calls `getvalue()`, which at this rung materialises
    # ~20 GB of ANONYMOUS memory per 10 GB member for no benefit, and produces
    # an archive no consumer can memmap. `atomic_save_new_npy` streams from the
    # scratch memmap in chunks and the result is memmappable, which is what a
    # 100M trainer needs. Registered in the round file in advance.
    edge_paths = {
        "sources": atomic_save_new_npy(
            os.path.join(output, "edges-k15-fuzzy-src.i32.npy"), src,
            immutable=True,
        ),
        "targets": atomic_save_new_npy(
            os.path.join(output, "edges-k15-fuzzy-dst.i32.npy"), dst,
            immutable=True,
        ),
        "weights": atomic_save_new_npy(
            os.path.join(output, "edges-k15-fuzzy-wts.f32.npy"), wts,
            immutable=True,
        ),
    }
    header_path = atomic_save_new_npz(
        os.path.join(output, "edges-k15-fuzzy-header.npz"), immutable=True,
        compressed=False,
        n_nodes=np.asarray(ROWS, dtype=np.int64),
        k=np.asarray(GRAPH_K, dtype=np.int64),
        directed_edges=np.asarray(directed_edges, dtype=np.int64),
    )
    fuzzy_record = {
        key: value for key, value in fuzzy.items()
        if key not in ("src", "dst", "weights", "scratch")
    }
    del src, dst, wts
    gc.collect()
    shutil.rmtree(fuzzy["scratch"], ignore_errors=True)

    scratch_bytes = 3 * 2 * ROWS * GRAPH_K * 4
    receipt = prompt_contract.seal(json_safe(json_scrub({
        "schema": FUZZY_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": str(manifest["release_sha"]),
        "capabilities": [FUZZY_CAPABILITY, CANONICAL_CAPABILITY],
        "rows": ROWS,
        "k": GRAPH_K,
        "spill": SPILL,
        "clusters": REGISTERED_SELECTED_CLUSTERS,
        "cell": REGISTERED_SELECTED_CELL,
        "registered_inheritance": REGISTERED_INHERITANCE,
        "inheritance": inheritance,
        "locality_verdict_that_permitted_this_node": verdict,
        "fuzzy": fuzzy_record,
        "fuzzy_seconds": float(fuzzy_seconds),
        "directed_edges": directed_edges,
        "weight_distribution": weights,
        "symmetrised_degree": degrees,
        "canonicalization": canonical,
        "post_canonical_tripwire": tripwire,
        "outputs": {
            "ids": expected_input_signature(ids_out),
            "edges_header": expected_input_signature(header_path),
            **{
                f"edges_{name}": expected_input_signature(path)
                for name, path in edge_paths.items()
            },
        },
        "wall_guards": {
            "descending_sort": sort_guard.receipt(),
            "fuzzy_symmetrisation": fuzzy_guard.receipt(),
        },
        "host_watchdog": watchdog.receipt(),
        "io": {
            "physical_read_bytes": int(
                io_after["read_bytes"] - io_before["read_bytes"]
            ),
            "physical_write_bytes": int(
                io_after["write_bytes"] - io_before["write_bytes"]
            ),
            "peak_scratch_bytes_architectural": int(scratch_bytes),
            "pass_term_note": (
                "the symmetrisation is a PASS-shaped consumer - two sequential "
                "walks of a 6 GB id array and one of a 6 GB cosine array, then "
                "one stripe-major walk of the membership arrays - and the "
                "51-pass model's unit is right for it. The GATHER term is "
                "priced separately, from the locality node's own measured "
                "sample, because a random neighbour gather is a different "
                "instrument and the 51-pass model has no term for it"
            ),
        },
        "bulk_input_memmap_attestation": _memmap_attestation({
            "graph_ids": ids, "builder_cosines": cosines,
        }),
        "cuvs_calls": 0,
        "cuda_context_created": False,
        "child_processes_launched": 0,
        "signal_delivered": False,
        "abort_policy": (
            "in-band cooperative flag only, polled every fuzzy stripe and every "
            "sort block; no SIGTERM, no SIGKILL, no ptrace, and no child "
            "process exists to signal"
        ),
        "safety_note": SAFETY_NOTE,
        "scope_note": SCOPE_NOTE,
        "gpu_hours_cap": GPU_HOURS_CAP,
        "gpu_hours_cap_note": GPU_HOURS_CAP_NOTE,
        "training_performed": False,
        "gate_registered": False,
        "adoption_claimed": False,
        "map_quality_claimed": False,
        "performance": {"total_wall_s": time.monotonic() - started},
    })))
    atomic_write_new_json(
        os.path.join(output, FUZZY_FILE), receipt, immutable=True
    )
    if not tripwire["holds"]:
        raise Round0242Error(
            "R0242 POST-CANONICALIZATION DEGREE-ZERO TRIPWIRE FAILED at "
            f"100,000,000 rows: {tripwire['zero_degree_rows']} rows carry no "
            "canonical edge. This is the v1 failure mode (R0034: 2,779,481 "
            "rows, ~1.85%) at the exact step where it arose, and the evidence "
            "is sealed beside this failure. The graph must not be used."
        )


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    action = str(job.get("action") or "")
    if action == LOCALITY_ACTION:
        run_locality(active, job)
    elif action == FUZZY_ACTION:
        run_fuzzy(active, job)
    else:
        raise Round0242Error(f"R{ROUND_ID} does not authorize action {action!r}")


__all__ = [
    "FUZZY_ACTION",
    "LOCALITY_ACTION",
    "run_fuzzy",
    "run_job",
    "run_locality",
]
