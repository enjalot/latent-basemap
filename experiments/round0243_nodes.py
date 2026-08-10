"""Execute R0243 — the tie-aware locality settlement, then the fuzzy graph.

Two nodes:

* `residual_100000k` (Part A) re-runs R0242's own locality analysis on the
  TIE-AWARE loss vector R0242 sealed and never joined, beside the strict
  figures, so the difference is explicit rather than inferred. It costs almost
  nothing: every per-row vector already exists on disk and is bound by hash. It
  also discharges three things review-0242-01 named — it seals a reproduced
  `c = 400` reachability vector and the spill assignment it came from, it
  measures the realised exposure distribution so the re-expressed hot-spot
  guard's ability to bind is a measurement, and it prices a SORTED gather at
  two anchor counts instead of extrapolating an unsorted one.
* `fuzzy_100000k` (Part B) runs only if Part A's sealed verdict permits it, and
  re-reads that verdict through `read_sealed` rather than trusting job order.
  It symmetrises with UMAP's own law, reports symmetrised degree ONCE, and runs
  the R0215 degree-zero tripwire AFTER canonicalization — the check the v2
  rebuild exists for, which has never run at `100,000,000` rows.

Every registered check is IMPORTED, never re-typed: `verify_inheritance`,
`_readonly_memmap`, `_blocked_descending_sort`, `_fuzzy_symmetrise_blocked`,
`_check_runner_abort`, R0242's `loss_decomposition`, `cluster_locality_test`,
`cluster_rate_table`, `_dispersion`, `canonical_undirected_degrees`,
`post_canonical_tripwire`, `symmetrised_degree_once`, `weight_distribution`,
`gather_price`, `StageGuard0242`, `_HostWatchdog`, `_memmap_attestation`,
`_cluster_assignment`, `partition_reachability`, `partition_agreement`.

This module starts no child process, creates no CUDA context outside R0242's
registered torch transcription of R0226's `_kmeans`/`_assign`, hands cuVS
nothing, and contains no signalling construct of any kind.
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
from basemap.round0238_rung5 import (
    GRAPH_K,
    SPILL,
    TRUTH_PROBE_ROWS,
    TRUTH_PROBE_SEED,
    json_safe,
    truth_probe_query_rows,
)
from basemap.round0240_rung5 import REGISTERED_REACHABILITY_CEILING_C400
from basemap.round0241_qualify import (
    GPU_HOURS_CAP_NOTE,
    REGISTERED_SELECTED_CELL,
)
from basemap.round0242_locality import (
    StageGuard0242,
    partition_agreement,
    partition_reachability,
)
from basemap.round0243_residual import (
    CANONICAL_CAPABILITY,
    CLUSTERS,
    CONCENTRATION_TOP_M,
    DIMENSION,
    FUZZY_CAPABILITY,
    FUZZY_DEADLINE_S,
    FUZZY_FILE,
    FUZZY_SCHEMA,
    GPU_HOURS_CAP,
    HALT_CELL_TIE_AWARE_BUILDER_RATE,
    HALT_GLOBAL_TIE_AWARE_BUILDER_RATE,
    HALT_RULE_NOTE,
    HALT_SINGLE_CLUSTER_EXPOSURE_MULTIPLE,
    HALT_SINGLE_CLUSTER_SHARE,
    PARTITION_SEED,
    PERMUTATIONS,
    PERMUTATION_SEED,
    R0242_LOCALITY_SHA256,
    R0242_PRIMARY_CLUSTER_SHA256,
    R0242_TIE_AWARE_VECTOR_SHA256,
    R0242_UNSORTED_PHYSICAL_RATE_BYTES_PER_S,
    REGISTERED_INHERITANCE,
    RESIDUAL_CAPABILITY,
    RESIDUAL_DEADLINE_S,
    RESIDUAL_EARLY_FILE,
    RESIDUAL_EARLY_SCHEMA,
    RESIDUAL_FILE,
    RESIDUAL_SCHEMA,
    ROUND_ID,
    ROWS,
    Round0243Error,
    SAFETY_NOTE,
    SCOPE_NOTE,
    SORTED_GATHER_NOTE,
    SUBSTRATE_BYTES,
    TIE_AWARE_NOTE,
    canonical_undirected_degrees,
    cluster_locality_test,
    cluster_rate_table,
    exposure_profile,
    full_gather_ceiling,
    hot_cell_scan,
    io_counters,
    json_scrub,
    loss_decomposition,
    map_harm_assessment,
    observed_dispersion,
    post_canonical_tripwire,
    residual_verdict,
    sorted_gather_price,
    strict_reproduction_gate,
    symmetrised_degree_once,
    weight_distribution,
)
from experiments.round0238_nodes import (
    FUZZY_STRIPE_ROWS,
    _blocked_descending_sort,
    _check_runner_abort,
    _fuzzy_symmetrise_blocked,
)
from experiments.round0241_nodes import _readonly_memmap, verify_inheritance
from experiments.round0242_nodes import (
    EDGE_BLOCK,
    _cluster_assignment,
    _HostWatchdog,
    _memmap_attestation,
    _meminfo,
)

RESIDUAL_ACTION = "residual_100000k"
FUZZY_ACTION = "fuzzy_100000k"

#: The sorted-gather instrument. Two anchor counts: R0242's own `500,000`, so
#: the sorted and unsorted measurements are directly comparable, and
#: `10,000,000`, which review-0242-01/F8 named as the point that fits the
#: crossover without a 200x extrapolation.
SORTED_GATHER_SEED = 243_100
SORTED_GATHER_ANCHORS = (500_000, 10_000_000)
SORTED_GATHER_READ_BLOCK = 250_000
SORTED_GATHER_ID_BLOCK = 1_000_000
PRIMARY_LABEL_BLOCK = 10_000_000

#: The populations, in the order they are analysed and reported.
POPULATIONS = (
    "total_loss",
    "partition_limited_loss",
    "builder_loss_inside_partition",
)


class StageGuard0243(StageGuard0242):
    """R0242's corrected wall guard, unchanged, under this round's name.

    `StageGuard0242` already carries review-0241-01/F5's two corrections: the
    wall is stamped at stage completion rather than at seal time, and
    `deadline_reached` is a measurement rather than a hardcoded literal. It is
    subclassed rather than edited because R0242's file is a reviewed artifact
    and this round modifies nothing in it.
    """


def _bound_path(job: Mapping[str, Any], key: str, *, label: str) -> str:
    """Bind an input at its FULL signature and return its path."""
    reference = dict(job[key])
    path = str(reference.get("canonical_path") or "")
    if not path or not reference.get("sha256"):
        raise Round0243Error(f"{label} must be bound at a full sha256 signature")
    if expected_input_signature(path) != reference:
        raise Round0243Error(f"{label} content changed")
    return path


def _bound_vector(
    job: Mapping[str, Any], key: str, *, label: str, expect_sha256: str | None = None
) -> np.ndarray:
    """Load one of R0242's sealed per-row vectors as a read-only memmap."""
    reference = dict(job[key])
    path = _bound_path(job, key, label=label)
    if expect_sha256 is not None and str(reference.get("sha256")) != expect_sha256:
        raise Round0243Error(
            f"{label} hashes to {reference.get('sha256')!r}, registered "
            f"{expect_sha256!r}"
        )
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(array, np.memmap) or array.flags.writeable:
        raise Round0243Error(f"{label} is not a read-only np.memmap")
    return array


def _cache_state() -> dict[str, int]:
    values = _meminfo()
    return {
        "cached_bytes": int(values.get("Cached", 0)),
        "mem_available_bytes": int(values.get("MemAvailable", 0)),
        "mem_free_bytes": int(values.get("MemFree", 0)),
    }


def _sorted_gather(
    *,
    anchors: np.ndarray,
    ids: np.ndarray,
    host: np.ndarray,
    budget_s: float,
) -> dict[str, Any]:
    """Gather every anchor's k neighbours off the substrate in SORTED id order.

    The point of the instrument, from review-0242-01/F8: an unsorted gather of
    `500,000` anchors already physically read `92.94%` of the substrate, so its
    `12.39x` amplification was within `7.6%` of its arithmetic ceiling and
    cannot be carried `200x`. A gather whose targets are sorted and
    deduplicated reads each touched row ONCE, so its physical read is bounded
    above by the substrate itself however many anchors it serves. This measures
    that, rather than fitting a saturating rate.

    Nothing is retained: a checksum is accumulated so the pages are genuinely
    read, and every block is released before the next.
    """
    anchors = np.asarray(anchors, dtype=np.int64)
    targets: list[np.ndarray] = []
    for begin in range(0, anchors.size, SORTED_GATHER_ID_BLOCK):
        end = min(begin + SORTED_GATHER_ID_BLOCK, anchors.size)
        block = np.asarray(ids[anchors[begin:end]], dtype=np.int64).ravel()
        targets.append(np.unique(block))
        del block
        _check_runner_abort("R0243 sorted gather: id collection")
    distinct = np.unique(np.concatenate(targets)) if targets else np.empty(
        0, dtype=np.int64
    )
    del targets
    gc.collect()
    if distinct.size and (
        int(distinct.min()) < 0 or int(distinct.max()) >= int(ROWS)
    ):
        raise Round0243Error("R0243 sorted gather found an out-of-range target id")

    #: The guard is built AFTER the distinct set is known, so its `units_total`
    #: is the work it will actually do rather than an upper bound that would
    #: make its own calibration prediction pessimistic and refuse the stage.
    guard = StageGuard0243(
        label=f"sorted gather, {anchors.size} anchors",
        units_total=max(1, -(-int(distinct.size) // SORTED_GATHER_READ_BLOCK)),
        budget_s=float(budget_s), deadline_s=RESIDUAL_DEADLINE_S,
        abort_check=_check_runner_abort,
    )
    cache_before = _cache_state()
    io_before = io_counters()
    started = time.monotonic()
    checksum = 0.0
    for begin in range(0, distinct.size, SORTED_GATHER_READ_BLOCK):
        end = min(begin + SORTED_GATHER_READ_BLOCK, distinct.size)
        block = np.asarray(host[distinct[begin:end]])
        checksum += float(block[:, 0].sum(dtype=np.float64))
        del block
        guard.unit_done(f"sorted gather rows [{begin}, {end})")
    wall = time.monotonic() - started
    io_after = io_counters()
    guard.stop()
    cache_after = _cache_state()
    priced = sorted_gather_price(
        anchors=int(anchors.size),
        neighbours_per_row=GRAPH_K,
        row_bytes=DIMENSION * 4,
        distinct_rows_touched=int(distinct.size),
        substrate_bytes=SUBSTRATE_BYTES,
        wall_s=wall,
        physical_read_bytes=int(
            io_after["read_bytes"] - io_before["read_bytes"]
        ),
        label=(
            f"{anchors.size}-anchor k15 neighbour gather, targets SORTED and "
            "deduplicated, off the 153.6 GB substrate"
        ),
    )
    priced["checksum"] = checksum
    priced["wall_guard"] = guard.receipt()
    priced["page_cache_before"] = cache_before
    priced["page_cache_after"] = cache_after
    priced["page_cache_caveat"] = (
        "/proc/self/io read_bytes counts block-layer reads, so any part of the "
        "substrate already resident in page cache is NOT counted. This host "
        "holds ~118 GB of page cache against a 153.6 GB substrate, so the "
        "physical figure here is a LOWER bound on a cold read and the wall is "
        "a lower bound on a cold wall. The distinct-bytes figure beside it is "
        "arithmetic and is cache-independent."
    )
    del distinct
    gc.collect()
    return priced


# --------------------------------------------------------------------------- #
# node A — the tie-aware settlement
# --------------------------------------------------------------------------- #
def run_residual(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    """Part A — the same analysis, on the scale R0241 headlines."""
    manifest = active["manifest"]
    if str(manifest.get("round_id")) != ROUND_ID:
        raise Round0243Error("R0243 handler received another queue")
    started = time.monotonic()
    watchdog = _HostWatchdog()
    inheritance = verify_inheritance(job)
    output = create_fresh_directory(str(job["outputs"][0]), label="R0243 residual")

    # ---- bind every input, including R0242's sealed receipt and vectors ------ #
    r0242_locality = prompt_contract.read_sealed(
        _bound_path(job, "r0242_locality", label="R0243 R0242 locality receipt"),
        label="R0243 R0242 locality receipt",
    )
    if str(dict(job["r0242_locality"]).get("sha256")) != R0242_LOCALITY_SHA256:
        raise Round0243Error(
            "R0243 bound a loss-locality receipt that is not the one R0242 "
            "sealed"
        )
    if str(r0242_locality.get("round_id")) != "0242":
        raise Round0243Error("R0243 read a locality receipt from another round")

    probe_cluster = _bound_vector(
        job, "r0242_probe_cluster", label="R0243 probe cluster labels"
    )
    strict_recall = _bound_vector(
        job, "r0242_probe_strict_recall", label="R0243 probe strict recall"
    )
    tie_recall = _bound_vector(
        job, "r0242_probe_tie_aware_recall",
        label="R0243 probe tie-aware recall",
        expect_sha256=R0242_TIE_AWARE_VECTOR_SHA256,
    )
    probe_in_degree = _bound_vector(
        job, "r0242_probe_in_degree", label="R0243 probe in-degree"
    )
    primary_cluster = _bound_vector(
        job, "r0242_primary_cluster", label="R0243 primary cluster labels",
        expect_sha256=R0242_PRIMARY_CLUSTER_SHA256,
    )

    truth = prompt_contract.read_sealed(
        str(inheritance["truth"]["source"]["canonical_path"]), label="R0243 truth"
    )
    probe_rows = np.load(
        prompt_contract.verify_signature(
            dict(truth["outputs"]["query_rows"]), label="R0243 probe rows"
        ),
        allow_pickle=False,
    )
    truth_ids = np.load(
        prompt_contract.verify_signature(
            dict(truth["outputs"]["ids"]), label="R0243 truth ids"
        ),
        allow_pickle=False,
    )
    if not np.array_equal(probe_rows, truth_probe_query_rows(
        rows=ROWS, size=TRUTH_PROBE_ROWS, seed=TRUTH_PROBE_SEED
    )):
        raise Round0243Error(
            "R0243 sealed probe rows are not R0238's registered uniform draw"
        )
    reachability = prompt_contract.read_sealed(
        str(inheritance["reachability"]["source"]["canonical_path"]),
        label="R0243 reachability",
    )
    cell = next(iter(reachability["cells"]))
    sealed_reach = np.load(
        prompt_contract.verify_signature(
            dict(cell["strict_vector"]), label="R0243 sealed reachability vector"
        ),
        allow_pickle=False,
    )

    labels = np.asarray(probe_cluster, dtype=np.int64)
    if labels.size != int(probe_rows.size):
        raise Round0243Error(
            f"R0243 cluster labels cover {labels.size} of {probe_rows.size} "
            "probe rows"
        )

    # ---- stage 1 (H0): the strict scale, reproduced from R0242's sealed bytes -- #
    strict_decomposition = loss_decomposition(
        strict=np.asarray(strict_recall, dtype=np.float64),
        reachability=sealed_reach, k=GRAPH_K,
    )
    strict_vectors = strict_decomposition.pop("vectors")
    strict_populations = {
        "total_loss": (strict_vectors["lost"], strict_vectors["exposure_all"]),
        "partition_limited_loss": (
            strict_vectors["partition_lost"], strict_vectors["exposure_all"]
        ),
        "builder_loss_inside_partition": (
            strict_vectors["builder_lost"], strict_vectors["exposure_builder"]
        ),
    }
    strict_observed = {
        name: observed_dispersion(
            labels=labels, missing=missing, exposure=exposure,
            clusters=CLUSTERS, top_m=CONCENTRATION_TOP_M,
        )
        for name, (missing, exposure) in strict_populations.items()
    }
    reproduction = strict_reproduction_gate(
        measured_decomposition=strict_decomposition,
        sealed_decomposition=r0242_locality["decomposition"],
        measured_dispersion=strict_observed,
        sealed_tests=r0242_locality["cluster_locality_tests"],
    )
    if not reproduction["agree"]:
        raise Round0243Error(
            "R0243 STOP (H0): the strict decomposition and dispersion "
            "recomputed here do not reproduce R0242's sealed values on "
            f"{reproduction['disagreements']}. The tie-aware re-analysis would "
            "stratify a loss vector that is not the one R0242 published."
        )
    _check_runner_abort("R0243 strict reproduction gate passed")
    watchdog.poll("after the strict reproduction gate")

    # ---- stage 2: the TIE-AWARE scale, through the same decomposition ------- #
    tie_decomposition = loss_decomposition(
        strict=np.asarray(tie_recall, dtype=np.float64),
        reachability=sealed_reach, k=GRAPH_K,
    )
    tie_vectors = tie_decomposition.pop("vectors")
    tie_populations = {
        "total_loss": (tie_vectors["lost"], tie_vectors["exposure_all"]),
        "partition_limited_loss": (
            tie_vectors["partition_lost"], tie_vectors["exposure_all"]
        ),
        "builder_loss_inside_partition": (
            tie_vectors["builder_lost"], tie_vectors["exposure_builder"]
        ),
    }
    forgiveness = {
        "strict_total_missing_edges": int(
            strict_decomposition["total_missing_edges"]
        ),
        "tie_aware_total_missing_edges": int(
            tie_decomposition["total_missing_edges"]
        ),
        "strict_builder_missing_edges": int(
            strict_decomposition["builder_missing_edges"]
        ),
        "tie_aware_builder_missing_edges": int(
            tie_decomposition["builder_missing_edges"]
        ),
        "strict_partition_forced_missing_edges": int(
            strict_decomposition["partition_forced_missing_edges"]
        ),
        "tie_aware_partition_forced_missing_edges": int(
            tie_decomposition["partition_forced_missing_edges"]
        ),
        "note": (
            "the same imported loss_decomposition, run twice: once on R0242's "
            "sealed strict recall vector and once on its sealed tie-aware "
            "vector, against the same R0238 sealed reachability. This is the "
            "round's OWN decomposition rerun, which review-0242-01/F3 asked "
            "for and estimated conservatively as min(tie_missing, "
            "builder_missing)"
        ),
    }

    tie_tests: dict[str, Any] = {}
    tie_tables: dict[str, Any] = {}
    for name in POPULATIONS:
        missing, exposure = tie_populations[name]
        tie_tests[name] = cluster_locality_test(
            labels=labels, missing=missing, exposure=exposure,
            clusters=CLUSTERS, permutations=PERMUTATIONS, seed=PERMUTATION_SEED,
            top_m=CONCENTRATION_TOP_M, population=f"tie_aware_{name}",
            poll=_check_runner_abort,
        )
        tie_tables[name] = cluster_rate_table(
            labels=labels, missing=missing, exposure=exposure, clusters=CLUSTERS,
        )
        _check_runner_abort(f"R0243 tie-aware cluster test: {name}")
        watchdog.poll(f"after the tie-aware cluster test for {name}")

    # ---- stage 3: the exposure profile and the magnitude arms ---------------- #
    builder_missing_tie, builder_exposure = tie_populations[
        "builder_loss_inside_partition"
    ]
    builder_missing_strict, _ = strict_populations["builder_loss_inside_partition"]
    profile = exposure_profile(
        labels=labels, exposure=builder_exposure, clusters=CLUSTERS,
        guard_multiple=HALT_SINGLE_CLUSTER_EXPOSURE_MULTIPLE,
    )
    tie_scan = hot_cell_scan(
        labels=labels, missing=builder_missing_tie, exposure=builder_exposure,
        clusters=CLUSTERS,
        cell_rate_threshold=HALT_CELL_TIE_AWARE_BUILDER_RATE,
        share_threshold=HALT_SINGLE_CLUSTER_SHARE,
        exposure_multiple=HALT_SINGLE_CLUSTER_EXPOSURE_MULTIPLE,
    )
    strict_scan = hot_cell_scan(
        labels=labels, missing=builder_missing_strict, exposure=builder_exposure,
        clusters=CLUSTERS,
        cell_rate_threshold=HALT_CELL_TIE_AWARE_BUILDER_RATE,
        share_threshold=HALT_SINGLE_CLUSTER_SHARE,
        exposure_multiple=HALT_SINGLE_CLUSTER_EXPOSURE_MULTIPLE,
    )
    strict_builder_sealed = dict(
        r0242_locality["cluster_locality_tests"]["builder_loss_inside_partition"]
    )
    verdict = residual_verdict(
        reproduction=reproduction,
        tie_aware_scan=tie_scan,
        tie_aware_builder_test=tie_tests["builder_loss_inside_partition"],
        strict_builder_test=strict_builder_sealed,
        global_rate_threshold=HALT_GLOBAL_TIE_AWARE_BUILDER_RATE,
    )
    harm = map_harm_assessment(
        strict_decomposition=strict_decomposition,
        tie_aware_decomposition=tie_decomposition,
        tie_aware_scan=tie_scan,
        strict_scan=strict_scan,
        probe_rows=int(probe_rows.size),
        k=GRAPH_K,
    )

    #: The three cells review-0242-01 named, reported on both scales side by
    #: side with their in-degree interiors, because "which cells and how much"
    #: is what Phase 3 will act on.
    named_cells: list[dict[str, Any]] = []
    probe_degree = np.asarray(probe_in_degree, dtype=np.int64)
    #: A blocked bincount over R0242's sealed 100,000,000-row primary-cluster
    #: vector. Measured, not sampled: review-0242-01/F5's point is that cell
    #: 168 is an ORDINARY-sized cell, and a sampled estimate of that would be
    #: the wrong instrument for the correction it makes.
    partition_sizes = np.zeros(int(CLUSTERS), dtype=np.int64)
    for begin in range(0, int(primary_cluster.size), PRIMARY_LABEL_BLOCK):
        end = min(begin + PRIMARY_LABEL_BLOCK, int(primary_cluster.size))
        chunk = np.asarray(primary_cluster[begin:end], dtype=np.int64)
        partition_sizes += np.bincount(chunk, minlength=int(CLUSTERS))[
            : int(CLUSTERS)
        ]
        del chunk
        _check_runner_abort("R0243 primary-cluster bincount")
    partition_size_profile = {
        "rows_counted": int(partition_sizes.sum()),
        "clusters": int(CLUSTERS),
        "mean": float(partition_sizes.mean()),
        "min": int(partition_sizes.min()),
        "p25": float(np.percentile(partition_sizes, 25)),
        "p50": float(np.percentile(partition_sizes, 50)),
        "p75": float(np.percentile(partition_sizes, 75)),
        "max": int(partition_sizes.max()),
        "note": (
            "review-0242-01/F5: R0242 described cluster 168 as 'ONE cluster "
            "holding 0.29 percent of rows', which reads as small. Every cell "
            "in a c = 400 partition of 100,000,000 rows holds about 0.25%. "
            "This is the realised distribution, measured by a blocked "
            "bincount over the sealed label vector"
        ),
    }
    for cluster in (168, 9, 285):
        mask = labels == int(cluster)
        rows_here = int(mask.sum())
        if rows_here == 0:
            continue
        exposure_here = float(builder_exposure[mask].sum())
        named_cells.append({
            "cluster": int(cluster),
            "probe_rows": rows_here,
            "exposure_edges": exposure_here,
            "strict_builder_missing_edges": int(
                builder_missing_strict[mask].sum()
            ),
            "tie_aware_builder_missing_edges": int(
                builder_missing_tie[mask].sum()
            ),
            "strict_builder_rate": float(
                builder_missing_strict[mask].sum() / exposure_here
            ) if exposure_here else None,
            "tie_aware_builder_rate": float(
                builder_missing_tie[mask].sum() / exposure_here
            ) if exposure_here else None,
            "mean_in_degree": float(probe_degree[mask].mean()),
            "zero_in_degree_fraction": float((probe_degree[mask] == 0).mean()),
            "rows_in_the_100m_partition": int(partition_sizes[int(cluster)]),
            "percentile_of_cell_size": float(
                (partition_sizes <= partition_sizes[int(cluster)]).mean() * 100.0
            ),
        })
    watchdog.poll("after the magnitude arms")

    # ---- EARLY WRITE: the science, the instant it exists --------------------- #
    early_path = os.path.join(output, RESIDUAL_EARLY_FILE)
    atomic_write_new_json(early_path, prompt_contract.seal(json_safe(json_scrub({
        "schema": RESIDUAL_EARLY_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": str(manifest["release_sha"]),
        "rows": ROWS,
        "k": GRAPH_K,
        "clusters": CLUSTERS,
        "strict_reproduction_gate": reproduction,
        "strict_decomposition": strict_decomposition,
        "strict_observed_dispersion": strict_observed,
        "tie_aware_decomposition": tie_decomposition,
        "tie_forgiveness": forgiveness,
        "tie_aware_cluster_locality_tests": tie_tests,
        "exposure_profile": profile,
        "tie_aware_hot_cell_scan": tie_scan,
        "strict_hot_cell_scan": strict_scan,
        "residual_verdict": verdict,
        "map_harm_assessment": harm,
        "named_cells": named_cells,
        "partition_size_profile": partition_size_profile,
        "is_complete": False,
        "completed_by": RESIDUAL_FILE,
        "why_this_file_exists": (
            "review-0240-01/F2 and R0242 attempt 1: every stage that produces "
            "a result persists it before a later stage can fail"
        ),
    }))), immutable=True)

    # ---- stage 4: the SORTED gather, measured at two anchor counts ----------- #
    substrate_path = prompt_contract.verify_signature(
        dict(prompt_contract.read_sealed(
            str(inheritance["substrate"]["source"]["canonical_path"]),
            label="R0243 substrate manifest",
        )["substrate"]),
        label="R0243 substrate",
    )
    host = _readonly_memmap(
        substrate_path, label="R0243 substrate", shape=(ROWS, DIMENSION)
    )
    ids = _readonly_memmap(
        str(inheritance["graph"]["ids"]["canonical_path"]),
        label="R0243 graph ids", shape=(ROWS, GRAPH_K),
    )
    budget = float(job.get("stage_budget_s") or RESIDUAL_DEADLINE_S)
    rng = np.random.default_rng(SORTED_GATHER_SEED)
    sorted_gathers: dict[str, Any] = {}
    for anchor_count in SORTED_GATHER_ANCHORS:
        anchors = np.sort(rng.choice(
            int(ROWS), size=min(int(anchor_count), int(ROWS)), replace=False
        )).astype(np.int64)
        sorted_gathers[str(anchor_count)] = _sorted_gather(
            anchors=anchors, ids=ids, host=host, budget_s=budget
        )
        del anchors
        gc.collect()
        watchdog.poll(f"after the {anchor_count}-anchor sorted gather")
    largest = sorted_gathers[str(SORTED_GATHER_ANCHORS[-1])]
    delivered = float(largest["distinct_bytes"]) / max(
        float(largest["wall_s"]), 1e-9
    )
    gather_ceiling = full_gather_ceiling(
        substrate_bytes=SUBSTRATE_BYTES,
        measured_delivered_rate_bytes_per_s=delivered,
        measured_physical_read_fraction_of_substrate=float(
            largest["physical_read_as_fraction_of_substrate"]
        ),
        r0242_unsorted_physical_rate_bytes_per_s=(
            R0242_UNSORTED_PHYSICAL_RATE_BYTES_PER_S
        ),
    )

    # ---- stage 5: the discharge review-0242-01/F9.3 asked for --------------- #
    # R0242's strongest partition claim - 99.9486% of 500,000 rows bit-identical
    # to R0238's sealed vector - rested on a reproduced reachability vector that
    # was never sealed, so no reviewer could check it. That realisation is gone.
    # This round re-realises the partition once and SEALS both the spill
    # assignment and the reproduced reachability vector, so the equivalent claim
    # here is checkable from bytes. It does NOT gate Part B: the tie-aware
    # analysis above is stratified by R0242's SEALED labels, and Part B consumes
    # only the two graph arrays. A disagreement is published as a finding.
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
    fresh_primary = assignment[:, 0].astype(np.int16)
    fresh_probe_cluster = assignment[
        np.asarray(probe_rows, dtype=np.int64), 0
    ].astype(np.int64)
    label_identity = {
        "probe_rows_compared": int(labels.size),
        "probe_rows_with_identical_primary_cluster": int(
            (fresh_probe_cluster == labels).sum()
        ),
        "probe_row_primary_cluster_identity_fraction": float(
            (fresh_probe_cluster == labels).mean()
        ),
        "note": (
            "cluster ids are a labelling of one k-means realisation, so two "
            "realisations agreeing on a row's id at all is a strong statement "
            "about the realisation, not a weak one; it is reported as a "
            "measurement and gates nothing"
        ),
    }
    del assignment
    gc.collect()
    watchdog.poll("after the reachability reproduction")

    vector_dir = ensure_data_directory(os.path.join(output, "vectors"))
    saved = {
        "reproduced_partition_reachability": atomic_save_new_npy(
            os.path.join(vector_dir, "reproduced-strict-c400.f64.npy"),
            np.asarray(reproduced_reach, dtype=np.float64), immutable=True,
        ),
        "reproduced_primary_cluster": atomic_save_new_npy(
            os.path.join(vector_dir, "reproduced-primary-cluster-c400.i16.npy"),
            fresh_primary, immutable=True,
        ),
        "probe_tie_aware_missing_edges": atomic_save_new_npy(
            os.path.join(vector_dir, "probe-tie-aware-missing-edges.i16.npy"),
            tie_vectors["lost"].astype(np.int16), immutable=True,
        ),
        "probe_tie_aware_builder_missing_edges": atomic_save_new_npy(
            os.path.join(
                vector_dir, "probe-tie-aware-builder-missing-edges.i16.npy"
            ),
            tie_vectors["builder_lost"].astype(np.int16), immutable=True,
        ),
    }
    del fresh_primary
    gc.collect()

    receipt = prompt_contract.seal(json_safe(json_scrub({
        "schema": RESIDUAL_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": str(manifest["release_sha"]),
        "capability": RESIDUAL_CAPABILITY,
        "rows": ROWS,
        "k": GRAPH_K,
        "spill": SPILL,
        "clusters": CLUSTERS,
        "cell": REGISTERED_SELECTED_CELL,
        "registered_inheritance": REGISTERED_INHERITANCE,
        "inheritance": inheritance,
        "halt_rule": HALT_RULE_NOTE,
        "tie_aware_note": TIE_AWARE_NOTE,
        "strict_reproduction_gate": reproduction,
        "strict_decomposition": strict_decomposition,
        "strict_observed_dispersion": strict_observed,
        "tie_aware_decomposition": tie_decomposition,
        "tie_forgiveness": forgiveness,
        "tie_aware_cluster_locality_tests": tie_tests,
        "tie_aware_cluster_rate_tables": tie_tables,
        "exposure_profile": profile,
        "tie_aware_hot_cell_scan": tie_scan,
        "strict_hot_cell_scan": strict_scan,
        "named_cells": named_cells,
        "partition_size_profile": partition_size_profile,
        "residual_verdict": verdict,
        "map_harm_assessment": harm,
        "sorted_gathers": sorted_gathers,
        "full_sorted_gather_prediction": gather_ceiling,
        "partition": partition,
        "partition_agreement_with_r0238": agreement_partition,
        "partition_label_identity_with_r0242": label_identity,
        "partition_discharge_note": (
            "review-0242-01/F9.3: R0242's reproduced reachability vector was "
            "not sealed, so its 99.9486% row-identity claim was the one line "
            "in that round a reviewer could not check. This round seals its "
            "own reproduced vector and the primary-cluster labels it came "
            "from. It gates nothing here: the tie-aware analysis is "
            "stratified by R0242's SEALED probe labels and Part B consumes "
            "only the two graph arrays."
        ),
        "vectors_saved": {
            name: expected_input_signature(path) for name, path in saved.items()
        },
        "residual_first_write": expected_input_signature(early_path),
        "bulk_input_memmap_attestation": _memmap_attestation({
            "substrate": host, "graph_ids": ids,
            "r0242_tie_aware_recall": tie_recall,
            "r0242_primary_cluster": primary_cluster,
        }),
        "host_watchdog": watchdog.receipt(),
        "cuvs_calls": 0,
        "cuda_context_created": True,
        "cuda_context_note": (
            "one CUDA context, created by R0242's registered torch "
            "transcription of R0226's _kmeans/_assign for the partition "
            "re-realisation and by nothing else. cuVS is not called and no "
            "managed-memory handle to an anonymous buffer is ever created"
        ),
        "child_processes_launched": 0,
        "signal_delivered": False,
        "abort_policy": (
            "in-band cooperative flag only, polled every sorted-gather block, "
            "every assignment block and every 500 permutations; no SIGTERM, no "
            "SIGKILL, no ptrace, and no child process exists to signal"
        ),
        "safety_note": SAFETY_NOTE,
        "scope_note": SCOPE_NOTE,
        "sorted_gather_note": SORTED_GATHER_NOTE,
        "gpu_hours_cap": GPU_HOURS_CAP,
        "gpu_hours_cap_note": GPU_HOURS_CAP_NOTE,
        "training_performed": False,
        "gate_registered": False,
        "adoption_claimed": False,
        "map_quality_claimed": False,
        "performance": {"total_wall_s": time.monotonic() - started},
    })))
    atomic_write_new_json(
        os.path.join(output, RESIDUAL_FILE), receipt, immutable=True
    )


# --------------------------------------------------------------------------- #
# node B — the fuzzy symmetrisation and the post-canonicalization tripwire
# --------------------------------------------------------------------------- #
def run_fuzzy(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    """Part B — symmetrise, then run the tripwire where the v1 defect arose."""
    manifest = active["manifest"]
    if str(manifest.get("round_id")) != ROUND_ID:
        raise Round0243Error("R0243 handler received another queue")
    started = time.monotonic()
    watchdog = _HostWatchdog()
    inheritance = verify_inheritance(job)

    # Part A's receipt is an INTRA-queue artifact, so it cannot be bound at a
    # hash the manifest was written before it existed. It is bound instead by
    # its own seal - `read_sealed` recomputes `identity_sha256` over the
    # content and refuses a tampered file - plus its registered schema, round
    # id and release commit. It does not consult job state.
    residual_path = str(job["residual_reference"])
    if not os.path.isfile(residual_path):
        raise Round0243Error(
            f"R0243 Part B cannot run: Part A's receipt is absent at "
            f"{residual_path}"
        )
    residual = prompt_contract.read_sealed(
        residual_path, label="R0243 residual receipt"
    )
    if str(residual.get("schema")) != RESIDUAL_SCHEMA or str(
        residual.get("round_id")
    ) != ROUND_ID:
        raise Round0243Error("R0243 Part B read a receipt from another round")
    if str(residual.get("release_sha")) != str(manifest["release_sha"]):
        raise Round0243Error(
            "R0243 Part B and Part A did not run from the same release commit"
        )
    verdict = dict(residual["residual_verdict"])
    if not verdict.get("part_b_may_run"):
        raise Round0243Error(
            "R0243 STOP before Part B: the registered residual rule halted it. "
            f"H0 reproduction {verdict['h0_strict_reproduction_agrees']}, "
            f"H1 global tie-aware builder rate "
            f"{verdict['h1_global_tie_aware_builder_rate']} against "
            f"{verdict['h1_threshold']}, H2 cells firing all three arms "
            f"{verdict['h2_cells_firing_all_three_arms']} "
            f"{verdict['h2_firing_clusters']}. This is the registered "
            "behaviour, not a failure."
        )

    output = create_fresh_directory(str(job["outputs"][0]), label="R0243 fuzzy")
    ids = _readonly_memmap(
        str(inheritance["graph"]["ids"]["canonical_path"]),
        label="R0243 graph ids", shape=(ROWS, GRAPH_K),
    )
    cosines = _readonly_memmap(
        str(inheritance["graph"]["cosines"]["canonical_path"]),
        label="R0243 builder cosines", shape=(ROWS, GRAPH_K),
    )

    budget = float(job.get("stage_budget_s") or FUZZY_DEADLINE_S)
    io_before = io_counters()
    sort_guard = StageGuard0243(
        label="descending sort", units_total=1, budget_s=budget,
        deadline_s=FUZZY_DEADLINE_S, abort_check=_check_runner_abort,
    )
    ids_sorted, cos_sorted = _blocked_descending_sort(
        ids=ids, cosines=cosines, rows=ROWS
    )
    sort_guard.unit_done("descending sort")
    sort_guard.stop()
    watchdog.poll("after the descending sort")

    dists = np.maximum((1.0 - cos_sorted).astype(np.float32), 0.0)
    if not np.isfinite(dists).all():
        raise Round0243Error("R0243 candidate distances are not finite")
    del cos_sorted
    gc.collect()
    watchdog.poll("after the distance transform")

    import umap.umap_ as umap_api

    ids_out = atomic_save_new_npy(
        os.path.join(output, "graph-k15-ids.i32.npy"), ids_sorted, immutable=True
    )
    fuzzy_guard = StageGuard0243(
        label="fuzzy symmetrisation",
        units_total=-(-ROWS // FUZZY_STRIPE_ROWS),
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
        raise Round0243Error(
            "R0243 fuzzy weights are invalid: "
            f"min {weights['min']}, max {weights['max']}, "
            f"{weights['non_finite_entries']} non-finite"
        )
    watchdog.poll("after the weight scan")

    # Symmetrised degree is reported ONCE. `A + A^T - A o A^T` is symmetric, so
    # in-degree and out-degree are the same number for every row; reporting
    # them as two gating quantities shipped as an identity from R0237 to R0240
    # (review-0240-01/F1). The identity is CROSS-CHECKED on a seeded sample and
    # published as a cross-check, not as a second measurement.
    degrees = symmetrised_degree_once(
        src=src, dst=dst, rows=ROWS, block=EDGE_BLOCK
    )
    symmetrised_degree = degrees.pop("degree")
    if not degrees["identity_cross_check"][
        "in_degree_equals_out_degree_on_every_sampled_row"
    ]:
        raise Round0243Error(
            "R0243 symmetrisation produced an asymmetric adjacency: in-degree "
            "and out-degree differ on a sampled row, which the set operation "
            "A + A^T - A o A^T makes impossible"
        )
    del symmetrised_degree
    gc.collect()
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

    # Three streamed, memmappable `.npy` arrays plus a small header `.npz`, not
    # one bulk `.npz`: `zipfile` cannot stream a member, so the archive path
    # materialises about 20 GB of ANONYMOUS memory per 10 GB member inside an
    # `io.BytesIO` for no benefit and yields an archive no 100M trainer can
    # memmap. Registered in the round file in advance.
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
        "clusters": CLUSTERS,
        "cell": REGISTERED_SELECTED_CELL,
        "registered_inheritance": REGISTERED_INHERITANCE,
        "inheritance": inheritance,
        "residual_verdict_that_permitted_this_node": verdict,
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
                "51-pass model's unit is right for it. It performs NO "
                "substrate gather at all: the fuzzy distances come from the "
                "builder's sealed cosines. The GATHER term is therefore priced "
                "separately in Part A's receipt, and priced as a SORTED gather "
                "whose physical read is bounded above by the substrate itself"
            ),
            "gather_term_lives_in": "Part A: sorted_gathers, "
                                    "full_sorted_gather_prediction",
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
        raise Round0243Error(
            "R0243 POST-CANONICALIZATION DEGREE-ZERO TRIPWIRE FAILED at "
            f"100,000,000 rows: {tripwire['zero_degree_rows']} rows carry no "
            "canonical edge. This is the v1 failure mode (R0034: 2,779,481 "
            "rows, ~1.85%) at the exact step where it arose, and the evidence "
            "is sealed beside this failure. The graph must not be used."
        )


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    action = str(job.get("action") or "")
    if action == RESIDUAL_ACTION:
        run_residual(active, job)
    elif action == FUZZY_ACTION:
        run_fuzzy(active, job)
    else:
        raise Round0243Error(f"R{ROUND_ID} does not authorize action {action!r}")


__all__ = [
    "FUZZY_ACTION",
    "POPULATIONS",
    "RESIDUAL_ACTION",
    "SORTED_GATHER_ANCHORS",
    "SORTED_GATHER_SEED",
    "StageGuard0243",
    "run_fuzzy",
    "run_job",
    "run_residual",
]
