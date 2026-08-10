"""Execute R0241 — qualify the 100,000,000-row k15 neighbour graph. Two nodes.

Nothing is built, assembled or symmetrised. R0240's sealed graph arrays and
build-ladder receipt, and R0238's substrate, uniform truth probe and `c = 400`
reachability ceiling, are bound at their FULL sha256 signatures and every
registered literal about them is asserted before a single row is read.

* `tripwire_100000k` (no CUDA context; one 6 GB int32 array) runs the **R0215
  degree-zero tripwire, for the first time at this scale**, over all
  `100,000,000` rows in BOTH directions, plus self-loops, duplicates,
  out-of-range ids and rows below `k`. It runs FIRST because R0240 proved the
  cheapest check in the queue can be held hostage behind the most expensive one.
* `recall_100000k` (no CUDA context; `500,000` scattered anchors) recomputes
  candidate cosines for the registered probe's rows ONLY, independently of the
  builder's accumulator, scores strict and tie-aware containment against R0238's
  sealed brute-force truth, adjudicates every `min = 0.0` row individually, and
  publishes the probe's sampling power honestly. It then discharges the three
  arithmetic debts - the per-rung re-derivation, the `c = 400` per-seed movement
  table and the measured tolerance chain - from sealed receipts only.

Every registered check is IMPORTED, never re-typed: `_graph_validity_blocked`
(which wraps R0220's `graph_validity`), `_score_probe`, `strict_containment_rows`,
`tie_aware_rows`, `summarize`, `zero_recall_forensic`, `truth_probe_query_rows`,
`rung_derivation`, `imbalance_tolerance`, `carry_distance` and
`replicate_grid_table`. What this module writes is the one measurement no
reviewed function makes - per-row **in-degree** over the raw directed graph -
and the poll points and wall guard R0240's qualification node did not have.

This module starts no child process, creates no CUDA context, hands cuVS
nothing, and contains no signalling construct of any kind.
"""
from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap import round0113_prompt_contrast as prompt_contract
from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_write_new_json, create_fresh_directory
from basemap.round0220_cuvs_qualification import _first_occurrence_mask
from basemap.round0235_rung2 import fit_device_law, rung_derivation
from basemap.round0236_rung3 import carry_distance, imbalance_tolerance
from basemap.round0238_rung5 import (
    GRAPH_K,
    IMBALANCE_REPLICATE_SEEDS,
    PHASE2_RUNGS,
    PRIMARY_IMBALANCE_SEED,
    SPILL,
    TRUTH_PROBE_ROWS,
    TRUTH_PROBE_SEED,
    ZERO_RECALL_FORENSIC_NOTE,
    json_safe,
    replicate_grid_table,
    truth_probe_query_rows,
)
from basemap.round0240_rung5 import (
    REGISTERED_REACHABILITY_CEILING_C400,
    verify_inherited_reachability,
    verify_inherited_substrate,
    verify_inherited_truth,
)
from basemap.round0241_qualify import (
    CPU_ARITHMETIC_NOTE,
    DEBTS_NOTE,
    EARLY_PERSIST_NOTE,
    DIMENSION,
    DEGREE_SEMANTICS_NOTE,
    GPU_HOURS_CAP,
    GPU_HOURS_CAP_NOTE,
    GRAPH_UNDER_TEST,
    HUBNESS_NOTE,
    INHERITANCE_NOTE,
    NO_SYMMETRISATION_NOTE,
    PROBE_GATHER_BLOCK,
    RECALL_CAPABILITY,
    RECALL_DEADLINE_S,
    RECALL_EARLY_FILE,
    RECALL_EARLY_SCHEMA,
    RECALL_FILE,
    RECALL_NOTE,
    RECALL_SCHEMA,
    REGISTERED_GRAPH_ARRAY_BYTES,
    REGISTERED_GRAPH_COS_SHA256,
    REGISTERED_GRAPH_IDS_SHA256,
    REGISTERED_LADDER_RECEIPT_SHA256,
    REGISTERED_SELECTED_CELL,
    REGISTERED_SELECTED_CLUSTERS,
    ROUND_ID,
    ROWS,
    Round0241Error,
    SAFETY_NOTE,
    SCOPE_NOTE,
    SYMMETRISED_DEGREE_BOUND_NOTE,
    SYMMETRISED_IDENTITY_NOTE,
    StageGuard,
    TRIPWIRE_BLOCK,
    TRIPWIRE_CAPABILITY,
    TRIPWIRE_DEADLINE_S,
    TRIPWIRE_FILE,
    TRIPWIRE_SCHEMA,
    VACUITY_NOTE,
    WALL_GUARD_NOTE,
    cross_check_structural,
    per_seed_movement,
    probe_power_statement,
    recall_verdict,
    reconcile_in_degree,
    tolerance_chain,
    tripwire_verdict,
)
from experiments.round0238_nodes import (
    _check_runner_abort,
    _graph_validity_blocked,
    _inherited_replicate_grid,
    _score_probe,
)

TRIPWIRE_ACTION = "tripwire_100000k"
RECALL_ACTION = "recall_100000k"


# --------------------------------------------------------------------------- #
# binding: every input at its full signature, checked before any measurement
# --------------------------------------------------------------------------- #
def _bound(
    job: Mapping[str, Any], key: str, *, label: str
) -> tuple[str, dict[str, Any]]:
    """Resolve one hash-bound input, hashing it exactly once.

    The comparison is `prompt_contract.verify_signature`'s own - the observed
    signature must equal the declared one field for field - but the observed
    signature is returned rather than discarded, so a 6 GB array is read once
    per node instead of three times.
    """
    reference = dict(job[key])
    if not reference.get("sha256"):
        raise Round0241Error(
            f"{label} must be bound at a full sha256 signature; R0241 inherits "
            "sealed artifacts across queues and never intra-queue"
        )
    path = str(reference.get("canonical_path") or "")
    if not path:
        raise Round0241Error(f"{label} signature carries no canonical path")
    observed = expected_input_signature(path)
    if observed != reference:
        raise Round0241Error(f"{label} content changed")
    return path, observed


def _bound_sealed(
    job: Mapping[str, Any], key: str, *, label: str
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    path, signature = _bound(job, key, label=label)
    return path, signature, prompt_contract.read_sealed(path, label=label)


def _readonly_memmap(path: str, *, label: str, shape: tuple[int, ...]) -> np.ndarray:
    """Open an array read-only and assert it really is a read-only memmap.

    This round hands cuVS nothing, so the precondition is satisfied vacuously.
    It is asserted anyway: the rule earned at R0232 is that the assertion is
    what makes the property checkable, and a round that quietly stops asserting
    it is how the property gets lost.
    """
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(array, np.memmap) or array.flags.writeable:
        raise Round0241Error(f"{label} is not a read-only np.memmap")
    if array.shape != shape:
        raise Round0241Error(f"{label} has shape {array.shape}, expected {shape}")
    return array


def verify_inheritance(job: Mapping[str, Any]) -> dict[str, Any]:
    """Assert every registered literal about every inherited artifact.

    Fail-closed and cheap in JSON, exact in bytes: the three R0238 manifests are
    checked field by field through `basemap.round0240_rung5`'s registered
    verifiers, and R0240's ladder receipt and both graph arrays are checked at
    their full sha256 signatures. A mismatch raises and the round STOPS.
    """
    _substrate_path, substrate_signature, substrate = _bound_sealed(
        job, "substrate_manifest", label="R0241 inherited substrate manifest"
    )
    _truth_path, truth_signature, truth = _bound_sealed(
        job, "truth_reference", label="R0241 inherited truth probe"
    )
    _reach_path, reach_signature, reachability = _bound_sealed(
        job, "reachability_reference", label="R0241 inherited reachability"
    )
    _ladder_path, ladder_signature, ladder = _bound_sealed(
        job, "ladder_reference", label="R0241 inherited R0240 build ladder"
    )
    if str(ladder_signature.get("sha256")) != REGISTERED_LADDER_RECEIPT_SHA256:
        raise Round0241Error(
            "STOP: the build-ladder receipt this round binds is not the one "
            f"R0240 sealed ({ladder_signature.get('sha256')!r} against the "
            f"registered {REGISTERED_LADDER_RECEIPT_SHA256!r})"
        )
    if int(ladder.get("rows", -1)) != ROWS:
        raise Round0241Error("R0240 ladder receipt is not the 100,000,000 rung")
    selected = int(
        (ladder.get("cluster_selection") or {}).get("selected_clusters", -1)
    )
    if selected != REGISTERED_SELECTED_CLUSTERS:
        raise Round0241Error(
            f"R0240 selected c = {selected}, registered "
            f"{REGISTERED_SELECTED_CLUSTERS}"
        )

    _ids_path, ids_signature = _bound(
        job, "graph_ids", label="R0241 graph neighbour ids"
    )
    _cos_path, cos_signature = _bound(
        job, "graph_cos", label="R0241 builder cosines"
    )
    for signature, digest, label in (
        (ids_signature, REGISTERED_GRAPH_IDS_SHA256, "graph-k15-ids.i32.npy"),
        (cos_signature, REGISTERED_GRAPH_COS_SHA256, "graph-k15-cos.f32.npy"),
    ):
        if str(signature.get("sha256")) != digest or int(
            signature.get("bytes", -1)
        ) != REGISTERED_GRAPH_ARRAY_BYTES:
            raise Round0241Error(
                f"STOP: {label} is {signature.get('sha256')!r} at "
                f"{signature.get('bytes')} bytes, registered {digest!r} at "
                f"{REGISTERED_GRAPH_ARRAY_BYTES}. The graph this round "
                "qualifies is not the graph R0240 built"
            )

    return {
        "note": INHERITANCE_NOTE,
        "substrate": {
            "source": substrate_signature,
            **verify_inherited_substrate(substrate),
        },
        "truth": {
            "source": truth_signature,
            **verify_inherited_truth(truth),
        },
        "reachability": {
            "source": reach_signature,
            **verify_inherited_reachability(reachability),
        },
        "build_ladder": {
            "source": ladder_signature,
            "verified": True,
            "round_id_of_source": str(ladder.get("round_id")),
            "queue_of_source": "/data/latent-basemap/runs/round-0240/queue",
            "selected_clusters": selected,
            "selected_cell": REGISTERED_SELECTED_CELL,
            "receipt_round_id_disclosure": (
                "the receipt stamps round_id 0238 because R0240 called R0238's "
                "run_ladder unchanged; it is an R0240 artifact by queue, "
                "release commit and output path, as R0240's own attestation "
                "states"
            ),
        },
        "graph": {
            "ids": ids_signature,
            "cosines": cos_signature,
            "verified": True,
            "rows": ROWS,
            "k": GRAPH_K,
        },
    }


# --------------------------------------------------------------------------- #
# node A — the R0215 degree-zero tripwire, in BOTH directions
# --------------------------------------------------------------------------- #
def _degree_pass(ids: np.ndarray, *, guard: StageGuard) -> dict[str, Any]:
    """Per-row usable OUT-degree and per-row IN-degree over the raw graph.

    No reviewed function builds an in-degree accumulator, which is why R0238's
    qualification could only reach in-degree through the `~4.3 h` fuzzy
    symmetrisation. This is the missing pass and it is one sequential walk of
    the id array.

    "Usable" is R0220's own definition, reproduced from its own
    `_first_occurrence_mask` rather than re-derived: an entry counts when it is
    the first occurrence of that id within the row, is in range, and is not the
    row itself. The same mask drives both accumulators, so out-degree and
    in-degree are two projections of one relation and cannot disagree about
    which edges exist.
    """
    rows = int(ids.shape[0])
    out_degree = np.zeros(rows, dtype=np.uint8)
    in_degree_usable = np.zeros(rows, dtype=np.int32)
    in_degree_raw = np.zeros(rows, dtype=np.int32)
    totals = {
        "out_of_range_entries": 0, "rows_with_out_of_range": 0,
        "self_loop_entries": 0, "rows_with_self_loop": 0,
        "duplicate_entries": 0, "rows_with_duplicates": 0,
    }
    width: int | None = None
    scanned = 0
    for begin in range(0, rows, TRIPWIRE_BLOCK):
        end = min(begin + TRIPWIRE_BLOCK, rows)
        stripe = np.asarray(ids[begin:end]).astype(np.int64)
        width = int(stripe.shape[1])
        out_of_range = (stripe < 0) | (stripe >= rows)
        self_ids = np.arange(begin, end, dtype=np.int64)[:, None]
        self_loops = (stripe == self_ids) & ~out_of_range
        fresh = _first_occurrence_mask(stripe)
        usable = fresh & ~out_of_range & ~self_loops
        out_degree[begin:end] = usable.sum(axis=1).astype(np.uint8)
        in_degree_usable += np.bincount(
            stripe[usable], minlength=rows
        ).astype(np.int32)
        in_degree_raw += np.bincount(
            stripe[~out_of_range], minlength=rows
        ).astype(np.int32)
        totals["out_of_range_entries"] += int(out_of_range.sum())
        totals["rows_with_out_of_range"] += int(out_of_range.any(axis=1).sum())
        totals["self_loop_entries"] += int(self_loops.sum())
        totals["rows_with_self_loop"] += int(self_loops.any(axis=1).sum())
        totals["duplicate_entries"] += int((~fresh).sum())
        totals["rows_with_duplicates"] += int((~fresh).any(axis=1).sum())
        scanned += end - begin
        del stripe, out_of_range, self_ids, self_loops, fresh, usable
        guard.unit_done(f"degree pass rows [{begin}, {end})")
    if scanned != rows or width is None:
        raise Round0241Error(
            f"R0241 degree pass covered {scanned} of {rows} rows"
        )
    out_zero = out_degree == 0
    in_zero = in_degree_usable == 0
    #: One sort, reused for every hubness statistic. `np.percentile` would
    #: partition the 400 MB array once per quantile.
    ordered = np.sort(in_degree_raw)
    total_in = int(ordered.sum(dtype=np.int64))
    top_percent = max(1, rows // 100)
    hubness = {
        str(q): float(ordered[min(rows - 1, int(q / 100.0 * rows))])
        for q in (50, 75, 90, 99, 99.9, 99.99)
    }
    top_share = (
        float(ordered[-top_percent:].sum(dtype=np.int64)) / float(total_in)
        if total_in else None
    )
    del ordered
    return {
        "own_structural": {
            "rows": rows,
            "width": width,
            "min_usable_degree": int(out_degree.min()),
            "rows_below_k": int((out_degree < GRAPH_K).sum()),
            "zero_degree_rows": int(out_zero.sum()),
            **totals,
        },
        "out_degree": {
            "graph": GRAPH_UNDER_TEST,
            "definition": (
                "usable entries per row - first occurrence within the row, in "
                "range, not the row itself. It is exactly k for every row "
                "unless the builder emitted a degenerate entry"
            ),
            "expected_by_construction": GRAPH_K,
            "zero_rows": int(out_zero.sum()),
            "min": int(out_degree.min()),
            "max": int(out_degree.max()),
            "mean": float(out_degree.mean()),
            "median": float(np.median(out_degree)),
            "rows_below_k": int((out_degree < GRAPH_K).sum()),
        },
        "in_degree_usable": {
            "graph": GRAPH_UNDER_TEST,
            "definition": (
                "raw in-degree counting only usable entries, so an in-edge "
                "that is a self-loop or an intra-row duplicate does not "
                "rescue a row from isolation"
            ),
            "zero_rows": int(in_zero.sum()),
            "min": int(in_degree_usable.min()),
            "max": int(in_degree_usable.max()),
            "mean": float(in_degree_usable.mean()),
            "median": float(np.median(in_degree_usable)),
        },
        "in_degree_raw": {
            "graph": GRAPH_UNDER_TEST,
            "definition": (
                "how many rows name this row as a neighbour, counting every "
                "in-range entry exactly as a plain bincount over the id array "
                "does; this is the quantity review-0240-01/F1 measured"
            ),
            "zero_rows": int((in_degree_raw == 0).sum()),
            "min": int(in_degree_raw.min()),
            "max": int(in_degree_raw.max()),
            "mean": float(in_degree_raw.mean()),
            "hubness_percentiles": hubness,
            "rows_at_or_above_1000": int((in_degree_raw >= 1_000).sum()),
            "share_of_edges_in_top_1_percent_of_rows": top_share,
        },
        "symmetrised": {
            "isolated_rows": int((out_zero & in_zero).sum()),
            "degree_lower_bound_min": int(
                np.maximum(out_degree.astype(np.int64), in_degree_usable).min()
            ),
            "degree_upper_bound_min": int(
                (out_degree.astype(np.int64) + in_degree_usable).min()
            ),
            "degree_upper_bound_mean": float(
                (out_degree.astype(np.int64) + in_degree_usable).mean()
            ),
            "bound_note": SYMMETRISED_DEGREE_BOUND_NOTE,
            "identity_note": SYMMETRISED_IDENTITY_NOTE,
        },
        "edge_totals": {
            "usable_directed_edges": int(in_degree_usable.sum(dtype=np.int64)),
            "entries_scanned": int(rows) * int(width),
        },
    }


def run_tripwire(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    """The R0215 degree-zero tripwire at 100,000,000 rows, for the first time."""
    manifest = active["manifest"]
    if str(manifest.get("round_id")) != ROUND_ID:
        raise Round0241Error("R0241 handler received another queue")
    started = time.monotonic()
    inheritance = verify_inheritance(job)
    output = create_fresh_directory(str(job["outputs"][0]), label="R0241 tripwire")

    ids_path = str(inheritance["graph"]["ids"]["canonical_path"])
    ids = _readonly_memmap(
        ids_path, label="R0241 graph ids", shape=(ROWS, GRAPH_K)
    )
    if ids.dtype != np.int32:
        raise Round0241Error(f"R0241 graph ids are {ids.dtype}, expected int32")

    units = -(-ROWS // TRIPWIRE_BLOCK)
    budget = float(job.get("stage_budget_s") or TRIPWIRE_DEADLINE_S)
    own_guard = StageGuard(
        label="tripwire degree pass", units_total=units, budget_s=budget,
        deadline_s=TRIPWIRE_DEADLINE_S, abort_check=_check_runner_abort,
    )
    measured = _degree_pass(ids, guard=own_guard)

    # Predict the reviewed scan from THIS stage's own measured rate before
    # paying for it. R0240's lesson: a bound that is only checked at admission
    # is not a bound.
    reviewed_guard = StageGuard(
        label="tripwire reviewed structural scan", units_total=units,
        budget_s=budget, deadline_s=TRIPWIRE_DEADLINE_S,
        abort_check=_check_runner_abort,
    )
    projection = dict(own_guard.predict())
    projection["applied_to"] = "the reviewed blocked structural scan"
    if projection["predicted_with_margin_s"] > budget:
        raise Round0241Error(
            "R0241 refuses the reviewed structural scan a priori: predicted "
            f"{projection['predicted_with_margin_s']:.1f} s against a budget of "
            f"{budget:.1f} s, on this node's own measured rate. Reported, not "
            "trimmed."
        )
    reviewed = _graph_validity_blocked(ids, rows=ROWS, block=TRIPWIRE_BLOCK)
    reviewed_guard.units_done = units
    cross_check = cross_check_structural(
        reviewed=reviewed, own=measured["own_structural"]
    )

    verdict = tripwire_verdict(
        out_degree_zero_rows=measured["out_degree"]["zero_rows"],
        in_degree_zero_rows=measured["in_degree_raw"]["zero_rows"],
        symmetrised_isolated_rows=measured["symmetrised"]["isolated_rows"],
        raw_zero_degree_rows=int(reviewed["zero_degree_rows"]),
    )
    reconciliation = reconcile_in_degree(
        measured_zero_in_degree_rows=measured["in_degree_raw"]["zero_rows"],
        measured_in_degree_max=measured["in_degree_raw"]["max"],
        measured_in_degree_mean=measured["in_degree_raw"]["mean"],
        measured_out_degree_min=measured["out_degree"]["min"],
        measured_out_degree_zero_rows=measured["out_degree"]["zero_rows"],
    )
    receipt = prompt_contract.seal(json_safe({
        "schema": TRIPWIRE_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": str(manifest["release_sha"]),
        "capability": TRIPWIRE_CAPABILITY,
        "rows": ROWS,
        "k": GRAPH_K,
        "spill": SPILL,
        "clusters": REGISTERED_SELECTED_CLUSTERS,
        "cell": REGISTERED_SELECTED_CELL,
        "inheritance": inheritance,
        "graph_under_test": GRAPH_UNDER_TEST,
        "degree_semantics_note": DEGREE_SEMANTICS_NOTE,
        "vacuity_note": VACUITY_NOTE,
        "hubness_note": HUBNESS_NOTE,
        "symmetrised_identity_note": SYMMETRISED_IDENTITY_NOTE,
        "symmetrised_degree_bound_note": SYMMETRISED_DEGREE_BOUND_NOTE,
        "r0215_tripwire": verdict,
        "in_degree_reconciliation": reconciliation,
        "reviewed_structural_scan": reviewed,
        "own_degree_pass": measured,
        "structural_cross_check": cross_check,
        "wall_guard": {
            "degree_pass": own_guard.receipt(),
            "reviewed_scan_projection": projection,
            "note": WALL_GUARD_NOTE,
        },
        "cuvs_inputs_asserted_memmap": True,
        "cuda_context_created": False,
        "child_processes_launched": 0,
        "signal_delivered": False,
        "abort_policy": (
            "in-band cooperative flag only; no SIGTERM, no SIGKILL, no ptrace, "
            "and no child process exists to signal"
        ),
        "safety_note": SAFETY_NOTE,
        "scope_note": SCOPE_NOTE,
        "no_symmetrisation_note": NO_SYMMETRISATION_NOTE,
        "gpu_hours_cap": GPU_HOURS_CAP,
        "gpu_hours_cap_note": GPU_HOURS_CAP_NOTE,
        "training_performed": False,
        "performance": {"total_wall_s": time.monotonic() - started},
    }))
    atomic_write_new_json(
        os.path.join(output, TRIPWIRE_FILE), receipt, immutable=True
    )
    if not verdict["holds"]:
        raise Round0241Error(
            "R0241 R0215 DEGREE-ZERO TRIPWIRE FAILED at 100,000,000 rows: "
            f"{verdict['symmetrised_isolated_rows']} symmetrised-isolated rows, "
            f"{verdict['out_degree_zero_rows']} with no usable out-edge, "
            f"{verdict['raw_graph_zero_degree_rows']} raw zero-degree rows. "
            "The evidence is sealed beside this failure and the graph must not "
            "be used. This is the v1 failure mode and it is reported, not "
            "explained away."
        )


# --------------------------------------------------------------------------- #
# node B — recall on the registered probe's rows, and only those
# --------------------------------------------------------------------------- #
class _ProbeRowView:
    """Present probe-only arrays under the registered global row ids.

    `_score_probe` reads its two bulk inputs exactly once each, as
    `array[probe_rows]`. This adapter lets that reviewed function be called
    unchanged on a `500,000`-row gather instead of a `100,000,000`-row one, and
    it is fail-closed: any index that is not the registered probe draw raises.
    So the adapter is also an assertion that the rows scored are the rows
    registered, which is the property review-0227-01 exists to protect.
    """

    def __init__(self, *, rows: np.ndarray, values: np.ndarray) -> None:
        self._rows = np.asarray(rows)
        self._values = np.asarray(values)
        if self._values.shape[0] != self._rows.shape[0]:
            raise Round0241Error("probe view rows and values disagree in length")

    @property
    def shape(self) -> tuple[int, ...]:
        return self._values.shape

    def __getitem__(self, index: Any) -> np.ndarray:
        candidate = np.asarray(index)
        if candidate.shape != self._rows.shape or not np.array_equal(
            candidate, self._rows
        ):
            raise Round0241Error(
                "R0241 probe view was indexed by something other than the "
                "registered uniform probe draw"
            )
        return self._values


def _probe_candidate_cosines(
    *,
    probe_rows: np.ndarray,
    ids: np.ndarray,
    host: np.ndarray,
    guard: StageGuard,
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute each probe row's 15 candidate cosines from the substrate.

    Independent of the builder's accumulator, which is the point
    (review-0216-01): the builder ran on the GPU through cuVS, this runs in CPU
    float32 numpy, and the two share nothing but the substrate bytes.

    This is R0238's `_candidate_cosines` restricted to the probe's rows and
    given the poll point it lacks. R0240's abort took `2 h 52 m` to reach a safe
    point precisely because that function has none, and the gather is the
    longest stage in the node.
    """
    count = int(probe_rows.size)
    out_cos = np.empty((count, GRAPH_K), dtype=np.float32)
    out_ids = np.empty((count, GRAPH_K), dtype=np.int32)
    for begin in range(0, count, PROBE_GATHER_BLOCK):
        end = min(begin + PROBE_GATHER_BLOCK, count)
        anchors_at = np.asarray(probe_rows[begin:end], dtype=np.int64)
        anchor = np.asarray(host[anchors_at], dtype=np.float32)
        neighbour_ids = np.asarray(ids[anchors_at], dtype=np.int32)
        out_ids[begin:end] = neighbour_ids
        flat = neighbour_ids.reshape(-1).astype(np.int64)
        order = np.argsort(flat, kind="stable")
        gathered = np.empty((flat.size, DIMENSION), dtype=np.float32)
        gathered[order] = host[flat[order]]
        out_cos[begin:end] = np.einsum(
            "bd,bkd->bk", anchor,
            gathered.reshape(end - begin, GRAPH_K, DIMENSION),
        )
        del anchor, neighbour_ids, flat, order, gathered
        guard.unit_done(f"probe gather rows [{begin}, {end})")
    if not np.isfinite(out_cos).all():
        raise Round0241Error("R0241 recomputed candidate cosines are not finite")
    return out_ids, out_cos


def _builder_agreement(
    *, cos_path: str, probe_rows: np.ndarray, recomputed: np.ndarray
) -> dict[str, Any]:
    """The builder's emitted cosines against the independent recompute.

    Never the basis of a recall number - every recall figure uses the
    recompute - but the comparison is free on the probe's rows and it says
    whether the two arithmetics agree at this rung.
    """
    builder = _readonly_memmap(
        cos_path, label="R0241 builder cosines", shape=(ROWS, GRAPH_K)
    )
    block = np.asarray(builder[np.asarray(probe_rows, dtype=np.int64)])
    difference = np.abs(
        block.astype(np.float64) - recomputed.astype(np.float64)
    )
    return {
        "population": "the registered probe's rows only",
        "rows_compared": int(probe_rows.size),
        "max_absolute_difference": float(difference.max()),
        "mean_absolute_difference": float(difference.mean()),
        "note": (
            "the builder ran on the GPU through cuVS; this recompute is CPU "
            "float32 numpy. Every recall number in this artifact uses the "
            "recompute"
        ),
    }


def _derivations(
    *, job: Mapping[str, Any], ladder: Mapping[str, Any]
) -> dict[str, Any]:
    """The three arithmetic debts, discharged from sealed receipts only."""
    selection = dict(ladder.get("cluster_selection") or {})
    candidates = {
        int(entry["clusters"]): entry
        for entry in (selection.get("candidates_considered") or [])
    }
    selected = int(selection["selected_clusters"])
    chosen = candidates[selected]

    own_by_seed = {
        int(cell["seed"]): float(cell["imbalance_max_over_mean"])
        for cell in (ladder.get("measured_imbalance") or {}).get("cells") or []
        if int(cell.get("clusters") or -1) == selected
        and int(cell.get("rows") or -1) == ROWS
    }
    if sorted(own_by_seed) != sorted(int(s) for s in IMBALANCE_REPLICATE_SEEDS):
        raise Round0241Error(
            f"R0240's sealed grid carries seeds {sorted(own_by_seed)} at "
            f"c = {selected}, registered {sorted(IMBALANCE_REPLICATE_SEEDS)}"
        )

    inherited_grid, inherited_grid_source = _inherited_replicate_grid(job)
    r0237_by_seed = {
        int(seed): float(value)
        for seed, value in (
            inherited_grid.get(50_000_000, {}).get(selected, {})
        ).items()
    }

    grid: dict[int, dict[int, dict[int, float]]] = {
        int(rows): {
            int(clusters): dict(seeds) for clusters, seeds in cells.items()
        }
        for rows, cells in inherited_grid.items()
    }
    grid.setdefault(ROWS, {})[selected] = dict(own_by_seed)
    sources_by_rows = {
        int(rows): (
            "measured in R0240, 5 seeds, sealed and hash-bound" if rows == ROWS
            else "measured in R0236/R0237, sealed and hash-bound"
        )
        for rows in grid
    }
    merged = replicate_grid_table(grid, sources_by_rows=sources_by_rows)
    merged["merged_from"] = inherited_grid_source

    inherited_law = dict(ladder["inherited_device_law"])
    setting = dict(ladder.get("nn_descent") or {})
    own_points: list[dict[str, Any]] = []
    for cell in ladder.get("ladder") or []:
        build = dict(cell.get("build") or {})
        largest = int((build.get("cluster_sizes") or {}).get("max") or 0)
        device_bytes = build.get("device_wide_peak_over_baseline_bytes")
        if not build.get("fit") or largest <= 0 or not device_bytes:
            continue
        own_points.append({
            "source": f"R0240 ladder {cell['cell']}",
            "cell": cell["cell"],
            "rows": ROWS,
            "clusters": int(cell["clusters"]),
            "spill": SPILL,
            "graph_degree": setting.get("graph_degree"),
            "intermediate_graph_degree": setting.get(
                "intermediate_graph_degree"
            ),
            "max_iterations": setting.get("max_iterations"),
            "max_cluster_rows": largest,
            "device_bytes": float(device_bytes),
        })
    laws = [fit_device_law(
        list(inherited_law["points"]) + own_points,
        label="all-sealed-gd64-igd256-it40-with-r0240",
    )]

    worst_now = {selected: float(max(own_by_seed.values()))}
    primary_now = {selected: float(own_by_seed[PRIMARY_IMBALANCE_SEED])}
    rungs: dict[str, Any] = {}
    for rung in PHASE2_RUNGS:
        with_margin = rung_derivation(
            rung=int(rung), imbalance_by_c=worst_now,
            imbalance_source=(
                f"worst of {len(IMBALANCE_REPLICATE_SEEDS)} replicate seeds "
                f"MEASURED at N = {ROWS:,}, s = {SPILL}, from R0240's sealed grid"
            ),
            laws=laws, apply_margin=True,
        )
        tolerances = {
            str(int(clusters)): imbalance_tolerance(
                rung=int(rung), clusters=int(clusters), imbalance=float(value),
                laws=laws,
            )
            for clusters, value in sorted(worst_now.items())
        }
        rungs[str(int(rung))] = {
            "with_margin": with_margin,
            "with_margin_primary_seed": rung_derivation(
                rung=int(rung), imbalance_by_c=primary_now,
                imbalance_source=(
                    f"primary seed {PRIMARY_IMBALANCE_SEED} measured at "
                    f"N = {ROWS:,}, s = {SPILL}"
                ),
                laws=laws, apply_margin=True,
            ),
            "point_estimate_no_margin": rung_derivation(
                rung=int(rung), imbalance_by_c=worst_now,
                imbalance_source=(
                    f"worst of {len(IMBALANCE_REPLICATE_SEEDS)} replicate seeds "
                    f"measured at N = {ROWS:,}, s = {SPILL}"
                ),
                laws=laws, apply_margin=False,
            ),
            "imbalance_tolerance_by_c": tolerances,
            "carry": carry_distance(measured_at_rows=ROWS, rung=int(rung)),
            "selected_tolerance": (
                tolerances.get(str(int(with_margin["selected_clusters"])))
                if with_margin.get("selected_clusters") is not None else None
            ),
        }

    return {
        "note": DEBTS_NOTE,
        "device_law": laws[0],
        "per_rung_derivation": rungs,
        "per_seed_movement_at_c400": per_seed_movement(
            r0240_by_seed=own_by_seed,
            r0237_by_seed=r0237_by_seed or None,
        ),
        "merged_imbalance_grid": merged,
        "tolerance_chain": tolerance_chain(
            r0240_admissible_max_cluster_rows=float(
                chosen["admissible_max_cluster_rows"]
            ),
            r0240_guarded_max_cluster_rows=float(
                chosen["guarded_max_cluster_rows"]
            ),
            r0240_measured_imbalance=float(chosen["measured_imbalance"]),
            imbalance_margin=(
                float(chosen["guarded_max_cluster_rows"])
                / float(chosen["measured_imbalance"])
                / float(chosen["mean_cluster_rows"])
            ),
            mean_cluster_rows=float(chosen["mean_cluster_rows"]),
        ),
    }


def run_recall(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    """Strict and tie-aware recall on the registered 500,000-row uniform probe."""
    manifest = active["manifest"]
    if str(manifest.get("round_id")) != ROUND_ID:
        raise Round0241Error("R0241 handler received another queue")
    started = time.monotonic()
    inheritance = verify_inheritance(job)
    ladder_path = str(inheritance["build_ladder"]["source"]["canonical_path"])
    ladder = prompt_contract.read_sealed(ladder_path, label="R0241 build ladder")
    output = create_fresh_directory(str(job["outputs"][0]), label="R0241 recall")

    truth_path = str(inheritance["truth"]["source"]["canonical_path"])
    truth = prompt_contract.read_sealed(truth_path, label="R0241 truth")
    rows_path = prompt_contract.verify_signature(
        dict(truth["outputs"]["query_rows"]), label="R0241 probe rows"
    )
    truth_ids_path = prompt_contract.verify_signature(
        dict(truth["outputs"]["ids"]), label="R0241 truth ids"
    )
    truth_cos_path = prompt_contract.verify_signature(
        dict(truth["outputs"]["cosines"]), label="R0241 truth cosines"
    )
    probe_rows = np.load(rows_path, allow_pickle=False)
    truth_ids = np.load(truth_ids_path, allow_pickle=False)
    truth_cos = np.load(truth_cos_path, allow_pickle=False)
    if truth_ids.shape != (TRUTH_PROBE_ROWS, GRAPH_K):
        raise Round0241Error("R0241 truth arrays have the wrong shape")
    registered_draw = truth_probe_query_rows(
        rows=ROWS, size=TRUTH_PROBE_ROWS, seed=TRUTH_PROBE_SEED
    )
    if not np.array_equal(probe_rows, registered_draw):
        raise Round0241Error(
            "R0241 sealed probe rows are not R0238's registered uniform draw at "
            f"seed {TRUTH_PROBE_SEED}. This round scores the ALREADY-REGISTERED "
            "probe and never a fresh one"
        )
    kth = truth_cos[:, GRAPH_K - 1].astype(np.float64)
    truth_best = truth_cos[:, 0].astype(np.float64)
    del truth_cos

    substrate_path = prompt_contract.verify_signature(
        dict(prompt_contract.read_sealed(
            str(inheritance["substrate"]["source"]["canonical_path"]),
            label="R0241 substrate manifest",
        )["substrate"]),
        label="R0241 substrate",
    )
    host = _readonly_memmap(
        substrate_path, label="R0241 substrate", shape=(ROWS, DIMENSION)
    )
    ids = _readonly_memmap(
        str(inheritance["graph"]["ids"]["canonical_path"]),
        label="R0241 graph ids", shape=(ROWS, GRAPH_K),
    )

    units = -(-int(probe_rows.size) // PROBE_GATHER_BLOCK)
    budget = float(job.get("stage_budget_s") or RECALL_DEADLINE_S)
    guard = StageGuard(
        label="probe gather", units_total=units, budget_s=budget,
        deadline_s=RECALL_DEADLINE_S, abort_check=_check_runner_abort,
    )
    probe_ids, probe_cos = _probe_candidate_cosines(
        probe_rows=probe_rows, ids=ids, host=host, guard=guard,
    )
    _check_runner_abort("R0241 scoring the probe")

    selected = _score_probe(
        ids=_ProbeRowView(rows=probe_rows, values=probe_ids),
        candidate_cos=_ProbeRowView(rows=probe_rows, values=probe_cos),
        probe_rows=probe_rows, truth_ids=truth_ids, kth=kth,
        truth_best=truth_best,
    )
    _check_runner_abort("R0241 scored the probe")
    verdict = recall_verdict(
        tie_aware=selected["tie_aware"], strict=selected["strict"]
    )
    power = probe_power_statement(
        mean=float(selected["tie_aware"]["mean"]),
        sd=_tie_aware_sd(probe_cos=probe_cos, probe_ids=probe_ids, kth=kth),
    )

    # ---- PERSIST THE SCIENCE THE MOMENT IT EXISTS ----------------------------
    # review-0240-01/F2: R0240 DID compute recall and then lost it, because the
    # node persisted nothing until after the last long stage and raised inside
    # the structural scan 44 lines later. "A single write before the structural
    # scan would have salvaged that round." This is that write, and it happens
    # before the builder-agreement pass and before any derivation can fail.
    early_path = os.path.join(output, RECALL_EARLY_FILE)
    atomic_write_new_json(early_path, prompt_contract.seal(json_safe({
        "schema": RECALL_EARLY_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": str(manifest["release_sha"]),
        "rows": ROWS,
        "k": GRAPH_K,
        "clusters": REGISTERED_SELECTED_CLUSTERS,
        "cell": REGISTERED_SELECTED_CELL,
        "probe_rows": TRUTH_PROBE_ROWS,
        "probe_seed": TRUTH_PROBE_SEED,
        "graph_ids_sha256": REGISTERED_GRAPH_IDS_SHA256,
        "recall": selected,
        "recall_verdict": verdict,
        "probe_power": power,
        "wall_guard": {"probe_gather": guard.receipt()},
        "why_this_file_exists": EARLY_PERSIST_NOTE,
        "is_complete": False,
        "completed_by": RECALL_FILE,
    })), immutable=True)

    agreement = _builder_agreement(
        cos_path=str(inheritance["graph"]["cosines"]["canonical_path"]),
        probe_rows=probe_rows, recomputed=probe_cos,
    )
    derivations = _derivations(job=job, ladder=ladder)

    receipt = prompt_contract.seal(json_safe({
        "schema": RECALL_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": str(manifest["release_sha"]),
        "capability": RECALL_CAPABILITY,
        "rows": ROWS,
        "k": GRAPH_K,
        "spill": SPILL,
        "clusters": REGISTERED_SELECTED_CLUSTERS,
        "cell": REGISTERED_SELECTED_CELL,
        "probe_rows": TRUTH_PROBE_ROWS,
        "probe_seed": TRUTH_PROBE_SEED,
        "probe_is_the_registered_r0238_draw": True,
        "inheritance": inheritance,
        "recall": selected,
        "recall_verdict": verdict,
        "probe_power": power,
        "builder_cosine_agreement": agreement,
        "reachability_ceiling_same_rows": REGISTERED_REACHABILITY_CEILING_C400,
        "zero_recall_forensic_note": ZERO_RECALL_FORENSIC_NOTE,
        "derivations": derivations,
        "wall_guard": {"probe_gather": guard.receipt(), "note": WALL_GUARD_NOTE},
        "recall_note": RECALL_NOTE,
        "recall_first_write": expected_input_signature(early_path),
        "early_persist_note": EARLY_PERSIST_NOTE,
        "cpu_arithmetic_note": CPU_ARITHMETIC_NOTE,
        "cuvs_inputs_asserted_memmap": True,
        "cuda_context_created": False,
        "child_processes_launched": 0,
        "signal_delivered": False,
        "abort_policy": (
            "in-band cooperative flag only; no SIGTERM, no SIGKILL, no ptrace, "
            "and no child process exists to signal"
        ),
        "safety_note": SAFETY_NOTE,
        "scope_note": SCOPE_NOTE,
        "no_symmetrisation_note": NO_SYMMETRISATION_NOTE,
        "fuzzy_weights_produced": False,
        "edges_produced": False,
        "gpu_hours_cap": GPU_HOURS_CAP,
        "gpu_hours_cap_note": GPU_HOURS_CAP_NOTE,
        "training_performed": False,
        "gate_registered": False,
        "adoption_claimed": False,
        "map_quality_claimed": False,
        "performance": {"total_wall_s": time.monotonic() - started},
    }))
    atomic_write_new_json(
        os.path.join(output, RECALL_FILE), receipt, immutable=True
    )
    if not verdict["holds"]:
        raise Round0241Error(
            "R0241 recall FAILED its registered floors: tie-aware mean "
            f"{verdict['tie_aware_mean']} against {verdict['mean_floor']}, p10 "
            f"{verdict['tie_aware_p10']} against {verdict['p10_floor']}. The "
            "evidence is sealed beside this failure."
        )


def _tie_aware_sd(
    *, probe_cos: np.ndarray, probe_ids: np.ndarray, kth: np.ndarray
) -> float:
    """Population sd of the per-row tie-aware score, for the standard error.

    Computed from the same reviewed `tie_aware_rows` the recall number uses, so
    the uncertainty describes the estimator that was actually reported.
    """
    from basemap.round0220_cuvs_qualification import tie_aware_rows

    scores = tie_aware_rows(
        probe_cos.astype(np.float64), probe_ids, np.asarray(kth, dtype=np.float64)
    )
    return float(np.std(scores, ddof=1))


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    action = str(job.get("action") or "")
    if action == TRIPWIRE_ACTION:
        run_tripwire(active, job)
    elif action == RECALL_ACTION:
        run_recall(active, job)
    else:
        raise Round0241Error(f"R{ROUND_ID} does not authorize action {action!r}")


__all__ = [
    "RECALL_ACTION",
    "TRIPWIRE_ACTION",
    "run_job",
    "run_recall",
    "run_tripwire",
    "verify_inheritance",
]
