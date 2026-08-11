"""Execute R0258 — the 100M graph load and edge preparation, on a CUDA-holding node.

Two nodes.

`controls_0258` (CPU)
    The guards this round ships, each with a positive control that plants the
    defect: five structural plants the chunk-loop guard must refuse and a
    substring predicate must accept; five numeric plants the bitwise comparison
    must refuse and an `np.allclose` comparison mostly accepts. Then the
    dispatch-derived install/gate census, re-run after this round wires an
    `AbortPollGate` into `round0238_nodes.run_assemble` and
    `round0113_nodes.run_train` — the two entries review-0254-01 §F named as the
    ones that will hold the card for hours and emit nothing.

`graphload_0258` (GPU)
    The measurement. A live CUDA context is created and held for the whole node,
    because a process that maps a CUDA library is the one a terminal cannot stop,
    and therefore the only one whose stop latency is worth measuring. Three arms
    over R0243's real `2,511,103,254`-edge fuzzy graph — `unpolled_control`
    (the shipped calls, no abort read anywhere), `polled_buffered` and
    `polled_o_direct` — five stages each, `n = 3` repetitions, round-robin with a
    rotating start (R0254's schedule, for R0254's reason: an experiment that runs
    all of arm A then all of arm B measures the device's history).

Nothing is trained. Measuring the load and the edge preparation does not require
completing a fit and a ~10 h train is not this round's business.

Safety: every bulk input is opened read-only; no process is signalled; no child
process is started; cuVS is handed nothing; no subprocess is wrapped in a
timeout. The host guard gates on ANONYMOUS bytes, never RSS.
"""
from __future__ import annotations

import gc
import json
import os
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.output_safety import create_fresh_directory
from basemap.round0238_rung5 import json_safe
from basemap.round0242_locality import host_anonymous_bytes, io_counters, json_scrub
from basemap.round0245_guard import EnforcedHostWatchdog
from basemap.round0251_trainer_setup import PollRecorder
from basemap.round0252_stoppability import gap_report
from basemap.round0253_coverage import CoverageLedger, coverage_summary
from basemap.round0253_stop_hooks import (
    NOT_A_FAMILY_CELL,
    THE_INSTRUMENT_IS_DEFEATABLE,
    install_stop_hooks,
    over_the_ceiling,
    registered_ceiling_s,
)
from basemap.round0254_dispatch import (
    SCOPE_MODULES,
    assert_derived_entries_install,
    derived_entries,
    dispatch_census,
    entry_install_audit,
    entry_tuples,
    gate_census,
)
from basemap import round0258_graph_load as graph_load
from basemap.round0258_graph_load import (
    ARMS,
    ARM_POLLED_BUFFERED,
    ARM_POLLED_O_DIRECT,
    ARM_UNPOLLED_CONTROL,
    DIRECTED_EDGES,
    EDGE_ARRAYS,
    EDGE_HEADER_PATH,
    K,
    READ_CHUNK_BYTES,
    ROWS,
    Round0258GraphLoadError,
    SCAN_CHUNK_ELEMENTS,
    STAGES,
    THE_RESIDUE_THAT_CANNOT_BE_CHUNKED,
    WHAT_THE_UNPOLLED_ARM_IS,
    arm_schedule,
    assert_bitwise_identity_controls,
    assert_chunk_loops_poll,
    assert_structural_defect_controls,
    bitwise_identical,
    interval_summary,
)
from experiments.round0253_nodes import _cuda_maps_evidence, force_remove_tree
from experiments.round0251_nodes import (
    _guard_tail_reported,
    _node_gate,
    _receipt_envelope as _r0251_envelope,
    _score_gate_without_raising,
    _seal,
    _start_node,
)


ROUND_ID = "0258"

CONTROLS_ACTION = "round0258_graph_load_controls"
GRAPHLOAD_ACTION = "round0258_100m_graph_load_and_edge_prep"

CONTROLS_CAPABILITY = "round0258-graph-load-poll-controls-v1"
GRAPHLOAD_CAPABILITY = "round0258-100m-graph-load-and-edge-prep-intervals-v1"

CONTROLS_SCHEMA = "round0258-graph-load-poll-controls-v1"
GRAPHLOAD_SCHEMA = "round0258-100m-graph-load-and-edge-prep-v1"

REPETITIONS = 3

#: The anonymous budget this node declares. Arithmetic, not a round number:
#: the three edge arrays resident are `3 x 10,044,413,016 = 30,133,239,048` B;
#: the largest transient beside them is R0107's boolean temporary at
#: `2,511,103,254` B, giving a load-and-scan peak of `32,644,342,302` B
#: (`30.40` GiB). The CDF stage frees the two endpoint arrays first and peaks at
#: `10,044,413,016 + 20,088,826,032 = 30,133,239,048` B. `40` GiB is above both
#: and below the registered `max_declared_anonymous_budget_bytes` of `60` GiB.
NODE_ANON_BUDGET_BYTES = 40 * (1 << 30)

#: Refuse to start an arm without this much MemAvailable. The measured peak is
#: `~30.4` GiB; `56` GiB is the peak plus the whole 30 GB graph's page cache
#: plus headroom, so an arm that would have driven this box toward swap does not
#: start at all. Swap growth is what wedged R0224.
MIN_MEM_AVAILABLE_BYTES = 56 * (1 << 30)

SAFETY_NOTE = (
    "no node in this module signals any process, starts a child process, hands "
    "cuVS anything, or wraps a subprocess in a timeout. Every bulk input is "
    "opened read-only -- np.load without mmap_mode materialises a read-only "
    "copy of immutable bytes and never writes back, and the memmap stages use "
    "R0230's reader rule (isinstance np.memmap and not writeable, asserted on "
    "the object). The graph-load node holds a live CUDA context for its whole "
    "wall on purpose: /proc/<pid>/maps matches libcuda, so this process must "
    "never be signalled, which is exactly the case whose stop latency matters."
)

WHY_THIS_IS_THE_REAL_ARTIFACT = (
    "R0253 D3 recorded that R0238's 100,000,000-row substrate and R0240's k15 "
    "graph 'no longer exist on this box, and rebuilding them is ~7 GPU-h'. They "
    "do exist, fully allocated, at their sealed signatures, and have since they "
    "were built -- so this round measures the real bytes and spends no GPU-h "
    "rebuilding anything. The three edge arrays are re-verified here against "
    "R0243's sealed sha256 values rather than trusted from their size, because "
    "the runner's input preflight checks size only."
)


class Round0258NodeError(RuntimeError):
    """R0258 fails closed."""


def _receipt_envelope(manifest: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(_r0251_envelope(manifest))
    body["round_id"] = ROUND_ID
    return body


def _node_guard(label: str, *, interval_s: float = 0.05) -> EnforcedHostWatchdog:
    """This round's own guard: the R0251 helper declares a 16 GiB budget, which
    a node that materialises 30 GB of edges would trip on its first stage."""
    return EnforcedHostWatchdog(
        anonymous_budget_bytes=NODE_ANON_BUDGET_BYTES,
        interval_s=float(interval_s),
        label=label,
    )


def _mem_available_bytes() -> int:
    with open("/proc/meminfo", "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise Round0258NodeError("R0258 could not read MemAvailable")


def _require_headroom(*, label: str) -> dict[str, Any]:
    available = _mem_available_bytes()
    if available < MIN_MEM_AVAILABLE_BYTES:
        raise Round0258NodeError(
            f"R0258 refuses to start {label}: MemAvailable {available} B is below "
            f"the required {MIN_MEM_AVAILABLE_BYTES} B. An arm that would drive "
            "this box toward swap does not start."
        )
    return {"mem_available_bytes": available,
            "required_bytes": MIN_MEM_AVAILABLE_BYTES}


def _verify_edge_arrays(*, poll: Any) -> dict[str, Any]:
    """Re-verify the three edge arrays against R0243's sealed sha256.

    The hash is chunk-polled through the release's own `sha256_file`, which
    R0252 made pollable -- so proving the artifact's identity is itself not an
    unpollable interval. `expected_input_signature` is not used because it also
    stats and would conflate two things.
    """
    from basemap import artifact_identity

    previous = artifact_identity.set_abort_poll(poll)
    try:
        verified: dict[str, Any] = {}
        for name, spec in sorted(EDGE_ARRAYS.items()):
            started = time.monotonic()
            digest = artifact_identity.sha256_file(spec["path"])
            wall = time.monotonic() - started
            size = os.path.getsize(spec["path"])
            verified[name] = {
                "canonical_path": spec["path"],
                "bytes": size,
                "sha256": digest,
                "matches_the_r0243_seal": digest == spec["sha256"],
                "size_matches_the_r0243_seal": size == spec["bytes"],
                "hash_wall_s": wall,
                "hash_wall_over_the_ceiling": over_the_ceiling(wall),
            }
            poll(f"R0258 {name} verified against the R0243 seal")
    finally:
        artifact_identity.set_abort_poll(previous)
    unmatched = sorted(n for n, v in verified.items()
                       if not (v["matches_the_r0243_seal"]
                               and v["size_matches_the_r0243_seal"]))
    if unmatched:
        raise Round0258NodeError(
            f"R0258 refuses to measure a graph that is not R0243's: {unmatched}"
        )
    header = np.load(EDGE_HEADER_PATH, allow_pickle=False)
    declared = {key: int(np.asarray(header[key])) for key in sorted(header.files)}
    if (declared.get("n_nodes") != ROWS or declared.get("k") != K
            or declared.get("directed_edges") != DIRECTED_EDGES):
        raise Round0258NodeError(
            f"R0258 header disagrees with the registered rung: {declared}"
        )
    return {
        "schema": "round0258-edge-array-identity-v1",
        "arrays": verified,
        "header": declared,
        "why": WHY_THIS_IS_THE_REAL_ARTIFACT,
    }


# --------------------------------------------------------------------------- #
# the five stages, per arm
# --------------------------------------------------------------------------- #


def _timed(function: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    started = time.monotonic()
    value = function(*args, **kwargs)
    return value, time.monotonic() - started


def _one_arm(*, arm: str, repetition: int, poll: Any) -> dict[str, Any]:
    """One repetition of one arm: the five stages, timed individually.

    review-0254-01 §L.7's rule is the reason every stage below is timed
    DIRECTLY, beside the poll-gap series rather than inside it: *a poll-gap is
    not a syscall duration*. R0253 attributed a 2 GiB `fsync` to a 64 MiB block
    write because it published gaps and named them by the site that closed them.
    Here the gap series and the per-stage walls are two independent records of
    the same node and are published side by side.
    """
    if arm not in ARMS:
        raise Round0258NodeError(f"R0258 does not know arm {arm!r}")
    polled = arm != ARM_UNPOLLED_CONTROL
    o_direct = arm == ARM_POLLED_O_DIRECT
    stage_poll = poll if polled else None

    eviction: dict[str, float] = {}
    for name, spec in sorted(EDGE_ARRAYS.items()):
        eviction[name] = graph_load.evict_page_cache(spec["path"])
    poll(f"R0258 {arm} rep {repetition} page cache evicted")

    anonymous: dict[str, int] = {"before": host_anonymous_bytes()["anonymous_bytes"]}
    stages: dict[str, Any] = {}

    # ---- S1: load ---------------------------------------------------------- #
    loaded: dict[str, np.ndarray] = {}
    per_array_load: dict[str, float] = {}
    for name, spec in sorted(EDGE_ARRAYS.items()):
        if polled:
            array, wall = _timed(
                graph_load.load_array_polled, spec["path"],
                poll=stage_poll, chunk_bytes=READ_CHUNK_BYTES, o_direct=o_direct,
            )
        else:
            array, wall = _timed(graph_load.load_array_unpolled, spec["path"])
        if array.shape != (DIRECTED_EDGES,):
            raise Round0258NodeError(
                f"R0258 {name} loaded shape {array.shape}, expected "
                f"({DIRECTED_EDGES},)"
            )
        loaded[name] = array
        per_array_load[name] = wall
    stages[graph_load.STAGE_LOAD] = {
        "wall_s": float(sum(per_array_load.values())),
        "per_array_wall_s": dict(sorted(per_array_load.items())),
        "widest_single_array_s": max(per_array_load.values()),
        "bytes": sum(spec["bytes"] for spec in EDGE_ARRAYS.values()),
    }
    anonymous["after_load"] = host_anonymous_bytes()["anonymous_bytes"]
    poll(f"R0258 {arm} rep {repetition} load complete")

    # ---- S2: endpoint bounds ---------------------------------------------- #
    bounds: dict[str, Any] = {}
    bounds_walls: dict[str, float] = {}
    for name in ("sources", "targets"):
        if polled:
            value, wall = _timed(
                graph_load.endpoint_bounds_polled, loaded[name],
                poll=stage_poll, chunk=SCAN_CHUNK_ELEMENTS,
            )
        else:
            value, wall = _timed(graph_load.endpoint_bounds_unpolled, loaded[name])
        bounds[name] = {"min": int(value[0]), "max": int(value[1])}
        bounds_walls[name] = wall
    stages[graph_load.STAGE_BOUNDS] = {
        "wall_s": float(sum(bounds_walls.values())),
        "per_array_wall_s": dict(sorted(bounds_walls.items())),
        "widest_single_array_s": max(bounds_walls.values()),
        "observed": bounds,
    }
    poll(f"R0258 {arm} rep {repetition} bounds complete")

    # ---- S3: weight validity ---------------------------------------------- #
    if polled:
        validity, validity_wall = _timed(
            graph_load.weight_validity_polled, loaded["weights"],
            poll=stage_poll, chunk=SCAN_CHUNK_ELEMENTS,
        )
    else:
        validity, validity_wall = _timed(
            graph_load.weight_validity_unpolled, loaded["weights"]
        )
    stages[graph_load.STAGE_VALIDITY] = {
        "wall_s": validity_wall,
        "all_finite": bool(validity[0]),
        "any_non_positive": bool(validity[1]),
        "any_above_one": bool(validity[2]),
        "boolean_temporary_bytes": (
            SCAN_CHUNK_ELEMENTS if polled else DIRECTED_EDGES
        ),
    }
    anonymous["after_validity"] = host_anonymous_bytes()["anonymous_bytes"]
    poll(f"R0258 {arm} rep {repetition} validity complete")

    # ---- S4: contiguous int32 copies --------------------------------------- #
    # Two cases, and they differ by a factor the accounting must not hide.
    # From an already-contiguous int32 ndarray `np.ascontiguousarray` returns the
    # SAME OBJECT and costs nothing; from a read-only memmap it copies 10 GB.
    # A 100M loader that hands the sampler a memmap -- which is the only loader
    # that fits R0244's 1.97 GB anonymous peak -- pays the second.
    from_array, from_array_wall = (
        _timed(graph_load.contiguous_int32_polled, loaded["sources"],
               poll=stage_poll, chunk=SCAN_CHUNK_ELEMENTS)
        if polled else
        _timed(graph_load.contiguous_int32_unpolled, loaded["sources"])
    )
    copied_from_array = from_array is not loaded["sources"]
    del from_array
    gc.collect()

    memmapped = graph_load.open_readonly_memmap(
        EDGE_ARRAYS["sources"]["path"], label="sources"
    )
    from_memmap, from_memmap_wall = (
        _timed(graph_load.contiguous_int32_polled, memmapped,
               poll=stage_poll, chunk=SCAN_CHUNK_ELEMENTS)
        if polled else
        _timed(graph_load.contiguous_int32_unpolled, memmapped)
    )
    memmap_copy_matches = bitwise_identical(from_memmap, loaded["sources"])
    del from_memmap, memmapped
    gc.collect()
    stages[graph_load.STAGE_CONTIGUOUS] = {
        "wall_s": float(from_array_wall + from_memmap_wall),
        "from_a_resident_int32_array_wall_s": from_array_wall,
        "from_a_resident_int32_array_copied": copied_from_array,
        "from_a_read_only_memmap_wall_s": from_memmap_wall,
        "the_memmap_copy_is_bitwise_identical_to_the_loaded_array": (
            memmap_copy_matches
        ),
        "widest_single_call_s": max(from_array_wall, from_memmap_wall),
    }
    poll(f"R0258 {arm} rep {repetition} contiguous copies complete")

    # ---- S5: the float64 sampling CDF -------------------------------------- #
    # The endpoint arrays are released first: the CDF is `20,088,826,032` B and
    # holding all four would put this node's anonymous peak above its declared
    # budget. The stages are therefore measured with the arrays each stage needs
    # resident, and the combined residency a single-process trainer would hold
    # is reported as ARITHMETIC below, never as a measurement.
    weights = loaded.pop("weights")
    loaded.clear()
    gc.collect()
    anonymous["before_cdf"] = host_anonymous_bytes()["anonymous_bytes"]
    if polled:
        (cdf, total, residue_s), cdf_wall = _timed(
            graph_load.weight_cdf_polled, weights,
            poll=stage_poll, chunk=SCAN_CHUNK_ELEMENTS,
        )
    else:
        (cdf, total), cdf_wall = _timed(graph_load.weight_cdf_unpolled, weights)
        residue_s = None
    anonymous["after_cdf"] = host_anonymous_bytes()["anonymous_bytes"]
    cdf_tail = float(cdf[-1])
    cdf_head = float(cdf[0])
    cdf_bytes = int(cdf.nbytes)
    cdf_sha_input = np.asarray(cdf[:: max(1, DIRECTED_EDGES // 4096)])
    cdf_probe = float(cdf_sha_input.sum())
    del cdf, weights
    gc.collect()
    stages[graph_load.STAGE_CDF] = {
        "wall_s": cdf_wall,
        "float64_bytes": cdf_bytes,
        "weight_total": total,
        "cdf_first": cdf_head,
        "cdf_last": cdf_tail,
        "cdf_probe_sum_over_4096_samples": cdf_probe,
        "unpollable_np_sum_s": residue_s,
        "unpollable_np_sum_over_the_ceiling": (
            over_the_ceiling(residue_s) if residue_s is not None else None
        ),
        "the_residue_that_cannot_be_chunked": THE_RESIDUE_THAT_CANNOT_BE_CHUNKED,
    }
    anonymous["after"] = host_anonymous_bytes()["anonymous_bytes"]
    poll(f"R0258 {arm} rep {repetition} CDF complete")

    walls = {name: float(body["wall_s"]) for name, body in stages.items()}
    widest_single_call = max(
        [stages[graph_load.STAGE_LOAD]["widest_single_array_s"],
         stages[graph_load.STAGE_BOUNDS]["widest_single_array_s"],
         stages[graph_load.STAGE_VALIDITY]["wall_s"],
         stages[graph_load.STAGE_CONTIGUOUS]["widest_single_call_s"],
         stages[graph_load.STAGE_CDF]["wall_s"]]
    )
    return {
        "arm": arm,
        "repetition": repetition,
        "polled": polled,
        "o_direct": o_direct,
        "page_cache_eviction_s": dict(sorted(eviction.items())),
        "page_cache_eviction_total_s": float(sum(eviction.values())),
        "stages": stages,
        "stage_wall_s": walls,
        "total_stage_wall_s": float(sum(walls.values())),
        "widest_single_shipped_call_s": widest_single_call,
        "widest_single_shipped_call_over_the_ceiling": over_the_ceiling(
            widest_single_call
        ),
        "host_anonymous_bytes_by_stage": dict(sorted(anonymous.items())),
        "peak_host_anonymous_bytes_observed": max(anonymous.values()),
    }


# --------------------------------------------------------------------------- #
# node A -- the controls and the install/gate census
# --------------------------------------------------------------------------- #


def run_controls(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0258 round0258_nodes.run_controls")
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0258NodeError("R0258 controls handler received another queue")
    node_id = str(active.get("node_id") or "controls_0258")
    ledger = CoverageLedger(node=node_id)
    label = "R0258 graph-load poll controls"
    abort_flag = _start_node(label)
    output = create_fresh_directory(str(job["outputs"][0]), label="R0258 controls")

    window = ledger.window("R0258 controls stage")
    guard = _node_guard(label)
    gate = _node_gate(label, training_performed=False)
    with guard:
        gate.start()
        recorder = PollRecorder(gate=gate, clock=time.monotonic)
        recorder.anchor("R0258 controls stage entered")
        wrapped = window.wrap(recorder)

        structural = assert_structural_defect_controls()
        wrapped("R0258 structural plants refused")
        numeric = assert_bitwise_identity_controls()
        wrapped("R0258 numeric plants refused")
        chunk_audit = assert_chunk_loops_poll()
        wrapped("R0258 chunk-loop audit complete")

        census = dispatch_census()
        derived = derived_entries(SCOPE_MODULES, census)
        entries = entry_tuples(derived)
        installs = entry_install_audit(entries)
        install_guard = assert_derived_entries_install(SCOPE_MODULES, census)
        gates = gate_census(entries)
        wrapped("R0258 dispatch install and gate census complete")
        gate.finish("R0258 controls stage end")
    window.close()
    tail = _guard_tail_reported(guard, label=label)
    scored = _score_gate_without_raising(gate, tail, label=label)
    report = gap_report(recorder.records, arm="controls")

    both = sorted(
        row["entry"] for row in gates["entries"]
        if row["constructs_a_gate"] and row["installs_effectively"]
    )
    coverage = ledger.receipt()
    body = dict(_receipt_envelope(active["manifest"]))
    body.update({
        "schema": CONTROLS_SCHEMA,
        "capability": CONTROLS_CAPABILITY,
        "capabilities": [CONTROLS_CAPABILITY],
        "abort_flag_precondition": abort_flag,
        "node": node_id,
        "gate_registered": False,
        "is_a_family_cell": False,
        "not_a_family_cell": NOT_A_FAMILY_CELL,
        "the_instrument_is_defeatable": THE_INSTRUMENT_IS_DEFEATABLE,
        "safety_note": SAFETY_NOTE,
        "registered_ceiling_s": registered_ceiling_s(),
        "structural_defect_controls": structural,
        "numeric_defect_controls": numeric,
        "chunk_loop_audit": chunk_audit,
        "install_and_gate": {
            "install_guard": install_guard,
            "entry_count": int(gates["entries_audited"]),
            "effective_installs": int(
                installs["entries_with_an_effective_install"]
            ),
            "every_entry_installs_effectively": bool(
                installs["every_entry_installs_effectively"]
            ),
            "entries_that_construct_a_gate": int(
                gates["entries_that_construct_a_gate"]
            ),
            "entries_that_both_install_and_gate": int(
                gates["entries_that_both_install_and_gate"]
            ),
            "the_entries_that_both_install_and_gate": both,
            "entries_that_install_but_construct_no_gate": sorted(
                gates["entries_that_install_but_construct_no_gate"]
            ),
            "scope_modules": list(SCOPE_MODULES),
            "scope_note": (
                "the derived entry list is every module-level run_* attribute "
                "reachable from run_job in the scope modules. review-0254-01 "
                "D.3 blocked the claim that this bounds a queue: nothing "
                "compares a queue's handler_module set to SCOPE_MODULES. That "
                "remains open and this round does not claim otherwise."
            ),
        },
        "training_performed": False,
        "cuvs_calls": 0,
        "child_processes_launched": 0,
        "signal_delivered": False,
        "gap_report": report,
        "enforcement_poll_spacing": scored,
        "guard_tail": tail,
        "poll_coverage": coverage,
        "observed_span_s": coverage["observed_span_s"],
        "node_wall_s": coverage["node_wall_s"],
    })
    _seal(output, f"{node_id}-graph-load-controls.json", json_scrub(json_safe(body)))


# --------------------------------------------------------------------------- #
# node B -- the 100M measurement, on a node holding a live CUDA context
# --------------------------------------------------------------------------- #


def run_graph_load(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0258 round0258_nodes.run_graph_load")
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0258NodeError("R0258 graph-load handler received another queue")
    node_id = str(active.get("node_id") or "graphload_0258")
    ledger = CoverageLedger(node=node_id)
    label = "R0258 100M graph load and edge preparation"
    abort_flag = _start_node(label)
    output = create_fresh_directory(str(job["outputs"][0]), label="R0258 graph load")
    node_started = time.monotonic()

    import torch

    if not torch.cuda.is_available():
        raise Round0258NodeError("R0258 graph-load node requires a live CUDA device")
    resident = torch.zeros(1024, 1024, device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()
    cuda_context = {
        "device": torch.cuda.get_device_name(0),
        "resident_tensor_bytes": int(resident.numel() * resident.element_size()),
        "torch_cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated()),
        "maps": _cuda_maps_evidence(),
        "held_for": (
            "the whole node. The context is created before the first read and "
            "released after the last, so every interval below is measured in the "
            "state a training node is in -- the state that cannot be signalled."
        ),
    }
    if not cuda_context["maps"]["this_process_maps_a_cuda_library"]:
        raise Round0258NodeError(
            "R0258 graph-load node holds no CUDA mapping; the measurement would "
            "not be the case it claims to be"
        )

    io_before = io_counters()
    headroom = _require_headroom(label=label)
    schedule = arm_schedule(ARMS, REPETITIONS)
    runs: list[dict[str, Any]] = []
    identity: dict[str, Any] | None = None
    try:
        for index, (repetition, arm) in enumerate(schedule):
            _require_headroom(label=f"{label} {arm} rep {repetition}")
            window = ledger.window(f"R0258 {arm} rep {repetition}")
            guard = _node_guard(label)
            gate = _node_gate(label, training_performed=False)
            with guard:
                gate.start()
                recorder = PollRecorder(gate=gate, clock=time.monotonic)
                recorder.anchor(f"R0258 {arm} rep {repetition} entered")
                wrapped = window.wrap(recorder)
                if identity is None:
                    identity = _verify_edge_arrays(poll=wrapped)
                    wrapped("R0258 edge arrays verified against the R0243 seal")
                run = _one_arm(arm=arm, repetition=repetition, poll=wrapped)
                gate.finish(f"R0258 {arm} rep {repetition} end")
            window.close()
            tail = _guard_tail_reported(guard, label=label)
            run["enforcement_poll_spacing"] = _score_gate_without_raising(
                gate, tail, label=label
            )
            run["guard_tail"] = tail
            run["gap_report"] = gap_report(recorder.records, arm=arm)
            run["schedule_index"] = index
            runs.append(run)
            gc.collect()
    finally:
        del resident
        torch.cuda.empty_cache()
    io_after = io_counters()
    node_wall = time.monotonic() - node_started

    verdict = arm_verdict(runs)
    coverage = ledger.receipt(node_wall_s=node_wall)
    body = dict(_receipt_envelope(active["manifest"]))
    body.update({
        "schema": GRAPHLOAD_SCHEMA,
        "capability": GRAPHLOAD_CAPABILITY,
        "capabilities": [GRAPHLOAD_CAPABILITY],
        "abort_flag_precondition": abort_flag,
        "node": node_id,
        "gate_registered": False,
        "is_a_family_cell": False,
        "not_a_family_cell": NOT_A_FAMILY_CELL,
        "the_instrument_is_defeatable": THE_INSTRUMENT_IS_DEFEATABLE,
        "safety_note": SAFETY_NOTE,
        "registered_ceiling_s": registered_ceiling_s(),
        "cuda_context": cuda_context,
        "rows": ROWS,
        "k": K,
        "directed_edges": DIRECTED_EDGES,
        "edge_array_identity": identity,
        "read_chunk_bytes": READ_CHUNK_BYTES,
        "scan_chunk_elements": SCAN_CHUNK_ELEMENTS,
        "repetitions": REPETITIONS,
        "schedule": [list(pair) for pair in schedule],
        "what_the_unpolled_arm_is": WHAT_THE_UNPOLLED_ARM_IS,
        "mem_available_at_start": headroom,
        "declared_anonymous_budget_bytes": NODE_ANON_BUDGET_BYTES,
        # Sealed BY ARM, not as a flat indexed list. `rounds/report.py::_is_control`
        # classifies on the flattened dotted KEY PATH, so a control run sealed at
        # `runs[0].gap_report.widest_gap_s` carries no control token and the census
        # ranks it as SHIPPED -- which is exactly what the first attempt of this
        # node did, putting the unpolled control's 34.517681 s at the head of the
        # shipped table. Keying by arm puts `unpolled_control` in the path of every
        # figure the control produces.
        "runs_by_arm": {
            arm: [run for run in runs if run["arm"] == arm]
            for arm in sorted({run["arm"] for run in runs})
        },
        "why_the_runs_are_keyed_by_arm": (
            "rounds/report.py::_is_control matches on the flattened dotted key "
            "path, not on the arm value. A flat `runs[N]` list hides the arm from "
            "the path and makes the census rank a registered control as a shipped "
            "arm. This is the R0252 mistake and the first attempt of this node "
            "reproduced it; the run is preserved and reported."
        ),
        "verdict": verdict,
        "combined_residency_is_arithmetic_not_a_measurement": {
            "three_edge_arrays_bytes": 3 * (DIRECTED_EDGES * 4),
            "float64_cdf_bytes": DIRECTED_EDGES * 8,
            "sum_bytes": 3 * (DIRECTED_EDGES * 4) + DIRECTED_EDGES * 8,
            "note": (
                "a single-process trainer that holds the endpoint arrays AND the "
                "sampling CDF at once needs this many anonymous bytes. This node "
                "does not hold them at once -- it frees the endpoint arrays "
                "before the CDF stage so its own peak stays inside its declared "
                "budget -- so this line is ARITHMETIC over measured array sizes "
                "and is not a measurement of anything this node did."
            ),
        },
        "io_delta": {
            key: int(io_after.get(key, 0) - io_before.get(key, 0))
            for key in sorted(set(io_before) | set(io_after))
        },
        "training_performed": False,
        "cuvs_calls": 0,
        "child_processes_launched": 0,
        "signal_delivered": False,
        "poll_coverage": coverage,
        "observed_span_s": coverage["observed_span_s"],
        "node_wall_s": coverage["node_wall_s"],
        "coverage_summary": coverage_summary([coverage]),
    })
    _seal(output, f"{node_id}-100m-graph-load.json", json_scrub(json_safe(body)))


def arm_verdict(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold the runs into a per-arm ladder.

    review-0254-01 §B.1 is why `worst_single_interval_s` is a max over EVERY
    timed interval class rather than over a chosen subset: R0254's own statistic
    omitted its in-loop flushes, which were the largest intervals in three of its
    six arms, and that single omission made its headline, its best arm and its
    improvement factor wrong. Here the fold is over all five stage walls and both
    per-array sub-walls, and it is cross-checked against the census's own
    `gap_report.widest_gap_s` for the same run, which sees the same intervals by
    a different route.
    """
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_arm.setdefault(run["arm"], []).append(run)
    ceiling = registered_ceiling_s()
    entries: dict[str, Any] = {}
    for arm, arm_runs in sorted(by_arm.items()):
        intervals: list[float] = []
        for run in arm_runs:
            stages = run["stages"]
            intervals.extend(stages[graph_load.STAGE_LOAD]["per_array_wall_s"].values())
            intervals.extend(
                stages[graph_load.STAGE_BOUNDS]["per_array_wall_s"].values()
            )
            intervals.append(stages[graph_load.STAGE_VALIDITY]["wall_s"])
            intervals.append(
                stages[graph_load.STAGE_CONTIGUOUS][
                    "from_a_resident_int32_array_wall_s"]
            )
            intervals.append(
                stages[graph_load.STAGE_CONTIGUOUS]["from_a_read_only_memmap_wall_s"]
            )
            intervals.append(stages[graph_load.STAGE_CDF]["wall_s"])
        worst = max(intervals)
        census_gaps = [float(run["gap_report"]["widest_gap_s"]) for run in arm_runs]
        entries[arm] = {
            "repetitions": len(arm_runs),
            "worst_single_interval_s": worst,
            "worst_single_interval_over_the_ceiling": worst / ceiling,
            "interval_summary": interval_summary(intervals, ceiling_s=ceiling),
            "census_widest_gap_s": max(census_gaps),
            "census_widest_gap_over_the_ceiling": max(census_gaps) / ceiling,
            "per_repetition_total_stage_wall_s": [
                float(run["total_stage_wall_s"]) for run in arm_runs
            ],
            "every_repetition_under_the_ceiling": all(
                value <= ceiling for value in intervals
            ),
        }
    shipped = {
        arm: entry for arm, entry in entries.items() if arm != ARM_UNPOLLED_CONTROL
    }
    best = min(shipped, key=lambda a: shipped[a]["worst_single_interval_s"]) if shipped else None
    control = entries.get(ARM_UNPOLLED_CONTROL)
    return {
        "schema": "round0258-graph-load-arm-verdict-v1",
        "registered_ceiling_s": ceiling,
        "by_arm": entries,
        "best_shipped_arm": best,
        "best_shipped_worst_interval_s": (
            shipped[best]["worst_single_interval_s"] if best else None
        ),
        "best_shipped_worst_interval_over_the_ceiling": (
            shipped[best]["worst_single_interval_over_the_ceiling"] if best else None
        ),
        "improvement_over_the_unpolled_control": (
            control["worst_single_interval_s"] / shipped[best]["worst_single_interval_s"]
            if best and control else None
        ),
        "arms_whose_every_interval_stayed_under_the_ceiling": sorted(
            arm for arm, entry in entries.items()
            if entry["every_repetition_under_the_ceiling"]
        ),
    }


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0258 round0258_nodes.run_job")
    action = str(job.get("action") or "")
    if action == CONTROLS_ACTION:
        run_controls(active, job)
    elif action == GRAPHLOAD_ACTION:
        run_graph_load(active, job)
    else:
        raise Round0258NodeError(f"R0258 does not know action {action!r}")
