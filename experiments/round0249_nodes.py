"""Execute R0249 — two real bugs, a discovered scope, and a bound that throttles.

Two CPU-shaped nodes. Neither trains anything, builds anything, reads a bulk
array, or creates a CUDA context:

* `guardfix_0249` — the two in-process defects review-0248-01 §B named as real,
  each with a control in the reviewer's shape: the four disclosure attributes
  are read-only (the reviewer's two assignments now raise), and `replay`'s
  false `enforcement="clamped"` is corrected to `declared`, so the weakening
  record reaches the sealing gate and refuses a replayed verdict **without
  reading any disclosure attribute**. Plus the non-vacuity control that shows
  `require()` still discriminates between replays, and the inventory whose
  module scope is now DISCOVERED rather than hand-written.
* `external_0249` — the bound moved from `memory.max` to `memory.high`. A plain
  CPU allocator asks for `8x` its `memory.high` and must **survive**, with the
  kernel's `memory.events` `high` counter as the evidence and its own
  allocation rate above and below the limit as the measure of the back-pressure.
  The identical allocator under R0248's `memory.max` is killed, once, for the
  contrast. Then the escape battery runs in `root-scope` (all five refused) and
  in `user-scope` (three succeed), and the fail-closed control shows an
  unplaceable mode refusing rather than silently becoming a weaker one.

**Never** hand a `memory.max` to a process holding a CUDA context: a kernel OOM
kill of a CUDA holder is the wedge this box has been rebooted for twice. That is
the whole reason the production limit is `memory.high` now. Both nodes run with
no GPU work, and every allocator imports no array library at all.

Every registered check is IMPORTED, never re-typed. The `external_0249` node
starts child processes deliberately — that is the mechanism under test — and no
node in this module delivers a signal on any path.
"""
from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any

from basemap import round0113_prompt_contrast as prompt_contract
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0238_rung5 import GRAPH_K, json_safe
from basemap.round0241_qualify import GPU_HOURS_CAP_NOTE
from basemap.round0242_locality import json_scrub
from basemap.round0245_guard import (
    EnforcedHostWatchdog,
    require_enforceable_abort_flag,
)
from basemap.round0246_guard import (
    AbortPollGate,
    measured_slope_from_trace,
    require_abort_flag_landed,
    require_enforcement_evidence,
    require_live_sampler,
)
from basemap.round0247_registry import (
    GPU_HOURS_CAP,
    ROWS,
    SAFETY_PARAMETER_CLASS_NOTE,
    Round0247Error,
    registered_bounds,
    registered_value,
    registry_fingerprint,
    safety_parameter_inventory,
    verify_registry,
)
from basemap.round0248_inventory import require_inventory_complete
from basemap.round0249_external import (
    CONTROL_MEMORY_HIGH_BYTES,
    DEFAULT_EXTERNAL_MEMORY_MODE,
    MEMORY_HIGH_NOTE,
    cgroup_self_report,
    external_memory_limit_declaration,
    external_memory_mode_availability,
    run_escape_battery,
    run_fail_closed_control,
    run_memory_high_throttle_control,
)
from basemap.round0249_guard import (
    ACCEPTED_RESIDUAL_RISKS,
    GUARD_SCOPE_NOTE,
    guard_fix_receipt,
    run_declared_enforcement_control,
    run_discovery_inventory_control,
    run_legitimate_replay_still_scores,
    run_readonly_disclosure_control,
    run_sealing_arm_control,
    tracker_disclosure_properties_are_read_only,
)
from experiments.round0238_nodes import _check_runner_abort

ROUND_ID = "0249"

GUARDFIX_ACTION = "guardfix_0249"
EXTERNAL_ACTION = "external_0249"

GUARDFIX_CAPABILITY = "round0249-disclosure-and-enforcement-closure-v1"
EXTERNAL_CAPABILITY = "round0249-external-memory-high-bound-v1"

GUARDFIX_FILE = "guard-fix.json"
EXTERNAL_FILE = "external-memory-high-bound.json"

GUARDFIX_SCHEMA = "round0249-guard-fix-v1"
EXTERNAL_SCHEMA = "round0249-external-memory-high-bound-v1"

#: The node's own anonymous budget, as R0247's and R0248's nodes declared it.
NODE_ANON_BUDGET_BYTES = 16 * (1 << 30)

SCOPE_NOTE = (
    "R0249 trains nothing, builds nothing, registers no gate on a map, adopts "
    "nothing, measures no displacement and reads no bulk array. It closes the "
    "two in-process defects review-0248-01 named as real, replaces the "
    "hand-written guarded-module list with discovery, and moves the external "
    "memory bound from memory.max (which OOM-kills, and charges the page cache "
    "this memmap-heavy workload produces by the terabyte) onto memory.high "
    "(which throttles and cannot kill a CUDA holder), with root-scope as the "
    "default and no silent downgrade."
)
SAFETY_NOTE = (
    "no bulk input is opened at all; nothing is handed to cuVS; no signal is "
    "delivered on any path. The external-bound node starts child processes on "
    "purpose - that is the mechanism - and every one is a plain CPU allocator "
    "with no array library imported. " + MEMORY_HIGH_NOTE
)


def _receipt_envelope(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "round_id": ROUND_ID,
        "release_sha": str(manifest["release_sha"]),
        "rows": ROWS,
        "k": GRAPH_K,
        "gpu_hours_cap": GPU_HOURS_CAP,
        "gpu_hours_cap_note": GPU_HOURS_CAP_NOTE,
        "scope_note": SCOPE_NOTE,
        "safety_note": SAFETY_NOTE,
        "safety_parameter_class_note": SAFETY_PARAMETER_CLASS_NOTE,
        "guard_scope_note": GUARD_SCOPE_NOTE,
        "registry_fingerprint": registry_fingerprint(),
        #: review-0248-01 §E.4: `applied: true` was a claim about argv. Every
        #: R0249 receipt carries the limit the KERNEL applied to this process,
        #: read back from its own cgroup.
        "external_memory_limit_as_the_kernel_applied_it": cgroup_self_report(),
        "training_performed": False,
        "gate_registered": False,
        "adoption_claimed": False,
        "map_quality_claimed": False,
        "displacement_measured": False,
        "cuvs_calls": 0,
        "cuda_context_created": False,
        "signal_delivered": False,
    }


def _seal(output: str, name: str, body: Mapping[str, Any]) -> None:
    atomic_write_new_json(
        os.path.join(output, name),
        prompt_contract.seal(json_safe(json_scrub(dict(body)))),
        immutable=True,
    )


def _start_node(label: str) -> dict[str, Any]:
    verify_registry(label=label)
    return require_enforceable_abort_flag(label=label)


def _node_guard(label: str) -> EnforcedHostWatchdog:
    return EnforcedHostWatchdog(
        anonymous_budget_bytes=NODE_ANON_BUDGET_BYTES,
        interval_s=0.05,
        label=label,
    )


def _node_gate(label: str) -> AbortPollGate:
    return AbortPollGate(
        inner=_check_runner_abort,
        headroom_bytes=int(registered_value("max_declared_headroom_bytes")),
        label=label,
        training_performed=True,
    )


def _guard_tail(watchdog: EnforcedHostWatchdog, *, label: str) -> dict[str, Any]:
    receipt = watchdog.receipt()
    return {
        "host_watchdog": receipt,
        "sampler_liveness": require_live_sampler(receipt, label=label),
        "abort_flag_landing": require_abort_flag_landed(receipt, label=label),
    }


def _close_gate(
    gate: AbortPollGate, tail: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    enforcement = gate.require(
        measured_slope_bytes_per_s=measured_slope_from_trace(
            tail["host_watchdog"]["anonymous_trace_by_second"]
        )
    )
    enforcement["enforcement_evidence"] = require_enforcement_evidence(
        enforcement, label=label
    )
    return enforcement


def _run_under_the_node_guard(label: str, steps: int = 12) -> dict[str, Any]:
    guard = _node_guard(label)
    gate = _node_gate(label)
    with guard:
        gate.start()
        guard.poll(f"{label} start")
        for step in range(steps):
            gate(f"{label} step {step}")
            time.sleep(0.1)
        guard.poll(f"{label} end")
        gate.finish()
    tail = _guard_tail(guard, label=label)
    return {"tail": tail, "enforcement": _close_gate(gate, tail, label=label)}


# --------------------------------------------------------------------------- #
# node 1 — the two defects, and the scope that is no longer a list
# --------------------------------------------------------------------------- #
def run_guardfix(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    manifest = active["manifest"]
    if str(manifest.get("round_id")) != ROUND_ID:
        raise Round0247Error("R0249 handler received another queue")
    started = time.monotonic()
    abort_flag = _start_node("R0249 guard fix")
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0249 guardfix"
    )
    repo_root = str(manifest["repo_root"])

    inventory = require_inventory_complete(repo_root=repo_root)
    _check_runner_abort("R0249 derived the inventory over the discovered scope")
    disclosure = run_readonly_disclosure_control()
    _check_runner_abort("R0249 read-only disclosure control")
    sealing = run_sealing_arm_control()
    _check_runner_abort("R0249 sealing arm control")
    declared = run_declared_enforcement_control()
    _check_runner_abort("R0249 declared enforcement control")
    non_vacuity = run_legitimate_replay_still_scores()
    _check_runner_abort("R0249 replay non-vacuity control")
    discovery = run_discovery_inventory_control(repo_root=repo_root)
    _check_runner_abort("R0249 discovery inventory control")

    guarded = _run_under_the_node_guard("R0249 guardfix node tail")
    closure = guard_fix_receipt(
        inventory=inventory, disclosure=disclosure, sealing=sealing,
        declared=declared, non_vacuity=non_vacuity, discovery=discovery,
    )
    body = dict(_receipt_envelope(manifest))
    body.update({
        "schema": GUARDFIX_SCHEMA,
        "capabilities": [GUARDFIX_CAPABILITY],
        "child_processes_launched": 0,
        "abort_flag_precondition": abort_flag,
        "safety_parameter_inventory": safety_parameter_inventory(),
        "disclosure_fields_are_read_only_properties": (
            tracker_disclosure_properties_are_read_only()
        ),
        "closure": closure,
        "external_memory_limit_declared_by_this_queue": dict(
            manifest.get("external_memory_limit") or {}
        ),
        **registered_bounds([
            "replay",
            "r0246_max_poll_spacing_s",
            "min_binding_slope_bytes_per_s",
            "max_declared_headroom_bytes",
            "max_declared_anonymous_budget_bytes",
        ]),
        "enforcement_poll_spacing": guarded["enforcement"],
        **guarded["tail"],
        "performance": {"total_wall_s": time.monotonic() - started},
    })
    _seal(output, GUARDFIX_FILE, body)


# --------------------------------------------------------------------------- #
# node 2 — the bound that throttles instead of killing
# --------------------------------------------------------------------------- #
def run_external(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    manifest = active["manifest"]
    if str(manifest.get("round_id")) != ROUND_ID:
        raise Round0247Error("R0249 handler received another queue")
    started = time.monotonic()
    abort_flag = _start_node("R0249 external memory bound")
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0249 external"
    )
    workspace = create_fresh_directory(
        os.path.join(output, "control-workspace"), label="R0249 external control"
    )
    #: review-0248-01 §G item 7: R0248 claimed this ran first in BOTH nodes and
    #: it ran in one. It runs in both now.
    inventory = require_inventory_complete(repo_root=str(manifest["repo_root"]))
    _check_runner_abort("R0249 derived the inventory in the external node too")

    mode = str(job.get("external_control_mode") or DEFAULT_EXTERNAL_MEMORY_MODE)
    availability = external_memory_mode_availability(mode)
    _check_runner_abort("R0249 checked the external memory mode")
    fail_closed = run_fail_closed_control()
    _check_runner_abort("R0249 fail-closed mode control")
    throttle = run_memory_high_throttle_control(
        workspace=workspace, mode=mode,
        high_bytes=int(job.get("external_control_high_bytes")
                       or CONTROL_MEMORY_HIGH_BYTES),
    )
    _check_runner_abort("R0249 memory.high throttle control")
    escapes = run_escape_battery(
        workspace=workspace, mode=mode,
        high_bytes=int(job.get("external_control_high_bytes")
                       or CONTROL_MEMORY_HIGH_BYTES),
    )
    _check_runner_abort("R0249 escape battery")

    guarded = _run_under_the_node_guard("R0249 external node tail")
    body = dict(_receipt_envelope(manifest))
    body.update({
        "schema": EXTERNAL_SCHEMA,
        "capabilities": [EXTERNAL_CAPABILITY],
        #: three transient scopes for the throttle control, one per escape
        #: battery mode. Counted from the receipts, not asserted.
        "child_processes_launched": 3 + 1 + int(
            bool(escapes["the_other_mode"].get("ran"))
        ),
        "abort_flag_precondition": abort_flag,
        "inventory": inventory,
        "external_memory_mode": mode,
        "external_memory_mode_availability": availability,
        "production_limit_declaration": external_memory_limit_declaration(
            mode=mode
        ),
        "external_memory_limit_declared_by_this_queue": dict(
            manifest.get("external_memory_limit") or {}
        ),
        "fail_closed_control": fail_closed,
        "throttle_control": throttle,
        "escape_battery": escapes,
        "accepted_residual_risks": [
            dict(row) for row in ACCEPTED_RESIDUAL_RISKS
        ],
        "what_the_switch_to_memory_high_costs": (
            "a hard stop. Under memory.max the kernel STOPPED a runaway by "
            "killing it; under memory.high the kernel only makes continuing "
            "expensive, so the thing that stops a runaway is the cooperative "
            "in-process guard writing the abort flag. What is bought is that "
            "the bound can never OOM-kill a process holding a CUDA context, "
            "which is the failure that has cost this box two reboots, and that "
            "the limit no longer has to be sized against a denominator "
            "(anon + page cache + kernel memory) different from the one the "
            "in-process budget counts (anonymous only)"
        ),
        "what_the_external_bound_does_not_cover": (
            "poll spacing, abort latency, GPU memory, and receipt honesty. A "
            "node under a memory.high can still publish a receipt it invented, "
            "still report a poll spacing it never measured, and still script "
            "its own clock. The in-process guards of R0244-R0249 are kept "
            "because they catch MISTAKES - the real risk from a runner "
            "following instructions in good faith. Neither defends against a "
            "runner that fabricates; only independent recomputation by a "
            "reviewer who did not run the round does"
        ),
        **registered_bounds([
            "max_declared_anonymous_budget_bytes",
            "external_memory_limit_margin_bytes",
        ]),
        "enforcement_poll_spacing": guarded["enforcement"],
        **guarded["tail"],
        "performance": {"total_wall_s": time.monotonic() - started},
    })
    _seal(output, EXTERNAL_FILE, body)


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    action = str(job.get("action") or "")
    if action == GUARDFIX_ACTION:
        run_guardfix(active, job)
    elif action == EXTERNAL_ACTION:
        run_external(active, job)
    else:
        raise Round0247Error(f"R{ROUND_ID} does not authorize action {action!r}")


__all__ = [
    "EXTERNAL_ACTION",
    "EXTERNAL_CAPABILITY",
    "EXTERNAL_FILE",
    "EXTERNAL_SCHEMA",
    "GUARDFIX_ACTION",
    "GUARDFIX_CAPABILITY",
    "GUARDFIX_FILE",
    "GUARDFIX_SCHEMA",
    "NODE_ANON_BUDGET_BYTES",
    "ROUND_ID",
    "run_external",
    "run_guardfix",
    "run_job",
]
