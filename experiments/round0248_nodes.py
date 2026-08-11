"""Execute R0248 — close the four gaps, and move the memory bound out of process.

Two CPU-shaped nodes. Neither trains anything, builds anything, reads a bulk
array, or creates a CUDA context:

* `gapguard_0248` — the mechanically derived inventory, plus one positive
  control per gap review-0247-01 found, each in the reviewer's shape: the two
  observation-gap module globals reassigned, the sampler's anonymous ceiling
  reassigned, the abort-reader marker applied, and a `replay=True` gate sealed.
  Plus the self-attack battery, including the attacks that still succeed.
* `external_0248` — the bound that is not a Python object. A plain CPU
  allocator is placed under a cgroup v2 `memory.max` with `memory.swap.max=0`
  and asks for eight times its limit **with the in-process guard disabled**; it
  is killed by the kernel and `journalctl -k` names the scope as the
  `oom_memcg`. A second child under the same limit survives, so the kill is the
  limit and not the allocator. Then the escape battery runs inside the scope in
  both modes and publishes what it can and cannot do.

**Never** hand a cgroup limit to a process holding a CUDA context: a kernel OOM
kill of a CUDA holder is the wedge this box has been rebooted for twice. Both
nodes run with no GPU work, and the allocator imports no array library at all.

Every registered check is IMPORTED, never re-typed. The `external_0248` node
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
from basemap.round0248_external import (
    CONTROL_MEMORY_MAX_BYTES,
    EXTERNAL_BOUND_NOTE,
    cgroup_v2_memory_available,
    external_memory_limit_declaration,
    run_external_memory_bound_control,
)
from basemap.round0248_guard import (
    GUARD_SCOPE_NOTE,
    gap_closure_receipt,
    run_gap1_observation_gap_control,
    run_gap2_sampler_bytes_control,
    run_gap3_abort_reader_control,
    run_gap4_replay_control,
    run_self_attack_battery,
)
from basemap.round0248_inventory import require_inventory_complete
from experiments.round0238_nodes import _check_runner_abort

ROUND_ID = "0248"

GAPGUARD_ACTION = "gapguard_0248"
EXTERNAL_ACTION = "external_0248"

GAPGUARD_CAPABILITY = "round0248-registry-comparison-site-closure-v1"
EXTERNAL_CAPABILITY = "round0248-external-cgroup-memory-bound-v1"

GAPGUARD_FILE = "gap-closure.json"
EXTERNAL_FILE = "external-memory-bound.json"

GAPGUARD_SCHEMA = "round0248-gap-closure-v1"
EXTERNAL_SCHEMA = "round0248-external-memory-bound-v1"

#: The node's own anonymous budget, as R0247's nodes declared it.
NODE_ANON_BUDGET_BYTES = 16 * (1 << 30)

SCOPE_NOTE = (
    "R0248 trains nothing, builds nothing, registers no gate on a map, adopts "
    "nothing, measures no displacement and reads no bulk array. It closes the "
    "four bounds review-0247-01 found that never entered the decision, derives "
    "the inventory from the source instead of by hand, and places the node's "
    "memory bound outside the process on a cgroup v2 memory.max."
)
SAFETY_NOTE = (
    "no bulk input is opened at all; nothing is handed to cuVS; no signal is "
    "delivered on any path. The external-bound node starts child processes on "
    "purpose - that is the mechanism - and every one is a plain CPU allocator "
    "with no array library imported, because a kernel OOM kill of a CUDA "
    "holder is the wedge this machine has been rebooted for twice. "
    + EXTERNAL_BOUND_NOTE
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
# node 1 — the four gaps, and the inventory that is no longer written by hand
# --------------------------------------------------------------------------- #
def run_gapguard(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    manifest = active["manifest"]
    if str(manifest.get("round_id")) != ROUND_ID:
        raise Round0247Error("R0248 handler received another queue")
    started = time.monotonic()
    abort_flag = _start_node("R0248 gap closure")
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0248 gapguard"
    )
    repo_root = str(manifest["repo_root"])

    inventory = require_inventory_complete(repo_root=repo_root)
    _check_runner_abort("R0248 derived the inventory mechanically")
    gap1 = run_gap1_observation_gap_control()
    _check_runner_abort("R0248 gap-1 control")
    gap2 = run_gap2_sampler_bytes_control()
    _check_runner_abort("R0248 gap-2 control")
    gap3 = run_gap3_abort_reader_control()
    _check_runner_abort("R0248 gap-3 control")
    gap4 = run_gap4_replay_control(repo_root=repo_root)
    _check_runner_abort("R0248 gap-4 control")
    self_attacks = run_self_attack_battery()
    _check_runner_abort("R0248 self-attack battery")

    guarded = _run_under_the_node_guard("R0248 gapguard node tail")
    closure = gap_closure_receipt(
        inventory=inventory, gap1=gap1, gap2=gap2, gap3=gap3, gap4=gap4,
        self_attacks=self_attacks,
    )
    body = dict(_receipt_envelope(manifest))
    body.update({
        "schema": GAPGUARD_SCHEMA,
        "capabilities": [GAPGUARD_CAPABILITY],
        "child_processes_launched": 0,
        "abort_flag_precondition": abort_flag,
        "safety_parameter_inventory": safety_parameter_inventory(),
        "closure": closure,
        "external_memory_limit_declared_by_this_queue": dict(
            manifest.get("external_memory_limit") or {}
        ),
        **registered_bounds([
            "watchdog_max_observation_gap_s",
            "watchdog_max_mean_observation_gap_s",
            "sampler_max_anonymous_bytes",
            "replay",
            "external_memory_limit_margin_bytes",
        ]),
        "enforcement_poll_spacing": guarded["enforcement"],
        **guarded["tail"],
        "performance": {"total_wall_s": time.monotonic() - started},
    })
    _seal(output, GAPGUARD_FILE, body)


# --------------------------------------------------------------------------- #
# node 2 — the bound that is not a Python object
# --------------------------------------------------------------------------- #
def run_external(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    manifest = active["manifest"]
    if str(manifest.get("round_id")) != ROUND_ID:
        raise Round0247Error("R0248 handler received another queue")
    started = time.monotonic()
    abort_flag = _start_node("R0248 external memory bound")
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0248 external"
    )
    workspace = create_fresh_directory(
        os.path.join(output, "control-workspace"), label="R0248 external control"
    )

    availability = cgroup_v2_memory_available()
    _check_runner_abort("R0248 checked cgroup v2 availability")
    control = run_external_memory_bound_control(
        workspace=workspace,
        mode=str(job.get("external_control_mode") or "root-scope"),
        max_bytes=int(job.get("external_control_max_bytes")
                      or CONTROL_MEMORY_MAX_BYTES),
    )
    _check_runner_abort("R0248 external memory bound control")

    guarded = _run_under_the_node_guard("R0248 external node tail")
    body = dict(_receipt_envelope(manifest))
    body.update({
        "schema": EXTERNAL_SCHEMA,
        "capabilities": [EXTERNAL_CAPABILITY],
        #: three transient scopes in root-scope mode, four when the other-mode
        #: escape arm also runs. Counted, not asserted.
        "child_processes_launched": 3 + int(
            bool(control["the_other_mode_escape_arm"]["returncode"] is not None)
        ),
        "abort_flag_precondition": abort_flag,
        "cgroup_v2_availability": availability,
        "production_limit_declaration": external_memory_limit_declaration(),
        "external_memory_limit_declared_by_this_queue": dict(
            manifest.get("external_memory_limit") or {}
        ),
        "control": control,
        "what_the_external_bound_does_not_cover": (
            "poll spacing, abort latency, GPU memory, and receipt honesty. A "
            "node under a memory.max can still publish a receipt it invented, "
            "still report a poll spacing it never measured, and still script "
            "its own clock. The in-process guards of R0244-R0247 are kept "
            "because they catch MISTAKES - the real risk from a runner "
            "following instructions in good faith - and the external limit "
            "catches the runaway allocation that has taken this box down "
            "twice. Neither defends against a runner that fabricates; only "
            "independent recomputation by a reviewer who did not run the round "
            "does"
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
    if action == GAPGUARD_ACTION:
        run_gapguard(active, job)
    elif action == EXTERNAL_ACTION:
        run_external(active, job)
    else:
        raise Round0247Error(f"R{ROUND_ID} does not authorize action {action!r}")


__all__ = [
    "EXTERNAL_ACTION",
    "EXTERNAL_CAPABILITY",
    "EXTERNAL_FILE",
    "EXTERNAL_SCHEMA",
    "GAPGUARD_ACTION",
    "GAPGUARD_CAPABILITY",
    "GAPGUARD_FILE",
    "GAPGUARD_SCHEMA",
    "NODE_ANON_BUDGET_BYTES",
    "ROUND_ID",
    "run_external",
    "run_gapguard",
    "run_job",
]
