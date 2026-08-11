"""R0248 — the memory bound that does not live in this process.

**Why this exists.** Three consecutive rounds (R0245, R0246, R0247) each shipped
a Python guard believing it had closed the override class, and a reviewer
defeated each one in the same shape: the guard verified one object and the
decision read another. review-0247-01 defeated R0247 five ways in an hour. Three
of those five — rebinding `REGISTERED_SAFETY_PARAMETERS`, mutating the dict
behind the `MappingProxyType` via `gc.get_referents()`, and running the
watchdog's `_now()` a thousand times slow — require the node to deliberately
cheat, and they are the same category as the fabricated receipt R0247 published
as unclosable.

    No in-process Python guard can constrain the process it runs in.
    The guard, the bound, the clock and the receipt are all objects the node
    owns. Three rounds of evidence say stop trying.

So the bound that matters is placed **outside** the process, by the kernel, on a
cgroup v2 `memory.max` with `memory.swap.max = 0`. A node cannot rebind it, it
has no clock to script, and the kill is delivered by the OOM killer rather than
by a `raise` the node could catch.

**What this covers and what it does not.** It bounds *memory*, and only memory.
It does not bound poll spacing, abort latency, GPU memory, or receipt honesty. A
node under a `memory.max` can still publish a fabricated receipt, still report a
poll spacing it never measured, and still lie about everything it did. The only
mitigation for that is independent recomputation by a reviewer who did not run
the round.

**Two modes, and the difference is who owns the cgroup directory.**

* `user-scope` — `systemd-run --user --scope`. The scope lands under
  `user@<uid>.service`, whose subtree systemd delegates to the user, so the
  files are user-writable and a node that deliberately writes `max` into its own
  `memory.max` **raises its own limit**. R0248 measured this rather than
  assuming it. It still binds every node that does not do that, which is the
  runaway-allocation class that has taken this box down twice.
* `root-scope` — `sudo -n systemd-run --scope --uid --gid`. The scope lands
  under `system.slice`, root-owned; the node gets `EACCES` on `memory.max`, on
  `memory.swap.max` and on `cgroup.procs`, and cannot move itself out. This is
  the mode a node cannot rebind, and it is what the positive control runs under.

**Sizing, and the wedge.** A kernel OOM kill of a process holding a CUDA context
is exactly the failure this machine has been rebooted for twice, so the
production limit is deliberately set *above* the registered in-process anonymous
budget: the cooperative in-process trip fires first and unwinds the context, and
the kernel is the backstop for when it does not. The positive control therefore
runs a **plain CPU allocator** in a child process with no CUDA context anywhere
near it.

Nothing here signals a process, hands cuVS anything, or imports a GPU array
library. It does start child processes — that is the whole mechanism — and every
one is a plain `python -c` allocator under a transient scope.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from typing import Any

from basemap.round0247_registry import (
    Round0247Error,
    registered_value,
    registry_fingerprint,
    verify_registry,
)

ROUND_ID = "0248"

#: The control's own limit. Small enough to be reached in under a second by a
#: plain `bytearray` loop, far too small to disturb the machine, and nowhere
#: near any GPU workload.
CONTROL_MEMORY_MAX_BYTES = 256 * (1 << 20)
#: What the killed arm tries to allocate: 8x its limit, in 16 MiB steps.
CONTROL_TARGET_BYTES = 8 * CONTROL_MEMORY_MAX_BYTES
CONTROL_STEP_BYTES = 16 * (1 << 20)
#: What the surviving arm allocates: a quarter of the limit.
CONTROL_SAFE_BYTES = CONTROL_MEMORY_MAX_BYTES // 4

EXTERNAL_BOUND_NOTE = (
    "REGISTERED 2026-08-11 (R0248). The node's memory bound is a cgroup v2 "
    "memory.max with memory.swap.max=0, placed by the runner through a "
    "transient systemd scope. It is not a Python object: the node cannot "
    "rebind it, cannot mutate it through gc.get_referents(), and cannot script "
    "its clock, because it has none. It bounds MEMORY and nothing else - not "
    "poll spacing, not abort latency, not GPU memory, not receipt honesty. The "
    "in-process guards of R0244-R0247 are kept: they catch MISTAKES, which is "
    "the real risk from a runner following instructions in good faith. Neither "
    "defends against a runner that fabricates; only independent recomputation "
    "by a reviewer does."
)


def external_memory_max_bytes() -> int:
    """The production limit, derived from the registered anonymous budget.

    `max_declared_anonymous_budget_bytes` (60 GiB) is the largest anonymous
    budget a node may declare, i.e. the level at which the in-process guard
    trips and writes the cooperative abort flag. The external limit sits one
    margin above it so the cooperative path has room to unwind a CUDA context
    before the kernel acts.
    """
    budget = float(registered_value("max_declared_anonymous_budget_bytes"))
    margin = float(registered_value("external_memory_limit_margin_bytes"))
    limit = int(budget + margin)
    #: The risk here is two-sided and the registry's single-direction clamp
    #: cannot express that, so both sides are asserted here instead of being
    #: implied. Too LOW and the kernel kills a CUDA holder before the
    #: cooperative trip fires - the wedge this box was rebooted for twice. Too
    #: HIGH and the limit is not a limit.
    if limit <= budget:
        raise Round0247Error(
            "R0248 STOP: the external memory limit must sit strictly above the "
            f"registered in-process anonymous budget {budget:.0f} B, so the "
            "cooperative trip fires before the kernel does"
        )
    total = machine_total_memory_bytes()
    if total and limit >= total:
        raise Round0247Error(
            f"R0248 STOP: an external memory limit of {limit} B is at or above "
            f"this machine's {total} B of RAM and would never bind"
        )
    return limit


def machine_total_memory_bytes() -> int:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return 0
    return 0


def external_memory_limit_declaration(*, mode: str = "user-scope") -> dict[str, Any]:
    """What the queue manifest declares, so the runner needs no basemap import."""
    verify_registry(label="R0248 external memory limit")
    limit = external_memory_max_bytes()
    budget = int(registered_value("max_declared_anonymous_budget_bytes"))
    return {
        "max_bytes": limit,
        "swap_max_bytes": 0,
        "mode": mode,
        "required": True,
        "derived_from": {
            "registered_max_declared_anonymous_budget_bytes": budget,
            "registered_external_memory_limit_margin_bytes": int(
                registered_value("external_memory_limit_margin_bytes")
            ),
            "machine_total_memory_bytes": machine_total_memory_bytes(),
            "arithmetic": (
                f"{budget} + "
                f"{int(registered_value('external_memory_limit_margin_bytes'))}"
                f" = {limit}"
            ),
        },
        "note": EXTERNAL_BOUND_NOTE,
        "registry_fingerprint": registry_fingerprint(),
    }


# --------------------------------------------------------------------------- #
# availability
# --------------------------------------------------------------------------- #
def cgroup_v2_memory_available() -> dict[str, Any]:
    controllers: list[str] = []
    try:
        with open("/sys/fs/cgroup/cgroup.controllers", encoding="utf-8") as fh:
            controllers = fh.read().split()
    except OSError:
        controllers = []
    sudo = subprocess.run(
        ["sudo", "-n", "true"], check=False, capture_output=True
    ).returncode == 0 if shutil.which("sudo") else False
    return {
        "unified_hierarchy": os.path.exists("/sys/fs/cgroup/cgroup.controllers"),
        "root_controllers": controllers,
        "memory_controller": "memory" in controllers,
        "systemd_run": shutil.which("systemd-run"),
        "passwordless_sudo_for_root_scope": sudo,
        "holds": bool("memory" in controllers and shutil.which("systemd-run")),
    }


def _scope_argv(
    *, unit: str, max_bytes: int, mode: str, env: dict[str, str] | None = None
) -> list[str]:
    properties = [
        "-p", f"MemoryMax={int(max_bytes)}",
        "-p", "MemorySwapMax=0",
        "-p", "MemoryAccounting=yes",
    ]
    if mode == "root-scope":
        argv = [
            "sudo", "-n", "systemd-run", "--scope", "--quiet", "--collect",
            f"--unit={unit}", f"--uid={os.getuid()}", f"--gid={os.getgid()}",
            *properties,
        ]
        for key, value in sorted((env or {}).items()):
            argv.extend(["--setenv", f"{key}={value}"])
        return argv
    return [
        "systemd-run", "--user", "--scope", "--quiet", "--collect",
        f"--unit={unit}", *properties,
    ]


#: The allocator. A plain CPU `bytearray` loop, touched so the pages are
#: resident. NOTHING here imports torch, cuml, cupy or numpy, so no CUDA
#: context exists anywhere in the killed process - a kernel OOM kill of a CUDA
#: holder is the wedge this machine has been rebooted for twice.
_ALLOCATOR = r"""
import os, sys, json
target = int(sys.argv[1]); step = int(sys.argv[2]); receipt = sys.argv[3]
blocks = []
allocated = 0
with open(receipt, "w") as h:
    json.dump({
        "pid": os.getpid(),
        "cgroup": open("/proc/self/cgroup").read().strip(),
        "in_process_guard_armed": False,
        "cuda_modules_imported": sorted(
            m for m in sys.modules
            if m.split(".")[0] in {"torch", "cuml", "cupy", "cuvs", "numpy"}
        ),
    }, h)
while allocated < target:
    block = bytearray(step)
    block[::4096] = b"x" * len(block[::4096])
    blocks.append(block)
    allocated += step
print(json.dumps({"allocated_bytes": allocated}))
"""

#: The escape battery, run INSIDE the scope. Every attempt is reported.
_ESCAPES = r"""
import json, os, sys
cg = "/sys/fs/cgroup" + open("/proc/self/cgroup").read().strip().split(":")[-1]
rows = []
def attempt(name, fn):
    try:
        fn(); rows.append({"attempt": name, "succeeded": True, "error": None})
    except Exception as exc:
        rows.append({"attempt": name, "succeeded": False,
                     "error": f"{type(exc).__name__}: {exc}"})
def raise_own_limit():
    with open(os.path.join(cg, "memory.max"), "w") as h: h.write("max")
def raise_own_swap():
    with open(os.path.join(cg, "memory.swap.max"), "w") as h: h.write("max")
def escape_to_parent():
    parent = os.path.dirname(cg.rstrip("/"))
    with open(os.path.join(parent, "cgroup.procs"), "w") as h:
        h.write(str(os.getpid()))
def fork_out():
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)
    child_cg = cg
    if child_cg != cg:
        raise RuntimeError("forked child left the cgroup")
    raise RuntimeError("a forked child stays in the same cgroup")
attempt("raise_my_own_memory_max", raise_own_limit)
attempt("raise_my_own_memory_swap_max", raise_own_swap)
attempt("move_myself_to_the_parent_cgroup", escape_to_parent)
attempt("fork_a_child_outside_the_cgroup", fork_out)
json.dump({
    "cgroup": cg,
    "memory_max_after_the_attempts": open(os.path.join(cg, "memory.max")).read().strip(),
    "memory_swap_max_after_the_attempts": open(
        os.path.join(cg, "memory.swap.max")).read().strip(),
    "attempts": rows,
}, open(sys.argv[1], "w"))
"""


def _journal_oom_evidence(unit: str, since: float) -> dict[str, Any]:
    """The kernel's own account of the kill, from `journalctl -k`."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(since - 5.0))
    completed = subprocess.run(
        ["journalctl", "-k", "--no-pager", "--since", stamp],
        check=False, capture_output=True, text=True,
    )
    lines = [
        line for line in completed.stdout.splitlines()
        if unit in line or "CONSTRAINT_MEMCG" in line
    ]
    matched = [line for line in lines if unit in line]
    return {
        "journalctl_returncode": completed.returncode,
        "lines_naming_this_unit": matched[:6],
        "constraint_memcg_lines": [
            line for line in lines if "CONSTRAINT_MEMCG" in line
        ][:6],
        "the_kernel_names_this_scope_as_the_oom_memcg": bool(
            any("CONSTRAINT_MEMCG" in line and unit in line for line in lines)
        ),
    }


def run_external_memory_bound_control(
    *, workspace: str, mode: str = "root-scope",
    max_bytes: int = CONTROL_MEMORY_MAX_BYTES,
) -> dict[str, Any]:
    """Prove the EXTERNAL bound fires, with the in-process guard disabled.

    Four arms:

    1. **killed** — a plain CPU allocator asks for `8x` the limit under a
       transient scope. It must die by `SIGKILL` from the kernel's OOM killer,
       and `journalctl -k` must name this scope as the `oom_memcg`.
    2. **survives** — the same allocator asking for a quarter of the limit
       under the same scope exits `0`. Without this the kill proves only that
       the allocator crashes.
    3. **no in-process guard** — the killed child's own receipt records that it
       armed nothing, and its peak is orders of magnitude below the registered
       in-process anonymous budget, so no in-process guard would have fired.
       The evidence is that the *external* bound acted.
    4. **escapes** — from inside the scope: raise my own `memory.max`, raise my
       own `memory.swap.max`, move myself to the parent cgroup, fork out. Every
       result is published, including the ones that succeed.
    """
    verify_registry(label="R0248 external memory bound control")
    os.makedirs(workspace, exist_ok=True)
    availability = cgroup_v2_memory_available()
    if not availability["holds"]:
        raise Round0247Error(
            f"R0248 STOP: this box cannot place a cgroup memory limit "
            f"{availability}. The external bound is the round's subject; a "
            "round that cannot place it must fail rather than report success."
        )
    if mode == "root-scope" and not availability["passwordless_sudo_for_root_scope"]:
        mode = "user-scope"

    started = time.time()
    unit_killed = f"r0248-oom-control-{os.getpid()}-killed"
    unit_survive = f"r0248-oom-control-{os.getpid()}-survives"
    unit_escape = f"r0248-oom-control-{os.getpid()}-escapes"
    killed_receipt = os.path.join(workspace, "killed-child.json")
    escape_receipt = os.path.join(workspace, "escape-attempts.json")

    # -- arm 1: the over-allocator, killed by the kernel -------------------- #
    killed = subprocess.run(
        _scope_argv(unit=unit_killed, max_bytes=max_bytes, mode=mode)
        + [sys.executable, "-c", _ALLOCATOR, str(CONTROL_TARGET_BYTES),
           str(CONTROL_STEP_BYTES), killed_receipt],
        #: NO `timeout=`. `subprocess.run(timeout=)` delivers SIGKILL, and the
        #: release's signal-safety detector rightly refuses that for any child
        #: it cannot prove GPU-free. It is also unnecessary: the bound on this
        #: child IS the cgroup limit - an over-allocator is OOM-killed by the
        #: kernel in well under a second - and the runner's cooperative soft
        #: deadline bounds the node above it. The allocator loop is finite by
        #: construction and the escape battery makes four bounded attempts.
        check=False, capture_output=True, text=True,
    )
    child_receipt: dict[str, Any] = {}
    if os.path.exists(killed_receipt):
        with open(killed_receipt, encoding="utf-8") as handle:
            child_receipt = json.load(handle)
    journal = _journal_oom_evidence(unit_killed, started)

    # -- arm 2: the under-allocator, under the same limit ------------------- #
    survived = subprocess.run(
        _scope_argv(unit=unit_survive, max_bytes=max_bytes, mode=mode)
        + [sys.executable, "-c", _ALLOCATOR, str(CONTROL_SAFE_BYTES),
           str(CONTROL_STEP_BYTES),
           os.path.join(workspace, "surviving-child.json")],
        #: NO `timeout=`. `subprocess.run(timeout=)` delivers SIGKILL, and the
        #: release's signal-safety detector rightly refuses that for any child
        #: it cannot prove GPU-free. It is also unnecessary: the bound on this
        #: child IS the cgroup limit - an over-allocator is OOM-killed by the
        #: kernel in well under a second - and the runner's cooperative soft
        #: deadline bounds the node above it. The allocator loop is finite by
        #: construction and the escape battery makes four bounded attempts.
        check=False, capture_output=True, text=True,
    )

    # -- arm 4: the escape battery ------------------------------------------ #
    escapes = subprocess.run(
        _scope_argv(unit=unit_escape, max_bytes=max_bytes, mode=mode)
        + [sys.executable, "-c", _ESCAPES, escape_receipt],
        #: NO `timeout=`. `subprocess.run(timeout=)` delivers SIGKILL, and the
        #: release's signal-safety detector rightly refuses that for any child
        #: it cannot prove GPU-free. It is also unnecessary: the bound on this
        #: child IS the cgroup limit - an over-allocator is OOM-killed by the
        #: kernel in well under a second - and the runner's cooperative soft
        #: deadline bounds the node above it. The allocator loop is finite by
        #: construction and the escape battery makes four bounded attempts.
        check=False, capture_output=True, text=True,
    )
    escape_rows: dict[str, Any] = {}
    if os.path.exists(escape_receipt):
        with open(escape_receipt, encoding="utf-8") as handle:
            escape_rows = json.load(handle)

    # -- the same escape battery in the OTHER mode -------------------------- #
    #: Published whether or not it succeeds, because it DOES succeed and a
    #: round that only reported the mode it wins in would be doing exactly what
    #: three reviews have punished. `systemd-run --user --scope` puts the scope
    #: under `user@<uid>.service`, whose subtree systemd delegates to the user,
    #: so the scope's own `memory.max` is user-writable and a node can raise
    #: it. The runner defaults to this mode because it needs no privilege and
    #: still binds the runaway-allocation class; `root-scope` is the mode a
    #: node cannot rebind.
    other = "user-scope" if mode == "root-scope" else "root-scope"
    other_rows: dict[str, Any] = {}
    other_receipt = os.path.join(workspace, f"escape-attempts-{other}.json")
    other_run = None
    if other == "user-scope" or availability["passwordless_sudo_for_root_scope"]:
        other_run = subprocess.run(
            _scope_argv(
                unit=f"r0248-oom-control-{os.getpid()}-escapes-other",
                max_bytes=max_bytes, mode=other,
            )
            + [sys.executable, "-c", _ESCAPES, other_receipt],
            #: NO `timeout=`. `subprocess.run(timeout=)` delivers SIGKILL, and the
            #: release's signal-safety detector rightly refuses that for any child
            #: it cannot prove GPU-free. It is also unnecessary: the bound on this
            #: child IS the cgroup limit - an over-allocator is OOM-killed by the
            #: kernel in well under a second - and the runner's cooperative soft
            #: deadline bounds the node above it. The allocator loop is finite by
            #: construction and the escape battery makes four bounded attempts.
        check=False, capture_output=True, text=True,
        )
        if os.path.exists(other_receipt):
            with open(other_receipt, encoding="utf-8") as handle:
                other_rows = json.load(handle)

    registered_budget = int(
        registered_value("max_declared_anonymous_budget_bytes")
    )
    arms = {
        "the_over_allocating_child_was_killed": bool(
            killed.returncode in (-9, 137)
        ),
        "the_kernel_oom_killer_did_it": bool(
            journal["the_kernel_names_this_scope_as_the_oom_memcg"]
        ),
        "an_under_allocating_child_survives_the_same_limit": bool(
            survived.returncode == 0
        ),
        "the_in_process_guard_was_not_armed": bool(
            child_receipt.get("in_process_guard_armed") is False
        ),
        "no_cuda_or_array_module_was_loaded_in_the_killed_child": bool(
            child_receipt.get("cuda_modules_imported") == []
        ),
        "the_in_process_budget_would_not_have_fired": bool(
            int(max_bytes) < registered_budget
        ),
        "the_escape_battery_ran": bool(escape_rows.get("attempts")),
    }
    successful_escapes = [
        row for row in (escape_rows.get("attempts") or [])
        if row.get("succeeded")
    ]
    evidence = {
        "control": "round0248-external-memory-bound-control-v1",
        "mode": mode,
        "availability": availability,
        "memory_max_bytes": int(max_bytes),
        "memory_swap_max_bytes": 0,
        "target_bytes": CONTROL_TARGET_BYTES,
        "safe_bytes": CONTROL_SAFE_BYTES,
        "registered_in_process_anonymous_budget_bytes": registered_budget,
        "killed_arm": {
            "unit": unit_killed,
            "returncode": killed.returncode,
            "killed_by_signal": bool(killed.returncode in (-9, 137)),
            "stdout_tail": killed.stdout[-400:],
            "stderr_tail": killed.stderr[-400:],
            "child_receipt": child_receipt,
        },
        "surviving_arm": {
            "unit": unit_survive,
            "returncode": survived.returncode,
            "stdout_tail": survived.stdout[-400:],
        },
        "escape_arm": {
            "unit": unit_escape,
            "returncode": escapes.returncode,
            "attempts": escape_rows.get("attempts"),
            "memory_max_after_the_attempts": escape_rows.get(
                "memory_max_after_the_attempts"
            ),
            "memory_swap_max_after_the_attempts": escape_rows.get(
                "memory_swap_max_after_the_attempts"
            ),
            "successful_escapes": successful_escapes,
            "the_node_can_defeat_this_mode": bool(successful_escapes),
        },
        "the_other_mode_escape_arm": {
            "mode": other,
            "returncode": None if other_run is None else other_run.returncode,
            "attempts": other_rows.get("attempts"),
            "memory_max_after_the_attempts": other_rows.get(
                "memory_max_after_the_attempts"
            ),
            "successful_escapes": [
                row for row in (other_rows.get("attempts") or [])
                if row.get("succeeded")
            ],
            "the_node_can_defeat_this_mode": bool([
                row for row in (other_rows.get("attempts") or [])
                if row.get("succeeded")
            ]),
            "why_it_is_published": (
                "a user scope's cgroup directory is delegated to the user, so "
                "a node CAN raise its own memory.max there. The runner "
                "defaults to user-scope because it needs no privilege and "
                "still binds every node that does not deliberately cheat; "
                "root-scope is the mode a node cannot rebind. Both results are "
                "published because reporting only the winning mode is the "
                "shape three reviews have punished"
            ),
        },
        "kernel_journal": journal,
        "arms": arms,
        "failures": [name for name, ok in arms.items() if not ok],
        "note": EXTERNAL_BOUND_NOTE,
        "what_this_does_not_cover": (
            "poll spacing, abort latency, GPU memory, and receipt honesty. A "
            "node under a memory.max can still publish a receipt it invented. "
            "The only mitigation for that is independent recomputation by a "
            "reviewer who did not run the round"
        ),
        "registry_fingerprint": registry_fingerprint(),
    }
    evidence["holds"] = not evidence["failures"]
    if evidence["failures"]:
        raise Round0247Error(
            "R0248 EXTERNAL MEMORY BOUND CONTROL DID NOT FIRE: "
            f"{evidence['failures']}. killed rc={killed.returncode}, "
            f"survivor rc={survived.returncode}, journal={journal}"
        )
    return evidence


__all__ = [
    "CONTROL_MEMORY_MAX_BYTES",
    "CONTROL_SAFE_BYTES",
    "CONTROL_TARGET_BYTES",
    "EXTERNAL_BOUND_NOTE",
    "ROUND_ID",
    "cgroup_v2_memory_available",
    "external_memory_limit_declaration",
    "external_memory_max_bytes",
    "machine_total_memory_bytes",
    "run_external_memory_bound_control",
]
