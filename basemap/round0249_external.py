"""R0249 — the external memory bound, moved off a kill and onto back-pressure.

R0248 placed every node under a cgroup v2 **`memory.max`** with
`memory.swap.max = 0`. review-0248-01 §E measured that bound honestly and found
two things wrong with the *choice of file*, neither of which is a bug in the
mechanism:

1. **`memory.max` charges page cache**, not just anonymous memory. Measured
   under a `268,435,456` B scope reading `2` GiB from `/data`:
   `memory.current 268,423,168` of which `file 261,234,688` and
   `anon 5,627,904`, and a re-read cost `0.381` s against `0.069` s unlimited —
   a `5.5x` penalty from the cgroup evicting the cache it had just filled. This
   program's workload is *deliberately* memmap-heavy: terabytes stream through
   page cache while anonymous memory sits near 4 GB. The in-process budget the
   external limit was sized against — `max_declared_anonymous_budget_bytes` —
   counts anonymous only. So "the limit sits above the in-process budget, so the
   cooperative trip fires first" was an assumption about the *composition* of
   the node's memory, not an arithmetic consequence of `68 > 60` GiB.
2. **`memory.max` OOM-kills.** Killing a process that holds a live CUDA context
   is exactly the failure that has wedged this box into an unrecoverable UVM
   deadlock twice, each time costing a reboot. A safety device whose failure
   mode is the accident it was installed to prevent is the wrong device.

**R0249 switches to `memory.high`.** A `memory.high` breach cannot OOM-kill: the
kernel throttles the allocating task and reclaims, and the process survives. The
positive control below plants a plain CPU allocator that exceeds its
`memory.high` by `8x` and requires it to **survive**, with the kernel's own
`memory.events` `high` counter as the evidence that the limit acted.

**State the trade plainly.** We have traded a hard stop for back-pressure. Under
`memory.max` the kernel stopped a runaway; under `memory.high` the kernel only
makes continuing expensive, and **the thing that stops a runaway is now the
cooperative in-process guard** — the R0244 watchdog writing the abort flag,
which the node reads and unwinds through the normal Python path. The kernel's
role has changed from executioner to brake. What we gain is that the brake
cannot wedge the card, and what we lose is that a node which ignores its own
guard *and* keeps allocating is bounded only by the machine.

**And the mode default changes.** review-0248-01 §E.6 measured that a
`user-scope` node lifts its own limit with two `open(...,"w").write("max")`
calls, leaving no override record and no receipt field — weaker than the Python
read-only properties the external bound was added to outrank. `root-scope`
returns `EACCES` on all three escape attempts. `root-scope` is now the default,
and there is **no silent fallback**: a mode that cannot be placed refuses.

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
from basemap.round0248_external import (
    cgroup_v2_memory_available,
    external_memory_max_bytes,
    machine_total_memory_bytes,
)

ROUND_ID = "0249"

#: The default mode, and the reason it changed. review-0248-01 §E.6.
DEFAULT_EXTERNAL_MEMORY_MODE = "root-scope"
KNOWN_EXTERNAL_MEMORY_MODES = ("root-scope", "user-scope")

#: The control's own limit. Small enough to be crossed in under a second by a
#: plain `bytearray` loop, far too small to disturb the machine, and nowhere
#: near any GPU workload.
CONTROL_MEMORY_HIGH_BYTES = 256 * (1 << 20)
#: What the throttled arm ASKS for: 8x its `memory.high`, in 4 MiB steps.
CONTROL_TARGET_BYTES = 8 * CONTROL_MEMORY_HIGH_BYTES
CONTROL_STEP_BYTES = 4 * (1 << 20)
#: How long the throttled arm is allowed to keep trying before it stops and
#: reports. **This bound is the whole design.** A first attempt at this control
#: asked for `8x` the limit with no wall bound: the child was still alive and
#: still allocating after `162` s, having managed `295` MB, with the `high`
#: counter at `6,798` and a measured rate of about `4` MB/min. That is the
#: back-pressure working — anonymous pages with `memory.swap.max=0` are not
#: reclaimable, so the kernel can only stall the allocator — but "run it to
#: completion" is not a control, it is a hang. The arm therefore MEASURES the
#: throttle (unthrottled bytes/s below the limit against throttled bytes/s
#: above it) instead of waiting for it to relent.
CONTROL_THROTTLE_WALL_BUDGET_S = 20.0
#: The fraction of `memory.high` the child fills at full speed first, so the
#: unthrottled rate is measured on the same allocator in the same process.
CONTROL_UNTHROTTLED_FRACTION = 0.75
#: The throttled rate must be at most this fraction of the unthrottled rate for
#: the arm to hold. Two orders of magnitude, against a measured `~1000x`.
CONTROL_MAX_THROTTLED_RATE_RATIO = 0.01
#: What the untripped arm allocates: a quarter of the limit. Without this the
#: `high` counter proves only that the counter exists.
CONTROL_SAFE_BYTES = CONTROL_MEMORY_HIGH_BYTES // 4
#: How many attempts `_ESCAPES` makes. A battery reporting fewer than this has
#: not run, whatever its return code says.
_ESCAPE_ATTEMPTS = 5

MEMORY_HIGH_NOTE = (
    "REGISTERED 2026-08-11 (R0249). The node's external memory bound is a "
    "cgroup v2 memory.high with memory.swap.max=0 and NO memory.max, placed by "
    "the runner through a root-owned transient systemd scope. memory.high "
    "throttles and reclaims; it cannot OOM-kill, which is the failure that has "
    "cost this box two reboots. The trade is explicit: the kernel supplies "
    "back-pressure and the COOPERATIVE in-process guard is what stops a "
    "runaway. It bounds MEMORY and nothing else - not poll spacing, not abort "
    "latency, not GPU memory, not receipt honesty. Neither this nor any "
    "in-process guard defends against a runner that fabricates; only "
    "independent recomputation by a reviewer does."
)


class Round0249Error(RuntimeError):
    """R0249 fails closed."""


# --------------------------------------------------------------------------- #
# reading a cgroup back — so `applied` is a measurement, not a claim about argv
# --------------------------------------------------------------------------- #
def _read_cgroup_file(cgroup_dir: str, name: str) -> str | None:
    try:
        with open(os.path.join(cgroup_dir, name), encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return None


def _parse_events(raw: str | None) -> dict[str, int]:
    events: dict[str, int] = {}
    for line in (raw or "").splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                events[parts[0]] = int(parts[1])
            except ValueError:
                continue
    return events


def own_cgroup_directory() -> str | None:
    try:
        with open("/proc/self/cgroup", encoding="utf-8") as handle:
            relative = handle.read().strip().split(":")[-1]
    except OSError:
        return None
    if not relative:
        return None
    return "/sys/fs/cgroup" + relative


def cgroup_self_report() -> dict[str, Any]:
    """What the kernel actually applied to THIS process, read from its cgroup.

    review-0248-01 §E.4: R0248's `applied: true` was written by the runner when
    the argv prefix was *constructed*, and neither node ever read back its own
    cgroup. The transient scopes are `--collect`ed and gone, so the claim was
    unverifiable post hoc. This is the read-back; every R0249 node puts it in
    its receipt, so the limit in the receipt is the limit the kernel imposed.
    """
    directory = own_cgroup_directory()
    if directory is None:
        return {
            "cgroup": None,
            "readable": False,
            "why": "no /proc/self/cgroup entry",
        }
    events = _parse_events(_read_cgroup_file(directory, "memory.events"))
    high = _read_cgroup_file(directory, "memory.high")
    maximum = _read_cgroup_file(directory, "memory.max")
    return {
        "cgroup": directory,
        "readable": os.path.isdir(directory),
        "memory_high": high,
        "memory_max": maximum,
        "memory_swap_max": _read_cgroup_file(directory, "memory.swap.max"),
        "memory_current": _read_cgroup_file(directory, "memory.current"),
        "memory_events": events,
        "memory_stat_head": (
            _read_cgroup_file(directory, "memory.stat") or ""
        ).splitlines()[:6],
        "a_memory_high_limit_is_in_force": bool(high not in (None, "max")),
        "no_memory_max_kill_limit_is_in_force": bool(maximum in (None, "max")),
        "times_the_high_limit_was_breached": int(events.get("high", 0)),
        "times_the_kernel_oom_killed_in_this_cgroup": int(
            events.get("oom_kill", 0)
        ),
        "note": MEMORY_HIGH_NOTE,
    }


# --------------------------------------------------------------------------- #
# the mode, and the refusal to downgrade it
# --------------------------------------------------------------------------- #
def external_memory_mode_availability(mode: str) -> dict[str, Any]:
    """Can this box place a limit in exactly this mode? No opinion about others."""
    base = cgroup_v2_memory_available()
    if mode not in KNOWN_EXTERNAL_MEMORY_MODES:
        return {
            "mode": mode, "available": False,
            "why": f"unknown external memory mode {mode!r}", "base": base,
        }
    if not base["holds"]:
        return {
            "mode": mode, "available": False,
            "why": "this box cannot place a cgroup v2 memory limit at all",
            "base": base,
        }
    if mode == "root-scope" and not base["passwordless_sudo_for_root_scope"]:
        return {
            "mode": mode, "available": False,
            "why": "root-scope needs passwordless sudo -n and it is unavailable",
            "base": base,
        }
    return {"mode": mode, "available": True, "why": "ok", "base": base}


def require_external_memory_mode(
    mode: str = DEFAULT_EXTERNAL_MEMORY_MODE,
    *,
    availability: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed on a mode this box cannot place. **Never downgrade.**

    R0248's control silently rewrote `root-scope` to `user-scope` when
    `sudo -n` was unavailable — a downgrade to a bound the node can lift in two
    writes, with the receipt still reporting a limit. There is no such path
    here: an unavailable mode raises, and the caller decides.
    """
    state = availability or external_memory_mode_availability(mode)
    if not state["available"]:
        raise Round0249Error(
            f"R0249 STOP: external memory mode {mode!r} cannot be placed on "
            f"this box ({state['why']}). R0249 does NOT fall back to a weaker "
            "mode: under user-scope the node's own limit file is user-writable "
            "and two writes remove the bound, so a receipt reporting a limit "
            "the node can lift is worse than no limit. Re-declare the mode "
            "explicitly if a weaker bound is genuinely intended."
        )
    return state


def external_memory_limit_declaration(
    *, mode: str = DEFAULT_EXTERNAL_MEMORY_MODE,
) -> dict[str, Any]:
    """What the queue manifest declares. Sized from the registry, never typed."""
    verify_registry(label="R0249 external memory limit")
    require_external_memory_mode(mode)
    limit = external_memory_max_bytes()
    budget = int(registered_value("max_declared_anonymous_budget_bytes"))
    margin = int(registered_value("external_memory_limit_margin_bytes"))
    return {
        "max_bytes": limit,
        "swap_max_bytes": 0,
        "mode": mode,
        "required": True,
        "limit_file": "memory.high",
        "enforcement": "throttle_and_reclaim",
        "can_oom_kill_the_node": False,
        "derived_from": {
            "registered_max_declared_anonymous_budget_bytes": budget,
            "registered_external_memory_limit_margin_bytes": margin,
            "machine_total_memory_bytes": machine_total_memory_bytes(),
            "arithmetic": f"{budget} + {margin} = {limit}",
        },
        "note": MEMORY_HIGH_NOTE,
        "what_the_switch_costs": (
            "a hard stop. memory.max stopped a runaway by killing it; "
            "memory.high only throttles, so the thing that STOPS a runaway is "
            "the cooperative in-process guard writing the abort flag. The "
            "kernel is the brake, not the executioner"
        ),
        "registry_fingerprint": registry_fingerprint(),
    }


# --------------------------------------------------------------------------- #
# the children
# --------------------------------------------------------------------------- #
def _scope_argv(*, unit: str, properties: list[str], mode: str) -> list[str]:
    if mode == "root-scope":
        return [
            "sudo", "-n", "systemd-run", "--scope", "--quiet", "--collect",
            f"--unit={unit}", f"--uid={os.getuid()}", f"--gid={os.getgid()}",
            *properties,
        ]
    return [
        "systemd-run", "--user", "--scope", "--quiet", "--collect",
        f"--unit={unit}", *properties,
    ]


def _high_properties(high_bytes: int) -> list[str]:
    #: MemoryHigh and deliberately no MemoryMax. Setting both puts the kill back.
    return [
        "-p", f"MemoryHigh={int(high_bytes)}",
        "-p", "MemorySwapMax=0",
        "-p", "MemoryAccounting=yes",
    ]


def _max_properties(max_bytes: int) -> list[str]:
    #: The R0248 shape, used ONCE, by the contrast arm, on a plain CPU
    #: allocator that holds no CUDA context, to show what we moved away from.
    return [
        "-p", f"MemoryMax={int(max_bytes)}",
        "-p", "MemorySwapMax=0",
        "-p", "MemoryAccounting=yes",
    ]


#: The allocator. A plain CPU `bytearray` loop, touched so the pages are
#: resident, which then READS ITS OWN CGROUP and reports what the kernel did to
#: it. NOTHING here imports torch, cuml, cupy or numpy, so no CUDA context
#: exists anywhere in this process.
_ALLOCATOR = r"""
import os, sys, json, time
target = int(sys.argv[1]); step = int(sys.argv[2]); receipt = sys.argv[3]
free_target = int(sys.argv[4]); budget_s = float(sys.argv[5])
cg = "/sys/fs/cgroup" + open("/proc/self/cgroup").read().strip().split(":")[-1]
def read(name):
    try:
        return open(os.path.join(cg, name)).read().strip()
    except OSError:
        return None
def events():
    rows = {}
    for line in (read("memory.events") or "").splitlines():
        parts = line.split()
        if len(parts) == 2:
            rows[parts[0]] = int(parts[1])
    return rows
def grab(n):
    block = bytearray(n)
    block[::4096] = b"x" * len(block[::4096])
    return block
before = events()
blocks = []
allocated = 0
# phase 1: below the limit, at full speed. This is the denominator.
free_started = time.monotonic()
while allocated < min(free_target, target):
    blocks.append(grab(step)); allocated += step
free_elapsed = max(time.monotonic() - free_started, 1e-9)
free_bytes = allocated
events_after_phase_1 = events()
current_after_phase_1 = read("memory.current")
# phase 2: past the limit, bounded by wall time rather than by success. The
# kernel throttles here; the process must SURVIVE, not finish.
throttled_started = time.monotonic()
deadline = throttled_started + budget_s
while allocated < target and time.monotonic() < deadline:
    blocks.append(grab(step)); allocated += step
throttled_elapsed = max(time.monotonic() - throttled_started, 1e-9)
throttled_bytes = allocated - free_bytes
after = events()
free_rate = free_bytes / free_elapsed
throttled_rate = throttled_bytes / throttled_elapsed
json.dump({
    "pid": os.getpid(),
    "cgroup": cg,
    "allocated_bytes": allocated,
    "requested_bytes": target,
    "reached_the_request": bool(allocated >= target),
    "unthrottled_bytes": free_bytes,
    "unthrottled_wall_s": free_elapsed,
    "unthrottled_bytes_per_s": free_rate,
    "throttled_bytes": throttled_bytes,
    "throttled_wall_s": throttled_elapsed,
    "throttled_bytes_per_s": throttled_rate,
    "throttled_rate_over_unthrottled_rate": (
        throttled_rate / free_rate if free_rate > 0 else None
    ),
    "memory_high": read("memory.high"),
    "memory_max": read("memory.max"),
    "memory_swap_max": read("memory.swap.max"),
    "memory_current": read("memory.current"),
    "memory_current_after_phase_1": current_after_phase_1,
    "memory_events_before": before,
    "memory_events_after_phase_1": events_after_phase_1,
    "memory_events_after": after,
    "in_process_guard_armed": False,
    "cuda_modules_imported": sorted(
        m for m in sys.modules
        if m.split(".")[0] in {"torch", "cuml", "cupy", "cuvs", "numpy"}
    ),
}, open(receipt, "w"))
print(json.dumps({"allocated_bytes": allocated, "survived": True}))
"""

#: The escape battery, run INSIDE the scope. Every attempt is reported, and
#: `memory.high` is now one of the files a node would have to be able to write.
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
def write(name, value):
    def _do():
        with open(os.path.join(cg, name), "w") as h:
            h.write(value)
    return _do
def escape_to_parent():
    parent = os.path.dirname(cg.rstrip("/"))
    with open(os.path.join(parent, "cgroup.procs"), "w") as h:
        h.write(str(os.getpid()))
def fork_out():
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)
    raise RuntimeError("a forked child stays in the same cgroup")
def read(name):
    try:
        return open(os.path.join(cg, name)).read().strip()
    except OSError as exc:
        return f"unreadable: {exc}"
attempt("raise_my_own_memory_high", write("memory.high", "max"))
attempt("set_my_own_memory_max_to_max", write("memory.max", "max"))
attempt("raise_my_own_memory_swap_max", write("memory.swap.max", "max"))
attempt("move_myself_to_the_parent_cgroup", escape_to_parent)
attempt("fork_a_child_outside_the_cgroup", fork_out)
json.dump({
    "cgroup": cg,
    "memory_high_after_the_attempts": read("memory.high"),
    "memory_max_after_the_attempts": read("memory.max"),
    "memory_swap_max_after_the_attempts": read("memory.swap.max"),
    "attempts": rows,
}, open(sys.argv[1], "w"))
"""


def _journal_oom_lines(unit: str, since: float) -> dict[str, Any]:
    """The kernel's account. Under `memory.high` there must be NOTHING here."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(since - 5.0))
    completed = subprocess.run(
        ["journalctl", "-k", "--no-pager", "--since", stamp],
        check=False, capture_output=True, text=True,
    )
    naming = [line for line in completed.stdout.splitlines() if unit in line]
    return {
        "journalctl_returncode": completed.returncode,
        "lines_naming_this_unit": naming[:6],
        "the_kernel_oom_killed_this_scope": bool(
            any("CONSTRAINT_MEMCG" in line for line in naming)
        ),
    }


def _user_scope_environment() -> dict[str, str]:
    """`systemd-run --user` needs the session bus, and a root-scope node has none.

    R0249, found by attacking the round's own first run. `external_0249` runs
    under a ROOT-owned scope, whose environment carries neither
    `XDG_RUNTIME_DIR` nor `DBUS_SESSION_BUS_ADDRESS`, so the `user-scope` half
    of the escape battery died with `Failed to connect to bus: No medium found`
    and published **zero attempts** while its `the_node_can_defeat_this_mode`
    field read `false` — a receipt saying user-scope is safe, which it is not.
    That is the "report only the mode that wins" shape three reviews have
    punished, arrived at by accident. The addresses are derived here the same
    way `roundrun` derives them, and never invented.
    """
    environment = dict(os.environ)
    runtime_dir = environment.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    environment.setdefault("XDG_RUNTIME_DIR", runtime_dir)
    environment.setdefault(
        "DBUS_SESSION_BUS_ADDRESS", f"unix:path={runtime_dir}/bus"
    )
    return environment


def _run_child(
    argv: list[str], *, env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    #: NO `timeout=`. `subprocess.run(timeout=)` delivers SIGKILL, and the
    #: release's signal-safety detector rightly refuses that for any child it
    #: cannot prove GPU-free. The allocator loop is finite by construction, the
    #: escape battery makes five bounded attempts, and the runner's cooperative
    #: soft deadline bounds the node above all of them.
    return subprocess.run(
        argv, check=False, capture_output=True, text=True, env=env
    )


def _load(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------- #
# the positive control: throttled and ALIVE
# --------------------------------------------------------------------------- #
def run_memory_high_throttle_control(
    *, workspace: str, mode: str = DEFAULT_EXTERNAL_MEMORY_MODE,
    high_bytes: int = CONTROL_MEMORY_HIGH_BYTES,
) -> dict[str, Any]:
    """A plain CPU allocator that exceeds `memory.high` is THROTTLED, not killed.

    Arms, all measured:

    1. **survives** — the over-allocator exits `0` after asking for `8x` its
       `memory.high`. Under R0248's `memory.max` this same child is killed with
       `returncode -9`; arm 9 runs exactly that, once, for the contrast.
    2. **the kernel acted** — the child's own `memory.events` `high` counter
       rose while it allocated. This is the kernel's account of the throttle,
       read from inside the scope, not inferred from wall time.
    3. **the back-pressure is quantified** — the same allocator in the same
       process measures its own rate below the limit and above it; above it the
       rate must collapse by at least two orders of magnitude.
    4. **the cgroup really sat at its limit** — `memory.current` at or above
       `memory.high` when the child stopped, so the throttle is not an artefact
       of an allocator that never got there.
    5. **nothing was OOM-killed** — `memory.events` `oom` and `oom_kill` are
       both `0`, and `journalctl -k` names this scope nowhere.
    6. **there is no kill limit at all** — the child reads `memory.max` as
       `max`. If both files were set, the kill would still be armed.
    7. **swap is pinned at zero** — back-pressure cannot be absorbed by swap.
    8. **an under-allocating child does not trip it** — a second child at a
       quarter of the limit exits `0` with the `high` counter unmoved. Without
       this the counter proves only that a counter exists.
    9. **contrast** — the same allocator, same size, under R0248's `memory.max`
       is killed. One child, plain CPU, no CUDA anywhere near it.
    10. **no in-process guard could have acted** — the control limit is orders
        of magnitude below the registered in-process anonymous budget, and the
        child's receipt records that it armed nothing and imported no array
        library.
    """
    verify_registry(label="R0249 memory.high throttle control")
    os.makedirs(workspace, exist_ok=True)
    availability = require_external_memory_mode(mode)

    started = time.time()
    stem = f"r0249-high-control-{os.getpid()}"
    throttled_receipt = os.path.join(workspace, "throttled-child.json")
    quiet_receipt = os.path.join(workspace, "under-allocating-child.json")
    killed_receipt = os.path.join(workspace, "memory-max-contrast-child.json")
    unthrottled_target = int(high_bytes * CONTROL_UNTHROTTLED_FRACTION)

    def _allocator_argv(
        receipt: str, target: int, *, free_target: int, budget_s: float,
    ) -> list[str]:
        return [
            sys.executable, "-c", _ALLOCATOR, str(int(target)),
            str(CONTROL_STEP_BYTES), receipt, str(int(free_target)),
            f"{budget_s:.3f}",
        ]

    # -- arms 1-7: the over-allocator under memory.high --------------------- #
    unit_throttled = f"{stem}-throttled"
    throttled = _run_child(
        _scope_argv(
            unit=unit_throttled, properties=_high_properties(high_bytes),
            mode=mode,
        )
        + _allocator_argv(
            throttled_receipt, CONTROL_TARGET_BYTES,
            free_target=unthrottled_target,
            budget_s=CONTROL_THROTTLE_WALL_BUDGET_S,
        )
    )
    throttled_child = _load(throttled_receipt)
    journal = _journal_oom_lines(unit_throttled, started)

    # -- arm 8: the under-allocator, same limit ----------------------------- #
    quiet = _run_child(
        _scope_argv(
            unit=f"{stem}-under", properties=_high_properties(high_bytes),
            mode=mode,
        )
        + _allocator_argv(
            quiet_receipt, CONTROL_SAFE_BYTES, free_target=CONTROL_SAFE_BYTES,
            budget_s=1.0,
        )
    )
    quiet_child = _load(quiet_receipt)

    # -- arm 9: the contrast, under R0248's memory.max ---------------------- #
    unit_killed = f"{stem}-memory-max-contrast"
    killed = _run_child(
        _scope_argv(
            unit=unit_killed, properties=_max_properties(high_bytes), mode=mode,
        )
        + _allocator_argv(
            killed_receipt, CONTROL_TARGET_BYTES,
            free_target=unthrottled_target,
            budget_s=CONTROL_THROTTLE_WALL_BUDGET_S,
        )
    )

    high_before = int(
        (throttled_child.get("memory_events_before") or {}).get("high", 0)
    )
    high_after = int(
        (throttled_child.get("memory_events_after") or {}).get("high", 0)
    )
    quiet_high_delta = int(
        (quiet_child.get("memory_events_after") or {}).get("high", 0)
    ) - int((quiet_child.get("memory_events_before") or {}).get("high", 0))
    after = throttled_child.get("memory_events_after") or {}
    rate_ratio = throttled_child.get("throttled_rate_over_unthrottled_rate")
    registered_budget = int(
        registered_value("max_declared_anonymous_budget_bytes")
    )

    arms = {
        "the_over_allocating_child_survived": bool(throttled.returncode == 0),
        "the_kernel_throttled_it": bool(high_after > high_before),
        "the_throttle_collapsed_the_allocation_rate": bool(
            rate_ratio is not None
            and float(rate_ratio) <= CONTROL_MAX_THROTTLED_RATE_RATIO
        ),
        "the_cgroup_was_held_at_its_high_watermark": bool(
            int(throttled_child.get("memory_current") or 0) >= int(high_bytes)
        ),
        "nothing_was_oom_killed_in_the_scope": bool(
            int(after.get("oom", 0)) == 0 and int(after.get("oom_kill", 0)) == 0
        ),
        "the_kernel_journal_names_no_oom_kill_for_this_scope": bool(
            not journal["the_kernel_oom_killed_this_scope"]
        ),
        "no_memory_max_kill_limit_was_set": bool(
            throttled_child.get("memory_max") == "max"
        ),
        "swap_is_pinned_at_zero": bool(
            throttled_child.get("memory_swap_max") == "0"
        ),
        "an_under_allocating_child_does_not_trip_the_limit": bool(
            quiet.returncode == 0 and quiet_high_delta == 0
        ),
        "the_same_allocator_under_memory_max_is_killed": bool(
            killed.returncode in (-9, 137)
        ),
        "the_in_process_guard_was_not_armed": bool(
            throttled_child.get("in_process_guard_armed") is False
        ),
        "no_cuda_or_array_module_was_loaded_in_any_child": bool(
            throttled_child.get("cuda_modules_imported") == []
        ),
        "the_in_process_budget_would_not_have_fired": bool(
            int(high_bytes) < registered_budget
        ),
    }
    evidence = {
        "control": "round0249-memory-high-throttle-control-v1",
        "mode": mode,
        "availability": availability,
        "memory_high_bytes": int(high_bytes),
        "memory_swap_max_bytes": 0,
        "target_bytes": CONTROL_TARGET_BYTES,
        "safe_bytes": CONTROL_SAFE_BYTES,
        "unthrottled_phase_target_bytes": unthrottled_target,
        "throttled_phase_wall_budget_s": CONTROL_THROTTLE_WALL_BUDGET_S,
        "max_permitted_throttled_rate_ratio": CONTROL_MAX_THROTTLED_RATE_RATIO,
        "registered_in_process_anonymous_budget_bytes": registered_budget,
        "throttled_arm": {
            "unit": unit_throttled,
            "returncode": throttled.returncode,
            "stdout_tail": throttled.stdout[-400:],
            "stderr_tail": throttled.stderr[-400:],
            "memory_events_high_before": high_before,
            "memory_events_high_after": high_after,
            "memory_events_high_delta": high_after - high_before,
            "unthrottled_bytes_per_s": throttled_child.get(
                "unthrottled_bytes_per_s"
            ),
            "throttled_bytes_per_s": throttled_child.get(
                "throttled_bytes_per_s"
            ),
            "throttled_rate_over_unthrottled_rate": rate_ratio,
            "reached_the_request": throttled_child.get("reached_the_request"),
            "child_receipt": throttled_child,
        },
        "under_allocating_arm": {
            "returncode": quiet.returncode,
            "memory_events_high_delta": quiet_high_delta,
            "child_receipt": quiet_child,
        },
        "memory_max_contrast_arm": {
            "unit": unit_killed,
            "returncode": killed.returncode,
            "killed_by_signal": bool(killed.returncode in (-9, 137)),
            "what_it_shows": (
                "the identical allocator at the identical size under R0248's "
                "MemoryMax. This is the behaviour R0249 moved away from, "
                "measured rather than asserted, on a plain CPU child that "
                "holds no CUDA context"
            ),
        },
        "kernel_journal": journal,
        "arms": arms,
        "failures": [name for name, ok in arms.items() if not ok],
        "note": MEMORY_HIGH_NOTE,
        "what_this_does_not_cover": (
            "poll spacing, abort latency, GPU memory, and receipt honesty. A "
            "node under a memory.high can still publish a receipt it invented. "
            "The only mitigation for that is independent recomputation by a "
            "reviewer who did not run the round"
        ),
        "what_the_switch_costs": (
            "memory.high does not STOP anything. A node that ignores its own "
            "cooperative guard and keeps allocating is slowed, not halted, and "
            "is then bounded only by the machine. That is the price of never "
            "OOM-killing a CUDA holder"
        ),
        "registry_fingerprint": registry_fingerprint(),
    }
    evidence["holds"] = not evidence["failures"]
    if evidence["failures"]:
        raise Round0249Error(
            "R0249 MEMORY.HIGH THROTTLE CONTROL DID NOT FIRE: "
            f"{evidence['failures']}. throttled rc={throttled.returncode}, "
            f"under rc={quiet.returncode}, contrast rc={killed.returncode}, "
            f"high {high_before}->{high_after}, rate ratio {rate_ratio}"
        )
    return evidence


# --------------------------------------------------------------------------- #
# the escape battery, and the refusal to downgrade
# --------------------------------------------------------------------------- #
def run_escape_battery(
    *, workspace: str, mode: str = DEFAULT_EXTERNAL_MEMORY_MODE,
    high_bytes: int = CONTROL_MEMORY_HIGH_BYTES,
    also_run_the_other_mode: bool = True,
) -> dict[str, Any]:
    """Five attempts from inside the scope, in the default mode and the other.

    Under `root-scope` all five must fail. The `user-scope` run is published
    whether or not it succeeds — it does succeed, that is exactly why the
    default moved — because reporting only the mode that wins is the shape
    three reviews have punished.
    """
    verify_registry(label="R0249 escape battery")
    os.makedirs(workspace, exist_ok=True)
    require_external_memory_mode(mode)
    stem = f"r0249-escape-{os.getpid()}"

    def _battery(this_mode: str) -> dict[str, Any]:
        receipt = os.path.join(workspace, f"escape-{this_mode}.json")
        completed = _run_child(
            _scope_argv(
                unit=f"{stem}-{this_mode}",
                properties=_high_properties(high_bytes), mode=this_mode,
            )
            + [sys.executable, "-c", _ESCAPES, receipt],
            #: a `user-scope` scope needs the session bus, which a root-scope
            #: node's environment does not carry. Derived, never invented.
            env=_user_scope_environment() if this_mode != "root-scope" else None,
        )
        rows = _load(receipt)
        attempts = rows.get("attempts") or []
        succeeded = [row for row in attempts if row.get("succeeded")]
        ran = len(attempts) >= _ESCAPE_ATTEMPTS
        return {
            "mode": this_mode,
            "ran": ran,
            "returncode": completed.returncode,
            "stderr_tail": completed.stderr[-300:],
            "attempts": attempts,
            "attempts_run": len(attempts),
            "successful_escapes": succeeded,
            #: `None`, not `False`, when nothing ran. A battery that produced
            #: no attempts says NOTHING about whether the mode is defeatable,
            #: and publishing `false` there is a claim of safety the run did
            #: not earn.
            "the_node_can_defeat_this_mode": bool(succeeded) if ran else None,
            "memory_high_after_the_attempts": rows.get(
                "memory_high_after_the_attempts"
            ),
            "memory_max_after_the_attempts": rows.get(
                "memory_max_after_the_attempts"
            ),
            "memory_swap_max_after_the_attempts": rows.get(
                "memory_swap_max_after_the_attempts"
            ),
        }

    default_mode = _battery(mode)
    other = "user-scope" if mode == "root-scope" else "root-scope"
    other_mode: dict[str, Any] = {
        "mode": other, "ran": False, "attempts": [], "attempts_run": 0,
        "the_node_can_defeat_this_mode": None,
        "why_it_did_not_run": "this box cannot place a scope in that mode",
    }
    if also_run_the_other_mode and external_memory_mode_availability(
        other
    )["available"]:
        other_mode = _battery(other)

    arms = {
        "the_battery_ran_in_the_default_mode": bool(
            default_mode["attempts_run"] >= _ESCAPE_ATTEMPTS
        ),
        "no_escape_succeeds_in_the_default_mode": bool(
            not default_mode["successful_escapes"]
        ),
        "the_memory_high_limit_survived_the_attempts": bool(
            default_mode["memory_high_after_the_attempts"]
            not in (None, "max")
        ),
        #: R0249 self-attack: this arm used to read `bool(other_mode["ran"])`
        #: and `ran` was set by the CALLER, so a battery that died on
        #: "Failed to connect to bus" published `attempts: []` and still passed
        #: — with `the_node_can_defeat_this_mode: false` beside it. It now
        #: requires the full battery to have actually executed.
        "the_other_mode_was_measured_and_published": bool(
            other_mode["attempts_run"] >= _ESCAPE_ATTEMPTS
        ),
        #: and the losing mode must LOSE. `user-scope` is defeatable; a run
        #: that says otherwise has not run the battery.
        "the_other_mode_result_is_not_a_silent_empty": bool(
            other_mode["the_node_can_defeat_this_mode"] is not None
        ),
    }
    evidence = {
        "control": "round0249-escape-battery-v1",
        "default_mode": default_mode,
        "the_other_mode": other_mode,
        "why_the_default_moved": (
            "review-0248-01 §E.6. Under user-scope the scope's cgroup "
            "directory is delegated to the user, so two writes lift the bound "
            "and leave no override record, no fingerprint mismatch and no "
            "receipt field - weaker than the Python read-only properties this "
            "bound was added to outrank. Under root-scope every write is EACCES"
        ),
        "arms": arms,
        "failures": [name for name, ok in arms.items() if not ok],
        "registry_fingerprint": registry_fingerprint(),
    }
    evidence["holds"] = not evidence["failures"]
    if evidence["failures"]:
        raise Round0249Error(
            f"R0249 ESCAPE BATTERY DID NOT FIRE: {evidence['failures']}. "
            f"default {default_mode.get('successful_escapes')}"
        )
    return evidence


def run_fail_closed_control() -> dict[str, Any]:
    """A mode that cannot be placed REFUSES; it does not become a weaker mode.

    The planted defect is R0248's own line
    `if mode == "root-scope" and not availability[...]: mode = "user-scope"`,
    which turned an unavailable strong mode into a silently weaker one. Here
    the same situation is presented three ways and every one must raise.
    """
    verify_registry(label="R0249 fail-closed control")
    rows: list[dict[str, Any]] = []

    def _attempt(name: str, **kwargs: Any) -> None:
        try:
            state = require_external_memory_mode(**kwargs)
            rows.append({
                "attempt": name, "refused": False,
                "returned_mode": state.get("mode"), "error": None,
            })
        except Round0249Error as error:
            rows.append({
                "attempt": name, "refused": True, "returned_mode": None,
                "error": str(error)[:220],
            })

    #: 1. an unplaceable root-scope, presented exactly as R0248 saw it
    _attempt(
        "root_scope_without_passwordless_sudo",
        mode="root-scope",
        availability={
            "mode": "root-scope", "available": False,
            "why": "root-scope needs passwordless sudo -n and it is unavailable",
            "base": {},
        },
    )
    #: 2. a box with no cgroup v2 memory controller at all
    _attempt(
        "no_cgroup_v2_memory_controller",
        mode="root-scope",
        availability={
            "mode": "root-scope", "available": False,
            "why": "this box cannot place a cgroup v2 memory limit at all",
            "base": {},
        },
    )
    #: 3. a mode nobody has implemented
    _attempt("an_unknown_mode", mode="best-effort")

    #: and the negative half: the mode this box CAN place is not refused.
    placeable = external_memory_mode_availability(DEFAULT_EXTERNAL_MEMORY_MODE)
    accepted = None
    if placeable["available"]:
        accepted = require_external_memory_mode(
            DEFAULT_EXTERNAL_MEMORY_MODE
        )["mode"]

    arms = {
        "every_unplaceable_mode_is_refused": bool(
            all(row["refused"] for row in rows)
        ),
        "no_refusal_returned_a_weaker_mode": bool(
            all(row["returned_mode"] is None for row in rows)
        ),
        "a_placeable_mode_is_still_accepted": bool(
            accepted == DEFAULT_EXTERNAL_MEMORY_MODE
            or not placeable["available"]
        ),
    }
    evidence = {
        "control": "round0249-fail-closed-mode-control-v1",
        "attempts": rows,
        "the_mode_this_box_can_place": placeable,
        "accepted_mode": accepted,
        "planted": (
            "the R0248 downgrade: an unavailable root-scope silently becoming "
            "user-scope, which is a bound the node can lift in two writes"
        ),
        "arms": arms,
        "failures": [name for name, ok in arms.items() if not ok],
        "registry_fingerprint": registry_fingerprint(),
    }
    evidence["holds"] = not evidence["failures"]
    if evidence["failures"]:
        raise Round0249Error(
            f"R0249 FAIL-CLOSED CONTROL DID NOT FIRE: {evidence['failures']}"
        )
    return evidence


def sudo_is_available() -> bool:
    if not shutil.which("sudo"):
        return False
    return subprocess.run(
        ["sudo", "-n", "true"], check=False, capture_output=True
    ).returncode == 0


__all__ = [
    "CONTROL_MEMORY_HIGH_BYTES",
    "CONTROL_SAFE_BYTES",
    "CONTROL_TARGET_BYTES",
    "DEFAULT_EXTERNAL_MEMORY_MODE",
    "KNOWN_EXTERNAL_MEMORY_MODES",
    "MEMORY_HIGH_NOTE",
    "ROUND_ID",
    "Round0249Error",
    "cgroup_self_report",
    "external_memory_limit_declaration",
    "external_memory_mode_availability",
    "own_cgroup_directory",
    "require_external_memory_mode",
    "run_escape_battery",
    "run_fail_closed_control",
    "run_memory_high_throttle_control",
    "sudo_is_available",
]
