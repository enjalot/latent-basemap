"""Signal-free supervision for children that may hold a CUDA context.

Why this module exists
----------------------
`subprocess.run(..., timeout=N)` is implemented in CPython as `Popen.kill()` on
expiry, and `Popen.kill()` is `send_signal(SIGKILL)`. A `SIGKILL` delivered to a
process inside a CUDA/UVM call is what deadlocked RCU and put PID 1 into `D`
state twice on this box, each time costing a reboot. R0238's imbalance grid ran
under `subprocess.run(..., timeout=3600)`, the bound **expired**, and a `SIGKILL`
reached a live cuML child; the machine survived by luck and both of R0238's
results claimed no signal had been delivered.

This module is the replacement. It is the build path's supervision pattern —
already reviewed and certified correct at R0233 through R0238 — factored out so
that *every* GPU child gets it, not only the build child.

The contract
------------
- The parent uses `Popen` and `communicate()` **with no timeout**, so CPython
  has no code path that signals the child.
- The bound is expressed as a **cooperative flag file**. A watchdog thread
  writes the flag when the deadline passes; the child polls it between units of
  work and unwinds its own CUDA context through the normal Python path.
- The parent never calls `kill()`, `terminate()`, `send_signal()`, `os.kill()`
  or `os.killpg()`, and never attaches `ptrace`/`py-spy` to the child.
- A child that ignores the flag is **left running**. That is deliberate: on this
  driver a hung CUDA child is recoverable and a `SIGKILL`ed one may not be.

`emit_child_abort_preamble()` generates the child half of the contract: the
polling helper the generated probe scripts call between cells.
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
from typing import Any, Callable, Sequence


__all__ = [
    "CooperativeChildResult",
    "emit_child_abort_preamble",
    "run_gpu_child_cooperative",
]


class CooperativeChildResult:
    """What a signal-free supervised child produced.

    Deliberately shaped like `subprocess.CompletedProcess` (`returncode`,
    `stdout`, `stderr`) so call sites read the same, plus the fields that prove
    no signal was delivered.
    """

    def __init__(
        self,
        *,
        returncode: int,
        stdout: str,
        stderr: str,
        elapsed_s: float,
        deadline_s: float,
        flag_written: bool,
        flag_written_at_s: float | None,
        escalations: list[str],
        io_readings: dict[str, Any] | None = None,
        snapshot_before: dict[str, Any] | None = None,
        snapshot_after: dict[str, Any] | None = None,
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.elapsed_s = elapsed_s
        self.deadline_s = deadline_s
        self.flag_written = flag_written
        self.flag_written_at_s = flag_written_at_s
        self.escalations = escalations
        self.io_readings = io_readings
        self.snapshot_before = snapshot_before
        self.snapshot_after = snapshot_after

    def supervision_receipt(self) -> dict[str, Any]:
        """The evidence a result may quote about how this child was bounded."""
        return {
            "abort_policy": (
                "in-band cooperative flag only; no SIGTERM, no SIGKILL, no ptrace"
            ),
            "bound_mechanism": "cooperative-flag-file",
            "deadline_s": self.deadline_s,
            "elapsed_s": self.elapsed_s,
            "cooperative_flag_written": self.flag_written,
            "cooperative_flag_written_at_s": self.flag_written_at_s,
            "watchdog_escalations": list(self.escalations),
            "signal_delivered": False,
        }

    def io_receipt(self) -> dict[str, Any]:
        """The instruments R0238's acceptance check 14 registered and never got.

        Check 14 asked for child `read_bytes`, `/data` device counters and
        `mincore` substrate residency before and after the cell. Those existed
        only inside the build-child path, which never ran, so the check was
        structurally unable to produce a value. Every cooperatively supervised
        child now carries them, or records plainly that it was not instrumented
        — `instrumented: false` is a visible absence rather than a silent one.
        """
        instrumented = (
            self.io_readings is not None
            or self.snapshot_before is not None
            or self.snapshot_after is not None
        )
        return {
            "instrumented": instrumented,
            "child_io": self.io_readings,
            "snapshot_before": self.snapshot_before,
            "snapshot_after": self.snapshot_after,
        }


class _CooperativeDeadline(threading.Thread):
    """Writes a flag file when the deadline passes. Never signals anything.

    This class has no `os.kill`, no `terminate()`, no `kill()` and no
    `send_signal()`, and `check_signal_safety.py` asserts that mechanically.
    """

    def __init__(
        self,
        *,
        flag_path: str,
        deadline_s: float,
        poll_s: float,
        process: subprocess.Popen,
    ) -> None:
        super().__init__(daemon=True, name="cooperative-deadline")
        self._flag_path = flag_path
        self._deadline_s = float(deadline_s)
        self._poll = float(poll_s)
        self._process = process
        self._halt = threading.Event()
        self.flag_written = False
        self.flag_written_at_s: float | None = None
        self.escalations: list[str] = []

    def _trip(self, reason: str) -> None:
        """Write the cooperative flag. This is the only escalation that exists."""
        if self.flag_written:
            return
        tmp = f"{self._flag_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(reason)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self._flag_path)
        self.flag_written = True
        self.escalations.append("cooperative-flag")

    def run(self) -> None:
        started = time.monotonic()
        while not self._halt.is_set():
            if self._process.poll() is not None:
                return
            elapsed = time.monotonic() - started
            if elapsed >= self._deadline_s and not self.flag_written:
                self.flag_written_at_s = elapsed
                self._trip(
                    f"cell exceeded its {self._deadline_s:.0f} s deadline; "
                    "cooperative flag set, no signal sent"
                )
            self._halt.wait(self._poll)

    def halt(self) -> None:
        self._halt.set()
        self.join(timeout=self._poll * 2 + 5.0)


def emit_child_abort_preamble(flag_path: str) -> str:
    """The child half of the contract, for injection into a generated script.

    The generated probe scripts call `_check_abort()` between cells. On seeing
    the flag the child raises, unwinds, drops its CUDA context through the
    normal interpreter path and writes whatever it completed — which is strictly
    more than a `SIGKILL` would have left behind.
    """
    return f'''
_ABORT_FLAG = {flag_path!r}


class _CooperativeAbort(Exception):
    """The parent asked us to stop. Raised so the CUDA context unwinds normally."""


def _check_abort():
    """Poll the cooperative flag. The parent never signals; this is the bound."""
    if os.path.exists(_ABORT_FLAG):
        raise _CooperativeAbort(
            "cooperative abort flag observed; unwinding this child's CUDA "
            "context through the normal path"
        )
'''


def run_gpu_child_cooperative(
    argv: Sequence[str],
    *,
    cwd: str,
    env: dict[str, str],
    flag_path: str,
    deadline_s: float,
    poll_s: float = 1.0,
    sampler_factory: Callable[[int], Any] | None = None,
    snapshot: Callable[[], dict[str, Any]] | None = None,
) -> CooperativeChildResult:
    """Run a child that may hold a CUDA context, bounded without any signal.

    Replaces `subprocess.run(argv, timeout=deadline_s)`. The bound still exists
    — it is just expressed cooperatively, so the child tears its own context
    down instead of being `SIGKILL`ed inside a UVM call.

    `sampler_factory(pid)` returns an object with `start()`, `halt()` and
    `readings()` — R0237/R0238's `_IoSampler`. `snapshot()` is called once
    before launch and once after reap; it is where `/data` device counters and
    `mincore` residency are captured. Both are optional so this module stays
    free of any dependency on `experiments/`, and both are recorded in
    `io_receipt()` so an uninstrumented cell is visibly uninstrumented.
    """
    if os.path.exists(flag_path):
        os.unlink(flag_path)
    snapshot_before = snapshot() if snapshot is not None else None
    started = time.monotonic()
    process = subprocess.Popen(
        list(argv),
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    watchdog = _CooperativeDeadline(
        flag_path=flag_path,
        deadline_s=deadline_s,
        poll_s=poll_s,
        process=process,
    )
    sampler = sampler_factory(process.pid) if sampler_factory is not None else None
    watchdog.start()
    if sampler is not None:
        sampler.start()
    try:
        # No `timeout=`: this is the whole point. CPython has no path from here
        # that signals the child.
        stdout, stderr = process.communicate()
    finally:
        if sampler is not None:
            sampler.halt()
        watchdog.halt()
    return CooperativeChildResult(
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
        elapsed_s=time.monotonic() - started,
        deadline_s=float(deadline_s),
        flag_written=watchdog.flag_written,
        flag_written_at_s=watchdog.flag_written_at_s,
        escalations=list(watchdog.escalations),
        io_readings=sampler.readings() if sampler is not None else None,
        snapshot_before=snapshot_before,
        snapshot_after=snapshot() if snapshot is not None else None,
    )
