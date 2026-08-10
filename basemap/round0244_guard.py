"""R0244 — a host-memory watchdog that is actually running during the peak.

review-0243-01 §6 located the defect precisely and it is a **safety** gap, not
a reporting one:

* `_HostWatchdog` (R0242) has **no thread and no timer**. It is evaluated only
  where `poll()` is called.
* `run_fuzzy` calls `poll()` at exactly six stage boundaries.
* The stripe loop inside `_fuzzy_symmetrise_blocked` polls the cooperative
  abort flag every stripe and **never polls the memory guard**.

So during `compute_membership_strengths` + the stripe-wise set operation — the
stage that holds roughly `35 GiB` — the conjunctive guard was **not evaluated
at all**. The `12,058,918,912` B the R0243 receipt publishes is not an
under-read of the peak; it is the value at the *preceding boundary*
(`ids_sorted` + `dists`), and the stage's true maximum was never sampled by
anything. Nothing was in danger on that run (`0` swap growth, `MemAvailable`
never below `116.8 GB`) — but the guard could not have known, and the next
thing this program runs is a ~10 h training node whose only protection against
swap is that guard.

This module fixes all three halves of it without editing one byte of R0242:

1. **A real sampling thread.** `ThreadedHostWatchdog` subclasses R0242's
   `_HostWatchdog`, so the conjunctive rule, its thresholds and its receipt
   fields are INHERITED rather than re-typed, and adds a daemon thread that
   samples `/proc/self/status` and `/proc/meminfo` every
   `WATCHDOG_SAMPLE_INTERVAL_S`. The peak is therefore observed inside a stage
   that has no call site at all.
2. **In-stage enforcement, through the mechanism the program already
   sanctions.** A thread cannot raise into the main thread, so observation
   alone would still leave enforcement at the next boundary. On a trip the
   sampler writes the runner's **cooperative abort flag** — the same file
   `<queue_root>/logs/<node>.abort` an operator writes — which the stripe loop
   already polls through `_check_runner_abort` once per stripe. That converts a
   boundary-latency guard into a per-stripe one with **no signal, no ptrace, no
   child, and no edit to a reviewed module**. `poll()` remains the direct
   enforcement call site for code this round owns.
3. **A positive control that is required to fire.**
   `run_watchdog_positive_control` plants an allocation that exceeds a declared
   budget and fails closed unless the thread observes it, the flag lands, and
   `poll()` raises. A guard nobody has seen fire is not evidence — that rule
   has five precedents in this program.

The instrument also publishes, from the same run, **what a boundary-only
instrument would have reported** (`boundary_peak_anonymous_bytes`, updated only
inside `poll()`) beside what the thread saw. That is what makes the comparison
against R0243's `12.06 GB` a measurement rather than an argument.

Nothing here signals anything, starts a child process, touches the GPU, or
imports `cupy`.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any

from basemap.gpu_child_supervision import (
    RUNNER_ABORT_FLAG_ENV,
    runner_abort_reason,
)
from basemap.round0242_locality import host_anonymous_bytes
from experiments.round0242_nodes import (
    WATCHDOG_ANON_BYTES,
    WATCHDOG_MEM_AVAILABLE_BYTES,
    WATCHDOG_SWAP_GROWTH_BYTES,
    _HostWatchdog,
    _meminfo,
    _swap_used,
)

ROUND_ID = "0244"
ROWS = 100_000_000
GPU_HOURS_CAP = 2.0


class Round0244Error(RuntimeError):
    """R0244 fails closed."""


#: How often the sampling thread reads `/proc`. `250 ms` over R0243's
#: `173.3 s` symmetrisation is `~693` samples inside a stage that R0242's
#: instrument sampled `0` times.
WATCHDOG_SAMPLE_INTERVAL_S = 0.25

#: How many samples the receipt carries. The thread keeps a per-second maximum
#: so a `10 h` node's trace stays a receipt rather than a dataset.
WATCHDOG_TRACE_SECONDS = 4_000

#: The NEW arm. R0242's rule is conjunctive (swap growth AND pressure) and is
#: inherited unchanged — it is the machine-level rule and this round does not
#: touch it. What it cannot express is a stage exceeding the footprint its own
#: architecture predicts while the machine is still comfortable, which is
#: exactly the state R0243's symmetrisation was in and exactly the state a
#: `10 h` trainer would be in on the way to swap. A node declares the budget it
#: designed for; exceeding it is a defect in the node, not in the machine.
WATCHDOG_DEFAULT_ANON_BUDGET_BYTES = 64 * (1 << 30)

#: The budget for the R0243 symmetrisation stage, re-run here to measure its
#: true peak. R0243's own architectural accounting is `ids_sorted` (6 GB) +
#: `dists` (6 GB) + `compute_membership_strengths`'s four `1.5e9`-entry outputs
#: (24 GB) + the `np.repeat` layout assertion (6 GB) ~= 42 GB, and the operator
#: observation during the run was `~35 GiB`. `64 GiB` is above both and far
#: below the `~110 GiB` at which this host would begin to swap.
FUZZY_STAGE_ANON_BUDGET_BYTES = 64 * (1 << 30)

WATCHDOG_NOTE = (
    "R0242's conjunctive rule is INHERITED UNCHANGED and still gates on "
    "ANONYMOUS bytes rather than RSS. What R0244 adds is (a) a daemon sampling "
    "thread, so the guard is evaluated INSIDE a stage that contains no call "
    "site, (b) a node-declared anonymous budget arm, and (c) in-stage "
    "ENFORCEMENT by writing the runner's cooperative abort flag, which every "
    "reviewed stripe loop in this release already polls once per stripe. No "
    "signal is involved at any point and no reviewed module is edited."
)


def _now() -> float:
    return time.monotonic()


class ThreadedHostWatchdog(_HostWatchdog):
    """R0242's guard, evaluated continuously instead of at six boundaries."""

    def __init__(
        self,
        *,
        anonymous_budget_bytes: int = WATCHDOG_DEFAULT_ANON_BUDGET_BYTES,
        interval_s: float = WATCHDOG_SAMPLE_INTERVAL_S,
        abort_flag_path: str | None = None,
        label: str = "R0244 node",
    ) -> None:
        super().__init__()
        self.anonymous_budget_bytes = int(anonymous_budget_bytes)
        self.interval_s = float(interval_s)
        self.abort_flag_path = abort_flag_path
        self.label = str(label)
        #: What a boundary-only instrument — R0242's — would have reported from
        #: THIS run. Updated only inside `poll()`.
        self.boundary_peak_anonymous_bytes = 0
        self.boundary_polls = 0
        self.samples = 0
        self.started_at_s: float | None = None
        self.stopped_at_s: float | None = None
        self.trip_reason: str | None = None
        self.trip_rule: str | None = None
        self.trip_anonymous_bytes: int | None = None
        self.trip_observed_after_s: float | None = None
        self.abort_flag_written = False
        self.abort_flag_error: str | None = None
        #: review-0244-01 A1: `_loop` caught only `OSError`, so any other
        #: exception killed the sampler silently and the guard degraded to
        #: R0242's boundary-only behaviour with nothing raised and nothing
        #: logged. A dead sampler must never read as "no problem observed", so
        #: the death is recorded here and `poll()` raises on it.
        self.thread_death: str | None = None
        self._trace: dict[int, int] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- the thread ------------------------------------------------------- #
    def start(self) -> "ThreadedHostWatchdog":
        if self._thread is not None:
            raise Round0244Error("R0244 watchdog is already sampling")
        self.started_at_s = _now()
        self._thread = threading.Thread(
            target=self._loop, name="r0244-host-watchdog", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5.0 + self.interval_s)
        self.stopped_at_s = _now()
        self._thread = None

    def __enter__(self) -> "ThreadedHostWatchdog":
        return self.start()

    def __exit__(self, *_exc: Any) -> None:
        self.stop()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample()
            except OSError:
                #: `/proc` is never unavailable on this box, but a sampling
                #: thread that dies takes the guard with it, which is the
                #: failure this round exists to remove.
                pass
            except BaseException as error:  # noqa: BLE001 - see _record_death
                self._record_death(error)
                return
            self._stop.wait(self.interval_s)
        try:
            self._sample()
        except OSError:
            pass
        except BaseException as error:  # noqa: BLE001 - see _record_death
            self._record_death(error)

    def _record_death(self, error: BaseException) -> None:
        """A sampler that stops sampling is a guard that stopped guarding.

        review-0244-01 A1 killed the thread with a `ValueError` and watched it
        vanish: `thread_alive_after_death False`, `samples_recorded 1`, nothing
        raised, nothing logged. The record is written under the lock so
        `poll()` and `receipt()` both see it, and `poll()` raises on it.
        """
        with self._lock:
            if self.thread_death is None:
                self.thread_death = (
                    f"{type(error).__name__}: {error}. The R0244 sampling "
                    "thread stopped sampling; the host guard is no longer "
                    "being evaluated between call sites."
                )

    def _sample(self) -> dict[str, int]:
        """One observation. Never raises into the main thread."""
        host = host_anonymous_bytes()
        values = _meminfo()
        anonymous = int(host["anonymous_bytes"])
        rss = int(host["rss_bytes"])
        swap_growth = _swap_used(values) - self.baseline_swap_bytes
        available = int(values.get("MemAvailable", 0))
        with self._lock:
            self.samples += 1
            self.peak_anonymous_bytes = max(self.peak_anonymous_bytes, anonymous)
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
            self.peak_swap_growth_bytes = max(
                self.peak_swap_growth_bytes, swap_growth
            )
            self.minimum_mem_available_bytes = min(
                self.minimum_mem_available_bytes, available
            )
            if self.started_at_s is not None:
                second = int(_now() - self.started_at_s)
                if second < WATCHDOG_TRACE_SECONDS:
                    self._trace[second] = max(
                        self._trace.get(second, 0), anonymous
                    )
            pressure = (
                anonymous > WATCHDOG_ANON_BYTES
                or available < WATCHDOG_MEM_AVAILABLE_BYTES
            )
            conjunctive = swap_growth > WATCHDOG_SWAP_GROWTH_BYTES and pressure
            over_budget = anonymous > self.anonymous_budget_bytes
            if (conjunctive or over_budget) and not self.fired:
                self.fired = True
                self.trip_rule = (
                    "conjunctive-swap-and-pressure" if conjunctive
                    else "node-declared-anonymous-budget"
                )
                self.trip_anonymous_bytes = anonymous
                self.trip_observed_after_s = (
                    None if self.started_at_s is None
                    else _now() - self.started_at_s
                )
                self.trip_reason = (
                    f"R0244 host watchdog tripped in-stage on {self.label} "
                    f"({self.trip_rule}): anonymous {anonymous} B against a "
                    f"budget of {self.anonymous_budget_bytes} B, swap growth "
                    f"{swap_growth} B, MemAvailable {available} B. Observed by "
                    f"the sampling thread, not at a stage boundary."
                )
                self._request_cooperative_abort()
        return {
            "anonymous_bytes": anonymous,
            "rss_bytes": rss,
            "swap_growth_bytes": swap_growth,
            "mem_available_bytes": available,
        }

    def _request_cooperative_abort(self) -> None:
        """Ask, in band, through the flag every reviewed stripe loop polls."""
        path = self.abort_flag_path
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(self.trip_reason or "R0244 host watchdog tripped")
            self.abort_flag_written = True
        except OSError as error:  # pragma: no cover - filesystem failure
            self.abort_flag_error = str(error)

    # -- the enforcement call site ---------------------------------------- #
    def poll(self, where: str) -> None:
        """Enforce. Also records what a boundary-only instrument would see."""
        sample = self._sample()
        with self._lock:
            self.polls += 1
            self.boundary_polls += 1
            self.boundary_peak_anonymous_bytes = max(
                self.boundary_peak_anonymous_bytes, sample["anonymous_bytes"]
            )
            fired = self.fired
            reason = self.trip_reason
            death = self.thread_death
        if death is not None:
            raise Round0244Error(
                f"R0244 host watchdog SAMPLING THREAD DIED: {death} Observed "
                f"at {where}. A dead sampler does not mean 'no problem "
                "observed'; the node must stop rather than continue behind a "
                "guard that is no longer running."
            )
        if fired:
            raise Round0244Error(
                f"{reason} Observed at {where}. Unwinding in band; no signal "
                "was delivered and no child process exists."
            )

    def raise_if_fired(self, where: str) -> None:
        with self._lock:
            fired = self.fired
            reason = self.trip_reason
        if fired:
            raise Round0244Error(f"{reason} Observed at {where}.")

    def raise_if_thread_died(self, where: str) -> None:
        """The visible-failure half of the A1 fix, callable without sampling."""
        with self._lock:
            death = self.thread_death
        if death is not None:
            raise Round0244Error(
                f"R0244 host watchdog SAMPLING THREAD DIED: {death} Observed "
                f"at {where}."
            )

    # -- evidence ---------------------------------------------------------- #
    def receipt(self) -> dict[str, Any]:
        base = super().receipt()
        with self._lock:
            trace = sorted(self._trace.items())
            elapsed = (
                0.0 if self.started_at_s is None
                else float((self.stopped_at_s or _now()) - self.started_at_s)
            )
            expected = elapsed / self.interval_s if self.interval_s > 0 else 0.0
            base.update({
                "instrument": "round0244-threaded-host-watchdog-v1",
                "note": WATCHDOG_NOTE,
                "label": self.label,
                "sampling_thread": True,
                "sample_interval_s": self.interval_s,
                "samples": int(self.samples),
                "sampled_wall_s": elapsed,
                "expected_samples_at_interval": expected,
                "sample_coverage": (
                    None if expected <= 0 else float(self.samples) / expected
                ),
                #: review-0244-01 A1. `sampling_thread_alive` is the field a
                #: node gates on; `thread_death` is why. Both are False/None on
                #: a healthy run, so a receipt that carries a death cannot be
                #: read as a clean one.
                "sampling_thread_alive": bool(self.thread_death is None),
                "thread_death": self.thread_death,
                "boundary_polls": int(self.boundary_polls),
                "boundary_peak_anonymous_bytes": int(
                    self.boundary_peak_anonymous_bytes
                ),
                "thread_peak_anonymous_bytes": int(self.peak_anonymous_bytes),
                "anonymous_budget_bytes": int(self.anonymous_budget_bytes),
                "budget_headroom_bytes": int(
                    self.anonymous_budget_bytes - self.peak_anonymous_bytes
                ),
                "trip_rule": self.trip_rule,
                "trip_reason": self.trip_reason,
                "trip_anonymous_bytes": self.trip_anonymous_bytes,
                "trip_observed_after_s": self.trip_observed_after_s,
                "abort_flag_path": self.abort_flag_path,
                "abort_flag_written": bool(self.abort_flag_written),
                "abort_flag_error": self.abort_flag_error,
                "anonymous_trace_by_second": [
                    [int(second), int(value)] for second, value in trace
                ],
            })
        return base


def boundary_only_understatement(receipt: dict[str, Any]) -> dict[str, Any]:
    """How much a boundary-only instrument would have missed, measured.

    This is the quantity review-0243-01 asked for: the same run, sampled two
    ways, so the `12.06 GB` R0243 published can be compared against the true
    peak without either number being an argument.
    """
    thread_peak = int(receipt["thread_peak_anonymous_bytes"])
    boundary_peak = int(receipt["boundary_peak_anonymous_bytes"])
    return {
        "thread_peak_anonymous_bytes": thread_peak,
        "boundary_peak_anonymous_bytes": boundary_peak,
        "understatement_bytes": thread_peak - boundary_peak,
        "understatement_multiple": (
            None if boundary_peak <= 0 else thread_peak / float(boundary_peak)
        ),
        "boundary_polls": int(receipt["boundary_polls"]),
        "thread_samples": int(receipt["samples"]),
        "note": (
            "both columns come from ONE run of the same stage. The boundary "
            "column is updated only inside poll(), which is what R0242's "
            "instrument does; the thread column is updated every "
            f"{receipt['sample_interval_s']} s regardless of call sites."
        ),
    }


def run_watchdog_positive_control(
    *,
    flag_path: str,
    plant_bytes: int = 1 << 30,
    headroom_bytes: int = 512 << 20,
    interval_s: float = 0.05,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """Plant an over-budget allocation and PROVE the guard catches it.

    Fails closed: this function raises unless the sampling thread observes the
    planted allocation, the cooperative abort flag lands on disk,
    `runner_abort_reason()` reports it (the exact mechanism by which a reviewed
    stripe loop unwinds), and `poll()` then raises.

    The allocation is real anonymous memory — a written `uint8` buffer — not a
    mocked `/proc` reading, because a guard proven against a stub is not proven.
    """
    import numpy as np

    if runner_abort_reason() is not None:
        raise Round0244Error(
            "R0244 will not run its watchdog positive control while a real "
            "cooperative abort is already requested"
        )
    baseline = int(host_anonymous_bytes()["anonymous_bytes"])
    budget = baseline + int(headroom_bytes)
    watchdog = ThreadedHostWatchdog(
        anonymous_budget_bytes=budget,
        interval_s=float(interval_s),
        abort_flag_path=flag_path,
        label="R0244 watchdog positive control",
    )
    saved_env = os.environ.get(RUNNER_ABORT_FLAG_ENV)
    planted: Any = None
    started = _now()
    try:
        os.environ[RUNNER_ABORT_FLAG_ENV] = flag_path
        watchdog.start()
        planted = np.ones(int(plant_bytes), dtype=np.uint8)
        planted[:] = 7
        deadline = started + float(timeout_s)
        while _now() < deadline:
            with watchdog._lock:  # noqa: SLF001 - the control inspects its subject
                if watchdog.fired:
                    break
            time.sleep(interval_s / 2.0)
        observed_after_s = _now() - started
        anonymous_with_plant = int(host_anonymous_bytes()["anonymous_bytes"])
        flag_reason = runner_abort_reason()
        poll_raised = False
        poll_message = None
        try:
            watchdog.poll("R0244 positive control enforcement site")
        except Round0244Error as error:
            poll_raised = True
            poll_message = str(error)
        loop_raised = _stripe_loop_control(watchdog=watchdog)
        receipt = watchdog.receipt()
    finally:
        watchdog.stop()
        planted = None
        if saved_env is None:
            os.environ.pop(RUNNER_ABORT_FLAG_ENV, None)
        else:
            os.environ[RUNNER_ABORT_FLAG_ENV] = saved_env
        try:
            os.remove(flag_path)
        except OSError:
            pass

    evidence = {
        "control": "round0244-watchdog-positive-control-v1",
        "planted_bytes": int(plant_bytes),
        "baseline_anonymous_bytes": baseline,
        "declared_budget_bytes": int(budget),
        "anonymous_with_plant_bytes": anonymous_with_plant,
        "budget_exceeded_by_bytes": anonymous_with_plant - int(budget),
        "watchdog_fired": bool(receipt["fired"]),
        "trip_rule": receipt["trip_rule"],
        "trip_anonymous_bytes": receipt["trip_anonymous_bytes"],
        "observed_after_s": observed_after_s,
        "thread_observed_after_s": receipt["trip_observed_after_s"],
        "sample_interval_s": float(interval_s),
        "samples_taken": int(receipt["samples"]),
        "abort_flag_written": bool(receipt["abort_flag_written"]),
        "abort_flag_reason_readable": flag_reason is not None,
        "poll_raised": poll_raised,
        "poll_message": poll_message,
        "stripe_loop_unwound_at_unit": loop_raised,
        "timeout_s": float(timeout_s),
    }
    failures = [
        name for name, ok in (
            ("watchdog_fired", evidence["watchdog_fired"]),
            ("budget_exceeded", evidence["budget_exceeded_by_bytes"] > 0),
            ("abort_flag_written", evidence["abort_flag_written"]),
            ("abort_flag_readable", evidence["abort_flag_reason_readable"]),
            ("poll_raised", evidence["poll_raised"]),
            ("stripe_loop_unwound", loop_raised is not None),
        ) if not ok
    ]
    evidence["failures"] = failures
    evidence["holds"] = not failures
    if failures:
        raise Round0244Error(
            "R0244 watchdog POSITIVE CONTROL DID NOT FIRE: "
            f"{failures}. A guard that cannot be shown catching a planted "
            "defect is untested at its only job, and no multi-hour node may "
            "run behind it."
        )
    return evidence


def _stripe_loop_control(*, watchdog: ThreadedHostWatchdog) -> int | None:
    """A stripe-shaped loop that polls the flag exactly as R0238's does.

    Returns the unit index at which it unwound, or `None` if it ran to
    completion. This is the half that proves ENFORCEMENT reaches inside a
    stage: the loop below never touches the watchdog object at all, it only
    calls `runner_abort_reason()`, which is what
    `experiments.round0238_nodes._check_runner_abort` calls once per stripe.
    """
    for unit in range(50):
        if runner_abort_reason() is not None:
            return unit
    watchdog.raise_if_fired("R0244 stripe-loop control completed")
    return None


__all__ = [
    "FUZZY_STAGE_ANON_BUDGET_BYTES",
    "GPU_HOURS_CAP",
    "ROUND_ID",
    "ROWS",
    "Round0244Error",
    "ThreadedHostWatchdog",
    "WATCHDOG_DEFAULT_ANON_BUDGET_BYTES",
    "WATCHDOG_NOTE",
    "WATCHDOG_SAMPLE_INTERVAL_S",
    "WATCHDOG_TRACE_SECONDS",
    "boundary_only_understatement",
    "run_watchdog_positive_control",
]
