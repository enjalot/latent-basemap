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
from basemap.round0247_registry import (
    REGISTERED_SAFETY_PARAMETERS,
    clamp,
    override_records,
    registered_bounds,
    registered_value,
    registry_fingerprint,
    verify_registry,
)
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

#: review-0245-01 A1-bis. `_loop` caught `OSError` FIRST and `pass`ed, so the
#: reviewer's planted `OSError` left the thread alive, sampling nothing, with
#: `sampling_thread_alive: true` and `poll()` quiet. `/proc` reads do not fail
#: on this box, but a guard must not be unfalsifiable, so the tolerance is a
#: REGISTERED number rather than "forever".
#:
#: Basis — the same inequality the whole poll-spacing argument rests on. `n`
#: consecutive failures leave the guard blind for `n * WATCHDOG_SAMPLE_INTERVAL_S`
#: seconds, which at R0244's own measured `11,767,996,416` B/s permits
#: `n * 0.25 * 11767996416` B of growth. Against R0244's sealed
#: `budget_headroom_bytes` of `29,548,888,064` B the largest admissible `n` is
#: `29548888064 / (0.25 * 11767996416) = 10.04`, i.e. `10`. The registered value
#: is `3`, which is `3.35x` stricter than the derived maximum — it is not a
#: number chosen to let anything pass.
#:
#: R0247: this is now a *mirror* of the registry entry, not the authority.
#: `_sample_once` reads `REGISTERED_SAFETY_PARAMETERS` directly, so assigning
#: `round0244_guard.WATCHDOG_MAX_CONSECUTIVE_SAMPLE_FAILURES = 10**9` from an
#: importer - the module-global form of the override class - no longer moves the
#: tolerance. `tests/test_round0247_contract.py` plants exactly that.
WATCHDOG_MAX_CONSECUTIVE_SAMPLE_FAILURES = int(
    REGISTERED_SAFETY_PARAMETERS["watchdog_max_consecutive_sample_failures"].value
)

#: The NEW arm. R0242's rule is conjunctive (swap growth AND pressure) and is
#: inherited unchanged — it is the machine-level rule and this round does not
#: touch it. What it cannot express is a stage exceeding the footprint its own
#: architecture predicts while the machine is still comfortable, which is
#: exactly the state R0243's symmetrisation was in and exactly the state a
#: `10 h` trainer would be in on the way to swap. A node declares the budget it
#: designed for; exceeding it is a defect in the node, not in the machine.
#:
#: R0247 (REGISTERED CHANGE, direction: TIGHTER). This was `64 GiB`, which sits
#: ABOVE R0242's own `60 GiB` anonymous pressure threshold - so a node taking
#: the default had a budget arm that could never fire before the machine rule
#: did, i.e. no budget arm at all. The default is now the threshold itself,
#: which is also `max_declared_anonymous_budget_bytes` in the R0247 registry.
#: Basis: a node-declared budget above the level at which the machine is in
#: trouble is not a budget. Nothing that passed before can fail now except a
#: stage whose anonymous peak is between `60` and `64 GiB`, which on a `123 GiB`
#: box is already inside R0242's pressure arm.
WATCHDOG_DEFAULT_ANON_BUDGET_BYTES = int(
    REGISTERED_SAFETY_PARAMETERS["max_declared_anonymous_budget_bytes"].value
)

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
        verify_registry(label=f"R0244 watchdog for {label}")
        #: review-0246-01 A: `interval_s` was a keyword nothing gated on, and a
        #: node declaring `5.0` reported `thread_sample_coverage 0.9994` while
        #: buying `1.99x` the sealed headroom. It is clamped at the registered
        #: `0.25` s and the attempt is recorded; the receipt reports the
        #: REGISTERED interval under `registered_*` and the declared one under
        #: `declared_*`, and the coverage denominator is derived from the
        #: registered value and measured wall time, never from this argument.
        interval_effective, interval_record = clamp(
            "watchdog_sample_interval_s", interval_s,
            site="ThreadedHostWatchdog(interval_s=)", label=str(label),
        )
        budget_effective, budget_record = clamp(
            "max_declared_anonymous_budget_bytes", anonymous_budget_bytes,
            site="ThreadedHostWatchdog(anonymous_budget_bytes=)",
            label=str(label),
        )
        self.safety_overrides = override_records(
            [interval_record, budget_record]
        )
        self.anonymous_budget_bytes = int(budget_effective)
        self.declared_anonymous_budget_bytes = int(anonymous_budget_bytes)
        self.interval_s = float(interval_effective)
        self.declared_interval_s = float(interval_s)
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
        #: review-0245-01 A1-bis: `samples` counts BOTH the thread's samples and
        #: the ones `poll()` takes at stage boundaries, so `sample_coverage`
        #: could be held high by a main thread polling often behind a sampler
        #: that is doing nothing. These count the thread's work only, and they
        #: are what the coverage floor is evaluated on.
        self.thread_samples = 0
        self.sample_failures = 0
        self.consecutive_sample_failures = 0
        self.max_consecutive_sample_failures = 0
        #: review-0246-01 A, the denominator defect. Coverage is a RATIO to an
        #: interval the caller declares, so it is not a bound. These are the
        #: same quantity in seconds of measured wall clock: the widest and the
        #: mean interval between two successful THREAD samples, timestamped by
        #: the sampling thread itself. Nothing a caller declares enters them.
        self.max_thread_sample_gap_s = 0.0
        self.gap_after_the_last_thread_sample_s: float | None = None
        self._last_thread_sample_s: float | None = None
        self._trace: dict[int, int] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- the thread ------------------------------------------------------- #
    def start(self) -> "ThreadedHostWatchdog":
        if self._thread is not None:
            raise Round0244Error("R0244 watchdog is already sampling")
        self.started_at_s = _now()
        #: The first observation gap runs from stage start, not from the first
        #: sample - the same anchoring the poll gate uses, for the same reason.
        self._last_thread_sample_s = self.started_at_s
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
        #: And the last observation gap runs to stage end, so a sampler that
        #: stops early cannot hide the tail interval.
        with self._lock:
            if self._last_thread_sample_s is not None:
                tail = float(self.stopped_at_s - self._last_thread_sample_s)
                self.gap_after_the_last_thread_sample_s = tail
                self.max_thread_sample_gap_s = max(
                    self.max_thread_sample_gap_s, tail
                )
        self._thread = None

    def __enter__(self) -> "ThreadedHostWatchdog":
        return self.start()

    def __exit__(self, *_exc: Any) -> None:
        self.stop()

    def _loop(self) -> None:
        """Sample until asked to stop. ANY other exit is a recorded death.

        review-0245-01 A1-bis: the previous form caught `OSError` before the
        `BaseException` arm and `pass`ed forever, so a `/proc` failure left the
        thread alive and sampling nothing with the receipt still reading
        `sampling_thread_alive: true`. Now a bounded number of consecutive
        failures is tolerated, the tolerance is registered, and the `finally`
        makes *every* way out of this function visible — including a return that
        raises nothing at all.
        """
        try:
            while not self._stop.is_set():
                if not self._sample_once():
                    return
                self._stop.wait(self.interval_s)
            self._sample_once()
        finally:
            self._record_exit()

    def _sample_once(self) -> bool:
        """One thread-side observation. `False` means the guard has died."""
        #: R0247: read the tolerance from the REGISTRY, not from this module's
        #: global. Assigning the global was the module-global form of the
        #: override class and it no longer reaches the decision.
        tolerance = int(
            REGISTERED_SAFETY_PARAMETERS[
                "watchdog_max_consecutive_sample_failures"
            ].value
        )
        try:
            self._sample()
        except OSError as error:
            with self._lock:
                self.sample_failures += 1
                self.consecutive_sample_failures += 1
                self.max_consecutive_sample_failures = max(
                    self.max_consecutive_sample_failures,
                    self.consecutive_sample_failures,
                )
                exhausted = self.consecutive_sample_failures > tolerance
            if exhausted:
                self._record_death(
                    error,
                    context=(
                        f"{tolerance + 1} consecutive sampling failures"
                    ),
                )
                return False
            return True
        except BaseException as error:  # noqa: BLE001 - see _record_death
            self._record_death(error)
            return False
        now = _now()
        with self._lock:
            self.thread_samples += 1
            self.consecutive_sample_failures = 0
            if self._last_thread_sample_s is not None:
                gap = float(now - self._last_thread_sample_s)
                if gap > self.max_thread_sample_gap_s:
                    self.max_thread_sample_gap_s = gap
            self._last_thread_sample_s = now
        return True

    def _record_exit(self) -> None:
        """A sampler that returns without being asked to stop is a dead guard."""
        if self._stop.is_set():
            return
        self._record_death(
            RuntimeError("the sampling thread returned on its own"),
            context="an unrequested thread exit",
        )

    def _record_death(
        self, error: BaseException, *, context: str | None = None
    ) -> None:
        """A sampler that stops sampling is a guard that stopped guarding.

        review-0244-01 A1 killed the thread with a `ValueError` and watched it
        vanish: `thread_alive_after_death False`, `samples_recorded 1`, nothing
        raised, nothing logged. The record is written under the lock so
        `poll()` and `receipt()` both see it, and `poll()` raises on it.
        """
        with self._lock:
            if self.thread_death is None:
                cause = (
                    f"{type(error).__name__}: {error}" if context is None
                    else f"{context} ({type(error).__name__}: {error})"
                )
                self.thread_death = (
                    f"{cause}. The R0244 sampling thread stopped sampling; the "
                    "host guard is no longer being evaluated between call "
                    "sites."
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
            #: R0248: read the registered anonymous pressure threshold at the
            #: comparison site. `WATCHDOG_ANON_BYTES` is registered as
            #: `max_declared_anonymous_budget_bytes` and was compared here as a
            #: bare imported module global - the mechanically derived inventory
            #: found this one; no review had.
            pressure = (
                anonymous > registered_value(
                    "max_declared_anonymous_budget_bytes"
                )
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
            #: review-0246-01 A. THE denominator fix: the expected sample count
            #: is derived from measured wall time and the REGISTERED interval,
            #: so a node declaring a wider interval no longer buys itself a
            #: coverage of 0.9994. `expected_samples_at_interval` above is kept
            #: for continuity with R0244/R0245/R0246 receipts and is NOT what
            #: any gate reads.
            registered_interval = float(
                REGISTERED_SAFETY_PARAMETERS[
                    "watchdog_sample_interval_s"
                ].value
            )
            expected_registered = (
                elapsed / registered_interval if registered_interval > 0
                else 0.0
            )
            mean_gap = (
                elapsed / float(self.thread_samples)
                if self.thread_samples > 0 else None
            )
            base.update({
                "instrument": "round0244-threaded-host-watchdog-v2",
                "note": WATCHDOG_NOTE,
                "label": self.label,
                "sampling_thread": True,
                "sample_interval_s": self.interval_s,
                "declared_sample_interval_s": self.declared_interval_s,
                "declared_anonymous_budget_bytes": int(
                    self.declared_anonymous_budget_bytes
                ),
                #: The two denominator-free arms. Both are measured in seconds
                #: of wall clock by the sampling thread itself; neither is a
                #: ratio to anything the node declares.
                "max_thread_sample_gap_s": float(self.max_thread_sample_gap_s),
                "mean_thread_sample_gap_s": mean_gap,
                "gap_after_the_last_thread_sample_s": (
                    self.gap_after_the_last_thread_sample_s
                ),
                "expected_samples_at_the_registered_interval": (
                    expected_registered
                ),
                "thread_sample_coverage_at_the_registered_interval": (
                    None if expected_registered <= 0
                    else float(self.thread_samples) / expected_registered
                ),
                "safety_overrides": [
                    dict(record) for record in self.safety_overrides
                ],
                "registry_fingerprint": registry_fingerprint(),
                **registered_bounds([
                    "watchdog_sample_interval_s",
                    "watchdog_max_consecutive_sample_failures",
                    "watchdog_max_observation_gap_s",
                    "watchdog_max_mean_observation_gap_s",
                    "max_declared_anonymous_budget_bytes",
                ]),
                "samples": int(self.samples),
                "sampled_wall_s": elapsed,
                "expected_samples_at_interval": expected,
                "sample_coverage": (
                    None if expected <= 0 else float(self.samples) / expected
                ),
                #: review-0245-01 A1-bis. `sample_coverage` above counts the
                #: boundary `poll()` samples too, so it cannot be gated on: a
                #: main thread polling often keeps it high behind a sampler
                #: doing nothing. These are the thread's own numbers.
                "thread_samples": int(self.thread_samples),
                "thread_sample_coverage": (
                    None if expected <= 0
                    else float(self.thread_samples) / expected
                ),
                "sample_failures": int(self.sample_failures),
                "max_consecutive_sample_failures": int(
                    self.max_consecutive_sample_failures
                ),
                "max_consecutive_sample_failures_allowed": int(
                    REGISTERED_SAFETY_PARAMETERS[
                        "watchdog_max_consecutive_sample_failures"
                    ].value
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
        #: review-0246-01 G: this emitted `receipt["samples"]` - the
        #: BOUNDARY-INCLUSIVE count - under the name of the thread-only field,
        #: so two different quantities shared a key across two receipts. Both
        #: are now named for what they are.
        "samples_including_boundary_polls": int(receipt["samples"]),
        "thread_samples": int(receipt["thread_samples"]),
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
    "WATCHDOG_MAX_CONSECUTIVE_SAMPLE_FAILURES",
    "WATCHDOG_NOTE",
    "WATCHDOG_SAMPLE_INTERVAL_S",
    "WATCHDOG_TRACE_SECONDS",
    "boundary_only_understatement",
    "run_watchdog_positive_control",
]
