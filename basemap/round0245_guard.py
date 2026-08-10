"""R0245 — close the three guard defects that stand between us and a ~10 h node.

review-0244-01 accepted R0244 and named seven defects. Three of them are guard
defects, and all three are the same failure class this program keeps shipping:
a protection whose *absence* is silent.

1. **A1 — the sampler thread can die without a trace.** `_loop` caught only
   `OSError`; the reviewer subclassed the watchdog so `_sample()` raised a
   `ValueError` and watched `thread_alive_after_death False`,
   `samples_recorded 1`, nothing raised and nothing logged. The guard silently
   reverted to R0242's boundary-only behaviour and no node gated on the
   `sample_coverage` field that would have revealed it. Fixed at source in
   `basemap/round0244_guard.py` (catch `BaseException`, record the death, raise
   from `poll()`, publish `sampling_thread_alive`); the positive control here
   plants the death and proves the failure surfaces.

2. **A3 — in-stage enforcement is silently optional.** R0244's re-measurement
   stage took `abort_flag_path=os.environ.get("ROUNDRUN_ABORT_FLAG")` and
   asserted nothing. With the variable unset the guard trips, writes nothing,
   and no stripe loop can ever learn of it — the reviewer reproduced exactly
   that state (`fired True`, `abort_flag_written False`, `abort_flag_error
   None`). `require_enforceable_abort_flag()` fails closed *before* a node
   starts, and `EnforcedHostWatchdog` refuses to be constructed without a
   writable flag path.

3. **A2 — the binding number is stripe latency times allocation slope, not the
   sampling interval.** R0244 reported `0.25 s` sampling and disclosed a
   `3.485 s` enforcement latency, but never multiplied that latency by the
   slope its own sealed trace exhibits. At `11,767,996,416 B/s` a `3.485 s`
   stripe permits `~41 GB` of growth *after* the guard has decided to stop,
   against `29,548,888,064 B` of declared headroom. The safety number is

       max_poll_spacing_s = headroom_bytes / slope_bytes_per_s

   which for R0243's own stage is `2.5109 s` — smaller than the stripe. So the
   fix is the *poll spacing*, not a faster sampler: sampling faster narrows the
   observation gap and does nothing at all to the enforcement gap.
   `require_poll_spacing()` is the executable form, and the positive control
   runs a synthetic stage allocating faster than the measured slope and shows
   the guard stopping it inside the headroom — and shows the same stage
   breaching the headroom when the spacing violates the requirement.

Nothing here signals anything, starts a child process, or touches the GPU;
no GPU array library is imported on any path. The guard axis stays **anonymous, never RSS**, and R0242's
conjunctive machine rule and all three of its thresholds are inherited
unchanged: this module adds no threshold and changes no rule.
"""
from __future__ import annotations

import os
import threading
import time
from collections.abc import Mapping
from typing import Any

from basemap.gpu_child_supervision import (
    RUNNER_ABORT_FLAG_ENV,
    runner_abort_reason,
)
from basemap.round0242_locality import host_anonymous_bytes
from basemap.round0244_guard import (
    ThreadedHostWatchdog,
    WATCHDOG_SAMPLE_INTERVAL_S,
)
from basemap.round0247_registry import (
    clamp,
    is_registered_abort_reader,
    override_records,
    verify_registry,
)
from experiments.round0242_nodes import _meminfo

ROUND_ID = "0245"
ROWS = 100_000_000
GPU_HOURS_CAP = 1.0


class Round0245Error(RuntimeError):
    """R0245 fails closed."""


# --------------------------------------------------------------------------- #
# A2 — the binding number, derived from R0244's own sealed trace
# --------------------------------------------------------------------------- #
#: The steepest one-second rise in `anonymous_trace_by_second` of R0244's sealed
#: `host-watchdog.json` (seconds 37 -> 38 of the symmetrisation). The node
#: RECOMPUTES this from the receipt and refuses to publish if it disagrees, so
#: the constant is a registration rather than a transcription.
R0244_MEASURED_SLOPE_BYTES_PER_S = 11_767_996_416

#: `budget_headroom_bytes` from the same receipt: the node declared
#: `68,719,476,736` B and the thread measured a peak of `39,170,588,672` B.
R0244_BUDGET_HEADROOM_BYTES = 29_548_888_064

#: `stage_walls.fuzzy_symmetrisation_s / fuzzy.stripes` from the same receipt.
#: This is the spacing at which `_check_runner_abort` — the only in-stage
#: enforcement site that exists — is reached.
R0244_STRIPE_LATENCY_S = 174.2581459750072 / 50.0

#: The floor a synthetic slope control must beat to be a control for THIS
#: stage. Anything slower proves the guard against a defect gentler than the
#: one already measured.
SLOPE_CONTROL_MIN_BYTES_PER_S = float(R0244_MEASURED_SLOPE_BYTES_PER_S)

#: review-0245-01 D. The correction to R0245 bound the gate to the stage's own
#: measured slope with a `max(slope, 1.0)` B/s floor, and the reviewer replayed
#: attempt 1's `5.828025072987657` s gap through it at `measured_slope = 0.0`:
#: the binding slope became `1.0` B/s, the permitted spacing `47,244,640,256` s,
#: and the gate that had just fired on its author's code passed the same gap.
#: A floor of `1.0` B/s is a number chosen so that nothing fails.
#:
#: The registered minimum binding slope is instead the fastest allocation slope
#: this program has ever MEASURED — the steepest one-second rise in R0244's
#: sealed `anonymous_trace_by_second`, `11,767,996,416` B/s. Basis: a stage's
#: own differentiated trace is a lower bound on what it could allocate (bursts
#: inside one trace bin are invisible to a one-second difference, which is
#: exactly why the DiD node measured `0.0`), so the gate must bind on at least
#: the fastest rate observed on this machine. A stage that really does allocate
#: faster is scored on its own larger number; nothing is ever scored on less.
MIN_BINDING_SLOPE_BYTES_PER_S = float(R0244_MEASURED_SLOPE_BYTES_PER_S)

MIN_BINDING_SLOPE_BASIS = (
    "the minimum binding slope is 11,767,996,416 B/s, the steepest one-second "
    "rise in R0244's sealed anonymous_trace_by_second (seconds 37 -> 38 of the "
    "R0243 symmetrisation). It replaces the max(slope, 1.0) B/s floor R0245's "
    "correction introduced, under which a stage whose differentiated trace is "
    "flat was permitted 47,244,640,256 s between cooperative-abort reads and "
    "the gate could not fail. A one-second difference cannot see a burst inside "
    "a single bin, so a measured 0.0 B/s is not evidence of a slow stage; the "
    "worst rate this machine has been measured at is the floor."
)


def binding_slope_bytes_per_s(measured_slope_bytes_per_s: float) -> float:
    """The slope a gate binds on: the stage's own, never below the worst case."""
    return max(
        float(measured_slope_bytes_per_s), MIN_BINDING_SLOPE_BYTES_PER_S
    )

POLL_SPACING_NOTE = (
    "the guard's safety number is enforcement latency times allocation slope, "
    "not the sampling interval. Sampling every 0.25 s bounds how late the trip "
    "is OBSERVED; it does not bound how much a stage allocates between the "
    "trip and the next site that can act on it. For a node that declares an "
    "anonymous budget, the requirement is slope * poll_spacing <= headroom, "
    "i.e. poll_spacing <= headroom / slope. Sampling faster does not help; "
    "polling the flag more often inside the unit does, and so does declaring a "
    "smaller budget."
)


def poll_spacing_requirement(
    *,
    slope_bytes_per_s: float,
    headroom_bytes: int,
    poll_spacing_s: float,
) -> dict[str, Any]:
    """`poll_spacing <= headroom / slope`, computed rather than asserted.

    `headroom_bytes` is what may still be allocated after the guard has decided
    to stop without harm — for a node declaring an anonymous budget that is the
    distance from the budget to the level at which the machine is in trouble,
    and R0244 published it as `budget_headroom_bytes`.
    """
    slope = float(slope_bytes_per_s)
    headroom = float(headroom_bytes)
    spacing = float(poll_spacing_s)
    if slope <= 0.0 or headroom <= 0.0 or spacing <= 0.0:
        raise Round0245Error(
            "R0245 poll-spacing requirement needs a positive slope, headroom "
            f"and spacing, got {slope}, {headroom}, {spacing}"
        )
    max_spacing = headroom / slope
    permitted = slope * spacing
    return {
        "slope_bytes_per_s": slope,
        "headroom_bytes": int(headroom_bytes),
        "poll_spacing_s": spacing,
        "max_poll_spacing_s": max_spacing,
        "permitted_growth_after_trip_bytes": permitted,
        "permitted_growth_over_headroom_multiple": permitted / headroom,
        "spacing_over_requirement_multiple": spacing / max_spacing,
        "minimum_polls_per_unit": int(-(-spacing // max_spacing)),
        "requirement_holds": bool(spacing <= max_spacing),
        "note": POLL_SPACING_NOTE,
    }


def require_poll_spacing(
    *,
    slope_bytes_per_s: float,
    headroom_bytes: int,
    poll_spacing_s: float,
    label: str,
) -> dict[str, Any]:
    """The executable form. A node that cannot meet it must not start."""
    verdict = poll_spacing_requirement(
        slope_bytes_per_s=slope_bytes_per_s,
        headroom_bytes=headroom_bytes,
        poll_spacing_s=poll_spacing_s,
    )
    if not verdict["requirement_holds"]:
        raise Round0245Error(
            f"R0245 REFUSES {label}: its guard is polled every "
            f"{verdict['poll_spacing_s']} s, which at the measured slope of "
            f"{verdict['slope_bytes_per_s']} B/s permits "
            f"{verdict['permitted_growth_after_trip_bytes']:.0f} B of growth "
            f"after the trip against {verdict['headroom_bytes']} B of "
            f"headroom. Poll at most every {verdict['max_poll_spacing_s']} s "
            f"({verdict['minimum_polls_per_unit']} polls per unit) or declare "
            "a smaller budget."
        )
    return verdict


class AbortPollTracker:
    """Times the spacing of cooperative-abort reads — the enforcement sites.

    The watchdog's sampling thread *writes* the flag; a stage acts on it only
    where it *reads* the flag, which in every reviewed stage in this release is
    `_check_runner_abort`. So the enforcement spacing is the spacing of those
    reads, not the spacing of `poll()` and certainly not the sampling interval.
    Wrapping the abort check is what makes that spacing a measurement.
    """

    def __init__(
        self,
        *,
        inner: Any,
        headroom_bytes: int,
        label: str,
        slope_bytes_per_s: float = SLOPE_CONTROL_MIN_BYTES_PER_S,
        clock: Any = None,
        replay: bool = False,
    ) -> None:
        verify_registry(label=f"R0245 poll tracker for {label}")
        #: An injectable clock so a gap can be REPLAYED exactly rather than
        #: slept through. R0246 replays attempt 1's 5.828025072987657 s gap
        #: through this class with a scripted clock; the default is the real one.
        #:
        #: R0247 attack a2-r0247-1: a scripted clock is a construction path that
        #: replaces the measurement itself. `clock=lambda: 0.0` scores every gap
        #: at zero and the gate reports a perfect spacing for a stage that
        #: allocated for hours. A scripted clock is now permitted only on a gate
        #: explicitly declared `replay=True`, whose verdict is marked
        #: `replay_only` and which a node may not seal as enforcement evidence.
        self.replay = bool(replay)
        self.clock_is_the_registered_monotonic_clock = bool(clock is None)
        self._clock = clock if clock is not None else time.monotonic
        self.inner = inner
        #: R0247 attack a2-r0247-2: the gate times calls to ITSELF, so a gate
        #: wrapping a no-op publishes a flawless spacing for a stage that never
        #: reads the cooperative abort flag. `inner` must be a registered abort
        #: reader on any gate that is not a replay.
        self.inner_is_a_registered_abort_reader = bool(
            is_registered_abort_reader(inner)
        )
        #: review-0246-01 C and D3. Both of these were keywords with registered
        #: defaults and no bound, and the reviewer combined them: a `1 << 50`
        #: headroom with an overridden ceiling passed R0245's own blocker gap.
        headroom_effective, headroom_record = clamp(
            "max_declared_headroom_bytes", headroom_bytes,
            site="AbortPollTracker(headroom_bytes=)", label=str(label),
        )
        slope_effective, slope_record = clamp(
            "min_binding_slope_bytes_per_s", slope_bytes_per_s,
            site="AbortPollTracker(slope_bytes_per_s=)", label=str(label),
        )
        self.safety_overrides = override_records([headroom_record, slope_record])
        self.declared_headroom_bytes = int(headroom_bytes)
        self.declared_slope_bytes_per_s = float(slope_bytes_per_s)
        self.headroom_bytes = int(headroom_effective)
        self.slope_bytes_per_s = float(slope_effective)
        self.label = str(label)
        self.polls = 0
        self.max_gap_s = 0.0
        self.total_gap_s = 0.0
        self.max_gap_at: str | None = None
        self._last: float | None = None

    def __call__(self, where: str) -> None:
        self._record(float(self._clock()), where)

    def _record(self, now: float, where: str) -> None:
        """The bookkeeping, separated so a subclass can read the clock once."""
        if self._last is not None:
            gap = now - self._last
            self.total_gap_s += gap
            if gap > self.max_gap_s:
                self.max_gap_s = gap
                self.max_gap_at = str(where)
        self._last = now
        self.polls += 1
        self.inner(where)

    def verdict(
        self, *, measured_slope_bytes_per_s: float | None = None
    ) -> dict[str, Any]:
        """Score the widest gap twice: at the stage's OWN slope and at the
        worst slope this program has ever measured.

        The safety condition is `slope * spacing <= headroom` with the slope
        that actually applies, so the stage's own differentiated anonymous
        trace is the number the gate must use. R0244's `11,767,996,416` B/s is
        a different stage's slope; it is reported beside it as the
        worst-case bound a future ALLOCATING stage would have to meet, and a
        stage that exceeds it says so in its receipt rather than being scored
        against a slope it never exhibited.
        """
        spacing = self.max_gap_s if self.max_gap_s > 0.0 else 1e-9
        worst_case = poll_spacing_requirement(
            slope_bytes_per_s=self.slope_bytes_per_s,
            headroom_bytes=self.headroom_bytes,
            poll_spacing_s=spacing,
        )
        own = None
        if measured_slope_bytes_per_s is not None:
            own = poll_spacing_requirement(
                slope_bytes_per_s=binding_slope_bytes_per_s(
                    measured_slope_bytes_per_s
                ),
                headroom_bytes=self.headroom_bytes,
                poll_spacing_s=spacing,
            )
        return {
            "label": self.label,
            #: R0247: what the caller ASKED for is published beside what the
            #: gate actually bound on, so an override is visible in the receipt
            #: instead of being echoed under a `registered_*` name.
            "declared_headroom_bytes": int(self.declared_headroom_bytes),
            "effective_headroom_bytes": int(self.headroom_bytes),
            "declared_worst_case_slope_bytes_per_s": float(
                self.declared_slope_bytes_per_s
            ),
            "effective_worst_case_slope_bytes_per_s": float(
                self.slope_bytes_per_s
            ),
            "safety_overrides": [
                dict(record) for record in self.safety_overrides
            ],
            "clock_is_the_registered_monotonic_clock": bool(
                self.clock_is_the_registered_monotonic_clock
            ),
            "inner_is_a_registered_abort_reader": bool(
                self.inner_is_a_registered_abort_reader
            ),
            "replay_only": bool(self.replay),
            "binding_slope_floor_bytes_per_s": MIN_BINDING_SLOPE_BYTES_PER_S,
            "binding_slope_floor_basis": MIN_BINDING_SLOPE_BASIS,
            "enforcement_polls": self.polls,
            "max_gap_between_enforcement_polls_s": self.max_gap_s,
            "mean_gap_between_enforcement_polls_s": (
                self.total_gap_s / float(self.polls - 1) if self.polls > 1
                else 0.0
            ),
            "widest_gap_before": self.max_gap_at,
            "measured_slope_bytes_per_s": (
                None if measured_slope_bytes_per_s is None
                else float(measured_slope_bytes_per_s)
            ),
            "requirement": own if own is not None else worst_case,
            "own_slope_requirement": own,
            "worst_case_requirement": worst_case,
            "meets_the_r0244_worst_case_slope": bool(
                worst_case["requirement_holds"]
            ),
            "note": POLL_SPACING_NOTE,
        }

    def require(
        self, *, measured_slope_bytes_per_s: float | None = None
    ) -> dict[str, Any]:
        """Fail closed on the stage's own slope. A stage that does not also
        meet the worst-case bound keeps the fact in its receipt."""
        verdict = self.verdict(
            measured_slope_bytes_per_s=measured_slope_bytes_per_s
        )
        binding = verdict["requirement"]
        if not binding["requirement_holds"]:
            raise Round0245Error(
                f"R0245 STOP: {self.label} left "
                f"{verdict['max_gap_between_enforcement_polls_s']} s between "
                "cooperative-abort reads, which at "
                f"{binding['slope_bytes_per_s']} B/s permits "
                f"{binding['permitted_growth_after_trip_bytes']:.0f}"
                f" B of growth against {self.headroom_bytes} B of headroom. "
                "The widest gap followed: " + str(verdict["widest_gap_before"])
            )
        return verdict


def r0244_stripe_verdict() -> dict[str, Any]:
    """R0243's own stage, scored against the requirement it did not compute."""
    verdict = poll_spacing_requirement(
        slope_bytes_per_s=R0244_MEASURED_SLOPE_BYTES_PER_S,
        headroom_bytes=R0244_BUDGET_HEADROOM_BYTES,
        poll_spacing_s=R0244_STRIPE_LATENCY_S,
    )
    verdict.update({
        "stage": "R0243/R0244 fuzzy symmetrisation, 50 stripes",
        "sampling_interval_s": WATCHDOG_SAMPLE_INTERVAL_S,
        "observation_gap_bytes": (
            float(R0244_MEASURED_SLOPE_BYTES_PER_S) * WATCHDOG_SAMPLE_INTERVAL_S
        ),
        "why_the_interval_is_not_the_number": POLL_SPACING_NOTE,
    })
    return verdict


def slope_from_trace(trace: Any) -> dict[str, Any]:
    """Steepest one-second rise in a sealed `anonymous_trace_by_second`.

    R0244 published the trace and never differentiated it. This is the
    differentiation, run against the sealed bytes so the registered constant is
    checkable rather than transcribed.
    """
    points = [(int(second), int(value)) for second, value in trace]
    if len(points) < 2:
        raise Round0245Error("R0245 slope needs at least two trace points")
    points.sort()
    best_rise = 0
    best_at = points[0][0]
    for earlier, later in zip(points, points[1:]):
        span = later[0] - earlier[0]
        if span <= 0:
            continue
        rise = (later[1] - earlier[1]) / span
        if rise > best_rise:
            best_rise = rise
            best_at = earlier[0]
    return {
        "trace_points": len(points),
        "max_rise_bytes_per_s": float(best_rise),
        "max_rise_starts_at_second": int(best_at),
        "peak_bytes": int(max(value for _second, value in points)),
    }


# --------------------------------------------------------------------------- #
# A3 — enforcement is a precondition, not an environment coincidence
# --------------------------------------------------------------------------- #
ABORT_FLAG_NOTE = (
    "review-0244-01 A3: the re-measurement stage read ROUNDRUN_ABORT_FLAG with "
    "os.environ.get and asserted nothing, so with the variable unset the guard "
    "would trip, write nothing, and no stripe loop could learn of it. Any node "
    "that declares a budget must prove, before it starts, that its trip can "
    "reach an enforcement site."
)


ABORT_FLAG_PROBE_NOTE = (
    "review-0245-01 A3-bis: the precondition validated os.path.dirname(path) "
    "and never the path itself, so ROUNDRUN_ABORT_FLAG pointed at an existing "
    "writable DIRECTORY passed it, the watchdog armed, the guard tripped, and "
    "abort_flag_written came back false with '[Errno 21] Is a directory'. "
    "Existence of a writable directory is not writability of a path. This "
    "creates and removes a probe file AT THE PATH, and refuses anything that "
    "already exists and is not a writable regular file - a directory, a FIFO "
    "(which would also have blocked the trip's open() forever), a device node, "
    "a dangling symlink, or a read-only file."
)


def probe_abort_flag_path(
    path: str, *, label: str = "R0245 node"
) -> dict[str, Any]:
    """Prove the trip can be WRITTEN here, do not infer it from a directory."""
    try:
        if os.path.lexists(path):
            if not os.path.isfile(path):
                raise Round0245Error(
                    f"R0245 REFUSES to start {label}: the cooperative abort "
                    f"flag path {path!r} already exists and is not a regular "
                    "file, so a host-guard trip could not be written to it. "
                    + ABORT_FLAG_PROBE_NOTE
                )
            if not os.access(path, os.W_OK):
                raise Round0245Error(
                    f"R0245 REFUSES to start {label}: the cooperative abort "
                    f"flag path {path!r} exists and is not writable, so a "
                    "host-guard trip could not be written to it. "
                    + ABORT_FLAG_PROBE_NOTE
                )
            return {
                "abort_flag_path": path,
                "path_is_writable": True,
                "probe": "an existing writable regular file was left in place",
                "probe_file_created_and_removed": False,
                "an_abort_is_already_pending_at_this_path": True,
                "note": ABORT_FLAG_PROBE_NOTE,
            }
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.write(handle, b"R0246 abort-flag writability probe\n")
        finally:
            os.close(handle)
        os.remove(path)
    except OSError as error:
        raise Round0245Error(
            f"R0245 REFUSES to start {label}: the cooperative abort flag path "
            f"{path!r} could not be written ({error}), so a host-guard trip "
            "would land nowhere and the node would run unprotected. "
            + ABORT_FLAG_PROBE_NOTE
        ) from error
    return {
        "abort_flag_path": path,
        "path_is_writable": True,
        "probe": "a probe file was created at the path and removed",
        "probe_file_created_and_removed": True,
        "an_abort_is_already_pending_at_this_path": False,
        "note": ABORT_FLAG_PROBE_NOTE,
    }


def require_enforceable_abort_flag(
    *, env: str = RUNNER_ABORT_FLAG_ENV, label: str = "R0245 node"
) -> dict[str, Any]:
    """Fail closed and loudly when the trip has nowhere to land.

    The runner publishes `<queue_root>/logs/<node>.abort` to every node. If the
    variable is missing, empty, or points somewhere unwritable, the guard is
    observation-only and the node must refuse to start rather than run
    unprotected for hours.
    """
    path = os.environ.get(env)
    if not path:
        raise Round0245Error(
            f"R0245 REFUSES to start {label}: {env} is unset, so a host-guard "
            "trip could write no cooperative abort flag and no stripe loop "
            "could ever act on it. The node would run unprotected. "
            + ABORT_FLAG_NOTE
        )
    directory = os.path.dirname(path) or "."
    if not os.path.isdir(directory):
        raise Round0245Error(
            f"R0245 REFUSES to start {label}: {env} is {path!r} but its "
            f"directory {directory!r} does not exist, so the trip cannot land."
        )
    if not os.access(directory, os.W_OK):
        raise Round0245Error(
            f"R0245 REFUSES to start {label}: {env} is {path!r} and its "
            f"directory {directory!r} is not writable, so the trip cannot land."
        )
    probe = probe_abort_flag_path(path, label=label)
    return {
        "env": env,
        "abort_flag_path": path,
        "directory_is_writable": bool(os.access(directory, os.W_OK)),
        "path_is_writable": bool(probe["path_is_writable"]),
        "path_writability_probe": probe,
        "an_abort_is_already_pending": bool(runner_abort_reason() is not None),
        "note": ABORT_FLAG_NOTE,
        "path_probe_note": ABORT_FLAG_PROBE_NOTE,
    }


class EnforcedHostWatchdog(ThreadedHostWatchdog):
    """R0244's threaded guard, which now cannot be built unenforceable.

    Everything else is inherited: the conjunctive machine rule, its three
    thresholds, the anonymous axis, the sampling thread, the trip record and
    the receipt. The only difference is that `abort_flag_path` is a required,
    validated precondition instead of an `os.environ.get` that may be `None`.
    """

    def __init__(self, *, abort_flag_path: str | None = None, **kwargs: Any) -> None:
        resolved = abort_flag_path or os.environ.get(RUNNER_ABORT_FLAG_ENV)
        label = str(kwargs.get("label") or "R0245 node")
        if not resolved:
            raise Round0245Error(
                f"R0245 REFUSES to arm a host watchdog for {label} with no "
                "cooperative abort flag path: a trip that writes nothing "
                "cannot stop a stripe loop. " + ABORT_FLAG_NOTE
            )
        directory = os.path.dirname(resolved) or "."
        os.makedirs(directory, exist_ok=True)
        if not os.access(directory, os.W_OK):
            raise Round0245Error(
                f"R0245 REFUSES to arm a host watchdog for {label}: "
                f"{directory!r} is not writable, so a trip cannot land."
            )
        #: review-0245-01 A3-bis: arming is the last moment before the guard is
        #: relied on, so the path is probed HERE too and not only in the node
        #: precondition. A directory, FIFO, device node or read-only file at
        #: this path refuses the arm instead of producing `fired True /
        #: abort_flag_written False` hours later.
        self.abort_flag_path_probe = probe_abort_flag_path(resolved, label=label)
        super().__init__(abort_flag_path=resolved, **kwargs)
        self.enforcement_is_a_precondition = True


# --------------------------------------------------------------------------- #
# the three positive controls
# --------------------------------------------------------------------------- #
class _DyingWatchdog(EnforcedHostWatchdog):
    """A watchdog whose sampler raises a NON-`OSError` on its second sample.

    This is the reviewer's own attack, made a first-class control: `ValueError`
    is a plausible `/proc` parse failure and it is exactly what R0244's `_loop`
    did not catch.
    """

    def __init__(self, *, die_after: int = 1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.die_after = int(die_after)
        self._sample_calls = 0

    def _sample(self) -> dict[str, int]:
        self._sample_calls += 1
        if self._sample_calls > self.die_after and threading.current_thread() is not \
                threading.main_thread():
            raise ValueError(
                "planted /proc parse failure (R0245 thread-death control)"
            )
        return super()._sample()


def run_thread_death_positive_control(
    *, flag_path: str, timeout_s: float = 30.0, interval_s: float = 0.05
) -> dict[str, Any]:
    """Kill the sampler with a non-`OSError` and prove the failure surfaces.

    Fails closed unless: the thread stops sampling, the receipt says
    `sampling_thread_alive` is false and carries the death, and `poll()` raises
    rather than returning quietly. Before R0245 the same plant produced
    `poll() still raises: True` only when the budget had ALREADY been crossed —
    with a healthy budget it returned silently and the node ran on.
    """
    watchdog = _DyingWatchdog(
        die_after=1,
        anonymous_budget_bytes=1 << 62,
        interval_s=float(interval_s),
        abort_flag_path=flag_path,
        label="R0245 thread-death positive control",
    )
    started = time.monotonic()
    watchdog.start()
    deadline = started + float(timeout_s)
    while time.monotonic() < deadline:
        with watchdog._lock:  # noqa: SLF001 - the control inspects its subject
            if watchdog.thread_death is not None:
                break
        time.sleep(interval_s / 2.0)
    thread = watchdog._thread  # noqa: SLF001 - the control inspects its subject
    thread_alive = bool(thread is not None and thread.is_alive())
    samples_after_death = int(watchdog.samples)
    poll_raised = False
    poll_message = None
    try:
        watchdog.poll("R0245 thread-death control enforcement site")
    except Exception as error:  # noqa: BLE001 - the raise is the evidence
        poll_raised = True
        poll_message = str(error)
    receipt = watchdog.receipt()
    watchdog.stop()
    try:
        os.remove(flag_path)
    except OSError:
        pass

    evidence = {
        "control": "round0245-sampler-thread-death-positive-control-v1",
        "planted": "ValueError raised inside the sampling thread's _sample()",
        "observed_after_s": time.monotonic() - started,
        "thread_death_recorded": bool(receipt["thread_death"] is not None),
        "thread_death": receipt["thread_death"],
        "thread_still_running": thread_alive,
        "receipt_says_sampler_alive": bool(receipt["sampling_thread_alive"]),
        "samples_before_the_death": samples_after_death,
        "poll_raised": poll_raised,
        "poll_message": poll_message,
        "budget_was_never_crossed": bool(receipt["fired"] is False),
        "why": (
            "a guard whose sampler can vanish without a trace is the failure "
            "class the guard exists to prevent. The plant leaves the budget "
            "untouched, so nothing but the death itself can make the node stop."
        ),
    }
    failures = [
        name for name, ok in (
            ("thread_death_recorded", evidence["thread_death_recorded"]),
            ("thread_stopped", not evidence["thread_still_running"]),
            ("receipt_reports_a_dead_sampler",
             not evidence["receipt_says_sampler_alive"]),
            ("poll_raised", evidence["poll_raised"]),
            ("death_not_masked_by_a_trip",
             evidence["budget_was_never_crossed"]),
        ) if not ok
    ]
    evidence["failures"] = failures
    evidence["holds"] = not failures
    if failures:
        raise Round0245Error(
            "R0245 THREAD-DEATH POSITIVE CONTROL DID NOT FIRE: "
            f"{failures}. A sampler that can die silently makes every "
            "downstream memory receipt unfalsifiable."
        )
    return evidence


def run_missing_flag_positive_control(*, label: str = "R0245 node") -> dict[str, Any]:
    """Unset `ROUNDRUN_ABORT_FLAG` and prove the node refuses to start.

    Restores the environment in a `finally`, exactly as R0244's control does.
    """
    saved = os.environ.get(RUNNER_ABORT_FLAG_ENV)
    precondition_raised = False
    precondition_message = None
    watchdog_raised = False
    watchdog_message = None
    try:
        os.environ.pop(RUNNER_ABORT_FLAG_ENV, None)
        try:
            require_enforceable_abort_flag(label=label)
        except Round0245Error as error:
            precondition_raised = True
            precondition_message = str(error)
        try:
            EnforcedHostWatchdog(label=label)
        except Round0245Error as error:
            watchdog_raised = True
            watchdog_message = str(error)
    finally:
        if saved is None:
            os.environ.pop(RUNNER_ABORT_FLAG_ENV, None)
        else:
            os.environ[RUNNER_ABORT_FLAG_ENV] = saved

    evidence = {
        "control": "round0245-missing-abort-flag-positive-control-v1",
        "planted": f"{RUNNER_ABORT_FLAG_ENV} removed from the environment",
        "precondition_refused_to_start": precondition_raised,
        "precondition_message": precondition_message,
        "watchdog_refused_to_arm": watchdog_raised,
        "watchdog_message": watchdog_message,
        "environment_restored": bool(
            os.environ.get(RUNNER_ABORT_FLAG_ENV) == saved
        ),
        "why": (
            "with the variable unset R0244's stage trips, writes nothing, "
            "records no error, and no stripe loop can act on it. Refusing to "
            "start is the only reading of that state that is not a silent "
            "downgrade."
        ),
    }
    failures = [
        name for name, ok in (
            ("precondition_refused", evidence["precondition_refused_to_start"]),
            ("watchdog_refused", evidence["watchdog_refused_to_arm"]),
            ("environment_restored", evidence["environment_restored"]),
        ) if not ok
    ]
    evidence["failures"] = failures
    evidence["holds"] = not failures
    if failures:
        raise Round0245Error(
            "R0245 MISSING-FLAG POSITIVE CONTROL DID NOT FIRE: "
            f"{failures}. In-stage enforcement would remain optional."
        )
    return evidence


def _touch_anonymous(nbytes: int, threads: int) -> list[Any]:
    """Allocate and WRITE `nbytes` of anonymous memory as fast as this box can.

    A `numpy` `fill` releases the GIL, so the write scales with threads; this
    box reaches `~16 GB/s` at four, which is what makes a control against the
    measured `11.77 GB/s` slope possible at all.
    """
    import numpy as np

    per = int(nbytes) // int(threads)
    held: list[Any] = [None] * int(threads)

    def work(index: int) -> None:
        block = np.empty(per, dtype=np.uint8)
        block.fill(np.uint8(index % 251 + 1))
        held[index] = block

    workers = [
        threading.Thread(target=work, args=(index,), daemon=True)
        for index in range(int(threads))
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    return held


def run_allocation_slope_positive_control(
    *,
    flag_path: str,
    trip_at_bytes: int = 2 << 30,
    compliant_headroom_bytes: int = 6 << 30,
    compliant_unit_bytes: int = 1 << 30,
    compliant_max_units: int = 8,
    breaching_headroom_bytes: int = 512 << 20,
    breaching_unit_bytes: int = 2 << 30,
    breaching_max_units: int = 4,
    threads: int = 4,
    interval_s: float = 0.05,
    min_slope_bytes_per_s: float = SLOPE_CONTROL_MIN_BYTES_PER_S,
) -> dict[str, Any]:
    """A stage allocating faster than the measured slope, run in two arms.

    Arm 1 — **compliant**: the unit is small enough that the spacing between
    cooperative-abort reads satisfies `spacing <= headroom / slope`. The guard
    must fire AND the stage must unwind while still inside its headroom.

    Arm 2 — **breaching**: the identical stage with a larger unit, so the
    spacing violates the requirement. The stage must overshoot its headroom.
    That arm is what makes the requirement non-vacuous: a rule every
    configuration satisfies is not a rule.

    The two arms differ ONLY in poll spacing and the headroom they are held to,
    which is precisely the claim. Both allocate real, written anonymous memory
    and free it immediately, and the control refuses to run at all unless the
    machine has room for the worst case plus a wide margin — a control must
    never be the thing that swaps this box.
    """
    #: R0247: a control's own bar is a safety parameter - lowering it lets a
    #: slow synthetic stage certify the guard against a defect gentler than the
    #: one already measured.
    min_slope_effective, min_slope_record = clamp(
        "slope_control_min_bytes_per_s", min_slope_bytes_per_s,
        site="run_allocation_slope_positive_control(min_slope_bytes_per_s=)",
        label="R0245 allocation-slope control",
    )
    min_slope_bytes_per_s = float(min_slope_effective)
    available = int(_meminfo().get("MemAvailable", 0))
    worst_case = int(trip_at_bytes) + max(
        int(compliant_max_units) * int(compliant_unit_bytes),
        int(breaching_max_units) * int(breaching_unit_bytes),
    )
    needed = worst_case + (32 << 30)
    if available < needed:
        raise Round0245Error(
            f"R0245 REFUSES to run the slope control: MemAvailable is "
            f"{available} B and an arm could reach {worst_case} B; {needed} B "
            "is required so the control can never be the cause of swap on this "
            "box."
        )

    arms: dict[str, Any] = {}
    for arm, headroom, unit_bytes, max_units in (
        ("compliant", int(compliant_headroom_bytes),
         int(compliant_unit_bytes), int(compliant_max_units)),
        ("breaching", int(breaching_headroom_bytes),
         int(breaching_unit_bytes), int(breaching_max_units)),
    ):
        arms[arm] = _slope_arm(
            flag_path=flag_path,
            trip_at_bytes=int(trip_at_bytes),
            headroom_bytes=headroom,
            unit_bytes=unit_bytes,
            threads=int(threads),
            max_units=max_units,
            interval_s=float(interval_s),
        )

    fastest = max(float(arms[name]["measured_slope_bytes_per_s"]) for name in arms)
    compliant = arms["compliant"]
    breaching = arms["breaching"]
    evidence = {
        "control": "round0245-allocation-slope-positive-control-v1",
        "planted": (
            "a synthetic stage allocating written anonymous memory faster than "
            "the slope R0244's own sealed trace exhibits"
        ),
        "min_slope_bytes_per_s": float(min_slope_bytes_per_s),
        "safety_overrides": [
            dict(record) for record in override_records([min_slope_record])
        ],
        "fastest_measured_slope_bytes_per_s": fastest,
        "slope_over_the_r0244_measurement": (
            fastest / float(min_slope_bytes_per_s)
        ),
        "arms": arms,
        "why": POLL_SPACING_NOTE,
    }
    failures = [
        name for name, ok in (
            ("stage_is_at_least_as_fast_as_the_measured_slope",
             fastest >= float(min_slope_bytes_per_s)),
            ("compliant_arm_meets_the_requirement",
             bool(compliant["requirement"]["requirement_holds"])),
            ("compliant_guard_fired", bool(compliant["watchdog_fired"])),
            ("compliant_flag_landed", bool(compliant["abort_flag_written"])),
            ("compliant_loop_unwound",
             compliant["unwound_at_unit"] is not None),
            ("compliant_stayed_inside_headroom",
             bool(compliant["growth_after_trip_bytes"]
                  <= compliant["headroom_bytes"])),
            ("breaching_arm_violates_the_requirement",
             not bool(breaching["requirement"]["requirement_holds"])),
            ("breaching_guard_fired", bool(breaching["watchdog_fired"])),
            ("breaching_overshot_its_headroom",
             bool(breaching["growth_after_trip_bytes"]
                  > breaching["headroom_bytes"])),
        ) if not ok
    ]
    evidence["failures"] = failures
    evidence["holds"] = not failures
    if failures:
        raise Round0245Error(
            "R0245 ALLOCATION-SLOPE POSITIVE CONTROL DID NOT FIRE: "
            f"{failures}. Either the guard does not stop a fast stage inside "
            "its headroom, or the poll-spacing requirement is vacuous."
        )
    return evidence


def _slope_arm(
    *,
    flag_path: str,
    trip_at_bytes: int,
    headroom_bytes: int,
    unit_bytes: int,
    threads: int,
    max_units: int,
    interval_s: float,
) -> dict[str, Any]:
    """One arm: allocate `unit_bytes` per unit, read the flag between units.

    The unit wall IS the enforcement spacing, so the arm's poll spacing is a
    measurement of the stage rather than a parameter of the harness.
    """
    if runner_abort_reason() is not None:
        raise Round0245Error(
            "R0245 will not run its slope control while a real cooperative "
            "abort is already requested"
        )
    baseline = int(host_anonymous_bytes()["anonymous_bytes"])
    budget = baseline + int(trip_at_bytes)
    saved_env = os.environ.get(RUNNER_ABORT_FLAG_ENV)
    watchdog = EnforcedHostWatchdog(
        anonymous_budget_bytes=budget,
        interval_s=float(interval_s),
        abort_flag_path=flag_path,
        label="R0245 allocation-slope control",
    )
    held: list[Any] = []
    unwound_at: int | None = None
    anonymous_at_trip: int | None = None
    final_anonymous = baseline
    unit_walls: list[float] = []
    try:
        os.environ[RUNNER_ABORT_FLAG_ENV] = flag_path
        watchdog.start()
        for unit in range(int(max_units)):
            #: The stripe loop's shape: act on the flag between units of work,
            #: never inside one. This is `_check_runner_abort`'s contract.
            if runner_abort_reason() is not None:
                unwound_at = unit
                break
            unit_started = time.monotonic()
            held.append(_touch_anonymous(int(unit_bytes), int(threads)))
            unit_walls.append(time.monotonic() - unit_started)
            with watchdog._lock:  # noqa: SLF001 - the control inspects its subject
                if watchdog.fired and anonymous_at_trip is None:
                    anonymous_at_trip = int(watchdog.trip_anonymous_bytes or 0)
        final_anonymous = int(host_anonymous_bytes()["anonymous_bytes"])
    finally:
        watchdog.stop()
        held = []
        if saved_env is None:
            os.environ.pop(RUNNER_ABORT_FLAG_ENV, None)
        else:
            os.environ[RUNNER_ABORT_FLAG_ENV] = saved_env
        try:
            os.remove(flag_path)
        except OSError:
            pass

    receipt = watchdog.receipt()
    trip_bytes = int(receipt["trip_anonymous_bytes"] or 0)
    spacing = max(unit_walls) if unit_walls else float("inf")
    mean_wall = (
        sum(unit_walls) / float(len(unit_walls)) if unit_walls else float("inf")
    )
    measured_slope = (
        float(unit_bytes) / mean_wall
        if mean_wall not in (0.0, float("inf")) else 0.0
    )
    requirement = poll_spacing_requirement(
        slope_bytes_per_s=binding_slope_bytes_per_s(measured_slope),
        headroom_bytes=int(headroom_bytes),
        poll_spacing_s=max(spacing, 1e-9),
    )
    return {
        "baseline_anonymous_bytes": baseline,
        "declared_budget_bytes": budget,
        "headroom_bytes": int(headroom_bytes),
        "units_allocated": len(unit_walls),
        "unit_bytes": int(unit_bytes),
        "threads": int(threads),
        "max_unit_wall_s": spacing,
        "mean_unit_wall_s": mean_wall,
        "measured_slope_bytes_per_s": measured_slope,
        "watchdog_fired": bool(receipt["fired"]),
        "trip_rule": receipt["trip_rule"],
        "trip_anonymous_bytes": trip_bytes,
        "trip_observed_after_s": receipt["trip_observed_after_s"],
        "abort_flag_written": bool(receipt["abort_flag_written"]),
        "unwound_at_unit": unwound_at,
        "final_anonymous_bytes": final_anonymous,
        "growth_after_trip_bytes": max(final_anonymous - trip_bytes, 0),
        "sampling_thread_alive": bool(receipt["sampling_thread_alive"]),
        "requirement": requirement,
    }


def guard_receipt(
    *,
    abort_flag: Mapping[str, Any],
    thread_death: Mapping[str, Any],
    missing_flag: Mapping[str, Any],
    slope: Mapping[str, Any],
    stripe: Mapping[str, Any],
) -> dict[str, Any]:
    """The three fixes and their three controls, in one falsifiable block."""
    arms = {
        "sampler_death_surfaces": bool(thread_death["holds"]),
        "unset_flag_refuses_to_start": bool(missing_flag["holds"]),
        "fast_stage_stopped_inside_its_headroom": bool(slope["holds"]),
        "r0243_stripe_spacing_violates_the_requirement": not bool(
            stripe["requirement_holds"]
        ),
        "enforcement_path_is_a_precondition": bool(
            abort_flag["directory_is_writable"]
        ),
    }
    return {
        "instrument": "round0245-guard-closure-v1",
        "abort_flag_precondition": dict(abort_flag),
        "thread_death_control": dict(thread_death),
        "missing_flag_control": dict(missing_flag),
        "allocation_slope_control": dict(slope),
        "r0243_stripe_against_the_requirement": dict(stripe),
        "arms": arms,
        "holds": all(arms.values()),
        "guard_axis": "anonymous, never RSS",
        "thresholds_and_rule": (
            "R0242's conjunctive rule and its three thresholds are INHERITED "
            "unchanged through R0244's ThreadedHostWatchdog. R0245 adds no "
            "threshold and edits no rule; it makes the sampler's death "
            "visible, the enforcement path mandatory, and the poll spacing a "
            "computed requirement."
        ),
    }


__all__ = [
    "ABORT_FLAG_NOTE",
    "ABORT_FLAG_PROBE_NOTE",
    "MIN_BINDING_SLOPE_BASIS",
    "MIN_BINDING_SLOPE_BYTES_PER_S",
    "AbortPollTracker",
    "binding_slope_bytes_per_s",
    "probe_abort_flag_path",
    "EnforcedHostWatchdog",
    "GPU_HOURS_CAP",
    "POLL_SPACING_NOTE",
    "R0244_BUDGET_HEADROOM_BYTES",
    "R0244_MEASURED_SLOPE_BYTES_PER_S",
    "R0244_STRIPE_LATENCY_S",
    "ROUND_ID",
    "ROWS",
    "Round0245Error",
    "SLOPE_CONTROL_MIN_BYTES_PER_S",
    "guard_receipt",
    "poll_spacing_requirement",
    "r0244_stripe_verdict",
    "require_enforceable_abort_flag",
    "require_poll_spacing",
    "run_allocation_slope_positive_control",
    "run_missing_flag_positive_control",
    "run_thread_death_positive_control",
    "slope_from_trace",
]
