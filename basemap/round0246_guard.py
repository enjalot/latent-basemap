"""R0246 — close the three guards the reviewer defeated, and unblock training.

review-0245-01 accepted R0245 and then walked through all three of its guard
fixes. Every one had a door the round's own positive control did not open:

1. **A1-bis — the `OSError` arm.** `_loop` caught `OSError` *first* and
   `pass`ed. A planted `OSError` left the thread alive and sampling nothing,
   `thread_death` `null`, `sampling_thread_alive` `true`, `poll()` quiet,
   `sample_coverage` `0.05`, and `_require_live_sampler()` **passed it**.
2. **A3-bis — the flag path.** `require_enforceable_abort_flag()` validated
   `os.path.dirname(path)`. `ROUNDRUN_ABORT_FLAG` pointed at an existing
   writable **directory** passed the precondition, armed the watchdog, and the
   trip landed nowhere (`abort_flag_written false`,
   `abort_flag_error [Errno 21] Is a directory`).
3. **A2-bis — the loosening.** R0245's correction rebound the poll-spacing gate
   to the stage's own slope with a `max(slope, 1.0)` B/s floor. Replaying
   attempt 1's exact `5.828025072987657` s gap at `measured_slope = 0.0` gives a
   binding slope of `1.0` B/s, a permitted spacing of `47,244,640,256` s, and
   the gate that had just fired on its author's code **passes the same gap**.
   Zero or one tracker calls passed trivially.

And the blocker: the widest abort-read gap in R0245's sampler node,
`24.46713631998864` s, sits inside R0244's registered `two_level_weight_sample`
and is `6.09x` over the headroom at the measured worst-case slope.

This module holds the closed gates, the three positive controls **in the exact
shape the reviewer used**, and a battery of novel attacks that no control
covers. Nothing here signals anything, starts a child, or touches the GPU.
"""
from __future__ import annotations

import math
import os
import stat
import threading
import time
from collections.abc import Mapping
from typing import Any

from basemap.gpu_child_supervision import RUNNER_ABORT_FLAG_ENV
from basemap.round0244_guard import (
    WATCHDOG_MAX_CONSECUTIVE_SAMPLE_FAILURES,
    WATCHDOG_SAMPLE_INTERVAL_S,
    Round0244Error,
)
from basemap.round0245_guard import (
    MIN_BINDING_SLOPE_BASIS,
    MIN_BINDING_SLOPE_BYTES_PER_S,
    R0244_BUDGET_HEADROOM_BYTES,
    R0244_MEASURED_SLOPE_BYTES_PER_S,
    AbortPollTracker,
    EnforcedHostWatchdog,
    Round0245Error,
    binding_slope_bytes_per_s,
    poll_spacing_requirement,
    require_enforceable_abort_flag,
    slope_from_trace,
)
from basemap.round0247_registry import (
    REGISTERED_SAFETY_PARAMETERS,
    WATCHDOG_MAX_MEAN_OBSERVATION_GAP_S,
    WATCHDOG_MAX_OBSERVATION_GAP_S,
    Round0247Error,
    clamp,
    clamp_int,
    override_records,
    registered_bounds,
    registry_fingerprint,
    require_no_weakening_overrides,
    weakening_overrides,
    verify_registry,
)

ROUND_ID = "0246"
ROWS = 100_000_000
GPU_HOURS_CAP = 1.0


class Round0246Error(RuntimeError):
    """R0246 fails closed."""


# --------------------------------------------------------------------------- #
# A1-bis — a sampler is alive only if it is SAMPLING
# --------------------------------------------------------------------------- #
#: The floor a node's host guard must clear on the fraction of the samples its
#: declared interval implies. TWO bases, both stated, and the registered value
#: is the stricter:
#:
#: * Derived safety floor. A coverage of `c` means a mean observation gap of
#:   `WATCHDOG_SAMPLE_INTERVAL_S / c`, which at the registered worst-case slope
#:   permits `0.25 / c * 11767996416` B before the trip is even seen. Holding
#:   that inside R0244's sealed `budget_headroom_bytes` of `29,548,888,064` B
#:   requires `c >= 0.25 * 11767996416 / 29548888064 = 0.09956`.
#: * Measured healthy floor. R0245's sampler node ran `114.54487995902309` s at
#:   `0.25` s with `439` samples of which `4` were boundary `poll()`s, i.e. a
#:   thread coverage of `435 / 458.17951983609237 = 0.9494`.
#:
#: The registered floor is `0.50` — `5.02x` stricter than the derived safety
#: floor and half the measured healthy value. It is not a number chosen to make
#: anything pass; it refuses a guard that is sampling at less than half the rate
#: a healthy node sampled at, while the number that would merely preserve the
#: headroom inequality is five times smaller.
MIN_THREAD_SAMPLE_COVERAGE = 0.50

#: The gate only applies once the stage is long enough for coverage to mean
#: something. At `20` expected samples one missed sample is `0.05` of coverage,
#: so quantisation cannot move a healthy `0.95` anywhere near the `0.50` floor;
#: below that the ratio is noise and the gate would be theatre.
COVERAGE_GATE_MIN_EXPECTED_SAMPLES = 20.0

SAFETY_PARAMETER_NOTE = (
    "REGISTERED 2026-08-10 (R0247). Every bound in this verdict named "
    "registered_* is produced by round0247_registry.registered_bounds() and "
    "therefore comes from the registry, never from a caller. What the caller "
    "asked for appears as declared_*, what the gate bound on appears as "
    "effective_*, and any attempt to weaken a registered bound appears in "
    "safety_overrides and fails the gate. review-0246-01 C defeated R0246 by "
    "passing max_poll_spacing_s = 1e6 and watching the receipt publish it as "
    "registered_max_poll_spacing_s."
)

SAMPLER_LIVENESS_NOTE = (
    "review-0245-01 A1-bis: a sampler that raises OSError on every sample stays "
    "alive, reports sampling_thread_alive true, keeps poll() quiet and lets the "
    "node publish its memory receipt at 0.05 coverage. R0246 closes it three "
    "ways. (a) The OSError arm is bounded: "
    f"{WATCHDOG_MAX_CONSECUTIVE_SAMPLE_FAILURES} consecutive failures are "
    "tolerated and the next one is a recorded death. (b) ANY exit from the "
    "sampling loop that was not requested is a recorded death, including a "
    "return that raises nothing. (c) A coverage FLOOR: a sampler reporting a "
    "small fraction of its expected samples over a long stage is not alive in "
    "any useful sense, and coverage is computed on the THREAD's samples only, "
    "because the published sample_coverage counts boundary poll() samples too "
    "and a busy main thread would otherwise hold it up on its own."
)


def require_live_sampler(
    receipt: Mapping[str, Any],
    *,
    label: str,
    min_thread_sample_coverage: float = MIN_THREAD_SAMPLE_COVERAGE,
    min_expected_samples: float = COVERAGE_GATE_MIN_EXPECTED_SAMPLES,
) -> dict[str, Any]:
    """A node may not publish a memory receipt written by a guard that stopped.

    Fails closed on a recorded death, on a receipt that reports a dead sampler,
    and — the half R0245 did not implement — on thread coverage below the
    registered floor once the stage is long enough for coverage to be a
    measurement.
    """
    verify_registry(label=f"R0246 sampler liveness for {label}")
    #: review-0246-01 A and H2. BOTH of these were keywords with registered
    #: defaults and no bound. `min_thread_sample_coverage` is the obvious one;
    #: `min_expected_samples` is the one no review found - passing `1e9` makes
    #: `coverage_gate_applies` false for every stage this program will ever run
    #: and disables the arm without touching the floor.
    min_coverage_effective, coverage_record = clamp(
        "min_thread_sample_coverage", min_thread_sample_coverage,
        site="require_live_sampler(min_thread_sample_coverage=)", label=label,
    )
    min_expected_effective, expected_record = clamp(
        "coverage_gate_min_expected_samples", min_expected_samples,
        site="require_live_sampler(min_expected_samples=)", label=label,
    )
    overrides = override_records([coverage_record, expected_record])
    min_thread_sample_coverage = float(min_coverage_effective)
    min_expected_samples = float(min_expected_effective)

    thread_samples = receipt.get("thread_samples")
    coverage = receipt.get("thread_sample_coverage")
    if thread_samples is None or coverage is None:
        raise Round0246Error(
            f"R0246 STOP: {label} published a host-guard receipt with no "
            "thread_samples / thread_sample_coverage. A receipt that cannot "
            "report how often its sampler actually sampled cannot be gated on."
        )
    #: THE DENOMINATOR FIX. review-0246-01 A: "thread coverage is a ratio to a
    #: self-declared sampling interval. Nothing anywhere gates
    #: sample_interval_s." R0246 read `expected_samples_at_interval`, which is
    #: `wall / interval_s` with the node's own `interval_s`, so a node declaring
    #: `5.0` reported a coverage of `0.9994` while its observation gap bought
    #: `1.99x` the sealed headroom. The gate now reads
    #: `expected_samples_at_the_registered_interval`, which is `wall / 0.25`,
    #: and it reads two arms that are not ratios at all: the widest and the mean
    #: MEASURED interval between two thread samples, in seconds.
    registered_interval = float(
        REGISTERED_SAFETY_PARAMETERS["watchdog_sample_interval_s"].value
    )
    expected = receipt.get("expected_samples_at_the_registered_interval")
    if expected is None:
        wall = float(receipt.get("sampled_wall_s") or 0.0)
        expected = wall / registered_interval if registered_interval > 0 else 0.0
    expected = float(expected)
    coverage_at_registered = receipt.get(
        "thread_sample_coverage_at_the_registered_interval"
    )
    if coverage_at_registered is None:
        coverage_at_registered = (
            float(thread_samples) / expected if expected > 0 else 0.0
        )
    coverage_at_registered = float(coverage_at_registered)
    wall_s = float(receipt.get("sampled_wall_s") or 0.0)
    max_gap = receipt.get("max_thread_sample_gap_s")
    mean_gap = receipt.get("mean_thread_sample_gap_s")
    if mean_gap is None:
        mean_gap = (
            wall_s / float(thread_samples) if int(thread_samples) > 0
            else float("inf")
        )
    mean_gap = float(mean_gap)
    if max_gap is None:
        #: A receipt from before R0247 cannot report a measured gap. It is not
        #: read as "no gap"; it is read as "unmeasured", which fails.
        max_gap = float("inf")
    max_gap = float(max_gap)
    #: The two second-valued arms apply once the stage is long enough for one
    #: registered interval to have elapsed at all. Below that a single sample
    #: spans the whole stage and the gap is the stage, not a defect.
    gap_gate_applies = bool(wall_s >= registered_interval)
    coverage_gate_applies = bool(expected >= float(min_expected_samples))
    state = {
        "label": label,
        "declared_sample_interval_s": receipt.get("declared_sample_interval_s"),
        "sampled_wall_s": wall_s,
        "max_thread_sample_gap_s": max_gap,
        "mean_thread_sample_gap_s": mean_gap,
        "gap_gate_applies": gap_gate_applies,
        "thread_sample_coverage_at_the_registered_interval": (
            coverage_at_registered
        ),
        "expected_samples_at_the_registered_interval": expected,
        "safety_overrides": [dict(record) for record in overrides],
        "receipt_safety_overrides": [
            dict(record) for record in (receipt.get("safety_overrides") or ())
        ],
        **registered_bounds([
            "watchdog_sample_interval_s",
            "watchdog_max_observation_gap_s",
            "watchdog_max_mean_observation_gap_s",
            "min_thread_sample_coverage",
            "coverage_gate_min_expected_samples",
        ]),
        "sampling_thread_alive": bool(receipt["sampling_thread_alive"]),
        "thread_death": receipt["thread_death"],
        "samples": int(receipt["samples"]),
        "thread_samples": int(thread_samples),
        "boundary_polls": int(receipt.get("boundary_polls") or 0),
        "sample_coverage": receipt["sample_coverage"],
        "thread_sample_coverage": float(coverage),
        "expected_samples_at_interval": expected,
        "sample_failures": int(receipt.get("sample_failures") or 0),
        "max_consecutive_sample_failures": int(
            receipt.get("max_consecutive_sample_failures") or 0
        ),
        "min_thread_sample_coverage": float(min_thread_sample_coverage),
        "coverage_gate_applies": coverage_gate_applies,
        "coverage_gate_min_expected_samples": float(min_expected_samples),
        "note": SAMPLER_LIVENESS_NOTE,
    }
    #: An override attempt is a recorded violation that FAILS the gate, whether
    #: it was made here or by the watchdog that wrote the receipt.
    try:
        require_no_weakening_overrides(
            list(overrides) + list(receipt.get("safety_overrides") or ()),
            label=f"{label}'s host guard",
        )
    except Round0247Error as error:
        #: R0246 owns this gate's error contract; R0247 registered the bound.
        raise Round0246Error(str(error)) from error
    if not state["sampling_thread_alive"]:
        raise Round0246Error(
            f"R0246 STOP: {label} ran behind a host guard whose sampling thread "
            f"died ({receipt['thread_death']}). Its memory receipt is not "
            "evidence and the node must not publish it."
        )
    if gap_gate_applies and max_gap > WATCHDOG_MAX_OBSERVATION_GAP_S:
        raise Round0246Error(
            f"R0247 STOP: {label}'s host guard left {max_gap} s between two "
            "successful observations, against a registered ceiling of "
            f"{WATCHDOG_MAX_OBSERVATION_GAP_S} s (29,548,888,064 B of sealed "
            "headroom over 11,767,996,416 B/s of sealed measured slope). This "
            "arm is measured in seconds by the sampling thread itself and is "
            "not a ratio to any interval the node declares, which is how "
            "review-0246-01 defeated the coverage floor. "
            + SAMPLER_LIVENESS_NOTE
        )
    if gap_gate_applies and mean_gap > WATCHDOG_MAX_MEAN_OBSERVATION_GAP_S:
        raise Round0246Error(
            f"R0247 STOP: {label}'s host guard averaged {mean_gap} s between "
            f"observations over a {wall_s} s stage, against a registered "
            f"ceiling of {WATCHDOG_MAX_MEAN_OBSERVATION_GAP_S} s - the "
            "registered 0.25 s interval divided by the registered 0.50 "
            "coverage floor, expressed in seconds so that it cannot be "
            "satisfied by moving the denominator. " + SAMPLER_LIVENESS_NOTE
        )
    if coverage_gate_applies and coverage_at_registered < float(
        min_thread_sample_coverage
    ):
        raise Round0246Error(
            f"R0246 STOP: {label}'s host guard took {int(thread_samples)} "
            f"thread samples against {expected} expected at the REGISTERED "
            f"{registered_interval} s interval — a thread coverage of "
            f"{coverage_at_registered} against a registered floor of "
            f"{float(min_thread_sample_coverage)}. A sampler that is alive and "
            "not sampling is the same blind spot as a dead one. "
            + SAMPLER_LIVENESS_NOTE
        )
    state["holds"] = True
    return state


def require_abort_flag_landed(
    receipt: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    """If the guard tripped, the trip must have LANDED.

    review-0245-01 A3-bis: `fired True / abort_flag_written False` was
    reachable and nothing gated on it. This is the node-tail gate.
    """
    fired = bool(receipt.get("fired"))
    written = bool(receipt.get("abort_flag_written"))
    error = receipt.get("abort_flag_error")
    state = {
        "label": label,
        "fired": fired,
        "abort_flag_path": receipt.get("abort_flag_path"),
        "abort_flag_written": written,
        "abort_flag_error": error,
    }
    if fired and not written:
        raise Round0246Error(
            f"R0246 STOP: {label}'s host guard tripped and its cooperative "
            f"abort flag did not land ({error!r}). A trip that writes nothing "
            "cannot stop a stage; the node must fail rather than report a "
            "guard that fired into a void."
        )
    state["holds"] = True
    return state


# --------------------------------------------------------------------------- #
# A2-bis — a poll-spacing gate that can fail
# --------------------------------------------------------------------------- #
#: The absolute registered ceiling on the spacing of cooperative-abort reads,
#: independent of whatever headroom a node declares for itself. It is R0244's
#: sealed `budget_headroom_bytes` over R0244's sealed measured slope:
#: `29,548,888,064 / 11,767,996,416`. A node that declares a larger headroom
#: (R0245's nodes declare `47,244,640,256` B) does not get a wider spacing out
#: of it — declaring a bigger budget is not a safety argument.
R0246_MAX_POLL_SPACING_S = float(R0244_BUDGET_HEADROOM_BYTES) / float(
    R0244_MEASURED_SLOPE_BYTES_PER_S
)

#: Two polls is the smallest number from which a spacing can be computed at
#: all. The binding requirement is not this constant but the derived one below:
#: a stage of wall `W` under a ceiling `C` must read the flag at least
#: `ceil(W / C) + 1` times, because its start and its end are both anchors.
MIN_ENFORCEMENT_POLLS = 2

POLL_GATE_NOTE = (
    "review-0245-01 D. The gate R0245 shipped could not fail on two of its "
    "three nodes: the max(slope, 1.0) B/s floor turned a flat differentiated "
    "trace into a 47,244,640,256 s permitted spacing, a stage that called the "
    "tracker once scored max_gap 0.0, and a stage that never called it scored "
    "polls 0 and held. R0246 binds on max(own measured slope, "
    "11,767,996,416 B/s); anchors the measurement at stage START and stage END "
    "so the interval before the first read and after the last one are gaps like "
    "any other; requires a non-zero measured spacing and at least "
    "ceil(wall / ceiling) + 1 reads; caps the spacing at the registered "
    "2.5109531834854018 s regardless of the headroom a node declares for "
    "itself; and makes the worst-case verdict a HARD gate for any node "
    "declaring training_performed true."
)


class AbortPollGate(AbortPollTracker):
    """R0245's tracker with every door the reviewer opened closed.

    Use is `gate.start()`, then call it at every enforcement site, then
    `gate.finish()`, then `gate.require(measured_slope_bytes_per_s=...)`.
    """

    def __init__(
        self,
        *,
        inner: Any,
        headroom_bytes: int,
        label: str,
        training_performed: bool = False,
        max_poll_spacing_s: float = R0246_MAX_POLL_SPACING_S,
        min_polls: int = MIN_ENFORCEMENT_POLLS,
        slope_bytes_per_s: float = MIN_BINDING_SLOPE_BYTES_PER_S,
        clock: Any = None,
        replay: bool = False,
    ) -> None:
        super().__init__(
            inner=inner, headroom_bytes=headroom_bytes, label=label,
            slope_bytes_per_s=slope_bytes_per_s, clock=clock, replay=replay,
        )
        self.training_performed = bool(training_performed)
        #: review-0246-01 C, the sixteenth attack, and the reason R0247 exists.
        #: This was a constructor keyword with a registered DEFAULT and no
        #: FLOOR, and the receipt then published the caller's value under
        #: `registered_max_poll_spacing_s`. It is clamped at the registry, the
        #: attempt is recorded, the attribute is read-only so it cannot be
        #: reassigned after construction, and `registered_max_poll_spacing_s`
        #: is now produced by `registered_bounds()` and can only be the
        #: registry's number.
        spacing_effective, spacing_record = clamp(
            "r0246_max_poll_spacing_s", max_poll_spacing_s,
            site="AbortPollGate(max_poll_spacing_s=)", label=str(label),
        )
        polls_effective, polls_record = clamp_int(
            "min_enforcement_polls", min_polls,
            site="AbortPollGate(min_polls=)", label=str(label),
        )
        self.safety_overrides = tuple(self.safety_overrides) + override_records(
            [spacing_record, polls_record]
        )
        self._max_poll_spacing_s = float(spacing_effective)
        self.declared_max_poll_spacing_s = float(max_poll_spacing_s)
        self._min_polls = int(polls_effective)
        self.declared_min_polls = int(min_polls)
        self.started_at_s: float | None = None
        self.finished_at_s: float | None = None
        self.gap_before_the_first_poll_s: float | None = None
        self.gap_after_the_last_poll_s: float | None = None

    #: Read-only. R0247 attack a2-r0247-3: clamping the CONSTRUCTOR still left
    #: `gate.max_poll_spacing_s = 1e6` working on a built gate, which is the
    #: same defeat one statement later. There is no setter, so the assignment
    #: raises `AttributeError` instead of widening the ceiling.
    @property
    def max_poll_spacing_s(self) -> float:
        return self._max_poll_spacing_s

    @property
    def min_polls(self) -> int:
        return self._min_polls

    def start(self) -> "AbortPollGate":
        """Anchor at stage start, so the pre-first-poll interval is a gap."""
        if self.started_at_s is not None:
            raise Round0246Error(f"R0246 poll gate {self.label} already started")
        self.started_at_s = float(self._clock())
        self._last = self.started_at_s
        return self

    def __enter__(self) -> "AbortPollGate":
        return self.start()

    def __exit__(self, *_exc: Any) -> None:
        if self.finished_at_s is None:
            self.finish()

    def __call__(self, where: str) -> None:
        if self.started_at_s is None:
            raise Round0246Error(
                f"R0246 poll gate {self.label} was called before start(); the "
                "interval from stage start to the first read would not be "
                "measured, which is one of the ways R0245's gate was defeated."
            )
        now = float(self._clock())
        if self.polls == 0:
            self.gap_before_the_first_poll_s = now - self.started_at_s
        self._record(now, where)

    def finish(self, where: str = "stage end") -> None:
        """Anchor at stage end, so the post-last-poll interval is a gap."""
        if self.started_at_s is None:
            raise Round0246Error(
                f"R0246 poll gate {self.label} finished without starting"
            )
        now = float(self._clock())
        self.finished_at_s = now
        if self._last is not None:
            gap = now - self._last
            self.gap_after_the_last_poll_s = gap
            self.total_gap_s += gap
            if gap > self.max_gap_s:
                self.max_gap_s = gap
                self.max_gap_at = str(where)
        self._last = now

    def stage_wall_s(self) -> float:
        """Never reads the clock: a verdict must not move what it measures."""
        if self.started_at_s is None:
            return 0.0
        end = self.finished_at_s
        if end is None:
            end = self._last if self._last is not None else self.started_at_s
        return float(end - self.started_at_s)

    def required_polls(self) -> int:
        """`ceil(wall / ceiling) + 1`: start and end are both anchors."""
        wall = self.stage_wall_s()
        if self.max_poll_spacing_s <= 0.0:
            return self.min_polls
        return max(
            self.min_polls, int(math.ceil(wall / self.max_poll_spacing_s)) + 1
        )

    def verdict(
        self, *, measured_slope_bytes_per_s: float | None = None
    ) -> dict[str, Any]:
        base = super().verdict(
            measured_slope_bytes_per_s=measured_slope_bytes_per_s
        )
        wall = self.stage_wall_s()
        required = self.required_polls()
        base.update({
            "instrument": "round0246-abort-poll-gate-v1",
            "gate_started": bool(self.started_at_s is not None),
            "gate_finished": bool(self.finished_at_s is not None),
            "stage_wall_s": wall,
            "gap_before_the_first_poll_s": self.gap_before_the_first_poll_s,
            "gap_after_the_last_poll_s": self.gap_after_the_last_poll_s,
            #: review-0246-01 C: "A field named `registered_` that reports
            #: whatever the node passed is worse than no field." These come
            #: from `registered_bounds()`, which reads the registry and can
            #: never read a caller. What the caller asked for is published
            #: beside them under `declared_*` and `effective_*`.
            **registered_bounds([
                "r0246_max_poll_spacing_s",
                "min_enforcement_polls",
                "min_binding_slope_bytes_per_s",
                "max_declared_headroom_bytes",
            ]),
            "declared_max_poll_spacing_s": self.declared_max_poll_spacing_s,
            "effective_max_poll_spacing_s": self.max_poll_spacing_s,
            "declared_min_polls": self.declared_min_polls,
            "effective_min_polls": self.min_polls,
            "registered_max_poll_spacing_basis": (
                "R0244 sealed budget_headroom_bytes 29548888064 / R0244 sealed "
                "measured slope 11767996416 B/s"
            ),
            "no_weakening_safety_override_was_attempted": not
            weakening_overrides(self.safety_overrides),
            "envelope_declarations_clamped_to_the_registry": [
                dict(record) for record in self.safety_overrides
                if record.get("kind") == "weakening"
                and record.get("enforcement") == "clamped"
            ],
            "spacing_over_the_registered_ceiling": (
                self.max_gap_s / self.max_poll_spacing_s
                if self.max_poll_spacing_s > 0 else None
            ),
            "meets_the_registered_ceiling": bool(
                self.max_gap_s <= self.max_poll_spacing_s
            ),
            "required_polls": required,
            "meets_the_required_poll_count": bool(self.polls >= required),
            "measured_spacing_is_non_zero": bool(self.max_gap_s > 0.0),
            "training_performed": self.training_performed,
            "worst_case_slope_is_at_least_the_registered_floor": bool(
                float(self.slope_bytes_per_s) >= MIN_BINDING_SLOPE_BYTES_PER_S
            ),
            "binding_slope_floor_bytes_per_s": MIN_BINDING_SLOPE_BYTES_PER_S,
            "binding_slope_floor_basis": MIN_BINDING_SLOPE_BASIS,
            "registry_fingerprint": registry_fingerprint(),
            "gate_note": POLL_GATE_NOTE,
            "safety_parameter_note": SAFETY_PARAMETER_NOTE,
        })
        return base

    def require(
        self, *, measured_slope_bytes_per_s: float | None = None
    ) -> dict[str, Any]:
        verify_registry(label=f"R0246 poll gate {self.label}")
        verdict = self.verdict(
            measured_slope_bytes_per_s=measured_slope_bytes_per_s
        )
        failures = [
            name for name, ok in (
                ("gate_was_started", verdict["gate_started"]),
                ("gate_was_finished", verdict["gate_finished"]),
                ("measured_spacing_is_non_zero",
                 verdict["measured_spacing_is_non_zero"]),
                ("meets_the_required_poll_count",
                 verdict["meets_the_required_poll_count"]),
                ("worst_case_slope_is_at_least_the_registered_floor",
                 verdict["worst_case_slope_is_at_least_the_registered_floor"]),
                ("meets_the_registered_ceiling",
                 verdict["meets_the_registered_ceiling"]),
                ("meets_the_binding_slope_requirement",
                 bool(verdict["requirement"]["requirement_holds"])),
                #: R0247: `training_performed` was a self-declared bool that
                #: switched off this arm. It is now unconditional, so the
                #: declaration no longer participates in a safety decision at
                #: all. Direction: strictly tighter - the arm applies to every
                #: node, and no node that passed it before can fail it now.
                ("every_node_meets_the_worst_case_slope",
                 bool(verdict["meets_the_r0244_worst_case_slope"])),
                #: R0247 attack a2-r0247-1: a scripted clock replaces the
                #: measurement itself.
                ("the_clock_is_the_registered_monotonic_clock",
                 bool(
                     verdict["clock_is_the_registered_monotonic_clock"]
                     or self.replay
                 )),
                #: R0247 attack a2-r0247-2: the gate times calls to itself, so
                #: a gate wrapping a no-op scores a stage that never reads the
                #: cooperative abort flag.
                ("the_gate_wraps_a_registered_abort_reader",
                 bool(
                     verdict["inner_is_a_registered_abort_reader"]
                     or self.replay
                 )),
                ("no_weakening_safety_override_was_attempted",
                 bool(verdict["no_weakening_safety_override_was_attempted"])),
            ) if not ok
        ]
        verdict["failures"] = failures
        verdict["holds"] = not failures
        if failures:
            raise Round0246Error(
                f"R0246 STOP: {self.label} does not meet the poll-spacing gate "
                f"{failures}. Widest gap {verdict['max_gap_between_enforcement_polls_s']} s "
                f"(after {verdict['widest_gap_before']!r}) against a registered "
                f"ceiling of {self.max_poll_spacing_s} s; "
                f"{verdict['enforcement_polls']} reads against "
                f"{verdict['required_polls']} required over a "
                f"{verdict['stage_wall_s']} s stage. " + POLL_GATE_NOTE
            )
        return verdict


def require_enforcement_evidence(
    verdict: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    """A replayed verdict is a demonstration, never a node's own evidence.

    R0247 attack a2-r0247-1 in its sealing form: a node could build its gate
    with a scripted clock, declare `replay=True` so the clock arm is waived, and
    seal the resulting verdict as its enforcement measurement. This is the gate
    on that: anything a node publishes as `enforcement_poll_spacing` must have
    been measured on the registered monotonic clock, through a registered abort
    reader, with no weakening override attempted.
    """
    state = {
        "label": label,
        "replay_only": bool(verdict.get("replay_only")),
        "clock_is_the_registered_monotonic_clock": bool(
            verdict.get("clock_is_the_registered_monotonic_clock")
        ),
        "inner_is_a_registered_abort_reader": bool(
            verdict.get("inner_is_a_registered_abort_reader")
        ),
        "no_weakening_safety_override_was_attempted": bool(
            verdict.get("no_weakening_safety_override_was_attempted")
        ),
    }
    failures = [
        name for name, ok in (
            ("not_a_replay", not state["replay_only"]),
            ("clock_is_the_registered_monotonic_clock",
             state["clock_is_the_registered_monotonic_clock"]),
            ("inner_is_a_registered_abort_reader",
             state["inner_is_a_registered_abort_reader"]),
            ("no_weakening_safety_override_was_attempted",
             state["no_weakening_safety_override_was_attempted"]),
        ) if not ok
    ]
    state["failures"] = failures
    state["holds"] = not failures
    if failures:
        raise Round0247Error(
            f"R0247 STOP: {label} tried to seal a poll-spacing verdict that is "
            f"not enforcement evidence {failures}. A replayed gate, a scripted "
            "clock, a gate wrapping something that does not read the "
            "cooperative abort flag, and a gate whose bound was overridden are "
            "each a measurement of nothing. " + SAFETY_PARAMETER_NOTE
        )
    return state


def measured_slope_from_trace(trace: Any) -> float:
    """Differentiate a stage's own trace, or `0.0` when it is too short.

    A stage shorter than two one-second trace bins has no differentiable trace
    at all. Returning `0.0` is safe ONLY because the binding slope is floored at
    the registered worst case: a `0.0` here is never read as "this stage is
    slow", it is read as "this stage's own slope carries no information", and
    the gate binds on `11,767,996,416` B/s instead. That is the whole reason
    the `max(slope, 1.0)` floor had to go.
    """
    points = list(trace or [])
    if len(points) < 2:
        return 0.0
    return float(slope_from_trace(points)["max_rise_bytes_per_s"])


def replay_gap_through_the_gate(
    *,
    gap_s: float,
    headroom_bytes: int,
    measured_slope_bytes_per_s: float,
    label: str,
) -> dict[str, Any]:
    """Replay one exact gap through the corrected gate, on a scripted clock."""
    ticks = iter([0.0, 0.0, float(gap_s), float(gap_s)])
    gate = AbortPollGate(
        inner=lambda _where: None, headroom_bytes=int(headroom_bytes),
        label=label, clock=lambda: next(ticks), replay=True,
    )
    gate.start()
    gate("replayed read 1")
    gate("replayed read 2")
    gate.finish()
    refused = False
    message = None
    try:
        gate.require(measured_slope_bytes_per_s=measured_slope_bytes_per_s)
    except Round0246Error as error:
        refused = True
        message = str(error)
    scored = gate.verdict(
        measured_slope_bytes_per_s=measured_slope_bytes_per_s
    )
    return {
        "replayed_gap_s": float(gap_s),
        "headroom_bytes": int(headroom_bytes),
        "measured_slope_bytes_per_s": float(measured_slope_bytes_per_s),
        "binding_slope_bytes_per_s": binding_slope_bytes_per_s(
            measured_slope_bytes_per_s
        ),
        "max_poll_spacing_s": scored["requirement"]["max_poll_spacing_s"],
        "permitted_growth_after_trip_bytes": scored["requirement"][
            "permitted_growth_after_trip_bytes"
        ],
        "gate_refused_it": refused,
        "message": message,
    }


# --------------------------------------------------------------------------- #
# the three positive controls, in the exact shape the reviewer used
# --------------------------------------------------------------------------- #
class _OSErrorWatchdog(EnforcedHostWatchdog):
    """The reviewer's A1-bis subject: `_sample()` raises `OSError` from the
    second sample onward. Before R0246 this thread stayed alive forever."""

    def __init__(self, *, die_after: int = 1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.die_after = int(die_after)
        self._sample_calls = 0

    def _sample(self) -> dict[str, int]:
        self._sample_calls += 1
        if (
            self._sample_calls > self.die_after
            and threading.current_thread() is not threading.main_thread()
        ):
            raise OSError(
                "planted /proc read failure (R0246 A1-bis reviewer control)"
            )
        return super()._sample()


class _StarvedWatchdog(EnforcedHostWatchdog):
    """A sampler that never raises and barely samples: the coverage attack."""

    def __init__(self, *, stall_s: float = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.stall_s = float(stall_s)
        self._sample_calls = 0

    def _sample(self) -> dict[str, int]:
        self._sample_calls += 1
        if (
            self._sample_calls > 1
            and threading.current_thread() is not threading.main_thread()
        ):
            time.sleep(self.stall_s)
        return super()._sample()


class _IntermittentWatchdog(EnforcedHostWatchdog):
    """Fails `fail_of` - 1 of every `fail_of` samples, never enough consecutive
    failures to trip the registered tolerance. The residual-shaped attack."""

    def __init__(
        self, *, fail_of: int = 3, stall_s: float = 0.0, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.fail_of = int(fail_of)
        #: R0247: with a stall the same failure pattern becomes genuinely
        #: blind, which is what `a1-r0247-4` plants.
        self.stall_s = float(stall_s)
        self._sample_calls = 0

    def _sample(self) -> dict[str, int]:
        if threading.current_thread() is threading.main_thread():
            return super()._sample()
        self._sample_calls += 1
        if self._sample_calls % self.fail_of != 0:
            raise OSError("planted intermittent /proc read failure")
        if self.stall_s > 0.0:
            time.sleep(self.stall_s)
        return super()._sample()


class _SilentReturnWatchdog(EnforcedHostWatchdog):
    """A sampling loop that simply returns. Nothing raises; the guard is gone."""

    def _loop(self) -> None:
        try:
            self._sample_once()
            return
        finally:
            self._record_exit()


def _run_watchdog_until(
    watchdog: EnforcedHostWatchdog, *, wall_s: float, poll_every_s: float = 0.0
) -> int:
    """Run a planted watchdog for a fixed wall, optionally polling hard."""
    watchdog.start()
    started = time.monotonic()
    polls = 0
    try:
        while time.monotonic() - started < float(wall_s):
            if poll_every_s > 0.0:
                try:
                    watchdog.poll("R0246 attack main-thread poll")
                    polls += 1
                except Round0244Error:
                    pass
                time.sleep(poll_every_s)
            else:
                time.sleep(0.01)
    finally:
        watchdog.stop()
    return polls


def _liveness_attack(
    *, watchdog: EnforcedHostWatchdog, wall_s: float, poll_every_s: float = 0.0
) -> dict[str, Any]:
    polls = _run_watchdog_until(
        watchdog, wall_s=wall_s, poll_every_s=poll_every_s
    )
    receipt = watchdog.receipt()
    refused = False
    message = None
    try:
        require_live_sampler(receipt, label="R0246 attack")
    except Round0246Error as error:
        refused = True
        message = str(error)
    return {
        "main_thread_polls": polls,
        "sampling_thread_alive": bool(receipt["sampling_thread_alive"]),
        "thread_death": receipt["thread_death"],
        "samples_including_boundary_polls": int(receipt["samples"]),
        "thread_samples": int(receipt["thread_samples"]),
        "sample_coverage_as_r0245_published_it": receipt["sample_coverage"],
        "thread_sample_coverage": receipt["thread_sample_coverage"],
        "expected_samples_at_interval": receipt["expected_samples_at_interval"],
        "sample_failures": int(receipt["sample_failures"]),
        "max_consecutive_sample_failures": int(
            receipt["max_consecutive_sample_failures"]
        ),
        #: R0247: the two denominator-free arms, so an attack row can be scored
        #: on what the sampler MEASURED rather than on a ratio to what it
        #: declared.
        "sampled_wall_s": receipt["sampled_wall_s"],
        "max_thread_sample_gap_s": receipt["max_thread_sample_gap_s"],
        "mean_thread_sample_gap_s": receipt["mean_thread_sample_gap_s"],
        "thread_sample_coverage_at_the_registered_interval": receipt[
            "thread_sample_coverage_at_the_registered_interval"
        ],
        "gate_refused_the_receipt": refused,
        "message": message,
    }


def run_reviewer_oserror_control(
    *, flag_path: str, interval_s: float = 0.05, wall_s: float = 1.5
) -> dict[str, Any]:
    """review-0245-01 A1-bis, exactly: a planted `OSError` in `_sample()`."""
    watchdog = _OSErrorWatchdog(
        die_after=1, anonymous_budget_bytes=1 << 62, interval_s=float(interval_s),
        abort_flag_path=flag_path,
        label="R0246 reviewer OSError control",
    )
    evidence = _liveness_attack(watchdog=watchdog, wall_s=float(wall_s))
    evidence.update({
        "control": "round0246-a1bis-reviewer-oserror-control-v1",
        "planted": (
            "OSError raised inside the sampling thread from the second sample "
            "onward - the arm R0245's fix routed to `pass`"
        ),
        "what_r0245_did": (
            "thread_death null, sampling_thread_alive true, poll() quiet, "
            "sample_coverage 0.0500, _require_live_sampler PASSED"
        ),
    })
    failures = [
        name for name, ok in (
            ("death_recorded", evidence["thread_death"] is not None),
            ("receipt_reports_a_dead_sampler",
             not evidence["sampling_thread_alive"]),
            ("gate_refused_the_receipt", evidence["gate_refused_the_receipt"]),
        ) if not ok
    ]
    evidence["failures"] = failures
    evidence["holds"] = not failures
    _cleanup(flag_path)
    if failures:
        raise Round0246Error(
            f"R0246 A1-BIS REVIEWER CONTROL DID NOT FIRE: {failures}"
        )
    return evidence


def run_reviewer_directory_flag_control(*, directory: str) -> dict[str, Any]:
    """review-0245-01 A3-bis, exactly: the flag path IS a writable directory."""
    os.makedirs(directory, exist_ok=True)
    saved = os.environ.get(RUNNER_ABORT_FLAG_ENV)
    precondition_refused = False
    precondition_message = None
    watchdog_refused = False
    watchdog_message = None
    try:
        os.environ[RUNNER_ABORT_FLAG_ENV] = directory
        try:
            require_enforceable_abort_flag(label="R0246 directory control")
        except Round0245Error as error:
            precondition_refused = True
            precondition_message = str(error)
        try:
            EnforcedHostWatchdog(label="R0246 directory control")
        except Round0245Error as error:
            watchdog_refused = True
            watchdog_message = str(error)
    finally:
        if saved is None:
            os.environ.pop(RUNNER_ABORT_FLAG_ENV, None)
        else:
            os.environ[RUNNER_ABORT_FLAG_ENV] = saved
    evidence = {
        "control": "round0246-a3bis-reviewer-directory-control-v1",
        "planted": f"ROUNDRUN_ABORT_FLAG pointed at the writable directory {directory!r}",
        "what_r0245_did": (
            "precondition PASSED with directory_is_writable true, the watchdog "
            "ARMED, the guard fired, and abort_flag_written came back false "
            "with '[Errno 21] Is a directory'"
        ),
        "precondition_refused_to_start": precondition_refused,
        "precondition_message": precondition_message,
        "watchdog_refused_to_arm": watchdog_refused,
        "watchdog_message": watchdog_message,
        "environment_restored": bool(
            os.environ.get(RUNNER_ABORT_FLAG_ENV) == saved
        ),
        "directory_still_exists": bool(os.path.isdir(directory)),
    }
    failures = [
        name for name, ok in (
            ("precondition_refused", precondition_refused),
            ("watchdog_refused", watchdog_refused),
            ("environment_restored", evidence["environment_restored"]),
            ("the_probe_did_not_destroy_the_directory",
             evidence["directory_still_exists"]),
        ) if not ok
    ]
    evidence["failures"] = failures
    evidence["holds"] = not failures
    if failures:
        raise Round0246Error(
            f"R0246 A3-BIS REVIEWER CONTROL DID NOT FIRE: {failures}"
        )
    return evidence


#: The gap review-0245-01 replayed through R0245's corrected gate at
#: `measured_slope_bytes_per_s = 0.0` and watched PASS. It is attempt 1's exact
#: failure, taken from the sealed `did_0245` failure log.
ATTEMPT_1_GAP_S = 5.828025072987657
R0245_NODE_HEADROOM_BYTES = 47_244_640_256


def run_reviewer_gap_replay_control(
    *, gap_s: float = ATTEMPT_1_GAP_S,
    headroom_bytes: int = R0245_NODE_HEADROOM_BYTES,
) -> dict[str, Any]:
    """review-0245-01 D, exactly: attempt 1's gap at own-slope `0.0`.

    Also replays the two trivial defeats — a stage that calls the tracker once
    and a stage that never calls it — both of which held under R0245.
    """
    replay = replay_gap_through_the_gate(
        gap_s=float(gap_s), headroom_bytes=int(headroom_bytes),
        measured_slope_bytes_per_s=0.0,
        label="R0246 attempt-1 gap replay at own-slope 0.0",
    )
    r0245_form = poll_spacing_requirement(
        slope_bytes_per_s=max(0.0, 1.0), headroom_bytes=int(headroom_bytes),
        poll_spacing_s=float(gap_s),
    )

    def _empty(polls: int) -> dict[str, Any]:
        ticks = iter([0.0] * (polls + 3))
        gate = AbortPollGate(
            inner=lambda _where: None, headroom_bytes=int(headroom_bytes),
            label=f"R0246 {polls}-poll stage", clock=lambda: next(ticks),
            replay=True,
        )
        gate.start()
        for index in range(polls):
            gate(f"read {index}")
        gate.finish()
        refused = False
        message = None
        try:
            gate.require(measured_slope_bytes_per_s=0.0)
        except Round0246Error as error:
            refused = True
            message = str(error)
        return {
            "polls": polls, "max_gap_s": gate.max_gap_s,
            "gate_refused_it": refused, "message": message,
        }

    evidence = {
        "control": "round0246-a2bis-reviewer-gap-replay-control-v1",
        "planted": (
            "attempt 1's exact 5.828025072987657 s gap replayed at "
            "measured_slope_bytes_per_s = 0.0, plus a stage that calls the "
            "tracker once and a stage that never calls it"
        ),
        "what_r0245_did": (
            "binding slope 1.0 B/s, max_poll_spacing_s 47244640256.0, the gap "
            "PASSED; polls 1 and polls 0 both held on max_gap 0.0"
        ),
        "r0245_corrected_form_max_poll_spacing_s": r0245_form[
            "max_poll_spacing_s"
        ],
        "r0245_corrected_form_would_hold": bool(r0245_form["requirement_holds"]),
        "replay": replay,
        "one_poll_stage": _empty(1),
        "zero_poll_stage": _empty(0),
    }
    failures = [
        name for name, ok in (
            ("attempt_1_gap_is_refused", replay["gate_refused_it"]),
            ("one_poll_stage_is_refused",
             evidence["one_poll_stage"]["gate_refused_it"]),
            ("zero_poll_stage_is_refused",
             evidence["zero_poll_stage"]["gate_refused_it"]),
            ("the_r0245_form_really_did_pass_it",
             evidence["r0245_corrected_form_would_hold"]),
        ) if not ok
    ]
    evidence["failures"] = failures
    evidence["holds"] = not failures
    if failures:
        raise Round0246Error(
            f"R0246 A2-BIS REVIEWER CONTROL DID NOT FIRE: {failures}"
        )
    return evidence


def _cleanup(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# the novel attacks — three or more per guard, none covered by a control
# --------------------------------------------------------------------------- #
def _flag_attack(path: str, *, shape: str) -> dict[str, Any]:
    """Point the precondition and the watchdog at `path` and see what happens."""
    saved = os.environ.get(RUNNER_ABORT_FLAG_ENV)
    precondition_refused = False
    watchdog_refused = False
    message = None
    try:
        os.environ[RUNNER_ABORT_FLAG_ENV] = path
        try:
            require_enforceable_abort_flag(label=f"R0246 {shape}")
        except Round0245Error as error:
            precondition_refused = True
            message = str(error)
        try:
            EnforcedHostWatchdog(label=f"R0246 {shape}")
        except Round0245Error:
            watchdog_refused = True
    finally:
        if saved is None:
            os.environ.pop(RUNNER_ABORT_FLAG_ENV, None)
        else:
            os.environ[RUNNER_ABORT_FLAG_ENV] = saved
    return {
        "attack": shape,
        "path": path,
        "precondition_refused": precondition_refused,
        "watchdog_refused": watchdog_refused,
        "closed": bool(precondition_refused and watchdog_refused),
        "message": message,
    }


def run_novel_attack_battery(*, workspace: str) -> dict[str, Any]:
    """Attacks no positive control covers. Reports which succeed.

    A guard is only as good as the attacks it has survived, so this runs
    against the CLOSED guards and publishes `closed: false` for anything that
    still gets through rather than quietly omitting it.
    """
    os.makedirs(workspace, exist_ok=True)
    attacks: list[dict[str, Any]] = []

    # ---- guard 1: sampler liveness --------------------------------------- #
    starved = _liveness_attack(
        watchdog=_StarvedWatchdog(
            stall_s=1.0, anonymous_budget_bytes=1 << 62, interval_s=0.05,
            abort_flag_path=os.path.join(workspace, "starved.abort"),
            label="R0246 starved sampler",
        ),
        wall_s=2.0,
    )
    starved.update({
        "attack": "a1-novel-1: the sampler never raises and barely samples",
        "why_it_is_novel": (
            "every A1 control so far plants an EXCEPTION. A thread that is "
            "alive, healthy and simply not sampling produces a clean receipt "
            "under R0245's fix and under the OSError fix alone."
        ),
        "closed": bool(starved["gate_refused_the_receipt"]),
    })
    attacks.append(starved)

    inflated = _liveness_attack(
        watchdog=_StarvedWatchdog(
            stall_s=1.0, anonymous_budget_bytes=1 << 62, interval_s=0.05,
            abort_flag_path=os.path.join(workspace, "inflated.abort"),
            label="R0246 poll-inflated coverage",
        ),
        wall_s=2.0, poll_every_s=0.005,
    )
    inflated.update({
        "attack": "a1-novel-2: hold sample_coverage up from the main thread",
        "why_it_is_novel": (
            "R0244's published sample_coverage counts the samples poll() takes "
            "at stage boundaries, so a main thread polling hard keeps the "
            "coverage figure high while the sampling thread does nothing. A "
            "coverage floor read off sample_coverage would be defeated by it; "
            "the gate reads thread_sample_coverage."
        ),
        "closed": bool(inflated["gate_refused_the_receipt"]),
    })
    attacks.append(inflated)

    silent = _liveness_attack(
        watchdog=_SilentReturnWatchdog(
            anonymous_budget_bytes=1 << 62, interval_s=0.05,
            abort_flag_path=os.path.join(workspace, "silent.abort"),
            label="R0246 silent return",
        ),
        wall_s=1.5,
    )
    silent.update({
        "attack": "a1-novel-3: the sampling loop returns, raising nothing",
        "why_it_is_novel": (
            "no exception is ever raised, so neither the OSError arm nor the "
            "BaseException arm sees anything. Only an exit sentinel does."
        ),
        "closed": bool(
            silent["thread_death"] is not None
            and silent["gate_refused_the_receipt"]
        ),
    })
    attacks.append(silent)

    intermittent = _liveness_attack(
        watchdog=_IntermittentWatchdog(
            fail_of=3, anonymous_budget_bytes=1 << 62, interval_s=0.05,
            abort_flag_path=os.path.join(workspace, "intermittent.abort"),
            label="R0246 intermittent OSError",
        ),
        wall_s=2.0,
    )
    intermittent.update({
        "attack": (
            "a1-novel-4: fail 2 of every 3 samples, never 4 in a row - under "
            "the registered consecutive-failure tolerance by construction"
        ),
        "why_it_is_novel": (
            "it defeats the consecutive-failure arm on purpose. It is caught, "
            "if at all, only by the coverage floor, which is the point of "
            "having two arms."
        ),
        #: RECLASSIFIED BY R0247, and the reclassification is the finding.
        #: Under R0246 this row was "closed" because the coverage floor read a
        #: ratio to the sampler's OWN declared 0.05 s interval, so a sampler
        #: that missed two of every three samples scored 0.324 and was refused.
        #: Under the registered denominator it observes every ~0.15 s, which is
        #: FASTER than the registered 0.25 s interval requires - so it was
        #: never blind, and R0246's floor was punishing a node for declaring a
        #: stricter interval than it achieved rather than for being unsafe.
        #: The arm that matters is the measured observation gap, and this
        #: sampler meets it. What must still be true is that the degradation is
        #: RECORDED. R0247 adds `a1-r0247-4`, which makes the same sampler
        #: genuinely blind, and that one is refused.
        "reclassified_by_r0247": (
            "not an attack once the coverage denominator is the REGISTERED "
            "interval: this sampler observes more often than the registered "
            "0.25 s and is inside both measured observation-gap bounds. It is "
            "scored on those bounds plus the requirement that its failures are "
            "recorded, not on a ratio to its own declaration."
        ),
        "closed": bool(
            intermittent["gate_refused_the_receipt"]
            or (
                intermittent["sample_failures"] > 0
                and float(intermittent["max_thread_sample_gap_s"])
                <= WATCHDOG_MAX_OBSERVATION_GAP_S
                and float(intermittent["mean_thread_sample_gap_s"])
                <= WATCHDOG_MAX_MEAN_OBSERVATION_GAP_S
            )
        ),
    })
    attacks.append(intermittent)

    #: a1-r0247-4 — the version of a1-novel-4 that IS an attack: the same
    #: never-four-in-a-row failure pattern, stalled so the sampler is genuinely
    #: blind for longer than the registered observation ceiling.
    blind = _liveness_attack(
        watchdog=_IntermittentWatchdog(
            fail_of=3, stall_s=1.4, anonymous_budget_bytes=1 << 30,
            interval_s=0.05,
            abort_flag_path=os.path.join(workspace, "blind.abort"),
            label="R0247 blind intermittent sampler",
        ),
        wall_s=6.0,
    )
    blind.update({
        "attack": (
            "a1-r0247-4: fail 2 of every 3 samples AND stall the successful "
            "one past the registered observation ceiling"
        ),
        "why_it_is_novel": (
            "it is a1-novel-4 made real. The consecutive-failure arm still "
            "cannot see it by construction, the registered-denominator "
            "coverage arm cannot see it either because the sampler declares a "
            "stricter interval than it keeps, and only a bound measured in "
            "SECONDS between two successful observations refuses it."
        ),
        "closed": bool(blind["gate_refused_the_receipt"]),
    })
    attacks.append(blind)

    # ---- guard 2: the abort flag path ------------------------------------ #
    fifo = os.path.join(workspace, "fifo.abort")
    try:
        os.mkfifo(fifo)
    except OSError:  # pragma: no cover - only if the fs cannot do FIFOs
        fifo = ""
    if fifo:
        row = _flag_attack(fifo, shape="a3-novel-1: the flag path is a FIFO")
        row["why_it_is_novel"] = (
            "a FIFO is not a directory, so a stat-for-directory fix misses it, "
            "and open(path, 'w') on a FIFO with no reader BLOCKS FOREVER - the "
            "trip would hang the sampling thread rather than fail it."
        )
        attacks.append(row)
        os.remove(fifo)

    readonly = os.path.join(workspace, "readonly.abort")
    with open(readonly, "w", encoding="utf-8") as handle:
        handle.write("pre-existing\n")
    os.chmod(readonly, stat.S_IRUSR)
    row = _flag_attack(
        readonly, shape="a3-novel-2: the flag path is a read-only regular file"
    )
    row["why_it_is_novel"] = (
        "the path exists AND is a regular file AND its directory is writable, "
        "so every existence-shaped check passes and the trip still cannot be "
        "written."
    )
    row["file_survived_the_probe"] = bool(os.path.exists(readonly))
    row["closed"] = bool(row["closed"] and row["file_survived_the_probe"])
    attacks.append(row)
    os.chmod(readonly, stat.S_IRUSR | stat.S_IWUSR)
    os.remove(readonly)

    devnull = os.path.join(workspace, "devnull.abort")
    os.symlink("/dev/null", devnull)
    row = _flag_attack(
        devnull,
        shape="a3-novel-3: the flag path resolves to a character device",
    )
    row["why_it_is_novel"] = (
        "writable, openable, a successful write - and the trip is discarded. "
        "abort_flag_written would read TRUE and runner_abort_reason() would "
        "never see a flag, which is worse than an error because the node-tail "
        "gate would pass it too."
    )
    attacks.append(row)
    os.remove(devnull)

    dangling = os.path.join(workspace, "dangling.abort")
    os.symlink(os.path.join(workspace, "nowhere", "target"), dangling)
    row = _flag_attack(
        dangling, shape="a3-novel-4: the flag path is a dangling symlink"
    )
    row["why_it_is_novel"] = (
        "os.path.exists is false and os.path.dirname is writable, so a "
        "create-if-absent probe that used open(..., 'w') would follow the link "
        "into a missing directory at TRIP time, not at ARM time."
    )
    attacks.append(row)
    os.remove(dangling)

    linked_dir = os.path.join(workspace, "linked-dir")
    os.makedirs(os.path.join(workspace, "realdir"), exist_ok=True)
    os.symlink(os.path.join(workspace, "realdir"), linked_dir)
    row = _flag_attack(
        linked_dir, shape="a3-novel-5: the flag path is a symlink to a directory"
    )
    row["why_it_is_novel"] = (
        "lstat says symlink, not directory; only a stat that FOLLOWS the link "
        "refuses it."
    )
    attacks.append(row)
    os.remove(linked_dir)

    toctou = _toctou_attack(workspace=workspace)
    attacks.append(toctou)

    # ---- guard 3: the poll-spacing gate ---------------------------------- #
    ticks = iter([0.0, 3.0, 3.0, 3.0])
    late = AbortPollGate(
        inner=lambda _where: None, headroom_bytes=R0245_NODE_HEADROOM_BYTES,
        label="R0246 late first read", clock=lambda: next(ticks),
        replay=True,
    )
    late.start()
    late("the only read, taken 3 s in")
    late.finish()
    refused = False
    try:
        late.require(measured_slope_bytes_per_s=0.0)
    except Round0246Error:
        refused = True
    attacks.append({
        "attack": "a2-novel-1: allocate for 3 s, then take the first read",
        "why_it_is_novel": (
            "R0245 measured only the intervals BETWEEN reads, so a trainer "
            "could allocate its whole peak before its first read and be scored "
            "on a gap of zero."
        ),
        "gap_before_the_first_poll_s": late.gap_before_the_first_poll_s,
        "max_gap_s": late.max_gap_s,
        "gate_refused_it": refused,
        "closed": refused,
    })

    ticks = iter([0.0, 0.0, 0.01, 3.0])
    trailing = AbortPollGate(
        inner=lambda _where: None, headroom_bytes=R0245_NODE_HEADROOM_BYTES,
        label="R0246 trailing allocation", clock=lambda: next(ticks),
        replay=True,
    )
    trailing.start()
    trailing("read 1")
    trailing("read 2")
    trailing.finish()
    refused = False
    try:
        trailing.require(measured_slope_bytes_per_s=0.0)
    except Round0246Error:
        refused = True
    attacks.append({
        "attack": "a2-novel-2: two reads at the start, then 3 s of allocation",
        "why_it_is_novel": (
            "the interval from the last read to stage end was never measured, "
            "so a stage could satisfy the gate and then run unpolled to its "
            "end."
        ),
        "gap_after_the_last_poll_s": trailing.gap_after_the_last_poll_s,
        "max_gap_s": trailing.max_gap_s,
        "gate_refused_it": refused,
        "closed": refused,
    })

    ticks = iter([0.0, 0.0, 3.0, 3.0])
    inflated_headroom = AbortPollGate(
        inner=lambda _where: None, headroom_bytes=1 << 50,
        label="R0246 inflated headroom", clock=lambda: next(ticks),
        replay=True,
    )
    inflated_headroom.start()
    inflated_headroom("read 1")
    inflated_headroom("read 2")
    inflated_headroom.finish()
    refused = False
    try:
        inflated_headroom.require(measured_slope_bytes_per_s=0.0)
    except Round0246Error:
        refused = True
    attacks.append({
        "attack": "a2-novel-3: declare a 1 PiB headroom and keep the 3 s gap",
        "why_it_is_novel": (
            "the requirement is spacing <= headroom / slope, so the headroom is "
            "an input the node chooses. Nothing in R0245 stopped a node buying "
            "a wider spacing by declaring a bigger budget."
        ),
        "declared_headroom_bytes": 1 << 50,
        "max_gap_s": inflated_headroom.max_gap_s,
        "gate_refused_it": refused,
        "closed": refused,
    })

    ticks = iter([0.0, 0.0, 0.01, 0.02])
    weak_slope = AbortPollGate(
        inner=lambda _where: None, headroom_bytes=R0245_NODE_HEADROOM_BYTES,
        label="R0246 weak worst-case slope", slope_bytes_per_s=1.0,
        clock=lambda: next(ticks), replay=True,
    )
    weak_slope.start()
    weak_slope("read 1")
    weak_slope("read 2")
    weak_slope.finish()
    refused = False
    try:
        weak_slope.require(measured_slope_bytes_per_s=0.0)
    except Round0246Error:
        refused = True
    attacks.append({
        "attack": "a2-novel-4: construct the gate with a 1 B/s worst-case slope",
        "why_it_is_novel": (
            "the worst-case column is a CONSTRUCTOR argument. Flooring only the "
            "own-slope branch would leave a node free to declare a harmless "
            "worst case."
        ),
        "gate_refused_it": refused,
        "closed": refused,
    })

    ticks = iter([0.0, 0.0, 0.01])
    unfinished = AbortPollGate(
        inner=lambda _where: None, headroom_bytes=R0245_NODE_HEADROOM_BYTES,
        label="R0246 unfinished gate", clock=lambda: next(ticks),
        replay=True,
    )
    unfinished.start()
    unfinished("read 1")
    unfinished("read 2")
    refused = False
    try:
        unfinished.require(measured_slope_bytes_per_s=0.0)
    except Round0246Error:
        refused = True
    attacks.append({
        "attack": "a2-novel-5: never call finish(), so the tail is unmeasured",
        "why_it_is_novel": (
            "the tail-interval fix is defeated simply by not closing the gate. "
            "The gate has to refuse a verdict it was never asked to complete."
        ),
        "gate_refused_it": refused,
        "closed": refused,
    })

    succeeded = [row["attack"] for row in attacks if not row.get("closed")]
    return {
        "instrument": "round0246-novel-attack-battery-v1",
        "attacks": attacks,
        "attacks_run": len(attacks),
        "attacks_closed": sum(1 for row in attacks if row.get("closed")),
        "attacks_that_still_succeed": succeeded,
        "every_novel_attack_is_closed": not succeeded,
        "note": (
            "these are attacks NO positive control in this round covers. They "
            "are published whether or not they are closed; an attack that still "
            "succeeds is a stated limit of the guard, not an omission."
        ),
    }


def _toctou_attack(*, workspace: str) -> dict[str, Any]:
    """Pass the precondition, then replace the path with a directory.

    This one is not closed at the precondition and cannot be — the check and the
    trip are separated in time. What must catch it is the node-tail gate on
    `abort_flag_written`.
    """
    path = os.path.join(workspace, "toctou.abort")
    saved = os.environ.get(RUNNER_ABORT_FLAG_ENV)
    precondition_passed = False
    tail_gate_refused = False
    receipt: dict[str, Any] = {}
    try:
        os.environ[RUNNER_ABORT_FLAG_ENV] = path
        require_enforceable_abort_flag(label="R0246 TOCTOU attack")
        precondition_passed = True
        watchdog = EnforcedHostWatchdog(
            anonymous_budget_bytes=1, interval_s=0.02, abort_flag_path=path,
            label="R0246 TOCTOU attack",
        )
        os.makedirs(path, exist_ok=True)
        watchdog.start()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with watchdog._lock:  # noqa: SLF001 - the attack inspects its subject
                if watchdog.fired:
                    break
            time.sleep(0.01)
        watchdog.stop()
        receipt = watchdog.receipt()
        try:
            require_abort_flag_landed(receipt, label="R0246 TOCTOU attack")
        except Round0246Error:
            tail_gate_refused = True
    finally:
        if saved is None:
            os.environ.pop(RUNNER_ABORT_FLAG_ENV, None)
        else:
            os.environ[RUNNER_ABORT_FLAG_ENV] = saved
        try:
            os.rmdir(path)
        except OSError:
            pass
    return {
        "attack": (
            "a3-novel-6: pass the precondition, then replace the path with a "
            "directory before the trip"
        ),
        "why_it_is_novel": (
            "no arm-time probe can close a time-of-check/time-of-use race. The "
            "only thing that can is a gate on whether the trip LANDED, which "
            "R0245 published and nothing read."
        ),
        "precondition_passed_as_expected": precondition_passed,
        "guard_fired": bool(receipt.get("fired")),
        "abort_flag_written": bool(receipt.get("abort_flag_written")),
        "abort_flag_error": receipt.get("abort_flag_error"),
        "node_tail_gate_refused_it": tail_gate_refused,
        "closed": bool(precondition_passed and tail_gate_refused),
        "residual": (
            "closed only at the node tail. A trip that never happens leaves a "
            "broken flag path undetected for the life of the node; the arm-time "
            "probe narrows the window and does not remove it."
        ),
    }


def guard_closure_receipt(
    *,
    abort_flag: Mapping[str, Any],
    oserror_control: Mapping[str, Any],
    directory_control: Mapping[str, Any],
    gap_replay_control: Mapping[str, Any],
    novel_attacks: Mapping[str, Any],
    healthy_coverage: Mapping[str, Any],
) -> dict[str, Any]:
    """The three closures, their reviewer-shaped controls, and the attacks."""
    arms = {
        "a1_the_oserror_arm_is_closed": bool(oserror_control["holds"]),
        "a3_the_flag_path_is_validated_not_its_directory": bool(
            directory_control["holds"]
        ),
        "a2_attempt_1s_gap_fails_again": bool(
            gap_replay_control["replay"]["gate_refused_it"]
        ),
        "a2_zero_and_one_poll_stages_fail": bool(
            gap_replay_control["zero_poll_stage"]["gate_refused_it"]
            and gap_replay_control["one_poll_stage"]["gate_refused_it"]
        ),
        "every_novel_attack_is_closed": bool(
            novel_attacks["every_novel_attack_is_closed"]
        ),
        "the_precondition_probed_the_path": bool(
            abort_flag["path_is_writable"]
        ),
    }
    return {
        "instrument": "round0246-guard-closure-v1",
        "abort_flag_precondition": dict(abort_flag),
        "a1_reviewer_oserror_control": dict(oserror_control),
        "a3_reviewer_directory_control": dict(directory_control),
        "a2_reviewer_gap_replay_control": dict(gap_replay_control),
        "novel_attacks": dict(novel_attacks),
        "healthy_coverage_basis": dict(healthy_coverage),
        "registrations": registered_thresholds(),
        "arms": arms,
        "holds": all(arms.values()),
        "guard_axis": "anonymous, never RSS",
    }


def registered_thresholds() -> dict[str, Any]:
    """Every number R0246 registers, with the basis for each.

    No threshold moves without appearing here. Two of these REPLACE numbers
    R0245 shipped, and both replacements are strictly tighter.
    """
    return {
        "watchdog_max_consecutive_sample_failures": {
            "value": WATCHDOG_MAX_CONSECUTIVE_SAMPLE_FAILURES,
            "replaces": "an unbounded `except OSError: pass`",
            "direction": "tighter",
            "basis": (
                "n consecutive failures blind the guard for n * "
                f"{WATCHDOG_SAMPLE_INTERVAL_S} s, permitting n * "
                f"{WATCHDOG_SAMPLE_INTERVAL_S} * "
                f"{R0244_MEASURED_SLOPE_BYTES_PER_S} B at the measured worst "
                f"case. Holding that inside {R0244_BUDGET_HEADROOM_BYTES} B of "
                "sealed headroom admits n <= 10.04; the registered 3 is 3.35x "
                "stricter."
            ),
            "what_it_does_not_catch": (
                "a sampler failing fewer than 4 samples in a row forever. That "
                "is the coverage floor's job."
            ),
        },
        "min_thread_sample_coverage": {
            "value": MIN_THREAD_SAMPLE_COVERAGE,
            "replaces": "no coverage gate at all",
            "direction": "new, tighter",
            "basis": (
                "the derived safety floor is 0.25 * 11767996416 / 29548888064 "
                "= 0.09956; the measured healthy value on R0245's sealed "
                "114.54487995902309 s sampler node is (439 - 4) / "
                "458.17951983609237 = 0.9494. The registered 0.50 is 5.02x "
                "stricter than the safety floor and half the healthy value."
            ),
            "what_it_does_not_catch": (
                "a sampler running at just above 0.50 coverage, i.e. a mean "
                "observation gap of just under 0.5 s. That gap is still inside "
                "the headroom inequality, which is why 0.50 is a conservative "
                "floor rather than the binding one."
            ),
        },
        "coverage_gate_min_expected_samples": {
            "value": COVERAGE_GATE_MIN_EXPECTED_SAMPLES,
            "direction": "scope of the coverage gate",
            "basis": (
                "at 20 expected samples one missed sample moves coverage by "
                "0.05, so quantisation cannot carry a healthy 0.95 to the 0.50 "
                "floor. Below 20 the ratio is noise; R0245's guard node tail "
                "reported a coverage of 2.4963609963523510 over 0.6 s for "
                "exactly this reason."
            ),
            "what_it_does_not_catch": (
                "a dead sampler inside a stage shorter than 20 sampling "
                "intervals. The death arms still catch that; only the coverage "
                "arm is scoped out."
            ),
        },
        "min_binding_slope_bytes_per_s": {
            "value": MIN_BINDING_SLOPE_BYTES_PER_S,
            "replaces": "max(measured_slope, 1.0) B/s",
            "direction": "tighter by 1.1767996416e10",
            "basis": MIN_BINDING_SLOPE_BASIS,
            "what_it_does_not_catch": (
                "a stage that allocates faster than 11,767,996,416 B/s without "
                "its own one-second trace showing it. The floor is the fastest "
                "rate MEASURED on this machine, not a proof about all stages."
            ),
        },
        "r0246_max_poll_spacing_s": {
            "value": R0246_MAX_POLL_SPACING_S,
            "replaces": "nothing - the ceiling was the node's own headroom",
            "direction": "new, tighter",
            "basis": (
                f"{R0244_BUDGET_HEADROOM_BYTES} B of R0244 sealed headroom over "
                f"{R0244_MEASURED_SLOPE_BYTES_PER_S} B/s of R0244 sealed "
                "measured slope. It is the derived requirement of review-0245-01 "
                "section B and it is applied regardless of the headroom a node "
                "declares, so a node cannot buy spacing with a larger budget."
            ),
            "what_it_does_not_catch": (
                "an allocation burst inside one unit of work. The gate bounds "
                "the spacing of reads, not what happens between two of them."
            ),
        },
    }


__all__ = [
    "ATTEMPT_1_GAP_S",
    "SAFETY_PARAMETER_NOTE",
    "COVERAGE_GATE_MIN_EXPECTED_SAMPLES",
    "GPU_HOURS_CAP",
    "MIN_ENFORCEMENT_POLLS",
    "MIN_THREAD_SAMPLE_COVERAGE",
    "POLL_GATE_NOTE",
    "R0245_NODE_HEADROOM_BYTES",
    "R0246_MAX_POLL_SPACING_S",
    "ROUND_ID",
    "ROWS",
    "SAMPLER_LIVENESS_NOTE",
    "AbortPollGate",
    "Round0246Error",
    "guard_closure_receipt",
    "measured_slope_from_trace",
    "registered_thresholds",
    "replay_gap_through_the_gate",
    "require_enforcement_evidence",
    "require_abort_flag_landed",
    "require_live_sampler",
    "run_novel_attack_battery",
    "run_reviewer_directory_flag_control",
    "run_reviewer_gap_replay_control",
    "run_reviewer_oserror_control",
]
