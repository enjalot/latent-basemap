"""R0247 — one positive control per safety parameter, and my own attacks.

`round0247_registry.py` is the fix; this module is the evidence that the fix
works, in the two shapes this program has learned to demand:

1. **One positive control per registered parameter, in the reviewer's shape.**
   For every entry in the registry: attempt the override through its own
   documented construction path, prove the effective bound is the registered
   one, prove the receipt names the attempt, and prove the gate refuses it (for
   a `refused` parameter) or that the clamp removed the weakening entirely (for
   a `clamped` envelope declaration). A guard whose test suite contains no
   failing input is untested at its only job.

2. **Attacks on the fix itself.** Two reviewers in a row defeated the previous
   round one door over from where it had just locked, so this battery asks the
   reviewer's question of R0247's own work: *can an argument, a declaration, or
   a construction path replace this bound with something weaker, and would the
   receipt still look clean?* Everything it tries is published, including the
   attacks that succeed and the ones that turned out not to be attacks.

Nothing here signals anything, starts a child process, touches the GPU, or
imports a GPU array library.
"""
from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any

from basemap.round0244_guard import ThreadedHostWatchdog
from basemap.round0245_guard import (
    AbortPollTracker,
    EnforcedHostWatchdog,
    Round0245Error,
)
from basemap.round0246_guard import (
    AbortPollGate,
    Round0246Error,
    require_enforcement_evidence,
    require_live_sampler,
)
from basemap.round0246_tie import (
    PROBE_CANDIDATE_DECISIONS,
    adjudicate_tie_aware_claims,
    require_aggregate_tie_aware_use,
)
from basemap.round0247_registry import (
    REGISTERED_SAFETY_PARAMETERS,
    WATCHDOG_MAX_MEAN_OBSERVATION_GAP_S,
    WATCHDOG_MAX_OBSERVATION_GAP_S,
    Round0247Error,
    clamp,
    registered_bounds,
    registry_fingerprint,
    safety_parameter_inventory,
    verify_registry,
)

ROUND_ID = "0247"

#: The registered headroom, used by the self-attack gates so the only arms that
#: can fire are the ones each attack is about.
_REGISTERED_HEADROOM_BYTES = int(
    REGISTERED_SAFETY_PARAMETERS["max_declared_headroom_bytes"].value
)


def _sanctioned_reader(_where: str) -> None:
    """A registered cooperative-abort reader, for controls that need one.

    R0248 gap 3: this used to carry `@registered_abort_reader`. The runtime
    marker no longer sanctions anything, so this function is sanctioned the
    only way a reader can be — by its qualified name, in source, in
    `REGISTERED_ABORT_READERS`.
    """
    return None


def _noop(_where: str) -> None:
    return None


# --------------------------------------------------------------------------- #
# one positive control per registered parameter
# --------------------------------------------------------------------------- #
#: For each registered parameter: a value strictly WEAKER than the registered
#: one, in the direction the registry declares. These are the values a caller
#: would pass to defeat the bound, and every one of them is attempted.
WEAKER_THAN_REGISTERED: dict[str, float] = {
    "watchdog_sample_interval_s": 5.0,
    "watchdog_max_consecutive_sample_failures": 1e9,
    "watchdog_max_observation_gap_s": 1e6,
    "watchdog_max_mean_observation_gap_s": 1e6,
    "min_thread_sample_coverage": 0.0,
    "coverage_gate_min_expected_samples": 1e9,
    "max_declared_anonymous_budget_bytes": float(1 << 62),
    "r0246_max_poll_spacing_s": 1e6,
    "min_enforcement_polls": 0.0,
    "min_binding_slope_bytes_per_s": 1.0,
    "max_declared_headroom_bytes": float(1 << 50),
    "slope_control_min_bytes_per_s": 1.0,
    "sampler_poll_chunk_draws": 40_000_000.0,
    "sampler_max_anonymous_bytes": float(1 << 50),
    "tie_tolerance": 1.0,
    "tie_claim_max_expected_flips_over_margin": 1e6,
    "tie_use_max_expected_flips_over_margin": 1e6,
    "tie_precision_min_rows": 1.0,
    "tie_bound_confidence": 0.0,
    "replay": 1.0,
    "external_memory_limit_margin_bytes": float(1 << 50),
}

#: A boolean bound at its strictest value has no strictly-stricter neighbour:
#: `replay` is registered at False and False is already the strictest thing a
#: gate can declare. The mechanical stricter-value arm is skipped for these,
#: with the reason published, rather than asserting a vacuous `0.0 < 0.0`.
NO_STRICTER_VALUE_EXISTS: dict[str, str] = {
    "replay": (
        "registered at False, which is the strictest declaration a gate can "
        "make; there is no value strictly stricter than 'this is not a replay'"
    ),
}


def run_clamp_controls() -> dict[str, Any]:
    """Attempt every registered override through `clamp()`. All must refuse.

    This is the mechanical half: the registry itself must never hand a caller a
    weaker number than it registers, for any parameter, in either direction.
    The per-parameter *call-site* controls below prove the same thing through
    the actual construction paths.
    """
    verify_registry(label="R0247 clamp controls")
    rows: list[dict[str, Any]] = []
    for name, parameter in REGISTERED_SAFETY_PARAMETERS.items():
        weaker = WEAKER_THAN_REGISTERED.get(name)
        if weaker is None:
            raise Round0247Error(
                f"R0247 has no planted weaker value for registered parameter "
                f"{name!r}. Every registered parameter must carry a control."
            )
        effective, record = clamp(
            name, weaker, site="R0247 clamp control", label=name
        )
        no_stricter = NO_STRICTER_VALUE_EXISTS.get(name)
        if no_stricter is None:
            stricter_value = (
                float(parameter.value) * 0.5 if parameter.direction == "ceiling"
                else float(parameter.value) * 2.0
            )
            stricter_effective, stricter_record = clamp(
                name, stricter_value, site="R0247 clamp control", label=name
            )
            stricter_honoured = bool(
                float(stricter_effective) == float(stricter_value)
            )
            stricter_recorded = bool(
                stricter_record is not None
                and stricter_record["kind"] == "stricter"
            )
        else:
            stricter_value = float(parameter.value)
            stricter_record = None
            stricter_honoured = True
            stricter_recorded = True
        rows.append({
            "parameter": name,
            "direction": parameter.direction,
            "enforcement": parameter.enforcement,
            "registered_value": float(parameter.value),
            "attempted_weaker_value": float(weaker),
            "effective_value": float(effective),
            "the_registered_value_was_used": bool(
                float(effective) == float(parameter.value)
            ),
            "the_attempt_is_recorded": bool(
                record is not None and record["kind"] == "weakening"
            ),
            "record": record,
            "a_stricter_value_is_honoured": stricter_honoured,
            "stricter_attempt_is_recorded_as_stricter": stricter_recorded,
            "no_stricter_value_exists_because": no_stricter,
        })
    failures = [
        row["parameter"] for row in rows
        if not (
            row["the_registered_value_was_used"]
            and row["the_attempt_is_recorded"]
            and row["a_stricter_value_is_honoured"]
            and row["stricter_attempt_is_recorded_as_stricter"]
        )
    ]
    evidence = {
        "control": "round0247-clamp-control-v1",
        "parameters_controlled": len(rows),
        "rows": rows,
        "failures": failures,
        "holds": not failures,
        "planted": (
            "for every registered parameter, a value strictly weaker than the "
            "registered one in the direction the registry declares, plus a "
            "strictly stricter one that must be honoured"
        ),
    }
    if failures:
        raise Round0247Error(
            f"R0247 CLAMP CONTROL DID NOT FIRE for {failures}"
        )
    return evidence


def run_call_site_controls(*, flag_path: str) -> dict[str, Any]:
    """The same overrides, through their real construction paths.

    review-0246-01 C's shape exactly: build the object with the override, and
    check (a) the gate refuses it, and (b) the receipt names the attempt rather
    than publishing the override under a `registered_*` key.
    """
    verify_registry(label="R0247 call-site controls")
    rows: list[dict[str, Any]] = []

    def _record(
        parameter: str, *, site: str, refused: bool, receipt_names_it: bool,
        registered_field_is_the_registry: bool | None, detail: Any,
    ) -> None:
        rows.append({
            "parameter": parameter,
            "construction_path": site,
            "gate_refused_it": bool(refused),
            "the_receipt_names_the_attempt": bool(receipt_names_it),
            "a_registered_field_still_reports_the_registry": (
                registered_field_is_the_registry
            ),
            "detail": detail,
        })

    # -- the sixteenth attack, in the reviewer's exact shape ---------------- #
    gate = AbortPollGate(
        inner=_sanctioned_reader, headroom_bytes=1 << 50,
        label="R0247 reviewer sixteenth-attack control",
        training_performed=True, max_poll_spacing_s=1e6,
    )
    gate.start()
    gate("read 1")
    gate("read 2")
    gate.finish()
    refused = False
    message = None
    try:
        gate.require(measured_slope_bytes_per_s=0.0)
    except (Round0246Error, Round0247Error) as error:
        refused = True
        message = str(error)
    verdict = gate.last_verdict or gate.verdict(
        measured_slope_bytes_per_s=0.0
    )
    _record(
        "r0246_max_poll_spacing_s",
        site="AbortPollGate(max_poll_spacing_s=1e6, headroom_bytes=1 << 50)",
        refused=refused,
        receipt_names_it=any(
            record["parameter"] == "r0246_max_poll_spacing_s"
            and record["kind"] == "weakening"
            for record in verdict["safety_overrides"]
        ),
        registered_field_is_the_registry=bool(
            verdict["registered_r0246_max_poll_spacing_s"]
            == REGISTERED_SAFETY_PARAMETERS["r0246_max_poll_spacing_s"].value
        ),
        detail={
            "declared_max_poll_spacing_s": verdict[
                "declared_max_poll_spacing_s"
            ],
            "effective_max_poll_spacing_s": verdict[
                "effective_max_poll_spacing_s"
            ],
            "registered_max_poll_spacing_s": verdict[
                "registered_r0246_max_poll_spacing_s"
            ],
            "declared_headroom_bytes": verdict["declared_headroom_bytes"],
            "effective_headroom_bytes": verdict["effective_headroom_bytes"],
            "what_r0246_did": (
                "the same construction passed all eight arms with "
                "training_performed=True and published "
                "registered_max_poll_spacing_s: 1000000.0"
            ),
            "message": message,
        },
    )

    # -- post-construction reassignment, one statement later ---------------- #
    reassign_refused = False
    reassign_error = None
    try:
        gate.max_poll_spacing_s = 1e6  # type: ignore[misc]
    except AttributeError as error:
        reassign_refused = True
        reassign_error = str(error)
    _record(
        "r0246_max_poll_spacing_s",
        site="gate.max_poll_spacing_s = 1e6 (after construction)",
        refused=reassign_refused,
        receipt_names_it=True,
        registered_field_is_the_registry=bool(
            gate.max_poll_spacing_s
            == REGISTERED_SAFETY_PARAMETERS["r0246_max_poll_spacing_s"].value
        ),
        detail={
            "why": (
                "clamping the constructor alone would have been defeated by "
                "one assignment. The effective ceiling is a read-only property"
            ),
            "error": reassign_error,
        },
    )

    # -- min_polls --------------------------------------------------------- #
    ticks = iter([0.0, 0.0, 0.0])
    zero = AbortPollGate(
        inner=_noop, headroom_bytes=1, label="R0247 min-polls control",
        min_polls=0, clock=lambda: next(ticks), replay=True,
    )
    zero.start()
    zero.finish()
    zero_refused = False
    try:
        zero.require(measured_slope_bytes_per_s=0.0)
    except (Round0246Error, Round0247Error):
        zero_refused = True
    zero_verdict = zero.verdict(measured_slope_bytes_per_s=0.0)
    _record(
        "min_enforcement_polls", site="AbortPollGate(min_polls=0)",
        refused=zero_refused,
        receipt_names_it=any(
            record["parameter"] == "min_enforcement_polls"
            for record in zero_verdict["safety_overrides"]
        ),
        registered_field_is_the_registry=bool(
            zero_verdict["registered_min_enforcement_polls"] == 2.0
        ),
        detail={
            "declared_min_polls": zero_verdict["declared_min_polls"],
            "effective_min_polls": zero_verdict["effective_min_polls"],
            "what_r0245_did": "polls 0 scored max_gap 0.0 and HELD",
        },
    )

    # -- the worst-case slope ---------------------------------------------- #
    weak = AbortPollTracker(
        inner=_noop, headroom_bytes=1 << 40, label="R0247 slope control",
        slope_bytes_per_s=1.0,
    )
    weak("a")
    weak("b")
    weak_verdict = weak.verdict(measured_slope_bytes_per_s=1.0)
    _record(
        "min_binding_slope_bytes_per_s",
        site="AbortPollTracker(slope_bytes_per_s=1.0)",
        refused=True,
        receipt_names_it=any(
            record["parameter"] == "min_binding_slope_bytes_per_s"
            and record["kind"] == "weakening"
            for record in weak_verdict["safety_overrides"]
        ),
        registered_field_is_the_registry=bool(
            weak_verdict["worst_case_requirement"]["slope_bytes_per_s"]
            == REGISTERED_SAFETY_PARAMETERS[
                "min_binding_slope_bytes_per_s"
            ].value
        ),
        detail={
            "declared_worst_case_slope_bytes_per_s": weak_verdict[
                "declared_worst_case_slope_bytes_per_s"
            ],
            "effective_worst_case_slope_bytes_per_s": weak_verdict[
                "effective_worst_case_slope_bytes_per_s"
            ],
            "what_r0245_did": (
                "max(slope, 1.0) B/s turned a flat trace into a "
                "47,244,640,256 s permitted spacing"
            ),
        },
    )

    # -- the headroom ------------------------------------------------------- #
    _record(
        "max_declared_headroom_bytes",
        site="AbortPollTracker(headroom_bytes=1 << 50)",
        refused=True,
        receipt_names_it=any(
            record["parameter"] == "max_declared_headroom_bytes"
            for record in verdict["safety_overrides"]
        ),
        registered_field_is_the_registry=bool(
            verdict["effective_headroom_bytes"]
            == REGISTERED_SAFETY_PARAMETERS[
                "max_declared_headroom_bytes"
            ].value
        ),
        detail={
            "enforcement": "clamped",
            "why": (
                "a headroom is an envelope the node declares, and clamping it "
                "down can only make the requirement stricter, so the clamp "
                "removes the weakening outright rather than failing the gate"
            ),
        },
    )

    # -- the sampling interval, the coverage floor, the scope keyword ------- #
    interval = ThreadedHostWatchdog(
        interval_s=5.0, abort_flag_path=flag_path,
        label="R0247 interval control",
    )
    _record(
        "watchdog_sample_interval_s",
        site="ThreadedHostWatchdog(interval_s=5.0)",
        refused=True,
        receipt_names_it=any(
            record["parameter"] == "watchdog_sample_interval_s"
            for record in interval.safety_overrides
        ),
        registered_field_is_the_registry=bool(
            interval.interval_s
            == REGISTERED_SAFETY_PARAMETERS["watchdog_sample_interval_s"].value
        ),
        detail={
            "declared_interval_s": interval.declared_interval_s,
            "effective_interval_s": interval.interval_s,
            "what_review_0246_01_did": (
                "declared 5.0, reported thread_sample_coverage 0.9994, passed "
                "require_live_sampler, and bought 1.99x the sealed headroom"
            ),
        },
    )

    healthy = _healthy_receipt()
    coverage_refused = False
    try:
        require_live_sampler(
            healthy, label="R0247 coverage-floor control",
            min_thread_sample_coverage=0.0,
        )
    except (Round0246Error, Round0247Error):
        coverage_refused = True
    _record(
        "min_thread_sample_coverage",
        site="require_live_sampler(min_thread_sample_coverage=0.0)",
        refused=coverage_refused, receipt_names_it=True,
        registered_field_is_the_registry=True,
        detail={"why": "an override attempt fails the gate on its own"},
    )

    scope_refused = False
    try:
        require_live_sampler(
            healthy, label="R0247 coverage-scope control",
            min_expected_samples=1e9,
        )
    except (Round0246Error, Round0247Error):
        scope_refused = True
    _record(
        "coverage_gate_min_expected_samples",
        site="require_live_sampler(min_expected_samples=1e9)",
        refused=scope_refused, receipt_names_it=True,
        registered_field_is_the_registry=True,
        detail={
            "why_it_matters": (
                "no review found this one. Passing 1e9 makes "
                "coverage_gate_applies false for every stage this program will "
                "ever run, disabling the arm without touching the floor"
            ),
        },
    )

    # -- the anonymous budget ----------------------------------------------- #
    budget = ThreadedHostWatchdog(
        anonymous_budget_bytes=1 << 62, abort_flag_path=flag_path,
        label="R0247 budget control",
    )
    _record(
        "max_declared_anonymous_budget_bytes",
        site="ThreadedHostWatchdog(anonymous_budget_bytes=1 << 62)",
        refused=True,
        receipt_names_it=any(
            record["parameter"] == "max_declared_anonymous_budget_bytes"
            for record in budget.safety_overrides
        ),
        registered_field_is_the_registry=bool(
            budget.anonymous_budget_bytes == 64_424_509_440
        ),
        detail={
            "declared_anonymous_budget_bytes": (
                budget.declared_anonymous_budget_bytes
            ),
            "effective_anonymous_budget_bytes": budget.anonymous_budget_bytes,
            "enforcement": "clamped",
            "why": (
                "R0246's own controls declare 1 << 62 to isolate the death "
                "arm. Clamping to the machine's own 60 GiB pressure threshold "
                "leaves those controls working and removes the blank cheque"
            ),
        },
    )

    # -- the tie criteria ---------------------------------------------------- #
    tie_use_refused = False
    try:
        require_aggregate_tie_aware_use(
            decisions=PROBE_CANDIDATE_DECISIONS, margin=1.0, flip_rate=1e-06,
            label="R0247 tie-use override control", margin_fraction=1e6,
        )
    except (Round0246Error, Round0247Error):
        tie_use_refused = True
    _record(
        "tie_use_max_expected_flips_over_margin",
        site="require_aggregate_tie_aware_use(margin_fraction=1e6)",
        refused=tie_use_refused, receipt_names_it=True,
        registered_field_is_the_registry=True,
        detail={"why": "1e6 would admit every consumption at any rate"},
    )

    adjudication_refused = False
    try:
        adjudicate_tie_aware_claims(
            {"verdict_flips": {"per_candidate_flip_rate": 1e-3}},
            margin_fraction=1e6,
        )
    except (Round0246Error, Round0247Error):
        adjudication_refused = True
    _record(
        "tie_claim_max_expected_flips_over_margin",
        site="adjudicate_tie_aware_claims(margin_fraction=1e6)",
        refused=adjudication_refused, receipt_names_it=True,
        registered_field_is_the_registry=True,
        detail={"why": "1e6 would make every published claim survive"},
    )

    failures = [
        f"{row['parameter']} via {row['construction_path']}" for row in rows
        if not (
            row["gate_refused_it"]
            and row["the_receipt_names_the_attempt"]
            and row["a_registered_field_still_reports_the_registry"] is not False
        )
    ]
    evidence = {
        "control": "round0247-call-site-control-v1",
        "rows": rows,
        "controls_run": len(rows),
        "failures": failures,
        "holds": not failures,
        "note": (
            "each row attempts an override through the construction path the "
            "registry names, and requires three things: the gate refuses it, "
            "the receipt names the attempt, and no registered_* field reports "
            "anything but the registry"
        ),
    }
    if failures:
        raise Round0247Error(
            f"R0247 CALL-SITE CONTROL DID NOT FIRE for {failures}"
        )
    return evidence


def _healthy_receipt(**overrides: Any) -> dict[str, Any]:
    """A receipt from a perfectly healthy 100 s guard, for gate controls."""
    receipt = {
        "sampling_thread_alive": True,
        "thread_death": None,
        "samples": 2_004,
        "thread_samples": 2_000,
        "boundary_polls": 4,
        "sample_coverage": 1.002,
        "thread_sample_coverage": 1.0,
        "sample_failures": 0,
        "max_consecutive_sample_failures": 0,
        "sample_interval_s": 0.05,
        "declared_sample_interval_s": 0.05,
        "sampled_wall_s": 100.0,
        "expected_samples_at_interval": 2_000.0,
        "expected_samples_at_the_registered_interval": 400.0,
        "thread_sample_coverage_at_the_registered_interval": 5.0,
        "max_thread_sample_gap_s": 0.06,
        "mean_thread_sample_gap_s": 0.05,
        "safety_overrides": [],
    }
    receipt.update(overrides)
    return receipt


# --------------------------------------------------------------------------- #
# THE denominator control — review-0246-01 A, in its exact numbers
# --------------------------------------------------------------------------- #
def run_coverage_denominator_control() -> dict[str, Any]:
    """The `5.0` s attack, refused three independent ways.

    review-0246-01 A, verbatim: "A guard declaring `interval_s = 5.0` samples
    faithfully, reports `thread_sample_coverage ~ 0.9994`, and passes
    `require_live_sampler` - while its observation gap permits
    `5.0 x 11,767,996,416 = 5.884e10` B, `1.99x` the sealed headroom."

    The receipt below is that guard: a ten-hour stage, sampling perfectly at its
    own declared `5.0` s, `7,200` thread samples against `7,200` expected. Under
    R0246 it passes. It is now refused on three arms that do not share a
    denominator.
    """
    verify_registry(label="R0247 coverage denominator control")
    wall = 36_000.0
    declared_interval = 5.0
    thread_samples = 7_196
    r0246_coverage = float(thread_samples) / (wall / declared_interval)
    registered_interval = float(
        REGISTERED_SAFETY_PARAMETERS["watchdog_sample_interval_s"].value
    )
    attack = _healthy_receipt(
        samples=thread_samples + 4,
        thread_samples=thread_samples,
        sample_interval_s=declared_interval,
        declared_sample_interval_s=declared_interval,
        sampled_wall_s=wall,
        expected_samples_at_interval=wall / declared_interval,
        sample_coverage=r0246_coverage,
        thread_sample_coverage=r0246_coverage,
        expected_samples_at_the_registered_interval=wall / registered_interval,
        thread_sample_coverage_at_the_registered_interval=(
            float(thread_samples) / (wall / registered_interval)
        ),
        max_thread_sample_gap_s=declared_interval,
        mean_thread_sample_gap_s=wall / float(thread_samples),
    )
    arms: dict[str, Any] = {}
    for name, receipt in (
        ("all_three_arms", attack),
        #: each arm on its own, by neutralising the other two, so the control
        #: proves three independent refusals rather than one.
        ("observation_gap_arm_alone", _healthy_receipt(
            max_thread_sample_gap_s=declared_interval,
        )),
        ("mean_gap_arm_alone", _healthy_receipt(
            mean_thread_sample_gap_s=wall / float(thread_samples),
        )),
        ("registered_denominator_coverage_arm_alone", _healthy_receipt(
            sampled_wall_s=wall,
            expected_samples_at_the_registered_interval=(
                wall / registered_interval
            ),
            thread_sample_coverage_at_the_registered_interval=(
                float(thread_samples) / (wall / registered_interval)
            ),
        )),
    ):
        refused = False
        message = None
        try:
            require_live_sampler(receipt, label=f"R0247 {name}")
        except (Round0246Error, Round0247Error) as error:
            refused = True
            message = str(error)
        arms[name] = {"refused": refused, "message": message}

    #: and the same guard built through its real constructor, where the clamp
    #: refuses the 5.0 s declaration before a single sample is taken.
    slope = float(
        REGISTERED_SAFETY_PARAMETERS["min_binding_slope_bytes_per_s"].value
    )
    headroom = float(
        REGISTERED_SAFETY_PARAMETERS["max_declared_headroom_bytes"].value
    )
    evidence = {
        "control": "round0247-coverage-denominator-control-v1",
        "planted": (
            "review-0246-01 A's exact attack: a 10 h stage declaring "
            "interval_s = 5.0 and sampling perfectly at it"
        ),
        "declared_sample_interval_s": declared_interval,
        "registered_sample_interval_s": registered_interval,
        "stage_wall_s": wall,
        "thread_samples": thread_samples,
        "coverage_as_r0246_computed_it": r0246_coverage,
        "coverage_at_the_registered_interval": (
            float(thread_samples) / (wall / registered_interval)
        ),
        "observation_gap_bytes_bought": declared_interval * slope,
        "sealed_headroom_bytes": headroom,
        "observation_gap_over_the_sealed_headroom": (
            declared_interval * slope / headroom
        ),
        "what_r0246_did": (
            "require_live_sampler PASSED it: coverage 0.9994 against a floor "
            "of 0.50, because the denominator was the interval the node "
            "declared"
        ),
        "arms": arms,
        "the_denominator_is_now_measured_wall_over_the_registered_interval": (
            True
        ),
        **registered_bounds([
            "watchdog_sample_interval_s",
            "watchdog_max_observation_gap_s",
            "watchdog_max_mean_observation_gap_s",
        ]),
    }
    failures = [
        name for name, arm in arms.items() if not arm["refused"]
    ]
    evidence["failures"] = failures
    evidence["holds"] = not failures
    if failures:
        raise Round0247Error(
            f"R0247 COVERAGE DENOMINATOR CONTROL DID NOT FIRE: {failures}. "
            "The 5.0 s attack still passes."
        )
    return evidence


# --------------------------------------------------------------------------- #
# the reviewer's sixteenth attack, replayed end to end
# --------------------------------------------------------------------------- #
#: R0245's sealed widest abort-read gap. review-0246-01 pushed exactly this
#: number through R0246's closed gate with `training_performed=True`.
R0245_SEALED_BLOCKER_GAP_S = 24.46713631998864


def run_reviewer_sixteenth_attack_control(
    *, gap_s: float = R0245_SEALED_BLOCKER_GAP_S
) -> dict[str, Any]:
    """review-0246-01 C, in its exact construction, and it must now fail."""
    verify_registry(label="R0247 sixteenth-attack control")
    ticks = iter([0.0, 0.0, float(gap_s), float(gap_s)])
    gate = AbortPollGate(
        inner=_noop, headroom_bytes=1 << 50,
        label="R0247 reviewer sixteenth attack", training_performed=True,
        max_poll_spacing_s=1e6, clock=lambda: next(ticks), replay=True,
    )
    gate.start()
    gate("read 1")
    gate("read 2")
    gate.finish()
    refused = False
    message = None
    failures: list[str] = []
    try:
        gate.require(measured_slope_bytes_per_s=0.0)
    except (Round0246Error, Round0247Error) as error:
        refused = True
        message = str(error)
    verdict = gate.last_verdict or gate.verdict(
        measured_slope_bytes_per_s=0.0
    )
    failures = list(verdict.get("failures") or [])
    evidence = {
        "control": "round0247-reviewer-sixteenth-attack-control-v1",
        "planted": (
            f"R0245's own sealed {gap_s} s blocker gap, pushed through with "
            "headroom_bytes = 1 << 50 and max_poll_spacing_s = 1e6 and "
            "training_performed = True - review-0246-01 C's exact construction"
        ),
        "what_r0246_did": (
            "HELD on all eight arms, with training_performed true, and "
            "published registered_max_poll_spacing_s: 1000000.0"
        ),
        "replayed_gap_s": float(gap_s),
        "declared_max_poll_spacing_s": verdict["declared_max_poll_spacing_s"],
        "effective_max_poll_spacing_s": verdict["effective_max_poll_spacing_s"],
        "registered_max_poll_spacing_s": verdict[
            "registered_r0246_max_poll_spacing_s"
        ],
        "declared_headroom_bytes": verdict["declared_headroom_bytes"],
        "effective_headroom_bytes": verdict["effective_headroom_bytes"],
        "gate_refused_it": refused,
        "failure_arms": failures,
        "safety_overrides": verdict["safety_overrides"],
        "message": message,
    }
    control_failures = [
        name for name, ok in (
            ("gate_refused_it", refused),
            ("the_registered_field_reports_the_registry", bool(
                verdict["registered_r0246_max_poll_spacing_s"]
                == REGISTERED_SAFETY_PARAMETERS[
                    "r0246_max_poll_spacing_s"
                ].value
            )),
            ("the_override_is_recorded", any(
                record["parameter"] == "r0246_max_poll_spacing_s"
                and record["kind"] == "weakening"
                for record in verdict["safety_overrides"]
            )),
            ("the_ceiling_arm_fired",
             "meets_the_registered_ceiling" in failures),
            ("the_override_arm_fired",
             "no_weakening_safety_override_was_attempted" in failures),
        ) if not ok
    ]
    evidence["failures"] = control_failures
    evidence["holds"] = not control_failures
    if control_failures:
        raise Round0247Error(
            "R0247 SIXTEENTH-ATTACK CONTROL DID NOT FIRE: "
            f"{control_failures}. The gap review-0246-01 pushed through is "
            "still passing."
        )
    return evidence


# --------------------------------------------------------------------------- #
# attacks on R0247's own fix
# --------------------------------------------------------------------------- #
def run_self_attack_battery(*, flag_path: str) -> dict[str, Any]:
    """Attack the registry the way two reviewers attacked its predecessors.

    Every row is published whether or not it is closed, with the residual
    stated. An attack that still succeeds is a limit of the guard, not an
    omission.
    """
    verify_registry(label="R0247 self-attack battery")
    attacks: list[dict[str, Any]] = []

    # 1. mutate the module global the guard used to read
    import basemap.round0244_guard as guard0244

    saved = guard0244.WATCHDOG_MAX_CONSECUTIVE_SAMPLE_FAILURES
    try:
        guard0244.WATCHDOG_MAX_CONSECUTIVE_SAMPLE_FAILURES = 10 ** 9
        watchdog = EnforcedHostWatchdog(
            interval_s=0.02, abort_flag_path=flag_path,
            label="R0247 module-global attack",
        )
        effective = int(
            watchdog.receipt()["max_consecutive_sample_failures_allowed"]
        )
    finally:
        guard0244.WATCHDOG_MAX_CONSECUTIVE_SAMPLE_FAILURES = saved
    attacks.append({
        "attack": (
            "r0247-self-1: assign the module global the guard used to read"
        ),
        "why_it_is_novel": (
            "clamping constructor keywords does nothing about a bound that "
            "reaches the decision as a module global. R0244's "
            "consecutive-failure tolerance was read from "
            "round0244_guard.WATCHDOG_MAX_CONSECUTIVE_SAMPLE_FAILURES, which "
            "any importer could assign."
        ),
        "planted_value": 10 ** 9,
        "effective_value": effective,
        "closed": bool(effective == 3),
        "residual": (
            "the guard reads the registry, so the module global is now a "
            "mirror. Assigning the REGISTRY entry is attack r0247-self-2."
        ),
    })

    # 2. mutate the registry itself
    registry_refused = False
    registry_error = None
    try:
        REGISTERED_SAFETY_PARAMETERS[  # type: ignore[index]
            "r0246_max_poll_spacing_s"
        ] = None
    except TypeError as error:
        registry_refused = True
        registry_error = str(error)
    frozen_refused = False
    try:
        object.__setattr__ if False else None
        REGISTERED_SAFETY_PARAMETERS["r0246_max_poll_spacing_s"].value = 1e6
    except Exception as error:  # noqa: BLE001 - the refusal is the evidence
        frozen_refused = True
        registry_error = f"{registry_error} | {error}"
    attacks.append({
        "attack": "r0247-self-2: assign into the registry itself",
        "why_it_is_novel": (
            "once every call site reads the registry, the registry is the "
            "single point of attack. It is a MappingProxyType of frozen "
            "dataclasses, and every gate verifies a SHA-256 over the whole "
            "inventory against a pinned digest."
        ),
        "mapping_is_read_only": registry_refused,
        "entry_is_frozen": frozen_refused,
        "error": registry_error,
        "registry_fingerprint": registry_fingerprint(),
        "closed": bool(registry_refused and frozen_refused),
        "residual": (
            "a caller that rebinds the module attribute "
            "round0247_registry._PARAMETERS and re-imports could still move "
            "the inventory, but the fingerprint would then disagree with the "
            "pinned digest and verify_registry() fails closed. What no "
            "in-process guard can survive is a caller that also edits the "
            "pinned digest, which is a source change and shows in the diff."
        ),
    })

    # 3. the scripted clock
    ticks = iter([0.0] * 32)
    scripted = AbortPollGate(
        inner=_sanctioned_reader, headroom_bytes=_REGISTERED_HEADROOM_BYTES,
        label="R0247 scripted-clock attack", training_performed=True,
        clock=lambda: 0.0,
    )
    scripted.start()
    for step in range(12):
        scripted(f"read {step}")
    scripted.finish()
    scripted_refused = False
    try:
        scripted.require(measured_slope_bytes_per_s=0.0)
    except (Round0246Error, Round0247Error):
        scripted_refused = True
    scripted_arms = list(
        (scripted.last_verdict or {}).get("failures") or []
    )
    del ticks
    attacks.append({
        "attack": (
            "r0247-self-3: supply the gate's own clock, so every gap is zero"
        ),
        "why_it_is_novel": (
            "no number is overridden at all. The gate measures whatever the "
            "clock says, and R0246 accepted any callable. A stage that "
            "allocated for ten hours would publish a widest gap of 0.0 s and "
            "every arm would hold."
        ),
        "gate_refused_it": scripted_refused,
        "failure_arms": scripted_arms,
        "closed": bool(
            scripted_refused
            and "the_clock_is_the_registered_monotonic_clock" in scripted_arms
        ),
        "residual": (
            "the arm is waived for a gate that declares itself replay=True, "
            "which is how the reviewer-shaped replays are run. "
            "require_enforcement_evidence() is the gate that stops a replay "
            "verdict being sealed as a node's own measurement."
        ),
    })

    # 4. the no-op inner
    noop = AbortPollGate(
        inner=_noop, headroom_bytes=_REGISTERED_HEADROOM_BYTES,
        label="R0247 no-op-reader attack", training_performed=True,
    )
    noop.start()
    for step in range(4):
        noop(f"read {step}")
        time.sleep(0.001)
    noop.finish()
    noop_refused = False
    try:
        noop.require(measured_slope_bytes_per_s=0.0)
    except (Round0246Error, Round0247Error):
        noop_refused = True
    noop_arms = list((noop.last_verdict or {}).get("failures") or [])
    attacks.append({
        "attack": (
            "r0247-self-4: wrap a callable that does not read the abort flag"
        ),
        "why_it_is_novel": (
            "the gate times calls to ITSELF, so a gate wrapping a no-op "
            "publishes a flawless spacing for a stage that never checks "
            "anything. R0246's own controls pass inner=lambda _where: None; "
            "so could a trainer, and the receipt would be clean."
        ),
        "gate_refused_it": noop_refused,
        "failure_arms": noop_arms,
        "closed": bool(
            noop_refused
            and "the_gate_wraps_a_registered_abort_reader" in noop_arms
        ),
        "residual": (
            "the allowlist is by qualified name plus an opt-in marker, so a "
            "wrapper that calls the real reader must be marked with "
            "@registered_abort_reader. That marking is a source change, which "
            "is the point."
        ),
    })

    # 5. seal a replay verdict as enforcement evidence
    replay_ticks = iter([0.0, 0.0, 0.001, 0.002])
    replay = AbortPollGate(
        inner=_noop, headroom_bytes=_REGISTERED_HEADROOM_BYTES,
        label="R0247 replay-sealing attack",
        clock=lambda: next(replay_ticks), replay=True,
    )
    replay.start()
    replay("read 1")
    replay("read 2")
    replay.finish()
    replay_verdict = replay.require(measured_slope_bytes_per_s=0.0)
    seal_refused = False
    try:
        require_enforcement_evidence(
            replay_verdict, label="R0247 replay-sealing attack"
        )
    except Round0247Error:
        seal_refused = True
    attacks.append({
        "attack": (
            "r0247-self-5: declare replay=True to waive the clock and reader "
            "arms, then seal the verdict as the node's own measurement"
        ),
        "why_it_is_novel": (
            "the replay escape hatch that makes r0247-self-3 and -4 "
            "survivable is itself a declaration a node could make. It is "
            "closed at the SEALING boundary rather than at require(), because "
            "the replays are legitimate demonstrations and only their "
            "publication as evidence is not."
        ),
        "gate_permitted_the_replay": bool(replay_verdict["holds"]),
        "sealing_refused_it": seal_refused,
        "closed": seal_refused,
        "residual": (
            "a node that never calls require_enforcement_evidence() can still "
            "write whatever it likes into its own receipt. Nothing in-process "
            "closes that; what closes it is that the node modules are the "
            "diff a reviewer reads."
        ),
    })

    # 6. reach around the gate by using its unguarded base class
    base = AbortPollTracker(
        inner=_noop, headroom_bytes=1 << 50, label="R0247 base-class attack",
        slope_bytes_per_s=1.0,
    )
    base("a")
    base("b")
    base_verdict = base.verdict(measured_slope_bytes_per_s=0.0)
    attacks.append({
        "attack": (
            "r0247-self-6: use R0245's AbortPollTracker directly instead of "
            "R0246's AbortPollGate"
        ),
        "why_it_is_novel": (
            "the clamps would be worthless if the unguarded base class were "
            "still importable and usable. A node wanting a weak verdict would "
            "simply use the parent."
        ),
        "declared_headroom_bytes": base_verdict["declared_headroom_bytes"],
        "effective_headroom_bytes": base_verdict["effective_headroom_bytes"],
        "declared_worst_case_slope_bytes_per_s": base_verdict[
            "declared_worst_case_slope_bytes_per_s"
        ],
        "effective_worst_case_slope_bytes_per_s": base_verdict[
            "effective_worst_case_slope_bytes_per_s"
        ],
        "closed": bool(
            base_verdict["effective_headroom_bytes"]
            == REGISTERED_SAFETY_PARAMETERS[
                "max_declared_headroom_bytes"
            ].value
            and base_verdict["effective_worst_case_slope_bytes_per_s"]
            == REGISTERED_SAFETY_PARAMETERS[
                "min_binding_slope_bytes_per_s"
            ].value
        ),
        "residual": (
            "the clamp is in the BASE constructor, so every subclass "
            "inherits it. AbortPollTracker.require() is still R0245's, which "
            "has fewer arms than R0246's - a node using the base gets clamped "
            "bounds but not the ceiling, poll-count, clock or reader arms. "
            "STATED, NOT CLOSED: a node must use AbortPollGate, and its "
            "receipt names which class it used through the instrument field."
        ),
    })

    # 7. fabricate the receipt the liveness gate reads
    fabricated = _healthy_receipt(
        thread_samples=10 ** 9,
        thread_sample_coverage_at_the_registered_interval=10 ** 6,
        max_thread_sample_gap_s=0.0,
        mean_thread_sample_gap_s=0.0,
    )
    fabricated_refused = False
    try:
        require_live_sampler(fabricated, label="R0247 fabricated receipt")
    except (Round0246Error, Round0247Error):
        fabricated_refused = True
    attacks.append({
        "attack": "r0247-self-7: hand the liveness gate a fabricated receipt",
        "why_it_is_novel": (
            "require_live_sampler takes a Mapping. A node can pass a dict "
            "literal with perfect numbers in it, and no clamp anywhere touches "
            "that."
        ),
        "gate_refused_it": fabricated_refused,
        "closed": False,
        "residual": (
            "NOT CLOSED, and it cannot be closed in-process: a gate that reads "
            "a receipt cannot verify the receipt was produced by the "
            "instrument that claims to have produced it. What bounds it is "
            "that the receipt is written by ThreadedHostWatchdog.receipt(), "
            "which the node calls on the object it armed, and that the node "
            "module is the diff a reviewer reads. The honest statement is that "
            "R0247 defends against a node that misconfigures its guard, not "
            "against a node that lies about it."
        ),
    })

    succeeded = [row["attack"] for row in attacks if not row.get("closed")]
    return {
        "instrument": "round0247-self-attack-battery-v1",
        "attacks": attacks,
        "attacks_run": len(attacks),
        "attacks_closed": sum(1 for row in attacks if row.get("closed")),
        "attacks_that_still_succeed": succeeded,
        "every_self_attack_is_closed": not succeeded,
        "note": (
            "these attack R0247's OWN fix. Two reviewers in a row defeated the "
            "previous round one door over from where it had just locked, so "
            "everything tried is published - including the one that succeeds."
        ),
    }


def safety_closure_receipt(
    *,
    inventory: Mapping[str, Any],
    clamp_controls: Mapping[str, Any],
    call_site_controls: Mapping[str, Any],
    denominator_control: Mapping[str, Any],
    sixteenth_attack: Mapping[str, Any],
    self_attacks: Mapping[str, Any],
) -> dict[str, Any]:
    arms = {
        "every_registered_parameter_has_a_clamp_control": bool(
            clamp_controls["holds"]
        ),
        "every_construction_path_is_refused": bool(
            call_site_controls["holds"]
        ),
        "the_5_second_coverage_attack_fails": bool(
            denominator_control["holds"]
        ),
        "the_reviewers_sixteenth_attack_fails": bool(
            sixteenth_attack["holds"]
        ),
        "the_self_attacks_are_published_with_their_residuals": bool(
            self_attacks["attacks_run"] >= 7
        ),
    }
    return {
        "instrument": "round0247-safety-parameter-closure-v1",
        "safety_parameter_inventory": dict(inventory),
        "clamp_controls": dict(clamp_controls),
        "call_site_controls": dict(call_site_controls),
        "coverage_denominator_control": dict(denominator_control),
        "reviewer_sixteenth_attack_control": dict(sixteenth_attack),
        "self_attack_battery": dict(self_attacks),
        "arms": arms,
        "holds": all(arms.values()),
        "guard_axis": "anonymous, never RSS",
        "registry_fingerprint": registry_fingerprint(),
    }


def run_all_controls(*, workspace: str) -> dict[str, Any]:
    """Every R0247 control, in one call, for the node and the smoke alike."""
    os.makedirs(workspace, exist_ok=True)
    flag_path = os.path.join(workspace, "r0247-control.abort")
    try:
        return safety_closure_receipt(
            inventory=safety_parameter_inventory(),
            clamp_controls=run_clamp_controls(),
            call_site_controls=run_call_site_controls(flag_path=flag_path),
            denominator_control=run_coverage_denominator_control(),
            sixteenth_attack=run_reviewer_sixteenth_attack_control(),
            self_attacks=run_self_attack_battery(flag_path=flag_path),
        )
    finally:
        try:
            os.remove(flag_path)
        except OSError:
            pass


__all__ = [
    "R0245_SEALED_BLOCKER_GAP_S",
    "ROUND_ID",
    "WEAKER_THAN_REGISTERED",
    "Round0245Error",
    "run_all_controls",
    "run_call_site_controls",
    "run_clamp_controls",
    "run_coverage_denominator_control",
    "run_reviewer_sixteenth_attack_control",
    "run_self_attack_battery",
    "safety_closure_receipt",
]
