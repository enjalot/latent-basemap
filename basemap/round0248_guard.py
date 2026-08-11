"""R0248 — the four gaps review-0247-01 found, each with its positive control.

review-0247-01 defeated R0247 five ways. Two of the five split into different
kinds and R0248 treats them differently.

**Four genuine gaps — bounds that never entered the decision, fixed here.**

1. `watchdog_max_observation_gap_s` and `watchdog_max_mean_observation_gap_s`
   were the substance of R0247 §B and were compared against **module globals**
   at `round0246_guard:278,289`. Two assignments and review-0246-01 A's `5.0` s
   attack passed, while the receipt still printed `2.5109531834854018`.
2. `sampler_max_anonymous_bytes` was registered as parameter 14 with the
   override path *"a module global compared in the node's verdict arms"* — and
   that is exactly how it stayed enforced.
3. `registered_abort_reader` was an exported one-call decorator that made any
   callable pass both `require()` and `require_enforcement_evidence()`.
4. The inventory missed `replay`, a self-declared bool that waives two arms of
   `require()` — the shape R0247 retired `training_performed` for.

Each has a control below **in the reviewer's shape**: make the assignment, or
the call, and prove the decision does not move.

**Three defeats that require the node to cheat — not chased here.** Rebinding
`REGISTERED_SAFETY_PARAMETERS`, mutating the dict behind the `MappingProxyType`
through `gc.get_referents()`, and running the watchdog clock `1000x` slow are
the same category as the fabricated receipt R0247 published as unclosable. The
fingerprint bug (it hashed `_PARAMETERS` while the decisions read
`REGISTERED_SAFETY_PARAMETERS`) is fixed because hashing the wrong object is a
bug, not a defence; the `gc` and clock attacks are re-run below and **still
succeed**, and they are reported as succeeding. The bound that does not live in
this process is `basemap.round0248_external`.

The in-process guards are all kept. They catch **mistakes** — a bound
registered and then compared against its mirror, a guard armed with the wrong
interval — which is the real risk from a runner following instructions in good
faith. The external cgroup limit catches the runaway allocation that has taken
this machine down twice. Neither defends against a runner that fabricates; only
independent recomputation by a reviewer who did not run the round does.

Nothing here signals anything, starts a child process, touches the GPU, or
imports a GPU array library.
"""
from __future__ import annotations

import gc
import os
from collections.abc import Mapping
from typing import Any

import basemap.round0244_prereq as prereq0244
import basemap.round0246_guard as guard0246
from basemap.round0246_guard import (
    AbortPollGate,
    Round0246Error,
    require_enforcement_evidence,
    require_live_sampler,
)
from basemap.round0247_guard import _healthy_receipt
from basemap.round0247_registry import (
    REGISTERED_SAFETY_PARAMETERS,
    Round0247Error,
    abort_reader_sanction,
    is_registered_abort_reader,
    marker_applications,
    registered_abort_reader,
    registered_bounds,
    registered_value,
    registry_fingerprint,
    unsanctioned_marker_applications,
    verify_registry,
)
from basemap.round0248_external import EXTERNAL_BOUND_NOTE
from basemap.round0248_inventory import (
    derive_inventory,
    require_inventory_complete,
)

ROUND_ID = "0248"

GUARD_SCOPE_NOTE = (
    "R0248 splits the five defeats of review-0247-01 into two kinds. FOUR are "
    "bugs: a registered bound compared against its module-level mirror (two "
    "observation-gap bounds and the sampler's anonymous ceiling), a runtime "
    "allowlist extension, and a missing inventory entry. They are fixed here "
    "and each carries a control in the reviewer's shape. THREE require the "
    "node to deliberately cheat - rebinding the registry mapping, mutating the "
    "dict behind the MappingProxyType, and scripting the module clock - and no "
    "in-process Python guard can constrain the process it runs in. Those are "
    "re-run and published with their outcomes, including the ones that still "
    "succeed. " + EXTERNAL_BOUND_NOTE
)


def _sanctioned_reader(_where: str) -> None:
    """A registered cooperative-abort reader, sanctioned BY NAME in source."""
    return None


def _noop(_where: str) -> None:
    return None


def _the_5_second_receipt() -> dict[str, Any]:
    """review-0246-01 A's exact attack receipt: 10 h at a declared 5.0 s."""
    wall = 36_000.0
    declared = 5.0
    thread_samples = 7_196
    registered_interval = registered_value("watchdog_sample_interval_s")
    return _healthy_receipt(
        samples=thread_samples + 4,
        thread_samples=thread_samples,
        sample_interval_s=declared,
        declared_sample_interval_s=declared,
        sampled_wall_s=wall,
        expected_samples_at_interval=wall / declared,
        sample_coverage=float(thread_samples) / (wall / declared),
        thread_sample_coverage=float(thread_samples) / (wall / declared),
        expected_samples_at_the_registered_interval=wall / registered_interval,
        thread_sample_coverage_at_the_registered_interval=(
            float(thread_samples) / (wall / registered_interval)
        ),
        max_thread_sample_gap_s=declared,
        mean_thread_sample_gap_s=wall / float(thread_samples),
    )


# --------------------------------------------------------------------------- #
# gap 1 — the two observation-gap bounds were module globals
# --------------------------------------------------------------------------- #
def run_gap1_observation_gap_control() -> dict[str, Any]:
    """review-0247-01 A.3, verbatim: two assignments and the `5.0` s passes."""
    verify_registry(label="R0248 gap-1 control")
    receipt = _the_5_second_receipt()

    def _try(label: str) -> dict[str, Any]:
        try:
            state = require_live_sampler(receipt, label=label)
        except (Round0246Error, Round0247Error) as error:
            return {"refused": True, "message": str(error)[:400]}
        return {"refused": False, "state_holds": bool(state.get("holds"))}

    unpatched = _try("R0248 gap-1 control, unpatched")

    saved_max = guard0246.WATCHDOG_MAX_OBSERVATION_GAP_S
    saved_mean = guard0246.WATCHDOG_MAX_MEAN_OBSERVATION_GAP_S
    try:
        #: THE REVIEWER'S TWO ASSIGNMENTS.
        guard0246.WATCHDOG_MAX_OBSERVATION_GAP_S = 1.0e6
        guard0246.WATCHDOG_MAX_MEAN_OBSERVATION_GAP_S = 1.0e6
        patched = _try("R0248 gap-1 control, module globals reassigned")
        try:
            reported = require_live_sampler(
                _healthy_receipt(), label="R0248 gap-1 healthy under patch"
            )
        except (Round0246Error, Round0247Error) as error:
            reported = {"refused": str(error)[:200]}
    finally:
        guard0246.WATCHDOG_MAX_OBSERVATION_GAP_S = saved_max
        guard0246.WATCHDOG_MAX_MEAN_OBSERVATION_GAP_S = saved_mean
    verify_registry(label="R0248 gap-1 control, after restore")

    arms = {
        "the_attack_is_refused_unpatched": bool(unpatched["refused"]),
        "the_attack_is_still_refused_after_both_assignments": bool(
            patched["refused"]
        ),
        "the_comparison_site_reports_the_registry": bool(
            isinstance(reported, dict)
            and reported.get(
                "registered_max_observation_gap_s_at_the_comparison_site"
            ) == registered_value("watchdog_max_observation_gap_s")
        ),
    }
    evidence = {
        "control": "round0248-gap1-observation-gap-control-v1",
        "gap": (
            "review-0247-01 A.3: round0246_guard:278 and :289 compared against "
            "the imported module globals WATCHDOG_MAX_OBSERVATION_GAP_S and "
            "WATCHDOG_MAX_MEAN_OBSERVATION_GAP_S. Two assignments and "
            "review-0246-01 A's 5.0 s receipt passed, while "
            "registered_watchdog_max_observation_gap_s in the same receipt "
            "still read 2.5109531834854018"
        ),
        "fix": (
            "both comparisons call registered_value() and resolve the registry "
            "at the moment of the comparison"
        ),
        "planted": "guard0246.WATCHDOG_MAX_OBSERVATION_GAP_S = 1e6, and the mean",
        "unpatched": unpatched,
        "after_both_assignments": patched,
        "comparison_site_fields": {
            key: value for key, value in (
                reported.items() if isinstance(reported, dict) else ()
            ) if "at_the_comparison_site" in key
        },
        **registered_bounds([
            "watchdog_max_observation_gap_s",
            "watchdog_max_mean_observation_gap_s",
        ]),
        "arms": arms,
    }
    evidence["failures"] = [name for name, ok in arms.items() if not ok]
    evidence["holds"] = not evidence["failures"]
    if evidence["failures"]:
        raise Round0247Error(
            f"R0248 GAP-1 CONTROL DID NOT FIRE: {evidence['failures']}"
        )
    return evidence


# --------------------------------------------------------------------------- #
# gap 2 — the sampler's anonymous ceiling was a bare literal
# --------------------------------------------------------------------------- #
def run_gap2_sampler_bytes_control() -> dict[str, Any]:
    """`round0244_prereq:112` was a bare literal and the nodes compared it."""
    verify_registry(label="R0248 gap-2 control")
    registered = registered_value("sampler_max_anonymous_bytes")
    #: A peak one byte over the registered ceiling: the arm must be False.
    peak = int(registered) + 1
    unpatched_arm = bool(peak <= prereq0244.sampler_max_anonymous_bytes())

    saved = prereq0244.SAMPLER_MAX_ANONYMOUS_BYTES
    try:
        prereq0244.SAMPLER_MAX_ANONYMOUS_BYTES = float(1 << 50)
        patched_accessor = prereq0244.sampler_max_anonymous_bytes()
        patched_arm = bool(peak <= prereq0244.sampler_max_anonymous_bytes())
        the_mirror_moved = bool(
            prereq0244.SAMPLER_MAX_ANONYMOUS_BYTES == float(1 << 50)
        )
    finally:
        prereq0244.SAMPLER_MAX_ANONYMOUS_BYTES = saved
    verify_registry(label="R0248 gap-2 control, after restore")

    arms = {
        "an_over_budget_peak_fails_the_arm_unpatched": not unpatched_arm,
        "an_over_budget_peak_still_fails_after_the_assignment": not patched_arm,
        "the_accessor_still_returns_the_registered_value": bool(
            patched_accessor == registered
        ),
        "the_module_global_did_move_so_the_control_is_not_vacuous": bool(
            the_mirror_moved
        ),
    }
    evidence = {
        "control": "round0248-gap2-sampler-bytes-control-v1",
        "gap": (
            "review-0247-01 A.3: sampler_max_anonymous_bytes is registered as "
            "parameter 14 with the override path 'a module global compared in "
            "the node's verdict arms', and round0244_prereq:112 was a bare "
            "literal that round0244/0245/0246_nodes.py compared against. "
            "Registering it changed nothing"
        ),
        "fix": (
            "the literal is a mirror of the registry and all three verdict "
            "arms call sampler_max_anonymous_bytes(), which reads the registry "
            "at the comparison site"
        ),
        "planted": "round0244_prereq.SAMPLER_MAX_ANONYMOUS_BYTES = 1 << 50",
        "registered_ceiling_bytes": registered,
        "peak_used_in_the_arm_bytes": peak,
        "accessor_under_the_assignment": patched_accessor,
        **registered_bounds(["sampler_max_anonymous_bytes"]),
        "arms": arms,
    }
    evidence["failures"] = [name for name, ok in arms.items() if not ok]
    evidence["holds"] = not evidence["failures"]
    if evidence["failures"]:
        raise Round0247Error(
            f"R0248 GAP-2 CONTROL DID NOT FIRE: {evidence['failures']}"
        )
    return evidence


# --------------------------------------------------------------------------- #
# gap 3 — the runtime-extensible abort-reader allowlist
# --------------------------------------------------------------------------- #
def _gate_arms(gate: AbortPollGate) -> tuple[bool, list[str], dict[str, Any]]:
    gate.start()
    gate("read 1")
    gate("read 2")
    gate.finish()
    refused = False
    try:
        gate.require(measured_slope_bytes_per_s=0.0)
    except (Round0246Error, Round0247Error):
        refused = True
    verdict = gate.last_verdict or gate.verdict(measured_slope_bytes_per_s=0.0)
    return refused, list(verdict.get("failures") or []), verdict


def run_gap3_abort_reader_control() -> dict[str, Any]:
    """review-0247-01 A.4: one call made a no-op pass both gates."""
    verify_registry(label="R0248 gap-3 control")

    def marked_noop(_where: str) -> None:
        return None

    before = is_registered_abort_reader(marked_noop)
    #: THE REVIEWER'S ONE CALL.
    registered_abort_reader(marked_noop)
    after = is_registered_abort_reader(marked_noop)
    #: and the same thing without the exported function, by raw setattr.
    def setattr_noop(_where: str) -> None:
        return None

    setattr(setattr_noop, "_r0247_registered_abort_reader", True)
    after_setattr = is_registered_abort_reader(setattr_noop)

    headroom = int(registered_value("max_declared_headroom_bytes"))
    refused, arms_fired, verdict = _gate_arms(AbortPollGate(
        inner=marked_noop, headroom_bytes=headroom,
        label="R0248 gap-3 marked-no-op gate", training_performed=True,
    ))
    seal_refused = False
    seal_failures: list[str] = []
    try:
        require_enforcement_evidence(verdict, label="R0248 gap-3 sealing")
    except Round0247Error as error:
        seal_refused = True
        seal_failures = [
            name for name in (
                "inner_is_a_registered_abort_reader",
                "the_abort_reader_is_sanctioned_by_registered_name",
            ) if name in str(error)
        ]

    #: the sanctioned reader must still pass, or the fix is a wrecking ball.
    sanctioned_ok = is_registered_abort_reader(_sanctioned_reader)
    unsanctioned = [
        row for row in unsanctioned_marker_applications()
        if row["qualified_name"].endswith("marked_noop")
    ]

    control_arms = {
        "the_marker_did_not_sanction_before": not before,
        "the_marker_does_not_sanction_after_the_call": not after,
        "raw_setattr_does_not_sanction_either": not after_setattr,
        "the_gate_refuses_the_marked_no_op": bool(refused),
        "the_reader_arm_fired": bool(
            "the_gate_wraps_a_registered_abort_reader" in arms_fired
        ),
        "sealing_refuses_it": bool(seal_refused),
        "sealing_names_the_sanction_mechanism": bool(seal_failures),
        "the_application_is_a_recorded_violation": bool(unsanctioned),
        "a_name_registered_reader_still_passes": bool(sanctioned_ok),
    }
    evidence = {
        "control": "round0248-gap3-abort-reader-control-v1",
        "gap": (
            "review-0247-01 A.4: registered_abort_reader is public, in "
            "__all__, and sets an attribute that is_registered_abort_reader "
            "honoured. One call made a no-op pass require() AND "
            "require_enforcement_evidence - the gate R0247 positioned as the "
            "backstop for exactly this attack"
        ),
        "fix": (
            "sanction is by qualified name against the source-level "
            "REGISTERED_ABORT_READERS set. The attribute sanctions nothing; "
            "applying it is recorded in MARKER_APPLICATIONS and published; and "
            "every verdict now carries abort_reader_sanction naming WHICH "
            "mechanism sanctioned the reader, which is review-0247-01 H5"
        ),
        "planted": "registered_abort_reader(marked_noop), then raw setattr",
        "is_registered_abort_reader_before": before,
        "is_registered_abort_reader_after_the_call": after,
        "is_registered_abort_reader_after_raw_setattr": after_setattr,
        "gate_failure_arms": arms_fired,
        "sealing_refused": seal_refused,
        "abort_reader_sanction_in_the_verdict": verdict.get(
            "abort_reader_sanction"
        ),
        "recorded_marker_applications": list(marker_applications()),
        "unsanctioned_marker_applications": list(
            unsanctioned_marker_applications()
        ),
        "arms": control_arms,
    }
    evidence["failures"] = [name for name, ok in control_arms.items() if not ok]
    evidence["holds"] = not evidence["failures"]
    if evidence["failures"]:
        raise Round0247Error(
            f"R0248 GAP-3 CONTROL DID NOT FIRE: {evidence['failures']}"
        )
    return evidence


# --------------------------------------------------------------------------- #
# gap 4 — `replay`, and the inventory that is now derived
# --------------------------------------------------------------------------- #
def run_gap4_replay_control(*, repo_root: str) -> dict[str, Any]:
    """review-0247-01 A.6: a self-declared bool waived two arms, unregistered."""
    verify_registry(label="R0248 gap-4 control")
    ticks = iter([0.0, 0.0, 0.001, 0.002])
    headroom = int(registered_value("max_declared_headroom_bytes"))
    gate = AbortPollGate(
        inner=_noop, headroom_bytes=headroom,
        label="R0248 gap-4 replay gate",
        clock=lambda: next(ticks), replay=True,
    )
    gate.start()
    gate("read 1")
    gate("read 2")
    gate.finish()
    verdict = gate.require(measured_slope_bytes_per_s=0.0)
    seal_refused = False
    seal_message = None
    try:
        require_enforcement_evidence(verdict, label="R0248 gap-4 sealing")
    except Round0247Error as error:
        seal_refused = True
        seal_message = str(error)[:400]

    inventory = derive_inventory(repo_root=repo_root)
    waivers = inventory["arm_waivers"]
    replay_rows = [
        row for row in waivers["rows"] if row["waived_by"] == "replay"
    ]
    arms = {
        "replay_is_registered": bool(
            "replay" in REGISTERED_SAFETY_PARAMETERS
        ),
        "the_registered_value_is_false": bool(
            registered_value("replay") == 0.0
        ),
        "the_declaration_is_recorded_in_the_verdict": bool(any(
            record.get("parameter") == "replay"
            for record in verdict["safety_overrides"]
        )),
        "the_verdict_names_the_arms_it_waived": bool(
            sorted(verdict["gate_arms_waived_by_declaration"]) == sorted([
                "the_clock_is_the_registered_monotonic_clock",
                "the_gate_wraps_a_registered_abort_reader",
            ])
        ),
        "sealing_refuses_a_waived_verdict": bool(seal_refused),
        "the_mechanical_derivation_finds_both_waivers": bool(
            len(replay_rows) == 2
        ),
        "every_derived_waiver_names_a_registered_parameter": bool(
            waivers["holds"]
        ),
    }
    evidence = {
        "control": "round0248-gap4-replay-control-v1",
        "gap": (
            "review-0247-01 A.6: replay=True waives BOTH "
            "the_clock_is_the_registered_monotonic_clock and "
            "the_gate_wraps_a_registered_abort_reader inside require() itself, "
            "and it was not one of the 19 - the same shape R0247 retired "
            "training_performed for"
        ),
        "fix": (
            "replay is registered (value False, enforcement 'clamped' so the "
            "legitimate reviewer-shaped replays still run); the declaration is "
            "recorded in safety_overrides; the verdict NAMES the arms it "
            "waived; require_enforcement_evidence refuses on "
            "no_gate_arm_was_waived_by_a_declaration; and the waiver set is "
            "DERIVED from the source rather than listed by hand"
        ),
        "declared_replay": verdict["declared_replay"],
        "registered_replay": verdict["registered_replay"],
        "gate_arms_waived_by_declaration": verdict[
            "gate_arms_waived_by_declaration"
        ],
        "replay_declaration_record": verdict["replay_declaration"],
        "sealing_refused": seal_refused,
        "sealing_message": seal_message,
        "mechanically_derived_waivers": replay_rows,
        "arms": arms,
    }
    evidence["failures"] = [name for name, ok in arms.items() if not ok]
    evidence["holds"] = not evidence["failures"]
    if evidence["failures"]:
        raise Round0247Error(
            f"R0248 GAP-4 CONTROL DID NOT FIRE: {evidence['failures']}"
        )
    return evidence


# --------------------------------------------------------------------------- #
# attacks on R0248's own work — including the ones that still succeed
# --------------------------------------------------------------------------- #
def run_self_attack_battery() -> dict[str, Any]:
    """Re-run review-0247-01's five defeats against R0248, and publish."""
    verify_registry(label="R0248 self-attack battery")
    attacks: list[dict[str, Any]] = []
    import basemap.round0247_registry as registry

    # A.1 — rebind REGISTERED_SAFETY_PARAMETERS
    from dataclasses import replace as _replace

    saved_mapping = registry.REGISTERED_SAFETY_PARAMETERS
    caught = None
    effective_after = None
    try:
        forged = dict(saved_mapping)
        forged["r0246_max_poll_spacing_s"] = _replace(
            forged["r0246_max_poll_spacing_s"], value=1.0e6
        )
        registry.REGISTERED_SAFETY_PARAMETERS = forged  # type: ignore[assignment]
        try:
            effective_after = registry.registered_value(
                "r0246_max_poll_spacing_s"
            )
        except Round0247Error as error:
            caught = str(error)[:240]
    finally:
        registry.REGISTERED_SAFETY_PARAMETERS = saved_mapping
    verify_registry(label="R0248 after A.1")
    attacks.append({
        "attack": (
            "r0248-self-1 (review-0247-01 A.1): rebind "
            "REGISTERED_SAFETY_PARAMETERS, the object the decisions read"
        ),
        "what_r0247_did": (
            "registry_fingerprint() hashed the module-level tuple _PARAMETERS "
            "while every guard resolved REGISTERED_SAFETY_PARAMETERS, so this "
            "moved all 19 clamps at once with the pinned digest unchanged and "
            "zero overrides recorded"
        ),
        "effective_ceiling_after_the_rebind": effective_after,
        "verify_registry_error": caught,
        "closed": bool(caught is not None and effective_after is None),
        "residual": (
            "the fingerprint now covers the mapping, so a rebind fails closed "
            "at the comparison site. This is a fix to the FINGERPRINT'S OWN "
            "CLAIM, not a defence: a caller that also edits the pinned digest "
            "wins, and that is a source change in the diff"
        ),
    })

    # A.2 — mutate the dict behind the MappingProxyType
    referents = [
        item for item in gc.get_referents(saved_mapping) if isinstance(item, dict)
    ]
    mutated = None
    fingerprint_after = None
    gate_after = None
    if referents:
        underlying = referents[0]
        saved_entry = underlying.get("r0246_max_poll_spacing_s")
        try:
            underlying["r0246_max_poll_spacing_s"] = _replace(
                saved_entry, value=1.0e6
            )
            mutated = float(
                registry.REGISTERED_SAFETY_PARAMETERS[
                    "r0246_max_poll_spacing_s"
                ].value
            )
            fingerprint_after = registry_fingerprint()
            try:
                gate_after = registered_value("r0246_max_poll_spacing_s")
            except Round0247Error:
                gate_after = "refused"
        finally:
            underlying["r0246_max_poll_spacing_s"] = saved_entry
    verify_registry(label="R0248 after A.2")
    attacks.append({
        "attack": (
            "r0248-self-2 (review-0247-01 A.2): gc.get_referents() hands out "
            "the dict behind the MappingProxyType; mutate it in place"
        ),
        "the_bound_moved_to": mutated,
        "the_fingerprint_after_the_mutation": fingerprint_after,
        "the_comparison_site_returned": gate_after,
        "closed": bool(gate_after == "refused"),
        "residual": (
            "STILL SUCCEEDS as a mutation, and the fingerprint now CATCHES it "
            "because the digest covers the mapping's contents - but a caller "
            "who mutates can also mutate the pinned digest, and nothing "
            "in-process survives that. This is the category R0248 stops "
            "chasing: no in-process Python guard can constrain the process it "
            "runs in. The bound that does not live in this process is the "
            "cgroup memory.max in basemap.round0248_external, and it covers "
            "MEMORY only"
        ),
    })

    # A.5 — the scripted module clock on the host watchdog
    import basemap.round0244_guard as guard0244

    saved_now = guard0244._now
    scripted_wall = None
    try:
        guard0244._now = lambda: saved_now() * 1e-3  # type: ignore[assignment]
        scripted_wall = float(guard0244._now())
    finally:
        guard0244._now = saved_now  # type: ignore[assignment]
    attacks.append({
        "attack": (
            "r0248-self-3 (review-0247-01 A.5): assign round0244_guard._now to "
            "a clock that runs 1000x slow, collapsing the measured wall below "
            "the registered interval so all three liveness arms switch "
            "themselves off by their own applicability predicates"
        ),
        "the_module_clock_is_assignable": True,
        "scripted_clock_reading": scripted_wall,
        "closed": False,
        "residual": (
            "NOT CLOSED and NOT CHASED. A module-level function is an "
            "attribute of a module object the node owns, exactly like the "
            "module-level constants R0248 just moved into the registry - and "
            "moving the clock into the registry would only move the assignment "
            "one level up, which is what R0245, R0246 and R0247 each did in "
            "turn. The honest statement is that a node which scripts its own "
            "clock is a node that lies, and the only mitigation is a reviewer "
            "who recomputes"
        ),
    })

    # the fabricated receipt, still
    fabricated = _healthy_receipt(
        thread_samples=10 ** 9,
        thread_sample_coverage_at_the_registered_interval=10 ** 6,
        max_thread_sample_gap_s=0.0,
        mean_thread_sample_gap_s=0.0,
    )
    fabricated_refused = False
    try:
        require_live_sampler(fabricated, label="R0248 fabricated receipt")
    except (Round0246Error, Round0247Error):
        fabricated_refused = True
    attacks.append({
        "attack": "r0248-self-4: hand the liveness gate a fabricated receipt",
        "gate_refused_it": fabricated_refused,
        "closed": False,
        "residual": (
            "unchanged from R0247 and unclosable in process. A gate that reads "
            "a receipt cannot verify the instrument that produced it. The "
            "external cgroup bound does not help: it bounds memory, not "
            "honesty. Only independent recomputation by a reviewer does"
        ),
    })

    # the external bound, from inside — the summary; the measurement is in
    # basemap.round0248_external's escape battery
    attacks.append({
        "attack": (
            "r0248-self-5: defeat the EXTERNAL bound from inside the node - "
            "raise my own memory.max, raise my own memory.swap.max, move "
            "myself to the parent cgroup, fork out"
        ),
        "closed": None,
        "residual": (
            "mode-dependent and MEASURED, not asserted: see "
            "round0248-external-memory-bound-control-v1. Under a root-owned "
            "scope every attempt is refused with EACCES. Under a user scope "
            "the first two SUCCEED, because systemd delegates the user "
            "manager's subtree; the runner defaults to user-scope because it "
            "needs no privilege and still binds every node that does not "
            "deliberately cheat"
        ),
    })

    succeeded = [row["attack"] for row in attacks if row.get("closed") is False]
    return {
        "instrument": "round0248-self-attack-battery-v1",
        "attacks": attacks,
        "attacks_run": len(attacks),
        "attacks_closed": sum(1 for row in attacks if row.get("closed")),
        "attacks_that_still_succeed": succeeded,
        "note": GUARD_SCOPE_NOTE,
    }


def gap_closure_receipt(
    *,
    inventory: Mapping[str, Any],
    gap1: Mapping[str, Any],
    gap2: Mapping[str, Any],
    gap3: Mapping[str, Any],
    gap4: Mapping[str, Any],
    self_attacks: Mapping[str, Any],
) -> dict[str, Any]:
    arms = {
        "the_two_observation_gap_bounds_read_the_registry": bool(gap1["holds"]),
        "the_sampler_ceiling_reads_the_registry": bool(gap2["holds"]),
        "the_runtime_abort_reader_marker_sanctions_nothing": bool(gap3["holds"]),
        "replay_is_registered_and_the_waivers_are_derived": bool(gap4["holds"]),
        "the_inventory_is_derived_mechanically_and_is_complete": bool(
            inventory["holds"]
        ),
        "every_self_attack_is_published_with_its_outcome": bool(
            self_attacks["attacks_run"] >= 5
        ),
    }
    return {
        "instrument": "round0248-gap-closure-v1",
        "mechanical_inventory": dict(inventory),
        "gap1_observation_gap": dict(gap1),
        "gap2_sampler_bytes": dict(gap2),
        "gap3_abort_reader": dict(gap3),
        "gap4_replay": dict(gap4),
        "self_attack_battery": dict(self_attacks),
        "arms": arms,
        "holds": all(arms.values()),
        "scope_note": GUARD_SCOPE_NOTE,
        "guard_axis": "anonymous, never RSS",
        "registry_fingerprint": registry_fingerprint(),
    }


def run_all_controls(*, repo_root: str, workspace: str) -> dict[str, Any]:
    """Every R0248 in-process control, for the node and the smoke alike."""
    os.makedirs(workspace, exist_ok=True)
    return gap_closure_receipt(
        inventory=require_inventory_complete(repo_root=repo_root),
        gap1=run_gap1_observation_gap_control(),
        gap2=run_gap2_sampler_bytes_control(),
        gap3=run_gap3_abort_reader_control(),
        gap4=run_gap4_replay_control(repo_root=repo_root),
        self_attacks=run_self_attack_battery(),
    )


__all__ = [
    "GUARD_SCOPE_NOTE",
    "ROUND_ID",
    "gap_closure_receipt",
    "run_all_controls",
    "run_gap1_observation_gap_control",
    "run_gap2_sampler_bytes_control",
    "run_gap3_abort_reader_control",
    "run_gap4_replay_control",
    "run_self_attack_battery",
]
