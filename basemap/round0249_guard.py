"""R0249 — the two in-process defects review-0248-01 found, with their controls.

**Scope, deliberately narrow.** Four rounds (R0245–R0248) have each closed the
previous review's defeats and been defeated again. Every remaining in-process
defeat the reviewers hold open requires a node to *deliberately cheat*: rebind
`REGISTERED_SAFETY_PARAMETERS`, mutate the dict behind the `MappingProxyType`
with `gc.get_referents()`, script a module-level clock, or fabricate a receipt
outright. Those are **not** the threat this program faces — the runners here
follow instructions in good faith and make mistakes — and four rounds of
evidence say Python cannot be made tamper-proof against itself. R0249 fixes what
is genuinely broken, moves the memory bound somewhere a mistake cannot defeat
(`basemap.round0249_external`), and stops.

Two defects, both from review-0248-01 §B:

1. **The disclosure fields were plain mutable attributes.** R0247 had already
   established the remedy on this exact class — `AbortPollGate.max_poll_spacing_s`
   is a read-only property — and R0248 then added three new caller-visible facts
   (`replay`, `clock_is_the_registered_monotonic_clock`,
   `inner_is_a_registered_abort_reader`) as ordinary instance attributes. The
   reviewer's two statements —

       g.replay = False
       g.clock_is_the_registered_monotonic_clock = True

   — turned a replayed, scripted-clock gate into sealed enforcement evidence.
   All four are read-only now (`safety_overrides` too, because the independent
   arm below is computed from it).

2. **`replay` was registered `enforcement="clamped"`, which is false three ways**
   and, worse, switched off the arm that would have caught defect 1 on its own:
   `weakening_overrides()` filters on `refused`, so the *weakening* `replay`
   record was invisible and `no_weakening_safety_override_was_attempted`
   reported `True` on a replayed gate. `replay` is now registered `declared`, a
   third enforcement class whose enforcement is at the SEALING boundary:
   `sealing_refused_overrides()` carries it, and
   `require_enforcement_evidence()` refuses on it **without reading a single
   disclosure attribute**.

Why `declared` rather than simply `refused`: registering it `refused` would make
`require()` itself fail on every replay gate, which would make every
reviewer-shaped replay control in R0246–R0248 pass for a reason that has nothing
to do with the gap it plants — nine currently-meaningful `gate_refused_it` arms
would become vacuous in one edit. The fix has to bite where the damage is, which
is sealing, and `run_legitimate_replay_still_scores()` below is the non-vacuity
control that proves it did not bite anywhere else.

Nothing here signals a process, starts a child, touches the GPU, or reads a bulk
array.
"""
from __future__ import annotations

import os
from typing import Any

from basemap.round0245_guard import AbortPollTracker
from basemap.round0246_guard import (
    AbortPollGate,
    replay_gap_through_the_gate,
    require_enforcement_evidence,
)
from basemap.round0247_guard import _sanctioned_reader, run_clamp_controls
from basemap.round0247_registry import (
    ENFORCEMENT_CLAMPED,
    ENFORCEMENT_DECLARED,
    REGISTERED_SAFETY_PARAMETERS,
    Round0247Error,
    registered_bounds,
    registered_value,
    registry_fingerprint,
    sealing_refused_overrides,
    verify_registry,
    weakening_overrides,
)
from basemap.round0248_inventory import (
    derive_gate_constant_inventory,
    discover_registry_regime_modules,
    discover_round_modules,
    require_inventory_complete,
)

ROUND_ID = "0249"

GUARD_SCOPE_NOTE = (
    "R0249 closes the two in-process defects review-0248-01 named as real - "
    "four mutable disclosure attributes, and one false enforcement "
    "classification that switched off the arm which would have caught them - "
    "and replaces the hand-written GUARDED_MODULES list with discovery. It "
    "does NOT chase the scripted module clock, the gc.get_referents() "
    "mutation, or the fabricated receipt. Each of those requires the node to "
    "deliberately cheat; each has moved one level up every time a round has "
    "tried; and the mitigation that actually works for all three is "
    "independent recomputation from sealed bytes by a reviewer who did not "
    "run the round. They are carried as ACCEPTED RESIDUAL RISK, named in the "
    "receipt, not as closed."
)

ACCEPTED_RESIDUAL_RISKS: tuple[dict[str, str], ...] = (
    {
        "risk": "a scripted module-level clock",
        "shape": (
            "basemap.round0244_guard._now is an attribute of a module object "
            "the node owns; assigning a 1000x-slow function to it makes every "
            "measured interval a fiction"
        ),
        "why_it_is_not_chased": (
            "registering the callable moves the assignment one level up, which "
            "is what R0245, R0246 and R0247 each did in turn and each was "
            "defeated for. A node that scripts its clock is cheating, not "
            "mistaken, and cheating is not the class this program fails at"
        ),
        "the_mitigation_that_works": (
            "independent recomputation from the sealed artifacts by a reviewer "
            "who did not run the round. Every review since R0245 has done "
            "exactly this and it is what has found every one of these defects"
        ),
    },
    {
        "risk": "gc.get_referents() mutation of the registry mapping",
        "shape": (
            "the dict behind REGISTERED_SAFETY_PARAMETERS' MappingProxyType is "
            "reachable through the garbage collector, and mutating it moves a "
            "bound while the fingerprint can be recomputed to match"
        ),
        "why_it_is_not_chased": (
            "a caller who can mutate the mapping can mutate the digest, so no "
            "in-process check bounds it. What R0248's fingerprint fix did buy "
            "is real and is kept: the digest now covers the object the "
            "decisions read, so an ACCIDENTAL divergence fails closed"
        ),
        "the_mitigation_that_works": (
            "the external bound of basemap.round0249_external, which is not a "
            "Python object at all, plus independent recomputation by a reviewer"
        ),
    },
    {
        "risk": "a fabricated receipt",
        "shape": (
            "a node can write any JSON it likes into its own artifact and seal "
            "it. No guard in this process can distinguish a measured receipt "
            "from an invented one"
        ),
        "why_it_is_not_chased": (
            "R0247 published this as unclosable in process and nothing since "
            "has changed that. It is the ceiling on what any in-process guard "
            "family can be worth"
        ),
        "the_mitigation_that_works": (
            "independent recomputation from sealed bytes by a reviewer who did "
            "not run the round - which is the protocol's fourth safeguard and "
            "the reason it exists"
        ),
    },
)


def _registered_headroom() -> int:
    return int(registered_value("max_declared_headroom_bytes"))


def _registered_slope_floor() -> float:
    return float(registered_value("min_binding_slope_bytes_per_s"))


def _replayed_gate(label: str) -> AbortPollGate:
    """review-0248-01 §B's subject: scripted clock, `replay=True`, real reader."""
    ticks = iter([0.0, 0.0, 0.001, 0.002])
    gate = AbortPollGate(
        inner=_sanctioned_reader, headroom_bytes=_registered_headroom(),
        label=label, clock=lambda: next(ticks), replay=True,
    )
    gate.start()
    gate("replayed read 1")
    gate("replayed read 2")
    gate.finish()
    return gate


# --------------------------------------------------------------------------- #
# defect 1 — the disclosure fields are read-only
# --------------------------------------------------------------------------- #
def run_readonly_disclosure_control() -> dict[str, Any]:
    """review-0248-01 §B, replayed exactly, then extended.

    The reviewer made **two** assignments and sealed a replayed, scripted-clock
    gate as enforcement evidence. This makes those two and two more — the
    abort-reader disclosure and the override tuple the new sealing arm reads —
    and every one must raise `AttributeError` at the statement that makes it.
    Then the sealing gate is asked for the verdict anyway, and must refuse.
    """
    verify_registry(label="R0249 read-only disclosure control")
    gate = _replayed_gate("R0249 review-0248-01 §B replay")
    attempts: list[dict[str, Any]] = []
    for attribute, value in (
        ("replay", False),
        ("clock_is_the_registered_monotonic_clock", True),
        ("inner_is_a_registered_abort_reader", True),
        ("safety_overrides", ()),
    ):
        before = getattr(gate, attribute)
        error = None
        assigned = False
        try:
            setattr(gate, attribute, value)
            assigned = True
        except AttributeError as exc:
            error = f"{type(exc).__name__}: {exc}"
        attempts.append({
            "attribute": attribute,
            "attempted_value": repr(value),
            "value_before": repr(before),
            "value_after": repr(getattr(gate, attribute)),
            "the_assignment_was_refused": bool(not assigned),
            "the_value_is_unchanged": bool(
                repr(getattr(gate, attribute)) == repr(before)
            ),
            "error": error,
        })

    verdict = gate.require(
        measured_slope_bytes_per_s=_registered_slope_floor()
    )
    sealed = False
    seal_failures: list[str] = []
    try:
        require_enforcement_evidence(verdict, label="R0249 disclosure control")
        sealed = True
    except Round0247Error as error:
        seal_failures = list(
            (gate.last_verdict or {}).get("failures") or []
        )
        seal_message = str(error)
    else:
        seal_message = None

    arms = {
        "every_disclosure_assignment_raises": bool(
            all(row["the_assignment_was_refused"] for row in attempts)
        ),
        "no_disclosure_value_moved": bool(
            all(row["the_value_is_unchanged"] for row in attempts)
        ),
        "the_verdict_still_reports_it_as_a_replay": bool(
            verdict["replay_only"]
        ),
        "the_waived_arms_are_still_named": bool(
            verdict["gate_arms_waived_by_declaration"]
        ),
        "the_seal_is_refused": bool(not sealed),
    }
    evidence = {
        "control": "round0249-readonly-disclosure-control-v1",
        "planted": (
            "review-0248-01 §B's two assignments on a gate built with a "
            "scripted clock, replay=True and the name-registered abort "
            "reader, plus two more of the same shape"
        ),
        "what_it_did_before_r0249": (
            "replay_only -> False, gate_arms_waived_by_declaration -> [], and "
            "require_enforcement_evidence() SEALED the verdict"
        ),
        "attempts": attempts,
        "verdict_replay_only": bool(verdict["replay_only"]),
        "verdict_waived_arms": list(
            verdict["gate_arms_waived_by_declaration"]
        ),
        "seal_refused": bool(not sealed),
        "seal_message": seal_message,
        "seal_failure_arms": seal_failures,
        "arms": arms,
        "failures": [name for name, ok in arms.items() if not ok],
        "registry_fingerprint": registry_fingerprint(),
    }
    evidence["holds"] = not evidence["failures"]
    if evidence["failures"]:
        raise Round0247Error(
            f"R0249 READ-ONLY DISCLOSURE CONTROL DID NOT FIRE: "
            f"{evidence['failures']}"
        )
    return evidence


# --------------------------------------------------------------------------- #
# defect 2 — the enforcement classification, and the arm it switched off
# --------------------------------------------------------------------------- #
def run_sealing_arm_control() -> dict[str, Any]:
    """The independent catch: refuse a replay from the OVERRIDE RECORD alone.

    The read-only properties above close the assignment. This closes the class
    behind it: even handed a verdict dictionary in which **every** disclosure
    field has been flipped to the innocent value — `replay_only: False`,
    `gate_arms_waived_by_declaration: []`,
    `clock_is_the_registered_monotonic_clock: True` — the sealing gate must
    still refuse, on `no_sealing_blocking_override_was_attempted`, which is
    computed from the `replay` declaration record built at construction.

    That is review-0248-01's own recommendation ("makes the pre-existing
    weakening arm refuse a replayed gate whether or not `self.replay` survives
    to `require()`"), routed through the `declared` class so it fires at
    sealing without making every legitimate replay control vacuous.
    """
    verify_registry(label="R0249 sealing arm control")
    gate = _replayed_gate("R0249 sealing arm")
    verdict = gate.require(
        measured_slope_bytes_per_s=_registered_slope_floor()
    )
    #: every disclosure field flipped to the innocent value, in a COPY - the
    #: gate itself refuses the assignment now, so this is the strictly harder
    #: case of a caller who hands the sealing gate a laundered dictionary.
    laundered = dict(verdict)
    laundered["replay_only"] = False
    laundered["gate_arms_waived_by_declaration"] = []
    laundered["clock_is_the_registered_monotonic_clock"] = True
    laundered["inner_is_a_registered_abort_reader"] = True

    def _seal(body: dict[str, Any], label: str) -> dict[str, Any]:
        try:
            require_enforcement_evidence(body, label=label)
        except Round0247Error as error:
            message = str(error)
            arms = [
                arm for arm in (
                    "not_a_replay",
                    "clock_is_the_registered_monotonic_clock",
                    "inner_is_a_registered_abort_reader",
                    "no_weakening_safety_override_was_attempted",
                    "no_sealing_blocking_override_was_attempted",
                    "no_gate_arm_was_waived_by_a_declaration",
                    "the_abort_reader_is_sanctioned_by_registered_name",
                ) if f"'{arm}'" in message
            ]
            return {"refused": True, "failure_arms": arms,
                    "message": message[:400]}
        return {"refused": False, "failure_arms": [], "message": None}

    honest = _seal(dict(verdict), "R0249 honest replay verdict")
    laundered_result = _seal(laundered, "R0249 laundered replay verdict")

    records = list(verdict.get("sealing_blocking_overrides") or [])
    arms = {
        "the_honest_replay_verdict_is_refused": bool(honest["refused"]),
        "the_laundered_verdict_is_still_refused": bool(
            laundered_result["refused"]
        ),
        "the_laundered_refusal_is_on_the_override_record_alone": bool(
            laundered_result["failure_arms"]
            == ["no_sealing_blocking_override_was_attempted"]
        ),
        "the_blocking_record_is_the_replay_declaration": bool(
            [row["parameter"] for row in records] == ["replay"]
        ),
        "the_record_is_a_weakening": bool(
            all(row.get("kind") == "weakening" for row in records) and records
        ),
    }
    evidence = {
        "control": "round0249-sealing-arm-control-v1",
        "planted": (
            "a verdict from a replayed, scripted-clock gate with every "
            "disclosure field rewritten to the innocent value"
        ),
        "honest_verdict": honest,
        "laundered_verdict": laundered_result,
        "sealing_blocking_overrides": records,
        "no_sealing_blocking_override_was_attempted": bool(
            verdict["no_sealing_blocking_override_was_attempted"]
        ),
        "arms": arms,
        "failures": [name for name, ok in arms.items() if not ok],
        "registry_fingerprint": registry_fingerprint(),
    }
    evidence["holds"] = not evidence["failures"]
    if evidence["failures"]:
        raise Round0247Error(
            f"R0249 SEALING ARM CONTROL DID NOT FIRE: {evidence['failures']}"
        )
    return evidence


def run_declared_enforcement_control() -> dict[str, Any]:
    """`replay`'s classification, corrected — and every way it was false, closed.

    review-0248-01 §B finding 3 named three:

    1. its only call site is `record_declaration()`, which does not clamp;
    2. `run_clamp_controls()` proved a clamp for it through `clamp()`, which no
       call site uses for it;
    3. `weakening_overrides()` filters on `refused`, so the weakening record
       never reached `no_weakening_safety_override_was_attempted`.

    And a fourth the reviewer named separately: the sealed receipt carried
    `registered_replay_enforcement: "clamped"`, a claim about `replay` that the
    code contradicted by design.
    """
    verify_registry(label="R0249 declared enforcement control")
    parameter = REGISTERED_SAFETY_PARAMETERS["replay"]
    clamp_controls = run_clamp_controls()
    replay_row = next(
        row for row in clamp_controls["rows"] if row["parameter"] == "replay"
    )
    gate = _replayed_gate("R0249 declared enforcement")
    verdict = gate.require(
        measured_slope_bytes_per_s=_registered_slope_floor()
    )
    bounds = registered_bounds(["replay"])

    arms = {
        "replay_is_registered_declared": bool(
            parameter.enforcement == ENFORCEMENT_DECLARED
        ),
        "replay_is_no_longer_registered_clamped": bool(
            parameter.enforcement != ENFORCEMENT_CLAMPED
        ),
        "its_control_runs_through_its_real_call_site": bool(
            replay_row["controlled_through"] == "record_declaration"
        ),
        "no_clamp_is_claimed_for_it": bool(
            replay_row["the_registered_value_was_used"] is False
            and replay_row["the_declaration_stands"] is True
        ),
        "the_weakening_record_reaches_the_sealing_gate": bool(
            sealing_refused_overrides(gate.safety_overrides)
        ),
        "it_still_does_not_fail_require": bool(
            not weakening_overrides(gate.safety_overrides)
            and verdict["holds"]
        ),
        "the_receipt_no_longer_publishes_clamped_for_replay": bool(
            bounds["registered_replay_enforcement"] == ENFORCEMENT_DECLARED
        ),
        "it_is_absent_from_the_clamped_envelope_list": bool(
            "replay" not in [
                row["parameter"]
                for row in verdict["envelope_declarations_clamped_to_the_registry"]
            ]
        ),
    }
    evidence = {
        "control": "round0249-declared-enforcement-control-v1",
        "registered_enforcement": parameter.enforcement,
        "enforcement_before_r0249": ENFORCEMENT_CLAMPED,
        "clamp_control_row": {
            key: value for key, value in replay_row.items() if key != "record"
        },
        "clamp_controls_summary": {
            "parameters_controlled": clamp_controls["parameters_controlled"],
            "through_clamp": clamp_controls[
                "parameters_controlled_through_clamp"
            ],
            "through_record_declaration": clamp_controls[
                "parameters_controlled_through_record_declaration"
            ],
        },
        "registered_bounds_block": bounds,
        "arms": arms,
        "failures": [name for name, ok in arms.items() if not ok],
        "registry_fingerprint": registry_fingerprint(),
    }
    evidence["holds"] = not evidence["failures"]
    if evidence["failures"]:
        raise Round0247Error(
            "R0249 DECLARED ENFORCEMENT CONTROL DID NOT FIRE: "
            f"{evidence['failures']}"
        )
    return evidence


def run_legitimate_replay_still_scores() -> dict[str, Any]:
    """The NON-VACUITY control for the classification change.

    Registering `replay` as `refused` — the obvious one-word fix — would have
    made `require()` refuse every replay gate unconditionally, and the nine
    `gate_refused_it` arms in R0246's reviewer-shaped controls would then have
    read `True` for a reason unrelated to the defect each one plants. This
    checks the opposite: a replay of a gap that is INSIDE the ceiling still
    passes `require()`, and a replay of R0245 attempt-1's `5.828…` s gap still
    fails it, on the spacing arm rather than on an override.
    """
    verify_registry(label="R0249 replay non-vacuity control")
    headroom = _registered_headroom()
    slope = _registered_slope_floor()
    ceiling = float(registered_value("r0246_max_poll_spacing_s"))

    inside = replay_gap_through_the_gate(
        gap_s=ceiling / 10.0, headroom_bytes=headroom,
        measured_slope_bytes_per_s=slope,
        label="R0249 replay of a gap inside the ceiling",
    )
    attempt_1 = replay_gap_through_the_gate(
        gap_s=5.828025072987657, headroom_bytes=headroom,
        measured_slope_bytes_per_s=0.0,
        label="R0249 replay of R0245 attempt-1's gap",
    )
    arms = {
        "a_replay_inside_the_ceiling_still_passes_require": bool(
            not inside["gate_refused_it"]
        ),
        "attempt_1s_gap_is_still_refused": bool(attempt_1["gate_refused_it"]),
        "and_it_is_refused_on_the_spacing_arm": bool(
            "meets_the_registered_ceiling" in (attempt_1["message"] or "")
        ),
        "the_refusal_is_not_an_override_arm": bool(
            "no_weakening_safety_override_was_attempted"
            not in (attempt_1["message"] or "")
            and "no_sealing_blocking_override_was_attempted"
            not in (attempt_1["message"] or "")
        ),
    }
    evidence = {
        "control": "round0249-replay-non-vacuity-control-v1",
        "why_it_exists": (
            "review-0248-01 recommended registering replay as `refused`. That "
            "would refuse every replay gate at require(), turning nine "
            "meaningful gate_refused_it arms in R0246 into arms that cannot "
            "fail. The `declared` class bites at sealing instead, and this is "
            "the control that shows require() still discriminates"
        ),
        "replay_inside_the_ceiling": inside,
        "replay_of_attempt_1": attempt_1,
        "registered_max_poll_spacing_s": ceiling,
        "arms": arms,
        "failures": [name for name, ok in arms.items() if not ok],
        "registry_fingerprint": registry_fingerprint(),
    }
    evidence["holds"] = not evidence["failures"]
    if evidence["failures"]:
        raise Round0247Error(
            "R0249 REPLAY NON-VACUITY CONTROL DID NOT FIRE: "
            f"{evidence['failures']}"
        )
    return evidence


# --------------------------------------------------------------------------- #
# the discovered module scope
# --------------------------------------------------------------------------- #
def run_discovery_inventory_control(*, repo_root: str) -> dict[str, Any]:
    """The list is gone, and the thing it hid is fixed.

    review-0248-01 §D.3 pointed R0248's own derivation at the modules
    `GUARDED_MODULES` did not name and got exactly one hit:
    `experiments/round0242_nodes.py:246`, `WATCHDOG_ANON_BYTES`, against a
    second literal copy of a registered number. Three arms:

    * the scope is **discovered**: every `round0*.py` under `basemap/` and
      `experiments/` is scanned, and the count is reported rather than declared;
    * the discovered scope contains the modules the hand-written list waived,
      including `round0242_nodes.py`;
    * **zero** bare registered symbols anywhere in it, and a planted one in a
      module outside the registry regime is still caught — which is the exact
      position the R0242 defect was in.
    """
    verify_registry(label="R0249 discovery inventory control")
    inventory = require_inventory_complete(repo_root=repo_root)
    constants = inventory["gate_constants"]
    discovered = discover_round_modules(repo_root=repo_root)
    regime = discover_registry_regime_modules(repo_root=repo_root)

    arms = {
        "the_scope_is_discovered_not_declared": bool(
            len(discovered) == constants["discovery"]["round_modules_scanned"]
        ),
        "the_scope_is_wider_than_the_retired_hand_written_list": bool(
            len(discovered) > 13
        ),
        "the_module_the_hand_list_omitted_is_in_scope": bool(
            "experiments/round0242_nodes.py" in discovered
        ),
        "and_it_is_now_in_the_triage_scope_too": bool(
            "experiments/round0242_nodes.py" in regime
        ),
        "the_modules_the_hand_list_waived_are_in_the_triage_scope": bool(
            {"basemap/round0246_tie.py", "basemap/round0247_precision.py"}
            <= set(regime)
        ),
        "no_bare_registered_symbol_anywhere_in_the_release": bool(
            not constants["bare_registered_symbols_anywhere_in_the_release"]
        ),
        "no_untriaged_constant_in_the_registry_regime": bool(
            not constants["untriaged_in_the_registry_regime"]
        ),
        "the_derivation_still_finds_defects": bool(
            _planted_defect_is_caught(repo_root=repo_root)
        ),
    }
    evidence = {
        "control": "round0249-discovery-inventory-control-v1",
        "round_modules_discovered": len(discovered),
        "registry_regime_modules_discovered": len(regime),
        "registry_regime_modules": list(regime),
        "comparisons_over_the_whole_scope": constants["discovery"][
            "comparisons_over_the_whole_scope"
        ],
        "comparisons_in_the_registry_regime": constants[
            "comparisons_against_module_level_constants"
        ],
        "distinct_symbols_in_the_registry_regime": len(
            constants["distinct_symbols"]
        ),
        "bare_registered_symbols_anywhere": constants[
            "bare_registered_symbols_anywhere_in_the_release"
        ],
        "what_the_hand_written_list_hid": (
            "experiments/round0242_nodes.py:246 - "
            "int(host['anonymous_bytes']) > WATCHDOG_ANON_BYTES, with "
            "WATCHDOG_ANON_BYTES = 60 * (1 << 30) at :157, a second literal "
            "copy of the registered max_declared_anonymous_budget_bytes. It is "
            "byte-for-byte the rule R0248 fixed at round0244_guard:404 and it "
            "sat one file outside the list"
        ),
        "arms": arms,
        "failures": [name for name, ok in arms.items() if not ok],
        "registry_fingerprint": registry_fingerprint(),
    }
    evidence["holds"] = not evidence["failures"]
    if evidence["failures"]:
        raise Round0247Error(
            "R0249 DISCOVERY INVENTORY CONTROL DID NOT FIRE: "
            f"{evidence['failures']}"
        )
    return evidence


def _planted_defect_is_caught(*, repo_root: str) -> bool:
    """Plant `round0242_nodes:246`'s exact shape in a throwaway tree.

    Written into a temporary directory that mimics the discovery layout, in a
    module that does **not** import the registry — so it is outside the triage
    scope, exactly where the real defect was — and the wide scope must catch it
    anyway. The tree is removed afterwards; nothing is written into the release.
    """
    import shutil  # noqa: PLC0415 - control-local, never on a node's hot path
    import tempfile  # noqa: PLC0415 - same

    planted = tempfile.mkdtemp(prefix="r0249-inventory-control-")
    try:
        directory = os.path.join(planted, "experiments")
        os.makedirs(directory, exist_ok=True)
        with open(
            os.path.join(directory, "round0242_planted.py"), "w",
            encoding="utf-8",
        ) as handle:
            handle.write(
                "WATCHDOG_ANON_BYTES = 60 * (1 << 30)\n"
                "def poll(host):\n"
                "    return int(host['anonymous_bytes']) > WATCHDOG_ANON_BYTES\n"
            )
        derived = derive_gate_constant_inventory(repo_root=planted)
        bare = derived["bare_registered_symbols_anywhere_in_the_release"]
        return bool(
            derived["holds"] is False
            and [row["symbol"] for row in bare] == ["WATCHDOG_ANON_BYTES"]
            and bare[0]["module"] == "experiments/round0242_planted.py"
            and bare[0]["module"] not in derived["modules_declared"]
        )
    finally:
        shutil.rmtree(planted, ignore_errors=True)


# --------------------------------------------------------------------------- #
# the round's receipt
# --------------------------------------------------------------------------- #
def guard_fix_receipt(
    *,
    inventory: dict[str, Any],
    disclosure: dict[str, Any],
    sealing: dict[str, Any],
    declared: dict[str, Any],
    non_vacuity: dict[str, Any],
    discovery: dict[str, Any],
) -> dict[str, Any]:
    controls = {
        "readonly_disclosure": disclosure,
        "sealing_arm": sealing,
        "declared_enforcement": declared,
        "replay_non_vacuity": non_vacuity,
        "discovery_inventory": discovery,
    }
    failures = [
        name for name, control in controls.items() if not control["holds"]
    ]
    return {
        "instrument": "round0249-guard-fix-v1",
        "scope_note": GUARD_SCOPE_NOTE,
        "inventory": inventory,
        "controls": controls,
        "accepted_residual_risks": [dict(row) for row in ACCEPTED_RESIDUAL_RISKS],
        "what_is_not_closed": [row["risk"] for row in ACCEPTED_RESIDUAL_RISKS],
        "failures": failures,
        "holds": not failures,
        "registry_fingerprint": registry_fingerprint(),
    }


def tracker_disclosure_properties_are_read_only() -> dict[str, bool]:
    """Source-level statement of the fix, for the receipt."""
    return {
        name: isinstance(getattr(AbortPollTracker, name, None), property)
        for name in (
            "replay",
            "clock_is_the_registered_monotonic_clock",
            "inner_is_a_registered_abort_reader",
            "safety_overrides",
        )
    }


__all__ = [
    "ACCEPTED_RESIDUAL_RISKS",
    "GUARD_SCOPE_NOTE",
    "ROUND_ID",
    "guard_fix_receipt",
    "run_declared_enforcement_control",
    "run_discovery_inventory_control",
    "run_legitimate_replay_still_scores",
    "run_readonly_disclosure_control",
    "run_sealing_arm_control",
    "tracker_disclosure_properties_are_read_only",
]
