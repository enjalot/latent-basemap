"""R0248 contract — the four gaps, the derived inventory, the external bound."""
from __future__ import annotations

import ast
import glob
import os
import subprocess

import pytest

import basemap.round0244_prereq as prereq0244
import basemap.round0246_guard as guard0246
import basemap.round0247_registry as registry
from basemap.round0246_guard import (
    AbortPollGate,
    Round0246Error,
    require_enforcement_evidence,
    require_live_sampler,
)
from basemap.round0247_guard import _healthy_receipt
from basemap.round0248_external import (
    EXTERNAL_BOUND_NOTE,
    cgroup_v2_memory_available,
    external_memory_limit_declaration,
    external_memory_max_bytes,
    machine_total_memory_bytes,
)
from basemap.round0248_guard import (
    _sanctioned_reader,
    run_gap1_observation_gap_control,
    run_gap2_sampler_bytes_control,
    run_gap3_abort_reader_control,
    run_gap4_replay_control,
    run_self_attack_battery,
)
from basemap.round0248_inventory import (
    GUARDED_MODULES,
    derive_arm_waiver_inventory,
    derive_gate_constant_inventory,
    require_inventory_complete,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R0248_FILES = (
    "basemap/round0248_inventory.py",
    "basemap/round0248_guard.py",
    "basemap/round0248_external.py",
    "experiments/round0248_nodes.py",
    "experiments/prepare_round0248_queue.py",
)


# --------------------------------------------------------------------------- #
# the registry
# --------------------------------------------------------------------------- #
def test_the_fingerprint_covers_the_object_the_decisions_read() -> None:
    """review-0247-01 A.1: it hashed `_PARAMETERS` and the guards read the map."""
    state = registry.verify_registry()
    assert state["holds"] is True
    assert state["the_fingerprint_covers_the_object_the_decisions_read"] is True
    saved = registry.REGISTERED_SAFETY_PARAMETERS
    try:
        forged = dict(saved)
        forged.pop("replay")
        registry.REGISTERED_SAFETY_PARAMETERS = forged  # type: ignore[assignment]
        with pytest.raises(registry.Round0247Error):
            registry.registered_value("r0246_max_poll_spacing_s")
    finally:
        registry.REGISTERED_SAFETY_PARAMETERS = saved
    assert registry.verify_registry()["holds"] is True


def test_replay_is_a_registered_safety_parameter_at_false() -> None:
    assert "replay" in registry.REGISTERED_SAFETY_PARAMETERS
    assert registry.registered_value("replay") == 0.0
    assert registry.REGISTERED_SAFETY_PARAMETERS["replay"].direction == "ceiling"


def test_the_retired_marker_sanctions_nothing_and_is_recorded() -> None:
    def stranger(_where: str) -> None:
        return None

    assert registry.is_registered_abort_reader(stranger) is False
    registry.registered_abort_reader(stranger)
    assert registry.is_registered_abort_reader(stranger) is False
    sanction = registry.abort_reader_sanction(stranger)
    assert sanction["mechanism"] == "retired_marker_only"
    assert sanction["marked_and_unsanctioned"] is True
    assert any(
        row["qualified_name"].endswith("stranger")
        for row in registry.unsanctioned_marker_applications()
    )
    #: and the name-registered reader still works, or the fix is a wrecking ball
    assert registry.is_registered_abort_reader(_sanctioned_reader) is True
    assert registry.abort_reader_sanction(
        _sanctioned_reader
    )["mechanism"] == "registered_name"


# --------------------------------------------------------------------------- #
# the mechanically derived inventory
# --------------------------------------------------------------------------- #
def test_the_inventory_is_derived_and_complete_at_this_checkout() -> None:
    inventory = require_inventory_complete(repo_root=REPO)
    assert inventory["holds"] is True
    assert inventory["gate_constants"][
        "comparisons_against_module_level_constants"] > 40
    assert inventory["arm_waivers"]["arms_waivable_by_a_declaration"] >= 2


def test_every_guard_module_in_the_tree_is_in_guarded_modules() -> None:
    """A guard module added and not enumerated is itself the defect."""
    found = {
        os.path.relpath(path, REPO)
        for path in glob.glob(os.path.join(REPO, "basemap", "round024[4-8]*.py"))
    }
    #: the tie/precision/did/sampler helpers hold no gate comparisons of their
    #: own; the ones that do are enumerated.
    missing = {
        name for name in found
        if name.endswith(("_guard.py", "_prereq.py", "_registry.py",
                          "_external.py", "_inventory.py"))
    } - set(GUARDED_MODULES) - {"basemap/round0248_inventory.py"}
    assert not missing, sorted(missing)


def test_the_derivation_catches_a_planted_bare_registered_comparison(
    tmp_path,
) -> None:
    """The positive control for the DERIVATION itself."""
    module = tmp_path / "planted_guard.py"
    module.write_text(
        "R0246_MAX_POLL_SPACING_S = 2.5\n"
        "def gate(gap):\n"
        "    return gap <= R0246_MAX_POLL_SPACING_S\n",
        encoding="utf-8",
    )
    import basemap.round0248_inventory as inventory_module

    saved = inventory_module.GUARDED_MODULES
    try:
        inventory_module.GUARDED_MODULES = ("planted_guard.py",)
        derived = derive_gate_constant_inventory(repo_root=str(tmp_path))
    finally:
        inventory_module.GUARDED_MODULES = saved
    assert derived["holds"] is False
    assert derived["defects"][0]["status"] == "bare_registered_symbol"
    assert derived["defects"][0]["symbol"] == "R0246_MAX_POLL_SPACING_S"


def test_the_derivation_catches_a_planted_unregistered_arm_waiver(
    tmp_path,
) -> None:
    module = tmp_path / "planted_waiver.py"
    module.write_text(
        "class G:\n"
        "    def require(self):\n"
        "        return [n for n, ok in (\n"
        "            ('an_arm', bool(self.measured or self.pretend)),\n"
        "        ) if not ok]\n",
        encoding="utf-8",
    )
    import basemap.round0248_inventory as inventory_module

    saved = inventory_module.GUARDED_MODULES
    try:
        inventory_module.GUARDED_MODULES = ("planted_waiver.py",)
        derived = derive_arm_waiver_inventory(repo_root=str(tmp_path))
    finally:
        inventory_module.GUARDED_MODULES = saved
    assert derived["holds"] is False
    #: both operands of the disjunction are reported: the derivation does
    #: not decide which one is "the real measurement", it demands that
    #: every name capable of switching the arm on be registered.
    assert {row["waived_by"] for row in derived["defects"]} == {
        "measured", "pretend"
    }


# --------------------------------------------------------------------------- #
# the four gap controls
# --------------------------------------------------------------------------- #
def test_gap1_two_assignments_no_longer_pass_the_5_second_attack() -> None:
    evidence = run_gap1_observation_gap_control()
    assert evidence["holds"] is True
    assert evidence["after_both_assignments"]["refused"] is True


def test_gap1_without_the_fix_the_assignment_would_have_worked() -> None:
    """The derivation, not the guard, is what makes this non-vacuous.

    The fix removes the module global from the decision, so there is no state
    in which the assignment succeeds any more. What is asserted here is that
    the mirror really does move — i.e. the control is planting a live value —
    and that the decision does not follow it.
    """
    saved = guard0246.WATCHDOG_MAX_OBSERVATION_GAP_S
    try:
        guard0246.WATCHDOG_MAX_OBSERVATION_GAP_S = 1.0e6
        assert guard0246.WATCHDOG_MAX_OBSERVATION_GAP_S == 1.0e6
        with pytest.raises(Round0246Error):
            require_live_sampler(
                _healthy_receipt(max_thread_sample_gap_s=5.0, sampled_wall_s=36000.0,
                                 mean_thread_sample_gap_s=5.0),
                label="R0248 mirror moved",
            )
    finally:
        guard0246.WATCHDOG_MAX_OBSERVATION_GAP_S = saved


def test_gap2_the_sampler_ceiling_is_read_from_the_registry() -> None:
    evidence = run_gap2_sampler_bytes_control()
    assert evidence["holds"] is True
    assert prereq0244.sampler_max_anonymous_bytes() == registry.registered_value(
        "sampler_max_anonymous_bytes"
    )


def test_gap3_a_marked_no_op_fails_both_gates() -> None:
    evidence = run_gap3_abort_reader_control()
    assert evidence["holds"] is True
    assert "the_gate_wraps_a_registered_abort_reader" in evidence[
        "gate_failure_arms"
    ]
    assert evidence["sealing_refused"] is True


def test_gap4_a_replay_verdict_names_the_arms_it_waived() -> None:
    evidence = run_gap4_replay_control(repo_root=REPO)
    assert evidence["holds"] is True
    assert evidence["registered_replay"] == 0.0
    assert sorted(evidence["gate_arms_waived_by_declaration"]) == [
        "the_clock_is_the_registered_monotonic_clock",
        "the_gate_wraps_a_registered_abort_reader",
    ]


def test_a_non_replay_gate_with_a_registered_reader_still_seals() -> None:
    """The four fixes must not make honest enforcement evidence unsealable."""
    gate = AbortPollGate(
        inner=_sanctioned_reader,
        headroom_bytes=int(
            registry.registered_value("max_declared_headroom_bytes")
        ),
        label="R0248 honest gate", training_performed=True,
    )
    gate.start()
    for step in range(4):
        gate(f"read {step}")
    gate.finish()
    verdict = gate.require(measured_slope_bytes_per_s=0.0)
    state = require_enforcement_evidence(verdict, label="R0248 honest gate")
    assert state["holds"] is True
    assert state["abort_reader_sanctioned_by_registered_name"] is True
    assert state["gate_arms_waived_by_declaration"] == []


# --------------------------------------------------------------------------- #
# the external bound
# --------------------------------------------------------------------------- #
def test_the_production_limit_is_derived_and_sits_above_the_in_process_budget(
) -> None:
    limit = external_memory_max_bytes()
    budget = registry.registered_value("max_declared_anonymous_budget_bytes")
    assert limit > budget
    total = machine_total_memory_bytes()
    assert total == 0 or limit < total
    declaration = external_memory_limit_declaration()
    assert declaration["max_bytes"] == limit
    assert declaration["swap_max_bytes"] == 0
    assert declaration["required"] is True
    assert EXTERNAL_BOUND_NOTE in declaration["note"]


def test_this_box_can_place_a_cgroup_memory_limit() -> None:
    availability = cgroup_v2_memory_available()
    assert availability["memory_controller"] is True
    assert availability["systemd_run"]
    assert availability["holds"] is True


def test_the_runner_places_the_limit_and_declares_what_it_cannot_do() -> None:
    """The runner change is the mechanism; read it, do not assume it."""
    runner = os.path.expanduser("~/code/workshop/rounds/runner.py")
    source = open(runner, encoding="utf-8").read()
    assert "external_memory_limit_argv" in source
    assert "MemorySwapMax=0" in source
    assert "external_memory_limit" in source
    tree = ast.parse(source)
    names = {
        node.name for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    assert {"external_memory_limit_argv", "external_memory_limit_available"} <= names


# --------------------------------------------------------------------------- #
# the self-attack battery publishes what still succeeds
# --------------------------------------------------------------------------- #
def test_the_self_attacks_publish_the_ones_that_still_succeed() -> None:
    battery = run_self_attack_battery()
    assert battery["attacks_run"] >= 5
    succeeded = battery["attacks_that_still_succeed"]
    #: the scripted module clock and the fabricated receipt are NOT closed and
    #: the round says so. A battery in which everything passes is a battery
    #: that was not trying.
    assert any("scripted" in row or "_now" in row for row in succeeded)
    assert any("fabricated" in row for row in succeeded)
    assert registry.verify_registry()["holds"] is True


# --------------------------------------------------------------------------- #
# hygiene
# --------------------------------------------------------------------------- #
def test_no_r0248_file_contains_a_signalling_construct() -> None:
    """Read the CODE, not the prose. R0248's docstrings discuss SIGKILL."""
    forbidden_calls = {
        "kill", "killpg", "terminate", "send_signal", "pthread_kill",
        "raise_signal",
    }
    for name in R0248_FILES:
        tree = ast.parse(
            open(os.path.join(REPO, name), encoding="utf-8").read(),
            filename=name,
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                target = node.func
                called = (
                    target.attr if isinstance(target, ast.Attribute)
                    else target.id if isinstance(target, ast.Name) else ""
                )
                assert called not in forbidden_calls, (
                    f"{name}:{node.lineno} calls {called}"
                )
            if isinstance(node, ast.Attribute) and node.attr.startswith("SIG"):
                raise AssertionError(f"{name}:{node.lineno} names {node.attr}")


def test_the_allocator_imports_no_array_library() -> None:
    """A kernel OOM kill of a CUDA holder is the wedge; the child holds none."""
    from basemap.round0248_external import _ALLOCATOR

    for banned in ("numpy", "torch", "cupy", "cuml", "cuvs"):
        assert f"import {banned}" not in _ALLOCATOR


def test_this_round_adds_only_its_own_files_and_the_declared_edits() -> None:
    repo = REPO
    committed = subprocess.run(
        ["git", "-C", repo, "diff", "--name-only",
         "0941b3776442cfdf00575f84c2688d63d28a5611", "HEAD"],
        check=False, capture_output=True, text=True,
    ).stdout.split()
    worktree = [
        line[3:] for line in subprocess.run(
            ["git", "-C", repo, "status", "--porcelain"],
            check=False, capture_output=True, text=True,
        ).stdout.splitlines() if line
    ]
    allowed = {
        #: the declared edits to reviewed modules — every one is a comparison
        #: site that had to move from a module global onto the registry
        "basemap/round0244_guard.py",
        "basemap/round0244_prereq.py",
        "basemap/round0245_guard.py",
        "basemap/round0246_guard.py",
        "basemap/round0247_guard.py",
        "basemap/round0247_registry.py",
        "experiments/round0244_nodes.py",
        "experiments/round0245_nodes.py",
        "experiments/round0246_nodes.py",
        "tests/test_round0245_cpu_smoke.py",
        "tests/test_round0246_contract.py",
        "tests/test_round0247_contract.py",
        #: R0248's own files
        "basemap/round0248_inventory.py",
        "basemap/round0248_guard.py",
        "basemap/round0248_external.py",
        "experiments/round0248_nodes.py",
        "experiments/prepare_round0248_queue.py",
        "tests/test_round0248_contract.py",
        "tests/test_round0248_cpu_smoke.py",
    }
    changed = set(committed + worktree)
    assert changed <= allowed, sorted(changed - allowed)
