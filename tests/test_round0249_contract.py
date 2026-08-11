"""R0249 contract — the two defects, the discovered scope, the memory.high bound.

Every guard R0249 adds ships a positive control that plants the defect and
proves the guard catches it, and every one of those controls is exercised here.
Three of them are review-0248-01's own attacks, run verbatim.
"""
from __future__ import annotations

import ast
import glob
import os
import subprocess

import pytest

import basemap.round0247_registry as registry
import basemap.round0249_external as external
import basemap.round0249_guard as guard
from basemap.round0245_guard import AbortPollTracker
from basemap.round0246_guard import (
    AbortPollGate,
    require_enforcement_evidence,
)
from basemap.round0247_guard import _sanctioned_reader, run_clamp_controls
from basemap.round0248_inventory import (
    NOT_A_BOUND,
    derive_gate_constant_inventory,
    discover_registry_regime_modules,
    discover_round_modules,
    require_inventory_complete,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R0249_FILES = (
    "basemap/round0249_guard.py",
    "basemap/round0249_external.py",
    "experiments/round0249_nodes.py",
    "experiments/prepare_round0249_queue.py",
)
_HEADROOM = int(
    registry.REGISTERED_SAFETY_PARAMETERS["max_declared_headroom_bytes"].value
)
_SLOPE_FLOOR = float(
    registry.REGISTERED_SAFETY_PARAMETERS[
        "min_binding_slope_bytes_per_s"].value
)


def _replayed_gate(label: str = "R0249 contract replay") -> AbortPollGate:
    ticks = iter([0.0, 0.0, 0.001, 0.002])
    gate = AbortPollGate(
        inner=_sanctioned_reader, headroom_bytes=_HEADROOM, label=label,
        clock=lambda: next(ticks), replay=True,
    )
    gate.start()
    gate("read 1")
    gate("read 2")
    gate.finish()
    return gate


# --------------------------------------------------------------------------- #
# A. defect 1 — the mutable disclosure attributes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("attribute,value", [
    ("replay", False),
    ("clock_is_the_registered_monotonic_clock", True),
    ("inner_is_a_registered_abort_reader", True),
    ("safety_overrides", ()),
])
def test_each_disclosure_field_is_read_only(attribute, value) -> None:
    gate = _replayed_gate()
    before = getattr(gate, attribute)
    with pytest.raises(AttributeError):
        setattr(gate, attribute, value)
    assert getattr(gate, attribute) == before


def test_they_are_properties_on_the_class_not_just_absent_setters() -> None:
    for name in (
        "replay", "clock_is_the_registered_monotonic_clock",
        "inner_is_a_registered_abort_reader", "safety_overrides",
    ):
        assert isinstance(getattr(AbortPollTracker, name), property), name


def test_review_0248_01_section_b_verbatim_no_longer_seals() -> None:
    """The reviewer's exact two statements, then the seal they bought."""
    gate = _replayed_gate("review-0248-01 §B")
    with pytest.raises(AttributeError):
        gate.replay = False
    with pytest.raises(AttributeError):
        gate.clock_is_the_registered_monotonic_clock = True
    verdict = gate.require(measured_slope_bytes_per_s=_SLOPE_FLOOR)
    assert verdict["replay_only"] is True
    assert verdict["gate_arms_waived_by_declaration"]
    with pytest.raises(registry.Round0247Error):
        require_enforcement_evidence(verdict, label="review-0248-01 §B")


def test_the_readonly_disclosure_control_fires() -> None:
    evidence = guard.run_readonly_disclosure_control()
    assert evidence["holds"] is True
    assert all(row["the_assignment_was_refused"]
               for row in evidence["attempts"])
    assert evidence["seal_refused"] is True


# --------------------------------------------------------------------------- #
# B. defect 2 — the enforcement classification and the arm it switched off
# --------------------------------------------------------------------------- #
def test_replay_is_registered_declared_not_clamped() -> None:
    parameter = registry.REGISTERED_SAFETY_PARAMETERS["replay"]
    assert parameter.enforcement == registry.ENFORCEMENT_DECLARED
    assert parameter.enforcement != registry.ENFORCEMENT_CLAMPED


def test_the_receipt_no_longer_claims_a_clamp_for_replay() -> None:
    block = registry.registered_bounds(["replay"])
    assert block["registered_replay_enforcement"] == "declared"


def test_the_declaration_reaches_the_sealing_set_but_not_the_require_set(
) -> None:
    gate = _replayed_gate()
    assert registry.sealing_refused_overrides(gate.safety_overrides)
    assert not registry.weakening_overrides(gate.safety_overrides)


def test_a_laundered_verdict_is_refused_on_the_override_record_alone() -> None:
    """Every disclosure field flipped; the sealing gate must still refuse."""
    gate = _replayed_gate()
    verdict = dict(gate.require(measured_slope_bytes_per_s=_SLOPE_FLOOR))
    verdict["replay_only"] = False
    verdict["gate_arms_waived_by_declaration"] = []
    verdict["clock_is_the_registered_monotonic_clock"] = True
    verdict["inner_is_a_registered_abort_reader"] = True
    with pytest.raises(registry.Round0247Error) as excinfo:
        require_enforcement_evidence(verdict, label="laundered")
    assert "no_sealing_blocking_override_was_attempted" in str(excinfo.value)


def test_the_clamp_control_routes_declared_through_record_declaration() -> None:
    evidence = run_clamp_controls()
    assert evidence["holds"] is True
    assert evidence["parameters_controlled_through_record_declaration"] == 1
    row = next(r for r in evidence["rows"] if r["parameter"] == "replay")
    assert row["controlled_through"] == "record_declaration"
    assert row["the_declaration_stands"] is True
    assert row["the_weakening_record_reaches_the_sealing_gate"] is True


def test_the_declared_enforcement_control_fires() -> None:
    assert guard.run_declared_enforcement_control()["holds"] is True


def test_a_legitimate_replay_still_discriminates() -> None:
    """Non-vacuity: `require()` did not become a rubber stamp OR a wall."""
    evidence = guard.run_legitimate_replay_still_scores()
    assert evidence["holds"] is True
    assert evidence["replay_inside_the_ceiling"]["gate_refused_it"] is False
    assert evidence["replay_of_attempt_1"]["gate_refused_it"] is True


def test_the_sealing_arm_control_fires() -> None:
    evidence = guard.run_sealing_arm_control()
    assert evidence["holds"] is True
    assert evidence["laundered_verdict"]["failure_arms"] == [
        "no_sealing_blocking_override_was_attempted"
    ]


def test_the_registry_fingerprint_is_pinned_over_the_corrected_class() -> None:
    state = registry.verify_registry()
    assert state["holds"] is True
    assert len(registry.REGISTERED_SAFETY_PARAMETERS) == 21
    #: `enforcement` is one of the six fields the digest covers, so the
    #: reclassification is registered rather than silent.
    assert any(
        row["name"] == "replay" and row["enforcement"] == "declared"
        for row in registry.registry_rows()
    )


# --------------------------------------------------------------------------- #
# C. the discovered module scope
# --------------------------------------------------------------------------- #
def test_the_hand_written_module_list_is_gone() -> None:
    assert not hasattr(
        __import__("basemap.round0248_inventory", fromlist=["x"]),
        "GUARDED_MODULES",
    )


def test_the_scope_is_discovered_from_the_tree() -> None:
    discovered = set(discover_round_modules(repo_root=REPO))
    on_disk = {
        os.path.relpath(path, REPO)
        for directory in ("basemap", "experiments")
        for path in glob.glob(os.path.join(REPO, directory, "round0*.py"))
    }
    assert discovered == on_disk
    assert len(discovered) > 300


def test_round0242_nodes_is_in_scope_and_clean() -> None:
    """review-0248-01 §D.3's seventh bare registered symbol."""
    inventory = require_inventory_complete(repo_root=REPO)
    constants = inventory["gate_constants"]
    assert not constants["bare_registered_symbols_anywhere_in_the_release"]
    assert "experiments/round0242_nodes.py" in discover_round_modules(
        repo_root=REPO)
    assert "experiments/round0242_nodes.py" in discover_registry_regime_modules(
        repo_root=REPO)


def test_the_comparison_at_round0242_nodes_246_reads_the_registry() -> None:
    """Read the source, not the receipt: the `if` must call the registry."""
    path = os.path.join(REPO, "experiments", "round0242_nodes.py")
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "registered_value"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "max_declared_anonymous_budget_bytes"
    ]
    assert calls, "the anonymous term must resolve the registry"


def test_a_bare_registered_symbol_outside_the_regime_is_still_caught(
    tmp_path,
) -> None:
    planted = tmp_path / "experiments"
    planted.mkdir()
    (planted / "round0242_planted.py").write_text(
        "WATCHDOG_ANON_BYTES = 60 * (1 << 30)\n"
        "def poll(host):\n"
        "    return int(host['anonymous_bytes']) > WATCHDOG_ANON_BYTES\n",
        encoding="utf-8",
    )
    derived = derive_gate_constant_inventory(repo_root=str(tmp_path))
    assert derived["holds"] is False
    bare = derived["bare_registered_symbols_anywhere_in_the_release"]
    assert [row["symbol"] for row in bare] == ["WATCHDOG_ANON_BYTES"]
    #: and it is NOT in the triage scope, which is the whole point
    assert bare[0]["module"] not in derived["modules_declared"]


def test_the_discovery_inventory_control_fires() -> None:
    evidence = guard.run_discovery_inventory_control(repo_root=REPO)
    assert evidence["holds"] is True
    assert evidence["round_modules_discovered"] > 300
    assert not evidence["bare_registered_symbols_anywhere"]


def test_every_triage_entry_matches_a_real_symbol_or_is_declared_unused(
) -> None:
    """`NOT_A_BOUND` is the last hand-written list; keep it honest."""
    inventory = require_inventory_complete(repo_root=REPO)
    seen = {row["symbol"] for row in inventory["gate_constants"]["rows"]}
    #: entries that match nothing are stale, not dangerous — assert the ones
    #: R0249 added do match something, so the round's own additions are live.
    for name in (
        "REGISTRY_READERS", "REGISTRY_REGIME_MARKER", "LOCALITY_ACTION",
        "FUZZY_ACTION", "LOCALITY_SCHEMA", "KNOWN_EXTERNAL_MEMORY_MODES",
        "CONTROL_MAX_THROTTLED_RATE_RATIO", "DEFAULT_EXTERNAL_MEMORY_MODE",
    ):
        assert name in NOT_A_BOUND, name
        assert name in seen, name


# --------------------------------------------------------------------------- #
# D. the external bound
# --------------------------------------------------------------------------- #
def test_the_declaration_is_memory_high_and_cannot_kill() -> None:
    declaration = external.external_memory_limit_declaration()
    assert declaration["limit_file"] == "memory.high"
    assert declaration["can_oom_kill_the_node"] is False
    assert declaration["swap_max_bytes"] == 0
    assert declaration["mode"] == "root-scope"
    #: derived from the registry, never typed
    budget = int(registry.registered_value(
        "max_declared_anonymous_budget_bytes"))
    margin = int(registry.registered_value(
        "external_memory_limit_margin_bytes"))
    assert declaration["max_bytes"] == budget + margin
    assert declaration["derived_from"]["arithmetic"] == (
        f"{budget} + {margin} = {budget + margin}")


def test_the_default_mode_is_root_scope() -> None:
    assert external.DEFAULT_EXTERNAL_MEMORY_MODE == "root-scope"


def test_an_unplaceable_mode_refuses_and_never_downgrades() -> None:
    with pytest.raises(external.Round0249Error, match="does NOT fall back"):
        external.require_external_memory_mode(
            "root-scope",
            availability={"mode": "root-scope", "available": False,
                          "why": "sudo -n is unavailable", "base": {}},
        )
    with pytest.raises(external.Round0249Error):
        external.require_external_memory_mode("best-effort")


def test_the_fail_closed_control_fires() -> None:
    evidence = external.run_fail_closed_control()
    assert evidence["holds"] is True
    assert all(row["refused"] for row in evidence["attempts"])
    assert all(row["returned_mode"] is None for row in evidence["attempts"])


def test_the_allocator_and_escape_scripts_import_no_array_library() -> None:
    for source in (external._ALLOCATOR, external._ESCAPES):
        for banned in ("numpy", "torch", "cupy", "cuml", "cuvs"):
            assert f"import {banned}" not in source


def test_no_r0249_module_sets_memory_max_on_a_production_path() -> None:
    """`MemoryMax` appears exactly once, in the contrast arm's helper."""
    path = os.path.join(REPO, "basemap", "round0249_external.py")
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    assert source.count("MemoryMax=") == 1
    assert "_max_properties" in source
    #: and the production declaration never routes through it
    assert "MemoryHigh=" in source


def test_cgroup_self_report_reads_the_kernel_not_the_argv() -> None:
    report = external.cgroup_self_report()
    assert "memory_high" in report
    assert "memory_events" in report
    assert "times_the_high_limit_was_breached" in report


# --------------------------------------------------------------------------- #
# E. safety and scope
# --------------------------------------------------------------------------- #
def test_no_r0249_module_signals_anything() -> None:
    for name in R0249_FILES:
        path = os.path.join(REPO, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                called = getattr(node.func, "attr", None) or getattr(
                    node.func, "id", None)
                assert called not in {
                    "kill", "killpg", "terminate", "send_signal", "pkill",
                }, f"{name}:{node.lineno} calls {called}"
            if isinstance(node, ast.Attribute) and node.attr.startswith("SIG"):
                raise AssertionError(f"{name}:{node.lineno} names {node.attr}")


def test_no_subprocess_timeout_anywhere_in_r0249() -> None:
    """`subprocess.run(timeout=)` delivers SIGKILL. R0248 shipped one and the
    release's own detector caught it; the bound on these children is the
    cgroup limit plus the runner's cooperative soft deadline."""
    for name in R0249_FILES:
        path = os.path.join(REPO, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    assert keyword.arg != "timeout", f"{name}:{node.lineno}"


def test_the_accepted_residual_risks_are_named_with_their_mitigation() -> None:
    """R0249 does not claim these are closed; it says what does close them."""
    risks = guard.ACCEPTED_RESIDUAL_RISKS
    assert len(risks) == 3
    named = " ".join(row["risk"] for row in risks)
    assert "clock" in named and "gc.get_referents" in named
    assert "fabricated receipt" in named
    for row in risks:
        assert row["why_it_is_not_chased"]
        assert "independent recomputation" in row["the_mitigation_that_works"]


def test_this_round_adds_only_its_own_files_and_the_declared_edits() -> None:
    """R0249's scope over R0248's release `6359243`, plus the worktree."""
    committed = subprocess.run(
        ["git", "-C", REPO, "diff", "--name-only",
         "6359243baf2afc5b31156f05cc321f1fe0b93879", "HEAD"],
        check=False, capture_output=True, text=True,
    ).stdout.split()
    worktree = [
        line[3:] for line in subprocess.run(
            ["git", "-C", REPO, "status", "--porcelain"],
            check=False, capture_output=True, text=True,
        ).stdout.splitlines() if line
    ]
    allowed = {
        #: the declared edits to reviewed modules
        "basemap/round0245_guard.py",
        "basemap/round0246_guard.py",
        "basemap/round0247_guard.py",
        "basemap/round0247_registry.py",
        "basemap/round0248_inventory.py",
        "experiments/round0242_nodes.py",
        "tests/test_round0246_contract.py",
        "tests/test_round0247_contract.py",
        "tests/test_round0248_contract.py",
        #: R0249's own files
        "basemap/round0249_guard.py",
        "basemap/round0249_external.py",
        "experiments/round0249_nodes.py",
        "experiments/prepare_round0249_queue.py",
        "tests/test_round0249_contract.py",
        "tests/test_round0249_cpu_smoke.py",
    }
    changed = set(committed + worktree)
    assert changed <= allowed, sorted(changed - allowed)
