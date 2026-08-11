"""R0253 contract tests.

Every guard this round adds ships a positive control that plants the defect and
proves the guard catches it.  The three guards are:

1. the stop hook is installed as the FIRST statement of every registered
   ladder-train node entry (`entry_install_audit`),
2. the write loop contains an abort read (`assert_write_loop_polls`),
3. every hooked module is actually hooked at the moment work starts
   (`assert_stop_hooks_installed`).

and each of `test_*_positive_control_*` below plants exactly the shape the guard
is supposed to reject.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
import sys
import textwrap
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap import artifact_identity, panel_v2
from basemap.round0252_stoppability import (
    Round0252CooperativeAbort,
    measure_stop_latency,
)
from basemap.round0253_coverage import CoverageLedger, coverage_summary
from basemap.round0253_stop_hooks import (
    BINDING_CALLS,
    HOOKED_MODULES,
    REGISTERED_ENTRIES,
    Round0253CooperativeAbort,
    Round0253Error,
    RunnerAbortPoll,
    assert_every_entry_installs,
    assert_stop_hooks_installed,
    entry_install_audit,
    install_stop_hooks,
    reachability_audit,
    stop_hooks_state,
)
from basemap.round0253_write_path import (
    ABORT_POLL_SITES,
    Round0253WriteError,
    assert_write_loop_polls,
    write_loop_polls,
    write_sized_file,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSHOP_REPORT = "/home/enjalot/code/workshop/rounds/report.py"


@pytest.fixture(autouse=True)
def _unhooked():
    """Every test starts and ends with both modules unhooked."""
    artifact_identity.set_abort_poll(None)
    panel_v2.set_abort_poll(None)
    yield
    artifact_identity.set_abort_poll(None)
    panel_v2.set_abort_poll(None)


# --------------------------------------------------------------------------- #
# A. the install
# --------------------------------------------------------------------------- #


def test_the_release_starts_unhooked():
    """The default is None in both modules, which is why an install is needed."""
    state = stop_hooks_state()
    assert state["modules_not_installed"] == sorted(HOOKED_MODULES)
    assert not state["every_hooked_module_is_installed"]


def test_every_registered_ladder_entry_installs_the_hook_first():
    audit = assert_every_entry_installs()
    assert audit["entries_audited"] == len(REGISTERED_ENTRIES)
    assert audit["entries_missing_the_install"] == []
    # The two the review named by name, explicitly.
    named = {f"{row['module']}.{row['function']}": row for row in audit["entries"]}
    assert named["experiments.round0250_nodes.run_train"]["installs_the_hook_first"]
    assert named["experiments.round0218_nodes.run_panel"]["installs_the_hook_first"]
    assert "score_panel" in named["experiments.round0218_nodes.run_panel"][
        "binding_calls_in_this_entry"
    ]


def test_positive_control_an_entry_without_the_install_fails_the_audit(tmp_path):
    """Plant the defect: a node entry that does not install the hook."""
    planted = tmp_path / "planted_nodes.py"
    planted.write_text(textwrap.dedent(
        '''
        from basemap.artifact_identity import expected_input_signature

        def run_planted(active, job):
            """A ladder-train-shaped entry with the install removed."""
            return expected_input_signature(job["path"])
        '''
    ))
    sys.path.insert(0, str(tmp_path))
    try:
        audit = entry_install_audit([("planted_nodes", "run_planted")])
        assert audit["entries_missing_the_install"] == ["planted_nodes.run_planted"]
        assert not audit["every_registered_entry_installs_the_hook_first"]
        with pytest.raises(Round0253Error):
            assert_every_entry_installs([("planted_nodes", "run_planted")])
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("planted_nodes", None)


def test_positive_control_an_install_after_the_binding_call_fails_the_audit(tmp_path):
    """The R0252 shape exactly: the hook goes in, but only after the long call."""
    planted = tmp_path / "late_nodes.py"
    planted.write_text(textwrap.dedent(
        '''
        from basemap.artifact_identity import expected_input_signature
        from basemap.round0253_stop_hooks import install_stop_hooks

        def run_late(active, job):
            signature = expected_input_signature(job["path"])
            install_stop_hooks(label="too late")
            return signature
        '''
    ))
    sys.path.insert(0, str(tmp_path))
    try:
        audit = entry_install_audit([("late_nodes", "run_late")])
        assert audit["entries_missing_the_install"] == ["late_nodes.run_late"]
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("late_nodes", None)


@pytest.mark.parametrize("module_name,function_name", REGISTERED_ENTRIES)
def test_the_real_entry_path_installs_the_hook(module_name, function_name):
    """Resolve the handler exactly as the runner does, call it, check the hooks.

    `rounds/runner.py`'s `--node` path is three lines:
    `importlib.import_module(job["handler_module"])`, `getattr(module,
    job["handler_callable"])`, `handler(active, job)`.  This reproduces those
    three lines against a deliberately empty job, so the handler raises almost
    immediately -- but AFTER its first statement.  If the install is absent, or
    stops being the first statement, both modules are still `None` when the
    exception arrives and this fails.
    """
    assert stop_hooks_state()["modules_not_installed"] == sorted(HOOKED_MODULES)
    module = importlib.import_module(module_name)
    handler = getattr(module, function_name)
    with pytest.raises(BaseException):
        handler({}, {})
    state = stop_hooks_state()
    assert state["every_hooked_module_is_installed"], (
        f"{module_name}.{function_name} raised with "
        f"{state['modules_not_installed']} unhooked"
    )


def test_positive_control_the_real_entry_path_check_fails_without_the_install(tmp_path):
    """The same three lines against a handler with no install: hooks stay None."""
    planted = tmp_path / "unhooked_nodes.py"
    planted.write_text(textwrap.dedent(
        '''
        def run_unhooked(active, job):
            raise RuntimeError("no install here")
        '''
    ))
    sys.path.insert(0, str(tmp_path))
    try:
        module = importlib.import_module("unhooked_nodes")
        handler = getattr(module, "run_unhooked")
        with pytest.raises(RuntimeError):
            handler({}, {})
        assert not stop_hooks_state()["every_hooked_module_is_installed"]
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("unhooked_nodes", None)


def test_the_installed_hook_raises_out_of_a_real_integrity_hash(tmp_path, monkeypatch):
    """Behavioural, not structural: the installed hook stops a real hash."""
    target = tmp_path / "sample.bin"
    write_sized_file(str(target), 48 << 20, seed=7, poll=None, digest=False)
    flag = tmp_path / "node.abort"
    monkeypatch.setenv("ROUNDRUN_ABORT_FLAG", str(flag))
    install_stop_hooks(label="test")
    # No flag: the hash completes and yields the same digest as an unhooked one.
    hooked = artifact_identity.expected_input_signature(str(target))
    artifact_identity.set_abort_poll(None)
    plain = artifact_identity.expected_input_signature(str(target))
    assert hooked == plain
    # Flag present: the hash raises and yields NO digest.
    install_stop_hooks(label="test")
    flag.write_text('{"reason":"test"}')
    with pytest.raises(Round0253CooperativeAbort):
        artifact_identity.expected_input_signature(str(target))


def test_the_guard_raises_when_a_module_is_left_unhooked():
    install_stop_hooks(label="test")
    assert_stop_hooks_installed(label="test")
    panel_v2.set_abort_poll(None)
    with pytest.raises(Round0253Error) as error:
        assert_stop_hooks_installed(label="test")
    assert "basemap.panel_v2" in str(error.value)


def test_the_binding_entry_census_is_published_as_a_lower_bound():
    census = reachability_audit(os.path.join(REPO_ROOT, "experiments"))
    assert census["entries_that_can_bind"] > census["entries_with_the_install"]
    assert census["entries_with_the_install"] >= 16
    assert 0.0 < census["coverage_fraction"] < 1.0
    assert "lower bound" in census["what_this_is_not"]


def test_a_runner_abort_poll_allocates_no_digest_and_names_no_signal():
    poll = RunnerAbortPoll(label="test")
    poll("somewhere")          # no flag in the environment: a no-op
    receipt = poll.receipt()
    assert receipt["reads"] == 1
    assert receipt["signal_delivered"] is False


# --------------------------------------------------------------------------- #
# B. the write path
# --------------------------------------------------------------------------- #


def test_the_write_loop_polls():
    guard = assert_write_loop_polls()
    assert guard["write_loops_found"] >= 1
    assert guard["poll_calls_inside_a_write_loop"] >= 1


def test_positive_control_a_write_loop_without_a_poll_is_caught(tmp_path):
    """Plant R0252's `_write_sized_file` shape and prove the guard rejects it."""
    source = textwrap.dedent(
        '''
        def write_sized_file(path, total_bytes, *, seed, poll=None, digest=True):
            fd = 0
            written = 0
            while written < total_bytes:
                written += os.write(fd, b"x")
        '''
    )
    tree = ast.parse(source)
    function = tree.body[0]
    loops = [n for n in ast.walk(function) if isinstance(n, ast.While)]
    write_loops = [
        loop for loop in loops
        if any(isinstance(i, ast.Call) and isinstance(i.func, ast.Attribute)
               and i.func.attr == "write" for i in ast.walk(loop))
    ]
    polled = [
        n for loop in write_loops for stmt in loop.body for n in ast.walk(stmt)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "poll"
    ]
    assert write_loops and not polled, (
        "the planted control must contain a write loop with no poll, which is "
        "the exact shape review-0252-01 §H measured at 144.50039366001147 s"
    )


def test_the_two_write_arms_produce_identical_bytes(tmp_path):
    size = 200 << 20
    unpolled_path = tmp_path / "unpolled.bin"
    polled_path = tmp_path / "polled.bin"
    reads: list[str] = []
    unpolled = write_sized_file(str(unpolled_path), size, seed=11, poll=None)
    polled = write_sized_file(str(polled_path), size, seed=11, poll=reads.append)
    assert unpolled["content_sha256"] == polled["content_sha256"]
    assert unpolled["bytes"] == polled["bytes"] == size
    assert unpolled["abort_reads_inside_the_write"] == 0
    assert polled["abort_reads_inside_the_write"] >= 3
    assert set(reads) <= set(ABORT_POLL_SITES)
    # And the file really is on the device, evidenced by the inode, not a literal.
    assert unpolled["fully_allocated"] and polled["fully_allocated"]
    digest = hashlib.sha256(unpolled_path.read_bytes()).hexdigest()
    assert digest == unpolled["content_sha256"]


def test_a_stop_planted_mid_write_is_observed_and_leaves_no_file(tmp_path):
    target = tmp_path / "aborted.bin"

    def run(poll):
        try:
            write_sized_file(str(target), 3 << 30, seed=13, poll=poll, digest=False)
        finally:
            if target.exists():
                target.unlink()

    outcome = measure_stop_latency(
        label="test write", flag_path=str(tmp_path / "stop.abort"),
        delay_s=0.25, run=run,
    )
    assert outcome["the_work_stopped_cooperatively"] is True
    assert outcome["stop_latency_s"] is not None
    assert not target.exists()


def test_an_unpolled_write_cannot_be_stopped_at_all(tmp_path):
    """The control that gives the fix its meaning: no poll, no observation."""
    target = tmp_path / "unstoppable.bin"

    def run(poll):
        # `poll` is deliberately not passed to the writer: R0252's shape.
        write_sized_file(str(target), 200 << 20, seed=17, poll=None, digest=False)

    outcome = measure_stop_latency(
        label="test unpolled write", flag_path=str(tmp_path / "stop2.abort"),
        delay_s=0.05, run=run,
    )
    assert outcome["the_work_stopped_cooperatively"] is False
    assert outcome["the_work_ran_to_completion_instead"] is True
    assert outcome["stop_latency_s"] is None


def test_the_write_path_refuses_a_non_callable_poll(tmp_path):
    with pytest.raises(Round0253WriteError):
        write_sized_file(str(tmp_path / "x.bin"), 1 << 20, seed=1, poll=object())


# --------------------------------------------------------------------------- #
# C. coverage
# --------------------------------------------------------------------------- #


def test_the_ledger_emits_observed_span_s_as_a_union_not_a_sum():
    clock = {"t": 0.0}
    ledger = CoverageLedger(node="test", clock=lambda: clock["t"])
    first = ledger.window("a")
    second = ledger.window("b")
    clock["t"] = 1.0
    first.observe()
    clock["t"] = 2.0
    second.observe()
    clock["t"] = 5.0
    first.observe()
    clock["t"] = 6.0
    second.observe()
    # windows are [1, 5] and [2, 6]: the sum is 8, the union is 5.
    assert ledger.observed_span_s() == pytest.approx(5.0)
    receipt = ledger.receipt(node_wall_s=10.0)
    assert receipt["observed_span_s"] == pytest.approx(5.0)
    assert receipt["uncovered_wall_s"] == pytest.approx(5.0)
    assert receipt["covered_fraction"] == pytest.approx(0.5)


def test_a_window_with_one_read_covers_nothing():
    clock = {"t": 0.0}
    ledger = CoverageLedger(node="test", clock=lambda: clock["t"])
    window = ledger.window("single")
    clock["t"] = 3.0
    window.observe()
    assert window.span_s == 0.0
    assert ledger.observed_span_s() == 0.0
    assert window.receipt()["covers_nothing_because_it_saw_fewer_than_two_reads"]


def test_a_wrapped_poll_that_raises_still_contributes_its_observation():
    clock = {"t": 0.0}
    ledger = CoverageLedger(node="test", clock=lambda: clock["t"])
    window = ledger.window("raising")

    def poll(where):
        if clock["t"] >= 4.0:
            raise Round0252CooperativeAbort("stop")

    wrapped = window.wrap(poll)
    wrapped("a")
    clock["t"] = 4.0
    with pytest.raises(Round0252CooperativeAbort):
        wrapped("b")
    assert window.observations == 2
    assert window.span_s == pytest.approx(4.0)


def test_coverage_summary_folds_nodes():
    ledger = CoverageLedger(node="n")
    summary = coverage_summary([ledger.receipt(node_wall_s=2.0)])
    assert summary["nodes"] == 1
    assert summary["total_node_wall_s"] == pytest.approx(2.0)


@pytest.mark.skipif(not os.path.exists(WORKSHOP_REPORT),
                    reason="roundreport not present on this machine")
def test_roundreport_reads_observed_span_s_and_reports_unknown_without_it(tmp_path):
    """The point of §C: the tool already looks, and until now found nothing.

    This is both the check and its own positive control -- the second half plants
    R0252's state (an artifact with no covered span) and requires `roundreport`
    to report the node as uninstrumented.
    """
    spec = importlib.util.spec_from_file_location("r0253_report", WORKSHOP_REPORT)
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    terminal = {"nodes": [{"node": "writepath_0253", "wall_s": 1000.0}]}

    without = artifacts / "writepath_0253-polled-write-path.json"
    without.write_text(json.dumps({"gap_report": {"widest_gap_s": 0.004}}))
    rows = report.collect_coverage(str(artifacts), terminal)
    assert rows and rows[0]["covered_s"] is None

    without.write_text(json.dumps({
        "gap_report": {"widest_gap_s": 0.004},
        "observed_span_s": 940.0,
        "node_wall_s": 1000.0,
    }))
    rows = report.collect_coverage(str(artifacts), terminal)
    assert rows[0]["covered_s"] == pytest.approx(940.0)
    assert rows[0]["uncovered_s"] == pytest.approx(60.0)


# --------------------------------------------------------------------------- #
# process rules
# --------------------------------------------------------------------------- #


R0253_FILES = (
    "basemap/round0253_stop_hooks.py",
    "basemap/round0253_coverage.py",
    "basemap/round0253_write_path.py",
    "experiments/round0253_nodes.py",
    "experiments/prepare_round0253_queue.py",
)


def test_no_hidden_sigkill_in_any_round0253_file():
    """`subprocess.run(..., timeout=N)` is `Popen.kill()`; the AST guard is blind."""
    for name in R0253_FILES:
        path = os.path.join(REPO_ROOT, name)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
        offenders = [
            int(node.lineno)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and any(kw.arg == "timeout" for kw in node.keywords)
        ]
        assert not offenders, f"{name} passes timeout= at line(s) {offenders}"


def test_positive_control_a_planted_timeout_call_is_caught(tmp_path):
    """Plant the hidden SIGKILL and prove the check sees it — and that a comment
    mentioning `timeout=` does not trip it, which a textual grep would."""
    planted = tmp_path / "planted.py"
    planted.write_text(textwrap.dedent(
        '''
        import subprocess
        # deliberately mentions timeout= in prose only
        def safe():
            return subprocess.run(["true"], check=False)
        def unsafe():
            return subprocess.run(["true"], check=False, timeout=5)
        '''
    ))
    tree = ast.parse(planted.read_text())
    offenders = [
        int(node.lineno) for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and any(kw.arg == "timeout" for kw in node.keywords)
    ]
    assert len(offenders) == 1


def test_the_round0253_nodes_never_write_the_runners_abort_flag():
    path = os.path.join(REPO_ROOT, "experiments/round0253_nodes.py")
    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()
    assert "ROUNDRUN_ABORT_FLAG" in source  # the install probe restores it
    assert ".abort" in source
    # every flag this round writes lives under the node's own scratch root
    assert 'os.path.join(scratch, "write-stop.abort")' in source
    assert 'os.path.join(scratch, "install-probe.abort")' in source


def test_binding_calls_are_the_ones_r0252_instrumented():
    assert "score_panel" in BINDING_CALLS
    assert "expected_input_signature" in BINDING_CALLS
    assert "ordered_array_sha256" in BINDING_CALLS


def test_the_declared_write_sites_all_appear_in_the_writer():
    guard = write_loop_polls()
    path = os.path.join(REPO_ROOT, "basemap/round0253_write_path.py")
    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()
    for site in guard["declared_sites"]:
        assert site in source
