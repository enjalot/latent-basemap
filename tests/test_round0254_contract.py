"""R0254 contract — the guards, and controls that plant the defect they catch.

Every guard this round adds ships a positive control that (a) plants the defect
in a **real importable module** or a real source string, (b) passes it to the
**shipped** function rather than to a copy of its logic, and (c) is itself
verified to pass when the shipped path is weakened — so a control that could not
fail is a test failure.

(c) is the standard review-0253-01 §I asked for.  R0253 claimed four positive
controls and one of them, `test_positive_control_a_write_loop_without_a_poll_is_
caught`, re-implemented the guard's AST walk in the test body and would have
passed had `write_loop_polls` returned `True` unconditionally.  The
`memory.high` work merged into the runner on 2026-08-11 did this properly for
all four of its controls; this file matches that standard.
"""
from __future__ import annotations

import ast
import os
import sys
import textwrap

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from basemap import round0254_dispatch as dispatch
from basemap import round0254_writeback as writeback
from basemap.round0253_stop_hooks import install_stop_hooks
from basemap.round0253_write_path import (
    write_loop_polls as r0253_write_loop_polls,
)
from experiments.round0254_nodes import (
    PLANTED_DEFECTS,
    PLANTED_HONEST,
    WRITEBACK_ACTION,
    DISPATCH_ACTION,
)


# --------------------------------------------------------------------------- #
# helpers: plant a real module, and reproduce R0253's weak predicate
# --------------------------------------------------------------------------- #


#: `roundreport`'s own module, read from the workshop checkout by AST so the
#: test binds the tool's real constants without importing it (it is not on the
#: release's import path).
ROUNDREPORT_SOURCE = "/home/enjalot/code/workshop/rounds/report.py"


def _report_constant(name: str) -> tuple:
    if not os.path.exists(ROUNDREPORT_SOURCE):  # pragma: no cover
        pytest.skip(f"roundreport source not present at {ROUNDREPORT_SOURCE}")
    tree = ast.parse(open(ROUNDREPORT_SOURCE, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return tuple(ast.literal_eval(node.value))
    raise AssertionError(f"{name} is no longer defined in {ROUNDREPORT_SOURCE}")


def _plant(tmp_path, name: str, source: str) -> str:
    """Write a real, importable module and return its dotted name."""
    package = tmp_path / "planted_pkg"
    package.mkdir(exist_ok=True)
    (package / "__init__.py").write_text("")
    (package / f"{name}.py").write_text(source)
    root = str(tmp_path)
    if root not in sys.path:
        sys.path.insert(0, root)
    return f"planted_pkg.{name}"


def _unplant(tmp_path, module_name: str) -> None:
    sys.modules.pop(module_name, None)
    sys.modules.pop("planted_pkg", None)
    root = str(tmp_path)
    if root in sys.path:
        sys.path.remove(root)


def _r0253_calls_install(statement) -> bool:
    """R0253's `_calls_install`, verbatim, as the weakened predicate.

    `round0253_stop_hooks.py:287--300`: walk the first statement, match on
    `Name.id` or `Attribute.attr`. review-0253-01 §A.3 got three shapes past it.
    This exists so every planted defect below can be shown to be a defect the
    OLD code accepted — which is what makes the new refusals evidence rather
    than tautology.
    """
    if statement is None:
        return False
    for node in ast.walk(statement):
        if isinstance(node, ast.Call):
            func = node.func
            name = (
                func.id if isinstance(func, ast.Name)
                else func.attr if isinstance(func, ast.Attribute)
                else None
            )
            if name == "install_stop_hooks":
                return True
    return False


def _first_statement_of(source: str, function: str):
    tree = ast.parse(source)
    node = next(
        item for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == function
    )
    body = list(node.body)
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    return body[0] if body else None


# --------------------------------------------------------------------------- #
# §C. the auditor, and the five planted defects
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name,source", list(PLANTED_DEFECTS))
def test_positive_control_a_planted_install_defect_is_refused(tmp_path, name, source):
    """Each planted shape must fail the SHIPPED auditor, with a reason."""
    module_name = _plant(tmp_path, name, source)
    try:
        verdict = dispatch.install_effectiveness(module_name, "run_job")
    finally:
        _unplant(tmp_path, module_name)
    assert verdict["the_install_is_effective"] is False, (
        f"{name}: the shipped auditor accepted a planted defect"
    )
    assert verdict["why_not"], f"{name}: refused without naming a reason"


@pytest.mark.parametrize("name,source", list(PLANTED_DEFECTS))
def test_the_control_discriminates_because_r0253_s_auditor_accepted_it(name, source):
    """The control is only evidence if the weakened predicate accepts the defect.

    Three of these five (`dead_branch`, `module_level_shadow`, `deferred_lambda`)
    are review-0253-01 §A.3's shapes verbatim and it reported all three passing.
    This asserts that directly against R0253's own logic, so the new refusal
    above is a change in behaviour rather than a check that never mattered.
    """
    statement = _first_statement_of(source, "run_job")
    assert _r0253_calls_install(statement) is True, (
        f"{name}: R0253's predicate already refused this, so the new refusal "
        "proves nothing"
    )


def test_the_auditor_still_accepts_an_honest_install(tmp_path):
    """Without this the five refusals above would be a broken audit, not a fix."""
    name, source = PLANTED_HONEST
    module_name = _plant(tmp_path, name, source)
    try:
        verdict = dispatch.install_effectiveness(module_name, "run_job")
    finally:
        _unplant(tmp_path, module_name)
    assert verdict["the_install_is_effective"] is True, verdict["why_not"]


def test_the_node_s_own_planted_defect_harness_runs_end_to_end(tmp_path):
    """The node helper itself, in one package directory, as the node runs it.

    Regression: the first launch of `dispatch_0254` died with
    `ModuleNotFoundError: round0254_planted.module_level_shadow`. `importlib`'s
    `FileFinder` caches a package directory's listing on first import and
    revalidates it only on an mtime change with 1 s granularity, so writing the
    second planted module *after* importing the first is invisible on a fast
    disk. The per-test `_plant` helper above uses a fresh directory each time
    and could not see it. This calls the node's own helper.
    """
    from experiments.round0254_nodes import _plant_and_audit

    report = _plant_and_audit(str(tmp_path))
    assert report["every_planted_defect_was_caught"] is True, report["controls"]
    assert report["the_honest_install_still_passes"] is True
    assert report["defects_caught_by_the_shipped_auditor"] == len(PLANTED_DEFECTS)


def test_the_auditor_refuses_an_install_imported_from_the_wrong_module(tmp_path):
    module_name = _plant(
        tmp_path, "wrong_source",
        "from basemap.round0253_coverage import CoverageLedger as install_stop_hooks\n"
        "\n\n"
        "def run_job(active, job):\n"
        "    install_stop_hooks(node='planted')\n"
        "    return None\n",
    )
    try:
        verdict = dispatch.install_effectiveness(module_name, "run_job")
    finally:
        _unplant(tmp_path, module_name)
    assert verdict["the_install_is_effective"] is False
    assert verdict["resolves_to_the_release_function"] is False


def test_the_auditor_refuses_an_install_after_the_binding_call(tmp_path):
    """R0252's exact shape: the install runs after the 153.6 GB hash."""
    module_name = _plant(
        tmp_path, "install_after_the_hash",
        "from basemap.artifact_identity import expected_input_signature\n"
        "from basemap.round0253_stop_hooks import install_stop_hooks\n"
        "\n\n"
        "def run_job(active, job):\n"
        "    signature = expected_input_signature(job['path'])\n"
        "    install_stop_hooks(label='too late')\n"
        "    return signature\n",
    )
    try:
        verdict = dispatch.install_effectiveness(module_name, "run_job")
    finally:
        _unplant(tmp_path, module_name)
    assert verdict["the_install_is_effective"] is False
    assert verdict["install_is_the_first_statement_unconditionally"] is False


# --------------------------------------------------------------------------- #
# §B. the entry list is derived, not nominated
# --------------------------------------------------------------------------- #


def test_the_dispatch_census_reads_real_queue_manifests():
    census = dispatch.dispatch_census()
    assert census["queue_manifests_scanned"] > 0
    assert census["distinct_dispatched_handlers"] > 0
    # The finding this round exists to fix: the runner resolves `run_job`.
    assert census["dispatched_callables_by_name"].get("run_job", 0) > 0


def test_every_derived_entry_installs_effectively():
    guard = dispatch.assert_derived_entries_install()
    assert guard["audit"]["every_entry_installs_effectively"] is True
    assert guard["audit"]["entries_audited"] == guard["derived"]["entry_count"]


def test_the_derived_list_contains_the_callable_the_runner_resolves():
    """`run_job` is what `rounds/runner.py` getattrs. R0253 excluded it by name."""
    derived = dispatch.derived_entries()
    entries = {f"{row['module']}.{row['function']}" for row in derived["entries"]}
    for module in dispatch.SCOPE_MODULES:
        assert f"{module}.run_job" in entries, f"{module} has no run_job entry"


def test_the_derived_list_grows_when_a_dispatch_table_grows(tmp_path):
    """The completeness check R0253's hand-written list could not have.

    A new action in a `run_job` dispatch table must enlarge the derived list by
    itself. Planted as a real module so the derivation runs on real source.
    """
    module_name = _plant(
        tmp_path, "growing_module",
        "from basemap.round0253_stop_hooks import install_stop_hooks\n"
        "\n\n"
        "def run_alpha(active, job):\n"
        "    install_stop_hooks(label='alpha')\n"
        "    return None\n"
        "\n\n"
        "def run_beta(active, job):\n"
        "    install_stop_hooks(label='beta')\n"
        "    return None\n"
        "\n\n"
        "def run_job(active, job):\n"
        "    install_stop_hooks(label='job')\n"
        "    if job['action'] == 'a':\n"
        "        return run_alpha(active, job)\n"
        "    return run_beta(active, job)\n",
    )
    try:
        census = {"handlers": [
            {"module": module_name, "callable": "run_job", "queue_jobs": 1}
        ]}
        derived = dispatch.derived_entries((module_name,), census)
        names = {row["function"] for row in derived["entries"]}
    finally:
        _unplant(tmp_path, module_name)
    assert names == {"run_job", "run_alpha", "run_beta"}


def test_positive_control_an_undispatched_delegate_without_the_install_fails(tmp_path):
    """Plant R0253's §A.4 defect: a delegate the dispatch table reaches, uninstalled."""
    module_name = _plant(
        tmp_path, "uninstalled_delegate",
        "from basemap.round0253_stop_hooks import install_stop_hooks\n"
        "\n\n"
        "def run_orphan(active, job):\n"
        "    return None\n"
        "\n\n"
        "def run_job(active, job):\n"
        "    install_stop_hooks(label='job')\n"
        "    return run_orphan(active, job)\n",
    )
    try:
        census = {"handlers": [
            {"module": module_name, "callable": "run_job", "queue_jobs": 1}
        ]}
        with pytest.raises(dispatch.Round0254DispatchError) as error:
            dispatch.assert_derived_entries_install((module_name,), census)
    finally:
        _unplant(tmp_path, module_name)
    assert "run_orphan" in str(error.value)


# --------------------------------------------------------------------------- #
# §D. installation is not enforcement
# --------------------------------------------------------------------------- #


def test_the_gate_census_counts_install_and_gate_separately(tmp_path):
    module_name = _plant(
        tmp_path, "gate_shapes",
        "from basemap.round0253_stop_hooks import install_stop_hooks\n"
        "from experiments.round0251_nodes import _node_gate\n"
        "\n\n"
        "def run_with_gate(active, job):\n"
        "    install_stop_hooks(label='g')\n"
        "    gate = _node_gate('planted', training_performed=False)\n"
        "    return gate\n"
        "\n\n"
        "def run_without_gate(active, job):\n"
        "    install_stop_hooks(label='n')\n"
        "    return None\n",
    )
    try:
        census = dispatch.gate_census([
            (module_name, "run_with_gate"), (module_name, "run_without_gate")
        ])
    finally:
        _unplant(tmp_path, module_name)
    assert census["entries_that_install_effectively"] == 2
    assert census["entries_that_construct_a_gate"] == 1
    assert census["entries_that_both_install_and_gate"] == 1
    assert census["entries_that_install_but_construct_no_gate"] == [
        f"{module_name}.run_without_gate"
    ]


def test_the_gate_census_follows_a_module_local_helper(tmp_path):
    module_name = _plant(
        tmp_path, "indirect_gate",
        "from basemap.round0253_stop_hooks import install_stop_hooks\n"
        "from experiments.round0251_nodes import _node_gate\n"
        "\n\n"
        "def _make_gate():\n"
        "    return _node_gate('planted', training_performed=False)\n"
        "\n\n"
        "def run_indirect(active, job):\n"
        "    install_stop_hooks(label='i')\n"
        "    return _make_gate()\n",
    )
    try:
        census = dispatch.gate_census([(module_name, "run_indirect")])
    finally:
        _unplant(tmp_path, module_name)
    assert census["entries_that_construct_a_gate"] == 1
    assert census["entries"][0]["gate_constructed_in"] == "_make_gate"


def test_the_live_install_and_gate_census_reports_both_numbers():
    derived = dispatch.derived_entries()
    census = dispatch.gate_census(dispatch.entry_tuples(derived))
    assert census["entries_audited"] == derived["entry_count"]
    assert census["entries_that_install_effectively"] == derived["entry_count"]
    # Not asserted equal: that is the finding, whatever it turns out to be.
    assert census["entries_that_both_install_and_gate"] <= census["entries_audited"]


# --------------------------------------------------------------------------- #
# §A. the write-stall probe
# --------------------------------------------------------------------------- #


def test_the_write_unit_is_not_a_variable():
    """review-0253-01 §D falsified the smaller-unit remedy; it is not retried."""
    assert writeback.WRITE_BLOCK_BYTES == 64 << 20
    for arm in writeback.SHIPPED_ARMS:
        assert writeback.ARMS[arm].receipt()["write_block_bytes"] == 64 << 20


def test_the_arms_vary_only_the_flush_discipline():
    cadences = {
        writeback.ARMS[arm].flush_every_bytes for arm in writeback.SHIPPED_ARMS
    }
    assert cadences == {
        writeback.CADENCE_2GIB, writeback.CADENCE_256MIB,
        writeback.CADENCE_32MIB, None,
    }
    assert writeback.ARMS[writeback.ARM_O_DIRECT].o_direct is True
    assert writeback.ARMS[writeback.ARM_SYNC_FILE_RANGE].sync_file_range is True
    assert writeback.ARMS[writeback.ARM_UNPOLLED_CONTROL].polls is False


def test_the_schedule_rotates_so_no_arm_owns_a_device_state():
    width = len(writeback.SHIPPED_ARMS)
    schedule = writeback.arm_schedule(writeback.SHIPPED_ARMS, 5)
    assert len(schedule) == 5 * width
    firsts = [schedule[index * width][1] for index in range(5)]
    assert len(set(firsts)) > 1, "every repetition started with the same arm"
    for arm in writeback.SHIPPED_ARMS:
        assert sum(1 for _r, name in schedule if name == arm) == 5


def test_the_control_arm_site_string_is_what_roundreport_ranks_as_a_control():
    """Read from the tool's own source, so a rename there fails here."""
    hints = _report_constant("_CONTROL_HINTS")
    assert any(hint in writeback.ARM_UNPOLLED_CONTROL for hint in hints)
    for arm in writeback.SHIPPED_ARMS:
        assert not any(hint in arm for hint in hints), (
            f"shipped arm {arm} would be ranked as a control by roundreport"
        )


def test_a_real_write_records_every_syscall_interval(tmp_path):
    """The instrument, at a size a test can afford: 192 MiB, three blocks."""
    path = str(tmp_path / "small.bin")
    reads: list[str] = []
    created = writeback.write_arm(
        path, arm=writeback.ARM_FSYNC_32MIB, total_bytes=3 * writeback.WRITE_BLOCK_BYTES,
        seed=7, poll=reads.append,
    )
    assert created["bytes"] == 3 * writeback.WRITE_BLOCK_BYTES
    assert created["blocks_written"] == 3
    assert created["fully_allocated"] is True
    assert created["allocated_blocks_512b"] * 512 >= created["bytes"]
    assert created["per_write_interval"]["count"] >= 3
    assert created["final_flush_s"] > 0.0
    assert created["the_widest_single_syscall_s"] > 0.0
    assert writeback.ABORT_POLL_SITE_BLOCK in reads
    assert writeback.ABORT_POLL_SITE_FINAL in reads


def test_o_direct_and_sync_file_range_write_the_same_bytes(tmp_path):
    size = 2 * writeback.WRITE_BLOCK_BYTES
    digests = {}
    for arm in (writeback.ARM_O_DIRECT, writeback.ARM_SYNC_FILE_RANGE,
                writeback.ARM_FSYNC_2GIB):
        path = str(tmp_path / f"{arm}.bin")
        created = writeback.write_arm(path, arm=arm, total_bytes=size, seed=11,
                                      poll=lambda _where: None)
        digests[arm] = (created["block_sha256"], created["blocks_written"],
                        created["bytes"])
        os.unlink(path)
    assert len(set(digests.values())) == 1, digests


def test_the_unpolled_control_issues_no_abort_read(tmp_path):
    path = str(tmp_path / "control.bin")
    reads: list[str] = []
    created = writeback.write_arm(
        path, arm=writeback.ARM_UNPOLLED_CONTROL,
        total_bytes=writeback.WRITE_BLOCK_BYTES, seed=3, poll=reads.append,
    )
    assert created["abort_reads_inside_the_write"] == 0
    assert reads == []


def test_positive_control_a_write_loop_without_a_poll_is_caught_by_the_guard():
    """Planted source, run through the SHIPPED guard — not through a copy of it."""
    planted = textwrap.dedent(
        """
        def write_arm(path, *, arm, total_bytes, seed, poll=None, block=None):
            fd = os.open(path, os.O_WRONLY)
            written = 0
            while written < total_bytes:
                written += os.write(fd, b"x")
            os.close(fd)
            return {}
        """
    )
    guard = writeback.write_loop_polls(planted)
    assert guard["the_write_loop_polls"] is False
    assert guard["checked"] == "planted source"
    with pytest.raises(writeback.Round0254WritebackError):
        writeback.assert_write_loop_polls(planted)


def test_the_guard_accepts_the_shipped_writer():
    guard = writeback.assert_write_loop_polls()
    assert guard["the_write_loop_polls"] is True
    assert guard["checked"] == "the shipped writer"
    assert guard["write_loops_found"] >= 1


def test_r0253_s_write_guard_can_now_be_pointed_at_planted_source():
    """review-0253-01 §I: the fix is a `source=` argument, not a new guard."""
    planted = textwrap.dedent(
        """
        def write_sized_file(path, total_bytes, *, seed, poll=None, digest=True):
            fd = os.open(path, os.O_WRONLY)
            written = 0
            while written < total_bytes:
                written += os.write(fd, b"x")
            os.close(fd)
            return {}
        """
    )
    assert r0253_write_loop_polls(planted)["the_write_loop_polls"] is False
    assert r0253_write_loop_polls()["the_write_loop_polls"] is True


def test_the_stall_verdict_ranks_arms_on_the_maximum_not_the_median():
    runs = [
        {"arm": "fsync_2gib", "final_flush_s": 0.1,
         "write_wall_s": 40.0,
         "per_write_interval": {"max_s": 2.0, "median_s": 0.02, "p99_s": 1.5,
                                "count": 700, "intervals_over_the_ceiling": 0}},
        {"arm": "fsync_2gib", "final_flush_s": 0.2,
         "write_wall_s": 41.0,
         "per_write_interval": {"max_s": 1.0, "median_s": 0.02, "p99_s": 0.8,
                                "count": 700, "intervals_over_the_ceiling": 0}},
        # A tiny median but a worse maximum: this is exactly the 8 MiB result
        # review-0253-01 §D reported, and the verdict must not prefer it.
        {"arm": "o_direct", "final_flush_s": 0.01,
         "write_wall_s": 30.0,
         "per_write_interval": {"max_s": 3.0, "median_s": 0.001, "p99_s": 0.002,
                                "count": 700, "intervals_over_the_ceiling": 1}},
    ]
    verdict = writeback.stall_verdict(runs)
    assert verdict["by_arm"]["fsync_2gib"]["worst_write_s"] == 2.0
    assert verdict["by_arm"]["o_direct"]["worst_write_s"] == 3.0
    assert verdict["best_shipped_arm"] == "fsync_2gib"
    assert verdict["worst_shipped_arm"] == "o_direct"


def test_the_verdict_says_plainly_when_nothing_bounds_the_stall():
    ceiling = writeback.registered_ceiling_s()
    runs = [
        {"arm": arm, "final_flush_s": 0.1, "write_wall_s": 40.0,
         "per_write_interval": {"max_s": ceiling * 1.5, "median_s": 0.02,
                                "p99_s": 1.0, "count": 700,
                                "intervals_over_the_ceiling": 1}}
        for arm in writeback.SHIPPED_ARMS
    ]
    verdict = writeback.stall_verdict(runs)
    assert verdict["something_bounds_the_stall_below_the_ceiling"] is False
    assert verdict["arms_whose_every_repetition_stayed_under_the_ceiling"] == []


def test_the_dirty_page_settings_are_read_and_never_written():
    settings = writeback.dirty_page_settings()
    assert settings["read_only"] is True
    assert settings["sysctl"]["dirty_ratio"] is not None
    source = open(
        os.path.join(REPO_ROOT, "basemap", "round0254_writeback.py"),
        encoding="utf-8",
    ).read()
    assert "sysctl -w" not in source
    for pattern in ("open('/proc/sys/vm", 'open("/proc/sys/vm'):
        for hit in source.split(pattern)[1:]:
            assert '"w"' not in hit.split(")")[0]


# --------------------------------------------------------------------------- #
# coverage, safety and the standing rules
# --------------------------------------------------------------------------- #


def test_both_nodes_emit_a_covered_span_key_roundreport_reads():
    source = open(
        os.path.join(REPO_ROOT, "experiments", "round0254_nodes.py"),
        encoding="utf-8",
    ).read()
    assert source.count('"observed_span_s": coverage["observed_span_s"]') == 2
    assert "observed_span_s" in _report_constant("_COVERED_SPAN_KEYS")


def test_no_hidden_sigkill_in_this_round_s_files():
    """`subprocess.run(..., timeout=N)` is `Popen.kill()`. AST, not grep."""
    for relative in ("basemap/round0254_writeback.py",
                     "basemap/round0254_dispatch.py",
                     "experiments/round0254_nodes.py",
                     "experiments/prepare_round0254_queue.py"):
        path = os.path.join(REPO_ROOT, relative)
        tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (func.attr if isinstance(func, ast.Attribute)
                    else func.id if isinstance(func, ast.Name) else "")
            if name in {"run", "call", "check_call", "check_output"}:
                assert not any(kw.arg == "timeout" for kw in node.keywords), (
                    f"{relative}: subprocess timeout= is a hidden SIGKILL"
                )
            if name in {"kill", "terminate", "send_signal"}:
                raise AssertionError(f"{relative}: {name}() on a child process")


def test_no_signal_is_delivered_anywhere_in_this_round():
    for relative in ("basemap/round0254_writeback.py",
                     "basemap/round0254_dispatch.py",
                     "experiments/round0254_nodes.py"):
        source = open(os.path.join(REPO_ROOT, relative), encoding="utf-8").read()
        assert "os.kill" not in source
        assert "signal.SIGKILL" not in source
        assert "pkill" not in source


def test_the_scratch_is_on_data_not_root():
    from experiments.round0254_nodes import SCRATCH_ROOT
    assert SCRATCH_ROOT.startswith("/data/")


def test_the_round_registers_nothing():
    from basemap.round0247_registry import registry_fingerprint
    assert registry_fingerprint() == (
        "2f61d1ed00996b5e6b20a5712b0b0c0903eb9e4a6e9a896b2235faf635ffe020"
    )


def test_the_actions_are_the_two_this_round_authorizes():
    from experiments.round0254_nodes import ACTIONS
    assert set(ACTIONS) == {DISPATCH_ACTION, WRITEBACK_ACTION}


def test_run_job_installs_the_hook_before_it_dispatches():
    """The whole point of §B: the callable the runner resolves carries the install."""
    verdict = dispatch.install_effectiveness("experiments.round0254_nodes", "run_job")
    assert verdict["the_install_is_effective"] is True, verdict["why_not"]
    assert install_stop_hooks is not None
