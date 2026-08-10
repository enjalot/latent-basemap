"""R0239 — the signal-safety detector, and proof it catches what the AST guard misses.

A detector nobody has seen fail is not evidence. R0234 established the pattern:
every instrument gets a positive control that proves it fires. So this file
plants deliberate hazards and asserts the detector reports them, plants the
*exact* construct that reached a live cuML child in R0238 and asserts it is
unwaivable, and — the calibration that matters — shows the pre-existing AST
check passing the same planted instance the new detector fails.
"""
from __future__ import annotations

import ast
import importlib
import json
import os
import re
import subprocess
import sys
import textwrap
import time

import pytest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basemap.gpu_child_supervision import (  # noqa: E402
    emit_child_abort_preamble,
    run_gpu_child_cooperative,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DETECTOR = os.path.join(REPO, "experiments", "check_signal_safety.py")

sys.path.insert(0, os.path.join(REPO, "experiments"))
import check_signal_safety as detector  # noqa: E402


#: The modules R0239 remediated. Every future round must keep these clean.
REMEDIATED = (
    "experiments/round0233_nodes.py",
    "experiments/round0235_nodes.py",
    "experiments/round0236_nodes.py",
    "experiments/round0237_nodes.py",
    "experiments/round0238_nodes.py",
    "basemap/gpu_child_supervision.py",
)


# --------------------------------------------------------------------------- #
# positive controls — the detector must be seen to fail
# --------------------------------------------------------------------------- #
def _write(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return str(path)


def test_the_exact_r0238_construct_is_caught_and_is_unwaivable(tmp_path):
    """The literal call that SIGKILLed a live cuML child. This is the control."""
    planted = _write(tmp_path, "planted_r0238.py", '''
        import subprocess
        CUML_LAUNCHER = "/data/latent-basemap/cuml-env/cuml_py"

        def _measure_imbalance_grid(script, repo_root, cache_root):
            completed = subprocess.run(
                [CUML_LAUNCHER, script], cwd=repo_root,
                env=_child_environment(cache_root), capture_output=True, text=True,
                timeout=3_600,
            )
            return completed
    ''')
    findings = detector.scan(planted)
    assert len(findings) == 1, findings
    finding = findings[0]
    assert finding.kind == "SIGNALLING_TIMEOUT"
    assert finding.gpu is True
    assert finding.fatal is True
    assert "GPU-CHILD" in finding.classification()

    # and it gates: the CLI exits nonzero
    proc = subprocess.run(
        [sys.executable, DETECTOR, planted],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 1, proc.stdout
    assert "SIGNALLING_TIMEOUT" in proc.stdout
    assert "GPU-CHILD" in proc.stdout


def test_a_waiver_cannot_rescue_a_gpu_child(tmp_path):
    """The escape hatch must not exist for a CUDA child. No annotation saves it."""
    planted = _write(tmp_path, "planted_waived.py", '''
        import subprocess
        CUML_LAUNCHER = "/data/latent-basemap/cuml-env/cuml_py"

        def probe(script):
            return subprocess.run(
                # signal-safe: I promise this is fine (it is not)
                [CUML_LAUNCHER, script], capture_output=True, timeout=60,
            )
    ''')
    findings = detector.scan(planted)
    assert len(findings) == 1
    assert findings[0].gpu is True
    assert findings[0].fatal is True, "a GPU child must be unwaivable"


def test_every_signalling_timeout_variant_is_caught(tmp_path):
    """`run`, `call`, `check_call` and `check_output` all kill on expiry."""
    for call in ("run", "call", "check_call", "check_output"):
        planted = _write(tmp_path, f"planted_{call}.py", f'''
            import subprocess
            def probe(script):
                return subprocess.{call}([CUML_LAUNCHER, script], timeout=30)
        ''')
        findings = detector.scan(planted)
        assert len(findings) == 1, (call, findings)
        assert findings[0].kind == "SIGNALLING_TIMEOUT", call
        assert findings[0].gpu is True, call


def test_direct_signals_are_caught_and_never_waivable(tmp_path):
    planted = _write(tmp_path, "planted_signals.py", '''
        import os, signal
        def stop(process, pid, pgid):
            process.terminate()          # signal-safe: not a real waiver
            process.kill()
            process.send_signal(signal.SIGKILL)
            os.kill(pid, signal.SIGTERM)
            os.killpg(pgid, signal.SIGKILL)
            signal.alarm(30)
    ''')
    findings = detector.scan(planted)
    kinds = {f.kind for f in findings}
    assert kinds == {"DIRECT_SIGNAL"}
    assert len(findings) >= 6, [str(f) for f in findings]
    assert all(f.fatal for f in findings), "direct signals are never waivable"


def test_a_shell_signal_binary_is_caught(tmp_path):
    planted = _write(tmp_path, "planted_pkill.py", '''
        import subprocess
        def stop():
            subprocess.Popen(["pkill", "-9", "cuml_py"])
    ''')
    findings = detector.scan(planted)
    assert any(f.kind == "DIRECT_SIGNAL" and f.fatal for f in findings), findings


# --------------------------------------------------------------------------- #
# R0240 — the evasions review-0239-01/F1 planted, and the detector missed
#
# Every test below was run against the ORIGINAL detector (beaedb8) and FAILS
# there. A control that passes both before and after proves nothing, so each of
# these is a control the hardening is *seen* to have earned.
# --------------------------------------------------------------------------- #
def _gpu_hazards(findings):
    return [f for f in findings if f.kind == "SIGNALLING_TIMEOUT" and f.gpu]


def test_evasion1_an_aliased_subprocess_module_cannot_hide_a_timeout(tmp_path):
    """`import subprocess as sp`. The old regex knew one prefix: `subprocess`."""
    planted = _write(tmp_path, "planted_alias_module.py", '''
        import subprocess as sp
        CUML_LAUNCHER = "/data/latent-basemap/cuml-env/cuml_py"

        def probe(script):
            return sp.run([CUML_LAUNCHER, script], timeout=3_600)
    ''')
    hazards = _gpu_hazards(detector.scan(planted))
    assert len(hazards) == 1, [str(f) for f in detector.scan(planted)]
    assert hazards[0].fatal is True


def test_evasion2_a_from_import_alias_cannot_hide_a_timeout(tmp_path):
    """`from subprocess import run as _launch`: the callee is not even spelled."""
    planted = _write(tmp_path, "planted_alias_func.py", '''
        from subprocess import run as _launch
        CUML_LAUNCHER = "/data/latent-basemap/cuml-env/cuml_py"

        def probe(script):
            return _launch([CUML_LAUNCHER, script], timeout=3_600)
    ''')
    hazards = _gpu_hazards(detector.scan(planted))
    assert len(hazards) == 1, [str(f) for f in detector.scan(planted)]
    assert hazards[0].fatal is True


def test_evasion3_a_timeout_smuggled_through_kwargs_unpacking_is_caught(tmp_path):
    """The word `timeout` never appears inside the call span at all."""
    planted = _write(tmp_path, "planted_kwargs.py", '''
        import subprocess
        CUML_LAUNCHER = "/data/latent-basemap/cuml-env/cuml_py"

        def probe(script, workdir):
            kwargs = {"cwd": workdir, "timeout": 3_600}
            return subprocess.run([CUML_LAUNCHER, script], **kwargs)
    ''')
    hazards = _gpu_hazards(detector.scan(planted))
    assert len(hazards) == 1, [str(f) for f in detector.scan(planted)]
    assert hazards[0].fatal is True


def test_evasion4_a_backslash_continuation_before_run_is_caught(tmp_path):
    """`subprocess\\` newline `.run(...)` — the lookbehind saw only the dot."""
    planted = _write(tmp_path, "planted_continuation.py", '''
        import subprocess
        CUML_LAUNCHER = "/data/latent-basemap/cuml-env/cuml_py"

        def probe(script):
            return subprocess\\
                .run([CUML_LAUNCHER, script], timeout=3600)
    ''')
    hazards = _gpu_hazards(detector.scan(planted))
    assert len(hazards) == 1, [str(f) for f in detector.scan(planted)]
    assert hazards[0].fatal is True


@pytest.mark.parametrize("argv", [
    '["timeout", "-s", "KILL", "3600", CUML_LAUNCHER, script]',
    '["timeout", "-k", "30", "3600", CUML_LAUNCHER, script]',
    '["/usr/bin/timeout", "3600", CUML_LAUNCHER, script]',
])
def test_evasion5_the_gnu_timeout_binary_in_argv_is_caught(tmp_path, argv):
    """No `timeout=` kwarg anywhere: coreutils delivers the signal instead.

    This is the one the reviewer flagged as most dangerous — `timeout -s KILL`
    is a *more* direct route to SIGKILLing a cuML child than `subprocess.run`.
    """
    planted = _write(tmp_path, "planted_timeout_binary.py", f'''
        import subprocess
        CUML_LAUNCHER = "/data/latent-basemap/cuml-env/cuml_py"

        def probe(script):
            return subprocess.run({argv}, capture_output=True)
    ''')
    hazards = _gpu_hazards(detector.scan(planted))
    assert len(hazards) == 1, [str(f) for f in detector.scan(planted)]
    assert hazards[0].fatal is True, "a GPU child under timeout(1) is unwaivable"


@pytest.mark.parametrize("source", [
    '''
    import functools, subprocess
    CUML_LAUNCHER = "/data/latent-basemap/cuml-env/cuml_py"
    launch = functools.partial(subprocess.run, timeout=3_600)

    def probe(script):
        return launch([CUML_LAUNCHER, script])
    ''',
    '''
    import subprocess
    from functools import partial as p
    CUML_LAUNCHER = "/data/latent-basemap/cuml-env/cuml_py"
    launch = p(subprocess.run, timeout=3_600)

    def probe(script):
        return launch([CUML_LAUNCHER, script])
    ''',
])
def test_evasion6_functools_partial_cannot_curry_a_signalling_timeout(
    tmp_path, source,
):
    """The bound is armed where the child is never named."""
    planted = _write(tmp_path, "planted_partial.py", source)
    findings = detector.scan(planted)
    hazards = [f for f in findings if f.kind == "SIGNALLING_TIMEOUT"]
    assert len(hazards) == 1, [str(f) for f in findings]
    assert hazards[0].fatal is True


def test_evasion7_a_signalling_timeout_inside_a_generated_child_script_is_caught(
    tmp_path,
):
    """`blank_noncode()` erases child scripts by design — so they are re-scanned.

    These modules build the GPU child as a string and hand it to the launcher. A
    `subprocess.run(..., timeout=)` living only in that string is still a
    SIGKILL aimed at a CUDA context. A *plain* literal is the form the old
    detector could never see: `blank_noncode()` blanks it entirely.
    """
    planted = _write(tmp_path, "planted_child_script.py", """
        import subprocess

        BODY = '''
        import subprocess
        subprocess.run(["/data/latent-basemap/cuml-env/cuml_py", "in.py"], timeout=3_600)
        '''

        def build(script_path):
            with open(script_path, "w") as handle:
                handle.write(BODY)
            return script_path
    """)
    findings = detector.scan(planted)
    hazards = _gpu_hazards(findings)
    assert len(hazards) == 1, [str(f) for f in findings]
    assert hazards[0].fatal is True
    assert "generated child script" in hazards[0].snippet


def test_evasion7b_a_generated_child_script_built_as_an_fstring_is_caught(tmp_path):
    """The same, interpolated. The launcher arrives through `{CUML_LAUNCHER!r}`."""
    planted = _write(tmp_path, "planted_child_fstring.py", '''
        import subprocess
        CUML_LAUNCHER = "/data/latent-basemap/cuml-env/cuml_py"

        def build(script_path, workdir):
            body = f"""
        import subprocess
        subprocess.run([{CUML_LAUNCHER!r}, "inner.py"], timeout=3_600)
        """
            with open(script_path, "w") as handle:
                handle.write(body)
            return script_path
    ''')
    findings = detector.scan(planted)
    hazards = _gpu_hazards(findings)
    assert len(hazards) == 1, [str(f) for f in findings]
    assert hazards[0].fatal is True


@pytest.mark.parametrize("source", [
    'import os\ndef stop(pid):\n    os.system("kill -9 %d" % pid)\n',
    'import os\ndef stop():\n    os.system("pkill -f cuml_py")\n',
    'import os\ndef stop():\n    os.popen("killall -9 cuml_py").read()\n',
    'import os as _o\ndef stop(pid):\n    _o.system("kill -9 %d" % pid)\n',
])
def test_evasion8_a_shell_kill_through_os_system_is_caught(tmp_path, source):
    """A signal delivered by `/bin/sh` is still a signal."""
    planted = _write(tmp_path, "planted_os_system.py", source)
    findings = detector.scan(planted)
    shellouts = [
        f for f in findings
        if f.kind == "DIRECT_SIGNAL" and "shell signal" in f.snippet
    ]
    assert len(shellouts) == 1, [str(f) for f in findings]
    assert shellouts[0].fatal is True


def test_evasion9_getattr_with_a_split_literal_signal_method_is_caught(tmp_path):
    """Python folds `"ki" "ll"` to `kill`, so the AST sees what the text hides."""
    planted = _write(tmp_path, "planted_getattr_split.py", '''
        def stop(proc):
            getattr(proc, "ki" "ll")()
    ''')
    findings = detector.scan(planted)
    hits = [f for f in findings if "getattr" in f.snippet]
    assert len(hits) == 1, [str(f) for f in findings]
    assert hits[0].kind == "DIRECT_SIGNAL" and hits[0].fatal is True


@pytest.mark.parametrize("attribute", ["kill", "terminate", "send_signal"])
def test_evasion9b_getattr_with_a_variable_signal_method_is_caught(
    tmp_path, attribute,
):
    """The attribute name held in a local, not written at the call site."""
    planted = _write(tmp_path, "planted_getattr_var.py", f'''
        _METHOD = "{attribute}"

        def stop(proc):
            getattr(proc, _METHOD)()
    ''')
    findings = detector.scan(planted)
    hits = [f for f in findings if "getattr" in f.snippet]
    assert len(hits) == 1, [str(f) for f in findings]
    assert hits[0].kind == "DIRECT_SIGNAL" and hits[0].fatal is True


# --------------------------------------------------------------------------- #
# R0240 — the two defects review-0239-01 found in the detector's own verdicts
# --------------------------------------------------------------------------- #
def test_defect_timeout_none_arms_nothing_and_is_not_reported(tmp_path):
    """`timeout=None` is CPython's "wait forever". Reporting it was noise."""
    planted = _write(tmp_path, "planted_timeout_none.py", '''
        import subprocess
        def extents(path):
            return subprocess.run(["filefrag", path], timeout=None)
    ''')
    assert detector.scan(planted) == []


def test_defect_a_timeout_variable_that_might_be_none_is_still_reported(tmp_path):
    """The fix must be literal-only: an unknown name can hold a real bound."""
    planted = _write(tmp_path, "planted_timeout_var.py", '''
        import subprocess
        def extents(path, deadline_s):
            return subprocess.run(["filefrag", path], timeout=deadline_s)
    ''')
    findings = detector.scan(planted)
    assert len(findings) == 1, [str(f) for f in findings]
    assert findings[0].kind == "SIGNALLING_TIMEOUT" and findings[0].fatal


def test_defect_os_kill_is_counted_once_not_twice(tmp_path):
    """F2: `os.kill(` matched both the `os.kill` and the generic `.kill(` rule.

    Every `os.kill` site was therefore emitted twice, inflating the
    `unwaived hazard(s)` tally and the published inventory built from it.
    """
    planted = _write(tmp_path, "planted_double_count.py", '''
        import os, signal
        def stop(pid):
            os.kill(pid, signal.SIGKILL)
    ''')
    findings = detector.scan(planted)
    assert len(findings) == 1, [str(f) for f in findings]
    assert findings[0].kind == "DIRECT_SIGNAL"
    assert "os.kill" in findings[0].snippet, "the most specific name must survive"
    assert findings[0].fatal is True


# --------------------------------------------------------------------------- #
# the calibration — the old guard passes exactly what the new one catches
# --------------------------------------------------------------------------- #
def test_the_ast_guard_is_blind_to_the_construct_the_detector_catches(tmp_path):
    """This is *why* R0239 exists, asserted rather than asserted-about.

    R0238's `test_the_watchdog_is_conjunctive_and_still_cannot_signal` matches
    call *names*. `subprocess.run` is not a forbidden name, so the delegated
    kill sails through. The new detector matches the argument *span*.
    """
    planted = _write(tmp_path, "planted_blind.py", '''
        import subprocess
        CUML_LAUNCHER = "/data/latent-basemap/cuml-env/cuml_py"

        def probe(script):
            return subprocess.run([CUML_LAUNCHER, script], timeout=3600)
    ''')

    # the old check, reproduced verbatim in shape
    tree = ast.parse(open(planted, encoding="utf-8").read())
    calls = {
        ast.unparse(node.func) for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    ast_guard_fired = (
        any(name in calls for name in ("os.kill", "signal.signal", "os.killpg"))
        or any(c.endswith((".terminate", ".kill", ".send_signal")) for c in calls)
    )
    assert ast_guard_fired is False, "the AST guard is blind here — that is the point"

    # the new check
    assert any(f.fatal for f in detector.scan(planted)), "the new detector must fire"


def test_prose_about_sigkill_does_not_produce_a_finding(tmp_path):
    """These modules discuss `os.kill` at length. A noisy guard gets ignored."""
    planted = _write(tmp_path, "planted_prose.py", '''
        """This module never calls os.kill, terminate() or kill().

        SIGKILL on a CUDA context is what deadlocked RCU. We use
        subprocess.run(..., timeout=3600) nowhere.
        """
        FORBIDDEN = "os.kill"
        # never do process.kill() or subprocess.run(x, timeout=5)
        def safe():
            return 1
    ''')
    assert detector.scan(planted) == []


def test_a_non_gpu_child_may_be_waived_but_only_explicitly(tmp_path):
    unannotated = _write(tmp_path, "planted_bare.py", '''
        import subprocess
        def extents(path):
            return subprocess.run(["filefrag", path], timeout=60)
    ''')
    assert all(f.fatal for f in detector.scan(unannotated)), "bare call must fail"

    annotated = _write(tmp_path, "planted_annotated.py", '''
        import subprocess
        def extents(path):
            return subprocess.run(
                # signal-safe: filefrag never opens the CUDA driver
                ["filefrag", path], timeout=60,
            )
    ''')
    findings = detector.scan(annotated)
    assert len(findings) == 1
    assert findings[0].fatal is False
    assert "filefrag" in findings[0].waiver


def test_a_waiver_cannot_leak_onto_the_next_unannotated_call(tmp_path):
    """R0239 correction to the interrupted attempt's own detector.

    It searched for the waiver comment in `source[start:end + 200]`, so a
    correctly waived `filefrag` call silently excused any unannotated call that
    began within 200 characters of it. The waiver must live INSIDE the span.
    """
    planted = _write(tmp_path, "planted_leak.py", '''
        import subprocess
        def extents(path):
            first = subprocess.run(
                # signal-safe: filefrag never opens the CUDA driver
                ["filefrag", path], timeout=60,
            )
            second = subprocess.run(["stat", path], timeout=60)
            return first, second
    ''')
    findings = detector.scan(planted)
    assert len(findings) == 2, findings
    waived, leaked = findings
    assert waived.fatal is False, "the annotated call is waived"
    assert leaked.waiver is None, "the waiver must not reach the next call"
    assert leaked.fatal is True, "an unannotated call must still gate"


def test_a_direct_signal_is_classified_by_what_its_function_supervises(tmp_path):
    """R0239 correction: a bare `.kill()` has no argv, so the span lies.

    The interrupted attempt classified `_terminate_cooperatively`'s last-resort
    `SIGKILL` — on a child the function's own docstring says holds a CUDA
    context — as a *non-GPU child*, because the eight-character span
    `process.kill()` contains no marker.
    """
    planted = _write(tmp_path, "planted_context.py", '''
        def _terminate_cooperatively(process, escalations):
            """Stop a build that holds a CUDA context without wedging the kernel."""
            process.terminate()
            process.kill()

        def _stop_a_plain_helper(process):
            """Stop a text-munging subprocess that never touches the card."""
            process.kill()
    ''')
    findings = detector.scan(planted)
    by_line = {f.line: f for f in findings}
    gpu = [f for f in findings if f.gpu]
    plain = [f for f in findings if not f.gpu]
    assert len(gpu) == 2, [str(f) for f in findings]
    assert len(plain) == 1, [str(f) for f in findings]
    assert all(f.fatal for f in findings), "direct signals gate either way"


def test_the_real_build_child_kill_paths_are_classified_as_gpu():
    """The live instance of the bug above, in the modules that actually have it.

    R0226/R0224's `_terminate_cooperatively` escalates to `process.kill()` on a
    cuML build child. Rungs 1-5 do not call it — `FlagWatchdog` overrides the
    signal away — but it is still in the tree and it must not be reported as a
    non-GPU call site.
    """
    for relative in ("experiments/round0226_nodes.py", "experiments/round0224_nodes.py"):
        findings = detector.scan(os.path.join(REPO, relative))
        gpu_kills = [
            f for f in findings if f.kind == "DIRECT_SIGNAL" and f.gpu
        ]
        assert gpu_kills, f"{relative}: build-child signal paths must classify as GPU"


def test_communicate_and_wait_timeouts_are_not_hazards(tmp_path):
    """They raise and leave the child running — the whole basis of the fix."""
    planted = _write(tmp_path, "planted_communicate.py", '''
        import subprocess
        def supervise(process, timeout_s):
            try:
                return process.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                return process.communicate()
    ''')
    assert detector.scan(planted) == []


# --------------------------------------------------------------------------- #
# the release must be clean, and stay clean
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("relative", REMEDIATED)
def test_the_remediated_modules_carry_no_unwaived_hazard(relative):
    path = os.path.join(REPO, relative)
    findings = detector.scan(path)
    fatal = [str(f) for f in findings if f.fatal]
    assert not fatal, "\n".join(fatal)


def test_no_cuml_launch_anywhere_in_the_release_is_signal_bounded():
    """The rule with no escape hatch, asserted across every node module."""
    offenders = []
    experiments = os.path.join(REPO, "experiments")
    for name in sorted(os.listdir(experiments)):
        if not name.endswith("_nodes.py"):
            continue
        for finding in detector.scan(os.path.join(experiments, name)):
            if finding.gpu and finding.kind == "SIGNALLING_TIMEOUT":
                offenders.append(str(finding))
    assert not offenders, "\n".join(offenders)


# --------------------------------------------------------------------------- #
# the replacement actually works, end to end, with no GPU
# --------------------------------------------------------------------------- #
def test_a_cooperative_child_completes_normally(tmp_path):
    script = tmp_path / "child.py"
    out = tmp_path / "out.json"
    script.write_text(
        "import json, os\n"
        + emit_child_abort_preamble(str(tmp_path / "abort.flag"))
        + "cells = []\naborted = False\n"
          "try:\n"
          "  for i in range(3):\n"
          "    _check_abort()\n"
          "    cells.append(i)\n"
          "except _CooperativeAbort:\n"
          "    aborted = True\n"
          f"with open({str(out)!r}, 'w') as h:\n"
          "    json.dump({'cells': cells, 'aborted_cooperatively': aborted}, h)\n",
        encoding="utf-8",
    )
    result = run_gpu_child_cooperative(
        [sys.executable, str(script)], cwd=str(tmp_path), env=dict(os.environ),
        flag_path=str(tmp_path / "abort.flag"), deadline_s=60.0, poll_s=0.05,
    )
    assert result.returncode == 0, result.stderr
    assert result.flag_written is False
    assert result.escalations == []
    assert json.loads(out.read_text()) == {
        "cells": [0, 1, 2], "aborted_cooperatively": False
    }
    receipt = result.supervision_receipt()
    assert receipt["signal_delivered"] is False
    assert receipt["bound_mechanism"] == "cooperative-flag-file"


def test_the_deadline_aborts_cooperatively_and_delivers_no_signal(tmp_path):
    """The R0238 scenario, replayed: the bound expires against a running child.

    Under `subprocess.run(timeout=…)` this is where the SIGKILL went. Here the
    child observes a flag, unwinds, and writes what it finished — and the exit
    status proves it was never signalled (a SIGKILLed child returns -9).
    """
    script = tmp_path / "child.py"
    out = tmp_path / "out.json"
    script.write_text(
        "import json, os, time\n"
        + emit_child_abort_preamble(str(tmp_path / "abort.flag"))
        + "cells = []\naborted = False\n"
          "try:\n"
          "  for i in range(200):\n"
          "    _check_abort()\n"
          "    time.sleep(0.05)\n"
          "    cells.append(i)\n"
          "except _CooperativeAbort:\n"
          "    aborted = True\n"
          f"with open({str(out)!r}, 'w') as h:\n"
          "    json.dump({'cells': cells, 'aborted_cooperatively': aborted}, h)\n",
        encoding="utf-8",
    )
    started = time.monotonic()
    result = run_gpu_child_cooperative(
        [sys.executable, str(script)], cwd=str(tmp_path), env=dict(os.environ),
        flag_path=str(tmp_path / "abort.flag"), deadline_s=0.5, poll_s=0.05,
    )
    elapsed = time.monotonic() - started

    assert result.flag_written is True
    assert result.escalations == ["cooperative-flag"]
    # a signalled child reports a negative returncode; this one exited cleanly
    assert result.returncode == 0, result.stderr
    assert elapsed < 10.0, "the child stopped early, so the bound is real"

    payload = json.loads(out.read_text())
    assert payload["aborted_cooperatively"] is True
    assert 0 < len(payload["cells"]) < 200, payload["cells"]
    assert result.supervision_receipt()["signal_delivered"] is False


def test_the_supervisor_itself_contains_no_signal_path():
    """Belt and braces: the fix may not reintroduce what it removes."""
    source = os.path.join(REPO, "basemap", "gpu_child_supervision.py")
    assert [f for f in detector.scan(source) if f.fatal] == []
    tree = ast.parse(open(source, encoding="utf-8").read())
    calls = {
        ast.unparse(node.func) for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert not any(
        c.endswith((".terminate", ".kill", ".send_signal")) for c in calls
    ), sorted(calls)
    assert "os.kill" not in calls and "os.killpg" not in calls


# --------------------------------------------------------------------------- #
# acceptance check 14 — the instruments must be able to capture something
# --------------------------------------------------------------------------- #
def test_the_io_instruments_capture_around_a_supervised_child(tmp_path):
    """R0238's check 14 registered instruments that produced no value anywhere.

    They lived only in the build-child path, which never ran, so the check was
    structurally unable to fail *or* to report. The supervisor now takes them,
    and this proves they are actually invoked and land in the receipt.
    """
    calls: list[str] = []

    class _Sampler:
        def __init__(self, pid):
            self.pid = pid
            calls.append("sampler-constructed")

        def start(self):
            calls.append("sampler-start")

        def halt(self):
            calls.append("sampler-halt")

        def readings(self):
            return {"child_io_read_bytes": 4096, "pid": self.pid}

    def _snapshot():
        calls.append("snapshot")
        return {"seq": len(calls)}

    script = tmp_path / "child.py"
    script.write_text("print('done')\n", encoding="utf-8")
    result = run_gpu_child_cooperative(
        [sys.executable, str(script)], cwd=str(tmp_path), env=dict(os.environ),
        flag_path=str(tmp_path / "abort.flag"), deadline_s=60.0, poll_s=0.05,
        sampler_factory=_Sampler, snapshot=_snapshot,
    )
    assert result.returncode == 0, result.stderr
    assert calls.count("snapshot") == 2, calls
    assert "sampler-start" in calls and "sampler-halt" in calls, calls

    receipt = result.io_receipt()
    assert receipt["instrumented"] is True
    assert receipt["child_io"]["child_io_read_bytes"] == 4096
    assert receipt["snapshot_before"] is not None
    assert receipt["snapshot_after"] is not None
    assert receipt["snapshot_before"] != receipt["snapshot_after"]


def test_an_uninstrumented_child_reports_its_absence_rather_than_hiding_it(tmp_path):
    """A check that cannot fail is the defect. Absence must be visible."""
    script = tmp_path / "child.py"
    script.write_text("print('done')\n", encoding="utf-8")
    result = run_gpu_child_cooperative(
        [sys.executable, str(script)], cwd=str(tmp_path), env=dict(os.environ),
        flag_path=str(tmp_path / "abort.flag"), deadline_s=60.0, poll_s=0.05,
    )
    receipt = result.io_receipt()
    assert receipt["instrumented"] is False
    assert receipt["child_io"] is None


@pytest.mark.parametrize("relative", [
    "experiments/round0237_nodes.py",
    "experiments/round0238_nodes.py",
])
def test_the_grid_and_reachability_nodes_pass_the_io_instruments(relative):
    """Check 14's instruments must reach the nodes whose cells are the point."""
    source = open(os.path.join(REPO, relative), encoding="utf-8").read()
    assert source.count("sampler_factory=lambda pid: _IoSampler(pid=pid)") == 2, relative
    assert source.count('"data_block_device": _data_block_device_stat()') == 2, relative
    assert source.count("_page_cache_resident_bytes(substrate_path)") == 2, relative
    # and the readings must be sealed, not merely collected
    assert source.count('"io_instruments": completed.io_receipt()') == 2, relative
    assert source.count('"supervision": completed.supervision_receipt()') == 2, relative


# --------------------------------------------------------------------------- #
# the generated child scripts must still be valid Python
# --------------------------------------------------------------------------- #
def _child_script_sources(module_path: str) -> list[str]:
    """Every `body = f'''…'''` in a node module, with placeholders substituted.

    Substitution keeps the text syntactically checkable: this asserts that the
    `try:` / `except _CooperativeAbort:` wrappers and the re-indented loops
    R0239 introduced did not break the generated scripts.
    """
    tree = ast.parse(open(module_path, encoding="utf-8").read())
    out: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.JoinedStr)):
            continue
        if not (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            continue
        if node.targets[0].id != "body":
            continue
        text = ""
        for piece in node.value.values:
            if isinstance(piece, ast.Constant):
                text += piece.value
            else:
                text += "None"
        out.append(text)
    return out


@pytest.mark.parametrize("relative", [
    "experiments/round0233_nodes.py",
    "experiments/round0235_nodes.py",
    "experiments/round0236_nodes.py",
    "experiments/round0237_nodes.py",
    "experiments/round0238_nodes.py",
])
def test_generated_child_scripts_are_syntactically_valid(relative):
    path = os.path.join(REPO, relative)
    scripts = _child_script_sources(path)
    assert scripts, f"no generated child script found in {relative}"
    for index, source in enumerate(scripts):
        compile(source, f"<{relative}:body{index}>", "exec")


@pytest.mark.parametrize("relative", [
    "experiments/round0233_nodes.py",
    "experiments/round0235_nodes.py",
    "experiments/round0236_nodes.py",
    "experiments/round0237_nodes.py",
    "experiments/round0238_nodes.py",
])
def test_every_generated_gpu_child_polls_the_cooperative_flag(relative):
    """A flag nobody polls is not a bound."""
    source = open(os.path.join(REPO, relative), encoding="utf-8").read()
    assert "emit_child_abort_preamble(flag_path)" in source
    assert "_check_abort()" in source
    assert "except _CooperativeAbort:" in source
    assert "run_gpu_child_cooperative(" in source


# --------------------------------------------------------------------------- #
# the shared infrastructure that launches every node
# --------------------------------------------------------------------------- #
#: The round runner lives in a different (private) repo, so this is a property
#: check on an absolute path that may not exist on another machine — never a
#: pinned line number or count, both of which its author would churn.
ROUND_RUNNER = "/home/enjalot/code/workshop/rounds/runner.py"


def test_the_shared_round_runner_never_signal_bounds_a_gpu_node():
    """The detector is only worth having if it is pointed at what launches us.

    The runner spawns every node in the program, so a signalling bound there
    reaches a cuML child through its process group. What it *may* have is a
    guarded escalation that runs only after the node is proven free of any CUDA
    context: that is a `DIRECT_SIGNAL`, it is fatal by design, and it must
    classify as non-GPU. What it may not have is anything classified GPU-CHILD,
    or a bare `proc.kill()` / `proc.terminate()` on the node process.
    """
    if not os.path.exists(ROUND_RUNNER):
        pytest.skip(f"{ROUND_RUNNER} is not present on this machine")
    findings = detector.scan(ROUND_RUNNER)

    gpu = [str(f) for f in findings if f.gpu]
    assert not gpu, "the runner must never signal-bound a GPU child:\n" + "\n".join(gpu)

    marked = [
        str(f) for f in findings
        if f.kind == "SIGNALLING_TIMEOUT"
        and any(marker in f.snippet for marker in detector._GPU_MARKERS)
    ]
    assert not marked, "\n".join(marked)

    with open(ROUND_RUNNER, encoding="utf-8") as handle:
        code = detector.blank_noncode(handle.read())
    bare = re.findall(r"\bproc\s*\.\s*(kill|terminate|send_signal)\s*\(", code)
    assert not bare, f"bare signal on the node process: {sorted(set(bare))}"


def test_the_child_preamble_compiles_and_raises_on_the_flag(tmp_path):
    flag = tmp_path / "abort.flag"
    namespace: dict = {"os": os}
    exec(compile(emit_child_abort_preamble(str(flag)), "<preamble>", "exec"), namespace)
    check = namespace["_check_abort"]
    abort = namespace["_CooperativeAbort"]
    check()  # no flag, no raise
    flag.write_text("stop", encoding="utf-8")
    with pytest.raises(abort):
        check()


# --------------------------------------------------------------------------- #
# review-0239-02/F4 — the runner's cooperative flag must actually be polled
# --------------------------------------------------------------------------- #
def test_the_runners_abort_flag_reaches_a_supervised_gpu_child(tmp_path,
                                                               monkeypatch):
    """The runner wrote a flag and nothing read it, so the bound was one-phase.

    The runner publishes `ROUNDRUN_ABORT_FLAG` to every node. Here the parent
    watchdog observes it well before the cell's own deadline and relays it onto
    the cell flag; the child sees it, unwinds, and exits cleanly — the exit
    status is the proof that no signal was delivered.
    """
    runner_flag = tmp_path / "node.abort"
    monkeypatch.setenv("ROUNDRUN_ABORT_FLAG", str(runner_flag))
    script = tmp_path / "child.py"
    out = tmp_path / "out.json"
    script.write_text(
        "import json, os, time\n"
        + emit_child_abort_preamble(str(tmp_path / "abort.flag"))
        + "cells = []\naborted = False\n"
          "try:\n"
          "  for i in range(200):\n"
          "    _check_abort()\n"
          "    time.sleep(0.05)\n"
          "    cells.append(i)\n"
          "except _CooperativeAbort:\n"
          "    aborted = True\n"
          f"with open({str(out)!r}, 'w') as h:\n"
          "    json.dump({'cells': cells, 'aborted_cooperatively': aborted}, h)\n",
        encoding="utf-8",
    )
    runner_flag.write_text('{"reason":"operator abort"}', encoding="utf-8")
    started = time.monotonic()
    result = run_gpu_child_cooperative(
        [sys.executable, str(script)], cwd=str(tmp_path),
        env={k: v for k, v in os.environ.items()
             if k != "ROUNDRUN_ABORT_FLAG"},   # parent-side relay only
        flag_path=str(tmp_path / "abort.flag"),
        deadline_s=600.0,                      # nowhere near expiring
        poll_s=0.05,
    )
    elapsed = time.monotonic() - started

    assert result.runner_abort_observed is True, (
        "the runner's abort flag was never polled by the supervisor")
    assert result.flag_written is True
    assert result.escalations == ["cooperative-flag"]
    assert result.returncode == 0, result.stderr
    assert elapsed < 30.0, "the runner's request did not bound the child"
    payload = json.loads(out.read_text())
    assert payload["aborted_cooperatively"] is True
    assert len(payload["cells"]) < 200
    assert result.supervision_receipt()["runner_abort_observed"] is True
    assert result.supervision_receipt()["signal_delivered"] is False


def test_the_generated_child_polls_the_runners_flag_itself(tmp_path,
                                                           monkeypatch):
    """Belt and braces: the child does not depend on its parent's watchdog."""
    runner_flag = tmp_path / "node.abort"
    namespace: dict = {"os": os}
    exec(compile(emit_child_abort_preamble(str(tmp_path / "abort.flag")),
                 "<preamble>", "exec"), namespace)
    check = namespace["_check_abort"]
    abort = namespace["_CooperativeAbort"]
    monkeypatch.setenv("ROUNDRUN_ABORT_FLAG", str(runner_flag))
    check()  # the runner has asked for nothing yet
    runner_flag.write_text("stop", encoding="utf-8")
    with pytest.raises(abort):
        check()


def test_the_real_build_child_polls_the_runners_flag(tmp_path, monkeypatch):
    """`basemap/round0233_build.py` is the child every rung's build delegates to.

    R0235 → R0236 → R0237 → R0238 all bottom out in this module's `_check_abort`,
    so this one call site covers the 100M build child.
    """
    build = importlib.import_module("basemap.round0233_build")
    runner_flag = tmp_path / "node.abort"
    monkeypatch.setenv("ROUNDRUN_ABORT_FLAG", str(runner_flag))
    build._check_abort(None, where="start")   # nothing set: no abort
    runner_flag.write_text("stop", encoding="utf-8")
    with pytest.raises(build.CooperativeAbort):
        build._check_abort(None, where="start")


def test_a_node_parent_loop_observes_the_runners_flag(tmp_path, monkeypatch):
    """`run_truth` holds a torch CUDA context in the NODE process itself."""
    nodes = importlib.import_module("experiments.round0238_nodes")
    runner_flag = tmp_path / "node.abort"
    monkeypatch.setenv("ROUNDRUN_ABORT_FLAG", str(runner_flag))
    nodes._check_runner_abort("exact-truth database chunk")
    runner_flag.write_text("stop", encoding="utf-8")
    with pytest.raises(nodes.Round0238Error):
        nodes._check_runner_abort("exact-truth database chunk")


def test_the_build_watchdog_relays_the_runners_flag_without_a_signal(
    tmp_path, monkeypatch,
):
    """The 100M build child is supervised by `MemAwareFlagWatchdog`, not by
    `run_gpu_child_cooperative`, so it needs its own poll."""
    nodes = importlib.import_module("experiments.round0238_nodes")
    # No GPU, no /proc reads: this test is about the flag, nothing else.
    monkeypatch.setattr(nodes, "_nvidia_smi_device_bytes", lambda: 0)
    monkeypatch.setattr(nodes, "_nvidia_smi_per_process_bytes", lambda _pid: 0)
    monkeypatch.setattr(nodes, "_proc_memory_bytes", lambda _pid: (0, 0))
    monkeypatch.setattr(nodes, "_swap_used_bytes", lambda: 0)
    monkeypatch.setattr(nodes, "_mem_available_bytes", lambda: 1 << 40)
    cell_flag = tmp_path / "abort.flag"
    runner_flag = tmp_path / "node.abort"
    monkeypatch.setenv("ROUNDRUN_ABORT_FLAG", str(runner_flag))
    watchdog = nodes.MemAwareFlagWatchdog(
        flag_path=str(cell_flag), pid=os.getpid(), poll_s=0.05,
        host_anon_budget_bytes=1 << 60, swap_growth_abort_bytes=1 << 60,
        device_baseline_bytes=0, swap_baseline_bytes=0,
    )
    watchdog.sample_once()
    assert not cell_flag.exists() and watchdog.aborted is False

    runner_flag.write_text("stop", encoding="utf-8")
    watchdog.sample_once()
    assert cell_flag.exists(), "the runner's request never reached the cell"
    assert watchdog.aborted is True
    assert watchdog.escalations == ["cooperative-flag"]
    assert watchdog.readings()["runner_abort_observed"] is True
