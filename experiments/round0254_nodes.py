#!/usr/bin/env python3
"""R0254 node handlers — what bounds the stall, and where the poll really goes.

Two nodes, both CPU.  review-0253-01 §K.1 made the ordering binding: the write
stall's distribution "is the only item that can invalidate everything
downstream", and items 2--4 are wasted if it says the interval cannot be bounded.
So the write measurement is the round's centre and the GPU stays idle.

* `dispatch_0254` (CPU) — §B/§C/§D.  R0253 installed the stop hook on `19`
  hand-nominated entries and proved it three ways, but the proof resolved
  `run_train` / `run_panel` while the runner resolves `run_job`, and **0 of the
  dispatched handlers on this box were among the 19** (review-0253-01 §A.2).
  This node reads every `queue.json`, derives the entry list from what the runner
  actually resolved plus the transitive `run_*` delegation closure, audits each
  derived entry for an install that is *effective* rather than merely present,
  plants the three shapes review-0253-01 §A.3 got past R0253's auditor, and
  counts how many derived entries both install **and** construct a gate.
* `writeback_0254` (CPU) — §A.  Six flush disciplines at the `49,152,000,000` B
  rung that breached, five repetitions each, round-robin with a rotating start so
  no arm owns a device state, plus one unpolled control.  The write unit is held
  at 64 MiB: review-0253-01 §D falsified the smaller-unit remedy and
  `plan-minilm-100m-v2.md` now records it, so retrying it would spend the disk
  budget re-deriving a known negative.

Both nodes emit `observed_span_s` at the top level of their artifact, so
`roundreport`'s poll-coverage table prints a real percentage rather than UNKNOWN.
No node writes the runner's abort flag; the stop-latency control writes its own.
No signal is delivered on any path, and no CUDA library is loaded by either node.
"""
from __future__ import annotations

import importlib
import os
import shutil
import sys
import time
from collections.abc import Mapping
from typing import Any

from basemap.output_safety import create_fresh_directory
from basemap.round0242_locality import io_counters
from basemap.round0251_trainer_setup import PollRecorder
from basemap.round0252_stoppability import (
    STOP_CONTROL_DELAY_S,
    gap_report,
    measure_stop_latency,
)
from basemap.round0253_coverage import CoverageLedger, coverage_summary
from basemap.round0253_stop_hooks import (
    NOT_A_FAMILY_CELL,
    THE_INSTRUMENT_IS_DEFEATABLE,
    install_stop_hooks,
    over_the_ceiling,
    registered_ceiling_s,
    stop_hooks_state,
)
from basemap.round0254_dispatch import (
    DISPATCH_CAPABILITY,
    DISPATCH_SCHEMA,
    SCOPE_MODULES,
    WRITEBACK_CAPABILITY,
    WRITEBACK_SCHEMA,
    assert_derived_entries_install,
    derived_entries,
    dispatch_census,
    entry_install_audit,
    entry_tuples,
    gate_census,
    install_effectiveness,
    scope_residual,
)
from basemap.round0254_writeback import (
    ARM_UNPOLLED_CONTROL,
    ARMS,
    SHIPPED_ARMS,
    STALL_SIZE_BYTES,
    arm_schedule,
    assert_write_loop_polls,
    dirty_page_settings,
    stall_verdict,
    write_arm,
)
from experiments.round0251_nodes import (
    _guard_tail_reported,
    _node_gate,
    _node_guard,
    _receipt_envelope as _r0251_envelope,
    _score_gate_without_raising,
    _seal,
    _start_node,
)


ROUND_ID = "0254"

DISPATCH_ACTION = "round0254_dispatch_derived_install"
WRITEBACK_ACTION = "round0254_write_stall_distribution"
ACTIONS: tuple[str, ...] = (DISPATCH_ACTION, WRITEBACK_ACTION)

#: Scratch for the 49 GB files.  On `/data`, never `/`, one file at a time,
#: removed by the node whether it succeeds or fails.
SCRATCH_ROOT = "/data/tmp/round0254-scratch"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS_DIR = os.path.join(REPO_ROOT, "experiments")

#: Repetitions per shipped arm.  review-0253-01 §K.1 asked for `n >= 5`.
REPETITIONS = 5


class Round0254NodeError(RuntimeError):
    """The registered R0254 node contract changed."""


def _receipt_envelope(manifest: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(_r0251_envelope(manifest))
    body["round_id"] = ROUND_ID
    return body


def force_remove_tree(root: str) -> dict[str, Any]:
    """Remove a tree, chmod-ing anything the output-safety layer made read-only."""
    removed = {"root": root, "existed": os.path.isdir(root), "chmods": 0}
    if not os.path.isdir(root):
        return removed
    for base, directories, files in os.walk(root, topdown=False):
        for name in files + directories:
            target = os.path.join(base, name)
            try:
                os.chmod(target, 0o700)
                removed["chmods"] += 1
            except OSError:
                pass
    try:
        os.chmod(root, 0o700)
    except OSError:
        pass
    shutil.rmtree(root, ignore_errors=True)
    removed["removed"] = not os.path.isdir(root)
    return removed


def _free_bytes(path: str) -> int:
    stat = os.statvfs(path)
    return int(stat.f_bavail) * int(stat.f_frsize)


def _anonymous_bytes() -> int:
    """Host ANONYMOUS memory, never RSS (`plan-minilm-100m-v2.md`)."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return -1


def _no_cuda_here() -> dict[str, Any]:
    """The PRESENCE rule, applied to this process, expecting an empty result.

    Both nodes are CPU. The rule is checked rather than asserted so the receipt
    says what was observed: `plan-minilm-100m-v2.md` records a 2026-08-11 case
    where a process launched with `CUDA_VISIBLE_DEVICES=""` still mapped ten CUDA
    objects, so the environment variable is not evidence.
    """
    import re

    matches: set[str] = set()
    try:
        with open("/proc/self/maps", "r", encoding="utf-8") as handle:
            for line in handle:
                for hit in re.findall(
                    r"/[^ ]*(?:libcuda|libcuvs|libcudart|libcublas|libcudnn|nvidia)[^ ]*",
                    line, flags=re.IGNORECASE,
                ):
                    matches.add(hit)
    except OSError:
        pass
    return {
        "schema": "round0254-cuda-absence-v1",
        "pid": os.getpid(),
        "mapped_cuda_objects": sorted(matches),
        "this_process_maps_a_cuda_library": bool(matches),
        "rule": (
            "PRESENCE, not a count. Read from /proc/self/maps, not from "
            "CUDA_VISIBLE_DEVICES, because the environment variable is not "
            "evidence (plan-minilm-100m-v2.md, 2026-08-11)."
        ),
    }


# --------------------------------------------------------------------------- #
# B / C / D. dispatch-derived install, an auditor that works, and the gate count
# --------------------------------------------------------------------------- #


#: The three shapes review-0253-01 §A.3 planted against R0253's `_calls_install`
#: and that all passed it, plus two more of the same class.  Each is written to a
#: real importable module and run through the SHIPPED auditor -- not through a
#: copy of its logic -- so a control that could not fail is a node failure.
PLANTED_DEFECTS: tuple[tuple[str, str], ...] = (
    (
        "dead_branch",
        "from basemap.round0253_stop_hooks import install_stop_hooks\n"
        "\n\n"
        "def run_job(active, job):\n"
        "    if False:\n"
        "        install_stop_hooks(label='planted')\n"
        "    return None\n",
    ),
    (
        "module_level_shadow",
        "from basemap.round0253_stop_hooks import install_stop_hooks\n"
        "\n\n"
        "def install_stop_hooks(**kwargs):\n"
        "    return {}\n"
        "\n\n"
        "def run_job(active, job):\n"
        "    install_stop_hooks(label='planted')\n"
        "    return None\n",
    ),
    (
        "deferred_lambda",
        "from basemap.round0253_stop_hooks import install_stop_hooks\n"
        "\n\n"
        "def run_job(active, job):\n"
        "    hook = lambda: install_stop_hooks(label='planted')\n"
        "    return hook\n",
    ),
    (
        "guarded_by_try",
        "from basemap.round0253_stop_hooks import install_stop_hooks\n"
        "\n\n"
        "def run_job(active, job):\n"
        "    try:\n"
        "        install_stop_hooks(label='planted')\n"
        "    except Exception:\n"
        "        pass\n"
        "    return None\n",
    ),
    (
        "function_local_shadow",
        "from basemap.round0253_stop_hooks import install_stop_hooks\n"
        "\n\n"
        "def run_job(active, job):\n"
        "    install_stop_hooks(label='planted')\n"
        "\n"
        "    def install_stop_hooks(**kwargs):\n"
        "        return {}\n"
        "    return None\n",
    ),
)

#: The honest install, for the negative-control's negative control: if this one
#: did not pass, the audit would be rejecting everything and the five failures
#: above would prove nothing.
PLANTED_HONEST = (
    "honest_install",
    "from basemap.round0253_stop_hooks import install_stop_hooks\n"
    "\n\n"
    "def run_job(active, job):\n"
    "    install_stop_hooks(label='planted')\n"
    "    return None\n",
)


def _plant_and_audit(scratch: str) -> dict[str, Any]:
    """Write real modules with the defect, import them, run the SHIPPED auditor."""
    package = os.path.join(scratch, "round0254_planted")
    os.makedirs(package, exist_ok=True)
    with open(os.path.join(package, "__init__.py"), "w", encoding="utf-8") as handle:
        handle.write("")
    if scratch not in sys.path:
        sys.path.insert(0, scratch)
    # Every file is written BEFORE the first import, and the finder's directory
    # cache is invalidated once afterwards. `importlib`'s `FileFinder` caches a
    # package directory's listing on first import and revalidates it only on an
    # mtime change with 1 s granularity, so writing the second module after
    # importing the first raises `ModuleNotFoundError` on a fast disk. That is a
    # property of the harness, not of the guard under test.
    for name, source in (*PLANTED_DEFECTS, PLANTED_HONEST):
        with open(os.path.join(package, f"{name}.py"), "w", encoding="utf-8") as handle:
            handle.write(source)
    importlib.invalidate_caches()

    rows: list[dict[str, Any]] = []
    try:
        for name, _source in (*PLANTED_DEFECTS, PLANTED_HONEST):
            module_name = f"round0254_planted.{name}"
            verdict = install_effectiveness(module_name, "run_job")
            rows.append({
                "planted": name,
                "module": module_name,
                "the_shipped_auditor_says_the_install_is_effective":
                    verdict["the_install_is_effective"],
                "why_not": verdict["why_not"],
                "expected_to_be_caught": name != PLANTED_HONEST[0],
            })
    finally:
        if scratch in sys.path:
            sys.path.remove(scratch)
        for name, _source in (*PLANTED_DEFECTS, PLANTED_HONEST):
            sys.modules.pop(f"round0254_planted.{name}", None)
        sys.modules.pop("round0254_planted", None)

    caught = [
        row for row in rows
        if row["expected_to_be_caught"]
        and not row["the_shipped_auditor_says_the_install_is_effective"]
    ]
    honest = [
        row for row in rows
        if not row["expected_to_be_caught"]
        and row["the_shipped_auditor_says_the_install_is_effective"]
    ]
    return {
        "schema": "round0254-planted-install-defect-controls-v1",
        "controls": rows,
        "defects_planted": len(PLANTED_DEFECTS),
        "defects_caught_by_the_shipped_auditor": len(caught),
        "the_honest_install_still_passes": bool(honest),
        "every_planted_defect_was_caught": len(caught) == len(PLANTED_DEFECTS),
        "how_this_differs_from_r0253_s_control": (
            "Each planted module is written to disk, imported, and passed to the "
            "SHIPPED `install_effectiveness`. review-0253-01 §I blocked R0253's "
            "equivalent claim because one of its four controls re-implemented the "
            "guard's AST walk in the test body and would have passed had the "
            "guard returned True unconditionally. Nothing here re-implements "
            "anything; if `install_effectiveness` returned True unconditionally, "
            "all five of these would fail."
        ),
        "the_three_r0253_shapes": (
            "`dead_branch`, `module_level_shadow` and `deferred_lambda` are "
            "review-0253-01 §A.3's three shapes verbatim. All three PASSED "
            "R0253's `entry_install_audit`."
        ),
    }


def run_dispatch(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    """§B/§C/§D. Derive the entry list from dispatch; audit it; count the gates."""
    install_stop_hooks(label="R0254 round0254_nodes.run_dispatch")
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0254NodeError("R0254 dispatch handler received another queue")
    node_id = str(active.get("node_id") or "dispatch_0254")
    ledger = CoverageLedger(node=node_id)
    label = "R0254 dispatch-derived stop-hook install"
    abort_flag = _start_node(label)
    output = create_fresh_directory(str(job["outputs"][0]), label="R0254 dispatch")
    scratch = create_fresh_directory(SCRATCH_ROOT, label="R0254 dispatch scratch")
    cuda = _no_cuda_here()

    # The gate and the coverage window are constructed BEFORE any of the work,
    # not after it. R0252's severe defect (review-0252-01 §H) was that its
    # expensive stage ran before a gate existed, so no census could see it, and a
    # first cut of this node reproduced that shape in miniature: it did every
    # audit first and then polled three times in a row, which sealed
    # `observed_span_s = 1.12e-05` against a `1.577` s node -- a coverage of
    # `0.0007%`, quantitative and useless. Each stage below is followed by a real
    # abort read, so the span the ledger reports is the audit itself.
    window = ledger.window("R0254 dispatch audit stage")
    guard_ctx = _node_guard(label)
    gate = _node_gate(label, training_performed=False)
    try:
        with guard_ctx:
            gate.start()
            recorder = PollRecorder(gate=gate, clock=time.monotonic)
            recorder.anchor("R0254 dispatch stage entered")
            wrapped = window.wrap(recorder)

            census = dispatch_census()
            wrapped("R0254 queue-manifest dispatch census complete")
            derived = derived_entries(SCOPE_MODULES, census)
            wrapped("R0254 entry list derived from the dispatch path")
            entries = entry_tuples(derived)
            audit = entry_install_audit(entries)
            wrapped("R0254 effective-install audit complete")
            guard = assert_derived_entries_install(SCOPE_MODULES, census)
            wrapped("R0254 derived-entry install guard complete")
            gates = gate_census(entries)
            wrapped("R0254 install-and-gate census complete")
            residual = scope_residual(census, SCOPE_MODULES)
            wrapped("R0254 scope residual complete")
            planted = _plant_and_audit(scratch)
            wrapped("R0254 planted-defect controls complete")
            if not planted["every_planted_defect_was_caught"]:
                raise Round0254NodeError(
                    "R0254: the shipped install auditor did not catch every "
                    f"planted defect: {planted['controls']}"
                )
            if not planted["the_honest_install_still_passes"]:
                raise Round0254NodeError(
                    "R0254: the shipped install auditor rejects an honest "
                    "install, so its five refusals prove nothing"
                )

            # How many of the entries the runner has ACTUALLY dispatched now
            # carry an effective install -- review-0253-01 §A.2's `0 of 206`.
            in_scope_dispatched = [
                (str(row["module"]), str(row["callable"]))
                for row in census["handlers"]
                if str(row["module"]) in set(SCOPE_MODULES)
            ]
            dispatched_audit = entry_install_audit(in_scope_dispatched)
            wrapped("R0254 dispatched-handler audit complete")
            state = stop_hooks_state()
            wrapped("R0254 stop-hook state read")
            gate.finish("R0254 dispatch stage end")
        window.close()
        tail = _guard_tail_reported(guard_ctx, label=label)
        scored = _score_gate_without_raising(gate, tail, label=label)
        report = gap_report(recorder.records, arm="dispatch")
    finally:
        force_remove_tree(scratch)

    coverage = ledger.receipt()
    body = dict(_receipt_envelope(active["manifest"]))
    body.update({
        "schema": DISPATCH_SCHEMA,
        "capability": DISPATCH_CAPABILITY,
        "capabilities": [DISPATCH_CAPABILITY],
        "abort_flag_precondition": abort_flag,
        "node": node_id,
        "gate_registered": False,
        "is_a_family_cell": False,
        "not_a_family_cell": NOT_A_FAMILY_CELL,
        "the_instrument_is_defeatable": THE_INSTRUMENT_IS_DEFEATABLE,
        "cuda_absence": cuda,
        "dispatch_census": census,
        "derived_entry_list": derived,
        "effective_install_audit": audit,
        "derived_entry_install_guard": {
            "schema": guard["schema"],
            "entries_audited": guard["audit"]["entries_audited"],
            "every_entry_installs_effectively":
                guard["audit"]["every_entry_installs_effectively"],
        },
        "dispatched_handlers_in_scope_audit": {
            "handlers_audited": dispatched_audit["entries_audited"],
            "handlers_with_an_effective_install":
                dispatched_audit["entries_with_an_effective_install"],
            "handlers_without": dispatched_audit["entries_without_an_effective_install"],
            "what_r0253_measured_here": (
                "0 of 206. review-0253-01 §A.2: none of R0253's 19 registered "
                "entries was ever a dispatched handler, because the runner "
                "resolves run_job and run_job was excluded from R0253's census "
                "by name."
            ),
        },
        "install_and_gate_census": gates,
        "scope_residual": residual,
        "planted_defect_controls": planted,
        "stop_hooks_state_after_install": state,
        "gap_report": report,
        "enforcement_poll_spacing": scored,
        "guard_tail": tail,
        "poll_coverage": coverage,
        "observed_span_s": coverage["observed_span_s"],
        "node_wall_s": coverage["node_wall_s"],
        "what_this_node_proves": (
            "That the entry list is derived from the runner's own dispatch record "
            "rather than nominated: every (handler_module, handler_callable) pair "
            "in every queue.json on this box, closed transitively over the "
            "module-level run_* functions those callables delegate to. That every "
            "derived entry carries an install which is unconditional, unshadowed "
            "at module and function scope, and resolves at runtime to the release "
            "function -- checked by the same function that refuses all five "
            "planted defects, three of which passed R0253's auditor. And how many "
            "of those entries also construct a gate, which is a different and "
            "smaller number."
        ),
        "what_this_node_does_not_prove": (
            "That the modules in scope are the modules a Phase 2 ladder train will "
            "dispatch: that queue does not exist yet and the module list is this "
            "round's judgement, published beside the full dispatched-module count. "
            "That an entry reaching a binding call or a gate through another "
            "module's helper is covered: both walks are module-local and are lower "
            "bounds. And it does not make the AbortPollGate verdicts sound -- "
            "review-0249-01 §B.1/§B.2 remain open by owner decision."
        ),
    })
    _seal(output, f"{node_id}-dispatch-derived-install.json", body)


# --------------------------------------------------------------------------- #
# A. what bounds the write stall
# --------------------------------------------------------------------------- #


def _one_run(*, arm: str, repetition: int, scratch: str, ledger: CoverageLedger,
             label: str) -> dict[str, Any]:
    spec = ARMS[arm]
    path = os.path.join(scratch, f"stall-{arm}-{repetition}.bin")
    site = f"{label} {arm}"
    guard = _node_guard(site)
    gate = _node_gate(site, training_performed=False)
    window = ledger.window(f"{site} rep {repetition}")
    try:
        with guard:
            gate.start()
            recorder = PollRecorder(gate=gate, clock=time.monotonic)
            recorder.anchor(f"R0254 {arm} write entered")
            wrapped = window.wrap(recorder)
            created = write_arm(
                path, arm=arm, seed=20260811 + repetition,
                poll=(wrapped if spec.polls else None),
            )
            wrapped(f"R0254 {arm} file written")
            gate.finish(f"R0254 {arm} write stage end")
        window.close()
        tail = _guard_tail_reported(guard, label=site)
        scored = _score_gate_without_raising(gate, tail, label=site)
        report = gap_report(recorder.records, arm=arm)
    finally:
        if os.path.exists(path):
            os.unlink(path)
    return {
        "arm": arm,
        "repetition": repetition,
        **{key: value for key, value in created.items() if key != "path"},
        "gap_report": report,
        "enforcement_poll_spacing": scored,
        "guard_tail": tail,
        "coverage_window": window.receipt(),
        "free_bytes_after_unlink": _free_bytes("/data"),
        "host_anonymous_bytes": _anonymous_bytes(),
    }


def run_writeback(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    """§A. Measure the stall distribution across flush disciplines."""
    install_stop_hooks(label="R0254 round0254_nodes.run_writeback")
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0254NodeError("R0254 writeback handler received another queue")
    node_id = str(active.get("node_id") or "writeback_0254")
    ledger = CoverageLedger(node=node_id)
    label = "R0254 write stall"
    abort_flag = _start_node(label)
    output = create_fresh_directory(str(job["outputs"][0]), label="R0254 writeback")
    cuda = _no_cuda_here()
    guard_receipt = assert_write_loop_polls()
    repetitions = int(job.get("repetitions", REPETITIONS))
    size = int(job.get("stall_size_bytes", STALL_SIZE_BYTES))
    if size != STALL_SIZE_BYTES:
        raise Round0254NodeError(
            f"R0254 measures the rung that breached ({STALL_SIZE_BYTES} B), not "
            f"{size} B"
        )
    dirty = dirty_page_settings()
    io_before = io_counters()
    free_before = _free_bytes("/data")

    scratch = create_fresh_directory(SCRATCH_ROOT, label="R0254 writeback scratch")
    runs: list[dict[str, Any]] = []
    stop_controls: list[dict[str, Any]] = []
    try:
        # One warm-up write, discarded. review-0253-01 §C.3: its runs 1--2 on a
        # quiescent device produced a 0.42 s maximum and runs 3--7 produced
        # 1.5--2.3 s. An arm that runs first on a cold device is not comparable
        # to one that runs after 100 GB of traffic, so every measured arm starts
        # after the same warm-up and the schedule rotates on top of that.
        warmup = _one_run(arm=ARMS[SHIPPED_ARMS[0]].name, repetition=-1,
                          scratch=scratch, ledger=ledger, label=f"{label} warmup")
        for repetition, arm in arm_schedule(SHIPPED_ARMS, repetitions):
            runs.append(_one_run(arm=arm, repetition=repetition, scratch=scratch,
                                 ledger=ledger, label=label))
        control = _one_run(arm=ARM_UNPOLLED_CONTROL, repetition=0, scratch=scratch,
                           ledger=ledger, label=f"{label} unpolled_control")

        # A stop planted mid-write, on this node's OWN flag path, through the
        # best shipped arm. Not the runner's flag.
        verdict_so_far = stall_verdict(runs)
        best_arm = str(verdict_so_far["best_shipped_arm"])
        control_path = os.path.join(scratch, "stall-stop-control.bin")

        def run_under_poll(poll, _path=control_path, _arm=best_arm):
            try:
                write_arm(_path, arm=_arm, seed=20260815, poll=poll)
            finally:
                if os.path.exists(_path):
                    os.unlink(_path)

        # The stop control writes another 49 GB and it is instrumented too:
        # `FlagFileAbortPoll` forwards every read to `inner`, so the control's
        # own window records the same span the census would otherwise miss.
        stop_window = ledger.window(f"{label} stop control under {best_arm}")
        outcome = measure_stop_latency(
            label=f"write of the {STALL_SIZE_BYTES} B rung under {best_arm}",
            flag_path=os.path.join(scratch, "stall-stop.abort"),
            delay_s=STOP_CONTROL_DELAY_S,
            run=run_under_poll,
            inner=stop_window.observe,
        )
        stop_window.close()
        if not outcome["the_work_stopped_cooperatively"]:
            raise Round0254NodeError(
                "R0254 write stop control did not stop: the polled write path ran "
                "to completion with the flag planted"
            )
        stop_controls.append(outcome)
    finally:
        force_remove_tree(scratch)
    io_after = io_counters()

    verdict = stall_verdict(runs)
    ceiling = registered_ceiling_s()
    coverage = ledger.receipt()
    body = dict(_receipt_envelope(active["manifest"]))
    body.update({
        "schema": WRITEBACK_SCHEMA,
        "capability": WRITEBACK_CAPABILITY,
        "capabilities": [WRITEBACK_CAPABILITY],
        "abort_flag_precondition": abort_flag,
        "node": node_id,
        "gate_registered": False,
        "is_a_family_cell": False,
        "not_a_family_cell": NOT_A_FAMILY_CELL,
        "the_instrument_is_defeatable": THE_INSTRUMENT_IS_DEFEATABLE,
        "cuda_absence": cuda,
        "write_loop_poll_guard": guard_receipt,
        "registered_ceiling_s": ceiling,
        "stall_size_bytes": size,
        "repetitions_per_arm": repetitions,
        "schedule": [
            {"repetition": repetition, "arm": arm}
            for repetition, arm in arm_schedule(SHIPPED_ARMS, repetitions)
        ],
        "warmup_run": warmup,
        "runs": runs,
        "unpolled_control_run": control,
        "stall_verdict": verdict,
        "stop_latency": stop_controls,
        "dirty_page_settings": dirty,
        "io_counters": {"before": io_before, "after": io_after},
        "disk": {
            "free_bytes_before": free_before,
            "free_bytes_after": _free_bytes("/data"),
            "peak_scratch_bytes": size,
            "one_file_at_a_time": True,
            "scratch_is_on_data_not_root": SCRATCH_ROOT.startswith("/data/"),
        },
        "poll_coverage": coverage,
        "observed_span_s": coverage["observed_span_s"],
        "node_wall_s": coverage["node_wall_s"],
        "the_write_unit_is_not_a_variable_here": (
            "review-0253-01 §D measured R0253's proposed remedy and falsified it: "
            "an 8 MiB unit moved the median 11x and the p99 ~700x and the maximum "
            "not at all (1.576/1.793 s at 8 MiB against 1.475/1.567/2.298 s at 64 "
            "MiB under matched load). plan-minilm-100m-v2.md records it. This "
            "round holds the unit at 64 MiB and varies the flush discipline "
            "instead."
        ),
        "what_this_node_measures": (
            "The DISTRIBUTION of the per-write interval under six flush "
            "disciplines, not one maximum, at the one size that breached. The "
            "final fsync is timed as its own named interval in every arm because "
            "review-0253-01 §E found it ranked second in R0253's own census at "
            "0.962761x and absent from R0253's result."
        ),
        "what_this_node_does_not_measure": (
            "vm.dirty_ratio / vm.dirty_bytes tuning, which is global, root-owned "
            "and shared with every other job on this workstation; the settings in "
            "force are read and published instead. And it does not bound the "
            "stall: a maximum over a finite number of draws is a lower bound on "
            "the true maximum and rises with the number of draws."
        ),
    })
    _seal(output, f"{node_id}-write-stall-distribution.json", body)


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0254 round0254_nodes.run_job")
    action = str(job.get("action") or "")
    if action == DISPATCH_ACTION:
        run_dispatch(active, job)
    elif action == WRITEBACK_ACTION:
        run_writeback(active, job)
    else:
        raise Round0254NodeError(f"R{ROUND_ID} does not authorize action {action!r}")


def node_coverage_summary(receipts) -> dict[str, Any]:
    return coverage_summary(receipts)


__all__ = [
    "ACTIONS",
    "DISPATCH_ACTION",
    "PLANTED_DEFECTS",
    "PLANTED_HONEST",
    "REPETITIONS",
    "ROUND_ID",
    "Round0254NodeError",
    "SCRATCH_ROOT",
    "WRITEBACK_ACTION",
    "force_remove_tree",
    "node_coverage_summary",
    "over_the_ceiling",
    "run_dispatch",
    "run_job",
    "run_writeback",
]
