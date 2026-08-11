#!/usr/bin/env python3
"""R0253 node handlers — install the stop hook, poll the write path, count coverage.

review-0252-01 left three things that a multi-hour 100M training node needs and
does not have.  This round does all three and measures the result.

* `hookinstall_0253` (CPU) — §A.  `set_abort_poll` was called from exactly one
  file in the whole repository: R0252's own measurement node.  This node runs
  through the real runner entry path with the install now sitting as the first
  statement of every ladder-train node entry, publishes what is installed at
  handler entry, audits the release's own AST for every registered entry, and
  enumerates every `run_*` entry in `experiments/` that can bind at all.
* `writepath_0253` (CPU) — §B, the severe one.  `_write_sized_file` had no abort
  read and the gate was not constructed until after it returned, so R0252's own
  cooperative stop took `93.135091` s `= 37.09x` the ceiling and no census saw
  it.  This node measures the write path at four real sizes up to
  `153,600,000,000` B in both arms, times a stop planted mid-write, and audits
  the release for siblings — every raw write loop, and everything that runs
  before a gate exists.
* `cudasetup_0253` (GPU) — §D.  A node holding a live CUDA context, doing a
  100M-rung setup on a real `153,600,000,000` B substrate-sized file: the
  integrity hash, the read-only memmap open, a sampler-order gather and the model
  allocation, with its own setup gap measured at its own rung.  This is the case
  that cannot be signalled, which is why it is the case that matters.

Every node emits `observed_span_s` at the top level of its artifact, so
`roundreport`'s poll-coverage table prints a real percentage instead of
`not instrumented` — the blind spot closed at the source rather than at the tool.

No node writes the runner's abort flag; every stop-latency control writes its own
flag path.  No signal is delivered on any path.
"""
from __future__ import annotations

import ast
import gc
import json
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from basemap import artifact_identity
from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import create_fresh_directory
from basemap.round0238_rung5 import json_safe
from basemap.round0242_locality import io_counters, json_scrub
from basemap.round0251_trainer_setup import PollRecorder
from basemap.round0217_minilm_2m_seed_family import (
    train_config as r0217_train_config,
)
from basemap.round0250_seed_extension_n16 import ROWS as R0217_ROWS
from basemap.round0250_trainer_loops import short_rung_config
from basemap.round0252_stoppability import (
    STOP_CONTROL_DELAY_S,
    STOP_CONTROL_UPDATES,
    TARGET_DIMENSION,
    TARGET_ROWS,
    TARGET_SUBSTRATE_BYTES,
    gap_reduction,
    gap_report,
    measure_stop_latency,
    size_law,
)
from basemap.round0253_coverage import CoverageLedger, coverage_summary
from basemap.round0253_stop_hooks import (
    CUDA_CAPABILITY,
    CUDA_SCHEMA,
    INSTALL_CAPABILITY,
    INSTALL_SCHEMA,
    NOT_A_FAMILY_CELL,
    REGISTERED_ENTRIES,
    THE_INSTRUMENT_IS_DEFEATABLE,
    WRITE_CAPABILITY,
    WRITE_SCHEMA,
    Round0253Error,
    assert_every_entry_installs,
    assert_stop_hooks_installed,
    install_stop_hooks,
    over_the_ceiling,
    reachability_audit,
    registered_ceiling_s,
    stop_hooks_state,
)
from basemap.round0253_write_path import (
    ARM_POLLED,
    ARM_UNPOLLED,
    assert_write_loop_polls,
    write_sized_file,
)
from experiments.round0113_nodes import _new_model
from experiments.round0230_nodes import _open_substrate, _sealed_graph
from experiments.round0251_nodes import (
    _guard_tail_reported,
    _node_gate,
    _node_guard,
    _receipt_envelope as _r0251_envelope,
    _score_gate_without_raising,
    _seal,
    _start_node,
)


ROUND_ID = "0253"

INSTALL_ACTION = "round0253_stop_hook_install"
WRITE_ACTION = "round0253_polled_write_path"
CUDA_ACTION = "round0253_cuda_holding_setup_gap"
ACTIONS: tuple[str, ...] = (INSTALL_ACTION, WRITE_ACTION, CUDA_ACTION)

#: Scratch for the 100M-scale files.  On `/data`, never `/`, and removed by the
#: node whether it succeeds or fails.
SCRATCH_ROOT = "/data/tmp/round0253-scratch"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS_DIR = os.path.join(REPO_ROOT, "experiments")
BASEMAP_DIR = os.path.join(REPO_ROOT, "basemap")

#: Sizes the write path is measured at, in bytes.  The last is the real target.
WRITE_LADDER_BYTES: tuple[int, ...] = (
    3_072_000_000,
    12_288_000_000,
    49_152_000_000,
    TARGET_SUBSTRATE_BYTES,
)

#: Rows the CUDA-holding node gathers from the 100M-row memmap, in sampler order.
CUDA_GATHER_ROWS = 200_000
CUDA_GATHER_BATCH = 4_096


class Round0253NodeError(RuntimeError):
    """The registered R0253 node contract changed."""


def _receipt_envelope(manifest: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(_r0251_envelope(manifest))
    body["round_id"] = ROUND_ID
    return body


def _evict(path: str) -> dict[str, Any]:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return {
        "path": path,
        "bytes": os.path.getsize(path),
        "method": "posix_fadvise(POSIX_FADV_DONTNEED) on the measured file itself",
        "note": (
            "advisory: 'cold' means 'asked to be cold'. A warmer cache makes the "
            "read faster, so an un-evicted arm UNDER-states the unpolled interval."
        ),
    }


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


# --------------------------------------------------------------------------- #
# A. the install
# --------------------------------------------------------------------------- #


def _cooperative_stop_is_observable_here(*, scratch: str) -> dict[str, Any]:
    """Plant a flag on the INSTALLED hook's own path and time the observation.

    Not a re-measurement of R0252's mid-hash latency: this proves that the hook
    the ladder-train entries now install is the one that reads the runner's flag,
    by pointing `ROUNDRUN_ABORT_FLAG` at a control path this function owns,
    hashing a small real file through the installed hook, and requiring the raise
    to come out of `expected_input_signature`.
    """
    from basemap.round0253_stop_hooks import Round0253CooperativeAbort

    probe = os.path.join(scratch, "install-probe.bin")
    write_sized_file(probe, 256 << 20, seed=20260812, poll=None, digest=False)
    flag = os.path.join(scratch, "install-probe.abort")
    previous_flag = os.environ.get("ROUNDRUN_ABORT_FLAG")
    observed: dict[str, Any] = {}
    try:
        os.environ["ROUNDRUN_ABORT_FLAG"] = flag
        with open(f"{flag}.staging", "w", encoding="utf-8") as handle:
            handle.write('{"reason":"R0253 install probe"}\n')
            handle.flush()
            os.fsync(handle.fileno())
        planted = time.monotonic()
        os.replace(f"{flag}.staging", flag)
        raised = None
        try:
            expected_input_signature(probe)
        except Round0253CooperativeAbort as error:
            raised = f"{type(error).__name__}: {error}"
        latency = time.monotonic() - planted
        observed = {
            "probe_bytes": os.path.getsize(probe),
            "the_installed_hook_raised_inside_expected_input_signature": raised is not None,
            "raised": raised,
            "observation_latency_s": latency,
            "observation_latency_over_the_ceiling": over_the_ceiling(latency),
            "signal_delivered": False,
            "flag_path": flag,
            "this_flag_is_not_the_runner_s": True,
        }
    finally:
        if previous_flag is None:
            os.environ.pop("ROUNDRUN_ABORT_FLAG", None)
        else:
            os.environ["ROUNDRUN_ABORT_FLAG"] = previous_flag
        for path in (flag, f"{flag}.staging", probe):
            if os.path.exists(path):
                os.unlink(path)
    if not observed["the_installed_hook_raised_inside_expected_input_signature"]:
        raise Round0253NodeError(
            "R0253: the installed hook did not observe a planted abort inside "
            "expected_input_signature; the install is decorative"
        )
    return observed


def run_install(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    """§A. Prove the stop hook is installed on the path a ladder train runs."""
    installed = install_stop_hooks(label="R0253 hookinstall node")
    # `install_stop_hooks` returns the PREVIOUS hook per module, read before it
    # replaced anything, so the state at handler entry is recovered from the
    # install itself rather than from a separate read that would have to run
    # before it -- and the install must be the first statement, because that is
    # what `entry_install_audit` requires of every registered entry.
    entry_state = {
        "schema": "round0253-stop-hook-state-at-handler-entry-v1",
        "previous_hook_by_module": dict(installed["previous_hook_by_module"]),
        "every_hooked_module_was_unhooked_at_handler_entry": bool(
            installed["the_process_started_unhooked"]
        ),
        "why_this_is_read_from_the_install": (
            "The runner dispatches this handler directly; nothing between "
            "interpreter start and this line touches `set_abort_poll`. A "
            "separate pre-install read would itself have to precede the install "
            "and would fail the round's own first-statement audit."
        ),
    }
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0253NodeError("R0253 install handler received another queue")
    node_id = str(active.get("node_id") or "hookinstall_0253")
    ledger = CoverageLedger(node=node_id)
    label = "R0253 ladder-node stop-hook install"
    abort_flag = _start_node(label)
    output = create_fresh_directory(str(job["outputs"][0]), label="R0253 install")
    scratch = create_fresh_directory(SCRATCH_ROOT, label="R0253 install scratch")

    try:
        audit = assert_every_entry_installs()
        census = reachability_audit(EXPERIMENTS_DIR)
        after_install = assert_stop_hooks_installed(label=label)
        probe = _cooperative_stop_is_observable_here(scratch=scratch)

        guard = _node_guard(label)
        gate = _node_gate(label, training_performed=False)
        window = ledger.window("R0253 install verification stage")
        with guard:
            gate.start()
            recorder = PollRecorder(gate=gate, clock=time.monotonic)
            recorder.anchor("R0253 install stage entered")
            wrapped = window.wrap(recorder)
            wrapped("R0253 entry audit complete")
            previous = artifact_identity.set_abort_poll(wrapped)
            try:
                # A real hash of a real file, through the very hook the ladder
                # entries install, so this stage's coverage is not a formality.
                sample = os.path.join(scratch, "install-sample.bin")
                write_sized_file(sample, 1 << 30, seed=20260813, poll=wrapped,
                                 digest=False)
                _evict(sample)
                sample_signature = expected_input_signature(sample)
                os.unlink(sample)
            finally:
                artifact_identity.set_abort_poll(previous)
            wrapped("R0253 install stage complete")
            gate.finish("R0253 install stage end")
        window.close()
        tail = _guard_tail_reported(guard, label=label)
        scored = _score_gate_without_raising(gate, tail, label=label)
        report = gap_report(recorder.records, arm="polled")
    finally:
        force_remove_tree(scratch)

    coverage = ledger.receipt()
    body = dict(_receipt_envelope(active["manifest"]))
    body.update({
        "schema": INSTALL_SCHEMA,
        "capability": INSTALL_CAPABILITY,
        "capabilities": [INSTALL_CAPABILITY],
        "abort_flag_precondition": abort_flag,
        "node": node_id,
        "gate_registered": False,
        "is_a_family_cell": False,
        "not_a_family_cell": NOT_A_FAMILY_CELL,
        "the_instrument_is_defeatable": THE_INSTRUMENT_IS_DEFEATABLE,
        "stop_hook_state_at_handler_entry": entry_state,
        "stop_hook_install": {
            key: value for key, value in installed.items() if key != "poll"
        },
        "stop_hook_state_after_install": after_install,
        "registered_entries": [
            {"module": module, "function": function}
            for module, function in REGISTERED_ENTRIES
        ],
        "entry_install_audit": audit,
        "binding_entry_census": census,
        "the_installed_hook_observes_a_planted_abort": probe,
        "sample_hash_through_the_installed_hook": sample_signature,
        "gap_report": report,
        "enforcement_poll_spacing": scored,
        "guard_tail": tail,
        "poll_coverage": coverage,
        "observed_span_s": coverage["observed_span_s"],
        "node_wall_s": coverage["node_wall_s"],
        "what_this_node_proves": (
            "The runner dispatched this handler through its own `--node` entry "
            "path. `stop_hook_state_at_handler_entry` was read as the FIRST "
            "statement of `run_install`, before the install line, and shows the "
            "process starts unhooked -- so the install is doing work rather than "
            "restating a default. `entry_install_audit` then reads the release's "
            "AST and requires the install to be the FIRST statement of every "
            "registered ladder-train entry, and "
            "`the_installed_hook_observes_a_planted_abort` requires the installed "
            "hook to raise out of a real `expected_input_signature` call."
        ),
        "what_this_node_does_not_prove": (
            "That every entry in `binding_entry_census` is covered: it is not, "
            "and the count of those that are not is published. That an entry "
            "reaching a binding call through a helper in another module is "
            "covered: the census counts direct calls only and is a lower bound. "
            "And it does not make the AbortPollGate verdicts sound -- "
            "review-0249-01 §B.1/§B.2 remain open."
        ),
    })
    _seal(output, f"{node_id}-stop-hook-install.json", body)


# --------------------------------------------------------------------------- #
# B. the write path
# --------------------------------------------------------------------------- #


def _write_arm(*, arm: str, path: str, size: int, label: str,
               ledger: CoverageLedger) -> dict[str, Any]:
    """One sized write, with or without the between-block abort read."""
    guard = _node_guard(f"{label} {arm}")
    gate = _node_gate(f"{label} {arm}", training_performed=False)
    window = ledger.window(f"{label} {arm} {size}")
    with guard:
        gate.start()
        recorder = PollRecorder(gate=gate, clock=time.monotonic)
        recorder.anchor("R0253 write stage entered")
        wrapped = window.wrap(recorder)
        created = write_sized_file(
            path, size, seed=20260811 + (size % 9973),
            poll=(wrapped if arm == ARM_POLLED else None),
        )
        wrapped("R0253 file written")
        gate.finish(f"R0253 {arm} write stage end")
    window.close()
    tail = _guard_tail_reported(guard, label=f"{label} {arm}")
    scored = _score_gate_without_raising(gate, tail, label=f"{label} {arm}")
    report = gap_report(recorder.records, arm=arm)
    return {
        "arm": arm,
        "creation": created,
        "gap_report": report,
        "enforcement_poll_spacing": scored,
        "guard_tail": tail,
        "coverage_window": window.receipt(),
    }


def _raw_write_loop_census() -> dict[str, Any]:
    """Every raw `os.write` / `.write(` loop in the release, and whether it polls.

    review-0252-01 §H names one un-polled write path.  "Anything that runs before
    the gate exists is the same class", so this enumerates the class instead of
    fixing only the named member: for every function in `basemap/` and
    `experiments/` that issues a write inside a loop, does that loop also contain
    a call to a poll?
    """
    rows: list[dict[str, Any]] = []
    for directory in (BASEMAP_DIR, EXPERIMENTS_DIR):
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".py"):
                continue
            path = os.path.join(directory, name)
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    tree = ast.parse(handle.read(), filename=path)
            except (SyntaxError, OSError):
                continue
            for function in ast.walk(tree):
                if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for loop in ast.walk(function):
                    if not isinstance(loop, (ast.While, ast.For)):
                        continue
                    writes = [
                        inner for inner in ast.walk(loop)
                        if isinstance(inner, ast.Call)
                        and isinstance(inner.func, ast.Attribute)
                        and inner.func.attr in {"write", "tofile"}
                    ]
                    if not writes:
                        continue
                    polls = [
                        inner for inner in ast.walk(loop)
                        if isinstance(inner, ast.Call)
                        and (
                            (isinstance(inner.func, ast.Name)
                             and inner.func.id in {"poll", "_poll_abort", "wrapped", "recorder"})
                            or (isinstance(inner.func, ast.Attribute)
                                and inner.func.attr in {"anchor", "observe", "_poll_abort"})
                        )
                    ]
                    rows.append({
                        "file": os.path.relpath(path, REPO_ROOT),
                        "function": function.name,
                        "loop_lineno": int(loop.lineno),
                        "write_calls": len(writes),
                        "poll_calls_in_the_same_loop": len(polls),
                        "the_loop_polls": bool(polls),
                    })
    unpolled = [row for row in rows if not row["the_loop_polls"]]
    return {
        "schema": "round0253-raw-write-loop-census-v1",
        "write_loops_found": len(rows),
        "write_loops_that_poll": len(rows) - len(unpolled),
        "write_loops_that_do_not_poll": len(unpolled),
        "unpolled_loops": unpolled[:60],
        "unpolled_loops_truncated_at": 60,
        "what_this_is": (
            "A syntactic census, not a proof of safety. A loop that writes a few "
            "hundred bytes of JSON is in this list and is harmless; the class that "
            "matters is a loop whose total is gigabytes. The census is published "
            "so the residual is a number, and the sizes below say which members "
            "of it can actually breach the ceiling."
        ),
    }


def _pre_gate_audit() -> dict[str, Any]:
    """What runs in a registered entry BEFORE its first gate exists.

    R0252's severe defect was not only that `_write_sized_file` never polled: it
    was that it ran before `_node_gate` was constructed, so even a poll would have
    had nothing to report to. With the install now the first statement of each
    entry, the pre-gate region is hooked; this audit says how large that region
    is per entry and what it contains.
    """
    rows: list[dict[str, Any]] = []
    for module_name, function_name in REGISTERED_ENTRIES:
        module = __import__(module_name, fromlist=["*"])
        source_path = getattr(module, "__file__", None)
        if not source_path:
            continue
        with open(source_path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=source_path)
        target = next(
            (node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             and node.name == function_name),
            None,
        )
        if target is None:
            continue
        gate_lineno = None
        for node in ast.walk(target):
            if isinstance(node, ast.Call):
                name = (
                    node.func.id if isinstance(node.func, ast.Name)
                    else node.func.attr if isinstance(node.func, ast.Attribute)
                    else None
                )
                if name in {"_node_gate", "AbortPollGate"}:
                    gate_lineno = int(node.lineno)
                    break
        install_lineno = None
        for node in ast.walk(target):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id == "install_stop_hooks":
                install_lineno = int(node.lineno)
                break
        rows.append({
            "entry": f"{module_name}.{function_name}",
            "function_lineno": int(target.lineno),
            "install_lineno": install_lineno,
            "first_gate_lineno": gate_lineno,
            "the_entry_has_no_gate_at_all": gate_lineno is None,
            "pre_gate_source_lines": (
                None if gate_lineno is None or install_lineno is None
                else max(0, gate_lineno - install_lineno)
            ),
            "the_hook_is_installed_before_the_gate": (
                install_lineno is not None
                and (gate_lineno is None or install_lineno < gate_lineno)
            ),
        })
    return {
        "schema": "round0253-pre-gate-audit-v1",
        "entries": rows,
        "entries_where_the_hook_precedes_the_gate": sum(
            1 for row in rows if row["the_hook_is_installed_before_the_gate"]
        ),
        "entries_with_no_gate_at_all": sum(
            1 for row in rows if row["the_entry_has_no_gate_at_all"]
        ),
        "why_this_matters": (
            "An abort read is only useful if something is there to read. R0252's "
            "gate was constructed after the write, so the 144.50039366001147 s the "
            "write took could not appear in any gap series even in principle. "
            "Installing the hook as the entry's first statement makes the whole "
            "body pollable, including the pre-gate region; a gate is still needed "
            "for the ceiling VERDICT, but not for the node to stop."
        ),
    }


def run_writepath(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    """§B. Measure the un-polled write path, then poll it, then re-measure."""
    install_stop_hooks(label="R0253 writepath node")
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0253NodeError("R0253 write handler received another queue")
    node_id = str(active.get("node_id") or "writepath_0253")
    ledger = CoverageLedger(node=node_id)
    label = "R0253 polled write path"
    abort_flag = _start_node(label)
    output = create_fresh_directory(str(job["outputs"][0]), label="R0253 write")
    guard_receipt = assert_write_loop_polls()
    sizes = [int(value) for value in job["write_ladder_bytes"]]
    if TARGET_SUBSTRATE_BYTES not in sizes:
        raise Round0253NodeError(
            "R0253 write node must measure the 100M substrate size itself, not "
            f"project to it: {TARGET_SUBSTRATE_BYTES} not in {sizes}"
        )

    io_before = io_counters()
    scratch = create_fresh_directory(SCRATCH_ROOT, label="R0253 write scratch")
    ladder: list[dict[str, Any]] = []
    stop_controls: list[dict[str, Any]] = []
    try:
        for size in sizes:
            path = os.path.join(scratch, f"write-{size}.bin")
            unpolled = _write_arm(arm=ARM_UNPOLLED, path=path, size=size,
                                  label=label, ledger=ledger)
            os.unlink(path)
            polled = _write_arm(arm=ARM_POLLED, path=path, size=size,
                                label=label, ledger=ledger)
            if (unpolled["creation"]["content_sha256"]
                    != polled["creation"]["content_sha256"]):
                raise Round0253NodeError(
                    f"R0253 write rung {size}: the polled arm wrote different "
                    "bytes from the unpolled arm"
                )
            ladder.append({
                "rung": f"write_{size}",
                "bytes": size,
                "arms": {ARM_UNPOLLED: unpolled, ARM_POLLED: polled},
                "the_two_arms_wrote_identical_bytes": True,
                "reduction": gap_reduction(
                    before=unpolled["gap_report"], after=polled["gap_report"]
                ),
            })
            os.unlink(path)

            if size == TARGET_SUBSTRATE_BYTES:
                control_path = os.path.join(scratch, "write-stop-control.bin")

                def run_write_under_poll(poll, _path=control_path, _size=size):
                    try:
                        write_sized_file(_path, _size, seed=20260814, poll=poll,
                                         digest=False)
                    finally:
                        if os.path.exists(_path):
                            os.unlink(_path)

                outcome = measure_stop_latency(
                    label="write of a 100M-scale substrate",
                    flag_path=os.path.join(scratch, "write-stop.abort"),
                    delay_s=STOP_CONTROL_DELAY_S,
                    run=run_write_under_poll,
                )
                # The round's abort rule: a stop control that does not stop is a
                # HARD failure of its node, not a null in a table.
                if not outcome["the_work_stopped_cooperatively"]:
                    raise Round0253NodeError(
                        "R0253 write stop control did not stop: the polled write "
                        "path ran to completion with the flag planted"
                    )
                stop_controls.append(outcome)
    finally:
        force_remove_tree(scratch)
    io_after = io_counters()

    at_target = next(entry for entry in ladder if entry["bytes"] == TARGET_SUBSTRATE_BYTES)
    polled_points = [
        {"bytes": entry["bytes"],
         "widest_gap_s": entry["arms"][ARM_POLLED]["gap_report"]["widest_gap_s"]}
        for entry in ladder
    ]
    unpolled_points = [
        {"bytes": entry["bytes"],
         "widest_gap_s": entry["arms"][ARM_UNPOLLED]["gap_report"]["widest_gap_s"]}
        for entry in ladder
    ]

    coverage = ledger.receipt()
    body = dict(_receipt_envelope(active["manifest"]))
    body.update({
        "schema": WRITE_SCHEMA,
        "capability": WRITE_CAPABILITY,
        "capabilities": [WRITE_CAPABILITY],
        "abort_flag_precondition": abort_flag,
        "node": node_id,
        "gate_registered": False,
        "is_a_family_cell": False,
        "not_a_family_cell": NOT_A_FAMILY_CELL,
        "the_instrument_is_defeatable": THE_INSTRUMENT_IS_DEFEATABLE,
        "write_loop_poll_guard": guard_receipt,
        "registered_ceiling_s": registered_ceiling_s(),
        "ladder": ladder,
        "at_the_100m_size": {
            "unpolled_widest_gap_s": at_target["arms"][ARM_UNPOLLED]["gap_report"]["widest_gap_s"],
            "unpolled_widest_gap_over_the_ceiling": at_target["arms"][ARM_UNPOLLED]["gap_report"]["widest_gap_over_the_ceiling"],
            "polled_widest_gap_s": at_target["arms"][ARM_POLLED]["gap_report"]["widest_gap_s"],
            "polled_widest_gap_over_the_ceiling": at_target["arms"][ARM_POLLED]["gap_report"]["widest_gap_over_the_ceiling"],
            "reduction": at_target["reduction"],
        },
        "polled_size_law": size_law(polled_points),
        "unpolled_size_law": size_law(unpolled_points),
        "stop_latency": stop_controls,
        "raw_write_loop_census": _raw_write_loop_census(),
        "pre_gate_audit": _pre_gate_audit(),
        "io_counters": {"before": io_before, "after": io_after},
        "poll_coverage": coverage,
        "observed_span_s": coverage["observed_span_s"],
        "node_wall_s": coverage["node_wall_s"],
        "what_r0252_reported_here": (
            "Nothing. review-0252-01 §H recovered `144.50039366001147` s = "
            "`57.55x` the registered ceiling from R0252's own sealed "
            "`write_wall_s`, and `93.135091` s = `37.09x` from the timestamps of "
            "its attempt-1 cooperative stop. Neither number appears in R0252's "
            "round file, its result or any census, because code that never polls "
            "emits no gap."
        ),
        "the_unpolled_arm_is_r0252_s_writer": (
            "Same 64 MiB pseudorandom block, same os.write loop, same fsync + "
            "POSIX_FADV_DONTNEED every 2 GiB, no fallocate and no ftruncate. The "
            "two arms are required to produce the identical content digest at "
            "every rung, which is the receipt that the poll changed no byte."
        ),
    })
    _seal(output, f"{node_id}-polled-write-path.json", body)


# --------------------------------------------------------------------------- #
# C. the setup gap on a node that is holding a live CUDA context
# --------------------------------------------------------------------------- #


def _cuda_maps_evidence() -> dict[str, Any]:
    """The PRESENCE rule, applied to this process: does it map a CUDA library?"""
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
        "schema": "round0253-cuda-presence-v1",
        "pid": os.getpid(),
        "mapped_cuda_objects": sorted(matches)[:20],
        "this_process_maps_a_cuda_library": bool(matches),
        "rule": (
            "PRESENCE, not a count (plan-minilm-100m-v2.md). Any non-zero match "
            "means this process must never be signalled. It is recorded here "
            "because a CUDA-holding node is exactly the case a terminal cannot "
            "stop, which is why its setup gap is the one that matters."
        ),
    }


def run_cudasetup(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    """§D. The 100M-rung setup gap, measured on a node holding a CUDA context."""
    install_stop_hooks(label="R0253 cudasetup node")
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0253NodeError("R0253 cuda handler received another queue")
    node_id = str(active.get("node_id") or "cudasetup_0253")
    ledger = CoverageLedger(node=node_id)
    label = "R0253 CUDA-holding 100M setup gap"
    abort_flag = _start_node(label)
    output = create_fresh_directory(str(job["outputs"][0]), label="R0253 cuda")

    import torch

    if not torch.cuda.is_available():
        raise Round0253NodeError("R0253 cuda node requires a live CUDA device")
    resident = torch.zeros(1024, 1024, device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()
    context = {
        "device": torch.cuda.get_device_name(0),
        "resident_tensor_bytes": int(resident.numel() * resident.element_size()),
        "torch_cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated()),
        "maps": _cuda_maps_evidence(),
    }
    if not context["maps"]["this_process_maps_a_cuda_library"]:
        raise Round0253NodeError(
            "R0253 cuda node holds no CUDA mapping; the measurement would not be "
            "the case it claims to be"
        )

    scratch = create_fresh_directory(SCRATCH_ROOT, label="R0253 cuda scratch")
    substrate = os.path.join(scratch, "substrate-100m.f32")
    io_before = io_counters()
    try:
        guard = _node_guard(label)
        gate = _node_gate(label, training_performed=False)
        window = ledger.window("R0253 CUDA-holding 100M setup")
        with guard:
            gate.start()
            recorder = PollRecorder(gate=gate, clock=time.monotonic)
            recorder.anchor("R0253 CUDA-holding setup entered")
            wrapped = window.wrap(recorder)
            previous = artifact_identity.set_abort_poll(wrapped)
            try:
                created = write_sized_file(
                    substrate, TARGET_SUBSTRATE_BYTES, seed=20260815,
                    poll=wrapped, digest=False,
                )
                wrapped("R0253 100M substrate written")

                eviction = _evict(substrate)
                hash_started = time.monotonic()
                signature = expected_input_signature(substrate)
                hash_wall = time.monotonic() - hash_started

                open_started = time.monotonic()
                memmap = np.memmap(
                    substrate, dtype=np.float32, mode="r",
                    shape=(TARGET_ROWS, TARGET_DIMENSION),
                )
                open_wall = time.monotonic() - open_started
                wrapped("R0253 100M read-only memmap opened")

                rng = np.random.default_rng(20260816)
                rows = np.sort(rng.choice(TARGET_ROWS, size=CUDA_GATHER_ROWS,
                                          replace=False))
                gather_started = time.monotonic()
                checksum = 0.0
                for start in range(0, len(rows), CUDA_GATHER_BATCH):
                    wrapped("R0253 sampler-order gather batch")
                    batch = np.asarray(memmap[rows[start:start + CUDA_GATHER_BATCH]],
                                       dtype=np.float32)
                    device_batch = torch.from_numpy(batch).to("cuda",
                                                              non_blocking=False)
                    checksum += float(device_batch.abs().sum().item())
                    del device_batch
                gather_wall = time.monotonic() - gather_started
                torch.cuda.synchronize()
                wrapped("R0253 sampler-order gather complete")

                # The treatment's REAL model, built from R0217's canonical config
                # through the release's own `_new_model`, so "model allocation"
                # is the allocation a training node performs and not a stand-in.
                # Its cost does not depend on N -- it is reported beside the
                # N-dependent stages, never as one of them.
                treatment_started = time.monotonic()
                template_graph = _sealed_graph(job)
                template_source, template_signature = _open_substrate(template_graph)
                template, _template_sha = r0217_train_config(
                    seed=int(job["template_seed"]),
                    graph_signature=template_graph["signature"],
                    graph_manifest_signature=template_graph["manifest_signature"],
                    substrate_signature=template_signature,
                    graph_edges=template_graph["directed_edges"],
                    rows=R0217_ROWS,
                )
                treatment_edges = int(template_graph["directed_edges"])
                del template_source, template_graph
                gc.collect()
                treatment_wall = time.monotonic() - treatment_started
                wrapped("R0253 treatment config resolved")
                model_config, _model_sha, _model_identity = short_rung_config(
                    template, updates=STOP_CONTROL_UPDATES
                )
                model_started = time.monotonic()
                model = _new_model(model_config)
                # `_new_model` constructs the estimator; the network itself is
                # materialised by `_init_model`, which is what `fit()` calls and
                # therefore what "model allocation" means. Nothing is fitted.
                model._init_model(TARGET_DIMENSION)
                torch.cuda.synchronize()
                model_wall = time.monotonic() - model_started
                wrapped("R0253 model allocated on device")
                parameters = int(sum(p.numel() for p in model.model.parameters()))
                parameter_devices = sorted({
                    str(p.device) for p in model.model.parameters()
                })
                del memmap
            finally:
                artifact_identity.set_abort_poll(previous)
            gate.finish("R0253 CUDA-holding setup end")
        window.close()
        tail = _guard_tail_reported(guard, label=label)
        scored = _score_gate_without_raising(gate, tail, label=label)
        report = gap_report(recorder.records, arm="polled")
    finally:
        del resident
        force_remove_tree(scratch)
    io_after = io_counters()

    coverage = ledger.receipt()
    ceiling = registered_ceiling_s()
    body = dict(_receipt_envelope(active["manifest"]))
    body.update({
        "schema": CUDA_SCHEMA,
        "capability": CUDA_CAPABILITY,
        "capabilities": [CUDA_CAPABILITY],
        "abort_flag_precondition": abort_flag,
        "node": node_id,
        "gate_registered": False,
        "is_a_family_cell": False,
        "not_a_family_cell": NOT_A_FAMILY_CELL,
        "the_instrument_is_defeatable": THE_INSTRUMENT_IS_DEFEATABLE,
        "cuda_context": context,
        "registered_ceiling_s": ceiling,
        "target_rows": TARGET_ROWS,
        "target_dimension": TARGET_DIMENSION,
        "target_substrate_bytes": TARGET_SUBSTRATE_BYTES,
        "stages": {
            "write_wall_s": created["write_wall_s"],
            "write_wall_over_the_ceiling": over_the_ceiling(created["write_wall_s"]),
            "hash_wall_s": hash_wall,
            "hash_wall_over_the_ceiling": over_the_ceiling(hash_wall),
            "memmap_open_wall_s": open_wall,
            "memmap_open_wall_over_the_ceiling": over_the_ceiling(open_wall),
            "gather_wall_s": gather_wall,
            "gather_wall_over_the_ceiling": over_the_ceiling(gather_wall),
            "model_allocation_wall_s": model_wall,
            "model_allocation_wall_over_the_ceiling": over_the_ceiling(model_wall),
        },
        "the_2m_rung_stages_measured_here_and_NOT_carried_up": {
            "treatment_graph_load_and_config_wall_s": treatment_wall,
            "treatment_graph_load_and_config_wall_over_the_ceiling": over_the_ceiling(
                treatment_wall
            ),
            "directed_edges": treatment_edges,
            "rows": int(R0217_ROWS),
            "why_it_is_here": (
                "The model this node allocates is the treatment's real model, so "
                "its config has to be resolved, which means loading the sealed 2M "
                "graph. That load is a 2,000,000-row measurement and is reported "
                "as one. It is NOT a 100M graph-load figure and nothing in this "
                "artifact extrapolates it."
            ),
        },
        "creation": created,
        "page_cache": eviction,
        "substrate_signature": signature,
        "gather": {
            "rows_gathered": int(CUDA_GATHER_ROWS),
            "batch_rows": int(CUDA_GATHER_BATCH),
            "readonly_memmap": True,
            "device_checksum": checksum,
            "why_a_memmap": (
                "Read-only np.memmap is a standing SAFETY precondition on this "
                "box, not an optimisation: both GPU wedges were UVM page-fault "
                "deadlocks against anonymous host buffers."
            ),
        },
        "model": {
            "parameters": parameters,
            "parameter_devices": parameter_devices,
            "input_dimension": int(TARGET_DIMENSION),
            "allocated_through": "ParametricUMAP._init_model, the call fit() makes",
            "nothing_was_fitted": True,
        },
        "gap_report": report,
        "enforcement_poll_spacing": scored,
        "guard_tail": tail,
        "io_counters": {"before": io_before, "after": io_after},
        "poll_coverage": coverage,
        "observed_span_s": coverage["observed_span_s"],
        "node_wall_s": coverage["node_wall_s"],
        "what_is_measured_here": (
            "A node holding a live CUDA context -- verified by the PRESENCE rule "
            "on /proc/self/maps, not by an environment variable -- performing a "
            "100M-rung setup on a real 153,600,000,000 B file: the write, the "
            "integrity hash, the read-only memmap open, a sampler-order gather to "
            "device, and the model allocation. Every one of those is polled and "
            "every gap between polls is in `gap_report`."
        ),
        "what_is_NOT_measured_here": (
            "The 100M graph load and edge-list preparation at their own rung. "
            "R0238's 100,000,000-row substrate and R0240's k15 graph no longer "
            "exist on this box, and rebuilding them is ~7 GPU-h against this "
            "round's 3.0 GPU-h cap. Their 2M-rung values -- graph load 0.4017 s, "
            "edge-list preparation 0.0371 s, model allocation 0.0269 s -- are NOT "
            "carried here, and no figure in this artifact is a projection of them. "
            "review-0250-01 and review-0251-01's prerequisite is therefore "
            "PARTIALLY met: the CUDA-holding node and its own-rung setup hash, "
            "memmap and gather are measured; the graph load is not."
        ),
        "no_signal_was_delivered": True,
        "cuvs_calls": 0,
    })
    _seal(output, f"{node_id}-cuda-holding-setup-gap.json", body)


# --------------------------------------------------------------------------- #


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0254 round0253_nodes.run_job")
    action = str(job.get("action") or "")
    if action == INSTALL_ACTION:
        run_install(active, job)
    elif action == WRITE_ACTION:
        run_writepath(active, job)
    elif action == CUDA_ACTION:
        run_cudasetup(active, job)
    else:
        raise Round0253NodeError(f"R{ROUND_ID} does not authorize action {action!r}")


def summarise_coverage(paths: Sequence[str]) -> dict[str, Any]:
    """Fold the three nodes' sealed coverage blocks into one statement."""
    receipts = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            receipts.append(dict(json.load(handle)["poll_coverage"]))
    return coverage_summary(receipts)


__all__ = [
    "ACTIONS",
    "CUDA_ACTION",
    "INSTALL_ACTION",
    "Round0253NodeError",
    "WRITE_ACTION",
    "WRITE_LADDER_BYTES",
    "force_remove_tree",
    "run_cudasetup",
    "run_install",
    "run_job",
    "run_writepath",
    "summarise_coverage",
]
