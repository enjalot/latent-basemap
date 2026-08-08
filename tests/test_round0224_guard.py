"""R0224 addendum 2 — the predictive guard, the watchdog, and the ascent.

The first R0224 attempt had an OOM *catch* and no OOM *prediction*. The
`nnd-gd32-igd48-it20` cell at `N=16,000,000` in `materialize` mode reached
`46.7 GB` host RSS, exhausted all `7 GB` of swap, ran `36:54` against `35.47 s`
for the same setting at `8M`, and was SIGKILLed. Because it held a CUDA context
the kill left a UVM teardown thread uninterruptible, RCU deadlocked, PID 1 went
into `D` state, and the box needed a SysRq reboot.

These tests pin the three things that must never regress:

1. the guard **refuses that exact cell before launching it**;
2. the guard does **not** refuse any of the nine cells that already measured
   cleanly, so the fix costs no evidence;
3. a stopped cell is stopped **cooperatively** — SIGTERM into a Python
   exception — and never SIGKILLed as a first resort.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basemap.round0224_cuvs_memory import (  # noqa: E402
    DIMENSION,
    GUARD_DEVICE_BUDGET_BYTES,
    GUARD_HOST_RSS_BUDGET_BYTES,
    GUARD_SWAP_ABORT_BYTES,
    SWEEP_GRAPH_DEGREE,
    SWEEP_INTERMEDIATE_DEGREES,
    guard_decision,
    predict_footprint,
    summarize_sweep,
    sweep_settings,
)
from experiments.round0224_nodes import (  # noqa: E402
    BuildWatchdog,
    _terminate_cooperatively,
    refused_cell,
    run_ascending_sweep,
    skipped_cell,
)

GIB = 1024 ** 3

#: The nine `(rows, igd)` cells that completed in the first attempt. Regression
#: evidence: the guard must still launch every one of them, so the fix costs no
#: measurement. Only the coordinates are listed — the instrument readings live
#: in the sealed build receipts and are never hand-copied into a test.
MEASURED_CELLS = tuple(
    (rows, igd)
    for rows in (2_000_000, 4_000_000, 8_000_000)
    for igd in (48, 96, 128)
)


# --------------------------------------------------------------------------- #
# the predictive guard
# --------------------------------------------------------------------------- #


def test_guard_refuses_the_cell_that_took_the_box_down() -> None:
    """16M x igd48 in materialize mode: refused, with a reason, before launch."""
    decision = guard_decision(
        rows=16_000_000,
        dimension=DIMENSION,
        graph_degree=32,
        intermediate_degree=48,
        dataset_mode="materialize",
    )
    assert decision["refused_a_priori"] is True
    assert decision["allowed"] is False
    assert decision["refusal_reasons"], "a refusal must carry its reason"
    # It is the device budget that catches it: the structural model puts the
    # working set at ~30.5 GiB on a 32 GiB card, which is why the real cell
    # thrashed through UVM for 37 minutes instead of raising a clean OOM.
    assert decision["device_over_budget"] is True


def test_guard_allows_every_cell_that_already_measured_cleanly() -> None:
    """The fix must not cost a single one of the nine existing measurements."""
    for rows, igd in MEASURED_CELLS:
        decision = guard_decision(
            rows=rows,
            dimension=DIMENSION,
            graph_degree=SWEEP_GRAPH_DEGREE,
            intermediate_degree=igd,
            dataset_mode="materialize",
        )
        assert decision["allowed"] is True, (rows, igd, decision["refusal_reasons"])


def test_guard_host_model_reproduces_the_measured_load_footprint() -> None:
    """The `2 x dataset` host term is calibrated, not guessed.

    The first attempt recorded `rss_after_load_bytes = 6,623,383,552` at `2M`
    against a baseline of `479,277,056`. A resident copy alone would be
    `3,072,000,000`; the copy *plus* the file pages it was read from is
    `6,144,000,000`, and `6,144,000,000 + 479,277,056 = 6,623,277,056` — within
    `106,496` bytes of the measurement.
    """
    prediction = predict_footprint(
        rows=2_000_000,
        dimension=DIMENSION,
        graph_degree=SWEEP_GRAPH_DEGREE,
        intermediate_degree=48,
        dataset_mode="materialize",
    )
    measured_after_load = 6_623_383_552
    baseline = 479_277_056
    modelled_load = (
        prediction["resident_copy_bytes"] + prediction["predicted_file_backed_bytes"]
    )
    assert abs((modelled_load + baseline) - measured_after_load) < 1_000_000


def test_guard_ceiling_falls_as_intermediate_degree_rises() -> None:
    """Reachability is igd-ordered: a heavier graph runs out of card sooner."""
    ceilings = {}
    for igd in SWEEP_INTERMEDIATE_DEGREES:
        reachable = [
            rows
            for rows in range(1_000_000, 20_000_001, 1_000_000)
            if guard_decision(
                rows=rows,
                dimension=DIMENSION,
                graph_degree=SWEEP_GRAPH_DEGREE,
                intermediate_degree=igd,
                dataset_mode="memmap",
            )["allowed"]
        ]
        ceilings[igd] = max(reachable)
    assert ceilings[48] > ceilings[96] > ceilings[128], ceilings


def test_guard_refuses_100m_for_every_registered_setting() -> None:
    """A 100M x 384 fp32 substrate is 153.6 GB. Nothing here reaches it."""
    for igd in SWEEP_INTERMEDIATE_DEGREES:
        for mode in ("materialize", "memmap"):
            decision = guard_decision(
                rows=100_000_000,
                dimension=DIMENSION,
                graph_degree=SWEEP_GRAPH_DEGREE,
                intermediate_degree=igd,
                dataset_mode=mode,
            )
            assert decision["refused_a_priori"] is True, (igd, mode)


def test_budgets_are_the_registered_ones() -> None:
    assert GUARD_DEVICE_BUDGET_BYTES == 24 * GIB
    assert GUARD_HOST_RSS_BUDGET_BYTES == 60 * GIB
    assert GUARD_SWAP_ABORT_BYTES == 1 * GIB


# --------------------------------------------------------------------------- #
# a refusal is data
# --------------------------------------------------------------------------- #


def test_a_refusal_is_recorded_as_a_measurement() -> None:
    config = {
        "setting_id": "nnd-gd32-igd48-it20-n16000000",
        "rows": 16_000_000,
        "dimension": DIMENSION,
        "intermediate_graph_degree": 48,
        "graph_degree": 32,
        "max_iterations": 20,
        "metric": "sqeuclidean",
        "dataset_mode": "materialize",
    }
    decision = guard_decision(
        rows=16_000_000,
        dimension=DIMENSION,
        graph_degree=32,
        intermediate_degree=48,
        dataset_mode="materialize",
    )
    cell = refused_cell(config, decision)
    assert cell["fit"] is False
    assert cell["refused_a_priori"] is True
    assert cell["error_type"] == "RefusedAPriori"
    assert cell["guard"]["prediction"]["predicted_device_bytes"] > 0
    summary = summarize_sweep(
        measurements=[cell],
        device_total_bytes=34_359_738_368,
        host_total_bytes=132_000_000_000,
    )
    assert len(summary["refused_cells"]) == 1
    assert summary["refused_cells"][0]["rows"] == 16_000_000
    assert summary["refused_cells"][0]["reasons"]


# --------------------------------------------------------------------------- #
# the ascent
# --------------------------------------------------------------------------- #


def test_ascent_stops_an_igd_at_its_first_failure() -> None:
    """No larger N is attempted for a setting that already failed."""
    attempted: list[tuple[int, int]] = []

    def run_cell(config, setting):
        attempted.append(
            (int(setting["rows"]), int(setting["intermediate_graph_degree"]))
        )
        # igd 128 fails at 4M; igd 48 and 96 always succeed.
        failed = (
            int(setting["intermediate_graph_degree"]) == 128
            and int(setting["rows"]) >= 4_000_000
        )
        return {
            "rows": int(setting["rows"]),
            "intermediate_graph_degree": int(setting["intermediate_graph_degree"]),
            "fit": not failed,
            "error_type": "MemoryError" if failed else None,
        }

    results = run_ascending_sweep(
        settings=sweep_settings(),
        make_config=lambda setting: dict(
            setting, setting_id=setting["id"], dimension=DIMENSION
        ),
        run_cell=run_cell,
    )
    # igd128 was attempted at 2M and 4M and never above.
    igd128_attempted = sorted(rows for rows, igd in attempted if igd == 128)
    assert igd128_attempted == [2_000_000, 4_000_000]
    # The other settings were unaffected by igd128's failure.
    assert max(rows for rows, igd in attempted if igd == 48) == 16_000_000
    # Every registered cell still appears in the record.
    assert len(results) == len(sweep_settings())
    skipped = [item for item in results if item.get("skipped_after_failure_at_smaller_n")]
    assert {int(item["rows"]) for item in skipped} == {8_000_000, 12_000_000, 16_000_000}
    assert all(int(item["intermediate_graph_degree"]) == 128 for item in skipped)


def test_ascent_runs_smallest_n_first() -> None:
    order = [int(item["rows"]) for item in sweep_settings()]
    assert order == sorted(order), "cells must be attempted in ascending N"


def test_a_skipped_cell_states_why() -> None:
    cell = skipped_cell(
        {
            "setting_id": "x",
            "rows": 16_000_000,
            "dimension": DIMENSION,
            "intermediate_graph_degree": 128,
            "graph_degree": 32,
            "max_iterations": 20,
            "metric": "sqeuclidean",
            "dataset_mode": "memmap",
        },
        "igd 128 did not complete at 8,000,000 rows",
    )
    assert cell["fit"] is False
    assert cell["skipped_after_failure_at_smaller_n"] is True
    assert "8,000,000" in cell["skip_reason"]


# --------------------------------------------------------------------------- #
# the watchdog and the cooperative abort
# --------------------------------------------------------------------------- #


def test_watchdog_trips_on_swap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap above the threshold aborts the cell, and SIGTERM is what is sent."""
    import experiments.round0224_nodes as nodes

    signalled: list[tuple[int, int]] = []
    monkeypatch.setattr(nodes, "_nvidia_smi_device_bytes", lambda: 1 * GIB)
    monkeypatch.setattr(nodes, "_nvidia_smi_per_process_bytes", lambda pid: 0)
    monkeypatch.setattr(nodes, "_proc_memory_bytes", lambda pid: (2 * GIB, 1 * GIB))
    monkeypatch.setattr(nodes, "_swap_used_bytes", lambda: 4 * GIB)
    monkeypatch.setattr(
        nodes.os, "kill", lambda pid, sig: signalled.append((pid, sig))
    )

    watchdog = BuildWatchdog(
        pid=os.getpid(),
        poll_s=0.01,
        host_rss_budget_bytes=GUARD_HOST_RSS_BUDGET_BYTES,
        swap_abort_bytes=GUARD_SWAP_ABORT_BYTES,
        device_baseline_bytes=0,
    )
    watchdog.start()
    deadline = time.time() + 5.0
    while watchdog.abort_reason is None and time.time() < deadline:
        time.sleep(0.01)
    watchdog.stop()
    watchdog.join(timeout=5)

    assert watchdog.abort_reason is not None
    assert "swap" in watchdog.abort_reason
    assert signalled and signalled[0][1] == signal.SIGTERM
    assert "SIGKILL" not in " ".join(watchdog.escalations)
    readings = watchdog.readings()
    assert readings["watchdog_aborted"] is True
    assert readings["system_swap_peak_bytes"] == 4 * GIB
    assert readings["device_wide_peak_bytes"] == 1 * GIB


def test_watchdog_trips_on_host_rss(monkeypatch: pytest.MonkeyPatch) -> None:
    import experiments.round0224_nodes as nodes

    signalled: list[tuple[int, int]] = []
    monkeypatch.setattr(nodes, "_nvidia_smi_device_bytes", lambda: 0)
    monkeypatch.setattr(nodes, "_nvidia_smi_per_process_bytes", lambda pid: 0)
    monkeypatch.setattr(
        nodes, "_proc_memory_bytes", lambda pid: (70 * GIB, 70 * GIB)
    )
    monkeypatch.setattr(nodes, "_swap_used_bytes", lambda: 0)
    monkeypatch.setattr(
        nodes.os, "kill", lambda pid, sig: signalled.append((pid, sig))
    )

    watchdog = BuildWatchdog(
        pid=os.getpid(),
        poll_s=0.01,
        host_rss_budget_bytes=GUARD_HOST_RSS_BUDGET_BYTES,
        swap_abort_bytes=GUARD_SWAP_ABORT_BYTES,
        device_baseline_bytes=0,
    )
    watchdog.start()
    deadline = time.time() + 5.0
    while watchdog.abort_reason is None and time.time() < deadline:
        time.sleep(0.01)
    watchdog.stop()
    watchdog.join(timeout=5)

    assert watchdog.abort_reason is not None and "RSS" in watchdog.abort_reason
    assert signalled and signalled[0][1] == signal.SIGTERM


def test_watchdog_does_not_trip_when_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    import experiments.round0224_nodes as nodes

    monkeypatch.setattr(nodes, "_nvidia_smi_device_bytes", lambda: 8 * GIB)
    monkeypatch.setattr(nodes, "_nvidia_smi_per_process_bytes", lambda pid: 0)
    monkeypatch.setattr(nodes, "_proc_memory_bytes", lambda pid: (10 * GIB, 5 * GIB))
    monkeypatch.setattr(nodes, "_swap_used_bytes", lambda: 0)

    watchdog = BuildWatchdog(
        pid=os.getpid(),
        poll_s=0.01,
        host_rss_budget_bytes=GUARD_HOST_RSS_BUDGET_BYTES,
        swap_abort_bytes=GUARD_SWAP_ABORT_BYTES,
        device_baseline_bytes=2 * GIB,
    )
    watchdog.start()
    time.sleep(0.2)
    watchdog.stop()
    watchdog.join(timeout=5)

    assert watchdog.abort_reason is None
    assert watchdog.escalations == []
    readings = watchdog.readings()
    assert readings["device_wide_peak_over_baseline_bytes"] == 6 * GIB
    assert readings["watchdog_samples"] > 1


def test_watchdog_samples_are_not_gil_starved(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point of moving the sampler into the parent.

    The in-process 5 ms sampler took 1-2 samples per build because
    `nn_descent.build` holds the GIL. The parent samples a *different* process
    while blocked in `communicate()`, which releases the GIL, so it keeps
    sampling for the entire build.
    """
    import experiments.round0224_nodes as nodes

    monkeypatch.setattr(nodes, "_nvidia_smi_device_bytes", lambda: 1 * GIB)
    monkeypatch.setattr(nodes, "_nvidia_smi_per_process_bytes", lambda pid: 0)
    monkeypatch.setattr(nodes, "_proc_memory_bytes", lambda pid: (1 * GIB, 1 * GIB))
    monkeypatch.setattr(nodes, "_swap_used_bytes", lambda: 0)

    watchdog = BuildWatchdog(
        pid=os.getpid(),
        poll_s=0.01,
        host_rss_budget_bytes=GUARD_HOST_RSS_BUDGET_BYTES,
        swap_abort_bytes=GUARD_SWAP_ABORT_BYTES,
        device_baseline_bytes=0,
    )
    watchdog.start()
    # Hold the GIL in a way a C extension would not: a tight pure-Python loop
    # still yields, but a blocking sleep in the main thread is the honest
    # analogue of the parent waiting on `communicate()`.
    time.sleep(0.5)
    watchdog.stop()
    watchdog.join(timeout=5)
    assert watchdog.samples > 10, watchdog.samples


def test_cooperative_termination_sends_sigterm_before_sigkill() -> None:
    """A process holding a CUDA context is never SIGKILLed as a first resort."""
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import signal, sys, time
                def handler(signum, frame):
                    # Unwind, the way the build script does.
                    sys.exit(17)
                signal.signal(signal.SIGTERM, handler)
                time.sleep(60)
                """
            ),
        ]
    )
    time.sleep(0.5)
    escalations: list[str] = []
    _terminate_cooperatively(child, escalations)
    assert child.returncode == 17, "the child must exit through its own handler"
    assert escalations == ["SIGTERM"]
    assert not any("SIGKILL" in item for item in escalations)


def test_build_script_turns_sigterm_into_a_python_exception() -> None:
    """The child-side half of the cooperative abort."""
    script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "basemap",
        "round0224_cuvs_memory_build.py",
    )
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import runpy, sys, os, signal, time
                module = runpy.run_path({script!r})
                module["_install_sigterm_handler"]()
                try:
                    time.sleep(30)
                except module["CooperativeAbort"] as exc:
                    print("raised:" + type(exc).__name__)
                    sys.exit(23)
                sys.exit(1)
                """
            ),
            ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(1.0)
    child.terminate()
    stdout, _stderr = child.communicate(timeout=30)
    assert child.returncode == 23, stdout
    assert "raised:CooperativeAbort" in stdout
