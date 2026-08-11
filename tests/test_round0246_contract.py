"""R0246 contract — the three closures, and the attacks that must fail.

Every test here plants a defect. review-0245-01 defeated all three of R0245's
guards *outside* their own positive controls, so this file carries the
reviewer's exact shapes first and then a set of novel shapes no control covers.
"""
from __future__ import annotations

import os
import stat
import threading

import numpy as np
import pytest

from basemap import round0244_guard as guard0244
from basemap import round0245_guard as guard0245
from basemap import round0246_guard as guard
from basemap import round0246_tie as tie
from basemap.round0244_prereq import (
    _stable_counting_order,
    two_level_weight_sample,
    weight_block_profile,
)


# --------------------------------------------------------------------------- #
# closure 1 — the OSError arm, the exit sentinel, the coverage floor
# --------------------------------------------------------------------------- #
def test_the_reviewer_oserror_control_fires(tmp_path) -> None:
    """review-0245-01 A1-bis, in its exact shape."""
    evidence = guard.run_reviewer_oserror_control(
        flag_path=str(tmp_path / "oserror.abort"), interval_s=0.02, wall_s=1.0
    )
    assert evidence["holds"] is True
    assert evidence["thread_death"] is not None
    assert evidence["sampling_thread_alive"] is False
    assert evidence["gate_refused_the_receipt"] is True
    #: the tolerance is bounded, not infinite, and the bound is registered
    assert evidence["max_consecutive_sample_failures"] >= (
        guard0244.WATCHDOG_MAX_CONSECUTIVE_SAMPLE_FAILURES
    )


def test_an_intermittent_oserror_below_the_tolerance_does_not_kill_the_thread(
    tmp_path,
) -> None:
    """The tolerance is real: a single failure must not fail a node."""
    watchdog = guard._IntermittentWatchdog(
        fail_of=2, anonymous_budget_bytes=1 << 62, interval_s=0.02,
        abort_flag_path=str(tmp_path / "intermittent.abort"),
        label="intermittent",
    )
    guard._run_watchdog_until(watchdog, wall_s=0.6)
    receipt = watchdog.receipt()
    assert receipt["sampling_thread_alive"] is True
    assert receipt["sample_failures"] > 0
    assert receipt["max_consecutive_sample_failures"] <= (
        guard0244.WATCHDOG_MAX_CONSECUTIVE_SAMPLE_FAILURES
    )


def test_a_sampling_loop_that_simply_returns_is_a_recorded_death(tmp_path) -> None:
    watchdog = guard._SilentReturnWatchdog(
        anonymous_budget_bytes=1 << 62, interval_s=0.02,
        abort_flag_path=str(tmp_path / "silent.abort"), label="silent",
    )
    guard._run_watchdog_until(watchdog, wall_s=0.4)
    receipt = watchdog.receipt()
    assert receipt["thread_death"] is not None
    assert "unrequested thread exit" in receipt["thread_death"]
    with pytest.raises(guard0244.Round0244Error):
        watchdog.poll("after the silent return")


def test_a_normal_stop_is_not_a_death(tmp_path) -> None:
    """The exit sentinel must not fire on the ordinary path."""
    watchdog = guard0245.EnforcedHostWatchdog(
        anonymous_budget_bytes=1 << 62, interval_s=0.02,
        abort_flag_path=str(tmp_path / "normal.abort"), label="normal",
    )
    guard._run_watchdog_until(watchdog, wall_s=0.3)
    receipt = watchdog.receipt()
    assert receipt["thread_death"] is None
    assert receipt["sampling_thread_alive"] is True
    assert receipt["thread_samples"] > 0
    guard.require_live_sampler(receipt, label="normal")


def test_the_coverage_floor_refuses_a_starved_sampler(tmp_path) -> None:
    """The half of review-0244-01 A1's sentence R0245 did not implement."""
    watchdog = guard._StarvedWatchdog(
        stall_s=1.0, anonymous_budget_bytes=1 << 62, interval_s=0.02,
        abort_flag_path=str(tmp_path / "starved.abort"), label="starved",
    )
    guard._run_watchdog_until(watchdog, wall_s=1.2)
    receipt = watchdog.receipt()
    assert receipt["sampling_thread_alive"] is True  # nothing raised
    assert receipt["thread_sample_coverage"] < guard.MIN_THREAD_SAMPLE_COVERAGE
    with pytest.raises(guard.Round0246Error):
        guard.require_live_sampler(receipt, label="starved")


def test_boundary_polls_cannot_hold_the_coverage_figure_up(tmp_path) -> None:
    """`sample_coverage` counts poll() samples; the gate must not read it."""
    watchdog = guard._StarvedWatchdog(
        stall_s=1.0, anonymous_budget_bytes=1 << 62, interval_s=0.02,
        abort_flag_path=str(tmp_path / "inflated.abort"), label="inflated",
    )
    guard._run_watchdog_until(watchdog, wall_s=1.2, poll_every_s=0.002)
    receipt = watchdog.receipt()
    assert receipt["sample_coverage"] > guard.MIN_THREAD_SAMPLE_COVERAGE
    assert receipt["thread_sample_coverage"] < guard.MIN_THREAD_SAMPLE_COVERAGE
    with pytest.raises(guard.Round0246Error):
        guard.require_live_sampler(receipt, label="inflated")


def test_the_coverage_gate_is_scoped_out_of_a_very_short_stage() -> None:
    """Below the registered expected-sample count the ratio is noise."""
    #: R0247 (declared edit): the denominator is the REGISTERED 0.25 s
    #: interval against measured wall time, never the interval the receipt
    #: declares, so these fixtures carry a wall and the expected count is
    #: derived from it.
    short = {
        "sampling_thread_alive": True, "thread_death": None, "samples": 2,
        "thread_samples": 1, "sample_coverage": 0.4,
        "thread_sample_coverage": 0.2, "expected_samples_at_interval": 5.0,
        "boundary_polls": 1, "sampled_wall_s": 0.24,
        "expected_samples_at_the_registered_interval": 0.96,
        "thread_sample_coverage_at_the_registered_interval": 1.04,
        "max_thread_sample_gap_s": 0.24, "mean_thread_sample_gap_s": 0.24,
    }
    assert guard.require_live_sampler(short, label="short")["holds"] is True
    long = dict(
        short, sampled_wall_s=25.0,
        expected_samples_at_the_registered_interval=100.0,
        thread_sample_coverage_at_the_registered_interval=0.01,
        max_thread_sample_gap_s=2.0, mean_thread_sample_gap_s=0.4,
    )
    with pytest.raises(guard.Round0246Error):
        guard.require_live_sampler(long, label="long")


def test_a_receipt_without_thread_coverage_cannot_be_gated_on() -> None:
    with pytest.raises(guard.Round0246Error):
        guard.require_live_sampler(
            {
                "sampling_thread_alive": True, "thread_death": None,
                "samples": 5, "sample_coverage": 1.0,
                "expected_samples_at_interval": 100.0,
            },
            label="old receipt",
        )


# --------------------------------------------------------------------------- #
# closure 2 — validate the flag PATH, not its directory
# --------------------------------------------------------------------------- #
def test_the_reviewer_directory_control_fires(tmp_path) -> None:
    """review-0245-01 A3-bis, in its exact shape."""
    evidence = guard.run_reviewer_directory_flag_control(
        directory=str(tmp_path / "flag-as-a-directory")
    )
    assert evidence["holds"] is True
    assert evidence["precondition_refused_to_start"] is True
    assert evidence["watchdog_refused_to_arm"] is True
    assert evidence["directory_still_exists"] is True


def test_a_writable_path_passes_and_leaves_nothing_behind(
    monkeypatch, tmp_path
) -> None:
    path = tmp_path / "node.abort"
    monkeypatch.setenv("ROUNDRUN_ABORT_FLAG", str(path))
    accepted = guard0245.require_enforceable_abort_flag(label="ok")
    assert accepted["path_is_writable"] is True
    assert accepted["path_writability_probe"][
        "probe_file_created_and_removed"
    ] is True
    assert not path.exists()


def test_a_pending_abort_flag_is_not_destroyed_by_the_probe(
    monkeypatch, tmp_path
) -> None:
    """A real operator abort at the path must survive the precondition."""
    path = tmp_path / "node.abort"
    path.write_text("operator abort", encoding="utf-8")
    monkeypatch.setenv("ROUNDRUN_ABORT_FLAG", str(path))
    accepted = guard0245.require_enforceable_abort_flag(label="pending")
    assert accepted["path_is_writable"] is True
    assert accepted["path_writability_probe"][
        "probe_file_created_and_removed"
    ] is False
    assert path.read_text(encoding="utf-8") == "operator abort"


@pytest.mark.parametrize("shape", ["fifo", "readonly", "dangling", "linked_dir"])
def test_novel_flag_path_shapes_are_refused(monkeypatch, tmp_path, shape) -> None:
    path = tmp_path / f"{shape}.abort"
    if shape == "fifo":
        os.mkfifo(path)
    elif shape == "readonly":
        path.write_text("x", encoding="utf-8")
        os.chmod(path, stat.S_IRUSR)
    elif shape == "dangling":
        os.symlink(tmp_path / "nowhere" / "target", path)
    else:
        (tmp_path / "realdir").mkdir()
        os.symlink(tmp_path / "realdir", path)
    monkeypatch.setenv("ROUNDRUN_ABORT_FLAG", str(path))
    with pytest.raises(guard0245.Round0245Error):
        guard0245.require_enforceable_abort_flag(label=shape)
    with pytest.raises(guard0245.Round0245Error):
        guard0245.EnforcedHostWatchdog(label=shape)
    if shape == "readonly":
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


def test_a_trip_that_does_not_land_fails_the_node_tail() -> None:
    with pytest.raises(guard.Round0246Error):
        guard.require_abort_flag_landed(
            {
                "fired": True, "abort_flag_written": False,
                "abort_flag_error": "[Errno 21] Is a directory",
                "abort_flag_path": "/tmp/adir",
            },
            label="planted",
        )
    assert guard.require_abort_flag_landed(
        {"fired": False, "abort_flag_written": False, "abort_flag_error": None},
        label="never fired",
    )["holds"] is True


# --------------------------------------------------------------------------- #
# closure 3 — the poll-spacing gate can fail again
# --------------------------------------------------------------------------- #
def test_attempt_1s_gap_fails_again() -> None:
    """The single most important assertion in this round."""
    replay = guard.replay_gap_through_the_gate(
        gap_s=guard.ATTEMPT_1_GAP_S,
        headroom_bytes=guard.R0245_NODE_HEADROOM_BYTES,
        measured_slope_bytes_per_s=0.0,
        label="attempt 1 replay",
    )
    assert replay["gate_refused_it"] is True
    assert replay["binding_slope_bytes_per_s"] == (
        guard0245.MIN_BINDING_SLOPE_BYTES_PER_S
    )
    #: and the form R0245 shipped really did pass exactly this gap
    loosened = guard0245.poll_spacing_requirement(
        slope_bytes_per_s=max(0.0, 1.0),
        headroom_bytes=guard.R0245_NODE_HEADROOM_BYTES,
        poll_spacing_s=guard.ATTEMPT_1_GAP_S,
    )
    assert loosened["requirement_holds"] is True
    assert loosened["max_poll_spacing_s"] == float(
        guard.R0245_NODE_HEADROOM_BYTES
    )


def test_the_reviewer_gap_replay_control_fires() -> None:
    evidence = guard.run_reviewer_gap_replay_control()
    assert evidence["holds"] is True
    assert evidence["replay"]["gate_refused_it"] is True
    assert evidence["one_poll_stage"]["gate_refused_it"] is True
    assert evidence["zero_poll_stage"]["gate_refused_it"] is True


def test_the_binding_slope_floor_is_the_measured_worst_case() -> None:
    assert guard0245.MIN_BINDING_SLOPE_BYTES_PER_S == float(
        guard0245.R0244_MEASURED_SLOPE_BYTES_PER_S
    )
    assert guard0245.binding_slope_bytes_per_s(0.0) == (
        guard0245.MIN_BINDING_SLOPE_BYTES_PER_S
    )
    assert guard0245.binding_slope_bytes_per_s(2.0e10) == 2.0e10
    assert guard.R0246_MAX_POLL_SPACING_S == pytest.approx(
        2.5109531834854018
    )


def _scripted_gate(ticks, **kwargs):
    stream = iter(list(ticks))
    #: R0247 (declared edit): a scripted clock and a no-op `inner` are now
    #: construction paths that a node may not use as evidence, so every
    #: scripted gate in this file declares itself a replay. `require()` then
    #: waives the clock and reader arms and applies every substantive one, and
    #: `require_enforcement_evidence()` is what refuses to seal the result.
    return guard.AbortPollGate(
        inner=lambda _where: None, label="scripted", replay=True,
        clock=lambda: next(stream), **kwargs,
    )


def test_the_gate_measures_the_interval_before_the_first_read() -> None:
    gate = _scripted_gate(
        [0.0, 3.0, 3.0, 3.0], headroom_bytes=guard.R0245_NODE_HEADROOM_BYTES
    )
    gate.start()
    gate("first read, 3 s in")
    gate.finish()
    assert gate.gap_before_the_first_poll_s == pytest.approx(3.0)
    with pytest.raises(guard.Round0246Error):
        gate.require(measured_slope_bytes_per_s=0.0)


def test_the_gate_measures_the_interval_after_the_last_read() -> None:
    gate = _scripted_gate(
        [0.0, 0.0, 0.01, 3.0], headroom_bytes=guard.R0245_NODE_HEADROOM_BYTES
    )
    gate.start()
    gate("read 1")
    gate("read 2")
    gate.finish()
    assert gate.gap_after_the_last_poll_s == pytest.approx(2.99)
    with pytest.raises(guard.Round0246Error):
        gate.require(measured_slope_bytes_per_s=0.0)


def test_an_inflated_headroom_cannot_buy_a_wider_spacing() -> None:
    gate = _scripted_gate([0.0, 0.0, 3.0, 3.0], headroom_bytes=1 << 50)
    gate.start()
    gate("read 1")
    gate("read 2")
    gate.finish()
    scored = gate.verdict(measured_slope_bytes_per_s=0.0)
    #: R0247 (declared edit): the declared headroom is itself clamped at the
    #: registered 29,548,888,064 B, so the requirement arm now refuses it too.
    #: review-0246-01 C combined an inflated headroom with an overridden
    #: ceiling; both arms are independent now.
    assert scored["declared_headroom_bytes"] == 1 << 50
    assert scored["effective_headroom_bytes"] == 29_548_888_064
    assert scored["requirement"]["requirement_holds"] is False
    assert scored["meets_the_registered_ceiling"] is False
    with pytest.raises(guard.Round0246Error):
        gate.require(measured_slope_bytes_per_s=0.0)


def test_a_weak_worst_case_slope_is_refused() -> None:
    gate = _scripted_gate(
        [0.0, 0.0, 0.01, 0.02],
        headroom_bytes=guard.R0245_NODE_HEADROOM_BYTES,
        slope_bytes_per_s=1.0,
    )
    gate.start()
    gate("read 1")
    gate("read 2")
    gate.finish()
    with pytest.raises(guard.Round0246Error):
        gate.require(measured_slope_bytes_per_s=0.0)


def test_a_gate_that_was_never_finished_is_refused() -> None:
    gate = _scripted_gate(
        [0.0, 0.0, 0.01], headroom_bytes=guard.R0245_NODE_HEADROOM_BYTES
    )
    gate.start()
    gate("read 1")
    gate("read 2")
    with pytest.raises(guard.Round0246Error):
        gate.require(measured_slope_bytes_per_s=0.0)


def test_a_gate_called_before_start_is_refused() -> None:
    gate = _scripted_gate(
        [0.0], headroom_bytes=guard.R0245_NODE_HEADROOM_BYTES
    )
    with pytest.raises(guard.Round0246Error):
        gate("read without a start")


def test_a_compliant_stage_passes_the_gate() -> None:
    """The gate is not merely strict: a real compliant stage must pass."""
    ticks = [0.0, 0.0] + [0.5 * step for step in range(1, 13)] + [6.0]
    gate = _scripted_gate(ticks, headroom_bytes=guard.R0245_NODE_HEADROOM_BYTES)
    gate.start()
    for step in range(12):
        gate(f"read {step}")
    gate.finish()
    verdict = gate.require(measured_slope_bytes_per_s=0.0)
    assert verdict["holds"] is True
    assert verdict["meets_the_registered_ceiling"] is True
    assert verdict["meets_the_r0244_worst_case_slope"] is True
    assert verdict["enforcement_polls"] >= verdict["required_polls"]


def test_the_novel_attack_battery_closes_every_attack(tmp_path) -> None:
    battery = guard.run_novel_attack_battery(
        workspace=str(tmp_path / "attacks")
    )
    assert battery["attacks_run"] >= 16
    assert battery["attacks_that_still_succeed"] == []
    assert battery["every_novel_attack_is_closed"] is True


def test_every_registered_threshold_carries_a_basis_and_a_blind_spot() -> None:
    registered = guard.registered_thresholds()
    assert len(registered) == 5
    for name, entry in registered.items():
        assert entry["basis"], name
        assert entry["what_it_does_not_catch"], name
        assert entry["value"] is not None, name


# --------------------------------------------------------------------------- #
# the blocker — the poll fix inside two_level_weight_sample is inert
# --------------------------------------------------------------------------- #
def test_the_chunked_stable_order_is_argsort_stable() -> None:
    rng = np.random.default_rng(246)
    keys = rng.integers(0, 41, size=50_000).astype(np.int64)
    reference = np.argsort(keys, kind="stable")
    for chunk in (1, 7, 1_000, 50_000, 200_000):
        mine = _stable_counting_order(
            keys, key_count=41, chunk=chunk, abort_check=None
        )
        assert np.array_equal(reference, mine), chunk


def _reference_two_level(weights, *, profile, draws, seed):
    """The pre-R0246 algorithm, re-typed here ONLY as the equality reference."""
    array = np.asarray(weights)
    block = int(profile["block"])
    block_sums = np.asarray(profile["block_sums"], dtype=np.float64)
    total = float(profile["total_weight"])
    rng = np.random.default_rng(int(seed))
    block_cdf = np.cumsum(block_sums)
    block_cdf[-1] = total
    chosen_block = np.searchsorted(
        block_cdf, rng.random(int(draws)) * total, side="right"
    ).astype(np.int64)
    np.clip(chosen_block, 0, block_sums.size - 1, out=chosen_block)
    order = np.argsort(chosen_block, kind="stable")
    sorted_blocks = chosen_block[order]
    boundaries = np.flatnonzero(np.diff(sorted_blocks)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [sorted_blocks.size]])
    edge_index = np.empty(int(draws), dtype=np.int64)
    for start, end in zip(starts, ends):
        index = int(sorted_blocks[start])
        lo = index * block
        hi = min(lo + block, array.size)
        chunk = np.asarray(array[lo:hi], dtype=np.float64)
        within = np.cumsum(chunk)
        picks = np.searchsorted(
            within, rng.random(end - start) * float(within[-1]), side="right"
        )
        np.clip(picks, 0, chunk.size - 1, out=picks)
        edge_index[order[start:end]] = lo + picks
    return {
        "edge_index": edge_index,
        "sampled_weights": np.asarray(array[edge_index], dtype=np.float64),
        "distinct_edges_drawn": int(np.unique(edge_index).size),
    }


def test_the_polled_sampler_draws_exactly_what_the_unpolled_one_drew() -> None:
    rng = np.random.default_rng(246_002)
    weights = np.maximum(
        rng.beta(0.7, 2.0, size=200_000).astype(np.float32), np.float32(1e-6)
    )
    profile = weight_block_profile(weights, block=4_096)
    reference = _reference_two_level(
        weights, profile=profile, draws=50_000, seed=17
    )
    seen: list[str] = []
    polled = two_level_weight_sample(
        weights, profile=profile, draws=50_000, seed=17,
        abort_check=seen.append, poll_chunk_draws=1_000,
    )
    assert np.array_equal(reference["edge_index"], polled["edge_index"])
    assert np.array_equal(
        reference["sampled_weights"], polled["sampled_weights"]
    )
    assert (
        reference["distinct_edges_drawn"] == polled["distinct_edges_drawn"]
    )
    #: and it really did poll, per unit, not once per 128 blocks
    assert len(seen) > 50


def test_the_sampler_still_works_without_an_abort_check() -> None:
    rng = np.random.default_rng(246_003)
    weights = np.maximum(
        rng.beta(0.7, 2.0, size=40_000).astype(np.float32), np.float32(1e-6)
    )
    profile = weight_block_profile(weights, block=4_096)
    sample = two_level_weight_sample(
        weights, profile=profile, draws=5_000, seed=3
    )
    assert sample["edge_index"].size == 5_000
    assert 0 < sample["distinct_edges_drawn"] <= 5_000


# --------------------------------------------------------------------------- #
# the tie-aware decision
# --------------------------------------------------------------------------- #
def test_the_aggregate_only_gate_routes_both_ways() -> None:
    evidence = tie.tie_use_positive_control()
    assert evidence["holds"] is True
    assert evidence["aggregate_use_permitted"] is True
    assert evidence["small_count_use_refused"] is True


def test_the_ledger_covers_both_rounds_and_every_entry_is_priced() -> None:
    rounds = {entry["round"] for entry in tie.TIE_AWARE_CLAIM_LEDGER}
    assert {"0241", "0243"} <= rounds
    for entry in tie.TIE_AWARE_CLAIM_LEDGER:
        assert entry["decisions"] > 0, entry["claim"]
        assert entry["margin"] > 0, entry["claim"]
        assert entry["margin_note"], entry["claim"]
        assert entry["kind"], entry["claim"]


def test_adjudication_separates_aggregates_from_small_counts() -> None:
    """At a plausible flip rate the aggregates stand and the per-row ones do not."""
    #: a planted rate of the order review-0241-01 F4 observed directly
    profile = {"verdict_flips": {"per_candidate_flip_rate": 5e-07}}
    verdict = tie.adjudicate_tie_aware_claims(profile)
    by_claim = {row["claim"]: row for row in verdict["claims"]}
    assert verdict["every_aggregate_claim_survives"] is True
    assert by_claim[
        "tie_aware_rows_at_zero = 25, exactly"
    ]["survives"] is False
    assert by_claim[
        "tie-aware mean recall 0.9979422666666667 over the 500,000-row probe"
    ]["survives"] is True
    assert verdict["claims_that_do_not_survive"]


def test_a_zero_flip_rate_would_rescue_everything() -> None:
    """The adjudication is driven by the measurement, not by the ledger."""
    verdict = tie.adjudicate_tie_aware_claims(
        {"verdict_flips": {"per_candidate_flip_rate": 0.0}}
    )
    assert verdict["claims_that_do_not_survive"] == []


def test_the_precision_profile_measures_flips_on_a_tiny_world() -> None:
    rng = np.random.default_rng(246_004)
    rows, dim, k = 200, 16, 15
    substrate = rng.normal(size=(rows, dim)).astype(np.float32)
    substrate /= np.linalg.norm(substrate, axis=1, keepdims=True)
    graph_ids = np.stack([
        np.array(
            [(row + offset) % rows for offset in range(1, k + 1)],
            dtype=np.int32,
        )
        for row in range(rows)
    ])
    probe_rows = np.arange(rows, dtype=np.int64)
    truth_ids = graph_ids.astype(np.int32)
    truth_cos = np.zeros((rows, k), dtype=np.float32)
    for row in range(rows):
        truth_cos[row] = np.sort(
            substrate[row] @ substrate[graph_ids[row]].T
        )[::-1]
    profile = tie.tie_aware_precision_profile(
        substrate=substrate, graph_ids=graph_ids, probe_query_rows=probe_rows,
        truth_ids=truth_ids, truth_cosines=truth_cos, sample_rows=200,
        block=50,
    )
    assert profile["matched_candidate_truth_pairs"] > 0
    assert profile["candidate_decisions_scored"] == 200 * k
    assert profile["float64"]["abs_delta_p99"] <= (
        profile["float32"]["abs_delta_p99"]
    )
    assert profile["verdict_flips"]["per_candidate_flip_rate"] is not None


def test_the_tolerance_was_not_raised() -> None:
    """R0246 registers a rule; it does not move R0243's published numbers."""
    from basemap.round0227_low_c_contract import TIE_TOLERANCE

    assert TIE_TOLERANCE == 1e-06
    assert "NOT raise TIE_TOLERANCE" in tie.TIE_AGGREGATE_ONLY_RULE


# --------------------------------------------------------------------------- #
# nothing in this round signals anything
# --------------------------------------------------------------------------- #
def test_no_r0246_file_contains_a_signalling_construct() -> None:
    """Read the source. Do not delegate this to a detector."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = [
        "basemap/round0246_guard.py",
        "basemap/round0246_tie.py",
        "experiments/round0246_nodes.py",
        "experiments/prepare_round0246_queue.py",
    ]
    forbidden = (
        "os.kill", "signal.", "SIGKILL", "SIGTERM", "pkill", "killpg",
        "subprocess.Popen", "terminate()",
    )
    for name in files:
        with open(os.path.join(here, name), encoding="utf-8") as handle:
            source = handle.read()
        for token in forbidden:
            assert token not in source, f"{name} contains {token!r}"


def test_this_round_adds_only_its_own_files_and_the_declared_edits() -> None:
    """R0246 imports rounds 0215-0245 read-only except the declared edits."""
    import subprocess as _sp  # noqa: PLC0415 - test-side git query only

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    committed = _sp.run(
        ["git", "-C", repo, "diff", "--name-only",
         "c94a1401dc33b71e045925bf28cfde543457f9d9", "HEAD"],
        check=False, capture_output=True, text=True,
    ).stdout.split()
    worktree = [
        line[3:] for line in _sp.run(
            ["git", "-C", repo, "status", "--porcelain"],
            check=False, capture_output=True, text=True,
        ).stdout.splitlines() if line
    ]
    allowed = {
        #: the declared edits to reviewed modules, each named in the round file
        "basemap/round0244_guard.py",
        "basemap/round0244_prereq.py",
        "basemap/round0245_guard.py",
        "tests/test_round0244_contract.py",
        "tests/test_round0245_contract.py",
        #: R0246's own files
        "basemap/round0246_guard.py",
        "basemap/round0246_tie.py",
        "experiments/round0246_nodes.py",
        "experiments/prepare_round0246_queue.py",
        "tests/test_round0246_contract.py",
        "tests/test_round0246_cpu_smoke.py",
        #: R0247's declared edits and its own files. R0247 makes every safety
        #: parameter in R0244-R0246 non-overridable, which necessarily touches
        #: the modules that own them; every one is a diff in result-0247.
        "basemap/round0247_registry.py",
        "basemap/round0247_guard.py",
        "basemap/round0247_precision.py",
        "basemap/round0246_tie.py",
        "experiments/round0247_nodes.py",
        "experiments/prepare_round0247_queue.py",
        "tests/test_round0247_contract.py",
        "tests/test_round0247_cpu_smoke.py",
        #: R0248's declared edits and its own files. R0248 routes the two
        #: observation-gap bounds and SAMPLER_MAX_ANONYMOUS_BYTES through the
        #: registry AT THE COMPARISON SITE, retires the runtime abort-reader
        #: marker, registers `replay`, and derives the inventory mechanically;
        #: every one is a diff in result-0248.
        "basemap/round0248_inventory.py",
        "basemap/round0248_guard.py",
        "basemap/round0248_external.py",
        "experiments/round0248_nodes.py",
        "experiments/prepare_round0248_queue.py",
        "experiments/round0244_nodes.py",
        "experiments/round0245_nodes.py",
        "experiments/round0246_nodes.py",
        "tests/test_round0248_contract.py",
        "tests/test_round0248_cpu_smoke.py",
    }
    changed = set(committed + worktree)
    assert changed <= allowed, sorted(changed - allowed)


def test_the_watchdog_edit_moves_no_r0242_threshold_or_rule(tmp_path) -> None:
    """R0246 edits R0244's loop. R0242's machine rule must be untouched."""
    from experiments.round0242_nodes import (
        WATCHDOG_ANON_BYTES,
        WATCHDOG_MEM_AVAILABLE_BYTES,
        WATCHDOG_SWAP_GROWTH_BYTES,
    )

    assert WATCHDOG_ANON_BYTES == 64_424_509_440  # 60 GiB, R0242's
    assert WATCHDOG_MEM_AVAILABLE_BYTES == 17_179_869_184
    assert WATCHDOG_SWAP_GROWTH_BYTES == 4_294_967_296
    watchdog = guard0245.EnforcedHostWatchdog(
        anonymous_budget_bytes=1 << 62, interval_s=0.02,
        abort_flag_path=str(tmp_path / "rule.abort"), label="rule",
    )
    guard._run_watchdog_until(watchdog, wall_s=0.2)
    receipt = watchdog.receipt()
    #: R0247 (declared edit): the receipt gained the two measured
    #: observation-gap fields and the registered-interval denominator, so the
    #: instrument version moved. R0242's three thresholds are asserted above
    #: and are untouched.
    assert receipt["instrument"] == "round0244-threaded-host-watchdog-v2"
    assert receipt["sample_interval_s"] == 0.02
    assert threading.active_count() >= 1
