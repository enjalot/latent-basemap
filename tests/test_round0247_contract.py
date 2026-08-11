"""R0247 contract — the class fix, its controls, and the attacks on it.

Every test here plants a defect and requires the guard to catch it. A guard
whose suite contains no failing input is untested at its only job; that rule has
six precedents in this program and two of them were found by reviewers after the
round shipped believing it was done.
"""
from __future__ import annotations

import math
import os
import subprocess

import numpy as np
import pytest

from basemap import round0244_guard as guard0244
from basemap import round0245_guard as guard0245
from basemap import round0246_guard as guard0246
from basemap import round0246_tie as tie
from basemap import round0247_guard as controls
from basemap import round0247_precision as precision
from basemap import round0247_registry as registry


# --------------------------------------------------------------------------- #
# A. the registry itself
# --------------------------------------------------------------------------- #
def test_the_registry_fingerprint_matches_its_pin() -> None:
    """Moving a bound without registering the move fails closed."""
    assert registry.registry_fingerprint() == (
        registry.REGISTERED_REGISTRY_SHA256
    )
    assert registry.verify_registry()["holds"] is True


def test_a_moved_registry_fails_every_gate(monkeypatch) -> None:
    monkeypatch.setattr(
        registry, "REGISTERED_REGISTRY_SHA256", "0" * 64, raising=True
    )
    with pytest.raises(registry.Round0247Error):
        registry.verify_registry(label="planted")


def test_the_registry_is_read_only_and_its_entries_are_frozen() -> None:
    with pytest.raises(TypeError):
        registry.REGISTERED_SAFETY_PARAMETERS["r0246_max_poll_spacing_s"] = None
    with pytest.raises(Exception):
        registry.REGISTERED_SAFETY_PARAMETERS[
            "r0246_max_poll_spacing_s"
        ].value = 1e6


def test_every_registered_parameter_carries_a_basis_and_a_blind_spot() -> None:
    for name, parameter in registry.REGISTERED_SAFETY_PARAMETERS.items():
        assert parameter.basis, name
        assert parameter.override_path, name
        assert parameter.what_it_does_not_catch, name
        assert parameter.role, name
        assert parameter.direction in (registry.CEILING, registry.FLOOR), name
        #: R0249 adds the third class, `declared` — see
        #: `SafetyParameter.enforcement` and review-0248-01 §B finding 3.
        assert parameter.enforcement in (
            registry.ENFORCEMENT_REFUSED,
            registry.ENFORCEMENT_CLAMPED,
            registry.ENFORCEMENT_DECLARED,
        ), name


def test_every_registered_parameter_has_a_planted_weaker_value() -> None:
    """A registry entry with no control is an unattacked guard."""
    assert set(controls.WEAKER_THAN_REGISTERED) == set(
        registry.REGISTERED_SAFETY_PARAMETERS
    )
    for name, weaker in controls.WEAKER_THAN_REGISTERED.items():
        parameter = registry.REGISTERED_SAFETY_PARAMETERS[name]
        assert parameter.weaker_than_registered(weaker), name


def test_registered_bounds_cannot_report_a_callers_value() -> None:
    block = registry.registered_bounds(["r0246_max_poll_spacing_s"])
    assert block["registered_r0246_max_poll_spacing_s"] == pytest.approx(
        29_548_888_064 / 11_767_996_416
    )
    assert block["registered_registry_sha256"] == (
        registry.REGISTERED_REGISTRY_SHA256
    )
    with pytest.raises(registry.Round0247Error):
        registry.registered_bounds(["not_a_registered_parameter"])


def test_clamp_refuses_an_unregistered_parameter() -> None:
    with pytest.raises(registry.Round0247Error):
        registry.clamp("not_registered", 1.0, site="test")


def test_a_stricter_caller_is_honoured_and_recorded() -> None:
    value, record = registry.clamp(
        "r0246_max_poll_spacing_s", 0.5, site="test"
    )
    assert value == 0.5
    assert record["kind"] == "stricter"


# --------------------------------------------------------------------------- #
# B. one positive control per parameter
# --------------------------------------------------------------------------- #
def test_the_clamp_control_fires_for_every_registered_parameter() -> None:
    evidence = controls.run_clamp_controls()
    assert evidence["holds"] is True
    assert evidence["parameters_controlled"] == len(
        registry.REGISTERED_SAFETY_PARAMETERS
    )
    for row in evidence["rows"]:
        assert row["the_attempt_is_recorded"] is True
        if row["enforcement"] == registry.ENFORCEMENT_DECLARED:
            #: R0249: a `declared` flag is NOT substituted — clamping it would
            #: make a replay receipt report `replay: false`. What is asserted
            #: instead is what `record_declaration()` really does.
            assert row["controlled_through"] == "record_declaration"
            assert row["the_registered_value_was_used"] is False
            assert row["the_declaration_stands"] is True
            assert row["the_weakening_record_reaches_the_sealing_gate"] is True
        else:
            assert row["controlled_through"] == "clamp"
            assert row["the_registered_value_was_used"] is True


def test_the_call_site_controls_fire(tmp_path) -> None:
    evidence = controls.run_call_site_controls(
        flag_path=str(tmp_path / "control.abort")
    )
    assert evidence["holds"] is True
    assert evidence["controls_run"] >= 10
    for row in evidence["rows"]:
        assert row["gate_refused_it"] is True
        assert row["the_receipt_names_the_attempt"] is True


# --------------------------------------------------------------------------- #
# C. review-0246-01 C — the sixteenth attack
# --------------------------------------------------------------------------- #
def test_the_reviewers_sixteenth_attack_is_refused() -> None:
    evidence = controls.run_reviewer_sixteenth_attack_control()
    assert evidence["holds"] is True
    assert evidence["gate_refused_it"] is True
    assert evidence["declared_max_poll_spacing_s"] == 1e6
    assert evidence["effective_max_poll_spacing_s"] == pytest.approx(
        2.5109531834854018
    )
    assert evidence["registered_max_poll_spacing_s"] == pytest.approx(
        2.5109531834854018
    )
    assert "meets_the_registered_ceiling" in evidence["failure_arms"]
    assert (
        "no_weakening_safety_override_was_attempted"
        in evidence["failure_arms"]
    )


def test_the_ceiling_cannot_be_reassigned_after_construction() -> None:
    gate = guard0246.AbortPollGate(
        inner=controls._sanctioned_reader, headroom_bytes=1 << 20,
        label="reassignment",
    )
    with pytest.raises(AttributeError):
        gate.max_poll_spacing_s = 1e6
    with pytest.raises(AttributeError):
        gate.min_polls = 0


def test_the_worst_case_arm_no_longer_depends_on_training_performed() -> None:
    """`training_performed` was a self-declaration that switched off an arm."""
    ticks = iter([0.0, 0.0, 3.0, 3.0])
    gate = guard0246.AbortPollGate(
        inner=controls._noop,
        headroom_bytes=int(registry.REGISTERED_SAFETY_PARAMETERS[
            "max_declared_headroom_bytes"
        ].value),
        label="not a training node", training_performed=False,
        clock=lambda: next(ticks), replay=True,
    )
    gate.start()
    gate("read 1")
    gate("read 2")
    gate.finish()
    with pytest.raises(guard0246.Round0246Error):
        gate.require(measured_slope_bytes_per_s=0.0)
    assert "every_node_meets_the_worst_case_slope" in (
        gate.last_verdict["failures"]
    )


# --------------------------------------------------------------------------- #
# D. review-0246-01 A — the coverage denominator
# --------------------------------------------------------------------------- #
def test_the_five_second_coverage_attack_is_refused_three_ways() -> None:
    evidence = controls.run_coverage_denominator_control()
    assert evidence["holds"] is True
    assert evidence["declared_sample_interval_s"] == 5.0
    assert evidence["coverage_as_r0246_computed_it"] > 0.99
    assert evidence["observation_gap_over_the_sealed_headroom"] == (
        pytest.approx(5.0 * 11_767_996_416 / 29_548_888_064)
    )
    assert evidence["observation_gap_over_the_sealed_headroom"] > 1.9
    assert len(evidence["arms"]) == 4
    for arm in evidence["arms"].values():
        assert arm["refused"] is True


def test_the_interval_is_clamped_at_construction(tmp_path) -> None:
    watchdog = guard0244.ThreadedHostWatchdog(
        interval_s=5.0, abort_flag_path=str(tmp_path / "flag.abort"),
        label="clamped interval",
    )
    assert watchdog.interval_s == 0.25
    assert watchdog.declared_interval_s == 5.0
    assert [
        record["parameter"] for record in watchdog.safety_overrides
    ] == ["watchdog_sample_interval_s"]
    #: and the default budget is now the registered one, so a node that
    #: declares nothing records nothing. R0244's default was 64 GiB, ABOVE
    #: R0242's own 60 GiB pressure threshold - a budget arm that could not fire.
    assert watchdog.anonymous_budget_bytes == 64_424_509_440
    assert guard0244.WATCHDOG_DEFAULT_ANON_BUDGET_BYTES == 64_424_509_440


def test_the_receipt_reports_the_registered_interval_not_the_declared_one(
    tmp_path,
) -> None:
    watchdog = guard0245.EnforcedHostWatchdog(
        interval_s=0.02, abort_flag_path=str(tmp_path / "flag.abort"),
        label="registered denominator",
    )
    guard0246._run_watchdog_until(watchdog, wall_s=0.4)
    receipt = watchdog.receipt()
    assert receipt["registered_watchdog_sample_interval_s"] == 0.25
    assert receipt["declared_sample_interval_s"] == 0.02
    assert receipt["max_thread_sample_gap_s"] > 0.0
    assert receipt["mean_thread_sample_gap_s"] > 0.0
    assert receipt["expected_samples_at_the_registered_interval"] == (
        pytest.approx(receipt["sampled_wall_s"] / 0.25)
    )


def test_a_receipt_with_no_measured_gap_cannot_be_gated_on() -> None:
    """A pre-R0247 receipt reads as UNMEASURED, never as 'no gap'."""
    stale = controls._healthy_receipt()
    stale.pop("max_thread_sample_gap_s")
    with pytest.raises(guard0246.Round0246Error):
        guard0246.require_live_sampler(stale, label="stale receipt")


def test_the_coverage_scope_keyword_cannot_disable_the_arm() -> None:
    with pytest.raises(guard0246.Round0246Error):
        guard0246.require_live_sampler(
            controls._healthy_receipt(), label="scope override",
            min_expected_samples=1e9,
        )


# --------------------------------------------------------------------------- #
# E. attacks on R0247's own fix
# --------------------------------------------------------------------------- #
def test_the_self_attack_battery_publishes_the_one_that_succeeds(
    tmp_path,
) -> None:
    battery = controls.run_self_attack_battery(
        flag_path=str(tmp_path / "self.abort")
    )
    assert battery["attacks_run"] >= 7
    #: the fabricated receipt is NOT closed and is published as such. If this
    #: assertion ever changes, the round has either closed it or hidden it.
    assert battery["attacks_that_still_succeed"] == [
        "r0247-self-7: hand the liveness gate a fabricated receipt"
    ]
    for row in battery["attacks"]:
        assert row["residual"], row["attack"]


def test_the_module_global_no_longer_reaches_the_decision(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        guard0244, "WATCHDOG_MAX_CONSECUTIVE_SAMPLE_FAILURES", 10 ** 9
    )
    watchdog = guard0245.EnforcedHostWatchdog(
        interval_s=0.02, abort_flag_path=str(tmp_path / "flag.abort"),
        label="module global",
    )
    assert watchdog.receipt()["max_consecutive_sample_failures_allowed"] == 3


def test_a_scripted_clock_cannot_be_a_nodes_measurement() -> None:
    gate = guard0246.AbortPollGate(
        inner=controls._sanctioned_reader,
        headroom_bytes=int(registry.REGISTERED_SAFETY_PARAMETERS[
            "max_declared_headroom_bytes"
        ].value),
        label="scripted", training_performed=True, clock=lambda: 0.0,
    )
    gate.start()
    for step in range(4):
        gate(f"read {step}")
    gate.finish()
    with pytest.raises(guard0246.Round0246Error):
        gate.require(measured_slope_bytes_per_s=0.0)
    assert "the_clock_is_the_registered_monotonic_clock" in (
        gate.last_verdict["failures"]
    )


def test_a_gate_that_does_not_read_the_abort_flag_is_refused() -> None:
    gate = guard0246.AbortPollGate(
        inner=controls._noop,
        headroom_bytes=int(registry.REGISTERED_SAFETY_PARAMETERS[
            "max_declared_headroom_bytes"
        ].value),
        label="no-op reader", training_performed=True,
    )
    gate.start()
    gate("read 1")
    gate("read 2")
    gate.finish()
    with pytest.raises(guard0246.Round0246Error):
        gate.require(measured_slope_bytes_per_s=0.0)
    assert "the_gate_wraps_a_registered_abort_reader" in (
        gate.last_verdict["failures"]
    )


def test_the_registered_abort_reader_is_the_one_the_stripe_loops_call() -> None:
    from experiments.round0238_nodes import _check_runner_abort

    assert registry.is_registered_abort_reader(_check_runner_abort) is True
    assert registry.is_registered_abort_reader(controls._noop) is False
    assert registry.is_registered_abort_reader(None) is False


def test_a_replay_verdict_cannot_be_sealed_as_enforcement_evidence() -> None:
    ticks = iter([0.0, 0.0, 0.001, 0.002])
    gate = guard0246.AbortPollGate(
        inner=controls._noop,
        headroom_bytes=int(registry.REGISTERED_SAFETY_PARAMETERS[
            "max_declared_headroom_bytes"
        ].value),
        label="replay", clock=lambda: next(ticks), replay=True,
    )
    gate.start()
    gate("read 1")
    gate("read 2")
    gate.finish()
    verdict = gate.require(measured_slope_bytes_per_s=0.0)
    assert verdict["holds"] is True
    with pytest.raises(registry.Round0247Error):
        guard0246.require_enforcement_evidence(verdict, label="replay")


def test_the_unguarded_base_class_is_clamped_too() -> None:
    tracker = guard0245.AbortPollTracker(
        inner=controls._noop, headroom_bytes=1 << 50, label="base",
        slope_bytes_per_s=1.0,
    )
    tracker("a")
    tracker("b")
    verdict = tracker.verdict(measured_slope_bytes_per_s=0.0)
    assert verdict["effective_headroom_bytes"] == 29_548_888_064
    assert verdict["effective_worst_case_slope_bytes_per_s"] == (
        11_767_996_416.0
    )


# --------------------------------------------------------------------------- #
# F. the sampler chunk
# --------------------------------------------------------------------------- #
def test_the_sampler_poll_chunk_is_clamped() -> None:
    from basemap.round0244_prereq import (
        two_level_weight_sample,
        weight_block_profile,
    )

    rng = np.random.default_rng(247)
    weights = np.maximum(
        rng.beta(0.7, 2.0, size=20_000).astype(np.float32), np.float32(1e-6)
    )
    profile = weight_block_profile(weights, block=1_024)
    sample = two_level_weight_sample(
        weights, profile=profile, draws=5_000, seed=3,
        poll_chunk_draws=40_000_000,
    )
    assert sample["declared_poll_chunk_draws"] == 40_000_000
    assert sample["poll_chunk_draws"] == 2_000_000
    assert [
        record["parameter"] for record in sample["safety_overrides"]
    ] == ["sampler_poll_chunk_draws"]


# --------------------------------------------------------------------------- #
# G. the ledger, the bound and the reconciled rule
# --------------------------------------------------------------------------- #
def test_the_poisson_bound_reduces_to_the_rule_of_three() -> None:
    assert precision.poisson_upper_bound(0) == pytest.approx(-math.log(0.05))
    assert precision.poisson_upper_bound(0) == pytest.approx(2.9957322735)
    #: and it is monotone in the observed count
    limits = [precision.poisson_upper_bound(k) for k in range(6)]
    assert limits == sorted(limits)


def test_a_weaker_confidence_is_refused() -> None:
    with pytest.raises(registry.Round0247Error):
        precision.poisson_upper_bound(0, confidence=0.5)


def _profile(decisions: int, flips: int = 0) -> dict:
    return {
        "candidate_decisions_scored": decisions,
        "verdict_flips": {
            "total": flips,
            "per_candidate_flip_rate": flips / decisions,
        },
    }


def test_the_sealed_adjudication_reproduces_r0246s_eight() -> None:
    """review-0246-01 E: 'survive: 6 | do NOT survive: 8', not seven."""
    sealed = precision.sealed_bound_adjudication(_profile(300_000))
    assert sealed["flip_rate_bound"]["upper_bound_flip_rate"] == (
        pytest.approx(2.99573227355399 / 300_000)
    )
    assert sealed["claims_that_survive_at_the_bound"] == 6
    assert sealed["claims_that_do_not_survive_at_the_bound"] == 8
    assert len(sealed["claims_that_do_not_survive_at_the_bound_names"]) == 8
    #: and the already-repaired claim is COUNTED and separately labelled,
    #: which is exactly the off-by-one the reviewer found
    assert len(sealed["already_repaired_among_the_non_survivors"]) == 1
    assert len(sealed["corrections_owed"]) == 7


def test_the_whole_probe_tightens_the_bound_by_25x() -> None:
    sealed = precision.sealed_bound_adjudication(_profile(7_500_000))
    assert sealed["flip_rate_bound"]["upper_bound_flip_rate"] == (
        pytest.approx(2.99573227355399 / 7_500_000)
    )
    assert sealed["claims_that_survive_at_the_bound"] == 11
    assert sealed["claims_that_do_not_survive_at_the_bound"] == 3
    #: every claim that still fails is a per-row or small-integer use, which is
    #: precisely what the registered aggregate-only rule forbids
    failing_kinds = {
        row["kind"] for row in
        sealed["at_the_retrospective_criterion"]["claims"]
        if not row["survives"]
    }
    assert all("aggregate" not in kind for kind in failing_kinds)


def test_the_bound_adjudication_is_a_receipt_not_prose() -> None:
    sealed = precision.sealed_bound_adjudication(_profile(7_500_000))
    #: every count in it is recomputable from the sealed rows
    rows = sealed["at_the_retrospective_criterion"]["claims"]
    assert sum(1 for row in rows if row["survives"]) == (
        sealed["claims_that_survive_at_the_bound"]
    )
    rate = sealed["flip_rate_bound"]["upper_bound_flip_rate"]
    for row in rows:
        assert row["expected_flipped_decisions"] == pytest.approx(
            row["decisions"] * rate
        )
        assert row["survives"] == (
            row["expected_flips_over_margin"] <= 1.0
        )


def test_the_two_criteria_are_registered_and_differ_by_100x() -> None:
    assert tie.TIE_CLAIM_MAX_EXPECTED_FLIPS_OVER_MARGIN == 1.0
    assert tie.TIE_USE_MAX_EXPECTED_FLIPS_OVER_MARGIN == 0.01
    assert "1% of the margin" in tie.TIE_AGGREGATE_ONLY_RULE
    assert (
        "TIE_USE_MAX_EXPECTED_FLIPS_OVER_MARGIN = 0.01"
        in tie.TIE_AGGREGATE_ONLY_RULE
    )
    assert (
        "TIE_CLAIM_MAX_EXPECTED_FLIPS_OVER_MARGIN = 1.0"
        in tie.TIE_AGGREGATE_ONLY_RULE
    )


def test_the_admission_gate_now_applies_the_sealed_one_percent() -> None:
    """R0246 quoted 1% and applied 1.0. The gate applies 1% now."""
    with pytest.raises(guard0246.Round0246Error):
        tie.require_aggregate_tie_aware_use(
            decisions=7_500_000, margin=29.15, flip_rate=1e-06,
            label="R0243's 2,915 count at R0246's planted rate",
        )
    permitted = tie.require_aggregate_tie_aware_use(
        decisions=7_500_000, margin=29.15, flip_rate=1e-08,
        label="the same count at a rate 100x smaller",
    )
    assert permitted["use_is_permitted"] is True
    assert permitted["margin_fraction"] == 0.01


def test_the_tie_tolerance_cannot_be_widened_by_a_caller() -> None:
    rng = np.random.default_rng(11)
    substrate = rng.normal(size=(64, 8)).astype(np.float32)
    substrate /= np.linalg.norm(substrate, axis=1, keepdims=True)
    graph_ids = np.stack([
        np.array([(row + off) % 64 for off in range(1, 16)], dtype=np.int64)
        for row in range(64)
    ])
    probe_rows = np.arange(64, dtype=np.int64)
    truth_ids = graph_ids.astype(np.int64)
    truth_cos = np.einsum(
        "bd,bkd->bk", substrate[probe_rows], substrate[truth_ids]
    ).astype(np.float32)
    with pytest.raises(registry.Round0247Error):
        tie.tie_aware_precision_profile(
            substrate=substrate, graph_ids=graph_ids,
            probe_query_rows=probe_rows, truth_ids=truth_ids,
            truth_cosines=truth_cos, sample_rows=64, tolerance=1.0,
        )


def test_the_adjudication_margin_fraction_cannot_be_widened() -> None:
    with pytest.raises(registry.Round0247Error):
        tie.adjudicate_tie_aware_claims(
            _profile(7_500_000, flips=7_500), margin_fraction=1e6
        )


# --------------------------------------------------------------------------- #
# H. the precision fix
# --------------------------------------------------------------------------- #
def test_the_recompute_separates_storage_from_arithmetic() -> None:
    """review-0246-01 F's claim, tested rather than asserted."""
    rng = np.random.default_rng(2470)
    substrate = rng.normal(size=(256, precision.SUBSTRATE_DIMENSION))
    substrate /= np.linalg.norm(substrate, axis=1, keepdims=True)
    substrate = substrate.astype(np.float32)
    probe_rows = np.arange(256, dtype=np.int64)
    truth_ids = np.stack([
        np.array([(row + off) % 256 for off in range(1, 16)], dtype=np.int64)
        for row in range(256)
    ])
    #: a float32 COMPUTATION, exactly as R0238's truth cosines are
    stored = np.einsum(
        "bd,bkd->bk", substrate[probe_rows], substrate[truth_ids]
    ).astype(np.float32)
    recompute = precision.recompute_truth_cosines_f64(
        substrate=substrate, probe_query_rows=probe_rows, truth_ids=truth_ids,
        block=64,
    )
    floor = precision.cosine_noise_floor(
        stored_f32=stored, recomputed_f64=recompute["cosines"],
        substrate=substrate, probe_query_rows=probe_rows, truth_ids=truth_ids,
        control_rows=256, block=64,
    )
    #: the stored residual is float32 ARITHMETIC, not the container: it sits
    #: well above the pure quantisation column
    assert floor["stored_vs_recomputed"]["p99"] > (
        floor["storage_quantisation"]["p99"]
    )
    assert floor["the_residual_is_arithmetic_not_storage"] is True
    #: and float64 is orders of magnitude below both
    assert floor["float64_arithmetic"]["p99"] < 1e-12
    assert floor["float32_half_ulp_at_one"] == 2.0 ** -25


def test_the_defensible_tolerance_is_stated_and_not_applied() -> None:
    floor = {
        "float64_arithmetic": {"p99": 1e-16, "max": 4e-16},
        "stored_vs_recomputed": {"p99": 5.336e-07},
        "storage_quantisation": {"p99": 2.98e-08},
    }
    stated = precision.defensible_tolerance(floor)
    assert stated["the_tolerance_was_not_moved"] is True
    assert stated["current_tie_tolerance"] == 1e-6
    assert stated["smallest_defensible_tolerance"] == pytest.approx(4e-13)
    assert stated["current_tolerance_over_the_float64_floor"] == (
        pytest.approx(1e10)
    )
    #: and the current tolerance sits at under 2x its stored-reference floor,
    #: which is the condition R0246 shipped under
    assert stated["current_tolerance_over_the_stored_floor"] < 2.0


# --------------------------------------------------------------------------- #
# I. scope
# --------------------------------------------------------------------------- #
def test_no_r0247_file_contains_a_signalling_construct() -> None:
    """Read the source, do not delegate to a detector."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    banned = (
        "os.kill", "signal.", "SIGKILL", "SIGTERM", "pkill", "kill -9",
        "subprocess.Popen", "ptrace", "py-spy", "cupy",
    )
    for name in (
        "basemap/round0247_registry.py",
        "basemap/round0247_guard.py",
        "basemap/round0247_precision.py",
        "experiments/round0247_nodes.py",
    ):
        with open(os.path.join(repo, name), encoding="utf-8") as handle:
            source = handle.read()
        for token in banned:
            assert token not in source, f"{name} contains {token!r}"


def test_this_round_adds_only_its_own_files_and_the_declared_edits() -> None:
    """R0247 imports rounds 0215-0246 read-only except the declared edits."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    #: R0249: pinned to R0247's OWN commit range, for the reason given in the
    #: same test in `test_round0246_contract.py`. R0247's release is
    #: `0941b37`, which is R0248's base.
    committed = subprocess.run(
        ["git", "-C", repo, "diff", "--name-only",
         "f636769370e254e5883ec69a37eb5e49502d9381",
         "0941b3776442cfdf00575f84c2688d63d28a5611"],
        check=False, capture_output=True, text=True,
    ).stdout.split()
    worktree: list[str] = []
    allowed = {
        #: the declared edits to reviewed modules - every one is where the
        #: class fix belongs, and every one is a diff in result-0247
        "basemap/round0244_guard.py",
        "basemap/round0244_prereq.py",
        "basemap/round0245_guard.py",
        "basemap/round0246_guard.py",
        "basemap/round0246_tie.py",
        "tests/test_round0245_contract.py",
        "tests/test_round0246_contract.py",
        "tests/test_round0246_cpu_smoke.py",
        #: R0247's own files
        "basemap/round0247_registry.py",
        "basemap/round0247_guard.py",
        "basemap/round0247_precision.py",
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
        "tests/test_round0245_cpu_smoke.py",
    }
    changed = set(committed + worktree)
    assert changed <= allowed, sorted(changed - allowed)


def test_r0242s_machine_rule_and_thresholds_are_untouched() -> None:
    from experiments.round0242_nodes import (
        WATCHDOG_ANON_BYTES,
        WATCHDOG_MEM_AVAILABLE_BYTES,
        WATCHDOG_SWAP_GROWTH_BYTES,
    )

    assert WATCHDOG_ANON_BYTES == 64_424_509_440
    assert WATCHDOG_MEM_AVAILABLE_BYTES == 17_179_869_184
    assert WATCHDOG_SWAP_GROWTH_BYTES == 4_294_967_296


def test_no_registered_threshold_moved_without_being_registered() -> None:
    """Every number R0244-R0246 registered is still its registered value."""
    assert guard0244.WATCHDOG_SAMPLE_INTERVAL_S == 0.25
    assert guard0244.WATCHDOG_MAX_CONSECUTIVE_SAMPLE_FAILURES == 3
    assert guard0245.MIN_BINDING_SLOPE_BYTES_PER_S == 11_767_996_416.0
    assert guard0245.R0244_BUDGET_HEADROOM_BYTES == 29_548_888_064
    assert guard0246.MIN_THREAD_SAMPLE_COVERAGE == 0.50
    assert guard0246.COVERAGE_GATE_MIN_EXPECTED_SAMPLES == 20.0
    assert guard0246.R0246_MAX_POLL_SPACING_S == pytest.approx(
        2.5109531834854018
    )
    assert guard0246.MIN_ENFORCEMENT_POLLS == 2
    #: and the two R0247 introduces are derived from the same sealed pair
    assert registry.WATCHDOG_MAX_OBSERVATION_GAP_S == pytest.approx(
        29_548_888_064 / 11_767_996_416
    )
    assert registry.WATCHDOG_MAX_MEAN_OBSERVATION_GAP_S == pytest.approx(
        0.25 / 0.50
    )
