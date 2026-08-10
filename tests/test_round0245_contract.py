"""R0245 contract — the fixes are gates, and the gates fail on planted defects.

Every check this round adds ships a positive control here or in
`basemap/round0245_*.py`, and every one of those controls plants the defect
rather than describing it. The file also holds the source-level assertions that
no static tool can express: that no R0245 file contains a signalling construct,
that the round re-types no registered threshold, and that the one edit to a
reviewed module changes no rule.
"""
from __future__ import annotations

import ast
import io
import os
import tokenize

import numpy as np
import pytest

from basemap import round0244_guard as guard0244
from basemap import round0245_did as did
from basemap import round0245_guard as guard
from basemap import round0245_sampler as sampler
from basemap.round0227_low_c_contract import TIE_TOLERANCE
from basemap.round0244_prereq import (
    DID_ALPHA,
    DID_DECISION_RULE,
    DID_MAPS_PER_ARM,
    SAMPLER_MIN_CHI_SQUARE_P,
)
from experiments import round0245_nodes as nodes
from experiments.round0242_nodes import (
    WATCHDOG_ANON_BYTES,
    WATCHDOG_MEM_AVAILABLE_BYTES,
    WATCHDOG_SWAP_GROWTH_BYTES,
    _HostWatchdog,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: The files that run INSIDE a node. The queue-preparation script is not one
#: of them: it runs pytest before the queue exists, exactly as R0244's did, and
#: R0244's contract test scoped the same way.
NODE_PATH_FILES = (
    "basemap/round0245_guard.py",
    "basemap/round0245_did.py",
    "basemap/round0245_sampler.py",
    "experiments/round0245_nodes.py",
)
FORBIDDEN_TOKENS = frozenset({
    "signal", "kill", "killpg", "SIGKILL", "SIGTERM", "SIGINT", "terminate",
    "subprocess", "multiprocessing", "ptrace", "pkill", "psutil",
})


# --------------------------------------------------------------------------- #
# safety, at the source level
# --------------------------------------------------------------------------- #
def _sources() -> dict[str, str]:
    out = {}
    for name in NODE_PATH_FILES:
        with open(os.path.join(REPO, name), encoding="utf-8") as handle:
            out[name] = handle.read()
    return out


def test_no_node_path_file_contains_a_signalling_construct() -> None:
    """`tokenize`-based, so a hazard NAMED in prose cannot excuse one in code."""
    hits = []
    for name, source in _sources().items():
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type in (tokenize.STRING, tokenize.COMMENT):
                continue
            if token.string in FORBIDDEN_TOKENS:
                hits.append((name, token.start[0], token.string))
    assert hits == [], hits


def test_no_node_path_file_imports_cupy() -> None:
    for name, source in _sources().items():
        assert "cupy" not in source, f"{name} mentions cupy"


# --------------------------------------------------------------------------- #
# the R0244 edit changes no threshold and no rule
# --------------------------------------------------------------------------- #
def test_the_r0244_watchdog_edit_moves_no_threshold_and_no_rule() -> None:
    """The A1 fix is behavioural, not numerical: the inherited conjunctive rule
    and all three of its thresholds must still be exactly R0242's."""
    watchdog = guard0244.ThreadedHostWatchdog(
        abort_flag_path=None, label="threshold inheritance"
    )
    receipt = watchdog.receipt()
    assert receipt["swap_growth_threshold_bytes"] == WATCHDOG_SWAP_GROWTH_BYTES
    assert receipt["anonymous_threshold_bytes"] == WATCHDOG_ANON_BYTES
    assert receipt["mem_available_threshold_bytes"] == WATCHDOG_MEM_AVAILABLE_BYTES
    assert receipt["guard_axis"] == "anonymous, never RSS"
    assert receipt["rule"] == _HostWatchdog().receipt()["rule"]
    assert isinstance(watchdog, _HostWatchdog)
    assert receipt["sampling_thread_alive"] is True
    assert receipt["thread_death"] is None


def test_a_healthy_watchdog_poll_does_not_raise() -> None:
    """The A1 fix must not make a working guard fail — the negative case."""
    watchdog = guard0244.ThreadedHostWatchdog(
        anonymous_budget_bytes=1 << 62, interval_s=0.05,
        abort_flag_path=None, label="healthy",
    )
    with watchdog:
        watchdog.poll("healthy guard")
        watchdog.poll("healthy guard again")
    assert watchdog.receipt()["sampling_thread_alive"] is True


# --------------------------------------------------------------------------- #
# fix 1 — the sampler thread's death is a visible failure
# --------------------------------------------------------------------------- #
def test_thread_death_positive_control_fires(tmp_path) -> None:
    evidence = guard.run_thread_death_positive_control(
        flag_path=str(tmp_path / "death.abort")
    )
    assert evidence["holds"] is True
    assert evidence["thread_death_recorded"] is True
    assert evidence["thread_still_running"] is False
    assert evidence["receipt_says_sampler_alive"] is False
    assert evidence["poll_raised"] is True
    #: The plant never crosses the budget, so nothing but the death itself
    #: could have made the node stop.
    assert evidence["budget_was_never_crossed"] is True


def test_a_non_oserror_in_the_sampler_makes_poll_raise(tmp_path) -> None:
    dying = guard._DyingWatchdog(
        die_after=1, anonymous_budget_bytes=1 << 62, interval_s=0.02,
        abort_flag_path=str(tmp_path / "d.abort"), label="raw plant",
    )
    with dying:
        deadline = 0
        while dying.thread_death is None and deadline < 500:
            deadline += 1
            dying._stop.wait(0.01)
        with pytest.raises(guard0244.Round0244Error):
            dying.poll("after the plant")
        with pytest.raises(guard0244.Round0244Error):
            dying.raise_if_thread_died("after the plant")


def test_a_node_refuses_to_publish_behind_a_dead_sampler() -> None:
    dead = {
        "sampling_thread_alive": False,
        "thread_death": "ValueError: planted",
        "samples": 1,
        "sample_coverage": 0.1,
    }
    with pytest.raises(guard.Round0245Error):
        nodes._require_live_sampler(dead, label="planted")
    alive = dict(dead, sampling_thread_alive=True, thread_death=None)
    assert nodes._require_live_sampler(alive, label="ok")["samples"] == 1


# --------------------------------------------------------------------------- #
# fix 2 — an unset abort flag refuses to start
# --------------------------------------------------------------------------- #
def test_missing_abort_flag_positive_control_fires(monkeypatch) -> None:
    evidence = guard.run_missing_flag_positive_control()
    assert evidence["holds"] is True
    assert evidence["precondition_refused_to_start"] is True
    assert evidence["watchdog_refused_to_arm"] is True


def test_require_enforceable_abort_flag_accepts_a_writable_path(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("ROUNDRUN_ABORT_FLAG", str(tmp_path / "node.abort"))
    accepted = guard.require_enforceable_abort_flag(label="ok")
    assert accepted["directory_is_writable"] is True
    monkeypatch.setenv("ROUNDRUN_ABORT_FLAG", str(tmp_path / "gone" / "n.abort"))
    with pytest.raises(guard.Round0245Error):
        guard.require_enforceable_abort_flag(label="missing directory")


def test_every_node_action_refuses_to_start_without_the_flag(
    monkeypatch, tmp_path
) -> None:
    """The precondition is on the node path, not only in a helper."""
    monkeypatch.delenv("ROUNDRUN_ABORT_FLAG", raising=False)
    active = {"manifest": {"round_id": "0245", "release_sha": "0" * 40}}
    for action in (nodes.GUARD_ACTION, nodes.DID_ACTION, nodes.SAMPLER_ACTION):
        with pytest.raises(guard.Round0245Error):
            nodes.run_job(
                active,
                {"action": action, "outputs": [str(tmp_path / action)]},
            )


# --------------------------------------------------------------------------- #
# fix 3 — the poll-spacing requirement, derived and non-vacuous
# --------------------------------------------------------------------------- #
def test_the_requirement_is_headroom_over_slope() -> None:
    verdict = guard.poll_spacing_requirement(
        slope_bytes_per_s=1_000_000_000.0, headroom_bytes=2_000_000_000,
        poll_spacing_s=1.5,
    )
    assert verdict["max_poll_spacing_s"] == pytest.approx(2.0)
    assert verdict["permitted_growth_after_trip_bytes"] == pytest.approx(1.5e9)
    assert verdict["requirement_holds"] is True
    breach = guard.poll_spacing_requirement(
        slope_bytes_per_s=1_000_000_000.0, headroom_bytes=2_000_000_000,
        poll_spacing_s=3.0,
    )
    assert breach["requirement_holds"] is False
    assert breach["minimum_polls_per_unit"] == 2
    with pytest.raises(guard.Round0245Error):
        guard.require_poll_spacing(
            slope_bytes_per_s=1_000_000_000.0, headroom_bytes=2_000_000_000,
            poll_spacing_s=3.0, label="planted breach",
        )


def test_r0243s_own_stripe_violates_the_derived_requirement() -> None:
    """The number the round exists to state: R0243's stripe is too wide."""
    verdict = guard.r0244_stripe_verdict()
    assert verdict["slope_bytes_per_s"] == float(
        guard.R0244_MEASURED_SLOPE_BYTES_PER_S
    )
    assert verdict["headroom_bytes"] == guard.R0244_BUDGET_HEADROOM_BYTES
    assert verdict["requirement_holds"] is False
    assert verdict["max_poll_spacing_s"] < verdict["poll_spacing_s"]
    assert (
        verdict["permitted_growth_after_trip_bytes"] > verdict["headroom_bytes"]
    )
    #: Sampling faster is not the fix: the observation gap is small already.
    assert verdict["observation_gap_bytes"] < verdict["headroom_bytes"]


def test_slope_from_trace_differentiates_a_trace() -> None:
    slope = guard.slope_from_trace([[0, 0], [1, 5], [2, 25], [3, 30]])
    assert slope["max_rise_bytes_per_s"] == 20.0
    assert slope["max_rise_starts_at_second"] == 1
    assert slope["peak_bytes"] == 30
    with pytest.raises(guard.Round0245Error):
        guard.slope_from_trace([[0, 0]])


def test_abort_poll_tracker_measures_and_refuses(monkeypatch) -> None:
    seen: list[str] = []
    tracker = guard.AbortPollTracker(
        inner=seen.append, headroom_bytes=1, label="planted",
        slope_bytes_per_s=1_000_000_000.0,
    )
    tracker("a")
    tracker("b")
    assert tracker.polls == 2
    assert seen == ["a", "b"]
    #: 1 B of headroom at 1 GB/s allows 1 ns between reads, which no pair of
    #: Python statements can meet, so this must refuse.
    with pytest.raises(guard.Round0245Error):
        tracker.require()
    generous = guard.AbortPollTracker(
        inner=seen.append, headroom_bytes=1 << 40, label="generous",
        slope_bytes_per_s=1.0,
    )
    generous("a")
    generous("b")
    assert generous.require()["requirement"]["requirement_holds"] is True

    #: the gate binds on the stage's OWN measured slope: a stage that really
    #: did allocate at 20 GB/s between two reads must be refused even when the
    #: worst-case column would have passed it.
    own = guard.AbortPollTracker(
        inner=seen.append, headroom_bytes=1, label="own slope",
        slope_bytes_per_s=1.0,
    )
    own("a")
    own("b")
    #: worst-case column set to 1 B/s passes; the stage's own 20 GB/s does not
    assert own.verdict(measured_slope_bytes_per_s=1.0)[
        "worst_case_requirement"
    ]["requirement_holds"] is True
    with pytest.raises(guard.Round0245Error):
        own.require(measured_slope_bytes_per_s=2.0e10)
    scored = own.verdict(measured_slope_bytes_per_s=1.0)
    assert scored["own_slope_requirement"] is not None
    assert scored["worst_case_requirement"]["slope_bytes_per_s"] == 1.0


# --------------------------------------------------------------------------- #
# fix 4 — the decision map is a gate
# --------------------------------------------------------------------------- #
def test_did_decision_routes_every_registered_branch() -> None:
    evidence = did.did_decision_positive_controls()
    assert evidence["holds"] is True
    assert evidence["unpowered_nulls_returned_indeterminate"] == 3
    verdicts = {row["case"]: row["verdict"] for row in evidence["cases"]}
    assert verdicts["harmful_effect"] == did.HARMFUL
    assert verdicts["powered_null"] == did.HARMLESS
    assert verdicts["unpowered_null_no_control"] == did.INDETERMINATE


def test_an_unpowered_null_is_never_harmless() -> None:
    """The clause the review said must become code."""
    for control in (None, {"planted_displacement": 0.05, "rejected": False,
                           "same_maps": True}):
        verdict = did.did_decision(
            stratification="genuine", difference_in_differences=0.0001,
            holm_adjusted_p=0.9, placebo_sd=0.01, power_control=control,
        )
        assert verdict["verdict"] == did.INDETERMINATE
        assert verdict["power_was_demonstrated_on_the_same_maps"] is False
    assert "INDETERMINATE, never harmless" in DID_DECISION_RULE
    assert did.DECISION_SOURCE_RULE is DID_DECISION_RULE


def test_did_decision_rejects_impossible_inputs() -> None:
    for kwargs in (
        {"holm_adjusted_p": 1.5},
        {"holm_adjusted_p": float("nan")},
        {"placebo_sd": -1.0},
    ):
        payload = {
            "stratification": "genuine", "difference_in_differences": 0.01,
            "holm_adjusted_p": 0.2, "placebo_sd": 0.01,
        }
        payload.update(kwargs)
        with pytest.raises(guard.Round0245Error):
            did.did_decision(**payload)


def test_the_family_decision_applies_holm_then_the_map() -> None:
    family = did.did_family_decision(per_stratification={
        "genuine": {
            "p_one_sided": 0.001, "difference_in_differences": 0.04,
            "placebo_sd": 0.01,
            "power_control": {"planted_displacement": 0.05, "rejected": True,
                              "same_maps": True},
        },
        "tie_forgiven": {
            "p_one_sided": 0.9, "difference_in_differences": 0.001,
            "placebo_sd": 0.01,
            "power_control": {"planted_displacement": 0.05, "rejected": True,
                              "same_maps": True},
        },
    })
    assert family["holm"]["alpha"] == DID_ALPHA
    assert family["verdicts"]["genuine"] == did.HARMFUL
    assert family["overall_verdict"] == did.HARMFUL


# --------------------------------------------------------------------------- #
# fix 5 — the arm assignment
# --------------------------------------------------------------------------- #
def _planted_vectors() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = 400
    strict = np.zeros(rows, dtype=np.int64)
    tie = np.zeros(rows, dtype=np.int64)
    strict[::10] = 2
    tie[::10] = 2
    strict[5::10] = 1
    #: the planted defect: a row with tie-aware loss and no strict loss, the
    #: exact shape of probe 21785.
    tie[7] = 1
    #: ten density levels with forty rows each, so every decile that carries a
    #: loss row also carries intact companions for R0228's exact match.
    kth = np.repeat(np.linspace(0.30, 0.99, 10), rows // 10)
    return strict, tie, kth


def test_the_clamp_removes_the_row_that_sat_in_both_arms() -> None:
    strict, tie, _kth = _planted_vectors()
    audit = did.tie_monotonicity_audit(
        strict_builder_missing=strict, tie_aware_builder_missing=tie
    )
    assert audit["rows_with_tie_above_strict"] == 1
    assert audit["violating_rows"] == [7]
    assert audit["rows_in_both_the_treated_and_control_arms_before_the_clamp"] == [7]
    assert audit["genuine_rows_after_the_clamp"] == (
        audit["genuine_rows_before_the_clamp"] - 1
    )
    assert audit["treated_plus_forgiven_equals_strict_loss_rows"] is True
    effective = did.tie_effective_builder_missing(
        strict_builder_missing=strict, tie_aware_builder_missing=tie
    )
    assert int(effective[7]) == 0
    assert np.all(effective <= strict)


def test_v1_populations_overlap_and_v2_do_not() -> None:
    """The positive control for the arm-assignment fix: the same planted rows
    through the unclamped definitions must be caught by the disjointness gate,
    and through the clamped ones must pass it."""
    from basemap.round0244_prereq import did_populations as v1

    strict, tie, kth = _planted_vectors()
    overlapping = v1(
        strict_builder_missing=strict, tie_aware_builder_missing=tie,
        kth_cosine=kth, sample_rows=50,
    )
    with pytest.raises(guard.Round0245Error):
        did.require_disjoint_arms(overlapping)
    fixed = did.did_populations_v2(
        strict_builder_missing=strict, tie_aware_builder_missing=tie,
        kth_cosine=kth, sample_rows=50,
    )
    assert fixed["arm_disjointness"]["arms_are_disjoint"] is True
    assert fixed["arm_disjointness"]["rows_on_both_sides"] == 0


def test_the_forensics_finds_a_planted_tolerance_demotion() -> None:
    """A candidate that IS in truth but whose recomputed cosine falls one
    tolerance below the k-th must be reported as the cause."""
    rng = np.random.default_rng(245)
    rows, dim, k = 6, 8, 4
    substrate = rng.normal(size=(rows, dim)).astype(np.float32)
    substrate /= np.linalg.norm(substrate, axis=1, keepdims=True)
    graph_ids = np.stack([
        np.array([(row + offset) % rows for offset in range(1, k + 1)])
        for row in range(rows)
    ]).astype(np.int32)
    probe_rows = np.arange(rows, dtype=np.int64)
    truth_ids = graph_ids.astype(np.int32)
    truth_cos = np.zeros((rows, k), dtype=np.float32)
    for row in range(rows):
        cosines = substrate[row] @ substrate[graph_ids[row]].T
        truth_cos[row] = np.sort(cosines)[::-1]
    #: raise the k-th truth cosine of row 0 just above every candidate, which
    #: is the numerical situation the two rows are in
    truth_cos[0, -1] = np.float32(float(truth_cos[0, -1]) + 5.0 * TIE_TOLERANCE)
    report = did.tie_violation_forensics(
        probe_rows_to_explain=[0], substrate=substrate, graph_ids=graph_ids,
        probe_query_rows=probe_rows, truth_ids=truth_ids,
        truth_cosines=truth_cos,
    )
    entry = report["rows"][0]
    assert entry["strict_containment_count"] == k
    assert entry["tie_aware_count"] < entry["strict_containment_count"]
    assert entry["candidates_in_truth_scored_invalid_by_the_tolerance"] >= 1


# --------------------------------------------------------------------------- #
# finding 6 — the permutation's cost is computed, not asserted
# --------------------------------------------------------------------------- #
def test_permutation_design_cost_prices_both_designs() -> None:
    cost = did.permutation_design_cost(trials=400, maps_per_arm=DID_MAPS_PER_ARM)
    assert cost["unrestricted_labellings"] == 252
    assert cost["paired_sign_flip_patterns"] == 32
    assert cost["paired_test_can_reject_at_this_n"] is False
    assert cost["smallest_arm_that_can_reject_paired"] == 8
    assert cost["smallest_arm_that_can_reject_unrestricted"] == 5
    assert 0.0 <= cost["simulation"]["realised_one_sided_rate"] <= 1.0


# --------------------------------------------------------------------------- #
# finding 7 — the sampler's power and blind spot
# --------------------------------------------------------------------------- #
def _tiny_profile(seed: int = 245) -> tuple[np.ndarray, dict]:
    from basemap.round0244_prereq import weight_block_profile

    rng = np.random.default_rng(seed)
    weights = rng.beta(0.7, 2.0, size=60_000).astype(np.float32)
    weights = np.maximum(weights, np.float32(1e-6))
    weights[rng.integers(0, weights.size, size=400)] = np.float32(1.0)
    profile = weight_block_profile(weights, block=4_096)
    return weights, profile


def test_required_draws_inverts_the_chi_square_relation() -> None:
    p_cells = np.array([0.5, 0.5])
    same = sampler.required_draws_for_arm(p_cells=p_cells, q_cells=p_cells)
    assert same["required_draws"] is None
    assert same["arm_can_never_reject_this_sampler"] is True
    q_cells = np.array([0.6, 0.4])
    apart = sampler.required_draws_for_arm(p_cells=p_cells, q_cells=q_cells)
    assert apart["required_draws"] > 0
    #: a smaller discrepancy must need strictly more draws
    closer = sampler.required_draws_for_arm(
        p_cells=p_cells, q_cells=np.array([0.55, 0.45])
    )
    assert closer["required_draws"] > apart["required_draws"]


def test_the_draw_floor_refuses_a_draw_count_below_the_family() -> None:
    weights, profile = _tiny_profile()
    battery = sampler.mis_sampler_battery(
        weights, profile=profile, draws=200_000, seed=1
    )
    generous = 200_000_000
    floor = sampler.sampler_draw_floor(
        profile=profile, battery=battery, registered_draws=generous
    )
    assert floor["registered_draws_clear_the_floor"] is True
    assert floor["required_draws_floor"] > 0.0
    with pytest.raises(guard.Round0245Error):
        sampler.sampler_draw_floor(
            profile=profile, battery=battery,
            registered_draws=int(floor["required_draws_floor"]) // 100,
        )


def test_the_family_catches_the_subtle_mis_samplers() -> None:
    weights, profile = _tiny_profile()
    battery = sampler.mis_sampler_battery(
        weights, profile=profile, draws=400_000, seed=7
    )
    caught = set(battery["caught"])
    assert "uniform_positions" in caught
    assert "block_by_weight_uniform_within" in caught
    #: and the correct sampler passes the same arms at the same draw count
    reference = sampler.true_sampler_reference(
        weights, profile=profile, draws=400_000, seed=11
    )
    assert reference["holds"] is True


def test_the_blind_spot_is_computed_and_reported() -> None:
    weights, profile = _tiny_profile()
    blind = sampler.certified_weight_floor(
        profile=profile, draws=400_000, min_p=SAMPLER_MIN_CHI_SQUARE_P
    )
    assert blind["certified_above_weight"] >= 0.0
    assert blind["prefix_scan"]
    required = [
        row["required_draws"] for row in blind["prefix_scan"]
        if row["required_draws"] is not None
    ]
    #: removing the very bottom of the distribution is the hardest thing to
    #: see, which is exactly the blind spot the capability must record
    assert required[0] == max(required)
    assert any(
        row["detectable_at_the_registered_draws"]
        for row in blind["prefix_scan"]
    )


def test_block_dispersion_explains_the_weak_arm() -> None:
    _weights, profile = _tiny_profile()
    dispersion = sampler.block_profile_dispersion(profile)
    assert dispersion["block_sum_coefficient_of_variation"] > 0.0
    assert dispersion["uniform_position_noncentrality"] > 0.0


# --------------------------------------------------------------------------- #
# the round re-types nothing it inherits
# --------------------------------------------------------------------------- #
def test_no_registered_constant_is_re_typed_in_this_round() -> None:
    """Registered names must be imported. A re-typed threshold is a second
    source of truth and this program has been bitten by one."""
    protected = {
        "TIE_TOLERANCE", "DID_ALPHA", "DID_MAPS_PER_ARM", "DID_SAMPLE_ROWS",
        "DID_SD_EQUIVALENCE_BOUND", "DID_TESTS_IN_FAMILY", "SAMPLER_DRAWS",
        "SAMPLER_BLOCK_EDGES", "SAMPLER_MAX_ABS_Z", "SAMPLER_MIN_CHI_SQUARE_P",
        "SAMPLER_EPOCHS", "WATCHDOG_ANON_BYTES", "WATCHDOG_SWAP_GROWTH_BYTES",
        "WATCHDOG_MEM_AVAILABLE_BYTES", "WATCHDOG_SAMPLE_INTERVAL_S",
        "GRAPH_K", "TRUTH_PROBE_ROWS", "DIMENSION",
    }
    for name in (
        "basemap/round0245_guard.py", "basemap/round0245_did.py",
        "basemap/round0245_sampler.py", "experiments/round0245_nodes.py",
    ):
        with open(os.path.join(REPO, name), encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assert target.id not in protected, (
                            f"{name} re-types the registered name {target.id}"
                        )


def test_the_did_gate_imports_r0244s_populations_rather_than_redefining_them() -> None:
    from basemap.round0244_prereq import did_populations as registered

    source = did.did_populations_v2.__doc__ or ""
    assert "imported" in source.lower()
    assert registered.__module__ == "basemap.round0244_prereq"
    assert did.did_populations_v2.__module__ == "basemap.round0245_did"
    assert TIE_TOLERANCE == 1e-6


def test_this_round_adds_only_its_own_files_and_the_one_declared_edit() -> None:
    """R0245 imports rounds 0215-0244 read-only except for the one edit the
    mandate authorises: the A1 fix inside R0244's watchdog module."""
    import subprocess as _sp  # noqa: PLC0415 - test-side git query only

    #: R0246 note: the endpoint was `HEAD` plus the working tree, so a later
    #: round adding a file turned this assertion red. Pinned to R0245's own
    #: release commit - the range it was written to check.
    committed = _sp.run(
        ["git", "-C", REPO, "diff", "--name-only",
         "8f159e5bcc81dbbd9079b026a5791908e82a4612",
         "c94a1401dc33b71e045925bf28cfde543457f9d9"],
        check=False, capture_output=True, text=True,
    ).stdout.split()
    worktree: list[str] = []
    allowed = {
        #: the single authorised edit to a reviewed module
        "basemap/round0244_guard.py",
        "basemap/round0245_guard.py",
        "basemap/round0245_did.py",
        "basemap/round0245_sampler.py",
        "experiments/round0245_nodes.py",
        "experiments/prepare_round0245_queue.py",
        "tests/test_round0245_contract.py",
        "tests/test_round0245_cpu_smoke.py",
    }
    changed = set(committed + worktree)
    assert changed <= allowed, sorted(changed - allowed)
