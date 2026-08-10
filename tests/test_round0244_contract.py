"""R0244 contract — every guard this round adds, shown catching a planted defect.

`AGENT_STARTUP.md`: "Any guard, detector, or tripwire the round adds must ship
a positive control — a test that plants the defect and proves the guard catches
it. A guard whose test suite contains no failing input is untested at its only
job." This round adds four, and each one below is exercised on an input that
must make it fail as well as on one that must not.
"""
from __future__ import annotations

import ast
import io
import os
import tokenize
import uuid

import numpy as np
import pytest

from basemap.round0244_guard import (
    Round0244Error,
    ThreadedHostWatchdog,
    boundary_only_understatement,
    run_watchdog_positive_control,
)
from basemap.round0244_prereq import (
    DID_ALPHA,
    NEAR_IDENTICAL_JACCARD,
    R0243_TOTAL_WEIGHT,
    SAMPLER_MAX_ABS_Z,
    classify_text_pair,
    did_populations,
    did_registration,
    did_requirement,
    excerpt,
    jaccard,
    permutation_resolution,
    sampling_fidelity,
    two_level_weight_sample,
    uniform_sample_control,
    weight_block_profile,
)
from experiments.round0242_nodes import _HostWatchdog

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE_PATH_FILES = (
    "basemap/round0244_guard.py",
    "basemap/round0244_prereq.py",
    "experiments/round0244_nodes.py",
)
SMOKE_ROOT = "/data/latent-basemap/tests"


@pytest.fixture()
def scratch():
    import shutil

    os.makedirs(SMOKE_ROOT, exist_ok=True)
    root = os.path.join(SMOKE_ROOT, f"round0244-contract-{uuid.uuid4().hex}")
    os.makedirs(root)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# A. the host watchdog
# --------------------------------------------------------------------------- #
def test_watchdog_positive_control_fires_on_a_planted_allocation(scratch):
    """The whole point: a guard nobody has seen fire is not evidence."""
    evidence = run_watchdog_positive_control(
        flag_path=os.path.join(scratch, "planted.abort"),
        plant_bytes=768 << 20, headroom_bytes=256 << 20,
        interval_s=0.02, timeout_s=60.0,
    )
    assert evidence["holds"] is True
    assert evidence["watchdog_fired"] is True
    assert evidence["budget_exceeded_by_bytes"] > 0
    assert evidence["trip_rule"] == "node-declared-anonymous-budget"
    assert evidence["abort_flag_written"] is True
    assert evidence["abort_flag_reason_readable"] is True
    assert evidence["poll_raised"] is True
    #: The stripe-shaped loop unwinds at its FIRST unit — that is the property
    #: R0243's stage did not have, because nothing inside it read the guard.
    assert evidence["stripe_loop_unwound_at_unit"] == 0
    assert evidence["observed_after_s"] < 60.0


def test_watchdog_does_not_fire_on_a_clean_stage(scratch):
    """The negative half. A guard that always fires is also not evidence."""
    watchdog = ThreadedHostWatchdog(
        anonymous_budget_bytes=1 << 62, interval_s=0.02,
        abort_flag_path=os.path.join(scratch, "clean.abort"),
        label="clean",
    )
    with watchdog:
        keep = np.zeros(1 << 20, dtype=np.uint8)
        keep[:] = 1
        for index in range(5):
            watchdog.poll(f"clean unit {index}")
        del keep
    receipt = watchdog.receipt()
    assert receipt["fired"] is False
    assert receipt["trip_reason"] is None
    assert receipt["abort_flag_written"] is False
    assert not os.path.exists(os.path.join(scratch, "clean.abort"))


def test_watchdog_samples_without_any_call_site(scratch):
    """The thread must observe a peak that NO poll() is anywhere near.

    This is R0243's defect in miniature: allocate inside a stretch with no
    boundary, and show the boundary column misses what the thread column sees.
    """
    import time

    watchdog = ThreadedHostWatchdog(
        anonymous_budget_bytes=1 << 62, interval_s=0.02, label="in-stage",
    )
    with watchdog:
        watchdog.poll("before the stage")
        transient = np.ones(600 << 20, dtype=np.uint8)
        transient[:] = 3
        time.sleep(0.30)
        peak_while_held = int(watchdog.peak_anonymous_bytes)
        del transient
        import gc

        gc.collect()
        time.sleep(0.20)
        watchdog.poll("after the stage")
    receipt = watchdog.receipt()
    comparison = boundary_only_understatement(receipt)
    assert receipt["samples"] > 5
    assert peak_while_held > receipt["boundary_peak_anonymous_bytes"]
    assert comparison["understatement_bytes"] > 0
    assert comparison["understatement_multiple"] > 1.0


def test_threaded_watchdog_inherits_r0242s_rule_rather_than_retyping_it():
    assert issubclass(ThreadedHostWatchdog, _HostWatchdog)
    receipt = ThreadedHostWatchdog(anonymous_budget_bytes=1 << 62).receipt()
    #: R0242's conjunctive thresholds, unchanged, still in the receipt.
    assert receipt["swap_growth_threshold_bytes"] == 4 * (1 << 30)
    assert receipt["anonymous_threshold_bytes"] == 60 * (1 << 30)
    assert receipt["mem_available_threshold_bytes"] == 16 * (1 << 30)
    assert receipt["guard_axis"] == "anonymous, never RSS"


# --------------------------------------------------------------------------- #
# D. the edge list as a sampling distribution
# --------------------------------------------------------------------------- #
def _planted_weights(seed: int = 7, edges: int = 200_000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    weights = rng.beta(0.6, 2.0, size=edges).astype(np.float32)
    weights = np.maximum(weights, np.float32(1e-6))
    weights[rng.integers(0, edges, size=edges // 40)] = np.float32(1.0)
    return weights


def test_sampling_fidelity_accepts_a_correct_two_level_draw():
    weights = _planted_weights()
    profile = weight_block_profile(weights, block=4_096, bins=20, epochs=200)
    sample = two_level_weight_sample(
        weights, profile=profile, draws=400_000, seed=11
    )
    verdict = sampling_fidelity(profile=profile, sample=sample)
    assert verdict["holds"] is True
    assert abs(verdict["mean_weight_z"]) <= SAMPLER_MAX_ABS_Z
    #: The arm that separates weight-proportional from uniform: E[w] under
    #: weight-proportional sampling is sum(w^2)/sum(w), strictly above the
    #: arithmetic mean for any non-constant weight vector.
    assert verdict["expected_mean_weight"] > verdict["arithmetic_mean_weight"]


def test_sampling_fidelity_rejects_a_uniform_mis_sampler():
    """The positive control. A check that cannot reject is decoration."""
    weights = _planted_weights()
    profile = weight_block_profile(weights, block=4_096, bins=20, epochs=200)
    control = uniform_sample_control(
        weights, profile=profile, draws=200_000, seed=13
    )
    assert control["rejected"] is True
    assert control["arms"]["mean_weight_z_within_bound"] is False


def test_weight_profile_keeps_no_more_than_one_float_per_block():
    weights = _planted_weights(edges=100_000)
    profile = weight_block_profile(weights, block=4_096, bins=20, epochs=200)
    assert profile["blocks"] == -(-100_000 // 4_096)
    assert profile["block_sums_bytes"] == profile["blocks"] * 8
    assert profile["block_sums_bytes"] < weights.nbytes // 100
    assert profile["total_weight"] == pytest.approx(
        float(np.asarray(weights, dtype=np.float64).sum()), rel=1e-12
    )
    assert (
        profile["edges_sampled_at_least_once_in_a_run"]
        + profile["edges_never_sampled_in_a_run"] == profile["edges"]
    )


def test_r0243_total_weight_is_a_bound_constant_not_a_recomputation():
    assert R0243_TOTAL_WEIGHT == 875131479.5054033


# --------------------------------------------------------------------------- #
# B. the DiD registration
# --------------------------------------------------------------------------- #
def test_did_resolution_is_computed_and_small_arms_are_refused():
    """review-0228-01: a design whose floor exceeds its own correction cannot
    produce any outcome. That arithmetic is done here BEFORE the arm is fixed."""
    assert permutation_resolution(treated_maps=3, null_maps=3)[
        "can_reject_under_the_family_correction"
    ] is False
    assert permutation_resolution(treated_maps=4, null_maps=4)[
        "can_reject_under_the_family_correction"
    ] is False
    assert permutation_resolution(treated_maps=5, null_maps=5)[
        "can_reject_under_the_family_correction"
    ] is True
    registration = did_registration()
    assert registration["smallest_arm_that_can_reject"] == 5
    assert registration["selected_design"]["labellings"] == 252
    assert registration["alpha_family_wise"] == DID_ALPHA
    with pytest.raises(Round0244Error):
        did_registration(arm_sizes=(3,), maps_per_arm=3)


def test_did_decision_rule_refuses_to_call_an_unpowered_null_harmless():
    rule = did_registration()["decision_rule"]
    assert "INDETERMINATE" in rule
    assert "never harmless" in rule
    assert "power" in rule


def test_did_requirement_states_a_cost_rather_than_a_hope():
    requirement = did_requirement()
    assert requirement["maps_required"] == 5
    assert requirement["gather_bytes"] == 0
    assert requirement["estimated_gpu_hours"] == 50.0
    assert requirement["fits_in_the_rung_cap"] is False
    assert requirement["label"] == "prediction"


def test_did_populations_are_row_sets_and_carry_no_displacement():
    rng = np.random.default_rng(244)
    rows = 4_000
    strict = np.zeros(rows, dtype=np.int64)
    tie = np.zeros(rows, dtype=np.int64)
    strict[:600] = rng.integers(1, 5, size=600)
    tie[:200] = strict[:200]
    cosine = rng.uniform(0.2, 0.95, size=rows)
    populations = did_populations(
        strict_builder_missing=strict, tie_aware_builder_missing=tie,
        kth_cosine=cosine, sample_rows=150,
    )
    assert populations["displacement_computed"] is False
    assert populations["genuine_rows_total"] == 200
    assert populations["tie_forgiven_rows_total"] == 400
    assert populations["placebo_halves_disjoint"] is True
    assert (
        populations["genuine"]["decile_counts_lost"]
        == populations["genuine"]["decile_counts_control"]
    )
    with pytest.raises(Round0244Error):
        did_populations(
            strict_builder_missing=np.zeros(rows, dtype=np.int64),
            tie_aware_builder_missing=np.zeros(rows, dtype=np.int64),
            kth_cosine=cosine,
        )


# --------------------------------------------------------------------------- #
# C. the near-duplicate classifier
# --------------------------------------------------------------------------- #
def test_near_duplicate_classifier_separates_the_registered_categories():
    body = "the quick brown fox jumps over the lazy dog " * 6
    assert classify_text_pair(body, body)["category"] == "identical"
    tweaked = body.replace("lazy dog", "lazy cat", 1)
    near = classify_text_pair(body, tweaked)
    assert near["category"] == "near_identical"
    assert near["jaccard_char_5gram"] >= NEAR_IDENTICAL_JACCARD
    other = "quantum chromodynamics confines colour charge at low energy " * 6
    assert classify_text_pair(body, other)["category"] == "distinct_text"
    assert jaccard("", "") == 1.0
    assert jaccard("abcdefgh", "") == 0.0
    assert excerpt("a  b\n c", chars=100) == "a b c"
    assert excerpt("x" * 500).endswith(" ...")


# --------------------------------------------------------------------------- #
# process: nothing re-typed, nothing signalling
# --------------------------------------------------------------------------- #
FORBIDDEN_TOKENS = frozenset({
    "signal", "kill", "killpg", "SIGKILL", "SIGTERM", "SIGINT", "terminate",
    "subprocess", "multiprocessing", "ptrace", "pkill", "psutil",
})


def test_no_node_path_file_contains_a_signalling_construct():
    """`tokenize`-based, so a hazard NAMED in prose cannot excuse one in code."""
    for relative in NODE_PATH_FILES:
        with open(os.path.join(REPO, relative), encoding="utf-8") as handle:
            source = handle.read()
        hits = []
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type in (tokenize.STRING, tokenize.COMMENT):
                continue
            if token.string in FORBIDDEN_TOKENS:
                hits.append((relative, token.start[0], token.string))
        assert hits == [], hits


REVIEWED_NAMES = frozenset({
    "loss_decomposition", "post_canonical_tripwire", "symmetrised_degree_once",
    "canonical_undirected_degrees", "weight_distribution",
    "density_matched_control", "true_neighbour_scatter", "map_scale",
    "exact_displacement_permutation", "holm_bonferroni",
    "_fuzzy_symmetrise_blocked", "_blocked_descending_sort",
    "_check_runner_abort", "_memmap_attestation", "_readonly_memmap",
    "verify_inheritance", "io_counters", "host_anonymous_bytes",
})


def test_no_registered_check_is_re_typed_in_this_round():
    for relative in NODE_PATH_FILES:
        with open(os.path.join(REPO, relative), encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=relative)
        defined = {
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        overlap = defined & REVIEWED_NAMES
        assert overlap == set(), (relative, overlap)


def test_this_round_adds_only_new_files():
    """R0244 imports rounds 0215-0243 read-only; it edits none of them."""
    import subprocess as _sp  # noqa: PLC0415 - test-side git query only

    committed = _sp.run(
        ["git", "-C", REPO, "diff", "--name-only", "370f715", "HEAD"],
        check=False, capture_output=True, text=True,
    ).stdout.split()
    worktree = [
        line[3:] for line in _sp.run(
            ["git", "-C", REPO, "status", "--porcelain"],
            check=False, capture_output=True, text=True,
        ).stdout.splitlines() if line
    ]
    changed = committed + worktree
    allowed = {
        "basemap/round0244_guard.py",
        "basemap/round0244_prereq.py",
        "experiments/round0244_nodes.py",
        "experiments/prepare_round0244_queue.py",
        "tests/test_round0244_contract.py",
        "tests/test_round0244_cpu_smoke.py",
    }
    assert set(changed) <= allowed, sorted(set(changed) - allowed)
