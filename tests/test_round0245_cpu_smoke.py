"""R0245 CPU smoke — all three node entry paths, executed end to end.

R0216 died on a `NameError`, R0236 on an arity mismatch, R0242 attempt 1 on a
module absent from the release venv. Each lived on a path nothing executed
until the node reached it, and no static check can see any of them. So this
file builds a miniature world — real `.npy` files, a real sealed R0244-shaped
watchdog receipt, a real weight array — and calls `run_job` for each of the
three actions at `400` rows instead of `100,000,000`.

Nothing is stubbed. The three guard positive controls run for real against
live anonymous memory, the DiD decision gate routes its planted branches, the
arm-assignment clamp runs on planted vectors that contain the exact shape of
probe `21785`, and the mis-sampler family is drawn and scored by the imported
fidelity check.
"""
from __future__ import annotations

import os
import shutil
import uuid

import numpy as np
import pytest

from basemap import round0113_prompt_contrast as prompt_contract
from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_save_new_npy, atomic_write_new_json
from basemap.round0238_rung5 import json_safe
from basemap import round0245_guard as guard_module
from experiments import round0245_nodes as nodes

TINY_ROWS = 400
TINY_PROBE = 400
TINY_K = 15
TINY_DIM = 16
RELEASE = "0" * 40
SMOKE_ROOT = "/data/latent-basemap/tests"

#: The tiny world's stand-in for R0244's sealed trace: a peak of 900 and a
#: steepest one-second rise of 400.
TINY_TRACE = [[0, 100], [1, 200], [2, 600], [3, 900], [4, 850]]
TINY_TRACE_SLOPE = 400
TINY_TRACE_PEAK = 900
TINY_HEADROOM = 100


def _npy(path: str, array: np.ndarray) -> dict:
    atomic_save_new_npy(path, array)
    return expected_input_signature(path)


class _TinyWorld:
    """Every byte the three nodes read, at 400 rows."""

    def __init__(self, root: str) -> None:
        os.makedirs(root, exist_ok=True)
        self.root = root
        rng = np.random.default_rng(245_000)

        substrate = rng.normal(size=(TINY_ROWS, TINY_DIM)).astype(np.float32)
        substrate /= np.linalg.norm(substrate, axis=1, keepdims=True)
        self.substrate_matrix = substrate
        self.substrate = _npy(
            os.path.join(root, "substrate.f32.npy"), substrate
        )

        graph_ids = np.stack([
            np.array(
                [(row + offset) % TINY_ROWS for offset in range(1, TINY_K + 1)],
                dtype=np.int32,
            )
            for row in range(TINY_ROWS)
        ])
        self.graph_ids = _npy(
            os.path.join(root, "graph-ids.i32.npy"), graph_ids
        )
        probe_rows = np.arange(TINY_PROBE, dtype=np.int64)
        self.probe_rows = _npy(
            os.path.join(root, "probe-query-rows.i64.npy"), probe_rows
        )

        truth_ids = graph_ids.astype(np.int32)
        truth_cos = np.zeros((TINY_PROBE, TINY_K), dtype=np.float32)
        for row in range(TINY_PROBE):
            cosines = substrate[row] @ substrate[graph_ids[row]].T
            truth_cos[row] = np.sort(cosines)[::-1]
        #: probe 7 is the tiny world's probe 21785: its k-th truth cosine sits
        #: a few tolerances above every candidate's recomputed cosine, so the
        #: value test demotes a candidate the id test counts.
        truth_cos[7, -1] = np.float32(float(truth_cos[7, -1]) + 5e-6)
        self.truth_ids = _npy(os.path.join(root, "truth-ids.i32.npy"), truth_ids)
        self.truth_cos = _npy(os.path.join(root, "truth-cos.f32.npy"), truth_cos)

        #: Ten density levels of forty rows, so every decile that carries loss
        #: also carries intact companions for R0228's exact match.
        strict = np.zeros(TINY_PROBE, dtype=np.int16)
        tie = np.zeros(TINY_PROBE, dtype=np.int16)
        strict[::10] = 2
        tie[::10] = 2
        strict[5::10] = 1
        tie[7] = 1  # the planted tie > strict row
        self.strict_vector = strict
        self.tie_vector = tie
        self.strict_builder = _npy(
            os.path.join(root, "probe-builder-missing.i16.npy"), strict
        )
        self.tie_builder = _npy(
            os.path.join(root, "probe-tie-builder-missing.i16.npy"), tie
        )

        #: A sealed receipt shaped exactly like R0244's.
        receipt_path = os.path.join(root, "host-watchdog.json")
        atomic_write_new_json(
            receipt_path,
            prompt_contract.seal(json_safe({
                "schema": "round0244-threaded-host-watchdog-and-true-peak-v1",
                "host_watchdog": {
                    "anonymous_trace_by_second": TINY_TRACE,
                    "budget_headroom_bytes": TINY_HEADROOM,
                    "thread_peak_anonymous_bytes": TINY_TRACE_PEAK,
                },
                "fuzzy": {"stripes": 50},
                "stage_walls": {"fuzzy_symmetrisation_s": 174.0},
            })),
        )
        self.watchdog_receipt = expected_input_signature(receipt_path)

        edges = 60_000
        weights = rng.beta(0.7, 2.0, size=edges).astype(np.float32)
        weights = np.maximum(weights, np.float32(1e-6))
        weights[rng.integers(0, edges, size=400)] = np.float32(1.0)
        self.edge_weights = weights
        self.edges_wts = _npy(os.path.join(root, "edges-wts.f32.npy"), weights)


def _patch(monkeypatch, world: _TinyWorld) -> None:
    monkeypatch.setattr(nodes, "ROWS", TINY_ROWS)
    monkeypatch.setattr(nodes, "TRUTH_PROBE_ROWS", TINY_PROBE)
    monkeypatch.setattr(nodes, "DIMENSION", TINY_DIM)
    monkeypatch.setattr(nodes, "COSINE_AGREEMENT_ROWS", 100)
    monkeypatch.setattr(
        nodes, "R0244_TRIP_TRACE_SLOPE_BYTES_PER_S", TINY_TRACE_SLOPE
    )
    monkeypatch.setattr(
        nodes, "R0244_WATCHDOG_RECEIPT_PEAK_BYTES", TINY_TRACE_PEAK
    )
    monkeypatch.setattr(nodes, "R0244_BUDGET_HEADROOM_BYTES", TINY_HEADROOM)
    monkeypatch.setattr(
        nodes, "R0243_STRICT_BUILDER_MISSING_EDGES",
        int(np.asarray(world.strict_vector, dtype=np.int64).sum()),
    )
    monkeypatch.setattr(
        nodes, "R0243_TIE_AWARE_BUILDER_MISSING_EDGES",
        int(np.asarray(world.tie_vector, dtype=np.int64).sum()),
    )
    monkeypatch.setattr(nodes, "SAMPLER_BLOCK_EDGES", 4_096)


@pytest.fixture()
def scratch():
    os.makedirs(SMOKE_ROOT, exist_ok=True)
    root = os.path.join(SMOKE_ROOT, f"round0245-{uuid.uuid4().hex}")
    os.makedirs(root)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture()
def world(scratch) -> _TinyWorld:
    return _TinyWorld(os.path.join(scratch, "world"))


@pytest.fixture()
def armed(monkeypatch, scratch):
    """Every node needs an enforceable cooperative abort flag to start."""
    logs = os.path.join(scratch, "logs")
    os.makedirs(logs, exist_ok=True)
    monkeypatch.setenv("ROUNDRUN_ABORT_FLAG", os.path.join(logs, "node.abort"))
    return logs


ACTIVE = {"manifest": {"round_id": "0245", "release_sha": RELEASE}}


def test_guard_entry_path_runs_end_to_end(monkeypatch, scratch, world, armed):
    """Drives all three positive controls against live anonymous memory."""
    _patch(monkeypatch, world)
    output = os.path.join(scratch, "artifacts-guard")
    nodes.run_job(ACTIVE, {
        "action": nodes.GUARD_ACTION,
        "outputs": [output],
        "r0244_watchdog_receipt": world.watchdog_receipt,
    })
    receipt = prompt_contract.read_sealed(
        os.path.join(output, nodes.GUARD_FILE), label="tiny guard"
    )
    assert receipt["schema"] == nodes.GUARD_SCHEMA
    closure = receipt["closure"]
    assert closure["holds"] is True
    assert closure["thread_death_control"]["holds"] is True
    assert closure["missing_flag_control"]["holds"] is True
    assert closure["allocation_slope_control"]["holds"] is True
    #: the compliant arm stops inside its headroom, the breaching one does not
    arms = closure["allocation_slope_control"]["arms"]
    assert (
        arms["compliant"]["growth_after_trip_bytes"]
        <= arms["compliant"]["headroom_bytes"]
    )
    assert (
        arms["breaching"]["growth_after_trip_bytes"]
        > arms["breaching"]["headroom_bytes"]
    )
    assert arms["compliant"]["requirement"]["requirement_holds"] is True
    assert arms["breaching"]["requirement"]["requirement_holds"] is False
    #: and R0243's own stripe is scored against the derived requirement
    assert closure["r0243_stripe_against_the_requirement"][
        "requirement_holds"
    ] is False
    assert receipt["r0244_slope_binding"]["slope_agrees"] is True
    assert receipt["sampler_liveness"]["sampling_thread_alive"] is True
    #: the controls clean up after themselves
    assert not os.path.exists(os.path.join(output, "slope-control.abort"))
    assert not os.path.exists(os.path.join(output, "thread-death-control.abort"))


def test_guard_node_refuses_a_receipt_whose_slope_is_not_the_registered_one(
    monkeypatch, scratch, world, armed
):
    """A slope re-derived from another run's trace measures nothing here."""
    _patch(monkeypatch, world)
    monkeypatch.setattr(nodes, "R0244_TRIP_TRACE_SLOPE_BYTES_PER_S", 1)
    with pytest.raises(nodes.Round0245Error):
        nodes.run_job(ACTIVE, {
            "action": nodes.GUARD_ACTION,
            "outputs": [os.path.join(scratch, "artifacts-guard-bad")],
            "r0244_watchdog_receipt": world.watchdog_receipt,
        })


def _did_job(world: _TinyWorld, output: str) -> dict:
    return {
        "action": nodes.DID_ACTION,
        "outputs": [output],
        "r0242_probe_builder_missing_edges": world.strict_builder,
        "r0243_probe_tie_aware_builder_missing_edges": world.tie_builder,
        "truth_cos": world.truth_cos,
        "truth_ids": world.truth_ids,
        "probe_query_rows": world.probe_rows,
        "substrate_array": world.substrate,
        "graph_ids": world.graph_ids,
    }


def test_did_entry_path_runs_end_to_end(monkeypatch, scratch, world, armed):
    _patch(monkeypatch, world)
    output = os.path.join(scratch, "artifacts-did")
    nodes.run_job(ACTIVE, _did_job(world, output))
    receipt = prompt_contract.read_sealed(
        os.path.join(output, nodes.DID_FILE), label="tiny did"
    )
    assert receipt["schema"] == nodes.DID_SCHEMA
    assert receipt["displacement_measured"] is False
    assert receipt["decision_gate"]["holds"] is True
    assert receipt["decision_gate"]["unpowered_nulls_returned_indeterminate"] == 3
    audit = receipt["tie_monotonicity_audit"]
    assert audit["rows_with_tie_above_strict"] == 1
    assert audit["violating_rows"] == [7]
    assert audit["genuine_rows_after_the_clamp"] == (
        audit["genuine_rows_before_the_clamp"] - 1
    )
    assert receipt["arm_disjointness"]["arms_are_disjoint"] is True
    assert receipt["arm_disjointness"]["population_partition"][
        "populations_partition_the_probe"
    ] is True
    forensics = receipt["tie_violation_forensics"]
    assert forensics["rows"][0]["probe_row"] == 7
    assert forensics["rows"][0][
        "candidates_in_truth_scored_invalid_by_the_tolerance"
    ] >= 1
    assert receipt["cosine_agreement_profile"]["matched_candidate_truth_pairs"] > 0
    assert receipt["permutation_design_cost"]["unrestricted_labellings"] == 252
    enforcement = receipt["enforcement_poll_spacing"]
    assert enforcement["requirement"]["requirement_holds"] is True
    assert enforcement["own_slope_requirement"] is not None
    assert enforcement["measured_slope_bytes_per_s"] is not None
    assert enforcement["worst_case_requirement"]["slope_bytes_per_s"] == float(
        guard_module.R0244_MEASURED_SLOPE_BYTES_PER_S
    )
    assert os.path.exists(
        os.path.join(output, "vectors", "did-v2-placebo-a-rows.i64.npy")
    )


def test_did_node_refuses_a_vector_that_is_not_r0243s(
    monkeypatch, scratch, world, armed
):
    """A different loss vector cannot be re-analysed as if it were this one."""
    _patch(monkeypatch, world)
    monkeypatch.setattr(nodes, "R0243_STRICT_BUILDER_MISSING_EDGES", 1)
    with pytest.raises(nodes.Round0245Error):
        nodes.run_job(
            ACTIVE, _did_job(world, os.path.join(scratch, "artifacts-did-bad"))
        )


def test_sampler_entry_path_runs_end_to_end(monkeypatch, scratch, world, armed):
    _patch(monkeypatch, world)
    output = os.path.join(scratch, "artifacts-sampler")
    nodes.run_job(ACTIVE, {
        "action": nodes.SAMPLER_ACTION,
        "outputs": [output],
        "edges_wts": world.edges_wts,
        "draws": 8_000_000,
    })
    receipt = prompt_contract.read_sealed(
        os.path.join(output, nodes.SAMPLER_FILE), label="tiny sampler"
    )
    assert receipt["schema"] == nodes.SAMPLER_SCHEMA
    assert receipt["correct_sampler_reference"]["holds"] is True
    assert "uniform_positions" in receipt["mis_sampler_battery"]["caught"]
    assert receipt["draw_floor"]["registered_draws_clear_the_floor"] is True
    assert receipt["limits"]["certified_above_weight"] >= 0.0
    assert receipt["verdict_arms"]["correct_sampler_passes_at_a_fresh_seed"]
    assert receipt["edges_republished"] is False
    assert receipt["sampler_liveness"]["sampling_thread_alive"] is True


def test_sampler_node_refuses_a_draw_count_below_the_family(
    monkeypatch, scratch, world, armed
):
    """The draw count is part of the check, and the node fails closed on it."""
    _patch(monkeypatch, world)
    with pytest.raises(nodes.Round0245Error):
        nodes.run_job(ACTIVE, {
            "action": nodes.SAMPLER_ACTION,
            "outputs": [os.path.join(scratch, "artifacts-sampler-bad")],
            "edges_wts": world.edges_wts,
            "draws": 2_000,
        })


def test_run_job_refuses_an_unknown_action(scratch, armed) -> None:
    with pytest.raises(nodes.Round0245Error):
        nodes.run_job(ACTIVE, {"action": "not-a-round-0245-node"})
