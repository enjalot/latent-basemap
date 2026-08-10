"""R0246 CPU smoke — all three node entry paths, executed end to end.

R0216 died on a `NameError`, R0236 on an arity mismatch, R0242 attempt 1 on a
module absent from the release venv. Each lived on a path nothing executed
until the node reached it. So this file builds a miniature world — real `.npy`
files, a real sealed R0245-shaped sampler receipt, a real weight array — and
calls `run_job` for each of the three actions at `400` rows instead of
`100,000,000`.

Nothing is stubbed: the reviewer's three controls run for real against live
threads and a live filesystem, the novel attack battery runs, the polled
sampler draws and is scored by the imported fidelity check, and the tie-aware
precision profile recomputes cosines in both precisions.
"""
from __future__ import annotations

import os
import shutil
import uuid

import numpy as np
import pytest

from basemap import round0113_prompt_contrast as prompt_contract
from basemap import round0246_guard as guard
from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_save_new_npy, atomic_write_new_json
from basemap.round0238_rung5 import json_safe
from experiments import round0246_nodes as nodes

TINY_ROWS = 400
TINY_PROBE = 400
TINY_K = 15
TINY_DIM = 16
RELEASE = "0" * 40
SMOKE_ROOT = "/data/latent-basemap/tests"

#: The tiny world's stand-in for R0245's sealed sampler receipt.
TINY_SAMPLES = 100
TINY_BOUNDARY_POLLS = 4
TINY_EXPECTED_SAMPLES = 104.0


def _npy(path: str, array: np.ndarray) -> dict:
    atomic_save_new_npy(path, array)
    return expected_input_signature(path)


class _TinyWorld:
    """Every byte the three nodes read, at 400 rows."""

    def __init__(self, root: str) -> None:
        os.makedirs(root, exist_ok=True)
        self.root = root
        rng = np.random.default_rng(246_000)

        substrate = rng.normal(size=(TINY_ROWS, TINY_DIM)).astype(np.float32)
        substrate /= np.linalg.norm(substrate, axis=1, keepdims=True)
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
            truth_cos[row] = np.sort(
                substrate[row] @ substrate[graph_ids[row]].T
            )[::-1]
        self.truth_ids = _npy(os.path.join(root, "truth-ids.i32.npy"), truth_ids)
        self.truth_cos = _npy(os.path.join(root, "truth-cos.f32.npy"), truth_cos)

        receipt_path = os.path.join(root, "sampler-power.json")
        atomic_write_new_json(
            receipt_path,
            prompt_contract.seal(json_safe({
                "schema": "round0245-sampler-power-and-limits-v1",
                "host_watchdog": {
                    "samples": TINY_SAMPLES,
                    "boundary_polls": TINY_BOUNDARY_POLLS,
                    "expected_samples_at_interval": TINY_EXPECTED_SAMPLES,
                    "sampled_wall_s": 26.0,
                    "sample_coverage": TINY_SAMPLES / TINY_EXPECTED_SAMPLES,
                },
            })),
        )
        self.sampler_receipt = expected_input_signature(receipt_path)

        edges = 60_000
        weights = rng.beta(0.7, 2.0, size=edges).astype(np.float32)
        weights = np.maximum(weights, np.float32(1e-6))
        weights[rng.integers(0, edges, size=400)] = np.float32(1.0)
        self.edges_wts = _npy(os.path.join(root, "edges-wts.f32.npy"), weights)
        #: what the tiny sampler node must reproduce, drawn here once
        from basemap.round0244_prereq import (
            two_level_weight_sample,
            weight_block_profile,
        )

        profile = weight_block_profile(weights, block=4_096)
        self.tiny_distinct = int(
            two_level_weight_sample(
                weights, profile=profile, draws=200_000, seed=7
            )["distinct_edges_drawn"]
        )


def _patch(monkeypatch, world: _TinyWorld) -> None:
    monkeypatch.setattr(nodes, "ROWS", TINY_ROWS)
    monkeypatch.setattr(nodes, "TRUTH_PROBE_ROWS", TINY_PROBE)
    monkeypatch.setattr(nodes, "DIMENSION", TINY_DIM)
    monkeypatch.setattr(nodes, "SAMPLER_BLOCK_EDGES", 4_096)
    monkeypatch.setattr(nodes, "R0245_SEALED_SAMPLER_SAMPLES", TINY_SAMPLES)
    monkeypatch.setattr(
        nodes, "R0245_SEALED_SAMPLER_BOUNDARY_POLLS", TINY_BOUNDARY_POLLS
    )
    monkeypatch.setattr(
        nodes, "R0245_SEALED_SAMPLER_EXPECTED_SAMPLES", TINY_EXPECTED_SAMPLES
    )
    monkeypatch.setattr(
        nodes, "R0245_SEALED_DISTINCT_EDGES_DRAWN", world.tiny_distinct
    )


@pytest.fixture()
def scratch():
    os.makedirs(SMOKE_ROOT, exist_ok=True)
    root = os.path.join(SMOKE_ROOT, f"round0246-{uuid.uuid4().hex}")
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


ACTIVE = {"manifest": {"round_id": "0246", "release_sha": RELEASE}}


def test_guard_entry_path_runs_end_to_end(monkeypatch, scratch, world, armed):
    """All three reviewer controls and the novel battery, for real."""
    _patch(monkeypatch, world)
    output = os.path.join(scratch, "artifacts-guard")
    nodes.run_job(ACTIVE, {
        "action": nodes.GUARD_ACTION,
        "outputs": [output],
        "r0245_sampler_receipt": world.sampler_receipt,
    })
    receipt = prompt_contract.read_sealed(
        os.path.join(output, nodes.GUARD_FILE), label="tiny guard"
    )
    assert receipt["schema"] == nodes.GUARD_SCHEMA
    closure = receipt["closure"]
    assert closure["holds"] is True
    assert closure["a1_reviewer_oserror_control"]["holds"] is True
    assert closure["a3_reviewer_directory_control"]["holds"] is True
    assert closure["a2_reviewer_gap_replay_control"]["holds"] is True
    assert closure["novel_attacks"]["attacks_that_still_succeed"] == []
    assert closure["arms"]["a2_attempt_1s_gap_fails_again"] is True
    #: the coverage floor's basis is recomputed from the sealed receipt
    basis = closure["healthy_coverage_basis"]
    assert basis["thread_samples"] == TINY_SAMPLES - TINY_BOUNDARY_POLLS
    #: and the node's own guard passed all three node-tail gates
    assert receipt["sampler_liveness"]["holds"] is True
    assert receipt["abort_flag_landing"]["holds"] is True
    assert receipt["enforcement_poll_spacing"]["holds"] is True
    assert receipt["enforcement_poll_spacing"][
        "meets_the_registered_ceiling"
    ] is True


def test_guard_node_refuses_a_receipt_that_is_not_the_registered_basis(
    monkeypatch, scratch, world, armed
):
    _patch(monkeypatch, world)
    monkeypatch.setattr(nodes, "R0245_SEALED_SAMPLER_SAMPLES", 1)
    with pytest.raises(nodes.Round0246Error):
        nodes.run_job(ACTIVE, {
            "action": nodes.GUARD_ACTION,
            "outputs": [os.path.join(scratch, "artifacts-guard-bad")],
            "r0245_sampler_receipt": world.sampler_receipt,
        })


def _sampler_job(world: _TinyWorld, output: str) -> dict:
    return {
        "action": nodes.SAMPLER_ACTION,
        "outputs": [output],
        "edges_wts": world.edges_wts,
        "draws": 200_000,
        "seed": 7,
    }


def test_sampler_entry_path_runs_end_to_end(monkeypatch, scratch, world, armed):
    _patch(monkeypatch, world)
    output = os.path.join(scratch, "artifacts-sampler")
    nodes.run_job(ACTIVE, _sampler_job(world, output))
    receipt = prompt_contract.read_sealed(
        os.path.join(output, nodes.SAMPLER_FILE), label="tiny sampler"
    )
    assert receipt["schema"] == nodes.SAMPLER_SCHEMA
    assert receipt["sealed_draw_reproduction"]["distinct_agrees"] is True
    assert receipt["sealed_draw_reproduction"]["correct_sampler_still_passes"]
    gap = receipt["abort_read_gap_against_both_slopes"]
    assert gap["widest_abort_read_gap_s"] > 0.0
    assert gap["worst_case"]["meets_the_r0244_worst_case_slope"] is True
    assert gap["r0245_sealed_widest_gap_s"] == pytest.approx(24.46713631998864)
    assert receipt["the_first_multi_hour_training_node_is_unblocked"] is True
    assert receipt["edges_republished"] is False
    assert receipt["sampler_liveness"]["holds"] is True


def test_sampler_node_refuses_a_draw_that_is_not_the_sealed_one(
    monkeypatch, scratch, world, armed
):
    """A poll fix that changed the science must fail the node."""
    _patch(monkeypatch, world)
    monkeypatch.setattr(nodes, "R0245_SEALED_DISTINCT_EDGES_DRAWN", 1)
    with pytest.raises(nodes.Round0246Error):
        nodes.run_job(
            ACTIVE,
            _sampler_job(world, os.path.join(scratch, "artifacts-sampler-bad")),
        )


def test_tie_entry_path_runs_end_to_end(monkeypatch, scratch, world, armed):
    _patch(monkeypatch, world)
    output = os.path.join(scratch, "artifacts-tie")
    nodes.run_job(ACTIVE, {
        "action": nodes.TIE_ACTION,
        "outputs": [output],
        "truth_cos": world.truth_cos,
        "truth_ids": world.truth_ids,
        "probe_query_rows": world.probe_rows,
        "substrate_array": world.substrate,
        "graph_ids": world.graph_ids,
        "tie_rows": 200,
        "tie_seed": 246_005,
    })
    receipt = prompt_contract.read_sealed(
        os.path.join(output, nodes.TIE_FILE), label="tiny tie"
    )
    assert receipt["schema"] == nodes.TIE_SCHEMA
    assert receipt["the_tolerance_was_not_raised"] is True
    profile = receipt["tie_precision_profile"]
    assert profile["candidate_decisions_scored"] == 200 * TINY_K
    assert profile["verdict_flips"]["per_candidate_flip_rate"] is not None
    adjudication = receipt["claim_adjudication"]
    assert adjudication["claims_examined"] == len(
        __import__(
            "basemap.round0246_tie", fromlist=["TIE_AWARE_CLAIM_LEDGER"]
        ).TIE_AWARE_CLAIM_LEDGER
    )
    assert receipt["aggregate_only_control"]["holds"] is True
    assert receipt["sampler_liveness"]["holds"] is True


def test_every_node_action_refuses_to_start_without_the_flag(
    monkeypatch, scratch
):
    monkeypatch.delenv("ROUNDRUN_ABORT_FLAG", raising=False)
    for action in (
        nodes.GUARD_ACTION, nodes.SAMPLER_ACTION, nodes.TIE_ACTION,
    ):
        with pytest.raises(guard.Round0245Error):
            nodes.run_job(
                ACTIVE,
                {"action": action, "outputs": [os.path.join(scratch, action)]},
            )


def test_run_job_refuses_an_unknown_action(scratch, armed) -> None:
    with pytest.raises(nodes.Round0246Error):
        nodes.run_job(ACTIVE, {"action": "not-a-round-0246-node"})
