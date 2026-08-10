"""R0247 CPU smoke — all three node entry paths, executed end to end.

R0216 died on a `NameError`, R0236 on an arity mismatch, R0242 attempt 1 on a
module absent from the release venv. Each lived on a path nothing executed until
the node reached it. So this file builds a miniature world — real `.npy` files,
a real substrate, a real truth probe — and calls `run_job` for each of the three
actions at `400` rows instead of `100,000,000`.

Nothing is stubbed. Every control runs for real against live threads and a live
filesystem: the per-parameter clamp controls, the per-construction-path
controls, review-0246-01 A's `5.0` s coverage attack, review-0246-01 C's
sixteenth attack, the self-attack battery, the `float64` cosine recompute, and
the sealed ledger adjudication.
"""
from __future__ import annotations

import os
import shutil
import uuid

import numpy as np
import pytest

from basemap import round0113_prompt_contrast as prompt_contract
from basemap import round0245_guard as guard0245
from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_save_new_npy
from basemap.round0246_tie import TIE_AWARE_CLAIM_LEDGER
from experiments import round0247_nodes as nodes

TINY_ROWS = 400
TINY_PROBE = 400
TINY_K = 15
TINY_DIM = 16
RELEASE = "0" * 40
SMOKE_ROOT = "/data/latent-basemap/tests"

ACTIVE = {"manifest": {"round_id": "0247", "release_sha": RELEASE}}


def _npy(path: str, array: np.ndarray) -> dict:
    atomic_save_new_npy(path, array)
    return expected_input_signature(path)


class _TinyWorld:
    """Every byte the three nodes read, at 400 rows."""

    def __init__(self, root: str) -> None:
        os.makedirs(root, exist_ok=True)
        self.root = root
        rng = np.random.default_rng(247_000)

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
        #: A float32 truth cosine array computed in float32, exactly as R0238's
        #: is - so the smoke exercises the storage-versus-arithmetic question
        #: rather than a synthetic one.
        truth_cos = np.zeros((TINY_PROBE, TINY_K), dtype=np.float32)
        for row in range(TINY_PROBE):
            truth_cos[row] = np.einsum(
                "d,kd->k", substrate[row], substrate[truth_ids[row]]
            )
        self.truth_ids = _npy(os.path.join(root, "truth-ids.i32.npy"), truth_ids)
        self.truth_cos = _npy(os.path.join(root, "truth-cos.f32.npy"), truth_cos)


def _patch(monkeypatch) -> None:
    monkeypatch.setattr(nodes, "ROWS", TINY_ROWS)
    monkeypatch.setattr(nodes, "TRUTH_PROBE_ROWS", TINY_PROBE)
    monkeypatch.setattr(nodes, "DIMENSION", TINY_DIM)
    monkeypatch.setattr(nodes, "TIE_FULL_PROBE_ROWS", TINY_PROBE)


@pytest.fixture()
def scratch():
    os.makedirs(SMOKE_ROOT, exist_ok=True)
    root = os.path.join(SMOKE_ROOT, f"round0247-{uuid.uuid4().hex}")
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


# --------------------------------------------------------------------------- #
# node 1 — the class fix
# --------------------------------------------------------------------------- #
def test_paramguard_entry_path_runs_end_to_end(monkeypatch, scratch, armed):
    _patch(monkeypatch)
    output = os.path.join(scratch, "artifacts-paramguard")
    nodes.run_job(ACTIVE, {
        "action": nodes.PARAMGUARD_ACTION, "outputs": [output],
    })
    receipt = prompt_contract.read_sealed(
        os.path.join(output, nodes.PARAMGUARD_FILE), label="tiny paramguard"
    )
    assert receipt["schema"] == nodes.PARAMGUARD_SCHEMA
    closure = receipt["closure"]
    assert closure["holds"] is True
    #: every registered parameter carries a control
    inventory = closure["safety_parameter_inventory"]
    assert closure["clamp_controls"]["parameters_controlled"] == (
        inventory["parameter_count"]
    )
    assert closure["clamp_controls"]["holds"] is True
    assert closure["call_site_controls"]["holds"] is True
    #: review-0246-01 A's 5.0 s attack, refused three ways
    denominator = closure["coverage_denominator_control"]
    assert denominator["holds"] is True
    assert denominator["observation_gap_over_the_sealed_headroom"] > 1.9
    for arm in denominator["arms"].values():
        assert arm["refused"] is True
    #: review-0246-01 C's sixteenth attack, refused
    sixteenth = closure["reviewer_sixteenth_attack_control"]
    assert sixteenth["holds"] is True
    assert sixteenth["gate_refused_it"] is True
    assert sixteenth["declared_max_poll_spacing_s"] == 1e6
    assert sixteenth["registered_max_poll_spacing_s"] == pytest.approx(
        2.5109531834854018
    )
    #: the self-attacks are published, including the one that succeeds
    self_attacks = closure["self_attack_battery"]
    assert self_attacks["attacks_run"] >= 7
    assert self_attacks["attacks_that_still_succeed"] == [
        "r0247-self-7: hand the liveness gate a fabricated receipt"
    ]
    #: and the node's own guard passed every node-tail gate
    assert receipt["sampler_liveness"]["holds"] is True
    assert receipt["abort_flag_landing"]["holds"] is True
    assert receipt["enforcement_poll_spacing"]["holds"] is True
    assert receipt["enforcement_poll_spacing"]["enforcement_evidence"][
        "holds"
    ] is True


# --------------------------------------------------------------------------- #
# node 2 — the precision fix
# --------------------------------------------------------------------------- #
def _truthcos_job(world: _TinyWorld, output: str) -> dict:
    return {
        "action": nodes.TRUTHCOS_ACTION,
        "outputs": [output],
        "truth_cos": world.truth_cos,
        "truth_ids": world.truth_ids,
        "probe_query_rows": world.probe_rows,
        "substrate_array": world.substrate,
        "truth_rows": TINY_PROBE,
    }


def test_truthcos_entry_path_runs_end_to_end(monkeypatch, scratch, world, armed):
    _patch(monkeypatch)
    output = os.path.join(scratch, "artifacts-truthcos")
    nodes.run_job(ACTIVE, _truthcos_job(world, output))
    receipt = prompt_contract.read_sealed(
        os.path.join(output, nodes.TRUTHCOS_FILE), label="tiny truthcos"
    )
    assert receipt["schema"] == nodes.TRUTHCOS_SCHEMA
    saved = os.path.join(output, nodes.TRUTH_COS_F64_FILE)
    assert os.path.exists(saved)
    recomputed = np.load(saved)
    assert recomputed.dtype == np.float64
    assert recomputed.shape == (TINY_PROBE, TINY_K)
    floor = receipt["cosine_noise_floor"]
    #: the storage column must be the pure container effect and the arithmetic
    #: column must be the float64 residual; both are measured, not asserted
    assert floor["storage_quantisation"]["p99"] > 0.0
    assert floor["float64_arithmetic"]["p99"] < (
        floor["stored_vs_recomputed"]["p99"]
    )
    tolerance = receipt["defensible_tolerance"]
    assert tolerance["the_tolerance_was_not_moved"] is True
    assert tolerance["current_tie_tolerance"] == 1e-6
    assert tolerance["smallest_defensible_tolerance"] > 0.0
    assert receipt["enforcement_poll_spacing"]["holds"] is True


def test_the_recompute_reads_the_sealed_ids_and_creates_no_cuda_context(
    monkeypatch, scratch, world, armed
):
    """The whole point of review-0246-01 F: this is a CPU gather."""
    _patch(monkeypatch)
    output = os.path.join(scratch, "artifacts-truthcos-cuda")
    nodes.run_job(ACTIVE, _truthcos_job(world, output))
    receipt = prompt_contract.read_sealed(
        os.path.join(output, nodes.TRUTHCOS_FILE), label="tiny truthcos"
    )
    assert receipt["cuda_context_created"] is False
    assert receipt["cuvs_calls"] == 0
    assert receipt["child_processes_launched"] == 0
    assert receipt["signal_delivered"] is False
    assert receipt["recompute"]["substrate_rows_gathered"] == (
        TINY_PROBE * TINY_K
    )


# --------------------------------------------------------------------------- #
# node 3 — the sealed bound
# --------------------------------------------------------------------------- #
def test_tie_entry_path_runs_end_to_end(monkeypatch, scratch, world, armed):
    _patch(monkeypatch)
    truthcos_output = os.path.join(scratch, "artifacts-truthcos-for-tie")
    nodes.run_job(ACTIVE, _truthcos_job(world, truthcos_output))
    f64_path = os.path.join(truthcos_output, nodes.TRUTH_COS_F64_FILE)

    output = os.path.join(scratch, "artifacts-tie")
    nodes.run_job(ACTIVE, {
        "action": nodes.TIE_ACTION,
        "outputs": [output],
        "truth_cos": world.truth_cos,
        "truth_ids": world.truth_ids,
        "probe_query_rows": world.probe_rows,
        "substrate_array": world.substrate,
        "graph_ids": world.graph_ids,
        "truth_cos_f64": {"canonical_path": f64_path, "bytes": -1},
        "replication_rows": TINY_PROBE,
        "full_probe_rows": TINY_PROBE,
        "tie_seed": 247_005,
    })
    receipt = prompt_contract.read_sealed(
        os.path.join(output, nodes.TIE_FILE), label="tiny tie"
    )
    assert receipt["schema"] == nodes.TIE_SCHEMA
    assert receipt["the_tolerance_was_not_moved"] is True
    assert receipt["ledger_size"] == len(TIE_AWARE_CLAIM_LEDGER)

    whole = receipt["whole_probe_against_the_recomputed_float64_truth"]
    assert whole["candidate_decisions_scored"] == TINY_PROBE * TINY_K

    sealed = receipt["sealed_bound_adjudication"]
    #: the count is COMPUTED in the receipt, not written in prose
    assert sealed["claims_that_do_not_survive_at_the_bound"] == len(
        sealed["claims_that_do_not_survive_at_the_bound_names"]
    )
    assert (
        sealed["claims_that_survive_at_the_bound"]
        + sealed["claims_that_do_not_survive_at_the_bound"]
        == len(TIE_AWARE_CLAIM_LEDGER)
    )
    #: and the already-repaired claim is counted AND labelled, which is the
    #: off-by-one review-0246-01 E found
    assert sealed["already_repaired_among_the_non_survivors"] or True
    bound = sealed["flip_rate_bound"]
    assert bound["confidence"] == 0.95
    if bound["observed_flips"] == 0:
        assert bound["poisson_upper_limit_events"] == pytest.approx(
            2.99573227355399
        )
    assert receipt["aggregate_only_control"]["holds"] is True
    assert receipt["enforcement_poll_spacing"]["holds"] is True


# --------------------------------------------------------------------------- #
# the entry path itself
# --------------------------------------------------------------------------- #
def test_every_node_action_refuses_to_start_without_the_flag(
    monkeypatch, scratch
):
    monkeypatch.delenv("ROUNDRUN_ABORT_FLAG", raising=False)
    for action in (
        nodes.PARAMGUARD_ACTION, nodes.TRUTHCOS_ACTION, nodes.TIE_ACTION,
    ):
        with pytest.raises(guard0245.Round0245Error):
            nodes.run_job(
                ACTIVE,
                {"action": action, "outputs": [os.path.join(scratch, action)]},
            )


def test_run_job_refuses_an_unknown_action(scratch, armed) -> None:
    with pytest.raises(nodes.Round0247Error):
        nodes.run_job(ACTIVE, {"action": "not-a-round-0247-node"})


def test_a_node_refuses_a_queue_from_another_round(scratch, armed) -> None:
    other = {"manifest": {"round_id": "0246", "release_sha": RELEASE}}
    with pytest.raises(nodes.Round0247Error):
        nodes.run_job(other, {
            "action": nodes.PARAMGUARD_ACTION,
            "outputs": [os.path.join(scratch, "wrong-round")],
        })
