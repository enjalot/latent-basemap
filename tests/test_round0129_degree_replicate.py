"""Focused contract tests for the conditional R0129 replicate."""
from __future__ import annotations

import json
import os

import pytest

from basemap.artifact_identity import expected_input_signature
from basemap.round0113_prompt_contrast import read_sealed, seal
from basemap.round0124_degree_bridge import (
    ATTEMPT1_EVIDENCE,
    ATTEMPT1_RELEASE_SHA,
    OUTCOME_INCONCLUSIVE,
    OUTCOME_MATERIAL,
    OUTCOME_NOT_MATERIAL,
)
from basemap.round0129_degree_replicate import (
    GRAPH_PROVENANCE_SCHEMA,
    TRAINING_SEED,
    Round0129Error,
    config_equivalence,
    train_config,
    verify_graph_provenance,
)
from experiments import prepare_round0129_queue, round0113_nodes


R0115_GRAPH = (
    "/data/latent-basemap/runs/round-0115/queue-attempt-2/"
    "artifacts/raw/graph/graph-manifest.json"
)
R0117_CONFIG = (
    "/data/latent-basemap/runs/round-0117/queue/artifacts/raw/train/"
    "production-config.json"
)
R0124_GRAPH = ATTEMPT1_EVIDENCE["graph_manifest"]["canonical_path"]


def test_seed43_config_is_graph_only_against_exact_r0117_control():
    treatment_graph = read_sealed(R0124_GRAPH, label="R0124 k15 graph")
    treatment, _sha = train_config(
        graph_signature=treatment_graph["graph"],
        graph_manifest_signature=expected_input_signature(R0124_GRAPH),
        graph_edges=int(treatment_graph["directed_edge_count"]),
        retained_rows=int(treatment_graph["retained_rows"]),
    )
    with open(R0117_CONFIG, encoding="utf-8") as handle:
        control_receipt = json.load(handle)
    proof = config_equivalence(
        treatment=treatment, control=control_receipt["config"]
    )
    assert treatment["optimizer"]["seed"] == TRAINING_SEED
    assert treatment["optimizer"]["positive_rng_seed"] == TRAINING_SEED
    assert treatment["graph"]["k"] == 15
    assert control_receipt["config"]["graph"]["k"] == 50
    assert proof["sampling_law_equal_after_graph_fields"] is True
    assert proof["identical_realized_edge_draws_claimed"] is False


def test_prompt_panel_registers_r0129_seed43_without_fallback():
    active = {"manifest": {"round_id": "0129"}}
    assert round0113_nodes._training_seed(
        active, {"training_seed": TRAINING_SEED}
    ) == TRAINING_SEED
    with pytest.raises(Exception, match="training seed changed"):
        round0113_nodes._training_seed(active, {"training_seed": 42})


def _graph_provenance_value() -> dict:
    return {
        "schema": GRAPH_PROVENANCE_SCHEMA,
        "source_round_id": "0124",
        "source_attempt": 1,
        "source_release_sha": ATTEMPT1_RELEASE_SHA,
        "source_terminal_verdict": "failed-after-successful-graph-node",
        "graph_rebuilt": False,
        "evidence": {
            key: dict(ATTEMPT1_EVIDENCE[key])
            for key in (
                "queue_manifest",
                "runner_terminal",
                "graph_done_marker",
                "graph_manifest",
                "graph",
                "topology_probe",
            )
        },
    }


def test_graph_provenance_authenticates_successful_node_inside_failed_attempt(
    monkeypatch,
):
    value = _graph_provenance_value()
    by_path = {
        signature["canonical_path"]: signature
        for signature in value["evidence"].values()
    }
    monkeypatch.setattr(
        "basemap.round0129_degree_replicate.expected_input_signature",
        lambda path: dict(by_path[path]),
    )
    queue_sha = value["evidence"]["queue_manifest"]["sha256"]
    records = {
        value["evidence"]["queue_manifest"]["canonical_path"]: {
            "schema": "round0124-fineweb-2m-degree-bridge-queue-v1",
            "round_id": "0124",
            "release_sha": ATTEMPT1_RELEASE_SHA,
        },
        value["evidence"]["runner_terminal"]["canonical_path"]: {
            "schema": "slim-runner-terminal-v3",
            "round_id": "0124",
            "verdict": "failed",
            "queue_manifest_sha256": queue_sha,
            "queue_manifest_unchanged": True,
            "release_checkout_unchanged": True,
            "boundary_problems": [],
            "completed_jobs": ["build_k15_graph"],
        },
        value["evidence"]["graph_done_marker"]["canonical_path"]: {
            "schema": "slim-runner-done-v2",
            "node": "build_k15_graph",
            "returncode": 0,
            "queue_manifest_sha256": queue_sha,
            "release_sha": ATTEMPT1_RELEASE_SHA,
        },
    }
    monkeypatch.setattr(
        "basemap.round0129_degree_replicate._read_json",
        lambda path, label: records[path],
    )
    monkeypatch.setattr(
        "basemap.round0129_degree_replicate.read_sealed",
        lambda path, label: {
            "round_id": "0124",
            "release_sha": ATTEMPT1_RELEASE_SHA,
            "graph": value["evidence"]["graph"],
            "topology_probe": value["evidence"]["topology_probe"],
        },
    )
    assert verify_graph_provenance(value) == value
    changed = _graph_provenance_value()
    changed["evidence"]["graph"]["sha256"] = "0" * 64
    with pytest.raises(Round0129Error, match="provenance contract changed"):
        verify_graph_provenance(changed)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_trigger_bundle(tmp_path, *, outcome: str):
    release = "a" * 40
    decision_dir = tmp_path / "artifacts" / "degree-bridge-decision"
    decision_path = decision_dir / "decision.json"
    decision = seal(
        {
            "schema": "round0124-fineweb-2m-degree-bridge-decision-v1",
            "round_id": "0124",
            "release_sha": release,
            "retry_provenance": {"source": "test"},
            "registered_selector": {"outcome": outcome},
            "capabilities_produced": [
                "jina-fineweb-2m-native-k15-degree-bridge-v1"
            ],
        }
    )
    _write_json(decision_path, decision)
    queue_path = tmp_path / "queue.json"
    jobs = [
        {"id": "train_k15_treatment", "outputs": [str(tmp_path / "train")]},
        {
            "id": "decide_degree_bridge",
            "outputs": [str(decision_dir)],
        },
    ]
    queue = {
        "schema": "round0124-fineweb-2m-degree-bridge-retry-queue-v1",
        "round_id": "0124",
        "release_sha": release,
        "retry_provenance": {"source": "test"},
        "jobs": jobs,
    }
    _write_json(queue_path, queue)
    queue_sig = expected_input_signature(str(queue_path))
    terminal_path = tmp_path / "runner-terminal.json"
    terminal = {
        "schema": "slim-runner-terminal-v3",
        "round_id": "0124",
        "verdict": "succeeded",
        "required_jobs": [job["id"] for job in jobs],
        "completed_jobs": [job["id"] for job in jobs],
        "nodes": [
            {"node": job["id"], "validation_problems": []} for job in jobs
        ],
        "queue_manifest_sha256": queue_sig["sha256"],
        "queue_manifest_sha256_at_finish": queue_sig["sha256"],
        "queue_manifest_unchanged": True,
        "release_checkout_unchanged": True,
        "gpu_wall_accounting_complete": True,
        "boundary_problems": [],
        "release_checkout": {"head": release},
        "release_checkout_at_finish": {"head": release},
    }
    _write_json(terminal_path, terminal)
    terminal_sig = expected_input_signature(str(terminal_path))
    decision_sig = expected_input_signature(str(decision_path))
    result_path = tmp_path / "result-0124-2026-07-31.md"
    result_path.write_text(
        "---\n"
        'round_id: "0124"\n'
        "status: complete\n"
        f'release_commit: "{release}"\n'
        f'queue_manifest: "gsv:{queue_path}"\n'
        f'queue_manifest_sha256: "{queue_sig["sha256"]}"\n'
        "capabilities_produced: "
        '["jina-fineweb-2m-native-k15-degree-bridge-v1"]\n'
        "---\n"
        f"queue {queue_sig['sha256']} terminal {terminal_sig['sha256']} "
        f"decision {decision_sig['sha256']}\n"
    )
    result_sig = expected_input_signature(str(result_path))
    review_path = tmp_path / "review-0124-2026-07-31.md"
    review_path.write_text(
        "---\n"
        'round_id: "0124"\n'
        "status: accepted\n"
        f'result: "{result_path.name}"\n'
        f'result_sha256: "{result_sig["sha256"]}"\n'
        f'verified_release_commit: "{release}"\n'
        "releases: "
        '["capability:jina-fineweb-2m-native-k15-degree-bridge-v1"]\n'
        "---\n"
        "Prose mentions k15-materially-degrades-native-density and "
        "k15-does-not-materially-degrade-native-density, but prose is not "
        "the selector.\n"
    )
    return {
        "review": str(review_path),
        "review_sha": expected_input_signature(str(review_path))["sha256"],
        "queue": str(queue_path),
        "terminal": str(terminal_path),
        "decision": str(decision_path),
    }


def test_structured_inconclusive_trigger_ignores_positive_negative_prose(
    tmp_path,
):
    bundle = _write_trigger_bundle(tmp_path, outcome=OUTCOME_INCONCLUSIVE)
    evidence = prepare_round0129_queue._require_inconclusive_r0124_review(
        bundle["review"],
        expected_sha256=bundle["review_sha"],
        queue_path=bundle["queue"],
        terminal_path=bundle["terminal"],
        decision_path=bundle["decision"],
    )
    assert set(evidence) == {"review", "result", "queue", "terminal", "decision"}


@pytest.mark.parametrize("outcome", [OUTCOME_MATERIAL, OUTCOME_NOT_MATERIAL])
def test_positive_or_negative_sealed_outcome_cannot_release_inconclusive_branch(
    tmp_path, outcome
):
    bundle = _write_trigger_bundle(tmp_path, outcome=outcome)
    with pytest.raises(RuntimeError, match="not the inconclusive branch"):
        prepare_round0129_queue._require_inconclusive_r0124_review(
            bundle["review"],
            expected_sha256=bundle["review_sha"],
            queue_path=bundle["queue"],
            terminal_path=bundle["terminal"],
            decision_path=bundle["decision"],
        )
