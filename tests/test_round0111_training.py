from __future__ import annotations

import hashlib
import json

import pytest

from basemap.artifact_identity import expected_input_signature
from basemap.round0107_training import Round0107Error, train_config
from experiments import round0111_nodes
from experiments.prepare_round0111_queue import (
    GRAPH_MANIFEST,
    R0106_REVIEW,
    R0106_REVIEW_SHA256,
    SEED,
    _require_successful_r0109_terminal,
)


def _graph() -> tuple[dict, dict]:
    manifest = {
        "directed_edge_count": 1_000,
        "compact_mapping": {"sha256": "a" * 64},
        "outputs": {
            "sources": {"sha256": "b" * 64},
            "targets": {"sha256": "c" * 64},
            "weights": {"sha256": "d" * 64},
        },
    }
    signature = {
        "kind": "file",
        "canonical_path": "/data/synthetic-graph.json",
        "bytes": 1,
        "sha256": "e" * 64,
    }
    return manifest, signature


def test_seed44_config_changes_only_registered_identity_and_seed() -> None:
    graph, signature = _graph()
    control, control_sha = train_config(
        graph_manifest=graph,
        graph_signature=signature,
    )
    treatment, treatment_sha = train_config(
        graph_manifest=graph,
        graph_signature=signature,
        seed=44,
        schema="round0111-diverse-jina-train-config-v1",
    )
    assert control_sha != treatment_sha
    assert treatment["schema"] == "round0111-diverse-jina-train-config-v1"
    assert treatment["optimizer"]["seed"] == 44
    normalized = dict(treatment)
    normalized["schema"] = control["schema"]
    normalized["optimizer"] = dict(treatment["optimizer"])
    normalized["optimizer"]["seed"] = control["optimizer"]["seed"]
    assert normalized == control


def test_round0111_wrapper_binds_seed_and_receipt_schemas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict = {}

    def fake(active, job, **contract):
        observed.update(contract)
        return {"round_id": contract["round_id"]}

    monkeypatch.setattr(round0111_nodes, "run_train_contract", fake)
    value = round0111_nodes.run_job(
        {"manifest": {"round_id": "0111"}},
        {"action": "train_diverse_jina_seed44"},
    )
    assert value == {"round_id": "0111"}
    assert observed == {
        "round_id": "0111",
        "seed": 44,
        "train_config_schema": "round0111-diverse-jina-train-config-v1",
        "production_config_schema": "round0111-production-config-v1",
        "train_receipt_schema": (
            "round0111-diverse-jina-train-receipt-v1"
        ),
        "output_label": "R0111 seed-44 diverse-Jina train output",
    }
    with pytest.raises(Round0107Error):
        round0111_nodes.run_job(
            {"manifest": {"round_id": "0109"}},
            {"action": "train_diverse_jina_seed44"},
        )


def test_r0109_terminal_ordering_requires_clean_success(tmp_path) -> None:
    path = tmp_path / "runner-terminal.json"
    terminal = {
        "schema": "slim-runner-terminal-v3",
        "round_id": "0109",
        "verdict": "succeeded",
        "completed_jobs": ["a"],
        "required_jobs": ["a"],
        "release_checkout_unchanged": True,
        "queue_manifest_unchanged": True,
    }
    path.write_text(json.dumps(terminal))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert _require_successful_r0109_terminal(
        str(path), expected_sha256=digest
    ) == expected_input_signature(path)
    terminal["verdict"] = "failed"
    path.write_text(json.dumps(terminal))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="terminal training"):
        _require_successful_r0109_terminal(
            str(path), expected_sha256=digest
        )


def test_round0111_binds_reviewed_graph_and_seed42_review() -> None:
    assert SEED == 44
    assert GRAPH_MANIFEST.endswith(
        "/round-0106/queue-attempt-3/artifacts/"
        "canonical-fuzzy-graph/graph-manifest.json"
    )
    assert expected_input_signature(R0106_REVIEW)["sha256"] == (
        R0106_REVIEW_SHA256
    )
