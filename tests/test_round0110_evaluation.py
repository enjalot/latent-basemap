from __future__ import annotations

import hashlib
import inspect
import json

import pytest

from basemap.round0108_evaluation import (
    MAP_KEY as SEED42_MAP_KEY,
    load_reviewed_model,
)
from experiments import round0108_nodes, round0110_nodes
from experiments.prepare_round0110_queue import (
    R0108_SELECTION,
    _frontmatter,
    _require_clean_terminal,
)


def test_reviewed_model_loader_defaults_preserve_r0108_seed42_contract() -> None:
    parameters = inspect.signature(load_reviewed_model).parameters
    assert parameters["expected_train_round_id"].default == "0107"
    assert (
        parameters["expected_train_receipt_schema"].default
        == "round0107-diverse-jina-train-receipt-v1"
    )
    assert (
        parameters["expected_production_config_schema"].default
        == "round0107-production-config-v1"
    )
    assert parameters["expected_seed"].default == 42


def test_seed43_loader_binds_exact_r0109_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict = {}

    def fake(**kwargs):
        observed.update(kwargs)
        return {"model": "seed43"}

    monkeypatch.setattr(round0110_nodes, "load_reviewed_model", fake)
    value = round0110_nodes._seed43_model(
        train_output="/train",
        graph_manifest_path="/graph",
        graph_manifest_sha256="a" * 64,
    )
    assert value == {"model": "seed43"}
    assert observed == {
        "train_output": "/train",
        "graph_manifest_path": "/graph",
        "graph_manifest_sha256": "a" * 64,
        "expected_train_round_id": "0109",
        "expected_train_receipt_schema": (
            "round0109-diverse-jina-train-receipt-v1"
        ),
        "expected_production_config_schema": (
            "round0109-production-config-v1"
        ),
        "expected_seed": 43,
    }


def test_seed43_wrapper_changes_only_evaluation_identity(
) -> None:
    original = round0108_nodes.R0108_EVALUATION_CONTRACT
    selected = round0110_nodes._seed43_job({"action": "score"})
    contract = selected["evaluation_node_contract"]
    assert contract["round_id"] == "0110"
    assert contract["map_key"] == "r0109-diverse-jina-25m-seed43"
    assert contract["core_schema"].startswith("round0110-")
    assert contract["ood_schema"].startswith("round0110-")
    assert contract["train_round_id"] == "0109"
    assert contract["seed"] == 43
    assert round0108_nodes.R0108_EVALUATION_CONTRACT is original


def test_run_job_rejects_cross_round_dispatch() -> None:
    with pytest.raises(Exception, match="exact round"):
        round0110_nodes.run_job(
            {"manifest": {"round_id": "0108"}},
            {"action": "transform_seed43"},
        )


def test_round0110_uses_exact_r0108_selection_path() -> None:
    assert R0108_SELECTION.endswith(
        "/round-0108/queue-attempt-3/inputs/registered-selections.npz"
    )
    assert SEED42_MAP_KEY == "r0107-diverse-jina-25m-seed42"


def test_frontmatter_parser_reads_quoted_values(tmp_path) -> None:
    path = tmp_path / "review.md"
    path.write_text(
        '---\nround_id: "0109"\nstatus: accepted\n---\n# Review\n'
    )
    assert _frontmatter(str(path)) == {
        "round_id": "0109",
        "status": "accepted",
    }


def test_clean_terminal_requires_complete_unchanged_success(tmp_path) -> None:
    path = tmp_path / "runner-terminal.json"
    terminal = {
        "schema": "slim-runner-terminal-v3",
        "round_id": "0108",
        "verdict": "succeeded",
        "completed_jobs": ["a"],
        "required_jobs": ["a"],
        "release_checkout_unchanged": True,
        "queue_manifest_unchanged": True,
    }
    path.write_text(json.dumps(terminal))
    signature = _require_clean_terminal(str(path), round_id="0108")
    assert signature["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    terminal["completed_jobs"] = []
    path.write_text(json.dumps(terminal))
    with pytest.raises(RuntimeError, match="clean success"):
        _require_clean_terminal(str(path), round_id="0108")
