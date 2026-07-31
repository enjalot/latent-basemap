from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from basemap.round0108_evaluation import seal
from experiments import round0119_nodes
from experiments.prepare_round0119_queue import (
    REQUIRED_REVIEWS,
    _clean_terminal,
)


def _write_json(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return expected_input_signature(str(path))


def test_authenticate_model_binds_train_config_and_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = {
        "schema": "test-config-v1",
        "arm": "raw",
        "optimizer": {"seed": 43},
        "model": {
            "architecture": "residual_bottleneck",
            "input_dimension": 768,
            "hidden_dimension": 2048,
            "hidden_layers": 3,
            "output_dimension": 2,
            "use_batchnorm": False,
            "use_dropout": False,
            "low_dim_kernel": "legacy_lp",
            "a": 1.0,
            "b": 1.0,
        },
    }
    config_sha = sha256_bytes(canonical_json(config))
    config_path = tmp_path / "production-config.json"
    config_signature = _write_json(
        config_path,
        {
            "schema": "test-config-receipt-v1",
            "round_id": "0117",
            "config": config,
            "config_sha256": config_sha,
        },
    )
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"model")
    model_signature = expected_input_signature(str(model_path))
    train_path = tmp_path / "train-receipt.json"
    train_signature = _write_json(
        train_path,
        seal({
            "schema": "test-train-v1",
            "round_id": "0117",
            "arm": "raw",
            "training_seed": 43,
            "production_config": config_signature,
            "production_config_sha256": config_sha,
            "model": model_signature,
        }),
    )

    class FakeModel:
        architecture = "residual_bottleneck"
        input_dim = 768
        hidden_dim = 2048
        n_layers = 3
        n_components = 2
        use_batchnorm = False
        use_dropout = False
        low_dim_kernel = "legacy_lp"
        a = 1.0
        b = 1.0

    from basemap.pumap import parametric_umap

    monkeypatch.setattr(
        parametric_umap.ParametricUMAP,
        "load",
        lambda path, device: FakeModel(),
    )
    spec = {
        "key": "current_2m_seed43",
        "group": "current_2m",
        "round_id": "0117",
        "seed": 43,
        "arm": "raw",
        "train_schema": "test-train-v1",
        "config_receipt_schema": "test-config-receipt-v1",
        "config_receipt_round_id": "0117",
        "config_schema": "test-config-v1",
        "training_population": "population",
        "training_graph": "graph",
        "training_dose": "dose",
        "train_receipt": train_signature,
        "production_config": config_signature,
        "model": model_signature,
    }
    bundle = round0119_nodes._authenticate_model(spec)
    assert bundle["train"] == train_signature
    assert bundle["production_config"] == config_signature
    assert bundle["model_signature"] == model_signature

    changed = json.loads(config_path.read_text(encoding="utf-8"))
    changed["config"]["optimizer"]["seed"] = 42
    config_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(round0119_nodes.Round0119Error, match="bytes changed"):
        round0119_nodes._authenticate_model(spec)


def test_clean_terminal_requires_r0117_completion(tmp_path: Path) -> None:
    path = tmp_path / "runner-terminal.json"
    terminal = {
        "schema": "slim-runner-terminal-v3",
        "round_id": "0117",
        "verdict": "succeeded",
        "completed_jobs": ["raw", "document", "decision"],
        "required_jobs": ["raw", "document", "decision"],
        "release_checkout_unchanged": True,
        "queue_manifest_unchanged": True,
    }
    path.write_text(json.dumps(terminal), encoding="utf-8")
    assert _clean_terminal(str(path), round_id="0117")["bytes"] > 0
    terminal["completed_jobs"] = ["raw"]
    path.write_text(json.dumps(terminal), encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean success"):
        _clean_terminal(str(path), round_id="0117")


def test_registered_cell_and_group_order_is_frozen() -> None:
    assert round0119_nodes.CELL_ORDER == (
        "historical_2m_seed42",
        "historical_2m_seed43",
        "current_2m_seed42",
        "current_2m_seed43",
        "current_25m_seed42",
        "current_25m_seed43",
    )
    assert set(round0119_nodes.GROUPS) == {
        "historical_2m",
        "current_2m",
        "current_25m",
    }
    assert REQUIRED_REVIEWS == ("0037", "0038", "0110", "0115", "0117")
    assert "0111" not in REQUIRED_REVIEWS
    assert "0118" not in REQUIRED_REVIEWS
