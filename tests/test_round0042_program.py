from __future__ import annotations

import pytest

from basemap.round0042_program import (
    Round0042ProgramError,
    train_config_from_graph,
)


def _manifest() -> dict:
    return {
        "schema": "minilm-canonical-source-major-k15-v1",
        "round_id": "0041",
        "row_count": 30_000_000,
        "input_k": 15,
        "inputs": {
            "eligibility": {
                "sha256": (
                    "834089fcbd9a722cec4f05be6382ed8430d27280e7e23ca085"
                    "5785e3f48ea5e2"
                )
            }
        },
        "summary": {
            "input_edge_count": 450_000_000,
            "eligibility_excluded_source_count": 218_242,
            "eligibility_retained_row_count": 29_781_758,
            "retained_positive_source_count": 29_781_619,
            "zero_degree_retained_source_count": 139,
            "valid_canonical_edge_count": 444_198_115,
            "duplicate_destinations_mapped": 2_524_873,
        },
    }


def test_train_config_is_the_registered_r0021_isolation() -> None:
    config, digest = train_config_from_graph(
        _manifest(),
        graph_manifest_path="/data/canonical.json",
        graph_manifest_sha256="a" * 64,
    )
    assert len(digest) == 64
    assert config["optimizer"]["seed"] == 42
    assert (
        config["optimizer"]["successful_positive_lr_updates"] == 500_000
    )
    assert config["model"]["hidden_dimension"] == 2048
    assert config["execution"]["required_pipeline"] == (
        "device_fp16_canonical"
    )
    assert config["execution"]["expected_pipeline_stamp"][
        "positive_source_count"
    ] == 29_781_619
    assert config["execution"]["duplicate_control"][
        "zero_degree_retained_sources_excluded"
    ] == 139
    assert config["execution"]["performance_windows"] == 200
    assert config["execution"][
        "performance_abort_latency_at_floor_seconds_max"
    ] == 63.0
    assert "duplicate_multiplicity" not in config["execution"]


def test_train_config_fails_on_graph_geometry_drift() -> None:
    manifest = _manifest()
    manifest["summary"]["valid_canonical_edge_count"] -= 1
    with pytest.raises(Round0042ProgramError, match="geometry"):
        train_config_from_graph(
            manifest,
            graph_manifest_path="/data/canonical.json",
            graph_manifest_sha256="a" * 64,
        )


def test_round0042_modules_do_not_mutate_cuda_visibility(monkeypatch) -> None:
    import importlib
    import os

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "sentinel")
    importlib.import_module("experiments.round0042_nodes")
    importlib.import_module("experiments.prepare_round0042_queue")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "sentinel"
