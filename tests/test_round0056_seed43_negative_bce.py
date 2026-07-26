from __future__ import annotations

import importlib
import inspect

import pytest

from basemap.round0056_program import (
    Round0056ProgramError,
    selected_arm,
    train_config_from_graph,
)


def _graph() -> dict:
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


def _comparison(selection: str) -> dict:
    return {
        "schema": "round0051-negative-bce-calibration-v1",
        "round_id": "0051",
        "selection": selection,
        "interpretation": {
            "external_ood_adoption_gate_run": False,
        },
    }


def test_selection_is_exactly_the_r0051_candidate() -> None:
    assert selected_arm(_comparison("negative-0p50-candidate")) == (
        "negative_0p50"
    )
    assert selected_arm(_comparison("negative-0p25-candidate")) == (
        "negative_0p25"
    )
    with pytest.raises(Round0056ProgramError):
        selected_arm(_comparison("retain-baseline-1p00"))
    with pytest.raises(Round0056ProgramError):
        selected_arm(_comparison("invalid-isolation"))


def test_config_changes_seed43_edge_arm_only_by_selected_loss() -> None:
    config, digest = train_config_from_graph(
        _graph(),
        graph_manifest_path="/data/graph.json",
        graph_manifest_sha256="a" * 64,
        arm="negative_0p25",
    )
    assert len(digest) == 64
    assert config["optimizer"]["seed"] == 43
    assert config["optimizer"]["negative_bce_multiplier"] == 0.25
    assert config["optimizer"]["positive_bce_multiplier"] == 1.0
    stamp = config["execution"]["expected_pipeline_stamp"]
    assert stamp["pipeline"] == "device_fp16_canonical_edge_uniform"
    assert stamp["positive_source_sampling"] == (
        "degree-proportional-over-positive-sources"
    )
    assert config["execution"]["expected_loss_stamp"] == {
        "loss_class": "NormalizedClassWeightedBCELoss",
        "positive_multiplier": 1.0,
        "negative_multiplier": 0.25,
        "reduction": "weighted-sum-over-weight-sum",
        "positive_threshold": 0.5,
    }


def test_round0056_is_one_train_with_direct_live_profiler() -> None:
    from experiments import prepare_round0056_queue as queue_prep

    source = inspect.getsource(queue_prep.prepare_round0056)
    assert "gpu_hours_cap=2.25" in source
    assert '"total": 5_900.0' in source
    assert source.count('"action": "train"') == 1
    assert '"action": "canary"' not in source
    assert queue_prep.RELEASE_ROOT == (
        "/home/enjalot/code/latent-basemap-run"
    )


def test_round0056_modules_do_not_mutate_cuda_visibility(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "sentinel")
    importlib.import_module("basemap.round0056_program")
    importlib.import_module("experiments.round0056_nodes")
    importlib.import_module("experiments.prepare_round0056_queue")
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "sentinel"
