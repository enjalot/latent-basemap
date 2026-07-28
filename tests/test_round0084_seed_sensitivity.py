from __future__ import annotations

import copy
import inspect
import json

from basemap.round0075_training import SUCCESSFUL_UPDATES
from basemap.round0084_program import (
    CONFIG_SCHEMA,
    SEED,
    seed43_config_from_seed42,
)
from experiments import prepare_round0084_queue, round0084_nodes


SUBSTRATE = (
    "/data/latent-basemap/runs/round-0071/queue/artifacts/"
    "balanced-90m-int8-substrate/balanced-90m-substrate-v1.json"
)
GRAPH = (
    "/data/latent-basemap/runs/round-0073/queue/artifacts/"
    "native-graph-balanced-90m/canonical-graph-v1.json"
)
BASELINE_RECEIPT = (
    "/data/latent-basemap/runs/round-0075/queue/artifacts/"
    "train-balanced-90m/train-receipt.json"
)


def test_seed43_config_changes_only_seed_and_receipt_metadata() -> None:
    with open(BASELINE_RECEIPT, encoding="utf-8") as handle:
        baseline = json.load(handle)["production_config"]
    with open(SUBSTRATE, encoding="utf-8") as handle:
        substrate = json.load(handle)
    with open(GRAPH, encoding="utf-8") as handle:
        graph = json.load(handle)
    config, digest = seed43_config_from_seed42(
        baseline,
        graph_manifest=graph,
        graph_manifest_path=GRAPH,
        graph_manifest_sha256=(
            "d8ec25e2887926d11af6da7b6c6c4bf07d1fa9adedfc9f84d2c1c5baf07fcef5"
        ),
        substrate_manifest=substrate,
        substrate_manifest_path=SUBSTRATE,
        substrate_manifest_sha256=(
            "032e3c6396e26e0f2ff0db81f764330e4e84175d337d164ab63ae9c7ddeec6d2"
        ),
    )
    assert len(digest) == 64
    assert config["schema"] == CONFIG_SCHEMA
    assert config["optimizer"]["seed"] == SEED == 43
    assert (
        config["optimizer"]["successful_positive_lr_updates"]
        == SUCCESSFUL_UPDATES
        == 1_493_293
    )
    assert (
        config["execution"]["expected_pipeline_stamp"]
        == baseline["execution"]["expected_pipeline_stamp"]
    )
    assert config["graph"] == baseline["graph"]
    assert config["row_universe"] == baseline["row_universe"]
    assert config["model"] == baseline["model"]

    normalized = copy.deepcopy(config)
    normalized["schema"] = baseline["schema"]
    normalized["phrase"] = baseline["phrase"]
    normalized["optimizer"]["seed"] = 42
    del normalized["execution"]["seed_sensitivity_treatment"]
    del normalized["decision_thresholds"]["one_seed_contrast_only"]
    del normalized["decision_thresholds"][
        "does_not_establish_seed_noise_band"
    ]
    normalized["execution"]["scale_transition"] = copy.deepcopy(
        baseline["execution"]["scale_transition"]
    )
    assert normalized == baseline


def test_comparison_contract_is_descriptive_not_a_noise_estimator() -> None:
    source = inspect.getsource(round0084_nodes.run_comparison)
    assert '"one_paired_seed_contrast": True' in source
    assert '"estimates_variance": False' in source
    assert '"establishes_error_bar": False' in source
    assert "twice" not in source.lower()
    assert "absolute_delta" in source
    assert "signed_delta_seed43_minus_seed42" in source


def test_queue_is_one_train_plus_two_matched_evaluation_lanes() -> None:
    source = inspect.getsource(prepare_round0084_queue.prepare_round0084)
    assert source.count('action="train"') == 1
    assert source.count('action="transform"') == 2
    assert source.count('action="panel"') == 2
    assert source.count('action="comparison"') == 1
    assert "gpu_hours_cap=5.5" in source
    assert '"minilm-balanced-90m-seed43-sensitivity-v1"' in source
    assert '"one_contrast_is_not_variance_or_error_bar": True' in source


def test_historical_dependencies_match_actual_consumers() -> None:
    assert list(prepare_round0084_queue.REVIEWS) == [
        "0071",
        "0073",
        "0075",
        "0076",
    ]
    assert prepare_round0084_queue.CAPABILITIES["0071"].endswith(
        "90m-int8-input-v1"
    )
    assert prepare_round0084_queue.CAPABILITIES["0073"].endswith(
        "90m-gpu-native-graph-v1"
    )
    assert prepare_round0084_queue.CAPABILITIES["0075"].endswith(
        "trained-model-seed42-v1"
    )
    assert prepare_round0084_queue.CAPABILITIES["0076"].endswith(
        "30m-45m-60m-90m-scale-geometry-v1"
    )


def test_handler_restores_shared_trainer_globals() -> None:
    source = inspect.getsource(round0084_nodes.run_train)
    assert "trainer.ROUND_ID = ROUND_ID" in source
    assert "trainer.SEED = SEED" in source
    assert "finally:" in source
    assert 'trainer.ROUND_ID = previous["round_id"]' in source
    assert 'trainer.SEED = previous["seed"]' in source
