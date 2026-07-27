from __future__ import annotations

import inspect

import json

import numpy as np

from experiments import prepare_round0070_queue, round0070_nodes


LEGACY_COORDINATES = (
    "/data/latent-basemap/runs/round-0019/queue/artifacts/coordinates/"
    "actual-transform.json"
)


def test_legacy_coordinates_bind_actual_reviewed_model() -> None:
    with open(LEGACY_COORDINATES, encoding="utf-8") as handle:
        receipt = json.load(handle)
    assert (
        receipt["actual_transform"]["model_signature"]["sha256"]
        == "2f5eb27582e26735491b4bed9417cf27992bb213ef942e433a5bcba97d481a32"
    )


def test_density_summary_reports_exact_registered_corpus_strata() -> None:
    high = np.linspace(1.0, 10.0, 10_000)
    low = high * 2.0
    labels = np.repeat(
        np.asarray(["fineweb", "redpajama", "pile"], dtype="<U10"),
        [3_334, 3_333, 3_333],
    )
    summary = round0070_nodes._density_summary(high, low, labels)
    assert summary["correlation"] == 1.0
    assert summary["anchors"] == 10_000
    assert {
        key: value["anchors"]
        for key, value in summary["by_corpus"].items()
    } == {"fineweb": 3_334, "redpajama": 3_333, "pile": 3_333}


def test_factorial_classification_covers_registered_explanations() -> None:
    universe = round0070_nodes.classify_factorial({
        "legacy_original": 0.80,
        "legacy_representative": 0.10,
        "modern_original": 0.79,
        "modern_representative": 0.11,
    })
    assert universe["classification"] == "data-universe-dominant"

    model = round0070_nodes.classify_factorial({
        "legacy_original": 0.80,
        "legacy_representative": 0.79,
        "modern_original": 0.10,
        "modern_representative": 0.11,
    })
    assert model["classification"] == "model-training-dominant"

    interaction = round0070_nodes.classify_factorial({
        "legacy_original": 0.80,
        "legacy_representative": 0.10,
        "modern_original": 0.10,
        "modern_representative": 0.10,
    })
    assert interaction["classification"] == "model-by-universe-interaction"


def test_queue_is_one_cross_transform_one_reference_one_factorial() -> None:
    source = inspect.getsource(prepare_round0070_queue.prepare_round0070)
    assert source.count('action="modern_transform"') == 1
    assert source.count('action="original_reference"') == 1
    assert source.count('action="density_factorial"') == 1
    assert 'action="train"' not in source
    assert "gpu_hours_cap=1.0" in source
    assert '"authorizes_larger_training_rung": False' in source


def test_train_bundle_validator_uses_registered_round0064_label() -> None:
    queue_source = inspect.getsource(prepare_round0070_queue.prepare_round0070)
    node_source = inspect.getsource(round0070_nodes._balanced_bundle)
    assert 'label="r0061-30m"' in queue_source
    assert 'label="r0061-30m"' in node_source
    assert "r0061-balanced-30m" not in queue_source
    assert "r0061-balanced-30m" not in node_source


def test_factorial_uses_identical_global_anchors_and_exact_radii() -> None:
    source = inspect.getsource(round0070_nodes.run_density_factorial)
    assert "selector.compact_to_global(compact_anchors)" in source
    assert source.count("_self_knn(") == 1
    assert "want_dist=True" in source
    assert "modern int8 density does not replay" in source


def test_no_training_handler_contract() -> None:
    source = inspect.getsource(round0070_nodes)
    assert '"training_performed": False' in source
    assert "optimizer" not in inspect.getsource(
        round0070_nodes.run_modern_transform
    )
    assert "ParametricUMAP.load" not in source
