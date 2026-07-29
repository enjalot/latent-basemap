from __future__ import annotations

import importlib
import inspect

import pytest

from basemap.round0034_pipeline import GRAPH_SCHEMA
from basemap.round0065_substrates import SUBSETS, SUBSTRATE_SCHEMA
from basemap.round0068_training import (
    PIPELINE_SCHEMA,
    SAMPLER_CLASS,
    successful_updates_for_tier,
    train_config_from_capabilities,
)


def _capability_fixture(tier: str):
    spec = SUBSETS[tier]
    rows = spec["row_count"]
    excluded = spec["eligibility_summary"]["excluded_row_count"]
    retained = spec["eligibility_summary"]["retained_row_count"]
    eligibility = {
        "canonical_path": f"/data/{tier}-eligibility.npz",
        "bytes": 123,
        "sha256": "e" * 64,
    }
    substrate = {
        "schema": SUBSTRATE_SCHEMA,
        "round_id": "0065",
        "tier": tier,
        "row_count": rows,
        "dimension": 384,
        "global_150m_intervals": [
            list(value) for value in spec["intervals"]
        ],
        "outputs": {
            "int8": {
                "canonical_path": f"/data/{tier}.i8",
                "bytes": rows * 384,
                "sha256": "a" * 64,
            },
            "scales": {
                "canonical_path": f"/data/{tier}.f16",
                "bytes": rows * 2,
                "sha256": "b" * 64,
            },
            "eligibility": eligibility,
        },
    }
    graph = {
        "schema": GRAPH_SCHEMA,
        "round_id": "0067",
        "tier": tier,
        "row_count": rows,
        "input_k": 15,
        "inputs": {"eligibility": eligibility},
        "summary": {
            "eligibility_excluded_source_count": excluded,
            "eligibility_retained_row_count": retained,
            "retained_positive_source_count": retained,
            "zero_degree_retained_source_count": 0,
            "valid_canonical_edge_count": retained * 15,
            "degree_histogram": {"0": excluded, "15": retained},
        },
    }
    return substrate, graph


@pytest.mark.parametrize(
    ("tier", "updates"),
    [("45m", 748_757), ("120m", 1_982_221)],
)
def test_selected_tier_updates_are_coverage_aligned(
    tier: str,
    updates: int,
) -> None:
    assert successful_updates_for_tier(tier) == updates
    substrate, graph = _capability_fixture(tier)
    config, config_sha = train_config_from_capabilities(
        tier=tier,
        graph_manifest=graph,
        graph_manifest_path=f"/data/{tier}-graph.json",
        graph_manifest_sha256="c" * 64,
        substrate_manifest=substrate,
        substrate_manifest_path=f"/data/{tier}-substrate.json",
        substrate_manifest_sha256="d" * 64,
    )
    assert len(config_sha) == 64
    assert config["optimizer"]["successful_positive_lr_updates"] == updates
    stamp = config["execution"]["expected_pipeline_stamp"]
    assert stamp["schema"] == PIPELINE_SCHEMA
    assert stamp["sampler_class"] == SAMPLER_CLASS
    assert stamp["positive_source_count"] == (
        SUBSETS[tier]["eligibility_summary"]["retained_row_count"]
    )
    assert stamp["valid_canonical_edge_count"] == (
        stamp["positive_source_count"] * 15
    )


def test_round0068_is_one_dynamic_training_job() -> None:
    from experiments import prepare_round0068_queue

    source = inspect.getsource(
        prepare_round0068_queue.prepare_round0068
    )
    assert source.count('"action": "train_selected_tier"') == 1
    assert 'cap = 3.0 if tier == "45m" else 7.5' in source
    assert 'p90 = 10_800.0 if tier == "45m" else 27_000.0' in source
    assert '"standalone_canary": False' in source
    assert '"training_wall_only": True' in source
    assert "tier = decision" in source


def test_round0068_receipts_exact_training_accounting() -> None:
    from experiments import round0068_nodes

    source = inspect.getsource(round0068_nodes.run_train)
    for field in (
        "lr_horizon",
        "positive_lr_optimizer_steps",
        "optimizer_steps_succeeded",
        "amp_overflow_skips",
        "nonfinite_loss_skips",
        "nonfinite_gradient_skips",
        "pipeline_runtime",
        "host_prefetch_consumer_batches",
    ):
        assert field in source
    loader = inspect.getsource(round0068_nodes._load_pipeline)
    assert (
        'graph["manifest"].get("inputs", {}).get("scale_decision")'
        in loader
    )
    assert '!= decision["signature"]' in loader


def test_round0068_flattens_actual_runtime_counters() -> None:
    from experiments import round0068_nodes

    runtime = {
        key: index + 1
        for index, key in enumerate(
            round0068_nodes._DYNAMIC_PIPELINE_COUNTERS
        )
    }
    accounting = {
        f"pipeline_{key}": 0
        for key in round0068_nodes._DYNAMIC_PIPELINE_COUNTERS
    }
    round0068_nodes._synchronize_flattened_runtime_counters(
        accounting,
        runtime,
    )
    assert accounting == {
        f"pipeline_{key}": value
        for key, value in runtime.items()
    }


def test_round0068_modules_do_not_mutate_cuda_visibility(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "sentinel")
    importlib.import_module("basemap.round0068_training")
    importlib.import_module("experiments.round0068_nodes")
    importlib.import_module("experiments.prepare_round0068_queue")
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "sentinel"
