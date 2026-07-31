from __future__ import annotations

import inspect

from basemap.round0052_program import (
    SUCCESSFUL_UPDATES,
    train_config_from_capabilities,
)
from experiments import prepare_round0063_queue, round0052_nodes
from tests.test_round0052_balanced60m_train import _capabilities


def test_r0062_graph_is_an_exact_training_capability() -> None:
    graph, substrate = _capabilities()
    graph["round_id"] = "0062"
    config, digest = train_config_from_capabilities(
        graph,
        graph_manifest_path="/data/round0062-graph.json",
        graph_manifest_sha256="c" * 64,
        substrate_manifest=substrate,
        substrate_manifest_path="/data/substrate.json",
        substrate_manifest_sha256="d" * 64,
    )
    assert len(digest) == 64
    assert SUCCESSFUL_UPDATES == 997_248
    assert config["graph"]["path"] == "/data/round0062-graph.json"
    assert config["execution"]["required_pipeline"] == (
        "host_int8_canonical"
    )
    assert config["execution"]["expected_pipeline_stamp"][
        "positive_source_count"
    ] == 59_399_288


def test_r0063_queue_is_canonical_and_has_no_extra_canary() -> None:
    source = inspect.getsource(
        prepare_round0063_queue.prepare_round0063
    )
    assert prepare_round0063_queue.ROUND_ID == "0063"
    assert prepare_round0063_queue.RELEASE_ROOT == (
        "/home/enjalot/code/latent-basemap-run"
    )
    assert "gpu_hours_cap=3.5" in source
    assert '"standalone_canary": False' in source
    assert '"action": "canary"' not in source
    assert '"round0063-train-receipt-v1"' in source
    assert '"total": 10_500.0' in source


def test_shared_trainer_accepts_only_registered_rounds() -> None:
    source = inspect.getsource(round0052_nodes.run_job)
    assert '{ROUND_ID, "0063"}' in source
    assert '"round_id": round_id' in inspect.getsource(
        round0052_nodes.run_train
    )
