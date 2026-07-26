from __future__ import annotations

import inspect

from basemap.round0055_program import (
    SUCCESSFUL_UPDATES,
    train_config_from_capabilities,
)
from experiments import prepare_round0061_queue, round0055_nodes
from tests.test_round0055_matched30m_train import _capabilities


def test_r0060_graph_is_an_exact_training_capability() -> None:
    graph, substrate = _capabilities()
    graph["round_id"] = "0060"
    config, digest = train_config_from_capabilities(
        graph,
        graph_manifest_path="/data/round0060-graph.json",
        graph_manifest_sha256="c" * 64,
        substrate_manifest=substrate,
        substrate_manifest_path="/data/substrate.json",
        substrate_manifest_sha256="d" * 64,
    )
    assert len(digest) == 64
    assert SUCCESSFUL_UPDATES == 500_003
    assert config["graph"]["path"] == "/data/round0060-graph.json"
    assert config["execution"]["required_pipeline"] == (
        "host_int8_canonical"
    )
    assert config["execution"]["expected_pipeline_stamp"][
        "positive_source_count"
    ] == 29_781_754


def test_r0061_queue_is_canonical_and_has_no_extra_canary() -> None:
    source = inspect.getsource(
        prepare_round0061_queue.prepare_round0061
    )
    assert prepare_round0061_queue.ROUND_ID == "0061"
    assert prepare_round0061_queue.RELEASE_ROOT == (
        "/home/enjalot/code/latent-basemap-run"
    )
    assert "gpu_hours_cap=2.0" in source
    assert '"standalone_canary": False' in source
    assert '"action": "canary"' not in source
    assert '"round0061-train-receipt-v1"' in source


def test_shared_trainer_accepts_only_registered_rounds() -> None:
    source = inspect.getsource(round0055_nodes.run_job)
    assert '{ROUND_ID, "0061"}' in source
    assert '"round_id": round_id' in inspect.getsource(
        round0055_nodes.run_train
    )
