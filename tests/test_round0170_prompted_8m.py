"""Contract tests for the fp16-IVF prompted 8M replacement rung."""
from __future__ import annotations

from basemap.round0170_prompted_8m import (
    GRAPH_VECTOR_STORAGE,
    ROUND_ID,
    scale_train_config,
)
from experiments import round0166_nodes, round0170_nodes


def test_r0170_config_registers_only_capacity_representation() -> None:
    signature = {"canonical_path": "/future/graph", "sha256": "a" * 64}
    config, digest = scale_train_config(
        graph_signature=signature,
        graph_manifest_signature=signature,
        graph_edges=123,
        retained_rows=7_952_419,
    )
    assert len(digest) == 64
    assert config["schema"] == "round0170-prompted-8m-train-config-v1"
    assert config["execution"]["graph_vector_storage"] == GRAPH_VECTOR_STORAGE
    assert config["paired_invariant"]["graph_vector_storage"] == GRAPH_VECTOR_STORAGE
    assert config["optimizer"]["successful_positive_lr_updates"] == 500_000
    assert config["graph"]["k"] == 50


def test_r0170_dispatch_binds_fp16_faiss(monkeypatch) -> None:
    observed = {}
    monkeypatch.setattr(
        round0166_nodes,
        "run_job",
        lambda active, job: observed.update({
            "round_id": round0166_nodes.ROUND_ID,
            "index": round0166_nodes.GRAPH_INDEX_DESCRIPTION,
            "config": round0166_nodes.scale_train_config,
        }),
    )
    round0170_nodes.run_job(
        {"manifest": {"round_id": ROUND_ID}}, {"action": "build_graph_and_reference"}
    )
    assert observed["round_id"] == ROUND_ID
    assert observed["index"].endswith("fp16 vector storage")
    assert observed["config"] is scale_train_config
