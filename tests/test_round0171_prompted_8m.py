"""Contract tests for the exact sharded-fp32 prompted 8M replacement."""
from __future__ import annotations

import numpy as np
import pytest

from basemap.round0171_prompted_8m import (
    GRAPH_EXECUTION,
    GRAPH_VECTOR_STORAGE,
    ROUND_ID,
    scale_train_config,
)
from experiments import round0166_nodes, round0171_nodes


def test_merge_ann_topk_is_global_and_deterministic_at_ties() -> None:
    sims, ids = round0166_nodes._merge_ann_topk(
        np.asarray([[0.9, 0.8, 0.7], [0.5, 0.4, 0.3]], dtype=np.float32),
        np.asarray([[9, 8, 7], [50, 40, 30]], dtype=np.int64),
        np.asarray([[0.95, 0.8, 0.6], [0.5, 0.45, 0.2]], dtype=np.float32),
        np.asarray([[10, 2, 6], [5, 45, 20]], dtype=np.int64),
        k=3,
    )
    np.testing.assert_allclose(sims, [[0.95, 0.9, 0.8], [0.5, 0.5, 0.45]])
    np.testing.assert_array_equal(ids, [[10, 9, 2], [5, 50, 45]])


def test_merge_ann_topk_rejects_invalid_candidates() -> None:
    with pytest.raises(RuntimeError, match="invalid candidates"):
        round0166_nodes._merge_ann_topk(
            np.asarray([[1.0, np.nan]], dtype=np.float32),
            np.asarray([[0, 1]], dtype=np.int64),
            np.asarray([[0.9, 0.8]], dtype=np.float32),
            np.asarray([[2, 3]], dtype=np.int64),
            k=2,
        )


def test_r0171_config_registers_capacity_execution_only() -> None:
    signature = {"canonical_path": "/future/graph", "sha256": "a" * 64}
    config, digest = scale_train_config(
        graph_signature=signature,
        graph_manifest_signature=signature,
        graph_edges=123,
        retained_rows=7_952_419,
    )
    assert len(digest) == 64
    assert config["schema"] == "round0171-prompted-8m-train-config-v1"
    assert config["execution"]["graph_execution"] == GRAPH_EXECUTION
    assert config["execution"]["graph_vector_storage"] == GRAPH_VECTOR_STORAGE
    assert config["paired_invariant"]["graph_vector_storage"] == GRAPH_VECTOR_STORAGE
    assert config["optimizer"]["successful_positive_lr_updates"] == 500_000
    assert config["graph"]["k"] == 50


def test_r0171_dispatch_binds_fp32_sharded_faiss(monkeypatch) -> None:
    observed = {}
    binding_names = (
        "ROUND_ID",
        "CAPABILITY",
        "Round0166Error",
        "GRAPH_SCHEMA",
        "QUERY_SCHEMA",
        "TRAIN_SCHEMA",
        "EVALUATION_SCHEMA",
        "PRODUCTION_CONFIG_SCHEMA",
        "GRAPH_INDEX_DESCRIPTION",
        "GRAPH_REFERENCE_ROW_ORDER",
        "GRAPH_REFERENCE_ANCHOR_NAMESPACE",
        "GRAPH_SHARD_ROWS",
        "scale_decision",
        "scale_train_config",
        "_faiss_gpu_options",
    )
    before = {name: getattr(round0166_nodes, name) for name in binding_names}
    monkeypatch.setattr(
        round0166_nodes,
        "run_job",
        lambda active, job: observed.update({
            "round_id": round0166_nodes.ROUND_ID,
            "index": round0166_nodes.GRAPH_INDEX_DESCRIPTION,
            "shard_rows": round0166_nodes.GRAPH_SHARD_ROWS,
            "config": round0166_nodes.scale_train_config,
        }),
    )
    try:
        round0171_nodes.run_job(
            {"manifest": {"round_id": ROUND_ID}},
            {"action": "build_graph_and_reference"},
        )
        assert observed["round_id"] == ROUND_ID
        assert "fp32 vector storage" in observed["index"]
        assert "exact global top-k" in observed["index"]
        assert observed["shard_rows"] == 4_000_000
        assert observed["config"] is scale_train_config
    finally:
        for name, value in before.items():
            setattr(round0166_nodes, name, value)
