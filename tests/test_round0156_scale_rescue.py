"""CUDA-hidden tests for the R0156 retagged scale treatment."""
from __future__ import annotations

from basemap.round0156_scale_rescue import (
    CAPABILITY,
    GRAPH_DEGREE,
    GRAPH_K,
    PARENT_CAPABILITY,
    PARENT_ROUND_ID,
    OUTCOME_PASS,
    PIPELINE,
    PIPELINE_SCHEMA,
    POSITIVE_DESTINATION_POLICY,
    RETAINED_ROWS,
    ROUND_ID,
    TRAIN_CONFIG_SCHEMA,
)
from basemap.round0107_training import SAMPLER_CLASS, train_config
from basemap.round0108_evaluation import CompactInt8DequantizedArray
from experiments import prepare_round0156_queue, round0156_nodes
from experiments import round0106_nodes


def test_r0156_uses_scale_native_k15_not_small_map_k50() -> None:
    assert GRAPH_K == 15
    assert GRAPH_DEGREE == "variable-symmetric-fuzzy-k15-topology"


def test_r0156_retags_parent_and_release_contract() -> None:
    original_round = round0156_nodes.base.ROUND_ID
    with round0156_nodes._configured():
        assert round0156_nodes.base.ROUND_ID == ROUND_ID
        assert round0156_nodes.base.CAPABILITY == CAPABILITY
        assert round0156_nodes.base.PARENT_ROUND_ID == PARENT_ROUND_ID == "0155"
        assert round0156_nodes.base.PARENT_CAPABILITY == PARENT_CAPABILITY
        assert round0156_nodes.base.PARENT_CENSUS_FIELD == "r0155_census"
        assert round0156_nodes.base.GRAPH_DEGREE == GRAPH_DEGREE
        assert round0156_nodes.contract.FULL_25M_TEST_ON_PASS is False
        quality = {"passed": True}
        decision = round0156_nodes.contract.build_decision(
            validity_checks={"all": True}, quality=quality
        )
        assert decision["outcome"] == OUTCOME_PASS
        assert decision["atlas_rescue_candidate_released"] is True
        assert decision["full_25m_prefix_drop_test_released"] is False
    assert round0156_nodes.base.ROUND_ID == original_round


def test_r0156_queue_wrapper_uses_new_paths_and_handler() -> None:
    original_round = prepare_round0156_queue.base.ROUND_ID
    with prepare_round0156_queue._configured():
        assert prepare_round0156_queue.base.ROUND_ID == ROUND_ID
        assert prepare_round0156_queue.base.ROUND_ROOT.endswith("round-0156")
        assert prepare_round0156_queue.base.R0151_CENSUS.endswith("/census.json")
        assert "/round-0155/" in prepare_round0156_queue.base.R0151_CENSUS
        assert prepare_round0156_queue.base.HANDLER_MODULE == "experiments.round0156_nodes"
        assert prepare_round0156_queue.base._issued_round == prepare_round0156_queue._issued_round
    assert prepare_round0156_queue.base.ROUND_ID == original_round


def test_r0156_graph_helper_registers_exact_universe_and_contract(tmp_path) -> None:
    """Catch the real graph-shard admission path before spending GPU time."""
    contract = round0106_nodes.GraphNodeContract(
        round_id=ROUND_ID,
        k=GRAPH_K,
        n_neighbors=GRAPH_K + 1,
        shard_schema="r0156-test-shard-v1",
        part_schema="r0156-test-part-v1",
        graph_schema="r0156-test-graph-v1",
    )
    artifact = tmp_path / "missing-shard.npz"
    receipt = tmp_path / "missing-shard.npz.receipt.json"
    assert round0106_nodes._validate_shard(
        str(artifact),
        str(receipt),
        part="groups-a",
        shard=0,
        compact_start=0,
        compact_stop=1,
        contract_sha256="0" * 64,
        contract=contract,
        universe_rows=12_485_206,
    ) is None


def test_r0156_real_train_config_variant_is_registered() -> None:
    signature = {
        "canonical_path": "/synthetic/r0156-graph.json",
        "bytes": 1,
        "sha256": "0" * 64,
        "kind": "file",
    }
    graph = {
        "directed_edge_count": 295_373_928,
        "schema": "round0156-historical-prefix-fuzzy-graph-v1",
        "round_id": ROUND_ID,
        "compact_mapping": signature,
        "outputs": [signature, signature, signature],
    }
    config, digest = train_config(
        graph_manifest=graph,
        graph_signature=signature,
        schema=TRAIN_CONFIG_SCHEMA,
        update_rule="ceil(actual-R0156-directed-fuzzy-edges/409)",
        positive_destination_policy=POSITIVE_DESTINATION_POLICY,
        graph_degree=GRAPH_DEGREE,
        compact_retained_rows=RETAINED_ROWS,
        pipeline=PIPELINE,
        pipeline_schema=PIPELINE_SCHEMA,
        sampler_class=SAMPLER_CLASS,
    )
    assert config["input"]["rows"] == RETAINED_ROWS
    assert config["execution"]["expected_pipeline_stamp"]["pipeline"] == PIPELINE
    assert config["optimizer"]["successful_positive_lr_updates"] == 722_186
    assert len(digest) == 64


def test_r0156_compact_mapping_is_registered_for_posttrain_evaluation(
    monkeypatch,
) -> None:
    """Catch the real transform admission that follows the expensive train."""
    import numpy as np
    from basemap import round0108_evaluation

    class _MemmapStub:
        def __init__(self, *_args, **kwargs):
            self.shape = kwargs["shape"]

    monkeypatch.setattr(round0108_evaluation, "validate_substrate_manifest", lambda **_: {
        "payloads": {
            "int8": {"canonical_path": "/synthetic/embeddings.i8"},
            "scales": {"canonical_path": "/synthetic/scales.f16"},
        }
    })
    monkeypatch.setattr(round0108_evaluation.np, "memmap", _MemmapStub)
    class _MappingStub:
        ndim = 1
        dtype = np.dtype("int64")

        def __len__(self):
            return RETAINED_ROWS

        def __getitem__(self, key):
            if key == 0:
                return 0
            if key == -1:
                return RETAINED_ROWS - 1
            if isinstance(key, slice):
                return np.array([1, 2] if key.start == 1 else [0, 1], dtype=np.int64)
            raise AssertionError(key)

    mapping = _MappingStub()
    source = CompactInt8DequantizedArray(mapping)
    assert source.shape == (RETAINED_ROWS, 768)
