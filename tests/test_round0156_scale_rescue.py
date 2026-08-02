"""CUDA-hidden tests for the R0156 retagged scale treatment."""
from __future__ import annotations

from basemap.round0156_scale_rescue import (
    CAPABILITY,
    GRAPH_DEGREE,
    GRAPH_K,
    PARENT_CAPABILITY,
    PARENT_ROUND_ID,
    OUTCOME_PASS,
    ROUND_ID,
)
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
