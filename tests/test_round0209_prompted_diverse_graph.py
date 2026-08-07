from __future__ import annotations

import pytest

from basemap import round0169_prompted_diverse as diverse
from basemap.round0209_prompted_diverse_graph import (
    CAPABILITY,
    ESTIMATED_DIRECTED_EDGES,
    EXPECTED_GRAPH_SHARDS,
    GRAPH_SCHEMA,
    GRAPH_SHARD_ROWS,
    ROUND_ID,
    ROWS,
    Round0209Error,
    plausible_directed_edges,
)
from experiments import round0169_nodes as base
from experiments import round0209_nodes as nodes


def test_graph_law_is_the_accepted_diverse_law_unchanged() -> None:
    assert ROWS == diverse.ROWS == 12_474_331
    assert (
        diverse.GRAPH_K,
        diverse.GRAPH_NLIST,
        diverse.GRAPH_NPROBE,
        diverse.GRAPH_NPROBE_GRID,
        diverse.GRAPH_TRAIN_SEED,
        diverse.GRAPH_QUALITY_SEED,
        diverse.GRAPH_MEAN_RECALL_FLOOR,
        diverse.GRAPH_P10_RECALL_FLOOR,
    ) == (50, 8_192, 64, (16, 32, 64, 128, 256), 113, 114, 0.90, 0.80)


def test_shard_plan_covers_the_population_exactly_once() -> None:
    assert EXPECTED_GRAPH_SHARDS[0][0] == 0
    assert EXPECTED_GRAPH_SHARDS[-1][1] == ROWS
    for (_start, stop), (next_start, _next_stop) in zip(
        EXPECTED_GRAPH_SHARDS, EXPECTED_GRAPH_SHARDS[1:]
    ):
        assert stop == next_start
    assert all(stop - start <= GRAPH_SHARD_ROWS for start, stop in EXPECTED_GRAPH_SHARDS)
    assert sum(stop - start for start, stop in EXPECTED_GRAPH_SHARDS) == ROWS


def test_edge_plausibility_band() -> None:
    assert plausible_directed_edges(ESTIMATED_DIRECTED_EDGES)
    assert plausible_directed_edges(int(ESTIMATED_DIRECTED_EDGES * 0.9))
    assert not plausible_directed_edges(int(ESTIMATED_DIRECTED_EDGES * 0.5))
    assert not plausible_directed_edges(int(ESTIMATED_DIRECTED_EDGES * 2))
    assert not plausible_directed_edges(0)


def test_node_rejects_any_other_action() -> None:
    for action in ("train_prompted_diverse_u12", "evaluate_prompted_diverse_u12", ""):
        with pytest.raises(Round0209Error):
            nodes.run_job({"manifest": {"round_id": ROUND_ID}}, {"action": action})


def test_configure_rebinds_only_identity_and_schema() -> None:
    original = (base.ROUND_ID, base.GRAPH_SCHEMA, base.CAPABILITY)
    try:
        nodes._configure()
        assert base.ROUND_ID == ROUND_ID == "0209"
        assert base.GRAPH_SCHEMA == GRAPH_SCHEMA
        assert base.CAPABILITY == CAPABILITY
        assert base.GRAPH_SHARD_ROWS == GRAPH_SHARD_ROWS
        assert base.EXPECTED_GRAPH_SHARDS == EXPECTED_GRAPH_SHARDS
    finally:
        base.ROUND_ID, base.GRAPH_SCHEMA, base.CAPABILITY = original


def test_round0209_error_is_a_diverse_error() -> None:
    assert issubclass(Round0209Error, diverse.Round0169Error)
