from __future__ import annotations

import math

import pytest

from basemap.round0169_prompted_diverse import ROWS as DIVERSE_ROWS
from basemap.round0202_h4096_nested_dose_ladder import (
    FULL_GRAPH_EDGES,
    FULL_SUCCESSFUL_UPDATES,
)
from basemap.round0210_prompted_diverse_low_dose import (
    CAPABILITY,
    HIDDEN_DIMENSION,
    POSITIVE_ROWS_PER_UPDATE,
    ROUND_ID,
    ROWS,
    Round0210Error,
    SEED,
    SUPERSEDED_FIXED_UPDATES,
    TARGET_POSITIVE_DRAWS_PER_EDGE,
    achieved_draws_per_edge,
    low_dose_train_config,
    successful_updates_for_edges,
)
from experiments import round0210_nodes as nodes


GRAPH = {"canonical_path": "/data/edges.npz", "bytes": 1, "sha256": "a" * 64}
MANIFEST = {"canonical_path": "/data/graph.json", "bytes": 1, "sha256": "b" * 64}
ESTIMATED_EDGES = 946_013_908


def _config(edges: int = ESTIMATED_EDGES) -> dict:
    config, _digest = low_dose_train_config(
        graph_signature=GRAPH,
        graph_manifest_signature=MANIFEST,
        graph_edges=edges,
        retained_rows=ROWS,
    )
    return config


def test_dose_rule_is_the_accepted_r0184_rule() -> None:
    assert successful_updates_for_edges(FULL_GRAPH_EDGES) == FULL_SUCCESSFUL_UPDATES
    assert FULL_SUCCESSFUL_UPDATES == 1_000_000
    assert math.isclose(
        achieved_draws_per_edge(
            updates=FULL_SUCCESSFUL_UPDATES, edge_count=FULL_GRAPH_EDGES
        ),
        TARGET_POSITIVE_DRAWS_PER_EDGE,
        rel_tol=1e-15,
    )


def test_memo_estimate_reproduces_the_memo_update_count() -> None:
    assert successful_updates_for_edges(ESTIMATED_EDGES) == 1_568_621


def test_achieved_dose_tracks_the_target_across_plausible_edge_counts() -> None:
    for edges in (
        int(ESTIMATED_EDGES * 0.85),
        ESTIMATED_EDGES,
        int(ESTIMATED_EDGES * 1.2),
    ):
        updates = successful_updates_for_edges(edges)
        achieved = achieved_draws_per_edge(updates=updates, edge_count=edges)
        assert math.isclose(
            achieved, TARGET_POSITIVE_DRAWS_PER_EDGE, rel_tol=1e-6, abs_tol=0.0
        )


def test_config_replaces_only_the_horizon() -> None:
    config = _config()
    assert config["optimizer"]["successful_positive_lr_updates"] == 1_568_621
    assert config["optimizer"]["successful_positive_lr_updates"] != SUPERSEDED_FIXED_UPDATES
    assert config["model"]["hidden_dimension"] == HIDDEN_DIMENSION == 2048
    assert config["paired_invariant"]["rows"] == ROWS == DIVERSE_ROWS == 12_474_331
    assert config["paired_invariant"]["seed"] == SEED == 42
    assert (
        config["execution"]["expected_pipeline_stamp"]["compact_retained_rows"] == ROWS
    )
    assert config["dose_registration"]["source_round"] == "0184"
    assert config["dose_registration"]["active_graph_edges"] == ESTIMATED_EDGES
    assert config["dose_registration"]["positive_rows_per_update"] == POSITIVE_ROWS_PER_UPDATE


def test_config_is_deterministic() -> None:
    first, first_digest = low_dose_train_config(
        graph_signature=GRAPH,
        graph_manifest_signature=MANIFEST,
        graph_edges=ESTIMATED_EDGES,
        retained_rows=ROWS,
    )
    second, second_digest = low_dose_train_config(
        graph_signature=GRAPH,
        graph_manifest_signature=MANIFEST,
        graph_edges=ESTIMATED_EDGES,
        retained_rows=ROWS,
    )
    assert first == second and first_digest == second_digest


def test_wrong_population_fails_closed() -> None:
    with pytest.raises(Round0210Error):
        low_dose_train_config(
            graph_signature=GRAPH,
            graph_manifest_signature=MANIFEST,
            graph_edges=ESTIMATED_EDGES,
            retained_rows=7_952_419,
        )


def test_nonpositive_edges_fail_closed() -> None:
    for edges in (0, -1):
        with pytest.raises(Round0210Error):
            successful_updates_for_edges(edges)


def test_node_rejects_another_action_or_queue() -> None:
    with pytest.raises(Round0210Error):
        nodes.run_job(
            {"manifest": {"round_id": ROUND_ID}},
            {"action": "build_graph_and_reference"},
        )
    with pytest.raises(Round0210Error):
        nodes.run_train(
            {"manifest": {"round_id": "0169"}},
            {"action": "train_prompted_diverse_u12_low_dose"},
        )


def test_capability_and_round_identity() -> None:
    assert ROUND_ID == "0210"
    assert CAPABILITY == "jina-prompted-diverse-u12-map-seed42-low-dose-v1"
