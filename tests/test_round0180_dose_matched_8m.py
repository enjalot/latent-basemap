"""Contract and end-to-end CPU smoke tests for R0180."""
from __future__ import annotations

import pytest

from basemap.round0180_dose_matched_8m import (
    ACHIEVED_POSITIVE_DRAWS_PER_EDGE,
    BASELINE_GRAPH_EDGES,
    BASELINE_SUCCESSFUL_UPDATES,
    RETAINED_ROWS,
    ROUND_ID,
    SUCCESSFUL_UPDATES,
    TARGET_GRAPH_EDGES,
    TARGET_POSITIVE_DRAWS_PER_EDGE,
    scale_train_config,
)
from experiments import round0166_nodes, round0180_nodes
from tests.test_round0166_cpu_smoke import (
    _run_train_seal_reload_panel_cpu_smoke,
)


def test_exact_dose_horizon_is_first_whole_update_at_or_above_target() -> None:
    assert SUCCESSFUL_UPDATES == 2_026_478
    assert (
        (SUCCESSFUL_UPDATES - 1) * BASELINE_GRAPH_EDGES
        < BASELINE_SUCCESSFUL_UPDATES * TARGET_GRAPH_EDGES
        <= SUCCESSFUL_UPDATES * BASELINE_GRAPH_EDGES
    )
    assert ACHIEVED_POSITIVE_DRAWS_PER_EDGE >= TARGET_POSITIVE_DRAWS_PER_EDGE


def test_config_changes_only_the_registered_dose_horizon() -> None:
    signature = {
        "kind": "file",
        "canonical_path": "/future/r0171-graph",
        "bytes": 1,
        "sha256": "a" * 64,
    }
    config, digest = scale_train_config(
        graph_signature=signature,
        graph_manifest_signature=signature,
        graph_edges=TARGET_GRAPH_EDGES,
        retained_rows=RETAINED_ROWS,
    )
    assert len(digest) == 64
    assert config["optimizer"]["successful_positive_lr_updates"] == SUCCESSFUL_UPDATES
    assert config["paired_invariant"]["successful_positive_lr_updates"] == SUCCESSFUL_UPDATES
    assert config["dose_registration"]["target_graph_round"] == "0171"
    assert config["dose_registration"]["rounding"].startswith("ceiling")
    assert config["graph"]["k"] == 50
    assert config["optimizer"]["seed"] == 42


def test_dispatch_binds_reused_graph_and_forbids_rebuild(monkeypatch) -> None:
    observed = {}
    names = (
        "ROUND_ID",
        "CAPABILITY",
        "SUCCESSFUL_UPDATES",
        "HOST_RSS_LIMIT_GIB",
        "Round0166Error",
        "GRAPH_SCHEMA",
        "QUERY_SCHEMA",
        "TRAIN_SCHEMA",
        "EVALUATION_SCHEMA",
        "PRODUCTION_CONFIG_SCHEMA",
        "GRAPH_INDEX_DESCRIPTION",
        "GRAPH_REFERENCE_ROW_ORDER",
        "GRAPH_REFERENCE_ANCHOR_NAMESPACE",
        "GRAPH_SOURCE_ROUND_ID",
        "GRAPH_BUILT_IN_ROUND",
        "scale_decision",
        "scale_train_config",
    )
    before = {name: getattr(round0166_nodes, name) for name in names}
    monkeypatch.setattr(
        round0166_nodes,
        "run_job",
        lambda active, job: observed.update({
            "round_id": round0166_nodes.ROUND_ID,
            "updates": round0166_nodes.SUCCESSFUL_UPDATES,
            "graph_round": round0166_nodes.GRAPH_SOURCE_ROUND_ID,
            "graph_built": round0166_nodes.GRAPH_BUILT_IN_ROUND,
        }),
    )
    try:
        round0180_nodes.run_job(
            {"manifest": {"round_id": ROUND_ID}},
            {"action": "train_prompted_8m"},
        )
        assert observed == {
            "round_id": "0180",
            "updates": 2_026_478,
            "graph_round": "0171",
            "graph_built": False,
        }
        with pytest.raises(RuntimeError, match="does not authorize"):
            round0180_nodes.run_job(
                {"manifest": {"round_id": ROUND_ID}},
                {"action": "build_graph_and_reference"},
            )
    finally:
        for name, value in before.items():
            setattr(round0166_nodes, name, value)


def test_r0180_train_seal_reload_panel_cpu_smoke(monkeypatch, tmp_path) -> None:
    names = (
        "ROUND_ID",
        "CAPABILITY",
        "SUCCESSFUL_UPDATES",
        "HOST_RSS_LIMIT_GIB",
        "Round0166Error",
        "GRAPH_SCHEMA",
        "QUERY_SCHEMA",
        "TRAIN_SCHEMA",
        "EVALUATION_SCHEMA",
        "PRODUCTION_CONFIG_SCHEMA",
        "GRAPH_INDEX_DESCRIPTION",
        "GRAPH_REFERENCE_ROW_ORDER",
        "GRAPH_REFERENCE_ANCHOR_NAMESPACE",
        "GRAPH_SOURCE_ROUND_ID",
        "GRAPH_BUILT_IN_ROUND",
        "scale_decision",
        "scale_train_config",
    )
    before = {name: getattr(round0166_nodes, name) for name in names}
    try:
        round0180_nodes._configure()
        with monkeypatch.context() as smoke_patch:
            _run_train_seal_reload_panel_cpu_smoke(
                smoke_patch,
                tmp_path,
                config_graph_edges=TARGET_GRAPH_EDGES,
            )
    finally:
        for name, value in before.items():
            setattr(round0166_nodes, name, value)
