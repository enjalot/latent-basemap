"""Contract, policy, and end-to-end CPU smoke tests for R0184."""
from __future__ import annotations

import pytest

from basemap.round0166_prompted_8m import METRICS
from basemap.round0184_prompted_8m_dose_midpoint import (
    ACHIEVED_POSITIVE_DRAWS_PER_EDGE,
    RETAINED_ROWS,
    ROUND_ID,
    SUCCESSFUL_UPDATES,
    TARGET_GRAPH_EDGES,
    diagnostic_scale_decision,
    scale_train_config,
)
from experiments import round0166_nodes, round0184_nodes
from tests.test_round0166_cpu_smoke import _run_train_seal_reload_panel_cpu_smoke


def test_midpoint_dose_is_exactly_one_million_updates() -> None:
    assert SUCCESSFUL_UPDATES == 1_000_000
    assert ACHIEVED_POSITIVE_DRAWS_PER_EDGE == pytest.approx(
        1_000_000 * 409 / 603_086_368
    )


def test_config_changes_only_the_registered_dose_horizon() -> None:
    signature = {
        "kind": "file",
        "canonical_path": "/accepted/r0171-graph",
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
    assert config["optimizer"]["successful_positive_lr_updates"] == 1_000_000
    assert config["paired_invariant"]["successful_positive_lr_updates"] == 1_000_000
    assert config["dose_registration"]["curve_population_round"] == "0171"
    assert config["dose_registration"]["lower_point"]["round"] == "0171"
    assert config["dose_registration"]["upper_point"]["round"] == "0180"
    assert config["graph"]["k"] == 50
    assert config["optimizer"]["seed"] == 42


def test_metric_miss_is_valid_diagnostic_but_execution_miss_is_not() -> None:
    native = {metric: 0.5 for metric in METRICS}
    matched = {metric: 0.5 for metric in METRICS}
    baseline = {metric: 1.0 for metric in METRICS}
    floors = {metric: 1.0 for metric in METRICS}
    metric_decision = diagnostic_scale_decision(
        native=native,
        matched_2m=matched,
        baseline_2m=baseline,
        prompted_floors=floors,
    )
    assert metric_decision["passed"] is False
    assert metric_decision["metric_gates_required_for_capability"] is False

    valid = round0166_nodes._finalize_scale_decision(
        metric_decision, {"finite": True, "accounting": True}
    )
    assert valid["metric_gates_passed"] is False
    assert valid["execution_gates_passed"] is True
    assert valid["passed"] is True
    assert valid["outcome"] == "prompted-english-8m-dose-readout-valid"

    invalid = round0166_nodes._finalize_scale_decision(
        metric_decision, {"finite": True, "accounting": False}
    )
    assert invalid["passed"] is False
    assert invalid["outcome"] == "prompted-english-8m-execution-invalid"


def test_historical_scale_decisions_remain_metric_gated() -> None:
    decision = round0166_nodes._finalize_scale_decision(
        {"passed": False}, {"execution": True}
    )
    assert decision["metric_gates_required_for_capability"] is True
    assert decision["passed"] is False
    assert decision["outcome"] == "prompted-english-8m-scale-rung-not-qualified"

    with pytest.raises(RuntimeError, match="release policy"):
        round0166_nodes._finalize_scale_decision(
            {"passed": True, "metric_gates_required_for_capability": "no"},
            {"execution": True},
        )


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
            "metric_policy": round0166_nodes.scale_decision,
        }),
    )
    try:
        round0184_nodes.run_job(
            {"manifest": {"round_id": ROUND_ID}},
            {"action": "train_prompted_8m"},
        )
        assert observed == {
            "round_id": "0184",
            "updates": 1_000_000,
            "graph_round": "0171",
            "graph_built": False,
            "metric_policy": diagnostic_scale_decision,
        }
        with pytest.raises(RuntimeError, match="does not authorize"):
            round0184_nodes.run_job(
                {"manifest": {"round_id": ROUND_ID}},
                {"action": "build_graph_and_reference"},
            )
    finally:
        for name, value in before.items():
            setattr(round0166_nodes, name, value)


def test_r0184_train_seal_reload_panel_cpu_smoke(monkeypatch, tmp_path) -> None:
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
        round0184_nodes._configure()
        with monkeypatch.context() as smoke_patch:
            _run_train_seal_reload_panel_cpu_smoke(
                smoke_patch,
                tmp_path,
                config_graph_edges=TARGET_GRAPH_EDGES,
            )
    finally:
        for name, value in before.items():
            setattr(round0166_nodes, name, value)
