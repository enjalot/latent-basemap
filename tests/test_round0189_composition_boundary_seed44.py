from __future__ import annotations

import json

import numpy as np
import pytest

from basemap.artifact_identity import expected_input_signature
from basemap.round0187_composition_nested_ladder import (
    PRIMARY_METRICS,
    REQUIRED_TRAIN_CHECKS,
    RUNG_ROWS,
    TARGET_POSITIVE_DRAWS_PER_EDGE,
)
from basemap.round0189_composition_boundary_seed44 import (
    ROUND_ID,
    SEED,
    Round0189Error,
    boundary_decision,
    successful_updates_for_edges,
    train_checks_close,
    train_config,
)
from basemap import round0113_prompt_contrast as prompt_contract
from experiments import round0166_nodes as q2
from experiments import round0189_nodes as nodes
from tests.test_round0166_cpu_smoke import _run_train_seal_reload_panel_cpu_smoke


HALF_GRAPH = (
    "/data/latent-basemap/runs/round-0187/queue-correction-1/artifacts/"
    "half-graph/graph-manifest.json"
)
FULL_GRAPH = (
    "/data/latent-basemap/runs/round-0171/queue/artifacts/"
    "fuzzy-k50-graph-and-reference/graph-manifest.json"
)


def _graph(path: str) -> dict:
    return prompt_contract.read_sealed(path, label="test graph")


def _config(rung: str, path: str) -> dict:
    graph = _graph(path)
    config, _digest = train_config(
        rung=rung,
        graph_signature=graph["graph"],
        graph_manifest_signature=expected_input_signature(path),
        graph_edges=int(graph["directed_edge_count"]),
        retained_rows=RUNG_ROWS[rung],
    )
    return config


def _metrics(pile_ffr: float, *, scale: float = 1.0) -> dict[str, float]:
    return {
        metric: (pile_ffr if metric == "pile_ffr" else (0.5 + i / 100) * scale)
        for i, metric in enumerate(PRIMARY_METRICS)
    }


def test_round_and_seed_are_frozen() -> None:
    assert ROUND_ID == "0189"
    assert SEED == 44


def test_exact_r0187_dose_horizons_are_reproduced() -> None:
    assert successful_updates_for_edges(300_567_710) == 1_009_962
    assert successful_updates_for_edges(603_086_368) == 2_026_478


@pytest.mark.parametrize("rung,path", [("half", HALF_GRAPH), ("full", FULL_GRAPH)])
def test_config_changes_only_seed_bound_fields(rung: str, path: str) -> None:
    config = _config(rung, path)
    graph = _graph(path)
    expected_updates = successful_updates_for_edges(int(graph["directed_edge_count"]))
    stamp = config["execution"]["expected_pipeline_stamp"]
    assert config["paired_invariant"]["seed"] == 44
    assert config["optimizer"]["seed"] == 44
    assert config["optimizer"]["positive_rng_seed"] == 44
    assert config["optimizer"]["negative_rng_seed"] == 11_300_044
    assert stamp["positive_rng_seed"] == 44
    assert stamp["negative_rng_seed"] == 11_300_044
    assert stamp["compact_retained_rows"] == RUNG_ROWS[rung]
    assert config["input"]["rows"] == RUNG_ROWS[rung]
    assert config["optimizer"]["successful_positive_lr_updates"] == expected_updates
    assert np.isclose(
        config["execution"]["target_positive_draws_per_edge"],
        TARGET_POSITIVE_DRAWS_PER_EDGE,
        rtol=0,
        atol=1e-15,
    )
    assert config["model"]["hidden_dimension"] == 2048
    serialized = json.dumps(config, sort_keys=True)
    assert "seed43" not in serialized
    assert "11300043" not in serialized


def test_invalid_rung_and_cardinality_fail_closed() -> None:
    graph = _graph(HALF_GRAPH)
    signature = expected_input_signature(HALF_GRAPH)
    with pytest.raises(Round0189Error):
        train_config(
            rung="quarter",
            graph_signature=graph["graph"],
            graph_manifest_signature=signature,
            graph_edges=int(graph["directed_edge_count"]),
            retained_rows=RUNG_ROWS["quarter"],
        )
    with pytest.raises(Round0189Error):
        train_config(
            rung="half",
            graph_signature=graph["graph"],
            graph_manifest_signature=signature,
            graph_edges=int(graph["directed_edge_count"]),
            retained_rows=RUNG_ROWS["half"] - 1,
        )


def test_nonvacuous_train_checks() -> None:
    assert train_checks_close({key: True for key in REQUIRED_TRAIN_CHECKS})
    assert not train_checks_close({})


def test_seed44_positive_replay_is_reported_without_aggregate_claim() -> None:
    decision = boundary_decision(
        seed42={"half": _metrics(0.4556), "full": _metrics(0.4358)},
        seed44={"half": _metrics(0.5000), "full": _metrics(0.4800)},
    )
    assert decision["outcome"] == (
        "composition-controlled-size-regression-seed44-positive"
    )
    assert decision["seed44_confirms_registered_regression"] is True
    assert "R0188" in decision["follow_up"]


def test_seed44_negative_replay_is_reported_without_capacity_decision() -> None:
    decision = boundary_decision(
        seed42={"half": _metrics(0.4556), "full": _metrics(0.4358)},
        seed44={"half": _metrics(0.5000), "full": _metrics(0.4900)},
    )
    assert decision["outcome"] == (
        "composition-controlled-size-regression-seed44-negative"
    )
    assert decision["seed44_confirms_registered_regression"] is False


def test_changed_seed42_premise_fails_closed() -> None:
    with pytest.raises(Round0189Error, match="R0187 Pile-FFR trigger"):
        boundary_decision(
            seed42={"half": _metrics(0.4556), "full": _metrics(0.4500)},
            seed44={"half": _metrics(0.5000), "full": _metrics(0.4800)},
        )


Q2_BINDINGS = (
    "ROUND_ID",
    "CAPABILITY",
    "SEED",
    "SUCCESSFUL_UPDATES",
    "HOST_RSS_LIMIT_GIB",
    "Round0166Error",
    "GRAPH_SCHEMA",
    "TRAIN_SCHEMA",
    "PRODUCTION_CONFIG_SCHEMA",
    "GRAPH_INDEX_DESCRIPTION",
    "GRAPH_REFERENCE_ROW_ORDER",
    "GRAPH_REFERENCE_ANCHOR_NAMESPACE",
    "GRAPH_SOURCE_ROUND_ID",
    "GRAPH_BUILT_IN_ROUND",
    "POPULATION_READER",
    "MIN_SCALE_ROWS_EXCLUSIVE",
    "ScalePromptTrainingInput",
    "scale_train_config",
)


def test_q2_configuration_binds_seed44_and_reviewed_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in Q2_BINDINGS:
        monkeypatch.setattr(q2, name, getattr(q2, name))
    nodes._configure_q2("half", {"rung": "half", "graph_manifest": HALF_GRAPH})
    assert q2.ROUND_ID == "0189"
    assert q2.SEED == 44
    assert q2.SUCCESSFUL_UPDATES == 1_009_962
    assert q2.GRAPH_SCHEMA == "round0187-composition-nested-fuzzy-graph-half-v1"
    assert q2.PRODUCTION_CONFIG_SCHEMA == (
        "round0189-half-seed44-production-config-v1"
    )
    assert q2.GRAPH_SOURCE_ROUND_ID == "0187"
    assert q2.GRAPH_BUILT_IN_ROUND is False


def test_seed44_train_seal_reload_panel_cpu_smoke(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    for name in Q2_BINDINGS:
        monkeypatch.setattr(q2, name, getattr(q2, name))
    nodes._configure_q2("full", {"rung": "full", "graph_manifest": FULL_GRAPH})
    _run_train_seal_reload_panel_cpu_smoke(
        monkeypatch,
        tmp_path,
        config_graph_edges=603_086_368,
        expected_seed=44,
    )


def test_unknown_action_fails_before_execution() -> None:
    with pytest.raises(Round0189Error, match="does not authorize"):
        nodes.run_job({"manifest": {"round_id": "0189"}}, {"action": "h4096"})
