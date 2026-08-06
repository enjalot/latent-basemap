from __future__ import annotations

import pytest

from basemap.artifact_identity import expected_input_signature
from basemap.round0187_composition_nested_ladder import PRIMARY_METRICS, RUNG_ROWS
from basemap.round0202_h4096_nested_dose_ladder import (
    FULL_GRAPH_EDGES,
    FULL_SUCCESSFUL_UPDATES,
    HIDDEN_DIMENSION,
    ROUND_ID,
    TARGET_POSITIVE_DRAWS_PER_EDGE,
    Round0202Error,
    ladder_summary,
    successful_updates_for_edges,
    train_config,
)
from basemap import round0113_prompt_contrast as prompt_contract
from experiments import round0166_nodes as q2
from experiments import round0188_nodes as evaluator
from experiments import round0202_nodes as nodes
from tests.test_round0166_cpu_smoke import _run_train_seal_reload_panel_cpu_smoke


GRAPHS = {
    "quarter": (
        "/data/latent-basemap/runs/round-0187/queue-correction-1/artifacts/"
        "quarter-graph/graph-manifest.json"
    ),
    "half": (
        "/data/latent-basemap/runs/round-0187/queue-correction-1/artifacts/"
        "half-graph/graph-manifest.json"
    ),
}
EXPECTED_UPDATES = {"quarter": 247_234, "half": 498_383}


def _metrics(value: float) -> dict[str, float]:
    return {metric: value for metric in PRIMARY_METRICS}


def test_exact_r0191_dose_horizons() -> None:
    assert ROUND_ID == "0202"
    assert successful_updates_for_edges(FULL_GRAPH_EDGES) == FULL_SUCCESSFUL_UPDATES
    for rung, path in GRAPHS.items():
        graph = prompt_contract.read_sealed(path, label=f"accepted {rung} graph")
        edges = int(graph["directed_edge_count"])
        updates = successful_updates_for_edges(edges)
        assert updates == EXPECTED_UPDATES[rung]
        achieved = updates * prompt_contract.POSITIVE_ROWS_PER_UPDATE / edges
        assert achieved >= TARGET_POSITIVE_DRAWS_PER_EDGE
        assert achieved - TARGET_POSITIVE_DRAWS_PER_EDGE < (
            prompt_contract.POSITIVE_ROWS_PER_UPDATE / edges
        )


@pytest.mark.parametrize("rung", ("quarter", "half"))
def test_config_freezes_nested_recipe_and_changes_width_and_dose(rung: str) -> None:
    path = GRAPHS[rung]
    graph = prompt_contract.read_sealed(path, label=f"accepted {rung} graph")
    config, digest = train_config(
        rung=rung,
        graph_signature=graph["graph"],
        graph_manifest_signature=expected_input_signature(path),
        graph_edges=int(graph["directed_edge_count"]),
        retained_rows=RUNG_ROWS[rung],
    )
    assert len(digest) == 64
    assert config["model"]["hidden_dimension"] == HIDDEN_DIMENSION == 4096
    assert config["optimizer"]["seed"] == 42
    assert config["optimizer"]["successful_positive_lr_updates"] == (
        EXPECTED_UPDATES[rung]
    )
    assert config["input"]["rows"] == RUNG_ROWS[rung]
    assert config["execution"]["expected_pipeline_stamp"]["sampler_class"] == (
        "PromptWeightedJinaSampler"
    )
    assert config["execution"]["target_positive_draws_per_edge"] == (
        TARGET_POSITIVE_DRAWS_PER_EDGE
    )
    assert config["dose_registration"]["source_round"] == "0191"


def test_ladder_summary_reports_step_and_compound_retention_without_deciding() -> None:
    summary = ladder_summary({
        "quarter": _metrics(1.0),
        "half": _metrics(0.98),
        "full": _metrics(0.95),
    })
    registered = summary["registered_pile_ffr_retentions"]
    assert registered["half_over_quarter"] == pytest.approx(0.98)
    assert registered["full_over_half"] == pytest.approx(0.95 / 0.98)
    assert registered["full_over_quarter"] == pytest.approx(0.95)
    assert summary["decision_deferred_to_track_a3"] is True
    with pytest.raises(Round0202Error, match="cells changed"):
        ladder_summary({"quarter": _metrics(1.0), "half": _metrics(0.98)})


def test_q2_configuration_and_train_seal_reload_panel_cpu_smoke(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    # Preserve every shared global that R0202 intentionally configures.
    for name in (
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
    ):
        monkeypatch.setattr(q2, name, getattr(q2, name))
    nodes._configure_q2("quarter", {"graph_manifest": GRAPHS["quarter"]})
    assert q2.ROUND_ID == ROUND_ID
    assert q2.SUCCESSFUL_UPDATES == EXPECTED_UPDATES["quarter"]
    assert q2.GRAPH_SOURCE_ROUND_ID == "0187"
    assert q2.GRAPH_BUILT_IN_ROUND is False
    _run_train_seal_reload_panel_cpu_smoke(
        monkeypatch,
        tmp_path,
        config_graph_edges=149_103_268,
        config_retained_rows=RUNG_ROWS["quarter"],
        expected_seed=42,
    )


def test_evaluator_dispatch_and_unknown_action_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "ROUND_ID",
        "CAPABILITY",
        "SEED",
        "RUNGS",
        "EVALUATION_SCHEMA",
        "Round0188Error",
        "_configure_q2",
        "_load_seed43_model",
    ):
        monkeypatch.setattr(evaluator, name, getattr(evaluator, name))
    nodes._configure_evaluator()
    assert evaluator.ROUND_ID == ROUND_ID
    assert evaluator.RUNGS == ("quarter", "half")
    with pytest.raises(Round0202Error, match="does not authorize"):
        nodes.run_job(
            {"manifest": {"round_id": ROUND_ID}}, {"action": "build_graph"}
        )
