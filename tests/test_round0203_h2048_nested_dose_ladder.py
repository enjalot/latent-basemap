from __future__ import annotations

import pytest

from basemap.artifact_identity import expected_input_signature
from basemap.round0187_composition_nested_ladder import (
    RUNG_ROWS,
    train_config as high_dose_train_config,
)
from basemap.round0203_h2048_nested_dose_ladder import (
    HIDDEN_DIMENSION,
    ROUND_ID,
    TARGET_POSITIVE_DRAWS_PER_EDGE,
    Round0203Error,
    successful_updates_for_edges,
    train_config,
)
from basemap import round0113_prompt_contrast as prompt_contract
from experiments import round0166_nodes as q2
from experiments import round0202_nodes as delegate
from experiments import round0203_nodes as nodes
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


@pytest.mark.parametrize("rung", ("quarter", "half"))
def test_low_dose_config_changes_horizon_not_h2048_recipe(rung: str) -> None:
    path = GRAPHS[rung]
    graph = prompt_contract.read_sealed(path, label=f"accepted {rung} graph")
    kwargs = {
        "rung": rung,
        "graph_signature": graph["graph"],
        "graph_manifest_signature": expected_input_signature(path),
        "graph_edges": int(graph["directed_edge_count"]),
        "retained_rows": RUNG_ROWS[rung],
    }
    high, _ = high_dose_train_config(**kwargs)
    low, digest = train_config(**kwargs)
    assert ROUND_ID == "0203"
    assert len(digest) == 64
    assert low["model"] == high["model"]
    assert low["model"]["hidden_dimension"] == HIDDEN_DIMENSION == 2048
    assert low["graph"] == high["graph"]
    assert low["input"] == high["input"]
    assert low["optimizer"]["seed"] == high["optimizer"]["seed"] == 42
    assert low["optimizer"]["successful_positive_lr_updates"] == (
        EXPECTED_UPDATES[rung]
    )
    assert low["execution"]["expected_pipeline_stamp"] == (
        high["execution"]["expected_pipeline_stamp"]
    )
    assert low["execution"]["target_positive_draws_per_edge"] == (
        TARGET_POSITIVE_DRAWS_PER_EDGE
    )
    assert low["dose_registration"]["source_round"] == "0184"


def test_exact_horizons_use_same_rational_as_r0202() -> None:
    for rung, path in GRAPHS.items():
        graph = prompt_contract.read_sealed(path, label=f"accepted {rung} graph")
        assert successful_updates_for_edges(int(graph["directed_edge_count"])) == (
            EXPECTED_UPDATES[rung]
        )


def test_delegate_and_train_seal_reload_panel_cpu_smoke(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    for name in (
        "ROUND_ID",
        "CAPABILITY",
        "SEED",
        "RUNGS",
        "HIDDEN_DIMENSION",
        "HOST_RSS_LIMIT_GIB",
        "EVALUATION_SCHEMA",
        "SYNTHESIS_SCHEMA",
        "TARGET_POSITIVE_DRAWS_PER_EDGE",
        "Round0202Error",
        "successful_updates_for_edges",
        "train_config",
        "train_schema",
    ):
        monkeypatch.setattr(delegate, name, getattr(delegate, name))
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
    nodes._configure_delegate()
    delegate._configure_q2("quarter", {"graph_manifest": GRAPHS["quarter"]})
    assert delegate.ROUND_ID == q2.ROUND_ID == ROUND_ID
    assert q2.SUCCESSFUL_UPDATES == EXPECTED_UPDATES["quarter"]
    config, _digest = q2.scale_train_config(
        graph_signature={
            "kind": "file",
            "canonical_path": "/g",
            "bytes": 1,
            "sha256": "a" * 64,
        },
        graph_manifest_signature={
            "kind": "file",
            "canonical_path": "/m",
            "bytes": 1,
            "sha256": "b" * 64,
        },
        graph_edges=149_103_268,
        retained_rows=RUNG_ROWS["quarter"],
    )
    assert config["model"]["hidden_dimension"] == 2048
    _run_train_seal_reload_panel_cpu_smoke(
        monkeypatch,
        tmp_path,
        config_graph_edges=149_103_268,
        config_retained_rows=RUNG_ROWS["quarter"],
        expected_seed=42,
    )


def test_unknown_action_fails_closed() -> None:
    with pytest.raises(Round0203Error, match="does not authorize"):
        nodes.run_job(
            {"manifest": {"round_id": ROUND_ID}}, {"action": "train_h4096"}
        )
