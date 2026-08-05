from __future__ import annotations

import copy
import json

import pytest

from basemap.artifact_identity import expected_input_signature
from basemap.round0187_composition_nested_ladder import (
    PRIMARY_METRICS,
    RUNG_ROWS,
    train_config as seed42_train_config,
)
from basemap.round0192_quarter_seed_family import (
    GATE_METRICS,
    ROUND_ID,
    ROWS,
    SEEDS,
    Round0192Error,
    seed_family,
    successful_updates_for_edges,
    train_config,
)
from basemap import round0113_prompt_contrast as prompt_contract
from experiments import round0166_nodes as q2
from experiments import round0192_nodes as nodes
from tests.test_round0166_cpu_smoke import _run_train_seal_reload_panel_cpu_smoke


QUARTER_GRAPH = (
    "/data/latent-basemap/runs/round-0187/queue-correction-1/artifacts/"
    "quarter-graph/graph-manifest.json"
)
SEED42_EVALUATION = (
    "/data/latent-basemap/runs/round-0187/queue-correction-1/artifacts/"
    "quarter-common-core-evaluation/common-core-evaluation.json"
)


def _graph() -> dict:
    return prompt_contract.read_sealed(QUARTER_GRAPH, label="quarter graph")


def _config(seed: int) -> dict:
    graph = _graph()
    return train_config(
        seed=seed,
        graph_signature=graph["graph"],
        graph_manifest_signature=expected_input_signature(QUARTER_GRAPH),
        graph_edges=int(graph["directed_edge_count"]),
        retained_rows=ROWS,
    )[0]


def _evaluation(seed: int, offset: float = 0.0) -> dict:
    primary = {
        metric: 0.45 + index / 100 + offset
        for index, metric in enumerate(PRIMARY_METRICS)
    }
    return {
        "primary_metrics": primary,
        "diagnostic_metrics": {
            "mixed_density": 0.20 + offset,
            "mixed_projection_ffr": 0.52 + offset,
        },
        "pile_ood": {"ffr": 0.52 + offset},
        "seed": seed,
    }


def test_exact_quarter_horizon_and_seed_set() -> None:
    graph = _graph()
    assert ROUND_ID == "0192"
    assert SEEDS == (43, 44)
    assert ROWS == RUNG_ROWS["quarter"] == 1_988_104
    assert int(graph["directed_edge_count"]) == 149_103_268
    assert successful_updates_for_edges(int(graph["directed_edge_count"])) == 501_014


@pytest.mark.parametrize("seed", SEEDS)
def test_config_changes_only_seed_bound_fields(seed: int) -> None:
    graph = _graph()
    signature = expected_input_signature(QUARTER_GRAPH)
    seed42, _ = seed42_train_config(
        rung="quarter",
        graph_signature=graph["graph"],
        graph_manifest_signature=signature,
        graph_edges=int(graph["directed_edge_count"]),
        retained_rows=ROWS,
    )
    treatment = _config(seed)
    assert treatment["input"] == seed42["input"]
    assert treatment["graph"] == seed42["graph"]
    assert treatment["model"] == seed42["model"]
    assert treatment["optimizer"]["successful_positive_lr_updates"] == 501_014
    assert treatment["optimizer"]["seed"] == seed
    assert treatment["optimizer"]["positive_rng_seed"] == seed
    assert treatment["optimizer"]["negative_rng_seed"] == 11_300_000 + seed
    stamp = treatment["execution"]["expected_pipeline_stamp"]
    assert stamp["positive_rng_seed"] == seed
    assert stamp["negative_rng_seed"] == 11_300_000 + seed
    normalized = copy.deepcopy(treatment)
    normalized["schema"] = seed42["schema"]
    normalized["paired_invariant"] = copy.deepcopy(seed42["paired_invariant"])
    normalized["optimizer"]["seed"] = 42
    normalized["optimizer"]["positive_rng_seed"] = 42
    normalized["optimizer"]["negative_rng_seed"] = 11_300_042
    normalized["execution"]["expected_pipeline_stamp"]["positive_rng_seed"] = 42
    normalized["execution"]["expected_pipeline_stamp"]["negative_rng_seed"] = 11_300_042
    normalized["execution"]["scale_change"] = seed42["execution"]["scale_change"]
    assert normalized == seed42


def test_family_binds_three_seeds_but_does_not_register_gates() -> None:
    family = seed_family(
        evaluations={
            42: _evaluation(42, 0.00),
            43: _evaluation(43, 0.01),
            44: _evaluation(44, 0.02),
        }
    )
    assert family["outcome"] == "mixed-quarter-three-seed-family-complete"
    assert family["gate_registration_deferred_to_reviewed_cpu_round"] is True
    assert set(family["gate_metric_cells"]) == {"42", "43", "44"}
    assert set(family["descriptive_summaries"]) == set(GATE_METRICS)
    assert family["descriptive_summaries"]["density_v2"]["mean"] == pytest.approx(0.21)
    assert family["descriptive_summaries"]["density_v2"]["sample_sd_ddof1"] == pytest.approx(0.01)


def test_changed_seed_set_or_projection_binding_fails_closed() -> None:
    with pytest.raises(Round0192Error, match="seed family"):
        seed_family(evaluations={42: _evaluation(42), 43: _evaluation(43)})
    bad = _evaluation(42)
    bad["pile_ood"]["ffr"] = 0.5
    with pytest.raises(Round0192Error, match="projection FFR"):
        seed_family(evaluations={42: bad, 43: _evaluation(43), 44: _evaluation(44)})


@pytest.mark.parametrize("seed", SEEDS)
def test_seed_train_seal_reload_panel_cpu_smoke(
    seed: int, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    names = (
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
    for name in names:
        monkeypatch.setattr(q2, name, getattr(q2, name))
    nodes._configure_q2(
        "quarter", {"seed": seed, "graph_manifest": QUARTER_GRAPH}
    )
    assert q2.ROUND_ID == ROUND_ID
    assert q2.SEED == seed
    assert q2.SUCCESSFUL_UPDATES == 501_014
    assert q2.GRAPH_SOURCE_ROUND_ID == "0187"
    assert q2.GRAPH_BUILT_IN_ROUND is False
    _run_train_seal_reload_panel_cpu_smoke(
        monkeypatch,
        tmp_path,
        config_graph_edges=149_103_268,
        config_retained_rows=ROWS,
        expected_seed=seed,
    )


def test_accepted_seed42_evaluation_has_required_gate_vector() -> None:
    with open(SEED42_EVALUATION, encoding="utf-8") as handle:
        evaluation = json.load(handle)
    prompt_contract.validate_seal(evaluation, label="accepted seed42 evaluation")
    family = seed_family(
        evaluations={
            42: evaluation,
            43: _evaluation(43),
            44: _evaluation(44),
        }
    )
    assert family["gate_metric_cells"]["42"]["density_v2"] == 0.2022
    assert family["gate_metric_cells"]["42"]["projection_ffr"] == 0.5198


def test_unknown_action_fails_before_execution() -> None:
    with pytest.raises(Round0192Error, match="does not authorize"):
        nodes.run_job({"manifest": {"round_id": ROUND_ID}}, {"action": "gate"})
