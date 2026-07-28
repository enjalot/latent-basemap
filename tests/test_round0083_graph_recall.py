from __future__ import annotations

import inspect
import json

import numpy as np

from basemap.round0083_program import (
    NPROBES,
    SUCCESSFUL_UPDATES,
    train_config_from_graph,
)
from experiments import prepare_round0083_queue, round0083_nodes


SUBSTRATE = (
    "/data/latent-basemap/runs/round-0053/queue/artifacts/"
    "balanced-30m-int8-substrate/balanced-30m-int8-substrate-v1.json"
)
BASELINE_GRAPH = (
    "/data/latent-basemap/runs/round-0060/queue/artifacts/"
    "native-graph-balanced-30m/canonical-graph-v1.json"
)


def test_graph_treatment_config_preserves_r0061_recipe() -> None:
    graph = json.load(open(BASELINE_GRAPH, encoding="utf-8"))
    graph["round_id"] = "0083"
    graph["candidate_generator"]["nprobe"] = 16
    graph["quality"]["mean_recall_at_15_unambiguous"] = 0.85
    substrate = json.load(open(SUBSTRATE, encoding="utf-8"))
    config, digest = train_config_from_graph(
        graph,
        graph_manifest_path="/tmp/graph.json",
        graph_manifest_sha256="a" * 64,
        substrate_manifest=substrate,
        substrate_manifest_path=SUBSTRATE,
        substrate_manifest_sha256="b" * 64,
    )
    assert len(digest) == 64
    assert config["optimizer"]["seed"] == 42
    assert (
        config["optimizer"]["successful_positive_lr_updates"]
        == SUCCESSFUL_UPDATES
        == 500_003
    )
    assert config["execution"]["graph_recall_treatment"] == {
        "round_id": "0083",
        "nprobe": 16,
        "candidate_recall_at_15_unambiguous": 0.85,
        "baseline_nprobe": 64,
        "baseline_candidate_recall_at_15_unambiguous": 0.9224609375000001,
        "only_intended_difference_from_r0061": (
            "canonical graph neighbor identities induced by fixed nprobe"
        ),
    }
    stamp = config["execution"]["expected_pipeline_stamp"]
    assert stamp["positive_source_count"] == 29_781_754
    assert stamp["valid_canonical_edge_count"] == 446_726_310
    assert stamp["source_edge_uniform_equivalent"] is True


def test_full_graph_overlap_reports_the_actual_treatment(
    tmp_path, monkeypatch
) -> None:
    rows = 4
    k = round0083_nodes.K
    baseline = np.tile(np.arange(k, dtype="<i4"), (rows, 1))
    treatment = baseline.copy()
    treatment[1, -3:] = np.asarray([20, 21, 22], dtype="<i4")
    baseline_path = tmp_path / "baseline.i32"
    treatment_path = tmp_path / "treatment.i32"
    baseline.tofile(baseline_path)
    treatment.tofile(treatment_path)
    monkeypatch.setattr(round0083_nodes, "ROW_COUNT", rows)
    monkeypatch.setattr(round0083_nodes, "EXPECTED_RETAINED_ROWS", rows - 1)
    summary = round0083_nodes._graph_overlap(
        treatment_path=str(treatment_path),
        baseline_path=str(baseline_path),
        excluded=np.asarray([0], dtype=np.int64),
    )
    assert summary["sources"] == 3
    assert summary["overlap_count_histogram"] == {"12": 1, "15": 2}
    assert summary["identical_neighbor_set_fraction"] == 2 / 3
    assert summary["mean_neighbor_overlap_fraction"] == 42 / 45


def test_sensitivity_classification_is_ordered_by_measured_recall() -> None:
    def cells(low_pass: bool, high_pass: bool) -> dict:
        return {
            "16": {
                "candidate_recall_at_15_unambiguous": 0.85,
                "passed": low_pass,
            },
            "32": {
                "candidate_recall_at_15_unambiguous": 0.89,
                "passed": high_pass,
            },
        }

    assert round0083_nodes.classify_sensitivity(
        cells(True, True)
    )["verdict"] == "insensitive-through-lowest-tested-recall"
    assert round0083_nodes.classify_sensitivity(
        cells(False, True)
    )["verdict"] == "sensitive-between-tested-recalls"
    assert round0083_nodes.classify_sensitivity(
        cells(False, False)
    )["verdict"] == "current-floor-load-bearing-within-tested-range"
    assert round0083_nodes.classify_sensitivity(
        cells(True, False)
    )["verdict"] == "nonmonotonic-map-outcome-requires-follow-up"


def test_queue_has_two_fixed_graph_train_panel_cells() -> None:
    source = inspect.getsource(
        prepare_round0083_queue.prepare_round0083
    )
    assert NPROBES == (16, 32)
    assert "gpu_hours_cap=6.0" in source
    assert "planning_bands_not_admission_gates" in source
    assert 'action="build_graph"' in source
    assert 'action="train"' in source
    assert 'action="panel"' in source
    assert "successful_updates=SUCCESSFUL_UPDATES" in source
    assert "batch_size=8_192" in source


def test_graph_builder_stamps_sampler_and_actual_dose() -> None:
    source = inspect.getsource(round0083_nodes.run_build_graph)
    assert "_graph_overlap(" in source
    assert '"uniform retained source, then uniform one of 15 destinations;' in source
    assert '"mean_recall_at_15_unambiguous"' in source
    assert '"planning_band_is_not_an_admission_gate": True' in source
    assert "exact_rerank" in source


def test_treatment_does_not_adapt_nprobe_or_change_seed() -> None:
    qualify = inspect.getsource(round0083_nodes.run_qualification)
    train = inspect.getsource(round0083_nodes.run_train)
    assert "for nprobe in NPROBES" in qualify
    assert "selected_nprobe" not in qualify
    assert "optimizer" not in train
    assert "seed" not in train.lower()


def test_handler_uses_manifest_round_and_registered_actions() -> None:
    source = inspect.getsource(round0083_nodes.run_job)
    assert 'active.get("manifest", {}).get("round_id")' in source
    assert 'active.get("round_id")' not in source
    for action in (
        "qualify",
        "build_graph",
        "train",
        "transform",
        "panel",
        "comparison",
    ):
        assert f'action == "{action}"' in source
