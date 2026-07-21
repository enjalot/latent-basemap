"""Focused CPU-only checks for the matched R0030 sampling round."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from basemap.artifact_identity import canonical_json, expected_input_signature, sha256_bytes
from basemap.round0030_program import (
    GRAPH_EFFECTIVE_EDGES,
    GRAPH_RESIDENT_EDGES,
    R0023_SEED_SPREAD,
    train_config_for_arm,
)
from experiments import round0030_compare as comparison
from experiments import run_round0014_node as node
from experiments.prepare_round0030_queue import _jobs


def _leaf_differences(left: Any, right: Any, prefix: str = "") -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        assert set(left) == set(right)
        out: set[str] = set()
        for key in left:
            path = f"{prefix}.{key}" if prefix else key
            out.update(_leaf_differences(left[key], right[key], path))
        return out
    return {prefix} if left != right else set()


def test_round0030_arm_configs_differ_only_by_registered_sampling_fields() -> None:
    uniform, uniform_sha = train_config_for_arm("uniform")
    fuzzy, fuzzy_sha = train_config_for_arm("fuzzy")
    assert uniform_sha != fuzzy_sha
    assert uniform["optimizer"]["seed"] == fuzzy["optimizer"]["seed"] == 42
    assert uniform["graph"]["path"] == fuzzy["graph"]["path"]
    assert uniform["graph"]["sha256"] == fuzzy["graph"]["sha256"]
    assert uniform["graph"]["directed_edges_effective"] == GRAPH_EFFECTIVE_EDGES
    assert fuzzy["graph"]["directed_edges"] == GRAPH_RESIDENT_EDGES
    assert uniform["execution"]["required_pipeline"] == "hybrid"
    assert fuzzy["execution"]["required_pipeline"] == "hybrid"
    assert uniform["execution"]["expected_pipeline_stamp"] == {
        "pipeline": "hybrid",
        "sampler_class": "HostStreamEdgeSampler",
        "positive_sampling": "uniform",
        "uniform_with_replacement": True,
        "positive_with_replacement": True,
        "weighted_requested": False,
        "weighted_effective": False,
        "x_residency": "device_fp16",
        "multiplicity_positive_source_sampling": (
            "uniform_over_retained_source_edges_with_replacement"
        ),
        "multiplicity_graph_degree": "variable_or_weighted_edge_universe",
    }
    assert fuzzy["execution"]["expected_pipeline_stamp"]["positive_sampling"] == (
        "weighted_with_replacement"
    )
    assert fuzzy["execution"]["expected_pipeline_stamp"]["weighted_effective"] is True
    expected_differences = set(
        uniform["execution"]["matched_arm_invariants"]["same_except"]
    )
    assert _leaf_differences(uniform, fuzzy) == expected_differences


def test_round0030_seed_spreads_are_metric_specific() -> None:
    assert R0023_SEED_SPREAD["ffr"] == pytest.approx(0.0222)
    assert R0023_SEED_SPREAD["density"] == pytest.approx(0.0194)
    assert R0023_SEED_SPREAD["purity_k1024"] == pytest.approx(0.0562)
    assert R0023_SEED_SPREAD["recall_at_50"] == pytest.approx(0.00021)


def test_round0030_queue_runs_both_trains_before_evaluation(tmp_path: Path) -> None:
    templates = {arm: str(tmp_path / f"{arm}.json") for arm in ("uniform", "fuzzy")}
    jobs = _jobs(artifacts=str(tmp_path / "artifacts"), templates=templates, inputs=[])
    ids = [job["id"] for job in jobs]
    assert ids[:3] == ["sampler_canary", "uniform_train_30m", "fuzzy_train_30m"]
    assert ids[-1] == "comparison"
    assert jobs[-1]["node_policy"]["gpu_required"] is False
    assert sum(job["handler"] == "train_seed42_30m" for job in jobs) == 2
    assert sum(job["handler"] == "round0030_ood_panel" for job in jobs) == 2
    assert all(job["canary_output"].endswith("sampler-canary")
               for job in jobs if job["handler"] == "train_seed42_30m")


def test_round0030_canary_uses_admission_not_fit() -> None:
    source = inspect.getsource(node._run_round0030_sampler_canary)
    assert "_prepare_edge_list_training" in source
    assert ".fit(" not in source
    assert "optimizer_updates\": 0" in source


def _write_sealed(path: Path, body: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {**body, "identity_sha256": sha256_bytes(canonical_json(body))}
    path.write_text(json.dumps(value), encoding="utf-8")


def _comparison_fixture(
    root: Path, *, fuzzy_ffr_delta: float, fuzzy_ood_delta: float = 0.0
) -> dict[str, dict[str, str]]:
    graph_signature = expected_input_signature(str(root / "graph"))
    outputs: dict[str, dict[str, str]] = {}
    for arm in ("uniform", "fuzzy"):
        config, config_sha = train_config_for_arm(arm)
        arm_root = root / arm
        model = arm_root / "train" / "model.pt"
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(arm.encode("ascii"))
        stamp = config["execution"]["expected_pipeline_stamp"]
        stats = {
            **{f"pipeline_{key}": value for key, value in stamp.items()},
            "n_pos_edges": GRAPH_EFFECTIVE_EDGES,
            "positive_lr_optimizer_steps": 500_000,
            "optimizer_steps_succeeded": 500_000,
            "budget_satisfied": True,
            "pipeline_multiplicity_cap_artifact_sha256": (
                "9511ceca802da603bfbfe9164f8c6ffd7006df82df17b9499d4ed33288fde7cb"
            ),
            "pipeline_multiplicity_positive_edges_resident": GRAPH_RESIDENT_EDGES,
            "pipeline_multiplicity_positive_edges_effective": GRAPH_EFFECTIVE_EDGES,
            "verified_hashes": {"graph_sha256": comparison.GRAPH_SHA256},
        }
        _write_sealed(
            arm_root / "train" / "train-receipt.json",
            {
                "schema": f"round0030-{arm}-train-receipt-v1",
                "production_config_sha256": config_sha,
                "model": expected_input_signature(str(model)),
                "train_stats": stats,
            },
        )
        ffr = 0.45 + (fuzzy_ffr_delta if arm == "fuzzy" else 0.0)
        _write_sealed(
            arm_root / "panel" / "panel.json",
            {
                "schema": f"round0030-{arm}-registered-panel-v1",
                "production_config_sha256": config_sha,
                "registered_inputs": {"graph": graph_signature},
                "panel": {
                    "ffr": ffr,
                    "density": 0.78,
                    "purity": {"k256": 1.12, "k1024": 0.92},
                },
                "recall@10": 0.0037,
                "recall@50": 0.0046,
                "projection": {"proj_ffr": 0.42},
                "selector_passed": True,
            },
        )
        retention = 0.60 + (fuzzy_ood_delta if arm == "fuzzy" else 0.0)
        _write_sealed(
            arm_root / "ood" / "universality-panel-v1.json",
            {
                "schema": "universality-panel-v1",
                "map": {
                    "label": f"round0030-{arm}",
                    "model": expected_input_signature(str(model)),
                    "coordinate_receipt": {"sha256": arm},
                },
                "probes": {
                    "dadabase": {
                        "status": "included", "retention": retention,
                        "verdict": "amber",
                    },
                    "trec-covid": {
                        "status": "included", "retention": 0.38,
                        "verdict": "failure",
                    },
                    "scifact": {"status": "excluded"},
                },
            },
        )
        outputs[arm] = {
            "train": str(arm_root / "train"),
            "panel": str(arm_root / "panel"),
            "ood_panel": str(arm_root / "ood"),
        }
    return outputs


@pytest.mark.parametrize(
    ("ffr_delta", "expected"),
    [(0.03, "adopt"), (0.01, "inconclusive/replicate"), (-0.03, "reject")],
)
def test_round0030_registered_decision_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ffr_delta: float, expected: str
) -> None:
    graph = tmp_path / "graph"
    graph.write_bytes(b"graph")
    graph_sha = expected_input_signature(str(graph))["sha256"]
    monkeypatch.setattr(comparison, "GRAPH_PATH", str(graph))
    monkeypatch.setattr(comparison, "GRAPH_SHA256", graph_sha)
    outputs = _comparison_fixture(tmp_path, fuzzy_ffr_delta=ffr_delta)
    result = comparison.build_comparison(
        release_sha="a" * 40, arm_outputs=outputs)
    assert result["decision"] == expected
    assert result["paired_replication_required"] is (expected == "inconclusive/replicate")


def test_round0030_ood_regression_rejects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = tmp_path / "graph"
    graph.write_bytes(b"graph")
    graph_sha = expected_input_signature(str(graph))["sha256"]
    monkeypatch.setattr(comparison, "GRAPH_PATH", str(graph))
    monkeypatch.setattr(comparison, "GRAPH_SHA256", graph_sha)
    outputs = _comparison_fixture(
        tmp_path, fuzzy_ffr_delta=0.03, fuzzy_ood_delta=-0.03)
    result = comparison.build_comparison(
        release_sha="a" * 40, arm_outputs=outputs)
    assert result["decision"] == "reject"
    assert any(reason.startswith("ood-dadabase") for reason in result["reject_reasons"])
