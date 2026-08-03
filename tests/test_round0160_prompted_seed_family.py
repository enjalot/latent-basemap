"""CUDA-hidden contracts for the R0160 prompted seed family."""
from __future__ import annotations

import pytest

from basemap.round0113_prompt_contrast import RETAINED_ROWS, Round0113Error, train_config
from basemap.round0160_prompted_seed_family import (
    METRICS,
    SEEDS,
    Round0160Error,
    build_family_evidence,
    metric_view,
)
from experiments.round0113_nodes import _graph_execution_round_id, _training_seed


GRAPH = {"canonical_path": "/tmp/graph", "kind": "file", "bytes": 1, "sha256": "a" * 64}
MANIFEST = {"canonical_path": "/tmp/manifest", "kind": "file", "bytes": 1, "sha256": "b" * 64}


def _changed(left, right, prefix: str = "") -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        assert set(left) == set(right)
        output: set[str] = set()
        for key in left:
            output.update(_changed(left[key], right[key], f"{prefix}.{key}" if prefix else key))
        return output
    return {prefix} if left != right else set()


@pytest.mark.parametrize("seed", [44, 45])
def test_prompted_new_seed_changes_only_registered_rng_fields(seed: int) -> None:
    kwargs = {
        "arm": "document",
        "graph_signature": GRAPH,
        "graph_manifest_signature": MANIFEST,
        "graph_edges": 150_000_000,
        "retained_rows": RETAINED_ROWS,
    }
    baseline, _ = train_config(**kwargs, seed=42)
    treatment, _ = train_config(**kwargs, seed=seed)
    assert _changed(baseline, treatment) == {
        "paired_invariant.seed",
        "optimizer.seed",
        "optimizer.positive_rng_seed",
        "optimizer.negative_rng_seed",
        "execution.expected_pipeline_stamp.positive_rng_seed",
        "execution.expected_pipeline_stamp.negative_rng_seed",
    }


def test_round0160_registry_allows_only_seeds44_and45_on_r0115_graph() -> None:
    active = {"manifest": {"round_id": "0160"}}
    for seed in (44, 45):
        assert _training_seed(active, {"training_seed": seed}) == seed
        assert _graph_execution_round_id(active, {"graph_execution_round_id": "0115"}) == "0115"
    for value in (None, 42, 43, 46, True):
        with pytest.raises(Round0113Error, match="training seed"):
            _training_seed(active, {"training_seed": value})


def test_prompted_metric_view_uses_symmetric_purity_fidelity() -> None:
    panel = {"density": 0.23, "ffr": 0.64, "purity": {"k256": 1.1, "k1024": 0.9}}
    score = {"projections": {"matched": {"ffr": 0.58, "recall_at_10": 0.012}}}
    values = metric_view(panel=panel, native_score=score)
    assert set(values) == set(METRICS)
    assert values["purity_fidelity_k256"] == pytest.approx(1 / 1.1)
    assert values["purity_fidelity_k1024"] == pytest.approx(0.9)


def test_family_evidence_requires_all_four_seed_cells() -> None:
    cells = {
        seed: {"seed": seed, "decision_metrics": {metric: 0.1 + seed / 1000 for metric in METRICS}}
        for seed in SEEDS
    }
    evidence = build_family_evidence(cells)
    assert evidence["seeds"] == list(SEEDS)
    assert evidence["gate_registered"] is False
    with pytest.raises(Round0160Error, match="incomplete"):
        build_family_evidence({42: cells[42]})
