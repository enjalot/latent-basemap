"""CUDA-hidden tests for the R0154 raw seed replay."""
from __future__ import annotations

import pytest

from basemap.round0140_subsystem_bisection import (
    CURRENT_GRAPH_CURRENT_HOST,
    METRICS,
)
from basemap.round0154_seed_variance import (
    CAPABILITY,
    SEEDS,
    Round0154Error,
    build_seed_evidence,
    raw_seed_train_config,
)


GRAPH = {"canonical_path": "/tmp/graph", "kind": "file", "bytes": 1, "sha256": "a" * 64}
MANIFEST = {"canonical_path": "/tmp/manifest", "kind": "file", "bytes": 1, "sha256": "b" * 64}


def test_seed_configs_change_only_registered_seed_fields() -> None:
    first, first_sha = raw_seed_train_config(
        seed=44,
        graph_signature=GRAPH,
        graph_manifest_signature=MANIFEST,
        graph_edges=151_013_146,
    )
    second, second_sha = raw_seed_train_config(
        seed=45,
        graph_signature=GRAPH,
        graph_manifest_signature=MANIFEST,
        graph_edges=151_013_146,
    )
    assert first_sha != second_sha
    assert first["paired_invariant"]["seed"] == first["optimizer"]["seed"] == 44
    assert second["paired_invariant"]["seed"] == second["optimizer"]["seed"] == 45
    assert first["execution"] == second["execution"]
    assert first["graph"] == second["graph"]
    assert first["model"] == second["model"]


@pytest.mark.parametrize("seed", [42, 43, 46])
def test_seed_config_rejects_unregistered_seed(seed: int) -> None:
    with pytest.raises(Round0154Error, match="seed or cell"):
        raw_seed_train_config(
            seed=seed,
            graph_signature=GRAPH,
            graph_manifest_signature=MANIFEST,
            graph_edges=151_013_146,
        )


def _panel(seed: int, offset: float) -> dict:
    decision = {
        "ffr": 0.57 + offset,
        "purity_fidelity_k256": 0.91 + offset,
        "purity_fidelity_k1024": 0.97 + offset,
        "projection_ffr": 0.53 + offset,
        "ood_recall_at_10": 0.0098 + offset,
    }
    assert set(decision) == set(METRICS)
    return {
        "round_id": "0154",
        "cells": {
            CURRENT_GRAPH_CURRENT_HOST: {
                "seed": seed,
                "decision_metrics": decision,
                "panel": {
                    "density": 0.56 + offset,
                    "ffr": decision["ffr"],
                    "purity": {
                        "k256": decision["purity_fidelity_k256"],
                        "k1024": decision["purity_fidelity_k1024"],
                    },
                },
                "projection": {
                    "ffr": decision["projection_ffr"],
                    "recall_at_10": decision["ood_recall_at_10"],
                },
            }
        },
    }


def test_seed_evidence_closes_both_cells_without_proposing_margin() -> None:
    panels = {44: _panel(44, 0.0), 45: _panel(45, 0.001)}
    densities = {
        f"seed{seed}": {
            "seed": seed,
            "density_v2": {"correlation": 0.18 + 0.001 * (seed - 44)},
            "clears_registered_floor": True,
        }
        for seed in SEEDS
    }
    result = build_seed_evidence(panels, densities)
    assert result["capability"] == CAPABILITY
    assert result["margin_or_floor_proposed"] is False
    assert result["floor_changed"] is False
    assert result["seed45_minus_seed44"]["density_v2"] == pytest.approx(0.001)


def test_seed_evidence_rejects_incomplete_matrix() -> None:
    with pytest.raises(Round0154Error, match="incomplete"):
        build_seed_evidence({}, {})
