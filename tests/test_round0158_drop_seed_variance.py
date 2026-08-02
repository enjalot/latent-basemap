"""CUDA-hidden contract tests for R0158 drop-only seed calibration."""
from __future__ import annotations

import copy

import pytest

from basemap.round0140_subsystem_bisection import METRICS
from basemap.round0149_drop_only import TREATMENT
from basemap.round0158_drop_seed_variance import (
    CAPABILITY,
    ROUND_ID,
    SEEDS,
    Round0158Error,
    build_seed_evidence,
    drop_seed_train_config,
)


def _signature(name: str) -> dict[str, object]:
    return {
        "canonical_path": f"/tmp/{name}",
        "kind": "file",
        "bytes": 10,
        "sha256": name[0] * 64,
    }


def test_drop_config_changes_only_registered_seed() -> None:
    kwargs = {
        "graph_signature": _signature("graph"),
        "graph_manifest_signature": _signature("manifest"),
        "graph_edges": 100,
        "source_sha256": "s" * 64,
        "selection_sha256": "x" * 64,
    }
    first, _ = drop_seed_train_config(seed=44, **kwargs)
    second, _ = drop_seed_train_config(seed=45, **kwargs)
    assert first["paired_invariant"]["seed"] == 44
    assert second["paired_invariant"]["seed"] == 45
    normalized_first = copy.deepcopy(first)
    normalized_second = copy.deepcopy(second)
    for value in (normalized_first, normalized_second):
        value["paired_invariant"]["seed"] = 0
        value["optimizer"]["seed"] = 0
        value["causal_matrix"]["replication_seed"] = 0
        value["schema"] = "normalized"
    assert normalized_first == normalized_second
    with pytest.raises(Round0158Error, match="seed changed"):
        drop_seed_train_config(seed=46, **kwargs)


def _panel(seed: int, base: float) -> dict[str, object]:
    metrics = {key: base + index / 100 for index, key in enumerate(METRICS)}
    return {
        "round_id": ROUND_ID,
        "cells": {
            TREATMENT: {
                "seed": seed,
                "panel": {
                    "ffr": metrics["ffr"],
                    "purity": {
                        "k256": metrics["purity_fidelity_k256"],
                        "k1024": metrics["purity_fidelity_k1024"],
                    },
                    "density": 0.2,
                },
                    "projection": {
                        "ffr": metrics["projection_ffr"],
                        "recall_at_10": metrics["ood_recall_at_10"],
                },
            }
        },
    }


def test_evidence_is_measurement_only() -> None:
    panels = {44: _panel(44, 0.50), 45: _panel(45, 0.51)}
    density = {
        f"seed{seed}": {
            "seed": seed,
            "density_v2": {"correlation": 0.18 + (seed - 44) / 100},
            "clears_registered_floor": True,
        }
        for seed in SEEDS
    }
    evidence = build_seed_evidence(panels, density)
    assert evidence["capability"] == CAPABILITY
    assert evidence["margin_or_floor_proposed"] is False
    assert evidence["floor_changed"] is False
    assert evidence["seed45_minus_seed44"]["ffr"] == pytest.approx(0.01)
