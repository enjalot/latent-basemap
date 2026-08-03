"""Frozen four-seed native-prompted calibration design for Round 0160."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np


ROUND_ID = "0160"
CAPABILITY = "jina-fineweb-2m-prompted-seed42-45-family-v1"
ROWS = 1_993_761
DIMENSION = 768
SEEDS = (42, 43, 44, 45)
NEW_SEEDS = (44, 45)
METRICS = (
    "density_v2",
    "ffr",
    "purity_fidelity_k256",
    "purity_fidelity_k1024",
    "projection_ffr",
    "heldout_recall_at_10",
)


class Round0160Error(RuntimeError):
    """Raised when the R0160 prompted-seed contract changes."""


def purity_fidelity(value: Any) -> float:
    """Make purity ratios symmetric around the ideal value of one."""
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise Round0160Error("prompted purity ratio must be finite and positive")
    return math.exp(-abs(math.log(number)))


def metric_view(*, panel: Mapping[str, Any], native_score: Mapping[str, Any]) -> dict[str, float]:
    purity = panel.get("purity")
    projections = native_score.get("projections")
    matched = projections.get("matched") if isinstance(projections, Mapping) else None
    if not isinstance(purity, Mapping) or not isinstance(matched, Mapping):
        raise Round0160Error("prompted family cell lacks purity or held-out projection")
    values = {
        "density_v2": float(panel["density"]),
        "ffr": float(panel["ffr"]),
        "purity_fidelity_k256": purity_fidelity(purity["k256"]),
        "purity_fidelity_k1024": purity_fidelity(purity["k1024"]),
        "projection_ffr": float(matched["ffr"]),
        "heldout_recall_at_10": float(matched["recall_at_10"]),
    }
    if set(values) != set(METRICS) or not np.isfinite(tuple(values.values())).all():
        raise Round0160Error("prompted family decision metrics are incomplete")
    return values


def build_family_evidence(cells: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    if set(cells) != set(SEEDS):
        raise Round0160Error("prompted four-seed family is incomplete")
    output: dict[str, Any] = {}
    for seed in SEEDS:
        cell = cells[seed]
        metrics = cell.get("decision_metrics")
        if int(cell.get("seed", -1)) != seed or not isinstance(metrics, Mapping):
            raise Round0160Error(f"prompted seed-{seed} identity changed")
        if set(metrics) != set(METRICS):
            raise Round0160Error(f"prompted seed-{seed} metrics changed")
        output[f"seed{seed}"] = dict(cell)
    return {
        "schema": "round0160-prompted-four-seed-family-evidence-v1",
        "round_id": ROUND_ID,
        "capability": CAPABILITY,
        "population": {
            "rows": ROWS,
            "dimension": DIMENSION,
            "embedding_convention": "Document: ",
        },
        "seeds": list(SEEDS),
        "new_training_seeds": list(NEW_SEEDS),
        "metrics": list(METRICS),
        "cells": output,
        "gate_registered": False,
        "gate_method": "deferred to preregistered R0161 mean-minus-2-sample-sd",
        "raw_floor_changed": False,
        "training_performed": True,
    }
