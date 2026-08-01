"""Frozen decision arithmetic for the R0134 density functional showdown."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


ROUND_ID = "0134"
PANEL_SCHEMA = "round0134-jina-functional-showdown-panel-v1"
DECISION_SCHEMA = "round0134-jina-functional-showdown-decision-v1"
CAPABILITY = "jina-density-functional-showdown-v1"

HISTORICAL_SEED42 = "historical_r0037_seed42"
HISTORICAL_SEED43 = "historical_r0038_seed43"
CURRENT_R0104_SEED42 = "current_r0104_fp16_seed42"
CURRENT_RAW_SEED42 = "current_r0115_raw_seed42"
CURRENT_RAW_SEED43 = "current_r0117_raw_seed43"
CELL_ORDER = (
    HISTORICAL_SEED42,
    HISTORICAL_SEED43,
    CURRENT_R0104_SEED42,
    CURRENT_RAW_SEED42,
    CURRENT_RAW_SEED43,
)

METRIC_ORDER = (
    "ffr",
    "purity_fidelity_k256",
    "purity_fidelity_k1024",
    "projection_ffr",
    "ood_recall_at_10",
)
COMPARISON_TOLERANCE = 1.0e-12


class Round0134Error(RuntimeError):
    """Raised when R0134 evidence or selector inputs are malformed."""


def purity_fidelity(ratio: float) -> float:
    """Convert a purity ratio into symmetric fidelity around the ideal 1.0.

    Raw purity is map-label agreement divided by high-dimensional agreement.
    Ratios above one are over-separation, not unbounded improvement.  This
    multiplicative score treats ``x`` and ``1/x`` equally and is maximized at
    one, while preserving the raw ratios in the panel receipt.
    """
    value = float(ratio)
    if not math.isfinite(value) or value <= 0.0:
        raise Round0134Error("purity ratio must be finite and positive")
    return math.exp(-abs(math.log(value)))


def decision_metrics(cell: Mapping[str, Any]) -> dict[str, float]:
    panel = cell.get("panel")
    projection = cell.get("projection")
    if not isinstance(panel, Mapping) or not isinstance(projection, Mapping):
        raise Round0134Error("functional cell is missing panel/projection metrics")
    purity = panel.get("purity")
    if not isinstance(purity, Mapping):
        raise Round0134Error("functional cell is missing purity metrics")
    values = {
        "ffr": float(panel["ffr"]),
        "purity_fidelity_k256": purity_fidelity(float(purity["k256"])),
        "purity_fidelity_k1024": purity_fidelity(float(purity["k1024"])),
        "projection_ffr": float(projection["ffr"]),
        "ood_recall_at_10": float(projection["recall_at_10"]),
    }
    if any(not math.isfinite(value) for value in values.values()):
        raise Round0134Error("functional decision metric is nonfinite")
    return values


def _mean_metrics(
    cells: Mapping[str, Mapping[str, Any]], keys: Sequence[str]
) -> dict[str, float]:
    if not keys:
        raise Round0134Error("functional aggregate cannot be empty")
    rows = [decision_metrics(cells[key]) for key in keys]
    return {
        metric: sum(row[metric] for row in rows) / len(rows)
        for metric in METRIC_ORDER
    }


def _contrast(
    *,
    name: str,
    cells: Mapping[str, Mapping[str, Any]],
    historical: Sequence[str],
    current: Sequence[str],
) -> dict[str, Any]:
    historical_metrics = _mean_metrics(cells, historical)
    current_metrics = _mean_metrics(cells, current)
    metrics: dict[str, Any] = {}
    for metric in METRIC_ORDER:
        delta = current_metrics[metric] - historical_metrics[metric]
        metrics[metric] = {
            "historical": historical_metrics[metric],
            "current": current_metrics[metric],
            "delta_current_minus_historical": delta,
            "current_at_least_historical": delta >= -COMPARISON_TOLERANCE,
        }
    return {
        "name": name,
        "historical_cells": list(historical),
        "current_cells": list(current),
        "aggregation": "arithmetic mean after per-cell metric computation",
        "metrics": metrics,
        "all_functional_metrics_current_at_least_historical": all(
            row["current_at_least_historical"] for row in metrics.values()
        ),
    }


def build_decision(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if tuple(cells) != CELL_ORDER:
        raise Round0134Error("functional cells are missing or reordered")
    contrasts = {
        "pre_r0115_seed42": _contrast(
            name="pre_r0115_seed42",
            cells=cells,
            historical=(HISTORICAL_SEED42,),
            current=(CURRENT_R0104_SEED42,),
        ),
        "raw_current_two_seed": _contrast(
            name="raw_current_two_seed",
            cells=cells,
            historical=(HISTORICAL_SEED42, HISTORICAL_SEED43),
            current=(CURRENT_RAW_SEED42, CURRENT_RAW_SEED43),
        ),
    }
    passed = all(
        contrast["all_functional_metrics_current_at_least_historical"]
        for contrast in contrasts.values()
    )
    failures = [
        f"{name}:{metric}"
        for name, contrast in contrasts.items()
        for metric, row in contrast["metrics"].items()
        if not row["current_at_least_historical"]
    ]
    return {
        "schema": DECISION_SCHEMA,
        "round_id": ROUND_ID,
        "selector": {
            "metrics": list(METRIC_ORDER),
            "purity_direction": (
                "exp(-abs(log(raw purity ratio))); one is ideal, so both "
                "under- and over-separation reduce fidelity"
            ),
            "point_comparison": "current >= historical",
            "absolute_tolerance": COMPARISON_TOLERANCE,
            "no_posthoc_margin": True,
        },
        "contrasts": contrasts,
        "failed_cells": failures,
        "outcome": (
            "current-recipe-functionally-noninferior"
            if passed
            else "historical-recipe-functionally-better"
        ),
        "density_v3_calibration_authorized": passed,
        "fuzzy_graph_or_sampler_bridges_authorized": not passed,
        "training_performed": False,
        "registered_density_floor_changed": False,
        "map_registry_state_changed": False,
    }
