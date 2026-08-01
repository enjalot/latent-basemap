"""Decision contract for the R0137 high-recall fuzzy-graph bridge."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .round0134_functional_showdown import (
    COMPARISON_TOLERANCE,
    METRIC_ORDER,
    decision_metrics,
)


ROUND_ID = "0137"
PANEL_SCHEMA = "round0137-high-recall-graph-functional-panel-v1"
DECISION_SCHEMA = "round0137-high-recall-graph-bridge-decision-v1"
CAPABILITY = "jina-current-2m-high-recall-graph-bridge-v1"

HISTORICAL = "historical_r0037_seed42"
CONTROL = "current_r0104_fp16_seed42"
TREATMENT = "current_high_recall_graph_seed42"
CELL_ORDER = (HISTORICAL, CONTROL, TREATMENT)


class Round0137Error(RuntimeError):
    """The R0137 graph-bridge contract was violated."""


def _contrast(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    baseline = decision_metrics(left)
    candidate = decision_metrics(right)
    metrics: dict[str, Any] = {}
    for metric in METRIC_ORDER:
        delta = candidate[metric] - baseline[metric]
        metrics[metric] = {
            "baseline": baseline[metric],
            "candidate": candidate[metric],
            "delta_candidate_minus_baseline": delta,
            "candidate_at_least_baseline": delta >= -COMPARISON_TOLERANCE,
        }
    return {
        "metrics": metrics,
        "candidate_at_least_baseline_on_all_metrics": all(
            row["candidate_at_least_baseline"] for row in metrics.values()
        ),
    }


def build_decision(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if tuple(cells) != CELL_ORDER:
        raise Round0137Error("graph-bridge functional cells are reordered")
    versus_control = _contrast(cells[CONTROL], cells[TREATMENT])
    versus_historical = _contrast(cells[HISTORICAL], cells[TREATMENT])
    restores_historical = versus_historical[
        "candidate_at_least_baseline_on_all_metrics"
    ]
    preserves_control = versus_control[
        "candidate_at_least_baseline_on_all_metrics"
    ]
    restores = restores_historical and preserves_control
    regresses_control = not versus_control[
        "candidate_at_least_baseline_on_all_metrics"
    ]
    if restores:
        outcome = "high-recall-graph-sufficient-to-restore-function"
    elif regresses_control:
        outcome = "high-recall-graph-regresses-current-control"
    else:
        outcome = "high-recall-graph-insufficient-to-restore-function"
    return {
        "schema": DECISION_SCHEMA,
        "round_id": ROUND_ID,
        "selector": {
            "metrics": list(METRIC_ORDER),
            "point_comparison": "candidate >= baseline",
            "absolute_tolerance": COMPARISON_TOLERANCE,
            "restoration_requires_all_metrics_vs_historical_and_control": True,
            "no_density_floor_or_posthoc_margin": True,
        },
        "treatment_vs_current_control": versus_control,
        "treatment_vs_historical_target": versus_historical,
        "outcome": outcome,
        "high_recall_graph_sufficient": restores,
        "restores_historical_on_all_metrics": restores_historical,
        "preserves_current_control_on_all_metrics": preserves_control,
        "sampler_bridge_authorized": not restores,
        "training_performed": True,
        "registered_density_floor_changed": False,
        "map_registry_state_changed": False,
    }
