"""Frozen candidate selector for the conditional historical-row-policy bridge.

This module does not authorize execution.  It prepares the narrow comparison
that is relevant only if accepted R0140 evidence says the exact historical row
universe restores function with the current graph and current host trainer.
The treatment retains historical shuffle order and corpus composition while
removing R0087-ineligible exact copies with size-preserving replacement.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .round0140_subsystem_bisection import (
    CURRENT_GRAPH_CURRENT_HOST,
    METRICS,
    RESTORATION_FLOORS,
    metric_view,
)


ROUND_ID = "0147"
CAPABILITY = "jina-2m-historical-row-policy-duplicate-control-v1"
TREATMENT = "eligible_historical_current_graph_current_host"
ROWS = 2_000_000


class Round0147Error(RuntimeError):
    """The conditional row-policy contrast is malformed."""


def _floor_test(values: Mapping[str, float]) -> dict[str, Any]:
    metrics = {
        key: {
            "observed": float(values[key]),
            "floor": float(RESTORATION_FLOORS[key]),
            "passed": float(values[key]) >= RESTORATION_FLOORS[key],
        }
        for key in METRICS
    }
    return {
        "metrics": metrics,
        "passed_all": all(item["passed"] for item in metrics.values()),
    }


def _selection_guard(summary: Mapping[str, Any]) -> dict[str, int]:
    fields = {
        key: int(summary.get(key, -1))
        for key in (
            "target_rows",
            "historical_stream_rows",
            "scan_rows",
            "skipped_excluded_rows",
            "raw_prefix_excluded_rows",
            "replacement_rows_beyond_raw_prefix",
        )
    }
    if (
        fields["target_rows"] != ROWS
        or fields["historical_stream_rows"] < ROWS
        or fields["scan_rows"] != ROWS + fields["skipped_excluded_rows"]
        or fields["raw_prefix_excluded_rows"] <= 0
        or fields["replacement_rows_beyond_raw_prefix"]
        != fields["raw_prefix_excluded_rows"]
        or fields["skipped_excluded_rows"]
        < fields["raw_prefix_excluded_rows"]
    ):
        raise Round0147Error("historical eligibility selection does not close")
    return fields


def build_decision(
    cells: Mapping[str, Mapping[str, Any]],
    *,
    selection_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Decide whether exact-copy control preserves the R0140 restoration.

    The control is the accepted R0140 current-graph/current-host map on the raw
    historical 2M rows.  The only new trained cell uses the same graph/trainer
    semantics on a size-preserving eligible selection.  Passing establishes
    compatibility, not that duplicate control caused (or cured) the original
    regression.
    """
    if set(cells) != {CURRENT_GRAPH_CURRENT_HOST, TREATMENT}:
        raise Round0147Error("R0147 decision cells are missing or unexpected")
    selection = _selection_guard(selection_summary)
    values = {key: metric_view(cells[key]) for key in cells}
    if not all(
        np.isfinite(value)
        for metrics in values.values()
        for value in metrics.values()
    ):
        raise Round0147Error("R0147 metrics must be finite")
    restoration = {key: _floor_test(values[key]) for key in cells}
    if not restoration[CURRENT_GRAPH_CURRENT_HOST]["passed_all"]:
        raise Round0147Error(
            "R0147 activation requires an accepted restoring R0140 control"
        )
    treatment_restores = restoration[TREATMENT]["passed_all"]
    if treatment_restores:
        outcome = "eligible-historical-row-policy-restores"
        next_action = (
            "define-and-preregister-a-diverse-scale-analogue-before-rescue"
        )
    else:
        outcome = "eligible-historical-row-policy-does-not-restore"
        next_action = "decompose-exclusion-and-replacement-policy-before-scale"
    return {
        "schema": "round0147-historical-row-policy-decision-v1",
        "round_id": ROUND_ID,
        "capability": CAPABILITY,
        "selector": {
            "metrics": list(METRICS),
            "restoration_floors": RESTORATION_FLOORS,
            "all_metrics_required": True,
            "density_diagnostic_only": True,
            "selection": selection,
        },
        "metrics": values,
        "restoration": restoration,
        "paired_eligible_minus_raw_historical": {
            key: values[TREATMENT][key]
            - values[CURRENT_GRAPH_CURRENT_HOST][key]
            for key in METRICS
        },
        "outcome": outcome,
        "next_action": next_action,
        "duplicate_control_compatible_with_restoration": treatment_restores,
        "duplicate_control_causal_claimed": False,
        "diverse_scale_transfer_claimed": False,
        "registered_density_floor_changed": False,
        "map_registry_state_changed": False,
    }
