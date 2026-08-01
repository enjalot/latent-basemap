"""Frozen design and selector for the R0140 Jina subsystem bisection.

All new cells train on the exact R0037 2M row universe.  The extra
current-graph/current-host cell is deliberate: R0037 and R0104/R0115 used
different row selections, so the campaign's original two-cell shorthand could
not distinguish a graph effect from a row-universe effect.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from .artifact_identity import canonical_json, sha256_bytes
from .round0104_training import train_config as r0104_train_config


ROUND_ID = "0140"
CAPABILITY = "jina-2m-subsystem-bisection-v1"
ROWS = 2_000_000
DIMENSION = 768
SEED = 42
GRAPH_K = 50
SUCCESSFUL_UPDATES = 500_000

CURRENT_GRAPH_CURRENT_HOST = "current_graph_current_host"
HISTORICAL_GRAPH_CURRENT_HOST = "historical_graph_current_host"
HISTORICAL_GRAPH_DEVICE_REPRO = "historical_graph_device_reproduction"
NEW_CELLS = (
    CURRENT_GRAPH_CURRENT_HOST,
    HISTORICAL_GRAPH_CURRENT_HOST,
    HISTORICAL_GRAPH_DEVICE_REPRO,
)

METRICS = (
    "ffr",
    "purity_fidelity_k256",
    "purity_fidelity_k1024",
    "projection_ffr",
    "ood_recall_at_10",
)

# Preregistered from the accepted R0134 seed-42/43 historical cells.  Each
# margin is max(the declared metric floor, 2x the observed seed movement).
HISTORICAL_SEED42 = {
    "ffr": 0.5698,
    "purity_fidelity_k256": 0.9298000929800094,
    "purity_fidelity_k1024": 0.9481,
    "projection_ffr": 0.5315,
    "ood_recall_at_10": 0.01021,
}
HISTORICAL_MARGINS = {
    "ffr": 0.01,
    "purity_fidelity_k256": 0.038438284667,
    "purity_fidelity_k1024": 0.0338,
    "projection_ffr": 0.015,
    "ood_recall_at_10": 0.0015,
}
HISTORICAL_FLOORS = {
    key: HISTORICAL_SEED42[key] - HISTORICAL_MARGINS[key] for key in METRICS
}

# These are contextual controls from different training-row universes.  The
# stronger seed-42 value is used per metric; neither is called a paired cell.
CURRENT_CONTEXT_FLOORS = {
    "ffr": 0.4457,
    "purity_fidelity_k256": 0.7966,
    "purity_fidelity_k1024": 0.6818,
    "projection_ffr": 0.4371,
    "ood_recall_at_10": 0.00946,
}
RESTORATION_FLOORS = {
    key: max(HISTORICAL_FLOORS[key], CURRENT_CONTEXT_FLOORS[key])
    for key in METRICS
}

GRAPH_NLIST = 8_192
GRAPH_TRAIN_ROWS = 262_144
GRAPH_TRAIN_SEED = 104
GRAPH_QUALITY_ROWS = 4_096
GRAPH_QUALITY_SEED = 105
GRAPH_NPROBE_GRID = (16, 32, 64, 128, 256)
GRAPH_MEAN_RECALL_FLOOR = 0.90
GRAPH_P10_RECALL_FLOOR = 0.80

TRAIN_MINIMUM_UPDATES_PER_S = 60.0
TRAIN_WARNING_UPDATES_PER_S = 75.0
PERFORMANCE_WARMUP_UPDATES = 200
PERFORMANCE_WINDOWS = 5


class Round0140Error(RuntimeError):
    """The registered R0140 bisection contract was violated."""


def historical_preprocessing_stamp() -> dict[str, Any]:
    body = {
        "schema": "round0140-historical-row-input-preprocessing-v1",
        "source_rows": [0, ROWS],
        "source_dimension": DIMENSION,
        "effective_dimension": DIMENSION,
        "compute_dtype": "<f4",
        "operation": "exact-r0037-fp16-to-device-fp32",
        "l2_renormalized_for_training": False,
        "row_universe": "R0037-jina-en-2M-nested-exact-order",
    }
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}


def host_train_config(
    *,
    cell: str,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    graph_edges: int,
) -> tuple[dict[str, Any], str]:
    if cell not in {CURRENT_GRAPH_CURRENT_HOST, HISTORICAL_GRAPH_CURRENT_HOST}:
        raise Round0140Error(f"not a host cell: {cell}")
    config, _ = r0104_train_config(
        "fp16_control",
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        graph_edges=graph_edges,
    )
    config = copy.deepcopy(config)
    stamp = historical_preprocessing_stamp()
    config.update({
        "schema": "round0140-fixed-row-host-train-config-v1",
        "arm": cell,
        "causal_matrix": {
            "row_universe": "historical-r0037-exact",
            "graph_subsystem": (
                "current-r0104-style" if cell == CURRENT_GRAPH_CURRENT_HOST
                else "historical-r0037-byte-exact"
            ),
            "trainer_subsystem": "current-r0104-host",
        },
        "input_preprocessing": stamp,
    })
    config["paired_invariant"]["sampler"] = "PairedHostWeightedJinaSampler"
    config["execution"]["minimum_train_upd_s"] = TRAIN_MINIMUM_UPDATES_PER_S
    config["execution"]["warning_train_upd_s"] = TRAIN_WARNING_UPDATES_PER_S
    config["execution"]["performance_windows"] = PERFORMANCE_WINDOWS
    config["execution"]["expected_pipeline_stamp"].update({
        "source_representation": "fp16-control",
        "row_universe": "R0037-jina-en-2M-nested-exact-order",
        "source_sha256": (
            "7941f827eac6ac38ad45301198dc238a9fd7bbe16204c36a4031ce63a4115007"
        ),
    })
    return config, sha256_bytes(canonical_json(config))


def metric_view(cell: Mapping[str, Any]) -> dict[str, float]:
    panel = cell.get("panel")
    projection = cell.get("projection")
    if not isinstance(panel, Mapping) or not isinstance(projection, Mapping):
        raise Round0140Error("functional cell lacks panel/projection")
    purity = panel.get("purity")
    if not isinstance(purity, Mapping):
        raise Round0140Error("functional cell lacks purity")

    def fidelity(value: Any) -> float:
        number = float(value)
        if not np.isfinite(number) or number <= 0:
            raise Round0140Error("purity ratio must be finite and positive")
        return math.exp(-abs(math.log(number)))

    values = {
        "ffr": float(panel["ffr"]),
        "purity_fidelity_k256": fidelity(purity["k256"]),
        "purity_fidelity_k1024": fidelity(purity["k1024"]),
        "projection_ffr": float(projection["ffr"]),
        "ood_recall_at_10": float(projection["recall_at_10"]),
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise Round0140Error("functional metrics must be finite")
    return values


def _floor_test(values: Mapping[str, float]) -> dict[str, Any]:
    cells = {
        key: {
            "observed": float(values[key]),
            "floor": RESTORATION_FLOORS[key],
            "passed": float(values[key]) >= RESTORATION_FLOORS[key],
        }
        for key in METRICS
    }
    return {"metrics": cells, "passed_all": all(v["passed"] for v in cells.values())}


def _paired_delta(
    control: Mapping[str, float], treatment: Mapping[str, float]
) -> dict[str, float]:
    return {key: float(treatment[key]) - float(control[key]) for key in METRICS}


def build_decision(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if set(cells) != set(NEW_CELLS):
        raise Round0140Error("R0140 decision cells are missing or unexpected")
    values = {key: metric_view(cells[key]) for key in NEW_CELLS}
    restoration = {key: _floor_test(values[key]) for key in NEW_CELLS}
    current_restores = restoration[CURRENT_GRAPH_CURRENT_HOST]["passed_all"]
    graph_swap_restores = restoration[HISTORICAL_GRAPH_CURRENT_HOST]["passed_all"]
    reproduction_restores = restoration[HISTORICAL_GRAPH_DEVICE_REPRO]["passed_all"]

    if not reproduction_restores:
        outcome = "historical-recipe-not-reproduced-current-release"
        next_action = "audit-reproduction-before-causal-claim"
    elif graph_swap_restores and not current_restores:
        outcome = "historical-graph-subsystem-restores"
        next_action = "validate-winning-graph-subsystem-at-12.5m"
    elif current_restores and graph_swap_restores:
        outcome = "historical-row-universe-restores-with-current-trainer"
        next_action = "recover-and-test-row-policy-on-current-population"
    elif not current_restores and not graph_swap_restores:
        outcome = "current-host-trainer-subsystem-does-not-restore"
        next_action = "validate-historical-trainer-subsystem-before-scale"
    else:
        outcome = "subsystem-interaction-unresolved"
        next_action = "issue-one-cell-current-graph-historical-device-interaction"

    return {
        "schema": "round0140-jina-subsystem-bisection-decision-v1",
        "round_id": ROUND_ID,
        "selector": {
            "metrics": list(METRICS),
            "historical_seed42": HISTORICAL_SEED42,
            "historical_margins": HISTORICAL_MARGINS,
            "historical_floors": HISTORICAL_FLOORS,
            "current_context_floors": CURRENT_CONTEXT_FLOORS,
            "restoration_floors": RESTORATION_FLOORS,
            "all_metrics_required": True,
            "density_diagnostic_only": True,
        },
        "metrics": values,
        "restoration": restoration,
        "paired_graph_delta_historical_minus_current": _paired_delta(
            values[CURRENT_GRAPH_CURRENT_HOST],
            values[HISTORICAL_GRAPH_CURRENT_HOST],
        ),
        "paired_trainer_delta_device_minus_host_on_historical_graph": _paired_delta(
            values[HISTORICAL_GRAPH_CURRENT_HOST],
            values[HISTORICAL_GRAPH_DEVICE_REPRO],
        ),
        "outcome": outcome,
        "next_action": next_action,
        "historical_reproduction_restores": reproduction_restores,
        "historical_graph_current_host_restores": graph_swap_restores,
        "current_graph_current_host_restores": current_restores,
        "row_universe_fixed_across_new_cells": True,
        "cross_round_training_row_equivalence_claimed": False,
        "registered_density_floor_changed": False,
        "map_registry_state_changed": False,
    }
