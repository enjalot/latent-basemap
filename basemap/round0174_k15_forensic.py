"""Frozen design for the R0174 historical-row fuzzy-k15 forensic."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np

from .artifact_identity import canonical_json, sha256_bytes
from .round0140_subsystem_bisection import (
    CURRENT_GRAPH_CURRENT_HOST as R0140_K50_CELL,
    RESTORATION_FLOORS,
    host_train_config as r0140_host_train_config,
    metric_view,
)


ROUND_ID = "0174"
CAPABILITY = "jina-historical-rows-current-trainer-k15-forensic-v1"
CELL = "historical_rows_current_graph_k15_current_host"
ROWS = 2_000_000
DIMENSION = 768
SEED = 42
GRAPH_K = 15
SUCCESSFUL_UPDATES = 500_000


class Round0174Error(RuntimeError):
    """The registered R0174 graph-degree forensic contract was violated."""


def host_train_config(
    *,
    cell: str,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    graph_edges: int,
) -> tuple[dict[str, Any], str]:
    """Retarget the reviewed R0140 current-host recipe to only fuzzy k."""
    if cell != CELL:
        raise Round0174Error(f"unknown R0174 cell: {cell!r}")
    config, _ = r0140_host_train_config(
        cell=R0140_K50_CELL,
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        graph_edges=graph_edges,
    )
    config = copy.deepcopy(config)
    config.update({
        "schema": "round0174-historical-row-k15-host-train-config-v1",
        "arm": CELL,
        "causal_matrix": {
            "row_universe": "historical-r0037-exact",
            "graph_subsystem": "current-ivf-fuzzy-k15",
            "trainer_subsystem": "current-r0104-host",
            "only_change_from_r0140_current_graph_current_host": "graph_k_50_to_15",
        },
    })
    config["graph"]["k"] = GRAPH_K
    expected = config["execution"]["expected_pipeline_stamp"]
    expected.update({
        "positive_destination_policy": "queue-local-fp16-fuzzy-k15",
        "graph_degree": "variable-fuzzy-k15-edge-universe",
    })
    config["execution"]["graph_degree_treatment"] = {
        "control_k": 50,
        "treatment_k": GRAPH_K,
        "dose_rule_changed": False,
    }
    return config, sha256_bytes(canonical_json(config))


def build_decision(
    *,
    treatment: Mapping[str, Any],
    k50_control: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify whether fuzzy k15 alone breaks R0140 restoration."""
    observed = metric_view(treatment)
    baseline = metric_view(k50_control)
    if set(observed) != set(RESTORATION_FLOORS) or set(baseline) != set(
        RESTORATION_FLOORS
    ):
        raise Round0174Error("R0174 metric set changed")
    gates: dict[str, Any] = {}
    for metric, floor in RESTORATION_FLOORS.items():
        value = float(observed[metric])
        control = float(baseline[metric])
        if not np.isfinite(value) or not np.isfinite(control):
            raise Round0174Error("R0174 decision contains nonfinite metrics")
        gates[metric] = {
            "observed_k15": value,
            "r0140_k50_control": control,
            "paired_delta_k15_minus_k50": value - control,
            "restoration_floor": float(floor),
            "passed": value >= float(floor),
        }
    restores = all(cell["passed"] for cell in gates.values())
    return {
        "schema": "round0174-k15-forensic-decision-v1",
        "round_id": ROUND_ID,
        "cell": CELL,
        "registered_gates": gates,
        "passed_all_restoration_floors": restores,
        "outcome": (
            "k15-maintains-restoration-on-historical-rows"
            if restores
            else "k15-breaks-restoration-on-historical-rows"
        ),
        "interpretation": (
            "At 2M on historical rows, fuzzy graph degree k15 alone does not "
            "reproduce the prior scale-rung functional failure. This does not "
            "rule out a graph-degree-by-scale interaction."
            if restores
            else
            "At 2M on historical rows, changing only fuzzy graph degree from "
            "k50 to k15 is sufficient to break at least one registered "
            "restoration gate. This supports graph degree as a causal "
            "candidate but does not establish scale transfer."
        ),
        "density_role": "diagnostic-only; no density floor is changed",
        "production_or_publishing": False,
    }


__all__ = [
    "CAPABILITY",
    "CELL",
    "DIMENSION",
    "GRAPH_K",
    "ROUND_ID",
    "ROWS",
    "Round0174Error",
    "build_decision",
    "host_train_config",
]
