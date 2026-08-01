"""Seed-43 replication for the raw-versus-drop-only historical-row contrast.

R0149 supplies the seed-42 contrast.  This module freezes a second, paired
seed over the same two row populations and graphs.  It can authorize a scale
candidate only when both arms restore at seed 43 and the drop-only arm also
restored at seed 42; a discordant seed remains explicitly inconclusive.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from .artifact_identity import canonical_json, sha256_bytes
from .round0140_subsystem_bisection import (
    CURRENT_GRAPH_CURRENT_HOST,
    METRICS,
    RESTORATION_FLOORS,
    host_train_config,
    metric_view,
)
from .round0149_drop_only import (
    CAPABILITY as R0149_CAPABILITY,
    TREATMENT as DROP_ONLY,
    treatment_train_config as r0149_train_config,
)


ROUND_ID = "0150"
CAPABILITY = "jina-2m-drop-only-seed-replication-v1"
SEED = 43
RAW = CURRENT_GRAPH_CURRENT_HOST
CELLS = (RAW, DROP_ONLY)


class Round0150Error(RuntimeError):
    """The registered seed replay or its evidence is malformed."""


def _retag_seed(config: Mapping[str, Any], *, schema: str) -> tuple[dict[str, Any], str]:
    value = copy.deepcopy(dict(config))
    paired = value.get("paired_invariant")
    optimizer = value.get("optimizer")
    if (
        not isinstance(paired, dict)
        or not isinstance(optimizer, dict)
        or paired.get("seed") != 42
        or optimizer.get("seed") != 42
    ):
        raise Round0150Error("parent training config seed changed")
    paired["seed"] = SEED
    optimizer["seed"] = SEED
    value["schema"] = schema
    causal = value.get("causal_matrix")
    if not isinstance(causal, dict):
        raise Round0150Error("parent training config causal matrix changed")
    causal["replication_seed"] = SEED
    causal["graph_reused_byte_exact"] = True
    return value, sha256_bytes(canonical_json(value))


def raw_seed43_train_config(
    *,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    graph_edges: int,
) -> tuple[dict[str, Any], str]:
    config, _digest = host_train_config(
        cell=RAW,
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        graph_edges=graph_edges,
    )
    return _retag_seed(config, schema="round0150-raw-historical-seed43-train-v1")


def drop_seed43_train_config(
    *,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    graph_edges: int,
    source_sha256: str,
    selection_sha256: str,
) -> tuple[dict[str, Any], str]:
    config, _digest = r0149_train_config(
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        graph_edges=graph_edges,
        source_sha256=source_sha256,
        selection_sha256=selection_sha256,
    )
    return _retag_seed(config, schema="round0150-drop-only-seed43-train-v1")


def _floor_test(values: Mapping[str, float]) -> dict[str, Any]:
    metrics = {
        key: {
            "observed": float(values[key]),
            "floor": float(RESTORATION_FLOORS[key]),
            "passed": float(values[key]) >= float(RESTORATION_FLOORS[key]),
        }
        for key in METRICS
    }
    return {
        "metrics": metrics,
        "passed_all": all(item["passed"] for item in metrics.values()),
    }


def build_decision(
    r0149_decision: Mapping[str, Any],
    seed43_cells: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Select only a replicated restoration, replicated failure, or no claim."""
    if set(seed43_cells) != set(CELLS):
        raise Round0150Error("seed-43 replay cells are missing or unexpected")
    if any(seed43_cells[key].get("seed") != SEED for key in CELLS):
        raise Round0150Error("seed-43 replay cell seed changed")
    if (
        r0149_decision.get("round_id") != "0149"
        or r0149_decision.get("capability") != R0149_CAPABILITY
        or r0149_decision.get("outcome")
        not in {
            "drop-only-historical-row-policy-restores",
            "drop-only-historical-row-policy-does-not-restore",
        }
    ):
        raise Round0150Error("accepted R0149 activation changed")

    seed42_values = r0149_decision.get("metrics")
    seed42_tests = r0149_decision.get("restoration")
    if (
        not isinstance(seed42_values, Mapping)
        or not isinstance(seed42_tests, Mapping)
        or set(seed42_values) != {
            RAW,
            "eligible_historical_current_graph_current_host",
            DROP_ONLY,
        }
        or seed42_tests.get(RAW, {}).get("passed_all") is not True
    ):
        raise Round0150Error("R0149 seed-42 control evidence changed")
    seed42_drop_passed = seed42_tests.get(DROP_ONLY, {}).get("passed_all") is True
    expected_r0149_outcome = (
        "drop-only-historical-row-policy-restores"
        if seed42_drop_passed
        else "drop-only-historical-row-policy-does-not-restore"
    )
    if r0149_decision.get("outcome") != expected_r0149_outcome:
        raise Round0150Error("R0149 outcome does not match its selector cells")

    seed43_values = {key: metric_view(seed43_cells[key]) for key in CELLS}
    seed43_tests = {key: _floor_test(seed43_values[key]) for key in CELLS}
    raw_seed43_passed = seed43_tests[RAW]["passed_all"]
    drop_seed43_passed = seed43_tests[DROP_ONLY]["passed_all"]

    if raw_seed43_passed and seed42_drop_passed and drop_seed43_passed:
        outcome = "drop-only-restoration-replicates-across-seeds"
        next_action = "preregister-12.5m-drop-only-rescue-rung"
        scale_candidate_released = True
    elif raw_seed43_passed and not seed42_drop_passed and not drop_seed43_passed:
        outcome = "drop-only-restoration-fails-across-seeds"
        next_action = "do-not-scale-drop-only-close-row-removal-path"
        scale_candidate_released = False
    else:
        outcome = "drop-only-restoration-is-seed-sensitive-or-control-inconclusive"
        next_action = "no-scale-transfer-from-row-policy-seed-replay"
        scale_candidate_released = False

    seed42_metric_values = {
        key: {metric: float(seed42_values[key][metric]) for metric in METRICS}
        for key in (RAW, DROP_ONLY)
    }
    return {
        "schema": "round0150-drop-only-seed-replication-decision-v1",
        "round_id": ROUND_ID,
        "capability": CAPABILITY,
        "selector": {
            "metrics": list(METRICS),
            "restoration_floors": dict(RESTORATION_FLOORS),
            "all_metrics_required_per_arm_and_seed": True,
            "scale_candidate_requires_raw_seed43_and_drop_only_both_seeds": True,
            "density_diagnostic_only": True,
            "ood_diagnostic_only": True,
        },
        "r0149_outcome": r0149_decision["outcome"],
        "metrics": {"seed42": seed42_metric_values, "seed43": seed43_values},
        "restoration": {
            "seed42": {
                RAW: dict(seed42_tests[RAW]),
                DROP_ONLY: dict(seed42_tests[DROP_ONLY]),
            },
            "seed43": seed43_tests,
        },
        "drop_only_minus_raw": {
            "seed42": {
                metric: seed42_metric_values[DROP_ONLY][metric]
                - seed42_metric_values[RAW][metric]
                for metric in METRICS
            },
            "seed43": {
                metric: seed43_values[DROP_ONLY][metric]
                - seed43_values[RAW][metric]
                for metric in METRICS
            },
        },
        "outcome": outcome,
        "next_action": next_action,
        "drop_only_scale_candidate_released": scale_candidate_released,
        "unique_causal_factor_claimed": False,
        "density_floor_changed": False,
        "map_registry_state_changed": False,
        "production_or_publishing_claimed": False,
    }
