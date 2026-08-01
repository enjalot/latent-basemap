"""Frozen drop-only decomposition of the R0147 historical row-policy result.

R0147 removed ineligible exact-family members and replaced them from beyond
the historical 2M prefix.  This round removes the same prefix members without
replacement.  It therefore adds one trained map, but it is still a bundled
population/cardinality/induced-graph contrast and cannot identify a unique
causal mechanism.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np

from .artifact_identity import canonical_json, ordered_array_sha256, sha256_bytes
from .round0104_training import negative_sampling_stamp
from .round0140_subsystem_bisection import (
    CURRENT_GRAPH_CURRENT_HOST,
    METRICS,
    RESTORATION_FLOORS,
    host_train_config,
    metric_view,
)
from .round0147_row_policy import TREATMENT as SIZE_PRESERVING_TREATMENT


ROUND_ID = "0149"
CAPABILITY = "jina-2m-historical-drop-only-decomposition-v1"
TREATMENT = "drop_only_historical_current_graph_current_host"
RAW_PREFIX_ROWS = 2_000_000
RAW_PREFIX_EXCLUDED_ROWS = 10_367
ROWS = RAW_PREFIX_ROWS - RAW_PREFIX_EXCLUDED_ROWS
DIMENSION = 768
ROW_UNIVERSE = (
    "r0037-historical-2m-prefix-after-r0087-exact-family-exclusion-no-replacement"
)
CELLS = (
    CURRENT_GRAPH_CURRENT_HOST,
    SIZE_PRESERVING_TREATMENT,
    TREATMENT,
)


class Round0149Error(RuntimeError):
    """The registered drop-only decomposition is malformed."""


def derive_drop_only_selection(
    arrays: Mapping[str, np.ndarray],
    *,
    parent_summary: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Take exactly the eligible members of the raw historical 2M prefix."""
    required = {
        "historical_positions",
        "pre_shuffle_positions",
        "corpus_ids",
        "dataset_rows",
        "global_rows",
    }
    if set(arrays) != required:
        raise Round0149Error("R0147 parent selection arrays changed")
    values = {key: np.asarray(arrays[key]) for key in sorted(required)}
    if any(len(value) != RAW_PREFIX_ROWS or value.ndim != 1 for value in values.values()):
        raise Round0149Error("R0147 parent selection geometry changed")
    historical = values["historical_positions"]
    if historical.dtype != np.dtype("int64") or np.any(historical[1:] <= historical[:-1]):
        raise Round0149Error("R0147 historical positions are not strictly ordered")
    keep = historical < RAW_PREFIX_ROWS
    kept_indices = np.flatnonzero(keep)
    if (
        int(parent_summary.get("target_rows", -1)) != RAW_PREFIX_ROWS
        or int(parent_summary.get("raw_prefix_excluded_rows", -1))
        != RAW_PREFIX_EXCLUDED_ROWS
        or len(kept_indices) != ROWS
        or not np.array_equal(kept_indices, np.arange(ROWS, dtype=np.int64))
        or historical[0] != 0
        or historical[ROWS - 1] >= RAW_PREFIX_ROWS
        or historical[ROWS] < RAW_PREFIX_ROWS
    ):
        raise Round0149Error("drop-only prefix closure failed")
    selected = {
        key: np.ascontiguousarray(value[:ROWS]) for key, value in values.items()
    }
    summary = {
        "target_rows": ROWS,
        "raw_prefix_rows": RAW_PREFIX_ROWS,
        "raw_prefix_excluded_rows": RAW_PREFIX_EXCLUDED_ROWS,
        "eligible_rows_retained": ROWS,
        "replacement_rows_beyond_raw_prefix": 0,
        "historical_position_start": int(historical[0]),
        "historical_position_stop_exclusive": RAW_PREFIX_ROWS,
        "last_retained_historical_position": int(historical[ROWS - 1]),
        "first_parent_replacement_historical_position": int(historical[ROWS]),
        "historical_order_preserved": True,
        "size_preserving": False,
        "parent_selection_target_rows": int(parent_summary["target_rows"]),
        "array_sha256": {
            key: ordered_array_sha256(value) for key, value in selected.items()
        },
    }
    return selected, summary


def treatment_preprocessing_stamp(
    *, source_sha256: str, selection_sha256: str
) -> dict[str, Any]:
    body = {
        "schema": "round0149-drop-only-historical-input-preprocessing-v1",
        "source_rows": [0, ROWS],
        "source_dimension": DIMENSION,
        "effective_dimension": DIMENSION,
        "compute_dtype": "<f4",
        "operation": "exact-r0147-staged-fp16-prefix-to-device-fp32",
        "l2_renormalized_for_training": False,
        "row_universe": ROW_UNIVERSE,
        "raw_prefix_rows": RAW_PREFIX_ROWS,
        "excluded_rows": RAW_PREFIX_EXCLUDED_ROWS,
        "size_preserving": False,
        "replacement_rows": 0,
        "source_sha256": source_sha256,
        "selection_sha256": selection_sha256,
    }
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}


def treatment_train_config(
    *,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    graph_edges: int,
    source_sha256: str,
    selection_sha256: str,
) -> tuple[dict[str, Any], str]:
    config, _ = host_train_config(
        cell=CURRENT_GRAPH_CURRENT_HOST,
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        graph_edges=graph_edges,
    )
    config = copy.deepcopy(config)
    stamp = treatment_preprocessing_stamp(
        source_sha256=source_sha256,
        selection_sha256=selection_sha256,
    )
    config.update({
        "schema": "round0149-drop-only-historical-host-train-config-v1",
        "arm": TREATMENT,
        "causal_matrix": {
            "row_policy": "historical-prefix-exact-family-exclusion-no-replacement",
            "population_rows": ROWS,
            "cardinality_change_vs_raw": -RAW_PREFIX_EXCLUDED_ROWS,
            "graph_subsystem": "current-r0104-style-rebuilt-on-drop-only-population",
            "trainer_subsystem": "current-r0104-host",
            "row_policy_includes_induced_graph_change": True,
            "unique_causal_factor_isolated": False,
        },
        "input_preprocessing": stamp,
    })
    config["paired_invariant"]["rows"] = ROWS
    config["execution"]["expected_pipeline_stamp"].update({
        "negative_sampling": negative_sampling_stamp(ROWS),
        "source_representation": "fp16-control",
        "row_universe": ROW_UNIVERSE,
        "source_sha256": source_sha256,
        "selection_sha256": selection_sha256,
    })
    return config, sha256_bytes(canonical_json(config))


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


def _selection_guard(summary: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "target_rows": ROWS,
        "raw_prefix_rows": RAW_PREFIX_ROWS,
        "raw_prefix_excluded_rows": RAW_PREFIX_EXCLUDED_ROWS,
        "eligible_rows_retained": ROWS,
        "replacement_rows_beyond_raw_prefix": 0,
        "historical_position_start": 0,
        "historical_position_stop_exclusive": RAW_PREFIX_ROWS,
        "historical_order_preserved": True,
        "size_preserving": False,
        "parent_selection_target_rows": RAW_PREFIX_ROWS,
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise Round0149Error("drop-only selection receipt changed")
    return expected


def build_decision(
    cells: Mapping[str, Mapping[str, Any]],
    *,
    selection_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Decide the bounded three-cell decomposition without a causal overclaim."""
    if set(cells) != set(CELLS):
        raise Round0149Error("R0149 decision cells are missing or unexpected")
    selection = _selection_guard(selection_summary)
    values = {key: metric_view(cells[key]) for key in CELLS}
    restoration = {key: _floor_test(values[key]) for key in CELLS}
    if not restoration[CURRENT_GRAPH_CURRENT_HOST]["passed_all"]:
        raise Round0149Error("accepted R0140 raw historical control no longer restores")
    if restoration[SIZE_PRESERVING_TREATMENT]["passed_all"]:
        raise Round0149Error("accepted R0147 negative treatment unexpectedly restores")

    drop_restores = restoration[TREATMENT]["passed_all"]
    if drop_restores:
        outcome = "drop-only-historical-row-policy-restores"
        next_action = "replacement-cardinality-package-remains-suspect-no-scale-claim"
    else:
        outcome = "drop-only-historical-row-policy-does-not-restore"
        next_action = "row-policy-restoration-unresolved-consider-seed-replay-before-scale"
    return {
        "schema": "round0149-drop-only-decomposition-decision-v1",
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
        "drop_only_minus_raw_historical": {
            key: values[TREATMENT][key] - values[CURRENT_GRAPH_CURRENT_HOST][key]
            for key in METRICS
        },
        "drop_only_minus_size_preserving": {
            key: values[TREATMENT][key] - values[SIZE_PRESERVING_TREATMENT][key]
            for key in METRICS
        },
        "outcome": outcome,
        "next_action": next_action,
        "drop_only_compatible_with_restoration": drop_restores,
        "size_preserving_compatible_with_restoration": False,
        "unique_causal_factor_claimed": False,
        "diverse_scale_transfer_claimed": False,
        "registered_density_floor_changed": False,
        "map_registry_state_changed": False,
    }
