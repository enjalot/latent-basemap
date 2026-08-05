"""Frozen contract for the R0188 seed-43 half-to-full replay."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import canonical_json, sha256_bytes
from basemap import round0113_prompt_contrast as r0113
from basemap.round0187_composition_nested_ladder import (
    DIMENSION,
    FULL_GRAPH_EDGES,
    FULL_SUCCESSFUL_UPDATES,
    MULTIPLICITY_POLICY,
    POSITIVE_ROWS_PER_UPDATE,
    PRIMARY_METRICS,
    REQUIRED_TRAIN_CHECKS,
    RETENTION_RATIO,
    RUNG_COUNTS,
    RUNG_ROWS,
    TARGET_POSITIVE_DRAWS_PER_EDGE,
)


ROUND_ID = "0188"
SEED = 43
RUNGS = ("half", "full")
CAPABILITY = (
    "jina-document-english-composition-controlled-half-full-seed43-replay-v1"
)
EVALUATION_SCHEMA = "round0188-composition-boundary-evaluation-v1"
SYNTHESIS_SCHEMA = "round0188-composition-boundary-seed43-decision-v1"
TRAIN_SCHEMA_PREFIX = "round0188-composition-boundary-seed43-train-receipt"


class Round0188Error(RuntimeError):
    """The preregistered R0188 contract changed or failed authentication."""


def successful_updates_for_edges(edge_count: int) -> int:
    """Match R0180/R0187's exact consumed-positive-draws-per-edge dose."""
    if edge_count <= 0:
        raise Round0188Error("edge count must be positive")
    numerator = FULL_SUCCESSFUL_UPDATES * int(edge_count)
    return (numerator + FULL_GRAPH_EDGES - 1) // FULL_GRAPH_EDGES


def train_schema(rung: str) -> str:
    if rung not in RUNGS:
        raise Round0188Error("unknown boundary rung")
    return f"{TRAIN_SCHEMA_PREFIX}-{rung}-v1"


def train_config(
    *,
    rung: str,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    graph_edges: int,
    retained_rows: int,
) -> tuple[dict[str, Any], str]:
    """Clone the R0187 treatment and change only the model seed."""
    if rung not in RUNGS or retained_rows != RUNG_ROWS[rung]:
        raise Round0188Error("seed replay rung/cardinality changed")
    updates = successful_updates_for_edges(graph_edges)
    config, _ = r0113.train_config(
        "document",
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        graph_edges=graph_edges,
        # R0113 authenticates its original population before descendants replace
        # every population-bound field below.
        retained_rows=r0113.RETAINED_ROWS,
        seed=SEED,
    )
    config = copy.deepcopy(config)
    config["schema"] = f"round0188-{rung}-seed43-train-config-v1"
    config["paired_invariant"] = {
        "rung": rung,
        "rows": retained_rows,
        "dimension": DIMENSION,
        "seed": SEED,
        "successful_positive_lr_updates": updates,
        "dose_rule": (
            "ceil(R0180_successful_updates * active_edges / "
            "R0180_directed_edges)"
        ),
        "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
        "graph_policy": "byte-exact accepted R0187/R0171 fuzzy-k50 graph",
        "sampler": r0113.SAMPLER_CLASS,
        "hidden_dimension": 2048,
        "only_model_treatment_relative_to_r0187": "seed 42 -> 43",
    }
    config["input"].update({
        "rows": retained_rows,
        "representation": "prompted-document-host-fp16",
        "multiplicity_policy": MULTIPLICITY_POLICY,
        "composition": dict(RUNG_COUNTS[rung]),
        "nested_population_schema": (
            "round0187-composition-nested-population-v1"
            if rung == "half"
            else "round0165-prompted-english-frozen-prefix-population-v1"
        ),
    })
    expected = config["execution"]["expected_pipeline_stamp"]
    expected["negative_sampling"] = (
        f"uniform-{retained_rows}-compact-representatives-nonself"
    )
    expected["compact_retained_rows"] = retained_rows
    expected["multiplicity_policy"] = MULTIPLICITY_POLICY
    config["optimizer"]["successful_positive_lr_updates"] = updates
    config["execution"].update({
        "scale_change": (
            "none relative to the matching R0187 rung; seed 43 is the sole "
            "model treatment and graph/population/dose/recipe remain frozen"
        ),
        "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
        "achieved_positive_draws_per_edge": (
            updates * POSITIVE_ROWS_PER_UPDATE / graph_edges
        ),
        "graph_vector_storage": "gpu-ivfflat-fp32-complete-shard-search",
        "graph_execution": "all-row-shards-shared-quantizer-global-topk",
    })
    config["dose_registration"] = {
        "source_round": "0180",
        "source_graph_edges": FULL_GRAPH_EDGES,
        "source_successful_updates": FULL_SUCCESSFUL_UPDATES,
        "positive_rows_per_update": POSITIVE_ROWS_PER_UPDATE,
        "active_graph_edges": graph_edges,
        "successful_updates": updates,
        "rounding": "ceiling to first whole successful update at/above target",
        "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
        "achieved_positive_draws_per_edge": (
            updates * POSITIVE_ROWS_PER_UPDATE / graph_edges
        ),
    }
    return config, sha256_bytes(canonical_json(config))


def train_checks_close(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == REQUIRED_TRAIN_CHECKS
        and all(value[key] is True for key in REQUIRED_TRAIN_CHECKS)
    )


def boundary_decision(
    *,
    seed42: Mapping[str, Mapping[str, float]],
    seed43: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Decide whether seed 43 confirms R0187's Pile-FFR boundary loss."""
    normalized: dict[str, dict[str, dict[str, float]]] = {}
    for seed, cells in (("seed42", seed42), ("seed43", seed43)):
        if set(cells) != set(RUNGS):
            raise Round0188Error(f"{seed} boundary rung set changed")
        normalized[seed] = {
            rung: {key: float(value) for key, value in cells[rung].items()}
            for rung in RUNGS
        }
        if any(
            set(metrics) != set(PRIMARY_METRICS)
            for metrics in normalized[seed].values()
        ):
            raise Round0188Error(f"{seed} primary metric set changed")
    values = [
        value
        for cells in normalized.values()
        for metrics in cells.values()
        for value in metrics.values()
    ]
    if not np.isfinite(values).all() or any(value <= 0 for value in values):
        raise Round0188Error("boundary metrics must be finite and positive")

    retention = {
        seed: {
            metric: cells["full"][metric] / cells["half"][metric]
            for metric in PRIMARY_METRICS
        }
        for seed, cells in normalized.items()
    }
    seed42_trigger = retention["seed42"]["pile_ffr"] < RETENTION_RATIO
    seed43_trigger = retention["seed43"]["pile_ffr"] < RETENTION_RATIO
    if not seed42_trigger:
        raise Round0188Error("accepted R0187 Pile-FFR trigger changed")
    if seed43_trigger:
        outcome = "composition-controlled-size-regression-confirmed-two-seed"
        follow_up = "one targeted h4096 half-to-full sibling may be preregistered"
    else:
        outcome = "composition-controlled-size-regression-not-replicated"
        follow_up = "capacity work remains blocked; record seed sensitivity"
    return {
        "outcome": outcome,
        "registered_metric": "pile_ffr",
        "registered_boundary": "half_to_full",
        "retention_floor": RETENTION_RATIO,
        "cells": normalized,
        "retention": retention,
        "seed42_trigger_reproduced_from_r0187": seed42_trigger,
        "seed43_confirms_registered_regression": seed43_trigger,
        "capacity_follow_up_activated": seed43_trigger,
        "other_seed43_metric_misses_diagnostic": [
            metric
            for metric in PRIMARY_METRICS
            if metric != "pile_ffr" and retention["seed43"][metric] < RETENTION_RATIO
        ],
        "follow_up": follow_up,
    }


__all__ = [
    "CAPABILITY",
    "EVALUATION_SCHEMA",
    "ROUND_ID",
    "RUNGS",
    "SEED",
    "SYNTHESIS_SCHEMA",
    "Round0188Error",
    "boundary_decision",
    "successful_updates_for_edges",
    "train_checks_close",
    "train_config",
    "train_schema",
]
