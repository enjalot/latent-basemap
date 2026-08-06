"""Frozen contract for the h4096 quarter/half matched-dose ladder."""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import canonical_json, sha256_bytes
from basemap.round0187_composition_nested_ladder import (
    PRIMARY_METRICS,
    RUNG_ROWS,
    train_config as h2048_nested_train_config,
)
from basemap.round0191_full_width_contrast import (
    HOST_RSS_LIMIT_GIB,
    MINIMUM_TRAIN_UPDATES_PER_S,
    WARNING_TRAIN_UPDATES_PER_S,
)
from basemap import round0113_prompt_contrast as prompt_contract


ROUND_ID = "0202"
SEED = 42
HIDDEN_DIMENSION = 4096
RUNGS = ("quarter", "half")
FULL_GRAPH_EDGES = 603_086_368
FULL_SUCCESSFUL_UPDATES = 1_000_000
POSITIVE_ROWS_PER_UPDATE = prompt_contract.POSITIVE_ROWS_PER_UPDATE
TARGET_POSITIVE_DRAWS_PER_EDGE = (
    FULL_SUCCESSFUL_UPDATES * POSITIVE_ROWS_PER_UPDATE / FULL_GRAPH_EDGES
)
CAPABILITY = "jina-document-english-h4096-composition-nested-dose-ladder-v1"
TRAIN_SCHEMA_PREFIX = "round0202-h4096-composition-nested-train-receipt"
EVALUATION_SCHEMA = "round0202-h4096-composition-nested-common-core-evaluation-v1"
SYNTHESIS_SCHEMA = "round0202-h4096-composition-nested-ladder-summary-v1"


class Round0202Error(RuntimeError):
    """The preregistered R0202 width-by-N treatment changed."""


def train_schema(rung: str) -> str:
    if rung not in RUNGS:
        raise Round0202Error("h4096 nested rung changed")
    return f"{TRAIN_SCHEMA_PREFIX}-{rung}-v1"


def successful_updates_for_edges(edge_count: int) -> int:
    """Ceil the exact R0191 full-rung consumed-draws/edge rational."""
    if edge_count <= 0:
        raise Round0202Error("edge count must be positive")
    numerator = FULL_SUCCESSFUL_UPDATES * int(edge_count)
    return (numerator + FULL_GRAPH_EDGES - 1) // FULL_GRAPH_EDGES


def train_config(
    *,
    rung: str,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    graph_edges: int,
    retained_rows: int,
) -> tuple[dict[str, Any], str]:
    """Clone the accepted nested recipe; change width and the registered dose."""
    if rung not in RUNGS or retained_rows != RUNG_ROWS[rung]:
        raise Round0202Error("h4096 nested rung/cardinality changed")
    config, _digest = h2048_nested_train_config(
        rung=rung,
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        graph_edges=graph_edges,
        retained_rows=retained_rows,
    )
    config = copy.deepcopy(config)
    updates = successful_updates_for_edges(graph_edges)
    achieved = updates * POSITIVE_ROWS_PER_UPDATE / graph_edges
    config["schema"] = f"round0202-{rung}-h4096-matched-dose-train-config-v1"
    config["model"]["hidden_dimension"] = HIDDEN_DIMENSION
    config["optimizer"]["successful_positive_lr_updates"] = updates
    config["paired_invariant"].update({
        "hidden_dimension": HIDDEN_DIMENSION,
        "successful_positive_lr_updates": updates,
        "dose_rule": (
            "ceil(R0191_successful_updates * active_edges / "
            "R0191_directed_edges)"
        ),
        "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
        "only_treatments_relative_to_r0187": (
            "hidden_dimension 2048 -> 4096 and dose 1.3743131099922326 -> "
            "exact R0191 0.6781781544098838 draws/directed-edge"
        ),
    })
    config["execution"].update({
        "scale_change": (
            "composition-preserving nested N at h4096 and the exact accepted "
            "R0191 full-rung dose; population, graph, seed, prompt, precision, "
            "sampler, optimizer, and common evaluation core frozen"
        ),
        "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
        "achieved_positive_draws_per_edge": achieved,
        "minimum_train_upd_s": MINIMUM_TRAIN_UPDATES_PER_S,
        "warning_train_upd_s": WARNING_TRAIN_UPDATES_PER_S,
        "width_by_n_role": "h4096 quarter/half cells paired to accepted R0191 full",
    })
    config["dose_registration"] = {
        "source_round": "0191",
        "source_graph_edges": FULL_GRAPH_EDGES,
        "source_successful_updates": FULL_SUCCESSFUL_UPDATES,
        "positive_rows_per_update": POSITIVE_ROWS_PER_UPDATE,
        "active_graph_edges": graph_edges,
        "successful_updates": updates,
        "rounding": "ceiling to first whole successful update at/above target",
        "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
        "achieved_positive_draws_per_edge": achieved,
    }
    return config, sha256_bytes(canonical_json(config))


def ladder_summary(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Report the h4096 N response without making the cross-width A3 decision."""
    expected_rungs = {"quarter", "half", "full"}
    if set(cells) != expected_rungs:
        raise Round0202Error("h4096 ladder cells changed")
    normalized: dict[str, dict[str, float]] = {}
    for rung in ("quarter", "half", "full"):
        if set(cells[rung]) != set(PRIMARY_METRICS):
            raise Round0202Error(f"{rung} metric set changed")
        normalized[rung] = {}
        for metric, value in cells[rung].items():
            number = float(value)
            if not math.isfinite(number) or number <= 0:
                raise Round0202Error(f"{rung}/{metric} is not finite and positive")
            normalized[rung][metric] = number
    retentions = {
        metric: {
            "half_over_quarter": normalized["half"][metric]
            / normalized["quarter"][metric],
            "full_over_half": normalized["full"][metric]
            / normalized["half"][metric],
            "full_over_quarter": normalized["full"][metric]
            / normalized["quarter"][metric],
        }
        for metric in PRIMARY_METRICS
    }
    return {
        "cells": normalized,
        "retentions": retentions,
        "registered_metric": "pile_ffr",
        "registered_pile_ffr_retentions": retentions["pile_ffr"],
        "decision_deferred_to_track_a3": True,
    }


__all__ = [
    "CAPABILITY",
    "EVALUATION_SCHEMA",
    "FULL_GRAPH_EDGES",
    "FULL_SUCCESSFUL_UPDATES",
    "HIDDEN_DIMENSION",
    "HOST_RSS_LIMIT_GIB",
    "ROUND_ID",
    "RUNGS",
    "SEED",
    "SYNTHESIS_SCHEMA",
    "TARGET_POSITIVE_DRAWS_PER_EDGE",
    "Round0202Error",
    "ladder_summary",
    "successful_updates_for_edges",
    "train_config",
    "train_schema",
]
