"""Frozen contract for the h2048 quarter/half R0184-dose ladder."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import canonical_json, sha256_bytes
from basemap.round0187_composition_nested_ladder import (
    RUNG_ROWS,
    train_config as high_dose_train_config,
)
from basemap.round0202_h4096_nested_dose_ladder import (
    FULL_GRAPH_EDGES,
    FULL_SUCCESSFUL_UPDATES,
    POSITIVE_ROWS_PER_UPDATE,
    RUNGS,
    TARGET_POSITIVE_DRAWS_PER_EDGE,
    ladder_summary,
)


ROUND_ID = "0203"
SEED = 42
HIDDEN_DIMENSION = 2048
HOST_RSS_LIMIT_GIB = 28.0
CAPABILITY = "jina-document-english-h2048-composition-nested-low-dose-ladder-v1"
TRAIN_SCHEMA_PREFIX = "round0203-h2048-composition-nested-train-receipt"
EVALUATION_SCHEMA = "round0203-h2048-composition-nested-common-core-evaluation-v1"
SYNTHESIS_SCHEMA = "round0203-h2048-composition-nested-low-dose-summary-v1"


class Round0203Error(RuntimeError):
    """The preregistered R0203 matched-low-dose treatment changed."""


def train_schema(rung: str) -> str:
    if rung not in RUNGS:
        raise Round0203Error("h2048 nested rung changed")
    return f"{TRAIN_SCHEMA_PREFIX}-{rung}-v1"


def successful_updates_for_edges(edge_count: int) -> int:
    if edge_count <= 0:
        raise Round0203Error("edge count must be positive")
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
    """Clone R0187 and change only the dose horizon and bound descriptions."""
    if rung not in RUNGS or retained_rows != RUNG_ROWS[rung]:
        raise Round0203Error("h2048 nested rung/cardinality changed")
    config, _digest = high_dose_train_config(
        rung=rung,
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        graph_edges=graph_edges,
        retained_rows=retained_rows,
    )
    config = copy.deepcopy(config)
    updates = successful_updates_for_edges(graph_edges)
    achieved = updates * POSITIVE_ROWS_PER_UPDATE / graph_edges
    config["schema"] = f"round0203-{rung}-h2048-low-dose-train-config-v1"
    config["optimizer"]["successful_positive_lr_updates"] = updates
    config["paired_invariant"].update({
        "successful_positive_lr_updates": updates,
        "dose_rule": (
            "ceil(R0184_successful_updates * active_edges / "
            "R0184_directed_edges)"
        ),
        "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
        "only_treatment_relative_to_r0187": (
            "dose 1.3743131099922326 -> exact accepted R0184/R0191 "
            "0.6781781544098838 draws/directed-edge"
        ),
    })
    config["execution"].update({
        "scale_change": (
            "composition-preserving nested N at h2048 and exact accepted "
            "R0184 full-rung dose; population, graph, seed, model, prompt, "
            "precision, sampler, optimizer, and common evaluation core frozen"
        ),
        "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
        "achieved_positive_draws_per_edge": achieved,
        "width_by_n_role": "h2048 low-dose pair to accepted R0184 full",
    })
    config["dose_registration"] = {
        "source_round": "0184",
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


__all__ = [
    "CAPABILITY",
    "EVALUATION_SCHEMA",
    "HIDDEN_DIMENSION",
    "HOST_RSS_LIMIT_GIB",
    "ROUND_ID",
    "RUNGS",
    "SEED",
    "SYNTHESIS_SCHEMA",
    "TARGET_POSITIVE_DRAWS_PER_EDGE",
    "Round0203Error",
    "ladder_summary",
    "successful_updates_for_edges",
    "train_config",
    "train_schema",
]
