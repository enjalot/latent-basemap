"""Execute R0180 by changing only the R0171 successful-update horizon."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from basemap.round0180_dose_matched_8m import (
    CAPABILITY,
    HOST_RSS_LIMIT_GIB,
    ROUND_ID,
    SUCCESSFUL_UPDATES,
    Round0180Error,
    scale_decision,
    scale_train_config,
)
from experiments import round0166_nodes as q2


GRAPH_SCHEMA = "round0171-prompted-8m-fuzzy-graph-v1"
QUERY_SCHEMA = "round0171-prompted-8m-heldout-query-v1"
TRAIN_SCHEMA = "round0180-prompted-8m-dose-matched-train-receipt-v1"
EVALUATION_SCHEMA = "round0180-prompted-8m-dose-matched-evaluation-v1"
PRODUCTION_CONFIG_SCHEMA = "round0180-prompted-8m-dose-matched-config-v1"
GRAPH_INDEX_DESCRIPTION = (
    "two row-disjoint GPU IndexIVFFlat/IP shards with fp32 vector storage, "
    "one shared coarse quantizer, and exact global top-k merge"
)
ALLOWED_ACTIONS = {"train_prompted_8m", "evaluate_prompted_8m"}


def _configure() -> None:
    bindings = {
        "ROUND_ID": ROUND_ID,
        "CAPABILITY": CAPABILITY,
        "SUCCESSFUL_UPDATES": SUCCESSFUL_UPDATES,
        "HOST_RSS_LIMIT_GIB": HOST_RSS_LIMIT_GIB,
        "Round0166Error": Round0180Error,
        "GRAPH_SCHEMA": GRAPH_SCHEMA,
        "QUERY_SCHEMA": QUERY_SCHEMA,
        "TRAIN_SCHEMA": TRAIN_SCHEMA,
        "EVALUATION_SCHEMA": EVALUATION_SCHEMA,
        "PRODUCTION_CONFIG_SCHEMA": PRODUCTION_CONFIG_SCHEMA,
        "GRAPH_INDEX_DESCRIPTION": GRAPH_INDEX_DESCRIPTION,
        "GRAPH_REFERENCE_ROW_ORDER": "R0165 frozen-prefix prompted compact order",
        "GRAPH_REFERENCE_ANCHOR_NAMESPACE": "R0165 compact IDs",
        "GRAPH_SOURCE_ROUND_ID": "0171",
        "GRAPH_BUILT_IN_ROUND": False,
        "scale_decision": scale_decision,
        "scale_train_config": scale_train_config,
    }
    for name, value in bindings.items():
        setattr(q2, name, value)


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    action = str(job.get("action") or "")
    if action not in ALLOWED_ACTIONS:
        raise Round0180Error(f"R0180 does not authorize action {action!r}")
    _configure()
    q2.run_job(active, job)


__all__ = ["run_job"]
