"""Execute the diagnostic R0184 1M-update dose-response point."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from basemap.round0184_prompted_8m_dose_midpoint import (
    CAPABILITY,
    HOST_RSS_LIMIT_GIB,
    ROUND_ID,
    SUCCESSFUL_UPDATES,
    Round0184Error,
    diagnostic_scale_decision,
    scale_train_config,
)
from experiments import round0166_nodes as q2


GRAPH_SCHEMA = "round0171-prompted-8m-fuzzy-graph-v1"
QUERY_SCHEMA = "round0171-prompted-8m-heldout-query-v1"
TRAIN_SCHEMA = "round0184-prompted-8m-dose-midpoint-train-receipt-v1"
EVALUATION_SCHEMA = "round0184-prompted-8m-dose-midpoint-evaluation-v1"
PRODUCTION_CONFIG_SCHEMA = "round0184-prompted-8m-dose-midpoint-config-v1"
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
        "Round0166Error": Round0184Error,
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
        "scale_decision": diagnostic_scale_decision,
        "scale_train_config": scale_train_config,
    }
    for name, value in bindings.items():
        setattr(q2, name, value)


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    action = str(job.get("action") or "")
    if action not in ALLOWED_ACTIONS:
        raise Round0184Error(f"R0184 does not authorize action {action!r}")
    _configure()
    q2.run_job(active, job)


__all__ = ["run_job"]
