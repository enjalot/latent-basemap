"""Execute R0171 by binding Q2 to exact sharded fp32 IVF search."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from basemap.round0171_prompted_8m import (
    CAPABILITY,
    GRAPH_EXECUTION,
    ROUND_ID,
    Round0171Error,
    scale_decision,
    scale_train_config,
)
from experiments import round0166_nodes as q2


GRAPH_SCHEMA = "round0171-prompted-8m-fuzzy-graph-v1"
QUERY_SCHEMA = "round0171-prompted-8m-heldout-query-v1"
TRAIN_SCHEMA = "round0171-prompted-8m-train-receipt-v1"
EVALUATION_SCHEMA = "round0171-prompted-8m-scale-evaluation-v1"
PRODUCTION_CONFIG_SCHEMA = "round0171-prompted-8m-production-config-v1"
GRAPH_INDEX_DESCRIPTION = (
    "two row-disjoint GPU IndexIVFFlat/IP shards with fp32 vector storage, "
    "one shared coarse quantizer, and exact global top-k merge"
)
GRAPH_SHARD_ROWS = 4_000_000


def _fp32_gpu_options(faiss: Any) -> Any:
    options = faiss.GpuClonerOptions()
    options.indicesOptions = faiss.INDICES_64_BIT
    options.useFloat16 = False
    options.usePrecomputed = True
    return options


def _configure() -> None:
    bindings = {
        "ROUND_ID": ROUND_ID,
        "CAPABILITY": CAPABILITY,
        "Round0166Error": Round0171Error,
        "GRAPH_SCHEMA": GRAPH_SCHEMA,
        "QUERY_SCHEMA": QUERY_SCHEMA,
        "TRAIN_SCHEMA": TRAIN_SCHEMA,
        "EVALUATION_SCHEMA": EVALUATION_SCHEMA,
        "PRODUCTION_CONFIG_SCHEMA": PRODUCTION_CONFIG_SCHEMA,
        "GRAPH_INDEX_DESCRIPTION": GRAPH_INDEX_DESCRIPTION,
        "GRAPH_REFERENCE_ROW_ORDER": "R0165 frozen-prefix prompted compact order",
        "GRAPH_REFERENCE_ANCHOR_NAMESPACE": "R0165 compact IDs",
        "GRAPH_SHARD_ROWS": GRAPH_SHARD_ROWS,
        "scale_decision": scale_decision,
        "scale_train_config": scale_train_config,
        "_faiss_gpu_options": _fp32_gpu_options,
    }
    for name, value in bindings.items():
        setattr(q2, name, value)


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    _configure()
    q2.run_job(active, job)


__all__ = ["GRAPH_EXECUTION", "run_job"]
