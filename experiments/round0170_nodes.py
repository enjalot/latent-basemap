"""Execute R0170 by binding the reviewed Q2 kernel to fp16 IVF storage."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from basemap.round0170_prompted_8m import (
    CAPABILITY,
    GRAPH_VECTOR_STORAGE,
    ROUND_ID,
    Round0170Error,
    scale_decision,
    scale_train_config,
)
from experiments import round0166_nodes as q2


GRAPH_SCHEMA = "round0170-prompted-8m-fuzzy-graph-v1"
QUERY_SCHEMA = "round0170-prompted-8m-heldout-query-v1"
TRAIN_SCHEMA = "round0170-prompted-8m-train-receipt-v1"
EVALUATION_SCHEMA = "round0170-prompted-8m-scale-evaluation-v1"
PRODUCTION_CONFIG_SCHEMA = "round0170-prompted-8m-production-config-v1"
GRAPH_INDEX_DESCRIPTION = "GPU IndexIVFFlat/IP fp16 vector storage"


def _fp16_gpu_options(faiss: Any) -> Any:
    options = faiss.GpuClonerOptions()
    options.indicesOptions = faiss.INDICES_64_BIT
    options.useFloat16 = True
    options.usePrecomputed = True
    return options


def _configure() -> None:
    bindings = {
        "ROUND_ID": ROUND_ID,
        "CAPABILITY": CAPABILITY,
        "Round0166Error": Round0170Error,
        "GRAPH_SCHEMA": GRAPH_SCHEMA,
        "QUERY_SCHEMA": QUERY_SCHEMA,
        "TRAIN_SCHEMA": TRAIN_SCHEMA,
        "EVALUATION_SCHEMA": EVALUATION_SCHEMA,
        "PRODUCTION_CONFIG_SCHEMA": PRODUCTION_CONFIG_SCHEMA,
        "GRAPH_INDEX_DESCRIPTION": GRAPH_INDEX_DESCRIPTION,
        "GRAPH_REFERENCE_ROW_ORDER": "R0165 frozen-prefix prompted compact order",
        "GRAPH_REFERENCE_ANCHOR_NAMESPACE": "R0165 compact IDs",
        "scale_decision": scale_decision,
        "scale_train_config": scale_train_config,
        "_faiss_gpu_options": _fp16_gpu_options,
    }
    for name, value in bindings.items():
        setattr(q2, name, value)


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    _configure()
    q2.run_job(active, job)
