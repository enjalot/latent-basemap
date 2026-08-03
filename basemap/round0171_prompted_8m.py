"""Frozen contracts for the sharded-fp32 replacement of the prompted 8M rung."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import canonical_json, sha256_bytes
from basemap.round0166_prompted_8m import *  # noqa: F403
from basemap.round0166_prompted_8m import (
    Round0166Error,
    scale_decision,
    scale_train_config as _base_train_config,
)


ROUND_ID = "0171"
CAPABILITY = "jina-document-english-8m-prompted-map-seed42-sharded-fp32-ivf-v1"
GRAPH_VECTOR_STORAGE = "gpu-ivfflat-fp32-two-shard-exact-merge"
GRAPH_EXECUTION = "two-row-disjoint-shards-shared-quantizer-global-topk"


class Round0171Error(Round0166Error):
    """The registered R0171 replacement contract changed."""


def scale_train_config(
    *,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    graph_edges: int,
    retained_rows: int,
) -> tuple[dict[str, Any], str]:
    """Bind the unchanged Q2 science to its capacity-safe graph execution."""
    config, _digest = _base_train_config(
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        graph_edges=graph_edges,
        retained_rows=retained_rows,
    )
    config = copy.deepcopy(config)
    config["schema"] = "round0171-prompted-8m-train-config-v1"
    config["paired_invariant"]["graph_vector_storage"] = GRAPH_VECTOR_STORAGE
    config["execution"]["graph_vector_storage"] = GRAPH_VECTOR_STORAGE
    config["execution"]["graph_execution"] = GRAPH_EXECUTION
    config["execution"]["capacity_correction"] = (
        "R0166 monolithic fp32 IVF exceeded the single 32GiB GPU and R0170's "
        "GpuClonerOptions.useFloat16 premise was false for IndexIVFFlat; search "
        "two disjoint fp32 shards built from one trained coarse quantizer and "
        "merge every shard's candidates into the exact global top-k"
    )
    return config, sha256_bytes(canonical_json(config))


__all__ = [
    "CAPABILITY",
    "GRAPH_EXECUTION",
    "GRAPH_VECTOR_STORAGE",
    "ROUND_ID",
    "Round0171Error",
    "scale_decision",
    "scale_train_config",
]
