"""Frozen contracts for the fp16-IVF replacement of the prompted 8M rung."""
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


ROUND_ID = "0170"
CAPABILITY = "jina-document-english-8m-prompted-map-seed42-fp16-ivf-v1"
GRAPH_VECTOR_STORAGE = "gpu-ivfflat-fp16"


class Round0170Error(Round0166Error):
    """The registered R0170 replacement contract changed."""


def scale_train_config(
    *,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    graph_edges: int,
    retained_rows: int,
) -> tuple[dict[str, Any], str]:
    config, _digest = _base_train_config(
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        graph_edges=graph_edges,
        retained_rows=retained_rows,
    )
    config = copy.deepcopy(config)
    config["schema"] = "round0170-prompted-8m-train-config-v1"
    config["paired_invariant"]["graph_vector_storage"] = GRAPH_VECTOR_STORAGE
    config["execution"]["graph_vector_storage"] = GRAPH_VECTOR_STORAGE
    config["execution"]["capacity_correction"] = (
        "R0166 fp32 IVF payload exceeded the single 32GiB GPU; store IVF "
        "vectors as fp16 while retaining exact fp32 recall qualification"
    )
    return config, sha256_bytes(canonical_json(config))


__all__ = [
    "CAPABILITY",
    "GRAPH_VECTOR_STORAGE",
    "ROUND_ID",
    "Round0170Error",
    "scale_decision",
    "scale_train_config",
]
