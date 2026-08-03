"""Frozen contract for the dose-matched prompted-English 8M rung."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import canonical_json, sha256_bytes
from basemap.round0113_prompt_contrast import POSITIVE_ROWS_PER_UPDATE
from basemap.round0171_prompted_8m import (
    GRAPH_EXECUTION,
    GRAPH_VECTOR_STORAGE,
    Round0171Error,
    scale_decision,
    scale_train_config as _base_train_config,
)


ROUND_ID = "0180"
CAPABILITY = "jina-document-english-8m-prompted-map-seed42-dose-matched-v1"
SEED = 42
RETAINED_ROWS = 7_952_419
BASELINE_GRAPH_EDGES = 148_801_612
TARGET_GRAPH_EDGES = 603_086_368
BASELINE_SUCCESSFUL_UPDATES = 500_000
SUCCESSFUL_UPDATES = (
    BASELINE_SUCCESSFUL_UPDATES * TARGET_GRAPH_EDGES
    + BASELINE_GRAPH_EDGES
    - 1
) // BASELINE_GRAPH_EDGES
TARGET_POSITIVE_DRAWS_PER_EDGE = (
    BASELINE_SUCCESSFUL_UPDATES
    * POSITIVE_ROWS_PER_UPDATE
    / BASELINE_GRAPH_EDGES
)
ACHIEVED_POSITIVE_DRAWS_PER_EDGE = (
    SUCCESSFUL_UPDATES * POSITIVE_ROWS_PER_UPDATE / TARGET_GRAPH_EDGES
)
HOST_RSS_LIMIT_GIB = 28.0


class Round0180Error(Round0171Error):
    """The registered dose-matched R0180 contract changed."""


def scale_train_config(
    *,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    graph_edges: int,
    retained_rows: int,
) -> tuple[dict[str, Any], str]:
    """Change only the successful-update horizon on the byte-exact R0171 graph."""
    if graph_edges != TARGET_GRAPH_EDGES or retained_rows != RETAINED_ROWS:
        raise Round0180Error("R0180 graph or population cardinality changed")
    config, _digest = _base_train_config(
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        graph_edges=graph_edges,
        retained_rows=retained_rows,
    )
    config = copy.deepcopy(config)
    config["schema"] = "round0180-prompted-8m-dose-matched-train-config-v1"
    config["paired_invariant"].update({
        "successful_positive_lr_updates": SUCCESSFUL_UPDATES,
        "dose_rule": (
            "ceil(500000 * 603086368 / 148801612) successful updates; "
            "match the accepted R0115 seed-42 positive-draws-per-edge exposure"
        ),
        "graph_reuse": "byte-exact accepted R0171 sharded-fp32 graph",
    })
    config["optimizer"]["successful_positive_lr_updates"] = SUCCESSFUL_UPDATES
    config["execution"].update({
        "scale_change": (
            "dose horizon only; R0171 population, graph, seed, sampler, model, "
            "optimizer, precision, and panel are byte/config exact"
        ),
        "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
        "achieved_positive_draws_per_edge": ACHIEVED_POSITIVE_DRAWS_PER_EDGE,
    })
    config["dose_registration"] = {
        "baseline_round": "0115",
        "baseline_graph_edges": BASELINE_GRAPH_EDGES,
        "baseline_successful_updates": BASELINE_SUCCESSFUL_UPDATES,
        "positive_rows_per_update": POSITIVE_ROWS_PER_UPDATE,
        "target_graph_round": "0171",
        "target_graph_edges": TARGET_GRAPH_EDGES,
        "rounding": "ceiling to the first whole successful update at or above target",
        "successful_updates": SUCCESSFUL_UPDATES,
        "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
        "achieved_positive_draws_per_edge": ACHIEVED_POSITIVE_DRAWS_PER_EDGE,
    }
    return config, sha256_bytes(canonical_json(config))


__all__ = [
    "ACHIEVED_POSITIVE_DRAWS_PER_EDGE",
    "BASELINE_GRAPH_EDGES",
    "BASELINE_SUCCESSFUL_UPDATES",
    "CAPABILITY",
    "GRAPH_EXECUTION",
    "GRAPH_VECTOR_STORAGE",
    "HOST_RSS_LIMIT_GIB",
    "RETAINED_ROWS",
    "ROUND_ID",
    "Round0180Error",
    "SEED",
    "SUCCESSFUL_UPDATES",
    "TARGET_GRAPH_EDGES",
    "TARGET_POSITIVE_DRAWS_PER_EDGE",
    "scale_decision",
    "scale_train_config",
]
