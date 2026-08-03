"""Frozen contract for the diagnostic 1M-update prompted-English 8M dose point."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import canonical_json, sha256_bytes
from basemap.round0113_prompt_contrast import POSITIVE_ROWS_PER_UPDATE
from basemap.round0180_dose_matched_8m import (
    RETAINED_ROWS,
    TARGET_GRAPH_EDGES,
    TARGET_POSITIVE_DRAWS_PER_EDGE,
)
from basemap.round0171_prompted_8m import (
    GRAPH_EXECUTION,
    GRAPH_VECTOR_STORAGE,
    Round0171Error,
    scale_decision as _quality_decision,
    scale_train_config as _base_train_config,
)


ROUND_ID = "0184"
CAPABILITY = "jina-document-english-8m-prompted-dose-midpoint-readout-v1"
SEED = 42
SUCCESSFUL_UPDATES = 1_000_000
ACHIEVED_POSITIVE_DRAWS_PER_EDGE = (
    SUCCESSFUL_UPDATES * POSITIVE_ROWS_PER_UPDATE / TARGET_GRAPH_EDGES
)
HOST_RSS_LIMIT_GIB = 28.0


class Round0184Error(Round0171Error):
    """The registered diagnostic dose-midpoint contract changed."""


def scale_train_config(
    *,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    graph_edges: int,
    retained_rows: int,
) -> tuple[dict[str, Any], str]:
    """Change only the horizon on the byte-exact accepted R0171 graph."""
    if graph_edges != TARGET_GRAPH_EDGES or retained_rows != RETAINED_ROWS:
        raise Round0184Error("R0184 graph or population cardinality changed")
    config, _digest = _base_train_config(
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        graph_edges=graph_edges,
        retained_rows=retained_rows,
    )
    config = copy.deepcopy(config)
    config["schema"] = "round0184-prompted-8m-dose-midpoint-train-config-v1"
    config["paired_invariant"].update({
        "successful_positive_lr_updates": SUCCESSFUL_UPDATES,
        "dose_rule": (
            "exactly 1000000 successful updates on the accepted R0171 graph; "
            "diagnostic midpoint between the 500000-update R0171 cell and "
            "the 2026478-update dose-matched R0180 cell"
        ),
        "graph_reuse": "byte-exact accepted R0171 sharded-fp32 graph",
    })
    config["optimizer"]["successful_positive_lr_updates"] = SUCCESSFUL_UPDATES
    config["execution"].update({
        "scale_change": (
            "dose horizon only; population, graph, seed, sampler, model, "
            "optimizer, precision, and panel remain byte/config exact"
        ),
        "achieved_positive_draws_per_edge": ACHIEVED_POSITIVE_DRAWS_PER_EDGE,
        "dose_readout_role": "diagnostic-only; registered quality floors are reported, not gated",
    })
    config["dose_registration"] = {
        "curve_population_round": "0171",
        "graph_edges": TARGET_GRAPH_EDGES,
        "positive_rows_per_update": POSITIVE_ROWS_PER_UPDATE,
        "successful_updates": SUCCESSFUL_UPDATES,
        "achieved_positive_draws_per_edge": ACHIEVED_POSITIVE_DRAWS_PER_EDGE,
        "lower_point": {
            "round": "0171",
            "successful_updates": 500_000,
            "positive_draws_per_edge": (
                500_000 * POSITIVE_ROWS_PER_UPDATE / TARGET_GRAPH_EDGES
            ),
        },
        "upper_point": {
            "round": "0180",
            "successful_updates": 2_026_478,
            "positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
        },
        "role": "publication-oriented dose-response readout; not a quality gate",
    }
    return config, sha256_bytes(canonical_json(config))


def diagnostic_scale_decision(
    *,
    native: Mapping[str, float],
    matched_2m: Mapping[str, float],
    baseline_2m: Mapping[str, float],
    prompted_floors: Mapping[str, float],
) -> dict[str, Any]:
    """Report frozen quality cells without making them capability gates."""
    decision = _quality_decision(
        native=native,
        matched_2m=matched_2m,
        baseline_2m=baseline_2m,
        prompted_floors=prompted_floors,
    )
    return {
        **decision,
        "metric_gates_required_for_capability": False,
        "readout_role": (
            "diagnostic dose-response point; quality gates are reported verbatim "
            "but neither passing nor failing them changes execution acceptance"
        ),
    }


__all__ = [
    "ACHIEVED_POSITIVE_DRAWS_PER_EDGE",
    "CAPABILITY",
    "GRAPH_EXECUTION",
    "GRAPH_VECTOR_STORAGE",
    "HOST_RSS_LIMIT_GIB",
    "RETAINED_ROWS",
    "ROUND_ID",
    "Round0184Error",
    "SEED",
    "SUCCESSFUL_UPDATES",
    "TARGET_GRAPH_EDGES",
    "diagnostic_scale_decision",
    "scale_train_config",
]
