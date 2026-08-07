"""Frozen contract for the low-dose h2048 prompted-diverse U12 train stage.

R0207's accepted factorial resolved the scaling question as
``both-widths-flat-at-low-dose`` and registered the operating rule the owner
campaign then adopted: scale N at hidden dimension 2048 and the exact accepted
R0184/R0191 dose of 0.6781781544098838 positive draws per directed edge.  R0169
preregistered the diverse U12 rung before that evidence existed and fixed its
horizon at 500,000 successful updates; R0210 replaces exactly that one number
with the registered dose rule and changes nothing else about the recipe.

The dose rule is the R0202/R0203 rule, imported rather than restated::

    updates = ceil(1,000,000 * active_directed_edges / 603,086,368)

The denominator and numerator are the accepted R0184 full-rung graph and its
successful-update count, so the achieved draws-per-edge reproduces the
registered target to within one update of rounding.  The active edge count is
read from the sealed R0209 graph receipt, never estimated: the R0207 memo
requires that "the final update horizon is recomputed from the sealed prompted
graph edge count".
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import canonical_json, sha256_bytes
from basemap.round0202_h4096_nested_dose_ladder import (
    FULL_GRAPH_EDGES,
    FULL_SUCCESSFUL_UPDATES,
    TARGET_POSITIVE_DRAWS_PER_EDGE,
)
from basemap.round0169_prompted_diverse import (
    DIMENSION,
    HOST_RSS_LIMIT_GIB,
    ROWS,
    SEED,
    Round0169Error,
    diverse_train_config,
)
from basemap import round0113_prompt_contrast as r0113


ROUND_ID = "0210"
CAPABILITY = "jina-prompted-diverse-u12-map-seed42-low-dose-v1"
TRAIN_SCHEMA = "round0210-prompted-diverse-u12-low-dose-train-receipt-v1"
PRODUCTION_CONFIG_SCHEMA = (
    "round0210-prompted-diverse-u12-low-dose-production-config-v1"
)
HIDDEN_DIMENSION = 2048
POSITIVE_ROWS_PER_UPDATE = r0113.POSITIVE_ROWS_PER_UPDATE
GRAPH_CAPABILITY = "jina-prompted-diverse-u12-fuzzy-k50-graph-v1"
GRAPH_SCHEMA = "round0209-prompted-diverse-u12-fuzzy-graph-v1"
#: R0169's superseded horizon, kept only so a test can assert R0210 changed it.
SUPERSEDED_FIXED_UPDATES = 500_000


class Round0210Error(Round0169Error):
    """The registered low-dose prompted-diverse train contract changed."""


def successful_updates_for_edges(edge_count: int) -> int:
    """The accepted R0184/R0202 low-dose horizon for an active edge count."""
    if int(edge_count) <= 0:
        raise Round0210Error("R0210 active directed-edge count must be positive")
    numerator = FULL_SUCCESSFUL_UPDATES * int(edge_count)
    return (numerator + FULL_GRAPH_EDGES - 1) // FULL_GRAPH_EDGES


def achieved_draws_per_edge(*, updates: int, edge_count: int) -> float:
    if int(edge_count) <= 0 or int(updates) <= 0:
        raise Round0210Error("R0210 dose arithmetic requires positive inputs")
    return int(updates) * POSITIVE_ROWS_PER_UPDATE / int(edge_count)


def low_dose_train_config(
    *,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    graph_edges: int,
    retained_rows: int,
) -> tuple[dict[str, Any], str]:
    """Clone the R0169 diverse recipe and change only the dose horizon."""
    if retained_rows != ROWS:
        raise Round0210Error("R0210 population cardinality changed")
    config, _digest = diverse_train_config(
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        graph_edges=graph_edges,
        retained_rows=retained_rows,
    )
    config = copy.deepcopy(config)
    if int(config["model"]["hidden_dimension"]) != HIDDEN_DIMENSION:
        raise Round0210Error("R0210 requires the registered h2048 recipe")
    updates = successful_updates_for_edges(graph_edges)
    achieved = achieved_draws_per_edge(updates=updates, edge_count=graph_edges)
    if not math.isclose(
        achieved, TARGET_POSITIVE_DRAWS_PER_EDGE, rel_tol=1.0e-6, abs_tol=0.0
    ):
        raise Round0210Error("R0210 achieved dose left the registered target")
    config["schema"] = "round0210-prompted-diverse-u12-low-dose-train-config-v1"
    config["optimizer"]["successful_positive_lr_updates"] = updates
    config["paired_invariant"].update({
        "successful_positive_lr_updates": updates,
        "dose_rule": (
            "ceil(R0184_successful_updates * active_edges / R0184_directed_edges)"
        ),
        "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
        "only_treatment_relative_to_r0169": (
            "fixed 500,000 successful updates -> exact accepted R0184/R0191 "
            "0.6781781544098838 draws/directed-edge at the sealed R0209 edge count"
        ),
    })
    config["execution"].update({
        "scale_change": (
            "exact accepted R0132 U12 diverse population at h2048 and the "
            "accepted R0184 low dose; recipe, k50 graph law, seed, prompt, "
            "precision, sampler, and optimizer frozen"
        ),
        "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
        "achieved_positive_draws_per_edge": achieved,
        "width_by_n_role": (
            "R0207 both-widths-flat-at-low-dose operating rule applied to the "
            "12,474,331-row diverse rung"
        ),
    })
    config["dose_registration"] = {
        "source_round": "0184",
        "source_graph_edges": FULL_GRAPH_EDGES,
        "source_successful_updates": FULL_SUCCESSFUL_UPDATES,
        "positive_rows_per_update": POSITIVE_ROWS_PER_UPDATE,
        "active_graph_edges": int(graph_edges),
        "successful_updates": updates,
        "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
        "achieved_positive_draws_per_edge": achieved,
        "superseded_r0169_fixed_updates": SUPERSEDED_FIXED_UPDATES,
    }
    return config, sha256_bytes(canonical_json(config))


__all__ = [
    "CAPABILITY",
    "DIMENSION",
    "GRAPH_CAPABILITY",
    "GRAPH_SCHEMA",
    "HIDDEN_DIMENSION",
    "HOST_RSS_LIMIT_GIB",
    "POSITIVE_ROWS_PER_UPDATE",
    "PRODUCTION_CONFIG_SCHEMA",
    "ROUND_ID",
    "ROWS",
    "Round0210Error",
    "SEED",
    "SUPERSEDED_FIXED_UPDATES",
    "TARGET_POSITIVE_DRAWS_PER_EDGE",
    "TRAIN_SCHEMA",
    "achieved_draws_per_edge",
    "low_dose_train_config",
    "successful_updates_for_edges",
]
