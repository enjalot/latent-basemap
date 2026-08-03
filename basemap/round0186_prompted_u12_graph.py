"""Frozen contract for the prompted U12 fuzzy-k50 graph and dose plan."""
from __future__ import annotations

from basemap.round0113_prompt_contrast import POSITIVE_ROWS_PER_UPDATE
from basemap.round0169_prompted_diverse import (
    DIMENSION,
    GRAPH_EXECUTION,
    GRAPH_K,
    GRAPH_MEAN_RECALL_FLOOR,
    GRAPH_NLIST,
    GRAPH_NPROBE,
    GRAPH_NPROBE_GRID,
    GRAPH_P10_RECALL_FLOOR,
    GRAPH_QUALITY_ROWS,
    GRAPH_QUALITY_SEED,
    GRAPH_TRAIN_ROWS,
    GRAPH_TRAIN_SEED,
    GRAPH_VECTOR_STORAGE,
    HOST_RSS_LIMIT_GIB,
    ROWS,
    Round0169Error,
)


ROUND_ID = "0186"
CAPABILITY = "jina-document-diverse-u12-prompted-fuzzy-k50-dose-plan-v1"
GRAPH_SCHEMA = "round0186-prompted-diverse-u12-fuzzy-graph-v1"
DOSE_PLAN_SCHEMA = "round0186-prompted-diverse-u12-dose-plan-v1"
BASELINE_GRAPH_EDGES = 148_801_612
BASELINE_SUCCESSFUL_UPDATES = 500_000
REFERENCE_UPDATES_PER_SECOND = 109.5
REFERENCE_THROUGHPUT_ROUND = "0171"
EVALUATION_ALLOWANCE_SECONDS = 900.0


class Round0186Error(Round0169Error):
    """The registered prompted U12 graph/dose-plan contract changed."""


def successful_updates_for_edges(graph_edges: int) -> int:
    """Return the first whole update at or above the R0115 per-edge dose."""
    if graph_edges <= 0:
        raise Round0186Error("graph edge count must be positive")
    return (
        BASELINE_SUCCESSFUL_UPDATES * graph_edges
        + BASELINE_GRAPH_EDGES
        - 1
    ) // BASELINE_GRAPH_EDGES


def positive_draws_per_edge(*, successful_updates: int, graph_edges: int) -> float:
    if successful_updates <= 0 or graph_edges <= 0:
        raise Round0186Error("dose inputs must be positive")
    return successful_updates * POSITIVE_ROWS_PER_UPDATE / graph_edges


__all__ = [
    "BASELINE_GRAPH_EDGES",
    "BASELINE_SUCCESSFUL_UPDATES",
    "CAPABILITY",
    "DIMENSION",
    "DOSE_PLAN_SCHEMA",
    "EVALUATION_ALLOWANCE_SECONDS",
    "GRAPH_EXECUTION",
    "GRAPH_K",
    "GRAPH_MEAN_RECALL_FLOOR",
    "GRAPH_NLIST",
    "GRAPH_NPROBE",
    "GRAPH_NPROBE_GRID",
    "GRAPH_P10_RECALL_FLOOR",
    "GRAPH_QUALITY_ROWS",
    "GRAPH_QUALITY_SEED",
    "GRAPH_SCHEMA",
    "GRAPH_TRAIN_ROWS",
    "GRAPH_TRAIN_SEED",
    "GRAPH_VECTOR_STORAGE",
    "HOST_RSS_LIMIT_GIB",
    "REFERENCE_THROUGHPUT_ROUND",
    "REFERENCE_UPDATES_PER_SECOND",
    "ROUND_ID",
    "ROWS",
    "Round0186Error",
    "positive_draws_per_edge",
    "successful_updates_for_edges",
]
