"""Frozen contract for the prompted-diverse U12 fuzzy k50 graph stage.

The R0207 design memo prices the 12,474,331-row prompted-diverse rung as a
sharded-fp32 graph followed by a low-dose h2048 train.  The memo's own note
that "the final update horizon is recomputed from the sealed prompted graph
edge count" makes the graph a hard prerequisite of the train horizon, so R0209
runs the graph as its own queue and R0210 consumes its sealed edge count.  The
campaign authorizes exactly this split ("split graph and train into sequential
queues rather than trimming either").

Every element of the graph law is the accepted R0169 diverse law, imported
rather than restated: fuzzy k50, IVF8192, nprobe 64 with the 16/32/64/128/256
qualification grid, graph seeds 113/114, one shared trained fp32 coarse
quantizer cloned into four row-disjoint <=4M-row GPU shards, and an exact
global top-k merge by similarity descending then global ID ascending.  R0209
changes no threshold and adds no check; it changes only which round owns the
graph artifact.
"""
from __future__ import annotations

from basemap.round0169_prompted_diverse import (  # noqa: F401 - re-exported law
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
    SEED,
    Round0169Error,
)


ROUND_ID = "0209"
CAPABILITY = "jina-prompted-diverse-u12-fuzzy-k50-graph-v1"
GRAPH_SCHEMA = "round0209-prompted-diverse-u12-fuzzy-graph-v1"
GRAPH_SHARD_ROWS = 4_000_000
EXPECTED_GRAPH_SHARDS = (
    (0, 4_000_000),
    (4_000_000, 8_000_000),
    (8_000_000, 12_000_000),
    (12_000_000, ROWS),
)
#: The R0207 memo's edge estimate, carried forward as a plausibility bound
#: only.  The sealed receipt's actual count is what R0210's dose consumes.
ESTIMATED_DIRECTED_EDGES = 946_013_908
DIRECTED_EDGE_PLAUSIBILITY = (0.80, 1.25)


class Round0209Error(Round0169Error):
    """The registered prompted-diverse graph stage changed."""


def plausible_directed_edges(edge_count: int) -> bool:
    """True when a sealed edge count is close enough to the memo estimate.

    This is a reporting guard, not a science threshold: the graph node already
    fails closed on degree, finiteness, weight range, and recall.  A count far
    from the estimate means the population or the k50 law drifted, and R0210's
    dose would silently follow it.
    """
    low, high = DIRECTED_EDGE_PLAUSIBILITY
    return (
        int(edge_count) >= int(ESTIMATED_DIRECTED_EDGES * low)
        and int(edge_count) <= int(ESTIMATED_DIRECTED_EDGES * high)
    )


__all__ = [
    "CAPABILITY",
    "DIMENSION",
    "DIRECTED_EDGE_PLAUSIBILITY",
    "ESTIMATED_DIRECTED_EDGES",
    "EXPECTED_GRAPH_SHARDS",
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
    "GRAPH_SHARD_ROWS",
    "GRAPH_TRAIN_ROWS",
    "GRAPH_TRAIN_SEED",
    "GRAPH_VECTOR_STORAGE",
    "HOST_RSS_LIMIT_GIB",
    "ROUND_ID",
    "ROWS",
    "Round0209Error",
    "SEED",
    "plausible_directed_edges",
]
