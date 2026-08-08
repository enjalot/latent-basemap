"""R0228 geometry — the measurement the panel will not surface.

The panel scores four global aggregates over 4,000 anchors. The fear
review-0227-01 raised is *local*: the residual loss of `cluster-spill-nnd` is
`99.6-99.9%` mutual, regionally clustered (neighbour-loss autocorrelation
`0.8327`) and not the partition cut, and R0215 showed that edge absence is what
produced the v1 150M concentration-clumps. A 4,000-anchor mean can be untouched
while a region of the map has folded.

Three instruments, all alignment-free.

## 1. Clumps — R0215's own method, applied to a 2M map

`HEATMAP_BINS x HEATMAP_BINS` occupancy over coordinates trimmed at
`COORD_TRIM_PERCENTILE`, the `>= CLUMP_DENSITY_PERCENTILE` mask, and the
connected components of that mask under 8-connectivity. R0215's constants are
imported, not restated. What is compared across arms is the component count, the
largest component's share of clumped rows, and the maximum bin occupancy.

## 2. Zero-degree rows

The R0215 tripwire, per graph. Reported per configuration even when it is `0`,
because R0227's review singled out publishing it everywhere as that round's
strongest unqualified result and dropping it here would be a regression.

## 3. Displacement of the rows that actually lost edges

This is the round's most important measurement, and it needs care, because **two
maps from different seeds are not comparable pointwise**. Parametric UMAP is not
identifiable: there is no rigid transform relating seed 42's map to seed 43's, so
a cross-map per-row displacement — with or without Procrustes — measures seed
noise, not graph damage. Any statistic here must be computable *within* one map.

The statistic is the **true-neighbour scatter**:

    s(r) = mean over the row's 15 EXACT high-D neighbours of ||y_r - y_j||
           divided by the map's own RMS radius about its centroid

`s(r)` asks "in this map, how far apart did the row and its true neighbours
land?", is dimensionless, and is invariant to the translation, rotation and
uniform scaling that differ between maps. Truth comes from R0220's sealed exact
k15 arrays, so the *same* neighbour set is used in every map — including maps
trained on graphs that never contained those edges. That is the point: a lost
edge should show up as its endpoints drifting apart.

The comparison is a difference in differences, and the control is what makes it
mean anything:

* `lost` — rows missing at least one of their 15 exact neighbours in this
  configuration's candidate graph.
* `control` — rows missing none, **density-matched to `lost`** on deciles of the
  row's own true 15th-best cosine. Without this match, "rows that lost edges have
  higher scatter" could be nothing more than "sparse rows have higher scatter",
  which is already known to be true (R0227's density deciles).
* The **null arm** is the eight exact-graph maps, scored on the *same* row sets.
  Those maps were trained on graphs where nothing was lost, so any gap they show
  is confounding rather than damage.

    delta = (s_lost - s_control) in candidate maps
          - (s_lost - s_control) in exact-graph maps

`delta ~ 0` means the missing edges did not displace their rows beyond what the
row selection alone explains. `delta > 0` means they did, and by how much, in
units of the map's own radius.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .round0215_v1_forensic import (
    CLUMP_DENSITY_PERCENTILE,
    COORD_TRIM_PERCENTILE,
    HEATMAP_BINS,
    population_stats,
)


ROUND_ID = "0228"

#: Rows sampled per population for the scatter statistic. R0215 used 2,000 per
#: population at 150M; 20,000 here because the statistic is cheap at 2M and a
#: tighter standard error is the difference between seeing a 0.01-radius effect
#: and not.
SCATTER_SAMPLE_ROWS = 20_000
SCATTER_SAMPLE_SEED = 228
#: Deciles of the row's own true 15th-best cosine, the same density axis R0226
#: and R0227 used for their recall deciles.
DENSITY_DECILES = 10

#: `s` is divided by this, so it is dimensionless and comparable across maps.
SCALE_DEFINITION = "RMS Euclidean radius of the map about its own centroid"

CLUMP_DEFINITION = (
    f"occupancy histogram over {HEATMAP_BINS}x{HEATMAP_BINS} bins spanning the "
    f"{COORD_TRIM_PERCENTILE[0]}-{COORD_TRIM_PERCENTILE[1]} coordinate "
    f"percentiles; clump mask = bins at or above the "
    f"{CLUMP_DENSITY_PERCENTILE}th percentile of OCCUPIED bin counts; components "
    "under 8-connectivity (R0215's registered constants, imported)"
)
SCATTER_DEFINITION = (
    "s(r) = mean_j ||y_r - y_j|| over the row's 15 exact high-D neighbours from "
    "R0220's sealed truth, divided by the map's RMS radius about its centroid. "
    "Alignment-free and computed within a single map, because parametric UMAP is "
    "not identifiable across seeds and a cross-map per-row displacement measures "
    "seed noise rather than graph damage."
)
NULL_ARM_NOTE = (
    "the eight exact-graph maps are scored on the SAME row sets; they were "
    "trained on graphs where nothing was lost, so the gap they show between the "
    "lost and control populations is confounding, and the difference in "
    "differences against them is the damage attributable to the missing edges"
)


class Round0228GeometryError(RuntimeError):
    """The registered R0228 geometry contract changed."""


def map_scale(coordinates: np.ndarray) -> float:
    """RMS radius about the centroid. Positive for any non-degenerate map."""
    array = np.asarray(coordinates, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise Round0228GeometryError("R0228 geometry needs (n, 2) coordinates")
    centred = array - array.mean(axis=0, keepdims=True)
    scale = float(np.sqrt((centred ** 2).sum(axis=1).mean()))
    if not math.isfinite(scale) or scale <= 0.0:
        raise Round0228GeometryError("R0228 map has no spatial extent")
    return scale


def clump_profile(
    coordinates: np.ndarray,
    *,
    bins: int = HEATMAP_BINS,
    trim: Sequence[float] = COORD_TRIM_PERCENTILE,
    percentile: float = CLUMP_DENSITY_PERCENTILE,
) -> dict[str, Any]:
    """R0215's clump detector: heatmap, `>= p99.9` mask, connected components."""
    from scipy import ndimage

    array = np.asarray(coordinates, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise Round0228GeometryError("R0228 clump profile needs (n, 2) coordinates")
    lo_pct, hi_pct = float(trim[0]), float(trim[1])
    x_lo, x_hi = np.percentile(array[:, 0], [lo_pct, hi_pct])
    y_lo, y_hi = np.percentile(array[:, 1], [lo_pct, hi_pct])
    if not (x_hi > x_lo and y_hi > y_lo):
        raise Round0228GeometryError("R0228 trimmed coordinate range is degenerate")
    counts, _x_edges, _y_edges = np.histogram2d(
        array[:, 0],
        array[:, 1],
        bins=int(bins),
        range=[[float(x_lo), float(x_hi)], [float(y_lo), float(y_hi)]],
    )
    occupied = counts[counts > 0]
    if occupied.size == 0:
        raise Round0228GeometryError("R0228 heatmap has no occupied bins")
    threshold = float(np.percentile(occupied, float(percentile)))
    mask = counts >= threshold
    structure = np.ones((3, 3), dtype=bool)
    labels, components = ndimage.label(mask, structure=structure)
    if int(components) <= 0:
        raise Round0228GeometryError("R0228 clump mask has no components")
    flat_labels = labels.ravel()
    component_bins = np.bincount(flat_labels, minlength=int(components) + 1)[1:]
    component_rows = np.bincount(
        flat_labels, weights=counts.ravel(), minlength=int(components) + 1
    )[1:].astype(np.float64)
    clumped_rows = float(component_rows.sum())
    total_binned = float(counts.sum())
    return {
        "definition": CLUMP_DEFINITION,
        "bins": int(bins),
        "coordinate_trim_percentile": [lo_pct, hi_pct],
        "clump_density_percentile": float(percentile),
        "rows_inside_trimmed_range": total_binned,
        "occupied_bins": int(occupied.size),
        "median_occupied_bin_count": float(np.median(occupied)),
        "max_bin_count": float(counts.max()),
        "clump_threshold_count": threshold,
        "clump_bins": int(mask.sum()),
        "clump_components": int(components),
        "largest_component_bins": int(component_bins.max()),
        "largest_component_rows": float(component_rows.max()),
        "clumped_rows": clumped_rows,
        "clumped_row_fraction": clumped_rows / total_binned if total_binned else 0.0,
        "largest_component_share_of_clumped_rows": (
            float(component_rows.max()) / clumped_rows if clumped_rows else 0.0
        ),
    }


def true_neighbour_scatter(
    coordinates: np.ndarray,
    truth_ids: np.ndarray,
    rows: np.ndarray,
    *,
    scale: float | None = None,
) -> np.ndarray:
    """`s(r)` for the given rows, in units of the map's own RMS radius."""
    coords = np.asarray(coordinates, dtype=np.float32)
    selected = np.asarray(rows, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0:
        raise Round0228GeometryError("R0228 scatter needs a nonempty row selection")
    if int(selected.min()) < 0 or int(selected.max()) >= coords.shape[0]:
        raise Round0228GeometryError("R0228 scatter row selection is out of range")
    neighbours = np.asarray(truth_ids[selected], dtype=np.int64)
    if neighbours.ndim != 2:
        raise Round0228GeometryError("R0228 truth ids must be (n, k)")
    anchor = coords[selected][:, None, :]
    neighbour_xy = coords[neighbours.ravel()].reshape(
        neighbours.shape[0], neighbours.shape[1], 2
    )
    distances = np.sqrt(
        ((neighbour_xy.astype(np.float64) - anchor.astype(np.float64)) ** 2).sum(axis=2)
    )
    divisor = float(scale) if scale is not None else map_scale(coords)
    return np.asarray(distances.mean(axis=1) / divisor, dtype=np.float64)


def density_matched_control(
    *,
    lost_mask: np.ndarray,
    kth_cosine: np.ndarray,
    sample_rows: int = SCATTER_SAMPLE_ROWS,
    deciles: int = DENSITY_DECILES,
    seed: int = SCATTER_SAMPLE_SEED,
) -> dict[str, Any]:
    """Sample `lost` rows and a control matched on true-density deciles.

    Without the match, a scatter difference between rows that lost edges and rows
    that did not is partly a restatement of "sparse rows lose more edges", which
    R0226/R0227 already measured. The match removes that channel by construction.
    """
    lost = np.asarray(lost_mask, dtype=bool)
    cosine = np.asarray(kth_cosine, dtype=np.float64)
    if lost.shape != cosine.shape:
        raise Round0228GeometryError("R0228 control needs one cosine per row")
    lost_rows = np.flatnonzero(lost)
    kept_rows = np.flatnonzero(~lost)
    if lost_rows.size == 0:
        raise Round0228GeometryError(
            "R0228 control needs at least one row carrying loss"
        )
    if kept_rows.size == 0:
        raise Round0228GeometryError("R0228 control needs at least one intact row")
    edges = np.quantile(cosine, np.linspace(0.0, 1.0, int(deciles) + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    bin_of = np.digitize(cosine, edges[1:-1], right=False)

    rng = np.random.default_rng(int(seed))
    take = min(int(sample_rows), int(lost_rows.size))
    chosen_lost = np.sort(rng.choice(lost_rows, size=take, replace=False))
    wanted = np.bincount(bin_of[chosen_lost], minlength=int(deciles))
    control_parts: list[np.ndarray] = []
    shortfall: dict[str, int] = {}
    for index in range(int(deciles)):
        need = int(wanted[index])
        if need == 0:
            continue
        pool = kept_rows[bin_of[kept_rows] == index]
        if pool.size == 0:
            shortfall[str(index)] = need
            continue
        got = min(need, int(pool.size))
        if got < need:
            shortfall[str(index)] = need - got
        control_parts.append(rng.choice(pool, size=got, replace=False))
    if not control_parts:
        raise Round0228GeometryError("R0228 density-matched control is empty")
    chosen_control = np.sort(np.concatenate(control_parts))
    return {
        "lost_rows_total": int(lost_rows.size),
        "intact_rows_total": int(kept_rows.size),
        "sample_rows_requested": int(sample_rows),
        "lost_sample": chosen_lost,
        "control_sample": chosen_control,
        "deciles": int(deciles),
        "decile_counts_lost": [int(value) for value in wanted],
        "decile_counts_control": [
            int((bin_of[chosen_control] == index).sum()) for index in range(int(deciles))
        ],
        "control_shortfall_by_decile": shortfall,
        "matched_exactly": not shortfall,
        "seed": int(seed),
    }


def displacement_summary(
    *,
    lost_scatter: Mapping[str, Sequence[float]],
    control_scatter: Mapping[str, Sequence[float]],
    candidate_maps: Sequence[str],
    exact_maps: Sequence[str],
) -> dict[str, Any]:
    """Difference in differences: candidate gap minus exact-graph gap."""
    missing = [
        name
        for name in list(candidate_maps) + list(exact_maps)
        if name not in lost_scatter or name not in control_scatter
    ]
    if missing:
        raise Round0228GeometryError(f"R0228 displacement is missing maps {missing}")
    per_map = {
        name: {
            "lost": population_stats(list(lost_scatter[name])),
            "control": population_stats(list(control_scatter[name])),
            "gap_lost_minus_control": (
                statistics.fmean([float(v) for v in lost_scatter[name]])
                - statistics.fmean([float(v) for v in control_scatter[name]])
            ),
        }
        for name in list(candidate_maps) + list(exact_maps)
    }
    candidate_gaps = [per_map[name]["gap_lost_minus_control"] for name in candidate_maps]
    exact_gaps = [per_map[name]["gap_lost_minus_control"] for name in exact_maps]
    if len(candidate_gaps) < 2 or len(exact_gaps) < 2:
        raise Round0228GeometryError("R0228 displacement needs >= 2 maps per arm")
    candidate_mean = statistics.fmean(candidate_gaps)
    exact_mean = statistics.fmean(exact_gaps)
    exact_sd = statistics.stdev(exact_gaps)
    return {
        "definition": SCATTER_DEFINITION,
        "scale_definition": SCALE_DEFINITION,
        "null_arm_note": NULL_ARM_NOTE,
        "per_map": per_map,
        "candidate_maps": list(candidate_maps),
        "exact_maps": list(exact_maps),
        "candidate_gap_mean": candidate_mean,
        "candidate_gap_sd": statistics.stdev(candidate_gaps),
        "exact_gap_mean": exact_mean,
        "exact_gap_sd": exact_sd,
        "difference_in_differences": candidate_mean - exact_mean,
        "difference_in_differences_in_exact_sd": (
            (candidate_mean - exact_mean) / exact_sd if exact_sd > 0 else None
        ),
        "units": "map RMS radii",
    }


__all__ = [
    "CLUMP_DEFINITION",
    "CLUMP_DENSITY_PERCENTILE",
    "COORD_TRIM_PERCENTILE",
    "DENSITY_DECILES",
    "HEATMAP_BINS",
    "NULL_ARM_NOTE",
    "ROUND_ID",
    "Round0228GeometryError",
    "SCALE_DEFINITION",
    "SCATTER_DEFINITION",
    "SCATTER_SAMPLE_ROWS",
    "SCATTER_SAMPLE_SEED",
    "clump_profile",
    "density_matched_control",
    "displacement_summary",
    "map_scale",
    "population_stats",
    "true_neighbour_scatter",
]
