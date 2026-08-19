"""Two map-quality metrics the registered gates cannot see: COLLAPSE and FOG.

Both are CPU-only, deterministic given a seed, and safe on memmapped
coordinate files. They measure two *different* failure directions:

  COLLAPSE  the map has folded into point-like beads. Local neighbourhoods
            are far tighter than the map's own extent. Registered FFR
            actually *rewards* this (a bead makes every true neighbour a
            2D neighbour), so no registered metric can see it.

  FOG       the map is a diffuse haze: a large share of the point mass sits
            in bins holding less than 1% of the peak bin's count, i.e.
            between the clusters rather than in them.

A collapsed map has LOW fog (all the mass is in a few dense bins) and LOW
collapse statistic. A foggy map has HIGH fog and a normal collapse
statistic. Neither metric substitutes for the other; a gate needs both.

Protocol provenance
-------------------
COLLAPSE hardens ``tissue_metrics()`` in
``experiments/sandbox/heldout_eval.py`` and the inline copy in
``experiments/sandbox/knobs_2m.py::run_arm`` (both compute
``median(d_10) / p90 map radius`` over a 20k sample, rng seed 0).

The N-ADJUSTMENT is new here. In 2D at fixed occupied area the median
distance to the k-th nearest neighbour scales as 1/sqrt(N), so the raw
ratio is not comparable across map sizes. Multiplying by sqrt(N) removes
that scaling: healthy maps hold ``r10/R * sqrt(N) ~ 1.0-1.2`` across
2M -> 6.25M, while collapsed maps sit far below at an N-independent floor.

FOG reuses the exact binning of ``experiments/map_renders.py``
(``robust_extent`` percentile extent with 2% pad, ``binned_counts`` 1024x1024
histogram with edge clipping) and the ``low_density_mass_fraction``
definition from ``heldout_eval.tissue_metrics``.

Memory rule
-----------
Coordinate files are (N, 2) fp32 and may be memmapped. Nothing here
materialises the full array unless ``N * 16`` bytes stays under 256 MB
(``N * 16`` is what ``scipy.spatial.cKDTree`` costs, since it promotes to
float64 internally). Above that threshold the collapse statistic is
measured on a seeded uniform subsample of ``MAX_TREE_ROWS`` rows and
adjusted with ``sqrt(n_effective)`` -- which is exactly what the
sqrt(N) adjustment is for, so the statistic is unchanged in expectation.
FOG never materialises the array at all: extent and histogram are both
chunked passes over the memmap.

No GPU. No imports of torch, cuml, or faiss.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

__all__ = [
    "MADN_CONSISTENCY",
    "MAX_MATERIALIZE_BYTES",
    "MAX_TREE_ROWS",
    "load_coords",
    "map_collapse",
    "map_fog",
    "madn",
    "robust_floor",
    "robust_ceiling",
]

# --- protocol constants -------------------------------------------------

#: scaled median absolute deviation consistency constant for the normal
#: distribution; the program's established robust scale (R0231/R0234/R0255).
MADN_CONSISTENCY = 1.4826

#: memory ceiling for materialising a coordinate array (see module docstring).
MAX_MATERIALIZE_BYTES = 256 * 1024 * 1024
#: N above which the collapse statistic switches to a seeded subsample.
MAX_TREE_ROWS = MAX_MATERIALIZE_BYTES // 16  # 16_777_216

#: chunk size for streaming passes over a memmap; matches map_renders.py.
CHUNK_ROWS = 2_000_000
#: rows sampled (by stride) when estimating the percentile extent; matches
#: map_renders.robust_extent(sample_rows=...).
EXTENT_SAMPLE_ROWS = 2_000_000

COLLAPSE_K = 10          # the 10th nearest neighbour
COLLAPSE_RADIUS_PCT = 90  # p90 radius about the centroid = "map radius"


# --- io -----------------------------------------------------------------

def load_coords(path) -> np.memmap:
    """Memmap an (N, 2) coordinate .npy. Never reads the payload."""
    arr = np.load(str(path), mmap_mode="r")
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{path}: expected (N, 2) coordinates, got {arr.shape}")
    return arr


def _as_2d(xy) -> Any:
    arr = xy
    if not hasattr(arr, "shape"):
        arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"expected (N, 2) coordinates, got shape {arr.shape}")
    return arr


# --- robust statistics (shared with the calibrator) ---------------------

def madn(values: Sequence[float] | np.ndarray, axis: int = -1) -> np.ndarray | float:
    """Scaled median absolute deviation, consistency constant 1.4826."""
    v = np.asarray(values, dtype=np.float64)
    centre = np.median(v, axis=axis, keepdims=True)
    return MADN_CONSISTENCY * np.median(np.abs(v - centre), axis=axis)


def robust_floor(values: Sequence[float] | np.ndarray, k: float) -> float:
    """One-sided lower floor: ``median - k * MAD_n``."""
    v = np.asarray(values, dtype=np.float64)
    return float(np.median(v) - k * madn(v))


def robust_ceiling(values: Sequence[float] | np.ndarray, k: float) -> float:
    """One-sided upper ceiling: ``median + k * MAD_n``."""
    v = np.asarray(values, dtype=np.float64)
    return float(np.median(v) + k * madn(v))


# --- COLLAPSE -----------------------------------------------------------

def map_collapse(
    xy,
    sample_size: int = 20_000,
    rng_seed: int = 0,
    k_neighbor: int = COLLAPSE_K,
    radius_pct: int = COLLAPSE_RADIUS_PCT,
    workers: int = 8,
    max_tree_rows: int = MAX_TREE_ROWS,
) -> dict:
    """Bead-collapse statistic for a 2D map.

    Returns a dict carrying the raw ratio ``r10 / R``, the N-adjusted gate
    statistic ``r10 / R * sqrt(n_effective)``, and every protocol parameter
    needed to reproduce the number.

    The gate direction is ONE-SIDED FROM BELOW on
    ``r10_over_radius_times_sqrt_n``: too small means collapsed.
    """
    from scipy.spatial import cKDTree

    arr = _as_2d(xy)
    n_rows = int(arr.shape[0])
    if n_rows <= k_neighbor + 1:
        raise ValueError(
            f"need more than {k_neighbor + 1} rows for a {k_neighbor}-NN statistic, got {n_rows}"
        )

    rng = np.random.default_rng(rng_seed)

    # Base set: the full map when it fits the memory rule, otherwise a seeded
    # uniform subsample. The sqrt(n) adjustment makes the two commensurate.
    subsampled = n_rows > max_tree_rows
    if subsampled:
        base_idx = np.sort(rng.choice(n_rows, size=max_tree_rows, replace=False))
        base = np.asarray(arr[base_idx], dtype=np.float32)
    else:
        base = np.asarray(arr, dtype=np.float32)
    n_eff = int(base.shape[0])

    # Query sample drawn from the base set (so the k-th neighbour is measured
    # within the same point process the tree indexes). Indices are sorted:
    # the statistic is an order-free median, so sorting only speeds the read.
    n_sample = int(min(sample_size, n_eff))
    sample_idx = np.sort(rng.choice(n_eff, size=n_sample, replace=False))
    sample = base[sample_idx]

    tree = cKDTree(base)
    # k_neighbor + 1 because every query point is itself in the tree at d = 0.
    dists, _ = tree.query(sample, k=k_neighbor + 1, workers=workers)
    r_k = float(np.median(dists[:, k_neighbor]))

    centroid = base.mean(axis=0)
    radius = float(np.percentile(np.linalg.norm(base - centroid, axis=1), radius_pct))

    ratio = r_k / radius if radius > 0 else float("nan")
    return {
        "metric": "collapse",
        "statistic": "r10_over_radius_times_sqrt_n",
        "gate_direction": "one_sided_lower_floor",
        "n_rows": n_rows,
        "n_effective": n_eff,
        "subsampled_for_tree": bool(subsampled),
        "max_tree_rows": int(max_tree_rows),
        "sample_size": n_sample,
        "rng_seed": int(rng_seed),
        "k_neighbor": int(k_neighbor),
        "radius_pct": int(radius_pct),
        "workers": int(workers),
        "r10_median": r_k,
        "map_radius_p90": radius,
        "r10_over_map_radius_median": ratio,
        "r10_over_radius_times_sqrt_n": ratio * math.sqrt(n_eff),
    }


# --- FOG ----------------------------------------------------------------

def _robust_extent(arr, extent_pct: tuple[float, float], sample_rows: int) -> list[float]:
    """Percentile extent with 2% pad; one outlier island cannot shrink it.

    Byte-for-byte the rule in ``experiments/map_renders.robust_extent``.
    """
    n = int(arr.shape[0])
    if n > sample_rows:
        idx = np.linspace(0, n - 1, sample_rows).astype(np.int64)
        pts = np.asarray(arr[idx], dtype=np.float32)
    else:
        pts = np.asarray(arr, dtype=np.float32)
    lo, hi = extent_pct
    x0, x1 = np.percentile(pts[:, 0], [lo, hi])
    y0, y1 = np.percentile(pts[:, 1], [lo, hi])
    pad_x = 0.02 * (x1 - x0) or 1.0
    pad_y = 0.02 * (y1 - y0) or 1.0
    return [float(x0 - pad_x), float(x1 + pad_x), float(y0 - pad_y), float(y1 + pad_y)]


def _binned_counts(arr, extent: list[float], bins: int) -> np.ndarray:
    """Chunked 2D histogram; off-extent points clip into the edge bins.

    Byte-for-byte the rule in ``experiments/map_renders.binned_counts``.
    """
    x0, x1, y0, y1 = extent
    edges_x = np.linspace(x0, x1, bins + 1)
    edges_y = np.linspace(y0, y1, bins + 1)
    counts = np.zeros((bins, bins), dtype=np.int64)
    for i in range(0, int(arr.shape[0]), CHUNK_ROWS):
        chunk = np.asarray(arr[i:i + CHUNK_ROWS], dtype=np.float32)
        cx = np.clip(chunk[:, 0], x0, x1)
        cy = np.clip(chunk[:, 1], y0, y1)
        h, _, _ = np.histogram2d(cx, cy, bins=[edges_x, edges_y])
        counts += h.astype(np.int64)
    return counts


def map_fog(
    xy,
    bins: int = 1024,
    extent_pct: tuple[float, float] = (0.1, 99.9),
    threshold: float = 0.01,
) -> dict:
    """Diffuse-haze statistic: point mass sitting in near-empty bins.

    ``fog`` is the fraction of total binned mass in *occupied* bins whose
    count is below ``threshold * peak_bin_count`` (with an absolute floor of
    1 count, so a map whose peak bin is tiny cannot report spurious fog).

    The gate direction is ONE-SIDED FROM ABOVE: too large means haze.
    Deterministic -- the extent sample is a fixed stride, no rng.

    Two validity diagnostics ship with every result, because the inherited
    definition has a hard degeneracy:

    ``resolution_levels``   Bin counts are integers, so the number of usable
        "low density" levels is ``floor(threshold * peak)``. When the peak
        bin holds fewer than ``1 / threshold`` points (100 at the default
        1%), NO occupied bin can fall below the cutoff and fog is identically
        0.0000 regardless of how hazy the map is. ``degenerate`` flags that.
        Measured examples: the cuML 1M reference has peak = 198, i.e. ONE
        usable level -- its celebrated fog of 0.040 sits a single integer
        count above the degeneracy. The sandbox arm ``umap-md035-x2`` has
        peak = 99 and reports fog 0.0000 while being one of the haziest maps
        in the sandbox.

    ``peak_bin_count`` itself is a corpus property as much as a map property:
    on the 2M mixed MiniLM substrate the peak bin of every umap-kernel arm is
    the same exact-duplicate group of 1377 identical rows, so the 1% cutoff
    is pinned by duplicate text rather than by the map's densest tissue.

    Neither is fixed here -- changing the definition would break
    comparability with the sandbox and cuML reference values -- but both are
    reported so a caller can refuse a degenerate measurement.
    """
    arr = _as_2d(xy)
    n_rows = int(arr.shape[0])
    if n_rows == 0:
        raise ValueError("empty coordinate array")

    extent = _robust_extent(arr, extent_pct, EXTENT_SAMPLE_ROWS)
    counts = _binned_counts(arr, extent, bins)

    total = int(counts.sum())
    occupied = counts > 0
    peak = int(counts.max())
    cutoff_raw = peak * threshold
    cutoff = max(cutoff_raw, 1.0)
    low = occupied & (counts < cutoff)
    fog = float(counts[low].sum() / total) if total else float("nan")
    resolution_levels = int(math.floor(cutoff_raw))

    return {
        "metric": "fog",
        "statistic": "low_density_mass_fraction",
        "gate_direction": "one_sided_upper_ceiling",
        "n_rows": n_rows,
        "bins": int(bins),
        "extent_pct": [float(extent_pct[0]), float(extent_pct[1])],
        "threshold": float(threshold),
        "extent": extent,
        "total_points_binned": total,
        "peak_bin_count": peak,
        "low_bin_cutoff": float(cutoff),
        "low_bin_cutoff_raw": float(cutoff_raw),
        "absolute_floor_binding": bool(cutoff_raw < 1.0),
        "resolution_levels": resolution_levels,
        "degenerate": bool(resolution_levels < 1),
        "occupied_bins": int(occupied.sum()),
        "occupied_bin_fraction": float(occupied.mean()),
        "low_density_bins": int(low.sum()),
        "low_density_mass_fraction": fog,
        "fog": fog,
    }


def map_quality(xy, **kwargs) -> dict:
    """Both metrics over one coordinate array (single memmap, two passes)."""
    collapse_kw = {k: v for k, v in kwargs.items() if k in
                   ("sample_size", "rng_seed", "k_neighbor", "radius_pct",
                    "workers", "max_tree_rows")}
    fog_kw = {k: v for k, v in kwargs.items() if k in
              ("bins", "extent_pct", "threshold")}
    return {"collapse": map_collapse(xy, **collapse_kw), "fog": map_fog(xy, **fog_kw)}
