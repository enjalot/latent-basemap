"""density_v3 — repaired local-crowding agreement metric (CPU-only).

WHAT THE METRIC MEASURES (unchanged from density_v2)
---------------------------------------------------
For a set of *anchor* rows we compare how crowded each anchor's neighbourhood
is in the source space against how crowded it is on the map:

    r_hd(a) = mean Euclidean distance from anchor ``a`` to its k=15 nearest
              self-excluded neighbours in the high-D substrate
    r_2d(a) = the same quantity computed on the 2-D map coordinates

A map that preserves local density structure makes ``r_hd`` and ``r_2d`` agree
monotonically across anchors.  ``density_v2`` scored that agreement as

    Pearson( log(r_hd + 1e-12), log(r_2d + 1e-12) )                    (v2)

over a fixed anchor set drawn by ``np.random.RandomState(seed).choice``.

THE DEFECT THIS MODULE REPAIRS
------------------------------
Review 0225 (``latent-labs/basemap-100m/review-0225-2026-08-08-01.md``) showed
that the 4,000-anchor v2 population for the ``minilm-mixed-2m`` universe contains
exactly one anchor — substrate row ``1449227``, which has 1,377 exact duplicate
rows — whose ``r_hd`` is exactly ``0``.  Its duplicates also collapse onto one
map coordinate, so ``r_2d`` is exactly ``0`` too.  With ``eps = 1e-12`` that
anchor sits at ``(-27.63, -27.63)``: ~27 log-units away from the other 3,999
points on both axes, on the identity diagonal.  It is a perfect leverage point.

Consequences measured on the eight defining cells:

    family mean       0.4421  ->  0.1544   when that one anchor is dropped
    2 sigma / mean     1.93%  ->  26.03%
    cell ranking      Pearson -0.047 between the two versions

i.e. **one anchor in four thousand supplied ~two thirds of the metric's value**,
and because it was byte-identical across cells it acted as a large additive
constant that compressed all between-cell variation.  ``density_v2`` was
therefore demoted to descriptive-only.

THE v3 REPAIR
-------------
Three changes, all in the *anchor policy* and the *statistic*, none in the
underlying quantity being measured:

1. **Degenerate anchors are excluded at source.**  An anchor is eligible only
   if ``r_hd > EPS_HD`` (default ``1e-3``).  The threshold is not a knife edge:
   on both measured universes the ``r_hd`` distribution has an empty band three
   orders of magnitude wide between the duplicate cluster and the real corpus.
   Measured on a 10,000-row pool of ``minilm-mixed-2m``, the nine smallest radii
   are ``0 (x6), 2.9e-5, 2.3e-4, 2.3e-4`` and then the tenth is ``0.188``;
   on the ``cuml-1m`` subsample they are ``0 (x5), 5.5e-4`` and then ``0.051``.
   ``1e-3`` sits inside that gap in both, so exact duplicates *and* the
   float-noise near-duplicates they drag along are removed, and nothing else is.
   The exclusion is a property of the *source space alone*, so the anchor set is
   identical for every map of a given substrate — maps stay comparable.

2. **A larger, deterministic anchor set.**  ``n_anchors`` defaults to 8,000
   (2x v2).  Drawing rule, fully documented and seeded:

       pool    = sorted( Generator(PCG64(anchor_seed)).choice(
                           N, size=ceil(n_anchors * POOL_FACTOR),
                           replace=False) )
       anchors = the first ``n_anchors`` members of ``pool``, in ascending row
                 order, that satisfy ``r_hd > EPS_HD``

   The 1.25x pool oversample exists purely so that removing degenerate rows
   still leaves a full-size anchor set; if it does not, the shortfall is
   reported rather than silently topped up (topping up would make the anchor
   set depend on the degeneracy rate in a way that is hard to audit).

3. **A rank statistic is primary.**  ``spearman`` is the reported ``value``.
   Spearman is invariant to any monotone transform of either radius, so it
   cannot be moved by a single anchor's *magnitude* — only by its *ordinal
   position*, which is bounded by ``1/n``.  ``pearson_log`` is kept for
   continuity with v2, but two things change: the ``1e-12`` log floor is gone,
   and both log-radius vectors are **winsorized** at the ``[WINSOR_Q,
   1 - WINSOR_Q]`` quantiles (default 0.1%, i.e. 8 anchors per tail at
   n = 8,000) before the correlation.  This is needed because degeneracy also
   appears on the *map* side: 2-3 anchors per 2M map are non-degenerate in
   high-D but land on an exactly-duplicated 2-D coordinate, so ``r_2d == 0``.
   A map is allowed to collapse points on top of each other — that is real,
   measurable behaviour, and it still costs the map rank — but it may not
   thereby manufacture a -27 log-unit leverage point.  Without winsorization
   the v3 ``pearson_log`` still moves 6-17% under leave-one-out; with it, both
   statistics stay under 1%.

Both statistics ship with a full leave-one-anchor-out sweep, so the sensitivity
that broke v2 is a *reported number* rather than something a later reviewer has
to rediscover.

INPUT MODES
-----------
``density_v3(xy, substrate_or_radii, ...)`` accepts high-D information as:

  (a) a precomputed per-row radius array, shape ``(N,)`` — one ``r_hd`` per
      substrate row (mode ``"radii_all_rows"``); or a ``(anchor_ids, radii)``
      pair for an explicit anchor population (mode ``"radii_explicit"``);
  (b) a substrate array / memmap / ``.npy`` path of shape ``(N, D)``, ``D > 2``
      (mode ``"substrate"``).  Anchor radii are then computed exactly on CPU by
      chunked brute force: a BLAS matmul selects ``k + OVERSELECT + 1``
      candidates per corpus chunk (exact for unit-norm rows, where squared
      Euclidean distance is a monotone function of the inner product), then the
      candidate vectors are gathered and their distances recomputed by
      per-dimension accumulation in fp32.  The per-dimension rerank is not
      optional: the ``2 - 2s`` matmul shortcut catastrophically cancels for
      duplicate rows and reports ~1e-4 where the true distance is 0, which is
      exactly the degeneracy this metric must detect.

The substrate is never materialised — only bounded chunk slices of the memmap
are read.  Measured cost on gsv (Ryzen 9 9950X, 8 BLAS threads) for a 10,000
anchor pool against 2,000,000 x 384 fp32 rows: ~75 s wall, ~1.3 GB peak.

The 2-D pass uses ``scipy.spatial.cKDTree`` (exact Euclidean) rather than a
matmul, for the same cancellation reason.

This module is CPU-only by construction: it imports nothing from ``basemap``
and touches no CUDA API.  Run it with ``CUDA_VISIBLE_DEVICES=""``.
"""
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata

__all__ = [
    "K_DENSITY",
    "EPS_HD",
    "DEFAULT_N_ANCHORS",
    "POOL_FACTOR",
    "WINSOR_Q",
    "DensityV3Error",
    "draw_anchor_pool",
    "high_d_radii",
    "low_d_radii",
    "density_v3",
    "density_v2_legacy",
]

K_DENSITY = 15
OVERSELECT = 8
EPS_HD = 1e-3
DEFAULT_N_ANCHORS = 8_000
POOL_FACTOR = 1.25
WINSOR_Q = 0.001
LEGACY_LOG_EPSILON = 1e-12
DEFAULT_THREADS = 8
DEFAULT_CORPUS_CHUNK = 200_000
DEFAULT_ANCHOR_TILE = 1_250


class DensityV3Error(RuntimeError):
    """A density_v3 contract was violated."""


# ── small statistics ────────────────────────────────────────────────────────

def _check_pair(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 1 or right.shape != left.shape or len(left) < 3:
        raise DensityV3Error("correlation inputs are malformed")
    if not (np.isfinite(left).all() and np.isfinite(right).all()):
        raise DensityV3Error("correlation inputs are not finite")
    return left, right


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    """Pearson r, with an explicit variance-collapse failure."""
    left, right = _check_pair(left, right)
    left = left - left.mean()
    right = right - right.mean()
    denominator = math.sqrt(float(left @ left) * float(right @ right))
    if not denominator > 0.0 or not math.isfinite(denominator):
        raise DensityV3Error("Pearson input variance collapsed")
    value = float((left @ right) / denominator)
    if not math.isfinite(value):
        raise DensityV3Error("Pearson correlation is nonfinite")
    return value


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman rho = Pearson on average-tie ranks."""
    left, right = _check_pair(left, right)
    return pearson(rankdata(left), rankdata(right))


def _loo_pearson(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Exact leave-one-out Pearson for every index, vectorised O(n)."""
    left, right = _check_pair(left, right)
    n = len(left)
    sx, sy = left.sum(), right.sum()
    sxx, syy, sxy = float(left @ left), float(right @ right), float(left @ right)
    m = n - 1
    ax, ay = sx - left, sy - right
    axx, ayy, axy = sxx - left * left, syy - right * right, sxy - left * right
    cov = axy - ax * ay / m
    vx = axx - ax * ax / m
    vy = ayy - ay * ay / m
    denominator = np.sqrt(np.maximum(vx, 0.0) * np.maximum(vy, 0.0))
    if not np.all(denominator > 0.0):
        raise DensityV3Error("leave-one-out Pearson variance collapsed")
    return cov / denominator


def _loo_spearman(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Exact leave-one-out Spearman for every index (re-ranks each time)."""
    left, right = _check_pair(left, right)
    n = len(left)
    keep = np.ones(n, dtype=bool)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        keep[i] = False
        out[i] = pearson(rankdata(left[keep]), rankdata(right[keep]))
        keep[i] = True
    return out


def _loo_block(value: float, loo: np.ndarray) -> dict[str, Any]:
    shift = np.abs(loo - value)
    worst = int(np.argmax(shift))
    scale = max(abs(value), 1e-12)
    return {
        "max_absolute_shift": float(shift.max()),
        "max_relative_shift": float(shift.max() / scale),
        "worst_anchor_position": worst,
        "value_without_worst_anchor": float(loo[worst]),
        "mean_absolute_shift": float(shift.mean()),
    }


def _winsorize(
    values: np.ndarray, q: float
) -> tuple[np.ndarray, tuple[float, float], tuple[int, int]]:
    """Clip ``values`` to their ``[q, 1-q]`` quantiles, keeping them positive.

    Returns ``(clipped, (lo, hi), (n_low, n_high))``.  If the lower quantile is
    non-positive (a map that collapsed more than ``q`` of the anchors onto
    duplicate coordinates) the bound falls back to the smallest strictly
    positive value, so ``log`` stays finite and the collapse still costs the map
    rank rather than being silently rescued.
    """
    values = np.asarray(values, dtype=np.float64)
    if not 0.0 <= q < 0.5:
        raise DensityV3Error("winsor_q must be in [0, 0.5)")
    lo = float(np.quantile(values, q)) if q > 0 else float(values.min())
    hi = float(np.quantile(values, 1.0 - q)) if q > 0 else float(values.max())
    if lo <= 0.0:
        positive = values[values > 0]
        if not len(positive):
            raise DensityV3Error("all radii are zero; the map is degenerate")
        lo = float(positive.min())
    if not hi > lo:
        raise DensityV3Error("winsorization bounds collapsed")
    return (
        np.clip(values, lo, hi),
        (lo, hi),
        (int((values < lo).sum()), int((values > hi).sum())),
    )


# ── anchor policy ───────────────────────────────────────────────────────────

def draw_anchor_pool(
    n_rows: int,
    n_anchors: int = DEFAULT_N_ANCHORS,
    anchor_seed: int = 0,
    pool_factor: float = POOL_FACTOR,
) -> np.ndarray:
    """Deterministic candidate pool: ``ceil(n_anchors * pool_factor)`` distinct
    rows drawn without replacement by ``Generator(PCG64(anchor_seed))``,
    returned in ascending row order.  Depends only on
    ``(n_rows, n_anchors, anchor_seed, pool_factor)``."""
    if n_rows < 3 or n_anchors < 3:
        raise DensityV3Error("anchor draw needs n_rows >= 3 and n_anchors >= 3")
    size = min(int(n_rows), int(math.ceil(n_anchors * pool_factor)))
    rng = np.random.default_rng(anchor_seed)
    return np.sort(rng.choice(int(n_rows), size=size, replace=False)).astype(np.int64)


# ── radii ───────────────────────────────────────────────────────────────────

def _set_blas_threads(threads: int | None) -> None:
    if threads is None:
        return
    try:  # optional; env vars are the documented fallback
        from threadpoolctl import threadpool_limits
    except ImportError:
        return
    threadpool_limits(limits=int(threads))


def low_d_radii(
    xy: np.ndarray,
    anchor_ids: np.ndarray,
    k: int = K_DENSITY,
    threads: int = DEFAULT_THREADS,
) -> np.ndarray:
    """Mean Euclidean distance from each anchor to its ``k`` nearest
    self-excluded rows of ``xy``.  Exact (cKDTree, no matmul cancellation).

    Self-exclusion follows the panel_v2 convention: query ``k + 1`` neighbours,
    drop the entry whose row id equals the anchor's, and take the first ``k`` of
    what remains.  When the anchor has exact coordinate duplicates the tree may
    tie-break to a duplicate instead of the anchor itself; the returned distance
    is 0 either way, so the radius is unchanged.
    """
    coords = np.asarray(xy, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise DensityV3Error("xy must be (N, d>=2)")
    anchor_ids = np.asarray(anchor_ids, dtype=np.int64)
    if anchor_ids.ndim != 1 or len(anchor_ids) == 0:
        raise DensityV3Error("anchor_ids must be a non-empty 1-D array")
    if anchor_ids.min() < 0 or anchor_ids.max() >= len(coords):
        raise DensityV3Error("anchor_ids out of range for xy")
    if len(coords) <= k:
        raise DensityV3Error("xy has too few rows for k neighbours")
    tree = cKDTree(coords)
    dist, idx = tree.query(coords[anchor_ids], k=k + 1, workers=int(threads))
    out = np.empty(len(anchor_ids), dtype=np.float64)
    for position in range(len(anchor_ids)):
        keep = idx[position] != anchor_ids[position]
        row = dist[position][keep]
        out[position] = (row[:k] if len(row) >= k else dist[position][:k]).mean()
    if not np.isfinite(out).all() or np.any(out < 0):
        raise DensityV3Error("low-D radii are malformed")
    return out


def high_d_radii(
    substrate: Any,
    anchor_ids: np.ndarray,
    k: int = K_DENSITY,
    *,
    corpus_chunk: int = DEFAULT_CORPUS_CHUNK,
    anchor_tile: int = DEFAULT_ANCHOR_TILE,
    overselect: int = OVERSELECT,
    threads: int = DEFAULT_THREADS,
    assume_unit_norm: bool = True,
    progress: Any = None,
) -> np.ndarray:
    """Exact CPU mean-k-NN radius for ``anchor_ids`` against the full substrate.

    Two-stage, matching the panel_v2 convention:
      1. candidate selection by chunked BLAS matmul (top ``k + overselect + 1``
         by inner product; exact ordering for unit-norm rows);
      2. exact rerank by per-dimension fp32 accumulation over the gathered
         candidate vectors, which is the only stage allowed to decide whether a
         radius is zero.

    ``substrate`` may be an ndarray, an ``np.memmap``, or a path to a ``.npy``
    file (opened with ``mmap_mode='r'``).  Only bounded chunks are read into
    memory; the full substrate is never materialised.
    """
    if isinstance(substrate, (str, os.PathLike)):
        substrate = np.load(substrate, mmap_mode="r", allow_pickle=False)
    if getattr(substrate, "ndim", 0) != 2 or substrate.shape[1] < 3:
        raise DensityV3Error("substrate must be (N, D) with D >= 3")
    anchor_ids = np.asarray(anchor_ids, dtype=np.int64)
    n_rows, dims = int(substrate.shape[0]), int(substrate.shape[1])
    if anchor_ids.ndim != 1 or len(anchor_ids) == 0:
        raise DensityV3Error("anchor_ids must be a non-empty 1-D array")
    if anchor_ids.min() < 0 or anchor_ids.max() >= n_rows:
        raise DensityV3Error("anchor_ids out of range for substrate")
    if n_rows <= k:
        raise DensityV3Error("substrate has too few rows for k neighbours")
    _set_blas_threads(threads)

    m = len(anchor_ids)
    cand = min(n_rows, k + int(overselect) + 1)
    queries = np.ascontiguousarray(np.asarray(substrate[anchor_ids], dtype=np.float32))
    if assume_unit_norm:
        norms = np.linalg.norm(queries, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise DensityV3Error(
                "assume_unit_norm=True but anchor rows are not unit-norm"
            )
    best_score = np.full((m, cand), -np.inf, dtype=np.float32)
    best_index = np.full((m, cand), -1, dtype=np.int64)
    chunk = max(1, int(corpus_chunk))
    tile = max(1, int(anchor_tile))

    for start in range(0, n_rows, chunk):
        stop = min(start + chunk, n_rows)
        block = np.ascontiguousarray(np.asarray(substrate[start:stop], dtype=np.float32))
        if not assume_unit_norm:
            block_sq = (block * block).sum(1)
        for t0 in range(0, m, tile):
            t1 = min(t0 + tile, m)
            scores = queries[t0:t1] @ block.T           # (tile, chunk)
            if not assume_unit_norm:
                # rank by -d2/2 so the "largest score" convention still holds
                scores = scores - 0.5 * block_sq[None, :]
            take = min(cand, stop - start)
            part = np.argpartition(scores, -take, axis=1)[:, -take:]
            local_score = np.take_along_axis(scores, part, axis=1)
            local_index = part.astype(np.int64) + start
            merged_score = np.concatenate([best_score[t0:t1], local_score], axis=1)
            merged_index = np.concatenate([best_index[t0:t1], local_index], axis=1)
            sel = np.argpartition(merged_score, -cand, axis=1)[:, -cand:]
            best_score[t0:t1] = np.take_along_axis(merged_score, sel, axis=1)
            best_index[t0:t1] = np.take_along_axis(merged_index, sel, axis=1)
            del scores, part, local_score, local_index, merged_score, merged_index, sel
        del block
        if progress is not None:
            progress(stop, n_rows)

    # exact rerank: per-dimension accumulation, never `2 - 2s`
    out = np.empty(m, dtype=np.float64)
    rerank_tile = max(1, min(m, int(2_000_000_000 // (cand * dims * 4 * 3))))
    for t0 in range(0, m, rerank_tile):
        t1 = min(t0 + rerank_tile, m)
        flat = best_index[t0:t1].reshape(-1)
        if np.any(flat < 0):
            raise DensityV3Error("candidate selection left empty slots")
        neigh = np.asarray(substrate[flat], dtype=np.float32).reshape(t1 - t0, cand, dims)
        diff = neigh - queries[t0:t1][:, None, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff, dtype=np.float32)
        dist = np.sqrt(np.maximum(d2.astype(np.float64), 0.0))
        order = np.argsort(dist, axis=1, kind="stable")
        dist = np.take_along_axis(dist, order, axis=1)
        ids = np.take_along_axis(best_index[t0:t1], order, axis=1)
        for position in range(t1 - t0):
            keep = ids[position] != anchor_ids[t0 + position]
            row = dist[position][keep]
            if len(row) < k:
                row = dist[position][:k]
            out[t0 + position] = row[:k].mean()
        del neigh, diff, d2, dist, order, ids
    if not np.isfinite(out).all() or np.any(out < 0):
        raise DensityV3Error("high-D radii are malformed")
    return out


# ── the metric ──────────────────────────────────────────────────────────────

def _resolve_high_d(
    substrate_or_radii: Any,
    n_rows: int,
    n_anchors: int,
    anchor_seed: int,
    pool_factor: float,
    k: int,
    threads: int,
    radii_kwargs: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return ``(pool_ids, pool_r_hd, mode)``."""
    if isinstance(substrate_or_radii, tuple):
        ids, radii = substrate_or_radii
        ids = np.asarray(ids, dtype=np.int64)
        radii = np.asarray(radii, dtype=np.float64)
        if ids.shape != radii.shape or ids.ndim != 1:
            raise DensityV3Error("(anchor_ids, radii) must be matching 1-D arrays")
        order = np.argsort(ids, kind="stable")
        return ids[order], radii[order], "radii_explicit"

    if isinstance(substrate_or_radii, (str, os.PathLike)):
        substrate_or_radii = np.load(substrate_or_radii, mmap_mode="r", allow_pickle=False)

    array = substrate_or_radii
    if getattr(array, "ndim", None) == 1:
        radii = np.asarray(array, dtype=np.float64)
        if len(radii) != n_rows:
            raise DensityV3Error("per-row radii length must equal len(xy)")
        pool = draw_anchor_pool(n_rows, n_anchors, anchor_seed, pool_factor)
        return pool, radii[pool], "radii_all_rows"

    if getattr(array, "ndim", None) == 2 and array.shape[1] > 2:
        if len(array) != n_rows:
            raise DensityV3Error("substrate row count must equal len(xy)")
        pool = draw_anchor_pool(n_rows, n_anchors, anchor_seed, pool_factor)
        radii = high_d_radii(array, pool, k=k, threads=threads, **radii_kwargs)
        return pool, radii, "substrate"

    raise DensityV3Error(
        "substrate_or_radii must be (N,) radii, (N,D>2) substrate, "
        "an .npy path, or an (anchor_ids, radii) tuple"
    )


def density_v3(
    xy: np.ndarray,
    substrate_or_radii: Any,
    anchor_seed: int = 0,
    n_anchors: int = DEFAULT_N_ANCHORS,
    *,
    k: int = K_DENSITY,
    eps_hd: float = EPS_HD,
    pool_factor: float = POOL_FACTOR,
    winsor_q: float = WINSOR_Q,
    threads: int = DEFAULT_THREADS,
    leave_one_out: bool = True,
    radii_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Repaired density metric.  See the module docstring for the full policy.

    Returns a dict with (at least)::

        spearman            primary value; rank agreement of r_hd vs r_2d
        pearson_log         v2-continuity value on clamped log radii
        n_anchors           anchors actually scored
        n_excluded_degenerate_hd   pool rows dropped for r_hd <= eps_hd
        leave_one_out       per-statistic max absolute / relative shift
    """
    coords = np.asarray(xy)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise DensityV3Error("xy must be (N, d>=2)")
    n_rows = len(coords)
    pool_ids, pool_r_hd, mode = _resolve_high_d(
        substrate_or_radii, n_rows, n_anchors, anchor_seed, pool_factor,
        k, threads, radii_kwargs or {},
    )

    if np.any(pool_r_hd < 0) or not np.isfinite(pool_r_hd).all():
        raise DensityV3Error("high-D radii are malformed")

    eligible = pool_r_hd > float(eps_hd)
    n_degenerate = int((~eligible).sum())
    kept_ids = pool_ids[eligible][:n_anchors]
    kept_r_hd = pool_r_hd[eligible][:n_anchors]
    if len(kept_ids) < 3:
        raise DensityV3Error("fewer than three eligible anchors survive")

    r_2d = low_d_radii(coords, kept_ids, k=k, threads=threads)

    hd_w, hd_bounds, hd_counts = _winsorize(kept_r_hd, winsor_q)
    ld_w, ld_bounds, ld_counts = _winsorize(r_2d, winsor_q)
    log_hd = np.log(hd_w)
    log_2d = np.log(ld_w)
    value_spearman = spearman(kept_r_hd, r_2d)
    value_pearson = pearson(log_hd, log_2d)

    result: dict[str, Any] = {
        "schema": "density-v3-2026-08-13",
        "spearman": value_spearman,
        "pearson_log": value_pearson,
        "value": value_spearman,
        "primary_statistic": "spearman",
        "k": int(k),
        "anchor_seed": int(anchor_seed),
        "n_anchors_requested": int(n_anchors),
        "n_anchors": int(len(kept_ids)),
        "n_pool": int(len(pool_ids)),
        "pool_factor": float(pool_factor),
        "n_excluded_degenerate_hd": n_degenerate,
        "excluded_rows_sample": [int(v) for v in pool_ids[~eligible][:16]],
        "anchor_shortfall": int(max(0, n_anchors - len(kept_ids))),
        "eps_hd": float(eps_hd),
        "winsor_q": float(winsor_q),
        "winsorization": {
            "r_hd_bounds": [float(v) for v in hd_bounds],
            "r_hd_n_low": hd_counts[0], "r_hd_n_high": hd_counts[1],
            "r_2d_bounds": [float(v) for v in ld_bounds],
            "r_2d_n_low": ld_counts[0], "r_2d_n_high": ld_counts[1],
            "applies_to": "pearson_log only; spearman uses raw radii",
        },
        "high_d_mode": mode,
        "anchor_rule": (
            "sorted Generator(PCG64(anchor_seed)).choice(N, ceil(n_anchors*"
            "pool_factor), replace=False); keep the first n_anchors pool rows "
            "in ascending order with r_hd > eps_hd"
        ),
        "radii": {
            "r_hd_min": float(kept_r_hd.min()),
            "r_hd_median": float(np.median(kept_r_hd)),
            "r_hd_max": float(kept_r_hd.max()),
            "r_2d_min": float(r_2d.min()),
            "r_2d_median": float(np.median(r_2d)),
            "r_2d_max": float(r_2d.max()),
            "n_r2d_zero": int((r_2d == 0).sum()),
        },
        "anchor_ids_sha256_prefix": _ids_digest(kept_ids),
    }
    if leave_one_out:
        result["leave_one_out"] = {
            "spearman": _loo_block(value_spearman, _loo_spearman(kept_r_hd, r_2d)),
            "pearson_log": _loo_block(value_pearson, _loo_pearson(log_hd, log_2d)),
        }
    return result


def density_v2_legacy(
    xy: np.ndarray,
    substrate_or_radii: Any,
    anchor_seed: int = 0,
    n_anchors: int = DEFAULT_N_ANCHORS,
    *,
    k: int = K_DENSITY,
    pool_factor: float = POOL_FACTOR,
    threads: int = DEFAULT_THREADS,
    leave_one_out: bool = True,
    radii_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The v2 statistic, kept only for side-by-side comparison.

    Reproduces the released definition exactly — ``Pearson(log(r_hd + 1e-12),
    log(r_2d + 1e-12))`` with **no** degenerate-anchor exclusion — over the same
    anchor pool v3 would start from, so the two numbers differ only by policy.
    """
    coords = np.asarray(xy)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise DensityV3Error("xy must be (N, d>=2)")
    n_rows = len(coords)
    pool_ids, pool_r_hd, mode = _resolve_high_d(
        substrate_or_radii, n_rows, n_anchors, anchor_seed, pool_factor,
        k, threads, radii_kwargs or {},
    )
    ids = pool_ids[:n_anchors]
    r_hd = pool_r_hd[:n_anchors]
    r_2d = low_d_radii(coords, ids, k=k, threads=threads)
    log_hd = np.log(r_hd + LEGACY_LOG_EPSILON)
    log_2d = np.log(r_2d + LEGACY_LOG_EPSILON)
    value = pearson(log_hd, log_2d)
    out: dict[str, Any] = {
        "schema": "density-v2-legacy-2026-08-13",
        "pearson_log": value,
        "value": value,
        "spearman": spearman(r_hd, r_2d),
        "log_epsilon": LEGACY_LOG_EPSILON,
        "n_anchors": int(len(ids)),
        "n_degenerate_hd_included": int((r_hd <= EPS_HD).sum()),
        "n_r2d_zero": int((r_2d == 0).sum()),
        "high_d_mode": mode,
        "anchor_seed": int(anchor_seed),
    }
    if leave_one_out:
        out["leave_one_out"] = {
            "pearson_log": _loo_block(value, _loo_pearson(log_hd, log_2d)),
        }
    return out


def _ids_digest(ids: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(ids, dtype=np.int64)).tobytes()
    ).hexdigest()[:16]
