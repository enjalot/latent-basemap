"""OOD confidence for parametric-UMAP projections (CPU-only, importable).

Question this module answers: does a *cheap* per-point novelty score predict the
*expensive* per-point placement error of a parametric-UMAP map for a query point?

If it does, latent-scope can flag "don't trust this point's position" at project
time without ever computing the ground-truth neighbourhood-recall error.

The three quantities, all as pure functions of ``(substrate, radii, index, model)``:

  * ``novelty_score``          — Signal A: cheap. query's mean k-NN distance to
                                  the training substrate, normalised by the local
                                  training scale (mean k-NN radius of those
                                  neighbours). >1 = query sits in sparser territory
                                  than the training points around it.
  * ``pointwise_placement_error`` — ground truth: 1 - recall of the query's high-D
                                  training neighbours inside its 2D disc.
  * ``seed_ensemble_spread``   — Signal B: also cheap-ish. disagreement of the
                                  query's 2D position across a seed family after a
                                  single procrustes alignment of the training maps.

Everything runs on CPU. The kNN index is FAISS-cpu when available, otherwise a
chunked exact BLAS scan (identical results for unit-norm rows).

Distances are Euclidean on unit-norm rows: for cosine similarity ``s`` between
unit vectors, ``d = sqrt(max(2 - 2s, 0))``. This matches ``density_v3.high_d_radii``,
which produces the ``substrate_radii`` fed in here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

try:  # FAISS-cpu is the fast path; chunked-exact BLAS is the fallback.
    import faiss  # type: ignore

    _HAVE_FAISS = True
except Exception:  # pragma: no cover
    faiss = None  # type: ignore
    _HAVE_FAISS = False


# ── kNN index over the training substrate ────────────────────────────────────

@dataclass
class _ExactIndex:
    """Chunked exact inner-product index (fallback when FAISS is absent)."""

    substrate: np.ndarray
    chunk: int = 262_144

    @property
    def ntotal(self) -> int:
        return int(self.substrate.shape[0])

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(queries, dtype=np.float32)
        m = q.shape[0]
        n = self.ntotal
        kk = min(k, n)
        best_s = np.full((m, kk), -np.inf, dtype=np.float32)
        best_i = np.full((m, kk), -1, dtype=np.int64)
        for start in range(0, n, self.chunk):
            stop = min(start + self.chunk, n)
            block = np.ascontiguousarray(self.substrate[start:stop], dtype=np.float32)
            sims = q @ block.T                                  # (m, blk)
            take = min(kk, stop - start)
            part = np.argpartition(sims, -take, axis=1)[:, -take:]
            cand_s = np.take_along_axis(sims, part, axis=1)
            cand_i = part.astype(np.int64) + start
            merged_s = np.concatenate([best_s, cand_s], axis=1)
            merged_i = np.concatenate([best_i, cand_i], axis=1)
            sel = np.argpartition(merged_s, -kk, axis=1)[:, -kk:]
            best_s = np.take_along_axis(merged_s, sel, axis=1)
            best_i = np.take_along_axis(merged_i, sel, axis=1)
            del block, sims, part, cand_s, cand_i, merged_s, merged_i, sel
        order = np.argsort(-best_s, axis=1, kind="stable")       # descending sim
        best_s = np.take_along_axis(best_s, order, axis=1)
        best_i = np.take_along_axis(best_i, order, axis=1)
        return best_s, best_i


def build_index(substrate: Any, *, threads: int | None = None) -> Any:
    """Build a CPU cosine / inner-product kNN index over the training substrate.

    ``substrate`` may be an ``(N, D)`` array, an ``np.memmap``, or a path to a
    ``.npy``. Rows are assumed unit-norm (inner product == cosine). Returns a
    FAISS ``IndexFlatIP`` if faiss is importable, else a chunked-exact fallback
    with the same ``.search(queries, k) -> (sims, ids)`` contract.
    """
    if isinstance(substrate, (str, os.PathLike)):
        substrate = np.load(substrate, mmap_mode="r", allow_pickle=False)
    if getattr(substrate, "ndim", 0) != 2:
        raise ValueError("substrate must be 2-D (N, D)")
    d = int(substrate.shape[1])
    if _HAVE_FAISS:
        if threads is not None:
            faiss.omp_set_num_threads(int(threads))
        index = faiss.IndexFlatIP(d)
        # faiss needs a contiguous fp32 buffer; materialise once (~3 GB @ 2M×384).
        index.add(np.ascontiguousarray(substrate, dtype=np.float32))
        return index
    return _ExactIndex(np.asarray(substrate))


def _search(index: Any, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    q = np.ascontiguousarray(queries, dtype=np.float32)
    sims, ids = index.search(q, k)
    return np.asarray(sims, dtype=np.float32), np.asarray(ids, dtype=np.int64)


def _sim_to_dist(sims: np.ndarray) -> np.ndarray:
    """Euclidean distance on unit-norm rows from cosine similarity."""
    return np.sqrt(np.maximum(2.0 - 2.0 * sims, 0.0))


# ── Signal A: cheap novelty score ────────────────────────────────────────────

@dataclass
class NoveltyResult:
    ratio: np.ndarray          # raw novelty ratio (>1 == sparser than local training)
    percentile: np.ndarray | None  # 0..1 rank vs a reference (holdout) distribution
    query_mean_dist: np.ndarray    # mean k-NN Euclidean distance to training
    local_scale: np.ndarray        # mean substrate radius of those k neighbours


def novelty_score(
    query_X: np.ndarray,
    index: Any,
    substrate_radii: np.ndarray,
    *,
    k: int = 15,
    ref_ratios: np.ndarray | None = None,
) -> NoveltyResult:
    """Signal A. Per-query novelty = (query's mean k-NN distance to training) /
    (mean training k-NN radius of those k neighbours).

    A ratio near 1 means the query sits at the same local density as the training
    points nearest it; a ratio >1 means it lands in sparser / less-supported
    territory than the map ever saw there.

    If ``ref_ratios`` is given (typically the in-corpus holdout's ratios), each
    query also gets a 0..1 percentile = fraction of the reference below it.
    """
    substrate_radii = np.asarray(substrate_radii, dtype=np.float64)
    sims, ids = _search(index, np.asarray(query_X, dtype=np.float32), k)
    dists = _sim_to_dist(sims)                       # (m, k)
    query_mean_dist = dists.mean(axis=1)             # cheap proxy for query radius
    local_scale = substrate_radii[ids].mean(axis=1)  # local training scale
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = query_mean_dist / local_scale
    ratio = np.where(np.isfinite(ratio), ratio, np.nan)

    percentile = None
    if ref_ratios is not None:
        ref = np.sort(np.asarray(ref_ratios, dtype=np.float64))
        percentile = np.searchsorted(ref, ratio, side="right") / max(len(ref), 1)
        percentile = percentile.astype(np.float64)
    return NoveltyResult(
        ratio=ratio.astype(np.float64),
        percentile=percentile,
        query_mean_dist=query_mean_dist.astype(np.float64),
        local_scale=local_scale.astype(np.float64),
    )


# ── Ground truth: pointwise placement error ──────────────────────────────────

def pointwise_placement_error(
    query_X: np.ndarray,
    query_xy: np.ndarray,
    substrate: Any,
    substrate_xy: np.ndarray,
    index: Any,
    *,
    k: int = 15,
    disc_frac: float = 0.001,
    kdtree: Any | None = None,
    workers: int = 8,
) -> np.ndarray:
    """Per-query placement error = 1 - recall of the query's high-D training
    neighbours inside its 2D disc.

    Mirrors ``knobs_2m.quick_ffr`` conventions (k_true=15 high-D truth; 2D disc =
    0.1% of the training rows), but pointwise and for arbitrary (test) queries:

      1. high-D truth  = query's ``k`` nearest TRAINING points (via ``index``);
      2. 2D disc       = the ``disc = max(int(N*disc_frac), 1)`` training points
                         nearest ``query_xy`` (KDTree over ``substrate_xy``);
      3. error         = fraction of the ``k`` truth ids NOT inside the disc.

    Returns an array of per-query error in [0, 1]. ``substrate`` is accepted for
    API symmetry (the high-D truth already comes from ``index``); only its row
    count is used.
    """
    from scipy.spatial import cKDTree

    substrate_xy = np.asarray(substrate_xy, dtype=np.float32)
    n_rows = int(substrate_xy.shape[0])
    disc = max(int(n_rows * disc_frac), 1)

    _, truth_ids = _search(index, np.asarray(query_X, dtype=np.float32), k)  # (m, k)

    if kdtree is None:
        kdtree = cKDTree(substrate_xy)
    _, near2d = kdtree.query(np.asarray(query_xy, dtype=np.float32), k=disc, workers=workers)
    if near2d.ndim == 1:
        near2d = near2d[:, None]

    m = truth_ids.shape[0]
    err = np.empty(m, dtype=np.float64)
    for i in range(m):
        hits = np.isin(truth_ids[i], near2d[i], assume_unique=False).sum()
        err[i] = 1.0 - hits / truth_ids.shape[1]
    return err


# ── Signal B: seed-ensemble spread ───────────────────────────────────────────

def _similarity_transform(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Umeyama similarity transform mapping ``src`` onto ``dst`` (rotation +
    reflection + uniform scale + translation), least-squares. Returns
    ``(scale, R, t)`` such that ``scale * src @ R.T + t ≈ dst``."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    cov = (dst_c.T @ src_c) / src.shape[0]
    U, S, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:                # keep it a proper alignment, allow reflection
        U2 = U.copy()
        U2[:, -1] *= -1
        R = U2 @ Vt
        S = S.copy()
        S[-1] *= -1
    var_s = (src_c ** 2).sum() / src.shape[0]
    scale = float(S.sum() / var_s) if var_s > 0 else 1.0
    t = mu_d - scale * (R @ mu_s)
    return scale, R, t


def _apply_transform(xy: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return scale * (np.asarray(xy, dtype=np.float64) @ R.T) + t


def seed_ensemble_spread(
    query_proj_list: Sequence[np.ndarray],
    train_xy_list: Sequence[np.ndarray],
    *,
    fit_sample: int = 100_000,
    seed: int = 42,
) -> np.ndarray:
    """Signal B. Per-query disagreement across a seed family after one procrustes
    alignment of the training maps.

    ``query_proj_list`` — the SAME queries projected through each seed model
    (each ``(m, 2)``); ``train_xy_list`` — each seed's training 2D coordinates
    (each ``(N, 2)``, row-aligned across seeds). The first seed is the reference;
    every other seed's map is aligned to it with a similarity transform fit on the
    training coords, and that transform is applied to its query projection. The
    per-query spread is the mean pairwise distance across the aligned positions.
    """
    if len(query_proj_list) != len(train_xy_list):
        raise ValueError("query_proj_list and train_xy_list must be same length")
    if len(query_proj_list) < 2:
        raise ValueError("need >= 2 seed models for an ensemble spread")

    ref_train = np.asarray(train_xy_list[0], dtype=np.float64)
    n = ref_train.shape[0]
    rng = np.random.default_rng(seed)
    fit_ids = (rng.choice(n, size=min(fit_sample, n), replace=False)
               if fit_sample and fit_sample < n else np.arange(n))

    aligned = [np.asarray(query_proj_list[0], dtype=np.float64)]
    for j in range(1, len(query_proj_list)):
        src = np.asarray(train_xy_list[j], dtype=np.float64)[fit_ids]
        dst = ref_train[fit_ids]
        scale, R, t = _similarity_transform(src, dst)
        aligned.append(_apply_transform(query_proj_list[j], scale, R, t))

    stack = np.stack(aligned, axis=0)          # (S, m, 2)
    s = stack.shape[0]
    m = stack.shape[1]
    acc = np.zeros(m, dtype=np.float64)
    pairs = 0
    for a in range(s):
        for b in range(a + 1, s):
            acc += np.linalg.norm(stack[a] - stack[b], axis=1)
            pairs += 1
    return acc / max(pairs, 1)


__all__ = [
    "build_index",
    "novelty_score",
    "NoveltyResult",
    "pointwise_placement_error",
    "seed_ensemble_spread",
]
