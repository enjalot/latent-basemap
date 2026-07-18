"""Consolidated evaluation harness for parametric-UMAP basemaps.

This module replaces the three divergent metric implementations that used to
live in ``experiments/run_experiment.py`` (``compute_metrics``),
``autoresearch/prepare.py`` (``evaluate``) and ``validate_umap.py``. It is the
single source of truth for basemap evaluation, implementing the metric panel
described in ``latent-labs/guides/plan-basemap-atlas.md`` §2.2 and the stability
playbook of §4.

Design principles
-----------------
* **Full-corpus neighbours, never within-subsample.** The biggest defect in the
  old code was computing "held-out kNN" by k-NN'ing a small subsample against
  itself. Here, true neighbours are always computed against the *full* dataset
  with FAISS (CPU ``IndexFlatL2``). Anchors are sampled, but their neighbours
  are ranked against every row.
* **CPU-only, memmap-friendly.** ``X_high`` may be a ``np.memmap`` or a
  :class:`basemap.data_loader.MemmapArrayConcatenator`. We never materialise the
  full float32 array in our own code -- rows are read lazily in blocks. FAISS's
  flat index keeps its own copy (that *is* the index); everything else samples
  rows. Respect the >=2 GiB rule: keep transient buffers small.
* **Everything is a plain function over numpy arrays** so the panel is trivially
  unit-testable on synthetic data (swiss roll, blobs).

Metric families (plan §2.1)
---------------------------
A. **Map fidelity** -- is the 2D layout faithful to the high-D structure of the
   corpus it was trained on? (kNN recall, trustworthiness + continuity, Spearman
   distance correlation, triplet accuracy, density preservation).
B. **Projection fidelity** -- does a *new* point land near its true neighbours in
   the existing map? (out-of-sample recall into the training map).
C. **Stability** -- same rows, two maps: Procrustes alignment, anchor kNN
   overlap, per-point drift.

Plus cluster/task-level metrics (Leiden of the high-D graph + neighbourhood hit,
silhouette, linear-probe gap), PCA-2D / random-projection floors, per-point
diagnostics parquet, and spatially-resolved T/C bins.

CLI::

    python -m basemap.eval score \
        --coords run_dir/coords.parquet \
        --embeddings /data/embeddings/<dataset>/train \
        --out metrics.json \
        --per-point diagnostics.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Row access helpers (memmap-safe)
# ─────────────────────────────────────────────────────────────────────────────


def read_rows(X, idx) -> np.ndarray:
    """Read ``X[idx]`` as a contiguous float32 array.

    Works for ``np.ndarray``, ``np.memmap`` and
    :class:`MemmapArrayConcatenator` (which vectorises 1D integer-array
    indexing). Always passes an ndarray index so the concatenator takes its fast
    path (its slice path loops row-by-row).
    """
    idx = np.asarray(idx)
    rows = X[idx]
    return np.ascontiguousarray(np.asarray(rows), dtype=np.float32)


def _n_rows(X) -> int:
    return int(X.shape[0])


def _dim(X) -> int:
    return int(X.shape[1])


def sample_indices(n: int, count: int, seed: int = 42) -> np.ndarray:
    """Deterministically sample ``count`` distinct row ids from ``range(n)``."""
    if count >= n:
        return np.arange(n)
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(n, size=count, replace=False))


# ─────────────────────────────────────────────────────────────────────────────
# FAISS full-corpus neighbours
# ─────────────────────────────────────────────────────────────────────────────


def build_flat_l2_index(X, block: int = 100_000):
    """Build an exact L2 flat index over the full corpus, adding rows in blocks.

    We never hold more than ``block`` rows (float32) as a transient buffer; the
    index itself stores ``n*d*4`` bytes (unavoidable for exact search). For
    corpora where that exceeds memory, an ANN index would be substituted, but
    the plan targets CPU FAISS at <=1M rows which is fine.
    """
    import faiss

    dim = _dim(X)
    n = _n_rows(X)
    index = faiss.IndexFlatL2(dim)
    for start in range(0, n, block):
        end = min(start + block, n)
        blk = read_rows(X, np.arange(start, end))
        index.add(blk)
    return index


def _knn_excluding_self(index, queries: np.ndarray, query_ids: np.ndarray, k: int) -> np.ndarray:
    """Query ``index`` and drop each query's own id from its neighbour list.

    ``queries`` are rows that are *also* present in the index (transductive
    case). Returns an ``[m, k]`` int array of neighbour ids (full-corpus ids).
    """
    D, I = index.search(np.ascontiguousarray(queries, dtype=np.float32), k + 1)
    out = np.empty((len(queries), k), dtype=np.int64)
    for r in range(len(queries)):
        row = I[r]
        row = row[row != query_ids[r]]
        if len(row) < k:  # self not found (shouldn't happen); pad by dropping farthest
            row = I[r][:k]
        out[r] = row[:k]
    return out


def _knn_external(index, queries: np.ndarray, k: int) -> np.ndarray:
    """Query ``index`` with points that are NOT in the index (inductive case)."""
    D, I = index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
    return I.astype(np.int64)


def knn_ids_full(X, anchor_idx: np.ndarray, k: int, index=None) -> np.ndarray:
    """True k nearest neighbours of ``anchor_idx`` among the *full* corpus X."""
    if index is None:
        index = build_flat_l2_index(X)
    q = read_rows(X, anchor_idx)
    return _knn_excluding_self(index, q, anchor_idx, k)


# ─────────────────────────────────────────────────────────────────────────────
# Family A: map fidelity
# ─────────────────────────────────────────────────────────────────────────────


def _gpu_knn_ids(X, anchor_idx: np.ndarray, k: int, device="cuda", achunk: int = 128,
                 return_dist: bool = False):
    """Exact L2 k-NN of ``anchor_idx`` among the full corpus X, self-excluded —
    same result as ``knn_ids_full`` but via chunked GPU matmul (~100x faster at
    scale). fp32 (fp16 misranks near-unit-norm neighbours). Corpus resident on
    device (fp32 ≈ N*d*4 B; fits to ~10M on 32 GB, >10M needs corpus chunking).

    ``return_dist``: also return (m,k) true L2 distances to those neighbours
    (radii for density_preservation) — computed from the *same* topk matrix, so
    recall and density share one GPU pass instead of two full-corpus kNNs.
    """
    import torch
    Xt = torch.from_numpy(np.ascontiguousarray(np.asarray(X))).to(device, torch.float32)
    # ||row||^2 in blocks — (Xt*Xt) would materialize a full second copy of Xt.
    xn = torch.empty(Xt.shape[0], device=device)
    for j in range(0, Xt.shape[0], 1_000_000):
        xn[j:j + 1_000_000] = Xt[j:j + 1_000_000].pow(2).sum(1)
    aidx = np.asarray(anchor_idx)
    aidx_t = torch.as_tensor(aidx, device=device, dtype=torch.long)
    A = Xt.index_select(0, aidx_t)
    # cap anchor-chunk so the (achunk x N) distance matrix stays small at scale
    # (128 x 8M x 4B = 3.8 GB would OOM on top of the 24.6 GB fp32 corpus).
    N, D = Xt.shape
    achunk = min(achunk, max(8, int(2e8 // max(N, 1))))
    # Low-dim spaces (2-D atlas coords) need EXACT ranking: the ||q||^2+||x||^2
    # -2q·x expansion can't resolve near-zero distances between duplicate points
    # (fp cancellation on values that are truly 0), which corrupts the smallest
    # radii and destroys the density log-log correlation. Unit-norm high-D has
    # no such cancellation, so it keeps the fast matmul path.
    low_dim = D <= 16
    out = np.empty((len(aidx), k), dtype=np.int64)
    dout = np.empty((len(aidx), k), dtype=np.float64) if return_dist else None
    for i in range(0, A.shape[0], achunk):
        q = A[i:i + achunk]
        if low_dim:
            d2 = torch.zeros((q.shape[0], N), device=device)   # exact sum of sq diffs
            for c in range(D):
                diff = q[:, c:c + 1] - Xt[:, c]                # (c,N) broadcast
                d2.addcmul_(diff, diff)
        else:
            # rank by L2^2 = ||q||^2 + ||x||^2 - 2 q·x; the ||q||^2 term is
            # constant per row (irrelevant to ranking). In-place to avoid copies.
            d2 = q @ Xt.T                      # (c,N)
            d2.mul_(-2.0).add_(xn)            # -> ||x||^2 - 2 q·x  (rank-equiv)
        ti = torch.topk(d2, k + 1, dim=1, largest=False).indices   # (c,k+1) on GPU
        if return_dist:
            # Exact L2 to the k+1 candidates by *gathering* them and subtracting
            # directly — NOT the ||q||^2+||x||^2-2q·x expansion, which loses all
            # precision for small low-D radii (coords O(10), gaps O(0.01) →
            # catastrophic fp32 cancellation). Only k+1 neighbours, so cheap.
            nb = Xt.index_select(0, ti.reshape(-1)).reshape(ti.shape[0], k + 1, -1)
            tv = (nb - q[:, None, :]).norm(dim=2).cpu().numpy()     # (c,k+1) exact
        ti = ti.cpu().numpy()
        s = aidx[i:i + achunk]
        for r in range(ti.shape[0]):
            keep = ti[r] != s[r]
            row = ti[r][keep][:k]
            if len(row) == k:
                out[i + r] = row
                if return_dist:
                    dout[i + r] = tv[r][keep][:k]
            else:                              # anchor not in its own top-(k+1)
                out[i + r] = ti[r][:k]
                if return_dist:
                    dout[i + r] = tv[r][:k]
    del Xt, xn, A
    torch.cuda.empty_cache()
    return (out, dout) if return_dist else out


def gpu_score(X_high, Z_low, anchor_idx: np.ndarray, k_recall=(10, 50),
              k_density: int = 15, device="cuda"):
    """Fused GPU scoring — recall@k *and* density_preservation from a single
    high-D and a single low-D full-corpus kNN pass (two passes total instead of
    four). The recall neighbours (indices) and the density radii (distances) both
    come out of the same topk matrix, so density is essentially free once recall
    is computed. Matches ``knn_recall`` + ``density_preservation`` semantics
    (exact L2, self-excluded, radius = mean dist to k nearest). Returns a dict
    ``{"recall@10":…, "recall@50":…, "density":…}``.
    """
    kr = tuple(k_recall) if not isinstance(k_recall, int) else (k_recall,)
    kmax = max(max(kr), k_density)
    hi_idx, hi_dist = _gpu_knn_ids(X_high, anchor_idx, kmax, device=device, return_dist=True)
    lo_idx, lo_dist = _gpu_knn_ids(Z_low, anchor_idx, kmax, device=device, return_dist=True)
    m = len(anchor_idx)
    out = {}
    for k in kr:
        per = np.fromiter(
            (len(np.intersect1d(hi_idx[i, :k], lo_idx[i, :k])) / k for i in range(m)),
            dtype=np.float64, count=m)
        out[f"recall@{k}"] = float(per.mean())
    # density: mean radius to k nearest (self already excluded, dists ascending)
    rh = hi_dist[:, :k_density].mean(axis=1)
    rl = lo_dist[:, :k_density].mean(axis=1)
    eps = 1e-12
    log_rh, log_rl = np.log(rh + eps), np.log(rl + eps)
    out["density"] = (float(np.corrcoef(log_rh, log_rl)[0, 1])
                      if np.std(log_rh) > eps and np.std(log_rl) > eps else float("nan"))
    return out


def knn_recall(
    X_high,
    Z_low,
    anchor_idx: np.ndarray,
    k: int,
    high_index=None,
    low_index=None,
    true_neighbors: Optional[np.ndarray] = None,
    use_gpu: bool = False,
):
    """kNN recall: fraction of each anchor's true high-D neighbours that are also
    among its 2D neighbours, both computed against the **full** dataset.

    ``use_gpu=True`` computes both neighbour sets via GPU matmul (validated to
    match the CPU FAISS path; ~100-400x faster at scale). ``true_neighbors`` can
    be precomputed and passed to skip recomputing the (map-independent) high-D
    neighbours across many maps of the same corpus.

    Returns ``(mean_recall, per_anchor_recall)``.
    """
    if use_gpu:
        if true_neighbors is None:
            true_neighbors = _gpu_knn_ids(X_high, anchor_idx, k)
        pred = _gpu_knn_ids(Z_low, anchor_idx, k)
    else:
        if high_index is None:
            high_index = build_flat_l2_index(X_high)
        if low_index is None:
            low_index = build_flat_l2_index(Z_low)
        if true_neighbors is None:
            true_neighbors = knn_ids_full(X_high, anchor_idx, k, index=high_index)
        q_low = read_rows(Z_low, anchor_idx)
        pred = _knn_excluding_self(low_index, q_low, anchor_idx, k)

    per = np.empty(len(anchor_idx), dtype=np.float64)
    for i in range(len(anchor_idx)):
        per[i] = len(np.intersect1d(true_neighbors[i], pred[i], assume_unique=False)) / k
    return float(per.mean()), per


def _rank_matrix(D: np.ndarray) -> np.ndarray:
    """Rank of every point from every row (self -> 0), ascending distance."""
    return np.argsort(np.argsort(D, axis=1), axis=1)


def _tc_core(rank_source: np.ndarray, nn_target: np.ndarray, k: int):
    """Trustworthiness/continuity core (rank-based, matches sklearn's T).

    ``rank_source`` is the [m, m] rank matrix in the *ranking* space; ``nn_target``
    is [m, k] neighbour ids in the *neighbour* space (excluding self). Returns
    ``(global_score, per_point_score)``.
    """
    m = rank_source.shape[0]
    per = np.zeros(m, dtype=np.float64)
    for i in range(m):
        r = rank_source[i, nn_target[i]]
        viol = r[r > k] - k
        per[i] = viol.sum()
    denom = k * (2 * m - 3 * k - 1)
    if denom <= 0:
        return 1.0, np.ones(m)
    per_score = 1.0 - (2.0 / denom) * per
    global_score = 1.0 - (2.0 / (m * denom)) * per.sum()
    return float(global_score), per_score


def trustworthiness_continuity(X_high, Z_low, k: int, idx: Optional[np.ndarray] = None):
    """Compute trustworthiness AND continuity (rank-based) on a point subset.

    Both are computed exactly on the same set of points in both spaces (the
    standard definition, matching :func:`sklearn.manifold.trustworthiness` for
    T). Because exact ranks require full pairwise distances, this runs on a
    subsample of up to ``len(idx)`` points; the kNN *recall* metric above is the
    full-corpus one. Continuity is trustworthiness with the two spaces swapped.

    Returns dict with ``trustworthiness``, ``continuity`` and per-point arrays.
    """
    from sklearn.metrics import pairwise_distances

    if idx is None:
        idx = np.arange(_n_rows(X_high))
    Xs = read_rows(X_high, idx)
    Zs = read_rows(Z_low, idx)
    m = len(idx)
    k = min(k, m - 1)

    D_high = pairwise_distances(Xs)
    D_low = pairwise_distances(Zs)
    rank_high = _rank_matrix(D_high)
    rank_low = _rank_matrix(D_low)

    # neighbours (exclude self at rank 0)
    nn_low = np.argsort(D_low, axis=1)[:, 1 : k + 1]
    nn_high = np.argsort(D_high, axis=1)[:, 1 : k + 1]

    T, per_T = _tc_core(rank_high, nn_low, k)  # false neighbours penalised in high-D rank
    C, per_C = _tc_core(rank_low, nn_high, k)  # missing neighbours penalised in low-D rank

    return {
        "trustworthiness": T,
        "continuity": C,
        "per_point_trustworthiness": per_T,
        "per_point_continuity": per_C,
        "idx": idx,
        "k": k,
    }


def spearman_distance_correlation(X_high, Z_low, n_pairs: int = 100_000, seed: int = 42):
    """Spearman correlation of pairwise L2 distances on sampled pairs.

    Spearman (rank) rather than Pearson because concentrated embedding spaces are
    dominated by mid-range distances (plan §2.2).
    """
    from scipy.stats import spearmanr

    n = _n_rows(X_high)
    rng = np.random.RandomState(seed)
    i = rng.randint(0, n, n_pairs)
    j = rng.randint(0, n, n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]

    Xi, Xj = read_rows(X_high, i), read_rows(X_high, j)
    Zi, Zj = read_rows(Z_low, i), read_rows(Z_low, j)
    dh = np.linalg.norm(Xi - Xj, axis=1)
    dl = np.linalg.norm(Zi - Zj, axis=1)
    rho, _ = spearmanr(dh, dl)
    return float(rho)


def triplet_accuracy(X_high, Z_low, n_triplets: int = 100_000, seed: int = 42):
    """PaCMAP/TriMap-style triplet accuracy (plan §2.2, lit ``28-trimap.md``).

    For random triplets (i, j, l): fraction where the 2D layout preserves the
    high-D order of d(i, j) vs d(i, l).
    """
    n = _n_rows(X_high)
    rng = np.random.RandomState(seed)
    i = rng.randint(0, n, n_triplets)
    j = rng.randint(0, n, n_triplets)
    l = rng.randint(0, n, n_triplets)
    mask = (i != j) & (i != l) & (j != l)
    i, j, l = i[mask], j[mask], l[mask]

    Xi, Xj, Xl = read_rows(X_high, i), read_rows(X_high, j), read_rows(X_high, l)
    Zi, Zj, Zl = read_rows(Z_low, i), read_rows(Z_low, j), read_rows(Z_low, l)
    dh_ij = np.linalg.norm(Xi - Xj, axis=1)
    dh_il = np.linalg.norm(Xi - Xl, axis=1)
    dl_ij = np.linalg.norm(Zi - Zj, axis=1)
    dl_il = np.linalg.norm(Zi - Zl, axis=1)
    agree = (np.sign(dh_ij - dh_il) == np.sign(dl_ij - dl_il))
    return float(agree.mean())


def density_preservation(X_high, Z_low, anchor_idx: np.ndarray, k: int = 15,
                         high_index=None, low_index=None, use_gpu=False):
    """densMAP-style density preservation (lit ``05-densmap.md``).

    Local radius = mean distance to the k nearest neighbours. Reports the
    Pearson correlation of log local radii (high-D vs 2D). Returns
    ``(corr, log_r_high, log_r_low)`` where the log-radius arrays align with
    ``anchor_idx``.

    ``use_gpu=True`` computes exact radii via chunked GPU matmul (100x faster and
    numerically correct). PREFER IT for low-D atlas coords: the default FAISS path
    (``IndexFlatL2`` batched search) evaluates L2 through the ``||x||^2+||q||^2
    -2xq`` BLAS expansion, which catastrophically cancels for near-duplicate 2-D
    points (coords O(10), true dist^2 O(1e-6) -> clamped to 0 by ``max(.,0)``).
    That manufactures spurious zero radii and biases the correlation (upward at
    low N, downward at high density) — the CPU numbers are NOT reliable at scale.
    """
    if use_gpu:
        _, rh = _gpu_knn_ids(np.asarray(X_high), anchor_idx, k, return_dist=True)
        _, rl = _gpu_knn_ids(np.asarray(Z_low), anchor_idx, k, return_dist=True)
        rh, rl = rh.mean(axis=1), rl.mean(axis=1)
        eps = 1e-12
        log_rh, log_rl = np.log(rh + eps), np.log(rl + eps)
        corr = (float(np.corrcoef(log_rh, log_rl)[0, 1])
                if np.std(log_rh) > eps and np.std(log_rl) > eps else float("nan"))
        return corr, log_rh, log_rl

    import faiss

    if high_index is None:
        high_index = build_flat_l2_index(X_high)
    if low_index is None:
        low_index = build_flat_l2_index(Z_low)

    qh = read_rows(X_high, anchor_idx)
    ql = read_rows(Z_low, anchor_idx)
    Dh, _ = high_index.search(np.ascontiguousarray(qh), k + 1)
    Dl, _ = low_index.search(np.ascontiguousarray(ql), k + 1)
    # FAISS returns squared L2; take sqrt, drop self (col 0)
    rh = np.sqrt(np.maximum(Dh[:, 1:], 0)).mean(axis=1)
    rl = np.sqrt(np.maximum(Dl[:, 1:], 0)).mean(axis=1)
    eps = 1e-12
    log_rh = np.log(rh + eps)
    log_rl = np.log(rl + eps)
    if np.std(log_rh) < eps or np.std(log_rl) < eps:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(log_rh, log_rl)[0, 1])
    return corr, log_rh, log_rl


# ─────────────────────────────────────────────────────────────────────────────
# Family B: projection fidelity (out-of-sample)
# ─────────────────────────────────────────────────────────────────────────────


def projection_fidelity(X_train, Z_train, X_query, Z_query, k: int = 10,
                        train_high_index=None, train_low_index=None):
    """Out-of-sample projection fidelity (plan §2.1 family B).

    For each held-out query point: find its true high-D k-NN **among the training
    corpus** (full-corpus FAISS), then measure recall of those neighbours among
    the query's 2D k-NN within the training-map points.

    Returns ``(mean_recall, per_query_recall)``.
    """
    if train_high_index is None:
        train_high_index = build_flat_l2_index(X_train)
    if train_low_index is None:
        train_low_index = build_flat_l2_index(Z_train)

    qh = read_rows(X_query, np.arange(_n_rows(X_query)))
    ql = read_rows(Z_query, np.arange(_n_rows(Z_query)))
    true_nn = _knn_external(train_high_index, qh, k)   # ids into training corpus
    pred_nn = _knn_external(train_low_index, ql, k)

    per = np.empty(len(qh), dtype=np.float64)
    for i in range(len(qh)):
        per[i] = len(np.intersect1d(true_nn[i], pred_nn[i])) / k
    return float(per.mean()), per


# ─────────────────────────────────────────────────────────────────────────────
# Family C: stability
# ─────────────────────────────────────────────────────────────────────────────


def procrustes_align(A: np.ndarray, B: np.ndarray):
    """Optimal rigid alignment of ``B`` onto ``A`` (rotation, reflection, scale).

    Finds ``s, R, t`` minimising ``||A - (s B R + t)||_F`` where ``R`` is
    orthogonal (reflections allowed). Returns
    ``(B_aligned, disparity, transform)`` where ``disparity`` is the normalised
    residual ``||A_c - s B_c R||^2 / ||A_c||^2`` (0 == perfect) and ``transform``
    holds ``rotation``, ``scale`` and ``translation``.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    muA = A.mean(axis=0)
    muB = B.mean(axis=0)
    Ac = A - muA
    Bc = B - muB

    # optimal orthogonal R: argmin ||Ac - Bc R||  ->  R = U V^T from SVD(Bc^T Ac)
    M = Bc.T @ Ac
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt
    normB2 = (Bc ** 2).sum()
    scale = S.sum() / normB2 if normB2 > 0 else 1.0
    B_aligned = scale * (Bc @ R) + muA
    normA2 = (Ac ** 2).sum()
    resid = ((Ac - scale * (Bc @ R)) ** 2).sum()
    disparity = float(resid / normA2) if normA2 > 0 else 0.0
    return B_aligned, disparity, {"rotation": R, "scale": float(scale), "translation": muA - scale * (muB @ R)}


def anchor_knn_overlap(Z_a, Z_b, anchor_idx: np.ndarray, k: int = 10):
    """Jaccard overlap of the 2D k-NN sets of each anchor between two maps.

    Both neighbour sets are computed in 2D against the full point sets (the
    stability metric of plan §4 / lit ``139-sae-seed-stability.md``). Returns
    ``(mean_jaccard, per_anchor_jaccard)``.
    """
    idx_a = build_flat_l2_index(Z_a)
    idx_b = build_flat_l2_index(Z_b)
    qa = read_rows(Z_a, anchor_idx)
    qb = read_rows(Z_b, anchor_idx)
    nn_a = _knn_excluding_self(idx_a, qa, anchor_idx, k)
    nn_b = _knn_excluding_self(idx_b, qb, anchor_idx, k)
    per = np.empty(len(anchor_idx), dtype=np.float64)
    for i in range(len(anchor_idx)):
        sa, sb = set(nn_a[i].tolist()), set(nn_b[i].tolist())
        union = len(sa | sb)
        per[i] = len(sa & sb) / union if union else 1.0
    return float(per.mean()), per


def per_point_drift(A_aligned: np.ndarray, B_aligned: np.ndarray) -> np.ndarray:
    """Per-point Euclidean displacement between two aligned maps."""
    return np.linalg.norm(np.asarray(A_aligned) - np.asarray(B_aligned), axis=1)


def compare_maps(Z_a, Z_b, k: int = 10, n_anchors: int = 10_000, seed: int = 42):
    """Stability panel comparing two maps of the *same* rows.

    Returns dict with ``procrustes_disparity``, ``anchor_knn_overlap``,
    ``mean_drift`` plus per-point arrays (``drift`` over all rows,
    ``knn_overlap`` over the anchor subset with its ``anchor_idx``).
    """
    Za = read_rows(Z_a, np.arange(_n_rows(Z_a)))
    Zb = read_rows(Z_b, np.arange(_n_rows(Z_b)))
    Zb_aligned, disparity, _ = procrustes_align(Za, Zb)
    drift = per_point_drift(Za, Zb_aligned)

    anchor_idx = sample_indices(_n_rows(Z_a), n_anchors, seed=seed)
    mean_ov, per_ov = anchor_knn_overlap(Za, Zb, anchor_idx, k=k)
    return {
        "procrustes_disparity": disparity,
        "anchor_knn_overlap": mean_ov,
        "mean_drift": float(drift.mean()),
        "per_point_drift": drift,
        "per_anchor_knn_overlap": per_ov,
        "anchor_idx": anchor_idx,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cluster / task level
# ─────────────────────────────────────────────────────────────────────────────


def dataset_hash(X, n_sample: int = 2000) -> str:
    """Stable short hash of a dataset from a strided row sample + shape."""
    n = _n_rows(X)
    idx = np.linspace(0, n - 1, min(n_sample, n)).astype(np.int64)
    rows = read_rows(X, idx)
    h = hashlib.sha1()
    h.update(np.array([n, _dim(X)], dtype=np.int64).tobytes())
    h.update(rows.tobytes())
    return h.hexdigest()[:16]


def leiden_labels(X_high, k: int = 15, resolution: float = 1.0, seed: int = 42,
                  cache_dir: Optional[str] = None, ds_hash: Optional[str] = None):
    """Leiden-cluster the high-D kNN graph (lit ``07-umap-connectivity-clustering.md``).

    The high-D kNN graph is the source of truth. Labels are cached per dataset
    hash under ``cache_dir`` so the (potentially expensive) clustering runs once.
    Returns an int label array of length n.
    """
    if ds_hash is None:
        ds_hash = dataset_hash(X_high)
    cache_path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"leiden_{ds_hash}_k{k}_r{resolution}.npy")
        if os.path.exists(cache_path):
            return np.load(cache_path)

    import faiss
    import igraph as ig
    import leidenalg

    n = _n_rows(X_high)
    index = build_flat_l2_index(X_high)
    Xq = read_rows(X_high, np.arange(n))
    _, I = index.search(np.ascontiguousarray(Xq), k + 1)
    # build undirected edge list (drop self column)
    src = np.repeat(np.arange(n), k)
    dst = I[:, 1:].reshape(-1)
    edges = np.stack([src, dst], axis=1)
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]

    g = ig.Graph(n=n, edges=edges.tolist(), directed=False)
    part = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution, seed=seed,
    )
    labels = np.array(part.membership, dtype=np.int64)
    if cache_path is not None:
        np.save(cache_path, labels)
    return labels


def neighborhood_hit(Z_low, labels: np.ndarray, anchor_idx: np.ndarray, k: int = 15,
                     low_index=None):
    """Fraction of each anchor's 2D k-NN that share its cluster label."""
    if low_index is None:
        low_index = build_flat_l2_index(Z_low)
    q = read_rows(Z_low, anchor_idx)
    nn = _knn_excluding_self(low_index, q, anchor_idx, k)
    per = np.empty(len(anchor_idx), dtype=np.float64)
    for i, a in enumerate(anchor_idx):
        per[i] = np.mean(labels[nn[i]] == labels[a])
    return float(per.mean()), per


def silhouette_2d(Z_low, labels: np.ndarray, max_samples: int = 10_000, seed: int = 42):
    """Silhouette of the high-D cluster labels rendered in 2D."""
    from sklearn.metrics import silhouette_score

    n = _n_rows(Z_low)
    idx = sample_indices(n, max_samples, seed=seed)
    Z = read_rows(Z_low, idx)
    lab = labels[idx]
    uniq = np.unique(lab)
    if len(uniq) < 2:
        return float("nan")
    return float(silhouette_score(Z, lab))


def probe_gap(X_high, Z_low, labels: np.ndarray, max_samples: int = 20_000,
              seed: int = 42):
    """Linear-probe gap (lit ``113-linear-probes.md``).

    Logistic regression predicting the cluster label from high-D vs from 2D
    coords; the accuracy gap quantifies information lost by the projection.
    Returns dict with ``acc_high``, ``acc_low`` and ``gap`` (= acc_high - acc_low).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    n = _n_rows(X_high)
    idx = sample_indices(n, max_samples, seed=seed)
    lab = labels[idx]
    # keep classes with >=2 members so the split is well-defined
    uniq, counts = np.unique(lab, return_counts=True)
    keep = np.isin(lab, uniq[counts >= 2])
    idx, lab = idx[keep], lab[keep]
    if len(np.unique(lab)) < 2:
        return {"acc_high": float("nan"), "acc_low": float("nan"), "gap": float("nan")}

    Xh = read_rows(X_high, idx)
    Zl = read_rows(Z_low, idx)
    tr, te = train_test_split(np.arange(len(idx)), test_size=0.3, random_state=seed,
                              stratify=lab)

    def _acc(feat):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(feat[tr], lab[tr])
        return float((clf.predict(feat[te]) == lab[te]).mean())

    acc_high = _acc(Xh)
    acc_low = _acc(Zl)
    return {"acc_high": acc_high, "acc_low": acc_low, "gap": acc_high - acc_low}


# ─────────────────────────────────────────────────────────────────────────────
# Floors (PCA-2D, Gaussian random projection)
# ─────────────────────────────────────────────────────────────────────────────


def pca_2d(X_high, seed: int = 42) -> np.ndarray:
    """Deterministic PCA-2D floor embedding."""
    from sklearn.decomposition import PCA

    X = read_rows(X_high, np.arange(_n_rows(X_high)))
    return PCA(n_components=2, random_state=seed).fit_transform(X).astype(np.float32)


def random_projection_2d(X_high, seed: int = 42) -> np.ndarray:
    """Gaussian random-projection 2D floor embedding."""
    from sklearn.random_projection import GaussianRandomProjection

    X = read_rows(X_high, np.arange(_n_rows(X_high)))
    return GaussianRandomProjection(n_components=2, random_state=seed).fit_transform(X).astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# Global structure: Grassmann Score (lit ``172-generalizable-spectral-umap.md``)
# ─────────────────────────────────────────────────────────────────────────────


def _knn_graph_symmetric(P: np.ndarray, k: int):
    """Binary symmetric kNN adjacency (scipy CSR) from points ``P`` ``[m, d]``.

    Each point connects to its ``k`` nearest neighbours (self excluded); the
    result is symmetrised with a binary OR so ``W[i, j] == W[j, i]``.
    """
    import faiss
    from scipy.sparse import csr_matrix

    P = np.ascontiguousarray(P, dtype=np.float32)
    m = len(P)
    k = min(k, m - 1)
    index = faiss.IndexFlatL2(P.shape[1])
    index.add(P)
    _, I = index.search(P, k + 1)
    rows = np.repeat(np.arange(m), k)
    cols = I[:, 1:].reshape(-1)  # drop self column
    data = np.ones(len(rows), dtype=np.float64)
    A = csr_matrix((data, (rows, cols)), shape=(m, m))
    A = A.maximum(A.T)  # binary symmetrisation
    return A


def laplacian_eigenvectors(P: np.ndarray, k: int = 15, t: int = 10):
    """First ``t`` non-trivial eigenvectors of the normalized graph Laplacian.

    Builds a symmetric kNN graph of ``P``, forms the symmetric normalized
    Laplacian ``L_sym = I - D^{-1/2} W D^{-1/2}`` and returns its ``t``
    eigenvectors of smallest eigenvalue, **skipping** the trivial (constant)
    eigenvector at eigenvalue 0. Returns an ``[m, t]`` orthonormal array.

    Raises on eigensolver non-convergence (caller is expected to catch).
    """
    from scipy.sparse import csgraph
    from scipy.sparse.linalg import eigsh

    W = _knn_graph_symmetric(P, k)
    L = csgraph.laplacian(W, normed=True).tocsr().astype(np.float64)

    # Smallest ``t+1`` eigenvalues (including the trivial one). Shift-invert
    # around sigma just below 0 is the robust way to reach the low end of a
    # PSD Laplacian's spectrum; fall back to a plain smallest-algebraic solve.
    try:
        vals, vecs = eigsh(L, k=t + 1, sigma=-1e-3, which="LM", tol=0, maxiter=10_000)
    except Exception:
        vals, vecs = eigsh(L, k=t + 1, which="SA", tol=1e-6, maxiter=20_000)
    order = np.argsort(vals)
    vecs = vecs[:, order]
    return vecs[:, 1 : t + 1]  # drop the constant eigenvector


def grassmann_score(X_high, Z_low, t: int = 10, sample: int = 10_000, k: int = 15,
                    seed: int = 42):
    """Grassmann Score global-structure metric (note 172, lower is better).

    On a shared subsample of rows, compares the subspaces spanned by the first
    ``t`` non-trivial Laplacian eigenvectors of the high-D data ``X`` and of the
    2D embedding ``Y``. The principal angles ``theta_i`` between the two
    ``[m, t]`` eigenvector matrices come from the singular values (= ``cos``
    of the angles) of ``V_X^T V_Y``. Returns a dict with:

    * ``grassmann_distance`` = ``sqrt(sum theta_i^2)`` (the note's ``d_Gr``);
    * ``grassmann_affinity_error`` = ``1 - mean(cos^2 theta_i)`` (in ``[0, 1]``).

    Both are lower-is-better. On eigensolver non-convergence returns ``None``
    (with a warning) so the panel degrades gracefully.
    """
    n = _n_rows(X_high)
    idx = sample_indices(n, sample, seed=seed)
    Xs = read_rows(X_high, idx)
    Zs = read_rows(Z_low, idx)
    try:
        Vx = laplacian_eigenvectors(Xs, k=k, t=t)
        Vy = laplacian_eigenvectors(Zs, k=k, t=t)
    except Exception as e:  # ArpackError / ArpackNoConvergence / linalg failures
        import warnings
        warnings.warn(f"grassmann_score: Laplacian eigensolver failed ({e!r}); returning None")
        return None

    s = np.linalg.svd(Vx.T @ Vy, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)  # singular values are cos(theta_i) in [0, 1]
    thetas = np.arccos(s)
    return {
        "grassmann_distance": float(np.sqrt(np.sum(thetas ** 2))),
        "grassmann_affinity_error": float(1.0 - np.mean(s ** 2)),
        "t": int(t),
        "sample": int(len(idx)),
        "k": int(k),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Collapse index: degenerate-layout diagnostics
# ─────────────────────────────────────────────────────────────────────────────


def collapse_index(Z_low, grid: int = 64, clip=(0.5, 99.5)):
    """Diagnose degenerate collapsed layouts (tight clumps + empty space).

    Collapsed maps pack points into a few near-overlapping clumps surrounded by
    empty canvas, which inflates trustworthiness while destroying the usable
    spread of the map. Two complementary scalars capture that:

    * ``collapse_nn_ratio`` -- median 2D nearest-neighbour distance divided by
      the bounding-box diagonal of the percentile-clipped extent. Tiny values
      (points sit on top of each other relative to the canvas) flag collapse.
    * ``occupancy_entropy`` -- Shannon entropy of the point-count distribution
      over a ``grid x grid`` lattice on the clipped extent, normalized by
      ``log(grid*grid)``. ``1.0`` == points uniformly fill the canvas; ``-> 0``
      == everything crammed into a handful of cells.

    ``clip`` percentiles resist a few runaway outliers stretching the extent.
    Returns a dict with both scalars plus the raw extent for diagnostics.
    """
    import faiss

    n = _n_rows(Z_low)
    Z = read_rows(Z_low, np.arange(n)).astype(np.float32)
    x, y = Z[:, 0], Z[:, 1]
    xlo, xhi = np.percentile(x, clip)
    ylo, yhi = np.percentile(y, clip)
    xspan = max(xhi - xlo, 1e-12)
    yspan = max(yhi - ylo, 1e-12)
    diag = float(np.hypot(xspan, yspan))

    # (a) median nearest-neighbour distance / bbox diagonal
    index = faiss.IndexFlatL2(2)
    index.add(np.ascontiguousarray(Z))
    D, _ = index.search(np.ascontiguousarray(Z), 2)
    nn_dist = np.sqrt(np.maximum(D[:, 1], 0.0))
    nn_ratio = float(np.median(nn_dist) / diag) if diag > 0 else float("nan")

    # (b) occupancy entropy over the clipped grid (outliers clamp to edge cells)
    xi = np.clip(((x - xlo) / xspan * grid).astype(np.int64), 0, grid - 1)
    yi = np.clip(((y - ylo) / yspan * grid).astype(np.int64), 0, grid - 1)
    counts = np.zeros(grid * grid, dtype=np.float64)
    np.add.at(counts, xi * grid + yi, 1.0)
    occupied = counts[counts > 0]
    p = occupied / occupied.sum()
    entropy = float(-(p * np.log(p)).sum() / np.log(grid * grid))

    return {
        "collapse_nn_ratio": nn_ratio,
        "occupancy_entropy": entropy,
        "n_occupied_cells": int(len(occupied)),
        "bbox_diagonal": diag,
        "grid": int(grid),
    }



# ─────────────────────────────────────────────────────────────────────────────
# Spatially-resolved trustworthiness / continuity
# ─────────────────────────────────────────────────────────────────────────────


def spatial_tc_bins(Z_points: np.ndarray, per_T: np.ndarray, per_C: np.ndarray,
                    gridsize: int = 30):
    """Grid-bin per-point T/C for rendering a spatial heatmap (lit ``19-tvisne.md``).

    ``Z_points`` are the 2D coords of the points for which ``per_T``/``per_C``
    were computed. Returns a dict of arrays suitable for rendering: bin edges,
    per-bin mean T, mean C and counts (NaN where empty).
    """
    Z = np.asarray(Z_points, dtype=np.float64)
    x, y = Z[:, 0], Z[:, 1]
    x_edges = np.linspace(x.min(), x.max(), gridsize + 1)
    y_edges = np.linspace(y.min(), y.max(), gridsize + 1)
    xi = np.clip(np.digitize(x, x_edges) - 1, 0, gridsize - 1)
    yi = np.clip(np.digitize(y, y_edges) - 1, 0, gridsize - 1)

    counts = np.zeros((gridsize, gridsize), dtype=np.int64)
    sumT = np.zeros((gridsize, gridsize), dtype=np.float64)
    sumC = np.zeros((gridsize, gridsize), dtype=np.float64)
    np.add.at(counts, (xi, yi), 1)
    np.add.at(sumT, (xi, yi), per_T)
    np.add.at(sumC, (xi, yi), per_C)
    with np.errstate(invalid="ignore"):
        meanT = np.where(counts > 0, sumT / counts, np.nan)
        meanC = np.where(counts > 0, sumC / counts, np.nan)
    return {
        "x_edges": x_edges,
        "y_edges": y_edges,
        "count": counts,
        "mean_trustworthiness": meanT,
        "mean_continuity": meanC,
        "gridsize": gridsize,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PanelConfig:
    k_list: tuple = (10, 50)
    n_anchors: int = 10_000
    tc_subsample: int = 5_000       # points for exact rank-based T/C
    n_pairs: int = 100_000          # for Spearman distance correlation
    n_triplets: int = 100_000
    density_k: int = 15
    leiden_k: int = 15
    leiden_resolution: float = 1.0
    silhouette_max: int = 10_000
    probe_max: int = 20_000
    seed: int = 42
    do_clusters: bool = True
    do_density: bool = True
    do_floors: bool = False
    do_grassmann: bool = True
    do_collapse: bool = True
    grassmann_t: int = 10
    grassmann_sample: int = 10_000
    grassmann_k: int = 15
    collapse_grid: int = 64
    cache_dir: Optional[str] = None
    spatial_gridsize: int = 30


def score_map(X_high, Z_low, config: Optional[PanelConfig] = None, labels=None):
    """Run the full map-fidelity panel (families A + cluster level) on one map.

    Parameters
    ----------
    X_high : array-like ``[n, d]`` float32 (may be memmap / concatenator).
    Z_low  : array-like ``[n, 2]``.
    config : :class:`PanelConfig`.
    labels : optional precomputed cluster labels ``[n]`` (else Leiden is run when
             ``config.do_clusters``).

    Returns ``(metrics, per_point_df, extras)`` where ``metrics`` is a nested
    JSON-serialisable dict, ``per_point_df`` is a pandas DataFrame of per-row
    diagnostics, and ``extras`` holds arrays not written to JSON (spatial bins,
    cluster labels).
    """
    import pandas as pd

    if config is None:
        config = PanelConfig()
    n = _n_rows(X_high)
    assert _n_rows(Z_low) == n, "coords and embeddings must have the same #rows"

    t0 = time.time()
    metrics: dict = {"n": n, "dim": _dim(X_high)}
    extras: dict = {}

    # shared indices
    high_index = build_flat_l2_index(X_high)
    low_index = build_flat_l2_index(Z_low)

    anchor_idx = sample_indices(n, config.n_anchors, seed=config.seed)
    Z_anchor = read_rows(Z_low, anchor_idx)

    # per-point columns keyed by row id (anchor subset)
    per_cols: dict = {"row_id": anchor_idx}

    # ── Family A: kNN recall at each k ──
    metrics["knn_recall"] = {}
    for k in config.k_list:
        true_nn = knn_ids_full(X_high, anchor_idx, k, index=high_index)
        mean_r, per_r = knn_recall(X_high, Z_low, anchor_idx, k,
                                   high_index=high_index, low_index=low_index,
                                   true_neighbors=true_nn)
        metrics["knn_recall"][f"k{k}"] = mean_r
        per_cols[f"knn_recall_k{k}"] = per_r

    # ── Trustworthiness + continuity (rank-based, subsample) ──
    tc_idx = sample_indices(n, config.tc_subsample, seed=config.seed + 1)
    metrics["trustworthiness"] = {}
    metrics["continuity"] = {}
    tc_first = None
    for k in config.k_list:
        tc = trustworthiness_continuity(X_high, Z_low, k, idx=tc_idx)
        metrics["trustworthiness"][f"k{k}"] = tc["trustworthiness"]
        metrics["continuity"][f"k{k}"] = tc["continuity"]
        if tc_first is None:
            tc_first = tc  # keep the first k for spatial bins / per-point
    extras["tc"] = tc_first

    # ── Global structure ──
    metrics["spearman_distance_correlation"] = spearman_distance_correlation(
        X_high, Z_low, n_pairs=config.n_pairs, seed=config.seed)
    metrics["triplet_accuracy"] = triplet_accuracy(
        X_high, Z_low, n_triplets=config.n_triplets, seed=config.seed)

    # ── Density ──
    if config.do_density:
        corr, log_rh, log_rl = density_preservation(
            X_high, Z_low, anchor_idx, k=config.density_k,
            high_index=high_index, low_index=low_index)
        metrics["density_preservation"] = corr
        per_cols["density_log_radius_high"] = log_rh
        per_cols["density_log_radius_low"] = log_rl

    # ── Global structure: Grassmann Score (note 172) ──
    if config.do_grassmann:
        metrics["grassmann"] = grassmann_score(
            X_high, Z_low, t=config.grassmann_t, sample=config.grassmann_sample,
            k=config.grassmann_k, seed=config.seed)  # None on eigensolver failure

    # ── Collapse index (degenerate-layout diagnostics) ──
    if config.do_collapse:
        metrics["collapse"] = collapse_index(Z_low, grid=config.collapse_grid)

    # ── Cluster / task level ──
    if config.do_clusters:
        if labels is None:
            labels = leiden_labels(
                X_high, k=config.leiden_k, resolution=config.leiden_resolution,
                seed=config.seed, cache_dir=config.cache_dir)
        extras["labels"] = labels
        metrics["n_clusters"] = int(len(np.unique(labels)))
        nh_mean, nh_per = neighborhood_hit(Z_low, labels, anchor_idx,
                                           k=config.leiden_k, low_index=low_index)
        metrics["neighborhood_hit"] = nh_mean
        per_cols["neighborhood_hit"] = nh_per
        per_cols["cluster_label"] = labels[anchor_idx].astype(np.float64)
        metrics["silhouette_2d"] = silhouette_2d(
            Z_low, labels, max_samples=config.silhouette_max, seed=config.seed)
        metrics["probe_gap"] = probe_gap(
            X_high, Z_low, labels, max_samples=config.probe_max, seed=config.seed)

    # ── per-point diagnostics for the T/C subsample (spatially resolved) ──
    # merge T/C per-point into the diagnostics frame on row_id
    per_df = pd.DataFrame({c: np.asarray(v) for c, v in per_cols.items()})
    if tc_first is not None:
        tc_df = pd.DataFrame({
            "row_id": tc_first["idx"],
            "trustworthiness": tc_first["per_point_trustworthiness"],
            "continuity": tc_first["per_point_continuity"],
        })
        per_df = per_df.merge(tc_df, on="row_id", how="outer").sort_values("row_id").reset_index(drop=True)

        # spatial bins from the T/C subsample
        Z_tc = read_rows(Z_low, tc_first["idx"])
        extras["spatial_bins"] = spatial_tc_bins(
            Z_tc, tc_first["per_point_trustworthiness"],
            tc_first["per_point_continuity"], gridsize=config.spatial_gridsize)

    metrics["elapsed_s"] = round(time.time() - t0, 2)
    return metrics, per_df, extras


def compute_floors(X_high, config: Optional[PanelConfig] = None):
    """Run PCA-2D and Gaussian-random-projection floors through the panel.

    Returns ``{"pca": metrics, "random_projection": metrics}``. Cluster metrics
    are disabled for speed (floors are a fidelity reference). Anything a trained
    map does not clearly beat is not working (plan §2.3).
    """
    if config is None:
        config = PanelConfig()
    floor_cfg = PanelConfig(**{**config.__dict__, "do_clusters": False, "do_floors": False})
    out = {}
    for name, fn in (("pca", pca_2d), ("random_projection", random_projection_2d)):
        Z = fn(X_high, seed=config.seed)
        m, _, _ = score_map(X_high, Z, config=floor_cfg)
        out[name] = m
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Loading / CLI
# ─────────────────────────────────────────────────────────────────────────────


def load_coords(path: str):
    """Load a coordinates parquet. Expects ``x``/``y`` columns (as written by the
    latent-scope exporter). An optional ``row_id`` column maps coords rows to
    embedding rows; otherwise positional alignment is assumed. Returns
    ``(Z [n,2] float32, row_id [n] int64 or None)``.
    """
    import pandas as pd

    df = pd.read_parquet(path)
    cols = {c.lower(): c for c in df.columns}
    xcol = cols.get("x")
    ycol = cols.get("y")
    if xcol is None or ycol is None:
        raise ValueError(f"coords parquet {path} must have x and y columns; got {list(df.columns)}")
    Z = np.stack([df[xcol].to_numpy(), df[ycol].to_numpy()], axis=1).astype(np.float32)
    row_id = df[cols["row_id"]].to_numpy().astype(np.int64) if "row_id" in cols else None
    return Z, row_id


def load_embeddings(path: str, dim: Optional[int] = None):
    """Load embeddings as a memmap-backed array.

    ``path`` may be a single ``.npy`` file (opened with ``mmap_mode='r'``) or a
    directory of shards (loaded via :class:`MemmapArrayConcatenator`, read-only).
    """
    if os.path.isdir(path):
        from basemap.data_loader import MemmapArrayConcatenator
        import glob

        if dim is None:
            first = sorted(glob.glob(os.path.join(path, "*.npy")))
            if not first:
                raise ValueError(f"no .npy shards in {path}")
            dim = int(np.load(first[0], mmap_mode="r").shape[1])
        return MemmapArrayConcatenator([path], input_dim=dim)
    return np.load(path, mmap_mode="r")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serialisable: {type(o)}")


def check_unit_norm(X, sample=10000, tol=0.01, seed=0):
    """Sample row norms and report unit-norm status.

    The whole pipeline treats L2 / cosine / dot as rank-equivalent, which is
    only true for unit-normalized embeddings. Any space that fails this check
    must be explicitly normalized (or the metric choice reconsidered) before
    its numbers are trusted — see plan-basemap-atlas.md §3.
    """
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(_n_rows(X), min(sample, _n_rows(X)), replace=False))
    norms = np.linalg.norm(read_rows(X, idx).astype(np.float32), axis=1)
    stats = {"mean": float(norms.mean()), "std": float(norms.std()),
             "min": float(norms.min()), "max": float(norms.max())}
    stats["is_unit_norm"] = bool(abs(stats["mean"] - 1.0) < tol and stats["std"] < tol)
    if not stats["is_unit_norm"]:
        import warnings
        warnings.warn(
            f"embeddings are NOT unit-normalized (mean norm {stats['mean']:.4f}, "
            f"std {stats['std']:.4f}): L2/cosine/dot are no longer rank-equivalent; "
            "distance-based metrics depend on the metric convention. Normalize the "
            "space or interpret with care.")
    return stats


def cmd_score(args):
    from basemap.round0005_retirement import refuse_retired_launcher
    refuse_retired_launcher("basemap/eval.py")


def _cmd_score_fixture_only(args):
    """Private CPU-only compatibility body; never registered as a CLI command."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("eval fixture scorer requires CUDA_VISIBLE_DEVICES='' exactly")
    X = load_embeddings(args.embeddings, dim=args.dim)
    Z, row_id = load_coords(args.coords)

    # Align embeddings to coords rows.
    if row_id is not None:
        # read only the referenced rows into a compact array (lazy)
        X = read_rows(X, row_id)
    elif _n_rows(X) != len(Z):
        raise ValueError(
            f"row count mismatch: embeddings {_n_rows(X)} vs coords {len(Z)}; "
            "provide a row_id column in the coords parquet to map them")

    config = PanelConfig(
        n_anchors=args.n_anchors,
        tc_subsample=args.tc_subsample,
        do_floors=args.floors,
        do_clusters=not args.no_clusters,
        do_grassmann=not args.no_grassmann,
        do_collapse=not args.no_collapse,
        grassmann_t=args.grassmann_t,
        grassmann_sample=args.grassmann_sample,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )
    if args.k:
        config.k_list = tuple(args.k)

    metrics, per_df, extras = score_map(X, Z, config=config)
    metrics["norm_check"] = check_unit_norm(X)

    if args.floors:
        metrics["floors"] = compute_floors(X, config=config)

    # write spatial bins alongside metrics (as lists)
    if "spatial_bins" in extras:
        sb = extras["spatial_bins"]
        metrics["spatial_bins"] = {
            "gridsize": sb["gridsize"],
            "count": sb["count"].tolist(),
            "mean_trustworthiness": np.where(np.isnan(sb["mean_trustworthiness"]), None,
                                             sb["mean_trustworthiness"]).tolist(),
            "mean_continuity": np.where(np.isnan(sb["mean_continuity"]), None,
                                        sb["mean_continuity"]).tolist(),
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2, default=_json_default)
    print(f"wrote metrics -> {args.out}")

    if args.per_point:
        per_df.to_parquet(args.per_point, index=False)
        print(f"wrote per-point diagnostics -> {args.per_point} "
              f"({len(per_df)} rows, cols={list(per_df.columns)})")

    # brief summary
    print(json.dumps({k: v for k, v in metrics.items()
                      if k not in ("spatial_bins", "floors")}, indent=2, default=_json_default))


def build_parser():
    p = argparse.ArgumentParser(prog="basemap.eval", description="Basemap evaluation harness")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("score", help="score a single map's fidelity")
    s.add_argument("--coords", required=True, help="coords parquet with x,y (+ optional row_id)")
    s.add_argument("--embeddings", required=True, help=".npy file or directory of shards")
    s.add_argument("--out", required=True, help="output metrics.json")
    s.add_argument("--per-point", default=None, help="output per-point diagnostics parquet")
    s.add_argument("--dim", type=int, default=None, help="embedding dim (inferred if omitted)")
    s.add_argument("--n-anchors", type=int, default=10_000)
    s.add_argument("--tc-subsample", type=int, default=5_000)
    s.add_argument("--k", type=int, nargs="*", default=None, help="k values (default 10 50)")
    s.add_argument("--floors", action="store_true", help="also compute PCA / random-projection floors")
    s.add_argument("--no-clusters", action="store_true", help="skip Leiden cluster metrics")
    s.add_argument("--no-grassmann", action="store_true", help="skip Grassmann global-structure score")
    s.add_argument("--no-collapse", action="store_true", help="skip collapse-index diagnostics")
    s.add_argument("--grassmann-t", type=int, default=10, help="# Laplacian eigenvectors for Grassmann score")
    s.add_argument("--grassmann-sample", type=int, default=10_000, help="subsample size for Grassmann score")
    s.add_argument("--cache-dir", default=None, help="dir to cache Leiden labels")
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_score)
    return p


def main(argv=None):
    from basemap.round0005_retirement import refuse_retired_launcher
    refuse_retired_launcher("basemap/eval.py")
    args = build_parser().parse_args(argv)
    args.func(args)


def _main_fixture_only(argv=None):
    """Exercise the legacy CPU implementation without exposing an executable lane."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("eval fixture main requires CUDA_VISIBLE_DEVICES='' exactly")
    args = build_parser().parse_args(argv)
    if args.func is not cmd_score:
        raise RuntimeError("eval fixture parser selected a non-fixture command")
    return _cmd_score_fixture_only(args)


if __name__ == "__main__":
    main()
