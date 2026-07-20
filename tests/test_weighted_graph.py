"""Focused tests for experiments/build_weighted_graph.py.

Small synthetic checks that (a) the symmetrization t-conorm join reproduces
scipy's, (b) partitioning is join-invariant, (c) the full per-chunk pipeline
matches umap.fuzzy_simplicial_set on identical topology, and (d) the sharded
gather is order-preserving.
"""
import os
import sys

import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.build_weighted_graph import (
    ShardedEmbeddings, symmetrize_bucket, _pair_bucket,
    fuzzy_directed_from_knn)
from experiments.weighted_graph_validate import build_exact_knn_gpu, _knn_to_16col


def _scipy_tconorm(rows, cols, vals, n):
    P = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    P.eliminate_zeros()
    T = P.transpose()
    S = P + T - P.multiply(T)
    S.eliminate_zeros()
    return S.tocoo()


def test_symmetrize_bucket_matches_scipy():
    # hand-crafted directed membership with mutual + one-directional edges
    rows = np.array([0, 0, 1, 2, 3], dtype=np.int32)
    cols = np.array([1, 2, 0, 3, 2], dtype=np.int32)
    vals = np.array([0.8, 0.5, 0.4, 0.9, 0.3], dtype=np.float32)
    n = 4
    ui, uj, uw = symmetrize_bucket(rows, cols, vals, n)
    mine = {(int(a), int(b)): float(w) for a, b, w in zip(ui, uj, uw)}
    coo = _scipy_tconorm(rows, cols, vals, n)
    ref = {(int(r), int(c)): float(v) for r, c, v in zip(coo.row, coo.col, coo.data)}
    assert set(mine) == set(ref)
    for e in ref:
        assert abs(mine[e] - ref[e]) < 1e-6, (e, mine[e], ref[e])
    # mutual pair (0,1)/(1,0): sym = .8 + .4 - .8*.4 = 0.88 both directions
    assert abs(mine[(0, 1)] - 0.88) < 1e-6
    assert abs(mine[(1, 0)] - 0.88) < 1e-6
    # one-directional (0,2)/(2,0)=(only 0->2 .5): sym=.5 both directions, and
    # (2,3)/(3,2) mutual = .9+.3-.27=0.93
    assert abs(mine[(2, 0)] - 0.5) < 1e-6
    assert abs(mine[(2, 3)] - 0.93) < 1e-6


def test_partition_join_invariance():
    rng = np.random.RandomState(0)
    n = 500
    k = 10
    src = np.repeat(np.arange(n), k).astype(np.int32)
    dst = rng.randint(0, n, size=n * k).astype(np.int32)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    w = rng.uniform(0.01, 1.0, size=len(src)).astype(np.float32)
    # single-bucket reference
    ui0, uj0, uw0 = symmetrize_bucket(src, dst, w, n)
    ref = {(int(a), int(b)): float(x) for a, b, x in zip(ui0, uj0, uw0)}
    # partitioned
    P = 8
    a = np.minimum(src, dst); b = np.maximum(src, dst)
    buckets = _pair_bucket(a, b, n, P)
    merged = {}
    for p in range(P):
        m = buckets == p
        if not m.any():
            continue
        ui, uj, uw = symmetrize_bucket(src[m], dst[m], w[m], n)
        for aa, bb, xx in zip(ui, uj, uw):
            merged[(int(aa), int(bb))] = float(xx)
    assert set(merged) == set(ref)
    for e in ref:
        assert abs(merged[e] - ref[e]) < 1e-6


def test_full_pipeline_matches_umap():
    pytest.importorskip("torch")
    from umap.umap_ import fuzzy_simplicial_set
    rng = np.random.RandomState(1)
    n = 800
    k = 15
    X = rng.randn(n, 48).astype(np.float32)
    neighbors, dists = build_exact_knn_gpu(X, k, device="cpu")
    knn_i, knn_d = _knn_to_16col(neighbors, dists)
    n_neighbors = k + 1
    rows, cols, vals, _, _, _ = fuzzy_directed_from_knn(knn_i, knn_d, n_neighbors)
    ui, uj, uw = symmetrize_bucket(rows, cols, vals, n)
    mine = {(int(a), int(b)): float(x) for a, b, x in zip(ui, uj, uw)}
    graph, _, _ = fuzzy_simplicial_set(
        X, n_neighbors=n_neighbors, random_state=42, metric="cosine",
        knn_indices=knn_i.astype(np.int64), knn_dists=knn_d.astype(np.float64))
    coo = graph.tocoo()
    ref = {(int(r), int(c)): float(v) for r, c, v in zip(coo.row, coo.col, coo.data)}
    assert set(mine) == set(ref), (len(mine), len(ref))
    maxdiff = max(abs(mine[e] - ref[e]) for e in ref)
    assert maxdiff < 1e-4, maxdiff


def test_membership_matches_umap_full_and_offset():
    """fuzzy_directed_from_knn must (a) equal umap.compute_membership_strengths on
    a full 0..n block and (b) emit correct GLOBAL source ids for a sub-block that
    does NOT start at 0 — the bug that corrupted the first 30M build."""
    pytest.importorskip("torch")
    from umap.umap_ import smooth_knn_dist, compute_membership_strengths
    rng = np.random.RandomState(7)
    n, k = 600, 15
    X = rng.randn(n, 32).astype(np.float32)
    neighbors, dists = build_exact_knn_gpu(X, k, device="cpu")
    knn_i, knn_d = _knn_to_16col(neighbors, dists)  # col0 = global self id
    nn = k + 1

    # (a) full block vs umap's own kernel
    sig, rho = smooth_knn_dist(knn_d.astype(np.float32), float(nn), local_connectivity=1.0)
    ur, uc, uv, _ = compute_membership_strengths(knn_i, knn_d.astype(np.float32), sig, rho, False)
    umap_full = {(int(r), int(c)): float(v)
                 for r, c, v in zip(ur, uc, uv) if v > 0}
    mr, mc, mv, _, _, _ = fuzzy_directed_from_knn(knn_i, knn_d, nn)
    mine_full = {(int(a), int(b)): float(w) for a, b, w in zip(mr, mc, mv)}
    assert set(mine_full) == set(umap_full)
    assert max(abs(mine_full[e] - umap_full[e]) for e in umap_full) < 1e-4

    # (b) offset sub-block [200:260): sources must be the GLOBAL ids 200..259 and
    #     equal the full-block edges restricted to those sources.
    lo, hi = 200, 260
    sr, sc, sv, _, _, _ = fuzzy_directed_from_knn(knn_i[lo:hi], knn_d[lo:hi], nn)
    sub = {(int(a), int(b)): float(w) for a, b, w in zip(sr, sc, sv)}
    assert set(int(a) for a in sr) <= set(range(lo, hi))
    assert min(int(a) for a in sr) >= lo and max(int(a) for a in sr) < hi
    expected = {e: w for e, w in umap_full.items() if lo <= e[0] < hi}
    assert set(sub) == set(expected), (len(sub), len(expected))
    assert max(abs(sub[e] - expected[e]) for e in expected) < 1e-4


def test_sharded_gather_order(tmp_path):
    a = np.arange(20 * 4, dtype=np.float16).reshape(20, 4)
    p0 = tmp_path / "s0.npy"; p1 = tmp_path / "s1.npy"
    np.save(p0, a[:12]); np.save(p1, a[12:])
    emb = ShardedEmbeddings([str(p0), str(p1)], expected_dim=4)
    assert len(emb) == 20
    ids = np.array([15, 0, 11, 12, 3])
    got = emb.gather(ids, out_dtype=np.float32)
    assert np.array_equal(got, a[ids].astype(np.float32))
