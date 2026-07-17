"""Tests for the consolidated basemap evaluation harness (basemap/eval.py).

Uses synthetic data (swiss roll, blobs). Key invariants checked:

* trustworthiness matches ``sklearn.manifold.trustworthiness`` within 1e-6, and
  continuity matches sklearn's trustworthiness with the spaces swapped;
* a perfect (identity) embedding scores ~1.0 on recall / T / C, while a random
  embedding scores near the theoretical floor;
* Procrustes alignment recovers a known rotation exactly.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basemap import eval as ev  # noqa: E402


# ─── fixtures ────────────────────────────────────────────────────────────────


def swiss_roll(n=800, seed=0):
    from sklearn.datasets import make_swiss_roll

    X, color = make_swiss_roll(n_samples=n, noise=0.05, random_state=seed)
    return X.astype(np.float32), color


def blobs_hi(n=600, d=20, centers=6, seed=0):
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=n, n_features=d, centers=centers,
                      cluster_std=1.0, random_state=seed)
    return X.astype(np.float32), y


def blobs_2d(n=600, centers=6, seed=0):
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=n, n_features=2, centers=centers,
                      cluster_std=1.0, random_state=seed)
    return X.astype(np.float32), y


# ─── trustworthiness / continuity vs sklearn ─────────────────────────────────


@pytest.mark.parametrize("k", [5, 10, 15])
def test_trustworthiness_matches_sklearn(k):
    from sklearn.manifold import trustworthiness as sk_tw

    X, _ = swiss_roll(n=400, seed=1)
    rng = np.random.RandomState(2)
    Z = (X[:, :2] + 0.3 * rng.randn(len(X), 2)).astype(np.float32)  # a non-trivial embedding

    tc = ev.trustworthiness_continuity(X, Z, k=k)
    sk_T = sk_tw(X, Z, n_neighbors=k)
    sk_C = sk_tw(Z, X, n_neighbors=k)  # continuity == trustworthiness with spaces swapped

    assert abs(tc["trustworthiness"] - sk_T) < 1e-6
    assert abs(tc["continuity"] - sk_C) < 1e-6


def test_per_point_tc_mean_equals_global():
    X, _ = swiss_roll(n=300, seed=3)
    Z = X[:, :2].astype(np.float32)
    tc = ev.trustworthiness_continuity(X, Z, k=10)
    assert abs(tc["per_point_trustworthiness"].mean() - tc["trustworthiness"]) < 1e-9
    assert abs(tc["per_point_continuity"].mean() - tc["continuity"]) < 1e-9


# ─── perfect vs random embedding ─────────────────────────────────────────────


def test_perfect_embedding_scores_high():
    # 2D blobs: the "embedding" is the data itself -> near-perfect fidelity.
    X, _ = blobs_2d(n=500, centers=6, seed=4)
    Z = X.copy()

    anchors = ev.sample_indices(len(X), 200, seed=0)
    mean_r, _ = ev.knn_recall(X, Z, anchors, k=10)
    assert mean_r > 0.999

    tc = ev.trustworthiness_continuity(X, Z, k=10, idx=ev.sample_indices(len(X), 300, seed=0))
    assert tc["trustworthiness"] > 0.999
    assert tc["continuity"] > 0.999

    assert ev.triplet_accuracy(X, Z, n_triplets=20_000) > 0.999
    assert ev.spearman_distance_correlation(X, Z, n_pairs=20_000) > 0.999


def test_random_embedding_scores_near_floor():
    X, _ = blobs_hi(n=500, d=20, centers=6, seed=5)
    rng = np.random.RandomState(7)
    Z = rng.randn(len(X), 2).astype(np.float32)

    anchors = ev.sample_indices(len(X), 200, seed=0)
    mean_r, _ = ev.knn_recall(X, Z, anchors, k=10)
    # theoretical floor for recall of 10 neighbours out of ~500 is ~10/500 = 0.02
    assert mean_r < 0.10

    tc = ev.trustworthiness_continuity(X, Z, k=10, idx=ev.sample_indices(len(X), 300, seed=0))
    assert tc["trustworthiness"] < 0.75  # random ~0.5, well below a real map

    # triplet accuracy of a random map ~ 0.5
    tri = ev.triplet_accuracy(X, Z, n_triplets=20_000)
    assert 0.4 < tri < 0.6


# ─── kNN recall uses full corpus, not the anchor subsample ───────────────────


def test_knn_recall_full_corpus():
    # Two well-separated blobs in 2D; a perfect embedding gives recall ~1 even
    # when only a handful of anchors are scored -- neighbours come from the full
    # set, not the anchor subset.
    X, _ = blobs_2d(n=400, centers=8, seed=9)
    Z = X.copy()
    anchors = ev.sample_indices(len(X), 5, seed=1)  # tiny anchor set
    mean_r, per = ev.knn_recall(X, Z, anchors, k=10)
    assert per.shape == (5,)
    assert mean_r > 0.999


# ─── Procrustes recovers a known rotation ────────────────────────────────────


def test_procrustes_recovers_rotation():
    rng = np.random.RandomState(11)
    A = rng.randn(200, 2)
    theta = 0.7
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    t = np.array([3.0, -1.5])
    B = A @ R.T + t  # rotate + translate

    B_aligned, disparity, tr = ev.procrustes_align(A, B)
    assert disparity < 1e-12
    assert np.allclose(B_aligned, A, atol=1e-8)


def test_procrustes_handles_scale_and_reflection():
    rng = np.random.RandomState(12)
    A = rng.randn(150, 2)
    # reflection + scale + rotation
    theta = -1.2
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    reflect = np.array([[1, 0], [0, -1]])
    B = (A @ (R @ reflect).T) * 2.5 + np.array([1.0, 1.0])
    B_aligned, disparity, tr = ev.procrustes_align(A, B)
    assert disparity < 1e-10
    assert np.allclose(B_aligned, A, atol=1e-6)


# ─── stability: identical maps ───────────────────────────────────────────────


def test_compare_identical_maps():
    X, _ = blobs_2d(n=300, centers=5, seed=13)
    out = ev.compare_maps(X, X.copy(), k=10, n_anchors=100)
    assert out["procrustes_disparity"] < 1e-12
    assert out["mean_drift"] < 1e-6
    assert out["anchor_knn_overlap"] > 0.999


def test_compare_rotated_map_is_stable():
    X, _ = blobs_2d(n=300, centers=5, seed=14)
    theta = 0.5
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    Xr = (X @ R.T + np.array([5.0, 2.0])).astype(np.float32)
    out = ev.compare_maps(X, Xr, k=10, n_anchors=100)
    # rotation is gauge freedom -> perfectly stable after alignment
    assert out["procrustes_disparity"] < 1e-8
    assert out["anchor_knn_overlap"] > 0.999  # kNN overlap is rotation-invariant


# ─── projection fidelity ─────────────────────────────────────────────────────


def test_projection_fidelity_perfect_and_random():
    X, _ = blobs_2d(n=500, centers=6, seed=15)
    n_train = 400
    X_train, X_query = X[:n_train], X[n_train:]
    Z_train = X_train.copy()

    # perfect projection: query 2D == query high-D (which is 2D here)
    Z_query = X_query.copy()
    mean_r, per = ev.projection_fidelity(X_train, Z_train, X_query, Z_query, k=10)
    assert mean_r > 0.999
    assert per.shape == (len(X_query),)

    # random projection of the queries -> near floor
    rng = np.random.RandomState(3)
    Z_query_rand = rng.randn(len(X_query), 2).astype(np.float32)
    mean_r2, _ = ev.projection_fidelity(X_train, Z_train, X_query, Z_query_rand, k=10)
    assert mean_r2 < 0.15


# ─── density preservation ────────────────────────────────────────────────────


def test_density_preservation_perfect():
    X, _ = blobs_2d(n=400, centers=6, seed=16)
    Z = X.copy()
    anchors = ev.sample_indices(len(X), 200, seed=0)
    corr, log_rh, log_rl = ev.density_preservation(X, Z, anchors, k=15)
    assert corr > 0.999
    assert log_rh.shape == (200,)


# ─── clustering / floors / full panel smoke test ─────────────────────────────


def test_leiden_and_cluster_metrics(tmp_path):
    X, y = blobs_hi(n=400, d=20, centers=6, seed=17)
    labels = ev.leiden_labels(X, k=15, cache_dir=str(tmp_path))
    assert len(labels) == len(X)
    assert len(np.unique(labels)) >= 2
    # cache hit returns same labels
    labels2 = ev.leiden_labels(X, k=15, cache_dir=str(tmp_path))
    assert np.array_equal(labels, labels2)

    Z = X[:, :2].astype(np.float32)  # PCA-ish 2D view
    anchors = ev.sample_indices(len(X), 200, seed=0)
    nh_mean, _ = ev.neighborhood_hit(Z, labels, anchors, k=15)
    assert 0.0 <= nh_mean <= 1.0
    pg = ev.probe_gap(X, Z, labels, max_samples=400)
    assert pg["acc_high"] >= pg["acc_low"] - 0.2  # high-D probe should be >= 2D (allow slack)


def test_score_map_full_panel(tmp_path):
    X, y = blobs_hi(n=500, d=20, centers=6, seed=18)
    Z = X[:, :2].astype(np.float32)
    cfg = ev.PanelConfig(n_anchors=200, tc_subsample=300, n_pairs=5000,
                         n_triplets=5000, cache_dir=str(tmp_path))
    metrics, per_df, extras = ev.score_map(X, Z, config=cfg)

    assert "knn_recall" in metrics and "k10" in metrics["knn_recall"]
    assert "trustworthiness" in metrics and "continuity" in metrics
    assert "spearman_distance_correlation" in metrics
    assert "triplet_accuracy" in metrics
    assert "density_preservation" in metrics
    assert "neighborhood_hit" in metrics
    assert "probe_gap" in metrics
    assert "row_id" in per_df.columns
    assert "knn_recall_k10" in per_df.columns
    assert "trustworthiness" in per_df.columns
    assert "spatial_bins" in extras
    sb = extras["spatial_bins"]
    assert sb["mean_trustworthiness"].shape == (cfg.spatial_gridsize, cfg.spatial_gridsize)


def test_floors_run(tmp_path):
    X, y = blobs_hi(n=400, d=20, centers=6, seed=19)
    cfg = ev.PanelConfig(n_anchors=150, tc_subsample=200, n_pairs=3000,
                         n_triplets=3000, cache_dir=str(tmp_path))
    floors = ev.compute_floors(X, config=cfg)
    assert "pca" in floors and "random_projection" in floors
    # PCA should beat random projection on kNN recall
    assert floors["pca"]["knn_recall"]["k10"] >= floors["random_projection"]["knn_recall"]["k10"]


# ─── CLI / IO round-trip ─────────────────────────────────────────────────────


def test_private_cpu_fixture_score_roundtrip(tmp_path):
    import pandas as pd

    X, y = blobs_hi(n=400, d=16, centers=6, seed=20)
    Z = X[:, :2].astype(np.float32)

    emb_path = tmp_path / "emb.npy"
    np.save(emb_path, X)
    coords_path = tmp_path / "coords.parquet"
    pd.DataFrame({"x": Z[:, 0], "y": Z[:, 1]}).to_parquet(coords_path)

    out = tmp_path / "metrics.json"
    per = tmp_path / "diag.parquet"
    ev._main_fixture_only([
        "score", "--coords", str(coords_path), "--embeddings", str(emb_path),
        "--out", str(out), "--per-point", str(per),
        "--n-anchors", "150", "--tc-subsample", "200",
        "--cache-dir", str(tmp_path / "cache"),
    ])
    assert out.exists() and per.exists()
    import json
    m = json.load(open(out))
    assert "knn_recall" in m
    d = pd.read_parquet(per)
    assert "row_id" in d.columns
