"""Tests for the extra global-structure / degeneracy metrics in basemap/eval.py.

Covers the two metrics added on top of the core fidelity panel:

* **Grassmann Score** (note 172) -- ~0 when the 2D embedding preserves the
  spectral subspace of low-dim data (identity-like), clearly > 0 for a random
  2D scatter that destroys global structure.
* **Collapse index** -- cleanly separates a healthy, uniformly-spread blob from
  a synthetic collapsed layout (a few tight gaussian clumps in a huge extent):
  the collapsed map has a much smaller ``collapse_nn_ratio`` and much lower
  ``occupancy_entropy``.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basemap import eval as ev  # noqa: E402


# ─── fixtures ────────────────────────────────────────────────────────────────


def blobs_2d(n=2000, centers=8, seed=0):
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=n, n_features=2, centers=centers,
                      cluster_std=1.0, random_state=seed)
    return X.astype(np.float32), y


def collapsed_layout(n=2000, n_clumps=50, extent=1000.0, clump_std=0.02, seed=0):
    """A degenerate layout: ``n_clumps`` extremely tight gaussian blobs scattered
    across a huge empty canvas -- the pathology the collapse index must catch."""
    rng = np.random.RandomState(seed)
    centers = rng.uniform(0, extent, size=(n_clumps, 2))
    assign = rng.randint(0, n_clumps, size=n)
    Z = centers[assign] + clump_std * rng.randn(n, 2)
    return Z.astype(np.float32)


# ─── Grassmann Score ─────────────────────────────────────────────────────────


def test_grassmann_identity_near_zero_vs_random():
    # Intrinsically 2D data; an identity-like embedding (Z == X) spans the exact
    # same Laplacian subspace -> Grassmann distance ~ 0.
    X, _ = blobs_2d(n=2000, centers=8, seed=1)
    Z_id = X.copy()
    gs_id = ev.grassmann_score(X, Z_id, t=10, sample=2000, k=15, seed=0)
    assert gs_id is not None
    assert gs_id["grassmann_distance"] < 1e-6
    assert gs_id["grassmann_affinity_error"] < 1e-8

    # A random 2D scatter destroys the global/spectral structure -> clearly > 0.
    rng = np.random.RandomState(3)
    Z_rand = rng.randn(len(X), 2).astype(np.float32)
    gs_rand = ev.grassmann_score(X, Z_rand, t=10, sample=2000, k=15, seed=0)
    assert gs_rand is not None
    assert gs_rand["grassmann_distance"] > 0.5
    assert gs_rand["grassmann_affinity_error"] > 0.1
    # and the identity map is unambiguously better on both scalars
    assert gs_id["grassmann_distance"] < gs_rand["grassmann_distance"]
    assert gs_id["grassmann_affinity_error"] < gs_rand["grassmann_affinity_error"]


def test_grassmann_bounds_and_keys():
    X, _ = blobs_2d(n=1500, centers=6, seed=5)
    gs = ev.grassmann_score(X, X.copy(), t=8, sample=1500, k=15, seed=0)
    assert set(gs) >= {"grassmann_distance", "grassmann_affinity_error", "t", "sample", "k"}
    assert gs["t"] == 8 and gs["sample"] == 1500
    # affinity error lives in [0, 1]; distance is non-negative
    assert 0.0 <= gs["grassmann_affinity_error"] <= 1.0
    assert gs["grassmann_distance"] >= 0.0


# ─── Collapse index ──────────────────────────────────────────────────────────


def test_collapse_index_separates_healthy_from_collapsed():
    # Healthy: one broad, roughly uniform gaussian blob filling the canvas.
    rng = np.random.RandomState(7)
    Z_healthy = rng.randn(4000, 2).astype(np.float32)
    healthy = ev.collapse_index(Z_healthy, grid=64)

    # Collapsed: 50 pinprick clumps scattered across a 1000x extent.
    Z_collapsed = collapsed_layout(n=4000, n_clumps=50, extent=1000.0,
                                   clump_std=0.02, seed=7)
    collapsed = ev.collapse_index(Z_collapsed, grid=64)

    # (a) NN ratio: collapsed points overlap relative to the canvas -> tiny.
    assert collapsed["collapse_nn_ratio"] < healthy["collapse_nn_ratio"]
    assert collapsed["collapse_nn_ratio"] < 1e-4
    assert healthy["collapse_nn_ratio"] > 1e-3

    # (b) occupancy entropy: collapsed fills few cells -> low; healthy -> high.
    assert collapsed["occupancy_entropy"] < healthy["occupancy_entropy"]
    assert collapsed["occupancy_entropy"] < 0.5
    assert healthy["occupancy_entropy"] > 0.7
    # far fewer occupied cells in the collapsed layout
    assert collapsed["n_occupied_cells"] < healthy["n_occupied_cells"]


def test_collapse_index_uniform_is_high_entropy():
    rng = np.random.RandomState(9)
    Z = rng.uniform(0, 1, size=(5000, 2)).astype(np.float32)
    ci = ev.collapse_index(Z, grid=32)
    # A truly uniform spread should approach entropy 1.0 and occupy most cells.
    assert ci["occupancy_entropy"] > 0.9
    assert ci["n_occupied_cells"] > 0.8 * 32 * 32


# ─── panel wiring ────────────────────────────────────────────────────────────


def test_score_map_reports_new_metrics(tmp_path):
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=1500, n_features=20, centers=6,
                      cluster_std=1.0, random_state=11)
    X = X.astype(np.float32)
    Z = X[:, :2].astype(np.float32)
    cfg = ev.PanelConfig(n_anchors=200, tc_subsample=300, n_pairs=3000,
                         n_triplets=3000, grassmann_sample=1500,
                         cache_dir=str(tmp_path))
    metrics, _, _ = ev.score_map(X, Z, config=cfg)
    assert "grassmann" in metrics and metrics["grassmann"] is not None
    assert "grassmann_distance" in metrics["grassmann"]
    assert "collapse" in metrics
    assert "collapse_nn_ratio" in metrics["collapse"]
    assert "occupancy_entropy" in metrics["collapse"]
