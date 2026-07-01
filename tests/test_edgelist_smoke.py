"""
Smoke tests for the precomputed edge-list training path (Work Package 2).

Exercises:
  * ParametricUMAP.fit(precomputed_edges_path=...) directly, and
  * the full experiments.run_experiment end-to-end path,

on a small synthetic gaussian-blobs dataset with an exact-kNN edge .npz built
with sklearn. Everything runs on CPU.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ─── Shared synthetic-data helpers (reused by test_persistence.py) ────────────

def make_blobs_data(n=5000, dim=32, centers=8, seed=0):
    """Well-separated gaussian blobs in `dim` dimensions."""
    from sklearn.datasets import make_blobs
    X, labels = make_blobs(
        n_samples=n, n_features=dim, centers=centers,
        cluster_std=1.0, random_state=seed,
    )
    return X.astype(np.float32), labels.astype(np.int64)


def build_knn_edges(X, k=10):
    """Exact-kNN directed edge list matching the build_*_index_modal.py npz
    schema: int32 sources/targets, float32 weights, scalar n_nodes/k."""
    from sklearn.neighbors import NearestNeighbors
    n = len(X)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)
    idx = idx[:, 1:]  # drop self
    sources = np.repeat(np.arange(n, dtype=np.int32), k)
    targets = idx.reshape(-1).astype(np.int32)
    weights = np.full(len(sources), 1.0 / k, dtype=np.float32)
    return sources, targets, weights, n


def write_blobs_dataset(root: Path, n=5000, dim=32, centers=8, k=10, seed=0):
    """Materialise a memmap-style .npy shard + edge .npz under `root`.

    Returns (memmap_dir, edges_path, labels).
    """
    root = Path(root)
    memmap_dir = root / "blobs_32d"
    memmap_dir.mkdir(parents=True, exist_ok=True)
    X, labels = make_blobs_data(n=n, dim=dim, centers=centers, seed=seed)
    np.save(memmap_dir / "data-00000-of-00001.npy", X)

    sources, targets, weights, n_nodes = build_knn_edges(X, k=k)
    edges_path = root / "blobs_edges_k10.npz"
    np.savez_compressed(
        edges_path, sources=sources, targets=targets, weights=weights,
        n_nodes=n_nodes, k=k,
    )
    np.save(root / "labels.npy", labels)
    return str(memmap_dir), str(edges_path), labels


def silhouette_2d(Z, labels):
    from sklearn.metrics import silhouette_score
    return float(silhouette_score(np.asarray(Z), np.asarray(labels)))


# ─── Direct fit() edge-list path ──────────────────────────────────────────────

def test_fit_edge_list_path_separates_blobs(tmp_path):
    from basemap.pumap.parametric_umap import ParametricUMAP

    X, labels = make_blobs_data(n=5000, dim=32, centers=8, seed=0)
    sources, targets, weights, n_nodes = build_knn_edges(X, k=10)
    edges_path = tmp_path / "edges.npz"
    np.savez_compressed(edges_path, sources=sources, targets=targets,
                        weights=weights, n_nodes=n_nodes, k=10)

    pumap = ParametricUMAP(
        architecture="residual_bottleneck",
        hidden_dim=128, n_layers=3, n_components=2,
        a=1.0, b=1.0, correlation_weight=0.0,
        learning_rate=0.01, n_epochs=30, batch_size=512,
        pos_ratio=0.2, device="cpu", use_amp=False,
        positive_target_mode="binary",
        lr_schedule="cosine", warmup_steps=50, total_steps_estimate=3000,
    )
    pumap.fit(X, precomputed_edges_path=str(edges_path),
              random_state=42, verbose=False)

    Z = pumap.transform(X)
    assert Z.shape == (5000, 2)
    assert np.isfinite(Z).all(), "projection contains NaN/inf"

    sil = silhouette_2d(Z, labels)
    assert sil > 0.2, f"2D silhouette {sil:.3f} not above random floor (0.2)"


def test_edge_list_iterator_rejects_self_and_neighbors():
    """The on-the-fly negative sampler must never emit self-pairs, and with
    reject_neighbors it must never emit a positive edge as a negative."""
    from basemap.pumap.parametric_umap.datasets.edge_list_dataset import (
        EdgeListBalancedIterator, build_edge_key_set,
    )
    n_nodes = 200
    rng = np.random.RandomState(1)
    sources = rng.randint(0, n_nodes, size=1000).astype(np.int32)
    targets = rng.randint(0, n_nodes, size=1000).astype(np.int32)
    weights = np.full(1000, 0.1, dtype=np.float32)
    edge_set = build_edge_key_set(sources, targets, n_nodes)

    it = EdgeListBalancedIterator(
        sources, targets, weights, n_nodes=n_nodes,
        pos_ratio=0.2, batch_size=256, shuffle=True, random_state=7,
        positive_target_mode="binary", edge_set=edge_set,
    )
    n_batches = 0
    for edge_batch, labels in it:
        # negatives are the entries with label 0.0
        for (s, d), lab in zip(edge_batch, labels):
            if lab == 0.0:
                assert s != d, "self-pair emitted as negative"
                key = int(s) * n_nodes + int(d)
                assert key not in edge_set, "positive edge emitted as negative"
        n_batches += 1
    assert n_batches == len(it)


# ─── End-to-end run_experiment path ──────────────────────────────────────────

def _load_smoke_config(tmp_path):
    from experiments.experiment_config import load_config
    cfg_path = REPO_ROOT / "experiments" / "configs" / "test_edgelist_smoke.yaml"
    memmap_dir, edges_path, labels = write_blobs_dataset(tmp_path)
    cfg = load_config(str(cfg_path))
    cfg.data.memmap_dirs = [memmap_dir]
    cfg.data.precomputed_edges_path = edges_path
    cfg.logging.results_dir = str(tmp_path / "results")
    return cfg, labels


def test_run_experiment_end_to_end(tmp_path):
    from experiments.run_experiment import run_single_experiment

    cfg, labels = _load_smoke_config(tmp_path)
    results = run_single_experiment(cfg)

    # Locate the run dir.
    results_root = Path(cfg.logging.results_dir)
    run_dirs = sorted(results_root.glob(f"{cfg.name}_*"))
    assert run_dirs, "no run directory created"
    run_dir = run_dirs[-1]

    coords_path = run_dir / "coords.parquet"
    model_path = run_dir / "model.pt"
    assert coords_path.exists(), "coords.parquet was not persisted"
    assert model_path.exists(), "model.pt was not persisted"

    # Training produced finite metrics.
    metrics = results["metrics_train"]
    assert all(np.isfinite(v) for v in metrics.values())

    # Coordinates separate the blobs. ls_index maps rows back to labels.
    import pyarrow.parquet as pq
    tbl = pq.read_table(coords_path)
    df = tbl.to_pandas()
    assert set(df.columns) == {"x", "y", "ls_index"}
    Z = df[["x", "y"]].to_numpy()
    assert np.isfinite(Z).all()
    sil = silhouette_2d(Z, labels[df["ls_index"].to_numpy()])
    assert sil > 0.2, f"end-to-end 2D silhouette {sil:.3f} not above 0.2"
