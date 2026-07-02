"""
Tests for the Phase 1 objective-experiment variants (plan-basemap-atlas.md
§4.2 anchored initialization, §6 Phase 1 mid-near pair loss).

Both variants compose with the precomputed edge-list training path + binary
targets + run persistence. Everything runs on CPU against synthetic
gaussian-blobs, reusing the helpers from test_edgelist_smoke.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.test_edgelist_smoke import (  # noqa: E402
    make_blobs_data, build_knn_edges, write_blobs_dataset, silhouette_2d,
)


# ─── (a) Anchored init: pretrain output correlates with PCA targets ──────────

def test_anchored_pretrain_matches_pca_targets():
    """After the ~PCA pretrain phase (before any UMAP loss), the encoder's
    projection should match the deterministic PCA-2D targets up to a rigid
    transform: Procrustes disparity < 0.5."""
    import torch
    from scipy.spatial import procrustes
    from basemap.pumap.parametric_umap import ParametricUMAP
    from basemap.pumap.parametric_umap.datasets.covariates_datasets import (
        VariableDataset,
    )

    torch.manual_seed(0)
    X, labels = make_blobs_data(n=4000, dim=32, centers=8, seed=0)

    pumap = ParametricUMAP(
        architecture="residual_bottleneck", hidden_dim=128, n_layers=3,
        n_components=2, device="cpu", use_amp=False, batch_size=256,
        clip_grad_norm=1.0, anchored_init="pca", anchored_init_epochs=40,
        anchored_init_lr=1e-3,
    )
    pumap._init_model(X.shape[1])
    n_train = X.shape[0]

    targets = pumap._compute_pca_anchor_targets(X, n_train, random_state=42)
    assert targets.shape == (n_train, 2)
    assert np.isfinite(targets).all()
    # RMS radius scaled to ~5.
    rms = float(np.sqrt(np.mean(np.sum(targets ** 2, axis=1))))
    assert 4.0 < rms < 6.0, f"anchor RMS radius {rms:.2f} not ~5"

    dataset = VariableDataset(X).to("cpu")
    pumap._anchored_pretrain(dataset, targets, n_train, random_state=42)

    # Projection straight out of the pretrain phase (mark fitted so transform
    # is allowed; no UMAP-loss training has happened yet).
    pumap.is_fitted = True
    Z = pumap.transform(X)
    assert np.isfinite(Z).all()

    _, _, disparity = procrustes(targets, Z)
    assert disparity < 0.5, (
        f"post-pretrain projection Procrustes disparity {disparity:.3f} "
        "not below 0.5 (encoder did not learn the PCA targets)"
    )


def test_anchor_targets_are_deterministic():
    """The PCA anchor targets are deterministic across calls (fixed sign
    convention + fixed subsample seed)."""
    from basemap.pumap.parametric_umap import ParametricUMAP

    X, _ = make_blobs_data(n=3000, dim=32, centers=6, seed=1)
    pumap = ParametricUMAP(n_components=2, device="cpu")
    t1 = pumap._compute_pca_anchor_targets(X, len(X), random_state=42)
    t2 = pumap._compute_pca_anchor_targets(X, len(X), random_state=42)
    np.testing.assert_allclose(t1, t2, rtol=0, atol=0)


# ─── (b) Mid-near training runs (no NaN) and still separates blobs ───────────

def test_midnear_fit_separates_blobs_no_nan():
    from basemap.pumap.parametric_umap import ParametricUMAP

    X, labels = make_blobs_data(n=4000, dim=32, centers=8, seed=0)
    sources, targets, weights, n_nodes = build_knn_edges(X, k=10)
    edges = {"sources": sources, "targets": targets, "weights": weights,
             "n_nodes": n_nodes, "k": 10}
    import tempfile, os
    tmp = tempfile.mkdtemp()
    edges_path = os.path.join(tmp, "edges.npz")
    np.savez_compressed(edges_path, **edges)

    pumap = ParametricUMAP(
        architecture="residual_bottleneck", hidden_dim=128, n_layers=3,
        n_components=2, a=1.0, b=1.0, correlation_weight=0.0,
        learning_rate=0.01, n_epochs=12, batch_size=1024, pos_ratio=0.2,
        device="cpu", use_amp=False, positive_target_mode="binary",
        lr_schedule="cosine", warmup_steps=30, total_steps_estimate=5000,
        midnear_enabled=True, mn_pairs_per_batch=0, mn_weight_scale=1.0,
    )
    pumap.fit(X, precomputed_edges_path=edges_path, random_state=42,
              verbose=False)

    Z = pumap.transform(X)
    assert Z.shape == (X.shape[0], 2)
    assert np.isfinite(Z).all(), "mid-near projection contains NaN/inf"
    sil = silhouette_2d(Z, labels)
    assert sil > 0.2, f"mid-near 2D silhouette {sil:.3f} not above 0.2"


# ─── (c) Both variants compose with binary targets + persistence e2e ─────────

def _run_variant_config(cfg_name, tmp_path):
    from experiments.experiment_config import load_config
    from experiments.run_experiment import run_single_experiment

    cfg_path = REPO_ROOT / "experiments" / "configs" / cfg_name
    # Keep the CPU smoke fast: fewer rows/edges (blobs are trivially separable).
    memmap_dir, edges_path, labels = write_blobs_dataset(tmp_path, n=3000)
    cfg = load_config(str(cfg_path))
    cfg.data.memmap_dirs = [memmap_dir]
    cfg.data.precomputed_edges_path = edges_path
    cfg.logging.results_dir = str(tmp_path / "results")
    results = run_single_experiment(cfg)

    run_dirs = sorted(Path(cfg.logging.results_dir).glob(f"{cfg.name}_*"))
    assert run_dirs, "no run directory created"
    return cfg, results, run_dirs[-1], labels


def test_anchored_end_to_end_persists_targets(tmp_path):
    import pyarrow.parquet as pq

    cfg, results, run_dir, labels = _run_variant_config(
        "test_anchored_smoke.yaml", tmp_path)

    # Composes with binary targets.
    assert cfg.train.positive_target_mode == "binary"
    assert cfg.train.anchored_init == "pca"

    # Persistence: coords + model + anchor targets.
    coords_path = run_dir / "coords.parquet"
    model_path = run_dir / "model.pt"
    anchor_path = run_dir / "anchor_targets.parquet"
    assert coords_path.exists(), "coords.parquet missing"
    assert model_path.exists(), "model.pt missing"
    assert anchor_path.exists(), "anchor_targets.parquet missing"

    # Manifest records the new options.
    eff = results["run_manifest"]["effective_config"]["train"]
    assert eff["anchored_init"] == "pca"
    assert "anchored_init_epochs" in eff

    # Anchor targets schema + finiteness.
    atbl = pq.read_table(anchor_path).to_pandas()
    assert set(atbl.columns) == {"x", "y", "ls_index"}
    assert np.isfinite(atbl[["x", "y"]].to_numpy()).all()

    # Blobs still separate.
    ctbl = pq.read_table(coords_path).to_pandas()
    Z = ctbl[["x", "y"]].to_numpy()
    assert np.isfinite(Z).all()
    sil = silhouette_2d(Z, labels[ctbl["ls_index"].to_numpy()])
    assert sil > 0.2, f"anchored e2e silhouette {sil:.3f} not above 0.2"


def test_midnear_end_to_end(tmp_path):
    import pyarrow.parquet as pq

    cfg, results, run_dir, labels = _run_variant_config(
        "test_midnear_smoke.yaml", tmp_path)

    assert cfg.train.positive_target_mode == "binary"
    assert cfg.train.midnear_enabled is True

    coords_path = run_dir / "coords.parquet"
    model_path = run_dir / "model.pt"
    assert coords_path.exists() and model_path.exists()
    # No anchor targets when anchored_init is off.
    assert not (run_dir / "anchor_targets.parquet").exists()

    eff = results["run_manifest"]["effective_config"]["train"]
    assert eff["midnear_enabled"] is True
    assert "mn_weight_scale" in eff

    ctbl = pq.read_table(coords_path).to_pandas()
    Z = ctbl[["x", "y"]].to_numpy()
    assert np.isfinite(Z).all()
    sil = silhouette_2d(Z, labels[ctbl["ls_index"].to_numpy()])
    assert sil > 0.2, f"mid-near e2e silhouette {sil:.3f} not above 0.2"
