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


# ─── Reference-atlas distillation (plan §4.3): file targets + anchor hold ─────

def _write_teacher_parquet(path, X, seed=0, shuffle=False, with_ls_index=True):
    """Write a learnable teacher layout (PCA-2D of X) as a coords parquet.

    Returns the teacher coordinates in *original X row order* so callers can
    Procrustes-compare a projection (also in row order) against them. When
    ``shuffle`` is set the parquet rows are permuted, so alignment can only be
    recovered via the ``ls_index`` column — exercising the by-ls_index path.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from sklearn.decomposition import PCA

    coords = PCA(n_components=2, random_state=seed).fit_transform(
        X.astype(np.float64)).astype(np.float32)
    n = len(coords)
    ls = np.arange(n, dtype=np.int64)
    if shuffle:
        perm = np.random.RandomState(seed).permutation(n)
        coords_w, ls_w = coords[perm], ls[perm]
    else:
        coords_w, ls_w = coords, ls
    cols = {"x": pa.array(coords_w[:, 0], type=pa.float32()),
            "y": pa.array(coords_w[:, 1], type=pa.float32())}
    if with_ls_index:
        cols["ls_index"] = pa.array(ls_w, type=pa.int64())
    pq.write_table(pa.table(cols), str(path))
    return coords


def test_file_pretrain_reproduces_teacher_layout(tmp_path):
    """(a) After the ~file pretrain phase (before any UMAP loss), the encoder's
    projection reproduces the teacher layout up to a rigid transform: Procrustes
    disparity < 0.1. The teacher parquet rows are shuffled, so this also checks
    ls_index alignment."""
    from scipy.spatial import procrustes
    from basemap.pumap.parametric_umap import ParametricUMAP
    from basemap.pumap.parametric_umap.datasets.covariates_datasets import (
        VariableDataset,
    )

    X, _ = make_blobs_data(n=2500, dim=32, centers=8, seed=0)
    teacher_path = tmp_path / "teacher.parquet"
    coords = _write_teacher_parquet(teacher_path, X, seed=0, shuffle=True,
                                    with_ls_index=True)

    pumap = ParametricUMAP(
        architecture="residual_bottleneck", hidden_dim=128, n_layers=3,
        n_components=2, device="cpu", use_amp=False, batch_size=256,
        clip_grad_norm=1.0, anchored_init="file",
        anchored_init_path=str(teacher_path),
        anchored_init_epochs=200, anchored_init_lr=1e-3,
    )
    pumap._init_model(X.shape[1])
    n_train = X.shape[0]

    targets = pumap._compute_anchor_targets(X, n_train, random_state=42)
    assert targets.shape == (n_train, 2)
    assert np.isfinite(targets).all()
    # Scaled to RMS radius ~5 (same convention as the PCA path).
    rms = float(np.sqrt(np.mean(np.sum(targets ** 2, axis=1))))
    assert 4.0 < rms < 6.0, f"file-target RMS radius {rms:.2f} not ~5"
    assert pumap.anchor_scale_ is not None and pumap.anchor_scale_ > 0
    # Targets align to the original X row order despite the shuffled parquet.
    _, _, teacher_disp = procrustes(coords, targets)
    assert teacher_disp < 1e-6, (
        f"scaled targets not a rigid transform of the teacher (disparity "
        f"{teacher_disp:.2e}) — ls_index alignment broken")

    dataset = VariableDataset(X).to("cpu")
    pumap._anchored_pretrain(dataset, targets, n_train, random_state=42)

    pumap.is_fitted = True
    Z = pumap.transform(X)
    assert np.isfinite(Z).all()

    _, _, disparity = procrustes(targets, Z)
    assert disparity < 0.1, (
        f"post-pretrain projection Procrustes disparity {disparity:.3f} "
        "not below 0.1 (encoder did not reproduce the teacher layout)")


def test_anchor_hold_keeps_layout_near_targets_with_midnear(tmp_path):
    """(b) With a high anchor_hold_weight the final layout stays near the teacher
    targets (Procrustes disparity < 0.2), and training does not NaN with mid-near
    enabled."""
    from scipy.spatial import procrustes
    from basemap.pumap.parametric_umap import ParametricUMAP

    X, labels = make_blobs_data(n=3000, dim=32, centers=8, seed=0)
    sources, targets_e, weights, n_nodes = build_knn_edges(X, k=10)
    edges_path = tmp_path / "edges.npz"
    np.savez_compressed(edges_path, sources=sources, targets=targets_e,
                        weights=weights, n_nodes=n_nodes, k=10)
    teacher_path = tmp_path / "teacher.parquet"
    coords = _write_teacher_parquet(teacher_path, X, seed=0, shuffle=False,
                                    with_ls_index=True)

    pumap = ParametricUMAP(
        architecture="residual_bottleneck", hidden_dim=128, n_layers=3,
        n_components=2, a=1.0, b=1.0, correlation_weight=0.0,
        learning_rate=0.01, n_epochs=20, batch_size=512, pos_ratio=0.2,
        device="cpu", use_amp=False, positive_target_mode="binary",
        lr_schedule="cosine", warmup_steps=30, total_steps_estimate=6000,
        anchored_init="file", anchored_init_path=str(teacher_path),
        anchored_init_epochs=0,  # hold-only: no pretrain, pure ongoing distill
        anchor_hold_weight=30.0, anchor_hold_fraction=0.1,
        midnear_enabled=True, mn_pairs_per_batch=0, mn_weight_scale=1.0,
    )
    pumap.fit(X, precomputed_edges_path=str(edges_path), random_state=42,
              verbose=False)

    Z = pumap.transform(X)
    assert np.isfinite(Z).all(), "anchor-hold + midnear projection has NaN/inf"
    _, _, disparity = procrustes(coords, Z)
    assert disparity < 0.2, (
        f"anchor-hold final layout Procrustes disparity {disparity:.3f} "
        "not below 0.2 (high hold weight did not pin the layout to the teacher)")


def test_fast_path_anchor_hold_midnear_cpu(tmp_path):
    """(c) The GPU-resident fast path (forced True on CPU) composes with file
    anchored-init, anchor-hold, and mid-near, and still separates the blobs."""
    from basemap.pumap.parametric_umap import ParametricUMAP

    X, labels = make_blobs_data(n=3000, dim=32, centers=8, seed=0)
    sources, targets_e, weights, n_nodes = build_knn_edges(X, k=10)
    edges_path = tmp_path / "edges.npz"
    np.savez_compressed(edges_path, sources=sources, targets=targets_e,
                        weights=weights, n_nodes=n_nodes, k=10)
    teacher_path = tmp_path / "teacher.parquet"
    _write_teacher_parquet(teacher_path, X, seed=0, shuffle=False,
                           with_ls_index=True)

    pumap = ParametricUMAP(
        architecture="residual_bottleneck", hidden_dim=128, n_layers=3,
        n_components=2, a=1.0, b=1.0, correlation_weight=0.0,
        learning_rate=0.01, n_epochs=15, batch_size=512, pos_ratio=0.2,
        device="cpu", use_amp=False, positive_target_mode="binary",
        lr_schedule="cosine", warmup_steps=30, total_steps_estimate=3000,
        anchored_init="file", anchored_init_path=str(teacher_path),
        anchored_init_epochs=3, anchored_init_lr=1e-3,
        anchor_hold_weight=1.0, anchor_hold_fraction=0.1,
        midnear_enabled=True, mn_pairs_per_batch=0, mn_weight_scale=1.0,
        gpu_resident_data="true",
    )
    pumap.fit(X, precomputed_edges_path=str(edges_path), random_state=42,
              verbose=False)
    assert pumap._fast_device_path is True, "fast path did not engage"
    assert pumap._anchor_targets_dev is not None
    Z = pumap.transform(X)
    assert Z.shape == (3000, 2)
    assert np.isfinite(Z).all(), "fast-path distillation projection has NaN/inf"
    sil = silhouette_2d(Z, labels)
    assert sil > 0.2, f"fast-path distillation 2D silhouette {sil:.3f} not > 0.2"


def test_hold_without_target_source_raises(tmp_path):
    """anchor_hold_weight>0 with anchored_init='none' is a config error."""
    from basemap.pumap.parametric_umap import ParametricUMAP

    X, _ = make_blobs_data(n=800, dim=16, centers=4, seed=0)
    sources, targets_e, weights, n_nodes = build_knn_edges(X, k=6)
    edges_path = tmp_path / "edges.npz"
    np.savez_compressed(edges_path, sources=sources, targets=targets_e,
                        weights=weights, n_nodes=n_nodes, k=6)
    pumap = ParametricUMAP(
        n_components=2, device="cpu", use_amp=False, n_epochs=1,
        batch_size=256, positive_target_mode="binary", lr_schedule="cosine",
        total_steps_estimate=100, anchored_init="none", anchor_hold_weight=5.0,
    )
    with pytest.raises(ValueError, match="anchor_hold_weight"):
        pumap.fit(X, precomputed_edges_path=str(edges_path), random_state=0,
                  verbose=False)


def test_anchored_file_end_to_end_manifest_scale(tmp_path):
    """End-to-end run_experiment with the file-mode + hold config: persists
    coords + model + anchor targets, and records the RMS scale factor in the
    manifest."""
    import pyarrow.parquet as pq
    from experiments.experiment_config import load_config
    from experiments.run_experiment import run_single_experiment

    memmap_dir, edges_path, labels = write_blobs_dataset(tmp_path, n=3000)
    # Teacher aligned to the blobs the runner will load (same seed/params).
    X, _ = make_blobs_data(n=3000, dim=32, centers=8, seed=0)
    teacher_path = tmp_path / "teacher_coords.parquet"
    _write_teacher_parquet(teacher_path, X, seed=0, shuffle=False,
                           with_ls_index=True)

    cfg_path = REPO_ROOT / "experiments" / "configs" / "test_anchored_file_smoke.yaml"
    cfg = load_config(str(cfg_path))
    cfg.data.memmap_dirs = [memmap_dir]
    cfg.data.precomputed_edges_path = edges_path
    cfg.train.anchored_init_path = str(teacher_path)
    cfg.logging.results_dir = str(tmp_path / "results")
    results = run_single_experiment(cfg)

    run_dirs = sorted(Path(cfg.logging.results_dir).glob(f"{cfg.name}_*"))
    assert run_dirs, "no run directory created"
    run_dir = run_dirs[-1]

    assert cfg.train.anchored_init == "file"
    assert cfg.train.anchor_hold_weight > 0

    coords_path = run_dir / "coords.parquet"
    model_path = run_dir / "model.pt"
    anchor_path = run_dir / "anchor_targets.parquet"
    assert coords_path.exists() and model_path.exists()
    assert anchor_path.exists(), "anchor_targets.parquet missing for file mode"

    # Manifest records the new options + the computed scale factor.
    eff = results["run_manifest"]["effective_config"]["train"]
    assert eff["anchored_init"] == "file"
    assert eff["anchor_hold_weight"] == cfg.train.anchor_hold_weight
    assert "anchor_hold_fraction" in eff
    scale = results["run_manifest"]["anchor_scale"]
    assert scale is not None and scale > 0, "anchor_scale not recorded in manifest"

    # Persisted anchor targets are RMS ~5 (scaled teacher).
    atbl = pq.read_table(anchor_path).to_pandas()
    assert set(atbl.columns) == {"x", "y", "ls_index"}
    xy = atbl[["x", "y"]].to_numpy()
    assert np.isfinite(xy).all()
    rms = float(np.sqrt(np.mean((xy ** 2).sum(axis=1))))
    assert 4.0 < rms < 6.0, f"persisted anchor RMS {rms:.2f} not ~5"

    ctbl = pq.read_table(coords_path).to_pandas()
    Z = ctbl[["x", "y"]].to_numpy()
    assert np.isfinite(Z).all()
    sil = silhouette_2d(Z, labels[ctbl["ls_index"].to_numpy()])
    assert sil > 0.2, f"file-mode e2e silhouette {sil:.3f} not above 0.2"
