"""Tests for the GPU-resident fast input path (edge_list_dataset.DeviceEdgeSampler
+ ParametricUMAP gpu_resident_data flag).

All tests run on CPU: the fast path degrades gracefully to CPU tensors (fp32
storage) when forced with gpu_resident_data=True, so we can validate its
semantics without a GPU. The default "auto" mode stays legacy on CPU, which is
covered by the existing edge-list smoke tests.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.test_edgelist_smoke import (  # noqa: E402
    make_blobs_data, build_knn_edges, silhouette_2d,
)


def _write_edges(tmp_path, X, k=10):
    sources, targets, weights, n_nodes = build_knn_edges(X, k=k)
    edges_path = tmp_path / "edges.npz"
    np.savez_compressed(edges_path, sources=sources, targets=targets,
                        weights=weights, n_nodes=n_nodes, k=k)
    return str(edges_path)


# ── DeviceEdgeSampler unit behaviour ──────────────────────────────────────────

def test_device_sampler_negatives_never_self_pair():
    import torch
    from basemap.pumap.parametric_umap.datasets.edge_list_dataset import (
        DeviceArrayDataset, DeviceEdgeSampler,
    )
    X, _ = make_blobs_data(n=500, dim=16, centers=5, seed=0)
    dd = DeviceArrayDataset(X, "cpu")
    src = np.repeat(np.arange(500, dtype=np.int64), 4)
    dst = np.random.RandomState(0).randint(0, 500, size=len(src)).astype(np.int64)
    s = DeviceEdgeSampler(dd, src, dst, None, n_nodes=500, pos_ratio=0.25,
                          batch_size=128, random_state=1,
                          positive_target_mode="binary", device="cpu")
    for _ in range(50):
        neg_src, neg_dst = s._sample_negatives(1000)
        assert torch.all(neg_src != neg_dst), "self-pair emitted as negative"
        assert int(neg_src.min()) >= 0 and int(neg_src.max()) < 500
        assert int(neg_dst.min()) >= 0 and int(neg_dst.max()) < 500


def test_device_sampler_batch_shapes_and_labels():
    import torch
    from basemap.pumap.parametric_umap.datasets.edge_list_dataset import (
        DeviceArrayDataset, DeviceEdgeSampler,
    )
    X, _ = make_blobs_data(n=400, dim=8, centers=4, seed=1)
    dd = DeviceArrayDataset(X, "cpu")
    src = np.repeat(np.arange(400, dtype=np.int64), 5)
    dst = np.random.RandomState(2).randint(0, 400, size=len(src)).astype(np.int64)
    bs, pr = 100, 0.2
    s = DeviceEdgeSampler(dd, src, dst, None, n_nodes=400, pos_ratio=pr,
                          batch_size=bs, random_state=3,
                          positive_target_mode="binary", device="cpu")
    num_pos = max(1, int(bs * pr))
    src_feats, dst_feats, targets = next(iter(s))
    assert src_feats.shape == (bs, 8) and dst_feats.shape == (bs, 8)
    assert src_feats.dtype == torch.float32  # gathered as fp32
    # First num_pos are positives (label 1), the rest negatives (label 0).
    assert int((targets == 1.0).sum()) == num_pos
    assert int((targets == 0.0).sum()) == bs - num_pos
    assert len(s) == int(np.ceil(len(src) / num_pos))


def test_device_sampler_explicit_uniform_replacement_is_threshold_independent(monkeypatch):
    import torch
    from basemap.pumap.parametric_umap.datasets.edge_list_dataset import (
        DeviceArrayDataset, DeviceEdgeSampler,
    )
    monkeypatch.setenv("PER_BATCH_EDGE_THRESHOLD", "999999999")
    X = np.arange(80, dtype=np.float32).reshape(10, 8)
    dd = DeviceArrayDataset(X, "cpu")
    src = np.arange(10, dtype=np.int32)
    dst = np.roll(src, -1)
    sampler = DeviceEdgeSampler(
        dd, src, dst, None, n_nodes=10, pos_ratio=0.25,
        batch_size=8, random_state=42, positive_target_mode="binary",
        uniform_with_replacement=True, device="cpu")
    src_feats, dst_feats, labels = next(iter(sampler))
    assert sampler._per_batch is True
    assert sampler.perm is None
    assert src_feats.shape == dst_feats.shape == (8, 8)
    assert torch.equal(labels, torch.tensor([1.0, 1.0, 0.0, 0.0,
                                             0.0, 0.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="cannot be combined"):
        DeviceEdgeSampler(
            dd, src, dst, np.ones(10, dtype=np.float32), n_nodes=10,
            weighted_edge_sampling=True, uniform_with_replacement=True,
            device="cpu")


def test_decide_gpu_resident_modes():
    from basemap.pumap.parametric_umap import ParametricUMAP
    # auto on CPU -> legacy
    p = ParametricUMAP(device="cpu", gpu_resident_data="auto")
    use, _ = p._decide_gpu_resident(1000, 32, 5000, None, False)
    assert use is False
    # forced True -> fast even on CPU
    p = ParametricUMAP(device="cpu", gpu_resident_data=True)
    use, _ = p._decide_gpu_resident(1000, 32, 5000, None, False)
    assert use is True
    # forced but edge_set present (reject_neighbors) -> legacy
    use, _ = p._decide_gpu_resident(1000, 32, 5000, {1, 2}, False)
    assert use is False
    # explicit false -> legacy
    p = ParametricUMAP(device="cpu", gpu_resident_data="false")
    use, _ = p._decide_gpu_resident(1000, 32, 5000, None, False)
    assert use is False


# ── End-to-end: forced fast path trains + separates blobs on CPU ──────────────

def _fit_blobs(gpu_resident, midnear, tmp_path, seed=0):
    from basemap.pumap.parametric_umap import ParametricUMAP
    X, labels = make_blobs_data(n=3000, dim=32, centers=8, seed=seed)
    edges_path = _write_edges(tmp_path, X, k=10)
    pumap = ParametricUMAP(
        architecture="residual_bottleneck", hidden_dim=128, n_layers=3,
        n_components=2, a=1.0, b=1.0, correlation_weight=0.0,
        learning_rate=0.01, n_epochs=15, batch_size=512, pos_ratio=0.2,
        device="cpu", use_amp=False, positive_target_mode="binary",
        lr_schedule="cosine", warmup_steps=30, total_steps_estimate=3000,
        midnear_enabled=midnear, mn_pairs_per_batch=0, mn_weight_scale=1.0,
        gpu_resident_data=("true" if gpu_resident else "false"),
    )
    pumap.fit(X, precomputed_edges_path=edges_path, random_state=42,
              verbose=False)
    assert pumap._fast_device_path is bool(gpu_resident)
    Z = pumap.transform(X)
    assert Z.shape == (3000, 2)
    assert np.isfinite(Z).all()
    return silhouette_2d(Z, labels)


def test_fast_path_separates_blobs_cpu(tmp_path):
    sil = _fit_blobs(gpu_resident=True, midnear=False, tmp_path=tmp_path)
    assert sil > 0.2, f"fast-path 2D silhouette {sil:.3f} not above 0.2"


def test_fast_path_midnear_separates_blobs_cpu(tmp_path):
    sil = _fit_blobs(gpu_resident=True, midnear=True, tmp_path=tmp_path)
    assert sil > 0.2, f"fast-path+midnear 2D silhouette {sil:.3f} not above 0.2"


def test_fast_path_probability_targets_runs(tmp_path):
    """Fast path supports probability (weighted) targets end-to-end."""
    from basemap.pumap.parametric_umap import ParametricUMAP
    X, labels = make_blobs_data(n=2000, dim=16, centers=6, seed=2)
    edges_path = _write_edges(tmp_path, X, k=8)
    pumap = ParametricUMAP(
        architecture="mlp", hidden_dim=64, n_layers=2, n_components=2,
        a=1.0, b=1.0, correlation_weight=0.0, learning_rate=0.01,
        n_epochs=12, batch_size=256, pos_ratio=0.2, device="cpu",
        use_amp=False, positive_target_mode="probability",
        lr_schedule="cosine", warmup_steps=20, total_steps_estimate=2000,
        gpu_resident_data=True,
    )
    pumap.fit(X, precomputed_edges_path=edges_path, random_state=7,
              verbose=False)
    assert pumap._fast_device_path is True
    Z = pumap.transform(X)
    assert np.isfinite(Z).all()
