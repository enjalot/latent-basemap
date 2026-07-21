"""L0.5 — close the directly-demonstrated admission holes.

Covers the owner's REORDER_ACCEPTED finding and the related fail-open gaps:
ordered shard identity (reject reorder + duplicate basenames), a multi-shard
manifest without an order list, the strengthened SHA-cache identity + atomic
sidecar, HostStream degenerate-weight fail-closed, and the invariant that
admission runs BEFORE model allocation.
"""
import sys, os, numpy as np, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from basemap.graph_validation import (graph_manifest_v2, validate_graph_content,
                                       _cached_stream_sha, stream_sha)


def _two_shard_graph(tmp_path, dim=8, n=120):
    X = np.random.RandomState(0).randn(n, dim).astype('float32')
    h = n // 2
    s0 = tmp_path / "data-00000.npy"; np.save(s0, X[:h])
    s1 = tmp_path / "data-00001.npy"; np.save(s1, X[h:])
    s = np.repeat(np.arange(n), 3).astype('int32')
    t = np.random.RandomState(1).randint(0, n, 3 * n).astype('int32')
    gp = str(tmp_path / "edges.npz"); np.savez(gp, sources=s, targets=t, n_nodes=n, k=3)
    man = graph_manifest_v2(s, t, n, X=X, graph_path=gp, data_paths=[str(s0), str(s1)], k=3)
    return gp, str(s0), str(s1), man


def test_l0_ordered_shards_accepted(tmp_path):
    gp, s0, s1, man = _two_shard_graph(tmp_path)
    trusted = validate_graph_content(gp, man, shard_paths=[s0, s1])   # correct order → ok
    assert trusted["data_shard_order"] == ["data-00000.npy", "data-00001.npy"]


def test_l0_reversed_shards_raise(tmp_path):
    # The exact REORDER_ACCEPTED bug: manifest order [a,b], loaded [b,a].
    gp, s0, s1, man = _two_shard_graph(tmp_path)
    with pytest.raises(ValueError, match="order.*!=.*order|reordered"):
        validate_graph_content(gp, man, shard_paths=[s1, s0])


def test_l0_duplicate_basename_shards_raise(tmp_path):
    gp, s0, s1, man = _two_shard_graph(tmp_path)
    da = tmp_path / "a"; db = tmp_path / "b"; da.mkdir(); db.mkdir()
    import shutil
    pa = str(da / "data-00000.npy"); pb = str(db / "data-00000.npy")
    shutil.copy(s0, pa); shutil.copy(s0, pb)
    with pytest.raises(ValueError, match="duplicate shard basenames"):
        validate_graph_content(gp, man, shard_paths=[pa, pb])


def test_l0_multishard_without_order_raises(tmp_path):
    gp, s0, s1, man = _two_shard_graph(tmp_path)
    del man["data_shards"]                       # legacy dict-only, >1 shard
    with pytest.raises(ValueError, match="lacks an ordered"):
        validate_graph_content(gp, man, shard_paths=[s0, s1])


def test_l0_cache_identity_rejects_same_size_mtime_replacement(tmp_path):
    # size+mtime alone is fooled by an in-place same-size replacement with a
    # restored mtime; the strengthened identity (adds ctime/inode/dev) must not
    # return the stale sha.
    p = str(tmp_path / "blob.bin")
    with open(p, "wb") as f:
        f.write(b"A" * 4096)
    st0 = os.stat(p)
    sha_a = _cached_stream_sha(p)
    assert sha_a == stream_sha(p)
    with open(p, "wb") as f:
        f.write(b"B" * 4096)                     # same size, different content
    os.utime(p, ns=(st0.st_atime_ns, st0.st_mtime_ns))   # restore mtime
    sha_b = _cached_stream_sha(p)
    assert sha_b == stream_sha(p) and sha_b != sha_a      # not the stale sha
    assert not any(x.endswith(".tmp") or ".tmp." in x for x in os.listdir(tmp_path))  # atomic


def test_l0_hoststream_degenerate_weights_fail_closed():
    from basemap.pumap.parametric_umap.datasets.edge_list_dataset import (
        HostStreamEdgeSampler, DeviceArrayDataset)
    n = 6
    X = np.random.RandomState(0).randn(n, 4).astype('float32')
    ds = DeviceArrayDataset(X, "cpu")
    s = np.arange(6).astype('int32') % n; t = (s + 1) % n

    def mk(w):
        return HostStreamEdgeSampler(ds, s, t, np.asarray(w, 'float32'), n_nodes=n,
                                     batch_size=4, pos_ratio=0.5, device="cpu",
                                     weighted_edge_sampling=True, n_workers=1)
    with pytest.raises(ValueError, match="non-positive weight total"):
        mk(np.zeros(6))
    with pytest.raises(ValueError, match="negative edge weights"):
        mk([1, 1, -1, 1, 1, 1])
    with pytest.raises(ValueError, match="non-finite"):
        mk([1, 1, np.inf, 1, 1, 1])


@pytest.mark.parametrize("weighted", [False, True])
def test_l0_hoststream_retained_nodes_apply_to_variable_degree_edges(weighted):
    from basemap.pumap.parametric_umap.datasets.edge_list_dataset import (
        HostStreamEdgeSampler, DeviceArrayDataset)
    n = 5
    X = np.random.RandomState(4).randn(n, 3).astype("float32")
    ds = DeviceArrayDataset(X, "cpu")
    sources = np.array([0, 0, 1, 2, 2, 2, 3, 4, 4], dtype="int32")
    targets = np.array([1, 2, 0, 0, 3, 4, 2, 0, 3], dtype="int32")
    weights = np.array([1, 2, 50, 3, 4, 5, 60, 6, 7], dtype="float32")
    retained = np.array([0, 2, 4], dtype="int32")
    sampler = HostStreamEdgeSampler(
        ds, sources, targets, weights, n_nodes=n, batch_size=16,
        pos_ratio=0.5, device="cpu", weighted_edge_sampling=weighted,
        retained_node_rows=retained, n_workers=0)
    assert sampler.n_pos == 7 and sampler.excluded_positive_edges == 2
    draws = sampler._draw_positive_indices(np.random.default_rng(8), 200_000)
    assert set(sources[draws].tolist()) <= set(retained.tolist())
    observed = np.bincount(draws, minlength=len(sources)) / len(draws)
    allowed = np.isin(sources, retained)
    expected = (weights.astype("float64") * allowed if weighted
                else allowed.astype("float64"))
    expected /= expected.sum()
    assert np.allclose(observed, expected, atol=0.006), (observed, expected)
    neg_s, neg_t = sampler._sample_negatives(20_000)
    assert set(neg_s.numpy().tolist()) <= set(retained.tolist())
    assert set(neg_t.numpy().tolist()) <= set(retained.tolist())
    assert np.all(neg_s.numpy() != neg_t.numpy())


def test_l0_hoststream_retained_nodes_require_source_sorted_edges():
    from basemap.pumap.parametric_umap.datasets.edge_list_dataset import (
        HostStreamEdgeSampler, DeviceArrayDataset)
    X = np.random.RandomState(5).randn(4, 3).astype("float32")
    ds = DeviceArrayDataset(X, "cpu")
    with pytest.raises(ValueError, match="source-sorted"):
        HostStreamEdgeSampler(
            ds, np.array([1, 0, 2], dtype="int32"),
            np.array([0, 2, 1], dtype="int32"), np.ones(3, dtype="float32"),
            n_nodes=4, device="cpu", retained_node_rows=np.array([0, 2, 3]),
            n_workers=0)


def test_l0_admission_precedes_model_allocation():
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    m = ParametricUMAP(a=1., b=1., correlation_weight=0.0, n_epochs=1, batch_size=8,
                       total_steps_estimate=8, lr_schedule='cosine', warmup_steps=0,
                       device='cpu', positive_target_mode='binary', use_amp=False,
                       require_full_budget=False)
    m._init_model(8)                              # allocate BEFORE admission → invariant broken
    with pytest.raises(RuntimeError, match="before model allocation|BEFORE model"):
        m._prepare_edge_list_training(None, "x.npz", 10, False, 0)


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
