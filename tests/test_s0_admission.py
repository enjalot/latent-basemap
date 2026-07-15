"""S0 (closure seal): input + sampler admission fails closed.

- graph/shard content: a required manifest with data_shard_sha rejects a missing,
  mutated, extra, or absent shard, and a required manifest without graph_sha raises;
- weighted sampler: large-imbalance weights produce the expected sampled frequency,
  constant weights are uniform-equivalent, and non-finite/negative/non-positive
  totals raise rather than collapse every draw onto one edge.
"""
import sys, os, numpy as np, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from basemap.graph_validation import graph_manifest_v2, validate_graph_content


def _graph_and_shards(tmp_path, dim=8, n=200):
    X = np.random.RandomState(0).randn(n, dim).astype('float32')
    shard = tmp_path / "data-00000.npy"; np.save(shard, X)
    s = np.repeat(np.arange(n), 3).astype('int32'); t = np.random.RandomState(1).randint(0, n, 3 * n).astype('int32')
    gp = str(tmp_path / "edges.npz"); np.savez(gp, sources=s, targets=t, n_nodes=n, k=3)
    man = graph_manifest_v2(s, t, n, X=X, graph_path=gp, data_paths=[str(shard)], k=3)
    return gp, str(shard), man


def test_s0_shard_content_verified(tmp_path):
    gp, shard, man = _graph_and_shards(tmp_path)
    assert "data-00000.npy" in man["data_shard_sha"]
    validate_graph_content(gp, man, shard_paths=[shard])            # matches → ok


def test_s0_missing_shard_paths_raises(tmp_path):
    gp, shard, man = _graph_and_shards(tmp_path)
    with pytest.raises(ValueError, match="NO shard paths|refuse to train"):
        validate_graph_content(gp, man, shard_paths=[])             # loader gave none


def test_s0_mutated_shard_raises(tmp_path):
    gp, shard, man = _graph_and_shards(tmp_path)
    Xb = np.load(shard); Xb[5] += 9.0; np.save(shard, Xb)
    try:
        os.remove(shard + ".shacache.json")
    except OSError:
        pass
    with pytest.raises(ValueError, match="sha .* != manifest|data changed"):
        validate_graph_content(gp, man, shard_paths=[shard])


def test_s0_extra_shard_raises(tmp_path):
    gp, shard, man = _graph_and_shards(tmp_path)
    extra = tmp_path / "data-00001.npy"; np.save(extra, np.zeros((10, 8), 'float32'))
    with pytest.raises(ValueError, match="absent from the manifest|extra"):
        validate_graph_content(gp, man, shard_paths=[shard, str(extra)])


def test_s0_required_manifest_without_graph_sha_raises(tmp_path):
    from basemap.graph_validation import graph_manifest
    X = np.random.RandomState(0).randn(100, 8).astype('float32')
    s = np.arange(100).astype('int32'); t = (s + 1) % 100
    man_v1 = graph_manifest(s, t, 100, X=X)                          # v1: no graph_sha
    gp = str(tmp_path / "e.npz"); np.savez(gp, sources=s, targets=t, n_nodes=100)
    with pytest.raises(ValueError, match="no graph_sha"):
        validate_graph_content(gp, man_v1, require_manifest_sha=True)
    validate_graph_content(gp, man_v1, require_manifest_sha=False)   # test escape hatch


# ── weighted sampler ─────────────────────────────────────────────────────────────

def _sampler(weights, n=6):
    from basemap.pumap.parametric_umap.datasets.edge_list_dataset import DeviceEdgeSampler, DeviceArrayDataset
    X = np.random.RandomState(0).randn(n, 4).astype('float32')
    ds = DeviceArrayDataset(X, "cpu")
    s = np.arange(len(weights)).astype('int32') % n
    t = (s + 1) % n
    return DeviceEdgeSampler(ds, s, t, np.asarray(weights, 'float32'), n_nodes=n,
                             weighted_edge_sampling=True, device="cpu")


def test_s0_weighted_sampler_matches_frequency():
    # a large-imbalance weight vector must be sampled at ~proportional frequency
    w = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 5.0], dtype='float32')
    smp = _sampler(w)
    draws = smp._draw_idx(200_000).cpu().numpy()
    freq = np.bincount(draws, minlength=len(w)) / len(draws)
    expected = w / w.sum()
    assert np.allclose(freq, expected, atol=0.02), (freq, expected)


def test_s0_constant_weights_uniform_equivalent():
    w = np.full(6, 0.3, dtype='float32')
    smp = _sampler(w)
    freq = np.bincount(smp._draw_idx(120_000).cpu().numpy(), minlength=6) / 120_000
    assert np.allclose(freq, np.full(6, 1 / 6), atol=0.02), freq


def test_s0_degenerate_weights_raise():
    with pytest.raises(ValueError, match="non-positive weight total"):
        _sampler(np.zeros(6, 'float32'))
    with pytest.raises(ValueError, match="negative edge weights"):
        _sampler(np.array([1, 1, -1, 1, 1, 1], 'float32'))
    with pytest.raises(ValueError, match="non-finite"):
        _sampler(np.array([1, 1, np.inf, 1, 1, 1], 'float32'))


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
