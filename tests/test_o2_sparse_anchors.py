"""O2 — sparse anchor_ids API: explicit landmark-hold mechanism for growth
training (e.g. the 2M->4M step). Mirrors tests/test_p0_2_3_trainer.py's style:
tiny synthetic data, a real ParametricUMAP, precomputed edge-list fit() on CPU.

Covers:
  (a) sparse targets are allocated only for landmarks, not n_train.
  (b) the hold term samples ONLY landmark ids (instrumented via
      _debug_track_hold_samples).
  (c) fail-closed BEFORE training on non-unique / out-of-bounds / length-
      mismatch / non-finite / empty landmark sets.
  (d) deterministic landmark-hold sampling given a seed.
  (e) anchor_holdout_fraction reserves ids that are never sampled.
  (f) landmark id/target hashes + counts land in _train_stats.
  (g) backfill_4m_manifest's synthetic manifest passes validate_graph_content.
"""
import sys, os, numpy as np, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from basemap.pumap.parametric_umap.core import ParametricUMAP
from basemap.graph_validation import graph_manifest_v2, write_manifest, validate_graph_content


def _edges(n, e, seed=0):
    rng = np.random.RandomState(seed)
    return (rng.randint(0, n, e).astype(np.int32), rng.randint(0, n, e).astype(np.int32),
            rng.rand(e).astype(np.float32))


def _edges_with_manifest(tmp_path, n=200, e=4000, dim=8, seed=1):
    X = np.random.RandomState(seed).randn(n, dim).astype(np.float32)
    s, t, w = _edges(n, e, seed)
    ep = str(tmp_path / "edges.npz")
    np.savez(ep, sources=s, targets=t, weights=w, n_nodes=n, k=15)
    write_manifest(ep + ".manifest.json", graph_manifest_v2(s, t, n, X=X, graph_path=ep, k=15))
    return X, ep


def _landmark_npz(tmp_path, n_train, n_landmarks=40, seed=0, n_components=2, name="anchors.npz"):
    rng = np.random.RandomState(seed)
    ids = rng.choice(n_train, size=n_landmarks, replace=False).astype(np.int64)
    targets = (rng.randn(n_landmarks, n_components) * 5.0).astype(np.float32)
    p = str(tmp_path / name)
    np.savez(p, anchor_ids=ids, anchor_targets=targets)
    return p, ids, targets


def _umap(anchor_ids_path, **kw):
    base = dict(a=1., b=1., correlation_weight=0.0, n_epochs=2, batch_size=32,
                total_steps_estimate=20, lr_schedule='cosine', warmup_steps=0, device='cpu',
                positive_target_mode='binary', use_amp=False, require_full_budget=False,
                anchor_hold_weight=1.0, anchor_hold_fraction=0.25,
                anchor_ids_path=anchor_ids_path)
    base.update(kw)
    return ParametricUMAP(**base)


# ── (a) sparse allocation scoped to landmarks, not n_train ──────────────────

def test_sparse_targets_allocated_only_for_landmarks(tmp_path):
    X, ep = _edges_with_manifest(tmp_path, n=200)
    ap, ids, targets = _landmark_npz(tmp_path, n_train=200, n_landmarks=40)
    m = _umap(ap, gpu_resident_data=True)
    m.fit(X, precomputed_edges_path=ep)
    assert tuple(m._anchor_targets_dev.shape) == (40, 2)
    assert m.anchor_targets_.shape == (40, 2)
    assert m._train_stats['anchor_landmark_count'] == 40
    assert m._train_stats['non_anchor_row_count'] == 160


# ── (b) hold term samples ONLY landmark ids ──────────────────────────────

@pytest.mark.parametrize("resident", [True, False])
def test_hold_samples_only_landmark_ids(tmp_path, resident):
    X, ep = _edges_with_manifest(tmp_path, n=200)
    ap, ids, targets = _landmark_npz(tmp_path, n_train=200, n_landmarks=40)
    m = _umap(ap, gpu_resident_data=resident)
    m._debug_track_hold_samples = True
    m.fit(X, precomputed_edges_path=ep)
    seen = set(m._hold_sampled_row_ids)
    assert seen, "no hold samples recorded"
    assert seen <= set(ids.tolist())


# ── (c) fail-closed BEFORE training ──────────────────────────────────────

def test_nonunique_ids_raise(tmp_path):
    X, ep = _edges_with_manifest(tmp_path, n=200)
    ids = np.array([1, 2, 2, 3], dtype=np.int64)
    targets = np.random.RandomState(0).randn(4, 2).astype(np.float32)
    ap = str(tmp_path / "dup.npz"); np.savez(ap, anchor_ids=ids, anchor_targets=targets)
    m = _umap(ap)
    with pytest.raises(ValueError, match="duplicate"):
        m.fit(X, precomputed_edges_path=ep)
    assert not m.is_fitted


def test_out_of_bounds_ids_raise(tmp_path):
    X, ep = _edges_with_manifest(tmp_path, n=200)
    ids = np.array([1, 2, 3, 500], dtype=np.int64)   # 500 >= n_train=200
    targets = np.random.RandomState(0).randn(4, 2).astype(np.float32)
    ap = str(tmp_path / "oob.npz"); np.savez(ap, anchor_ids=ids, anchor_targets=targets)
    m = _umap(ap)
    with pytest.raises(ValueError, match="out of range"):
        m.fit(X, precomputed_edges_path=ep)
    assert not m.is_fitted


def test_length_mismatch_raises(tmp_path):
    X, ep = _edges_with_manifest(tmp_path, n=200)
    ids = np.array([1, 2, 3], dtype=np.int64)
    targets = np.random.RandomState(0).randn(4, 2).astype(np.float32)   # 4 != 3
    ap = str(tmp_path / "mismatch.npz"); np.savez(ap, anchor_ids=ids, anchor_targets=targets)
    m = _umap(ap)
    with pytest.raises(ValueError, match="length"):
        m.fit(X, precomputed_edges_path=ep)
    assert not m.is_fitted


def test_nonfinite_targets_raise(tmp_path):
    X, ep = _edges_with_manifest(tmp_path, n=200)
    ids = np.array([1, 2, 3, 4], dtype=np.int64)
    targets = np.random.RandomState(0).randn(4, 2).astype(np.float32)
    targets[2, 0] = np.nan
    ap = str(tmp_path / "nonfinite.npz"); np.savez(ap, anchor_ids=ids, anchor_targets=targets)
    m = _umap(ap)
    with pytest.raises(ValueError, match="non-finite"):
        m.fit(X, precomputed_edges_path=ep)
    assert not m.is_fitted


def test_empty_landmark_set_raises(tmp_path):
    X, ep = _edges_with_manifest(tmp_path, n=200)
    ids = np.zeros(0, dtype=np.int64)
    targets = np.zeros((0, 2), dtype=np.float32)
    ap = str(tmp_path / "empty.npz"); np.savez(ap, anchor_ids=ids, anchor_targets=targets)
    m = _umap(ap)
    with pytest.raises(ValueError, match="empty"):
        m.fit(X, precomputed_edges_path=ep)
    assert not m.is_fitted


# ── (d) deterministic sampling given a seed ──────────────────────────────

def test_deterministic_hold_sampling_given_seed(tmp_path):
    X, ep = _edges_with_manifest(tmp_path, n=200)
    ap, ids, targets = _landmark_npz(tmp_path, n_train=200, n_landmarks=40)

    def run():
        m = _umap(ap, gpu_resident_data=True)
        m._debug_track_hold_samples = True
        m.fit(X, precomputed_edges_path=ep, random_state=7)
        return list(m._hold_sampled_row_ids)

    seq1 = run()
    seq2 = run()
    assert len(seq1) > 0
    assert seq1 == seq2


# ── (e) anchor_holdout_fraction reserves ids never sampled ───────────────

def test_holdout_fraction_reserves_ids(tmp_path):
    X, ep = _edges_with_manifest(tmp_path, n=200)
    ap, ids, targets = _landmark_npz(tmp_path, n_train=200, n_landmarks=40)
    m = _umap(ap, gpu_resident_data=True, anchor_holdout_fraction=0.5)
    m._debug_track_hold_samples = True
    m.fit(X, precomputed_edges_path=ep, random_state=3)
    assert m.anchor_holdout_ids_.shape[0] == 20
    assert m.anchor_ids_.shape[0] == 20
    seen = set(m._hold_sampled_row_ids)
    assert seen, "no hold samples recorded"
    assert seen.isdisjoint(set(m.anchor_holdout_ids_.tolist()))
    assert seen <= set(m.anchor_ids_.tolist())
    assert m._train_stats['anchor_landmark_holdout_count'] == 20
    assert m._train_stats['anchor_landmark_active_count'] == 20


def test_holdout_fraction_deterministic_split(tmp_path):
    X, ep = _edges_with_manifest(tmp_path, n=200)
    ap, ids, targets = _landmark_npz(tmp_path, n_train=200, n_landmarks=40)

    def run():
        m = _umap(ap, gpu_resident_data=True, anchor_holdout_fraction=0.5)
        m.fit(X, precomputed_edges_path=ep, random_state=3)
        return sorted(m.anchor_holdout_ids_.tolist())

    assert run() == run()


def test_holdout_fraction_out_of_range_raises(tmp_path):
    X, ep = _edges_with_manifest(tmp_path, n=200)
    ap, ids, targets = _landmark_npz(tmp_path, n_train=200, n_landmarks=40)
    m = _umap(ap, gpu_resident_data=True, anchor_holdout_fraction=1.0)
    with pytest.raises(ValueError, match="anchor_holdout_fraction"):
        m.fit(X, precomputed_edges_path=ep)
    assert not m.is_fitted


# ── (f) hashes / counts land in _train_stats ──────────────────────────────

def test_train_stats_records_landmark_hashes(tmp_path):
    X, ep = _edges_with_manifest(tmp_path, n=200)
    ap, ids, targets = _landmark_npz(tmp_path, n_train=200, n_landmarks=40)
    m = _umap(ap, gpu_resident_data=True)
    m.fit(X, precomputed_edges_path=ep)
    s = m._train_stats
    assert isinstance(s['anchor_landmark_id_hash'], str) and len(s['anchor_landmark_id_hash']) > 0
    assert isinstance(s['anchor_landmark_target_hash'], str) and len(s['anchor_landmark_target_hash']) > 0
    assert s['anchor_landmark_path'] == ap
    assert s['anchor_landmark_count'] == 40
    assert s['n_train_rows'] == 200


# ── config semantics: mutual exclusivity + hold-weight requirement ────────

def test_anchor_ids_path_and_anchored_init_mutually_exclusive(tmp_path):
    X, ep = _edges_with_manifest(tmp_path, n=200)
    ap, ids, targets = _landmark_npz(tmp_path, n_train=200, n_landmarks=40)
    m = _umap(ap, anchored_init="pca")
    with pytest.raises(ValueError, match="alternative anchor-target"):
        m.fit(X, precomputed_edges_path=ep)
    assert not m.is_fitted


def test_anchor_ids_path_requires_hold_weight(tmp_path):
    X, ep = _edges_with_manifest(tmp_path, n=200)
    ap, ids, targets = _landmark_npz(tmp_path, n_train=200, n_landmarks=40)
    m = _umap(ap, anchor_hold_weight=0.0)
    with pytest.raises(ValueError, match="anchor_hold_weight"):
        m.fit(X, precomputed_edges_path=ep)
    assert not m.is_fitted


# ── backfill_4m_manifest synthetic unit test ───────────────────────────────

def test_backfill_4m_manifest_synthetic(tmp_path):
    from backfill_4m_manifest import build_manifest_for_asset

    n, dim = 60, 8
    X = np.random.RandomState(0).randn(n, dim).astype(np.float32)
    train_dir = tmp_path / "train"; train_dir.mkdir()
    np.save(train_dir / "data-00000.npy", X)
    s = np.repeat(np.arange(n), 5).astype(np.int32)
    t = np.random.RandomState(1).randint(0, n, len(s)).astype(np.int32)
    w = np.random.RandomState(2).rand(len(s)).astype(np.float32)
    graph_path = str(tmp_path / "edges_k50_fuzzy.npz")
    np.savez(graph_path, sources=s, targets=t, weights=w, n_nodes=n, k=5)
    asset = dict(name="synthetic-4m", graph=graph_path, train=str(train_dir),
                sample_idx=str(tmp_path / "sample_indices.npy"), dim=dim)

    man = build_manifest_for_asset(asset, cosine_probe=False)   # random edges: skip the kNN probe
    mpath = graph_path + ".manifest.json"
    assert os.path.exists(mpath)
    assert man['schema'] == 'graph_manifest.v2'
    assert 'graph_sha' in man
    assert man['data_shards'] == ['data-00000.npy']
    assert 'data-00000.npy' in man['data_shard_sha']

    trusted = validate_graph_content(
        graph_path, man, shard_paths=[str(train_dir / "data-00000.npy")])
    assert 'graph_sha' in trusted and 'data_shard_sha' in trusted


if __name__ == '__main__':
    pytest.main([__file__, '-q'])


# ── fable-finding regressions ─────────────────────────────────────────────────

def test_float_valued_anchor_ids_rejected(tmp_path):
    # a float64 round-trip of ids must NOT be silently floored onto wrong rows.
    X, ep = _edges_with_manifest(tmp_path, n=200)
    p = str(tmp_path / "float_ids.npz")
    np.savez(p, anchor_ids=np.array([1.7, 2.2, 3.9], dtype=np.float64),
             anchor_targets=np.zeros((3, 2), np.float32))
    m = _umap(p, gpu_resident_data=True)
    with pytest.raises(ValueError, match="integer-valued row indices"):
        m.fit(X, precomputed_edges_path=ep)


def test_integer_valued_float_anchor_ids_accepted(tmp_path):
    # exact-integer floats (e.g. [1.0, 2.0]) are fine.
    X, ep = _edges_with_manifest(tmp_path, n=200)
    p = str(tmp_path / "intfloat_ids.npz")
    np.savez(p, anchor_ids=np.array([1.0, 5.0, 9.0], dtype=np.float64),
             anchor_targets=np.zeros((3, 2), np.float32))
    m = _umap(p, gpu_resident_data=True)
    m.fit(X, precomputed_edges_path=ep)   # trains without error
    assert m._train_stats["anchor_landmark_count"] == 3


def test_anchor_ids_path_with_stray_anchored_init_path_raises(tmp_path):
    # sparse + a stray dense anchored_init_path (even anchored_init='none') must
    # fail loudly rather than silently ignore the dense file.
    X, ep = _edges_with_manifest(tmp_path, n=200)
    ap, ids, targets = _landmark_npz(tmp_path, n_train=200)
    m = _umap(ap, anchored_init_path="/some/teacher.parquet")
    with pytest.raises(ValueError, match="alternative anchor-target"):
        m.fit(X, precomputed_edges_path=ep)
