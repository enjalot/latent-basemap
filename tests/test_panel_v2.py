"""Panel v2 canonical evaluator tests (P0-C).

Covers: the pure FFR/recall formulas (non-tautological), exact ID alignment,
overselect+exact-rerank high-D neighbours on >k near-duplicates, projection
out-of-sample guard, multi-shard raw loader, per-metric masks, and equality of
the streamed panel against a brute fp64 reference.
"""
import sys, os, json, numpy as np, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from basemap import panel_v2 as pv


# ── pure formulas: FFR != recall@k by construction (fixes the tautological test) ──

def test_ffr_and_recall_are_distinct():
    # hiD true top-10 = ids 100..109. In the loD k_frac list they sit at
    # positions 11..20 (absent from loD top-10). recall@10 must be 0; FFR = 1.
    hi = np.arange(100, 110)[None, :]                       # (1,10)
    lo = np.concatenate([np.arange(200, 210), np.arange(100, 110)])[None, :]  # (1,20)
    assert pv.recall_at_k_from_neighbors(hi, lo, 10) == 0.0
    assert pv.ffr_from_neighbors(hi, lo, 10) == 1.0


def test_ffr_partial_overlap():
    hi = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    lo = np.array([[1, 2, 3, 4, 5, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]])
    assert pv.ffr_from_neighbors(hi, lo, 10) == 0.5


# ── ID alignment (exact, never sort-inferred) ────────────────────────────────────

def test_align_gather_by_z_ids():
    X = np.arange(20).reshape(10, 2).astype('float32')
    z_ids = np.array([9, 0, 5])
    Z = np.zeros((3, 2), 'float32')
    Xa, ids, note = pv.align_x_to_z(X, Z, None, z_ids)
    assert np.array_equal(Xa, X[z_ids]) and note == "gather_X_by_z_ids"


def test_align_reorder_same_universe():
    X = np.arange(12).reshape(6, 2).astype('float32')
    x_ids = np.array([5, 4, 3, 2, 1, 0])
    z_ids = np.array([0, 1, 2, 3, 4, 5])
    Z = np.zeros((6, 2), 'float32')
    Xa, ids, note = pv.align_x_to_z(X, Z, x_ids, z_ids)
    # row r of Xa must be the X row whose x_id == z_ids[r]
    assert np.array_equal(Xa[0], X[5]) and np.array_equal(Xa[5], X[0])
    assert note == "reorder_X_to_z"


def test_align_rejects_bad_ids():
    X = np.zeros((5, 2), 'float32'); Z = np.zeros((5, 2), 'float32')
    with pytest.raises(ValueError):
        pv.align_x_to_z(X, Z, None, np.array([0, 0, 1, 2, 3]))     # duplicate
    with pytest.raises(ValueError):
        pv.align_x_to_z(X, Z, None, np.array([0, 1, 2, 3, 99]))    # out of range
    with pytest.raises(ValueError):
        pv.align_x_to_z(X, Z, np.arange(5), None)                  # x_ids w/o z_ids


# ── high-D exactness: overselect+rerank on more than k near-duplicates ────────────

def test_hi_knn_exact_on_near_duplicates():
    # Anchor 0 has 20 near-duplicates (dist ~1e-3) and the rest far away. The fast
    # matmul expansion cancels near-dup distances; overselect+exact-rerank must
    # still return the true k nearest with monotone exact radii.
    rng = np.random.RandomState(0)
    base = rng.randn(1, 32).astype('float32')
    dups = base + rng.randn(20, 32).astype('float32') * 1e-3
    far = rng.randn(200, 32).astype('float32') * 5 + 50
    F = np.concatenate([base, dups, far]).astype('float32')       # row0 = anchor
    cfg = pv.PanelV2Config(overselect=8)
    ids, dist, guard = pv._self_knn(F, np.array([0]), 15, cfg, hi_dim=True, want_dist=True)
    # all 15 neighbours must be among the 20 true near-duplicates (rows 1..20)
    assert set(ids[0].tolist()).issubset(set(range(1, 21))), ids[0]
    assert np.all(np.diff(dist[0]) >= -1e-6), dist[0]             # radii sorted
    # exact radii ~ 1e-3 scale, NOT the ~0 the cancellation would give
    assert dist[0].max() < 0.5


def test_hi_knn_matches_brute_fp64():
    rng = np.random.RandomState(1)
    F = rng.randn(400, 24).astype('float32')
    a = np.array([3, 100, 250])
    ids, dist, _ = pv._self_knn(F, a, 10, pv.PanelV2Config(overselect=8),
                                hi_dim=True, want_dist=True)
    Fd = F.astype('float64')
    for r, ai in enumerate(a):
        d = np.linalg.norm(Fd - Fd[ai], axis=1); d[ai] = np.inf
        true = np.argsort(d)[:10]
        assert set(ids[r].tolist()) == set(true.tolist()), (ai, ids[r], true)


# ── multi-shard raw loader ────────────────────────────────────────────────────────

def test_multishard_raw_loader(tmp_path):
    dim = 8
    A = np.random.RandomState(0).randn(30, dim).astype('float32')
    B = np.random.RandomState(1).randn(20, dim).astype('float32')
    pa = tmp_path / "s0.bin"; pb = tmp_path / "s1.bin"
    A.tofile(pa); B.tofile(pb)
    F = pv.load_embeddings([str(pa), str(pb)], dim=dim)
    assert len(F) == 50
    assert np.allclose(F[0:30], A) and np.allclose(F[30:50], B)
    assert np.allclose(F[np.array([0, 35, 29, 49])], np.stack([A[0], B[5], A[29], B[19]]))
    # trailing-byte shard must be rejected, not silently truncated
    bad = tmp_path / "bad.bin"
    with open(bad, "wb") as fh:
        fh.write(A.tobytes() + b"\x00\x00\x00")
    with pytest.raises(ValueError, match="divisible"):
        pv.load_embeddings([str(bad)], dim=dim)


# ── per-metric masks + full-panel brute equality ─────────────────────────────────

def _brute_panel(X, Z, cfg):
    aidx = pv.sample_anchors(len(Z), cfg)
    kf = max(cfg.k_hit, int(np.ceil(cfg.frac * len(Z))))
    Xd = X.astype('float64'); Zd = Z.astype('float64')
    def knn(F, k):
        out = np.empty((len(aidx), k), np.int64)
        for r, ai in enumerate(aidx):
            d = np.linalg.norm(F - F[ai], axis=1); d[ai] = np.inf
            out[r] = np.argsort(d)[:k]
        return out
    hi = knn(Xd, cfg.k_hit); lo = knn(Zd, kf)
    ffr = np.mean([len(np.intersect1d(hi[i], lo[i])) / cfg.k_hit for i in range(len(aidx))])
    return round(float(ffr), 4)


def test_score_panel_matches_brute():
    rng = np.random.RandomState(3)
    X = rng.randn(1500, 16).astype('float32')
    Z = rng.randn(1500, 2).astype('float32')
    cfg = pv.PanelV2Config(frac=0.01, n_anchors=120, overselect=8, corpus_chunk=137)
    res = pv.score_panel(X, Z, config=cfg, provenance={"t": "unit"})
    assert abs(res["ffr"] - _brute_panel(X, Z, cfg)) <= 0.02, res["ffr"]
    assert res["recall@k"] <= res["ffr"] + 1e-9
    assert res["provenance"]["exactness"].startswith("hi:overselect")
    assert res["guards"]["coords_finite"] is True


def test_masks_separate_ffr_and_purity():
    rng = np.random.RandomState(4)
    X = rng.randn(800, 12).astype('float32'); Z = rng.randn(800, 2).astype('float32')
    C = rng.randn(16, 12).astype('float32')
    cfg = pv.PanelV2Config(frac=0.02, n_anchors=100, corpus_chunk=200)
    m = min(cfg.n_anchors, 800)
    masks = {"ffr": np.ones(m, bool), "purity": (np.arange(m) < 30)}
    res = pv.score_panel(X, Z, config=cfg, centroids_by_k={16: C},
                         anchor_masks=masks, provenance={"t": "mask"})
    assert res["n_ffr_anchors"] == m and res["n_purity_anchors"] == 30
    assert "k16" in res["purity"] and "k16" in res["centroid_hashes"]


# ── projection out-of-sample guard ───────────────────────────────────────────────

def test_projection_rejects_in_sample_queries():
    rng = np.random.RandomState(5)
    X = rng.randn(300, 10).astype('float32'); Z = rng.randn(300, 2).astype('float32')
    proj = {"Xq": X[:5], "Zq": Z[:5], "query_ids": np.arange(5)}   # ids ARE training rows
    with pytest.raises(ValueError, match="held-out|overlap"):
        pv.score_panel(X, Z, config=pv.PanelV2Config(n_anchors=50), projection=proj,
                       provenance={"t": "proj"})


def test_projection_scores_when_held_out():
    rng = np.random.RandomState(6)
    X = rng.randn(300, 10).astype('float32'); Z = rng.randn(300, 2).astype('float32')
    Xq = rng.randn(8, 10).astype('float32'); Zq = rng.randn(8, 2).astype('float32')
    proj = {"Xq": Xq, "Zq": Zq, "query_ids": np.arange(1000, 1008),
            "checkpoint_hash": "abc123"}
    res = pv.score_panel(X, Z, config=pv.PanelV2Config(frac=0.02, n_anchors=50),
                         projection=proj, provenance={"t": "proj"})
    assert "projection" in res and res["projection"]["proj_n_queries"] == 8
    assert 0.0 <= res["projection"]["proj_ffr"] <= 1.0
    assert res["projection"]["proj_checkpoint"] == "abc123"


def test_runner_and_cli_payloads_byte_equivalent(tmp_path):
    # Both entry points call score_panel, so metric payloads (minus runtime
    # telemetry) must be byte-identical. Simulate: direct arrays vs load-from-disk.
    import pandas as pd
    rng = np.random.RandomState(7)
    X = rng.randn(600, 12).astype('float32'); Z = rng.randn(600, 2).astype('float32')
    cfg = pv.PanelV2Config(frac=0.02, n_anchors=80, corpus_chunk=200)
    direct = pv.score_panel(X, Z, config=cfg, provenance={"caller": "runner"})
    # disk round-trip (CLI path)
    ep = tmp_path / "emb.bin"; X.tofile(ep)
    cp = tmp_path / "coords.parquet"
    pd.DataFrame({"x": Z[:, 0], "y": Z[:, 1], "ls_index": np.arange(600)}).to_parquet(cp)
    Xl = pv.load_embeddings([str(ep)], dim=12)
    Zl, zids = pv.load_coords(str(cp))
    cli = pv.score_panel(Xl, Zl, config=cfg, z_ids=zids, provenance={"caller": "cli"})

    telemetry = {"provenance", "guards"}
    dm = {k: v for k, v in direct.items() if k not in telemetry}
    cm = {k: v for k, v in cli.items() if k not in telemetry}
    assert json.dumps(dm, sort_keys=True) == json.dumps(cm, sort_keys=True), (dm, cm)


if __name__ == '__main__':
    import pytest as _p
    _p.main([__file__, '-q'])
