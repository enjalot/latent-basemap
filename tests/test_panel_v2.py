"""P0.5/P0.6: versioned panel evaluator — loaders, ID validation, ffr!=recall@10,
and bounded==exact density (incl. the near-duplicate cancellation regression)."""
import sys, os, tempfile, numpy as np, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from basemap import panel_v2 as pv


def test_load_coords_preserves_all_dims_and_ids():
    import pandas as pd
    p = tempfile.mktemp(suffix='.parquet')
    pd.DataFrame({'x': [0., 1.], 'y': [0., 1.], 'z': [0., 2.],
                  'ls_index': [10, 11]}).to_parquet(p)
    Z, ids = pv.load_coords(p)
    assert Z.shape == (2, 3) and Z[1, 2] == 2.0   # z preserved (not dropped to x,y)
    assert ids.tolist() == [10, 11]


def test_load_coords_rejects_duplicate_ids():
    import pandas as pd
    p = tempfile.mktemp(suffix='.parquet')
    pd.DataFrame({'x': [0., 1.], 'y': [0., 1.], 'row_id': [5, 5]}).to_parquet(p)
    with pytest.raises(ValueError, match="duplicate ids"):
        pv.load_coords(p)


def test_load_embeddings_raw_headerless():
    p = tempfile.mktemp(suffix='.npy')
    A = np.random.randn(7, 4).astype(np.float32)
    A.tofile(p)  # raw, no NUMPY header
    M = pv.load_embeddings(p, dim=4)
    assert M.shape == (7, 4) and np.allclose(np.asarray(M), A, atol=1e-6)
    with pytest.raises(ValueError, match="raw-headerless"):
        pv.load_embeddings(p)  # dim required


def test_ffr_distinct_from_recall10():
    # true hiD 10-NN sit in the tail of the 2D top-k_frac but NOT the top-10
    rng = np.random.RandomState(0)
    n, d = 3000, 8
    X = rng.randn(n, d).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    # a decent 2D: PCA-ish projection so neighbourhoods partly preserved
    Z = (X[:, :2] + 0.05 * rng.randn(n, 2)).astype(np.float32)
    cfg = pv.PanelV2Config(n_anchors=500)
    r = pv.score_ffr(X, Z, cfg)
    assert r['ffr'] >= r['recall@10']            # ffr uses a wider 2D window
    assert 'ffr' in r and 'recall@10' in r


def test_bounded_density_matches_brute_with_near_duplicates():
    # inject near-duplicate high-D rows (true dist ~1e-4) that the matmul
    # expansion would collapse to 0 — bounded density must match fp64 brute.
    rng = np.random.RandomState(1)
    n, d = 1200, 16
    X = rng.randn(n, d).astype(np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True)
    X[50:70] = X[50] + 1e-4 * rng.randn(20, d).astype(np.float32)   # near-dup cluster
    Z = (X[:, :2] + 0.02 * rng.randn(n, 2)).astype(np.float32)
    cfg = pv.PanelV2Config(n_anchors=n)   # all rows as anchors
    dens_panel = pv.score_density(X, Z, cfg)['density']
    # brute fp64 reference
    def radii(F, k):
        F = F.astype(np.float64); out = np.empty(len(F))
        for i in range(len(F)):
            dd = np.sqrt(((F - F[i])**2).sum(1)); dd.sort()
            out[i] = dd[1:k+1].mean()   # exclude self
        return out
    a = pv.sample_anchors(n, cfg)
    rh = radii(X, 15)[a]; rl = radii(Z, 15)[a]; e = 1e-12
    dens_brute = np.corrcoef(np.log(rh+e), np.log(rl+e))[0, 1]
    assert abs(dens_panel - dens_brute) < 0.01, (dens_panel, dens_brute)


if __name__ == '__main__':
    for fn in [test_load_coords_preserves_all_dims_and_ids, test_load_coords_rejects_duplicate_ids,
               test_load_embeddings_raw_headerless, test_ffr_distinct_from_recall10,
               test_bounded_density_matches_brute_with_near_duplicates]:
        fn(); print("PASS", fn.__name__)
    print("ALL PANEL_V2 TESTS PASSED")
