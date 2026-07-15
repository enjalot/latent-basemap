"""S2.5 acceptance — content-addressed high-D reference cache.

The controlling property: rescoring multiple maps of the same corpus reuses ONE
verified high-D reference, the cached path is numerically identical to inline,
and a drifted reference fails closed rather than mis-scoring.
"""
import numpy as np
import pytest

import basemap.panel_v2 as pv


def _fixture(seed=7, n=1200, d=12, k=16):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype("float32")
    Z1 = rng.randn(n, 2).astype("float32")
    Z2 = rng.randn(n, 2).astype("float32")
    C = rng.randn(k, d).astype("float32")
    return X, Z1, Z2, {k: C}


def _metrics(res):
    return (res["ffr"], res["recall@k"], res["purity"]["k16"], res["density"])


def test_cached_matches_inline_bit_for_bit():
    X, Z1, _, cents = _fixture()
    cfg = pv.PanelV2Config(frac=0.02, n_anchors=100, corpus_chunk=256)
    inline = pv.score_panel(X, Z1, config=cfg, centroids_by_k=cents, provenance={"t": "inline"})
    aidx = pv.sample_anchors(len(X), cfg)
    ref = pv.build_hiD_reference(X, aidx, cfg, cents)
    cached = pv.score_panel(X, Z1, config=cfg, centroids_by_k=cents,
                            hiD_reference=ref, provenance={"t": "cached"})
    assert _metrics(inline) == _metrics(cached)
    assert inline["purity_numerators"] == cached["purity_numerators"]
    assert cached["provenance"]["hiD_reference_reused"] is True
    assert inline["provenance"]["hiD_reference_reused"] is False


def test_one_reference_scores_many_maps():
    X, Z1, Z2, cents = _fixture()
    cfg = pv.PanelV2Config(frac=0.02, n_anchors=100, corpus_chunk=256)
    aidx = pv.sample_anchors(len(X), cfg)
    ref = pv.build_hiD_reference(X, aidx, cfg, cents)
    r1 = pv.score_panel(X, Z1, config=cfg, centroids_by_k=cents, hiD_reference=ref,
                        provenance={"map": 1})
    r2 = pv.score_panel(X, Z2, config=cfg, centroids_by_k=cents, hiD_reference=ref,
                        provenance={"map": 2})
    # Same reference key for both maps; different low-D results.
    assert r1["provenance"]["hiD_reference_key"] == r2["provenance"]["hiD_reference_key"]
    assert r1["provenance"]["hiD_reference_reused"] and r2["provenance"]["hiD_reference_reused"]
    assert _metrics(r1) != _metrics(r2)     # low-D differs per map


def test_reference_fails_closed_on_drifted_data():
    X, Z1, _, cents = _fixture()
    cfg = pv.PanelV2Config(frac=0.02, n_anchors=100, corpus_chunk=256)
    aidx = pv.sample_anchors(len(X), cfg)
    ref = pv.build_hiD_reference(X, aidx, cfg, cents)
    X2 = X.copy(); X2[::5] += 3.0          # mutate the corpus -> different fingerprint
    with pytest.raises(ValueError, match="key mismatch"):
        pv.score_panel(X2, Z1, config=cfg, centroids_by_k=cents,
                       hiD_reference=ref, provenance={"t": "drift"})


def test_reference_fails_closed_on_drifted_centroids():
    X, Z1, _, cents = _fixture()
    cfg = pv.PanelV2Config(frac=0.02, n_anchors=100, corpus_chunk=256)
    aidx = pv.sample_anchors(len(X), cfg)
    ref = pv.build_hiD_reference(X, aidx, cfg, cents)
    bad = {16: cents[16] + 1.0}
    with pytest.raises(ValueError, match="key mismatch"):
        pv.score_panel(X, Z1, config=cfg, centroids_by_k=bad,
                       hiD_reference=ref, provenance={"t": "cdrift"})


def test_reference_save_load_roundtrip(tmp_path):
    X, Z1, _, cents = _fixture()
    cfg = pv.PanelV2Config(frac=0.02, n_anchors=100, corpus_chunk=256)
    aidx = pv.sample_anchors(len(X), cfg)
    ref = pv.build_hiD_reference(X, aidx, cfg, cents)
    p = pv.save_hiD_reference(ref, str(tmp_path / "ref"))
    loaded = pv.load_hiD_reference(p)
    assert loaded["key"] == ref["key"]
    a = pv.score_panel(X, Z1, config=cfg, centroids_by_k=cents, hiD_reference=ref,
                       provenance={"t": "mem"})
    b = pv.score_panel(X, Z1, config=cfg, centroids_by_k=cents, hiD_reference=loaded,
                       provenance={"t": "disk"})
    assert _metrics(a) == _metrics(b)
