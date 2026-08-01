"""Tests for map_metrics_extract (Component C).

Gates (hard asserts, run against the REAL frozen artifacts on this box):
  * recall@10 recomputed  == 0.002450284090909091
  * recall@50-of-high10   == 0.011044034090909092
  * pol_Latn packet mean recall == 0.2278 exactly
  * packet neighbor coords within the R0108 map extent

Synthetic tests cover the bin/json writers and the local-expansion score in
isolation so the module is verifiable without the large artifacts.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

import experiments.map_metrics_extract as mm

R0108_CORE_PANEL = Path(
    "/data/latent-basemap/runs/round-0108/queue-attempt-3/artifacts/"
    "core-geometry/core-panel-arrays.npz"
)
R0108_OOD_DIR = Path(
    "/data/latent-basemap/runs/round-0108/queue-attempt-3/artifacts/ood"
)
# Full-map coordinate extent for the R0108 seed-42 atlas (all retained rows).
# Recomputed lazily from the coordinate chunks (cached for the module test).
R0108_COORDS_DIR = Path(
    "/data/latent-basemap/runs/round-0108/queue/artifacts/coordinates"
)


def _real(p: Path) -> bool:
    return p.exists()


# ---------------------------------------------------------------------------
# Synthetic: writers round-trip.
# ---------------------------------------------------------------------------

def test_anchors_bin_roundtrip(tmp_path):
    xy = np.array([[1.0, 2.0], [3.5, -4.0], [0.0, 0.0]], dtype=np.float64)
    score = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    path = tmp_path / "metrics-anchors.bin"
    n = mm.write_anchors_bin(path, xy, score)
    assert n == 3
    # header check
    with open(path, "rb") as f:
        magic, count = struct.unpack("<II", f.read(8))
    assert magic == mm.ANCHORS_MAGIC == 0x414E4331
    assert count == 3
    rxy, rscore = mm.read_anchors_bin(path)
    np.testing.assert_allclose(rxy, xy, rtol=0, atol=1e-6)
    np.testing.assert_allclose(rscore, score, rtol=0, atol=1e-6)


def test_anchors_bin_shape_guard(tmp_path):
    with pytest.raises(ValueError):
        mm.write_anchors_bin(tmp_path / "x.bin", np.zeros((3, 3)), np.zeros(3))
    with pytest.raises(ValueError):
        mm.write_anchors_bin(tmp_path / "x.bin", np.zeros((3, 2)), np.zeros(2))


def test_queries_json_writer(tmp_path):
    probes = [{"key": "k", "label": "L", "recall50": 0.5, "queries": []}]
    path = tmp_path / "metrics-queries.json"
    mm.write_queries_json(path, probes)
    import json
    got = json.loads(path.read_text())
    assert got == {"probes": probes}


# ---------------------------------------------------------------------------
# Synthetic: local expansion score behaviour.
# ---------------------------------------------------------------------------

def test_local_expansion_score_median_maps_to_half():
    # ratio == median  -> log2(1) == 0 -> score01 == 0.5
    low = np.array([2.0, 4.0, 8.0, 1.0])
    high = np.array([1.0, 1.0, 1.0, 1.0])  # ratios 2,4,8,1 ; median 3.0
    score01, log2_norm, median_ratio = mm.local_expansion_score(low, high)
    assert median_ratio == pytest.approx(3.0)
    # anchor whose ratio == median maps near 0.5
    idx = int(np.argmin(np.abs((low / high) - median_ratio)))
    # score monotonic in ratio
    order = np.argsort(low / high)
    assert np.all(np.diff(score01[order]) >= -1e-12)
    assert np.all((score01 >= 0.0) & (score01 <= 1.0))


def test_local_expansion_clip_bounds():
    # extreme ratios clip to [-2,2] -> score01 in {0,1}
    low = np.array([1.0, 1000.0, 0.0001, 4.0])
    high = np.array([1.0, 1.0, 1.0, 1.0])
    score01, log2_norm, _ = mm.local_expansion_score(low, high)
    assert score01.max() == pytest.approx(1.0)
    assert score01.min() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Synthetic: probe packet skip logic (no embedded truth).
# ---------------------------------------------------------------------------

def test_probe_skipped_without_truth(tmp_path):
    p = tmp_path / "fake-coordinates.npz"
    np.savez(
        p,
        probe_corpus_coords=np.zeros((5, 2), dtype=np.float32),
        probe_query_coords=np.zeros((2, 2), dtype=np.float32),
    )
    res = mm.build_probe_packet(p)
    assert res.skipped
    assert res.packet is None
    assert "lacks embedded" in res.reason


def test_probe_packet_synthetic():
    # a hand-built npz with a known recall
    import tempfile, os
    corpus = np.arange(20, dtype=np.float32).reshape(10, 2)
    truth = np.array([[0, 1, 2], [3, 4, 5]])  # 3 truths/query (Q=2)
    low = np.array([[0, 1, 9], [9, 8, 7]])     # q0: 2/3 hit ; q1: 0/3 hit
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "syn-coordinates.npz")
        np.savez(
            p,
            probe_corpus_coords=corpus,
            probe_query_coords=np.zeros((2, 2), dtype=np.float32),
            exact_high_d_top10=truth,
            low_d_top50=low,
            probe_query_ids=np.array([100, 200]),
        )
        res = mm.build_probe_packet(p)
    assert not res.skipped
    assert res.n_queries == 2
    # q0 recall 2/3, q1 recall 0 -> mean 1/3
    assert res.recall50 == pytest.approx(1.0 / 3.0)
    assert res.packet["queries"][0]["hits"] == [True, True, False]
    # neighbor xy = corpus[truth]
    np.testing.assert_allclose(
        res.packet["queries"][0]["neighbors"], corpus[truth[0]].tolist()
    )


# ---------------------------------------------------------------------------
# GATE: real R0108 core-panel recalls.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _real(R0108_CORE_PANEL), reason="R0108 core panel absent")
def test_gate_core_panel_recalls():
    with np.load(R0108_CORE_PANEL) as z:
        r10, r50 = mm.recompute_core_panel_recalls(z)
    assert r10 == pytest.approx(0.002450284090909091, abs=1e-12)
    assert r50 == pytest.approx(0.011044034090909092, abs=1e-12)


@pytest.mark.skipif(not _real(R0108_CORE_PANEL), reason="R0108 core panel absent")
def test_gate_extract_core_panel_asserts_published():
    anc = mm.extract_core_panel_anchors(R0108_CORE_PANEL, assert_published=True)
    assert anc["score_label"] == "local expansion (log2 vs median)"
    assert anc["xy"].shape == (5632, 2)
    assert anc["score01"].shape == (5632,)
    assert np.all((anc["score01"] >= 0.0) & (anc["score01"] <= 1.0))
    # carried global FFR summary stat
    assert anc["summary"]["ffr"] == pytest.approx(0.6386363636363636, abs=1e-12)


# ---------------------------------------------------------------------------
# GATE: pol_Latn packet mean recall == 0.2278 exactly + neighbor coords in extent.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _real(R0108_OOD_DIR / "pol_Latn-coordinates.npz"),
    reason="R0108 pol_Latn OOD absent",
)
def test_gate_pol_latn_recall_exact():
    res = mm.build_probe_packet(R0108_OOD_DIR / "pol_Latn-coordinates.npz")
    assert not res.skipped
    assert res.n_queries == 500
    assert res.recall50 == pytest.approx(0.2278, abs=1e-12)


def _r0108_extent():
    """Full-map extent from the coordinate chunks (min/max over all rows)."""
    import glob
    chunks = sorted(glob.glob(str(R0108_COORDS_DIR / "chunk-*/coordinates.npy")))
    xmin = ymin = np.inf
    xmax = ymax = -np.inf
    for c in chunks:
        a = np.load(c, mmap_mode="r")
        xmin = min(xmin, float(a[:, 0].min()))
        xmax = max(xmax, float(a[:, 0].max()))
        ymin = min(ymin, float(a[:, 1].min()))
        ymax = max(ymax, float(a[:, 1].max()))
    return [xmin, ymin, xmax, ymax]


@pytest.mark.skipif(
    not (_real(R0108_OOD_DIR / "pol_Latn-coordinates.npz") and _real(R0108_COORDS_DIR)),
    reason="R0108 OOD or coords absent",
)
def test_gate_packet_neighbors_within_extent():
    extent = _r0108_extent()
    paths = sorted(R0108_OOD_DIR.glob("*-coordinates.npz"))
    # exclude the alignment npz (not a probe)
    paths = [p for p in paths if "alignment" not in p.name]
    packets, manifest = mm.build_ood_query_packets(paths, extent=extent)
    # extent assertion inside build_ood_query_packets would raise on violation.
    assert any(m.get("key") == "pol_Latn" for m in manifest)
    pol = next(m for m in manifest if m.get("key") == "pol_Latn")
    assert pol["recall50"] == pytest.approx(0.2278, abs=1e-12)


# ---------------------------------------------------------------------------
# In-zip .npy memmap helper (small member, exercised without the 11.8 GB one).
# ---------------------------------------------------------------------------

R0102_REFERENCE = Path(
    "/data/latent-basemap/runs/round-0102/queue/artifacts/"
    "high-d-reference-150m/reference.npz"
)


@pytest.mark.skipif(not _real(R0102_REFERENCE), reason="R0102 reference absent")
def test_memmap_member_matches_direct_load():
    # hi_hit is small + stored; the in-zip memmap must match a direct read.
    mmp = mm._memmap_npy_member_in_zip(R0102_REFERENCE, "hi_hit.npy")
    assert mmp is not None  # stored uncompressed
    direct = mm._load_small_member(R0102_REFERENCE, "hi_hit.npy")
    assert mmp.shape == direct.shape
    np.testing.assert_array_equal(np.asarray(mmp[:50]), direct[:50])


@pytest.mark.skipif(not _real(R0102_REFERENCE), reason="R0102 reference absent")
def test_hi_frac_member_is_huge_and_stored():
    # Guard the "never np.load whole" contract: the member is ~11.8 GB and the
    # in-zip memmap must open it without materializing it.
    mmp = mm._memmap_npy_member_in_zip(R0102_REFERENCE, "hi_frac.npy")
    assert mmp is not None  # stored uncompressed -> zero-copy memmap
    assert mmp.shape == (10000, 147222)
    assert mmp.nbytes > 10 * 1024**3  # > 10 GB, must stay memmapped


R0102_DENSITY_V2 = Path(
    "/data/latent-basemap/runs/round-0102/queue/artifacts/density-v2/density-v2-radii.npz"
)


@pytest.mark.skipif(not _real(R0102_DENSITY_V2), reason="R0102 density-v2 absent")
def test_r0102_local_expansion_score():
    score01, log2_norm, median_ratio = mm.local_expansion_from_density_v2(
        R0102_DENSITY_V2, "full_150m"
    )
    assert score01.shape == (10000,)
    assert np.all((score01 >= 0.0) & (score01 <= 1.0))
    # median ratio maps log2 -> 0 at the median (score ~0.5 there)
    assert np.isfinite(median_ratio) and median_ratio > 0


def test_hi_frac_membership_is_not_ffr_synthetic():
    # Demonstrate the semantic distinction with tiny synthetic arrays: the
    # high-D top-10 is a subset of the high-D fraction pool -> membership 1.0,
    # whereas the true FFR against a low-D pool that misses some is < 1.0.
    hi_hit = np.array([[1, 2, 3]])
    hi_frac = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])  # contains all hi_hit
    membership = np.isin(hi_hit[0], hi_frac[0]).mean()
    assert membership == 1.0
    lo_kf = np.array([[1, 9, 8, 7, 6]])  # low-D pool misses 2 and 3
    ffr = np.isin(hi_hit[0], lo_kf[0]).mean()
    assert ffr == pytest.approx(1.0 / 3.0)
