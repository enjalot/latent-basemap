"""Synthetic tests for map_tiles.py (component A).

Run: uv run python -m pytest experiments/tests/test_map_tiles.py -q
No GPU, no /data access — fabricates a tiny coordinates dir under tmp_path.
"""

import os
import struct

import numpy as np
import pytest

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import map_tiles as mt


def _make_coords_dir(tmp_path, arrays):
    """Write a list of f32 [n,2] arrays as chunk-XXXXX/coordinates.npy."""
    root = tmp_path / "coordinates"
    root.mkdir()
    for i, a in enumerate(arrays):
        d = root / f"chunk-{i:05d}"
        d.mkdir()
        np.save(d / "coordinates.npy", np.asarray(a, dtype=np.float32))
    return str(root)


@pytest.fixture
def synth(tmp_path):
    rng = np.random.default_rng(0)
    # 10k points across 3 chunks, spread over a known extent
    pts = rng.uniform(low=[-5.0, 2.0], high=[15.0, 22.0], size=(10_000, 2)).astype(np.float32)
    # force exact corners so extent is deterministic
    pts[0] = [-5.0, 2.0]
    pts[1] = [15.0, 22.0]
    arrays = [pts[:4000], pts[4000:7000], pts[7000:]]
    coords = _make_coords_dir(tmp_path, arrays)
    src = mt.MapSource(coords, cache_dir=str(tmp_path / "cache"))
    return src, pts


def test_source_nrows_and_extent(synth):
    src, pts = synth
    assert src.nrows == 10_000
    xmin, ymin, xmax, ymax = src.extent()
    assert xmin == pytest.approx(-5.0)
    assert ymin == pytest.approx(2.0)
    assert xmax == pytest.approx(15.0)
    assert ymax == pytest.approx(22.0)
    # cached second call
    assert src.extent() == (xmin, ymin, xmax, ymax)


def test_extent_cache_sidecar(synth, tmp_path):
    src, _ = synth
    src.extent()
    files = os.listdir(tmp_path / "cache")
    assert any(f.startswith("extent-") and f.endswith(".json") for f in files)


@pytest.mark.parametrize("level", [64, 128, 256, 512, 1024])
def test_bin_counts_sum_equals_nrows(synth, level):
    src, _ = synth
    extent = src.extent()
    idx, cnt = mt.bin_counts(src, level, extent)
    assert int(cnt.sum()) == src.nrows
    # cell indices are within range and unique
    assert idx.max() < level * level
    assert len(np.unique(idx)) == len(idx)


def test_bin_all_levels_sum_and_pooling(synth):
    src, _ = synth
    extent = src.extent()
    levels = [64, 128, 256, 512, 1024]
    grids = mt.bin_all_levels(src, levels, extent)
    for lvl in levels:
        idx, cnt = grids[lvl]
        assert int(cnt.sum()) == src.nrows, f"level {lvl}"
    # pooling consistency: pooled coarse must equal direct coarse bincount
    for lvl in levels:
        idx, cnt = grids[lvl]
        didx, dcnt = mt.bin_counts(src, lvl, extent)
        dense_pool = np.zeros(lvl * lvl, dtype=np.int64)
        dense_pool[idx] = cnt
        dense_direct = np.zeros(lvl * lvl, dtype=np.int64)
        dense_direct[didx] = dcnt
        assert np.array_equal(dense_pool, dense_direct), f"pool mismatch at {lvl}"


def test_pooling_nested_totals(synth):
    """Sum over level L cells == sum over level 2L cells (both == nrows)."""
    src, _ = synth
    extent = src.extent()
    grids = mt.bin_all_levels(src, [128, 256], extent)
    assert int(grids[128][1].sum()) == int(grids[256][1].sum()) == src.nrows


def test_sample_bins_subset_of_nonempty(synth):
    src, _ = synth
    extent = src.extent()
    level = 64
    idx, cnt = mt.bin_counts(src, level, extent)
    nonempty = set(int(c) for c in idx)
    samples = mt.sample_bins(src, level, extent, k=3, rng_seed=1)
    # every sampled cell is nonempty
    assert set(samples.keys()) <= nonempty
    # every nonempty cell is covered (two-pass guarantees this)
    assert set(samples.keys()) == nonempty
    # at most k per cell, and each sampled row actually falls in that cell
    for cell, rows in samples.items():
        assert len(rows) <= 3
        xy = src.read_rows(rows)
        flat = mt._flat_cell(xy, level, extent)
        assert all(int(c) == cell for c in flat)


def test_sample_bins_forces_pass_two(tmp_path):
    """A cell with a single point far from the mass is rare; ensure pass-2
    sweep still covers it (use tiny max_candidates to stress it)."""
    pts = np.zeros((5000, 2), dtype=np.float32)
    pts[:] = [0.0, 0.0]
    pts[-1] = [100.0, 100.0]  # lone outlier in its own cell
    coords = _make_coords_dir(tmp_path, [pts])
    src = mt.MapSource(coords, cache_dir=str(tmp_path / "c"))
    extent = src.extent()
    level = 64
    idx, _ = mt.bin_counts(src, level, extent)
    samples = mt.sample_bins(src, level, extent, k=3, rng_seed=0,
                             max_candidates=10)
    assert set(samples.keys()) == set(int(c) for c in idx)


def test_row_filter_slice_counts(synth):
    src, _ = synth
    extent = src.extent()
    lo, hi = 1000, 4000
    idx, cnt = mt.bin_counts(src, 64, extent, row_filter=slice(lo, hi))
    assert int(cnt.sum()) == hi - lo


def test_row_filter_ids_counts(synth):
    src, _ = synth
    extent = src.extent()
    ids = np.array([5, 100, 4001, 9999, 7000], dtype=np.int64)
    idx, cnt = mt.bin_counts(src, 64, extent, row_filter=ids)
    assert int(cnt.sum()) == len(ids)


def test_subset_counts_equal_range_size(synth):
    """A slice subset's binned total equals the range size across all levels."""
    src, _ = synth
    extent = src.extent()
    lo, hi = 2500, 8500
    grids = mt.bin_all_levels(src, [64, 128, 256], extent, row_filter=slice(lo, hi))
    for lvl in (64, 128, 256):
        assert int(grids[lvl][1].sum()) == hi - lo


def test_grid_binary_round_trip(synth, tmp_path):
    src, _ = synth
    extent = src.extent()
    idx, cnt = mt.bin_counts(src, 128, extent)
    path = str(tmp_path / "grid-all-128.bin")
    mt.write_grid(path, 128, idx, cnt)
    level2, idx2, cnt2 = mt.read_grid(path)
    assert level2 == 128
    assert np.array_equal(idx, idx2)
    assert np.array_equal(cnt, cnt2)
    # header magic is BIN1
    with open(path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
    assert magic == mt.GRID_MAGIC == 0x42494E31


def test_points_binary_round_trip(synth, tmp_path):
    src, _ = synth
    xy = src.read_rows(np.arange(50))
    path = str(tmp_path / "points-x.bin")
    mt.write_points(path, xy)
    xy2 = mt.read_points(path)
    assert np.allclose(xy, xy2)
    with open(path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
    assert magic == mt.PTS_MAGIC == 0x50545331


def test_write_samples_supertile_bucketing(synth, tmp_path):
    src, _ = synth
    extent = src.extent()
    sample_level = 256
    super_tile = 16
    # fabricate a couple of cells with coord rows; use a fake map_kind path by
    # monkeypatching resolve_texts via minilm? Instead test bucketing math only.
    samples = {0: [1, 2], 255: [3], 256 * 255 + 255: [4]}

    # monkeypatch resolve_texts to avoid /data
    orig = mt.resolve_texts
    mt.resolve_texts = lambda kind, rows, cache_dir=None: [
        {"t": f"row{r}", "g": "grp", "r": int(r)} for r in rows]
    try:
        out_dir = str(tmp_path / "data")
        written = mt.write_samples(out_dir, "all", samples, sample_level,
                                   super_tile, "jina-25m")
    finally:
        mt.resolve_texts = orig

    names = sorted(os.path.basename(p) for p in written)
    # cell 0 -> tile 0_0 ; cell 255 -> cx255 -> sx15, cy0 -> sy0 -> 15_0
    # cell 256*255+255 -> cx255,cy255 -> 15_15
    assert "samples-all-0_0.json" in names
    assert "samples-all-15_0.json" in names
    assert "samples-all-15_15.json" in names

    import json
    with open(os.path.join(out_dir, "samples-all-0_0.json")) as f:
        payload = json.load(f)
    assert "0" in payload["cells"]
    assert len(payload["cells"]["0"]) == 2
    assert payload["cells"]["0"][0]["t"] == "row1"


def test_clean_text_strips_cls():
    assert mt._clean_text("[CLS] hello world") == "hello world"
    assert mt._clean_text("[CLS]tight") == "tight"
    long = "[CLS] " + "x" * 500
    assert len(mt._clean_text(long)) == mt.TEXT_MAX_CHARS
