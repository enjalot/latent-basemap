"""Synthetic-fixture tests for the map pack builder.

Everything runs on a tiny fabricated substrate (a few thousand rows, four
corpora, real parquet chunk files) so the whole suite is seconds of CPU. The
positive controls are the ones that matter:

* ``tile_index`` byte offsets really do delimit each tile's points
* corpus bits survive the pack/unpack round trip
* sidecar offsets fetch back the exact text a row started with
* a deliberately mis-sorted point file is caught by ``validate``
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

import map_pack as mp


N_ROWS = 70_000   # > 256^2 so the pyramid has more than one tile (Z=1)
CORPORA = ["fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2",
           "RedPajama-Data-V2-sample-10B-chunked-120-all-MiniLM-L6-v2",
           "pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2",
           "starcoderdata-code-chunked-120-all-MiniLM-L6-v2"]
SHARES = [0.4, 0.25, 0.25, 0.1]
SHARDS = [3, 2, 4, 1]          # per corpus; unique (rows, shards) signatures
ROWS_PER_SHARD = 30_000


def _text(corpus_code: int, shard: int, row: int) -> str:
    # deliberately includes non-ascii so the utf-8 offsets are exercised
    return f"c{corpus_code}/s{shard}/r{row} — prüfung éà {'x' * (row % 37)}"


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    """A miniature substrate + chunk tree + coordinates, all on tmp_path."""
    root = tmp_path_factory.mktemp("mappack")
    emb_root = root / "embeddings"
    chunk_root = root / "chunks"
    sub_dir = root / "runs" / "queue" / "artifacts" / "tiny-substrate-v1"
    sub_dir.mkdir(parents=True)

    rng = np.random.default_rng(7)
    counts = [int(round(N_ROWS * s)) for s in SHARES]
    counts[0] += N_ROWS - sum(counts)

    prov = np.zeros(N_ROWS, dtype=[("corpus", "u1"), ("shard", "<u2"), ("row", "<i8")])
    at = 0
    shard_sizes: dict[str, dict] = {}
    for code, (name, cnt, nsh) in enumerate(zip(CORPORA, counts, SHARDS)):
        (emb_root / name / "train").mkdir(parents=True)
        cds = name.replace("-all-MiniLM-L6-v2", "")
        (chunk_root / cds / "train").mkdir(parents=True)
        sizes = {}
        for s in range(nsh):
            stem = f"data-{s:05d}-of-{nsh:05d}"
            (emb_root / name / "train" / f"{stem}.npy").write_bytes(b"\0" * 16)
            sizes[f"{name}/train/{stem}.npy"] = 16
            _write_parquet(chunk_root / cds / "train" / f"{stem}.parquet", code, s)
        shard_sizes[name] = {"shard_sizes": sizes}
        # distinct (shard, row) draws, deliberately left unsorted: the 12.5M
        # substrate's provenance is not in (corpus, shard, row) order either
        flat = rng.choice(nsh * ROWS_PER_SHARD, size=cnt, replace=False)
        rng.shuffle(flat)
        prov["corpus"][at:at + cnt] = code
        prov["shard"][at:at + cnt] = flat // ROWS_PER_SHARD
        prov["row"][at:at + cnt] = flat % ROWS_PER_SHARD
        at += cnt
    np.save(sub_dir / "provenance.npy", prov)

    meta = {
        "capability": "tiny-substrate-v1",
        "rows": N_ROWS,
        "composition": {n: {"rows": c} for n, c in zip(CORPORA, counts)},
        "sources": {n: {"shards": s} for n, s in zip(CORPORA, SHARDS)},
        "selection": {"excluded_shards": {}},
    }
    (sub_dir / "substrate.json").write_text(json.dumps(meta))
    (sub_dir.parent.parent / "source-size-manifest.json").write_text(
        json.dumps({"corpora": shard_sizes}))

    # coordinates: three gaussian blobs plus a far outlier halo, so the trimmed
    # core actually has to trim something
    xy = np.concatenate([
        rng.normal([0, 0], 1.0, size=(N_ROWS // 2, 2)),
        rng.normal([6, 3], 0.6, size=(N_ROWS // 3, 2)),
        rng.normal([-4, 5], 1.5, size=(N_ROWS - N_ROWS // 2 - N_ROWS // 3, 2)),
    ]).astype(np.float32)
    xy[:20] *= 60.0
    coords = root / "coordinates.npy"
    np.save(coords, xy)

    return {"root": root, "coords": coords, "sub_dir": sub_dir,
            "emb_root": emb_root, "chunk_root": chunk_root, "prov": prov}


def _write_parquet(path, corpus_code, shard):
    import pyarrow as pa
    import pyarrow.parquet as pq
    texts = [_text(corpus_code, shard, r) for r in range(ROWS_PER_SHARD)]
    pq.write_table(pa.table({"chunk_text": pa.array(texts),
                             "chunk_token_count": pa.array(
                                 [len(t.split()) for t in texts], type=pa.int64())}), path)


@pytest.fixture(scope="module")
def built(fixture, tmp_path_factory, monkeypatch_module):
    out_root = tmp_path_factory.mktemp("packs")
    side_root = tmp_path_factory.mktemp("sidecars")
    monkeypatch_module.setattr(mp, "EMB_ROOT", fixture["emb_root"])
    monkeypatch_module.setattr(mp, "CHUNK_ROOT", fixture["chunk_root"])
    man = mp.build_pack(fixture["coords"], fixture["sub_dir"], "tiny-map",
                        out_root=out_root, sidecar_root=side_root, verbose=False)
    return {"manifest": man, "pack": out_root / "tiny-map",
            "sidecar": side_root / "tiny-substrate-v1"}


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch
    mpatch = MonkeyPatch()
    yield mpatch
    mpatch.undo()


# ---------------------------------------------------------------- unit bits
def test_choose_max_zoom_matches_spec_table():
    assert mp.choose_max_zoom(2_000_000) == 3
    assert mp.choose_max_zoom(12_500_000) == 4
    assert mp.choose_max_zoom(100_000_000) == 5


def test_morton_is_a_bijection_on_a_small_block():
    x, y = np.meshgrid(np.arange(16), np.arange(16))
    m = mp.morton8(x.ravel().astype(np.uint32), y.ravel().astype(np.uint32))
    assert len(np.unique(m)) == 256


def test_quantization_bin_is_an_exact_shift():
    extent = [-3.0, 5.0, -2.0, 6.0]
    rng = np.random.default_rng(0)
    pts = rng.uniform([-3, -2], [5, 6], size=(5000, 2)).astype(np.float32)
    qx, qy = mp.quantize(pts, extent)
    for z in range(0, 6):
        side = mp.TILE_BINS << z
        x = pts[:, 0].astype(np.float64)
        direct = np.clip(np.floor((x - extent[0]) / (extent[1] - extent[0]) * side
                                  ).astype(np.int64), 0, side - 1)
        assert np.array_equal(mp.bins_at(qx, z), direct)


def test_rank_within_groups():
    gids = np.array([1, 1, 1, 2, 2])
    prio = np.array([0.5, 0.1, 0.9, 0.3, 0.2])
    assert mp.rank_within(gids, prio).tolist() == [1, 0, 2, 1, 0]


def test_corpus_map_is_verified_not_assumed(fixture, monkeypatch_module):
    monkeypatch_module.setattr(mp, "EMB_ROOT", fixture["emb_root"])
    sub = mp.Substrate(fixture["sub_dir"])
    assert sub.corpus_map() == dict(enumerate(CORPORA))
    # break the bijection: two corpora with identical (rows, shards) signatures
    meta = json.loads((sub.dir / "substrate.json").read_text())
    meta["composition"][CORPORA[1]]["rows"] = meta["composition"][CORPORA[2]]["rows"]
    meta["sources"][CORPORA[1]]["shards"] = meta["sources"][CORPORA[2]]["shards"]
    bad = sub.dir.parent / "bad"
    bad.mkdir(exist_ok=True)
    (bad / "substrate.json").write_text(json.dumps(meta))
    shutil.copy(sub.dir / "provenance.npy", bad / "provenance.npy")
    with pytest.raises(ValueError, match="cannot resolve corpus names"):
        mp.Substrate(bad).corpus_map()


# ------------------------------------------------------------ pack contract
def test_manifest_shape(built):
    man = built["manifest"]
    assert man["pack_format_version"] == "1"
    assert man["n_points"] == N_ROWS
    assert man["corpus_codes"] == {str(i): c for i, c in enumerate(CORPORA)}
    assert sum(man["corpus_counts"].values()) == N_ROWS
    assert man["tiles"]["tile_bins"] == 256
    x0, x1, y0, y1 = man["frame"]["extent"]
    assert (x1 - x0) == pytest.approx(y1 - y0)  # square extent


def test_density_planes_sum_to_n_at_every_level(built):
    pack = built["pack"]
    for lvl in built["manifest"]["tiles"]["levels"]:
        idx = json.loads((pack / "density" / f"z{lvl['z']}" / "index.json").read_text())
        total = 0
        for tk, ent in idx["tiles"].items():
            tx, ty = (int(v) for v in tk.split("_"))
            for c in ent["corpora"]:
                raw = (pack / "density" / f"z{lvl['z']}" / f"{tx}_{ty}.{c}.u32").read_bytes()
                assert len(raw) == 256 * 256 * 4
                total += int(np.frombuffer(raw, dtype="<u4").sum())
        assert total == N_ROWS
        assert (pack / "density" / f"z{lvl['z']}" / f"{tx}_{ty}.png").is_file()


def test_tile_index_offsets_delimit_the_right_points(built):
    """Positive control: for every tile, the bytes its offsets name decode to
    exactly the points whose quantized coordinates fall inside that tile."""
    pack = built["pack"]
    man = built["manifest"]
    z = man["tiles"]["max_zoom"]
    t = 1 << z
    xy = np.fromfile(pack / "points" / "xy_id.bin", dtype=mp.POINT_DTYPE)
    off = np.fromfile(pack / "points" / "tile_index.u64", dtype="<u8")
    assert len(off) == t * t + 1
    assert int(off[-1]) == len(xy) * 8
    seen = 0
    for ti in range(t * t):
        a = int(off[ti]) // 8
        b = int(off[ti + 1]) // 8
        run = xy[a:b]
        seen += len(run)
        if not len(run):
            continue
        ix = run["x"] >> np.uint16(8 - z)
        iy = run["y"] >> np.uint16(8 - z)
        assert np.all((iy // 256) * t + (ix // 256) == ti)
    assert seen == N_ROWS


def test_corpus_bits_round_trip(built, fixture):
    xy = np.fromfile(built["pack"] / "points" / "xy_id.bin", dtype=mp.POINT_DTYPE)
    ids = xy["packed"] & np.uint32((1 << mp.ID_BITS) - 1)
    corp = xy["packed"] >> np.uint32(mp.ID_BITS)
    prov = fixture["prov"]
    assert len(np.unique(ids)) == N_ROWS
    assert np.array_equal(corp.astype(np.int64), prov["corpus"][ids].astype(np.int64))


def test_points_positions_match_the_quantized_coordinates(built, fixture):
    man = built["manifest"]
    xy = np.fromfile(built["pack"] / "points" / "xy_id.bin", dtype=mp.POINT_DTYPE)
    ids = (xy["packed"] & np.uint32((1 << mp.ID_BITS) - 1)).astype(np.int64)
    coords = np.load(fixture["coords"])
    qx, qy = mp.quantize(coords, man["frame"]["extent"])
    assert np.array_equal(xy["x"], qx[ids])
    assert np.array_equal(xy["y"], qy[ids])


def test_lod_is_stratified_and_min_zoom_ordered(built):
    lod = np.fromfile(built["pack"] / "points" / "lod.bin", dtype=mp.LOD_DTYPE)
    man = built["manifest"]
    assert lod.dtype.itemsize == 9
    assert len(lod) == man["lod"]["n_points"] <= max(N_ROWS // 4, 1)
    assert np.all(np.diff(lod["minz"].astype(int)) >= 0)
    assert lod["minz"].max() <= man["tiles"]["max_zoom"]
    offs = man["lod"]["min_zoom_offsets"]
    assert offs[0] == 0 and offs[-1] == len(lod) * 9
    ids = lod["packed"] & np.uint32((1 << mp.ID_BITS) - 1)
    assert len(np.unique(ids)) == len(lod)


def test_bin_samples_and_snippets_agree(built, fixture):
    pack = built["pack"]
    samples = json.loads((pack / "bins" / "samples_z0.json").read_text())
    snippets = json.loads((pack / "bins" / "snippets_z0.json").read_text())
    assert set(samples) == set(snippets)
    prov = fixture["prov"]
    for kbin, rows in list(samples.items())[:50]:
        assert len(rows) <= mp.BIN_SAMPLE_K
        for row, snip in zip(rows, snippets[kbin]):
            c, s, r = prov[row]
            assert snip == _text(int(c), int(s), int(r))[:mp.SNIPPET_CHARS]


# ------------------------------------------------------------- text sidecar
def test_sidecar_offsets_fetch_the_exact_text(built, fixture):
    side = built["sidecar"]
    offsets = np.fromfile(side / "offsets.u64", dtype="<u8")
    blob = np.memmap(side / "blob.utf8", dtype=np.uint8, mode="r")
    prov = fixture["prov"]
    assert len(offsets) == N_ROWS + 1
    assert np.all(np.diff(offsets.astype(np.int64)) >= 0)
    man = json.loads((side / "manifest.json").read_text())
    assert int(offsets[-1]) == man["blob_bytes"]
    for row in [0, 1, 17, N_ROWS // 3, N_ROWS - 1]:
        a, b = int(offsets[row]), int(offsets[row + 1])
        got = bytes(blob[a:b]).decode("utf-8")
        c, s, r = prov[row]
        assert got == _text(int(c), int(s), int(r))
    # exhaustive: every row, not just a sample
    for row in range(N_ROWS):
        a, b = int(offsets[row]), int(offsets[row + 1])
        c, s, r = prov[row]
        assert bytes(blob[a:b]).decode("utf-8") == _text(int(c), int(s), int(r))


def test_sidecar_validates(built):
    res = mp.validate_sidecar(built["sidecar"])
    assert res["ok"], res["failed"]


# ---------------------------------------------------------------- validator
def test_validate_passes_on_a_clean_pack(built):
    res = mp.validate_pack(built["pack"], full=True)
    assert res["ok"], res["failed"]


def _clone(pack, tmp_path):
    dst = tmp_path / "clone"
    shutil.copytree(pack, dst)
    return dst


def _rehash(pack):
    man = json.loads((pack / "manifest.json").read_text())
    man["files"] = mp.inventory(pack)
    (pack / "manifest.json").write_text(json.dumps(man, indent=1))


def test_validator_catches_a_planted_wrong_order_sort(built, tmp_path):
    """Swap two points across tiles, keeping every byte count identical, and
    re-stamp the inventory hashes — only the order invariant can catch it."""
    pack = _clone(built["pack"], tmp_path)
    xy = np.fromfile(pack / "points" / "xy_id.bin", dtype=mp.POINT_DTYPE)
    z = built["manifest"]["tiles"]["max_zoom"]
    tid = ((xy["y"] >> np.uint16(8 - z)) // 256).astype(np.int64) * (1 << z) + \
          ((xy["x"] >> np.uint16(8 - z)) // 256).astype(np.int64)
    diff = np.flatnonzero(tid != tid[0])
    assert diff.size, "fixture must span more than one tile"
    i, j = 0, int(diff[-1])
    xy[[i, j]] = xy[[j, i]]
    xy.tofile(pack / "points" / "xy_id.bin")
    _rehash(pack)
    res = mp.validate_pack(pack, full=True)
    assert not res["ok"]
    failed = {c["check"] for c in res["failed"]}
    assert "points_sorted" in failed


def test_validator_catches_a_shifted_tile_index(built, tmp_path):
    pack = _clone(built["pack"], tmp_path)
    off = np.fromfile(pack / "points" / "tile_index.u64", dtype="<u8")
    nz = np.flatnonzero(np.diff(off.astype(np.int64)) > 0)
    off[nz[0] + 1] += 8  # steal one point from the next tile's run
    off.tofile(pack / "points" / "tile_index.u64")
    _rehash(pack)
    res = mp.validate_pack(pack, full=True)
    assert not res["ok"]
    assert "tile_index_delimits_tiles" in {c["check"] for c in res["failed"]}


def test_validator_catches_a_corrupted_density_plane(built, tmp_path):
    pack = _clone(built["pack"], tmp_path)
    z = built["manifest"]["tiles"]["max_zoom"]
    idx = json.loads((pack / "density" / f"z{z}" / "index.json").read_text())
    tk, ent = next(iter(idx["tiles"].items()))
    tx, ty = (int(v) for v in tk.split("_"))
    p = pack / "density" / f"z{z}" / f"{tx}_{ty}.{ent['corpora'][0]}.u32"
    arr = np.frombuffer(p.read_bytes(), dtype="<u4").copy()
    arr[int(np.flatnonzero(arr > 0)[0])] += 1
    p.write_bytes(arr.tobytes())
    _rehash(pack)
    res = mp.validate_pack(pack, full=True)
    assert not res["ok"]


def test_validator_catches_a_tampered_file_hash(built, tmp_path):
    pack = _clone(built["pack"], tmp_path)
    lod = pack / "points" / "lod.bin"
    b = bytearray(lod.read_bytes())
    b[0] ^= 0xFF
    lod.write_bytes(bytes(b))
    res = mp.validate_pack(pack, full=True)
    assert not res["ok"]
    assert "inventory_sha256" in {c["check"] for c in res["failed"]}


def test_validator_catches_a_missing_file(built, tmp_path):
    pack = _clone(built["pack"], tmp_path)
    (pack / "points" / "lod.bin").unlink()
    res = mp.validate_pack(pack, full=False)
    assert not res["ok"]


def test_validator_catches_mismatched_corpus_counts(built, tmp_path):
    pack = _clone(built["pack"], tmp_path)
    man = json.loads((pack / "manifest.json").read_text())
    man["corpus_counts"]["0"] = int(man["corpus_counts"]["0"]) + 1
    (pack / "manifest.json").write_text(json.dumps(man, indent=1))
    res = mp.validate_pack(pack, full=False)
    assert not res["ok"]
    assert "corpus_bits_roundtrip" in {c["check"] for c in res["failed"]}


def test_skip_text_produces_a_pack_without_snippets(fixture, tmp_path, monkeypatch_module):
    monkeypatch_module.setattr(mp, "EMB_ROOT", fixture["emb_root"])
    monkeypatch_module.setattr(mp, "CHUNK_ROOT", fixture["chunk_root"])
    out = tmp_path / "packs"
    man = mp.build_pack(fixture["coords"], fixture["sub_dir"], "no-text",
                        out_root=out, sidecar_root=tmp_path / "sc",
                        skip_text=True, verbose=False)
    assert man["text"]["text_available"] is False
    assert not (out / "no-text" / "bins" / "snippets_z0.json").exists()
    assert mp.validate_pack(out / "no-text", full=True)["ok"]
