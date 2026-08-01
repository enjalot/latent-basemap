"""Unit tests for the map viewer orchestrator (component D).

The sibling tile/metric builders (components A & C) are monkeypatched with
fakes that emit the frozen on-disk formats, so these tests exercise D's own
logic: layer planning, manifest schema, template instantiation, thumbnails,
idempotency, the round-page splice, and the registry index card grid.
"""
from __future__ import annotations

import json
import struct
import sys
import types
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

GRID_MAGIC = 0x42494E31


# --------------------------------------------------------------- fakes ------

def _write_grid_bin(path: Path, level: int, cells: dict[int, int]) -> None:
    idx = sorted(cells)
    body = struct.pack("<4I", GRID_MAGIC, level, len(idx), 0)
    body += b"".join(struct.pack("<I", c) for c in idx)
    body += b"".join(struct.pack("<I", cells[c]) for c in idx)
    path.write_bytes(body)


def _fake_map_tiles():
    """Fakes matching the REAL component-A signatures the orchestrator calls."""
    import numpy as np
    mod = types.ModuleType("map_tiles")

    class MapSource:
        def __init__(self, coords_dir, cache_dir=None):
            self.coords_dir = Path(coords_dir)
            self.cache_dir = cache_dir
            self.nrows = 1000

        def extent(self):
            return (-5.0, -4.0, 6.0, 7.0)

    def subset_ranges(map_kind):
        if map_kind == "jina-25m":
            return {"eng": (0, 500), "lang-pol": (500, 1000)}
        return {"fineweb": (0, 500), "redpajama": (500, 1000)}

    def jina_subset_ranges():
        return {"eng": {"label": "English", "dataset": "fineweb"},
                "lang-pol": {"label": "Polish", "dataset": "fineweb2"}}

    def bin_all_levels(source, levels, extent, row_filter=None):
        out = {}
        for L in levels:
            # A few populated cells; enough for a real thumbnail render.
            cells = {0: 3, L + 1: 12, 2 * L + 2: 40, L * L - 1: 7}
            idx = np.array(sorted(cells), dtype=np.uint32)
            cnt = np.array([cells[c] for c in sorted(cells)], dtype=np.uint32)
            out[L] = (idx, cnt)
        return out

    def write_grid(path, level, idx, cnt):
        idx = list(int(x) for x in idx)
        cnt = list(int(x) for x in cnt)
        body = struct.pack("<4I", GRID_MAGIC, int(level), len(idx), 0)
        body += b"".join(struct.pack("<I", c) for c in idx)
        body += b"".join(struct.pack("<I", c) for c in cnt)
        Path(path).write_bytes(body)
        return path

    def sample_bins(source, level, extent, k=3, row_filter=None, rng_seed=0):
        return {0: [1, 2, 3]}

    def write_samples(out_dir, layer, samples_by_cell, sample_level, super_tile,
                      map_kind, cache_dir=None):
        p = Path(out_dir) / f"samples-{layer}-0_0.json"
        p.write_text(json.dumps({"cells": {"0": [{"t": "hi", "g": "eng", "r": 1}]}}))
        return [str(p)]

    def write_points(path, xy):
        Path(path).write_bytes(struct.pack("<2I", 0x50545331, len(xy)))
        return path

    mod.MapSource = MapSource
    mod.subset_ranges = subset_ranges
    mod.jina_subset_ranges = jina_subset_ranges
    mod.bin_all_levels = bin_all_levels
    mod.write_grid = write_grid
    mod.sample_bins = sample_bins
    mod.write_samples = write_samples
    mod.write_points = write_points
    return mod


def _fake_metrics():
    mod = types.ModuleType("map_metrics_extract")

    def _emit(data_dir):
        data_dir = Path(data_dir)
        (data_dir / "metrics-anchors.bin").write_bytes(struct.pack("<2I", 0x414E4331, 0))
        (data_dir / "metrics-queries.json").write_text(json.dumps({"probes": []}))
        return {"anchors": {"file": "metrics-anchors.bin", "count": 0,
                            "summary": {"ffr": 0.6386}},
                "probes": [{"key": "pol_Latn", "recall50": 0.2278}]}

    def build_r0108_metrics(core_panel_npz, ood_npz_paths, out_dir, *,
                            extent=None, labels=None, texts_resolver=None):
        return _emit(out_dir)

    def build_r0102_metrics(reference_npz, coords_dir, density_v2_npz,
                            ood_npz_paths, out_dir, *, extent=None, **kw):
        return _emit(out_dir)

    mod.build_r0108_metrics = build_r0108_metrics
    mod.build_r0102_metrics = build_r0102_metrics
    return mod


# -------------------------------------------------------------- fixtures ----

@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setitem(sys.modules, "map_tiles", _fake_map_tiles())
    monkeypatch.setitem(sys.modules, "map_metrics_extract", _fake_metrics())
    yield


@pytest.fixture
def registry(tmp_path):
    coords = tmp_path / "coords"
    (coords / "chunk-00000").mkdir(parents=True)
    # Metric-source artifact the orchestrator looks up beside the coords dir.
    core = coords.parent / "core-geometry" / "core-panel-arrays.npz"
    core.parent.mkdir(parents=True, exist_ok=True)
    core.write_bytes(b"stub")
    entry = {
        "map_id": "round-0108-r0107-diverse-jina-25m-seed42",
        "round_id": "0108",
        "kind": "round-map",
        "map_label": "25M diverse-Jina atlas — seed 42",
        "date": "2026-07-20T00:00:00+00:00",
        "evidence_status": "review:accepted",
        "n_rows": 25_000_000,
        "scientific_rows": 24_948_663,
        "dims": [768, 2],
        "training_round": "0107",
        "coordinates": {"dir": f"gsv:{coords}", "chunks": 1,
                        "receipt_sha256": "sha-abc-123"},
        "panel": {"ffr": 0.6386, "density": 0.72, "purity_k1024": None,
                  "decision_checks_all_pass": True,
                  "formula_version": "density-v2-jina-calibrated"},
    }
    return {"schema": "basemap-map-registry-v2", "maps": [entry],
            "counts": {}, "generated_utc": "2026-07-20T00:00:00+00:00"}


# ---------------------------------------------------------------- tests -----

def test_manifest_schema_and_files(patched, registry, tmp_path):
    import map_viewer
    site = tmp_path / "site"
    built = map_viewer.build_map_viewers(registry, site)

    assert len(built) == 1
    b = built[0]
    vdir = site / "viewer" / "round-0108-r0107-diverse-jina-25m-seed42"
    manifest = json.loads((vdir / "data" / "manifest.json").read_text())

    assert manifest["schema"] == "basemap-viewer-manifest-v1"
    for key in ("map_id", "round_id", "title", "rows_total", "rows_note",
                "extent", "levels", "sample_level", "super_tile", "layers",
                "metrics", "provenance", "coordinates_receipt_sha"):
        assert key in manifest, f"missing manifest key {key}"

    assert manifest["rows_total"] == 24_948_663
    assert manifest["extent"] == [-5.0, -4.0, 6.0, 7.0]
    assert manifest["sample_level"] == 256
    assert manifest["coordinates_receipt_sha"] == "sha-abc-123"
    assert manifest["provenance"]["training_round"] == "0107"
    assert manifest["provenance"]["eval_round"] == "0108"
    assert manifest["metrics"]["anchors"]["summary"]["ffr"] == 0.6386

    layer_keys = {layer["key"] for layer in manifest["layers"]}
    assert "all" in layer_keys
    assert "eng" in layer_keys and "lang-pol" in layer_keys
    all_layer = next(l for l in manifest["layers"] if l["key"] == "all")
    assert all_layer["kind"] == "grid"
    # Layer rows = summed binned counts (equals nrows for the full layer on
    # real data; the fake grid sums to 62).
    assert all_layer["rows"] == 62

    # Grid bins + samples + metric packets on disk.
    assert (vdir / "data" / "grid-all-256.bin").is_file()
    assert (vdir / "data" / "samples-all-0_0.json").is_file()
    assert (vdir / "data" / "metrics-anchors.bin").is_file()
    assert (vdir / "data" / "metrics-queries.json").is_file()

    # Thumbnail rendered by PIL from the level-256 base grid.
    thumb = vdir / "thumb.png"
    assert thumb.is_file() and thumb.stat().st_size > 0

    # Template instantiated (fallback), config token replaced.
    page = (vdir / "index.html").read_text()
    assert "__VIEWER_CONFIG__" not in page
    assert "window.VIEWER_CONFIG" in page
    assert '"manifest": "data/manifest.json"' in page or '"manifest":"data/manifest.json"' in page


def test_round_page_splice(patched, registry, tmp_path):
    import map_viewer
    site = tmp_path / "site"
    round_dir = site / "round-0108"
    round_dir.mkdir(parents=True)
    (round_dir / "index.html").write_text(
        '<!doctype html><h1>Round 0108</h1><p>existing body</p>')

    map_viewer.build_map_viewers(registry, site)
    body = (round_dir / "index.html").read_text()

    assert "existing body" in body  # original content preserved
    assert "<!-- viewer:start -->" in body and "<!-- viewer:end -->" in body
    assert "Interactive viewer" in body
    # Primary link is the React app; legacy vanilla viewer stays reachable.
    assert "../app/index.html#/map/round-0108-r0107-diverse-jina-25m-seed42" in body
    assert "../viewer/round-0108-r0107-diverse-jina-25m-seed42/index.html" in body
    assert "legacy viewer" in body


def test_round_page_splice_idempotent(patched, registry, tmp_path):
    import map_viewer
    site = tmp_path / "site"
    round_dir = site / "round-0108"
    round_dir.mkdir(parents=True)
    (round_dir / "index.html").write_text('<h1>Round 0108</h1>')

    map_viewer.build_map_viewers(registry, site)
    map_viewer.build_map_viewers(registry, site, force=True)
    body = (round_dir / "index.html").read_text()

    # Marker block appears exactly once even after a rebuild.
    assert body.count("<!-- viewer:start -->") == 1
    assert body.count("<!-- viewer:end -->") == 1


def test_idempotency_skips_rebuild(patched, registry, tmp_path):
    import map_viewer
    site = tmp_path / "site"
    map_viewer.build_map_viewers(registry, site)
    vdir = site / "viewer" / "round-0108-r0107-diverse-jina-25m-seed42"
    manifest_path = vdir / "data" / "manifest.json"
    first_gen = json.loads(manifest_path.read_text())["generated_utc"]

    # Second run: unchanged receipt sha -> manifest untouched.
    built = map_viewer.build_map_viewers(registry, site)
    assert len(built) == 1
    assert json.loads(manifest_path.read_text())["generated_utc"] == first_gen

    # force=True rewrites the manifest.
    map_viewer.build_map_viewers(registry, site, force=True)
    assert json.loads(manifest_path.read_text())["generated_utc"] != first_gen


def test_only_filter(patched, registry, tmp_path):
    import map_viewer
    site = tmp_path / "site"
    # A round_id outside the default allowlist and not in `only` builds nothing.
    assert map_viewer.build_map_viewers(registry, site, only=["9999"]) == []
    # Selecting by round_id builds it.
    assert len(map_viewer.build_map_viewers(registry, site, only=["0108"])) == 1


def test_index_card_injection(patched, registry, tmp_path):
    import map_viewer
    import map_registry
    site = tmp_path / "site"
    site.mkdir()
    index = site / "index.html"
    index.write_text(
        '<h1>registry</h1><!-- viewer-cards:start --><!-- viewer-cards:end -->'
        '<h2>Round maps</h2>')

    built = map_viewer.build_map_viewers(registry, site)
    map_registry._inject_viewer_cards(site, registry, built)
    body = index.read_text()

    assert "Interactive maps" in body
    assert 'class="cardgrid"' in body
    assert 'class="mapcard"' in body
    assert "open viewer" in body
    # Primary button targets the React app; legacy viewer is a secondary link.
    assert "app/index.html#/map/round-0108-r0107-diverse-jina-25m-seed42" in body
    assert "legacy viewer" in body
    assert "viewer/round-0108-r0107-diverse-jina-25m-seed42/index.html" in body
    # Metric chips: FFR value + density_v2 pass badge (0.72 >= 0.60 floor).
    assert "FFR 0.6386" in body
    assert "density_v2 0.7200 ✓" in body
    # Evidence badge and thumbnail wired to the viewer dir.
    assert "review:accepted" in body
    assert "viewer/round-0108-r0107-diverse-jina-25m-seed42/thumb.png" in body
    # Existing sections preserved.
    assert "<h2>Round maps</h2>" in body


def test_density_fail_chip(patched, tmp_path):
    import map_viewer
    import map_registry
    # A map whose density is below the floor gets a fail chip.
    coords = tmp_path / "c2"
    (coords / "chunk-00000").mkdir(parents=True)
    ref = coords.parent / "high-d-reference-150m" / "reference.npz"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_bytes(b"stub")
    entry = {
        "map_id": "round-0102-150m", "round_id": "0102", "kind": "round-map",
        "map_label": "150M MiniLM", "date": "2026-07-10T00:00:00+00:00",
        "evidence_status": "review:accepted", "n_rows": 150_000_000,
        "dims": [384, 2], "training_round": "0034",
        "coordinates": {"dir": f"gsv:{coords}", "receipt_sha256": "sha-2"},
        "panel": {"ffr": 0.5, "density": 0.41, "decision_checks_all_pass": False},
    }
    reg = {"maps": [entry]}
    site = tmp_path / "site2"
    built = map_viewer.build_map_viewers(reg, site)
    assert len(built) == 1
    # minilm map_kind rows_note mentions block corpora.
    manifest = json.loads(
        (site / "viewer" / "round-0102-150m" / "data" / "manifest.json").read_text())
    assert manifest["map_kind"] == "minilm-150m"
    assert "fineweb" in manifest["rows_note"]

    card = map_registry._viewer_card(built[0], entry)
    assert "density_v2 0.4100 ✗" in card
    assert 'chip bad' in card


# ------------------------------------------------------- maps-index tests ---

def test_maps_index_schema(patched, registry, tmp_path):
    import map_viewer
    site = tmp_path / "site"
    map_viewer.build_map_viewers(registry, site)

    index = json.loads((site / "maps-index.json").read_text())
    assert index["schema"] == "basemap-maps-index-v1"
    assert "generated_utc" in index
    assert isinstance(index["maps"], list) and len(index["maps"]) == 1
    m = index["maps"][0]
    for key in ("map_id", "title", "kind", "round_id", "rows_total", "rows_note",
                "data", "thumbnail", "evidence_status", "metrics", "probes"):
        assert key in m, f"maps-index entry missing {key}"
    # The 25M jina map is surfaced as an atlas kind.
    assert m["kind"] == "atlas"
    assert m["map_id"] == "round-0108-r0107-diverse-jina-25m-seed42"
    assert m["rows_total"] == 24_948_663
    assert m["data"] == "viewer/round-0108-r0107-diverse-jina-25m-seed42/data/"
    assert m["thumbnail"] == "viewer/round-0108-r0107-diverse-jina-25m-seed42/thumb.png"
    assert m["metrics"]["ffr"] == 0.6386
    assert m["metrics"]["density_v2"] == 0.72
    # Probe list carried from the metrics fragment.
    assert m["probes"] and m["probes"][0]["key"] == "pol_Latn"


# ------------------------------------------------ projection-map viewers ----

def _write_projection_npz(path: Path, *, corpus=100, queries=5, truth=True):
    import numpy as np
    rng = np.random.RandomState(0)
    arrays = {
        "probe_corpus_coords": rng.uniform(-3, 3, (corpus, 2)).astype(np.float32),
        "probe_query_coords": rng.uniform(-3, 3, (queries, 2)).astype(np.float32),
        "probe_corpus_ids": np.arange(corpus),
        "probe_query_ids": np.arange(queries),
    }
    if truth:
        arrays["exact_high_d_top10"] = rng.randint(0, corpus, (queries, 10))
        arrays["low_d_top50"] = rng.randint(0, corpus, (queries, 50))
    np.savez(path, **arrays)


@pytest.fixture
def projection_registry(tmp_path):
    import numpy as np
    # Base map coordinate chunks + a semantic-render sample-id sidecar.
    base = tmp_path / "base_coords"
    chunk = base / "chunk-00000"
    chunk.mkdir(parents=True)
    np.save(chunk / "coordinates.npy",
            np.random.RandomState(1).uniform(-4, 4, (120, 2)).astype(np.float32))
    sample_ids = tmp_path / "sample-semantic-ids.npy"
    np.save(sample_ids, np.arange(120, dtype=np.int64))

    npz = tmp_path / "pol_Latn-coordinates.npz"
    _write_projection_npz(npz, corpus=100, queries=5, truth=True)

    entry = {
        "map_id": "round-0108-r0107-diverse-jina-25m-seed42-pol-latn-projection",
        "round_id": "0108",
        "kind": "projection-map",
        "evidence_status": "review:accepted",
        "base_map": "25M diverse-Jina atlas — seed 42",
        "base_coordinates": {"dir": f"gsv:{base}"},
        "base_sample_ids": {"path": f"gsv:{sample_ids}"},
        "projection": {
            "probe": "pol_Latn",
            "display_name": "Held-out Polish",
            "coordinates": f"gsv:{npz}",
            "coordinate_signature": {"sha256": "proj-sha-1"},
            "corpus_rows": 100,
            "query_rows": 5,
            "ffr": 0.19,
            "control_ffr": 0.42,
            "retention": 0.88,
            "verdict": "projected",
        },
    }
    return {"schema": "basemap-map-registry-v2", "maps": [entry],
            "counts": {}, "generated_utc": "2026-08-01T00:00:00+00:00"}


def test_projection_manifest_build(projection_registry, tmp_path):
    # Real map_tiles + map_metrics_extract + projection_gallery (no monkeypatch).
    import map_viewer
    site = tmp_path / "site"
    built = map_viewer.build_map_viewers(projection_registry, site)
    # Round-map build list is empty; the projection viewer is built separately.
    assert built == []

    map_id = "round-0108-r0107-diverse-jina-25m-seed42-pol-latn-projection"
    data = site / "viewer" / map_id / "data"
    manifest = json.loads((data / "manifest.json").read_text())

    assert manifest["schema"] == "basemap-viewer-manifest-v1"
    assert manifest["map_kind"] == "projection"
    assert manifest["kind"] == "projection-map"
    assert manifest["coordinates_receipt_sha"] == "proj-sha-1"
    assert len(manifest["extent"]) == 4

    layers = {l["key"]: l for l in manifest["layers"]}
    assert set(layers) == {"base-context", "corpus", "queries"}
    assert layers["base-context"]["label"] == "training-map context"
    assert layers["base-context"]["kind"] == "points"
    # sampled_of honesty: 30k-recipe sample of the base map's rows.
    assert "sampled_of" in layers["base-context"]
    assert layers["base-context"]["sampled_of"] == 120
    assert layers["base-context"]["rows"] <= 120
    # accent passthrough survives into the manifest layer schema.
    assert layers["corpus"]["accent"] == "a1"
    assert layers["base-context"]["accent"] == "a2"

    # Point binaries on disk.
    assert (data / "points-base-context.bin").is_file()
    assert (data / "points-corpus.bin").is_file()
    assert (data / "points-queries.bin").is_file()

    # Embedded truth -> metrics-queries.json emitted with a probe packet.
    assert (data / "metrics-queries.json").is_file()
    packet = json.loads((data / "metrics-queries.json").read_text())
    assert packet["probes"] and packet["probes"][0]["key"] == "pol_Latn"
    assert manifest["metrics"]["probes"][0]["queries"] == 5

    # maps-index.json includes the projection entry.
    index = json.loads((site / "maps-index.json").read_text())
    proj_entries = [m for m in index["maps"] if m["kind"] == "projection-map"]
    assert len(proj_entries) == 1
    assert proj_entries[0]["map_id"] == map_id
    assert proj_entries[0]["data"] == f"viewer/{map_id}/data/"


def test_projection_manifest_no_truth_skips_queries(projection_registry, tmp_path):
    import map_viewer
    # Rewrite the npz without embedded truth: metrics-queries.json must be absent.
    npz = Path(projection_registry["maps"][0]["projection"]["coordinates"]
               .removeprefix("gsv:"))
    _write_projection_npz(npz, corpus=100, queries=5, truth=False)

    site = tmp_path / "site"
    map_viewer.build_map_viewers(projection_registry, site)
    map_id = projection_registry["maps"][0]["map_id"]
    data = site / "viewer" / map_id / "data"

    assert not (data / "metrics-queries.json").is_file()
    manifest = json.loads((data / "manifest.json").read_text())
    assert manifest["metrics"] == {}
    assert any("query metrics omitted" in s for s in manifest["skipped"])


def test_projection_idempotency(projection_registry, tmp_path):
    import map_viewer
    site = tmp_path / "site"
    map_viewer.build_map_viewers(projection_registry, site)
    map_id = projection_registry["maps"][0]["map_id"]
    manifest_path = site / "viewer" / map_id / "data" / "manifest.json"
    first = json.loads(manifest_path.read_text())["generated_utc"]

    # Unchanged coordinate signature -> manifest untouched.
    map_viewer.build_map_viewers(projection_registry, site)
    assert json.loads(manifest_path.read_text())["generated_utc"] == first

    # force rewrites.
    map_viewer.build_map_viewers(projection_registry, site, force=True)
    assert json.loads(manifest_path.read_text())["generated_utc"] != first
