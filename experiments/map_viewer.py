#!/usr/bin/env python3
"""Per-map interactive viewer builder + registry publish hook (component D).

Post-hoc map-registry tooling (see the map-viz design doc). Never a launch-path
dependency; must survive missing sibling modules and partial artifacts. Each
per-map data build is delegated to two sibling modules built in parallel:

  * ``map_tiles``            — binning / sampling / point writers (component A)
  * ``map_metrics_extract``  — anchor + OOD query packets (component C)

D owns the orchestration, template instantiation, thumbnails, idempotency, and
the ``map_registry.publish`` splice + index card grid.

--------------------------------------------------------------------------
Real library interfaces this module calls (reconciled by the integrator to the
signatures components A and C actually shipped):

  map_tiles.MapSource(coords_dir, cache_dir=None)
      .extent() -> (xmin, ymin, xmax, ymax)
      .nrows    -> int
  map_tiles.subset_ranges(map_kind) -> {key: (lo, hi)}   (coord-space slices)
  map_tiles.jina_subset_ranges() -> {key: {"label","dataset",...}}
  map_tiles.bin_all_levels(source, levels, extent, row_filter=None)
      -> {level: (idx u32[], cnt u32[])}
  map_tiles.write_grid(path, level, idx, cnt) -> path       (grid-<layer>-<L>.bin)
  map_tiles.sample_bins(source, level, extent, k=3, row_filter=None,
                        rng_seed=0) -> {cell: [coord_row, ...]}
  map_tiles.write_samples(out_dir, layer, samples_by_cell, sample_level,
                          super_tile, map_kind, cache_dir=None) -> [paths]
  map_tiles.write_points(path, xy) -> path                  (points-<layer>.bin)

  map_metrics_extract.build_r0108_metrics(core_panel_npz, ood_npz_paths,
                          out_dir, *, extent=None, labels=None,
                          texts_resolver=None) -> dict
  map_metrics_extract.build_r0102_metrics(reference_npz, coords_dir,
                          density_v2_npz, ood_npz_paths, out_dir, *,
                          extent=None, ...) -> dict
      each writes metrics-anchors.bin + metrics-queries.json and returns the
      manifest "metrics" block {"anchors": {...}, "probes": [...]}.
--------------------------------------------------------------------------
"""
from __future__ import annotations

import html
import json
import shutil
import struct
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_SCHEMA = "basemap-viewer-manifest-v1"
MAPS_INDEX_SCHEMA = "basemap-maps-index-v1"

# Label required by the addendum for the 30k deterministic base sample.
BASE_CONTEXT_LABEL = "training-map context"

# Optional per-layer passthrough fields (addendum: honesty + accent hints).
LAYER_PASSTHROUGH = ("sampled_of", "accent")

# Full-map grid resolution ladder. write_grid returns the levels it actually
# emitted (it may drop 1024 for a subset layer whose bin would exceed 2 MB).
LEVELS = [64, 128, 256, 512, 1024]
# Addendum v3 deep-zoom levels — base "all" layer ONLY (subsets stay <=1024).
# Any base-layer level whose whole sparse file would exceed 2.5 MB is written
# as 4x4 spatial tiles (map_tiles.write_grid_auto) and declared "tiled_levels".
FINE_LEVELS = [2048, 4096]
SAMPLE_LEVEL = 256
SUPER_TILE = 16

GRID_MAGIC = 0x42494E31  # "BIN1"

# Static server has no HTTP Range: every fetchable file must stay small. A
# grid bin is 16 + 8*ncells bytes; drop any level whose bin would exceed this.
MAX_GRID_BYTES = 2_400_000

# Human labels for the minilm-150m coord-space corpora blocks (subset_ranges).
MINILM_LABELS = {
    "fineweb": ("FineWeb-Edu", "corpus"),
    "redpajama": ("RedPajama-V2", "corpus"),
    "pile": ("The Pile", "corpus"),
}

# Newest accepted map per family (design doc). ``only`` overrides this.
DEFAULT_ALLOWLIST = {"0102", "0108", "0132"}

DENSITY_FLOOR = 0.60  # density_v2 pass threshold (registry density_at_least_0_60)

ASSETS_REL = "../../assets"  # viewer page lives at viewer/<map_id>/index.html


# --------------------------------------------------------------- helpers ----

def _strip(path: str | None) -> Path | None:
    if not path:
        return None
    return Path(path.removeprefix("gsv:"))


def _map_kind(entry: dict) -> str:
    dims = entry.get("dims") or []
    if dims and dims[0] == 768:
        return "jina-25m"
    return "minilm-150m"


def _rows_total(entry: dict) -> int | None:
    return entry.get("scientific_rows") or entry.get("n_rows")


def _rows_note(entry: dict, map_kind: str) -> str:
    if entry.get("scientific_rows"):
        return "all retained representatives, exact count — not a sample"
    if map_kind == "minilm-150m":
        return ("identity-order coordinates; block corpora "
                "[0,50M) fineweb · [50M,100M) redpajama · [100M,150M) pile")
    return "exact row count — not a sample"


def _title(entry: dict) -> str:
    return str(entry.get("map_label") or entry.get("map_id"))


def _viewer_rel(entry: dict) -> str:
    """Per-map viewer directory, keyed by map_id to avoid same-round collisions."""
    return f"viewer/{entry['map_id']}"


def _should_build(entry: dict, only) -> bool:
    if entry.get("kind") != "round-map":
        return False
    coords = _strip((entry.get("coordinates") or {}).get("dir"))
    if not coords:
        return False
    if only:
        wanted = set(only)
        return entry.get("map_id") in wanted or entry.get("round_id") in wanted
    return entry.get("round_id") in DEFAULT_ALLOWLIST


def _import_siblings():
    try:
        import map_tiles  # type: ignore
        import map_metrics_extract  # type: ignore
    except ImportError:  # package-qualified execution
        from experiments import map_tiles, map_metrics_extract  # type: ignore
    return map_tiles, map_metrics_extract


def _import_projection_gallery():
    """Reuse projection_gallery's sampling/id recipe (import only — never edit)."""
    try:
        import projection_gallery  # type: ignore
    except ImportError:  # package-qualified execution
        from experiments import projection_gallery  # type: ignore
    return projection_gallery


def _app_href(map_id: str, prefix: str = "") -> str:
    """Hash-routed React app URL for one map (relative to the given prefix)."""
    return f"{prefix}app/index.html#/map/{map_id}"


# ------------------------------------------------------------ thumbnails ----

def _write_thumbnail(grid_path: Path, out_png: Path, size: int = 256) -> bool:
    """Render a log-scaled single-hue PNG thumbnail from a grid-<layer>-L.bin.

    Reuses the frozen grid binary format so it stays consistent with the tile
    builder without a second code path. Empty cells keep the light card
    background; density interpolates one blue hue light -> dark, log-compressed.
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return False
    try:
        raw = grid_path.read_bytes()
    except OSError:
        return False
    if len(raw) < 16:
        return False
    magic, level, ncells, _reserved = struct.unpack("<4I", raw[:16])
    if magic != GRID_MAGIC or level <= 0:
        return False
    off = 16
    need = 16 + 8 * ncells
    if len(raw) < need:
        return False
    idx = np.frombuffer(raw, dtype="<u4", count=ncells, offset=off)
    cnt = np.frombuffer(raw, dtype="<u4", count=ncells, offset=off + 4 * ncells)
    grid = np.zeros(level * level, dtype=np.float64)
    if ncells:
        valid = idx < grid.size
        grid[idx[valid]] = cnt[valid]
    grid = grid.reshape(level, level)  # [cy, cx], data-space y
    grid = np.flipud(grid)             # screen orientation: top = max y
    peak = float(grid.max())
    if peak <= 0:
        return False
    t = np.log1p(grid) / np.log1p(peak)          # [0,1] log-compressed
    bg = np.array([248.0, 249.0, 252.0])         # light card background
    ink = np.array([26.0, 58.0, 138.0])          # single blue hue, dark end
    rgb = bg[None, None, :] * (1.0 - t[..., None]) + ink[None, None, :] * t[..., None]
    rgb[grid == 0] = bg
    img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    if level != size:
        img = img.resize((size, size), Image.NEAREST)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    return True


# --------------------------------------------------------------- assets -----

def _copy_assets(site_dir: Path) -> None:
    """Copy shared viewer assets (viewer.js/css) into <site>/assets/ once."""
    src = Path(__file__).resolve().parent / "viewer_assets"
    if not src.is_dir():
        return
    dst = site_dir / "assets"
    dst.mkdir(parents=True, exist_ok=True)
    for item in sorted(src.iterdir()):
        if not item.is_file() or "template" in item.name:
            continue
        target = dst / item.name
        if not target.exists() or target.stat().st_size != item.stat().st_size:
            shutil.copy2(item, target)


def _load_template() -> str:
    src = Path(__file__).resolve().parent / "viewer_assets" / "viewer_template.html"
    try:
        return src.read_text()
    except OSError:
        return _FALLBACK_TEMPLATE


def _instantiate(template: str, config: dict) -> str:
    blob = json.dumps(config, ensure_ascii=False)
    if "__VIEWER_CONFIG__" in template:
        return template.replace("__VIEWER_CONFIG__", blob)
    # Template author omitted the token; append a config script defensively so
    # a viewer that reads window.VIEWER_CONFIG still initializes.
    return template + f"\n<script>window.VIEWER_CONFIG={blob};</script>\n"


# --------------------------------------------------------- layer planning ---

def _apply_passthrough(layer: dict, plan: dict) -> None:
    """Copy optional honesty/accent fields (sampled_of, accent) into a manifest
    layer when the layer plan carries them. Grid layers without them are
    unchanged (addendum layer-schema addition)."""
    for field in LAYER_PASSTHROUGH:
        val = plan.get(field)
        if val is not None:
            layer[field] = val


def _subset_label(map_kind: str, key: str, mt) -> tuple[str, str | None]:
    """Human (label, group) for a subset key, best-effort."""
    if map_kind == "minilm-150m":
        return MINILM_LABELS.get(key, (key, "corpus"))
    try:
        meta = mt.jina_subset_ranges().get(key, {})
        return meta.get("label", key), meta.get("dataset")
    except Exception:
        return key, None


def _plan_layers(entry: dict, map_kind: str, mt) -> list[dict]:
    layers = [{
        "key": "all", "label": "Full corpus", "kind": "grid",
        "row_filter": None, "group": None, "samples": True,
    }]
    try:
        ranges = mt.subset_ranges(map_kind) or {}
    except Exception as exc:  # subset ranges are best-effort
        print(f"  subset_ranges({map_kind}) failed: {exc}")
        ranges = {}
    for key, span in ranges.items():
        try:
            lo, hi = int(span[0]), int(span[1])
        except (TypeError, ValueError, IndexError):
            continue
        label, group = _subset_label(map_kind, key, mt)
        layers.append({
            "key": key, "label": label, "kind": "grid",
            "row_filter": slice(lo, hi), "group": group, "samples": False,
        })
    return layers


def _linked_projections(registry: dict, entry: dict) -> list[dict]:
    """OOD projection-maps whose base map is this round-map's coordinate dir."""
    coords = _strip((entry.get("coordinates") or {}).get("dir"))
    if not coords:
        return []
    out = []
    for m in registry.get("maps", []):
        if m.get("kind") != "projection-map":
            continue
        base = _strip((m.get("base_coordinates") or {}).get("dir"))
        if base is not None and base == coords:
            out.append(m)
    return out


# --------------------------------------------------------- grid / metrics ---

def _emit_grids(mt, source, data: Path, layer_key: str, extent, row_filter,
                skipped=None, base=False):
    """Write the grid files for one layer.

    Subset layers (base=False): plain grid-<layer>-<L>.bin for LEVELS, dropping
    any level that would exceed the static-fetch size cap (unchanged v1 rule).

    Base "all" layer (base=True): LEVELS + FINE_LEVELS, each written plain or
    as 4x4 spatial tiles by map_tiles.write_grid_auto's 2.5 MB rule (nothing is
    dropped). Returns (plain_levels, tiled_levels, rows) where tiled_levels is
    the manifest fragment [{"level": L, "split": 4}, ...].
    """
    levels = LEVELS + FINE_LEVELS if base else LEVELS
    grids = mt.bin_all_levels(source, levels, extent, row_filter=row_filter)
    # Every level sums to the same row total; use the coarsest.
    rows = int(grids[levels[0]][1].sum())
    emitted, tiled = [], []
    for lvl in levels:
        idx, cnt = grids[lvl]
        if base:
            result = mt.write_grid_auto(str(data), layer_key, lvl, idx, cnt)
            if result["tiled"]:
                tiled.append({"level": lvl, "split": result["split"]})
            else:
                emitted.append(lvl)
            continue
        if 16 + 8 * len(idx) > MAX_GRID_BYTES:
            msg = (f"{layer_key}: L{lvl} grid omitted "
                   f"({16 + 8 * len(idx)}B > {MAX_GRID_BYTES}B static-fetch cap)")
            print(f"  {msg}")
            if skipped is not None:
                skipped.append(msg)
            continue
        mt.write_grid(str(data / f"grid-{layer_key}-{lvl}.bin"), lvl, idx, cnt)
        emitted.append(lvl)
    return emitted, tiled, rows


def _metric_artifacts(map_kind: str, coords_dir: Path) -> dict:
    """Locate the metric-source npz files by walking up from the coords dir."""
    art = coords_dir.parent  # .../artifacts
    if map_kind == "jina-25m":
        ood = [p for p in sorted((art / "ood").glob("*-coordinates.npz"))
               if "alignment" not in p.name]
        return {"kind": "r0108",
                "core_panel_npz": art / "core-geometry" / "core-panel-arrays.npz",
                "ood": ood}
    ood = sorted(art.glob("ood-*/**/panel/*-coordinates.npz"))
    return {"kind": "r0102",
            "reference_npz": art / "high-d-reference-150m" / "reference.npz",
            "density_v2_npz": art / "density-v2" / "density-v2-radii.npz",
            "ood": ood}


def _build_metrics(map_kind: str, coords_dir: Path, data: Path, mm, extent,
                   skipped=None) -> dict:
    a = _metric_artifacts(map_kind, coords_dir)
    if a["kind"] == "r0108":
        if not a["core_panel_npz"].is_file():
            msg = f"anchor/query metrics omitted: core panel npz missing ({a['core_panel_npz'].name})"
            print(f"  {msg}")
            if skipped is not None:
                skipped.append(msg)
            return {}
        result = mm.build_r0108_metrics(
            a["core_panel_npz"], a["ood"], data, extent=extent)
    else:
        if not a["reference_npz"].is_file():
            msg = f"anchor/query metrics omitted: high-d reference npz missing ({a['reference_npz'].name})"
            print(f"  {msg}")
            if skipped is not None:
                skipped.append(msg)
            return {}
        result = mm.build_r0102_metrics(
            a["reference_npz"], coords_dir, a["density_v2_npz"], a["ood"], data,
            extent=extent)
    return result if isinstance(result, dict) else {}


def _load_probe_corpus_xy(npz_path: Path):
    import numpy as np
    with np.load(npz_path) as z:
        if "probe_corpus_coords" not in z:
            return None
        return np.asarray(z["probe_corpus_coords"], dtype=np.float32)


# ------------------------------------------------------- index descriptors --

def _probe_summaries(manifest: dict) -> list[dict]:
    """Compact per-probe list for maps-index.json (key/label/queries/recall50)."""
    out = []
    for p in ((manifest.get("metrics") or {}).get("probes") or []):
        if isinstance(p, dict):
            out.append({k: p.get(k) for k in ("key", "label", "queries", "recall50")
                        if k in p})
    return out


def _round_descriptor(entry: dict, manifest: dict, viewer_rel: str,
                      thumb_exists: bool) -> dict:
    """Descriptor consumed by both the registry card grid and maps-index.json."""
    map_kind = manifest.get("map_kind")
    kind = "atlas" if map_kind == "jina-25m" else "round-map"
    panel = (manifest.get("provenance") or {}).get("panel") or {}
    thumb_rel = f"{viewer_rel}/thumb.png"
    probes = _probe_summaries(manifest)
    return {
        "map_id": entry["map_id"],
        "round_id": entry.get("round_id"),
        "kind": kind,
        "title": manifest.get("title") or _title(entry),
        "date": entry.get("date"),
        "rows_total": manifest.get("rows_total"),
        "rows_note": manifest.get("rows_note"),
        "evidence_status": entry.get("evidence_status"),
        "data": f"{viewer_rel}/data/",
        "thumbnail": thumb_rel if thumb_exists else None,
        "metrics": {"ffr": panel.get("ffr"), "density_v2": panel.get("density")},
        "probes": probes,
        # Filter tags: the keys of this map's OOD probes (addendum v3).
        "tags": sorted({p["key"] for p in probes if p.get("key")}),
        # Retained for the registry card grid (legacy viewer link + thumb).
        "viewer_rel": f"{viewer_rel}/index.html",
        "thumb_rel": thumb_rel,
    }


def _projection_descriptor(entry: dict, manifest: dict, viewer_rel: str,
                           thumb_exists: bool) -> dict:
    proj = entry.get("projection", {}) or {}
    probes = _probe_summaries(manifest)
    if not probes:
        probes = [{"key": proj.get("probe"), "label": proj.get("display_name")}]
    thumb_rel = f"{viewer_rel}/thumb.png"
    return {
        "map_id": entry["map_id"],
        "round_id": entry.get("round_id"),
        "kind": "projection-map",
        "title": manifest.get("title"),
        "date": entry.get("date"),
        "rows_total": manifest.get("rows_total"),
        "rows_note": manifest.get("rows_note"),
        "evidence_status": entry.get("evidence_status"),
        "data": f"{viewer_rel}/data/",
        "thumbnail": thumb_rel if thumb_exists else None,
        "metrics": {"ffr": proj.get("ffr"), "control_ffr": proj.get("control_ffr")},
        "probes": probes,
        # Filter tag: this projection's probe key (addendum v3).
        "tags": [proj["probe"]] if proj.get("probe") else [],
        "viewer_rel": f"{viewer_rel}/index.html",
        "thumb_rel": thumb_rel,
    }


def write_maps_index(site_dir: Path, round_built: list[dict],
                     projection_built: list[dict]) -> dict:
    """Emit maps-index.json (schema basemap-maps-index-v1) at the site root.

    One entry per map that has viewer data — atlas/round maps and the new
    projection-map manifests. Consumed by the React gallery route.
    """
    maps = []
    for d in list(round_built) + list(projection_built):
        maps.append({
            "map_id": d.get("map_id"),
            "title": d.get("title"),
            "kind": d.get("kind"),
            "round_id": d.get("round_id"),
            "date": d.get("date"),
            "rows_total": d.get("rows_total"),
            "rows_note": d.get("rows_note"),
            "data": d.get("data"),
            "thumbnail": d.get("thumbnail"),
            "evidence_status": d.get("evidence_status"),
            "metrics": d.get("metrics") or {},
            "probes": d.get("probes") or [],
            "tags": d.get("tags") or [],
        })
    index = {
        "schema": MAPS_INDEX_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "maps": maps,
    }
    (site_dir / "maps-index.json").write_text(json.dumps(index, indent=1))
    return index


# --------------------------------------------------------------- build ------

def _build_one(entry: dict, registry: dict, site_dir: Path,
               mt, mm, *, force: bool) -> dict | None:
    map_id = entry["map_id"]
    map_kind = _map_kind(entry)
    coords_dir = _strip((entry.get("coordinates") or {}).get("dir"))
    receipt_sha = (entry.get("coordinates") or {}).get("receipt_sha256")

    viewer_rel = _viewer_rel(entry)
    out = site_dir / viewer_rel
    data = out / "data"
    manifest_path = data / "manifest.json"

    # Idempotency: skip a rebuild when the coordinates receipt is unchanged.
    if not force and manifest_path.is_file():
        try:
            prev = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            prev = {}
        if prev.get("coordinates_receipt_sha") == receipt_sha and receipt_sha:
            return _round_descriptor(entry, prev, viewer_rel,
                                     (out / "thumb.png").is_file())  # already current

    if coords_dir is None or not coords_dir.is_dir():
        print(f"  {map_id}: coordinates dir missing ({coords_dir}); skip")
        return None

    data.mkdir(parents=True, exist_ok=True)
    cache_dir = out / "_cache"

    try:
        source = mt.MapSource(coords_dir, cache_dir=cache_dir)
        extent = list(source.extent())
    except Exception as exc:
        print(f"  {map_id}: MapSource/extent failed: {exc}; skip")
        return None

    # Honest record of anything this build omits; surfaced in the viewer header.
    skipped: list[str] = []

    manifest_layers: list[dict] = []
    for layer in _plan_layers(entry, map_kind, mt):
        is_base = layer["key"] == "all"
        try:
            emitted, tiled, rows = _emit_grids(
                mt, source, data, layer["key"], extent, layer["row_filter"],
                skipped=skipped, base=is_base)
        except Exception as exc:
            print(f"  {map_id}: grids({layer['key']}) failed: {exc}")
            if is_base:
                return None  # base layer is mandatory
            skipped.append(f"subset layer '{layer['key']}' omitted: grid build failed ({exc})")
            continue
        if not emitted and not tiled:
            if is_base:
                return None
            skipped.append(f"subset layer '{layer['key']}' omitted: no grid level within size cap")
            continue
        lyr = {
            "key": layer["key"], "label": layer["label"], "kind": "grid",
            "rows": rows, "levels": emitted,
        }
        if tiled:
            lyr["tiled_levels"] = tiled
        if layer["group"]:
            lyr["group"] = layer["group"]
        _apply_passthrough(lyr, layer)
        manifest_layers.append(lyr)
        if layer["samples"]:
            try:
                samples = mt.sample_bins(source, SAMPLE_LEVEL, extent, k=3,
                                         rng_seed=42, row_filter=layer["row_filter"])
                mt.write_samples(data, layer["key"], samples, SAMPLE_LEVEL,
                                 SUPER_TILE, map_kind, cache_dir=cache_dir)
            except Exception as exc:
                print(f"  {map_id}: samples({layer['key']}) failed: {exc}")
                skipped.append(f"text samples for '{layer['key']}' omitted ({exc})")

    # Linked OOD probe corpora as point layers.
    for proj in _linked_projections(registry, entry):
        npz = _strip(proj.get("projection", {}).get("coordinates"))
        probe = proj.get("projection", {}).get("probe", proj["map_id"])
        key = f"probe-{probe}"
        if npz is None or not npz.is_file():
            skipped.append(f"probe point layer '{key}' omitted: coordinates npz missing")
            continue
        try:
            xy = _load_probe_corpus_xy(npz)
            if xy is None or len(xy) == 0:
                skipped.append(f"probe point layer '{key}' omitted: no probe_corpus_coords")
                continue
            mt.write_points(str(data / f"points-{key}.bin"), xy)
        except Exception as exc:
            print(f"  {map_id}: write_points({key}) failed: {exc}")
            skipped.append(f"probe point layer '{key}' omitted: write failed ({exc})")
            continue
        manifest_layers.append({
            "key": key,
            "label": proj["projection"].get("display_name", key),
            "kind": "points", "rows": int(len(xy)), "group": "held-out",
            "accent": "a1",
        })

    # Metric packets (anchors + OOD queries).
    metrics = {}
    try:
        metrics = _build_metrics(map_kind, coords_dir, data, mm, extent,
                                 skipped=skipped)
    except Exception as exc:
        print(f"  {map_id}: build_metrics failed: {exc}")
        skipped.append(f"anchor/query metrics omitted: build failed ({exc})")

    panel = entry.get("panel", {})
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "map_id": map_id,
        "round_id": entry.get("round_id"),
        "title": _title(entry),
        "rows_total": _rows_total(entry),
        "rows_note": _rows_note(entry, map_kind),
        "map_kind": map_kind,
        "extent": extent,
        "levels": LEVELS,
        "sample_level": SAMPLE_LEVEL,
        "super_tile": SUPER_TILE,
        "layers": manifest_layers,
        "metrics": metrics,
        "skipped": skipped,
        "provenance": {
            "training_round": entry.get("training_round"),
            "eval_round": entry.get("round_id"),
            "evidence_status": entry.get("evidence_status"),
            "panel": {
                "ffr": panel.get("ffr"),
                "density": panel.get("density"),
                "purity_k1024": panel.get("purity_k1024"),
                "decision_checks_all_pass": panel.get("decision_checks_all_pass"),
                "formula_version": panel.get("formula_version"),
            },
        },
        "coordinates_receipt_sha": receipt_sha,
    }
    manifest_path.write_text(json.dumps(manifest, indent=1))

    # Thumbnail from the level-256 base grid.
    thumb_written = _write_thumbnail(data / "grid-all-256.bin", out / "thumb.png")

    # Instantiate the page from B's template.
    config = {
        # Keys consumed by viewer.js (component B's runtime contract).
        "dataDir": "data",
        "back": "../../index.html",
        # Extra metadata retained for provenance / future use (harmless to JS).
        "manifest": "data/manifest.json",
        "assets": ASSETS_REL,
        "map_id": map_id,
        "title": _title(entry),
        "round_id": entry.get("round_id"),
        "registry_href": "../../index.html",
        "round_href": f"../../round-{entry.get('round_id')}/index.html",
    }
    (out / "index.html").write_text(_instantiate(_load_template(), config))
    return _round_descriptor(entry, manifest, viewer_rel, thumb_written)


# ------------------------------------------------- projection-map viewers ---

def _should_build_projection(entry: dict, only) -> bool:
    if entry.get("kind") != "projection-map":
        return False
    if not _strip((entry.get("projection") or {}).get("coordinates")):
        return False
    if only:
        wanted = set(only)
        return entry.get("map_id") in wanted or entry.get("round_id") in wanted
    return True


def _base_total_rows(entry: dict, registry: dict, fallback: int | None) -> int | None:
    """Total row count of the base map a projection was projected through.

    The honest denominator for the 30k ``training-map context`` sample. Matches
    the projection's base coordinate dir to the base round-map's registry entry;
    falls back to the semantic-render sample pool size, then to ``fallback``.
    """
    base = _strip((entry.get("base_coordinates") or {}).get("dir"))
    if base is not None:
        for m in registry.get("maps", []):
            if m.get("kind") != "round-map":
                continue
            cdir = _strip((m.get("coordinates") or {}).get("dir"))
            if cdir is None:
                continue
            if cdir == base or base == cdir or cdir in base.parents or base in cdir.parents:
                rows = _rows_total(m)
                if rows:
                    return int(rows)
    pool = _base_sample_pool(entry)
    if pool:
        return int(pool)
    return int(fallback) if fallback else None


def _base_sample_pool(entry: dict) -> int | None:
    sid = _strip((entry.get("base_sample_ids") or {}).get("path"))
    if sid is None or not sid.is_file():
        return None
    try:
        import numpy as np
        return int(len(np.load(sid, allow_pickle=False, mmap_mode="r")))
    except Exception:
        return None


def _write_points_thumbnail(layers, out_png: Path, extent, size: int = 256) -> bool:
    """Scatter base/corpus/query points to a small PNG (or skip -> False)."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return False
    layers = [(np.asarray(xy, dtype=np.float64), rgb) for xy, rgb in layers if len(xy)]
    if not layers:
        return False
    xmin, ymin, xmax, ymax = extent
    if not (xmax > xmin and ymax > ymin):
        return False
    img = np.empty((size, size, 3), dtype=np.uint8)
    img[:] = np.array([248, 249, 252], dtype=np.uint8)  # light card background
    for xy, rgb in layers:
        m = np.isfinite(xy).all(axis=1)
        xy = xy[m]
        if not len(xy):
            continue
        px = ((xy[:, 0] - xmin) / (xmax - xmin) * (size - 1)).astype(int)
        py = ((xy[:, 1] - ymin) / (ymax - ymin) * (size - 1)).astype(int)
        py = size - 1 - py  # screen orientation: top = max y
        ok = (px >= 0) & (px < size) & (py >= 0) & (py < size)
        img[py[ok], px[ok]] = np.array(rgb, dtype=np.uint8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, mode="RGB").save(out_png)
    return True


def _build_projection_one(entry: dict, registry: dict, site_dir: Path,
                          mt, mm, pg, *, force: bool) -> dict | None:
    """Build viewer/<map_id>/data for one projection-map (same manifest contract)."""
    import numpy as np

    map_id = entry["map_id"]
    proj = entry.get("projection", {}) or {}
    npz = _strip(proj.get("coordinates"))
    viewer_rel = _viewer_rel(entry)
    out = site_dir / viewer_rel
    data = out / "data"
    manifest_path = data / "manifest.json"
    receipt_sha = (proj.get("coordinate_signature") or {}).get("sha256")

    # Idempotency: skip when the immutable coordinate npz signature is unchanged.
    if not force and manifest_path.is_file():
        try:
            prev = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            prev = {}
        if receipt_sha and prev.get("coordinates_receipt_sha") == receipt_sha:
            return _projection_descriptor(entry, prev, viewer_rel,
                                          (out / "thumb.png").is_file())

    if npz is None or not npz.is_file():
        return None
    data.mkdir(parents=True, exist_ok=True)
    skipped: list[str] = []

    with np.load(npz, allow_pickle=False) as archive:
        corpus_xy = np.asarray(archive["probe_corpus_coords"], dtype=np.float32)
        query_xy = np.asarray(archive["probe_query_coords"], dtype=np.float32)
    corpus_total = int(len(corpus_xy))
    corpus_rows = pg._sample_rows(corpus_total, pg.CORPUS_SAMPLE_N,
                                  label=map_id + ":corpus")
    corpus_sample = corpus_xy[corpus_rows]

    # Base-context: the 30k deterministic sample from projection_gallery's recipe.
    try:
        base_ids, base_xy = pg._base_coordinates(entry)
    except Exception as exc:
        base_ids = np.empty((0,), np.int64)
        base_xy = np.empty((0, 2), np.float32)
        skipped.append(f"base-context layer omitted: {exc}")
    base_total = _base_total_rows(entry, registry, len(base_ids))

    layers: list[dict] = []
    if len(base_xy):
        mt.write_points(str(data / "points-base-context.bin"), base_xy)
        base_layer = {
            "key": "base-context", "label": BASE_CONTEXT_LABEL, "kind": "points",
            "rows": int(len(base_xy)), "group": "context", "accent": "a2",
        }
        if base_total:
            base_layer["sampled_of"] = int(base_total)
        layers.append(base_layer)
    elif not skipped:
        skipped.append("base-context layer omitted: base coordinates unavailable")

    display = proj.get("display_name") or "probe"
    mt.write_points(str(data / "points-corpus.bin"), corpus_sample)
    corpus_layer = {
        "key": "corpus", "label": f"{display} corpus", "kind": "points",
        "rows": int(len(corpus_sample)), "group": "probe", "accent": "a1",
    }
    if corpus_total > len(corpus_sample):
        corpus_layer["sampled_of"] = corpus_total
    layers.append(corpus_layer)

    mt.write_points(str(data / "points-queries.bin"), query_xy)
    layers.append({
        "key": "queries", "label": f"{display} held-out queries", "kind": "points",
        "rows": int(len(query_xy)), "group": "probe-queries", "accent": "a2",
    })

    # Extent over the union of all point layers (grid extent has no meaning here).
    parts = [a for a in (base_xy, corpus_sample, query_xy) if len(a)]
    allxy = np.concatenate(parts, axis=0) if parts else np.zeros((1, 2), np.float32)
    finite = allxy[np.isfinite(allxy).all(axis=1)]
    if len(finite):
        xmin, ymin = (float(v) for v in finite.min(axis=0))
        xmax, ymax = (float(v) for v in finite.max(axis=0))
        px = (xmax - xmin or 1.0) * 0.04
        py = (ymax - ymin or 1.0) * 0.04
        extent = [xmin - px, ymin - py, xmax + px, ymax + py]
    else:
        extent = [-1.0, -1.0, 1.0, 1.0]

    # metrics-queries.json only when the npz embeds exact_high_d_top10 + low_d_top50.
    metrics: dict = {}
    try:
        res = mm.build_probe_packet(npz, key=proj.get("probe") or map_id,
                                    label=display)
        if res.skipped:
            skipped.append(f"query metrics omitted: {res.reason}")
        else:
            mm.write_queries_json(data / "metrics-queries.json", [res.packet])
            metrics = {"probes": [{"key": res.key, "label": res.label,
                                   "queries": res.n_queries,
                                   "recall50": res.recall50}]}
    except Exception as exc:
        skipped.append(f"query metrics omitted: build failed ({exc})")

    title = f"{display} on {entry.get('base_map') or 'basemap'}"
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "map_id": map_id,
        "round_id": entry.get("round_id"),
        "title": title,
        "rows_total": proj.get("corpus_rows") or corpus_total,
        "rows_note": ("OOD probe corpus projected through the registered base map; "
                      "browser point layers are deterministic samples — metrics "
                      "come from the full registered panel"),
        "map_kind": "projection",
        "kind": "projection-map",
        "base_map": entry.get("base_map"),
        "extent": extent,
        "levels": [],
        "sample_level": SAMPLE_LEVEL,
        "super_tile": SUPER_TILE,
        "layers": layers,
        "metrics": metrics,
        "skipped": skipped,
        "provenance": {
            "eval_round": entry.get("round_id"),
            "evidence_status": entry.get("evidence_status"),
            "base_map": entry.get("base_map"),
            "probe": proj.get("probe"),
            "ffr": proj.get("ffr"),
            "control_ffr": proj.get("control_ffr"),
            "retention": proj.get("retention"),
            "verdict": proj.get("verdict"),
        },
        "coordinates_receipt_sha": receipt_sha,
    }
    manifest_path.write_text(json.dumps(manifest, indent=1))

    thumb_written = _write_points_thumbnail(
        [(base_xy, (139, 146, 156)), (corpus_sample, (49, 120, 198)),
         (query_xy, (229, 72, 77))],
        out / "thumb.png", extent)

    # A minimal landing page so the viewer dir is not a bare 404; the React app
    # is the primary surface and reads data/ directly.
    config = {
        "dataDir": "data", "manifest": "data/manifest.json", "assets": ASSETS_REL,
        "map_id": map_id, "title": title, "round_id": entry.get("round_id"),
        "registry_href": "../../index.html", "app_href": _app_href(map_id, "../../"),
    }
    (out / "index.html").write_text(_instantiate(_load_template(), config))

    return _projection_descriptor(entry, manifest, viewer_rel, thumb_written)


def build_projection_viewers(registry: dict, site_dir: Path, mt, mm,
                             only=None, force: bool = False) -> list[dict]:
    """Build viewer/<map_id>/data manifests for every registry projection-map.

    Reuses projection_gallery's sampling/id recipe by import (never edits it or
    the legacy projections/ pages). Returns one descriptor per built projection.
    """
    site_dir = Path(site_dir)
    targets = [e for e in registry.get("maps", []) if _should_build_projection(e, only)]
    if not targets:
        return []
    try:
        pg = _import_projection_gallery()
    except Exception as exc:
        print(f"  projection viewers skipped (import failed): {exc}")
        return []

    built: list[dict] = []
    for entry in targets:
        try:
            descriptor = _build_projection_one(entry, registry, site_dir,
                                                mt, mm, pg, force=force)
        except Exception as exc:  # one bad projection must not sink the rest
            print(f"  projection viewer failed for {entry.get('map_id')}: {exc}")
            descriptor = None
        if descriptor:
            built.append(descriptor)
    return built


def build_map_viewers(registry: dict, site_dir: Path,
                      only=None, force: bool = False) -> list[dict]:
    """Build interactive viewers for allowlisted round/atlas maps.

    Returns one descriptor per round/atlas map that has a viewer (built or
    already-current), used by the registry index card grid. Also builds the
    projection-map viewers, splices a ``viewer:start/end`` block into each round
    page, and emits maps-index.json at the site root.
    """
    site_dir = Path(site_dir)
    mt, mm = _import_siblings()
    _copy_assets(site_dir)

    built: list[dict] = []
    for entry in registry.get("maps", []):
        if not _should_build(entry, only):
            continue
        try:
            descriptor = _build_one(entry, registry, site_dir, mt, mm, force=force)
        except Exception as exc:  # one bad map must not sink the rest
            print(f"  viewer build failed for {entry.get('map_id')}: {exc}")
            descriptor = None
        if descriptor:
            built.append(descriptor)

    _splice_round_pages(site_dir, built)

    projection_built = build_projection_viewers(
        registry, site_dir, mt, mm, only=only, force=force)

    write_maps_index(site_dir, built, projection_built)
    return built


def _splice_round_pages(site_dir: Path, built: list[dict]) -> None:
    import re
    by_round: dict[str, list[dict]] = {}
    for item in built:
        by_round.setdefault(item["round_id"], []).append(item)
    for round_id, items in by_round.items():
        page_dir = site_dir / f"round-{round_id}"
        page_dir.mkdir(exist_ok=True)
        links = "".join(
            f'<li><a href="../{html.escape(_app_href(i["map_id"]))}">'
            f'{html.escape(i["title"] or i["map_id"])}</a> — interactive viewer'
            f' · <a href="../{html.escape(i["viewer_rel"])}">legacy viewer</a></li>'
            for i in items
        )
        block = ("<!-- viewer:start -->"
                 '<h2>Interactive viewer</h2><ul>' + links + "</ul>"
                 "<!-- viewer:end -->")
        target = page_dir / "index.html"
        if target.is_file():
            body = target.read_text()
            if "<!-- viewer:start -->" in body:
                body = re.sub(r"<!-- viewer:start -->.*?<!-- viewer:end -->",
                              block, body, flags=re.DOTALL)
            else:
                body += block
            target.write_text(body)
        else:
            target.write_text(
                '<!doctype html><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width, initial-scale=1">'
                f"<title>round {round_id}</title>"
                '<p><a href="../index.html">← all maps</a></p>'
                f"<h1>Round {round_id}</h1>{block}"
            )


# ------------------------------------------------------ fallback template ---

_FALLBACK_TEMPLATE = """<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>basemap viewer</title>
<style>
:root{color-scheme:light dark;--fg:#1a1d21;--bg:#fff;--muted:#667;--line:#e2e5ea;--card:#f6f7f9}
@media (prefers-color-scheme:dark){:root{--fg:#e6e8eb;--bg:#121417;--muted:#9aa1ab;--line:#2a2f36;--card:#1b1f24}}
body{font:15px/1.5 system-ui,sans-serif;color:var(--fg);background:var(--bg);margin:0 auto;max-width:1200px;padding:18px}
a{color:inherit}.muted{color:var(--muted)}
#plot{width:100%;height:min(74vh,780px);display:block;background:#fff;border:1px solid var(--line);border-radius:8px}
@media (prefers-color-scheme:dark){#plot{background:#0d0f12}}
</style>
<script>window.VIEWER_CONFIG=__VIEWER_CONFIG__;</script>
<p><a href="../../index.html">← registry</a> · <a href="data/manifest.json">manifest</a></p>
<h1 id="title">basemap viewer</h1>
<p class="muted" id="rows"></p>
<canvas id="plot"></canvas>
<script>
(async()=>{
 const cfg=window.VIEWER_CONFIG||{};
 const M=await (await fetch(cfg.manifest||'data/manifest.json')).json();
 document.getElementById('title').textContent=M.title||M.map_id||'basemap';
 document.getElementById('rows').textContent=(M.rows_total?M.rows_total.toLocaleString()+' rows — ':'')+(M.rows_note||'');
 // Minimal base-grid render so the fallback page is never blank.
 async function grid(L){const b=await(await fetch(`data/grid-all-${L}.bin`)).arrayBuffer();
  const h=new Uint32Array(b,0,4);if(h[0]!==0x42494E31)return null;const n=h[2];
  const idx=new Uint32Array(b,16,n),cnt=new Uint32Array(b,16+4*n,n);return{L:h[1],idx,cnt};}
 const levels=M.levels||[256];const lv=levels.includes(256)?256:levels[levels.length-1];
 const g=await grid(lv);const c=document.getElementById('plot'),r=c.getBoundingClientRect(),d=devicePixelRatio||1;
 c.width=r.width*d;c.height=r.height*d;const ctx=c.getContext('2d');ctx.scale(d,d);
 if(g){let mx=0;for(const v of g.cnt)mx=Math.max(mx,v);const cw=r.width/g.L,ch=r.height/g.L;
  for(let i=0;i<g.idx.length;i++){const cell=g.idx[i],cx=cell%g.L,cy=(cell-cx)/g.L;
   const t=Math.log1p(g.cnt[i])/Math.log1p(mx);ctx.fillStyle=`rgb(${248-222*t|0},${249-191*t|0},${252-114*t|0})`;
   ctx.fillRect(cx*cw,(g.L-1-cy)*ch,Math.ceil(cw),Math.ceil(ch));}}
})();
</script>
"""


if __name__ == "__main__":  # manual staging build
    import argparse
    from map_registry import _load_json, REGISTRY_PATH, SITE_DIR, scan  # type: ignore

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--site", default=str(SITE_DIR))
    args = ap.parse_args()
    reg = _load_json(REGISTRY_PATH) or scan()
    result = build_map_viewers(reg, Path(args.site), only=args.only, force=args.force)
    print(f"built {len(result)} viewers -> {args.site}")
