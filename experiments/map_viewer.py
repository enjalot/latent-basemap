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

# Full-map grid resolution ladder. write_grid returns the levels it actually
# emitted (it may drop 1024 for a subset layer whose bin would exceed 2 MB).
LEVELS = [64, 128, 256, 512, 1024]
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

def _emit_grids(mt, source, data: Path, layer_key: str, extent, row_filter):
    """Write grid-<layer>-<L>.bin for every level, dropping any that would
    exceed the static-fetch size cap. Returns the emitted level list."""
    grids = mt.bin_all_levels(source, LEVELS, extent, row_filter=row_filter)
    emitted, rows = [], 0
    for lvl in LEVELS:
        idx, cnt = grids[lvl]
        if 16 + 8 * len(idx) > MAX_GRID_BYTES:
            print(f"  {layer_key}: L{lvl} bin {16 + 8 * len(idx)}B > cap; dropped")
            continue
        mt.write_grid(str(data / f"grid-{layer_key}-{lvl}.bin"), lvl, idx, cnt)
        emitted.append(lvl)
        rows = int(cnt.sum())
    return emitted, rows


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


def _build_metrics(map_kind: str, coords_dir: Path, data: Path, mm, extent) -> dict:
    a = _metric_artifacts(map_kind, coords_dir)
    if a["kind"] == "r0108":
        if not a["core_panel_npz"].is_file():
            print(f"  metrics: core panel npz missing ({a['core_panel_npz']})")
            return {}
        result = mm.build_r0108_metrics(
            a["core_panel_npz"], a["ood"], data, extent=extent)
    else:
        if not a["reference_npz"].is_file():
            print(f"  metrics: reference npz missing ({a['reference_npz']})")
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
    thumb_rel = f"{viewer_rel}/thumb.png"

    built = {
        "map_id": map_id, "round_id": entry.get("round_id"),
        "viewer_rel": f"{viewer_rel}/index.html", "thumb_rel": thumb_rel,
        "title": _title(entry), "rows_total": _rows_total(entry),
        "evidence_status": entry.get("evidence_status"),
    }

    # Idempotency: skip a rebuild when the coordinates receipt is unchanged.
    if not force and manifest_path.is_file():
        try:
            prev = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            prev = {}
        if prev.get("coordinates_receipt_sha") == receipt_sha and receipt_sha:
            return built  # already current

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

    manifest_layers: list[dict] = []
    for layer in _plan_layers(entry, map_kind, mt):
        try:
            emitted, rows = _emit_grids(mt, source, data, layer["key"], extent,
                                        layer["row_filter"])
        except Exception as exc:
            print(f"  {map_id}: grids({layer['key']}) failed: {exc}")
            if layer["key"] == "all":
                return None  # base layer is mandatory
            continue
        if not emitted:
            if layer["key"] == "all":
                return None
            continue
        lyr = {
            "key": layer["key"], "label": layer["label"], "kind": "grid",
            "rows": rows, "levels": emitted,
        }
        if layer["group"]:
            lyr["group"] = layer["group"]
        manifest_layers.append(lyr)
        if layer["samples"]:
            try:
                samples = mt.sample_bins(source, SAMPLE_LEVEL, extent, k=3,
                                         rng_seed=42, row_filter=layer["row_filter"])
                mt.write_samples(data, layer["key"], samples, SAMPLE_LEVEL,
                                 SUPER_TILE, map_kind, cache_dir=cache_dir)
            except Exception as exc:
                print(f"  {map_id}: samples({layer['key']}) failed: {exc}")

    # Linked OOD probe corpora as point layers.
    for proj in _linked_projections(registry, entry):
        npz = _strip(proj.get("projection", {}).get("coordinates"))
        probe = proj.get("projection", {}).get("probe", proj["map_id"])
        key = f"probe-{probe}"
        if npz is None or not npz.is_file():
            continue
        try:
            xy = _load_probe_corpus_xy(npz)
            if xy is None or len(xy) == 0:
                continue
            mt.write_points(str(data / f"points-{key}.bin"), xy)
        except Exception as exc:
            print(f"  {map_id}: write_points({key}) failed: {exc}")
            continue
        manifest_layers.append({
            "key": key,
            "label": proj["projection"].get("display_name", key),
            "kind": "points", "rows": int(len(xy)), "group": "held-out",
        })

    # Metric packets (anchors + OOD queries).
    metrics = {}
    try:
        metrics = _build_metrics(map_kind, coords_dir, data, mm, extent)
    except Exception as exc:
        print(f"  {map_id}: build_metrics failed: {exc}")

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
    _write_thumbnail(data / "grid-all-256.bin", out / "thumb.png")

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
    return built


def build_map_viewers(registry: dict, site_dir: Path,
                      only=None, force: bool = False) -> list[dict]:
    """Build interactive viewers for allowlisted round/atlas maps.

    Returns one descriptor per map that has a viewer (built or already-current),
    used by the registry index card grid. Splices a ``viewer:start/end`` block
    into each round page, mirroring the projection gallery.
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
            f'<li><a href="../{html.escape(i["viewer_rel"])}">'
            f'{html.escape(i["title"])}</a> — interactive viewer</li>'
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
