#!/usr/bin/env python3
"""Recent-maps gallery (owner ask via overseer 2026-09-04): browsable interactive viewer pages for this week's
sandbox maps (MONET draws, their-UMAP-on-our-champion, NeoMME text + joint-modality raw/centered, exp-2c cells),
reusing the existing map_tiles + map_viewer primitives (NOT _build_one, which misclassifies sandbox maps as
minilm-150m and resolves FineWeb/Pile text tooltips — garbage for these). Coverage, not new UI.

Each map -> viewer/<map_id>/ (base density grid + optional per-category color subset layers via row_filter id
arrays) + a thumbnail; a thin recent/index.html links them. Also post-merges maps-index.json (union, since
write_maps_index clobbers). Idempotent + skip-if-no-coords (re-run as more maps land).

CPU-only, read-only over coords. Usage: build_recent_maps_gallery.py [--only <map_id substr>]"""
import sys, json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import map_tiles as mt
import map_viewer as mv

SB = Path("/data/latent-basemap/sandbox")
SITE = Path.home() / ".agent/basemap-maps"
REL = "sandbox"                     # viewer dir prefix / index folder

# color: None (density only) | ("modality", npy_path) | ("source", ids_json_path)
SPECS = [
    {"ds": "monet-random-clip-2m",   "title": "MONET random draw — CLIP-512 (champion 2M)"},
    {"ds": "monet-random-dino-2m",   "title": "MONET random draw — DINOv2-1536 (champion 2M)"},
    {"ds": "monet-draw-random-clip", "title": "MONET draw:random — CLIP (champion)"},
    {"ds": "monet-draw-sscd-clip",   "title": "MONET draw:sscd — CLIP (champion)"},
    {"ds": "monet-draw-annfaiss-clip","title": "MONET draw:annfaiss — CLIP (champion)"},
    {"ds": "monet-draw-theirfaiss-clip","title": "MONET draw:theirfaiss — CLIP (champion)"},
    {"ds": "monet-theirumap-clip",   "title": "their-UMAP rows, OUR champion — CLIP"},
    {"ds": "monet-theirumap-dino",   "title": "their-UMAP rows, OUR champion — DINOv2"},
    {"ds": "theirumap-published-1m", "title": "THEIR published UMAP-1M layout (Jasper MONET) — flip vs our champion, same rows",
     "coords_override": "/data2/monet/theirumap/their-layout"},
    {"ds": "monet-neomme-fineweb-2m", "title": "NeoMME-1024 fineweb text — RAW (champion 2M)"},
    {"ds": "monet-neomme-fineweb-2m-centered", "title": "NeoMME-1024 fineweb text — CENTERED (champion 2M)"},
    {"ds": "monet-neomme-pairs-500k", "title": "NeoMME joint image+text 500K — RAW (color by modality)",
     "color": ("modality", "/data2/monet/neomme-pairs-500k/modality.npy")},
    {"ds": "monet-neomme-pairs-500k-centered", "title": "NeoMME joint 500K — CENTERED (color by modality)",
     "color": ("modality", "/data2/monet/neomme-pairs-500k-centered/modality.npy")},
    {"ds": "monet-neomme-pairs-500k-2c-w05", "title": "exp-2c pair-edge w0.5x (color by modality)",
     "color": ("modality", "/data2/monet/neomme-pairs-500k-centered/modality.npy")},
    {"ds": "monet-neomme-pairs-500k-2c-w1", "title": "exp-2c pair-edge w1x (color by modality)",
     "color": ("modality", "/data2/monet/neomme-pairs-500k-centered/modality.npy")},
    {"ds": "monet-neomme-pairs-500k-2c-w2", "title": "exp-2c pair-edge w2x (color by modality)",
     "color": ("modality", "/data2/monet/neomme-pairs-500k-centered/modality.npy")},
]
MOD_LABELS = {0: "image rows", 1: "text rows"}


def _color_layers(source, extent, color, data: Path):
    """Categorical color = one grid subset layer per class, row_filter = id array of that class."""
    if not color:
        return []
    kind, path = color
    if kind == "modality":
        field = np.load(path); labels = MOD_LABELS
    else:                                              # source: factorize strings from ids.json
        srcs = json.load(open(path)).get("sources", [])
        uniq = sorted(set(srcs)); code = {s: i for i, s in enumerate(uniq)}
        field = np.array([code[s] for s in srcs], dtype=np.int32); labels = {i: s for s, i in code.items()}
    layers = []
    vals, counts = np.unique(field, return_counts=True)
    order = vals[np.argsort(-counts)][:12]             # top-12 categories by size
    for v in order:
        ids = np.nonzero(field == v)[0].astype(np.int64)
        g = mt.bin_all_levels(source, mv.LEVELS, extent, row_filter=ids)
        lv = []
        for lvl in mv.LEVELS:
            i2, c2 = g[lvl]
            if 16 + 8 * len(i2) > mv.MAX_GRID_BYTES:
                continue
            mt.write_grid(str(data / f"grid-{kind}-{v}-{lvl}.bin"), lvl, i2, c2)
            lv.append(lvl)
        if lv:
            layers.append({"key": f"{kind}-{v}", "label": str(labels.get(int(v), v)),
                           "kind": "grid", "rows": int(g[mv.LEVELS[0]][1].sum()),
                           "levels": lv, "group": kind})
    return layers


def build_one(spec) -> dict | None:
    ds = spec["ds"]
    coords_dir = Path(spec["coords_override"]) if spec.get("coords_override") else SB / ds / "champion-bs16k"
    if not (coords_dir / "coordinates.npy").is_file():
        return None
    map_id = f"{REL}-{ds}"
    out = SITE / "viewer" / map_id; data = out / "data"; data.mkdir(parents=True, exist_ok=True)
    mv._copy_assets(SITE)
    source = mt.MapSource(str(coords_dir), cache_dir=str(out / "_cache"))
    extent = list(source.extent())
    # base density
    grids = mt.bin_all_levels(source, mv.LEVELS + mv.FINE_LEVELS, extent, row_filter=None)
    emitted, tiled = [], []
    for lvl in mv.LEVELS + mv.FINE_LEVELS:
        idx, cnt = grids[lvl]
        res = mt.write_grid_auto(str(data), "all", lvl, idx, cnt)
        (tiled.append({"level": lvl, "split": res["split"]}) if res["tiled"] else emitted.append(lvl))
    rows_all = int(grids[mv.LEVELS[0]][1].sum())
    base = {"key": "all", "label": "All points", "kind": "grid", "rows": rows_all, "levels": emitted}
    if tiled:
        base["tiled_levels"] = tiled
    layers = [base] + _color_layers(source, extent, spec.get("color"), data)
    # ffr for display
    ffr = None
    sj = coords_dir / "summary.json"
    if sj.is_file():
        try:
            ffr = json.load(open(sj)).get("quick_ffr_v2") or json.load(open(sj)).get("quick_ffr_at_0.1pct")
        except Exception:
            ffr = None
    manifest = {"schema": mv.MANIFEST_SCHEMA, "generated_utc": datetime.now(timezone.utc).isoformat(),
                "map_id": map_id, "title": spec["title"], "rows_total": rows_all,
                "rows_note": "exact rows" + ("; colored by " + spec["color"][0] if spec.get("color") else ""),
                "map_kind": "sandbox", "extent": extent, "levels": mv.LEVELS,
                "sample_level": mv.SAMPLE_LEVEL, "super_tile": mv.SUPER_TILE,
                "layers": layers, "metrics": ({"quick_ffr@0.1pct": round(ffr, 4)} if ffr else {}), "skipped": []}
    (data / "manifest.json").write_text(json.dumps(manifest, indent=1))
    mv._write_thumbnail(data / "grid-all-256.bin", out / "thumb.png")
    cfg = {"dataDir": "data", "back": "../../" + REL + "/index.html", "manifest": "data/manifest.json",
           "assets": mv.ASSETS_REL, "map_id": map_id, "title": spec["title"]}
    (out / "index.html").write_text(mv._instantiate(mv._load_template(), cfg))
    return {"map_id": map_id, "title": spec["title"], "ffr": round(ffr, 4) if ffr else None,
            "rows": rows_all, "colored": spec.get("color", [None])[0] if spec.get("color") else None}


def _scan_cards():
    """Assemble the full card list from ALL built sandbox pages on disk (so --only keeps the index complete)."""
    cards = []
    for mdir in sorted((SITE / "viewer").glob(f"{REL}-*")):
        mf = mdir / "data" / "manifest.json"
        if not mf.is_file():
            continue
        m = json.loads(mf.read_text()); met = m.get("metrics", {})
        ffr = next((v for k, v in met.items() if "ffr" in k.lower()), None)
        colored = next((l.get("group") for l in m.get("layers", []) if l.get("group")), None)
        cards.append({"map_id": mdir.name, "title": m.get("title", mdir.name),
                      "ffr": ffr, "rows": m.get("rows_total", 0), "colored": colored})
    return cards


def build_index(cards):
    idx = SITE / REL; idx.mkdir(parents=True, exist_ok=True)
    items = "".join(
        f'<a class="card" href="../viewer/{c["map_id"]}/index.html">'
        f'<img loading="lazy" src="../viewer/{c["map_id"]}/thumb.png">'
        f'<div class="t">{c["title"]}</div>'
        f'<div class="m">{c["rows"]:,} pts'
        + (f' · FFR {c["ffr"]}' if c["ffr"] else '')
        + (f' · color: {c["colored"]}' if c["colored"] else '') + '</div></a>'
        for c in cards)
    html = f"""<!doctype html><meta charset=utf-8><title>Recent basemaps (last 2 days)</title>
<style>body{{font-family:system-ui;max-width:1400px;margin:1.5rem auto;padding:0 1rem;color:#1a202c}}
h1{{margin:.2rem 0}}.sub{{color:#4a5568;margin-top:0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:1rem;margin-top:1rem}}
.card{{border:1px solid #e2e8f0;border-radius:8px;padding:.6rem;text-decoration:none;color:#2d3748;background:#fff}}
.card:hover{{border-color:#d69e2e;box-shadow:0 0 0 2px #f6e05e55}}
.card img{{width:100%;display:block;border-radius:4px;background:#f7fafc}}
.t{{font-weight:600;font-size:.9rem;margin-top:.4rem}}.m{{color:#718096;font-size:.8rem;margin-top:.2rem}}</style>
<h1>Recent basemaps</h1><p><a href="../latest-images/">Latest image comparisons: DINO vs CLIP at 103.8M, and the DINO 2M / 6M ladder</a></p><p class="sub">This week's sandbox maps — MONET draws, their-UMAP vs our champion,
NeoMME text + joint-modality (raw/centered), exp-2c pair-edge sweep. Click a card to pan/zoom; color layers via
the in-viewer dropdown. Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.</p>
<div class="grid">{items}</div>"""
    (idx / "index.html").write_text(html)
    return idx / "index.html"


def main():
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    cards = []
    for spec in SPECS:
        if only and only not in spec["ds"]:
            continue
        c = build_one(spec)
        if c:
            cards.append(c); print(f"built {c['map_id']} ({c['rows']:,} pts, ffr={c['ffr']}, color={c['colored']})", flush=True)
        else:
            print(f"skip {spec['ds']} (no coords yet)", flush=True)
    page = build_index(_scan_cards())      # index from ALL built pages, not just this run's
    # post-merge into maps-index.json (union; write_maps_index clobbers)
    mi = SITE / "maps-index.json"
    if mi.is_file():
        idx = json.loads(mi.read_text()); existing = {m.get("map_id") for m in idx.get("maps", [])}
        for c in cards:
            if c["map_id"] not in existing:
                idx.setdefault("maps", []).append({
                    "map_id": c["map_id"], "title": c["title"], "kind": "sandbox-map", "round_id": None,
                    "date": None, "rows_total": c["rows"], "rows_note": "sandbox",
                    "data": f"viewer/{c['map_id']}/data/", "thumbnail": f"viewer/{c['map_id']}/thumb.png",
                    "evidence_status": "sandbox", "metrics": {}, "probes": [], "tags": [c["colored"]] if c["colored"] else []})
        mi.write_text(json.dumps(idx, indent=1))
    print(f"\nGALLERY: {len(cards)} maps -> {page}\nURL: http://gsv.local:8800/basemap-maps/{REL}/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
