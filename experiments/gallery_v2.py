#!/usr/bin/env python3
"""Gallery v2: interactive linked small-multiples over registry maps.

Called by map_registry.py publish (or standalone). For each group of round
maps sharing a row universe (same n_rows), draws a common seed-0 20k-row
sample through every map, Procrustes-aligns each to the newest accepted map,
and emits a self-contained canvas viewer with linked hover and per-point
drift coloring.

Fixes the v1 gallery's defects (plan-map-inspection.md): provenance travels
in the manifest (round, release, config, panel scores, sample hash), output
dirs are per-generation immutable in content (regenerated wholesale, never
hand-mutated), and the spec used is persisted next to the output.
"""
from __future__ import annotations

import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SAMPLE_SEED = 0
SAMPLE_N = 20_000
SCHEMA = "basemap-gallery-v2-manifest-v1"


def _gather(coords_dir: Path, rows: np.ndarray) -> np.ndarray:
    chunks = sorted(coords_dir.glob("chunk-*/coordinates.npy"))
    if not chunks:
        raise FileNotFoundError(f"no coordinate chunks under {coords_dir}")
    sizes = [np.load(c, mmap_mode="r").shape[0] for c in chunks]
    offsets = np.cumsum([0] + sizes)
    out = np.empty((len(rows), 2), np.float32)
    ci = np.searchsorted(offsets, rows, side="right") - 1
    for c in np.unique(ci):
        m = ci == c
        arr = np.load(chunks[c], mmap_mode="r")
        out[m] = arr[rows[m] - offsets[c]]
    return out


def _procrustes(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Similarity-transform align x onto ref (translation, rotation/reflection, scale)."""
    mu_r, mu_x = ref.mean(0), x.mean(0)
    r0, x0 = ref - mu_r, x - mu_x
    nr, nx = np.linalg.norm(r0), np.linalg.norm(x0)
    if nx == 0 or nr == 0:
        return x.copy()
    r0, x0 = r0 / nr, x0 / nx
    u, s, vt = np.linalg.svd(x0.T @ r0)
    rot = u @ vt
    return (x0 @ rot) * nr * s.sum() + mu_r


def build_compare_groups(registry: dict, site_dir: Path) -> list[dict]:
    maps = [m for m in registry["maps"] if m["kind"] == "round-map" and m.get("n_rows")]
    groups: dict[int, list[dict]] = {}
    for m in maps:
        groups.setdefault(int(m["n_rows"]), []).append(m)

    built = []
    for n_rows, group in sorted(groups.items(), reverse=True):
        group = sorted(group, key=lambda m: (("accepted" in m["evidence_status"]), m.get("date") or ""),
                       reverse=True)
        ref_map = group[0]
        slug = f"n{n_rows // 1_000_000}m" if n_rows >= 1_000_000 else f"n{n_rows}"
        out = site_dir / "compare" / slug
        (out / "data").mkdir(parents=True, exist_ok=True)

        rng = np.random.RandomState(SAMPLE_SEED)
        rows = np.sort(rng.choice(n_rows, min(SAMPLE_N, n_rows), replace=False)).astype(np.int64)
        sample_sha = hashlib.sha256(rows.tobytes()).hexdigest()

        ref_xy = None
        panels = []
        for m in group:
            coords_dir = Path(m["coordinates"]["dir"].removeprefix("gsv:"))
            try:
                xy = _gather(coords_dir, rows).astype(np.float64)
            except (FileNotFoundError, OSError) as e:
                print(f"  skip {m['map_id']}: {e}")
                continue
            if ref_xy is None:
                ref_xy = xy
                aligned, drift = xy, np.zeros(len(rows), np.float32)
            else:
                aligned = _procrustes(ref_xy, xy)
                drift = np.linalg.norm(aligned - ref_xy, axis=1).astype(np.float32)
            (out / "data" / f"{m['map_id']}.xy.bin").write_bytes(aligned.astype(np.float32).tobytes())
            (out / "data" / f"{m['map_id']}.drift.bin").write_bytes(drift.tobytes())
            panels.append({
                "map_id": m["map_id"],
                "round_id": m["round_id"],
                "date": m.get("date"),
                "evidence_status": m["evidence_status"],
                "release_sha": m.get("release_sha"),
                "architecture": f'{m.get("architecture")} h{m.get("hidden_dim")}',
                "panel": {k: m["panel"].get(k) for k in
                          ("ffr", "density", "purity_k256", "purity_k1024", "proj_ffr")},
                "model_sha256": (m.get("model") or {}).get("sha256"),
                "coordinates_dir": m["coordinates"]["dir"],
                "is_reference": m is group[0],
                "mean_drift": float(drift.mean()),
                "files": {"xy": f"data/{m['map_id']}.xy.bin", "drift": f"data/{m['map_id']}.drift.bin"},
            })
        if not panels:
            continue

        manifest = {
            "schema": SCHEMA,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "n_rows": n_rows,
            "sample": {"seed": SAMPLE_SEED, "n": len(rows), "ids_sha256": sample_sha},
            "alignment": "similarity-procrustes-to-reference",
            "reference_map": ref_map["map_id"],
            "panels": panels,
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
        # persist the generating spec (v1 gallery never did)
        (out / "spec.json").write_text(json.dumps({
            "generator": "experiments/gallery_v2.py",
            "registry": "/data/latent-basemap/maps.json",
            "registry_generated_utc": registry.get("generated_utc"),
            "group_n_rows": n_rows,
            "map_ids": [p["map_id"] for p in panels],
        }, indent=1))
        (out / "index.html").write_text(_viewer_html(manifest))
        built.append({"slug": slug, "n_rows": n_rows, "panels": len(panels), "dir": str(out)})
        print(f"  compare/{slug}: {len(panels)} panels, sample {len(rows)} rows")
    return built


def _viewer_html(manifest: dict) -> str:
    title = f"basemap compare · {manifest['n_rows']:,} rows"
    return f"""<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ color-scheme: light dark; --fg:#1a1d21; --bg:#fff; --muted:#667; --line:#e2e5ea; --card:#f6f7f9; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --fg:#e6e8eb; --bg:#121417; --muted:#9aa1ab; --line:#2a2f36; --card:#1b1f24; }} }}
body {{ font: 14px/1.5 system-ui, sans-serif; color: var(--fg); background: var(--bg);
       margin: 0 auto; max-width: 1400px; padding: 18px; }}
h1 {{ font-size: 1.2rem; }} a {{ color: inherit; }} .muted {{ color: var(--muted); }}
#panels {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); gap: 14px; }}
.panel {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 10px; }}
.panel h3 {{ margin: 2px 0 2px; font-size: 13px; }}
.panel .sub {{ font-size: 11.5px; color: var(--muted); margin-bottom: 6px; overflow-wrap: anywhere; }}
canvas {{ width: 100%; aspect-ratio: 1; display: block; border-radius: 6px; background: #fff; cursor: crosshair; }}
@media (prefers-color-scheme: dark) {{ canvas {{ background: #0d0f12; }} }}
.controls {{ margin: 10px 0 14px; display: flex; gap: 14px; flex-wrap: wrap; align-items: center; }}
#tip {{ position: fixed; pointer-events: none; background: var(--card); border: 1px solid var(--line);
       border-radius: 6px; padding: 4px 8px; font-size: 12px; display: none; z-index: 5; }}
</style>
<h1>{html.escape(title)}</h1>
<p class="muted">Common seed-{manifest['sample']['seed']} sample of {manifest['sample']['n']:,} rows ·
aligned to <b>{html.escape(manifest['reference_map'])}</b> (similarity Procrustes) ·
sample sha <code>{manifest['sample']['ids_sha256'][:12]}</code> ·
<a href="../../index.html">registry</a> · <a href="manifest.json">manifest</a> · <a href="spec.json">spec</a></p>
<div class="controls">
  <label>color: <select id="colorBy"><option value="drift">drift vs reference</option><option value="none">none</option></select></label>
  <label>point size: <input id="ptSize" type="range" min="0.5" max="3" step="0.25" value="1"></label>
  <label><input id="clipExtent" type="checkbox" checked> clip axes to p0.5–p99.5 (outlier islands off-frame)</label>
  <span class="muted" id="hoverInfo">hover links panels by row</span>
</div>
<div id="panels"></div>
<div id="tip"></div>
<script>
const M = {json.dumps(manifest)};
const N = M.sample.n;
const panels = [];
let extent = null, hoverIdx = -1;

async function loadBin(url) {{
  const buf = await (await fetch(url)).arrayBuffer();
  return new Float32Array(buf);
}}

function computeExtent(all, clip) {{
  if (clip) {{
    const xs=[], ys=[];
    for (const xy of all) for (let i=0;i<N;i++) {{ xs.push(xy[2*i]); ys.push(xy[2*i+1]); }}
    xs.sort((a,b)=>a-b); ys.sort((a,b)=>a-b);
    const lo=Math.floor(0.005*(xs.length-1)), hi=Math.floor(0.995*(xs.length-1));
    const x0=xs[lo], x1=xs[hi], y0=ys[lo], y1=ys[hi];
    const px=(x1-x0)*0.04, py=(y1-y0)*0.04;
    return [x0-px, x1+px, y0-py, y1+py];
  }}
  let x0=1e30,x1=-1e30,y0=1e30,y1=-1e30;
  for (const xy of all) for (let i=0;i<N;i++) {{
    const x=xy[2*i], y=xy[2*i+1];
    if (x<x0)x0=x; if (x>x1)x1=x; if (y<y0)y0=y; if (y>y1)y1=y;
  }}
  const px=(x1-x0)*0.03, py=(y1-y0)*0.03;
  return [x0-px, x1+px, y0-py, y1+py];
}}

function driftColor(t) {{ // t in [0,1] -> blue-grey to orange-red
  const r = Math.round(70 + 185*t), g = Math.round(110 - 30*t), b = Math.round(190 - 150*t);
  return `rgb(${{r}},${{g}},${{b}})`;
}}

function draw(p) {{
  const c = p.canvas, ctx = c.getContext('2d');
  const dpr = devicePixelRatio || 1, w = c.clientWidth, h = c.clientHeight;
  c.width = w*dpr; c.height = h*dpr; ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  const [x0,x1,y0,y1] = extent, sx = w/(x1-x0), sy = h/(y1-y0);
  const size = +document.getElementById('ptSize').value;
  const colorBy = document.getElementById('colorBy').value;
  const maxD = p.maxDrift || 1;
  ctx.globalAlpha = 0.55;
  for (let i=0;i<N;i++) {{
    const x = (p.xy[2*i]-x0)*sx, y = h-(p.xy[2*i+1]-y0)*sy;
    ctx.fillStyle = colorBy==='drift' && p.drift ? driftColor(Math.min(1, p.drift[i]/maxD)) : '#4a7dbd';
    ctx.fillRect(x, y, size, size);
  }}
  ctx.globalAlpha = 1;
  if (hoverIdx >= 0) {{
    const x = (p.xy[2*hoverIdx]-x0)*sx, y = h-(p.xy[2*hoverIdx+1]-y0)*sy;
    ctx.strokeStyle = '#e5484d'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(x, y, 6, 0, 7); ctx.stroke();
  }}
}}

function drawAll() {{ panels.forEach(draw); }}

function nearest(p, mx, my) {{
  const c = p.canvas, w = c.clientWidth, h = c.clientHeight;
  const [x0,x1,y0,y1] = extent, sx = w/(x1-x0), sy = h/(y1-y0);
  let best = -1, bd = 15*15;
  for (let i=0;i<N;i++) {{
    const dx = (p.xy[2*i]-x0)*sx - mx, dy = h-(p.xy[2*i+1]-y0)*sy - my;
    const d = dx*dx + dy*dy;
    if (d < bd) {{ bd = d; best = i; }}
  }}
  return best;
}}

(async () => {{
  const host = document.getElementById('panels');
  for (const pm of M.panels) {{
    const div = document.createElement('div'); div.className = 'panel';
    const scores = pm.panel || {{}};
    div.innerHTML = `<h3>${{pm.map_id}} ${{pm.is_reference ? '· reference' : ''}}</h3>
      <div class="sub">${{pm.architecture}} · ${{pm.evidence_status}} · ffr ${{scores.ffr ?? '—'}} ·
      dens ${{scores.density ?? '—'}} · proj ${{scores.proj_ffr ?? '—'}} ·
      drift μ ${{pm.mean_drift.toFixed(2)}}<br>release <code>${{(pm.release_sha||'').slice(0,12)}}</code></div>
      <canvas></canvas>`;
    host.appendChild(div);
    const canvas = div.querySelector('canvas');
    const xy = await loadBin(pm.files.xy);
    const drift = await loadBin(pm.files.drift);
    const sorted = Array.from(drift).sort((a,b)=>a-b);
    const p = {{ pm, canvas, xy, drift, maxDrift: sorted[Math.floor(0.98*(N-1))] || 1 }};
    panels.push(p);
    canvas.addEventListener('mousemove', e => {{
      const r = canvas.getBoundingClientRect();
      hoverIdx = nearest(p, e.clientX-r.left, e.clientY-r.top);
      const tip = document.getElementById('tip');
      if (hoverIdx >= 0) {{
        tip.style.display = 'block';
        tip.style.left = (e.clientX+14)+'px'; tip.style.top = (e.clientY+14)+'px';
        tip.textContent = `row #${{hoverIdx}} · drift ${{p.drift[hoverIdx].toFixed(3)}}`;
      }} else tip.style.display = 'none';
      drawAll();
    }});
    canvas.addEventListener('mouseleave', () => {{
      hoverIdx = -1; document.getElementById('tip').style.display='none'; drawAll();
    }});
  }}
  const setExtent = () => {{ extent = computeExtent(panels.map(p => p.xy), document.getElementById('clipExtent').checked); }};
  setExtent();
  drawAll();
  document.getElementById('colorBy').onchange = drawAll;
  document.getElementById('ptSize').oninput = drawAll;
  document.getElementById('clipExtent').onchange = () => {{ setExtent(); drawAll(); }};
  addEventListener('resize', drawAll);
}})();
</script>
"""
