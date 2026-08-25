#!/usr/bin/env python3
"""build_ladder_gallery.py — the scale-ladder gallery page.

One page, chronological by rung (2M -> 100M): every promoted-recipe seed map
plus that rung's cuML reference card inline (owner request 2026-08-19,
replacing the separate kernels/cuML comparison pages for this purpose).

CPU-only: renders any missing density.png from coordinates via map_renders
(niced; safe next to a training GPU). Emits a static page under
~/.agent/basemap-maps/gallery/ (LAN) that is also copied into the gh-pages
site by mapviewer/scripts/publish_ghpages.sh.

Cells are declarative in LADDER below; metrics shown are the sealed/summary
numbers recorded next to each artifact where available (no recomputation).
"""

from __future__ import annotations

import html
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from map_renders import binned_counts, render_png, robust_extent  # noqa: E402

import numpy as np  # noqa: E402

OUT = Path.home() / ".agent/basemap-maps/gallery"
SANDBOX = Path("/data/latent-basemap/sandbox")
RUNS = Path("/data/latent-basemap/runs")


@dataclass
class Cell:
    label: str
    coords: Path | None = None       # render from these if png missing
    png: Path | None = None          # pre-rendered image to copy
    caption: str = ""
    kind: str = "seed"               # seed | cuml | preview | pending
    metrics: dict = field(default_factory=dict)


def r0265_seed(seed: int) -> Cell:
    art = None
    for q in sorted(RUNS.glob("round-0265/queue*/artifacts/"
                              f"minilm-mixed-2m-fneg-x4-md000-seed{seed}-r0265-v1")):
        if (q / "coordinates.npy").exists():
            art = q
    return Cell(label=f"2M seed {seed}", coords=art / "coordinates.npy" if art else None,
                caption="R0265 family cell (x4 dose)")


def sandbox_cell(rung: str, arm: str, label: str, caption: str) -> Cell:
    d = SANDBOX / f"{rung}-knobs" / arm
    m = {}
    sj = d / "summary.json"
    if sj.exists():
        s = json.loads(sj.read_text())
        m = {"quick-FFR": round(s.get("quick_ffr_at_0.1pct", 0), 4)}
    return Cell(label=label, coords=d / "coordinates.npy",
                png=d / "density.png" if (d / "density.png").exists() else None,
                caption=caption, metrics=m)


def cuml_cell(rung_dir: str, label: str) -> Cell:
    d = SANDBOX / "cuml-ref" / rung_dir
    return Cell(label=f"cuML reference", coords=d / "coordinates.npy",
                caption=f"non-parametric full-substrate reference ({label})",
                kind="cuml")


def r0267_seed(seed: int) -> Cell:
    corr = "queue-correction-3" if seed == 42 else "queue-correction-4"
    d = (RUNS / "round-0267" / corr / "artifacts" /
         f"minilm-mixed-50000k-fneg-x2-md000-hostint8-seed{seed}-r0267-v1")
    mets = {42: (1.0860, 0.2472, 0.5594), 43: (1.0326, 0.1165, 0.5539),
            44: (0.9234, 0.1631, 0.5495)}[seed]
    return Cell(label=f"50M seed {seed}", coords=d / "coordinates.npy",
                caption="R0267 sealed (x2 dose, host-int8)", kind="seed",
                metrics={"spacing": mets[0], "fog": mets[1], "FFR": mets[2]})


LADDER: list[tuple[str, str, list[Cell]]] = [
    ("2M", "13-seed calibration family (R0265, x4 dose) — the gate family",
     [r0265_seed(s) for s in range(42, 55)] + [cuml_cell("2m", "2M")]),
    ("6.25M", "scale checks (sandbox, P1/P2)",
     [sandbox_cell("6250k", "umap-md000-x4-fneg10", "6.25M x4 (pinning cell)",
                   "P2 pinning cell"),
      sandbox_cell("6250k", "umap-md000-x2-fneg10", "6.25M x2", "P1 scale pair"),
      sandbox_cell("6250k", "umap-md000-x2-fneg10-seed43", "6.25M x2 seed 43",
                   "seed replicate"),
      cuml_cell("6250k", "6.25M")]),
    ("12.5M", "third scale point (sandbox, P1)",
     [sandbox_cell("12500k", "umap-md000-x2-fneg10", "12.5M x2", "P1 drift point"),
      cuml_cell("12500k", "12.5M")]),
    ("25M", "host-int8 evidence cell (P5/P1)",
     [sandbox_cell("25000k", "umap-md000-x2-fneg10-hostint8", "25M x2 host-int8",
                   "fourth x2 point; int8 residency")]),
    ("50M", "staging rung — 3 seeds, 50M_PASS (R0267); cuML infeasible >=12.5M "
            "on this card (waiver 2026-08-17)",
     [r0267_seed(42), r0267_seed(43), r0267_seed(44)]),
    ("100M", "flagship (R0268) — 3 seeds training; preview from the orphaned "
             "pre-retrain train (non-evidence)",
     [Cell(label="100M seed 42 PREVIEW",
           coords=RUNS / "round-0268/salvage/seed42/coordinates.npy",
           caption="orphaned attempt-1 train; tripwire preview only",
           kind="preview",
           metrics={"spacing": 1.0139, "fog": 0.2040}),
      Cell(label="100M seed 42", kind="pending", caption="retraining"),
      Cell(label="100M seed 43", kind="pending", caption="queued"),
      Cell(label="100M seed 44", kind="pending", caption="queued")]),
]

BADGE = {"seed": ("sealed/sandbox", "#2b6cb0"), "cuml": ("cuML ref", "#6b46c1"),
         "preview": ("NON-EVIDENCE", "#c05621"), "pending": ("pending", "#718096")}


def ensure_png(cell: Cell, dest: Path) -> bool:
    if dest.exists():
        return True
    if cell.png and cell.png.exists():
        shutil.copy2(cell.png, dest)
        return True
    if cell.coords and cell.coords.exists():
        xy = np.load(cell.coords, mmap_mode="r")
        render_png(binned_counts(xy, robust_extent(xy)), dest)
        return True
    return False


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sections = []
    for rung, blurb, cells in LADDER:
        cards = []
        for c in cells:
            slug = c.label.lower().replace(" ", "-").replace(".", "_")
            png = OUT / f"{slug}.png"
            have = c.kind != "pending" and ensure_png(c, png)
            btxt, bcol = BADGE[c.kind]
            mets = " · ".join(f"{k} {v}" for k, v in c.metrics.items())
            body = (f'<img src="{png.name}" loading="lazy">' if have else
                    '<div class="pending">training…</div>')
            cards.append(
                f'<figure class="{c.kind}">{body}'
                f'<figcaption><b>{html.escape(c.label)}</b> '
                f'<span class="badge" style="background:{bcol}">{btxt}</span><br>'
                f'<small>{html.escape(c.caption)}'
                f'{("<br>" + html.escape(mets)) if mets else ""}</small>'
                f'</figcaption></figure>')
            print(("ok  " if have else "skip") + f"  {c.label}")
        sections.append(
            f'<section><h2>{rung}</h2><p>{html.escape(blurb)}</p>'
            f'<div class="grid">{"".join(cards)}</div></section>')

    (OUT / "index.html").write_text(f"""<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>latent-basemap — the scale ladder</title>
<script>
 // Google Analytics — only on the public site; the same page is served on
 // gsv.local / localhost and those views should not be counted.
 if (location.hostname === 'enjalot.github.io') {{
   var gaScript = document.createElement('script');
   gaScript.async = true;
   gaScript.src = 'https://www.googletagmanager.com/gtag/js?id=G-DZJP5PPBF7';
   document.head.appendChild(gaScript);
   window.dataLayer = window.dataLayer || [];
   function gtag() {{ dataLayer.push(arguments); }}
   gtag('js', new Date());
   gtag('config', 'G-DZJP5PPBF7');
 }}
</script>
<style>
 body {{ font-family: system-ui; max-width: 1500px; margin: 2rem auto; padding: 0 1rem; color: #1a202c; }}
 h1 {{ margin-bottom: .2rem }} .sub {{ color: #4a5568; margin-top: 0 }}
 section {{ margin: 2.2rem 0 }} h2 {{ border-bottom: 2px solid #e2e8f0; padding-bottom: .3rem }}
 .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 14px }}
 figure {{ margin: 0 }} img {{ width: 100%; border: 1px solid #e2e8f0; border-radius: 6px; background: #fff }}
 figure.cuml img {{ border-color: #6b46c1 }} figure.preview img {{ border-color: #c05621 }}
 .badge {{ color: #fff; border-radius: 4px; padding: 1px 6px; font-size: .7rem; vertical-align: middle }}
 .pending {{ aspect-ratio: 1; display: grid; place-items: center; border: 1px dashed #a0aec0;
             border-radius: 6px; color: #718096 }}
 figcaption {{ padding: .3rem .1rem }}
</style>
<h1>The scale ladder</h1>
<p class="sub">Every promoted-recipe map from 2M to 100M rows, in training-scale order, with each
rung's non-parametric cuML reference inline. Interactive maps: <a href="../viewer/">viewer</a>.</p>
{"".join(sections)}
<p><small>Renders are per-map robust-extent density bins (log ramp). cuML references stop at
12.5M — the hardware ceiling for full-substrate non-parametric UMAP on one 32 GB card.
Metrics shown are the sealed values where the artifact is round evidence, or sandbox
summaries otherwise. 100M preview is the orphaned pre-retrain train: healthy but formally
non-evidence; the retrained seeds replace it as they land.</small></p>
""")
    print(f"page: {OUT}/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
