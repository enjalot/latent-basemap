"""The kernel program's dedicated visual + metric comparison page.

Collects every sandbox arm (2M and 6.25M rungs), the cuML reference, and the
sealed legacy baselines onto one page at
``http://gsv.local:8800/basemap-maps/sandbox/kernels/``: a render grid grouped
by kernel family, the full metric table, and a Pareto scatter of held-out
fidelity against the collapse metric. Re-run any time; it only reads
summaries, heldout-eval.json and cached renders.
"""

from __future__ import annotations

import html
import json
import shutil
import sys
from pathlib import Path

SANDBOX = Path("/data/latent-basemap/sandbox")
SITE = Path.home() / ".agent/basemap-maps/sandbox/kernels"
HELDOUT = SANDBOX / "2m-knobs/heldout-eval.json"

GROUPS = [
    ("references", "References",
     "cuML (non-parametric target look) and the sealed legacy baseline."),
    ("legacy", "legacy_lp family",
     "The registered kernel: 1/(1+a·‖Δ‖_2b). Collapses regardless of dose/a/b."),
    ("umap", "umap kernel — min_dist sweep",
     "1/(1+a·r^2b), (a,b) fitted per min_dist. The plateau radius is the "
     "diffuseness dial: md000 tight → md050 diffuse."),
    ("gcauchy", "gcauchy — tail exponent",
     "(1+a·r^2b)^(-α). α=1 is the umap kernel; α<1 heavier tail (more "
     "inter-cluster room), α>1 lighter."),
    ("spectrum", "attraction/repulsion spectrum probes",
     "pos_ratio moved at fixed horizon; dose covaries (documented per card)."),
    ("6250k", "6.25M rung checks",
     "Winning configs re-trained at 3.1x scale on the sealed R0233 graph; "
     "compare against R0257's legacy maps."),
]


def classify(name: str, rung: str) -> str:
    if rung == "6250k":
        return "6250k"
    if name.startswith("gc-"):
        return "gcauchy"
    if "pos02" in name or "pos15" in name:
        return "spectrum"
    if name.startswith("umap"):
        return "umap"
    return "legacy"


def collect() -> list[dict]:
    heldout = {}
    if HELDOUT.is_file():
        heldout = json.loads(HELDOUT.read_text()).get("results", {})
    cards = []
    for rung, root in (("2m", SANDBOX / "2m-knobs"), ("6250k", SANDBOX / "6250k-knobs")):
        if not root.is_dir():
            continue
        for arm_dir in sorted(root.iterdir()):
            s_path, png = arm_dir / "summary.json", arm_dir / "density.png"
            if not (s_path.is_file() and png.is_file()):
                continue
            s = json.loads(s_path.read_text())
            h = heldout.get(arm_dir.name, {}) if rung == "2m" else {}
            cards.append({
                "name": arm_dir.name, "rung": rung, "png": png,
                "group": classify(arm_dir.name, rung),
                "quick_ffr": s.get("quick_ffr_at_0.1pct"),
                "dose": s.get("draws_per_edge"),
                "heldout_ffr": h.get("heldout_ffr"),
                "net_minus_regressor": h.get("net_minus_regressor"),
                "r10": h.get("r10_over_map_radius_median")
                       or s.get("r10_over_map_radius_median"),
                "tissue": h.get("low_density_mass_fraction"),
            })
    cuml_png = Path.home() / ".agent/basemap-maps/sandbox/cuml-1m/panel-0.png"
    if cuml_png.is_file():
        h = heldout.get("cuml-1m-reference", {})
        cards.append({"name": "cuML 1M (RAPIDS, non-parametric)", "rung": "1m",
                      "png": cuml_png, "group": "references",
                      "quick_ffr": None, "dose": None, "heldout_ffr": None,
                      "net_minus_regressor": None,
                      "r10": h.get("r10_over_map_radius_median"),
                      "tissue": h.get("low_density_mass_fraction")})
    sealed = Path("/data/latent-basemap/render-cache"
                  "/round-0217-minilm-mixed-2m-map-seed42-low-dose-v1/density.png")
    if sealed.is_file():
        cards.append({"name": "sealed R0217 seed 42 (registered recipe)",
                      "rung": "2m", "png": sealed, "group": "references",
                      "quick_ffr": 0.2764, "dose": 0.6782, "heldout_ffr": None,
                      "net_minus_regressor": None, "r10": 0.00012, "tissue": None})
    return cards


def fmt(v, digits=4):
    return f"{v:.{digits}f}" if isinstance(v, float) else "—"


def pareto_svg(cards: list[dict]) -> str:
    """Held-out FFR (y, up = better) vs collapse r10/R (x, log; right = healthier
    neighborhoods, far right = diffuse). quick-FFR fallback when heldout absent."""
    import math
    pts = [(c["r10"], c["heldout_ffr"] or c["quick_ffr"], c["name"], c["group"])
           for c in cards if c["r10"] and (c["heldout_ffr"] or c["quick_ffr"])]
    if len(pts) < 3:
        return ""
    xs = [math.log10(p[0]) for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = min(xs) - 0.15, max(xs) + 0.15
    y0, y1 = min(ys) - 0.02, max(ys) + 0.02
    W, H, M = 640, 380, 48
    sx = lambda x: M + (x - x0) / (x1 - x0) * (W - 2 * M)
    sy = lambda y: H - M - (y - y0) / (y1 - y0) * (H - 2 * M)
    colors = {"legacy": "#b25b4c", "umap": "#2b6cb0", "gcauchy": "#2f855a",
              "spectrum": "#805ad5", "references": "#666", "6250k": "#b7791f"}
    dots = "".join(
        f'<circle cx="{sx(math.log10(x)):.1f}" cy="{sy(y):.1f}" r="5" '
        f'fill="{colors.get(g, "#333")}" opacity="0.85"><title>{html.escape(n)}: '
        f'ffr {y:.4f}, r10/R {x:.5f}</title></circle>'
        f'<text x="{sx(math.log10(x)) + 7:.1f}" y="{sy(y) + 3:.1f}" font-size="9" '
        f'fill="#555">{html.escape(n[:18])}</text>'
        for x, y, n, g in pts)
    legend = "".join(
        f'<circle cx="{M + i * 110}" cy="16" r="5" fill="{c}"/>'
        f'<text x="{M + i * 110 + 9}" y="20" font-size="11">{g}</text>'
        for i, (g, c) in enumerate(colors.items()) if any(p[3] == g for p in pts))
    return (f'<svg viewBox="0 0 {W} {H}" style="max-width:100%;background:#fafafa;'
            f'border:1px solid #eee;border-radius:6px">{legend}'
            f'<line x1="{M}" y1="{H-M}" x2="{W-M}" y2="{H-M}" stroke="#999"/>'
            f'<line x1="{M}" y1="{M}" x2="{M}" y2="{H-M}" stroke="#999"/>'
            f'<text x="{W//2}" y="{H-10}" font-size="12" text-anchor="middle">'
            f'collapse r10/R (log; left = beads, right = diffuse)</text>'
            f'<text x="14" y="{H//2}" font-size="12" transform="rotate(-90 14 {H//2})" '
            f'text-anchor="middle">fidelity (held-out FFR, else quick-FFR)</text>{dots}</svg>')


def main() -> int:
    cards = collect()
    SITE.mkdir(parents=True, exist_ok=True)
    sections = []
    for key, title, blurb in GROUPS:
        group = [c for c in cards if c["group"] == key]
        if not group:
            continue
        figs = []
        for c in group:
            name = f"{c['rung']}-{c['name'].replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}.png"
            shutil.copy2(c["png"], SITE / name)
            caption = (f"quick {fmt(c['quick_ffr'])} · heldout {fmt(c['heldout_ffr'])} · "
                       f"net−reg {fmt(c['net_minus_regressor'])} · r10/R {fmt(c['r10'], 5)} · "
                       f"tissue {fmt(c['tissue'])}")
            figs.append(
                f'<figure style="margin:0"><img src="{name}" loading="lazy" '
                f'style="width:100%;border:1px solid #ddd;border-radius:6px">'
                f'<figcaption><b>{html.escape(c["name"])}</b>'
                f'<br><small>{caption}</small></figcaption></figure>')
        sections.append(
            f"<h2>{html.escape(title)}</h2><p><small>{html.escape(blurb)}</small></p>"
            f'<div style="display:grid;grid-template-columns:repeat(auto-fill,'
            f'minmax(300px,1fr));gap:14px">{"".join(figs)}</div>')

    rows = "".join(
        f"<tr><td>{html.escape(c['name'])}</td><td>{c['rung']}</td><td>{c['group']}</td>"
        f'<td style="text-align:right">{fmt(c["dose"], 3)}</td>'
        f'<td style="text-align:right">{fmt(c["quick_ffr"])}</td>'
        f'<td style="text-align:right">{fmt(c["heldout_ffr"])}</td>'
        f'<td style="text-align:right">{fmt(c["net_minus_regressor"])}</td>'
        f'<td style="text-align:right">{fmt(c["r10"], 5)}</td>'
        f'<td style="text-align:right">{fmt(c["tissue"])}</td></tr>'
        for c in sorted(cards, key=lambda c: (c["group"], c["name"])))

    (SITE / "index.html").write_text(
        '<!doctype html><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>kernel exploration</title>"
        '<body style="font-family:system-ui;max-width:1360px;margin:2rem auto;padding:0 1rem">'
        "<h1>Kernel exploration (sandbox)</h1>"
        "<p>Same sealed graph, different rendering law. Every arm is the rung's "
        "registered treatment with only the named kernel/dose knobs changed. "
        "Metrics: quick-FFR (sealed edges as truth), held-out FFR + regressor "
        "guard (20k sealed reserve rows), r10/R collapse, tissue mass. "
        "See experiments/sandbox/PLAN3-kernels.md.</p>"
        "<h2>Fidelity vs collapse</h2>" + pareto_svg(cards)
        + "".join(sections)
        + "<h2>All metrics</h2>"
        '<div style="overflow-x:auto"><table style="border-collapse:collapse" '
        'border="0" cellpadding="6"><tr><th>arm</th><th>rung</th><th>family</th>'
        "<th>dose</th><th>quick ffr</th><th>heldout ffr</th><th>net−reg</th>"
        f"<th>r10/R</th><th>tissue</th></tr>{rows}</table></div>")
    print(f"kernel page: {SITE}/index.html "
          f"(http://gsv.local:8800/basemap-maps/sandbox/kernels/) — {len(cards)} cards")
    return 0


if __name__ == "__main__":
    sys.exit(main())
