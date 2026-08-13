"""The kernel program's dedicated visual + metric comparison page.

Collects every sandbox arm (2M and 6.25M rungs), the cuML reference, and the
sealed legacy baseline onto one page at
``http://gsv.local:8800/basemap-maps/sandbox/kernels/``: an ALIGNED render
grid grouped by kernel family, the full metric table, and a Pareto scatter.

Alignment (owner request 2026-08-13): maps of one rung share row indexing, so
each map is similarity-Procrustes-aligned (translation/rotation/reflection/
scale — the same ``_procrustes`` math the compare pages use) onto the rung's
reference map, fitted on a 200k shared-row sample and applied to all rows,
then binned on the REFERENCE's extent. Rotated-looking siblings land in one
frame. The cuML card aligns through its known row subset of the 2M substrate.

Aligned renders are cached at
``/data/latent-basemap/sandbox/.aligned-cache/`` keyed on the coordinate
file's identity and the reference's; re-running is cheap after first build.
"""

from __future__ import annotations

import html
import json
import shutil
import sys
from pathlib import Path

import numpy as np

SANDBOX = Path("/data/latent-basemap/sandbox")
SITE = Path.home() / ".agent/basemap-maps/sandbox/kernels"
HELDOUT = SANDBOX / "2m-knobs/heldout-eval.json"
CACHE = SANDBOX / ".aligned-cache"
REGISTRY = Path("/data/latent-basemap/maps.json")
CACHE_VERSION = 1
FIT_SAMPLE = 200_000

EXPERIMENTS = Path(__file__).resolve().parents[1]
for p in (str(EXPERIMENTS), str(EXPERIMENTS / "sandbox")):
    if p not in sys.path:
        sys.path.insert(0, p)

GROUPS = [
    ("references", "References",
     "cuML (non-parametric target look) and the sealed legacy baseline, "
     "aligned into the 2M frame."),
    ("legacy", "legacy_lp family",
     "The registered kernel: 1/(1+a·‖Δ‖_2b). Collapses regardless of dose/a/b."),
    ("umap", "umap kernel — min_dist sweep",
     "1/(1+a·r^2b), (a,b) fitted per min_dist. The plateau radius is the "
     "diffuseness dial: md000 tight → md050 diffuse."),
    ("gcauchy", "gcauchy — tail exponent",
     "(1+a·r^2b)^(-α). α=1 is the umap kernel; α<1 heavier tail, α>1 lighter."),
    ("spectrum", "attraction/repulsion spectrum probes",
     "pos_ratio moved at fixed horizon; dose covaries (documented per card)."),
    ("fog", "fog program (PLAN4)",
     "Mechanisms aimed at the clean-AND-healthy corner: mid-near pairs (mn), "
     "density term (dw), fog-targeted negatives (fneg), on the winner config."),
    ("6250k", "6.25M rung checks",
     "Winning configs at 3.1x scale on the sealed R0233 graph; compare "
     "against R0257's legacy maps."),
]


def classify(name: str, rung: str) -> str:
    if rung == "6250k":
        return "6250k"
    if any(tag in name for tag in ("-mn", "-dw", "-fneg")):
        return "fog"
    if name.startswith("gc-"):
        return "gcauchy"
    if "pos02" in name or "pos15" in name:
        return "spectrum"
    if name.startswith("umap"):
        return "umap"
    return "legacy"


def _registry_coords(map_id: str) -> Path | None:
    try:
        registry = json.loads(REGISTRY.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    for m in registry["maps"]:
        if m.get("map_id") == map_id:
            p = Path(((m.get("coordinates") or {}).get("file") or "").removeprefix("gsv:"))
            return p if p.is_file() else None
    return None


def collect() -> list[dict]:
    heldout = {}
    if HELDOUT.is_file():
        heldout = json.loads(HELDOUT.read_text()).get("results", {})
    cards = []
    for rung, root in (("2m", SANDBOX / "2m-knobs"), ("6250k", SANDBOX / "6250k-knobs")):
        if not root.is_dir():
            continue
        for arm_dir in sorted(root.iterdir()):
            s_path, coords = arm_dir / "summary.json", arm_dir / "coordinates.npy"
            if not (s_path.is_file() and coords.is_file()):
                continue
            s = json.loads(s_path.read_text())
            h = heldout.get(arm_dir.name, {}) if rung == "2m" else {}
            cards.append({
                "name": arm_dir.name, "rung": rung, "coords": coords, "rows": None,
                "group": classify(arm_dir.name, rung),
                "quick_ffr": s.get("quick_ffr_at_0.1pct"),
                "dose": s.get("draws_per_edge"),
                "heldout_ffr": h.get("heldout_ffr"),
                "net_minus_regressor": h.get("net_minus_regressor"),
                "r10": h.get("r10_over_map_radius_median")
                       or s.get("r10_over_map_radius_median"),
                "tissue": h.get("low_density_mass_fraction"),
            })
    sealed = _registry_coords("round-0217-minilm-mixed-2m-map-seed42-low-dose-v1")
    if sealed is not None:
        cards.append({"name": "sealed R0217 seed 42 (registered recipe)",
                      "rung": "2m", "coords": sealed, "rows": None,
                      "group": "references", "quick_ffr": 0.2764, "dose": 0.6782,
                      "heldout_ffr": None, "net_minus_regressor": None,
                      "r10": 0.00011, "tissue": None})
    cuml_xy = SANDBOX / "cuml-1m/cuml-xy.npy"
    cuml_rows = SANDBOX / "cuml-1m/rows.npy"
    if cuml_xy.is_file() and cuml_rows.is_file():
        h = heldout.get("cuml-1m-reference", {})
        cards.append({"name": "cuML 1M (RAPIDS, non-parametric)", "rung": "2m",
                      "coords": cuml_xy, "rows": cuml_rows, "group": "references",
                      "quick_ffr": None, "dose": None, "heldout_ffr": None,
                      "net_minus_regressor": None,
                      "r10": h.get("r10_over_map_radius_median"),
                      "tissue": h.get("low_density_mass_fraction")})
    return cards


REFERENCE_PREFERENCE = ("umap-md000-x4", "umap-md000-x2", "umap-dose-x2")


def pick_reference(cards: list[dict], rung: str) -> dict | None:
    rung_cards = [c for c in cards if c["rung"] == rung and c["rows"] is None]
    for name in REFERENCE_PREFERENCE:
        for c in rung_cards:
            if c["name"] == name:
                return c
    return rung_cards[0] if rung_cards else None


def fit_similarity(ref_s: np.ndarray, x_s: np.ndarray):
    """The compare pages' _procrustes math, returned as a reusable transform."""
    mu_r, mu_x = ref_s.mean(0), x_s.mean(0)
    r0, x0 = ref_s - mu_r, x_s - mu_x
    nr, nx = np.linalg.norm(r0), np.linalg.norm(x0)
    if nx == 0 or nr == 0:
        return lambda xy: xy.copy()
    u, s, vt = np.linalg.svd((x0 / nx).T @ (r0 / nr))
    rot = u @ vt
    scale = nr * s.sum() / nx
    return lambda xy: ((xy - mu_x) @ rot) * scale + mu_r


def _cache_key(card: dict, ref: dict) -> dict:
    def ident(p: Path) -> list:
        st = p.stat()
        return [str(p), st.st_size, st.st_mtime_ns]
    return {"v": CACHE_VERSION, "map": ident(card["coords"]),
            "ref": ident(ref["coords"])}


def aligned_render(card: dict, ref: dict, ref_xy: np.ndarray,
                   ref_extent: list, ref_sample_idx: np.ndarray) -> Path:
    """Render this card aligned onto the reference, cached."""
    from map_renders import binned_counts, render_png

    CACHE.mkdir(parents=True, exist_ok=True)
    slug = f"{card['rung']}-{card['name']}".replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    png = CACHE / f"{slug}.png"
    meta = CACHE / f"{slug}.json"
    key = _cache_key(card, ref)
    if png.is_file() and meta.is_file():
        try:
            if json.loads(meta.read_text()).get("key") == key:
                return png
        except (OSError, json.JSONDecodeError):
            pass

    xy = np.asarray(np.load(card["coords"], mmap_mode="r"), dtype=np.float32)
    if card is ref:
        aligned = xy
    elif card["rows"] is not None:
        # cuML: its rows index the reference's row space.
        rows = np.load(card["rows"])
        rng = np.random.default_rng(1)
        idx = rng.choice(len(rows), min(FIT_SAMPLE, len(rows)), replace=False)
        transform = fit_similarity(ref_xy[rows[idx]], xy[idx])
        aligned = transform(xy)
    else:
        transform = fit_similarity(ref_xy[ref_sample_idx], xy[ref_sample_idx])
        aligned = transform(xy)
    render_png(binned_counts(aligned, ref_extent), png)
    meta.write_text(json.dumps({"key": key, "aligned_to": ref["name"],
                                "extent": ref_extent}))
    return png


def fmt(v, digits=4):
    return f"{v:.{digits}f}" if isinstance(v, float) else "—"


def pareto_svg(cards: list[dict]) -> str:
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
              "spectrum": "#805ad5", "fog": "#c05299", "references": "#666",
              "6250k": "#b7791f"}
    dots = "".join(
        f'<circle cx="{sx(math.log10(x)):.1f}" cy="{sy(y):.1f}" r="5" '
        f'fill="{colors.get(g, "#333")}" opacity="0.85"><title>{html.escape(n)}: '
        f'ffr {y:.4f}, r10/R {x:.5f}</title></circle>'
        f'<text x="{sx(math.log10(x)) + 7:.1f}" y="{sy(y) + 3:.1f}" font-size="9" '
        f'fill="#555">{html.escape(n[:18])}</text>'
        for x, y, n, g in pts)
    legend = "".join(
        f'<circle cx="{M + i * 100}" cy="16" r="5" fill="{c}"/>'
        f'<text x="{M + i * 100 + 9}" y="20" font-size="11">{g}</text>'
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

    refs: dict[str, tuple] = {}
    rng = np.random.default_rng(1)
    for rung in ("2m", "6250k"):
        ref = pick_reference(cards, rung)
        if ref is None:
            continue
        ref_xy = np.asarray(np.load(ref["coords"], mmap_mode="r"), dtype=np.float32)
        from map_renders import robust_extent
        extent = robust_extent(ref_xy)
        sample_idx = np.sort(rng.choice(len(ref_xy),
                                        min(FIT_SAMPLE, len(ref_xy)), replace=False))
        refs[rung] = (ref, ref_xy, extent, sample_idx)

    sections = []
    for key, title, blurb in GROUPS:
        group = [c for c in cards if c["group"] == key]
        if not group:
            continue
        figs = []
        for c in group:
            if c["rung"] not in refs:
                continue
            ref, ref_xy, extent, sample_idx = refs[c["rung"]]
            png = aligned_render(c, ref, ref_xy, extent, sample_idx)
            name = png.name
            shutil.copy2(png, SITE / name)
            caption = (f"quick {fmt(c['quick_ffr'])} · heldout {fmt(c['heldout_ffr'])} · "
                       f"net−reg {fmt(c['net_minus_regressor'])} · r10/R {fmt(c['r10'], 5)} · "
                       f"tissue {fmt(c['tissue'])}")
            aligned_note = "" if c is ref else f' · aligned to {html.escape(ref["name"])}'
            figs.append(
                f'<figure style="margin:0"><img src="{name}" loading="lazy" '
                f'style="width:100%;border:1px solid #ddd;border-radius:6px">'
                f'<figcaption><b>{html.escape(c["name"])}</b>'
                f'<br><small>{caption}{aligned_note}</small></figcaption></figure>')
        if not figs:
            continue
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
        "All renders are similarity-Procrustes aligned to the rung's reference "
        "map and share its extent, so siblings sit in one frame. "
        "See experiments/sandbox/PLAN3-kernels.md and PLAN4-fog.md.</p>"
        "<h2>Fidelity vs collapse</h2>" + pareto_svg(cards)
        + "".join(sections)
        + "<h2>All metrics</h2>"
        '<div style="overflow-x:auto"><table style="border-collapse:collapse" '
        'border="0" cellpadding="6"><tr><th>arm</th><th>rung</th><th>family</th>'
        "<th>dose</th><th>quick ffr</th><th>heldout ffr</th><th>net−reg</th>"
        f"<th>r10/R</th><th>tissue</th></tr>{rows}</table></div>")
    print(f"kernel page: {SITE}/index.html "
          f"(http://gsv.local:8800/basemap-maps/sandbox/kernels/) — {len(cards)} cards, aligned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
