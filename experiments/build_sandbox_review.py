#!/usr/bin/env python3
"""build_sandbox_review.py — the sandbox arm-review page (owner request 2026-08-20).

One page over EVERY sandbox arm (all rungs) so the owner can pick one or two
arms to explore further — including the ones that looked better aesthetically
but scored below the promoted recipe on quick-FFR. Cards show the density
render next to the knobs + metrics; a shortlist star and a per-arm notes box
persist in localStorage, and an export button copies the shortlist as JSON.

CPU-only, no recomputation: renders + metrics come from each arm's existing
density.png + summary.json. Output: ~/.agent/basemap-maps/sandbox/index.html
(replaces the old per-rung landing; the per-rung pages remain untouched).
"""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

SANDBOX = Path("/data/latent-basemap/sandbox")
OUT = Path.home() / ".agent/basemap-maps/sandbox"
IMG = OUT / "img"

RUNGS = [("2m-knobs", "2M"), ("6250k-knobs", "6.25M"),
         ("12500k-knobs", "12.5M"), ("25000k-knobs", "25M"),
         ("500k-crosscheck", "500K")]

#: technique groups, in page order: (title, blurb, matcher on the arm name).
#: First match wins, so put the specific families before the generic ones.
GROUPS = [
    ("umap-0.6dev sweep",
     "rank-window hard negatives / tanh repulsion cap / kernel annealing "
     "(plan §3; upstream review in the session scratchpad)",
     lambda n: "rankneg" in n or "tanh" in n or "anneal" in n),
    ("aesthetic × fneg cross",
     "owner picks 2026-08-21: the looser kernels re-run WITH the promoted "
     "fneg mechanism",
     lambda n: n in {"umap-md005-x2-fneg10", "umap-md020-x2-fneg10",
                     "gc-a2-md000-x2-fneg10", "gc-a2-md005-x2-fneg10",
                     "gc-a2-md020-x2-fneg10"}),
    ("500K upstream cross-check",
     "same 500K rows, same induced 2M-exact-graph truth, same instrument: "
     "umap-learn 0.6dev (CPU, saw only the 500K) vs our 2M-trained maps "
     "sliced to those rows",
     lambda n: n == "upstream-06dev" or n.endswith("-slice")),
    ("external baselines",
     "published competitors run on our substrate, scored on our instruments "
     "(sandbox reads, not the pre-registered baseline rounds)",
     lambda n: "paramrepulsor" in n),
    ("far-negative band (fneg)",
     "the promoted mechanism: mid-range negatives' BCE up-weighted",
     lambda n: "fneg" in n),
    ("3D output", "PLAN7 phase A", lambda n: "-3d" in n),
    ("host-int8 residency", "X as int8 rows + fp16 scales in host RAM",
     lambda n: "hostint8" in n),
    ("density weighting", "PLAN4 density-targeted loss terms",
     lambda n: "-dw" in n),
    ("mid-near pairs", "PaCMAP-style mid-near attraction",
     lambda n: "-mn" in n),
    ("gcauchy kernel", "heavier-tailed low-dim kernel (alpha != 1)",
     lambda n: n.startswith("gc-")),
    ("min_dist ladder", "umap kernel (a, b) fits across min_dist",
     lambda n: n.startswith("umap-md") or n.startswith("umap-mind")),
    ("pos-ratio", "positive fraction of the batch",
     lambda n: "pos0" in n or "pos1" in n),
    ("baseline & dose", "replay baseline, dose multiples, raw kernel knobs",
     lambda n: True),
]


def group_of(name: str) -> str:
    for title, _, match in GROUPS:
        if match(name):
            return title
    return GROUPS[-1][0]

#: arms on the promoted-recipe lineage (context badge, not a judgment)
PROMOTED_HINT = ("fneg10",)

#: quick-FFR of the promoted 2M cell, shown as the reference line
REFERENCES = [
    ("2M promoted (x4 fneg) — sandbox twin", "2m-knobs/umap-md000-x4-fneg10"),
    ("2M sealed R0265 seed 42", "gallery:2m-seed-42.png"),
]


def load_arms() -> list[dict]:
    arms = []
    for rung_dir, rung_label in RUNGS:
        for arm_dir in sorted((SANDBOX / rung_dir).iterdir()):
            sj = arm_dir / "summary.json"
            if not sj.exists():
                continue
            s = json.loads(sj.read_text())
            arms.append({
                "id": f"{rung_dir}/{arm_dir.name}",
                "rung": rung_label,
                "name": arm_dir.name,
                "png": arm_dir / "density.png",
                "extra_pngs": sorted(arm_dir.glob("density-??.png")),
                "ffr": s.get("quick_ffr_at_0.1pct"),
                "spacing": s.get("r10_over_map_radius_median"),
                "overrides": s.get("overrides") or {},
                "dose": s.get("dose_multiplier"),
                "seed": s.get("seed"),
                "wall_s": s.get("wall_s"),
                "n_components": s.get("n_components", 2),
                "promoted": any(h in arm_dir.name for h in PROMOTED_HINT),
            })
    return arms


def knobs_line(a: dict) -> str:
    parts = [f"{k}={v}" for k, v in a["overrides"].items()]
    if a.get("dose"):
        parts.append(f"dose x{a['dose']}")
    if a.get("seed") not in (None, 42):
        parts.append(f"seed {a['seed']}")
    if a.get("n_components", 2) != 2:
        parts.append(f"{a['n_components']}D")
    return ", ".join(parts) or "baseline knobs"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(exist_ok=True)
    arms = load_arms()

    cards_by_group: dict[str, list[str]] = {}
    n_cards = 0
    for a in arms:
        slug = a["id"].replace("/", "__")
        imgs = []
        for p in ([a["png"]] + a["extra_pngs"]):
            if p and p.exists():
                dest = IMG / f"{slug}--{p.name}"
                if not dest.exists():
                    shutil.copy2(p, dest)
                imgs.append(f"img/{dest.name}")
        if not imgs:
            continue
        ffr = f"{a['ffr']:.4f}" if a["ffr"] is not None else "—"
        spacing = f"{a['spacing']:.5f}" if a["spacing"] is not None else "—"
        wall = f"{a['wall_s']/60:.0f} min" if a["wall_s"] else "—"
        img_tags = "".join(
            f'<img src="{u}" loading="lazy" class="{"secondary" if i else "primary"}">'
            for i, u in enumerate(imgs))
        badge = ('<span class="badge lineage">promoted lineage</span>'
                 if a["promoted"] else "")
        if a["n_components"] != 2:
            badge += '<span class="badge d3">3D</span>'
        cards_by_group.setdefault(group_of(a["name"]), []).append(f"""
<figure class="card" data-id="{html.escape(a['id'])}" data-rung="{a['rung']}"
        data-ffr="{a['ffr'] if a['ffr'] is not None else -1}">
  <div class="imgs">{img_tags}</div>
  <figcaption>
    <div class="head"><b>{html.escape(a['name'])}</b>
      <span class="rung">{a['rung']}</span>{badge}
      <button class="star" title="shortlist">☆</button></div>
    <div class="knobs">{html.escape(knobs_line(a))}</div>
    <div class="mets">quick-FFR <b>{ffr}</b> · spacing {spacing} · {wall}</div>
    <textarea class="note" placeholder="notes…"></textarea>
  </figcaption>
</figure>""")
        n_cards += 1

    ref_html = ""
    twin = next((a for a in arms if a["id"] == "2m-knobs/umap-md000-x4-fneg10"), None)
    if twin:
        ref_html = (f"<p class='ref'>Reference: the promoted 2M recipe scores "
                    f"quick-FFR <b>{twin['ffr']:.4f}</b> in this sandbox "
                    f"(sealed R0265 renders in the <a href='../gallery/'>gallery</a>). "
                    f"Anything below that traded metric for looks — that trade is "
                    f"what this page is for judging.</p>")

    def slug(title: str) -> str:
        return "".join(c if c.isalnum() else "-" for c in title.lower())

    toc_items, sections = [], []
    for title, blurb, _ in GROUPS:
        group_cards = cards_by_group.get(title)
        if not group_cards:
            continue
        sid = slug(title)
        toc_items.append(f'<a href="#{sid}">{html.escape(title)} '
                         f'({len(group_cards)})</a>')
        sections.append(
            f'<section id="{sid}"><h2>{html.escape(title)} '
            f'<small>{len(group_cards)}</small></h2>'
            f'<p class="blurb">{html.escape(blurb)}</p>'
            f'<div class="grid">{"".join(group_cards)}</div></section>')
    toc_html = f'<nav class="toc">{" · ".join(toc_items)}</nav>'

    (OUT / "index.html").write_text(f"""<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>sandbox arm review — pick maps to explore</title>
<style>
 body {{ font-family: system-ui; max-width: 1560px; margin: 1.5rem auto; padding: 0 1rem; color: #1a202c; }}
 h1 {{ margin: 0 0 .2rem }} .sub {{ color: #4a5568; margin-top: 0 }}
 .ref {{ background: #f7fafc; border-left: 3px solid #2b6cb0; padding: .5rem .8rem }}
 .toc {{ margin: .8rem 0; line-height: 1.9 }}
 .toc a {{ background: #edf2f7; border-radius: 4px; padding: 2px 8px; margin-right: 4px;
           text-decoration: none; color: #2d3748; font-size: .85rem; white-space: nowrap }}
 .toc a:hover {{ background: #e2e8f0 }}
 section {{ margin: 1.6rem 0 }}
 section h2 {{ border-bottom: 2px solid #e2e8f0; padding-bottom: .25rem; margin-bottom: .25rem }}
 section h2 small {{ color: #a0aec0; font-weight: normal }}
 .blurb {{ color: #4a5568; margin: .2rem 0 .8rem; font-size: .9rem }}
 .bar {{ display: flex; gap: .8rem; align-items: center; margin: 1rem 0; flex-wrap: wrap }}
 .bar select, .bar button {{ font-size: .9rem; padding: .25rem .5rem }}
 .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); gap: 16px }}
 .card {{ margin: 0; border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; background: #fff }}
 .card.starred {{ border-color: #d69e2e; box-shadow: 0 0 0 2px #f6e05e55 }}
 .imgs img.primary {{ width: 100%; display: block }}
 .imgs img.secondary {{ width: 33.33%; display: inline-block }}
 figcaption {{ padding: .45rem .6rem }}
 .head {{ display: flex; gap: .5rem; align-items: baseline }}
 .head b {{ font-size: .95rem }}
 .rung {{ color: #4a5568; font-size: .8rem }}
 .badge {{ color: #fff; border-radius: 4px; padding: 1px 6px; font-size: .68rem }}
 .badge.lineage {{ background: #2b6cb0 }} .badge.d3 {{ background: #6b46c1 }}
 .star {{ margin-left: auto; background: none; border: none; font-size: 1.15rem; cursor: pointer; color: #a0aec0 }}
 .card.starred .star {{ color: #d69e2e }}
 .knobs {{ color: #4a5568; font-size: .82rem; margin-top: .15rem }}
 .mets {{ font-size: .85rem; margin-top: .2rem }}
 .note {{ width: 100%; margin-top: .35rem; font-size: .8rem; border: 1px solid #e2e8f0;
          border-radius: 4px; padding: .3rem; min-height: 1.6rem; resize: vertical }}
 #exported {{ white-space: pre-wrap; background: #f7fafc; padding: .6rem; border-radius: 6px;
              display: none; font-family: monospace; font-size: .8rem }}
</style>
<h1>Sandbox arm review</h1>
<p class="sub">Every sandbox arm across 2M–25M, renders next to knobs + metrics. Star the ones
worth exploring further; notes and stars persist in this browser. All arms retain
<code>model.pt</code> + coordinates on disk, so any pick can be revived (bigger renders,
interactive pack, scale-up) without retraining.</p>
{ref_html}
<div class="bar">
  <label>sort <select id="sort">
    <option value="name">name</option>
    <option value="ffr-desc">quick-FFR ↓</option>
    <option value="ffr-asc">quick-FFR ↑</option>
  </select></label>
  <label>rung <select id="rung">
    <option value="">all</option><option>2M</option><option>6.25M</option>
    <option>12.5M</option><option>25M</option>
  </select></label>
  <label><input type="checkbox" id="onlystars"> shortlist only</label>
  <button id="export">export shortlist</button>
  <span id="count"></span>
</div>
{toc_html}
<div id="exported"></div>
{"".join(sections)}
<script>
const LS = "sandbox-review-v1";
const state = JSON.parse(localStorage.getItem(LS) || '{{"stars":{{}},"notes":{{}}}}');
const save = () => localStorage.setItem(LS, JSON.stringify(state));
const cards = [...document.querySelectorAll(".card")];
for (const c of cards) {{
  const id = c.dataset.id;
  const star = c.querySelector(".star"), note = c.querySelector(".note");
  const paint = () => {{
    c.classList.toggle("starred", !!state.stars[id]);
    star.textContent = state.stars[id] ? "★" : "☆";
  }};
  note.value = state.notes[id] || "";
  star.onclick = () => {{ state.stars[id] = !state.stars[id]; save(); paint(); apply(); }};
  note.oninput = () => {{ state.notes[id] = note.value; save(); }};
  paint();
}}
function apply() {{
  const rung = document.getElementById("rung").value;
  const only = document.getElementById("onlystars").checked;
  const sort = document.getElementById("sort").value;
  let vis = 0;
  for (const sec of document.querySelectorAll("section")) {{
    const grid = sec.querySelector(".grid");
    const secCards = [...grid.querySelectorAll(".card")].sort((a, b) => {{
      if (sort === "name") return a.dataset.id.localeCompare(b.dataset.id);
      const fa = +a.dataset.ffr, fb = +b.dataset.ffr;
      return sort === "ffr-desc" ? fb - fa : fa - fb;
    }});
    let secVis = 0;
    for (const c of secCards) {{
      grid.appendChild(c);
      const show = (!rung || c.dataset.rung === rung) && (!only || state.stars[c.dataset.id]);
      c.style.display = show ? "" : "none";
      if (show) secVis++;
    }}
    sec.style.display = secVis ? "" : "none";
    const toc = document.querySelector(`.toc a[href="#${{sec.id}}"]`);
    if (toc) toc.style.display = secVis ? "" : "none";
    vis += secVis;
  }}
  document.getElementById("count").textContent = vis + " arms";
}}
for (const id of ["sort", "rung", "onlystars"]) document.getElementById(id).onchange = apply;
document.getElementById("export").onclick = () => {{
  const out = Object.keys(state.stars).filter(k => state.stars[k])
    .map(id => ({{ id, note: state.notes[id] || "" }}));
  const el = document.getElementById("exported");
  el.style.display = "block";
  el.textContent = JSON.stringify(out, null, 1);
  navigator.clipboard && navigator.clipboard.writeText(el.textContent);
}};
apply();
</script>
""")
    print(f"{n_cards} arm cards in {len(sections)} groups -> {OUT}/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
