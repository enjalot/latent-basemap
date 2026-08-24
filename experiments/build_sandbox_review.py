#!/usr/bin/env python3
"""build_sandbox_review.py — the sandbox arm-review page.

2026-08-24 overhaul (owner): per-card date/time (ET), a fixed 2-column
parameter grid identical across all cards (n/a shown as –), min_dist as a
first-class derived property, optional grouping (technique vs flat list),
DATASET-FIRST group assignment (arm names repeat across datasets — the
redditmix-in-images bug class), and an initiative-based taxonomy for the
MiniLM rungs replacing the overgrown min_dist-ladder catch-all.

CPU-only, no recomputation. Output: ~/.agent/basemap-maps/sandbox/index.html
"""

from __future__ import annotations

import datetime
import html
import json
import shutil
import zoneinfo
from pathlib import Path

SANDBOX = Path("/data/latent-basemap/sandbox")
OUT = Path.home() / ".agent/basemap-maps/sandbox"
IMG = OUT / "img"
ET = zoneinfo.ZoneInfo("America/New_York")

#: (dir, rung label, embedding model, corpus)
RUNGS = [
    ("2m-knobs", "2M", "MiniLM", "mixed-4"),
    ("6250k-knobs", "6.25M", "MiniLM", "mixed-4"),
    ("12500k-knobs", "12.5M", "MiniLM", "mixed-4"),
    ("25000k-knobs", "25M", "MiniLM", "mixed-4"),
    ("500k-crosscheck", "500K", "MiniLM", "mixed-4"),
    ("bl-siglip-1m", "BL 1.08M", "SigLIP2", "BL books"),
    ("sisap-clip-2m", "LAION 2M", "CLIP ViT-L/14", "LAION"),
    ("sisap-clip-2m-dedup", "LAION 2M dedup", "CLIP ViT-L/14", "LAION dedup"),
    ("jina-en-2m", "jina EN 2M", "jina-v5-nano", "mixed-3 EN"),
    ("jina-multi-2m", "jina multi 2M", "jina-v5-nano", "EN+20 langs"),
    ("jina-multi-6m", "jina multi 6.25M", "jina-v5-nano", "EN+20 langs"),
    ("minilm-redditmix-2m", "redditmix 2M", "MiniLM", "mixed-4+reddit"),
    ("distill-grid", "capacity grid", "MiniLM", "mixed-4"),
]

#: DATASET-FIRST assignment: non-MiniLM-knob dirs group by what they ARE.
DATASET_GROUPS = {
    "bl-siglip-1m": "image embeddings (BL SigLIP)",
    "sisap-clip-2m": "image embeddings (LAION CLIP)",
    "sisap-clip-2m-dedup": "image embeddings (LAION CLIP)",
    "jina-en-2m": "jina v5 nano (768-d text)",
    "jina-multi-2m": "jina v5 nano (768-d text)",
    "jina-multi-6m": "jina v5 nano (768-d text)",
    "minilm-redditmix-2m": "register mixture (redditmix)",
    "500k-crosscheck": "500K upstream cross-check",
    "distill-grid": "capacity-vs-scale grid",
}

#: aesthetic-cross explicit names (checked before the mechanism matchers)
_CROSS = {"umap-md005-x2-fneg10", "umap-md020-x2-fneg10",
          "gc-a2-md000-x2-fneg10", "gc-a2-md005-x2-fneg10",
          "gc-a2-md020-x2-fneg10", "umap-md005-x2-fneg10-tanh4",
          "umap-md010-x2-fneg10-tanh4"}

#: MiniLM-rung name taxonomy, first match wins, initiative-ordered.
NAME_GROUPS = [
    ("teacher distillation & multilevel",
     "distill / distill-init / multilevel-init experiments",
     lambda n: n.startswith(("distill-", "distillinit-", "mlinit-"))),
    ("external baselines",
     "upstream 0.6dev + ParamRepulsor on our substrates/instruments",
     lambda n: n.startswith("upstream-") or "paramrepulsor" in n),
    ("architecture sweep",
     "width/depth/arch under the x2 composed core",
     lambda n: n.startswith(("core-h", "core-L")) or n == "core-mlp"),
    ("composition screen",
     "x2 composed core + each previously-rejected lever",
     lambda n: "-core" in n),
    ("aesthetic & kernel cross",
     "looser kernels x promoted mechanisms (incl. the anti-collapse looks)",
     lambda n: n in _CROSS),
    ("champion & efficiency program",
     "the 2026-08-22/24 initiative: dose/tanh/pos factorial, winners, bs16k, "
     "width economics, scaling diagnostics",
     lambda n: ("winner" in n or "bs16k" in n or "pos10" in n
                or "tanh4-pos02" in n or "-x8-" in n or "-x16-" in n
                or "x4-fneg10-tanh" in n or n.endswith("-probt")
                or n.endswith("-fneg10-wes"))),
    ("umap-0.6dev mechanism sweep",
     "rank-window negatives / tanh cap / annealing ports at x2",
     lambda n: "rankneg" in n or "tanh" in n or "anneal" in n),
    ("3D output", "PLAN7 phase A", lambda n: "-3d" in n),
    ("host-int8 residency", "X as int8 in host RAM", lambda n: "hostint8" in n),
    ("density weighting", "PLAN4 density terms", lambda n: "-dw" in n),
    ("mid-near pairs", "PaCMAP-style mid-near", lambda n: "-mn" in n),
    ("far-negative band (fneg)",
     "the promoted mechanism era", lambda n: "fneg" in n),
    ("gcauchy kernel", "heavier-tailed kernel", lambda n: n.startswith("gc-")),
    ("min_dist ladder", "the original kernel-fit sweep",
     lambda n: n.startswith(("umap-md", "umap-mind"))),
    ("pos-ratio (legacy)", "early positive-fraction probes",
     lambda n: n.startswith("umap-pos")),
    ("baseline & dose", "replay baseline, dose multiples, raw kernel knobs",
     lambda n: True),
]

GROUP_ORDER = [
    "champion & efficiency program", "capacity-vs-scale grid",
    "teacher distillation & multilevel", "umap-0.6dev mechanism sweep",
    "composition screen", "architecture sweep", "aesthetic & kernel cross",
    "register mixture (redditmix)", "jina v5 nano (768-d text)",
    "image embeddings (LAION CLIP)", "image embeddings (BL SigLIP)",
    "500K upstream cross-check", "external baselines",
    "far-negative band (fneg)", "3D output", "host-int8 residency",
    "density weighting", "mid-near pairs", "gcauchy kernel",
    "min_dist ladder", "pos-ratio (legacy)", "baseline & dose",
]
GROUP_BLURBS = {t: b for t, b, _ in NAME_GROUPS}
GROUP_BLURBS.update({
    "image embeddings (BL SigLIP)": "recipes on the BL SigLIP2 space",
    "image embeddings (LAION CLIP)": "recipes on LAION CLIP768 (incl. dedup)",
    "jina v5 nano (768-d text)": "recipes on the prompted jina spaces",
    "register mixture (redditmix)": "the 80/20 reddit-mix substrate",
    "500K upstream cross-check": "same rows, same truth: upstream vs sliced",
    "capacity-vs-scale grid": "distill capacity meter: width x scale",
})

#: umap-kernel a-value -> min_dist label
A_TO_MD = {1.9328: "0.0", 1.8404: "0.025", 1.7502: "0.05", 1.5769: "0.1",
           1.2621: "0.2", 0.8741: "0.35", 0.583: "0.5"}

PROMOTED_HINT = ("fneg10",)


def group_of(rung_dir: str, name: str) -> str:
    if rung_dir in DATASET_GROUPS:
        return DATASET_GROUPS[rung_dir]
    for title, _, match in NAME_GROUPS:
        if match(name):
            return title
    return "baseline & dose"


def load_arms() -> list[dict]:
    arms = []
    for rung_dir, rung_label, model, corpus in RUNGS:
        if not (SANDBOX / rung_dir).is_dir():
            continue
        for arm_dir in sorted((SANDBOX / rung_dir).iterdir()):
            sj = arm_dir / "summary.json"
            if not sj.exists():
                continue
            s = json.loads(sj.read_text())
            ts = None
            if s.get("started_utc"):
                try:
                    ts = datetime.datetime.fromisoformat(
                        s["started_utc"]).timestamp()
                except ValueError:
                    ts = None
            if ts is None:
                ts = sj.stat().st_mtime
            arms.append({
                "ts": ts,
                "id": f"{rung_dir}/{arm_dir.name}",
                "rung_dir": rung_dir,
                "rung": rung_label,
                "model": model,
                "corpus": corpus,
                "name": arm_dir.name,
                "png": arm_dir / "density.png",
                "extra_pngs": sorted(arm_dir.glob("density-??.png")),
                "ffr": s.get("quick_ffr_at_0.1pct"),
                "spacing": s.get("r10_over_map_radius_median"),
                "overrides": s.get("overrides") or {},
                "summary": s,
                "dose": s.get("dose_multiplier"),
                "seed": s.get("seed"),
                "wall_s": s.get("wall_s"),
                "n_components": s.get("n_components", 2),
                "promoted": any(h in arm_dir.name for h in PROMOTED_HINT),
            })
    return arms


def param_rows(a: dict) -> list[tuple[str, str]]:
    """The fixed property set, identical order on every card."""
    o = a["overrides"]
    s = a["summary"]

    def get(*keys):
        for k in keys:
            if k in o:
                return o[k]
            if k in s:
                return s[k]
        return None

    kern = get("low_dim_kernel")
    alpha = get("kernel_alpha")
    if kern == "gcauchy" and alpha is not None:
        kern = f"gcauchy α={alpha}"
    md = A_TO_MD.get(round(float(o["a"]), 4)) if "a" in o else None
    dose = a.get("dose")
    wall = f"{a['wall_s']/60:.0f} min" if a.get("wall_s") else None

    def fmt(v):
        return "–" if v is None else str(v)

    return [
        ("kernel", fmt(kern)), ("min_dist", fmt(md)),
        ("dose", fmt(f"×{dose}" if dose else None)),
        ("fneg", fmt(get("fneg_weight"))),
        ("tanh γ", fmt(get("neg_tanh_gamma"))),
        ("rankneg", fmt(get("rankneg_window"))),
        ("pos_ratio", fmt(get("pos_ratio"))),
        ("batch", fmt(get("batch_size"))),
        ("width", fmt(get("hidden_dim", "width"))),
        ("layers", fmt(get("n_layers"))),
        ("arch", fmt(get("architecture", "init"))),
        ("wall", fmt(wall)),
    ]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(exist_ok=True)
    arms = load_arms()

    cards_by_group: dict[str, list[str]] = {}
    audit: dict[str, list[str]] = {}
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
        g = group_of(a["rung_dir"], a["name"])
        audit.setdefault(g, []).append(a["id"])
        ffr = f"{a['ffr']:.4f}" if a["ffr"] is not None else "—"
        when = datetime.datetime.fromtimestamp(a["ts"], tz=ET).strftime(
            "%b %d %I:%M %p ET")
        img_tags = "".join(
            f'<img src="{u}" loading="lazy" class="{"secondary" if i else "primary"}">'
            for i, u in enumerate(imgs))
        badge = ('<span class="badge lineage">promoted lineage</span>'
                 if a["promoted"] else "")
        if a["n_components"] != 2:
            badge += '<span class="badge d3">3D</span>'
        params = "".join(
            f'<div class="pk">{html.escape(k)}</div>'
            f'<div class="pv">{html.escape(v)}</div>'
            for k, v in param_rows(a))
        cards_by_group.setdefault(g, []).append(f"""
<figure class="card" data-id="{html.escape(a['id'])}" data-rung="{a['rung']}"
        data-model="{html.escape(a['model'])}" data-corpus="{html.escape(a['corpus'])}"
        data-ts="{a['ts']:.0f}" data-group="{html.escape(g)}"
        data-ffr="{a['ffr'] if a['ffr'] is not None else -1}">
  <div class="imgs">{img_tags}</div>
  <figcaption>
    <div class="head"><b>{html.escape(a['name'])}</b>
      <span class="rung">{a['rung']}</span>
      <span class="meta">{html.escape(a['model'])} · {html.escape(a['corpus'])}</span>{badge}
      <button class="star" title="shortlist">☆</button></div>
    <div class="when">{when} · quick-FFR <b>{ffr}</b></div>
    <div class="params">{params}</div>
    <textarea class="note" placeholder="notes…"></textarea>
  </figcaption>
</figure>""")
        n_cards += 1

    twin = next((a for a in arms if a["id"] == "2m-knobs/umap-md000-x4-fneg10"), None)
    ref_html = (f"<p class='ref'>Reference: promoted 2M recipe quick-FFR "
                f"<b>{twin['ffr']:.4f}</b>; champion (x8+tanh4+pos10+rankneg) "
                f"0.4600; upstream ceilings 0.4798 / 0.6779 / 0.6701 "
                f"(MiniLM / jina-multi / CLIP).</p>") if twin else ""

    def slug_of(title: str) -> str:
        return "".join(c if c.isalnum() else "-" for c in title.lower())

    toc_items, sections = [], []
    for title in GROUP_ORDER:
        group_cards = cards_by_group.pop(title, None)
        if not group_cards:
            continue
        sid = slug_of(title)
        toc_items.append(f'<a href="#{sid}">{html.escape(title)} '
                         f'({len(group_cards)})</a>')
        sections.append(
            f'<section id="{sid}"><h2>{html.escape(title)} '
            f'<small>{len(group_cards)}</small></h2>'
            f'<p class="blurb">{html.escape(GROUP_BLURBS.get(title, ""))}</p>'
            f'<div class="grid">{"".join(group_cards)}</div></section>')
    for title, group_cards in cards_by_group.items():  # anything unmapped
        sid = slug_of(title)
        sections.append(f'<section id="{sid}"><h2>{html.escape(title)}</h2>'
                        f'<div class="grid">{"".join(group_cards)}</div></section>')
    toc_html = f'<nav class="toc">{" · ".join(toc_items)}</nav>'
    model_opts = "".join(f"<option>{html.escape(m)}</option>" for m in
                         sorted({a["model"] for a in arms}))
    corpus_opts = "".join(f"<option>{html.escape(c)}</option>" for c in
                          sorted({a["corpus"] for a in arms}))

    (OUT / "index.html").write_text(f"""<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>sandbox arm review</title>
<style>
 body {{ font-family: system-ui; max-width: 1560px; margin: 1.5rem auto; padding: 0 1rem; color: #1a202c; }}
 h1 {{ margin: 0 0 .2rem }} .sub {{ color: #4a5568; margin-top: 0 }}
 .ref {{ background: #f7fafc; border-left: 3px solid #2b6cb0; padding: .5rem .8rem }}
 .toc {{ margin: .8rem 0; line-height: 1.9 }}
 .toc a {{ background: #edf2f7; border-radius: 4px; padding: 2px 8px; margin-right: 4px;
           text-decoration: none; color: #2d3748; font-size: .85rem; white-space: nowrap }}
 section {{ margin: 1.6rem 0 }}
 section h2 {{ border-bottom: 2px solid #e2e8f0; padding-bottom: .25rem; margin-bottom: .25rem }}
 section h2 small {{ color: #a0aec0; font-weight: normal }}
 .blurb {{ color: #4a5568; margin: .2rem 0 .8rem; font-size: .9rem }}
 .bar {{ display: flex; gap: .8rem; align-items: center; margin: 1rem 0; flex-wrap: wrap }}
 .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); gap: 14px }}
 .card {{ margin: 0; border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; background: #fff }}
 .card.starred {{ border-color: #d69e2e; box-shadow: 0 0 0 2px #f6e05e55 }}
 .imgs img.primary {{ width: 100%; display: block }}
 .imgs img.secondary {{ width: 33.33%; display: inline-block }}
 figcaption {{ padding: .45rem .6rem }}
 .head {{ display: flex; gap: .5rem; align-items: baseline; flex-wrap: wrap }}
 .head b {{ font-size: .93rem }}
 .rung {{ color: #4a5568; font-size: .8rem }}
 .meta {{ color: #718096; font-size: .72rem; background: #f7fafc; border-radius: 4px;
          padding: 1px 5px; white-space: nowrap }}
 .badge {{ color: #fff; border-radius: 4px; padding: 1px 6px; font-size: .68rem }}
 .badge.lineage {{ background: #2b6cb0 }} .badge.d3 {{ background: #6b46c1 }}
 .star {{ margin-left: auto; background: none; border: none; font-size: 1.15rem; cursor: pointer; color: #a0aec0 }}
 .card.starred .star {{ color: #d69e2e }}
 .when {{ color: #4a5568; font-size: .8rem; margin: .2rem 0 }}
 .params {{ display: grid; grid-template-columns: auto 1fr auto 1fr; gap: 1px 8px;
            font-size: .76rem; background: #f7fafc; border-radius: 5px; padding: .35rem .5rem }}
 .pk {{ color: #718096 }} .pv {{ font-variant-numeric: tabular-nums }}
 .note {{ width: 100%; margin-top: .35rem; font-size: .8rem; border: 1px solid #e2e8f0;
          border-radius: 4px; padding: .3rem; min-height: 1.6rem; resize: vertical }}
 #exported {{ white-space: pre-wrap; background: #f7fafc; padding: .6rem; border-radius: 6px;
              display: none; font-family: monospace; font-size: .8rem }}
 #flat {{ display: none }}
</style>
<h1>Sandbox arm review</h1>
<p class="sub">{n_cards} arms. Stars + notes persist in this browser; export copies the shortlist.</p>
{ref_html}
<div class="bar">
  <label>sort <select id="sort">
    <option value="newest">newest first</option>
    <option value="name">name</option>
    <option value="ffr-desc">quick-FFR ↓</option>
    <option value="ffr-asc">quick-FFR ↑</option>
  </select></label>
  <label>group <select id="groupby">
    <option value="technique">by initiative</option>
    <option value="none">flat list</option>
  </select></label>
  <label>rung <select id="rung"><option value="">all</option>
    <option>2M</option><option>6.25M</option><option>12.5M</option><option>25M</option>
    <option>500K</option><option>BL 1.08M</option><option>LAION 2M</option>
    <option>LAION 2M dedup</option><option>jina EN 2M</option>
    <option>jina multi 2M</option><option>jina multi 6.25M</option>
    <option>redditmix 2M</option><option>capacity grid</option>
  </select></label>
  <label>model <select id="model"><option value="">all</option>{model_opts}</select></label>
  <label>corpus <select id="corpus"><option value="">all</option>{corpus_opts}</select></label>
  <label><input type="checkbox" id="onlystars"> shortlist only</label>
  <button id="export">export shortlist</button>
  <span id="count"></span>
</div>
{toc_html}
<div id="exported"></div>
<section id="flat"><h2>all arms</h2><div class="grid"></div></section>
{"".join(sections)}
<script>
const LS = "sandbox-review-v1";
const state = JSON.parse(localStorage.getItem(LS) || '{{"stars":{{}},"notes":{{}}}}');
const save = () => localStorage.setItem(LS, JSON.stringify(state));
const cards = [...document.querySelectorAll("section:not(#flat) .card")];
const home = new Map(cards.map(c => [c, c.parentNode]));
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
const cmp = sort => (a, b) => {{
  if (sort === "name") return a.dataset.id.localeCompare(b.dataset.id);
  if (sort === "newest") return +b.dataset.ts - +a.dataset.ts;
  const fa = +a.dataset.ffr, fb = +b.dataset.ffr;
  return sort === "ffr-desc" ? fb - fa : fa - fb;
}};
const visible = c => {{
  const rung = document.getElementById("rung").value;
  const model = document.getElementById("model").value;
  const corpus = document.getElementById("corpus").value;
  const only = document.getElementById("onlystars").checked;
  return (!rung || c.dataset.rung === rung)
    && (!model || c.dataset.model === model)
    && (!corpus || c.dataset.corpus === corpus)
    && (!only || state.stars[c.dataset.id]);
}};
function apply() {{
  const sort = document.getElementById("sort").value;
  const flatMode = document.getElementById("groupby").value === "none";
  const flat = document.getElementById("flat");
  const secs = [...document.querySelectorAll("section:not(#flat)")];
  let vis = 0;
  if (flatMode) {{
    const grid = flat.querySelector(".grid");
    [...cards].sort(cmp(sort)).forEach(c => grid.appendChild(c));
    for (const c of cards) {{
      const show = visible(c);
      c.style.display = show ? "" : "none";
      if (show) vis++;
    }}
    flat.style.display = "";
    secs.forEach(s => s.style.display = "none");
    document.querySelector(".toc").style.display = "none";
  }} else {{
    for (const c of cards) home.get(c).appendChild(c);
    flat.style.display = "none";
    document.querySelector(".toc").style.display = "";
    for (const sec of secs) {{
      const grid = sec.querySelector(".grid");
      const sc = [...grid.querySelectorAll(".card")].sort(cmp(sort));
      let secVis = 0;
      for (const c of sc) {{
        grid.appendChild(c);
        const show = visible(c);
        c.style.display = show ? "" : "none";
        if (show) secVis++;
      }}
      sec.style.display = secVis ? "" : "none";
      const toc = document.querySelector(`.toc a[href="#${{sec.id}}"]`);
      if (toc) toc.style.display = secVis ? "" : "none";
      vis += secVis;
    }}
    if (sort === "newest") {{
      const anchor = secs[0].parentNode;
      secs.sort((a, b) => {{
        const mx = s => Math.max(0, ...[...s.querySelectorAll(".card")]
          .filter(c => c.style.display !== "none").map(c => +c.dataset.ts));
        return mx(b) - mx(a);
      }}).forEach(sec => anchor.appendChild(sec));
    }}
  }}
  document.getElementById("count").textContent = vis + " arms";
}}
for (const id of ["sort", "groupby", "rung", "model", "corpus", "onlystars"])
  document.getElementById(id).onchange = apply;
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
    print("\nGROUP AUDIT:")
    for g in GROUP_ORDER:
        if g in audit:
            names = ", ".join(i.split("/")[1] for i in audit[g][:8])
            print(f"  {g} ({len(audit[g])}): {names}"
                  + (" …" if len(audit[g]) > 8 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
