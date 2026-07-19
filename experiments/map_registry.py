#!/usr/bin/env python3
"""Map registry: index every trained basemap with provenance, publish a browsable site.

Read-only, post-hoc tooling (see latent-labs/guides/plan-map-inspection.md).
Never a launch-path dependency; must work with roundwatch down.

  uv run python experiments/map_registry.py scan      # -> /data/latent-basemap/maps.json
  uv run python experiments/map_registry.py publish   # -> ~/.agent/basemap-maps/ (gsv.local:8800/basemap-maps/)

The scanner keys off receipt presence (queue.json / render-manifest.json),
never a fixed tree, because only rounds 0014+ share the modern layout.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

RUNS_DIR = Path("/data/latent-basemap/runs")
CHECKPOINT_DIR = Path("/data/checkpoints/pumap")
LEDGER_DIR = Path.home() / "code/latent-labs/basemap-100m"
REGISTRY_PATH = Path("/data/latent-basemap/maps.json")
SITE_DIR = Path.home() / ".agent/basemap-maps"
SITE_URL = "http://gsv.local:8800/basemap-maps"

SCHEMA = "basemap-map-registry-v1"


# ---------------------------------------------------------------- ledger ----

def _front_matter(path: Path) -> dict:
    """Minimal YAML front-matter reader (flat `key: value` lines only)."""
    out: dict = {}
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return out
    m = re.match(r"\A---\n(.*?)\n---\n", text, re.S)
    if not m:
        return out
    for line in m.group(1).splitlines():
        kv = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$", line)
        if not kv:
            continue
        key, raw = kv.group(1), kv.group(2).strip()
        out[key] = raw.strip('"')
    return out


def ledger_status() -> dict:
    """round_id -> {round, result, review} front-matter status strings + doc names."""
    rounds: dict = {}
    if not LEDGER_DIR.is_dir():
        return rounds
    for doc in sorted(LEDGER_DIR.glob("*.md")):
        m = re.match(r"(round|result|review)-(\d{4})-", doc.name)
        if not m:
            continue
        kind, rid = m.group(1), m.group(2)
        fm = _front_matter(doc)
        entry = rounds.setdefault(rid, {})
        # keep the newest doc of each kind (suffix -01 style reissues sort last)
        entry[kind] = {"doc": doc.name, "status": fm.get("status", "unknown")}
    return rounds


def evidence_status(rid: str, ledger: dict) -> str:
    entry = ledger.get(rid, {})
    if "review" in entry:
        return f"review:{entry['review']['status']}"
    if "result" in entry:
        return "result:pending-review"
    if "round" in entry:
        return f"round:{entry['round']['status']}"
    return "unregistered"


# ----------------------------------------------------------------- scan -----

def _load_json(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _relpath(p: Path) -> str:
    return f"gsv:{p}"


def scan_modern_round(round_dir: Path, ledger: dict) -> list[dict]:
    """Rounds with a queue/ manifest (0014+ layout). One entry per trained map."""
    rid_m = re.match(r"round-(\d{4})", round_dir.name)
    rid = rid_m.group(1) if rid_m else round_dir.name
    art = round_dir / "queue/artifacts"
    queue = _load_json(round_dir / "queue/queue.json") or {}
    entries = []

    receipt = _load_json(art / "train/train-receipt.json")
    if not receipt:
        return entries
    prof = receipt.get("performance_profile", {})
    base = prof.get("baseline_key", {})
    stats = receipt.get("train_stats", {})
    config = _load_json(art / "train/production-config.json") or {}
    model_cfg = (config.get("config") or {}).get("model", {})
    panel = _load_json(art / "panel/panel.json") or {}
    render_manifest = _load_json(art / "semantic-renders/render-manifest.json") or {}
    transform = _load_json(art / "coordinates/actual-transform.json") or {}

    model = receipt.get("model", {})
    png = art / "semantic-renders/seed42-map.png"
    pngs = sorted((art / "semantic-renders").glob("*.png")) if (art / "semantic-renders").is_dir() else []
    coords_dir = art / "coordinates"
    coord_chunks = sorted(coords_dir.glob("chunk-*/coordinates.npy")) if coords_dir.is_dir() else []

    train_done = _load_json(art / "train_seed42_30m.done.json") or {}
    finished = train_done.get("finished")
    if not finished:
        for dm in sorted(art.glob("*.done.json")):
            j = _load_json(dm) or {}
            finished = j.get("finished") or finished

    p = panel.get("panel", {})
    proj = panel.get("projection", {})
    purity = p.get("purity", {}) if isinstance(p.get("purity"), dict) else {}
    checks = panel.get("decision_checks", {})

    entries.append({
        "map_id": f"round-{rid}-seed{(config.get('config') or {}).get('seed', stats.get('seed', 42))}",
        "round_id": rid,
        "kind": "round-map",
        "date": finished,
        "evidence_status": evidence_status(rid, ledger),
        "n_rows": base.get("n"),
        "dims": [base.get("d"), base.get("n_components")],
        "architecture": model_cfg.get("architecture"),
        "hidden_dim": base.get("hidden_dim") or model_cfg.get("hidden_dimension"),
        "kernel": base.get("kernel"),
        "pipeline": base.get("pipeline"),
        "precision": "bf16" if stats.get("amp_dtype") == "bfloat16" else ("fp16" if base.get("use_amp") else "fp32"),
        "updates": stats.get("optimizer_steps_succeeded"),
        "updates_per_s": stats.get("updates_per_s") or prof.get("rate_median"),
        "model": {"path": _relpath(art / "train/model.pt"), "sha256": model.get("sha256"), "bytes": model.get("bytes")},
        "coordinates": {
            "dir": _relpath(coords_dir),
            "chunks": len(coord_chunks),
            "receipt_sha256": (render_manifest.get("coordinate_stream") or {}).get("sha256")
                               or _sha_of(transform),
        },
        "panel": {
            "path": _relpath(art / "panel/panel.json"),
            "ffr": p.get("ffr"),
            "density": p.get("density"),
            "purity_k256": purity.get("k256"),
            "purity_k1024": purity.get("k1024"),
            "proj_ffr": proj.get("proj_ffr"),
            "proj_knn_ffr": proj.get("proj_knn_regressor_ffr"),
            "decision_checks_all_pass": bool(checks) and all(bool(v) for v in checks.values()),
            "formula_version": p.get("formula_version"),
        },
        "renders": [{"path": _relpath(x), "bytes": x.stat().st_size} for x in pngs],
        "render_diagnostics": render_manifest.get("diagnostics"),
        "release_sha": ((queue.get("release") or {}).get("sha")
                        or queue.get("release_sha")
                        or (panel.get("panel", {}).get("provenance") or {}).get("code_commit")),
        "run_dir": _relpath(round_dir),
    })
    return entries


def _sha_of(obj) -> str | None:
    if isinstance(obj, dict):
        return obj.get("sha256") or obj.get("identity_sha256")
    return None


def scan_legacy_renders(round_dir: Path, ledger: dict) -> list[dict]:
    """Rounds with a top-level renders/ dir (round-0001 style) but no queue artifacts."""
    rid_m = re.match(r"round-(\d{4})", round_dir.name)
    rid = rid_m.group(1) if rid_m else round_dir.name
    renders_dir = round_dir / "renders"
    manifest = _load_json(renders_dir / "render-manifest.json")
    pngs = sorted(renders_dir.glob("*.png"))
    if not pngs:
        return []
    return [{
        "map_id": f"round-{rid}-legacy-renders",
        "round_id": rid,
        "kind": "legacy-renders",
        "date": datetime.fromtimestamp(pngs[0].stat().st_mtime, tz=timezone.utc).isoformat(),
        "evidence_status": evidence_status(rid, ledger),
        "renders": [{"path": _relpath(x), "bytes": x.stat().st_size} for x in pngs],
        "render_manifest": bool(manifest),
        "run_dir": _relpath(round_dir),
    }]


def scan_checkpoints() -> list[dict]:
    """Pre-round checkpoints in /data/checkpoints/pumap (best-effort, no metrics)."""
    if not CHECKPOINT_DIR.is_dir():
        return []
    out = []
    for pt in sorted(CHECKPOINT_DIR.glob("*.pt")):
        st = pt.stat()
        out.append({
            "map_id": f"checkpoint-{pt.stem}",
            "kind": "pre-round-checkpoint",
            "date": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
            "evidence_status": "pre-protocol",
            "model": {"path": _relpath(pt), "bytes": st.st_size},
        })
    return out


def scan() -> dict:
    ledger = ledger_status()
    maps: list[dict] = []
    if RUNS_DIR.is_dir():
        for round_dir in sorted(RUNS_DIR.glob("round-*")):
            if (round_dir / "queue/artifacts").is_dir():
                maps += scan_modern_round(round_dir, ledger)
            elif (round_dir / "renders").is_dir():
                maps += scan_legacy_renders(round_dir, ledger)
    maps += scan_checkpoints()
    return {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "round_maps": sum(1 for m in maps if m["kind"] == "round-map"),
            "legacy_render_sets": sum(1 for m in maps if m["kind"] == "legacy-renders"),
            "pre_round_checkpoints": sum(1 for m in maps if m["kind"] == "pre-round-checkpoint"),
        },
        "maps": maps,
    }


# --------------------------------------------------------------- publish ----

CSS = """
:root { color-scheme: light dark; --fg:#1a1d21; --bg:#fff; --muted:#667; --line:#e2e5ea;
        --card:#f6f7f9; --ok:#0a7d33; --warn:#a15c00; --bad:#b3261e; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e6e8eb; --bg:#121417; --muted:#9aa1ab; --line:#2a2f36; --card:#1b1f24;
          --ok:#4ccb7a; --warn:#e0a34e; --bad:#e5776f; } }
* { box-sizing: border-box; }
body { font: 15px/1.5 system-ui, sans-serif; color: var(--fg); background: var(--bg);
       margin: 0 auto; max-width: 1100px; padding: 24px 20px 80px; }
h1 { font-size: 1.5rem; } h2 { font-size: 1.15rem; margin-top: 2rem; }
a { color: inherit; } small, .muted { color: var(--muted); }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; }
th { position: sticky; top: 0; background: var(--bg); }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.scroll { overflow-x: auto; }
.badge { padding: 1px 8px; border-radius: 9px; font-size: 12px; background: var(--card); }
.accepted { color: var(--ok); } .partial, .pending { color: var(--warn); } .rejected { color: var(--bad); }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; margin: 12px 0; }
img.render { max-width: 100%; border: 1px solid var(--line); border-radius: 6px; background: #fff; }
code { font-size: 12.5px; background: var(--card); padding: 1px 5px; border-radius: 4px; }
dl { display: grid; grid-template-columns: max-content 1fr; gap: 3px 14px; margin: 8px 0; }
dt { color: var(--muted); } dd { margin: 0; overflow-wrap: anywhere; }
"""


def _badge(status: str) -> str:
    cls = "muted"
    if "accepted" in status: cls = "accepted"
    elif "partial" in status or "pending" in status: cls = "partial"
    elif "rejected" in status: cls = "rejected"
    return f'<span class="badge {cls}">{html.escape(status)}</span>'


def _fmt(v, digits=4):
    if v is None: return "—"
    if isinstance(v, float): return f"{v:.{digits}f}"
    if isinstance(v, int) and v >= 1_000_000: return f"{v/1e6:.0f}M"
    return html.escape(str(v))


def publish(registry: dict) -> None:
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    round_maps = [m for m in registry["maps"] if m["kind"] == "round-map"]
    legacy = [m for m in registry["maps"] if m["kind"] == "legacy-renders"]
    checkpoints = [m for m in registry["maps"] if m["kind"] == "pre-round-checkpoint"]

    rows = []
    for m in sorted(round_maps, key=lambda x: x.get("date") or "", reverse=True):
        p = m["panel"]
        page = f'round-{m["round_id"]}/index.html'
        rows.append(
            f'<tr><td><a href="{page}">{html.escape(m["map_id"])}</a></td>'
            f'<td>{(m.get("date") or "")[:10]}</td>'
            f'<td class="num">{_fmt(m.get("n_rows"))}</td>'
            f'<td>h{m.get("hidden_dim")} {html.escape(str(m.get("architecture") or ""))}</td>'
            f'<td class="num">{_fmt(p.get("ffr"))}</td>'
            f'<td class="num">{_fmt(p.get("density"))}</td>'
            f'<td class="num">{_fmt(p.get("purity_k1024"))}</td>'
            f'<td class="num">{_fmt(p.get("proj_ffr"))}</td>'
            f'<td>{_badge(m["evidence_status"])}</td></tr>'
        )
    legacy_rows = [
        f'<tr><td><a href="round-{m["round_id"]}/index.html">{html.escape(m["map_id"])}</a></td>'
        f'<td>{(m.get("date") or "")[:10]}</td><td class="num">{len(m["renders"])} renders</td>'
        f'<td>{_badge(m["evidence_status"])}</td></tr>'
        for m in legacy
    ]
    ckpt_rows = [
        f'<tr><td>{html.escape(m["map_id"])}</td><td>{(m.get("date") or "")[:10]}</td>'
        f'<td class="num">{m["model"]["bytes"]/1e6:.0f} MB</td>'
        f'<td><code>{html.escape(m["model"]["path"])}</code></td></tr>'
        for m in checkpoints
    ]

    index = f"""<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>basemap maps</title><style>{CSS}</style>
<h1>Basemap map registry</h1>
<p class="muted">Generated {registry["generated_utc"][:19]}Z from
<code>/data/latent-basemap/maps.json</code> · {registry["counts"]["round_maps"]} round maps ·
{registry["counts"]["legacy_render_sets"]} legacy render sets ·
{registry["counts"]["pre_round_checkpoints"]} pre-protocol checkpoints ·
<a href="../basemap-gallery/">old gallery (2026-07-01)</a> ·
<a href="http://gsv.local:8710/">roundwatch</a></p>
<h2>Round maps</h2>
<div class="scroll"><table>
<tr><th>map</th><th>date</th><th>N</th><th>net</th><th>ffr</th><th>density</th>
<th>purity k1024</th><th>proj ffr</th><th>evidence</th></tr>
{''.join(rows) or '<tr><td colspan=9 class="muted">none</td></tr>'}
</table></div>
<h2>Legacy render sets</h2>
<div class="scroll"><table><tr><th>set</th><th>date</th><th>contents</th><th>evidence</th></tr>
{''.join(legacy_rows) or '<tr><td colspan=4 class="muted">none</td></tr>'}</table></div>
<h2>Pre-protocol checkpoints</h2>
<div class="scroll"><table><tr><th>checkpoint</th><th>date</th><th>size</th><th>path</th></tr>
{''.join(ckpt_rows) or '<tr><td colspan=4 class="muted">none</td></tr>'}</table></div>
"""
    compare_links = ""
    try:
        from gallery_v2 import build_compare_groups
        groups = build_compare_groups(registry, SITE_DIR)
        if groups:
            items = " · ".join(
                f'<a href="compare/{g["slug"]}/index.html">{g["n_rows"]:,} rows ({g["panels"]} maps)</a>'
                for g in groups)
            compare_links = f'<h2>Interactive comparison</h2><p>Linked small-multiples on a common 20k sample: {items}</p>'
    except Exception as e:  # viewer generation is best-effort; the index must still publish
        compare_links = f'<p class="muted">compare viewer generation failed: {html.escape(str(e))}</p>'
    index = index.replace("<h2>Round maps</h2>", compare_links + "\n<h2>Round maps</h2>", 1)

    (SITE_DIR / "index.html").write_text(index)

    for m in round_maps + legacy:
        page_dir = SITE_DIR / f'round-{m["round_id"]}'
        page_dir.mkdir(exist_ok=True)
        img_tags = []
        for r in m.get("renders", []):
            src = Path(r["path"].removeprefix("gsv:"))
            if src.is_file():
                dst = page_dir / src.name
                if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                    shutil.copy2(src, dst)
                img_tags.append(f'<p><img class="render" src="{src.name}" alt="{src.name}">'
                                f'<br><small>{src.name}</small></p>')
        panel = m.get("panel", {})
        dl_items = []
        for label, val in [
            ("evidence", m["evidence_status"]), ("date", m.get("date")),
            ("N rows", m.get("n_rows")), ("architecture",
             f'{m.get("architecture")} h{m.get("hidden_dim")} → {m.get("dims")}' if m.get("architecture") else None),
            ("kernel", m.get("kernel")), ("pipeline", m.get("pipeline")),
            ("precision", m.get("precision")), ("updates", m.get("updates")),
            ("updates/s", m.get("updates_per_s")), ("release", m.get("release_sha")),
            ("model", (m.get("model") or {}).get("path")),
            ("model sha256", (m.get("model") or {}).get("sha256")),
            ("coordinates", (m.get("coordinates") or {}).get("dir")),
            ("panel file", panel.get("path")), ("panel version", panel.get("formula_version")),
            ("run dir", m.get("run_dir")),
        ]:
            if val is not None:
                dl_items.append(f"<dt>{label}</dt><dd>{_fmt(val) if isinstance(val,(int,float)) else html.escape(str(val))}</dd>")
        metrics = ""
        if panel.get("ffr") is not None:
            metrics = (f'<div class="card"><b>Panel</b><dl>'
                       f'<dt>ffr@0.1%</dt><dd>{_fmt(panel.get("ffr"))}</dd>'
                       f'<dt>density</dt><dd>{_fmt(panel.get("density"))}</dd>'
                       f'<dt>purity k256 / k1024</dt><dd>{_fmt(panel.get("purity_k256"))} / {_fmt(panel.get("purity_k1024"))}</dd>'
                       f'<dt>proj ffr (vs kNN reg)</dt><dd>{_fmt(panel.get("proj_ffr"))} (vs {_fmt(panel.get("proj_knn_ffr"))})</dd>'
                       f'<dt>all decision checks</dt><dd>{"PASS" if panel.get("decision_checks_all_pass") else "see panel.json"}</dd>'
                       f'</dl></div>')
        page = f"""<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(m["map_id"])}</title><style>{CSS}</style>
<p><a href="../index.html">← all maps</a></p>
<h1>{html.escape(m["map_id"])}</h1>
{metrics}
<div class="card"><b>Provenance</b><dl>{''.join(dl_items)}</dl></div>
<h2>Renders</h2>
{''.join(img_tags) or '<p class="muted">no renders on disk</p>'}
"""
        (page_dir / "index.html").write_text(page)

    print(f"published {len(round_maps)+len(legacy)} map pages -> {SITE_DIR}  ({SITE_URL}/)")


# ------------------------------------------------------------------ main ----

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["scan", "publish"])
    args = ap.parse_args()
    if args.command == "scan":
        reg = scan()
        REGISTRY_PATH.write_text(json.dumps(reg, indent=1))
        print(f"wrote {REGISTRY_PATH}: {reg['counts']}")
    else:
        reg = _load_json(REGISTRY_PATH)
        if reg is None or reg.get("schema") != SCHEMA:
            reg = scan()
            REGISTRY_PATH.write_text(json.dumps(reg, indent=1))
            print(f"(re)scanned -> {REGISTRY_PATH}")
        publish(reg)


if __name__ == "__main__":
    main()
