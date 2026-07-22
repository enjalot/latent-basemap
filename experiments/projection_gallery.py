#!/usr/bin/env python3
"""Build interactive, provenance-bearing explorers for OOD projections.

This is deliberately post-hoc map-registry tooling.  It consumes immutable
projection NPZs and panel receipts, samples only for browser rendering, and
never participates in experiment admission or changes the scientific score.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    from experiments.gallery_v2 import _gather
except ModuleNotFoundError:  # direct `python experiments/map_registry.py`
    from gallery_v2 import _gather


SCHEMA = "basemap-projection-explorer-v1"
CORPUS_SAMPLE_N = 20_000
BASE_SAMPLE_N = 30_000


def _path(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value.removeprefix("gsv:"))


def _seed(label: str) -> int:
    return int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)


def _sample_rows(n_rows: int, maximum: int, *, label: str) -> np.ndarray:
    if n_rows <= maximum:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.RandomState(_seed(label))
    return np.sort(rng.choice(n_rows, maximum, replace=False)).astype(np.int64)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _write_xy(path: Path, value: np.ndarray) -> dict[str, Any]:
    xy = np.ascontiguousarray(value, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError(f"projection explorer coordinates are invalid: {xy.shape}")
    path.write_bytes(xy.tobytes())
    return {
        "path": f"data/{path.name}",
        "rows": int(len(xy)),
        "dtype": "<f4",
        "shape": [int(len(xy)), 2],
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _ids(array: np.ndarray, rows: np.ndarray) -> list[str]:
    return [str(value) for value in np.asarray(array)[rows].tolist()]


def _short(value: Any, maximum: int = 320) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= maximum else text[: maximum - 1] + "…"


def _dadabase_labels(
    source: Path, corpus_ids: list[str], query_ids: list[str]
) -> tuple[list[dict], list[dict]]:
    import pandas as pd

    frame = pd.read_parquet(source, columns=["joke", "score"])

    def one(value: str) -> dict:
        try:
            row = frame.iloc[int(value)]
            return {"id": value, "label": _short(row["joke"]), "score": int(row["score"])}
        except (ValueError, IndexError, TypeError):
            return {"id": value, "label": value}

    return [one(value) for value in corpus_ids], [one(value) for value in query_ids]


def _arrow_rows(path: Path, wanted: set[str], *, title: bool) -> dict[str, str]:
    """Read only labels for requested IDs from a local HF Arrow stream."""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    found: dict[str, str] = {}
    with pa.memory_map(str(path), "r") as source:
        try:
            reader = ipc.open_stream(source)
        except pa.ArrowInvalid:
            reader = ipc.open_file(source)
        for batch in reader:
            names = batch.schema.names
            ids = batch.column(names.index("_id")).to_pylist()
            titles = batch.column(names.index("title")).to_pylist() if title and "title" in names else None
            texts = batch.column(names.index("text")).to_pylist() if "text" in names else [""] * len(ids)
            for index, raw_id in enumerate(ids):
                value = str(raw_id)
                if value not in wanted:
                    continue
                heading = titles[index] if titles is not None else ""
                body = texts[index]
                found[value] = _short(heading or body)
            if len(found) == len(wanted):
                break
    return found


def _trec_labels(corpus_ids: list[str], query_ids: list[str]) -> tuple[list[dict], list[dict]]:
    root = Path("/data/hf/datasets/mteb___trec-covid")
    corpus_path = next(root.joinpath("corpus").rglob("*.arrow"), None)
    query_path = next(root.joinpath("queries").rglob("*.arrow"), None)
    corpus = _arrow_rows(corpus_path, set(corpus_ids), title=True) if corpus_path else {}
    queries = _arrow_rows(query_path, set(query_ids), title=False) if query_path else {}
    return (
        [{"id": value, "label": corpus.get(value, value)} for value in corpus_ids],
        [{"id": value, "label": queries.get(value, value)} for value in query_ids],
    )


def _labels(
    entry: dict, archive: Any, corpus_rows: np.ndarray, query_rows: np.ndarray
) -> tuple[list[dict], list[dict]]:
    corpus_key = "probe_corpus_ids" if "probe_corpus_ids" in archive.files else "probe_corpus_rows"
    query_key = "probe_query_ids" if "probe_query_ids" in archive.files else "probe_query_rows"
    corpus_ids = _ids(archive[corpus_key], corpus_rows)
    query_ids = _ids(archive[query_key], query_rows)
    probe = entry["projection"]["probe"]
    inputs = entry["projection"].get("inputs") or {}
    if probe == "dadabase":
        source = _path((inputs.get("texts") or {}).get("canonical_path"))
        if source and source.is_file():
            return _dadabase_labels(source, corpus_ids, query_ids)
    if probe == "trec-covid":
        try:
            return _trec_labels(corpus_ids, query_ids)
        except (ImportError, OSError, ValueError):
            pass
    return (
        [{"id": value, "label": value} for value in corpus_ids],
        [{"id": value, "label": value} for value in query_ids],
    )


def _base_coordinates(entry: dict) -> tuple[np.ndarray, np.ndarray]:
    coordinates = _path((entry.get("base_coordinates") or {}).get("dir"))
    sample_ids = _path((entry.get("base_sample_ids") or {}).get("path"))
    if not coordinates or not sample_ids or not coordinates.is_dir() or not sample_ids.is_file():
        return np.empty((0,), np.int64), np.empty((0, 2), np.float32)
    ids = np.load(sample_ids, allow_pickle=False)
    chosen = _sample_rows(len(ids), BASE_SAMPLE_N, label=entry["map_id"] + ":base")
    rows = np.asarray(ids[chosen], dtype=np.int64)
    return rows, _gather(coordinates, rows)


def build_projection_explorers(registry: dict, site_dir: Path) -> list[dict]:
    entries = [item for item in registry["maps"] if item.get("kind") == "projection-map"]
    built: list[dict] = []
    by_round: dict[str, list[dict]] = {}
    for entry in entries:
        archive_path = _path(entry["projection"].get("coordinates"))
        if not archive_path or not archive_path.is_file():
            continue
        out = site_dir / "projections" / entry["map_id"]
        data_dir = out / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        with np.load(archive_path, allow_pickle=False) as archive:
            corpus_xy = np.asarray(archive["probe_corpus_coords"], dtype=np.float32)
            query_xy = np.asarray(archive["probe_query_coords"], dtype=np.float32)
            corpus_rows = _sample_rows(
                len(corpus_xy), CORPUS_SAMPLE_N, label=entry["map_id"] + ":corpus"
            )
            query_rows = np.arange(len(query_xy), dtype=np.int64)
            corpus_labels, query_labels = _labels(entry, archive, corpus_rows, query_rows)
            base_ids, base_xy = _base_coordinates(entry)
            files = {
                "base": _write_xy(data_dir / "base.xy.bin", base_xy),
                "corpus": _write_xy(data_dir / "corpus.xy.bin", corpus_xy[corpus_rows]),
                "queries": _write_xy(data_dir / "queries.xy.bin", query_xy),
            }
            label_body = {"corpus": corpus_labels, "queries": query_labels}
            labels_path = data_dir / "labels.json"
            labels_path.write_text(json.dumps(label_body, ensure_ascii=False, separators=(",", ":")))
            files["labels"] = {
                "path": "data/labels.json",
                "bytes": labels_path.stat().st_size,
                "sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
            }
            sample = {
                "base_source_rows": int(len(base_ids)),
                "base_source_rows_sha256": _array_sha256(base_ids),
                "probe_corpus_source_rows": int(len(corpus_rows)),
                "probe_corpus_source_rows_sha256": _array_sha256(corpus_rows),
                "probe_query_source_rows": int(len(query_rows)),
                "probe_query_source_rows_sha256": _array_sha256(query_rows),
            }

        manifest = {
            "schema": SCHEMA,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "map_id": entry["map_id"],
            "round_id": entry["round_id"],
            "evidence_status": entry["evidence_status"],
            "base_map": entry.get("base_map"),
            "projection": entry["projection"],
            "source_coordinate_artifact": entry["projection"].get("coordinate_signature"),
            "sample": sample,
            "files": files,
            "sampling_note": (
                "Browser rendering uses deterministic base/probe samples; all probe queries are shown. "
                "Scientific metrics come unchanged from the full registered panel."
            ),
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
        (out / "index.html").write_text(_viewer_html(entry))
        page = {
            "map_id": entry["map_id"],
            "round_id": entry["round_id"],
            "probe": entry["projection"]["probe"],
            "path": f"projections/{entry['map_id']}/index.html",
        }
        built.append(page)
        by_round.setdefault(entry["round_id"], []).append(page)

    for round_id, pages in by_round.items():
        page_dir = site_dir / f"round-{round_id}"
        page_dir.mkdir(exist_ok=True)
        block = "<!-- projections:start --><h2>Projection explorers</h2><ul>" + "".join(
            f'<li><a href="../{html.escape(item["path"])}">{html.escape(item["map_id"])}</a></li>'
            for item in pages
        ) + "</ul><!-- projections:end -->"
        target = page_dir / "index.html"
        if target.is_file():
            body = target.read_text()
            if "<!-- projections:start -->" in body:
                body = re.sub(
                    r"<!-- projections:start -->.*?<!-- projections:end -->",
                    block,
                    body,
                    flags=re.DOTALL,
                )
            else:
                # Migrate pages emitted before the replaceable section markers.
                body = re.sub(
                    r"<h2>Projection explorers</h2><ul>.*?</ul>",
                    "",
                    body,
                    flags=re.DOTALL,
                )
                body += block
            target.write_text(body)
        else:
            target.write_text(
                '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
                f"<title>round {round_id} projections</title><style>{_BASIC_CSS}</style>"
                '<p><a href="../index.html">← all maps</a></p>'
                f"<h1>Round {round_id}</h1>{block}"
            )
    return built


_BASIC_CSS = """
:root { color-scheme: light dark; --fg:#1a1d21; --bg:#fff; --muted:#667; --line:#e2e5ea; --card:#f6f7f9; }
@media (prefers-color-scheme: dark) { :root { --fg:#e6e8eb; --bg:#121417; --muted:#9aa1ab; --line:#2a2f36; --card:#1b1f24; } }
body { font:14px/1.5 system-ui,sans-serif; color:var(--fg); background:var(--bg); margin:0 auto; max-width:1280px; padding:18px; }
a { color:inherit; } .muted { color:var(--muted); } code { font-size:12px; }
"""


def _viewer_html(entry: dict) -> str:
    title = f"{entry['projection']['display_name']} on {entry.get('base_map') or 'basemap'}"
    return f'''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>{_BASIC_CSS}
.stats {{ display:flex; gap:10px; flex-wrap:wrap; margin:8px 0 12px; }}
.pill {{ border:1px solid var(--line); background:var(--card); border-radius:12px; padding:2px 9px; }}
.controls {{ display:flex; gap:13px; flex-wrap:wrap; align-items:center; margin:10px 0; }}
#plot {{ width:100%; height:min(72vh,760px); display:block; background:#fff; border:1px solid var(--line); border-radius:8px; cursor:crosshair; }}
@media (prefers-color-scheme: dark) {{ #plot {{ background:#0d0f12; }} }}
#tip {{ position:fixed; max-width:430px; display:none; pointer-events:none; background:var(--card); border:1px solid var(--line); border-radius:7px; padding:6px 9px; z-index:5; }}
select {{ max-width:min(620px,90vw); }}
</style>
<p><a href="../../index.html">← registry</a> · <a href="manifest.json">manifest</a> · <a href="../../round-{entry['round_id']}/index.html">round {entry['round_id']}</a></p>
<h1>{html.escape(title)}</h1>
<p class="muted">Interactive rendering is a deterministic browser sample; full-corpus metrics and artifact identities remain in the immutable panel and manifest.</p>
<div class="stats" id="stats"></div>
<div class="controls">
 <label><input id="showBase" type="checkbox" checked> training-map context</label>
 <label><input id="showCorpus" type="checkbox" checked> probe corpus</label>
 <label><input id="showQueries" type="checkbox" checked> held-out queries</label>
 <button id="fitAll">fit all</button><button id="fitProbe">fit probe</button><button id="reset">reset</button>
 <label>query <select id="query"><option value="">choose…</option></select></label>
</div>
<canvas id="plot"></canvas><div id="tip"></div>
<script>
let M,L,D={{}}, view=null, drag=null, selected=-1;
const C={{base:'#8b929c',corpus:'#3178c6',query:'#e5484d',selected:'#f5a524'}};
async function bin(file){{const b=await (await fetch(file.path)).arrayBuffer();return new Float32Array(b)}}
function extent(names){{let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity;for(const n of names){{const a=D[n];for(let i=0;i<a.length;i+=2){{x0=Math.min(x0,a[i]);x1=Math.max(x1,a[i]);y0=Math.min(y0,a[i+1]);y1=Math.max(y1,a[i+1])}}}}if(!isFinite(x0))return[-1,1,-1,1];const px=(x1-x0||1)*.04,py=(y1-y0||1)*.04;return[x0-px,x1+px,y0-py,y1+py]}}
function sx(x,w){{return(x-view[0])*w/(view[1]-view[0])}} function sy(y,h){{return h-(y-view[2])*h/(view[3]-view[2])}}
function draw(){{const c=document.querySelector('#plot'),r=c.getBoundingClientRect(),d=devicePixelRatio||1;c.width=r.width*d;c.height=r.height*d;const g=c.getContext('2d');g.scale(d,d);g.clearRect(0,0,r.width,r.height);
 function points(name,color,size,alpha){{const a=D[name];g.fillStyle=color;g.globalAlpha=alpha;for(let i=0;i<a.length;i+=2)g.fillRect(sx(a[i],r.width),sy(a[i+1],r.height),size,size)}}
 if(showBase.checked)points('base',C.base,1,.22);if(showCorpus.checked)points('corpus',C.corpus,1.5,.55);if(showQueries.checked)points('queries',C.query,4,.9);g.globalAlpha=1;
 if(selected>=0){{const a=D.queries,x=sx(a[2*selected],r.width),y=sy(a[2*selected+1],r.height);g.strokeStyle=C.selected;g.lineWidth=3;g.beginPath();g.arc(x,y,9,0,7);g.stroke()}}}}
function fit(names){{view=extent(names);draw()}}
function nearest(mx,my){{const c=plot.getBoundingClientRect();let best=null,bd=12*12;for(const name of ['queries','corpus']){{if(name==='queries'&&!showQueries.checked||name==='corpus'&&!showCorpus.checked)continue;const a=D[name];for(let i=0;i<a.length;i+=2){{const dx=sx(a[i],c.width)-mx,dy=sy(a[i+1],c.height)-my,q=dx*dx+dy*dy;if(q<bd){{bd=q;best=[name,i/2]}}}}}}return best}}
(async()=>{{M=await(await fetch('manifest.json')).json();L=await(await fetch(M.files.labels.path)).json();for(const n of ['base','corpus','queries'])D[n]=await bin(M.files[n]);
 const p=M.projection, vals=[['evidence',M.evidence_status],['probe FFR',p.ffr],['matched control',p.control_ffr],['retention',p.retention],['verdict',p.verdict],['corpus rows',p.corpus_rows],['queries',p.query_rows]];stats.innerHTML=vals.map(v=>`<span class="pill"><b>${{v[0]}}</b> ${{v[1]??'—'}}</span>`).join('');
 L.queries.forEach((q,i)=>{{const o=document.createElement('option');o.value=i;o.textContent=`${{q.id}} · ${{q.label}}`;query.appendChild(o)}});fit(['base','corpus','queries']);
 query.onchange=()=>{{selected=query.value===''?-1:+query.value;if(selected>=0){{const a=D.queries,x=a[2*selected],y=a[2*selected+1],dx=(view[1]-view[0])*.08,dy=(view[3]-view[2])*.08;view=[x-dx,x+dx,y-dy,y+dy]}}draw()}};
 for(const id of ['showBase','showCorpus','showQueries'])document.getElementById(id).onchange=draw;fitAll.onclick=()=>fit(['base','corpus','queries']);fitProbe.onclick=()=>fit(['corpus','queries']);reset.onclick=()=>{{selected=-1;query.value='';fit(['base','corpus','queries'])}};
 plot.onwheel=e=>{{e.preventDefault();const r=plot.getBoundingClientRect(),fx=(e.clientX-r.left)/r.width,fy=1-(e.clientY-r.top)/r.height,cx=view[0]+fx*(view[1]-view[0]),cy=view[2]+fy*(view[3]-view[2]),z=e.deltaY>0?1.18:.84;view=[cx+(view[0]-cx)*z,cx+(view[1]-cx)*z,cy+(view[2]-cy)*z,cy+(view[3]-cy)*z];draw()}};
 plot.onmousedown=e=>{{drag=[e.clientX,e.clientY,view.slice()]}};addEventListener('mouseup',()=>drag=null);addEventListener('mousemove',e=>{{if(drag){{const r=plot.getBoundingClientRect(),dx=(e.clientX-drag[0])*(drag[2][1]-drag[2][0])/r.width,dy=(e.clientY-drag[1])*(drag[2][3]-drag[2][2])/r.height;view=[drag[2][0]-dx,drag[2][1]-dx,drag[2][2]+dy,drag[2][3]+dy];draw();return}}const r=plot.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom){{tip.style.display='none';return}}const hit=nearest(e.clientX-r.left,e.clientY-r.top);if(!hit){{tip.style.display='none';return}}const q=L[hit[0]][hit[1]];tip.style.display='block';tip.style.left=(e.clientX+13)+'px';tip.style.top=(e.clientY+13)+'px';tip.textContent=`${{hit[0]}} · ${{q.id}} · ${{q.label}}`;}});addEventListener('resize',draw);
}})();
</script>'''
