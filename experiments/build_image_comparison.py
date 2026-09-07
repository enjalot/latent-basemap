#!/usr/bin/env python3
"""Refresh the completed MONET 2D comparisons, CPU only, with exact-count tiles.

Run with the repository venv. Sources are read-only; publication goes to
~/.agent/basemap-maps. Re-running replaces these five pages and the comparison
index. No GPU inference, model fitting, or 3D packaging is performed.
"""
import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import map_tiles as mt
import map_viewer as mv
import build_recent_maps_gallery as recent

SB = Path('/data/latent-basemap/sandbox')
SITE = Path.home() / '.agent/basemap-maps'
SPECS = [
    ('dino-104m', 'DINOv2 · 103.82M images · 2M-trained head',
     'fullcorpus-dino-2d/coords.f32.npy', 2008321, 1536),
    ('clip-104m', 'CLIP · 103.82M images · 4M-trained head',
     'monet-clip-fullcorpus-proj-4m-20260905/coords.f32.npy', 4000000, 512),
    ('dino-2m', 'DINOv2 · 2.01M training images · 1536 dimensions',
     'monet-random-dino-2m/champion-bs16k/coordinates.npy', 2008321, 1536),
    ('dino-6m', 'DINOv2 · 6M training images · 1536 dimensions',
     'monet-random-dino-6m/champion-bs16k/coordinates.npy', 6000000, 1536),
    ('dino-6m-pca768', 'DINOv2 · 6M training images · PCA 768 dimensions',
     'monet-random-dino-6m-pca768/champion-bs16k/coordinates.npy', 6000000, 768),
]


class Source(mt.MapSource):
    """Bound every binning allocation to a million rows, including flat 104M files."""
    def __init__(self, path):
        self.array = np.load(path, mmap_mode='r')
        if self.array.ndim != 2 or self.array.shape[1] != 2:
            raise ValueError(f'Expected a 2D map: {path}')
        self.nrows = len(self.array)
        self._extent = None
        self.cache_dir = None

    def iter_chunks(self):
        for start in range(0, self.nrows, 1_000_000):
            block = self.array[start:start + 1_000_000]
            if not np.isfinite(block).all():
                raise ValueError(f'Nonfinite coordinates at block {start}')
            yield start, block


def build(spec):
    key, title, relative, trained, dimensions = spec
    path = SB / relative
    source = Source(path)
    expected = 103816750 if key.endswith('104m') else trained
    assert source.nrows == expected, (key, source.nrows, expected)
    map_id = 'sandbox-' + (key if key.endswith('104m') else path.parent.parent.name)
    out = SITE / 'viewer' / map_id
    data = out / 'data'
    data.mkdir(parents=True, exist_ok=True)
    receipt = path.parent / 'manifest.json'
    if not receipt.exists():
        receipt = path.parent.parent / 'ladder-score.json'
    if not receipt.exists():
        receipt = SB / 'd2-rescore-20260905.json'
    evidence = json.loads(receipt.read_text())
    if key.endswith('104m'):
        assert evidence['dim'] == 2 and evidence['n_rows'] == expected
    scores = {}
    if key.startswith('dino-6m'):
        scores = {'v2 FFR (own training truth)': evidence['v2_ffr_own_truth'],
                  'Held-out reception (6M reference)': evidence['heldout_reception_recall@15']}
    elif key == 'dino-2m':
        score = next(x for x in evidence['maps'] if x['dataset'] == 'monet-random-dino-2m'
                     and x['arm'] == 'champion-bs16k')
        scores = {'v2 FFR (own training truth)': score['quick_ffr_v2']}
    extent = list(source.extent())
    grids = mt.bin_all_levels(source, mv.LEVELS + mv.FINE_LEVELS, extent)
    emitted, tiled = [], []
    for level, (idx, count) in grids.items():
        assert int(count.astype(np.uint64).sum()) == expected, (key, level)
        result = mt.write_grid_auto(str(data), 'all', level, idx, count)
        if result['tiled']:
            tiled.append({'level': level, 'split': result['split']})
        else:
            emitted.append(level)
    layer = {'key': 'all', 'label': 'All images', 'kind': 'grid', 'rows': expected,
             'levels': sorted(emitted), 'tiled_levels': sorted(tiled, key=lambda x: x['level'])}
    note = f'{trained:,}-row training set; {dimensions} input dimensions; exact-count density, no image thumbnails'
    manifest = {'schema': mv.MANIFEST_SCHEMA, 'generated_utc': datetime.now(timezone.utc).isoformat(),
                'map_id': out.name, 'title': title, 'rows_total': expected, 'rows_note': note,
                'map_kind': 'sandbox', 'extent': extent, 'levels': mv.LEVELS,
                'sample_level': mv.SAMPLE_LEVEL, 'super_tile': mv.SUPER_TILE,
                'samples_available': False, 'layers': [layer], 'metrics': scores, 'skipped': [],
                'source': {'coordinates': 'gsv:' + str(path), 'bytes': path.stat().st_size,
                           'mtime_ns': path.stat().st_mtime_ns, 'receipt': 'source-receipt.json',
                           'receipt_sha256': hashlib.sha256(receipt.read_bytes()).hexdigest()},
                'training_rows': trained, 'input_dimensions': dimensions}
    (data / 'source-receipt.json').write_bytes(receipt.read_bytes())
    (data / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    mv._write_thumbnail(data / 'grid-all-256.bin', out / 'thumb.png')
    cfg = {'dataDir': 'data', 'back': '../../latest-images/', 'manifest': 'data/manifest.json',
           'assets': '../../image-assets', 'map_id': out.name, 'title': title}
    page = mv._instantiate(mv._load_template(), cfg).replace('../../assets/', '../../image-assets/')
    page = page.replace('id="tabs"', 'id="tabs" style="display:none"')
    (out / 'index.html').write_text(page)
    print(f'Built {key}: {expected:,} rows; all 7 grid levels conserve counts', flush=True)
    return {'key': key, 'title': title, 'url': '../viewer/' + out.name + '/',
            'rows': expected, 'trained': trained, 'dimensions': dimensions, 'scores': scores}


def publish_index(maps):
    out = SITE / 'latest-images'
    out.mkdir(exist_ok=True)
    cards = ''.join(f'<a href="{m["url"]}"><img src="{m["url"]}thumb.png" alt=""><b>{html.escape(m["title"])}</b></a>' for m in maps)
    rows = ''.join('<tr><td>' + html.escape(m['title']) + '</td><td>' +
                   '</td><td>'.join(f'{v:.5f}' for v in m['scores'].values()) +
                   ('</td><td>—' if len(m['scores']) == 1 else '') + '</td></tr>' for m in maps[2:])
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Latest MONET image maps — DINO and CLIP</title>
<style>body{font:16px/1.5 system-ui;margin:0;background:#f8fafc;color:#182333}main{max-width:1600px;margin:auto;padding:24px}h1{margin:0}p{max-width:1050px}a{color:#185c91}button,select{font:inherit;padding:8px;margin:4px 8px 4px 0}button{cursor:pointer}.panes{display:grid;grid-template-columns:1fr 1fr;gap:16px}.pane{min-width:0}iframe{width:100%;height:720px;border:1px solid #bcc8d3;background:white}select{max-width:100%}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}.cards a{background:white;padding:12px;text-decoration:none}.cards img{width:100%}table{border-collapse:collapse}td,th{text-align:left;padding:10px;border-bottom:1px solid #cdd5df}.scroll{overflow:auto}.meta{color:#526375;font-size:14px}@media(max-width:850px){.panes{grid-template-columns:1fr}iframe{height:620px}main{padding:14px}}</style>
<main><a href="../sandbox/">All recent experiments</a><h1>Latest MONET image maps</h1>
<p>Compare the full 103,816,750-image corpus in DINOv2 and CLIP, or inspect the completed DINO training ladder. These are interactive <b>2D density maps</b>: every image contributes to the counts; thumbnails and individual-image browsing are not included here.</p>
<p>The full-corpus DINO export uses the <b>2M-trained 1536-dimensional head</b>; CLIP uses the <b>4M-trained 512-dimensional head</b>. The two 6M DINO heads are complete, but their full-corpus projections are not yet available. There is no completed 12M DINO rung in this snapshot.</p>
<button id="full">DINO vs CLIP · 103.8M</button><button id="ladder">DINO · 6M full vs PCA</button><button id="growth">DINO · 2M vs 6M</button>
<p class="meta">Each pane pans and zooms independently. Frames are not aligned; the 2M and 6M views contain different populations. Compare structure, not point displacement. Density intensity is relative to each map.</p>
<div class="panes"><div class="pane"><label>Left map <select id="left"></select></label> <a id="left-link">Open alone</a><iframe id="left-frame" title="Left map"></iframe></div><div class="pane"><label>Right map <select id="right"></select></label> <a id="right-link">Open alone</a><iframe id="right-frame" title="Right map"></iframe></div></div>
<h2>Completed DINO ladder</h2><div class="scroll"><table><thead><tr><th>Training map</th><th>v2 FFR · own truth</th><th>Held-out reception</th></tr></thead><tbody>ROWS</tbody></table></div>
<p class="meta">The 6M arms use the same training draw and held-out validation population; PCA changes the high-dimensional neighborhood truth. Reception uses 6,000 held-out queries into the 6M training reference. FFR uses a 0.1% map-neighbor budget, which grows with reference size. These scores are not a fixed-budget scaling comparison.</p>
<h2>Individual maps</h2><div class="cards">CARDS</div><p class="meta">Updated STAMP. Coordinates and score receipts are linked in each map's <a href="catalog.json">manifest / catalog</a>. Existing <a href="http://gsv.local:5300/?dataset=monet-clip-basemap-full-4m-512">CLIP 103.8M image browser</a>.</p></main>
<script>const maps=MAPDATA;const params=new URLSearchParams(location.search);function set(side,key){const m=maps.find(m=>m.key===key)||maps[0];document.getElementById(side).value=m.key;document.getElementById(side+'-frame').src=m.url;document.getElementById(side+'-link').href=m.url;params.set(side,m.key);history.replaceState(null,'','?'+params)}for(const side of ['left','right']){const s=document.getElementById(side);for(const m of maps){s.add(new Option(m.title,m.key))}s.onchange=()=>set(side,s.value)}function pair(a,b){set('left',a);set('right',b)}document.getElementById('full').onclick=()=>pair('dino-104m','clip-104m');document.getElementById('ladder').onclick=()=>pair('dino-6m','dino-6m-pca768');document.getElementById('growth').onclick=()=>pair('dino-2m','dino-6m');const left=params.get('left')||'dino-104m',right=params.get('right')||'clip-104m';pair(left,right);</script></html>'''
    page = page.replace('ROWS', rows).replace('CARDS', cards).replace('STAMP', datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')).replace('MAPDATA', json.dumps(maps))
    (out / 'index.html').write_text(page)
    (out / 'catalog.json').write_text(json.dumps(maps, indent=2))
    recent.build_index(recent._scan_cards())


def main():
    assets = SITE / 'image-assets'
    assets.mkdir(exist_ok=True)
    original = Path(__file__).parent / 'viewer_assets'
    (assets / 'viewer.css').write_bytes((original / 'viewer.css').read_bytes())
    js = (original / 'viewer.js').read_text()
    js = js.replace('const MIN_CELL_PX = 7;', 'const MIN_CELL_PX = 1;')
    js = js.replace('async function getSamples(cx, cy) {', 'async function getSamples(cx, cy) {\n    if (S.manifest.samples_available === false) return null;')
    js = js.replace('Hover a bin for row count and text samples.', 'Hover a bin for its image count. This density view has no thumbnail samples.')
    (assets / 'viewer.js').write_text(js)
    publish_index([build(spec) for spec in SPECS])
    print('http://gsv.local:8800/basemap-maps/latest-images/', flush=True)


if __name__ == '__main__':
    main()
