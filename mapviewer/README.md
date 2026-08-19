# mapviewer — interactive viewer for latent-basemap map packs

Viewer **v2** for [PLAN6](../experiments/sandbox/PLAN6-interactive-maps.md): a
**fully static** pan/zoom map over a map pack. No backend, no database — the
page reads the same artifact directory whether it is served from gh-pages, from
GCS, or from `~/.agent` on this box.

Plain Vite + vanilla TypeScript. Layered rendering: WebGL2 base raster, a 2D
overlay canvas, DOM controls. Build output is ~58 KB of JS (the projection
runtime is separate and loaded only on request).

```
npm install
npm run fixtures     # synthesize a tiny pack into ./packs (uv + numpy + pillow)
npm run dev          # http://localhost:5195/
npm run build        # -> dist/
npm run smoke        # headless build verification + screenshots
bash scripts/deploy.sh
```

Node 24 is required: `source ~/.nvm/nvm.sh` first (system node is too old).

---

## What works

| # | feature | state |
| --- | --- | --- |
| 1 | pan/zoom density raster, client-side recomposition from per-corpus u32 planes, corpus toggles, dominant-corpus mode | **done** |
| 2 | hover bin highlight + side panel with composition percentages and sampled snippets | **done** |
| 3 | LOD points past the zoom cutover; deep points per-viewport | **done** |
| 4 | click a point → full text via the sidecar | **done**, but needs a published sidecar (see below) |
| 5 | explicit-ask loading with sizes, streamed progress bars, session byte counter | **done** |
| 6 | map switcher from `packs/index.json` | **done** |
| 7 | layered WebGL2 renderer — no base redraw on hover, no per-frame point work | **v2** |
| 8 | in-browser projection: text → MiniLM → the pack's map head → a dot on the map | **v2** |

Not built (deliberately out of scope): TF-IDF bin terms (PLAN6 build-order
item 4), the published-profile packer.

### Verified against

* the real `sandbox-2m-umap-md000-x4-fneg10` pack (2M rows, Z=3) and
  `sandbox-12500k-umap-md000-x2-fneg10` (12.5M rows, Z=4) served from
  `http://gsv.local:8800/basemap-maps/viewer/` — density, hover, LOD points,
  deep points and projection all work; text is unavailable because the sidecar
  is not published there (below).
* synthetic fixture packs, in both range modes.

---

## Rendering architecture (v2)

v1 drew everything into one 2D canvas and called `requestDraw()` from the
mousemove handler. Every pointer move therefore re-blitted every density tile
and re-iterated every LOD point (485k–1.8M `fillRect` calls per frame), which is
exactly the flicker and the lag the owner reported. v2 splits the work by how
often it changes:

| layer | element | tech | redrawn when |
| --- | --- | --- | --- |
| base | `#map` | WebGL2 (`src/gl.ts`) | camera moves, or data changes |
| overlay | `#overlay` | canvas 2D (`src/overlay.ts`) | pointer moves, selection, markers |
| DOM | `#controls`, `#hud`, `#side` | HTML | state changes |

* **Density raster.** The compose worker's RGBA goes straight into a GL texture,
  one `texImage2D` per tile **per recompose** — a corpus toggle or colour-mode
  change re-uploads, panning and zooming do not. Missing tiles still magnify the
  matching sub-rect of a coarser ancestor, now as a texture sub-rect.
* **Points.** `lod.bin` / `xy_id.bin` become interleaved VBOs uploaded once per
  data change and drawn as `gl.POINTS` with the camera in two uniforms. Corpus
  visibility is a bitmask uniform; the LOD cutover is a `uMaxMinZ` uniform *and*
  a draw-count prefix, because `lod.bin` is sorted by `min_zoom` — at z0 that is
  25k vertices instead of 485k.
* **Overlay.** Hover highlight, pinned bin, selected point and projection
  markers. `clearRect` + a handful of strokes; ~0.7 ms per draw.
* **One rAF.** `src/render.ts` owns the only `requestAnimationFrame` in the app
  and coalesces `markBase` / `markOverlay` / `markView` into at most one draw of
  each per frame. The side panel is coalesced the same way and hover rebuilds
  only the `bin` section, not the whole panel.
* **Fallback.** No WebGL2 → `src/raster2d.ts`, same interface and the same
  upload-once contract, with a point budget. The active backend is reported in
  the side panel and in `window.mapviewerConfig.backend`.

### `window.__renderStats` — the dev counter

```js
window.__resetRenderStats();
// … mouse around for 2 s …
window.__renderStats
// { baseDraws: 0, overlayDraws: 88, tileUploads: 0, pointUploads: 0,
//   frames: 88, maxOverlayMs: 0.7, maxBaseMs: 0, … }
```

`baseDraws`, `tileUploads` and `pointUploads` staying at 0 across a mousemove
sweep is the regression test for the flicker fix; `scripts/smoke.mjs` asserts
exactly that, plus zero long tasks > 50 ms while panning with points on.

---

## The range-request caveat (important)

**`http://gsv.local:8800/` does NOT honour `Range`.** It is
`python3 -m http.server` (`moonshine-web.service`), which answers `200` with the
whole body no matter what `Range` you send:

```
$ curl -s -D - -o /dev/null -H "Range: bytes=0-7" http://gsv.local:8800/sites/registry.json
HTTP/1.0 200 OK
Content-Length: 25204          # <- the entire file, not 8 bytes
```

So the viewer supports **two modes** and picks one per data source at load time
by probing a real binary in the pack:

| mode | when | how a byte range is read |
| --- | --- | --- |
| `http-range` | probe returns `206` (gh-pages, GCS, nginx, S3) | one `Range:` request |
| `chunked` | probe returns anything else | the covering `<path>.part{N}` files (fixed size, LRU-cached in memory) are fetched and spliced |
| `unavailable` | no Range **and** no published parts | ranged features are hidden and the panel says so |

The active mode per source is shown in the side panel and recorded in
`window.mapviewerConfig.sources`. `scripts/mirror_pack.py` generates the
`.part{N}` files when publishing a pack, so chunked mode works on any dumb
static host; a Range-capable host ignores the parts entirely.

Cost note: in chunked mode a KB-scale lookup costs one whole part (4 MB for real
packs). The panel reports what each lookup actually cost, so this is visible
rather than silent.

---

## Data contract

Two producers, one contract. `src/adapt.ts` normalises both into the viewer's
`Manifest` type:

* `basemap-map-pack-v1` — `fixtures/make_fixture_pack.py` (this repo). The
  reference implementation of the contract as the viewer reads it.
* `pack_format_version: "1"` — `experiments/mappack/map_pack.py` (the real
  builder). See `experiments/mappack/REPORT.md`.

Byte formats are identical between the two:

```
density/z{z}/{x}_{y}.{corpus}.u32   256*256 u32 LE, row-major, Y-DOWN
density/z{z}/{x}_{y}.png            combined YlGnBu render (convenience only —
                                    the viewer recomposes from the planes)
density/z{z}/index.json             which tiles/planes exist at this level
points/xy_id.bin                    8 B: u16 qx, u16 qy, u32 (corpus<<28 | row_id)
points/tile_index.u64               N_tiles+1 u64 offsets, row-major tile order
points/lod.bin                      9 B: the 8 B record + u8 min_zoom
bins/samples_z{k}.json              bin key -> [row_id, ...]
bins/snippets_z{k}.json             bin key -> ["text", ...]
text/offsets.u64, text/blob.utf8    row -> byte extent; one ranged read per lookup
model/map_head.onnx             ONNX map head, fp32, dynamic batch (optional)
model/models.json               mappack-models-v1: head io names, frame, parity
```

Where the two disagree, the adapter handles it:

* **offset units.** Real `tile_index.u64` and `lod.min_zoom_offsets` are **byte**
  offsets; the fixture's are record offsets. `tile_index.unit` disambiguates.
* **bin keys.** Real packs key `"{bin_x}_{bin_y}"` (files are already per level);
  the fixture keys `"{z}_{bin_x}_{bin_y}"`. `src/bins.ts` accepts both, plus a
  tile-nested form.
* **level max count.** Real levels carry `png_log_peak` = `log1p(max)`; the
  viewer wants the raw count.
* **empty tiles/planes are omitted** in real packs. The viewer reads
  `density/z{z}/index.json` and never 404-probes.
* **coordinates.** `u = (x - xmin)/(xmax - xmin)`, `v = (ymax - y)/(ymax - ymin)`
  — v is Y-down, matching tile, bin and quantization order. Real packs square
  the extent about the trimmed core; the viewer just uses `frame.extent`.
* **the map head is not in `files`.** It is exported by a separate step, so
  `scripts/mirror_pack.py` declares it in a `model` block on the published
  manifest and `src/adapt.ts` passes it through.

Bin previews are only sampled at `z0` in real packs, so hover walks **up** the
pyramid for snippets (the panel labels which level answered) while composition
counts come from the current level, also falling back up when a fine bin is
empty.

---

## In-browser projection (v2)

"Where would my text land?", wired into the side panel from
[`projection-poc/`](projection-poc/README.md):

```
text -> MiniLM (ONNX, onnxruntime-web WASM) -> 384-d unit vector
     -> the pack's model/map_head.onnx (fp32, dynamic batch) -> (x, y)
     -> the pack frame -> (u, v) -> a dot on the OVERLAY canvas
```

Behind the same explicit-ask pattern as the point tiers, labelled with real
sizes: **int8 encoder by default (~79 MB total)**, fp32 (~152 MB) behind an
`encoder precision` details menu. Markers persist in a clickable list (click to
fly to one), and `clear` empties it. Nothing leaves the browser; there is no
backend and no GPU.

### The frame contract (read before touching the placement maths)

The POC drew onto `density.png`, whose placement is **rotated 180°**
(`col = (1-u)*w`, `row = v*h`, from `models.json.frame.png_placement`). The
viewer does **not** draw that PNG — its world space is the pack's own
quantization grid:

```
u = (x - xmin) / (xmax - xmin)
v = (ymax - y) / (ymax - ymin)        # y measured downward
```

which is exactly `map_pack.quantize()`. So the dot is placed with the **pack
manifest** extent and **no x-flip**; applying the PNG rotation here would mirror
it across the map. Note also that `models.json.frame.extent` is the render frame
of `density.png` and differs from the pack's *squared* extent — the viewer uses
the pack's, and `src/projection.ts` says so at the top.

Verified two ways in `scripts/smoke.mjs`: numerically against the POC's python
reference, and empirically by flying to each landing and asserting the density
bin under it is non-empty (a mirrored frame would drop in-distribution text into
white space).

### Where the assets come from

| asset | source | how it is published |
| --- | --- | --- |
| `vendor/ort/`, `vendor/transformers/` | `projection-poc/vendor/` | `deploy.sh` rsync |
| `models/Xenova/all-MiniLM-L6-v2/` | `projection-poc/models/` | `deploy.sh` rsync |
| `model/map_head.onnx`, `model/models.json` | the pack itself | `mirror_pack.py` symlink |

`mirror_pack.py` also writes a `model` block into the published manifest
(`models_json`, `map_head`, `map_head_bytes`, `encoder`). The viewer reads that
block and never probes for the head — a 404 is a smoke-check failure, and packs
without a map head simply say *"this pack ships no map head"*.

### Gotchas carried over from the POC

- `ort.env.wasm.wasmPaths` must be an **absolute** URL: ORT resolves it against
  its own module URL, so a relative path becomes `/vendor/ort/vendor/ort/…` and
  session creation fails with "no available backend".
- `ort.env.wasm.numThreads = 1` — threads need COOP/COEP, which a plain static
  host (gh-pages, `python3 -m http.server`) cannot send.
- transformers.js pinned to **3.x**, tokenizer only, `allowRemoteModels = false`
  — 4.x moves tokenization into a second WASM package.
- Embedding semantics must match sentence-transformers: mean-pool over the
  attention mask, then L2 normalise.

---

## Layout

```
src/
  main.ts        wiring, controls, side panel, loading-UX asks, projection panel
  map.ts         camera, scene building, pointer interaction, hit testing
  render.ts      the single rAF loop + window.__renderStats
  gl.ts          WebGL2 base layer: tile textures + point sprite buffers
  raster2d.ts    canvas-2D fallback for the same interface
  overlay.ts     2D overlay: hover, selection, projection markers
  density.ts     tile store: fetch planes, cache, drive the compose worker
  worker/compose.worker.ts   u32 planes -> RGBA (combined / dominant-corpus)
  points.ts      LOD tier + deep tier + the LOD spatial index (buildLodIndex)
  projection.ts  text -> MiniLM (ONNX/WASM) -> map head -> world coords
  bins.ts        per-bin samples/snippets with pyramid fallback
  text.ts        sidecar lookup: 16 B into offsets.u64, then the blob extent
  net.ts         session byte accounting, progress streams, RangeReader
  adapt.ts       manifest normalisation for both producers
  types.ts, palette.ts, style.css
public/vendor  -> ../projection-poc/vendor    (symlink, gitignored)
public/models  -> ../projection-poc/models    (symlink, gitignored)
fixtures/make_fixture_pack.py   synthetic pack generator (contract reference)
scripts/deploy.sh               build + publish to ~/.agent/basemap-maps
scripts/mirror_pack.py          publish one real pack (symlinks + .part files)
scripts/smoke.mjs               headless verification + screenshots
```

---

## Deploying

```bash
bash scripts/deploy.sh
# -> http://gsv.local:8800/basemap-maps/viewer/
```

Produces:

```
~/.agent/basemap-maps/viewer/          the built site
~/.agent/basemap-maps/viewer/packs ->  ../packs
~/.agent/basemap-maps/packs/<map_id>/  real packs mirrored (symlinks + .partN)
~/.agent/basemap-maps/packs/index.json merged index driving the map switcher
```

Real packs are mirrored, not copied: `density/`, `bins/` and `model/` are
symlinks to `/data/latent-basemap/mappacks/<id>/`, `points/` is a real directory
holding symlinks plus the generated `.part{N}` slices, and `manifest.json` is a
rewritten copy carrying the `chunking` block. Nothing is ever written into
`/data/latent-basemap/mappacks/`.

Plus, unless `PROJECTION=0`:

```
~/.agent/basemap-maps/viewer/vendor/   onnxruntime-web + transformers.js (13 MB)
~/.agent/basemap-maps/viewer/models/   MiniLM fp32 + int8 + tokenizer (109 MB)
```

These are rsynced from `projection-poc/{vendor,models}` (gitignored, and
deliberately **not** part of `vite build` — `build.copyPublicDir` is off so a
rebuild doesn't copy 122 MB). `npm run dev` serves the same paths through the
`public/` symlinks.

The site rsync uses `--delete` but excludes `packs`, `vendor`, `models` and
`round-*`, because other tools publish sibling directories into the same viewer
root.

Env: `DEST`, `REAL_PACKS`, `WITH_FIXTURES=0`, `WITH_SIDECAR=1`, `PROJECTION=0`.

### Text sidecar

Sidecars are per **substrate**, live outside the pack
(`/data/latent-basemap/textsidecar/<substrate_key>/`) and are large — 923 MB for
the 2M substrate, 5.7 GB for the 12.5M one. `deploy.sh` skips them by default,
and the viewer then reports *"no text sidecar reachable from this source"*
instead of offering broken lookups.

To enable click-through text locally (doubles the sidecar's disk usage, because
chunked mode needs `.part` files):

```bash
WITH_SIDECAR=1 bash scripts/deploy.sh
```

On a Range-capable host no parts are needed — link the sidecar and set
`--max-split-bytes 0`.

### gh-pages

`vite.config.ts` sets `base: "./"`, so `dist/` works from any subdirectory. Copy
or symlink the publishable tiers (density pyramid + bin previews + LOD points)
into `dist/packs/` and push. Deep points belong on GCS with a CORS rule for the
gh-pages origin; point the viewer at them with `?packs=<https url>`.

For projection on gh-pages, copy `projection-poc/{vendor,models}` next to
`index.html` (real copies — gh-pages does not follow symlinks); `src/projection.ts`
resolves both against `document.baseURI`. COOP/COEP are not available there,
which is why `numThreads = 1` is not negotiable.

---

## URL parameters

| param | meaning |
| --- | --- |
| `?packs=<url>` | pack root to read `index.json` from (default `packs/`) |
| `?map=<map_id>` | open a specific pack |

## Smoke check

```bash
node scripts/smoke.mjs                       # chunked mode (mimics gsv:8800)
node scripts/smoke.mjs --range               # http-range mode
node scripts/smoke.mjs --no-projection       # skip the ~79 MB model download
node scripts/smoke.mjs --url http://gsv.local:8800/basemap-maps/viewer/ \
                       --map sandbox-2m-umap-md000-x4-fneg10 --out shots-real
node scripts/smoke.mjs --url ... --projection-precision fp32
```

Uses `playwright-core` with the cached `chrome-headless-shell` and
`LD_LIBRARY_PATH=~/.cache/lib` for `libasound.so.2`. It drives the real UI —
hover, corpus toggle, dominant mode, both point tiers, a point click, the
projection panel, the map switcher — asserts **zero console errors, page errors,
failed requests and 4xx/5xx responses**, and writes numbered screenshots.

The v2 checks are the ones worth watching:

| check | what it proves |
| --- | --- |
| `mousemove: zero base-layer redraws` | a 2 s scripted sweep leaves `baseDraws`/`tileUploads`/`pointUploads` at 0 |
| `mousemove: overlay layer did redraw` | the overlay *did* follow the cursor (~88 draws, < 1.5 ms each) |
| `pan with LOD points: no long task > 50 ms` | PerformanceObserver longtask count is 0 while dragging with points on |
| `pan re-used the uploaded point buffer` | no VBO re-upload during a pan |
| `projection matches python reference` | browser (x, y) vs `projection-poc/reference/comparison.json` |
| `projection frame contract` | u right, v down, PACK extent, no PNG rotation |
| `projections land in non-empty density bins` | empirical check that the dot is not mirrored |

Latest results (2026-08-14, `chrome-headless-shell` + SwiftShader, so the GPU
numbers are a worst case):

| pack | checks | mousemove | pan | projection (int8) |
| --- | --- | --- | --- | --- |
| fixture-blobs (250k, chunked + range) | 16/16 | 0 base draws / 88 overlay | 0 long tasks | n/a |
| sandbox-2m (2M, Z=3) | 22/22 | 0 base draws / 87 overlay | 0 long tasks | worst Δxy 0.0934 = **0.139 %** of extent diag |
| sandbox-12500k (12.5M, Z=4) | 20/20 | 0 base draws / 88 overlay | 0 long tasks | 5/5 land in non-empty bins |

With the fp32 encoder the 2M pack reproduces the python reference exactly
(worst Δxy `0.0000`, i.e. below float printing precision).
