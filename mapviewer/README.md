# mapviewer — interactive viewer for latent-basemap map packs

Viewer v1 for [PLAN6](../experiments/sandbox/PLAN6-interactive-maps.md): a
**fully static** pan/zoom map over a map pack. No backend, no database — the
page reads the same artifact directory whether it is served from gh-pages, from
GCS, or from `~/.agent` on this box.

Plain Vite + vanilla TypeScript + canvas 2D. Build output is ~35 KB of JS.

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

Not built (deliberately out of v1 scope): TF-IDF bin terms (PLAN6 build-order
item 4), in-browser projection (the packs ship `model/map_head.onnx`, nothing
reads it yet), the published-profile packer.

### Verified against

* the real `sandbox-2m-umap-md000-x4-fneg10` pack (2M rows, Z=3) served from
  `http://gsv.local:8800/basemap-maps/viewer/` — density, hover, LOD points and
  deep points all work; text is unavailable because the sidecar is not published
  there (below).
* synthetic fixture packs, in both range modes.

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

Bin previews are only sampled at `z0` in real packs, so hover walks **up** the
pyramid for snippets (the panel labels which level answered) while composition
counts come from the current level, also falling back up when a fine bin is
empty.

---

## Layout

```
src/
  main.ts        wiring, controls, side panel, loading-UX asks
  map.ts         camera, canvas renderer, pointer interaction, hit testing
  density.ts     tile store: fetch planes, cache, drive the compose worker
  worker/compose.worker.ts   u32 planes -> RGBA (combined / dominant-corpus)
  points.ts      LOD tier + deep tier (per-viewport ranged reads)
  bins.ts        per-bin samples/snippets with pyramid fallback
  text.ts        sidecar lookup: 16 B into offsets.u64, then the blob extent
  net.ts         session byte accounting, progress streams, RangeReader
  adapt.ts       manifest normalisation for both producers
  types.ts, palette.ts, style.css
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

Env: `DEST`, `REAL_PACKS`, `WITH_FIXTURES=0`, `WITH_SIDECAR=1`.

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
node scripts/smoke.mjs --url http://gsv.local:8800/basemap-maps/viewer/ \
                       --map sandbox-2m-umap-md000-x4-fneg10 --out shots-real
```

Uses `playwright-core` with the cached `chrome-headless-shell` and
`LD_LIBRARY_PATH=~/.cache/lib` for `libasound.so.2`. It drives the real UI —
hover, corpus toggle, dominant mode, both point tiers, a point click, the map
switcher — asserts **zero console errors, page errors, failed requests and 4xx/5xx
responses**, and writes numbered screenshots.
