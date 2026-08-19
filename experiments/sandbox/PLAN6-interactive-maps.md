# PLAN6 spec: interactive map viewing — bins with previews, points on demand, text reachback

**Status: SPEC ONLY (owner-directed 2026-08-14). No building yet. No GPU
anywhere in this plan.** Everything is framed as precompute/storage
tradeoffs and sampling-vs-lookup choices, designed for reuse across every
future map and run.

## Design keystone: the gilbert pattern

Everything is **static files + HTTP range requests** — no backend, no
database. The same artifact directory ("map pack") serves two profiles:
- **local hands-on** (gsv.local:8800): full-fidelity tiers on /data
- **published demo** (HF/CDN): the small tiers only, same viewer code

This is the architecture gilbert already proved at genome scale: layers on
a static CDN, range requests into binary blobs, API only where unavoidable
(here: nowhere).

## The map pack (one per map, built by `map_pack.py`, CPU-only)

Input: a registry map_id → coordinates.npy + the substrate's provenance
(corpus, shard, row per map-row) + /data/chunks parquets. Output, versioned:

```
pack/
  manifest.json            map identity, extent, frame transform, tier inventory
  density/z{0..Z}/…        multi-res count tiles (PNG render + u32 counts)
  points/tile-index.bin    Morton/tile-sorted point file + per-tile offsets
  points/xy-id.bin         quantized xy (u16 in-tile) + row id, sorted by tile
  bins/samples-z{k}.json   per-bin sampled row ids (reservoir K≈4)
  bins/snippets-z{k}.json  per-bin sampled text snippets (~140 chars)
  bins/terms-z{k}.json     per-bin TF-IDF top terms (labels, no clustering)
  text/offsets.u64         (local tier) row → byte offset
  text/blob.txt            (local tier) all chunk texts, one range request per lookup
```

### The one sort that makes everything cheap

Sort row ids once by spatial tile (Morton order at the finest zoom). Then:
- "all points in this viewport" = a few contiguous byte ranges (range request)
- per-bin samples = the first K rows of each bin's run (or reservoir for
  unbiasedness)
- density tiles = a counting pass over the same order
100M rows sort + bin on CPU ≈ 10–20 min, once per map.

### Storage by tier (the sampling-vs-lookup dial)

| tier | contents | 2M map | 100M map | publishable? |
| --- | --- | --- | --- | --- |
| **preview** | density pyramid + coarse-bin (256²) samples/snippets/terms | ~15 MB | ~30 MB | yes — this IS the demo |
| **points** | LOD point sample (density-stratified, ~1–5M pts, min-zoom tagged) | ~25 MB | ~60 MB | yes |
| **deep-points** | full tile-sorted xy+id | 24 MB | 1.2 GB | optional (HF can host; range requests work) |
| **text sidecar** | offsets + full chunk text blob, PER SUBSTRATE (shared by all maps on it) | ~1 GB | ~48 GB | local only; publish substitutes sampled-text JSON (~50–150 MB) |

Key reuse point: the text sidecar is per-SUBSTRATE, not per-map — build it
once for the 100M substrate and every map trained on it reaches back through
the same blob. Same for provenance. Only coordinates-derived tiers rebuild
per map (~minutes).

### Sampling vs lookup, resolved per interaction

- **"What's in this region?" (hover, coarse)** → precomputed: bin snippets +
  TF-IDF terms at z≤k. Zero latency, tiny storage, works published.
- **"Show me points here" (zoomed)** → LOD sample tiles first; past the
  cutover zoom, range-request the deep-points file (local + published-if-
  hosted).
- **"Show me THIS point's text" (click)** → lookup: one range request into
  the text sidecar (local), or the sampled-snippet fallback (published).
  Never precompute all text into the viewer tier; never query parquet live
  (row-group reads are MBs per hit — the sidecar's offset design is the fix).

### Bin summaries without clustering

Per-bin TF-IDF top terms (unigram/bigram) over the bin's sampled texts,
aggregated up the pyramid — the latent-taxonomy top-tokens machinery pattern,
pure CPU, storage trivial. Gives latent-scope-style region labels from the
grid itself; the visually-obvious clusters label themselves through their
dominant bins. Optional later (idle-GPU, explicitly out of scope now):
LLM labels per coarse region via ollama.

## The viewer (one codebase, two data profiles)

Pan/zoom canvas (deck.gl or hand-rolled, same class as the compare app):
- density tiles as the base raster (the bin style stays)
- crossfade to points past the LOD cutover
- hover: bin preview panel (snippets + terms), latent-scope-style side panel
- click: full text via sidecar range request (local) / snippet (published)
- per-corpus tinting via provenance (color by fineweb/RPJ/pile/code — the
  "what data goes where" question answered visually)
- **stretch, flagged for owner interest**: in-browser "where would my text
  land" — MiniLM (22M params) + the map head (11.8M) both run under ONNX in
  the browser (transformers.js); type text, see it land. Zero backend, zero
  GPU, pure demo magic. The R0026 ONNX parity work is the precedent; needs
  an export path for umap-kernel/fneg checkpoints (CPU, small).

Registry integration: map pages and the kernels page grow an "explore" link
per map once its pack exists.

## latent-scope: project datasets through a basemap

Prior art exists: the latent-scope `basemap-projection` branch (ls-basemap +
ls-umap-align, checkpoints at /data/checkpoints/pumap). Spec to revive it
against current reality:
1. **Projector loading**: support current sandbox/round checkpoints —
   requires the kernel-era fields (umap kernel a/b, kernel_alpha, fneg
   params — the load() provenance fix already landed). CPU transform only;
   MiniLM embedding of user datasets is CPU-fine at latent-scope dataset
   sizes.
2. **Frame consistency**: the pack manifest carries the map's canonical
   frame (and its Procrustes transform into the reference frame used on the
   kernels page); latent-scope projects INTO that frame and draws the
   pack's density tiles as the underlay — user data over the basemap, the
   lens-cover product in miniature.
3. **Works with sandbox models now**: nothing here waits for promotion —
   any checkpoint in the registry (or sandbox dirs) is loadable; the pack
   provides the underlay.
4. Confidence overlay (later, from the master plan): TwoNN-based per-point
   projection confidence tinting.

## Build order when green-lit (all CPU)

1. `map_pack.py` core: sort + density pyramid + bin samples/snippets +
   text sidecar (per substrate). First target: the 12.5M fneg map (the
   current showpiece) + the 2M winner.
2. Viewer v1: tiles + hover previews + corpus tinting, served at
   gsv.local:8800 next to the kernels page.
3. Deep-points + click-through text.
4. TF-IDF bin terms.
5. latent-scope projector revival (parallel track, independent).
6. Publish profile + (stretch) in-browser projection.

## Owner decisions (2026-08-14) — all open choices resolved

- **Hosting: gh-pages static site, likely in latent-basemap.** Small tiers
  (density pyramid, bin previews/terms, LOD points, model files) live on
  gh-pages; **deep-points on GCS** with range requests. GCS needs a CORS
  config for the gh-pages origin; egress stays modest because deep-points
  are fetched per-viewport as byte ranges — a session touches a few MB,
  never the whole file.
- **In-browser projection ships in v1.** MiniLM runs officially in
  transformers.js (Xenova ONNX export exists); the map head (~12M params)
  exports to ONNX (R0026 precedent; a small CPU export path for
  umap-kernel/fneg checkpoints is a build item). Type text → embed → project
  → marker lands on the map. Model assets ~35–50 MB one-time, Cache-API
  cached.
- **Loading UX: explicit-ask with size labels and progress.** Instant on
  load: density pyramid (few MB). On-interaction: bin preview JSONs. Behind
  explicit asks with stated sizes + progress bars (content-length streams):
  "point mode" (per-viewport GCS ranges, ~X MB as you pan), "load projection
  models (~45 MB)", sampled-text pack. Per-click text lookups are KB-scale
  range requests — no ask needed.

## Corpus filter/color — bin AND point level (owner question, answered yes)

Provenance gives every row a corpus tag (u8). Two precomputed structures
make filtering/coloring free at runtime:

- **Point level: pack corpus into the id word.** Row ids ≤100M fit 27 bits,
  so id+corpus pack into one u32 — point record = xy(u16×2) + packed u32 =
  **8 B/point** (revised: 100M deep-points = 800 MB, not 1.2 GB). Filtering
  = mask on the corpus bits; coloring = palette on them. Same scheme in the
  LOD sample tier.
- **Bin level: per-corpus count planes.** Each density level stores counts
  per corpus (4×u32/bin ≈ +16 MB/level at 1024²; pyramid ≈ +21 MB total).
  The client recomposes density from any subset of planes — corpus toggles,
  color-by-dominant-corpus, and hover previews showing bin composition
  ("62% fineweb / 21% pile / …") all come from the same planes with no
  server and no re-render pipeline.

This also generalizes forward: projected user datasets (latent-scope
overlays) and future register slices (PLAN5's Reddit/QA corpora) are just
additional planes/tags in the same scheme.
