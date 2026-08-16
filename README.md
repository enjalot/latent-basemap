# latent-basemap

**A basemap is a world map for an embedding space.** Train a small neural
projection head once over a large, mixed corpus, and every text you embed
afterwards lands on the same stable 2D frame — in milliseconds, with no
graph construction, even in the browser. The frame is the product: you can
learn its geography, link to places on it, and project new data onto it for
as long as the encoder exists.

<!-- TODO(publish): hero image — the Procrustes-aligned scale ladder -->

This repo is the home for:

- **trained basemap heads** for `all-MiniLM-L6-v2` (384-dim) at 2M–100M
  training rows <!-- TODO(D7): HF links when model repos are named -->
- **the training recipe + code** to build your own basemap over any
  embedding collection
- **map packs + a static viewer** for exploring the maps (density, points,
  text reachback, in-browser projection)
- **the paper** and its interactive companions
  <!-- TODO(D2): gh-pages links — /maps /concepts /ui /paper -->

## Use a trained basemap

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from basemap.pumap.parametric_umap.core import ParametricUMAP

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
basemap = ParametricUMAP.load("checkpoints/basemap-minilm.pt")  # TODO(D7): HF path

texts = ["a recipe for sourdough bread", "gradient descent convergence proof"]
xy = basemap.transform(encoder.encode(texts, normalize_embeddings=True))
# xy: (2, 2) float32 — coordinates on the shared frame
```

`transform` batches internally and never materializes its input, so a
memory-mapped `(N, 384)` array projects at any N. No neighbor graph, no
refitting: projection is a single forward pass (~11.8M-param MLP).

The same head runs **in the browser** via ONNX (`experiments/mappack/onnx/`
exports it; the viewer's projection panel pairs it with a quantized
in-browser MiniLM — ~80 MB total, no server).

## Explore the maps

The viewer (`mapviewer/`) renders map packs — self-contained static-file
bundles (density tiers, Morton-sorted point tiles, LOD, text sidecars)
served by any host that supports HTTP range requests.

<!-- TODO(D2): live demo links once gh-pages + GCS deep points are up -->

Locally: `cd mapviewer && npm install && npm run dev`, then point it at a
pack built with `experiments/mappack/map_pack.py`.

## Train your own

You need two inputs:

1. **Embeddings** — an `(N, D)` float32 array (memmap fine). Ours are
   384-dim MiniLM over 120-token chunks.
2. **A k-NN graph** — exact-k15 fuzzy edges over those embeddings. We build
   ours with GPU NN-descent (cuVS) and verify recall against brute-force
   truth on a probe; any builder works if recall is honest.

Then:

```python
from basemap.pumap.parametric_umap.core import ParametricUMAP

model = ParametricUMAP(
    n_components=2, hidden_dim=2048, n_layers=3, n_neighbors=15,
    architecture="residual_bottleneck",
    # the recipe (see the paper for why each field matters):
    low_dim_kernel="umap", a=1.9328, b=0.7905,   # umap kernel at min_dist 0
    fneg_weight=1.0,                             # fog-targeted negative reweighting
    # dose: total optimizer updates ≈ 2× successful_updates_for_edges(E)
)
model.fit(X, precomputed_edges_path="edges-k15-fuzzy.npz", random_state=42)
model.save("my-basemap.pt")
```

The four treatment fields that define our promoted recipe, all validated by
pre-registered gates on 13-seed families:

| field | value | what it does |
| --- | --- | --- |
| output kernel | umap `(a=1.9328, b=0.7905)` | min_dist-0 attraction/repulsion shape |
| dose | ×2–×4 draws/edge | training budget in graph-relative units |
| fneg | weight 1.0, band `[0.1,0.4]`×p90 radius | clears low-density "fog" without collapse |
| edge sampling | uniform | positives drawn uniformly over edges |

Cost on one RTX 5090 (32 GB), measured:

| rows | dose | wall |
| ---: | :--: | ---: |
| 2M | ×4 | ~50 min |
| 6.25M | ×4 | ~2.5 h |
| 12.5M | ×2 | ~2.5 h |
| 50M | ×2 | ~12 h (host-int8) |
| 100M | ×2 | ~24 h (est., host-int8) |

Past ~20M rows fp32 no longer fits VRAM: pass `x_residency="host_int8"` to
keep the substrate in host RAM as int8 rows + fp16 per-row scales (bit-exact
encoding, map-level fidelity validated against fp32 siblings).

Watch two instruments while you train (`experiments/metrics/`): **collapse**
(`r10·√N` — healthy maps sit near ~1.0; a slide toward 0 is the failure that
looks fine in a thumbnail) and **fog** (low-density mass — the haze of
misplaced points between clusters). Both are cheap and catch the two ways a
map lies to you.

## The paper

<!-- TODO(D6): PDF + interactive links + BibTeX once title/authors settle -->

Draft source lives in `paper/`. The experimental evidence behind every claim
is sealed under a pre-registration protocol in the research logs
(`latent-labs`, separate repo).

## Repo map

Supported surface:

- `basemap/pumap/parametric_umap/` — the model (fit / transform / save / load)
- `experiments/mappack/` — pack builder, text sidecars, ONNX export
- `mapviewer/` — the viewer + in-browser projection
- `experiments/metrics/` — collapse / fog / purity instruments

Research scaffolding (kept for reproducibility, not needed to train):
`experiments/sandbox/` (the knob-sweep program), `*_modal.py` (legacy cloud
scripts), round/receipt tooling referenced by the paper's protocol section.

## License & citation

<!-- TODO(owner): license for code / models / packs; BibTeX after arXiv -->
