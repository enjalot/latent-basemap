---
title: "Latent Basemap: Learning Large UMAPs"
author:
  - Ian Johnson
  - Claude
  - ChatGPT
bibliography: references.bib
# Build: single source for BOTH the arXiv PDF (pandoc + tectonic, sae-papers
# pipeline) and the interactive gh-pages version (moonshine still renders the
# same source with figures swapped for interactives).
---

<!-- SKELETON STATUS 2026-08-16: sections 1-4, 6 draftable now from sealed
     results; section 5's headline numbers wait on R0267 (50M) terminal and
     the owner's 100M decision. Every number that lands in this file must
     trace to a sealed result (resultcheck) or carry a [sandbox] label. -->

# Abstract

<!-- Write LAST. One paragraph: the basemap concept; recipe stable 2M->100M
     on one consumer GPU; calibrated instruments + pre-registered gates;
     static-file distribution + in-browser projection; models and demo
     released. -->

# 1. Introduction

A UMAP projection is usually a byproduct: embed a dataset, run the
optimizer, look at the picture, throw the coordinates away. The frame the
optimizer found — which regions ended up where, what is adjacent to what —
dies with the run. Embed a new document and there is no way to place it on
yesterday's picture; re-run the optimizer and the picture itself changes.
For a single analysis this is fine. For a map it is disqualifying: maps are
useful precisely because they stay put while the world moves across them.

This paper treats the frame as the product. A *basemap* is a parametric
projection head [@sainburg2021parametric] trained once over a large, mixed
corpus of embeddings, then held fixed. After training, any text the encoder
can embed lands on the same frame through a single forward pass of an
11.8M-parameter MLP — no neighbor graph, no optimization, no access to the
training data. The cost structure is sharply asymmetric: training the frame
takes hours on one consumer GPU, but projection takes milliseconds and runs
anywhere, including in a web browser alongside a quantized copy of the
encoder itself. Under that asymmetry it becomes rational to spend heavily
on one good frame, because the spend amortizes over every future
projection. The frame becomes learnable geography: a place users revisit,
link into, and navigate by memory, the way they navigate a city map rather
than re-derive it.

Two things have to hold for this to work. The frame must be *good* — a
faithful, legible layout of the encoder's semantic range — and it must be
good *at scale*, because a general-purpose frame has to be trained on
enough of that range to have geography for whatever arrives later. Both
requirements push in the same direction: large training sets. Ours grow
from 2M to 100M embedded text chunks, and the central empirical question of
the paper is whether map quality survives that growth. It is not obvious
that it should. The failure modes of neighbor embeddings at scale are
quiet: maps collapse gradually toward their centers, or fill with a fog of
misplaced points, while still looking plausible in a thumbnail. Much of our
method is therefore not the recipe itself but the instrumentation that
makes these failures visible, measurable, and gateable before a
hundred-GPU-hour run is trusted.

We make three contributions. First, a **training recipe** whose map quality
holds flat from 2M to 100M rows on a single RTX 5090: a UMAP output kernel
fit at zero minimum distance, training budget expressed in graph-relative
units (draws per edge), a fog-targeted reweighting of negative examples
that clears low-density haze without inducing collapse, uniform edge
sampling, and an int8 host-residency scheme that carries training past the
GPU's memory ceiling with map-level fidelity we verify rather than assume.
Second, **instruments and protocol**: cheap, calibrated statistics for
collapse and fog with failure modes planted to prove they can fail,
pre-registered gates fit per treatment family, and a rounds discipline —
registration before numbers exist, sealed receipts, adversarial review —
that we argue is what makes a hundred-GPU-hour claim trustworthy at all.
Third, a **distribution story**: a static-file map format servable from any
host that supports range requests, a viewer with corpus-level filtering
down to individual points and their source text, and in-browser projection
(~80 MB total) that turns any typed phrase into coordinates on the shared
frame with no server round-trip.

# 2. Related work

**Neighbor embeddings.** The lineage runs from t-SNE
[@vandermaaten2008tsne] through LargeVis [@tang2016largevis] to UMAP
[@mcinnes2018umap]: preserve a k-nearest-neighbor structure from the
high-dimensional space while a repulsive term keeps the layout from
collapsing. All three produce non-parametric embeddings — coordinates for
the training points only. Parametric UMAP [@sainburg2021parametric]
replaces the free embedding with a neural network trained on the same
objective, which is what makes a reusable frame possible at all: the
network is the map. Our architecture and objective are parametric UMAP's;
our departures are the scale, the output-kernel and negative-sampling
treatment, and the instrumentation. We also inherit a known sensitivity:
global structure in these methods is determined largely by initialization
[@kobak2021initialization], which for a frame meant to stay put is a
feature to manage deliberately rather than an artifact to ignore.

**Attraction, repulsion, and what the optimizer actually does.** Damrich
and Hamprecht [@damrich2021umaploss] showed that UMAP's implementation
optimizes a different loss than its theory describes, with negative
sampling as the decisive term — a finding our fog-targeted negative
reweighting (§3.2) builds on directly: we modify *which* negatives matter,
at distances where misplaced points accumulate, rather than how many are
drawn. The broader family of force modifications is well charted: heavier-
tailed output kernels sharpen cluster separation [@kobak2019heavytailed],
PaCMAP's taxonomy of near, mid-near, and far pairs isolates which force
ranges shape local versus global structure [@wang2021pacmap], TriMap
frames the problem through ordinal triplets [@amid2019trimap], and densMAP
adds a density-preservation term [@narayan2021densmap]. We evaluate
mid-near and density terms as ablations against our fneg mechanism and
find neither clears fog without cost at our scales. The negative-sampling
device itself descends from word2vec [@mikolov2013distributed].

**Scale.** GPU implementations brought UMAP from hours to minutes at
millions of points [@nolet2021gpuumap]; recent out-of-core work extends
exact-recall graph construction and layout past device memory to the
hundred-million regime [@outofcore2025umap]. We use the out-of-core
NN-descent approach for graph building at 50M+ rows, and full-substrate
cuML maps as per-rung non-parametric references up to the memory ceiling —
the benchmark our parametric maps must match on instruments while adding
the projection property cuML cannot: placing points it has never seen.
Approximate-neighbor systems [@johnson2019faiss] and scalable t-SNE
[@policar2019opentsne] solve adjacent problems in the same regime.

**Embedding-map interfaces.** Nomic Atlas [@atlas] demonstrated
web-served maps over large embedding collections; latent-scope
[@latentscope] couples projection, clustering, and labeling into a local
exploration workflow. Both render a map *of a dataset*. Our emphasis is
the complement: a frame published independently of any dataset, onto which
anyone's data — or a single typed sentence — can be projected client-side.
The distribution format (§6) borrows the cartographic pattern of tiled,
range-requested static files so that hosting a basemap requires no
server-side computation at all.

# 3. Method: the basemap recipe

<!-- [sealed evidence: R0265/R0266 + sandbox program] -->

## 3.1 Setup

- Substrate: 384-dim all-MiniLM-L6-v2 [@wang2020minilm; @reimers2019sbert]
  over 120-token chunks of FineWeb-Edu [@penedo2024fineweb], RedPajama-V2
  [@together2023redpajama], The Pile [@gao2020pile]; nested-prefix ladder
  2M→100M (one physical file carries every rung).
- Graph: exact-k15 fuzzy graph via out-of-core cuVS NN-descent
  [@outofcore2025umap], recall@15 receipts vs brute-force truth per rung
  (e.g. 0.9975 strict at 50M).
- Head: ResidualBottleneckMLP 384→2048→2, 11.8M params.

## 3.2 The four-field treatment

- Output kernel: umap (a,b) fit at min_dist 0 — and why the legacy
  low-power kernel collapses at scale. <!-- sandbox kernel program -->
- Dose: draws/edge as the budget unit; the ×2/×4 choice.
- fneg: BCE up-weighting of negatives at [0.1,0.4]×p90 map radius —
  the fog mechanism, gradient-equivalence to oversampling, telemetry
  (band-hit fraction, loss share). The only repulsion-class lever among our
  ablations (mid-near [@wang2021pacmap], density [@narayan2021densmap])
  that clears fog without collapse. <!-- FIG: fneg ablation before/after -->
- Uniform edge sampling — and the incident that made it a registered field
  (the seed-42 cross-check that caught fuzzy-weight sampling; §4.3).

## 3.3 Host-int8 residency (past the fp32 VRAM ceiling)

- int8 rows + fp16 per-row scales; bit-exact encoder; per-batch dequant on
  device; pre-sealed substrate loads (train-time encode does not survive
  50M). Map-level fidelity: int8 vs fp32 seed-paired sibling Δcollapse
  0.0011 on a 0.247 seed range. <!-- sealed: R0266 -->

# 4. Instruments and protocol

<!-- [sealed: R0263/R0264/R0265] — a methods contribution in its own right -->

## 4.1 The instruments

- Collapse r10·√N: packing-law derivation, the failure mode invisible in
  thumbnails; per-family calibration law (floors are fit per treatment
  family, never inherited across). <!-- FIG: planted-failure calibration -->
- Fog: low-density mass; fail-closed on degeneracy (a hazy map that reads
  fog 0.0 is "not measurable", never a pass).
- Purity (k256 two-sided / k1024 one-sided), held-out FFR.

## 4.2 The rounds protocol

- Pre-registration before numbers exist; sealed receipts; adversarial
  review; non-vacuous gates (every gate must be failable by construction);
  robust floors (median − k·MAD_n, 95/95 multipliers).

## 4.3 Honest incidents as method

- The sampler catch (0.84 GPU-h), effects-not-absolutes, salvage-with-
  provenance. Position: the incident record is evidence the gates work.

# 5. Experiments: the scale ladder

<!-- [sealed through 25M + P1 analysis; 50M = R0267; 100M = pending owner] -->

## 5.1 Does map quality survive scale?

- The drift question; saturating-relaxation analysis (joint fit, shared λ,
  residual bootstrap); asymptote bands ≥0.9 as the pre-registered go
  criterion. <!-- FIG: drift curves + bands; TODO numbers after R0267 -->

## 5.2 The gate family

- 13 seeds at 2M: floors/bands table; all-pass with every gate failable.

## 5.3 Reference frames

- cuML full-substrate references per rung up to the VRAM ceiling
  [@nolet2021gpuumap]; same-frame Procrustes comparison [@schonemann1966].
  <!-- FIG: aligned ladder, parametric vs cuML -->

## 5.4 Representativeness

- MiniLM's training mix (model card) vs our corpus; MS MARCO
  [@nguyen2016msmarco] as in-model/out-of-corpus probe; Reddit
  [@volske2017tldr] and GOOAQ [@khashabi2021gooaq] coverage cells.

## 5.5 Cost

- GPU-h ladder table, measured; the whole program on one consumer GPU.

# 6. Distribution: maps anyone can serve

<!-- [built: PLAN6] -->

- Map packs: static files + HTTP range requests; density tiers, Morton-
  sorted point tiles (8B/point), LOD, text sidecars (100% reachback,
  cosine-1.0000 alignment verification).
- Viewer: layered rendering (WebGL base + canvas overlay), corpus
  filter/color at bin and point level.
- In-browser projection: quantized MiniLM (transformers.js) + ONNX head —
  ~80 MB, no server; text → coordinates on the shared frame in the browser.
  <!-- FIG: viewer + projection panel screenshot -->

# 7. Limitations and future work

- 2D legibility vs 3D volume (the 3D track; teleport navigation).
- Frame evolution across encoder versions; Procrustes-aligned migrations.
- Single-encoder scope (MiniLM-L6-v2); corpus gaps from §5.4.

# 8. Reproducibility

- One consumer GPU; sealed configs; rounds ledger; code + models released.

# References

<!-- references.bib; keep lit-index pointers in comments there -->
