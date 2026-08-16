---
title: "Basemaps: Stable Parametric Projections of Embedding Space at 100M Scale on One GPU"
# TODO(owner): title decision — alternate: "A World Map for an Embedding Space"
# TODO(owner): author list, affiliations, contact
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

<!-- [draft now] -->

- The frame/data separation: per-dataset UMAP runs produce throwaway frames;
  a basemap trains the frame ONCE over a large mixed corpus, then any text
  projects onto it in milliseconds, forever. The map becomes learnable
  geography: stable landmarks, linkable places.
- The cost asymmetry that makes this work: training is hours on one GPU, but
  projection is a single forward pass of an 11.8M-param MLP — no graph, no
  optimization, browser-feasible [@sainburg2021parametric].
- Why scale (100M rows) matters for a general-purpose frame: coverage of the
  encoder's semantic range; the representativeness question (§5.4).
- Contributions: (1) a training recipe whose map quality holds flat 2M→100M
  on one RTX 5090, incl. the fneg mechanism and host-int8 residency; (2)
  calibrated, failable instruments + a pre-registration protocol for map
  quality; (3) a static-file distribution format + in-browser projection.

# 2. Related work

<!-- [draft now] -->

- Neighbor embeddings: t-SNE [@vandermaaten2008tsne], LargeVis
  [@tang2016largevis], UMAP [@mcinnes2018umap], parametric UMAP
  [@sainburg2021parametric]; initialization & global structure
  [@kobak2021initialization].
- Attraction/repulsion theory: UMAP's true loss [@damrich2021umaploss] —
  closest kin to our fneg analysis; heavy-tailed kernels
  [@kobak2019heavytailed]; PaCMAP's forces taxonomy [@wang2021pacmap];
  TriMap [@amid2019trimap]; densMAP [@narayan2021densmap]; SGNS negative
  sampling lineage [@mikolov2013distributed].
- GPU/scale systems: cuML UMAP [@nolet2021gpuumap]; out-of-core GPU UMAP
  [@outofcore2025umap]; FAISS [@johnson2019faiss]; openTSNE
  [@policar2019opentsne].
- Large-map visualization: Nomic Atlas [@atlas], latent-scope
  [@latentscope]. Position: those systems visualize a dataset; we publish a
  reusable frame.

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
