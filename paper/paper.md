---
title: "Latent Basemap: A Reusable Parametric UMAP Frame Trained on 100 Million Text Embeddings"
author:
  - Ian Johnson
date: "Draft for collaborator review, August 2026"
bibliography: references.bib
link-citations: true
geometry: margin=1in
---

<!--
This manuscript is written from the expected post-R0268 state. Values that can
only come from the sealed 100M result use {{100M_*}} tokens. See
RESULTS_CHECKLIST.md before sharing or rendering a release candidate.
-->

# Abstract

Dimensionality-reduction maps usually bind coordinates to one dataset and one
optimization run. We study a reusable alternative: train a parametric projection
on a large reference corpus, freeze it, and use the resulting two-dimensional
coordinate frame for later data. Our reference frame maps 384-dimensional
`all-MiniLM-L6-v2` embeddings through an 11.8-million-parameter residual MLP. We
train it on a composition-controlled ladder ending at 100 million text chunks.
The objective uses a UMAP low-dimensional kernel with binary positive targets,
uniform positive-edge sampling, uniform random negatives, and extra weight on
negative pairs that currently occupy a middle-distance band. Per-row int8 host
storage keeps the input matrix off the GPU without materially changing the map.
We evaluate scale with separate measures of map contraction, low-density haze,
and held-out neighborhood placement. Thresholds are fitted on a 13-seed family
before the 50M and 100M runs. At 50M, all three seeds pass: mean normalized
10-neighbor spacing is 1.0140, fog ranges from 0.1165 to 0.2472, and held-out
fixed-fraction recall ranges from 0.5495 to 0.5594. At 100M, the three-seed
family also passes, with mean spacing {{100M_COLLAPSE_MEAN}}, fog
{{100M_FOG_RANGE}}, and held-out fixed-fraction recall {{100M_FFR_RANGE}}.
The frozen head exports to ONNX and can run with the encoder in a browser. These
results establish one fixed encoder-head pair as a practical reference frame;
they do not establish invariance across retraining, encoder changes, or
out-of-distribution text.

# 1. Introduction

A conventional UMAP run produces coordinates for one finite dataset
[@mcinnes2018umap]. Adding data later requires either an out-of-sample procedure
or a new optimization. A new optimization can move old points, which makes saved
views, annotations, and spatial memory hard to reuse. Parametric UMAP replaces
the free coordinates with a learned function and can place new observations
without optimizing the full layout again [@sainburg2021parametric]. This paper
asks how far that operating model can be pushed for text embeddings.

We call a frozen encoder and projection head a *latent basemap*. For a versioned
text encoder $E$ and a trained head $f_\theta$, the coordinate of text $x$ is

$$
z(x) = f_\theta(E(x)) \in \mathbb{R}^2.
$$

Once both models are frozen, the same embedding produces the same coordinate.
The frame can support long-lived overlays, links, and annotations. It also has a
useful cost profile: graph construction and training happen once, while each new
point requires only encoder inference and one MLP forward pass.

This definition of stability is narrow. It applies to a fixed encoder, head,
preprocessing contract, and coordinate transform. We do not claim that two
independent training runs align automatically, that an updated encoder preserves
the frame, or that arbitrary vectors can be projected meaningfully. Those are
lifecycle and coverage questions, respectively.

Training a reference frame at 100M rows creates three practical problems. First,
the nearest-neighbor graph must be accurate enough that graph defects do not
become layout features. Second, a parametric neighbor-embedding objective can
produce maps that look plausible at thumbnail scale while point spacing
contracts or diffuse mass accumulates between dense regions. Third, the input
matrix exceeds device memory even when the MLP and edge sampler fit comfortably.
Our work addresses these problems together.

The paper makes four contributions:

1. We specify a UMAP-derived training recipe for a fixed text-embedding frame.
   The recipe differs from stock Parametric UMAP in ways that matter for its
   interpretation: positive edges are sampled uniformly, positive targets are
   binary, fuzzy membership weights do not enter the loss, and negative examples
   in a measured distance band receive twice the usual loss weight.
2. We train the same recipe on a composition-controlled ladder from 2M to 100M
   rows on one 32 GB RTX 5090. A symmetric per-row int8 representation stores the
   100M by 384 input matrix in 38.6 GB of host memory and dequantizes only sampled
   rows on the GPU.
3. We define and calibrate three inexpensive checks for failure modes relevant to
   a reusable map: normalized local spacing, low-density mass, and placement of
   held-out queries near their high-dimensional neighbors. A 13-seed family sets
   the thresholds before the 50M and 100M evaluations.
4. We implement a static map package and browser viewer. The package combines
   multiresolution density, point records, source-text lookup, and an ONNX copy of
   the trained head. Hosting requires static files and byte-range requests, with
   chunked files as a fallback.

The central empirical result is limited but useful: the measured map failures do
not worsen between 2M and 100M under this recipe, and held-out placement improves
at 50M. The result supports a fixed reference frame as a deployable artifact. It
does not show that the layout is optimal or that our custom measurements cover
every distortion a user might care about.

# 2. Related work

## 2.1 Parametric neighbor embeddings

t-SNE, LargeVis, and UMAP construct low-dimensional layouts from local
high-dimensional relationships [@vandermaaten2008tsne; @tang2016largevis;
@mcinnes2018umap]. Their standard forms optimize coordinates attached to the
training observations. Parametric UMAP instead learns a neural mapping and
supports direct inference on unseen observations [@sainburg2021parametric].
Neural Network Projections learn a similar train-once mapping from examples of a
non-parametric projection [@espadoto2020nnp]. Recent work finds that parametric
neighbor embeddings can lose local detail because their negative pairs receive
insufficient repulsion. ParamRepulsor addresses that problem with hard-negative
mining and a stronger repulsive loss [@huang2024paramrepulsor].

Our method belongs to this parametric family, but its training loss is not an
implementation of the full UMAP cross-entropy over fuzzy memberships. The graph
defines which pairs count as positive; the optimizer then uses binary targets and
uniform edge sampling. We therefore describe the objective as *UMAP-derived* and
state its departures explicitly in Section 3.3.

Out-of-sample placement also has failure modes of its own. Islam and Fleischer
measure peripheral drift in UMAP transforms and report that parameterization can
reduce it [@islam2026oosumap]. Our held-out evaluation tests whether projected
queries land within a coarse neighborhood of their true corpus neighbors. It
does not test every form of peripheral drift.

## 2.2 Attraction, repulsion, and negative selection

The sampled UMAP implementation optimizes a loss that differs from the nominal
fuzzy-set cross-entropy, with negative sampling playing a central role
[@damrich2021umaploss]. The attraction-repulsion spectrum provides a broader
account of how these forces change local and global structure [@bohm2022attraction].
Related methods change the output kernel [@kobak2019heavytailed], introduce
mid-near pairs [@wang2021pacmap], use triplet comparisons [@amid2019trimap], or
add density preservation [@narayan2021densmap]. Contrastive formulations connect
t-SNE and UMAP through their sampling and loss choices [@damrich2023contrastive].

Distance-dependent negative selection is established in contrastive learning
[@robinson2021hard], and ParamRepulsor is the closest dimensionality-reduction
precedent for our negative treatment [@huang2024paramrepulsor]. Our `fneg`
treatment is a simpler, band-limited reweighting aimed at an observed haze
artifact. A negative pair receives extra weight only when its current 2D distance
falls between 0.1 and 0.4 times a batch radius. We do not claim hard-negative
mining as a new idea. The experiment asks whether this restricted treatment
removes the measured haze without driving the map toward collapse.

## 2.3 Reference frames and large maps

Fixed basemaps have precedents outside text embedding. Scientometric overlay maps
place new publication sets on a shared map of science [@rafols2010overlay].
Single-cell reference systems map query cells into a frozen atlas
[@kang2021symphony]. OpenAlex Mapper applies the same pattern to scholarly text at
roughly 250,000 reference documents [@noichl2024openalexmapper]. These systems
establish the reference-frame concept. Our work tests one implementation at a
larger training scale with local browser inference.

Large static embedding maps provide a second precedent. A public biomedical
t-SNE atlas contains about 21 million abstracts [@gonzalezmarquez2024landscape].
Hackerverse publishes a 40M-point tiled map with semantic navigation
[@hackerverse2024]. NOMAD Projection maps more than 100M multilingual Wikipedia
documents using a non-parametric multi-GPU method [@duderstadt2025nomad]. Recent
out-of-core UMAP work scales graph construction and non-parametric layout beyond
100M vectors [@outofcore2026umap]. Our 100M result is different in two respects:
the trained artifact is a projection function, and the training run uses one
consumer GPU after graph construction.

GPU UMAP [@nolet2021gpuumap], FAISS [@johnson2019faiss], and openTSNE
[@policar2024opentsne] address complementary parts of the scale problem. We use
cuVS-style clustered candidate search and local NN-descent for graph construction,
then exact fp32 reranking of candidates. The resulting graph is approximate and
is qualified by sampled exact recall. We do not call it an exact graph.

## 2.4 Evaluation and delivery

The reliability literature warns against reading a low-dimensional plot as a
literal account of the source space. scDEED uses null comparisons to identify
dubious local patterns [@xia2024scdeed]. LOO-map studies continuity under data
perturbation [@liu2024loomap], and the Gap Index measures distortion in empty
regions [@ros2026gapindex]. A recent field review collects broader evaluation and
reporting recommendations [@debodt2025review]. Our collapse and fog statistics
target narrower failures observed in this project. They are operational gates,
not replacements for trustworthiness, continuity, or task-specific validation.

For delivery, Nomic Atlas and latent-scope provide interfaces for large embedding
collections [@atlas; @latentscope]. WizMap and Embedding Atlas show scalable web
interaction over embedding data [@wang2023wizmap; @ren2025embeddingatlas]. Our map
package follows the static, byte-range-addressed pattern used by PMTiles
[@pmtiles]. Its distinguishing feature is that the browser can also run the same
encoder-head pair used to define the frame.

# 3. Method

## 3.1 Frame contract

The frame is defined by five versioned components:

1. text preprocessing and 120-token chunking;
2. the `sentence-transformers/all-MiniLM-L6-v2` encoder;
3. L2 normalization of its 384-dimensional output;
4. the learned projection head;
5. the affine transform from model coordinates to published map coordinates.

Changing any component creates a new frame version. The head accepts any vector
that follows this embedding contract, but useful placement also depends on
coverage by the encoder and training corpus. The viewer should eventually expose
an out-of-distribution score; the current prototype does not.

## 3.2 Corpus and graph

The 100M training substrate contains 40M FineWeb-Edu chunks, 25M RedPajama-V2
chunks, 25M chunks from The Pile, and 10M code chunks from `bigcode/starcoderdata`
[@penedo2024fineweb; @weber2024redpajama; @gao2020pile; @li2023starcoder]. Each
production rung from 6.25M through 100M preserves the 40/25/25/10 proportions
and nests within the next rung. The 2M calibration family uses a separate,
composition-matched draw. Rows are sampled across the available shards rather
than taken as leading prefixes. Zero vectors are rejected and replaced. A
separate 200,000-row reserve, 50,000 rows per corpus, is disjoint from training.
The held-out placement panel uses 500 query rows per corpus from that reserve.

For each rung we build a directed cosine $k$-nearest-neighbor graph with $k=15$.
At large rungs, vectors are assigned to multiple balanced clusters, local
NN-descent graphs are built per cluster, candidates are merged, and final
candidate distances are recomputed in fp32. We then form a fuzzy symmetric union
using the UMAP graph construction. This procedure is approximate because cluster
assignment and local search can omit candidates.

We qualify the approximate graph against brute-force fp32 neighbors on a fixed
uniform probe. At 100M, the probe contains 500,000 rows. Tie-aware recall@15 is
0.99794 and strict recall@15 is 0.99590. The directed graph has no invalid,
self-loop, duplicate-within-row, or zero-out-degree entries; its fuzzy symmetric
union has no isolated rows. The fuzzy graph contains 2,511,103,254 directed edge
records after symmetrization.

## 3.3 Projection head and loss

The projection head is a residual bottleneck MLP. It maps 384 inputs to a
2048-unit layer, reduces to a 1536-unit neck, applies two residual 1536-unit
blocks, expands to 2048 units, and outputs two coordinates. Linear layers use
ReLU activations except at the output. The model has 11,809,282 parameters and no
batch normalization or dropout.

For a pair $(i,j)$ with projected distance $r_{ij}$, the low-dimensional
similarity is the UMAP kernel

$$
q_{ij} = \frac{1}{1 + a(r_{ij}^2)^b},
$$

with $a=1.9328$ and $b=0.7905$, the curve fitted for `min_dist=0`. Each update
contains 8,192 pairs: 409 positive graph edges and 7,783 random non-self pairs.
Positive edges are drawn uniformly with replacement from the fuzzy symmetric
edge list and receive target 1. Random pairs receive target 0. Fuzzy membership
weights affect graph topology but do not weight the training loss.

The base pair loss is binary cross-entropy. For negative pair $p$, let $d_p$ be
its current 2D distance. Let $R$ be the 90th percentile of endpoint radii around
the endpoint centroid in that update. The `fneg` weight is

$$
w_p =
\begin{cases}
2, & y_p=0 \text{ and } 0.1R \le d_p \le 0.4R,\\
1, & \text{otherwise}.
\end{cases}
$$

The update loss is $\sum_p w_p\,\mathrm{BCE}(q_p,y_p)/\sum_p w_p$. This doubles
the contribution of negatives already in the measured middle-distance haze band.
The band moves with the current map scale because $R$ is recomputed each update.

We optimize with AdamW at learning rate $10^{-3}$, 200 successful-update warmup,
a cosine schedule, bf16 autocast, and gradient-norm clipping at 1.0. The
correlation term in the package implementation is disabled. These choices and
the binary, uniformly sampled objective are why we avoid describing the loss as
stock Parametric UMAP.

## 3.4 Training dose

We express the training horizon relative to a registered base update budget that
scales with the active directed-edge count. A dose label such as $\times2$ means
twice that base update budget; it does not mean two literal visits to each edge.
At 50M, $\times2$ is 4,162,228 successful updates. At 100M, it is 8,327,508
successful updates. With 409 positive pairs per update, both correspond to about
1.356 positive draws per directed edge. The 2M calibration family uses
$\times4$, about 2.713 draws per edge. Receipts count successful optimizer
updates, excluding skipped or non-finite attempts.

Four fields define the promoted treatment: the UMAP output kernel, the dose, the
middle-distance negative weight, and uniform positive-edge sampling. We freeze
the remaining architecture and optimizer fields across the scale ladder.

## 3.5 Host int8 residency

Above about 20M rows, even fp16 input residency competes with the graph and
training state for 32 GB of VRAM. We encode each normalized input row $x$ with a
stored fp16 scale $s=\max |x|/127$ and signed int8 values
$\operatorname{clip}(\operatorname{round}(x/s),-127,127)$. Sampled rows are
gathered in host memory and dequantized on the GPU for each batch. The 100M input
uses 38.4 GB for int8 values and 0.2 GB for scales.

This is a lossy representation. We test its effect on the final map rather than
calling it lossless. On a seed-paired 2M run, int8 changes normalized spacing by
0.0011 compared with a 0.247 range across the 13-seed fp32 family. The int8 map
also remains inside all five calibrated family bands (Section 5.2).

# 4. Evaluation

## 4.1 Experimental design

We separate exploratory selection from confirmatory scale evaluation. Sandbox
experiments compare output kernels, doses, `fneg`, mid-near pairs, and a
density-matching term. Those runs select one treatment but do not set its final
thresholds. We then train that treatment at 2M for 13 seeds (42 through 54).
This family defines the metric bands. The 50M and 100M evaluations each use three
seeds (42, 43, and 44) and apply thresholds fixed before training.

The 100M evaluation was preregistered after the 50M result. A run passes when the
three-seed mean spacing lies in its scale band and every seed clears the spacing
backstop, fog ceiling, and held-out placement floor. Purity is descriptive at
50M and 100M because those substrates do not share the exact 2M reference rows
used to fit the purity bands. We do not apply a threshold calibrated on one row
identity to another.

Each accepted cell binds the input substrate, graph, config, code release, seed,
model, coordinates, and evaluation output by digest. At 50M, seed 42 completed
its full training horizon and saved its model and coordinates, but a post-training
memory assertion stopped the process before it wrote the standard train receipt.
That cell is retained as log-derived evidence; seeds 43 and 44 have complete
receipts. The rounds machinery also records failed setup attempts and corrected
instruments. These records are useful for audit, but they are not a scientific
contribution by themselves.

## 4.2 Metrics

**Normalized 10-neighbor spacing (collapse).** We build the neighbor index on all
points up to 16,777,216 rows. Larger maps use a fixed seeded subsample of that
size. We then sample 20,000 query points with the same fixed seed, compute each
query's distance to its tenth 2D neighbor, and take the median $r_{10}$. Let $R$
be the 90th percentile radius around the indexed map centroid. For indexed
population $N_{\mathrm{eff}}$, the statistic is

$$
C = \frac{r_{10}}{R}\sqrt{N_{\mathrm{eff}}}.
$$

The square-root factor removes the first-order packing effect in two dimensions.
Low values indicate contraction into dense beads. At 50M and 100M, this is a
fixed-size Monte Carlo estimate rather than a full-population nearest-neighbor
calculation. The statistic is invariant to translation and isotropic scaling,
but it can still depend on sampling, duplicates, and map shape.

**Low-density mass (fog).** We estimate the 0.1st to 99.9th percentile extent
from at most two million deterministically selected rows, add 2% padding, and
compute a 1024 by 1024 histogram of all points. Off-extent coordinates are
clipped into edge bins. Let $m$ be the peak bin count. Fog is the fraction of all
binned points in occupied bins with counts below $0.01m$. If $m<100$, the
threshold has no integer resolution and the measurement is marked degenerate. A
degenerate fog result cannot pass.

**Held-out fixed-fraction recall (FFR).** For each reserve query, we find its true
top 10 cosine neighbors in the full high-dimensional training substrate. We
project the query through the frozen head and retrieve the closest 0.1% of the
map by 2D distance. FFR is the mean fraction of the ten true neighbors contained
in that discovery set. The discovery set contains 2,000 points at 2M, 50,000 at
50M, and 100,000 at 100M. FFR measures coarse neighborhood placement. It is much
more permissive than recall@10 and should not be reported as recall@10.

**Purity fidelity.** We assign source-space labels from fixed $k$-centroid
partitions at $k=256$ and $k=1024$. For sampled anchors, we compare label
agreement in a 0.1%-of-$N$ map neighborhood with agreement in the corresponding
high-dimensional neighborhood. The reported ratio is map agreement divided by
high-dimensional agreement. A ratio above one means the map over-separates those
labels. The $k=256$ criterion is therefore two-sided; the $k=1024$ criterion uses
a lower floor. At large rungs we report these ratios without a verdict because
the frozen reference rows changed.

## 4.3 Calibration and decision rules

For each one-sided metric, the family threshold has the form
$\operatorname{median} \pm k\,\mathrm{MAD}_n$, where
$\mathrm{MAD}_n=1.4826\operatorname{median}|x_i-\operatorname{median}(x)|$.
The multiplier is calibrated by simulation for a 95/95 tolerance target under a
Gaussian null. At $n=13$, the one-sided multiplier is 3.7364 and the two-sided
multiplier is 4.4524. Planted bad inputs verify that each rule can fail.

The scale-level spacing rule adds a second check. A joint saturating-relaxation
fit to the 2M, 6.25M, 12.5M, and 25M spacing results estimates an asymptotic band
for the $\times2$ treatment. We widen that band by
$1.96\,\sigma_{\mathrm{family}}/\sqrt{3}$ for a three-seed mean. The resulting
50M and 100M band is [0.8650, 1.0505], with a per-seed backstop of 0.8129. The
fit uses only seven total scale-dose points and two residual degrees of freedom,
so we treat it as a guard against drift, not a precise scaling law.

| measure | rule used at 50M and 100M | source |
|---|---:|---|
| mean normalized spacing | 0.8650 to 1.0505 | widened scale fit |
| per-seed normalized spacing | at least 0.8129 | 13-seed family |
| fog | at most 0.41207 | 13-seed family |
| held-out FFR | at least 0.39906 | 13-seed family |
| purity fidelity | descriptive | reference lineage differs |

## 4.4 Comparisons

We compute full-substrate cuML UMAP references where the non-parametric layout
fits the available hardware. These references show a tradeoff: cuML leaves very
little low-density mass, while the parametric head gives higher in-sample
coarse-neighborhood placement on the available comparison. cuML has no learned
transform in this setup, so it cannot enter the held-out projection comparison.
We do not require the two methods to match on every custom metric, since one
optimizes free coordinates and the other fits a reusable function.

The current study lacks a scale-matched stock Parametric UMAP, ParamRepulsor, or
NOMAD run under a shared benchmark. We return to this gap in Section 7.

# 5. Results

## 5.1 Selecting the negative treatment

The exploratory ablation isolates a recurring haze between dense regions. At 2M
and dose $\times4$, adding the middle-distance negative weight reduces fog from
0.4587 to 0.3706. Normalized spacing changes from 1.1834 to 1.0620 and remains
well above the later family floor. The treated map's held-out FFR is 0.4111.

At 6.25M and dose $\times2$, the same treatment reduces fog from 0.4509 to
0.2860, raises the exploratory in-substrate FFR from 0.3924 to 0.4051, and gives
normalized spacing 0.9346 instead of 1.0551. Mid-near and density-matching arms
did not improve the joint fog and FFR result in this sandbox. Because those arms
were not repeated as a full seed family, we use them only to explain treatment
selection. A direct ParamRepulsor comparison remains necessary.

The treatment effect is not a monotone improvement on every visual statistic.
Stronger repulsion removes diffuse mass but also changes overall spacing. This is
why the promoted decision requires both an upper fog limit and a lower spacing
limit.

## 5.2 The 2M gate family

All 13 seeds of the promoted 2M treatment pass the thresholds fitted from that
family. The table reports the observed family distribution and the registered
rule.

| measure | median | MADn | observed range | registered rule |
|---|---:|---:|---:|---:|
| normalized spacing | 1.0288 | 0.0578 | 0.8825 to 1.1300 | at least 0.8129 |
| fog | 0.3366 | 0.0202 | 0.2653 to 0.3660 | at most 0.4121 |
| held-out FFR | 0.4196 | 0.00549 | 0.4154 to 0.4285 | at least 0.3991 |
| purity fidelity, k=256 | 1.0867 | -- | within 1.0517 to 1.1228 | two-sided band |
| purity fidelity, k=1024 | -- | -- | all above 0.8807 | at least 0.8807 |

The family is a calibration population, not an independent estimate of a false
failure rate in arbitrary data. Its main use is operational: it fixes the scale
of expected seed variation before expensive runs.

## 5.3 Scale behavior

The pre-50M ladder does not show progressive contraction. At dose $\times2$,
normalized spacing is 1.1904 at 2M, 0.9346 at 6.25M, 0.9930 at 12.5M, and 0.9672
at 25M. A saturating-relaxation fit estimates an asymptote of 0.9649 with a
bootstrap band of [0.9303, 0.9851]. The $\times4$ series has three points and an
estimated asymptote of 0.9907 with band [0.9608, 1.0159]. Both estimates exceed
the preregistered go floor of 0.9. The bootstrap is fragile because the series is
short; the later seed families are more informative than the fitted rate
parameter.

At 50M, the $\times2$ host-int8 treatment passes every gated criterion.

| seed | normalized spacing | fog | held-out FFR |
|---:|---:|---:|---:|
| 42 | 1.0860 | 0.2472 | 0.5594 |
| 43 | 1.0326 | 0.1165 | 0.5539 |
| 44 | 0.9234 | 0.1631 | 0.5495 |
| decision | mean 1.0140, pass | all pass | all pass |

Seed 42 is above the narrow fitted asymptote band, but the three-seed mean lies
inside the preregistered seed-widened band and every seed clears the family
backstop. Held-out FFR is about 0.55, compared with a family median near 0.42 at
2M. Since the low-dimensional discovery set remains fixed at 0.1% of the corpus,
this is evidence of better coarse placement rather than an artifact of holding a
fixed number of discovered points.

At 100M, the three-seed family passes the same preregistered decision rule. Mean
normalized spacing is {{100M_COLLAPSE_MEAN}}, with per-seed values
{{100M_COLLAPSE_VALUES}}. Fog ranges {{100M_FOG_RANGE}}, and held-out FFR ranges
{{100M_FFR_RANGE}}. All spacing values clear 0.8129, all fog values remain below
0.41207, and all FFR values remain above 0.39906. Purity is
{{100M_PURITY_SUMMARY}} and remains descriptive because its reference row
identity differs from the 2M calibration family.

<!-- FIGURE 1: aligned density renders for 2M, 12.5M, 50M, and 100M. -->
<!-- FIGURE 2: spacing and fog across N, with seed points and preregistered bands. -->

## 5.4 Graph quality

The 100M graph qualification measures the topology before training. On the fixed
500,000-row probe, strict recall@15 is 0.99590 and tie-aware recall@15 is 0.99794.
The 100M graph has zero rows with missing outgoing neighbors and zero isolated
rows after fuzzy symmetrization. Twenty-five probe rows have tie-aware recall
zero; 24 belong to duplicate or near-duplicate families, while one is a genuine
candidate-search miss. Reporting that tail matters because a high mean can hide
individual failures.

These results support the graph as a high-recall approximation. They do not make
it exact. The graph builder and the projection head also solve separate problems:
graph recall validates training input, while the map metrics evaluate the learned
two-dimensional frame.

## 5.5 Non-parametric references

At 2M, a cuML reference has normalized spacing 1.718, fog 0.0397, and exploratory
in-sample quick FFR 0.2852. At 6.25M, the corresponding values are 1.908, 0.0399,
and 0.3266. The promoted 6.25M parametric map has more fog (0.2860) but higher
in-sample quick FFR (0.4051). This comparison concerns coordinates assigned to
training rows; it is distinct from the reserve-projection FFR used by the scale
gate. It shows that low background mass and coarse neighborhood containment
encode different preferences.

The instruments in this comparison are not a standard benchmark suite. We avoid
the stronger claim that the parametric map matches or exceeds non-parametric UMAP
in general.

## 5.6 Precision and cost

The seed-paired 2M host-int8 map records normalized spacing 1.1289, fog 0.3370,
held-out FFR 0.4212, k=1024 purity fidelity 0.9304, and k=256 purity fidelity
1.0817. Every value lies inside the corresponding fp32 family band. The spacing
difference from its fp32 sibling is 0.0011; the fog difference is 0.029.

On one RTX 5090, the 13-seed 2M family costs 10.85 GPU-hours in total, or about
50 minutes per seed. The 50M runs take about 11.9 GPU-hours per seed. The 100M
runs take {{100M_TRAIN_HOURS}} GPU-hours per seed and
{{100M_TOTAL_HOURS}} GPU-hours across the three training seeds. These figures
cover model training and exclude prior embedding and graph construction.

# 6. Static distribution and browser projection

The trained head is small relative to the corpus, but a useful map also needs
density, individual points, and source lookup. We package those assets as static
files. A map manifest records the frame extent and file hashes. Density is stored
as a multiresolution pyramid with separate uint32 planes per source corpus. Point
records use two uint16 coordinates and one uint32 identifier, eight bytes per
point, sorted by finest-tile order and then Morton order. A level-of-detail file
provides bounded point samples before deep zoom. Text sidecars map row identifiers
to byte ranges in a UTF-8 blob.

For the 2M and 12.5M prototype packs, the current packer validates that density
counts sum to the corpus size at every level, point records fall in their declared
tiles, identifiers round-trip, and file hashes match. On 256 sampled rows,
re-embedding the sidecar text reproduces the source embedding at cosine 1.0000.
The stored text is the lowercased, detokenized chunk that the encoder saw; it is
not necessarily the verbatim source document.

The viewer uses WebGL2 for density and points, a small canvas overlay for hover
and selections, and DOM controls for filters and details. A range-capable host can
serve point and text slices directly. For hosts that ignore HTTP `Range`, the
publisher can emit fixed-size part files and the viewer splices the requested
bytes locally.

The projection path exports the MLP to a 47.2 MB fp32 ONNX model. For the 2M and
12.5M prototype heads, ONNX and PyTorch differ by less than $10^{-4}$ in any
coordinate on 10,000 random unit vectors. The final 100M head still needs the same
export check. The browser pairs a verified head with MiniLM through
Transformers.js and ONNX Runtime Web. The fp32 browser profile is about 143 MB.
An optional int8 encoder profile is about 78 MB; on five test strings its encoder
cosine relative to fp32 is 0.988 to 0.994 and its map displacement is at most
0.139% of the frame diagonal. We therefore treat the int8 browser encoder as a
size-quality tradeoff, not an exact reproduction.

# 7. Limitations

**One encoder and one corpus mixture.** The result applies to
`all-MiniLM-L6-v2` and the 40/25/25/10 training mix. The mix is dominated by web
and English-language material, with a 10% code allocation. It overweights code
relative to the encoder's reported training mix and underrepresents conversational
and question-answering sources. A planned MS MARCO, Reddit, and GOOAQ coverage
audit is not complete [@nguyen2016msmarco; @volske2017tldr;
@khashabi2021gooaq]. We make no universality claim.

**Fixed-version stability only.** Repeated inference is deterministic for fixed
weights and preprocessing. We have not measured coordinate drift across retrained
heads, new corpus snapshots, or new encoder versions. A production basemap needs
versioned migrations, alignment criteria, and a policy for retiring old frames.

**Custom metrics.** Normalized spacing and fog were built around failures found in
this project. FFR uses a large discovery neighborhood and does not measure
fine-grained recall. Purity depends on a frozen reference and could not be gated
at 50M or 100M. Standard trustworthiness, continuity, neighborhood hit, visual
task studies, and perturbation tests would make the evaluation easier to compare
with other work.

**Missing matched baselines.** We have not run stock Parametric UMAP,
ParamRepulsor, NNP, NOMAD, or the recent out-of-core non-parametric UMAP system on
the same 100M substrate and hardware budget. The cuML references stop at 6.25M.
The current evidence supports within-recipe scale behavior, not a state-of-the-art
ranking.

**Approximate graph.** Sampled recall is high, but at least one probed row has a
genuine complete miss. Graph errors may concentrate in rare or sparse regions
that an aggregate recall estimate underweights.

**Two-dimensional interpretation.** A clear map can still distort distances,
density, and topology. The frame is an interface for exploration, not a metric
space for downstream inference. We have a separate 3D experiment, but it is not
part of the present claim.

**Release constraints.** The viewer and packer work on local artifacts. Public
checkpoints, map packs, source-text policies, licenses, and durable hosting are
release tasks. Until those are complete, the distribution section describes a
tested prototype rather than a public service.

# 8. Reproducibility and evidence

The repository contains the projection implementation, metric code, map packer,
viewer, and manuscript. The final 50M and 100M round modules live on the research
branch `basemap-100m/round-0208`; this publication branch has diverged and should
not be treated as the exact 100M execution checkout. The matching preregistrations,
results, reviews, and artifact digests are stored in the companion `latent-labs`
repository. `REVIEWER_GUIDE.md` gives a short path through those records.

The main evidence anchors are:

| claim | record |
|---|---|
| 2M 13-seed thresholds | R0265 |
| host-int8 map fidelity | R0266 |
| 50M three-seed pass | R0267 |
| 100M preregistration and result | R0268 |
| 100M graph recall and degree checks | R0241 and R0243 |
| static map package | `experiments/mappack/REPORT.md` |
| ONNX and browser parity | `experiments/mappack/onnx/REPORT.md` |

We will publish the selected checkpoint, production config, map manifest, and
artifact hashes with the paper. Dataset redistribution will follow the licenses
and access terms of each source corpus.

# AI-assisted work disclosure

Claude and ChatGPT were used for code generation, experiment orchestration,
literature triage, and manuscript editing. The human author designed the research
program, reviewed generated work and experimental decisions, and accepts
responsibility for the claims and released artifacts. This disclosure should be
revised to match the policy of the submission venue.

# References
