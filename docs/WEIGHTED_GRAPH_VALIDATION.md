# Weighted (fuzzy) neighbor graphs at 30M–405M — build + validation

**Status:** 2026-07-20. Implements `latent-labs/guides/spec-weighted-graph-build.md`
(Path A: real UMAP fuzzy weights on the existing IVF_PQ topology). Tool:
`experiments/build_weighted_graph.py` (+ `experiments/weighted_graph_validate.py`,
tests in `tests/test_weighted_graph.py`). Honest write-up of what was built, the
V1–V4 battery, the 405M / Path-B feasibility memo, and one blocker (the 150M
artifact is blocked by upstream fineweb data corruption — see §150M).

This work does **not** claim weighted training helps at 30M — that is the research
side's weighted-vs-uniform A/B. It produces the artifacts that make it possible.

## Headline

| item | result |
| --- | --- |
| `edges_30m_k15_fuzzy.npz` | **built + validated**, 738,221,242 directed edges, 5.66 GB, weight median 0.186 |
| V1 exactness (100k) | **PASS** — edge sets identical to `fuzzy_simplicial_set`, weight max-abs-diff 1.2e-07, sigma/rho diff 0.0 |
| V2 topology honesty (3M) | recall@15 **0.254–0.258** (nprobe 32/64/128) — IVF_PQ topology is PQ-quantization-limited, NOT nprobe-limited |
| V3 consumer contract (30M) | **PASS** — CDF builds, no constant-weights warning, draws ∝ weight (decile corr 1.0) |
| V4 spot physical (30M) | **PASS** — 20/20 monotone decay, 60/60 mutual-pair symmetrized dominance |
| `edges_150m_k15_fuzzy` | **BLOCKED** — local fineweb-120 shard 37 is truncated; ~14M of 150M nodes are misaligned with the graph topology and unrecoverable locally (Modal wound down). Tool is ready; needs a clean fineweb re-pull/re-embed. |
| 405M / Path-B | measured: distance-recompute is **disk-random-read bound** above RAM; Path-B (better-recall topology) recommended given V2. |

## What the tool does

The ship-scale graphs carry constant `1/k` weights — the trainer's
`weighted_edge_sampling=True` path detects "constant weights → uniform-equivalent"
and gains nothing. This tool rebuilds them with genuine UMAP fuzzy membership
weights on the **same neighbor topology** (schema-identical output → trainer needs
zero changes), on one RTX 5090, streamed and resumable.

| phase | device | what |
| --- | --- | --- |
| A | GPU | Per node's k neighbors (source-sorted, exactly k/node → reshape `(n,k)`), gather both endpoint vectors, recompute **exact cosine distances** on GPU, sort ascending, prepend self-column `(i, 0)`. Emit per-chunk forward directed membership edges (resumable `.done` markers). |
| B | CPU | Partition forward edges by a splitmix64 hash of the canonical pair `{i,j}` into P disk buckets — both `(i,j)` and `(j,i)` land together. |
| C | CPU (parallel) | Per bucket, probabilistic t-conorm `W = W + Wᵀ − W∘Wᵀ` (`set_op_mix_ratio=1.0`), emitting **both** directed edges per pair — exactly `fuzzy_simplicial_set(...).tocoo()`. |
| D | CPU | Assemble single `.npz` (30M, globally `(src,dst)`-sorted) or sharded `part-*.npz`+`index.json` (150M) + provenance manifest. |

### Why the math is correct by construction
sigma/rho come from umap-learn's own `smooth_knn_dist` (v0.5.12), applied per
node-chunk — id-independent, so chunking is exact. Membership is emitted here
(not via umap's `compute_membership_strengths`) because that function keys the
source id and self-mask off the POSITIONAL row index, which only equals the
global id for a chunk starting at node 0; we key off the real global id in the
self-column so any chunk offset is correct. The per-value formula and edge-case
order match umap exactly and are validated against `compute_membership_strengths`
in the tests and against `fuzzy_simplicial_set` end-to-end in V1.

*(This positional-index coupling caused a first 30M build to be corrupt — every
chunk emitted local 0-based source ids and failed self-detection, inflating
hub-node degree and injecting self-loops. Caught by V4, fixed, regression-tested
(`test_membership_matches_umap_full_and_offset`), and rebuilt. V1 alone did not
catch it because it processes rows 0..n−1 where local index == global id.)*

### Neighbor-count convention (log2(16), not log2(15))
The topology stores **k = 15 real neighbors** per node (self excluded; 0.043% of
30M nodes carry a self-loop from the approximate index — handled by umap's
self→0→eliminate_zeros). UMAP's convention is `n_neighbors` columns INCLUDING
self, so faithful reproduction prepends a self-column and uses `n_neighbors =
k+1 = 16`, target `log2(16) = 4.0` — matching how the k=50 reference artifact was
built (`n_neighbors=50` → 49 real neighbors). `--target-neighbors` overrides it.

### Chunk-mean edge case (quantified)
`smooth_knn_dist`'s `mean_distances` is per-call and used only in the `rho==0`
fallback (a node whose every neighbor is at distance 0). The builder counts
`rho==0` rows; at 30M this count is **0**, so chunking is provably identical to a
single call.

## V1 — exactness at 100k

Exact GPU cosine kNN (k=15) on the first 100k rows, full builder pipeline vs
`fuzzy_simplicial_set` on the **identical** precomputed topology.

| metric | value |
| --- | --- |
| exact GPU kNN vs sklearn (mean Jaccard@15) | **1.0000** |
| n_neighbors / target | 16 / 4.0 |
| my edges / oracle edges | 2,130,704 / 2,130,704 |
| edges only-mine / only-oracle | 0 / 0 (**identical**) |
| weight max-abs-diff | **1.19e-07** (≤ 1e-4 ✓) |
| sigma / rho max-abs-diff | **0.0 / 0.0** |
| rho==0 rows | 0 |

**PASS.** (`docs/weighted_graph_validation/v1_100k.json`)

## V2 — topology honesty at 3M

`faiss_ivf_pq_3m.index` (IVFPQ, nlist=1732, PQ M=48×8-bit) vs exact GPU kNN,
50k sampled nodes. Base = round-0010 rows [0:3M] = fineweb[:3M] (= the 3M index's
rows; confirmed by the high edge/random cosine margin).

| nprobe | recall@15 | recall p10 | weight-MAD (shared edges) | median wt exact / approx |
| --- | --- | --- | --- | --- |
| 32 (150M setting) | 0.2536 | 0.067 | 0.295 | 0.178 / 0.175 |
| 64 (30M setting) | 0.2568 | 0.067 | 0.292 | 0.178 / 0.175 |
| 128 | 0.2580 | 0.067 | 0.291 | 0.178 / 0.175 |

**Reading.** Recall is flat (~0.254–0.258) as nprobe quadruples: the topology is
**PQ-quantization-limited, not nprobe-limited**. The uniform *and* fuzzy 30M/150M
graphs sit on a kNN that overlaps exact kNN by only ~1/4. Fuzzy-weight
*distributions* are stable (median 0.178 vs 0.175), but per-shared-edge weights
differ by ~0.29 because a changed neighbor set recalibrates rho/sigma.

**Implication.** Path A faithfully upgrades the *shipped* topology to fuzzy
weights (valid for a same-topology weighted-vs-uniform A/B), but the topology
itself is lossy. Per the spec's `recall < 0.9` trigger, **Path B is warranted**
(see memo). The lever is PQ compression, not nprobe.
(`docs/weighted_graph_validation/v2_3m_nprobe{32,64,128}.json`)

## V3 — consumer contract at 30M

Trainer's `DeviceEdgeSampler(weighted_edge_sampling=True)` on the 30M artifact.

| metric | value |
| --- | --- |
| load time / peak RSS | 17.1 s / 9.5 GB |
| CDF built / endpoint | yes / 0.99999999999 (max backstep 5.6e-16, GPU-cumsum fp noise) |
| "constant weights" warning | **none** |
| weight median / mean / max | 0.186 / 0.283 / 1.0 |
| 10M draws: decile draw-freq vs expected | **corr 1.0, max abs diff 1.4e-4** |

**PASS.** Draws are proportional to fuzzy weight across all weight deciles; the
per-edge `corrcoef` is Poisson-noise-limited at 10M draws over ~1e9 edges and is
intentionally not the gate. (`docs/weighted_graph_validation/v3_30m.json`)

## V4 — spot physical check (30M)

20 random nodes, per-node membership recomputed from the topology vs the shipped
artifact.

| check | result |
| --- | --- |
| highest-weight neighbor == smallest-distance neighbor | 20/20 |
| weights monotonically decay with distance | 20/20 |
| mutual-pair symmetrized weight ≥ either directed membership | 60/60 |

**PASS.** (`docs/weighted_graph_validation/v4_30m.json`)

## Artifacts

- `gsv:/data/checkpoints/pumap/edges_30m_k15_fuzzy.npz` — 738,221,242 directed
  edges (24.6/node), 5.66 GB, keys `sources`/`targets` (int32),
  `weights` (float32, median 0.186, p10 0.050, p90 0.744, max 1.0),
  `n_nodes=30000000`, `k=15`, `nprobe=64`. Globally `(src,dst)`-sorted.
  Loads unchanged via the trainer's `load_edge_arrays`.
- `gsv:/data/checkpoints/pumap/edges_30m_k15_fuzzy.npz.manifest.json` — input
  edge-npz sha256, ordered shard list, params (k, metric, n_neighbors=16,
  target=4.0, tol 1e-5, nprobe inherited 64), per-array sha256, weight summary,
  build resources (wall 502 s, peak RSS 28.3 GB, peak VRAM 7.15 GB), commit.
- Build: `uv run python experiments/build_weighted_graph.py build --edges
  .../edges_30m_k15.npz --embeddings-dir <30 round-0010 chunk dirs> --workdir
  .../_wg_30m --out .../edges_30m_k15_fuzzy.npz --chunk-size 150000
  --partitions 64 --phase-c-workers 12`.

## 150M — BLOCKED on upstream data integrity

The 150M graph indexes `fineweb[:50M] + redpajama[:50M] + pile[:50M]` (dict order,
first-N rows of each corpus's sorted `*.npy` shards; confirmed from
`build_150m_index_modal.py`). Recomputing distances needs those embeddings.

**Finding.** The local corpora at `/data/embeddings/*-chunked-120-all-MiniLM-L6-v2/train`
are RAW headerless normalised **fp32** `(N,384)` buffers (not `.npy`). redpajama
(84.1M rows) and pile (227.6M rows) are clean and byte-aligned. fineweb (99
shards, 93.9M rows) has **shard `data-00037` truncated mid-float** (802 trailing
bytes past a whole-row boundary). Alignment probe (edge-endpoint cosine vs random
on the 150M graph):

| node range | corpus | edge cos | rand cos | margin |
| --- | --- | --- | --- | --- |
| 10M–30M | fineweb pre-shard37 | 0.617 | 0.020 | **0.597 (aligned)** |
| 40M–49M | fineweb post-shard37 | 0.099 | 0.020 | **0.080 (BROKEN)** |
| 60M–90M | redpajama | 0.652 | 0.024 | 0.627 (aligned) |
| 110M–140M | pile | 0.711 | 0.019 | 0.692 (aligned) |

fineweb is aligned through shard 36 (node 34.97M); beyond the corrupt shard 37
(~35.88M) the local rows no longer correspond to the graph's node ids. A
single-offset repair does **not** realign shards 38+ (δ-search over
35M–49M found no cosine peak; all δ ≈ 0.31 vs 0.6 aligned), which means the
damage is not just shard 37's missing tail — other shards are likely truncated by
whole rows (staying 1536-divisible, so undetectable without a size manifest).

**Consequence.** ~14M of the 150M nodes (fineweb ~35.9M–50M, 9.4%) cannot be
given correct fuzzy weights from local data. No clean fineweb source covering
rows 30M–50M exists locally (only the round-0010 fp16 pack = fineweb[:30M]), and
Modal is wound down, so re-pulling the shard is not possible here. **I did not
ship a knowingly-corrupt artifact.**

**To finish the 150M once fineweb is clean** (re-pull the Modal `embeddings`
volume shards, or re-embed fineweb[:50M] chunks, verifying the alignment probe
above returns margin > 0.5 for every corpus range):

```
uv run python experiments/build_weighted_graph.py build \
  --edges /data/checkpoints/pumap/edges_150m_k15.npz \
  --corpus "<fineweb-120>/train:50000000" "<redpajama-120>/train:50000000" \
           "<pile-120>/train:50000000" \
  --raw-dtype '<f4' --workdir /data/checkpoints/pumap/_wg_150m \
  --out /data/checkpoints/pumap/edges_150m_k15_fuzzy \
  --sharded --chunk-size 150000 --partitions 128 --phase-c-workers 12
```

The tool's 150M path (corpus mix, raw-fp32 loader, sharded output + `index.json`
+ `load_sharded_edges` glue, parallel join) is implemented and exercised; the
sharded schema is format-identical to the 30M single file proven by V3.

## 405M / Path-B feasibility memo (measured)

**Distance-recompute cost is dominated by scattered reads, not GPU.** At 30M the
embeddings (23 GB fp16) fit the page cache and Phase A ran at ~1.3 s per 150k-node
chunk (whole 30M build: 502 s incl. symmetrization). Above RAM the picture
inverts — measured random-gather throughput against the clean 84M-row (129 GB)
redpajama corpus:

| trial (cold→warming) | 2.4M random fp32 rows | rate |
| --- | --- | --- |
| 0 | 117.8 s | 0.02M rows/s |
| 1–2 | 78–91 s | 0.03M rows/s |
| 3 | 43.2 s | 0.06M rows/s |

Each node-chunk gathers ~2.4M rows (self + 15 neighbors × 150k). Extrapolating at
~0.03M rows/s (working set > 100 GB RAM never fully warms):

- **150M**: 1000 chunks × ~80 s ≈ **~22 h** of gather-bound Phase A (GPU ~idle).
- **405M**: ~2700 chunks × ~80 s ≈ **~60 h** as-is.

**Levers (in order):**
1. **Sequential reads.** The reads are random because neighbor ids scatter. Sort
   each chunk's gather ids *within shard* (or external-sort the whole
   `edge→target` list once) so the memmap is read near-sequentially — NVMe
   sequential is ~100× random-4K. This is the single biggest win and is a small
   addition to `ShardedEmbeddings.gather`. Estimated Phase A then bounded by
   sequential bandwidth: 405M × 16 × 384 × 4 B ≈ 10 TB of reads / ~2 GB/s ≈ 1.4 h.
2. **fp16 on disk.** Halves bytes (405M → 620 GB → 310 GB). Combine with (1).
3. **Symmetrization already scales:** Phase B/C are partitioned and parallel;
   30M join ran 64 buckets in ~4 min on 12 workers. 405M ≈ 6.75B forward edges →
   ~80 GB partitions, P=256, 12–16 workers ≈ a few hours. RAM-safe (per-bucket).
   Output ~8–9B directed edges → sharded (~110 GB) — within `/data`.

**Path-B decision (topology).** V2 shows the shipped IVF_PQ is ~25% recall@15 and
PQ-limited. If the research side wants maps trained on a *faithful* neighborhood
(not just faithful weights on an approximate neighborhood), rebuild the topology:

| option | fit in 32 GB VRAM? | expected recall@15 | note |
| --- | --- | --- | --- |
| IVF-**Flat** (no PQ), GPU, higher nprobe | 405M×384 fp16 = 310 GB ✗ VRAM; shardable | ≫0.9 achievable | reads dominated by full vectors; do in shards |
| cuVS CAGRA (graph ANN) | 30M×384 fp16 (23 GB) + graph ✓; 150M/405M ✗ | ~0.95+ | must shard/stream above ~40M |
| two-pass IVF + exact re-rank of top-64 | ✓ (re-rank is cheap on GPU) | ~0.9+ | cheapest upgrade; reuses existing IVF centroids |

Recommendation: for a fresh 405M, **fuse** a higher-recall topology build with the
weight computation in one sequential pass (rank candidates, exact re-rank top-64
on GPU, then rho/sigma/membership on the re-ranked neighbors) — this both fixes
V2's recall gap and sidesteps the random-read tax, since the candidate vectors
are read once per shard rather than gathered randomly per edge.

## Reproduce the battery

```
uv run python experiments/build_weighted_graph.py validate-v1 --embeddings-list <round0010 chunk-00000>/embeddings.npy --n 100000 --k 15
uv run python experiments/build_weighted_graph.py validate-v2 --embeddings-list <round0010 chunk-0000{0,1,2}>/embeddings.npy --index .../faiss_ivf_pq_3m.index --n-base 3000000 --n-sample 50000 --nprobe 64
uv run python experiments/build_weighted_graph.py validate-v3 --artifact .../edges_30m_k15_fuzzy.npz --n-draw 10000000
uv run python experiments/build_weighted_graph.py validate-v4 --edges .../edges_30m_k15.npz --artifact .../edges_30m_k15_fuzzy.npz --embeddings-dir <30 round-0010 chunk dirs> --n-nodes 20
uv run python -m pytest tests/test_weighted_graph.py
```
