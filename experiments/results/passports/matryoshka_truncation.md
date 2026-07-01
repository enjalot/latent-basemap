# Matryoshka truncation test — jina-v5-nano English

Plan `latent-labs/guides/plan-basemap-atlas.md` §3 decision point. jina-v5-nano is matryoshka-trained; if a truncated-dim k=50 graph reproduces the full-768 graph, large-scale graph construction gets cheaper.

**Sample:** 100,000 rows from `gsv:/data/embeddings/fineweb-edu-sample-10BT-chunked-500-jina-v5-nano/train/` (exact faiss IndexFlatIP on L2-normalized vectors, k=50).  Reduced from the planned 200k to 100,000: exact faiss IndexFlatIP search on this CPU box parallelizes poorly (~1.7 of 32 cores), so a full 200k×200k×768 build ran >8 min for the full-dim graph alone; 100k keeps the three-build test tractable (~a few minutes each) without changing the overlap conclusion. 

| truncation | mean edge overlap | median edge overlap | recall@10-of-50 |
| --- | --- | --- | --- |
| 768 (full) | 1.0000 | 1.0000 | 1.0000 |
| 256 (re-normalized) | 0.6734 | 0.6800 | 0.9618 |
| 128 (re-normalized) | 0.5140 | 0.5200 | 0.8412 |

Decision threshold (plan §3): >=90% edge overlap at 256d means graph construction at 20M+ can use truncated dims.

## Verdict

**FAIL** — 256d does not reach the 90% edge-overlap threshold; full-dim graph construction is required, or a higher truncation dim should be tested.

Note the split signal at 256d: recall@10-of-50 is high (0.962) while full k=50 edge overlap is only 0.673. The first ~256 matryoshka dims preserve each point's *closest* neighbors, but the broader 50-neighbor set reshuffles substantially — so truncation is safe for a small-k nearest-neighbor lookup but not for reproducing the dense k=50 UMAP graph, whose mid-range edges carry the manifold structure. Graph construction at 20M+ should therefore use full 768 dims (or an ANN index on full dims), not truncated dims.

At 128d edge overlap drops further to 0.5140 (recall@10 0.8412) — below the 90% bar, shown for the cost/quality curve.
