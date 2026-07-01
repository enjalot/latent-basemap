# Space passport — jina-v5-nano-768-ar

**Model:** `jina-embeddings-v5-text-nano`  |  **Dim:** 768  |  **Per-token:** False

**Source:** `gsv:/data/embeddings/fineweb2-arb_Arab-chunked-500-jina-v5-nano/train/`  
**Rows on disk:** 2,000,000 (1 shards, dtype float16)

> Pooled sentence embeddings, 500-token chunks, fineweb2 Arabic (arb_Arab).

## Metrics

| metric | value |
| --- | --- |
| mean cosine (random pairs) | +0.1741 |
| mean \|cosine\| | 0.1741 |
| std cosine | 0.0866 |
| cosine p1 / p50 / p99 | +0.029 / +0.159 / +0.450 |
| mean vector norm | 1.0000 |
| norm CV | 0.0000 |
| L_uniform (Wang–Isola, t=2) | -3.2374 |
| TwoNN intrinsic dim | 28.32 |
| PCA participation ratio | 128.3 (of 768) |
| %var top-2 / 10 / 50 | 7.8% / 20.2% / 45.9% |
| eigenvalue tail slope (ranks 10–100) | -0.710 |
| kNN in-degree skewness (k=50) | 2.192 |
| kNN in-degree max / mean | 533 / 50.0 |
| hubs (in-deg ≥ 2k) / anti-hubs (0) | 4323 / 41 |

*Samples: master 100,000 rows; TwoNN 20,000; PCA 50,000; hubness 50,000. Wall time 64.9s.*

## Interpretation

Anisotropic cone (mean cos +0.174, 8% variance in top-2 PCA): a dominant axis exists that a map can lock onto, which helps determinism but Euclidean UMAP may over-weight that axis; cosine/centering worth checking. Intrinsic dimension is high (TwoNN 28.3, PCA participation ratio 128 of 768), so 2D can only preserve local neighborhoods, not global geometry -- favors a dense kNN graph (k=50) and %-of-ceiling framing. High hubness (in-degree skew 2.2, max 533): a few hub points dominate the kNN graph, a known kNN-quality concern that can distort attraction; mutual-kNN or hubness-aware edge weighting may help.
