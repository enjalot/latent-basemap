# Space passport — jina-v5-nano-768-zh

**Model:** `jina-embeddings-v5-text-nano`  |  **Dim:** 768  |  **Per-token:** False

**Source:** `gsv:/data/embeddings/fineweb2-cmn_Hani-chunked-500-jina-v5-nano/train/`  
**Rows on disk:** 2,000,000 (1 shards, dtype float16)

> Pooled sentence embeddings, 500-token chunks, fineweb2 Chinese (cmn_Hani).

## Metrics

| metric | value |
| --- | --- |
| mean cosine (random pairs) | +0.3407 |
| mean \|cosine\| | 0.3407 |
| std cosine | 0.0999 |
| cosine p1 / p50 / p99 | +0.113 / +0.341 / +0.568 |
| mean vector norm | 1.0000 |
| norm CV | 0.0000 |
| L_uniform (Wang–Isola, t=2) | -2.5577 |
| TwoNN intrinsic dim | 52.16 |
| PCA participation ratio | 125.2 (of 768) |
| %var top-2 / 10 / 50 | 7.7% / 21.1% / 45.8% |
| eigenvalue tail slope (ranks 10–100) | -0.723 |
| kNN in-degree skewness (k=50) | 4.015 |
| kNN in-degree max / mean | 1274 / 50.0 |
| hubs (in-deg ≥ 2k) / anti-hubs (0) | 6478 / 768 |

*Samples: master 100,000 rows; TwoNN 20,000; PCA 50,000; hubness 50,000. Wall time 66.1s.*

## Interpretation

Anisotropic cone (mean cos +0.341, 8% variance in top-2 PCA): a dominant axis exists that a map can lock onto, which helps determinism but Euclidean UMAP may over-weight that axis; cosine/centering worth checking. Intrinsic dimension is high (TwoNN 52.2, PCA participation ratio 125 of 768), so 2D can only preserve local neighborhoods, not global geometry -- favors a dense kNN graph (k=50) and %-of-ceiling framing. High hubness (in-degree skew 4.0, max 1274): a few hub points dominate the kNN graph, a known kNN-quality concern that can distort attraction; mutual-kNN or hubness-aware edge weighting may help.
