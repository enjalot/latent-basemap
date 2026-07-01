# Space passport — mxbai-colbert-64-pile

**Model:** `mxbai-edge-colbert-v0-32m`  |  **Dim:** 64  |  **Per-token:** True

**Source:** `gsv:/data/embeddings/pile-uncopyrighted-chunked-500-mxbai-edge-32m/train/`  
**Rows on disk:** 390,852,269 (26 shards, dtype float16)

> PER-TOKEN ColBERT late-interaction vectors (one vector per token, not per chunk), 500-token chunks, pile-uncopyrighted. Included as a known-anisotropic contrast substrate; sampling draws random tokens.

## Metrics

| metric | value |
| --- | --- |
| mean cosine (random pairs) | +0.8874 |
| mean \|cosine\| | 0.9195 |
| std cosine | 0.2460 |
| cosine p1 / p50 / p99 | -0.914 / +0.925 / +0.989 |
| mean vector norm | 1.0000 |
| norm CV | 0.0000 |
| L_uniform (Wang–Isola, t=2) | -0.3227 |
| TwoNN intrinsic dim | 17.16 |
| PCA participation ratio | 10.2 (of 64) |
| %var top-2 / 10 / 50 | 32.9% / 49.4% / 92.6% |
| eigenvalue tail slope (ranks 10–64) | -0.752 |
| kNN in-degree skewness (k=50) | 2.748 |
| kNN in-degree max / mean | 907 / 50.0 |
| hubs (in-deg ≥ 2k) / anti-hubs (0) | 7119 / 570 |

*Samples: master 100,000 rows; TwoNN 20,000; PCA 50,000; hubness 50,000. Wall time 16.7s.*

## Interpretation

Anisotropic cone (mean cos +0.887, 33% variance in top-2 PCA): a dominant axis exists that a map can lock onto, which helps determinism but Euclidean UMAP may over-weight that axis; cosine/centering worth checking. Intrinsic dimension is comparatively low and concentrated (TwoNN 17.2, PCA participation ratio 10 of 64, 33% of variance already in the top-2 axes): a few directions dominate, so a 2D map can capture more of the coarse geometry, but the strong anisotropy above means Euclidean structure is skewed. High hubness (in-degree skew 2.7, max 907): a few hub points dominate the kNN graph, a known kNN-quality concern that can distort attraction; mutual-kNN or hubness-aware edge weighting may help. NOTE: these are per-token ColBERT vectors, not pooled chunk embeddings -- geometry is token-level and not directly comparable to the pooled sentence spaces; deferred as a basemap substrate.
