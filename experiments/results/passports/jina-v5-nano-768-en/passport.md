# Space passport — jina-v5-nano-768-en

**Model:** `jina-embeddings-v5-text-nano`  |  **Dim:** 768  |  **Per-token:** False

**Source:** `gsv:/data/embeddings/fineweb-edu-sample-10BT-chunked-500-jina-v5-nano/train/`  
**Rows on disk:** 2,890,362 (11 shards, dtype float16)

> Pooled sentence embeddings, 500-token chunks, fineweb-edu (English). Primary target model.

## Metrics

| metric | value |
| --- | --- |
| mean cosine (random pairs) | +0.1156 |
| mean \|cosine\| | 0.1161 |
| std cosine | 0.0673 |
| cosine p1 / p50 / p99 | -0.006 / +0.107 / +0.327 |
| mean vector norm | 1.0000 |
| norm CV | 0.0000 |
| L_uniform (Wang–Isola, t=2) | -3.4979 |
| TwoNN intrinsic dim | 26.42 |
| PCA participation ratio | 202.3 (of 768) |
| %var top-2 / 10 / 50 | 4.7% / 15.4% / 37.8% |
| eigenvalue tail slope (ranks 10–100) | -0.624 |
| kNN in-degree skewness (k=50) | 1.652 |
| kNN in-degree max / mean | 328 / 50.0 |
| hubs (in-deg ≥ 2k) / anti-hubs (0) | 3538 / 17 |

*Samples: master 100,000 rows; TwoNN 20,000; PCA 50,000; hubness 50,000. Wall time 64.7s.*

## Interpretation

Mildly anisotropic (mean cos +0.116, mean|cos| 0.116, 5% variance in top-2 PCA): some global orientation but no strong dominant axis; expect moderate layout determinacy. Intrinsic dimension is high (TwoNN 26.4, PCA participation ratio 202 of 768), so 2D can only preserve local neighborhoods, not global geometry -- favors a dense kNN graph (k=50) and %-of-ceiling framing. Hubness is moderate (in-degree skew 1.7, max 328): the kNN graph is reasonably balanced, no special hubness mitigation indicated.
