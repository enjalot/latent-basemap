# Space passport — minilm-384-fineweb

**Model:** `all-MiniLM-L6-v2`  |  **Dim:** 384  |  **Per-token:** False

**Source:** `gsv:/data/embeddings/fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2/train/`  
**Rows on disk:** 50,999,581 (54 shards, dtype float32)

> Pooled sentence embeddings, 120-token chunks, fineweb-edu (English).

## Metrics

| metric | value |
| --- | --- |
| mean cosine (random pairs) | +0.0271 |
| mean \|cosine\| | 0.0665 |
| std cosine | 0.0847 |
| cosine p1 / p50 / p99 | -0.136 / +0.017 / +0.276 |
| mean vector norm | 0.9999 |
| norm CV | 0.0095 |
| L_uniform (Wang–Isola, t=2) | -3.8274 |
| TwoNN intrinsic dim | 39.00 |
| PCA participation ratio | 142.7 (of 384) |
| %var top-2 / 10 / 50 | 5.1% / 17.5% / 46.6% |
| eigenvalue tail slope (ranks 10–100) | -0.566 |
| kNN in-degree skewness (k=50) | 1.836 |
| kNN in-degree max / mean | 505 / 50.0 |
| hubs (in-deg ≥ 2k) / anti-hubs (0) | 4098 / 22 |

*Samples: master 100,000 rows; TwoNN 20,000; PCA 50,000; hubness 50,000. Wall time 43.1s.*

## Interpretation

Mildly anisotropic (mean cos +0.027, mean|cos| 0.066, 5% variance in top-2 PCA): some global orientation but no strong dominant axis; expect moderate layout determinacy. Intrinsic dimension is high (TwoNN 39.0, PCA participation ratio 143 of 384), so 2D can only preserve local neighborhoods, not global geometry -- favors a dense kNN graph (k=50) and %-of-ceiling framing. Hubness is moderate (in-degree skew 1.8, max 505): the kNN graph is reasonably balanced, no special hubness mitigation indicated.
