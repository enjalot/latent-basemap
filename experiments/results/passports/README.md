# Space passports — comparison

Pre-flight geometry diagnostics per embedding space (plan `latent-labs/guides/plan-basemap-atlas.md` §3). CPU-only, ≤100k-row samples. All spaces store L2-normalized vectors on disk, so mean norm ≈ 1 and norm CV ≈ 0.

## Comparison table

| space | dim | mean cos | mean\|cos\| | L_uniform | TwoNN ID | PCA PR | %var@2 | %var@10 | tail slope | hub skew | max in-deg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minilm-384-fineweb | 384 | +0.027 | 0.066 | -3.827 | 39.0 | 143 | 5.1% | 17.5% | -0.57 | 1.84 | 505 |
| jina-v5-nano-768-en | 768 | +0.116 | 0.116 | -3.498 | 26.4 | 202 | 4.7% | 15.4% | -0.62 | 1.65 | 328 |
| jina-v5-nano-768-zh | 768 | +0.341 | 0.341 | -2.558 | 52.2 | 125 | 7.7% | 21.1% | -0.72 | 4.02 | 1274 |
| jina-v5-nano-768-ar | 768 | +0.174 | 0.174 | -3.237 | 28.3 | 128 | 7.8% | 20.2% | -0.71 | 2.19 | 533 |
| mxbai-colbert-64-pile | 64 | +0.887 | 0.919 | -0.323 | 17.2 | 10 | 32.9% | 49.4% | -0.75 | 2.75 | 907 |

Reading the columns: **mean cos / mean|cos|** near 0 ⇒ isotropic (no dominant axis); **L_uniform** more negative ⇒ more uniform on the sphere; **TwoNN ID / PCA PR** high ⇒ genuinely high-dimensional (2D preserves locals only); **%var@2** low ⇒ layout underdetermined; **hub skew** high ⇒ kNN-graph quality concern.

## Per-space interpretation

### minilm-384-fineweb

Mildly anisotropic (mean cos +0.027, mean|cos| 0.066, 5% variance in top-2 PCA): some global orientation but no strong dominant axis; expect moderate layout determinacy. Intrinsic dimension is high (TwoNN 39.0, PCA participation ratio 143 of 384), so 2D can only preserve local neighborhoods, not global geometry -- favors a dense kNN graph (k=50) and %-of-ceiling framing. Hubness is moderate (in-degree skew 1.8, max 505): the kNN graph is reasonably balanced, no special hubness mitigation indicated.

### jina-v5-nano-768-en

Mildly anisotropic (mean cos +0.116, mean|cos| 0.116, 5% variance in top-2 PCA): some global orientation but no strong dominant axis; expect moderate layout determinacy. Intrinsic dimension is high (TwoNN 26.4, PCA participation ratio 202 of 768), so 2D can only preserve local neighborhoods, not global geometry -- favors a dense kNN graph (k=50) and %-of-ceiling framing. Hubness is moderate (in-degree skew 1.7, max 328): the kNN graph is reasonably balanced, no special hubness mitigation indicated.

### jina-v5-nano-768-zh

Anisotropic cone (mean cos +0.341, 8% variance in top-2 PCA): a dominant axis exists that a map can lock onto, which helps determinism but Euclidean UMAP may over-weight that axis; cosine/centering worth checking. Intrinsic dimension is high (TwoNN 52.2, PCA participation ratio 125 of 768), so 2D can only preserve local neighborhoods, not global geometry -- favors a dense kNN graph (k=50) and %-of-ceiling framing. High hubness (in-degree skew 4.0, max 1274): a few hub points dominate the kNN graph, a known kNN-quality concern that can distort attraction; mutual-kNN or hubness-aware edge weighting may help.

### jina-v5-nano-768-ar

Anisotropic cone (mean cos +0.174, 8% variance in top-2 PCA): a dominant axis exists that a map can lock onto, which helps determinism but Euclidean UMAP may over-weight that axis; cosine/centering worth checking. Intrinsic dimension is high (TwoNN 28.3, PCA participation ratio 128 of 768), so 2D can only preserve local neighborhoods, not global geometry -- favors a dense kNN graph (k=50) and %-of-ceiling framing. High hubness (in-degree skew 2.2, max 533): a few hub points dominate the kNN graph, a known kNN-quality concern that can distort attraction; mutual-kNN or hubness-aware edge weighting may help.

### mxbai-colbert-64-pile

Anisotropic cone (mean cos +0.887, 33% variance in top-2 PCA): a dominant axis exists that a map can lock onto, which helps determinism but Euclidean UMAP may over-weight that axis; cosine/centering worth checking. Intrinsic dimension is comparatively low and concentrated (TwoNN 17.2, PCA participation ratio 10 of 64, 33% of variance already in the top-2 axes): a few directions dominate, so a 2D map can capture more of the coarse geometry, but the strong anisotropy above means Euclidean structure is skewed. High hubness (in-degree skew 2.7, max 907): a few hub points dominate the kNN graph, a known kNN-quality concern that can distort attraction; mutual-kNN or hubness-aware edge weighting may help. NOTE: these are per-token ColBERT vectors, not pooled chunk embeddings -- geometry is token-level and not directly comparable to the pooled sentence spaces; deferred as a basemap substrate.

---

See `matryoshka_truncation.md` for the jina truncated-dim graph-overlap decision test, and each `<space-name>/passport.md` for the full metric table.
