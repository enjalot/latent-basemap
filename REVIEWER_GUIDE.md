# Reviewer guide

This repository is a research draft for a fixed parametric UMAP frame trained on
100 million MiniLM text embeddings. The main question is whether the method and
evaluation support that claim, not whether the rounds infrastructure is elegant.

## Ten-minute route

1. Read the [abstract and introduction](paper/paper.md#abstract).
2. Read [Sections 3.3-3.5](paper/paper.md#33-projection-head-and-loss) for the
   exact objective, dose, and host-int8 representation.
3. Read [Sections 4.2-4.3](paper/paper.md#42-metrics) for the metric definitions
   and decision rules.
4. Read [Sections 5.2-5.3](paper/paper.md#52-the-2m-gate-family) for the 2M
   calibration family, 50M result, and assumed-success 100M result.
5. Read [Section 7](paper/paper.md#7-limitations) before judging the scope of the claims.

The manuscript is about 5,000 words. The full research ledger is available for
audit, but it is not required to understand the paper.

## Status on 2026-08-18

- The 2M 13-seed calibration family is sealed (R0265).
- Host-int8 map fidelity is sealed (R0266).
- The 50M three-seed run passes its preregistered spacing, fog, and held-out FFR
  criteria (R0267).
- The 100M graph is built and qualified. Its sampled strict recall@15 is 0.99590.
- The final 100M three-seed training round (R0268) is running. An infrastructure
  signal killed the first seed after training and before its evidence receipt was
  complete; that checkpoint is excluded from evidence and the seed is being
  retrained under a persistent systemd user service.
- The paper is written from the expected completed state. Unsealed values are
  visible `{{100M_*}}` tokens. [The result checklist](paper/RESULTS_CHECKLIST.md)
  names the only acceptable source for each value.

## The claim in one paragraph

For a fixed `all-MiniLM-L6-v2` encoder, we train an 11.8M-parameter residual MLP
to map normalized 384-dimensional embeddings into one reusable 2D frame. The
training graph uses UMAP fuzzy topology, but the optimization uses uniformly
sampled positive edges with binary targets and uniformly sampled random
negatives. Negative pairs in a current 2D distance band receive twice the BCE
weight. A 13-seed family calibrates checks for map contraction, diffuse
low-density mass, and coarse placement of held-out queries. The recipe passes at
50M and is being confirmed at 100M on one RTX 5090 using per-row int8 host
storage.

## What we do not claim

- The objective is not stock Parametric UMAP and does not optimize fuzzy edge
  weights directly.
- The 100M neighbor graph is high-recall approximate, not exact.
- Held-out FFR is not recall@10. It asks whether true top-10 neighbors occur in
  the nearest 0.1% of the map.
- A frozen head is stable only within a versioned encoder-head contract. We have
  not established stability across retraining or encoder updates.
- The result covers one encoder and one corpus mixture.
- There is no matched 100M ParamRepulsor, NOMAD, or stock Parametric UMAP
  baseline yet.
- Purity is descriptive at 50M and 100M because the reference row identity
  differs from the 2M calibration family.

## Feedback requested

1. **Nomenclature.** Is "UMAP-derived parametric objective" the right description
   for binary targets and uniform edge sampling on a UMAP fuzzy topology? Which
   parts should still be described as Parametric UMAP?
2. **Negative treatment.** Is the positioning relative to ParamRepulsor, hard
   negative mining, and the known sampled UMAP loss accurate? What comparison
   would be most informative?
3. **Evaluation.** Are normalized 10-neighbor spacing, fog, and held-out FFR
   defensible operational gates? Which standard metric or planted failure is
   missing?
4. **Scale claim.** Does the 2M calibration plus three seeds at 50M and 100M
   support the stated within-recipe scale conclusion?
5. **Paper boundary.** Should the fixed-frame system, training treatment, and
   static browser delivery remain one paper, or is one component distracting
   from the dimensionality-reduction result?
6. **Frame lifecycle.** What minimum experiment would support a useful statement
   about retraining or version migration?

## Evidence map

| topic | readable record |
|---|---|
| 2M 13-seed family and thresholds | [R0265 result](https://github.com/enjalot/latent-labs/blob/main/basemap-100m/result-0265-2026-08-15.md) |
| host-int8 map fidelity | [R0266 result](https://github.com/enjalot/latent-labs/blob/main/basemap-100m/result-0266-2026-08-15.md) |
| 50M three-seed result | [R0267 result](https://github.com/enjalot/latent-labs/blob/main/basemap-100m/result-0267-2026-08-17.md) |
| 100M preregistration | [100M flagship plan](https://github.com/enjalot/latent-labs/blob/main/basemap-100m/plan-100m-flagship-2026-08-17.md) |
| 100M graph qualification | [R0241 result](https://github.com/enjalot/latent-labs/blob/main/basemap-100m/result-0241-2026-08-10.md) |
| 100M run interruption | [incident note](https://github.com/enjalot/latent-labs/blob/main/logs/process/2026-08-18_round0268-session-scope-incident.md) |
| prior-art differentiation | [research summaries](https://github.com/enjalot/latent-labs/tree/main/research/markdowns) |
| map pack validation | [local report](experiments/mappack/REPORT.md) |
| ONNX and browser parity | [local report](experiments/mappack/onnx/REPORT.md) |

The result reports contain extensive execution and receipt detail. Their opening
"Digest" sections contain the scientific result; later sections are the audit
trail.

## Code map

- [`basemap/pumap/parametric_umap/core.py`](basemap/pumap/parametric_umap/core.py)
  contains the projection loss and training loop.
- [`basemap/pumap/parametric_umap/models/mlp.py`](basemap/pumap/parametric_umap/models/mlp.py)
  contains the residual bottleneck head.
- [`basemap/panel_v2.py`](basemap/panel_v2.py) contains FFR and purity scoring.
- [`experiments/mappack/`](experiments/mappack/) contains the static package and
  ONNX export.
- [`mapviewer/`](mapviewer/) contains the browser viewer.
- The R0265-R0268 production modules are included on `main`. Commit `d3ac5c4`
  is the frozen checkout used for the current 100M execution; use it for
  byte-level comparison with run receipts.

## Known gaps before submission

- Fill and verify the sealed 100M values.
- Add matched modern parametric baselines or narrow the comparison claim further.
- Complete the corpus-coverage and out-of-distribution analysis.
- Add the aligned scale-ladder and viewer figures marked in the manuscript.
- Publish a checkpoint, config, small map pack, and one-command inference example.
- Choose a repository, model, and data license. The root repository currently has
  no license; the BSD file under `basemap/pumap/` covers that nested package only.
- Tag the integrated review release after the 100M values are sealed and filled.
