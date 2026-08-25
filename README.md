# latent-basemap

`latent-basemap` trains a small parametric UMAP-style head as a reusable 2D
coordinate frame for text embeddings. The current study uses
`sentence-transformers/all-MiniLM-L6-v2` and a composition-controlled training
set of up to 100 million text chunks.

This is a research repository, not a packaged release. Start with the
[`REVIEWER_GUIDE.md`](REVIEWER_GUIDE.md) for the short scientific path or the
[`paper`](paper/paper.md) for the full argument.

## Current result

As of 2026-08-18:

| stage | status | result |
|---|---|---|
| 2M, 13 seeds | sealed | family thresholds for spacing, fog, held-out FFR, and purity |
| host-int8 check | sealed | seed-paired map remains inside all fp32 family bands |
| 50M, 3 seeds | sealed pass | mean spacing 1.0140; fog 0.1165-0.2472; FFR 0.5495-0.5594 |
| 100M graph | sealed pass | strict recall@15 0.99590 on a fixed 500K-row probe |
| 100M, 3 seeds | running | manuscript uses explicit `{{100M_*}}` result tokens |

The final 100M run uses one RTX 5090. A session-management signal interrupted an
early attempt after training but before a complete evidence receipt; that
checkpoint is excluded and the seed is being retrained. See the
[`100M incident note`](https://github.com/enjalot/latent-labs/blob/main/logs/process/2026-08-18_round0268-session-scope-incident.md).

## What the model does

For a versioned encoder $E$ and projection head $f$, the frame is

```text
text -> 120-token preprocessing -> MiniLM -> L2-normalized 384D vector -> 2D head
```

The head is an 11,809,282-parameter residual bottleneck MLP. Once the encoder,
head, preprocessing, and map-coordinate transform are frozen, new text can be
placed by inference alone. No neighbor graph or layout optimization is required
at projection time.

"Stable" means repeated inference within that fixed version. It does not mean
that independent training runs, new corpus snapshots, or new encoder versions
produce the same frame.

## Inspect a checkpoint

The selected checkpoint is not public yet. With a collaborator-provided
checkpoint:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ./basemap/pumap
python -m pip install sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer
from parametric_umap import ParametricUMAP

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
basemap = ParametricUMAP.load("/path/to/basemap-model.pt", device="cpu")

texts = [
    "a recipe for sourdough bread",
    "a proof about gradient descent convergence",
]
x = encoder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
xy = basemap.transform(x)
print(xy.shape)  # (2, 2)
```

`transform` reads array-like inputs in batches, so a memory-mapped embedding
matrix does not need to be copied into RAM at once. The returned coordinate array
is materialized.

## Training recipe

The promoted treatment is defined by these fields:

| field | value |
|---|---|
| graph | approximate high-recall cosine k15, exact fp32 candidate rerank, UMAP fuzzy union |
| head | residual bottleneck MLP, 384 -> 2048 -> 1536 residual neck -> 2048 -> 2 |
| low-dimensional kernel | UMAP curve, `a=1.9328`, `b=0.7905` (`min_dist=0`) |
| batch | 8,192 pairs: 409 positive edges, 7,783 uniform non-self negatives |
| positive treatment | uniform edge sampling, binary target 1 |
| negative treatment | binary target 0; 2x BCE weight in `[0.1, 0.4]` times batch p90 radius |
| optimizer | AdamW, lr `1e-3`, 200-update warmup, cosine decay, bf16, clip norm 1 |
| 50M/100M dose | 2x the calibrated base update horizon, about 1.356 positive draws per directed edge |
| large-input storage | per-row symmetric int8 values with fp16 scales in host memory |

This is a UMAP-derived objective, not stock Parametric UMAP. Fuzzy edge weights
define graph topology but do not weight the binary training loss.

The exact 50M and 100M production modules are on branch
`basemap-100m/round-0208`, release `d3ac5c4`. This publication branch diverged
from that research branch. Do not infer the final execution config from a short
constructor example or from package defaults.

## Evaluation

The scale decision uses three primary checks:

- **Normalized 10-neighbor spacing:** median 2D tenth-neighbor distance divided
  by p90 map radius, multiplied by `sqrt(N_eff)`. At large rungs, `N_eff` is a
  fixed seeded sample of 16,777,216 rows. Low values detect contraction.
- **Fog:** fraction of points in occupied 1024x1024 bins below 1% of the peak-bin
  count. Degenerate measurements fail closed.
- **Held-out FFR:** fraction of a query's true high-dimensional top-10 neighbors
  found in its closest 0.1% of the 2D map. This is a coarse placement measure,
  not recall@10.

Purity fidelity is gated on the 2M reference family and reported descriptively at
50M/100M because the large substrates use a different reference-row identity.
Definitions and thresholds are in [Section 4 of the paper](paper/paper.md).

## Viewer and static map packs

[`mapviewer/`](mapviewer/) is a Vite and TypeScript viewer for static map packs.
It supports multiresolution density, source-corpus filtering, level-of-detail
points, source-text lookup, and local text projection through MiniLM plus an ONNX
map head.

```bash
cd mapviewer
npm install
npm run fixtures
npm run dev
```

The viewer defaults to `http://localhost:5195/`. Node 24 is required. The checked
in fixtures are synthetic; real packs and model files are not committed. See the
[`map pack report`](experiments/mappack/REPORT.md),
[`ONNX report`](experiments/mappack/onnx/REPORT.md), and
[`viewer README`](mapviewer/README.md).

## Repository map

- `paper/` contains the Markdown manuscript, bibliography, and final-result
  checklist.
- `basemap/pumap/parametric_umap/` contains the PyTorch projection package.
- `basemap/panel_v2.py` contains the shared FFR and purity evaluator.
- `experiments/metrics/` contains map-level metric tools.
- `experiments/mappack/` contains the static pack builder and ONNX export.
- `mapviewer/` contains the browser viewer.
- `experiments/sandbox/` contains treatment-selection experiments.
- `basemap/round*`, `experiments/round*`, and the companion `latent-labs` repo
  contain the preregistered research history and evidence machinery.

Most readers should stay in the paper, reviewer guide, reports, and the files
named above. The hundreds of round modules preserve decisions and execution
contracts; they are not the shortest route to the method.

## Checks

Focused model and metric tests on this publication branch:

```bash
PYTHONPATH=. .venv/bin/pytest -q \
  tests/test_low_dim_kernel.py \
  tests/test_edgelist_smoke.py \
  tests/test_panel_v2.py \
  tests/test_persistence.py
```

Paper build:

```bash
cd paper
pandoc paper.md --citeproc --pdf-engine=tectonic -o /tmp/latent-basemap-paper.pdf
```

Viewer build and smoke test:

```bash
cd mapviewer
npm install
npm run build
npm run smoke
```

## Evidence

Readable result records live in
[`enjalot/latent-labs`](https://github.com/enjalot/latent-labs). The reviewer
guide links the specific 2M, 50M, graph, and 100M records. Large substrates,
graphs, coordinates, and checkpoints remain in the internal artifact store; each
record includes content digests and execution receipts.

## License and citation

The root repository, model, and map-pack licenses have not been selected. The BSD
license under `basemap/pumap/` applies to that nested upstream-derived package,
not automatically to the whole repository. External sharing is currently for
friendly review, not redistribution. Citation metadata will be added with the
release tag.
