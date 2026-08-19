# Sandbox: 2M visual-quality knobs + 1M cuML reference (owner-directed, prepared 2026-08-12)

**Status: PREPARED, NOT RUN.** Owner asked for these to be planned and staged
while the round program owns the GPU. Everything below launches with one
command when the card is free; every script refuses to start if `nvidia-smi`
shows a compute process (the round runner keeps priority).

## What we are trying to alleviate (owner observation)

On the 2M and 6.25M binned density maps: (a) clusters connected by visible
lines of density, and (b) clusters that look spread out / diffuse.

Why the current recipe plausibly produces both:

- **The low-D kernel is heavy-tailed.** `legacy_lp` with the shipped
  `a = 1.0, b = 1.0` is similarity `1/(1 + ‖Δ‖)` — it decays *linearly* near
  zero and very slowly at range. Attraction saturates gently (loose packing)
  and the repulsion gradient at mid-range is weak (inter-cluster tissue is
  cheap to keep). In UMAP terms this is a large-min_dist regime.
- **A parametric map interpolates.** The MLP is a continuous function of the
  embedding; input space between clusters must land somewhere between them in
  2D. Non-parametric UMAP can tear those regions apart, an MLP cannot — some
  filament signal is the price of projectability, and the cuML reference run
  below measures exactly how much.
- **Bridge rows are real.** k = 15 fuzzy edges give fringe rows neighbors in
  two clusters; those rows sit between basins and drag density with them.
- **Low dose (0.6782 draws/edge) is an under-convergence choice.** It was
  registered because *high* dose degraded big rungs at h2048 (the dose ×
  capacity interaction), but at 2M the historical high-dose maps were the
  healthy ones. Diffuse clusters are consistent with stopping early.

## Experiment 1 — `knobs_2m.py`: one-variable arms at 2M, seed 42

All arms reuse the **sealed R0216 substrate and exact k15 fuzzy graph**
(`round-0216/queue-correction-3`, zero degree-zero rows) and R0217's exact
treatment otherwise; each arm changes one knob. ~113 upd/s ⇒ ~12 min per
80k-update arm. Outputs to `/data/latent-basemap/sandbox/2m-knobs/<arm>/`:
model, full-2M coordinates, config, fit log, binned `density.png`, and a
quick FFR@0.1% (20k queries, sealed graph edges as the high-D truth — same
spirit as the panel, not the sealed panel itself).

| arm | change | question | cost |
| --- | --- | --- | ---: |
| `replay-baseline` | none (R0217 seed 42 re-run) | does the harness reproduce the sealed map? trust gate for the others | ~12 min |
| `dose-x2` | horizon 80,163 → 160,326 (draws/edge 0.678 → 1.356, the historical high dose) | is the diffuseness under-convergence? | ~24 min |
| `kernel-a4` | `a` 1.0 → 4.0 | shrink the kernel's distance scale (min_dist↓ analog): tighter clusters, more filament contrast? | ~12 min |
| `kernel-b2` | `b` 1.0 → 2.0 | sharpen the falloff (‖Δ‖ → ‖Δ‖⁴ tail): stronger mid-range repulsion, thinner tissue? | ~12 min |
| `umap-kernel` | `low_dim_kernel` → `umap`, `a=1.577, b=0.895` (the standard fit for min_dist 0.1) | does the textbook kernel look better even though it measured worse at 8M? | ~12 min |

Total ≈ 1.2 GPU-h. Run serially:

```bash
cd ~/code/latent-basemap
.venv/bin/python experiments/sandbox/knobs_2m.py --arm replay-baseline   # first, always
.venv/bin/python experiments/sandbox/knobs_2m.py --arm dose-x2
.venv/bin/python experiments/sandbox/knobs_2m.py --arm kernel-a4
.venv/bin/python experiments/sandbox/knobs_2m.py --arm kernel-b2
.venv/bin/python experiments/sandbox/knobs_2m.py --arm umap-kernel
```

`--dry-run` validates paths, prints the exact constructor kwargs and horizon,
and diffs the unchanged fields against R0217's sealed train receipt without
touching CUDA. Judge `replay-baseline` before believing any other arm: its
density render and quick-FFR should be visually/metrically indistinguishable
from the sealed R0217 seed-42 map (bitwise equality is not expected).

**Read the results as visuals first.** The quick-FFR guards against an arm
that looks tidy by scrambling neighborhoods (the July lesson: a map can look
better and be worse). Every render lands on one comparison page:
`http://gsv.local:8800/basemap-maps/sandbox/2m-knobs/`.

## Experiment 2 — `cuml_1m.py`: non-parametric reference on identical rows

The cuML GPU UMAP is the "what would a free (non-parametric) layout do"
ceiling: it may tear inter-cluster tissue apart because nothing forces it to
be a continuous function. Comparing it against our parametric maps on the
**same 1M rows** separates "filaments are the parametric price" from
"filaments are a recipe defect".

Stages (one command each; stage 2 runs under `/data/latent-basemap/cuml_py`):

```bash
.venv/bin/python experiments/sandbox/cuml_1m.py sample   # CPU: 1M uniform rows (seed 0) from the sealed 2M substrate
.venv/bin/python experiments/sandbox/cuml_1m.py umap     # GPU few min: cuML UMAP n_neighbors=15, min_dist=0.1
.venv/bin/python experiments/sandbox/cuml_1m.py page     # CPU: binned renders + comparison page
```

The page shows, binned identically on the same 1M rows:
1. cuML UMAP direct (non-parametric reference),
2. our parametric exact-graph map (R0217 seed 42, view restricted to the rows),
3. our parametric cuVS-graph map (R0223 seed 42, same restriction).

Caveat stated on the page: cuML trains on the 1M subset while ours trained on
2M and are viewed on the subset — close enough for structure comparison, not
a matched experiment. A matched 1M parametric train is a cheap follow-up arm
(`knobs_2m.py` would need a 1M graph; build via cluster-spill or cuVS in
seconds if wanted).

## Already answered without GPU

The cuVS-graph 2M maps are **already rendered** (registry-side binned
renders): `round-0223-…cuvs-igd48-map-seed4{2,3,4}…` pages, and the drift
analysis showed swapping exact→cuVS moves points less than a seed reroll.

## Promotion path

These are sandbox artifacts (no round, no capability, no gate). If an arm
looks clearly better *and* holds quick-FFR, promote it by issuing a proper 2M
round under the standing protocol (seed family, sealed panel, review) before
letting it near the ladder recipe.
