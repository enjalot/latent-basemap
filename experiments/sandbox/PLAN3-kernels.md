# Sandbox phase 3: kernel exploration (owner-budgeted, prepared 2026-08-12)

**Owner budget: 6 GPU-h of kernel arms at 2M, then 2 GPU-h of 6.25M checks.**
Owner also asked for a dedicated visual+metric page — that is
`build_kernels_page.py` → `http://gsv.local:8800/basemap-maps/sandbox/kernels/`
(render grid by family, full metric table, fidelity-vs-collapse Pareto
scatter). Rebuild it after arms complete; it only reads summaries and evals.

## Where 2a/2b landed (the baseline for this phase)

D1 passed: `umap-dose-x2` quick-FFR 0.2849 > replay-baseline 0.2764, and its
held-out FFR 0.3326 beats the registered recipe's 0.3253 — without collapsing
(r10/R 0.00101 vs legacy's 0.0001). D2 passed with nuance: plain `umap-kernel`
has net−regressor −0.001 (the July failure reproduced exactly — an undertrained
umap map is just a picture); dose x2 restores +0.0154. Legacy's much larger
+0.054–0.068 guard margins come partly from collapse itself. The min_dist
sweep (mind0 0.0 / kernel 0.1 / mind05 0.5) is monotone in r10/R (0.00084 /
0.00110 / 0.00200) and tissue (0.54 / 0.81 / 0.997): **min_dist — the kernel's
plateau radius, expressed through the fitted (a,b) pair — is the diffuseness
dial.** mind05 at dose x1 craters fidelity (quick 0.188, guard −0.030): too
diffuse at this scale, retested at 6.25M below per owner curiosity.

## Phase 3a — the kernel grid at 2M (~4.4 GPU-h of the 6)

All at dose x2 (the established convergence), seed 42, sealed R0216
substrate/graph. Arms exist in `knobs_2m.py`; ~24 min each:

| axis | arms | question |
| --- | --- | --- |
| min_dist (plateau) | `umap-md000-x2` `umap-md005-x2` `umap-md020-x2` `umap-md035-x2` `umap-md050-x2` (md010 = existing `umap-dose-x2`) | locate the sweet spot on the tightness↔diffuseness dial; is fidelity monotone or peaked? |
| tail exponent (gcauchy `(1+a·r^2b)^-α`) | `gc-a05-md000-x2` `gc-a05-md010-x2` `gc-a2-md000-x2` `gc-a2-md010-x2` | α<1 = heavier tail = more inter-cluster room (ft-SNE direction); α>1 lighter. α=1 is exactly the umap kernel |
| attraction/repulsion spectrum | `umap-pos02-x2` `umap-pos15-x2` | negative-sampling ratio is a hidden kernel dial (Damrich & Hamprecht); dose covaries at fixed horizon — 0.54 and 4.07 draws/edge respectively, stated on the page |

## Phase 3b — evals + page (CPU, after each batch of arms)

```bash
.venv/bin/python experiments/sandbox/heldout_eval.py eval
.venv/bin/python experiments/sandbox/build_kernels_page.py
```

## Phase 3c — reserve (~1.6 GPU-h of the 6)

Spend on what the grid reveals, in this order of preference:
1. One finer min_dist cell between the two best md values (24 min).
2. `gc` at the best md with α 0.75 or 1.5, whichever side won (24 min).
3. A seed-43 replicate of the single best arm (24 min) — look stability.
4. `umap-dose-x4` on the best arm only if its dose trend is still rising.

## Phase 3d — 6.25M checks (the 2 GPU-h budget)

On the sealed R0233 substrate + fuzzy graph (`--rung 6250k`), compare against
R0257's legacy maps (already rendered, dose x1, 255,142 updates):

1. **Overall winner at dose x2** (`--rung 6250k --arm <winner>`, ~75 min).
2. **Runner-up from a different family at dose x1** (~38 min).
3. **`umap-md050-x2` → at 6.25M dose x1** (~38 min) — owner is curious how
   the diffuse end behaves at higher N (prediction: diffuseness is
   kernel-intrinsic, roughly N-independent; the disc fraction shrinking with
   N may still improve its relative FFR).

Run 1 and 2; run 3 only if wall time remains inside the 2 h cap. Every 6250k
summary carries quick-FFR and r10/R (computed in-run); held-out eval at
6.25M would need a fresh truth build against the 6.25M substrate — not
prepared, report as future work rather than improvising.

## Reading the results

The deliverable is the **Pareto frontier**, not a single winner: fidelity
(held-out FFR primary, quick-FFR fallback) against collapse (r10/R), with
tissue and the regressor guard as tie-breakers. Specific things to look for:
- Does fidelity peak at an interior min_dist (md005–md020), or is the
  ordering monotone toward tight?
- Does a heavy tail (α=0.5) buy visible inter-cluster separation at small
  fidelity cost — the ft-SNE prediction — or does it shred the guard?
- Do the spectrum probes move the map the way the Böhm/Berens/Kobak spectrum
  predicts (more attraction → tighter, more repulsion → inflated)?
- At 6.25M: does the 2M frontier ordering survive scale (the July lesson
  says check, not assume)?

## Boundaries (unchanged)

Sandbox only: no rounds, no capabilities, no touching registered recipes.
GPU refusal when a compute process exists; issued rounds keep priority.
Write-once arm dirs; a firing receipt assertion means stop and report.
Promotion of any kernel is an owner decision with known costs (new treatment
⇒ new gate family; int8-path replication; a registered collapse/tissue
metric). Literature grounding: latent-labs research/markdowns — UMAP (0001),
t-SNE (0002), densMAP (0005), PaCMAP (0027), plus the three kernel-theory
papers added 2026-08-12 (attraction-repulsion spectrum; UMAP's true loss;
heavy-tailed kernels).
