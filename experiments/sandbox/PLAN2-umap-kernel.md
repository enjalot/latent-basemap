# Sandbox phase 2: the umap-kernel program (prepared 2026-08-12)

**Trigger (owner, 2026-08-12):** on the phase-1 renders, `umap-kernel` is the
only parametric map whose structure looks right — continuous regions instead
of micro-clusters joined by filaments. cuML's non-parametric 1M reference
looks right too; `replay-baseline`, the cuVS sibling, and the a/b arms all
show the beads-and-filaments look. So the look is not "the price of being
parametric" in general — it is a property of the `legacy_lp` kernel.

**What `umap-kernel` is:** a full parametric map — R0217's sealed treatment
(same substrate, exact fuzzy graph, h2048, binary targets, low dose, seed 42)
with the low-D kernel switched `legacy_lp → umap` at `a=1.577, b=0.895`
(the standard fit for `min_dist = 0.1`). Its checkpoint projects new data
like any registered map. It is a candidate treatment, not a reference.

## The collapse mechanism — measured, 2026-08-12

Owner observation: every map except `umap-kernel` is "useless as a map —
everything collapses" into beads joined by filaments, while the cuVS paper
(lit 0197) shows healthy maps. Resolution, in three measurements:

- **Collapse metric** (median 2D radius to the 10th neighbor / p90 map
  radius, 20k sample): every `legacy_lp` map sits at ~1e-4 — a typical
  point's ten nearest neighbors fit inside 0.01% of the map radius, i.e.
  point-like beads. `sealed R0217` 0.00012, `cuVS-graph parametric` 0.00010,
  `dose-x2` 0.00009, `kernel-a4` 0.00010. The umap-kernel arm: **0.00110**;
  the cuML 1M reference: **0.00167**. The kernel moves this number 10–15x;
  graph choice, dose, and legacy `a`/`b` do not move it at all.
- **The graph is exonerated.** Exact-graph and cuVS-graph parametric maps
  collapse identically. In lit 0197, cuVS builds the *graph* and cuML's UMAP
  optimizer does the *layout* — their pictures are umap-kernel pictures. Our
  cuML reference is that method (RAPIDS cuML uses nn-descent internally),
  and the substrate is exactly unit-norm (measured: sd 0.0000), so cuML's
  euclidean metric coincides with our cosine graph. The baseline is proper.
- **Why the kernel does it:** `legacy_lp` similarity `1/(1 + a·‖Δ‖_{2b})` is
  degree-1 in distance near zero, so the attractive gradient on a positive
  pair never vanishes — pairs keep pulling until numerical zero and every
  tight neighborhood collapses to a bead; rows with edges into two beads
  stretch into filaments. The umap kernel `1/(1 + a·r^{2b})` has a vanishing
  gradient at r→0 (for 2b > 1) — the min_dist plateau, invented for exactly
  this — so neighborhoods keep finite area. Rescaling legacy's `a`/`b`
  changes the norm, not the degree-1 behavior, which is why `kernel-a4` and
  `kernel-b2` didn't help.

Two program-level consequences, flagged for the owner:

1. **FFR rewards collapse.** `dose-x2` is simultaneously the most collapsed
   and the best-scoring arm: a bead packs all true neighbors into the 0.1%
   disc trivially. The program's headline fidelity metric cannot see this
   failure mode, and `density_v2` (which could) is benched for the anchor
   defect. The `r10_over_map_radius_median` metric now computed by
   `heldout_eval.py` is the candidate gate for it.
2. **The two-sided purity band was smelling this.** R0257's 6.25M maps
   breached `k256` *above 1.0* — map neighborhoods purer than the high-D
   space — which is the collapse direction, and the R0260 one-sided ruling
   explicitly excuses that direction. Once a collapse metric is registered,
   the one-sided ruling deserves a revisit.

**The tension to resolve (history matters here):** the July kernel decision
kept `legacy_lp` at 8M for two reasons — higher FFR (0.578 vs 0.536), and the
kNN-regressor guard: a held-out point placed by averaging its high-D
neighbors' 2D positions nearly matched the umap-kernel map (net value only
+0.02–0.07) while the legacy map beat the regressor by +0.15–0.19. Phase 1
reproduces the fidelity gap under the current recipe (quick-FFR 0.254 vs
0.276) — but `dose-x2` also showed +0.030 is recoverable from convergence
alone, and the guard has never been run under the current graph/dose/target
semantics. Neither July result automatically stands.

## Phase 2a — can the fidelity come back? (GPU, ~1.5 h, run in this order)

New arms in `knobs_2m.py`, one knob each off the `umap-kernel` config:

| arm | change vs `umap-kernel` | question | cost |
| --- | --- | --- | ---: |
| `umap-dose-x2` | horizon ×2 (the historical high dose) | does convergence close the 0.022 quick-FFR gap while keeping the look? **the headline arm** | ~24 min |
| `umap-mind0` | `a=1.929, b=0.7915` (min_dist 0.0) | tightest packing the kernel allows — more fidelity from sharper attraction? | ~12 min |
| `umap-mind05` | `a=0.583, b=1.334` (min_dist 0.5) | the loose end, brackets the family | ~12 min |
| `umap-dose-x4` | horizon ×4 | only if `umap-dose-x2` moves fidelity but not enough | ~47 min |

Decision rule **D1**: if the best umap-family arm reaches quick-FFR ≥ 0.276
(the replay-baseline value) with the look intact, the visual win costs
nothing vs the registered recipe, and phase 2c is justified immediately.
Between 0.26 and 0.276 it is an owner trade-off; below 0.26 the kernel is
buying looks with real neighborhood damage and the burden of proof flips.

## Phase 2b — does the pretty map earn its keep as a projector? (one-time ~2 GPU-min, then CPU)

`heldout_eval.py`, two stages:

- `truth` (GPU, once): exact top-10 cosine neighbors in the 2M substrate for
  a 20k subset of R0233's sealed 200k held-out reserve (brute force, chunked).
  Shared across all arms and all future 2M sandbox work.
- `eval` (CPU, per arm): project the 20k held-out rows through the arm's
  checkpoint and compute
  1. **held-out FFR@0.1%** — do unseen rows land near their true neighbors?
  2. **the kNN-regressor guard** — place each held-out row at the mean 2D
     position of its true neighbors instead, and compute the same FFR. The
     difference `model − regressor` is the net's added value, the July guard
     re-run under current semantics.
  3. **tissue metrics** from the binned full-map grid: occupied-bin fraction
     (spread) and low-density tissue mass (fraction of points in bins under
     1% of peak — a first registrable number for "filaments").

Decision rule **D2**: the winning umap arm must beat its own kNN regressor on
held-out FFR by a clearly positive margin. If the regressor matches the net
(the July failure mode), the map is a picture, not a projector, and the
program should know that before falling in love with it. Note the caveat:
the reserve's composition (25/25/25/25 per corpus) differs from the training
mix (40/25/25/10); it is a fair *between-arm* comparison, not an absolute.

Run `eval` on `replay-baseline`, `dose-x2`, and every umap-family arm, so the
guard has the legacy contrast the July decision had. Also compute tissue
metrics for the cuML map (target values for "looks right").

## Phase 2c — stability and scale (GPU, ~1.7 h, only after D1+D2 pass)

- **Seeds 43/44 of the winning config** (`--seed`, ~24 min): does the look
  survive a reroll, and does the arm-to-arm drift stay in the seed band?
- **One 6.25M cell** on the sealed rung-1 substrate/graph (R0233), winning
  config, seed 42 (~40 min): the July umap-vs-legacy gap *widened* with
  scale — this is the kill-shot check before anyone imagines 100M with this
  kernel. Compare its render against R0257's legacy maps and its quick-FFR
  against their panel values.

## Phase 2d — promotion (rounds, owner decision, not sandbox)

If 2a–2c hold: changing the kernel is a **treatment change**. That means a
proper round family: seeds at 2M for a new gate family (the n=29 legacy
family does not carry over), int8-path replication (the 100M rung trains on
host-int8; R0262 already owes a 2M int8-fidelity cell — share the round), and
a registered tissue/visual metric so "looks right" is a number the next
regression can be caught by. Sequencing vs the 100M train is the owner's
call: train 100M on the registered legacy recipe now and treat the kernel as
a v2 retrain question, or hold the flagship until this resolves.

## Launch

```bash
cd ~/code/latent-basemap
.venv/bin/python experiments/sandbox/knobs_2m.py --arm umap-dose-x2
.venv/bin/python experiments/sandbox/knobs_2m.py --arm umap-mind0
.venv/bin/python experiments/sandbox/knobs_2m.py --arm umap-mind05
.venv/bin/python experiments/sandbox/heldout_eval.py truth        # GPU, once
.venv/bin/python experiments/sandbox/heldout_eval.py eval         # CPU, all arms
# then, if D1+D2 pass:
.venv/bin/python experiments/sandbox/knobs_2m.py --arm umap-dose-x2 --seed 43
.venv/bin/python experiments/sandbox/knobs_2m.py --arm umap-dose-x2 --seed 44
```

Same guards as phase 1: GPU refusal when a compute process exists, write-once
arm dirs, receipt assertion on every non-arm field. Results append to
`http://gsv.local:8800/basemap-maps/sandbox/2m-knobs/`.
