# Sandbox phase 4: the fog program — reaching the clean-AND-healthy corner (owner-funded 2026-08-13)

**Owner budget: 10 hours WALL CLOCK, total, including GPU time.** The executor
is expected to overlap: launch a GPU arm, code the next mechanism while it
trains. GPU inside that wall is expected to land around 6 h; the wall is the
only hard cap. Promotion remains frozen until this program and the metric
registrations are considered together by the owner.

## The target

cuML's profile is simultaneously **clean** (fog/tissue 0.040 — crisp
inter-cluster space) and **healthy** (non-collapsed neighborhoods,
r10·√N ≈ 1.0–1.2). Our best parametric arm (`umap-md000-x4`) is healthy
(1.05–1.18) and the best-fidelity map in the program, but 11× foggier
(tissue 0.459). The fog is the parametric-continuity price: an MLP cannot
tear inter-cluster space, so it fills it with haze. This program tests
whether added loss mechanisms can cut the fog without breaking anything.

**Success rule (early exit):** an arm with tissue ≤ 0.15 AND held-out FFR
≥ 0.38 AND r10·√N in [0.9, 1.3] at 2M. If reached, stop spending on new
mechanisms and use remaining wall for a seed replicate + one 6.25M cell of
the winner. If not reached by wall end, the deliverable is the frontier and
an honest account of what each mechanism bought.

**Base config for every arm:** `umap-md000-x4` (min_dist 0, dose ×4) — the
program winner. Change exactly one mechanism per arm.

## Track 4A — mid-near pairs (PaCMAP-style) · run FIRST, machinery exists

`ParametricUMAP` already has the dormant mxbai-era machinery:
`midnear_enabled`, `mn_pairs_per_batch` (core.py constructor),
`_sample_midnear_features` / `_sample_midnear_features_device`
(~core.py:1029+, PaCMAP candidate selection). Steps:

1. **CPU smoke first** (CUDA hidden): construct with `midnear_enabled=True`,
   tiny synthetic X, verify the code path runs end-to-end and stamp what
   loss weight/schedule it applies (read the loss section; document it in
   the arm summary — this code predates the current program and its exact
   semantics must be stated, not assumed).
2. Three GPU arms (~47 min each at ×4): `mn_pairs_per_batch` at roughly
   {small, medium, large} relative to the 409 positives/batch — pick values
   after reading the mechanism; name arms `umap-md000-x4-mn<value>`.
3. Historical caution to carry in the writeup: July's forensics found
   mid-near's density rescue at 8M was mostly a metric artifact — but that
   was the legacy kernel judged by broken metrics. The fog metric makes it a
   clean question now.

## Track 4B — densMAP-style local-density term · code while 4A trains

Add an opt-in loss term aligning each point's local 2D radius with its
high-D radius (lit 0005). Implementation contract (same spirit as the
gcauchy addition — opt-in constructor params, legacy paths untouched):

1. Precompute per-row high-D kNN radius r_hd once from the sealed 2M
   substrate: distance to the k=15th cosine neighbor. The sealed exact graph
   gives the neighbor IDs (edges npz); radii need one gather+dot pass over
   the substrate — GPU ~1–2 min as a standalone script, or reuse the
   heldout-truth machinery pattern. Cache to
   /data/latent-basemap/sandbox/density-radii-2m.npy.
2. Loss: for each batch's source rows, penalize disagreement between
   log r_2d (distance to the k-th neighbor within-batch is too noisy — use
   the kernel-radius surrogate densMAP uses: local Var/E of q over the
   batch pairs, or a documented simpler surrogate; state the choice) and
   log r_hd, weighted by `density_weight`. Constructor params:
   `density_weight=0.0` (off by default), `density_radii_path=""`.
3. Three GPU arms: `density_weight` log-spaced {low, mid, high} →
   `umap-md000-x4-dw<value>`. CPU smoke with CUDA hidden before any launch.

## Track 4C — fog-targeted negatives · only with remaining wall

Oversample negative pairs whose CURRENT 2D distance is mid-range (where fog
lives) instead of uniform-random. Cheapest honest version: after each
epoch's first half, sample extra negatives from pairs binned at 0.1–0.4 of
map radius. This is speculative (no direct literature anchor; spectrum
papers adjacent). One or two arms max (`umap-md000-x4-fneg`), only if 4A/4B
leave wall.

## Eval + reporting (unchanged harness)

After each arm: in-run quick-FFR + r10 (automatic), then
`heldout_eval.py eval` and `build_kernels_page.py` (the page gains the new
arms automatically). Final deliverable message: per-arm table (heldFFR,
net−regressor, r10·√N, tissue, quick), visual read vs the cuML card, which
mechanism moved fog and at what cost, and wall/GPU accounting against the
10 h budget.

## Boundaries

- Sandbox only: no rounds, no capabilities, no registered-recipe changes.
- Core changes: opt-in parameters only, default-off, legacy/umap paths
  byte-untouched for existing configs; CPU smoke before every GPU launch of
  new code; write-once arm dirs; receipt assertion fires ⇒ stop and report.
- GPU refusal guard stays; issued rounds keep priority.
- Do NOT touch `experiments/metrics/` — a parallel workstream (metric
  options A/B/C) owns that directory today.
- Wall accounting starts at your acknowledgment; report spend with the
  results and stop at 10 h regardless of state.
