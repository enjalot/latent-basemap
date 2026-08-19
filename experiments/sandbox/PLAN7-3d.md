# PLAN7 spec: the 3D basemap track (owner-directed 2026-08-15, NOT scheduled)

**Status: PLAN ONLY. The owner will trigger execution later.** A separate
curiosity track — nothing here enters the 2D promotion program's evidence
chain, gates, or drift models, all of which are 2D-calibrated by
construction.

## The vision (owner's words, kept close)

These maps act as a **stable world map**; a 3D volume is a world you can
become familiar with — and the in-browser projection turns navigation into
**teleportation: type a concept, project it, and fly to it**. The extra
dimension buys real fidelity (the crowding problem relaxes as r³), and the
interface potential — an explorable, memorable volume with semantic
jump-to — is the product experiment.

## Phase A — training arms (2M sandbox, ~1.5 GPU-h total)

`n_components=3` is already a constructor param; the objective, kernel, and
fneg band (p90-radius-relative) are dimension-agnostic. Arms, each the 2D
twin's cost:

| arm | config | cost | question |
| --- | --- | ---: | --- |
| `3d-md000-x4-fneg10` | the promoted recipe, output 3 | ~47 min | the headline: the promoted recipe in 3D |
| `3d-md000-x4` | no fneg | ~47 min | does fog even need fneg in 3D? (tissue can route through empty volume) |
| `3d-md010-x2` | min_dist 0.1, dose ×2 | ~25 min | is min_dist 0 still optimal when crowding relaxes? |
| cuML-3D reference | `n_components=3` on the same substrate | ~4 min | the non-parametric benchmark, same discipline as 2D |

Optional later: one 6.25M cell for scale behavior (3D drift exponents are
their own question — see Phase B).

## Phase B — 3D instruments (CPU, ~half a day, BEFORE judging any arm)

- **Quick-FFR works unchanged** (cKDTree is dimension-general; disc is
  count-based). Expect higher values in 3D — that is the phenomenon, and
  cross-dimension FFR comparisons are apples-to-oranges by construction;
  say so wherever both appear.
- **Collapse metric re-derivation**: 2D's `r10·√N` packing normalization
  becomes `r10·N^(1/3)` in 3D. The healthy band must be re-referenced from
  the Phase A arms + cuML-3D (the 2D band [~0.9, 1.3] means nothing here).
  Verify the exponent empirically with a subsample-scaling check before
  trusting it (the 2D program did: measured 0.93 of predicted packing).
- **Fog in voxels**: 256³ occupancy (16.7M voxels) replaces the 1024² grid;
  re-derive the low-density threshold and the integer-degeneracy guard for
  voxel counts (sparser by construction). Report beside, never against, 2D
  fog numbers.
- Seed-replicate discipline carries over unchanged (one 3D replicate before
  believing any surprising 3D number).

## Phase C — pack + viewer 3D mode (CPU, the larger build)

- **Pack**: point records become u16 xyz + packed u32 = 10 B/point;
  LOD-only (no 2D tiling — spatial index = Morton-3D on a coarse voxel
  grid for viewport/frustum fetch). Density tiers become either 3-plane
  orthogonal projections (binned with existing 2D code — the cheap first
  render) plus an optional low-res voxel brick for volume impressions.
  Text sidecar/provenance: unchanged (dimension-independent — the big
  reuse win).
- **Viewer**: the v2 architecture was built for this — points are already
  VBOs drawn through camera uniforms. 3D mode = perspective camera
  (orbit/fly), LOD by distance, corpus tint/filter as-is, overlay layer
  renders picked-point/marker billboards. Hover previews shift from bins
  to nearest-neighbors-of-ray or coarse-voxel previews (samples/snippets
  machinery reusable keyed by voxel).
- **Alignment**: Procrustes generalizes (same SVD, 3×3 rotation) — 3D maps
  of one substrate can share a frame; a canonical orientation convention
  (e.g., corpus-centroid axes) is worth fixing early so the "stable world"
  stays visually stable across retrains.

## Phase D — teleportation (the interface experiment)

Type text → MiniLM (already in-browser) → 3D map head (same ONNX export
path; output dim 3) → **animate the camera flight to the landing point**,
leaving a marker trail. Add: a "you are here" breadcrumb list, jump-back,
and optionally project a whole pasted list (a dataset teaser — latent-scope
in miniature). The projection PoC needs only the head swap and a camera
path; the encoder, tokenizer, and loading UX are done work.

## Discipline

- Separate track: no 2D gate, drift model, or calibrated band applies; no
  3D number enters the promotion evidence; cuML-3D + own references only.
- Same sandbox rules: write-once arms, GPU refusal when rounds run,
  receipt-diff vs the sealed checkpoint (n_components is the arm's knob),
  CPU smoke for new eval code, positive controls for the re-derived
  instruments (planted 3D collapse/fog cases must fail).
- Rendering-for-the-record: every 3D arm still emits the three orthogonal
  2D projections so the aligned kernels-page discipline (visual record,
  cheap diffing) survives even before the 3D viewer exists.

## Cost summary when triggered

~1.5 GPU-h (Phase A) + ~half-day CPU (instruments) + the viewer 3D mode as
the main build (comparable to viewer v2's scope). Phases A+B alone answer
"is 3D better and by how much" with renders to look at; C+D make it a
place to stand.

## Open questions parked for execution time

- Does fneg earn its place in 3D, or does volume dissolve the fog problem?
- min_dist optimum in 3D (crowding-relaxed).
- Publish profile: 3D deep-points are 1.25× the 2D bytes at the same N —
  fine for GCS; decide whether the demo defaults to 2D or 3D.
- Whether the 100M-era "true atlas" is 3D-with-derived-2D-views — a
  step-5+ question, explicitly not this plan's.
