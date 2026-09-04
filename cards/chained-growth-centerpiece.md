# Chained Growth — validation-block centerpiece (DRAFT for overseer review, 2026-09-04)

**Status: PREREG DRAFT — do not run GPU until reviewed.** Replaces/absorbs the old valbatch design (overseer G).

## Motivation — the persistence gap
The anchored-λ harness saves **coords only**, not the model / preprocessing / alignment-frame state. So every
"update" so far started from a fresh fit of the previous *coordinates*, never from the previous *system state*.
We have therefore **never demonstrated update-from-update**: an anchored map that ingests new arrivals, persists,
and is then itself the substrate the next batch updates from. That chained property — not single-step anchoring —
is what a living basemap service actually requires. This experiment builds the persistence and measures the
chain.

## Build (prereq, no science until this lands)
Extend the anchored harness to persist and reload a **MapState**: (1) the parametric model weights, (2) the
input preprocessing (the `_norm` stats / int8 quantization params — exact, so a reload is bitwise-equivalent to
the in-memory model), (3) the coordinate **frame** (the RIGID gauge from review-item B — rotation+translation,
no scale; learned scale stored separately), (4) the fuzzy-graph build params + kNN config, (5) a receipt (source
commit, resolved config, input/output hashes, gen key — review-item A5). Assert on reload: transform(x_ref)
reproduces the pre-save coords within 1e-4 (bitwise-ish), else fail closed.

## Protocol — the chain
Corpora: define the arrival stream from the existing evolbench tranches (numeric defn below). Stages:
1. **T0** — fit anchored map on the base cohort; persist MapState_0.
2. **ordinary arrival** — an in-distribution batch transformed BY MapState_0 (no trigger); persist coords.
3. **OOD-A trigger** — an out-of-distribution cohort (reddit) arrives; the trigger fires; anchored fine-tune
   from MapState_0 with anchor_hold_weight=w (champion w=0.02 from the λ frontier) -> **MapState_1** (persist).
4. **arrivals-by-#1** — the next in-distribution batch transformed BY MapState_1 (proving #1 is usable as a
   live head, not just a checkpoint).
5. **OOD-B trigger** — a second OOD cohort (CA or bluesky) arrives; anchored fine-tune **FROM MapState_1** ->
   **MapState_2** (persist). This is the update-from-update step the whole experiment exists to demonstrate.

## Per-stage metrics (numeric prereg — every prediction gets a threshold, review-item F)
All FFR = v2 @0.1% on each cohort's own exact-k15 truth. All churn/drift in the RIGID shared frame (item B),
one frame defined in a shared module consumed by both scorer and viz exporter.
- **per-cohort FFR**: each stage's map scored on (a) T0-retained cohort, (b) each arrived cohort, (c) the active
  OOD cohort. Predict: OOD-cohort FFR after anchored update >= frozen-baseline OOD FFR + 0.08 (absolute).
- **active-vs-holdout anchor drift**: mean coord displacement of anchor rows, split into anchor-active (in the
  fine-tune's anchor set) vs holdout (anchors held out of the loss). Predict: holdout drift <= 1.5x active drift
  (anchoring generalizes to unseen anchors, not just memorizes the active set).
- **cumulative old-coord churn**: |coord(stage_k) - coord(stage_0)| for surviving T0 rows, in rigid frame,
  normalized by frame diameter. Predict: cumulative churn at MapState_2 <= 2x the single-step churn (chaining
  does not compound churn super-linearly).
- **new-to-old reception**: arrived rows transformed by the PRIOR MapState (never trained on them) -> recall@15
  vs their own truth. Predict: reception recall@15 >= 0.6x a from-scratch fit's recall on the same rows.
- **trigger false-positive rate**: run the trigger detector on a NO-SHIFT in-distribution stream (T0 resampled).
  Predict: FPR <= 0.05 over N>=20 no-shift batches.
- **full latency/cost per stage**: load + preproc + index(kNN/fuzzy) + transform + align + serialize + store,
  each timed separately (not just train wall). Report as the service-cost profile.

## Baselines (each stage, same cohorts, same frame + v2 scorer)
- **frozen** (MapState_0 transforms everything, never updated).
- **full parametric retrain** (champion recipe from scratch on the cumulative corpus at each stage).
- **fixed-seed full UMAP** (non-parametric; random_state fixed — the item-C noise-floor baseline; also run
  same-snapshot rerun pairs to get the pure-optimizer churn floor, so growth-churn is reported ABOVE that floor).
- **reduced AlignedUMAP** if feasible (transductive incremental baseline; note if it cannot do held-out
  reception — that inability is itself the differentiator for the parametric approach).

## Numeric definitions to lock before running (F)
- "S2" = the second OOD trigger cohort's post-update stage (MapState_2); "cohort"/"batch" sizes; the exact
  tranche->stage mapping; trigger detector + its threshold; anchor set size + holdout fraction; w=0.02.
- FFR instrument = knobs_2m.quick_ffr_v2, exact knn_indices where present.

## Risks / open questions for review
- Does bitwise-reproducible reload hold across the int8 quantization path? (assert; if not, use fp substrate for
  the chain and note the residency caveat.)
- Frame accumulation: rigid gauge avoids the scale-collapse (item B), but rotation drift could accumulate over
  the chain — the cumulative-churn metric + the shared frame module are the guard; verify on a 2-step dry run
  before the full chain.
- Cost of the persistence assert (a full transform of x_ref each save) — acceptable at these sizes.

**Requested before GPU:** overseer sign-off on (1) the tranche->stage mapping + cohort sizes, (2) the numeric
thresholds above, (3) the MapState persistence contract, (4) baseline set (esp. whether reduced AlignedUMAP is in).
