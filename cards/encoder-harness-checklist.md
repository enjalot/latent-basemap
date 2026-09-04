# Encoder harness checklist — evaluating a new encoder on a corpus

The standard battery for putting a new embedding encoder into the basemap comparison (owner's compare-embedding-
models muscle; overseer 2026-09-04). Each step is one line + the failure it guards against. Run top-to-bottom;
cone_stats gates whether the map is even meaningful. (Destined for a latent-labs guide once the cone step is
empirically validated by the NeoMME exp-1b/2b centered re-maps.)

1. **Embed with a VERIFIED prompt form.** Use the encoder's documented query/document template and ASSERT on the
   produced token ids (e.g. the doc-prefix token is present) — not on the call succeeding. *Failure guarded:*
   loose-`**kwargs` HF processors warn-and-ignore a bad prompt kwarg and silently emit plain-text embeddings; a
   try/except fallback never fires. (See NeoMME `<doc>` bug; `feedback_assert_on_output_not_call`.)

2. **cone_stats BEFORE mapping.** Compute the L2 norm of the substrate mean over unit-normalized rows (per-
   modality means too, if multimodal). If mean-norm **> ~0.3**, the space is an anisotropic cone: mean-center
   (per-substrate; per-modality if mixed) + renormalize, and report BOTH raw and centered numbers. *Failure
   guarded:* a strong shared component makes cosine-kNN map the cone, not the content — depressing FFR and
   forcing spurious modality/topic islands. (NeoMME mean-norm 0.84-0.95; `center_substrate.py`.)

3. **Truth graph.** Build the exact-kNN truth (k=15) on the substrate for scoring; verify the faiss metric
   convention before trusting neighbor order. *Failure guarded:* an IP-vs-L2 metric mixup inverts near/far and
   silently corrupts every downstream FFR/rarity number.

4. **Champion 2M map.** Train the champion recipe (md000/dose4/rankneg=25%-of-N/bs16k/pos0.10/fneg1.0/tanh4.0) on
   a 2M slice. *Failure guarded:* off-recipe hyperparameters make cross-encoder FFR numbers non-comparable.

5. **quick_ffr_v2 @0.1%.** Score the map's neighborhood preservation on its own exact truth. *Failure guarded:*
   the training-time `quick_ffr_at_0.1pct` and the rescored `quick_ffr_v2` are different keys — read v2 if present
   else the training key, or aggregators log None.

6. **Cluster spectrum.** k-means cluster-coverage / rare-region fraction over the substrate. *Failure guarded:* a
   coverage metric that SATURATES at large k hides real diversity differences; annotate saturation.

7. **Probe reception.** Push a fixed held-out probe set through the trained head, recall@15 vs identical truth.
   *Failure guarded:* in-sample FFR can look fine while the head generalizes poorly to unseen points.

8. **Forensics (as needed).** Blob/sink forensics on suspicious map regions (are dense sinks real topic manifolds
   or junk?). *Failure guarded:* treating a coherent topic cluster as an artifact (or vice-versa).

Assets/instruments: `experiments/sandbox/center_substrate.py` (cone removal), `neomme_exp2_analyze.py`
(`anisotropy_diagnostic`: raw / joint- / per-modality-centered margins + cone_stats), `knobs_2m.quick_ffr_v2`,
`image_map_pipeline.py` (knn/fuzzy/train).
