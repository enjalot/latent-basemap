# Substrate-draw universality (owner-approved 2026-08-30)

## Question
A1's cross-scale head audit used NESTED heads (2M ⊂ 6.25M etc.), so it measured EXTRAPOLATION, not
draw variance. This is the first INDEPENDENT-DRAW comparison: three disjoint composition-matched 2M
MiniLM substrates (A/B/C), trained with the SAME seed → bit-identical inits → any output difference
is purely the DATA DRAW. The number to put next to P1.5's seed-variance (same substrate, seeds 42/43):
data-lever vs init-lever.

## Slices (build_draw_universality.py)
- Three 2M substrates at the baseline 40/25/25/10 mix (fineweb/redpajama/pile/starcoder =
  800k/500k/500k/200k), drawn from the 150M pool.
- Disjoint BY CONSTRUCTION: one rng(42) draws 3× the per-corpus count in a single call, random-split
  first→A next→B next→C, so A/B/C are pairwise disjoint (and NOT a seed-42 replay of the baseline —
  baseline coords are coord-excluded per corpus).
- Provenance (corpus,shard,row) + manifest per slice; `draw-univ-proofs.json` holds all proofs.

## Proofs (all zero)
| check | method | A | B | C |
| --- | --- | --- | --- | --- |
| A∩B / A∩C / B∩C | coord | 0 | 0 | 0 |
| probe-code-heldout ∩ slice | coord | 0 | 0 | 0 |
| a1-common-neutral ∩ slice | **content** | 0 | 0 | 0 |

## The a1 catch — why proofs-first mattered (owner emphasis)
a1-common-neutral is NOT common-corpus: it draws from the SAME base corpora (fineweb/rpj/pile/starcoder
quotas) but with an INCOMPATIBLE coordinate system (its `corpus` code0 has maxshard 149 vs fineweb's
99 — a different code ordering + shard enumeration). So a1 CANNOT be coord-excluded, and a
(corpus,shard,row) intersection against it is meaningless.

Content-equality exclusion (exact void-row match, coordinate-system-independent) was the right — and
only — fix. It removed **19,435 rows** whose embedding content the fresh slice draws would otherwise
have shared with a1 (fineweb 4,978 / redpajama 5,224 / pile 9,233; starcoder 0 — a1's code quota was
already held-out). Without it, ~19k a1 EVAL rows would have been in-sample TRAINING rows across all
three heads — invisible contamination under a mismatched coordinate system. This is exactly the
"bitten us repeatedly" hazard the owner demanded proofs-first to catch; it is now structurally closed.

## Trains (pplan_drawuniv.sh) — queued behind P1.6
Each slice: build k15 truth (knn+fuzzy) → train champion-bs16k (dose4, rankneg 500k=25% of 2M), SEED 42.
Validity gate (asserted at score time): the three `init_state_sha256` MUST be equal — same seed →
bit-identical init → any FFR/geometry difference is the data draw alone. Full champion, full horizon
(the A2 lesson: no proxy horizons).

## Evaluation (p_drawuniv_score.py) — rotation
- Shared eval: a1-common-neutral (truth exists), projected through all three heads.
- Rotation: each head evaluated on the slices it did NOT train on (A on B,C; B on A,C; C on A,B), scored
  on each eval slice's own k15 truth.
- Readouts: (1) FFR SPREAD across heads on identical truth = quality draw-variance; (2) procrustes-
  aligned per-point deviation + spread-ratio = geometric agreement; (3) cross-vs-self member advantage
  (head-A on slice-B truth [unseen] vs head-B on slice-B truth [member]).
- Context table: draw-variance side-by-side with P1.5 seed-variance (same substrate, seeds 42/43).

## RESULT (draw-univ-score.json, 2026-08-30)
Validity gate PASSES: all three inits bit-identical (`cb6fb9a944e4ec3b`) → every difference is the data draw.
- **Data-draw variance < seed variance**: a1 FFR range across A/B/C = **0.0037** (std 0.0017) vs baseline
  @42-vs-@43 = **0.0080**. Swapping the entire disjoint 2M substrate moves quality ~½ as much as
  changing the seed. At 2M composition-matched the substrate draw is essentially a NON-lever.
- **Geometry** (procrustes on a1, 250k pts): mean per-point deviation ~0.0012 across ALL pairs
  (draw pairs AND the seed pair) — geometrically draw ≈ seed ≈ tiny; spread-ratio 1.03–1.10.
- **Member advantage** (self in-sample − cross unseen): +0.062 / +0.069 / +0.069 (A/B/C). Each head
  scores ~0.50 on its own slice, ~0.44 on any disjoint slice; cross-FFRs tight (0.434–0.441) =
  draw-universal generalization.
- Load-bearing: the a1 content-exclusion (19,435 rows) kept those out of the cross (unseen) numbers;
  otherwise they'd be in-sample members corrupting the member-advantage measure.
Takeaways: (1) a single 2M draw is trustworthy (substrate-draw is not a quality lever); (2) ~0.065
in-sample optimism at 2M; (3) seed is the larger (still small, 0.008) variance axis — matches P1.5's
two-seed robustness framing.
