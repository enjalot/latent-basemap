# Seeding correction — image_map_pipeline unseeded init (4th external review, 2026-08-28)

## The bug
`experiments/sandbox/image_map_pipeline.py` NEVER called `torch.manual_seed` before the
model init (`ParametricUMAP._init_model`), unlike `knobs_2m.py:643` which does. So **every
arm trained on the image-runner started from a RANDOM weight init**, even at "fixed seed."
The `random_state` passed to `fit()` seeded the sampler/edge draws but NOT the MLP init.

Fixed in commit 6b8618b: `torch.manual_seed(arm_seed)+cuda` before construction/fit +
`init_state_sha256` recorded in every summary.json (verified reproducible: seed42 init hash
identical across runs, seed43 differs). All image-runner arms from 6b8618b forward are seeded.

## Consequence — the "0.0184 floor" was invalid
The determinism control compared `champion-bs16k-hostint8` vs `-rep2` at "seed 42" and got
Δ=0.0184, which I attributed to GPU/int8-path nondeterminism. But both runs had DIFFERENT
random inits, so 0.0184 **conflated init variance with training nondeterminism**. The true
same-seed floor is UNMEASURED until a seeded same-seed×2 repeat (queued).

## DEMOTED TO PROVISIONAL (were judged against an invalid floor / are within likely init variance)
- **device-int8 quality-parity "SEAL"** (deviceint8-parity.json): the 0.0025 vs the 0.0184
  floor is meaningless (floor invalid). device-int8 quality-parity is REOPENED, pending the
  controlled seeded floor + the P1 two-seed finalist rerun. (device-int8 THROUGHPUT 0.987 is
  UNAFFECTED — timing, not quality.)
- **"int8 path is non-deterministic / verify_repro is resident-only"** (determinism-control.json):
  UNSUPPORTED — the 0.0184 may be pure init variance. Attribution withdrawn.
- **jina bmix10 maximin win** (+0.001), **MiniLM bmix10cp win** (+0.0037), **bmix30 loss**
  (−0.0059), **A4 neck deltas** (−0.0008/−0.0045), **width-ladder increments** (+0.0128/+0.0048),
  **md010 delta** (−0.0069): all are image-runner effects SMALLER than a plausible init-variance
  band, so their SIGN is provisional pending seeded reruns. (The DIRECTION of the social ladder
  and the mechanism still hold at the group level — see surviving.)

## SURVIVING — unaffected by the init bug
- **Everything knobs_2m-side** (2m-knobs, 6250k-knobs, int8fac, dose/rankneg grids, the champion
  transfer scoreboard): knobs_2m seeds the init (`:643`) — these are valid.
- **Image-runner effects ≫ the plausible init-variance floor** (sign robust):
  - champion transfer scoreboard ordering; **exposure +0.044** (dose8 vs base); **A2 horizon
    +0.03→+0.07** (H320/H640/full); **ceiling monotone code-collapse** (code-heldout 0.134→0.108→
    0.099 across bmix30/40/50 — a >0.03 monotone trend, far above init variance); the
    **code/language-DISPLACEMENT mechanism** (bmix30/40/50 lose the maximin by growing margins;
    the displacement→small-register-collapse effect is group-level and large).
  - **gate-2 throughput** (device-int8 0.987, C1-fast 0.57−0.74, batch 0.83, legacy 0.64/0.77):
    timing, not quality — unaffected by init seeding.
  - **transform-gap "small head serves the atlas"** curve: the GAP is trained-at-N vs transform,
    both image-runner, so the ABSOLUTE gap carries an init-variance band, but the qualitative
    result (gap saturates, ~0.01−0.03 at 12.5−100M) is far larger than the floor at the big rungs
    and holds; the recipe-clean 2M→6.25M headline (+0.069/+0.093) also ≫ floor.

## Pending (P0/P1)
- Seeded same-seed×2 repeat → the TRUE floor (replaces 0.0184).
- P1: two-seed controlled reruns of the FINALISTS only (jina baseline-vs-bmix10, MiniLM
  baseline-vs-bmix10cp) with group-level gates + projection as a required non-regression gate.
  Only after those do the bmix10/bmix10cp "wins" and the device-int8 parity get re-sealed (or not).

---

## Update 2026-08-29 — P0.1 core init-hash hook + P0.3(b) full-disc transform gap

### init-hash hook (commit bb6e1e7) supersedes the post-fit-only fingerprint
`core.py._init_model` now hashes the freshly-initialized weights → `self.init_state_sha256`
(runs AFTER core's own init, so no L0.5 admission-guard conflict — the reason the earlier
external pre-call attempt failed). This is the TRUE pre-init hash: at fixed seed it is
reproducible (42==42=046ccfe2…) and seed-sensitive (43=b05fcba3…), so it isolates INIT
variance unambiguously — a property the post-fit `trained_state_sha256` cannot give (post-fit
varies with training nondeterminism even at fixed seed). Surfaced into both runners' summaries
+ the checkpoint save_dict. The resident-floor twins (D384/D768, same-seed×2) use init-match as
the seeding proof and trained-match as the zero-floor test; result in `floor-result.json`.

### P0.3(b) — the 2M-SLICE transform-gap instrument UNDERSTATED the gap 2-3× (commit 09e2daa)
The transform-gap audit's big-rung numbers came from a 2M random subsample with a FRESH
in-slice k15 truth. The full-disc fallback (score the head-transformed FULL rung against the
rung's SEALED full k15 truth at disc 0.1%×N) shows the slice was OPTIMISTIC:

| rung | slice gap | FULL-DISC gap | transform FFR | trained FFR | retention | unseen≈overall |
| ---- | --------- | ------------- | ------------- | ----------- | --------- | -------------- |
| 25M  | 0.0062    | **0.0194**    | 0.4720        | 0.4914      | 96.0%     | 0.4703 vs 0.4720 |
| 50M  | 0.0196    | **0.0380**    | 0.4839        | 0.5219      | 92.7%     | 0.4823 vs 0.4839 |

A fresh k15 within a 2M-of-N subsample keeps only ~4-8% of each row's true neighbors → a
sparser/easier "truth" that flatters both maps and compresses their difference. **Corrections:**
- The projector "small head serves the atlas" retention is ~93-96% (full-disc), NOT the
  slice-based ~98%. The product still works; the honest number is lower.
- member_frac tiny (8.2%/4.2%) and UNSEEN ≈ OVERALL at both rungs → the projection quality is
  genuine out-of-sample, not inflated by nested-prefix training members.
- gap/extrap ≈ constant (0.00155/0.00152) → gap scales ~linearly; 100M (50× extrap) ≈ 0.076
  gap / ~87% retention by LINEAR EXTRAPOLATION (100M full-disc infeasible — a caveat, not a
  measurement).
- SURVIVING line 45 above ("~0.01−0.03 at 12.5−100M ... holds") is REVISED: the true full-disc
  gaps are ~0.02−0.04 at 25−50M; the qualitative "gap stays small / saturates" still holds.

### Floor result (P0.1) + int8 re-seal — the invalid floors are RETIRED
`floor-result.json`: resident-D384 (e944eca4), resident-D768 (4bc231c1), host_int8-D768 (7d1b01c5),
device_int8-D768 (7d1b01c5) are ALL same-seed BITWISE-DETERMINISTIC (a==b) → every resident/int8
floor is ZERO. Cross device_int8 == host_int8 (7d1b01c5) BITWISE-IDENTICAL → device-int8 parity
re-seals exactly. The 0.0184 AND 0.0025 numbers were pure unseeded-init variance and are RETIRED.
device-int8 stays the 30M gate-2 answer (throughput 0.987 + bitwise-identical quality to host_int8).
The int8 trained hash (7d1b01c5) ≠ resident (4bc231c1) is the genuine deterministic int8 weight tax.

### P1.5 MiniLM finalist — the provisional bmix10cp "win" is RESOLVED: NO-ADOPT
Two-seed seeded reruns (p15-minilm-verdict.json), resident floor=0 so per-seed comparisons EXACT:
- OWN-MAP maximin: bmix10cp wins BOTH seeds (worst_delta +0.0071 s42 / +0.0025 s43, same sign;
  mean +0.0127/+0.0064; social registers +0.03−0.04; code-heldout holds → code-preserving worked).
- PROJECTION (6.25M projector maximin): REGRESSES both seeds (−0.0019 s42 / −0.0100 s43).
- VERDICT: NO-ADOPT (own-map win but projection non-regression GATE fails). Baseline stands. The
  earlier provisional "+0.0037 bmix10cp win" is superseded: it's an own-map effect that does NOT
  transfer to the atlas projector.

### P1.5 jina finalist (bmix10) — NO-ADOPT (double reason)
Two-seed seeded reruns (p15-jina-verdict.json), resident-D768 floor=0:
- maximin SPLIT SIGN: seed42 +0.0079 (bmix10 wins), seed43 −0.0001 (baseline wins) → seed-scale,
  no-adopt. AND projection regresses both seeds (proj_6250k −0.0135/−0.0141).
- Social registers gain robustly both seeds (reddit/twitter/bluesky/ca +0.01−0.03); EN base pays
  the displacement (fineweb/pile −0.005−0.012); languages small/mixed.
- VERDICT: NO-ADOPT. Undisplaced baseline stands for BOTH spaces (MiniLM + jina). The provisional
  jina "+0.001 bmix10 win" is superseded. Two-seed + projection-gate caught what a single-seed
  single-metric read would have "confirmed."

### P1.6 head-size (jina) — head SIZE matters; small heads don't serve the atlas
p16-headsize-results.json. Composition-matched heads (exact member/unseen masks) vs the SEEDED direct
6.25M reference (0.7054 — now the canonical jina-6m map, replaces unseeded 0.6686):
| head | proj FFR | unseen | collapse | fog | occ | retention | gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2M | 0.6172 | 0.5800 | 1.017 | 0.238 | 0.265 | 87.5% | FAIL |
| 4M | 0.6684 | 0.6455 | 1.010 | 0.112 | 0.209 | 94.8% | FAIL (near-boundary → seed-43) |
| direct | 0.7054 | — | 0.895 | 0.106 | 0.125 | ref | — |
Monotone in size on EVERY axis (FFR up, fog/collapse/occupancy down). Neither small head clears the
deploy gate (≤0.01 gap OR ≥97% retention, no group −0.005) — the atlas needs training at scale, not a
projected small head. 4M near-boundary → seed-43 replicate of the decisive cell (jina-4m-head/champion-s43).
4M SEED-43 REPLICATE (SEALED): 4M@42 gap 0.0369/ret 0.9477 vs 4M@43 gap 0.0371/ret 0.9474 — agree
within 0.0003, BOTH FAIL. The near-boundary verdict is seed-robust; 4M reproducibly misses the gate.

### P2 seeded arch pair (jina 2M) — h3072-neck625 vs h2048, capacity reallocation
p2-arch-jina-confirm-results.json (seed 42, both checkpointed, PCIe x16). h3072-neck625 (wider +
bottleneck) vs h2048:
- maximin (worst reg = cmn_Hani): 0.0763 → 0.0845 = **+0.0083** (h3072 WINS the maximin); mean **−0.003**.
- Coherent capacity REALLOCATION toward hard non-Latin scripts (jpn +0.019, cmn +0.008, hin +0.004),
  away from Latin (ind −0.021, ell −0.013, swe −0.010, pol −0.009, kor −0.007, deu −0.007). ±0.02 swings
  are above the ~0.008 seed band → the direction is real; a genuine maximin-vs-mean tradeoff.
- On the standing maximin criterion h3072 modestly confirms, but the NET maximin delta is at seed scale
  (cross-arch, single seed). FLAGSHIP: recommend a seed-43 arch replicate before committing the 17h
  x8-h3072 6.25M run (two-seed discipline, per P1.5/P1.6). Now checkpointable → 17h is safe.

### Draw universality (MiniLM 2M) — substrate-draw is not a quality lever
draw-univ-score.json: 3 disjoint composition-matched 2M draws, same seed (init bit-identical). a1 FFR
draw-variance 0.0037 (std 0.0017) < seed-variance 0.0080; procrustes mean-pointdev ~0.0012 (draw ≈ seed);
member advantage ~0.065. A single 2M draw is trustworthy. (Image-space sisap-CLIP replication built +
proven, queued.)
