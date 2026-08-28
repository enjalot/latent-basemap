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
