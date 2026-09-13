# Card084 — prospective bounded-attraction readiness, CPU only

Status: READY_FOR_ROOT_PROTOCOL_REVIEW, not GPU admitted. Base code 909e291;
branch card084-bounded-attraction. Runner 01a09ad8-aec2-73c1-b615-ec89835b1c01.
Root freezes arms, calibration, quality gates, release and scoring. No training
or GPU calibration was performed. No edits to Card081–083 runtimes or logs.

## Mechanism and comparison

Propose two new arms, pseudo-Huber and quadratic extra attraction, each from the
original Card023 half-scale 2M 3D head, with the Card081 uniform both-on recipe
(fneg_weight=1, neg_tanh_gamma=4, inverse-CDF uniform negatives, binary positives,
base negative weight=1, seed42, batch16384, positive fraction .1, LR1e-4,
weight decay .01, clipping1). Propose matched 60K successful positive-LR updates
and 20/40/60K durable snapshots; root sets final dose. Compare both against
081uniform and each other; no Card083 endpoint initialization or selection.
A coefficient-zero replay canary is required before admission, not a third sweep.

For d=y_i-y_j, s=sqrt(r_i*r_j), zeta=||d||/s and u=zeta²:
Q=u/2; H=delta²(sqrt(1+u/delta²)-1)=u/(sqrt(1+u/delta²)+1).
Loss addition is c times the arithmetic mean over sampled positive graph edges,
not all pairs and not weighted by the negative band normalization. Repeated
positive edges retain production multiplicity. Radii and positive mask detach;
both student endpoints receive gradients. No teacher network or teacher gradient.
Compute in fp32 after AMP forwards (double retained for CPU tests), before clipping.

Per-edge gradients with respect to y_i are d/s² for Q and
 d/[s² sqrt(1+u/delta²)] for H; y_j gets the opposite. At d=0 both
are exactly zero with finite derivative; near zero H agrees to leading order
with Q. Effective-distance force is zeta for Q, and
zeta/sqrt(1+(zeta/delta)²) for H, approaching delta at infinity.
The native-coordinate H bound is delta/s per edge, NOT a radius-independent
bound, nor a bound on shared neural gradients. The loss itself is unbounded.
The rationalized expression avoids cancellation and sqrt(u) zero gradients.
Nonfinite arithmetic at distances beyond finite precision remains a STOP,
not a clamping or silent nan-to-num policy for this new term.

The paper's appendix A.6 equation26 adds beta*||d||³/3, not a quadratic
penalty. This pilot tests a different, bounded force family; it does not
replicate that experiment or transfer its coefficient. Direct-coordinate
contraction theory does not imply Adam-head contraction or semantic quality.
Source: https://arxiv.org/html/2503.09101v3#A6
Local review: /home/enjalot/code/latent-labs/research/markdowns/0220-shape-of-attraction-umap.md.

## Delta and fixed-head calibration plan

Propose delta=0.9139534189451107, the linear-interpolated training-only p90 of
all 5,000 edges in the existing audited panel (seed83083). Panel SHA256
cdbcbadeaeb4c3a08dc737ddf430cb981523541e3da9d561bdd96825c0003ffe;
original head SHA256 cde4a26b385eea214270aca80633e49a54a9d90efa25bf282562b52f016bd6b9.
Use the receipt's input/graph/radii hashes and independent exclusion audit.
Do not tune delta on query results or recompute from later heads. No new bank.

Prespecify eight calibration batches: batches indexed0–7 from a fresh production
Card081 uniform loader initialized at seed42, batch16384, exact current
positive count1638, same graph ordering and conditional inverse-CDF negatives.
Capture and hash ordered source/destination ids, targets, detached radius
products and loader RNG state before and after each batch before gradient work.
This is a calibration-only replay using training rows; restore fresh seed42
for training. Never advance the training loader to obtain calibration batches.
Assert all positive pairs are graph edges and all endpoint global IDs exclude
ref/validation IDs using the existing substrate mapping/exclusion instrument.
Keep fp16 stored feature values cast to fp32, with no extra normalization.
Do not allocate a full CPU float32 2M feature table; mmap/gather one batch,
threads2, peak RSS cap8GiB for this preparation (ordinary admitted jobs32GiB).

At the FIXED ORIGINAL head (no optimizer updates, no head preparation), for each
batch b compute the exact baseline pairwise loss including both negative
components, using the production forward/precision and weighted denominator.
Use autograd.grad over every trainable model parameter to obtain the global
L2 norm G_b. Compute separate unit-coefficient positive-only extra-loss norms
A_Qb and A_Hb on the SAME forward and pair batch. Exclude weight decay, AMP
scaling and clipping from these norms; unscale if needed. Norm accumulation
in float64. Any nonfinite/nonpositive G or A or coefficient => STOP.
Set one coefficient per family: c_F = 0.1 / median_b(A_Fb/G_b).
Thus the median of eight added/baseline global-gradient ratios is exactly10%;
individual batch ratios are not forced equal. Save all eight raw norms,
ratios, coefficients, min/median/max/IQR, gradient cosine with baseline, and
combined-gradient norms/clipping incidence. Do not retune to equalize Adam
steps or later norms. Matching is an initial raw-gradient scale comparison,
not an equal trajectory or equal effective-update claim. Root decides execution
of GPU calibration and whether spread requires STOP rather than arbitrary tuning.
No numerical c has been selected in this CPU readiness task.

## Minimal hook, identity and required admission work

core.py calls add_attraction after baseline loss only when the optional
_card084_attraction dictionary exists and its coefficient is nonzero. Reuse
src_embeddings/dst_embeddings, binary-positive targets and existing detached
_pair_scale. No additional forward, sampled edges, RNG draws or mutable counters.
The helper returns the identical base Tensor before even validating inputs when
c=0. Existing runtime default is absent; production off-path equivalence still
needs a full canary, including identical states, losses, RNG and optimizer steps.
The enabled helper uses small device validation synchronizations; measured
full-dose preflight must include this overhead, rather than claiming zero cost.

Before release implement Card084-specific configuration/admission/resume wrappers
binding family, exact coefficient/delta, positive-mean normalization, calibration
receipt and batch hashes, original head/warm-state hashes, input/graph/radii/
uniform-q hashes, source manifest and loaded-module hashes, precision, dose,
seed and all baseline hyperparameters. Reconstruct the stateless hook explicitly
from that bound identity on resume; reject missing or mismatched identity.
The hook is NOT automatically serialized in current generic model save/load.
Do not use inherited Card081–083 entrypoints or source manifests for Card084.
A resumed-vs-uninterrupted canary, default-off equivalence canary, measured
full-dose preflight and refreshed immutable runtime manifest remain required.
Root owns GPU release, lease/rolling-ledger accounting and frozen scoring.

## Prospective guards and interpretable claims

Retain root's unchanged multi-aspect metric bundle and original/uniform guards;
recommend the existing sparse-tail recall and radius-retention gates as mandatory,
with simultaneous uncertainty handling for two attraction comparisons. Root must
freeze exact thresholds and endpoint before results. AUC improvement alone is
insufficient; do not relax the .005 meaningful-effect gate because nearby prior
results were smaller. Report the paired H-minus-Q comparison even if neither
beats uniform. Single-seed results remain development evidence; query bootstraps
are conditional on trained heads, not seed replication.

On the fixed training panel record longest1%/10% force shares, stratifying by
radius/sparse tail, at the admitted snapshots without selection. Current native
coordinate shares: Q22.74%/85.90%, H5.07%/48.93%, using identical distance tails.
This establishes force-profile concentration only. It cannot establish that
long edges are errors, that H improves sparse recall, or that Adam contracts
pairs. Testable pilot claim: at matched initial neural-gradient scale, limiting
far-edge effective-distance force changes quality and sparse-tail retention
relative to quadratic stronger attraction under the same recipe. Any claim
about actual pair contraction requires measured before/after distances; any
claim of seed robustness requires independent trained seeds.

## CPU validation and review artifacts

experiments/sandbox/test_card084_cpu.py: fp32/fp64 analytic coordinate gradients
at zero,1e-10,1,1e6; finite loss/gradients; negative exclusion; detached radius
teacher inputs; double finite-difference gradcheck including coincident points;
translation/rotation value invariance and gradient covariance; coefficient-zero
object/RNG identity; stateless repeated result. All20 named checks pass.
These are mathematical/unit contracts, not a production resume or GPU audit.
Artifacts: /data/latent-basemap/sandbox/overseer-codex/card084-readiness/
(cpu-tests.json, panel.json, receipt.json). No scoring bundle is forked.
