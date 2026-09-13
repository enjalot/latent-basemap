# Card084 frozen implementation for root release review

Runner UUID01a09ad8-aec2-73c1-b615-ec89835b1c01. Root accepted implementation;
GPU release remains absent and required. No device work was performed.

Frozen: quadratic and pseudo_huber, original023 tensors, both-on081uniform,
60,000 successful positive-LR1e-4 updates, snapshots20/40/60K,
delta0.9139534189451107. Eight actual production batches0–7 at seed42,
fixed original head, c=0.1/median(A/G). Any nonfinite/nonpositive norm or invalid
coefficient, or max/min raw A/G>10, STOP. No omission, sweep or retuning.
All eight initial raw-gradient cosines, baseline and combined norms/clipping
indicators are recorded; no claim of matching Adam updates. Root owns scoring:
two97.5% AUC intervals >=.005 versus081uniform, all083 original/uniform guards;
direct H-Q contrast descriptive. This runtime has no scorer.

## Implemented paths

- bounded_attraction.py and core.py: default-off stateless production-pair hook;
  finite per-positive values and extra scalar checked; extra coordinate backward
  checked; enabled combined loss and parameter-gradient nonfinites STOP before
  any optimizer update, including AMP overflows. This is deliberately stricter
  than inherited AMP retry behavior. Coefficient0 bypasses all new loss checks.
- card084_calibration.py and gpu_card084_calibrate.py: exact production callback
  before backward/optimizer, same frozen parent on all8 batches, actual graph
  positive rows verified against PERM, all endpoint global IDs exclude ref/val;
  ordered pair/target/radius/RNG-before/RNG-after artifacts hashed. Training
  restarts fresh seed42. No separately sampled training bank or extra forward.
- card084_common.py / fit.py / resume.py / run_card084_arm.py: bound family,
  coefficient, delta, calibration and batch hashes, training-panel bank SHA,
  dose, graph/radii/q/input/source/parent identity; reconstruct hook before fit
  on resume. Checkpoint equality rejects changes before configuration/updates.
  Generic inference model save does not carry a training hook; checkpoint and
  admission reconstruct it. Prepared head tensors equal original023 exactly.
- gpu_card084_graph_canary.py: actual production1536D/2048hidden/3D model,
  batch16384 and sampler on512-node synthetic graph using real training feature
  rows/radii. Exact model/Adam/scheduler/scaler/sampler/global+aux RNG, loop
  counters and hook reconstruction from both mid and real epoch checkpoints.
  Wrong family/coefficient/delta/calibration/source/bank/dose/batches rejected.
  Absent-hook versus coefficient0 full-state control, enabled divergence.
  This is not a full2M-epoch resume twin; full2M is the separate throughput test.
- gpu_card084_preflight.py: per-arm full2M500/3500-update fits and full checkpoint
  serialization; conservative slope/setup and epoch/snapshot reserve. Full60K
  admitted only if both estimates fit remaining cumulative caps and deadline.
- run_card084_chain.py / card084_budget.py: no release => no leases/device work;
  nonblocking flock bothleases, monitor global VRAM<30GiB, treeRSS<32GiB and
  available host RAM>4GiB. Enforce deadline23:50:55Z,4200s total and1800s per arm
  inclusive of attributable stages; shared calibration/canary/controller split
  equally. Reservations debit the sole rolling ledger before work, settle to
  measured elapsed afterwards; negative settlement entries are refunds of the
  reservation, not negative GPU usage. Durable journal replay is idempotent;
  if the controller crashes, unrefunded reservation remains charged, preventing
  unaccounted retries. Root can reconcile abandoned reservations explicitly.
  Both ledger writes use window-ledger-write.lock. No new budget ledger replaces
  the rolling ledger; card084-ledger.json is its card-specific accounting view.

Training features stay mmap-backed on host; existing DeviceArrayDataset casts
stored fp16 values to fp32 per upload chunk and returns them to GPUfp16 storage,
identical values to inherited full-host-fp32 path. No extra normalization. GPU
stage host cap32GiB; CPU preparation copies only the parent state and hashes
inputs. The8GiB proposed CPU diagnostic cap does not constrain a released GPU
stage's existing chunked upload; all stages use root's32GiB RSS bound.

## Root release contract and launch

Root creates O/card084-release.json with PASS=true, card="084",
runtime_sha=SHA256(card084-runtime-sha.json), immutable files mapping (absolute
path to SHA256), and exact limits:
{"card_gpu_s":4200,"per_arm_gpu_s":1800,"global_vram_gib":30,"rss_gib":32,"deadline":"2026-09-13T23:50:55Z"}.
Suggested immutable file map is O/card084-readiness/release-candidate.json;
this is a candidate, never a release. Root may add scorer/protocol files.
Do not add mutable output calibration/ledger files to the pre-calibration root
release; post-calibration SHA and eight batch contents bind every fit identity.
A failed calibration leaves durable batch artifacts and no PASS receipt; root
reviews before any retry. A valid calibration is reused by hash on chain restart.

Root launches the chain as a durable systemd unit with MemoryMax=32G,
KillMode=control-group and an overall deadline. The chain itself owns both
flocks; do not wrap it in a process already holding either lease:
/home/enjalot/code/latent-basemap/.venv/bin/python /data/latent-basemap/sandbox/card084-code/experiments/sandbox/run_card084_chain.py
It exports the exact root release SHA to GPU stages; direct stage execution
without that bound release environment fails. This document is not permission
to launch and the runner has not created a unit or root release.

Remaining device evidence:8-batch calibration, production-path canary, measured
full-dose admission and full training. CPU checks cannot certify GPU numerics,
throughput, memory peaks or resume twins. Any failure is durable STOP, not an
automatic alternate recipe, reduced dose or GPU reallocation.
