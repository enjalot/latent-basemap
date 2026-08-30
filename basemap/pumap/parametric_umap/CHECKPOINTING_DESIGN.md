# Resumable checkpointing design (#11, owner-driven, 2026-08-30)

Opt-in, default-off, periodic. Acceptance bar (sharpened by resident-floor=0 determinism):
**interrupt a short SEEDED run at a checkpoint, resume, and assert the final
`trained_state_sha256` is BITWISE-IDENTICAL to the uninterrupted twin.** Resume must be invisible
to the hashes. Hard gate before any >6h-class run (30M / seeded 6.25M flagship).

## Mutable-state inventory (core.py fit(), the main UMAP phase ~1737–2319)
A bitwise-identical resume must checkpoint AND restore EVERY stream the loop reads:
- **model** — `self.model.state_dict()`.
- **optimizer** — `AdamW.state_dict()` (moment buffers + step counts).
- **scheduler** — cosine LambdaLR; its `.state_dict()` OR the exact drivers `st["scheduler_steps"]`
  + `lr_horizon` + `warmup_steps` (the lambda is a pure fn of scheduler_steps).
- **AMP scaler** — `GradScaler.state_dict()` (scale + growth tracker); None on bf16/cpu.
- **torch GLOBAL RNG** — `torch.get_rng_state()` (CPU) AND `torch.cuda.get_rng_state_all()` — the
  model forward + any torch.* draws consume this.
- **aux generators (×3, each numpy RandomState AND a device torch.Generator on the fast path):**
  - mid-near: `mn_rng` (np seed rs+104729) + `mn_gen` (torch, same).
  - density: `dens_rng` (rs+60013) + `dens_gen`.
  - anchor-hold: `hold_rng` (rs+92821) + `hold_gen`.
  numpy: `.get_state()`; torch.Generator: `.get_state()`.
- **sampler / DataLoader RNG** — THE crux. edge_list_dataset / DeviceEdgeSampler draws the per-batch
  edges + on-the-fly negatives. Must capture its generator state (see Open Q1).
- **loop position** — `epoch`, `global_step`, and the full `st` (=self._train_stats) counter dict
  (optimizer_steps_attempted/succeeded, positive_lr_optimizer_steps, scheduler_steps, executed_iters).
- **stop driver** — the loop ends on `st["scheduler_steps"] >= lr_horizon`; resume must continue
  counting from the restored scheduler_steps, not restart.

## Strategy: EPOCH-BOUNDARY checkpoints (not mid-epoch)
The loop is `for epoch in range(n_epochs): for batch in loader:`. Checkpoint at the TOP of the epoch
loop (before consuming any batch of that epoch), every `checkpoint_every_epochs` (or wall-clock). This
sidesteps mid-epoch batch-skipping: resume reconstructs the loader with the restored sampler generator
and runs `for epoch in range(resumed_epoch, n_epochs)`. For a >6h run, ~30-min epoch-granularity is
ample. (Mid-epoch resume is a later refinement; epoch-boundary meets the gate.)

## Checkpoint format
`<out>/ckpt-epoch{e}.pt` (torch.save), atomic write (tmp+rename). Fields = the inventory above +
`init_state_sha256` (provenance) + config fingerprint (assert same recipe on resume). Keep last K
(default 2) + delete older. `resume_from=<path>` kwarg (or auto-detect newest ckpt in out dir when
`resume=True`). default-off: `checkpoint_every_epochs=0` disables entirely.

## Resume flow (fit())
1. If resuming: load ckpt BEFORE building optimizer/scheduler/gens; set a `_resume` state object.
2. Build model/optimizer/scheduler/scaler/gens as normal, THEN overwrite each from the ckpt
   (load_state_dict / set_state). Restore torch global + cuda RNG last (after any construction-time
   draws). Restore `st`, `global_step`, and start the epoch loop at `resumed_epoch`.
3. The init hash is NOT recomputed on resume (init already happened pre-interrupt); carry it forward.

## Acceptance test (the gate)
`test_checkpoint_resume_bitwise.py`: a SHORT seeded run (e.g. minilm 2M champion, H≈2000 steps, or a
tiny fixture) with `checkpoint_every_epochs=1`. (a) Run uninterrupted → `trained_state_sha256` = TW.
(b) Run with a hard stop after epoch k (SIGTERM-style / raise), then resume from ckpt-epoch{k} →
`trained_state_sha256` = TR. ASSERT TR == TW (bitwise). Also assert intermediate: the resumed model's
params immediately after load == the twin's params at the same epoch boundary. Gate FAILS if any RNG
stream was missed (the hash diverges) — that's the point.

## Open questions to resolve at implementation
- **Q1 sampler RNG**: does DeviceEdgeSampler / edge_list_dataset reshuffle per epoch from a torch
  Generator whose state advances (→ must capture+restore state) or re-seed per epoch from a fixed
  seed (→ only epoch index needed)? Read datasets/edge_list_dataset.py + DeviceEdgeSampler. This
  determines whether epoch-boundary restore is sufficient. (host_int8/device_int8 samplers too.)
- **Q2 cudnn.benchmark=True** (line 1723): algo autotuning. Resident determinism at floor=0 held WITH
  benchmark on, so algo selection is stable given fixed shapes → resume (same shapes) is fine. Verify
  in the acceptance test; if it flakes, set benchmark=False under checkpointing.
- **Q3 DataLoader workers**: if num_workers>0, per-worker RNG must be captured. Prefer num_workers=0
  under checkpointing (the device path is already in-process) to keep RNG in the main process.
- **Q4 anchored-pretrain phase** (_anchored_pretrain, before the main loop): if a run checkpoints, it
  is always past pretrain (pretrain is short + pre-main); checkpoints only cover the main phase, and
  resume skips pretrain (its output is in the restored model). Assert resume never re-runs pretrain.

Implementation order (delegate): after draw-univ + 4M-seed43, BEFORE the seeded arch pair
(checkpointing gates the flagship >6h run that arch would trigger).
