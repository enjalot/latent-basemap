"""Card033 two-point full4M device timing:1000 and4000 successful updates, each with a real mid-fit checkpoint.
Slope separates setup from steady updates; setup and explicit epoch/checkpoint/endpoint reserves are added
back for both800K production arms. GPU caches are released between trials so resident admission sees actual
available memory. Every measured/preparation second remains charged by the chain. No dose truncation.
"""
import os, sys, json, math, time, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import card033_validate as V
import torch
from basemap.pumap.parametric_umap.core import ParametricUMAP

SB = V.SB; OC = V.OC; DATA = V.DATA
CHAMPION = SB / "dino-arrival-t0/champion-bs16k/model.pt"; INIT3D = V.INIT3D
WIN = OC / "cards-24h-window-ledger.json"; CARD = OC / "card033-ledger.json"
SEED = V.SEED; RANKNEG = V.RANKNEG; DOSE = V.DOSE; BATCH = V.BATCH; POS_RATIO = V.POS_RATIO
STEPS = 4000; STEP_CKPT_AT = 2000
GPU_CAP = 32400; PER_ARM_CAP = 14500
EPOCH_CKPT_WRITE_S = 5.0; STEP_CKPT_WRITE_S = 5.0; ENDPOINT_OVERHEAD_S = 240.0; SLACK_PER_ARM_S = 300.0
DEADLINE = dt.datetime.fromisoformat("2026-09-13T01:52:44+00:00").timestamp()
_X = None


def _preload():
    global _X
    t = time.monotonic(); _X = np.asarray(np.load(DATA / "substrate.f16.npy", mmap_mode="r"), np.float32)
    _ = float(_X[0, 0]) + float(_X[-1, -1])     # force materialization
    return time.monotonic() - t


def _measured_fit(ckpt_dir):
    import gc;gc.collect();torch.cuda.empty_cache()
    radii = np.load(DATA / "r_actual.npy").astype(np.float32)
    init = torch.load(str(INIT3D), map_location="cpu", weights_only=False)["model_state"]
    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed_all(SEED)
    p = ParametricUMAP.load(str(CHAMPION), device="cuda"); p.model = None; p.n_components = 3
    assert p.fneg_weight==1.0 and p.neg_tanh_gamma==4.0 and p.positive_target_mode=="binary"
    p.learning_rate = 0.001; p.lr_schedule = "constant"; p.batch_size = BATCH; p.warmup_steps = 0
    p.n_epochs = 100000; p.rankneg_window = RANKNEG; p._max_train_steps = STEPS
    p.gpu_resident_vram_budget_gb = 14.0
    p.x_residency = "auto"; p.required_input_pipeline = "device"
    for a, v in (("anchor_ids_path", ""), ("anchor_hold_weight", 0.0), ("replay_bank_path", ""),
                 ("replay_weight", 0.0), ("deriv_bank_path", ""), ("deriv_weight", 0.0)):
        if hasattr(p, a): setattr(p, a, v)
    p._card013_radii = radii; p._checkpoint_step_targets = {STEP_CKPT_AT}
    free0, total = torch.cuda.mem_get_info(); torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); t0 = time.monotonic()
    p.fit(_X, precomputed_edges_path=str(DATA / "edges-fixed15.npz"), random_state=SEED, verbose=False,
          warm_start_state=init, checkpoint_every_epochs=1, checkpoint_dir=str(ckpt_dir))
    torch.cuda.synchronize(); wall = time.monotonic() - t0
    assert p._train_stats["positive_lr_optimizer_steps"] == STEPS
    assert (getattr(p, "_pipeline_info", {}) or {}).get("x_residency") == "device_fp16", "pipeline not device_fp16"
    proc_peak = torch.cuda.max_memory_allocated() / 2**30
    free1, _ = torch.cuda.mem_get_info(); global_used = (total - free1) / 2**30
    # observe (not claim-epoch) the mid-fit step checkpoint write
    ck = ckpt_dir / f"ckpt-step{STEP_CKPT_AT}.pt"; ck_ok = ck.exists()
    if ck_ok:
        o = torch.load(ck, map_location="cpu", weights_only=False)
        ck_ok = int(o.get("global_step", -1)) == STEP_CKPT_AT and all(bool(torch.isfinite(t).all()) for t in o["model"].values())
    return wall, proc_peak, global_used, ck_ok, (total / 2**30)


def _ledger_spent(path, key):
    try:
        v = float(json.loads(Path(path).read_text())[key])
    except Exception as e:
        raise RuntimeError(f"malformed/missing ledger {path}: {e!r} — fail closed")
    assert math.isfinite(v) and v >= 0, f"ledger {path}[{key}] not finite/nonnegative"
    return v


def main():
    global STEPS, STEP_CKPT_AT
    t_pf0 = time.monotonic()
    free,total=torch.cuda.mem_get_info();assert (total-free)/2**30<12,"leave18GiB GPU headroom"
    prep_wall = _preload()     # cold24GB host alloc OUTSIDE the timed fit (still on the GPU-charged clock, recorded)
    import tempfile
    with tempfile.TemporaryDirectory(dir=str(SB)) as td:
        first=Path(td)/"first";first.mkdir();second=Path(td)/"second";second.mkdir()
        STEPS=1000;STEP_CKPT_AT=500
        w1,p1,g1,k1,t1=_measured_fit(first)
        STEPS=4000;STEP_CKPT_AT=2000
        fit_wall,proc_peak,global_used,ck_ok,total_gb=_measured_fit(second)
    assert k1 and ck_ok, "both real step checkpoints required"
    per_step=(fit_wall-w1)/3000
    assert math.isfinite(per_step) and per_step>0, "invalid two-point slope"
    setup=max(0.,w1-1000*per_step)
    it_per_s=1./per_step
    proc_peak=max(proc_peak,p1);global_used=max(global_used,g1)

    steps_per_epoch = math.ceil(V.N * 15 / int(BATCH * POS_RATIO))  # 4M×15 directed edges, 1638 pos/batch ≈36,631
    n_epochs = math.ceil(DOSE / steps_per_epoch)
    # Add setup and explicit epoch work after estimating the steady-state slope.
    epoch_ckpt_reserve = n_epochs * EPOCH_CKPT_WRITE_S
    epoch_rank_reserve = n_epochs * 10.0
    step_ckpt_reserve = len(V.STEP_CKPTS) * STEP_CKPT_WRITE_S
    per_arm_est = per_step * DOSE + setup + epoch_rank_reserve + epoch_ckpt_reserve + step_ckpt_reserve + ENDPOINT_OVERHEAD_S + SLACK_PER_ARM_S

    card_spent = _ledger_spent(CARD, "batch_spent_s")             # already includes the actual canary
    preflight_wall = time.monotonic() - t_pf0
    effective_spent = card_spent + preflight_wall                 # no double-charged canary reserve
    cap_room = GPU_CAP - effective_spent
    deadline_room = DEADLINE - time.time()
    window_room = 86400 - _ledger_spent(WIN,"spent_s") - preflight_wall
    two_arms = 2 * per_arm_est
    checks = {"step_ckpt_observed": bool(ck_ok),
              "per_arm_le_cap": per_arm_est <= PER_ARM_CAP,
              "fits_gpu_cap": two_arms <= cap_room,
              "fits_deadline": (two_arms + 60) <= deadline_room,
              "fits_window_ledger": (two_arms + 60) <= window_room,
              "global_vram_lt_30gb": global_used < 30.0,
              "rate_finite_positive": math.isfinite(it_per_s) and it_per_s > 0}
    R = {"schema": "card033-preflight-2026-09-12", "at": dt.datetime.now(dt.timezone.utc).isoformat(),
         "measure_steps": STEPS, "step_ckpt_at": STEP_CKPT_AT, "prep_preload_wall_s": prep_wall,
         "fit_wall_s": fit_wall, "first1000_wall_s":w1,"estimated_setup_s":setup,"per_step_s": per_step, "it_per_s": it_per_s, "dose": DOSE,
         "steps_per_epoch_est": steps_per_epoch, "n_epochs_est": n_epochs,
         "reserves_s": {"epoch_rank":epoch_rank_reserve,"setup":setup,"epoch_ckpt": epoch_ckpt_reserve, "step_ckpt": step_ckpt_reserve,
                        "endpoint": ENDPOINT_OVERHEAD_S, "slack": SLACK_PER_ARM_S,
                        "note": "Steady slope excludes initial setup, which is added back. Epoch ranks reserve10s each and checkpoint writes5s each; these are explicit allowances, not measured epoch timings."},
         "per_arm_estimate_s": per_arm_est, "per_arm_cap_s": PER_ARM_CAP, "two_arms_s": two_arms,
         "proc_peak_vram_gb": round(proc_peak, 3), "global_vram_used_gb": round(global_used, 3), "gpu_total_gb": round(total_gb, 3),
         "gpu_cap_s": GPU_CAP, "card_spent_s": card_spent, "preflight_wall_s": preflight_wall,
         "effective_spent_s": effective_spent, "cap_room_s": cap_room, "deadline_room_s": deadline_room,"window_room_s":window_room,
         "checks": checks,
         "decision": "Both 800K arms + their checkpoint/endpoint reserves must fit the remaining 32,400s cap and deadline. On a miss admission STOPS with NO dose truncation.",
         "PASS": bool(all(checks.values()))}
    (OC / "card033-preflight.json").write_text(json.dumps(R, indent=1)); print(json.dumps(R, indent=1), flush=True)
    return 0 if R["PASS"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
