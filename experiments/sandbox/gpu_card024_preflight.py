"""Card024 throughput preflight (per card024 "Measure throughput and admit all three full 60K arms within
remaining cap and deadline, no automatic dose truncation"). Warm/load ONCE, measure TWO windows (1000 + 3000
positive updates) with REAL checkpointing on the most expensive arm (nce_learned: logit loss + learned
scalar). A 300K epoch is ~2747 steps, so the 3000-step window spans an epoch boundary + rank rebuild + epoch
checkpoint. Conservative per_step = max(2-fit slope, long whole-fit average); setup >= 0. Projects ALL THREE
60K arms + charged prep against the 7200s shared cap (and window/deadline) and each arm's 1800s cumulative
per-arm cap; missing/degenerate throughput STOPS (no fallback, no dose truncation). GPU; root runs. Usage:
gpu_card024_preflight.py
"""
import os, sys, json, math, time, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import card024_validate as V
import torch
from basemap.pumap.parametric_umap.core import ParametricUMAP

SB = V.SB; OC = V.OC; CHAMPION = V.CHAMPION; SUB = V.SUB; GRAPH = V.GRAPH; INIT = V.INIT
WIN = OC / "cards-24h-window-ledger.json"; CARD = OC / "card024-ledger.json"
SEED = V.SEED; DOSE = V.DOSE; BATCH = V.BATCH; POS_RATIO = V.POS_RATIO
W1, W2 = 1000, 3000; GPU_CAP = 7200; PER_ARM_CAP = 1800; WIN_CAP = 86400; N_ARMS = 3
EPOCH_CKPT_WRITE_S = 3.0; STEP_CKPT_WRITE_S = 3.0; ENDPOINT_OVERHEAD_S = 60.0; SLACK_PER_ARM_S = 120.0
DEADLINE = dt.datetime.fromisoformat("2026-09-13T01:52:44+00:00").timestamp()
_X = None; _WARM = None


def _preload():
    global _X, _WARM
    t = time.monotonic()
    _X = np.asarray(np.load(SUB, mmap_mode="r"), np.float32); _ = float(_X[0, 0]) + float(_X[-1, -1])
    _WARM = V.check_init()
    return time.monotonic() - t


def _measured_fit(steps, ckpt_dir):
    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed_all(SEED)
    p = ParametricUMAP.load(str(CHAMPION), device="cuda"); p.model = None; p.n_components = V.NC
    p.learning_rate = V.LR; p.lr_schedule = "constant"; p.batch_size = BATCH; p.warmup_steps = 0
    p.n_epochs = 100000; p.rankneg_window = 0; p._max_train_steps = steps
    p.x_residency = "auto"; p.required_input_pipeline = "device"
    for a, v in (("anchor_ids_path", ""), ("anchor_hold_weight", 0.0), ("replay_bank_path", ""),
                 ("replay_weight", 0.0), ("deriv_bank_path", ""), ("deriv_weight", 0.0),
                 ("fneg_weight", 0.0), ("neg_tanh_gamma", 0.0), ("midnear_enabled", False),
                 ("density_weight", 0.0), ("correlation_weight", 0.0)):
        if hasattr(p, a): setattr(p, a, v)
    p._card024_mode = "nce_learned"; p._checkpoint_step_targets = {min(steps, 500)}
    _, total = torch.cuda.mem_get_info(); torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); t0 = time.monotonic()
    p.fit(_X, precomputed_edges_path=str(GRAPH), random_state=SEED, verbose=False, warm_start_state=_WARM,
          checkpoint_every_epochs=1, checkpoint_dir=str(ckpt_dir))
    torch.cuda.synchronize(); wall = time.monotonic() - t0
    assert p._train_stats["positive_lr_optimizer_steps"] == steps
    assert (getattr(p, "_pipeline_info", {}) or {}).get("x_residency") == "device_fp16", "pipeline not device_fp16"
    free1, _ = torch.cuda.mem_get_info(); return wall, torch.cuda.max_memory_allocated() / 2**30, (total - free1) / 2**30, total / 2**30


def _spent(path, key):
    try: v = float(json.loads(Path(path).read_text())[key])
    except Exception as e: raise RuntimeError(f"malformed ledger {path}: {e!r} — fail closed")
    assert math.isfinite(v) and v >= 0; return v


def main():
    import tempfile
    t_pf0 = time.monotonic(); prep = _preload()
    with tempfile.TemporaryDirectory(dir=str(SB)) as td:
        w1, pk1, gu1, total = _measured_fit(W1, Path(td) / "m1")
        w2, pk2, gu2, _ = _measured_fit(W2, Path(td) / "m2")
    assert math.isfinite(w1) and math.isfinite(w2) and w2 > w1 > 0, "non-finite/degenerate windows — stop"
    raw_slope = (w2 - w1) / (W2 - W1)
    per_step = max(raw_slope, w2 / W2); setup = max(0.0, w1 - per_step * W1)     # conservative; unequal cold setup guard
    assert math.isfinite(per_step) and per_step > 0, "non-finite/nonpositive per-step — stop"
    itps = 1.0 / per_step
    steps_per_epoch = math.ceil((V.N * 15) / (BATCH * POS_RATIO)); n_epochs = math.ceil(DOSE / steps_per_epoch)
    per_arm = setup + per_step * DOSE + n_epochs * EPOCH_CKPT_WRITE_S + len(V.STEP_CKPTS) * STEP_CKPT_WRITE_S + ENDPOINT_OVERHEAD_S + SLACK_PER_ARM_S

    card_spent = _spent(CARD, "batch_spent_s"); win_spent = _spent(WIN, "spent_s")
    preflight_wall = time.monotonic() - t_pf0
    cap_room = GPU_CAP - (card_spent + preflight_wall); win_room = WIN_CAP - (win_spent + preflight_wall)
    deadline_room = DEADLINE - time.time(); all_arms = N_ARMS * per_arm; peak_global = round(max(gu1, gu2), 3)
    checks = {"windows_valid": bool(w2 > w1 > 0), "rate_finite_positive": bool(math.isfinite(itps) and itps > 0),
              "per_arm_le_cap": bool(per_arm <= PER_ARM_CAP), "three_arms_fit_cap": bool(all_arms <= cap_room),
              "three_arms_fit_window": bool(all_arms <= win_room), "three_arms_fit_deadline": bool((all_arms + 60) <= deadline_room),
              "global_vram_lt_30gb": bool(peak_global < 30.0)}
    R = {"schema": "card024-preflight-2026-09-12", "at": dt.datetime.now(dt.timezone.utc).isoformat(),
         "windows": {str(W1): w1, str(W2): w2}, "preload_wall_s": prep, "raw_two_fit_slope_s": raw_slope,
         "per_step_s": per_step, "setup_s": setup, "it_per_s": itps, "dose": DOSE, "n_arms": N_ARMS,
         "steps_per_epoch_est": steps_per_epoch, "n_epochs_est": n_epochs, "per_arm_estimate_s": per_arm,
         "per_arm_cap_s": PER_ARM_CAP, "three_arms_s": all_arms, "gpu_cap_s": GPU_CAP, "card_spent_s": card_spent,
         "cap_room_s": cap_room, "window_room_s": win_room, "deadline_room_s": deadline_room,
         "proc_peak_vram_gb": round(max(pk1, pk2), 3), "global_vram_used_gb": peak_global, "gpu_total_gb": round(total, 3),
         "preflight_wall_s": preflight_wall, "checks": checks,
         "note": "Two independent fits; conservative per_step=max(slope, long whole-fit avg), setup>=0. "
                 "Checkpointing ON; 3000 steps span a 300K epoch. Measured on nce_learned (most expensive). "
                 "No fallback throughput; missing/degenerate STOPS; no dose truncation.",
         "PASS": bool(all(checks.values()))}
    (OC / "card024-preflight.json").write_text(json.dumps(R, indent=2))
    print(json.dumps({k: R[k] for k in ("per_step_s", "per_arm_estimate_s", "three_arms_s", "cap_room_s", "PASS")}, indent=1), flush=True)
    return 0 if R["PASS"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
