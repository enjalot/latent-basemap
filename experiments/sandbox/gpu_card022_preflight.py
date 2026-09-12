"""Card022 throughput preflight (per root prelaunch review §3 — this replaces 120-step setup-inclusive
timing, the Card009 cost-estimation failure class). Warm/load ONCE, then measure TWO windows (1000 + 3000 =
4000 positive updates) with the FULL production 20K bank and REAL checkpointing on the shape_floor arm (the
more expensive one). A 300K epoch is ~2747 steps, so the 3000-step window spans an epoch boundary + rank
rebuild + epoch checkpoint. A 2-point linear fit separates fixed setup (intercept) from steady per-step
(slope). Projects BOTH full 60K arms + charged prep against the 4500s cap (and window/deadline); missing or
malformed measured throughput STOPS (no fallback). Emits full-precision per_step for the chain. GPU; root runs
before either arm. Usage: gpu_card022_preflight.py
"""
import os, sys, json, math, time, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import card022_validate as V
import torch
from basemap.pumap.parametric_umap.core import ParametricUMAP

SB = V.SB; OC = V.OC; PARENT = V.PARENT; SUB = V.SUB; GRAPH = V.GRAPH; BANKD = V.BANKD
WIN = OC / "cards-24h-window-ledger.json"; CARD = OC / "card022-ledger.json"
SEED = V.SEED; RANKNEG = V.RANKNEG; DOSE = V.DOSE; BATCH = V.BATCH; POS_RATIO = V.POS_RATIO
W1, W2 = 1000, 3000; GPU_CAP = 4500; WIN_CAP = 86400
EPOCH_CKPT_WRITE_S = 3.0; STEP_CKPT_WRITE_S = 3.0; ENDPOINT_OVERHEAD_S = 60.0; SLACK_PER_ARM_S = 120.0
DEADLINE = dt.datetime.fromisoformat("2026-09-13T01:52:44+00:00").timestamp()
_X = None; _WARM = None; _BANK = None


def _preload():
    global _X, _WARM, _BANK
    t = time.monotonic()
    _X = np.asarray(np.load(SUB, mmap_mode="r"), np.float32); _ = float(_X[0, 0]) + float(_X[-1, -1])
    _WARM = torch.load(str(PARENT), map_location="cpu", weights_only=False)["model_state_dict"]
    _BANK = {"X": np.asarray(np.load(BANKD / "X.npy", mmap_mode="r"), np.float16), "tau": np.asarray(np.load(BANKD / "tau.npy"), np.float32)}
    return time.monotonic() - t


def _measured_fit(steps, ckpt_dir):
    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed_all(SEED)
    p = ParametricUMAP.load(str(PARENT), device="cuda"); p.model = None
    p.learning_rate = V.LR; p.lr_schedule = "constant"; p.batch_size = BATCH; p.warmup_steps = 0
    p.n_epochs = 100000; p.rankneg_window = RANKNEG; p._max_train_steps = steps
    p.x_residency = "auto"; p.required_input_pipeline = "device"
    for a, v in (("anchor_ids_path", ""), ("anchor_hold_weight", 0.0), ("replay_bank_path", ""),
                 ("replay_weight", 0.0), ("deriv_bank_path", ""), ("deriv_weight", 0.0)):
        if hasattr(p, a): setattr(p, a, v)
    p._shape_bank = {"X": _BANK["X"], "tau": _BANK["tau"]}; p._shape_weight = 1.0
    p._shape_eps = float(V.EPSILON); p._shape_centers_per_step = V.CENTERS_PER_STEP
    p._checkpoint_step_targets = {min(steps, 500)}                # exercise a real step-ckpt write in-window
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
    except Exception as e: raise RuntimeError(f"malformed/missing ledger {path}: {e!r} — fail closed")
    assert math.isfinite(v) and v >= 0, f"ledger {path}[{key}] not finite/nonnegative"; return v


def main():
    import tempfile
    t_pf0 = time.monotonic(); prep = _preload()
    with tempfile.TemporaryDirectory(dir=str(SB)) as td:
        w1, pk1, gu1, total = _measured_fit(W1, Path(td) / "m1")
        w2, pk2, gu2, _ = _measured_fit(W2, Path(td) / "m2")
    assert math.isfinite(w1) and math.isfinite(w2) and w2 > w1 > 0, "non-finite/degenerate windows — stop"
    per_step = (w2 - w1) / (W2 - W1); setup = w1 - per_step * W1
    assert math.isfinite(per_step) and per_step > 0, "non-finite/nonpositive per-step — stop"
    itps = 1.0 / per_step
    steps_per_epoch = math.ceil((V.N * 15) / (BATCH * POS_RATIO)); n_epochs = math.ceil(DOSE / steps_per_epoch)
    per_arm = setup + per_step * DOSE + n_epochs * EPOCH_CKPT_WRITE_S + len(V.STEP_CKPTS) * STEP_CKPT_WRITE_S + ENDPOINT_OVERHEAD_S + SLACK_PER_ARM_S

    card_spent = _spent(CARD, "batch_spent_s") if CARD.exists() else 0.0     # calib + canary already charged
    win_spent = _spent(WIN, "spent_s") if WIN.exists() else 0.0
    preflight_wall = time.monotonic() - t_pf0
    cap_room = GPU_CAP - (card_spent + preflight_wall)
    win_room = WIN_CAP - (win_spent + preflight_wall); deadline_room = DEADLINE - time.time()
    two_arms = 2 * per_arm; peak_global = round(max(gu1, gu2), 3)
    checks = {"windows_valid": bool(w2 > w1 > 0), "rate_finite_positive": bool(math.isfinite(itps) and itps > 0),
              "both_arms_fit_cap": bool(two_arms <= cap_room), "both_arms_fit_window": bool(two_arms <= win_room),
              "both_arms_fit_deadline": bool((two_arms + 60) <= deadline_room), "global_vram_lt_30gb": bool(peak_global < 30.0)}
    R = {"schema": "card022-preflight-2026-09-12", "at": dt.datetime.now(dt.timezone.utc).isoformat(),
         "windows": {str(W1): w1, str(W2): w2}, "preload_wall_s": prep, "per_step_s": per_step, "setup_s": setup,
         "it_per_s": itps, "dose": DOSE, "steps_per_epoch_est": steps_per_epoch, "n_epochs_est": n_epochs,
         "per_arm_estimate_s": per_arm, "two_arms_s": two_arms, "gpu_cap_s": GPU_CAP, "card_spent_s": card_spent,
         "cap_room_s": cap_room, "window_room_s": win_room, "deadline_room_s": deadline_room,
         "proc_peak_vram_gb": round(max(pk1, pk2), 3), "global_vram_used_gb": peak_global, "gpu_total_gb": round(total, 3),
         "preflight_wall_s": preflight_wall, "checks": checks,
         "reserves_note": "per_step (windowed, checkpointing ON, 3000-step window spans a 300K epoch) already "
                          "carries steady rank-rebuild + one in-window ckpt; explicit reserves add the remaining "
                          "epoch/step checkpoint writes + endpoint. No fallback throughput; missing/degenerate STOPS.",
         "decision": "Both 60K arms + charged prep must fit 4500s cap/window/deadline. On a miss admission STOPS with NO dose truncation.",
         "PASS": bool(all(checks.values()))}
    (OC / "card022-preflight.json").write_text(json.dumps(R, indent=2)); print(json.dumps({k: R[k] for k in ("per_step_s", "per_arm_estimate_s", "two_arms_s", "cap_room_s", "PASS")}, indent=1), flush=True)
    return 0 if R["PASS"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
