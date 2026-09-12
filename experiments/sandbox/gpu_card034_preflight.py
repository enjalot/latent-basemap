"""Card034 throughput preflight (per production review item 5). Root runs on GPU. Measures the REAL
production engine (card034_engine.GroupedEngine) — actual step math, actual GRAD_CLIP, exact frozen
coefficient, actual nonfinite handling, and a FULL checkpoint payload (model/Adam/scaler/scalar/sampler/RNG),
not a toy loop. Warm/load once (graph/PERM/substrate setup timed + charged conservatively). Two windows
(1000 + 3000 positive updates) on the FULL graph with real checkpointing; conservative per_step =
max(2-fit slope, long whole-fit average); setup >= 0. Projects ALL THREE 60K arms + charged prep against
7200s / 1800s-per-arm / window / deadline; missing/degenerate/overflow STOPS (no fallback, no dose truncation).
Usage: gpu_card034_preflight.py
"""
import os, sys, json, math, time, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import card034_validate as V
import card034_grouped as G
import card034_engine as E
import torch

SB = V.SB; OC = V.OC; CHAMPION = V.CHAMPION; SUB = V.SUB; GRAPH = V.GRAPH; INIT = V.INIT
ROOT = str(Path(__file__).resolve().parents[2])
CARD = OC / "card034-ledger.json"; WIN = OC / "cards-24h-window-ledger.json"
W1, W2 = 1000, 3000; GPU_CAP = 7200; PER_ARM_CAP = 1800; WIN_CAP = 86400; N_ARMS = 3
EPOCH_CKPT_WRITE_S = 3.0; STEP_CKPT_WRITE_S = 3.0; ENDPOINT_OVERHEAD_S = 60.0; SLACK_PER_ARM_S = 150.0
DEADLINE = dt.datetime.fromisoformat("2026-09-13T01:52:44+00:00").timestamp()
_X = None; _WARM = None; _E = None


def _preload():
    global _X, _WARM, _E
    t = time.monotonic()
    _X = torch.tensor(np.asarray(np.load(SUB, mmap_mode="r"), np.float16), device="cuda")
    _WARM = torch.load(str(INIT), map_location="cpu", weights_only=False)["model_state"]
    ez = np.load(GRAPH); _E = (ez["sources"], ez["targets"]); return time.monotonic() - t


def _measured_fit(steps, ckpt_dir):
    coeff = V.calibrated_coeff()
    torch.manual_seed(V.SEED); np.random.seed(V.SEED); torch.cuda.manual_seed_all(V.SEED)
    ident = V.expected_identity("grouped_infonce", ROOT)     # most expensive arm
    engine = E.GroupedEngine("grouped_infonce", ident, coeff, "cuda", CHAMPION, _WARM)
    sampler = G.GroupedSampler(V.N, _E[0], _E[1], seed=V.SEED); prev_epoch = sampler.epoch; pending = False
    _, total = torch.cuda.mem_get_info(); torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); t0 = time.monotonic()
    while engine.success < steps:
        heads, tails = sampler.next_block()
        if sampler.epoch > prev_epoch: pending = True; prev_epoch = sampler.epoch
        if engine.step(_X, heads, tails):
            s = engine.success
            if pending: torch.save(engine.ckpt_dict(sampler, True), Path(ckpt_dir) / f"ckpt-epoch{sampler.epoch}.pt"); pending = False    # FULL payload
            if s == min(steps, 500): torch.save(engine.ckpt_dict(sampler, True), Path(ckpt_dir) / "ckpt-step.pt")                          # FULL payload
        assert engine.stats["nonfinite_skips"] < 300, "preflight: bounded overflow/nonfinite failure — stop"
    torch.cuda.synchronize(); wall = time.monotonic() - t0
    free1, _ = torch.cuda.mem_get_info(); return wall, torch.cuda.max_memory_allocated() / 2**30, (total - free1) / 2**30, total / 2**30


def _spent(path, key):
    if not Path(path).exists(): return 0.0
    try: v = float(json.loads(Path(path).read_text())[key])
    except Exception as e: raise RuntimeError(f"malformed ledger {path}: {e!r} — fail closed")
    assert math.isfinite(v) and v >= 0; return v


def main():
    import tempfile
    t_pf0 = time.monotonic(); prep = _preload()
    with tempfile.TemporaryDirectory(dir=str(SB)) as td:
        d1 = Path(td) / "m1"; d2 = Path(td) / "m2"; d1.mkdir(); d2.mkdir()
        w1, pk1, gu1, total = _measured_fit(W1, d1)
        w2, pk2, gu2, _ = _measured_fit(W2, d2)
    assert math.isfinite(w1) and math.isfinite(w2) and w2 > w1 > 0, "non-finite/degenerate windows — stop"
    per_step = max((w2 - w1) / (W2 - W1), w2 / W2); setup = max(0.0, w1 - per_step * W1) + prep    # include real setup/PERM/load
    assert math.isfinite(per_step) and per_step > 0, "non-finite/nonpositive per-step — stop"
    itps = 1.0 / per_step
    spe = math.ceil((V.N * 15) / V.BLOCK_POS); n_epochs = math.ceil(V.DOSE / spe)
    per_arm = setup + per_step * V.DOSE + n_epochs * EPOCH_CKPT_WRITE_S + len(V.STEP_CKPTS) * STEP_CKPT_WRITE_S + ENDPOINT_OVERHEAD_S + SLACK_PER_ARM_S
    card_spent = _spent(CARD, "batch_spent_s"); win_spent = _spent(WIN, "spent_s")
    preflight_wall = time.monotonic() - t_pf0
    cap_room = GPU_CAP - (card_spent + preflight_wall); win_room = WIN_CAP - (win_spent + preflight_wall)
    deadline_room = DEADLINE - time.time(); all_arms = N_ARMS * per_arm; peak_global = round(max(gu1, gu2), 3)
    checks = {"windows_valid": bool(w2 > w1 > 0), "rate_finite_positive": bool(math.isfinite(itps) and itps > 0),
              "per_arm_le_cap": bool(per_arm <= PER_ARM_CAP), "three_arms_fit_cap": bool(all_arms <= cap_room),
              "three_arms_fit_window": bool(all_arms <= win_room), "three_arms_fit_deadline": bool((all_arms + 60) <= deadline_room),
              "global_vram_lt_30gb": bool(peak_global < 30.0)}
    R = {"schema": "card034-preflight-2026-09-12", "at": dt.datetime.now(dt.timezone.utc).isoformat(),
         "engine": "card034_engine.GroupedEngine (production)", "windows": {str(W1): w1, str(W2): w2}, "preload_wall_s": prep,
         "per_step_s": per_step, "setup_s": setup, "it_per_s": itps, "dose": V.DOSE, "n_arms": N_ARMS,
         "steps_per_epoch_est": spe, "n_epochs_est": n_epochs, "per_arm_estimate_s": per_arm, "per_arm_cap_s": PER_ARM_CAP,
         "three_arms_s": all_arms, "gpu_cap_s": GPU_CAP, "card_spent_s": card_spent, "cap_room_s": cap_room,
         "window_room_s": win_room, "deadline_room_s": deadline_room, "proc_peak_vram_gb": round(max(pk1, pk2), 3),
         "global_vram_used_gb": peak_global, "gpu_total_gb": round(total, 3), "preflight_wall_s": preflight_wall,
         "note": "Real production engine + FULL checkpoint payload; conservative per_step=max(slope, long whole-fit "
                 "avg); setup includes graph/PERM/substrate load. No fallback; degenerate/overflow STOPS; no truncation.",
         "PASS": bool(all(checks.values())), "checks": checks}
    (OC / "card034-preflight.json").write_text(json.dumps(R, indent=2))
    print(json.dumps({k: R[k] for k in ("per_step_s", "per_arm_estimate_s", "three_arms_s", "cap_room_s", "PASS")}, indent=1), flush=True)
    return 0 if R["PASS"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
