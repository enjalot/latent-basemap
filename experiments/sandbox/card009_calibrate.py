"""Card009 coefficient calibration + cross-device JVP floor (GPU, under flock; charged to 009).
Per the admission review: both preservation losses are ZERO at teacher init, so calibrate at a
MOVED student state (the real 70K IN snapshot — same recipe family), NOT at init. Measures:
  - cross-device floor (at T0 INIT): the deriv term uses GPU-JVP student Jv vs the CPU-cached teacher
    Jv, so the init deriv-loss floor is a GPU-vs-CPU numeric residual (not 0); record it + the replay
    init floor, at deriv subbatch chunk sizes 64/128/256 (CPU-fidelity 0 does not prove cross-device).
  - added preservation-gradient norms at the MOVED state: ||d/dparams (deriv_term)|| and
    ||d/dparams (replay_term)|| at weight 1, on independent subbatches.
Then freezes ONE coefficient per treatment so the three arms carry a COMPARABLE added preservation-
gradient norm: ordinary OUT (replay w=0.02); derivative (replay 0.02 + deriv w_d, w_d fixed); stronger
pointwise (replay 0.02 + extra Δ chosen so Δ·||∇replay|| == w_d·||∇deriv||). Writes card009-calibration.json.
Usage: card009_calibrate.py
"""
import os, sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import torch
from basemap.pumap.parametric_umap.core import ParametricUMAP

SB = Path("/data/latent-basemap/sandbox"); OC = SB / "overseer-codex"
T0 = SB / "dino-arrival-t0/champion-bs16k/model.pt"
MOVED = SB / "dino-arrival-t0/replay-updates/snapshots-in/model-step70000.pt"   # real moved student (same recipe family)
R0 = 33.6717; W_R = 0.02          # base OUT replay weight (all arms)
DEV = "cuda"
# Calibration target: the derivative arm and the stronger-pointwise arm each ADD a preservation-
# gradient norm equal to the ORDINARY arm's base replay-preservation norm (W_R*||grad_replay||),
# i.e. ~double the preservation, matched across the two treatments. w_d and the extra replay weight
# are then derived from the measured moved-state gradient norms.


def _norm(a):
    a = np.asarray(a, np.float32); return a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-12)


def deriv_grad_norm(model, X, V, S, JVt):
    for p in model.parameters():
        p.requires_grad_(True); p.grad = None
    _, jv = torch.autograd.functional.jvp(lambda i: model(i), X, V, create_graph=True)
    loss = ((S.unsqueeze(1) * (jv.float() - JVt) / R0) ** 2).sum(1).mean()
    loss.backward()
    return float(sum(p.grad.norm() for p in model.parameters() if p.grad is not None)), float(loss.item())


def replay_grad_norm(model, X, T):
    for p in model.parameters():
        p.requires_grad_(True); p.grad = None
    z = model(X).float()
    loss = (z - T).pow(2).sum(1).mean()
    loss.backward()
    return float(sum(p.grad.norm() for p in model.parameters() if p.grad is not None)), float(loss.item())


def main():
    if not torch.cuda.is_available():
        print(json.dumps({"error": "no cuda"})); return 3
    db = np.load(OC / "card009_deriv_bank.npz"); rb = np.load(OC / "card006_out_bank.npz")
    dX = torch.from_numpy(np.asarray(db["deriv_X"], np.float32)).to(DEV)
    dV = torch.from_numpy(np.asarray(db["deriv_dir"], np.float32)).to(DEV)
    dS = torch.from_numpy(np.asarray(db["deriv_scale"], np.float32)).to(DEV)
    dJV = torch.from_numpy(np.asarray(db["deriv_teacher_jv"], np.float32)).to(DEV)
    rX = torch.from_numpy(_norm(np.asarray(rb["replay_X"], np.float32))).to(DEV)
    rT = torch.from_numpy(np.asarray(rb["replay_targets"], np.float32)).to(DEV)
    rng = np.random.default_rng(909)

    # ---- cross-device floor at T0 INIT (deriv term uses GPU JVP vs CPU-cached teacher Jv) ----
    t0 = ParametricUMAP.load(str(T0), device=DEV); t0.model.eval()
    floor = {}
    for cs in (64, 128, 256):
        idx = torch.from_numpy(rng.choice(dX.shape[0], cs, replace=False)).to(DEV)
        with torch.no_grad():
            _, jv = torch.autograd.functional.jvp(lambda i: t0.model(i), dX[idx], dV[idx])
            dloss = ((dS[idx].unsqueeze(1) * (jv.float() - dJV[idx]) / R0) ** 2).sum(1).mean().item()
        floor[f"deriv_loss_init_sub{cs}"] = float(dloss)
    ridx = torch.from_numpy(rng.choice(rX.shape[0], 819, replace=False)).to(DEV)
    with torch.no_grad():
        rloss_init = (t0.model(rX[ridx]).float() - rT[ridx]).pow(2).sum(1).mean().item()
    floor["replay_loss_init_819"] = float(rloss_init)

    # ---- added gradient norms at the MOVED state ----
    moved = ParametricUMAP.load(str(MOVED), device=DEV); moved.model.train()
    di = torch.from_numpy(rng.choice(dX.shape[0], 128, replace=False)).to(DEV)
    g_deriv, l_deriv = deriv_grad_norm(moved.model, dX[di], dV[di], dS[di], dJV[di])
    moved2 = ParametricUMAP.load(str(MOVED), device=DEV); moved2.model.train()
    ri = torch.from_numpy(rng.choice(rX.shape[0], 819, replace=False)).to(DEV)
    g_replay, l_replay = replay_grad_norm(moved2.model, rX[ri], rT[ri])

    target_added = W_R * g_replay                          # base replay-preservation gradient norm
    w_d = target_added / g_deriv if g_deriv > 0 else None   # deriv weight matching that added norm
    delta_pointwise = target_added / g_replay if g_replay > 0 else None   # = W_R (double the replay preservation)
    out = {"schema": "card009-calibration-2026-09-11", "R0": R0, "moved_state": str(MOVED),
           "cross_device_floor": floor,
           "moved_state_grad_norms_w1": {"deriv_term": round(g_deriv, 6), "replay_term": round(g_replay, 6),
                                         "deriv_loss_moved": round(l_deriv, 6), "replay_loss_moved": round(l_replay, 6)},
           "target_added_preservation_grad_norm": round(target_added, 6),
           "frozen_coefficients": {
               "ordinary_out": {"replay_weight": W_R, "deriv_weight": 0.0},
               "derivative": {"replay_weight": W_R, "deriv_weight": round(w_d, 6)},
               "stronger_pointwise": {"replay_weight": round(W_R + delta_pointwise, 6), "deriv_weight": 0.0,
                                      "extra_replay_weight": round(delta_pointwise, 6)}},
           "calibration_logic": f"target added preservation-grad-norm = W_R*||grad_replay|| = {W_R}*{g_replay:.4g} = "
                                f"{target_added:.4g} (the ordinary arm's base replay preservation). derivative arm: "
                                f"w_d = {target_added:.4g}/{g_deriv:.4g} = {w_d:.4g}; stronger-pointwise arm: extra replay "
                                f"delta = {target_added:.4g}/{g_replay:.4g} = {delta_pointwise:.4g}. Both add ~equal preservation grad.",
           "note": "one frozen setting per treatment; both losses are 0 at teacher init so calibration is at the moved state. "
                   "cross-device deriv init floor is a GPU-JVP-vs-CPU-cached-teacher residual, not 0 (recorded per chunk)."}
    (OC / "card009-calibration.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"cross_device_floor": floor, "grad_norms": out["moved_state_grad_norms_w1"],
                      "frozen_coefficients": out["frozen_coefficients"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
