"""Card008 LR-schedule canary (CPU). Verifies the three finishing schedules that branch from
the 70K IN snapshot, each for another 70K successful updates:
  1 constant 1e-4  ·  2 cosine 1e-4->1e-5 (floor)  ·  3 constant = discrete average of arm-2 LRs.
Two checks:
 A. ANALYTIC over the real H=70000 horizon using core's EXACT cosine lambda (with lr_min floor):
    arm2 LR[0]=1e-4, LR[H-1]≈1e-5, strictly decreasing, min>0 (no zero-LR counted update); and
    arm3 constant = mean(arm2 LRs) (the value to register, ~5.5e-5).
 B. REAL tiny CPU fits confirm CORE applies the floor: lr_schedule=cosine + lr_min=1e-5 ends at
    ~1e-5 (not 0) via train_stats.lr_used_last; lr_min=0 ends near 0; plateau stays constant.
Writes card008-lr-canary.json (incl the registered arm-3 constant). Usage: tiny_lr_schedule_canary.py
"""
import os, sys, json, tempfile
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import torch
from basemap.pumap.parametric_umap.core import ParametricUMAP

OC = Path("/data/latent-basemap/sandbox/overseer-codex")
BASE_LR, LR_MIN, H = 1e-4, 1e-5, 70000
N, D, K = 1500, 64, 10


def cosine_seq(base, lr_min, H, W=0):
    floor = (lr_min / base) if (lr_min and base > 0) else 0.0
    u = np.arange(H)
    lam = floor + (1.0 - floor) * 0.5 * (1.0 + np.cos(np.pi * np.minimum((u - W) / max(1, H - W), 1.0)))
    return base * lam


def tiny_fit(schedule, lr, lr_min, Hs):
    rng = np.random.default_rng(0); X = rng.standard_normal((N, D)).astype(np.float32)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)
    S = Xn @ Xn.T; np.fill_diagonal(S, -1); nn = np.argsort(-S, axis=1)[:, :K]
    td = Path(tempfile.mkdtemp(prefix="lrcan_")); e = td / "e.npz"
    np.savez(e, sources=np.repeat(np.arange(N), K).astype(np.int32), targets=nn.reshape(-1).astype(np.int32),
             weights=np.ones(N * K, np.float32), n_nodes=np.int64(N))
    torch.manual_seed(0); np.random.seed(0)
    m = ParametricUMAP(n_components=2, n_epochs=40, batch_size=256, low_dim_kernel="umap",
                       learning_rate=lr, lr_schedule=schedule, lr_min=lr_min)
    m.total_steps_estimate = Hs; m._max_train_steps = Hs
    m.fit(X.astype(np.float32), precomputed_edges_path=str(e), random_state=0, verbose=False)
    ts = dict(m._train_stats)
    return {"lr_first": ts.get("lr_used_first"), "lr_last": ts.get("lr_used_last"),
            "lr_min": ts.get("lr_used_min"), "lr_max": ts.get("lr_used_max"),
            "executed": ts.get("executed_iters")}


def main():
    # ---- A. analytic over the real horizon ----
    seq = cosine_seq(BASE_LR, LR_MIN, H)
    arm3_const = float(seq.mean())
    A = {"arm2_lr_first": float(seq[0]), "arm2_lr_last": float(seq[-1]), "arm2_lr_min": float(seq.min()),
         "strictly_decreasing": bool(np.all(np.diff(seq) < 0)), "no_zero_lr": bool(seq.min() > 0),
         "arm3_registered_constant_lr": repr(arm3_const),
         "arm3_matches_launch_value": bool(arm3_const == 0.000055000642857142866),
         "checks": {"first_is_1e-4": bool(abs(seq[0] - 1e-4) < 1e-9),
                    "last_is_~1e-5": bool(abs(seq[-1] - 1e-5) < 2e-9),
                    "arm3_in_5.4e-5..5.6e-5": bool(5.4e-5 <= arm3_const <= 5.6e-5)}}
    A_pass = all(A["checks"].values()) and A["strictly_decreasing"] and A["no_zero_lr"] and A["arm3_matches_launch_value"]

    # ---- B. real tiny fits confirm core applies the floor ----
    Hs = 300
    cos_floor = tiny_fit("cosine", BASE_LR, LR_MIN, Hs)
    cos_zero = tiny_fit("cosine", BASE_LR, 0.0, Hs)
    const1 = tiny_fit("constant", BASE_LR, 0.0, Hs)          # arm1: truly-constant schedule
    const3 = tiny_fit("constant", arm3_const, 0.0, Hs)       # arm3: truly-constant at the mean LR
    B = {"cosine_floor": cos_floor, "cosine_zero": cos_zero, "constant_arm1": const1, "constant_arm3": const3}
    def is_const(c, lr):
        return (c["lr_min"] is not None and c["lr_max"] is not None
                and abs(c["lr_min"] - c["lr_max"]) < 1e-12 and abs(c["lr_min"] - lr) < 1e-12)
    B_pass = bool(cos_floor["lr_first"] and abs(cos_floor["lr_first"] - BASE_LR) < 1e-9
                  and abs(cos_floor["lr_last"] - LR_MIN) / LR_MIN < 0.05        # floor applied (~1e-5)
                  and cos_zero["lr_last"] < cos_floor["lr_last"]                 # no-floor ends lower
                  and is_const(const1, BASE_LR) and is_const(const3, arm3_const) # TRUE constants (min==max==lr)
                  and cos_floor["executed"] == Hs)

    out = {"schema": "card008-lr-canary-2026-09-11", "horizon_H": H, "base_lr": BASE_LR, "lr_min": LR_MIN,
           "analytic": A, "real_tiny_fits": B, "A_pass": bool(A_pass), "B_pass": B_pass,
           "PASS": bool(A_pass and B_pass),
           "note": "arm1=constant 1e-4; arm2=cosine 1e-4->1e-5 (lr_min floor); arm3=constant "
                   f"{round(arm3_const, 10)} (mean of arm2 LR sequence). No zero-LR counted update."}
    print(json.dumps(out, indent=1))
    OC.mkdir(parents=True, exist_ok=True); (OC / "card008-lr-canary.json").write_text(json.dumps(out, indent=1))
    return 0 if out["PASS"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
