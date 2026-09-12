"""Card022 coefficient calibration (per card022-local-shape-floor.md). Freezes ONE shape coefficient BEFORE
either production arm by matching the added shape-gradient global-L2 norm to 10% of the ordinary
pairwise-loss gradient at the SAME starting head (the Card013 baseline endpoint), over eight fixed
actual-recipe calibration batches with a disjoint, recorded calibration RNG. Uses the core's default-off
calibration probe (real batches, real umap pairwise loss, real shape term). coefficient = median finite
positive (0.10 * ||grad_pairwise|| / ||grad_shape||). If the required ratio is undefined/unstable/nonfinite
(fewer than the required finite positive ratios, or any non-finite gradient), STOP and report — no strength
sweep, no evaluation-score tuning. GPU; root runs before the shape_floor arm. Usage: calibrate_card022.py
"""
import os, sys, json, time, math, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import card022_validate as V
import torch
from basemap.pumap.parametric_umap.core import ParametricUMAP

ROOT = Path(__file__).resolve().parents[2]
OC = V.OC; PARENT = V.PARENT; SUB = V.SUB; GRAPH = V.GRAPH; BANKD = V.BANKD
CALIB_SEED = 2202; N_BATCHES = 8; TARGET_FRAC = 0.10


def main():
    assert torch.cuda.is_available(), "calibration is the device path"
    fok, bad = V.runtime_manifest_check(ROOT); assert fok, f"frozen runtime mismatch {bad}"
    assert V.full_sha(PARENT) == V.TEACHER_SHA and V.full_sha(SUB) == V.SUB_SHA256 and V.full_sha(GRAPH) == V.GRAPH_SHA256
    assert V.full_sha(BANKD / "X.npy") == V.BANK_X_SHA256 and V.full_sha(BANKD / "tau.npy") == V.BANK_TAU_SHA256
    parent = torch.load(str(PARENT), map_location="cpu", weights_only=False); warm = parent["model_state_dict"]
    bank = {"X": np.asarray(np.load(BANKD / "X.npy", mmap_mode="r"), np.float16), "tau": np.asarray(np.load(BANKD / "tau.npy"), np.float64)}

    torch.manual_seed(CALIB_SEED); np.random.seed(CALIB_SEED); torch.cuda.manual_seed_all(CALIB_SEED)
    p = ParametricUMAP.load(str(PARENT), device="cuda"); p.model = None
    assert p.n_components == V.NC
    p.learning_rate = V.LR; p.lr_schedule = "constant"; p.batch_size = V.BATCH; p.warmup_steps = 0
    p.n_epochs = 100000; p._max_train_steps = N_BATCHES + 2; p.rankneg_window = V.RANKNEG
    p.x_residency = "auto"; p.required_input_pipeline = "device"
    V.validate_parent_recipe(p)
    for a, v in (("anchor_ids_path", ""), ("anchor_hold_weight", 0.0), ("replay_bank_path", ""),
                 ("replay_weight", 0.0), ("deriv_bank_path", ""), ("deriv_weight", 0.0)):
        if hasattr(p, a): setattr(p, a, v)
    p._shape_bank = bank; p._shape_weight = 1.0; p._shape_eps = float(V.EPSILON); p._shape_centers_per_step = V.CENTERS_PER_STEP
    p._calib_probe = {"n": N_BATCHES}; p._calib_records = []

    init_param_sha = V.state_sha(warm)     # pristine parent head; must be unchanged after calibration
    X = np.asarray(np.load(SUB, mmap_mode="r"), np.float32)
    t0 = time.time()
    p.fit(X, precomputed_edges_path=str(GRAPH), random_state=CALIB_SEED, verbose=False, warm_start_state=warm)
    recs = list(p._calib_records); ts = dict(getattr(p, "_train_stats", {}) or {})
    assert (getattr(p, "_pipeline_info", {}) or {}).get("x_residency") == "device_fp16", "pipeline not device_fp16"

    # FIXED-HEAD proof: no optimizer updates for ANY of the 8 batches
    final_param_sha = V.state_sha(p.model.state_dict())
    head_unchanged = bool(final_param_sha == init_param_sha)
    pos_steps = int(ts.get("positive_lr_optimizer_steps", -1))
    warm_hash_ok = bool(getattr(p, "warm_start_sha256", None))     # parent weights applied
    # require ALL 8 batches finite positive — a nonfinite ratio must NOT be silently dropped
    raw = recs
    distinct_batches = len({r["pair_batch_sha256"] for r in recs}) == N_BATCHES
    all8 = (len(recs) == N_BATCHES and
            all(math.isfinite(r["grad_pairwise_l2"]) and math.isfinite(r["grad_shape_l2"]) and
                r["grad_pairwise_l2"] > 0 and r["grad_shape_l2"] > 0 for r in recs))
    ratios = [TARGET_FRAC * r["grad_pairwise_l2"] / r["grad_shape_l2"] for r in recs] if all8 else []
    coeff = float(np.median(ratios)) if all8 else None
    spread = ({"min": float(min(ratios)), "max": float(max(ratios)), "mean": float(np.mean(ratios)),
               "std": float(np.std(ratios)), "ratios": ratios} if all8 else None)
    fractions = [coeff * r["grad_shape_l2"] / r["grad_pairwise_l2"] for r in recs] if all8 else []
    # Prospective operational bound: no calibration batch may receive an added
    # gradient larger than the ordinary gradient (10x intended median fraction).
    stable = bool(all8 and distinct_batches and head_unchanged and pos_steps == 0 and coeff is not None and math.isfinite(coeff) and coeff > 0 and max(fractions) <= 1.0)
    R = {"schema": "card022-calibration-2026-09-12", "at": dt.datetime.now(dt.timezone.utc).isoformat(),
         "calibration_seed": CALIB_SEED, "n_batches": N_BATCHES, "target_fraction": TARGET_FRAC,
         "n_records": len(recs), "raw_norms": raw, "eight_distinct_batches": distinct_batches,
         "realized_gradient_fractions": fractions, "max_allowed_gradient_fraction": 1.0, "all8_finite_positive": all8,
         "head_unchanged": head_unchanged, "init_param_sha": init_param_sha, "final_param_sha": final_param_sha,
         "positive_optimizer_steps": pos_steps, "warm_hash_recorded": warm_hash_ok,
         "coefficient": coeff, "ratio_spread": spread, "wall_s": round(time.time() - t0, 1),
         "method": "coefficient = median(0.10 * ||grad_pairwise|| / ||grad_shape||) over EXACTLY 8 fixed "
                   "actual-recipe batches at the pristine Card013 baseline head (no optimizer updates; "
                   "init==final param hash, 0 positive steps); all 8 finite positive required (no drop); "
                   "disjoint recorded calibration RNG (seed 2202); no strength sweep; not tuned on scores.",
         "PASS": stable}
    if not stable:
        R["stop_reason"] = ("STOP: " + ("head moved during calibration; " if not head_unchanged else "")
                            + (f"positive_optimizer_steps={pos_steps}; " if pos_steps != 0 else "")
                            + ("not all 8 ratios finite positive; " if not all8 else "")
                            + "no strength sweep, no dose change")
    (OC / "card022-calibration.json").write_text(json.dumps(R, indent=2))
    print(json.dumps({"coefficient": coeff, "all8_finite_positive": all8, "head_unchanged": head_unchanged,
                      "positive_optimizer_steps": pos_steps, "PASS": stable}, indent=1), flush=True)
    return 0 if stable else 3


if __name__ == "__main__":
    raise SystemExit(main())
