"""Card022 device canary (real 2M... 300K continuation path, short dose) — per card022-local-shape-floor.md.
Proves, before production (root runs on GPU):
  1. OFF is BITWISE-identical on the real rank-window/device path: hook attrs unset == bank set with weight 0
     (gated branch not entered, no shape_gen draw, no extra forward);
  2. ON (weight>0 + bank) DIVERGES from OFF (the covariance-floor term is active);
  3. GENUINE resume twin, shape ON: a step-checkpointed run resumed mid-flight reaches a BITWISE-identical
     endpoint. The mid-run checkpoint is LOADED and asserted (step in range, step flag, identity bound,
     model finite, and the independent bank-sampler RNG present);
  4. wrong-BANK, wrong-WEIGHT, wrong-SEED and wrong-ARM resumes FAIL CLOSED on the SPECIFIC admission-identity
     error (message match), not any ValueError.
Uses a small bank slice for speed (mechanism, not production identity). Exit 0 = PASS. Usage: gpu_card022_canary.py
"""
import os, sys, json, hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import card022_validate as V
import torch
from basemap.pumap.parametric_umap.core import ParametricUMAP

SB = V.SB; OC = V.OC; PARENT = V.PARENT; SUB = V.SUB; GRAPH = V.GRAPH; BANKD = V.BANKD
SEED = V.SEED; SHORT = 120; CKPT_AT = 60; RANKNEG = V.RANKNEG; IDMISS = "admission-identity mismatch"
_X = None; _WARM = None; _BANK = None


def _prep():
    global _X, _WARM, _BANK
    if _X is None: _X = np.asarray(np.load(SUB, mmap_mode="r"), np.float32)
    if _WARM is None: _WARM = torch.load(str(PARENT), map_location="cpu", weights_only=False)["model_state_dict"]
    if _BANK is None:
        bx = np.asarray(np.load(BANKD / "X.npy", mmap_mode="r")[:512], np.float16); bt = np.asarray(np.load(BANKD / "tau.npy")[:512], np.float64)
        _BANK = {"X": bx, "tau": bt, "sha": hashlib.sha256(bx.tobytes()).hexdigest()[:16]}


def _identity(arm, weight, bank_sha, seed):
    return {"card": "card022-canary", "arm": arm, "weight": float(weight), "bank_sha": bank_sha, "seed": int(seed)}


def _run(arm, weight, seed=SEED, resume_from=None, ckpt_targets=None, ckpt_dir=None, force_bank=False):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    p = ParametricUMAP.load(str(PARENT), device="cuda"); p.model = None
    p.learning_rate = V.LR; p.lr_schedule = "constant"; p.batch_size = V.BATCH; p.warmup_steps = 0
    p.n_epochs = 100000; p.rankneg_window = RANKNEG; p._max_train_steps = SHORT
    p.x_residency = "auto"; p.required_input_pipeline = "device"
    V.validate_parent_recipe(p)
    for a, v in (("anchor_ids_path", ""), ("anchor_hold_weight", 0.0), ("replay_bank_path", ""),
                 ("replay_weight", 0.0), ("deriv_bank_path", ""), ("deriv_weight", 0.0)):
        if hasattr(p, a): setattr(p, a, v)
    bank_sha = None
    if weight > 0 or force_bank:      # force_bank configures the bank even at weight 0 (real OFF-parity test)
        p._shape_bank = {"X": _BANK["X"], "tau": _BANK["tau"]}
        p._shape_eps = float(V.EPSILON); p._shape_centers_per_step = V.CENTERS_PER_STEP
        if weight > 0: bank_sha = _BANK["sha"]
    p._shape_weight = float(weight)
    p._card012_identity = _identity(arm, weight, bank_sha, seed)
    if ckpt_targets: p._checkpoint_step_targets = set(ckpt_targets)
    kw = {}
    if ckpt_dir: kw.update(checkpoint_dir=str(ckpt_dir), checkpoint_every_epochs=1)
    if resume_from: kw.update(resume_from=str(resume_from))
    p.fit(_X, precomputed_edges_path=str(GRAPH), random_state=seed, verbose=False,
          warm_start_state=(None if resume_from else _WARM), **kw)
    assert p._train_stats["positive_lr_optimizer_steps"] == SHORT
    assert (getattr(p, "_pipeline_info", {}) or {}).get("x_residency") == "device_fp16", "pipeline not device_fp16"
    assert all(torch.isfinite(t).all() for t in p.model.state_dict().values())
    return V.state_sha(p.model.state_dict())


def main():
    global _BANK
    _prep(); R = {"schema": "card022-canary-2026-09-12", "short_dose": SHORT, "ckpt_at": CKPT_AT, "bank_sha": _BANK["sha"]}
    import tempfile
    _off_tmp = tempfile.TemporaryDirectory(dir=str(SB))
    off_dir = Path(_off_tmp.name)
    sha_unset = _run("ordinary", 0.0, ckpt_targets={SHORT}, ckpt_dir=off_dir / "unset")                      # hook attrs unset (no bank configured)
    sha_bankw0 = _run("ordinary", 0.0, force_bank=True, ckpt_targets={SHORT}, ckpt_dir=off_dir / "zero")    # REAL bank configured but weight 0 (no upload/gen/forward)
    R["off_bitwise_identical"] = bool(sha_unset == sha_bankw0)   # configured-bank/weight0 == unset (real OFF parity + RNG parity)
    def eq(a,b):
        if torch.is_tensor(a): return torch.is_tensor(b) and torch.equal(a,b)
        if isinstance(a,(list,tuple)): return isinstance(b,(list,tuple)) and len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
        return a == b
    ca, cb = [torch.load(off_dir / tag / f"ckpt-step{SHORT}.pt", map_location="cpu", weights_only=False) for tag in ["unset","zero"]]
    R["off_rng_byte_equal"] = all(eq(ca[k],cb[k]) for k in ["torch_rng","cuda_rng","loader_gen","mn_gen","dens_gen","hold_gen","shape_gen"])
    _off_tmp.cleanup()
    sha_rerun = _run("ordinary", 0.0)                      # deterministic rerun — a SEPARATE check
    R["deterministic_rerun"] = bool(sha_rerun == sha_unset)
    sha_base = sha_unset
    sha_on = _run("shape_floor", 1.0)
    R["on_diverges"] = bool(sha_on != sha_base)

    import tempfile, time
    with tempfile.TemporaryDirectory(dir=str(SB)) as td:
        cdir = Path(td) / "ckpts"; cdir.mkdir(parents=True)
        _t = time.monotonic(); sha_full = _run("shape_floor", 1.0, ckpt_targets={CKPT_AT}, ckpt_dir=cdir)
        R["shape_on_wall_s"] = round(time.monotonic() - _t, 2)
        R["it_per_s"] = round(SHORT / R["shape_on_wall_s"], 2) if R["shape_on_wall_s"] > 0 else None
        ck = cdir / f"ckpt-step{CKPT_AT}.pt"; assert ck.exists(), "mid-run step checkpoint not written"
        o = torch.load(ck, map_location="cpu", weights_only=False)
        R["ckpt_global_step"] = int(o.get("global_step", -1)); R["ckpt_step_in_range"] = bool(0 < R["ckpt_global_step"] < SHORT)
        R["ckpt_step_flag"] = bool(o.get("step_checkpoint")); R["ckpt_identity_bound"] = o.get("card012_identity") == _identity("shape_floor", 1.0, _BANK["sha"], SEED)
        R["ckpt_has_shape_gen"] = o.get("shape_gen") is not None
        V.validate_resume_payload(o, CKPT_AT)
        R["ckpt_full_resume_schema"] = True
        R["ckpt_shape_exposure"] = o["train_stats"].get("shape_successful_steps") == CKPT_AT
        R["ckpt_model_finite"] = all(bool(torch.isfinite(t).all()) for t in o["model"].values())
        sha_resume = _run("shape_floor", 1.0, resume_from=ck, ckpt_dir=cdir)
        R["resume_twin_bitwise"] = bool(sha_resume == sha_full)

        def _reject(arm, weight, seed):
            try:
                _run(arm, weight, seed=seed, resume_from=ck, ckpt_dir=cdir); return False
            except ValueError as e:
                return IDMISS in str(e)
            except Exception:
                return False
        R["wrong_weight_rejected"] = _reject("shape_floor", 2.0, SEED)     # different weight ⇒ identity mismatch
        R["wrong_seed_rejected"] = _reject("shape_floor", 1.0, SEED + 1)
        R["wrong_arm_rejected"] = _reject("ordinary", 0.0, SEED)
        # wrong bank: same arm/weight/seed but a different bank slice ⇒ different bank_sha in identity
        good = _BANK; _BANK = {"X": good["X"][:256], "tau": good["tau"][:256], "sha": hashlib.sha256(good["X"][:256].tobytes()).hexdigest()[:16]}
        R["wrong_bank_rejected"] = _reject("shape_floor", 1.0, SEED); _BANK = good

    keys = ["off_rng_byte_equal", "ckpt_full_resume_schema", "ckpt_shape_exposure", "off_bitwise_identical", "deterministic_rerun", "on_diverges", "ckpt_step_in_range", "ckpt_step_flag", "ckpt_identity_bound",
            "ckpt_has_shape_gen", "ckpt_model_finite", "resume_twin_bitwise", "wrong_weight_rejected",
            "wrong_seed_rejected", "wrong_arm_rejected", "wrong_bank_rejected"]
    R["PASS"] = bool(all(R[k] for k in keys))
    (OC / "card022-canary.json").write_text(json.dumps(R, indent=1)); print(json.dumps(R, indent=1), flush=True)
    return 0 if R["PASS"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
