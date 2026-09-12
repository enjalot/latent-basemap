"""Card018 device canary (real 2M fit path, short dose) — per card018-scale-2m.md + root review. Proves:
  1. radius==ONES reproduces the baseline fit BITWISE, actual radii DIVERGE (kernel active);
  2. GENUINE resume twin, radii ACTIVE: step-checkpointed run resumed mid-flight reaches a BITWISE-identical
     endpoint (not a hash assertion). The mid-run checkpoint is LOADED and asserted: global_step in (0,total),
     step_checkpoint flag set, identity bound, model finite — not mere existence;
  3. wrong-arm and wrong-radius resumes FAIL CLOSED on the SPECIFIC admission-identity error (message match),
     not any ValueError.
Precision is asserted device_fp16 on every fit. Exit 0 = PASS. Usage: gpu_card018_canary.py
"""
import os, sys, json, hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import card018_validate as V
import torch
from basemap.pumap.parametric_umap.core import ParametricUMAP

SB = V.SB; OC = V.OC; DATA = V.DATA
CHAMPION = SB / "dino-arrival-t0/champion-bs16k/model.pt"; INIT3D = V.INIT3D
SEED = V.SEED; SHORT = 120; CKPT_AT = 60; RANKNEG = V.RANKNEG
WARM_PARAM_SHA = V.WARM_PARAM_SHA; INIT_SHA = V.INIT_NAMED_SHA; IDMISS = "admission-identity mismatch"
_X = None


def _substrate():
    global _X
    if _X is None: _X = np.asarray(np.load(DATA / "substrate.f16.npy", mmap_mode="r"), np.float32)
    return _X


def _identity(arm, radii):
    return {"card": "card018", "arm": arm,
            "radii_sha": (hashlib.sha256(np.ascontiguousarray(radii).tobytes()).hexdigest()[:16] if radii is not None else None),
            "init_sha256": INIT_SHA, "rankneg_window": RANKNEG}


def _fresh(arm, radii):
    init = torch.load(str(INIT3D), map_location="cpu", weights_only=False); assert init["init_state_sha256"] == INIT_SHA
    p = ParametricUMAP.load(str(CHAMPION), device="cuda"); p.model = None; p.n_components = 3
    p.learning_rate = 0.001; p.lr_schedule = "constant"; p.batch_size = 16384; p.warmup_steps = 0
    p.n_epochs = 100000; p.rankneg_window = RANKNEG; p.x_residency = "auto"; p.required_input_pipeline = "device"
    for a, v in (("anchor_ids_path", ""), ("anchor_hold_weight", 0.0), ("replay_bank_path", ""),
                 ("replay_weight", 0.0), ("deriv_bank_path", ""), ("deriv_weight", 0.0)):
        if hasattr(p, a): setattr(p, a, v)
    if radii is not None: p._card013_radii = radii
    p._card012_identity = _identity(arm, radii)
    return p, init["model_state"]


def _run(arm, radii, steps, warm, resume_from=None, ckpt_targets=None, ckpt_dir=None):
    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed_all(SEED)
    p, init_state = _fresh(arm, radii); p._max_train_steps = steps
    if ckpt_targets: p._checkpoint_step_targets = set(ckpt_targets)
    kw = {}
    if ckpt_dir: kw.update(checkpoint_dir=str(ckpt_dir), checkpoint_every_epochs=1)
    if resume_from: kw.update(resume_from=str(resume_from))
    p.fit(_substrate(), precomputed_edges_path=str(DATA / "edges-fixed15.npz"), random_state=SEED, verbose=False,
          warm_start_state=(init_state if warm else None), **kw)
    if warm: assert p.warm_start_sha256 == WARM_PARAM_SHA, "warm parameter hash mismatch"
    assert p._train_stats["positive_lr_optimizer_steps"] == steps
    assert p.model.proj_out.out_features == 3
    assert (getattr(p, "_pipeline_info", {}) or {}).get("x_residency") == "device_fp16", "pipeline not device_fp16"
    assert all(torch.isfinite(t).all() for t in p.model.state_dict().values())
    return V.state_sha(p.model.state_dict())


def main():
    ones = np.ones(V.N, np.float32); actual = np.load(DATA / "r_actual.npy").astype(np.float32)
    R = {"schema": "card018-canary-2026-09-12", "short_dose": SHORT, "ckpt_at": CKPT_AT}
    sha_base = _run("ordinary3d", None, SHORT, warm=True)
    sha_ones = _run("actual3d", ones, SHORT, warm=True)
    sha_act = _run("actual3d", actual, SHORT, warm=True)
    R["radius1_bitwise_baseline"] = bool(sha_ones == sha_base); R["actual_diverges"] = bool(sha_act != sha_base)

    import tempfile
    with tempfile.TemporaryDirectory(dir=str(SB)) as td:
        cdir = Path(td) / "ckpts"; cdir.mkdir(parents=True)
        sha_full = _run("actual3d", actual, SHORT, warm=True, ckpt_targets={CKPT_AT}, ckpt_dir=cdir)
        ck = cdir / f"ckpt-step{CKPT_AT}.pt"; assert ck.exists(), "mid-run step checkpoint not written"
        obj = torch.load(ck, map_location="cpu", weights_only=False)     # LOAD + assert, not existence alone
        R["ckpt_global_step"] = int(obj.get("global_step", -1))
        R["ckpt_step_in_range"] = bool(0 < R["ckpt_global_step"] < SHORT)
        R["ckpt_step_flag"] = bool(obj.get("step_checkpoint"))
        R["ckpt_identity_bound"] = obj.get("card012_identity") == _identity("actual3d", actual)
        R["ckpt_model_finite"] = all(bool(torch.isfinite(t).all()) for t in obj["model"].values())
        sha_resume = _run("actual3d", actual, SHORT, warm=True, resume_from=ck, ckpt_dir=cdir)
        R["resume_twin_bitwise"] = bool(sha_resume == sha_full)

        def _reject_specific(arm, radii):
            try:
                _run(arm, radii, SHORT, warm=True, resume_from=ck, ckpt_dir=cdir); return False
            except ValueError as e:
                return IDMISS in str(e)     # SPECIFIC admission-identity error, not any ValueError
            except Exception:
                return False
        R["wrong_arm_rejected"] = _reject_specific("ordinary3d", None)
        R["wrong_radius_rejected"] = _reject_specific("actual3d", ones)

    R["PASS"] = bool(R["radius1_bitwise_baseline"] and R["actual_diverges"] and R["ckpt_step_in_range"]
                     and R["ckpt_step_flag"] and R["ckpt_identity_bound"] and R["ckpt_model_finite"]
                     and R["resume_twin_bitwise"] and R["wrong_arm_rejected"] and R["wrong_radius_rejected"])
    (OC / "card018-canary.json").write_text(json.dumps(R, indent=1)); print(json.dumps(R, indent=1), flush=True)
    return 0 if R["PASS"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
