"""Card024 device canary (real 300K fixed15 path, short dose) — per card024 "Validation before GPU
production". Proves, before production (root runs on GPU):
  1. DEFAULT-OFF bitwise identity: mode unset == mode 'umap_uniform' (both plain BCE, no scalar, no extra
     terms) — the card024 code present but off leaves the path unchanged;
  2. beta0 reproduces NEG exactly in MODEL gradients: a 1-step neg_fixed and a 1-step nce_learned reach a
     BITWISE-identical model (beta=0 at step 0 ⇒ identical model gradient), and the nce scalar has moved;
  3. ON divergence: over a short run nce_learned diverges from neg_fixed (the learned scalar is active);
  4. GENUINE resume twin: an nce_learned step-checkpointed run resumed mid-flight reaches a BITWISE-identical
     endpoint AND recovers the scalar; the checkpoint carries card024_beta and the beta optimizer group has
     weight_decay 0;
  5. wrong-family resume FAILS CLOSED on the specific admission-identity error before state restoration.
Exit 0 = PASS. Usage: gpu_card024_canary.py
"""
import os, sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import card024_validate as V
import torch
from basemap.pumap.parametric_umap.core import ParametricUMAP

SB = V.SB; OC = V.OC; CHAMPION = V.CHAMPION; SUB = V.SUB; GRAPH = V.GRAPH; INIT = V.INIT
SEED = V.SEED; IDMISS = "admission-identity mismatch"
_X = None; _WARM = None


def _prep():
    global _X, _WARM
    if _X is None: _X = np.asarray(np.load(SUB, mmap_mode="r"), np.float32)
    if _WARM is None: _WARM = torch.load(str(INIT), map_location="cpu", weights_only=False)["model_state"]


def _identity(mode, seed): return {"card": "card024-canary", "mode": mode, "seed": int(seed)}


def _run(mode, short, seed=SEED, resume_from=None, ckpt_targets=None, ckpt_dir=None):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    p = ParametricUMAP.load(str(CHAMPION), device="cuda"); p.model = None; p.n_components = V.NC
    p.learning_rate = V.LR; p.lr_schedule = "constant"; p.batch_size = V.BATCH; p.warmup_steps = 0
    p.n_epochs = 100000; p.rankneg_window = 0; p._max_train_steps = short
    p.x_residency = "auto"; p.required_input_pipeline = "device"
    for a, v in (("anchor_ids_path", ""), ("anchor_hold_weight", 0.0), ("replay_bank_path", ""),
                 ("replay_weight", 0.0), ("deriv_bank_path", ""), ("deriv_weight", 0.0),
                 ("fneg_weight", 0.0), ("neg_tanh_gamma", 0.0), ("midnear_enabled", False),
                 ("density_weight", 0.0), ("correlation_weight", 0.0)):
        if hasattr(p, a): setattr(p, a, v)
    if mode is not None: p._card024_mode = mode
    p._card012_identity = _identity(mode, seed)
    if ckpt_targets: p._checkpoint_step_targets = set(ckpt_targets)
    kw = {}
    if ckpt_dir: kw.update(checkpoint_dir=str(ckpt_dir), checkpoint_every_epochs=1)
    if resume_from: kw.update(resume_from=str(resume_from))
    p.fit(_X, precomputed_edges_path=str(GRAPH), random_state=seed, verbose=False,
          warm_start_state=(None if resume_from else _WARM), **kw)
    assert p._train_stats["positive_lr_optimizer_steps"] == short
    assert (getattr(p, "_pipeline_info", {}) or {}).get("x_residency") == "device_fp16", "pipeline not device_fp16"
    assert all(torch.isfinite(t).all() for t in p.model.state_dict().values())
    _b = getattr(p, "_card024_beta", None)
    return V.state_sha(p.model.state_dict()), (float(_b.detach()) if isinstance(_b, torch.Tensor) else None), p


def main():
    _prep(); R = {"schema": "card024-canary-2026-09-12"}
    # (1) default-off bitwise identity
    sha_off, _, _ = _run(None, 60); sha_uu, _, _ = _run("umap_uniform", 60)
    R["off_bitwise_identical"] = bool(sha_off == sha_uu)
    # (2) beta0 == NEG in model gradients (1 step) + scalar moved
    sha_neg1, _, _ = _run("neg_fixed", 1); sha_nce1, beta_nce1, _ = _run("nce_learned", 1)
    R["beta0_model_grad_equals_neg"] = bool(sha_neg1 == sha_nce1)
    R["nce_scalar_moved_after_step"] = bool(beta_nce1 is not None and abs(beta_nce1) > 0)
    # (3) ON divergence over a short run
    sha_neg60, _, _ = _run("neg_fixed", 60); sha_nce60, _, _ = _run("nce_learned", 60)
    R["on_diverges"] = bool(sha_neg60 != sha_nce60)

    import tempfile
    with tempfile.TemporaryDirectory(dir=str(SB)) as td:
        cdir = Path(td) / "ckpts"; cdir.mkdir(parents=True)
        sha_full, beta_full, _ = _run("nce_learned", 120, ckpt_targets={60}, ckpt_dir=cdir)
        ck = cdir / "ckpt-step60.pt"; assert ck.exists(), "mid-run step checkpoint not written"
        o = torch.load(ck, map_location="cpu", weights_only=False)
        R["ckpt_step_in_range"] = bool(0 < int(o.get("global_step", -1)) < 120)
        R["ckpt_has_scalar"] = o.get("card024_beta") is not None
        R["ckpt_identity_bound"] = o.get("card012_identity") == _identity("nce_learned", SEED)
        # scalar zero weight decay in the beta optimizer group (last group holds the scalar)
        wds = [g.get("weight_decay") for g in o["optimizer"]["param_groups"]]
        R["scalar_zero_weight_decay"] = bool(wds and wds[-1] == 0.0)
        sha_resume, beta_resume, _ = _run("nce_learned", 120, resume_from=ck, ckpt_dir=cdir)
        R["resume_twin_bitwise"] = bool(sha_resume == sha_full)
        R["resume_recovers_scalar"] = bool(beta_resume is not None and beta_full is not None and abs(beta_resume - beta_full) < 1e-9)

        def _reject(mode):
            try:
                _run(mode, 120, resume_from=ck, ckpt_dir=cdir); return False
            except ValueError as e:
                return IDMISS in str(e)
            except Exception:
                return False
        R["wrong_family_neg_rejected"] = _reject("neg_fixed")
        R["wrong_family_umap_rejected"] = _reject("umap_uniform")

    keys = ["off_bitwise_identical", "beta0_model_grad_equals_neg", "nce_scalar_moved_after_step", "on_diverges",
            "ckpt_step_in_range", "ckpt_has_scalar", "ckpt_identity_bound", "scalar_zero_weight_decay",
            "resume_twin_bitwise", "resume_recovers_scalar", "wrong_family_neg_rejected", "wrong_family_umap_rejected"]
    R["PASS"] = bool(all(R[k] for k in keys))
    (OC / "card024-canary.json").write_text(json.dumps(R, indent=1)); print(json.dumps(R, indent=1), flush=True)
    return 0 if R["PASS"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
