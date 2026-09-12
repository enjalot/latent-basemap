"""Card024 canonical-validator self-test (CPU, no GPU). Builds structurally-valid umap_uniform + nce_learned
fixtures and asserts strict_validate_arm PASSES them, then applies independent single-fault mutations that must
each FAIL closed: wrong dose, warm-init drift, precision int8, identity drift, endpoint != 60K snapshot, stale
snapshot, corrupt step ckpt, missing resumable ckpt, and (nce_learned) an unmoved scalar + a step ckpt missing
the learned scalar. Usage: card024_selftest.py
"""
import os, sys, json, shutil, tempfile
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import card024_validate as V
import torch

ROOT = Path(__file__).resolve().parents[2]
V.N = 64  # tiny structurally-identical fixtures; production validator keeps 300K


def _sd(seed):
    g = torch.Generator().manual_seed(seed)
    return {"hidden.weight": torch.randn(4, 4, generator=g), "proj_out.weight": torch.randn(2, 4, generator=g),
            "proj_out.bias": torch.randn(2, generator=g)}


def build(base, arm):
    td = Path(base); (td / arm / "ckpts").mkdir(parents=True, exist_ok=True)
    ident = V.expected_identity(arm, ROOT)
    snap = {s: _sd(1000 + i) for i, s in enumerate(V.SNAP_STEPS)}
    for s in V.SNAP_STEPS:
        torch.save({"model_state_dict": snap[s], "n_components": V.NC, "learning_rate": V.LR, "lr_schedule": "constant"}, td / arm / f"model-step{s}.pt")
    torch.save({"model_state_dict": snap[60000]}, td / f"model-{arm}.pt")
    for s in V.STEP_CKPTS:
        ck = {"global_step": s, "step_checkpoint": True, "card012_identity": ident, "model": (snap[s] if s in snap else _sd(9000 + s))}
        if arm == "nce_learned": ck["card024_beta"] = torch.tensor(0.05)
        torch.save(ck, td / arm / "ckpts" / f"ckpt-step{s}.pt")
    np.save(td / f"coords-{arm}.npy", np.zeros((V.N, V.NC), "f4"))
    fb = (0.05 if arm == "nce_learned" else (0.0 if arm == "neg_fixed" else None))
    traj = ([{"step": s, "beta": 0.01 * i} for i, s in enumerate(V.SNAP_STEPS, 1)] if arm == "nce_learned" else [])
    man = {"arm": arm, "mode": V.MODE[arm], "card012_identity": ident, "warm_init_sha256": V.INIT_SHA,
           "executed_steps": V.DOSE, "train_stats": {"positive_lr_optimizer_steps": V.DOSE, "card024_beta_traj": traj},
           "lr_used_min": V.LR, "lr_used_max": V.LR, "final_beta": fb, "pipeline_info": {"x_residency": "device_fp16"},
           "trained_sha256": V.state_sha(snap[60000]), "loaded_modules": {"verified_frozen_runtime": True, "all_basemap_under_root": True}}
    (td / f"admission-{arm}.json").write_text(json.dumps({"arm": arm, "card012_identity": ident, "warm_init_sha256": V.INIT_SHA}))
    os.utime(td / f"admission-{arm}.json", (0, 0)); (td / f"manifest-{arm}.json").write_text(json.dumps(man))
    return td


def expect_fail(base, arm, mutate, label):
    td = Path(tempfile.mkdtemp(dir=base)); shutil.copytree(Path(base) / arm, td / arm)
    for f in (f"coords-{arm}.npy", f"model-{arm}.pt", f"manifest-{arm}.json", f"admission-{arm}.json"):
        shutil.copy(Path(base) / f, td / f)
    os.utime(td / f"admission-{arm}.json", (0, 0)); mutate(td)
    try:
        V.strict_validate_arm(arm, ROOT, base=td)
    except Exception as e:
        return f"FAIL-CLOSED ok: {label} -> {type(e).__name__}"
    raise SystemExit(f"GUARD DID NOT FIRE: {label}")


def _man(td, arm, **kw):
    m = json.loads((td / f"manifest-{arm}.json").read_text()); m.update(kw); (td / f"manifest-{arm}.json").write_text(json.dumps(m))


def main():
    base = tempfile.mkdtemp(prefix="card024-selftest-", dir="/tmp"); results = []
    build(base, "umap_uniform")
    assert V.strict_validate_arm("umap_uniform", ROOT, base=base)["PASS"]; results.append("PASS umap_uniform valid")
    results.append(expect_fail(base, "umap_uniform", lambda td: _man(td, "umap_uniform", executed_steps=59999), "wrong dose"))
    results.append(expect_fail(base, "umap_uniform", lambda td: _man(td, "umap_uniform", warm_init_sha256="0" * 16), "warm-init drift"))
    results.append(expect_fail(base, "umap_uniform", lambda td: _man(td, "umap_uniform", pipeline_info={"x_residency": "device_int8"}), "precision int8"))
    def _idmut(td):
        m = json.loads((td / "manifest-umap_uniform.json").read_text()); m["card012_identity"] = dict(m["card012_identity"]); m["card012_identity"]["dose"] = 1; (td / "manifest-umap_uniform.json").write_text(json.dumps(m))
    results.append(expect_fail(base, "umap_uniform", _idmut, "identity drift"))
    results.append(expect_fail(base, "umap_uniform", lambda td: torch.save({"model_state_dict": _sd(7)}, td / "model-umap_uniform.pt"), "endpoint != 60K snapshot"))
    results.append(expect_fail(base, "umap_uniform", lambda td: os.utime(td / "admission-umap_uniform.json", None), "stale snapshot"))
    results.append(expect_fail(base, "umap_uniform", lambda td: (td / "umap_uniform/ckpts/ckpt-step40000.pt").write_bytes(b"garbage"), "corrupt step ckpt"))
    results.append(expect_fail(base, "umap_uniform", lambda td: (td / "umap_uniform/ckpts/ckpt-step60000.pt").unlink(), "missing 60K resumable ckpt"))

    build(base, "nce_learned")
    assert V.strict_validate_arm("nce_learned", ROOT, base=base)["PASS"]; results.append("PASS nce_learned valid")
    results.append(expect_fail(base, "nce_learned", lambda td: _man(td, "nce_learned", final_beta=0.0), "nce scalar never moved"))
    def _drop_beta(td):
        p = td / "nce_learned/ckpts/ckpt-step40000.pt"; ck = torch.load(p, map_location="cpu", weights_only=False); ck.pop("card024_beta", None); torch.save(ck, p)
    results.append(expect_fail(base, "nce_learned", _drop_beta, "nce step ckpt missing scalar"))

    shutil.rmtree(base, ignore_errors=True)
    out = {"schema": "card024-selftest-2026-09-12", "PASS": True, "n_cases": len(results), "results": results}
    (V.OC / "card024-selftest.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
