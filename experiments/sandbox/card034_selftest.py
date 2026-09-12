"""Card034 canonical-validator self-test (CPU, no GPU). Builds structurally-valid grouped_umap + grouped_nce
+ grouped_infonce fixtures and asserts strict_validate_arm PASSES them, then applies independent single-fault
mutations that must each FAIL closed: wrong dose, warm-init drift, precision, identity drift, endpoint != 60K
snapshot, stale snapshot, corrupt step ckpt, missing resumable ckpt, missing epoch ckpt, 9:1 exposure
violation, and (grouped_nce) unmoved scalar + a step ckpt missing the scalar. Usage: card034_selftest.py
"""
import os, sys, json, shutil, tempfile
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import card034_validate as V
import torch

ROOT = Path(__file__).resolve().parents[2]
V.N = 64  # tiny structurally-identical fixtures; production validator keeps 300K


def _sd(seed):
    g = torch.Generator().manual_seed(seed)
    return {"proj_out.weight": torch.randn(2, 4, generator=g), "proj_out.bias": torch.randn(2, generator=g)}


def build(base, arm):
    td = Path(base); (td / arm / "ckpts").mkdir(parents=True, exist_ok=True)
    ident = V.expected_identity(arm, ROOT)
    snap = {s: _sd(1000 + i) for i, s in enumerate(V.SNAP_STEPS)}
    for s in V.SNAP_STEPS:
        torch.save({"model_state_dict": snap[s], "n_components": V.NC}, td / arm / f"model-step{s}.pt")
    torch.save({"model_state_dict": snap[60000]}, td / f"model-{arm}.pt")
    ss = {"perm": np.arange(10), "cursor": 0, "epoch": 1}
    for s in V.STEP_CKPTS:
        ck = {"schema": "card034-ckpt-2026-09-12", "arm": arm, "global_step": s, "epoch": 1, "step_checkpoint": True,
              "identity": ident, "coeff": ident.get("infonce_coeff") or 1.0, "model": (snap[s] if s in snap else _sd(9000 + s)),
              "optimizer": {"state": {}, "param_groups": []}, "scaler": {}, "sampler_state": ss,
              "beta": (torch.tensor(0.03) if arm == "grouped_nce" else None)}
        torch.save(ck, td / arm / "ckpts" / f"ckpt-step{s}.pt")
    torch.save({"global_step": 2747, "epoch": 1, "identity": ident, "model": _sd(5), "optimizer": {"state": {}, "param_groups": []},
                "scaler": {}, "sampler_state": ss, "step_checkpoint": False, "beta": (torch.tensor(0.03) if arm == "grouped_nce" else None)},
               td / arm / "ckpts" / "ckpt-epoch1.pt")
    np.save(td / f"coords-{arm}.npy", np.zeros((V.N, V.NC), "f4"))
    fb = (0.03 if arm == "grouped_nce" else None)
    man = {"arm": arm, "mode": V.MODE[arm], "identity": ident, "warm_init_sha256": V.INIT_SHA, "executed_steps": V.DOSE,
           "train_stats": {"positive_lr_optimizer_steps": V.DOSE, "exposure_probes": [{"step": 1, "noise_per_pos": 9}, {"step": 30000, "noise_per_pos": 9}, {"step": 60000, "noise_per_pos": 9}]},
           "lr_used_min": V.LR, "lr_used_max": V.LR, "weight_decay": V.WEIGHT_DECAY, "grad_clip": V.GRAD_CLIP,
           "final_beta": fb, "infonce_coeff": ident.get("infonce_coeff"), "pipeline": "device_fp16",
           "trained_sha256": V.state_sha(snap[60000]), "loaded_modules": {"verified_frozen_runtime": True, "all_basemap_under_root": True}}
    (td / f"admission-{arm}.json").write_text(json.dumps({"arm": arm, "identity": ident, "warm_init_sha256": V.INIT_SHA}))
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
    base = tempfile.mkdtemp(prefix="card034-selftest-", dir="/tmp"); results = []
    for arm in V.ARMS:
        build(base, arm); assert V.strict_validate_arm(arm, ROOT, base=base)["PASS"]; results.append(f"PASS {arm} valid")
    A = "grouped_umap"
    results.append(expect_fail(base, A, lambda td: _man(td, A, executed_steps=59999), "wrong dose"))
    results.append(expect_fail(base, A, lambda td: _man(td, A, warm_init_sha256="0" * 16), "warm-init drift"))
    results.append(expect_fail(base, A, lambda td: _man(td, A, pipeline="host_int8"), "precision"))
    def _idmut(td):
        m = json.loads((td / f"manifest-{A}.json").read_text()); m["identity"] = dict(m["identity"]); m["identity"]["dose"] = 1; (td / f"manifest-{A}.json").write_text(json.dumps(m))
    results.append(expect_fail(base, A, _idmut, "identity drift"))
    results.append(expect_fail(base, A, lambda td: torch.save({"model_state_dict": _sd(7)}, td / f"model-{A}.pt"), "endpoint != 60K snapshot"))
    results.append(expect_fail(base, A, lambda td: os.utime(td / f"admission-{A}.json", None), "stale snapshot"))
    results.append(expect_fail(base, A, lambda td: (td / f"{A}/ckpts/ckpt-step40000.pt").write_bytes(b"garbage"), "corrupt step ckpt"))
    results.append(expect_fail(base, A, lambda td: (td / f"{A}/ckpts/ckpt-step60000.pt").unlink(), "missing 60K resumable ckpt"))
    results.append(expect_fail(base, A, lambda td: (td / f"{A}/ckpts/ckpt-epoch1.pt").unlink(), "missing epoch ckpt"))
    def _exp(td):
        m = json.loads((td / f"manifest-{A}.json").read_text()); m["train_stats"]["exposure_probes"][1]["noise_per_pos"] = 8; (td / f"manifest-{A}.json").write_text(json.dumps(m))
    results.append(expect_fail(base, A, _exp, "9:1 exposure violated"))
    NC = "grouped_nce"
    results.append(expect_fail(base, NC, lambda td: _man(td, NC, final_beta=0.0), "nce scalar never moved"))
    def _drop_beta(td):
        p = td / f"{NC}/ckpts/ckpt-step40000.pt"; ck = torch.load(p, map_location="cpu", weights_only=False); ck["beta"] = None; torch.save(ck, p)
    results.append(expect_fail(base, NC, _drop_beta, "nce step ckpt missing scalar"))

    shutil.rmtree(base, ignore_errors=True)
    out = {"schema": "card034-selftest-2026-09-12", "PASS": True, "n_cases": len(results), "results": results}
    (V.OC / "card034-selftest.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
