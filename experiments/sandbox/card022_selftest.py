"""Card022 canonical-validator self-test (per card022 "validator corruption tests as feasible without GPU").
No GPU. Builds structurally-valid card022-train fixtures and asserts card022_validate.strict_validate_arm
PASSES them, then applies independent single-fault mutations that must each FAIL closed: wrong dose, teacher/
warm-provenance drift, wrong precision, identity drift, endpoint != 60K snapshot, stale snapshot, corrupt step
checkpoint, missing resumable ckpt, and (shape_floor) zero weight + a step ckpt missing the bank-sampler RNG.
The shape_floor cases use a temp calibration file via a monkeypatched V.CALIB so no GPU output is needed.
Usage: card022_selftest.py
"""
import os, sys, json, shutil, tempfile
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import card022_validate as V
import torch

ROOT = Path(__file__).resolve().parents[2]


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
        if arm == "shape_floor": ck["shape_gen"] = torch.zeros(8, dtype=torch.uint8)   # bank-sampler RNG present
        torch.save(ck, td / arm / "ckpts" / f"ckpt-step{s}.pt")
    np.save(td / f"coords-{arm}.npy", np.zeros((V.N, V.NC), "f4"))
    man = {"arm": arm, "card012_identity": ident, "teacher_sha256": V.TEACHER_SHA, "executed_steps": V.DOSE,
           "train_stats": {"positive_lr_optimizer_steps": V.DOSE}, "lr_used_min": V.LR, "lr_used_max": V.LR,
           "pipeline_info": {"x_residency": "device_fp16"}, "trained_sha256": V.state_sha(snap[60000]),
           "shape_weight": (ident["shape_weight"] if arm == "shape_floor" else 0.0),
           "loaded_modules": {"verified_frozen_runtime": True, "all_basemap_under_root": True}}
    (td / f"admission-{arm}.json").write_text(json.dumps({"arm": arm, "card012_identity": ident, "teacher_sha256": V.TEACHER_SHA}))
    os.utime(td / f"admission-{arm}.json", (0, 0))
    (td / f"manifest-{arm}.json").write_text(json.dumps(man))
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
    base = tempfile.mkdtemp(prefix="card022-selftest-", dir="/tmp"); results = []
    # ordinary arm: calibration-free general guards
    build(base, "ordinary")
    assert V.strict_validate_arm("ordinary", ROOT, base=base)["PASS"]; results.append("PASS ordinary valid")
    results.append(expect_fail(base, "ordinary", lambda td: _man(td, "ordinary", executed_steps=59999), "wrong dose"))
    results.append(expect_fail(base, "ordinary", lambda td: _man(td, "ordinary", teacher_sha256="0" * 64), "teacher/warm drift"))
    results.append(expect_fail(base, "ordinary", lambda td: _man(td, "ordinary", pipeline_info={"x_residency": "device_int8"}), "precision int8"))
    results.append(expect_fail(base, "ordinary", lambda td: _man(td, "ordinary", shape_weight=0.5), "ordinary carries shape weight"))
    def _idmut(td):
        m = json.loads((td / "manifest-ordinary.json").read_text()); m["card012_identity"] = dict(m["card012_identity"]); m["card012_identity"]["dose"] = 1; (td / "manifest-ordinary.json").write_text(json.dumps(m))
    results.append(expect_fail(base, "ordinary", _idmut, "identity drift"))
    results.append(expect_fail(base, "ordinary", lambda td: torch.save({"model_state_dict": _sd(7)}, td / "model-ordinary.pt"), "endpoint != 60K snapshot"))
    results.append(expect_fail(base, "ordinary", lambda td: os.utime(td / "admission-ordinary.json", None), "stale snapshot"))
    results.append(expect_fail(base, "ordinary", lambda td: (td / "ordinary/ckpts/ckpt-step40000.pt").write_bytes(b"garbage"), "corrupt step ckpt"))
    results.append(expect_fail(base, "ordinary", lambda td: (td / "ordinary/ckpts/ckpt-step60000.pt").unlink(), "missing 60K resumable ckpt"))

    # shape_floor arm: temp calibration via monkeypatched V.CALIB (no GPU output needed)
    calib = Path(base) / "calib.json"; calib.write_text(json.dumps({"PASS": True, "coefficient": 0.037}))
    V.CALIB = calib
    build(base, "shape_floor")
    assert V.strict_validate_arm("shape_floor", ROOT, base=base)["PASS"]; results.append("PASS shape_floor valid")
    results.append(expect_fail(base, "shape_floor", lambda td: _man(td, "shape_floor", shape_weight=0.0), "shape_floor zero weight"))
    def _drop_gen(td):
        p = td / "shape_floor/ckpts/ckpt-step40000.pt"; ck = torch.load(p, map_location="cpu", weights_only=False); ck.pop("shape_gen", None); torch.save(ck, p)
    results.append(expect_fail(base, "shape_floor", _drop_gen, "step ckpt missing bank-sampler RNG"))

    shutil.rmtree(base, ignore_errors=True)
    out = {"schema": "card022-selftest-2026-09-12", "PASS": True, "n_cases": len(results), "results": results}
    (V.OC / "card022-selftest.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
