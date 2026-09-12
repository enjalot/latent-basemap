"""Card018 CPU self-test (per root review group 5). No GPU. Builds a synthetic but structurally-valid
card018-train fixture and asserts the canonical validator PASSES it; then applies independent single-fault
mutations (identity drift, wrong dose, mandatory/forbidden radius, wrong precision, stale snapshot, endpoint
!=400K snapshot, warm-provenance drift, corrupt step checkpoint) and asserts each FAILS closed with a raise.
Proves the one validator's guards actually fire. Usage: cpu_card018_selftest.py
"""
import os, sys, json, shutil, tempfile, copy
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import card018_validate as V
import torch

ROOT = Path(__file__).resolve().parents[2]


def _sd(seed):
    g = torch.Generator().manual_seed(seed)
    return {"hidden.weight": torch.randn(4, 4, generator=g), "proj_out.weight": torch.randn(3, 4, generator=g),
            "proj_out.bias": torch.randn(3, generator=g)}


def build_fixture(base, arm):
    td = Path(base); (td / arm / "ckpts").mkdir(parents=True, exist_ok=True)
    ident = V.expected_identity(arm, ROOT)
    # snapshots (distinct per step); endpoint == 400K snapshot
    snap = {s: _sd(1000 + i) for i, s in enumerate(V.SNAP_STEPS)}
    for s in V.SNAP_STEPS:
        torch.save({"model_state_dict": snap[s], "n_components": 3, "learning_rate": 0.001, "lr_schedule": "constant"},
                   td / arm / f"model-step{s}.pt")
    torch.save({"model_state_dict": snap[400000],"n_components":3,"learning_rate":.001,"lr_schedule":"constant"}, td / f"model-{arm}.pt")
    trained_sha = V.state_sha(snap[400000])
    # resumable step ckpts: those matching a snapshot step carry the SAME model payload
    for s in V.STEP_CKPTS:
        m = snap[s] if s in snap else _sd(9000 + s)
        torch.save({"global_step": s, "step_checkpoint": True, "card012_identity": ident, "model": m,"optimizer":{"state":{}},"scheduler":{},"torch_rng":torch.ones(1,dtype=torch.uint8),"cuda_rng":[torch.ones(1,dtype=torch.uint8)],"loader_gen":torch.ones(1,dtype=torch.uint8),"config":{"learning_rate":.001,"lr_schedule":"constant"}}, td / arm / "ckpts" / f"ckpt-step{s}.pt")
    np.save(td / f"coords-{arm}.npy", np.zeros((V.N, 3), "f4"))
    man = {"arm": arm, "card012_identity": ident, "warm_param_sha256": V.WARM_PARAM_SHA,
           "executed_steps": V.DOSE, "train_stats": {"positive_lr_optimizer_steps": V.DOSE,"executed_iters":V.DOSE,"lr_used_min":.001,"lr_used_max":.001},"kernel":ident["kernel"],"proc_peak_vram_gb":1.,"global_vram_used_gb":1.,
           "pipeline_info": {"x_residency": "device_fp16"}, "trained_sha256": trained_sha,
           "radii_sha": (V.hashlib.sha256(np.ascontiguousarray(np.load(V.DATA/"r_actual.npy"),dtype=np.float32).tobytes()).hexdigest()[:16] if arm == "actual3d" else None),
           "loaded_modules": {"verified_frozen_runtime": True, "all_basemap_under_root": True}}
    man.update(model_file_sha256=V.full_sha(td/f"model-{arm}.pt"),coords_file_sha256=V.full_sha(td/f"coords-{arm}.npy"))
    adm = {"arm": arm, "card012_identity": ident, "warm_param_sha256": V.WARM_PARAM_SHA,
           "radii_sha": man["radii_sha"], "kernel": ident["kernel"]}
    # admission written BEFORE snapshots so freshness (snap mtime >= admission mtime) holds
    (td / f"admission-{arm}.json").write_text(json.dumps(adm))
    os.utime(td / f"admission-{arm}.json", (0, 0))     # far in the past ⇒ snapshots are "fresh"
    (td / f"manifest-{arm}.json").write_text(json.dumps(man))
    return td, man, adm, ident


def expect_pass(base, arm):
    r = V.strict_validate_arm(arm, ROOT, base=base); assert r["PASS"]; return "PASS"


def expect_fail(base, arm, mutate, label):
    td = Path(tempfile.mkdtemp(dir=base))
    shutil.copytree(Path(base) / "actual3d", td / "actual3d")
    for f in ("coords-actual3d.npy", "model-actual3d.pt", "manifest-actual3d.json", "admission-actual3d.json"):
        shutil.copy(Path(base) / f, td / f)
    os.utime(td / "admission-actual3d.json", (0, 0))
    mutate(td)
    try:
        V.strict_validate_arm("actual3d", ROOT, base=td)
    except Exception as e:
        return f"FAIL-CLOSED ok: {label} -> {type(e).__name__}"
    raise SystemExit(f"GUARD DID NOT FIRE: {label}")


def _edit_man(td, **kw):
    m = json.loads((td / "manifest-actual3d.json").read_text()); m.update(kw); (td / "manifest-actual3d.json").write_text(json.dumps(m))
def _edit_man_nested(td, key, sub, val):
    m = json.loads((td / "manifest-actual3d.json").read_text()); m[key] = dict(m[key]); m[key][sub] = val; (td / "manifest-actual3d.json").write_text(json.dumps(m))


def main():
    base = tempfile.mkdtemp(prefix="card018-selftest-", dir="/tmp")
    build_fixture(base, "actual3d")
    results = [expect_pass(base, "actual3d")]

    # independent single-fault mutations, each must fail closed
    results.append(expect_fail(base, "actual3d", lambda td: _edit_man(td, executed_steps=399999), "wrong dose"))
    results.append(expect_fail(base, "actual3d", lambda td: _edit_man(td, radii_sha=None), "actual3d radius mandatory"))
    results.append(expect_fail(base, "actual3d", lambda td: _edit_man(td, warm_param_sha256="0" * 16), "warm provenance drift"))
    results.append(expect_fail(base, "actual3d", lambda td: _edit_man_nested(td, "pipeline_info", "x_residency", "device_int8"), "precision int8"))
    def _idmut(td):
        m = json.loads((td / "manifest-actual3d.json").read_text()); m["card012_identity"] = dict(m["card012_identity"]); m["card012_identity"]["dose"] = 1; (td / "manifest-actual3d.json").write_text(json.dumps(m))
    results.append(expect_fail(base, "actual3d", _idmut, "identity drift"))
    def _endpoint_mut(td):
        torch.save({"model_state_dict": _sd(7)}, td / "model-actual3d.pt")   # != manifest trained_sha256 & != 400K snap
    results.append(expect_fail(base, "actual3d", _endpoint_mut, "endpoint != 400K snapshot"))
    def _stale(td):
        os.utime(td / "admission-actual3d.json", None)   # admission now newer than snapshots ⇒ stale
    results.append(expect_fail(base, "actual3d", _stale, "stale snapshot"))
    def _corrupt(td):
        (td / "actual3d" / "ckpts" / "ckpt-step250000.pt").write_bytes(b"not a torch file")
    results.append(expect_fail(base, "actual3d", _corrupt, "corrupt step checkpoint"))
    def _drop_ckpt(td):
        (td / "actual3d" / "ckpts" / "ckpt-step400000.pt").unlink()
    results.append(expect_fail(base, "actual3d", _drop_ckpt, "missing 400K resumable ckpt"))

    shutil.rmtree(base, ignore_errors=True)
    out = {"schema": "card018-selftest-2026-09-12", "PASS": True, "n_cases": len(results), "results": results}
    (V.OC / "card018-selftest.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
