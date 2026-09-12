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
V.N = 64  # structurally identical tiny fixtures; production validator retains300K


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
        m=ck["model"]; ids=list(range(len(m)))
        ck.update(schema="pumap-ckpt-2026-08-30", epoch=0,
                  config={"batch_size":V.BATCH,"random_state":V.SEED,"architecture":"residual_bottleneck","learning_rate":V.LR,"lr_schedule":"constant","replay_enabled":False,"deriv_enabled":False},
                  train_stats={"executed_iters":s,"positive_lr_optimizer_steps":s,"optimizer_steps_succeeded":s,"shape_successful_steps":s if arm=="shape_floor" else 0},
                  optimizer={"param_groups":[{"params":ids,"lr":V.LR,"betas":(.9,.999),"eps":1e-8,"weight_decay":.01,"amsgrad":False}],"state":{i:{"step":torch.tensor(float(s)),"exp_avg":torch.zeros_like(v),"exp_avg_sq":torch.zeros_like(v)} for i,v in zip(ids,m.values())}},
                  scheduler={"base_lrs":[V.LR],"last_epoch":s,"_step_count":s+1,"_last_lr":[V.LR],"lr_lambdas":[None]},
                  torch_rng=torch.Generator().manual_seed(0).get_state(),cuda_rng=[torch.zeros(16,dtype=torch.uint8)],loader_gen=torch.zeros(16,dtype=torch.uint8),
                  loader_perm=torch.arange(15*V.N),loader_pos_idx=0,loader_batch_no=s,loader_rank_of_node=torch.arange(V.N),loader_node_at_rank=torch.arange(V.N),rankneg_scale=1.0)
        if arm == "shape_floor": ck["shape_gen"] = torch.zeros(16, dtype=torch.uint8)   # bank-sampler RNG present
        torch.save(ck, td / arm / "ckpts" / f"ckpt-step{s}.pt")
    np.save(td / f"coords-{arm}.npy", np.zeros((V.N, V.NC), "f4"))
    man = {"arm": arm, "card012_identity": ident, "teacher_sha256": V.TEACHER_SHA, "executed_steps": V.DOSE,
           "warm_start_param_sha256": V.WARM_PARAM_SHA, "train_stats": {"positive_lr_optimizer_steps": V.DOSE, "shape_successful_steps":V.DOSE if arm=="shape_floor" else 0,"shape_clouds_successful":V.DOSE*V.CENTERS_PER_STEP if arm=="shape_floor" else 0,"shape_positive_loss_steps":V.DOSE if arm=="shape_floor" else 0,"shape_loss_sum":1.0 if arm=="shape_floor" else 0.0}, "lr_used_min": V.LR, "lr_used_max": V.LR,
           "pipeline_info": {"x_residency": "device_fp16"}, "trained_sha256": V.state_sha(snap[60000]),
           "shape_weight": (ident["shape_weight"] if arm == "shape_floor" else 0.0),
           "loaded_modules": {"verified_frozen_runtime": True, "all_basemap_under_root": True}}
    (td / f"admission-{arm}.json").write_text(json.dumps({"arm": arm, "card012_identity": ident, "teacher_sha256": V.TEACHER_SHA,"warm_start_param_sha256":V.WARM_PARAM_SHA}))
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

    def ckmut(td, fn):
        p=td/"shape_floor/ckpts/ckpt-step40000.pt";c=torch.load(p,map_location="cpu",weights_only=False);fn(c);torch.save(c,p)
    cases=[
       ("missing optimizer",lambda c:c.pop("optimizer")),
       ("empty Adam state",lambda c:c["optimizer"].update(state={})),
       ("wrong Adam moment",lambda c:c["optimizer"]["state"][0].update(exp_avg=torch.zeros(2))),
       ("missing scheduler",lambda c:c.pop("scheduler")),
       ("wrong scheduler LR",lambda c:c["scheduler"].update(_last_lr=[.1])),
       ("bad CPU RNG",lambda c:c.update(torch_rng=torch.zeros(2,dtype=torch.uint8))),
       ("bad shape RNG",lambda c:c.update(shape_gen=torch.zeros(8,dtype=torch.uint8))),
       ("missing loader permutation",lambda c:c.pop("loader_perm")),
       ("wrong rank inverse",lambda c:c.update(loader_node_at_rank=torch.zeros(V.N,dtype=torch.int64))),
       ("wrong shape exposure",lambda c:c["train_stats"].update(shape_successful_steps=0))]
    for label,fn in cases:results.append(expect_fail(base,"shape_floor",lambda td,fn=fn:ckmut(td,fn),label))
    results.append(expect_fail(base,"shape_floor",lambda td:_man(td,"shape_floor",warm_start_param_sha256="wrong"),"actual warm mismatch"))
    results.append(expect_fail(base,"shape_floor",lambda td:_man(td,"shape_floor",train_stats={"positive_lr_optimizer_steps":V.DOSE}),"missing actual exposure"))
    shutil.rmtree(base, ignore_errors=True)
    out = {"schema": "card022-selftest-2026-09-12", "PASS": True, "n_cases": len(results), "results": results}
    (V.OC / "card022-selftest.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
