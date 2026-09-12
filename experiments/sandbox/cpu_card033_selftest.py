"""Card033 CPU self-test (per root review group 5). No GPU. Builds a synthetic but structurally-valid
card033-train fixture and asserts the canonical validator PASSES it; then applies independent single-fault
mutations (identity drift, wrong dose, mandatory/forbidden radius, wrong precision, stale snapshot, endpoint
!=800K snapshot, warm-provenance drift, corrupt step checkpoint) and asserts each FAILS closed with a raise.
Proves the one validator's guards actually fire. Usage: cpu_card033_selftest.py
"""
import os, sys, json, shutil, tempfile, copy
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import card033_validate as V
import torch

ROOT = Path(__file__).resolve().parents[2]


def _sd(seed):
    g = torch.Generator().manual_seed(seed)
    return {"hidden.weight": torch.randn(4, 4, generator=g), "proj_out.weight": torch.randn(3, 4, generator=g),
            "proj_out.bias": torch.randn(3, generator=g)}


def build_fixture(base, arm):
    td = Path(base); (td / arm / "ckpts").mkdir(parents=True, exist_ok=True)
    ident = V.expected_identity(arm, ROOT)
    # snapshots (distinct per step); endpoint == 800K snapshot
    snap = {s: _sd(1000 + i) for i, s in enumerate(V.SNAP_STEPS)}
    for s in V.SNAP_STEPS:
        torch.save({"model_state_dict": snap[s], "n_components": 3, "learning_rate": 0.001, "lr_schedule": "constant"},
                   td / arm / f"model-step{s}.pt")
    torch.save({"model_state_dict": snap[800000],"n_components":3,"learning_rate":.001,"lr_schedule":"constant"}, td / f"model-{arm}.pt")
    trained_sha = V.state_sha(snap[800000])
    # resumable step ckpts: those matching a snapshot step carry the SAME model payload
    for s in V.STEP_CKPTS:
        m = snap[s] if s in snap else _sd(9000 + s)
        parameters=[torch.nn.Parameter(v.clone()) for v in m.values()]
        opt=torch.optim.AdamW(parameters,lr=V.LR);sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
        ost=opt.state_dict();ost['state']={i:{'step':torch.tensor(float(s)),'exp_avg':torch.zeros_like(v),'exp_avg_sq':torch.zeros_like(v)} for i,v in enumerate(m.values())}
        torch.save({"schema":"pumap-ckpt-2026-08-30","epoch":1,"global_step":s,"step_checkpoint":True,"card012_identity":ident,"model":m,
                    "optimizer":ost,"scheduler":sch.state_dict(),"torch_rng":torch.get_rng_state(),"cuda_rng":[torch.arange(16,dtype=torch.uint8)],"loader_gen":torch.arange(16,dtype=torch.uint8),
                    "scaler":{"scale":1024.,"growth_factor":2.,"backoff_factor":.5,"growth_interval":2000,"_growth_tracker":1},
                    "config":{"learning_rate":V.LR,"lr_schedule":"constant","batch_size":V.BATCH,"random_state":V.SEED,"architecture":"residual_bottleneck","replay_enabled":False,"deriv_enabled":False},
                    "train_stats":{"executed_iters":s,"positive_lr_optimizer_steps":s,"optimizer_steps_succeeded":s},
                    "loader_perm":torch.arange(V.N*15),"loader_pos_idx":0,"loader_batch_no":0,"loader_rank_of_node":torch.arange(V.N),"loader_node_at_rank":torch.arange(V.N),"rankneg_scale":1.},td/arm/"ckpts"/f"ckpt-step{s}.pt")
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
    torch.set_num_threads(2)
    original_n=V.N;V.N=32 # Scaled schema fixture; real production validator remains N=4M.
    base = tempfile.mkdtemp(prefix="card033-selftest-", dir="/tmp")
    # Only the validator fixture uses synthetic graph/radius metadata; production admission remains untouched.
    saved=(V.DATA,V.DATA_MANIFEST,V._GRAPH,V.EDGES_SHA256,V.R_ACTUAL_SHA256)
    V.DATA=Path(base)/"fixture-data";V.DATA.mkdir();np.save(V.DATA/"r_actual.npy",np.ones(32,"f4"))
    V.DATA_MANIFEST=V.DATA/"manifest.json";V.DATA_MANIFEST.write_text('{"PASS":true,"fixture":true}')
    V._GRAPH={"PASS":True};V.EDGES_SHA256="e"*64;V.R_ACTUAL_SHA256=V.full_sha(V.DATA/"r_actual.npy")
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
        torch.save({"model_state_dict": _sd(7),"n_components":3,"learning_rate":.001,"lr_schedule":"constant"}, td / "model-actual3d.pt")   # != manifest trained_sha256 & != 800K snap
    results.append(expect_fail(base, "actual3d", _endpoint_mut, "endpoint != 800K snapshot"))
    def _stale(td):
        os.utime(td / "admission-actual3d.json", None)   # admission now newer than snapshots ⇒ stale
    results.append(expect_fail(base, "actual3d", _stale, "stale snapshot"))
    def _corrupt(td):
        (td / "actual3d" / "ckpts" / "ckpt-step300000.pt").write_bytes(b"not a torch file")
    results.append(expect_fail(base, "actual3d", _corrupt, "corrupt step checkpoint"))
    def _drop_ckpt(td):
        (td / "actual3d" / "ckpts" / "ckpt-step800000.pt").unlink()
    results.append(expect_fail(base, "actual3d", _drop_ckpt, "missing 800K resumable ckpt"))

    def ck_mut(td,fn):
        p=td/"actual3d"/"ckpts"/"ckpt-step300000.pt";z=torch.load(p,map_location="cpu",weights_only=False);fn(z);torch.save(z,p)
    for label,fn in [
        ("wrong negative recipe",lambda z:z["card012_identity"].update(fneg_weight=0.0)),
        ("wrong resident budget",lambda z:z["card012_identity"].update(gpu_resident_vram_budget_gb=9.0)),
        ("missing AMP scaler",lambda z:z.pop("scaler")),
        ("empty Adam",lambda z:z.update(optimizer={"state":{},"param_groups":[]})),
        ("missing scheduler",lambda z:z.update(scheduler={})),
        ("invalid CPU RNG",lambda z:z.update(torch_rng=torch.ones(1,dtype=torch.uint8))),
        ("invalid CUDA RNG",lambda z:z.update(cuda_rng=[torch.ones(1,dtype=torch.uint8)])),
        ("invalid loader RNG",lambda z:z.update(loader_gen=torch.ones(1,dtype=torch.uint8))),
        ("missing permutation",lambda z:z.pop("loader_perm")),
        ("missing cursor",lambda z:z.pop("loader_pos_idx")),
        ("missing rank order",lambda z:z.pop("loader_rank_of_node")),
        ("wrong Adam dose",lambda z:next(iter(z["optimizer"]["state"].values()))["step"].fill_(1)),
        ("wrong moment shape",lambda z:next(iter(z["optimizer"]["state"].values())).update(exp_avg=torch.zeros(1))),
    ]:results.append(expect_fail(base,"actual3d",lambda td,fn=fn:ck_mut(td,fn),label))

    shutil.rmtree(base, ignore_errors=True)
    V.N=original_n
    V.DATA,V.DATA_MANIFEST,V._GRAPH,V.EDGES_SHA256,V.R_ACTUAL_SHA256=saved
    out = {"schema": "card033-selftest-2026-09-12", "PASS": True, "n_cases": len(results), "results": results}
    (V.OC / "card033-selftest.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
