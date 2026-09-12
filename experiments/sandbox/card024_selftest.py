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
    td=Path(base);(td/arm/"ckpts").mkdir(parents=True,exist_ok=True);ident=V.expected_identity(arm,ROOT)
    snap={step:_sd(1000+i) for i,step in enumerate(V.SNAP_STEPS)};nce=arm=="nce_learned";beta=.125 if nce else (0.0 if arm=="neg_fixed" else None)
    def stats(step):
        return {"executed_iters":step,"positive_lr_optimizer_steps":step,"optimizer_steps_succeeded":step,
          "card024_objective_steps":step,"card024_noise_slots":step*(V.BATCH-int(V.BATCH*V.POS_RATIO)),
          "card024_positive_slots":step*min(V.N*15,int(V.BATCH*V.POS_RATIO)),"card024_scalar_steps":step if nce else 0,
          "card024_scalar_absgrad_sum":step*.1 if nce else 0,"card024_scalar_absgrad_max":.1 if nce else 0}
    for step in V.SNAP_STEPS:
        torch.save({"model_state_dict":snap[step],"n_components":V.NC,"learning_rate":V.LR,"lr_schedule":"constant","card024_mode":arm,"card024_beta":beta},td/arm/f"model-step{step}.pt")
    torch.save({"model_state_dict":snap[60000],"card024_mode":arm,"card024_beta":beta},td/f"model-{arm}.pt")
    for step in V.STEP_CKPTS:
        m=snap[step];values=list(m.values())+([torch.tensor(beta)] if nce else []);ids=list(range(len(values)));groups=[{"params":ids[:len(m)],"lr":V.LR,"betas":(.9,.999),"eps":1e-8,"weight_decay":.01,"amsgrad":False}]
        if nce:groups.append({**groups[0],"params":[ids[-1]],"weight_decay":0.0})
        ck={"global_step":step,"step_checkpoint":True,"card012_identity":ident,"model":m,"card024_beta":torch.tensor(beta) if beta is not None else None,
            "schema":"pumap-ckpt-2026-08-30","epoch":0,"train_stats":stats(step),
            "config":{"batch_size":V.BATCH,"random_state":V.SEED,"architecture":"residual_bottleneck","learning_rate":V.LR,"lr_schedule":"constant","replay_enabled":False,"deriv_enabled":False},
            "optimizer":{"param_groups":groups,"state":{i:{"step":torch.tensor(float(step)),"exp_avg":torch.zeros_like(v),"exp_avg_sq":torch.zeros_like(v)} for i,v in zip(ids,values)}},
            "scheduler":{"base_lrs":[V.LR]*len(groups),"last_epoch":step,"_step_count":step+1,"_last_lr":[V.LR]*len(groups),"lr_lambdas":[None]*len(groups)},
            "scaler":{"scale":65536.,"growth_factor":2.,"backoff_factor":.5,"growth_interval":2000,"_growth_tracker":0},
            "torch_rng":torch.Generator().manual_seed(0).get_state(),"cuda_rng":[torch.zeros(16,dtype=torch.uint8)],"loader_gen":torch.zeros(16,dtype=torch.uint8),
            "loader_perm":torch.arange(15*V.N),"loader_pos_idx":0,"loader_batch_no":step,"loader_rank_of_node":None,"loader_node_at_rank":None,"rankneg_scale":None}
        torch.save(ck,td/arm/"ckpts"/f"ckpt-step{step}.pt")
    np.save(td/f"coords-{arm}.npy",np.zeros((V.N,V.NC),"f4"));ts=stats(V.DOSE);ts['card024_beta_traj']=[{"step":step,"beta":beta} for step in V.SNAP_STEPS] if beta is not None else []
    man={"arm":arm,"mode":arm,"card012_identity":ident,"warm_init_sha256":V.INIT_SHA,"warm_start_param_sha256":V.INIT_SHA,
       "executed_steps":V.DOSE,"train_stats":ts,"lr_used_min":V.LR,"lr_used_max":V.LR,"final_beta":beta,"pipeline_info":{"x_residency":"device_fp16"},
       "trained_sha256":V.state_sha(snap[60000]),"loaded_modules":{"verified_frozen_runtime":True,"all_basemap_under_root":True}}
    (td/f"admission-{arm}.json").write_text(json.dumps({"arm":arm,"card012_identity":ident,"warm_init_sha256":V.INIT_SHA,"warm_start_param_sha256":V.INIT_SHA}))
    os.utime(td/f"admission-{arm}.json",(0,0));(td/f"manifest-{arm}.json").write_text(json.dumps(man));return td


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

    def ckmut(td, fn):
        p=td/"nce_learned/ckpts/ckpt-step40000.pt";c=torch.load(p,map_location="cpu",weights_only=False);fn(c);torch.save(c,p)
    cases=[("missing optimizer",lambda c:c.pop("optimizer")),("empty Adam state",lambda c:c["optimizer"].update(state={})),
      ("wrong scalar moment",lambda c:c["optimizer"]["state"][3].update(exp_avg=torch.zeros(2))),
      ("wrong scalar dose",lambda c:c["optimizer"]["state"][3].update(step=torch.tensor(1.))),
      ("wrong scalar decay",lambda c:c["optimizer"]["param_groups"][-1].update(weight_decay=.01)),
      ("missing scheduler",lambda c:c.pop("scheduler")),("wrong scheduler LR",lambda c:c["scheduler"].update(_last_lr=[.1,.1])),
      ("missing scaler",lambda c:c.pop("scaler")),("bad CPU RNG",lambda c:c.update(torch_rng=torch.zeros(2,dtype=torch.uint8))),
      ("bad loader RNG",lambda c:c.update(loader_gen=torch.zeros(8,dtype=torch.uint8))),
      ("missing loader permutation",lambda c:c.pop("loader_perm")),("unwanted ranked sampler",lambda c:c.update(loader_rank_of_node=torch.arange(V.N))),
      ("nonfinite beta",lambda c:c.update(card024_beta=torch.tensor(float('nan')))),
      ("wrong objective dose",lambda c:c["train_stats"].update(card024_objective_steps=0)),
      ("missing scalar exposure",lambda c:c["train_stats"].update(card024_scalar_steps=0))]
    for label,fn in cases:results.append(expect_fail(base,"nce_learned",lambda td,fn=fn:ckmut(td,fn),label))
    results.append(expect_fail(base,"nce_learned",lambda td:_man(td,"nce_learned",warm_start_param_sha256="wrong"),"actual warm mismatch"))
    shutil.rmtree(base, ignore_errors=True)
    out = {"schema": "card024-selftest-2026-09-12", "PASS": True, "n_cases": len(results), "results": results}
    (V.OC / "card024-selftest.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
