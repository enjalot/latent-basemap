"""Card024 canonical strict validator + identity (per card024-contrastive-normalization.md). ONE source of
truth reused by trainer, canary, chain. CPU-only.

expected_identity(arm) is the immutable admission identity bound into every checkpoint: the contrastive family
MODE, the qhat kernel (a,b), the fresh 2D init hash, the 300K substrate + fixed15 graph hashes, uniform
nonself noise semantics (rankneg 0), seed/LR/dose/batch/pos_ratio/precision, the beta initialization + scalar
optimizer spec (nce_learned), and the runtime-manifest hash. Deterministic ⇒ a resume rebuilds the same dict
and the core admission guard rejects a wrong family / scalar config before state restoration.
strict_validate_arm does full completion validation (exact dose, warm init, precision, endpoint==60K snapshot,
snapshot progress+freshness, resumable step ckpts with bound identity + — nce_learned — the recovered scalar,
and scalar EXPOSURE: beta must have moved for nce_learned, stayed 0 for neg_fixed, be absent for umap_uniform).
Corrupt checkpoints fail closed; idempotent completion is decided ONLY by this validator.
"""
import hashlib, json
from pathlib import Path

SB = Path("/data/latent-basemap/sandbox"); OC = SB / "overseer-codex"
CHAMPION = SB / "dino-arrival-t0/champion-bs16k/model.pt"
SUBD = Path("/data/latent-basemap/substrates/card010-adaptive")
SUB = SUBD / "substrate.f16.npy"; GRAPH = SUBD / "edges-fixed15.npz"; INIT = SUBD / "init-card010.pt"
TD_DEFAULT = SB / "card024-train"
INIT_SHA = "589895f037d406ae"
INIT_FILE_SHA = "7d313c265cb659c954951d79ec395fa4d267f0f34a0bfb6aafe0dd23d5985825"
SUB_SHA256 = "873d76e35eb2151e966c1c4dabb05e113c53f9cbc6e8473d87b6bd25029e5f45"
GRAPH_SHA256 = "d214a839b07113dff2c29b225da9f38008f86a0b2cb3662a39d14bc0542d4450"
A, B = 1.9328, 0.7905
N = 300000; NC = 2; DOSE = 60000; LR = 1e-3; BATCH = 16384; POS_RATIO = 0.1; RANKNEG = 0; SEED = 42
SNAP_STEPS = [20000, 40000, 60000]; STEP_CKPTS = [20000, 40000, 60000]
ARMS = ["umap_uniform", "neg_fixed", "nce_learned"]
MODE = {"umap_uniform": "umap_uniform", "neg_fixed": "neg_fixed", "nce_learned": "nce_learned"}
KERNEL = {"umap_uniform": "BCE(qhat) uniform-noise 2D", "neg_fixed": "BCE-logit log(qhat)-beta, beta fixed 0",
          "nce_learned": "BCE-logit log(qhat)-beta, beta learned"}


def full_sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    return h.hexdigest()


def state_sha(sd):
    import numpy as np
    h = hashlib.sha256()
    for k in sorted(sd): h.update(k.encode()); h.update(np.ascontiguousarray(sd[k].detach().cpu().numpy()).tobytes())
    return h.hexdigest()[:16]


def runtime_manifest_path(ROOT): return Path(ROOT) / "card024-runtime-sha.json"
def runtime_manifest_check(ROOT):
    exp = json.loads(runtime_manifest_path(ROOT).read_text())
    bad = [n for n, h in exp.items() if full_sha(Path(ROOT) / n) != h]
    return (len(bad) == 0, bad)


def loaded_basemap_under_root(ROOT):
    import sys
    ROOT = Path(ROOT).resolve(); out = {}
    for name, mod in list(sys.modules.items()):
        if not name.startswith("basemap"): continue
        f = getattr(mod, "__file__", None)
        if not f: continue
        rp = Path(f).resolve()
        assert str(rp).startswith(str(ROOT)), f"loaded basemap module {name} OUTSIDE worktree: {rp}"
        out[name] = {"path": str(rp), "sha16": full_sha(rp)[:16]}
    return out


def noise_recipe():
    import math
    npos=int(BATCH*POS_RATIO);nnoise=BATCH-npos;edges=15*N;full,tail=divmod(edges,npos);batches=full+bool(tail)
    return {"full_positive_slots":npos,"noise_slots_per_batch":nnoise,"full_ratio":nnoise/npos,
            "tail_positive_slots":tail,"tail_ratio":nnoise/tail if tail else None,
            "epoch_positive_slots":edges,"epoch_noise_slots":batches*nnoise,"epoch_pooled_ratio":batches*nnoise/edges,
            "positive_mode":"PERM binary fixed15","noise":"uniform ordered nonself, independent draws",
            "grouped_negatives":False,"endpoint_reuse":False,"per_batch_replacement":False,
            "beta_interpretation":"constant classifier intercept; nominal and tail priors disclosed; no batch-dependent beta offset"}

def check_init():
    import torch
    assert full_sha(INIT)==INIT_FILE_SHA, "full init file hash"
    z=torch.load(INIT,map_location="cpu",weights_only=False);h=hashlib.sha256()
    for v in z["model_state"].values():h.update(v.detach().numpy().tobytes())
    assert h.hexdigest()[:16]==INIT_SHA, "actual init parameter hash"
    return z["model_state"]


def expected_identity(arm, ROOT):
    assert arm in ARMS, arm
    ident = {"card": "card024", "arm": arm, "mode": MODE[arm], "kernel": KERNEL[arm], "kernel_a": A, "kernel_b": B,
             "init_sha256": INIT_SHA, "init_file_sha256": INIT_FILE_SHA, "sampler_recipe": noise_recipe(), "substrate_sha256": SUB_SHA256, "graph_sha256": GRAPH_SHA256,
             "noise_semantics": "uniform nonself ordered pairs (rankneg_window=0; no fneg/neg_tanh/rank weighting)",
             "seed": SEED, "lr": LR, "lr_schedule": "constant", "dose": DOSE, "rankneg_window": RANKNEG,
             "batch_size": BATCH, "pos_ratio": POS_RATIO, "n_components": NC, "precision": "device_fp16",
             "x_residency": "auto", "required_input_pipeline": "device",
             "runtime_manifest_sha256": full_sha(runtime_manifest_path(ROOT))}
    if arm == "nce_learned":
        ident.update({"beta_init": 0.0, "scalar_learned": True, "scalar_lr": LR, "scalar_weight_decay": 0.0})
    elif arm == "neg_fixed":
        ident.update({"beta_init": 0.0, "scalar_learned": False, "scalar_lr": None, "scalar_weight_decay": None})
    else:
        ident.update({"beta_init": None, "scalar_learned": None, "scalar_lr": None, "scalar_weight_decay": None})
    return ident


def validate_resume_payload(ck, step, n_nodes=None):
    """Validate the actual PERM/device checkpoint schema observed in real Card019 step20K.
    CPU RNG is load-tested; CUDA Philox states are checked against this runtime's observed16-byte schema.
    This does not substitute for the mandatory real GPU resume twin.
    """
    import torch, numpy as np
    n_nodes = N if n_nodes is None else n_nodes
    assert ck.get("schema") == "pumap-ckpt-2026-08-30", "checkpoint schema"
    assert isinstance(ck.get("epoch"), int) and ck["epoch"] >= 0, "missing epoch"
    ts = ck.get("train_stats", {})
    assert ts.get("executed_iters") == ts.get("positive_lr_optimizer_steps") == ts.get("optimizer_steps_succeeded") == step, "checkpoint successful counters"
    cfg = ck["config"]
    assert cfg.get("batch_size") == BATCH and cfg.get("random_state") == SEED and cfg.get("architecture") == "residual_bottleneck", "checkpoint sampler/model config"
    assert cfg.get("learning_rate") == LR and cfg.get("lr_schedule") == "constant", "checkpoint LR identity"
    assert cfg.get("replay_enabled") is False and cfg.get("deriv_enabled") is False, "unexpected preservation treatment"
    is_nce=ck.get("card012_identity",{}).get("arm",ck.get("card012_identity",{}).get("mode"))=="nce_learned"
    scalar=ck.get("card024_beta")
    if is_nce:assert torch.is_tensor(scalar) and scalar.shape==() and bool(torch.isfinite(scalar)), "learned scalar invalid"
    tensors=list(ck["model"].values())+([scalar] if is_nce else [])
    opt = ck["optimizer"]; groups = opt.get("param_groups", []); states = opt.get("state", {})
    assert groups and states, "empty optimizer state/groups"
    pids = [i for g in groups for i in g.get("params", [])]
    assert len(pids) == len(set(pids)) == len(tensors) and set(pids) == set(states), "optimizer parameter identity"
    assert len(groups)==(2 if is_nce else 1), "optimizer group count"
    if is_nce:assert len(groups[-1]["params"])==1 and groups[-1]["weight_decay"]==0, "scalar optimizer group"
    assert groups[0]["weight_decay"]==.01, "parent model AdamW decay"
    for g in groups:
        assert all(k in g for k in ["lr", "betas", "eps", "weight_decay", "amsgrad", "params"]), "incomplete Adam group"
        assert g["lr"] == LR and g["eps"] > 0 and len(g["betas"]) == 2, "Adam group configuration"
    for pid, parameter in zip(pids, tensors):
        state = states[pid]
        assert all(k in state for k in ["step", "exp_avg", "exp_avg_sq"]), "missing Adam moments"
        assert torch.is_tensor(state["step"]) and state["step"].numel() == 1 and float(state["step"]) == step, "Adam successful dose"
        for key in ["exp_avg", "exp_avg_sq"]:
            assert state[key].shape == parameter.shape and bool(torch.isfinite(state[key]).all()), "Adam moment shape/nonfinite"
    sched = ck["scheduler"]
    assert all(k in sched for k in ["base_lrs", "last_epoch", "_step_count", "_last_lr", "lr_lambdas"]), "incomplete constant scheduler"
    assert len(sched["base_lrs"]) == len(groups) == len(sched["_last_lr"]) and all(x == LR for x in sched["base_lrs"] + sched["_last_lr"]), "scheduler LR mismatch"
    assert isinstance(sched["last_epoch"], int) and isinstance(sched["_step_count"], int) and sched["_step_count"] >= 1, "scheduler counters"
    cpu = ck["torch_rng"]
    assert torch.is_tensor(cpu) and cpu.dtype == torch.uint8 and cpu.ndim == 1, "CPU RNG dtype/shape"
    torch.Generator(device="cpu").set_state(cpu.cpu())
    def cuda_state(v):
        return torch.is_tensor(v) and v.dtype == torch.uint8 and v.ndim == 1 and v.numel() == 16
    assert isinstance(ck["cuda_rng"], (list, tuple)) and len(ck["cuda_rng"]) >= 1 and all(cuda_state(v) for v in ck["cuda_rng"]), "CUDA RNG schema"
    assert cuda_state(ck["loader_gen"]), "device loader RNG schema"
    for k in ["loader_perm", "loader_pos_idx", "loader_batch_no"]:
        assert k in ck and ck[k] is not None, "missing PERM/rank continuation field: " + k
    perm = ck["loader_perm"]; n_edges = 15 * n_nodes
    assert torch.is_tensor(perm) and perm.dtype == torch.int64 and perm.shape == (n_edges,), "PERM shape/dtype"
    assert int(perm.min()) >= 0 and int(perm.max()) < n_edges, "PERM bounds"
    assert isinstance(ck["loader_pos_idx"], int) and 0 <= ck["loader_pos_idx"] < n_edges + int(BATCH * POS_RATIO), "PERM cursor"
    assert isinstance(ck["loader_batch_no"], int) and ck["loader_batch_no"] >= 0, "loader batch cursor"
    assert ck.get("loader_rank_of_node") is None and ck.get("loader_node_at_rank") is None, "uniform arm carried rank order"
    scaler=ck.get("scaler")
    assert isinstance(scaler,dict) and all(k in scaler for k in ["scale","growth_factor","backoff_factor","growth_interval","_growth_tracker"]), "AMP scaler state absent"
    assert all(np.isfinite(v) for v in scaler.values()) and scaler["scale"]>0, "invalid scaler state"
    assert ts.get("card024_objective_steps")==step and ts.get("card024_noise_slots")==step*(BATCH-int(BATCH*POS_RATIO)), "objective/noise exposure"
    assert 0<ts.get("card024_positive_slots",0)<=step*int(BATCH*POS_RATIO), "positive exposure"
    if is_nce:
        assert ts.get("card024_scalar_steps")==step and ts.get("card024_scalar_absgrad_sum",0)>0, "scalar gradient exposure"
        assert np.isfinite(ts["card024_scalar_absgrad_sum"]) and np.isfinite(ts["card024_scalar_absgrad_max"]), "scalar gradient summary nonfinite"


def strict_validate_arm(arm, ROOT, base=None):
    import torch, numpy as np
    base = Path(base) if base else TD_DEFAULT
    ad = base / f"admission-{arm}.json"; mp = base / f"manifest-{arm}.json"
    assert ad.exists() and mp.exists(), f"missing admission/manifest for {arm}"
    adm = json.loads(ad.read_text()); man = json.loads(mp.read_text()); exp = expected_identity(arm, ROOT)
    assert adm.get("card012_identity") == exp, f"{arm}: admission identity != canonical"
    assert man.get("card012_identity") == exp, f"{arm}: manifest identity != canonical"
    assert man.get("warm_init_sha256") == INIT_SHA, f"{arm}: warm init != fresh 2D 589895f0"
    assert man.get("warm_start_param_sha256")==INIT_SHA and adm.get("warm_start_param_sha256")==INIT_SHA, "applied init parameter mismatch"
    ts = man.get("train_stats", {})
    assert ts.get("card024_objective_steps")==DOSE and ts.get("card024_noise_slots")==DOSE*noise_recipe()["noise_slots_per_batch"], "production objective exposure"
    assert 0<ts.get("card024_positive_slots",0)<=DOSE*noise_recipe()["full_positive_slots"], "positive slots absent"
    if arm=="nce_learned":
        assert ts.get("card024_scalar_steps")==DOSE and ts.get("card024_scalar_absgrad_sum",0)>0, "full-dose scalar exposure"
        assert np.isfinite(ts["card024_scalar_absgrad_sum"]), "scalar gradient nonfinite"
    assert man.get("executed_steps") == ts.get("positive_lr_optimizer_steps") == DOSE, f"{arm}: dose != {DOSE}"
    assert man.get("pipeline_info", {}).get("x_residency") == "device_fp16", f"{arm}: pipeline not device_fp16"
    assert abs(float(man.get("lr_used_min", 0)) - LR) < 1e-12 and abs(float(man.get("lr_used_max", 0)) - LR) < 1e-12, "LR not 1e-3"
    lm = man.get("loaded_modules", {})
    assert lm.get("verified_frozen_runtime") and lm.get("all_basemap_under_root"), f"{arm}: runtime not frozen/under-root"
    ok, bad = runtime_manifest_check(ROOT); assert ok, f"frozen runtime drifted: {bad}"

    ep = torch.load(base / f"model-{arm}.pt", map_location="cpu", weights_only=False)
    assert ep.get("card024_mode")==MODE[arm], "endpoint objective metadata"
    assert ep.get("card024_beta")==man.get("final_beta"), "endpoint scalar metadata"
    ep_sd = ep["model_state_dict"]; assert all(bool(torch.isfinite(t).all()) for t in ep_sd.values()), "endpoint non-finite"
    ep_sha = state_sha(ep_sd); assert ep_sha == man.get("trained_sha256"), f"{arm}: endpoint sha != manifest"
    ad_mtime = ad.stat().st_mtime; snap_sha = {}
    for s in SNAP_STEPS:
        p = base / arm / f"model-step{s}.pt"; assert p.exists(), f"missing snapshot {s}"
        o = torch.load(p, map_location="cpu", weights_only=False); sd = o["model_state_dict"]
        assert o["n_components"] == NC and o["learning_rate"] == LR and o["lr_schedule"] == "constant", f"snapshot {s} config"
        assert all(bool(torch.isfinite(t).all()) for t in sd.values()) and p.stat().st_mtime >= ad_mtime, f"snapshot {s} bad/stale"
        assert o.get("card024_mode")==MODE[arm], "snapshot objective metadata"
        if arm=="nce_learned": assert np.isfinite(o.get("card024_beta",np.nan)), "snapshot scalar metadata"
        snap_sha[s] = state_sha(sd)
    assert len(set(snap_sha.values())) == len(SNAP_STEPS), f"{arm}: snapshots did not progress"
    assert snap_sha[60000] == ep_sha, f"{arm}: endpoint != 60K snapshot"

    ck_receipt = {}
    for s in STEP_CKPTS:
        p = base / arm / "ckpts" / f"ckpt-step{s}.pt"; assert p.exists(), f"missing resumable step ckpt {s}"
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:
            raise AssertionError(f"corrupt step checkpoint {s}: {e!r}")
        validate_resume_payload(ck,s,n_nodes=N)
        assert int(ck.get("global_step", -1)) == s, f"step ckpt {s} global_step mismatch"
        assert bool(ck.get("step_checkpoint")), f"ckpt {s} not marked step_checkpoint"
        assert ck.get("card012_identity") == exp, f"step ckpt {s} identity != canonical"
        if arm == "nce_learned":
            assert ck.get("card024_beta") is not None, f"nce_learned step ckpt {s} missing learned scalar"
            sn=torch.load(base / arm / f"model-step{s}.pt",map_location="cpu",weights_only=False)
            assert float(ck["card024_beta"])==sn["card024_beta"], "snapshot/checkpoint scalar mismatch"
        m = ck["model"]; assert all(bool(torch.isfinite(t).all()) for t in m.values()), f"step ckpt {s} non-finite"
        msha = state_sha(m)
        if s in snap_sha: assert msha == snap_sha[s], f"step ckpt {s} model payload != snapshot"
        ck_receipt[s] = msha

    # scalar EXPOSURE: no scientific result from a silently inactive scalar
    traj = ts.get("card024_beta_traj", []); final_beta = man.get("final_beta")
    if arm == "nce_learned":
        assert [x["step"] for x in traj]==SNAP_STEPS and final_beta is not None and np.isfinite(final_beta) and abs(float(final_beta)) > 0, f"nce_learned scalar never moved (beta={final_beta})"
        assert traj[-1]["beta"]==final_beta, "final trajectory beta mismatch"
        for x in traj:
            q=torch.load(base / arm / f"model-step{x['step']}.pt",map_location="cpu",weights_only=False)
            assert x["beta"]==q["card024_beta"], "trajectory/snapshot beta mismatch"
    elif arm == "neg_fixed":
        assert (final_beta is None) or float(final_beta) == 0.0, "neg_fixed scalar must stay 0"
    else:
        assert final_beta is None, "umap_uniform must have no scalar"

    cf = base / f"coords-{arm}.npy"; coords = np.load(cf, mmap_mode="r")
    assert coords.shape == (N, NC) and bool(np.isfinite(coords).all()), f"{arm}: coords bad"
    return {"arm": arm, "PASS": True, "model_state_sha": ep_sha, "model_file_sha": full_sha(base / f"model-{arm}.pt"),
            "coords_file_sha": full_sha(cf), "final_beta": final_beta, "snapshot_sha": {str(k): v for k, v in snap_sha.items()},
            "step_ckpt_sha": {str(k): v for k, v in ck_receipt.items()}, "identity": exp}
