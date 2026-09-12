"""Card038 canonical strict validator + identity. ONE source of truth reused by the trainer, canary and
preflight. CPU-only. Root's separate scorer (score_card038_exact.py) imports SUBD/SUB/GRAPH/TD_DEFAULT/ARMS/
expected_identity/runtime_manifest_check/full_sha from here.

Two arms at different WIDTHS and DOSES on the proven core (card033 @ 8b7ecab): wide2048 (H2048, 14,170,627
params, 60K successful positive-LR updates) and compact1024 (H1024, 4,332,803 params, 180K). 3D output, fresh
per-width init at seed 42 (persisted + hashed), half-strength local radii (sqrt of Card013 r_actual) via the
Hook-C kernel, original sampler, LR .001, AdamW wd .01, clip 1, rankneg 75000, fneg 1, tanh 4, batch 16384,
pos .1. Width is bound by the ACTUAL parameter tensors/count, not a metadata string. Checkpoints use the
core card012 schema (pumap-ckpt-2026-08-30); validate_ckpt_payload deep-validates them before restore and at
completion.
"""
import hashlib, json
from pathlib import Path

SB = Path("/data/latent-basemap/sandbox"); OC = SB / "overseer-codex"
CHAMPION = SB / "dino-arrival-t0/champion-bs16k/model.pt"
SUBD = Path("/data/latent-basemap/substrates/card010-adaptive")
SUB = SUBD / "substrate.f16.npy"; GRAPH = SUBD / "edges-fixed15.npz"
RADII = SB / "card014-radii/r_actual_half.npy"; INITD = SB / "card038-init"
TD_DEFAULT = SB / "card038-train"
SUB_SHA256 = "873d76e35eb2151e966c1c4dabb05e113c53f9cbc6e8473d87b6bd25029e5f45"
GRAPH_SHA256 = "d214a839b07113dff2c29b225da9f38008f86a0b2cb3662a39d14bc0542d4450"
R_HALF_SHA256 = "72dde55fb6124afb97afa43745231683279b329b44aa57b7074c8a2f8fc12924"
INIT_STATE_SHA = {"wide2048": "b7ba2ffeeb794bd4", "compact1024": "257455f65f1e0147"}
INIT_FILE_SHA = {"wide2048": "03c472172836a5ccc4c0f607b2f437d5072c37bc6b351d439165d9b9e402f098",
                 "compact1024": "30f73b6d9bf88c5d026fec1f6402250043849dcabbdd917dc7f352a5b5760069"}
INIT_PARAM_SHA = {"wide2048":"eab90e470348ec786365bfffd0e9236fe7cad3eb9c037992d1017c8625a27929", "compact1024":"73e970f75a8e82fe0bc1e9f817653621b1df83e0780305209bfadd53847f4805"}
WIDTH = {"wide2048": 2048, "compact1024": 1024}; NPARAM = {"wide2048": 14170627, "compact1024": 4332803}
DOSE = {"wide2048": 60000, "compact1024": 180000}
SNAP = {"wide2048": [20000, 40000, 60000], "compact1024": [60000, 120000, 180000]}
N = 300000; NC = 3; LR = 1e-3; WEIGHT_DECAY = 0.01; GRAD_CLIP = 1.0; BATCH = 16384; POS_RATIO = 0.1
RANKNEG = 75000; FNEG = 1.0; TANH = 4.0; SEED = 42; N_POS_EDGES = N * 15
ARMS = ["wide2048", "compact1024"]
KERNEL = "half local-scale d2/(r_i*r_j), radii = sqrt(Card013 r_actual)"


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


def runtime_manifest_path(ROOT): return Path(ROOT) / "card038-runtime-sha.json"
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
        assert rp.is_relative_to(ROOT), f"loaded basemap module {name} OUTSIDE worktree: {rp}"
        out[name] = {"path": str(rp), "sha16": full_sha(rp)[:16]}
    return out


def expected_identity(arm, ROOT):
    assert arm in ARMS, arm
    return {"base_config_artifact_sha":full_sha(CHAMPION),"gpu_resident_vram_budget_gb":14.0,"fneg_weight":1.0,"neg_tanh_gamma":4.0,"positive_target_mode":"binary","observe_exposure":True,"n_layers":3,"neck_fraction":.75,"low_dim_kernel":"umap","warm_param_sha256":INIT_PARAM_SHA[arm][:16],"card": "card038", "arm": arm, "width_hidden_dim": WIDTH[arm], "n_params": NPARAM[arm], "n_components": NC,
            "dose": DOSE[arm], "kernel": KERNEL, "radii_sha256": R_HALF_SHA256,
            "init_state_sha256": INIT_STATE_SHA[arm], "init_file_sha256": INIT_FILE_SHA[arm],
            "substrate_sha256": SUB_SHA256, "graph_sha256": GRAPH_SHA256, "seed": SEED, "lr": LR, "lr_schedule": "constant",
            "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP, "batch_size": BATCH, "pos_ratio": POS_RATIO,
            "rankneg_window": RANKNEG, "fneg": FNEG, "tanh_gamma": TANH, "precision": "device_fp16",
            "x_residency": "auto", "required_input_pipeline": "device", "runtime_manifest_sha256": full_sha(runtime_manifest_path(ROOT))}


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
    ident = ck.get("card012_identity", {})
    assert ident.get("gpu_resident_vram_budget_gb") == 14.0, "resident budget identity drift"
    assert ident.get("fneg_weight")==1.0 and ident.get("neg_tanh_gamma")==4.0 and ident.get("positive_target_mode")=="binary", "negative/positive recipe identity drift"
    assert cfg.get("batch_size") == BATCH and cfg.get("random_state") == SEED and cfg.get("architecture") == "residual_bottleneck", "checkpoint sampler/model config"
    assert cfg.get("replay_enabled") is False and cfg.get("deriv_enabled") is False, "unexpected preservation treatment"
    opt = ck["optimizer"]; groups = opt.get("param_groups", []); states = opt.get("state", {})
    assert groups and states, "empty optimizer state/groups"
    pids = [i for g in groups for i in g.get("params", [])]
    assert len(pids) == len(set(pids)) == len(ck["model"]) and set(pids) == set(states), "optimizer parameter identity"
    for g in groups:
        assert all(k in g for k in ["lr", "betas", "eps", "weight_decay", "amsgrad", "params"]), "incomplete Adam group"
        assert g["lr"] == LR and g["weight_decay"] == WEIGHT_DECAY and g["eps"] > 0 and len(g["betas"]) == 2, "Adam group configuration"
    for pid, parameter in zip(pids, ck["model"].values()):
        state = states[pid]
        assert all(k in state for k in ["step", "exp_avg", "exp_avg_sq"]), "missing Adam moments"
        assert torch.is_tensor(state["step"]) and state["step"].numel() == 1 and float(state["step"]) == step, "Adam successful dose"
        for key in ["exp_avg", "exp_avg_sq"]:
            assert state[key].shape == parameter.shape and bool(torch.isfinite(state[key]).all()), "Adam moment shape/nonfinite"
    sched = ck["scheduler"]
    assert all(k in sched for k in ["base_lrs", "last_epoch", "_step_count", "_last_lr", "lr_lambdas"]), "incomplete constant scheduler"
    assert len(sched["base_lrs"]) == len(groups) == len(sched["_last_lr"]) and all(x == LR for x in sched["base_lrs"] + sched["_last_lr"]), "scheduler LR mismatch"
    assert isinstance(sched["last_epoch"], int) and isinstance(sched["_step_count"], int) and sched["_step_count"] >= 1, "scheduler counters"
    scaler=ck.get("scaler")
    assert isinstance(scaler,dict) and all(k in scaler for k in ["scale","growth_factor","backoff_factor","growth_interval","_growth_tracker"]), "missing AMP scaler continuation state"
    assert all(np.isfinite(scaler[k]) for k in scaler) and scaler["scale"]>0, "invalid AMP scaler"
    cpu = ck["torch_rng"]
    assert torch.is_tensor(cpu) and cpu.dtype == torch.uint8 and cpu.ndim == 1, "CPU RNG dtype/shape"
    torch.Generator(device="cpu").set_state(cpu.cpu())
    def cuda_state(v):
        return torch.is_tensor(v) and v.dtype == torch.uint8 and v.ndim == 1 and v.numel() == 16
    assert isinstance(ck["cuda_rng"], (list, tuple)) and len(ck["cuda_rng"]) >= 1 and all(cuda_state(v) for v in ck["cuda_rng"]), "CUDA RNG schema"
    assert cuda_state(ck["loader_gen"]), "device loader RNG schema"
    for k in ["loader_perm", "loader_pos_idx", "loader_batch_no", "loader_rank_of_node", "loader_node_at_rank", "rankneg_scale"]:
        assert k in ck and ck[k] is not None, "missing PERM/rank continuation field: " + k
    perm = ck["loader_perm"]; n_edges = 15 * n_nodes
    assert torch.is_tensor(perm) and perm.dtype == torch.int64 and perm.shape == (n_edges,), "PERM shape/dtype"
    assert int(perm.min()) >= 0 and int(perm.max()) < n_edges, "PERM bounds"
    assert np.array_equal(np.sort(perm.cpu().numpy()),np.arange(n_edges)), "PERM duplicates/missing entries"
    assert isinstance(ck["loader_pos_idx"], int) and 0 <= ck["loader_pos_idx"] < n_edges + int(BATCH * POS_RATIO), "PERM cursor"
    assert isinstance(ck["loader_batch_no"], int) and ck["loader_batch_no"] >= 0, "loader batch cursor"
    rank, node = ck["loader_rank_of_node"], ck["loader_node_at_rank"]
    assert all(torch.is_tensor(v) and v.dtype == torch.int64 and v.shape == (n_nodes,) for v in [rank, node]), "rank-order shape/dtype"
    assert int(rank.min()) >= 0 and int(rank.max()) < n_nodes and int(node.min()) >= 0 and int(node.max()) < n_nodes, "rank-order bounds"
    assert torch.equal(rank[node], torch.arange(n_nodes)), "rank-order inverse mismatch"
    assert np.isfinite(ck["rankneg_scale"]) and ck["rankneg_scale"] > 0, "rank scale invalid"

def validate_ckpt_payload(ck, arm, ROOT, identity, expect_step=None, n_nodes=None):
    import torch
    assert ck.get('card012_identity')==identity, 'admission-identity mismatch (width/dose/radius/seed/data)'
    gs=int(ck['global_step'])
    if expect_step is not None:assert gs==expect_step, 'checkpoint step mismatch'
    validate_resume_payload(ck,gs,n_nodes=n_nodes)
    expected=expected_init(arm)
    assert set(ck['model'])==set(expected) and all(ck['model'][k].shape==expected[k].shape for k in expected), 'checkpoint architecture shapes'
    assert all(bool(torch.isfinite(t).all()) for t in ck['model'].values()), 'nonfinite model'
    assert ck['config']['lr_schedule']=='constant' and ck['config']['learning_rate']==LR, 'checkpoint LR'
    if identity.get('observe_exposure'):
        ts=ck['train_stats']
        assert 0<ts['card038_successful_positive']<=ts['card038_attempted_positive'], 'positive exposure'
        assert 0<ts['card038_successful_negative']<=ts['card038_attempted_negative'], 'negative exposure'
        assert ts.get('card038_attempted_probes'), 'missing exposure probe'
    return True


def expected_init(arm):
    import torch
    p=INITD/f'init-{arm}.pt';assert full_sha(p)==INIT_FILE_SHA[arm]
    sd=torch.load(p,map_location='cpu',weights_only=False)['model_state']
    assert state_sha(sd)==INIT_STATE_SHA[arm] and sum(t.numel() for t in sd.values())==NPARAM[arm], 'actual init tensors'
    assert param_sha(sd)==INIT_PARAM_SHA[arm], 'raw-parameter init identity'
    return sd


def param_sha(sd):
    import numpy as np
    h=hashlib.sha256()
    for t in sd.values():h.update(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def strict_validate_arm(arm, ROOT, base=None):
    import torch, numpy as np
    base = Path(base) if base else TD_DEFAULT
    ad = base / f"admission-{arm}.json"; mp = base / f"manifest-{arm}.json"
    assert ad.exists() and mp.exists(), f"missing admission/manifest for {arm}"
    adm = json.loads(ad.read_text()); man = json.loads(mp.read_text()); exp = expected_identity(arm, ROOT)
    assert adm.get("card012_identity") == exp, f"{arm}: admission identity != canonical"
    assert man.get("card012_identity") == exp, f"{arm}: manifest identity != canonical"
    assert man.get("init_state_sha256") == INIT_STATE_SHA[arm] and man.get("init_payload_sha256") == INIT_STATE_SHA[arm], f"{arm}: init hash"
    ts = man.get("train_stats", {})
    assert man.get("executed_steps") == ts.get("positive_lr_optimizer_steps") == DOSE[arm], f"{arm}: dose != {DOSE[arm]}"
    assert man.get("pipeline_info", {}).get("x_residency") == "device_fp16", f"{arm}: pipeline not device_fp16"
    assert abs(float(man.get("lr_used_min", 0)) - LR) < 1e-12 and abs(float(man.get("lr_used_max", 0)) - LR) < 1e-12, "LR not 1e-3"
    assert man.get("n_params") == NPARAM[arm] and man.get("width_hidden_dim") == WIDTH[arm], f"{arm}: width/param count"
    lm = man.get("loaded_modules", {})
    assert lm.get("verified_frozen_runtime") and lm.get("all_basemap_under_root"), f"{arm}: runtime not frozen/under-root"
    ok, bad = runtime_manifest_check(ROOT); assert ok, f"frozen runtime drifted: {bad}"

    # endpoint: actual param count binds width; == final-dose snapshot
    ep_obj=torch.load(base/f'model-{arm}.pt',map_location='cpu',weights_only=False)
    assert ep_obj['hidden_dim']==WIDTH[arm] and ep_obj['n_components']==NC and ep_obj['learning_rate']==LR and ep_obj['lr_schedule']=='constant', 'endpoint configuration'
    ep=ep_obj['model_state_dict'];expected=expected_init(arm)
    assert set(ep)==set(expected) and all(ep[k].shape==expected[k].shape for k in expected), 'endpoint architecture shapes'
    assert man['warm_param_sha256']==INIT_PARAM_SHA[arm][:16], 'actual warm-start provenance'

    assert int(sum(t.numel() for t in ep.values())) == NPARAM[arm], f"{arm}: endpoint param count != width"
    assert all(bool(torch.isfinite(t).all()) for t in ep.values()), "endpoint non-finite"
    assert ep["proj_out.weight"].shape[0] == NC, "endpoint not 3D"
    ep_sha = state_sha(ep); assert ep_sha == man.get("trained_sha256"), f"{arm}: endpoint sha != manifest"
    ad_mtime = ad.stat().st_mtime;assert mp.stat().st_mtime>=ad_mtime and (base/f"model-{arm}.pt").stat().st_mtime>=ad_mtime, "stale endpoint/manifest"
    assert lm.get("runtime_manifest_sha256")==full_sha(runtime_manifest_path(ROOT)), "runtime binding"
    for row in lm.get("basemap_modules",{}).values():
        path=Path(row["path"]).resolve();assert path.is_relative_to(Path(ROOT).resolve()) and full_sha(path)[:16]==row["sha16"], "loaded-module path/hash"
    snap_sha = {}
    for s in SNAP[arm]:
        p = base / arm / f"model-step{s}.pt"; assert p.exists(), f"missing snapshot {s}"
        o = torch.load(p, map_location="cpu", weights_only=False); sd = o.get("model_state_dict", o.get("model_state"))
        assert o.get("n_components") == NC, f"snapshot {s} n_components"
        assert int(sum(t.numel() for t in sd.values())) == NPARAM[arm], f"snapshot {s} width"
        assert all(bool(torch.isfinite(t).all()) for t in sd.values()) and p.stat().st_mtime >= ad_mtime, f"snapshot {s} bad/stale"
        snap_sha[s] = state_sha(sd)
    assert len(set(snap_sha.values())) == len(SNAP[arm]), f"{arm}: snapshots did not progress"
    assert snap_sha[DOSE[arm]] == ep_sha, f"{arm}: endpoint != final-dose snapshot"

    for s in SNAP[arm]:
        p = base / arm / "ckpts" / f"ckpt-step{s}.pt"; assert p.exists(), f"missing resumable step ckpt {s}"
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:
            raise AssertionError(f"corrupt step checkpoint {s}: {e!r}")
        validate_ckpt_payload(ck, arm, ROOT, exp, expect_step=s)
        assert bool(ck.get("step_checkpoint")), f"ckpt {s} not marked step_checkpoint"
        assert p.stat().st_mtime>=ad_mtime and state_sha(ck["model"])==snap_sha[s], "checkpoint freshness/snapshot match"
    epoch_cks = sorted((base / arm / "ckpts").glob("ckpt-epoch*.pt"))
    assert epoch_cks, f"{arm}: no epoch checkpoint"
    for ep_p in epoch_cks:
        assert ep_p.stat().st_mtime>=ad_mtime, "stale epoch checkpoint"
        validate_ckpt_payload(torch.load(ep_p, map_location="cpu", weights_only=False), arm, ROOT, exp)

    assert all(ts[k]==ck["train_stats"][k] for k in ts if k.startswith("card038_")), "exposure checkpoint/manifest mismatch"
    cf = base / f"coords-{arm}.npy"; coords = np.load(cf, mmap_mode="r")
    assert coords.shape == (N, NC) and bool(np.isfinite(coords).all()), f"{arm}: coords bad"
    return {"arm": arm, "PASS": True, "model_state_sha": ep_sha, "model_file_sha": full_sha(base / f"model-{arm}.pt"),
            "coords_file_sha": full_sha(cf), "n_params": NPARAM[arm], "snapshot_sha": {str(k): v for k, v in snap_sha.items()}, "identity": exp}


def ParametricUMAP_load_state(path):
    """Load a saved ParametricUMAP endpoint and return its model state_dict (schema-tolerant)."""
    import torch
    o = torch.load(path, map_location="cpu", weights_only=False)
    return o.get("model_state_dict", o.get("model_state"))
