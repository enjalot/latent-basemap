"""Card022 canonical strict validator + identity (per card022-local-shape-floor.md + root's bank-hash
directive). ONE source of truth reused by trainer, calibration, chain and scorer. CPU-only.

expected_identity(arm) is the immutable admission identity bound into every checkpoint: FULL (64-char)
content hashes of the parent (teacher) endpoint, the 300K substrate + fixed15 graph, and — for shape_floor —
the shape bank X + tau + manifest, plus epsilon, the frozen calibration coefficient, seed/LR/dose/precision.
Deterministic ⇒ a resume rebuilds the same dict and the core admission guard fails closed on drift (wrong
bank / weight / seed / arm). strict_validate_arm does full completion validation and fails closed on any
corrupt checkpoint. Idempotent completion is decided ONLY by this validator, never manifest dose alone.
"""
import hashlib, json
from pathlib import Path

SB = Path("/data/latent-basemap/sandbox"); OC = SB / "overseer-codex"
PARENT = SB / "card013-train/model-baseline.pt"
SUBD = Path("/data/latent-basemap/substrates/card010-adaptive")
SUB = SUBD / "substrate.f16.npy"; GRAPH = SUBD / "edges-fixed15.npz"
BANKD = Path("/data/latent-basemap/substrates/card022-shape-bank")
BANK_MANIFEST = BANKD / "manifest.json"
TD_DEFAULT = SB / "card022-train"; CALIB = OC / "card022-calibration.json"
# FULL content hashes (parent from the bank teacher_sha; graph/X/tau from the bank manifest; substrate computed)
TEACHER_SHA = "2b9c6d654bd63e0209d2b59667ef33e325905b5875a0dfd180d0166e69d00cbc"
SUB_SHA256 = "873d76e35eb2151e966c1c4dabb05e113c53f9cbc6e8473d87b6bd25029e5f45"
GRAPH_SHA256 = "d214a839b07113dff2c29b225da9f38008f86a0b2cb3662a39d14bc0542d4450"
BANK_X_SHA256 = "62014603c2120df04fd799079172c6107e0a3d23b0a758909e48404d391e3535"
BANK_TAU_SHA256 = "47c551c77d415acba554a2737f4aa381f9c5d7b3a3429089d8a69dd3e9664f4b"
EPSILON = 2.0141069984559376e-10
WARM_PARAM_SHA = "89eddf741fb28b40"
N = 300000; DIM = 1536; NC = 2; DOSE = 60000; LR = 1e-4; BATCH = 16384; POS_RATIO = 0.1
RANKNEG = 75000; SEED = 42; CENTERS_PER_STEP = 16; SHAPE_SAMPLER_SEED = SEED + 20220512  # core: random_state + 20220512
SNAP_STEPS = [20000, 40000, 60000]; STEP_CKPTS = [20000, 40000, 60000]
ARMS = ["ordinary", "shape_floor"]
KERNEL = {"ordinary": "baseline continuation 2D", "shape_floor": "covariance-floor relu(tau-q)^2 2D"}


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


def runtime_manifest_path(ROOT): return Path(ROOT) / "card022-runtime-sha.json"


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
        assert str(rp).startswith(str(ROOT)), f"loaded basemap module {name} resolves OUTSIDE worktree: {rp}"
        out[name] = {"path": str(rp), "sha16": full_sha(rp)[:16]}
    return out


PARENT_RECIPE = {"architecture":"residual_bottleneck","neck_fraction":0.75,"input_dim":1536,"n_components":2,
 "hidden_dim":2048,"n_layers":3,"n_neighbors":15,"a":1.9328,"b":0.7905,"low_dim_kernel":"umap",
 "kernel_alpha":1.0,"correlation_weight":0.0,"use_batchnorm":False,"use_dropout":False,
 "clip_grad_norm":1.0,"clip_grad_value":None,"pos_ratio":0.1,"positive_target_mode":"binary",
 "density_weight":0.0,"midnear_enabled":False,"fneg_weight":1.0,"fneg_lo":0.1,"fneg_hi":0.4,
 "rankneg_window":75000,"rankneg_exclude_neighbors":False,"neg_tanh_gamma":4.0,"kernel_anneal_frac":0.0}
def validate_parent_recipe(p):
    for k,v in PARENT_RECIPE.items():
        assert getattr(p,k,None)==v, f"inherited recipe mismatch {k}: {getattr(p,k,None)!r} != {v!r}"
    assert getattr(p,"_kernel_radii",None) is None, "unexpected local-scale radii"
    return dict(PARENT_RECIPE)


def calibrated_weight():
    """Frozen shape coefficient from the calibration step (raises if absent — required before shape_floor)."""
    c = json.loads(CALIB.read_text()); assert c.get("PASS"), "calibration not PASS"
    w = float(c["coefficient"]); assert w > 0 and w == w and w != float("inf"), "calibration coefficient invalid"
    return w


def expected_identity(arm, ROOT):
    assert arm in ARMS, arm
    common = {"card": "card022", "arm": arm, "kernel": KERNEL[arm],
              "teacher_sha256": TEACHER_SHA, "warm_param_sha256": WARM_PARAM_SHA, "substrate_sha256": SUB_SHA256, "graph_sha256": GRAPH_SHA256,
              "seed": SEED, "lr": LR, "lr_schedule": "constant", "dose": DOSE, "rankneg_window": RANKNEG,
              "batch_size": BATCH, "pos_ratio": POS_RATIO, "n_components": NC, "inherited_recipe": PARENT_RECIPE,
              "precision": "device_fp16", "x_residency": "auto", "required_input_pipeline": "device",
              "runtime_manifest_sha256": full_sha(runtime_manifest_path(ROOT))}
    if arm == "shape_floor":
        common.update({"shape_bank_X_sha256": BANK_X_SHA256, "shape_bank_tau_sha256": BANK_TAU_SHA256,
                       "shape_bank_manifest_sha256": full_sha(BANK_MANIFEST), "epsilon": EPSILON,
                       "shape_centers_per_step": CENTERS_PER_STEP, "shape_sampler_seed": SHAPE_SAMPLER_SEED,
                       "shape_weight": calibrated_weight()})
    else:
        common.update({"shape_bank_X_sha256": None, "shape_bank_tau_sha256": None,
                       "shape_bank_manifest_sha256": None, "epsilon": None,
                       "shape_centers_per_step": None, "shape_sampler_seed": None, "shape_weight": 0.0})
    return common


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
    opt = ck["optimizer"]; groups = opt.get("param_groups", []); states = opt.get("state", {})
    assert groups and states, "empty optimizer state/groups"
    pids = [i for g in groups for i in g.get("params", [])]
    assert len(pids) == len(set(pids)) == len(ck["model"]) and set(pids) == set(states), "optimizer parameter identity"
    for g in groups:
        assert all(k in g for k in ["lr", "betas", "eps", "weight_decay", "amsgrad", "params"]), "incomplete Adam group"
        assert g["lr"] == LR and g["eps"] > 0 and len(g["betas"]) == 2, "Adam group configuration"
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
    assert isinstance(ck["loader_pos_idx"], int) and 0 <= ck["loader_pos_idx"] < n_edges + int(BATCH * POS_RATIO), "PERM cursor"
    assert isinstance(ck["loader_batch_no"], int) and ck["loader_batch_no"] >= 0, "loader batch cursor"
    rank, node = ck["loader_rank_of_node"], ck["loader_node_at_rank"]
    assert all(torch.is_tensor(v) and v.dtype == torch.int64 and v.shape == (n_nodes,) for v in [rank, node]), "rank-order shape/dtype"
    assert int(rank.min()) >= 0 and int(rank.max()) < n_nodes and int(node.min()) >= 0 and int(node.max()) < n_nodes, "rank-order bounds"
    assert torch.equal(rank[node], torch.arange(n_nodes)), "rank-order inverse mismatch"
    assert np.isfinite(ck["rankneg_scale"]) and ck["rankneg_scale"] > 0, "rank scale invalid"


def strict_validate_arm(arm, ROOT, base=None):
    import torch, numpy as np
    base = Path(base) if base else TD_DEFAULT
    ad = base / f"admission-{arm}.json"; mp = base / f"manifest-{arm}.json"
    assert ad.exists() and mp.exists(), f"missing admission/manifest for {arm}"
    adm = json.loads(ad.read_text()); man = json.loads(mp.read_text()); exp = expected_identity(arm, ROOT)

    assert adm.get("card012_identity") == exp, f"{arm}: admission identity != canonical"
    assert man.get("card012_identity") == exp, f"{arm}: manifest identity != canonical"
    # intervention mandatory for shape_floor, forbidden for ordinary (unconditional)
    if arm == "shape_floor":
        assert exp["shape_weight"] > 0 and exp["shape_bank_X_sha256"] == BANK_X_SHA256, "shape_floor must apply the bank"
        assert man.get("shape_weight", 0) > 0, "shape_floor manifest weight not positive"
    else:
        assert exp["shape_weight"] == 0.0 and exp["shape_bank_X_sha256"] is None, "ordinary must NOT apply a bank"
        assert float(man.get("shape_weight", 0) or 0) == 0.0, "ordinary manifest carries a shape weight"
    # warm provenance bound to the parent (teacher) endpoint
    assert man.get("teacher_sha256") == TEACHER_SHA, f"{arm}: warm-start provenance != card013 baseline endpoint"
    assert man.get("warm_start_param_sha256") == WARM_PARAM_SHA, "actual warm parameter identity"
    assert adm.get("warm_start_param_sha256") == WARM_PARAM_SHA, "admitted warm parameter identity"
    ts = man.get("train_stats", {})
    exposed = int(ts.get("shape_successful_steps", 0))
    if arm == "shape_floor":
        assert exposed == DOSE and ts.get("shape_clouds_successful") == DOSE * CENTERS_PER_STEP, "shape exposure absent/incomplete"
        assert ts.get("shape_positive_loss_steps", 0) > 0 and np.isfinite(ts.get("shape_loss_sum", np.nan)), "no finite positive shape loss exposure"
    else:
        assert exposed == 0 and ts.get("shape_clouds_successful", 0) == 0, "ordinary shape exposure"
    assert man.get("executed_steps") == ts.get("positive_lr_optimizer_steps") == DOSE, f"{arm}: dose != {DOSE}"
    assert man.get("pipeline_info", {}).get("x_residency") == "device_fp16", f"{arm}: pipeline not device_fp16"
    assert abs(float(man.get("lr_used_min", 0)) - LR) < 1e-12 and abs(float(man.get("lr_used_max", 0)) - LR) < 1e-12, "LR not 1e-4"
    lm = man.get("loaded_modules", {})
    assert lm.get("verified_frozen_runtime") and lm.get("all_basemap_under_root"), f"{arm}: runtime not frozen/under-root"
    ok, bad = runtime_manifest_check(ROOT); assert ok, f"frozen runtime drifted at validate: {bad}"

    ep = torch.load(base / f"model-{arm}.pt", map_location="cpu", weights_only=False)
    ep_sd = ep["model_state_dict"]; assert all(bool(torch.isfinite(t).all()) for t in ep_sd.values()), "endpoint non-finite"
    ep_sha = state_sha(ep_sd); assert ep_sha == man.get("trained_sha256"), f"{arm}: endpoint sha != manifest"
    ad_mtime = ad.stat().st_mtime; snap_sha = {}
    for s in SNAP_STEPS:
        p = base / arm / f"model-step{s}.pt"; assert p.exists(), f"missing snapshot {s}"
        o = torch.load(p, map_location="cpu", weights_only=False); sd = o["model_state_dict"]
        assert o["n_components"] == NC and o["learning_rate"] == LR and o["lr_schedule"] == "constant", f"snapshot {s} config"
        assert all(bool(torch.isfinite(t).all()) for t in sd.values()) and p.stat().st_mtime >= ad_mtime, f"snapshot {s} bad/stale"
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
        validate_resume_payload(ck, s, n_nodes=N)
        assert int(ck.get("global_step", -1)) == s, f"step ckpt {s} global_step mismatch"
        assert bool(ck.get("step_checkpoint")), f"ckpt {s} not marked step_checkpoint"
        assert ck.get("card012_identity") == exp, f"step ckpt {s} identity != canonical"
        if arm == "shape_floor":
            sg = ck.get("shape_gen")
            assert torch.is_tensor(sg) and sg.dtype == torch.uint8 and sg.shape == (16,), "shape sampler CUDA RNG schema"
            assert ck["train_stats"].get("shape_successful_steps") == s, "checkpoint shape exposure"
        m = ck["model"]; assert all(bool(torch.isfinite(t).all()) for t in m.values()), f"step ckpt {s} non-finite"
        msha = state_sha(m)
        if s in snap_sha: assert msha == snap_sha[s], f"step ckpt {s} model payload != snapshot"
        ck_receipt[s] = msha

    cf = base / f"coords-{arm}.npy"; coords = np.load(cf, mmap_mode="r")
    assert coords.shape == (N, NC) and bool(np.isfinite(coords).all()), f"{arm}: coords bad"
    return {"arm": arm, "PASS": True, "model_state_sha": ep_sha, "model_file_sha": full_sha(base / f"model-{arm}.pt"),
            "coords_file_sha": full_sha(cf), "snapshot_sha": {str(k): v for k, v in snap_sha.items()},
            "step_ckpt_sha": {str(k): v for k, v in ck_receipt.items()}, "identity": exp}
