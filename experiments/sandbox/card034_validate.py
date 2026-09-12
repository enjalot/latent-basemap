"""Card034 canonical strict validator + identity. ONE source of truth reused by the standalone trainer,
calibration, canary and chain. CPU-only.

expected_identity(arm) is the immutable admission identity bound into every checkpoint: the grouped objective
MODE, the phi kernel (a,b), the fresh 2D init hash, the 300K substrate + fixed15 graph hashes, the grouped
sampler recipe (block_pos, n_noise=9, uniform nonself), seed/LR/weight-decay/grad-clip/dose/precision, the
runtime-manifest hash, and — grouped_nce — the learned-scalar spec, — grouped_infonce — the frozen init-scale
coefficient. Deterministic ⇒ a resume rebuilds it and the trainer rejects a wrong objective/coefficient/seed/
data/scalar before restore. strict_validate_arm does full completion validation over the STANDALONE checkpoint
schema (exact 60K successful positive-LR dose, fresh init, constant LR, endpoint==60K snapshot, snapshot
progress+freshness, resumable step+epoch ckpts with bound identity + sampler state + recovered scalar (nce),
recorded 9:1 exposure). No manifest-only completion; corrupt checkpoints fail closed.
"""
import hashlib, json
from pathlib import Path

SB = Path("/data/latent-basemap/sandbox"); OC = SB / "overseer-codex"
CHAMPION = SB / "dino-arrival-t0/champion-bs16k/model.pt"
SUBD = Path("/data/latent-basemap/substrates/card010-adaptive")
SUB = SUBD / "substrate.f16.npy"; GRAPH = SUBD / "edges-fixed15.npz"; INIT = SUBD / "init-card010.pt"
TD_DEFAULT = SB / "card034-train"; CALIB = OC / "card034-calibration.json"
INIT_SHA = "589895f037d406ae"
SUB_SHA256 = "873d76e35eb2151e966c1c4dabb05e113c53f9cbc6e8473d87b6bd25029e5f45"
GRAPH_SHA256 = "d214a839b07113dff2c29b225da9f38008f86a0b2cb3662a39d14bc0542d4450"
A, B = 1.9328, 0.7905
N = 300000; NC = 2; DOSE = 60000; LR = 1e-3; WEIGHT_DECAY = 0.01; GRAD_CLIP = 1.0
BLOCK_POS = 1638; N_NOISE = 9; SEED = 42
SNAP_STEPS = [20000, 40000, 60000]; STEP_CKPTS = [20000, 40000, 60000]
ARMS = ["grouped_umap", "grouped_nce", "grouped_infonce"]
MODE = {a: a for a in ARMS}
KERNEL = {"grouped_umap": "grouped BCE(phi) 1pos/9noise", "grouped_nce": "grouped BCE-logit log(phi)-beta",
          "grouped_infonce": "grouped logsumexp(log phi over 10) - log phi_pos"}


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


def runtime_manifest_path(ROOT): return Path(ROOT) / "card034-runtime-sha.json"
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


def calibrated_coeff():
    """Frozen InfoNCE init-scale coefficient from the calibration step (raises if absent/invalid)."""
    c = json.loads(CALIB.read_text()); assert c.get("PASS"), "calibration not PASS"
    w = float(c["coefficient"]); assert w > 0 and w == w and w != float("inf"), "calibration coefficient invalid"
    return w


def expected_identity(arm, ROOT):
    assert arm in ARMS, arm
    ident = {"card": "card034", "arm": arm, "mode": MODE[arm], "kernel": KERNEL[arm], "kernel_a": A, "kernel_b": B,
             "init_sha256": INIT_SHA, "init_file_sha256": full_sha(INIT), "substrate_sha256": SUB_SHA256, "graph_sha256": GRAPH_SHA256,
             "sampler": "grouped shared-PERM positive stream; 9 uniform nonself noise/positive (with replacement)",
             "block_pos": BLOCK_POS, "n_noise": N_NOISE, "seed": SEED, "lr": LR, "lr_schedule": "constant",
             "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP, "dose": DOSE, "n_components": NC,
             "precision": "device_fp16", "runtime_manifest_sha256": full_sha(runtime_manifest_path(ROOT))}
    if arm == "grouped_nce":
        ident.update({"beta_init": 0.0, "scalar_learned": True, "scalar_lr": LR, "scalar_weight_decay": 0.0,
                      "infonce_coeff": None, "infonce_calib_sha256": None})
    elif arm == "grouped_infonce":
        ident.update({"beta_init": None, "scalar_learned": False, "scalar_lr": None, "scalar_weight_decay": None,
                      "infonce_coeff": calibrated_coeff(), "infonce_calib_sha256": full_sha(CALIB)})
    else:
        ident.update({"beta_init": None, "scalar_learned": False, "scalar_lr": None, "scalar_weight_decay": None,
                      "infonce_coeff": None, "infonce_calib_sha256": None})
    return ident


def validate_ckpt_payload(ck, arm, ROOT, identity, n_nodes, expect_beta, expect_step=None, model_sd=None, expect_perm_len=None):
    """Deep canonical checkpoint validation (items 4/5). Adam groups LR/WD + per-parameter finite moments whose
    SHAPES match the actual parameters (incl. the scalar), correct step counter; scalar param/value; scaler
    STRUCTURE (finite positive scale + _growth_tracker); CPU/CUDA RNG STRUCTURE (uint8 tensor / list of uint8
    tensors); full valid sampler permutation of the EXPECTED length + cursor/epoch + numpy generator-state
    STRUCTURE; live (== global_step) stats; finite model. Reused BEFORE restore and at completion. Raises on
    any drift."""
    import torch, numpy as np
    assert ck.get("schema") == "card034-ckpt-2026-09-12", "ckpt schema"
    assert ck.get("identity") == identity, "ckpt identity != canonical (wrong objective/coeff/seed/data/scalar)"
    gs = int(ck["global_step"]); assert gs >= 0 and isinstance(ck.get("epoch"), int) and ck["epoch"] >= 0, "global_step/epoch"
    if expect_step is not None: assert gs == int(expect_step), f"ckpt step {gs} != {expect_step}"
    ts = ck.get("train_stats", {})
    assert int(ts.get("positive_lr_optimizer_steps", -1)) == gs, "stale/zero checkpoint stats (must equal global_step)"
    ck_model = ck["model"]; assert all(bool(torch.isfinite(t).all()) for t in ck_model.values()), "nonfinite model in ckpt"
    # actual ordered parameter shapes (model params in state_dict order, then the scalar) for moment mapping
    ref = model_sd if model_sd is not None else expected_model_state()
    assert set(ck_model)==set(ref) and all(ck_model[k].shape==ref[k].shape for k in ref), "model parameter keys/shapes mismatch"
    param_shapes = [tuple(v.shape) for v in ref.values()]
    if expect_beta: param_shapes.append(())                       # scalar beta has shape ()
    opt = ck.get("optimizer", {}); groups = opt.get("param_groups", []); state = opt.get("state", {})
    assert groups and state, "empty optimizer state/groups"
    assert groups[0]["lr"] == LR and groups[0]["weight_decay"] == WEIGHT_DECAY, "model Adam group lr/wd"
    if expect_beta:
        assert len(groups) == 2 and groups[1]["lr"] == LR and groups[1]["weight_decay"] == 0.0, "scalar Adam group lr/wd"
        assert ck.get("beta") is not None and bool(torch.isfinite(ck["beta"]).all()) and tuple(ck["beta"].shape) == (), "missing/nonfinite/nonscalar beta"
    else:
        assert len(groups) == 1, "unexpected extra optimizer group"
    pids = [i for g in groups for i in g["params"]]
    assert len(pids) == len(set(pids)) == len(param_shapes) and set(pids) == set(state), "optimizer parameter identity/count"
    for j, pid in enumerate(pids):
        st = state[pid]; assert all(k in st for k in ("step", "exp_avg", "exp_avg_sq")), "missing Adam moments"
        assert float(st["step"]) == gs, "Adam step counter != successful dose"
        want = torch.Size(param_shapes[j])                        # moment shape == actual parameter shape
        assert tuple(st["exp_avg"].shape) == tuple(want) and tuple(st["exp_avg_sq"].shape) == tuple(want), "Adam moment shape != parameter shape"
        assert bool(torch.isfinite(st["exp_avg"]).all()) and bool(torch.isfinite(st["exp_avg_sq"]).all()), "nonfinite Adam moment"
    # scaler structure
    sc = ck.get("scaler", {}); assert isinstance(sc, dict) and "scale" in sc and "_growth_tracker" in sc, "scaler structure"
    _scale = float(sc["scale"]); assert np.isfinite(_scale) and _scale > 0, "scaler scale not finite positive"
    assert int(sc["_growth_tracker"]) >= 0 and float(sc["growth_factor"]) > 1 and 0 < float(sc["backoff_factor"]) < 1 and int(sc["growth_interval"]) > 0, "invalid scaler settings"
    # RNG structure
    tr = ck.get("torch_rng"); assert torch.is_tensor(tr) and tr.dtype == torch.uint8 and tr.ndim==1 and tr.numel()==torch.get_rng_state().numel(), "torch RNG structure"
    cr = ck.get("cuda_rng"); assert isinstance(cr, (list, tuple)) and len(cr) >= 1 and all(torch.is_tensor(x) and x.dtype == torch.uint8 and x.ndim==1 and x.numel()>=16 for x in cr), "cuda RNG structure"
    # sampler permutation + cursor/epoch + generator-state structure
    ss = ck.get("sampler_state", {}); perm = np.asarray(ss.get("perm"))
    assert perm.ndim == 1 and perm.shape[0] > 0 and np.array_equal(np.sort(perm), np.arange(perm.shape[0])), "sampler perm not a valid permutation"
    if expect_perm_len is None: expect_perm_len=n_nodes*15
    assert perm.shape[0] == int(expect_perm_len), f"perm length {perm.shape[0]} != {expect_perm_len}"
    assert 0 <= int(ss["cursor"]) <= perm.shape[0] and int(ss["epoch"]) >= 0, "sampler cursor/epoch"
    for gk in ("perm_gen", "noise_gen"):
        assert isinstance(ss.get(gk), dict) and ss[gk].get("bit_generator")=="PCG64", f"sampler {gk} generator-state structure"
        gen=np.random.default_rng();gen.bit_generator.state=ss[gk]  # real deserialization before restore
    assert int(ss["epoch"])==ck["epoch"], "epoch/sampler mismatch"
    expected_coeff=identity.get("infonce_coeff") or identity.get("coeff") or 1.0
    assert float(ck["coeff"])==float(expected_coeff), "payload coefficient mismatch"
    return True


def expected_model_state():
    import torch
    state=torch.load(INIT,map_location='cpu',weights_only=False)['model_state']
    assert state_sha(state)==INIT_SHA, 'warm-init actual tensors changed'
    return state


def strict_validate_arm(arm, ROOT, base=None):
    import torch, numpy as np
    base = Path(base) if base else TD_DEFAULT
    ad = base / f"admission-{arm}.json"; mp = base / f"manifest-{arm}.json"
    assert ad.exists() and mp.exists(), f"missing admission/manifest for {arm}"
    adm = json.loads(ad.read_text()); man = json.loads(mp.read_text()); exp = expected_identity(arm, ROOT)
    assert adm.get("identity") == exp, f"{arm}: admission identity != canonical"
    assert man.get("identity") == exp, f"{arm}: manifest identity != canonical"
    assert man.get("warm_init_sha256") == INIT_SHA, f"{arm}: warm init != fresh 2D 589895f0"
    ts = man.get("train_stats", {})
    assert man.get("executed_steps") == ts.get("positive_lr_optimizer_steps") == DOSE, f"{arm}: dose != {DOSE}"
    # explicit experimental pipeline receipt bound to actual Xt observations (not a borrowed core fit receipt)
    pr = man.get("pipeline_receipt", {})
    assert pr.get("x_residency") == "device_fp16" and pr.get("verified") and "cuda" in str(pr.get("device", "")) \
        and "float16" in str(pr.get("dtype", "")) and tuple(pr.get("shape", ())) == (N, 1536), f"{arm}: pipeline receipt invalid"
    assert abs(float(man.get("lr_used_min", 0)) - LR) < 1e-12 and abs(float(man.get("lr_used_max", 0)) - LR) < 1e-12, "LR not 1e-3"
    assert float(man.get("weight_decay", -1)) == WEIGHT_DECAY and float(man.get("grad_clip", -1)) == GRAD_CLIP, "wd/clip mismatch"
    lm = man.get("loaded_modules", {})
    assert lm.get("verified_frozen_runtime") and lm.get("all_basemap_under_root"), f"{arm}: runtime not frozen/under-root"
    ok, bad = runtime_manifest_check(ROOT); assert ok, f"frozen runtime drifted: {bad}"

    ep = torch.load(base / f"model-{arm}.pt", map_location="cpu", weights_only=False)
    ep_sd = ep["model_state_dict"]; assert all(bool(torch.isfinite(t).all()) for t in ep_sd.values()), "endpoint non-finite"
    ep_sha = state_sha(ep_sd); assert ep_sha == man.get("trained_sha256"), f"{arm}: endpoint sha != manifest"
    ad_mtime = ad.stat().st_mtime; snap_sha = {}
    for s in SNAP_STEPS:
        p = base / arm / f"model-step{s}.pt"; assert p.exists(), f"missing snapshot {s}"
        o = torch.load(p, map_location="cpu", weights_only=False); sd = o["model_state_dict"]
        assert o.get("n_components") == NC, f"snapshot {s} n_components"
        assert all(bool(torch.isfinite(t).all()) for t in sd.values()) and p.stat().st_mtime >= ad_mtime, f"snapshot {s} bad/stale"
        snap_sha[s] = state_sha(sd)
    assert len(set(snap_sha.values())) == len(SNAP_STEPS), f"{arm}: snapshots did not progress"
    assert snap_sha[60000] == ep_sha, f"{arm}: endpoint != 60K snapshot"

    expect_beta = (arm == "grouped_nce")
    ck_receipt = {}
    for s in STEP_CKPTS:
        p = base / arm / "ckpts" / f"ckpt-step{s}.pt"; assert p.exists(), f"missing resumable step ckpt {s}"
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:
            raise AssertionError(f"corrupt step checkpoint {s}: {e!r}")
        validate_ckpt_payload(ck, arm, ROOT, exp, N, expect_beta, expect_step=s, model_sd=expected_model_state(), expect_perm_len=N * 15)   # DEEP
        assert bool(ck.get("step_checkpoint")), f"ckpt {s} not marked step_checkpoint"
        assert p.stat().st_mtime>=ad_mtime, "stale step checkpoint"
        msha = state_sha(ck["model"])
        if s in snap_sha: assert msha == snap_sha[s], f"step ckpt {s} model payload != snapshot"
        ck_receipt[s] = msha
    # retained epoch checkpoint(s) deep-validated too (not just filename existence)
    epoch_cks = sorted((base / arm / "ckpts").glob("ckpt-epoch*.pt"))
    assert epoch_cks, f"{arm}: no epoch checkpoint"
    for ep_p in epoch_cks:
        eck = torch.load(ep_p, map_location="cpu", weights_only=False)
        assert ep_p.stat().st_mtime>=ad_mtime, "stale epoch checkpoint"
        validate_ckpt_payload(eck, arm, ROOT, exp, N, expect_beta, model_sd=expected_model_state(), expect_perm_len=N * 15)   # coherent full state
    assert ck["train_stats"]==ts, "final checkpoint stats != manifest"
    # init payload hash: manifest must record the hash of the actual warm tensors == INIT_SHA (not trusted metadata)
    assert adm.get("init_payload_sha256")==INIT_SHA and man.get("init_payload_sha256") == INIT_SHA, f"{arm}: init payload hash not verified/recorded"

    # recorded 9:1 exposure at BOTH attempted-step and successful-step probes (AMP differences kept honest)
    succ_probes = ts.get("exposure_probes", []); att_probes = ts.get("attempted_probes", [])
    assert [q.get('step') for q in succ_probes]==[1,30000,60000], 'wrong/missing successful probes'
    assert [q.get('attempted_step') for q in att_probes]==[1,30000,60000], 'wrong/missing attempted probes'
    assert ts['attempted_steps']==DOSE+ts['amp_skips']+ts['nonfinite_skips'], 'attempt/skip count mismatch'
    assert ts['successful_noise']==9*ts['successful_positive'] and ts['attempted_noise']==9*ts['attempted_positive'], 'actual exposure not9:1'
    assert 0<ts['successful_positive']<=ts['attempted_positive']<=ts['attempted_steps']*BLOCK_POS, 'invalid actual exposure'

    assert succ_probes and all(abs(float(q.get("noise_per_pos", -1)) - N_NOISE) < 1e-9 and q.get("id_digest") for q in succ_probes), f"{arm}: successful-step 9:1 exposure not recorded/violated"
    assert att_probes and all(abs(float(q.get("noise_per_pos", -1)) - N_NOISE) < 1e-9 and q.get("id_digest") for q in att_probes), f"{arm}: attempted-step 9:1 exposure not recorded/violated"
    # scalar exposure (nce): beta must have moved
    if arm == "grouped_nce":
        fb = man.get("final_beta"); assert fb is not None and abs(float(fb)) > 0, "grouped_nce scalar never moved"
    else:
        assert man.get("final_beta") in (None, 0.0), f"{arm} must have no learned scalar"

    cf = base / f"coords-{arm}.npy"; coords = np.load(cf, mmap_mode="r")
    assert coords.shape == (N, NC) and bool(np.isfinite(coords).all()), f"{arm}: coords bad"
    return {"arm": arm, "PASS": True, "model_state_sha": ep_sha, "model_file_sha": full_sha(base / f"model-{arm}.pt"),
            "coords_file_sha": full_sha(cf), "final_beta": man.get("final_beta"), "infonce_coeff": exp.get("infonce_coeff"),
            "snapshot_sha": {str(k): v for k, v in snap_sha.items()}, "step_ckpt_sha": {str(k): v for k, v in ck_receipt.items()}, "identity": exp}
