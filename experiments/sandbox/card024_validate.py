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


def expected_identity(arm, ROOT):
    assert arm in ARMS, arm
    ident = {"card": "card024", "arm": arm, "mode": MODE[arm], "kernel": KERNEL[arm], "kernel_a": A, "kernel_b": B,
             "init_sha256": INIT_SHA, "substrate_sha256": SUB_SHA256, "graph_sha256": GRAPH_SHA256,
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


def strict_validate_arm(arm, ROOT, base=None):
    import torch, numpy as np
    base = Path(base) if base else TD_DEFAULT
    ad = base / f"admission-{arm}.json"; mp = base / f"manifest-{arm}.json"
    assert ad.exists() and mp.exists(), f"missing admission/manifest for {arm}"
    adm = json.loads(ad.read_text()); man = json.loads(mp.read_text()); exp = expected_identity(arm, ROOT)
    assert adm.get("card012_identity") == exp, f"{arm}: admission identity != canonical"
    assert man.get("card012_identity") == exp, f"{arm}: manifest identity != canonical"
    assert man.get("warm_init_sha256") == INIT_SHA, f"{arm}: warm init != fresh 2D 589895f0"
    ts = man.get("train_stats", {})
    assert man.get("executed_steps") == ts.get("positive_lr_optimizer_steps") == DOSE, f"{arm}: dose != {DOSE}"
    assert man.get("pipeline_info", {}).get("x_residency") == "device_fp16", f"{arm}: pipeline not device_fp16"
    assert abs(float(man.get("lr_used_min", 0)) - LR) < 1e-12 and abs(float(man.get("lr_used_max", 0)) - LR) < 1e-12, "LR not 1e-3"
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
        assert int(ck.get("global_step", -1)) == s, f"step ckpt {s} global_step mismatch"
        assert bool(ck.get("step_checkpoint")), f"ckpt {s} not marked step_checkpoint"
        assert ck.get("card012_identity") == exp, f"step ckpt {s} identity != canonical"
        if arm == "nce_learned":
            assert ck.get("card024_beta") is not None, f"nce_learned step ckpt {s} missing learned scalar"
        m = ck["model"]; assert all(bool(torch.isfinite(t).all()) for t in m.values()), f"step ckpt {s} non-finite"
        msha = state_sha(m)
        if s in snap_sha: assert msha == snap_sha[s], f"step ckpt {s} model payload != snapshot"
        ck_receipt[s] = msha

    # scalar EXPOSURE: no scientific result from a silently inactive scalar
    traj = ts.get("card024_beta_traj", []); final_beta = man.get("final_beta")
    if arm == "nce_learned":
        assert traj and final_beta is not None and abs(float(final_beta)) > 0, f"nce_learned scalar never moved (beta={final_beta})"
    elif arm == "neg_fixed":
        assert (final_beta is None) or float(final_beta) == 0.0, "neg_fixed scalar must stay 0"
    else:
        assert final_beta is None, "umap_uniform must have no scalar"

    cf = base / f"coords-{arm}.npy"; coords = np.load(cf, mmap_mode="r")
    assert coords.shape == (N, NC) and bool(np.isfinite(coords).all()), f"{arm}: coords bad"
    return {"arm": arm, "PASS": True, "model_state_sha": ep_sha, "model_file_sha": full_sha(base / f"model-{arm}.pt"),
            "coords_file_sha": full_sha(cf), "final_beta": final_beta, "snapshot_sha": {str(k): v for k, v in snap_sha.items()},
            "step_ckpt_sha": {str(k): v for k, v in ck_receipt.items()}, "identity": exp}
