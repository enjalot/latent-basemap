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
             "init_sha256": INIT_SHA, "substrate_sha256": SUB_SHA256, "graph_sha256": GRAPH_SHA256,
             "sampler": "grouped shared-PERM positive stream; 9 uniform nonself noise/positive (with replacement)",
             "block_pos": BLOCK_POS, "n_noise": N_NOISE, "seed": SEED, "lr": LR, "lr_schedule": "constant",
             "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP, "dose": DOSE, "n_components": NC,
             "precision": "device_fp16", "runtime_manifest_sha256": full_sha(runtime_manifest_path(ROOT))}
    if arm == "grouped_nce":
        ident.update({"beta_init": 0.0, "scalar_learned": True, "scalar_lr": LR, "scalar_weight_decay": 0.0, "infonce_coeff": None})
    elif arm == "grouped_infonce":
        ident.update({"beta_init": None, "scalar_learned": False, "scalar_lr": None, "scalar_weight_decay": None,
                      "infonce_coeff": calibrated_coeff()})
    else:
        ident.update({"beta_init": None, "scalar_learned": False, "scalar_lr": None, "scalar_weight_decay": None, "infonce_coeff": None})
    return ident


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
    assert man.get("pipeline") == "device_fp16", f"{arm}: pipeline not device_fp16"
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

    ck_receipt = {}
    for s in STEP_CKPTS:
        p = base / arm / "ckpts" / f"ckpt-step{s}.pt"; assert p.exists(), f"missing resumable step ckpt {s}"
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:
            raise AssertionError(f"corrupt step checkpoint {s}: {e!r}")
        assert int(ck.get("global_step", -1)) == s, f"step ckpt {s} global_step mismatch"
        assert bool(ck.get("step_checkpoint")), f"ckpt {s} not marked step_checkpoint"
        assert ck.get("identity") == exp, f"step ckpt {s} identity != canonical"
        assert ck.get("sampler_state") is not None and "perm" in ck["sampler_state"], f"step ckpt {s} missing sampler state"
        assert ck.get("optimizer") is not None, f"step ckpt {s} missing optimizer state"
        if arm == "grouped_nce":
            assert ck.get("beta") is not None, f"grouped_nce step ckpt {s} missing learned scalar"
        m = ck["model"]; assert all(bool(torch.isfinite(t).all()) for t in m.values()), f"step ckpt {s} non-finite"
        msha = state_sha(m)
        if s in snap_sha: assert msha == snap_sha[s], f"step ckpt {s} model payload != snapshot"
        ck_receipt[s] = msha
    # at least one epoch checkpoint present (auditable epoch/step state)
    assert any((base / arm / "ckpts").glob("ckpt-epoch*.pt")), f"{arm}: no epoch checkpoint"

    # recorded 9:1 exposure at early/middle/final probes
    probes = ts.get("exposure_probes", [])
    assert probes and all(abs(float(pr.get("noise_per_pos", -1)) - N_NOISE) < 1e-9 for pr in probes), f"{arm}: 9:1 exposure not recorded/violated"
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
