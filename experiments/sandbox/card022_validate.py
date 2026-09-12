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


def calibrated_weight():
    """Frozen shape coefficient from the calibration step (raises if absent — required before shape_floor)."""
    c = json.loads(CALIB.read_text()); assert c.get("PASS"), "calibration not PASS"
    w = float(c["coefficient"]); assert w > 0 and w == w and w != float("inf"), "calibration coefficient invalid"
    return w


def expected_identity(arm, ROOT):
    assert arm in ARMS, arm
    common = {"card": "card022", "arm": arm, "kernel": KERNEL[arm],
              "teacher_sha256": TEACHER_SHA, "substrate_sha256": SUB_SHA256, "graph_sha256": GRAPH_SHA256,
              "seed": SEED, "lr": LR, "lr_schedule": "constant", "dose": DOSE, "rankneg_window": RANKNEG,
              "batch_size": BATCH, "pos_ratio": POS_RATIO, "n_components": NC,
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
    ts = man.get("train_stats", {})
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
        assert int(ck.get("global_step", -1)) == s, f"step ckpt {s} global_step mismatch"
        assert bool(ck.get("step_checkpoint")), f"ckpt {s} not marked step_checkpoint"
        assert ck.get("card012_identity") == exp, f"step ckpt {s} identity != canonical"
        if arm == "shape_floor":
            assert ck.get("shape_gen") is not None, f"shape_floor step ckpt {s} missing bank-sampler RNG"
        m = ck["model"]; assert all(bool(torch.isfinite(t).all()) for t in m.values()), f"step ckpt {s} non-finite"
        msha = state_sha(m)
        if s in snap_sha: assert msha == snap_sha[s], f"step ckpt {s} model payload != snapshot"
        ck_receipt[s] = msha

    cf = base / f"coords-{arm}.npy"; coords = np.load(cf, mmap_mode="r")
    assert coords.shape == (N, NC) and bool(np.isfinite(coords).all()), f"{arm}: coords bad"
    return {"arm": arm, "PASS": True, "model_state_sha": ep_sha, "model_file_sha": full_sha(base / f"model-{arm}.pt"),
            "coords_file_sha": full_sha(cf), "snapshot_sha": {str(k): v for k, v in snap_sha.items()},
            "step_ckpt_sha": {str(k): v for k, v in ck_receipt.items()}, "identity": exp}
