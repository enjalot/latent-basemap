"""Card023 arm trainer (per card023-scale-2m.md + root review). Two matched fresh 3D heads from the frozen
Card015 init on the 2M substrate: ordinary3d (baseline kernel) / actual3d (d²/(r_i·r_j) detached, Hook C).
n_components=3, constant LR .001, batch 16384, pos_ratio .10, rankneg 500000 (25% of 2M), seed 42, EXACTLY
400000 successful updates. Inference snaps 50/100/200/300/400K; GENUINE resumable checkpoints at 50K step
intervals INCLUDING 400K + epoch boundaries, with the canonical immutable admission identity bound in.

Root-review repairs: explicit x_residency='auto' + required_input_pipeline='device' and a post-fit assert
that the pipeline is actually device_fp16 (not int8); FULL content-hash identity incl. code + data-manifest +
LR/seed/dose (card023_validate.expected_identity); ALL loaded basemap modules verified UNDER this worktree;
admission written ONCE and never rewritten on resume (resume attempts recorded separately, with rebuilt
identity matched to the original before fitting); latest-checkpoint corruption fails closed; idempotent skip
only after strict validation. Usage: run_card023_arm.py <ordinary3d|actual3d> <STEPS>
"""
import os, sys, time, json, hashlib, subprocess, re, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import card023_validate as V

ROOT = Path(__file__).resolve().parents[2]
SB = V.SB; OC = V.OC; DATA = V.DATA
CHAMPION = SB / "dino-arrival-t0/champion-bs16k/model.pt"
INIT3D = V.INIT3D; OUTD = SB / "card023-train"
SEED = V.SEED; LR = V.LR; BATCH = V.BATCH; RANKNEG = V.RANKNEG; N_EXPECT = V.N
SNAPS = tuple(V.SNAP_STEPS); STEP_CKPTS = set(V.STEP_CKPTS)
RMAP = V.ARM_RADII; INIT_SHA = V.INIT_NAMED_SHA; WARM_PARAM_SHA = V.WARM_PARAM_SHA


def _latest_ckpt(cdir):
    """Highest-global_step resumable/epoch checkpoint. Fail CLOSED on any corruption (never skip to older)."""
    import torch
    if not cdir.is_dir(): return None, 0
    cands = []
    for f in cdir.glob("ckpt-step*.pt"):
        m = re.search(r"ckpt-step(\d+)\.pt$", f.name)
        if m:
            try: ck=torch.load(f,map_location="cpu",weights_only=False)
            except Exception as e: raise RuntimeError(f"corrupt step checkpoint {f}: {e!r}")
            assert int(ck["global_step"])==int(m.group(1)), "step filename/state mismatch"
            cands.append((int(m.group(1)),f))
    for f in cdir.glob("ckpt-epoch*.pt"):
        try: gs = int(torch.load(f, map_location="cpu", weights_only=False)["global_step"])
        except Exception as e: raise RuntimeError(f"corrupt epoch checkpoint {f}: {e!r} — fail closed")
        cands.append((gs, f))
    if not cands: return None, 0
    gs, f = max(cands, key=lambda t: t[0])
    try: ck = torch.load(f, map_location="cpu", weights_only=False)   # verify chosen loads fully
    except Exception as e: raise RuntimeError(f"corrupt latest checkpoint {f}: {e!r} — fail closed")
    assert int(ck["global_step"]) == gs, f"chosen ckpt {f} global_step {ck['global_step']} != {gs}"
    return f, gs


def main():
    arm = sys.argv[1]; steps = int(sys.argv[2]); assert arm in RMAP and steps == V.DOSE, "arm/dose"
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    import basemap.pumap.parametric_umap.core as _core
    import basemap.pumap.parametric_umap.datasets.edge_list_dataset as _eld
    import torch
    OUTD.mkdir(parents=True, exist_ok=True)

    # idempotent completion: ONLY via the canonical strict validator (never manifest dose alone)
    if (OUTD / f"manifest-{arm}.json").exists():
        V.strict_validate_arm(arm, ROOT)
        print(f"[card023 {arm}] already complete + strict-valid; skip", flush=True); return 0

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "card023 arms are the production device path; refuse CPU fallback"
    free,total=torch.cuda.mem_get_info();assert (total-free)/2**30<12, "leave18GiB GPU headroom"
    # actual on-disk data files match the admitted FULL hashes (the honest binding behind the identity)
    assert V.full_sha(DATA / "edges-fixed15.npz") == V.EDGES_SHA256, "2M edges hash"
    assert V.full_sha(DATA / "substrate.f16.npy") == V.SUB_SHA256, "2M substrate hash"
    init_obj = torch.load(str(INIT3D), map_location="cpu", weights_only=False)
    init_state = init_obj["model_state"]
    assert init_obj["init_state_sha256"] == INIT_SHA and init_obj["n_components"] == 3, "Card015 3D init identity"
    radii = None; radii_sha = None
    if RMAP[arm] is not None:
        assert V.full_sha(DATA / RMAP[arm]) == V.R_ACTUAL_SHA256, "radius file identity"
        radii = np.load(DATA / RMAP[arm]).astype(np.float32); radii_sha = hashlib.sha256(np.ascontiguousarray(radii).tobytes()).hexdigest()[:16]
        assert radii.shape[0] == N_EXPECT and np.isfinite(radii).all() and (radii > 0).all(), "radii invalid"

    # frozen runtime + ALL loaded basemap modules must resolve under THIS worktree
    fok, bad = V.runtime_manifest_check(ROOT); assert fok, f"frozen-runtime hash mismatch {bad} — fail closed"
    basemap_mods = V.loaded_basemap_under_root(ROOT)     # asserts every basemap module under ROOT
    githead = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    loaded = {"core": {"path": _core.__file__, "sha": V.full_sha(_core.__file__)[:16]},
              "edge_list_dataset": {"path": _eld.__file__, "sha": V.full_sha(_eld.__file__)[:16]},
              "run_card023_arm": {"path": __file__, "sha": V.full_sha(__file__)[:16]},
              "git_head": githead, "worktree_root": str(ROOT), "verified_frozen_runtime": True,
              "all_basemap_under_root": True, "basemap_modules": basemap_mods,
              "runtime_manifest_sha256": V.full_sha(V.runtime_manifest_path(ROOT))}

    identity = V.expected_identity(arm, ROOT)     # canonical immutable admission identity
    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed_all(SEED); torch.cuda.reset_peak_memory_stats()
    pumap = ParametricUMAP.load(str(CHAMPION), device=dev); pumap.model = None
    pumap.n_components = 3
    pumap.learning_rate = LR; pumap.lr_schedule = "constant"; pumap.batch_size = BATCH; pumap.warmup_steps = 0
    pumap.n_epochs = 100000; pumap._max_train_steps = steps; pumap.rankneg_window = RANKNEG
    pumap.x_residency = "auto"; pumap.required_input_pipeline = "device"     # fail closed off the device path
    assert abs(float(getattr(pumap, "pos_ratio", -1)) - V.POS_RATIO) < 1e-9, "champion pos_ratio must be .10"
    for a, v in (("anchor_ids_path", ""), ("anchor_hold_weight", 0.0), ("replay_bank_path", ""),
                 ("replay_weight", 0.0), ("deriv_bank_path", ""), ("deriv_weight", 0.0)):
        if hasattr(pumap, a): setattr(pumap, a, v)
    if radii is not None: pumap._card013_radii = radii
    pumap._checkpoint_step_targets = set(STEP_CKPTS)
    pumap._card012_identity = identity

    SNAPDIR = OUTD / arm; SNAPDIR.mkdir(exist_ok=True); CKPTDIR = SNAPDIR / "ckpts"; CKPTDIR.mkdir(exist_ok=True)
    resume_from, resume_gs = _latest_ckpt(CKPTDIR)

    # admission is IMMUTABLE: write once on the fresh run; on resume, match rebuilt identity to it + log attempt
    adm_path = OUTD / f"admission-{arm}.json"
    if resume_from is None:
        adm = {"schema": "card023-admission-2026-09-12", "arm": arm, "written_before_steps": True, "n_components": 3,
               "warm_param_sha256": WARM_PARAM_SHA, "shared_init_sha256": INIT_SHA, "derived_from_2d": "589895f037d406ae",
               "substrate_sha": V.SUB_SHA256[:16], "edges_sha": V.EDGES_SHA256[:16], "lr": LR, "lr_schedule": "constant",
               "batch_size": BATCH, "pos_ratio": V.POS_RATIO, "seed": SEED, "steps": steps, "rankneg_window": RANKNEG,
               "radii_file": (str(DATA / RMAP[arm]) if radii is not None else None), "radii_sha": radii_sha,
               "kernel": identity["kernel"], "snapshots": list(SNAPS), "step_checkpoints": sorted(STEP_CKPTS),
               "checkpoint_every_epochs": 1, "precision": "device_fp16", "card012_identity": identity,
               "loaded_modules": loaded, "n_rows": N_EXPECT}
        assert not adm_path.exists(), "fresh run but admission already exists"
        adm_path.write_text(json.dumps(adm, indent=1))
    else:
        orig = json.loads(adm_path.read_text())
        assert orig.get("card012_identity") == identity, "resume rebuilt identity != immutable original admission"
        rec = {"at": dt.datetime.now(dt.timezone.utc).isoformat(), "resume_from": str(resume_from),
               "resume_global_step": resume_gs, "identity_matches_original": True,
               "original_warm_provenance": orig.get("warm_param_sha256"), "loaded_modules": loaded}
        (OUTD / f"resume-{arm}-{int(time.time())}.json").write_text(json.dumps(rec, indent=1))

    X = np.asarray(np.load(DATA / "substrate.f16.npy", mmap_mode="r"), np.float32); n = X.shape[0]; assert n == N_EXPECT
    t0 = time.time()
    pumap.fit(X, precomputed_edges_path=str(DATA / "edges-fixed15.npz"), random_state=SEED, verbose=False,
              warm_start_state=init_state, snapshot_steps=SNAPS, snapshot_dir=str(SNAPDIR),
              checkpoint_every_epochs=1, checkpoint_dir=str(CKPTDIR), resume_from=(str(resume_from) if resume_from else None))
    wall = time.time() - t0

    ts = dict(getattr(pumap, "_train_stats", {}) or {}); pinfo = dict(getattr(pumap, "_pipeline_info", {}) or {})
    exec_steps = int(ts.get("executed_iters", 0)); assert exec_steps == steps, f"{exec_steps} != {steps}"
    assert int(ts.get("positive_lr_optimizer_steps", -1)) == steps, "positive-LR count != dose"
    assert abs(ts.get("lr_used_min", 0) - LR) < 1e-12 and abs(ts.get("lr_used_max", 0) - LR) < 1e-12, "LR not .001"
    assert pinfo.get("x_residency") == "device_fp16", f"pipeline not device_fp16: {pinfo.get('x_residency')}"
    assert getattr(pumap, "warm_start_sha256", None) == WARM_PARAM_SHA, "warm-start param hash != 3D init"
    proc_peak = torch.cuda.max_memory_allocated() / 2**30
    free, total = torch.cuda.mem_get_info(); global_used = (total - free) / 2**30
    assert global_used < 30.0, f"global VRAM {global_used} GiB exceeds 30 GiB cap"

    coords = np.asarray(pumap.transform(X, batch_size=8192), np.float32); assert coords.shape == (n, 3)
    np.save(OUTD / f"coords-{arm}.npy", coords); pumap.save(str(OUTD / f"model-{arm}.pt"))
    man = {"schema": "card023-arm-2026-09-12", "arm": arm, "n": int(n), "n_components": 3, "executed_steps": exec_steps,
           "warm_param_sha256": pumap.warm_start_sha256, "shared_init_sha256": INIT_SHA, "trained_sha256": V.state_sha(pumap.model.state_dict()),
           "lr_used_min": ts.get("lr_used_min"), "lr_used_max": ts.get("lr_used_max"), "radii_sha": radii_sha,
           "kernel": identity["kernel"], "snapshots": list(SNAPS), "step_checkpoints": sorted(STEP_CKPTS),
           "pipeline_info": pinfo, "loaded_modules": loaded, "card012_identity": identity,
           "resumed_from": (str(resume_from) if resume_from else None), "resume_global_step": resume_gs,
           "train_wall_s": round(wall, 1), "it_per_s": round(exec_steps / wall, 2) if wall > 0 else None,
           "proc_peak_vram_gb": proc_peak, "global_vram_used_gb": global_used, "stop_reason": ts.get("stop_reason"),
           "train_stats": ts, "model_file_sha256": V.full_sha(OUTD/f"model-{arm}.pt"), "coords_file_sha256": V.full_sha(OUTD/f"coords-{arm}.npy")}
    (OUTD / f"manifest-{arm}.json").write_text(json.dumps(man, indent=1))
    V.strict_validate_arm(arm, ROOT)     # fail closed here if the completed arm is not strictly valid
    print(f"[card023 {arm}] steps={exec_steps} {wall:.0f}s 3D proc={proc_peak}GB global={global_used}GB "
          f"pipe={pinfo.get('x_residency')} resumed={bool(resume_from)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
