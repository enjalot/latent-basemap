"""Card022 arm trainer (per card022-local-shape-floor.md + root review/bank). Two matched 60K CONTINUATION
arms from the frozen Card013 baseline 2D endpoint on the fixed 300K full-DINO1536 draw:
  - ordinary:    matched continuation, no new term (default-off hook);
  - shape_floor: identical continuation + the covariance-floor term (calibrated coefficient).
Same original binary fixed15 graph, seed 42, batch 16384, pos_ratio .1, rankneg 75000, constant LR 1e-4
reapplied after load, FRESH AdamW, device_fp16, EXACTLY 60000 successful positive-LR steps. n_components=2.
Snapshots 20/40/60K; resumable step checkpoints at 20/40/60K + epoch boundaries with the canonical immutable
identity + (shape_floor) the independent bank-sampler RNG bound in. Full content-hash identity (parent/graph/
substrate + shape bank X/tau/manifest + epsilon + calibrated weight) via card022_validate.expected_identity.
Auto-resumes from the latest checkpoint. Usage: run_card022_arm.py <ordinary|shape_floor> <STEPS>
"""
import os, sys, time, json, hashlib, subprocess, re, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import card022_validate as V

ROOT = Path(__file__).resolve().parents[2]
SB = V.SB; OC = V.OC; PARENT = V.PARENT; SUB = V.SUB; GRAPH = V.GRAPH; BANKD = V.BANKD
OUTD = SB / "card022-train"
SEED = V.SEED; LR = V.LR; BATCH = V.BATCH; RANKNEG = V.RANKNEG; N_EXPECT = V.N; NC = V.NC
SNAPS = tuple(V.SNAP_STEPS); STEP_CKPTS = set(V.STEP_CKPTS); ARMS = V.ARMS


def _latest_ckpt(cdir):
    """Highest-global_step resumable/epoch checkpoint. Fail CLOSED on any corruption."""
    import torch
    if not cdir.is_dir(): return None, 0
    cands = []
    for f in cdir.glob("ckpt-step*.pt"):
        m = re.search(r"ckpt-step(\d+)\.pt$", f.name)
        if m:
            try: ck = torch.load(f, map_location="cpu", weights_only=False)
            except Exception as e: raise RuntimeError(f"corrupt step checkpoint {f}: {e!r}")
            assert int(ck["global_step"]) == int(m.group(1)), "step filename/state mismatch"
            cands.append((int(m.group(1)), f))
    for f in cdir.glob("ckpt-epoch*.pt"):
        try: gs = int(torch.load(f, map_location="cpu", weights_only=False)["global_step"])
        except Exception as e: raise RuntimeError(f"corrupt epoch checkpoint {f}: {e!r} — fail closed")
        cands.append((gs, f))
    if not cands: return None, 0
    gs, f = max(cands, key=lambda t: t[0])
    try: ck = torch.load(f, map_location="cpu", weights_only=False)
    except Exception as e: raise RuntimeError(f"corrupt latest checkpoint {f}: {e!r} — fail closed")
    assert int(ck["global_step"]) == gs; return f, gs


def main():
    arm = sys.argv[1]; steps = int(sys.argv[2]); assert arm in ARMS and steps == V.DOSE, "arm/dose"
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    import basemap.pumap.parametric_umap.core as _core
    import basemap.pumap.parametric_umap.datasets.edge_list_dataset as _eld
    import torch
    OUTD.mkdir(parents=True, exist_ok=True)

    if (OUTD / f"manifest-{arm}.json").exists():
        V.strict_validate_arm(arm, ROOT); print(f"[card022 {arm}] already complete + strict-valid; skip", flush=True); return 0

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "card022 arms are the production device path; refuse CPU fallback"
    free, total = torch.cuda.mem_get_info(); assert (total - free) / 2**30 < 12, "leave GPU headroom before load"
    assert V.full_sha(SUB) == V.SUB_SHA256, "300K substrate hash"
    assert V.full_sha(GRAPH) == V.GRAPH_SHA256, "fixed15 graph hash"
    assert V.full_sha(PARENT) == V.TEACHER_SHA, "card013 baseline endpoint (teacher) hash"

    # warm-start = the parent's weights (continuation); core allocates fresh model + injects after admission
    parent = torch.load(str(PARENT), map_location="cpu", weights_only=False)
    warm_state = parent["model_state_dict"]

    radii = None  # (none for card022; the intervention is the shape floor)
    shape_bank = None; shape_weight = 0.0
    if arm == "shape_floor":
        assert V.full_sha(BANKD / "X.npy") == V.BANK_X_SHA256, "shape bank X hash"
        assert V.full_sha(BANKD / "tau.npy") == V.BANK_TAU_SHA256, "shape bank tau hash"
        bx = np.load(BANKD / "X.npy", mmap_mode="r"); btau = np.load(BANKD / "tau.npy")
        assert bx.shape == (20000, 16, V.DIM) and btau.shape == (20000,), "bank shapes"
        shape_bank = {"X": np.asarray(bx, np.float16), "tau": np.asarray(btau, np.float32)}
        shape_weight = V.calibrated_weight()

    fok, bad = V.runtime_manifest_check(ROOT); assert fok, f"frozen-runtime hash mismatch {bad} — fail closed"
    basemap_mods = V.loaded_basemap_under_root(ROOT)
    githead = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    loaded = {"core": {"path": _core.__file__, "sha": V.full_sha(_core.__file__)[:16]},
              "edge_list_dataset": {"path": _eld.__file__, "sha": V.full_sha(_eld.__file__)[:16]},
              "run_card022_arm": {"path": __file__, "sha": V.full_sha(__file__)[:16]},
              "git_head": githead, "worktree_root": str(ROOT), "verified_frozen_runtime": True,
              "all_basemap_under_root": True, "basemap_modules": basemap_mods,
              "runtime_manifest_sha256": V.full_sha(V.runtime_manifest_path(ROOT))}
    identity = V.expected_identity(arm, ROOT)

    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed_all(SEED); torch.cuda.reset_peak_memory_stats()
    pumap = ParametricUMAP.load(str(PARENT), device=dev); pumap.model = None   # keep parent CONFIG, inject weights as warm-start
    assert pumap.n_components == NC, f"parent must be {NC}D"
    pumap.learning_rate = LR; pumap.lr_schedule = "constant"; pumap.batch_size = BATCH; pumap.warmup_steps = 0
    pumap.n_epochs = 100000; pumap._max_train_steps = steps; pumap.rankneg_window = RANKNEG
    pumap.x_residency = "auto"; pumap.required_input_pipeline = "device"
    assert abs(float(getattr(pumap, "pos_ratio", -1)) - V.POS_RATIO) < 1e-9, "parent pos_ratio must be .1"
    for a, v in (("anchor_ids_path", ""), ("anchor_hold_weight", 0.0), ("replay_bank_path", ""),
                 ("replay_weight", 0.0), ("deriv_bank_path", ""), ("deriv_weight", 0.0)):
        if hasattr(pumap, a): setattr(pumap, a, v)
    if shape_bank is not None:
        pumap._shape_bank = shape_bank; pumap._shape_weight = float(shape_weight)
        pumap._shape_eps = float(V.EPSILON); pumap._shape_centers_per_step = V.CENTERS_PER_STEP
    pumap._checkpoint_step_targets = set(STEP_CKPTS); pumap._card012_identity = identity

    SNAPDIR = OUTD / arm; SNAPDIR.mkdir(exist_ok=True); CKPTDIR = SNAPDIR / "ckpts"; CKPTDIR.mkdir(exist_ok=True)
    resume_from, resume_gs = _latest_ckpt(CKPTDIR)
    adm_path = OUTD / f"admission-{arm}.json"
    if resume_from is None:
        adm = {"schema": "card022-admission-2026-09-12", "arm": arm, "written_before_steps": True, "n_components": NC,
               "teacher_sha256": V.TEACHER_SHA, "substrate_sha": V.SUB_SHA256[:16], "graph_sha": V.GRAPH_SHA256[:16],
               "lr": LR, "lr_schedule": "constant", "batch_size": BATCH, "pos_ratio": V.POS_RATIO, "seed": SEED,
               "steps": steps, "rankneg_window": RANKNEG, "shape_weight": shape_weight, "kernel": identity["kernel"],
               "epsilon": (V.EPSILON if arm == "shape_floor" else None),
               "shape_bank_X_sha256": (V.BANK_X_SHA256 if arm == "shape_floor" else None),
               "shape_bank_tau_sha256": (V.BANK_TAU_SHA256 if arm == "shape_floor" else None),
               "snapshots": list(SNAPS), "step_checkpoints": sorted(STEP_CKPTS), "checkpoint_every_epochs": 1,
               "precision": "device_fp16", "card012_identity": identity, "loaded_modules": loaded, "n_rows": N_EXPECT}
        assert not adm_path.exists(); adm_path.write_text(json.dumps(adm, indent=1))
    else:
        orig = json.loads(adm_path.read_text())
        assert orig.get("card012_identity") == identity, "resume rebuilt identity != immutable original admission"
        (OUTD / f"resume-{arm}-{int(time.time())}.json").write_text(json.dumps(
            {"at": dt.datetime.now(dt.timezone.utc).isoformat(), "resume_from": str(resume_from),
             "resume_global_step": resume_gs, "identity_matches_original": True,
             "original_teacher_provenance": orig.get("teacher_sha256"), "loaded_modules": loaded}, indent=1))

    X = np.asarray(np.load(SUB, mmap_mode="r"), np.float32); n = X.shape[0]; assert n == N_EXPECT
    t0 = time.time()
    pumap.fit(X, precomputed_edges_path=str(GRAPH), random_state=SEED, verbose=False,
              warm_start_state=(None if resume_from else warm_state), snapshot_steps=SNAPS, snapshot_dir=str(SNAPDIR),
              checkpoint_every_epochs=1, checkpoint_dir=str(CKPTDIR), resume_from=(str(resume_from) if resume_from else None))
    wall = time.time() - t0

    ts = dict(getattr(pumap, "_train_stats", {}) or {}); pinfo = dict(getattr(pumap, "_pipeline_info", {}) or {})
    exec_steps = int(ts.get("executed_iters", 0)); assert exec_steps == steps, f"{exec_steps} != {steps}"
    assert int(ts.get("positive_lr_optimizer_steps", -1)) == steps, "positive-LR count != dose"
    assert abs(ts.get("lr_used_min", 0) - LR) < 1e-12 and abs(ts.get("lr_used_max", 0) - LR) < 1e-12, "LR not 1e-4"
    assert pinfo.get("x_residency") == "device_fp16", f"pipeline not device_fp16: {pinfo.get('x_residency')}"
    if resume_from is None:
        assert getattr(pumap, "warm_start_sha256", None), "warm-start param hash not recorded"
    proc_peak = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    free, total = torch.cuda.mem_get_info(); global_used = round((total - free) / 2**30, 3)
    assert global_used < 30.0, f"global VRAM {global_used} GiB exceeds 30 GiB cap"

    coords = np.asarray(pumap.transform(X, batch_size=8192), np.float32); assert coords.shape == (n, NC)
    np.save(OUTD / f"coords-{arm}.npy", coords); pumap.save(str(OUTD / f"model-{arm}.pt"))
    man = {"schema": "card022-arm-2026-09-12", "arm": arm, "n": int(n), "n_components": NC, "executed_steps": exec_steps,
           "teacher_sha256": V.TEACHER_SHA, "warm_start_param_sha256": getattr(pumap, "warm_start_sha256", None),
           "trained_sha256": V.state_sha(pumap.model.state_dict()), "lr_used_min": ts.get("lr_used_min"),
           "lr_used_max": ts.get("lr_used_max"), "shape_weight": shape_weight, "kernel": identity["kernel"],
           "snapshots": list(SNAPS), "step_checkpoints": sorted(STEP_CKPTS), "pipeline_info": pinfo,
           "loaded_modules": loaded, "card012_identity": identity, "resumed_from": (str(resume_from) if resume_from else None),
           "resume_global_step": resume_gs, "train_wall_s": round(wall, 1),
           "it_per_s": round(exec_steps / wall, 2) if wall > 0 else None, "proc_peak_vram_gb": proc_peak,
           "global_vram_used_gb": global_used, "stop_reason": ts.get("stop_reason"), "train_stats": ts}
    (OUTD / f"manifest-{arm}.json").write_text(json.dumps(man, indent=1))
    V.strict_validate_arm(arm, ROOT)
    print(f"[card022 {arm}] steps={exec_steps} {wall:.0f}s 2D proc={proc_peak}GB global={global_used}GB "
          f"pipe={pinfo.get('x_residency')} w={shape_weight} resumed={bool(resume_from)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
