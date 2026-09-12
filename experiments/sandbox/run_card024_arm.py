"""Card024 arm trainer (per card024-contrastive-normalization.md). Three fresh 300K / 2D / 60K arms from the
fresh 2D init (589895f037d406ae) on the fixed 300K DINO draw + binary directed fixed15 graph. SAME uniform
nonself noise sampler + graph-positive stream for all arms; NO rank-window/fneg/anchors/replay/shape/radius/
derivative/mid-near/density/correlation. n_components=2, FP16 device inputs, batch 16384, pos_ratio .1, seed
42, constant LR .001, exactly 60000 successful positive-LR updates, fresh AdamW (parent weight decay).
  - umap_uniform: BCE(qhat) (default-off core path);
  - neg_fixed:    BCE-logit log(qhat)-beta, beta fixed 0;
  - nce_learned:  BCE-logit, beta learned (own AdamW group, LR .001, wd 0).
Snapshots 20/40/60K; resumable step ckpts (20/40/60K) + epoch, binding the family/scalar identity and (nce)
the learned scalar. Auto-resumes. Usage: run_card024_arm.py <umap_uniform|neg_fixed|nce_learned> <STEPS>
"""
import os, sys, time, json, hashlib, subprocess, re, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import card024_validate as V

ROOT = Path(__file__).resolve().parents[2]
SB = V.SB; OC = V.OC; CHAMPION = V.CHAMPION; SUB = V.SUB; GRAPH = V.GRAPH; INIT = V.INIT
OUTD = SB / "card024-train"
SEED = V.SEED; LR = V.LR; BATCH = V.BATCH; N_EXPECT = V.N; NC = V.NC
SNAPS = tuple(V.SNAP_STEPS); STEP_CKPTS = set(V.STEP_CKPTS); ARMS = V.ARMS; MODE = V.MODE


def _latest_ckpt(cdir):
    import torch
    if not cdir.is_dir(): return None, 0
    cands = []
    for f in cdir.glob("ckpt-step*.pt"):
        m = re.search(r"ckpt-step(\d+)\.pt$", f.name)
        if m:
            try: ck = torch.load(f, map_location="cpu", weights_only=False)
            except Exception as e: raise RuntimeError(f"corrupt step checkpoint {f}: {e!r}")
            assert int(ck["global_step"]) == int(m.group(1)); cands.append((int(m.group(1)), f))
    for f in cdir.glob("ckpt-epoch*.pt"):
        try: gs = int(torch.load(f, map_location="cpu", weights_only=False)["global_step"])
        except Exception as e: raise RuntimeError(f"corrupt epoch checkpoint {f}: {e!r} — fail closed")
        cands.append((gs, f))
    if not cands: return None, 0
    gs, f = max(cands, key=lambda t: t[0])
    ck = torch.load(f, map_location="cpu", weights_only=False); assert int(ck["global_step"]) == gs
    return f, gs


def main():
    arm = sys.argv[1]; steps = int(sys.argv[2]); assert arm in ARMS and steps == V.DOSE, "arm/dose"
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    import basemap.pumap.parametric_umap.core as _core
    import basemap.pumap.parametric_umap.datasets.edge_list_dataset as _eld
    import torch
    OUTD.mkdir(parents=True, exist_ok=True)
    if (OUTD / f"manifest-{arm}.json").exists():
        V.strict_validate_arm(arm, ROOT); print(f"[card024 {arm}] already complete + strict-valid; skip", flush=True); return 0

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "card024 arms are the production device path; refuse CPU fallback"
    free, total = torch.cuda.mem_get_info(); assert (total - free) / 2**30 < 12, "leave GPU headroom before load"
    assert V.full_sha(SUB) == V.SUB_SHA256, "300K substrate hash"
    assert V.full_sha(GRAPH) == V.GRAPH_SHA256, "fixed15 graph hash"
    ez = np.load(GRAPH); n_edges = ez["sources"].shape[0]
    assert n_edges == N_EXPECT * 15, f"expected 15 outgoing/row: {n_edges} != {N_EXPECT*15}"
    init_obj = torch.load(str(INIT), map_location="cpu", weights_only=False)
    warm_state = init_obj["model_state"]; assert init_obj["init_state_sha256"] == V.INIT_SHA, "fresh 2D init identity"

    fok, bad = V.runtime_manifest_check(ROOT); assert fok, f"frozen-runtime hash mismatch {bad} — fail closed"
    basemap_mods = V.loaded_basemap_under_root(ROOT)
    githead = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    loaded = {"core": {"path": _core.__file__, "sha": V.full_sha(_core.__file__)[:16]},
              "edge_list_dataset": {"path": _eld.__file__, "sha": V.full_sha(_eld.__file__)[:16]},
              "run_card024_arm": {"path": __file__, "sha": V.full_sha(__file__)[:16]},
              "git_head": githead, "worktree_root": str(ROOT), "verified_frozen_runtime": True,
              "all_basemap_under_root": True, "basemap_modules": basemap_mods,
              "runtime_manifest_sha256": V.full_sha(V.runtime_manifest_path(ROOT))}
    identity = V.expected_identity(arm, ROOT)

    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed_all(SEED); torch.cuda.reset_peak_memory_stats()
    pumap = ParametricUMAP.load(str(CHAMPION), device=dev); pumap.model = None; pumap.n_components = NC
    pumap.learning_rate = LR; pumap.lr_schedule = "constant"; pumap.batch_size = BATCH; pumap.warmup_steps = 0
    pumap.n_epochs = 100000; pumap._max_train_steps = steps
    pumap.rankneg_window = 0                       # uniform nonself noise (no rank window)
    pumap.x_residency = "auto"; pumap.required_input_pipeline = "device"
    assert abs(float(getattr(pumap, "pos_ratio", -1)) - V.POS_RATIO) < 1e-9, "pos_ratio must be .1"
    for a, v in (("anchor_ids_path", ""), ("anchor_hold_weight", 0.0), ("replay_bank_path", ""),
                 ("replay_weight", 0.0), ("deriv_bank_path", ""), ("deriv_weight", 0.0),
                 ("fneg_weight", 0.0), ("neg_tanh_gamma", 0.0), ("midnear_enabled", False),
                 ("density_weight", 0.0), ("correlation_weight", 0.0)):
        if hasattr(pumap, a): setattr(pumap, a, v)
    pumap._card024_mode = MODE[arm]                # the family switch (default-off otherwise)
    pumap._checkpoint_step_targets = set(STEP_CKPTS); pumap._card012_identity = identity

    SNAPDIR = OUTD / arm; SNAPDIR.mkdir(exist_ok=True); CKPTDIR = SNAPDIR / "ckpts"; CKPTDIR.mkdir(exist_ok=True)
    resume_from, resume_gs = _latest_ckpt(CKPTDIR)
    adm_path = OUTD / f"admission-{arm}.json"
    if resume_from is None:
        adm = {"schema": "card024-admission-2026-09-12", "arm": arm, "mode": MODE[arm], "written_before_steps": True,
               "n_components": NC, "warm_init_sha256": V.INIT_SHA, "substrate_sha": V.SUB_SHA256[:16], "graph_sha": V.GRAPH_SHA256[:16],
               "lr": LR, "lr_schedule": "constant", "batch_size": BATCH, "pos_ratio": V.POS_RATIO, "seed": SEED,
               "steps": steps, "rankneg_window": 0, "noise": "uniform nonself", "kernel_a": V.A, "kernel_b": V.B,
               "snapshots": list(SNAPS), "step_checkpoints": sorted(STEP_CKPTS), "checkpoint_every_epochs": 1,
               "precision": "device_fp16", "card012_identity": identity, "loaded_modules": loaded, "n_rows": N_EXPECT}
        assert not adm_path.exists(); adm_path.write_text(json.dumps(adm, indent=1))
    else:
        orig = json.loads(adm_path.read_text())
        assert orig.get("card012_identity") == identity, "resume rebuilt identity != immutable original admission"
        (OUTD / f"resume-{arm}-{int(time.time())}.json").write_text(json.dumps(
            {"at": dt.datetime.now(dt.timezone.utc).isoformat(), "resume_from": str(resume_from),
             "resume_global_step": resume_gs, "identity_matches_original": True, "loaded_modules": loaded}, indent=1))

    X = np.asarray(np.load(SUB, mmap_mode="r"), np.float32); n = X.shape[0]; assert n == N_EXPECT
    t0 = time.time()
    pumap.fit(X, precomputed_edges_path=str(GRAPH), random_state=SEED, verbose=False,
              warm_start_state=(None if resume_from else warm_state), snapshot_steps=SNAPS, snapshot_dir=str(SNAPDIR),
              checkpoint_every_epochs=1, checkpoint_dir=str(CKPTDIR), resume_from=(str(resume_from) if resume_from else None))
    wall = time.time() - t0

    ts = dict(getattr(pumap, "_train_stats", {}) or {}); pinfo = dict(getattr(pumap, "_pipeline_info", {}) or {})
    exec_steps = int(ts.get("executed_iters", 0)); assert exec_steps == steps, f"{exec_steps} != {steps}"
    assert int(ts.get("positive_lr_optimizer_steps", -1)) == steps, "positive-LR count != dose"
    assert abs(ts.get("lr_used_min", 0) - LR) < 1e-12 and abs(ts.get("lr_used_max", 0) - LR) < 1e-12, "LR not 1e-3"
    assert pinfo.get("x_residency") == "device_fp16", f"pipeline not device_fp16: {pinfo.get('x_residency')}"
    _beta = getattr(pumap, "_card024_beta", None)
    final_beta = (float(_beta.detach()) if isinstance(_beta, torch.Tensor) else None)
    if arm == "nce_learned":
        assert final_beta is not None and abs(final_beta) > 0, "nce_learned scalar never moved (inactive)"
    proc_peak = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    free, total = torch.cuda.mem_get_info(); global_used = round((total - free) / 2**30, 3)
    assert global_used < 30.0, f"global VRAM {global_used} GiB exceeds 30 GiB cap"

    coords = np.asarray(pumap.transform(X, batch_size=8192), np.float32); assert coords.shape == (n, NC)
    np.save(OUTD / f"coords-{arm}.npy", coords); pumap.save(str(OUTD / f"model-{arm}.pt"))
    man = {"schema": "card024-arm-2026-09-12", "arm": arm, "mode": MODE[arm], "n": int(n), "n_components": NC,
           "executed_steps": exec_steps, "warm_init_sha256": V.INIT_SHA, "warm_start_param_sha256": getattr(pumap, "warm_start_sha256", None),
           "trained_sha256": V.state_sha(pumap.model.state_dict()), "lr_used_min": ts.get("lr_used_min"), "lr_used_max": ts.get("lr_used_max"),
           "final_beta": final_beta, "kernel_a": V.A, "kernel_b": V.B, "snapshots": list(SNAPS), "step_checkpoints": sorted(STEP_CKPTS),
           "pipeline_info": pinfo, "loaded_modules": loaded, "card012_identity": identity,
           "resumed_from": (str(resume_from) if resume_from else None), "resume_global_step": resume_gs,
           "train_wall_s": round(wall, 1), "it_per_s": round(exec_steps / wall, 2) if wall > 0 else None,
           "proc_peak_vram_gb": proc_peak, "global_vram_used_gb": global_used, "stop_reason": ts.get("stop_reason"), "train_stats": ts}
    (OUTD / f"manifest-{arm}.json").write_text(json.dumps(man, indent=1))
    V.strict_validate_arm(arm, ROOT)
    print(f"[card024 {arm}] steps={exec_steps} {wall:.0f}s 2D beta={final_beta} proc={proc_peak}GB global={global_used}GB "
          f"pipe={pinfo.get('x_residency')} resumed={bool(resume_from)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
