"""Card038 arm trainer (per card038-direct-compact-graph.md). Uses the PROVEN core fit (card033 @ 8b7ecab) +
original sampler + Hook-C half-strength local radii — NOT Card034's experimental engine. Two arms at
different widths/doses from fresh per-width 3D inits (seed 42): wide2048 (H2048, 60K) / compact1024 (H1024,
180K). Shared recipe: LR .001 constant, AdamW wd .01, grad clip 1, seed 42, device FP16 bank/AMP, batch 16384,
pos .1, rankneg 75000, fneg 1, tanh 4, half local-scale kernel; all other interventions OFF. Wide snapshots
20/40/60K, compact 60/120/180K; genuine step + epoch checkpoints (card012 schema). Immutable admission;
auto-resume rejects wrong width/dose/radius before restore (identity guard). Width is bound by the actual
parameter tensors/count. Usage: run_card038_arm.py <wide2048|compact1024> <STEPS>
"""
import os, sys, time, json, subprocess, re, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import card038_validate as V
from card038_fit import configure_pumap

ROOT = Path(__file__).resolve().parents[2]
SB = V.SB; OC = V.OC; CHAMPION = V.CHAMPION; SUB = V.SUB; GRAPH = V.GRAPH; RADII = V.RADII; INITD = V.INITD
OUTD = V.TD_DEFAULT


def _latest_ckpt(cdir):
    import torch
    if not cdir.is_dir(): return None, 0
    cands = []
    for f in list(cdir.glob("ckpt-step*.pt")) + list(cdir.glob("ckpt-epoch*.pt")):
        try:
            obj=torch.load(f, map_location="cpu", weights_only=False);gs=int(obj["global_step"])
            if f.name.startswith("ckpt-step"):assert gs==int(f.stem.removeprefix("ckpt-step")) and obj.get("step_checkpoint") is True, "checkpoint filename/step identity"
        except Exception as e: raise RuntimeError(f"corrupt checkpoint {f}: {e!r} — fail closed")
        cands.append((gs, f))
    if not cands: return None, 0
    gs, f = max(cands, key=lambda t: t[0]); return f, gs


def main():
    arm = sys.argv[1]; steps = int(sys.argv[2]); assert arm in V.ARMS and steps == V.DOSE[arm], "arm/dose"
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    import basemap.pumap.parametric_umap.core as _core
    import basemap.pumap.parametric_umap.datasets.edge_list_dataset as _eld
    import torch
    OUTD.mkdir(parents=True, exist_ok=True)
    if (OUTD / f"manifest-{arm}.json").exists():
        V.strict_validate_arm(arm, ROOT); print(f"[card038 {arm}] already complete + strict-valid; skip", flush=True); return 0

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "card038 arms are the production device path; refuse CPU fallback"
    assert V.full_sha(SUB) == V.SUB_SHA256 and V.full_sha(GRAPH) == V.GRAPH_SHA256, "data identity"
    assert V.full_sha(RADII) == V.R_HALF_SHA256, "half-radii identity"
    init_path = INITD / f"init-{arm}.pt"; assert V.full_sha(init_path) == V.INIT_FILE_SHA[arm], "init file hash"
    init_obj = torch.load(str(init_path), map_location="cpu", weights_only=False); warm = init_obj["model_state"]
    init_payload = V.state_sha(warm); assert init_payload == V.INIT_STATE_SHA[arm], "init payload hash"
    assert init_obj["hidden_dim"] == V.WIDTH[arm] and init_obj["n_components"] == V.NC, "init width/dim"
    radii = np.load(RADII).astype(np.float32); assert radii.shape[0] == V.N and np.isfinite(radii).all() and (radii > 0).all()

    fok, bad = V.runtime_manifest_check(ROOT); assert fok, f"frozen-runtime mismatch {bad}"
    basemap_mods = V.loaded_basemap_under_root(ROOT)
    githead = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    loaded = {"core": {"path": _core.__file__, "sha": V.full_sha(_core.__file__)[:16]},
              "edge_list_dataset": {"path": _eld.__file__, "sha": V.full_sha(_eld.__file__)[:16]},
              "run_card038_arm": {"path": __file__, "sha": V.full_sha(__file__)[:16]},
              "git_head": githead, "worktree_root": str(ROOT), "verified_frozen_runtime": True,
              "all_basemap_under_root": True, "basemap_modules": basemap_mods, "runtime_manifest_sha256": V.full_sha(V.runtime_manifest_path(ROOT))}
    identity = V.expected_identity(arm, ROOT)
    SNAPS = tuple(V.SNAP[arm]); STEP_CKPTS = set(V.SNAP[arm])

    torch.manual_seed(V.SEED); np.random.seed(V.SEED); torch.cuda.manual_seed_all(V.SEED); torch.cuda.reset_peak_memory_stats()
    pumap = ParametricUMAP.load(str(CHAMPION), device=dev); pumap.model = None
    configure_pumap(pumap,arm,steps,radii,identity,STEP_CKPTS)

    SNAPDIR = OUTD / arm; SNAPDIR.mkdir(exist_ok=True); CKPTDIR = SNAPDIR / "ckpts"; CKPTDIR.mkdir(exist_ok=True)
    resume_from, resume_gs = _latest_ckpt(CKPTDIR)
    adm_path = OUTD / f"admission-{arm}.json"
    if resume_from is None:
        adm = {"schema": "card038-admission-2026-09-12", "arm": arm, "width_hidden_dim": V.WIDTH[arm], "n_params": V.NPARAM[arm],
               "written_before_steps": True, "n_components": V.NC, "init_state_sha256": V.INIT_STATE_SHA[arm], "init_payload_sha256": init_payload,
               "radii_sha256": V.R_HALF_SHA256, "dose": steps, "snapshots": list(SNAPS), "card012_identity": identity, "loaded_modules": loaded}
        assert not adm_path.exists(); adm_path.write_text(json.dumps(adm, indent=1))
    else:
        V.validate_ckpt_payload(torch.load(resume_from,map_location="cpu",weights_only=False),arm,ROOT,identity,expect_step=resume_gs)
        orig = json.loads(adm_path.read_text()); assert orig.get("card012_identity") == identity, "immutable admission drift"
        (OUTD / f"resume-{arm}-{int(time.time())}.json").write_text(json.dumps(
            {"at": dt.datetime.now(dt.timezone.utc).isoformat(), "resume_from": str(resume_from), "resume_global_step": resume_gs, "loaded_modules": loaded}, indent=1))

    X = np.asarray(np.load(SUB, mmap_mode="r"), np.float32); n = X.shape[0]; assert n == V.N
    t0 = time.time()
    pumap.fit(X, precomputed_edges_path=str(GRAPH), random_state=V.SEED, verbose=False,
              warm_start_state=(None if resume_from else warm), snapshot_steps=SNAPS, snapshot_dir=str(SNAPDIR),
              checkpoint_every_epochs=1, checkpoint_dir=str(CKPTDIR), resume_from=(str(resume_from) if resume_from else None))
    wall = time.time() - t0

    ts = dict(getattr(pumap, "_train_stats", {}) or {}); pinfo = dict(getattr(pumap, "_pipeline_info", {}) or {})
    exec_steps = int(ts.get("executed_iters", 0)); assert exec_steps == steps, f"{exec_steps} != {steps}"
    assert int(ts.get("positive_lr_optimizer_steps", -1)) == steps, "positive-LR count != dose"
    assert abs(ts.get("lr_used_min", 0) - V.LR) < 1e-12 and abs(ts.get("lr_used_max", 0) - V.LR) < 1e-12, "LR not 1e-3"
    assert pinfo.get("x_residency") == "device_fp16", f"pipeline not device_fp16: {pinfo.get('x_residency')}"
    if resume_from is None: assert pumap.warm_start_sha256==V.INIT_PARAM_SHA[arm][:16], "actual warm hash mismatch"
    nparam = int(sum(t.numel() for t in pumap.model.parameters())); assert nparam == V.NPARAM[arm], "param count != width"
    proc_peak = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    free, total = torch.cuda.mem_get_info(); global_used = round((total - free) / 2**30, 3)
    assert global_used < 30.0, f"global VRAM {global_used} GiB exceeds 30 GiB cap"
    coords = np.asarray(pumap.transform(X, batch_size=8192), np.float32); assert coords.shape == (n, V.NC)
    np.save(OUTD / f"coords-{arm}.npy", coords); pumap.save(str(OUTD / f"model-{arm}.pt"))
    loaded["basemap_modules"]=V.loaded_basemap_under_root(ROOT)
    man = {"schema": "card038-arm-2026-09-12", "arm": arm, "width_hidden_dim": V.WIDTH[arm], "n_params": nparam, "n": n,
           "n_components": V.NC, "executed_steps": exec_steps, "init_state_sha256": V.INIT_STATE_SHA[arm], "init_payload_sha256": init_payload,
           "warm_param_sha256": V.INIT_PARAM_SHA[arm][:16], "trained_sha256": V.state_sha(pumap.model.state_dict()), "lr_used_min": V.LR, "lr_used_max": V.LR,
           "radii_sha256": V.R_HALF_SHA256, "pipeline_info": pinfo, "snapshots": list(SNAPS), "loaded_modules": loaded,
           "card012_identity": identity, "resumed_from": (str(resume_from) if resume_from else None),
           "train_wall_s": round(wall, 1), "it_per_s": round(exec_steps / wall, 2) if wall > 0 else None,
           "proc_peak_vram_gb": proc_peak, "global_vram_used_gb": global_used, "train_stats": ts}
    (OUTD / f"manifest-{arm}.json").write_text(json.dumps(man, indent=1))
    V.strict_validate_arm(arm, ROOT)
    print(f"[card038 {arm}] H{V.WIDTH[arm]} params={nparam} steps={exec_steps} {wall:.0f}s proc={proc_peak}GB global={global_used}GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
