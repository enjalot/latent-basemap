"""Card034 arm trainer (per card034 + production review). Thin driver over the ONE production engine
(card034_engine.GroupedEngine) reused by the canary and preflight. Three fresh 300K/2D/60K arms from the
fresh 2D init 589895f0, shared grouped 9:1 sampler, constant LR .001, AdamW wd .01, MODEL grad clip 1, FP16
device bank + AMP, EXACTLY 60000 successful positive-LR updates. Checkpoints are written at a COHERENT
POST-STEP boundary (model + sampler both advanced past the same block — no skipped/duplicated batch), with
live stats + full optimizer/scaler/scalar/sampler/RNG payload. Snapshots 20/40/60K. Auto-resume deep-validates
BEFORE restore and rejects wrong objective/coeff/seed/data/scalar. The actual warm tensors are hashed (not
trusted metadata). Usage: run_card034_arm.py <grouped_umap|grouped_nce|grouped_infonce> <STEPS>
"""
import os, sys, time, json, re, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import card034_validate as V
import card034_grouped as G
import card034_engine as E

ROOT = Path(__file__).resolve().parents[2]
SB = V.SB; OC = V.OC; CHAMPION = V.CHAMPION; SUB = V.SUB; GRAPH = V.GRAPH; INIT = V.INIT
OUTD = SB / "card034-train"
DOSE = V.DOSE; SNAPS = set(V.SNAP_STEPS); STEP_CKPTS = set(V.STEP_CKPTS); ARMS = V.ARMS


def _latest_ckpt(cdir):
    import torch
    if not cdir.is_dir(): return None, 0
    cands = []
    for f in list(cdir.glob("ckpt-step*.pt")) + list(cdir.glob("ckpt-epoch*.pt")):
        try: gs = int(torch.load(f, map_location="cpu", weights_only=False)["global_step"])
        except Exception as e: raise RuntimeError(f"corrupt checkpoint {f}: {e!r} — fail closed")
        cands.append((gs, f))
    if not cands: return None, 0
    gs, f = max(cands, key=lambda t: t[0]); return f, gs


def main():
    arm = sys.argv[1]; steps = int(sys.argv[2]); assert arm in ARMS and steps == DOSE, "arm/dose"
    import torch
    import basemap.pumap.parametric_umap.core as _core
    OUTD.mkdir(parents=True, exist_ok=True)
    if (OUTD / f"manifest-{arm}.json").exists():
        V.strict_validate_arm(arm, ROOT); print(f"[card034 {arm}] already complete + strict-valid; skip", flush=True); return 0

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "card034 arms are the production device path; refuse CPU fallback"
    assert V.full_sha(SUB) == V.SUB_SHA256 and V.full_sha(GRAPH) == V.GRAPH_SHA256, "data identity"
    init_obj = torch.load(str(INIT), map_location="cpu", weights_only=False); warm = init_obj["model_state"]
    init_payload = V.init_payload_sha(warm); assert init_payload == V.INIT_SHA, f"init payload hash {init_payload} != {V.INIT_SHA}"   # hash actual tensors
    if arm == "grouped_infonce":
        assert V.full_sha(V.CALIB) == V.expected_identity(arm, ROOT)["infonce_calib_sha256"], "calibration content hash drift"
    fok, bad = V.runtime_manifest_check(ROOT); assert fok, f"frozen-runtime mismatch {bad}"
    basemap_mods = V.loaded_basemap_under_root(ROOT)
    import subprocess
    githead = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    loaded = {"core": {"path": _core.__file__, "sha": V.full_sha(_core.__file__)[:16]},
              "run_card034_arm": {"path": __file__, "sha": V.full_sha(__file__)[:16]},
              "card034_grouped": {"path": G.__file__, "sha": V.full_sha(G.__file__)[:16]},
              "card034_engine": {"path": E.__file__, "sha": V.full_sha(E.__file__)[:16]},
              "git_head": githead, "worktree_root": str(ROOT), "verified_frozen_runtime": True,
              "all_basemap_under_root": True, "basemap_modules": basemap_mods,
              "runtime_manifest_sha256": V.full_sha(V.runtime_manifest_path(ROOT))}
    identity = V.expected_identity(arm, ROOT)
    coeff = V.calibrated_coeff() if arm == "grouped_infonce" else 1.0

    torch.manual_seed(V.SEED); np.random.seed(V.SEED); torch.cuda.manual_seed_all(V.SEED); torch.cuda.reset_peak_memory_stats()
    engine = E.GroupedEngine(arm, identity, coeff, dev, CHAMPION, warm)
    ez = np.load(GRAPH); src = ez["sources"]; tgt = ez["targets"]; assert src.shape[0] == V.N * 15, "15 outgoing/row"
    X = np.asarray(np.load(SUB, mmap_mode="r"), np.float32); assert X.shape == (V.N, 1536)
    Xt = torch.tensor(X, dtype=torch.float16, device=dev)
    sampler = G.GroupedSampler(V.N, src, tgt, seed=V.SEED)
    SNAPDIR = OUTD / arm; SNAPDIR.mkdir(exist_ok=True); CKPTDIR = SNAPDIR / "ckpts"; CKPTDIR.mkdir(exist_ok=True)

    def _write_ckpt(name, step_checkpoint):
        st = engine.ckpt_dict(sampler, step_checkpoint)
        tmp = CKPTDIR / f"ckpt-{name}.pt.tmp"; torch.save(st, tmp); os.replace(tmp, CKPTDIR / f"ckpt-{name}.pt")
    def _prune_epochs():
        eps = sorted(CKPTDIR.glob("ckpt-epoch*.pt"), key=lambda q: int(re.search(r"epoch(\d+)", q.name).group(1)))
        for old in eps[:-2]:
            try: old.unlink()
            except OSError: pass

    resume_from, resume_gs = _latest_ckpt(CKPTDIR)
    adm_path = OUTD / f"admission-{arm}.json"
    if resume_from is None:
        adm = {"schema": "card034-admission-2026-09-12", "arm": arm, "mode": V.MODE[arm], "written_before_steps": True,
               "warm_init_sha256": V.INIT_SHA, "init_payload_sha256": init_payload, "identity": identity,
               "loaded_modules": loaded, "n_rows": V.N, "snapshots": sorted(SNAPS), "step_checkpoints": sorted(STEP_CKPTS), "coeff": coeff}
        assert not adm_path.exists(); adm_path.write_text(json.dumps(adm, indent=1))
    else:
        ck = torch.load(resume_from, map_location=dev, weights_only=False)
        engine.restore(ck, sampler, ROOT, V.N)                 # deep-validate BEFORE restore (fail closed on drift)
        orig = json.loads(adm_path.read_text()); assert orig.get("identity") == identity, "immutable admission drift"
        (OUTD / f"resume-{arm}-{int(time.time())}.json").write_text(json.dumps(
            {"at": dt.datetime.now(dt.timezone.utc).isoformat(), "resume_from": str(resume_from), "resume_global_step": resume_gs, "loaded_modules": loaded}, indent=1))

    t0 = time.time(); prev_epoch = sampler.epoch; pending_epoch = False
    while engine.success < DOSE:
        heads, tails = sampler.next_block()
        if sampler.epoch > prev_epoch: pending_epoch = True; prev_epoch = sampler.epoch
        if engine.step(Xt, heads, tails):                      # coherent: model + sampler both advanced past this block
            s = engine.success
            if pending_epoch: _write_ckpt(f"epoch{sampler.epoch}", True); _prune_epochs(); pending_epoch = False
            if s in SNAPS: engine.p.is_fitted = True; engine.p.save(str(SNAPDIR / f"model-step{s}.pt"))
            if s in STEP_CKPTS: _write_ckpt(f"step{s}", True)
    wall = time.time() - t0

    st = engine.stats; assert st["positive_lr_optimizer_steps"] == DOSE, "dose"
    receipt = E.pipeline_receipt(Xt)     # explicit device-bank observation (CUDA/fp16/(N,1536)); not a borrowed core fit receipt
    proc_peak = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    free, total = torch.cuda.mem_get_info(); global_used = round((total - free) / 2**30, 3)
    assert global_used < 30.0, f"global VRAM {global_used} GiB exceeds 30 GiB cap"
    coords = np.asarray(engine.p.transform(X, batch_size=8192), np.float32); assert coords.shape == (V.N, V.NC)
    np.save(OUTD / f"coords-{arm}.npy", coords); engine.p.is_fitted = True; engine.p.save(str(OUTD / f"model-{arm}.pt"))
    final_beta = engine.final_beta()
    if arm == "grouped_nce": assert final_beta is not None and abs(final_beta) > 0, "grouped_nce scalar never moved"
    man = {"schema": "card034-arm-2026-09-12", "arm": arm, "mode": V.MODE[arm], "n": V.N, "n_components": V.NC,
           "executed_steps": engine.success, "warm_init_sha256": V.INIT_SHA, "init_payload_sha256": init_payload,
           "trained_sha256": V.state_sha(engine.model.state_dict()), "lr_used_min": V.LR, "lr_used_max": V.LR,
           "weight_decay": V.WEIGHT_DECAY, "grad_clip": V.GRAD_CLIP, "final_beta": final_beta,
           "infonce_coeff": (coeff if arm == "grouped_infonce" else None), "pipeline_receipt": receipt,
           "snapshots": sorted(SNAPS), "step_checkpoints": sorted(STEP_CKPTS), "loaded_modules": loaded, "identity": identity,
           "resumed_from": (str(resume_from) if resume_from else None), "train_wall_s": round(wall, 1),
           "it_per_s": round(engine.success / wall, 2) if wall > 0 else None, "proc_peak_vram_gb": proc_peak,
           "global_vram_used_gb": global_used, "train_stats": st}
    (OUTD / f"manifest-{arm}.json").write_text(json.dumps(man, indent=1))
    V.strict_validate_arm(arm, ROOT)
    print(f"[card034 {arm}] steps={engine.success} amp_skips={st['amp_skips']} nonfinite={st['nonfinite_skips']} "
          f"{wall:.0f}s beta={final_beta} coeff={coeff} proc={proc_peak}GB global={global_used}GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
