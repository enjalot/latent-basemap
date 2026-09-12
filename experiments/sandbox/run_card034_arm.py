"""Card034 standalone arm trainer (per card034-grouped-relative-objective.md). Self-contained loop reusing
the existing model architecture + ParametricUMAP checkpoint export (NO core edits). Three fresh 300K/2D/60K
arms from the fresh 2D init 589895f0 on the fixed 300K DINO draw (unit-norm rows) + fixed15 graph, driven by
the shared grouped sampler (card034_grouped): shared PERM positive stream, exactly 9 uniform nonself noise per
positive, head forwarded once + shared across its 10 candidates. Constant LR .001, AdamW model weight_decay
.01, model grad-norm clip 1, FP16 device bank + AMP, EXACTLY 60000 successful positive-LR updates (AMP skips
charged separately). grouped_nce adds a learned scalar beta (own group, LR .001, wd 0); grouped_infonce
multiplies its loss by the frozen init-scale coefficient. Snapshots 20/40/60K; genuine step + epoch
checkpoints (model/opt/scaler/beta/sampler/RNG) binding the full identity; auto-resume rejects a wrong
objective/coeff/seed/data/scalar before restore. Usage: run_card034_arm.py <grouped_umap|grouped_nce|grouped_infonce> <STEPS>
"""
import os, sys, time, json, subprocess, re, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import card034_validate as V
import card034_grouped as G

ROOT = Path(__file__).resolve().parents[2]
SB = V.SB; OC = V.OC; CHAMPION = V.CHAMPION; SUB = V.SUB; GRAPH = V.GRAPH; INIT = V.INIT
OUTD = SB / "card034-train"
SEED = V.SEED; LR = V.LR; WD = V.WEIGHT_DECAY; CLIP = V.GRAD_CLIP; N_EXPECT = V.N; NC = V.NC
DOSE = V.DOSE; SNAPS = set(V.SNAP_STEPS); STEP_CKPTS = set(V.STEP_CKPTS); ARMS = V.ARMS
PROBE_STEPS = {1, DOSE // 2, DOSE}


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
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    import basemap.pumap.parametric_umap.core as _core
    import torch, torch.nn as nn
    from torch.optim import AdamW
    OUTD.mkdir(parents=True, exist_ok=True)
    if (OUTD / f"manifest-{arm}.json").exists():
        V.strict_validate_arm(arm, ROOT); print(f"[card034 {arm}] already complete + strict-valid; skip", flush=True); return 0

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "card034 arms are the production device path; refuse CPU fallback"
    assert V.full_sha(SUB) == V.SUB_SHA256 and V.full_sha(GRAPH) == V.GRAPH_SHA256, "data identity"
    init_obj = torch.load(str(INIT), map_location="cpu", weights_only=False)
    assert init_obj["init_state_sha256"] == V.INIT_SHA, "fresh 2D init identity"; warm = init_obj["model_state"]
    fok, bad = V.runtime_manifest_check(ROOT); assert fok, f"frozen-runtime mismatch {bad}"
    basemap_mods = V.loaded_basemap_under_root(ROOT)
    githead = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    loaded = {"core": {"path": _core.__file__, "sha": V.full_sha(_core.__file__)[:16]},
              "run_card034_arm": {"path": __file__, "sha": V.full_sha(__file__)[:16]},
              "card034_grouped": {"path": G.__file__, "sha": V.full_sha(G.__file__)[:16]},
              "git_head": githead, "worktree_root": str(ROOT), "verified_frozen_runtime": True,
              "all_basemap_under_root": True, "basemap_modules": basemap_mods,
              "runtime_manifest_sha256": V.full_sha(V.runtime_manifest_path(ROOT))}
    identity = V.expected_identity(arm, ROOT)
    coeff = V.calibrated_coeff() if arm == "grouped_infonce" else 1.0

    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed_all(SEED); torch.cuda.reset_peak_memory_stats()
    p = ParametricUMAP.load(str(CHAMPION), device=dev); p.model = None; p.n_components = NC
    p.learning_rate = LR; p.lr_schedule = "constant"
    p._init_model(1536); p.model.load_state_dict(warm); model = p.model
    beta = None; groups = [{"params": list(model.parameters()), "lr": LR, "weight_decay": WD}]
    if arm == "grouped_nce":
        beta = nn.Parameter(torch.zeros((), device=dev)); groups.append({"params": [beta], "lr": LR, "weight_decay": 0.0})
    opt = AdamW(groups); scaler = torch.amp.GradScaler(dev, enabled=True)

    ez = np.load(GRAPH); src = ez["sources"]; tgt = ez["targets"]; assert src.shape[0] == N_EXPECT * 15, "15 outgoing/row"
    X = np.asarray(np.load(SUB, mmap_mode="r"), np.float32); assert X.shape == (N_EXPECT, 1536)
    Xt = torch.tensor(X, dtype=torch.float16, device=dev)
    sampler = G.GroupedSampler(N_EXPECT, src, tgt, seed=SEED)
    SNAPDIR = OUTD / arm; SNAPDIR.mkdir(exist_ok=True); CKPTDIR = SNAPDIR / "ckpts"; CKPTDIR.mkdir(exist_ok=True)

    stats = {"positive_lr_optimizer_steps": 0, "amp_skips": 0, "exposure_probes": [], "lr_used_min": LR, "lr_used_max": LR}
    success = 0; last_epoch = 0; consec_bad = 0

    def _ck_state(name):
        st = {"schema": "card034-ckpt-2026-09-12", "arm": arm, "mode": V.MODE[arm], "global_step": int(success),
              "epoch": int(sampler.epoch), "step_checkpoint": bool(name.startswith("step")), "identity": identity,
              "coeff": coeff, "model": model.state_dict(), "optimizer": opt.state_dict(),
              "scaler": scaler.state_dict(), "beta": (beta.detach().cpu() if beta is not None else None),
              "sampler_state": sampler.state(), "torch_rng": torch.get_rng_state(),
              "cuda_rng": torch.cuda.get_rng_state_all(), "train_stats": dict(stats)}
        tmp = CKPTDIR / f"ckpt-{name}.pt.tmp"; torch.save(st, tmp); os.replace(tmp, CKPTDIR / f"ckpt-{name}.pt")
    def _prune_epochs():
        eps = sorted(CKPTDIR.glob("ckpt-epoch*.pt"), key=lambda q: int(re.search(r"epoch(\d+)", q.name).group(1)))
        for old in eps[:-2]:
            try: old.unlink()
            except OSError: pass
    def _export(path):
        p.is_fitted = True; p.save(str(path))

    resume_from, resume_gs = _latest_ckpt(CKPTDIR)
    adm_path = OUTD / f"admission-{arm}.json"
    if resume_from is None:
        adm = {"schema": "card034-admission-2026-09-12", "arm": arm, "mode": V.MODE[arm], "written_before_steps": True,
               "warm_init_sha256": V.INIT_SHA, "identity": identity, "loaded_modules": loaded, "n_rows": N_EXPECT,
               "snapshots": sorted(SNAPS), "step_checkpoints": sorted(STEP_CKPTS), "coeff": coeff}
        assert not adm_path.exists(); adm_path.write_text(json.dumps(adm, indent=1))
    else:
        ck = torch.load(resume_from, map_location=dev, weights_only=False)
        assert ck.get("identity") == identity, "resume identity mismatch — wrong objective/coeff/seed/data/scalar; fail closed BEFORE restore"
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); scaler.load_state_dict(ck["scaler"])
        if beta is not None and ck.get("beta") is not None:
            with torch.no_grad(): beta.copy_(ck["beta"].to(dev))
        sampler.load_state(ck["sampler_state"]); torch.set_rng_state(ck["torch_rng"].to("cpu", torch.uint8))
        torch.cuda.set_rng_state_all([s.to("cpu", torch.uint8) for s in ck["cuda_rng"]])
        stats = dict(ck["train_stats"]); success = int(ck["global_step"]); last_epoch = int(sampler.epoch)
        orig = json.loads(adm_path.read_text()); assert orig.get("identity") == identity, "immutable admission drift"
        (OUTD / f"resume-{arm}-{int(time.time())}.json").write_text(json.dumps(
            {"at": dt.datetime.now(dt.timezone.utc).isoformat(), "resume_from": str(resume_from), "resume_global_step": resume_gs, "loaded_modules": loaded}, indent=1))

    t0 = time.time()
    while success < DOSE:
        heads, tails = sampler.next_block()
        if sampler.epoch > last_epoch:                       # epoch boundary checkpoint (auditable)
            _ck_state(f"epoch{sampler.epoch}"); _prune_epochs(); last_epoch = sampler.epoch
        h = torch.as_tensor(heads, device=dev); tl = torch.as_tensor(tails, device=dev)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            head_emb = model(Xt.index_select(0, h))                                  # head forwarded ONCE
            tail_emb = model(Xt.index_select(0, tl.reshape(-1))).reshape(h.shape[0], G.GROUP, NC)
        radial = G.radial_from_emb(head_emb, tail_emb)                                 # guarded FP32; shared head across 10
        if arm == "grouped_umap": loss = G.grouped_umap_loss(radial)
        elif arm == "grouped_nce": loss = G.grouped_nce_loss(radial, beta)
        else: loss = coeff * G.grouped_infonce_loss(radial)
        stepped = False
        if bool(torch.isfinite(loss)):
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            prev = scaler.get_scale(); scaler.step(opt); scaler.update()
            stepped = scaler.get_scale() >= prev              # not an AMP-overflow skip -> successful positive-LR update
        if stepped:
            consec_bad = 0; success += 1
            if success in PROBE_STEPS:
                stats["exposure_probes"].append({"step": int(success), "n_pos": int(h.shape[0]), "noise_per_pos": int(tl.shape[1] - 1)})
            if success in SNAPS: _export(SNAPDIR / f"model-step{success}.pt")
            if success in STEP_CKPTS: _ck_state(f"step{success}")
        else:
            stats["amp_skips"] += 1; consec_bad += 1
            assert consec_bad < 300, f"aborting: {consec_bad} consecutive non-finite/AMP-skipped steps (nonfinite not concealed)"
    wall = time.time() - t0

    stats["positive_lr_optimizer_steps"] = success
    proc_peak = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    free, total = torch.cuda.mem_get_info(); global_used = round((total - free) / 2**30, 3)
    assert global_used < 30.0, f"global VRAM {global_used} GiB exceeds 30 GiB cap"
    coords = np.asarray(p.transform(X, batch_size=8192), np.float32); assert coords.shape == (N_EXPECT, NC)
    np.save(OUTD / f"coords-{arm}.npy", coords); _export(OUTD / f"model-{arm}.pt")
    final_beta = (float(beta.detach()) if beta is not None else None)
    if arm == "grouped_nce": assert final_beta is not None and abs(final_beta) > 0, "grouped_nce scalar never moved"
    man = {"schema": "card034-arm-2026-09-12", "arm": arm, "mode": V.MODE[arm], "n": N_EXPECT, "n_components": NC,
           "executed_steps": success, "warm_init_sha256": V.INIT_SHA, "trained_sha256": V.state_sha(model.state_dict()),
           "lr_used_min": LR, "lr_used_max": LR, "weight_decay": WD, "grad_clip": CLIP, "final_beta": final_beta,
           "infonce_coeff": (coeff if arm == "grouped_infonce" else None), "pipeline": "device_fp16",
           "snapshots": sorted(SNAPS), "step_checkpoints": sorted(STEP_CKPTS), "loaded_modules": loaded, "identity": identity,
           "resumed_from": (str(resume_from) if resume_from else None), "train_wall_s": round(wall, 1),
           "it_per_s": round(success / wall, 2) if wall > 0 else None, "proc_peak_vram_gb": proc_peak,
           "global_vram_used_gb": global_used, "train_stats": stats}
    (OUTD / f"manifest-{arm}.json").write_text(json.dumps(man, indent=1))
    V.strict_validate_arm(arm, ROOT)
    print(f"[card034 {arm}] steps={success} amp_skips={stats['amp_skips']} {wall:.0f}s beta={final_beta} "
          f"coeff={coeff} proc={proc_peak}GB global={global_used}GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
