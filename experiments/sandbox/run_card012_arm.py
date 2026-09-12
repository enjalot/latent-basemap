"""Card012 arm trainer (per card012-prereg.md + Hook B reviews). DINO T0 warm-start, UNCHANGED 2.4M final
graph, original active-anchor split, constant LR 1e-4, replay .02 (819/step), 140K updates. Initial bank =
original OUT bank; REFRESH at 35K/70K/105K via the real selector (build_card012_bank.select) for the arm:
  uniform         : 200K uniform within per-source quotas.
  error_directed  : 100K uniform + 100K highest-error-20%/source (native squared student-T0 displacement).
Step-checkpoints at each refresh for genuine resumability. Admission identity is DERIVED from the ACTUAL
runtime (arm / refresh-callback-enabled / cadence / pool+teacher/selection-seed / INITIAL bank sha) and the
support manifest is bound. Frozen executable checkout + loaded-module hashes recorded.
Usage: run_card012_arm.py <uniform|error_directed> <STEPS>
"""
import os, sys, time, json, hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import build_card012_bank as BANK

SB = Path("/data/latent-basemap/sandbox"); OC = SB / "overseer-codex"
HEAD = SB / "dino-arrival-t0/champion-bs16k/model.pt"
EDGES = SB / "dino-arrival-final/edges-k15-fuzzy.npz"
ANCHOR = SB / "dino-arrival-t0/anchor.npz"
OUT_BANK = OC / "card006_out_bank.npz"
SUB = Path("/data/latent-basemap/substrates/dino-arrival-final/substrate.f16.npy")
POOLMAN = OC / "card012-pool-manifest.json"
OUTD = SB / "card012-train"; BANKDIR = OUTD / "banks"
SEED = 42; SELSEED = 12012; LR = 1e-4
# refresh cadence (env-overridable ONLY for the bounded smoke test; production uses 35K/70K/105K)
REFRESH_STEPS = tuple(int(s) for s in os.environ.get("CARD012_REFRESH_STEPS", "35000,70000,105000").split(","))


def _fsha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def _state_sha(model):
    h = hashlib.sha256()
    for k in sorted(model.state_dict()):
        h.update(k.encode()); h.update(model.state_dict()[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def main():
    arm = sys.argv[1]; steps = int(sys.argv[2]); assert arm in ("uniform", "error_directed")
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    import torch
    CKPTDIR = OUTD / "ckpt" / arm            # PER-ARM isolation (was shared -> arms overwrote each other)
    for d in (OUTD, BANKDIR, CKPTDIR): d.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    RESUME = os.environ.get("CARD012_RESUME", "")   # durable resume path (fail-closed via admission identity)
    poolman = json.loads(POOLMAN.read_text())
    ob = np.load(OUT_BANK)
    initial_bank_sha = BANK._content_sha(ob["replay_ids"], np.asarray(ob["replay_X"], np.float16),
                                         np.asarray(ob["replay_targets"], np.float32))
    torch.manual_seed(SEED); np.random.seed(SEED)
    if dev == "cuda": torch.cuda.manual_seed_all(SEED); torch.cuda.reset_peak_memory_stats()

    pumap = ParametricUMAP.load(str(HEAD), device=dev)
    warm = {k: v.detach().clone() for k, v in pumap.model.state_dict().items()}
    warm_sha = _state_sha(pumap.model); pumap.model = None
    pumap.learning_rate = LR; pumap.lr_schedule = "constant"; pumap.batch_size = 16384; pumap.warmup_steps = 0
    pumap.n_epochs = 100000; pumap._max_train_steps = steps
    # original active-anchor split/radius + OUT replay (matching the anchored comparator)
    pumap.anchor_ids_path = str(ANCHOR); pumap.anchor_hold_weight = 0.02
    pumap.anchor_hold_fraction = 0.05; pumap.anchor_holdout_fraction = 0.10
    pumap.anchored_init = "none"; pumap.anchored_init_path = ""
    pumap.replay_bank_path = str(OUT_BANK); pumap.replay_weight = 0.02; pumap.replay_fraction = 0.05; pumap.replay_seed = 51549
    for a, v in (("deriv_bank_path", ""), ("deriv_weight", 0.0)):
        if hasattr(pumap, a): setattr(pumap, a, v)

    # refresh callback: real selector for the arm; restore model train-mode after the eval() scoring.
    stage_meta = {}

    def refresh(pu, step):
        was_training = pu.model.training
        bankpath = BANKDIR / f"{arm}-step{step}.npz"
        meta = BANK.select(arm, pu.model, dev, bankpath, seed=SELSEED + step)   # PER-STAGE seed -> genuinely new refresh
        if was_training: pu.model.train()
        stage_meta[step] = meta
        (OUTD / f"stage-{arm}-{step}.json").write_text(json.dumps({"step": step, **meta}, indent=1))
        return str(bankpath)

    pumap._replay_refresh_fn = refresh
    pumap._replay_refresh_steps = set(REFRESH_STEPS)
    pumap._checkpoint_step_targets = set(REFRESH_STEPS)     # checkpoint AFTER each refresh (resumable)
    # admission identity DERIVED from the actual runtime configuration (not a copied expected dict)
    pumap._card012_identity = {
        "arm": arm, "refresh_enabled": bool(pumap._replay_refresh_fn is not None),
        "refresh_steps": sorted(pumap._replay_refresh_steps),
        "pool_ids_sha": poolman["pool_ids_sha"], "teacher_sha": poolman["teacher_sha"],
        "selection_seed": SELSEED, "initial_bank_sha": initial_bank_sha}

    X = np.asarray(np.load(SUB, mmap_mode="r"), np.float32)
    n = X.shape[0]
    frozen = OC / "card012-code-frozen"; frozen.mkdir(exist_ok=True)
    import basemap.pumap.parametric_umap.core as _core
    import basemap.pumap.parametric_umap.datasets.edge_list_dataset as _eld
    loaded_module_shas = {"core": _fsha(_core.__file__), "edge_list_dataset": _fsha(_eld.__file__),
                          "run_card012_arm": _fsha(__file__), "build_card012_bank": _fsha(BANK.__file__)}
    # EXECUTION PINNING: if a frozen manifest exists, the LOADED modules must match it byte-for-byte
    # (fail-closed) — so the executed code == the frozen checkout, not just a post-hoc source copy.
    _frozen_man = OC / "card012-code-frozen" / "manifest.json"
    if _frozen_man.exists():
        _fm = json.loads(_frozen_man.read_text())
        _mm = {k: v for k, v in loaded_module_shas.items() if k in _fm.get("module_shas", {})}
        assert _mm == {k: _fm["module_shas"][k] for k in _mm}, \
            f"card012 execution-pinning FAIL: loaded modules {_mm} != frozen {_fm.get('module_shas')}"
    admission = {"schema": "card012-admission-2026-09-11", "arm": arm, "written_before_steps": True,
                 "warm_start_sha": warm_sha, "head_sha": _fsha(HEAD), "edges_sha": _fsha(EDGES),
                 "anchor_sha": _fsha(ANCHOR), "lr": LR, "lr_schedule": "constant", "batch_size": 16384,
                 "seed": SEED, "steps": steps, "replay_weight": 0.02, "replay_fraction": 0.05,
                 "anchor_hold_weight": 0.02, "identity": pumap._card012_identity,
                 "initial_bank_sha": initial_bank_sha, "support_manifest": poolman.get("support_manifest"),
                 "pool_manifest_sha": _fsha(POOLMAN), "loaded_module_shas": loaded_module_shas,
                 "n_rows": int(n)}
    (OUTD / f"admission-{arm}.json").write_text(json.dumps(admission, indent=1))

    t0 = time.time()
    _fit_kw = dict(precomputed_edges_path=str(EDGES), random_state=SEED, verbose=False, checkpoint_dir=str(CKPTDIR))
    if RESUME:                                     # durable resume (admission identity fails closed on mismatch)
        _fit_kw["resume_from"] = RESUME
    else:
        _fit_kw["warm_start_state"] = warm
    pumap.fit(X, **_fit_kw)
    wall = time.time() - t0
    ts = dict(getattr(pumap, "_train_stats", {}) or {})
    exec_steps = int(ts.get("executed_iters", 0)); assert exec_steps == steps, f"{exec_steps} != {steps}"
    assert abs(ts.get("lr_used_min", 0) - LR) < 1e-12 and abs(ts.get("lr_used_max", 0) - LR) < 1e-12, "LR not constant"
    refreshes = ts.get("replay_refreshes", [])
    assert len(refreshes) == len(REFRESH_STEPS), f"expected {len(REFRESH_STEPS)} refreshes, got {len(refreshes)}"
    peak_vram = round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev == "cuda" else None
    coords = np.asarray(pumap.transform(X, batch_size=8192), np.float32)
    np.save(OUTD / f"coords-{arm}.npy", coords); pumap.save(str(OUTD / f"model-{arm}.pt"))
    man = {"schema": "card012-arm-2026-09-11", "arm": arm, "n": int(n), "executed_steps": exec_steps,
           "warm_start_sha": warm_sha, "trained_sha256": _state_sha(pumap.model),
           "lr_used_min": ts.get("lr_used_min"), "lr_used_max": ts.get("lr_used_max"),
           "replay_refreshes": refreshes, "stage_meta": {str(k): v for k, v in stage_meta.items()},
           "identity": pumap._card012_identity, "train_wall_s": round(wall, 1),
           "it_per_s": round(exec_steps / wall, 2) if wall > 0 else None, "peak_vram_gb": peak_vram,
           "stop_reason": ts.get("stop_reason"), "resumed_from": RESUME or None,
           "train_stats": ts}                      # FULL train_stats persisted (reporting completeness)
    (OUTD / f"manifest-{arm}.json").write_text(json.dumps(man, indent=1))
    print(f"[card012 {arm}] steps={exec_steps} {wall:.0f}s vram={peak_vram}GB refreshes={len(refreshes)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
