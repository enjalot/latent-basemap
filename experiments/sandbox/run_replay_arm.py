"""Cards 006/007 replay-arm runner. Same anchored-update mechanism as run_jina_update.py
(warm-start ORIGINAL head, FINAL graph, anchor λ=w, seed, budget, LR — hold_gen keeps the
edge/negative stream matched across arms) PLUS the off-graph replay term. IN vs OUT arm
differ ONLY by which replay bank is passed. GPU/device path.

Env (in addition to run_jina_update's EVOLBENCH_LAMBDA_*):
  REPLAY_BANK       replay bank .npz (IN or OUT) — required, weight>0
  REPLAY_WEIGHT     default 0.02
  REPLAY_FRACTION   default 0.05  (819/step at bs16384)
  REPLAY_SEED       default 51549
  SNAPSHOT_STEPS    comma-sep attempted-step targets for inference-only snapshots (e.g. 35000,70000,140000)
  SNAPSHOT_DIR      dir for model-step{N}.pt (default OUTD)
  CHECKPOINT_DIR    optional epoch-boundary resume checkpoints
  CKPT_EVERY_EPOCHS default 0 (off) — set >0 with CHECKPOINT_DIR for resumable runs
  RESUME_FROM       optional epoch checkpoint to resume
Usage: run_replay_arm.py <w_anchor> <n_epochs>
"""
import os, sys, time, json, hashlib
from pathlib import Path

# ── Card008 code isolation ──────────────────────────────────────────────────
# card008's three schedule arms must all use ONE immutable trainer (commit dd3038a)
# even while core.py changes for card009. The live chain invokes THIS mutable file;
# for a card008 OUTD we re-exec the FROZEN dd3038a entrypoint (its own _paths imports
# the frozen core) BEFORE importing core/_paths or reading any other env. execv keeps
# the PID and the held GPU-flock fd. All other cards keep the normal live path.
_C8_FROZEN_ENTRY = "/data/latent-basemap/sandbox/overseer-codex/card008-code-dd3038a/experiments/sandbox/run_replay_arm.py"
if ("card008" in os.environ.get("EVOLBENCH_LAMBDA_OUTD", "")
        and os.environ.get("_CARD008_DISPATCHED") != "1"
        and os.path.exists(_C8_FROZEN_ENTRY)
        and os.path.realpath(__file__) != os.path.realpath(_C8_FROZEN_ENTRY)):
    os.environ["_CARD008_DISPATCHED"] = "1"
    sys.stderr.write(f"[card008-dispatch] routing to FROZEN dd3038a entrypoint: {_C8_FROZEN_ENTRY}\n"); sys.stderr.flush()
    os.execv(sys.executable, [sys.executable, _C8_FROZEN_ENTRY, *sys.argv[1:]])

import numpy as np

HEAD = Path(os.environ["EVOLBENCH_LAMBDA_HEAD"]); EDGES = Path(os.environ["EVOLBENCH_LAMBDA_EDGES"])
ANCHOR = Path(os.environ.get("EVOLBENCH_LAMBDA_ANCHOR", "")); OUTD = Path(os.environ["EVOLBENCH_LAMBDA_OUTD"])
TRANCHE = os.environ["EVOLBENCH_LAMBDA_TRANCHE_PATHS"]; SEED = int(os.environ.get("EVOLBENCH_LAMBDA_SEED", "42"))
RBANK = Path(os.environ["REPLAY_BANK"]); RWEIGHT = float(os.environ.get("REPLAY_WEIGHT", "0.02"))
RFRAC = float(os.environ.get("REPLAY_FRACTION", "0.05")); RSEED = int(os.environ.get("REPLAY_SEED", "51549"))
SNAP = tuple(int(s) for s in os.environ.get("SNAPSHOT_STEPS", "").split(",") if s.strip())
SNAPDIR = Path(os.environ.get("SNAPSHOT_DIR", str(OUTD)))
CKPTDIR = os.environ.get("CHECKPOINT_DIR", ""); CKPT_EVERY = int(os.environ.get("CKPT_EVERY_EPOCHS", "0"))
RESUME = os.environ.get("RESUME_FROM", "")


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def _state_hash(model):
    h = hashlib.sha256()
    for k in sorted(model.state_dict()):
        h.update(k.encode()); h.update(model.state_dict()[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _file_sha(path, full=True):
    p = Path(path)
    if not full:                                  # cheap identity for large files (edges/substrate)
        st = p.stat(); return f"size{st.st_size}-mtime{int(st.st_mtime)}"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _bank_content_sha(path):                       # MUST match core's ids+X+targets hash
    b = np.load(path); rids = np.asarray(b["replay_ids"]); rx = np.asarray(b["replay_X"]); rt = np.asarray(b["replay_targets"], np.float32)
    return hashlib.sha256(np.ascontiguousarray(np.sort(rids.astype(np.int64))).tobytes()
                          + np.ascontiguousarray(rx).tobytes()
                          + np.ascontiguousarray(rt.astype(np.float32)).tobytes()).hexdigest()[:16]


def main():
    w = float(sys.argv[1]); n_epochs = int(sys.argv[2])
    sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    import torch
    OUTD.mkdir(parents=True, exist_ok=True); SNAPDIR.mkdir(parents=True, exist_ok=True)
    for f in ([HEAD, EDGES, RBANK] + ([ANCHOR] if w > 0 else [])):
        if not Path(f).exists():
            raise SystemExit(f"missing prerequisite: {f}")
    if RWEIGHT > 0 and not RBANK.exists():
        raise SystemExit(f"replay enabled but bank missing: {RBANK}")   # fail-closed (core re-asserts)
    torch.manual_seed(SEED); np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED); torch.cuda.reset_peak_memory_stats()

    pumap = ParametricUMAP.load(str(HEAD), device="cuda")
    warm_hash = _state_hash(pumap.model)
    warm_state = {k: v.detach().clone() for k, v in pumap.model.state_dict().items()}
    pumap.model = None
    if w > 0:
        pumap.anchor_ids_path = str(ANCHOR); pumap.anchor_hold_weight = w
        pumap.anchor_hold_fraction = 0.05; pumap.anchor_holdout_fraction = 0.10
        pumap.anchored_init = "none"; pumap.anchored_init_path = ""
    else:
        pumap.anchor_ids_path = ""; pumap.anchor_hold_weight = 0.0
    # off-graph replay
    pumap.replay_bank_path = str(RBANK); pumap.replay_weight = RWEIGHT
    pumap.replay_fraction = RFRAC; pumap.replay_seed = RSEED
    pumap.batch_size = 16384; pumap.n_epochs = max(n_epochs, 50); pumap.warmup_steps = 0
    pumap._max_train_steps = int(os.environ.get("EVOLBENCH_LAMBDA_MAXSTEPS", "140000"))
    # Optional LR-schedule control (card008 finishing comparison). Defaults leave the
    # loaded/plateau behavior untouched. LR_SCHEDULE=cosine needs TOTAL_STEPS_EST (the
    # cosine horizon in successful updates); LR_MIN sets the cosine floor (0=to zero).
    if os.environ.get("LR"):
        pumap.learning_rate = float(os.environ["LR"])
    if os.environ.get("LR_SCHEDULE"):
        pumap.lr_schedule = os.environ["LR_SCHEDULE"]
    if os.environ.get("LR_MIN"):
        pumap.lr_min = float(os.environ["LR_MIN"])
    if os.environ.get("TOTAL_STEPS_EST"):
        pumap.total_steps_estimate = int(os.environ["TOTAL_STEPS_EST"])

    _paths = [p for p in TRANCHE.split(",") if p]
    X = np.asarray(np.load(_paths[0], mmap_mode="r"), np.float32) if len(_paths) == 1 \
        else np.concatenate([np.asarray(np.load(p, mmap_mode="r"), np.float32) for p in _paths])
    if os.environ.get("PRENORMED") != "1":
        X = _norm(X)
    n = X.shape[0]
    # ── Frozen admission receipt BEFORE any optimizer step (review ref 4) ──
    tag = os.environ.get("EVOLBENCH_LAMBDA_TAG", f"w{w:g}")
    admission = {"schema": "cards006-007-admission-2026-09-10", "tag": tag, "written_before_steps": True,
                 "warm_start_hash": warm_hash, "head_sha": _file_sha(HEAD), "anchor_sha": _file_sha(ANCHOR) if w > 0 else None,
                 "edges_id": _file_sha(EDGES, full=False), "substrate_id": _file_sha(_paths[0], full=False) if len(_paths) == 1 else [_file_sha(p, full=False) for p in _paths],
                 "replay_bank": str(RBANK), "replay_bank_content_sha": _bank_content_sha(RBANK),
                 "replay_weight": RWEIGHT, "replay_fraction": RFRAC, "replay_seed": RSEED,
                 "anchor_lambda": w, "anchor_hold_fraction": 0.05, "anchor_holdout_fraction": 0.10,
                 "batch_size": 16384, "seed": SEED, "max_train_steps": pumap._max_train_steps,
                 "lr_schedule": pumap.lr_schedule, "learning_rate": pumap.learning_rate,
                 "lr_min": pumap.lr_min, "total_steps_estimate": int(getattr(pumap, "total_steps_estimate", 0)),
                 "snapshot_steps": list(SNAP), "n_rows": int(n), "prenormed": os.environ.get("PRENORMED") == "1"}
    (OUTD / f"admission-{tag}.json").write_text(json.dumps(admission, indent=1))
    fit_kw = dict(precomputed_edges_path=str(EDGES), random_state=SEED, verbose=False, warm_start_state=warm_state)
    if SNAP:
        fit_kw.update(snapshot_steps=SNAP, snapshot_dir=str(SNAPDIR))
    if CKPTDIR and CKPT_EVERY > 0:
        fit_kw.update(checkpoint_dir=CKPTDIR, checkpoint_every_epochs=CKPT_EVERY)
    if RESUME:
        fit_kw.update(resume_from=RESUME)
    t0 = time.time()
    pumap.fit(X, **fit_kw)
    wall = time.time() - t0; trained_hash = _state_hash(pumap.model)
    steps = int(pumap._train_stats.get("executed_iters", 0)) if hasattr(pumap, "_train_stats") else 0
    peak_vram_gb = round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None
    its = round(steps / wall, 2) if wall > 0 else None
    proj_140k_h = round(140000 / its / 3600, 3) if its else None
    coords = np.asarray(pumap.transform(X, batch_size=8192), np.float32)
    np.save(OUTD / f"coords-{tag}.npy", coords); pumap.save(str(OUTD / f"model-{tag}.pt"))
    for attr, nm in (("anchor_holdout_ids_", "anchor_holdout_ids"), ("anchor_ids_", "anchor_active_ids")):
        v = getattr(pumap, attr, None)
        if v is not None:
            np.save(OUTD / f"{nm}-{tag}.npy", np.asarray(v))
    r = np.linalg.norm(coords.astype(np.float64) - np.median(coords, 0), axis=1)
    ts = dict(getattr(pumap, "_train_stats", {}) or {})   # persist ALL train_stats (review ref 3)
    pipe = dict(getattr(pumap, "_pipeline_info", {}) or {})   # observed residency/pipeline (not a claimed flag)
    man = {"schema": "cards006-007-replay-arm-2026-09-10", "tag": tag, "w_anchor": w, "n": int(n),
           "replay_bank": str(RBANK), "replay_bank_sha": getattr(pumap, "_replay_bank_sha", None),
           "replay_bank_content_sha_admission": admission["replay_bank_content_sha"],
           "replay_weight": RWEIGHT, "replay_fraction": RFRAC, "replay_seed": RSEED,
           "max_train_steps": pumap._max_train_steps, "executed_steps": steps, "batch_size": 16384, "seed": SEED,
           "lr_schedule": pumap.lr_schedule, "learning_rate": pumap.learning_rate, "lr_min": pumap.lr_min,
           "total_steps_estimate": int(getattr(pumap, "total_steps_estimate", 0)),
           "train_wall_s": round(wall, 1), "it_per_s": its, "projected_140k_gpu_h": proj_140k_h,
           "peak_vram_gb": peak_vram_gb, "warm_start_hash": warm_hash, "trained_hash": trained_hash,
           "warm_start_changed": bool(warm_hash != trained_hash), "anchored": bool(w > 0),
           "snapshot_steps": list(SNAP), "head": str(HEAD), "edges": str(EDGES),
           "observed_x_residency": pipe.get("x_residency"), "observed_pipeline": pipe.get("pipeline"),
           "observed_positive_sampling": pipe.get("positive_sampling"),
           "grouped_negatives_env": os.environ.get("GROUPED_NEGATIVES"), "endpoint_reuse_env": os.environ.get("ENDPOINT_REUSE"),
           "train_stats": ts, "stop_reason": ts.get("stop_reason"),
           "coords_p50_radius": round(float(np.percentile(r, 50)), 3)}
    (OUTD / f"manifest-{tag}.json").write_text(json.dumps(man, indent=1))
    print(f"[replay-arm {tag}] w={w} n={n:,} steps={steps} {wall:.0f}s it/s={its} "
          f"proj140k={proj_140k_h}h vram={peak_vram_gb}GB changed={warm_hash!=trained_hash}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
