"""Card011 arm trainer (per card011-prereg.md + preflight-review-2 fixes). Warm-starts the card010
fixed15 head (WEIGHTS) with a UNIFORM optimizer reset, same fixed15 graph + 300K inputs, 60K constant-LR
1e-4 updates, no anchors/replay. Arms differ ONLY in the injected 5% of negative slots:
  ordinary          : no injection.
  collision         : verified-collision pool; re-mines from the CURRENT map at 0/20K/40K and PERSISTS its
                      per-stage histogram+pairs (the reference the random arm matches). MUST run before random.
  verified_random   : matches the COLLISION arm's EXACT persisted histogram at each stage (0/20K/40K).
Binds inject identity (frac/seed/refresh-steps/per-stage pool content SHA) in the admission. Single clean
run per arm (no mid-run resume; on failure the chain re-runs the arm idempotently).
Usage: run_card011_arm.py <ordinary|collision|verified_random> <STEPS>
"""
import os, sys, time, json, hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import build_card011_pools as P

SB = Path("/data/latent-basemap/sandbox")
HEAD = SB / "card010-train/model-fixed15.pt"
SUB = Path("/data/latent-basemap/substrates/card010-adaptive/substrate.f16.npy")
EDGES = Path("/data/latent-basemap/substrates/card010-adaptive/edges-fixed15.npz")
OUTD = SB / "card011-train"
SEED = 42; INJECT_FRAC = 0.05; REFRESH_STEPS = (20000, 40000); LR = 1e-4


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _pairs_sha(src, dst):
    return hashlib.sha256(np.ascontiguousarray(src.astype(np.int64)).tobytes()
                          + np.ascontiguousarray(dst.astype(np.int64)).tobytes()).hexdigest()[:16]


def _state_sha(model):
    h = hashlib.sha256()
    for k in sorted(model.state_dict()):
        h.update(k.encode()); h.update(model.state_dict()[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def main():
    tag = sys.argv[1]; steps = int(sys.argv[2])
    assert tag in ("ordinary", "collision", "verified_random")
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    import torch
    OUTD.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert _sha(EDGES) == "d214a839b07113dff2c29b225da9f38008f86a0b2cb3662a39d14bc0542d4450", "fixed15 edges sha mismatch"
    sub, knn_idx, knn_dist, src_code = P.load_static()
    X = np.asarray(np.load(SUB, mmap_mode="r"), np.float32)                    # unit-normed (PRENORMED)
    n = X.shape[0]
    torch.manual_seed(SEED); np.random.seed(SEED)
    if dev == "cuda":
        torch.cuda.manual_seed_all(SEED); torch.cuda.reset_peak_memory_stats()

    pumap = ParametricUMAP.load(str(HEAD), device=dev)                         # fixed15 config+weights
    warm_state = {k: v.detach().clone() for k, v in pumap.model.state_dict().items()}
    warm_sha = _state_sha(pumap.model)
    pumap.model = None                                                        # uniform optimizer reset via warm_start_state
    pumap.learning_rate = LR; pumap.lr_schedule = "constant"; pumap.batch_size = 16384
    pumap.warmup_steps = 0; pumap.n_epochs = 10000; pumap._max_train_steps = steps
    for attr, val in (("anchor_ids_path", ""), ("anchor_hold_weight", 0.0),
                      ("replay_bank_path", ""), ("replay_weight", 0.0),
                      ("deriv_bank_path", ""), ("deriv_weight", 0.0)):
        if hasattr(pumap, attr):
            setattr(pumap, attr, val)

    # ── injection config (instance attrs read by the isolated fit() hook) ──
    stage_shas = {}
    if tag != "ordinary":
        pumap._inject_frac_cfg = INJECT_FRAC
        pumap._inject_refresh_steps = REFRESH_STEPS
        z0 = np.load(P.STAGE_DIR / "collision-stage0.npz")                     # step-0 collisions (from preflight)
        if tag == "collision":
            ps, pd = z0["src"], z0["dst"]
            stage_shas[0] = _pairs_sha(ps, pd)

            def refresh(pu, step):
                coords = np.asarray(pu.transform(X, batch_size=8192), np.float64)
                coll = P.mine_collisions(coords, sub, knn_idx, knn_dist, src_code, device=dev)
                via = P.viability(coll, src_code)                              # refresh viability, fail-closed
                if not via["viable"]:
                    raise RuntimeError(f"card011 collision refresh viability FAIL at step {step}: {via}")
                P._persist_stage(step, coll)                                   # reference for the random arm
                stage_shas[step] = _pairs_sha(coll["src"], coll["dst"])
                (OUTD / f"refresh-viability-{step}.json").write_text(json.dumps(via, indent=1))
                return coll["src"], coll["dst"]
        else:  # verified_random: match the COLLISION arm's EXACT persisted histogram at each stage
            rng0 = np.random.default_rng(SEED)
            ps, pd = P.sample_matched_random(z0["hist"], sub, knn_idx, knn_dist, src_code, rng0, device=dev)
            stage_shas[0] = _pairs_sha(ps, pd)
            np.savez(P.STAGE_DIR / "random-stage0.npz", src=ps, dst=pd)         # persist actual random pool

            def refresh(pu, step):
                hist = P.load_stage_hist(step)                                 # collision arm must have run first
                rng = np.random.default_rng(SEED + step)
                rs, rd = P.sample_matched_random(hist, sub, knn_idx, knn_dist, src_code, rng, device=dev)
                stage_shas[step] = _pairs_sha(rs, rd)
                np.savez(P.STAGE_DIR / f"random-stage{step}.npz", src=rs, dst=rd)   # persist actual random pool
                return rs, rd
        pumap._inject_pool = (ps.astype(np.int64), pd.astype(np.int64))
        pumap._inject_refresh_fn = refresh

    admission = {"schema": "card011-admission-2026-09-11", "tag": tag, "written_before_steps": True,
                 "warm_start_sha": warm_sha, "head": str(HEAD), "edges_sha256": _sha(EDGES),
                 "lr": LR, "lr_schedule": "constant", "batch_size": 16384, "seed": SEED, "steps": steps,
                 "inject_frac": (INJECT_FRAC if tag != "ordinary" else 0.0),
                 "inject_gen_seed": SEED + 918273, "refresh_steps": list(REFRESH_STEPS),
                 "initial_pool_sha": stage_shas.get(0), "rankneg_window": getattr(pumap, "rankneg_window", None),
                 "no_anchors_no_replay": True}
    (OUTD / f"admission-{tag}.json").write_text(json.dumps(admission, indent=1))

    t0 = time.time()
    pumap.fit(X, precomputed_edges_path=str(EDGES), random_state=SEED, verbose=False, warm_start_state=warm_state)
    wall = time.time() - t0
    trained_sha = _state_sha(pumap.model)
    ts = dict(getattr(pumap, "_train_stats", {}) or {})
    exec_steps = int(ts.get("executed_iters", 0))
    assert exec_steps == steps, f"executed {exec_steps} != {steps}"
    assert abs(ts.get("lr_used_min", 0) - LR) < 1e-12 and abs(ts.get("lr_used_max", 0) - LR) < 1e-12, "LR not constant 1e-4"
    if tag != "ordinary":                                                     # zero-exposure intervention must FAIL
        assert int(ts.get("inject_slots_applied", 0)) > 0, "injection configured but NEVER applied (zero exposure)"
        assert int(ts.get("inject_batches_applied", 0)) >= steps - 50, \
            f"injection applied to only {ts.get('inject_batches_applied')} of ~{steps} batches"
    peak_vram = round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev == "cuda" else None
    coords = np.asarray(pumap.transform(X, batch_size=8192), np.float32)
    np.save(OUTD / f"coords-{tag}.npy", coords); pumap.save(str(OUTD / f"model-{tag}.pt"))
    man = {"schema": "card011-arm-2026-09-11", "tag": tag, "n": int(n), "executed_steps": exec_steps,
           "warm_start_sha": warm_sha, "trained_sha256": trained_sha, "warm_start_changed": bool(warm_sha != trained_sha),
           "lr": LR, "lr_used_min": ts.get("lr_used_min"), "lr_used_max": ts.get("lr_used_max"),
           "inject_frac": admission["inject_frac"], "inject_stage_shas": {str(k): v for k, v in stage_shas.items()},
           "inject_refreshes": ts.get("inject_refreshes"),
           "inject_slots_applied": int(ts.get("inject_slots_applied", 0)),
           "inject_batches_applied": int(ts.get("inject_batches_applied", 0)), "train_wall_s": round(wall, 1),
           "it_per_s": round(exec_steps / wall, 2) if wall > 0 else None, "peak_vram_gb": peak_vram,
           "stop_reason": ts.get("stop_reason")}
    (OUTD / f"manifest-{tag}.json").write_text(json.dumps(man, indent=1))
    print(f"[card011 {tag}] steps={exec_steps} {wall:.0f}s vram={peak_vram}GB changed={warm_sha != trained_sha} "
          f"refreshes={ts.get('inject_refreshes')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
