"""Card013 arm trainer (per card013-prereg.md). card010's SAME 300K DINO draw + fixed15 graph + the SAME
fresh shared init (589895f037d406ae), champion config, explicit constant LR=0.001, 60K updates, snapshots
20K/40K/60K (+30K for equal-time context). Three branches differ ONLY in the local-scale kernel radii:
  baseline        : existing kernel (pair_scale=None).
  actual_radii    : d²/(r_i·r_j), r from card013-radii/r_actual.npy (RMS 15-NN d_HD / train-p95).
  shuffled_radii  : same per-source radius DISTRIBUTION permuted within source (r_shuffled.npy).
Radii DETACHED (autodiff through delta only); radius array bound (sha) in the admission. No anchors/replay/
deriv. Usage: run_card013_arm.py <baseline|actual_radii|shuffled_radii> <STEPS>
"""
import os, sys, time, json, hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()

SB = Path("/data/latent-basemap/sandbox")
CHAMPION = SB / "dino-arrival-t0/champion-bs16k/model.pt"
SUBD = Path("/data/latent-basemap/substrates/card010-adaptive")
SUB = SUBD / "substrate.f16.npy"; EDGES = SUBD / "edges-fixed15.npz"; INIT = SUBD / "init-card010.pt"
RADII = SB / "card013-radii"; OUTD = SB / "card013-train"
OC = SB / "overseer-codex"; SEED = 42; LR = 0.001; SNAPS = (20000, 30000, 40000, 60000)


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def _arr_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def _state_sha(model):
    h = hashlib.sha256()
    for k in sorted(model.state_dict()):
        h.update(k.encode()); h.update(model.state_dict()[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def main():
    arm = sys.argv[1]; steps = int(sys.argv[2])
    assert arm in ("baseline", "actual_radii", "shuffled_radii")
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    import torch
    OUTD.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert _sha(EDGES) == "d214a839b07113df", f"fixed15 edges sha mismatch: {_sha(EDGES)}"   # 16-char prefix
    init_obj = torch.load(str(INIT), map_location="cpu", weights_only=False)
    init_state = init_obj["model_state"]; init_sha = init_obj["init_state_sha256"]
    radii = None; radii_sha = None
    if arm != "baseline":
        rf = "r_actual.npy" if arm == "actual_radii" else "r_shuffled.npy"
        radii = np.load(RADII / rf).astype(np.float32); radii_sha = _arr_sha(radii)
        assert radii.shape[0] == 300000 and np.isfinite(radii).all() and (radii > 0).all()
    torch.manual_seed(SEED); np.random.seed(SEED)
    if dev == "cuda": torch.cuda.manual_seed_all(SEED); torch.cuda.reset_peak_memory_stats()

    pumap = ParametricUMAP.load(str(CHAMPION), device=dev)   # champion config verbatim
    pumap.model = None
    pumap.learning_rate = LR; pumap.lr_schedule = "constant"; pumap.batch_size = 16384; pumap.warmup_steps = 0
    pumap.n_epochs = 10000; pumap._max_train_steps = steps
    pumap.rankneg_window = 75000     # same 25%-of-N regime as card010 fixed15
    for a, v in (("anchor_ids_path", ""), ("anchor_hold_weight", 0.0), ("replay_bank_path", ""),
                 ("replay_weight", 0.0), ("deriv_bank_path", ""), ("deriv_weight", 0.0)):
        if hasattr(pumap, a): setattr(pumap, a, v)
    if radii is not None:
        pumap._card013_radii = radii   # loop uploads to device + enables id-stash + gathers pair_scale

    X = np.asarray(np.load(SUB, mmap_mode="r"), np.float32)   # already unit-normed
    n = X.shape[0]
    SNAPDIR = OUTD / arm; SNAPDIR.mkdir(exist_ok=True)
    admission = {"schema": "card013-admission-2026-09-12", "arm": arm, "written_before_steps": True,
                 "shared_init_sha256": init_sha, "expected_init": "589895f037d406ae",
                 "edges_sha": _sha(EDGES), "substrate_sha": _sha(SUB), "lr": LR, "lr_schedule": "constant",
                 "batch_size": 16384, "seed": SEED, "steps": steps, "rankneg_window": 75000,
                 "radii_file": (str(RADII / ("r_actual.npy" if arm == "actual_radii" else "r_shuffled.npy")) if radii is not None else None),
                 "radii_sha": radii_sha, "snapshots": list(SNAPS), "n_rows": int(n),
                 "kernel": "d2/(r_i*r_j) detached" if radii is not None else "baseline umap"}
    (OUTD / f"admission-{arm}.json").write_text(json.dumps(admission, indent=1))

    t0 = time.time()
    pumap.fit(X, precomputed_edges_path=str(EDGES), random_state=SEED, verbose=False,
              warm_start_state=init_state, snapshot_steps=SNAPS, snapshot_dir=str(SNAPDIR))
    wall = time.time() - t0
    ts = dict(getattr(pumap, "_train_stats", {}) or {})
    exec_steps = int(ts.get("executed_iters", 0)); assert exec_steps == steps, f"{exec_steps} != {steps}"
    assert abs(ts.get("lr_used_min", 0) - LR) < 1e-12 and abs(ts.get("lr_used_max", 0) - LR) < 1e-12, "LR not constant 0.001"
    warm_sha = getattr(pumap, "warm_start_sha256", None); assert warm_sha == init_sha, f"warm {warm_sha} != init {init_sha}"
    peak_vram = round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev == "cuda" else None
    coords = np.asarray(pumap.transform(X, batch_size=8192), np.float32)
    np.save(OUTD / f"coords-{arm}.npy", coords); pumap.save(str(OUTD / f"model-{arm}.pt"))
    man = {"schema": "card013-arm-2026-09-12", "arm": arm, "n": int(n), "executed_steps": exec_steps,
           "shared_init_sha256": init_sha, "warm_start_sha256": warm_sha, "trained_sha256": _state_sha(pumap.model),
           "lr_used_min": ts.get("lr_used_min"), "lr_used_max": ts.get("lr_used_max"),
           "radii_sha": radii_sha, "kernel": admission["kernel"], "snapshots": list(SNAPS),
           "train_wall_s": round(wall, 1), "it_per_s": round(exec_steps / wall, 2) if wall > 0 else None,
           "peak_vram_gb": peak_vram, "stop_reason": ts.get("stop_reason"), "train_stats": ts}
    (OUTD / f"manifest-{arm}.json").write_text(json.dumps(man, indent=1))
    print(f"[card013 {arm}] steps={exec_steps} {wall:.0f}s vram={peak_vram}GB kernel={admission['kernel']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
