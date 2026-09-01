"""Evolution benchmark — ARM A (ours) + its bounds (plan §2). GPU (.venv/torch).

Arm A holds unchanged points bitwise-still between retrains (a FROZEN head transforms arrivals in
seconds) and retrains only on the drift trigger. Generalized over a RETRAIN SCHEDULE {k: head_path}:
the active head at snapshot k is the most-recent scheduled head with j<=k. Because a fixed head's
transform is deterministic, the shared first-n_{k-1} points are BITWISE-IDENTICAL between consecutive
snapshots that share a head → churn 0 by construction; only a head SWITCH (retrain) moves them, and
there we procrustes-align the new head's frame to the previous snapshot (ls-umap-align style).

Schedules (via SCHEDULE_JSON env, {k: head_path}):
  frozen (floor)       : {0: S0-head}                        — never retrains, churn 0, quality decays.
  drift-triggered (A)  : {0: S0-head, 3: S3-head}            — retrains once, at the T3 OOD injection.
  retrain-every (ceil) : {0:S0, 1:S1, ..., 5:S5}             — best quality, worst churn.

Emits per snapshot: coords + transform wall (placement latency) + a head_switch flag. Churn/quality
are computed by the scorer from these coords vs the snapshot truths.
Env: EVOLBENCH_K, SCHEDULE_JSON, ARM_LABEL. Output -> <sandbox>/evolbench-armA-<label>/coords-S{k}.npy + manifest.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

SB = Path("/data/latent-basemap/sandbox")
K = int(os.environ.get("EVOLBENCH_K", "5"))
LABEL = os.environ.get("ARM_LABEL", "frozen")
SCHEDULE = {int(k): v for k, v in json.loads(os.environ.get("SCHEDULE_JSON", '{"0": ""}')).items()}


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def _load_Sk(k):
    parts = ["T0"] + [f"T{j}" for j in range(1, k + 1)]
    return _norm(np.concatenate([
        np.asarray(np.load(f"/data/latent-basemap/substrates/evolbench/{t}/substrate.f32.npy",
                           mmap_mode="r"), dtype=np.float32) for t in parts]))


def _procrustes(src, ref, src_shared):
    mu_s = src_shared.mean(0); mu_r = ref.mean(0)
    A = src_shared - mu_s; B = ref - mu_r
    U, S, Vt = np.linalg.svd(A.T @ B); R = U @ Vt
    scale = S.sum() / max((A ** 2).sum(), 1e-9)
    return (src - mu_s) @ (scale * R) + mu_r


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    # default S0 head path if schedule[0] is empty
    if not SCHEDULE.get(0):
        SCHEDULE[0] = str(SB / "evolbench-S0/champion-bs16k/model.pt")
    outd = SB / f"evolbench-armA-{LABEL}"; outd.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "evolbench-armA-2026-08-31", "label": LABEL, "schedule": {str(k): v for k, v in SCHEDULE.items()}, "snapshots": []}
    active = None; active_k = None; prev_coords = None; prev_n = 0
    for k in range(K + 1):
        if k in SCHEDULE:                       # head switch (retrain) at this snapshot
            active = ParametricUMAP.load(SCHEDULE[k], device="cuda"); active_k = k
        X = _load_Sk(k); n = int(X.shape[0])
        t0 = time.time()
        coords = np.asarray(active.transform(X, batch_size=8192), dtype=np.float32)
        wall = time.time() - t0
        switched = (active_k == k) and (prev_coords is not None)
        if switched:                            # align the new head's frame to the previous snapshot
            coords = _procrustes(coords, prev_coords, coords[:prev_n])
        np.save(outd / f"coords-S{k}.npy", coords)
        manifest["snapshots"].append({"k": k, "n": n, "transform_wall_s": round(wall, 2),
                                      "head_switch": bool(active_k == k), "active_head_k": active_k})
        print(f"armA[{LABEL}] S{k}: n={n:,} transform {wall:.1f}s "
              f"{'(RETRAIN switch, procrustes)' if switched else '(frozen)'}", flush=True)
        prev_coords = coords; prev_n = n
        del X
    (outd / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\nwrote {outd}/manifest.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
