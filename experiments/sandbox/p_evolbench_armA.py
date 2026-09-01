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


def _fit_procrustes(src_shared, ref_shared):
    """Fit the similarity transform (scale*R + t) aligning src_shared->ref_shared (both the SAME shared
    points, row-aligned). Returns (mu_s, scaleR, mu_r) to APPLY to any coords in the src head's frame:
    (X - mu_s) @ scaleR + mu_r. Fit ONCE at a head switch, then reused for every snapshot that head
    produces — otherwise later snapshots of the same head stay in the head's RAW frame while the previous
    (aligned) snapshot is in the aligned frame, injecting a spurious churn spike at the first non-switch
    snapshot after a retrain (the 2026-09-01 S4=0.945 artifact)."""
    mu_s = src_shared.mean(0); mu_r = ref_shared.mean(0)
    A = src_shared - mu_s; B = ref_shared - mu_r
    U, S, Vt = np.linalg.svd(A.T @ B); R = U @ Vt
    scale = S.sum() / max((A ** 2).sum(), 1e-9)
    return mu_s, (scale * R).astype(np.float32), mu_r


def _apply_xform(src, xform):
    mu_s, scaleR, mu_r = xform
    return ((src - mu_s) @ scaleR + mu_r).astype(np.float32)


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _paths import ensure_paths; ensure_paths()  # repo root on sys.path (parents[2], not [1])
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    # default S0 head path if schedule[0] is empty
    if not SCHEDULE.get(0):
        SCHEDULE[0] = str(SB / "evolbench-S0/champion-bs16k/model.pt")
    outd = SB / f"evolbench-armA-{LABEL}"; outd.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "evolbench-armA-2026-08-31", "label": LABEL, "schedule": {str(k): v for k, v in SCHEDULE.items()}, "snapshots": []}
    active = None; active_k = None; active_xform = None; prev_coords = None; prev_n = 0
    for k in range(K + 1):
        if k in SCHEDULE:                       # head switch (retrain) at this snapshot
            active = ParametricUMAP.load(SCHEDULE[k], device="cuda"); active_k = k
            active_xform = None                 # refit the alignment for the NEW head at this snapshot
        X = _load_Sk(k); n = int(X.shape[0])
        t0 = time.time()
        coords = np.asarray(active.transform(X, batch_size=8192), dtype=np.float32)
        wall = time.time() - t0
        switched = (active_k == k) and (prev_coords is not None)
        if switched and active_xform is None:   # fit ONCE at the switch: shared points -> prev timeline
            active_xform = _fit_procrustes(coords[:prev_n], prev_coords)
        if active_xform is not None:            # apply this head's alignment to EVERY snapshot it makes
            coords = _apply_xform(coords, active_xform)
        np.save(outd / f"coords-S{k}.npy", coords)
        manifest["snapshots"].append({"k": k, "n": n, "transform_wall_s": round(wall, 2),
                                      "head_switch": bool(active_k == k), "active_head_k": active_k})
        print(f"armA[{LABEL}] S{k}: n={n:,} transform {wall:.1f}s "
              f"{'(RETRAIN switch, procrustes fit)' if switched else ('(aligned frame)' if active_xform is not None else '(frozen)')}",
              flush=True)
        prev_coords = coords; prev_n = n
        del X
    (outd / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\nwrote {outd}/manifest.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
