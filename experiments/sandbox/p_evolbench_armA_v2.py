"""Evolution benchmark — ARM A v2, APPEND-ONLY service model (5th review #5). GPU (.venv/torch).

The product operation is: a tranche ARRIVES, we place ONLY the new rows through the frozen head and
APPEND them to the persisted map — the cumulative corpus is NEVER re-projected between retrains (old
points are bitwise-identical, churn 0 by construction). Only a retrain (drift trigger) re-projects all
rows with the new head + procrustes-aligns to the previous frame.

Two latencies are reported at every snapshot, and BOTH at a retrain trigger:
  place_wall_s  = immediate frozen placement of the arriving tranche (what the user sees instantly),
                  incl. load+normalize+transform+append+write of the new rows only.
  update_wall_s = time-to-updated-map at a retrain: full re-transform of the cumulative corpus with the
                  new head + procrustes alignment + write (None when no retrain that snapshot).
Head GPU-hours (the offline train cost) are carried by the scorer via HEAD_GPU_H, not here.

Schedule via SCHEDULE_JSON {k: head_path} (frozen {"0":""}; drift-triggered {"0":"","3":S3head}). Substrate
dir + tranche layout via EVOLBENCH_SUBDIR (default evolbench); arriving tranche at snapshot k = substrate T{k}.
Env: EVOLBENCH_K, SCHEDULE_JSON, ARM_LABEL, EVOLBENCH_SUBDIR. Output evolbench-armA-<label>-v2/coords-S{k}.npy
+ manifest (place_wall_s/update_wall_s/head_switch/active_head_k per snapshot)."""
import json, os, sys, time
from pathlib import Path
import numpy as np

SB = Path("/data/latent-basemap/sandbox")
SUBDIR = os.environ.get("EVOLBENCH_SUBDIR", "/data/latent-basemap/substrates/evolbench")
K = int(os.environ.get("EVOLBENCH_K", "5"))
LABEL = os.environ.get("ARM_LABEL", "frozen")
SCHEDULE = {int(k): v for k, v in json.loads(os.environ.get("SCHEDULE_JSON", '{"0": ""}')).items()}


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def _tranche(k):
    """The rows ARRIVING at snapshot k = substrate T{k} (T0 at k=0)."""
    return _norm(np.asarray(np.load(f"{SUBDIR}/T{k}/substrate.f32.npy", mmap_mode="r"), dtype=np.float32))


def _load_Sk(k):
    parts = ["T0"] + [f"T{j}" for j in range(1, k + 1)]
    return _norm(np.concatenate([np.asarray(np.load(f"{SUBDIR}/{t}/substrate.f32.npy", mmap_mode="r"),
                                            dtype=np.float32) for t in parts]))


def _fit_procrustes(src_shared, ref_shared):
    mu_s = src_shared.mean(0); mu_r = ref_shared.mean(0)
    A = src_shared - mu_s; B = ref_shared - mu_r
    U, S, Vt = np.linalg.svd(A.T @ B); R = U @ Vt
    scale = S.sum() / max((A ** 2).sum(), 1e-9)
    return mu_s, (scale * R).astype(np.float32), mu_r


def _apply(src, xf):
    mu_s, sR, mu_r = xf
    return ((src - mu_s) @ sR + mu_r).astype(np.float32)


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _paths import ensure_paths; ensure_paths()
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    if not SCHEDULE.get(0):
        SCHEDULE[0] = str(SB / "evolbench-S0/champion-bs16k/model.pt")
    outd = SB / f"evolbench-armA-{LABEL}-v2"; outd.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "evolbench-armA-v2-2026-09-01", "label": LABEL,
                "schedule": {str(k): v for k, v in SCHEDULE.items()}, "snapshots": []}
    active = None; active_k = None; active_xform = None
    persisted = None                      # coords for rows already placed, in the active frame
    for k in range(K + 1):
        switch = k in SCHEDULE
        # ---- immediate frozen placement of the ARRIVING tranche with the CURRENT (old) head ----
        Xn = _tranche(k); place = None
        if active is not None:            # k>=1: place new rows through the still-frozen old head
            t0 = time.time()
            nc = np.asarray(active.transform(Xn, batch_size=8192), dtype=np.float32)
            if active_xform is not None:
                nc = _apply(nc, active_xform)
            immediate = np.concatenate([persisted, nc])
            place = time.time() - t0
        else:                             # k==0: first head places T0
            active = ParametricUMAP.load(SCHEDULE[0], device="cuda"); active_k = 0
            t0 = time.time()
            immediate = np.asarray(active.transform(Xn, batch_size=8192), dtype=np.float32)
            place = time.time() - t0
        update = None
        # ---- retrain (drift trigger): re-project the cumulative corpus with the NEW head + align ----
        if switch and k != 0:
            t1 = time.time()
            active = ParametricUMAP.load(SCHEDULE[k], device="cuda"); active_k = k; active_xform = None
            X_all = _load_Sk(k)
            full = np.asarray(active.transform(X_all, batch_size=8192), dtype=np.float32)
            # align the new head's frame to the previous (persisted+immediate) frame on the SHARED rows
            active_xform = _fit_procrustes(full[:persisted.shape[0]], persisted) if persisted is not None else None
            coords = _apply(full, active_xform) if active_xform is not None else full
            update = time.time() - t1
            del X_all
        else:
            coords = immediate
        np.save(outd / f"coords-S{k}.npy", coords)
        persisted = coords
        manifest["snapshots"].append({"k": k, "n": int(coords.shape[0]),
                                      "place_wall_s": round(place, 2) if place is not None else None,
                                      "update_wall_s": round(update, 2) if update is not None else None,
                                      "head_switch": bool(k in SCHEDULE),
                                      "active_head_k": active_k})
        print(f"armA-v2[{LABEL}] S{k}: n={coords.shape[0]:,} place {place:.1f}s "
              f"{('UPDATE(retrain) '+format(update,'.1f')+'s' if update else 'append-only')}", flush=True)
    (outd / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\nwrote {outd}/manifest.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
