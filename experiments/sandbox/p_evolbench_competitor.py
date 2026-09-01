"""Evolution benchmark — frame-preserving COMPETITORS (5th review #6-7). umap06dev-env (umap-learn 0.6, CPU).

The standing baseline rule requires competitors that try to preserve the frame, not just full-refit ones.
Two modes, both at min_dist=0 to MATCH our md000 champion (#6 — the 0.1 mismatch is removed):
  full_timeline   : refit umap-learn on EACH Sk, procrustes-align to the PREVIOUS snapshot on the shared
                    first-n_{k-1} rows. The honest CPU competitor that actually maps well (unlike cuVS on
                    MiniLM-384) but pays full churn + full recompute every snapshot. Scored kind "B".
  frozen_transform: fit umap-learn on S0 ONCE, then .transform() the arriving tranche each snapshot and
                    APPEND (umap-learn's own parametric-ish frozen analog of our head). Low churn by
                    construction; quality is umap-learn's out-of-sample transform. Scored kind "A" (member=S0).

Emits coords-S{k}.npy + manifest (wall_s per snapshot; frozen_transform also splits fit vs transform wall).
Env: EVOLBENCH_K, MODE (full_timeline|frozen_transform), EVOLBENCH_SUBDIR, MIN_DIST (default 0.0).
Output: evolbench-competitor-umap-<MODE>/.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

SB = Path("/data/latent-basemap/sandbox")
SUBDIR = os.environ.get("EVOLBENCH_SUBDIR", "/data/latent-basemap/substrates/evolbench")
K = int(os.environ.get("EVOLBENCH_K", "5"))
MODE = os.environ.get("MODE", "full_timeline")
MIN_DIST = float(os.environ.get("MIN_DIST", "0.0"))


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def _tranche(k):
    return _norm(np.asarray(np.load(f"{SUBDIR}/T{k}/substrate.f32.npy", mmap_mode="r"), dtype=np.float32))


def _load_Sk(k):
    parts = ["T0"] + [f"T{j}" for j in range(1, k + 1)]
    return _norm(np.concatenate([np.asarray(np.load(f"{SUBDIR}/{t}/substrate.f32.npy", mmap_mode="r"),
                                            dtype=np.float32) for t in parts]))


def _procrustes(src, ref_shared, src_shared):
    mu_s = src_shared.mean(0); mu_r = ref_shared.mean(0)
    A = src_shared - mu_s; B = ref_shared - mu_r
    U, S, Vt = np.linalg.svd(A.T @ B); R = U @ Vt
    scale = S.sum() / max((A ** 2).sum(), 1e-9)
    return ((src - mu_s) @ (scale * R) + mu_r).astype(np.float32)


def main():
    import umap
    outd = SB / f"evolbench-competitor-umap-{MODE}"; outd.mkdir(parents=True, exist_ok=True)
    man = {"schema": f"evolbench-competitor-umap-{MODE}-2026-09-01", "min_dist": MIN_DIST, "snapshots": []}
    mk = lambda: umap.UMAP(n_neighbors=15, min_dist=MIN_DIST, n_components=2, metric="cosine", random_state=None)
    if MODE == "frozen_transform":
        prev = None; prev_n = 0; reducer = None
        for k in range(K + 1):
            Xn = _tranche(k); t0 = time.time()
            if reducer is None:
                reducer = mk(); coords = np.asarray(reducer.fit_transform(Xn), dtype=np.float32); fit_w = time.time() - t0
                man["snapshots"].append({"k": k, "n": int(coords.shape[0]), "wall_s": round(fit_w, 1),
                                         "fit_wall_s": round(fit_w, 1), "transform_wall_s": None})
            else:
                nc = np.asarray(reducer.transform(Xn), dtype=np.float32); tw = time.time() - t0
                coords = np.concatenate([prev, nc])
                man["snapshots"].append({"k": k, "n": int(coords.shape[0]), "wall_s": round(tw, 1),
                                         "transform_wall_s": round(tw, 1)})
            np.save(outd / f"coords-S{k}.npy", coords); prev = coords; prev_n = coords.shape[0]
            print(f"comp[{MODE}] S{k}: n={coords.shape[0]:,} {man['snapshots'][-1]['wall_s']}s", flush=True)
    else:  # full_timeline
        prev = None; prev_n = 0
        for k in range(K + 1):
            X = _load_Sk(k); t0 = time.time()
            coords = np.asarray(mk().fit_transform(X), dtype=np.float32); w = time.time() - t0
            if prev is not None:
                coords = _procrustes(coords, prev, coords[:prev_n])
            np.save(outd / f"coords-S{k}.npy", coords)
            man["snapshots"].append({"k": k, "n": int(coords.shape[0]), "wall_s": round(w, 1)})
            print(f"comp[{MODE}] S{k}: n={coords.shape[0]:,} fit {w:.0f}s", flush=True)
            prev = coords[:coords.shape[0]].copy(); prev_n = coords.shape[0]
    (outd / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"\nwrote {outd}/manifest.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
