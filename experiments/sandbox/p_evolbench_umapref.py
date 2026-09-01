#!/usr/bin/env python3
"""Evolution benchmark — umap-learn 0.6 CPU quality REFERENCE (overseer 2026-09-01, option c).
Two anchor points only (S0, S5), not a full arm — anchors the quality axis for the MiniLM readout as the
working-approximate ceiling (umap-learn 0.6 = the substrate's ~0.45 FFR reference). CPU, runs in parallel
with the GPU arm B. Usage: p_evolbench_umapref.py <k>  -> evolbench-umapref/coords-S{k}.npy + wall."""
import os, sys, time
from pathlib import Path
import numpy as np

SUBDIR = os.environ.get("EVOLBENCH_SUBDIR", "/data/latent-basemap/substrates/evolbench")
OUTD = Path(os.environ.get("EVOLBENCH_UMAPREF_DIR", "/data/latent-basemap/sandbox/evolbench-umapref"))


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def _load_Sk(k):
    parts = ["T0"] + [f"T{j}" for j in range(1, k + 1)]
    return _norm(np.concatenate([
        np.asarray(np.load(f"{SUBDIR}/{t}/substrate.f32.npy", mmap_mode="r"), dtype=np.float32)
        for t in parts]))


def main():
    k = int(sys.argv[1]); OUTD.mkdir(parents=True, exist_ok=True)
    import umap
    X = _load_Sk(k); n = X.shape[0]
    t0 = time.time()
    coords = np.asarray(umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric="cosine",
                                  random_state=None).fit_transform(X), dtype=np.float32)
    wall = time.time() - t0
    np.save(OUTD / f"coords-S{k}.npy", coords)
    print(f"umapref S{k}: n={n:,} wall={wall/60:.1f}min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
