#!/usr/bin/env python3
"""umap-learn 0.6 reference map for the cuVS diagnostic (the working-approximate ceiling). umap06dev-env.
Usage: OUTDIR  -> reads Xnorm.f32.npy, writes coords-umap06.npy (pynndescent graph + umap-learn layout)."""
import sys, time
from pathlib import Path
import numpy as np


def main():
    outd = Path(sys.argv[1]); X = np.load(outd / "Xnorm.f32.npy")
    import umap
    t0 = time.time()
    coords = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric="cosine",
                       random_state=None).fit_transform(X)
    coords = np.asarray(coords, dtype=np.float32)
    np.save(outd / "coords-umap06.npy", coords)
    r = np.linalg.norm(coords.astype(np.float64) - np.median(coords, 0), axis=1)
    print(f"umap06[{outd.name}]: wall={time.time()-t0:.1f}s radius p50={np.percentile(r,50):.2f} "
          f"max={r.max():.2e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
