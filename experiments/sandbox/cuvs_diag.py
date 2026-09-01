#!/usr/bin/env python3
"""cuVS/cuml diagnostic matrix (graph-vs-layout, dimension axis, plumbing). cuml-env, GPU.
Modes (per substrate sample, N rows, dim D):
  prep   SUBSTRATE D N OUTDIR   -> save Xnorm.f32.npy (L2-normed sample) + exact_k15.npy (cuml brute cosine)
  recall OUTDIR                 -> cuVS nn_descent graph recall vs exact_k15 (set-intersection)
  umap   OUTDIR TAG KW_JSON     -> cuml UMAP with kwargs KW_JSON -> coords-<TAG>.npy; prints n_epochs_/flags
Truth + coords are saved to disk; FFR is scored uniformly afterward by the .venv scorer for one metric."""
import json, sys, time
from pathlib import Path
import numpy as np


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def prep(substrate, D, N, outd):
    outd = Path(outd); outd.mkdir(parents=True, exist_ok=True)
    raw = np.load(substrate, mmap_mode="r")
    N = min(int(N), raw.shape[0])
    X = _norm(np.asarray(raw[:N], dtype=np.float32))
    assert X.shape[1] == int(D), f"dim {X.shape[1]} != {D}"
    np.save(outd / "Xnorm.f32.npy", X)
    from cuml.neighbors import NearestNeighbors
    t0 = time.time()
    nn = NearestNeighbors(n_neighbors=16, algorithm="brute", metric="cosine").fit(X)
    _, idx = nn.kneighbors(X)
    idx = np.asarray(idx)[:, 1:16].astype(np.int32)   # drop self, keep 15
    np.save(outd / "exact_k15.npy", idx)
    print(f"prep {outd.name}: N={N:,} D={X.shape[1]} contig={X.flags['C_CONTIGUOUS']} "
          f"dtype={X.dtype} range=[{X.min():.3f},{X.max():.3f}] exact-knn {time.time()-t0:.1f}s", flush=True)
    return 0


def recall(outd):
    outd = Path(outd)
    X = np.load(outd / "Xnorm.f32.npy")
    exact = np.load(outd / "exact_k15.npy")
    from cuvs.neighbors import nn_descent
    import cupy as cp
    Xd = cp.asarray(X)
    idx_map = {}
    for degree in (32, 64, 128):
        try:
            params = nn_descent.IndexParams(graph_degree=degree, metric="sqeuclidean")
            t0 = time.time()
            g = nn_descent.build(params, Xd)
            gi = cp.asarray(g.graph).get()[:, :15].astype(np.int32)
        except Exception as e:
            print(f"recall degree={degree}: BUILD ERR {e}", flush=True); continue
        # per-row overlap of the 15 nn_descent neighbors vs the 15 exact
        ov = np.array([len(set(gi[i]).intersection(exact[i])) for i in range(0, len(gi), max(1, len(gi)//20000))])
        print(f"recall degree={degree}: {ov.mean()/15:.3f}  build {time.time()-t0:.1f}s", flush=True)
    return 0


def umap(outd, tag, kw_json):
    outd = Path(outd); X = np.load(outd / "Xnorm.f32.npy")
    kw = json.loads(kw_json)
    from cuml.manifold import UMAP
    print(f"umap[{tag}] input: contig={X.flags['C_CONTIGUOUS']} dtype={X.dtype} shape={X.shape}", flush=True)
    t0 = time.time()
    reducer = UMAP(**kw)
    coords = np.asarray(reducer.fit_transform(X), dtype=np.float32)
    wall = time.time() - t0
    np.save(outd / f"coords-{tag}.npy", coords)
    ne = getattr(reducer, "n_epochs_", getattr(reducer, "n_epochs", "?"))
    r = np.linalg.norm(coords.astype(np.float64) - np.median(coords, 0), axis=1)
    print(f"umap[{tag}]: wall={wall:.1f}s n_epochs_={ne} kw={kw} "
          f"radius p50={np.percentile(r,50):.1f} max={r.max():.2e}", flush=True)
    return 0


def main():
    mode = sys.argv[1]
    if mode == "prep":   return prep(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    if mode == "recall": return recall(sys.argv[2])
    if mode == "umap":   return umap(sys.argv[2], sys.argv[3], sys.argv[4])
    raise SystemExit(f"bad mode {mode}")


if __name__ == "__main__":
    raise SystemExit(main())
