#!/usr/bin/env python3
"""Diagnostic: why does armB's cuVS UMAP produce a degenerate map (FFR 0.003, coord blowup to 2e7)?
Fit full 4M S0 with a named config in a FRESH process (pool hygiene), save coords + print coord-range
stats. Score separately with .venv quick_ffr_v2 against the S0 truth. cuml-env. Usage: --config NAME."""
import sys, time
from pathlib import Path
import numpy as np

SUB = "/data/latent-basemap/substrates/evolbench"
OUT = Path("/data/latent-basemap/sandbox/cuvs-probe")


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def load_S0():
    return _norm(np.asarray(np.load(f"{SUB}/T0/substrate.f32.npy", mmap_mode="r"), dtype=np.float32))


CONFIGS = {
    # control: exactly what armB ran
    "control": dict(n_neighbors=15, n_components=2, n_epochs=500, min_dist=0.1,
                    random_state=42, build_algo="nn_descent"),
    # spectral init (prevents disconnected-component fling + global blowup)
    "spectral": dict(n_neighbors=15, n_components=2, n_epochs=500, min_dist=0.1,
                     random_state=42, build_algo="nn_descent", init="spectral"),
    # spectral, drop random_state (cuml's seeded path can be lower quality), fewer epochs
    "spectral_nors": dict(n_neighbors=15, n_components=2, n_epochs=200, min_dist=0.1,
                          build_algo="nn_descent", init="spectral"),
    # richer nn_descent graph (more neighbors during build) + spectral
    "spectral_k30": dict(n_neighbors=30, n_components=2, n_epochs=500, min_dist=0.1,
                         random_state=42, build_algo="nn_descent", init="spectral"),
    # the KNOWN-GOOD ceiling-setter (cuml_reference.py, benchmarks.md): DEFAULT build = exact
    # brute-force kNN (NOT nn_descent), random_state=0. In-core to 12.5M at 21GB per that benchmark.
    "reference": dict(n_neighbors=15, n_components=2, min_dist=0.1, random_state=0),
}


def main():
    cfg_name = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else "control"
    from cuml.manifold import UMAP
    OUT.mkdir(parents=True, exist_ok=True)
    X = load_S0(); n = X.shape[0]
    kw = CONFIGS[cfg_name]
    t0 = time.time()
    coords = np.asarray(UMAP(**kw).fit_transform(X), dtype=np.float32)
    wall = time.time() - t0
    np.save(OUT / f"coords-{cfg_name}.npy", coords)
    r = np.linalg.norm(coords.astype(np.float64) - np.median(coords, 0), axis=1)
    print(f"[{cfg_name}] n={n:,} wall={wall:.0f}s  radius p50={np.percentile(r,50):.1f} "
          f"p99.9={np.percentile(r,99.9):.1f} max={r.max():.3e}  cfg={kw}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
