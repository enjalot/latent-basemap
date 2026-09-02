"""D768 arm-B spot-check: cuml EXACT-build UMAP on d768-S0 (2M×768), to validate the nn_descent S0 map at
PRODUCTION scale in the actual benchmark (not just the 500K recall diagnostic). cuml-env, GPU. Same config
as arm-B (min_dist=0, spectral, 500ep) but EXACT graph (random_state=0 -> brute_force). Saves coords for the
.venv scorer to compare nnd-S0 vs exact-S0 FFR (acceptance |Δ| <= 0.02)."""
import time
from pathlib import Path
import numpy as np

SUB = "/data/latent-basemap/substrates/evolbench-d768/T0/substrate.f32.npy"
OUT = Path("/data/latent-basemap/sandbox/evolbench-d768-armB/coords-exact-S0.npy")


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def main():
    from cuml.manifold import UMAP
    X = _norm(np.asarray(np.load(SUB, mmap_mode="r"), dtype=np.float32))
    t0 = time.time()
    coords = np.asarray(UMAP(n_neighbors=15, n_components=2, min_dist=0.0, n_epochs=500,
                             random_state=0, init="spectral").fit_transform(X), dtype=np.float32)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT, coords)
    print(f"d768 exact-S0: n={X.shape[0]:,} wall={time.time()-t0:.0f}s -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
