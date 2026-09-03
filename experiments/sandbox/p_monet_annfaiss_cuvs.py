"""annfaiss draw-arm rarity via cuVS-GPU (owner MONET, overseer 2026-09-03). cuml-env. The arm = "self-built
index over the pool's CLIP-512"; implementation swapped CPU-IVF-Flat -> cuVS/cuml GPU kNN (bandwidth-bound on
CPU, minutes on GPU). Cosine via L2 on normalized fp16. Writes rarity_annfaiss.npy (mean kNN dist over the
full 19.3M pool, large=rare) + rarity_cuvs_sample.npy (a seeded 500K sample, for the CPU cross-check). Method
annotated in annfaiss-method.json. Usage: p_monet_annfaiss_cuvs.py [K=16] [SAMPLE=500000]."""
import sys, json, time
from pathlib import Path
import numpy as np

POOL = Path("/data2/monet/pool-20m"); OUT = Path("/data2/monet/draws"); OUT.mkdir(parents=True, exist_ok=True)
SEED = 42


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    nsample = int(sys.argv[2]) if len(sys.argv) > 2 else 500_000
    import cupy as cp
    from cuml.neighbors import NearestNeighbors
    clip = np.load(POOL / "clip512.f32.npy", mmap_mode="r"); N, d = clip.shape
    print(f"cuVS annfaiss: N={N:,} d={d}", flush=True)
    # load + L2-normalize on GPU as fp16 (19.3M x 512 x2 ~ 19.7GB)
    X = cp.asarray(np.asarray(clip, dtype=np.float32))
    X /= cp.linalg.norm(X, axis=1, keepdims=True).clip(1e-9)
    X = X.astype(cp.float16)
    t0 = time.time()
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="ivfflat")
    nn.fit(X)
    rar = np.empty(N, np.float32); B = 500_000
    for s in range(0, N, B):
        dist, _ = nn.kneighbors(X[s:s+B])           # euclidean on normalized; [:,0]=self
        rar[s:s+B] = cp.asnumpy(dist[:, 1:].mean(1))
        if s % 5_000_000 == 0:
            print(f"  {s:,}/{N:,} ({time.time()-t0:.0f}s)", flush=True)
    np.save(OUT / "rarity_annfaiss.npy", rar)
    # seeded sample for the cross-check
    samp = np.sort(np.random.default_rng(SEED).choice(N, nsample, replace=False))
    np.save(OUT / "annfaiss_sample_idx.npy", samp)
    np.save(OUT / "rarity_cuvs_sample.npy", rar[samp])
    (OUT / "annfaiss-method.json").write_text(json.dumps({
        "arm": "annfaiss", "impl": "cuVS/cuml NearestNeighbors ivfflat GPU", "metric": "euclidean-on-L2norm(cosine)",
        "dtype": "fp16", "k": k, "n": int(N), "wall_s": round(time.time() - t0, 1),
        "note": "swapped from CPU faiss IVF-Flat (bandwidth-bound); cross-check Spearman >=0.95 vs CPU on a "
                f"{nsample} sample gates the draw. Arm identity (self-built index over the pool) preserved."}, indent=1))
    print(f"cuVS annfaiss DONE: {N:,} rarity in {time.time()-t0:.0f}s -> rarity_annfaiss.npy", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
