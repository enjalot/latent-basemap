"""theirfaiss rarity via GPU search of THEIR SAME IVF-PQ index (owner MONET option A, overseer 2026-09-04).
faiss-gpu-cu12 env. NOT a substitution — identical index bytes + nprobe=64 + PQ codes; only the search HARDWARE
is GPU. Modes: 'test' = load their CLIP index -> index_cpu_to_gpu -> tiny search, verifying GpuIndexIVFPQ
supports m=64/d=512 on sm_120 (go/no-go for A). 'full' = search the 19.3M pool -> rarity_theirfaiss_gpu.npy
(mean kNN dist). 'sample <n>' = search a fixed sample (for the cross-check). Usage: p_theirfaiss_gpu.py <mode>."""
import sys, time, json
from pathlib import Path
import numpy as np

THEIR = "/data2/monet/retrieval-storage/clip/embedding_clip-vit-base-patch32.faiss"
POOL = Path("/data2/monet/pool-20m"); OUT = Path("/data2/monet/draws"); K = 16; SEED = 42


def _gpu_index():
    import faiss
    assert faiss.get_num_gpus() > 0, "no GPU visible to faiss"
    idx = faiss.read_index(THEIR)
    res = faiss.StandardGpuResources()
    co = faiss.GpuClonerOptions()
    try:
        co.useFloat16 = True                 # PQ lookup tables fp16 (memory); minor accum-order diff (gated)
    except Exception:
        pass
    g = faiss.index_cpu_to_gpu(res, 0, idx, co); g.nprobe = 64
    return g


def _rarity(g, X):
    n = X.shape[0]; rar = np.empty(n, np.float32); B = 500_000
    for s in range(0, n, B):
        q = np.ascontiguousarray(X[s:s+B], dtype=np.float32)
        D, _ = g.search(q, K + 1); D.sort(axis=1); rar[s:s+B] = D[:, 1:K+1].mean(1)
    return rar


def main():
    mode = sys.argv[1]
    clip = np.load(POOL / "clip512.f32.npy", mmap_mode="r"); N = clip.shape[0]
    g = _gpu_index()
    if mode == "test":
        t0 = time.time(); D, _ = g.search(np.ascontiguousarray(clip[:1000], np.float32), K + 1)
        print(f"TEST OK: GpuIndexIVFPQ m=64 d=512 search {D.shape} in {time.time()-t0:.2f}s (sm_120 supported)", flush=True)
        return 0
    if mode == "sample":
        ns = int(sys.argv[2]); samp = np.sort(np.random.default_rng(SEED).choice(N, ns, replace=False))
        np.save(OUT / "theirfaiss_xcheck_idx.npy", samp)
        np.save(OUT / "rarity_theirfaiss_gpu_sample.npy", _rarity(g, np.ascontiguousarray(clip[samp])))
        print(f"sample rarity {ns} saved", flush=True); return 0
    # full
    t0 = time.time(); rar = _rarity(g, clip)
    np.save(OUT / "rarity_theirfaiss_gpu.npy", rar)
    print(f"FULL theirfaiss GPU rarity {N:,} in {time.time()-t0:.0f}s -> rarity_theirfaiss_gpu.npy", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
