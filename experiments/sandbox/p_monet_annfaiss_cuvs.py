"""annfaiss draw-arm rarity via cuVS-GPU CAGRA (owner MONET, overseer 2026-09-03; OOM-safe rewrite). cuml-env.
NEVER materializes the pool as f32 on GPU (39.6GB > card). Chunked-loads CLIP-512, L2-normalizes, quantizes
to INT8 (19.3M×512 = 9.9GB — fits with margin, respects the >=2GB-shard OOM rule; full per-vector codes, NOT
PQ, so the arm stays a self-built exact-ish index, distinct from theirfaiss's PQ). CAGRA int8 kNN -> rarity =
mean kNN distance (large=rare). fp/int8 precision is fine for a RANK-based density (Spearman gate confirms).
Writes rarity_annfaiss.npy + a seeded 500K sample. Usage: p_monet_annfaiss_cuvs.py [K=16] [SAMPLE=500000]."""
import sys, json, time
from pathlib import Path
import numpy as np

POOL = Path("/data2/monet/pool-20m"); OUT = Path("/data2/monet/draws"); OUT.mkdir(parents=True, exist_ok=True)
SEED = 42


def _load_int8_chunked(clip, B=1_000_000):
    N, d = clip.shape
    q = np.empty((N, d), np.int8)                       # 9.9GB (int8), no 39.6GB f32 spike
    for s in range(0, N, B):
        c = np.array(clip[s:s+B], dtype=np.float32)  # copy: asarray on a ro-memmap gives a ro-view
        c /= np.linalg.norm(c, axis=1, keepdims=True).clip(1e-9)
        q[s:s+B] = np.clip(np.round(c * 127.0), -127, 127).astype(np.int8)
    return q


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    nsample = int(sys.argv[2]) if len(sys.argv) > 2 else 500_000
    import cupy as cp
    from cuvs.neighbors import cagra
    clip = np.load(POOL / "clip512.f32.npy", mmap_mode="r"); N, d = clip.shape
    print(f"cuVS CAGRA int8 annfaiss: N={N:,} d={d}", flush=True)
    q8 = _load_int8_chunked(clip)                        # CPU int8 9.9GB
    Xg = cp.asarray(q8)                                  # GPU int8 9.9GB
    t0 = time.time()
    idx = cagra.build(cagra.IndexParams(graph_degree=64, intermediate_graph_degree=96), Xg)
    print(f"CAGRA built in {time.time()-t0:.0f}s", flush=True)
    rar = np.empty(N, np.float32); B = 1_000_000
    sp = cagra.SearchParams(itopk_size=128)
    for s in range(0, N, B):
        D, _ = cagra.search(sp, idx, Xg[s:s+B], k + 1)
        Dn = cp.asnumpy(D).astype(np.float32)
        Dn.sort(axis=1)
        rar[s:s+B] = Dn[:, 1:k+1].mean(1)                # drop self (nearest); mean of next k
        if s % 5_000_000 == 0:
            print(f"  {s:,}/{N:,} ({time.time()-t0:.0f}s)", flush=True)
    np.save(OUT / "rarity_annfaiss.npy", rar)
    samp = np.sort(np.random.default_rng(SEED).choice(N, nsample, replace=False))
    np.save(OUT / "annfaiss_sample_idx.npy", samp)
    np.save(OUT / "rarity_cuvs_sample.npy", rar[samp])
    (OUT / "annfaiss-method.json").write_text(json.dumps({
        "arm": "annfaiss", "impl": "cuVS CAGRA int8 GPU (graph_degree=64)", "metric": "L2-on-int8(L2norm*127)",
        "k": k, "n": int(N), "wall_s": round(time.time()-t0, 1),
        "note": "swapped from CPU faiss IVF-Flat (bandwidth-bound); int8 quantization of L2-normalized CLIP "
                "(full per-vector codes, not PQ) -> rank-based density; Spearman>=0.95 vs CPU-fp32-exact on a "
                f"{nsample} sample gates the draw."}, indent=1))
    print(f"cuVS CAGRA annfaiss DONE: {N:,} in {time.time()-t0:.0f}s -> rarity_annfaiss.npy", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
