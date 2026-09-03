"""MONET diversity draws (owner priority, overseer 2026-09-03). CPU (faiss). Selects a 2M-row draw from the
19.3M light pool under four arms, so the eval can test whether an index-driven diversity draw beats random:
  - random      : uniform 2M.
  - sscd        : shipped-metadata. rarity = (1 - sscd_nn). sscd_nn is a SIMILARITY (HIGH=near-dup, confirmed
                  corr +0.40), so LOW sscd_nn = rare -> upweighted. No index.
  - annfaiss    : self-built EXACT index. IVF-Flat over the pool's CLIP-512 (cosine), k-NN mean distance =
                  inverse density; rarity = mean k-NN distance (large = sparse/rare).
  - theirfaiss  : publisher's IVF-PQ index (nprobe=64) over CLIP-512, same k-NN-distance rarity. PQ-compressed
                  => APPROXIMATE density (recorded as an arm condition — exactly a real customer index).
Selection: weighted sampling WITHOUT replacement via Gumbel-top-k (key = log(rarity) + Gumbel), seeded.
Separate idx files + a manifest (method/params/metric per arm + pairwise overlaps). Resumable rarity cache.
Usage: p_monet_draws.py [N_DRAW=2000000] [SEED=42] [K=16]."""
import json, sys, os, time
from pathlib import Path
import numpy as np

POOL = Path("/data2/monet/pool-20m")
OUT = Path("/data2/monet/draws")
THEIR_CLIP = "/data2/monet/retrieval-storage/clip/embedding_clip-vit-base-patch32.faiss"


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def _gumbel_topk(logw, k, rng):
    g = -np.log(-np.log(rng.random(logw.shape[0]).astype(np.float64) + 1e-12) + 1e-12)
    return np.argpartition(-(logw + g), k)[:k]


def _rarity_annfaiss(clip, k, cache):
    if cache.exists():
        return np.load(cache)
    import faiss
    faiss.omp_set_num_threads(16)
    n, d = clip.shape; B = 200_000
    nlist = 4096
    quant = faiss.IndexFlatIP(d); idx = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
    tr = _norm(np.asarray(clip[np.random.default_rng(0).choice(n, min(500_000, n), replace=False)], np.float32))
    idx.train(tr); del tr
    for s in range(0, n, B):                                   # add normalized batches (no full 39GB copy)
        idx.add(_norm(np.asarray(clip[s:s+B], np.float32)))
    idx.nprobe = 16
    rar = np.empty(n, np.float32)
    for s in range(0, n, B):
        D, _ = idx.search(_norm(np.asarray(clip[s:s+B], np.float32)), k + 1)   # IP sims desc; self first
        rar[s:s+B] = (1.0 - D[:, 1:].mean(1))                 # low mean-sim = rare
        if s % 2_000_000 == 0:
            print(f"  annfaiss {s:,}/{n:,}", flush=True)
    np.save(cache, rar); return rar


def _rarity_theirfaiss(clip, k, cache):
    if cache.exists():
        return np.load(cache)
    import faiss
    faiss.omp_set_num_threads(16)
    idx = faiss.read_index(THEIR_CLIP)
    try:
        idx.nprobe = 64
    except Exception:
        pass
    xq = np.asarray(clip, dtype=np.float32)                    # raw (match their index's build preprocessing)
    n = xq.shape[0]; rar = np.empty(n, np.float32); B = 200_000
    for s in range(0, n, B):
        D, _ = idx.search(xq[s:s+B], k + 1)                   # L2 dist asc; self may or may not be present
        d = np.sort(D, axis=1)[:, 1:k+1]                      # drop nearest (self/dup), keep next k
        rar[s:s+B] = d.mean(1)                                # large mean-dist = rare
        if s % 2_000_000 == 0:
            print(f"  theirfaiss {s:,}/{n:,}", flush=True)
    np.save(cache, rar); return rar


def main():
    n_draw = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    OUT.mkdir(parents=True, exist_ok=True)
    sscd_nn = np.load(POOL / "sscd_nn.npy")
    N = sscd_nn.shape[0]
    clip = np.load(POOL / "clip512.f32.npy", mmap_mode="r")
    print(f"pool N={N:,}, drawing {n_draw:,} per arm (seed {seed}, k={k})", flush=True)

    arms = {}
    arms["random"] = ("uniform", np.ones(N, np.float32))
    arms["sscd"] = ("1-sscd_nn (rare=low similarity)", np.clip(1.0 - sscd_nn, 1e-4, None).astype(np.float32))
    t0 = time.time()
    arms["annfaiss"] = ("IVFFlat cosine kNN mean-dist (exact vectors)", _rarity_annfaiss(clip, k, OUT / "rarity_annfaiss.npy"))
    print(f"annfaiss rarity in {time.time()-t0:.0f}s", flush=True); t0 = time.time()
    arms["theirfaiss"] = ("their IVF-PQ nprobe=64 kNN mean-dist (PQ-approx)", _rarity_theirfaiss(clip, k, OUT / "rarity_theirfaiss.npy"))
    print(f"theirfaiss rarity in {time.time()-t0:.0f}s", flush=True)

    sel = {}
    for arm, (method, rar) in arms.items():
        rng = np.random.default_rng(hash((seed, arm)) % (2**32))
        logw = np.log(np.clip(rar, 1e-8, None).astype(np.float64))
        idx = np.sort(_gumbel_topk(logw, n_draw, rng))
        np.save(OUT / f"{arm}.idx.npy", idx.astype(np.int64)); sel[arm] = idx
        print(f"  {arm}: {len(idx):,} selected  ({method})", flush=True)

    # pairwise overlaps (Jaccard)
    names = list(sel); ov = {}
    for i, a in enumerate(names):
        for b in names[i+1:]:
            inter = np.intersect1d(sel[a], sel[b], assume_unique=True).size
            ov[f"{a}∩{b}"] = {"intersect": int(inter),
                              "jaccard": round(inter / (2 * n_draw - inter), 4)}
    man = {"schema": "monet-draws-2026-09-03", "pool_rows": int(N), "n_draw": n_draw, "seed": seed, "k": k,
           "arms": {a: {"method": m, "idx_file": f"{a}.idx.npy"} for a, (m, _) in arms.items()},
           "sscd_nn_direction": "SIMILARITY (HIGH=near-dup); rarity=1-sscd_nn upweights LOW",
           "theirfaiss_condition": "PQ-compressed index -> approximate density (== a real customer index)",
           "pairwise_overlap": ov}
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1))
    print("pairwise overlaps:", json.dumps(ov), flush=True)
    print(f"DONE -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
