"""Held-out reception eval for the fast-path full-pool CLIP projection (owner image phase, 2026-09-05). CPU-only
(faiss-CPU + KDTree) — does NOT touch the GPU (3D twin trains concurrently).

Question (plan-monet-clip-demo-cost.md): does the frozen 2M champion head place UNSEEN pool rows into the
existing 2M map such that a held-out row's true high-D neighbors land near it in 2D? Reception recall@15:
for each held-out query, its 15 high-D nearest neighbors AMONG the 2M training rows (exact cosine) vs its 15
nearest training rows in the 2D projection; recall = overlap/15. A MEMBER baseline (in-sample training rows)
contextualizes held-out vs seen.

Existing map (reference): random-2m/clip-substrate.f32.npy (2M x512 high-D, unit-norm) + the 2M champion's
coordinates.npy (2M x2 training layout). Held-out = pool rows whose id is NOT in random-2m (the seed-42 2M
subset); their high-D = pool clip512[heldout], 2D = the full-pool projection coords[heldout].

Usage: heldout_reception_eval.py [NQ=6000] [SEED=0].
"""
import json, sys, time
from pathlib import Path
import numpy as np

POOL = Path("/data2/monet/pool-20m")
RM = Path("/data2/monet/random-2m")
CKPT2M = Path("/data/latent-basemap/sandbox/monet-random-clip-2m/champion-bs16k")
PROJ = Path("/data/latent-basemap/sandbox/monet-clip-fullpool-proj-20260905")
K = 15


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def main():
    nq = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    import faiss
    from scipy.spatial import cKDTree

    # reference (existing 2M map)
    ref_hd = _norm(np.asarray(np.load(RM / "clip-substrate.f32.npy", mmap_mode="r"), np.float32))
    ref_2d = np.asarray(np.load(CKPT2M / "coordinates.npy"), np.float64)
    n_ref = ref_hd.shape[0]
    assert ref_2d.shape[0] == n_ref, f"ref mismatch {ref_2d.shape} vs {n_ref}"
    print(f"[recep] reference (existing map): {n_ref:,} rows", flush=True)

    # random-2m ids (shard order = clip-substrate/coordinates row order)
    metas = sorted((RM / "shards").glob("*_meta.npz"))
    rm_ids = np.concatenate([np.load(m, allow_pickle=True)["id"] for m in metas])
    assert rm_ids.shape[0] == n_ref, f"rm_ids {rm_ids.shape[0]} != ref {n_ref}"
    # pool ids -> held-out positions (id not in random-2m)
    pool_ids = np.load(POOL / "id.npy", allow_pickle=True)
    N = pool_ids.shape[0]
    is_member = np.isin(pool_ids, rm_ids)   # C-level; both are unicode string arrays
    heldout_pos = np.where(~is_member)[0]
    member_pos = np.where(is_member)[0]
    print(f"[recep] pool {N:,}: members {member_pos.size:,} (random-2m subset), held-out {heldout_pos.size:,} "
          f"({heldout_pos.size/N*100:.1f}%)", flush=True)

    coords = np.load(PROJ / "coords.f32.npy", mmap_mode="r")
    pool_clip = np.load(POOL / "clip512.f32.npy", mmap_mode="r")

    # faiss exact IP over the 2M reference (unit vectors -> cosine); KDTree over 2M 2D layout
    t0 = time.time()
    index = faiss.IndexFlatIP(ref_hd.shape[1]); index.add(ref_hd)
    tree = cKDTree(ref_2d)
    print(f"[recep] index+tree built {time.time()-t0:.0f}s", flush=True)

    def recall_at_k(positions, exclude_self_in_ref):
        rng = np.random.default_rng(seed)
        q = rng.choice(positions, size=min(nq, positions.size), replace=False)
        q_hd = _norm(np.asarray(pool_clip[np.sort(q)], np.float32))  # sorted for memmap locality
        qs = np.sort(q)
        q_2d = np.asarray(coords[qs], np.float64)
        # high-D truth among reference (k+1 to drop self when member)
        kk = K + (1 if exclude_self_in_ref else 0)
        _, hd = index.search(q_hd, kk)
        # 2D neighbors among reference
        _, d2 = tree.query(q_2d, k=kk, workers=-1)
        rec = np.empty(qs.size)
        for i in range(qs.size):
            h = hd[i]; t = d2[i]
            if exclude_self_in_ref:
                # member query row exists in ref; its own ref index is the top hd hit — drop the nearest-identical
                h = h[h >= 0][:K + 1]
                t = t[:K + 1]
                # drop the self match (cos~1 / dist~0): remove the single closest in each
                h = h[1:K + 1] if h.size > K else h[:K]
                t = t[1:K + 1] if t.size > K else t[:K]
            else:
                h = h[:K]; t = t[:K]
            rec[i] = len(set(h.tolist()) & set(t.tolist())) / K
        return float(rec.mean()), int(qs.size)

    ho_rec, ho_n = recall_at_k(heldout_pos, exclude_self_in_ref=False)
    mem_rec, mem_n = recall_at_k(member_pos, exclude_self_in_ref=True)
    out = {"schema": "monet-clip-fullpool-heldout-reception-2026-09-05", "k": K, "nq": nq, "seed": seed,
           "n_reference": int(n_ref), "n_pool": int(N),
           "n_members": int(member_pos.size), "n_heldout": int(heldout_pos.size),
           "heldout_reception_recall@15": round(ho_rec, 4), "heldout_nq": ho_n,
           "member_reception_recall@15": round(mem_rec, 4), "member_nq": mem_n,
           "note": "reception = fraction of a query's high-D 15NN (among the 2M existing map) that are also its "
                   "2D 15NN in the frozen-head projection. Held-out rows never trained; members are the 2M draw."}
    (PROJ / "heldout-reception.json").write_text(json.dumps(out, indent=1))
    print(f"[recep] held-out recall@15 {ho_rec:.4f} (n={ho_n}) | member {mem_rec:.4f} (n={mem_n}) "
          f"-> {PROJ/'heldout-reception.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
