"""Three-way full-corpus reception (owner item b; 2026-09-05). CPU (faiss/KDTree). Measures reception recall@15
into a head's TRAINING map, stratified by how OOD each row is:
  member        = random-2m rows (in-sample; the head trained on these)
  pool-heldout  = pool rows NOT in random-2m (unseen by training, but in the pool draw)
  complement    = the 84.47M complement rows (DOUBLY unseen: never trained, never in the pool draw)
Shows the generalization gradient at 104M scale. Reception = fraction of a query's high-D 15NN (among the
training set) within its 0.1%-of-training 2D disc in the frozen-head full-corpus projection.

Head-parameterized: REF_HD (training high-D), REF_COORDS (training 2D layout), PROJ (full-corpus coords dir),
HEAD_TAG. Full-corpus row space: [0,19344847)=pool (high-D pool-20m/clip512), [19344847,N)=complement
(high-D pool-complement-88m/clip512). Usage: fullcorpus_reception.py [NQ=6000] [SEED=0].
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

POOL = Path("/data2/monet/pool-20m"); RM = Path("/data2/monet/random-2m")
COMP = Path("/data2/monet/pool-complement-88m")
REF_HD = Path(os.environ["REF_HD"]); REF_COORDS = Path(os.environ["REF_COORDS"])
PROJ = Path(os.environ["PROJ"]); TAG = os.environ["HEAD_TAG"]
N_POOL = 19_344_847
K = 15


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def main():
    nq = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    import faiss
    from scipy.spatial import cKDTree

    ref_hd = _norm(np.asarray(np.load(REF_HD, mmap_mode="r"), np.float32))
    ref_2d = np.asarray(np.load(REF_COORDS), np.float64)
    index = faiss.IndexFlatIP(512); index.add(ref_hd)
    tree = cKDTree(ref_2d); disc = max(int(round(ref_hd.shape[0] * 0.001)), K)
    coords = np.load(PROJ / "coords.f32.npy", mmap_mode="r"); N = coords.shape[0]
    pool_clip = np.load(POOL / "clip512.f32.npy", mmap_mode="r")
    comp_clip = np.load(COMP / "clip512.f32.npy", mmap_mode="r")

    # membership within the pool
    pool_ids = np.load(POOL / "id.npy", allow_pickle=True)
    rm_ids = np.concatenate([np.load(m, allow_pickle=True)["id"] for m in sorted((RM / "shards").glob("*_meta.npz"))])
    is_member = np.isin(pool_ids, rm_ids)
    member_pos = np.where(is_member)[0]; poolheld_pos = np.where(~is_member)[0]
    print(f"[{TAG}] member {member_pos.size:,} pool-heldout {poolheld_pos.size:,} complement {N-N_POOL:,}", flush=True)

    rng = np.random.default_rng(seed)

    def reception(rows, hd_getter, exclude_self):
        q = np.sort(rng.choice(rows, min(nq, len(rows)), replace=False))
        q_hd = _norm(hd_getter(q)); q_2d = np.asarray(coords[q], np.float64)
        _, hd = index.search(q_hd, K + (1 if exclude_self else 0))
        _, d2 = tree.query(q_2d, k=disc + (1 if exclude_self else 0), workers=-1)
        rec = np.empty(q.size)
        for i in range(q.size):
            h = hd[i]; t = d2[i]
            if exclude_self:
                h = h[1:]; t = t[1:]
            ds = set(int(x) for x in t[:disc]); rec[i] = sum(int(x) in ds for x in h[:K]) / K
        return round(float(rec.mean()), 4), int(q.size)

    t0 = time.time()
    m_rec, m_n = reception(member_pos, lambda q: np.asarray(pool_clip[q], np.float32), True)
    p_rec, p_n = reception(poolheld_pos, lambda q: np.asarray(pool_clip[q], np.float32), False)
    c_rec, c_n = reception(np.arange(N_POOL, N), lambda q: np.asarray(comp_clip[q - N_POOL], np.float32), False)

    out = {"schema": "fullcorpus-reception-3way-2026-09-05", "head": TAG, "k": K, "n_full": int(N),
           "member": {"reception": m_rec, "nq": m_n},
           "pool_heldout": {"reception": p_rec, "nq": p_n},
           "complement_doubly_unseen": {"reception": c_rec, "nq": c_n},
           "gradient": {"member_minus_poolheld": round(m_rec - p_rec, 4),
                        "poolheld_minus_complement": round(p_rec - c_rec, 4),
                        "member_minus_complement": round(m_rec - c_rec, 4)},
           "wall_s": round(time.time() - t0, 1),
           "note": "reception into the head's 2M training map; strata = increasing OOD. complement rows are "
                   "doubly unseen (never trained, never in the pool draw)."}
    (PROJ / "reception-3way.json").write_text(json.dumps(out, indent=1))
    print(f"[{TAG}] member {m_rec} | pool-heldout {p_rec} | complement {c_rec} "
          f"(gradient {m_rec-c_rec:+.4f} member->complement)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
