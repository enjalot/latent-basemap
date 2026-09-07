"""Three-way reception stamp for a 6M full-corpus DINO projection (owner 2026-09-07). CPU (faiss + a bit of numpy),
OFF-FLOCK. Stamps fullcorpus-dino-6m-{2d,pca768-2d,pca768-3d}/reception-3way.json — the projections were made by
another agent (2D) / this session (3d); this session STAMPS them.

reception recall@15 = fraction of a query's high-D 15NN (among the 6M REFERENCE = the trained-on rows) that are
also its 2D neighbors within the 0.1%-disc of the reference layout. Three cohorts by 6M-draw membership (full_pos):
  member                = full-corpus rows IN the 6M draw
  pool_heldout          = pool rows [0,19344847) NOT in the 6M draw
  complement_doubly_unseen = complement rows NOT in the 6M draw
High-D column = the DINO-1536 full column (pool+complement); for the PCA heads it is PCA-768-transformed (saved
components+mean) to match the reference space. The query's 2D = the full-corpus projection coords.

Env: PROJ=<fullcorpus dir> REF_HD=<6M substrate> REF_COORDS=<6M champion coords> [PCA_MODEL=<pca768-model.npz>].
Usage: fullcorpus_reception_6m.py [NQ=4000]
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

POOL_D = "/data2/monet/pool-20m/dino1536.f16.npy"; COMP_D = "/data2/monet/pool-complement-88m/dino1536.f16.npy"
FULL_POS = "/data2/monet/random-dino-6m/full_pos.npy"; N_POOL = 19_344_847; K = 15
PROJ = Path(os.environ["PROJ"]); REF_HD = os.environ["REF_HD"]; REF_COORDS = os.environ["REF_COORDS"]
PCA_MODEL = os.environ.get("PCA_MODEL")


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def main():
    nq = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    import faiss
    pool = np.load(POOL_D, mmap_mode="r"); comp = np.load(COMP_D, mmap_mode="r")
    coords_full = np.load(PROJ / "coords.f32.npy", mmap_mode="r")            # 103.8M x ncomp (this projection)
    # OOM FIX (giant-shard rule): the 6M substrate is L2=1.0 (prenormalized) -> SKIP _norm; keep it as a MMAP and
    # feed faiss index.add in 1M-row chunks (per-chunk f16->f32 cast) — never materialize the whole f32 array.
    ref_mm = np.load(REF_HD, mmap_mode="r"); nref = ref_mm.shape[0]
    ref_2d = np.asarray(np.load(REF_COORDS), np.float32)
    pm = np.load(PCA_MODEL) if PCA_MODEL else None
    comp_p = pm["components"].astype(np.float32) if pm is not None else None; mean_p = pm["mean"].astype(np.float32) if pm is not None else None

    t0 = time.time(); hdx = faiss.IndexFlatIP(ref_mm.shape[1])
    for i in range(0, nref, 1_000_000):
        hdx.add(np.ascontiguousarray(np.asarray(ref_mm[i:i + 1_000_000], np.float32)));
    print(f"[recep6m {PROJ.name}] faiss index {hdx.ntotal:,} added (chunked, no whole-array materialize)", flush=True)
    d2x = faiss.IndexFlatL2(ref_2d.shape[1]); d2x.add(np.ascontiguousarray(ref_2d)); disc = max(int(round(nref * 0.001)), K)
    print(f"[recep6m {PROJ.name}] ref {nref:,} dim {ref_mm.shape[1]} | faiss {time.time()-t0:.0f}s disc {disc}", flush=True)

    fp = np.load(FULL_POS); is_member = np.zeros(N_POOL + comp.shape[0], bool); is_member[fp] = True

    def q_highd(fpos):                                                       # fpos: full-corpus positions -> high-D (PCA if pca head)
        ispool = fpos < N_POOL
        out = np.empty((fpos.shape[0], pool.shape[1]), np.float32)
        if ispool.any(): out[ispool] = np.asarray(pool[fpos[ispool]], np.float32)
        if (~ispool).any(): out[~ispool] = np.asarray(comp[fpos[~ispool] - N_POOL], np.float32)
        out = _norm(out)
        if pm is not None: out = _norm((out - mean_p) @ comp_p)
        return out

    def reception(positions, exclude_self):
        rng = np.random.default_rng(0); q = np.sort(rng.choice(positions, min(nq, positions.size), replace=False))
        qhd = q_highd(q); q2d = np.ascontiguousarray(coords_full[q].astype(np.float32))
        _, hd = hdx.search(qhd, K + (1 if exclude_self else 0))
        _, dd = d2x.search(q2d, disc + (1 if exclude_self else 0))
        rec = np.empty(q.size)
        for i in range(q.size):
            h = hd[i][1:][:K] if exclude_self else hd[i][:K]; ds = set(int(x) for x in (dd[i][1:] if exclude_self else dd[i]))
            rec[i] = sum(int(x) in ds for x in h) / K
        return round(float(rec.mean()), 4), int(q.size)

    member = np.where(is_member)[0]
    poolheld = np.where(~is_member[:N_POOL])[0]
    compl = N_POOL + np.where(~is_member[N_POOL:])[0]
    print(f"[recep6m {PROJ.name}] member {member.size:,} pool-heldout {poolheld.size:,} complement {compl.size:,}", flush=True)
    m_rec, m_n = reception(member, True)          # members are in the ref -> drop self
    p_rec, p_n = reception(poolheld, False)
    c_rec, c_n = reception(compl, False)
    out = {"schema": "fullcorpus-reception-3way-6m-2026-09-07", "proj": str(PROJ), "ref_hd": REF_HD, "pca": PCA_MODEL,
           "k": K, "disc": disc, "nref": int(nref),
           "member": {"reception": m_rec, "nq": m_n}, "pool_heldout": {"reception": p_rec, "nq": p_n},
           "complement_doubly_unseen": {"reception": c_rec, "nq": c_n},
           "projected_by": "other-agent (2d) / this-session (pca768-3d)", "stamped_by": "this-session",
           "note": "recall@15 of a query's high-D 15NN (among the 6M trained-on ref) that are its 2D 0.1%-disc neighbors in the ref layout; PCA heads use PCA-768 high-D."}
    (PROJ / "reception-3way.json").write_text(json.dumps(out, indent=1))
    print(f"[recep6m {PROJ.name}] member {m_rec} pool-heldout {p_rec} complement {c_rec} -> {PROJ/'reception-3way.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
