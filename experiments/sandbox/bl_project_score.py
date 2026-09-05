"""BL OOD projection + scoring through a MONET head (owner item c, controlled design; 2026-09-05). GPU (project)
+ CPU (faiss/KDTree). Projects BL (British Library, 1.08M) and the pool-thumb control (200K) — BOTH thumb-embedded
via the byte-identical pipeline — through a frozen head, then:
  1) RECEPTION per BL segment vs the pool-thumb FLOOR: recall@15 of a query's high-D 15NN (among the head's
     TRAINING set) within its 0.1%-of-training 2D disc. BL_segment - pool_thumb_floor = corpus-isolated reception
     (the common thumb/config offset cancels because BOTH are thumb-embedded through the same pipeline).
  2) SINK FORENSICS (light): 2D occupancy of BL vs pool-thumb on a shared grid over the head's map extent —
     entropy + top-1% cell mass + effective coverage. A sink = BL piling into few cells (low entropy / high top-1%)
     relative to the pool-thumb control. Answers: does an OOD image corpus collapse in a frozen MONET map?

Head-parameterized: HEAD_MODEL, HEAD_COORDS (training 2D layout), HEAD_TRAIN_HD (training high-D), HEAD_TAG.
Usage: HEAD_MODEL=.. HEAD_COORDS=.. HEAD_TRAIN_HD=.. HEAD_TAG=2m-2d bl_project_score.py [NQ_PER=6000] [SEED=0]
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

BL = Path("/data2/monet/bl-clip")
OUT = Path("/data/latent-basemap/sandbox/bl-ood-20260905")
SEGMENTS = ["covers", "medium", "embellishments", "plates"]
K = 15
MODEL = Path(os.environ["HEAD_MODEL"]); COORDS = Path(os.environ["HEAD_COORDS"])
TRAIN_HD = Path(os.environ["HEAD_TRAIN_HD"]); TAG = os.environ["HEAD_TAG"]


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def _project(model_path, X):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _paths import ensure_paths; ensure_paths()
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    p = ParametricUMAP.load(str(model_path), device="cuda")
    return np.asarray(p.transform(X, batch_size=16384), dtype=np.float32)


def _reception(q_hd, q_2d, index, tree, disc):
    _, hd = index.search(q_hd, K)
    _, d2 = tree.query(q_2d, k=disc, workers=-1)
    rec = np.empty(q_hd.shape[0])
    for i in range(q_hd.shape[0]):
        ds = set(int(x) for x in d2[i]); rec[i] = sum(int(x) in ds for x in hd[i][:K]) / K
    return float(rec.mean())


def _occupancy(xy, extent, bins=256):
    (x0, x1), (y0, y1) = extent
    hx = np.clip(((xy[:, 0] - x0) / (x1 - x0 + 1e-9) * bins).astype(int), 0, bins - 1)
    hy = np.clip(((xy[:, 1] - y0) / (y1 - y0 + 1e-9) * bins).astype(int), 0, bins - 1)
    flat = hx * bins + hy
    counts = np.bincount(flat, minlength=bins * bins).astype(np.float64)
    p = counts / counts.sum(); nz = p[p > 0]
    ent = float(-(nz * np.log(nz)).sum()); ent_norm = ent / np.log(bins * bins)
    top1 = float(np.sort(p)[::-1][:max(1, (bins * bins) // 100)].sum())
    occ = float((counts > 0).mean())
    return {"entropy_nats": round(ent, 4), "entropy_norm": round(ent_norm, 4),
            "top1pct_cell_mass": round(top1, 4), "occupied_cell_frac": round(occ, 4)}


def main():
    nqp = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    import faiss
    from scipy.spatial import cKDTree
    OUT.mkdir(parents=True, exist_ok=True)

    bl_hd = _norm(np.asarray(np.load(BL / "bl-clip.f32.npy", mmap_mode="r"), np.float32))
    bl_seg = np.load(BL / "bl-segment.npy")
    ps_hd = _norm(np.asarray(np.load(BL / "poolthumb-clip.f32.npy", mmap_mode="r"), np.float32))
    ps_valid = np.load(BL / "poolthumb-valid.npy")

    t0 = time.time()
    bl_2d = _project(MODEL, bl_hd); ps_2d = _project(MODEL, ps_hd)
    print(f"[{TAG}] projected BL {bl_2d.shape} + pool-thumb {ps_2d.shape} in {time.time()-t0:.0f}s", flush=True)

    ref_hd = _norm(np.asarray(np.load(TRAIN_HD, mmap_mode="r"), np.float32))
    ref_2d = np.asarray(np.load(COORDS), np.float64)
    index = faiss.IndexFlatIP(512); index.add(ref_hd)
    tree = cKDTree(ref_2d)
    # cap the 2D disc at 2000 (was 0.1% of ref): keeps BL reception COMPARABLE across heads of different
    # training size (2M vs 4M ref) and bounds the KDTree k-query cost (4M's 0.1%=4000 ran ~1h under CPU load).
    disc = min(max(int(round(ref_hd.shape[0] * 0.001)), K), 2000)
    rng = np.random.default_rng(seed)

    def recep_of(hd_all, mask, n):
        pos = np.where(mask)[0] if mask is not None else np.arange(hd_all.shape[0])
        q = np.sort(rng.choice(pos, min(n, pos.size), replace=False))
        return _reception(hd_all[q], np.asarray((bl_2d if hd_all is bl_hd else ps_2d)[q], np.float64), index, tree, disc), int(q.size)

    ps_floor, ps_n = recep_of(ps_hd, ps_valid, nqp)
    seg_rec = {}
    for si, seg in enumerate(SEGMENTS):
        r, n = recep_of(bl_hd, bl_seg == si, nqp)
        seg_rec[seg] = {"reception": round(r, 4), "minus_floor": round(r - ps_floor, 4), "nq": n}
    bl_all, bl_n = recep_of(bl_hd, None, nqp)

    # sink forensics on the head's map extent (from training layout)
    ext = ((float(ref_2d[:, 0].min()), float(ref_2d[:, 0].max())),
           (float(ref_2d[:, 1].min()), float(ref_2d[:, 1].max())))
    occ = {"pool_thumb": _occupancy(ps_2d[ps_valid], ext), "bl_all": _occupancy(bl_2d, ext)}
    for si, seg in enumerate(SEGMENTS):
        occ[f"bl_{seg}"] = _occupancy(bl_2d[bl_seg == si], ext)

    out = {"schema": "bl-ood-score-2026-09-05", "head": TAG, "k": K,
           "reception": {"pool_thumb_floor": round(ps_floor, 4), "pool_thumb_nq": ps_n,
                         "bl_all": round(bl_all, 4), "bl_all_minus_floor": round(bl_all - ps_floor, 4),
                         "bl_by_segment": seg_rec},
           "sink_forensics": occ,
           "interpretation": "reception minus_floor < 0 = BL received WORSE than pool (corpus-OOD, offset-cancelled). "
                             "sink: BL entropy_norm << pool_thumb or top1pct >> pool_thumb = BL collapsing into sinks."}
    (OUT / f"score-{TAG}.json").write_text(json.dumps(out, indent=1))
    print(f"[{TAG}] floor {ps_floor:.4f} | BL all {bl_all:.4f} ({bl_all-ps_floor:+.4f}) | "
          f"by-seg " + " ".join(f"{s}={seg_rec[s]['minus_floor']:+.3f}" for s in SEGMENTS), flush=True)
    print(f"[{TAG}] sink: BL entropy_norm {occ['bl_all']['entropy_norm']} vs pool {occ['pool_thumb']['entropy_norm']} | "
          f"BL top1% {occ['bl_all']['top1pct_cell_mass']} vs pool {occ['pool_thumb']['top1pct_cell_mass']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
