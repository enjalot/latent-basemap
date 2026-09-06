"""Score one 6M DINO ladder arm (owner residency experiment 2026-09-06). Args: <arm_ds> <dim>.
- v2 FFR on own truth: quick_ffr_v2(coords, arm/edges-k15-fuzzy.npz, n_rows).
- held-out reception recall@15: project the val held-out set through the arm's champion head -> for each val query,
  its 15 high-D NN among the 6M TRAINING rows (faiss IP over the arm's substrate) vs its 2D 0.1%-disc in the 6M map.
  For the PCA arm, val is PCA-transformed (pca768-model) + renormed to match the substrate space.
Writes SANDBOX/<arm_ds>/ladder-score.json. Usage: exp_dino_6m_score.py monet-random-dino-6m 1536
"""
import json, sys, time
from pathlib import Path
import numpy as np

SB = Path("/data/latent-basemap/sandbox"); D6 = Path("/data2/monet/random-dino-6m")
K = 15


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def main():
    ds = sys.argv[1]; dim = int(sys.argv[2])
    arm = SB / ds; ckpt_dir = arm / "champion-bs16k"
    sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from knobs_2m import quick_ffr_v2
    import faiss
    from scipy.spatial import cKDTree

    coords = np.asarray(np.load(ckpt_dir / "coordinates.npy"), np.float32); n = coords.shape[0]
    ffr_v2 = quick_ffr_v2(coords, arm / "edges-k15-fuzzy.npz", n)
    print(f"[6m-score {ds}] v2 FFR (own truth) = {ffr_v2}", flush=True)

    # held-out reception: val through the head
    sub = np.load(D6 / ("pca768-substrate.f32.npy" if dim == 768 else "dino-substrate.f16.npy"), mmap_mode="r")
    val_hd = _norm(np.asarray(np.load(D6 / "val-dino.f16.npy", mmap_mode="r"), np.float32))   # 1536-d
    if dim == 768:
        m = np.load(D6 / "pca768-model.npz"); val_hd = _norm((val_hd - m["mean"]) @ m["components"])
    # project val through the champion head
    from _paths import ensure_paths; ensure_paths()
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    import torch
    pumap = ParametricUMAP.load(str(ckpt_dir / "model.pt"), device="cuda"); pumap.model.eval()
    rng = np.random.default_rng(0); q = np.sort(rng.choice(val_hd.shape[0], min(6000, val_hd.shape[0]), replace=False))
    with torch.no_grad():
        val_2d = pumap.model(torch.from_numpy(val_hd[q]).cuda()).cpu().numpy().astype(np.float64)
    # faiss IP over the 6M reference (chunked add), cKDTree over 6M coords
    t0 = time.time(); index = faiss.IndexFlatIP(dim)
    for i in range(0, n, 500_000):
        index.add(_norm(np.asarray(sub[i:i + 500_000], np.float32)))
    tree = cKDTree(coords.astype(np.float64)); disc = max(int(round(n * 0.001)), K)
    print(f"[6m-score {ds}] faiss+tree built {time.time()-t0:.0f}s, disc={disc}", flush=True)
    _, hd = index.search(val_hd[q], K)
    _, d2 = tree.query(val_2d, k=disc, workers=-1)
    rec = np.mean([len(set(int(x) for x in hd[i]) & set(int(x) for x in d2[i])) / K for i in range(len(q))])

    out = {"schema": "dino-6m-ladder-arm-2026-09-06", "arm": ds, "dim": dim, "n_rows": int(n),
           "v2_ffr_own_truth": round(float(ffr_v2), 5), "heldout_reception_recall@15": round(float(rec), 4),
           "heldout_nq": int(len(q)),
           "note": "reception: val held-out (never trained) projected through the champion head; recall of its 15 high-D "
                   "NN among the 6M training rows vs its 2D 0.1%-disc. PCA arm: val PCA-transformed+renormed first."}
    (arm / "ladder-score.json").write_text(json.dumps(out, indent=1))
    print(f"[6m-score {ds}] v2FFR {ffr_v2:.4f} | held-out reception {rec:.4f} -> ladder-score.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
