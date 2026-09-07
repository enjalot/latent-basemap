"""Score the 12M DINO→PCA-768 ladder rung (owner overnight run 2026-09-07). OFF-FLOCK (faiss CPU + a seconds-long
GPU projection). v2 FFR on own truth + held-out reception recall@15 (val already PCA-768 in the draw, so NO
transform here — unlike the 6M score which PCA-transforms a 1536-d val). Reception: project val through the
champion head -> each val query's 15 high-D NN among the 12M TRAINING rows (faiss IP over pca768-substrate) vs its
2D 0.1%-disc in the 12M map. Writes SANDBOX/monet-random-dino-12m-pca768/ladder-score.json.
Usage: exp_dino_12m_score.py
"""
import json, time
from pathlib import Path
import numpy as np

SB = Path("/data/latent-basemap/sandbox"); D = Path("/data2/monet/random-dino-12m")
DS = "monet-random-dino-12m-pca768"; K = 15


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def main():
    arm = SB / DS; ckpt = arm / "champion-bs16k"
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from knobs_2m import quick_ffr_v2
    import faiss, torch
    from _paths import ensure_paths; ensure_paths()
    from basemap.pumap.parametric_umap.core import ParametricUMAP

    coords = np.asarray(np.load(ckpt / "coordinates.npy"), np.float32); n = coords.shape[0]
    ffr_v2 = float(quick_ffr_v2(coords, arm / "edges-k15-fuzzy.npz", n))
    print(f"[12m-score] v2 FFR (own truth) = {ffr_v2:.5f}", flush=True)

    sub = np.load(D / "pca768-substrate.f16.npy", mmap_mode="r")           # 12M x768 (already PCA-768, renormed)
    val = _norm(np.asarray(np.load(D / "val-pca768.f16.npy", mmap_mode="r"), np.float32))
    t0 = time.time(); hdx = faiss.IndexFlatIP(sub.shape[1])
    for i in range(0, n, 1_000_000):                                       # chunked add (no whole-array f32 materialize)
        hdx.add(np.ascontiguousarray(np.asarray(sub[i:i + 1_000_000], np.float32)))
    d2x = faiss.IndexFlatL2(coords.shape[1]); d2x.add(np.ascontiguousarray(coords)); disc = max(int(round(n * 0.001)), K)
    print(f"[12m-score] faiss built {time.time()-t0:.0f}s | disc {disc}", flush=True)

    pumap = ParametricUMAP.load(str(ckpt / "model.pt"), device="cuda"); pumap.model.eval()
    with torch.no_grad():
        v2d = pumap.model(torch.from_numpy(val).cuda()).cpu().numpy().astype(np.float32)
    _, hd = hdx.search(val, K); _, dd = d2x.search(np.ascontiguousarray(v2d), disc)
    rec = float(np.mean([len(set(int(x) for x in hd[i]) & set(int(x) for x in dd[i])) / K for i in range(val.shape[0])]))
    out = {"schema": "monet-dino-12m-pca768-score-2026-09-07", "ds": DS, "n_rows": int(n),
           "v2_ffr_own_truth": round(ffr_v2, 5), "heldout_reception_at15": round(rec, 4), "n_val": int(val.shape[0]),
           "disc": disc, "pca_reused_from_6m": True,
           "note": "ladder rung 12M; v2 FFR own-truth + held-out reception (val already PCA-768). Compare to 2M/6M rungs."}
    (arm / "ladder-score.json").write_text(json.dumps(out, indent=1))
    print(f"[12m-score] v2FFR {ffr_v2:.5f} | held-out reception@15 {rec:.4f} -> {arm/'ladder-score.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
