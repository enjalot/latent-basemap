"""exp-2d image-only FFR baseline (overseer 2026-09-06). Two modes:

prep : build the WITHIN-image fuzzy graph for the 250K image rows from the relational run's within-modality-15
       truth (knn_indices.npy rows [0,250K) are already within-image, indices in [0,250K)). Writes
       monet-neomme-2d-imgbaseline/{edges-k15-fuzzy.npz, knn_indices.npy}. Then train via image_map_pipeline.

score: FFR of the trained 250K-image map vs that same within-15 image truth (recall of within-image k15 among the
       2D disc of N*0.1% neighbors, self excluded) — identical metric to exp2c_score._within_ffr, image rows only.
       Writes baseline_ffr.json with the guard: cost = baseline_image_ffr - joint_image_ffr (0.5185) must be ≤ 0.10.

Usage: exp2d_image_baseline.py {prep|score}
"""
import sys, json
from pathlib import Path
import numpy as np

SB = Path("/data/latent-basemap/sandbox")
REL = SB / "monet-neomme-2d-relational"; OUT = SB / "monet-neomme-2d-imgbaseline"
N = 250000; K = 15; JOINT_IMG_FFR = 0.5185


def prep():
    OUT.mkdir(parents=True, exist_ok=True)
    idx = np.load(REL / "knn_indices.npy")[:N]              # within-image k15 truth (indices already in [0,N))
    assert idx.max() < N, f"image-row truth references non-image index {idx.max()} — not within-modality"
    # knn_dists wasn't persisted by exp2d_stratified_knn → recompute within-image cosine dists from the substrate
    # (batched on GPU; materializing all N*15*1152 neighbor vectors at once would be ~17GB).
    import torch
    x = np.asarray(np.load("/data2/monet/exp2d-siglip-500k/substrate.f32.npy", mmap_mode="r")[:N], dtype=np.float32)
    xg = torch.nn.functional.normalize(torch.from_numpy(x).cuda(), dim=1)
    dst = np.empty((N, K), dtype=np.float32)
    for s in range(0, N, 8192):
        e = min(s + 8192, N)
        q = xg[s:e].unsqueeze(1)                            # (b,1,1152)
        nb = xg[torch.from_numpy(idx[s:e].astype(np.int64)).cuda()]  # (b,15,1152)
        dst[s:e] = (1.0 - (q * nb).sum(-1)).clamp_min(0.0).cpu().numpy()
    del xg
    np.save(OUT / "knn_indices.npy", idx); np.save(OUT / "knn_dists.npy", dst)
    from umap.umap_ import fuzzy_simplicial_set
    knn_i = np.concatenate([np.arange(N, dtype=np.int64)[:, None], idx.astype(np.int64)], 1)
    knn_d = np.concatenate([np.zeros((N, 1), np.float32), dst], 1)
    g, _, _ = fuzzy_simplicial_set(X=np.empty((N, 1), np.float32), n_neighbors=K + 1,
                                   random_state=42, metric="euclidean", knn_indices=knn_i, knn_dists=knn_d)
    g = g.tocoo()
    np.savez(OUT / "edges-k15-fuzzy.npz", sources=g.row.astype(np.int32), targets=g.col.astype(np.int32),
             weights=g.data.astype(np.float32), n_nodes=np.int64(N))
    print(f"[imgbaseline] prep: within-image fuzzy edges {len(g.row):,} ({len(g.row)/N:.1f}/node) -> {OUT}")
    return 0


def score():
    from scipy.spatial import cKDTree
    xy = np.asarray(np.load(OUT / "champion-bs16k" / "coordinates.npy"), dtype=np.float32)
    knn = np.load(OUT / "knn_indices.npy", mmap_mode="r")
    rng = np.random.default_rng(0); q = rng.choice(N, min(10000, N), replace=False)
    tree = cKDTree(xy); disc = max(K, int(round(N * 0.001)))
    _, nbr = tree.query(xy[q], k=disc + 1, workers=-1)
    hit = tot = 0
    for r, qi in enumerate(q):
        ds = set(int(x) for x in nbr[r] if x != qi)
        truth = [int(t) for t in knn[qi][:K] if t != qi]
        if truth:
            hit += sum(t in ds for t in truth); tot += len(truth)
    ffr = round(hit / tot, 4) if tot else None
    cost = round(ffr - JOINT_IMG_FFR, 4)
    res = {"baseline_image_ffr": ffr, "joint_image_ffr": JOINT_IMG_FFR, "cost": cost,
           "guard_passed": bool(cost <= 0.10),
           "note": "cost = baseline_image_ffr - joint_image_ffr = intra-image structure lost by adding cross-modal edges; prereg guard ≤0.10"}
    (OUT / "baseline_ffr.json").write_text(json.dumps(res, indent=1))
    print(f"[imgbaseline] baseline image FFR {ffr} | joint {JOINT_IMG_FFR} | cost {cost} | guard {'PASS' if res['guard_passed'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit({"prep": prep, "score": score}[sys.argv[1]]())
