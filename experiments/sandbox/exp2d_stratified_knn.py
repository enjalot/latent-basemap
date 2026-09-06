"""exp-2d stratified rank-composed kNN + fuzzy graph (owner 2026-09-06). GPU. THE CORE of exp-2d: per node, the
k/2=8 nearest WITHIN its own modality + k/2=8 nearest CROSS-modality (exact cosine, self excluded), assembled into
one k=16 neighbor list -> umap-learn fuzzy_simplicial_set (per-point rho/sigma = the gauge-absorption under test).
No pair supervision in the edges; the cross edges are SEMANTIC ranks. Generalizes exp-2c's 1-injected-pair-edge to
k semantic cross edges/node.

Writes into SANDBOX/<ds>/{knn_indices.npy, knn_dists.npy, edges-k15-fuzzy.npz} matching image_map_pipeline's format
so train() reuses it unchanged.

--holdout FRAC (default 0): the CLEAN generalization variant. Hold out FRAC of pair_ids; for a held-out node, if
its matched partner (row N+i / i) lands among its 8 cross neighbors, DROP it and shift in the next-nearest cross
neighbor -> the graph never DIRECTLY connects a held-out matched pair. matched_partner@k15 on held-out pairs then
measures whether the relational kernel pulls them adjacent via SHARED neighbors alone. Saves holdout_pairs.npy.

Usage: exp2d_stratified_knn.py <ds> [--holdout 0.1]  (main venv python; torch + umap-learn)
"""
import os, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path
import numpy as np, torch

SB = Path("/data/latent-basemap/sandbox"); SUB = "/data2/monet/exp2d-siglip-500k"
# stratification split (owner ratio sweep 2026-09-06): KWITHIN within-modality + KCROSS cross-modality per node.
# default 8+8; the sweep sets {12+4, 10+6} via env. Total k = KWITHIN+KCROSS (=16 across all cells).
KWITHIN = int(os.environ.get("EXP2D_KWITHIN", 8)); KCROSS = int(os.environ.get("EXP2D_KCROSS", 8)); SEED = 42


TRUTH_K = 15   # plain within-modality exact-k15 = the FFR quality-guard truth (saved as knn_indices.npy)


def _stratified(x, mod, N, holdout_ids=None):
    """x: (2N,D) f32 L2-normed on GPU. mod: (2N,) 0/1. Returns:
       strat_idx (2N,16)/strat_dst — 8 within + 8 cross (feeds fuzzy graph);
       truth_idx (2N,15) — plain within-modality exact-k15 (FFR quality-guard truth)."""
    n = x.shape[0]
    s_i = np.empty((n, KWITHIN + KCROSS), np.int32); s_d = np.empty((n, KWITHIN + KCROSS), np.float32)
    t_i = np.empty((n, TRUTH_K), np.int32)
    ho = set(holdout_ids.tolist()) if holdout_ids is not None else set()
    KW = max(TRUTH_K, KWITHIN) + 2        # within candidate pool (survive self, cover k15 truth + graph-KWITHIN)
    KC = KCROSS + 2                       # cross candidate pool (survive self / partner)
    q_chunk, d_chunk = 2048, 262_144
    with torch.no_grad():
        for qs in range(0, n, q_chunk):
            q = x[qs:qs + q_chunk]; qn = q.shape[0]
            qmod = mod[qs:qs + qn]
            rows = torch.arange(qs, qs + qn, device="cuda")
            bw_s = torch.full((qn, KW), -2.0, device="cuda"); bw_i = torch.zeros((qn, KW), dtype=torch.long, device="cuda")
            bc_s = torch.full((qn, KC), -2.0, device="cuda"); bc_i = torch.zeros((qn, KC), dtype=torch.long, device="cuda")
            for ds0 in range(0, n, d_chunk):
                db = x[ds0:ds0 + d_chunk]; dmod = mod[ds0:ds0 + db.shape[0]]
                sims = q @ db.T
                same = (dmod.unsqueeze(0) == qmod.unsqueeze(1))
                sw = torch.where(same, sims, torch.tensor(-2.0, device="cuda"))
                sc = torch.where(same, torch.tensor(-2.0, device="cuda"), sims)
                for (bs, bi, sm, kk) in ((bw_s, bw_i, sw, KW), (bc_s, bc_i, sc, KC)):
                    s, i = torch.topk(sm, min(kk, sm.shape[1]), dim=1)
                    cs, sel = torch.topk(torch.cat([bs, s], 1), kk, dim=1)
                    ci = torch.gather(torch.cat([bi, i + ds0], 1), 1, sel)
                    bs.copy_(cs); bi.copy_(ci)
            w_i15, _ = _take(bw_i, bw_s, rows, TRUTH_K, drop_partner=None)          # within-15 truth (FFR)
            w_ig, w_dg = _take(bw_i, bw_s, rows, KWITHIN, drop_partner=None)        # within-KWITHIN for graph
            partner = torch.where(rows < N, rows + N, rows - N)
            drop_p = partner.clone()
            if ho:
                pid = torch.where(rows < N, rows, rows - N)
                keep_drop = torch.tensor([int(p) in ho for p in pid.tolist()], device="cuda")
                drop_p = torch.where(keep_drop, partner, torch.full_like(partner, -1))
            c_ig, c_dg = _take(bc_i, bc_s, torch.full_like(rows, -1), KCROSS, drop_partner=drop_p)  # cross-KCROSS
            s_i[qs:qs + qn] = torch.cat([w_ig, c_ig], 1).cpu().numpy().astype(np.int32)
            s_d[qs:qs + qn] = torch.cat([w_dg, c_dg], 1).cpu().numpy().astype(np.float32)
            t_i[qs:qs + qn] = w_i15.cpu().numpy().astype(np.int32)
            if qs % (q_chunk * 40) == 0: print(f"  knn {qs}/{n}", flush=True)
    return s_i, np.clip(s_d, 0.0, None), t_i


def _take(bi, bs, self_rows, k, drop_partner):
    """From (qn,KE) candidates drop self_rows (and drop_partner if given), return top-k idx + (1-cos) dist."""
    keep = torch.ones_like(bi, dtype=torch.bool)
    keep &= bi != self_rows.unsqueeze(1)
    if drop_partner is not None:
        keep &= bi != drop_partner.unsqueeze(1)
    bs2 = torch.where(keep, bs, torch.tensor(-3.0, device="cuda"))
    s, sel = torch.topk(bs2, k, dim=1)
    i = torch.gather(bi, 1, sel)
    return i, (1.0 - s)


def main():
    ds = sys.argv[1]
    holf = float(sys.argv[sys.argv.index("--holdout") + 1]) if "--holdout" in sys.argv else 0.0
    out = SB / ds; out.mkdir(parents=True, exist_ok=True)
    x = np.load(f"{SUB}/substrate.f32.npy", mmap_mode="r")
    mod = np.load(f"{SUB}/modality.npy"); N = int((mod == 0).sum())
    print(f"[exp2d-knn] ds={ds} 2N={x.shape[0]} D={x.shape[1]} N={N} holdout={holf}", flush=True)
    xg = torch.from_numpy(np.ascontiguousarray(x)).float().cuda()
    xg = torch.nn.functional.normalize(xg, dim=1)
    modg = torch.from_numpy(mod.astype(np.int64)).cuda()
    ho_ids = None
    if holf > 0:
        rng = np.random.default_rng(SEED); ho_ids = np.sort(rng.choice(N, int(N * holf), replace=False)).astype(np.int64)
        np.save(out / "holdout_pairs.npy", ho_ids); print(f"[exp2d-knn] held out {len(ho_ids)} pairs' direct partner cross-edge", flush=True)
    t0 = time.time()
    strat_i, strat_d, truth_i = _stratified(xg, modg, N, ho_ids)
    np.save(out / "knn_strat.npy", strat_i)                          # provenance: 8within+8cross graph indices
    np.save(out / "knn_indices.npy", truth_i)                        # FFR quality-guard truth (within-modality k15)
    print(f"[exp2d-knn] stratified {KWITHIN}within+{KCROSS}cross (+within-{TRUTH_K} truth) in {(time.time()-t0)/60:.1f} min -> fuzzy", flush=True)
    from umap.umap_ import fuzzy_simplicial_set
    n = strat_i.shape[0]
    knn_i = np.concatenate([np.arange(n, dtype=np.int64)[:, None], strat_i.astype(np.int64)], 1)
    knn_d = np.concatenate([np.zeros((n, 1), np.float32), strat_d], 1)
    g, _, _ = fuzzy_simplicial_set(X=np.empty((n, 1), np.float32), n_neighbors=KWITHIN + KCROSS + 1,
                                   random_state=SEED, metric="euclidean", knn_indices=knn_i, knn_dists=knn_d)
    g = g.tocoo()
    np.savez(out / "edges-k15-fuzzy.npz", sources=g.row.astype(np.int32), targets=g.col.astype(np.int32),
             weights=g.data.astype(np.float32), n_nodes=np.int64(n))
    print(f"[exp2d-knn] DONE fuzzy edges {len(g.row):,} ({len(g.row)/n:.1f}/node) -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
