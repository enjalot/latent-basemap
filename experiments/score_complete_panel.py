"""Complete-panel decision scorer for the R1 kernel comparison (plan §R1).

The kernel A/B (run_r1_kernel.py) reports transductive ffr/recall/density. Before
any kernel DECISION the plan requires the FULL canonical panel:
  - purity at k∈{256,1024} against FROZEN centroids (computed once per substrate);
  - held-out PROJECTION fidelity (queries provably outside the training rows) +
    a random floor;
  - a kNN-regressor OOS baseline — the NUMAP trigger: if the neural map does not
    clearly beat non-parametric kNN regression on held-out projection, we do NOT
    claim method superiority.

Everything routes through basemap.panel_v2 for the transductive metrics and reuses
its exact ffr formula for projection, so the numbers are comparable to the A/B.

Usage:
  python experiments/score_complete_panel.py \
     --runs legacy=<dir> umap=<dir> stdcurve=<dir> \
     --testbed /data/latent-basemap/jina-en-200k \
     --source /data/embeddings/fineweb-edu-sample-10BT-chunked-500-jina-v5-nano/train \
     --out /data/latent-basemap/r1_kernel/complete_panel.json
"""
from __future__ import annotations
import argparse, os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from basemap.panel_v2 import (score_panel, PanelV2Config, load_embeddings, load_coords,
                              ffr_from_neighbors, recall_at_k_from_neighbors, _ids_hash)


def frozen_centroids(X, ks, cache_dir, seed=0, iters=25):
    """GPU k-means (random init + Lloyd) once per substrate; cached to disk so the
    purity labels are a frozen artifact, never silently regenerated."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    n = len(X)
    for k in ks:
        path = os.path.join(cache_dir, f"centroids_k{k}.npy")
        if os.path.exists(path):
            out[k] = np.load(path); continue
        rng = np.random.RandomState(seed)
        C = torch.from_numpy(np.asarray(X[np.sort(rng.choice(n, k, replace=False))],
                                        dtype=np.float32)).to(dev)
        for _ in range(iters):
            sums = torch.zeros_like(C); counts = torch.zeros(k, device=dev)
            for i in range(0, n, 100000):
                xb = torch.from_numpy(np.asarray(X[i:i + 100000], dtype=np.float32)).to(dev)
                lab = torch.cdist(xb, C).argmin(1)
                sums.index_add_(0, lab, xb); counts.index_add_(0, lab, torch.ones(len(xb), device=dev))
            nz = counts > 0
            C[nz] = sums[nz] / counts[nz, None]
        out[k] = C.cpu().numpy(); np.save(path, out[k])
    return out


def _cross_topk(Q, corpus, k, lo, cfg, q_tile=4096):
    """Top-k of each Q row over corpus (cross, not self). Tiles BOTH the query rows
    (q_tile) and the corpus (cfg.corpus_chunk) so peak VRAM is bounded — a single
    (all-queries × corpus) matrix is ~16 GB at 20k×200k and OOMs. lo=True → exact
    cdist on tiny coords; else normalised-matmul expansion."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cc = len(corpus) if lo else min(len(corpus), max(1, cfg.corpus_chunk))
    out = np.empty((len(Q), k), dtype=np.int64)
    for q0 in range(0, len(Q), q_tile):
        Qt = torch.from_numpy(np.asarray(Q[q0:q0 + q_tile], dtype=np.float32)).to(dev)
        qn = (Qt * Qt).sum(1, keepdim=True)
        best_d = torch.full((len(Qt), k), float("inf"), device=dev)
        best_i = torch.full((len(Qt), k), -1, dtype=torch.long, device=dev)
        for j in range(0, len(corpus), cc):
            Xc = torch.from_numpy(np.asarray(corpus[j:j + cc], dtype=np.float32)).to(dev)
            d2 = (torch.cdist(Qt, Xc) ** 2) if lo else (qn - 2.0 * (Qt @ Xc.T) + (Xc * Xc).sum(1))
            kloc = min(k, len(Xc))
            ld, li = torch.topk(d2, kloc, dim=1, largest=False)
            best_d = torch.cat([best_d, ld], 1); best_i = torch.cat([best_i, li + j], 1)
            best_d, sel = torch.topk(best_d, k, dim=1, largest=False)
            best_i = torch.gather(best_i, 1, sel); del Xc, d2
        out[q0:q0 + len(Qt)] = best_i.cpu().numpy(); del Qt, best_d, best_i
    return out


def projection_ffr(X, Z, Xq, Zq, cfg):
    """Held-out FFR: hi-D query→corpus top-k_hit vs projected-query→map top-k_frac,
    via the canonical ffr formula. Returns (ffr, recall@k)."""
    kf = max(cfg.k_hit, int(np.ceil(cfg.frac * len(Z))))
    hi = _cross_topk(Xq, X, cfg.k_hit, lo=False, cfg=cfg)
    lo = _cross_topk(Zq, Z, kf, lo=True, cfg=cfg)
    return (round(ffr_from_neighbors(hi, lo, cfg.k_hit), 4),
            round(recall_at_k_from_neighbors(hi, lo, cfg.k_hit), 5))


def knn_regress_coords(Xq, X, Z, cfg, k=15):
    """Non-parametric OOS map: each held-out query's 2D = mean of the map coords of
    its k nearest TRAIN rows in high-D. The baseline the neural map must beat."""
    nb = _cross_topk(Xq, X, k, lo=False, cfg=cfg)     # (nq, k) train-row ids
    return Z[nb].mean(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="label=run_dir pairs")
    ap.add_argument("--testbed", required=True)
    ap.add_argument("--source", required=True, help="dir of source shards for held-out queries")
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--n-holdout", type=int, default=20000)
    ap.add_argument("--frac", type=float, default=0.001)
    ap.add_argument("--n-anchors", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from basemap.pumap.parametric_umap.core import ParametricUMAP
    # cap corpus_chunk so query-tile × corpus-chunk cross matrices stay bounded
    # (4096 × 500k ≈ 2 GB fp32) even when the corpus is 2M+.
    cfg = PanelV2Config(frac=args.frac, n_anchors=args.n_anchors, corpus_chunk=500_000)
    runs = dict(kv.split("=", 1) for kv in args.runs)

    X = load_embeddings(os.path.join(args.testbed, "train"), dim=args.dim)
    si = np.load(os.path.join(args.testbed, "sample_indices.npy"))
    centroids = frozen_centroids(X, (256, 1024), args.testbed)

    # held-out queries: source rows NOT in the testbed sample (real OOS proof)
    src = load_embeddings(args.source, dim=args.dim)
    train_set = set(int(i) for i in si)
    rng = np.random.RandomState(args.seed)
    held = []
    while len(held) < args.n_holdout:
        cand = rng.randint(0, len(src), args.n_holdout * 2)
        held += [int(c) for c in cand if int(c) not in train_set]
    held = np.sort(np.array(held[:args.n_holdout], dtype=np.int64))
    assert len(set(held.tolist()) & train_set) == 0, "held-out overlaps training rows"
    Xq = np.asarray(src[held], dtype=np.float32)

    summary = {"testbed": args.testbed, "n": int(len(X)), "n_holdout": int(len(held)),
               "held_disjoint_from_train": True, "held_hash": _ids_hash(held),
               "frac": cfg.frac, "runs": {},
               "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    for label, rd in runs.items():
        t0 = time.time()
        Z, z_ids = load_coords(os.path.join(rd, "coords.parquet"))
        panel = score_panel(X, Z, config=cfg, centroids_by_k=centroids,
                            provenance={"scorer": "complete_panel", "run": os.path.basename(rd)})
        model = ParametricUMAP.load(os.path.join(rd, "model.pt"), device="cuda")
        Zq = np.asarray(model.transform(Xq), dtype=np.float32)
        proj_ffr, proj_rk = projection_ffr(X, Z, Xq, Zq, cfg)
        Zq_knn = knn_regress_coords(Xq, X, Z, cfg)
        knn_ffr, _ = projection_ffr(X, Z, Xq, Zq_knn, cfg)
        lo, hi = Z.min(0), Z.max(0)
        Zq_rand = (rng.rand(len(Xq), Z.shape[1]).astype(np.float32) * (hi - lo) + lo)
        floor_ffr, _ = projection_ffr(X, Z, Xq, Zq_rand, cfg)
        summary["runs"][label] = {
            "run_dir": os.path.basename(rd), "wall_s": round(time.time() - t0, 1),
            "ffr": panel["ffr"], "recall@k": panel["recall@k"],
            "purity_k256": (panel.get("purity") or {}).get("k256"),
            "purity_k1024": (panel.get("purity") or {}).get("k1024"),
            "density": panel["density"],
            "proj_ffr": proj_ffr, "proj_recall@k": proj_rk,
            "proj_knn_regressor_ffr": knn_ffr, "proj_random_floor_ffr": floor_ffr,
            "proj_beats_knn": bool(proj_ffr > knn_ffr),
            "proj_margin_over_knn": round(proj_ffr - knn_ffr, 4)}
        json.dump(summary, open(args.out, "w"), indent=1)
        r = summary["runs"][label]
        print(f"[panel] {label:10s} ffr={r['ffr']} purity1024={r['purity_k1024']} dens={r['density']} "
              f"| proj={r['proj_ffr']} knnReg={r['proj_knn_regressor_ffr']} floor={r['proj_random_floor_ffr']} "
              f"beats_knn={r['proj_beats_knn']}", flush=True)

    summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(summary, open(args.out, "w"), indent=1)
    print(f"[panel] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
