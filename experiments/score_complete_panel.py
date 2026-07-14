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
                              ffr_from_neighbors, recall_at_k_from_neighbors, _ids_hash,
                              cross_knn)


def _sha_file(path):
    import hashlib
    try:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except Exception:
        return None


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


def projection_ffr(X, Z, Xq, Zq, cfg):
    """Held-out FFR: hi-D query→corpus top-k_hit vs projected-query→map top-k_frac,
    via the canonical panel_v2.cross_knn + ffr formula (P0-4). Returns (ffr, r@k)."""
    kf = max(cfg.k_hit, int(np.ceil(cfg.frac * len(Z))))
    hi = cross_knn(Xq, X, cfg.k_hit, cfg, hi_dim=True)
    lo = cross_knn(Zq, Z, kf, cfg, hi_dim=False)
    return (round(ffr_from_neighbors(hi, lo, cfg.k_hit), 4),
            round(recall_at_k_from_neighbors(hi, lo, cfg.k_hit), 5))


def knn_regress_coords(Xq, X, Z, cfg, k=15):
    """Non-parametric OOS map: each held-out query's 2D = mean of the map coords of
    its k nearest TRAIN rows in high-D. The baseline the neural map must beat."""
    nb = cross_knn(Xq, X, k, cfg, hi_dim=True)         # (nq, k) train-row ids
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

    # P0-5: the GPU decision scorer must run under a held lease (controller or
    # in-process). Exempt when no CUDA (CPU scoring) or explicit unsafe override.
    import torch as _torch
    if _torch.cuda.is_available():
        from basemap.run_controller import require_active_lease
        require_active_lease()

    from basemap.pumap.parametric_umap.core import ParametricUMAP
    # cap corpus_chunk so query-tile × corpus-chunk cross matrices stay bounded
    # (4096 × 500k ≈ 2 GB fp32) even when the corpus is 2M+.
    cfg = PanelV2Config(frac=args.frac, n_anchors=args.n_anchors, corpus_chunk=500_000)
    runs = dict(kv.split("=", 1) for kv in args.runs)

    X = load_embeddings(os.path.join(args.testbed, "train"), dim=args.dim)
    si = np.load(os.path.join(args.testbed, "sample_indices.npy"))
    centroids = frozen_centroids(X, (256, 1024), args.testbed)

    # held-out queries: source rows NOT in the testbed sample (real OOS proof),
    # sampled WITHOUT replacement so the effective query count equals n_holdout
    # (P0-4 — the old with-replacement sampler yielded ~19.8k unique of 20k).
    src = load_embeddings(args.source, dim=args.dim)
    train_set = set(int(i) for i in si)
    complement = np.setdiff1d(np.arange(len(src), dtype=np.int64),
                              np.asarray(sorted(train_set), dtype=np.int64), assume_unique=False)
    if len(complement) < args.n_holdout:
        raise ValueError(f"only {len(complement)} held-out candidates < n_holdout {args.n_holdout}")
    rng = np.random.RandomState(args.seed)
    held = np.sort(rng.choice(complement, args.n_holdout, replace=False))
    assert len(np.unique(held)) == args.n_holdout, "held-out not unique"
    assert len(set(held.tolist()) & train_set) == 0, "held-out overlaps training rows"
    Xq = np.asarray(src[held], dtype=np.float32)

    import subprocess
    try:
        _commit = subprocess.check_output(["git", "-C", os.path.dirname(os.path.abspath(__file__)),
                                           "rev-parse", "HEAD"], text=True).strip()[:12]
        _dirty = bool(subprocess.check_output(["git", "-C", os.path.dirname(os.path.abspath(__file__)),
                                               "status", "--porcelain"], text=True).strip())
    except Exception:
        _commit = _dirty = None
    summary = {"testbed": args.testbed, "n": int(len(X)), "n_holdout": int(len(held)),
               "n_holdout_unique": int(len(np.unique(held))),
               "held_disjoint_from_train": True, "held_hash": _ids_hash(held),
               "source": args.source, "sample_indices_hash": _ids_hash(np.asarray(si, np.int64)),
               "frac": cfg.frac, "n_anchors": cfg.n_anchors, "seed": args.seed,
               "scorer_commit": _commit, "scorer_dirty": _dirty,
               "formula_version": cfg.formula_version, "runs": {},
               "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    for label, rd in runs.items():
        t0 = time.time()
        Z, z_ids = load_coords(os.path.join(rd, "coords.parquet"))
        # pass z_ids through — the exact-alignment contract must hold for the
        # decision scorer too (P0-4), not just the transductive panel.
        panel = score_panel(X, Z, config=cfg, z_ids=z_ids, centroids_by_k=centroids,
                            provenance={"scorer": "complete_panel", "run": os.path.basename(rd),
                                        "coords_sha": _sha_file(os.path.join(rd, "coords.parquet")),
                                        "model_sha": _sha_file(os.path.join(rd, "model.pt"))})
        Xa = X[np.asarray(z_ids, np.int64)] if z_ids is not None else X
        model = ParametricUMAP.load(os.path.join(rd, "model.pt"), device="cuda")
        Zq = np.asarray(model.transform(Xq), dtype=np.float32)
        proj_ffr, proj_rk = projection_ffr(Xa, Z, Xq, Zq, cfg)
        Zq_knn = knn_regress_coords(Xq, Xa, Z, cfg)
        knn_ffr, _ = projection_ffr(Xa, Z, Xq, Zq_knn, cfg)
        lo, hi = Z.min(0), Z.max(0)
        Zq_rand = (rng.rand(len(Xq), Z.shape[1]).astype(np.float32) * (hi - lo) + lo)
        floor_ffr, _ = projection_ffr(Xa, Z, Xq, Zq_rand, cfg)
        summary["runs"][label] = {
            "run_dir": os.path.basename(rd), "wall_s": round(time.time() - t0, 1),
            "ffr": panel["ffr"], "recall@k": panel["recall@k"],
            "purity_k256": (panel.get("purity") or {}).get("k256"),
            "purity_k1024": (panel.get("purity") or {}).get("k1024"),
            "density": panel["density"],
            "proj_ffr": proj_ffr, "proj_recall@k": proj_rk,
            "proj_knn_regressor_ffr": knn_ffr, "proj_random_floor_ffr": floor_ffr,
            "proj_beats_knn": bool(proj_ffr > knn_ffr),
            "proj_margin_over_knn": round(proj_ffr - knn_ffr, 4),
            # P0-4: retain the full panel audit trail, not just scalar leaves.
            "panel_full": panel}
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
