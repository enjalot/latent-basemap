"""P4 — durable kNN-regressor cost evidence (closure review). The NUMAP baseline's
cost must be measured, not assumed free. Our kNN-regressor uses a ZERO-BUILD
streamed brute search: the training X itself is the reference (no separate index
to build/add/load), so the cost is entirely query + coordinate regression + the
memory footprint of holding X. Records that contract explicitly and contrasts it
with the parametric map's constant-time transform.

Runs on GPU (needs a held lease).

Usage (via a held GpuLease / controller):
  python experiments/measure_knn_cost.py --out experiments/evidence/r1_rescore/knn_cost.json
"""
from __future__ import annotations
import argparse, os, sys, json, time, resource
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from basemap.panel_v2 import PanelV2Config, load_embeddings, cross_knn, load_coords, _ids_hash

SRC = "/data/embeddings/fineweb-edu-sample-10BT-chunked-500-jina-v5-nano/train"


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-holdout", type=int, default=20000)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--out", default="experiments/evidence/r1_rescore/knn_cost.json")
    args = ap.parse_args()
    cfg = PanelV2Config(corpus_chunk=500_000)
    rec = {"contract": "zero-build streamed brute hi-D search: the training X IS the reference "
                       "index (no separate build/add/load); cost = query + coordinate regression + "
                       "the memory to hold X.",
           "k": args.k, "testbeds": {}}
    for name, tb, dim in [("200k", "/data/latent-basemap/jina-en-200k", 768),
                          ("2m", "/data/latent-basemap/jina-en-2m", 768)]:
        X = load_embeddings(os.path.join(tb, "train"), dim=dim)
        si = np.load(os.path.join(tb, "sample_indices.npy"))
        src = load_embeddings(SRC, dim=dim)
        comp = np.setdiff1d(np.arange(len(src), dtype=np.int64),
                            np.asarray(sorted(set(int(i) for i in si)), np.int64))
        held = np.sort(np.random.RandomState(123).choice(comp, args.n_holdout, replace=False))
        Xq = np.asarray(src[held], dtype=np.float32)
        import glob
        rd = sorted(glob.glob(f"experiments/results/r1_kernel_{'2m_' if name=='2m' else ''}legacy_a1b1_s42_*"))[-1]
        Z, zid = load_coords(os.path.join(rd, "coords.parquet"))
        Xa = X[np.asarray(zid, np.int64)] if zid is not None else X
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        nb = cross_knn(Xq, Xa, args.k, cfg, hi_dim=True)      # the query step
        q_s = time.time() - t0
        t1 = time.time()
        Zq = Z[nb].mean(axis=1)                               # coordinate regression
        e2e = (time.time() - t1) + q_s
        rec["testbeds"][name] = {
            "n_train": int(len(Xa)), "n_query": int(args.n_holdout),
            "index_build_s": 0.0, "index_note": "no index — streamed brute search over X",
            "query_wall_s": round(q_s, 2), "queries_per_s": round(args.n_holdout / q_s, 1),
            "end_to_end_regression_wall_s": round(e2e, 2),
            "end_to_end_per_s": round(args.n_holdout / e2e, 1),
            "x_footprint_gb": round(len(Xa) * dim * (2 if getattr(X, "dtype", None) == np.float16 else 4) / 1e9, 3),
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else None,
            "held_hash": _ids_hash(held), "run_dir": os.path.basename(rd),
            "vs_parametric_transform": "parametric map = constant-time forward pass (~430k rows/s, "
                                       "14 MB checkpoint); kNN-regressor query scales O(n_train × D).",
        }
        print(f"[knn_cost] {name}: query {q_s:.2f}s ({args.n_holdout/q_s:.0f} q/s), e2e {e2e:.2f}s, "
              f"X {rec['testbeds'][name]['x_footprint_gb']} GB", flush=True)
    rec["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(rec, open(args.out, "w"), indent=1)
    print(f"[knn_cost] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
