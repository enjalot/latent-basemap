"""cuVS/cuML full-UMAP at 4M on MONET CLIP-512 — the realistic transductive comparison (owner 2026-09-05).
Runs in cuml-env (launch via /data/latent-basemap/cuml_py). At 512-dim cuML's fast ANN path (nn_descent) is
genuinely usable (unlike the artificial brute-force exact-kNN we forced at 384), so this is the fair fight.

SAME 4M rows as the gated parametric champion (identical substrate slice). Params matched to the md000/competitor
convention: n_neighbors=15, min_dist=0.0, cosine, n_components=2. build_algo left to cuML's default (auto) and
RECORDED (which ANN it picks at 512-dim). Transductive: transform=NO — reception/projection columns are
structurally n/a (noted; that asymmetry rides in every comparison table).

Produces (this script, cuml-env): coords-cuvs-4m-s<seed>.npy + partial manifest (wall, build_algo, n_epochs,
ANN-graph recall@15 vs the EXISTING 4M exact-k15 truth — the 512-dim datapoint for the 0.997@768 / 0.50@384
dimension curve). FFR-v2 + the same-snapshot churn floor (if 2 seeds) are scored afterward by the .venv scorer.

Usage: cuml_py cuvs_4m_umap.py <seed> [tag]
"""
import json, sys, time
from pathlib import Path
import numpy as np

SUB = "/data2/monet/random-clip-4m/clip-substrate.f32.npy"
EXACT_KNN = "/data/latent-basemap/sandbox/monet-random-clip-4m/knn_indices.npy"
OUT = Path("/data/latent-basemap/sandbox/cuvs-4m-20260905")


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    tag = sys.argv[2] if len(sys.argv) > 2 else f"s{seed}"
    OUT.mkdir(parents=True, exist_ok=True)
    from cuml.manifold import UMAP
    import cuml

    X = _norm(np.asarray(np.load(SUB, mmap_mode="r"), np.float32))
    n = X.shape[0]
    kw = dict(n_neighbors=15, min_dist=0.0, n_components=2, metric="cosine",
              build_algo="auto", random_state=seed, verbose=True)
    print(f"[cuvs-4m {tag}] {n:,}x{X.shape[1]} cuml {cuml.__version__} kw={kw}", flush=True)
    t0 = time.time()
    reducer = UMAP(**kw)
    coords = np.asarray(reducer.fit_transform(X), dtype=np.float32)
    wall = time.time() - t0
    np.save(OUT / f"coords-cuvs-4m-{tag}.npy", coords)
    build_algo_used = getattr(reducer, "build_algo", kw["build_algo"])
    n_epochs = getattr(reducer, "n_epochs_", getattr(reducer, "n_epochs", None))
    print(f"[cuvs-4m {tag}] wall {wall:.0f}s ({wall/60:.1f}min) build_algo={build_algo_used} n_epochs={n_epochs}", flush=True)

    # ANN-graph recall@15 vs the EXISTING 4M exact-k15 truth (512-dim datapoint)
    recall = None
    try:
        from cuvs.neighbors import nn_descent
        import cupy as cp
        exact = np.load(EXACT_KNN, mmap_mode="r")           # (n,15) exact cosine k15
        params = nn_descent.IndexParams(graph_degree=64, metric="sqeuclidean")  # normed -> sqeuclidean==cosine order
        g = nn_descent.build(params, cp.asarray(X))
        gi = cp.asarray(g.graph)[:, :15].get()
        # sample rows for recall (full 4M overlap is heavy)
        rng = np.random.default_rng(0); s = rng.choice(n, min(50000, n), replace=False)
        ov = np.mean([len(set(gi[i]) & set(np.asarray(exact[i][:15]))) for i in s]) / 15.0
        recall = round(float(ov), 4)
        print(f"[cuvs-4m {tag}] ANN-graph recall@15 vs exact (512-dim) = {recall}", flush=True)
    except Exception as e:
        print(f"[cuvs-4m {tag}] ANN recall skipped: {e}", flush=True)

    man = {"schema": "cuvs-4m-umap-2026-09-05", "tag": tag, "seed": seed, "n_rows": int(n),
           "cuml_version": cuml.__version__, "params": kw, "build_algo_used": str(build_algo_used),
           "n_epochs": n_epochs, "wall_s": round(wall, 1), "ann_graph_recall_at15_vs_exact_512d": recall,
           "transductive": True, "transform_supported": False,
           "note": "transductive full-UMAP; reception/projection columns structurally n/a. FFR-v2 + churn scored "
                   "by the .venv scorer vs monet-random-clip-4m exact-k15 truth. Compare FFR-v2 to parametric-4M 0.712 (v2).",
           "champion_4m_wall_s_ref": 12480}
    (OUT / f"manifest-{tag}.json").write_text(json.dumps(man, indent=1))
    print(f"[cuvs-4m {tag}] DONE -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
