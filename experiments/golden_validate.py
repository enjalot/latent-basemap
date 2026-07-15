"""Golden 2M validation gate for panel v2 (P0-C exit criterion).

Runs the canonical streamed `score_panel` on the real 2M jina-en corpus + a real
ceiling map, and compares its ffr / recall / density to an INDEPENDENT exact
reference (a different code path: single tiled pass, fp64 accumulation, plain
argsort, no overselect) over the SAME anchors. Records peak GPU/RSS so "bounded"
is an observed property, and writes a checked-in manifest with pre-registered
tolerances and a pass/fail verdict.

Usage:
  python experiments/golden_validate.py \
     --emb  /data/latent-basemap/jina-en-2m/train/data-00000.npy \
     --coords /data/latent-basemap/jina-en-2m/ceiling_umaplearn_k50.parquet \
     --n-anchors 512 --out experiments/golden/golden_2m.json
"""
from __future__ import annotations
import argparse, os, sys, json, time, resource
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from basemap.panel_v2 import (score_panel, PanelV2Config, load_coords, load_embeddings,
                              sample_anchors, ffr_from_neighbors,
                              recall_at_k_from_neighbors, align_x_to_z, cross_knn, _ids_hash)

# Pre-registered tolerances (streamed-vs-independent-fp64-reference). purity uses
# the APPROXIMATE k_frac membership pass so it gets a looser tolerance; ffr/density
# are exact so they get tight ones.
TOL = {"ffr": 0.01, "recall@k": 0.01, "density": 0.02, "purity_k1024": 0.05, "proj_ffr": 0.02}


def _exact_reference(Xa, Z, cfg, centroids=None, proj=None):
    """Independent exact kNN on the panel's anchors — fp64-accumulated tiled
    distances + argsort (no overselect, no expansion rerank). Returns ffr,
    recall@k, density over the SAME anchor set score_panel uses."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n = len(Z)
    aidx = sample_anchors(n, cfg)
    kf = max(cfg.k_hit, int(np.ceil(cfg.frac * n)))

    def knn(F, anchors, k, tile=200_000):
        m = len(anchors); D = F.shape[1]
        ids = np.empty((m, k), np.int64); rad = np.empty((m, k), np.float64)
        Q = torch.from_numpy(np.asarray(F[anchors], np.float32)).double().to(dev)
        best_d = torch.full((m, k + 1), float("inf"), dtype=torch.float64, device=dev)
        best_i = torch.full((m, k + 1), -1, dtype=torch.long, device=dev)
        for j in range(0, len(F), tile):
            Xc = torch.from_numpy(np.asarray(F[j:j + tile], np.float32)).double().to(dev)
            d2 = torch.cdist(Q, Xc) ** 2                     # exact fp64
            kloc = min(k + 1, len(Xc))
            ld, li = torch.topk(d2, kloc, dim=1, largest=False)
            best_d = torch.cat([best_d, ld], 1); best_i = torch.cat([best_i, li + j], 1)
            best_d, sel = torch.topk(best_d, k + 1, dim=1, largest=False)
            best_i = torch.gather(best_i, 1, sel); del Xc, d2
        bi = best_i.cpu().numpy(); bd = best_d.clamp_min(0).sqrt().cpu().numpy()
        for r in range(m):
            keep = bi[r] != anchors[r]
            ids[r] = bi[r][keep][:k]; rad[r] = bd[r][keep][:k]
        return ids, rad

    hi, hr = knn(Xa, aidx, cfg.k_hit)
    hif, _ = knn(Xa, aidx, kf)
    lo, lr = knn(Z, aidx, kf)
    ffr = ffr_from_neighbors(hif[:, :cfg.k_hit], lo, cfg.k_hit)
    r_at = recall_at_k_from_neighbors(hif[:, :cfg.k_hit], lo, cfg.k_hit)
    # density on k_density
    hid, hrd = knn(Xa, aidx, cfg.k_density)
    lod, lrd = knn(Z, aidx, cfg.k_density)
    eps = 1e-12
    dens = float(np.corrcoef(np.log(hrd.mean(1) + eps), np.log(lrd.mean(1) + eps))[0, 1])
    out = {"ffr": round(float(ffr), 4), "recall@k": round(float(r_at), 5),
           "density": round(dens, 4), "n_anchors": int(len(aidx))}
    # purity vs EXACT fp64 k_frac membership (hif/lo above), 1024 granularity
    if centroids is not None and 1024 in centroids:
        import torch
        C = torch.from_numpy(np.asarray(centroids[1024], np.float32)).to(dev)
        lab = np.empty(n, np.int32)
        for i in range(0, n, 65536):
            xb = torch.from_numpy(np.asarray(Xa[i:i + 65536], np.float32)).to(dev)
            lab[i:i + 65536] = torch.cdist(xb, C).argmin(1).cpu().numpy()
        alab = lab[aidx]
        hd = float((lab[hif] == alab[:, None]).mean()); mp = float((lab[lo] == alab[:, None]).mean())
        out["purity_k1024"] = round(mp / hd, 4) if hd else None
    # projection ffr vs fp64 cross-knn (held-out queries already projected)
    if proj is not None:
        Xq, Zq = proj["Xq"], proj["Zq"]
        hiq = _fp64_cross_topk(Xq, Xa, cfg.k_hit, dev)
        loq = _fp64_cross_topk(Zq, Z, kf, dev)
        out["proj_ffr"] = round(float(ffr_from_neighbors(hiq, loq, cfg.k_hit)), 4)
    return out


def _fp64_cross_topk(Q, corpus, k, dev, tile=200_000):
    import torch
    Qt = torch.from_numpy(np.asarray(Q, np.float32)).double().to(dev)
    best_d = torch.full((len(Q), k), float("inf"), dtype=torch.float64, device=dev)
    best_i = torch.full((len(Q), k), -1, dtype=torch.long, device=dev)
    for j in range(0, len(corpus), tile):
        Xc = torch.from_numpy(np.asarray(corpus[j:j + tile], np.float32)).double().to(dev)
        d2 = torch.cdist(Qt, Xc) ** 2
        ld, li = torch.topk(d2, min(k, len(Xc)), dim=1, largest=False)
        best_d = torch.cat([best_d, ld], 1); best_i = torch.cat([best_i, li + j], 1)
        best_d, sel = torch.topk(best_d, k, dim=1, largest=False)
        best_i = torch.gather(best_i, 1, sel); del Xc, d2
    return best_i.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--coords", required=True)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--frac", type=float, default=0.001)
    ap.add_argument("--n-anchors", type=int, default=512)
    ap.add_argument("--testbed", default=None, help="dir with sample_indices.npy (enables purity+projection)")
    ap.add_argument("--source", default=None, help="source shard dir for held-out projection queries")
    ap.add_argument("--model", default=None, help="model.pt to project held-out queries")
    ap.add_argument("--n-holdout", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = PanelV2Config(frac=args.frac, n_anchors=args.n_anchors, corpus_chunk=500_000)
    X = load_embeddings(args.emb, dim=args.dim)
    Z, z_ids = load_coords(args.coords)
    Xa, _, _ = align_x_to_z(X, Z, None, z_ids)

    # purity centroids (frozen k-means) + held-out projection, if a testbed is given
    centroids = proj = projection = None
    if args.testbed:
        from experiments.score_complete_panel import frozen_centroids
        centroids = frozen_centroids(Xa, (256, 1024), args.testbed)
    if args.testbed and args.source and args.model:
        from basemap.pumap.parametric_umap.core import ParametricUMAP
        si = np.load(os.path.join(args.testbed, "sample_indices.npy"))
        src = load_embeddings(args.source, dim=args.dim)
        comp = np.setdiff1d(np.arange(len(src), dtype=np.int64),
                            np.asarray(sorted(set(int(i) for i in si)), np.int64))
        held = np.sort(np.random.RandomState(7).choice(comp, args.n_holdout, replace=False))
        Xq = np.asarray(src[held], dtype=np.float32)
        Zq = np.asarray(ParametricUMAP.load(args.model, device="cuda").transform(Xq), dtype=np.float32)
        proj = {"Xq": Xq, "Zq": Zq}
        projection = {"Xq": Xq, "Zq": Zq, "query_ids": held + 10 ** 9}  # ids disjoint from 0..N
    print(f"[golden] X={X.shape} Z={Z.shape} n_anchors={cfg.n_anchors} frac={cfg.frac} "
          f"purity={'y' if centroids else 'n'} proj={'y' if proj else 'n'}", flush=True)

    t0 = time.time()
    panel = score_panel(X, Z, config=cfg, z_ids=z_ids, centroids_by_k=centroids, projection=projection,
                        provenance={"gate": "golden_2m", "emb": args.emb, "coords": args.coords})
    t_panel = time.time() - t0
    # flatten the streamed metrics we validate
    streamed = {"ffr": panel["ffr"], "recall@k": panel["recall@k"], "density": panel["density"]}
    if centroids:
        streamed["purity_k1024"] = (panel.get("purity") or {}).get("k1024")
    if projection:
        streamed["proj_ffr"] = (panel.get("projection") or {}).get("proj_ffr")
    print(f"[golden] streamed {t_panel:.1f}s: {streamed} peak_gpu_gb={panel['provenance']['peak_gpu_gb']}", flush=True)

    t1 = time.time()
    ref = _exact_reference(Xa, Z, cfg, centroids=centroids, proj=proj)
    t_ref = time.time() - t1
    print(f"[golden] exact reference {t_ref:.1f}s: {ref}", flush=True)

    metrics = [m for m in TOL if streamed.get(m) is not None and ref.get(m) is not None]
    deltas = {m: round(abs(streamed[m] - ref[m]), 5) for m in metrics}
    verdict = {m: bool(deltas[m] <= TOL[m]) for m in metrics}
    passed = all(verdict.values())
    manifest = {
        "gate": "panel_v2_golden_2m", "schema": panel["schema"],
        "formula_version": panel["formula_version"], "passed": passed,
        "config": {"frac": cfg.frac, "n_anchors": cfg.n_anchors, "k_hit": cfg.k_hit,
                   "k_frac": panel["k_frac"], "k_density": cfg.k_density,
                   "overselect": cfg.overselect, "corpus_chunk": cfg.corpus_chunk,
                   "anchor_seed": cfg.anchor_seed},
        "corpus": {"emb": args.emb, "coords": args.coords, "n": panel["n"],
                   "dim_hi": panel["n_dims_hi"], "dim_lo": panel["n_dims_lo"]},
        "metrics_validated": metrics,
        "streamed": {m: streamed[m] for m in metrics},
        "reference": {m: ref[m] for m in metrics},
        "deltas": deltas, "tolerances": {m: TOL[m] for m in metrics}, "per_metric_pass": verdict,
        "exactness": panel["provenance"].get("exactness"),
        "n_holdout": (int(len(proj["Xq"])) if proj else 0),
        "peak_gpu_gb": panel["provenance"]["peak_gpu_gb"],
        "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3),
        "wall_streamed_s": round(t_panel, 1), "wall_reference_s": round(t_ref, 1),
        "code_commit": panel["provenance"]["code_commit"],
        "code_dirty": panel["provenance"]["code_dirty"],
        "guards": panel["guards"],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(manifest, open(args.out, "w"), indent=1)
    print(f"[golden] {'PASS' if passed else 'FAIL'} deltas={deltas} tol={TOL} → {args.out}",
          flush=True)
    sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()
