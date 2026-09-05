"""Growth comparison: random-4M vs sscd-weighted-4M (owner queue #1, 2026-09-05). GPU (project) + CPU (faiss/KDTree).
The "how should a service grow a map" answer — single-variable (addition policy: random vs sscd-rarity-weighted;
identical 2M base). Both heads scored on the COMMON holdout (sscd-clip-4m/test_pos, 200K rows OUTSIDE BOTH 4M
unions -> genuinely unseen by both):
  - held-out FFR (internal): test high-D 15NN among test vs test 2D 0.1%-disc (cap 2000) after projection.
  - reception: test high-D 15NN among the head's TRAINING set vs test 2D disc in the head's map (coordinates.npy).
  - RARE-region coverage: same two metrics on the LOW-sscd_nn (rare) test subset — does the sscd draw improve
    placement of rare/diverse content (the whole point of diversity-weighting)?
  - per-source composition + achieved sscd_nn dist: from the two draw manifests.
Output: sandbox/growth-compare-20260905.json.
Usage: monet_growth_compare.py [NQ=8000].
"""
import json, sys, time
from pathlib import Path
import numpy as np

SB = Path("/data/latent-basemap/sandbox"); POOL = Path("/data2/monet/pool-20m")
COMMON_TEST = Path("/data2/monet/sscd-clip-4m/test-clip.f32.npy")   # common holdout (outside both unions)
TEST_POS = Path("/data2/monet/sscd-clip-4m/test_pos.npy")
HEADS = {
    "random-4m": {"model": SB/"monet-random-clip-4m/champion-bs16k/model.pt", "coords": SB/"monet-random-clip-4m/champion-bs16k/coordinates.npy",
                  "train_hd": Path("/data2/monet/random-clip-4m/clip-substrate.f32.npy"), "manifest": Path("/data2/monet/random-clip-4m/manifest.json")},
    "sscd-4m":   {"model": SB/"monet-sscd-clip-4m/champion-bs16k/model.pt", "coords": SB/"monet-sscd-clip-4m/champion-bs16k/coordinates.npy",
                  "train_hd": Path("/data2/monet/sscd-clip-4m/clip-substrate.f32.npy"), "manifest": Path("/data2/monet/sscd-clip-4m/manifest.json")},
}
K = 15


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x/n).astype(np.float32)


def _recall(q_hd, q_2d, index, tree, disc):
    _, hd = index.search(q_hd, K); _, d2 = tree.query(q_2d, k=disc, workers=-1)
    r = np.empty(q_hd.shape[0])
    for i in range(q_hd.shape[0]):
        ds = set(int(x) for x in d2[i]); r[i] = sum(int(x) in ds for x in hd[i][:K]) / K
    return round(float(r.mean()), 4)


def _project(model_path, X):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _paths import ensure_paths; ensure_paths()
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    return np.asarray(ParametricUMAP.load(str(model_path), device="cuda").transform(X, batch_size=16384), np.float32)


def main():
    nq = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    import faiss
    from scipy.spatial import cKDTree
    for h in HEADS.values():
        for k in ("model", "coords", "train_hd"):
            if not h[k].exists():
                raise SystemExit(f"missing {h[k]} — is the sscd-4M champion done?")

    test = _norm(np.asarray(np.load(COMMON_TEST), np.float32)); nt = test.shape[0]
    sscd_test = np.load(POOL/"sscd_nn.npy")[np.load(TEST_POS)]          # rare split by test sscd_nn (LOW=rare)
    rare_thr = np.nanpercentile(sscd_test, 25)                          # rarest quartile
    rare_mask = (sscd_test <= rare_thr) & ~np.isnan(sscd_test)
    rng = np.random.default_rng(0)
    all_q = np.sort(rng.choice(nt, min(nq, nt), replace=False))
    rare_pos = np.where(rare_mask)[0]; rare_q = np.sort(rng.choice(rare_pos, min(nq, rare_pos.size), replace=False))

    # internal FFR truth: test high-D 15NN among test
    ti = faiss.IndexFlatIP(512); ti.add(test); disc_int = min(max(int(round(nt*0.001)), K), 2000)
    out = {"schema": "growth-compare-random-vs-sscd-4m-2026-09-05", "k": K, "nq": nq,
           "common_holdout": "sscd-clip-4m test set (outside BOTH 4M unions)", "n_test": int(nt),
           "rare_def": f"test sscd_nn <= p25 ({round(float(rare_thr),4)}); n_rare {int(rare_mask.sum())}", "heads": {}}
    for name, h in HEADS.items():
        t0 = time.time()
        t2d = _project(h["model"], test)
        tree_int = cKDTree(np.asarray(t2d, np.float64))
        ffr_all = _recall(test[all_q], np.asarray(t2d[all_q], np.float64), ti, tree_int, disc_int)
        ffr_rare = _recall(test[rare_q], np.asarray(t2d[rare_q], np.float64), ti, tree_int, disc_int)
        ref_hd = _norm(np.asarray(np.load(h["train_hd"], mmap_mode="r"), np.float32))
        ref_2d = np.asarray(np.load(h["coords"]), np.float64)
        idx = faiss.IndexFlatIP(512); idx.add(ref_hd); tree = cKDTree(ref_2d)
        disc = min(max(int(round(ref_hd.shape[0]*0.001)), K), 2000)
        rec_all = _recall(test[all_q], np.asarray(t2d[all_q], np.float64), idx, tree, disc)
        rec_rare = _recall(test[rare_q], np.asarray(t2d[rare_q], np.float64), idx, tree, disc)
        man = json.loads(h["manifest"].read_text())
        add_sscd = man.get("addition_sscd_nn", {}).get("mean") or man.get("sscd_nn_distribution", {}).get("mean")
        out["heads"][name] = {"heldout_ffr": ffr_all, "heldout_ffr_rare": ffr_rare,
                              "reception": rec_all, "reception_rare": rec_rare,
                              "addition_sscd_nn_mean": add_sscd, "wall_s": round(time.time()-t0, 1)}
        print(f"[growth] {name}: FFR {ffr_all} (rare {ffr_rare}) reception {rec_all} (rare {rec_rare}) add_sscd {add_sscd}", flush=True)
        del ref_hd, idx, tree
    r, s = out["heads"]["random-4m"], out["heads"]["sscd-4m"]
    out["deltas_sscd_minus_random"] = {"ffr": round(s["heldout_ffr"]-r["heldout_ffr"], 4),
        "ffr_rare": round(s["heldout_ffr_rare"]-r["heldout_ffr_rare"], 4),
        "reception": round(s["reception"]-r["reception"], 4), "reception_rare": round(s["reception_rare"]-r["reception_rare"], 4)}
    out["verdict"] = ("sscd-weighted growth improves rare-region coverage" if out["deltas_sscd_minus_random"]["reception_rare"] > 0.01
                      else "sscd-weighted growth ~matches random on this holdout (see rare deltas)")
    (SB/"growth-compare-20260905.json").write_text(json.dumps(out, indent=1))
    print(f"[growth] DONE deltas(sscd-random): {out['deltas_sscd_minus_random']} -> {SB/'growth-compare-20260905.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
