"""Evolution λ-frontier scorer (5th review, PROMOTED). .venv (scipy). Assembles the (churn, OOD-gain, COST)
frontier over the anchored-fine-tune cells + the two endpoints (w=inf frozen, w=0 full-retrain).

Per map at S3: procrustes-align the anchored S2 rows [0:n2] to their S2 layout, measure residual churn
(the service aligns the new map to the old frame — ls-umap-align); cohort FFRs (overall, T0-retention
[0:n0], reddit-OOD [n2:n3]) via ONE cKDTree + bucketed query pass vs the exact S3 k15 truth; cost in
GPU-minutes. OOD-gain = reddit_FFR - frozen_reddit_FFR. Output: evolbench-lambda-frontier.json.
PROVISIONAL-PENDING-VALIDATION (single seed; noise floor = MiniLM S0-head seed-43 rerun, deferred batch)."""
import json, glob
from pathlib import Path
import numpy as np

import os
SB = Path("/data/latent-basemap/sandbox")
# dims env-parameterized for the D768/jina track (N0=2M, N2=2.8M, N=3.2M) vs MiniLM defaults.
N0 = int(os.environ.get("EVOLBENCH_N0", "4000000"))
N2 = int(os.environ.get("EVOLBENCH_N2", "5600000"))
N = int(os.environ.get("EVOLBENCH_N", "6400000"))
# env-parameterized for the md010 kernel probe: its cells pin to (and churn against) the md010 frozen S2
# layout, its frozen endpoint is the md010 head, and it has no full-retrain endpoint (skipped if absent).
import os
FROZEN_DIR = SB / os.environ.get("EVOLBENCH_FROZEN_DIR", "evolbench-armA-frozen")
LAMBDA_DIR = SB / os.environ.get("EVOLBENCH_LAMBDA_DIR", "lambda")
S2_LAYOUT = FROZEN_DIR / "coords-S2.npy"
S3_KNN = SB / os.environ.get("EVOLBENCH_S3_KNN_DS", "evolbench-S3") / "knn_indices.npy"
RETRAIN_COST_MIN = float(os.environ.get("EVOLBENCH_RETRAIN_COST_MIN", "313.0"))   # S3 head full train (min)
FROZEN_COST_MIN = 0.1      # frozen: placement only (seconds)


def _procrustes(src, ref):
    mu_s = src.mean(0); mu_r = ref.mean(0)
    U, S, Vt = np.linalg.svd((src - mu_s).T @ (ref - mu_r)); R = U @ Vt
    sc = S.sum() / max(((src - mu_s) ** 2).sum(), 1e-9)
    return ((src - mu_s) @ (sc * R) + mu_r)


def _cohort_ffr(xy, knn, cohorts, nq=40000):
    from scipy.spatial import cKDTree
    n = xy.shape[0]; disc = max(int(n * 0.001), 1)
    rng = np.random.default_rng(0); q = rng.choice(n, min(nq, n), replace=False)
    tree = cKDTree(xy); _, near = tree.query(xy[q], k=disc + 1, workers=8)
    hits = {c: 0.0 for c in cohorts}; tot = {c: 0 for c in cohorts}
    for qi in range(len(q)):
        qq = int(q[qi]); tr = np.asarray(knn[qq][:15]); tr = tr[tr != qq]
        if tr.size == 0:
            continue
        ds = set(int(x) for x in near[qi]); ds.discard(qq)
        rec = len(set(int(t) for t in tr) & ds) / tr.size
        for c, (lo, hi) in cohorts.items():
            if lo <= qq < hi:
                hits[c] += rec; tot[c] += 1
    return {c: (round(hits[c] / tot[c], 4) if tot[c] else None) for c in cohorts}


def _score_map(coords_path, s2, knn, cost_min, label):
    import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parent))
    from frame import churn as _frame_churn   # item B: RIGID gauge (no scale collapse) + canonical radius, shared with score_v2
    xy = np.asarray(np.load(coords_path), dtype=np.float32); n = xy.shape[0]
    disp, _ = _frame_churn(xy, s2.astype(np.float64))   # was _procrustes WITH-SCALE (collapsed radius 67.5->0.10)
    coh = _cohort_ffr(xy, knn, {"overall": (0, n), "T0_retention": (0, N0), "reddit": (N2, N)})
    return {"label": label, "n": int(n), "churn_mean": round(float(disp.mean()), 5),
            "churn_p95": round(float(np.percentile(disp, 95)), 5),
            "overall_ffr": coh["overall"], "T0_retention": coh["T0_retention"],
            "reddit_ffr": coh["reddit"], "cost_gpu_min": round(cost_min, 1)}


def main():
    s2 = np.asarray(np.load(S2_LAYOUT), dtype=np.float32)
    knn = np.load(S3_KNN, mmap_mode="r")
    rows = []
    # endpoint: frozen (w=inf)
    rows.append({**_score_map(FROZEN_DIR / "coords-S3.npy", s2, knn, FROZEN_COST_MIN, "w=inf(frozen)"), "w": float("inf")})
    # sweep cells
    def _wkey(p):   # sort by the leading numeric part of the tag; non-numeric tags sort last
        t = Path(p).stem.replace("coords-w", "")
        try:
            return -float(t.split("-")[0].split("_")[0])
        except ValueError:
            return 1.0
    for f in sorted(glob.glob(str(LAMBDA_DIR / "coords-w*.npy")), key=_wkey):
        tag = Path(f).stem.replace("coords-w", "")
        man = json.loads((LAMBDA_DIR / f"manifest-w{tag}.json").read_text())
        cost = float(man.get("train_wall_s", 0)) / 60.0
        try:
            wval = float(tag.split("-")[0].split("_")[0])
        except ValueError:
            wval = tag
        rows.append({**_score_map(f, s2, knn, cost, f"w={tag}"), "w": wval,
                     "warm_start_hash": man.get("warm_start_state_hash"),
                     "trained_hash": man.get("trained_state_hash"), "gen_key": man.get("gen_key")})
    # endpoint: full retrain (w=0). Path env-overridable (D768 = armA-d768-triggered-v2/coords-S3.npy);
    # md010 probe skips it (overlays on md000's). md000 default is the triggered arm.
    _retrain = SB / os.environ.get("EVOLBENCH_RETRAIN_COORDS", "evolbench-armA-triggered/coords-S3.npy")
    if _retrain.is_file() and os.environ.get("EVOLBENCH_SKIP_RETRAIN_ENDPOINT") != "1":
        rows.append({**_score_map(_retrain, s2, knn, RETRAIN_COST_MIN, "w=0(full-retrain)"), "w": 0.0})
    frozen_reddit = rows[0]["reddit_ffr"]
    for r in rows:
        r["ood_gain"] = round((r["reddit_ffr"] or 0) - (frozen_reddit or 0), 4)
    out = {"schema": "evolbench-lambda-frontier-2026-09-01",
           "_chunking": os.environ.get("EVOLBENCH_CHUNKING", ""),   # per-corpus chunk convention (cross-space read)
           "_PROVISIONAL": "PROVISIONAL-PENDING-VALIDATION (single seed; noise floor = MiniLM S0-head seed-43, deferred batch)",
           "note": "Service pricing tiers: placement=seconds (frozen), OOD-absorption=fine-tune minutes (cells), full-retrain=hours.",
           "frontier": rows}
    OUT = SB / os.environ.get("EVOLBENCH_LAMBDA_OUT", "evolbench-lambda-frontier.json"); OUT.write_text(json.dumps(out, indent=1, default=str))
    print("\n=== λ-FRONTIER (churn, OOD-gain, cost) ===", flush=True)
    print(f"{'cell':>16} {'churn':>8} {'reddit':>7} {'OODgain':>8} {'overall':>8} {'T0ret':>7} {'cost_min':>9}", flush=True)
    for r in rows:
        print(f"{r['label']:>16} {r['churn_mean']:>8.4f} {str(r['reddit_ffr']):>7} {r['ood_gain']:>8.4f} "
              f"{str(r['overall_ffr']):>8} {str(r['T0_retention']):>7} {r['cost_gpu_min']:>9.1f}", flush=True)
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
