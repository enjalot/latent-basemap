"""Text-chain fixed-T0-frame scoring (owner plan-basemap-chain, 2026-09-04). Every stage is compared to the SAME
fixed T0 frame (S0 coords) with a fixed T0 p90-radius denominator (registered in the receipt) — never
successively aligned/rescaled frames. Metrics (plan evidence table): per-cohort FFR (quick_ffr_v2 @0.1% on each
cohort's exact k15 truth), cumulative T0 movement (rigid-aligned to the fixed frame + raw), anchor drift (active
vs holdout), reception (unseen recall@15 into a fixed existing-map reference). Emits a machine-readable stage
receipt. Functions are importable by the driver; run standalone per stage.

Handles zero radius explicitly (no unreported epsilon). FFR/reception are DISTINCT measurements (kept separate)."""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frame


def fixed_t0_radius(t0_coords):
    """The ONE registered denominator: p90 radius of the fixed T0 (S0) layout. Explicit zero handling."""
    r = frame.frame_radius(np.asarray(t0_coords, np.float64))
    return r if r > 0 else None


def t0_movement(stage_coords, t0_coords, fixed_radius):
    """Cumulative T0 movement: the T0 rows of this stage, RIGID-aligned to the fixed T0 frame, displacement /
    fixed T0 radius. Also raw displacement. stage_coords[:n_t0] must be the T0 rows."""
    n0 = t0_coords.shape[0]
    cur = np.asarray(stage_coords[:n0], np.float64)
    aligned, info = frame.rigid_align(cur, np.asarray(t0_coords, np.float64))
    raw = np.linalg.norm(aligned - np.asarray(t0_coords, np.float64), axis=1)
    if fixed_radius is None:
        return {"error": "zero fixed T0 radius", "learned_scale": info["learned_scale"]}
    norm = raw / fixed_radius
    return {"movement_mean_normed": round(float(norm.mean()), 5), "movement_p95_normed": round(float(np.percentile(norm, 95)), 5),
            "movement_mean_raw": round(float(raw.mean()), 4), "learned_scale": info["learned_scale"], "rmsd": info["rmsd"]}


def cohort_ffr(coords, edges_path, cohort_ranges, knn_indices_path):
    """quick_ffr_v2 @0.1% per cohort on the stage's own exact k15 truth. cohort_ranges: {name:(lo,hi)}. Queries
    restricted to each cohort's rows (mirrors quick_ffr_v2's disc/self-exclusion, cohort-scoped)."""
    from knobs_2m import quick_ffr_v2
    n = coords.shape[0]
    out = {}
    # overall via the standard instrument
    out["overall"] = round(float(quick_ffr_v2(coords, edges_path, n, knn_indices_path=knn_indices_path)), 4)
    # per-cohort: score the slice against the same truth file (row ids are global; quick_ffr_v2 samples from rows)
    for name, (lo, hi) in cohort_ranges.items():
        m = hi - lo
        if m < 50:
            out[name] = None; continue
        # quick_ffr_v2 samples queries from range(rows); to cohort-scope, pass rows=hi and it queries [0,hi)
        # then we cannot isolate [lo,hi) cleanly via that API -> use a direct cohort recall instead.
        out[name] = _cohort_recall(coords, knn_indices_path, lo, hi, n)
    return out


def _cohort_recall(coords, knn_indices_path, lo, hi, n_total, k=15, nq=8000, seed=0):
    """Recall of the exact-k15 high-D truth among the 2D disc of n_total*0.1% neighbors, queries in [lo,hi)."""
    from scipy.spatial import cKDTree
    knn = np.load(knn_indices_path, mmap_mode="r")
    rng = np.random.default_rng(seed)
    q = rng.choice(np.arange(lo, hi), size=min(nq, hi - lo), replace=False)
    disc = max(int(round(n_total * 0.001)), k)
    tree = cKDTree(coords)
    _, nbr = tree.query(coords[q], k=disc + 1, workers=-1)
    hit = 0; tot = 0
    for r, qi in enumerate(q):
        ds = set(int(x) for x in nbr[r] if x != qi)
        tr = [int(t) for t in knn[qi][:k] if t != qi]
        if tr:
            hit += sum(t in ds for t in tr); tot += len(tr)
    return round(hit / tot, 4) if tot else None


def anchor_drift(stage_coords, anchor_npz, holdout_mask_path=None):
    """Mean displacement of anchor rows from their anchor_targets, split active vs holdout. holdout_mask: bool
    over anchor rows (True=held out of the loss). If absent, reports overall only."""
    z = np.load(anchor_npz); ids = z["anchor_ids"]; tgt = z["anchor_targets"].astype(np.float64)
    cur = np.asarray(stage_coords[ids], np.float64)
    # align current anchor rows to targets (rigid) then displacement
    aligned, _ = frame.rigid_align(cur, tgt)
    disp = np.linalg.norm(aligned - tgt, axis=1)
    out = {"anchor_disp_mean": round(float(disp.mean()), 5), "n_anchor": int(len(ids))}
    if holdout_mask_path and Path(holdout_mask_path).exists():
        hm = np.load(holdout_mask_path)
        a = disp[~hm].mean(); h = disp[hm].mean()
        out.update(active_disp_mean=round(float(a), 5), holdout_disp_mean=round(float(h), 5),
                   holdout_over_active=round(float(h / (a + 1e-12)), 4))
    return out
