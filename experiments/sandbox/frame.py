"""Shared RIGID coordinate gauge (review-item B, 2026-09-04) — the ONE frame consumed by the evolbench scorer,
the viz exporter, and the chained-growth harness (G). Fixes: (a) sequential Procrustes-WITH-SCALE collapsed the
frame (radius 67.5->0.10) by absorbing structure into shrinking scale; (b) scorer vs exporter normalized by
DIFFERENT radii -> different churn for identical coords. Here: rotation+translation ONLY (no scale applied),
learned scale REPORTED separately, and one canonical radius for normalization. 2D.

Import this everywhere a frame/alignment/churn is needed; never re-implement Procrustes or radius locally."""
import numpy as np


def rigid_align(src, ref, allow_reflection=False):
    """Best-fit RIGID transform (rotation[+optional reflection] + translation, NO scaling) mapping src onto ref.
    Orthogonal Procrustes on centered clouds. Returns (aligned_src[float32], info) where info has R (2x2), t,
    learned_scale (the scale that WOULD minimize residual — reported, NOT applied), and rmsd of the rigid fit."""
    src = np.asarray(src, np.float64); ref = np.asarray(ref, np.float64)
    assert src.shape == ref.shape and src.shape[1] == 2, f"2D same-shape required, got {src.shape} {ref.shape}"
    mu_s = src.mean(0); mu_r = ref.mean(0)
    S = src - mu_s; R_ = ref - mu_r
    M = R_.T @ S                                   # 2x2
    U, sv, Vt = np.linalg.svd(M)
    R = U @ Vt
    if not allow_reflection and np.linalg.det(R) < 0:   # force a proper rotation (no reflection)
        U = U.copy(); U[:, -1] *= -1; R = U @ Vt
    aligned = (R @ S.T).T + mu_r
    denom = float((S ** 2).sum())
    learned_scale = float(sv.sum() / denom) if denom > 1e-12 else 1.0   # optimal-if-allowed scale, REPORTED only
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - ref) ** 2, axis=1))))
    t = mu_r - R @ mu_s
    return aligned.astype(np.float32), {"R": R.tolist(), "t": t.tolist(),
                                        "learned_scale": round(learned_scale, 5), "rmsd": round(rmsd, 5)}


def frame_radius(xy):
    """THE canonical scale for churn normalization — 90th-percentile distance from centroid. Scorer AND exporter
    MUST use this identical function so the same coords give the same churn."""
    xy = np.asarray(xy, np.float64)
    return float(np.percentile(np.linalg.norm(xy - xy.mean(0), axis=1), 90))


def churn(cur, prev, align=True):
    """Per-row displacement of the shared leading rows cur[:len(prev)] vs prev, RIGID-aligned (rotation-invariant)
    then normalized by prev's canonical radius. Returns (disp[n_prev], info). align=False = raw (legacy compare)."""
    m = prev.shape[0]
    cur_m = np.asarray(cur[:m], np.float64)
    if align:
        cur_m, info = rigid_align(cur_m, prev)
    else:
        info = {"R": None, "t": None, "learned_scale": None, "rmsd": None}
    disp = np.linalg.norm(np.asarray(cur_m, np.float64) - np.asarray(prev, np.float64), axis=1)
    disp = disp / max(frame_radius(prev), 1e-9)
    return disp.astype(np.float32), info
