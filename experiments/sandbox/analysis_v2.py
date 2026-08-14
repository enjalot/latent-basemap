"""P1 analysis-v2 — fneg collapse-drift, leveling-capable JOINT fit (FROZEN).

Owner-approved 2026-08-13 (option a). This file is the pre-registration:
committed BEFORE the 25M x2 and 12.5M x4 cells exist, so the functional form,
the joint-fit structure, the bootstrap band, the fixed seed, and the decision
rule below are frozen, not chosen after seeing the data. Do not edit the model,
seed, or rule after the new cells' collapse values are readable.

Spec: gsv:/data/latent-basemap/sandbox/logs/analysis-v2-prereg-draft.md
Supersedes analysis-v1 (decomposition_go_no_go.py, linear-in-log10 N) for the
go/no-go — v1's linear form is mis-specified by the non-monotone x2 recovery
(2M 1.190 -> 6.25M 0.935 -> 12.5M 0.993). v1's NO-GO stands as published.

MODEL (per point, dose d in {x2,x4}, rows N, x=log10 N, x0=log10 2e6):
    y(N;d) = yinf_d + (y0_d - yinf_d) * exp(-lambda * (x - x0))
  ONE shared lambda across doses (same mechanism/substrate family); per-dose
  y0_d, yinf_d. Params = [lambda, y0_x2, yinf_x2, y0_x4, yinf_x4] (5).
  Fit set: x2 at 2M/6.25M/12.5M/25M (4) + x4 at 2M/6.25M/12.5M (3) = 7 points,
  2 residual dof; the x4 arm alone keeps 1 dof, so no zero-width interpolation.

BAND: nonparametric residual bootstrap, B=10000, SEED=20260813 (fixed). Pool
the 7 residuals, resample with replacement, refit the joint model, record
yinf_x2, yinf_x4, and the contrast (yinf_x4 - yinf_x2). 95% band = [2.5,97.5]
percentiles. Disclosed caveat: a bootstrap over 7 points (2 dof) is fragile —
honest-best-effort at this n, not an asymptotic guarantee; R0264's N-rule
inherits this caveat.

DECISION (frozen): GO iff yinf_x4 band lower >= 0.9 AND yinf_x2 band lower
>= 0.9 (the healthy floor; the asymptote, not a 100M point, is the gate).
Else NO-GO. Dose-interaction report: contrast band excludes 0 => dose shifts
the asymptote; covers 0 => x2 corroborates x4.
"""
import sys, math, json
sys.path.insert(0, "/home/enjalot/code/latent-basemap")
sys.path.insert(0, "/home/enjalot/code/latent-basemap/experiments")
import numpy as np
from scipy.optimize import least_squares
from metrics.collapse_fog import map_quality

SB = "/data/latent-basemap/sandbox"
X0 = math.log10(2_000_000)
FLOOR = 0.9
SEED = 20260813
B = 10000

CELLS = {  # (dose, N): coords path
    ("x2", 2_000_000):  f"{SB}/2m-knobs/umap-md000-x2-fneg10/coordinates.npy",
    ("x2", 6_250_000):  f"{SB}/6250k-knobs/umap-md000-x2-fneg10/coordinates.npy",
    ("x2", 12_500_000): f"{SB}/12500k-knobs/umap-md000-x2-fneg10/coordinates.npy",
    # 25M is the host-int8 arm: fp32 X (35.76 GiB) OOMs the 32 GB card, so >20M
    # rungs are int8-path (P5 passed: 2M int8-vs-fp32 delta collapse -0.040/fog
    # -0.052, < seed variation). Path corrected 2026-08-14 BEFORE the 25M int8
    # coords existed (prospective, not fit-to-data); fit form/band/seed/rule
    # byte-unchanged. The 6.25M int8-vs-fp32 delta is published beside the verdict.
    ("x2", 25_000_000): f"{SB}/25000k-knobs/umap-md000-x2-fneg10-hostint8/coordinates.npy",
    ("x4", 2_000_000):  f"{SB}/2m-knobs/umap-md000-x4-fneg10/coordinates.npy",
    ("x4", 6_250_000):  f"{SB}/6250k-knobs/umap-md000-x4-fneg10/coordinates.npy",
    ("x4", 12_500_000): f"{SB}/12500k-knobs/umap-md000-x4-fneg10/coordinates.npy",
}
# param vector p = [lambda, y0_x2, yinf_x2, y0_x4, yinf_x4]
PIDX = {"x2": (1, 2), "x4": (3, 4)}


def collapse(path):
    return map_quality(np.load(path, mmap_mode="r"))["collapse"]["r10_over_radius_times_sqrt_n"]


def model(p, dose, N):
    lam = p[0]; i0, iinf = PIDX[dose]
    return p[iinf] + (p[i0] - p[iinf]) * math.exp(-lam * (math.log10(N) - X0))


def residuals(p, pts):
    return [model(p, d, N) - y for (d, N, y) in pts]


def fit(pts):
    p0 = [0.5, 1.2, 0.95, 1.1, 0.95]
    lb = [0.0, -np.inf, -np.inf, -np.inf, -np.inf]
    ub = [np.inf] * 5
    r = least_squares(residuals, p0, args=(pts,), bounds=(lb, ub), max_nfev=10000)
    return r.x, np.array(r.fun)


def main():
    pts = []
    for (d, N), path in CELLS.items():
        try:
            pts.append((d, N, collapse(path)))
        except FileNotFoundError:
            print(f"MISSING {d}@{N}: {path}")
    have = {(d, N) for (d, N, _) in pts}
    need = set(CELLS)
    if have != need:
        print(f"\nHave {len(have)}/7 cells; missing {sorted(need - have)}. "
              "Re-run when both new cells land.")
        return 1

    p, resid = fit(pts)
    lam, y0x2, yinfx2, y0x4, yinfx4 = p
    print("x2 curve:", sorted([(N, round(y, 3)) for (d, N, y) in pts if d == "x2"]))
    print("x4 curve:", sorted([(N, round(y, 3)) for (d, N, y) in pts if d == "x4"]))
    print(f"\nJOINT FIT: lambda={lam:.4f}  yinf_x2={yinfx2:.4f}  yinf_x4={yinfx4:.4f}  "
          f"(y0_x2={y0x2:.4f} y0_x4={y0x4:.4f}); resid_rms={math.sqrt((resid**2).mean()):.4f}")

    rng = np.random.default_rng(SEED)
    yhat = np.array([model(p, d, N) for (d, N, _) in pts])
    boot = {"yinf_x2": [], "yinf_x4": [], "contrast": []}
    fails = 0
    for _ in range(B):
        e = rng.choice(resid, size=len(resid), replace=True)
        yb = yhat + e
        bpts = [(d, N, yb[i]) for i, (d, N, _) in enumerate(pts)]
        try:
            pb, _ = fit(bpts)
        except Exception:
            fails += 1; continue
        boot["yinf_x2"].append(pb[2]); boot["yinf_x4"].append(pb[4])
        boot["contrast"].append(pb[4] - pb[2])

    def band(key):
        a = np.array(boot[key]); return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))
    lo2, hi2 = band("yinf_x2"); lo4, hi4 = band("yinf_x4"); loc, hic = band("contrast")
    print(f"\nbootstrap B={B} seed={SEED} (failed fits: {fails})")
    print(f"  yinf_x2 = {yinfx2:.3f}  band [{lo2:.3f}, {hi2:.3f}]")
    print(f"  yinf_x4 = {yinfx4:.3f}  band [{lo4:.3f}, {hi4:.3f}]   <- promoted config")
    print(f"  contrast (x4-x2) = {yinfx4-yinfx2:+.3f}  band [{loc:.3f}, {hic:.3f}]  "
          f"({'excludes 0: dose shifts asymptote' if (loc>0 or hic<0) else 'covers 0: x2 corroborates x4'})")

    go = (lo4 >= FLOOR) and (lo2 >= FLOOR)
    print(f"\nVERDICT: {'GO' if go else 'NO-GO'}  "
          f"(yinf_x4 lower {lo4:.3f} {'>=' if lo4>=FLOOR else '<'} {FLOOR}; "
          f"yinf_x2 lower {lo2:.3f} {'>=' if lo2>=FLOOR else '<'} {FLOOR})")
    out = {"fit": {"lambda": lam, "yinf_x2": yinfx2, "yinf_x4": yinfx4,
                   "y0_x2": y0x2, "y0_x4": y0x4, "resid_rms": float(math.sqrt((resid**2).mean()))},
           "points": [(d, N, y) for (d, N, y) in pts],
           "bands": {"yinf_x2": [lo2, hi2], "yinf_x4": [lo4, hi4], "contrast": [loc, hic]},
           "bootstrap": {"B": B, "seed": SEED, "failed_fits": fails},
           "floor": FLOOR, "verdict": "GO" if go else "NO-GO",
           "caveat": "bootstrap over 7 points (2 dof) is fragile; honest-best-effort at this n"}
    json.dump(out, open(f"{SB}/logs/analysis_v2_result.json", "w"), indent=1)
    print(f"\nwrote {SB}/logs/analysis_v2_result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
