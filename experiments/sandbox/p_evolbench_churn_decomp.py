"""Churn decomposition metric (owner investigation program, overseer 2026-09-02). CPU. Owner: "movement from
cluster points shifting is more OK than moving cluster members — qualify this." Formalize with k-means (k≈1000)
on the FROZEN 2D map; per point Δ_i = pos_w − pos_frozen (procrustes-aligned); cluster translation T_c = mean
Δ over the cluster. Decompose (exact energy identity, cross-term vanishes per cluster since Σ(Δ_i−T_c)=0):
    Σ‖Δ_i‖² = Σ‖T_c(i)‖²  (COHERENT, rigid cluster motion)  +  Σ‖Δ_i−T_c(i)‖²  (INTERNAL, member reshuffle).
Plus MEMBERSHIP churn = fraction whose nearest frozen centroid changes between its frozen and w position.
Report the coherent/internal split per w across the ladder, per cohort (base [0:N2] vs reddit [N2:N]).
Prediction: anchored cells' tiny churn is mostly COHERENT; the retrain's 0.37 is heavily INTERNAL.
Usage: p_evolbench_churn_decomp.py [k]   (default k=1000). Output: evolbench-churn-decomp.json"""
import json, sys, glob, os
from pathlib import Path
import numpy as np

SB = Path("/data/latent-basemap/sandbox")
# dims env-parameterized for the D768 track (T0=2M, tranche 400k -> N2=2.8M, N=3.2M) vs MiniLM defaults.
N2 = int(os.environ.get("EVOLBENCH_N2", "5600000")); N = int(os.environ.get("EVOLBENCH_N", "6400000"))
KDEF = 1000
FROZEN = SB / os.environ.get("EVOLBENCH_DECOMP_FROZEN", "evolbench-armA-frozen/coords-S3.npy")
TRIGGERED = SB / os.environ.get("EVOLBENCH_DECOMP_TRIGGERED", "evolbench-armA-triggered/coords-S3.npy")
# EVOLBENCH_DECOMP_CELLS: optional json {tag: coords_path_rel_to_SB}. When set, decompose exactly these
# (D768 uses {"triggered": armA-d768-triggered-v2/coords-S3.npy, "armB": evolbench-d768-armB/coords-S3.npy})
# vs FROZEN; otherwise the default md000 ladder (lambda cells + the retrain endpoint).
_CELLS_ENV = os.environ.get("EVOLBENCH_DECOMP_CELLS", "")


def _procrustes_fit(src, ref):
    mu_s = src.mean(0); mu_r = ref.mean(0)
    U, S, Vt = np.linalg.svd((src - mu_s).T @ (ref - mu_r)); R = U @ Vt
    sc = S.sum() / max(((src - mu_s) ** 2).sum(), 1e-9)
    return mu_s, (sc * R), mu_r


def _apply(xy, fit):
    mu_s, scR, mu_r = fit
    return (xy - mu_s) @ scR + mu_r


def _decompose(xy_w, xy_f, cl, cent_f, rad):
    """Return coherent/internal energy split + membership churn, overall and per cohort."""
    fit = _procrustes_fit(xy_w[:N2].astype(np.float64), xy_f[:N2].astype(np.float64))
    wa = _apply(xy_w.astype(np.float64), fit).astype(np.float32)
    D = wa - xy_f                                                   # per-point displacement
    K = cent_f.shape[0]
    # cluster translation T_c = mean Δ over members
    Tc = np.zeros((K, 2), np.float64); cnt = np.zeros(K, np.int64)
    np.add.at(Tc, cl, D.astype(np.float64)); np.add.at(cnt, cl, 1)
    Tc /= np.maximum(cnt[:, None], 1)
    Tci = Tc[cl].astype(np.float32)
    coh = np.einsum("ij,ij->i", Tci, Tci)                          # ‖T_c‖² per point
    intn = np.einsum("ij,ij->i", D - Tci, D - Tci)                 # ‖Δ−T_c‖² per point
    # membership churn: nearest frozen centroid at frozen-pos (=cl) vs at w-pos
    from scipy.spatial import cKDTree
    tree = cKDTree(cent_f)
    _, cl_w = tree.query(wa, k=1, workers=8)
    changed = (cl_w != cl)

    def _split(mask):
        c = float(coh[mask].sum()); i = float(intn[mask].sum()); tot = c + i
        return {"n": int(mask.sum()),
                "churn_mean": round(float(np.linalg.norm(D[mask], axis=1).mean()) / max(rad, 1e-9), 5),
                "coherent_frac": round(c / tot, 4) if tot else None,
                "internal_frac": round(i / tot, 4) if tot else None,
                "membership_churn": round(float(changed[mask].mean()), 4)}
    base = np.zeros(len(D), bool); base[:N2] = True
    red = ~base
    return {"overall": _split(np.ones(len(D), bool)),
            "base": _split(base), "reddit": _split(red)}


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else KDEF
    xy_f = np.asarray(np.load(FROZEN), np.float32)
    rad = float(np.percentile(np.linalg.norm(xy_f - xy_f.mean(0), axis=1), 90))
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=10000, n_init=3, max_iter=100)
    cl = km.fit_predict(xy_f); cent_f = km.cluster_centers_.astype(np.float32)

    # cells to decompose: explicit set (D768 arms) or the default md000 ladder (λ cells + retrain endpoint)
    if _CELLS_ENV:
        cells = [(tag, np.asarray(np.load(SB / rel), np.float32)) for tag, rel in json.loads(_CELLS_ENV).items()]
    else:
        cells = [(Path(f).stem.replace("coords-w", ""), np.asarray(np.load(f), np.float32))
                 for f in sorted(glob.glob(str(SB / "lambda/coords-w*.npy")))]
        cells.append(("0(retrain)", np.asarray(np.load(TRIGGERED), np.float32)))
    out = {"schema": "evolbench-churn-decomp-2026-09-02", "k": k, "frozen_radius_p90": round(rad, 4),
           "dims": {"N2": N2, "N": N}, "frozen": str(FROZEN.relative_to(SB)),
           "note": "coherent=rigid cluster translation; internal=member reshuffle; energy-exact split.",
           "cells": {}}
    print(f"=== CHURN DECOMPOSITION (k={k}, N2={N2:,} N={N:,}) ===", flush=True)
    print(f"{'cell':>12} {'cohort':>7} {'churn':>8} {'coh%':>7} {'int%':>7} {'memb%':>7}", flush=True)
    for tag, xy_w in cells:
        dec = _decompose(xy_w, xy_f, cl, cent_f, rad)
        out["cells"][tag] = dec
        for coh in ("overall", "base", "reddit"):
            d = dec[coh]
            print(f"{tag:>12} {coh:>7} {d['churn_mean']:>8.4f} "
                  f"{(d['coherent_frac'] or 0)*100:>6.1f}% {(d['internal_frac'] or 0)*100:>6.1f}% "
                  f"{d['membership_churn']*100:>6.1f}%", flush=True)
    outp = SB / os.environ.get("EVOLBENCH_DECOMP_OUT", "evolbench-churn-decomp.json")
    outp.write_text(json.dumps(out, indent=1, default=str))
    print(f"wrote {outp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
