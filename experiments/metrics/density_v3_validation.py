"""Validation harness for density_v3 (CPU-only).

Scores three map families with both the repaired metric and the legacy
density_v2 statistic, and measures the leave-one-anchor-out sensitivity that
review-0225 identified as the defect.

Families
--------
legacy-2m   round-0217 seeds 42/43/44   (parametric UMAP, "collapsed" visually)
umap-2m     sandbox/2m-knobs umap-md000-x2, umap-dose-x2, umap-kernel
                                        (UMAP low-D kernel, "healthy but foggy")
cuml-1m     sandbox/cuml-1m             (non-parametric cuML UMAP, "clean")

The 2M family shares one substrate, so its high-D radii are computed once and
cached.  The cuML map lives in its own 1M-row universe (a row subsample of the
same substrate) and therefore gets its own radii; its values are NOT strictly
commensurate with the 2M rows — see REPORT-density.md.

Usage:
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 \
      .venv/bin/python experiments/metrics/density_v3_validation.py [--cache DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from density_v3 import (  # noqa: E402
    DEFAULT_N_ANCHORS,
    EPS_HD,
    POOL_FACTOR,
    density_v2_legacy,
    density_v3,
    draw_anchor_pool,
    high_d_radii,
    low_d_radii,
    pearson,
    spearman,
    _loo_block,
    _loo_pearson,
)

SUBSTRATE_2M = (
    "/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
    "minilm-mixed-2m-substrate-and-exact-k15-graph-v1/substrate.f32.npy"
)
PANEL_DIR = (
    "/data/latent-basemap/runs/round-0218/queue/artifacts/"
    "minilm-mixed-2m-seed-family-panel-v1"
)
SEALED_REFERENCE = os.path.join(PANEL_DIR, "minilm-2m-high-d-reference.npz")
KNOBS = "/data/latent-basemap/sandbox/2m-knobs"
CUML = "/data/latent-basemap/sandbox/cuml-1m"

MAPS_2M = [
    ("legacy-2m", "round-0217-seed42", os.path.join(PANEL_DIR, "coordinates-seed42.npy")),
    ("legacy-2m", "round-0217-seed43", os.path.join(PANEL_DIR, "coordinates-seed43.npy")),
    ("legacy-2m", "round-0217-seed44", os.path.join(PANEL_DIR, "coordinates-seed44.npy")),
    ("umap-2m", "umap-md000-x2", os.path.join(KNOBS, "umap-md000-x2/coordinates.npy")),
    ("umap-2m", "umap-dose-x2", os.path.join(KNOBS, "umap-dose-x2/coordinates.npy")),
    ("umap-2m", "umap-kernel", os.path.join(KNOBS, "umap-kernel/coordinates.npy")),
]

N_ANCHORS = DEFAULT_N_ANCHORS
ANCHOR_SEED = 0
THREADS = 8


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def cached_pool_radii(cache_dir: str, tag: str, substrate_path: str,
                      n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    return cached_pool_radii_seed(cache_dir, tag, substrate_path, n_rows,
                                  ANCHOR_SEED)


def cached_pool_radii_seed(cache_dir: str, tag: str, substrate_path: str,
                           n_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Anchor pool + exact high-D radii, cached on disk (map-independent)."""
    path = os.path.join(cache_dir, f"density_v3_hd_{tag}_seed{seed}"
                                   f"_n{N_ANCHORS}.npz")
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as archive:
            return archive["pool_ids"], archive["r_hd"]
    pool = draw_anchor_pool(n_rows, N_ANCHORS, seed, POOL_FACTOR)
    _log(f"computing exact high-D radii: {tag}, pool={len(pool)}, rows={n_rows}")
    started = time.time()
    substrate = np.load(substrate_path, mmap_mode="r", allow_pickle=False)
    radii = high_d_radii(substrate, pool, threads=THREADS)
    _log(f"  done in {time.time() - started:.1f}s; "
         f"degenerate(r_hd<={EPS_HD}) = {int((radii <= EPS_HD).sum())}")
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(path, pool_ids=pool, r_hd=radii)
    return pool, radii


def sealed_population_reproduction(coords_path: str) -> dict:
    """The exact v2 defect: 4,000 anchors, seed 123, sealed r_hd, eps=1e-12."""
    with np.load(SEALED_REFERENCE, allow_pickle=False) as archive:
        anchor_ids = np.asarray(archive["anchor_ids"], dtype=np.int64)
        r_hd = np.asarray(archive["r_hd"], dtype=np.float64)
    coords = np.load(coords_path, mmap_mode="r", allow_pickle=False)
    r_2d = low_d_radii(coords, anchor_ids, threads=THREADS)
    eps = 1e-12
    log_hd, log_2d = np.log(r_hd + eps), np.log(r_2d + eps)
    value = pearson(log_hd, log_2d)
    loo = _loo_pearson(log_hd, log_2d)
    eligible = r_hd > EPS_HD
    return {
        "density_v2_as_defined": value,
        "density_v2_degenerate_dropped": pearson(log_hd[eligible], log_2d[eligible]),
        "spearman_as_defined": spearman(r_hd, r_2d),
        "n_anchors": int(len(anchor_ids)),
        "n_degenerate": int((~eligible).sum()),
        "degenerate_rows": [int(v) for v in anchor_ids[~eligible]],
        "leave_one_out": _loo_block(value, loo),
        "worst_anchor_row": int(anchor_ids[int(np.argmax(np.abs(loo - value)))]),
    }


def score_map(coords_path: str, pool_ids: np.ndarray, r_hd: np.ndarray) -> dict:
    """v3 on its own anchor set, and the v2 statistic on the SAME anchors plus
    the degenerate pool rows v3 excluded — so the only difference between the
    two numbers is the anchor policy plus the log floor."""
    coords = np.load(coords_path, mmap_mode="r", allow_pickle=False)
    v3 = density_v3(
        coords, (pool_ids, r_hd), anchor_seed=ANCHOR_SEED, n_anchors=N_ANCHORS,
        threads=THREADS,
    )
    eligible = r_hd > EPS_HD
    v3_ids = pool_ids[eligible][:N_ANCHORS]
    legacy_ids = np.union1d(v3_ids, pool_ids[~eligible])
    lookup = {int(row): float(value) for row, value in zip(pool_ids, r_hd)}
    legacy_radii = np.array([lookup[int(row)] for row in legacy_ids])
    v2 = density_v2_legacy(
        coords, (legacy_ids, legacy_radii), anchor_seed=ANCHOR_SEED,
        n_anchors=len(legacy_ids), threads=THREADS,
    )
    return {"density_v3": v3, "density_v2_same_pool": v2}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=os.environ.get(
        "DENSITY_V3_CACHE",
        "/tmp/claude-1000/-home-enjalot-code/"
        "af9bdebc-c677-41a4-8156-9eab383e4bbc/scratchpad",
    ))
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "density_v3_results.json"))
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        _log("WARNING: CUDA_VISIBLE_DEVICES is not \"\" (this harness is CPU-only)")

    results: dict = {
        "schema": "density-v3-validation-2026-08-13",
        "anchor_seed": ANCHOR_SEED,
        "n_anchors": N_ANCHORS,
        "eps_hd": EPS_HD,
        "pool_factor": POOL_FACTOR,
        "substrate_2m": f"gsv:{SUBSTRATE_2M}",
        "sealed_reference": f"gsv:{SEALED_REFERENCE}",
        "maps": {},
        "defect_reproduction": {},
    }

    # ── defect reproduction on the original sealed anchor population ───────
    for _, name, path in MAPS_2M:
        _log(f"defect reproduction (4,000 sealed anchors): {name}")
        results["defect_reproduction"][name] = sealed_population_reproduction(path)

    # ── v3 + same-pool v2 on the 2M family ─────────────────────────────────
    pool_2m, r_hd_2m = cached_pool_radii(args.cache, "mixed2m", SUBSTRATE_2M, 2_000_000)
    results["pool_radii_2m"] = {
        "n_pool": int(len(pool_2m)),
        "n_degenerate": int((r_hd_2m <= EPS_HD).sum()),
        "degenerate_rows": [int(v) for v in pool_2m[r_hd_2m <= EPS_HD]],
        "quantiles": {q: float(np.quantile(r_hd_2m, q))
                      for q in (0.0, 0.0001, 0.001, 0.01, 0.5, 0.99, 1.0)},
    }
    for family, name, path in MAPS_2M:
        _log(f"scoring {name} ({family})")
        started = time.time()
        entry = score_map(path, pool_2m, r_hd_2m)
        entry["family"] = family
        entry["coordinates"] = f"gsv:{path}"
        entry["wall_s"] = round(time.time() - started, 1)
        results["maps"][name] = entry
        _log(f"  v3 spearman={entry['density_v3']['spearman']:.4f} "
             f"pearson_log={entry['density_v3']['pearson_log']:.4f} "
             f"| v2(same pool)={entry['density_v2_same_pool']['pearson_log']:.4f} "
             f"({entry['wall_s']}s)")

    # ── cuML reference, own 1M universe ────────────────────────────────────
    pool_1m, r_hd_1m = cached_pool_radii(
        args.cache, "cuml1m", os.path.join(CUML, "emb.f32.npy"), 1_000_000)
    results["pool_radii_1m"] = {
        "n_pool": int(len(pool_1m)),
        "n_degenerate": int((r_hd_1m <= EPS_HD).sum()),
        "degenerate_rows": [int(v) for v in pool_1m[r_hd_1m <= EPS_HD]],
        "note": "1M row subsample of the 2M substrate (rows.npy); own universe",
    }
    _log("scoring cuml-1m")
    started = time.time()
    entry = score_map(os.path.join(CUML, "cuml-xy.npy"), pool_1m, r_hd_1m)
    entry["family"] = "cuml-1m"
    entry["coordinates"] = f"gsv:{CUML}/cuml-xy.npy"
    entry["substrate"] = f"gsv:{CUML}/emb.f32.npy"
    entry["wall_s"] = round(time.time() - started, 1)
    results["maps"]["cuml-1m"] = entry
    _log(f"  v3 spearman={entry['density_v3']['spearman']:.4f} "
         f"pearson_log={entry['density_v3']['pearson_log']:.4f} "
         f"| v2(same pool)={entry['density_v2_same_pool']['pearson_log']:.4f}")

    # ── commensurate control: 2M maps restricted to the cuML 1M universe ───
    # cuml-1m lives in its own row subsample, so its number is not directly
    # comparable above.  Restricting a 2M map to the same 1,000,000 rows and
    # rescoring against the same 1M-universe radii makes it comparable.
    rows = np.load(os.path.join(CUML, "rows.npy"), allow_pickle=False)
    results["commensurate_1m"] = {
        "note": "2M maps restricted to the cuml-1m row subsample "
                f"(gsv:{CUML}/rows.npy); scored against the 1M-universe radii",
        "maps": {},
    }
    for name, path in (
        ("round-0217-seed42", MAPS_2M[0][2]),
        ("umap-md000-x2", MAPS_2M[3][2]),
    ):
        _log(f"commensurate 1M control: {name}")
        coords = np.asarray(np.load(path, mmap_mode="r", allow_pickle=False)[rows])
        entry = density_v3(coords, (pool_1m, r_hd_1m), anchor_seed=ANCHOR_SEED,
                           n_anchors=N_ANCHORS, threads=THREADS)
        results["commensurate_1m"]["maps"][name] = entry
        _log(f"  spearman={entry['spearman']:.4f} "
             f"pearson_log={entry['pearson_log']:.4f}")
    results["commensurate_1m"]["maps"]["cuml-1m"] = results["maps"]["cuml-1m"]["density_v3"]

    # ── knob sensitivity ──────────────────────────────────────────────────
    _log("knob sensitivity: anchor_seed / eps_hd / winsor_q")
    sensitivity: dict = {"anchor_seed": {}, "eps_hd": {}, "winsor_q": {}}
    probe = [("round-0217-seed42", MAPS_2M[0][2]), ("umap-md000-x2", MAPS_2M[3][2])]
    for seed in (0, 1, 2):
        pool_s, r_hd_s = cached_pool_radii_seed(args.cache, "mixed2m",
                                                SUBSTRATE_2M, 2_000_000, seed)
        sensitivity["anchor_seed"][str(seed)] = {}
        for name, path in probe:
            coords = np.load(path, mmap_mode="r", allow_pickle=False)
            entry = density_v3(coords, (pool_s, r_hd_s), anchor_seed=seed,
                               n_anchors=N_ANCHORS, threads=THREADS,
                               leave_one_out=False)
            sensitivity["anchor_seed"][str(seed)][name] = {
                "spearman": entry["spearman"], "pearson_log": entry["pearson_log"],
                "n_excluded": entry["n_excluded_degenerate_hd"]}
    for eps in (1e-6, 1e-4, 1e-3, 1e-2, 1e-1):
        sensitivity["eps_hd"][f"{eps:g}"] = {}
        for name, path in probe:
            coords = np.load(path, mmap_mode="r", allow_pickle=False)
            entry = density_v3(coords, (pool_2m, r_hd_2m), anchor_seed=ANCHOR_SEED,
                               n_anchors=N_ANCHORS, threads=THREADS,
                               eps_hd=eps, leave_one_out=False)
            sensitivity["eps_hd"][f"{eps:g}"][name] = {
                "spearman": entry["spearman"], "pearson_log": entry["pearson_log"],
                "n_excluded": entry["n_excluded_degenerate_hd"]}
    for q in (0.0, 0.0005, 0.001, 0.005, 0.01):
        sensitivity["winsor_q"][f"{q:g}"] = {}
        for name, path in probe:
            coords = np.load(path, mmap_mode="r", allow_pickle=False)
            entry = density_v3(coords, (pool_2m, r_hd_2m), anchor_seed=ANCHOR_SEED,
                               n_anchors=N_ANCHORS, threads=THREADS,
                               winsor_q=q, leave_one_out=True)
            sensitivity["winsor_q"][f"{q:g}"][name] = {
                "spearman": entry["spearman"], "pearson_log": entry["pearson_log"],
                "pearson_log_loo_rel": entry["leave_one_out"]["pearson_log"][
                    "max_relative_shift"],
                "pearson_log_loo_abs": entry["leave_one_out"]["pearson_log"][
                    "max_absolute_shift"]}
    results["sensitivity"] = sensitivity

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=1, sort_keys=True)
    _log(f"wrote {args.out}")

    print("\n" + "=" * 78)
    print(f"{'map':22s} {'family':10s} {'v3 rho':>8s} {'v3 r_log':>9s} "
          f"{'v2 pool':>8s} {'v3 LOO%':>8s} {'v2 LOO%':>8s} {'excl':>5s}")
    print("-" * 78)
    for name, entry in results["maps"].items():
        v3, v2 = entry["density_v3"], entry["density_v2_same_pool"]
        print(f"{name:22s} {entry['family']:10s} "
              f"{v3['spearman']:8.4f} {v3['pearson_log']:9.4f} "
              f"{v2['pearson_log']:8.4f} "
              f"{100 * v3['leave_one_out']['spearman']['max_relative_shift']:8.2f} "
              f"{100 * v2['leave_one_out']['pearson_log']['max_relative_shift']:8.2f} "
              f"{v3['n_excluded_degenerate_hd']:5d}")
    print("=" * 78)
    print("\nDefect reproduction (original 4,000-anchor sealed population):")
    for name, entry in results["defect_reproduction"].items():
        print(f"  {name:22s} v2={entry['density_v2_as_defined']:.4f} "
              f"dropped={entry['density_v2_degenerate_dropped']:.4f} "
              f"LOO max shift={100 * entry['leave_one_out']['max_relative_shift']:.1f}% "
              f"(row {entry['worst_anchor_row']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
