#!/usr/bin/env python3
"""Upstream 0.6dev (CPU, transductive) baselines for any pipeline dataset.

Usage: upstream_ds_run.py <dataset> fit|score
Writes sandbox/<dataset>/upstream-06dev/{coordinates.npy,fit_info.json,
density.png,summary.json}. Same truth + instrument as that dataset's arms.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

SANDBOX = Path("/data/latent-basemap/sandbox")


def fit(ds: str) -> int:
    import umap
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from image_map_pipeline import DATASETS, _norm

    d = SANDBOX / ds / "upstream-06dev"
    d.mkdir(parents=True, exist_ok=True)
    if (d / "coordinates.npy").exists():
        print("exists, skip")
        return 0
    X = np.ascontiguousarray(_norm(DATASETS[ds]["load"]()))
    t0 = time.time()
    xy = umap.UMAP(n_neighbors=15, min_dist=0.0, random_state=42,
                   verbose=True).fit_transform(X)
    wall = time.time() - t0
    np.save(d / "coordinates.npy", np.asarray(xy, dtype=np.float32))
    (d / "fit_info.json").write_text(json.dumps({
        "upstream": f"umap-learn {umap.__version__} (0.6dev clone)",
        "device": "cpu", "wall_s": wall,
        "settings": "n_neighbors 15, min_dist 0, seed 42, defaults"}, indent=1))
    print(f"{ds}: upstream fit {wall/60:.1f} min")
    return 0


def score(ds: str) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from knobs_2m import quick_ffr
    from map_renders import binned_counts, render_png, robust_extent

    d = SANDBOX / ds / "upstream-06dev"
    xy = np.load(d / "coordinates.npy")
    edges = SANDBOX / ds / "edges-k15-fuzzy.npz"
    ffr = quick_ffr(xy, edges, xy.shape[0])
    render_png(binned_counts(xy, robust_extent(xy)), d / "density.png")
    fi = json.loads((d / "fit_info.json").read_text())
    (d / "summary.json").write_text(json.dumps({
        "arm": "upstream-06dev", "rung": ds,
        "overrides": {"external_baseline": "umap-learn 0.6dev CPU, "
                                           "transductive (no encoder)"},
        "seed": 42, "wall_s": fi["wall_s"],
        "quick_ffr_at_0.1pct": float(ffr), "edges": str(edges),
        "note": "transductive ceiling reference; no sealed claim.",
    }, indent=1))
    print(f"{ds}/upstream-06dev: quick-FFR {ffr:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit({"fit": fit, "score": score}[sys.argv[2]](sys.argv[1]))
