#!/usr/bin/env python3
"""Generic register OOD probe: project a register sample through frozen maps.

Usage: register_ood_probe.py <dataset>   (reddit-2m | communityarchive-2m | ...)

The dataset must have its truth graph built (image_map_pipeline knn+fuzzy).
Each frozen map projects the sample; quick-FFR vs the register's OWN truth;
render in the map's frame. Outputs: sandbox/<dataset>-ood/<map>/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

SANDBOX = Path("/data/latent-basemap/sandbox")
MAPS = {
    "minilm-2m-r0265-seed42": Path("/data/checkpoints/pumap/maps/minilm-2m-r0265-seed42"),
    "minilm-50m-r0267-seed42": Path("/data/checkpoints/pumap/maps/minilm-50m-r0267-seed42"),
    "minilm-100m-r0268-preview-seed42": Path(
        "/data/checkpoints/pumap/maps/minilm-100m-r0268-preview-seed42"),
    "sandbox-fneg10-tanh4": Path(
        "/data/latent-basemap/sandbox/2m-knobs/umap-md000-x2-fneg10-tanh4"),
    "sandbox-composed-x8": Path(
        "/data/latent-basemap/sandbox/2m-knobs/umap-md000-x8-fneg10-tanh4-pos10"),
    # the redditmix maps join once trained (before/after register comparison)
    "redditmix-promoted": SANDBOX / "minilm-redditmix-2m/promoted-fneg10",
    "redditmix-composed-x8": SANDBOX / "minilm-redditmix-2m/composed-x8",
}


def main() -> int:
    ds = sys.argv[1]
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from image_map_pipeline import DATASETS, _norm
    from knobs_2m import quick_ffr
    from map_renders import binned_counts, render_png, robust_extent

    from basemap.pumap.parametric_umap.core import ParametricUMAP

    edges = SANDBOX / ds / "edges-k15-fuzzy.npz"
    assert edges.exists(), f"no truth graph for {ds}"
    x = _norm(DATASETS[ds]["load"]())
    n = x.shape[0]
    print(f"{ds}: {n:,} rows")
    out_root = SANDBOX / f"{ds}-ood"

    for map_id, pack in MAPS.items():
        if not (pack / "model.pt").exists():
            print(f"{map_id}: no model, skip")
            continue
        d = out_root / map_id
        if (d / "summary.json").exists():
            print(f"{map_id}: done, skip")
            continue
        d.mkdir(parents=True, exist_ok=True)
        model = ParametricUMAP.load(str(pack / "model.pt"), device="cuda")
        t0 = time.time()
        xy = np.asarray(model.transform(x, batch_size=8192), dtype=np.float32)
        np.save(d / "coordinates.npy", xy)
        ffr = quick_ffr(xy, edges, n)
        frame = robust_extent(np.load(pack / "coordinates.npy", mmap_mode="r"))
        render_png(binned_counts(xy, frame), d / "density.png")
        (d / "summary.json").write_text(json.dumps({
            "probe": f"{ds}-ood", "map": map_id, "rows": n,
            "quick_ffr_at_0.1pct": float(ffr), "wall_s": time.time() - t0,
            "edges": str(edges),
        }, indent=1))
        print(f"{map_id}: OOD-FFR {ffr:.4f}")
        del model
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
