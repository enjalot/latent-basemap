#!/usr/bin/env python3
"""Reddit OOD probe (owner order 2026-08-22): is our training mix costing us
on MiniLM's own home register?

Reddit conversational text is ~62% of all-MiniLM-L6-v2's training data and
near-zero in the fineweb/RPJ/pile mix (PLAN5). This measures what our frozen
maps do to it: project the reddit-2m sample (every 5th of the 10M tldr-17
chunks) through each canonical map and score quick-FFR against reddit's OWN
exact-k15 fuzzy truth (built by image_map_pipeline knn/fuzzy). High OOD-FFR =
the map generalizes to the register; a collapse vs in-corpus numbers = the
case for changing the large-scale training mix.

Outputs per map: /data/latent-basemap/sandbox/reddit-ood/<map>/{summary.json,
density.png (reddit in the MAP's own frame), coordinates.npy}.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REDDIT = Path("/data/latent-basemap/sandbox/reddit-2m")
OUT = Path("/data/latent-basemap/sandbox/reddit-ood")
MAPS = {
    "minilm-2m-r0265-seed42": Path("/data/checkpoints/pumap/maps/minilm-2m-r0265-seed42"),
    "minilm-50m-r0267-seed42": Path("/data/checkpoints/pumap/maps/minilm-50m-r0267-seed42"),
    "minilm-100m-r0268-preview-seed42": Path(
        "/data/checkpoints/pumap/maps/minilm-100m-r0268-preview-seed42"),
    "sandbox-fneg10-tanh4": Path(
        "/data/latent-basemap/sandbox/2m-knobs/umap-md000-x2-fneg10-tanh4"),
    "sandbox-composed-x8": Path(
        "/data/latent-basemap/sandbox/2m-knobs/umap-md000-x8-fneg10-tanh4-pos10"),
}


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from image_map_pipeline import _reddit_load
    from knobs_2m import quick_ffr
    from map_renders import binned_counts, render_png, robust_extent

    from basemap.pumap.parametric_umap.core import ParametricUMAP

    edges = REDDIT / "edges-k15-fuzzy.npz"
    assert edges.exists(), "run image_map_pipeline reddit-2m knn+fuzzy first"
    x = _reddit_load()
    n = x.shape[0]
    print(f"reddit sample: {n:,} rows")

    for map_id, pack in MAPS.items():
        d = OUT / map_id
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
            "probe": "reddit-ood", "map": map_id,
            "rows": n, "quick_ffr_at_0.1pct": float(ffr),
            "wall_s": time.time() - t0,
            "edges": str(edges),
            "note": "reddit-2m projected through the FROZEN map; truth = "
                    "reddit's own exact-k15 fuzzy graph; render in the map's "
                    "own frame (overlays its density.png).",
        }, indent=1))
        print(f"{map_id}: OOD-FFR {ffr:.4f} ({time.time()-t0:.0f}s)")
        del model
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
