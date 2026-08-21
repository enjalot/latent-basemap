#!/usr/bin/env python3
"""Project the 41.97M wikipedia MiniLM embeddings through the canonical maps.

plan-gpu-window-2026-08-21.md §1.2: runs in minutes on an idle GPU. For each
map pack under /data/checkpoints/pumap/maps/ (+ optional sandbox picks),
transform all wiki rows and render the wiki density in the MAP's own frame
(robust extent of the map's training coordinates), so the render overlays the
map's own density.png 1:1.

Output: /data/latent-basemap/projections/wiki/<map-id>/{coordinates.npy,
density.png, info.json}
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from map_renders import binned_counts, render_png, robust_extent  # noqa: E402

WIKI = sorted(glob.glob(
    "/data/embeddings/wikipedia-en-chunked-120-all-MiniLM-L6-v2/train/*.npy"))
MAPS = {p.name: p for p in Path("/data/checkpoints/pumap/maps").iterdir()
        if (p / "model.pt").exists()}
MAPS["sandbox-md005-fneg10"] = Path(
    "/data/latent-basemap/sandbox/2m-knobs/umap-md005-x2-fneg10")
OUT = Path("/data/latent-basemap/projections/wiki")
BATCH = 65536


def main() -> int:
    from basemap.pumap.parametric_umap.core import ParametricUMAP

    shards = [np.load(f, mmap_mode="r") for f in WIKI]
    total = sum(s.shape[0] for s in shards)
    print(f"{total:,} wiki rows, {len(shards)} shards -> {len(MAPS)} maps")
    for map_id, pack in MAPS.items():
        dest = OUT / map_id
        if (dest / "coordinates.npy").exists():
            print(f"{map_id}: exists, skip")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        model = ParametricUMAP.load(str(pack / "model.pt"), device="cuda")
        t0 = time.time()
        parts = []
        for s in shards:
            for i in range(0, s.shape[0], BATCH):
                parts.append(np.asarray(model.transform(
                    np.asarray(s[i:i + BATCH], dtype=np.float32),
                    batch_size=8192), dtype=np.float32))
        xy = np.concatenate(parts)
        assert xy.shape == (total, 2)
        np.save(dest / "coordinates.npy", xy)
        wall = time.time() - t0
        frame = robust_extent(np.load(pack / "coordinates.npy", mmap_mode="r"))
        render_png(binned_counts(xy, frame), dest / "density.png")
        finite = float(np.isfinite(xy).all(axis=1).mean())
        (dest / "info.json").write_text(json.dumps({
            "map": str(pack), "rows": total, "wall_s": wall,
            "finite_frac": finite,
            "frame": "the MAP's own robust extent (overlays its density.png)",
        }, indent=1))
        print(f"{map_id}: {wall/60:.1f} min, finite {finite:.4f}")
        del model, parts, xy
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
