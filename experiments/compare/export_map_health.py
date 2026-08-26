#!/usr/bin/env python3
"""Batch collapse+fog (map-health) for every sandbox arm, incl. upstreams.

Results cache to /data/latent-basemap/sandbox/map-health.json keyed
"<rung>/<arm>" with the coordinates.npy mtime — recomputed only when coords
change. Arm dirs are never written (write-once discipline); the JSON is a
sidecar. The review page and compare exporter read it.

CPU-only (cKDTree on a 20K sample + chunked histogram per collapse_fog's
memory rules). Run standalone or from the review watcher.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from metrics.collapse_fog import map_quality  # noqa: E402

SANDBOX = Path("/data/latent-basemap/sandbox")
OUT = SANDBOX / "map-health.json"


def main() -> int:
    health = {}
    if OUT.exists():
        health = json.loads(OUT.read_text())
    n_new = 0
    for rung_dir in sorted(SANDBOX.iterdir()):
        if not rung_dir.is_dir() or rung_dir.name == "logs":
            continue
        for arm_dir in sorted(rung_dir.iterdir()):
            cf = arm_dir / "coordinates.npy"
            if not arm_dir.is_dir() or not cf.exists():
                continue
            key = f"{rung_dir.name}/{arm_dir.name}"
            mtime = cf.stat().st_mtime
            if key in health and health[key].get("coords_mtime") == mtime:
                continue
            try:
                xy = np.load(cf, mmap_mode="r")
                if xy.ndim != 2 or xy.shape[1] != 2:
                    continue
                t0 = time.time()
                q = map_quality(xy)
                health[key] = {
                    "coords_mtime": mtime,
                    "fog": round(q["fog"]["fog"], 5),
                    "collapse_sqrt_n": round(
                        q["collapse"]["r10_over_radius_times_sqrt_n"], 4),
                    "occupied_bin_fraction": round(
                        q["fog"]["occupied_bin_fraction"], 5),
                }
                n_new += 1
                print(f"{key}: fog {health[key]['fog']:.4f} collapse "
                      f"{health[key]['collapse_sqrt_n']:.3f} occ "
                      f"{health[key]['occupied_bin_fraction']:.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
                if n_new % 10 == 0:
                    OUT.write_text(json.dumps(health, indent=0))
            except Exception as ex:
                print(f"SKIP {key}: {ex}", flush=True)
    OUT.write_text(json.dumps(health, indent=0))
    print(f"{n_new} computed, {len(health)} total -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
