#!/usr/bin/env python3
"""make_ls_map_packs.py — canonical map-pack dirs for latent-scope projection.

Builds /data/checkpoints/pumap/maps/<map-id>/ for the maps chosen as visual
benchmarks (plan-gpu-window-2026-08-21.md §5): symlinks to the checkpoint +
training coordinates (sealed round dirs stay untouched), a rendered 1024^2
density underlay, and a summary.json. latent-scope's ls-basemap picks these up
as run-directory packs (checkpoint-by-path) or via registry entries.

CPU-only; re-runnable (skips existing renders). The 50M/100M renders read the
coordinates as strided memmaps.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from map_renders import binned_counts, render_png, robust_extent  # noqa: E402

import numpy as np  # noqa: E402

OUT_ROOT = Path("/data/checkpoints/pumap/maps")
RUNS = Path("/data/latent-basemap/runs")
SANDBOX = Path("/data/latent-basemap/sandbox")

#: id -> (artifact dir, provenance note, evidence status)
MAPS = {
    "minilm-2m-r0265-seed42": (
        RUNS / "round-0265/queue-correction-3/artifacts/"
               "minilm-mixed-2m-fneg-x4-md000-seed42-r0265-v1",
        "sealed R0265 2M family cell, promoted recipe (x4 dose)", "sealed"),
    "minilm-50m-r0267-seed42": (
        RUNS / "round-0267/queue-correction-3/artifacts/"
               "minilm-mixed-50000k-fneg-x2-md000-hostint8-seed42-r0267-v1",
        "sealed R0267 50M staging cell (x2 dose, host-int8)", "sealed"),
    "minilm-100m-r0268-preview-seed42": (
        RUNS / "round-0268/rehearsal/artifacts/"
               "rehearsal-minilm-mixed-100000k-fneg-x2-md000-hostint8-seed42-r0268-v1",
        "100M flagship PREVIEW: byte-identical attempt-4 model via the R11 "
        "rehearsal transform; NON-EVIDENCE until R0268 seals, map itself is "
        "the real 100M projection", "preview-non-evidence"),
    # sandbox aesthetics candidates surface on the review page; add picks here.
}


def build(map_id: str, src: Path, note: str, status: str) -> None:
    model = src / "model.pt"
    coords = src / "coordinates.npy"
    assert model.exists() and coords.exists(), f"{src} incomplete"
    out = OUT_ROOT / map_id
    out.mkdir(parents=True, exist_ok=True)
    for name, target in (("model.pt", model), ("coordinates.npy", coords)):
        link = out / name
        if not link.exists():
            link.symlink_to(target)
    png = out / "density.png"
    if not png.exists():
        xy = np.load(coords, mmap_mode="r")
        render_png(binned_counts(xy, robust_extent(xy)), png)
    rows = int(np.load(coords, mmap_mode="r").shape[0])
    (out / "summary.json").write_text(json.dumps({
        "map_id": map_id,
        "rows": rows,
        "source_artifact": str(src),
        "provenance": note,
        "evidence_status": status,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "note": "latent-scope map pack: symlinked checkpoint + coordinates; "
                "sealed round artifacts untouched at source_artifact.",
    }, indent=1))
    print(f"{map_id}: {rows:,} rows -> {out}")


def main() -> int:
    for map_id, (src, note, status) in MAPS.items():
        build(map_id, src, note, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
