#!/usr/bin/env python3
"""umap-learn 0.6dev on the FULL 2M substrate (CPU) — the direct 2M-table row.

Follow-up to the 500K cross-check (upstream 0.5257 vs our sliced 0.4002):
run upstream at our sandbox scale, scored with the SAME truth + instrument as
every 2M arm (edges-k15-fuzzy, quick-FFR@0.1%, disc=2000). Transductive
asymmetry still applies (it embeds the rows it sees; no reusable encoder, no
projection of new corpora) — that's the trade the basemap program exists to
make, stated on the card.

  fit   (umap06dev-env, CPU): defaults + min_dist 0, seed 42.
  score (latent-basemap .venv): quick-FFR + render + sandbox summary in
        2m-knobs/upstream-06dev-2m/ (external-baselines group).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

SUB = Path("/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
           "minilm-mixed-2m-substrate-and-exact-k15-graph-v1")
OUT = Path("/data/latent-basemap/sandbox/2m-knobs/upstream-06dev-2m")
ROWS = 2_000_000


def fit() -> int:
    import umap

    OUT.mkdir(parents=True, exist_ok=True)
    # a WRITABLE copy — np.asarray over a read-only memmap keeps writeable=False
    # and pynndescent's numba signatures reject readonly arrays (3 GB, fine).
    X = np.array(np.load(SUB / "substrate.f32.npy", mmap_mode="r"))
    t0 = time.time()
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.0, verbose=True,
                        random_state=42)
    xy = reducer.fit_transform(X)
    wall = time.time() - t0
    np.save(OUT / "coordinates.npy", np.asarray(xy, dtype=np.float32))
    (OUT / "fit_info.json").write_text(json.dumps({
        "upstream": f"umap-learn {umap.__version__} (0.6dev clone, HEAD 67ca365)",
        "settings": {"n_neighbors": 15, "min_dist": 0.0, "random_state": 42,
                     "everything_else": "upstream defaults"},
        "device": "cpu",
        "rows": ROWS,
        "wall_s": wall,
    }, indent=1))
    print(f"upstream 0.6dev 2M: {wall/60:.1f} min")
    return 0


def score() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from knobs_2m import quick_ffr
    from map_renders import binned_counts, render_png, robust_extent

    xy = np.load(OUT / "coordinates.npy")
    ffr = quick_ffr(xy, SUB / "edges-k15-fuzzy.npz", ROWS)
    render_png(binned_counts(xy, robust_extent(xy)), OUT / "density.png")
    fi = json.loads((OUT / "fit_info.json").read_text())
    (OUT / "summary.json").write_text(json.dumps({
        "arm": "upstream-06dev-2m",
        "rung": "2m",
        "overrides": {"external_baseline": "umap-learn 0.6dev, CPU, defaults; "
                                           "transductive (no encoder)"},
        "seed": 42,
        "quick_ffr_at_0.1pct": float(ffr),
        "wall_s": fi["wall_s"],
        "substrate": str(SUB / "substrate.f32.npy"),
        "edges": str(SUB / "edges-k15-fuzzy.npz"),
        "note": "sandbox external baseline; not a round, no sealed claim.",
    }, indent=1))
    print(f"upstream-06dev-2m: quick-FFR {ffr:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit({"fit": fit, "score": score}[sys.argv[1]]())
