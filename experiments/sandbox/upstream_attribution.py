#!/usr/bin/env python3
"""Upstream 0.6dev attribution panel at 2M (reviewer recommendation 2026-08-22).

Our ports of upstream mechanisms were approximations (loss-space tanh vs
grad-space clip; per-trainer-epoch rank refresh vs upstream's cadence; ±W
window = 2x upstream's total range; annealing != recursive init). Before
porting more faithfully, attribute upstream's 0.4798 to its components by
ABLATING upstream itself — each run ~3 min CPU, zero GPU:

  default        (recursive init, adam, range=200K, adaptive neg scale) = 0.4798
  init-random    init="random"
  init-spectral  init="spectral"
  neg-uniform    negative_selection_range=None (uniform negatives)
  neg-fixedscale negative_sample_scale=1.0 (adaptive force calibration OFF)
  optimizer-sgd  optimizer="sgd" (pre-0.6 compatibility optimizer)

All runs share random_state=42 and identical data -> identical kNN stage, so
differences attribute to the ablated component. Scored with our quick-FFR on
the sealed 2M truth. Outputs: sandbox/2m-knobs/upstream-ablate-<name>/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

SUB = Path("/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
           "minilm-mixed-2m-substrate-and-exact-k15-graph-v1")
OUT_ROOT = Path("/data/latent-basemap/sandbox/2m-knobs")

ABLATIONS = {
    "upstream-ablate-init-random": {"init": "random"},
    "upstream-ablate-init-spectral": {"init": "spectral"},
    "upstream-ablate-neg-uniform": {"negative_selection_range": None},
    "upstream-ablate-neg-fixedscale": {"negative_sample_scale": 1.0},
    "upstream-ablate-optimizer-sgd": {"optimizer": "sgd"},
}


def main() -> int:
    import umap

    X = np.array(np.load(SUB / "substrate.f32.npy", mmap_mode="r"))
    for name, kw in ABLATIONS.items():
        d = OUT_ROOT / name
        if (d / "summary.json").exists():
            print(f"{name}: done, skip")
            continue
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.0, verbose=True,
                                random_state=42, **kw)
            xy = np.asarray(reducer.fit_transform(X), dtype=np.float32)
        except Exception as e:  # noqa: BLE001 — record invalid toggles honestly
            (d / "summary.json").write_text(json.dumps({
                "arm": name, "rung": "2m", "overrides": kw, "seed": 42,
                "error": f"{type(e).__name__}: {e}",
                "note": "upstream ablation failed to run",
            }, indent=1))
            print(f"{name}: FAILED {e}")
            continue
        wall = time.time() - t0
        np.save(d / "coordinates.npy", xy)

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from knobs_2m import quick_ffr
        from map_renders import binned_counts, render_png, robust_extent

        ffr = quick_ffr(xy, SUB / "edges-k15-fuzzy.npz", X.shape[0])
        render_png(binned_counts(xy, robust_extent(xy)), d / "density.png")
        (d / "summary.json").write_text(json.dumps({
            "arm": name, "rung": "2m",
            "overrides": {k: (v if v is not None else "None") for k, v in kw.items()},
            "seed": 42, "wall_s": wall,
            "quick_ffr_at_0.1pct": float(ffr),
            "note": "upstream 0.6dev self-ablation (CPU, transductive); "
                    "default run = upstream-06dev-2m (0.4798).",
        }, indent=1))
        print(f"{name}: quick-FFR {ffr:.4f} in {wall/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
