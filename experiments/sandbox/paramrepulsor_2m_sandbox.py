#!/usr/bin/env python3
"""ParamRepulsor 2M sandbox run — upstream defaults, one seed. NOT the R0270 round.

Two phases because two environments (plan-gpu-window-2026-08-21.md §6):

  fit   (this file, /data/latent-basemap/paramrepulsor-env python): fit
        upstream ParamPaCMAP @ be8df72 on the 2M sandbox substrate, save
        coordinates + encoder state + timing.
        NOTE: env is the pinned package set EXCEPT torch cu128 (the R0270
        cu124 lock has no sm_120 kernels — cannot run on the 5090 at all).
  score (latent-basemap .venv): quick-FFR + spacing on the coordinates,
        density render, sandbox summary.json -> shows on the review page.

Usage: paramrepulsor_2m_sandbox.py fit|score
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

SUBSTRATE = Path(
    "/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
    "minilm-mixed-2m-substrate-and-exact-k15-graph-v1/substrate.f32.npy")
EDGES = SUBSTRATE.parent / "edges-k15-fuzzy.npz"
OUT = Path("/data/latent-basemap/sandbox/2m-knobs/paramrepulsor-upstream")
SEED = 42
ROWS = 2_000_000
DIM = 384


def fit() -> int:
    import torch
    from parampacmap import ParamPaCMAP

    OUT.mkdir(parents=True, exist_ok=True)
    X = np.load(SUBSTRATE, mmap_mode="r")
    assert X.shape == (ROWS, DIM), X.shape
    X = np.asarray(X, dtype=np.float32)  # upstream wants a real array (3 GB)

    # upstream defaults; the ONLY study settings are seed + verbose (the same
    # two the R0270 recipe declares).
    reducer = ParamPaCMAP(seed=SEED, verbose=True)
    t0 = time.time()
    coords = reducer.fit_transform(X)
    wall = time.time() - t0
    np.save(OUT / "coordinates.npy", np.asarray(coords, dtype=np.float32))
    model = getattr(reducer, "model", None)
    if model is not None:
        torch.save(model.state_dict(), OUT / "encoder_state.pt")
    (OUT / "fit_info.json").write_text(json.dumps({
        "upstream": "parampacmap @ be8df72 (ParamRepulsor)",
        "settings": {"seed": SEED, "verbose": True, "everything_else": "upstream defaults"},
        "torch": torch.__version__,
        "env_note": "pinned package set except torch cu128 (cu124 lock has no "
                    "sm_120 kernels; R0270 evidence round will hit this too)",
        "rows": ROWS,
        "wall_s": wall,
    }, indent=1))
    print(f"fit done in {wall/60:.1f} min -> {OUT}")
    return 0


def score() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from knobs_2m import quick_ffr
    from map_renders import binned_counts, render_png, robust_extent

    coords = np.load(OUT / "coordinates.npy")
    fit_info = json.loads((OUT / "fit_info.json").read_text())
    ffr = quick_ffr(coords, EDGES, ROWS)
    r10 = None
    try:  # same spacing stat the arms report, if the helper exposes it
        from knobs_2m import r10_over_map_radius_median  # type: ignore
        r10 = r10_over_map_radius_median(coords)
    except ImportError:
        pass
    render_png(binned_counts(coords, robust_extent(coords)), OUT / "density.png")
    summary = {
        "arm": "paramrepulsor-upstream",
        "rung": "2m",
        "overrides": {"external_baseline": "ParamRepulsor be8df72, upstream defaults"},
        "seed": SEED,
        "quick_ffr_at_0.1pct": float(ffr),
        "wall_s": fit_info["wall_s"],
        "substrate": str(SUBSTRATE),
        "edges": str(EDGES),
        "note": "sandbox external baseline; not a round, no sealed claim. "
                "Fit in the cu128 variant env (see fit_info.json).",
    }
    if r10 is not None:
        summary["r10_over_map_radius_median"] = float(r10)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"quick-FFR {ffr:.4f} -> {OUT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit({"fit": fit, "score": score}[sys.argv[1]]())
