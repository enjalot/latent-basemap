#!/usr/bin/env python3
"""Reproducibility verification for the perf round (owner condition: any
adopted speedup must reproduce an existing run's output).

Re-runs the exact config of an EXISTING arm (umap-md010-x2-fneg10-tanh4,
FFR 0.3505) with the current code path, same seed sequence as knobs run_arm,
and compares coordinates + FFR against the saved artifact. Tonight this
establishes the reproducibility baseline; after any perf patch is adopted
into core, this script is the same-output gate.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ARM = "umap-md010-x2-fneg10-tanh4"
SAVED = Path("/data/latent-basemap/sandbox/2m-knobs") / ARM
OUT = Path("/data/latent-basemap/sandbox/perf-bench") / f"verify-{ARM}"


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from knobs_2m import ARMS, BASE_KWARGS, RUNGS, quick_ffr

    from basemap.pumap.parametric_umap.core import ParametricUMAP

    rung = RUNGS["2m"]
    overrides = dict(ARMS[ARM])
    dose = overrides.pop("dose", 1)
    import math
    kwargs = {**BASE_KWARGS, **overrides}
    horizon = round(dose * rung["base_horizon"])
    num_pos = int(BASE_KWARGS["batch_size"] * kwargs["pos_ratio"])
    kwargs["total_steps_estimate"] = horizon
    kwargs["n_epochs"] = max(1, math.ceil(
        horizon / math.ceil(rung["directed_edges"] / num_pos)))

    X = np.load(rung["substrate"], mmap_mode="r")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    p = ParametricUMAP(**kwargs)
    t0 = time.time()
    p.fit(np.asarray(X, dtype=np.float32),
          precomputed_edges_path=str(rung["edges"]), random_state=42)
    xy = np.asarray(p.transform(np.asarray(X, dtype=np.float32),
                                batch_size=8192), dtype=np.float32)
    wall = time.time() - t0
    np.save(OUT / "coordinates.npy", xy)

    ref = np.load(SAVED / "coordinates.npy")
    saved_ffr = json.loads((SAVED / "summary.json").read_text())[
        "quick_ffr_at_0.1pct"]
    ffr = quick_ffr(xy, rung["edges"], X.shape[0])
    bitwise = bool(np.array_equal(xy, ref))
    max_abs = float(np.max(np.abs(xy - ref))) if xy.shape == ref.shape else None
    report = {
        "arm": ARM, "wall_s": round(wall, 1),
        "bitwise_identical": bitwise,
        "max_abs_coord_diff": max_abs,
        "ffr_rerun": float(ffr), "ffr_saved": float(saved_ffr),
        "ffr_delta": float(ffr - saved_ffr),
        "verdict": "IDENTICAL" if bitwise else (
            "EQUIVALENT" if abs(ffr - saved_ffr) < 0.003 else "DIVERGED"),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
