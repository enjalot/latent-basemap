#!/usr/bin/env python3
"""T3 hybrid, form 1 (distill-init): start from the distilled encoder, finetune
with the UMAP loss (owner direction 2026-08-22; plan-teacher-distillation T3).

T1 showed pure regression reaches full-FFR 0.4328 (beats every trained arm)
but generalizes poorly (heldout 0.27 vs the trained recipe's ~0.42): teacher
gives global structure, the UMAP loss gives the generalizing function. This
composes them WITHOUT core changes: load the distill-huber-2x checkpoint as
the initial encoder (fit() keeps a pre-built self.model), then run the
promoted fneg recipe on the 2M graph at x1 and x2 dose.

Arms: distillinit-umap-x1 (80,163 updates), distillinit-umap-x2 (160,326).
Scored exactly like sandbox arms (quick-FFR + render + summary in 2m-knobs/).
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

SUB = Path("/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
           "minilm-mixed-2m-substrate-and-exact-k15-graph-v1")
DISTILLED = Path("/data/latent-basemap/sandbox/2m-knobs/distill-huber-2x/model.pt")
OUT_ROOT = Path("/data/latent-basemap/sandbox/2m-knobs")
SEED = 42
ARMS = {"distillinit-umap-x1": 1, "distillinit-umap-x2": 2}


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from knobs_2m import BASE_KWARGS, MD, quick_ffr
    from map_renders import binned_counts, render_png, robust_extent

    from basemap.pumap.parametric_umap.core import ParametricUMAP

    X = np.array(np.load(SUB / "substrate.f32.npy", mmap_mode="r"))
    edges = SUB / "edges-k15-fuzzy.npz"
    base_horizon = 80_163  # R0217 0.6782 draws/edge
    directed_edges = 48_344_648
    n_pos = int(BASE_KWARGS["batch_size"] * BASE_KWARGS["pos_ratio"])

    for arm, dose in ARMS.items():
        d = OUT_ROOT / arm
        if (d / "summary.json").exists():
            print(f"{arm}: done, skip")
            continue
        d.mkdir(parents=True, exist_ok=True)
        horizon = base_horizon * dose
        kwargs = dict(BASE_KWARGS)
        kwargs.update({"low_dim_kernel": "umap", **MD["000"],
                       "fneg_weight": 1.0,
                       "total_steps_estimate": horizon,
                       "n_epochs": max(1, math.ceil(
                           horizon / math.ceil(directed_edges / n_pos)))})
        p = ParametricUMAP(**kwargs)
        p._init_model(X.shape[1])
        state = torch.load(DISTILLED, map_location="cuda", weights_only=False)
        p.model.load_state_dict(state["model_state_dict"])
        print(f"{arm}: initialized from {DISTILLED.name}", flush=True)
        torch.manual_seed(SEED)
        t0 = time.time()
        p.fit(X, precomputed_edges_path=str(edges), random_state=SEED)
        xy = np.asarray(p.transform(X, batch_size=8192), dtype=np.float32)
        wall = time.time() - t0
        np.save(d / "coordinates.npy", xy)
        p.save(str(d / "model.pt"))
        render_png(binned_counts(xy, robust_extent(xy)), d / "density.png")
        ffr = quick_ffr(xy, edges, X.shape[0])
        (d / "summary.json").write_text(json.dumps({
            "arm": arm, "rung": "2m",
            "overrides": {"init": "distill-huber-2x", "low_dim_kernel": "umap",
                          **MD["000"], "fneg_weight": 1.0},
            "seed": SEED, "dose_multiplier": dose, "horizon_updates": horizon,
            "wall_s": wall, "quick_ffr_at_0.1pct": float(ffr),
            "substrate": str(SUB / "substrate.f32.npy"), "edges": str(edges),
            "note": "T3 hybrid form 1: distilled-teacher init + promoted UMAP "
                    "loss finetune; sandbox, no sealed claim.",
        }, indent=1))
        print(f"{arm}: FFR {ffr:.4f} in {wall/60:.1f} min", flush=True)
        del p
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
