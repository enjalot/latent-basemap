#!/usr/bin/env python3
"""Multilevel (coarse-to-fine) training — the faithful port of upstream
0.6dev's recursive init, stage 2 (owner order 2026-08-22).

Curriculum: train the encoder on the COARSEST level first (weakened kernel
a^1/4, b^1/4 — upstream's coarse-level exponents), carry the weights to each
finer level (the encoder evaluated on finer features IS the expansion
operator; no Procrustes needed — the one place the parametric setting is
cleaner than upstream), and finish on the full 2M graph at the full kernel
with the composed-core recipe.

Coarse levels are nearly free (their edge counts are ~1/16, 1/256, ... of the
full graph at the same draws/edge). Arms:

  mlinit-core-x2   coarse curriculum -> full graph at dose x2
  mlinit-core-x4   same -> dose x4

vs plain core-x2 = 0.3734 / core-only references. Scored like every sandbox
arm (quick-FFR + render + summary in 2m-knobs/).
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path("/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
           "minilm-mixed-2m-substrate-and-exact-k15-graph-v1")
ML = Path("/data/latent-basemap/substrates/multilevel-2m")
OUT_ROOT = Path("/data/latent-basemap/sandbox/2m-knobs")
SEED = 42
COARSE_EXP = 0.25          # upstream's a^1/4, b^1/4 at coarse levels
COARSE_DRAWS_PER_EDGE = 2.7128  # x4-equivalent on the tiny coarse graphs
ARMS = {"mlinit-core-x2": 2, "mlinit-core-x4": 4}


def _levels() -> list[Path]:
    lvls = sorted(ML.glob("level*"), key=lambda p: int(p.name[5:]))
    assert lvls, "run multilevel_coarsen.py first"
    return lvls[::-1]  # coarsest first


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from knobs_2m import BASE_KWARGS, MD, quick_ffr
    from map_renders import binned_counts, render_png, robust_extent

    from basemap.pumap.parametric_umap.core import ParametricUMAP

    n_pos05 = int(BASE_KWARGS["batch_size"] * 0.05)
    core = dict(fneg_weight=1.0, neg_tanh_gamma=4.0, pos_ratio=0.10)
    a0, b0 = MD["000"]["a"], MD["000"]["b"]

    for arm, final_dose in ARMS.items():
        d = OUT_ROOT / arm
        if (d / "summary.json").exists():
            print(f"{arm}: done, skip")
            continue
        d.mkdir(parents=True, exist_ok=True)
        t_all = time.time()
        state = None
        curriculum_log = []

        # ---- coarse levels, coarsest -> finest ----
        for lvl in _levels():
            meta = json.loads((lvl / "meta.json").read_text())
            e = meta["directed_edges"]
            n_pos = int(BASE_KWARGS["batch_size"] * core["pos_ratio"])
            horizon = max(200, int(round(COARSE_DRAWS_PER_EDGE * e / n_pos)))
            kwargs = dict(BASE_KWARGS)
            kwargs.update({"low_dim_kernel": "umap",
                           "a": a0 ** COARSE_EXP, "b": b0 ** COARSE_EXP,
                           **core,
                           "total_steps_estimate": horizon,
                           "n_epochs": max(1, math.ceil(
                               horizon * n_pos / e))})
            p = ParametricUMAP(**kwargs)
            if state is not None:
                orig = p._init_model

                def _init(nf, _o=orig, _p=p, _s=state):
                    _o(nf)
                    _p.model.load_state_dict(_s)

                p._init_model = _init
            X = np.load(lvl / "substrate.f32.npy")
            t0 = time.time()
            p.fit(X, precomputed_edges_path=str(lvl / "edges.npz"),
                  random_state=SEED)
            state = {k: v.detach().clone() for k, v in p.model.state_dict().items()}
            curriculum_log.append({"level": lvl.name, "nodes": meta["nodes"],
                                   "horizon": horizon,
                                   "wall_s": round(time.time() - t0, 1)})
            print(f"{arm} {lvl.name}: {meta['nodes']:,} nodes, "
                  f"{horizon} upd, {time.time()-t0:.0f}s", flush=True)
            del p
            torch.cuda.empty_cache()

        # ---- final level: full graph, full kernel, composed core ----
        e_full = 48_344_648
        n_pos = int(BASE_KWARGS["batch_size"] * core["pos_ratio"])
        horizon = int(round(final_dose * 0.6782 * e_full / n_pos05))
        kwargs = dict(BASE_KWARGS)
        kwargs.update({"low_dim_kernel": "umap", "a": a0, "b": b0, **core,
                       "total_steps_estimate": horizon,
                       "n_epochs": max(1, math.ceil(horizon * n_pos / e_full))})
        p = ParametricUMAP(**kwargs)
        orig = p._init_model

        def _init_final(nf, _o=orig, _p=p, _s=state):
            _o(nf)
            _p.model.load_state_dict(_s)

        p._init_model = _init_final
        X = np.array(np.load(SRC / "substrate.f32.npy", mmap_mode="r"))
        t0 = time.time()
        p.fit(X, precomputed_edges_path=str(SRC / "edges-k15-fuzzy.npz"),
              random_state=SEED)
        xy = np.asarray(p.transform(X, batch_size=8192), dtype=np.float32)
        np.save(d / "coordinates.npy", xy)
        p.save(str(d / "model.pt"))
        render_png(binned_counts(xy, robust_extent(xy)), d / "density.png")
        ffr = quick_ffr(xy, SRC / "edges-k15-fuzzy.npz", X.shape[0])
        (d / "summary.json").write_text(json.dumps({
            "arm": arm, "rung": "2m",
            "overrides": {"init": "multilevel-coarse-to-fine",
                          "coarse_kernel_exp": COARSE_EXP,
                          "low_dim_kernel": "umap", **MD["000"], **core},
            "seed": SEED, "dose_multiplier": final_dose,
            "horizon_updates": horizon,
            "curriculum": curriculum_log,
            "wall_s": time.time() - t_all,
            "quick_ffr_at_0.1pct": float(ffr),
            "substrate": str(SRC / "substrate.f32.npy"),
            "edges": str(SRC / "edges-k15-fuzzy.npz"),
            "note": "faithful multilevel-init port (label-prop coarsening, "
                    "weakened coarse kernels, encoder-as-expansion); sandbox.",
        }, indent=1))
        print(f"{arm}: FFR {ffr:.4f} total {(time.time()-t_all)/60:.1f} min",
              flush=True)
        del p
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
