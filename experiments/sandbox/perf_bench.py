#!/usr/bin/env python3
"""Training-throughput bench (owner order 2026-08-23): 10 experiments, ~1h.

Workload = the composed-core recipe (fneg10+tanh4+pos10) on the 500K
cross-check slice (subset_x.npy + edges-induced.npz — already on disk), fixed
3,000 updates per variant, measured as updates/s (median of the fit wall
minus setup). Baseline throughput at 2M is ~116 up/s; the encoder is small
(12M params) so launch overhead, optimizer fusion, and per-batch python are
prime suspects.

Variants (SAFE = intended numerics-preserving; TUNING = changes the recipe
and needs quality re-verification before adoption):
  v00-baseline        current code path
  v01-quiet           tqdm/log off                          SAFE
  v02-fused-adamw     AdamW(fused=True)                     SAFE
  v03-tf32            float32_matmul_precision('high')      SAFE-ish
  v04-compile         torch.compile(model)                  SAFE-ish (numerics)
  v05-compile-max     torch.compile(mode='max-autotune')    SAFE-ish
  v06-quiet+fused+tf32 combo                                SAFE
  v07-combo+compile   v06 + compile                         SAFE-ish
  v08-batch16k        batch 16384 (same draws/edge)         TUNING
  v09-batch32k        batch 32768                           TUNING

Env knobs consumed by core/this bench: PERF_QUIET, PERF_FUSED_ADAMW,
PERF_TF32, PERF_COMPILE, PERF_COMPILE_MODE (implemented as monkeypatches
here — no core edits; anything adopted gets a reviewed core patch + the 2M
same-output verification re-run afterwards).

Output: /data/latent-basemap/sandbox/perf-bench/results.json (+ per-variant
rows). Run under systemd on an idle GPU.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

CC = Path("/data/latent-basemap/sandbox/500k-crosscheck")
OUT = Path("/data/latent-basemap/sandbox/perf-bench")
UPDATES = 3_000
ROWS = 500_000

VARIANTS = [
    ("v00-baseline", {}),
    ("v01-quiet", {"quiet": True}),
    ("v02-fused-adamw", {"fused": True}),
    ("v03-tf32", {"tf32": True}),
    ("v04-compile", {"compile": "default"}),
    ("v05-compile-max", {"compile": "max-autotune"}),
    ("v06-combo-safe", {"quiet": True, "fused": True, "tf32": True}),
    ("v07-combo-compile", {"quiet": True, "fused": True, "tf32": True,
                           "compile": "default"}),
    ("v08-batch16k", {"quiet": True, "batch": 16384}),
    ("v09-batch32k", {"quiet": True, "batch": 32768}),
]


def run_variant(name: str, opts: dict) -> dict:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    from knobs_2m import BASE_KWARGS, MD

    from basemap.pumap.parametric_umap import core as C

    if opts.get("tf32"):
        torch.set_float32_matmul_precision("high")
    else:
        torch.set_float32_matmul_precision("highest")

    batch = opts.get("batch", BASE_KWARGS["batch_size"])
    kwargs = dict(BASE_KWARGS)
    kwargs.update({"low_dim_kernel": "umap", **MD["000"],
                   "fneg_weight": 1.0, "neg_tanh_gamma": 4.0,
                   "pos_ratio": 0.10, "batch_size": batch,
                   "total_steps_estimate": UPDATES,
                   "n_epochs": 1})
    p = C.ParametricUMAP(**kwargs)

    patches = []
    if opts.get("fused"):
        orig_adamw = torch.optim.AdamW

        def fused_adamw(params, **kw):
            kw["fused"] = True
            return orig_adamw(params, **kw)

        torch.optim.AdamW = fused_adamw
        patches.append(("adamw", orig_adamw))
    if opts.get("compile"):
        orig_init = p._init_model

        def _init(nf, _o=orig_init, _p=p, _m=opts["compile"]):
            _o(nf)
            _p.model = torch.compile(
                _p.model, mode=None if _m == "default" else _m)

        p._init_model = _init
    if opts.get("quiet"):
        import tqdm as tqdm_mod
        orig_tqdm = tqdm_mod.tqdm

        class _Silent(orig_tqdm):
            def __init__(self, *a, **kw):
                kw["disable"] = True
                super().__init__(*a, **kw)

        tqdm_mod.tqdm = _Silent
        C.tqdm = _Silent
        patches.append(("tqdm", orig_tqdm))

    X = np.load(CC / "subset_x.npy")
    t0 = time.time()
    try:
        p.fit(X, precomputed_edges_path=str(CC / "edges-induced.npz"),
              random_state=42)
        wall = time.time() - t0
        st = p._train_stats
        ups = int(st.get("successful_positive_lr_updates",
                         st.get("executed_positive_lr_updates", UPDATES)))
        result = {"variant": name, "opts": opts, "wall_s": round(wall, 1),
                  "updates": ups, "updates_per_s": round(ups / wall, 1),
                  "peak_vram_gb": round(
                      torch.cuda.max_memory_allocated() / 1e9, 2)}
    except Exception as e:  # noqa: BLE001 — a variant may just not work
        result = {"variant": name, "opts": opts,
                  "error": f"{type(e).__name__}: {str(e)[:200]}"}
    finally:
        for kind, orig in patches:
            if kind == "adamw":
                import torch as _t
                _t.optim.AdamW = orig
            elif kind == "tqdm":
                import tqdm as tqdm_mod
                tqdm_mod.tqdm = orig
                C.tqdm = orig
    return result


def main() -> int:
    import sys
    OUT.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else None
    results = []
    for name, opts in VARIANTS:
        if which and name != which:
            continue
        r = run_variant(name, opts)
        results.append(r)
        print(json.dumps(r), flush=True)
        (OUT / f"{name}.json").write_text(json.dumps(r, indent=1))
    if not which:
        (OUT / "results.json").write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
