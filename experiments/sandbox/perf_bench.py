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
import os
OUT = Path(os.environ.get("PERF_OUT",
           "/data/latent-basemap/sandbox/perf-bench"))
UPDATES = 3_000            # v1 (kept for the recorded run)
DELTA_UPDATES = (500, 10_500)  # v2 delta-method: setup cancels in the diff
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
        # core.py binds AdamW at import time (`from torch.optim import AdamW`,
        # core.py:10) and constructs the optimizer via that module-global name
        # (core.py:1654, and the anchored-init path at core.py:1137). Patching
        # torch.optim.AdamW is therefore invisible to the trainer — the alias
        # already captured the original class. Rebind the name in core's own
        # namespace so the fused kwarg actually reaches the constructed optimizer.
        orig_adamw = C.AdamW

        def fused_adamw(params, **kw):
            kw["fused"] = True
            return orig_adamw(params, **kw)

        C.AdamW = fused_adamw
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
    try:
        # delta method: run short then long with identical setup; the diff is
        # pure steady-state training. n_epochs sized so the horizon is
        # reached even at big batches (the v1 flaw).
        walls, ups_seen = [], []
        for target in DELTA_UPDATES:
            q = C.ParametricUMAP(**{**kwargs, "total_steps_estimate": target,
                                    "n_epochs": max(1, math.ceil(
                                        target * batch * kwargs["pos_ratio"]
                                        / 3_008_780) + 1)})
            if opts.get("compile"):
                orig_i = q._init_model

                def _ci(nf, _o=orig_i, _q=q, _m=opts["compile"]):
                    _o(nf)
                    _q.model = torch.compile(
                        _q.model, mode=None if _m == "default" else _m)

                q._init_model = _ci
            t0 = time.time()
            q.fit(X, precomputed_edges_path=str(CC / "edges-induced.npz"),
                  random_state=42)
            walls.append(time.time() - t0)
            st = q._train_stats
            ups_seen.append(int(st.get("optimizer_steps_succeeded")
                                or st.get("positive_lr_optimizer_steps")
                                or st.get("scheduler_steps") or target))
            del q
            torch.cuda.empty_cache()
        d_up = ups_seen[1] - ups_seen[0]
        d_wall = walls[1] - walls[0]
        ups_per_s = d_up / max(d_wall, 1e-9)
        result = {"variant": name, "opts": opts,
                  "updates": ups_seen, "walls_s": [round(w, 1) for w in walls],
                  "updates_per_s": round(ups_per_s, 1),
                  "edges_per_s": round(ups_per_s * batch),
                  "peak_vram_gb": round(
                      torch.cuda.max_memory_allocated() / 1e9, 2)}
    except Exception as e:  # noqa: BLE001 — a variant may just not work
        result = {"variant": name, "opts": opts,
                  "error": f"{type(e).__name__}: {str(e)[:200]}"}
    finally:
        for kind, orig in patches:
            if kind == "adamw":
                C.AdamW = orig
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
