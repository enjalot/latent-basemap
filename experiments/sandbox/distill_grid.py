#!/usr/bin/env python3
"""Capacity-vs-scale distill grid (owner order 2026-08-23).

The question: how does the FFR-optimal encoder width grow with data scale?
(h3072 saturated at 2M; extrapolation needed for the 30M program.) The
instrument: distillation as a pure capacity meter — regress a 0.6dev teacher
layout into encoders of several widths; regression quality isolates
REPRESENTATIONAL capacity from training dynamics. Chinchilla-style small
grid -> power-law fit -> extrapolate.

Grid: scales {500K, 1M, 2M} x widths {512, 1024, 2048, 3072, 4096},
20K Huber updates each (~2-4 min/cell, ~50 min total GPU).

Phases:
  prep (umap06dev env, CPU): build the 1M subset (every 2nd 2M row), its
        0.6dev teacher layout, and its induced truth graph. (500K and 2M
        teachers/truths already exist.)
  grid (.venv, GPU): the 15 distill cells + saturation-width fit.

Outputs: /data/latent-basemap/sandbox/distill-grid/<scale>-h<width>/summary.json
+ report.json (per-scale curves, saturation widths at 97.5% of max, fitted
log-log slope, extrapolations to 30M / 100M).
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

SUB2M = Path("/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
             "minilm-mixed-2m-substrate-and-exact-k15-graph-v1")
CC = Path("/data/latent-basemap/sandbox/500k-crosscheck")
GRID = Path("/data/latent-basemap/sandbox/distill-grid")
WIDTHS = (512, 1024, 2048, 3072, 4096)
UPDATES = 20_000
BATCH = 8192
SEED = 42

SCALES = {
    "500k": {"x": CC / "subset_x.npy",
             "teacher": CC / "upstream-06dev/coordinates.npy",
             "edges": CC / "edges-induced.npz", "rows": 500_000},
    "1m": {"x": GRID / "subset-1m/x.npy",
           "teacher": GRID / "subset-1m/teacher.npy",
           "edges": GRID / "subset-1m/edges-induced.npz", "rows": 1_000_000},
    "2m": {"x": SUB2M / "substrate.f32.npy",
           "teacher": Path("/data/latent-basemap/sandbox/2m-knobs/"
                           "upstream-06dev-2m/coordinates.npy"),
           "edges": SUB2M / "edges-k15-fuzzy.npz", "rows": 2_000_000},
}


def prep() -> int:
    import umap

    d = GRID / "subset-1m"
    d.mkdir(parents=True, exist_ok=True)
    if (d / "teacher.npy").exists():
        print("1m prep exists, skip")
        return 0
    X2 = np.load(SUB2M / "substrate.f32.npy", mmap_mode="r")
    idx = np.arange(0, 2_000_000, 2)
    X = np.ascontiguousarray(np.asarray(X2[idx], dtype=np.float32))
    np.save(d / "x.npy", X)
    npz = np.load(SUB2M / "edges-k15-fuzzy.npz")
    src, dst, w = npz["sources"], npz["targets"], npz["weights"]
    in_sub = np.zeros(2_000_000, dtype=bool)
    in_sub[idx] = True
    keep = in_sub[src] & in_sub[dst]
    remap = np.full(2_000_000, -1, dtype=np.int64)
    remap[idx] = np.arange(len(idx))
    np.savez(d / "edges-induced.npz",
             sources=remap[src[keep]].astype(np.int32),
             targets=remap[dst[keep]].astype(np.int32),
             weights=w[keep].astype(np.float32),
             n_nodes=np.int64(len(idx)))
    t0 = time.time()
    xy = umap.UMAP(n_neighbors=15, min_dist=0.0, random_state=SEED,
                   verbose=True).fit_transform(X)
    np.save(d / "teacher.npy", np.asarray(xy, dtype=np.float32))
    print(f"1m teacher in {(time.time()-t0)/60:.1f} min; "
          f"induced edges {int(keep.sum()):,}")
    return 0


def grid() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from knobs_2m import BASE_KWARGS, quick_ffr

    from basemap.pumap.parametric_umap.core import ParametricUMAP

    for sname, sc in SCALES.items():
        X = np.array(np.load(sc["x"], mmap_mode="r"), dtype=np.float32)
        t = np.load(sc["teacher"]).astype(np.float32)
        t = t - t.mean(axis=0)
        t *= 5.0 / max(float(np.sqrt((t ** 2).sum(axis=1).mean())), 1e-9)
        Xg = torch.from_numpy(X).cuda().half()
        Tg = torch.from_numpy(t).cuda()
        n = X.shape[0]
        for width in WIDTHS:
            d = GRID / f"{sname}-h{width}"
            if (d / "summary.json").exists():
                print(f"{sname}-h{width}: done, skip")
                continue
            d.mkdir(parents=True, exist_ok=True)
            p = ParametricUMAP(**{**BASE_KWARGS, "hidden_dim": width,
                                  "n_epochs": 1})
            p._init_model(X.shape[1])
            net = p.model
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda u: min(1.0, (u + 1) / 200) * 0.5 * (
                    1 + math.cos(math.pi * min(u / UPDATES, 1.0))))
            gen = torch.Generator(device="cuda").manual_seed(SEED)
            torch.manual_seed(SEED)
            t0 = time.time()
            net.train()
            for u in range(UPDATES):
                idx = torch.randint(0, n, (BATCH,), generator=gen,
                                    device="cuda")
                xb = Xg.index_select(0, idx).float()
                yb = Tg.index_select(0, idx)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = net(xb)
                loss = torch.nn.functional.huber_loss(pred.float(), yb, 1.0)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                sched.step()
            wall = time.time() - t0
            net.eval()
            with torch.no_grad():
                xy = np.concatenate(
                    [net(Xg[i:i + 65536].float()).cpu().numpy()
                     for i in range(0, n, 65536)]).astype(np.float32)
            ffr = quick_ffr(xy, sc["edges"], n)
            agree = float(np.median(np.linalg.norm(xy - t, axis=1)) / 5.0)
            (d / "summary.json").write_text(json.dumps({
                "scale": sname, "rows": n, "width": width,
                "updates": UPDATES, "wall_s": round(wall, 1),
                "quick_ffr_at_0.1pct": float(ffr),
                "teacher_agreement_median_r": agree,
            }, indent=1))
            print(f"{sname}-h{width}: FFR {ffr:.4f} agree {agree:.4f}R "
                  f"({wall/60:.1f} min)", flush=True)
            del p, net, opt
            torch.cuda.empty_cache()
        del Xg, Tg
        torch.cuda.empty_cache()

    # fit: saturation width per scale (smallest width >= 97.5% of scale max)
    report = {"scales": {}, "widths": list(WIDTHS)}
    sat = {}
    for sname, sc in SCALES.items():
        curve = {}
        for width in WIDTHS:
            f = GRID / f"{sname}-h{width}" / "summary.json"
            if f.exists():
                curve[width] = json.loads(f.read_text())["quick_ffr_at_0.1pct"]
        if not curve:
            continue
        mx = max(curve.values())
        sat_w = min(w for w, v in curve.items() if v >= 0.975 * mx)
        report["scales"][sname] = {"rows": sc["rows"], "curve": curve,
                                   "saturation_width": sat_w}
        sat[sc["rows"]] = sat_w
    if len(sat) >= 2:
        xs = np.log([*sat.keys()])
        ys = np.log([*sat.values()])
        slope, intercept = np.polyfit(xs, ys, 1)
        report["fit"] = {
            "log_log_slope": float(slope),
            "extrapolated_width_30m": float(np.exp(
                intercept + slope * np.log(30e6))),
            "extrapolated_width_100m": float(np.exp(
                intercept + slope * np.log(100e6))),
            "caveat": "3-point fit on one corpus; anchor at 6.25M before "
                      "trusting; exponent is corpus-dependent.",
        }
    (GRID / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report.get("fit", {}), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit({"prep": prep, "grid": grid}[sys.argv[1]]())
