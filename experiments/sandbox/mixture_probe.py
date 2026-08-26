#!/usr/bin/env python3
"""BROAD probe-register suite, step 3: score a MiniLM mixture sweep (GPU).

For each sweep MAP x each probe register, project the register's substrate
through the frozen map and score fold-faithfulness (OOD-FFR@0.1%) against that
register's OWN exact-k15 fuzzy truth graph.  A map that receives an out-of-
distribution register faithfully keeps that register's local neighborhoods
intact after projection.

The sweep varies the SOCIAL share of the training mixture.  We report, per map,
the WORST register FFR (the map's weakest-received corpus) and the MEAN FFR, and
across maps the MAXIMIN winner (the mixture whose worst register is least bad).
If the worst-register score peaks at an interior social share rather than at 0%
or 30%, the sweep has an interior optimum -- some social data helps broad OOD
reception, too much hurts.

The maps scored (``model.pt`` inside each dir under SANDBOX):

  minilm-mix-1m/rankfrac-25              social 0%   (baseline)
  minilm-rmix{10,20,30}-1m/champion-bs16k   social 10/20/30%, mixture family "rmix"
  minilm-bmix{10,20,30}-1m/champion-bs16k   social 10/20/30%, mixture family "bmix"

A map whose model.pt is missing is skipped (its whole row recorded null); a
register whose truth graph is missing is skipped for every map (null), since the
orchestrator builds knn+fuzzy on each register before this runs.

GPU script (ParametricUMAP.transform): the orchestrator runs it when the GPU is
free.  Import-safe: no torch / model load at import time.

Usage:
    mixture_probe.py                 # score the hardcoded sweep maps
    mixture_probe.py <map> [<map>..] # score only the named maps (keys below)
Output: /data/latent-basemap/sandbox/mixture-sweep-results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

SANDBOX = Path("/data/latent-basemap/sandbox")

# the eight BROAD probe registers (built by build_probe_registers.py).
REGISTERS = (
    "probe-reddit", "probe-ca", "probe-twitter", "probe-bluesky",
    "probe-wiki", "probe-ccweb", "probe-ccscience", "probe-code",
)
SUBSTRATES = Path("/data/latent-basemap/substrates")

# sweep map key -> (checkpoint dir under SANDBOX, mixture family, social pct).
# The baseline is the shared 0% point of both mixture families.
SWEEP_MAPS = {
    "minilm-mix-1m/rankfrac-25":        ("minilm-mix-1m/rankfrac-25", "baseline", 0),
    "minilm-rmix10-1m/champion-bs16k":  ("minilm-rmix10-1m/champion-bs16k", "rmix", 10),
    "minilm-rmix20-1m/champion-bs16k":  ("minilm-rmix20-1m/champion-bs16k", "rmix", 20),
    "minilm-rmix30-1m/champion-bs16k":  ("minilm-rmix30-1m/champion-bs16k", "rmix", 30),
    "minilm-bmix10-1m/champion-bs16k":  ("minilm-bmix10-1m/champion-bs16k", "bmix", 10),
    "minilm-bmix20-1m/champion-bs16k":  ("minilm-bmix20-1m/champion-bs16k", "bmix", 20),
    "minilm-bmix30-1m/champion-bs16k":  ("minilm-bmix30-1m/champion-bs16k", "bmix", 30),
}


def _interior_optima(per_map_worst: dict, per_map_mean: dict) -> dict:
    """For each mixture family (rmix, bmix, each including the 0% baseline),
    report whether worst_register_ffr / mean_ffr peaks at an INTERIOR social
    share (10 or 20%) rather than at an endpoint (0 or 30%)."""
    families = {"rmix": [], "bmix": []}
    for key, (_, fam, pct) in SWEEP_MAPS.items():
        for f in (("rmix", "bmix") if fam == "baseline" else (fam,)):
            families[f].append((pct, key))
    report = {}
    for fam, pts in families.items():
        pts = sorted(pts)
        def _series(metric):
            xs = [(pct, metric.get(key)) for pct, key in pts]
            xs = [(p, v) for p, v in xs if v is not None]
            if len(xs) < 3:
                return None
            best_p = max(xs, key=lambda pv: pv[1])[0]
            endpoints = {xs[0][0], xs[-1][0]}
            return {"shares_scored": [p for p, _ in xs],
                    "values": {p: v for p, v in xs},
                    "argmax_share": best_p,
                    "interior_optimum": best_p not in endpoints}
        report[fam] = {"worst_register_ffr": _series(per_map_worst),
                       "mean_ffr": _series(per_map_mean)}
    return report


def main(argv: list[str]) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from image_map_pipeline import _norm
    from knobs_2m import quick_ffr

    from basemap.pumap.parametric_umap.core import ParametricUMAP

    want = set(argv[1:])
    maps = {k: v for k, v in SWEEP_MAPS.items() if not want or k in want}

    coords_dir = SANDBOX / "mixture-sweep-coords"
    coords_dir.mkdir(parents=True, exist_ok=True)

    # cache each register's normalized substrate + truth-graph path (lazy).
    reg_cache: dict[str, tuple] = {}

    def _register(reg: str):
        if reg not in reg_cache:
            sub = SUBSTRATES / reg / "substrate.f32.npy"
            edges = SANDBOX / reg / "edges-k15-fuzzy.npz"
            if not sub.exists() or not edges.exists():
                reg_cache[reg] = (None, None, None)
            else:
                x = _norm(np.asarray(np.load(sub, mmap_mode="r"), dtype=np.float32))
                reg_cache[reg] = (x, edges, int(x.shape[0]))
        return reg_cache[reg]

    matrix: dict[str, dict] = {}
    for key, (subdir, fam, pct) in maps.items():
        model_pt = SANDBOX / subdir / "model.pt"
        if not model_pt.exists():
            print(f"{key}: no model.pt at {model_pt}, skip (null row)", flush=True)
            matrix[key] = None
            continue
        print(f"{key}: social={pct}% family={fam}", flush=True)
        model = ParametricUMAP.load(str(model_pt), device="cuda")
        row: dict[str, float | None] = {}
        for reg in REGISTERS:
            x, edges, n = _register(reg)
            if x is None:
                print(f"  {reg}: substrate or truth graph missing, null", flush=True)
                row[reg] = None
                continue
            t0 = time.time()
            xy = np.asarray(model.transform(x, batch_size=8192), dtype=np.float32)
            np.save(coords_dir / f"{key.replace('/', '__')}--{reg}.npy", xy)
            ffr = float(quick_ffr(xy, edges, n))
            row[reg] = ffr
            print(f"  {reg}: OOD-FFR {ffr:.4f} ({(time.time()-t0)/60:.1f} min)",
                  flush=True)
        matrix[key] = row
        del model

    # per-map worst + mean over the registers actually scored.
    per_map_worst: dict[str, float | None] = {}
    per_map_mean: dict[str, float | None] = {}
    for key, row in matrix.items():
        if row is None:
            per_map_worst[key] = per_map_mean[key] = None
            continue
        vals = [v for v in row.values() if v is not None]
        per_map_worst[key] = float(min(vals)) if vals else None
        per_map_mean[key] = float(np.mean(vals)) if vals else None

    # maximin winner: the map whose worst register is least bad.
    scored_worst = {k: v for k, v in per_map_worst.items() if v is not None}
    maximin_winner = (max(scored_worst, key=scored_worst.get)
                      if scored_worst else None)

    optima = _interior_optima(per_map_worst, per_map_mean)

    out = SANDBOX / "mixture-sweep-results.json"
    out.write_text(json.dumps({
        "schema": "mixture-broad-probe-2026-08-26",
        "registers": list(REGISTERS),
        "maps": {k: {"dir": str(SANDBOX / v[0]), "family": v[1],
                     "social_pct": v[2]} for k, v in SWEEP_MAPS.items()},
        "ffr_matrix": matrix,
        "per_map_worst_register_ffr": per_map_worst,
        "per_map_mean_ffr": per_map_mean,
        "maximin_winner": maximin_winner,
        "maximin_worst_register_ffr": (scored_worst.get(maximin_winner)
                                       if maximin_winner else None),
        "social_share_interior_optimum": optima,
        "note": "OOD-FFR@0.1%: each probe register projected through each frozen "
                "sweep map, scored vs the register's own exact-k15 fuzzy graph. "
                "per_map_worst_register_ffr = the map's weakest-received corpus; "
                "maximin_winner maximizes that worst case. social_share_interior_"
                "optimum flags, per mixture family (incl. the 0% baseline), whether "
                "the best score lands on an interior social share (10/20%) vs an "
                "endpoint (0/30%). Missing model.pt -> null row; missing register "
                "truth graph -> null cell.",
    }, indent=1))
    print(f"\nresults: {out}", flush=True)
    print(f"  per-map worst-register FFR: {per_map_worst}")
    print(f"  maximin winner: {maximin_winner} "
          f"({scored_worst.get(maximin_winner) if maximin_winner else None})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
