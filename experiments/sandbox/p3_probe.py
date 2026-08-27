#!/usr/bin/env python3
"""P3 curation-validation loop, step 2: OOD-probe the (a)/(b)/(c) champion maps.

Closes the loop opened by ``p3_build_and_scorecard.py``: that step froze a
predicted diversity order (by Vendi + kNN-ball radius) over the three substrates
BEFORE any (b)/(c) map was trained.  This step measures the maps' *generalization*
by projecting held-out MiniLM register corpora (reddit, community-archive) through
each frozen champion map and scoring fold-faithfulness (FFR) against each
register's OWN truth graph.  The resulting `probe_ffr_order` is then compared to
the scorecard's `predicted_diversity_order`: does a more diverse training
substrate yield a map that receives out-of-distribution data more faithfully?

The three champion maps:
  (a) current-mix   2m-knobs/umap-md000-x4bs16k-winner        (already trained)
  (b) curated       minilm-curated-2m/champion-bs16k          (trained by orchestrator)
  (c) random        minilm-random-2m/champion-bs16k           (trained by orchestrator)

A map whose model.pt is missing is skipped (its FFR recorded as null).

GPU script (ParametricUMAP.transform): the orchestrator runs it when the GPU is
free.  Modeled on ``register_ood_probe.py``.

Usage: p3_probe.py
Output: /data/latent-basemap/sandbox/p3-probe-results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

SANDBOX = Path("/data/latent-basemap/sandbox")
REGISTERS = ("reddit-2m", "communityarchive-2m")

# map key -> checkpoint directory (expects model.pt inside)
MAPS = {
    "a": SANDBOX / "2m-knobs/umap-md000-x4bs16k-winner",
    "b": SANDBOX / "minilm-curated-2m/champion-bs16k",
    "c": SANDBOX / "minilm-random-2m/champion-bs16k",
}


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from image_map_pipeline import DATASETS, _norm
    from knobs_2m import quick_ffr_v2 as quick_ffr  # v2 truth-selection (review 2026-08-27)

    from basemap.pumap.parametric_umap.core import ParametricUMAP

    # per-register: {map_key: ffr or None}
    results: dict[str, dict[str, float | None]] = {}
    coords_dir = SANDBOX / "p3-probe-coords"
    coords_dir.mkdir(parents=True, exist_ok=True)

    for ds in REGISTERS:
        edges = SANDBOX / ds / "edges-k15-fuzzy.npz"
        if not edges.exists():
            print(f"{ds}: no truth graph ({edges}), skip register")
            results[ds] = {k: None for k in MAPS}
            continue
        x = _norm(DATASETS[ds]["load"]())
        n = int(x.shape[0])
        print(f"{ds}: {n:,} register rows")
        results[ds] = {}
        for key, pack in MAPS.items():
            model_pt = pack / "model.pt"
            if not model_pt.exists():
                print(f"  {key}: no model.pt at {model_pt}, skip")
                results[ds][key] = None
                continue
            t0 = time.time()
            model = ParametricUMAP.load(str(model_pt), device="cuda")
            xy = np.asarray(model.transform(x, batch_size=8192), dtype=np.float32)
            np.save(coords_dir / f"{ds}--{key}.npy", xy)
            ffr = float(quick_ffr(xy, edges, n))
            results[ds][key] = ffr
            print(f"  {key}: OOD-FFR {ffr:.4f} ({(time.time()-t0)/60:.1f} min)")
            del model

    # per-map mean OOD-FFR across the registers where it was scored.
    per_map_mean: dict[str, float | None] = {}
    for key in MAPS:
        vals = [results[ds][key] for ds in REGISTERS
                if results.get(ds, {}).get(key) is not None]
        per_map_mean[key] = float(np.mean(vals)) if vals else None

    scored = {k: v for k, v in per_map_mean.items() if v is not None}
    probe_order = sorted(scored, key=lambda k: scored[k], reverse=True)

    # cross-reference the frozen scorecard's predicted order, if present.
    predicted_order = None
    scorecard = SANDBOX / "p3-scorecard.json"
    if scorecard.exists():
        try:
            predicted_order = json.loads(scorecard.read_text()).get(
                "predicted_diversity_order")
        except Exception:
            predicted_order = None

    out = SANDBOX / "p3-probe-results.json"
    out.write_text(json.dumps({
        "schema": "p3-probe-2026-08-25",
        "registers": REGISTERS,
        "maps": {k: str(v) for k, v in MAPS.items()},
        "per_register_ffr": results,
        "per_map_mean_ffr": per_map_mean,
        "probe_ffr_order": probe_order,
        "predicted_diversity_order": predicted_order,
        "note": "OOD-FFR@0.1%: register embeddings projected through each frozen "
                "champion map, scored vs the register's own exact-k15 fuzzy graph. "
                "Compare probe_ffr_order to predicted_diversity_order to test "
                "whether training-substrate diversity buys OOD generalization.",
    }, indent=1))
    print(f"probe results: {out}")
    print(f"  per-map mean OOD-FFR: {per_map_mean}")
    print(f"  probe_ffr_order: {probe_order}  (predicted: {predicted_order})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
