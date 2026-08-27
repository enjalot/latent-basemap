#!/usr/bin/env python3
"""P2, step 2: OOD-probe the jina-space champion map with two social-media
register substrates (reddit TL;DR, community-archive tweets).

Projects each document-prompted jina register substrate (built by
``p2_jina_embed.py``) through the frozen jina champion map and scores
fold-faithfulness (FFR@0.1%) against that register's OWN jina-space truth graph
(built by the orchestrator via ``image_map_pipeline`` knn+fuzzy on these
datasets).  A low FFR means the champion map fails to receive that register's
neighborhood structure.

The champion trained on NORMALIZED jina vectors, so we normalize the substrate
here too (``_norm`` from image_map_pipeline).

Modeled on ``register_ood_probe.py`` / ``p3_probe.py``.

GPU script (ParametricUMAP.transform): run by the orchestrator when the GPU is
free.  Import-safe: no torch/model load at import time.

Usage: p2_jina_probe.py
Output: /data/latent-basemap/sandbox/p2-jina-probe-results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

SANDBOX = Path("/data/latent-basemap/sandbox")
SUBSTRATES = Path("/data/latent-basemap/substrates")
CHAMPION = SANDBOX / "jina-multi-2m/champion-bs16k/model.pt"

# register dataset id -> substrate .f16.npy (from p2_jina_embed.py).  The truth
# graph is expected at SANDBOX/<ds>/edges-k15-fuzzy.npz.
REGISTERS = {
    "reddit-jina-250k": SUBSTRATES / "reddit-jina-250k" / "substrate.f16.npy",
    "ca-jina-250k": SUBSTRATES / "ca-jina-250k" / "substrate.f16.npy",
}


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from image_map_pipeline import _norm
    from knobs_2m import quick_ffr_v2 as quick_ffr  # v2 truth-selection (review 2026-08-27)

    from basemap.pumap.parametric_umap.core import ParametricUMAP

    assert CHAMPION.exists(), f"no champion model at {CHAMPION}"

    coords_dir = SANDBOX / "p2-jina-probe-coords"
    coords_dir.mkdir(parents=True, exist_ok=True)

    model = ParametricUMAP.load(str(CHAMPION), device="cuda")

    ffr_by_register: dict[str, float] = {}
    for ds, sub_path in REGISTERS.items():
        assert sub_path.exists(), f"no substrate for {ds}: {sub_path}"
        edges = SANDBOX / ds / "edges-k15-fuzzy.npz"
        assert edges.exists(), f"no truth graph for {ds}: {edges}"

        x = _norm(np.load(sub_path).astype(np.float32))
        n = int(x.shape[0])
        print(f"{ds}: {n:,} register rows", flush=True)

        t0 = time.time()
        xy = np.asarray(model.transform(x, batch_size=8192), dtype=np.float32)
        np.save(coords_dir / f"{ds}.npy", xy)
        ffr = float(quick_ffr(xy, edges, n))
        ffr_by_register[ds] = ffr
        print(f"  OOD-FFR {ffr:.4f} ({(time.time()-t0)/60:.1f} min)", flush=True)

    mean_ffr = float(np.mean(list(ffr_by_register.values())))
    worst_register = min(ffr_by_register, key=ffr_by_register.get)

    out = SANDBOX / "p2-jina-probe-results.json"
    payload: dict[str, object] = dict(ffr_by_register)
    payload.update({
        "mean": mean_ffr,
        "worst_register": worst_register,
        "schema": "p2-jina-probe-2026-08-25",
        "champion": str(CHAMPION),
        "note": "OOD-FFR@0.1%: document-prompted jina register substrates "
                "projected through the frozen jina champion map, scored vs each "
                "register's own exact-k15 fuzzy graph. Normalized jina vectors "
                "(champion trained on normalized input).",
    })
    out.write_text(json.dumps(payload, indent=1))
    print(f"probe results: {out}", flush=True)
    print(f"  per-register OOD-FFR: {ffr_by_register}")
    print(f"  mean={mean_ffr:.4f}  worst_register={worst_register}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
