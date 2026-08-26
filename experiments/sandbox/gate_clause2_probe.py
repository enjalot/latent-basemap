#!/usr/bin/env python3
"""h4096-gate clause 2 (external review 2026-08-26): OOD non-regression of the
width arm. Projects the reddit-jina-250k + ca-jina-250k register substrates
through the decomposition maps x8-h3072 (width) and x8-h2048 (exposure), scores
OOD-FFR on the FROZEN register truth graphs, and compares worst-register to the
champion-baseline P2 numbers (reddit 0.2219 / CA 0.1347). Non-regression =
neither register drops materially on the width arm vs the exposure arm / champion.

CPU-only (device="cpu") so it never competes with the concurrent GPU training.
Reuses p2_jina_probe's exact methodology (_norm, per-register truth graph, quick_ffr).
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

SANDBOX = Path("/data/latent-basemap/sandbox")
SUBSTRATES = Path("/data/latent-basemap/substrates")
MAPS = {
    "x8-h3072": SANDBOX / "jina-multi-2m/champion-x8-h3072/model.pt",
    "x8-h2048": SANDBOX / "jina-multi-2m/champion-x8-h2048/model.pt",
}
REGISTERS = {
    "reddit-jina-250k": SUBSTRATES / "reddit-jina-250k" / "substrate.f16.npy",
    "ca-jina-250k": SUBSTRATES / "ca-jina-250k" / "substrate.f16.npy",
}
# champion-baseline P2 OOD-FFR (the non-regression reference)
CHAMPION_P2 = {"reddit-jina-250k": 0.2219, "ca-jina-250k": 0.1347}


def main() -> int:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from image_map_pipeline import _norm
    from knobs_2m import quick_ffr
    from basemap.pumap.parametric_umap.core import ParametricUMAP

    reg_x, reg_edges = {}, {}
    for ds, sp in REGISTERS.items():
        reg_x[ds] = _norm(np.load(sp).astype(np.float32))
        reg_edges[ds] = SANDBOX / ds / "edges-k15-fuzzy.npz"
        assert reg_edges[ds].exists(), reg_edges[ds]

    out: dict[str, dict] = {}
    for mname, mpath in MAPS.items():
        assert mpath.exists(), mpath
        model = ParametricUMAP.load(str(mpath), device="cpu")
        out[mname] = {}
        for ds, x in reg_x.items():
            t0 = time.time()
            xy = np.asarray(model.transform(x, batch_size=8192), dtype=np.float32)
            ffr = float(quick_ffr(xy, reg_edges[ds], x.shape[0]))
            out[mname][ds] = ffr
            print(f"{mname} {ds}: OOD-FFR {ffr:.4f} ({(time.time()-t0)/60:.1f} min)", flush=True)

    # non-regression: width arm (x8-h3072) vs exposure arm (x8-h2048) and vs champion P2
    verdict = {}
    for ds in REGISTERS:
        w = out["x8-h3072"][ds]; e = out["x8-h2048"][ds]; c = CHAMPION_P2[ds]
        verdict[ds] = {"width_h3072": w, "exposure_h2048": e, "champion_p2": c,
                       "width_vs_exposure": round(w - e, 4),
                       "width_vs_champion": round(w - c, 4)}
    payload = {"schema": "h4096-gate-clause2-2026-08-26", "maps": out,
               "verdict": verdict,
               "non_regression_note": "clause 2 passes if the width arm (x8-h3072) does NOT drop "
                                      "materially on either register vs the exposure arm or the champion P2."}
    p = SANDBOX / "h4096-gate-clause2.json"
    p.write_text(json.dumps(payload, indent=1))
    print(f"wrote {p}")
    print(json.dumps(verdict, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
