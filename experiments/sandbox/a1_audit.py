#!/usr/bin/env python3
"""A1 cross-scale common-probe audit, step 2 (GPU, orchestrator-run).

ONE frozen common sample + ONE common truth graph + the broad register suite,
projected through EVERY surviving MiniLM parametric-UMAP head, all scored on one
instrument (quick-FFR@0.1%). Answers the operational question: does a cheap-N
head match the 100M head as a FUNCTION OF SCALE? Each head is scored on

  (a) the a1-common truth  — the 250K neutral pool sample frozen by
      build_a1_common.py; truth graph at sandbox/a1-common/edges-k15-fuzzy.npz.
  (b) each of the 8 BROAD register truths — reddit / ca / twitter / bluesky /
      wiki / ccweb / ccscience / code — each scored vs its OWN exact-k15 fuzzy
      graph (sandbox/probe-<reg>/edges-k15-fuzzy.npz).

For every head: _norm the RELEVANT substrate (a1-common for the common column,
each probe register for its own column), ParametricUMAP.load(model.pt,
device="cuda"), transform, quick_ffr vs that column's truth. Per head we also
report worst_register (its weakest-received corpus) and mean_register.

NOTE the heads SPAN RECIPE GENERATIONS: the 2M/6.25M/12.5M/25M knob-sandbox
winners and the 50M/100M round checkpoints were minted under evolving recipes.
The operational scale question is unaffected — each is a frozen map judged only
on how faithfully it receives the common sample and the registers. This caveat
is stamped into the results JSON.

Missing model.pt -> the whole head row is null and reported. Missing substrate or
truth graph for a column -> that cell is null.

GPU script (ParametricUMAP.transform): import-safe (no torch / model load at
import time); the orchestrator runs it when the GPU is free.

Usage:
    a1_audit.py                 # score every head
    a1_audit.py <head> [<head>..]  # score only the named heads (keys below)
Output: /data/latent-basemap/sandbox/a1-audit-results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

SANDBOX = Path("/data/latent-basemap/sandbox")
SUBSTRATES = Path("/data/latent-basemap/substrates")

# the frozen common probe sample (built by build_a1_common.py).
COMMON = "a1-common"
COMMON_SUBSTRATE = SUBSTRATES / COMMON / "substrate.f32.npy"
COMMON_EDGES = SANDBOX / COMMON / "edges-k15-fuzzy.npz"

# output column -> (substrate.f32.npy dir under SUBSTRATES, truth-graph dir under SANDBOX).
# For the registers the two dirs share the "probe-<reg>" name.
REGISTERS = {
    "reddit": "probe-reddit",
    "ca": "probe-ca",
    "twitter": "probe-twitter",
    "bluesky": "probe-bluesky",
    "wiki": "probe-wiki",
    "ccweb": "probe-ccweb",
    "ccscience": "probe-ccscience",
    "code": "probe-code",
}

# head key -> ordered list of candidate model.pt paths (first that loads wins).
# The 12.5M arm (umap-md000-x4-fneg10) and 25M arm (umap-md000-x2-fneg10-hostint8)
# were selected by highest quick_ffr_at_0.1pct in their knob-sandbox summaries
# (12.5M: 0.4667 x4 vs 0.4380 x2; 25M: sole arm 0.4775). The 100M head has two
# candidate checkpoints; whichever loads is used and its path recorded.
HEADS = {
    "2M": [SANDBOX / "2m-knobs/umap-md000-x4bs16k-winner/model.pt"],
    "6.25M-rank25": [SANDBOX / "6250k-knobs/umap-md000-x4bs16k-winner-rank25/model.pt"],
    "6.25M-norank": [SANDBOX / "6250k-knobs/umap-md000-x4bs16k-winner-norank/model.pt"],
    "12.5M": [SANDBOX / "12500k-knobs/umap-md000-x4-fneg10/model.pt"],
    "25M": [SANDBOX / "25000k-knobs/umap-md000-x2-fneg10-hostint8/model.pt"],
    "50M": [Path("/data/checkpoints/pumap/maps/minilm-50m-r0267-seed42/model.pt")],
    "100M": [
        Path("/data/checkpoints/pumap/maps/minilm-100m-r0268-preview-seed42/model.pt"),
        Path("/data/latent-basemap/runs/round-0268/attempt5/artifacts/"
             "minilm-mixed-100000k-fneg-x2-md000-hostint8-seed43-r0268-v1/model.pt"),
    ],
}


def main(argv: list[str]) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from image_map_pipeline import _norm
    from knobs_2m import quick_ffr

    from basemap.pumap.parametric_umap.core import ParametricUMAP

    want = set(argv[1:])
    heads = {k: v for k, v in HEADS.items() if not want or k in want}

    coords_dir = SANDBOX / "a1-audit-coords"
    coords_dir.mkdir(parents=True, exist_ok=True)

    # lazy, cached: column key -> (normed substrate, edges path, n) or (None, None, None).
    # Columns: "common" (the a1-common sample) + the 8 registers.
    col_cache: dict[str, tuple] = {}

    def _column(col: str):
        if col not in col_cache:
            if col == "common":
                sub, edges = COMMON_SUBSTRATE, COMMON_EDGES
            else:
                reg = REGISTERS[col]
                sub = SUBSTRATES / reg / "substrate.f32.npy"
                edges = SANDBOX / reg / "edges-k15-fuzzy.npz"
            if not sub.exists() or not edges.exists():
                print(f"  [{col}] substrate or truth graph missing "
                      f"(sub={sub.exists()} edges={edges.exists()}), null cell",
                      flush=True)
                col_cache[col] = (None, None, None)
            else:
                x = _norm(np.asarray(np.load(sub, mmap_mode="r"), dtype=np.float32))
                col_cache[col] = (x, edges, int(x.shape[0]))
        return col_cache[col]

    def _score(model, col: str, head: str) -> float | None:
        x, edges, n = _column(col)
        if x is None:
            return None
        t0 = time.time()
        xy = np.asarray(model.transform(x, batch_size=8192), dtype=np.float32)
        np.save(coords_dir / f"{head}--{col}.npy", xy)
        ffr = float(quick_ffr(xy, edges, n))
        print(f"  {col}: FFR {ffr:.4f} ({(time.time()-t0)/60:.1f} min)", flush=True)
        return ffr

    results: dict[str, dict] = {}
    head_model_paths: dict[str, str | None] = {}
    all_cols = ["common"] + list(REGISTERS)

    for head, candidates in heads.items():
        model_pt = next((p for p in candidates if p.exists()), None)
        if model_pt is None:
            print(f"{head}: no model.pt among {[str(p) for p in candidates]}, "
                  f"null row", flush=True)
            results[head] = {c: None for c in all_cols} | {
                "worst_register": None, "mean_register": None}
            head_model_paths[head] = None
            continue
        print(f"{head}: loading {model_pt}", flush=True)
        head_model_paths[head] = str(model_pt)
        model = ParametricUMAP.load(str(model_pt), device="cuda")
        row: dict[str, float | None] = {}
        row["common"] = _score(model, "common", head)
        for col in REGISTERS:
            row[col] = _score(model, col, head)
        reg_vals = [row[c] for c in REGISTERS if row[c] is not None]
        row["worst_register"] = float(min(reg_vals)) if reg_vals else None
        row["mean_register"] = float(np.mean(reg_vals)) if reg_vals else None
        results[head] = row
        del model

    out = SANDBOX / "a1-audit-results.json"
    out.write_text(json.dumps({
        "schema": "a1-cross-scale-audit-2026-08-26",
        "common_sample": {
            "substrate": str(COMMON_SUBSTRATE),
            "truth_graph": str(COMMON_EDGES),
            "desc": "250K random (seed 42) sample of the sealed 2M MiniLM pool "
                    "substrate; the neutral common instrument for every head.",
        },
        "registers": list(REGISTERS),
        "heads": {k: (head_model_paths.get(k)) for k in HEADS},
        "results": results,
        "note": "quick-FFR@0.1%: each frozen head projects the a1-common sample "
                "(scored vs the common truth graph) and each broad register "
                "(scored vs that register's OWN exact-k15 fuzzy graph). "
                "worst_register / mean_register summarize the 8 register columns. "
                "HEADS SPAN RECIPE GENERATIONS (2M/6.25M/12.5M/25M knob-sandbox "
                "winners + 50M/100M round checkpoints, evolving recipes); the "
                "operational scale question is unaffected since each head is a "
                "frozen map judged only on reception. Missing model.pt -> null "
                "row; missing substrate/truth graph -> null cell. 12.5M arm "
                "umap-md000-x4-fneg10 and 25M arm umap-md000-x2-fneg10-hostint8 "
                "chosen by highest summary quick_ffr_at_0.1pct.",
    }, indent=1))
    print(f"\nresults: {out}", flush=True)
    for head, row in results.items():
        print(f"  {head}: common={row.get('common')} "
              f"worst_register={row.get('worst_register')} "
              f"mean_register={row.get('mean_register')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
