#!/usr/bin/env python3
"""bmix30 FINALIST CONFIRMATION at 2M with MATCHED rows (owner 2026-08-27) — GPU.

The 1M social-mixture SCREENING sweep (mixture_probe.py) found bmix30 (30%
BALANCED social: reddit/CA/twitter/bluesky) the maximin winner over the broad
probe-register suite. This scorer CONFIRMS that lift at 2M with matched rows:
the minilm-bmix30-2m substrate is 1.4M base rows bit-identical to a subset of
the minilm-mixed-2m baseline + 600K balanced social, so the social 30% is the
SOLE delta between the two 2M heads scored here.

TWO 2M heads, both the champion-bs16k recipe (_CHAMPION_500K: dose4, rankneg
500K = 25% of 2M):

  baseline-0pct   sandbox/2m-knobs/umap-md000-x4bs16k-winner/model.pt
                  (recipe-identical to _CHAMPION_500K; the 0%-social 2M map)
  bmix30-30pct    sandbox/minilm-bmix30-2m/champion-bs16k/model.pt

DELIVERABLE 1 — register suite. For each head x each probe register: load the
register substrate (_norm), transform through the frozen head, score OOD-FFR@0.1%
(quick_ffr_v2, exact-k15 truth auto-resolved next to the register fuzzy npz).
The contaminated probe-code (1724-2252 exact-row overlap with every training
arm) stays EXCLUDED from the maximin/worst/mean; the 0-overlap probe-code-heldout
is INCLUDED. Full-matrix assert (mixture_probe #2): declare a verdict ONLY if
every head x maximin-register cell exists and is finite, else abort LOUDLY.
Reports per-head worst/mean, the per-register bmix30-baseline DELTA (does 30%
social confirm the screening lift at 2M?), and whether probe-code-heldout now
scores legitimately.

DELIVERABLE 2 — PROJECTOR scoring (owner add-on). Also score BOTH heads as
PROJECTORS: (a) receive the a1-common-neutral pool (held out for all heads) —
does the mixture-trained head receive the neutral pool better? (b) transform the
6.25M MiniLM substrate through both heads (lazy per-batch NormMemmap — the 9.6 GB
substrate never materializes) and score vs its sealed 6.25M truth at disc=0.1%xN
(the same instrument transform_gap_audit uses for this exact pair). Tests whether
a coverage-optimized substrate makes a better PROJECTOR head at 3.1x extrapolation.
Transformed coords exported for the compare page.

Import-safe: no torch / model load at import. GPU script; the orchestrator
(pplan_confirm.sh) runs it when the GPU is free.
Output: /data/latent-basemap/sandbox/bmix30-2m-confirm-results.json
Coords: /data/latent-basemap/sandbox/bmix30-2m-confirm-coords/
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np


def _np(o):
    """json.dumps default: numpy scalars (np.bool_/integer/floating) -> python."""
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return str(o)


SANDBOX = Path("/data/latent-basemap/sandbox")
SUBSTRATES = Path("/data/latent-basemap/substrates")

# The 8 BROAD probe registers + the 0-overlap held-out code register.
REGISTERS = (
    "probe-reddit", "probe-ca", "probe-twitter", "probe-bluesky",
    "probe-wiki", "probe-ccweb", "probe-ccscience", "probe-code",
    "probe-code-heldout",
)
# #8 (external review 2026-08-27): the contaminated probe-code overlaps every
# training arm -> EXCLUDE from worst/mean/maximin (still scored + reported for
# the record). probe-code-heldout (0 overlap, incl. bmix30-2m subset of baseline)
# REPLACES it in the maximin.
CODE_EXCLUDED = {"probe-code"}
MAXIMIN_REGS = [r for r in REGISTERS if r not in CODE_EXCLUDED]

# the two 2M heads (same champion-bs16k recipe; social 30% is the sole delta).
MAPS = {
    "baseline-0pct": {
        "model": SANDBOX / "2m-knobs/umap-md000-x4bs16k-winner/model.pt",
        "prefix": "baseline", "social_pct": 0},
    "bmix30-30pct": {
        "model": SANDBOX / "minilm-bmix30-2m/champion-bs16k/model.pt",
        "prefix": "bmix30-2m", "social_pct": 30},
}
BASELINE_KEY = "baseline-0pct"
BMIX_KEY = "bmix30-30pct"


class NormMemmap:
    """Lazy L2-row-normalizing view over a memmap (mirror transform_gap_audit).
    transform() slices X[i:j] per batch and casts to f32 but does NOT normalize;
    pre-_norm on the 9.6 GB 6.25M substrate would materialize it (>=2 GB rule).
    Normalizing inside __getitem__ keeps only one batch resident."""

    def __init__(self, mm):
        self._mm = mm
        self.shape = tuple(mm.shape)
        self.dtype = np.float32

    def __len__(self):
        return int(self.shape[0])

    def __getitem__(self, sl):
        chunk = np.asarray(self._mm[sl], dtype=np.float32)
        nrm = np.linalg.norm(chunk, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        return chunk / nrm


def _finite(v):
    return v is not None and np.isfinite(v)


def _resolve_6250k():
    """(substrate_path, edges_path, knn_indices_path_or_None) for the 6.25M rung.
    Edges resolve from the fixed path or the cluster-spill glob (knobs_2m RUNGS);
    knn_indices is co-located with the substrate or the edges if present, else
    None -> quick_ffr_v2 falls back to top-k-by-fuzzy-weight truth."""
    from knobs_2m import RUNGS
    rung = RUNGS["6250k"]
    sub = Path(rung["substrate"])
    edges = rung["edges"]
    if edges is None:
        hits = sorted(glob.glob(rung["edges_glob"]))
        edges = Path(hits[0]) if hits else None
    else:
        edges = Path(edges)
    knn = None
    for cand in (sub.parent / "knn_indices.npy",
                 (edges.parent / "knn_indices.npy") if edges else None):
        if cand is not None and cand.is_file():
            knn = cand
            break
    return sub, edges, knn


def main(argv: list[str]) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from image_map_pipeline import DATASETS, _norm
    from knobs_2m import quick_ffr_v2 as quick_ffr  # v2 truth-selection (review 2026-08-27)

    from basemap.pumap.parametric_umap.core import ParametricUMAP

    coords_dir = SANDBOX / "bmix30-2m-confirm-coords"
    coords_dir.mkdir(parents=True, exist_ok=True)

    # ---- register substrate + truth cache (lazy, mixture_probe pattern) ----
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

    # ---- projector inputs (resolve once) ----
    a1_sub = SUBSTRATES / "a1-common-neutral" / "substrate.f32.npy"
    a1_truth = SANDBOX / "a1-common-neutral" / "edges-k15-fuzzy.npz"
    a1_ready = a1_sub.exists() and a1_truth.exists()
    a1_x = a1_n = None
    if a1_ready:
        a1_x = _norm(np.asarray(np.load(a1_sub, mmap_mode="r"), dtype=np.float32))
        a1_n = int(a1_x.shape[0])

    sub6, edges6, knn6 = _resolve_6250k()
    proj6_ready = sub6.exists() and edges6 is not None and edges6.exists()

    matrix: dict[str, dict | None] = {}
    projectors: dict[str, dict] = {}

    for key, minfo in MAPS.items():
        model_pt = Path(minfo["model"])
        prefix = minfo["prefix"]
        if not model_pt.exists():
            print(f"{key}: no model.pt at {model_pt}, skip (null row)", flush=True)
            matrix[key] = None
            projectors[key] = {"status": f"SKIPPED (no model.pt at {model_pt})"}
            continue
        print(f"\n=== {key}: social={minfo['social_pct']}% ({model_pt}) ===", flush=True)
        model = ParametricUMAP.load(str(model_pt), device="cuda")

        # ---- DELIVERABLE 1: register suite ----
        row: dict[str, float | None] = {}
        for reg in REGISTERS:
            x, edges, n = _register(reg)
            if x is None:
                print(f"  {reg}: substrate or truth graph missing, null", flush=True)
                row[reg] = None
                continue
            t0 = time.time()
            xy = np.asarray(model.transform(x, batch_size=8192), dtype=np.float32)
            np.save(coords_dir / f"{prefix}--{reg}.npy", xy)
            ffr = float(quick_ffr(xy, edges, n))
            row[reg] = ffr
            mode = getattr(quick_ffr, "last_truth_mode", "?")
            print(f"  {reg}: OOD-FFR {ffr:.4f} [{mode}] ({(time.time()-t0)/60:.1f} min)",
                  flush=True)
        matrix[key] = row

        # ---- DELIVERABLE 2: projector scoring ----
        proj: dict[str, dict] = {}
        # (a) a1-common-neutral pool
        if a1_ready:
            t0 = time.time()
            xy = np.asarray(model.transform(a1_x, batch_size=8192), dtype=np.float32)
            np.save(coords_dir / f"{prefix}__proj-a1neutral.npy", xy)
            ffr = float(quick_ffr(xy, a1_truth, a1_n))
            proj["a1_neutral"] = {"ffr_v2": ffr, "rows": a1_n,
                                  "truth_mode": getattr(quick_ffr, "last_truth_mode", "?"),
                                  "coords_label": f"{prefix}__proj-a1neutral",
                                  "wall_s": round(time.time() - t0, 1)}
            print(f"  proj a1-common-neutral: FFR {ffr:.4f} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
        else:
            proj["a1_neutral"] = {"status": "SKIPPED (a1-common-neutral substrate/truth missing)"}

        # (b) 6.25M substrate @ 3.1x extrapolation, lazy NormMemmap, full disc=0.1%xN
        if proj6_ready:
            t0 = time.time()
            X6 = NormMemmap(np.load(sub6, mmap_mode="r"))
            n6 = len(X6)
            xy = np.asarray(model.transform(X6, batch_size=8192), dtype=np.float32)
            np.save(coords_dir / f"{prefix}__proj-6250k.npy", xy)
            ffr = float(quick_ffr(xy, edges6, n6,
                                  knn_indices_path=(knn6 if knn6 else None)))
            proj["proj_6250k"] = {
                "ffr_v2": ffr, "rows": n6, "extrap": round(n6 / 2_000_000, 3),
                "instrument": "FFR@0.1%-of-N (full-rung 6.25M sealed truth)",
                "truth": str(edges6), "knn_indices": (str(knn6) if knn6 else None),
                "truth_mode": getattr(quick_ffr, "last_truth_mode", "?"),
                "coords_label": f"{prefix}__proj-6250k",
                "wall_s": round(time.time() - t0, 1)}
            print(f"  proj 6.25M (x{n6/2_000_000:.2f}): FFR {ffr:.4f} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
        else:
            proj["proj_6250k"] = {"status": "SKIPPED (6.25M substrate/truth missing)"}

        projectors[key] = proj
        del model

    # ---- per-head worst / mean over the maximin (code-excluded) registers ----
    per_map_worst: dict[str, float | None] = {}
    per_map_mean: dict[str, float | None] = {}
    for key, row in matrix.items():
        vals = [row[r] for r in MAXIMIN_REGS if row and _finite(row.get(r))] if row else []
        per_map_worst[key] = float(min(vals)) if vals else None
        per_map_mean[key] = float(np.mean(vals)) if vals else None

    # ---- #2 full-matrix assert (mixture_probe): verdict ONLY if EVERY
    # head x maximin-register cell exists and is finite; else abort LOUDLY. ----
    missing = [(k, r) for k in MAPS for r in MAXIMIN_REGS
               if matrix.get(k) is None or not _finite((matrix.get(k) or {}).get(r))]
    if missing:
        (SANDBOX / "bmix30-2m-confirm-results.json").write_text(json.dumps({
            "schema": "bmix30-2m-confirm-2026-08-27", "ABORTED": True,
            "reason": "incomplete matrix: missing/non-finite head x maximin-register "
                      "cells; no confirmation verdict",
            "missing_cells": [f"{k}--{r}" for k, r in missing][:60],
            "ffr_matrix": matrix, "code_excluded_from_maximin": sorted(CODE_EXCLUDED),
            "projectors": projectors}, indent=1, default=_np))
        print(f"\nABORT: {len(missing)} missing/non-finite maximin cells -> "
              "NO confirmation verdict", flush=True)
        raise SystemExit(2)

    # ---- per-register bmix30 - baseline delta (does 30% social confirm the lift?) ----
    base_row = matrix[BASELINE_KEY]
    bmix_row = matrix[BMIX_KEY]
    per_register_delta = {}
    for r in MAXIMIN_REGS:
        b, m = base_row.get(r), bmix_row.get(r)
        per_register_delta[r] = (round(float(m) - float(b), 4)
                                 if _finite(b) and _finite(m) else None)

    # probe-code-heldout legitimacy: it now enters the maximin, so both heads must
    # have scored it finite (the whole point of building the 0-overlap register).
    heldout_base = base_row.get("probe-code-heldout")
    heldout_bmix = bmix_row.get("probe-code-heldout")
    heldout_legit = _finite(heldout_base) and _finite(heldout_bmix)

    # confirmation verdict: maximin (worst-register) not worse AND mean not worse.
    worst_delta = (round(per_map_worst[BMIX_KEY] - per_map_worst[BASELINE_KEY], 4)
                   if _finite(per_map_worst[BMIX_KEY]) and _finite(per_map_worst[BASELINE_KEY])
                   else None)
    mean_delta = (round(per_map_mean[BMIX_KEY] - per_map_mean[BASELINE_KEY], 4)
                  if _finite(per_map_mean[BMIX_KEY]) and _finite(per_map_mean[BASELINE_KEY])
                  else None)
    maximin_winner = max((k for k in MAPS), key=lambda k: per_map_worst[k])
    confirmed = bool(worst_delta is not None and worst_delta >= 0.0
                     and mean_delta is not None and mean_delta >= 0.0)

    # projector deltas (bmix30 - baseline), where both heads scored.
    def _proj_delta(name):
        b = (projectors.get(BASELINE_KEY, {}).get(name, {}) or {}).get("ffr_v2")
        m = (projectors.get(BMIX_KEY, {}).get(name, {}) or {}).get("ffr_v2")
        return (round(float(m) - float(b), 4)
                if _finite(b) and _finite(m) else None)

    out = SANDBOX / "bmix30-2m-confirm-results.json"
    out.write_text(json.dumps({
        "schema": "bmix30-2m-confirm-2026-08-27",
        "registers": list(REGISTERS),
        "maximin_registers": MAXIMIN_REGS,
        "code_excluded_from_maximin": sorted(CODE_EXCLUDED),
        "code_heldout_included_in_maximin": True,
        "maps": {k: {"model": str(v["model"]), "social_pct": v["social_pct"]}
                 for k, v in MAPS.items()},
        "ffr_matrix": matrix,
        "per_map_worst_register_ffr": per_map_worst,
        "per_map_mean_ffr": per_map_mean,
        "maximin_winner": maximin_winner,
        "per_register_delta_bmix30_minus_baseline": per_register_delta,
        "worst_register_delta_bmix30_minus_baseline": worst_delta,
        "mean_delta_bmix30_minus_baseline": mean_delta,
        "bmix30_confirms_screening_lift": confirmed,
        "probe_code_heldout": {
            "baseline": heldout_base, "bmix30": heldout_bmix,
            "delta": per_register_delta.get("probe-code-heldout"),
            "scores_legitimately": heldout_legit},
        "projectors": projectors,
        "projector_delta_bmix30_minus_baseline": {
            "a1_neutral": _proj_delta("a1_neutral"),
            "proj_6250k": _proj_delta("proj_6250k")},
        "note": (
            "OOD-FFR@0.1% (quick_ffr_v2): each probe register projected through each "
            "frozen 2M head, scored vs the register's own exact-k15 truth. Both heads are "
            "the champion-bs16k recipe (dose4, rankneg 500K = 25% of 2M); bmix30-2m adds "
            "30% BALANCED social to a base bit-identical to a subset of the 2M baseline, so "
            "social 30% is the sole delta. probe-code EXCLUDED from worst/mean/maximin "
            "(contaminated); probe-code-heldout (0 overlap) INCLUDED. bmix30_confirms_"
            "screening_lift = worst-register delta >=0 AND mean delta >=0. PROJECTOR block: "
            "both heads also score as projectors on a1-common-neutral (neutral pool "
            "reception) and on the 6.25M substrate at disc=0.1%xN (3.1x extrapolation, "
            "lazy NormMemmap, same instrument as transform_gap_audit) — tests whether the "
            "coverage-optimized substrate makes a better projector head."),
    }, indent=1, default=_np))
    print(f"\nresults: {out}", flush=True)
    print(f"  per-head worst-register FFR: {per_map_worst}")
    print(f"  per-head mean FFR:           {per_map_mean}")
    print(f"  worst-register delta (bmix30-baseline): {worst_delta}")
    print(f"  mean delta (bmix30-baseline):           {mean_delta}")
    print(f"  bmix30 confirms screening lift: {confirmed}")
    print(f"  probe-code-heldout scores legitimately: {heldout_legit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
