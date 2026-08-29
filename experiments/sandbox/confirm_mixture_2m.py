#!/usr/bin/env python3
"""REUSABLE mixture-arm CONFIRMATION scorer (owner STANDING CHANGE 2026-08-28:
projector-scoring is now a standard column for EVERY mixture arm).

Generalizes confirm_bmix30_2m.py: takes the mixture arm as a parameter
(argv[1] = the map key, or env MIXTURE_MAP) and scores {baseline-0pct, <that
mixture map>} on:

  (a) the full 9-register suite (8 broad probes + probe-code-heldout; the
      contaminated probe-code is EXCLUDED from the maximin, probe-code-heldout is
      INCLUDED) -> per-map worst/mean maximin + per-register delta vs baseline;
  (b) the PROJECTOR add-on: (a1) a1-common-neutral reception + (a2) 6.25M-substrate
      transform via lazy NormMemmap -> projector delta vs baseline.

Both mixture heads and their baseline are the champion-bs16k recipe (dose4,
rankneg = 25% of N). The mixture head's model.pt resolves as
  sandbox/<map key>/champion-bs16k/model.pt
The baseline is SCALE-SELECTABLE:
  2M arms -> the 2m-knobs winner  (sandbox/2m-knobs/umap-md000-x4bs16k-winner/model.pt)
  1M arms -> minilm-mix-1m/rankfrac-25 (sandbox/minilm-mix-1m/rankfrac-25/model.pt)

Supported mixture arms: minilm-bmix30-2m (2M), minilm-bmix10cp-2m (2M),
minilm-bmix40-1m (1M), minilm-bmix50-1m (1M).

All confirm_bmix30_2m.py logic is preserved: full-matrix assert (verdict ONLY if
every head x maximin-register cell exists and is finite, else abort LOUDLY),
NormMemmap lazy 6.25M transform, _np json default, exact FFR instrument strings.

Import-safe: no torch / model load at import. GPU script; the orchestrator runs
it when the GPU is free.
Output: /data/latent-basemap/sandbox/<prefix>-confirm-results.json
Coords: /data/latent-basemap/sandbox/<prefix>-confirm-coords/
"""
from __future__ import annotations

import glob
import json
import os
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
# the contaminated probe-code overlaps every training arm -> EXCLUDE from
# worst/mean/maximin (still scored + reported). probe-code-heldout (0 overlap)
# REPLACES it in the maximin.
CODE_EXCLUDED = {"probe-code"}
MAXIMIN_REGS = [r for r in REGISTERS if r not in CODE_EXCLUDED]

# ---- scale-selectable baseline ('0%-social' reference head) ----
SCALE_BASELINE = {
    "2m": {"model": SANDBOX / "2m-knobs/umap-md000-x4bs16k-winner/model.pt",
           "N": 2_000_000,
           "map_name": "minilm-mixed-2m (2m-knobs winner)"},
    "1m": {"model": SANDBOX / "minilm-mix-1m/rankfrac-25/model.pt",
           "N": 1_000_000,
           "map_name": "minilm-mix-1m/rankfrac-25"},
}

# ---- supported mixture arms. model.pt resolves as
#      sandbox/<arm>/champion-bs16k/model.pt ; baseline picked by scale. ----
ARM_REGISTRY = {
    "minilm-bmix30-2m":  {"scale": "2m", "social_pct": 30,
                          "prefix": "bmix30-2m",  "key": "bmix30-30pct"},
    "minilm-bmix10cp-2m": {"scale": "2m", "social_pct": 10,
                          "prefix": "bmix10cp-2m", "key": "bmix10cp-10pct"},
    "minilm-bmix40-1m":  {"scale": "1m", "social_pct": 40,
                          "prefix": "bmix40-1m",  "key": "bmix40-40pct"},
    "minilm-bmix50-1m":  {"scale": "1m", "social_pct": 50,
                          "prefix": "bmix50-1m",  "key": "bmix50-50pct"},
}
BASELINE_KEY = "baseline-0pct"


def resolve_arm(arm: str):
    """Return (MAPS, BMIX_KEY, prefix, scale, baseline_N, baseline_map_name) for
    the requested mixture arm. Does NOT touch the GPU or load any model."""
    if arm not in ARM_REGISTRY:
        raise SystemExit(
            f"unknown mixture arm {arm!r}; supported: {sorted(ARM_REGISTRY)}")
    info = ARM_REGISTRY[arm]
    scale = info["scale"]
    base = SCALE_BASELINE[scale]
    prefix = info["prefix"]
    bmix_key = info["key"]
    maps = {
        BASELINE_KEY: {
            "model": base["model"], "prefix": "baseline", "social_pct": 0},
        bmix_key: {
            "model": SANDBOX / arm / "champion-bs16k" / "model.pt",
            "prefix": prefix, "social_pct": info["social_pct"]},
    }
    return maps, bmix_key, prefix, scale, base["N"], base["map_name"]


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
    arm = (argv[1] if len(argv) > 1 else os.environ.get("MIXTURE_MAP", "")).strip()
    if not arm:
        raise SystemExit(
            "usage: confirm_mixture_2m.py <mixture-arm>  (or env MIXTURE_MAP); "
            f"supported: {sorted(ARM_REGISTRY)}")
    MAPS, BMIX_KEY, prefix, scale, baseline_N, baseline_map_name = resolve_arm(arm)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from image_map_pipeline import DATASETS, _norm  # noqa: F401 (parity w/ bmix30)
    from knobs_2m import quick_ffr_v2 as quick_ffr  # v2 truth-selection
    from knobs_2m import quick_ffr_v2_split  # P0.3c member/unseen split

    from basemap.pumap.parametric_umap.core import ParametricUMAP

    coords_dir = SANDBOX / f"{prefix}-confirm-coords"
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
        pfx = minfo["prefix"]
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
            np.save(coords_dir / f"{pfx}--{reg}.npy", xy)
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
            np.save(coords_dir / f"{pfx}__proj-a1neutral.npy", xy)
            ffr = float(quick_ffr(xy, a1_truth, a1_n))
            proj["a1_neutral"] = {"ffr_v2": ffr, "rows": a1_n,
                                  "truth_mode": getattr(quick_ffr, "last_truth_mode", "?"),
                                  "coords_label": f"{pfx}__proj-a1neutral",
                                  "wall_s": round(time.time() - t0, 1)}
            print(f"  proj a1-common-neutral: FFR {ffr:.4f} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
        else:
            proj["a1_neutral"] = {"status": "SKIPPED (a1-common-neutral substrate/truth missing)"}

        # (b) 6.25M substrate extrapolation, lazy NormMemmap, full disc=0.1%xN
        if proj6_ready:
            t0 = time.time()
            X6 = NormMemmap(np.load(sub6, mmap_mode="r"))
            n6 = len(X6)
            xy = np.asarray(model.transform(X6, batch_size=8192), dtype=np.float32)
            np.save(coords_dir / f"{pfx}__proj-6250k.npy", xy)
            # P0.3c (corrected 2026-08-29): the OPTIMISTIC projection case. The head's
            # training rows are a MEMBER of the 6.25M target — but the composition is by
            # STRIDE, not a 1M prefix: MiniLM-2M is the nested prefix of 6.25M, and the
            # smaller heads are STRIDES of that 2M (1M = every-2nd row, 500K = every-4th),
            # NOT prefixes (minilm-mix-1m is p4_slice_substrates stride=2, not a prefix).
            # So members = {0, stride, 2*stride, ...} below 2M, exact by construction.
            _stride = 2_000_000 // baseline_N          # 2m->1, 1m->2, 500k->4
            proj_member_mask = np.zeros(n6, dtype=bool)
            proj_member_mask[0:2_000_000:_stride] = True
            sp = quick_ffr_v2_split(xy, edges6, n6, member_mask=proj_member_mask,
                                    knn_indices_path=(knn6 if knn6 else None))
            ffr = float(sp["overall"])
            proj["proj_6250k"] = {
                "ffr_v2": ffr,
                "ffr_v2_member": sp["member"], "ffr_v2_unseen": sp["unseen"],
                "member_frac": sp["member_frac"],
                "n_member_queries": sp["n_member"], "n_unseen_queries": sp["n_unseen"],
                "member_note": (
                    f"{baseline_N:,}-row head = stride-{_stride} subsample of the 2M "
                    f"nested-prefix of 6.25M; members = indices 0,{_stride},{2*_stride},... "
                    f"below 2,000,000 (exact by construction, NOT a 1M prefix). "
                    f"member_frac ~ {baseline_N:,}/N."),
                "rows": n6, "extrap": round(n6 / baseline_N, 3),
                "instrument": "FFR@0.1%-of-N (full-rung 6.25M sealed truth)",
                "truth": str(edges6), "knn_indices": (str(knn6) if knn6 else None),
                "truth_mode": sp["truth_mode"],
                "coords_label": f"{pfx}__proj-6250k",
                "wall_s": round(time.time() - t0, 1)}
            print(f"  proj 6.25M (x{n6/baseline_N:.2f}): FFR {ffr:.4f} "
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

    out = SANDBOX / f"{prefix}-confirm-results.json"

    # ---- full-matrix assert (mixture_probe #2): verdict ONLY if EVERY
    # head x maximin-register cell exists and is finite; else abort LOUDLY. ----
    missing = [(k, r) for k in MAPS for r in MAXIMIN_REGS
               if matrix.get(k) is None or not _finite((matrix.get(k) or {}).get(r))]
    if missing:
        out.write_text(json.dumps({
            "schema": f"{prefix}-confirm-2026-08-28", "arm": arm, "ABORTED": True,
            "reason": "incomplete matrix: missing/non-finite head x maximin-register "
                      "cells; no confirmation verdict",
            "missing_cells": [f"{k}--{r}" for k, r in missing][:60],
            "ffr_matrix": matrix, "code_excluded_from_maximin": sorted(CODE_EXCLUDED),
            "projectors": projectors}, indent=1, default=_np))
        print(f"\nABORT: {len(missing)} missing/non-finite maximin cells -> "
              "NO confirmation verdict", flush=True)
        raise SystemExit(2)

    # ---- per-register mixture - baseline delta (does the social share confirm the lift?) ----
    base_row = matrix[BASELINE_KEY]
    bmix_row = matrix[BMIX_KEY]
    per_register_delta = {}
    for r in MAXIMIN_REGS:
        b, m = base_row.get(r), bmix_row.get(r)
        per_register_delta[r] = (round(float(m) - float(b), 4)
                                 if _finite(b) and _finite(m) else None)

    # probe-code-heldout legitimacy: it enters the maximin, so both heads must
    # have scored it finite (the whole point of the 0-overlap register).
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

    # projector deltas (mixture - baseline), where both heads scored.
    def _proj_delta(name):
        b = (projectors.get(BASELINE_KEY, {}).get(name, {}) or {}).get("ffr_v2")
        m = (projectors.get(BMIX_KEY, {}).get(name, {}) or {}).get("ffr_v2")
        return (round(float(m) - float(b), 4)
                if _finite(b) and _finite(m) else None)

    out.write_text(json.dumps({
        "schema": f"{prefix}-confirm-2026-08-28",
        "arm": arm,
        "scale": scale,
        "baseline_map": baseline_map_name,
        "baseline_N": baseline_N,
        "mixture_key": BMIX_KEY,
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
        "per_register_delta_mix_minus_baseline": per_register_delta,
        "worst_register_delta_mix_minus_baseline": worst_delta,
        "mean_delta_mix_minus_baseline": mean_delta,
        "mix_confirms_screening_lift": confirmed,
        "probe_code_heldout": {
            "baseline": heldout_base, "mix": heldout_bmix,
            "delta": per_register_delta.get("probe-code-heldout"),
            "scores_legitimately": heldout_legit},
        "projectors": projectors,
        "projector_delta_mix_minus_baseline": {
            "a1_neutral": _proj_delta("a1_neutral"),
            "proj_6250k": _proj_delta("proj_6250k")},
        "note": (
            f"OOD-FFR@0.1% (quick_ffr_v2): each probe register projected through each "
            f"frozen {scale.upper()} head, scored vs the register's own exact-k15 truth. "
            f"Both heads are the champion-bs16k recipe (dose4, rankneg = 25% of N); {arm} "
            f"adds {MAPS[BMIX_KEY]['social_pct']}% BALANCED social to a base bit-identical to "
            f"a subset of the {scale.upper()} baseline, so the social share is the sole delta. "
            f"probe-code EXCLUDED from worst/mean/maximin (contaminated); probe-code-heldout "
            f"(0 overlap) INCLUDED. mix_confirms_screening_lift = worst-register delta >=0 AND "
            f"mean delta >=0. PROJECTOR block: both heads also score as projectors on "
            f"a1-common-neutral (neutral pool reception) and on the 6.25M substrate at "
            f"disc=0.1%xN ({round(6_250_000/baseline_N,2)}x extrapolation, lazy NormMemmap, "
            f"same instrument as transform_gap_audit) — tests whether the coverage-optimized "
            f"substrate makes a better projector head."),
    }, indent=1, default=_np))
    print(f"\nresults: {out}", flush=True)
    print(f"  arm={arm} scale={scale} baseline={baseline_map_name}")
    print(f"  per-head worst-register FFR: {per_map_worst}")
    print(f"  per-head mean FFR:           {per_map_mean}")
    print(f"  worst-register delta (mix-baseline): {worst_delta}")
    print(f"  mean delta (mix-baseline):           {mean_delta}")
    print(f"  mix confirms screening lift: {confirmed}")
    print(f"  probe-code-heldout scores legitimately: {heldout_legit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
