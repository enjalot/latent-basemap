#!/usr/bin/env python3
"""jina 27-REGISTER mixture-arm CONFIRMATION scorer (JINA_SWEEP_PROPOSAL.md
2026-08-28). The jina analog of confirm_mixture_2m.py.

Scores {jina 0% baseline champion, <jina mixture arm>} on the 27-register jina
maximin suite — ALL 27 enter the maximin (languages IN the maximin is the whole
point; there is no contaminated code register on the jina side, so NO exclusions):

  20 languages   probe-lang-<lang>-jina   (<lang> = full fineweb2 code, e.g. arb_Arab)
   4 social      {reddit,ca,twitter,bluesky}-jina-250k   (reddit/ca exist)
   3 EN base     probe-{fineweb,rpj,pile}-jina

Both heads are the champion-bs16k recipe (= the jina-multi-2m champion recipe,
_CHAMPION_500K: md000, dose4, rankneg 500K = 25% of 2M, fneg1.0, tanh4.0, pos0.10,
bs16k). The 0% baseline head is the EXISTING sandbox/jina-multi-2m/champion-bs16k
(its substrate IS the undisplaced base). The mixture head resolves as
sandbox/<arm>/champion-bs16k/model.pt.

PROJECTOR column (owner resolution 2026-08-28), both heads:
  1. jina_6m_transform (PRIMARY, required): transform each head over the jina-6m
     (6.25M) substrate via lazy NormMemmap and score vs its jina-6m truth
     (sandbox/jina-multi-6m/edges-k15-fuzzy.npz). The true next-scale-up test.
  2. jina_neutral_pooled (SECONDARY, skip-clean-if-truth-absent): a fixed-seed
     pooled uniform neutral draw across the P-A/P-B/P-C holdouts, its own knn+fuzzy
     truth. Resolves substrates/jina-neutral-pooled/ + sandbox/jina-neutral-pooled/.
     PENDING; if absent at score time, SKIP cleanly (null + note) — NOT a blocker.
  NB: the MiniLM a1-common-neutral is MiniLM-space and is DELIBERATELY NOT used here.

All confirm_mixture_2m.py logic preserved: full-matrix assert (verdict ONLY if
every head x maximin-register cell exists and is finite, else abort LOUDLY), lazy
NormMemmap, _np json default, exact FFR instrument (quick_ffr_v2, _norm, D768).
Register substrates/truths are GPU prereqs (P-A/P-B/P-C) and may be PENDING: the
scorer resolves + prints exists/missing and skips missing registers cleanly (null
rows, mixture_probe pattern) rather than crashing.

Import-safe: no torch / model load at import. GPU script; the orchestrator runs
it when the GPU is free.
Output: /data/latent-basemap/sandbox/<arm>-jina-confirm-results.json
Coords: /data/latent-basemap/sandbox/<arm>-jina-confirm-coords/
"""
from __future__ import annotations

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

# ---- 27-register jina maximin suite (NO exclusions; languages IN the maximin) ----
_JINA_LANGS = ("arb_Arab", "ces_Latn", "cmn_Hani", "deu_Latn", "ell_Grek",
               "fra_Latn", "hin_Deva", "ind_Latn", "ita_Latn", "jpn_Jpan",
               "kor_Hang", "nld_Latn", "pol_Latn", "por_Latn", "rus_Cyrl",
               "spa_Latn", "swe_Latn", "tha_Thai", "tur_Latn", "vie_Latn")
LANG_REGISTERS = tuple(f"probe-lang-{l}-jina" for l in _JINA_LANGS)
SOCIAL_REGISTERS = ("reddit-jina-250k", "ca-jina-250k",
                    "twitter-jina-250k", "bluesky-jina-250k")
EN_REGISTERS = ("probe-fineweb-jina", "probe-rpj-jina", "probe-pile-jina")
REGISTERS = LANG_REGISTERS + SOCIAL_REGISTERS + EN_REGISTERS
assert len(REGISTERS) == 27, len(REGISTERS)
# ALL 27 enter the maximin — there is no jina contaminated-code analog.
MAXIMIN_REGS = list(REGISTERS)

# ---- 0% baseline head (the undisplaced jina-multi-2m champion) ----
BASELINE_MODEL = SANDBOX / "jina-multi-2m" / "champion-bs16k" / "model.pt"
BASELINE_MAP_NAME = "jina-multi-2m/champion-bs16k (0% social, undisplaced base)"
BASELINE_N = 2_000_000
BASELINE_KEY = "baseline-0pct"

# ---- supported jina mixture arms. model.pt resolves as
#      sandbox/<arm>/champion-bs16k/model.pt. ----
ARM_REGISTRY = {
    "jina-bmix10-2m": {"social_pct": 10, "key": "jina-bmix10-10pct"},
    "jina-bmix20-2m": {"social_pct": 20, "key": "jina-bmix20-20pct"},
    "jina-bmix30-2m": {"social_pct": 30, "key": "jina-bmix30-30pct"},
    "jina-rmix20-2m": {"social_pct": 20, "key": "jina-rmix20-20pct"},
}

# ---- projector (owner resolution 2026-08-28) ----
JINA6M_SUB = SUBSTRATES / "jina-prompted" / "substrate-6250k.f16.npy"
JINA6M_TRUTH = SANDBOX / "jina-multi-6m" / "edges-k15-fuzzy.npz"
JINA6M_KNN = SANDBOX / "jina-multi-6m" / "knn_indices.npy"
NEUTRAL_SUB = SUBSTRATES / "jina-neutral-pooled" / "substrate.f16.npy"
NEUTRAL_TRUTH = SANDBOX / "jina-neutral-pooled" / "edges-k15-fuzzy.npz"


def resolve_arm(arm: str):
    """Return (MAPS, BMIX_KEY, social_pct). No GPU / model load."""
    if arm not in ARM_REGISTRY:
        raise SystemExit(
            f"unknown jina mixture arm {arm!r}; supported: {sorted(ARM_REGISTRY)}")
    info = ARM_REGISTRY[arm]
    bmix_key = info["key"]
    maps = {
        BASELINE_KEY: {"model": BASELINE_MODEL, "prefix": "baseline",
                       "social_pct": 0},
        bmix_key: {"model": SANDBOX / arm / "champion-bs16k" / "model.pt",
                   "prefix": arm, "social_pct": info["social_pct"]},
    }
    return maps, bmix_key, info["social_pct"]


class NormMemmap:
    """Lazy L2-row-normalizing view over a memmap (mirror confirm_mixture_2m).
    transform() slices X[i:j] per batch and casts to f32; normalizing inside
    __getitem__ keeps only one batch resident (the 6.25M f16 substrate is ~9.6 GB;
    pre-_norm would violate the >=2 GB rule)."""

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


def _resolve_registers():
    """Print + return {reg: {substrate, truth, exists}} for all 27 registers.
    substrate file resolves substrate.f16.npy (jina convention) then .f32.npy."""
    resolved = {}
    print("\n[registers] resolving 27 jina probe registers "
          "(20 lang + 4 social + 3 EN):")
    groups = (("lang", LANG_REGISTERS), ("social", SOCIAL_REGISTERS),
              ("en", EN_REGISTERS))
    for gname, regs in groups:
        for reg in regs:
            sub16 = SUBSTRATES / reg / "substrate.f16.npy"
            sub32 = SUBSTRATES / reg / "substrate.f32.npy"
            sub = sub16 if sub16.exists() else sub32
            edges = SANDBOX / reg / "edges-k15-fuzzy.npz"
            ok = sub.exists() and edges.exists()
            resolved[reg] = {"group": gname, "substrate": str(sub),
                             "truth": str(edges), "exists": bool(ok)}
            print(f"    {'OK  ' if ok else 'PEND'} [{gname:6}] {reg}: "
                  f"sub={'y' if sub.exists() else 'n'} truth={'y' if edges.exists() else 'n'}")
    return resolved


def _resolve_p15_jina():
    """P1.5 two-seed override (4th-review, delegate 2026-08-29): score an explicit seeded
    (baseline, treatment) jina pair — both trained through THIS runner so the same-seed
    comparison is EXACT (resident-D768 floor=0). arm := P15_PREFIX so the output filename is
    {P15_PREFIX}-jina-confirm-results.json. Returns (MAPS, bmix_key, social_pct, arm) or None."""
    base = os.environ.get("P15_BASELINE_MODEL")
    if not base:
        return None
    arm = os.environ["P15_PREFIX"]
    bmix_key = os.environ.get("P15_TREATMENT_KEY", "treatment")
    social_pct = int(os.environ.get("P15_SOCIAL_PCT", "10"))
    maps = {
        BASELINE_KEY: {"model": Path(base), "prefix": "baseline", "social_pct": 0},
        bmix_key: {"model": Path(os.environ["P15_TREATMENT_MODEL"]), "prefix": arm,
                   "social_pct": social_pct},
    }
    return maps, bmix_key, social_pct, arm


def main(argv: list[str]) -> int:
    p15 = _resolve_p15_jina()
    if p15 is not None:
        MAPS, BMIX_KEY, social_pct, arm = p15
    else:
        arm = (argv[1] if len(argv) > 1 else os.environ.get("MIXTURE_MAP", "")).strip()
        if not arm:
            raise SystemExit(
                "usage: confirm_jina_mixture.py <jina-mixture-arm>  (or env "
                f"MIXTURE_MAP); supported: {sorted(ARM_REGISTRY)}")
        MAPS, BMIX_KEY, social_pct = resolve_arm(arm)

    reg_status = _resolve_registers()

    # projector input resolution (print exists/missing) -----------------------
    proj6_ready = JINA6M_SUB.exists() and JINA6M_TRUTH.exists()
    knn6 = JINA6M_KNN if JINA6M_KNN.exists() else None
    neutral_ready = NEUTRAL_SUB.exists() and NEUTRAL_TRUTH.exists()
    print("\n[projector] jina_6m_transform (PRIMARY, required):")
    print(f"    {'OK  ' if proj6_ready else 'MISS'} sub={JINA6M_SUB} "
          f"(exists={JINA6M_SUB.exists()})")
    print(f"         truth={JINA6M_TRUTH} (exists={JINA6M_TRUTH.exists()}) "
          f"knn={'y' if knn6 else 'n'}")
    print("[projector] jina_neutral_pooled (SECONDARY, skip-clean-if-absent):")
    print(f"    {'OK  ' if neutral_ready else 'PEND'} sub={NEUTRAL_SUB} "
          f"truth={NEUTRAL_TRUTH} (ready={neutral_ready})")
    _base_model = MAPS[BASELINE_KEY]["model"]  # P15-aware (may differ from module BASELINE_MODEL)
    print(f"\n[maps] baseline: {_base_model} (exists={_base_model.exists()})")
    print(f"[maps] mixture {arm}: {MAPS[BMIX_KEY]['model']} "
          f"(exists={MAPS[BMIX_KEY]['model'].exists()})")

    # heavy / GPU imports deferred to runtime (import-safe) --------------------
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from image_map_pipeline import _norm  # noqa: F401
    from knobs_2m import quick_ffr_v2 as quick_ffr  # v2 truth-selection
    from knobs_2m import quick_ffr_v2_split  # P0.3c member/unseen split
    from basemap.pumap.parametric_umap.core import ParametricUMAP

    coords_dir = SANDBOX / f"{arm}-jina-confirm-coords"
    coords_dir.mkdir(parents=True, exist_ok=True)

    # ---- register substrate + truth cache (lazy, mixture_probe pattern) ----
    reg_cache: dict[str, tuple] = {}

    def _register(reg: str):
        if reg not in reg_cache:
            info = reg_status[reg]
            sub = Path(info["substrate"])
            edges = Path(info["truth"])
            if not sub.exists() or not edges.exists():
                reg_cache[reg] = (None, None, None)
            else:
                x = _norm(np.asarray(np.load(sub, mmap_mode="r"), dtype=np.float32))
                reg_cache[reg] = (x, edges, int(x.shape[0]))
        return reg_cache[reg]

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
        print(f"\n=== {key}: social={minfo['social_pct']}% ({model_pt}) ===",
              flush=True)
        model = ParametricUMAP.load(str(model_pt), device="cuda")

        # ---- DELIVERABLE 1: 27-register suite ----
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
            print(f"  {reg}: OOD-FFR {ffr:.4f} [{mode}] "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
        matrix[key] = row

        # ---- DELIVERABLE 2: projector column ----
        proj: dict[str, dict] = {}
        # (1) jina_6m_transform (PRIMARY): 6.25M substrate, lazy NormMemmap.
        if proj6_ready:
            t0 = time.time()
            X6 = NormMemmap(np.load(JINA6M_SUB, mmap_mode="r"))
            n6 = len(X6)
            xy = np.asarray(model.transform(X6, batch_size=8192), dtype=np.float32)
            np.save(coords_dir / f"{pfx}__proj-jina6m.npy", xy)
            # P0.3c: OPTIMISTIC projection case — a 2M (BASELINE_N) head projecting
            # the larger jina-6m substrate whose first BASELINE_N rows are the head's
            # byte-identical nested-prefix training set. Split FFR into training
            # MEMBER (query index < BASELINE_N) vs UNSEEN so the number is honest.
            # sp["overall"] == the old quick_ffr_v2 number (same queries).
            sp = quick_ffr_v2_split(xy, JINA6M_TRUTH, n6, member_cutoff=BASELINE_N,
                                    knn_indices_path=(knn6 if knn6 else None))
            ffr = float(sp["overall"])
            proj["jina_6m_transform"] = {
                "ffr_v2": ffr,
                "ffr_v2_member": sp["member"], "ffr_v2_unseen": sp["unseen"],
                "member_frac": sp["member_frac"],
                "n_member_queries": sp["n_member"], "n_unseen_queries": sp["n_unseen"],
                "member_note": (
                    f"head trained on the first {BASELINE_N:,} nested-prefix rows; "
                    f"query index < {BASELINE_N:,} = training member (in-sample, "
                    f"optimistic), >= = unseen. member_frac ~ BASELINE_N/N."),
                "rows": n6, "extrap": round(n6 / BASELINE_N, 3),
                "instrument": "FFR@0.1%-of-N (jina-6m sealed truth)",
                "truth": str(JINA6M_TRUTH),
                "knn_indices": (str(knn6) if knn6 else None),
                "truth_mode": sp["truth_mode"],
                "coords_label": f"{pfx}__proj-jina6m",
                "wall_s": round(time.time() - t0, 1)}
            print(f"  proj jina-6m (x{n6/BASELINE_N:.2f}): FFR {ffr:.4f} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
        else:
            proj["jina_6m_transform"] = {
                "status": "SKIPPED (jina-6m substrate/truth missing)"}

        # (2) jina_neutral_pooled (SECONDARY): skip-clean if truth absent.
        if neutral_ready:
            t0 = time.time()
            xn = _norm(np.asarray(np.load(NEUTRAL_SUB, mmap_mode="r"),
                                  dtype=np.float32))
            nn = int(xn.shape[0])
            xy = np.asarray(model.transform(xn, batch_size=8192), dtype=np.float32)
            np.save(coords_dir / f"{pfx}__proj-neutralpooled.npy", xy)
            ffr = float(quick_ffr(xy, NEUTRAL_TRUTH, nn))
            proj["jina_neutral_pooled"] = {
                "ffr_v2": ffr, "rows": nn,
                "truth_mode": getattr(quick_ffr, "last_truth_mode", "?"),
                "coords_label": f"{pfx}__proj-neutralpooled",
                "wall_s": round(time.time() - t0, 1)}
            print(f"  proj jina-neutral-pooled: FFR {ffr:.4f} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
        else:
            proj["jina_neutral_pooled"] = {
                "status": "SKIPPED (jina-neutral-pooled substrate/truth missing; "
                          "PENDING prereq, non-blocking)"}

        projectors[key] = proj
        del model

    # ---- per-head worst / mean over the 27 maximin registers ----
    per_map_worst: dict[str, float | None] = {}
    per_map_mean: dict[str, float | None] = {}
    per_map_worst_reg: dict[str, str | None] = {}
    for key, row in matrix.items():
        vals = [(r, row[r]) for r in MAXIMIN_REGS
                if row and _finite(row.get(r))] if row else []
        if vals:
            wr, wv = min(vals, key=lambda t: t[1])
            per_map_worst[key] = float(wv)
            per_map_worst_reg[key] = wr
            per_map_mean[key] = float(np.mean([v for _, v in vals]))
        else:
            per_map_worst[key] = None
            per_map_worst_reg[key] = None
            per_map_mean[key] = None

    out = SANDBOX / f"{arm}-jina-confirm-results.json"

    # ---- full-matrix assert: verdict ONLY if EVERY head x maximin-register cell
    # exists and is finite; else abort LOUDLY (registers PENDING -> ABORTED). ----
    missing = [(k, r) for k in MAPS for r in MAXIMIN_REGS
               if matrix.get(k) is None or not _finite((matrix.get(k) or {}).get(r))]
    if missing:
        out.write_text(json.dumps({
            "schema": "jina-mix-confirm-2026-08-28", "arm": arm, "ABORTED": True,
            "reason": "incomplete matrix: missing/non-finite head x maximin-register "
                      "cells (register substrates/truths are GPU prereqs P-A/P-B/P-C, "
                      "likely still PENDING); no confirmation verdict",
            "missing_cells": [f"{k}--{r}" for k, r in missing][:80],
            "register_status": reg_status,
            "ffr_matrix": matrix,
            "maximin_registers": MAXIMIN_REGS,
            "projectors": projectors}, indent=1, default=_np))
        print(f"\nABORT: {len(missing)} missing/non-finite maximin cells -> "
              "NO confirmation verdict (registers likely PENDING)", flush=True)
        raise SystemExit(2)

    # ---- per-register mixture - baseline delta ----
    base_row = matrix[BASELINE_KEY]
    bmix_row = matrix[BMIX_KEY]
    per_register_delta = {}
    for r in MAXIMIN_REGS:
        b, m = base_row.get(r), bmix_row.get(r)
        per_register_delta[r] = (round(float(m) - float(b), 4)
                                 if _finite(b) and _finite(m) else None)

    worst_delta = (round(per_map_worst[BMIX_KEY] - per_map_worst[BASELINE_KEY], 4)
                   if _finite(per_map_worst[BMIX_KEY])
                   and _finite(per_map_worst[BASELINE_KEY]) else None)
    mean_delta = (round(per_map_mean[BMIX_KEY] - per_map_mean[BASELINE_KEY], 4)
                  if _finite(per_map_mean[BMIX_KEY])
                  and _finite(per_map_mean[BASELINE_KEY]) else None)
    maximin_winner = max((k for k in MAPS), key=lambda k: per_map_worst[k])
    confirmed = bool(worst_delta is not None and worst_delta >= 0.0
                     and mean_delta is not None and mean_delta >= 0.0)

    def _proj_delta(name):
        b = (projectors.get(BASELINE_KEY, {}).get(name, {}) or {}).get("ffr_v2")
        m = (projectors.get(BMIX_KEY, {}).get(name, {}) or {}).get("ffr_v2")
        return (round(float(m) - float(b), 4)
                if _finite(b) and _finite(m) else None)

    out.write_text(json.dumps({
        "schema": "jina-mix-confirm-2026-08-28",
        "arm": arm,
        "space": "jina-v5-nano document-prompted D768",
        "baseline_map": BASELINE_MAP_NAME,
        "baseline_N": BASELINE_N,
        "mixture_key": BMIX_KEY,
        "social_pct": social_pct,
        "registers": list(REGISTERS),
        "maximin_registers": MAXIMIN_REGS,
        "register_groups": {"languages": list(LANG_REGISTERS),
                            "social": list(SOCIAL_REGISTERS),
                            "en_base": list(EN_REGISTERS)},
        "no_exclusions_note": ("all 27 registers enter the maximin — languages are "
                               "the core of the jina maximin; there is no "
                               "contaminated-code register on the jina side."),
        "maps": {k: {"model": str(v["model"]), "social_pct": v["social_pct"]}
                 for k, v in MAPS.items()},
        "ffr_matrix": matrix,
        "per_map_worst_register_ffr": per_map_worst,
        "per_map_worst_register": per_map_worst_reg,
        "per_map_mean_ffr": per_map_mean,
        "maximin_winner": maximin_winner,
        "per_register_delta_mix_minus_baseline": per_register_delta,
        "worst_register_delta_mix_minus_baseline": worst_delta,
        "mean_delta_mix_minus_baseline": mean_delta,
        "mix_confirms_screening_lift": confirmed,
        "projectors": projectors,
        "projector_delta_mix_minus_baseline": {
            "jina_6m_transform": _proj_delta("jina_6m_transform"),
            "jina_neutral_pooled": _proj_delta("jina_neutral_pooled")},
        "note": (
            "OOD-FFR@0.1% (quick_ffr_v2): each of the 27 jina probe registers "
            "projected through each frozen 2M head, scored vs the register's own "
            "exact-k15 truth. Both heads are the champion-bs16k recipe (dose4, "
            f"rankneg = 25% of N); {arm} displaces only the EN 1M by {social_pct}% "
            "social while holding all 20 language blocks bit-identical to the 0% "
            "baseline, so the social share is the sole delta. ALL 27 registers "
            "(20 languages + 4 social + 3 EN base) enter worst/mean/maximin. "
            "mix_confirms_screening_lift = worst-register delta >=0 AND mean delta "
            ">=0. PROJECTOR column: jina_6m_transform (PRIMARY, 6.25M substrate at "
            f"disc=0.1%xN, {round(6_250_000/BASELINE_N,2)}x extrapolation, lazy "
            "NormMemmap) + jina_neutral_pooled (SECONDARY, skip-clean if its truth "
            "is still PENDING). The MiniLM a1-common-neutral is NOT used (wrong space)."),
    }, indent=1, default=_np))
    print(f"\nresults: {out}", flush=True)
    print(f"  arm={arm} baseline={BASELINE_MAP_NAME}")
    print(f"  per-head worst-register FFR: {per_map_worst}")
    print(f"  per-head worst register:     {per_map_worst_reg}")
    print(f"  per-head mean FFR:           {per_map_mean}")
    print(f"  worst-register delta (mix-baseline): {worst_delta}")
    print(f"  mean delta (mix-baseline):           {mean_delta}")
    print(f"  mix confirms screening lift: {confirmed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
