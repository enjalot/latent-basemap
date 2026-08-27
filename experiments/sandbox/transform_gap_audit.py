#!/usr/bin/env python3
"""Transform-vs-train gap audit (owner 2026-08-27, A1 follow-on) — INFERENCE ONLY.

For each (smaller-N head, larger-N rung) pair: transform the FULL rung-N substrate
through the SMALLER-N-trained head (pure batched forward pass), score against the
rung's OWN sealed truth with the same instrument the trained-at-N map used
(quick_ffr_v2 + collapse), and diff vs the trained-at-N map's score on that truth.
Deliverable: the gap-vs-extrapolation-factor curve + a health read on HOW transform
degrades (collapse fraction: fog/over-collapse vs the trained twin).

Recipe-clean HEADLINE pairs (carry the conclusion): MiniLM 2M→6.25M, jina 2M→6.25M.
Older MiniLM rungs (12.5M/25M/50M/100M) are recipe-MIXED across generations (same
caveat as A1) and are reported with that flag; a pair whose substrate OR truth is
missing is SKIPPED cleanly (recorded as skipped, never faked).

Transforms are batched forward passes — trivially interruptible and low-VRAM; the
orchestrator only invokes this when the GPU is free (yields to any queued train).
Transformed coordinates are exported as new rung "arms" labelled
`xform-from-<head>` so the provenance is obvious on the compare page.

GPU script. Output: /data/latent-basemap/sandbox/transform-gap-audit.json
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

SB = Path("/data/latent-basemap/sandbox")
CKPT = Path("/data/checkpoints/pumap")


def _resolve(p):
    """Resolve a path or a glob to an existing file, else None."""
    if p is None:
        return None
    p = str(p)
    if "*" in p:
        hits = sorted(glob.glob(p))
        return Path(hits[0]) if hits else None
    return Path(p) if Path(p).exists() else None


# Each pair: smaller-N HEAD (model.pt) transformed onto a larger-N RUNG substrate,
# scored vs that rung's sealed truth; reference = the trained-at-N map's coords.
def _pairs():
    from knobs_2m import RUNGS
    P = []
    # --- MiniLM: the 2M champion head projected up the ladder ---
    head2m = SB / "2m-knobs/umap-md000-x4bs16k-winner/model.pt"
    mini = [
        ("6.25M", "6250k", SB / "6250k-knobs/umap-md000-x4bs16k-winner-rank25", 3.125, True),
        ("12.5M", "12500k", SB / "12500k-knobs/umap-md000-x4-fneg10", 6.25, False),
        ("25M",  "25000k", SB / "25000k-knobs/umap-md000-x2-fneg10-hostint8", 12.5, False),
    ]
    for rung_label, rung_key, ref_dir, extrap, clean in mini:
        r = RUNGS.get(rung_key, {})
        sub = _resolve(r.get("substrate"))
        truth = _resolve(r.get("edges") or r.get("edges_glob"))
        P.append({
            "space": "MiniLM", "head": "2m-champion", "rung": rung_label,
            "extrap": extrap, "recipe_clean": clean,
            "head_pt": head2m, "substrate": sub, "truth": truth,
            "ref_coords": ref_dir / "coordinates.npy",
            "ref_summary": ref_dir / "summary.json",
        })
    # --- jina: the 2M champion head onto the 6.25M rung (recipe-clean headline) ---
    P.append({
        "space": "jina", "head": "jina-2m-champion", "rung": "6.25M",
        "extrap": 3.125, "recipe_clean": True,
        "head_pt": SB / "jina-multi-2m/champion-bs16k/model.pt",
        "substrate": None,   # loaded via DATASETS['jina-multi-6m'] below
        "substrate_ds": "jina-multi-6m",
        "truth": _resolve(SB / "jina-multi-6m/edges-k15-fuzzy.npz"),
        "ref_coords": SB / "jina-multi-6m/champion-bs16k/coordinates.npy",
        "ref_summary": SB / "jina-multi-6m/champion-bs16k/summary.json",
    })
    return P


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from image_map_pipeline import DATASETS, _norm
    from knobs_2m import quick_ffr_v2
    try:
        from analysis_v2 import collapse as collapse_frac
    except Exception:
        collapse_frac = None
    from basemap.pumap.parametric_umap.core import ParametricUMAP

    coords_dir = SB / "transform-gap-coords"
    coords_dir.mkdir(parents=True, exist_ok=True)

    def _score(xy, truth, n):
        out = {"ffr_v2": float(quick_ffr_v2(xy, truth, n))}
        if collapse_frac is not None:
            try:
                out["collapse"] = float(collapse_frac(xy))
            except Exception:
                out["collapse"] = None
        return out

    rows = []
    for p in _pairs():
        rec = {k: (str(v) if isinstance(v, Path) else v)
               for k, v in p.items() if k not in ("substrate_ds",)}
        # inventory gate — skip cleanly on any missing artifact
        head_pt = p["head_pt"]
        truth = p["truth"]
        missing = [k for k, v in (("head_pt", head_pt), ("truth", truth),
                                  ("ref_coords", p["ref_coords"])) if not (v and Path(v).exists())]
        # substrate: either a path or a DATASETS key
        sub = p.get("substrate")
        sub_ds = p.get("substrate_ds")
        if sub_ds is None and not (sub and Path(sub).exists()):
            missing.append("substrate")
        if missing:
            rec["status"] = f"SKIPPED (missing: {','.join(missing)})"
            rows.append(rec)
            print(f"SKIP {p['space']} {p['head']}→{p['rung']}: {rec['status']}", flush=True)
            continue

        t0 = time.time()
        X = (_norm(DATASETS[sub_ds]["load"]()) if sub_ds
             else _norm(np.asarray(np.load(sub, mmap_mode="r"), dtype=np.float32)))
        n = int(X.shape[0])
        model = ParametricUMAP.load(str(head_pt), device="cuda")
        xy = np.asarray(model.transform(X, batch_size=8192), dtype=np.float32)
        label = f"xform-from-{p['head']}"
        np.save(coords_dir / f"{p['space']}__{p['rung']}__{label}.npy", xy)
        transform_score = _score(xy, truth, n)

        ref_xy = np.asarray(np.load(p["ref_coords"]), dtype=np.float32)
        trained_score = _score(ref_xy, truth, n)
        # prefer the sealed FFR from the ref summary if present (same instrument)
        try:
            rs = json.loads(Path(p["ref_summary"]).read_text())
            trained_score["sealed_ffr_v1"] = rs.get("quick_ffr_at_0.1pct")
            trained_score["sealed_ffr_v2"] = rs.get("quick_ffr_v2")
        except Exception:
            pass

        gap = round(trained_score["ffr_v2"] - transform_score["ffr_v2"], 4)
        rec.update({
            "status": "OK", "rows": n, "coords_label": label,
            "transform": transform_score, "trained": trained_score,
            "gap_ffr_v2": gap, "gap_per_extrap": round(gap / p["extrap"], 5),
            "wall_s": round(time.time() - t0, 1),
        })
        rows.append(rec)
        print(f"OK {p['space']} {p['head']}→{p['rung']} x{p['extrap']}: "
              f"transform {transform_score['ffr_v2']:.4f} vs trained "
              f"{trained_score['ffr_v2']:.4f} → gap {gap:+.4f} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
        del model

    out = SB / "transform-gap-audit.json"
    out.write_text(json.dumps({
        "schema": "transform-vs-train-gap-2026-08-27",
        "note": "smaller-N head transformed onto larger-N rung, scored vs the rung's "
                "sealed truth (quick_ffr_v2 + collapse); gap = trained-at-N − transform. "
                "recipe_clean pairs (2M→6.25M both spaces) carry the headline; mixed-recipe "
                "rungs flagged. Transformed coords exported as xform-from-<head> arms.",
        "pairs": rows,
    }, indent=1))
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
