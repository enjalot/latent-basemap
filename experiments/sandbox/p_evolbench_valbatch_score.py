"""Evolution VALIDATION-BATCH scorer (overseer 2026-09-02). .venv (scipy). Produces the seed deltas the
λ-frontier's margin rule needs, on the SAME churn scale as p_evolbench_lambda_score:

  (1) PLACEMENT-tier floor: S0 head seed-43 vs seed-42 (identical 4M data, reseed only) -> procrustes,
      displacement / 90th-pct S0-cloud radius = optimizer-only churn floor for the frozen/placement side.
  (2) RETRAIN-corner seed delta: S3 head seed-43 (armA-triggered twin) scored through the frontier's own
      _score_map (S2 rows [0:N2] vs the S2 layout) -> directly comparable to the w=0(full-retrain) row.
      Δchurn, Δreddit_ffr between seed 42 and 43 = the retrain-corner seed noise.

Margin rule (#2): a λ cell is REAL only if its |gain| and churn exceed ~3x the corresponding seed floor;
cells inside the band are seed-noise. We annotate every frontier cell with a verdict.
Output: evolbench-valbatch.json  (PROVISIONAL until the 2nd-timeline draw folds in)."""
import json
from pathlib import Path
import numpy as np

SB = Path("/data/latent-basemap/sandbox")
N0 = 4_000_000; N2 = 5_600_000; N = 6_400_000
S2_LAYOUT = SB / "evolbench-armA-frozen" / "coords-S2.npy"
S3_KNN = SB / "evolbench-S3" / "knn_indices.npy"
S0_S42 = SB / "evolbench-S0" / "champion-bs16k" / "coordinates.npy"
S0_S43 = SB / "evolbench-S0" / "champion-bs16k-s43" / "coordinates.npy"
S3_S43 = SB / "evolbench-S3" / "champion-bs16k-s43" / "coordinates.npy"
MARGIN_K = 3.0   # a cell must beat 3x the seed floor to be "real"

# reuse the frontier scorer's exact churn + cohort definitions (same scale = valid margin comparison)
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from p_evolbench_lambda_score import _cohort_ffr, _score_map  # noqa: E402
import frame  # item B: RIGID gauge (no scale collapse), shared with score_v2/lambda_score/export_evolution_viz


def _floor_churn(a_path, b_path):
    """Optimizer-only churn: two independent trainings of the SAME map, RIGID-aligned (item B; was WITH-SCALE)."""
    a = np.asarray(np.load(a_path), dtype=np.float64)
    b = np.asarray(np.load(b_path), dtype=np.float64)
    n = min(a.shape[0], b.shape[0]); a = a[:n]; b = b[:n]
    disp, _ = frame.churn(a, b)   # rigid-aligned + canonical frame radius
    return {"n": int(n), "churn_mean": round(float(disp.mean()), 5),
            "churn_p95": round(float(np.percentile(disp, 95)), 5)}


def main():
    out = {"schema": "evolbench-valbatch-2026-09-02",
           "_PROVISIONAL": "PROVISIONAL-PENDING-2ND-TIMELINE (seed deltas complete; 2nd-draw robustness pending)"}

    # (1) placement-tier floor
    if S0_S43.is_file():
        out["floor_placement_S0_seed"] = _floor_churn(S0_S43, S0_S42)
    else:
        out["floor_placement_S0_seed"] = {"error": "S0 seed-43 coords absent"}

    # (2) retrain-corner seed delta: score S3-seed43 through the frontier scorer + diff vs seed-42 (w=0 row)
    s2 = np.asarray(np.load(S2_LAYOUT), dtype=np.float32)
    knn = np.load(S3_KNN, mmap_mode="r")
    s42 = _score_map(SB / "evolbench-armA-triggered" / "coords-S3.npy", s2, knn, 313.0, "w=0 seed42")
    if S3_S43.is_file():
        s43 = _score_map(S3_S43, s2, knn, 313.0, "w=0 seed43")
        d_churn = round(abs(s43["churn_mean"] - s42["churn_mean"]), 5)
        d_reddit = round(abs((s43["reddit_ffr"] or 0) - (s42["reddit_ffr"] or 0)), 4)
        d_overall = round(abs((s43["overall_ffr"] or 0) - (s42["overall_ffr"] or 0)), 4)
        out["retrain_corner_seed"] = {"seed42": s42, "seed43": s43,
                                      "delta_churn": d_churn, "delta_reddit_ffr": d_reddit,
                                      "delta_overall_ffr": d_overall}
    else:
        out["retrain_corner_seed"] = {"seed42": s42, "error": "S3 seed-43 coords absent"}
        d_churn = None; d_reddit = None

    # margin-rule verdicts on the existing frontier cells (gain floor = retrain-corner reddit seed noise;
    # churn floor = placement-tier seed churn — the smallest meaningful map motion)
    fr = SB / "evolbench-lambda-frontier.json"
    if fr.is_file() and d_reddit is not None:
        frontier = json.loads(fr.read_text())["frontier"]
        churn_floor = out["floor_placement_S0_seed"].get("churn_mean")
        gain_floor = d_reddit
        verdicts = []
        for r in frontier:
            g = abs(r.get("ood_gain") or 0); c = r.get("churn_mean") or 0
            gain_real = (g >= MARGIN_K * gain_floor) if gain_floor else None
            churn_real = (c >= MARGIN_K * churn_floor) if churn_floor else None
            verdicts.append({"cell": r["label"], "ood_gain": r.get("ood_gain"), "churn_mean": c,
                             "gain_over_floor": (round(g / gain_floor, 2) if gain_floor else None),
                             "churn_over_floor": (round(c / churn_floor, 2) if churn_floor else None),
                             "gain_real": gain_real, "churn_real": churn_real})
        out["margin_rule"] = {"gain_floor_reddit_ffr": gain_floor, "churn_floor_placement": churn_floor,
                              "margin_k": MARGIN_K, "cells": verdicts}

    (SB / "evolbench-valbatch.json").write_text(json.dumps(out, indent=1, default=str))
    print("=== VALIDATION BATCH — seed deltas ===", flush=True)
    print("placement floor (S0 s42 vs s43):", out.get("floor_placement_S0_seed"), flush=True)
    rc = out.get("retrain_corner_seed", {})
    print(f"retrain corner Δchurn={rc.get('delta_churn')} Δreddit={rc.get('delta_reddit_ffr')} "
          f"Δoverall={rc.get('delta_overall_ffr')}", flush=True)
    if "margin_rule" in out:
        print(f"margin rule (k={MARGIN_K}, gain_floor={out['margin_rule']['gain_floor_reddit_ffr']}, "
              f"churn_floor={out['margin_rule']['churn_floor_placement']}):", flush=True)
        for v in out["margin_rule"]["cells"]:
            print(f"  {v['cell']:>16} gain={v['ood_gain']} ({v['gain_over_floor']}x) real={v['gain_real']} | "
                  f"churn={v['churn_mean']} ({v['churn_over_floor']}x) real={v['churn_real']}", flush=True)
    print(f"wrote {SB / 'evolbench-valbatch.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
