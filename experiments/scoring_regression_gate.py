"""S2.3 — 2M scoring wall/peak regression gate.

The evaluator itself can regress (a lost byte-cap, a bad tile size, a stray full
materialization) and silently make 8M/30M scoring 5-25x slower or OOM. This gate
reads the persisted 2M scoring evidence and fails BEFORE any large-scale scoring
if wall time or peak memory drifts past a pinned envelope. It is CPU-only (reads
JSON), so it slots into the scoring DAG as a cheap mandatory node.

Baselines are the real 2026-07-15 v2.2 measurements (500k-tile default):
  golden 2M streamed ~5.5 s, peak_gpu ~5.6 GB; knn 2M e2e ~5.1 s, peak ~17.9 GB.

Usage:
  python experiments/scoring_regression_gate.py \
      --golden experiments/evidence/r1_rescore/golden_2m_extended_v22.json \
      --knn    experiments/evidence/r1_rescore/knn_cost.json \
      --out    experiments/evidence/r1_rescore/scoring_regression.json
"""
from __future__ import annotations
import argparse, os, sys, json

# name -> (baseline, hard_max). hard_max is the abort ceiling (generous headroom
# over the baseline to tolerate co-tenant jitter, but far below a real regression).
BASELINES = {
    "golden_2m_streamed_wall_s": (5.5, 15.0),
    "golden_2m_reference_wall_s": (12.7, 35.0),
    "golden_2m_peak_gpu_gb": (5.629, 9.0),
    "knn_2m_e2e_wall_s": (5.14, 15.0),
    "knn_2m_peak_gpu_gb": (17.946, 24.0),
}


def _load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="experiments/evidence/r1_rescore/golden_2m_extended_v22.json")
    ap.add_argument("--knn", default="experiments/evidence/r1_rescore/knn_cost.json")
    ap.add_argument("--out", default="experiments/evidence/r1_rescore/scoring_regression.json")
    args = ap.parse_args()

    g = _load(args.golden) or {}
    k = _load(args.knn) or {}
    knn2m = (k.get("testbeds") or {}).get("2m", {})
    observed = {
        "golden_2m_streamed_wall_s": g.get("wall_streamed_s"),
        "golden_2m_reference_wall_s": g.get("wall_reference_s"),
        "golden_2m_peak_gpu_gb": g.get("peak_gpu_gb"),
        "knn_2m_e2e_wall_s": knn2m.get("end_to_end_regression_wall_s"),
        "knn_2m_peak_gpu_gb": knn2m.get("peak_gpu_gb"),
    }
    checks = {}
    ok = True
    for name, (base, hard_max) in BASELINES.items():
        val = observed.get(name)
        passed = (val is not None) and (float(val) <= hard_max)
        checks[name] = {"observed": val, "baseline": base, "hard_max": hard_max,
                        "passed": bool(passed),
                        "ratio_vs_baseline": round(float(val) / base, 2) if val else None}
        ok = ok and passed
        flag = "PASS" if passed else "FAIL"
        print(f"  [{flag}] {name}: {val} (baseline {base}, max {hard_max})")
    out = {"gate": "scoring_2m_regression", "passed": bool(ok), "checks": checks,
           "rule": "2M evaluator wall/peak must stay within the pinned envelope "
                   "before any 8M/30M scoring; catches lost byte-caps / tile blowups."}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nscoring 2M regression passed = {ok}  ->  {args.out}")
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
