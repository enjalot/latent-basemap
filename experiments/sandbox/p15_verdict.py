#!/usr/bin/env python3
"""P1.5 two-seed finalist verdict (4th-review, delegate 2026-08-29).

Resident floor is ZERO, so per-seed treatment-vs-baseline comparisons are EXACT. Two seeds
therefore test effect ROBUSTNESS, not measurement noise: a different seed is a different init
trajectory, and a real effect must hold its sign across both.

Verdict rule (delegate):
  * both seeds SAME sign on the maximin (worst-register) delta -> SEALED win/loss.
  * SPLIT sign -> the effect is seed-scale; finalist is NO-ADOPT (treat as tie, prefer baseline
    for simplicity).
Plus the PROJECTION non-regression GATE (required): the 6.25M-projector maximin delta must not
regress (< -tol) in EITHER seed — a finalist that wins the own-map maximin but degrades the
atlas projector is not adopted.

Reads the two per-seed {prefix}-s{seed}-confirm-results.json and writes {prefix}-verdict.json.
Usage: p15_verdict.py <prefix-stem> [tol]   e.g. p15_verdict.py p15-minilm 0.0
"""
import json
import sys
from pathlib import Path

SB = Path("/data/latent-basemap/sandbox")


def _load(prefix_stem, seed):
    p = SB / f"{prefix_stem}-s{seed}-confirm-results.json"
    if not p.exists():
        return None, str(p)
    return json.loads(p.read_text()), str(p)


def _sign(x, tol=0.0):
    if x is None:
        return None
    if x > tol:
        return +1
    if x < -tol:
        return -1
    return 0


def main(argv):
    stem = argv[1] if len(argv) > 1 else "p15-minilm"
    tol = float(argv[2]) if len(argv) > 2 else 0.0
    seeds = [42, 43]
    per_seed = {}
    for s in seeds:
        d, path = _load(stem, s)
        if d is None:
            per_seed[s] = {"status": f"MISSING ({path})"}
            continue
        if d.get("ABORTED"):
            per_seed[s] = {"status": "ABORTED (incomplete matrix)", "path": path}
            continue
        proj = d.get("projector_delta_mix_minus_baseline", {})
        per_seed[s] = {
            "path": path,
            "worst_delta": d.get("worst_register_delta_mix_minus_baseline"),
            "mean_delta": d.get("mean_delta_mix_minus_baseline"),
            "per_register_delta": d.get("per_register_delta_mix_minus_baseline"),
            "proj_6250k_delta": proj.get("proj_6250k"),
            "proj_a1neutral_delta": proj.get("a1_neutral"),
            "maximin_winner": d.get("maximin_winner"),
        }

    complete = all("worst_delta" in per_seed[s] for s in seeds)
    out = {"schema": "p15-two-seed-verdict-2026-08-29", "prefix": stem, "tol": tol,
           "per_seed": per_seed, "complete": complete}

    if complete:
        signs = {s: _sign(per_seed[s]["worst_delta"], tol) for s in seeds}
        proj_signs = {s: _sign(per_seed[s]["proj_6250k_delta"], tol) for s in seeds}
        # projection non-regression gate: fail if any seed's projector maximin regresses (< -tol)
        proj_regressed = any(per_seed[s]["proj_6250k_delta"] is not None
                             and per_seed[s]["proj_6250k_delta"] < -tol for s in seeds)
        same_sign = signs[42] is not None and signs[42] == signs[43]
        both_positive = same_sign and signs[42] == +1
        both_nonneg = all(per_seed[s]["worst_delta"] is not None
                          and per_seed[s]["worst_delta"] >= -tol for s in seeds)

        if not same_sign:
            verdict = "NO-ADOPT (split sign — seed-scale effect; prefer baseline)"
        elif signs[42] == +1:
            verdict = ("SEALED WIN (both seeds positive maximin)"
                       if not proj_regressed else
                       "NO-ADOPT (own-map win but PROJECTION REGRESSES in >=1 seed)")
        elif signs[42] == -1:
            verdict = "SEALED LOSS (both seeds negative maximin — baseline wins)"
        else:  # both exactly 0 within tol
            verdict = "TIE (both seeds within tol) — prefer baseline"

        out.update({
            "maximin_delta_signs": signs, "same_sign": same_sign,
            "both_seeds_positive": both_positive, "both_seeds_nonneg_within_tol": both_nonneg,
            "projection_delta_signs": proj_signs, "projection_regressed": proj_regressed,
            "verdict": verdict})
        print(f"[{stem}] seed42 worst_delta={per_seed[42]['worst_delta']} "
              f"seed43 worst_delta={per_seed[43]['worst_delta']} | "
              f"proj42={per_seed[42]['proj_6250k_delta']} proj43={per_seed[43]['proj_6250k_delta']}")
        print(f"  -> {verdict}")
    else:
        out["verdict"] = "INCOMPLETE (missing per-seed results)"
        print(f"[{stem}] INCOMPLETE: {[per_seed[s].get('status') for s in seeds]}")

    (SB / f"{stem}-verdict.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {SB / f'{stem}-verdict.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
