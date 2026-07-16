"""G1 — content-bound CPU decision node for the 8M standard-curve vs legacy pair.

Applies the owner's pre-registered closeness bands to the shared-reference paired
scores. Legacy reference (seed-matched): FFR mean 0.5773 (sd 0.0007), purity-k1024
mean 0.78285 (sd 0.01534). Bands: max(0.005, 2·sd)=0.005 (FFR),
max(0.03, 2·sd)=0.03068 (purity).

Decision:
  - stdcurve FFR < 0.5723 AND purity_k1024 < 0.75217  -> finalize legacy_lp (no seed 43);
  - either primary axis inside its band or reversed     -> STOP, add seed 43 for owner;
  - density is secondary and cannot override both primary axes.
"""
from __future__ import annotations
import argparse, os, sys, json

LEGACY_FFR, FFR_BAND = 0.5773, 0.005
LEGACY_PUR, PUR_BAND = 0.78285, 0.03068
FFR_LO, PUR_LO = round(LEGACY_FFR - FFR_BAND, 4), round(LEGACY_PUR - PUR_BAND, 5)


def _get(scores, label):
    s = scores.get("seeds", scores.get("runs", {})).get(label)
    if not s:
        raise SystemExit(f"decision: no score entry for '{label}' in {list(scores.get('seeds', {}))}")
    pur = s.get("purity") or {}
    return {"ffr": s["ffr"], "purity_k1024": pur.get("k1024") if isinstance(pur, dict) else s.get("purity_k1024"),
            "density": s["density"],
            "hiD_reference_reused": s.get("hiD_reference_reused")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stdcurve-label", default="stdcurve")
    ap.add_argument("--legacy-label", default="legacy")
    args = ap.parse_args()
    scores = json.load(open(args.scores))
    sc = _get(scores, args.stdcurve_label)
    lg = _get(scores, args.legacy_label)
    # both maps must have reused the ONE shared reference (L0.4)
    if scores.get("hiD_reference_key") is None or not sc.get("hiD_reference_reused") \
            or not lg.get("hiD_reference_reused"):
        raise SystemExit("decision: paired scores did not both reuse the shared reference (L0.4).")

    ffr, pur, dens = sc["ffr"], sc["purity_k1024"], sc["density"]
    ffr_below = ffr < FFR_LO
    pur_below = pur < PUR_LO
    ffr_in_band = abs(ffr - LEGACY_FFR) <= FFR_BAND
    pur_in_band = abs(pur - LEGACY_PUR) <= PUR_BAND
    ffr_reversed = ffr > LEGACY_FFR + FFR_BAND
    pur_reversed = pur > LEGACY_PUR + PUR_BAND

    if ffr_below and pur_below:
        decision, needs_seed43 = "finalize_legacy_lp", False
    elif ffr_in_band or pur_in_band or ffr_reversed or pur_reversed:
        decision, needs_seed43 = "stop_add_seed43_for_owner", True
    else:
        # both primary axes clearly below but not both under the hard band -> still
        # ambiguous; be conservative and escalate.
        decision, needs_seed43 = "stop_add_seed43_for_owner", True

    out = {"gate": "stdcurve_vs_legacy_decision", "scores_path": os.path.abspath(args.scores),
           "hiD_reference_key": scores.get("hiD_reference_key"),
           "legacy_reference": {"ffr_mean": LEGACY_FFR, "purity_k1024_mean": LEGACY_PUR,
                                "ffr_band": FFR_BAND, "purity_band": PUR_BAND,
                                "ffr_floor": FFR_LO, "purity_floor": PUR_LO},
           "stdcurve": sc, "legacy_paired": lg,
           "primary_axes": {"ffr_below_floor": ffr_below, "purity_below_floor": pur_below,
                            "ffr_in_band": ffr_in_band, "purity_in_band": pur_in_band,
                            "ffr_reversed": ffr_reversed, "purity_reversed": pur_reversed},
           "density_secondary": dens,
           "decision": decision, "needs_seed43": needs_seed43}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"[decision] {decision} (needs_seed43={needs_seed43}) "
          f"stdcurve ffr={ffr} purity={pur} vs floors {FFR_LO}/{PUR_LO} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
