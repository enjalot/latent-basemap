"""Draw-variance (universality-of-timeline) report (overseer 2026-09-02). .venv. When the 2nd-timeline draw
lands, report it as DEVIATION-FROM-DRAW1 per metric alongside raw values, with the seed-variance band from
the validation batch as the comparison bar. |Δ_draw| <= seed band  =>  the headline is timeline-universal
(a different tranche split moves the metric no more than reseeding the optimizer does). That juxtaposition
IS the universality result. Reads evolbench-tradesurface-v2.json (draw1), evolbench-draw2-tradesurface-v2.json
(draw2), evolbench-valbatch.json (seed band). Output: evolbench-draw2-deviation.json."""
import json
from pathlib import Path

SB = Path("/data/latent-basemap/sandbox")
REDDIT_K = 3
# (draw1 label, draw2 label)
PAIRS = [("frozen", "armA-frozen", "armA-draw2-frozen"),
         ("triggered", "armA-triggered", "armA-draw2-triggered")]


def _final(arm):
    tr = arm["trajectory"]; last = tr[-1]
    reddit_s3 = next((r["cohorts"].get("reddit") for r in tr if r["k"] == REDDIT_K), None)
    return {"final_ffr": last["quality"]["ffr"], "cum_churn": last["cum_churn"],
            "reddit_s5": last["cohorts"].get("reddit"), "reddit_s3": reddit_s3}


def _band(valbatch):
    """Seed-variance band: retrain-corner (S3-head reseed) deltas for FFR/reddit; placement floor for churn.
    These bound how much a metric moves from optimizer reseeding alone — the bar the draw deviation must beat."""
    rc = valbatch.get("retrain_corner_seed", {})
    fp = valbatch.get("floor_placement_S0_seed", {})
    return {"ffr": rc.get("delta_overall_ffr"), "reddit": rc.get("delta_reddit_ffr"),
            "churn_triggered": rc.get("delta_churn"), "churn_frozen": fp.get("churn_mean")}


def _verdict(delta, band):
    if band is None or delta is None:
        return {"delta": delta, "band": band, "within": None, "ratio": None}
    r = abs(delta) / band if band else None
    return {"delta": round(delta, 4), "band": round(band, 4),
            "within": (abs(delta) <= band), "ratio": (round(r, 2) if r is not None else None)}


def main():
    d1p = SB / "evolbench-tradesurface-v2.json"
    d2p = SB / "evolbench-draw2-tradesurface-v2.json"
    vbp = SB / "evolbench-valbatch.json"
    if not d2p.is_file():
        raise SystemExit(f"draw2 surface not present yet: {d2p} (run after DRAW2_RESULT_SENTINEL)")
    d1 = json.loads(d1p.read_text())["arms"]
    d2 = json.loads(d2p.read_text())["arms"]
    band = _band(json.loads(vbp.read_text())) if vbp.is_file() else {}

    out = {"schema": "evolbench-draw2-deviation-2026-09-02",
           "_PROVISIONAL": "seed band from valbatch; both are single-draw/single-seed point estimates",
           "seed_band": band, "arms": {}}
    print("=== DRAW-VARIANCE (universality-of-timeline): draw2 Δ vs draw1, bar = seed band ===", flush=True)
    for kind, l1, l2 in PAIRS:
        if l1 not in d1 or l2 not in d2:
            print(f"  {kind}: MISSING ({l1} in draw1: {l1 in d1}; {l2} in draw2: {l2 in d2})", flush=True)
            continue
        a1 = _final(d1[l1]); a2 = _final(d2[l2])
        churn_band = band.get("churn_triggered") if kind == "triggered" else band.get("churn_frozen")
        row = {"draw1": a1, "draw2": a2,
               "d_final_ffr": _verdict((a2["final_ffr"] or 0) - (a1["final_ffr"] or 0), band.get("ffr")),
               "d_cum_churn": _verdict((a2["cum_churn"] or 0) - (a1["cum_churn"] or 0), churn_band),
               "d_reddit_s5": _verdict((a2["reddit_s5"] or 0) - (a1["reddit_s5"] or 0), band.get("reddit")),
               "d_reddit_s3": _verdict((a2["reddit_s3"] or 0) - (a1["reddit_s3"] or 0), band.get("reddit"))}
        out["arms"][kind] = row
        print(f"\n  [{kind}]  draw1 -> draw2   (Δ | band | within?)", flush=True)
        for m, lab in (("d_final_ffr", "final FFR"), ("d_cum_churn", "cum churn"),
                       ("d_reddit_s5", "reddit@S5"), ("d_reddit_s3", "reddit@S3")):
            v = row[m]; a = m.replace("d_", "").replace("_", "")
            k = {"d_final_ffr": "final_ffr", "d_cum_churn": "cum_churn",
                 "d_reddit_s5": "reddit_s5", "d_reddit_s3": "reddit_s3"}[m]
            print(f"    {lab:>10}: {a1[k]} -> {a2[k]}   Δ={v['delta']} band={v['band']} "
                  f"within={v['within']} ({v['ratio']}x)", flush=True)
    (SB / "evolbench-draw2-deviation.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote {SB / 'evolbench-draw2-deviation.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
