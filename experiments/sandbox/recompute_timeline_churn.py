"""Recompute the full-UMAP competitor timeline's churn with the frame.py instrument (overseer-approved
2026-09-05), so the Article-2 subtraction (timeline churn − optimizer floor) uses ONE gauge for both terms.
CPU, no fitting — just loads the saved per-snapshot coords and rigid-aligns consecutive pairs.

The competitor timeline (umap-learn 0.6, random_state=None, per-snapshot full refits) is at
evolbench-competitor-umap-full_timeline/coords-S{0..5}.npy (cumulative appends). Per-step churn = frame.churn
of the shared leading rows Sk→Sk+1 (rigid-aligned + frame-normed). Cumulative = sum of per-step means.

Output: sandbox/evolbench-competitor-umap-full_timeline/churn-framegauge.json.
"""
import json, sys
from pathlib import Path
import numpy as np

TL = Path("/data/latent-basemap/sandbox/evolbench-competitor-umap-full_timeline")
HERE = Path(__file__).resolve().parent


def main():
    sys.path.insert(0, str(HERE))
    import frame
    snaps = sorted(TL.glob("coords-S*.npy"), key=lambda p: int(p.stem.split("S")[1]))
    coords = {p.stem.split("-")[1]: np.load(p).astype(np.float64) for p in snaps}
    keys = [f"S{i}" for i in range(len(snaps))]
    per_step = []
    cum = 0.0
    for a, b in zip(keys[:-1], keys[1:]):
        disp, info = frame.churn(coords[b], coords[a])   # shared leading rows a->b, rigid-aligned + frame-normed
        m = float(disp.mean()); p95 = float(np.percentile(disp, 95))
        cum += m
        per_step.append({"step": f"{a}->{b}", "n_shared": int(coords[a].shape[0]),
                         "churn_mean": round(m, 5), "churn_p95": round(p95, 5),
                         "learned_scale": info.get("learned_scale")})
        print(f"[timeline] {a}->{b}: churn mean {m:.5f} p95 {p95:.5f}", flush=True)
    out = {"schema": "competitor-timeline-churn-framegauge-2026-09-05",
           "instrument": "frame.py churn (rigid-aligned rotation+translation, no scale; frame-normed)",
           "per_step": per_step, "cumulative_churn_framegauge": round(cum, 5),
           "note": "same gauge as the noise-floor pair; Article-2 growth-churn = cumulative − floor_mean."}
    (TL / "churn-framegauge.json").write_text(json.dumps(out, indent=1))
    print(f"[timeline] CUMULATIVE frame-gauge churn = {cum:.5f} -> {TL/'churn-framegauge.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
