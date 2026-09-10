"""Card-005 result-review corrections (CPU, from saved per-query arrays — NO re-projection). Fixes:
(1) gate BUG: per-arm AGGREGATE synthetic gain >=0.03 + each arm's OWN paired CI>0 (was per-source individual +
    anchored-CI-for-both). (2) natural pool-weighted reception aggregates, separate from equal-source cohort means.
(3) relabel cross-source as synthetic-to-non-arriving (carried from v1). (4) fold in the overseer's wholly-unseen
drift tail (card005-independent-audit + tail-check) — distinct from in-training-graph anchor-holdouts. Overall gate
stays FAIL. Writes card005-compare-v2.json (v1 preserved). Usage: card005_fix.py
"""
import json
from pathlib import Path
import numpy as np
from card005_gates import arrival_gate

OC = Path("/data/latent-basemap/sandbox/overseer-codex")
ARRIVING = ["synthetic-flux-klein", "synthetic-flux-schnell", "synthetic-z-image"]
OLD = ["laion", "coyo", "commoncatalog-cc-by", "megalith10m", "cc12m"]
# pool source counts (measured earlier) for natural (pool-weighted) aggregates; diffusion-aesthetic-4k diagnostic-only
POOL = {"laion": 8596445, "coyo": 3548209, "commoncatalog-cc-by": 2040197, "megalith10m": 1548838, "cc12m": 1119147,
        "synthetic-z-image": 1133890, "synthetic-flux-schnell": 734392, "synthetic-flux-klein": 622429}
BUDS = (250, 2000)


def main():
    z = np.load(OC / "card005-perq.npz", allow_pickle=True)
    src = z["val_source"].astype(str)
    v1 = json.load(open(OC / "card005-compare-v1.json"))

    def per(arm, B): return z[f"{arm}_B{B}"]
    # (1) CORRECTED gate: per-arm AGGREGATE arriving gain + each arm's OWN paired CI (equal-source support: 3x1200)
    gate = {}
    for arm in ("anchored", "unanchored"):
        old = {B: {c: float(per(arm, B)[src == c].mean() - per("frozen_t0", B)[src == c].mean()) for c in OLD} for B in BUDS}
        g = {"per_source_arriving_B250": {c: round(float(per(arm, 250)[src == c].mean() - per("frozen_t0", 250)[src == c].mean()), 4) for c in ARRIVING},
             "old_mean": {f"B{B}": round(float(np.mean(list(old[B].values()))), 4) for B in BUDS},
             "old_worst": {f"B{B}": round(float(min(old[B].values())), 4) for B in BUDS}}
        g.update(arrival_gate(per(arm, 250), per("frozen_t0", 250), src, ARRIVING))
        g["gate_old_stable"] = bool(all(np.mean(list(old[B].values())) >= -0.005 and min(old[B].values()) >= -0.01 for B in BUDS))
        gate[arm] = g

    # (2) natural pool-weighted reception aggregate per head/budget (vs equal-source cohort mean)
    tot = sum(POOL.values()); heads = ["frozen_t0", "anchored", "unanchored", "fresh_final"]
    natw, cohort = {}, {}
    for h in heads:
        natw[h] = {f"B{B}": round(float(sum(POOL[c] * per(h, B)[src == c].mean() for c in POOL) / tot), 4) for B in BUDS}
        cohort[h] = {f"B{B}": round(float(np.mean([per(h, B)[src == c].mean() for c in POOL])), 4) for B in BUDS}

    out = {"schema": "card005-compare-v2-2026-09-10-corrected",
           "corrections": ["gate: per-arm AGGREGATE synthetic gain>=0.03 + each arm's OWN paired CI (was per-source + anchored-CI-both)",
                           "natural pool-weighted aggregates added, separate from equal-source cohort means",
                           "cross-source relabeled synthetic-to-non-arriving",
                           "wholly-unseen drift tail folded in (distinct from in-training-graph anchor-holdouts)"],
           "reception_gates": gate,
           "arriving_agg_gain_B250": {"anchored": gate["anchored"]["arriving_agg_gain_B250"], "anchored_ci95": gate["anchored"]["arriving_agg_ci95_B250"],
                                      "fresh_gap_frac_recovered_audit": 0.393, "fresh_gain_audit": 0.021926},
           "natural_pool_weighted_reception": natw, "equal_source_cohort_reception": cohort,
           "cross_source_recovery_B250_RELABELED": {"metric": "synthetic-to-non-arriving: fraction of synthetic queries' true NN OUTSIDE all 3 synthetic sources (INCLUDES diagnostic diffusion-aesthetic-4k, NOT only the 5 real) — do not compare directly to card-004's Chinese cross-language stat without matching group semantics",
                                                     "values": v1.get("cross_source_recovery_B250")},
           "unseen_real_query_drift_tail": {"source": "overseer card005-independent-audit.json + card005-tail-check.json (not duplicated)",
                                            "scope": "WHOLLY-UNSEEN real seal queries (6000) — DISTINCT from the in-training-graph anchor-holdout rows the movement gate scored",
                                            "anchored": {"mean": 0.008, "p95": 0.012742, "p99": 0.194085, "n_gt_0.05": 164, "n_gt_0.10": 108, "by_source": {"laion": 35, "coyo": 23, "commoncatalog-cc-by": 28, "megalith10m": 36, "cc12m": 42}},
                                            "unanchored": {"mean": 0.65463, "p99": 1.5501}, "fresh": {"mean": 0.40884, "p99": 2.6298},
                                            "artifact_ruled_out": "raw vs L2-norm same p99 ~0.194; CPU-reload-vs-saved p99<5e-7 radius",
                                            "interpretation": "anchors help greatly vs alternatives, BUT the high-percentile unseen tail is REAL and WOULD EXCEED the 0.05 movement threshold if that deployment gate were applied to unseen queries. The card-005 movement-gate PASS (anchor-holdout mean 0.0029/p99 0.0091) is on in-training-graph rows ONLY and is NOT a universal-unseen-stability guarantee."},
           "measured_gpu": {"T0_s": 4035.27, "fresh_final_s": 4862.03, "anchored_s": 2406.4, "unanchored_s": 2298.6,
                            "training_gpu_h": 3.778, "note": "graph/load/projection overhead separate; caps met (not estimates)",
                            "T0_positive_lr_updates": 338332, "T0_amp_skips": 155, "fresh_final_positive_lr_updates": 407949},
           "OVERALL_PREREG_GATE_PASS": bool(gate["anchored"]["gate_new_source"] and gate["anchored"]["gate_old_stable"]
                                             and v1["movement_fixed_T0_radius"]["anchored"]["gate_movement_holdout"]),
           "note": "Overall FAIL unchanged. v1 (per-source gate bug) preserved at card005-compare-v1.json. Overseer owns canonical latent-labs entry."}
    (OC / "card005-compare-v2.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"gates": gate, "natural_vs_cohort_B250": {h: (natw[h]["B250"], cohort[h]["B250"]) for h in heads},
                      "overall": out["OVERALL_PREREG_GATE_PASS"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
