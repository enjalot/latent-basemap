"""Card013 scoring CLOSEOUT (per card013-completion-review). CPU reporting repairs computed from the v1
saved per-query arrays (NO re-projection): (1) SOURCE-STRATIFIED paired bootstrap (resample within each
source at fixed per-source counts) for the radius-corr differences; (2) equal-9-cohort aggregate recall
guards (not only pooled means); (3) per-DECILE (encoder-radius) recall/radius/corr diagnostics + per-SOURCE
corr/recall; (4) persist panel indices + ref/query IDs + finite checks + tie/self + val/ref-disjoint doc;
(5) corrected validator/provenance wording (existence-only pre-check + independent completion audit covers
loadable/fresh/inputs). Preserves v1 (card013-score.json -> card013-score-v1.json). No gate relaxation.
Usage: score_card013_closeout.py
"""
import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
from score_card009 import sha, write_json
from scipy.stats import spearmanr

OC = Path("/data/latent-basemap/sandbox/overseer-codex"); OUT = OC / "card013-scoring"
SEAL = Path("/data2/monet/eval-common-v2"); ARMS = ["baseline", "actual_radii", "shuffled_radii"]
OLD = ["laion", "coyo", "commoncatalog-cc-by", "megalith10m", "cc12m"]
BUDGETS = (50, 100, 250, 500, 1000, 2000); SEED = 13013; BOOT = 2000


def strat_boot_corr_diff(enc, m_a, m_b, groups, seed):
    """Source-stratified paired bootstrap of spearman(enc,m_a) - spearman(enc,m_b): resample WITHIN each
    source at its fixed count, pool, recompute both spearmans. Returns [lo, hi] CI95 + point diff."""
    rng = np.random.default_rng(seed)
    by_src = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
    draws = np.empty(BOOT)
    for i in range(BOOT):
        idx = np.concatenate([rng.choice(ix, ix.size, replace=True) for ix in by_src.values()])
        draws[i] = spearmanr(enc[idx], m_a[idx]).statistic - spearmanr(enc[idx], m_b[idx]).statistic
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))], float(np.mean(draws))


def main():
    t0 = time.time()
    z = np.load(OUT / "per-query.npz", allow_pickle=True)
    enc = z["enc_radius"]; groups = z["val_source"].astype(str); nq = enc.shape[0]
    mapr = {a: z[f"{a}_map_radius"] for a in ARMS}
    v1 = json.loads((OC / "card013-score.json").read_text())
    # preserve v1
    if not (OC / "card013-score-v1.json").exists():
        write_json(OC / "card013-score-v1.json", v1)

    # (1) source-stratified paired bootstrap
    ci_ab, d_ab = strat_boot_corr_diff(enc, mapr["actual_radii"], mapr["baseline"], groups, SEED)
    ci_as, d_as = strat_boot_corr_diff(enc, mapr["actual_radii"], mapr["shuffled_radii"], groups, SEED + 1)
    corr = {a: float(spearmanr(enc, mapr[a]).statistic) for a in ARMS}
    gain = corr["actual_radii"] - corr["baseline"]

    # (2) equal-9 recall guards + (3) per-source corr/recall
    def rec(a, b): return z[f"{a}_B{b}"]
    equal9 = {a: {str(b): float(np.mean([rec(a, b)[groups == g].mean() for g in np.unique(groups)])) for b in BUDGETS} for a in ARMS}
    per_source = {g: {"n": int((groups == g).sum()),
                      "corr": {a: float(spearmanr(enc[groups == g], mapr[a][groups == g]).statistic) for a in ARMS},
                      "B250": {a: float(rec(a, 250)[groups == g].mean()) for a in ARMS}} for g in np.unique(groups)}
    # equal-9 recall guard: actual within .005 of baseline on the equal-9 mean; cohorts within .01
    b250_eq_loss = equal9["baseline"]["250"] - equal9["actual_radii"]["250"]
    b2000_eq_loss = equal9["baseline"]["2000"] - equal9["actual_radii"]["2000"]
    cohort_loss = {str(b): {g: float(rec("baseline", b)[groups == g].mean() - rec("actual_radii", b)[groups == g].mean()) for g in np.unique(groups)} for b in (250, 2000)}
    cohort_ok = all(v <= .01 for d in cohort_loss.values() for v in d.values())

    # (3) per-DECILE (encoder-radius) diagnostics: recall/radius/corr within each decile
    edges = np.percentile(enc, np.arange(0, 101, 10)); dec_idx = np.clip(np.digitize(enc, edges[1:-1]), 0, 9)
    deciles = []
    for d in range(10):
        m = dec_idx == d
        deciles.append({"decile": d + 1, "n": int(m.sum()),
                        "enc_radius_mean": round(float(enc[m].mean()), 5),
                        "map_radius_mean": {a: round(float(mapr[a][m].mean()), 5) for a in ARMS},
                        "corr_within": {a: (round(float(spearmanr(enc[m], mapr[a][m]).statistic), 4) if m.sum() > 2 else None) for a in ARMS},
                        "B250": {a: round(float(rec(a, 250)[m].mean()), 5) for a in ARMS}})

    # continuity/severe from v1 (unchanged; guard is nonincrease, NOT an improvement claim)
    cont = v1["continuity"]; sev = v1["severe_frac"]
    cont_decl = cont["baseline"] - cont["actual_radii"]; sev_incr = sev["actual_radii"] - sev["baseline"]

    # (4) persistence: panel indices (deterministic), ref/query IDs, finite checks, disjoint doc
    rng = np.random.default_rng(SEED)
    panel = np.sort(np.concatenate([rng.choice(np.flatnonzero(groups == g), 200, replace=False) for g in np.unique(groups)]))
    ref_ids = np.load(SEAL / "ref_idx.npy"); val_ids = np.load(SEAL / "val_idx.npy")
    disjoint = bool(not np.isin(val_ids, ref_ids).any())
    finite_ok = bool(np.isfinite(enc).all() and all(np.isfinite(mapr[a]).all() for a in ARMS))
    np.savez(OUT / "closeout-persist.npz", panel_local=panel, panel_val_ids=val_ids[panel],
             ref_ids=ref_ids, val_ids=val_ids, enc_radius=enc, **{f"{a}_map_radius": mapr[a] for a in ARMS})

    gate = {"radius_corr_gain_ge_010": bool(gain >= .10),
            "radius_corr_gain_source_stratified_ci_positive": bool(ci_ab[0] > 0),
            "radius_corr_gt_shuffled_stratified": bool(ci_as[0] > 0),
            "equal9_B250_within_005": bool(b250_eq_loss <= .005), "equal9_B2000_within_005": bool(b2000_eq_loss <= .005),
            "no_cohort_recall_loss_gt_01": bool(cohort_ok),
            "continuity_within_005": bool(cont_decl <= .005), "severe_false_join_within_005": bool(sev_incr <= .005)}
    report = {"schema": "card013-score-v2-closeout-2026-09-12", "status": "SCORED", "supersedes": "card013-score-v1.json (preserved)",
              "primary": "held-out Spearman corr(encoder RMS radius, MAP RMS radius) vs ORIGINAL eval-common-v2 ref; SOURCE-STRATIFIED paired bootstrap",
              "radius_spearman": corr, "corr_gain_actual_minus_baseline": gain,
              "corr_gain_source_stratified_ci95": ci_ab, "corr_gain_source_stratified_mean": d_ab,
              "corr_vs_shuffled_source_stratified_ci95": ci_as,
              "equal9_recall": equal9, "equal9_B250_loss_vs_baseline": b250_eq_loss, "equal9_B2000_loss_vs_baseline": b2000_eq_loss,
              "worst_cohort_recall_loss": max(v for d in cohort_loss.values() for v in d.values()),
              "per_source": per_source, "encoder_radius_deciles_report": deciles,
              "continuity": cont, "continuity_decline_actual": cont_decl,
              "severe_frac": sev, "severe_frac_increase_actual": sev_incr,
              "severe_join_note": "Zero/low severe counts = an INSENSITIVE panel, NOT proof of absent false joins; the guard is nonincrease (|actual-baseline|<=.005), NOT evidence of improvement.",
              "persistence": {"panel_n": int(panel.size), "ref_ids": int(ref_ids.size), "val_ids": int(val_ids.size),
                              "val_ref_disjoint": disjoint, "finite_ok": finite_ok,
                              "tie_self_rule": "faiss/cKDTree fixed-k; queries held out from ref+training so no self; ties broken by index order",
                              "arrays": str(OUT / "closeout-persist.npz")},
              "validation_provenance": "pre-scoring validator checks dose/snapshot-EXISTENCE/finite-coords/admission/init/radius-hash; loadable-snapshot + freshness + input identity are covered by the INDEPENDENT completion audit (card013-completion-independent-audit.json), not the existence-only pre-check.",
              "matched_time_context": v1.get("matched_time_context"),
              "matched_time_note": "60K matched-DOSE with measured runtime 985-988s/arm (~60.7-60.9 upd/s, practically equal), NOT a separately selected equal-time endpoint.",
              "gate": gate, "GATE_PASS": bool(all(gate.values())),
              "note": "Scale-fidelity screen; single-seed development. Higher density corr with worse retrieval is a tradeoff, not promotion. Encoder local radius != intrinsic dimension/semantic density.",
              "uncertainty": f"{BOOT} SOURCE-STRATIFIED paired bootstrap draws (resample within source at fixed counts), seed {SEED}.",
              "scorer_v2_sha256": sha(__file__), "cpu_wall_s": time.time() - t0}
    write_json(OUT / "result-v2.json", report); write_json(OC / "card013-score.json", report)
    print(json.dumps({"radius_spearman": corr, "corr_gain": round(gain, 4), "stratified_ci": ci_ab,
                      "gate": gate, "GATE_PASS": all(gate.values())}, indent=1), flush=True)


if __name__ == "__main__":
    main()
