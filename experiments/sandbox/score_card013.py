"""Card013 scorer (per card013-launch-scoring-review). PRIMARY reference = the ORIGINAL eval-common-v2 250K
reference (ref_hd/ref_idx) + val queries + truth_val (encoder-k15 among REF) — NOT the 300K training draw.
Project the SAME reference+queries through each arm. Radii = each space's own fixed-15 neighbor distances
against that ORIGINAL reference: encoder radius (query's 15 enc-NN RMS among ref_hd) vs MAP radius (query's
15 map-NN RMS among the projected ref map coords). Primary metric = held-out Spearman corr(encoder radius,
MAP radius). Gate: corr(actual) >= corr(baseline)+.10 with positive paired bootstrap CI AND > shuffled;
recall B250/B2000 within .005 of baseline + every cohort within .01; continuity within .005; severe-false-
join within .005 (0/0 uninformative, never an improvement). All 9 cohorts, held-out queries. CPU.
Usage: score_card013.py
"""
import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "4"
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
from score_card009 import BUDGETS, sha, write_json, project, recall, ParametricUMAP
from score_card011 import ranks, reliability, severe_false_joins
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

SB = Path("/data/latent-basemap/sandbox"); OC = SB / "overseer-codex"; OUT = OC / "card013-scoring"
TRAIN = SB / "card013-train"; SEAL = Path("/data2/monet/eval-common-v2")
RADII = SB / "card013-radii"; ARMS = ["baseline", "actual_radii", "shuffled_radii"]
K = 15; SEED = 13013; BOOT = 2000; INIT_SHA = "589895f037d406ae"
OLD = ["laion", "coyo", "commoncatalog-cc-by", "megalith10m", "cc12m"]


def _norm(a):
    a = np.asarray(a, np.float32); return a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-12)


def prescoring_validator():
    C = {}
    mans = {a: json.loads((TRAIN / f"manifest-{a}.json").read_text()) for a in ARMS}
    C["all_60k"] = all(m["executed_steps"] == 60000 for m in mans.values())
    C["shared_init"] = len({m["shared_init_sha256"] for m in mans.values()}) == 1 and all(m["shared_init_sha256"] == INIT_SHA for m in mans.values())
    C["constant_lr_001"] = all(abs(m["lr_used_min"] - .001) < 1e-12 and abs(m["lr_used_max"] - .001) < 1e-12 for m in mans.values())
    C["radii_arms_diverge"] = mans["actual_radii"]["trained_sha256"] != mans["baseline"]["trained_sha256"] and mans["shuffled_radii"]["trained_sha256"] != mans["baseline"]["trained_sha256"]
    # admission radius sha bound to the persisted radii
    ra = json.loads((TRAIN / "admission-actual_radii.json").read_text()); rs = json.loads((TRAIN / "admission-shuffled_radii.json").read_text())
    import hashlib
    def ash(p): return hashlib.sha256(np.ascontiguousarray(np.load(p)).tobytes()).hexdigest()[:16]
    C["radii_hash_bound"] = ra["radii_sha"] == ash(RADII / "r_actual.npy") and rs["radii_sha"] == ash(RADII / "r_shuffled.npy")
    for a in ARMS:
        c = np.load(TRAIN / f"coords-{a}.npy", mmap_mode="r")
        C[f"{a}_finite"] = all(bool(np.isfinite(np.asarray(c[i:i+65536], np.float32)).all()) for i in range(0, len(c), 65536)) and c.shape == (300000, 2)
        for snap in (20000, 40000, 60000):
            C[f"{a}_snap{snap}"] = (TRAIN / a / f"model-step{snap}.pt").exists()
    return bool(all(C.values())), C


def main():
    import faiss
    t0 = time.time(); OUT.mkdir(exist_ok=True)
    ok, checks = prescoring_validator()
    write_json(OUT / "prescoring-validation.json", {"PASS": ok, "checks": checks})
    if not ok:
        print(json.dumps({"prescoring": {k: v for k, v in checks.items() if v is not True}, "PASS": ok}, indent=1))
        write_json(OC / "card013-score.json", {"status": "PRESCORING_FAIL", "checks": checks}); sys.exit(3)

    ref = _norm(np.asarray(np.load(SEAL / "ref_hd.f16.npy", mmap_mode="r"), np.float32))   # ORIGINAL 250K reference
    val = _norm(np.asarray(np.load(SEAL / "val_hd.f16.npy"), np.float32))
    truth = np.load(SEAL / "truth_val.npy"); groups = np.load(SEAL / "val_source.npy", allow_pickle=True).astype(str)
    ref_ids = np.load(SEAL / "ref_idx.npy"); val_ids = np.load(SEAL / "val_idx.npy")
    nq = val.shape[0]
    assert truth.shape == (nq, K) and (truth >= 0).all() and (truth < len(ref)).all()   # truth indexes REF
    # encoder radius (arm-independent): query's 15 enc-NN RMS distance among the ORIGINAL reference
    fi = faiss.IndexFlatL2(ref.shape[1]); fi.add(ref)
    d2enc, _ = fi.search(val, K); enc_radius = np.sqrt(d2enc.mean(axis=1))
    # fixed panel for continuity/severe-join (val queries; reused across arms, deterministic)
    rng = np.random.default_rng(SEED)
    panel = np.sort(np.concatenate([rng.choice(np.flatnonzero(groups == g), 200, replace=False) for g in np.unique(groups)]))
    H = val[panel]; hi, hr = ranks(cdist(H, H, "cosine"))
    km = faiss.Kmeans(H.shape[1], 16, niter=20, nredo=1, seed=SEED, verbose=False); km.train(H); _, li = km.index.search(H, 1); labels = li[:, 0]
    prov = {"scorer_sha256": sha(__file__), "reference": "eval-common-v2 ref_hd 250K (ORIGINAL common reference)",
            "instrument": {n: sha(SEAL / n) for n in ("ref_hd.f16.npy", "val_hd.f16.npy", "truth_val.npy", "val_source.npy")},
            "heads": {a: {"model_sha256": sha(TRAIN / f"model-{a}.pt"), "coords_sha256": sha(TRAIN / f"coords-{a}.npy"),
                          "it_per_s": json.loads((TRAIN / f"manifest-{a}.json").read_text()).get("it_per_s")} for a in ARMS}}
    write_json(OUT / "provenance.json", prov)

    heads = {}; perq = {"enc_radius": enc_radius, "val_source": groups, "val_ids": val_ids, "truth": truth}; mapr = {}
    for a in ARMS:
        model = ParametricUMAP.load(str(TRAIN / f"model-{a}.pt"), device="cpu").model.eval()
        for p in model.parameters(): p.requires_grad_(False)
        ref_xy = project(model, np.load(SEAL / "ref_hd.f16.npy", mmap_mode="r"), True)
        val_xy = project(model, np.load(SEAL / "val_hd.f16.npy", mmap_mode="r"), True)
        pq = recall(ref_xy, val_xy, truth)
        tree = cKDTree(np.asarray(ref_xy, np.float64))
        d2d, _ = tree.query(np.asarray(val_xy, np.float64), k=K, workers=6)
        mapr[a] = np.sqrt((d2d ** 2).mean(axis=1)); corr = float(spearmanr(enc_radius, mapr[a]).statistic)
        rel = reliability(np.asarray(val_xy[panel], np.float64), hi, hr, labels, km.centroids)
        sfj = severe_false_joins(H, np.asarray(val_xy[panel], np.float64))
        for b in BUDGETS: assert np.isfinite(pq[b]).all(); perq[f"{a}_B{b}"] = pq[b]
        perq[f"{a}_map_radius"] = mapr[a]
        heads[a] = {"radius_spearman": corr, "continuity": rel["continuity"], "trustworthiness": rel["trustworthiness"],
                    "severe_frac": sfj["severe_frac_of_map15"], "severe_count": sfj["severe_false_joins"],
                    "B_recall_real5": {str(b): float(pq[b][np.isin(groups, OLD)].mean()) for b in BUDGETS},
                    "B_recall_equal9": {str(b): float(np.mean([pq[b][groups == g].mean() for g in np.unique(groups)])) for b in BUDGETS}}
        write_json(OUT / f"{a}.json", heads[a]); del model
        print(json.dumps({"arm": a, "radius_spearman": round(corr, 4), "continuity": round(rel["continuity"], 4)}), flush=True)
    # encoder-radius deciles
    dec = [round(float(x), 5) for x in np.percentile(enc_radius, np.arange(10, 100, 10))]
    perq["radius_deciles_enc"] = np.array(dec)
    np.savez(OUT / "per-query.npz", **perq)

    # primary + paired bootstrap (actual - baseline, actual - shuffled)
    dab = np.empty(BOOT); das = np.empty(BOOT)
    for i in range(BOOT):
        ix = rng.integers(0, nq, nq)
        cb = spearmanr(enc_radius[ix], mapr["baseline"][ix]).statistic
        ca = spearmanr(enc_radius[ix], mapr["actual_radii"][ix]).statistic
        cs = spearmanr(enc_radius[ix], mapr["shuffled_radii"][ix]).statistic
        dab[i] = ca - cb; das[i] = ca - cs
    ci_ab = [float(np.percentile(dab, 2.5)), float(np.percentile(dab, 97.5))]
    ci_as = [float(np.percentile(das, 2.5)), float(np.percentile(das, 97.5))]
    gain = heads["actual_radii"]["radius_spearman"] - heads["baseline"]["radius_spearman"]
    def closs(b): return {g: float(perq[f"baseline_B{b}"][groups == g].mean() - perq[f"actual_radii_B{b}"][groups == g].mean()) for g in np.unique(groups)}
    b250 = float(perq["baseline_B250"].mean() - perq["actual_radii_B250"].mean())
    b2000 = float(perq["baseline_B2000"].mean() - perq["actual_radii_B2000"].mean())
    cont_decl = heads["baseline"]["continuity"] - heads["actual_radii"]["continuity"]
    sfj_incr = heads["actual_radii"]["severe_frac"] - heads["baseline"]["severe_frac"]
    gate = {"radius_corr_gain_ge_010": bool(gain >= .10), "radius_corr_gain_ci_positive": bool(ci_ab[0] > 0),
            "radius_corr_gt_shuffled": bool(ci_as[0] > 0),
            "B250_within_005": bool(b250 <= .005), "B2000_within_005": bool(b2000 <= .005),
            "no_cohort_loss_gt_01": bool(all(v <= .01 for bd in (250, 2000) for v in closs(bd).values())),
            "continuity_within_005": bool(cont_decl <= .005),
            "severe_false_join_within_005": bool(sfj_incr <= .005)}
    report = {"schema": "card013-score-2026-09-12", "status": "SCORED",
              "primary": "held-out Spearman corr(encoder RMS radius, MAP RMS radius) vs ORIGINAL eval-common-v2 ref; each space own k15",
              "radius_spearman": {a: heads[a]["radius_spearman"] for a in ARMS},
              "corr_gain_actual_minus_baseline": gain, "corr_gain_ci95": ci_ab,
              "corr_actual_minus_shuffled": heads["actual_radii"]["radius_spearman"] - heads["shuffled_radii"]["radius_spearman"], "corr_vs_shuffled_ci95": ci_as,
              "B250_loss_vs_baseline": b250, "B2000_loss_vs_baseline": b2000,
              "worst_cohort_recall_loss": max(v for bd in (250, 2000) for v in closs(bd).values()),
              "continuity": {a: heads[a]["continuity"] for a in ARMS}, "continuity_decline_actual": cont_decl,
              "severe_frac": {a: heads[a]["severe_frac"] for a in ARMS}, "severe_frac_increase_actual": sfj_incr,
              "severe_join_note": "0/0 severe-frac is uninformative (never an improvement); guard is |actual-baseline|<=.005",
              "B_recall_real5": {a: heads[a]["B_recall_real5"] for a in ARMS},
              "encoder_radius_deciles": dec,
              "matched_time_context": {a: json.loads((TRAIN / f"manifest-{a}.json").read_text()).get("it_per_s") for a in ARMS},
              "matched_time_note": "60K matched-DOSE primary; per-arm it/s above for equal-time context (do NOT call a slower arm's equal-step gain a matched-time gain). 20K/40K/60K(+30K) snapshots retained.",
              "gate": gate, "GATE_PASS": bool(all(gate.values())), "prescoring_validation": checks,
              "note": "Scale-fidelity screen; single-seed development. Higher density corr with worse retrieval is a tradeoff, not promotion. Encoder local radius != intrinsic dimension/semantic density.",
              "uncertainty": f"{BOOT} paired-query bootstrap draws, seed {SEED}.", "cpu_wall_s": time.time() - t0}
    write_json(OUT / "result.json", report); write_json(OC / "card013-score.json", report)
    print(json.dumps({"radius_spearman": report["radius_spearman"], "corr_gain": round(gain, 4), "gate": gate, "GATE_PASS": all(gate.values())}, indent=1), flush=True)


if __name__ == "__main__":
    main()
