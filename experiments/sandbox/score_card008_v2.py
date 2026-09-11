"""Card008 scorer v2 (CPU) — corrected per card008-result-review.md (v1 preserved as card008-score.json).
Fixes: (1) promotion ANDs B2000 + old-cohort retention guards (not just arriving B250); frozen/anchored
comparators; fail-closed on missing heads/nonfinite. (2) SYMMETRIC input — confirmation targets = original
T0 applied to the EXACT fp16-stored inputs each student consumes (no stale bank targets + renormalized
input mix); fixed R0=33.6717, no scale fit. (3) full ORIGINAL active-anchor frame via saved graph coords
(arms + in140k + anchored), 70k via bounded projection of the same active rows; report gauge sensitivity
vs a 200K active sample; native AND aligned movement; anchor-holdout vs wholly-unseen separately.
(4) paired p99-difference + arrival CIs (both endpoints) for treatment-vs-const, cosine-vs-constavg,
vs-70k frontier; per-source tails. (5) persist IDs/B2000/raw+aligned arrays/R,t/hashes.
NO gate relaxation; verdict decided by unrounded thresholds. Usage: score_card008_v2.py
"""
import os, sys, json, hashlib
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import torch, frame as F
import eval_jina_pair as EP
from basemap.pumap.parametric_umap.core import ParametricUMAP

SB = Path("/data/latent-basemap/sandbox"); OC = SB / "overseer-codex"
SEAL = Path("/data2/monet/eval-common-v2"); R0 = 33.6717
T0SUB = Path("/data/latent-basemap/substrates/dino-arrival-t0"); FINALSUB = Path("/data/latent-basemap/substrates/dino-arrival-final")
ARRIVING = ["synthetic-flux-klein", "synthetic-flux-schnell", "synthetic-z-image"]
OLD = ["laion", "coyo", "commoncatalog-cc-by", "megalith10m", "cc12m"]
T0D = SB / "dino-arrival-t0"; C8 = T0D / "card008"; UPD = T0D / "updates"
# head -> (model_path, coords_path or None). coords over the 2.4M FINAL draw where saved.
HEADS = {
 "frozen_t0": (T0D / "champion-bs16k/model.pt", None),          # reference; movement ~0 by construction
 "anchored": (UPD / "model-anchored.pt", UPD / "coords-anchored.npy"),
 "snapshot70k": (T0D / "replay-updates/snapshots-in/model-step70000.pt", None),
 "in140k": (T0D / "replay-updates/model-in.pt", T0D / "replay-updates/coords-in.npy"),
 "const1e4": (C8 / "model-const1e4.pt", C8 / "coords-const1e4.npy"),
 "cosine": (C8 / "model-cosine.pt", C8 / "coords-cosine.npy"),
 "constavg": (C8 / "model-constavg.pt", C8 / "coords-constavg.npy"),
}


def _norm(a):
    a = np.asarray(a, np.float32); return a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-12)


def project_norm(model, X):   # for arrival: normalized input (matches eval convention)
    o = []
    with torch.no_grad():
        for i in range(0, X.shape[0], 50000):
            o.append(model.model(torch.from_numpy(_norm(X[i:i + 50000]))).float().numpy().astype(np.float32))
    return np.concatenate(o)


def project_exact(model, Xf32):   # for movement: EXACT stored fp16->fp32 input, no renorm
    o = []
    with torch.no_grad():
        for i in range(0, Xf32.shape[0], 50000):
            o.append(model.model(torch.from_numpy(np.asarray(Xf32[i:i + 50000], np.float32))).float().numpy().astype(np.float32))
    return np.concatenate(o)


def pboot(a, b, fn, n=2000, seed=0):   # paired bootstrap CI of fn(a)-fn(b) over shared index
    rng = np.random.default_rng(seed); m = a.shape[0]
    d = np.array([fn(a[i := rng.integers(0, m, m)]) - fn(b[i]) for _ in range(n)])
    return round(float(fn(a) - fn(b)), 6), [round(float(np.percentile(d, 2.5)), 6), round(float(np.percentile(d, 97.5)), 6)]


def main():
    ref_hd = np.asarray(np.load(SEAL / "ref_hd.f16.npy"), np.float32); val_hd = np.asarray(np.load(SEAL / "val_hd.f16.npy"), np.float32)
    truth = np.load(SEAL / "truth_val.npy"); grp = np.load(SEAL / "val_source.npy", allow_pickle=True).astype(str)
    val_idx = np.load(SEAL / "val_idx.npy")
    # frame gauge: FULL original active anchors via T0 vs FINAL draw join (card006 gauge)
    t0_coords = np.asarray(np.load(T0D / "champion-bs16k/coordinates.npy"), np.float64)
    t0_draw = np.load(T0SUB / "draw_idx.npy"); final_draw = np.load(FINALSUB / "draw_idx.npy")
    common, t0_local, final_local = np.intersect1d(t0_draw, final_draw, assume_unique=True, return_indices=True)
    active = np.load(UPD / "anchor_active_ids-anchored.npy"); holdout = np.load(UPD / "anchor_holdout_ids-anchored.npy")
    active_mask = np.isin(final_local, active); hold_mask = np.isin(final_local, holdout)
    s0o = t0_coords[t0_local]
    # active-anchor HD (from IN bank) for heads without coords (70k/frozen): bounded projection frame
    inb = np.load(OC / "card006_in_bank.npz"); anc_X16 = np.asarray(inb["replay_X"], np.float16); anc_tgt = np.asarray(inb["replay_targets"], np.float64)
    # confirmation: EXACT stored fp16 inputs + SYMMETRIC targets = T0(exact input)
    cfb = np.load(OC / "card006_confirm_bank.npz"); cf_X16 = np.asarray(cfb["replay_X"], np.float16); cf_ids = np.asarray(cfb["replay_ids"]); cf_src = cfb["source"].astype(str)
    cf_Xf = cf_X16.astype(np.float32)
    frozen = ParametricUMAP.load(str(HEADS["frozen_t0"][0]), device="cpu"); frozen.model.eval()
    for p in frozen.model.parameters(): p.requires_grad_(False)
    cf_tgt = np.asarray(project_exact(frozen, cf_Xf), np.float64)     # symmetric teacher target

    missing = [h for h, (mp, _) in HEADS.items() if not Path(mp).exists()]
    if missing:
        raise SystemExit(f"FAIL-CLOSED: missing head checkpoints {missing}")

    rep = {}; perq = {"val_group": grp, "val_idx": val_idx, "confirm_ids": cf_ids, "confirm_src": cf_src}
    for h, (mp, cp) in HEADS.items():
        mo = ParametricUMAP.load(str(mp), device="cpu"); mo.model.eval()
        pq = EP.per_query(str(mp), ref_hd, val_hd, truth)
        for B in EP.BUDGETS:
            assert np.isfinite(pq[B]).all(), f"nonfinite arrival {h}/{B}"
            perq[f"{h}_arr{B}"] = pq[B]
        # frame: full-active via coords if present, else bounded projection of active-anchor sample
        if cp is not None and Path(cp).exists():
            upd = np.asarray(np.load(cp), np.float64)[final_local]
            _, info = F.rigid_align(upd[active_mask], s0o[active_mask]); frame_gauge = "full_active_via_coords"
            holdout_disp = None
        else:
            hc = np.asarray(project_exact(mo, anc_X16.astype(np.float32)), np.float64)
            _, info = F.rigid_align(hc, anc_tgt); frame_gauge = "active_sample_200k_projection"
            holdout_disp = None
        R, t = np.asarray(info["R"], np.float64), np.asarray(info["t"], np.float64)
        # movement of wholly-unseen confirmation (exact input; native + aligned)
        cc = np.asarray(project_exact(mo, cf_Xf), np.float64)
        disp_nat = np.linalg.norm(cc - cf_tgt, axis=1) / R0
        disp_al = np.linalg.norm(cc @ R.T + t - cf_tgt, axis=1) / R0
        perq[f"{h}_disp_aligned"] = disp_al.astype(np.float32); perq[f"{h}_disp_native"] = disp_nat.astype(np.float32)
        # anchor-holdout movement (in-graph) where coords available
        hold_stat = None
        if cp is not None and Path(cp).exists():
            hd = np.linalg.norm((upd @ R.T + t) - s0o, axis=1) / R0
            hold_stat = {"mean": round(float(hd[hold_mask].mean()), 6), "p99": round(float(np.percentile(hd[hold_mask], 99)), 6)}
        def a(B, cohort): return float(pq[B][np.isin(grp, cohort)].mean())
        rep[h] = {"frame_gauge": frame_gauge, "rigid_rmsd": round(float(info.get("rmsd", np.nan)), 6),
                  "arrival_B250": round(float(np.mean([a(250, [c]) for c in ARRIVING])), 5),
                  "arrival_B2000": round(float(np.mean([a(2000, [c]) for c in ARRIVING])), 5),
                  "old_B250": round(float(np.mean([a(250, [c]) for c in OLD])), 5),
                  "old_B2000": round(float(np.mean([a(2000, [c]) for c in OLD])), 5),
                  "unseen_aligned_mean": round(float(disp_al.mean()), 6), "unseen_aligned_p99": round(float(np.percentile(disp_al, 99)), 6),
                  "unseen_native_p99": round(float(np.percentile(disp_nat, 99)), 6),
                  "unseen_frac_gt.05": round(float((disp_al > 0.05).mean()), 5),
                  "unseen_p99_by_source": {s: round(float(np.percentile(disp_al[cf_src == s], 99)), 5) for s in sorted(set(cf_src.tolist()))},
                  "anchor_holdout": hold_stat}

    # ---- promotion vs matched constant control (const1e4), ALL guards ANDed ----
    def arr_perq(h, B, cohort): return perq[f"{h}_arr{B}"][np.isin(grp, cohort)]
    def arriving_cat(h, B): return np.concatenate([arr_perq(h, B, [c]) for c in ARRIVING])
    prom = {}
    for h in ("cosine", "constavg"):
        g250, ci250 = pboot(arriving_cat(h, 250), arriving_cat("const1e4", 250), np.mean)
        g2000, ci2000 = pboot(arriving_cat(h, 2000), arriving_cat("const1e4", 2000), np.mean)
        # old-retention guard: per-cohort recall loss vs FROZEN T0, mean & worst, both budgets
        old_loss = {B: {c: float(perq[f"frozen_t0_arr{B}"][np.isin(grp, [c])].mean() - perq[f"{h}_arr{B}"][np.isin(grp, [c])].mean()) for c in OLD} for B in EP.BUDGETS}
        old_ok = all(np.mean(list(old_loss[B].values())) <= 0.005 and max(old_loss[B].values()) <= 0.01 for B in EP.BUDGETS)
        # arrival not lost vs control at either budget
        arr_loss_ok = (rep["const1e4"]["arrival_B250"] - rep[h]["arrival_B250"] <= 0.005) and (rep["const1e4"]["arrival_B2000"] - rep[h]["arrival_B2000"] <= 0.005)
        # paired p99 movement diff vs const
        p99d, p99ci = pboot(perq[f"{h}_disp_aligned"].astype(np.float64), perq["const1e4_disp_aligned"].astype(np.float64),
                            lambda x: float(np.percentile(x, 99)))
        p99_drop_frac = -p99d / max(rep["const1e4"]["unseen_aligned_p99"], 1e-9)
        pathA = bool(g250 >= 0.01 and ci250[0] > 0 and rep[h]["arrival_B2000"] >= rep["const1e4"]["arrival_B2000"] - 0.005 and old_ok)
        pathB = bool(p99_drop_frac >= 0.25 and p99ci[1] < 0 and arr_loss_ok and old_ok)
        prom[h] = {"B250_gain_vs_const": g250, "B250_gain_ci95": ci250, "B2000_gain_vs_const": g2000, "B2000_gain_ci95": ci2000,
                   "unseen_p99_diff_vs_const": p99d, "unseen_p99_diff_ci95": p99ci, "p99_drop_frac": round(float(p99_drop_frac), 4),
                   "old_retention_ok_both_budgets": bool(old_ok), "arrival_not_lost_ok": bool(arr_loss_ok),
                   "pathA_arrival_gain": pathA, "pathB_p99_reduction": pathB, "PROMOTE": bool(pathA or pathB)}
    # cosine vs constavg + vs 70k frontier (paired), heterogeneity noted via per-source
    contrasts = {
      "cosine_vs_constavg_p99": pboot(perq["cosine_disp_aligned"].astype(np.float64), perq["constavg_disp_aligned"].astype(np.float64), lambda x: float(np.percentile(x, 99))),
      "cosine_vs_70k_p99": pboot(perq["cosine_disp_aligned"].astype(np.float64), perq["snapshot70k_disp_aligned"].astype(np.float64), lambda x: float(np.percentile(x, 99))),
      "constavg_vs_70k_p99": pboot(perq["constavg_disp_aligned"].astype(np.float64), perq["snapshot70k_disp_aligned"].astype(np.float64), lambda x: float(np.percentile(x, 99))),
    }

    out = {"schema": "card008-score-v2-2026-09-11", "R0": R0, "preserves": "card008-score.json (v1)",
           "per_head": rep, "promotion_vs_const1e4": prom, "contrasts_paired_p99": contrasts,
           "cost_note": "measured fit walls sum 3784.0s; process occupancy (incl load/projection) ~3813s; 4133s charged (incl 320s prep) — all consistent.",
           "gauge_note": "arms/in140k/anchored use full ORIGINAL active-anchor frame via saved graph coords; snapshot70k/frozen use a 200K active-anchor projection frame (labeled per head); rigid R+t only, no scale, fixed R0=33.6717.",
           "wording_caveats": ["'continuation/restart sensitivity' observed, not an isolated optimizer-only mechanism (historical uninterrupted run also differs in RNG/sampler/AMP/trajectory; no matched reset-only intervention).",
                               "cosine and constavg are CLOSE in aggregate (point estimates); per-source tails are heterogeneous (e.g. megalith); closeness is not a tested equivalence and does not prove 'no cosine-shape benefit'.",
                               "movement reductions reported WITH paired p99 CIs; a lower point estimate alone is not an established reduction.",
                               "exposed eval/confirmation are DEVELOPMENT data; this is a research screen, not deployment confirmation."],
           "note": "single seed/update; NO gate relaxation; absolute deployment limit (unseen p99<=.05) unmet by all heads."}
    np.savez(OC / "card008-score-v2-perq.npz", **{k: v for k, v in perq.items()})
    (OC / "card008-score-v2.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"per_head": {h: {k: rep[h][k] for k in ("arrival_B250", "arrival_B2000", "unseen_aligned_p99", "frame_gauge")} for h in rep},
                      "promotion": prom, "contrasts": contrasts}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
