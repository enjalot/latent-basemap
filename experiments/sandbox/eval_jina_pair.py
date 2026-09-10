"""Card-003 paired evaluator on the sealed MULTILINGUAL eval-common (CPU, off-flock). Scores TWO jina heads
(baseline A, candidate B) and reports the paired per-query per-cohort delta (B−A) at fixed B=250 & B=2000, with
EN mean/worst register, non-EN mean, cohort-balanced and NATURAL-corpus-weighted aggregates, and paired
query-bootstrap CIs. jina heads use L2-norm preprocessing only (no PCA). Reference stays balanced (the seal's).
All ladder draws excluded the seal ref∪val, so val is held out for every head.

Emits both card gates:
  --gate B (3D eligibility): B250 mean gain >=0.01, aggregate paired CI excludes 0, no cohort loses >0.01 at EITHER budget.
  --gate C (en40):           non-EN mean B250 gain >=0.005, CI excludes 0; EN mean AND each EN-register loss <=0.01 at
                             B250 & B2000; no cohort B2000 loss >0.01.
Usage: eval_jina_pair.py --a <ckptA> --b <ckptB> --labelA prop --labelB 3d --gate B --out <json>
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

SEAL = Path("/data2/monet/eval-common-multilingual"); K = 15; BUDGETS = (250, 2000); CHUNK = 50_000
NAT = {"en-fineweb-edu": 5_000_000, "en-redpajama": 5_000_000, "en-pile": 5_000_000}  # natural corpus weights
for _l in ("arb_Arab", "ces_Latn", "cmn_Hani", "deu_Latn", "ell_Grek", "fra_Latn", "hin_Deva", "ind_Latn",
           "ita_Latn", "jpn_Jpan", "kor_Hang", "nld_Latn", "pol_Latn", "por_Latn", "rus_Cyrl", "spa_Latn",
           "swe_Latn", "tha_Thai", "tur_Latn", "vie_Latn"):
    NAT[f"ml-{_l}"] = 750_000


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def per_query(ckpt, ref_hd, val_hd, truth):
    import torch, faiss
    sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    m = ParametricUMAP.load(str(ckpt), device="cpu"); m.model.eval()

    def proj(X):
        out = []
        with torch.no_grad():
            for i in range(0, X.shape[0], CHUNK):
                out.append(m.model(torch.from_numpy(_norm(X[i:i + CHUNK]))).numpy().astype(np.float32))
        return np.concatenate(out)
    rc = proj(ref_hd); vc = proj(val_hd)
    d2 = faiss.IndexFlatL2(rc.shape[1]); d2.add(np.ascontiguousarray(rc))
    _, nn = d2.search(np.ascontiguousarray(vc), max(BUDGETS))
    pq = {}
    for B in BUDGETS:
        pq[B] = np.array([len(set(int(x) for x in truth[i]) & set(int(x) for x in nn[i, :B])) / K
                          for i in range(nn.shape[0])])
    return pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--labelA", default="A"); ap.add_argument("--labelB", default="B")
    ap.add_argument("--gate", choices=["B", "C"], required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "4")
    import faiss; faiss.omp_set_num_threads(4)
    t0 = time.time()
    ref_hd = np.asarray(np.load(SEAL / "ref_hd.f16.npy"), np.float32)
    val_hd = np.asarray(np.load(SEAL / "val_hd.f16.npy"), np.float32)
    truth = np.load(SEAL / "truth_val.npy"); coh = np.load(SEAL / "val_cohort.npy", allow_pickle=True)
    pqA = per_query(a.a, ref_hd, val_hd, truth); pqB = per_query(a.b, ref_hd, val_hd, truth)
    cohs = sorted(set(coh.tolist())); rng = np.random.default_rng(0)
    tot_nat = sum(NAT[c] for c in cohs)

    out = {"schema": "jina-pair-eval-2026-09-10", "A": a.labelA, "B": a.labelB, "gate": a.gate,
           "n_val": int(val_hd.shape[0]), "per_budget": {}}
    for Bud in BUDGETS:
        dA, dB = pqA[Bud], pqB[Bud]; d = dB - dA
        per_src = {c: round(float(d[coh == c].mean()), 4) for c in cohs}
        boots = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(2000)])
        ci = (round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4))
        en = [c for c in cohs if c.startswith("en-")]; ml = [c for c in cohs if c.startswith("ml-")]
        out["per_budget"][f"B{Bud}"] = {
            "A_micro": round(float(dA.mean()), 4), "B_micro": round(float(dB.mean()), 4),
            "delta_micro": round(float(d.mean()), 4), "delta_paired_ci95": ci,
            "delta_cohort_balanced": round(float(np.mean(list(per_src.values()))), 4),
            "delta_corpus_weighted": round(float(sum(NAT[c] * per_src[c] for c in cohs) / tot_nat), 4),
            "EN_mean_delta": round(float(np.mean([per_src[c] for c in en])), 4),
            "EN_worst_delta": round(float(min(per_src[c] for c in en)), 4),
            "nonEN_mean_delta": round(float(np.mean([per_src[c] for c in ml])), 4),
            "worst_cohort_delta": round(float(min(per_src.values())), 4),
            "per_source_delta": per_src}
    # gates
    b250 = out["per_budget"]["B250"]; b2000 = out["per_budget"]["B2000"]
    if a.gate == "B":
        g = {"B250_mean_gain_ge_0.01": b250["delta_micro"] >= 0.01,
             "aggregate_ci_excludes_0": b250["delta_paired_ci95"][0] > 0,
             "no_cohort_loss_gt_0.01_either_budget": (b250["worst_cohort_delta"] >= -0.01 and b2000["worst_cohort_delta"] >= -0.01)}
    else:
        g = {"nonEN_B250_gain_ge_0.005": b250["nonEN_mean_delta"] >= 0.005,
             "nonEN_ci_excludes_0": b250["delta_paired_ci95"][0] > 0,
             "EN_mean_loss_le_0.01_both": (b250["EN_mean_delta"] >= -0.01 and b2000["EN_mean_delta"] >= -0.01),
             "each_EN_register_loss_le_0.01_both": (b250["EN_worst_delta"] >= -0.01 and b2000["EN_worst_delta"] >= -0.01),
             "no_cohort_B2000_loss_gt_0.01": b2000["worst_cohort_delta"] >= -0.01}
    out["gate_checks"] = g; out["gate_pass"] = all(g.values()); out["wall_s"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({"A": a.labelA, "B": a.labelB, "gate": a.gate, "pass": out["gate_pass"],
                      "B250_delta": b250["delta_micro"], "B250_ci": b250["delta_paired_ci95"],
                      "nonEN_B250": b250["nonEN_mean_delta"], "EN_B250": b250["EN_mean_delta"],
                      "checks": g}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
