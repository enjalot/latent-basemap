"""COMMON evaluator/scorer for the efficiency-experiment pilots (owner plan 2026-09-08). CPU (faiss + a small head
projection). The single decision instrument every pilot arm is scored by, against the sealed set from eval_seal.py.

Metrics (plan "common decision instrument"):
  * recall(k=15, B) = mean_q |H_15(q) ∩ M_B(q)| / 15, where H_15 = the query's ORIGINAL full-D 15-NN among the 250K
    reference (sealed truth) and M_B = the B nearest REFERENCE map points to the query's PROJECTED coord. FIXED
    B=250 and B=2000 (B=250 is also the 0.1% budget for a 250K reference — kept as the secondary-continuity number).
    Micro-averaged + per-provenance cohort + worst cohort. Query self-matches are impossible (queries ∉ reference).
  * trustworthiness + continuity @ k=15 on the fixed diagnostic sample (ranks capped at the sealed k=50).
  * bootstrap query-level 95% CI on micro recall (queries are correlated units, NOT training replicates).
  * movement (updates): mean/p95/p99 displacement of shared points vs one fixed radius (before+after alignment).

Score+deploy f_theta(x) — NEVER the unconstrained Z. Usage:
  eval_common.py --ckpt <model.pt>                    # project sealed ref+val through the head, score
  eval_common.py --coords-ref <npy> --coords-val <npy>  # score precomputed coords (already f_theta output)
"""
import argparse, json, time
from pathlib import Path
import numpy as np

import os
SEAL = Path(os.environ.get("EVAL_SEAL_DIR", "/data2/monet/eval-common")); K = 15; BUDGETS = (250, 2000); K_CAP = 50


def _load_seal():
    L = lambda n, **kw: np.load(SEAL / n, **kw)
    return dict(ref_hd=np.asarray(L("ref_hd.f16.npy"), np.float32), val_hd=np.asarray(L("val_hd.f16.npy"), np.float32),
                truth_val=L("truth_val.npy"), val_source=L("val_source.npy", allow_pickle=True),
                diag_idx=L("diag_idx.npy"), diag_knn_hd=L("diag_knn_hd.npy"))


def _project(ckpt, X, device="cuda"):
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _paths import ensure_paths; ensure_paths()
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    import torch
    m = ParametricUMAP.load(str(ckpt), device=device); m.model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], 100_000):
            t = torch.from_numpy(X[i:i + 100_000]).to(device)
            out.append(m.model(t).cpu().numpy().astype(np.float32))
    return np.concatenate(out)


def recall_at_B(ref_coords, val_coords, truth_val, val_source):
    import faiss
    d2 = faiss.IndexFlatL2(ref_coords.shape[1]); d2.add(np.ascontiguousarray(ref_coords))
    _, nn = d2.search(np.ascontiguousarray(val_coords), max(BUDGETS))         # (Q, maxB) ref-local map neighbors
    truth_sets = [set(int(x) for x in truth_val[i]) for i in range(truth_val.shape[0])]
    out = {}
    for B in BUDGETS:
        per_q = np.array([len(truth_sets[i] & set(int(x) for x in nn[i, :B])) / K for i in range(nn.shape[0])])
        cohorts = {}
        for s in np.unique(val_source):
            m = val_source == s
            cohorts[str(s)] = round(float(per_q[m].mean()), 4)
        # bootstrap 95% CI on the micro mean (query-level resample)
        rng = np.random.default_rng(0)
        boots = [per_q[rng.integers(0, per_q.size, per_q.size)].mean() for _ in range(1000)]
        ci = (round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4))
        out[f"recall@k15_B{B}"] = {"micro": round(float(per_q.mean()), 4), "ci95": ci,
                                   "worst_cohort": min(cohorts, key=cohorts.get), "worst_cohort_recall": min(cohorts.values()),
                                   "per_source": cohorts, "_per_q": per_q}
    return out


def trust_continuity(ref_coords, ref_hd, diag_idx, k=K):
    """T&C @ k SELF-CONTAINED on the diagnostic sample (Venna & Kaski, FULL ranks among the n_diag points, proper
    normalization). Recomputed here from the diagnostic points' original-D vectors + their map coords, so the
    penalty ranks span the full n_diag population (not capped) and the (2n-3k-1) normalizer is exact."""
    dh = np.ascontiguousarray(ref_hd[diag_idx]); dc = np.ascontiguousarray(ref_coords[diag_idx]); n = dh.shape[0]
    # full neighbor orderings within the sample (self excluded): high-D cosine (desc), map L2 (asc)
    sim = dh @ dh.T; np.fill_diagonal(sim, -np.inf); hd_order = np.argsort(-sim, axis=1)      # (n, n) neighbor ids by hd rank
    dist = ((dc[:, None, :] - dc[None, :, :]) ** 2).sum(-1); np.fill_diagonal(dist, np.inf); map_order = np.argsort(dist, axis=1)
    # rank lookups: hd_rank[i, j] = rank of j among i's high-D neighbors (1-based); same for map
    hd_rank = np.empty((n, n), np.int32); map_rank = np.empty((n, n), np.int32)
    rows = np.arange(n)[:, None]; ranks = np.arange(1, n + 1)[None, :]
    hd_rank[rows, hd_order] = ranks; map_rank[rows, map_order] = ranks
    norm = 2.0 / (n * k * (2 * n - 3 * k - 1)); t_pen = c_pen = 0.0
    for i in range(n):
        hd_topk = set(hd_order[i, :k].tolist()); map_topk = set(map_order[i, :k].tolist())
        for j in map_topk - hd_topk: t_pen += hd_rank[i, j] - k               # false map neighbor: penalize by its hd rank
        for j in hd_topk - map_topk: c_pen += map_rank[i, j] - k              # missed hd neighbor: penalize by its map rank
    return {"trustworthiness@15": round(1 - norm * t_pen, 4), "continuity@15": round(1 - norm * c_pen, 4),
            "n_diag": int(n), "method": "self-contained full-rank Venna-Kaski among the diagnostic sample"}


def movement(coords_before, coords_after, radius):
    """Mean/p95/p99 displacement of shared points as a fraction of a fixed original radius (updates/D)."""
    d = np.linalg.norm(coords_after - coords_before, axis=1) / radius
    return {"mean": round(float(d.mean()), 5), "p95": round(float(np.percentile(d, 95)), 5),
            "p99": round(float(np.percentile(d, 99)), 5), "radius": float(radius)}


def score(ref_coords, val_coords, seal=None, label="map"):
    seal = seal or _load_seal(); t0 = time.time()
    r = recall_at_B(ref_coords, val_coords, seal["truth_val"], seal["val_source"])
    for B in BUDGETS: r[f"recall@k15_B{B}"].pop("_per_q", None)               # drop the raw array from the report
    tc = trust_continuity(ref_coords, seal["ref_hd"], seal["diag_idx"])
    out = {"schema": "eval-common-score-2026-09-08", "label": label, **r, **tc, "score_wall_s": round(time.time() - t0, 1),
           "note": "recall@k15 at fixed B over the sealed 250K reference (original-D truth); B=250 is also the 0.1% "
                   "budget. f_theta(x) scored, never Z. T&C on the fixed diagnostic sample."}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt"); ap.add_argument("--coords-ref"); ap.add_argument("--coords-val")
    ap.add_argument("--label", default="map"); ap.add_argument("--out"); ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); seal = _load_seal()
    if a.ckpt:
        ref_coords = _project(a.ckpt, seal["ref_hd"], a.device); val_coords = _project(a.ckpt, seal["val_hd"], a.device)
    else:
        ref_coords = np.asarray(np.load(a.coords_ref), np.float32); val_coords = np.asarray(np.load(a.coords_val), np.float32)
    out = score(ref_coords, val_coords, seal, a.label)
    print(json.dumps(out, indent=1))
    if a.out: Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
