"""Card010 estimator + graphs + viability (CPU) — reads the PERSISTED phase-1 kNN (no rebuild).
Corrected per selector-review-v2 to match the pinned author/DADApy v0.2.0 references EXACTLY:
  - compute_kstar: vector port of dadapy v0.2.0 _compute_kstar. maxk is an EXPLICIT param (= self+60
    = 61). Loop j=4..maxk-1 computing dL@ksel=j-1; the pinned while checks dL BEFORE the volume calc,
    so a failure at the LAST tested ksel (j=maxk-1) is never acted on -> truncation returns maxk-1.
    Vector therefore registers a fail only for j in [4, maxk-2] (kstar=j-1); else truncation maxk-1.
    kstar is dadapy's self-inclusive count. Verified == literal scalar loop on CONSTRUCTED edge cases.
  - iterated binomial ID (author wrapper): n = count(dist < r_eff*r_kstar) INCLUDING self (STRICT <);
    ID = log((mean(n)-1)/(mean(kstar)-1))/log(r_eff); iterate on RAW kstar (no graph clip); fail on
    invalid p/ID. 2NN init. r_eff = min(.95, .2032**(1/ID)).
The [5,60] clamp is applied ONLY when building the graph (raw estimator output preserved). Usage: card010_kstar.py
"""
import os, sys, json, hashlib
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "6")
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()

SUB = Path("/data/latent-basemap/substrates/card010-adaptive"); OC = Path("/data/latent-basemap/sandbox/overseer-codex")
POOL = Path("/data2/monet/pool-20m"); SOURCES = ["laion", "coyo", "commoncatalog-cc-by", "megalith10m", "cc12m"]
DTHR = 23.928; KLO, KHI = 5, 60   # graph clamp only


def compute_kstar(dist, idx, ID, maxk):
    """Vector port of dadapy v0.2.0 _compute_kstar (self-inclusive dist/idx; maxk explicit)."""
    N = dist.shape[0]; first_fail = np.full(N, maxk, np.int64)
    for j in range(4, maxk - 1):                    # dL@ksel=j-1 is CHECKED at j+1<maxk -> j<=maxk-2
        ksel = j - 1
        vvi = np.maximum(dist[:, ksel], 1e-30) ** ID
        vvj = np.maximum(dist[idx[:, j], ksel], 1e-30) ** ID
        dL = -2.0 * ksel * (np.log(vvi) + np.log(vvj) - 2.0 * np.log(vvi + vvj) + np.log(4.0))
        newly = (dL >= DTHR) & (first_fail == maxk)
        first_fail[newly] = j
    return np.where(first_fail == maxk, maxk - 1, first_fail - 1).astype(np.int64)


def scalar_kstar_ref(dist, idx, ID, maxk):
    """Literal transcription of the pinned v0.2.0 scalar loop (reference for equivalence)."""
    N = dist.shape[0]; out = np.empty(N, np.int64)
    for i in range(N):
        j = 4; dL = 0.0
        while j < maxk and dL < DTHR:
            ksel = j - 1
            vvi = max(float(dist[i, ksel]), 1e-30) ** ID
            vvj = max(float(dist[idx[i, j], ksel]), 1e-30) ** ID
            dL = -2.0 * ksel * (np.log(vvi) + np.log(vvj) - 2.0 * np.log(vvi + vvj) + np.log(4.0))
            j += 1
        out[i] = (j - 1) if j == maxk else (j - 2)
    return out


def two_nn_id(dist):
    mu = dist[:, 2] / dist[:, 1].clip(1e-12)          # r2/r1 over real neighbors (cols 1,2; col0=self)
    mu = mu[np.isfinite(mu) & (mu > 1.0)]
    return float(mu.shape[0] / np.sum(np.log(mu)))


def binomial_id(dist, kstar, ID):
    """Author wrapper update: n=count(dist < r_eff*r_kstar) incl self (STRICT <);
    ID=log((mean(n)-1)/(mean(kstar)-1))/log(r_eff). Raw kstar (self-inclusive count)."""
    r_eff = min(0.95, 0.2032 ** (1.0 / max(ID, 1e-6)))
    N = dist.shape[0]
    r_kstar = dist[np.arange(N), kstar]               # self-inclusive index kstar
    n = (dist < (r_eff * r_kstar)[:, None]).sum(1)     # incl self; strict <
    num = float(n.mean()) - 1.0; den = float(kstar.mean()) - 1.0
    if not (num > 0 and den > 0):
        raise ValueError(f"invalid binomial ID inputs: mean(n)-1={num}, mean(kstar)-1={den}")
    ID_new = float(np.log(num / den) / np.log(r_eff))
    if not np.isfinite(ID_new) or ID_new <= 0:
        raise ValueError(f"invalid binomial ID {ID_new}")
    return ID_new, r_eff


def iterate_id(dist, idx, maxk, max_iter=10, tol=1e-3):
    ID = two_nn_id(dist); traj = [ID]
    for _ in range(max_iter):
        kstar = compute_kstar(dist, idx, ID, maxk)     # RAW (no clip)
        ID_new, r_eff = binomial_id(dist, kstar, ID)
        traj.append(ID_new)
        done = abs(ID_new - ID) < tol; ID = ID_new
        if done:
            break
    kstar = compute_kstar(dist, idx, ID, maxk)
    return ID, kstar, traj, bool(len(traj) > 1 and abs(traj[-1] - traj[-2]) < 1e-2)


def _canary():
    """Literal-equivalence on CONSTRUCTED edge cases (early fail, last-rank fail, all-pass truncation,
    ties) + known-ID synthetic recovery. Returns booleans that GATE admission."""
    import faiss
    out = {"cases": {}}
    maxk = 61
    # ENGINEERED geometry so kstar naturally spans the edge branches: a tight cluster (deep points ->
    # near-truncation kstar), a sparse halo (boundary -> early fail), and exact duplicates (ties). Real
    # kNN -> valid self-inclusive idx. The literal-equivalence assertion (vec==scalar) must hold on it.
    rng = np.random.default_rng(1)
    tight = rng.normal(0, 0.02, (400, 8)).astype(np.float32)
    halo = rng.normal(0, 3.0, (200, 8)).astype(np.float32)
    dups = np.repeat(rng.normal(0, 0.02, (20, 8)).astype(np.float32), 6, axis=0)      # 120 exact-tie rows
    Yc = np.concatenate([tight, halo, dups]); f = faiss.IndexFlatL2(8); f.add(Yc)
    dd, ii = f.search(Yc, maxk); dc = np.sqrt(np.clip(dd, 0, None)).astype(np.float64); dc[:, 0] = 0.0
    for ID in (2.0, 6.0):
        vec = compute_kstar(dc, ii.astype(np.int64), ID, maxk); sca = scalar_kstar_ref(dc, ii.astype(np.int64), ID, maxk)
        out["cases"][f"engineered_ID{ID}"] = {"equal": bool(np.array_equal(vec, sca)),
                                              "kstar_min": int(vec.min()), "kstar_max": int(vec.max()),
                                              "has_truncation": bool((vec == maxk - 1).any()), "has_early": bool((vec < 20).any())}
    # known-ID uniform cubes
    for D in (5, 10):
        rng = np.random.default_rng(0); Y = rng.random((8000, D)).astype(np.float32)
        f = faiss.IndexFlatL2(D); f.add(Y); dd, ii = f.search(Y, maxk)
        dist = np.sqrt(np.clip(dd, 0, None)).astype(np.float64); dist[:, 0] = 0.0
        ID, ks, tr, conv = iterate_id(dist, ii.astype(np.int64), maxk)
        vec = compute_kstar(dist, ii.astype(np.int64), ID, maxk); sca = scalar_kstar_ref(dist, ii.astype(np.int64), ID, maxk)
        out["cases"][f"uniform_{D}d"] = {"ID_est": round(ID, 3), "equal": bool(np.array_equal(vec, sca)), "converged": conv,
                                         "ID_in_range": bool(0.6 * D <= ID <= 1.4 * D)}
    out["all_equal"] = bool(all(c["equal"] for c in out["cases"].values()))
    out["synthetic_ID_ok"] = bool(out["cases"]["uniform_5d"]["ID_in_range"] and out["cases"]["uniform_10d"]["ID_in_range"])
    return out


def main():
    req = ["knn_idx.npy", "knn_dist.npy", "draw_ids.npy", "draw_source.npy", "substrate.f16.npy"]
    if not all((SUB / f).exists() for f in req):
        print("phase-1 kNN outputs incomplete — not ready"); return 4
    idx = np.load(SUB / "knn_idx.npy"); dist = np.load(SUB / "knn_dist.npy").astype(np.float64)
    dsrc = np.load(SUB / "draw_source.npy", allow_pickle=True).astype(str); draw = np.load(SUB / "draw_ids.npy")
    n, maxk = dist.shape
    # shape/consistency integrity (a partial write must not pass)
    if not (idx.shape == dist.shape and draw.shape[0] == n and dsrc.shape[0] == n and maxk >= 8):
        print(f"kNN shape mismatch idx{idx.shape} dist{dist.shape} draw{draw.shape} src{dsrc.shape}"); return 4
    # load integrity + pre-ID checks (BEFORE fitting)
    self_ok = bool((idx[:, 0] == np.arange(n)).all())
    dist[:, 0] = 0.0                                    # enforce exact self-distance 0 (IP roundoff)
    finite = bool(np.isfinite(dist).all())
    dup_r1 = float((dist[:, 1] <= 1e-6).mean())         # zero first real neighbor
    tie_r1r2 = float((np.abs(dist[:, 1] - dist[:, 2]) <= 1e-9).mean())   # tied r1==r2 (nonzero)
    canary = _canary()

    ID, kstar_raw, traj, converged = iterate_id(dist, idx.astype(np.int64), maxk)
    raw_trunc_mask = (kstar_raw >= (maxk - 1))          # RAW truncation at the reference maxk
    raw_trunc_frac = float(raw_trunc_mask.mean())
    kgraph = np.clip(kstar_raw, KLO, KHI)               # graph clamp applied SEPARATELY
    kmean = int(round(float(kgraph.mean())))

    from scipy.sparse import csr_matrix; from scipy.sparse.csgraph import connected_components
    def arm(kf):
        ks = kf if isinstance(kf, np.ndarray) else np.full(n, int(kf), np.int64)
        s = np.concatenate([np.full(int(ks[i]), i, np.int32) for i in range(n)])
        d = np.concatenate([idx[i, 1:1 + int(ks[i])] for i in range(n)]).astype(np.int32)
        A = csr_matrix((np.ones(s.shape[0], np.int8), (s, d)), shape=(n, n))
        ncomp, lab = connected_components(A, directed=True, connection="weak")
        _, cnt = np.unique(lab, return_counts=True); gc = float(cnt.max() / n)
        recip = float(A.multiply(A.T).nnz / A.nnz) if A.nnz else 0.0
        indeg = np.asarray(A.sum(0)).ravel(); outdeg = np.asarray(A.sum(1)).ravel()
        exp = {sc: {"in": round(float(indeg[dsrc == sc].mean()), 2), "out": round(float(outdeg[dsrc == sc].mean()), 2)} for sc in SOURCES}
        return {"edges": int(s.shape[0]), "mean_k": round(float(ks.mean()), 3), "giant_frac": gc,
                "n_components": ncomp, "reciprocal_frac": round(recip, 4), "source_exposure_in_out": exp}, (s, d)
    diag = {}; esets = {}
    for name, kf in [("fixed15", np.int64(15)), ("adaptive", kgraph), ("fixed_mean", np.int64(kmean))]:
        diag[name], esets[name] = arm(kf)

    # ANN recall vs exact — reconstruct the EXACT fp32 inputs from draw ids (match candidate build);
    # keep ORDERED exact neighbors, drop self/-1, slice K, then set.
    import faiss
    Xmm = np.load(POOL / "dino1536.f16.npy", mmap_mode="r")
    rng = np.random.default_rng(2); panel = np.sort(rng.choice(n, 3000, replace=False))
    Xf = (lambda a: (a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-12)).astype(np.float32))(np.asarray(Xmm[draw], np.float32))
    flat = faiss.IndexFlatIP(Xf.shape[1]); flat.add(Xf); _, Ie = flat.search(Xf[panel], maxk)
    rec = {}
    for K in (15, 30, 60):
        r = []
        for pi, gi in enumerate(panel):
            ex = [int(x) for x in Ie[pi] if int(x) != gi and int(x) >= 0][:K]
            ap = set(int(x) for x in idx[gi, 1:1 + K] if int(x) >= 0)
            r.append(len(set(ex) & ap) / K)
        rec[f"recall@{K}"] = float(np.mean(r))

    cap_frac = raw_trunc_frac; floor_frac = float((kgraph <= KLO).mean())
    # viability gates — UNROUNDED; evidence booleans ANDed; fail closed on integrity/convergence/reference
    stops = {"self_index_ok": self_ok, "finite": finite, "dup_r1<0.01": dup_r1 < 0.01, "tie_r1r2<0.01": tie_r1r2 < 0.01,
             "literal_equiv": canary["all_equal"], "synthetic_ID_ok": canary["synthetic_ID_ok"], "converged": converged,
             "raw_truncation<0.25": cap_frac < 0.25, "floor<0.10": floor_frac < 0.10,
             "adaptive_giant>=0.99": diag["adaptive"]["giant_frac"] >= 0.99, "ann_recall15>=0.98": rec["recall@15"] >= 0.98}
    viable = bool(all(stops.values()))
    out = {"schema": "card010-kstar-v2-2026-09-11", "n": n, "maxk": maxk, "reference": "DADApy v0.2.0 _compute_kstar + author binomial-ID wrapper (ported; verified vs literal scalar on constructed edge cases)",
           "canary": canary, "self_index_ok": self_ok, "finite": finite, "duplicate_r1_frac": dup_r1, "tie_r1_r2_frac": tie_r1r2,
           "ID_final": ID, "ID_trajectory": traj, "converged": converged,
           "kstar_raw_mean": float(kstar_raw.mean()), "kstar_raw_min": int(kstar_raw.min()), "kstar_raw_max": int(kstar_raw.max()),
           "raw_truncation_frac": raw_trunc_frac, "kgraph_mean": float(kgraph.mean()), "floor_frac": floor_frac, "fixed_mean_k": kmean,
           "arm_diagnostics": diag, "ann_recall_vs_exact_fp32": rec, "viability_stops": stops, "VIABLE": viable,
           "note": "kstar RAW (unclamped) drives ID + truncation gate; [5,60] clamp applied only for the graph. Verified vs literal scalar on constructed edge cases (early/last-rank fail, ties, truncation). ANN audit on exact fp32 inputs, ordered top-K."}
    np.save(SUB / "kstar_raw.npy", kstar_raw); np.save(SUB / "kgraph.npy", kgraph)
    (OC / "card010-kstar.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("canary", "ID_final", "ID_trajectory", "converged", "kstar_raw_min", "kstar_raw_max",
                                          "raw_truncation_frac", "ann_recall_vs_exact_fp32", "viability_stops", "VIABLE")}, indent=1, default=float))
    if viable:
        for name, (s, d) in esets.items():
            np.savez(SUB / f"edges-{name}.npz", sources=s, targets=d, weights=np.ones(s.shape[0], np.float32), n_nodes=np.int64(n))
        print("edge sets written (VIABLE)")
    else:
        print("VIABILITY STOP — honest diagnostic; no training graphs built")
    return 0 if viable else 3


if __name__ == "__main__":
    raise SystemExit(main())
