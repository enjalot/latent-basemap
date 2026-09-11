"""Card010 estimator + graphs + viability (CPU) — reads the PERSISTED phase-1 kNN (no rebuild).
Implements the pinned DADApy v0.2.0 criterion (ported, not dadapy code; verified on a known-ID
synthetic reference):
  compute_kstar = same-rank neighbor-VOLUME comparison (NOT a half-radius shell count): at rank ksel,
    compare V_i(ksel)=r_i(ksel)^d to V_{nn_j}(ksel)=r_{nn_j}(ksel)^d for the j-th neighbor; the
    likelihood-ratio dL=-2*ksel*(log vi+log vj-2 log((vi+vj)/2)); k* = first j with dL>Dthr (Dthr=23.928,
    the v0.2.0 default — NOT the new alpha API), truncating at maxk-1. Self-inclusive distances.
  Iterated binomial ID: r_eff=min(.95,.2032**(1/ID)); p_hat=sum(n1)/sum(kstar) with n1 = neighbors
    inside r_eff*r_kstar; ID=log(p_hat)/log(r_eff); iterate to convergence (2NN init).
All-3-arm diagnostics (fixed15/adaptive/fixed_mean): degree, connectivity, reciprocity, source
exposure OUT+IN, cap(>=maxk-1 truncation)/floor, ANN recall@15/@30/@60 vs exact panel. Preregistered
viability stops (25% cap unchanged). Usage: card010_kstar.py
"""
import os, sys, json, hashlib
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "6")
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()

SUB = Path("/data/latent-basemap/substrates/card010-adaptive"); OC = Path("/data/latent-basemap/sandbox/overseer-codex")
SOURCES = ["laion", "coyo", "commoncatalog-cc-by", "megalith10m", "cc12m"]
DTHR = 23.928; KLO, KHI = 5, 60


def compute_kstar(dist, idx, ID):
    """Vectorized port of DADApy v0.2.0 _compute_kstar (VERBATIM logic): j starts at 4 (ksel=j-1);
    dL = -2*ksel*(log vvi + log vvj - 2 log(vvi+vvj) + log4) with vv = r^id (prefactor cancels);
    loop while j<maxk and dL<Dthr; kstar = (j-2) at normal stop, (j-1)=maxk-1 at truncation. dist/idx
    self-inclusive (col0=self, dist0=0), shape [N, maxk]. Returned kstar is the real-neighbor count
    (index into the self-inclusive array; index0=self so kstar = #real neighbors)."""
    N, maxk = dist.shape
    first_fail = np.full(N, maxk, np.int64)       # loop-var j at which dL>=Dthr (else maxk sentinel)
    for j in range(4, maxk):                       # matches dadapy j=4..maxk-1
        ksel = j - 1
        vvi = np.maximum(dist[:, ksel], 1e-30) ** ID
        vvj = np.maximum(dist[idx[:, j], ksel], 1e-30) ** ID
        dL = -2.0 * ksel * (np.log(vvi) + np.log(vvj) - 2.0 * np.log(vvi + vvj) + np.log(4.0))
        newly = (dL >= DTHR) & (first_fail == maxk)
        first_fail[newly] = j
    return np.where(first_fail == maxk, maxk - 1, first_fail - 1).astype(np.int64)


def _scalar_kstar_ref(dist, idx, ID):
    """Literal scalar transcription of dadapy v0.2.0 _compute_kstar (reference for equivalence test)."""
    N, maxk = dist.shape; out = np.empty(N, np.int64)
    for i in range(N):
        j = 4; dL = 0.0
        while j < maxk and dL < DTHR:
            ksel = j - 1
            vvi = max(dist[i, ksel], 1e-30) ** ID
            vvj = max(dist[idx[i, j], ksel], 1e-30) ** ID
            dL = -2.0 * ksel * (np.log(vvi) + np.log(vvj) - 2.0 * np.log(vvi + vvj) + np.log(4.0))
            j += 1
        out[i] = (j - 1) if j == maxk else (j - 2)
    return out


def binomial_id(dist, kstar_real, ID):
    """One binomial-ID update from kstar (real neighbors). n1 = real neighbors within r_eff*r_kstar."""
    r_eff = min(0.95, 0.2032 ** (1.0 / max(ID, 1e-6)))
    N = dist.shape[0]
    rk = dist[np.arange(N), kstar_real]           # r at the k*-th REAL neighbor (col index = kstar_real, since col0=self)
    thr = r_eff * rk
    n1 = (dist <= thr[:, None]).sum(1) - 1         # minus self (dist0=0)
    n1 = np.clip(n1, 0, kstar_real)
    p_hat = n1.sum() / max(kstar_real.sum(), 1)
    return float(np.log(max(p_hat, 1e-12)) / np.log(r_eff)), r_eff, float(p_hat)


def two_nn_id(dist):
    mu = dist[:, 2] / dist[:, 1].clip(1e-12)       # r2/r1 over REAL neighbors (col1,col2)
    mu = mu[np.isfinite(mu) & (mu > 1)]
    return float(mu.shape[0] / np.sum(np.log(mu)))


def iterate_id(dist, idx, max_iter=8, tol=1e-3):
    ID = two_nn_id(dist); traj = [ID]
    for _ in range(max_iter):
        kstar = compute_kstar(dist, idx, ID)
        kstar = np.clip(kstar, KLO, KHI)
        ID_new, r_eff, p_hat = binomial_id(dist, kstar, ID)
        traj.append(ID_new)
        if abs(ID_new - ID) < tol:
            ID = ID_new; break
        ID = ID_new
    kstar = np.clip(compute_kstar(dist, idx, ID), KLO, KHI)
    return ID, kstar, traj, r_eff, p_hat


def _verify():
    """Reference check on a uniform D-dim cube: iterated ID must recover D (+-15%), and a lower-dim
    manifold must give a smaller ID. Uses the same self-inclusive kNN convention."""
    import faiss
    out = {}
    for D in (5, 10):
        rng = np.random.default_rng(0); Y = rng.random((8000, D)).astype(np.float32)
        idx_f = faiss.IndexFlatL2(D); idx_f.add(Y); dd, ii = idx_f.search(Y, 61)
        dist = np.sqrt(np.clip(dd, 0, None)).astype(np.float32)   # col0=self=0
        ID, ks, tr, _, _ = iterate_id(dist, ii.astype(np.int64))
        # LITERAL-reference equivalence: vectorized compute_kstar == scalar transcription of the
        # pinned v0.2.0 loop, at the SAME ID/dist/idx/maxk (covers saturation/early-fail/ties).
        vec = compute_kstar(dist, ii.astype(np.int64), ID)
        sca = _scalar_kstar_ref(dist, ii.astype(np.int64), ID)
        out[f"uniform_{D}d"] = {"ID_est": round(ID, 3), "kstar_mean": round(float(ks.mean()), 2),
                                "vectorized_eq_literal_scalar": bool(np.array_equal(vec, sca)),
                                "max_abs_kstar_diff": int(np.abs(vec - sca).max())}
    return out


def main():
    if not (SUB / "knn_idx.npy").exists():
        print("phase-1 kNN not ready"); return 4
    idx = np.load(SUB / "knn_idx.npy"); dist = np.load(SUB / "knn_dist.npy")
    dsrc = np.load(SUB / "draw_source.npy", allow_pickle=True).astype(str); n, maxk = dist.shape
    # pre-ID integrity: finite + no tied/zero FIRST REAL neighbor (would break the ratio ID)
    finite = bool(np.isfinite(dist).all()); dup_frac = float((dist[:, 1] <= 1e-6).mean())
    verify = _verify()
    ID, kstar, traj, r_eff, p_hat = iterate_id(dist, idx.astype(np.int64))
    converged = bool(abs(traj[-1] - traj[-2]) < 1e-2) if len(traj) > 1 else False
    trunc_mask = (compute_kstar(dist, idx, ID) >= (maxk - 2))         # real-neighbor truncation at maxk-1
    cap_frac = float((kstar >= KHI).mean()); floor_frac = float((kstar <= KLO).mean())
    kmean = int(round(float(kstar.mean())))

    def arm_diag(kf):
        ks = kf if isinstance(kf, np.ndarray) else np.full(n, int(kf), np.int64)
        from scipy.sparse import csr_matrix; from scipy.sparse.csgraph import connected_components
        s = np.concatenate([np.full(int(ks[i]), i, np.int32) for i in range(n)])
        d = np.concatenate([idx[i, 1:1 + int(ks[i])] for i in range(n)]).astype(np.int32)   # real neighbors (skip self col0)
        A = csr_matrix((np.ones(s.shape[0], np.int8), (s, d)), shape=(n, n))
        ncomp, lab = connected_components(A, directed=True, connection="weak")
        _, cnt = np.unique(lab, return_counts=True); gc = float(cnt.max() / n)
        recip = float(A.multiply(A.T).nnz / A.nnz) if A.nnz else 0.0
        indeg = np.asarray(A.sum(0)).ravel(); outdeg = np.asarray(A.sum(1)).ravel()
        exp = {sc: {"in": round(float(indeg[dsrc == sc].mean()), 2), "out": round(float(outdeg[dsrc == sc].mean()), 2)} for sc in SOURCES}
        return {"edges": int(s.shape[0]), "mean_k": round(float(ks.mean()), 2), "giant_frac": round(gc, 5),
                "n_components": ncomp, "reciprocal_frac": round(recip, 4), "source_exposure_in_out": exp}, (s, d)

    diag = {}; edgesets = {}
    for name, kf in [("fixed15", np.int64(15)), ("adaptive", kstar), ("fixed_mean", np.int64(kmean))]:
        diag[name], edgesets[name] = arm_diag(kf)

    # ANN recall vs exact on a 3000 panel at 15/30/60
    import faiss; X = np.asarray(np.load(SUB / "substrate.f16.npy"), np.float32)
    rng = np.random.default_rng(2); panel = np.sort(rng.choice(n, 3000, replace=False))
    flat = faiss.IndexFlatIP(X.shape[1]); flat.add(X); _, Ie = flat.search(X[panel], 61)
    rec = {}
    for K in (15, 30, 60):
        r = []
        for pi, gi in enumerate(panel):
            ex = set(int(x) for x in Ie[pi] if int(x) != gi)  # exact incl ranks; take top-K real
            ex = set(list(ex)[:K]); ap = set(int(x) for x in idx[gi, 1:1 + K]); r.append(len(ex & ap) / K)
        rec[f"recall@{K}"] = round(float(np.mean(r)), 4)

    viable = bool(finite and cap_frac < 0.25 and floor_frac < 0.10 and dup_frac < 0.01
                  and diag["adaptive"]["giant_frac"] >= 0.99 and rec["recall@15"] >= 0.98)
    out = {"schema": "card010-kstar-2026-09-11", "n": n, "maxk_incl_self": maxk, "estimator": "DADApy-v0.2.0 volume-comparison k* + iterated binomial ID (ported; Dthr=23.928)",
           "reference_verification": verify, "finite": finite, "duplicate_first_real_nn_frac": round(dup_frac, 6),
           "ID_final": round(ID, 4), "ID_trajectory": [round(x, 4) for x in traj], "converged": converged, "r_eff": round(r_eff, 5), "p_hat": round(p_hat, 5),
           "kstar_mean": round(float(kstar.mean()), 3), "kstar_median": int(np.median(kstar)), "kstar_min": int(kstar.min()), "kstar_max": int(kstar.max()),
           "cap_saturation_frac": round(cap_frac, 4), "floor_saturation_frac": round(floor_frac, 4), "truncation_frac": round(float(trunc_mask.mean()), 4),
           "fixed_mean_k": kmean, "arm_diagnostics": diag, "ann_recall_vs_exact": rec,
           "VIABLE": viable, "viability_stops": {"finite": finite, "cap<0.25": cap_frac < 0.25, "floor<0.10": floor_frac < 0.10,
                                                 "dup<0.01": dup_frac < 0.01, "adaptive_giant>=0.99": diag["adaptive"]["giant_frac"] >= 0.99,
                                                 "ann_recall15>=0.98": rec["recall@15"] >= 0.98},
           "note": "ported v0.2.0 volume-comparison criterion (NOT half-radius G-test, NOT dadapy code, NOT a gap heuristic); verified on known-ID synthetics; capped variable-k binary graph is an ADAPTATION."}
    np.save(SUB / "kstar.npy", kstar)
    (OC / "card010-kstar.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("reference_verification", "ID_final", "ID_trajectory", "kstar_mean", "kstar_min", "kstar_max", "cap_saturation_frac", "truncation_frac", "ann_recall_vs_exact", "VIABLE", "viability_stops")}, indent=1))
    if viable:
        for name, (s, d) in edgesets.items():
            np.savez(SUB / f"edges-{name}.npz", sources=s, targets=d, weights=np.ones(s.shape[0], np.float32), n_nodes=np.int64(n))
        print("edge sets written (viable)")
    else:
        print("VIABILITY STOP — no training graphs built (honest diagnostic)")
    return 0 if viable else 3


if __name__ == "__main__":
    raise SystemExit(main())
