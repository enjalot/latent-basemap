"""Card010 adaptive-neighborhood graph builder (CPU/faiss, off-flock heavy-CPU).
Fixed 300K DINO-1536 draw (real sources, seal-excluded); candidate kNN k_max=60; ABIDE-style
binomial k* local-uniformity criterion (2NN global ID + half-radius G-test, Dthr=23.928, clamp
[5,60]) — an implementation of the criterion from note 0243 (Di Noia et al 2026), NOT dadapy's code
and NOT a distance-gap heuristic. Builds three matched graphs (fixed15 / adaptive k* / fixed-mean-k),
runs the preregistered viability audits, and STOPS honestly if any fails. Outputs edges-*.npz +
card010-viability.json + draw ids. Usage: build_card010_graph.py
"""
import os, sys, json, hashlib
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "6")
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import faiss

POOL = Path("/data2/monet/pool-20m"); SEAL = Path("/data2/monet/eval-common-v2")
OUT = Path("/data/latent-basemap/substrates/card010-adaptive"); OUT.mkdir(parents=True, exist_ok=True)
OC = Path("/data/latent-basemap/sandbox/overseer-codex")
SOURCES = ["laion", "coyo", "commoncatalog-cc-by", "megalith10m", "cc12m"]
N_DRAW = 300000; KMAX = 60; DTHR = 23.928; KLO, KHI = 5, 60; SEED = 10010


def _norm(a):
    a = np.asarray(a, np.float32); return a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-12)


def two_nn_id(d1, d2):   # Facco 2NN MLE global ID from mu = r2/r1
    mu = (d2 / d1.clip(1e-12)); mu = mu[np.isfinite(mu) & (mu > 1)]
    return float(mu.shape[0] / np.sum(np.log(mu)))


def kstar_binomial(dist, ID):
    """ABIDE-style binomial k* (VECTORIZED over points): for each k, count neighbors inside the inner
    half-radius rk*2^(-1/ID); under local uniformity that count ~ Binomial(k,0.5); G-test stat Dk<DTHR.
    k* = largest k (from KLO) BEFORE the first k whose Dk>=DTHR; clamp [KLO,KHI]. Not a distance-gap rule."""
    n = dist.shape[0]; half = 2.0 ** (-1.0 / max(ID, 1e-6))
    first_fail = np.full(n, KHI + 1, np.int32)
    for k in range(KLO, KHI + 1):
        thr = dist[:, k - 1] * half
        m = (dist[:, :k] <= thr[:, None]).sum(1).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = np.where(m > 0, m * np.log(2.0 * m / k), 0.0)
            t2 = np.where(k - m > 0, (k - m) * np.log(2.0 * (k - m) / k), 0.0)
        Dk = 2.0 * (t1 + t2)
        newly = (Dk >= DTHR) & (first_fail == KHI + 1)
        first_fail[newly] = k
    return np.clip(first_fail - 1, KLO, KHI).astype(np.int32)


def edges_from_k(nbr, kfunc):
    """directed kNN edges: each row i -> its first k_i neighbors. kfunc(i)->k. binary weight."""
    n = nbr.shape[0]; src = []; dst = []
    ks = kfunc if isinstance(kfunc, np.ndarray) else np.full(n, kfunc, np.int32)
    for i in range(n):
        k = int(ks[i]); src.append(np.full(k, i, np.int32)); dst.append(nbr[i, :k].astype(np.int32))
    s = np.concatenate(src); d = np.concatenate(dst)
    return s, d, ks


def giant_component_frac(nbr, ks):
    """scipy connected components on the symmetrized adaptive graph -> largest fraction + #components."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = nbr.shape[0]
    src = np.concatenate([np.full(int(ks[i]), i, np.int32) for i in range(n)])
    dst = np.concatenate([nbr[i, :int(ks[i])] for i in range(n)]).astype(np.int32)
    A = coo_matrix((np.ones(src.shape[0], np.int8), (src, dst)), shape=(n, n))
    ncomp, labels = connected_components(A, directed=True, connection="weak")
    _, counts = np.unique(labels, return_counts=True)
    return float(counts.max() / n), int(ncomp)


def main():
    faiss.omp_set_num_threads(6); rng = np.random.default_rng(SEED)
    src_all = np.load(POOL / "source.npy", allow_pickle=True).astype(str)
    seal = np.zeros(src_all.shape[0], bool)
    seal[np.load(SEAL / "ref_idx.npy")] = True; seal[np.load(SEAL / "val_idx.npy")] = True
    # fixed proportional-ish draw: equal per real source (60K each -> 300K)
    per = N_DRAW // len(SOURCES); draw = []
    for s in SOURCES:
        cand = np.where((src_all == s) & (~seal))[0]
        draw.append(rng.choice(cand, per, replace=False))
    draw = np.sort(np.concatenate(draw)); n = draw.shape[0]
    Xmm = np.load(POOL / "dino1536.f16.npy", mmap_mode="r")
    X = _norm(np.asarray(Xmm[draw], np.float32)); dsrc = src_all[draw]
    print(f"{n} rows drawn; building candidate kNN k={KMAX} (IVF)", flush=True)

    d = X.shape[1]; nlist = 2048
    quant = faiss.IndexFlatIP(d); index = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(X); index.add(X); index.nprobe = 64
    S, I = index.search(X, KMAX + 1)                                  # +1 for self
    # drop self (col where I==row), keep KMAX
    nbr = np.empty((n, KMAX), np.int32); sim = np.empty((n, KMAX), np.float32)
    for i in range(n):
        row = I[i]; ss = S[i]; keep = row != i
        nbr[i] = row[keep][:KMAX]; sim[i] = ss[keep][:KMAX]
    dist = np.sqrt(np.clip(2.0 - 2.0 * sim, 0, None))                 # cosine -> euclidean on unit sphere
    print("kNN done; estimating 2NN ID + binomial k*", flush=True)

    ID = two_nn_id(dist[:, 0], dist[:, 1])
    kstar = kstar_binomial(dist, ID)
    kmean = int(round(float(kstar.mean())))
    print(f"ID={ID:.3f} k*: mean {kstar.mean():.2f} med {np.median(kstar)} min {kstar.min()} max {kstar.max()} fixed_mean={kmean}", flush=True)

    # ---- viability audits (preregistered numeric stops) ----
    from scipy.sparse import coo_matrix, csr_matrix
    from scipy.sparse.csgraph import connected_components
    cap_frac = float((kstar >= KHI).mean()); floor_frac = float((kstar <= KLO).mean())
    dup_frac = float((dist[:, 0] <= 1e-6).mean())
    # adaptive directed sparse graph (built once) -> connectivity, reciprocal fraction, indegree
    a_src = np.concatenate([np.full(int(kstar[i]), i, np.int32) for i in range(n)])
    a_dst = np.concatenate([nbr[i, :int(kstar[i])] for i in range(n)]).astype(np.int32)
    A = csr_matrix((np.ones(a_src.shape[0], np.int8), (a_src, a_dst)), shape=(n, n))
    ncomp, labels = connected_components(A, directed=True, connection="weak")
    _, counts = np.unique(labels, return_counts=True); gc_frac = float(counts.max() / n)
    recip = float(A.multiply(A.T).nnz / A.nnz) if A.nnz else 0.0
    indeg = np.asarray(A.sum(0)).ravel()
    src_exposure = {s: round(float(indeg[dsrc == s].mean()), 3) for s in SOURCES}
    # ANN fidelity: exact kNN@15 on a 3000-row panel (CPU flat over the 300K draw)
    panel = np.sort(rng.choice(n, 3000, replace=False))
    flat = faiss.IndexFlatIP(d); flat.add(X)
    _, Ie = flat.search(X[panel], 16)
    rec = []
    for pi, gi in enumerate(panel):
        ex = set(int(x) for x in Ie[pi] if int(x) != gi)
        ap = set(int(x) for x in nbr[gi, :15]); rec.append(len(ex & ap) / 15.0)
    ann_recall15 = float(np.mean(rec))

    audits = {"n": n, "ID_2nn": round(ID, 4), "kstar_mean": round(float(kstar.mean()), 3), "kstar_median": int(np.median(kstar)),
              "cap_saturation_frac": round(cap_frac, 4), "floor_saturation_frac": round(floor_frac, 4),
              "duplicate_1nn_frac": round(dup_frac, 6), "giant_component_frac": round(gc_frac, 5), "n_components": ncomp,
              "ann_recall_at15_vs_exact": round(ann_recall15, 4), "reciprocal_edge_frac": round(float(recip), 4),
              "fixed_mean_k": kmean, "per_source_indegree_exposure": src_exposure}
    viable = bool(cap_frac < 0.25 and floor_frac < 0.10 and dup_frac < 0.01 and gc_frac >= 0.99 and ann_recall15 >= 0.98)
    audits["VIABLE"] = viable
    audits["viability_stops"] = {"cap<0.25": cap_frac < 0.25, "floor<0.10": floor_frac < 0.10,
                                 "dup<0.01": dup_frac < 0.01, "giant>=0.99": gc_frac >= 0.99, "ann_recall>=0.98": ann_recall15 >= 0.98}

    np.save(OUT / "draw_ids.npy", draw)
    (OC / "card010-viability.json").write_text(json.dumps(audits, indent=1))
    print(json.dumps(audits, indent=1), flush=True)
    if not viable:
        print("VIABILITY STOP — not building training graphs (honest diagnostic).", flush=True)
        return 3
    # build the 3 matched edge sets
    for name, kf in [("fixed15", np.int32(15)), ("adaptive", kstar), ("fixed_mean", np.int32(kmean))]:
        s, dd, ks = edges_from_k(nbr, kf if isinstance(kf, np.ndarray) else np.full(n, int(kf), np.int32))
        np.savez(OUT / f"edges-{name}.npz", sources=s, targets=dd, weights=np.ones(s.shape[0], np.float32), n_nodes=np.int64(n))
        print(f"edges-{name}: {s.shape[0]} directed edges (mean k {ks.mean():.2f})", flush=True)
    # persist the draw substrate (normalized fp16) for training
    np.save(OUT / "substrate.f16.npy", X.astype(np.float16))
    (OC / "card010-viability.json").write_text(json.dumps({**audits, "kstar_ids_hash": hashlib.sha256(kstar.tobytes()).hexdigest()[:16],
                                                           "draw_ids_hash": hashlib.sha256(draw.tobytes()).hexdigest()[:16]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
