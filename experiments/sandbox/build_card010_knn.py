"""Card010 phase 1 — fixed draw + candidate kNN, PERSISTED before any estimator/viability work so
estimator repairs never rebuild the expensive search (per selector review). Self-inclusive distances
(dadapy convention: dist[:,0]=0, idx[:,0]=i). maxk=61 -> self + 60 real candidate neighbors.
Skips if outputs already exist. Usage: build_card010_knn.py
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
N_DRAW = 300000; MAXK = 61; SEED = 10010   # MAXK incl self -> 60 real candidate neighbors


def _norm(a):
    a = np.asarray(a, np.float32); return a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-12)


def main():
    if (OUT / "knn_idx.npy").exists() and (OUT / "knn_dist.npy").exists():
        print("kNN already persisted — skipping rebuild"); return 0
    faiss.omp_set_num_threads(6); rng = np.random.default_rng(SEED)
    src_all = np.load(POOL / "source.npy", allow_pickle=True).astype(str)
    seal = np.zeros(src_all.shape[0], bool)
    seal[np.load(SEAL / "ref_idx.npy")] = True; seal[np.load(SEAL / "val_idx.npy")] = True
    per = N_DRAW // len(SOURCES); draw = []
    for s in SOURCES:
        cand = np.where((src_all == s) & (~seal))[0]
        assert cand.size >= per, f"{s}: {cand.size}<{per}"
        draw.append(rng.choice(cand, per, replace=False))
    draw = np.sort(np.concatenate(draw)); n = draw.shape[0]
    Xmm = np.load(POOL / "dino1536.f16.npy", mmap_mode="r")
    X = _norm(np.asarray(Xmm[draw], np.float32)); dsrc = src_all[draw]
    print(f"{n} rows drawn; candidate kNN maxk={MAXK} (IVF nprobe=96)", flush=True)
    d = X.shape[1]; nlist = 2048
    quant = faiss.IndexFlatIP(d); index = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(X); index.add(X); index.nprobe = 96
    S, I = index.search(X, MAXK)
    # enforce self-inclusive convention: ensure self at column 0 (dist 0)
    idx = np.empty((n, MAXK), np.int32); sim = np.empty((n, MAXK), np.float32)
    for i in range(n):
        row = I[i]; ss = S[i]
        if row[0] != i:                                   # move self to front if ANN dropped/reordered it
            pos = np.where(row == i)[0]
            order = np.concatenate([[pos[0]], np.delete(np.arange(MAXK), pos[0])]) if pos.size else np.arange(MAXK)
            row = row[order]; ss = ss[order]
            if row[0] != i: row[0] = i; ss[0] = 1.0        # self missing from ANN list -> insert
        idx[i] = row; sim[i] = ss
    dist = np.sqrt(np.clip(2.0 - 2.0 * sim, 0.0, None)).astype(np.float32)   # unit-sphere cosine->euclid; self=0
    dup = float((dist[:, 1] <= 1e-6).mean())               # first REAL neighbor tied at 0
    np.save(OUT / "knn_idx.npy", idx); np.save(OUT / "knn_dist.npy", dist)
    np.save(OUT / "substrate.f16.npy", X.astype(np.float16)); np.save(OUT / "draw_ids.npy", draw)
    np.save(OUT / "draw_source.npy", dsrc)
    man = {"schema": "card010-knn-2026-09-11", "n": n, "maxk_incl_self": MAXK, "real_neighbors": MAXK - 1,
           "seed": SEED, "self_inclusive": True, "duplicate_first_real_nn_frac": round(dup, 6),
           "draw_ids_hash": hashlib.sha256(draw.tobytes()).hexdigest()[:16],
           "per_source": {s: int((dsrc == s).sum()) for s in SOURCES},
           "note": "self-inclusive (col0=self,dist0); IVF nprobe96 candidate search; exact-recall audited in the estimator step."}
    (OC / "card010-knn-manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
