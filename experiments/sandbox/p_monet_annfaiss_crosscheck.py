"""Cross-check the cuVS annfaiss rarity vs CPU faiss IVF-Flat on the SAME 500K sample (owner MONET gate).
.venv (faiss-cpu + scipy). Builds a CPU IVF-Flat over the sample's normalized CLIP-512, mean-kNN-dist rarity,
Spearman vs rarity_cuvs_sample.npy. Writes annfaiss-crosscheck.json; EXITS NONZERO if Spearman < 0.95 (gate:
stop before drawing). Usage: p_monet_annfaiss_crosscheck.py [K=16]."""
import sys, json
from pathlib import Path
import numpy as np

POOL = Path("/data2/monet/pool-20m"); OUT = Path("/data2/monet/draws")


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    import faiss
    from scipy.stats import spearmanr
    faiss.omp_set_num_threads(12)
    samp = np.load(OUT / "annfaiss_sample_idx.npy")
    cuvs = np.load(OUT / "rarity_cuvs_sample.npy")
    clip = np.load(POOL / "clip512.f32.npy", mmap_mode="r")
    xb = np.ascontiguousarray(clip[samp]).astype(np.float32)
    xb /= np.linalg.norm(xb, axis=1, keepdims=True).clip(1e-9)
    d = xb.shape[1]; nlist = 1024
    idx = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
    idx.train(xb); idx.add(xb); idx.nprobe = 32
    D, _ = idx.search(xb, k + 1)
    cpu = (1.0 - D[:, 1:].mean(1)).astype(np.float32)          # low mean-sim = rare (same sense as cuVS dist)
    rho = float(spearmanr(cuvs, cpu).statistic)
    ok = rho >= 0.95
    out = {"arm": "annfaiss", "n_sample": int(len(samp)), "spearman_cuvs_vs_cpu": round(rho, 4),
           "gate": ">=0.95", "PASS": bool(ok),
           "verdict": "cuVS annfaiss rarity method-consistent with CPU IVF-Flat — draw approved"
                      if ok else "STOP: cuVS vs CPU rarity ranking diverges (Spearman<0.95) — tell overseer before drawing"}
    (OUT / "annfaiss-crosscheck.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out), flush=True)
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
