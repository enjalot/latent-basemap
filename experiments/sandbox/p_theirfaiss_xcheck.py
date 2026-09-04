"""theirfaiss GPU-vs-CPU cross-check (owner MONET option A gate, overseer 2026-09-04). CPU faiss searches THEIR
index on the SAME ~200K sample rows the GPU used; Spearman GPU-vs-CPU rarity must be >=0.99 (same index+nprobe
=> near-identical; GPU IVFPQ has minor float-accum/table-precision diffs). EXIT NONZERO if below. .venv."""
import json
from pathlib import Path
import numpy as np

THEIR = "/data2/monet/retrieval-storage/clip/embedding_clip-vit-base-patch32.faiss"
POOL = Path("/data2/monet/pool-20m"); OUT = Path("/data2/monet/draws"); K = 16


def main():
    import faiss
    from scipy.stats import spearmanr
    faiss.omp_set_num_threads(16)
    samp = np.load(OUT / "theirfaiss_xcheck_idx.npy")
    gpu = np.load(OUT / "rarity_theirfaiss_gpu_sample.npy")
    clip = np.load(POOL / "clip512.f32.npy", mmap_mode="r")
    idx = faiss.read_index(THEIR); idx.nprobe = 64
    D, _ = idx.search(np.ascontiguousarray(clip[samp], np.float32), K + 1)
    D.sort(axis=1); cpu = D[:, 1:K+1].mean(1)
    rho = float(spearmanr(gpu, cpu).statistic); ok = rho >= 0.99
    out = {"arm": "theirfaiss", "n_sample": int(len(samp)), "spearman_gpu_vs_cpu": round(rho, 5),
           "gate": ">=0.99", "PASS": bool(ok),
           "verdict": "GPU search of their index agrees with CPU (same index/nprobe) — draw approved, hardware not part of arm identity"
                      if ok else "STOP: GPU-vs-CPU rarity diverges (<0.99) — faiss GPU IVFPQ precision issue, investigate before drawing"}
    (OUT / "theirfaiss-gpu-xcheck.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out), flush=True)
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
