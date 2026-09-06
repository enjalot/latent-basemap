"""PCA-768 of the 6M DINO draw (owner residency experiment via overseer 2026-09-06). GPU (exact PCA via batched
mean + covariance + eigh — 1536x1536 cov is tiny). Fits on the 6M draw ONLY, SAVES components+mean (a projection
head must apply this preprocessing at inference), transforms all 6M -> 768, RENORMS to unit length.

The residency question: 30M-DINO int8 is 46GB at 1536 (breaks 32GB device residency) vs 23GB at 768 (fits). This
measures what halving the dim costs in v2 FFR + held-out reception at 6M.

Reads /data2/monet/random-dino-6m/dino-substrate.f16.npy (6M x1536, unit-norm). Writes pca768-substrate.f32.npy
(6M x768, renormed) + pca768-model.npz (components 1536x768, mean 1536) + pca768-manifest.json.
Usage: exp_dino_6m_pca.py
"""
import json, time
from pathlib import Path
import numpy as np

D = Path("/data2/monet/random-dino-6m")
SUB = D / "dino-substrate.f16.npy"
DIM_IN, DIM_OUT, BATCH = 1536, 768, 100_000


def main():
    import torch
    x = np.load(SUB, mmap_mode="r"); n = x.shape[0]
    print(f"[pca768] fit PCA on {n:,}x{DIM_IN} -> {DIM_OUT}", flush=True)
    dev = "cuda"
    # pass 1: mean + covariance (sum X, sum X^T X), batched
    t0 = time.time()
    ssum = torch.zeros(DIM_IN, dtype=torch.float64, device=dev)
    cov = torch.zeros(DIM_IN, DIM_IN, dtype=torch.float64, device=dev)
    for i in range(0, n, BATCH):
        b = torch.from_numpy(np.asarray(x[i:i + BATCH], np.float32)).to(dev).double()
        ssum += b.sum(0); cov += b.T @ b
        del b
    mean = (ssum / n)
    cov = cov / n - torch.outer(mean, mean)                       # covariance of the (already unit-norm) rows
    cov = (cov + cov.T) / 2                                        # symmetrize
    evals, evecs = torch.linalg.eigh(cov)                         # ascending
    comp = evecs[:, -DIM_OUT:].flip(1)                            # top-768 components, descending (1536x768)
    ev_top = evals.flip(0)[:DIM_OUT]
    evr = float((ev_top.sum() / evals.sum()).item())              # explained-variance ratio at 768
    print(f"[pca768] fit {time.time()-t0:.0f}s | explained_variance_ratio@768 = {evr:.4f}", flush=True)

    meanc = mean.float(); compc = comp.float()
    out = np.lib.format.open_memmap(D / "pca768-substrate.f32.npy", mode="w+", dtype=np.float32, shape=(n, DIM_OUT))
    for i in range(0, n, BATCH):
        b = torch.from_numpy(np.asarray(x[i:i + BATCH], np.float32)).to(dev)
        y = (b - meanc) @ compc                                   # project
        y = torch.nn.functional.normalize(y, dim=1)               # RENORM to unit length (matches head training)
        out[i:i + b.shape[0]] = y.cpu().numpy().astype(np.float32)
        del b, y
    out.flush()
    np.savez(D / "pca768-model.npz", components=compc.cpu().numpy(), mean=meanc.cpu().numpy())
    (D / "pca768-manifest.json").write_text(json.dumps({
        "schema": "monet-dino-6m-pca768-2026-09-06", "n_rows": int(n), "dim_in": DIM_IN, "dim_out": DIM_OUT,
        "explained_variance_ratio_at_768": round(evr, 4), "fit_on": "the 6M draw only",
        "preprocessing": "y = normalize((x - mean) @ components); components 1536x768, mean 1536 in pca768-model.npz — a projection head MUST apply this at inference",
        "renormed": True}, indent=1))
    print(f"[pca768] DONE {n:,}x{DIM_OUT} renormed -> {D/'pca768-substrate.f32.npy'} (EVR {evr:.4f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
