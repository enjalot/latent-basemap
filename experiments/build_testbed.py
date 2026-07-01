"""Build a training testbed: sample N rows from an embedding shard dir,
materialize as a single fp32 .npy on /data, and build exact kNN edge .npz
files (GPU torch chunked matmul on normalized vectors = cosine) at several k.

Usage:
  python experiments/build_testbed.py \
      --src /data/embeddings/fineweb-edu-sample-10BT-chunked-500-jina-v5-nano/train \
      --out /data/latent-basemap/jina-en-200k --n 200000 --ks 15,30,50 --seed 42

Outputs under --out:
  train/data-00000.npy   (N, d) float32
  sample_indices.npy     (N,) int64 row indices into the concatenated source
  edges_k{K}.npz         sources/targets int32, weights=1/K float32, n_nodes, k
"""
import argparse
import os
import time

import numpy as np


def load_sample(src_dir, n, seed):
    shards = sorted(
        os.path.join(src_dir, f) for f in os.listdir(src_dir)
        if f.endswith(".npy")
    )
    sizes, dims = [], set()
    for s in shards:
        m = np.load(s, mmap_mode="r")
        sizes.append(m.shape[0])
        dims.add(m.shape[1:])
    (dim,) = {d[0] for d in dims}
    total = int(np.sum(sizes))
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(total, size=n, replace=False))
    print(f"{len(shards)} shards, {total:,} rows, d={dim}; sampling {n:,}")

    out = np.empty((n, dim), dtype=np.float32)
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    for si, s in enumerate(shards):
        lo, hi = offsets[si], offsets[si + 1]
        mask = (idx >= lo) & (idx < hi)
        if not mask.any():
            continue
        local = idx[mask] - lo
        m = np.load(s, mmap_mode="r")
        # gather in modest blocks to keep fp32 casts bounded
        rows = np.where(mask)[0]
        for b in range(0, len(local), 65536):
            out[rows[b:b + 65536]] = m[local[b:b + 65536]].astype(np.float32)
    return out, idx


def build_edges(X, ks, device="cuda", chunk=4096):
    import torch
    kmax = max(ks) + 1  # +1 for self
    n = X.shape[0]
    Xt = torch.from_numpy(X).to(device).half()
    Xt = torch.nn.functional.normalize(Xt, dim=1)
    nbrs = np.empty((n, kmax - 1), dtype=np.int32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, chunk):
            sims = Xt[i:i + chunk] @ Xt.T
            _, topi = torch.topk(sims, kmax, dim=1)
            topi = topi.cpu().numpy()
            # drop self wherever it appears in the top-kmax
            row = np.arange(i, i + topi.shape[0])[:, None]
            keep = topi != row
            # keep first kmax-1 non-self per row
            cleaned = np.empty((topi.shape[0], kmax - 1), dtype=np.int32)
            for r in range(topi.shape[0]):
                cleaned[r] = topi[r][keep[r]][: kmax - 1]
            nbrs[i:i + topi.shape[0]] = cleaned
    print(f"kNN (kmax={kmax - 1}) in {time.time() - t0:.1f}s")
    return nbrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--ks", default="15,30,50")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    ks = [int(k) for k in args.ks.split(",")]

    os.makedirs(os.path.join(args.out, "train"), exist_ok=True)
    X, idx = load_sample(args.src, args.n, args.seed)
    np.save(os.path.join(args.out, "sample_indices.npy"), idx)
    np.save(os.path.join(args.out, "train", "data-00000.npy"), X)

    nbrs = build_edges(X, ks, device=args.device)
    n = X.shape[0]
    for k in ks:
        sources = np.repeat(np.arange(n, dtype=np.int32), k)
        targets = nbrs[:, :k].reshape(-1).astype(np.int32)
        weights = np.full(n * k, 1.0 / k, dtype=np.float32)
        path = os.path.join(args.out, f"edges_k{k}.npz")
        np.savez(path, sources=sources, targets=targets, weights=weights,
                 n_nodes=n, k=k)
        print(f"wrote {path} ({n * k:,} edges)")


if __name__ == "__main__":
    main()
