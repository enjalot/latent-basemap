#!/usr/bin/env python
"""End-to-end positive control for the text sidecar, on real data.

Nothing in the pack pipeline proves that sidecar row i is the text that produced
substrate row i — the (corpus, shard, row) resolution and the parquet/npy
row alignment are both inferred. This script closes that loop the only way that
is really convincing: take a sample of substrate rows, pull their text out of
the sidecar, re-embed it with the same model the corpus was embedded with
(all-MiniLM-L6-v2, CPU), and compare against the sealed substrate vector.

A correct sidecar gives cosine ~1.0 for every sampled row. A shard-off-by-one or
a corpus mix-up gives ~0.0-0.3.

    CUDA_VISIBLE_DEVICES= python verify_text_alignment.py \
        --substrate-dir <dir> --sidecar <dir> [--n 64]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main() -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate-dir", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--n", type=int, default=64, help="rows sampled per corpus")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    sub_dir = Path(a.substrate_dir)
    side = Path(a.sidecar)
    prov = np.load(sub_dir / "provenance.npy", mmap_mode="r")
    vecs = np.load(sub_dir / "substrate.f32.npy", mmap_mode="r")
    offsets = np.memmap(side / "offsets.u64", dtype="<u8", mode="r")
    blob = np.memmap(side / "blob.utf8", dtype=np.uint8, mode="r")
    cmap = json.loads((side / "manifest.json").read_text())["corpus_codes"]

    rng = np.random.default_rng(a.seed)
    codes = np.asarray(prov["corpus"])
    rows = []
    for c in sorted(int(k) for k in cmap):
        pool = np.flatnonzero(codes == c)
        rows.extend(rng.choice(pool, size=min(a.n, pool.size), replace=False).tolist())
    rows = sorted(int(r) for r in rows)

    texts = [bytes(blob[int(offsets[r]):int(offsets[r + 1])]).decode("utf-8") for r in rows]
    empty = [r for r, t in zip(rows, texts) if not t]
    if empty:
        print(f"WARNING: {len(empty)} sampled rows have empty text")

    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(8)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    emb = model.encode(texts, batch_size=32, normalize_embeddings=True,
                       show_progress_bar=False)

    ref = np.asarray(vecs[rows], dtype=np.float32)
    ref /= np.linalg.norm(ref, axis=1, keepdims=True) + 1e-12
    cos = (emb * ref).sum(axis=1)

    print(f"{'corpus':<58} {'n':>4} {'min':>7} {'median':>7} {'frac>0.99':>10}")
    overall_ok = True
    for c in sorted(int(k) for k in cmap):
        m = np.asarray(prov["corpus"])[rows] == c
        if not m.any():
            continue
        sub = cos[m]
        ok = float((sub > 0.99).mean())
        overall_ok &= ok >= 0.95
        print(f"{cmap[str(c)]:<58} {m.sum():>4} {sub.min():>7.4f} "
              f"{np.median(sub):>7.4f} {ok:>10.3f}")
    print(f"\nALL: n={len(rows)} min={cos.min():.4f} median={np.median(cos):.4f} "
          f"frac>0.99={(cos > 0.99).mean():.3f}")
    print("PASS" if overall_ok else "FAIL")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
