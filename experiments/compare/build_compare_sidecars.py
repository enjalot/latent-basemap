#!/usr/bin/env python3
"""Build O(1) text sidecars for the compare-page content server.

For each listed chunk dir: stream every parquet's chunk_text in global row
order, truncate to 800 chars, and write offsets.u64 (N+1 x u64) + blob.utf8
under /data/latent-basemap/textsidecar/compare/<dir>/. The server prefers
these over parquet row-group decompression (content_server.compare_sidecar).

Streaming batches only — nothing large resident. Resumable per dir (skips
dirs whose offsets.u64 already exists). CPU/IO-only.
"""
from __future__ import annotations

import glob
import sys
import time
from pathlib import Path

import numpy as np

CHUNKS = Path("/data/chunks")
OUT = Path("/data/latent-basemap/textsidecar/compare")
TRUNC = 800

DIRS = [
    "communityarchive-tweets",
    "reddit-tldr17-chunked-120",
    "fineweb-edu-sample-10BT-chunked-500",
    "RedPajama-Data-V2-sample-10B-chunked-500",
    "pile-uncopyrighted-chunked-500",
] + [f"fineweb2-{l}-chunked-500" for l in (
    "arb_Arab", "ces_Latn", "cmn_Hani", "deu_Latn", "ell_Grek", "fra_Latn",
    "hin_Deva", "ind_Latn", "ita_Latn", "jpn_Jpan", "kor_Hang", "nld_Latn",
    "pol_Latn", "por_Latn", "rus_Cyrl", "spa_Latn", "swe_Latn", "tha_Thai",
    "tur_Latn", "vie_Latn")]

#: only the prefix of rows any resolver can reach (saves ~10x on the big
#: corpora): stride-resolvers touch stride*N rows, span-resolvers their span.
ROW_CAP = {
    "communityarchive-tweets": None,          # 8*i over full corpus
    "reddit-tldr17-chunked-120": None,        # 5*i over full corpus
    "fineweb-edu-sample-10BT-chunked-500": 333_334,
    "RedPajama-Data-V2-sample-10B-chunked-500": 333_333,
    "pile-uncopyrighted-chunked-500": 333_333,
}  # fineweb2 dirs: capped to 106_250 (jina-multi-2m 50K + 6m topup headroom)


def build(d: str) -> None:
    import pyarrow.parquet as pq
    out = OUT / d
    if (out / "offsets.u64").exists():
        print(f"{d}: exists, skip", flush=True)
        return
    cap = ROW_CAP.get(d, 106_250 if d.startswith("fineweb2-") else None)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    offs = [0]
    n = 0
    with open(out / "blob.tmp", "wb") as blob:
        pos = 0
        for f in sorted(glob.glob(f"{CHUNKS}/{d}/train/*.parquet")):
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=32_768,
                                         columns=["chunk_text"]):
                for t in batch.column(0).to_pylist():
                    b = (t or "")[:TRUNC].encode("utf-8", "replace")
                    blob.write(b)
                    pos += len(b)
                    offs.append(pos)
                    n += 1
                    if cap and n >= cap:
                        break
                if cap and n >= cap:
                    break
            if cap and n >= cap:
                break
    np.asarray(offs, dtype="<u8").tofile(out / "offsets.tmp")
    (out / "blob.tmp").rename(out / "blob.utf8")
    (out / "offsets.tmp").rename(out / "offsets.u64")
    mb = pos / 1e6
    print(f"{d}: {n:,} rows, {mb:.0f}MB, {time.time()-t0:.0f}s", flush=True)


def main() -> int:
    only = sys.argv[1:] or DIRS
    for d in only:
        build(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
