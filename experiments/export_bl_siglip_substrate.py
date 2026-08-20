#!/usr/bin/env python3
"""Export the BL SigLIP2 parquet embeddings to a training substrate (CPU-only).

Prepared 2026-08-20 for the owner GPU window (plan §4A). Reads the four
subset parquet dirs under /data/images/british-library-book-images/
siglip2-embeddings/ and writes:

  /data/latent-basemap/substrates/bl-siglip2-1m/
      substrate.f16.npy   # (N, 1152) float16, subset-ordered
      rows.parquet        # row-parallel: fname, date, image_type, subset,
                          #   source_filename, file_row_number
      corpus_ids.json     # subset -> corpus code (overlay planes)
      manifest.json

Row order is covers, medium, embellishments, plates (each in filename-sorted
parquet order) — deterministic and re-derivable. ~2.4 GB fp16.
"""
import glob
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq

SRC = "/data/images/british-library-book-images/siglip2-embeddings"
OUT = "/data/latent-basemap/substrates/bl-siglip2-1m"
SUBSETS = ("covers", "medium", "embellishments", "plates")
DIM = 1152


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    out_npy = os.path.join(OUT, "substrate.f16.npy")
    if os.path.exists(out_npy):
        print(f"{out_npy} exists; refusing to overwrite")
        return 1

    files = []
    for sub in SUBSETS:
        fs = sorted(glob.glob(f"{SRC}/{sub}/*.parquet"))
        assert fs, f"no parquet under {SRC}/{sub}"
        files += [(sub, f) for f in fs]
    total = sum(pq.ParquetFile(f).metadata.num_rows for _, f in files)
    print(f"{len(files)} files, {total:,} rows -> {out_npy}")

    arr = np.lib.format.open_memmap(
        out_npy + ".tmp.npy", mode="w+", dtype=np.float16, shape=(total, DIM))
    metas = []
    row = 0
    t0 = time.time()
    for sub, f in files:
        pf = pq.ParquetFile(f)
        for b in pf.iter_batches(batch_size=16384):
            d = b.to_pydict()
            emb = np.asarray(d.pop("embedding"), dtype=np.float32)
            assert emb.shape[1] == DIM, emb.shape
            n = emb.shape[0]
            arr[row:row + n] = emb.astype(np.float16)
            d["subset"] = [sub] * n
            metas.append(d)
            row += n
        print(f"  {sub}/{os.path.basename(f)} -> rows {row:,}", flush=True)
    assert row == total, (row, total)
    arr.flush()
    del arr
    os.rename(out_npy + ".tmp.npy", out_npy)

    import pyarrow as pa
    tables = [pa.table({k: v for k, v in m.items()}) for m in metas]
    rows_table = pa.concat_tables(tables)
    pq.write_table(rows_table, os.path.join(OUT, "rows.parquet"))

    corpus_ids = {sub: i for i, sub in enumerate(SUBSETS)}
    with open(os.path.join(OUT, "corpus_ids.json"), "w") as fh:
        json.dump(corpus_ids, fh, indent=1)
    manifest = {
        "source": SRC,
        "model": "siglip2 (as shipped with biglam/british-library-book-images)",
        "dim": DIM,
        "dtype": "float16",
        "rows": total,
        "row_order": "covers, medium, embellishments, plates; filename-sorted "
                     "parquets; batch order within file",
        "wall_s": time.time() - t0,
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"DONE {total:,} rows in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
