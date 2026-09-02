"""MONET 20M-pool CLIP-512 pull (owner MONET diversity, option A, overseer 2026-09-02). CPU/network only.
Pulls embedding_clip-vit-base-patch32 (512-d) for the FULL 20M light pool so diverse-annfaiss can build a
kNN density index over the same universe the random/diverse-sscd arms draw from (tests the owner's thesis:
a customer index over the whole corpus drives the draw). ~41 GB f32 on /data2.

ROW ALIGNMENT: the pool was read in parallel (completion order), so pool row i came from shards[prov_shard_
idx[i]] at prov_local_row[i]. We scatter each shard's CLIP into clip512[pool_rows_of_shard] via that
provenance -> clip512 is exactly row-aligned to sscd_nn.npy / id.npy etc. Resumable per shard.
Usage: monet_download_pool_clip.py [WORKERS=16]."""
import json, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import fsspec

REPO = "jasperai/monet"
OUT = Path("/data2/monet/pool-20m")
CLIP_COL = "embedding_clip-vit-base-patch32"


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    man = json.loads((OUT / "manifest.json").read_text())
    shards = man["shards"]
    shard_idx = np.load(OUT / "prov_shard_idx.npy")
    local_row = np.load(OUT / "prov_local_row.npy")
    N = shard_idx.shape[0]
    print(f"pool N={N:,}, {len(shards)} shards -> CLIP-512 (~{N*512*4/1e9:.1f} GB)", flush=True)

    clip = np.lib.format.open_memmap(OUT / "clip512.f32.npy", mode="r+" if (OUT / "clip512.f32.npy").exists()
                                     else "w+", dtype=np.float32, shape=(N, 512))
    done_path = OUT / "clip_done_shards.json"
    done = set(json.loads(done_path.read_text())) if done_path.exists() else set()
    # precompute pool-row indices per shard
    order = np.argsort(shard_idx, kind="stable")
    boundaries = {}
    si_sorted = shard_idx[order]
    starts = np.searchsorted(si_sorted, np.arange(len(shards)), side="left")
    ends = np.searchsorted(si_sorted, np.arange(len(shards)), side="right")
    for si in range(len(shards)):
        boundaries[si] = order[starts[si]:ends[si]]   # pool rows belonging to shard si

    fs = fsspec.filesystem("hf")

    def _fetch(si):
        pool_rows = boundaries[si]
        if len(pool_rows) == 0:
            return si, 0
        with fs.open(f"datasets/{REPO}/{shards[si]}", "rb") as fh:
            col = pq.read_table(fh, columns=[CLIP_COL])[CLIP_COL].to_pylist()
        arr = np.asarray(col, dtype=np.float32)
        clip[pool_rows] = arr[local_row[pool_rows]]   # disjoint rows -> thread-safe
        return si, len(pool_rows)

    todo = [si for si in range(len(shards)) if si not in done]
    print(f"{len(todo)} shards to fetch ({len(done)} already done)", flush=True)
    n = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch, si): si for si in todo}
        for fut in as_completed(futs):
            si, cnt = fut.result(); done.add(si); n += 1
            if n % 50 == 0:
                clip.flush(); done_path.write_text(json.dumps(sorted(done)))
                print(f"  {n}/{len(todo)} shards ({cnt} rows last)", flush=True)
    clip.flush(); done_path.write_text(json.dumps(sorted(done)))
    # sanity: no all-zero rows (unfilled)
    sample = clip[np.random.default_rng(0).choice(N, min(100000, N), replace=False)]
    zero_frac = float((np.linalg.norm(sample, axis=1) == 0).mean())
    (OUT / "clip512-manifest.json").write_text(json.dumps({
        "schema": "monet-pool-clip512-2026-09-02", "n_rows": int(N), "dim": 512, "dtype": "float32",
        "row_aligned_to": "pool-20m (sscd_nn.npy/id.npy) via prov_shard_idx+prov_local_row",
        "zero_row_frac_sampled": zero_frac, "n_shards": len(shards)}, indent=1))
    print(f"CLIP POOL DONE: {N:,}x512 -> {OUT/'clip512.f32.npy'}  zero-row frac {zero_frac:.5f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
