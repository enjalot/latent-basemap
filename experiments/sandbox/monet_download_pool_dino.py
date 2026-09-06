"""MONET 19.34M-pool DINOv2-vitg14 column pull, F16 (owner 2026-09-06). CPU/network, zero GPU. Mirrors
monet_download_pool_clip.py's prov-scatter, but the dino (1536-d) column, stored f16 into pool-20m/dino1536.f16.npy
row-aligned to clip512 (same prov). With the complement dino this completes the full-corpus DINO column for the
full-set projections (item 3). ~59 GB. Resumable per shard; handles the JSON-string-encoded embedding variant.
Usage: monet_download_pool_dino.py [WORKERS=3].
"""
import json, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, fsspec

REPO = "jasperai/monet"; OUT = Path("/data2/monet/pool-20m"); DINO_COL = "embedding_dinov2-vitg14"; DIM = 1536


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    man = json.loads((OUT / "manifest.json").read_text()); shards = man["shards"]
    shard_idx = np.load(OUT / "prov_shard_idx.npy"); local_row = np.load(OUT / "prov_local_row.npy")
    N = shard_idx.shape[0]
    print(f"pool N={N:,}, {len(shards)} shards -> DINO-1536 f16 (~{N*DIM*2/1e9:.1f} GB)", flush=True)
    dp = OUT / "dino1536.f16.npy"
    dino = np.lib.format.open_memmap(dp, mode="r+" if dp.exists() else "w+", dtype=np.float16, shape=(N, DIM))
    done_path = OUT / "dino_done_shards.json"
    done = set(json.loads(done_path.read_text())) if done_path.exists() else set()
    order = np.argsort(shard_idx, kind="stable"); si_sorted = shard_idx[order]
    starts = np.searchsorted(si_sorted, np.arange(len(shards)), side="left")
    ends = np.searchsorted(si_sorted, np.arange(len(shards)), side="right")
    boundaries = {si: order[starts[si]:ends[si]] for si in range(len(shards))}
    fs = fsspec.filesystem("hf")

    def _fetch(si):
        pr = boundaries[si]
        if len(pr) == 0:
            return si, 0
        with fs.open(f"datasets/{REPO}/{shards[si]}", "rb") as fh:
            col = pq.read_table(fh, columns=[DINO_COL])[DINO_COL].to_pylist()
        if col and isinstance(col[0], str):
            import json as _json; col = [_json.loads(c) for c in col]
        arr = np.asarray(col, dtype=np.float16)
        dino[pr] = arr[local_row[pr]]                 # disjoint rows -> thread-safe
        return si, len(pr)

    todo = [si for si in range(len(shards)) if si not in done]
    print(f"{len(todo)} shards to fetch ({len(done)} done), {workers} workers", flush=True)
    n = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch, si): si for si in todo}
        for fut in as_completed(futs):
            si, cnt = fut.result(); done.add(si); n += 1
            if n % 100 == 0:
                dino.flush(); done_path.write_text(json.dumps(sorted(done)))
                import shutil; free = shutil.disk_usage("/data2").free / 1e9
                print(f"  {n}/{len(todo)} shards  /data2 free {free:.0f}GB", flush=True)
    dino.flush(); done_path.write_text(json.dumps(sorted(done)))
    (OUT / "dino1536-manifest.json").write_text(json.dumps({
        "schema": "monet-pool-dino1536-f16-2026-09-06", "n_rows": int(N), "dim": DIM, "dtype": "float16",
        "row_aligned_to": "pool-20m clip512 via prov_shard_idx+prov_local_row",
        "cast": "f64->f16; head casts f16->f32 at projection", "n_shards": len(shards)}, indent=1))
    print(f"POOL DINO DONE: {N:,}x{DIM} f16 -> {dp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
