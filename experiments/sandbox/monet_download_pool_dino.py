"""MONET 19.34M-pool DINOv2-vitg14 column pull, F16 (owner 2026-09-06). CPU/network, zero GPU. Mirrors
monet_download_pool_clip.py's prov-scatter, but the dino (1536-d) column, stored f16 into pool-20m/dino1536.f16.npy
row-aligned to clip512 (same prov). With the complement dino this completes the full-corpus DINO column for the
full-set projections (item 3). ~59 GB. Resumable per shard; handles the JSON-string-encoded embedding variant.
Usage: monet_download_pool_dino.py [WORKERS=3].
"""
import json, sys, gc
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
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

    def _fetch(si):                                    # OOM FIX: fetch+gather in the worker, WRITE in main (so the
        pr = boundaries[si]                            # main loop can safely close+reopen the memmap between writes)
        if len(pr) == 0:
            return si, pr, None
        with fs.open(f"datasets/{REPO}/{shards[si]}", "rb") as fh:
            col = pq.read_table(fh, columns=[DINO_COL])[DINO_COL].to_pylist()
        if col and isinstance(col[0], str):
            import json as _json; col = [_json.loads(c) for c in col]
        arr = np.asarray(col, dtype=np.float16); del col
        g = arr[local_row[pr]]; del arr                # gather this shard's pool rows
        return si, pr, g

    todo = [si for si in range(len(shards)) if si not in done]
    print(f"{len(todo)} shards to fetch ({len(done)} done), {workers} workers", flush=True)
    dp = OUT / "dino1536.f16.npy"
    # OOM FIX (2026-09-06): bounded in-flight submission (never submit all upfront — retains every completed future's
    # gathered array) + close+reopen the memmap at each checkpoint (releases dirty pages). Mirror of the complement fix.
    n = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        it = iter(todo); inflight = {}
        for _ in range(min(2 * workers, len(todo))):
            try:
                inflight[ex.submit(_fetch, next(it))] = 1
            except StopIteration:
                break
        while inflight:
            dfs, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in dfs:
                del inflight[fut]
                try:
                    inflight[ex.submit(_fetch, next(it))] = 1
                except StopIteration:
                    pass
                rsi, pr, g = fut.result()
                if g is not None:
                    dino[pr] = g                       # disjoint rows; main-thread write
                del g, fut
                done.add(rsi); n += 1
                if n % 100 == 0:
                    dino.flush(); del dino; gc.collect()
                    dino = np.lib.format.open_memmap(dp, mode="r+", dtype=np.float16, shape=(N, DIM))
                    done_path.write_text(json.dumps(sorted(done)))
                    import shutil
                    free = shutil.disk_usage("/data2").free / 1e9
                    try:
                        rss = int([l.split()[1] for l in open("/proc/self/status") if l.startswith("VmRSS")][0]) / 1e6
                    except Exception:
                        rss = -1.0
                    print(f"  {n}/{len(todo)} shards  /data2 free {free:.0f}GB  RSS {rss:.1f}GB", flush=True)
    dino.flush(); done_path.write_text(json.dumps(sorted(done)))
    (OUT / "dino1536-manifest.json").write_text(json.dumps({
        "schema": "monet-pool-dino1536-f16-2026-09-06", "n_rows": int(N), "dim": DIM, "dtype": "float16",
        "row_aligned_to": "pool-20m clip512 via prov_shard_idx+prov_local_row",
        "cast": "f64->f16; head casts f16->f32 at projection", "n_shards": len(shards)}, indent=1))
    print(f"POOL DINO DONE: {N:,}x{DIM} f16 -> {dp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
