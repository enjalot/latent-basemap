"""Full-corpus DINOv2-vitg14 column download — complement (owner 2026-09-06). CPU/network only, zero GPU.
Downloads the embedding_dinov2-vitg14 (1536-d) column for the 84.47M complement, stored as F16 (259.5 GB;
f32's 519 GB won't fit — the head casts f16->f32 at projection with nil quality impact, recorded in the manifest).

DISK-SAFE: streams DIRECTLY into a preallocated f16 memmap using the complement's EXISTING offsets.npy row
layout — NO per-shard files (avoids the ~2x footprint that would breach the 300 GB headroom flag on the current
684 GB-free /data2). Row-aligned to pool-complement-88m/clip512.f32.npy (same shard order / offsets), so
dino1536.f16.npy[i] pairs with clip512[i]. Resumable per shard; per-1000-shard checkpoints; authenticated.

Handles the JSON-string-encoded embedding variant (same as the CLIP complement's shard 6552).
Usage: monet_download_dino.py [WORKERS=5].
"""
import json, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import fsspec

REPO = "jasperai/monet"
OUT = Path("/data2/monet/pool-complement-88m")
DINO_COL = "embedding_dinov2-vitg14"
DIM = 1536
HF_RETRIES = 6
CKPT_EVERY = 1000


def _fetch(gi, path, lo, n_expected):
    fs = fsspec.filesystem("hf"); last = None
    for attempt in range(HF_RETRIES):
        try:
            with fs.open(f"datasets/{REPO}/{path}", "rb") as fh:
                t = pq.read_table(fh, columns=[DINO_COL])
            break
        except Exception as e:
            last = e
            if attempt == HF_RETRIES - 1:
                raise RuntimeError(f"{path}: read failed after {HF_RETRIES}: {last}")
            time.sleep(min(120, 3 * 2 ** attempt))
    col = t[DINO_COL].to_pylist()
    if col and isinstance(col[0], str):
        import json as _json
        col = [_json.loads(c) for c in col]
    d = np.asarray(col, dtype=np.float16)
    if d.ndim != 2 or d.shape[1] != DIM:
        raise RuntimeError(f"{path}: dino shape {d.shape}")
    if d.shape[0] != n_expected:
        raise RuntimeError(f"{path}: row drift {n_expected} vs {d.shape[0]}")
    return gi, lo, d


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    full = json.loads((OUT / "full_shards.json").read_text())
    shuffle = full["shards"]; n_pool = full["n_pool"]
    complement = shuffle[n_pool:]
    offsets = np.load(OUT / "offsets.npy"); N = int(offsets[-1])
    print(f"[dino] complement {len(complement)} shards, N={N:,}, dino f16 {N*DIM*2/1e9:.1f} GB", flush=True)

    dpath = OUT / "dino1536.f16.npy"
    dino = np.lib.format.open_memmap(dpath, mode="r+" if dpath.exists() else "w+", dtype=np.float16, shape=(N, DIM))
    done_path = OUT / "dino_done_shards.json"
    done = set(json.loads(done_path.read_text())) if done_path.exists() else set()
    todo = [(n_pool + j, complement[j], int(offsets[j]), int(offsets[j + 1] - offsets[j]))
            for j in range(len(complement)) if (n_pool + j) not in done]
    print(f"[dino] {len(todo)} shards to fetch ({len(done)} done), {workers} workers", flush=True)

    t0 = time.time(); n_run = 0; errors = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch, gi, p, lo, n): gi for (gi, p, lo, n) in todo}
        for fut in as_completed(futs):
            gi = futs[fut]
            try:
                gi_, lo, d = fut.result()
            except Exception as e:
                errors.append(f"{gi}: {e}"); print(f"[dino] FAILED shard {gi}: {e}", flush=True); continue
            dino[lo:lo + d.shape[0]] = d
            done.add(gi); n_run += 1
            if n_run % CKPT_EVERY == 0 or n_run == len(todo):
                dino.flush(); done_path.write_text(json.dumps(sorted(done)))
                el = time.time() - t0; rate = n_run / el if el else 0
                import shutil
                free_gb = shutil.disk_usage("/data2").free / 1e9
                print(f"[dino] CKPT {len(done)}/{len(complement)} shards  {rate*60:.1f} sh/min  "
                      f"eta~{(len(todo)-n_run)/rate/3600:.1f}h  errors={len(errors)}  /data2 free {free_gb:.0f}GB", flush=True)
                if free_gb < 300:
                    print(f"[dino] WARNING /data2 free {free_gb:.0f}GB < 300GB flag", flush=True)
    dino.flush(); done_path.write_text(json.dumps(sorted(done)))
    manifest = {"schema": "monet-complement-dino1536-f16-2026-09-06", "col": DINO_COL, "dim": DIM,
                "dtype": "float16", "n_rows": N, "cast": "f64->f16 at download; head casts f16->f32 at projection (nil quality impact)",
                "row_aligned_to": "pool-complement-88m/clip512.f32.npy (same offsets/shard order)",
                "n_shards_done": len(done), "errors": errors}
    (OUT / "dino1536-manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[dino] DONE {len(done)}/{len(complement)} shards, {len(errors)} errors -> {dpath}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
