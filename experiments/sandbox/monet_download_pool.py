"""MONET 20M LIGHT candidate pool (owner MONET diversity draw, overseer 2026-09-02). CPU/network only.
Pulls ONLY light columns (id, source, sscd_nn, sscd_cluster_id, aesthetic) — NO embeddings — for a 20M-row
pool, the universe the random / diverse-sscd / diverse-annfaiss draws select from. Uses the SAME seed-42
shard shuffle as monet_download_sample.py, so the random-2m embedding sample is a subset of this pool.
Records per-row provenance (source shard + local row) so embeddings for the SELECTED UNION can be pulled
later. Parallel shard reads.

sscd_nn DIRECTION (empirically confirmed 2026-09-02): it is a NN SIMILARITY — HIGH = near-duplicate,
LOW = rare/diverse (corr with in-shard NN cosine = +0.40). Diversity weighting must UPWEIGHT LOW sscd_nn.
Usage: monet_download_pool.py [N_TARGET=20000000] [SEED=42] [WORKERS=16]."""
import json, sys, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import fsspec
from huggingface_hub import HfApi

REPO = "jasperai/monet"
OUT = Path("/data2/monet/pool-20m")
LIGHT = ["id", "source", "sscd_nn", "sscd_cluster_id", "aesthetic_jasperai"]


def _read_shard(path):
    fs = fsspec.filesystem("hf")
    with fs.open(f"datasets/{REPO}/{path}", "rb") as fh:
        t = pq.read_table(fh, columns=LIGHT)
    return path, t


def main():
    n_target = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000_000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    OUT.mkdir(parents=True, exist_ok=True)
    api = HfApi(); fs = fsspec.filesystem("hf")
    all_files = sorted(f.path for f in api.list_repo_tree(REPO, repo_type="dataset", recursive=True)
                       if getattr(f, "size", None) is not None and f.path.endswith(".parquet"))
    random.Random(seed).shuffle(all_files)   # SAME shuffle as the sample -> random-2m ⊂ pool
    # MONET shards are ~uniform (~9929 rows); pick shard count directly (skip a slow sequential meta pass).
    # Exact row count comes from the parallel reads; a small drift from 20M is fine for a candidate pool.
    n_shards = min(len(all_files), (n_target + 9928) // 9929)
    picked = all_files[:n_shards]
    print(f"pool: {len(picked)} shards for ~{n_target:,} rows (parallel read, {workers} workers)", flush=True)

    # pass 2: parallel light-column reads
    ids = []; src = []; nn = []; cid = []; aes = []; shard_idx = []; local_row = []
    order = {p: i for i, p in enumerate(picked)}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_read_shard, p): p for p in picked}
        for fut in as_completed(futs):
            p, t = fut.result(); si = order[p]; n = t.num_rows
            ids.append(np.array(t["id"].to_pylist()))
            src.append(np.array(t["source"].to_pylist()))
            nn.append(np.array(t["sscd_nn"].to_pylist(), dtype=np.float32))
            cid.append(np.array(t["sscd_cluster_id"].to_pylist()))
            aes.append(np.array(t["aesthetic_jasperai"].to_pylist(), dtype=np.float32))
            shard_idx.append(np.full(n, si, dtype=np.int32))
            local_row.append(np.arange(n, dtype=np.int32))
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(picked)} shards", flush=True)

    ids = np.concatenate(ids); src = np.concatenate(src); nn = np.concatenate(nn)
    cid = np.concatenate(cid); aes = np.concatenate(aes)
    shard_idx = np.concatenate(shard_idx); local_row = np.concatenate(local_row)
    np.save(OUT / "sscd_nn.npy", nn); np.save(OUT / "aesthetic.npy", aes)
    np.save(OUT / "prov_shard_idx.npy", shard_idx); np.save(OUT / "prov_local_row.npy", local_row)
    np.save(OUT / "id.npy", ids); np.save(OUT / "source.npy", src); np.save(OUT / "sscd_cluster_id.npy", cid)
    manifest = {"schema": "monet-pool-20m-2026-09-02", "repo": REPO, "seed": seed,
                "n_rows": int(nn.shape[0]), "n_shards": len(picked), "columns": LIGHT,
                "shards": picked,   # index i in prov_shard_idx -> picked[i]
                "sscd_nn_direction": "SIMILARITY (HIGH=near-duplicate, LOW=rare); confirmed corr +0.40 vs in-shard NN cosine",
                "note": "light cols only; embeddings pulled later for the selected union via (prov_shard_idx, prov_local_row).",
                "source_counts": {s: int((src == s).sum()) for s in np.unique(src)}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"POOL DONE: {nn.shape[0]:,} rows, {len(picked)} shards -> {OUT}", flush=True)
    print("source composition:", manifest["source_counts"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
