"""MONET full-corpus COMPLEMENT substrate pull (owner GO 2026-09-05, via overseer). CPU/network only — zero GPU.

Extends the 19.34M pool (pool-20m/) into ONE coherent ~103.8M substrate by pulling the 8,865 shards NOT in the
pool. Provenance CONTINUES the pool scheme: the pool is the first 2,015 shards of the seed-42 shuffle of all
10,880 repo parquet shards; the complement is shuffle[2015:], with ABSOLUTE global shard_idx (2015..10879) so a
single shards[] list (= the full shuffle) and the same (shard_idx, local_row) lookup covers the whole substrate.
Verified invariant: shuffle[:2015] == pool-20m/manifest.json['shards'] EXACTLY (asserted at startup).

ONE network read per shard (no footer pass, no double-read -> avoids HF 429 bursts). Per-shard artifacts, then a
pure-local finalize concatenates the coherent flat arrays:
  /data2/monet/pool-complement-88m/
    full_shards.json       the entire 10,880-shard seed-42 shuffle + n_pool boundary
    clip/{gi:05d}.f32.npy   per-shard (n,512) float32  (embedding_clip-vit-base-patch32, f64->f32; L2-normed)
    light/{gi:05d}.npz      per-shard id/source/sscd_nn/sscd_cluster_id/aesthetic
    done_shards.json        resumable: absolute gi's fully written
    -- after finalize (local, no network) --
    clip512.f32.npy         (N,512) float32  ~176 GB   row order = shard order (offsets.npy)
    prov_shard_idx.npy      (N,) int32  ABSOLUTE gi (2015..10879)
    prov_local_row.npy      (N,) int32
    sscd_nn.npy id.npy source.npy sscd_cluster_id.npy aesthetic.npy
    offsets.npy             (n_complement+1,) int64 row block per shard
    manifest.json validity.json  extension boundary + zero-clip-row accounting

CLIP dtype matches the pool (f32). A row's clip left all-zero (shouldn't happen; dataset embeddings are unit-norm)
would be counted in validity.json and excluded by a downstream mask — NO substitution. Per-1000-shard checkpoints.

Usage: monet_download_complement.py [WORKERS=4] [--finalize]. Resumable; re-run to continue. --finalize runs the
local concatenation only (auto-runs once all shards are fetched).
"""
import json, sys, time, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import fsspec
from huggingface_hub import HfApi

REPO = "jasperai/monet"
POOL_MANIFEST = Path("/data2/monet/pool-20m/manifest.json")
OUT = Path("/data2/monet/pool-complement-88m")
CLIP_COL = "embedding_clip-vit-base-patch32"
LIGHT = ["id", "source", "sscd_nn", "sscd_cluster_id", "aesthetic_jasperai"]
HF_RETRIES = 6
CKPT_EVERY = 1000


def _hf_token():
    for p in (Path.home() / ".cache/huggingface/token",):
        if p.exists():
            t = p.read_text().strip()
            if t:
                return t
    import os
    return os.environ.get("HF_TOKEN") or None


def _full_shuffle():
    pool = json.loads(POOL_MANIFEST.read_text())
    seed = pool["seed"]; pool_shards = pool["shards"]
    api = HfApi(token=_hf_token())   # authenticated -> higher /api rate limit than anonymous IP
    all_files = None
    for attempt in range(8):
        try:
            all_files = sorted(f.path for f in api.list_repo_tree(REPO, repo_type="dataset", recursive=True)
                               if getattr(f, "size", None) is not None and f.path.endswith(".parquet"))
            break
        except Exception as e:
            if attempt == 7:
                raise
            wait = min(310, 40 * 2 ** attempt)
            print(f"[complement] list_repo_tree rate-limited ({e}); wait {wait}s [retry {attempt+1}/8]", flush=True)
            time.sleep(wait)
    shuf = all_files[:]; random.Random(seed).shuffle(shuf)
    assert shuf[:len(pool_shards)] == pool_shards, "seed-42 shuffle prefix != pool shards — provenance broken"
    return shuf, len(pool_shards), seed


def _fetch_one(gi, path):
    fs = fsspec.filesystem("hf")
    last = None
    for attempt in range(HF_RETRIES):
        try:
            with fs.open(f"datasets/{REPO}/{path}", "rb") as fh:
                t = pq.read_table(fh, columns=[CLIP_COL] + LIGHT)
            break
        except Exception as e:
            last = e
            if attempt == HF_RETRIES - 1:
                raise RuntimeError(f"{path}: read failed after {HF_RETRIES}: {last}")
            time.sleep(min(120, 3 * 2 ** attempt))
    clip = np.asarray(t[CLIP_COL].to_pylist(), dtype=np.float32)
    if clip.ndim != 2 or clip.shape[1] != 512:
        raise RuntimeError(f"{path}: clip shape {clip.shape}")
    cp = OUT / "clip" / f"{gi:05d}.f32.npy"
    tmp = cp.parent / (cp.name + ".tmp")          # np.save to a file handle does NOT auto-append .npy
    with open(tmp, "wb") as f:
        np.save(f, clip)
    tmp.replace(cp)
    np.savez(OUT / "light" / f"{gi:05d}.npz",
             id=np.array(t["id"].to_pylist()), source=np.array(t["source"].to_pylist()),
             sscd_nn=np.asarray(t["sscd_nn"].to_pylist(), dtype=np.float32),
             sscd_cluster_id=np.array(t["sscd_cluster_id"].to_pylist()),
             aesthetic=np.asarray(t["aesthetic_jasperai"].to_pylist(), dtype=np.float32))
    return gi, clip.shape[0]


def finalize(shuffle, n_pool):
    complement = shuffle[n_pool:]
    counts = np.array([int(np.load(OUT / "clip" / f"{n_pool+j:05d}.f32.npy", mmap_mode="r").shape[0])
                       for j in range(len(complement))], dtype=np.int64)
    offsets = np.zeros(len(complement) + 1, dtype=np.int64); offsets[1:] = np.cumsum(counts)
    N = int(offsets[-1]); np.save(OUT / "offsets.npy", offsets)
    clip = np.lib.format.open_memmap(OUT / "clip512.f32.npy", mode="w+", dtype=np.float32, shape=(N, 512))
    nn = np.empty(N, np.float32); psi = np.empty(N, np.int32); plr = np.empty(N, np.int32)
    ids = np.empty(N, object); src = np.empty(N, object); cid = np.empty(N, object); aes = np.empty(N, np.float32)
    zero = 0
    for j in range(len(complement)):
        gi = n_pool + j; lo, hi = int(offsets[j]), int(offsets[j + 1]); n = hi - lo
        c = np.load(OUT / "clip" / f"{gi:05d}.f32.npy"); clip[lo:hi] = c
        zero += int((np.linalg.norm(c, axis=1) == 0).sum())
        z = np.load(OUT / "light" / f"{gi:05d}.npz", allow_pickle=True)
        nn[lo:hi] = z["sscd_nn"]; ids[lo:hi] = z["id"]; src[lo:hi] = z["source"]
        cid[lo:hi] = z["sscd_cluster_id"]; aes[lo:hi] = z["aesthetic"]
        psi[lo:hi] = gi; plr[lo:hi] = np.arange(n)
        if j % 1000 == 0:
            clip.flush(); print(f"  finalize {j}/{len(complement)}", flush=True)
    clip.flush()
    np.save(OUT / "sscd_nn.npy", nn); np.save(OUT / "prov_shard_idx.npy", psi); np.save(OUT / "prov_local_row.npy", plr)
    np.save(OUT / "id.npy", ids); np.save(OUT / "source.npy", src)
    np.save(OUT / "sscd_cluster_id.npy", cid); np.save(OUT / "aesthetic.npy", aes)
    src_counts = {s: int((src == s).sum()) for s in set(src.tolist())}
    manifest = {
        "schema": "monet-complement-substrate-2026-09-05", "repo": REPO, "extension_of": str(POOL_MANIFEST),
        "extension_boundary": {"pool_shards": n_pool,
                               "pool_rows": int(json.loads(POOL_MANIFEST.read_text())["n_rows"]),
                               "complement_shards": len(complement), "complement_first_global_shard_idx": n_pool,
                               "complement_rows": N,
                               "provenance": "ABSOLUTE global shard_idx continues at the pool boundary; one coherent "
                               "substrate = concat(pool arrays, complement arrays); shards[]=full seed-42 shuffle"},
        "clip": {"file": "clip512.f32.npy", "dim": 512, "dtype": "float32", "n_rows": N, "col": CLIP_COL,
                 "row_order": "shard order (offsets.npy)"},
        "source_counts": src_counts}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (OUT / "validity.json").write_text(json.dumps({"complement_rows": N, "zero_clip_rows": zero,
        "zero_clip_frac": round(zero / max(N, 1), 8), "policy": "no substitution; mask excludes zero-clip rows"}, indent=1))
    print(f"[complement] FINALIZE done: N={N:,}  zero-clip rows {zero}  -> {OUT/'clip512.f32.npy'}", flush=True)
    return N


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    workers = int(args[0]) if args else 4
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "clip").mkdir(exist_ok=True); (OUT / "light").mkdir(exist_ok=True)

    fsp = OUT / "full_shards.json"
    if fsp.exists():
        full = json.loads(fsp.read_text()); shuffle = full["shards"]; n_pool = full["n_pool"]
    else:
        shuffle, n_pool, seed = _full_shuffle()
        fsp.write_text(json.dumps({"shards": shuffle, "n_pool": n_pool, "seed": seed,
                                   "n_complement": len(shuffle) - n_pool}))
    complement = shuffle[n_pool:]

    if "--finalize" in sys.argv:
        return 0 if finalize(shuffle, n_pool) else 1

    done_path = OUT / "done_shards.json"
    done = set(json.loads(done_path.read_text())) if done_path.exists() else set()
    todo = [(n_pool + j, complement[j]) for j in range(len(complement)) if (n_pool + j) not in done]
    print(f"[complement] {len(complement)} complement shards (gi {n_pool}..{n_pool+len(complement)-1}), "
          f"{len(done)} done, {len(todo)} to go, {workers} workers", flush=True)

    t0 = time.time(); n_run = 0; errors = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_one, gi, p): gi for (gi, p) in todo}
        for fut in as_completed(futs):
            gi = futs[fut]
            try:
                gi_, n = fut.result()
            except Exception as e:
                errors.append(f"{gi}: {e}"); print(f"[complement] FAILED shard {gi}: {e}", flush=True); continue
            done.add(gi); n_run += 1
            if n_run % CKPT_EVERY == 0 or n_run == len(todo):
                done_path.write_text(json.dumps(sorted(done)))
                el = time.time() - t0; rate = n_run / el if el else 0
                eta = (len(todo) - n_run) / rate / 3600 if rate else float("inf")
                print(f"[complement] CKPT {len(done)}/{len(complement)} shards  {rate*60:.1f} sh/min  "
                      f"eta~{eta:.1f}h  errors={len(errors)}", flush=True)
    done_path.write_text(json.dumps(sorted(done)))
    print(f"[complement] fetch pass done: {len(done)}/{len(complement)} shards, {len(errors)} errors", flush=True)
    if len(done) == len(complement):
        print("[complement] all shards present -> finalize (local concat)", flush=True)
        finalize(shuffle, n_pool)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
