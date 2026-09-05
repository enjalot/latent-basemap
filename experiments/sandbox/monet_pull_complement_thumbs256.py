"""MONET complement thumbnails -> EXTEND the existing pool-20m-thumbs256 packed store (owner GO 2026-09-05).
CPU/network only — zero GPU. ADDITIVE: writes new shard files shards/{gi:04d} for gi in 2015..10879 (the
complement's ABSOLUTE global shard idx), never touching the pool's already-done 0..2014. One coherent thumb
store covers all ~103.8M rows; the packed thumb ref (shard_idx<<16 | local_row) is corpus-global and every
complement shard_idx (<=10879) fits the 16-bit upper half, so lsvoxel.MonetThumbStore reads it unchanged.

Format identical to scripts/pull_pool_thumbs256.py:
  shards/{gi:04d}.blob         concatenated 256px WEBP, HF row order
  shards/{gi:04d}.offsets.u64  n_rows+1 cumulative byte offsets (uint64 LE)
  shards/{gi:04d}.done         written last -> resumable
  shards/{gi:04d}.meta.json    n_rows, n_failed_decode, bytes, wall

NO black-image substitution: a row whose source thumbnail fails to decode gets a ZERO-LENGTH span
(offset[i+1]==offset[i]); failures are counted per shard (validity), never replaced with a synthetic image.
Per-1000-shard checkpoints. CPU-heavy (decode/resize/webp) -> launched under nice + CPUQuota so the GPU
trainer + backlog keep priority; runs slower while they're busy, by design.

Usage: monet_pull_complement_thumbs256.py [WORKERS=8]. Resumable; re-run to continue.
"""
from __future__ import annotations
import io, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

REPO = "jasperai/monet"
COMPLEMENT = Path("/data2/monet/pool-complement-88m")
THUMBS = Path("/data2/monet/pool-20m-thumbs256")          # SAME store, extended
SHARDS_DIR = THUMBS / "shards"
MAX_SIDE = 256
WEBP_QUALITY = 80
HF_RETRIES = 6
CKPT_EVERY = 1000


def _read_thumbnail_column(shard_path):
    import fsspec, pyarrow.parquet as pq
    fs = fsspec.filesystem("hf")
    last = None
    for attempt in range(HF_RETRIES):
        try:
            with fs.open(f"datasets/{REPO}/{shard_path}", "rb") as fh:
                return pq.read_table(fh, columns=["thumbnail"]).column("thumbnail").to_pylist()
        except Exception as e:
            last = e; time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"{shard_path}: giving up after {HF_RETRIES}: {last}")


def _downscale_webp(raw):
    from PIL import Image
    try:
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        buf = io.BytesIO(); im.save(buf, format="WEBP", quality=WEBP_QUALITY, method=4)
        return buf.getvalue()
    except Exception:
        return None


def process_shard(args):
    gi, shard_path = args
    base = SHARDS_DIR / f"{gi:04d}"
    if base.with_suffix(".done").exists():
        return {"shard_idx": gi, "skipped": True}
    t0 = time.time()
    thumbs = _read_thumbnail_column(shard_path); n = len(thumbs)
    offsets = np.zeros(n + 1, dtype="<u8"); chunks = []; n_failed = 0; total = 0
    for i, raw in enumerate(thumbs):
        out = _downscale_webp(raw) if raw else None
        if out is None:            # decode failure -> zero-length span, NO substitution
            n_failed += 1; out = b""
        chunks.append(out); total += len(out); offsets[i + 1] = total
    blob_tmp = base.with_suffix(".blob.tmp")
    with open(blob_tmp, "wb") as f:
        for c in chunks:
            f.write(c)
    os.replace(blob_tmp, base.with_suffix(".blob"))
    off_tmp = base.with_suffix(".offsets.u64.tmp"); offsets.tofile(off_tmp)
    os.replace(off_tmp, base.with_suffix(".offsets.u64"))
    meta = {"shard_idx": gi, "shard_path": shard_path, "n_rows": n, "n_failed_decode": n_failed,
            "bytes": total, "wall_s": round(time.time() - t0, 1)}
    base.with_suffix(".meta.json").write_text(json.dumps(meta))
    base.with_suffix(".done").touch()
    return {**meta, "skipped": False}


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    fsp = COMPLEMENT / "full_shards.json"
    if fsp.exists():
        full = json.loads(fsp.read_text()); shuffle = full["shards"]; n_pool = full["n_pool"]
    else:   # independent of the substrate stream: reproduce the verified seed-42 shuffle ourselves
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import monet_download_complement as sub
        shuffle, n_pool, _ = sub._full_shuffle()
    complement = shuffle[n_pool:]                     # gi = n_pool + j
    SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    todo = [(n_pool + j, complement[j]) for j in range(len(complement))
            if not (SHARDS_DIR / f"{n_pool + j:04d}.done").exists()]
    print(f"[comp-thumbs] complement {len(complement)} shards (gi {n_pool}..{n_pool+len(complement)-1}), "
          f"{len(complement)-len(todo)} done, {len(todo)} to go, {workers} workers", flush=True)

    t0 = time.time(); dc = 0; rows = 0; bt = 0; failed = 0; errors = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_shard, t): t for t in todo}
        for fut in as_completed(futs):
            gi, sp = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                errors.append(f"{gi} {sp}: {e}"); print(f"[comp-thumbs] FAILED {gi}: {e}", flush=True); continue
            if r.get("skipped"):
                continue
            dc += 1; rows += r["n_rows"]; bt += r["bytes"]; failed += r["n_failed_decode"]
            if dc % CKPT_EVERY == 0 or dc == len(todo):
                el = time.time() - t0; rate = dc / el if el else 0
                eta = (len(todo) - dc) / rate / 3600 if rate else float("inf")
                print(f"[comp-thumbs] CKPT {dc}/{len(todo)} shards  rows={rows:,}  out={bt/1e9:.1f}GB  "
                      f"failed_decode={failed} ({failed/max(rows,1)*100:.3f}%)  {rate*60:.1f} sh/min  "
                      f"eta~{eta:.1f}h  errors={len(errors)}", flush=True)

    # combined manifest across the FULL store (pool 0..n_pool-1 + complement), extension boundary recorded
    per_shard = []
    for gi in range(len(shuffle)):
        mp = SHARDS_DIR / f"{gi:04d}.meta.json"
        per_shard.append(json.loads(mp.read_text()) if mp.exists()
                         else {"shard_idx": gi, "shard_path": shuffle[gi], "missing": True})
    n_done = sum(1 for s in per_shard if not s.get("missing"))
    tot_fail = sum(s.get("n_failed_decode", 0) for s in per_shard if not s.get("missing"))
    tot_rows = sum(s.get("n_rows", 0) for s in per_shard if not s.get("missing"))
    manifest = {
        "schema": "monet-full-thumbs256-2026-09-05", "repo": REPO, "max_side_px": MAX_SIDE,
        "format": "webp", "webp_quality": WEBP_QUALITY,
        "extension_boundary": {"pool_shards": n_pool, "complement_first_global_shard_idx": n_pool,
                               "total_shards": len(shuffle)},
        "shard_order": "full seed-42 shuffle (pool-complement-88m/full_shards.json); gi = prov_shard_idx",
        "n_shards": len(shuffle), "n_shards_done": n_done,
        "lookup": "row -> (prov_shard_idx, prov_local_row) -> shards/{gi:04d}.offsets[lr:lr+2] -> blob slice",
        "validity": {"total_rows_done": tot_rows, "total_failed_decode": tot_fail,
                     "failed_decode_frac": round(tot_fail / max(tot_rows, 1), 6),
                     "policy": "zero-length span; NO black-image substitution"},
        "errors": errors,
    }
    (THUMBS / "manifest-full.json").write_text(json.dumps(manifest, indent=1))
    print(f"[comp-thumbs] DONE {dc} shards this run; store now {n_done}/{len(shuffle)}; "
          f"failed_decode {tot_fail:,}/{tot_rows:,} ({tot_fail/max(tot_rows,1)*100:.3f}%); errors={len(errors)}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
