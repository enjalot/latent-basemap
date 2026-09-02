"""MONET random 2M-sample downloader (owner MONET eval, overseer 2026-09-02). CPU/network only — NO GPU.
Pulls ONLY the embedding + provenance columns (never images/tar/vae/thumbnail) via column-projected parquet
reads over hf://, so a 2M sample is ~16 GB, not the 6.7 TB full parquet. Per-shard outputs (resumable) +
a manifest recording the exact source-shard list, per-shard row counts, draw seed, and columns — substrate
convention. Source column kept (MONET is 13.8M synthetic; must stay segmentable).

Cols: id, source, embedding_clip-vit-base-patch32 (512), embedding_dinov2-vitg14 (1536), sscd_nn,
sscd_cluster_id, aesthetic_jasperai. Usage: monet_download_sample.py [N_TARGET] [SEED]."""
import json, sys, time, random
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import fsspec
from huggingface_hub import HfApi

REPO = "jasperai/monet"
OUT = Path("/data2/monet/random-2m")
COLS = ["id", "source", "embedding_clip-vit-base-patch32", "embedding_dinov2-vitg14",
        "sscd_nn", "sscd_cluster_id", "aesthetic_jasperai"]


def main():
    n_target = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    (OUT / "shards").mkdir(parents=True, exist_ok=True)
    fs = fsspec.filesystem("hf")
    api = HfApi()

    # all parquet shards under v1.2.0/
    all_files = [f.path for f in api.list_repo_tree(REPO, repo_type="dataset", recursive=True)
                 if getattr(f, "size", None) is not None and f.path.endswith(".parquet")]
    all_files.sort()
    rng = random.Random(seed); rng.shuffle(all_files)
    print(f"{len(all_files)} parquet shards; drawing until >= {n_target:,} rows (seed {seed})", flush=True)

    # pass 1: pick shards + read num_rows (metadata only) until target reached
    picked = []; total = 0
    for p in all_files:
        try:
            with fs.open(f"datasets/{REPO}/{p}", "rb") as fh:
                nr = pq.ParquetFile(fh).metadata.num_rows
        except Exception as e:
            print(f"  skip {p}: {str(e)[:60]}", flush=True); continue
        picked.append((p, nr)); total += nr
        if total >= n_target:
            break
    print(f"picked {len(picked)} shards, {total:,} rows", flush=True)

    # pass 2: download the projected cols per shard (resumable)
    manifest_shards = []
    for i, (p, nr) in enumerate(picked):
        base = OUT / "shards" / f"{i:04d}"
        done_flag = base.with_suffix(".done")
        src = p.split("/")[1] if "/" in p else "?"      # v1.2.0/<source>/...
        if done_flag.exists():
            manifest_shards.append({"idx": i, "path": p, "rows": nr, "source": src}); continue
        t0 = time.time()
        with fs.open(f"datasets/{REPO}/{p}", "rb") as fh:
            tbl = pq.read_table(fh, columns=COLS)
        clip = np.array([r for r in tbl["embedding_clip-vit-base-patch32"].to_pylist()], dtype=np.float32)
        dino = np.array([r for r in tbl["embedding_dinov2-vitg14"].to_pylist()], dtype=np.float32)
        np.save(base.with_name(base.name + "_clip.npy"), clip)
        np.save(base.with_name(base.name + "_dino.npy"), dino)
        np.savez(base.with_name(base.name + "_meta.npz"),
                 id=np.array(tbl["id"].to_pylist()), source=np.array(tbl["source"].to_pylist()),
                 sscd_nn=np.array(tbl["sscd_nn"].to_pylist(), dtype=np.float32),
                 sscd_cluster_id=np.array(tbl["sscd_cluster_id"].to_pylist()),
                 aesthetic=np.array(tbl["aesthetic_jasperai"].to_pylist(), dtype=np.float32))
        done_flag.write_text("ok")
        manifest_shards.append({"idx": i, "path": p, "rows": int(clip.shape[0]), "source": src})
        print(f"  [{i+1}/{len(picked)}] {p} rows={clip.shape[0]} clip{clip.shape} dino{dino.shape} {time.time()-t0:.1f}s", flush=True)

    manifest = {"schema": "monet-random-sample-2026-09-02", "repo": REPO, "draw": "uniform-random-over-shards",
                "seed": seed, "n_target": n_target, "n_rows": int(sum(s["rows"] for s in manifest_shards)),
                "n_shards": len(manifest_shards), "columns": COLS,
                "clip_dim": 512, "dino_dim": 1536, "dtype": "float32",
                "note": "embedding + provenance columns ONLY (no images/tar/vae/thumbnail). source kept for "
                        "synthetic segmentation; sscd_nn/sscd_cluster_id shipped redundancy signal.",
                "shards": manifest_shards}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"DONE: {manifest['n_rows']:,} rows across {manifest['n_shards']} shards -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
