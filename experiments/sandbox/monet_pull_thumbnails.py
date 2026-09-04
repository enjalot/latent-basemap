"""Pull 250K MONET thumbnail+caption pairs for NeoMME exp-2 (joint-modality). CPU/network. Selects 250K of the
random-2m rows (seeded), maps each to its source MONET parquet via the random-2m manifest, and pulls the
thumbnail (binary) + caption (florence-2) + id + source for those rows -> /data2/monet/neomme-pairs-250k/
pairs.parquet. Column-projected reads (thumbnail is the fat 143MB/shard col — only the needed shards).
Resumable per source-parquet. Usage: monet_pull_thumbnails.py [N=250000] [SEED=13]."""
import sys, json
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, fsspec

REPO = "jasperai/monet"; RM = Path("/data2/monet/random-2m"); OUT = Path("/data2/monet/neomme-pairs-250k")


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 250_000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 13
    OUT.mkdir(parents=True, exist_ok=True)
    man = json.loads((RM / "manifest.json").read_text())
    shards = man["shards"]                                  # [{idx,path,rows,source}], in the sample order
    offs = np.cumsum([0] + [s["rows"] for s in shards]); total = int(offs[-1])
    sel = np.sort(np.random.default_rng(seed).choice(total, N, replace=False))
    fs = fsspec.filesystem("hf")
    tables = []; done = set(json.loads((OUT / "done.json").read_text())) if (OUT / "done.json").exists() else set()
    for si, s in enumerate(shards):
        lo, hi = offs[si], offs[si+1]
        rows = sel[(sel >= lo) & (sel < hi)] - lo          # local rows in this source parquet
        if len(rows) == 0 or si in done:
            continue
        with fs.open(f"datasets/{REPO}/{s["path"]}", "rb") as fh:
            t = pq.read_table(fh, columns=["id", "thumbnail", "caption_florence-2-large", "source"])
        sub = t.take(pa.array(rows)); (OUT / f"part-{si:04d}.parquet")
        pq.write_table(sub, OUT / f"part-{si:04d}.parquet")
        done.add(si); (OUT / "done.json").write_text(json.dumps(sorted(done)))
        print(f"  shard {si} ({s['path']}): {len(rows)} pairs", flush=True)
    # assemble
    parts = sorted(OUT.glob("part-*.parquet"))
    if parts:
        allt = pa.concat_tables([pq.read_table(p) for p in parts])
        pq.write_table(allt, OUT / "pairs.parquet")
        (OUT / "manifest.json").write_text(json.dumps({"schema": "monet-neomme-pairs-2026-09-04", "n": allt.num_rows,
            "cols": allt.schema.names, "seed": seed, "note": "thumbnail(binary)+caption(florence) pairs for NeoMME exp-2 joint-modality"}, indent=1))
        print(f"pairs assembled: {allt.num_rows} -> {OUT/'pairs.parquet'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
