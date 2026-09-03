"""Exact-embedding pull for scoring the publisher's MONET UMAP (owner MONET, overseer 2026-09-03). CPU/net.
The 1M umap-1m hashes touch only 107 dataset shards (coverage probe), so an EXACT pull is cheap (~2GB CLIP,
~7GB DINOv2). Pulls CLIP-512 FIRST (smaller → first their-layout FFR + half the source-space determination
sooner), then DINOv2-1536. Output is ROW-ALIGNED to umap-1m/points.parquet (embedding[i] ↔ that row's x/y),
with x/y + cluster_id/cluster_name carried alongside. PQ-reconstruct is NOT used as truth (24x lossy — rejected).
Usage: monet_their_umap_pull.py <clip|dino>."""
import sys, json, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
import fsspec

REPO = "jasperai/monet"
RS = Path("/data2/monet/retrieval-storage")
OUT = Path("/data2/monet/theirumap"); OUT.mkdir(parents=True, exist_ok=True)
COL = {"clip": ("embedding_clip-vit-base-patch32", 512), "dino": ("embedding_dinov2-vitg14", 1536)}


def main():
    space = sys.argv[1]; col, dim = COL[space]
    # umap-1m rows in order (+ carry x/y + clusters on the clip pass)
    up = pq.read_table(RS / "umap-data/umap-1m/points.parquet",
                       columns=["hash_perceptual", "x", "y", "cluster_id", "cluster_name"])
    uh = [h.decode() if isinstance(h, bytes) else h for h in up["hash_perceptual"].to_pylist()]
    pos = {h: i for i, h in enumerate(uh)}; N = len(uh)
    if space == "clip":
        np.save(OUT / "xy.npy", np.stack([up["x"].to_numpy(), up["y"].to_numpy()], 1).astype(np.float32))
        np.savez(OUT / "meta.npz", hash=np.array(uh), cluster_id=np.array(up["cluster_id"].to_pylist()),
                 cluster_name=np.array(up["cluster_name"].to_pylist()))

    # which dataset shards hold these hashes (vectorized join on index.parquet)
    idx = pq.read_table(RS / "index.parquet", columns=["hash_perceptual", "local_path"])
    sub = idx.filter(pc.is_in(idx["hash_perceptual"], value_set=pa.array(set(uh))))
    shards = sorted(set(sub["local_path"].to_pylist()))
    print(f"{space}: {N:,} umap rows across {len(shards)} shards", flush=True)

    fs = fsspec.filesystem("hf")
    emb = np.zeros((N, dim), np.float32); found = np.zeros(N, bool)
    t0 = time.time()
    for j, sp in enumerate(shards):
        with fs.open(f"datasets/{REPO}/v1.2.0/{sp}", "rb") as fh:
            t = pq.read_table(fh, columns=["hash_perceptual", col])
        hh = t["hash_perceptual"].to_pylist(); vv = t[col].to_pylist()
        for h, v in zip(hh, vv):
            i = pos.get(h)
            if i is not None and not found[i]:
                emb[i] = np.asarray(v, np.float32); found[i] = True
        if (j + 1) % 20 == 0:
            print(f"  {j+1}/{len(shards)} shards, found {found.sum():,}/{N:,} ({time.time()-t0:.0f}s)", flush=True)
    np.save(OUT / f"{space}.f32.npy", emb)
    manifest = {"schema": f"monet-theirumap-{space}-2026-09-03", "n_rows": N, "dim": dim,
                "found": int(found.sum()), "missing": int((~found).sum()), "n_shards": len(shards),
                "row_aligned_to": "umap-1m/points.parquet order (xy.npy / meta.npz)"}
    (OUT / f"{space}-manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{space} DONE: {found.sum():,}/{N:,} found, {(~found).sum():,} missing -> {OUT/(space+'.f32.npy')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
