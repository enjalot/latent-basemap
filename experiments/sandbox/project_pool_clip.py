"""Fast-path MONET CLIP image demo (owner GO 2026-09-05, via overseer): stream the full 19.34M-row pool CLIP-512
through the EXISTING 2M champion head -> 2D coords, row-aligned to the pool's id/thumb provenance. No training.
GPU (transform only). Labels everything "single-seed focused"/2M-head projection — separate dir, NEVER overwrites
the 2M receipts.

Preprocessing: transform() feeds X straight to the model (no internal norm); the pool clip is already unit-norm
(dataset embeddings), matching the champion's L2-normed training substrate, so the memmap is fed directly. Input
is streamed per batch (lazy memmap slice) — the 39.6 GB pool never materializes (>=2GB rule).

Row/image join: coords[i] corresponds to pool row i, hence pool id.npy[i] and thumb (prov_shard_idx[i],
prov_local_row[i]) packed as shard_idx<<16|local_row -> pool-20m-thumbs256. Recorded in the manifest.

Usage: project_pool_clip.py. Output: sandbox/monet-clip-fullpool-proj-20260905/{coords.f32.npy, manifest.json}.
"""
import json, os, sys, time, hashlib
from pathlib import Path
import numpy as np

# CKPT/OUT env-parameterized so the SAME streamer serves the 2M-2D head and the 2M-3D head (owner 2026-09-05).
# coords dim = the model's n_components (2 or 3), detected from the loaded head — NOT hardcoded.
CKPT = Path(os.environ.get("PROJ_CKPT", "/data/latent-basemap/sandbox/monet-random-clip-2m/champion-bs16k/model.pt"))
POOL = Path("/data2/monet/pool-20m")
OUT = Path(os.environ.get("PROJ_OUT", "/data/latent-basemap/sandbox/monet-clip-fullpool-proj-20260905"))
BATCH = 16384


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _paths import ensure_paths; ensure_paths()
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    import torch

    clip = np.load(POOL / "clip512.f32.npy", mmap_mode="r")     # (19.34M, 512) — lazy
    N = clip.shape[0]
    print(f"[proj] pool clip {clip.shape} {clip.dtype} | ckpt {CKPT}", flush=True)
    pumap = ParametricUMAP.load(str(CKPT), device="cuda")
    ckpt_sha = hashlib.sha256(CKPT.read_bytes()).hexdigest()[:16]
    ncomp = int(getattr(pumap, "n_components", 2))
    print(f"[proj] n_components={ncomp} -> coords ({N:,},{ncomp})", flush=True)

    coords = np.lib.format.open_memmap(OUT / "coords.f32.npy", mode="w+", dtype=np.float32, shape=(N, ncomp))
    t0 = time.time()
    with torch.no_grad():
        pumap.model.eval()
        for i in range(0, N, BATCH):
            j = min(i + BATCH, N)
            chunk = np.asarray(clip[i:j], dtype=np.float32)         # per-batch cast, off-RAM
            xy = pumap.model(torch.from_numpy(chunk).to("cuda")).cpu().numpy()
            coords[i:j] = xy.astype(np.float32)
            if (i // BATCH) % 200 == 0:
                el = time.time() - t0; rate = j / el if el else 0
                print(f"[proj] {j:,}/{N:,}  {rate/1e6:.2f}M rows/s  eta~{(N-j)/rate:.0f}s", flush=True)
    coords.flush()
    wall = time.time() - t0

    # validity + join manifest (coords row-aligned to pool id/prov)
    med = np.median(coords[:: max(N // 200000, 1)].astype(np.float64), 0)
    r = np.linalg.norm(coords[:: max(N // 200000, 1)].astype(np.float64) - med, axis=1)
    manifest = {
        "schema": "monet-clip-fullpool-proj-2026-09-05", "label": f"single-seed focused — 2M CLIP head (n_components={ncomp}), full-pool projection",
        "checkpoint": str(CKPT), "checkpoint_sha256_16": ckpt_sha, "trained_on_rows": 2008321,
        "n_rows": int(N), "coords_file": "coords.f32.npy", "dim": ncomp, "wall_s": round(wall, 1),
        "rows_per_s": round(N / wall, 1),
        "row_alignment": "coords[i] <-> pool-20m id.npy[i]; thumb = (prov_shard_idx[i]<<16|prov_local_row[i]) in pool-20m-thumbs256",
        "layout_radius_p50": round(float(np.percentile(r, 50)), 3),
        "note": "projection only; 2M receipts untouched. Held-out reception + viewer follow separately.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[proj] DONE {N:,} rows in {wall:.0f}s ({N/wall/1e6:.2f}M rows/s) -> {OUT/'coords.f32.npy'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
