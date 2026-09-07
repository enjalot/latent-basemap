"""Full-corpus (103.8M) projection through a MONET head (owner image phase, item b; 2026-09-05). GPU (transform).
Streams the WHOLE substrate — pool (19.34M) THEN complement (84.47M) — through a frozen head into ONE coherent
coords artifact, row-aligned to the combined provenance so the viewer's id/thumb join works across all 103.8M.
The "104M points on a map" artifact. Head-parameterized (PROJ_CKPT); coords dim = the head's n_components.

Row layout of the output (matches the coherent substrate contract):
  rows [0, 19_344_847)         = pool rows, in pool-20m order            (prov: pool-20m/prov_*)
  rows [19_344_847, 103.8M)    = complement rows, in complement order    (prov: pool-complement-88m/prov_*)

Inputs streamed lazily (never materializes a >CHUNK slice; the 39.6GB pool + 173GB complement stay on disk).
Requires the complement substrate FINALIZED (pool-complement-88m/clip512.f32.npy). Usage:
  PROJ_CKPT=<model.pt> PROJ_OUT=<dir> project_full_corpus.py
"""
import json, os, sys, time, hashlib
from pathlib import Path
import numpy as np

# Input columns parameterized (owner DINO endgame 2026-09-06): default CLIP-512, override to the DINO-1536 f16
# columns via POOL_SUB/COMP_SUB. All columns are pre-normed (L2=1.0) so the head's training _norm is matched; the
# f16 DINO column is cast f16->f32 per chunk in the loop (nil quality impact, recorded).
POOL_CLIP = Path(os.environ.get("POOL_SUB", "/data2/monet/pool-20m/clip512.f32.npy"))
COMP_CLIP = Path(os.environ.get("COMP_SUB", "/data2/monet/pool-complement-88m/clip512.f32.npy"))
CKPT = Path(os.environ["PROJ_CKPT"])
OUT = Path(os.environ["PROJ_OUT"])
BATCH = 16384


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not COMP_CLIP.exists():
        raise SystemExit(f"complement substrate not finalized yet: {COMP_CLIP}")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _paths import ensure_paths; ensure_paths()
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    import torch

    pool = np.load(POOL_CLIP, mmap_mode="r"); comp = np.load(COMP_CLIP, mmap_mode="r")
    n_pool, n_comp = pool.shape[0], comp.shape[0]
    N = n_pool + n_comp
    pumap = ParametricUMAP.load(str(CKPT), device="cuda")
    ncomp = int(getattr(pumap, "n_components", 2))
    ckpt_sha = hashlib.sha256(CKPT.read_bytes()).hexdigest()[:16]

    # Optional PCA preprocessing (owner 2026-09-07): a head trained on PCA-768 needs the FULL corpus to go through
    # the SAME saved components+mean before the head. Set PCA_MODEL=<pca768-model.npz>. f32->transform->renorm.
    PCA_MODEL = os.environ.get("PCA_MODEL"); comp_t = mean_t = None
    if PCA_MODEL:
        pm = np.load(PCA_MODEL); comp_t = torch.from_numpy(np.ascontiguousarray(pm["components"], np.float32)).cuda()
        mean_t = torch.from_numpy(np.ascontiguousarray(pm["mean"], np.float32)).cuda()
        def _pca(t):  # (b,1536)->(b,768) renormed, matching exp_dino_6m_pca.py's training-side prep
            return torch.nn.functional.normalize((t - mean_t) @ comp_t, dim=1)
        # ASSERT the transform matches the training-side prep: dim + a spot-check row vs the 6M substrate.
        assert comp_t.shape[1] == 768, f"PCA out-dim {comp_t.shape[1]} != 768"
        SPOT = os.environ.get("PCA_SPOTCHECK_SUB")  # 6M pca768 substrate + its full_pos to spot-check a shared row
        if SPOT:
            fp = np.load(os.environ["PCA_SPOTCHECK_FULLPOS"]); pj = int(fp[0])
            row = np.asarray((pool if pj < n_pool else comp)[pj if pj < n_pool else pj - n_pool], np.float32)
            got = _pca(torch.from_numpy(row[None]).cuda()).cpu().numpy()[0]
            ref = np.asarray(np.load(SPOT, mmap_mode="r")[0], np.float32)
            md = float(np.abs(got - ref).max()); assert md < 1e-3, f"PCA spot-check mismatch max|d|={md} (full_pos[0]={pj})"
            print(f"[full] PCA-768 spot-check OK (full_pos[0]={pj}, max|d| {md:.2e} vs 6M substrate)", flush=True)
    print(f"[full] pool {n_pool:,} + complement {n_comp:,} = {N:,} | head n_components={ncomp} | ckpt {ckpt_sha} | PCA={'yes' if PCA_MODEL else 'no'}", flush=True)

    coords = np.lib.format.open_memmap(OUT / "coords.f32.npy", mode="w+", dtype=np.float32, shape=(N, ncomp))
    t0 = time.time()
    pumap.model.eval()
    with torch.no_grad():
        base = 0
        for src, n_src, tag in ((pool, n_pool, "pool"), (comp, n_comp, "complement")):
            for i in range(0, n_src, BATCH):
                j = min(i + BATCH, n_src)
                chunk = torch.from_numpy(np.asarray(src[i:j], dtype=np.float32)).to("cuda")
                if PCA_MODEL:
                    chunk = _pca(chunk)
                coords[base + i:base + j] = pumap.model(chunk).cpu().numpy().astype(np.float32)
                if ((base + i) // BATCH) % 500 == 0:
                    el = time.time() - t0; done = base + j; rate = done / el if el else 0
                    print(f"[full] {tag} {done:,}/{N:,}  {rate/1e6:.2f}M rows/s  eta~{(N-done)/rate:.0f}s", flush=True)
            base += n_src
    coords.flush()
    wall = time.time() - t0

    s = coords[:: max(N // 200000, 1)].astype(np.float64)
    manifest = {
        "schema": "monet-clip-fullcorpus-proj-2026-09-05",
        "label": f"104M full-corpus projection — head n_components={ncomp}",
        "checkpoint": str(CKPT), "checkpoint_sha256_16": ckpt_sha, "dim": ncomp, "pca_model": PCA_MODEL,
        "n_rows": int(N), "n_pool": int(n_pool), "n_complement": int(n_comp),
        "row_layout": {"pool": [0, n_pool], "complement": [n_pool, N]},
        "row_alignment": "rows [0,n_pool) -> pool-20m id/prov; rows [n_pool,N) -> pool-complement-88m id/prov; "
                         "thumb = (prov_shard_idx<<16|prov_local_row) in the extended pool-20m-thumbs256 store",
        "wall_s": round(wall, 1), "rows_per_s": round(N / wall, 1),
        "finite_frac_sampled": round(float(np.isfinite(s).all(axis=1).mean()), 6),
        "note": "projection only; separate receipts. Three-way (member/pool-heldout/complement) reception scored separately.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[full] DONE {N:,} rows in {wall:.0f}s ({N/wall/1e6:.2f}M rows/s) -> {OUT/'coords.f32.npy'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
