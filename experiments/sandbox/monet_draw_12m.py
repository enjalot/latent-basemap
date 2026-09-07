"""Nested 12M DINO→PCA-768 draw for the ladder's 12M rung (owner via overseer 2026-09-07). CPU-only (gather +
CPU PCA transform, so it runs OFF-FLOCK — no GPU-during-I/O waste). Nested on the 6M draw (6M ⊂ 12M): members =
the 6M draw's full_pos, +6M new drawn from the full 103.8M column excluding the 6M. REUSES the 6M-fitted PCA
(random-dino-6m/pca768-model.npz) — NO refit — testing PCA transferability and matching how a 30M run would reuse
it (decision recorded in the card + manifest). PCA-768 substrate written f16 (disk flag: 12M×768×2 = 18.4GB).

Full-corpus row space (project_full_corpus contract): [0,19344847)=pool, [19344847,103.8M)=complement.
Output /data2/monet/random-dino-12m/: {pca768-substrate.f16.npy (12M×768, renormed), full_pos.npy, member_mask.npy
(marks the nested 6M), val_pos/test_pos.npy, val-pca768.f16.npy, test-pca768.f16.npy, id.npy, manifest.json}.
Usage: monet_draw_12m.py [N_TARGET=12000000] [SEED=42].
"""
import json, sys, time
from pathlib import Path
import numpy as np

POOL = Path("/data2/monet/pool-20m"); COMP = Path("/data2/monet/pool-complement-88m")
SIXM = Path("/data2/monet/random-dino-6m"); OUT = Path("/data2/monet/random-dino-12m")
PCA_MODEL = SIXM / "pca768-model.npz"
N_POOL = 19_344_847; DIM_IN, DIM_OUT, CHUNK = 1536, 768, 200_000


def main():
    n_target = int(sys.argv[1]) if len(sys.argv) > 1 else 12_000_000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    n_val = n_test = 200_000
    OUT.mkdir(parents=True, exist_ok=True)

    pool_d = np.load(POOL / "dino1536.f16.npy", mmap_mode="r"); comp_d = np.load(COMP / "dino1536.f16.npy", mmap_mode="r")
    comp_n = comp_d.shape[0]; N_full = N_POOL + comp_n
    member_pos = np.load(SIXM / "full_pos.npy").astype(np.int64)            # the nested 6M (full-corpus positions)
    assert member_pos.max() < N_full and np.unique(member_pos).size == member_pos.size
    print(f"full {N_full:,} | nested members (6M) {member_pos.size:,} | target {n_target:,}", flush=True)

    n_new = n_target - member_pos.size
    is_member = np.zeros(N_full, bool); is_member[member_pos] = True
    nonmember = np.where(~is_member)[0]
    rng = np.random.default_rng(seed); perm = rng.permutation(nonmember.size)
    new_pos = nonmember[perm[:n_new]]
    full_pos = np.sort(np.concatenate([member_pos, new_pos]))
    assert full_pos.size == n_target and np.unique(full_pos).size == n_target, "12M union not clean"
    rest = nonmember[perm[n_new:]]
    rng2 = np.random.default_rng(seed + 1); vt = rng2.permutation(rest.size)
    val_pos = np.sort(rest[vt[:n_val]]); test_pos = np.sort(rest[vt[n_val:n_val + n_test]])
    assert len(np.intersect1d(val_pos, full_pos)) == 0 and len(np.intersect1d(test_pos, full_pos)) == 0
    member_mask = is_member[full_pos]; assert member_mask.sum() == member_pos.size, "6M nesting broken"

    pm = np.load(PCA_MODEL); comp_p = pm["components"].astype(np.float32); mean_p = pm["mean"].astype(np.float32)
    assert comp_p.shape == (DIM_IN, DIM_OUT), f"PCA components {comp_p.shape} != {(DIM_IN, DIM_OUT)}"

    def gather_pca(positions, out_path):
        """Gather DINO f16 at SORTED full-corpus positions, CPU PCA-transform (reuse 6M model) + renorm -> f16."""
        n = positions.shape[0]; mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(n, DIM_OUT))
        for i in range(0, n, CHUNK):
            j = min(i + CHUNK, n); p = positions[i:j]; ispool = p < N_POOL
            x = np.empty((j - i, DIM_IN), np.float32)
            if ispool.any(): x[ispool] = np.asarray(pool_d[p[ispool]], np.float32)
            if (~ispool).any(): x[~ispool] = np.asarray(comp_d[p[~ispool] - N_POOL], np.float32)
            y = (x - mean_p) @ comp_p                                       # PCA project (CPU matmul, MKL-parallel)
            y /= (np.linalg.norm(y, axis=1, keepdims=True) + 1e-12)         # renorm (matches head training prep)
            mm[i:j] = y.astype(np.float16)
        mm.flush(); return n

    t0 = time.time()
    gather_pca(full_pos, OUT / "pca768-substrate.f16.npy")
    gather_pca(val_pos, OUT / "val-pca768.f16.npy"); gather_pca(test_pos, OUT / "test-pca768.f16.npy")
    wall = time.time() - t0

    np.save(OUT / "full_pos.npy", full_pos); np.save(OUT / "member_mask.npy", member_mask)
    np.save(OUT / "val_pos.npy", val_pos); np.save(OUT / "test_pos.npy", test_pos)
    pool_ids = np.load(POOL / "id.npy", allow_pickle=True); ids = np.empty(full_pos.size, dtype=object)
    pmask = full_pos < N_POOL; ids[pmask] = pool_ids[full_pos[pmask]]
    if (~pmask).any():
        comp_ids = np.load(COMP / "id.npy", allow_pickle=True); ids[~pmask] = comp_ids[full_pos[~pmask] - N_POOL]
    np.save(OUT / "id.npy", ids)
    (OUT / "manifest.json").write_text(json.dumps({
        "schema": "monet-random-dino-12m-pca768-2026-09-07", "seed": seed, "n_rows": int(full_pos.size),
        "n_members_nested_6m": int(member_pos.size), "n_new": int(n_new),
        "composition": {"pool_rows": int(pmask.sum()), "complement_rows": int((~pmask).sum())},
        "n_val": int(val_pos.size), "n_test": int(test_pos.size),
        "pca": {"reused_model": str(PCA_MODEL), "refit": False, "dim_in": DIM_IN, "dim_out": DIM_OUT,
                "why_reuse": "transferability test + matches 30M reuse; the 6M PCA is applied to the 12M draw, NOT refit"},
        "substrate": {"file": "pca768-substrate.f16.npy", "dtype": "float16", "renormed": True, "unit_norm": True},
        "gather_wall_s": round(wall, 1),
        "row_order": "sorted full-corpus position; member_mask marks the nested 6M"}, indent=1))
    print(f"12M DINO→PCA768 draw DONE: {full_pos.size:,} rows ({member_pos.size:,} nested + {n_new:,} new; "
          f"pool {int(pmask.sum()):,} / comp {int((~pmask).sum()):,}), val {val_pos.size:,} test {test_pos.size:,}, {wall:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
