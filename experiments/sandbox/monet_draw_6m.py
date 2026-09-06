"""Nested 6M DINO draw for the ladder's 6M rung (owner spec via overseer 2026-09-06). CPU-only, gathered from the
LOCAL full 103.8M DINO column (pool-20m/dino1536 + pool-complement-88m/dino1536). Same single-variable growth
pattern as the 4M CLIP design, but the +4M addition is drawn from the FULL corpus (pool+complement), not just pool.

Full-corpus row space (matches project_full_corpus contract): positions [0, 19_344_847)=pool, [19_344_847, 103.8M)=
complement. Construction (seed-42, deterministic):
  members = random-2m rows (id-join into pool) -> their POOL positions (all < 19.34M), the nested 2M
  new     = uniform draw of (6M - members) rows from the FULL column EXCLUDING members, seed 42
  6M      = sort(members ∪ new)  (full-corpus positions)
  val/test= disjoint uniform draws (seed 43) from the FULL column OUTSIDE the 6M union -> 200K each
Gather DINO f16 by the split: pos<19.34M -> pool dino1536[pos]; else complement dino1536[pos-19.34M].

Output: /data2/monet/random-dino-6m/{dino-substrate.f16.npy, full_pos.npy, member_mask.npy, id.npy, sscd_nn.npy
(pool-portion; complement lacks it -> NaN), val_pos.npy, test_pos.npy, val-dino.f16.npy, test-dino.f16.npy,
manifest.json}. Usage: monet_draw_6m.py [N_TARGET=6000000] [SEED=42].
"""
import json, sys, time
from pathlib import Path
import numpy as np

POOL = Path("/data2/monet/pool-20m"); COMP = Path("/data2/monet/pool-complement-88m")
RM = Path("/data2/monet/random-2m"); OUT = Path("/data2/monet/random-dino-6m")
N_POOL = 19_344_847; DIM = 1536; CHUNK = 200_000


def _gather_dino(pool_d, comp_d, positions, out_path):
    """Gather the DINO f16 column at full-corpus `positions` (SORTED) -> fresh f16 memmap, chunked (memmap locality)."""
    n = positions.shape[0]
    mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(n, DIM))
    for i in range(0, n, CHUNK):
        j = min(i + CHUNK, n); p = positions[i:j]
        is_pool = p < N_POOL
        out = np.empty((j - i, DIM), np.float16)
        if is_pool.any():
            out[is_pool] = pool_d[p[is_pool]]
        if (~is_pool).any():
            out[~is_pool] = comp_d[p[~is_pool] - N_POOL]
        mm[i:j] = out
    mm.flush(); return n


def main():
    n_target = int(sys.argv[1]) if len(sys.argv) > 1 else 6_000_000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    n_val = n_test = 200_000
    OUT.mkdir(parents=True, exist_ok=True)

    pool_ids = np.load(POOL / "id.npy", allow_pickle=True)
    comp_n = np.load(COMP / "dino1536.f16.npy", mmap_mode="r").shape[0]
    N_full = N_POOL + comp_n
    assert pool_ids.shape[0] == N_POOL, f"pool ids {pool_ids.shape[0]} != {N_POOL}"
    rm_ids = np.concatenate([np.load(m, allow_pickle=True)["id"] for m in sorted((RM / "shards").glob("*_meta.npz"))])
    member_pos = np.where(np.isin(pool_ids, rm_ids))[0].astype(np.int64)   # pool positions (< N_POOL)
    print(f"full corpus {N_full:,} (pool {N_POOL:,} + comp {comp_n:,}) | members {member_pos.size:,}", flush=True)

    n_new = n_target - member_pos.size
    assert 0 < n_new < N_full - member_pos.size, f"n_new {n_new} out of range"
    is_member_full = np.zeros(N_full, dtype=bool); is_member_full[member_pos] = True
    nonmember = np.where(~is_member_full)[0]                                # full positions available for the +new draw
    rng = np.random.default_rng(seed); perm = rng.permutation(nonmember.size)
    new_pos = nonmember[perm[:n_new]]
    sixm_pos = np.sort(np.concatenate([member_pos, new_pos]))
    assert sixm_pos.size == n_target and np.unique(sixm_pos).size == n_target, "6M union not clean"

    rest = nonmember[perm[n_new:]]
    rng2 = np.random.default_rng(seed + 1); vt = rng2.permutation(rest.size)
    val_pos = np.sort(rest[vt[:n_val]]); test_pos = np.sort(rest[vt[n_val:n_val + n_test]])
    assert len(np.intersect1d(val_pos, sixm_pos)) == 0 and len(np.intersect1d(test_pos, sixm_pos)) == 0
    assert len(np.intersect1d(val_pos, test_pos)) == 0

    member_mask = is_member_full[sixm_pos]
    assert member_mask.sum() == member_pos.size, "member nesting broken"

    # sscd_nn baseline — pool-portion only (complement has no sscd_nn); record composition
    sscd_pool = np.load(POOL / "sscd_nn.npy")
    sscd_6m = np.full(sixm_pos.size, np.nan, np.float64)
    pool_mask_6m = sixm_pos < N_POOL
    sscd_6m[pool_mask_6m] = sscd_pool[sixm_pos[pool_mask_6m]]
    sscd_fin = sscd_6m[~np.isnan(sscd_6m)]
    pcts = {f"p{p}": round(float(np.percentile(sscd_fin, p)), 5) for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)} if sscd_fin.size else {}
    n_pool_in_6m = int(pool_mask_6m.sum()); n_comp_in_6m = int((~pool_mask_6m).sum())

    np.save(OUT / "full_pos.npy", sixm_pos); np.save(OUT / "member_mask.npy", member_mask)
    np.save(OUT / "sscd_nn.npy", sscd_6m.astype(np.float32))
    np.save(OUT / "val_pos.npy", val_pos); np.save(OUT / "test_pos.npy", test_pos)
    # id.npy: pool rows -> pool id; complement rows -> complement id (load complement ids lazily only for those)
    ids = np.empty(sixm_pos.size, dtype=object)
    ids[pool_mask_6m] = pool_ids[sixm_pos[pool_mask_6m]]
    if n_comp_in_6m:
        comp_ids = np.load(COMP / "id.npy", allow_pickle=True)
        ids[~pool_mask_6m] = comp_ids[sixm_pos[~pool_mask_6m] - N_POOL]
    np.save(OUT / "id.npy", ids)

    pool_d = np.load(POOL / "dino1536.f16.npy", mmap_mode="r"); comp_d = np.load(COMP / "dino1536.f16.npy", mmap_mode="r")
    t0 = time.time()
    _gather_dino(pool_d, comp_d, sixm_pos, OUT / "dino-substrate.f16.npy")
    _gather_dino(pool_d, comp_d, val_pos, OUT / "val-dino.f16.npy")
    _gather_dino(pool_d, comp_d, test_pos, OUT / "test-dino.f16.npy")
    gather_wall = time.time() - t0

    manifest = {
        "schema": "monet-random-dino-6m-2026-09-06", "draw": "uniform-random from FULL 103.8M column, nested (members=random-2m)",
        "seed": seed, "n_target": n_target, "n_rows": int(sixm_pos.size),
        "n_members_nested": int(member_pos.size), "n_new": int(n_new),
        "composition": {"pool_rows": n_pool_in_6m, "complement_rows": n_comp_in_6m},
        "n_val": int(val_pos.size), "n_test": int(test_pos.size), "val_test_disjoint_from_6m": True,
        "gather_wall_s": round(gather_wall, 1),
        "row_order": "sorted FULL-corpus position (full_pos.npy); split at 19,344,847; member_mask marks the nested 2M",
        "sscd_nn_distribution": {"n_finite": int(sscd_fin.size), "note": "pool-portion only; complement rows have no sscd_nn (NaN)",
                                 "mean": round(float(sscd_fin.mean()), 5) if sscd_fin.size else None, "percentiles": pcts},
        "dino": {"file": "dino-substrate.f16.npy", "dim": DIM, "dtype": "float16", "unit_norm": True,
                 "col": "embedding_dinov2-vitg14", "gathered_from": "pool-20m/dino1536 + pool-complement-88m/dino1536"},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"6M DINO draw DONE: {sixm_pos.size:,} rows ({member_pos.size:,} nested + {n_new:,} new; "
          f"pool {n_pool_in_6m:,} / comp {n_comp_in_6m:,}), val {val_pos.size:,} test {test_pos.size:,}, gather {gather_wall:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
