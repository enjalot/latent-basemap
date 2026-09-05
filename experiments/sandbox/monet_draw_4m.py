"""Nested 4M random CLIP draw for the image-phase 4M champion (owner ruling 2026-09-05: RANDOM, uniform,
deterministic, nested 2M subset 4M). CPU-only, gathered from the LOCAL pool (no download). Keeps the 2M-vs-4M
comparison clean and the <=0.02 regression gate honest.

Construction (seed-42, deterministic):
  members  = the exact random-2m rows (id-join into the pool)            -> 2,008,321 rows (the nested 2M)
  new      = uniform draw of (4,000,000 - members) rows from the HELD-OUT pool complement, seed 42
  4M       = sort(members ∪ new)                                          -> 4,000,000 pool positions
  val/test = disjoint uniform draws (seed 43) from the pool OUTSIDE the 4M union -> 200K each, for the
             regression gate + reception eval on rows the 4M never saw.

Records (for the future sscd-weighted-draw comparison, overseer ask): the 4M draw's sscd_nn distribution
(percentiles + histogram) as the documented uniform baseline. Also the nested member_mask (which 4M rows are
the 2M), pool positions, ids. Substrate gathered in sorted-position chunks (memmap locality; >=2GB rule).

Output: /data2/monet/random-clip-4m/{clip-substrate.f32.npy, pool_pos.npy, member_mask.npy, id.npy,
sscd_nn.npy, val_pos.npy, test_pos.npy, val-clip.f32.npy, test-clip.f32.npy, manifest.json}.
Usage: monet_draw_4m.py [N_TARGET=4000000] [SEED=42] [N_VAL=200000] [N_TEST=200000].
"""
import json, sys, time
from pathlib import Path
import numpy as np

POOL = Path("/data2/monet/pool-20m")
RM = Path("/data2/monet/random-2m")
OUT = Path("/data2/monet/random-clip-4m")
CHUNK = 200_000


def _gather_stream(pool_clip, positions, out_path):
    """Write pool_clip[positions] (positions SORTED) to a fresh memmap in chunks — never materializes >CHUNK rows."""
    n = positions.shape[0]
    mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n, 512))
    for i in range(0, n, CHUNK):
        j = min(i + CHUNK, n)
        mm[i:j] = np.asarray(pool_clip[positions[i:j]], np.float32)
    mm.flush()
    return n


def main():
    n_target = int(sys.argv[1]) if len(sys.argv) > 1 else 4_000_000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    n_val = int(sys.argv[3]) if len(sys.argv) > 3 else 200_000
    n_test = int(sys.argv[4]) if len(sys.argv) > 4 else 200_000
    OUT.mkdir(parents=True, exist_ok=True)

    pool_ids = np.load(POOL / "id.npy", allow_pickle=True)
    N = pool_ids.shape[0]
    rm_ids = np.concatenate([np.load(m, allow_pickle=True)["id"] for m in sorted((RM / "shards").glob("*_meta.npz"))])
    is_member = np.isin(pool_ids, rm_ids)
    member_pos = np.where(is_member)[0]
    heldout_pos = np.where(~is_member)[0]
    print(f"pool {N:,} | members {member_pos.size:,} | held-out {heldout_pos.size:,}", flush=True)

    n_new = n_target - member_pos.size
    assert 0 < n_new <= heldout_pos.size, f"n_new {n_new} out of range"
    rng = np.random.default_rng(seed)
    perm = rng.permutation(heldout_pos.size)             # deterministic uniform draw over the complement
    new_pos = heldout_pos[perm[:n_new]]
    fourm_pos = np.sort(np.concatenate([member_pos, new_pos]))
    assert fourm_pos.size == n_target and np.unique(fourm_pos).size == n_target, "4M union not clean"

    # val/test from the pool OUTSIDE the 4M union (seed 43, disjoint from each other)
    rest = heldout_pos[perm[n_new:]]                     # complement rows not in the 4M
    rng2 = np.random.default_rng(seed + 1)
    vt = rng2.permutation(rest.size)
    val_pos = np.sort(rest[vt[:n_val]]); test_pos = np.sort(rest[vt[n_val:n_val + n_test]])
    assert len(np.intersect1d(val_pos, fourm_pos)) == 0 and len(np.intersect1d(test_pos, fourm_pos)) == 0
    assert len(np.intersect1d(val_pos, test_pos)) == 0

    # nested member_mask over the 4M rows (True = one of the original 2M)
    member_mask = is_member[fourm_pos]
    assert member_mask.sum() == member_pos.size, "member nesting broken"

    # sscd_nn distribution of the 4M draw (documented uniform baseline)
    sscd = np.load(POOL / "sscd_nn.npy")
    sscd_4m = sscd[fourm_pos].astype(np.float64)
    nan_frac = float(np.isnan(sscd_4m).mean())          # pool sscd_nn has ~0.4% NaN (no near-dup computed)
    sscd_fin = sscd_4m[~np.isnan(sscd_4m)]
    pcts = {f"p{p}": round(float(np.percentile(sscd_fin, p)), 5) for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)}
    hist, edges = np.histogram(sscd_fin, bins=20, range=(0.0, 1.0))

    # persist provenance + substrate
    np.save(OUT / "pool_pos.npy", fourm_pos); np.save(OUT / "member_mask.npy", member_mask)
    np.save(OUT / "id.npy", pool_ids[fourm_pos]); np.save(OUT / "sscd_nn.npy", sscd[fourm_pos].astype(np.float32))
    np.save(OUT / "val_pos.npy", val_pos); np.save(OUT / "test_pos.npy", test_pos)

    pool_clip = np.load(POOL / "clip512.f32.npy", mmap_mode="r")
    t0 = time.time()
    _gather_stream(pool_clip, fourm_pos, OUT / "clip-substrate.f32.npy")
    _gather_stream(pool_clip, val_pos, OUT / "val-clip.f32.npy")
    _gather_stream(pool_clip, test_pos, OUT / "test-clip.f32.npy")
    gather_wall = time.time() - t0

    manifest = {
        "schema": "monet-random-clip-4m-2026-09-05", "draw": "uniform-random, nested (members = random-2m)",
        "seed": seed, "n_target": n_target, "n_rows": int(fourm_pos.size),
        "n_members_nested": int(member_pos.size), "n_new": int(n_new),
        "n_val": int(val_pos.size), "n_test": int(test_pos.size),
        "val_test_disjoint_from_4m": True, "gather_wall_s": round(gather_wall, 1),
        "row_order": "sorted pool position (pool_pos.npy); member_mask marks the nested 2M rows",
        "sscd_nn_distribution": {"mean": round(float(sscd_fin.mean()), 5), "std": round(float(sscd_fin.std()), 5),
                                 "nan_frac": round(nan_frac, 5), "n_finite": int(sscd_fin.size), "percentiles": pcts,
                                 "hist_counts": hist.tolist(), "hist_edges": [round(float(e), 3) for e in edges],
                                 "note": "uniform-draw baseline (NaN sscd_nn excluded); a future sscd-weighted draw compares coverage against this"},
        "clip": {"file": "clip-substrate.f32.npy", "dim": 512, "dtype": "float32", "n_rows": int(fourm_pos.size),
                 "unit_norm": True, "col": "embedding_clip-vit-base-patch32"},
        "provenance": "pool_pos.npy indexes pool-20m rows; id.npy / thumb join identical to the pool.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"4M draw DONE: {fourm_pos.size:,} rows ({member_pos.size:,} nested + {n_new:,} new), "
          f"val {val_pos.size:,} test {test_pos.size:,}, gather {gather_wall:.0f}s", flush=True)
    print(f"  sscd_nn: mean {sscd_4m.mean():.4f} p50 {pcts['p50']} p90 {pcts['p90']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
