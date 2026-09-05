"""SSCD-weighted 4M draw — the GROWTH comparison (owner standing-queue #1, 2026-09-05). CPU, local pool.
Single-variable contrast vs the gated random-4M: the 2M BASE is identical (random-2m members, nested);
the ADDED 2M is drawn sscd-rarity-weighted (Gumbel-top-k) instead of uniform-random. Addition policy is the
ONLY variable -> isolates "how should a service grow a map: random vs diversity-weighted additions."

sscd_nn direction (confirmed 2026-09-02): HIGH = near-duplicate, LOW = rare/diverse. Diversity UPWEIGHTS LOW
sscd_nn. Gumbel-top-k without replacement over the held-out complement: key = (-sscd_nn / T) + Gumbel(0,1),
take top n_add. NaN sscd_nn EXCLUDED (documented). Records the ACHIEVED sscd_nn distribution vs the random-4M
uniform baseline (already in random-clip-4m/manifest.json: mean 0.451 / p50 0.409).

Common D3-style holdout: val/test drawn OUTSIDE BOTH 4M unions (pool minus (random-4M ∪ sscd-4M)) so both
heads are evaluated on identical genuinely-unseen rows. Output: /data2/monet/sscd-clip-4m/ (mirrors random-clip-4m).
Usage: monet_draw_4m_sscd.py [N_TARGET=4000000] [SEED=42] [T=0.1] [N_VAL=200000] [N_TEST=200000].
"""
import json, sys, time
from pathlib import Path
import numpy as np

POOL = Path("/data2/monet/pool-20m"); RM = Path("/data2/monet/random-2m")
RAND4M = Path("/data2/monet/random-clip-4m"); OUT = Path("/data2/monet/sscd-clip-4m")
CHUNK = 200_000


def _gather_stream(pool_clip, positions, out_path):
    n = positions.shape[0]
    mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n, 512))
    for i in range(0, n, CHUNK):
        j = min(i + CHUNK, n); mm[i:j] = np.asarray(pool_clip[positions[i:j]], np.float32)
    mm.flush(); return n


def main():
    n_target = int(sys.argv[1]) if len(sys.argv) > 1 else 4_000_000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    T = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    n_val = int(sys.argv[4]) if len(sys.argv) > 4 else 200_000
    n_test = int(sys.argv[5]) if len(sys.argv) > 5 else 200_000
    OUT.mkdir(parents=True, exist_ok=True)

    pool_ids = np.load(POOL / "id.npy", allow_pickle=True); N = pool_ids.shape[0]
    rm_ids = np.concatenate([np.load(m, allow_pickle=True)["id"] for m in sorted((RM / "shards").glob("*_meta.npz"))])
    is_member = np.isin(pool_ids, rm_ids)
    member_pos = np.where(is_member)[0]; heldout_pos = np.where(~is_member)[0]
    sscd = np.load(POOL / "sscd_nn.npy").astype(np.float64)

    # sscd-weighted addition via Gumbel-top-k over the held-out complement (NaN excluded)
    n_add = n_target - member_pos.size
    h_sscd = sscd[heldout_pos]
    valid = ~np.isnan(h_sscd)
    cand = heldout_pos[valid]; cand_sscd = h_sscd[valid]
    rng = np.random.default_rng(seed)
    logw = -cand_sscd / T                                   # LOW sscd (rare) -> HIGH weight
    g = rng.gumbel(size=cand.shape[0])
    keys = logw + g
    take = np.argpartition(-keys, n_add)[:n_add]            # top n_add without replacement
    sscd_add = cand[take]
    sscd_4m_pos = np.sort(np.concatenate([member_pos, sscd_add]))
    assert sscd_4m_pos.size == n_target and np.unique(sscd_4m_pos).size == n_target
    member_mask = is_member[sscd_4m_pos]
    assert member_mask.sum() == member_pos.size, "member nesting broken"

    # common holdout: outside BOTH random-4M and sscd-4M unions
    rand4m_pos = np.load(RAND4M / "pool_pos.npy")
    union_both = np.union1d(rand4m_pos, sscd_4m_pos)
    common_holdout = np.setdiff1d(np.arange(N), union_both, assume_unique=True)
    rng2 = np.random.default_rng(seed + 1); perm = rng2.permutation(common_holdout.size)
    val_pos = np.sort(common_holdout[perm[:n_val]]); test_pos = np.sort(common_holdout[perm[n_val:n_val + n_test]])
    assert len(np.intersect1d(val_pos, union_both)) == 0 and len(np.intersect1d(test_pos, union_both)) == 0

    # achieved sscd_nn distribution of the addition + full draw (NaN-excluded)
    add_s = sscd[sscd_add]; add_fin = add_s[~np.isnan(add_s)]
    full_s = sscd[sscd_4m_pos]; full_fin = full_s[~np.isnan(full_s)]
    def _pcts(a): return {f"p{p}": round(float(np.percentile(a, p)), 5) for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)}

    np.save(OUT / "pool_pos.npy", sscd_4m_pos); np.save(OUT / "member_mask.npy", member_mask)
    np.save(OUT / "id.npy", pool_ids[sscd_4m_pos]); np.save(OUT / "sscd_nn.npy", sscd[sscd_4m_pos].astype(np.float32))
    np.save(OUT / "val_pos.npy", val_pos); np.save(OUT / "test_pos.npy", test_pos)
    np.save(OUT / "common_holdout_pos.npy", common_holdout)

    pool_clip = np.load(POOL / "clip512.f32.npy", mmap_mode="r")
    t0 = time.time()
    _gather_stream(pool_clip, sscd_4m_pos, OUT / "clip-substrate.f32.npy")
    _gather_stream(pool_clip, val_pos, OUT / "val-clip.f32.npy")
    _gather_stream(pool_clip, test_pos, OUT / "test-clip.f32.npy")
    gather = time.time() - t0

    manifest = {
        "schema": "monet-sscd-clip-4m-2026-09-05", "draw": "sscd-rarity-weighted addition (Gumbel-top-k), nested (members = random-2m)",
        "single_variable_vs": "random-clip-4m (identical 2M base; addition policy is the only difference)",
        "seed": seed, "gumbel_temperature_T": T, "sscd_direction": "LOW sscd_nn=rare -> HIGH weight (log_w = -sscd_nn/T); NaN excluded",
        "n_target": n_target, "n_rows": int(sscd_4m_pos.size), "n_members_nested": int(member_pos.size), "n_add": int(n_add),
        "n_val": int(val_pos.size), "n_test": int(test_pos.size),
        "common_holdout_outside_BOTH_unions": int(common_holdout.size),
        "addition_sscd_nn": {"mean": round(float(add_fin.mean()), 5), "percentiles": _pcts(add_fin), "n_finite": int(add_fin.size),
                             "vs_random_baseline": "random-4M uniform addition sscd_nn ~mean 0.451/p50 0.409 (see random-clip-4m/manifest)"},
        "full_draw_sscd_nn": {"mean": round(float(full_fin.mean()), 5), "percentiles": _pcts(full_fin)},
        "clip": {"file": "clip-substrate.f32.npy", "dim": 512, "dtype": "float32", "n_rows": int(sscd_4m_pos.size), "unit_norm": True},
        "gather_wall_s": round(gather, 1)}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"sscd-4M DONE: {sscd_4m_pos.size:,} ({member_pos.size:,} nested + {n_add:,} sscd-weighted), "
          f"addition sscd_nn mean {add_fin.mean():.4f} (vs random ~0.451), common holdout {common_holdout.size:,}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
