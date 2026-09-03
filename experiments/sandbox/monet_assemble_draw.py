"""Assemble a MONET draw arm's CLIP-512 substrate (owner MONET item 3). CPU. For random/sscd, computes the
idx here (deterministic, identical to p_monet_draws) if absent; for annfaiss/theirfaiss, reads the idx that
p_monet_draws wrote. Then gathers pool clip512[idx] -> /data2/monet/draws/<arm>-clip.f32.npy (chunked read
of the sorted idx to keep the 39GB memmap access sequential-ish). Usage: monet_assemble_draw.py <arm>."""
import sys
from pathlib import Path
import numpy as np

POOL = Path("/data2/monet/pool-20m"); OUT = Path("/data2/monet/draws")
N_DRAW = 2_000_000; SEED = 42


def _gumbel_topk(logw, k, rng):
    g = -np.log(-np.log(rng.random(logw.shape[0]).astype(np.float64) + 1e-12) + 1e-12)
    return np.argpartition(-(logw + g), k)[:k]


def main():
    arm = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    idxf = OUT / f"{arm}.idx.npy"
    if not idxf.exists():
        # rarity per arm: random=uniform, sscd=1-sscd_nn, annfaiss/theirfaiss=rarity_<arm>.npy (density step)
        rf = OUT / f"rarity_{arm}.npy"
        if arm == "random":
            rar = np.ones(np.load(POOL / "sscd_nn.npy").shape[0], np.float32)
        elif arm == "sscd":
            rar = np.clip(1.0 - np.load(POOL / "sscd_nn.npy"), 1e-4, None).astype(np.float32)
        elif rf.exists():
            rar = np.load(rf).astype(np.float32)         # annfaiss (cuVS) / theirfaiss (their CPU index)
        else:
            raise SystemExit(f"{arm}: no idx and no rarity_{arm}.npy — produced by the density step; refuse to fabricate")
        rng = np.random.default_rng(hash((SEED, arm)) % (2**32))
        idx = np.sort(_gumbel_topk(np.log(np.clip(rar, 1e-8, None).astype(np.float64)), N_DRAW, rng)).astype(np.int64)
        np.save(idxf, idx)
    idx = np.load(idxf)
    out = OUT / f"{arm}-clip.f32.npy"
    if out.exists() and np.load(out, mmap_mode="r").shape[0] == idx.shape[0]:
        print(f"{arm}: substrate present ({idx.shape[0]:,})"); return 0
    clip = np.load(POOL / "clip512.f32.npy", mmap_mode="r")
    sub = np.empty((idx.shape[0], clip.shape[1]), np.float32)
    B = 100_000
    for s in range(0, idx.shape[0], B):
        sub[s:s+B] = clip[idx[s:s+B]]                        # idx sorted -> mostly sequential
    np.save(out, sub)
    print(f"{arm}: assembled {sub.shape} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
