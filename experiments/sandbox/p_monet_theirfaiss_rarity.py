"""Compute the theirfaiss draw-arm rarity ONLY (owner MONET, reorder 2026-09-03): search the publisher's
IVF-PQ (nprobe=64) with the 19.3M pool CLIP-512, kNN mean-distance = rarity. CPU (their index stays CPU by
design — it IS the publisher's production index). Writes rarity_theirfaiss.npy. Reuses p_monet_draws._rarity_theirfaiss."""
import sys, time; sys.path.insert(0, 'experiments/sandbox')
import numpy as np
from pathlib import Path
from p_monet_draws import _rarity_theirfaiss
POOL=Path("/data2/monet/pool-20m"); OUT=Path("/data2/monet/draws"); OUT.mkdir(exist_ok=True)
clip=np.load(POOL/"clip512.f32.npy", mmap_mode="r")
t0=time.time()
r=_rarity_theirfaiss(clip, 16, OUT/"rarity_theirfaiss.npy")
print(f"theirfaiss rarity DONE: {r.shape} in {time.time()-t0:.0f}s -> {OUT/'rarity_theirfaiss.npy'}", flush=True)
