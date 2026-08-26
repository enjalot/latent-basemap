#!/usr/bin/env python3
"""A1 cross-scale common-probe audit, step 1 (CPU): freeze ONE common sample.

The A1 audit asks a single operational question: does a cheap-N MiniLM
parametric-UMAP head match the 100M head as a function of training scale? To
answer it fairly every surviving head must be scored on the SAME frozen probe
sample against the SAME frozen truth graph — so no head gets a home-field
advantage from a probe drawn near its own training distribution.

This script freezes that common sample. It draws 250,000 rows (seed 42, random
without replacement) from the sealed 2M MiniLM pool substrate
(round-0216/queue-correction-3 ...minilm-mixed-2m...-v1/substrate.f32.npy).
That pool is the shared origin of every head's training substrate — all heads
were trained on pool-derived samples — so a sample of the pool is the neutral
common instrument. The orchestrator then runs the register pipeline's knn+fuzzy
on this substrate to freeze the a1-common truth graph
(sandbox/a1-common/edges-k15-fuzzy.npz), and a1_audit.py projects every head
through it.

NOTE the heads span recipe generations (2M/6.25M knob-sandbox winners through
the 50M/100M round checkpoints); the operational scale question is unaffected —
each head is a frozen map and is judged only on how faithfully it receives this
common sample.

CPU only (a memmap gather + one .npy write, seconds). Run:
    build_a1_common.py
Output:
    /data/latent-basemap/substrates/a1-common/substrate.f32.npy   (250000 x 384 f32)
    /data/latent-basemap/substrates/a1-common/manifest.json
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np

SEED = 42
N_SAMPLE = 250_000
POOL = Path(
    "/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
    "minilm-mixed-2m-substrate-and-exact-k15-graph-v1/substrate.f32.npy")
OUT_DIR = Path("/data/latent-basemap/substrates/a1-common")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pool = np.load(POOL, mmap_mode="r")
    n_pool, dim = int(pool.shape[0]), int(pool.shape[1])
    if N_SAMPLE > n_pool:
        raise SystemExit(f"pool has only {n_pool:,} rows < {N_SAMPLE:,} requested")

    rng = np.random.default_rng(SEED)
    idx = rng.choice(n_pool, size=N_SAMPLE, replace=False)
    idx.sort()   # ascending for efficient memmap gather; the SET is the random draw
    sample = np.ascontiguousarray(np.asarray(pool[idx], dtype=np.float32))

    out = OUT_DIR / "substrate.f32.npy"
    np.save(out, sample)

    manifest = {
        "schema": "a1-common-2026-08-26",
        "purpose": "frozen common probe sample for the A1 cross-scale head audit",
        "source_pool": str(POOL),
        "source_pool_rows": n_pool,
        "seed": SEED,
        "n_sample": N_SAMPLE,
        "dim": dim,
        "shape": list(sample.shape),
        "dtype": str(sample.dtype),
        "sampling": "np.random.default_rng(42).choice(replace=False), indices sorted ascending",
        "substrate": str(out),
        "truth_graph_pending": "/data/latent-basemap/sandbox/a1-common/edges-k15-fuzzy.npz",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "note": "The pool is the shared origin of every A1 head's training "
                "substrate, so a random sample of it is the neutral common "
                "instrument. Heads span recipe generations; the operational "
                "scale question (cheap-N vs 100M head) is unaffected.",
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))

    print(f"a1-common: wrote {out} shape={sample.shape} dtype={sample.dtype}")
    print(f"  sampled {N_SAMPLE:,} of {n_pool:,} pool rows (seed {SEED})")
    print(f"  manifest: {OUT_DIR / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
