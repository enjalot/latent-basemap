#!/usr/bin/env python3
"""Image-space universality: FOUR disjoint 2M sisap-CLIP slices A/B/C (heads) + D (shared eval)
(owner-approved 2026-08-30). CPU-only builder + disjointness proofs.

Replicates the MiniLM substrate-draw universality on CLIP768 (SISAP 30M pool). Homogeneous space
(no composition mix, no pool-drawn eval probes), so the build is a plain one-permutation partition:
one rng(42) draws 8M distinct indices from the 30M pool, split first 2M→A, next→B, next→C, next→D.
A/B/C/D are pairwise disjoint BY CONSTRUCTION. D is the shared neutral eval (disjoint from the 3
heads). Overlap with the existing every-15th sisap-clip-2m stride substrate is RECORDED (not excluded)
— that stride is not an eval set. Validity gate (at score time): the three HEAD init_state_sha256 must
be equal (D768 → different hash from MiniLM; equality-across-three is the gate).

Outputs (/data/latent-basemap/substrates/img-univ-{A,B,C,D}/): substrate.f32.npy (2M,768),
provenance.npy (int64 global row index into the 30M pool), manifest.json. Plus img-univ-proofs.json.
"""
import json
import time
from pathlib import Path

import numpy as np

H5 = "/data/embeddings/laion2b-en-clip768v2-sisap/laion2B-en-clip768v2-n=30M.h5"
DIM = 768
SEED = 42
PER_SLICE = 2_000_000
SLICES = ("A", "B", "C", "D")   # A/B/C heads, D shared eval
STRIDE = 15                     # existing sisap-clip-2m = pool[::15]
SUBSTRATES = Path("/data/latent-basemap/substrates")


def _draw_distinct(rng, T, k):
    acc = np.empty(0, dtype=np.int64)
    while acc.shape[0] < k:
        need = k - acc.shape[0]
        cand = rng.integers(0, T, size=int(need * 1.25) + 64, dtype=np.int64)
        acc = np.unique(np.concatenate([acc, cand]))
    rng.shuffle(acc)
    return acc[:k]


def main():
    import h5py
    for s in SLICES:
        if (SUBSTRATES / f"img-univ-{s}" / "substrate.f32.npy").exists():
            raise SystemExit(f"REFUSE overwrite: img-univ-{s}")
    t0 = time.time()
    with h5py.File(H5, "r") as f:
        emb = f["emb"]
        T = int(emb.shape[0])
        assert int(emb.shape[1]) == DIM, emb.shape
        rng = np.random.default_rng(SEED)
        need = PER_SLICE * len(SLICES)
        idx = _draw_distinct(rng, T, need)
        draws = {s: np.sort(idx[i * PER_SLICE:(i + 1) * PER_SLICE]) for i, s in enumerate(SLICES)}
        print(f"pool {T:,}; drew {need:,} distinct -> 4x{PER_SLICE:,}", flush=True)
        prov = {}
        for s in SLICES:
            d = SUBSTRATES / f"img-univ-{s}"; d.mkdir(parents=True, exist_ok=True)
            g = draws[s]
            out = np.lib.format.open_memmap(d / "substrate.f32.npy", mode="w+",
                                            dtype=np.float32, shape=(PER_SLICE, DIM))
            CH = 250_000
            for i in range(0, PER_SLICE, CH):
                sub_idx = g[i:i + CH]  # sorted increasing -> valid h5 fancy index
                out[i:i + len(sub_idx)] = np.asarray(emb[sub_idx], dtype=np.float32)
            out.flush(); del out
            np.save(d / "provenance.npy", g.astype(np.int64))
            prov[s] = g
            print(f"  slice {s}: wrote {PER_SLICE:,} rows", flush=True)

    # PROOFS: pairwise index intersections (single-source -> plain int64 sets)
    ksets = {s: set(prov[s].tolist()) for s in SLICES}
    pairwise = {}
    for i, a in enumerate(SLICES):
        for b in SLICES[i + 1:]:
            pairwise[f"{a}∩{b}"] = len(ksets[a] & ksets[b])
    # RECORD overlap with the existing every-15th stride substrate (index % 15 == 0). Acceptable.
    stride_overlap = {s: int((prov[s] % STRIDE == 0).sum()) for s in SLICES}
    proofs = {
        "schema": "img-univ-proofs-2026-08-30", "seed": SEED, "pool": H5, "pool_rows": T,
        "pairwise_slice_intersections": pairwise,
        "pairwise_disjoint": all(v == 0 for v in pairwise.values()),
        "existing_stride_overlap": {"note": ("count of slice rows that fall on the every-15th "
                                             "sisap-clip-2m stride; RECORDED not excluded (stride is "
                                             "not an eval set)."), **stride_overlap},
        "roles": {"A": "head", "B": "head", "C": "head", "D": "shared neutral eval"},
        "validity_gate_note": ("same seed 42 for the 3 HEAD trains -> init_state_sha256 must be equal "
                               "across A/B/C (D768 -> different hash from MiniLM's cb6fb9a9)."),
        "wall_s": round(time.time() - t0, 1),
    }
    (SUBSTRATES / "img-univ-proofs.json").write_text(json.dumps(proofs, indent=1))
    for s in SLICES:
        (SUBSTRATES / f"img-univ-{s}" / "manifest.json").write_text(json.dumps(
            {"slice": s, "role": proofs["roles"][s], "rows": PER_SLICE, "dim": DIM, "seed": SEED,
             "pairwise_slice_intersections": pairwise,
             "existing_stride_overlap": stride_overlap[s], "pool": H5}, indent=1))
    print(f"\n=== PROOFS ===\n  pairwise {pairwise} -> disjoint={proofs['pairwise_disjoint']}", flush=True)
    print(f"  existing-stride overlap {stride_overlap} (recorded, acceptable)", flush=True)
    print(f"wrote img-univ-proofs.json + 4 manifests ({proofs['wall_s']}s)", flush=True)
    return 0 if proofs["pairwise_disjoint"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
