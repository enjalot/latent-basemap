#!/usr/bin/env python3
"""Substrate-draw universality: THREE disjoint composition-matched 2M MiniLM slices (A/B/C)
(owner-approved 2026-08-30). CPU-only builder + full disjointness proofs.

Purpose: A1's head comparison used NESTED heads; this is the first INDEPENDENT-DRAW comparison.
Three 2M slices at the baseline 40/25/25/10 mix (fineweb/redpajama/pile/starcoder = 800k/500k/500k/200k),
drawn seed-42 but MUTUALLY DISJOINT and disjoint from the eval sets — BY CONSTRUCTION, not post-hoc:

  * one rng(42) draws 3x the per-corpus count in a SINGLE choice() per corpus, split first->A next->B
    next->C, so A/B/C are pairwise disjoint by construction (and NOT a rng-replay of the baseline draw).
  * the draw EXCLUDES, per corpus, the 2M baseline's own coords AND (for starcoder) the probe-code-heldout
    coords — so the slices can't accidentally reproduce the baseline or contaminate the code eval register.
  * base corpora (fineweb/rpj/pile/starcoder) are corpus codes 0-3; every social/common-corpus eval probe
    is a DIFFERENT corpus, so those are disjoint by corpus. Proven anyway where provenance exists.

Outputs per slice (/data/latent-basemap/substrates/draw-univ-{A,B,C}/): substrate.f32.npy (2M,384),
provenance.npy (corpus,shard,row), manifest.json (counts + ALL proofs). Plus draw-univ-proofs.json.
The three trains will share seed 42 -> init_state_sha256 MUST be equal (the experiment's validity gate,
checked at score time). NO train runs until these proofs are reviewed.
"""
from __future__ import annotations

import glob
import json
import os
import time
from pathlib import Path

import numpy as np

DIM = 384
SEED = 42
SUBSTRATES = Path("/data/latent-basemap/substrates")
E = "/data/embeddings"
CORPORA = {
    "fineweb":   f"{E}/fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2/train",
    "redpajama": f"{E}/RedPajama-Data-V2-sample-10B-chunked-120-all-MiniLM-L6-v2/train",
    "pile":      f"{E}/pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2/train",
    "starcoder": f"{E}/starcoderdata-code-chunked-120-all-MiniLM-L6-v2/train",
}
CORPUS_CODE = {"fineweb": 0, "redpajama": 1, "pile": 2, "starcoder": 3}
PER_SLICE = {"fineweb": 800_000, "redpajama": 500_000, "pile": 500_000, "starcoder": 200_000}
FILL_ORDER = ("fineweb", "redpajama", "pile", "starcoder")
SLICES = ("A", "B", "C")
PROV_DTYPE = np.dtype([("corpus", "u1"), ("shard", "<u2"), ("row", "<i8")])

BASELINE_PROV = Path("/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
                     "minilm-mixed-2m-substrate-and-exact-k15-graph-v1/provenance.npy")
CODE_HELDOUT_PROV = SUBSTRATES / "probe-code-heldout" / "provenance.npy"
A1_PROV = SUBSTRATES / "a1-common-neutral" / "provenance.npy"
# COORD-exclusion (same (corpus-code, sorted-shard, row) system as our draw): baseline (all corpora)
# + code-heldout (starcoder). a1-common-neutral draws from the SAME base corpora but with an
# INCOMPATIBLE coordinate system (its code ordering differs — verified: a1 code0 maxshard 149 vs
# fineweb's 99), so a1 CANNOT be coord-excluded; it is excluded by CONTENT (row-hash) instead, which
# is coordinate-system-independent. ~250k a1 rows over the pools => ~thousands of incidental content
# hits in a fresh 6M draw, so content-exclusion is REQUIRED, not cosmetic.
EXCLUDE_SOURCES = {
    "fineweb": [BASELINE_PROV],
    "redpajama": [BASELINE_PROV],
    "pile": [BASELINE_PROV],
    "starcoder": [BASELINE_PROV, CODE_HELDOUT_PROV],
}
# eval sets to PROVE disjointness against (exact where provenance exists). Slices are codes 0-3;
# social (reddit/ca/twitter/bluesky) + wiki/ccweb/ccscience have no provenance and are built from
# DIFFERENT pools (social / common-corpus), hence cross-corpus — reported as such, not silently scored.
EVAL_PROV = {
    "probe-code-heldout": CODE_HELDOUT_PROV,
    "a1-common-neutral": A1_PROV,
}


def _open_shard(path):
    with open(path, "rb") as fh:
        is_npy = fh.read(6) == b"\x93NUMPY"
    if is_npy:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    rows = os.path.getsize(path) // (DIM * 4)
    return np.memmap(path, dtype=np.float32, mode="r", shape=(rows, DIM))


def _corpus_shards(corpus):
    files = sorted(glob.glob(os.path.join(CORPORA[corpus], "*.npy")))
    if not files:
        raise FileNotFoundError(f"no shards for {corpus}")
    return [_open_shard(f) for f in files]


def _shard_offsets(shards):
    sr = [int(s.shape[0]) for s in shards]
    return np.concatenate([[0], np.cumsum(sr)]).astype(np.int64), sr


def _prov_globals(prov_path, corpus_code, shard_offsets):
    """Global indices (within the corpus's sorted-shard row space) of a provenance file's
    rows for one corpus code. Global = shard_offsets[shard] + row."""
    p = np.load(prov_path, mmap_mode="r", allow_pickle=False)
    m = p["corpus"] == corpus_code
    if not m.any():
        return np.empty(0, dtype=np.int64)
    return (shard_offsets[p["shard"][m].astype(np.int64)] + p["row"][m].astype(np.int64))


def _draw_distinct(rng, T, k):
    """k distinct uniform ints in [0,T) WITHOUT materializing a T-permutation (choice(replace=False)
    would allocate O(T) — fatal for the 227M pile pool). Batch-sample + unique + top-up, then a
    random k-subset (shuffle so it isn't index-ordered)."""
    acc = np.empty(0, dtype=np.int64)
    while acc.shape[0] < k:
        need = k - acc.shape[0]
        cand = rng.integers(0, T, size=int(need * 1.25) + 64, dtype=np.int64)
        acc = np.unique(np.concatenate([acc, cand]))
    rng.shuffle(acc)
    return acc[:k]


def _void_view(a):
    """(n, DIM) f32 -> (n,) void scalars for EXACT whole-row set ops (content identity,
    coordinate-system-independent)."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()


def _gather(shards, gidx, shard_offsets):
    """Gather rows at sorted global indices; per-shard ascending for memmap locality."""
    n = gidx.shape[0]
    shard_of = np.searchsorted(shard_offsets, gidx, side="right") - 1
    local = gidx - shard_offsets[shard_of]
    out = np.empty((n, DIM), dtype=np.float32)
    for si in range(len(shards)):
        mask = shard_of == si
        if not mask.any():
            continue
        loc = local[mask]
        order = np.argsort(loc)
        dest = np.nonzero(mask)[0][order]
        out[dest] = np.asarray(shards[si][loc[order]], dtype=np.float32)
    return out, shard_of.astype(np.uint16), local.astype(np.int64)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for s in SLICES:
        d = SUBSTRATES / f"draw-univ-{s}"
        if (d / "substrate.f32.npy").exists():
            raise SystemExit(f"REFUSE overwrite: {d}/substrate.f32.npy exists")
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    # a1-common-neutral rows as void scalars for EXACT content exclusion (coord-incompatible).
    a1_rows = np.asarray(np.load(A1_PROV.parent / "substrate.f32.npy", mmap_mode="r"), dtype=np.float32)
    a1_void = _void_view(a1_rows)
    print(f"  a1 content set: {a1_void.shape[0]:,} rows", flush=True)

    # --- per-corpus: draw (coord-excl) -> gather -> CONTENT-exclude a1 -> collect `need`, split A/B/C ---
    collected = {c: {} for c in FILL_ORDER}   # corpus -> slice -> (rows, shard_u16, local_i64)
    corpus_meta = {}
    for c in FILL_ORDER:
        shards = _corpus_shards(c)
        offs, sr = _shard_offsets(shards)
        T = int(offs[-1])
        code = CORPUS_CODE[c]
        excl = np.empty(0, dtype=np.int64)
        for src in EXCLUDE_SOURCES[c]:
            excl = np.union1d(excl, _prov_globals(src, code, offs))
        excl_sorted = np.unique(excl)
        need = PER_SLICE[c] * 3
        buf = int(need * 1.3) + excl_sorted.shape[0] + 200_000
        cand = _draw_distinct(rng, T, min(buf, T))
        cand = cand[~np.isin(cand, excl_sorted, assume_unique=False)]   # coord-exclude baseline/heldout
        rows, shard_of, local = _gather(shards, cand, offs)
        keep = ~np.isin(_void_view(rows), a1_void)                      # content-exclude a1
        a1_hits = int((~keep).sum())
        rows, shard_of, local = rows[keep][:need], shard_of[keep][:need], local[keep][:need]
        if rows.shape[0] < need:
            raise SystemExit(f"{c}: only {rows.shape[0]} after exclusions, need {need} (raise buf)")
        for i, s in enumerate(SLICES):
            sl = slice(i * PER_SLICE[c], (i + 1) * PER_SLICE[c])   # random partition (_draw_distinct shuffled)
            collected[c][s] = (rows[sl], shard_of[sl], local[sl])
        corpus_meta[c] = {"pool_rows": T, "coord_excluded": int(excl_sorted.shape[0]),
                          "a1_content_hits_removed": a1_hits, "per_slice": PER_SLICE[c], "shards": len(sr)}
        print(f"  {c}: pool {T:,} coord-excl {excl_sorted.shape[0]:,} a1-content-hits {a1_hits} "
              f"-> 3x{PER_SLICE[c]:,}", flush=True)
        del rows, shard_of, local, cand

    # --- assemble + write each slice (FILL_ORDER concat) ---
    slice_prov, slice_void = {}, {}
    for s in SLICES:
        d = SUBSTRATES / f"draw-univ-{s}"
        d.mkdir(parents=True, exist_ok=True)
        parts, provs = [], []
        for c in FILL_ORDER:
            rows, shard_of, local = collected[c][s]
            parts.append(rows)
            pr = np.empty(rows.shape[0], dtype=PROV_DTYPE)
            pr["corpus"] = CORPUS_CODE[c]; pr["shard"] = shard_of; pr["row"] = local
            provs.append(pr)
        sub = np.concatenate(parts)
        prov = np.concatenate(provs)
        assert sub.shape == (2_000_000, DIM), sub.shape
        np.save(d / "substrate.f32.npy", sub)
        np.save(d / "provenance.npy", prov)
        slice_prov[s] = prov
        slice_void[s] = _void_view(sub)
        print(f"  slice {s}: wrote substrate {sub.shape} + provenance ({prov.shape[0]:,})", flush=True)

    # --- PROOFS ---
    def _keys(prov):
        return set(zip(prov["corpus"].tolist(), prov["shard"].tolist(), prov["row"].tolist()))
    keysets = {s: _keys(slice_prov[s]) for s in SLICES}
    pairwise = {}
    for a, b in (("A", "B"), ("A", "C"), ("B", "C")):
        pairwise[f"{a}∩{b}"] = len(keysets[a] & keysets[b])
    eval_overlap = {}
    # probe-code-heldout: coord-compatible -> exact (corpus,shard,row) intersection.
    ep = np.load(CODE_HELDOUT_PROV, mmap_mode="r", allow_pickle=False)
    eks = _keys(ep)
    eval_overlap["probe-code-heldout"] = {"method": "coord", **{s: len(keysets[s] & eks) for s in SLICES}}
    # a1-common-neutral: coord-INCOMPATIBLE -> exact CONTENT (void-row) intersection.
    eval_overlap["a1-common-neutral"] = {"method": "content",
                                         **{s: int(np.isin(slice_void[s], a1_void).sum()) for s in SLICES}}
    # cross-corpus safety: which corpus codes appear in slices vs the eval provenances
    slice_codes = sorted(set(int(c) for s in SLICES for c in np.unique(slice_prov[s]["corpus"])))
    proofs = {
        "schema": "draw-univ-proofs-2026-08-30", "seed": SEED,
        "pairwise_slice_intersections": pairwise,
        "pairwise_disjoint": all(v == 0 for v in pairwise.values()),
        "eval_set_overlap": eval_overlap,
        "eval_disjoint": all(all(v[s] == 0 for s in SLICES) for v in eval_overlap.values()),
        "slice_corpus_codes": slice_codes,
        "note": ("base slices are corpus codes 0-3 (fineweb/rpj/pile/starcoder). probe-code-heldout is "
                 "starcoder(3) — proven disjoint by EXACT coord (corpus,shard,row) intersection. "
                 "a1-common-neutral draws from the same base corpora but with an INCOMPATIBLE coord "
                 "system, so it is CONTENT-excluded (exact void-row equality) and its overlap is proven "
                 "by content. social probes (reddit/ca/twitter/bluesky) + wiki/ccweb/ccscience are from "
                 "DIFFERENT pools (social/common-corpus), disjoint by corpus, no base-pool coords. "
                 "Baseline coords coord-excluded per corpus so slices are independent draws, not a "
                 "seed-42 replay of the baseline. Same seed 42 for all three trains -> init hashes MUST "
                 "match (validity gate, checked at score time)."),
        "corpus_meta": corpus_meta, "wall_s": round(time.time() - t0, 1),
    }
    (SUBSTRATES / "draw-univ-proofs.json").write_text(json.dumps(proofs, indent=1))
    for s in SLICES:
        man = {"slice": s, "rows": 2_000_000, "mix": PER_SLICE, "seed": SEED,
               "pairwise_slice_intersections": pairwise,
               "eval_set_overlap": {k: {"method": v["method"], "overlap": v[s]} for k, v in eval_overlap.items()},
               "corpus_meta": corpus_meta}
        (SUBSTRATES / f"draw-univ-{s}" / "manifest.json").write_text(json.dumps(man, indent=1))

    print("\n=== PROOFS ===", flush=True)
    print(f"  pairwise A∩B/A∩C/B∩C: {pairwise} -> disjoint={proofs['pairwise_disjoint']}", flush=True)
    print(f"  eval overlap: {eval_overlap} -> disjoint={proofs['eval_disjoint']}", flush=True)
    print(f"  slice corpus codes: {slice_codes}", flush=True)
    print(f"\nwrote draw-univ-proofs.json + 3 manifests ({proofs['wall_s']}s)", flush=True)
    return 0 if (proofs["pairwise_disjoint"] and proofs["eval_disjoint"]) else 3


if __name__ == "__main__":
    raise SystemExit(main())
