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
# a1-common-neutral draws from the SAME base corpora (codes 0-3) as the slices — NOT common-corpus —
# so it MUST be excluded from the draw (verified: its corpus codes are [0,1,2,3]). code-heldout is
# starcoder(3) only. Every base corpus excludes the baseline + a1; starcoder also excludes code-heldout.
EXCLUDE_SOURCES = {
    "fineweb": [BASELINE_PROV, A1_PROV],
    "redpajama": [BASELINE_PROV, A1_PROV],
    "pile": [BASELINE_PROV, A1_PROV],
    "starcoder": [BASELINE_PROV, A1_PROV, CODE_HELDOUT_PROV],
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

    # --- per-corpus disjoint 3x draw from the complement of excluded coords ---
    draws = {c: {} for c in FILL_ORDER}   # corpus -> {slice -> global idx array}
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
        # draw a buffer (need + room to survive excluded-coord removal), filter, take `need`
        buf = need + excl_sorted.shape[0] + 100_000
        cand = _draw_distinct(rng, T, min(buf, T))
        keep = cand[~np.isin(cand, excl_sorted, assume_unique=False)][:need]
        if keep.shape[0] < need:
            raise SystemExit(f"{c}: only {keep.shape[0]} after exclusion, need {need}")
        for i, s in enumerate(SLICES):
            draws[c][s] = np.sort(keep[i * PER_SLICE[c]:(i + 1) * PER_SLICE[c]])
        corpus_meta[c] = {"pool_rows": T, "excluded": int(excl_sorted.shape[0]),
                          "per_slice": PER_SLICE[c], "shards": len(sr)}
        print(f"  {c}: pool {T:,} excl {excl_sorted.shape[0]:,} -> 3x{PER_SLICE[c]:,} drawn", flush=True)

    # --- assemble + write each slice ---
    slice_prov = {}
    for s in SLICES:
        d = SUBSTRATES / f"draw-univ-{s}"
        d.mkdir(parents=True, exist_ok=True)
        parts, provs = [], []
        for c in FILL_ORDER:
            shards = _corpus_shards(c)
            offs, _ = _shard_offsets(shards)
            rows, shard_of, local = _gather(shards, draws[c][s], offs)
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
        print(f"  slice {s}: wrote substrate {sub.shape} + provenance ({prov.shape[0]:,})", flush=True)
        del sub, parts

    # --- PROOFS ---
    def _keys(prov):
        # pack (corpus,shard,row) -> a python set of tuples for exact intersection
        return set(zip(prov["corpus"].tolist(), prov["shard"].tolist(), prov["row"].tolist()))
    keysets = {s: _keys(slice_prov[s]) for s in SLICES}
    pairwise = {}
    for a, b in (("A", "B"), ("A", "C"), ("B", "C")):
        inter = len(keysets[a] & keysets[b])
        pairwise[f"{a}∩{b}"] = inter
    eval_overlap = {}
    for name, pth in EVAL_PROV.items():
        if not Path(pth).exists():
            eval_overlap[name] = "no provenance (cross-corpus by design)"
            continue
        ep = np.load(pth, mmap_mode="r", allow_pickle=False)
        eks = _keys(ep)
        eval_overlap[name] = {s: len(keysets[s] & eks) for s in SLICES}
    # cross-corpus safety: which corpus codes appear in slices vs the eval provenances
    slice_codes = sorted(set(int(c) for s in SLICES for c in np.unique(slice_prov[s]["corpus"])))
    proofs = {
        "schema": "draw-univ-proofs-2026-08-30", "seed": SEED,
        "pairwise_slice_intersections": pairwise,
        "pairwise_disjoint": all(v == 0 for v in pairwise.values()),
        "eval_set_overlap": eval_overlap,
        "eval_disjoint": all(isinstance(v, str) or all(x == 0 for x in v.values())
                             for v in eval_overlap.values()),
        "slice_corpus_codes": slice_codes,
        "note": ("base slices are corpus codes 0-3 (fineweb/rpj/pile/starcoder); social register probes "
                 "(reddit/ca/twitter/bluesky) + common-corpus probes (wiki/ccweb/ccscience/a1) are "
                 "DIFFERENT corpora, hence disjoint by corpus code — verified exactly where provenance "
                 "exists (probe-code-heldout, a1-common-neutral). Baseline coords excluded per corpus so "
                 "slices are independent draws, not a seed-42 replay of the baseline."),
        "corpus_meta": corpus_meta, "wall_s": round(time.time() - t0, 1),
    }
    (SUBSTRATES / "draw-univ-proofs.json").write_text(json.dumps(proofs, indent=1))
    for s in SLICES:
        man = {"slice": s, "rows": 2_000_000, "mix": PER_SLICE, "seed": SEED,
               "pairwise_slice_intersections": pairwise,
               "eval_set_overlap": {k: (v if isinstance(v, str) else v[s]) for k, v in eval_overlap.items()},
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
