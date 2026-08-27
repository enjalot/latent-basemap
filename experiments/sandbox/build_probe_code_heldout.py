#!/usr/bin/env python3
"""Build the HELD-OUT starcoder code probe register (CPU-only).

The existing `probe-code` register (build_probe_registers.py, mode="random") is
CONTAMINATED: it draws 250K random rows over the WHOLE starcoderdata corpus with
no regard for what the training substrates consumed, so it overlaps the training
rows of the mixture maps (measured 1724-2252 exact-row overlaps). It is therefore
EXCLUDED from the decision-time maximin.

This script freezes a provably-disjoint replacement, `probe-code-heldout`, so that
CODE can be scored legitimately at the 2M-confirmation DECISION point (bmix30-2m
vs the 2M baseline). Disjointness is with respect to the 2M baseline's starcoder
training rows:

    2M baseline provenance:
      /data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/
      minilm-mixed-2m-substrate-and-exact-k15-graph-v1/provenance.npy
    -- a (corpus,shard,row) coordinate list; corpus index 3 == starcoderdata
       (200,000 code rows = 10% of the 2M base mix, spread over shards 0-19).

Method (mirrors build_a1_common_neutral.py): EXACT index intersection, not
approximation. We enumerate the full starcoder (shard,row) coordinate space
(21 shards, 10,100,000 rows total), subtract the 200,000 baseline code coords,
and sample N_SAMPLE rows (seed 42) from the ~9.9M-row COMPLEMENT. The row space
is the global row index over the corpus's shards sorted by filename (glob +
sorted) then concatenated -- identical to build_mixture_substrates._open_shard /
_sample_corpus_rows ordering -- and each global index maps 1:1 to a (shard,row)
coordinate matching the baseline provenance encoding.

Disjointness proof: residual_overlap.json records the exact intersection between
the register's chosen (shard,row) code coords and the baseline-2m code coords = 0.
Because the bmix30-2m confirmation substrate draws its base rows as a matched
SUBSET of the 2M baseline's rows, its code rows are a subset of the baseline-2m
code rows, so this register is ALSO disjoint from bmix30-2m BY CONSTRUCTION
(0 overlap with a superset implies 0 overlap with any subset).

Output substrate.f32.npy is raw float32 (N_SAMPLE, 384), UNNORMALIZED (matches
build_probe_registers.py; mixture_probe._norm normalizes at load).

CPU-ONLY: pure slicing / set ops. A GPU pipeline owns the GPU; run with
CUDA_VISIBLE_DEVICES="". The knn/fuzzy truth-graph build is a SEPARATE GPU stage
queued later by the orchestrator -- do NOT run it here.

Write-once: refuses to overwrite an existing substrate.f32.npy.

Usage:
    CUDA_VISIBLE_DEVICES="" .venv/bin/python experiments/sandbox/build_probe_code_heldout.py
"""
from __future__ import annotations

import datetime
import glob
import json
import os
from pathlib import Path

import numpy as np

SEED = 42
N_SAMPLE = 250_000
DIM = 384

CORPUS_NAME = "starcoderdata-code-chunked-120-all-MiniLM-L6-v2"
CORPUS_INDEX = 3  # corpus id for starcoder in both R0216 and R0238 provenance
EMB_ROOT = Path("/data/embeddings")
SRC_DIR = EMB_ROOT / CORPUS_NAME / "train"

BASELINE_2M_PROVENANCE = (
    "/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
    "minilm-mixed-2m-substrate-and-exact-k15-graph-v1/provenance.npy")

OUT_DIR = Path("/data/latent-basemap/substrates/probe-code-heldout")


def _open_shard(path: str):
    """Read-only (rows, DIM) f32 view. Real .npy via np.load(mmap); headerless
    raw f32 via np.memmap with row count floored from size. Mirrors
    build_mixture_substrates._open_shard."""
    with open(path, "rb") as fh:
        is_npy = fh.read(6) == b"\x93NUMPY"
    if is_npy:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    rows = os.path.getsize(path) // (DIM * 4)
    return np.memmap(path, dtype=np.float32, mode="r", shape=(rows, DIM))


def _shard_index(path: str) -> int:
    # data-00007-of-00020.npy -> 7
    return int(os.path.basename(path).split("-")[1])


def main() -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    sub_path = OUT_DIR / "substrate.f32.npy"
    if sub_path.exists():
        raise SystemExit(f"REFUSE overwrite (write-once): {sub_path} already exists")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # --- enumerate the starcoder (shard,row) global row space -------------------
    files = sorted(glob.glob(str(SRC_DIR / "*.npy")))
    if not files:
        raise SystemExit(f"no shards under {SRC_DIR}")
    shard_ids = np.array([_shard_index(f) for f in files], dtype=np.int64)
    shard_rows = np.array([int(_open_shard(f).shape[0]) for f in files], dtype=np.int64)
    # global index over sorted-shard concatenation; offsets[i] = start of shard i
    offsets = np.concatenate([[0], np.cumsum(shard_rows)]).astype(np.int64)
    total = int(offsets[-1])
    print(f"starcoder corpus: {len(files)} shards, {total:,} rows total", flush=True)
    assert list(shard_ids) == sorted(shard_ids), "shard filenames not in index order"

    def coords_to_global(shard: np.ndarray, row: np.ndarray) -> np.ndarray:
        # map (shard_index, within-shard row) -> global row index. shard_index
        # equals position because filenames sort in index order (asserted above).
        return offsets[shard.astype(np.int64)] + row.astype(np.int64)

    def global_to_coords(gidx: np.ndarray):
        pos = np.searchsorted(offsets, gidx, side="right") - 1
        row = gidx - offsets[pos]
        return shard_ids[pos], row  # (shard_index, within-shard row)

    # --- baseline-2m code coords = the EXACT rows to exclude --------------------
    prov = np.asarray(np.load(BASELINE_2M_PROVENANCE, mmap_mode="r"))
    base_code = prov[prov["corpus"] == CORPUS_INDEX]
    base_shard = np.asarray(base_code["shard"], dtype=np.int64)
    base_row = np.asarray(base_code["row"], dtype=np.int64)
    print(f"baseline-2m code rows: {base_code.size:,} "
          f"(shards {int(base_shard.min())}-{int(base_shard.max())})", flush=True)
    assert base_shard.max() < len(files) and (base_row < shard_rows[base_shard]).all(), \
        "baseline code coord out of range for the on-disk starcoder shards"
    excl_global = np.unique(coords_to_global(base_shard, base_row))
    print(f"exclusion set (unique baseline code coords): {excl_global.size:,}", flush=True)

    # --- exact complement + seed-42 sample -------------------------------------
    all_global = np.arange(total, dtype=np.int64)
    held = np.setdiff1d(all_global, excl_global, assume_unique=True)
    print(f"held-out complement: {held.size:,} rows "
          f"({total:,} - {excl_global.size:,})", flush=True)
    if held.size < N_SAMPLE:
        raise SystemExit(f"complement {held.size:,} < N_SAMPLE {N_SAMPLE:,}")
    # sample positions in the complement (seed 42), sorted for sequential reads
    # -- same np.sort(rng.choice(..., replace=False)) convention as
    #    build_probe_registers._take_random, restricted to the complement pool.
    pos = np.sort(rng.choice(held.size, size=N_SAMPLE, replace=False))
    chosen_global = held[pos]
    ch_shard, ch_row = global_to_coords(chosen_global)

    # --- gather vectors (chosen_global ascending -> sequential shard reads) -----
    sample = np.empty((N_SAMPLE, DIM), dtype=np.float32)
    probe_prov = np.empty(N_SAMPLE, dtype=[("corpus", "u1"), ("shard", "<u2"),
                                           ("row", "<i8")])
    path_by_pos = {i: files[i] for i in range(len(files))}
    mm_cache: dict[int, np.ndarray] = {}
    # shard position (index into files) for each chosen global row
    ch_pos = np.searchsorted(offsets, chosen_global, side="right") - 1
    for i in range(N_SAMPLE):
        p = int(ch_pos[i])
        mm = mm_cache.get(p)
        if mm is None:
            mm = _open_shard(path_by_pos[p])
            mm_cache[p] = mm
        r = int(chosen_global[i] - offsets[p])
        sample[i] = np.asarray(mm[r], dtype=np.float32)
        probe_prov[i] = (CORPUS_INDEX, int(shard_ids[p]), r)

    assert sample.shape == (N_SAMPLE, DIM), sample.shape
    assert np.isfinite(sample).all(), "substrate contains non-finite values"

    np.save(sub_path, sample)
    np.save(OUT_DIR / "provenance.npy", probe_prov)

    # --- disjointness proof: exact intersection vs baseline-2m code coords ------
    K_SHARD = 10 ** 9  # shard in [0,~200), row in [0,~2e6): (shard,row) -> unique int64
    probe_keys = np.unique(ch_shard.astype(np.int64) * K_SHARD + ch_row.astype(np.int64))
    base_keys = np.unique(base_shard * K_SHARD + base_row)
    overlap = int(np.intersect1d(probe_keys, base_keys, assume_unique=True).size)
    print(f"OVERLAP probe-code-heldout vs baseline-2m code rows: {overlap}", flush=True)
    assert overlap == 0, f"DISJOINTNESS FAILED: {overlap} overlapping code rows"

    overlap_doc = {
        "schema": "probe-code-heldout-residual-overlap-2026-08-27",
        "probe_substrate": str(sub_path),
        "probe_rows": N_SAMPLE,
        "corpus": CORPUS_NAME,
        "method": "exact (shard,row) index intersection between the register's "
                  "chosen starcoder code coords and the 2M-baseline starcoder "
                  "training coords (corpus id 3).",
        "baseline_2m": {
            "provenance": BASELINE_2M_PROVENANCE,
            "code_training_rows": int(base_code.size),
            "overlap_rows": overlap,
            "overlap_pct_of_probe": round(100.0 * overlap / N_SAMPLE, 6),
        },
        "bmix30_2m_subset_note": (
            "The bmix30-2m confirmation substrate draws its BASE rows as a matched "
            "SUBSET of the 2M baseline's rows, so its code rows are a subset of the "
            "baseline-2m code rows above. Zero overlap with the (superset) "
            "baseline-2m code rows therefore implies ZERO overlap with bmix30-2m's "
            "code rows BY CONSTRUCTION -- this register is disjoint from BOTH sides "
            "of the 2M-confirmation decision map."),
        "note": "overlap is 0 by construction (register rows drawn from the EXACT "
                "complement of the baseline-2m code coords); this table is the "
                "decision-time disjointness proof.",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    (OUT_DIR / "residual_overlap.json").write_text(json.dumps(overlap_doc, indent=1))

    # --- manifest ---------------------------------------------------------------
    manifest = {
        "register": "probe-code-heldout",
        "schema": "probe-code-heldout-2026-08-27",
        "purpose": "HELD-OUT starcoder code probe register for the 2M-confirmation "
                   "(bmix30-2m vs baseline-2m) DECISION-time maximin. Replaces the "
                   "CONTAMINATED `probe-code` register (which overlapped training "
                   "rows and is excluded from the maximin).",
        "source_dir": str(SRC_DIR),
        "source_total_rows": total,
        "dim": DIM,
        "seed": SEED,
        "dtype": "float32",
        "unnormalized": True,
        "count": int(N_SAMPLE),
        "shape": list(sample.shape),
        "mode": "random-complement",
        "selection": f"random {N_SAMPLE:,} of the {held.size:,}-row complement "
                     f"(seed {SEED})",
        "sampling_protocol": (
            "global row index over glob+sorted starcoder shards (build_mixture_"
            "substrates ordering); enumerate all coords, subtract the exact "
            "baseline-2m code coords, np.sort(rng(42).choice(complement, "
            f"{N_SAMPLE}, replace=False))."),
        "complement_basis": {
            "baseline_2m_provenance": BASELINE_2M_PROVENANCE,
            "baseline_2m_code_rows_excluded": int(base_code.size),
            "complement_rows": int(held.size),
            "corpus_index": CORPUS_INDEX,
        },
        "disjointness": {
            "vs_baseline_2m_code": "0 overlap (proven; residual_overlap.json)",
            "vs_bmix30_2m_code": "0 overlap BY CONSTRUCTION (bmix30-2m code rows "
                                 "are a subset of baseline-2m code rows).",
        },
        "substrate": str(sub_path),
        "provenance": str(OUT_DIR / "provenance.npy"),
        "residual_overlap": str(OUT_DIR / "residual_overlap.json"),
        "note": "raw float32 (N,384), UNNORMALIZED (mixture_probe._norm normalizes "
                "at load). knn/fuzzy truth graph is a separate GPU stage queued by "
                "the orchestrator.",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))

    print(f"\nprobe-code-heldout: wrote {sub_path} shape={sample.shape} "
          f"dtype={sample.dtype}")
    print(f"  provenance:        {OUT_DIR / 'provenance.npy'}")
    print(f"  residual_overlap:  {OUT_DIR / 'residual_overlap.json'}  (overlap={overlap})")
    print(f"  manifest:          {OUT_DIR / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
