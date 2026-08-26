#!/usr/bin/env python3
"""Build 6 MiniLM 1M substrates for the social-mixture sweep (owner order 2026-08-26).

Base mix (fineweb-edu 40 / RedPajama 25 / Pile 25 / starcoder 10 -- same as
minilm-mix-1m).  Two compositions x three social shares s in {0.10,0.20,0.30}:

  reddit-only : (1-s) base + s reddit
                -> minilm-rmix10-1m / -rmix20-1m / -rmix30-1m
  balanced    : (1-s) base + s/4 each of reddit + CA + twitter + bluesky
                -> minilm-bmix10-1m / -bmix20-1m / -bmix30-1m

The 0% baseline is the existing minilm-mix-1m map -- NOT rebuilt here.

Mixture-preserving sampling mirrors p3_build_and_scorecard's `_open_shard`
(magic-byte sniff: headerless raw f32 for fineweb/redpajama/pile, real .npy for
starcoder + all social corpora) and `_sample_corpus_rows` (uniform global draw,
without replacement, np.random.default_rng(42)).

CRITICAL -- probe holdout: the probe suite scores OOD on HELDOUT rows of the 4
social corpora (reddit/CA/twitter/bluesky).  For EACH of those corpora the FIRST
300,000 global rows (sorted-shard concatenation order) are RESERVED as the probe
holdout; every training sample here is drawn ONLY from global offset >= 300000.
The reserved ranges are written to
`/data/latent-basemap/substrates/social-holdout-partition.json` so the probe
builder uses the disjoint front slice.  Base corpora need no holdout -- they are
not probe registers.

Write-once: each substrate.f32.npy is skipped if it already exists.  CPU-only
(set CUDA_VISIBLE_DEVICES="" before running); ~1-2 GB RAM per substrate via a
memmap gather + chunked shuffle.  Output float32 (1_000_000, 384).
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
TOTAL = 1_000_000
HOLDOUT = 300_000  # first 300k global rows of each social corpus reserved for probes

SUBSTRATES = Path("/data/latent-basemap/substrates")
HOLDOUT_JSON = SUBSTRATES / "social-holdout-partition.json"

E = "/data/embeddings"
CORPORA = {
    "fineweb":   f"{E}/fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2/train",
    "redpajama": f"{E}/RedPajama-Data-V2-sample-10B-chunked-120-all-MiniLM-L6-v2/train",
    "pile":      f"{E}/pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2/train",
    "starcoder": f"{E}/starcoderdata-code-chunked-120-all-MiniLM-L6-v2/train",
    "reddit":    f"{E}/reddit-tldr17-chunked-120-all-MiniLM-L6-v2/train",
    "CA":        f"{E}/communityarchive-tweets-all-MiniLM-L6-v2/train",
    "twitter":   f"{E}/twitter100m-chunked-120-all-MiniLM-L6-v2/train",
    "bluesky":   f"{E}/bluesky-5m-chunked-120-all-MiniLM-L6-v2/train",
}
BASE_MIX = {"fineweb": 0.40, "redpajama": 0.25, "pile": 0.25, "starcoder": 0.10}
SOCIAL = ("reddit", "CA", "twitter", "bluesky")
# offset applied to a corpus's global row space before sampling (holdout front slice)
CORPUS_OFFSET = {c: (HOLDOUT if c in SOCIAL else 0) for c in CORPORA}
# fixed, deterministic corpus fill order (base first, then social)
FILL_ORDER = ("fineweb", "redpajama", "pile", "starcoder", *SOCIAL)


# -- shard IO (mirrors p3_build_and_scorecard._open_shard) --------------------

def _open_shard(path: str):
    """Read-only (rows, DIM) f32 view: real .npy via np.load, headerless raw f32
    (fineweb/redpajama/pile) via np.memmap with row count floored from size."""
    with open(path, "rb") as fh:
        is_npy = fh.read(6) == b"\x93NUMPY"
    if is_npy:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    rows = os.path.getsize(path) // (DIM * 4)
    return np.memmap(path, dtype=np.float32, mode="r", shape=(rows, DIM))


def _corpus_shards(corpus: str):
    files = sorted(glob.glob(os.path.join(CORPORA[corpus], "*.npy")))
    if not files:
        raise FileNotFoundError(f"no shards for corpus {corpus} at {CORPORA[corpus]}")
    return [_open_shard(f) for f in files]


# -- mixture-preserving sampler (offset-aware _sample_corpus_rows) ------------

def _sample_corpus_rows(shards, n: int, rng, offset: int = 0, dim: int = DIM):
    """Gather ``n`` uniform-random rows without replacement from the global row
    space spanned by ``shards``, restricted to global indices >= ``offset``
    (the holdout front slice).  Returns (rows, record).  Per-shard gathers issue
    in ascending local-index order for memmap locality (same as the reference)."""
    shard_rows = [int(s.shape[0]) for s in shards]
    total = int(sum(shard_rows))
    avail = total - offset
    if n > avail:
        raise ValueError(f"need {n} rows but only {avail} available past offset "
                         f"{offset} (total {total})")
    gidx = np.sort(rng.choice(avail, size=n, replace=False)) + offset
    offsets = np.concatenate([[0], np.cumsum(shard_rows)]).astype(np.int64)
    shard_of = np.searchsorted(offsets, gidx, side="right") - 1
    local = gidx - offsets[shard_of]
    out = np.empty((n, dim), dtype=np.float32)
    for si in range(len(shards)):
        mask = shard_of == si
        if not mask.any():
            continue
        loc = local[mask]
        order = np.argsort(loc)
        dest = np.nonzero(mask)[0][order]
        out[dest] = np.asarray(shards[si][loc[order]], dtype=np.float32)
    record = {"total_rows": total, "offset": offset, "train_pool": avail,
              "sampled": n, "shard_rows": shard_rows}
    return out, record


# -- per-substrate count planning --------------------------------------------

def _base_counts(base_total: int) -> dict[str, int]:
    """Split ``base_total`` across the 4 base corpora by BASE_MIX; rounding slack
    to fineweb so the base block sums EXACTLY to ``base_total``."""
    counts = {c: int(round(base_total * f)) for c, f in BASE_MIX.items()}
    drift = base_total - sum(counts.values())
    counts["fineweb"] += drift
    return counts


def plan_counts(kind: str, s: float) -> dict[str, int]:
    """Per-corpus counts summing EXACTLY to TOTAL (1,000,000).

    kind='rmix': social = s*TOTAL reddit; kind='bmix': social = s/4 each of the
    four social corpora.  Base block absorbs any social rounding so the total is
    always exactly 1M."""
    if kind == "rmix":
        social = {"reddit": int(round(s * TOTAL))}
    elif kind == "bmix":
        per = int(round(s * TOTAL / 4))
        social = {c: per for c in SOCIAL}
    else:
        raise ValueError(kind)
    base_total = TOTAL - sum(social.values())
    counts = _base_counts(base_total)
    counts.update(social)
    assert sum(counts.values()) == TOTAL, (kind, s, sum(counts.values()))
    return counts


# -- build one substrate ------------------------------------------------------

def build_one(name: str, kind: str, s: float) -> dict:
    out_dir = SUBSTRATES / name
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_path = out_dir / "substrate.f32.npy"
    counts = plan_counts(kind, s)
    fill = [(c, counts[c]) for c in FILL_ORDER if counts.get(c, 0) > 0]

    if sub_path.exists():
        print(f"[{name}] {sub_path} exists, skip build")
        man = json.loads((out_dir / "manifest.json").read_text())
        return man

    print(f"[{name}] kind={kind} s={s} counts={counts}")
    t0 = time.time()
    rng = np.random.default_rng(SEED)  # fresh rng(42) per substrate, threaded in FILL_ORDER

    out = np.lib.format.open_memmap(str(sub_path), mode="w+",
                                    dtype=np.float32, shape=(TOTAL, DIM))
    subsets = np.empty(TOTAL, dtype=object)
    records: dict[str, dict] = {}
    pos = 0
    for corpus, cnt in fill:
        shards = _corpus_shards(corpus)
        rows, rec = _sample_corpus_rows(shards, cnt, rng,
                                        offset=CORPUS_OFFSET[corpus])
        out[pos:pos + cnt] = rows
        subsets[pos:pos + cnt] = corpus
        records[corpus] = rec
        pos += cnt
        print(f"    {corpus}: +{cnt:,} rows "
              f"(pool {rec['train_pool']:,} of {rec['total_rows']:,}, "
              f"offset {rec['offset']:,})", flush=True)
        del rows, shards
    assert pos == TOTAL, pos

    # shuffle (seed 42) so corpus blocks are not contiguous -- chunked to a temp
    # memmap then atomically replace, keeping RAM low.
    perm = rng.permutation(TOTAL)
    subsets = subsets[perm]
    tmp = sub_path.with_suffix(".shuf.tmp.npy")
    shuf = np.lib.format.open_memmap(str(tmp), mode="w+",
                                     dtype=np.float32, shape=(TOTAL, DIM))
    CH = 200_000
    for i in range(0, TOTAL, CH):
        shuf[i:i + CH] = out[perm[i:i + CH]]
    shuf.flush()
    del out, shuf
    os.replace(tmp, sub_path)
    np.save(out_dir / "subsets.npy", subsets)

    manifest = {
        "name": name, "kind": kind, "social_share": s,
        "rows": TOTAL, "dim": DIM, "dtype": "float32", "seed": SEED,
        "shuffled": True, "base_mix": BASE_MIX,
        "counts": counts, "per_corpus": records,
        "holdout_rows": HOLDOUT,
        "holdout_note": (
            "The 4 social corpora (reddit/CA/twitter/bluesky) reserve their "
            "FIRST 300,000 global rows (sorted-shard concatenation order) as a "
            "probe OOD holdout; ALL training rows here are drawn only from "
            "global offset >= 300000. Base corpora have no holdout. See "
            "social-holdout-partition.json."),
        "composition_note": (
            f"(1-s) base at 40/25/25/10 + social; kind={kind}, s={s}."),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    a = np.load(sub_path, mmap_mode="r")
    print(f"[{name}] built shape={a.shape} dtype={a.dtype} "
          f"in {(time.time()-t0)/60:.1f} min -> {sub_path}")
    del a
    return manifest


def write_holdout_partition() -> None:
    corpora = {}
    for c in SOCIAL:
        shards = _corpus_shards(c)
        total = int(sum(int(s.shape[0]) for s in shards))
        corpora[c] = {
            "path": CORPORA[c],
            "total_rows": total,
            "holdout_global_range": [0, HOLDOUT],
            "train_pool_global_range": [HOLDOUT, total],
            "train_pool_rows": total - HOLDOUT,
        }
        del shards
    doc = {
        "holdout_rows": HOLDOUT,
        "row_space": ("global row index over the corpus's shards sorted by "
                      "filename (glob + sorted), concatenated -- identical to "
                      "_open_shard / _sample_corpus_rows ordering"),
        "note": ("First 300,000 global rows of each social corpus are RESERVED "
                 "as probe OOD holdout. All training substrates in this sweep "
                 "draw ONLY from global offset >= 300000, so the probe builder "
                 "can safely score on rows [0, 300000)."),
        "corpora": corpora,
    }
    HOLDOUT_JSON.write_text(json.dumps(doc, indent=1))
    print(f"[holdout] wrote {HOLDOUT_JSON}")
    for c, d in corpora.items():
        print(f"    {c}: holdout [0,{HOLDOUT:,}) | train pool "
              f"[{HOLDOUT:,},{d['total_rows']:,}) = {d['train_pool_rows']:,}")


GRID = [
    ("minilm-rmix10-1m", "rmix", 0.10),
    ("minilm-rmix20-1m", "rmix", 0.20),
    ("minilm-rmix30-1m", "rmix", 0.30),
    ("minilm-bmix10-1m", "bmix", 0.10),
    ("minilm-bmix20-1m", "bmix", 0.20),
    ("minilm-bmix30-1m", "bmix", 0.30),
]


def main() -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    SUBSTRATES.mkdir(parents=True, exist_ok=True)
    write_holdout_partition()
    summary = {}
    for name, kind, s in GRID:
        man = build_one(name, kind, s)
        summary[name] = man["counts"]
    print("\n=== summary (per-corpus counts) ===")
    for name, counts in summary.items():
        print(f"{name}: sum={sum(counts.values()):,} {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
