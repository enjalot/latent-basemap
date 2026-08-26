#!/usr/bin/env python3
"""BROAD probe-register suite, step 1: slice the substrate holdouts (CPU-only).

Builds the eight MiniLM (384-d) OOD probe registers used to score a MiniLM
mixture sweep.  Each register is a ~250k-row f32 substrate carved out of a
source corpus so that projecting it through a frozen sweep map and scoring
fold-faithfulness (FFR) against the register's OWN truth graph measures how
faithfully that map RECEIVES out-of-distribution data.

Registers (source dir under /data/embeddings/<dir>/train/*.npy):

  probe-reddit    reddit-tldr17-chunked-120-all-MiniLM-L6-v2   (front-300k reserved slice)
  probe-ca        communityarchive-tweets-all-MiniLM-L6-v2     (front-300k)
  probe-twitter   twitter100m-chunked-120-all-MiniLM-L6-v2     (front-300k)
  probe-bluesky   bluesky-5m-chunked-120-all-MiniLM-L6-v2      (front-300k)
  probe-wiki      wikipedia-en-chunked-120-all-MiniLM-L6-v2    (random 250k, seed 42)
  probe-ccweb     common-corpus-web-chunked-120-all-MiniLM-L6-v2      (250k or all)
  probe-ccscience common-corpus-science-chunked-120-all-MiniLM-L6-v2  (250k or all)
  probe-code      starcoderdata-code-chunked-120-all-MiniLM-L6-v2     (random 250k, seed 42)

The four SOCIAL registers (reddit/ca/twitter/bluesky) MUST be disjoint from the
sweep's training rows.  A companion builder is expected to write
``substrates/social-holdout-partition.json`` reserving the FIRST 300,000 rows of
each social corpus for probes; the sweep trains only on rows AFTER that front.
If that file is absent this script falls back to reserving the front 300k itself
(and records the fallback in each social manifest).  Either way the social
registers are sliced from the reserved front range and asserted to lie strictly
inside [0, 300000), guaranteeing disjointness from the sweep training rows.

CPU-ONLY: this is pure slicing / IO.  A pipeline owns the GPU, so run with
``CUDA_VISIBLE_DEVICES=""``.  The knn/fuzzy truth-graph builds are a separate
GPU stage (image_map_pipeline knn|fuzzy) run later by the orchestrator.

Write-once / idempotent: a register whose substrate.f32.npy already exists is
left untouched.

Usage:
    CUDA_VISIBLE_DEVICES="" .venv/bin/python experiments/sandbox/build_probe_registers.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

DIM = 384
SEED = 42
TARGET = 250_000            # nominal per-register row count
HOLDOUT_FRONT = 300_000     # rows reserved at the front of each social corpus

SUBSTRATES = Path("/data/latent-basemap/substrates")
HOLDOUT_PARTITION = SUBSTRATES / "social-holdout-partition.json"
EMB = "/data/embeddings"

# name -> (source dir, mode).  mode is one of:
#   "front"        first TARGET rows out of the reserved front-HOLDOUT_FRONT
#   "random"       random TARGET rows over the whole corpus (seed 42)
#   "all_or_random" all rows if fewer than TARGET, else random TARGET (seed 42)
REGISTERS = {
    "probe-reddit":    ("reddit-tldr17-chunked-120-all-MiniLM-L6-v2", "front"),
    "probe-ca":        ("communityarchive-tweets-all-MiniLM-L6-v2", "front"),
    "probe-twitter":   ("twitter100m-chunked-120-all-MiniLM-L6-v2", "front"),
    "probe-bluesky":   ("bluesky-5m-chunked-120-all-MiniLM-L6-v2", "front"),
    "probe-wiki":      ("wikipedia-en-chunked-120-all-MiniLM-L6-v2", "random"),
    "probe-ccweb":     ("common-corpus-web-chunked-120-all-MiniLM-L6-v2", "all_or_random"),
    "probe-ccscience": ("common-corpus-science-chunked-120-all-MiniLM-L6-v2", "all_or_random"),
    "probe-code":      ("starcoderdata-code-chunked-120-all-MiniLM-L6-v2", "random"),
}
SOCIAL = {"probe-reddit", "probe-ca", "probe-twitter", "probe-bluesky"}


def _open_shard(path: str):
    """Read-only (rows, DIM) float32-castable view of a shard.

    Real .npy shards open via ``np.load(mmap_mode='r')``; headerless raw
    float32 buffers open via ``np.memmap`` with the row count floored from the
    file size (a truncated trailing partial row is dropped).  Mirrors
    ``p3_build_and_scorecard._open_shard``.
    """
    with open(path, "rb") as fh:
        is_npy = fh.read(6) == b"\x93NUMPY"
    if is_npy:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    rows = os.path.getsize(path) // (DIM * 4)
    return np.memmap(path, dtype=np.float32, mode="r", shape=(rows, DIM))


def _shards(src_dir: str):
    files = sorted(glob.glob(os.path.join(EMB, src_dir, "train", "*.npy")))
    if not files:
        raise FileNotFoundError(f"no shards under {EMB}/{src_dir}/train")
    return [_open_shard(f) for f in files]


def _take_front(shards, n: int) -> np.ndarray:
    """First ``n`` rows in shard order, cast to contiguous (n, DIM) float32.

    Global row indices of the returned rows are 0..n-1, so with n <= front
    reservation the slice is disjoint from any post-front training rows.
    """
    total = int(sum(int(s.shape[0]) for s in shards))
    n = int(min(n, total))
    out = np.empty((n, DIM), dtype=np.float32)
    filled = 0
    for s in shards:
        if filled >= n:
            break
        take = min(int(s.shape[0]), n - filled)
        out[filled:filled + take] = np.asarray(s[:take], dtype=np.float32)
        filled += take
    return out


def _take_random(shards, n: int, rng) -> tuple[np.ndarray, int]:
    """Random ``min(n, total)`` rows without replacement over the global row
    space of ``shards``.  Returns (rows, total_available)."""
    shard_rows = [int(s.shape[0]) for s in shards]
    total = int(sum(shard_rows))
    n = int(min(n, total))
    gidx = np.sort(rng.choice(total, size=n, replace=False))
    offsets = np.concatenate([[0], np.cumsum(shard_rows)]).astype(np.int64)
    shard_of = np.searchsorted(offsets, gidx, side="right") - 1
    local = gidx - offsets[shard_of]
    out = np.empty((n, DIM), dtype=np.float32)
    for i in range(len(shards)):
        m = shard_of == i
        if not np.any(m):
            continue
        li = local[m]
        out[np.nonzero(m)[0]] = np.asarray(shards[i][li], dtype=np.float32)
    return out, total


def _load_partition() -> tuple[dict, bool]:
    """Return (partition_map, is_fallback).

    partition_map maps corpus dir -> front-reserved row count.  If the companion
    partition file is missing we fall back to reserving HOLDOUT_FRONT ourselves.
    """
    if HOLDOUT_PARTITION.exists():
        try:
            data = json.loads(HOLDOUT_PARTITION.read_text())
            return data, False
        except Exception as e:  # pragma: no cover - defensive
            print(f"WARN: could not parse {HOLDOUT_PARTITION}: {e}; falling back")
    return {}, True


def _front_reserved(partition: dict, src_dir: str) -> int:
    """Front-reserved row count for a social corpus per the partition file, else
    the local HOLDOUT_FRONT fallback.  Accepts a few plausible schemas."""
    if not partition:
        return HOLDOUT_FRONT
    # companion schema: per-corpus holdout_global_range under corpora.<key>,
    # keyed by short name (reddit/CA/twitter/bluesky); plus a top-level
    # holdout_rows.  Also tolerate a few flatter forms.
    corpora = partition.get("corpora")
    if isinstance(corpora, dict):
        for node in corpora.values():
            if isinstance(node, dict) and node.get("path", "").rstrip("/").endswith(
                    src_dir + "/train"):
                rng = node.get("holdout_global_range")
                if isinstance(rng, (list, tuple)) and len(rng) == 2:
                    return int(rng[1]) - int(rng[0])
    if isinstance(partition.get("holdout_rows"), (int, float)):
        return int(partition["holdout_rows"])
    node = partition.get(src_dir)
    if node is None and isinstance(partition.get("reserved"), dict):
        node = partition["reserved"].get(src_dir)
    if isinstance(node, dict):
        for key in ("holdout_front", "front", "reserved_rows", "n"):
            if key in node:
                return int(node[key])
    if isinstance(node, (int, float)):
        return int(node)
    return HOLDOUT_FRONT


def build_one(name: str, src_dir: str, mode: str, partition: dict,
              is_fallback: bool) -> dict:
    out_dir = SUBSTRATES / name
    sub_path = out_dir / "substrate.f32.npy"
    manifest_path = out_dir / "manifest.json"
    if sub_path.exists():
        cnt = int(np.load(sub_path, mmap_mode="r").shape[0])
        print(f"{name}: exists ({cnt:,} rows), skip (write-once)")
        man = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        man["_status"] = "existing"
        man.setdefault("count", cnt)
        return man

    out_dir.mkdir(parents=True, exist_ok=True)
    shards = _shards(src_dir)
    total = int(sum(int(s.shape[0]) for s in shards))
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    manifest: dict = {
        "register": name,
        "source_dir": f"{EMB}/{src_dir}/train",
        "source_total_rows": total,
        "dim": DIM,
        "seed": SEED,
        "dtype": "float32",
        "mode": mode,
    }

    if mode == "front":
        front = _front_reserved(partition, src_dir)
        take = min(TARGET, front)
        rows = _take_front(shards, take)
        assert rows.shape[0] <= front, "probe slice exceeds reserved front"
        # global indices of a front slice are 0..count-1; assert strictly inside
        # the reserved front range -> disjoint from post-front training rows.
        assert rows.shape[0] - 1 < front, "front slice not inside reserved range"
        manifest.update({
            "holdout_front_reserved": int(front),
            "holdout_source": ("fallback:self-reserved-300k" if is_fallback
                               else str(HOLDOUT_PARTITION)),
            "holdout_range_used": [0, int(rows.shape[0])],
            "disjoint_from_training": True,
            "note": "social OOD register; rows sliced from the reserved front "
                    "range, disjoint from the sweep training rows (which start "
                    f"at global row {int(front)}).",
        })
    elif mode == "random":
        rows, _ = _take_random(shards, TARGET, rng)
        manifest.update({
            "selection": f"random {rows.shape[0]:,} of {total:,} (seed {SEED})",
            "note": "non-social OOD register; random global sample.",
        })
    elif mode == "all_or_random":
        if total <= TARGET:
            rows = _take_front(shards, total)   # all rows, in order
            manifest.update({
                "selection": f"all {rows.shape[0]:,} rows (corpus < {TARGET:,})",
                "note": "small corpus; used in full.",
            })
        else:
            rows, _ = _take_random(shards, TARGET, rng)
            manifest.update({
                "selection": f"random {rows.shape[0]:,} of {total:,} (seed {SEED})",
                "note": "non-social OOD register; random global sample.",
            })
    else:  # pragma: no cover
        raise ValueError(f"unknown mode {mode!r}")

    rows = np.ascontiguousarray(rows, dtype=np.float32)
    manifest["count"] = int(rows.shape[0])
    manifest["shape"] = list(rows.shape)
    np.save(sub_path, rows)
    manifest_path.write_text(json.dumps(manifest, indent=1))
    manifest["_status"] = "built"
    print(f"{name}: {rows.shape[0]:,} rows shape={rows.shape} "
          f"({mode}) in {time.time()-t0:.1f}s -> {sub_path}")
    return manifest


def main(argv: list[str]) -> int:
    partition, is_fallback = _load_partition()
    if is_fallback:
        print(f"NOTE: {HOLDOUT_PARTITION} absent -> falling back to self-reserving "
              f"the front {HOLDOUT_FRONT:,} rows of each social corpus.")
    else:
        print(f"using companion holdout partition {HOLDOUT_PARTITION}")

    only = set(argv[1:])
    summary = {}
    for name, (src_dir, mode) in REGISTERS.items():
        if only and name not in only:
            continue
        if not os.path.isdir(os.path.join(EMB, src_dir, "train")):
            print(f"{name}: MISSING source dir {EMB}/{src_dir}/train -> SKIP")
            summary[name] = {"_status": "missing_source"}
            continue
        summary[name] = build_one(name, src_dir, mode, partition, is_fallback)

    print("\n=== probe-register summary ===")
    for name, man in summary.items():
        st = man.get("_status")
        if st == "missing_source":
            print(f"  {name:16s} MISSING SOURCE")
            continue
        social = " (social, front<300k)" if name in SOCIAL else ""
        print(f"  {name:16s} {man.get('count'):>8,} rows{social}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
