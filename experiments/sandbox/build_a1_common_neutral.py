#!/usr/bin/env python3
"""A1 cross-scale audit, step 1 (CPU): freeze a NEUTRAL common probe.

Bug #5 (external review 2026-08-27). The old `a1-common` (built by
build_a1_common.py) is 250K rows sampled from the sealed 2M MiniLM pool
substrate (round-0216) -- i.e. it IS a subset of the 2M head's TRAINING set
(100% overlap for the 2M head; large overlap for the nested 6.25M..100M heads,
since those rungs are byte-identical-prefix draws from the same source pool).
Auditing on it unfairly favors lower-N heads.

This script freezes a probe that is HELD OUT from EVERY audited head. The heads
train on nested-prefix substrates drawn (uniform-without-replacement) from the
shared /data/embeddings MiniLM shards:

    2M    -> round-0216/queue-correction-3 ...minilm-mixed-2m...            (independent draw)
    6.25M -> round-0233 ...minilm-mixed-6250k-substrate-and-reserves-v1     (nested rung 1)
    12.5M -> round-0235 ...minilm-mixed-12500k-nested-substrate...          (nested rung 2)
    25M   -> round-0236/queue-correction-2 ...minilm-mixed-25000k-nested... (nested rung 3)
    50M   -> round-0237 ...minilm-mixed-50000k-nested...                    (nested rung 4)
    100M  -> round-0238 ...minilm-mixed-100000k-nested-substrate...         (nested rung 5)

The 6.25M..100M rungs are byte-identical prefixes of the 100M substrate
(verified via the R0238 nesting.ladder_prefix_ordered_sha256 chain), so the
100M training-row SET is a superset of every nested rung's. Therefore excluding
the union of

    (R0238 100M training provenance)  U  (R0216 2M training provenance)

removes every training row of every audited head from the candidate pool.

NEUTRAL-SOURCE NOTE / LIMITATION. The full 150M int8 pool (`minilm-int8-150m`)
is NOT on disk (checked 2026-08-27; only the per-head sealed substrates + the
source /data/embeddings shards exist). So we cannot sample "the pool minus the
100M head" from a single materialized pool file. Instead we sample directly
from the SOURCE MiniLM shards -- the same raw-headerless f32 shards every head's
substrate was drawn from -- and reject any (corpus, shard, row) that appears in
any head's training provenance. Each head's provenance is a sorted list of the
source coordinates it selected (decoupled from substrate row order), so the
exclusion is an EXACT index intersection, not an approximation. The resulting
probe is genuinely held out for all seven heads; residual_overlap.json records
the per-head intersection (0 by construction) as proof.

Composition matches the 100M training mix (fineweb .40 / redpajama .25 /
pile .25 / code .10) so the probe is in-distribution but never trained-on.

CPU only (a memmap gather + set intersections). Run:
    build_a1_common_neutral.py
Outputs:
    /data/latent-basemap/substrates/a1-common-neutral/substrate.f32.npy   (~250000 x 384 f32)
    /data/latent-basemap/substrates/a1-common-neutral/manifest.json
    /data/latent-basemap/substrates/a1-common-neutral/residual_overlap.json
    /data/latent-basemap/sandbox/a1-common-neutral/READY                   (marker, written last)
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

# canonical source-corpus ordering used by every provenance file's `corpus` index
# (verified empirically: R0238 provenance corpus 0 -> RedPajama, 3 -> starcoderdata,
#  matching the R0238 substrate.json `sources` dict order).
CORPUS_NAMES = [
    "RedPajama-Data-V2-sample-10B-chunked-120-all-MiniLM-L6-v2",
    "fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2",
    "pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2",
    "starcoderdata-code-chunked-120-all-MiniLM-L6-v2",
]
EMB_ROOT = Path("/data/embeddings")

# 100M training mix -> per-corpus probe quota (sums to N_SAMPLE).
CORPUS_QUOTA = {
    0: 62_500,   # redpajama .25
    1: 100_000,  # fineweb   .40
    2: 62_500,   # pile      .25
    3: 25_000,   # code      .10
}

# shards known-bad at 100M-substrate build time (excluded there); skip for sampling too.
BAD_SHARDS = {(1, 37)}  # fineweb data-00037-of-00099: partially-failed write, 24.89% zero rows

# head key -> training-provenance file. Used both to build the exclusion set
# (100M + 2M cover all rungs) and to report per-head residual overlap.
HEAD_PROVENANCE = {
    "2M": "/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
          "minilm-mixed-2m-substrate-and-exact-k15-graph-v1/provenance.npy",
    "6.25M": "/data/latent-basemap/runs/round-0233/queue/artifacts/"
             "minilm-mixed-6250k-substrate-and-reserves-v1/provenance.npy",
    "12.5M": "/data/latent-basemap/runs/round-0235/queue/artifacts/"
             "minilm-mixed-12500k-nested-substrate-and-reserves-v1/provenance.npy",
    "25M": "/data/latent-basemap/runs/round-0236/queue-correction-2/artifacts/"
           "minilm-mixed-25000k-nested-substrate-and-reserves-v1/provenance.npy",
    "50M": "/data/latent-basemap/runs/round-0237/queue/artifacts/"
           "minilm-mixed-50000k-nested-substrate-and-reserves-v1/provenance.npy",
    "100M": "/data/latent-basemap/runs/round-0238/queue/artifacts/"
            "minilm-mixed-100000k-nested-substrate-and-reserves-v1/provenance.npy",
}
# the two provenance files whose UNION is the full exclusion set.
EXCLUSION_HEADS = ["100M", "2M"]

OUT_DIR = Path("/data/latent-basemap/substrates/a1-common-neutral")
READY_DIR = Path("/data/latent-basemap/sandbox/a1-common-neutral")

# key encoding: corpus in [0,4), shard in [0,~200), row in [0,~2e6) -> unique int64.
K_CORPUS = 10 ** 15
K_SHARD = 10 ** 9


def _keys(prov: np.ndarray) -> np.ndarray:
    """(corpus,shard,row) structured provenance -> unique int64 keys."""
    return (prov["corpus"].astype(np.int64) * K_CORPUS
            + prov["shard"].astype(np.int64) * K_SHARD
            + prov["row"].astype(np.int64))


def _shard_index(path: str) -> int:
    # data-00037-of-00099.npy -> 37
    return int(os.path.basename(path).split("-")[1])


def _corpus_shards(ci: int) -> list[tuple[int, int, str]]:
    """on-disk (shard_idx, n_rows, path) for corpus ci, minus BAD_SHARDS."""
    out = []
    for p in sorted(glob.glob(str(EMB_ROOT / CORPUS_NAMES[ci] / "train" / "*.npy"))):
        s = _shard_index(p)
        if (ci, s) in BAD_SHARDS:
            continue
        n = os.path.getsize(p) // (DIM * 4)
        if n > 0:
            out.append((s, int(n), p))
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    READY_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # --- build exclusion set: union of 100M + 2M training provenance keys ------
    print("building exclusion set (union of every head's training rows)...", flush=True)
    excl_parts = []
    for h in EXCLUSION_HEADS:
        prov = np.load(HEAD_PROVENANCE[h], mmap_mode="r")
        k = _keys(np.asarray(prov))
        excl_parts.append(k)
        print(f"  {h}: {k.size:,} training rows", flush=True)
    exclusion = np.unique(np.concatenate(excl_parts))
    del excl_parts
    print(f"  exclusion set: {exclusion.size:,} unique source rows", flush=True)

    # corpora with few held-out rows (e.g. code: ~10M on disk, ~10M used by the
    # 100M head -> <100K held out) are enumerated exactly; large corpora with a
    # low exclusion rate are rejection-sampled. Threshold is on on-disk row count.
    ENUMERATE_MAX_ROWS = 20_000_000

    def _held_out_key(key: int) -> bool:
        i = np.searchsorted(exclusion, key)
        return not (i < exclusion.size and int(exclusion[i]) == key)

    def _not_degenerate(path_by_shard, mm_cache, s, r) -> bool:
        mm = mm_cache.get(s)
        if mm is None:
            mm = np.memmap(path_by_shard[s], dtype=np.float32, mode="r")
            mm_cache[s] = mm
        v = np.asarray(mm[r * DIM:(r + 1) * DIM], dtype=np.float32)
        return float(np.linalg.norm(v)) >= 0.5

    # --- draw the probe from source shards, rejecting excluded/degenerate rows -
    print(f"drawing {N_SAMPLE:,} neutral rows from source shards (seed {SEED})...", flush=True)
    chosen_keys: list[int] = []
    chosen_tuples: list[tuple[int, int, int]] = []
    seen: set[int] = set()

    for ci, quota in CORPUS_QUOTA.items():
        shards = _corpus_shards(ci)
        shard_ids = np.array([s for s, _, _ in shards], dtype=np.int64)
        shard_rows = np.array([n for _, n, _ in shards], dtype=np.int64)
        path_by_shard = {s: p for s, n, p in shards}
        mm_cache: dict[int, np.memmap] = {}
        total_rows = int(shard_rows.sum())
        got = 0

        if total_rows <= ENUMERATE_MAX_ROWS:
            # exact: enumerate all keys, subtract training, sample the complement.
            all_keys = np.concatenate([
                ci * K_CORPUS + s * K_SHARD + np.arange(n, dtype=np.int64)
                for s, n, _ in shards])
            held = np.setdiff1d(all_keys, exclusion, assume_unique=True)
            if held.size < quota:
                raise SystemExit(
                    f"corpus {ci}: only {held.size:,} held-out rows < quota {quota:,}")
            perm = rng.permutation(held.size)
            for idx in perm:
                if got >= quota:
                    break
                key = int(held[idx])
                s = int((key % K_CORPUS) // K_SHARD); r = int(key % K_SHARD)
                if not _not_degenerate(path_by_shard, mm_cache, s, r):
                    continue
                seen.add(key)
                chosen_keys.append(key)
                chosen_tuples.append((ci, s, r))
                got += 1
            if got < quota:
                raise SystemExit(f"corpus {ci}: exhausted held-out pool at {got}/{quota}")
        else:
            weights = shard_rows / shard_rows.sum()
            attempts = 0
            while got < quota:
                batch = max(quota - got, 4096) * 2
                si = rng.choice(len(shards), size=batch, p=weights)
                picks_shard = shard_ids[si]
                picks_row = (rng.random(batch) * shard_rows[si]).astype(np.int64)
                keys = ci * K_CORPUS + picks_shard * K_SHARD + picks_row
                loc = np.searchsorted(exclusion, keys)
                in_excl = (loc < exclusion.size) & (
                    exclusion[np.clip(loc, 0, exclusion.size - 1)] == keys)
                for j in range(batch):
                    if got >= quota:
                        break
                    if in_excl[j]:
                        continue
                    key = int(keys[j])
                    if key in seen:
                        continue
                    s = int(picks_shard[j]); r = int(picks_row[j])
                    if not _not_degenerate(path_by_shard, mm_cache, s, r):
                        continue
                    seen.add(key)
                    chosen_keys.append(key)
                    chosen_tuples.append((ci, s, r))
                    got += 1
                attempts += batch
                if attempts > quota * 200:
                    raise SystemExit(f"corpus {ci}: could not fill quota {quota} "
                                     f"(got {got}) after {attempts} draws")
        print(f"  corpus {ci} ({CORPUS_NAMES[ci].split('-')[0]}): {got:,} rows "
              f"({'enumerated' if total_rows <= ENUMERATE_MAX_ROWS else 'rejection-sampled'})",
              flush=True)

    n = len(chosen_tuples)
    assert n == N_SAMPLE, f"{n} != {N_SAMPLE}"

    # --- gather the vectors (sorted by tuple for efficient sequential reads) ---
    order = sorted(range(n), key=lambda i: chosen_tuples[i])
    sample = np.empty((n, DIM), dtype=np.float32)
    mm_cache = {}
    probe_prov = np.empty(n, dtype=[("corpus", "u1"), ("shard", "<u2"), ("row", "<i8")])
    for out_i, src_i in enumerate(order):
        ci, s, r = chosen_tuples[src_i]
        key = (ci, s)
        mm = mm_cache.get(key)
        if mm is None:
            mm = np.memmap(EMB_ROOT / CORPUS_NAMES[ci] / "train"
                           / os.path.basename(glob.glob(str(EMB_ROOT / CORPUS_NAMES[ci]
                             / "train" / f"data-{s:05d}-of-*.npy"))[0]),
                           dtype=np.float32, mode="r")
            mm_cache[key] = mm
        sample[out_i] = mm[r * DIM:(r + 1) * DIM]
        probe_prov[out_i] = (ci, s, r)

    out = OUT_DIR / "substrate.f32.npy"
    np.save(out, sample)
    np.save(OUT_DIR / "provenance.npy", probe_prov)
    probe_keys = np.sort(np.asarray(chosen_keys, dtype=np.int64))

    # --- residual overlap: exact intersection vs EACH head's training rows -----
    print("computing per-head residual overlap...", flush=True)
    residual = {}
    for h, ppath in HEAD_PROVENANCE.items():
        hk = np.unique(_keys(np.asarray(np.load(ppath, mmap_mode="r"))))
        inter = int(np.intersect1d(probe_keys, hk, assume_unique=False).size)
        residual[h] = {
            "training_rows": int(hk.size),
            "overlap_rows": inter,
            "overlap_pct_of_probe": round(100.0 * inter / n, 6),
            "provenance": ppath,
        }
        print(f"  {h:6s}: overlap {inter} / {n} probe rows "
              f"({100.0*inter/n:.6f}%)", flush=True)

    overlap_doc = {
        "schema": "a1-common-neutral-residual-overlap-2026-08-27",
        "probe_substrate": str(out),
        "probe_rows": n,
        "exclusion_source": "union of 100M (round-0238) + 2M (round-0216) training "
                            "provenance; covers all nested rungs (6.25M..50M are "
                            "byte-identical prefixes of the 100M substrate).",
        "method": "exact (corpus,shard,row) index intersection between the probe "
                  "and each head's sealed training provenance.npy",
        "per_head": residual,
        "note": "overlap is 0 by construction (probe rows were rejected if present "
                "in the 100M-or-2M exclusion union); table is the audit-integrity proof.",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    (OUT_DIR / "residual_overlap.json").write_text(json.dumps(overlap_doc, indent=1))

    # --- manifest --------------------------------------------------------------
    manifest = {
        "schema": "a1-common-neutral-2026-08-27",
        "purpose": "NEUTRAL common probe for the A1 cross-scale head audit -- "
                   "held out from every audited head's training set (Bug #5 fix).",
        "seed": SEED,
        "n_sample": n,
        "dim": DIM,
        "shape": list(sample.shape),
        "dtype": str(sample.dtype),
        "neutral_source": "source /data/embeddings MiniLM-L6-v2 shards (raw-headerless "
                          "f32), sampled uniformly per corpus MINUS every head's "
                          "training rows.",
        "full_pool_available": False,
        "full_pool_note": "minilm-int8-150m pool NOT on disk (checked 2026-08-27); "
                          "sampled from the source shards + exclusion instead. See "
                          "module docstring.",
        "composition_quota": {CORPUS_NAMES[c]: q for c, q in CORPUS_QUOTA.items()},
        "held_out_pool_note": "the code corpus (starcoderdata) is nearly fully "
                              "consumed by the 100M head (~10.0M of 10.1M on-disk "
                              "rows trained-on), so only ~98K code rows are held "
                              "out; its 25K quota is drawn by EXACT enumeration of "
                              "the held-out complement. The other three corpora are "
                              "rejection-sampled from their large held-out pools.",
        "exclusion_heads": EXCLUSION_HEADS,
        "excluded_bad_shards": [f"{CORPUS_NAMES[c]} shard {s}" for (c, s) in sorted(BAD_SHARDS)],
        "substrate": str(out),
        "provenance": str(OUT_DIR / "provenance.npy"),
        "residual_overlap": str(OUT_DIR / "residual_overlap.json"),
        "truth_graph_pending": "/data/latent-basemap/sandbox/a1-common-neutral/edges-k15-fuzzy.npz",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "note": "Replaces a1-common (which was a subset of the 2M head's training "
                "set). Every audited head sees this probe as held-out data.",
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))

    print(f"\na1-common-neutral: wrote {out} shape={sample.shape} dtype={sample.dtype}")
    print(f"  manifest:          {OUT_DIR / 'manifest.json'}")
    print(f"  residual_overlap:  {OUT_DIR / 'residual_overlap.json'}")

    # --- READY marker: only after probe + overlap table both exist -------------
    if out.exists() and (OUT_DIR / "residual_overlap.json").exists():
        (READY_DIR / "READY").write_text(
            f"a1-common-neutral built {manifest['created_utc']}\n"
            f"substrate={out}\nresidual_overlap={OUT_DIR / 'residual_overlap.json'}\n")
        print(f"  READY marker:      {READY_DIR / 'READY'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
