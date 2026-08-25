#!/usr/bin/env python3
"""P3 curation-validation loop, step 1: build substrates (b)/(c) + freeze scorecard.

Three substrates enter the loop:

  (a) current mix     the R0216 2M "queue-correction-3" substrate (already on disk)
  (b) curated 2M      built here: 4M candidate -> near-dup dedup -> inverse-density
                      subsample to 2M (soft head-capping, favors sparse regions)
  (c) random 2M       built here: plain uniform draw at the 40/25/25/10 mix

This script:
  1. builds (c) random and (b) curated (write-once / idempotent), and
  2. computes and FREEZES the (a)/(b)/(c) diversity scorecard to disk BEFORE any
     (b)/(c) map is trained -- a pre-registration guardrail so the predicted
     diversity order cannot be retrofitted to the OOD-probe result.

Mixture (FineWeb-edu 40% / RedPajama 25% / Pile 25% / starcoder 10%):
    random 2M candidate counts   800k / 500k / 500k / 200k
    curated 4M candidate counts   1.6M / 1.0M / 1.0M / 400k

Source embeddings are 384-d all-MiniLM-L6-v2, already unit-norm f32 -- the SAME
precision as (a).  fineweb/redpajama/pile shards are *headerless raw float32*
buffers (dim 384); starcoder shards are real .npy.  `_open_shard` handles both,
and floors a truncated shard to whole rows.

The GPU step (exact k15 kNN over the 4M candidate) is delegated to
`image_map_pipeline.knn`; the orchestrator runs this script when the GPU is free.
The pure helpers (`_sample_corpus_rows`, `_inverse_density_choice`,
`_dedup_survivors`) are import-safe and unit-tested on tiny arrays.

Usage:
    p3_build_and_scorecard.py            # build (c),(b) then freeze scorecard
    p3_build_and_scorecard.py random     # build (c) only
    p3_build_and_scorecard.py curated    # build (b) only
    p3_build_and_scorecard.py scorecard  # freeze scorecard only (needs a,b,c)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

DIM = 384
SEED = 42
K = 15
DEDUP_EPS = 0.005
GAMMA = 1.0

N_TARGET = 2_000_000          # (b) and (c) final row count
N_CANDIDATE = 4_000_000       # (b) candidate pool before dedup/subsample

SANDBOX = Path("/data/latent-basemap/sandbox")
SUBSTRATES = Path("/data/latent-basemap/substrates")
CANDIDATE_DS = "minilm-curated-candidate"   # temp ds name for the GPU knn build

# (a) current-mix substrate (already on disk).
A_PATH = Path(
    "/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
    "minilm-mixed-2m-substrate-and-exact-k15-graph-v1/substrate.f32.npy")
B_DIR = SUBSTRATES / "minilm-curated-2m"
C_DIR = SUBSTRATES / "minilm-random-2m"

CORPORA = {
    "fineweb": "/data/embeddings/fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2/train",
    "redpajama": "/data/embeddings/RedPajama-Data-V2-sample-10B-chunked-120-all-MiniLM-L6-v2/train",
    "pile": "/data/embeddings/pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2/train",
    "starcoder": "/data/embeddings/starcoderdata-code-chunked-120-all-MiniLM-L6-v2/train",
}
MIX = {"fineweb": 0.40, "redpajama": 0.25, "pile": 0.25, "starcoder": 0.10}


def _mix_counts(total: int) -> dict[str, int]:
    """Split ``total`` across corpora by MIX, giving any rounding slack to the
    largest share so the counts sum EXACTLY to ``total``."""
    counts = {c: int(round(total * f)) for c, f in MIX.items()}
    drift = total - sum(counts.values())
    counts["fineweb"] += drift
    return counts


# ── shard IO ──────────────────────────────────────────────────────────────────

def _open_shard(path: str):
    """Return a read-only (rows, DIM) float32 view of a shard.

    Real .npy shards (starcoder) open via ``np.load(mmap_mode='r')``; the
    fineweb/redpajama/pile shards are headerless raw float32 and open via
    ``np.memmap`` with the row count floored from the file size (one fineweb
    shard is truncated by 802 trailing bytes -- flooring drops the partial row).
    """
    with open(path, "rb") as fh:
        is_npy = fh.read(6) == b"\x93NUMPY"
    if is_npy:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    rows = os.path.getsize(path) // (DIM * 4)
    return np.memmap(path, dtype=np.float32, mode="r", shape=(rows, DIM))


def _corpus_shards(corpus: str):
    import glob
    files = sorted(glob.glob(os.path.join(CORPORA[corpus], "*.npy")))
    if not files:
        raise FileNotFoundError(f"no shards for corpus {corpus} at {CORPORA[corpus]}")
    return [_open_shard(f) for f in files]


# ── pure helpers (unit-tested) ─────────────────────────────────────────────────

def _sample_corpus_rows(shards, n: int, rng, dim: int = DIM):
    """Gather ``n`` uniform-random rows drawn without replacement from the global
    row space spanned by ``shards`` (a list of (rows_i, dim) array-likes).

    Returns ``(rows, record)`` where ``rows`` is a contiguous (n, dim) float32
    array and ``record`` documents the draw.  Pure w.r.t. the RNG: the same
    ``rng`` state + shard row counts always yield the same rows.  Works on tiny
    in-memory arrays (tests) and on memmaps (production) alike; per-shard gathers
    are issued in ascending local-index order for memmap locality.
    """
    shard_rows = [int(s.shape[0]) for s in shards]
    total = int(sum(shard_rows))
    n = int(min(n, total))
    gidx = np.sort(rng.choice(total, size=n, replace=False))
    offsets = np.concatenate([[0], np.cumsum(shard_rows)]).astype(np.int64)
    shard_of = np.searchsorted(offsets, gidx, side="right") - 1
    local = gidx - offsets[shard_of]
    out = np.empty((n, dim), dtype=np.float32)
    for si in range(len(shards)):
        mask = shard_of == si
        if not mask.any():
            continue
        loc = local[mask]
        order = np.argsort(loc)          # ascending local order for locality
        dest = np.nonzero(mask)[0][order]
        out[dest] = np.asarray(shards[si][loc[order]], dtype=np.float32)
    record = {"total_rows": total, "sampled": n, "shard_rows": shard_rows}
    return out, record


def _inverse_density_choice(mean_dists, n: int, rng, gamma: float = GAMMA):
    """Choose ``n`` row indices without replacement with probability proportional
    to ``mean_dists ** gamma`` (larger mean kNN distance = sparser = more likely
    kept).  Soft head-capping: dense regions are down-weighted, not clipped.

    Guards: non-finite / all-zero / degenerate weights fall back to a uniform
    draw; ``n >= len(mean_dists)`` returns all indices (sorted).
    """
    m = np.asarray(mean_dists, dtype=np.float64)
    N = len(m)
    if n >= N:
        return np.arange(N, dtype=np.int64)
    w = np.power(np.clip(m, 0.0, None), gamma)
    s = w.sum()
    if not np.isfinite(s) or s <= 0.0:
        w = np.ones(N, dtype=np.float64)
    p = w / w.sum()
    return np.sort(rng.choice(N, size=n, replace=False, p=p)).astype(np.int64)


def _dedup_survivors(groups, n_total: int):
    """Keep one representative row per near-dup group.

    ``groups`` is the per-row group-label array (``labels`` from
    ``dup_analysis.dup_groups``), length ``n_total``.  Returns the sorted row
    indices of the first-seen member of each group -- i.e. one survivor per
    connected component, near-duplicate rows dropped.
    """
    labels = np.asarray(groups)
    if len(labels) != n_total:
        raise ValueError(f"labels length {len(labels)} != n_total {n_total}")
    _, first_idx = np.unique(labels, return_index=True)
    return np.sort(first_idx).astype(np.int64)


# ── builders ───────────────────────────────────────────────────────────────────

def _write_memmap_from_corpora(out_path: Path, counts: dict[str, int], rng,
                               shuffle: bool):
    """Fill ``out_path`` (a fresh .npy memmap) with rows sampled per ``counts``.

    Records the per-corpus subset labels; optionally applies an in-place chunked
    permutation so corpus blocks are not contiguous.  Returns (records, subsets).
    """
    total = int(sum(counts.values()))
    out = np.lib.format.open_memmap(str(out_path), mode="w+",
                                    dtype=np.float32, shape=(total, DIM))
    subsets = np.empty(total, dtype=object)
    records = {}
    pos = 0
    for corpus, cnt in counts.items():
        shards = _corpus_shards(corpus)
        rows, rec = _sample_corpus_rows(shards, cnt, rng)
        out[pos:pos + len(rows)] = rows
        subsets[pos:pos + len(rows)] = corpus
        records[corpus] = rec
        pos += len(rows)
        print(f"    {corpus}: +{len(rows):,} rows (of {rec['total_rows']:,})", flush=True)
        del rows, shards
    if shuffle:
        perm = rng.permutation(total)
        subsets = subsets[perm]
        tmp = out_path.with_suffix(".shuf.tmp.npy")
        shuf = np.lib.format.open_memmap(str(tmp), mode="w+",
                                         dtype=np.float32, shape=(total, DIM))
        CH = 200_000
        for i in range(0, total, CH):
            shuf[i:i + CH] = out[perm[i:i + CH]]
        shuf.flush()
        del out, shuf
        os.replace(tmp, out_path)
    else:
        out.flush()
        del out
    return records, subsets


def build_random(rng) -> None:
    """(c) plain uniform 2M at the 40/25/25/10 mix, shuffled."""
    C_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = C_DIR / "substrate.f32.npy"
    if sub_path.exists():
        print(f"(c) random: {sub_path} exists, skip")
        return
    counts = _mix_counts(N_TARGET)
    print(f"(c) random 2M build: {counts}")
    t0 = time.time()
    records, _ = _write_memmap_from_corpora(sub_path, counts, rng, shuffle=True)
    (C_DIR / "manifest.json").write_text(json.dumps({
        "kind": "random", "rows": N_TARGET, "dim": DIM, "dtype": "float32",
        "mix": MIX, "counts": counts, "seed": SEED, "shuffled": True,
        "per_corpus": records,
        "note": "uniform random draw at the 40/25/25/10 mix; no dedup / no "
                "density reweighting; unit-norm f32 (same precision as a).",
    }, indent=1))
    print(f"(c) random: wrote {sub_path} in {(time.time()-t0)/60:.1f} min")


def build_curated(rng_cand, rng_density) -> None:
    """(b) 4M candidate -> dedup -> inverse-density subsample to 2M."""
    B_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = B_DIR / "substrate.f32.npy"
    if sub_path.exists():
        print(f"(b) curated: {sub_path} exists, skip")
        return

    # 1. candidate (write-once)
    cand_path = B_DIR / "candidate.f32.npy"
    cand_counts = _mix_counts(N_CANDIDATE)
    if cand_path.exists():
        print(f"(b) candidate: {cand_path} exists, skip build")
        cand_records = None
    else:
        print(f"(b) curated candidate 4M build: {cand_counts}")
        t0 = time.time()
        cand_records, _ = _write_memmap_from_corpora(
            cand_path, cand_counts, rng_cand, shuffle=False)
        print(f"(b) candidate: wrote {cand_path} in {(time.time()-t0)/60:.1f} min")

    n_cand = int(np.load(cand_path, mmap_mode="r").shape[0])

    # 2. exact k15 kNN over the candidate (GPU; via image_map_pipeline)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import image_map_pipeline as imp
    knn_dir = SANDBOX / CANDIDATE_DS
    if not (knn_dir / "knn_dists.npy").exists():
        imp.DATASETS[CANDIDATE_DS] = {
            "load": lambda: np.load(cand_path, mmap_mode="r"), "subsets": None}
        print(f"(b) curated: building exact k{K} knn over {n_cand:,} candidate rows (GPU)")
        imp.knn(CANDIDATE_DS)

    # 3. dedup: one representative per near-dup group
    from dup_analysis import dup_groups
    labels, counts = dup_groups(CANDIDATE_DS, DEDUP_EPS)
    survivors = _dedup_survivors(labels, n_cand)
    print(f"(b) curated: dedup {n_cand:,} -> {len(survivors):,} survivors "
          f"(eps={DEDUP_EPS}); {n_cand - len(survivors):,} near-dups dropped")

    # 4. inverse-density subsample survivors -> 2M
    knn_d = np.load(knn_dir / "knn_dists.npy", mmap_mode="r")
    mean_d = np.asarray(knn_d[survivors], dtype=np.float64).mean(axis=1)
    if len(survivors) < N_TARGET:
        raise RuntimeError(
            f"only {len(survivors):,} survivors < target {N_TARGET:,}; "
            "lower dedup eps or raise candidate size")
    chosen_local = _inverse_density_choice(mean_d, N_TARGET, rng_density, GAMMA)
    final_rows = np.sort(survivors[chosen_local])

    # 5. gather into the output substrate memmap (chunked; candidate never fully
    #    materialized)
    cand = np.load(cand_path, mmap_mode="r")
    out = np.lib.format.open_memmap(str(sub_path), mode="w+",
                                    dtype=np.float32, shape=(N_TARGET, DIM))
    CH = 200_000
    for i in range(0, N_TARGET, CH):
        out[i:i + CH] = np.asarray(cand[final_rows[i:i + CH]], dtype=np.float32)
    out.flush()
    del out
    np.save(B_DIR / "selected_candidate_rows.npy", final_rows.astype(np.int64))

    kept_mean = float(mean_d[chosen_local].mean())
    all_mean = float(mean_d.mean())
    (B_DIR / "manifest.json").write_text(json.dumps({
        "kind": "curated", "rows": N_TARGET, "dim": DIM, "dtype": "float32",
        "mix": MIX, "candidate_counts": cand_counts,
        "candidate_rows": n_cand, "seed": SEED,
        "dedup_eps": DEDUP_EPS, "survivors_after_dedup": int(len(survivors)),
        "near_dups_dropped": int(n_cand - len(survivors)),
        "inverse_density_gamma": GAMMA,
        "mean_knn_dist_kept": kept_mean, "mean_knn_dist_survivors": all_mean,
        "candidate_per_corpus": cand_records,
        "candidate_knn": str(knn_dir),
        "note": "4M candidate at the 40/25/25/10 mix -> union-find near-dup "
                "dedup (one rep/group) -> inverse-density subsample to 2M "
                "(P[keep] proportional to mean_k15_dist^gamma); soft head-cap, "
                "favors sparse regions. Unit-norm f32 (same precision as a).",
    }, indent=1))
    print(f"(b) curated: wrote {sub_path} "
          f"(kept mean k15 dist {kept_mean:.4f} vs survivor mean {all_mean:.4f})")


# ── scorecard ───────────────────────────────────────────────────────────────────

def _ensure_paths() -> None:
    """Make the sibling helper modules importable regardless of CWD."""
    here = Path(__file__).resolve().parent
    for p in (here, here.parent, here.parent / "metrics"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _substrate_scores(path: Path, anchor_ids: np.ndarray) -> dict:
    """vendi + kNN-ball radius summary for one substrate (CPU-only)."""
    _ensure_paths()
    from vendi import vendi_cosine
    from density_v3 import high_d_radii

    X = np.load(path, mmap_mode="r")
    n = int(X.shape[0])
    ids = anchor_ids[anchor_ids < n]
    # Drop degenerate (near-zero-norm) filler rows from the kNN-ball anchors: they are
    # kept in the substrate per the keep-degenerate-filler policy but are not unit-norm,
    # so they skew the radius metric and trip high_d_radii's unit-norm check. vendi_cosine
    # guards zero rows internally, so it is computed on the full substrate.
    anc = np.asarray(X[ids], dtype=np.float32)
    keep = np.linalg.norm(anc, axis=1) >= 0.5
    n_dropped = int((~keep).sum())
    ids = ids[keep]
    vendi = float(vendi_cosine(X))
    radii = high_d_radii(X, ids, k=K)
    return {
        "path": str(path), "rows": n,
        "degenerate_anchors_dropped": n_dropped,
        "vendi": vendi,
        "knn_ball_mean": float(np.mean(radii)),
        "knn_ball_median": float(np.median(radii)),
        "knn_ball_p10": float(np.percentile(radii, 10)),
        "knn_ball_p90": float(np.percentile(radii, 90)),
        "n_anchors": int(len(ids)),
    }


def write_scorecard() -> None:
    _ensure_paths()

    substrates = {"a": A_PATH, "b": B_DIR / "substrate.f32.npy",
                  "c": C_DIR / "substrate.f32.npy"}
    # deterministic anchor draw (seed 42); all three substrates are 2M rows.
    anchor_ids = np.sort(np.random.default_rng(SEED).choice(
        N_TARGET, size=20_000, replace=False)).astype(np.int64)

    scores: dict[str, dict] = {}
    for key, path in substrates.items():
        if not Path(path).exists():
            scores[key] = {"error": f"missing substrate: {path}", "path": str(path)}
            print(f"scorecard {key}: MISSING {path}")
            continue
        try:
            t0 = time.time()
            scores[key] = _substrate_scores(Path(path), anchor_ids)
            print(f"scorecard {key}: vendi={scores[key]['vendi']:.2f} "
                  f"knn_ball_mean={scores[key]['knn_ball_mean']:.4f} "
                  f"({(time.time()-t0)/60:.1f} min)")
        except Exception as e:  # never let one substrate crash the others
            scores[key] = {"error": f"{type(e).__name__}: {e}", "path": str(path)}
            print(f"scorecard {key}: ERROR {e}")

    ok = {k: v for k, v in scores.items() if "vendi" in v}
    predicted = sorted(ok, key=lambda k: ok[k]["vendi"], reverse=True)
    out = SANDBOX / "p3-scorecard.json"
    out.write_text(json.dumps({
        "frozen_utc_note": "pre-registration, computed before any (b)/(c) training",
        "schema": "p3-scorecard-2026-08-25",
        "k": K, "anchor_seed": SEED, "n_anchors_requested": 20_000,
        "substrates": scores,
        "predicted_diversity_order": predicted,
    }, indent=1))
    print(f"scorecard frozen: {out}  predicted diversity order (vendi desc): {predicted}")


def main() -> int:
    ss = np.random.SeedSequence(SEED)
    rng_random, rng_cand, rng_density = (np.random.default_rng(s) for s in ss.spawn(3))
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "random"):
        build_random(rng_random)
    if what in ("all", "curated"):
        build_curated(rng_cand, rng_density)
    if what in ("all", "scorecard"):
        write_scorecard()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
