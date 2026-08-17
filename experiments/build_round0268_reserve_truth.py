#!/usr/bin/env python3
"""Build the R0268 100M reserve-neighbour truth for the held-out FFR instrument.

The R0265 family FFR floor's instrument is the OUT-OF-SUBSTRATE reserve projection: the
held-out reserve queries `reserve.f32[reserve-query-rows]` projected THROUGH each trained
map, scored against those queries' EXACT top-10 cosine neighbours in the FULL substrate
(indices INTO the substrate == into each map's coordinates). This script builds that truth
for the 100M rung — the exact-cosine top-10 of the 2000 R0238 held-out reserve queries over
all 100,000,000 R0238 substrate rows.

Mirrors the sealed 50M reserve-truth recipe (round-0267/ffr-correction/reserve-truth-50m):
brute-force, CPU, streamed in row chunks, L2-normalise queries + each substrate chunk before
the dot product. Writes `truth-top10.npy` (2000×10 int64 indices), `truth-top10-scores.npy`
(2000×10 float32 cosines), and a `receipt.json` (params + reserve identity binding + sanity
checks + sha256s). This is a CPU-only, read-only build; it trains nothing and touches no GPU.

Run (after the round is issued, before `prepare_round0268_queue.py`):

    CUDA_VISIBLE_DEVICES="" /home/enjalot/code/latent-basemap/.venv/bin/python \
        experiments/build_round0268_reserve_truth.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from typing import Any

import numpy as np

SUBSTRATE = (
    "/data/latent-basemap/runs/round-0238/queue/artifacts/"
    "minilm-mixed-100000k-nested-substrate-and-reserves-v1/substrate.f32.npy"
)
RESERVE = (
    "/data/latent-basemap/runs/round-0238/queue/artifacts/"
    "minilm-mixed-100000k-nested-substrate-and-reserves-v1/reserve.f32.npy"
)
RESERVE_QUERY_ROWS = (
    "/data/latent-basemap/runs/round-0238/queue/artifacts/"
    "minilm-mixed-100000k-nested-substrate-and-reserves-v1/reserve-query-rows.i64.npy"
)
OUT_DIR = "/data/latent-basemap/runs/round-0268/ffr/reserve-truth-100m"

ROWS = 100_000_000
DIMENSION = 384
K_TRUE = 10
#: Substrate rows per chunk. Bounds the (n_queries × chunk) score matrix under ~2 GB float32
#: (2000 × 250,000 × 4 B ≈ 2.0 GB); 200,000 keeps it at ~1.6 GB with margin.
SUBSTRATE_CHUNK_ROWS = 200_000


def _sha256(path: str, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _l2_normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def build(*, out_dir: str = OUT_DIR, chunk_rows: int = SUBSTRATE_CHUNK_ROWS) -> dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    started = time.time()
    substrate = np.load(SUBSTRATE, mmap_mode="r", allow_pickle=False)
    if substrate.shape != (ROWS, DIMENSION) or substrate.dtype != np.float32:
        raise RuntimeError(f"substrate geometry changed: {substrate.shape} {substrate.dtype}")
    reserve = np.load(RESERVE, mmap_mode="r", allow_pickle=False)
    query_rows = np.load(RESERVE_QUERY_ROWS, allow_pickle=False).astype(np.int64, copy=False)
    if reserve.ndim != 2 or reserve.shape[1] != DIMENSION or query_rows.ndim != 1:
        raise RuntimeError("reserve / query-rows geometry changed")
    queries = _l2_normalise(np.asarray(reserve[query_rows], dtype=np.float32))
    n_q = int(queries.shape[0])

    # Running top-K over the streamed substrate: keep the best K scores + indices per query.
    best_scores = np.full((n_q, K_TRUE), -np.inf, dtype=np.float32)
    best_idx = np.full((n_q, K_TRUE), -1, dtype=np.int64)

    for start in range(0, ROWS, chunk_rows):
        stop = min(start + chunk_rows, ROWS)
        block = _l2_normalise(np.asarray(substrate[start:stop], dtype=np.float32))
        scores = queries @ block.T  # (n_q, block_rows) cosines
        block_idx = np.arange(start, stop, dtype=np.int64)
        # Merge this chunk's candidates with the running best, then re-select top-K.
        cand_scores = np.concatenate([best_scores, scores], axis=1)
        cand_idx = np.concatenate(
            [best_idx, np.broadcast_to(block_idx, (n_q, stop - start))], axis=1
        )
        # Partial top-K per row (descending): argpartition then sort the K survivors.
        part = np.argpartition(-cand_scores, K_TRUE - 1, axis=1)[:, :K_TRUE]
        rows_ax = np.arange(n_q)[:, None]
        top_scores = cand_scores[rows_ax, part]
        order = np.argsort(-top_scores, axis=1)
        best_scores = top_scores[rows_ax, order]
        best_idx = cand_idx[rows_ax, part][rows_ax, order]
        if (start // chunk_rows) % 25 == 0:
            print(json.dumps({
                "progress_rows": stop, "of": ROWS, "elapsed_s": round(time.time() - started, 1)
            }), flush=True)

    truth_path = os.path.join(out_dir, "truth-top10.npy")
    scores_path = os.path.join(out_dir, "truth-top10-scores.npy")
    np.save(truth_path, best_idx)
    np.save(scores_path, best_scores)

    sanity = {
        "all_finite": bool(np.isfinite(best_scores).all()),
        "scores_in_[-1,1]": bool((best_scores >= -1.0001).all() and (best_scores <= 1.0001).all()),
        "indices_in_[0,N)": bool((best_idx >= 0).all() and (best_idx < ROWS).all()),
        "no_negative_index": bool((best_idx >= 0).all()),
        "per_row_10_distinct_neighbours": bool(
            all(len(set(best_idx[i].tolist())) == K_TRUE for i in range(n_q))
        ),
        "top1_score_max": float(best_scores[:, 0].max()),
        "top1_score_min": float(best_scores[:, 0].min()),
        "note_own_index": (
            "queries index into reserve (separate array); truth indexes into substrate; by "
            "construction no query's own row appears. top1<1.0 confirms reserve is "
            "out-of-substrate (no exact duplicates)."
        ),
    }
    receipt = {
        "artifact": "round-0268 100M reserve-neighbour truth (held-out FFR)",
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "truth_top10_npy": truth_path,
        "truth_top10_sha256": _sha256(truth_path),
        "truth_top10_scores_npy": scores_path,
        "truth_top10_scores_sha256": _sha256(scores_path),
        "shape": [n_q, K_TRUE],
        "params": {
            "k_true": K_TRUE,
            "metric": "cosine, exact (brute-force, CPU, streamed)",
            "chunk_rows": chunk_rows,
            "n_queries": n_q,
            "n_substrate": ROWS,
            "normalisation": "L2-normalise queries and each substrate chunk before dot product",
        },
        "reserve_identity_binding": {
            "reserve_f32_npy": RESERVE,
            "reserve_f32_sha256": _sha256(RESERVE),
            "reserve_query_rows_i64_npy": RESERVE_QUERY_ROWS,
            "reserve_query_rows_sha256": _sha256(RESERVE_QUERY_ROWS),
            "substrate_f32_npy": SUBSTRATE,
            "note": (
                "queries = reserve.f32[reserve-query-rows]; truth = exact top-10 cosine "
                "substrate neighbours (out-of-substrate reserve projection recipe, mirrors "
                "the R0265 2M / R0267 50M reserve-truth meta)."
            ),
        },
        "sanity": sanity,
        "wall_s": round(time.time() - started, 1),
    }
    with open(os.path.join(out_dir, "receipt.json"), "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=1)
    print(json.dumps({
        "truth_top10_npy": truth_path,
        "truth_top10_sha256": receipt["truth_top10_sha256"],
        "sanity_ok": all(v for k, v in sanity.items() if isinstance(v, bool)),
        "wall_s": receipt["wall_s"],
    }))
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="build the R0268 100M reserve-neighbour truth")
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--chunk-rows", type=int, default=SUBSTRATE_CHUNK_ROWS)
    args = parser.parse_args(argv)
    build(out_dir=args.out_dir, chunk_rows=args.chunk_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
