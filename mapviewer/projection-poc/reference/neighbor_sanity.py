"""Semantic sanity check for the projection chain.

For each reference string: find its nearest substrate rows in embedding space
(cosine, over a streamed sample of the 2M substrate) and compare their *sealed*
map coordinates with the coordinate the projection chain produced.  If the
frame/extent contract and the map head are wired correctly, a text lands where
its nearest corpus rows already live -- distances well under a percent of the
map's extent diagonal.

    /data/latent-basemap/envs/mappack-onnx/bin/python neighbor_sanity.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

POC = Path(__file__).resolve().parent.parent
SUBSTRATE = Path("/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
                 "minilm-mixed-2m-substrate-and-exact-k15-graph-v1/substrate.f32.npy")
COORDS = Path("/data/latent-basemap/sandbox/2m-knobs/umap-md000-x4-fneg10/coordinates.npy")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, default=Path(__file__).parent / "reference.json")
    ap.add_argument("--rows", type=int, default=500_000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=100_000)
    args = ap.parse_args()

    ref = json.loads(args.reference.read_text())
    Q = np.asarray([r["embedding"] for r in ref["results"]], dtype=np.float32)
    diag = ref["extent_diagonal"]

    X = np.load(SUBSTRATE, mmap_mode="r")
    Y = np.load(COORDS, mmap_mode="r")
    n = min(args.rows, len(X))
    best_sim = np.full((len(Q), args.k), -2.0, dtype=np.float32)
    best_idx = np.zeros((len(Q), args.k), dtype=np.int64)
    for start in range(0, n, args.chunk):           # streamed: never materialise X
        stop = min(start + args.chunk, n)
        block = np.asarray(X[start:stop], dtype=np.float32)
        sims = Q @ block.T
        for q in range(len(Q)):
            merged_s = np.concatenate([best_sim[q], sims[q]])
            merged_i = np.concatenate([best_idx[q], np.arange(start, stop)])
            top = np.argpartition(-merged_s, args.k)[:args.k]
            best_sim[q], best_idx[q] = merged_s[top], merged_i[top]

    rows = []
    for q, r in enumerate(ref["results"]):
        xy = np.asarray(r["xy"], dtype=np.float32)
        nb = np.asarray(Y[np.sort(best_idx[q])], dtype=np.float32)
        d = np.linalg.norm(nb - xy, axis=1)
        rows.append({"text": r["text"][:44].replace("\n", " "),
                     "top_cosine": float(best_sim[q].max()),
                     "median_neighbor_distance_frac_of_extent": float(np.median(d) / diag),
                     "max_neighbor_distance_frac_of_extent": float(d.max() / diag)})
        print(f"{rows[-1]['top_cosine']:.3f}  "
              f"median {rows[-1]['median_neighbor_distance_frac_of_extent']*100:6.3f}%  "
              f"max {rows[-1]['max_neighbor_distance_frac_of_extent']*100:6.3f}%  "
              f"{rows[-1]['text']!r}")
    out = Path(__file__).parent / "neighbor_sanity.json"
    out.write_text(json.dumps({"rows_scanned": int(n), "k": args.k, "rows": rows},
                              indent=1) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
