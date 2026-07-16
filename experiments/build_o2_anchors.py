"""O2 phase 2 prep — build the sparse landmark anchor file for the self-growth
frontier from a trained 4M control map (the teacher).

SELF-CONTAINED growth surrogate (flagged for review): the jina-en-4M-nested corpus
is NOT a clean nesting of the 2M testbed, so there is no ground-truth 2M→4M row
correspondence. Instead we designate a DETERMINISTIC random subset of the 4M rows
as "old" landmarks and take their teacher 2D coordinates from a trained unanchored
4M control (seed 42). Holding those landmarks fixed while the rest train freely
tests the sparse-hold MECHANISM (does it keep old points stable + preserve quality)
without claiming a real 2M→4M growth.

Writes an .npz (anchor_ids int64, anchor_targets float32) consumable by
train.anchor_ids_path.
"""
from __future__ import annotations
import argparse, os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from basemap.panel_v2 import load_coords, _ids_hash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True, help="teacher control run dir (coords.parquet)")
    ap.add_argument("--n-landmarks", type=int, default=2_000_000, help="# old points held")
    ap.add_argument("--n-total", type=int, default=4_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    Z, zid = load_coords(os.path.join(args.control, "coords.parquet"))
    if len(Z) != args.n_total:
        raise SystemExit(f"control has {len(Z)} rows != n_total {args.n_total}")
    # coords are row-ordered when zid is identity; align teacher coords to row id.
    if zid is not None and not np.array_equal(np.asarray(zid), np.arange(len(Z))):
        order = np.argsort(np.asarray(zid))
        Z = Z[order]                       # now Z[row] = teacher coord for that row
    # DETERMINISTIC landmark subset of [0, n_total)
    rng = np.random.RandomState(args.seed)
    ids = np.sort(rng.choice(args.n_total, size=args.n_landmarks, replace=False)).astype(np.int64)
    targets = np.asarray(Z[ids], dtype=np.float32)
    if not np.all(np.isfinite(targets)):
        raise SystemExit("teacher targets contain non-finite coords — refuse (O2).")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, anchor_ids=ids, anchor_targets=targets)
    meta = {"gate": "o2_sparse_anchors", "control": os.path.abspath(args.control),
            "n_landmarks": int(len(ids)), "n_total": int(args.n_total), "seed": args.seed,
            "anchor_id_hash": _ids_hash(ids),
            "target_hash": _ids_hash(np.round(targets, 4)),
            "surrogate": "self-contained: landmarks are a deterministic random subset of the "
                         "4M rows; teacher coords from the seed-42 unanchored control. NOT a "
                         "literal 2M->4M growth (no provenance correspondence exists). FLAGGED.",
            "out": args.out}
    json.dump(meta, open(os.path.splitext(args.out)[0] + ".meta.json", "w"), indent=1)
    print(f"[o2_anchors] {len(ids)} landmarks (hash {meta['anchor_id_hash']}) from {args.control} "
          f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
