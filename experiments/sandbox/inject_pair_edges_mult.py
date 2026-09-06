"""exp-2c pair-edge injection by MULTIPLICITY (owner 2026-09-06; fixes the inert w_pair sweep). The champion's
uniform edge sampler IGNORES edge weights, so weighting pair edges (inject_pair_edges.py, w_mult) was inert
(all arms bit-identical). MULTIPLICITY works: duplicate each injected pair edge M times so the uniform sampler
draws it ~M x more often. Adds symmetric edges i <-> N+i, TILED M times, for the 90% TRAIN pairs only; the 10%
HELD-OUT pairs get NO injected edge — scoring on them tests whether the injection GENERALIZES (do held-out
partners land together without a direct edge) vs merely memorizes injected edges.

Usage: inject_pair_edges_mult.py <in_edges.npz> <out_edges.npz> <M> <N_pairs> [holdout_frac=0.1] [seed=0]
Writes out_edges.npz + <out_edges>.meta.json (M, train/holdout pair counts, holdout_ids for the scorer).
"""
import sys, json
from pathlib import Path
import numpy as np


def main():
    infile, outfile, M, N = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    hf = float(sys.argv[5]) if len(sys.argv) > 5 else 0.1
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    z = np.load(infile)
    src, tgt, wt = z["sources"], z["targets"], z["weights"]; n_nodes = int(z["n_nodes"])
    med = float(np.median(wt))                          # keep pair-edge weight at the within-modality median

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N); n_hold = int(round(N * hf))
    holdout = np.sort(perm[:n_hold]); train = np.sort(perm[n_hold:])   # pair ids (image row i, text row N+i)

    # tile each TRAIN pair's symmetric edge M times
    i = train; j = train + N
    add_src = np.tile(np.concatenate([i, j]), M)
    add_tgt = np.tile(np.concatenate([j, i]), M)
    add_wt = np.full(add_src.shape[0], med, dtype=np.float32)
    out_src = np.concatenate([src, add_src]); out_tgt = np.concatenate([tgt, add_tgt])
    out_wt = np.concatenate([wt, add_wt])
    np.savez(outfile, sources=out_src, targets=out_tgt, weights=out_wt, n_nodes=np.int64(n_nodes))

    hid = Path(outfile).with_suffix(".holdout_pairs.npy"); np.save(hid, holdout)
    meta = {"schema": "exp2c-mult-inject-2026-09-06", "source_graph": infile, "M": M,
            "median_edge_weight": round(med, 5), "N_pairs": N, "n_train_pairs": int(train.size),
            "n_holdout_pairs": int(holdout.size), "holdout_frac": hf, "seed": seed,
            "orig_edges": int(len(src)), "injected_pair_edges": int(add_src.shape[0]),
            "injected_per_train_pair": 2 * M, "holdout_pairs_file": str(hid),
            "note": "TRAIN pairs injected Mx (multiplicity, sampler respects it); HELD-OUT pairs no edge (generalization test)."}
    Path(outfile).with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[inject M={M}] +{add_src.shape[0]:,} pair edges ({train.size} train pairs x2x{M}); "
          f"{holdout.size} held-out pairs -> {outfile}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
