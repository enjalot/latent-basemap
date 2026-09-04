"""exp-2c: inject matched-pair edges into the centered fuzzy graph (owner-approved, overseer 2026-09-04).
Adds symmetric edges i <-> N+i for all N pairs to the existing centered kNN/fuzzy graph, at weight
w_pair = w_mult * median(existing edge weights). The rank probe showed the centered kNN is ~all within-modality,
so median(weights) is the within-modality reference; w_mult sweeps {0.5,1,2} the pair-vs-within-modality trade.
Usage: inject_pair_edges.py <in_edges.npz> <out_edges.npz> <w_mult> <N_pairs>"""
import sys, json
from pathlib import Path
import numpy as np


def main():
    infile, outfile, w_mult, N = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
    z = np.load(infile)
    src, tgt, wt = z["sources"], z["targets"], z["weights"]; n_nodes = int(z["n_nodes"])
    assert n_nodes == 2 * N, f"n_nodes {n_nodes} != 2*N {2*N}"
    med = float(np.median(wt)); w_pair = w_mult * med
    i = np.arange(N, dtype=np.int32)
    # symmetric pair edges: i -> N+i and N+i -> i
    add_src = np.concatenate([i, (i + N).astype(np.int32)])
    add_tgt = np.concatenate([(i + N).astype(np.int32), i])
    add_wt = np.full(2 * N, w_pair, dtype=np.float32)
    out_src = np.concatenate([src, add_src]); out_tgt = np.concatenate([tgt, add_tgt])
    out_wt = np.concatenate([wt, add_wt])
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    np.savez(outfile, sources=out_src, targets=out_tgt, weights=out_wt, n_nodes=np.int64(n_nodes))
    meta = {"source_graph": infile, "w_mult": w_mult, "median_edge_weight": round(med, 5),
            "w_pair": round(w_pair, 5), "orig_edges": int(len(src)), "injected_pair_edges": int(2 * N),
            "total_edges": int(len(out_src)), "note": "symmetric matched-pair edges i<->N+i on the centered graph"}
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
