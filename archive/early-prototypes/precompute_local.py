#!/usr/bin/env python3
"""
Precompute P_sym and negative edges for latent-scope datasets.

Usage:
  python precompute_local.py --dataset ls-squad --n-neighbors 15 50 100
  python precompute_local.py --dataset ls-fineweb-edu-100k --n-neighbors 25 100
"""
import argparse
import time
import os
import pickle
import logging
import numpy as np
import h5py

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

DATASETS = {
    "ls-squad": {
        "h5_path": os.path.expanduser("~/latent-scope-demo/ls-squad/embeddings/embedding-003.h5"),
        "h5_dataset": "embeddings",
    },
    "ls-fineweb-edu-100k": {
        "h5_path": os.path.expanduser("~/latent-scope-demo/ls-fineweb-edu-100k/embeddings/embedding-001.h5"),
        "h5_dataset": "embeddings",
    },
}


def precompute(dataset_name, n_neighbors_list, output_dir="data/precomputed", n_neg_samples=5):
    """Precompute P_sym and negative edges for a dataset at various n_neighbors values.

    n_neg_samples: multiplier for negative edges per positive edge.
    """
    from basemap.pumap.parametric_umap.utils.graph import compute_all_p_umap
    from basemap.pumap.parametric_umap.datasets.edge_dataset import EdgeDataset

    ds = DATASETS[dataset_name]
    os.makedirs(output_dir, exist_ok=True)

    # Load embeddings
    logging.info(f"Loading {dataset_name} from {ds['h5_path']}")
    t0 = time.time()
    with h5py.File(ds["h5_path"], "r") as f:
        X = f[ds["h5_dataset"]][:].astype(np.float32)
    load_time = time.time() - t0
    logging.info(f"Loaded {X.shape} in {load_time:.1f}s")

    timings = {"dataset": dataset_name, "shape": list(X.shape), "load_s": load_time}

    for nn in n_neighbors_list:
        logging.info(f"\n{'='*60}")
        logging.info(f"  n_neighbors={nn}")
        logging.info(f"{'='*60}")

        prefix = f"{output_dir}/{dataset_name}_nn{nn}"

        # ── P_sym ──
        psym_path = f"{prefix}_psym.pkl"
        logging.info(f"Computing P_sym (n_neighbors={nn})...")
        t0 = time.time()
        P_sym = compute_all_p_umap(X, k=nn)
        psym_time = time.time() - t0
        logging.info(f"P_sym computed in {psym_time:.1f}s — {P_sym.nnz:,} non-zero entries")

        with open(psym_path, "wb") as f:
            pickle.dump(P_sym, f)
        psym_size = os.path.getsize(psym_path) / (1024 * 1024)
        logging.info(f"Saved to {psym_path} ({psym_size:.1f} MB)")

        # ── Negative edges ──
        neg_path = f"{prefix}_negatives.pkl"
        logging.info(f"Computing negative edges...")
        t0 = time.time()
        ed = EdgeDataset(P_sym)
        n_pos = len(ed.pos_edges)
        ed.sample_negative_edges(random_state=42, n_processes=6)
        neg_time = time.time() - t0
        n_neg = len(ed.neg_edges)
        logging.info(f"Negative edges computed in {neg_time:.1f}s — {n_pos:,} pos, {n_neg:,} neg")

        with open(neg_path, "wb") as f:
            pickle.dump(ed.neg_edges, f)
        neg_size = os.path.getsize(neg_path) / (1024 * 1024)
        logging.info(f"Saved to {neg_path} ({neg_size:.1f} MB)")

        timings[f"nn{nn}"] = {
            "psym_s": psym_time,
            "psym_nnz": P_sym.nnz,
            "psym_mb": psym_size,
            "neg_s": neg_time,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "neg_mb": neg_size,
            "total_s": psym_time + neg_time,
        }

        logging.info(f"  Total for nn={nn}: {psym_time + neg_time:.1f}s")

    # Print summary
    logging.info(f"\n{'='*60}")
    logging.info(f"  PRECOMPUTATION SUMMARY: {dataset_name}")
    logging.info(f"{'='*60}")
    logging.info(f"  Dataset: {X.shape[0]:,} samples, {X.shape[1]} dims")
    logging.info(f"  Data load: {load_time:.1f}s")
    logging.info(f"")
    logging.info(f"  {'nn':>4} {'P_sym':>8} {'Neg edges':>10} {'Total':>8} {'Pos edges':>12} {'Neg edges':>12} {'Disk':>8}")
    logging.info(f"  {'-'*70}")
    for nn in n_neighbors_list:
        t = timings[f"nn{nn}"]
        disk = t["psym_mb"] + t["neg_mb"]
        logging.info(f"  {nn:>4} {t['psym_s']:>7.1f}s {t['neg_s']:>9.1f}s {t['total_s']:>7.1f}s {t['n_pos']:>12,} {t['n_neg']:>12,} {disk:>7.1f}MB")

    return timings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--n-neighbors", nargs="+", type=int, default=[15, 50, 100])
    parser.add_argument("--output-dir", default="data/precomputed")
    parser.add_argument("--n-neg-samples", type=int, default=5,
                        help="Negative edges per positive edge")
    args = parser.parse_args()
    precompute(args.dataset, args.n_neighbors, args.output_dir, args.n_neg_samples)
