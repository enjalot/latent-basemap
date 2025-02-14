#!/usr/bin/env python
"""
Local execution of negative edge precomputation (without Modal).
"""

import logging
import argparse
import numpy as np
import os

from basemap.data_loader import MemmapArrayConcatenator
from basemap.lancedb_loader import LanceDBLoader
from basemap.pumap.parametric_umap.utils.graph import compute_and_save_all_p_umap
from basemap.pumap.parametric_umap.datasets.edge_dataset import EdgeDataset

# Constants and defaults (modify these as needed)
# DATASET = ["/embeddings/wikipedia-en-chunked-120-all-MiniLM-L6-v2/train",]
DATASET = "/Users/enjalot/latent-scope-demo/ls-fineweb-edu-100k/lancedb"
TABLE = "scopes-001"
# D_IN = 384
D_IN = 768
N_NEIGHBORS = 15

CPU_CONCURRENCY = 64

# Path to save computed UMAP results (you can change this path as needed)
UMAP_RESULTS_FILE = "data/precomputed_umap_results_ls-fineweb-edu-100k.pkl"
# NEGATIVE_EDGES_FILE = "/checkpoints/pumap/wikipedia-en-chunked-120-all-MiniLM-L6-v2/precomputed_negatives.pkl"
NEGATIVE_EDGES_FILE = "data/precomputed_negatives_lance_ls-fineweb-edu-100k.pkl"

def main(random_state=0):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    print("Starting UMAP computation and negative edge precomputation locally with random_state:", random_state)
    
    # Load training data via memmap arrays (keeping memory usage low)
    # X_train = MemmapArrayConcatenator(DATASET, D_IN)
    X_train = LanceDBLoader(db_name=DATASET, table_name=TABLE, columns=["vector"])
    print("Loaded training data LanceDB with len:", len(X_train))
    
    # Compute (or load) the symmetric probability matrix (P_sym) for gathering positive edges
    print("Computing/loading symmetric probability matrix (P_sym)...")
    P_sym = compute_and_save_all_p_umap(X_train, k=N_NEIGHBORS, file_path=UMAP_RESULTS_FILE)
    
    # Create the EdgeDataset instance and display information on positive edges
    ed = EdgeDataset(P_sym)
    print("EdgeDataset created with", len(ed.pos_edges), "positive edges.")
    
    # Precompute negative edges and save them to file
    print("Precomputing negative edges and saving to", NEGATIVE_EDGES_FILE)
    ed.precompute_and_save_negative_edges(NEGATIVE_EDGES_FILE,
                                          random_state=random_state,
                                          n_processes=CPU_CONCURRENCY,
                                          verbose=True)
    print("Negative edge precomputation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Negative Edge Precomputation")
    parser.add_argument("--random-state", type=int, default=0,
                        help="Random state for precomputation")
    args = parser.parse_args()
    main(args.random_state) 