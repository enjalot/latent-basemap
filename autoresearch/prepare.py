"""
Autoresearch preparation & evaluation harness for Parametric UMAP.
DO NOT MODIFY THIS FILE — it defines the fixed evaluation contract.

Loads precomputed data and provides evaluation functions.
Usage:
    python prepare.py              # verify data is ready
    python prepare.py --info       # print dataset stats
"""

import os
import sys
import time
import pickle
import numpy as np
import h5py
from pathlib import Path

# ─── Fixed Constants ────────────────────────────────────────────────────────

TIME_BUDGET = 180          # 3 minutes wall clock training time
INPUT_DIM = 768            # nomic-embed-text-v1.5 dimension
OUTPUT_DIM = 2             # 2D output
RANDOM_SEED = 42
TRAIN_FRACTION = 0.8       # 80/20 train/test split

# Data paths
DATA_DIR = Path(os.path.expanduser("~/latent-scope-demo"))
CACHE_DIR = Path("data/precomputed")

DATASET = "ls-squad"
H5_PATH = DATA_DIR / "ls-squad" / "embeddings" / "embedding-003.h5"
H5_DATASET = "embeddings"
REFERENCE_UMAP_PATH = DATA_DIR / "ls-squad" / "umaps" / "umap-001.parquet"

# Use nn=100 precomputed graph (best results from sweep)
PSYM_PATH = CACHE_DIR / "ls-squad_nn100_psym.pkl"
NEGATIVES_PATH = CACHE_DIR / "ls-squad_nn100_negatives.pkl"


# ─── Data Loading ───────────────────────────────────────────────────────────

def load_data():
    """Load embeddings, split into train/test, return everything needed.

    Returns dict with keys:
        X_train, X_test: np.ndarray float32
        Z_ref_train, Z_ref_test: np.ndarray float32 (reference UMAP 2D coords)
        P_sym: scipy sparse matrix
        neg_edges: list of (i, j) tuples
        perm: permutation indices used for split
    """
    # Load embeddings
    with h5py.File(H5_PATH, "r") as f:
        X = f[H5_DATASET][:].astype(np.float32)

    # Load reference UMAP
    import pandas as pd
    Z_ref = pd.read_parquet(REFERENCE_UMAP_PATH)[['x', 'y']].values.astype(np.float32)

    # Load precomputed graph
    with open(PSYM_PATH, "rb") as f:
        P_sym = pickle.load(f)

    with open(NEGATIVES_PATH, "rb") as f:
        neg_edges = pickle.load(f)

    # Fixed train/test split — MUST use full dataset (no split) because
    # precomputed graph indices correspond to original row order
    # Instead, we'll use a held-out evaluation subset
    n = len(X)
    rng = np.random.RandomState(RANDOM_SEED)
    perm = rng.permutation(n)
    n_train = int(n * TRAIN_FRACTION)

    # For evaluation we use a random subset, but training uses ALL data
    # (graph indices must stay aligned)
    eval_idx = perm[n_train:]  # ~4192 samples for evaluation

    return {
        "X": X,
        "Z_ref": Z_ref,
        "P_sym": P_sym,
        "neg_edges": neg_edges,
        "eval_idx": eval_idx,
        "n_samples": n,
        "input_dim": X.shape[1],
    }


# ─── Evaluation ─────────────────────────────────────────────────────────────

def evaluate(model, X, Z_ref, eval_idx, device="mps"):
    """Evaluate a trained model. Returns dict of metrics.

    Args:
        model: nn.Module that maps (batch, 768) -> (batch, 2)
        X: full dataset (n, 768) numpy array
        Z_ref: reference UMAP coords (n, 2) numpy array
        eval_idx: indices of held-out evaluation samples
        device: torch device string

    Returns dict with:
        knn_10, knn_25, knn_50: KNN preservation at different k
        trustworthiness: sklearn trustworthiness
        dist_corr: sampled distance correlation (high-dim vs low-dim)
        ref_knn_overlap: KNN overlap with reference UMAP in 2D
        ref_procrustes: Procrustes disparity vs reference UMAP
    """
    import torch
    from sklearn.neighbors import NearestNeighbors
    from sklearn.manifold import trustworthiness as sk_trustworthiness
    from scipy.spatial import procrustes

    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        # Transform in batches to avoid OOM
        Z_all = []
        for i in range(0, len(X_tensor), 4096):
            Z_all.append(model(X_tensor[i:i+4096]).cpu().numpy())
        Z = np.concatenate(Z_all, axis=0)

    # Use eval subset for metrics
    X_eval = X[eval_idx]
    Z_eval = Z[eval_idx]
    Z_ref_eval = Z_ref[eval_idx]

    results = {}
    n = len(X_eval)

    # ── KNN Preservation at multiple k values ──
    for k in [10, 25, 50]:
        k_actual = min(k, n - 1)
        nn_h = NearestNeighbors(n_neighbors=k_actual+1, n_jobs=-1).fit(X_eval)
        nn_l = NearestNeighbors(n_neighbors=k_actual+1, n_jobs=-1).fit(Z_eval)
        _, idx_h = nn_h.kneighbors(X_eval)
        _, idx_l = nn_l.kneighbors(Z_eval)
        idx_h, idx_l = idx_h[:, 1:], idx_l[:, 1:]
        preserved = sum(len(set(idx_h[i]) & set(idx_l[i])) for i in range(n))
        results[f"knn_{k}"] = preserved / (n * k_actual)

    # ── Trustworthiness ──
    results["trustworthiness"] = float(sk_trustworthiness(X_eval, Z_eval, n_neighbors=10))

    # ── Sampled Distance Correlation ──
    rng = np.random.RandomState(42)
    n_pairs = min(10000, n * (n - 1) // 2)
    i = rng.randint(0, n, n_pairs)
    j = rng.randint(0, n, n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    dh = np.linalg.norm(X_eval[i] - X_eval[j], axis=1)
    dl = np.linalg.norm(Z_eval[i] - Z_eval[j], axis=1)
    results["dist_corr"] = float(np.corrcoef(dh, dl)[0, 1])

    # ── Reference UMAP comparison ──
    # Procrustes alignment
    sample_n = min(2000, n)
    rng2 = np.random.RandomState(42)
    sidx = rng2.choice(n, sample_n, replace=False)
    _, _, disparity = procrustes(Z_ref_eval[sidx], Z_eval[sidx])
    results["ref_procrustes"] = float(disparity)

    # KNN overlap with reference in 2D
    k_ref = 10
    nn_ref = NearestNeighbors(n_neighbors=k_ref+1, n_jobs=-1).fit(Z_ref_eval)
    nn_par = NearestNeighbors(n_neighbors=k_ref+1, n_jobs=-1).fit(Z_eval)
    _, idx_ref = nn_ref.kneighbors(Z_ref_eval)
    _, idx_par = nn_par.kneighbors(Z_eval)
    idx_ref, idx_par = idx_ref[:, 1:], idx_par[:, 1:]
    overlap = sum(len(set(idx_ref[i]) & set(idx_par[i])) for i in range(n))
    results["ref_knn_overlap"] = overlap / (n * k_ref)

    return results


def print_results(results):
    """Print results in the standard format for grep extraction."""
    print("---")
    for k, v in results.items():
        print(f"{k}: {v:.6f}")


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from basemap.round0005_retirement import refuse_retired_launcher
    refuse_retired_launcher("autoresearch/prepare.py")
    print("Checking data availability...")

    missing = []
    for name, path in [("H5 embeddings", H5_PATH), ("Reference UMAP", REFERENCE_UMAP_PATH),
                        ("P_sym", PSYM_PATH), ("Negatives", NEGATIVES_PATH)]:
        if not path.exists():
            missing.append(f"  MISSING: {name} at {path}")
            print(f"  ✗ {name}: {path}")
        else:
            size_mb = path.stat().st_size / (1024**2)
            print(f"  ✓ {name}: {path} ({size_mb:.1f} MB)")

    if missing:
        print("\nMissing files! Run precompute_local.py first:")
        print("  python precompute_local.py --dataset ls-squad --n-neighbors 100")
        sys.exit(1)

    if "--info" in sys.argv:
        data = load_data()
        print(f"\nDataset: {DATASET}")
        print(f"  Samples: {data['n_samples']:,}")
        print(f"  Input dim: {data['input_dim']}")
        print(f"  Eval samples: {len(data['eval_idx']):,}")
        print(f"  P_sym nnz: {data['P_sym'].nnz:,}")
        print(f"  Negative edges: {len(data['neg_edges']):,}")
        print(f"  Time budget: {TIME_BUDGET}s ({TIME_BUDGET//60} min)")

    print("\nReady for autoresearch!")
