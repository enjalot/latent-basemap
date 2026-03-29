#!/usr/bin/env python3
"""
Small-scale validation experiment for Parametric UMAP.

Generates synthetic data (swiss roll, s-curve, clusters), trains parametric UMAP,
and computes quality metrics to verify the model is actually learning.

Runs on CPU (macOS), MPS (Apple Silicon), or CUDA.

Usage:
    python validate_umap.py                           # Quick smoke test (1K points)
    python validate_umap.py --n-samples 5000          # Medium test
    python validate_umap.py --n-samples 10000 --n-epochs 20  # Thorough test
    python validate_umap.py --dataset all             # Run all synthetic datasets
"""

import argparse
import sys
import os
import time
import json
import numpy as np
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from basemap.pumap.parametric_umap import ParametricUMAP


# ─── Synthetic Data Generators ───────────────────────────────────────────────

def make_swiss_roll(n_samples=1000, noise=0.5, random_state=42):
    """Swiss roll — classic manifold learning test."""
    from sklearn.datasets import make_swiss_roll as _make_swiss_roll
    X, color = _make_swiss_roll(n_samples=n_samples, noise=noise, random_state=random_state)
    return X.astype(np.float32), color, "swiss_roll"


def make_s_curve(n_samples=1000, noise=0.1, random_state=42):
    """S-curve — another classic manifold."""
    from sklearn.datasets import make_s_curve as _make_s_curve
    X, color = _make_s_curve(n_samples=n_samples, noise=noise, random_state=random_state)
    return X.astype(np.float32), color, "s_curve"


def make_blobs(n_samples=1000, n_features=50, n_clusters=10, random_state=42):
    """High-dimensional clusters — tests cluster separation."""
    from sklearn.datasets import make_blobs as _make_blobs
    X, labels = _make_blobs(
        n_samples=n_samples, n_features=n_features, centers=n_clusters,
        cluster_std=1.0, random_state=random_state)
    return X.astype(np.float32), labels.astype(np.float32), "blobs_d50"


def make_mnist_subset(n_samples=1000, random_state=42):
    """MNIST digits — real-world high-dimensional test (784D)."""
    try:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(mnist.data), size=min(n_samples, len(mnist.data)), replace=False)
        X = mnist.data[idx].astype(np.float32) / 255.0
        labels = mnist.target[idx].astype(np.float32)
        return X, labels, "mnist"
    except Exception as e:
        print(f"  [SKIP] MNIST not available: {e}")
        return None, None, None


# ─── Quality Metrics ─────────────────────────────────────────────────────────

def trustworthiness(X_high, X_low, k=10):
    """
    Trustworthiness: measures whether k-nearest neighbors in the low-D embedding
    are also neighbors in high-D. Score in [0, 1]; 1 is perfect.
    """
    from sklearn.metrics import pairwise_distances
    n = X_high.shape[0]
    k = min(k, n - 1)

    D_high = pairwise_distances(X_high)
    D_low = pairwise_distances(X_low)

    # Ranks in high-D and low-D spaces
    ranks_high = np.argsort(np.argsort(D_high, axis=1), axis=1)

    nn_low = np.argsort(D_low, axis=1)[:, 1:k+1]

    penalty = 0.0
    for i in range(n):
        for j in nn_low[i]:
            rank = ranks_high[i, j]
            if rank > k:
                penalty += rank - k

    max_penalty = n * k * (2 * n - 3 * k - 1) / 2.0
    return 1.0 - (2.0 / max_penalty) * penalty if max_penalty > 0 else 1.0


def distance_correlation(X_high, X_low, n_sample=5000, random_state=42):
    """
    Pearson correlation between pairwise distances in high-D and low-D.
    Samples pairs to keep it fast.
    """
    rng = np.random.RandomState(random_state)
    n = X_high.shape[0]
    n_pairs = min(n_sample, n * (n - 1) // 2)

    idx_i = rng.randint(0, n, size=n_pairs)
    idx_j = rng.randint(0, n, size=n_pairs)
    # Avoid self-pairs
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]

    d_high = np.linalg.norm(X_high[idx_i] - X_high[idx_j], axis=1)
    d_low = np.linalg.norm(X_low[idx_i] - X_low[idx_j], axis=1)

    return float(np.corrcoef(d_high, d_low)[0, 1])


def knn_preservation(X_high, X_low, k=10):
    """
    Fraction of k-nearest neighbors preserved from high-D to low-D.
    """
    from sklearn.neighbors import NearestNeighbors
    n = X_high.shape[0]
    k = min(k, n - 1)

    nn_high = NearestNeighbors(n_neighbors=k+1).fit(X_high)
    nn_low = NearestNeighbors(n_neighbors=k+1).fit(X_low)

    _, idx_high = nn_high.kneighbors(X_high)
    _, idx_low = nn_low.kneighbors(X_low)

    # Skip self (index 0)
    idx_high = idx_high[:, 1:]
    idx_low = idx_low[:, 1:]

    preserved = 0
    for i in range(n):
        preserved += len(set(idx_high[i]) & set(idx_low[i]))

    return preserved / (n * k)


def cluster_quality(X_low, labels, n_clusters=None):
    """
    Silhouette score of the embedding using ground-truth labels.
    """
    from sklearn.metrics import silhouette_score
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) > 50:
        return None
    return float(silhouette_score(X_low, labels.astype(int)))


# ─── Random Baseline ─────────────────────────────────────────────────────────

def random_embedding(n_samples, n_components=2, random_state=42):
    """Random projection baseline — if our model can't beat this, something is very wrong."""
    rng = np.random.RandomState(random_state)
    return rng.randn(n_samples, n_components).astype(np.float32)


# ─── Main Experiment ─────────────────────────────────────────────────────────

def run_experiment(X, color, name, args):
    """Run a single dataset experiment."""
    print(f"\n{'='*60}")
    print(f"  Dataset: {name} — {X.shape[0]} samples, {X.shape[1]} features")
    print(f"{'='*60}")

    # Train/test split for generalization check
    n = X.shape[0]
    n_train = int(n * 0.8)
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    X_train, X_test = X[perm[:n_train]], X[perm[n_train:]]
    color_train, color_test = color[perm[:n_train]], color[perm[n_train:]]

    print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Train parametric UMAP
    pumap = ParametricUMAP(
        n_components=2,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_neighbors=args.n_neighbors,
        a=1.0,      # Standard UMAP curve params
        b=1.0,
        correlation_weight=args.corr_weight,
        learning_rate=args.learning_rate,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        device=args.device,
        pos_ratio=0.5,
    )

    t0 = time.time()
    pumap.fit(X_train, verbose=True, low_memory=(args.device == 'cpu'))
    train_time = time.time() - t0
    print(f"\n  Training time: {train_time:.1f}s")

    # Embed train and test
    Z_train = pumap.transform(X_train)
    Z_test = pumap.transform(X_test)

    # Random baseline
    Z_random_train = random_embedding(len(X_train))
    Z_random_test = random_embedding(len(X_test))

    # Compute metrics on TRAIN set
    print(f"\n  --- TRAIN set metrics ---")
    metrics_train = compute_metrics(X_train, Z_train, color_train, "pumap")
    metrics_random_train = compute_metrics(X_train, Z_random_train, color_train, "random")

    # Compute metrics on TEST set (generalization!)
    print(f"\n  --- TEST set metrics (generalization) ---")
    metrics_test = compute_metrics(X_test, Z_test, color_test, "pumap")
    metrics_random_test = compute_metrics(X_test, Z_random_test, color_test, "random")

    # Summary comparison
    print(f"\n  --- Summary: {name} ---")
    print(f"  {'Metric':<25} {'Parametric UMAP':>15} {'Random':>15} {'Pass?':>8}")
    print(f"  {'-'*63}")
    all_pass = True
    for key in metrics_train:
        if metrics_train[key] is None:
            continue
        pumap_val = metrics_test[key]
        rand_val = metrics_random_test[key]
        passed = pumap_val > rand_val if key != 'embedding_spread' else True
        status = "OK" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {key:<25} {pumap_val:>15.4f} {rand_val:>15.4f} {status:>8}")

    # Embedding stats
    print(f"\n  Embedding stats (train):")
    print(f"    Mean: {Z_train.mean(axis=0)}")
    print(f"    Std:  {Z_train.std(axis=0)}")
    print(f"    Range: [{Z_train.min():.2f}, {Z_train.max():.2f}]")

    # Save checkpoint
    if args.save_dir:
        save_path = Path(args.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        model_path = save_path / f"pumap_{name}.pt"
        pumap.save(str(model_path))
        print(f"\n  Model saved to {model_path}")

        # Save results
        results = {
            "dataset": name,
            "n_samples": int(n),
            "n_features": int(X.shape[1]),
            "train_time_s": train_time,
            "metrics_train": {k: float(v) if v is not None else None for k, v in metrics_train.items()},
            "metrics_test": {k: float(v) if v is not None else None for k, v in metrics_test.items()},
            "metrics_random": {k: float(v) if v is not None else None for k, v in metrics_random_test.items()},
            "all_pass": all_pass,
        }
        with open(save_path / f"results_{name}.json", "w") as f:
            json.dump(results, f, indent=2)

    return all_pass


def compute_metrics(X, Z, labels, label_prefix):
    """Compute all quality metrics."""
    k = min(10, X.shape[0] - 1)
    metrics = {}

    tw = trustworthiness(X, Z, k=k)
    print(f"    [{label_prefix}] Trustworthiness (k={k}): {tw:.4f}")
    metrics["trustworthiness"] = tw

    dc = distance_correlation(X, Z)
    print(f"    [{label_prefix}] Distance correlation: {dc:.4f}")
    metrics["distance_corr"] = dc

    knn = knn_preservation(X, Z, k=k)
    print(f"    [{label_prefix}] KNN preservation (k={k}): {knn:.4f}")
    metrics["knn_preservation"] = knn

    spread = float(Z.std())
    print(f"    [{label_prefix}] Embedding spread (std): {spread:.4f}")
    metrics["embedding_spread"] = spread

    cq = cluster_quality(Z, labels)
    if cq is not None:
        print(f"    [{label_prefix}] Silhouette score: {cq:.4f}")
    metrics["silhouette"] = cq

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Parametric UMAP validation experiments")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--dataset", type=str, default="swiss_roll",
                        choices=["swiss_roll", "s_curve", "blobs", "mnist", "all"])
    parser.add_argument("--n-epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden layer dimension")
    parser.add_argument("--n-layers", type=int, default=3, help="Number of hidden layers")
    parser.add_argument("--n-neighbors", type=int, default=15, help="k for k-NN graph")
    parser.add_argument("--corr-weight", type=float, default=0.1, help="Correlation loss weight")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda, mps)")
    parser.add_argument("--save-dir", type=str, default="validation_results", help="Save directory")
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    print(f"Using device: {args.device}")

    datasets = {
        "swiss_roll": lambda: make_swiss_roll(args.n_samples),
        "s_curve": lambda: make_s_curve(args.n_samples),
        "blobs": lambda: make_blobs(args.n_samples),
        "mnist": lambda: make_mnist_subset(args.n_samples),
    }

    if args.dataset == "all":
        to_run = list(datasets.keys())
    else:
        to_run = [args.dataset]

    all_results = {}
    for ds_name in to_run:
        X, color, name = datasets[ds_name]()
        if X is None:
            continue
        passed = run_experiment(X, color, name, args)
        all_results[name] = passed

    # Final summary
    print(f"\n{'='*60}")
    print(f"  VALIDATION SUMMARY")
    print(f"{'='*60}")
    for name, passed in all_results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:<20} {status}")

    n_pass = sum(all_results.values())
    n_total = len(all_results)
    print(f"\n  {n_pass}/{n_total} datasets passed all checks")

    if n_pass < n_total:
        print("\n  Some checks failed. This could indicate:")
        print("  - Too few training epochs (try --n-epochs 20)")
        print("  - Learning rate too high/low (try --learning-rate 1e-3)")
        print("  - Insufficient model capacity (try --hidden-dim 512)")
        sys.exit(1)


if __name__ == "__main__":
    main()
