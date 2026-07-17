#!/usr/bin/env python3
"""
Scaling experiments for Parametric UMAP.

Tests training at increasing scales to verify:
1. Loss continues to decrease at each scale
2. Quality metrics remain stable / improve with more data
3. Training time scales linearly (not quadratically) with data size
4. Memory usage is bounded by batch size, not dataset size

Designed to run the small tiers locally (macOS/CPU) and larger tiers on GPU.

Usage:
    # Quick local test — 1K and 5K only
    python scale_experiment.py --tiers small

    # Medium test on GPU — 1K through 100K
    python scale_experiment.py --tiers medium --device cuda

    # Full scaling curve — 1K through 1M (needs GPU + time)
    python scale_experiment.py --tiers large --device cuda

    # Custom sizes
    python scale_experiment.py --sizes 1000 5000 20000 --device mps
"""

import argparse
import sys
import os
import time
import json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from basemap.pumap.parametric_umap import ParametricUMAP


# ─── Data Generators ─────────────────────────────────────────────────────────

def make_clustered_embeddings(n_samples, n_features=384, n_clusters=50, random_state=42):
    """
    Simulate embedding-model output: high-dimensional data with cluster structure.
    Uses Gaussian mixture to mimic what real embedding models produce.
    """
    rng = np.random.RandomState(random_state)

    # Generate cluster centers spread out in embedding space
    centers = rng.randn(n_clusters, n_features).astype(np.float32) * 3.0

    # Assign samples to clusters
    labels = rng.randint(0, n_clusters, size=n_samples)

    # Generate samples around centers with some noise
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    for i in range(n_samples):
        X[i] = centers[labels[i]] + rng.randn(n_features).astype(np.float32) * 0.3

    # Normalize to unit norm (like real embedding models)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(norms, 1e-8)

    return X, labels


# ─── Metrics (lightweight versions for scaling) ─────────────────────────────

def fast_distance_correlation(X_high, X_low, n_pairs=10000, random_state=42):
    """Sampled pairwise distance correlation."""
    rng = np.random.RandomState(random_state)
    n = X_high.shape[0]
    idx_i = rng.randint(0, n, size=n_pairs)
    idx_j = rng.randint(0, n, size=n_pairs)
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]
    d_high = np.linalg.norm(X_high[idx_i] - X_high[idx_j], axis=1)
    d_low = np.linalg.norm(X_low[idx_i] - X_low[idx_j], axis=1)
    return float(np.corrcoef(d_high, d_low)[0, 1])


def fast_knn_preservation(X_high, X_low, k=10, n_sample=2000, random_state=42):
    """KNN preservation on a random subset for speed."""
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.RandomState(random_state)
    n = X_high.shape[0]
    n_sample = min(n_sample, n)
    idx = rng.choice(n, size=n_sample, replace=False)
    X_h = X_high[idx]
    X_l = X_low[idx]

    k = min(k, n_sample - 1)
    nn_h = NearestNeighbors(n_neighbors=k+1).fit(X_h)
    nn_l = NearestNeighbors(n_neighbors=k+1).fit(X_l)
    _, idx_h = nn_h.kneighbors(X_h)
    _, idx_l = nn_l.kneighbors(X_l)
    idx_h = idx_h[:, 1:]
    idx_l = idx_l[:, 1:]
    preserved = sum(len(set(idx_h[i]) & set(idx_l[i])) for i in range(n_sample))
    return preserved / (n_sample * k)


# ─── Scaling Experiment ──────────────────────────────────────────────────────

TIER_SIZES = {
    "small": [1000, 5000],
    "medium": [1000, 5000, 10000, 50000, 100000],
    "large": [1000, 5000, 10000, 50000, 100000, 500000, 1000000],
}


def run_scale_tier(n_samples, n_features, args):
    """Run training at a given scale and return metrics."""
    print(f"\n{'─'*60}")
    print(f"  Scale: {n_samples:,} samples × {n_features} features")
    print(f"{'─'*60}")

    # Generate data
    t0 = time.time()
    X, labels = make_clustered_embeddings(n_samples, n_features=n_features)
    gen_time = time.time() - t0
    print(f"  Data generation: {gen_time:.1f}s")

    # Split 80/20
    n_train = int(n_samples * 0.8)
    rng = np.random.RandomState(42)
    perm = rng.permutation(n_samples)
    X_train = X[perm[:n_train]]
    X_test = X[perm[n_train:]]
    labels_test = labels[perm[n_train:]]

    # Adjust hyperparams by scale
    batch_size = min(args.batch_size, n_train // 4)
    batch_size = max(batch_size, 64)

    # Scale epochs inversely with data size (more data = fewer epochs needed)
    n_epochs = max(2, args.n_epochs)
    if n_samples >= 100000:
        n_epochs = max(2, n_epochs // 2)
    if n_samples >= 500000:
        n_epochs = max(1, n_epochs // 4)

    pumap = ParametricUMAP(
        n_components=2,
        hidden_dim=args.hidden_dim,
        n_layers=3,
        n_neighbors=min(15, n_train - 1),
        a=1.0,
        b=1.0,
        correlation_weight=0.1,
        learning_rate=args.learning_rate,
        n_epochs=n_epochs,
        batch_size=batch_size,
        device=args.device,
        pos_ratio=0.5,
    )

    # Train
    low_memory = n_samples > 50000 or args.device == 'cpu'
    t0 = time.time()
    pumap.fit(X_train, verbose=True, low_memory=low_memory)
    train_time = time.time() - t0

    # Transform
    t0 = time.time()
    Z_test = pumap.transform(X_test)
    transform_time = time.time() - t0

    # Metrics
    dc = fast_distance_correlation(X_test, Z_test)
    knn = fast_knn_preservation(X_test, Z_test, k=10)

    # Memory
    if torch.cuda.is_available() and 'cuda' in str(args.device):
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
    else:
        peak_mem_gb = None

    result = {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_train": n_train,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "train_time_s": train_time,
        "transform_time_s": transform_time,
        "samples_per_sec": n_train * n_epochs / train_time,
        "distance_correlation": dc,
        "knn_preservation_k10": knn,
        "peak_gpu_memory_gb": peak_mem_gb,
        "embedding_std": float(Z_test.std()),
    }

    print(f"\n  Results:")
    print(f"    Train time:          {train_time:.1f}s ({result['samples_per_sec']:.0f} samples/sec)")
    print(f"    Transform time:      {transform_time:.2f}s ({len(X_test)/transform_time:.0f} samples/sec)")
    print(f"    Distance corr:       {dc:.4f}")
    print(f"    KNN preservation:    {knn:.4f}")
    if peak_mem_gb is not None:
        print(f"    Peak GPU memory:     {peak_mem_gb:.2f} GB")

    return result


def main():
    from basemap.round0005_retirement import refuse_retired_launcher
    refuse_retired_launcher("scale_experiment.py")
    parser = argparse.ArgumentParser(description="Parametric UMAP scaling experiments")
    parser.add_argument("--tiers", type=str, default="small", choices=["small", "medium", "large"])
    parser.add_argument("--sizes", type=int, nargs="+", default=None, help="Custom sizes")
    parser.add_argument("--n-features", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="scale_results")
    args = parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    print(f"Device: {args.device}")

    sizes = args.sizes if args.sizes else TIER_SIZES[args.tiers]
    print(f"Scale tiers: {[f'{s:,}' for s in sizes]}")

    results = []
    for n in sizes:
        r = run_scale_tier(n, args.n_features, args)
        results.append(r)

    # Summary table
    print(f"\n{'='*80}")
    print(f"  SCALING SUMMARY")
    print(f"{'='*80}")
    print(f"  {'N':>10} {'Train(s)':>10} {'samp/sec':>10} {'Dist Corr':>10} {'KNN Pres':>10} {'GPU(GB)':>8}")
    print(f"  {'─'*58}")
    for r in results:
        gpu = f"{r['peak_gpu_memory_gb']:.1f}" if r['peak_gpu_memory_gb'] else "N/A"
        print(f"  {r['n_samples']:>10,} {r['train_time_s']:>10.1f} {r['samples_per_sec']:>10.0f} "
              f"{r['distance_correlation']:>10.4f} {r['knn_preservation_k10']:>10.4f} {gpu:>8}")

    # Check scaling properties
    print(f"\n  Scaling analysis:")
    if len(results) >= 2:
        # Training time should scale roughly linearly
        for i in range(1, len(results)):
            size_ratio = results[i]["n_samples"] / results[i-1]["n_samples"]
            time_ratio = results[i]["train_time_s"] / max(results[i-1]["train_time_s"], 0.1)
            print(f"    {results[i-1]['n_samples']:,} → {results[i]['n_samples']:,}: "
                  f"size ×{size_ratio:.1f}, time ×{time_ratio:.1f} "
                  f"({'linear' if time_ratio < size_ratio * 1.5 else 'SUPERLINEAR'})")

    # Save
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "scale_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {save_dir / 'scale_results.json'}")

    # Guidance for billion-scale
    print(f"\n{'='*80}")
    print(f"  ROADMAP TO BILLION-SCALE TRAINING")
    print(f"{'='*80}")
    print("""
  To scale to billions of embeddings:

  1. PRECOMPUTE GRAPH OFFLINE (most expensive step):
     - Use approximate k-NN (FAISS IVF, ScaNN, or DiskANN) to build the graph
     - Shard across machines: each shard computes k-NN for its partition
     - Save P_sym as sparse matrix shards (CSR format)
     - This is O(N log N) with ANN, not O(N²)

  2. PRECOMPUTE NEGATIVE EDGES:
     - Use the distributed_sample_negative_edges_subtask() already in edges_modal.py
     - Each shard samples negatives from its P_sym partition
     - Save as edge lists (pickle or parquet)

  3. STREAMING DATA LOADER:
     - Don't load all embeddings into memory
     - Use memory-mapped numpy arrays (MemmapArrayConcatenator) or LanceDB
     - DataPrefetcher already handles async GPU transfer

  4. TRAINING AT SCALE:
     - The model itself is tiny (MLP) — training is I/O bound, not compute bound
     - Use large batch sizes (4096-16384) to amortize I/O
     - Multi-GPU: use DDP on the model, data-parallel on edges
     - Expected: ~100K-500K samples/sec on A100

  5. VALIDATION AT SCALE:
     - Can't compute full pairwise metrics at billion scale
     - Use SAMPLED metrics: random 10K-50K subset for quality checks
     - Track loss curves — should decrease monotonically
     - Spot-check: embed known clusters, verify separation
     - Compare with standard UMAP on a 100K subset as reference

  6. KEY SCALING CHECKPOINTS:
     ┌──────────────┬──────────────┬───────────────────────┐
     │ Scale        │ Expected     │ Validate              │
     ├──────────────┼──────────────┼───────────────────────┤
     │ 10K          │ Minutes      │ Full metrics          │
     │ 100K         │ ~30 min      │ Full metrics          │
     │ 1M           │ ~2-4 hours   │ Sampled metrics       │
     │ 10M          │ ~12 hours    │ Sampled + spot-check  │
     │ 100M         │ ~2-3 days    │ Sampled + spot-check  │
     │ 1B           │ ~1-2 weeks   │ Loss curve + samples  │
     └──────────────┴──────────────┴───────────────────────┘
""")


if __name__ == "__main__":
    main()
