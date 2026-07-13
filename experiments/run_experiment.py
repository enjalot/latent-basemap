#!/usr/bin/env python3
from __future__ import annotations
"""
Unified experiment runner for Parametric UMAP.

Usage:
    # Run a single experiment from config
    python -m experiments.run_experiment experiments/configs/swiss_roll_smoke.yaml

    # Run with CLI overrides
    python -m experiments.run_experiment experiments/configs/scaling_10k.yaml \
        --override train.n_epochs=20 train.batch_size=1024

    # Run a sweep
    python -m experiments.run_experiment experiments/configs/arch_sweep_base.yaml --sweep

    # List what a config would do without running
    python -m experiments.run_experiment experiments/configs/scaling_10k.yaml --dry-run
"""

import argparse
import sys
import os
import json
import time
import logging
import yaml
import platform
import socket
import subprocess
from pathlib import Path
from datetime import datetime

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.experiment_config import (
    ExperimentConfig, load_config, generate_sweep_configs
)
from experiments.artifact_cache import build_graph_cache_spec, write_manifest_if_missing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)


# ─── Run persistence ─────────────────────────────────────────────────────────

def _write_coords_parquet(path, Z, ls_index):
    """Write the final projection as coords.parquet with one float32 column
    per component (x, y[, z, c3, c4, ...]) and ls_index (int64 original row
    index). Historically hardcoded x,y, which silently dropped z on
    n_components=3 runs."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    Z = np.asarray(Z, dtype=np.float32)
    ls_index = np.asarray(ls_index, dtype=np.int64)
    names = ["x", "y", "z"] + [f"c{i}" for i in range(3, Z.shape[1])]
    cols = {names[i]: pa.array(Z[:, i], type=pa.float32()) for i in range(Z.shape[1])}
    cols["ls_index"] = pa.array(ls_index, type=pa.int64())
    pq.write_table(pa.table(cols), path)


def _write_anchor_targets_parquet(path, T, ls_index):
    """Write the deterministic anchored-init targets as anchor_targets.parquet
    (columns x, y, ls_index) so stability analysis can reuse the exact targets
    the encoder was pretrained on."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    T = np.asarray(T, dtype=np.float32)
    ls_index = np.asarray(ls_index, dtype=np.int64)
    table = pa.table({
        "x": pa.array(T[:, 0], type=pa.float32()),
        "y": pa.array(T[:, 1], type=pa.float32()),
        "ls_index": pa.array(ls_index, type=pa.int64()),
    })
    pq.write_table(table, path)


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_data(cfg: ExperimentConfig) -> np.ndarray:
    """Load data according to the data config. Returns (X, labels_or_None)."""
    dc = cfg.data

    if dc.source == "synthetic":
        return _load_synthetic(dc)
    elif dc.source == "h5":
        return _load_h5(dc)
    elif dc.source == "memmap":
        return _load_memmap(dc)
    elif dc.source == "lancedb":
        return _load_lancedb(dc)
    else:
        raise ValueError(f"Unknown data source: {dc.source}")


def _load_synthetic(dc):
    if np is None:
        raise RuntimeError("numpy is required to load synthetic data")
    from sklearn.datasets import make_swiss_roll, make_blobs
    n = dc.n_samples or 1000
    X, labels = make_blobs(n_samples=n, n_features=dc.input_dim, centers=20,
                           cluster_std=1.0, random_state=dc.random_seed)
    return X.astype(np.float32), labels.astype(np.float32)


def _load_h5(dc):
    if np is None:
        raise RuntimeError("numpy is required to load H5 data")
    import h5py

    if not dc.h5_path:
        raise ValueError("data.h5_path is required when data.source='h5'")

    logging.info("Loading H5 dataset %s:%s", dc.h5_path, dc.h5_dataset)
    with h5py.File(os.path.expanduser(dc.h5_path), "r") as f:
        X = f[dc.h5_dataset][:]

    X = np.asarray(X, dtype=np.float32)
    if dc.n_samples and dc.n_samples < len(X):
        rng = np.random.RandomState(dc.random_seed)
        idx = rng.choice(len(X), dc.n_samples, replace=False)
        X = X[idx]
    return X, None


def _load_memmap(dc):
    if np is None:
        raise RuntimeError("numpy is required to load memmap data")
    from basemap.data_loader import MemmapArrayConcatenator
    loader = MemmapArrayConcatenator(dc.memmap_dirs, dc.input_dim)
    logging.info(f"Memmap data shape: {loader.shape}")
    if dc.n_samples and dc.n_samples < len(loader):
        rng = np.random.RandomState(dc.random_seed)
        idx = rng.choice(len(loader), dc.n_samples, replace=False)
        X = loader[idx]
    else:
        # Keep the loader lazy — do NOT materialise the full corpus (>=2 GB
        # rule). Downstream indexing (splits, metrics, edge-list fit, transform)
        # handles the lazy MemmapArrayConcatenator per-batch/per-subset.
        X = loader
    return X, None


# ─── Reproducibility Metadata ────────────────────────────────────────────────

def _run_cmd(args):
    try:
        out = subprocess.check_output(args, cwd=Path(__file__).resolve().parents[1],
                                      stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except Exception:
        return None


def _file_manifest(path):
    if not path:
        return None
    p = Path(os.path.expanduser(path))
    if not p.exists():
        return {"path": str(p), "exists": False}
    stat = p.stat()
    return {
        "path": str(p),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


def _dir_manifest(paths):
    manifests = []
    for raw in paths or []:
        p = Path(os.path.expanduser(raw))
        entry = {"path": str(p), "exists": p.exists()}
        if p.exists():
            npy_files = sorted(p.glob("*.npy"))
            entry.update({
                "n_npy_files": len(npy_files),
                "size_bytes": sum(f.stat().st_size for f in npy_files),
                "first_files": [str(f) for f in npy_files[:5]],
            })
        manifests.append(entry)
    return manifests


def collect_run_manifest(cfg: ExperimentConfig, device: str, eval_mode: str) -> dict:
    dc = cfg.data
    manifest = {
        "git": {
            "commit": _run_cmd(["git", "rev-parse", "HEAD"]),
            "branch": _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty_status": _run_cmd(["git", "status", "--short"]),
        },
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "device": {
            "selected": device,
            "torch": getattr(torch, "__version__", None) if torch is not None else None,
            "cuda_available": torch.cuda.is_available() if torch is not None else False,
            "cuda_device_name": (
                torch.cuda.get_device_name(0)
                if torch is not None and torch.cuda.is_available()
                else None
            ),
        },
        "data_assets": {
            "h5": _file_manifest(dc.h5_path),
            "reference_umap": _file_manifest(dc.reference_umap_path),
            "precomputed_p_sym": _file_manifest(dc.precomputed_p_sym_path),
            "precomputed_negatives": _file_manifest(dc.precomputed_negatives_path),
            "precomputed_edges": _file_manifest(dc.precomputed_edges_path),
            "precomputed_index": _file_manifest(dc.precomputed_index_path),
            "memmap_dirs": _dir_manifest(dc.memmap_dirs),
        },
        "eval_contract": {
            "mode": eval_mode,
            "note": (
                "transductive_full_graph: train uses all rows because precomputed graph indices "
                "must align; metrics are sampled from rows seen during training."
                if eval_mode == "transductive_full_graph"
                else "holdout_rows: graph/model train on train rows only; metrics_test uses held-out rows."
            ),
        },
        "effective_config": cfg.to_dict(),
    }
    return manifest


def _load_lancedb(dc):
    if np is None:
        raise RuntimeError("numpy is required to load LanceDB data")
    from basemap.lancedb_loader import LanceDBLoader
    loader = LanceDBLoader(db_name=dc.lancedb_path, table_name=dc.lancedb_table,
                           columns=dc.lancedb_columns)
    logging.info(f"LanceDB data shape: {loader.shape}")
    X = np.asarray(loader).astype(np.float32)
    if dc.n_samples and dc.n_samples < len(X):
        rng = np.random.RandomState(dc.random_seed)
        idx = rng.choice(len(X), dc.n_samples, replace=False)
        X = X[idx]
    return X, None


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(X_high, X_low, cfg: ExperimentConfig, labels=None) -> dict:
    """Compute requested evaluation metrics."""
    ec = cfg.eval
    n = X_high.shape[0]
    sample_n = min(ec.metric_sample_size, n)

    # Subsample for speed on large datasets
    if sample_n < n:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, sample_n, replace=False)
        X_h = X_high[idx]
        X_l = X_low[idx]
        lab = labels[idx] if labels is not None else None
    else:
        X_h, X_l, lab = X_high, X_low, labels

    results = {}
    k = min(ec.knn_k, sample_n - 1)

    if "trustworthiness" in ec.metrics:
        from sklearn.manifold import trustworthiness as sk_tw
        results["trustworthiness"] = float(sk_tw(X_h, X_l, n_neighbors=k))

    if "distance_correlation" in ec.metrics:
        results["distance_correlation"] = _sampled_dist_corr(X_h, X_l)

    if "knn_preservation" in ec.metrics:
        results["knn_preservation"] = _knn_preservation(X_h, X_l, k)

    if "silhouette" in ec.metrics and lab is not None:
        from sklearn.metrics import silhouette_score
        unique = np.unique(lab)
        if 2 <= len(unique) <= 100:
            results["silhouette"] = float(silhouette_score(X_l, lab.astype(int)))

    return results


def _sampled_dist_corr(X_h, X_l, n_pairs=10000):
    rng = np.random.RandomState(42)
    n = len(X_h)
    n_pairs = min(n_pairs, n * (n - 1) // 2)
    i = rng.randint(0, n, n_pairs)
    j = rng.randint(0, n, n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    dh = np.linalg.norm(X_h[i] - X_h[j], axis=1)
    dl = np.linalg.norm(X_l[i] - X_l[j], axis=1)
    return float(np.corrcoef(dh, dl)[0, 1])


def _knn_preservation(X_h, X_l, k):
    from sklearn.neighbors import NearestNeighbors
    k = min(k, len(X_h) - 1)
    nn_h = NearestNeighbors(n_neighbors=k+1, n_jobs=-1).fit(X_h)
    nn_l = NearestNeighbors(n_neighbors=k+1, n_jobs=-1).fit(X_l)
    _, idx_h = nn_h.kneighbors(X_h)
    _, idx_l = nn_l.kneighbors(X_l)
    idx_h, idx_l = idx_h[:, 1:], idx_l[:, 1:]
    preserved = sum(len(set(idx_h[i]) & set(idx_l[i])) for i in range(len(X_h)))
    return preserved / (len(X_h) * k)


def set_global_seeds(seed: int):
    """Seed numpy and torch for reproducible experiment runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Standard UMAP Baseline ─────────────────────────────────────────────────

def run_umap_baseline(X_train, X_test, cfg: ExperimentConfig) -> dict:
    """Run standard UMAP on the same data as a comparison baseline."""
    try:
        import umap
    except ImportError:
        logging.warning("umap-learn not installed, skipping baseline. pip install umap-learn")
        return {}

    logging.info("Running standard UMAP baseline...")
    kwargs = cfg.eval.umap_kwargs.copy()
    t0 = time.time()
    reducer = umap.UMAP(**kwargs)
    Z_train = reducer.fit_transform(X_train)
    umap_train_time = time.time() - t0

    # Standard UMAP can't easily transform held-out data without parametric version,
    # but the UMAP library does support it
    t0 = time.time()
    Z_test = reducer.transform(X_test)
    umap_transform_time = time.time() - t0

    metrics_train = compute_metrics(X_train, Z_train, cfg)
    metrics_test = compute_metrics(X_test, Z_test, cfg)

    return {
        "train_time_s": umap_train_time,
        "transform_time_s": umap_transform_time,
        "metrics_train": metrics_train,
        "metrics_test": metrics_test,
    }


# ─── Core Runner ─────────────────────────────────────────────────────────────

def run_single_experiment(cfg: ExperimentConfig) -> dict:
    """Run one experiment end-to-end. Returns results dict."""
    run_dir = cfg.run_dir()
    os.makedirs(run_dir, exist_ok=True)

    # Save config for reproducibility
    cfg.to_yaml(os.path.join(run_dir, "config.yaml"))

    logging.info(f"=" * 60)
    logging.info(f"Experiment: {cfg.name}")
    logging.info(f"Results dir: {run_dir}")
    logging.info(f"=" * 60)

    # ── Load data ──
    set_global_seeds(cfg.data.random_seed)

    t0 = time.time()
    X, labels = load_data(cfg)
    data_load_time = time.time() - t0
    logging.info(f"Data loaded: {X.shape} in {data_load_time:.1f}s")

    # ── Train/test split ──
    # When precomputed graphs are provided, their edge indices correspond to the
    # original row order of X. Permuting X before passing it to fit() would
    # misalign the graph indices, so we skip the split in that case and train on
    # the full dataset.
    n = len(X)
    using_precomputed = bool(
        cfg.data.precomputed_p_sym_path
        or cfg.data.precomputed_negatives_path
        or cfg.data.precomputed_edges_path
        or cfg.data.precomputed_index_path
    )
    if using_precomputed:
        logging.warning(
            "Precomputed graph paths are set — skipping train/test split to keep "
            "graph indices aligned with X. All %d samples will be used for training.", n
        )
        X_train, X_test = X, None
        labels_train = labels
        labels_test = None
        # ls_index is the original row order (edge indices align to it).
        train_indices = np.arange(n, dtype=np.int64)
    else:
        n_train = int(n * 0.8)
        rng = np.random.RandomState(cfg.data.random_seed)
        perm = rng.permutation(n)
        X_train = X[perm[:n_train]]
        X_test = X[perm[n_train:]]
        labels_train = labels[perm[:n_train]] if labels is not None else None
        labels_test = labels[perm[n_train:]] if labels is not None else None
        # Map training rows back to their original dataset row indices.
        train_indices = perm[:n_train].astype(np.int64)

    if X_test is not None:
        logging.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    else:
        logging.info(f"Train: {X_train.shape} (no test split)")

    # ── Auto-detect device ──
    device = cfg.train.device
    if device is None:
        if torch is not None and torch.cuda.is_available():
            device = 'cuda'
        elif torch is not None and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    logging.info(f"Device: {device}")
    eval_mode = "transductive_full_graph" if using_precomputed else "holdout_rows"
    run_manifest = collect_run_manifest(cfg, device, eval_mode)
    graph_cache = build_graph_cache_spec(cfg, X_train, eval_mode)
    if graph_cache is not None:
        write_manifest_if_missing(graph_cache)
        run_manifest["graph_cache"] = graph_cache.to_dict()
        logging.info("Graph cache key: %s", graph_cache.key)
        logging.info("Graph cache dir: %s", graph_cache.artifact_dir)
    else:
        run_manifest["graph_cache"] = None

    # ── Build model ──
    if torch is None:
        raise RuntimeError("torch is required to run experiments")
    from basemap.pumap.parametric_umap import ParametricUMAP

    mc = cfg.model
    tc = cfg.train
    pumap = ParametricUMAP(
        n_components=mc.n_components,
        hidden_dim=mc.hidden_dim,
        n_layers=mc.n_layers,
        n_neighbors=cfg.data.n_neighbors,
        a=mc.a,
        b=mc.b,
        correlation_weight=tc.correlation_weight,
        learning_rate=tc.learning_rate,
        n_epochs=tc.n_epochs,
        batch_size=tc.batch_size,
        device=device,
        use_batchnorm=mc.use_batchnorm,
        use_dropout=mc.use_dropout,
        clip_grad_norm=tc.clip_grad_norm,
        clip_grad_value=tc.clip_grad_value,
        pos_ratio=tc.pos_ratio,
        architecture=mc.architecture,
        correlation_distance_transform=tc.correlation_distance_transform,
        lr_schedule=tc.lr_schedule,
        warmup_steps=tc.warmup_steps,
        total_steps_estimate=tc.total_steps_estimate,
        use_amp=tc.use_amp,
        positive_target_mode=tc.positive_target_mode,
        reject_neighbors=tc.reject_neighbors,
        anchored_init=tc.anchored_init,
        anchored_init_epochs=tc.anchored_init_epochs,
        anchored_init_lr=tc.anchored_init_lr,
        anchored_init_path=tc.anchored_init_path,
        anchor_hold_weight=tc.anchor_hold_weight,
        anchor_hold_fraction=tc.anchor_hold_fraction,
        midnear_enabled=tc.midnear_enabled,
        mn_pairs_per_batch=tc.mn_pairs_per_batch,
        mn_weight_scale=tc.mn_weight_scale,
        weighted_edge_sampling=tc.weighted_edge_sampling,
        gpu_resident_data=tc.gpu_resident_data,
        gpu_resident_vram_budget_gb=tc.gpu_resident_vram_budget_gb,
    )

    # Count parameters
    pumap._init_model(X_train.shape[1])
    n_params = sum(p.numel() for p in pumap.model.parameters())
    logging.info(f"Model parameters: {n_params:,}")

    # ── Wandb setup ──
    wandb_run_name = cfg.logging.wandb_run_name or f"{cfg.name}_{cfg.config_hash()}"

    # ── Train ──
    t0 = time.time()
    pumap.fit(
        X_train,
        low_memory=tc.low_memory,
        verbose=tc.verbose,
        n_processes=tc.n_processes,
        random_state=cfg.data.random_seed,
        resample_negatives=tc.resample_negatives,
        precomputed_p_sym_path=cfg.data.precomputed_p_sym_path,
        precomputed_negatives_path=cfg.data.precomputed_negatives_path,
        precomputed_edges_path=cfg.data.precomputed_edges_path,
        cache_p_sym_path=graph_cache.p_sym_path if graph_cache is not None else None,
        cache_negatives_path=graph_cache.negatives_path if graph_cache is not None else None,
        use_wandb=cfg.logging.use_wandb,
        wandb_project=cfg.logging.wandb_project,
        wandb_run_name=wandb_run_name,
    )
    train_time = time.time() - t0
    if graph_cache is not None:
        run_manifest["graph_cache"] = graph_cache.to_dict()
    # Record the anchored-init RMS-radius scale factor (plan §4.3): needed to
    # map persisted coords back onto the teacher layout's original units.
    run_manifest["anchor_scale"] = getattr(pumap, "anchor_scale_", None)
    n_train = len(X_train)
    samples_per_sec = n_train * tc.n_epochs / train_time

    # ── Transform ──
    t0 = time.time()
    Z_train = pumap.transform(X_train)
    Z_test = pumap.transform(X_test) if X_test is not None else None
    transform_time = time.time() - t0

    # ── Evaluate ──
    metrics_train = compute_metrics(X_train, Z_train, cfg, labels_train)
    metrics_test = {}
    if X_test is not None:
        logging.info("Computing metrics on test set...")
        metrics_test = compute_metrics(X_test, Z_test, cfg, labels_test)

    # ── Standard UMAP baseline ──
    umap_baseline = {}
    if cfg.eval.compare_umap and X_test is not None:
        umap_baseline = run_umap_baseline(X_train, X_test, cfg)
    elif cfg.eval.compare_umap:
        logging.warning("Skipping UMAP baseline: no test split available (precomputed graph mode).")

    # ── Compile results ──
    eval_arr = Z_test if Z_test is not None else Z_train
    results = {
        "config": cfg.to_dict(),
        "config_hash": cfg.config_hash(),
        "timestamp": datetime.now().isoformat(),
        "data": {
            "n_samples": n,
            "n_features": int(X.shape[1]),
            "n_train": n_train,
            "n_test": len(X_test) if X_test is not None else 0,
        },
        "model": {
            "n_params": n_params,
        },
        "timing": {
            "data_load_s": data_load_time,
            "train_s": train_time,
            "transform_s": transform_time,
            "samples_per_sec": samples_per_sec,
        },
        "metrics_train": metrics_train,
        "metrics_test": metrics_test,
        "embedding_stats": {
            "mean": eval_arr.mean(axis=0).tolist(),
            "std": eval_arr.std(axis=0).tolist(),
            "min": float(eval_arr.min()),
            "max": float(eval_arr.max()),
        },
        "umap_baseline": umap_baseline,
        "run_manifest": run_manifest,
    }

    # ── Save ──
    with open(os.path.join(run_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(run_dir, "manifest.json"), 'w') as f:
        json.dump(run_manifest, f, indent=2)

    # ── Run persistence (defaults ON) ──
    # Every run writes its final 2D coordinates + model checkpoint so it can be
    # re-scored / inspected / stability-tested without retraining. Past runs
    # saved neither — a real loss (see plan §2 caveat).
    model_saved = False
    if cfg.logging.persist_run:
        coords_path = os.path.join(run_dir, "coords.parquet")
        _write_coords_parquet(coords_path, Z_train, train_indices)
        logging.info("Saved coords: %s (%d rows)", coords_path, len(Z_train))
        pumap.save(os.path.join(run_dir, "model.pt"))
        model_saved = True

        # Persist anchored-init targets (if used) so cross-seed stability
        # analysis can reuse the exact deterministic targets.
        anchor_targets = getattr(pumap, "anchor_targets_", None)
        if anchor_targets is not None:
            anchor_path = os.path.join(run_dir, "anchor_targets.parquet")
            _write_anchor_targets_parquet(anchor_path, anchor_targets, train_indices)
            logging.info("Saved anchor targets: %s (%d rows)", anchor_path,
                         len(anchor_targets))

    if cfg.logging.save_model and not model_saved:
        pumap.save(os.path.join(run_dir, "model.pt"))

    if cfg.logging.save_embeddings:
        np.save(os.path.join(run_dir, "Z_train.npy"), Z_train)
        if Z_test is not None:
            np.save(os.path.join(run_dir, "Z_test.npy"), Z_test)

    # ── Print summary ──
    logging.info(f"\n{'─'*60}")
    logging.info(f"  RESULTS: {cfg.name}")
    logging.info(f"{'─'*60}")
    logging.info(f"  Params:           {n_params:,}")
    logging.info(f"  Train time:       {train_time:.1f}s ({samples_per_sec:.0f} samples/sec)")
    logging.info(f"  Transform time:   {transform_time:.2f}s")
    report_metrics = metrics_test if metrics_test else metrics_train
    for k, v in report_metrics.items():
        logging.info(f"  {k}: {v:.4f}")
    if umap_baseline.get("metrics_test"):
        logging.info(f"  --- UMAP baseline ---")
        for k, v in umap_baseline["metrics_test"].items():
            logging.info(f"  umap_{k}: {v:.4f}")
    logging.info(f"  Results saved to: {run_dir}")

    return results


# ─── Sweep Runner ────────────────────────────────────────────────────────────

def run_sweep(cfg: ExperimentConfig, sweep_file: str) -> list:
    """Run a parameter sweep defined in a YAML file."""
    with open(sweep_file) as f:
        sweep_def = yaml.safe_load(f)

    sweep_params = sweep_def.get("sweep", {})
    configs = generate_sweep_configs(cfg, sweep_params)

    logging.info(f"Running sweep with {len(configs)} configurations")
    all_results = []
    for i, c in enumerate(configs):
        logging.info(f"\n{'='*60}")
        logging.info(f"  Sweep run {i+1}/{len(configs)}: {c.name}")
        logging.info(f"{'='*60}")
        results = run_single_experiment(c)
        all_results.append(results)

    # Print comparison table
    _print_sweep_summary(all_results)

    return all_results


def _print_sweep_summary(results_list):
    """Print a table comparing all sweep runs."""
    logging.info(f"\n{'='*80}")
    logging.info(f"  SWEEP SUMMARY")
    logging.info(f"{'='*80}")

    header = f"  {'Name':<40} {'Params':>8} {'Train(s)':>9} {'Dist Corr':>10} {'KNN':>8}"
    logging.info(header)
    logging.info(f"  {'─'*75}")
    for r in results_list:
        name = r['config']['name'][:38]
        n_params = r['model']['n_params']
        train_s = r['timing']['train_s']
        metrics = r['metrics_test'] or r['metrics_train']
        dc = metrics.get('distance_correlation', 0)
        knn = metrics.get('knn_preservation', 0)
        logging.info(f"  {name:<40} {n_params:>8,} {train_s:>9.1f} {dc:>10.4f} {knn:>8.4f}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run parametric UMAP experiments")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Override config values: train.batch_size=1024 model.hidden_dim=256")
    parser.add_argument("--sweep", type=str, default=None,
                        help="Path to sweep YAML file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config and exit without running")
    args = parser.parse_args()

    # Parse overrides
    overrides = {}
    for o in args.override:
        key, val = o.split("=", 1)
        # Try numeric coercion
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                if val.lower() == 'true':
                    val = True
                elif val.lower() == 'false':
                    val = False
        overrides[key] = val

    cfg = load_config(args.config, overrides)

    if args.dry_run:
        print(yaml.dump(cfg.to_dict(), default_flow_style=False, sort_keys=False))
        print(f"Config hash: {cfg.config_hash()}")
        print(f"Run dir would be: {cfg.run_dir()}")
        return

    if args.sweep:
        run_sweep(cfg, args.sweep)
    else:
        run_single_experiment(cfg)


if __name__ == "__main__":
    main()
