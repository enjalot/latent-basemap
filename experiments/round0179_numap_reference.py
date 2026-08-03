#!/usr/bin/env python3
"""Run the unmodified NUMAP 0.2.3 GrEASE/residual reference path.

This file is intentionally executed by the private R0179 NUMAP toolchain,
not by the release checkout's Python environment.  It contains only the thin
adapter needed to load the frozen matrix, save coordinates/checkpoint bytes,
and stamp the package's actual execution path.
"""
from __future__ import annotations

import argparse
import glob
import importlib.metadata
import json
import os
import random
import re
import resource
import sys
import time
from typing import Any

import dill
import numpy as np
import torch

from numap import NUMAP


REAL_CONFIG: dict[str, Any] = {
    "n_neighbors": 10,
    "min_dist": 0.1,
    "metric": "cosine",
    "n_components": 2,
    "se_dim": 5,
    "se_neighbors": 10,
    "random_state": 42,
    "lr": 1.0e-3,
    "epochs": 10,
    "batch_size": 64,
    "num_workers": 0,
    "num_gpus": 1,
    "use_se": True,
    "use_residual_connections": True,
    "use_grease": True,
    "grease_batch_size": 1024,
    "grease_lr": 1.0e-3,
    "learn_from_se": True,
    "negative_sample_rate": 5,
    "use_concat": False,
    "use_alpha": False,
    "alpha": 0.0,
    "init_method": "identity",
    "grease_hiddens": [128, 256, 256],
    "use_true_eigenvectors": True,
}


def _seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")


def _package_versions() -> dict[str, str]:
    names = (
        "numap",
        "grease-embeddings",
        "torch",
        "pytorch-lightning",
        "pynndescent",
        "umap-learn",
        "numpy",
        "scikit-learn",
    )
    return {name: importlib.metadata.version(name) for name in names}


def _as_tensor(path: str, *, rows: int | None = None) -> torch.Tensor:
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if value.ndim != 2 or value.shape[1] <= 0:
        raise RuntimeError(f"malformed matrix at {path}: {value.shape}")
    if rows is not None and value.shape[0] != rows:
        raise RuntimeError(
            f"row count changed at {path}: {value.shape[0]} != {rows}"
        )
    # NUMAP 0.2.3 expects a torch.Tensor and retains the training tensor in
    # its fitted GrEASE object.  Own the bytes so a read-only mmap is never
    # exposed to package code.
    owned = np.array(value, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(owned).all():
        raise RuntimeError(f"nonfinite matrix at {path}")
    return torch.from_numpy(owned)


def _checkpoint_step(output: str) -> tuple[int, list[str]]:
    paths = sorted(
        glob.glob(os.path.join(output, "lightning_logs", "version_*", "checkpoints", "*.ckpt"))
    )
    steps: list[int] = []
    for path in paths:
        match = re.search(r"step=(\d+)", os.path.basename(path))
        if match:
            steps.append(int(match.group(1)))
    if not steps:
        raise RuntimeError("NUMAP Lightning checkpoint lacks a step stamp")
    return max(steps), paths


def _spectral_updates(model: NUMAP) -> int:
    trainer = model.grease._spectralnet.spectral_trainer
    observed: list[int] = []
    for state in trainer.optimizer.state.values():
        step = state.get("step")
        if step is None:
            continue
        observed.append(int(step.item() if hasattr(step, "item") else step))
    if not observed or min(observed) != max(observed) or observed[0] <= 0:
        raise RuntimeError("GrEASE optimizer update accounting is unavailable")
    return observed[0]


def _coordinate_summary(value: np.ndarray) -> dict[str, Any]:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all():
        raise RuntimeError("NUMAP coordinates are malformed")
    centered = value - value.mean(axis=0, keepdims=True)
    covariance = np.cov(value, rowvar=False)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "axis_standard_deviation": [float(item) for item in value.std(axis=0)],
        "radial_quantiles": [
            float(item)
            for item in np.quantile(
                np.linalg.norm(centered, axis=1), [0.0, 0.01, 0.5, 0.99, 1.0]
            )
        ],
        "covariance": np.asarray(covariance).tolist(),
    }


def _model_config(*, smoke: bool) -> dict[str, Any]:
    config = dict(REAL_CONFIG)
    config["grease_hiddens"] = list(REAL_CONFIG["grease_hiddens"])
    if smoke:
        # Same package path and control flags; one PUMAP epoch and a smaller
        # GrEASE network keep the pre-issuance CPU closure below two minutes.
        config.update({
            "epochs": 1,
            "num_gpus": 0,
            "grease_batch_size": 64,
            "grease_hiddens": [32],
            "se_dim": 2,
        })
    return config


def run_fit(
    *, matrix: str, queries: str, output: str, rows: int, smoke: bool
) -> dict[str, Any]:
    matrix = os.path.abspath(matrix)
    queries = os.path.abspath(queries)
    output = os.path.abspath(output)
    if os.path.exists(output):
        if not os.path.isdir(output) or os.listdir(output):
            raise RuntimeError(f"output must be a fresh empty directory: {output}")
    else:
        os.makedirs(output)
    _seed_everything(42)
    started = time.monotonic()
    X = _as_tensor(matrix, rows=rows)
    Q = _as_tensor(queries)
    if X.shape[1] != Q.shape[1]:
        raise RuntimeError("train/query feature dimensions differ")
    train_norms = torch.linalg.vector_norm(X, dim=1)
    query_norms = torch.linalg.vector_norm(Q, dim=1)
    if torch.any(train_norms <= 0) or torch.any(query_norms <= 0):
        raise RuntimeError("NUMAP inputs contain zero-norm rows")
    loaded = time.monotonic()

    # PyTorch Lightning otherwise writes its checkpoint relative to the
    # caller's checkout.  Keep every package side effect inside the declared
    # node output and make the checkpoint-step receipt discoverable.
    os.chdir(output)
    config = _model_config(smoke=smoke)
    requested_config = json.loads(json.dumps(config))
    model = NUMAP(**config)
    model.fit(X)
    fit_finished = time.monotonic()
    train_coordinates = np.asarray(model.transform(X, is_train=True), dtype=np.float32)
    train_transform_finished = time.monotonic()
    query_coordinates = np.asarray(model.transform(Q), dtype=np.float32)
    query_transform_finished = time.monotonic()
    train_summary = _coordinate_summary(train_coordinates)
    query_summary = _coordinate_summary(query_coordinates)
    if min(train_summary["axis_standard_deviation"]) <= 1.0e-6:
        raise RuntimeError("NUMAP train coordinates collapsed")
    if min(query_summary["axis_standard_deviation"]) <= 1.0e-6:
        raise RuntimeError("NUMAP query coordinates collapsed")

    train_path = os.path.join(output, "numap-train-coordinates.npy")
    query_path = os.path.join(output, "numap-query-coordinates.npy")
    checkpoint_path = os.path.join(output, "numap-model.dill")
    np.save(train_path, train_coordinates, allow_pickle=False)
    np.save(query_path, query_coordinates, allow_pickle=False)
    checkpoint_started = time.monotonic()
    with open(checkpoint_path, "wb") as handle:
        dill.dump(model, handle)
    checkpoint_finished = time.monotonic()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with open(checkpoint_path, "rb") as handle:
        reloaded = dill.load(handle)
    probe_rows = min(256, len(Q))
    reloaded_coordinates = np.asarray(
        reloaded.transform(Q[:probe_rows]), dtype=np.float32
    )
    reload_max_abs = float(
        np.max(np.abs(reloaded_coordinates - query_coordinates[:probe_rows]))
    )
    if not np.isfinite(reload_max_abs) or reload_max_abs > 1.0e-4:
        raise RuntimeError(
            f"reloaded NUMAP checkpoint drifted: max abs {reload_max_abs}"
        )
    pumap_updates, lightning_checkpoints = _checkpoint_step(output)
    spectral_updates = _spectral_updates(reloaded)
    spectral_batches_per_epoch = int(np.ceil((0.9 * rows) / config["grease_batch_size"]))
    if spectral_batches_per_epoch <= 0:
        raise RuntimeError("invalid GrEASE batch accounting")

    receipt = {
        "schema": "round0179-numap-reference-execution-v1",
        "mode": "smoke" if smoke else "real",
        "package_versions": _package_versions(),
        "config": requested_config,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "train_rows": int(rows),
        "query_rows": int(len(Q)),
        "dimension": int(X.shape[1]),
        "input_norms": {
            "train_minimum": float(train_norms.min()),
            "train_maximum": float(train_norms.max()),
            "query_minimum": float(query_norms.min()),
            "query_maximum": float(query_norms.max()),
        },
        "train_coordinates": train_summary,
        "query_coordinates": query_summary,
        "checkpoint": {
            "bytes": os.path.getsize(checkpoint_path),
            "reload_probe_rows": probe_rows,
            "reload_max_abs_error": reload_max_abs,
        },
        "train_accounting": {
            "selected_pipeline": (
                "numap==0.2.3 official-example GrEASE spectral extension + "
                "residual PUMAP encoder"
            ),
            "sampler_class": "numap.umap_pytorch.data.UMAPDataset via PyTorch DataLoader",
            "positive_sampling_semantics": (
                "package fuzzy_simplicial_set; weighted edges expanded by "
                "int(200*weight), shuffled once, then the first N edge slots "
                "sampled once per PUMAP epoch because UMAPDataset.__len__ is N"
            ),
            "negative_sampling_semantics": (
                f"package in-batch shuffle with negative_sample_rate={config['negative_sample_rate']}"
            ),
            "x_residency": (
                "owned host fp32 training tensor retained by GrEASE; mini-batches "
                "move to CUDA; PUMAP transforms execute on CUDA"
                if torch.cuda.is_available()
                else "owned host fp32 tensor; all smoke computation CPU-only"
            ),
            "grease_optimizer_updates": spectral_updates,
            "grease_architecture_actual": list(
                reloaded.grease._spectralnet.spectral_hiddens
            ),
            "grease_batches_per_full_epoch": spectral_batches_per_epoch,
            "grease_completed_epoch_equivalents": (
                spectral_updates / spectral_batches_per_epoch
            ),
            "grease_max_epochs": 200,
            "pumap_optimizer_updates": pumap_updates,
            "pumap_expected_updates": int(np.ceil(rows / config["batch_size"]))
            * int(config["epochs"]),
            "pumap_epochs": int(config["epochs"]),
            "lightning_checkpoints": [
                os.path.relpath(path, output) for path in lightning_checkpoints
            ],
        },
        "performance": {
            "load_seconds": loaded - started,
            "fit_seconds": fit_finished - loaded,
            "train_transform_seconds": train_transform_finished - fit_finished,
            "query_transform_seconds": query_transform_finished - train_transform_finished,
            "checkpoint_write_seconds": checkpoint_finished - checkpoint_started,
            "total_seconds": time.monotonic() - started,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024**2,
        },
        "paths": {
            "train_coordinates": os.path.basename(train_path),
            "query_coordinates": os.path.basename(query_path),
            "checkpoint": os.path.basename(checkpoint_path),
        },
    }
    receipt_path = os.path.join(output, "execution.json")
    with open(receipt_path, "x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return receipt


def _write_smoke_inputs(root: str) -> tuple[str, str]:
    rng = np.random.RandomState(42)
    train = rng.normal(size=(256, 16)).astype(np.float32)
    query = rng.normal(size=(32, 16)).astype(np.float32)
    train /= np.linalg.norm(train, axis=1, keepdims=True)
    query /= np.linalg.norm(query, axis=1, keepdims=True)
    matrix = os.path.join(root, "smoke-train.npy")
    queries = os.path.join(root, "smoke-query.npy")
    np.save(matrix, train, allow_pickle=False)
    np.save(queries, query, allow_pickle=False)
    return matrix, queries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix")
    parser.add_argument("--queries")
    parser.add_argument("--rows", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        if os.path.exists(args.output):
            raise RuntimeError("smoke output already exists")
        os.makedirs(args.output)
        matrix, queries = _write_smoke_inputs(args.output)
        fit_output = os.path.join(args.output, "fit")
        receipt = run_fit(
            matrix=matrix, queries=queries, output=fit_output, rows=256, smoke=True
        )
    else:
        if not args.matrix or not args.queries or args.rows is None:
            parser.error("real execution requires --matrix, --queries, and --rows")
        receipt = run_fit(
            matrix=args.matrix,
            queries=args.queries,
            output=args.output,
            rows=args.rows,
            smoke=False,
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
