#!/usr/bin/env python3
"""Fresh-train, same-process GrEASE batch-stability reference for R0206."""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from typing import Any

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from numap import NUMAP

from basemap.round0206_grease_fresh import (
    BATCH_TOLERANCE,
    INFERENCE_CHUNK_ROWS,
    REFERENCE_SCHEMA,
)
from experiments.round0179_numap_reference import (
    _as_tensor,
    _checkpoint_step,
    _coordinate_summary,
    _model_config,
    _package_versions,
    _seed_everything,
    _spectral_updates,
)
from experiments.round0181_numap_reference import (
    _install_treatment,
    _normalization_arrays,
)


def _chunked(callable_, values: torch.Tensor, *, rows: int) -> np.ndarray:
    output = []
    for start in range(0, len(values), rows):
        output.append(
            np.asarray(callable_(values[start : start + rows].clone()), dtype=np.float32)
        )
    return np.concatenate(output, axis=0)


def _write_smoke_inputs(root: str) -> tuple[str, str]:
    rng = np.random.RandomState(42)
    train = rng.normal(size=(256, 16)).astype(np.float32)
    queries = rng.normal(size=(64, 16)).astype(np.float32)
    train /= np.linalg.norm(train, axis=1, keepdims=True)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    matrix = os.path.join(root, "smoke-train.npy")
    query = os.path.join(root, "smoke-queries.npy")
    np.save(matrix, train, allow_pickle=False)
    np.save(query, queries, allow_pickle=False)
    return matrix, query


def run(
    *, matrix: str, queries: str, rows: int, scale: str, output: str,
    smoke: bool = False,
) -> dict[str, Any]:
    output = os.path.abspath(output)
    if os.path.exists(output):
        if not os.path.isdir(output) or os.listdir(output):
            raise RuntimeError("R0206 reference output must be fresh and empty")
    else:
        os.makedirs(output)
    _seed_everything(42)
    _install_treatment()
    started = time.monotonic()
    X = _as_tensor(matrix, rows=rows)
    Q = _as_tensor(queries)
    if X.shape[1] != Q.shape[1]:
        raise RuntimeError("R0206 train/query dimensions differ")
    loaded = time.monotonic()
    os.chdir(output)
    config = _model_config(smoke=smoke)
    requested_config = json.loads(json.dumps(config))
    model = NUMAP(**config)
    model.fit(X)
    fit_finished = time.monotonic()
    grease_network = model.grease._spectralnet.spec_net
    grease_network.eval()
    model.pumap.model.eval()
    chunk_rows = 16 if smoke else INFERENCE_CHUNK_ROWS

    grease_full = np.asarray(model.grease.transform(Q.clone()), dtype=np.float32)
    grease_chunked = _chunked(model.grease.transform, Q, rows=chunk_rows)
    numap_full = np.asarray(model.transform(Q.clone()), dtype=np.float32)
    numap_chunked = _chunked(model.transform, Q, rows=chunk_rows)
    grease_error = float(np.max(np.abs(grease_full - grease_chunked)))
    numap_error = float(np.max(np.abs(numap_full - numap_chunked)))
    stable = (
        np.isfinite(grease_error)
        and np.isfinite(numap_error)
        and max(grease_error, numap_error) <= BATCH_TOLERANCE
    )
    batch_finished = time.monotonic()

    paths: dict[str, str] = {}
    for name, value in {
        "grease-full.npy": grease_full,
        "grease-chunked.npy": grease_chunked,
        "numap-full.npy": numap_full,
        "numap-chunked.npy": numap_chunked,
    }.items():
        np.save(os.path.join(output, name), value, allow_pickle=False)
        paths[name.removesuffix(".npy").replace("-", "_")] = name

    train_summary = None
    query_summary = None
    normalization, normalization_paths = _normalization_arrays(model, output)
    normalization["statistics_stored_in_checkpoint"] = False
    normalization["statistics_stored_in_fitted_object"] = True
    if stable:
        train_coordinates = np.asarray(
            model.transform(_as_tensor(matrix, rows=rows), is_train=True),
            dtype=np.float32,
        )
        query_coordinates = numap_full
        train_summary = _coordinate_summary(train_coordinates)
        query_summary = _coordinate_summary(query_coordinates)
        if min(train_summary["axis_standard_deviation"]) <= 1.0e-6:
            raise RuntimeError("R0206 train coordinates collapsed")
        if min(query_summary["axis_standard_deviation"]) <= 1.0e-6:
            raise RuntimeError("R0206 query coordinates collapsed")
        np.save(
            os.path.join(output, "train-coordinates.npy"),
            train_coordinates,
            allow_pickle=False,
        )
        np.save(
            os.path.join(output, "query-coordinates.npy"),
            query_coordinates,
            allow_pickle=False,
        )
        paths["train_coordinates"] = "train-coordinates.npy"
        paths["query_coordinates"] = "query-coordinates.npy"
    transforms_finished = time.monotonic()

    pumap_updates, lightning_checkpoints = _checkpoint_step(output)
    for index, path in enumerate(lightning_checkpoints):
        paths[f"lightning_checkpoint_{index}"] = os.path.relpath(path, output)
    spectral_updates = _spectral_updates(model)
    spectral_batches_per_epoch = int(
        np.ceil((0.9 * rows) / config["grease_batch_size"])
    )
    receipt = {
        "schema": REFERENCE_SCHEMA,
        "mode": "smoke" if smoke else "real",
        "scale": scale,
        "package_versions": _package_versions(),
        "config": requested_config,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "train_rows": int(rows),
        "query_rows": int(len(Q)),
        "dimension": int(X.shape[1]),
        "normalization": normalization,
        "batch_stability": {
            "comparison": "same fresh fitted object; one full query call versus concatenated fixed-size calls",
            "full_query_rows": int(len(Q)),
            "chunk_rows": chunk_rows,
            "grease_max_abs_error": grease_error,
            "numap_max_abs_error": numap_error,
            "tolerance": BATCH_TOLERANCE,
            "passed": bool(stable),
        },
        "train_coordinates": train_summary,
        "query_coordinates": query_summary,
        "checkpoint_restore_performed": False,
        "dill_or_pickle_object_written": False,
        "train_accounting": {
            "selected_pipeline": (
                "numap==0.2.3 GrEASE spectral extension + residual PUMAP with "
                "stored train-time normalization; same-process fresh-model inference"
            ),
            "sampler_class": "numap.umap_pytorch.data.UMAPDataset via PyTorch DataLoader",
            "positive_sampling_semantics": (
                "package fuzzy_simplicial_set; weighted edges expanded by "
                "int(200*weight), shuffled once, then first N edge slots per epoch"
            ),
            "negative_sampling_semantics": (
                f"package in-batch shuffle with negative_sample_rate={config['negative_sample_rate']}"
            ),
            "x_residency": (
                "owned host fp32 with stored train normalization; mini-batches and transforms on CUDA"
                if torch.cuda.is_available()
                else "owned host fp32; CPU smoke only"
            ),
            "grease_optimizer_updates": spectral_updates,
            "grease_architecture_actual": list(
                model.grease._spectralnet.spectral_hiddens
            ),
            "grease_batches_per_full_epoch": spectral_batches_per_epoch,
            "grease_completed_epoch_equivalents": spectral_updates
            / spectral_batches_per_epoch,
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
            "batch_stability_seconds": batch_finished - fit_finished,
            "stable_transform_seconds": transforms_finished - batch_finished,
            "total_seconds": time.monotonic() - started,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024**2,
        },
        "paths": {**paths, **normalization_paths},
    }
    with open(os.path.join(output, "execution.json"), "x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix")
    parser.add_argument("--queries")
    parser.add_argument("--rows", type=int)
    parser.add_argument("--scale", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        if os.path.exists(args.output):
            raise RuntimeError("R0206 smoke output must not exist")
        os.makedirs(args.output)
        matrix, queries = _write_smoke_inputs(args.output)
        receipt = run(
            matrix=matrix,
            queries=queries,
            rows=256,
            scale="smoke",
            output=os.path.join(args.output, "fit"),
            smoke=True,
        )
    else:
        if not args.matrix or not args.queries or args.rows is None:
            parser.error("real execution requires matrix, queries, and rows")
        receipt = run(**vars(args))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
