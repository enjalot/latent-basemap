#!/usr/bin/env python3
"""Run the bounded NUMAP/GrEASE baseline with stored training normalization."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import dill
import json
import os
import resource
import sys
import time
from typing import Any, Iterator

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import grease._reduction as grease_reduction
from grease._reduction import GrEASE as PackageGrEASE
from numap import NUMAP

from basemap.round0181_fixed_normalization import NORMALIZATION_POLICY
from experiments.round0179_numap_reference import (
    REAL_CONFIG,
    _as_tensor,
    _checkpoint_step,
    _coordinate_summary,
    _model_config,
    _package_versions,
    _seed_everything,
    _spectral_updates,
    _write_smoke_inputs,
)


class StoredTrainNormalizationGrEASE(PackageGrEASE):
    """GrEASE with pointwise-stable inference under frozen train statistics."""

    normalization_mean: torch.Tensor | None = None
    normalization_std: torch.Tensor | None = None
    normalization_nonzero_mask: torch.Tensor | None = None
    normalization_training_rows: int | None = None

    @contextmanager
    def _package_identity_normalizer(self) -> Iterator[None]:
        previous = grease_reduction.normalize_data
        grease_reduction.normalize_data = lambda value: value
        try:
            yield
        finally:
            grease_reduction.normalize_data = previous

    def _capture_statistics(self, X: torch.Tensor) -> None:
        values = X.to(device="cpu", dtype=torch.float32)
        self.normalization_mean = values.mean(dim=0).detach().clone()
        self.normalization_std = values.std(dim=0, correction=1).detach().clone()
        self.normalization_nonzero_mask = self.normalization_std != 0
        self.normalization_training_rows = int(values.shape[0])
        if not torch.isfinite(self.normalization_mean).all() or not torch.isfinite(
            self.normalization_std
        ).all():
            raise RuntimeError("nonfinite GrEASE training normalization statistics")

    def _normalize_copy(self, X: torch.Tensor) -> torch.Tensor:
        if (
            self.normalization_mean is None
            or self.normalization_std is None
            or self.normalization_nonzero_mask is None
        ):
            raise RuntimeError("GrEASE training normalization statistics are absent")
        values = X.to(device="cpu", dtype=torch.float32).clone()
        mask = self.normalization_nonzero_mask
        values[:, mask] = (
            values[:, mask] - self.normalization_mean[mask]
        ) / self.normalization_std[mask]
        return values

    def _normalize_inplace(self, X: torch.Tensor) -> torch.Tensor:
        if X.device.type != "cpu" or X.dtype != torch.float32:
            raise RuntimeError(
                "fixed-normalization treatment requires the registered host-fp32 tensor"
            )
        if (
            self.normalization_mean is None
            or self.normalization_std is None
            or self.normalization_nonzero_mask is None
        ):
            raise RuntimeError("GrEASE training normalization statistics are absent")
        mask = self.normalization_nonzero_mask
        X[:, mask] = (
            X[:, mask] - self.normalization_mean[mask]
        ) / self.normalization_std[mask]
        return X

    def fit(self, X: torch.Tensor, y: torch.Tensor | None = None):
        self._capture_statistics(X)
        normalized = self._normalize_copy(X)
        with self._package_identity_normalizer():
            return super().fit(normalized, y)

    def transform(self, X: torch.Tensor) -> np.ndarray:
        # Package GrEASE mutates the caller tensor while standardizing it; NUMAP
        # then feeds that same standardized tensor to PUMAP.  Preserve that
        # behavior so only the source of mean/std changes relative to R0179.
        normalized = self._normalize_inplace(X)
        with self._package_identity_normalizer():
            return super().transform(normalized)


def _install_treatment() -> None:
    import importlib

    module = importlib.import_module("numap.numap")
    module.GrEASE = StoredTrainNormalizationGrEASE


def _normalization_arrays(
    model: NUMAP, output: str
) -> tuple[dict[str, Any], dict[str, str]]:
    grease = model.grease
    if not isinstance(grease, StoredTrainNormalizationGrEASE):
        raise RuntimeError("fixed-normalization GrEASE treatment was not selected")
    mean = grease.normalization_mean.detach().cpu().numpy().astype(np.float32)
    std = grease.normalization_std.detach().cpu().numpy().astype(np.float32)
    mask = grease.normalization_nonzero_mask.detach().cpu().numpy().astype(np.bool_)
    paths = {
        "normalization_mean": "grease-train-mean.npy",
        "normalization_std": "grease-train-std.npy",
        "normalization_nonzero_mask": "grease-train-nonzero-mask.npy",
    }
    np.save(os.path.join(output, paths["normalization_mean"]), mean, allow_pickle=False)
    np.save(os.path.join(output, paths["normalization_std"]), std, allow_pickle=False)
    np.save(
        os.path.join(output, paths["normalization_nonzero_mask"]),
        mask,
        allow_pickle=False,
    )
    return {
        "policy": NORMALIZATION_POLICY,
        "statistics_stored_in_checkpoint": True,
        "training_rows": grease.normalization_training_rows,
        "features": int(len(mean)),
        "torch_std_correction": 1,
        "zero_std_features": int((~mask).sum()),
        "mean_minimum": float(mean.min()),
        "mean_maximum": float(mean.max()),
        "std_minimum_nonzero": float(std[mask].min()) if mask.any() else None,
        "std_maximum": float(std.max()),
    }, paths


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
    _install_treatment()
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

    os.chdir(output)
    config = _model_config(smoke=smoke)
    requested_config = json.loads(json.dumps(config))
    model = NUMAP(**config)
    model.fit(X)
    fit_finished = time.monotonic()
    # NUMAP fit deliberately leaves X standardized for its PUMAP stage, matching
    # R0179.  Reload fresh source bytes so projection sees one normalization.
    train_coordinates = np.asarray(
        model.transform(_as_tensor(matrix, rows=rows), is_train=True),
        dtype=np.float32,
    )
    train_transform_finished = time.monotonic()
    query_coordinates = np.asarray(
        model.transform(_as_tensor(queries)), dtype=np.float32
    )
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
    normalization, normalization_paths = _normalization_arrays(model, output)
    checkpoint_started = time.monotonic()
    with open(checkpoint_path, "wb") as handle:
        dill.dump(model, handle)
    checkpoint_finished = time.monotonic()
    spectral_updates = _spectral_updates(model)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with open(checkpoint_path, "rb") as handle:
        reloaded = dill.load(handle)
    full_reload = np.asarray(
        reloaded.transform(_as_tensor(queries)), dtype=np.float32
    )
    probe_rows = min(256, max(1, len(Q) // 2))
    batch_reload = np.asarray(
        reloaded.transform(_as_tensor(queries)[:probe_rows]), dtype=np.float32
    )
    reload_full_max_abs = float(np.max(np.abs(full_reload - query_coordinates)))
    reload_batch_max_abs = float(
        np.max(np.abs(batch_reload - query_coordinates[:probe_rows]))
    )
    normalization["batch_composition_probe_rows"] = probe_rows
    if (
        not np.isfinite(reload_full_max_abs)
        or not np.isfinite(reload_batch_max_abs)
        or reload_full_max_abs > 1.0e-4
        or reload_batch_max_abs > 1.0e-4
    ):
        raise RuntimeError(
            "fixed-normalization reload guard failed: "
            f"full={reload_full_max_abs}, batch={reload_batch_max_abs}"
        )
    pumap_updates, lightning_checkpoints = _checkpoint_step(output)
    spectral_batches_per_epoch = int(
        np.ceil((0.9 * rows) / config["grease_batch_size"])
    )
    if spectral_batches_per_epoch <= 0:
        raise RuntimeError("invalid GrEASE batch accounting")

    receipt = {
        "schema": "round0181-numap-fixed-normalization-execution-v1",
        "mode": "smoke" if smoke else "real",
        "package_versions": _package_versions(),
        "config": requested_config,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "train_rows": int(rows),
        "query_rows": int(len(Q)),
        "dimension": int(X.shape[1]),
        "input_norms": {
            "train_minimum": float(train_norms.min()),
            "train_maximum": float(train_norms.max()),
            "query_minimum": float(query_norms.min()),
            "query_maximum": float(query_norms.max()),
        },
        "normalization": normalization,
        "train_coordinates": train_summary,
        "query_coordinates": query_summary,
        "checkpoint": {
            "bytes": os.path.getsize(checkpoint_path),
            "reload_full_rows": int(len(Q)),
            "reload_full_max_abs_error": reload_full_max_abs,
            "reload_batch_rows": probe_rows,
            "reload_batch_max_abs_error": reload_batch_max_abs,
        },
        "train_accounting": {
            "selected_pipeline": (
                "numap==0.2.3 GrEASE spectral extension + residual PUMAP with "
                "stored train-time feature normalization"
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
                "owned host fp32; stored train normalization on CPU; mini-batches "
                "and PUMAP transforms execute on CUDA"
                if torch.cuda.is_available()
                else "owned host fp32; fixed-normalization smoke is CPU-only"
            ),
            "grease_optimizer_updates": spectral_updates,
            "grease_architecture_actual": list(
                reloaded.grease._spectralnet.spectral_hiddens
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
            "train_transform_seconds": train_transform_finished - fit_finished,
            "query_transform_seconds": query_transform_finished
            - train_transform_finished,
            "checkpoint_write_seconds": checkpoint_finished - checkpoint_started,
            "total_seconds": time.monotonic() - started,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024**2,
        },
        "paths": {
            "train_coordinates": os.path.basename(train_path),
            "query_coordinates": os.path.basename(query_path),
            "checkpoint": os.path.basename(checkpoint_path),
            **normalization_paths,
        },
    }
    receipt_path = os.path.join(output, "execution.json")
    with open(receipt_path, "x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return receipt


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
        receipt = run_fit(
            matrix=matrix,
            queries=queries,
            output=os.path.join(args.output, "fit"),
            rows=256,
            smoke=True,
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
