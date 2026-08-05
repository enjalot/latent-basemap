#!/usr/bin/env python3
"""CPU-only GrEASE/NUMAP fixed-chunk inference diagnosis for R0196."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import dill
import io
import json
import os
import sys
import time
from typing import Any, Iterator

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from basemap.round0196_grease_batch_stable import (
    INFERENCE_CHUNK_ROWS,
    RELOAD_TOLERANCE,
    fixed_chunks,
)
from experiments.round0179_numap_reference import _as_tensor, _package_versions
from experiments.round0181_numap_reference import StoredTrainNormalizationGrEASE


@contextmanager
def _cpu_pickle_load() -> Iterator[None]:
    """Map tensors embedded by dill onto CPU without allocating CUDA."""
    original = torch.storage._load_from_bytes

    def load_from_bytes(value: bytes):
        return torch.load(
            io.BytesIO(value), map_location="cpu", weights_only=False
        )

    torch.storage._load_from_bytes = load_from_bytes
    try:
        yield
    finally:
        torch.storage._load_from_bytes = original


class BatchStableInferenceGrEASE(StoredTrainNormalizationGrEASE):
    """Minimal vendored inference patch: normalize once, then fixed network chunks."""

    inference_chunk_rows = INFERENCE_CHUNK_ROWS

    def _fixed_network_predict(self, X: torch.Tensor) -> np.ndarray:
        spectral = self._spectralnet
        if spectral.should_use_ae:
            raise RuntimeError("R0196 does not authorize the untested AE branch")
        network = spectral.spec_net
        network.eval()

        def apply(chunk: torch.Tensor) -> np.ndarray:
            with torch.no_grad():
                value = chunk.reshape(len(chunk), -1).to(spectral.device)
                return (
                    network(value, should_update_orth_weights=False)
                    .detach()
                    .cpu()
                    .numpy()
                )

        return fixed_chunks(X, apply, chunk_rows=self.inference_chunk_rows)

    def transform(self, X: torch.Tensor) -> np.ndarray:
        normalized = self._normalize_inplace(X)
        prediction = self._fixed_network_predict(normalized)
        if not self.should_true_eigenvectors:
            return prediction
        projected = prediction @ self.ortho_matrix
        if self.should_return_first_eigenvector:
            return projected
        scaled = projected @ np.diag((1 - self.eigenvalues) ** self.t)
        return scaled[:, 1:]


def _numap_transform(model: Any, X: torch.Tensor, *, chunk_pumap: bool) -> np.ndarray:
    spectral = np.asarray(model.grease.transform(X)[:, : model.se_dim], dtype=np.float32)
    joined = torch.cat([torch.from_numpy(spectral), X], dim=1)
    if chunk_pumap:
        return fixed_chunks(joined, model.pumap.transform)
    return np.asarray(model.pumap.transform(joined), dtype=np.float32)


def _errors(
    *,
    grease_full: np.ndarray,
    grease_probe: np.ndarray,
    numap_full: np.ndarray,
    numap_probe: np.ndarray,
    probe_rows: int,
) -> dict[str, float]:
    return {
        "grease_batch_max_abs_error": float(
            np.max(np.abs(grease_full[:probe_rows] - grease_probe))
        ),
        "numap_batch_max_abs_error": float(
            np.max(np.abs(numap_full[:probe_rows] - numap_probe))
        ),
    }


def run(*, checkpoint: str, queries: str, output: str) -> dict[str, Any]:
    if torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES") not in {
        "",
        "-1",
    }:
        raise RuntimeError("R0196 reference execution must hide CUDA")
    started = time.monotonic()
    Q = _as_tensor(queries)
    probe_rows = INFERENCE_CHUNK_ROWS
    if len(Q) < 2 * probe_rows:
        raise RuntimeError("R0196 query set is too small for the batch probe")
    with _cpu_pickle_load(), open(checkpoint, "rb") as handle:
        model = dill.load(handle)
    model.grease._spectralnet.device = torch.device("cpu")
    model.grease._spectralnet.spec_net.to("cpu").eval()
    model.pumap.model.to("cpu").eval()

    baseline_grease_full = np.asarray(
        model.grease.transform(Q.clone()), dtype=np.float32
    )
    baseline_grease_probe = np.asarray(
        model.grease.transform(Q[:probe_rows].clone()), dtype=np.float32
    )
    baseline_numap_full = np.asarray(model.transform(Q.clone()), dtype=np.float32)
    baseline_numap_probe = np.asarray(
        model.transform(Q[:probe_rows].clone()), dtype=np.float32
    )
    baseline = _errors(
        grease_full=baseline_grease_full,
        grease_probe=baseline_grease_probe,
        numap_full=baseline_numap_full,
        numap_probe=baseline_numap_probe,
        probe_rows=probe_rows,
    )

    model.grease.__class__ = BatchStableInferenceGrEASE
    fixed_grease_full = np.asarray(
        model.grease.transform(Q.clone()), dtype=np.float32
    )
    fixed_grease_probe = np.asarray(
        model.grease.transform(Q[:probe_rows].clone()), dtype=np.float32
    )
    grease_only_numap_full = _numap_transform(
        model, Q.clone(), chunk_pumap=False
    )
    grease_only_numap_probe = _numap_transform(
        model, Q[:probe_rows].clone(), chunk_pumap=False
    )
    fixed_grease = _errors(
        grease_full=fixed_grease_full,
        grease_probe=fixed_grease_probe,
        numap_full=grease_only_numap_full,
        numap_probe=grease_only_numap_probe,
        probe_rows=probe_rows,
    )
    fixed_both_numap_full = _numap_transform(model, Q.clone(), chunk_pumap=True)
    fixed_both_numap_probe = _numap_transform(
        model, Q[:probe_rows].clone(), chunk_pumap=True
    )
    fixed_both = _errors(
        grease_full=fixed_grease_full,
        grease_probe=fixed_grease_probe,
        numap_full=fixed_both_numap_full,
        numap_probe=fixed_both_numap_probe,
        probe_rows=probe_rows,
    )
    if all(
        fixed_grease[key] <= RELOAD_TOLERANCE
        for key in ("grease_batch_max_abs_error", "numap_batch_max_abs_error")
    ):
        selected = "fixed-256-row-grease-network"
    elif all(
        fixed_both[key] <= RELOAD_TOLERANCE
        for key in ("grease_batch_max_abs_error", "numap_batch_max_abs_error")
    ):
        selected = "fixed-256-row-grease-and-pumap-networks"
    else:
        selected = None
    receipt = {
        "schema": "round0196-grease-batch-stable-cpu-execution-v1",
        "device": "cpu",
        "source_checkpoint_round": "0181",
        "query_rows": int(len(Q)),
        "dimension": int(Q.shape[1]),
        "probe_rows": probe_rows,
        "reload_tolerance": RELOAD_TOLERANCE,
        "package_versions": _package_versions(),
        "candidates": {
            "baseline": baseline,
            "fixed_grease": fixed_grease,
            "fixed_grease_and_pumap": fixed_both,
        },
        "selected_patch": selected,
        "implementation": {
            "normalization": "stored R0181 train mean/std over full call",
            "grease_network_chunk_rows": INFERENCE_CHUNK_ROWS,
            "grease_orthonormalization_weights_updated_at_inference": False,
            "pumap_chunk_rows_when_selected": INFERENCE_CHUNK_ROWS,
            "batchnorm_or_dropout_layers_added": False,
            "vendored_patch": True,
        },
        "wall_seconds": time.monotonic() - started,
    }
    os.makedirs(output, exist_ok=False)
    with open(os.path.join(output, "execution.json"), "x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(**vars(args)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
