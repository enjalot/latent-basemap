#!/usr/bin/env python3
"""CUDA-hidden train-to-seal-to-reload smoke for conditional R0131."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from basemap.artifact_identity import canonical_json, expected_input_signature, sha256_bytes
from basemap.output_safety import atomic_write_new_json
from basemap.round0125_runtime_bridge import AuditedParametricUMAP
from basemap.round0131_runtime_factorial import (
    ARMS,
    PIPELINES,
    RESIDENT_FUSED,
    RESIDENT_SEPARATE,
    Round0131TrainingInput,
    select_outcome,
)


def _seal(body):
    value = dict(body)
    return {**value, "identity_sha256": sha256_bytes(canonical_json(value))}


def _tiny_model(*, pipeline: str, fused: bool) -> AuditedParametricUMAP:
    model = AuditedParametricUMAP(
        n_components=2,
        hidden_dim=32,
        n_layers=1,
        n_neighbors=4,
        a=1.0,
        b=1.0,
        low_dim_kernel="legacy_lp",
        correlation_weight=0.0,
        learning_rate=0.001,
        n_epochs=1,
        batch_size=16,
        device="cpu",
        use_batchnorm=False,
        use_dropout=False,
        clip_grad_norm=1.0,
        pos_ratio=0.25,
        architecture="residual_bottleneck",
        lr_schedule="cosine",
        warmup_steps=1,
        total_steps_estimate=8,
        require_full_budget=True,
        require_graph_manifest=False,
        required_input_pipeline=pipeline,
        use_amp=False,
        positive_target_mode="binary",
        reject_neighbors=False,
        weighted_edge_sampling=True,
        gpu_resident_data=True,
        gpu_resident_vram_budget_gb=1.0,
    )
    model._max_train_steps = 8
    model._bench_warmup = 0
    return model


def run_smoke(*, release_sha: str, output_path: str) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("R0131 smoke requires CUDA_VISIBLE_DEVICES=''")
    started = time.monotonic()
    rng = np.random.default_rng(13_131)
    rows = 23
    source = rng.normal(size=(rows, 768)).astype(np.float32)
    sources = np.repeat(np.arange(rows, dtype=np.int32), 2)
    targets = np.concatenate(
        ((np.arange(rows) + 1) % rows, (np.arange(rows) + 3) % rows)
    ).astype(np.int32)
    # Interleave source order to keep all ids represented while retaining a
    # non-divisible edge count relative to four positives per tiny batch.
    sources = np.concatenate((np.arange(rows), np.arange(rows))).astype(np.int32)
    weights = rng.uniform(0.1, 1.0, size=len(sources)).astype(np.float32)
    with tempfile.TemporaryDirectory(prefix="r0131-smoke-") as temp:
        graph_path = os.path.join(temp, "edges.npz")
        np.savez(
            graph_path,
            sources=sources,
            targets=targets,
            weights=weights,
            n_nodes=np.asarray(rows),
            k=np.asarray(2),
        )
        signature = expected_input_signature(graph_path)
        graph = {
            "sources": sources,
            "targets": targets,
            "weights": weights,
            "n_nodes": rows,
            "signature": signature,
            "manifest_signature": {
                "canonical_path": graph_path,
                "kind": "file",
                "bytes": signature["bytes"],
                "sha256": signature["sha256"],
            },
        }
        receipts = {}
        traces = {}
        for arm in ARMS:
            training = Round0131TrainingInput(
                source, graph, arm=arm, device="cpu", expected_rows=rows
            )
            model = _tiny_model(
                pipeline=PIPELINES[arm], fused=arm == RESIDENT_FUSED
            )
            model.fit(
                training,
                low_memory=False,
                verbose=False,
                n_processes=1,
                random_state=42,
                precomputed_edges_path=graph_path,
                use_wandb=False,
            )
            if training._sampler is not None:
                training._sampler.close()
            runtime = training.runtime_stamp()
            model_path = os.path.join(temp, f"{arm}.pt")
            model.save(model_path)
            loaded = AuditedParametricUMAP.load(model_path, device="cpu")
            coordinates = loaded.transform(source, batch_size=8)
            if coordinates.shape != (rows, 2) or not np.isfinite(coordinates).all():
                raise RuntimeError("R0131 smoke checkpoint/panel output is invalid")
            receipts[arm] = {
                "model": expected_input_signature(model_path),
                "positive_lr_updates": model._train_stats["positive_lr_optimizer_steps"],
                "budget_satisfied": model._train_stats["budget_satisfied"],
                "endpoint_forward": runtime["endpoint_forward"],
                "pipeline": runtime["pipeline"],
                "coordinates_sha256": sha256_bytes(
                    np.ascontiguousarray(coordinates).tobytes()
                ),
            }
            traces[arm] = runtime["stream_trace"]
        selector = select_outcome(
            r0125_outcome=(
                "device-path-restores-density-without-native-regression-at-seed42"
            ),
            correlations={
                "host_control": 0.15,
                RESIDENT_FUSED: 0.18,
                RESIDENT_SEPARATE: 0.19,
                "device_treatment": 0.20,
            },
            adjacent_ci99={
                "residency": (0.01, 0.05),
                "endpoint_forward": (-0.01, 0.03),
                "sampler_rng_epoch": (-0.02, 0.04),
            },
            execution_valid=True,
        )
        checks = {
            "both_actual_adapters_train_eight_successful_updates": all(
                receipt["positive_lr_updates"] == 8
                and receipt["budget_satisfied"] is True
                for receipt in receipts.values()
            ),
            "both_bounded_traces_are_complete": all(
                trace["batches_hashed"] == trace["requested_batches"] == 8
                for trace in traces.values()
            ),
            "checkpoint_reload_and_tiny_panel": all(
                len(receipt["coordinates_sha256"]) == 64
                for receipt in receipts.values()
            ),
            "identical_bounded_numpy_stream": (
                traces[RESIDENT_FUSED] == traces[RESIDENT_SEPARATE]
            ),
            "forward_modes_distinct": (
                receipts[RESIDENT_FUSED]["endpoint_forward"]
                == "fused-source-destination"
                and receipts[RESIDENT_SEPARATE]["endpoint_forward"]
                == "separate-source-destination"
            ),
            "selector_reached": (
                selector["outcome"]
                == "residency-transition-is-first-resolved-restoration"
            ),
            "nondivisible_edge_fixture": len(sources) % 4 != 0,
        }
    source_paths = [
        Path(__file__),
        Path(__file__).parents[1] / "basemap" / "round0131_runtime_factorial.py",
        Path(__file__).parents[1] / "experiments" / "round0131_nodes.py",
        Path(__file__).parents[1] / "experiments" / "prepare_round0131_queue.py",
    ]
    receipt = _seal({
        "schema": "round0131-cpu-preflight-v1",
        "round_id": "0131",
        "release_sha": release_sha,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "outcome": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "arms": receipts,
        "selector": selector,
        "source_files": [expected_input_signature(str(path)) for path in source_paths],
        "wall_seconds": time.monotonic() - started,
        "scientific_evidence": False,
    })
    atomic_write_new_json(output_path, receipt, immutable=True)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    receipt = run_smoke(release_sha=args.release_sha, output_path=args.output)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
