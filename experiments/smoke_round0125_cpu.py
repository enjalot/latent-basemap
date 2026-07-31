#!/usr/bin/env python3
"""CUDA-hidden train -> seal -> reload -> mini-panel preflight for R0125."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
from collections.abc import Mapping
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_build_new_file,
    atomic_save_new_npz,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.panel_v2 import PanelV2Config, score_panel
from basemap.round0125_runtime_bridge import (
    DEVICE_ARM,
    HOST_ARM,
    AuditedParametricUMAP,
    Round0125DeviceTrainingInput,
    Round0125HostTrainingInput,
    expected_device_endpoint_accounting,
    seal,
    select_outcome,
    validate_seal,
)


SMOKE_ROWS = 64
SMOKE_DIMENSION = 8
SMOKE_BATCH_SIZE = 8
SMOKE_POSITIVE_RATIO = 0.25
SMOKE_UPDATES = 9
SMOKE_SEED = 42


class _TinyHostDataset:
    """CPU endpoint provider with R0104's exact paired-slot interface."""

    def __init__(self, values: np.ndarray, *, buffer_rows: int) -> None:
        import torch

        self.values = np.asarray(values, dtype=np.float32)
        self.shape = self.values.shape
        self.device = "cpu"
        self.buffer_rows = int(buffer_rows)
        self.endpoint_gather_calls = 0
        self.source_rows_gathered = 0
        self.destination_rows_gathered = 0
        self.host_prefetch_batches_filled = 0
        self.host_prefetch_source_rows_filled = 0
        self.host_prefetch_destination_rows_filled = 0
        self._slots = [
            {
                "source": torch.empty((buffer_rows, self.shape[1])),
                "destination": torch.empty((buffer_rows, self.shape[1])),
            }
            for _ in range(2)
        ]

    def __len__(self) -> int:
        return len(self.values)

    def index_select(self, rows: Any):
        import torch

        if hasattr(rows, "detach"):
            rows = rows.detach().cpu().numpy()
        return torch.from_numpy(np.asarray(self.values[rows], dtype=np.float32))

    def fill_pair_slot(
        self, slot_index: int, source_rows: Any, destination_rows: Any
    ) -> int:
        left = np.asarray(source_rows, dtype=np.int64)
        right = np.asarray(destination_rows, dtype=np.int64)
        if (
            left.shape != right.shape
            or left.ndim != 1
            or len(left) > self.buffer_rows
            or np.any(left < 0)
            or np.any(left >= len(self))
            or np.any(right < 0)
            or np.any(right >= len(self))
        ):
            raise RuntimeError("invalid tiny host endpoint request")
        count = len(left)
        slot = self._slots[slot_index]
        slot["source"][:count].copy_(self.index_select(left))
        slot["destination"][:count].copy_(self.index_select(right))
        self.host_prefetch_batches_filled += 1
        self.host_prefetch_source_rows_filled += count
        self.host_prefetch_destination_rows_filled += count
        return count

    def transfer_pair_slot(self, slot_index: int, count: int):
        slot = self._slots[slot_index]
        self.endpoint_gather_calls += 1
        self.source_rows_gathered += count
        self.destination_rows_gathered += count
        return (
            slot["source"][:count].clone(),
            slot["destination"][:count].clone(),
        )

    def execution_stamp(self) -> dict[str, Any]:
        return {
            "source_representation": "fp16-control",
            "feature_residency": "host-mmap-fp16-source-shards",
            "device_conversion": "device-fp32-from-exact-fp16",
            "source_segments": ["synthetic-cpu-preflight"],
            "endpoint_gather_calls": self.endpoint_gather_calls,
            "source_rows_gathered": self.source_rows_gathered,
            "destination_rows_gathered": self.destination_rows_gathered,
            "host_prefetch_batches_filled": self.host_prefetch_batches_filled,
            "host_prefetch_source_rows_filled": (
                self.host_prefetch_source_rows_filled
            ),
            "host_prefetch_destination_rows_filled": (
                self.host_prefetch_destination_rows_filled
            ),
        }


def _model(required_pipeline: str) -> AuditedParametricUMAP:
    return AuditedParametricUMAP(
        n_components=2,
        hidden_dim=16,
        n_layers=1,
        n_neighbors=3,
        a=1.0,
        b=1.0,
        low_dim_kernel="legacy_lp",
        correlation_weight=0.0,
        learning_rate=0.01,
        n_epochs=2,
        batch_size=SMOKE_BATCH_SIZE,
        device="cpu",
        use_batchnorm=False,
        use_dropout=False,
        clip_grad_norm=1.0,
        clip_grad_value=None,
        pos_ratio=SMOKE_POSITIVE_RATIO,
        architecture="residual_bottleneck",
        correlation_distance_transform="raw",
        lr_schedule="cosine",
        warmup_steps=1,
        total_steps_estimate=SMOKE_UPDATES,
        require_full_budget=True,
        require_graph_manifest=True,
        required_input_pipeline=required_pipeline,
        use_amp=False,
        positive_target_mode="binary",
        reject_neighbors=False,
        anchored_init="none",
        anchor_hold_weight=0.0,
        midnear_enabled=False,
        mn_pairs_per_batch=0,
        weighted_edge_sampling=True,
        gpu_resident_data=required_pipeline == "device",
        gpu_resident_vram_budget_gb=0.0,
    )


def _metrics(panel: Mapping[str, Any]) -> dict[str, float]:
    return {
        "ffr": float(panel["ffr"]),
        "density": float(panel["density"]),
        "recall_at_10": float(panel["recall@k"]),
        "oos_proj_ffr": float(panel["ffr"]),
        "oos_proj_recall_at_10": float(panel["recall@k"]),
    }


def _source_files() -> list[dict[str, Any]]:
    root = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
    return [
        expected_input_signature(os.path.join(root, relative))
        for relative in (
            "basemap/round0104_training.py",
            "basemap/round0125_runtime_bridge.py",
            "experiments/round0125_nodes.py",
            "experiments/prepare_round0125_queue.py",
            "experiments/smoke_round0125_cpu.py",
        )
    ]


def _positive_updates(train_signature: Mapping[str, Any]) -> int:
    with open(train_signature["canonical_path"], encoding="utf-8") as handle:
        receipt = json.load(handle)
    validate_seal(receipt, label="R0125 CPU train receipt")
    return int(receipt["train_accounting"]["positive_lr_optimizer_steps"])


def run_smoke(
    *, release_sha: str, output_root: str, enforce_checkout: bool = False
) -> str:
    """Run both real adapters on CPU and return the sealed receipt path."""
    if re.fullmatch(r"[0-9a-f]{40}", release_sha) is None:
        raise ValueError("CPU preflight release SHA must be one full commit")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("R0125 CPU preflight requires CUDA_VISIBLE_DEVICES=''")
    if enforce_checkout:
        root = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
        observed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout
        if observed != release_sha or dirty:
            raise RuntimeError("CPU preflight requires the exact clean release checkout")

    import torch

    if torch.cuda.is_available():
        raise RuntimeError("CUDA-hidden CPU preflight unexpectedly discovered CUDA")
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    output = create_fresh_directory(output_root, label="R0125 CPU preflight")
    try:
        rng = np.random.default_rng(12_500)
        source = rng.normal(size=(SMOKE_ROWS, SMOKE_DIMENSION)).astype(np.float32)
        source /= np.linalg.norm(source, axis=1, keepdims=True)
        sources = np.arange(17, dtype=np.int32) % SMOKE_ROWS
        targets = (sources * 7 + 3) % SMOKE_ROWS
        weights = np.linspace(0.25, 1.0, len(sources), dtype=np.float32)
        graph_path = os.path.join(output, "tiny-graph.npz")
        atomic_save_new_npz(
            graph_path, immutable=True, sources=sources, targets=targets,
            weights=weights, n_nodes=np.asarray(SMOKE_ROWS),
            k=np.asarray(3),
        )
        manifest_path = os.path.join(output, "tiny-graph.manifest.json")
        atomic_write_new_json(
            manifest_path,
            {"schema": "round0125-cpu-preflight-graph-v1", "rows": SMOKE_ROWS},
            immutable=True,
        )
        graph = {
            "sources": sources,
            "targets": targets,
            "weights": weights,
            "n_nodes": SMOKE_ROWS,
            "signature": expected_input_signature(graph_path),
            "manifest_signature": expected_input_signature(manifest_path),
        }
        arms: dict[str, Any] = {}
        models: dict[str, AuditedParametricUMAP] = {}
        for arm in (DEVICE_ARM, HOST_ARM):
            if arm == DEVICE_ARM:
                training_input = Round0125DeviceTrainingInput(
                    source, graph, device="cpu", expected_rows=SMOKE_ROWS
                )
                required_pipeline = "device"
            else:
                training_input = Round0125HostTrainingInput(
                    _TinyHostDataset(source, buffer_rows=SMOKE_BATCH_SIZE),
                    graph,
                    expected_rows=SMOKE_ROWS,
                )
                required_pipeline = "host_weighted_jina_paired"
            random.seed(SMOKE_SEED)
            np.random.seed(SMOKE_SEED)
            torch.manual_seed(SMOKE_SEED)
            model = _model(required_pipeline)
            model._max_train_steps = SMOKE_UPDATES
            model.fit(
                training_input,
                low_memory=arm == HOST_ARM,
                verbose=False,
                n_processes=1,
                random_state=SMOKE_SEED,
                resample_negatives=False,
                precomputed_edges_path=graph_path,
                use_wandb=False,
            )
            if arm == HOST_ARM and training_input._last_sampler is not None:
                training_input._last_sampler.close()
            runtime = training_input.runtime_stamp()
            arm_root = create_fresh_directory(
                os.path.join(output, arm), label=f"R0125 CPU {arm}"
            )
            model_path = os.path.join(arm_root, "model.pt")
            atomic_build_new_file(model_path, model.save, immutable=True)
            train_receipt = seal({
                "schema": "round0125-cpu-preflight-train-v1",
                "arm": arm,
                "initial_model_state_sha256": model.initial_model_state_sha256,
                "train_accounting": model._train_stats,
                "exact_execution_receipt": runtime,
                "model": expected_input_signature(model_path),
            })
            train_path = os.path.join(arm_root, "train-receipt.json")
            atomic_write_new_json(train_path, train_receipt, immutable=True)
            with open(train_path, encoding="utf-8") as handle:
                validate_seal(json.load(handle), label=f"CPU {arm} train")
            loaded = AuditedParametricUMAP.load(model_path, device="cpu")
            coordinates = np.asarray(
                loaded.transform(source, batch_size=16), dtype=np.float32
            )
            panel = score_panel(
                source,
                coordinates,
                config=PanelV2Config(
                    frac=0.25,
                    k_hit=3,
                    k_density=3,
                    n_anchors=16,
                    corpus_chunk=32,
                    overselect=4,
                    block_elems=50_000,
                    rerank_byte_cap=4_000_000,
                    peak_byte_cap=8_000_000,
                ),
                provenance={"round_id": "0125", "mode": "cpu-preflight", "arm": arm},
            )
            panel_path = os.path.join(arm_root, "mini-panel.json")
            atomic_write_new_json(panel_path, seal(panel), immutable=True)
            arms[arm] = {
                "train": expected_input_signature(train_path),
                "model": expected_input_signature(model_path),
                "panel": expected_input_signature(panel_path),
                "initial_model_state_sha256": model.initial_model_state_sha256,
                "runtime": runtime,
                "metrics": _metrics(panel),
                "coordinates_finite": bool(np.isfinite(coordinates).all()),
                "coordinates_collapsed": bool(
                    np.any(coordinates.std(axis=0) <= 1e-8)
                ),
            }
            models[arm] = loaded

        edge_accounting = expected_device_endpoint_accounting(
            updates=SMOKE_UPDATES,
            graph_edges=len(sources),
            batch_size=SMOKE_BATCH_SIZE,
            positive_ratio=SMOKE_POSITIVE_RATIO,
        )
        device_runtime = arms[DEVICE_ARM]["runtime"]
        host_runtime = arms[HOST_ARM]["runtime"]
        density_delta = (
            arms[DEVICE_ARM]["metrics"]["density"]
            - arms[HOST_ARM]["metrics"]["density"]
        )
        selector = select_outcome(
            host_metrics=arms[HOST_ARM]["metrics"],
            device_metrics=arms[DEVICE_ARM]["metrics"],
            host_matched_density=arms[HOST_ARM]["metrics"]["density"],
            device_matched_density=arms[DEVICE_ARM]["metrics"]["density"],
            paired_delta_ci99=(density_delta - 0.01, density_delta + 0.01),
            execution_valid=True,
        )
        checks = {
            "cuda_hidden": not torch.cuda.is_available(),
            "both_exact_sampler_adapters_exercised": (
                device_runtime.get("sampler_class") == "DeviceEdgeSampler"
                and host_runtime.get("sampler_class")
                == "PairedHostWeightedJinaSampler"
            ),
            "identical_initial_model_state": (
                arms[DEVICE_ARM]["initial_model_state_sha256"]
                == arms[HOST_ARM]["initial_model_state_sha256"]
            ),
            "exact_successful_update_horizon": all(
                arms[arm]["runtime"]["stream_trace"]["batches_hashed"] == 8
                for arm in (DEVICE_ARM, HOST_ARM)
            ) and all(
                _positive_updates(arms[arm]["train"]) == SMOKE_UPDATES
                for arm in (DEVICE_ARM, HOST_ARM)
            ),
            "nondivisible_device_epoch_accounting": (
                edge_accounting["shortfall_per_completed_epoch"] == 1
                and edge_accounting["completed_epoch_boundaries"] == 1
                and edge_accounting["endpoint_rows_per_side"] == 71
                and device_runtime.get("source_rows_gathered") == 71
                and device_runtime.get("destination_rows_gathered") == 71
                and host_runtime.get("source_rows_gathered") == 72
                and host_runtime.get("destination_rows_gathered") == 72
            ),
            "checkpoint_reload_and_two_mini_panels": all(
                arms[arm]["coordinates_finite"]
                and not arms[arm]["coordinates_collapsed"]
                and np.isfinite(list(arms[arm]["metrics"].values())).all()
                for arm in (DEVICE_ARM, HOST_ARM)
            ),
            "selector_executed": bool(selector.get("outcome")),
            "sealed_train_and_panel_artifacts": all(
                all(
                    re.fullmatch(r"[0-9a-f]{64}", arms[arm][key]["sha256"])
                    is not None
                    for key in ("train", "model", "panel")
                )
                for arm in (DEVICE_ARM, HOST_ARM)
            ),
        }
        receipt = seal({
            "schema": "round0125-cpu-preflight-v1",
            "release_sha": release_sha,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "source_files": _source_files(),
            "graph": graph["signature"],
            "graph_manifest": graph["manifest_signature"],
            "arms": arms,
            "device_edge_accounting": edge_accounting,
            "selector": selector,
            "checks": checks,
            "outcome": "passed" if all(checks.values()) else "failed",
        })
        receipt_path = os.path.join(output, "cpu-preflight.json")
        atomic_write_new_json(receipt_path, receipt, immutable=True)
        if receipt["outcome"] != "passed":
            raise RuntimeError(f"R0125 CPU preflight failed: {checks}")
        return receipt_path
    finally:
        torch.set_num_threads(prior_threads)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    path = run_smoke(
        release_sha=args.release_sha,
        output_root=args.output,
        enforce_checkout=True,
    )
    print(json.dumps({"cpu_preflight_receipt": path}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
