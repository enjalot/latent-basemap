"""Conditional decomposition of R0125's host/device runtime bundle.

The two new arms fill the interior of a three-transition path while reusing
R0125's fresh corner arms:

    host numpy + host fp16 + fused forward
      -> host numpy + device fp16 + fused forward
      -> host numpy + device fp16 + separate forwards
      -> torch device sampler + device fp16 + separate forwards

Adjacent comparisons therefore change residency, endpoint-forward mode, and
sampler/RNG/epoch batching in that order.  The round is meaningful only after
an accepted positive R0125 bundle result.
"""
from __future__ import annotations

import math
import os
from collections.abc import Mapping
from typing import Any

import numpy as np

from .artifact_identity import canonical_json, sha256_bytes
from .round0104_training import InventoryFp16Array, preprocessing_stamp
from .round0125_runtime_bridge import (
    BATCH_SIZE,
    DIMENSION,
    GRAPH_EDGES,
    GRAPH_K,
    MATCHED_DENSITY_FLOOR,
    N_EPOCHS,
    POSITIVE_RATIO,
    ROWS,
    SEED,
    STREAM_TRACE_BATCHES,
    SUCCESSFUL_UPDATES,
    TRAIN_MINIMUM_UPDATES_PER_S,
    TRAIN_WARNING_UPDATES_PER_S,
    AuditedParametricUMAP,
    TracingDeviceArrayDataset,
    TracingPairedHostWeightedJinaSampler,
)


ROUND_ID = "0131"
CAPABILITY = "jina-fineweb-2m-runtime-component-localization-v1"
RESIDENT_FUSED = "numpy_device_fused"
RESIDENT_SEPARATE = "numpy_device_separate"
ARMS = (RESIDENT_FUSED, RESIDENT_SEPARATE)
PIPELINES = {
    RESIDENT_FUSED: "numpy_weighted_device_fp16_fused",
    RESIDENT_SEPARATE: "numpy_weighted_device_fp16_separate",
}
SAMPLER_CLASS = "Round0131NumpyDeviceSampler"
POSITIVE_R0125_OUTCOMES = {
    "device-path-restores-density-without-native-regression-at-seed42",
    "device-path-restores-density-but-regresses-native-panel-at-seed42",
}
PAIRED_BOOTSTRAP_DRAWS = 1_000
PAIRED_BOOTSTRAP_SEED = 12_501
PERFORMANCE_WARMUP_UPDATES = 200
PERFORMANCE_WINDOWS = math.ceil(
    (SUCCESSFUL_UPDATES - PERFORMANCE_WARMUP_UPDATES) / 2_500
)


class Round0131Error(RuntimeError):
    """The frozen R0131 component-localization contract changed."""


class DeviceResidentPairDataset:
    """Device-fp16 features with NumPy-produced pair ids and two id slots.

    The producer thread fills only CPU id arrays.  The training thread turns
    those ids into CUDA indices and gathers from one resident fp16 tensor.
    This retains the R0104 NumPy row generator and prefetch schedule without
    performing CUDA operations from the producer thread.
    """

    def __init__(
        self,
        source: Any,
        *,
        device: str,
        buffer_rows: int,
        expected_rows: int = ROWS,
    ) -> None:
        if source.shape != (expected_rows, DIMENSION) or buffer_rows <= 0:
            raise Round0131Error("invalid device-resident pair source")
        self.device = str(device)
        self.shape = source.shape
        self.buffer_rows = int(buffer_rows)
        self._resident = TracingDeviceArrayDataset(
            source, self.device, trace_batches=STREAM_TRACE_BATCHES
        )
        self._slots: list[tuple[np.ndarray, np.ndarray] | None] = [None, None]
        self.endpoint_gather_calls = 0
        self.source_rows_gathered = 0
        self.destination_rows_gathered = 0
        self.host_prefetch_batches_filled = 0
        self.host_prefetch_source_rows_filled = 0
        self.host_prefetch_destination_rows_filled = 0

    def __len__(self) -> int:
        return self.shape[0]

    def to(self, _device: str) -> "DeviceResidentPairDataset":
        return self

    def fill_pair_slot(
        self, slot_index: int, source_rows: Any, destination_rows: Any
    ) -> int:
        left = np.asarray(source_rows, dtype=np.int64)
        right = np.asarray(destination_rows, dtype=np.int64)
        if (
            not 0 <= slot_index < len(self._slots)
            or left.ndim != 1
            or right.shape != left.shape
            or len(left) > self.buffer_rows
            or np.any(left < 0)
            or np.any(left >= len(self))
            or np.any(right < 0)
            or np.any(right >= len(self))
        ):
            raise Round0131Error("device-resident endpoint ids are invalid")
        self._slots[slot_index] = (left.copy(), right.copy())
        count = len(left)
        self.host_prefetch_batches_filled += 1
        self.host_prefetch_source_rows_filled += count
        self.host_prefetch_destination_rows_filled += count
        return count

    def transfer_pair_slot(self, slot_index: int, count: int):
        import torch

        if not 0 <= slot_index < len(self._slots):
            raise Round0131Error("device-resident endpoint slot is invalid")
        pair = self._slots[slot_index]
        if pair is None or count != len(pair[0]):
            raise Round0131Error("device-resident endpoint slot is empty")
        left = torch.as_tensor(pair[0], dtype=torch.long, device=self.device)
        right = torch.as_tensor(pair[1], dtype=torch.long, device=self.device)
        source = self._resident.index_select(left)
        destination = self._resident.index_select(right)
        self.endpoint_gather_calls += 1
        self.source_rows_gathered += count
        self.destination_rows_gathered += count
        return source, destination

    def index_select(self, rows: Any):
        import torch

        if not torch.is_tensor(rows):
            rows = torch.as_tensor(rows, dtype=torch.long, device=self.device)
        return self._resident.index_select(rows.to(self.device, dtype=torch.long))

    def execution_stamp(self) -> dict[str, Any]:
        return {
            "source_representation": "fp16-control",
            "feature_residency": "device-fp16",
            "device_conversion": "resident-storage-to-device-fp32-on-gather",
            "endpoint_gather_calls": self.endpoint_gather_calls,
            "source_rows_gathered": self.source_rows_gathered,
            "destination_rows_gathered": self.destination_rows_gathered,
            "host_prefetch_batches_filled": self.host_prefetch_batches_filled,
            "host_prefetch_source_rows_filled": self.host_prefetch_source_rows_filled,
            "host_prefetch_destination_rows_filled": (
                self.host_prefetch_destination_rows_filled
            ),
            "stream_trace": self._resident.trace_receipt(),
        }


class Round0131NumpyDeviceSampler(TracingPairedHostWeightedJinaSampler):
    """R0104 NumPy weighted row law over device-resident fp16 features."""

    def __init__(self, *args: Any, component_arm: str, **kwargs: Any) -> None:
        if component_arm not in ARMS:
            raise Round0131Error(f"unknown component arm {component_arm!r}")
        self.component_arm = component_arm
        self.fused_endpoint_forward = component_arm == RESIDENT_FUSED
        super().__init__(*args, **kwargs)

    def execution_stamp(self) -> dict[str, Any]:
        stamp = super().execution_stamp()
        stamp.update({
            "schema": "round0131-numpy-device-pipeline-v1",
            "pipeline": PIPELINES[self.component_arm],
            "sampler_class": SAMPLER_CLASS,
            "component_arm": self.component_arm,
            "positive_destination_policy": "R0104-fp16-fuzzy-k50",
            "endpoint_forward": (
                "fused-source-destination"
                if self.fused_endpoint_forward
                else "separate-source-destination"
            ),
            "feature_residency": "device-fp16",
            "device_conversion": "resident-storage-to-device-fp32-on-gather",
        })
        return stamp


class Round0131TrainingInput:
    """Adapter exposing the two registered NumPy/device intermediate arms."""

    def __init__(
        self,
        source: Any,
        graph: Mapping[str, Any],
        *,
        arm: str,
        device: str,
        expected_rows: int = ROWS,
    ) -> None:
        if arm not in ARMS or source.shape != (expected_rows, DIMENSION):
            raise Round0131Error("intermediate training input changed")
        self.source = source
        self.graph = dict(graph)
        self.arm = arm
        self.device = str(device)
        self.shape = source.shape
        self._dataset: DeviceResidentPairDataset | None = None
        self._sampler: Round0131NumpyDeviceSampler | None = None

    def __len__(self) -> int:
        return self.shape[0]

    def to(self, _device: str) -> "Round0131TrainingInput":
        return self

    def index_select(self, rows: Any):
        if self._dataset is None:
            raise Round0131Error("intermediate dataset was not constructed")
        return self._dataset.index_select(rows)

    def prepare_round0034_training(
        self,
        *,
        edges_path: str,
        batch_size: int,
        pos_ratio: float,
        random_state: int,
        positive_target_mode: str,
        weighted_edge_sampling: bool,
        reject_neighbors: bool,
        required_input_pipeline: str | None,
    ):
        if (
            os.path.realpath(edges_path)
            != os.path.realpath(self.graph["signature"]["canonical_path"])
            or positive_target_mode != "binary"
            or not weighted_edge_sampling
            or reject_neighbors
            or required_input_pipeline != PIPELINES[self.arm]
        ):
            raise Round0131Error("intermediate pipeline request changed")
        dataset = DeviceResidentPairDataset(
            self.source,
            device=self.device,
            buffer_rows=batch_size,
            expected_rows=len(self),
        )
        sampler = Round0131NumpyDeviceSampler(
            dataset,
            sources=self.graph["sources"],
            targets=self.graph["targets"],
            weights=self.graph["weights"],
            n_nodes=self.graph["n_nodes"],
            batch_size=batch_size,
            pos_ratio=pos_ratio,
            random_state=random_state,
            graph_signature=self.graph["signature"],
            graph_manifest_signature=self.graph["manifest_signature"],
            arm="fp16_control",
            component_arm=self.arm,
            expected_rows=len(self),
        )
        self._dataset = dataset
        self._sampler = sampler
        return dataset, sampler, sampler.n_pos, sampler.execution_stamp(), {
            "graph": self.graph["signature"],
            "graph_manifest": self.graph["manifest_signature"],
            "source_representation": "fp16-control",
        }

    def runtime_stamp(self) -> dict[str, Any]:
        if self._sampler is None:
            raise Round0131Error("intermediate sampler was not constructed")
        return self._sampler.execution_stamp()


def _expected_pipeline(arm: str, graph_edges: int) -> dict[str, Any]:
    if arm not in ARMS:
        raise Round0131Error(f"unknown component arm {arm!r}")
    return {
        "schema": "round0131-numpy-device-pipeline-v1",
        "pipeline": PIPELINES[arm],
        "sampler_class": SAMPLER_CLASS,
        "component_arm": arm,
        "positive_sampling": "weighted_with_replacement",
        "positive_destination_policy": "R0104-fp16-fuzzy-k50",
        "negative_sampling": "uniform-2m-row-universe-nonself",
        "graph_degree": "variable-fuzzy-k50-edge-universe",
        "host_prefetch": "single-producer-two-pinned-slot",
        "endpoint_forward": (
            "fused-source-destination"
            if arm == RESIDENT_FUSED
            else "separate-source-destination"
        ),
        "weighted_requested": True,
        "weighted_effective": True,
        "uniform_with_replacement": False,
        "positive_with_replacement": True,
        "multiplicity_policy": "row_multiplicity_uncapped",
        "valid_canonical_edge_count": int(graph_edges),
        "source_representation": "fp16-control",
        "feature_residency": "device-fp16",
        "device_conversion": "resident-storage-to-device-fp32-on-gather",
    }


def train_config(
    arm: str,
    *,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    graph_edges: int = GRAPH_EDGES,
) -> tuple[dict[str, Any], str]:
    invariant = {
        "rows": ROWS,
        "dimension": DIMENSION,
        "source_payload_sha256": (
            "f4a0050e81a3755de84ba73405ba6823fa387f09a15d3ad299083fa60093f069"
        ),
        "seed": SEED,
        "graph": dict(graph_signature),
        "graph_manifest": dict(graph_manifest_signature),
        "graph_edges": int(graph_edges),
        "graph_k": GRAPH_K,
        "successful_positive_lr_updates": SUCCESSFUL_UPDATES,
        "batch_size": BATCH_SIZE,
        "positive_ratio": POSITIVE_RATIO,
        "input_preprocessing": preprocessing_stamp("fp16_control"),
        "numpy_row_generator": "numpy.default_rng-searchsorted-offset-v1",
    }
    config = {
        "schema": "round0131-runtime-component-config-v1",
        "arm": arm,
        "causal_invariant": invariant,
        "causal_invariant_sha256": sha256_bytes(canonical_json(invariant)),
        "model": {
            "architecture": "residual_bottleneck",
            "input_dimension": DIMENSION,
            "hidden_dimension": 2048,
            "hidden_layers": 3,
            "output_dimension": 2,
            "use_batchnorm": False,
            "use_dropout": False,
            "low_dim_kernel": "legacy_lp",
            "a": 1.0,
            "b": 1.0,
        },
        "optimizer": {
            "seed": SEED,
            "learning_rate": 0.001,
            "batch_size": BATCH_SIZE,
            "positive_ratio": POSITIVE_RATIO,
            "positive_target_mode": "binary",
            "weighted_edge_sampling": True,
            "correlation_weight": 0.0,
            "clip_grad_norm": 1.0,
            "use_amp": "bf16",
            "schedule": "cosine-v3-positive-budget",
            "warmup_successful_updates": PERFORMANCE_WARMUP_UPDATES,
            "successful_positive_lr_updates": SUCCESSFUL_UPDATES,
            "reject_neighbors": False,
            "n_epochs": N_EPOCHS,
        },
        "execution": {
            "required_pipeline": PIPELINES[arm],
            "gpu_resident_data": True,
            "gpu_resident_vram_budget_gb": 31.0,
            "minimum_train_upd_s": TRAIN_MINIMUM_UPDATES_PER_S,
            "warning_train_upd_s": TRAIN_WARNING_UPDATES_PER_S,
            "performance_subfloor_patience": 2,
            "performance_windows": PERFORMANCE_WINDOWS,
            "stream_trace_batches": STREAM_TRACE_BATCHES,
            "expected_pipeline_stamp": _expected_pipeline(arm, graph_edges),
        },
    }
    return config, sha256_bytes(canonical_json(config))


def new_model(config: Mapping[str, Any], *, device: str = "cuda"):
    model = config["model"]
    optimizer = config["optimizer"]
    execution = config["execution"]
    invariant = config["causal_invariant"]
    return AuditedParametricUMAP(
        n_components=model["output_dimension"],
        hidden_dim=model["hidden_dimension"],
        n_layers=model["hidden_layers"],
        n_neighbors=invariant["graph_k"],
        a=model["a"],
        b=model["b"],
        low_dim_kernel=model["low_dim_kernel"],
        correlation_weight=optimizer["correlation_weight"],
        learning_rate=optimizer["learning_rate"],
        n_epochs=optimizer["n_epochs"],
        batch_size=optimizer["batch_size"],
        device=device,
        use_batchnorm=model["use_batchnorm"],
        use_dropout=model["use_dropout"],
        clip_grad_norm=optimizer["clip_grad_norm"],
        clip_grad_value=None,
        pos_ratio=optimizer["positive_ratio"],
        architecture=model["architecture"],
        correlation_distance_transform="raw",
        lr_schedule="cosine",
        warmup_steps=optimizer["warmup_successful_updates"],
        total_steps_estimate=optimizer["successful_positive_lr_updates"],
        require_full_budget=True,
        require_graph_manifest=True,
        required_input_pipeline=execution["required_pipeline"],
        use_amp=optimizer["use_amp"],
        positive_target_mode=optimizer["positive_target_mode"],
        reject_neighbors=optimizer["reject_neighbors"],
        anchored_init="none",
        anchor_hold_weight=0.0,
        midnear_enabled=False,
        mn_pairs_per_batch=0,
        weighted_edge_sampling=optimizer["weighted_edge_sampling"],
        gpu_resident_data=True,
        gpu_resident_vram_budget_gb=execution["gpu_resident_vram_budget_gb"],
        graph_manifest_path=invariant["graph_manifest"]["canonical_path"],
        graph_manifest_sha256=invariant["graph_manifest"]["sha256"],
    )


def classify_interval(interval: tuple[float, float]) -> str:
    low, high = map(float, interval)
    if not np.isfinite([low, high]).all() or low > high:
        raise Round0131Error("invalid adjacent bootstrap interval")
    if low > 0.0:
        return "reliably-positive"
    if high <= 0.0:
        return "nonpositive"
    return "unresolved"


def select_outcome(
    *,
    r0125_outcome: str,
    correlations: Mapping[str, float],
    adjacent_ci99: Mapping[str, tuple[float, float]],
    execution_valid: bool,
) -> dict[str, Any]:
    expected_cells = {
        "host_control", RESIDENT_FUSED, RESIDENT_SEPARATE, "device_treatment"
    }
    if set(correlations) != expected_cells or set(adjacent_ci99) != {
        "residency",
        "endpoint_forward",
        "sampler_rng_epoch",
    }:
        raise Round0131Error("component selector cells changed")
    if r0125_outcome not in POSITIVE_R0125_OUTCOMES:
        raise Round0131Error("R0125 did not release the positive branch")
    statuses = {
        key: classify_interval(tuple(value))
        for key, value in adjacent_ci99.items()
    }
    clears = {
        key: float(value) >= MATCHED_DENSITY_FLOOR
        for key, value in correlations.items()
    }
    if not execution_valid:
        outcome = "invalid-execution"
    elif clears[RESIDENT_FUSED] and statuses["residency"] == "reliably-positive":
        outcome = "residency-transition-is-first-resolved-restoration"
    elif (
        not clears[RESIDENT_FUSED]
        and clears[RESIDENT_SEPARATE]
        and statuses["endpoint_forward"] == "reliably-positive"
    ):
        outcome = "endpoint-forward-transition-is-first-resolved-restoration"
    elif (
        not clears[RESIDENT_SEPARATE]
        and clears["device_treatment"]
        and statuses["sampler_rng_epoch"] == "reliably-positive"
    ):
        outcome = "sampler-rng-transition-is-first-resolved-restoration"
    else:
        outcome = "runtime-component-localization-inconclusive"
    return {
        "schema": "round0131-runtime-component-selector-v1",
        "outcome": outcome,
        "r0125_positive_outcome": r0125_outcome,
        "correlations": {key: float(value) for key, value in correlations.items()},
        "clears_registered_floor": clears,
        "adjacent_ci99": {
            key: list(map(float, value)) for key, value in adjacent_ci99.items()
        },
        "adjacent_classification": statuses,
        "execution_valid": bool(execution_valid),
        "native_intermediate_quality_tested": False,
        "production_runtime_adopted": False,
    }

