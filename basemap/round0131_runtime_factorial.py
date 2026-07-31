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
    DEVICE_ARM,
    DIMENSION,
    GRAPH_EDGES,
    GRAPH_K,
    HOST_ARM,
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
    expected_device_endpoint_accounting,
)


ROUND_ID = "0131"
CAPABILITY = "jina-fineweb-2m-runtime-component-localization-v1"
R0125_RELEASE_SHA = "ff5dfcde5632257aac355008a70bc330bab26bee"
RESIDENT_FUSED = "numpy_device_fused"
RESIDENT_SEPARATE = "numpy_device_separate"
ARMS = (RESIDENT_FUSED, RESIDENT_SEPARATE)
PATH_ARMS = (HOST_ARM, RESIDENT_FUSED, RESIDENT_SEPARATE, DEVICE_ARM)
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

_GRAPH_INVARIANT_KEYS = (
    "rows",
    "dimension",
    "source_payload_sha256",
    "graph",
    "graph_manifest",
    "graph_edges",
    "graph_k",
    "input_preprocessing",
)
_DOSE_ACCOUNTING = {
    "lr_horizon": SUCCESSFUL_UPDATES,
    "positive_lr_optimizer_steps": SUCCESSFUL_UPDATES,
    "scheduler_steps": SUCCESSFUL_UPDATES,
    "attempted_batches": SUCCESSFUL_UPDATES,
    "finite_loss_batches": SUCCESSFUL_UPDATES,
    "optimizer_steps_attempted": SUCCESSFUL_UPDATES,
    "optimizer_steps_succeeded": SUCCESSFUL_UPDATES,
    "amp_overflow_skips": 0,
    "nonfinite_loss_skips": 0,
    "nonfinite_gradient_skips": 0,
    "stop_reason": "lr_horizon",
    "budget_satisfied": True,
    "n_pos_edges": GRAPH_EDGES,
}
_COMMON_RUNTIME_SEMANTICS = {
    "positive_sampling": "weighted_with_replacement",
    "negative_sampling": "uniform-2m-row-universe-nonself",
    "graph_degree": "variable-fuzzy-k50-edge-universe",
    "weighted_requested": True,
    "weighted_effective": True,
    "uniform_with_replacement": False,
    "positive_with_replacement": True,
    "multiplicity_policy": "row_multiplicity_uncapped",
    "valid_canonical_edge_count": GRAPH_EDGES,
    "source_representation": "fp16-control",
}
_REGISTERED_PATH = {
    HOST_ARM: {
        "sampler_class": "PairedHostWeightedJinaSampler",
        "feature_residency": "host-mmap-fp16-source-shards",
        "device_conversion": "device-fp32-from-exact-fp16",
        "endpoint_forward": "fused-source-destination",
    },
    RESIDENT_FUSED: {
        "sampler_class": SAMPLER_CLASS,
        "feature_residency": "device-fp16",
        "device_conversion": "resident-storage-to-device-fp32-on-gather",
        "endpoint_forward": "fused-source-destination",
    },
    RESIDENT_SEPARATE: {
        "sampler_class": SAMPLER_CLASS,
        "feature_residency": "device-fp16",
        "device_conversion": "resident-storage-to-device-fp32-on-gather",
        "endpoint_forward": "separate-source-destination",
    },
    DEVICE_ARM: {
        "sampler_class": "DeviceEdgeSampler",
        "feature_residency": "device-fp16",
        "device_conversion": "resident-storage-to-device-fp32-on-gather",
        "endpoint_forward": "separate-source-destination",
    },
}


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


def _mapping_subset_matches(
    observed: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    return all(observed.get(key) == value for key, value in expected.items())


def _normalized_graph_invariant(config: Mapping[str, Any]) -> dict[str, Any] | None:
    invariant = config.get("causal_invariant")
    if not isinstance(invariant, Mapping):
        return None
    if any(key not in invariant for key in _GRAPH_INVARIANT_KEYS):
        return None
    return {key: invariant[key] for key in _GRAPH_INVARIANT_KEYS}


def _normalized_dose(
    config: Mapping[str, Any], train: Mapping[str, Any]
) -> dict[str, Any] | None:
    invariant = config.get("causal_invariant")
    optimizer = config.get("optimizer")
    accounting = train.get("train_accounting")
    if not all(isinstance(value, Mapping) for value in (invariant, optimizer, accounting)):
        return None
    registered = {
        "seed": invariant.get("seed"),
        "batch_size": invariant.get("batch_size"),
        "positive_ratio": invariant.get("positive_ratio"),
        "successful_positive_lr_updates": invariant.get(
            "successful_positive_lr_updates"
        ),
        "optimizer_batch_size": optimizer.get("batch_size"),
        "optimizer_positive_ratio": optimizer.get("positive_ratio"),
        "optimizer_successful_positive_lr_updates": optimizer.get(
            "successful_positive_lr_updates"
        ),
        "n_epochs": optimizer.get("n_epochs"),
    }
    observed = {key: accounting.get(key) for key in _DOSE_ACCOUNTING}
    return {"registered": registered, "observed": observed}


def _stream_digest(train: Mapping[str, Any]) -> dict[str, Any] | None:
    runtime = train.get("exact_execution_receipt")
    trace = runtime.get("stream_trace") if isinstance(runtime, Mapping) else None
    if not isinstance(trace, Mapping):
        return None
    value = {
        "batches_hashed": trace.get("batches_hashed"),
        "source_endpoint_ids_sha256": trace.get("source_endpoint_ids_sha256"),
        "destination_endpoint_ids_sha256": trace.get(
            "destination_endpoint_ids_sha256"
        ),
    }
    if (
        value["batches_hashed"] != STREAM_TRACE_BATCHES
        or any(
            not isinstance(value[key], str)
            or len(value[key]) != 64
            or any(character not in "0123456789abcdef" for character in value[key])
            for key in (
                "source_endpoint_ids_sha256",
                "destination_endpoint_ids_sha256",
            )
        )
    ):
        return None
    return value


def causal_execution_checks(
    *,
    active_environment_sha256: str,
    trains: Mapping[str, Mapping[str, Any]],
    configs: Mapping[str, Mapping[str, Any]],
) -> dict[str, bool]:
    """Verify the four cells really form the registered H->R->F->D path.

    The endpoint cells come from R0125 and the interior cells from R0131, so
    round-local config hashes are deliberately not compared.  Instead, this
    function compares the causal quantities after normalizing away schema and
    arm labels, while checking each arm's observed runtime against its own
    fully registered pipeline stamp.
    """

    exact_cells = set(trains) == set(PATH_ARMS) == set(configs)
    selected_trains = [trains.get(arm, {}) for arm in PATH_ARMS]
    selected_configs = [configs.get(arm, {}) for arm in PATH_ARMS]

    initial_hashes = [train.get("initial_model_state_sha256") for train in selected_trains]
    initial_hash_valid = all(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
        for value in initial_hashes
    )
    environments = [train.get("environment_freeze_sha256") for train in selected_trains]
    graph_invariants = [
        _normalized_graph_invariant(config) for config in selected_configs
    ]
    model_invariants = [config.get("model") for config in selected_configs]
    optimizer_invariants = [config.get("optimizer") for config in selected_configs]
    dose_invariants = [
        _normalized_dose(config, train)
        for config, train in zip(selected_configs, selected_trains, strict=True)
    ]
    causal_hashes_match_configs = all(
        isinstance(config.get("causal_invariant"), Mapping)
        and train.get("causal_invariant_sha256")
        == sha256_bytes(canonical_json(config["causal_invariant"]))
        for config, train in zip(selected_configs, selected_trains, strict=True)
    )
    receipt_graphs_match_configs = all(
        isinstance(config.get("causal_invariant"), Mapping)
        and train.get("graph") == config["causal_invariant"].get("graph")
        and train.get("graph_manifest")
        == config["causal_invariant"].get("graph_manifest")
        for config, train in zip(selected_configs, selected_trains, strict=True)
    )

    runtimes = {
        arm: train.get("exact_execution_receipt")
        for arm, train in zip(PATH_ARMS, selected_trains, strict=True)
    }
    pipeline_stamps_match_configs = all(
        isinstance(runtimes[arm], Mapping)
        and isinstance(configs.get(arm, {}).get("execution"), Mapping)
        and isinstance(
            configs[arm]["execution"].get("expected_pipeline_stamp"), Mapping
        )
        and _mapping_subset_matches(
            runtimes[arm], configs[arm]["execution"]["expected_pipeline_stamp"]
        )
        for arm in PATH_ARMS
    )
    common_runtime_semantics = all(
        isinstance(runtimes[arm], Mapping)
        and _mapping_subset_matches(runtimes[arm], _COMMON_RUNTIME_SEMANTICS)
        for arm in PATH_ARMS
    )
    registered_path_shape = all(
        isinstance(runtimes[arm], Mapping)
        and _mapping_subset_matches(runtimes[arm], _REGISTERED_PATH[arm])
        for arm in PATH_ARMS
    )
    runtime_graphs_match_configs = all(
        isinstance(runtimes[arm], Mapping)
        and isinstance(configs.get(arm, {}).get("causal_invariant"), Mapping)
        and runtimes[arm].get("graph")
        == configs[arm]["causal_invariant"].get("graph")
        and runtimes[arm].get("graph_manifest")
        == configs[arm]["causal_invariant"].get("graph_manifest")
        for arm in PATH_ARMS
    )
    numpy_streams = [_stream_digest(trains.get(arm, {})) for arm in PATH_ARMS[:3]]

    expected_full_rows = SUCCESSFUL_UPDATES * BATCH_SIZE
    device_endpoint_rows = expected_device_endpoint_accounting()[
        "endpoint_rows_per_side"
    ]
    endpoint_rows_match_path = all(
        isinstance(runtimes[arm], Mapping)
        and runtimes[arm].get("source_rows_gathered")
        == (expected_full_rows if arm != DEVICE_ARM else device_endpoint_rows)
        and runtimes[arm].get("destination_rows_gathered")
        == (expected_full_rows if arm != DEVICE_ARM else device_endpoint_rows)
        for arm in PATH_ARMS
    )

    return {
        "all_four_cells_present": exact_cells,
        "all_four_update0_model_hashes_equal": (
            exact_cells and initial_hash_valid and len(set(initial_hashes)) == 1
        ),
        "cross_round_environment_equal": (
            exact_cells
            and isinstance(active_environment_sha256, str)
            and len(active_environment_sha256) == 64
            and set(environments) == {active_environment_sha256}
        ),
        "normalized_graph_invariant_equal": (
            exact_cells
            and all(value is not None for value in graph_invariants)
            and all(value == graph_invariants[0] for value in graph_invariants[1:])
        ),
        "normalized_model_invariant_equal": (
            exact_cells
            and isinstance(model_invariants[0], Mapping)
            and all(value == model_invariants[0] for value in model_invariants[1:])
        ),
        "normalized_optimizer_invariant_equal": (
            exact_cells
            and isinstance(optimizer_invariants[0], Mapping)
            and all(
                value == optimizer_invariants[0]
                for value in optimizer_invariants[1:]
            )
        ),
        "normalized_registered_and_observed_dose_equal": (
            exact_cells
            and all(value is not None for value in dose_invariants)
            and all(value == dose_invariants[0] for value in dose_invariants[1:])
            and dose_invariants[0]["observed"] == _DOSE_ACCOUNTING
        ),
        "causal_invariant_hashes_match_configs": causal_hashes_match_configs,
        "train_receipt_graphs_match_configs": receipt_graphs_match_configs,
        "observed_pipeline_stamps_match_configs": pipeline_stamps_match_configs,
        "common_sampler_distribution_semantics_equal": common_runtime_semantics,
        "registered_h_r_f_d_path_shape": registered_path_shape,
        "runtime_graphs_match_configs": runtime_graphs_match_configs,
        "h_r_f_first8_numpy_endpoint_streams_equal": (
            all(value is not None for value in numpy_streams)
            and numpy_streams[0] == numpy_streams[1] == numpy_streams[2]
        ),
        "endpoint_rows_match_registered_path": endpoint_rows_match_path,
    }


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
