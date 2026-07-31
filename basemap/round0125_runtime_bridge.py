"""Causal device-versus-host runtime bridge over the exact R0104 tuple.

R0125 deliberately keeps the R0104 fp16 rows, fuzzy graph, model, seed,
optimizer, and successful-update horizon fixed.  Its only treatment is the
existing execution-path bundle: the legacy device-resident ``DeviceEdgeSampler``
versus R0104's host-prefetched ``PairedHostWeightedJinaSampler``.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import math
import os
import platform
import sys
from collections.abc import Mapping
from typing import Any

import numpy as np

from .artifact_identity import canonical_json, sha256_bytes
from .pumap.parametric_umap import ParametricUMAP
from .pumap.parametric_umap.datasets.edge_list_dataset import (
    DeviceArrayDataset,
    DeviceEdgeSampler,
)
from .round0104_training import (
    DIMENSION,
    GRAPH_K,
    InventoryFp16Array,
    PairedHostWeightedJinaSampler,
    QUERY_ROWS,
    QUERY_START,
    ROWS,
    SEED,
    SOURCE_FIRST2M_PAYLOAD_SHA256,
    SUCCESSFUL_UPDATES,
    preprocessing_stamp,
)


ROUND_ID = "0125"
CAPABILITY = "jina-fineweb-2m-runtime-path-density-bridge-v1"
ORIGINAL_RELEASE_SHA = "ff5dfcde5632257aac355008a70bc330bab26bee"
ARMS = ("device_treatment", "host_control")
DEVICE_ARM, HOST_ARM = ARMS
GRAPH_EDGES = 151_202_984
BATCH_SIZE = 8_192
POSITIVE_RATIO = 0.05
NUM_POSITIVE = max(1, int(BATCH_SIZE * POSITIVE_RATIO))
NUM_NEGATIVE = BATCH_SIZE - NUM_POSITIVE
N_EPOCHS = 2
NONINFERIORITY_RATIO = 0.97
MATCHED_DENSITY_FLOOR = 0.17589389755990817
PAIRED_BOOTSTRAP_DRAWS = 1_000
PAIRED_BOOTSTRAP_SEED = 12_501
STREAM_TRACE_BATCHES = 8
DEFAULT_PER_BATCH_EDGE_THRESHOLD = 400_000_000

R0104_RELEASE_SHA = "2b1b51746d4aeb01e9dd88b19aa6dc80ccbb8329"
R0104_SHARED_RECEIPT_SHA256 = (
    "934da48131bf58f890bae9ba1f09f4485e789d844ddf095c9164e67bf8e27869"
)
R0104_GRAPH_SHA256 = (
    "ac36aa60db5f2eeb40ceb52ff6d45ecf5dfa77717df6a26e498393a8265972fd"
)
R0104_GRAPH_MANIFEST_SHA256 = (
    "5903fcecb12495022549ffe357b8710ae2562a790dea4926c16ad887f4bee6da"
)
R0104_HIGH_D_REFERENCE_SHA256 = (
    "803205cbdaabf49c68806470be4eb0254d1b4b2bc6ad34d9a1ef8e55180062c5"
)
R0104_QUERY_TRUTH_SHA256 = (
    "a1dafdf662325bcc3cccf9e1156b74fea36570b6da331cc0fa455a639760418b"
)
R0104_QUERY_TRUTH_KEY = (
    "d5c34b7bb2596f5e51d45de9a3c85def9703ae730cc34761a9e19c75101c9666"
)
R0104_QUERY_TRUTH_PRODUCER_IMPLEMENTATION_SHA256 = (
    "42559e5ff24fceeb1f42927df65ded34f929fc70ad99c9db3fc79fc953293700"
)
R0104_QUERY_TRUTH_PRODUCER_BACKEND = "cuda"
R0104_QUERY_TRUTH_PATH = (
    "/data/latent-basemap/runs/round-0104/queue-attempt-3/artifacts/shared/"
    "oos-query-truth-k10.npz"
)
R0104_FP16_MODEL_SHA256 = (
    "36a7fb86784b6a891f7c73b83d008aead320a7729eea913efc117e4bcd5b3e08"
)
R0104_ACCEPTED_METRICS = {
    "ffr": 0.6227,
    "density": 0.644,
    "recall_at_10": 0.01507,
    "oos_proj_ffr": 0.5587000000000001,
    "oos_proj_recall_at_10": 0.0115,
}
DECISION_METRICS = tuple(R0104_ACCEPTED_METRICS)

R0122_RELEASE_SHA = "79c228e0b5d22027bf76a188c1f1daf895bb2aec"
R0122_PANEL_SHA256 = (
    "6192c648d838e7c2fa6ae901b528ed27125fe32762f5a73c4d9cc03680b15a61"
)
R0122_DECISION_SHA256 = (
    "6f2e7eab0124591977b72cb1e2f00367503b10f1aa49b16ed8b199f8de47ab50"
)
R0122_FP16_MATCHED_DENSITY = 0.15841710834170164

HOST_PIPELINE = "host_weighted_jina_paired"
HOST_SAMPLER = "PairedHostWeightedJinaSampler"
DEVICE_PIPELINE = "device"
DEVICE_SAMPLER = "DeviceEdgeSampler"
TRAIN_MINIMUM_UPDATES_PER_S = 60.0
TRAIN_WARNING_UPDATES_PER_S = 75.0
PERFORMANCE_WARMUP_UPDATES = 200
PERFORMANCE_WINDOW_UPDATES = 2_500
PERFORMANCE_WINDOWS = math.ceil(
    (SUCCESSFUL_UPDATES - PERFORMANCE_WARMUP_UPDATES)
    / PERFORMANCE_WINDOW_UPDATES
)


class Round0125Error(RuntimeError):
    """The frozen R0125 causal or execution contract changed."""


def seal(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    return {**value, "identity_sha256": sha256_bytes(canonical_json(value))}


def validate_seal(value: Mapping[str, Any], *, label: str) -> None:
    body = {key: item for key, item in value.items() if key != "identity_sha256"}
    if value.get("identity_sha256") != sha256_bytes(canonical_json(body)):
        raise Round0125Error(f"{label} identity seal is invalid")


def environment_freeze_receipt() -> dict[str, Any]:
    """Bind the exact Python environment shared by both training arms."""
    packages: list[dict[str, Any]] = []
    for distribution in metadata.distributions():
        name = str(distribution.metadata.get("Name") or "").strip()
        version = str(distribution.version or "").strip()
        if not name or not version:
            raise Round0125Error(
                "installed distribution lacks a stable name or version"
            )
        direct_url = distribution.read_text("direct_url.json")
        record = distribution.read_text("RECORD")
        packages.append({
            "name": name.lower().replace("_", "-"),
            "version": version,
            "direct_url_sha256": (
                sha256_bytes(direct_url.encode("utf-8"))
                if direct_url is not None else None
            ),
            "record_sha256": (
                sha256_bytes(record.encode("utf-8"))
                if record is not None else None
            ),
        })
    packages.sort(key=lambda item: (
        item["name"], item["version"],
        str(item["direct_url_sha256"]), str(item["record_sha256"]),
    ))
    body = {
        "schema": "round0125-python-environment-freeze-v1",
        "python_executable": os.path.abspath(sys.executable),
        "python_prefix": os.path.abspath(sys.prefix),
        "python_version": platform.python_version(),
        "packages": packages,
    }
    return {**body, "freeze_sha256": sha256_bytes(canonical_json(body))}


def validate_environment_freeze(expected: Mapping[str, Any]) -> dict[str, Any]:
    observed = environment_freeze_receipt()
    if observed != dict(expected):
        raise Round0125Error(
            "R0125 execution environment changed after queue preparation"
        )
    return observed


def state_dict_sha256(module: Any) -> str:
    """Stable byte digest of a freshly initialized torch module."""
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


class AuditedParametricUMAP(ParametricUMAP):
    """ParametricUMAP that records initial weights before the first update."""

    initial_model_state_sha256: str | None = None

    def _init_model(self, input_dim: int) -> None:
        super()._init_model(input_dim)
        self.initial_model_state_sha256 = state_dict_sha256(self.model)


def _update_index_digest(digest: Any, values: Any) -> None:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    array = np.ascontiguousarray(values, dtype=np.int64)
    digest.update(np.asarray([len(array)], dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))


class TracingDeviceArrayDataset(DeviceArrayDataset):
    """DeviceArrayDataset with bounded live endpoint-stream accounting."""

    def __init__(
        self,
        X: Any,
        device: str,
        *,
        trace_batches: int = STREAM_TRACE_BATCHES,
    ) -> None:
        super().__init__(X, device)
        self.trace_batches = int(trace_batches)
        self.index_select_calls = 0
        self.source_rows_gathered = 0
        self.destination_rows_gathered = 0
        self._source_digest = hashlib.sha256()
        self._destination_digest = hashlib.sha256()

    def index_select(self, idx: Any):
        role = "source" if self.index_select_calls % 2 == 0 else "destination"
        batch_index = self.index_select_calls // 2
        rows = int(idx.numel()) if hasattr(idx, "numel") else len(idx)
        if batch_index < self.trace_batches:
            _update_index_digest(
                self._source_digest if role == "source" else self._destination_digest,
                idx,
            )
        if role == "source":
            self.source_rows_gathered += rows
        else:
            self.destination_rows_gathered += rows
        self.index_select_calls += 1
        return super().index_select(idx)

    def trace_receipt(self) -> dict[str, Any]:
        return {
            "schema": "round0125-first-batches-endpoint-stream-v1",
            "requested_batches": self.trace_batches,
            "batches_hashed": min(
                self.trace_batches, self.index_select_calls // 2
            ),
            "source_endpoint_ids_sha256": self._source_digest.hexdigest(),
            "destination_endpoint_ids_sha256": (
                self._destination_digest.hexdigest()
            ),
            "index_select_calls": self.index_select_calls,
        }


class TracingPairedHostWeightedJinaSampler(PairedHostWeightedJinaSampler):
    """Exact R0104 host sampler plus a bounded first-batch digest."""

    def __init__(self, *args: Any, trace_batches: int = STREAM_TRACE_BATCHES,
                 **kwargs: Any) -> None:
        self.trace_batches = int(trace_batches)
        self._trace_rows_calls = 0
        self._trace_source_digest = hashlib.sha256()
        self._trace_destination_digest = hashlib.sha256()
        super().__init__(*args, **kwargs)

    def _rows(self) -> tuple[np.ndarray, np.ndarray]:
        source, destination = super()._rows()
        if self._trace_rows_calls < self.trace_batches:
            _update_index_digest(self._trace_source_digest, source)
            _update_index_digest(self._trace_destination_digest, destination)
        self._trace_rows_calls += 1
        return source, destination

    def execution_stamp(self) -> dict[str, Any]:
        stamp = super().execution_stamp()
        stamp.update({
            "trace_wrapper": type(self).__name__,
            "stream_trace": {
                "schema": "round0125-first-batches-endpoint-stream-v1",
                "requested_batches": self.trace_batches,
                "batches_hashed": min(
                    self.trace_batches, self._trace_rows_calls
                ),
                "source_endpoint_ids_sha256": (
                    self._trace_source_digest.hexdigest()
                ),
                "destination_endpoint_ids_sha256": (
                    self._trace_destination_digest.hexdigest()
                ),
            },
        })
        return stamp


class Round0125HostTrainingInput:
    """R0104 host runtime adapter, parameterized only for the CPU smoke."""

    def __init__(self, dataset: Any, graph: Mapping[str, Any], *,
                 expected_rows: int = ROWS) -> None:
        self.dataset = dataset
        self.graph = dict(graph)
        self.expected_rows = int(expected_rows)
        self.shape = dataset.shape
        self._last_sampler: TracingPairedHostWeightedJinaSampler | None = None
        if self.shape[0] != self.expected_rows:
            raise Round0125Error("host training input row geometry changed")

    def __len__(self) -> int:
        return len(self.dataset)

    def to(self, _device: str) -> "Round0125HostTrainingInput":
        return self

    def index_select(self, rows: Any):
        return self.dataset.index_select(rows)

    def prepare_round0034_training(
        self, *, edges_path: str, batch_size: int, pos_ratio: float,
        random_state: int, positive_target_mode: str,
        weighted_edge_sampling: bool, reject_neighbors: bool,
        required_input_pipeline: str | None,
    ):
        if (
            os.path.realpath(edges_path)
            != os.path.realpath(self.graph["signature"]["canonical_path"])
            or positive_target_mode != "binary"
            or not weighted_edge_sampling
            or reject_neighbors
            or required_input_pipeline != HOST_PIPELINE
        ):
            raise Round0125Error("host trainer pipeline request changed")
        sampler = TracingPairedHostWeightedJinaSampler(
            self.dataset,
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
            expected_rows=self.expected_rows,
        )
        self._last_sampler = sampler
        runtime = sampler.execution_stamp()
        return self, sampler, sampler.n_pos, runtime, {
            "graph": self.graph["signature"],
            "graph_manifest": self.graph["manifest_signature"],
            "source_representation": "fp16-control",
        }

    def runtime_stamp(self) -> dict[str, Any]:
        if self._last_sampler is None:
            raise Round0125Error("host sampler was not constructed")
        return self._last_sampler.execution_stamp()


class Round0125DeviceTrainingInput:
    """Exact fp16 source uploaded once for the legacy device sampler."""

    def __init__(self, source: Any, graph: Mapping[str, Any], *,
                 device: str, expected_rows: int = ROWS) -> None:
        self.source = source
        self.graph = dict(graph)
        self.device = str(device)
        self.expected_rows = int(expected_rows)
        self.shape = source.shape
        self._dataset: TracingDeviceArrayDataset | None = None
        self._sampler: DeviceEdgeSampler | None = None
        if self.shape[0] != self.expected_rows:
            raise Round0125Error("device training input row geometry changed")

    def __len__(self) -> int:
        return len(self.source)

    def to(self, _device: str) -> "Round0125DeviceTrainingInput":
        return self

    def index_select(self, rows: Any):
        if self._dataset is None:
            raise Round0125Error("device dataset was not constructed")
        return self._dataset.index_select(rows)

    def prepare_round0034_training(
        self, *, edges_path: str, batch_size: int, pos_ratio: float,
        random_state: int, positive_target_mode: str,
        weighted_edge_sampling: bool, reject_neighbors: bool,
        required_input_pipeline: str | None,
    ):
        threshold_override = os.environ.get("PER_BATCH_EDGE_THRESHOLD")
        if threshold_override is not None:
            raise Round0125Error(
                "R0125 requires the legacy default per-batch threshold; "
                "PER_BATCH_EDGE_THRESHOLD must be unset"
            )
        if (
            os.path.realpath(edges_path)
            != os.path.realpath(self.graph["signature"]["canonical_path"])
            or positive_target_mode != "binary"
            or not weighted_edge_sampling
            or reject_neighbors
            or required_input_pipeline != DEVICE_PIPELINE
        ):
            raise Round0125Error("device trainer pipeline request changed")
        dataset = TracingDeviceArrayDataset(self.source, self.device)
        sampler = DeviceEdgeSampler(
            dataset,
            self.graph["sources"],
            self.graph["targets"],
            self.graph["weights"],
            n_nodes=self.graph["n_nodes"],
            pos_ratio=pos_ratio,
            batch_size=batch_size,
            shuffle=True,
            random_state=random_state,
            positive_target_mode=positive_target_mode,
            weighted_edge_sampling=True,
            uniform_with_replacement=False,
            device=self.device,
        )
        if sampler._per_batch or sampler.sample_cdf is None:
            raise Round0125Error("legacy full-epoch weighted device path changed")
        self._dataset = dataset
        self._sampler = sampler
        runtime = self.runtime_stamp()
        return dataset, sampler, sampler.n_pos, runtime, {
            "graph": self.graph["signature"],
            "graph_manifest": self.graph["manifest_signature"],
            "source_representation": "fp16-control",
        }

    def runtime_stamp(self) -> dict[str, Any]:
        if self._dataset is None or self._sampler is None:
            raise Round0125Error("device sampler was not constructed")
        storage = str(self._dataset.storage_dtype).replace("torch.", "")
        return {
            "schema": "round0125-device-weighted-jina-pipeline-v1",
            "pipeline": DEVICE_PIPELINE,
            "sampler_class": DEVICE_SAMPLER,
            "trace_wrapper": type(self._dataset).__name__,
            "positive_sampling": "weighted_with_replacement",
            "positive_destination_policy": "R0104-fp16-fuzzy-k50",
            "negative_sampling": "uniform-2m-row-universe-nonself",
            "graph_degree": "variable-fuzzy-k50-edge-universe",
            "endpoint_forward": "separate-source-destination",
            "weighted_requested": True,
            "weighted_effective": True,
            "uniform_with_replacement": False,
            "positive_with_replacement": True,
            "multiplicity_policy": "row_multiplicity_uncapped",
            "valid_canonical_edge_count": self._sampler.n_pos,
            "source_representation": "fp16-control",
            "feature_residency": (
                "device-fp16" if storage == "float16" else f"device-{storage}"
            ),
            "device_conversion": "resident-storage-to-device-fp32-on-gather",
            "per_batch_edge_threshold": DEFAULT_PER_BATCH_EDGE_THRESHOLD,
            "per_batch_sampling": bool(self._sampler._per_batch),
            "full_epoch_weighted_draw": not self._sampler._per_batch,
            "graph": self.graph["signature"],
            "graph_manifest": self.graph["manifest_signature"],
            "endpoint_gather_calls": self._dataset.index_select_calls,
            "source_rows_gathered": self._dataset.source_rows_gathered,
            "destination_rows_gathered": self._dataset.destination_rows_gathered,
            "stream_trace": self._dataset.trace_receipt(),
        }


def expected_device_endpoint_accounting(
    *, updates: int = SUCCESSFUL_UPDATES, graph_edges: int = GRAPH_EDGES,
    batch_size: int = BATCH_SIZE, positive_ratio: float = POSITIVE_RATIO,
) -> dict[str, int]:
    num_positive = max(1, int(batch_size * positive_ratio))
    batches_per_epoch = math.ceil(graph_edges / num_positive)
    remainder = graph_edges % num_positive
    shortfall = (num_positive - remainder) % num_positive
    completed_epoch_boundaries = updates // batches_per_epoch
    shortened_rows = completed_epoch_boundaries * shortfall
    return {
        "positive_rows": updates * num_positive - shortened_rows,
        "negative_rows": updates * (batch_size - num_positive),
        "endpoint_rows_per_side": updates * batch_size - shortened_rows,
        "batches_per_epoch": batches_per_epoch,
        "short_last_batch_positive_rows": remainder or num_positive,
        "shortfall_per_completed_epoch": shortfall,
        "completed_epoch_boundaries": completed_epoch_boundaries,
    }


def _expected_pipeline(arm: str, graph_edges: int) -> dict[str, Any]:
    common = {
        "positive_sampling": "weighted_with_replacement",
        "negative_sampling": "uniform-2m-row-universe-nonself",
        "weighted_requested": True,
        "weighted_effective": True,
        "uniform_with_replacement": False,
        "positive_with_replacement": True,
        "multiplicity_policy": "row_multiplicity_uncapped",
        "valid_canonical_edge_count": int(graph_edges),
        "source_representation": "fp16-control",
    }
    if arm == HOST_ARM:
        return {
            **common,
            "pipeline": HOST_PIPELINE,
            "sampler_class": HOST_SAMPLER,
            "positive_destination_policy": "queue-local-fp16-fuzzy-k50",
            "graph_degree": "variable-fuzzy-k50-edge-universe",
            "host_prefetch": "single-producer-two-pinned-slot",
            "endpoint_forward": "fused-source-destination",
            "feature_residency": "host-mmap-fp16-source-shards",
            "device_conversion": "device-fp32-from-exact-fp16",
        }
    if arm == DEVICE_ARM:
        return {
            **common,
            "schema": "round0125-device-weighted-jina-pipeline-v1",
            "pipeline": DEVICE_PIPELINE,
            "sampler_class": DEVICE_SAMPLER,
            "positive_destination_policy": "R0104-fp16-fuzzy-k50",
            "graph_degree": "variable-fuzzy-k50-edge-universe",
            "endpoint_forward": "separate-source-destination",
            "feature_residency": "device-fp16",
            "device_conversion": "resident-storage-to-device-fp32-on-gather",
            "per_batch_edge_threshold": DEFAULT_PER_BATCH_EDGE_THRESHOLD,
            "per_batch_sampling": False,
            "full_epoch_weighted_draw": True,
        }
    raise Round0125Error(f"unknown R0125 arm {arm!r}")


def train_config(
    arm: str, *, graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any], graph_edges: int = GRAPH_EDGES,
) -> tuple[dict[str, Any], str]:
    expected_pipeline = _expected_pipeline(arm, graph_edges)
    invariant = {
        "rows": ROWS,
        "dimension": DIMENSION,
        "source_payload_sha256": SOURCE_FIRST2M_PAYLOAD_SHA256,
        "seed": SEED,
        "graph": dict(graph_signature),
        "graph_manifest": dict(graph_manifest_signature),
        "graph_edges": int(graph_edges),
        "graph_k": GRAPH_K,
        "successful_positive_lr_updates": SUCCESSFUL_UPDATES,
        "batch_size": BATCH_SIZE,
        "positive_ratio": POSITIVE_RATIO,
        "input_preprocessing": preprocessing_stamp("fp16_control"),
    }
    config = {
        "schema": "round0125-device-host-runtime-bridge-config-v1",
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
            "required_pipeline": (
                DEVICE_PIPELINE if arm == DEVICE_ARM else HOST_PIPELINE
            ),
            "gpu_resident_data": arm == DEVICE_ARM,
            "gpu_resident_vram_budget_gb": 31.0 if arm == DEVICE_ARM else 0.0,
            "minimum_train_upd_s": TRAIN_MINIMUM_UPDATES_PER_S,
            "warning_train_upd_s": TRAIN_WARNING_UPDATES_PER_S,
            "performance_subfloor_patience": 2,
            "performance_windows": PERFORMANCE_WINDOWS,
            "stream_trace_batches": STREAM_TRACE_BATCHES,
            "expected_pipeline_stamp": expected_pipeline,
        },
    }
    return config, sha256_bytes(canonical_json(config))


def select_outcome(
    *, host_metrics: Mapping[str, Any], device_metrics: Mapping[str, Any],
    host_matched_density: float, device_matched_density: float,
    paired_delta_ci99: tuple[float, float], execution_valid: bool,
) -> dict[str, Any]:
    if set(host_metrics) != set(DECISION_METRICS) or set(device_metrics) != set(
        DECISION_METRICS
    ):
        raise Round0125Error("native decision metric set changed")
    values = [
        *[float(host_metrics[key]) for key in DECISION_METRICS],
        *[float(device_metrics[key]) for key in DECISION_METRICS],
        float(host_matched_density), float(device_matched_density),
        float(paired_delta_ci99[0]), float(paired_delta_ci99[1]),
    ]
    if not all(np.isfinite(value) for value in values):
        raise Round0125Error("R0125 selector received nonfinite evidence")
    host_reproduction = {
        key: {
            "observed": float(host_metrics[key]),
            "historical": R0104_ACCEPTED_METRICS[key],
            "threshold": NONINFERIORITY_RATIO * R0104_ACCEPTED_METRICS[key],
            "passed": float(host_metrics[key])
            >= NONINFERIORITY_RATIO * R0104_ACCEPTED_METRICS[key],
        }
        for key in DECISION_METRICS
    }
    device_noninferiority = {
        key: {
            "host": float(host_metrics[key]),
            "device": float(device_metrics[key]),
            "threshold": NONINFERIORITY_RATIO * float(host_metrics[key]),
            "passed": float(device_metrics[key])
            >= NONINFERIORITY_RATIO * float(host_metrics[key]),
        }
        for key in DECISION_METRICS
    }
    host_reproduces_native = all(
        value["passed"] for value in host_reproduction.values()
    )
    host_fails_matched = host_matched_density < MATCHED_DENSITY_FLOOR
    device_clears_matched = device_matched_density >= MATCHED_DENSITY_FLOOR
    device_preserves_native = all(
        value["passed"] for value in device_noninferiority.values()
    )
    if not execution_valid:
        outcome = "invalid-execution"
    elif not host_reproduces_native or not host_fails_matched:
        outcome = "historical-host-baseline-not-reproduced"
    elif not device_clears_matched:
        outcome = "device-path-not-sufficient-at-seed42"
    elif paired_delta_ci99[0] > 0.0:
        outcome = (
            "device-path-restores-density-without-native-regression-at-seed42"
            if device_preserves_native else
            "device-path-restores-density-but-regresses-native-panel-at-seed42"
        )
    else:
        outcome = "device-path-effect-inconclusive-at-seed42"
    return {
        "schema": "round0125-device-host-runtime-selector-v1",
        "outcome": outcome,
        "host_historical_native_reproduction": host_reproduction,
        "host_reproduces_historical_native": host_reproduces_native,
        "host_matched_density": float(host_matched_density),
        "host_fails_matched_floor": host_fails_matched,
        "device_matched_density": float(device_matched_density),
        "device_clears_matched_floor": device_clears_matched,
        "paired_device_minus_host_density_ci99": list(paired_delta_ci99),
        "device_native_noninferiority": device_noninferiority,
        "device_preserves_native": device_preserves_native,
        "execution_valid": bool(execution_valid),
        "single_seed_path_bundle_only": True,
    }
