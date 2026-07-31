"""Self-contained paired fp16/int8 contract for Round 0104.

The original R0104 draft tried to compare R0103's first two million rows with
an R0037 map trained on a different selection.  This module deliberately has
no R0037 row or graph dependency.  It exposes the exact R0103 fp16 source
ordering, the matching int8 view, and one sampler used by both registered
arms.  The only arm-level difference is the stored input representation.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import threading
from collections.abc import Mapping
from typing import Any

import numpy as np

from .artifact_identity import (
    canonical_json,
    expected_input_signature,
    ordered_array_sha256,
    sha256_bytes,
)
from .round0034_pipeline import (
    HostInt8MaterializedArray,
    Round0034PipelineError,
)


ROUND_ID = "0104"
ROWS = 2_000_000
DIMENSION = 768
SEED = 42
QUERY_START = ROWS
QUERY_ROWS = 2_000
SUCCESSFUL_UPDATES = 500_000
GRAPH_K = 50

GRAPH_NLIST = 8_192
GRAPH_TRAIN_ROWS = 262_144
GRAPH_TRAIN_SEED = 104
GRAPH_QUALITY_ROWS = 4_096
GRAPH_QUALITY_SEED = 105
GRAPH_NPROBE_GRID = (16, 32, 64, 128, 256)
GRAPH_MEAN_RECALL_FLOOR = 0.90
GRAPH_P10_RECALL_FLOOR = 0.80

PANEL_ANCHORS = 4_000
PANEL_SEED = 123
NONINFERIORITY_RATIO = 0.97
DECISION_METRICS = (
    "ffr",
    "density",
    "recall_at_10",
    "oos_proj_ffr",
    "oos_proj_recall_at_10",
)

ARMS = ("fp16_control", "int8_treatment")
PIPELINE = "host_weighted_jina_paired"
PIPELINE_SCHEMA = "round0104-paired-host-weighted-jina-pipeline-v2"
SAMPLER_CLASS = "PairedHostWeightedJinaSampler"
TRAIN_MINIMUM_UPDATES_PER_S = 60.0
TRAIN_WARNING_UPDATES_PER_S = 75.0
PERFORMANCE_WARMUP_UPDATES = 200
PERFORMANCE_WINDOW_UPDATES = 2_500
PERFORMANCE_WINDOWS = math.ceil(
    (SUCCESSFUL_UPDATES - PERFORMANCE_WARMUP_UPDATES)
    / PERFORMANCE_WINDOW_UPDATES
)

SUBSTRATE_SCHEMA = "jina-diverse-25m-full768-int8-substrate-v1"
SUBSTRATE_ROWS = 25_000_000
SUBSTRATE_MANIFEST = (
    "/data/latent-basemap/runs/round-0103/queue/artifacts/"
    "jina-diverse-25m-full768-int8-substrate/"
    "jina-diverse-25m-full768-int8-substrate-v1.json"
)
SUBSTRATE_MANIFEST_SHA256 = (
    "b01bc7872cbb22e02b64afed1886bed607b21acd9ac0349caaa2fd88713cc7fa"
)
SUBSTRATE_IDENTITY_SHA256 = (
    "d54031a40766df240389c7e370084b59733c9882b89dbc03d239ed3c601fe37b"
)
SUBSTRATE_INT8_SHA256 = (
    "49479596e5de5c8adbba9a6e8811acdd7edcd65287202a328e01d9c3c7236ee2"
)
SUBSTRATE_SCALES_SHA256 = (
    "2f903f064af6b659195d9756ba6756635f1dc003291389614a1a22ff4799f8b6"
)
R0087_INVENTORY_SHA256 = (
    "364aaa2f7a5e886f9cacdb96d3ffef1bbe697148e1babe10eee7817af0fc7163"
)
R0087_INVENTORY_IDENTITY_SHA256 = (
    "6c73f781208d16a84fc9e619e66c89f8fe56375dad77f7eda75e795f85cfec9b"
)
SOURCE_FIRST2M_PAYLOAD_SHA256 = (
    "f4a0050e81a3755de84ba73405ba6823fa387f09a15d3ad299083fa60093f069"
)

_DYNAMIC_PIPELINE_COUNTERS = (
    "endpoint_gather_calls",
    "source_rows_gathered",
    "destination_rows_gathered",
    "host_prefetch_batches_filled",
    "host_prefetch_producer_batches",
    "host_prefetch_consumer_batches",
    "host_prefetch_source_rows_filled",
    "host_prefetch_destination_rows_filled",
)


class Round0104Error(Round0034PipelineError):
    """The registered paired R0104 contract was violated."""


def seal(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    return {**value, "identity_sha256": sha256_bytes(canonical_json(value))}


def validate_seal(value: Mapping[str, Any], *, label: str) -> None:
    body = {key: item for key, item in value.items() if key != "identity_sha256"}
    if value.get("identity_sha256") != sha256_bytes(canonical_json(body)):
        raise Round0104Error(f"{label} identity seal is invalid")


def verify_signature(signature: Any, *, label: str) -> str:
    if not isinstance(signature, Mapping):
        raise Round0104Error(f"{label} signature missing")
    path = str(signature.get("canonical_path") or "")
    if not path or expected_input_signature(path) != dict(signature):
        raise Round0104Error(f"{label} content changed")
    return path


def _manifest_output(
    manifest: Mapping[str, Any],
    key: str,
    *,
    verify_payload: bool,
) -> dict[str, Any]:
    signature = dict((manifest.get("outputs") or {}).get(key) or {})
    path = str(signature.get("canonical_path") or "")
    if not path:
        raise Round0104Error(f"substrate output {key!r} is missing")
    if verify_payload:
        if expected_input_signature(path) != signature:
            raise Round0104Error(f"substrate output {key!r} content changed")
    elif not os.path.isfile(path) or os.path.getsize(path) != int(
        signature.get("bytes", -1)
    ):
        raise Round0104Error(f"substrate output {key!r} missing/wrong size")
    return signature


def validate_substrate_manifest(*, verify_payloads: bool = False) -> dict[str, Any]:
    signature = expected_input_signature(SUBSTRATE_MANIFEST)
    if signature["sha256"] != SUBSTRATE_MANIFEST_SHA256:
        raise Round0104Error("R0103 substrate manifest bytes changed")
    with open(SUBSTRATE_MANIFEST, encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_seal(manifest, label="R0103 substrate manifest")
    outputs = manifest.get("outputs") or {}
    if (
        manifest.get("schema") != SUBSTRATE_SCHEMA
        or manifest.get("round_id") != "0103"
        or manifest.get("identity_sha256") != SUBSTRATE_IDENTITY_SHA256
        or int(manifest.get("row_count", -1)) != SUBSTRATE_ROWS
        or int(manifest.get("dimension", -1)) != DIMENSION
        or manifest.get("output_dtype") != "|i1"
        or manifest.get("scale_dtype") != "<f2"
        or outputs.get("int8", {}).get("sha256") != SUBSTRATE_INT8_SHA256
        or outputs.get("scales", {}).get("sha256") != SUBSTRATE_SCALES_SHA256
        or outputs.get("int8", {}).get("bytes") != SUBSTRATE_ROWS * DIMENSION
        or outputs.get("scales", {}).get("bytes") != SUBSTRATE_ROWS * 2
    ):
        raise Round0104Error("R0103 substrate manifest content changed")
    payloads = {
        key: _manifest_output(
            manifest,
            key,
            verify_payload=verify_payloads and key in {"int8", "scales"},
        )
        for key in ("int8", "scales", "labels", "reconstruction_sample")
    }
    return {"manifest": manifest, "signature": signature, "payloads": payloads}


def _load_inventory() -> tuple[dict[str, Any], dict[str, Any]]:
    substrate = validate_substrate_manifest(verify_payloads=False)
    signature = substrate["manifest"].get("inventory")
    if (
        not isinstance(signature, Mapping)
        or signature.get("sha256") != R0087_INVENTORY_SHA256
    ):
        raise Round0104Error("R0103 inventory binding changed")
    path = verify_signature(signature, label="R0087 inventory")
    with open(path, encoding="utf-8") as handle:
        inventory = json.load(handle)
    validate_seal(inventory, label="R0087 inventory")
    if inventory.get("identity_sha256") != R0087_INVENTORY_IDENTITY_SHA256:
        raise Round0104Error("R0087 inventory identity changed")
    return inventory, dict(signature)


def source_segments(start: int, stop: int) -> list[dict[str, Any]]:
    """Return exact R0087 fp16 shard slices covering ``[start, stop)``."""
    if not 0 <= start < stop <= SUBSTRATE_ROWS:
        raise ValueError("source segment interval is out of range")
    inventory, _signature = _load_inventory()
    ranges = (inventory.get("selection") or {}).get("ranges")
    if not isinstance(ranges, list) or not ranges:
        raise Round0104Error("R0087 inventory ranges are missing")
    selected: list[dict[str, Any]] = []
    cursor = start
    for item in ranges:
        global_start = int(item["global_row_start"])
        global_stop = int(item["global_row_stop"])
        if global_stop <= start:
            continue
        if global_start >= stop:
            break
        take_start = max(start, global_start)
        take_stop = min(stop, global_stop)
        if take_start != cursor or take_stop <= take_start:
            raise Round0104Error("R0087 source ranges are not contiguous")
        shard = item["shard"]
        local_start = int(item["shard_row_start"]) + (take_start - global_start)
        signature = {
            "canonical_path": os.path.realpath(str(shard["canonical_path"])),
            "kind": "file",
            "bytes": int(shard["bytes"]),
            "sha256": str(shard["sha256"]),
        }
        if (
            not os.path.isfile(signature["canonical_path"])
            or os.path.getsize(signature["canonical_path"]) != signature["bytes"]
        ):
            raise Round0104Error("R0087 source shard is missing/wrong size")
        selected.append(
            {
                "global_row_start": take_start,
                "global_row_stop": take_stop,
                "dataset": item.get("dataset"),
                "shard": signature,
                "shard_rows": int(shard["rows"]),
                "shard_row_start": local_start,
                "shard_row_stop": local_start + (take_stop - take_start),
            }
        )
        cursor = take_stop
        if cursor == stop:
            break
    if cursor != stop:
        raise Round0104Error("R0087 source ranges do not cover requested rows")
    return selected


class InventoryFp16Array:
    """Lazy exact fp16 view over an R0087 global-row interval."""

    def __init__(self, start: int, stop: int):
        self.start = int(start)
        self.stop = int(stop)
        self.segments = source_segments(start, stop)
        self.shape = (stop - start, DIMENSION)
        self.dtype = np.dtype("float16")
        self._offsets = np.asarray(
            [item["global_row_start"] - start for item in self.segments]
            + [stop - start],
            dtype=np.int64,
        )
        self._arrays: list[np.ndarray] = []
        for item in self.segments:
            array = np.load(
                item["shard"]["canonical_path"], mmap_mode="r", allow_pickle=False
            )
            if (
                array.dtype != self.dtype
                or array.ndim != 2
                or array.shape != (item["shard_rows"], DIMENSION)
            ):
                raise Round0104Error("R0087 source shard geometry changed")
            self._arrays.append(array)

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def shard_signatures(self) -> list[dict[str, Any]]:
        return [dict(item["shard"]) for item in self.segments]

    def __getitem__(self, key: Any) -> np.ndarray:
        scalar = isinstance(key, (int, np.integer))
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            rows = np.arange(start, stop, step, dtype=np.int64)
        else:
            rows = np.asarray([int(key)] if scalar else key, dtype=np.int64)
        shape = rows.shape
        flat = rows.reshape(-1)
        flat = np.where(flat < 0, flat + len(self), flat)
        if np.any(flat < 0) or np.any(flat >= len(self)):
            raise IndexError("R0104 fp16 source row is out of range")
        out = np.empty((len(flat), DIMENSION), dtype=np.float16)
        segment_ids = np.searchsorted(self._offsets, flat, side="right") - 1
        for segment_id in np.unique(segment_ids):
            mask = segment_ids == segment_id
            item = self.segments[int(segment_id)]
            local = (
                flat[mask]
                - self._offsets[int(segment_id)]
                + int(item["shard_row_start"])
            )
            out[mask] = self._arrays[int(segment_id)][local]
        shaped = out.reshape(shape + (DIMENSION,))
        return shaped[0] if scalar else shaped


class L2NormalizedArray:
    """Lazy fp32 row-normalized view used for cosine graph/scoring truth."""

    def __init__(self, source: Any):
        self.source = source
        self.shape = source.shape
        self.dtype = np.dtype("float32")

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, key: Any) -> np.ndarray:
        values = np.asarray(self.source[key], dtype=np.float32)
        norms = np.linalg.norm(values, axis=-1, keepdims=True)
        if not np.isfinite(values).all() or not np.isfinite(norms).all() or np.any(
            norms <= 0
        ):
            raise Round0104Error("source contains zero/nonfinite rows")
        return values / norms


def source_prefix_proof() -> dict[str, Any]:
    source = InventoryFp16Array(0, ROWS)
    digest = hashlib.sha256()
    for start in range(0, ROWS, 65_536):
        block = np.ascontiguousarray(
            source[start : min(start + 65_536, ROWS)], dtype=np.float16
        )
        digest.update(block.tobytes(order="C"))
    observed = digest.hexdigest()
    if observed != SOURCE_FIRST2M_PAYLOAD_SHA256:
        raise Round0104Error("R0103 first-2M fp16 source payload changed")
    _inventory, inventory_signature = _load_inventory()
    return {
        "schema": "round0104-r0103-first2m-source-proof-v2",
        "rows": ROWS,
        "dimension": DIMENSION,
        "dtype": "<f2",
        "payload_sha256": observed,
        "r0103_inventory": inventory_signature,
        "segments": source.segments,
        "cross_round_row_equivalence_claimed": False,
    }


class HostFp16MaterializedArray:
    """Host-fp16 endpoint provider matching the int8 provider's transfer law."""

    round0034_host_int8 = True

    def __init__(
        self,
        source: InventoryFp16Array,
        *,
        device: str,
        buffer_rows: int,
    ) -> None:
        import torch

        if source.shape != (ROWS, DIMENSION) or buffer_rows <= 0:
            raise Round0104Error("invalid host-fp16 training source")
        self.source = source
        self.shape = source.shape
        self.device = str(device)
        self.buffer_rows = int(buffer_rows)
        self.endpoint_gather_calls = 0
        self.source_rows_gathered = 0
        self.destination_rows_gathered = 0
        self.host_prefetch_batches_filled = 0
        self.host_prefetch_source_rows_filled = 0
        self.host_prefetch_destination_rows_filled = 0
        self._accounting_lock = threading.Lock()
        pin = "cuda" in self.device
        self._slots: list[dict[str, Any]] = []
        for _ in range(2):
            left = torch.empty(
                (buffer_rows, DIMENSION), dtype=torch.float16, pin_memory=pin
            )
            self._slots.append(
                {
                    "source": left,
                    "destination": torch.empty_like(left, pin_memory=pin),
                    "event": None,
                }
            )
        self._slot_index = 0

    def __len__(self) -> int:
        return self.shape[0]

    def to(self, _device: str) -> "HostFp16MaterializedArray":
        return self

    def _rows(
        self, source_rows: Any, destination_rows: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        left = np.asarray(source_rows, dtype=np.int64)
        right = np.asarray(destination_rows, dtype=np.int64)
        if (
            left.ndim != 1
            or right.shape != left.shape
            or len(left) > self.buffer_rows
            or np.any(left < 0)
            or np.any(left >= ROWS)
            or np.any(right < 0)
            or np.any(right >= ROWS)
        ):
            raise Round0104Error("fp16 endpoint rows are invalid")
        return left, right

    def fill_pair_slot(
        self, slot_index: int, source_rows: Any, destination_rows: Any
    ) -> int:
        left, right = self._rows(source_rows, destination_rows)
        if not 0 <= slot_index < len(self._slots):
            raise Round0104Error("fp16 endpoint slot is invalid")
        slot = self._slots[slot_index]
        if slot["event"] is not None:
            slot["event"].synchronize()
        count = len(left)
        slot["source"].numpy()[:count] = self.source[left]
        slot["destination"].numpy()[:count] = self.source[right]
        with self._accounting_lock:
            self.host_prefetch_batches_filled += 1
            self.host_prefetch_source_rows_filled += count
            self.host_prefetch_destination_rows_filled += count
        return count

    def transfer_pair_slot(self, slot_index: int, count: int):
        import torch

        if not 0 <= slot_index < len(self._slots) or not 0 <= count <= self.buffer_rows:
            raise Round0104Error("fp16 endpoint transfer is invalid")
        slot = self._slots[slot_index]
        non_blocking = "cuda" in self.device
        left = slot["source"][:count].to(
            self.device, non_blocking=non_blocking
        ).float()
        right = slot["destination"][:count].to(
            self.device, non_blocking=non_blocking
        ).float()
        if non_blocking:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            slot["event"] = event
        with self._accounting_lock:
            self.endpoint_gather_calls += 1
            self.source_rows_gathered += count
            self.destination_rows_gathered += count
        return left, right

    def gather_pairs(self, source_rows: Any, destination_rows: Any):
        slot = self._slot_index
        self._slot_index = (self._slot_index + 1) % len(self._slots)
        count = self.fill_pair_slot(slot, source_rows, destination_rows)
        return self.transfer_pair_slot(slot, count)

    def index_select(self, rows: Any):
        if hasattr(rows, "detach"):
            rows = rows.detach().cpu().numpy()
        values, _ = self.gather_pairs(rows, rows)
        return values

    def execution_stamp(self) -> dict[str, Any]:
        return {
            "source_representation": "fp16-control",
            "feature_residency": "host-mmap-fp16-source-shards",
            "device_conversion": "device-fp32-from-exact-fp16",
            "source_segments": self.source.segments,
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


def open_training_dataset(
    arm: str,
    *,
    verify_payloads: bool,
    buffer_rows: int,
    device: str = "cuda",
) -> tuple[Any, dict[str, Any]]:
    if arm == "fp16_control":
        substrate = validate_substrate_manifest(verify_payloads=False)
        return (
            HostFp16MaterializedArray(
                InventoryFp16Array(0, ROWS),
                device=device,
                buffer_rows=buffer_rows,
            ),
            substrate,
        )
    if arm != "int8_treatment":
        raise Round0104Error(f"unknown paired arm {arm!r}")
    substrate = validate_substrate_manifest(verify_payloads=verify_payloads)
    outputs = substrate["payloads"]
    encoded = np.memmap(
        outputs["int8"]["canonical_path"],
        dtype=np.int8,
        mode="r",
        shape=(SUBSTRATE_ROWS, DIMENSION),
    )[:ROWS]
    scales = np.memmap(
        outputs["scales"]["canonical_path"],
        dtype="<f2",
        mode="r",
        shape=(SUBSTRATE_ROWS,),
    )[:ROWS]
    dataset = HostInt8MaterializedArray(
        encoded,
        scales,
        device=device,
        signatures={"int8": outputs["int8"], "scales": outputs["scales"]},
        buffer_rows=buffer_rows,
    )
    return dataset, substrate


class Int8DequantizedArray:
    """Lazy fp32 view of an arbitrary contiguous R0103 int8 interval."""

    def __init__(self, start: int, stop: int):
        substrate = validate_substrate_manifest(verify_payloads=True)
        outputs = substrate["payloads"]
        self.encoded = np.memmap(
            outputs["int8"]["canonical_path"],
            dtype=np.int8,
            mode="r",
            shape=(SUBSTRATE_ROWS, DIMENSION),
        )
        self.scales = np.memmap(
            outputs["scales"]["canonical_path"],
            dtype="<f2",
            mode="r",
            shape=(SUBSTRATE_ROWS,),
        )
        self.start = int(start)
        self.stop = int(stop)
        self.shape = (stop - start, DIMENSION)
        self.dtype = np.dtype("float32")
        self.substrate = substrate

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key: Any) -> np.ndarray:
        scalar = isinstance(key, (int, np.integer))
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            rows = np.arange(start, stop, step, dtype=np.int64)
        else:
            rows = np.asarray([int(key)] if scalar else key, dtype=np.int64)
        shape = rows.shape
        flat = rows.reshape(-1)
        flat = np.where(flat < 0, flat + len(self), flat)
        if np.any(flat < 0) or np.any(flat >= len(self)):
            raise IndexError("int8 transform row is out of range")
        absolute = flat + self.start
        values = np.asarray(self.encoded[absolute], dtype=np.float32)
        values *= np.asarray(self.scales[absolute], dtype=np.float32)[:, None]
        shaped = values.reshape(shape + (DIMENSION,))
        return shaped[0] if scalar else shaped


def transform_array(arm: str, start: int, stop: int) -> tuple[Any, dict[str, Any]]:
    if arm == "fp16_control":
        return InventoryFp16Array(start, stop), validate_substrate_manifest(
            verify_payloads=False
        )
    if arm == "int8_treatment":
        value = Int8DequantizedArray(start, stop)
        return value, value.substrate
    raise Round0104Error(f"unknown paired arm {arm!r}")


def preprocessing_stamp(arm: str) -> dict[str, Any]:
    if arm not in ARMS:
        raise Round0104Error(f"unknown paired arm {arm!r}")
    body = {
        "schema": "round0104-paired-input-preprocessing-v2",
        "arm": arm,
        "source_rows": [0, ROWS],
        "source_dimension": DIMENSION,
        "effective_dimension": DIMENSION,
        "compute_dtype": "<f4",
        "operation": (
            "exact-fp16-to-device-fp32"
            if arm == "fp16_control"
            else "signed-int8-times-exact-fp16-row-scale-to-device-fp32"
        ),
        "l2_renormalized_for_training": False,
    }
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}


def panel_config():
    from .panel_v2 import PanelV2Config

    return PanelV2Config(
        frac=0.001,
        k_clust=(),
        k_density=15,
        k_hit=10,
        n_anchors=PANEL_ANCHORS,
        anchor_seed=PANEL_SEED,
        corpus_chunk=500_000,
        overselect=8,
        block_elems=500_000_000,
        rerank_byte_cap=2_000_000_000,
        rerank_scratch=3.0,
        peak_byte_cap=26_000_000_000,
    )


def train_config(
    arm: str,
    *,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    graph_edges: int,
) -> tuple[dict[str, Any], str]:
    if arm not in ARMS or graph_edges <= 0:
        raise Round0104Error("invalid paired train config input")
    stamp = preprocessing_stamp(arm)
    expected_pipeline = {
        "schema": PIPELINE_SCHEMA,
        "pipeline": PIPELINE,
        "sampler_class": SAMPLER_CLASS,
        "positive_sampling": "weighted_with_replacement",
        "positive_destination_policy": "queue-local-fp16-fuzzy-k50",
        "negative_sampling": "uniform-2m-row-universe-nonself",
        "graph_degree": "variable-fuzzy-k50-edge-universe",
        "host_prefetch": "single-producer-two-pinned-slot",
        "endpoint_forward": "fused-source-destination",
        "weighted_requested": True,
        "weighted_effective": True,
        "uniform_with_replacement": False,
        "positive_with_replacement": True,
        "multiplicity_policy": "row_multiplicity_uncapped",
        "valid_canonical_edge_count": int(graph_edges),
        "source_representation": (
            "fp16-control" if arm == "fp16_control" else "int8-treatment"
        ),
    }
    config = {
        "schema": "round0104-self-contained-paired-train-config-v2",
        "arm": arm,
        "paired_invariant": {
            "rows": ROWS,
            "dimension": DIMENSION,
            "seed": SEED,
            "graph": dict(graph_signature),
            "graph_manifest": dict(graph_manifest_signature),
            "successful_positive_lr_updates": SUCCESSFUL_UPDATES,
            "sampler": SAMPLER_CLASS,
        },
        "input_preprocessing": stamp,
        "graph": {
            "path": str(graph_signature["canonical_path"]),
            "sha256": str(graph_signature["sha256"]),
            "manifest_path": str(graph_manifest_signature["canonical_path"]),
            "manifest_sha256": str(graph_manifest_signature["sha256"]),
            "k": GRAPH_K,
            "directed_edges": int(graph_edges),
            "sampling": "fuzzy-weight-proportional-with-replacement",
            "positive_target_mode": "binary",
        },
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
            "batch_size": 8192,
            "positive_ratio": 0.05,
            "positive_target_mode": "binary",
            "weighted_edge_sampling": True,
            "correlation_weight": 0.0,
            "clip_grad_norm": 1.0,
            "use_amp": "bf16",
            "schedule": "cosine-v3-positive-budget",
            "warmup_successful_updates": PERFORMANCE_WARMUP_UPDATES,
            "successful_positive_lr_updates": SUCCESSFUL_UPDATES,
            "reject_neighbors": False,
        },
        "execution": {
            "device_count": 1,
            "required_pipeline": PIPELINE,
            "gpu_resident_data": False,
            "gpu_resident_vram_budget_gb": 0.0,
            "minimum_train_upd_s": TRAIN_MINIMUM_UPDATES_PER_S,
            "warning_train_upd_s": TRAIN_WARNING_UPDATES_PER_S,
            "performance_subfloor_patience": 2,
            "performance_windows": PERFORMANCE_WINDOWS,
            "expected_pipeline_stamp": expected_pipeline,
        },
    }
    return config, sha256_bytes(canonical_json(config))


class PairedHostWeightedJinaSampler:
    """One deterministic endpoint/sampling implementation for both arms."""

    fused_endpoint_forward = True

    def __init__(
        self,
        dataset: Any,
        *,
        sources: np.ndarray,
        targets: np.ndarray,
        weights: np.ndarray,
        n_nodes: int,
        batch_size: int,
        pos_ratio: float,
        random_state: int,
        graph_signature: Mapping[str, Any],
        graph_manifest_signature: Mapping[str, Any],
        arm: str,
        expected_rows: int = ROWS,
    ) -> None:
        import torch

        self.dataset = dataset
        self.sources = np.asarray(sources, dtype=np.int32)
        self.targets = np.asarray(targets, dtype=np.int32)
        self.weights = np.asarray(weights, dtype=np.float32)
        self.n_nodes = int(n_nodes)
        self.n_pos = len(self.sources)
        self.batch_size = int(batch_size)
        self.num_pos = max(1, int(batch_size * pos_ratio))
        self.num_neg = self.batch_size - self.num_pos
        self.rng = np.random.default_rng(int(random_state))
        self.graph_signature = dict(graph_signature)
        self.graph_manifest_signature = dict(graph_manifest_signature)
        self.arm = arm
        self.expected_rows = int(expected_rows)
        self.device = dataset.device
        self.batch_no = 0
        self._prefetch_executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._prefetch_future: concurrent.futures.Future[tuple[int, int]] | None = None
        self._producer_batches = 0
        self._consumer_batches = 0
        if (
            arm not in ARMS
            or self.expected_rows < 2
            or len(dataset) != self.expected_rows
            or self.n_nodes != self.expected_rows
            or self.sources.ndim != 1
            or self.targets.shape != self.sources.shape
            or self.weights.shape != self.sources.shape
            or self.n_pos <= 0
            or self.num_neg <= 0
            or self.sources.min(initial=0) < 0
            or self.targets.min(initial=0) < 0
            or self.sources.max(initial=0) >= self.expected_rows
            or self.targets.max(initial=0) >= self.expected_rows
            or not np.isfinite(self.weights).all()
            or np.any(self.weights <= 0)
        ):
            raise Round0104Error("paired sampler graph/dataset is invalid")
        cdf = np.cumsum(self.weights, dtype=np.float64)
        cdf /= float(cdf[-1])
        self.sample_cdf = cdf
        self._labels = torch.cat(
            (
                torch.ones(self.num_pos, dtype=torch.float32, device=self.device),
                torch.zeros(self.num_neg, dtype=torch.float32, device=self.device),
            )
        )

    def __len__(self) -> int:
        return int(math.ceil(self.n_pos / self.num_pos))

    def __iter__(self) -> "PairedHostWeightedJinaSampler":
        self.batch_no = 0
        if self._prefetch_executor is None:
            self._prefetch_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"r0104-{self.arm}-prefetch"
            )
        if self._prefetch_future is None:
            self._prefetch_future = self._prefetch_executor.submit(
                self._prefetch_one, 0
            )
        return self

    def _rows(self) -> tuple[np.ndarray, np.ndarray]:
        draws = self.rng.random(self.num_pos, dtype=np.float64)
        edge_ids = np.minimum(
            np.searchsorted(self.sample_cdf, draws, side="left"), self.n_pos - 1
        )
        source = np.asarray(self.sources[edge_ids], dtype=np.int64)
        destination = np.asarray(self.targets[edge_ids], dtype=np.int64)
        neg_source = self.rng.integers(
            0, self.n_nodes, size=self.num_neg, dtype=np.int64
        )
        offset = self.rng.integers(
            1, self.n_nodes, size=self.num_neg, dtype=np.int64
        )
        return (
            np.concatenate((source, neg_source)),
            np.concatenate((destination, (neg_source + offset) % self.n_nodes)),
        )

    def _prefetch_one(self, slot: int) -> tuple[int, int]:
        left, right = self._rows()
        count = self.dataset.fill_pair_slot(slot, left, right)
        self._producer_batches += 1
        return slot, count

    def __next__(self):
        if self.batch_no >= len(self):
            raise StopIteration
        self.batch_no += 1
        if self._prefetch_future is None or self._prefetch_executor is None:
            raise Round0104Error("paired prefetch was not initialized")
        slot, count = self._prefetch_future.result()
        left, right = self.dataset.transfer_pair_slot(slot, count)
        self._consumer_batches += 1
        if self.batch_no < len(self):
            self._prefetch_future = self._prefetch_executor.submit(
                self._prefetch_one, (slot + 1) % len(self.dataset._slots)
            )
        else:
            self._prefetch_future = None
        return left, right, self._labels

    def close(self) -> None:
        if self._prefetch_future is not None:
            self._prefetch_future.result()
        if self._prefetch_executor is not None:
            self._prefetch_executor.shutdown(wait=True, cancel_futures=True)
        self._prefetch_executor = None
        self._prefetch_future = None

    def execution_stamp(self) -> dict[str, Any]:
        dataset = self.dataset.execution_stamp()
        representation = dataset.get("source_representation")
        if representation is None and self.arm == "int8_treatment":
            representation = "int8-treatment"
        if representation not in {"fp16-control", "int8-treatment"}:
            raise Round0104Error("paired dataset representation stamp changed")
        return {
            "schema": PIPELINE_SCHEMA,
            "pipeline": PIPELINE,
            "sampler_class": SAMPLER_CLASS,
            "positive_sampling": "weighted_with_replacement",
            "positive_destination_policy": "queue-local-fp16-fuzzy-k50",
            "negative_sampling": "uniform-2m-row-universe-nonself",
            "graph_degree": "variable-fuzzy-k50-edge-universe",
            "host_prefetch": "single-producer-two-pinned-slot",
            "host_prefetch_producer_batches": self._producer_batches,
            "host_prefetch_consumer_batches": self._consumer_batches,
            "endpoint_forward": "fused-source-destination",
            "weighted_requested": True,
            "weighted_effective": True,
            "uniform_with_replacement": False,
            "positive_with_replacement": True,
            "multiplicity_policy": "row_multiplicity_uncapped",
            "valid_canonical_edge_count": self.n_pos,
            "source_representation": representation,
            "graph": self.graph_signature,
            "graph_manifest": self.graph_manifest_signature,
            **dataset,
        }


class Round0104TrainingInput:
    """ParametricUMAP adapter that keeps both paired arms on one sampler."""

    round0034_host_int8 = True

    def __init__(
        self,
        dataset: Any,
        graph: Mapping[str, Any],
        *,
        arm: str,
        required_pipeline: str,
    ) -> None:
        self.dataset = dataset
        self.graph = dict(graph)
        self.arm = arm
        self.required_pipeline = required_pipeline
        self.shape = dataset.shape
        self._last_sampler: PairedHostWeightedJinaSampler | None = None
        if self.shape != (ROWS, DIMENSION) or arm not in ARMS:
            raise Round0104Error("paired training input geometry changed")

    def __len__(self) -> int:
        return len(self.dataset)

    def to(self, _device: str) -> "Round0104TrainingInput":
        return self

    def index_select(self, rows: Any):
        return self.dataset.index_select(rows)

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
            or required_input_pipeline != self.required_pipeline
            or self.required_pipeline != PIPELINE
        ):
            raise Round0104Error("paired trainer pipeline request changed")
        sampler = PairedHostWeightedJinaSampler(
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
            arm=self.arm,
        )
        self._last_sampler = sampler
        runtime = sampler.execution_stamp()
        verified = {
            "graph": self.graph["signature"],
            "graph_manifest": self.graph["manifest_signature"],
            "source_representation": runtime["source_representation"],
        }
        return self, sampler, sampler.n_pos, runtime, verified

    def runtime_stamp(self) -> dict[str, Any]:
        if self._last_sampler is None:
            raise Round0104Error("paired sampler has not been constructed")
        return self._last_sampler.execution_stamp()


def synchronize_runtime_counters(
    accounting: dict[str, Any], runtime: Mapping[str, Any]
) -> None:
    for key in _DYNAMIC_PIPELINE_COUNTERS:
        flattened = f"pipeline_{key}"
        if key not in runtime or flattened not in accounting:
            raise Round0104Error(f"missing runtime accounting field {key}")
        accounting[flattened] = runtime[key]


def paired_decision(
    *,
    control: Mapping[str, Any],
    treatment: Mapping[str, Any],
) -> dict[str, Any]:
    control_metrics = dict(control.get("metrics") or {})
    treatment_metrics = dict(treatment.get("metrics") or {})
    if set(control_metrics) != set(DECISION_METRICS) or set(
        treatment_metrics
    ) != set(DECISION_METRICS):
        raise Round0104Error("paired decision metric set changed")
    metric_gates: dict[str, Any] = {}
    for metric in DECISION_METRICS:
        baseline = float(control_metrics[metric])
        observed = float(treatment_metrics[metric])
        if not np.isfinite(baseline) or not np.isfinite(observed):
            raise Round0104Error("paired decision metric is nonfinite")
        threshold = NONINFERIORITY_RATIO * baseline
        metric_gates[metric] = {
            "control": baseline,
            "treatment": observed,
            "threshold": threshold,
            "ratio": observed / baseline if baseline != 0 else None,
            "passed": observed >= threshold,
        }
    execution_gates = {
        arm: all(bool(value) for value in (report.get("execution_gates") or {}).values())
        for arm, report in (
            ("fp16_control", control),
            ("int8_treatment", treatment),
        )
    }
    passed = all(row["passed"] for row in metric_gates.values()) and all(
        execution_gates.values()
    )
    return {
        "schema": "round0104-self-contained-paired-decision-v2",
        "rows": ROWS,
        "seed": SEED,
        "successful_updates_per_arm": SUCCESSFUL_UPDATES,
        "noninferiority_ratio": NONINFERIORITY_RATIO,
        "metric_gates": metric_gates,
        "execution_gates": execution_gates,
        "passed": passed,
        "outcome": (
            "release-capability:jina-full768-host-int8-training-validation-v1"
            if passed
            else "terminal-negative-use-host-fp16-fallback"
        ),
    }


def sample_sha256(values: np.ndarray) -> str:
    return ordered_array_sha256(np.ascontiguousarray(values))
