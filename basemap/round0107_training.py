"""Training contract for the retained 25M diverse-Jina atlas."""
from __future__ import annotations

import concurrent.futures
import json
import math
import os
from collections.abc import Mapping
from typing import Any

import numpy as np

from .artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from .round0034_pipeline import HostInt8MaterializedArray
from .round0104_training import validate_substrate_manifest
from .round0106_graph import (
    GRAPH_SCHEMA,
    N_NEIGHBORS,
    RETAINED_ROWS,
)


ROUND_ID = "0107"
DIMENSION = 768
SEED = 42
BATCH_SIZE = 8_192
POSITIVE_RATIO = 0.05
POSITIVE_ROWS_PER_UPDATE = int(BATCH_SIZE * POSITIVE_RATIO)
PERFORMANCE_WARMUP_UPDATES = 200
PERFORMANCE_WINDOW_UPDATES = 2_500
TRAIN_MINIMUM_UPDATES_PER_S = 70.0
TRAIN_WARNING_UPDATES_PER_S = 80.0

PIPELINE = "host_weighted_jina_diverse_25m"
PIPELINE_SCHEMA = "round0107-host-weighted-jina-diverse-pipeline-v1"
SAMPLER_CLASS = "DiverseWeightedJinaSampler"
TRAIN_RECEIPT_SCHEMA = "round0107-diverse-jina-train-receipt-v1"


class Round0107Error(RuntimeError):
    """The R0107 training contract was violated."""


def seal(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    return {**value, "identity_sha256": sha256_bytes(canonical_json(value))}


def validate_seal(value: Mapping[str, Any], *, label: str) -> None:
    body = {key: item for key, item in value.items() if key != "identity_sha256"}
    if value.get("identity_sha256") != sha256_bytes(canonical_json(body)):
        raise Round0107Error(f"{label} identity seal is invalid")


def verify_signature(signature: Any, *, label: str) -> str:
    if not isinstance(signature, Mapping):
        raise Round0107Error(f"{label} signature missing")
    path = str(signature.get("canonical_path") or "")
    if not path or expected_input_signature(path) != dict(signature):
        raise Round0107Error(f"{label} content changed")
    return path


def successful_update_target(directed_edge_count: int) -> int:
    if directed_edge_count <= 0:
        raise Round0107Error("R0107 graph must contain positive edges")
    return (
        int(directed_edge_count) + POSITIVE_ROWS_PER_UPDATE - 1
    ) // POSITIVE_ROWS_PER_UPDATE


def performance_windows(successful_updates: int) -> int:
    return max(
        1,
        math.ceil(
            max(0, successful_updates - PERFORMANCE_WARMUP_UPDATES)
            / PERFORMANCE_WINDOW_UPDATES
        ),
    )


def load_graph_manifest(
    path: str,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise Round0107Error("R0106 graph manifest bytes changed")
    with open(signature["canonical_path"], encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_seal(manifest, label="R0106 graph manifest")
    outputs = manifest.get("outputs") or {}
    if (
        manifest.get("schema") != GRAPH_SCHEMA
        or manifest.get("round_id") != "0106"
        or int(manifest.get("retained_rows", -1)) != RETAINED_ROWS
        or int(manifest.get("dimension", -1)) != DIMENSION
        or int(manifest.get("k_real", -1)) != N_NEIGHBORS - 1
        or int(manifest.get("n_neighbors_including_self", -1))
        != N_NEIGHBORS
        or int(manifest.get("directed_edge_count", -1)) <= 0
        or set(outputs) != {"sources", "targets", "weights"}
        or (manifest.get("reciprocity_validation") or {}).get(
            "every_reverse_present_once"
        )
        is not True
    ):
        raise Round0107Error("R0106 graph manifest contract changed")
    paths = {
        key: verify_signature(outputs[key], label=f"R0106 graph {key}")
        for key in ("sources", "targets", "weights")
    }
    mapping_path = verify_signature(
        manifest.get("compact_mapping"), label="R0106 compact mapping"
    )
    edges = int(manifest["directed_edge_count"])
    arrays = {
        "sources": np.load(paths["sources"], mmap_mode="r", allow_pickle=False),
        "targets": np.load(paths["targets"], mmap_mode="r", allow_pickle=False),
        "weights": np.load(paths["weights"], mmap_mode="r", allow_pickle=False),
        "mapping": np.load(mapping_path, mmap_mode="r", allow_pickle=False),
    }
    if (
        arrays["sources"].shape != (edges,)
        or arrays["sources"].dtype != np.int32
        or arrays["targets"].shape != (edges,)
        or arrays["targets"].dtype != np.int32
        or arrays["weights"].shape != (edges,)
        or arrays["weights"].dtype != np.float32
        or arrays["mapping"].shape != (RETAINED_ROWS,)
        or arrays["mapping"].dtype != np.int64
    ):
        raise Round0107Error("R0106 graph arrays changed geometry")
    return {
        "manifest": manifest,
        "signature": signature,
        "arrays": arrays,
        "successful_updates": successful_update_target(edges),
    }


class CompactMappedInt8Array:
    """Lazy compact-ID view over R0103's original global int8 rows."""

    def __init__(self, source: np.ndarray, mapping: np.ndarray):
        if (
            source.ndim != 2
            or source.shape[1] != DIMENSION
            or source.dtype != np.int8
            or mapping.shape != (RETAINED_ROWS,)
            or mapping.dtype != np.int64
        ):
            raise Round0107Error("R0107 compact feature view is malformed")
        self.source = source
        self.mapping = mapping
        self.shape = (RETAINED_ROWS, DIMENSION)
        self.ndim = 2
        self.dtype = np.dtype("int8")

    def __len__(self) -> int:
        return RETAINED_ROWS

    def __getitem__(self, key: Any) -> np.ndarray:
        return self.source[self.mapping[key]]


class CompactHostInt8MaterializedArray(HostInt8MaterializedArray):
    """R0104-validated host pipeline with an explicit compact-ID mapping."""

    def __init__(
        self,
        *,
        mapping: np.ndarray,
        buffer_rows: int,
        device: str = "cuda",
    ) -> None:
        substrate = validate_substrate_manifest(verify_payloads=True)
        outputs = substrate["payloads"]
        encoded = np.memmap(
            outputs["int8"]["canonical_path"],
            dtype=np.int8,
            mode="r",
            shape=(25_000_000, DIMENSION),
        )
        global_scales = np.memmap(
            outputs["scales"]["canonical_path"],
            dtype="<f2",
            mode="r",
            shape=(25_000_000,),
        )
        compact_encoded = CompactMappedInt8Array(encoded, mapping)
        compact_scales = np.asarray(global_scales[mapping], dtype="<f2")
        super().__init__(
            compact_encoded,
            compact_scales,
            device=device,
            signatures={
                "int8": outputs["int8"],
                "scales": outputs["scales"],
            },
            buffer_rows=buffer_rows,
        )
        self.mapping = mapping
        self.substrate = substrate

    def execution_stamp(self) -> dict[str, Any]:
        return {
            **super().execution_stamp(),
            "source_representation": "int8-treatment",
            "feature_residency": (
                "host-mmap-global-int8-plus-compact-map-and-host-fp16-scale"
            ),
            "compact_mapping_rows": len(self.mapping),
            "compact_mapping_semantics": (
                "R0106-retained-compact-id-to-R0103-global-row"
            ),
        }


class DiverseWeightedJinaSampler:
    """Exact fuzzy-weight sampling via uniform-envelope rejection."""

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
        graph_signatures: Mapping[str, Any],
    ) -> None:
        import torch

        self.dataset = dataset
        self.sources = sources
        self.targets = targets
        self.weights = weights
        self.n_nodes = int(n_nodes)
        self.n_pos = int(len(sources))
        self.batch_size = int(batch_size)
        self.num_pos = max(1, int(batch_size * pos_ratio))
        self.num_neg = self.batch_size - self.num_pos
        self.rng = np.random.default_rng(int(random_state))
        self.graph_signatures = dict(graph_signatures)
        self.device = dataset.device
        self.batch_no = 0
        self._prefetch_executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._prefetch_future: concurrent.futures.Future[tuple[int, int]] | None = None
        self._producer_batches = 0
        self._consumer_batches = 0
        self._weight_proposals = 0
        self._weight_acceptances = 0
        self._weight_emitted_draws = 0
        self._accepted_buffer = np.empty(0, dtype=np.int64)
        self._rejection_iterations = 0
        if (
            len(dataset) != self.n_nodes
            or self.sources.ndim != 1
            or self.targets.shape != self.sources.shape
            or self.weights.shape != self.sources.shape
            or self.sources.dtype != np.int32
            or self.targets.dtype != np.int32
            or self.weights.dtype != np.float32
            or self.n_pos <= 0
            or self.num_neg <= 0
            or self.sources.min(initial=0) < 0
            or self.targets.min(initial=0) < 0
            or self.sources.max(initial=0) >= self.n_nodes
            or self.targets.max(initial=0) >= self.n_nodes
            or not np.isfinite(self.weights).all()
            or np.any(self.weights <= 0)
            or np.any(self.weights > 1)
        ):
            raise Round0107Error("R0107 sampler graph/dataset is invalid")
        self._labels = torch.cat(
            (
                torch.ones(self.num_pos, dtype=torch.float32, device=self.device),
                torch.zeros(self.num_neg, dtype=torch.float32, device=self.device),
            )
        )

    def __len__(self) -> int:
        return successful_update_target(self.n_pos)

    def __iter__(self) -> "DiverseWeightedJinaSampler":
        self.batch_no = 0
        if self._prefetch_executor is None:
            self._prefetch_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="r0107-weighted-prefetch"
            )
        if self._prefetch_future is None:
            self._prefetch_future = self._prefetch_executor.submit(
                self._prefetch_one, 0
            )
        return self

    def _draw_weighted_edge_ids(self, count: int) -> np.ndarray:
        """Sample proportional to w exactly because every w is in (0,1]."""
        output = np.empty(count, dtype=np.int64)
        buffered = min(count, len(self._accepted_buffer))
        if buffered:
            output[:buffered] = self._accepted_buffer[:buffered]
            self._accepted_buffer = self._accepted_buffer[buffered:]
        filled = buffered
        iterations = 0
        while filled < count:
            remaining = count - filled
            proposal_count = max(1_024, remaining * 3)
            edge_ids = self.rng.integers(
                0, self.n_pos, size=proposal_count, dtype=np.int64
            )
            uniforms = self.rng.random(proposal_count, dtype=np.float32)
            accepted = edge_ids[
                uniforms < np.asarray(self.weights[edge_ids], dtype=np.float32)
            ]
            take = min(remaining, len(accepted))
            if take:
                output[filled : filled + take] = accepted[:take]
                filled += take
            if take < len(accepted):
                self._accepted_buffer = np.concatenate(
                    (self._accepted_buffer, accepted[take:])
                )
            self._weight_proposals += proposal_count
            self._weight_acceptances += len(accepted)
            iterations += 1
            if iterations > 10_000:
                raise Round0107Error("R0107 weighted rejection did not progress")
        self._weight_emitted_draws += count
        self._rejection_iterations += iterations
        return output

    def _rows(self) -> tuple[np.ndarray, np.ndarray]:
        edge_ids = self._draw_weighted_edge_ids(self.num_pos)
        source = np.asarray(self.sources[edge_ids], dtype=np.int64)
        destination = np.asarray(self.targets[edge_ids], dtype=np.int64)
        negative_source = self.rng.integers(
            0, self.n_nodes, size=self.num_neg, dtype=np.int64
        )
        offset = self.rng.integers(
            1, self.n_nodes, size=self.num_neg, dtype=np.int64
        )
        return (
            np.concatenate((source, negative_source)),
            np.concatenate(
                (destination, (negative_source + offset) % self.n_nodes)
            ),
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
            raise Round0107Error("R0107 prefetch was not initialized")
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
        return {
            "schema": PIPELINE_SCHEMA,
            "pipeline": PIPELINE,
            "sampler_class": SAMPLER_CLASS,
            "positive_sampling": (
                "fuzzy_weight_proportional_with_replacement_via_exact_"
                "uniform_envelope_rejection"
            ),
            "positive_destination_policy": (
                "R0106-global-retained-fuzzy-tconorm-graph"
            ),
            "negative_sampling": (
                "uniform-24,948,663-compact-retained-rows-nonself"
            ),
            "graph_degree": "variable-symmetric-fuzzy-k15-topology",
            "host_prefetch": "single-producer-two-pinned-slot",
            "host_prefetch_producer_batches": self._producer_batches,
            "host_prefetch_consumer_batches": self._consumer_batches,
            "endpoint_forward": "fused-source-destination",
            "weighted_requested": True,
            "weighted_effective": True,
            "uniform_with_replacement": False,
            "positive_with_replacement": True,
            "weight_sampler": "uniform-envelope-rejection-max-weight-one",
            "weight_proposals": self._weight_proposals,
            "weight_acceptances": self._weight_acceptances,
            "weight_emitted_draws": self._weight_emitted_draws,
            "weight_buffered_draws": len(self._accepted_buffer),
            "weight_acceptance_rate": (
                self._weight_acceptances / self._weight_proposals
                if self._weight_proposals
                else None
            ),
            "weight_rejection_iterations": self._rejection_iterations,
            "valid_canonical_edge_count": self.n_pos,
            "compact_retained_rows": self.n_nodes,
            "graph": self.graph_signatures,
            **dataset,
        }


class Round0107TrainingInput:
    """ParametricUMAP adapter for the compact retained fuzzy graph."""

    round0034_host_int8 = True

    def __init__(
        self,
        dataset: Any,
        graph: Mapping[str, Any],
        *,
        required_pipeline: str,
    ) -> None:
        self.dataset = dataset
        self.graph = dict(graph)
        self.required_pipeline = required_pipeline
        self.shape = dataset.shape
        self._last_sampler: DiverseWeightedJinaSampler | None = None
        if self.shape != (RETAINED_ROWS, DIMENSION):
            raise Round0107Error("R0107 training input geometry changed")

    def __len__(self) -> int:
        return len(self.dataset)

    def to(self, _device: str) -> "Round0107TrainingInput":
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
            raise Round0107Error("R0107 trainer pipeline request changed")
        sampler = DiverseWeightedJinaSampler(
            self.dataset,
            sources=self.graph["sources"],
            targets=self.graph["targets"],
            weights=self.graph["weights"],
            n_nodes=RETAINED_ROWS,
            batch_size=batch_size,
            pos_ratio=pos_ratio,
            random_state=random_state,
            graph_signatures=self.graph["graph_signatures"],
        )
        self._last_sampler = sampler
        runtime = sampler.execution_stamp()
        verified = {
            "graph_manifest": self.graph["signature"],
            "graph_outputs": self.graph["graph_signatures"],
            "compact_mapping": self.graph["mapping_signature"],
            "source_representation": runtime["source_representation"],
        }
        return self, sampler, sampler.n_pos, runtime, verified

    def runtime_stamp(self) -> dict[str, Any]:
        if self._last_sampler is None:
            raise Round0107Error("R0107 sampler has not been constructed")
        return self._last_sampler.execution_stamp()


def train_config(
    *,
    graph_manifest: Mapping[str, Any],
    graph_signature: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    edges = int(graph_manifest["directed_edge_count"])
    updates = successful_update_target(edges)
    expected_pipeline = {
        "schema": PIPELINE_SCHEMA,
        "pipeline": PIPELINE,
        "sampler_class": SAMPLER_CLASS,
        "positive_sampling": (
            "fuzzy_weight_proportional_with_replacement_via_exact_"
            "uniform_envelope_rejection"
        ),
        "positive_destination_policy": (
            "R0106-global-retained-fuzzy-tconorm-graph"
        ),
        "negative_sampling": (
            "uniform-24,948,663-compact-retained-rows-nonself"
        ),
        "graph_degree": "variable-symmetric-fuzzy-k15-topology",
        "host_prefetch": "single-producer-two-pinned-slot",
        "endpoint_forward": "fused-source-destination",
        "weighted_requested": True,
        "weighted_effective": True,
        "uniform_with_replacement": False,
        "positive_with_replacement": True,
        "weight_sampler": "uniform-envelope-rejection-max-weight-one",
        "valid_canonical_edge_count": edges,
        "compact_retained_rows": RETAINED_ROWS,
        "source_representation": "int8-treatment",
    }
    config = {
        "schema": "round0107-diverse-jina-train-config-v1",
        "input": {
            "rows": RETAINED_ROWS,
            "dimension": DIMENSION,
            "representation": "signed-int8-plus-exact-fp16-row-scale",
            "compact_mapping": graph_manifest["compact_mapping"],
        },
        "graph": {
            "manifest": dict(graph_signature),
            "outputs": graph_manifest["outputs"],
            "directed_edges": edges,
            "n_neighbors_including_self": N_NEIGHBORS,
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
            "batch_size": BATCH_SIZE,
            "positive_ratio": POSITIVE_RATIO,
            "positive_rows_per_update": POSITIVE_ROWS_PER_UPDATE,
            "positive_target_mode": "binary",
            "weighted_edge_sampling": True,
            "correlation_weight": 0.0,
            "clip_grad_norm": 1.0,
            "use_amp": "bf16",
            "schedule": "cosine-v3-positive-budget",
            "warmup_successful_updates": PERFORMANCE_WARMUP_UPDATES,
            "successful_positive_lr_updates": updates,
            "update_rule": "ceil(directed_fuzzy_edges/409)",
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
            "performance_windows": performance_windows(updates),
            "expected_pipeline_stamp": expected_pipeline,
        },
    }
    return config, sha256_bytes(canonical_json(config))


_DYNAMIC_PIPELINE_COUNTERS = (
    "endpoint_gather_calls",
    "source_rows_gathered",
    "destination_rows_gathered",
    "host_prefetch_batches_filled",
    "host_prefetch_producer_batches",
    "host_prefetch_consumer_batches",
    "host_prefetch_source_rows_filled",
    "host_prefetch_destination_rows_filled",
    "weight_proposals",
    "weight_acceptances",
    "weight_emitted_draws",
    "weight_buffered_draws",
    "weight_acceptance_rate",
    "weight_rejection_iterations",
)


def synchronize_runtime_counters(
    accounting: dict[str, Any], runtime: Mapping[str, Any]
) -> None:
    for key in _DYNAMIC_PIPELINE_COUNTERS:
        flattened = f"pipeline_{key}"
        if key not in runtime or flattened not in accounting:
            raise Round0107Error(f"missing runtime accounting field {key}")
        accounting[flattened] = runtime[key]
