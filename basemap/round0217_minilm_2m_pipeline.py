"""Host-fp32 endpoint pipeline for the R0217 MiniLM 2M seed family.

The registered sampler is R0107's reviewed uniform-envelope rejection sampler,
unchanged: fuzzy weights live in `(0, 1]`, so accepting a uniformly proposed
edge with probability `w` samples exactly proportional to `w`. Only two things
are specialised here.

1. **Residency.** The R0113/R0166 endpoint array is a 768-dim *fp16* view of a
   compact jina matrix. R0216's substrate is 384-dim **fp32** and already
   L2-normalized, so it is served directly, lazily, from a memmap: 2M x 384 x 4
   is 3.07 GB, which the >= 2 GB rule says must never be materialised into an
   anonymous buffer. Resident memory is page cache plus two pinned 8192-row
   slots (~100 MB).

2. **RNG streams.** The negative stream is drawn from its own generator, seeded
   at `seed + NEGATIVE_RNG_SEED_OFFSET`, exactly as R0113 registered — so a
   seed replay perturbs both streams coherently and neither stream's state
   depends on how many rejection proposals the other consumed.

The sampler is constructed with R0107's registered pipeline labels so its
`_validate_legacy_or_r0132_variant` admission check sees the surface it knows;
the MiniLM-specific labels are applied in `execution_stamp`, which is what the
train receipt and the config's `expected_pipeline_stamp` compare against. That
is the same split R0113's `PromptWeightedJinaSampler` uses.
"""
from __future__ import annotations

import os
import threading
from collections.abc import Mapping
from typing import Any

import numpy as np

from .round0107_training import DiverseWeightedJinaSampler
from .round0217_minilm_2m_seed_family import (
    DIMENSION,
    FEATURE_RESIDENCY,
    GRAPH_K,
    MULTIPLICITY_POLICY,
    NEGATIVE_RNG_SEED_OFFSET,
    PIPELINE,
    PIPELINE_SCHEMA,
    POSITIVE_DESTINATION_POLICY,
    POSITIVE_TARGET_MODE,
    ROWS,
    Round0217Error,
    SAMPLER_CLASS,
)


class MiniLMHostFp32EndpointArray:
    """Pinned two-slot random endpoint gather over the sealed fp32 substrate."""

    round0034_host_int8 = True

    def __init__(
        self,
        source: np.ndarray,
        *,
        source_signature: Mapping[str, Any],
        buffer_rows: int,
        device: str = "cuda",
    ) -> None:
        import torch

        if (
            getattr(source, "ndim", 0) != 2
            or source.shape[1] != DIMENSION
            or source.dtype != np.float32
            or int(buffer_rows) <= 0
        ):
            raise Round0217Error("R0217 fp32 endpoint source is invalid")
        self.source = source
        self.shape = tuple(int(value) for value in source.shape)
        self.dtype = source.dtype
        self.device = device
        self.source_signature = dict(source_signature)
        self.buffer_rows = int(buffer_rows)
        self.endpoint_gather_calls = 0
        self.source_rows_gathered = 0
        self.destination_rows_gathered = 0
        self.host_prefetch_batches_filled = 0
        self.host_prefetch_source_rows_filled = 0
        self.host_prefetch_destination_rows_filled = 0
        self._lock = threading.Lock()
        pin = "cuda" in device
        self._slots: list[dict[str, Any]] = []
        for _ in range(2):
            left = torch.empty(
                (self.buffer_rows, DIMENSION), dtype=torch.float32, pin_memory=pin
            )
            self._slots.append({
                "source": left,
                "destination": torch.empty_like(left, pin_memory=pin),
                "event": None,
            })
        self._slot_index = 0

    def __len__(self) -> int:
        return int(self.shape[0])

    def to(self, _device: str) -> "MiniLMHostFp32EndpointArray":
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
            or np.any(left >= len(self))
            or np.any(right < 0)
            or np.any(right >= len(self))
        ):
            raise Round0217Error("R0217 endpoint row request is invalid")
        return left, right

    def fill_pair_slot(
        self, slot_index: int, source_rows: Any, destination_rows: Any
    ) -> int:
        left, right = self._rows(source_rows, destination_rows)
        if not 0 <= slot_index < len(self._slots):
            raise Round0217Error("R0217 endpoint slot is invalid")
        slot = self._slots[slot_index]
        if slot["event"] is not None:
            slot["event"].synchronize()
        count = len(left)
        slot["source"].numpy()[:count] = self.source[left]
        slot["destination"].numpy()[:count] = self.source[right]
        with self._lock:
            self.host_prefetch_batches_filled += 1
            self.host_prefetch_source_rows_filled += count
            self.host_prefetch_destination_rows_filled += count
        return count

    def transfer_pair_slot(self, slot_index: int, count: int):
        import torch

        if not 0 <= slot_index < len(self._slots) or not 0 <= count <= self.buffer_rows:
            raise Round0217Error("R0217 endpoint transfer is invalid")
        slot = self._slots[slot_index]
        left = slot["source"][:count].to(self.device, non_blocking=True)
        right = slot["destination"][:count].to(self.device, non_blocking=True)
        if "cuda" in self.device:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            slot["event"] = event
        else:
            slot["event"] = None
        with self._lock:
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
            "source_representation": "minilm-mixed-2m-fp32",
            "feature_residency": FEATURE_RESIDENCY,
            "device_conversion": "device-fp32-from-exact-fp32",
            "source": self.source_signature,
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


class MiniLMMixedWeightedSampler(DiverseWeightedJinaSampler):
    """R0107's rejection sampler with R0113's separate negative RNG stream."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        random_state = int(kwargs["random_state"])
        super().__init__(*args, **kwargs)
        self._positive_rng_seed = random_state
        self._negative_rng_seed = random_state + NEGATIVE_RNG_SEED_OFFSET
        self._negative_rng = np.random.default_rng(self._negative_rng_seed)

    def _rows(self) -> tuple[np.ndarray, np.ndarray]:
        edge_ids = self._draw_weighted_edge_ids(self.num_pos)
        source = np.asarray(self.sources[edge_ids], dtype=np.int64)
        destination = np.asarray(self.targets[edge_ids], dtype=np.int64)
        negative_source = self._negative_rng.integers(
            0, self.n_nodes, size=self.num_neg, dtype=np.int64
        )
        offset = self._negative_rng.integers(
            1, self.n_nodes, size=self.num_neg, dtype=np.int64
        )
        return (
            np.concatenate((source, negative_source)),
            np.concatenate(
                (destination, (negative_source + offset) % self.n_nodes)
            ),
        )

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
            "positive_destination_policy": POSITIVE_DESTINATION_POLICY,
            "negative_sampling": f"uniform-{self.n_nodes}-substrate-rows-nonself",
            "rng_stream_policy": (
                "separate-positive-rejection-and-negative-pair-streams"
            ),
            "positive_rng_seed": self._positive_rng_seed,
            "negative_rng_seed": self._negative_rng_seed,
            "graph_degree": f"variable-symmetric-fuzzy-k{GRAPH_K}-topology",
            "graph_search_neighbors_including_self": GRAPH_K + 1,
            "graph_nonself_degree": GRAPH_K,
            "host_prefetch": "single-producer-two-pinned-slot",
            "endpoint_forward": "fused-source-destination",
            "weighted_requested": True,
            "weighted_effective": True,
            "uniform_with_replacement": False,
            "positive_with_replacement": True,
            "weight_sampler": "uniform-envelope-rejection-max-weight-one",
            "weight_uniform_dtype": np.dtype("float64").str,
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
            "host_prefetch_producer_batches": self._producer_batches,
            "host_prefetch_consumer_batches": self._consumer_batches,
            "valid_canonical_edge_count": self.n_pos,
            "compact_retained_rows": self.n_nodes,
            "multiplicity_policy": MULTIPLICITY_POLICY,
            "graph": self.graph_signatures,
            **dataset,
        }


class MiniLMMixedTrainingInput:
    """ParametricUMAP adapter over the sealed R0216 substrate and graph."""

    round0034_host_int8 = True

    def __init__(
        self,
        dataset: MiniLMHostFp32EndpointArray,
        graph: Mapping[str, Any],
        *,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.graph = dict(graph)
        self.seed = int(seed)
        self.shape = dataset.shape
        self._last_sampler: MiniLMMixedWeightedSampler | None = None
        if (
            self.shape != (ROWS, DIMENSION)
            or int(self.graph.get("n_nodes", -1)) != len(dataset)
        ):
            raise Round0217Error("R0217 training input geometry changed")

    def __len__(self) -> int:
        return len(self.dataset)

    def to(self, _device: str) -> "MiniLMMixedTrainingInput":
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
            or positive_target_mode != POSITIVE_TARGET_MODE
            or not weighted_edge_sampling
            or reject_neighbors
            or required_input_pipeline != PIPELINE
            or int(random_state) != self.seed
        ):
            raise Round0217Error("R0217 trainer pipeline request changed")
        sampler = MiniLMMixedWeightedSampler(
            self.dataset,
            sources=self.graph["sources"],
            targets=self.graph["targets"],
            weights=self.graph["weights"],
            n_nodes=self.graph["n_nodes"],
            batch_size=batch_size,
            pos_ratio=pos_ratio,
            random_state=random_state,
            graph_signatures={
                "graph": self.graph["signature"],
                "manifest": self.graph["manifest_signature"],
            },
        )
        self._last_sampler = sampler
        runtime = sampler.execution_stamp()
        return (
            self,
            sampler,
            sampler.n_pos,
            runtime,
            {
                "graph": self.graph["signature"],
                "graph_manifest": self.graph["manifest_signature"],
                "source_representation": runtime["source_representation"],
            },
        )

    def runtime_stamp(self) -> dict[str, Any]:
        if self._last_sampler is None:
            raise Round0217Error("R0217 sampler has not been constructed")
        return self._last_sampler.execution_stamp()


__all__ = [
    "MiniLMHostFp32EndpointArray",
    "MiniLMMixedTrainingInput",
    "MiniLMMixedWeightedSampler",
]
