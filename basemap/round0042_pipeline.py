"""Device-resident fp16 canonical graph training primitives for Round 0042."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np


class Round0042PipelineError(RuntimeError):
    """The registered 30M canonical training contract was violated."""


class DeviceCanonicalSampler:
    """Uniform positive-source, uniform valid canonical-destination sampler.

    The compact source and negative-node universes plus the 30M x 15 target
    matrix live on the device.  Positive source exposure is therefore exactly
    independent of post-canonicalization degree, unlike uniform directed-edge
    sampling on a variable-degree graph.
    """

    def __init__(
        self,
        dataset: Any,
        *,
        targets: np.ndarray,
        degrees: np.ndarray,
        excluded_rows: np.ndarray,
        positive_source_count: int,
        valid_edge_count: int,
        batch_size: int,
        pos_ratio: float,
        random_state: int,
        graph_signature: Mapping[str, Any],
        eligibility_signature: Mapping[str, Any],
        device: str = "cuda",
        upload_chunk: int = 1_000_000,
    ) -> None:
        import torch

        self.dataset = dataset
        self.device = str(device)
        self.n_nodes = int(len(dataset))
        self.k = int(targets.shape[1]) if targets.ndim == 2 else -1
        self.batch_size = int(batch_size)
        self.num_pos = max(1, int(batch_size * pos_ratio))
        self.num_neg = self.batch_size - self.num_pos
        self.n_pos = int(valid_edge_count)
        self.source_n_pos = self.n_pos
        self.positive_source_count = int(positive_source_count)
        self.uniform_with_replacement = True
        self.perm = None
        self.graph_signature = dict(graph_signature)
        self.eligibility_signature = dict(eligibility_signature)
        excluded = np.asarray(excluded_rows, dtype=np.int64)
        if (
            targets.shape != (self.n_nodes, 15)
            or targets.dtype != np.dtype("<i4")
            or degrees.shape != (self.n_nodes,)
            or degrees.dtype != np.dtype("u1")
            or excluded.ndim != 1
            or not np.array_equal(excluded, np.unique(excluded))
            or (len(excluded) and (
                excluded[0] < 0 or excluded[-1] >= self.n_nodes
            ))
            or self.positive_source_count <= 1
            or self.n_pos < self.positive_source_count
            or int(np.asarray(degrees).sum(dtype=np.int64)) != self.n_pos
            or not 0 < pos_ratio < 1
            or self.num_neg <= 0
            or upload_chunk <= 0
        ):
            raise Round0042PipelineError(
                "canonical sampler geometry is invalid"
            )
        positive_sources = np.flatnonzero(
            np.asarray(degrees) > 0
        ).astype(np.int32, copy=False)
        if (
            len(positive_sources) != self.positive_source_count
            or np.intersect1d(positive_sources, excluded).size
            or np.any(np.asarray(degrees)[excluded] != 0)
        ):
            raise Round0042PipelineError(
                "canonical positive-source universe is invalid"
            )
        allowed = np.ones(self.n_nodes, dtype=bool)
        allowed[excluded] = False
        retained = np.flatnonzero(allowed).astype(np.int32, copy=False)
        del allowed
        if len(retained) < 2:
            raise Round0042PipelineError(
                "canonical retained universe is too small"
            )

        # Chunked copies avoid materializing another 1.8 GB host target array.
        self.targets_t = torch.empty(
            targets.shape, dtype=torch.int32, device=self.device
        )
        for start in range(0, self.n_nodes, int(upload_chunk)):
            stop = min(start + int(upload_chunk), self.n_nodes)
            block = np.asarray(targets[start:stop], dtype=np.int32)
            self.targets_t[start:stop].copy_(torch.from_numpy(block))
        self.degrees_t = torch.as_tensor(
            np.asarray(degrees, dtype=np.uint8),
            dtype=torch.uint8,
            device=self.device,
        )
        self.positive_source_rows_t = torch.as_tensor(
            positive_sources, dtype=torch.int32, device=self.device
        )
        self.retained_rows_t = torch.as_tensor(
            retained, dtype=torch.int32, device=self.device
        )
        self._labels = torch.cat((
            torch.ones(
                self.num_pos, dtype=torch.float32, device=self.device
            ),
            torch.zeros(
                self.num_neg, dtype=torch.float32, device=self.device
            ),
        ))
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(int(random_state))
        self.batch_no = 0

    def __len__(self) -> int:
        return int(np.ceil(self.n_pos / self.num_pos))

    def __iter__(self) -> "DeviceCanonicalSampler":
        self.batch_no = 0
        self.perm = None
        return self

    def _draw_slots(self, degrees):
        """Draw one uniform slot per row without a device synchronization.

        ``torch.rand`` is uniform on ``[0, 1)``; flooring after multiplying by
        each small integer degree gives the intended categorical law.  This
        avoids a rejection loop whose ``any().item()`` would serialize every
        training step against the host.
        """
        import torch

        return torch.floor(
            torch.rand(
                len(degrees),
                generator=self.gen,
                device=self.device,
                dtype=torch.float32,
            )
            * degrees.to(dtype=torch.float32)
        ).long()

    def _draw_positive_pairs(self, count: int):
        import torch

        positions = torch.randint(
            0,
            len(self.positive_source_rows_t),
            (int(count),),
            generator=self.gen,
            device=self.device,
        )
        source = self.positive_source_rows_t.index_select(
            0, positions
        ).long()
        degrees = self.degrees_t.index_select(0, source).long()
        slots = self._draw_slots(degrees)
        destination = self.targets_t[source, slots].long()
        return source, destination

    def _draw_negative_pairs(self, count: int):
        import torch

        universe = len(self.retained_rows_t)
        source_position = torch.randint(
            0,
            universe,
            (int(count),),
            generator=self.gen,
            device=self.device,
        )
        offset = torch.randint(
            1,
            universe,
            (int(count),),
            generator=self.gen,
            device=self.device,
        )
        destination_position = (source_position + offset) % universe
        return (
            self.retained_rows_t.index_select(
                0, source_position
            ).long(),
            self.retained_rows_t.index_select(
                0, destination_position
            ).long(),
        )

    def __next__(self):
        import torch

        if self.batch_no >= len(self):
            raise StopIteration
        self.batch_no += 1
        positive_source, positive_destination = self._draw_positive_pairs(
            self.num_pos
        )
        negative_source, negative_destination = self._draw_negative_pairs(
            self.num_neg
        )
        labels = self._labels
        source = torch.cat((positive_source, negative_source))
        destination = torch.cat((
            positive_destination, negative_destination
        ))
        return (
            self.dataset.index_select(source),
            self.dataset.index_select(destination),
            labels,
        )

    def execution_stamp(self) -> dict[str, Any]:
        return {
            "schema": "round0042-device-fp16-canonical-pipeline-v1",
            "pipeline": "device_fp16_canonical",
            "sampler_class": "DeviceCanonicalSampler",
            "x_residency": "device_fp16",
            "positive_sampling": (
                "uniform-retained-positive-source-then-uniform-valid-"
                "canonical-destination-with-replacement"
            ),
            "positive_source_count": self.positive_source_count,
            "valid_canonical_edge_count": self.n_pos,
            "graph_degree": (
                "variable-1-through-15;zero-degree-sources-excluded"
            ),
            "positive_destination_policy": (
                "R0020-duplicate-to-representative;"
                "zero-self-repeated-dropped"
            ),
            "negative_sampling": "uniform-R0020-retained-rows-nonself",
            "graph_manifest": self.graph_signature,
            "eligibility": self.eligibility_signature,
            "uniform_with_replacement": True,
            "positive_with_replacement": True,
            "weighted_requested": False,
            "weighted_effective": False,
        }


class Round0042TrainingInput:
    """Explicit adapter from the accepted 30M pack to the canonical sampler."""

    round0042_device_canonical = True
    expected_shape = (30_000_000, 384)

    def __init__(
        self,
        dataset: Any,
        *,
        graph: Mapping[str, Any],
        excluded_rows: np.ndarray,
        feature_signature: Mapping[str, Any],
    ) -> None:
        self.dataset = dataset
        self.graph = dict(graph)
        self.excluded_rows = np.asarray(excluded_rows, dtype=np.int64)
        self.feature_signature = dict(feature_signature)
        self.shape = dataset.tensor.shape
        self._last_sampler: DeviceCanonicalSampler | None = None
        manifest = self.graph.get("manifest") or {}
        summary = manifest.get("summary") or {}
        degrees = self.graph.get("degrees")
        if (
            tuple(self.shape) != tuple(self.expected_shape)
            or not isinstance(degrees, np.ndarray)
            or degrees.shape != (int(self.expected_shape[0]),)
            or self.excluded_rows.ndim != 1
            or not np.array_equal(
                self.excluded_rows, np.unique(self.excluded_rows)
            )
            or np.any(degrees[self.excluded_rows] != 0)
            or int(summary.get("eligibility_excluded_source_count", -1))
            != len(self.excluded_rows)
            or int(summary.get("eligibility_retained_row_count", -1))
            != int(self.expected_shape[0]) - len(self.excluded_rows)
        ):
            raise Round0042PipelineError(
                "canonical graph, feature matrix, and eligibility differ"
            )

    def __len__(self) -> int:
        return int(self.shape[0])

    def to(self, _device: str) -> "Round0042TrainingInput":
        return self

    def index_select(self, rows: Any):
        return self.dataset.index_select(rows)

    def prepare_round0042_training(
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
    ) -> tuple[
        "Round0042TrainingInput",
        DeviceCanonicalSampler,
        int,
        dict[str, Any],
        dict[str, Any],
    ]:
        signature = self.graph["signature"]
        manifest = self.graph["manifest"]
        summary = manifest["summary"]
        if (
            edges_path != signature["canonical_path"]
            or positive_target_mode != "binary"
            or weighted_edge_sampling
            or reject_neighbors
            or required_input_pipeline != "device_fp16_canonical"
        ):
            raise Round0042PipelineError(
                "R0042 requires the exact binary device-fp16 canonical path"
            )
        sampler = DeviceCanonicalSampler(
            self.dataset,
            targets=self.graph["targets"],
            degrees=self.graph["degrees"],
            excluded_rows=self.excluded_rows,
            positive_source_count=summary[
                "retained_positive_source_count"
            ],
            valid_edge_count=summary["valid_canonical_edge_count"],
            batch_size=batch_size,
            pos_ratio=pos_ratio,
            random_state=random_state,
            graph_signature=signature,
            eligibility_signature=manifest["inputs"]["eligibility"],
            device=str(self.dataset.device),
        )
        self._last_sampler = sampler
        return (
            self,
            sampler,
            sampler.n_pos,
            sampler.execution_stamp(),
            {
                "canonical_graph_manifest": signature,
                "canonical_targets": manifest["outputs"]["targets"],
                "canonical_degrees": manifest["outputs"]["degrees"],
                "eligibility": manifest["inputs"]["eligibility"],
                "features": self.feature_signature,
            },
        )

    def runtime_stamp(self) -> dict[str, Any]:
        if self._last_sampler is None:
            raise Round0042PipelineError("R0042 sampler was not constructed")
        return self._last_sampler.execution_stamp()
