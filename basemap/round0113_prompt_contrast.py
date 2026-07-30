"""Paired prompt-map contract for Round 0113."""
from __future__ import annotations

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
    sha256_bytes,
)
from .round0107_training import DiverseWeightedJinaSampler
from .round0112_prompt_substrate import (
    CONVENTIONS,
    DIMENSION,
    ELIGIBILITY_PATH,
    ELIGIBILITY_SHA256,
    ROWS,
    SUBSTRATE_SCHEMA,
    TEXT_ROOT,
)
from .round0114_prompt_recovery import RECOVERY_SCHEMA
from .round0104_training import source_segments


ROUND_ID = "0113"
ARMS = CONVENTIONS
SEED = 42
NEGATIVE_RNG_SEED_OFFSET = 11_300_000
QUERY_CANDIDATES = 4_096
QUERY_ROWS = 2_000
QUERY_SCAN_START = ROWS
QUERY_SCAN_LIMIT = ROWS + 25_000
POLISH_SOURCE_ROWS = 2_000_000
POLISH_QUERY_ROWS = 500
POLISH_QUERY_SEED = 127
POLISH_TEXT_PATH = (
    "/data/chunks/fineweb2-pol_Latn-chunked-500/train/000_00000.parquet"
)
POLISH_HISTORICAL_EMBEDDING_PATH = (
    "/data/embeddings/fineweb2-pol_Latn-chunked-500-jina-v5-nano/"
    "train/000_00000.npy"
)
POLISH_HISTORICAL_EMBEDDING_SHA256 = (
    "fac550b9a21409da7372c9f876761e059ea99f653db95e885f998f31e265b62f"
)
POLISH_HISTORICAL_MANIFEST_PATH = (
    "/data/embeddings/fineweb2-pol_Latn-chunked-500-jina-v5-nano/"
    "manifest.json"
)
POLISH_QUERY_ROWS_SHA256 = (
    "ae06ba5dd3e5ce3b1aafd18604b80c8d8575ea45a367f68871be6c80a99aa36b"
)

GRAPH_K = 50
GRAPH_NLIST = 8_192
GRAPH_TRAIN_ROWS = 262_144
GRAPH_TRAIN_SEED = 113
GRAPH_QUALITY_ROWS = 4_096
GRAPH_QUALITY_SEED = 114
GRAPH_NPROBE_GRID = (16, 32, 64, 128, 256)
GRAPH_NPROBE = 64
GRAPH_MEAN_RECALL_FLOOR = 0.90
GRAPH_P10_RECALL_FLOOR = 0.80

PANEL_ANCHORS = 4_000
PANEL_SEED = 123
BATCH_SIZE = 8_192
POSITIVE_RATIO = 0.05
POSITIVE_ROWS_PER_UPDATE = int(BATCH_SIZE * POSITIVE_RATIO)
SUCCESSFUL_UPDATES = 500_000
PERFORMANCE_WARMUP_UPDATES = 200
PERFORMANCE_WINDOW_UPDATES = 2_500
PERFORMANCE_WINDOWS = math.ceil(
    (SUCCESSFUL_UPDATES - PERFORMANCE_WARMUP_UPDATES)
    / PERFORMANCE_WINDOW_UPDATES
)
TRAIN_MINIMUM_UPDATES_PER_S = 70.0
TRAIN_WARNING_UPDATES_PER_S = 80.0

PIPELINE = "host_weighted_jina_prompt_contrast"
PIPELINE_SCHEMA = "round0113-host-weighted-jina-prompt-pipeline-v1"
SAMPLER_CLASS = "PromptWeightedJinaSampler"
GRAPH_SCHEMA = "round0113-prompt-arm-fuzzy-graph-v1"
ASSEMBLY_SCHEMA = "round0113-compact-prompt-arrays-v1"
QUERY_SCHEMA = "round0113-dual-prompt-query-reserve-v1"
QUERY_SELECTION_SCHEMA = "round0113-matched-query-selection-v1"
BASELINE_EXCLUDED_ROWS = 5_366
BASELINE_RETAINED_ROWS = ROWS - BASELINE_EXCLUDED_ROWS
# Finalize both values from the complete recovered source/raw/document exact-family
# union before issuance. The runtime census must reproduce this exact ordered
# selector identity without carrying a long row list in executable source.
PROMPT_UNION_EXTRA_EXCLUDED_ROWS = 873
PROMPT_UNION_EXTRA_EXCLUSIONS_SHA256 = (
    "aa6f446263e221ac6b127670fd3e6d2912fc481cf366146746a909ec5757d36a"
)
EXCLUDED_ROWS = BASELINE_EXCLUDED_ROWS + PROMPT_UNION_EXTRA_EXCLUDED_ROWS
RETAINED_ROWS = ROWS - EXCLUDED_ROWS

NONINFERIORITY_RATIO = 0.97
DECISION_METRICS = (
    "ffr",
    "density",
    "recall_at_10",
    "oos_recall_at_10",
    "oos_recall_at_50",
)


class Round0113Error(RuntimeError):
    """The registered R0113 prompt contrast was violated."""


def seal(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    return {**value, "identity_sha256": sha256_bytes(canonical_json(value))}


def validate_seal(value: Mapping[str, Any], *, label: str) -> None:
    body = {key: item for key, item in value.items() if key != "identity_sha256"}
    if value.get("identity_sha256") != sha256_bytes(canonical_json(body)):
        raise Round0113Error(f"{label} identity seal is invalid")


def verify_signature(signature: Any, *, label: str) -> str:
    if not isinstance(signature, Mapping):
        raise Round0113Error(f"{label} signature is missing")
    path = str(signature.get("canonical_path") or "")
    if not path or expected_input_signature(path) != dict(signature):
        raise Round0113Error(f"{label} content changed")
    return path


def read_sealed(path: str, *, label: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    validate_seal(value, label=label)
    return value


def load_substrate_manifest(
    path: str,
    *,
    expected_sha256: str,
    verify_chunks: bool = False,
) -> dict[str, Any]:
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise Round0113Error("paired substrate manifest bytes changed")
    manifest = read_sealed(path, label="reviewed dual-prompt substrate")
    conventions = manifest.get("conventions") or {}
    duplicate = manifest.get("duplicate_control") or {}
    schema_and_round = (manifest.get("schema"), manifest.get("round_id"))
    if (
        schema_and_round
        not in {
            (SUBSTRATE_SCHEMA, "0112"),
            (RECOVERY_SCHEMA, "0114"),
        }
        or int(manifest.get("row_count", -1)) != ROWS
        or int(manifest.get("dimension", -1)) != DIMENSION
        or set(conventions) != set(ARMS)
        or any(len((conventions[arm] or {}).get("chunks") or []) != 80 for arm in ARMS)
        or int(duplicate.get("excluded_exact_copy_rows", -1))
        != BASELINE_EXCLUDED_ROWS
        or int(duplicate.get("retained_representative_rows", -1))
        != BASELINE_RETAINED_ROWS
        or (duplicate.get("cohort_reconciliation") or {}).get(
            "outside_representative_rows_restored"
        )
        != 11
    ):
        raise Round0113Error("paired substrate contract changed")
    selector = dict(duplicate.get("selector") or {})
    selector_path = verify_signature(
        selector, label="paired cohort-local exclusion selector"
    )
    excluded = np.load(selector_path, mmap_mode="r", allow_pickle=False)
    if (
        excluded.shape != (BASELINE_EXCLUDED_ROWS,)
        or excluded.dtype != np.int64
        or excluded[0] < 0
        or excluded[-1] >= ROWS
        or np.any(excluded[1:] <= excluded[:-1])
    ):
        raise Round0113Error("paired cohort-local selector bytes changed")
    chunk_signatures: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        values = [dict(item) for item in conventions[arm]["chunks"]]
        for item in values:
            path_value = str(item.get("canonical_path") or "")
            if (
                not os.path.isfile(path_value)
                or os.path.getsize(path_value) != int(item.get("bytes", -1))
            ):
                raise Round0113Error(f"paired {arm} chunk missing/wrong size")
            if verify_chunks and expected_input_signature(path_value) != item:
                raise Round0113Error(f"paired {arm} chunk content changed")
        chunk_signatures[arm] = values
    return {
        "manifest": manifest,
        "signature": signature,
        "excluded": excluded,
        "selector": selector,
        "chunks": chunk_signatures,
    }


def baseline_compact_mapping(excluded: np.ndarray) -> np.ndarray:
    dropped = np.asarray(excluded, dtype=np.int64)
    if (
        dropped.shape != (BASELINE_EXCLUDED_ROWS,)
        or np.any(dropped[1:] <= dropped[:-1])
    ):
        raise Round0113Error("R0113 compact selector is malformed")
    keep = np.ones(ROWS, dtype=bool)
    keep[dropped] = False
    mapping = np.flatnonzero(keep).astype(np.int64)
    if mapping.shape != (BASELINE_RETAINED_ROWS,):
        raise Round0113Error("R0113 baseline compact mapping did not close")
    return mapping


def compact_mapping(
    excluded: np.ndarray,
    prompt_union_extra: np.ndarray,
) -> np.ndarray:
    baseline = baseline_compact_mapping(excluded)
    dropped = np.asarray(excluded, dtype=np.int64)
    extra = np.asarray(prompt_union_extra, dtype=np.int64)
    from .artifact_identity import ordered_array_sha256

    if (
        extra.shape != (PROMPT_UNION_EXTRA_EXCLUDED_ROWS,)
        or (len(extra) and np.any(extra[1:] <= extra[:-1]))
        or (len(extra) and (extra[0] < 0 or extra[-1] >= ROWS))
        or np.intersect1d(dropped, extra, assume_unique=True).size
        or ordered_array_sha256(extra)
        != PROMPT_UNION_EXTRA_EXCLUSIONS_SHA256
    ):
        raise Round0113Error("R0113 prompt-union selector is malformed")
    mapping = baseline[~np.isin(baseline, extra, assume_unique=True)]
    if (
        mapping.shape != (RETAINED_ROWS,)
        or np.intersect1d(mapping, extra, assume_unique=True).size
    ):
        raise Round0113Error("R0113 compact mapping did not close")
    return mapping


def query_candidate_rows() -> tuple[np.ndarray, dict[str, Any]]:
    """Select a fixed historical-family-clean reserve before fresh embedding."""
    signature = expected_input_signature(ELIGIBILITY_PATH)
    if signature["sha256"] != ELIGIBILITY_SHA256:
        raise Round0113Error("R0087 family table bytes changed")
    with np.load(ELIGIBILITY_PATH, allow_pickle=False) as archive:
        representatives = np.asarray(
            archive["representative_rows"], dtype=np.int64
        )
        offsets = np.asarray(archive["family_offsets"], dtype=np.int64)
        members = np.asarray(archive["member_rows"], dtype=np.int64)
    family_by_row: dict[int, tuple[int, bool]] = {}
    for family_index, representative in enumerate(representatives.tolist()):
        family = members[offsets[family_index] : offsets[family_index + 1]]
        candidates = family[
            (family >= QUERY_SCAN_START) & (family < QUERY_SCAN_LIMIT)
        ]
        if not len(candidates):
            continue
        touches_training = bool(np.any(family < ROWS))
        for row in candidates.tolist():
            family_by_row[int(row)] = (int(representative), touches_training)
    selected: list[int] = []
    used_families: set[Any] = set()
    rejected_training_copy = 0
    rejected_reserve_copy = 0
    for row in range(QUERY_SCAN_START, QUERY_SCAN_LIMIT):
        family = family_by_row.get(row)
        if family is not None and family[1]:
            rejected_training_copy += 1
            continue
        key: Any = family[0] if family is not None else ("singleton", row)
        if key in used_families:
            rejected_reserve_copy += 1
            continue
        used_families.add(key)
        selected.append(row)
        if len(selected) == QUERY_CANDIDATES:
            break
    rows = np.asarray(selected, dtype=np.int64)
    if (
        rows.shape != (QUERY_CANDIDATES,)
        or np.any(rows[1:] <= rows[:-1])
        or rows[0] < QUERY_SCAN_START
        or rows[-1] >= QUERY_SCAN_LIMIT
    ):
        raise Round0113Error("R0113 clean query reserve is exhausted")
    return rows, {
        "selector": (
            "ascending rows after training prefix; reject any R0087 exact "
            "family touching training and retain one row per reserve family"
        ),
        "scan_range": [QUERY_SCAN_START, QUERY_SCAN_LIMIT],
        "selected_rows": QUERY_CANDIDATES,
        "last_selected_global_row": int(rows[-1]),
        "rejected_training_family_rows": rejected_training_copy,
        "rejected_reserve_family_rows": rejected_reserve_copy,
        "eligibility": signature,
    }


def query_source_layout(rows: np.ndarray) -> list[dict[str, Any]]:
    """Bind only the source shards that contain the fixed query reserve."""
    import pyarrow.parquet as pq

    requested = np.asarray(rows, dtype=np.int64)
    if (
        requested.shape != (QUERY_CANDIDATES,)
        or np.any(requested[1:] <= requested[:-1])
    ):
        raise Round0113Error("R0113 query source rows are malformed")
    segments = source_segments(int(requested[0]), int(requested[-1]) + 1)
    layout: list[dict[str, Any]] = []
    for segment in segments:
        embedding = dict(segment["shard"])
        embedding_path = os.path.realpath(str(embedding["canonical_path"]))
        name = os.path.basename(embedding_path)
        if (
            segment.get("dataset")
            != "fineweb-edu-sample-10BT-chunked-500-jina-v5-nano"
            or not name.endswith(".npy")
        ):
            raise Round0113Error("R0113 query source dataset changed")
        text_path = os.path.realpath(
            os.path.join(TEXT_ROOT, name[:-4] + ".parquet")
        )
        if not os.path.isfile(text_path):
            raise Round0113Error("R0113 aligned query text shard is missing")
        text_rows = int(pq.ParquetFile(text_path).metadata.num_rows)
        if text_rows != int(segment["shard_rows"]):
            raise Round0113Error(
                "R0113 query text/embedding shard row counts differ"
            )
        layout.append(
            {
                "global_row_start": int(segment["global_row_start"]),
                "global_row_stop": int(segment["global_row_stop"]),
                "shard_row_start": int(segment["shard_row_start"]),
                "shard_row_stop": int(segment["shard_row_stop"]),
                "shard_rows": int(segment["shard_rows"]),
                "embedding": {
                    "canonical_path": embedding_path,
                    "kind": "file",
                    "bytes": int(embedding["bytes"]),
                    "sha256": str(embedding["sha256"]),
                },
                "text": expected_input_signature(text_path),
            }
        )
    coverage = np.zeros(len(requested), dtype=bool)
    for item in layout:
        coverage |= (
            (requested >= int(item["global_row_start"]))
            & (requested < int(item["global_row_stop"]))
        )
    if not np.all(coverage):
        raise Round0113Error("R0113 query source layout does not cover reserve")
    return layout


def polish_query_rows() -> np.ndarray:
    """Reproduce R0108's fixed 500-query held-out Polish panel."""
    selected = np.random.RandomState(POLISH_QUERY_SEED).choice(
        POLISH_SOURCE_ROWS, size=50_000, replace=False
    ).astype(np.int64)
    rows = np.sort(selected[49_500:])
    from .artifact_identity import ordered_array_sha256

    if (
        rows.shape != (POLISH_QUERY_ROWS,)
        or ordered_array_sha256(rows) != POLISH_QUERY_ROWS_SHA256
    ):
        raise Round0113Error("R0113 Polish query selector changed")
    return rows


class HostFp16EndpointArray:
    """Pinned two-slot random endpoint gather over one compact fp16 memmap."""

    round0034_host_int8 = True

    def __init__(
        self,
        source: np.ndarray,
        *,
        arm: str,
        source_signature: Mapping[str, Any],
        mapping_signature: Mapping[str, Any],
        buffer_rows: int,
        device: str = "cuda",
    ) -> None:
        import torch

        if (
            arm not in ARMS
            or source.ndim != 2
            or source.shape[1] != DIMENSION
            or source.dtype != np.float16
            or buffer_rows <= 0
        ):
            raise Round0113Error("R0113 compact fp16 endpoint source is invalid")
        self.source = source
        self.shape = source.shape
        self.dtype = source.dtype
        self.arm = arm
        self.device = device
        self.source_signature = dict(source_signature)
        self.mapping_signature = dict(mapping_signature)
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

    def to(self, _device: str) -> "HostFp16EndpointArray":
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
            raise Round0113Error("R0113 endpoint row request is invalid")
        return left, right

    def fill_pair_slot(
        self, slot_index: int, source_rows: Any, destination_rows: Any
    ) -> int:
        left, right = self._rows(source_rows, destination_rows)
        if not 0 <= slot_index < len(self._slots):
            raise Round0113Error("R0113 endpoint slot is invalid")
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
            raise Round0113Error("R0113 endpoint transfer is invalid")
        slot = self._slots[slot_index]
        left = slot["source"][:count].to(self.device, non_blocking=True).float()
        right = slot["destination"][:count].to(
            self.device, non_blocking=True
        ).float()
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
            "source_representation": f"{self.arm}-fp16",
            "feature_residency": "host-contiguous-compact-fp16-memmap",
            "device_conversion": "device-fp32-from-exact-fp16",
            "source": self.source_signature,
            "compact_mapping": self.mapping_signature,
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


class PromptWeightedJinaSampler(DiverseWeightedJinaSampler):
    """R0107's exact rejection sampler with prompt-specific stamps."""

    def __init__(self, *args: Any, arm: str, **kwargs: Any) -> None:
        self.arm = arm
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
            "arm": self.arm,
            "positive_sampling": (
                "fuzzy_weight_proportional_with_replacement_via_exact_"
                "uniform_envelope_rejection"
            ),
            "positive_destination_policy": (
                f"separate-{self.arm}-fp16-fuzzy-k50-graph"
            ),
            "negative_sampling": (
                f"uniform-{self.n_nodes}-compact-representatives-nonself"
            ),
            "rng_stream_policy": (
                "separate-positive-rejection-and-negative-pair-streams"
            ),
            "positive_rng_seed": self._positive_rng_seed,
            "negative_rng_seed": self._negative_rng_seed,
            "negative_row_pairs_identical_across_arms": True,
            "graph_degree": "variable-symmetric-fuzzy-k50-topology",
            "host_prefetch": "single-producer-two-pinned-slot",
            "host_prefetch_producer_batches": self._producer_batches,
            "host_prefetch_consumer_batches": self._consumer_batches,
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
            "valid_canonical_edge_count": self.n_pos,
            "compact_retained_rows": self.n_nodes,
            "multiplicity_policy": (
                "shared-source-raw-document-union-representative-only"
            ),
            "graph": self.graph_signatures,
            **dataset,
        }


class PromptTrainingInput:
    """ParametricUMAP adapter for one arm's separate graph."""

    round0034_host_int8 = True

    def __init__(
        self,
        dataset: HostFp16EndpointArray,
        graph: Mapping[str, Any],
        *,
        arm: str,
    ) -> None:
        self.dataset = dataset
        self.graph = dict(graph)
        self.arm = arm
        self.shape = dataset.shape
        self._last_sampler: PromptWeightedJinaSampler | None = None
        if (
            arm not in ARMS
            or self.shape != (RETAINED_ROWS, DIMENSION)
            or int(graph.get("n_nodes", -1)) != len(dataset)
        ):
            raise Round0113Error("R0113 training input geometry changed")

    def __len__(self) -> int:
        return len(self.dataset)

    def to(self, _device: str) -> "PromptTrainingInput":
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
            or required_input_pipeline != PIPELINE
        ):
            raise Round0113Error("R0113 trainer pipeline request changed")
        sampler = PromptWeightedJinaSampler(
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
            arm=self.arm,
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
            raise Round0113Error("R0113 sampler has not been constructed")
        return self._last_sampler.execution_stamp()


def train_config(
    arm: str,
    *,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    graph_edges: int,
    retained_rows: int,
) -> tuple[dict[str, Any], str]:
    if arm not in ARMS or graph_edges <= 0 or retained_rows != RETAINED_ROWS:
        raise Round0113Error("R0113 train config input is invalid")
    expected_pipeline = {
        "schema": PIPELINE_SCHEMA,
        "pipeline": PIPELINE,
        "sampler_class": SAMPLER_CLASS,
        "arm": arm,
        "positive_sampling": (
            "fuzzy_weight_proportional_with_replacement_via_exact_"
            "uniform_envelope_rejection"
        ),
        "positive_destination_policy": (
            f"separate-{arm}-fp16-fuzzy-k50-graph"
        ),
        "negative_sampling": (
            f"uniform-{retained_rows}-compact-representatives-nonself"
        ),
        "rng_stream_policy": (
            "separate-positive-rejection-and-negative-pair-streams"
        ),
        "positive_rng_seed": SEED,
        "negative_rng_seed": SEED + NEGATIVE_RNG_SEED_OFFSET,
        "negative_row_pairs_identical_across_arms": True,
        "graph_degree": "variable-symmetric-fuzzy-k50-topology",
        "host_prefetch": "single-producer-two-pinned-slot",
        "endpoint_forward": "fused-source-destination",
        "weighted_requested": True,
        "weighted_effective": True,
        "uniform_with_replacement": False,
        "positive_with_replacement": True,
        "weight_sampler": "uniform-envelope-rejection-max-weight-one",
        "weight_uniform_dtype": np.dtype("float64").str,
        "valid_canonical_edge_count": graph_edges,
        "compact_retained_rows": retained_rows,
        "multiplicity_policy": (
            "shared-source-raw-document-union-representative-only"
        ),
        "source_representation": f"{arm}-fp16",
    }
    config = {
        "schema": "round0113-prompt-arm-train-config-v1",
        "arm": arm,
        "paired_invariant": {
            "rows": retained_rows,
            "dimension": DIMENSION,
            "seed": SEED,
            "successful_positive_lr_updates": SUCCESSFUL_UPDATES,
            "graph_policy": "separate bytes; identical builder/parameters/seeds",
            "sampler": SAMPLER_CLASS,
        },
        "input": {
            "rows": retained_rows,
            "dimension": DIMENSION,
            "representation": f"fresh-local-{arm}-fp16",
            "multiplicity_policy": (
                "shared-source-raw-document-union-representative-only"
            ),
        },
        "graph": {
            "path": str(graph_signature["canonical_path"]),
            "sha256": str(graph_signature["sha256"]),
            "manifest_path": str(graph_manifest_signature["canonical_path"]),
            "manifest_sha256": str(graph_manifest_signature["sha256"]),
            "k": GRAPH_K,
            "nprobe": GRAPH_NPROBE,
            "directed_edges": graph_edges,
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
            "positive_rng_seed": SEED,
            "negative_rng_seed": SEED + NEGATIVE_RNG_SEED_OFFSET,
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
            raise Round0113Error(f"missing R0113 runtime field {key}")
        accounting[flattened] = runtime[key]


def load_graph(
    manifest_path: str,
    *,
    expected_sha256: str,
    arm: str,
) -> dict[str, Any]:
    signature = expected_input_signature(manifest_path)
    if signature["sha256"] != expected_sha256:
        raise Round0113Error(f"R0113 {arm} graph manifest bytes changed")
    manifest = read_sealed(manifest_path, label=f"R0113 {arm} graph manifest")
    search = manifest.get("search_qualification") or {}
    fixed_cell = (search.get("cells") or {}).get(str(GRAPH_NPROBE)) or {}
    if (
        arm not in ARMS
        or manifest.get("schema") != GRAPH_SCHEMA
        or manifest.get("round_id") != ROUND_ID
        or manifest.get("arm") != arm
        or int(manifest.get("retained_rows", -1)) != RETAINED_ROWS
        or int(manifest.get("dimension", -1)) != DIMENSION
        or int(manifest.get("k", -1)) != GRAPH_K
        or int(manifest.get("directed_edge_count", -1)) <= 0
        or int(search.get("selected_nprobe", -1)) != GRAPH_NPROBE
        or fixed_cell.get("passed") is not True
    ):
        raise Round0113Error(f"R0113 {arm} graph contract changed")
    graph_path = verify_signature(
        manifest.get("graph"), label=f"R0113 {arm} graph"
    )
    from .pumap.parametric_umap.datasets.edge_list_dataset import (
        load_edge_arrays,
    )

    sources, targets, weights, n_nodes = load_edge_arrays(
        graph_path, load_weights=True
    )
    if (
        weights is None
        or int(n_nodes) != RETAINED_ROWS
        or len(sources) != int(manifest["directed_edge_count"])
    ):
        raise Round0113Error(f"R0113 {arm} graph arrays changed")
    return {
        "manifest": manifest,
        "manifest_signature": signature,
        "signature": dict(manifest["graph"]),
        "sources": sources,
        "targets": targets,
        "weights": weights,
        "n_nodes": int(n_nodes),
    }


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


def paired_decision(
    control: Mapping[str, Any],
    treatment: Mapping[str, Any],
) -> dict[str, Any]:
    control_metrics = dict(control.get("metrics") or {})
    treatment_metrics = dict(treatment.get("metrics") or {})
    if (
        set(control_metrics) != set(DECISION_METRICS)
        or set(treatment_metrics) != set(DECISION_METRICS)
    ):
        raise Round0113Error("R0113 paired metric set changed")
    gates: dict[str, Any] = {}
    for metric in DECISION_METRICS:
        baseline = float(control_metrics[metric])
        observed = float(treatment_metrics[metric])
        if not np.isfinite(baseline) or not np.isfinite(observed):
            raise Round0113Error("R0113 paired metric is nonfinite")
        threshold = NONINFERIORITY_RATIO * baseline
        gates[metric] = {
            "raw": baseline,
            "document": observed,
            "document_minus_raw": observed - baseline,
            "threshold": threshold,
            "ratio": observed / baseline if baseline != 0 else None,
            "passed": observed >= threshold,
        }
    execution = {
        arm: all(
            bool(value)
            for value in (report.get("execution_gates") or {}).values()
        )
        for arm, report in (("raw", control), ("document", treatment))
    }
    return {
        "noninferiority_ratio": NONINFERIORITY_RATIO,
        "metric_gates": gates,
        "execution_gates": execution,
        "passed": all(item["passed"] for item in gates.values())
        and all(execution.values()),
        "projection_ffr_role": "diagnostic-only",
        "scope": (
            "one-seed 2M FineWeb prompt-transfer screen; not production "
            "readiness or full-corpus SAE evidence"
        ),
    }
