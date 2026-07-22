"""Canonical 150M graph and host-int8 training pipeline for Round 0034.

The historical 150M graph is a compressed ``npz`` containing redundant
source and constant-weight arrays.  This module validates those arrays by
streaming them from the ZIP members, then publishes only the data needed by
training: one source-major ``int32`` target matrix and one ``uint8`` degree per
row.  Duplicate destinations are mapped to the exact R0033 representative;
zero, self, and repeated canonical destinations are removed.

The training adapter keeps the R0025 int8 matrix and fp16 scales in host RAM.
One producer thread fills two owned pinned slots ahead of CUDA while every
consumed batch gathers *both* endpoint matrices and applies the exact per-row
scale on device. Source and destination rows share one model call because the
registered model has neither batch normalization nor dropout. The adapter is
compatible with the existing fast-loader loop but never uploads the full
feature matrix to CUDA.
"""
from __future__ import annotations

import contextlib
import concurrent.futures
import json
import math
import os
import threading
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import numpy as np

from .artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from .output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    refuse_existing,
)


ELIGIBILITY_SCHEMA = "minilm-int8-row-eligibility-v1"
GRAPH_SCHEMA = "minilm-canonical-source-major-k15-v1"
PIPELINE_SCHEMA = "round0034-host-int8-canonical-pipeline-v1"
CANARY_SCHEMA = "round0034-two-endpoint-no-update-canary-v1"

DEFAULT_ROWS = 150_000_000
DEFAULT_DIMENSION = 384
DEFAULT_K = 15
R0019_RETAINED_POSITIVE_SOURCES = 29_989_838
R0019_SUCCESSFUL_UPDATES = 500_000


class Round0034PipelineError(RuntimeError):
    """Fail-closed R0034 input/pipeline error."""


def coverage_aligned_successful_updates(
    retained_positive_sources: int,
    *,
    reference_sources: int = R0019_RETAINED_POSITIVE_SOURCES,
    reference_updates: int = R0019_SUCCESSFUL_UPDATES,
) -> int:
    """Scale R0019's successful-update horizon by positive-source coverage."""
    values = (retained_positive_sources, reference_sources, reference_updates)
    if any(isinstance(value, bool) or int(value) != value or value <= 0
           for value in values):
        raise ValueError("coverage-aligned update inputs must be positive integers")
    # Integer arithmetic avoids a float rounding error at the ceil boundary.
    return (int(reference_updates) * int(retained_positive_sources)
            + int(reference_sources) - 1) // int(reference_sources)


def _seal(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    return {**value, "identity_sha256": sha256_bytes(canonical_json(value))}


def _sorted_unique_in_range(
    value: Any,
    *,
    name: str,
    row_count: int,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.int64)
    if (
        array.ndim != 1
        or not np.array_equal(array, np.unique(array))
        or (len(array) and (array[0] < 0 or array[-1] >= row_count))
    ):
        raise Round0034PipelineError(
            f"eligibility {name} must be sorted, unique, and in range"
        )
    return array


def validate_eligibility_view(
    eligibility: Mapping[str, Any],
    *,
    row_count: int,
) -> dict[str, Any]:
    """Validate the generic result returned by R0033's released loader.

    Keeping this interface structural lets focused tests use a synthetic
    capability without copying or weakening the released R0033 loader.  The
    production entrypoint calls that loader first, including its full-file hash
    and internal-identity checks, and passes the returned mapping here.
    """
    metadata = eligibility.get("metadata")
    signature = eligibility.get("signature")
    if not isinstance(metadata, Mapping) or not isinstance(signature, Mapping):
        raise Round0034PipelineError("eligibility view lacks metadata/signature")
    if (
        metadata.get("schema") != ELIGIBILITY_SCHEMA
        or int(metadata.get("row_count", -1)) != row_count
        or not isinstance(signature.get("sha256"), str)
        or len(signature["sha256"]) != 64
    ):
        raise Round0034PipelineError("eligibility identity/geometry changed")

    zero = _sorted_unique_in_range(
        eligibility.get("zero_rows", []), name="zero_rows", row_count=row_count
    )
    excluded = _sorted_unique_in_range(
        eligibility.get("excluded_rows", []),
        name="excluded_rows",
        row_count=row_count,
    )
    duplicate = _sorted_unique_in_range(
        eligibility.get("duplicate_excluded_rows", []),
        name="duplicate_excluded_rows",
        row_count=row_count,
    )
    representatives = np.asarray(
        eligibility.get("duplicate_representative_rows", []), dtype=np.int64
    )
    if (
        representatives.shape != duplicate.shape
        or np.any(representatives < 0)
        or np.any(representatives >= row_count)
        or np.intersect1d(representatives, excluded).size
        or np.intersect1d(zero, duplicate).size
        or not np.array_equal(
            excluded, np.sort(np.concatenate((zero, duplicate)))
        )
    ):
        raise Round0034PipelineError(
            "eligibility duplicate-to-representative policy is invalid"
        )
    summary = metadata.get("summary") or {}
    if (
        int(summary.get("excluded_row_count", -1)) != len(excluded)
        or int(summary.get("retained_row_count", -1))
        != row_count - len(excluded)
    ):
        raise Round0034PipelineError("eligibility row accounting changed")
    return {
        "metadata": dict(metadata),
        "signature": dict(signature),
        "zero_rows": zero,
        "excluded_rows": excluded,
        "duplicate_excluded_rows": duplicate,
        "duplicate_representative_rows": representatives,
        "retained_row_count": row_count - len(excluded),
    }


@dataclass(frozen=True)
class NpyMemberHeader:
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    fortran_order: bool

    @property
    def values(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))


def _read_npy_header(handle: Any, name: str) -> NpyMemberHeader:
    from numpy.lib import format as fmt

    version = fmt.read_magic(handle)
    if version == (1, 0):
        shape, fortran_order, dtype = fmt.read_array_header_1_0(handle)
    elif version in {(2, 0), (3, 0)}:
        shape, fortran_order, dtype = fmt.read_array_header_2_0(handle)
    else:
        raise Round0034PipelineError(
            f"unsupported NPY version {version!r} for {name}"
        )
    return NpyMemberHeader(
        name=name,
        shape=tuple(int(value) for value in shape),
        dtype=np.dtype(dtype),
        fortran_order=bool(fortran_order),
    )


@contextlib.contextmanager
def _open_member(
    graph_path: str,
    member: str,
) -> Iterator[tuple[Any, NpyMemberHeader]]:
    with zipfile.ZipFile(graph_path) as archive:
        names = archive.namelist()
        if names.count(member) != 1:
            raise Round0034PipelineError(
                f"graph must contain exactly one {member!r} member"
            )
        with archive.open(member) as handle:
            header = _read_npy_header(handle, member)
            if header.fortran_order:
                raise Round0034PipelineError(
                    f"graph member {member} must be C ordered"
                )
            yield handle, header


def _read_exact(handle: Any, count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = int(count)
    while remaining:
        chunk = handle.read(remaining)
        if not chunk:
            raise Round0034PipelineError("truncated graph NPY member")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _scalar_member(graph_path: str, member: str, dtype: np.dtype) -> int:
    with _open_member(graph_path, member) as (handle, header):
        if header.shape != () or header.dtype != np.dtype(dtype):
            raise Round0034PipelineError(
                f"graph scalar {member} has geometry {header.shape}/{header.dtype}"
            )
        value = np.frombuffer(_read_exact(handle, header.dtype.itemsize),
                              dtype=header.dtype)[0]
        if handle.read(1):
            raise Round0034PipelineError(f"graph scalar {member} has trailing bytes")
        return int(value)


def _member_blocks(
    graph_path: str,
    member: str,
    *,
    dtype: np.dtype,
    values: int,
    block_values: int,
) -> Iterator[tuple[int, np.ndarray]]:
    expected_dtype = np.dtype(dtype)
    with _open_member(graph_path, member) as (handle, header):
        if header.shape != (values,) or header.dtype != expected_dtype:
            raise Round0034PipelineError(
                f"graph member {member} expected {(values,)}/{expected_dtype}, "
                f"got {header.shape}/{header.dtype}"
            )
        cursor = 0
        while cursor < values:
            take = min(block_values, values - cursor)
            payload = _read_exact(handle, take * expected_dtype.itemsize)
            yield cursor, np.frombuffer(payload, dtype=expected_dtype)
            cursor += take
        if handle.read(1):
            raise Round0034PipelineError(f"graph member {member} has trailing bytes")


def _verify_source_major(
    graph_path: str,
    *,
    row_count: int,
    k: int,
    block_rows: int,
) -> None:
    edge_count = row_count * k
    for edge_start, block in _member_blocks(
        graph_path,
        "sources.npy",
        dtype=np.dtype("<i4"),
        values=edge_count,
        block_values=block_rows * k,
    ):
        if edge_start % k or len(block) % k:
            raise Round0034PipelineError("source validation block lost k alignment")
        rows = len(block) // k
        first_row = edge_start // k
        expected = np.arange(first_row, first_row + rows, dtype=np.int32)
        matrix = block.reshape(rows, k)
        if not np.all(matrix == expected[:, None]):
            bad = np.argwhere(matrix != expected[:, None])[0]
            raise Round0034PipelineError(
                "graph is not source-major fixed-k at "
                f"source {first_row + int(bad[0])}, slot {int(bad[1])}"
            )


def _verify_uniform_weights(
    graph_path: str,
    *,
    edge_count: int,
    k: int,
    block_values: int,
) -> float:
    expected = np.float32(1.0 / k)
    observed: np.float32 | None = None
    for _start, block in _member_blocks(
        graph_path,
        "weights.npy",
        dtype=np.dtype("<f4"),
        values=edge_count,
        block_values=block_values,
    ):
        if not np.all(np.isfinite(block)) or np.any(block <= 0):
            raise Round0034PipelineError("graph weights are non-finite/non-positive")
        if observed is None:
            observed = block[0]
            if not np.isclose(observed, expected, rtol=0.0, atol=1e-7):
                raise Round0034PipelineError(
                    f"graph weight {float(observed)} is not 1/k={float(expected)}"
                )
        if not np.all(block == observed):
            raise Round0034PipelineError("graph weights are not exactly constant")
    if observed is None:
        raise Round0034PipelineError("graph weight member is empty")
    return float(observed)


class _FreshRawWriter:
    """Fresh raw-array writer; the manifest is the output commit marker."""

    def __init__(self, path: str, *, dtype: np.dtype, shape: tuple[int, ...]):
        self.path = refuse_existing(path, label="R0034 raw output")
        self.dtype = np.dtype(dtype)
        self.shape = tuple(int(value) for value in shape)
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        self.fd = os.open(self.path, flags, 0o600)
        os.ftruncate(self.fd, int(np.prod(self.shape)) * self.dtype.itemsize)
        self.array = np.memmap(
            self.path, dtype=self.dtype, mode="r+", shape=self.shape
        )
        self.closed = False

    def publish(self) -> None:
        if self.closed:
            return
        self.array.flush()
        os.fsync(self.fd)
        del self.array
        os.fchmod(self.fd, 0o444)
        os.close(self.fd)
        self.closed = True

    def abort(self) -> None:
        if self.closed:
            return
        with contextlib.suppress(Exception):
            del self.array
        with contextlib.suppress(OSError):
            os.close(self.fd)
        with contextlib.suppress(OSError):
            os.unlink(self.path)
        self.closed = True


def _membership(sorted_rows: np.ndarray, values: np.ndarray) -> np.ndarray:
    if not len(sorted_rows):
        return np.zeros(values.shape, dtype=np.bool_)
    positions = np.searchsorted(sorted_rows, values)
    bounded = positions < len(sorted_rows)
    answer = np.zeros(values.shape, dtype=np.bool_)
    answer[bounded] = sorted_rows[positions[bounded]] == values[bounded]
    return answer


def _map_duplicate_rows(
    values: np.ndarray,
    duplicate_rows: np.ndarray,
    representative_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mapped = np.asarray(values, dtype=np.int64).copy()
    if not len(duplicate_rows):
        return mapped, np.zeros(mapped.shape, dtype=np.bool_)
    positions = np.searchsorted(duplicate_rows, mapped)
    hit = positions < len(duplicate_rows)
    hit_indices = np.flatnonzero(hit)
    hit[hit_indices] = duplicate_rows[positions[hit_indices]] == mapped[hit_indices]
    mapped[hit] = representative_rows[positions[hit]]
    return mapped, hit


def _canonicalize_target_block(
    targets: np.ndarray,
    *,
    first_row: int,
    eligibility: Mapping[str, Any],
    row_count: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    rows = targets.shape[0]
    if targets.shape != (rows, k):
        raise Round0034PipelineError("target block has wrong k")
    if np.any(targets < 0) or np.any(targets >= row_count):
        bad = np.argwhere((targets < 0) | (targets >= row_count))[0]
        raise Round0034PipelineError(
            f"out-of-range target at source {first_row + int(bad[0])}, "
            f"slot {int(bad[1])}: {int(targets[tuple(bad)])}"
        )

    row_ids = np.arange(first_row, first_row + rows, dtype=np.int64)
    eligible_sources = ~_membership(eligibility["excluded_rows"], row_ids)
    output = np.full((rows, k), -1, dtype=np.int32)
    degrees = np.zeros(rows, dtype=np.uint8)
    counts = {
        "duplicate_destinations_mapped": 0,
        "zero_destinations_dropped": 0,
        "self_destinations_dropped": 0,
        "repeated_canonical_destinations_dropped": 0,
        "excluded_source_edges_dropped": int((~eligible_sources).sum()) * k,
    }
    row_positions = np.arange(rows, dtype=np.int64)
    for slot in range(k):
        raw = targets[:, slot].astype(np.int64, copy=False)
        zero = _membership(eligibility["zero_rows"], raw) & eligible_sources
        mapped, remapped = _map_duplicate_rows(
            raw,
            eligibility["duplicate_excluded_rows"],
            eligibility["duplicate_representative_rows"],
        )
        remapped &= eligible_sources
        self_edge = (mapped == row_ids) & eligible_sources & ~zero
        valid = eligible_sources & ~zero & ~self_edge
        repeated = np.zeros(rows, dtype=np.bool_)
        for prior in range(slot):
            repeated |= output[:, prior].astype(np.int64, copy=False) == mapped
        repeated &= valid
        valid &= ~repeated
        selected = row_positions[valid]
        output[selected, degrees[selected].astype(np.int64)] = mapped[valid].astype(
            np.int32, copy=False
        )
        degrees[selected] += 1
        counts["duplicate_destinations_mapped"] += int(remapped.sum())
        counts["zero_destinations_dropped"] += int(zero.sum())
        counts["self_destinations_dropped"] += int(self_edge.sum())
        counts["repeated_canonical_destinations_dropped"] += int(repeated.sum())
    return output, degrees, counts


def build_canonical_graph(
    *,
    graph_path: str,
    expected_graph_sha256: str,
    eligibility: Mapping[str, Any],
    output_root: str,
    row_count: int = DEFAULT_ROWS,
    k: int = DEFAULT_K,
    block_rows: int = 131_072,
) -> dict[str, Any]:
    """Stream-validate and canonicalize the historical 150M uniform graph."""
    if row_count <= 1 or k <= 0 or not (1 <= block_rows):
        raise ValueError("invalid canonical graph geometry")
    graph_path = os.path.realpath(graph_path)
    graph_signature = expected_input_signature(graph_path)
    if graph_signature["sha256"] != expected_graph_sha256:
        raise Round0034PipelineError("150M graph SHA-256 changed")
    view = validate_eligibility_view(eligibility, row_count=row_count)
    if _scalar_member(graph_path, "n_nodes.npy", np.dtype("<i8")) != row_count:
        raise Round0034PipelineError("graph n_nodes is not the requested universe")
    if _scalar_member(graph_path, "k.npy", np.dtype("<i8")) != k:
        raise Round0034PipelineError("graph k is not the requested fixed degree")

    output_root = create_fresh_directory(
        output_root, label="R0034 canonical graph output"
    )
    target_path = os.path.join(output_root, "canonical-targets.i32")
    degree_path = os.path.join(output_root, "valid-degrees.u8")
    target_writer = _FreshRawWriter(
        target_path, dtype=np.dtype("<i4"), shape=(row_count, k)
    )
    degree_writer = _FreshRawWriter(
        degree_path, dtype=np.dtype("u1"), shape=(row_count,)
    )
    started = time.monotonic()
    phase_timing: dict[str, float] = {}
    aggregate = Counter()
    degree_histogram = np.zeros(k + 1, dtype=np.int64)
    positive_source_count = 0
    valid_edge_count = 0
    try:
        phase = time.monotonic()
        _verify_source_major(
            graph_path, row_count=row_count, k=k, block_rows=block_rows
        )
        phase_timing["verify_source_major_seconds"] = time.monotonic() - phase

        phase = time.monotonic()
        uniform_weight = _verify_uniform_weights(
            graph_path,
            edge_count=row_count * k,
            k=k,
            block_values=block_rows * k,
        )
        phase_timing["verify_uniform_weights_seconds"] = time.monotonic() - phase

        phase = time.monotonic()
        expected_edge_start = 0
        for edge_start, flat in _member_blocks(
            graph_path,
            "targets.npy",
            dtype=np.dtype("<i4"),
            values=row_count * k,
            block_values=block_rows * k,
        ):
            if edge_start != expected_edge_start or edge_start % k or len(flat) % k:
                raise Round0034PipelineError("target stream lost source-major alignment")
            rows = len(flat) // k
            first_row = edge_start // k
            canonical, degrees, counts = _canonicalize_target_block(
                flat.reshape(rows, k),
                first_row=first_row,
                eligibility=view,
                row_count=row_count,
                k=k,
            )
            target_writer.array[first_row:first_row + rows] = canonical
            degree_writer.array[first_row:first_row + rows] = degrees
            aggregate.update(counts)
            degree_histogram += np.bincount(
                degrees, minlength=k + 1
            ).astype(np.int64, copy=False)
            positive_source_count += int(np.count_nonzero(degrees))
            valid_edge_count += int(degrees.sum(dtype=np.int64))
            expected_edge_start += len(flat)
        if expected_edge_start != row_count * k:
            raise Round0034PipelineError("target stream row accounting did not close")
        phase_timing["canonicalize_targets_seconds"] = time.monotonic() - phase
        target_writer.publish()
        degree_writer.publish()
    except BaseException:
        target_writer.abort()
        degree_writer.abort()
        raise

    zero_degree_retained = int(view["retained_row_count"] - positive_source_count)
    if zero_degree_retained < 0:
        raise Round0034PipelineError("positive-source accounting exceeded eligibility")
    dropped = sum(
        aggregate[name]
        for name in (
            "excluded_source_edges_dropped",
            "zero_destinations_dropped",
            "self_destinations_dropped",
            "repeated_canonical_destinations_dropped",
        )
    )
    if valid_edge_count + dropped != row_count * k:
        raise Round0034PipelineError("canonical graph edge accounting did not close")

    target_signature = expected_input_signature(target_path)
    degree_signature = expected_input_signature(degree_path)
    body = {
        "schema": GRAPH_SCHEMA,
        "round_id": "0034",
        "row_count": row_count,
        "input_k": k,
        "target_dtype": np.dtype("<i4").str,
        "target_shape": [row_count, k],
        "degree_dtype": np.dtype("u1").str,
        "degree_shape": [row_count],
        "source_semantics": "source-major-row-id; no materialized source array",
        "weight_semantics": (
            "input weights proven exactly constant 1/k; no materialized weight array"
        ),
        "sampling_semantics": (
            "uniform-retained-positive-source-then-uniform-valid-canonical-target"
        ),
        "destination_policy": (
            "map exact duplicate copies to representative; drop zero/self/repeated "
            "canonical target per source"
        ),
        "negative_policy": "uniform-R0033-retained-rows-nonself",
        "inputs": {
            "graph": graph_signature,
            "eligibility": view["signature"],
        },
        "outputs": {
            "targets": target_signature,
            "degrees": degree_signature,
        },
        "summary": {
            "input_edge_count": row_count * k,
            "eligibility_retained_row_count": view["retained_row_count"],
            "eligibility_excluded_source_count": len(view["excluded_rows"]),
            "retained_positive_source_count": positive_source_count,
            "zero_degree_retained_source_count": zero_degree_retained,
            "zero_degree_retained_source_fraction": (
                zero_degree_retained / view["retained_row_count"]
            ),
            "valid_canonical_edge_count": valid_edge_count,
            "degree_histogram": {
                str(key): int(value)
                for key, value in enumerate(degree_histogram)
                if value
            },
            **{key: int(value) for key, value in sorted(aggregate.items())},
            "uniform_input_weight": uniform_weight,
        },
        "timing": {
            **phase_timing,
            "total_seconds": time.monotonic() - started,
        },
    }
    manifest = _seal(body)
    manifest_path = os.path.join(output_root, "canonical-graph-v1.json")
    atomic_write_new_json(manifest_path, manifest, immutable=True)
    return {
        **manifest,
        "manifest": expected_input_signature(manifest_path),
    }


def load_canonical_graph(
    manifest_path: str,
    *,
    expected_sha256: str,
    expected_eligibility_sha256: str,
    row_count: int,
) -> dict[str, Any]:
    """Load and validate the immutable graph adapter without source/weight arrays."""
    manifest_signature = expected_input_signature(manifest_path)
    if manifest_signature["sha256"] != expected_sha256:
        raise Round0034PipelineError("canonical graph manifest SHA-256 changed")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    body = {key: value for key, value in manifest.items()
            if key != "identity_sha256"}
    summary = manifest.get("summary") or {}
    if (
        manifest.get("schema") != GRAPH_SCHEMA
        or manifest.get("identity_sha256") != sha256_bytes(canonical_json(body))
        or int(manifest.get("row_count", -1)) != row_count
        or int(manifest.get("input_k", -1)) != DEFAULT_K
        or manifest.get("inputs", {}).get("eligibility", {}).get("sha256")
        != expected_eligibility_sha256
        or int(summary.get("retained_positive_source_count", -1)) <= 0
        or int(summary.get("valid_canonical_edge_count", -1)) <= 0
    ):
        raise Round0034PipelineError("canonical graph manifest content changed")
    outputs = manifest.get("outputs") or {}
    targets_signature = expected_input_signature(
        outputs.get("targets", {}).get("canonical_path", "")
    )
    degrees_signature = expected_input_signature(
        outputs.get("degrees", {}).get("canonical_path", "")
    )
    if (
        targets_signature != outputs.get("targets")
        or degrees_signature != outputs.get("degrees")
        or targets_signature["bytes"] != row_count * DEFAULT_K * 4
        or degrees_signature["bytes"] != row_count
    ):
        raise Round0034PipelineError("canonical graph output bytes changed")
    targets = np.memmap(
        targets_signature["canonical_path"],
        dtype="<i4",
        mode="r",
        shape=(row_count, DEFAULT_K),
    )
    degrees = np.memmap(
        degrees_signature["canonical_path"],
        dtype="u1",
        mode="r",
        shape=(row_count,),
    )
    if (
        os.stat(targets_signature["canonical_path"]).st_mode & 0o222
        or os.stat(degrees_signature["canonical_path"]).st_mode & 0o222
    ):
        raise Round0034PipelineError("canonical graph raw artifacts are mutable")
    observed_sources = 0
    observed_edges = 0
    observed_histogram = np.zeros(DEFAULT_K + 1, dtype=np.int64)
    for start in range(0, row_count, 5_000_000):
        block = np.asarray(degrees[start:start + 5_000_000])
        if np.any(block > DEFAULT_K):
            raise Round0034PipelineError("canonical graph degree exceeds k")
        observed_sources += int(np.count_nonzero(block))
        observed_edges += int(block.sum(dtype=np.int64))
        observed_histogram += np.bincount(
            block, minlength=DEFAULT_K + 1
        ).astype(np.int64, copy=False)
    if (
        observed_sources != int(summary["retained_positive_source_count"])
        or observed_edges != int(summary["valid_canonical_edge_count"])
        or {
            str(key): int(value)
            for key, value in enumerate(observed_histogram)
            if value
        }
        != summary.get("degree_histogram")
    ):
        raise Round0034PipelineError("canonical graph degree accounting changed")
    return {
        "manifest": manifest,
        "signature": manifest_signature,
        "targets": targets,
        "degrees": degrees,
    }


class HostInt8MaterializedArray:
    """Host-RAM int8 features plus exact per-row fp16 scales.

    Two reusable pinned buffer pairs permit safe asynchronous H2D copies.  A
    CUDA event guards each slot before the host writes it again.
    """

    round0034_host_int8 = True

    def __init__(
        self,
        encoded: np.ndarray,
        scales: np.ndarray,
        *,
        device: str,
        signatures: Mapping[str, Any] | None = None,
        buffer_rows: int = 8192,
    ):
        import torch

        if (
            encoded.ndim != 2
            or encoded.dtype != np.dtype("int8")
            or scales.shape != (encoded.shape[0],)
            or scales.dtype != np.dtype("<f2")
            or not np.all(np.isfinite(scales))
            or np.any(scales <= 0)
        ):
            raise Round0034PipelineError(
                "host int8 array requires int8 rows and finite positive little-endian fp16 scales"
            )
        if buffer_rows <= 0:
            raise ValueError("buffer_rows must be positive")
        self.encoded = encoded
        self.scales = scales
        self.shape = encoded.shape
        self.device = str(device)
        self.signatures = dict(signatures or {})
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
            source_i8 = torch.empty(
                (buffer_rows, encoded.shape[1]), dtype=torch.int8, pin_memory=pin
            )
            destination_i8 = torch.empty_like(source_i8, pin_memory=pin)
            source_scale = torch.empty(
                (buffer_rows,), dtype=torch.float16, pin_memory=pin
            )
            destination_scale = torch.empty_like(source_scale, pin_memory=pin)
            self._slots.append({
                "source_i8": source_i8,
                "destination_i8": destination_i8,
                "source_scale": source_scale,
                "destination_scale": destination_scale,
                "event": None,
            })
        self._slot_index = 0

    @classmethod
    def from_files(
        cls,
        *,
        int8_path: str,
        int8_sha256: str,
        scales_path: str,
        scales_sha256: str,
        row_count: int,
        dimension: int,
        device: str,
        buffer_rows: int,
    ) -> "HostInt8MaterializedArray":
        int8_signature = expected_input_signature(int8_path)
        scales_signature = expected_input_signature(scales_path)
        if (
            int8_signature["sha256"] != int8_sha256
            or scales_signature["sha256"] != scales_sha256
            or int8_signature["bytes"] != row_count * dimension
            or scales_signature["bytes"] != row_count * 2
        ):
            raise Round0034PipelineError("R0025 host-int8 bytes changed")
        encoded = np.fromfile(int8_path, dtype=np.int8).reshape(row_count, dimension)
        scales = np.fromfile(scales_path, dtype="<f2", count=row_count)
        return cls(
            encoded,
            scales,
            device=device,
            signatures={"int8": int8_signature, "scales": scales_signature},
            buffer_rows=buffer_rows,
        )

    def __len__(self) -> int:
        return int(self.shape[0])

    def to(self, _device: str) -> "HostInt8MaterializedArray":
        return self

    def _wait_slot(self, slot: Mapping[str, Any]) -> None:
        event = slot.get("event")
        if event is not None:
            event.synchronize()

    def _validated_pair_rows(
        self,
        source_rows: np.ndarray,
        destination_rows: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        source_rows = np.asarray(source_rows, dtype=np.int64)
        destination_rows = np.asarray(destination_rows, dtype=np.int64)
        if (
            source_rows.ndim != 1
            or destination_rows.shape != source_rows.shape
            or len(source_rows) > self.buffer_rows
            or np.any(source_rows < 0)
            or np.any(source_rows >= len(self))
            or np.any(destination_rows < 0)
            or np.any(destination_rows >= len(self))
        ):
            raise Round0034PipelineError("endpoint gather row IDs are invalid")
        return source_rows, destination_rows

    def fill_pair_slot(
        self,
        slot_index: int,
        source_rows: np.ndarray,
        destination_rows: np.ndarray,
    ) -> int:
        """Fill one pinned slot on the host without issuing CUDA work."""
        source_rows, destination_rows = self._validated_pair_rows(
            source_rows, destination_rows
        )
        if slot_index < 0 or slot_index >= len(self._slots):
            raise Round0034PipelineError("endpoint gather slot is invalid")
        slot = self._slots[slot_index]
        self._wait_slot(slot)
        count = len(source_rows)
        slot["source_i8"].numpy()[:count] = self.encoded[source_rows]
        slot["destination_i8"].numpy()[:count] = self.encoded[destination_rows]
        slot["source_scale"].numpy()[:count] = self.scales[source_rows]
        slot["destination_scale"].numpy()[:count] = self.scales[destination_rows]
        with self._accounting_lock:
            self.host_prefetch_batches_filled += 1
            self.host_prefetch_source_rows_filled += count
            self.host_prefetch_destination_rows_filled += count
        return count

    def transfer_pair_slot(self, slot_index: int, count: int):
        """Transfer and dequantize a previously filled host slot."""
        import torch

        if (
            slot_index < 0
            or slot_index >= len(self._slots)
            or count < 0
            or count > self.buffer_rows
        ):
            raise Round0034PipelineError("endpoint transfer slot/count is invalid")
        slot = self._slots[slot_index]

        source_i8 = slot["source_i8"][:count].to(
            self.device, non_blocking="cuda" in self.device
        )
        destination_i8 = slot["destination_i8"][:count].to(
            self.device, non_blocking="cuda" in self.device
        )
        source_scale = slot["source_scale"][:count].to(
            self.device, non_blocking="cuda" in self.device
        )
        destination_scale = slot["destination_scale"][:count].to(
            self.device, non_blocking="cuda" in self.device
        )
        source = source_i8.float() * source_scale.float().view(-1, 1)
        destination = (
            destination_i8.float() * destination_scale.float().view(-1, 1)
        )
        if "cuda" in self.device:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            slot["event"] = event
        with self._accounting_lock:
            self.endpoint_gather_calls += 1
            self.source_rows_gathered += count
            self.destination_rows_gathered += count
        return source, destination

    def gather_pairs(self, source_rows: np.ndarray, destination_rows: np.ndarray):
        slot_index = self._slot_index
        self._slot_index = (self._slot_index + 1) % len(self._slots)
        count = self.fill_pair_slot(slot_index, source_rows, destination_rows)
        return self.transfer_pair_slot(slot_index, count)

    def index_select(self, rows: Any):
        if hasattr(rows, "detach"):
            rows = rows.detach().cpu().numpy()
        rows = np.asarray(rows, dtype=np.int64)
        values, _ = self.gather_pairs(rows, rows)
        return values

    def execution_stamp(self) -> dict[str, Any]:
        return {
            "feature_residency": "host-ram-int8-plus-fp16-scale",
            "dequantization": "device-fp32-int8-times-exact-row-fp16-scale",
            "int8_signature": self.signatures.get("int8"),
            "scale_signature": self.signatures.get("scales"),
            "endpoint_gather_calls": self.endpoint_gather_calls,
            "source_rows_gathered": self.source_rows_gathered,
            "destination_rows_gathered": self.destination_rows_gathered,
            "host_prefetch_batches_filled": self.host_prefetch_batches_filled,
            "host_prefetch_source_rows_filled": self.host_prefetch_source_rows_filled,
            "host_prefetch_destination_rows_filled": (
                self.host_prefetch_destination_rows_filled
            ),
        }


class HostInt8CanonicalSampler:
    """Uniform-source, uniform-valid-target sampler for the canonical graph."""

    def __init__(
        self,
        dataset: HostInt8MaterializedArray,
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
    ):
        import torch

        self.dataset = dataset
        self.targets = targets
        self.degrees = degrees
        self.excluded_rows = np.asarray(excluded_rows, dtype=np.int64)
        self.n_nodes = len(dataset)
        self.k = int(targets.shape[1])
        self.positive_source_count = int(positive_source_count)
        self.n_pos = int(valid_edge_count)
        self.batch_size = int(batch_size)
        self.num_pos = max(1, int(batch_size * pos_ratio))
        self.num_neg = batch_size - self.num_pos
        self.rng = np.random.default_rng(int(random_state))
        self.graph_signature = dict(graph_signature)
        self.eligibility_signature = dict(eligibility_signature)
        self.device = dataset.device
        self.batch_no = 0
        self.fused_endpoint_forward = True
        if (
            targets.shape != (self.n_nodes, DEFAULT_K)
            or targets.dtype != np.dtype("<i4")
            or degrees.shape != (self.n_nodes,)
            or degrees.dtype != np.dtype("u1")
            or self.positive_source_count <= 0
            or self.n_pos < self.positive_source_count
            or not 0 < pos_ratio < 1
            or self.num_neg <= 0
        ):
            raise Round0034PipelineError("canonical sampler geometry is invalid")
        self._label_device = torch.device(self.device)
        self._labels = torch.cat((
            torch.ones(self.num_pos, dtype=torch.float32, device=self._label_device),
            torch.zeros(self.num_neg, dtype=torch.float32, device=self._label_device),
        ))
        self._prefetch_enabled = "cuda" in str(self.device)
        self._prefetch_executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._prefetch_future: concurrent.futures.Future[tuple[int, int]] | None = None
        self._producer_batches = 0
        self._consumer_batches = 0

    def __len__(self) -> int:
        return int(math.ceil(self.n_pos / self.num_pos))

    def __iter__(self) -> "HostInt8CanonicalSampler":
        self.batch_no = 0
        if self._prefetch_enabled and self._prefetch_executor is None:
            self._start_prefetch()
        return self

    def _start_prefetch(self) -> None:
        self._prefetch_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="r0034-host-int8-prefetch"
        )
        self._prefetch_future = self._prefetch_executor.submit(
            self._prefetch_one, 0
        )

    def _draw_conditioned(self, count: int, predicate) -> np.ndarray:
        output = np.empty(count, dtype=np.int64)
        filled = 0
        while filled < count:
            remaining = count - filled
            proposal_count = max(remaining, int(math.ceil(remaining * 1.02)))
            proposal = self.rng.integers(
                0, self.n_nodes, size=proposal_count, dtype=np.int64
            )
            proposal = proposal[predicate(proposal)]
            take = min(remaining, len(proposal))
            if take:
                output[filled:filled + take] = proposal[:take]
                filled += take
        return output

    def _is_retained(self, values: np.ndarray) -> np.ndarray:
        return ~_membership(self.excluded_rows, values)

    def _draw_positive_pairs(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        source = self._draw_conditioned(
            count, lambda values: self.degrees[values] > 0
        )
        degree = np.asarray(self.degrees[source], dtype=np.int64)
        slots = np.empty(count, dtype=np.int64)
        for value in np.unique(degree):
            mask = degree == value
            slots[mask] = self.rng.integers(
                0, int(value), size=int(mask.sum()), dtype=np.int64
            )
        destination = np.asarray(self.targets[source, slots], dtype=np.int64)
        if (
            np.any(destination < 0)
            or np.any(source == destination)
            or not np.all(self._is_retained(source))
            or not np.all(self._is_retained(destination))
        ):
            raise Round0034PipelineError("canonical positive sampler invariant failed")
        return source, destination

    def _draw_negative_pairs(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        source = self._draw_conditioned(count, self._is_retained)
        destination = self._draw_conditioned(count, self._is_retained)
        same = source == destination
        while np.any(same):
            destination[same] = self._draw_conditioned(int(same.sum()), self._is_retained)
            same = source == destination
        return source, destination

    def _draw_batch_rows(self) -> tuple[np.ndarray, np.ndarray]:
        positive_source, positive_destination = self._draw_positive_pairs(
            self.num_pos
        )
        negative_source, negative_destination = self._draw_negative_pairs(
            self.num_neg
        )
        return (
            np.concatenate((positive_source, negative_source)),
            np.concatenate((positive_destination, negative_destination)),
        )

    def _prefetch_one(self, slot_index: int) -> tuple[int, int]:
        source, destination = self._draw_batch_rows()
        count = self.dataset.fill_pair_slot(slot_index, source, destination)
        self._producer_batches += 1
        return slot_index, count

    def _next_prefetched(self):
        assert self._prefetch_executor is not None
        assert self._prefetch_future is not None
        try:
            slot_index, count = self._prefetch_future.result()
        except BaseException as error:
            raise Round0034PipelineError(
                "R0034 host-int8 prefetch producer failed"
            ) from error
        values = self.dataset.transfer_pair_slot(slot_index, count)
        self._consumer_batches += 1
        if self.batch_no < len(self):
            # transfer_pair_slot records the CUDA event before the alternate
            # slot is submitted. Reusing this slot two batches later waits that
            # event, while the one pending host fill overlaps current GPU work.
            self._prefetch_future = self._prefetch_executor.submit(
                self._prefetch_one, (slot_index + 1) % len(self.dataset._slots)
            )
        else:
            self._prefetch_future = None
        return values

    def __next__(self):
        if self.batch_no >= len(self):
            raise StopIteration
        self.batch_no += 1
        if self._prefetch_enabled:
            source_values, destination_values = self._next_prefetched()
        else:
            source, destination = self._draw_batch_rows()
            source_values, destination_values = self.dataset.gather_pairs(
                source, destination
            )
            self._consumer_batches += 1
        return source_values, destination_values, self._labels

    def close(self) -> None:
        pending_error: BaseException | None = None
        if self._prefetch_future is not None:
            try:
                self._prefetch_future.result()
            except BaseException as error:
                pending_error = error
        if self._prefetch_executor is not None:
            self._prefetch_executor.shutdown(wait=True, cancel_futures=True)
            self._prefetch_executor = None
        self._prefetch_future = None
        if pending_error is not None:
            raise Round0034PipelineError(
                "R0034 host-int8 prefetch producer failed"
            ) from pending_error

    def execution_stamp(self) -> dict[str, Any]:
        return {
            "schema": PIPELINE_SCHEMA,
            "pipeline": "host_int8_canonical",
            "sampler_class": type(self).__name__,
            "positive_sampling": (
                "uniform-retained-positive-source-then-uniform-valid-canonical-"
                "destination-with-replacement"
            ),
            "positive_destination_policy": (
                "R0033-duplicate-to-representative;zero-self-repeated-dropped"
            ),
            "negative_sampling": "uniform-R0033-retained-rows-nonself",
            "graph_degree": "variable-1-through-15;zero-degree-sources-excluded",
            "x_residency": "host_int8_materialized",
            "host_prefetch": (
                "single-producer-two-pinned-slot"
                if self._prefetch_enabled else "disabled-noncuda"
            ),
            "host_prefetch_producer_batches": self._producer_batches,
            "host_prefetch_consumer_batches": self._consumer_batches,
            "endpoint_forward": "fused-source-destination",
            "graph_manifest": self.graph_signature,
            "eligibility": self.eligibility_signature,
            "positive_source_count": self.positive_source_count,
            "valid_canonical_edge_count": self.n_pos,
            **self.dataset.execution_stamp(),
        }


class Round0034TrainingInput:
    """Adapter handed directly to ``ParametricUMAP.fit`` for R0034."""

    round0034_host_int8 = True

    def __init__(
        self,
        dataset: HostInt8MaterializedArray,
        graph: Mapping[str, Any],
        eligibility: Mapping[str, Any],
    ):
        self.dataset = dataset
        self.graph = dict(graph)
        self.eligibility = validate_eligibility_view(
            eligibility, row_count=len(dataset)
        )
        self.shape = dataset.shape
        self._last_sampler: HostInt8CanonicalSampler | None = None
        if np.any(self.graph["degrees"][self.eligibility["excluded_rows"]] != 0):
            raise Round0034PipelineError(
                "canonical graph admits an R0033-excluded positive source"
            )
        summary = self.graph["manifest"]["summary"]
        if (
            int(summary["eligibility_retained_row_count"])
            != self.eligibility["retained_row_count"]
            or int(summary["eligibility_excluded_source_count"])
            != len(self.eligibility["excluded_rows"])
        ):
            raise Round0034PipelineError(
                "canonical graph and eligibility row accounting differ"
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def to(self, _device: str) -> "Round0034TrainingInput":
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
    ) -> tuple["Round0034TrainingInput", HostInt8CanonicalSampler, int,
               dict[str, Any], dict[str, Any]]:
        manifest = self.graph["manifest"]
        signature = self.graph["signature"]
        if os.path.realpath(edges_path) != signature["canonical_path"]:
            raise Round0034PipelineError(
                "trainer graph path is not the loaded canonical manifest"
            )
        if (
            positive_target_mode != "binary"
            or weighted_edge_sampling
            or reject_neighbors
            or required_input_pipeline != "host_int8_canonical"
        ):
            raise Round0034PipelineError(
                "R0034 requires binary uniform canonical sampling on the exact host-int8 pipeline"
            )
        summary = manifest["summary"]
        sampler = HostInt8CanonicalSampler(
            self.dataset,
            targets=self.graph["targets"],
            degrees=self.graph["degrees"],
            excluded_rows=self.eligibility["excluded_rows"],
            positive_source_count=summary["retained_positive_source_count"],
            valid_edge_count=summary["valid_canonical_edge_count"],
            batch_size=batch_size,
            pos_ratio=pos_ratio,
            random_state=random_state,
            graph_signature=signature,
            eligibility_signature=self.eligibility["signature"],
        )
        self._last_sampler = sampler
        stamp = sampler.execution_stamp()
        verified = {
            "canonical_graph_manifest": signature,
            "canonical_targets": manifest["outputs"]["targets"],
            "canonical_degrees": manifest["outputs"]["degrees"],
            "eligibility": self.eligibility["signature"],
            "int8": self.dataset.signatures.get("int8"),
            "scales": self.dataset.signatures.get("scales"),
        }
        return self, sampler, sampler.n_pos, stamp, verified

    def runtime_stamp(self) -> dict[str, Any]:
        if self._last_sampler is None:
            raise Round0034PipelineError("R0034 sampler has not been constructed")
        return self._last_sampler.execution_stamp()


def run_two_endpoint_no_update_canary(
    training_input: Round0034TrainingInput,
    *,
    graph_manifest_path: str,
    batch_size: int = 8192,
    pos_ratio: float = 0.05,
    random_state: int = 42,
    warmup_steps: int = 20,
    measured_steps: int = 100,
    minimum_batches_per_second: float = 90.0,
    minimum_headroom_gib: float = 1.5,
) -> dict[str, Any]:
    """Pull real production batches and exercise forward/backward, never step.

    The rate is named ``train_step_equivalents_per_second``: the canary runs
    the exact endpoint pipeline, model forward, BCE, backward, and gradient
    clipping but deliberately creates no optimizer and performs zero updates.
    """
    import statistics
    import torch

    if "cuda" not in training_input.dataset.device or not torch.cuda.is_available():
        raise Round0034PipelineError("R0034 production canary requires CUDA")
    if warmup_steps < 1 or measured_steps < 5:
        raise ValueError("canary requires warmup and at least five measured steps")
    from .pumap.parametric_umap.models.mlp import ResidualBottleneckMLP

    _dataset, sampler, _n_pos, stamp, verified = (
        training_input.prepare_round0034_training(
            edges_path=graph_manifest_path,
            batch_size=batch_size,
            pos_ratio=pos_ratio,
            random_state=random_state,
            positive_target_mode="binary",
            weighted_edge_sampling=False,
            reject_neighbors=False,
            required_input_pipeline="host_int8_canonical",
        )
    )
    device = torch.device(training_input.dataset.device)
    torch.manual_seed(random_state)
    model = ResidualBottleneckMLP(
        input_dim=training_input.shape[1],
        hidden_dim=2048,
        output_dim=2,
        num_layers=3,
    ).to(device)
    loss_fn = torch.nn.BCELoss()
    torch.cuda.reset_peak_memory_stats(device)
    iterator = iter(sampler)
    rates: list[float] = []
    windows = 5
    measured_per_window = measured_steps // windows
    remainder = measured_steps % windows
    checksum = torch.zeros((), dtype=torch.float32, device=device)

    def step() -> None:
        model.zero_grad(set_to_none=True)
        source, destination, labels = next(iterator)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            endpoint_count = len(source)
            both_xy = model(torch.cat((source, destination), dim=0))
            source_xy = both_xy[:endpoint_count]
            destination_xy = both_xy[endpoint_count:]
        # Match ParametricUMAP._low_dim_qs: the shipped legacy radial and BCE
        # inputs are explicitly fp32 even when the model forward uses bf16.
        radius = torch.norm(source_xy.float() - destination_xy.float(), dim=1)
        probability = torch.clamp(1.0 / (1.0 + radius), 1e-7, 1.0 - 1e-7)
        loss = loss_fn(probability.float(), labels.float())
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(norm)):
            raise Round0034PipelineError("R0034 canary produced non-finite loss/gradient")
        checksum.add_(loss.detach())

    for _ in range(warmup_steps):
        step()
    torch.cuda.synchronize(device)
    for window in range(windows):
        count = measured_per_window + (1 if window < remainder else 0)
        started = time.perf_counter()
        for _ in range(count):
            step()
        torch.cuda.synchronize(device)
        rates.append(count / (time.perf_counter() - started))
    sampler.close()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    median_rate = float(statistics.median(rates))
    headroom_gib = float(free_bytes) / (1024 ** 3)
    runtime_stamp = sampler.execution_stamp()
    expected_rows = (warmup_steps + measured_steps) * batch_size
    exact_gathers = (
        runtime_stamp["source_rows_gathered"] == expected_rows
        and runtime_stamp["destination_rows_gathered"] == expected_rows
    )
    body = {
        "schema": CANARY_SCHEMA,
        "round_id": "0034",
        "optimizer_updates": 0,
        "model": {
            "architecture": "residual_bottleneck",
            "input_dimension": training_input.shape[1],
            "hidden_dimension": 2048,
            "hidden_layers": 3,
            "output_dimension": 2,
            "low_dim_kernel": "legacy_lp",
            "autocast": "bf16",
        },
        "loop": {
            "batch_size": batch_size,
            "positive_ratio": pos_ratio,
            "warmup_train_step_equivalents": warmup_steps,
            "measured_train_step_equivalents": measured_steps,
            "operation": (
                "real-sampler-prefetched-two-endpoint-gather-dequant-fused-forward-"
                "bce-backward-clip-no-optimizer"
            ),
        },
        "pipeline_at_setup": stamp,
        "pipeline_after_measurement": runtime_stamp,
        "verified_hashes": verified,
        "train_step_equivalents_per_second_windows": rates,
        "train_step_equivalents_per_second_median": median_rate,
        "minimum_train_step_equivalents_per_second": minimum_batches_per_second,
        "post_setup_headroom_gib": headroom_gib,
        "minimum_post_setup_headroom_gib": minimum_headroom_gib,
        "total_device_memory_gib": float(total_bytes) / (1024 ** 3),
        "endpoint_gather_accounting_exact": exact_gathers,
        "checksum": float(checksum),
        "passed": bool(
            exact_gathers
            and median_rate >= minimum_batches_per_second
            and headroom_gib >= minimum_headroom_gib
        ),
    }
    return _seal(body)
