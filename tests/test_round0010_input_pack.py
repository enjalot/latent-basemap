from __future__ import annotations

import dataclasses
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from basemap.minilm_input_pack import (
    ENDPOINT_DTYPE,
    PackError,
    PlannedInterruption,
    RawMapMember,
    RawSourceMap,
    _validate_endpoint_array,
    _verify_constant_weights,
    canonical_sha256,
    file_identity,
    seal_record,
    sha256_file,
    stream_transform_to_npy_chunks,
    verify_raw_map_members,
    verify_sealed_record,
)


def _write_raw(path: Path, values: np.ndarray, *, global_start: int) -> RawMapMember:
    contiguous = np.ascontiguousarray(values, dtype="<f4")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contiguous.tobytes(order="C"))
    digest, _, _ = sha256_file(path)
    return RawMapMember(
        path=path.resolve(),
        corpus=path.stem,
        global_start=global_start,
        global_stop=global_start + len(contiguous),
        local_start=0,
        full_rows=len(contiguous),
        identity=file_identity(path),
        sha256=digest,
    )


def _small_source(tmp_path: Path) -> tuple[RawSourceMap, np.ndarray, list[RawMapMember]]:
    arrays = [
        np.arange(0, 20, dtype=np.float32).reshape(5, 4),
        np.arange(20, 32, dtype=np.float32).reshape(3, 4),
        np.arange(32, 56, dtype=np.float32).reshape(6, 4),
    ]
    cursor = 0
    members = []
    for index, array in enumerate(arrays):
        members.append(_write_raw(tmp_path / f"raw-{index}.npy", array, global_start=cursor))
        cursor += len(array)
    return (
        RawSourceMap(members, total_rows=cursor, dimension=4),
        np.concatenate(arrays),
        members,
    )


def test_module_is_torch_free_and_cuda_hidden() -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert "torch" not in sys.modules


def test_sealed_records_are_canonical_and_mutation_fails() -> None:
    record = seal_record({"b": [2, 1], "a": {"value": 3}})
    verify_sealed_record(record)
    assert record["receipt_sha256"] == canonical_sha256(
        {"a": {"value": 3}, "b": [2, 1]}
    )
    record["a"]["value"] = 4
    with pytest.raises(PackError, match="mismatch"):
        verify_sealed_record(record)


def test_headerless_loader_vectorizes_cross_shard_slices_and_indices(tmp_path: Path) -> None:
    source, expected, _ = _small_source(tmp_path)
    assert np.array_equal(source.read(3, 11), expected[3:11])
    indices = np.array([13, 0, 7, 7, 4, 9], dtype=np.int64)
    assert np.array_equal(source.take(indices), expected[indices])
    with pytest.raises(IndexError):
        source.read(-1, 2)
    with pytest.raises(IndexError):
        source.take([14])


def test_stream_transform_interrupt_resume_is_byte_identical_to_clean(
    tmp_path: Path,
) -> None:
    source, expected, _ = _small_source(tmp_path / "source")

    def transform(block: np.ndarray) -> np.ndarray:
        return np.column_stack((block[:, 0] + block[:, 1], block[:, 2] - block[:, 3]))

    kwargs = {
        "transform_id": "test-linear-v1",
        "output_dim": 2,
        "rows_per_chunk": 4,
        "read_block_rows": 2,
    }
    with pytest.raises(PlannedInterruption):
        stream_transform_to_npy_chunks(
            source,
            tmp_path / "resume",
            transform,
            interrupt_after_chunks=2,
            **kwargs,
        )
    resumed = stream_transform_to_npy_chunks(
        source, tmp_path / "resume", transform, **kwargs
    )
    clean = stream_transform_to_npy_chunks(
        source, tmp_path / "clean", transform, **kwargs
    )
    assert resumed["capability_sha256"] == clean["capability_sha256"]
    assert resumed["resumed_chunks_this_invocation"] == 2
    assert clean["created_chunks_this_invocation"] == 4
    observed = []
    for chunk in sorted((tmp_path / "resume").glob("chunk-*")):
        observed.append(np.load(chunk / "coordinates.npy", allow_pickle=False))
        verify_sealed_record(__import__("json").loads((chunk / "receipt.json").read_text()))
    expected_output = transform(expected)
    assert np.array_equal(np.concatenate(observed), expected_output)


def test_source_mutations_reorder_missing_duplicate_truncate_stale_and_hash_fail(
    tmp_path: Path,
) -> None:
    _, _, members = _small_source(tmp_path / "source")
    total = members[-1].global_stop
    with pytest.raises(PackError):
        verify_raw_map_members(
            list(reversed(members)), total_rows=total, dimension=4, full_hash=False
        )
    with pytest.raises(PackError):
        verify_raw_map_members(
            [members[0], members[2]], total_rows=total, dimension=4, full_hash=False
        )
    with pytest.raises(PackError):
        verify_raw_map_members(
            [members[0], members[0], *members[1:]],
            total_rows=total,
            dimension=4,
            full_hash=False,
        )

    truncated_path = tmp_path / "truncated.npy"
    truncated_path.write_bytes(members[0].path.read_bytes())
    truncated = dataclasses.replace(
        members[0],
        path=truncated_path.resolve(),
        identity=file_identity(truncated_path),
    )
    with truncated_path.open("r+b") as handle:
        handle.truncate(truncated_path.stat().st_size - 4)
    with pytest.raises(PackError, match="geometry"):
        verify_raw_map_members(
            [truncated],
            total_rows=truncated.global_stop,
            dimension=4,
            full_hash=False,
        )

    stale_path = tmp_path / "stale.npy"
    stale_path.write_bytes(members[0].path.read_bytes())
    stale = dataclasses.replace(
        members[0], path=stale_path.resolve(), identity=file_identity(stale_path)
    )
    os.utime(stale_path, ns=(stale.identity["mtime_ns"] + 10, stale.identity["mtime_ns"] + 10))
    with pytest.raises(PackError, match="stale"):
        verify_raw_map_members(
            [stale], total_rows=stale.global_stop, dimension=4, full_hash=False
        )

    mismatch_path = tmp_path / "mismatch.npy"
    mismatch_path.write_bytes(members[0].path.read_bytes())
    original_hash = hashlib.sha256(mismatch_path.read_bytes()).hexdigest()
    mismatch = dataclasses.replace(
        members[0],
        path=mismatch_path.resolve(),
        identity=file_identity(mismatch_path),
        sha256=original_hash,
    )
    with mismatch_path.open("r+b") as handle:
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 1]))
    with pytest.raises(PackError, match="hash mismatch"):
        verify_raw_map_members(
            [mismatch],
            total_rows=mismatch.global_stop,
            dimension=4,
            full_hash=True,
            check_identity=False,
        )


def _write_tiny_graph(path: Path, *, bad_source: bool = False, bad_weight: bool = False) -> None:
    n_nodes, k = 6, 2
    sources = np.repeat(np.arange(n_nodes, dtype=ENDPOINT_DTYPE), k)
    if bad_source:
        sources[3] = 0
    targets = (sources + np.tile(np.array([1, 2], dtype=ENDPOINT_DTYPE), n_nodes)) % n_nodes
    weights = np.full(len(sources), np.float32(1 / k), dtype="<f4")
    if bad_weight:
        weights[-1] = np.float32(0.25)
    np.savez_compressed(
        path, sources=sources, targets=targets, weights=weights, n_nodes=n_nodes, k=k, nprobe=1
    )


def test_small_endpoint_order_bounds_and_constant_weights(tmp_path: Path) -> None:
    graph = tmp_path / "graph.npz"
    _write_tiny_graph(graph)
    with np.load(graph, allow_pickle=False) as archive:
        np.save(tmp_path / "sources.npy", archive["sources"])
        np.save(tmp_path / "targets.npy", archive["targets"])
    source = _validate_endpoint_array(
        tmp_path / "sources.npy", role="sources", edge_count=12, n_nodes=6, k=2
    )
    target = _validate_endpoint_array(
        tmp_path / "targets.npy", role="targets", edge_count=12, n_nodes=6, k=2
    )
    weights = _verify_constant_weights(graph, edge_count=12, k=2)
    assert source["source_repeat_k_row_order_exact"] is True
    assert target["all_in_bounds"] is True
    assert weights["all_values_bitwise_equal"] is True
    assert weights["cdf_required"] is False


def test_endpoint_order_and_nonconstant_weight_mutations_fail(tmp_path: Path) -> None:
    bad_order = tmp_path / "bad-order.npz"
    _write_tiny_graph(bad_order, bad_source=True)
    with np.load(bad_order, allow_pickle=False) as archive:
        np.save(tmp_path / "bad-sources.npy", archive["sources"])
    with pytest.raises(PackError, match="row order mismatch"):
        _validate_endpoint_array(
            tmp_path / "bad-sources.npy",
            role="sources",
            edge_count=12,
            n_nodes=6,
            k=2,
        )
    bad_weight = tmp_path / "bad-weight.npz"
    _write_tiny_graph(bad_weight, bad_weight=True)
    with pytest.raises(PackError, match="nonconstant"):
        _verify_constant_weights(bad_weight, edge_count=12, k=2)
