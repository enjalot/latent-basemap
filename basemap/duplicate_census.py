"""Exact fp16 duplicate-family census for the accepted 30M MiniLM input pack."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .artifact_identity import (
    canonical_json,
    expected_input_signature,
    ordered_array_sha256,
    sha256_bytes,
)
from .duplicate_multiplicity import SCHEMA as DUPLICATE_CAP_SCHEMA
from .output_safety import atomic_save_new_npz, atomic_write_new_json, create_fresh_directory


ACCEPTED_INPUT_PACK_SHA256 = (
    "8f5a6ba8203aa583bbbdca3383f050e29c443ca0e25d628735bba873075bf7f2"
)
ROW_COUNT = 30_000_000
DIMENSION = 384
CHUNK_ROWS = 1_000_000
ROW_BYTES = DIMENSION * np.dtype("<f2").itemsize
KNOWN_R0019_ROWS = (16_008_908, 16_010_337, 16_013_677, 18_508_615)
KNOWN_R0019_COMPONENT_COUNTS = (9852, 307, 6, 1)
FINGERPRINT_DTYPE = np.dtype([("h0", "<u8"), ("h1", "<u8"), ("row", "<u4")])


@dataclass(frozen=True)
class MaterializedMember:
    chunk_index: int
    path: str
    global_start: int
    global_stop: int
    sha256: str
    bytes: int


def _seal(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}


def _digest128(row_bytes: bytes | memoryview) -> tuple[int, int]:
    digest = hashlib.sha256(row_bytes).digest()
    return int.from_bytes(digest[:8], "little"), int.from_bytes(digest[8:16], "little")


def load_materialized_members(manifest_path: str) -> list[MaterializedMember]:
    """Load and validate the accepted 30 one-million-row fp16 chunks."""
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("capability_hash_sha256") != ACCEPTED_INPUT_PACK_SHA256:
        raise ValueError("input-pack capability identity does not match 30m-input-pack-v1")
    members = manifest.get("capability_payload", {}).get(
        "materialized_fp16", {}
    ).get("ordered_members")
    if not isinstance(members, list) or len(members) != 30:
        raise ValueError("input-pack manifest does not declare exactly 30 fp16 chunks")
    out: list[MaterializedMember] = []
    cursor = 0
    for position, item in enumerate(members):
        shape = item.get("shape")
        if (
            item.get("chunk_index") != position
            or item.get("global_row_start") != cursor
            or item.get("global_row_stop") != cursor + CHUNK_ROWS
            or item.get("dtype") != "<f2"
            or shape != [CHUNK_ROWS, DIMENSION]
        ):
            raise ValueError(f"materialized chunk order/shape changed at {position}")
        out.append(
            MaterializedMember(
                chunk_index=position,
                path=os.path.realpath(str(item["path"])),
                global_start=int(item["global_row_start"]),
                global_stop=int(item["global_row_stop"]),
                sha256=str(item["sha256"]),
                bytes=int(item["size_bytes"]),
            )
        )
        cursor += CHUNK_ROWS
    if cursor != ROW_COUNT:
        raise ValueError(f"materialized row coverage ended at {cursor}")
    return out


def verify_member_files(members: Sequence[MaterializedMember]) -> list[dict[str, Any]]:
    """Full-hash every fp16 shard against the accepted manifest before use."""
    verified: list[dict[str, Any]] = []
    for member in members:
        signature = expected_input_signature(member.path)
        if signature["sha256"] != member.sha256 or signature["bytes"] != member.bytes:
            raise ValueError(f"materialized chunk hash/size mismatch: {member.path}")
        array = np.load(member.path, mmap_mode="r", allow_pickle=False)
        if tuple(array.shape) != (CHUNK_ROWS, DIMENSION) or array.dtype.str != "<f2":
            raise ValueError(f"materialized chunk ndarray contract changed: {member.path}")
        del array
        verified.append(
            {
                "chunk_index": member.chunk_index,
                "global_row_start": member.global_start,
                "global_row_stop": member.global_stop,
                "signature": signature,
            }
        )
    return verified


def write_row_fingerprints(
    members: Sequence[MaterializedMember],
    fingerprint_path: str,
    *,
    progress_every_chunks: int = 1,
) -> dict[str, Any]:
    """Hash every exact fp16 row as sha256(row_bytes) truncated to 128 bits."""
    started = time.monotonic()
    fingerprints = np.memmap(
        fingerprint_path, dtype=FINGERPRINT_DTYPE, mode="w+", shape=(ROW_COUNT,)
    )
    cursor = 0
    per_chunk: list[dict[str, Any]] = []
    for member in members:
        chunk_started = time.monotonic()
        array = np.load(member.path, mmap_mode="r", allow_pickle=False)
        if tuple(array.shape) != (CHUNK_ROWS, DIMENSION) or array.dtype.str != "<f2":
            raise ValueError(f"materialized chunk ndarray contract changed: {member.path}")
        if not array.flags.c_contiguous:
            raise ValueError(f"materialized chunk is not C contiguous: {member.path}")
        out = fingerprints[cursor : cursor + CHUNK_ROWS]
        for local_start in range(0, CHUNK_ROWS, 8192):
            block = np.ascontiguousarray(
                array[local_start : local_start + 8192], dtype="<f2"
            )
            raw = memoryview(block).cast("B")
            for local in range(block.shape[0]):
                h0, h1 = _digest128(raw[local * ROW_BYTES : (local + 1) * ROW_BYTES])
                out["h0"][local_start + local] = h0
                out["h1"][local_start + local] = h1
                out["row"][local_start + local] = member.global_start + local_start + local
        del array
        fingerprints.flush()
        per_chunk.append(
            {
                "chunk_index": member.chunk_index,
                "rows": CHUNK_ROWS,
                "wall_seconds": time.monotonic() - chunk_started,
            }
        )
        cursor += CHUNK_ROWS
        if progress_every_chunks and (member.chunk_index + 1) % progress_every_chunks == 0:
            print(
                f"duplicate-census: hashed chunk {member.chunk_index + 1}/30",
                flush=True,
            )
    if cursor != ROW_COUNT:
        raise RuntimeError(f"fingerprint writer ended at row {cursor}")
    return {
        "fingerprint_path": fingerprint_path,
        "fingerprint_dtype": FINGERPRINT_DTYPE.descr,
        "row_hash": "sha256(exact-768-byte-fp16-row) truncated to first 128 bits",
        "rows": ROW_COUNT,
        "per_chunk": per_chunk,
        "wall_seconds": time.monotonic() - started,
    }


def _row_bytes(row: int, members: Sequence[MaterializedMember], cache: dict[int, Any]) -> bytes:
    chunk_index = int(row) // CHUNK_ROWS
    member = members[chunk_index]
    if chunk_index not in cache:
        cache[chunk_index] = np.load(member.path, mmap_mode="r", allow_pickle=False)
    local = int(row) - member.global_start
    return np.ascontiguousarray(cache[chunk_index][local], dtype="<f2").tobytes()


def _split_exact_families(
    rows: np.ndarray, members: Sequence[MaterializedMember], cache: dict[int, Any]
) -> list[np.ndarray]:
    groups: dict[bytes, list[int]] = {}
    for row in rows.tolist():
        groups.setdefault(_row_bytes(int(row), members, cache), []).append(int(row))
    families = [
        np.asarray(sorted(group_rows), dtype=np.int64)
        for group_rows in groups.values()
        if len(group_rows) >= 2
    ]
    families.sort(key=lambda value: int(value[0]))
    return families


def find_duplicate_families(
    fingerprint_path: str,
    members: Sequence[MaterializedMember],
) -> dict[str, Any]:
    """Sort row fingerprints, then byte-verify every repeated-hash group."""
    started = time.monotonic()
    fingerprints = np.memmap(
        fingerprint_path, dtype=FINGERPRINT_DTYPE, mode="r", shape=(ROW_COUNT,)
    )
    print("duplicate-census: sorting 30M row fingerprints", flush=True)
    order = np.argsort(fingerprints, order=("h0", "h1"), kind="stable")
    sorted_fp = np.asarray(fingerprints[order])
    del order, fingerprints
    same = (
        (sorted_fp["h0"][1:] == sorted_fp["h0"][:-1])
        & (sorted_fp["h1"][1:] == sorted_fp["h1"][:-1])
    )
    repeated_positions = np.flatnonzero(same)
    exact_cache: dict[int, Any] = {}
    families: list[np.ndarray] = []
    repeated_hash_groups = 0
    hash_collision_splits = 0
    if repeated_positions.size:
        starts = repeated_positions[
            np.r_[True, repeated_positions[1:] != repeated_positions[:-1] + 1]
        ]
        for start in starts.tolist():
            stop = int(start) + 1
            while stop < ROW_COUNT and (
                sorted_fp["h0"][stop] == sorted_fp["h0"][start]
                and sorted_fp["h1"][stop] == sorted_fp["h1"][start]
            ):
                stop += 1
            rows = np.sort(sorted_fp["row"][start:stop].astype(np.int64))
            split = _split_exact_families(rows, members, exact_cache)
            repeated_hash_groups += 1
            if len(split) > 1:
                hash_collision_splits += len(split) - 1
            families.extend(split)
    del sorted_fp, same
    families.sort(key=lambda value: int(value[0]))
    representative_rows = np.asarray([int(rows[0]) for rows in families], dtype=np.int64)
    family_counts = np.asarray([len(rows) for rows in families], dtype=np.int64)
    offsets = np.zeros(len(families) + 1, dtype=np.int64)
    if families:
        offsets[1:] = np.cumsum(family_counts, dtype=np.int64)
        member_rows = np.concatenate(families).astype(np.int64, copy=False)
    else:
        member_rows = np.empty(0, dtype=np.int64)
    if member_rows.size and len(np.unique(member_rows)) != member_rows.size:
        raise RuntimeError("duplicate census produced overlapping exact families")
    total_family_rows = int(family_counts.sum()) if family_counts.size else 0
    unique_rows = ROW_COUNT - total_family_rows
    if total_family_rows + unique_rows != ROW_COUNT:
        raise RuntimeError("duplicate census row accounting does not close")
    return {
        "arrays": {
            "representative_rows": representative_rows,
            "family_counts": family_counts,
            "family_offsets": offsets,
            "member_rows": member_rows,
        },
        "summary": {
            "exact_family_count": int(len(families)),
            "repeated_hash_groups": repeated_hash_groups,
            "hash_collision_splits": hash_collision_splits,
            "total_rows_in_families": total_family_rows,
            "duplicated_copy_rows": int(np.maximum(family_counts - 1, 0).sum())
            if family_counts.size
            else 0,
            "unique_rows": unique_rows,
            "family_size_histogram": {
                str(size): int(count) for size, count in sorted(Counter(family_counts.tolist()).items())
            },
            "fraction_rows_in_families": total_family_rows / ROW_COUNT,
            "wall_seconds": time.monotonic() - started,
        },
    }


def _family_hashes(
    representative_rows: np.ndarray,
    members: Sequence[MaterializedMember],
) -> tuple[np.ndarray, np.ndarray]:
    h0 = np.empty(len(representative_rows), dtype="<u8")
    h1 = np.empty(len(representative_rows), dtype="<u8")
    cache: dict[int, Any] = {}
    for index, row in enumerate(representative_rows.tolist()):
        h0[index], h1[index] = _digest128(_row_bytes(int(row), members, cache))
    return h0, h1


def known_r0019_status(
    arrays: dict[str, np.ndarray], members: Sequence[MaterializedMember]
) -> list[dict[str, Any]]:
    member_rows = arrays["member_rows"]
    offsets = arrays["family_offsets"]
    representatives = arrays["representative_rows"]
    family_counts = arrays["family_counts"]
    row_to_family = {
        int(row): family_index
        for family_index in range(len(representatives))
        for row in member_rows[offsets[family_index] : offsets[family_index + 1]].tolist()
    }
    status: list[dict[str, Any]] = []
    cache: dict[int, Any] = {}
    for row, expected_component_count in zip(
        KNOWN_R0019_ROWS, KNOWN_R0019_COMPONENT_COUNTS, strict=True
    ):
        family_index = row_to_family.get(row)
        h0, h1 = _digest128(_row_bytes(int(row), members, cache))
        if family_index is None:
            status.append(
                {
                    "row": row,
                    "status": "singleton_not_reported_as_family",
                    "global_exact_family_count": 1,
                    "r0019_component_family_count": expected_component_count,
                    "row_hash128_hex": f"{h0:016x}{h1:016x}",
                }
            )
        else:
            status.append(
                {
                    "row": row,
                    "status": "reported_family",
                    "representative_row": int(representatives[family_index]),
                    "global_exact_family_count": int(family_counts[family_index]),
                    "r0019_component_family_count": expected_component_count,
                    "row_hash128_hex": f"{h0:016x}{h1:016x}",
                }
            )
    return status


def save_cap_npz(
    path: str,
    *,
    representative_rows: np.ndarray,
    family_counts: np.ndarray,
    member_rows: np.ndarray,
    family_offsets: np.ndarray,
    census_identity_sha256: str,
) -> str:
    exclusions: list[np.ndarray] = []
    for index in range(len(representative_rows)):
        rows = member_rows[family_offsets[index] : family_offsets[index + 1]]
        exclusions.append(rows[1:])
    excluded_rows = (
        np.sort(np.concatenate(exclusions)).astype(np.int64)
        if exclusions
        else np.empty(0, dtype=np.int64)
    )
    arrays = {
        "excluded_rows": excluded_rows,
        "representative_rows": representative_rows,
        "family_counts": family_counts,
    }
    payload = {
        "schema": DUPLICATE_CAP_SCHEMA,
        "row_count": ROW_COUNT,
        "fixed_edges_per_source": 15,
        "multiplicity_cap": 1,
        "positive_source_policy": "uniform-over-retained-rows-and-k-neighbor-slots",
        "positive_destination_policy": "original-authenticated-graph-target-row",
        "negative_node_policy": "uniform-over-retained-rows",
        "selection": {
            "source": "round0020-global-exact-fp16-census",
            "criterion": "every exact fp16 family with count >= 2",
            "exact_embedding_families": int(len(representative_rows)),
        },
        "excluded_row_count": int(len(excluded_rows)),
        "retained_row_count": ROW_COUNT - int(len(excluded_rows)),
        "effective_positive_edges": (ROW_COUNT - int(len(excluded_rows))) * 15,
        "array_sha256": {
            name: ordered_array_sha256(value) for name, value in arrays.items()
        },
        "inputs": {
            "census_identity_sha256": census_identity_sha256,
        },
    }
    metadata = _seal(payload)
    atomic_save_new_npz(
        path,
        immutable=True,
        metadata=np.asarray(canonical_json(metadata)),
        **arrays,
    )
    return path


def baseline_diagnostics(
    coordinates: Any,
    *,
    arrays: dict[str, np.ndarray],
    coordinate_signature: dict[str, Any],
    top_n: int = 50,
    sample_seed: int = 20260719,
    sample_size: int = 200_000,
) -> dict[str, Any]:
    family_counts = arrays["family_counts"]
    representative_rows = arrays["representative_rows"]
    member_rows = arrays["member_rows"]
    if len(family_counts):
        top = np.lexsort((representative_rows, -family_counts))[:top_n]
    else:
        top = np.empty(0, dtype=np.int64)
    rng = np.random.RandomState(sample_seed)
    sample_ids = np.sort(
        rng.choice(ROW_COUNT, sample_size, replace=False).astype(np.int64)
    )
    sample_ids = sample_ids[~np.isin(sample_ids, member_rows, assume_unique=True)]
    sample = np.asarray(coordinates[sample_ids], dtype=np.float64)
    reps = np.asarray(coordinates[representative_rows[top]], dtype=np.float64)
    if (
        sample.shape != (len(sample_ids), 2)
        or reps.shape != (len(top), 2)
        or not np.isfinite(sample).all()
        or not np.isfinite(reps).all()
    ):
        raise ValueError("invalid coordinate sample for R0019 global baseline")
    center = sample.mean(axis=0)
    covariance = np.cov(sample, rowvar=False)
    if covariance.shape != (2, 2) or np.linalg.det(covariance) <= 0:
        raise ValueError("R0019 baseline covariance is singular")
    inverse = np.linalg.inv(covariance)
    delta = reps - center
    distance = np.sqrt(np.einsum("ni,ij,nj->n", delta, inverse, delta))
    return {
        "schema": "r0019-global-duplicate-baseline-v1",
        "coordinate_receipt": coordinate_signature,
        "method": "fixed-sample-mahalanobis-excluding-union-of-all-census-family-rows",
        "sample_seed": sample_seed,
        "sample_size_requested": sample_size,
        "sample_size_effective": int(len(sample_ids)),
        "sample_ids_sha256": ordered_array_sha256(sample_ids),
        "excluded_union_family_rows": int(len(member_rows)),
        "reference_center": center.tolist(),
        "reference_covariance": covariance.tolist(),
        "top_family_count": int(len(top)),
        "families": [
            {
                "rank": int(rank + 1),
                "representative_row": int(representative_rows[index]),
                "family_count": int(family_counts[index]),
                "representative_coordinate": reps[rank].tolist(),
                "representative_mahalanobis": float(distance[rank]),
            }
            for rank, index in enumerate(top.tolist())
        ],
        "maximum_top_family_representative_mahalanobis": float(distance.max())
        if len(distance)
        else None,
    }


def build_duplicate_census(
    *,
    pack_manifest: str,
    output_root: str,
    coordinates: Any | None = None,
    coordinate_receipt_path: str | None = None,
) -> dict[str, Any]:
    create_fresh_directory(output_root, label="Round 0020 duplicate-census output")
    started = time.monotonic()
    manifest_signature = expected_input_signature(pack_manifest)
    members = load_materialized_members(pack_manifest)
    member_verification = verify_member_files(members)
    with tempfile.TemporaryDirectory(prefix="round0020-census-", dir=output_root) as temp:
        fingerprint_path = os.path.join(temp, "row-fingerprints.bin")
        fingerprint_receipt = write_row_fingerprints(members, fingerprint_path)
        duplicate = find_duplicate_families(fingerprint_path, members)
    arrays = duplicate["arrays"]
    family_h0, family_h1 = _family_hashes(arrays["representative_rows"], members)
    arrays["family_hash_h0"] = family_h0
    arrays["family_hash_h1"] = family_h1
    known_status = known_r0019_status(arrays, members)
    summary = duplicate["summary"]
    observed_counts = [
        int(item["global_exact_family_count"]) for item in known_status
    ]
    if observed_counts[:4] != list(KNOWN_R0019_COMPONENT_COUNTS):
        raise RuntimeError(
            f"known R0019 family counts changed: observed {observed_counts}, "
            f"expected {list(KNOWN_R0019_COMPONENT_COUNTS)}"
        )
    payload = {
        "schema": "global-duplicate-census-v1",
        "row_count": ROW_COUNT,
        "dimension": DIMENSION,
        "row_dtype": "<f2",
        "row_hash": "sha256(exact-768-byte-fp16-row) truncated to first 128 bits",
        "accepted_input_pack_sha256": ACCEPTED_INPUT_PACK_SHA256,
        "summary": summary,
        "known_r0019_rows": known_status,
        "array_sha256": {
            name: ordered_array_sha256(value) for name, value in arrays.items()
        },
        "inputs": {
            "input_pack_manifest": manifest_signature,
            "materialized_members": member_verification,
        },
    }
    census_metadata = _seal(payload)
    census_path = os.path.join(output_root, "global-duplicate-census-v1.npz")
    atomic_save_new_npz(
        census_path,
        immutable=True,
        metadata=np.asarray(canonical_json(census_metadata)),
        **arrays,
    )
    census_signature = expected_input_signature(census_path)
    cap_path = os.path.join(output_root, "global-cap-v1.npz")
    save_cap_npz(
        cap_path,
        representative_rows=arrays["representative_rows"],
        family_counts=arrays["family_counts"],
        member_rows=arrays["member_rows"],
        family_offsets=arrays["family_offsets"],
        census_identity_sha256=census_metadata["identity_sha256"],
    )
    outputs = {
        "census": expected_input_signature(census_path),
        "global_cap": expected_input_signature(cap_path),
    }
    baseline = None
    if coordinates is not None and coordinate_receipt_path is not None:
        coordinate_signature = expected_input_signature(coordinate_receipt_path)
        baseline_body = baseline_diagnostics(
            coordinates,
            arrays=arrays,
            coordinate_signature=coordinate_signature,
        )
        baseline = _seal(
            {
                **baseline_body,
                "inputs": {
                    "census_identity_sha256": census_metadata["identity_sha256"],
                    "coordinate_receipt": coordinate_signature,
                },
            }
        )
        baseline_path = os.path.join(output_root, "r0019-global-baseline.json")
        atomic_write_new_json(baseline_path, baseline, immutable=True)
        outputs["r0019_global_baseline"] = expected_input_signature(baseline_path)
    receipt_body = {
        "schema": "round0020-duplicate-census-receipt-v1",
        "round_id": "0020",
        "accepted_input_pack_sha256": ACCEPTED_INPUT_PACK_SHA256,
        "summary": summary,
        "known_r0019_rows": known_status,
        "outputs": outputs,
        "baseline_included": baseline is not None,
        "elapsed": {
            "fingerprinting": fingerprint_receipt,
            "total_wall_seconds": time.monotonic() - started,
        },
        "wall_seconds": time.monotonic() - started,
    }
    receipt = _seal(receipt_body)
    receipt_path = os.path.join(output_root, "receipt.json")
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(receipt_path)}
