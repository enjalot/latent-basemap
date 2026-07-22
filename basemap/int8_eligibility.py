"""Streaming exact-row eligibility census for the 150M MiniLM int8 universe.

The trainer consumes one signed-int8 row plus one fp16 scale per source row.
This module therefore defines exact duplicates over that complete encoded row,
not over only the int8 payload.  Hashes are only an acceleration structure:
every repeated fingerprint is verified against the exact 386 encoded bytes.
"""
from __future__ import annotations

import json
import os
import resource
import time
from collections import Counter
from typing import Any

import numpy as np
from numba import njit, prange

from .artifact_identity import (
    canonical_json,
    expected_input_signature,
    ordered_array_sha256,
    sha256_bytes,
)
from .output_safety import atomic_save_new_npz, atomic_write_new_json, create_fresh_directory


SCHEMA = "minilm-int8-row-eligibility-v1"
RECEIPT_SCHEMA = "round0033-int8-eligibility-receipt-v1"
DEFAULT_ROWS = 150_000_000
DEFAULT_DIMENSION = 384
R0025_MANIFEST_SHA256 = "38c3847f2811725d571d4861a74864598faa4c76f56caf81a5d3a89cdb4a3f7d"
R0025_CAPABILITY_IDENTITY = "f9d275573ec1b981bf7421c953b399e92aaeb906141a2ef5b46dcafe9e881738"
R0025_INT8_SHA256 = "2171e4bf3c21e7156435b4b4021ca62b2ef8a57d9404b2764e6e968d210b7090"
R0025_SCALES_SHA256 = "d282d4f5a5abbe17e981d957fce1cd9e227cbd67aa3262803542d496dbbecb49"

FINGERPRINT_DTYPE = np.dtype([("h0", "<u8"), ("h1", "<u8"), ("row", "<u4")])


def _seal(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}


@njit(parallel=True, cache=True)
def _fingerprint_kernel(
    encoded: np.ndarray,
    scale_bits: np.ndarray,
    out_h0: np.ndarray,
    out_h1: np.ndarray,
    zero_mask: np.ndarray,
) -> None:
    """Compute two deterministic 64-bit fingerprints and the all-zero mask.

    The pair is not treated as proof of equality.  Exact encoded-byte checks
    below make collisions harmless; the fingerprints only make candidates
    cheap to locate.
    """
    rows, dimension = encoded.shape
    for row in prange(rows):
        h0 = np.uint64(1469598103934665603)
        h1 = np.uint64(7809847782465536322)
        is_zero = True
        for column in range(dimension):
            value = np.uint64(np.uint8(encoded[row, column]))
            is_zero = is_zero and value == 0
            h0 = (h0 ^ value) * np.uint64(1099511628211)
            h1 = (h1 ^ (value + np.uint64(column + 1))) * np.uint64(
                14029467366897019727
            )
        scale = np.uint64(scale_bits[row])
        h0 = (h0 ^ (scale & np.uint64(255))) * np.uint64(1099511628211)
        h0 = (h0 ^ (scale >> np.uint64(8))) * np.uint64(1099511628211)
        h1 = (h1 ^ scale) * np.uint64(1609587929392839161)
        out_h0[row] = h0
        out_h1[row] = h1
        zero_mask[row] = is_zero


def fingerprint_encoded_rows(
    encoded: np.ndarray,
    scale_bits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sortable fingerprint records and an all-zero-encoded-row mask."""
    if encoded.ndim != 2 or encoded.dtype != np.dtype("int8"):
        raise ValueError("encoded input must be a two-dimensional int8 array")
    if scale_bits.shape != (encoded.shape[0],) or scale_bits.dtype != np.dtype("<u2"):
        raise ValueError("scale bits must be one little-endian uint16 per row")
    if encoded.shape[0] > np.iinfo(np.uint32).max:
        raise ValueError("fingerprint row IDs require at most uint32 rows")
    records = np.empty(encoded.shape[0], dtype=FINGERPRINT_DTYPE)
    zero_mask = np.empty(encoded.shape[0], dtype=np.bool_)
    records["row"] = np.arange(encoded.shape[0], dtype=np.uint32)
    _fingerprint_kernel(
        encoded,
        scale_bits,
        records["h0"],
        records["h1"],
        zero_mask,
    )
    return records, zero_mask


def _exact_key(encoded: np.ndarray, scale_bits: np.ndarray, row: int) -> bytes:
    return encoded[row].tobytes(order="C") + np.asarray(
        scale_bits[row], dtype="<u2"
    ).tobytes()


def find_exact_families(
    records: np.ndarray,
    encoded: np.ndarray,
    scale_bits: np.ndarray,
    zero_mask: np.ndarray,
) -> dict[str, Any]:
    """Sort fingerprints and byte-verify every nonzero repeated candidate."""
    started = time.monotonic()
    records.sort(order=("h0", "h1"), kind="stable")
    sort_seconds = time.monotonic() - started

    same = np.empty(max(len(records) - 1, 0), dtype=np.bool_)
    if len(records) > 1:
        same[:] = (
            (records["h0"][1:] == records["h0"][:-1])
            & (records["h1"][1:] == records["h1"][:-1])
        )
    repeated = np.flatnonzero(same)
    starts = (
        repeated[np.r_[True, repeated[1:] != repeated[:-1] + 1]]
        if repeated.size
        else np.empty(0, dtype=np.int64)
    )

    families: list[np.ndarray] = []
    repeated_fingerprint_groups = 0
    collision_splits = 0
    zero_candidates_skipped = 0
    for raw_start in starts.tolist():
        start = int(raw_start)
        stop = start + 2
        while stop - 1 < len(same) and same[stop - 1]:
            stop += 1
        candidate_rows = np.sort(records["row"][start:stop].astype(np.int64))
        nonzero_rows = candidate_rows[~zero_mask[candidate_rows]]
        zero_candidates_skipped += int(len(candidate_rows) - len(nonzero_rows))
        if len(nonzero_rows) < 2:
            continue
        repeated_fingerprint_groups += 1
        exact: dict[bytes, list[int]] = {}
        for row in nonzero_rows.tolist():
            exact.setdefault(_exact_key(encoded, scale_bits, row), []).append(row)
        verified = [
            np.asarray(group, dtype=np.int64)
            for group in exact.values()
            if len(group) >= 2
        ]
        if len(exact) > 1:
            collision_splits += len(exact) - 1
        families.extend(verified)

    del same, repeated, starts
    families.sort(key=lambda rows: int(rows[0]))
    representatives = np.asarray([rows[0] for rows in families], dtype=np.int64)
    counts = np.asarray([len(rows) for rows in families], dtype=np.int64)
    offsets = np.zeros(len(families) + 1, dtype=np.int64)
    if len(families):
        offsets[1:] = np.cumsum(counts, dtype=np.int64)
        member_rows = np.concatenate(families).astype(np.int64, copy=False)
    else:
        member_rows = np.empty(0, dtype=np.int64)
    duplicate_excluded = np.concatenate([rows[1:] for rows in families]) if families else np.empty(0, dtype=np.int64)
    duplicate_excluded = np.sort(duplicate_excluded).astype(np.int64, copy=False)
    zero_rows = np.flatnonzero(zero_mask).astype(np.int64, copy=False)
    excluded_rows = np.sort(np.concatenate((zero_rows, duplicate_excluded))).astype(
        np.int64, copy=False
    )
    if len(excluded_rows) != len(np.unique(excluded_rows)):
        raise RuntimeError("zero and duplicate exclusions overlap or repeat")
    if len(member_rows) != len(np.unique(member_rows)):
        raise RuntimeError("exact duplicate families overlap")

    row_count = int(encoded.shape[0])
    rows_in_families = int(counts.sum()) if len(counts) else 0
    unique_nonzero_rows = row_count - len(zero_rows) - rows_in_families
    retained_rows = row_count - len(excluded_rows)
    if unique_nonzero_rows + len(families) != retained_rows:
        raise RuntimeError("eligibility row accounting does not close")
    arrays = {
        "zero_rows": zero_rows,
        "excluded_rows": excluded_rows,
        "duplicate_excluded_rows": duplicate_excluded,
        "representative_rows": representatives,
        "family_counts": counts,
        "family_offsets": offsets,
        "member_rows": member_rows,
    }
    return {
        "arrays": arrays,
        "summary": {
            "row_count": row_count,
            "zero_row_count": int(len(zero_rows)),
            "exact_nonzero_family_count": int(len(families)),
            "rows_in_exact_nonzero_families": rows_in_families,
            "duplicate_copy_rows_excluded": int(len(duplicate_excluded)),
            "excluded_row_count": int(len(excluded_rows)),
            "retained_row_count": retained_rows,
            "unique_nonzero_rows": unique_nonzero_rows,
            "fraction_excluded": len(excluded_rows) / row_count,
            "family_size_histogram": {
                str(size): int(count)
                for size, count in sorted(Counter(counts.tolist()).items())
            },
            "repeated_fingerprint_groups": repeated_fingerprint_groups,
            "fingerprint_collision_splits": collision_splits,
            "zero_fingerprint_candidates_skipped": zero_candidates_skipped,
            "sort_and_verify_wall_seconds": time.monotonic() - started,
            "sort_wall_seconds": sort_seconds,
        },
    }


def _validate_r0025_manifest(
    manifest_path: str,
    *,
    rows: int,
    dimension: int,
    int8_path: str,
    scales_path: str,
    enforce_registered_identity: bool,
) -> dict[str, Any]:
    signature = expected_input_signature(manifest_path)
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    universe = manifest.get("universes", {}).get("minilm-int8-150m", {})
    expected = {
        "manifest_sha256": R0025_MANIFEST_SHA256,
        "identity_sha256": R0025_CAPABILITY_IDENTITY,
        "int8_sha256": R0025_INT8_SHA256,
        "scales_sha256": R0025_SCALES_SHA256,
    }
    if enforce_registered_identity and (
        signature["sha256"] != expected["manifest_sha256"]
        or manifest.get("identity_sha256") != expected["identity_sha256"]
        or universe.get("rows") != rows
        or universe.get("dimension") != dimension
        or os.path.realpath(universe.get("int8", {}).get("canonical_path", ""))
        != os.path.realpath(int8_path)
        or os.path.realpath(universe.get("scales", {}).get("canonical_path", ""))
        != os.path.realpath(scales_path)
        or universe.get("int8", {}).get("sha256") != expected["int8_sha256"]
        or universe.get("scales", {}).get("sha256") != expected["scales_sha256"]
    ):
        raise ValueError("R0025 150M int8 capability identity changed")
    return {"signature": signature, "capability_identity_sha256": manifest.get("identity_sha256")}


def build_int8_eligibility_census(
    *,
    manifest_path: str,
    int8_path: str,
    scales_path: str,
    output_root: str,
    rows: int = DEFAULT_ROWS,
    dimension: int = DEFAULT_DIMENSION,
    enforce_registered_identity: bool = True,
) -> dict[str, Any]:
    """Build one immutable eligibility/cap artifact for the encoded 150M rows."""
    output_root = os.path.realpath(output_root)
    create_fresh_directory(output_root, label="Round 0033 eligibility output")
    started = time.monotonic()
    manifest = _validate_r0025_manifest(
        manifest_path,
        rows=rows,
        dimension=dimension,
        int8_path=int8_path,
        scales_path=scales_path,
        enforce_registered_identity=enforce_registered_identity,
    )
    int8_signature = expected_input_signature(int8_path)
    scales_signature = expected_input_signature(scales_path)
    if enforce_registered_identity and (
        int8_signature["sha256"] != R0025_INT8_SHA256
        or scales_signature["sha256"] != R0025_SCALES_SHA256
    ):
        raise ValueError("R0025 150M int8 bytes changed")
    if int8_signature["bytes"] != rows * dimension or scales_signature["bytes"] != rows * 2:
        raise ValueError("encoded 150M file sizes do not match row geometry")

    encoded = np.memmap(int8_path, dtype=np.int8, mode="r", shape=(rows, dimension))
    scales = np.memmap(scales_path, dtype="<u2", mode="r", shape=(rows,))
    fingerprint_started = time.monotonic()
    records, zero_mask = fingerprint_encoded_rows(encoded, scales)
    fingerprint_seconds = time.monotonic() - fingerprint_started
    found = find_exact_families(records, encoded, scales, zero_mask)
    del records, zero_mask
    arrays = found["arrays"]
    summary = found["summary"]

    payload = {
        "schema": SCHEMA,
        "round_id": "0033",
        "universe": "minilm-int8-150m",
        "row_count": rows,
        "dimension": dimension,
        "encoded_row_contract": "384 signed-int8 bytes followed by exact little-endian fp16 scale bits",
        "zero_policy": "exclude every row whose 384-byte signed-int8 payload is all zero",
        "duplicate_policy": "retain lowest row id from every exact nonzero encoded family; exclude the rest",
        "positive_source_policy": "uniform-over-retained-source-rows-and-fixed-k-slots",
        "negative_node_policy": "uniform-over-retained-rows",
        "positive_destination_policy": "successor-must-drop-edges-to-excluded-invalid-zero-destinations",
        "summary": summary,
        "array_sha256": {
            name: ordered_array_sha256(value) for name, value in arrays.items()
        },
        "inputs": {
            "r0025_manifest": manifest,
            "int8": int8_signature,
            "scales": scales_signature,
        },
        "fingerprint": {
            "algorithm": "paired-64-bit-fnv-derived-candidate-index",
            "proof_semantics": "candidate-only; every repeated fingerprint byte-verified",
        },
    }
    metadata = _seal(payload)
    capability_path = os.path.join(output_root, "minilm-150m-row-eligibility-v1.npz")
    atomic_save_new_npz(
        capability_path,
        immutable=True,
        metadata=np.asarray(canonical_json(metadata)),
        **arrays,
    )
    capability_signature = expected_input_signature(capability_path)
    receipt_body = {
        "schema": RECEIPT_SCHEMA,
        "round_id": "0033",
        "capability": capability_signature,
        "capability_identity_sha256": metadata["identity_sha256"],
        "summary": summary,
        "timing": {
            "fingerprint_wall_seconds": fingerprint_seconds,
            "total_wall_seconds": time.monotonic() - started,
        },
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2),
    }
    receipt = _seal(receipt_body)
    receipt_path = os.path.join(output_root, "receipt.json")
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(receipt_path)}


def load_int8_eligibility(
    path: str,
    *,
    expected_sha256: str,
    row_count: int,
) -> dict[str, Any]:
    """Fail closed when a future trainer consumes the R0033 capability."""
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise ValueError("int8 eligibility artifact SHA-256 changed")
    with np.load(path, allow_pickle=False) as archive:
        expected_names = {
            "metadata",
            "zero_rows",
            "excluded_rows",
            "duplicate_excluded_rows",
            "representative_rows",
            "family_counts",
            "family_offsets",
            "member_rows",
        }
        if set(archive.files) != expected_names:
            raise ValueError("int8 eligibility artifact members changed")
        raw = archive["metadata"].item()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        metadata = json.loads(str(raw))
        arrays = {
            name: np.asarray(archive[name], dtype=np.int64)
            for name in expected_names
            if name != "metadata"
        }
    body = {key: value for key, value in metadata.items() if key != "identity_sha256"}
    array_hashes = {name: ordered_array_sha256(value) for name, value in arrays.items()}
    excluded = arrays["excluded_rows"]
    zero = arrays["zero_rows"]
    duplicate = arrays["duplicate_excluded_rows"]
    counts = arrays["family_counts"]
    offsets = arrays["family_offsets"]
    members = arrays["member_rows"]
    if (
        metadata.get("schema") != SCHEMA
        or metadata.get("row_count") != row_count
        or metadata.get("identity_sha256") != sha256_bytes(canonical_json(body))
        or metadata.get("array_sha256") != array_hashes
        or not np.array_equal(excluded, np.unique(excluded))
        or not np.array_equal(zero, np.unique(zero))
        or not np.array_equal(duplicate, np.unique(duplicate))
        or np.intersect1d(zero, duplicate).size
        or not np.array_equal(excluded, np.sort(np.concatenate((zero, duplicate))))
        or len(offsets) != len(counts) + 1
        or offsets[0] != 0
        or offsets[-1] != len(members)
        or not np.array_equal(np.diff(offsets), counts)
        or np.any(counts < 2)
        or metadata.get("summary", {}).get("excluded_row_count") != len(excluded)
        or metadata.get("summary", {}).get("retained_row_count") != row_count - len(excluded)
        or (len(excluded) and (excluded[0] < 0 or excluded[-1] >= row_count))
    ):
        raise ValueError("int8 eligibility artifact content is invalid")
    return {"signature": signature, "metadata": metadata, **arrays}
