"""Representative-only evaluation primitives for Round 0040.

The round does not retrain a map.  It removes exact duplicate copies from the
scientific row universe, keeps the first row of every non-zero exact family as
the geometry representative, and re-scores already-published coordinates.
All product rows remain addressable through the census artifact.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from .artifact_identity import (
    canonical_json,
    expected_input_signature,
    ordered_array_sha256,
    sha256_bytes,
)
from .output_safety import atomic_save_new_npz, atomic_write_new_json
from .round0036_pipeline import RetainedArrayView, RetainedRowSelector


ROUND_ID = "0040"
JINA_ROWS = 2_000_000
JINA_DIMENSION = 768
JINA_DTYPE = np.dtype("<f2")
JINA_CENSUS_SCHEMA = "round0040-jina-exact-family-census-v1"
JINA_CENSUS_RECEIPT_SCHEMA = "round0040-jina-census-receipt-v1"
FINGERPRINT_DTYPE = np.dtype(
    [("h0", "<u8"), ("h1", "<u8"), ("row", "<u4")]
)

MINILM_ROWS = 30_000_000
MINILM_DIMENSION = 384
MINILM_CAP_PATH = (
    "/data/latent-basemap/runs/round-0020/queue/artifacts/"
    "duplicate-census/global-cap-v1.npz"
)
MINILM_CAP_SHA256 = (
    "9511ceca802da603bfbfe9164f8c6ffd7006df82df17b9499d4ed33288fde7cb"
)
MINILM_CAP_RETAINED_ROWS = 29_781_758


def seal(body: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(body)
    return {**payload, "identity_sha256": sha256_bytes(canonical_json(payload))}


def validate_seal(value: Mapping[str, Any], *, label: str) -> None:
    body = {key: item for key, item in value.items()
            if key != "identity_sha256"}
    if value.get("identity_sha256") != sha256_bytes(canonical_json(body)):
        raise ValueError(f"{label} identity seal is invalid")


def _digest128(row_bytes: bytes | memoryview) -> tuple[int, int]:
    digest = hashlib.sha256(row_bytes).digest()
    return (
        int.from_bytes(digest[:8], "little"),
        int.from_bytes(digest[8:16], "little"),
    )


def _write_jina_fingerprints(
    array: np.ndarray,
    path: str,
    *,
    block_rows: int = 8192,
) -> dict[str, Any]:
    started = time.monotonic()
    row_count = int(len(array))
    row_bytes = int(array.shape[1]) * np.dtype(array.dtype).itemsize
    fingerprints = np.memmap(
        path, dtype=FINGERPRINT_DTYPE, mode="w+", shape=(row_count,)
    )
    zero_parts: list[np.ndarray] = []
    nonfinite_parts: list[np.ndarray] = []
    for start in range(0, row_count, block_rows):
        stop = min(start + block_rows, row_count)
        block = np.ascontiguousarray(array[start:stop])
        zero = np.flatnonzero(np.all(block == 0, axis=1))
        nonfinite = np.flatnonzero(~np.isfinite(block).all(axis=1))
        if len(zero):
            zero_parts.append((zero + start).astype(np.int64))
        if len(nonfinite):
            nonfinite_parts.append((nonfinite + start).astype(np.int64))
        raw = memoryview(block).cast("B")
        out = fingerprints[start:stop]
        for local in range(len(block)):
            begin = local * row_bytes
            h0, h1 = _digest128(raw[begin:begin + row_bytes])
            out["h0"][local] = h0
            out["h1"][local] = h1
            out["row"][local] = start + local
    fingerprints.flush()
    return {
        "row_hash": (
            f"sha256(exact-{row_bytes}-byte-row) truncated to first 128 bits"
        ),
        "zero_rows": (
            np.concatenate(zero_parts)
            if zero_parts else np.empty(0, dtype=np.int64)
        ),
        "nonfinite_rows": (
            np.concatenate(nonfinite_parts)
            if nonfinite_parts else np.empty(0, dtype=np.int64)
        ),
        "wall_seconds": time.monotonic() - started,
    }


def _exact_families_from_fingerprints(
    array: np.ndarray,
    fingerprint_path: str,
) -> dict[str, Any]:
    started = time.monotonic()
    row_count = int(len(array))
    fingerprints = np.memmap(
        fingerprint_path,
        dtype=FINGERPRINT_DTYPE,
        mode="r",
        shape=(row_count,),
    )
    order = np.argsort(fingerprints, order=("h0", "h1"), kind="stable")
    sorted_fp = np.asarray(fingerprints[order])
    del order, fingerprints
    same = (
        (sorted_fp["h0"][1:] == sorted_fp["h0"][:-1])
        & (sorted_fp["h1"][1:] == sorted_fp["h1"][:-1])
    )
    repeated = np.flatnonzero(same)
    starts = (
        repeated[np.r_[True, repeated[1:] != repeated[:-1] + 1]]
        if len(repeated) else np.empty(0, dtype=np.int64)
    )
    families: list[np.ndarray] = []
    repeated_hash_groups = 0
    hash_collision_splits = 0
    for start in starts.tolist():
        stop = int(start) + 1
        while stop < row_count and (
            sorted_fp["h0"][stop] == sorted_fp["h0"][start]
            and sorted_fp["h1"][stop] == sorted_fp["h1"][start]
        ):
            stop += 1
        rows = np.sort(sorted_fp["row"][start:stop].astype(np.int64))
        exact: dict[bytes, list[int]] = {}
        for row in rows.tolist():
            key = np.ascontiguousarray(
                array[int(row)]
            ).tobytes()
            exact.setdefault(key, []).append(int(row))
        split = [
            np.asarray(sorted(group), dtype=np.int64)
            for group in exact.values() if len(group) >= 2
        ]
        split.sort(key=lambda value: int(value[0]))
        repeated_hash_groups += 1
        hash_collision_splits += max(0, len(split) - 1)
        families.extend(split)
    del sorted_fp, same
    families.sort(key=lambda value: int(value[0]))
    representatives = np.asarray(
        [int(rows[0]) for rows in families], dtype=np.int64
    )
    counts = np.asarray([len(rows) for rows in families], dtype=np.int64)
    offsets = np.zeros(len(families) + 1, dtype=np.int64)
    if families:
        offsets[1:] = np.cumsum(counts, dtype=np.int64)
        members = np.concatenate(families).astype(np.int64, copy=False)
    else:
        members = np.empty(0, dtype=np.int64)
    if len(members) != len(np.unique(members)):
        raise RuntimeError("Jina exact-family census produced overlapping rows")
    return {
        "representative_rows": representatives,
        "family_counts": counts,
        "family_offsets": offsets,
        "member_rows": members,
        "repeated_hash_groups": repeated_hash_groups,
        "hash_collision_splits": hash_collision_splits,
        "wall_seconds": time.monotonic() - started,
    }


def build_jina_census(
    *,
    source_path: str,
    output_root: str,
    expected_source_sha256: str,
) -> dict[str, Any]:
    """Build the exact 2M Jina family census and representative selector."""
    started = time.monotonic()
    source = expected_input_signature(source_path)
    if source["sha256"] != expected_source_sha256:
        raise ValueError("Round 0040 Jina source bytes changed")
    array = np.load(source_path, mmap_mode="r", allow_pickle=False)
    if (
        array.shape != (JINA_ROWS, JINA_DIMENSION)
        or array.dtype != JINA_DTYPE
        or not array.flags.c_contiguous
    ):
        raise ValueError(
            f"Round 0040 Jina source header changed: {array.shape} {array.dtype}"
        )
    with tempfile.TemporaryDirectory(
        prefix="round0040-jina-census-", dir=output_root
    ) as temporary:
        fingerprints_path = os.path.join(temporary, "row-fingerprints.bin")
        fingerprint = _write_jina_fingerprints(array, fingerprints_path)
        families = _exact_families_from_fingerprints(
            array, fingerprints_path
        )

    family_counts = families["family_counts"]
    offsets = families["family_offsets"]
    members = families["member_rows"]
    duplicate_copies = (
        np.concatenate([
            members[offsets[index] + 1:offsets[index + 1]]
            for index in range(len(family_counts))
        ])
        if len(family_counts) else np.empty(0, dtype=np.int64)
    )
    invalid_rows = np.union1d(
        fingerprint["zero_rows"], fingerprint["nonfinite_rows"]
    ).astype(np.int64)
    excluded_rows = np.union1d(
        duplicate_copies, invalid_rows
    ).astype(np.int64)
    family_h0 = np.empty(len(family_counts), dtype="<u8")
    family_h1 = np.empty(len(family_counts), dtype="<u8")
    for index, row in enumerate(families["representative_rows"].tolist()):
        family_h0[index], family_h1[index] = _digest128(
            np.ascontiguousarray(array[int(row)], dtype=JINA_DTYPE).tobytes()
        )
    arrays = {
        "representative_rows": families["representative_rows"],
        "family_counts": family_counts,
        "family_offsets": offsets,
        "member_rows": members,
        "family_hash_h0": family_h0,
        "family_hash_h1": family_h1,
        "duplicate_copy_rows": np.sort(duplicate_copies),
        "zero_rows": fingerprint["zero_rows"],
        "nonfinite_rows": fingerprint["nonfinite_rows"],
        "excluded_rows": excluded_rows,
    }
    family_rows = int(family_counts.sum()) if len(family_counts) else 0
    summary = {
        "row_count": JINA_ROWS,
        "dimension": JINA_DIMENSION,
        "exact_family_count": int(len(family_counts)),
        "rows_in_exact_families": family_rows,
        "duplicate_copy_rows": int(len(duplicate_copies)),
        "zero_rows": int(len(fingerprint["zero_rows"])),
        "nonfinite_rows": int(len(fingerprint["nonfinite_rows"])),
        "excluded_rows": int(len(excluded_rows)),
        "representative_universe_rows": JINA_ROWS - int(len(excluded_rows)),
        "largest_family": (
            int(family_counts.max()) if len(family_counts) else 1
        ),
        "fraction_rows_in_families": family_rows / JINA_ROWS,
        "family_size_histogram": {
            str(size): int(count)
            for size, count in sorted(Counter(family_counts.tolist()).items())
        },
        "repeated_hash_groups": int(families["repeated_hash_groups"]),
        "hash_collision_splits": int(families["hash_collision_splits"]),
    }
    metadata_body = {
        "schema": JINA_CENSUS_SCHEMA,
        "round_id": ROUND_ID,
        "source": source,
        "selection": {
            "geometry_unit": "one-exact-nonzero-fp16-vector",
            "representative": "minimum-global-row-id",
            "duplicate_copies": "excluded-from-scientific-universe",
            "zero_or_nonfinite": "excluded-from-scientific-universe",
            "product_rows": "preserved-by-family-membership",
        },
        "summary": summary,
        "array_sha256": {
            name: ordered_array_sha256(value)
            for name, value in arrays.items()
        },
    }
    metadata = seal(metadata_body)
    artifact_path = os.path.join(
        output_root, "jina-exact-family-census-v1.npz"
    )
    atomic_save_new_npz(
        artifact_path,
        immutable=True,
        metadata=np.asarray(canonical_json(metadata)),
        **arrays,
    )
    receipt_body = {
        "schema": JINA_CENSUS_RECEIPT_SCHEMA,
        "round_id": ROUND_ID,
        "source": source,
        "census": expected_input_signature(artifact_path),
        "census_identity_sha256": metadata["identity_sha256"],
        "summary": summary,
        "elapsed": {
            "fingerprinting_wall_seconds": fingerprint["wall_seconds"],
            "family_resolution_wall_seconds": families["wall_seconds"],
            "total_wall_seconds": time.monotonic() - started,
        },
    }
    receipt = seal(receipt_body)
    receipt_path = os.path.join(output_root, "receipt.json")
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(receipt_path)}


def load_jina_census(receipt_path: str) -> dict[str, Any]:
    with open(receipt_path, encoding="utf-8") as handle:
        receipt = json.load(handle)
    validate_seal(receipt, label="Round 0040 Jina census receipt")
    signature = receipt.get("census")
    if (
        receipt.get("schema") != JINA_CENSUS_RECEIPT_SCHEMA
        or not isinstance(signature, dict)
        or expected_input_signature(signature.get("canonical_path", ""))
        != signature
    ):
        raise ValueError("Round 0040 Jina census receipt changed")
    with np.load(signature["canonical_path"], allow_pickle=False) as archive:
        raw = archive["metadata"].item()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        metadata = json.loads(str(raw))
        arrays = {
            name: np.asarray(archive[name])
            for name in archive.files if name != "metadata"
        }
    validate_seal(metadata, label="Round 0040 Jina census")
    hashes = {
        name: ordered_array_sha256(value)
        for name, value in arrays.items()
    }
    summary = metadata.get("summary") or {}
    excluded = arrays.get("excluded_rows")
    if (
        metadata.get("schema") != JINA_CENSUS_SCHEMA
        or metadata.get("identity_sha256")
        != receipt.get("census_identity_sha256")
        or metadata.get("array_sha256") != hashes
        or not isinstance(excluded, np.ndarray)
        or excluded.dtype != np.dtype("int64")
        or not np.array_equal(excluded, np.unique(excluded))
        or summary.get("representative_universe_rows")
        != JINA_ROWS - len(excluded)
    ):
        raise ValueError("Round 0040 Jina census content changed")
    return {
        "receipt": receipt,
        "metadata": metadata,
        "arrays": arrays,
        "signature": signature,
    }


class RepresentativeRowSelector(RetainedRowSelector):
    """Generic exact representative selector with a sealed source identity."""

    def __init__(
        self,
        excluded_rows: np.ndarray,
        *,
        row_count: int,
        source: Mapping[str, Any],
        policy: str,
    ) -> None:
        super().__init__(excluded_rows, row_count=row_count)
        self.source = dict(source)
        self.policy = str(policy)

    def identity(self) -> dict[str, Any]:
        return {
            "schema": "representative-row-selector-v1",
            "row_count": self.row_count,
            "representative_count": self.retained_count,
            "excluded_count": int(len(self.excluded_rows)),
            "excluded_rows_sha256": ordered_array_sha256(
                self.excluded_rows
            ),
            "policy": self.policy,
            "source": self.source,
        }


class RepresentativeArrayView(RetainedArrayView):
    """Compact representative row view accepted by the scale scorer."""

    round0036_retained_view = False
    representative_row_view = True

    def scale_admission_identity(self) -> dict[str, Any]:
        base = (
            self.base.scientific_identity()
            if hasattr(self.base, "scientific_identity")
            else None
        )
        return seal({
            "schema": "representative-row-scale-input-v1",
            "row_count": len(self),
            "dimensions": self.shape[1],
            "base": base,
            "selector": self.selector.identity(),
        })


class CachedShardedArray:
    """Lazy ordered shards with one memmap per member and cheap sorted gathers."""

    def __init__(
        self,
        members: Sequence[Mapping[str, Any]],
        *,
        row_count: int,
        dimension: int,
        dtype: str,
    ) -> None:
        self._members: list[dict[str, Any]] = []
        self._arrays: list[np.ndarray] = []
        cursor = 0
        expected_dtype = np.dtype(dtype)
        for position, raw in enumerate(members):
            path = os.path.realpath(str(raw["path"]))
            start = int(raw["global_row_start"])
            stop = int(raw["global_row_stop"])
            if start != cursor or stop <= start:
                raise ValueError(
                    f"cached shard order changed at member {position}"
                )
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if (
                array.shape != (stop - start, dimension)
                or array.dtype != expected_dtype
            ):
                raise ValueError(
                    f"cached shard geometry changed: {path}: "
                    f"{array.shape} {array.dtype}"
                )
            signature = raw.get("signature")
            if signature is None:
                signature = expected_input_signature(path)
            self._members.append({
                "path": path,
                "global_row_start": start,
                "global_row_stop": stop,
                "signature": dict(signature),
            })
            self._arrays.append(array)
            cursor = stop
        if cursor != row_count:
            raise ValueError("cached shard coverage is incomplete")
        self.shape = (row_count, dimension)
        self.dtype = expected_dtype

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key: Any) -> np.ndarray:
        if isinstance(key, tuple):
            rows, columns = key
            return self[rows][..., columns]
        if isinstance(key, (int, np.integer)):
            return self[np.asarray([int(key)], dtype=np.int64)][0]
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            rows = np.arange(start, stop, step, dtype=np.int64)
        else:
            rows = np.asarray(key, dtype=np.int64)
        shape = rows.shape
        flat = rows.reshape(-1)
        if np.any(flat < 0) or np.any(flat >= len(self)):
            raise IndexError("cached shard row is out of range")
        order = (
            None
            if len(flat) < 2 or np.all(flat[:-1] <= flat[1:])
            else np.argsort(flat, kind="stable")
        )
        ordered = flat if order is None else flat[order]
        gathered = np.empty((len(flat), self.shape[1]), dtype=self.dtype)
        target = gathered if order is None else np.empty_like(gathered)
        copied = 0
        for member, array in zip(self._members, self._arrays):
            low = member["global_row_start"]
            high = member["global_row_stop"]
            left = int(np.searchsorted(ordered, low, side="left"))
            right = int(np.searchsorted(ordered, high, side="left"))
            if right > left:
                target[left:right] = array[ordered[left:right] - low]
                copied += right - left
        if copied != len(flat):
            raise RuntimeError("cached shard gather did not cover every row")
        if order is not None:
            gathered[order] = target
        return gathered.reshape(shape + (self.shape[1],))

    def _reduce(self, op: Any, seed: float, *, axis: int | None):
        if axis not in (None, 0):
            raise ValueError("cached shard reduction supports axis 0 or all")
        value = np.full(self.shape[1], seed, dtype=np.float32)
        for array in self._arrays:
            value = op(value, op.reduce(array, axis=0))
        return value if axis == 0 else op.reduce(value)

    def min(self, axis: int | None = None):
        return self._reduce(np.minimum, np.inf, axis=axis)

    def max(self, axis: int | None = None):
        return self._reduce(np.maximum, -np.inf, axis=axis)

    def scientific_identity(self) -> dict[str, Any]:
        return {
            "kind": "ordered_shards",
            "shape": [int(value) for value in self.shape],
            "dtype": self.dtype.str,
            "shards": [
                {
                    "position": position,
                    "name": (
                        f"shard-{position:05d}-"
                        f"{os.path.basename(member['path'])}"
                    ),
                    "bytes": int(member["signature"]["bytes"]),
                    "sha256": str(member["signature"]["sha256"]),
                }
                for position, member in enumerate(self._members)
            ],
        }


def panel_config():
    from .panel_v2 import PanelV2Config

    return PanelV2Config(
        frac=0.001,
        k_clust=(256, 1024),
        k_density=15,
        k_hit=10,
        n_anchors=10_000,
        anchor_seed=123,
        corpus_chunk=500_000,
        overselect=8,
        block_elems=500_000_000,
        rerank_byte_cap=2_000_000_000,
        rerank_scratch=3.0,
        peak_byte_cap=26_000_000_000,
    )
