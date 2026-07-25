"""Balanced 60M MiniLM substrate primitives for Round 0049.

The released 150M MiniLM universe is laid out as 50M contiguous rows from
FineWeb, RedPajama, and Pile.  The next scale rung is the first 20M rows from
each corpus.  This module makes the non-contiguous 150M-to-compact-60M mapping
explicit and derives duplicate families *within that subset*.  A global
representative outside the subset must not make the only in-subset copy
ineligible.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .artifact_identity import (
    canonical_json,
    expected_input_signature,
    ordered_array_sha256,
    sha256_bytes,
)
from .int8_eligibility import SCHEMA as ELIGIBILITY_SCHEMA
from .output_safety import atomic_save_new_npz, atomic_write_new_json


ROUND_ID = "0049"
SOURCE_ROWS = 150_000_000
ROW_COUNT = 60_000_000
DIMENSION = 384
K = 15
CORPUS_INTERVALS = (
    (0, 20_000_000),
    (50_000_000, 70_000_000),
    (100_000_000, 120_000_000),
)
SOURCE_INT8_PATH = (
    "/data/latent-basemap/runs/round-0025/queue/artifacts/int8-shards/"
    "minilm-int8-150m/embeddings.i8"
)
SOURCE_SCALES_PATH = (
    "/data/latent-basemap/runs/round-0025/queue/artifacts/int8-shards/"
    "minilm-int8-150m/scales.f16"
)
SOURCE_INT8_SHA256 = (
    "2171e4bf3c21e7156435b4b4021ca62b2ef8a57d9404b2764e6e968d210b7090"
)
SOURCE_SCALES_SHA256 = (
    "d282d4f5a5abbe17e981d957fce1cd9e227cbd67aa3262803542d496dbbecb49"
)
SOURCE_ELIGIBILITY_PATH = (
    "/data/latent-basemap/runs/round-0033/queue/artifacts/eligibility/"
    "minilm-150m-row-eligibility-v1.npz"
)
SOURCE_ELIGIBILITY_SHA256 = (
    "cd9738d1cb35b7847923ec24e343583ac91dea4d76381ec28c8c2c8bf6412aca"
)
INDEX_PATH = "/data/checkpoints/pumap/faiss_ivf_pq_150m.index"
INDEX_SHA256 = (
    "7ed8ba062baf148b9b076f84c0089849ddb42610f0566a7c197f4c80852893c1"
)


class Round0049Error(RuntimeError):
    """The registered balanced-60M substrate contract was violated."""


def _seal(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    return {**value, "identity_sha256": sha256_bytes(canonical_json(value))}


def _normalize_intervals(
    intervals: Sequence[tuple[int, int]],
    *,
    source_rows: int,
) -> tuple[tuple[int, int], ...]:
    normalized = tuple((int(start), int(stop)) for start, stop in intervals)
    if (
        not normalized
        or any(start < 0 or stop <= start or stop > source_rows
               for start, stop in normalized)
        or any(normalized[index - 1][1] > normalized[index][0]
               for index in range(1, len(normalized)))
    ):
        raise ValueError("source intervals must be ordered, disjoint, and in range")
    return normalized


def compact_to_global(
    rows: Any,
    *,
    intervals: Sequence[tuple[int, int]] = CORPUS_INTERVALS,
    source_rows: int = SOURCE_ROWS,
) -> np.ndarray:
    """Map compact subset IDs to their exact global 150M IDs."""
    spans = _normalize_intervals(intervals, source_rows=source_rows)
    value = np.asarray(rows, dtype=np.int64)
    total = sum(stop - start for start, stop in spans)
    if np.any(value < 0) or np.any(value >= total):
        raise Round0049Error("compact row is outside the balanced subset")
    output = np.empty(value.shape, dtype=np.int64)
    cursor = 0
    for start, stop in spans:
        width = stop - start
        selected = (value >= cursor) & (value < cursor + width)
        output[selected] = start + (value[selected] - cursor)
        cursor += width
    return output


def global_to_compact(
    rows: Any,
    *,
    intervals: Sequence[tuple[int, int]] = CORPUS_INTERVALS,
    source_rows: int = SOURCE_ROWS,
) -> np.ndarray:
    """Map global IDs to compact IDs, returning -1 outside the subset."""
    spans = _normalize_intervals(intervals, source_rows=source_rows)
    value = np.asarray(rows, dtype=np.int64)
    if np.any(value < 0) or np.any(value >= source_rows):
        raise Round0049Error("global row is outside the source universe")
    output = np.full(value.shape, -1, dtype=np.int64)
    cursor = 0
    for start, stop in spans:
        selected = (value >= start) & (value < stop)
        output[selected] = cursor + (value[selected] - start)
        cursor += stop - start
    return output


def _read_metadata(raw: Any) -> dict[str, Any]:
    value = raw.item() if isinstance(raw, np.ndarray) else raw
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    parsed = json.loads(str(value))
    if not isinstance(parsed, dict):
        raise Round0049Error("eligibility metadata is not an object")
    return parsed


def _validate_source_eligibility(
    path: str = SOURCE_ELIGIBILITY_PATH,
    *,
    expected_sha256: str = SOURCE_ELIGIBILITY_SHA256,
    source_rows: int = SOURCE_ROWS,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise Round0049Error("R0033 eligibility bytes changed")
    with np.load(path, allow_pickle=False) as archive:
        expected = {
            "metadata",
            "zero_rows",
            "excluded_rows",
            "duplicate_excluded_rows",
            "duplicate_representative_rows",
            "representative_rows",
            "family_counts",
            "family_offsets",
            "member_rows",
        }
        if set(archive.files) != expected:
            raise Round0049Error("R0033 eligibility members changed")
        metadata = _read_metadata(archive["metadata"])
        arrays = {
            name: np.asarray(archive[name], dtype=np.int64)
            for name in expected
            if name != "metadata"
        }
    body = {
        key: value for key, value in metadata.items()
        if key != "identity_sha256"
    }
    if (
        metadata.get("schema") != ELIGIBILITY_SCHEMA
        or int(metadata.get("row_count", -1)) != source_rows
        or metadata.get("identity_sha256")
        != sha256_bytes(canonical_json(body))
        or metadata.get("array_sha256")
        != {
            name: ordered_array_sha256(value)
            for name, value in arrays.items()
        }
    ):
        raise Round0049Error("R0033 eligibility identity changed")
    counts = arrays["family_counts"]
    offsets = arrays["family_offsets"]
    members = arrays["member_rows"]
    if (
        len(offsets) != len(counts) + 1
        or offsets[0] != 0
        or offsets[-1] != len(members)
        or not np.array_equal(np.diff(offsets), counts)
        or np.any(counts < 2)
    ):
        raise Round0049Error("R0033 family geometry changed")
    return metadata, arrays, signature


def derive_subset_eligibility_arrays(
    source_arrays: Mapping[str, np.ndarray],
    *,
    intervals: Sequence[tuple[int, int]] = CORPUS_INTERVALS,
    source_rows: int = SOURCE_ROWS,
) -> dict[str, np.ndarray]:
    """Re-form exact families after restricting the global row universe."""
    counts = np.asarray(source_arrays["family_counts"], dtype=np.int64)
    offsets = np.asarray(source_arrays["family_offsets"], dtype=np.int64)
    members = np.asarray(source_arrays["member_rows"], dtype=np.int64)
    family_ids = np.repeat(
        np.arange(len(counts), dtype=np.int64),
        counts,
    )
    compact_members = global_to_compact(
        members,
        intervals=intervals,
        source_rows=source_rows,
    )
    inside = compact_members >= 0
    selected_counts = np.bincount(
        family_ids[inside],
        minlength=len(counts),
    ).astype(np.int64, copy=False)
    qualifying = selected_counts >= 2
    member_mask = inside & qualifying[family_ids]
    subset_members = compact_members[member_mask]
    subset_family_ids = family_ids[member_mask]
    subset_counts = selected_counts[qualifying]
    subset_offsets = np.zeros(len(subset_counts) + 1, dtype=np.int64)
    subset_offsets[1:] = np.cumsum(subset_counts, dtype=np.int64)
    if (
        len(subset_members) != int(subset_counts.sum())
        or (
            len(subset_family_ids)
            and np.any(subset_family_ids[1:] < subset_family_ids[:-1])
        )
    ):
        raise Round0049Error("subset family grouping lost source ordering")
    # R0033 families are ordered by their representative in the full 150M
    # universe.  When that representative is outside this subset, the first
    # retained member can move to a later corpus interval and that old family
    # order is no longer compact-row order.  Reorder whole variable-length
    # groups by their new in-subset representative.
    old_representatives = subset_members[subset_offsets[:-1]]
    family_order = np.argsort(old_representatives, kind="stable")
    if not np.array_equal(
        family_order,
        np.arange(len(family_order), dtype=np.int64),
    ):
        old_group = np.repeat(
            np.arange(len(subset_counts), dtype=np.int64),
            subset_counts,
        )
        new_rank = np.empty(len(subset_counts), dtype=np.int64)
        new_rank[family_order] = np.arange(
            len(subset_counts),
            dtype=np.int64,
        )
        member_order = np.lexsort(
            (subset_members, new_rank[old_group])
        )
        subset_members = subset_members[member_order]
        subset_counts = subset_counts[family_order]
        subset_offsets = np.zeros(
            len(subset_counts) + 1,
            dtype=np.int64,
        )
        subset_offsets[1:] = np.cumsum(
            subset_counts,
            dtype=np.int64,
        )
    starts = subset_offsets[:-1]
    representatives = subset_members[starts]
    is_copy = np.ones(len(subset_members), dtype=np.bool_)
    is_copy[starts] = False
    duplicate = subset_members[is_copy]
    repeated_representatives = np.repeat(representatives, subset_counts)
    duplicate_representatives = repeated_representatives[is_copy]
    order = np.argsort(duplicate, kind="stable")
    duplicate = duplicate[order]
    duplicate_representatives = duplicate_representatives[order]
    zero = global_to_compact(
        np.asarray(source_arrays["zero_rows"], dtype=np.int64),
        intervals=intervals,
        source_rows=source_rows,
    )
    zero = np.sort(zero[zero >= 0])
    excluded = np.sort(np.concatenate((zero, duplicate)))
    if (
        len(excluded) != len(np.unique(excluded))
        or len(subset_members) != len(np.unique(subset_members))
        or np.intersect1d(representatives, excluded).size
        or np.intersect1d(zero, duplicate).size
    ):
        raise Round0049Error("subset family/zero exclusions overlap")
    return {
        "zero_rows": zero.astype(np.int64, copy=False),
        "excluded_rows": excluded.astype(np.int64, copy=False),
        "duplicate_excluded_rows": duplicate.astype(np.int64, copy=False),
        "duplicate_representative_rows": duplicate_representatives.astype(
            np.int64, copy=False
        ),
        "representative_rows": representatives.astype(np.int64, copy=False),
        "family_counts": subset_counts.astype(np.int64, copy=False),
        "family_offsets": subset_offsets,
        "member_rows": subset_members.astype(np.int64, copy=False),
    }


def write_subset_eligibility(
    output_path: str,
    *,
    source_path: str = SOURCE_ELIGIBILITY_PATH,
    source_sha256: str = SOURCE_ELIGIBILITY_SHA256,
    intervals: Sequence[tuple[int, int]] = CORPUS_INTERVALS,
    source_rows: int = SOURCE_ROWS,
) -> dict[str, Any]:
    """Publish one loader-compatible compact eligibility capability."""
    _metadata, source_arrays, source_signature = _validate_source_eligibility(
        source_path,
        expected_sha256=source_sha256,
        source_rows=source_rows,
    )
    spans = _normalize_intervals(intervals, source_rows=source_rows)
    arrays = derive_subset_eligibility_arrays(
        source_arrays,
        intervals=spans,
        source_rows=source_rows,
    )
    rows = sum(stop - start for start, stop in spans)
    counts = arrays["family_counts"]
    rows_in_families = int(counts.sum())
    zero_count = len(arrays["zero_rows"])
    excluded_count = len(arrays["excluded_rows"])
    retained = rows - excluded_count
    unique_nonzero = rows - zero_count - rows_in_families
    if unique_nonzero + len(counts) != retained:
        raise Round0049Error("compact eligibility accounting does not close")
    summary = {
        "row_count": rows,
        "zero_row_count": zero_count,
        "exact_nonzero_family_count": len(counts),
        "rows_in_exact_nonzero_families": rows_in_families,
        "duplicate_copy_rows_excluded": len(arrays["duplicate_excluded_rows"]),
        "excluded_row_count": excluded_count,
        "retained_row_count": retained,
        "unique_nonzero_rows": unique_nonzero,
        "fraction_excluded": excluded_count / rows,
        "family_size_histogram": {
            str(size): int(count)
            for size, count in sorted(Counter(counts.tolist()).items())
        },
        "derived_from_exact_source_families": True,
    }
    body = {
        "schema": ELIGIBILITY_SCHEMA,
        "round_id": ROUND_ID,
        "universe": "minilm-int8-balanced-60m",
        "row_count": rows,
        "dimension": DIMENSION,
        "encoded_row_contract": (
            "384 signed-int8 bytes followed by exact little-endian fp16 scale bits"
        ),
        "zero_policy": "exclude every all-zero signed-int8 payload",
        "duplicate_policy": (
            "within the balanced subset retain the lowest global row from each "
            "exact nonzero encoded family and exclude the rest"
        ),
        "positive_source_policy": "registered by the consuming training round",
        "negative_node_policy": "uniform-over-retained-compact-rows",
        "positive_destination_policy": (
            "native representative-only search; no duplicate destination enters graph"
        ),
        "summary": summary,
        "array_sha256": {
            name: ordered_array_sha256(value)
            for name, value in arrays.items()
        },
        "inputs": {
            "r0033_eligibility": source_signature,
        },
        "global_to_compact": {
            "source_rows": source_rows,
            "intervals": [list(span) for span in spans],
            "compact_rows": rows,
        },
    }
    metadata = _seal(body)
    atomic_save_new_npz(
        output_path,
        immutable=True,
        metadata=np.asarray(canonical_json(metadata)),
        **arrays,
    )
    return {
        "metadata": metadata,
        "signature": expected_input_signature(output_path),
        "arrays": arrays,
    }


def validate_substrate_manifest(
    path: str,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    signature = expected_input_signature(path)
    if expected_sha256 is not None and signature["sha256"] != expected_sha256:
        raise Round0049Error("balanced-60M substrate manifest bytes changed")
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    body = {
        key: value for key, value in manifest.items()
        if key != "identity_sha256"
    }
    if (
        manifest.get("schema") != "round0049-balanced-60m-substrate-v1"
        or manifest.get("identity_sha256")
        != sha256_bytes(canonical_json(body))
        or int(manifest.get("row_count", -1)) != ROW_COUNT
        or int(manifest.get("dimension", -1)) != DIMENSION
    ):
        raise Round0049Error("balanced-60M substrate identity changed")
    outputs = manifest.get("outputs") or {}
    for key, size in (
        ("int8", ROW_COUNT * DIMENSION),
        ("scales", ROW_COUNT * 2),
    ):
        observed = expected_input_signature(
            outputs.get(key, {}).get("canonical_path", "")
        )
        if observed != outputs.get(key) or observed["bytes"] != size:
            raise Round0049Error(f"balanced-60M {key} bytes changed")
    eligibility = expected_input_signature(
        outputs.get("eligibility", {}).get("canonical_path", "")
    )
    if eligibility != outputs.get("eligibility"):
        raise Round0049Error("balanced-60M eligibility bytes changed")
    return {"manifest": manifest, "signature": signature}
