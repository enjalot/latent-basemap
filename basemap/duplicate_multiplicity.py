"""Exact duplicate-multiplicity treatment artifact for the 30M MiniLM rung."""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from .artifact_identity import (
    canonical_json,
    expected_input_signature,
    ordered_array_sha256,
    sha256_bytes,
)


SCHEMA = "round0019-exact-duplicate-cap.v1"


def load_duplicate_cap(
    path: str,
    *,
    expected_sha256: str,
    row_count: int,
    fixed_edges_per_source: int,
) -> dict[str, Any]:
    """Load and fully validate a source/negative multiplicity-cap artifact."""
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise ValueError("duplicate-cap artifact SHA-256 changed")
    with np.load(path, allow_pickle=False) as archive:
        names = set(archive.files)
        expected_names = {
            "metadata",
            "excluded_rows",
            "representative_rows",
            "family_counts",
        }
        if names != expected_names:
            raise ValueError(f"duplicate-cap members changed: {sorted(names)}")
        raw_metadata = archive["metadata"].item()
        if isinstance(raw_metadata, bytes):
            raw_metadata = raw_metadata.decode("utf-8")
        metadata = json.loads(str(raw_metadata))
        excluded = np.asarray(archive["excluded_rows"], dtype=np.int64)
        representatives = np.asarray(archive["representative_rows"], dtype=np.int64)
        family_counts = np.asarray(archive["family_counts"], dtype=np.int64)

    payload = {key: metadata[key] for key in metadata if key != "identity_sha256"}
    arrays = {
        "excluded_rows": excluded,
        "representative_rows": representatives,
        "family_counts": family_counts,
    }
    hashes = {name: ordered_array_sha256(value) for name, value in arrays.items()}
    if (
        metadata.get("schema") != SCHEMA
        or metadata.get("identity_sha256") != sha256_bytes(canonical_json(payload))
        or metadata.get("array_sha256") != hashes
        or metadata.get("row_count") != row_count
        or metadata.get("fixed_edges_per_source") != fixed_edges_per_source
        or metadata.get("multiplicity_cap") != 1
        or metadata.get("positive_source_policy")
        != "uniform-over-retained-rows-and-k-neighbor-slots"
        or metadata.get("negative_node_policy") != "uniform-over-retained-rows"
        or excluded.ndim != representatives.ndim
        or excluded.ndim != family_counts.ndim
        or not np.array_equal(excluded, np.unique(excluded))
        or not np.array_equal(representatives, np.sort(representatives))
        or len(representatives) != len(family_counts)
        or np.any(family_counts < 1)
        or len(excluded) != int(np.maximum(family_counts - 1, 0).sum())
        or (len(excluded) and (excluded[0] < 0 or excluded[-1] >= row_count))
        or (
            len(representatives)
            and (representatives[0] < 0 or representatives[-1] >= row_count)
        )
        or np.intersect1d(excluded, representatives).size
        or metadata.get("excluded_row_count") != len(excluded)
        or metadata.get("retained_row_count") != row_count - len(excluded)
        or metadata.get("effective_positive_edges")
        != (row_count - len(excluded)) * fixed_edges_per_source
    ):
        raise ValueError("duplicate-cap artifact content/semantics are invalid")
    return {
        "signature": signature,
        "metadata": metadata,
        "excluded_rows": excluded,
        "representative_rows": representatives,
        "family_counts": family_counts,
    }
