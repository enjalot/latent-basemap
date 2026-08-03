"""Pure frozen-prefix extension contract for prompted-English 8M."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .artifact_identity import canonical_json, ordered_array_sha256, sha256_bytes


ROUND_ID = "0165"
CAPABILITY = "jina-document-english-first8m-frozen-prefix-population-v1"
HOST_CAPABILITY = "jina-document-english-first8m-frozen-prefix-host-fp16-v1"
SCHEMA = "round0165-prompted-english-frozen-prefix-population-v1"
PREFIX_STOP = 2_000_000
SOURCE_ROWS = 8_000_000
DIMENSION = 768


class Round0165Error(RuntimeError):
    """Raised when the frozen-prefix extension contract changes."""


def frozen_prefix_extension(
    *,
    accepted_prefix: np.ndarray,
    prompted_only_mapping: np.ndarray,
    prior_three_relation_mapping: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    prefix = np.asarray(accepted_prefix, dtype=np.int64)
    prompted = np.asarray(prompted_only_mapping, dtype=np.int64)
    prior = np.asarray(prior_three_relation_mapping, dtype=np.int64)
    for label, values in (("prefix", prefix), ("prompted", prompted), ("prior", prior)):
        if (
            values.ndim != 1
            or len(values) == 0
            or np.any(values[1:] <= values[:-1])
            or values[0] < 0
            or values[-1] >= SOURCE_ROWS
        ):
            raise Round0165Error(f"{label} mapping is malformed")
    if prefix[-1] >= PREFIX_STOP or len(prefix) != 1_993_761:
        raise Round0165Error("accepted R0113 prefix identity changed")
    extension = prompted[prompted >= PREFIX_STOP]
    mapping = np.concatenate((prefix, extension))
    if np.any(mapping[1:] <= mapping[:-1]):
        raise Round0165Error("frozen-prefix extension is not strictly ordered")
    positions = np.searchsorted(prompted, mapping)
    if np.any(positions >= len(prompted)) or not np.array_equal(prompted[positions], mapping):
        raise Round0165Error("frozen-prefix extension is not a prompted-only subset")
    dropped = np.setdiff1d(prompted, mapping, assume_unique=True)
    added = np.setdiff1d(mapping, prior, assume_unique=True)
    prior_positions = np.searchsorted(mapping, prior)
    if (
        not np.all(prior_positions < len(mapping))
        or not np.array_equal(mapping[prior_positions], prior)
        or len(dropped) == 0
        or np.any(dropped >= PREFIX_STOP)
        or len(added) == 0
        or not np.array_equal(mapping[mapping < PREFIX_STOP], prefix)
    ):
        raise Round0165Error("frozen-prefix extension lineage did not close")
    excluded = np.setdiff1d(
        np.arange(SOURCE_ROWS, dtype=np.int64), mapping, assume_unique=True
    )
    report = {
        "selection_rule": (
            "byte-exact accepted R0113 representatives for canonical rows below 2M; "
            "R0164 exact-source-text-plus-Document-fp16 representatives at rows >=2M"
        ),
        "prefix_rows": len(prefix),
        "extension_rows": len(extension),
        "retained_rows": len(mapping),
        "excluded_rows": len(excluded),
        "dropped_prompted_only_prefix_rows": len(dropped),
        "added_over_r0163_rows": len(added),
        "mapping_ordered_sha256": ordered_array_sha256(mapping),
        "excluded_ordered_sha256": ordered_array_sha256(excluded),
        "dropped_ordered_sha256": ordered_array_sha256(dropped),
        "added_ordered_sha256": ordered_array_sha256(added),
    }
    return mapping, excluded, dropped, added, positions, report


def population_identity(*, mapping: np.ndarray, excluded: np.ndarray) -> str:
    body = {
        "schema": "round0165-frozen-prefix-population-identity-v1",
        "source_rows": SOURCE_ROWS,
        "prefix_stop": PREFIX_STOP,
        "dimension": DIMENSION,
        "dtype": "<f2",
        "selection_law": "frozen-R0113-prefix-plus-R0164-prompted-only-extension",
        "mapping_ordered_sha256": ordered_array_sha256(mapping),
        "excluded_ordered_sha256": ordered_array_sha256(excluded),
    }
    return sha256_bytes(canonical_json(body))
