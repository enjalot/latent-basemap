"""Pure contracts for the prompted-English 8M representative census."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .artifact_identity import canonical_json, ordered_array_sha256, sha256_bytes
from .round0162_prompted_english_staging import DIMENSION, DTYPE, VIEW_ROWS


ROUND_ID = "0163"
CAPABILITY = "jina-document-english-first8m-representative-population-v1"
HOST_CAPABILITY = "jina-document-english-first8m-host-fp16-v1"
SCHEMA = "round0163-prompted-english-8m-representatives-v1"
FAMILY_SOURCES = ("source_text", "raw_fp16", "document_fp16")
PROJECTION_POSITIONS = np.unique(
    np.linspace(0, DIMENSION - 1, 32, dtype=np.int64)
)


class Round0163Error(RuntimeError):
    """Raised when the Q1 representative-census contract is violated."""


def seal(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    return {**value, "identity_sha256": sha256_bytes(canonical_json(value))}


def validate_seal(value: Mapping[str, Any], *, label: str) -> None:
    body = {key: item for key, item in value.items() if key != "identity_sha256"}
    if value.get("identity_sha256") != sha256_bytes(canonical_json(body)):
        raise Round0163Error(f"{label} identity seal changed")


def union_representatives(
    families_by_source: Mapping[str, Sequence[Sequence[int]]],
    *,
    rows: int = VIEW_ROWS,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Union exact identity relations and retain the lowest canonical row.

    The family lists may overlap across source text, raw embeddings, and
    prompted embeddings.  Treating them as one transitive relation reproduces
    R0113's population law without relying on an arm-specific selector.
    """
    if set(families_by_source) != set(FAMILY_SOURCES) or rows <= 0:
        raise Round0163Error("prompted-English family sources are incomplete")
    parent: dict[int, int] = {}

    def find(row: int) -> int:
        root = parent.setdefault(row, row)
        while parent[root] != root:
            root = parent[root]
        while parent[row] != row:
            following = parent[row]
            parent[row] = root
            row = following
        return root

    def union(left: int, right: int) -> None:
        a = find(left)
        b = find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    normalized: dict[str, list[list[int]]] = {}
    for source in FAMILY_SOURCES:
        seen: set[tuple[int, ...]] = set()
        values: list[list[int]] = []
        for family in families_by_source[source]:
            members = tuple(sorted(set(int(value) for value in family)))
            if (
                len(members) < 2
                or members[0] < 0
                or members[-1] >= rows
                or members in seen
            ):
                raise Round0163Error(f"{source} exact family is malformed")
            seen.add(members)
            values.append(list(members))
            for member in members[1:]:
                union(members[0], member)
        values.sort(key=lambda family: (family[0], len(family), family))
        normalized[source] = values

    components: dict[int, list[int]] = {}
    for row in sorted(parent):
        components.setdefault(find(row), []).append(row)
    union_families = [
        sorted(family) for family in components.values() if len(family) >= 2
    ]
    union_families.sort(key=lambda family: (family[0], len(family), family))
    excluded = np.asarray(
        sorted(member for family in union_families for member in family[1:]),
        dtype=np.int64,
    )
    keep = np.ones(rows, dtype=bool)
    keep[excluded] = False
    mapping = np.flatnonzero(keep).astype(np.int64, copy=False)
    if (
        len(mapping) + len(excluded) != rows
        or (len(mapping) > 1 and np.any(mapping[1:] <= mapping[:-1]))
        or (len(excluded) > 1 and np.any(excluded[1:] <= excluded[:-1]))
    ):
        raise Round0163Error("prompted-English representative selection did not close")
    report = {
        "selection_rule": (
            "union complete source-text UTF-8, raw stored-fp16, and Document: "
            "stored-fp16 exact-family relations; retain the lowest canonical "
            "row in every transitive component"
        ),
        "source_family_counts": {
            source: len(normalized[source]) for source in FAMILY_SOURCES
        },
        "source_rows_in_nontrivial_families": {
            source: sum(len(family) for family in normalized[source])
            for source in FAMILY_SOURCES
        },
        "union_family_count": len(union_families),
        "rows_in_union_families": sum(len(family) for family in union_families),
        "maximum_union_family_size": max(
            (len(family) for family in union_families), default=1
        ),
        "union_family_examples": union_families[:32],
        "excluded_rows": len(excluded),
        "retained_rows": len(mapping),
        "excluded_ordered_sha256": ordered_array_sha256(excluded),
        "mapping_ordered_sha256": ordered_array_sha256(mapping),
    }
    return mapping, excluded, report


def embedding_text_relation(
    embedding_families: Sequence[Sequence[int]],
    text_families: Sequence[Sequence[int]],
) -> dict[str, Any]:
    text_family_by_row: dict[int, int] = {}
    for index, family in enumerate(text_families):
        for row in family:
            text_family_by_row[int(row)] = index
    explained = 0
    cross_text: list[list[int]] = []
    for family in embedding_families:
        groups = {text_family_by_row.get(int(row)) for row in family}
        if len(groups) == 1 and None not in groups:
            explained += 1
        else:
            cross_text.append([int(row) for row in family])
    return {
        "exact_embedding_families": len(embedding_families),
        "source_text_explained_families": explained,
        "cross_source_text_families": len(cross_text),
        "cross_source_text_family_examples": cross_text[:32],
    }


def population_identity(
    *,
    view_identity: str,
    mapping: np.ndarray,
    excluded: np.ndarray,
) -> str:
    body = {
        "schema": "round0163-prompted-english-population-identity-v1",
        "view_identity": str(view_identity),
        "source_rows": VIEW_ROWS,
        "dimension": DIMENSION,
        "dtype": DTYPE,
        "selection_law": "source-text/raw-fp16/document-fp16-transitive-union-lowest-row",
        "mapping_ordered_sha256": ordered_array_sha256(mapping),
        "excluded_ordered_sha256": ordered_array_sha256(excluded),
        "retained_rows": len(mapping),
    }
    return sha256_bytes(canonical_json(body))
