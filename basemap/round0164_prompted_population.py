"""Pure contracts for the prompted-only English 8M population decision."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .artifact_identity import canonical_json, ordered_array_sha256, sha256_bytes
from .round0162_prompted_english_staging import DIMENSION, DTYPE, VIEW_ROWS


ROUND_ID = "0164"
CAPABILITY = "jina-document-english-first8m-prompted-representative-population-v2"
HOST_CAPABILITY = "jina-document-english-first8m-prompted-host-fp16-v2"
SCHEMA = "round0164-prompted-english-8m-population-v1"
FAMILY_SOURCES = ("source_text", "document_fp16")


class Round0164Error(RuntimeError):
    """Raised when the prompted-only population contract is violated."""


def seal(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    return {**value, "identity_sha256": sha256_bytes(canonical_json(value))}


def prompted_representatives(
    families_by_source: Mapping[str, Sequence[Sequence[int]]],
    *,
    rows: int = VIEW_ROWS,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Union only identities observable in the prompted training universe.

    Equal source text controls corpus multiplicity; equal stored prompted-fp16
    rows control exact geometry multiplicity.  Raw, unprompted embeddings are
    deliberately absent because they are neither trained nor evaluated by Q2.
    """
    if set(families_by_source) != set(FAMILY_SOURCES) or rows <= 0:
        raise Round0164Error("prompted-only family sources are incomplete")
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
                raise Round0164Error(f"{source} exact family is malformed")
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
    if len(mapping) + len(excluded) != rows:
        raise Round0164Error("prompted-only representative selection did not close")
    report = {
        "selection_rule": (
            "union complete source-text UTF-8 and Document: stored-fp16 "
            "exact-family relations; retain lowest canonical row"
        ),
        "raw_unprompted_relation_used": False,
        "source_family_counts": {
            source: len(normalized[source]) for source in FAMILY_SOURCES
        },
        "union_family_count": len(union_families),
        "rows_in_union_families": sum(len(family) for family in union_families),
        "maximum_union_family_size": max(
            (len(family) for family in union_families), default=1
        ),
        "excluded_rows": len(excluded),
        "retained_rows": len(mapping),
        "excluded_ordered_sha256": ordered_array_sha256(excluded),
        "mapping_ordered_sha256": ordered_array_sha256(mapping),
    }
    return mapping, excluded, report


def population_identity(
    *, view_identity: str, mapping: np.ndarray, excluded: np.ndarray
) -> str:
    body = {
        "schema": "round0164-prompted-population-identity-v1",
        "view_identity": str(view_identity),
        "source_rows": VIEW_ROWS,
        "dimension": DIMENSION,
        "dtype": DTYPE,
        "selection_law": "source-text/document-fp16-transitive-union-lowest-row",
        "mapping_ordered_sha256": ordered_array_sha256(mapping),
        "excluded_ordered_sha256": ordered_array_sha256(excluded),
        "retained_rows": len(mapping),
    }
    return sha256_bytes(canonical_json(body))

