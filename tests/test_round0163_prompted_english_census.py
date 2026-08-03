"""Pure contract tests for the prompted-English representative census."""
from __future__ import annotations

import numpy as np

from basemap.round0163_prompted_english_census import (
    embedding_text_relation,
    population_identity,
    union_representatives,
)
from basemap.artifact_identity import expected_input_signature
from experiments.round0163_nodes import _validate_signature


def test_union_is_transitive_and_keeps_lowest_canonical_row() -> None:
    mapping, excluded, report = union_representatives({
        "source_text": [[2, 7], [11, 12]],
        "raw_fp16": [[7, 9]],
        "document_fp16": [[20, 25, 29]],
    }, rows=32)
    assert excluded.tolist() == [7, 9, 12, 25, 29]
    assert mapping.tolist() == [row for row in range(32) if row not in excluded]
    assert report["union_family_count"] == 3
    assert report["retained_rows"] == 27
    assert len(report["mapping_ordered_sha256"]) == 64


def test_embedding_text_relation_exposes_cross_text_collisions() -> None:
    report = embedding_text_relation(
        [[1, 2], [4, 9], [10, 11]],
        [[1, 2], [4, 5], [10, 11]],
    )
    assert report["source_text_explained_families"] == 2
    assert report["cross_source_text_families"] == 1
    assert report["cross_source_text_family_examples"] == [[4, 9]]


def test_population_identity_binds_view_and_both_selection_arrays() -> None:
    mapping = np.asarray([0, 2, 4], dtype=np.int64)
    excluded = np.asarray([1, 3], dtype=np.int64)
    baseline = population_identity(
        view_identity="a" * 64, mapping=mapping, excluded=excluded
    )
    assert baseline != population_identity(
        view_identity="b" * 64, mapping=mapping, excluded=excluded
    )
    assert baseline != population_identity(
        view_identity="a" * 64,
        mapping=np.asarray([0, 3, 4], dtype=np.int64),
        excluded=excluded,
    )


def test_payload_verifier_accepts_reviewed_nonidentity_metadata(tmp_path) -> None:
    path = tmp_path / "raw.npy"
    path.write_bytes(b"reviewed-payload")
    signature = expected_input_signature(str(path))
    _validate_signature({**signature, "rows": 1}, label="synthetic raw")
