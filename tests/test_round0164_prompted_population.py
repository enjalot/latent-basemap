"""Focused tests for the prompted-only R0164 population law."""
from __future__ import annotations

import numpy as np

from basemap.round0164_prompted_population import (
    population_identity,
    prompted_representatives,
)


def test_raw_only_alias_is_not_an_identity_relation() -> None:
    mapping, excluded, report = prompted_representatives(
        {
            "source_text": [[2, 7], [11, 12]],
            "document_fp16": [[7, 9], [20, 25, 29]],
        },
        rows=32,
    )
    assert excluded.tolist() == [7, 9, 12, 25, 29]
    assert mapping.tolist() == [row for row in range(32) if row not in excluded]
    assert report["raw_unprompted_relation_used"] is False


def test_dropping_a_relation_can_only_add_representatives() -> None:
    old_mapping = np.asarray([0, 1, 2, 4, 5], dtype=np.int64)
    new_mapping, _, _ = prompted_representatives(
        {"source_text": [], "document_fp16": []}, rows=6
    )
    positions = np.searchsorted(new_mapping, old_mapping)
    assert np.array_equal(new_mapping[positions], old_mapping)
    assert np.setdiff1d(new_mapping, old_mapping).tolist() == [3]


def test_population_identity_binds_prompted_mapping() -> None:
    mapping = np.asarray([0, 2, 4], dtype=np.int64)
    excluded = np.asarray([1, 3], dtype=np.int64)
    baseline = population_identity(
        view_identity="a" * 64, mapping=mapping, excluded=excluded
    )
    assert baseline != population_identity(
        view_identity="a" * 64,
        mapping=np.asarray([0, 3, 4], dtype=np.int64),
        excluded=excluded,
    )
