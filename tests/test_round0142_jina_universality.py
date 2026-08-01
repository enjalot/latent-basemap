from __future__ import annotations

import numpy as np
import pytest

from basemap.round0142_jina_universality import (
    DIMENSION,
    Round0142Error,
    canonical_representatives,
    fixed_separate_split,
    fixed_single_array_split,
    retention_verdict,
    shape_matched_control_split,
)


def vectors(rows: int, seed: int = 7) -> np.ndarray:
    values = np.random.RandomState(seed).normal(size=(rows, DIMENSION)).astype(
        np.float16
    )
    return values


def test_exact_families_are_canonicalized_before_split() -> None:
    values = vectors(1_000)
    values[10] = values[0]
    values[11] = values[0]
    corpus, queries, receipt = fixed_single_array_split(values, name="probe")
    assert len(corpus) + len(queries) == 998
    assert receipt["duplicate_control"]["excluded_exact_duplicate_rows"] == 2
    assert set(corpus.tolist()).isdisjoint(queries.tolist())


def test_shape_control_reuses_precomputed_canonicalization() -> None:
    values = vectors(1_200)
    values[4] = values[3]
    representatives, duplicates = canonical_representatives(values)
    corpus, queries, receipt = shape_matched_control_split(
        values,
        name="control",
        corpus_rows=900,
        query_rows=100,
        representatives=representatives,
        duplicate_control=duplicates,
    )
    assert len(corpus) == 900
    assert len(queries) == 100
    assert receipt["duplicate_control"]["excluded_exact_duplicate_rows"] == 1
    assert set(corpus.tolist()).isdisjoint(queries.tolist())


def test_separate_probe_drops_cross_split_exact_family() -> None:
    corpus = vectors(700)
    queries = vectors(100, seed=8)
    corpus[9] = queries[3]
    corpus_rows, query_rows, receipt = fixed_separate_split(
        corpus, queries, name="beir", maximum_corpus_rows=500
    )
    assert len(corpus_rows) == 500
    assert len(query_rows) == 100
    assert receipt["cross_split_families_removed"] == 1
    corpus_keys = {corpus[row].tobytes() for row in corpus_rows}
    query_keys = {queries[row].tobytes() for row in query_rows}
    assert corpus_keys.isdisjoint(query_keys)


@pytest.mark.parametrize(
    ("value", "expected"),
    ((0.70, "pass"), (0.50, "amber"), (0.499, "named-failure")),
)
def test_retention_verdict_is_literal(value: float, expected: str) -> None:
    assert retention_verdict(value) == expected


def test_invalid_retention_fails_closed() -> None:
    with pytest.raises(Round0142Error):
        retention_verdict(float("nan"))

