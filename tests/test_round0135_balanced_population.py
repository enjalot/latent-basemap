from __future__ import annotations

import numpy as np
import pytest

from basemap import round0135_balanced_population as subject


def _configure_small_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subject, "FINAL_ROWS", 8)
    monkeypatch.setattr(subject, "STAGED_ROWS", 10)
    monkeypatch.setattr(subject, "PADDING_DUPLICATE_ROWS", 2)
    monkeypatch.setattr(subject, "CANDIDATE_ROWS_PER_LANGUAGE", 6)
    monkeypatch.setattr(subject, "ENGLISH_DATASETS", ("e0", "e1", "e2"))
    monkeypatch.setattr(subject, "IN_MIX_LANGUAGES", ("l1",))
    monkeypatch.setattr(subject, "LANGUAGE_GROUPS", ("eng_Latn", "l1"))
    monkeypatch.setattr(subject, "GROUPS", ("e0", "e1", "e2", "l1"))


def _inventory() -> dict[str, dict[str, object]]:
    counts = {
        "e0": 2,
        "e1": 2,
        "e2": 2,
        "fineweb2-l1-chunked-500-jina-v5-nano": 6,
    }
    return {
        dataset: {
            "rows": rows,
            "shards": [{
                "canonical_path": f"/{dataset}.npy",
                "sha256": dataset,
                "bytes": rows * 4,
                "rows": rows,
            }],
        }
        for dataset, rows in counts.items()
    }


def _census() -> dict[str, object]:
    arrays = {
        "zero_rows": np.empty(0, dtype=np.int64),
        "nonfinite_rows": np.empty(0, dtype=np.int64),
        "excluded_rows": np.asarray([3, 7, 8], dtype=np.int64),
        "duplicate_excluded_rows": np.asarray([3, 7, 8], dtype=np.int64),
        "duplicate_representative_rows": np.asarray([0, 6, 6], dtype=np.int64),
        "representative_rows": np.asarray([0, 6], dtype=np.int64),
        "family_counts": np.asarray([2, 3], dtype=np.int64),
        "family_offsets": np.asarray([0, 2, 5], dtype=np.int64),
        "member_rows": np.asarray([0, 3, 6, 7, 8], dtype=np.int64),
    }
    return {
        "arrays": arrays,
        "summary": {
            "row_count": 12,
            "excluded_row_count": 3,
            "duplicate_copy_rows_excluded": 3,
            "fingerprint_collision_splits": 0,
        },
    }


def test_largest_remainder_is_deterministic_and_exact() -> None:
    assert subject.largest_remainder_equal(("a", "b", "c"), 8) == {
        "a": 3,
        "b": 3,
        "c": 2,
    }
    with pytest.raises(subject.Round0135Error):
        subject.largest_remainder_equal(("b", "a"), 8)


def test_candidate_selection_uses_group_order_and_dataset_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_small_contract(monkeypatch)
    selection = subject.build_candidate_selection(_inventory())
    assert selection["candidate_rows"] == 12
    assert selection["source_order"] == ["e0", "e1", "e2", "l1"]
    assert [item["group"] for item in selection["ranges"]] == [
        "e0", "e1", "e2", "l1"
    ]
    assert selection["ranges"][-1]["dataset"] == (
        "fineweb2-l1-chunked-500-jina-v5-nano"
    )


def test_canonicalization_precedes_quota_and_padding_stays_excluded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_small_contract(monkeypatch)
    selection = subject.build_candidate_selection(_inventory())
    population = subject.build_balanced_population(selection, _census())

    np.testing.assert_array_equal(
        population["final_candidate_rows"],
        np.asarray([0, 1, 2, 4, 6, 9, 10, 11]),
    )
    np.testing.assert_array_equal(
        population["padding_candidate_rows"], np.asarray([3, 7])
    )
    np.testing.assert_array_equal(
        population["complement_candidate_rows"], np.asarray([3, 5, 7, 8])
    )
    np.testing.assert_array_equal(
        population["staged_candidate_rows"],
        np.asarray([0, 1, 2, 4, 6, 9, 10, 11, 3, 7]),
    )
    np.testing.assert_array_equal(
        population["eligibility"]["excluded_rows"], np.asarray([8, 9])
    )
    np.testing.assert_array_equal(
        population["eligibility"]["duplicate_representative_rows"],
        np.asarray([0, 4]),
    )
    assert population["final_group_quotas"] == {
        "e0": 2,
        "e1": 1,
        "e2": 1,
        "l1": 4,
    }
    assert population["final_language_quotas"] == {
        "eng_Latn": 4,
        "l1": 4,
    }
    assert all(population["checks"].values())


def test_quota_shortfall_aborts_without_replenishment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_small_contract(monkeypatch)
    census = _census()
    arrays = census["arrays"]
    arrays["zero_rows"] = np.asarray([0, 1], dtype=np.int64)
    arrays["excluded_rows"] = np.asarray([0, 1, 3, 7, 8], dtype=np.int64)
    census["summary"]["excluded_row_count"] = 5
    selection = subject.build_candidate_selection(_inventory())
    with pytest.raises(subject.Round0135Error, match="canonical rows for quota"):
        subject.build_balanced_population(selection, census)


def test_padding_aborts_if_authentic_selected_families_are_insufficient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_small_contract(monkeypatch)
    monkeypatch.setattr(subject, "STAGED_ROWS", 12)
    monkeypatch.setattr(subject, "PADDING_DUPLICATE_ROWS", 4)
    selection = subject.build_candidate_selection(_inventory())
    with pytest.raises(subject.Round0135Error, match="authentic duplicate copies"):
        subject.build_balanced_population(selection, _census())


def test_census_requires_closed_family_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_small_contract(monkeypatch)
    census = _census()
    census["arrays"]["family_offsets"] = np.asarray([0, 1, 5])
    selection = subject.build_candidate_selection(_inventory())
    with pytest.raises(subject.Round0135Error, match="census is malformed"):
        subject.build_balanced_population(selection, census)
