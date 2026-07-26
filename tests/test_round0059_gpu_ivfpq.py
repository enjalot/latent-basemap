from __future__ import annotations

import numpy as np
import pytest

from experiments.round0059_nodes import (
    MEAN_RECALL_FLOOR,
    Round0059Error,
    _overlap,
    _selected_nprobe,
)


def _sweep(*, selected: int = 32, passed: bool = True) -> dict:
    return {
        "validity_passed": passed,
        "selected_nprobe": selected,
        "rows_by_nprobe": {
            str(selected): {
                "passes_mean_floor": True,
                "mean_recall_at_15_unambiguous": MEAN_RECALL_FLOOR,
            },
        },
    }


def test_selected_nprobe_accepts_only_passing_r0058_receipt() -> None:
    assert _selected_nprobe(_sweep()) == 32
    with pytest.raises(Round0059Error, match="passing nprobe"):
        _selected_nprobe(_sweep(passed=False))
    value = _sweep()
    value["rows_by_nprobe"]["32"]["passes_mean_floor"] = False
    with pytest.raises(Round0059Error, match="passing nprobe"):
        _selected_nprobe(value)


def test_engine_overlap_is_set_based_and_reports_exact_rows() -> None:
    left = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    right = np.asarray([[3, 2, 1], [4, 5, 7]], dtype=np.int32)
    observed = _overlap(left, right)
    assert observed["mean"] == pytest.approx(5 / 6)
    assert observed["p10"] == pytest.approx(0.7)
    assert observed["exact_row_fraction"] == 0.0


def test_engine_overlap_rejects_mismatched_geometry() -> None:
    with pytest.raises(Round0059Error, match="geometry"):
        _overlap(
            np.zeros((2, 3), dtype=np.int32),
            np.zeros((2, 4), dtype=np.int32),
        )


def test_faiss_id_range_copy_and_physical_filter_are_equivalent() -> None:
    faiss = pytest.importorskip("faiss")
    rng = np.random.RandomState(59)
    vectors = rng.normal(size=(60, 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    quantizer = faiss.IndexFlatIP(8)
    source = faiss.IndexIVFFlat(
        quantizer,
        8,
        4,
        faiss.METRIC_INNER_PRODUCT,
    )
    source.train(vectors)
    source.add_with_ids(vectors, np.arange(60, dtype=np.int64))
    destination = faiss.clone_index(source)
    destination.reset()
    for start, stop in ((0, 10), (20, 30), (40, 50)):
        source.copy_subset_to(
            destination,
            faiss.InvertedLists.SUBSET_TYPE_ID_RANGE,
            start,
            stop,
        )
    assert destination.ntotal == 30
    removed = destination.remove_ids(
        faiss.IDSelectorBatch(np.asarray([2, 22, 42], dtype=np.int64))
    )
    assert removed == 3
    assert destination.ntotal == 27
