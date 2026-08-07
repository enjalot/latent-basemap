"""The sharded ANN merge must ignore empty-shard slots, never rank them."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.round0166_nodes import Round0166Error, _merge_ann_topk


def _reference_merge(left_sims, left_ids, right_sims, right_ids, *, k):
    """The pre-fix behaviour, valid only when every slot carries a candidate."""
    sims = np.concatenate((left_sims, right_sims), axis=1).astype(np.float32)
    ids = np.concatenate((left_ids, right_ids), axis=1).astype(np.int64)
    order = np.lexsort((ids, -sims), axis=1)[:, :k]
    return (
        np.take_along_axis(sims, order, axis=1),
        np.take_along_axis(ids, order, axis=1),
    )


def test_all_present_is_byte_identical_to_the_previous_behaviour() -> None:
    rng = np.random.RandomState(0)
    left_sims = rng.rand(64, 8).astype(np.float32)
    right_sims = rng.rand(64, 8).astype(np.float32)
    left_ids = rng.randint(0, 10_000, size=(64, 8)).astype(np.int64)
    right_ids = rng.randint(0, 10_000, size=(64, 8)).astype(np.int64)
    got_sims, got_ids = _merge_ann_topk(
        left_sims, left_ids, right_sims, right_ids, k=5
    )
    want_sims, want_ids = _reference_merge(
        left_sims, left_ids, right_sims, right_ids, k=5
    )
    assert got_sims.tobytes() == want_sims.tobytes()
    assert got_ids.tobytes() == want_ids.tobytes()


def test_score_ties_still_break_on_ascending_global_id() -> None:
    left_sims = np.full((1, 4), 0.5, dtype=np.float32)
    right_sims = np.full((1, 4), 0.5, dtype=np.float32)
    left_ids = np.asarray([[7, 3, 9, 5]], dtype=np.int64)
    right_ids = np.asarray([[8, 2, 6, 4]], dtype=np.int64)
    _sims, ids = _merge_ann_topk(left_sims, left_ids, right_sims, right_ids, k=4)
    assert ids.tolist() == [[2, 3, 4, 5]]


def test_empty_shard_slots_are_excluded_not_ranked() -> None:
    """A shard with no candidate must not displace the other shard's rows."""
    left_sims = np.asarray([[0.9, 0.8, 0.7, 0.6]], dtype=np.float32)
    left_ids = np.asarray([[10, 11, 12, 13]], dtype=np.int64)
    # FAISS emits -1 ids with a large negative sentinel score for empty lists.
    right_sims = np.full((1, 4), -3.4028235e38, dtype=np.float32)
    right_ids = np.full((1, 4), -1, dtype=np.int64)
    sims, ids = _merge_ann_topk(left_sims, left_ids, right_sims, right_ids, k=4)
    assert ids.tolist() == [[10, 11, 12, 13]]
    assert np.array_equal(sims, left_sims)


def test_partially_filled_shard_contributes_only_its_real_candidates() -> None:
    left_sims = np.asarray([[0.9, 0.5, 0.4, 0.3]], dtype=np.float32)
    left_ids = np.asarray([[10, 11, 12, 13]], dtype=np.int64)
    right_sims = np.asarray([[0.95, 0.6, -1e38, -1e38]], dtype=np.float32)
    right_ids = np.asarray([[20, 21, -1, -1]], dtype=np.int64)
    _sims, ids = _merge_ann_topk(left_sims, left_ids, right_sims, right_ids, k=4)
    assert ids.tolist() == [[20, 10, 21, 11]]


def test_rows_short_of_k_across_every_shard_stay_negative() -> None:
    """The caller's completeness guard must still see -1, not a sentinel."""
    left_sims = np.asarray([[0.9, -1e38, -1e38, -1e38]], dtype=np.float32)
    left_ids = np.asarray([[10, -1, -1, -1]], dtype=np.int64)
    right_sims = np.asarray([[0.8, -1e38, -1e38, -1e38]], dtype=np.float32)
    right_ids = np.asarray([[20, -1, -1, -1]], dtype=np.int64)
    sims, ids = _merge_ann_topk(left_sims, left_ids, right_sims, right_ids, k=4)
    assert ids.tolist() == [[10, 20, -1, -1]]
    assert np.any(ids < 0)
    assert not np.isfinite(sims).all()


def test_a_nonfinite_score_on_a_real_candidate_still_fails_closed() -> None:
    left_sims = np.asarray([[0.9, np.nan, 0.7, 0.6]], dtype=np.float32)
    left_ids = np.asarray([[10, 11, 12, 13]], dtype=np.int64)
    right_sims = np.asarray([[0.5, 0.4, 0.3, 0.2]], dtype=np.float32)
    right_ids = np.asarray([[20, 21, 22, 23]], dtype=np.int64)
    with pytest.raises(Round0166Error):
        _merge_ann_topk(left_sims, left_ids, right_sims, right_ids, k=4)


def test_geometry_drift_still_fails_closed() -> None:
    with pytest.raises(Round0166Error):
        _merge_ann_topk(
            np.zeros((2, 3), dtype=np.float32),
            np.zeros((2, 3), dtype=np.int64),
            np.zeros((2, 3), dtype=np.float32),
            np.zeros((2, 3), dtype=np.int64),
            k=4,
        )
