from __future__ import annotations

import inspect

import numpy as np

from experiments import prepare_round0044_queue as queue_prep
from experiments import round0044_nodes as node


def test_explicit_self_removal_does_not_assume_rank_zero() -> None:
    raw = np.array([
        [7, 3, 9, 2, 4],
        [8, 6, 1, 5, 0],
    ])
    clean = node._clean_search_rows(
        raw,
        query_ids=np.array([3, 6]),
        width=4,
        n_base=10,
    )
    assert np.array_equal(
        clean,
        np.array([[7, 9, 2, 4], [8, 1, 5, 0]]),
    )


def test_candidate_rank_lookup_preserves_shortlist_order() -> None:
    candidates = np.array([
        [9, 4, 2, 7],
        [5, 1, 8, 3],
    ])
    truth = np.array([
        [2, 9, 0],
        [3, 8, 6],
    ])
    ranks = node._candidate_ranks(
        exact_neighbors=truth,
        candidates=candidates,
        n_base=10,
    )
    assert np.array_equal(
        ranks,
        np.array([[2, 0, 5], [3, 2, 5]], dtype=np.int32),
    )


def test_policy_prefers_smallest_qualifying_pq_width() -> None:
    coarse = {
        "64": {"mean_recall_at_15_unambiguous": 0.95},
    }
    pq = {
        "64": {
            "512": {"mean_recall_at_15_unambiguous": 0.91},
            "2048": {"mean_recall_at_15_unambiguous": 0.97},
        },
        "256": {
            "128": {"mean_recall_at_15_unambiguous": 0.92},
        },
    }
    decision = node._choose_candidate_policy(coarse, pq)
    assert decision["classification"] == (
        "ivfpq-shortlist-plus-exact-rerank"
    )
    assert decision["selected_nprobe"] == 256
    assert decision["selected_width"] == 128


def test_policy_falls_back_to_exact_vector_routing() -> None:
    coarse = {
        "16": {"mean_recall_at_15_unambiguous": 0.89},
        "32": {"mean_recall_at_15_unambiguous": 0.93},
    }
    pq = {
        "64": {
            "8192": {"mean_recall_at_15_unambiguous": 0.72},
        },
    }
    decision = node._choose_candidate_policy(coarse, pq)
    assert decision["classification"] == (
        "exact-vector-search-with-current-coarse-routing"
    )
    assert decision["selected_nprobe"] == 32
    assert decision["requires_new_exact_vector_generator"] is True


def test_round0044_queue_is_one_bounded_no_training_node() -> None:
    source = inspect.getsource(queue_prep.prepare_round0044)
    assert "gpu_hours_cap=0.75" in source
    assert '"training_performed"] = False' in source
    assert '"candidate_quality_sweep_3m"' in source
