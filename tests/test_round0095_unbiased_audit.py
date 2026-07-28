import numpy as np
import pytest

from basemap.round0095_unbiased_audit import (
    CORPUS_RANGES,
    MONOLITHIC_POLICIES,
    SAMPLE_ROWS,
    SAMPLE_SEED,
    SAMPLE_SHA256,
    SHARDED_POLICIES,
    sample_corpus_counts,
)
from experiments.round0095_nodes import _policy_metrics


def test_corrected_sample_contract_is_fixed() -> None:
    assert SAMPLE_ROWS == 4_096
    assert SAMPLE_SEED == 86
    assert SAMPLE_SHA256 == (
        "fab1613919b657a8116931b0fc336678576ea25ac3ce875b00576f860fa413fe"
    )
    assert CORPUS_RANGES == {
        "fineweb": (0, 50_000_000),
        "redpajama": (50_000_000, 100_000_000),
        "pile": (100_000_000, 150_000_000),
    }
    assert MONOLITHIC_POLICIES == (
        ("r0093_selected", 256, 1_536),
        ("r0093_highest_recall", 512, 1_536),
    )
    assert SHARDED_POLICIES == (
        ("r0094_strongest_registered", 96, 256),
    )


def test_sample_corpus_counts_cover_all_ranges() -> None:
    rows = np.asarray(
        [0, 49_999_999, 50_000_000, 99_999_999, 100_000_000, 149_999_999],
        dtype=np.int64,
    )
    assert sample_corpus_counts(rows) == {
        "fineweb": 2,
        "redpajama": 2,
        "pile": 2,
    }


def test_policy_metrics_report_corpus_recall_separately() -> None:
    sample = np.asarray(
        [10, 50_000_010, 100_000_010],
        dtype=np.int64,
    )
    exact = np.asarray(
        [
            list(range(0, 15)),
            list(range(20, 35)),
            list(range(40, 55)),
        ],
        dtype=np.int64,
    )
    selected = exact.copy()
    selected[2, -3:] = [70, 71, 72]
    metrics = _policy_metrics(
        selected,
        exact,
        sample=sample,
        unambiguous=np.ones(3, dtype=bool),
    )
    assert metrics["mean_recall_at_15_unambiguous"] == pytest.approx(14 / 15)
    assert metrics["by_corpus"]["fineweb"][
        "mean_recall_at_15_unambiguous"
    ] == 1.0
    assert metrics["by_corpus"]["redpajama"][
        "mean_recall_at_15_unambiguous"
    ] == 1.0
    assert metrics["by_corpus"]["pile"][
        "mean_recall_at_15_unambiguous"
    ] == 0.8
