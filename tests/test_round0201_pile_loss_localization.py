from __future__ import annotations

import numpy as np
import pytest

from basemap.round0201_pile_loss_localization import (
    ANCHORS,
    CAPABILITY,
    Round0201Error,
    synthesize,
)
from experiments import round0201_nodes


def _fixture():
    index = np.arange(ANCHORS)
    scores = {}
    for seed in (42, 43, 44):
        half = 0.5 + (index % 3) * 0.1
        full = half.copy()
        full[index % 2 == 0] -= 0.1
        full[index % 5 == 0] += 0.1
        scores[seed] = {"half": half, "full": full}
    labels = {256: index % 256, 1024: index % 1024}
    predictors = {
        "log_r2_r1": 0.02 + index / (ANCHORS * 100),
        "hubness_occurrence": (index % 17).astype(np.float64),
        "mixture_centroid_distance": 0.1 + index / (ANCHORS * 10),
    }
    return scores, labels, predictors


def test_corrected_synthesis_has_r0201_provenance() -> None:
    result = synthesize(*_fixture())
    assert result["schema"] == "round0201-pile-boundary-loss-localization-v1"
    assert result["round_id"] == "0201"
    assert result["capabilities"] == [CAPABILITY]
    assert result["supersedes_round"] == "0198"


def test_float64_rerank_resolves_fp32_only_boundary_tie() -> None:
    values = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1e-4], [2.0, 0.0]],
        dtype=np.float32,
    )
    neighbors, receipt = round0201_nodes._rerank_candidates_float64(
        values,
        values[[0]],
        np.asarray([[2, 1, 3]], dtype=np.int64),
        k=1,
    )
    assert neighbors.tolist() == [[1]]
    assert receipt["zero_boundary_gaps_float32_diagnostic"] == 1
    assert receipt["zero_boundary_gaps_float64"] == 0
    assert receipt["minimum_boundary_gap_squared_l2_float64"] > 0


def test_true_float64_boundary_tie_fails_closed() -> None:
    values = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [2.0, 0.0]],
        dtype=np.float32,
    )
    with pytest.raises(Round0201Error, match="remains tied"):
        round0201_nodes._rerank_candidates_float64(
            values,
            values[[0]],
            np.asarray([[1, 2, 3]], dtype=np.int64),
            k=1,
        )


def test_six_cell_fp32_diagnosis_is_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    round0201_nodes._SEARCH_RECEIPTS[:] = [
        {"zero_boundary_gaps_float32_diagnostic": count}
        for count in round0201_nodes.EXPECTED_FP32_BOUNDARY_TIES.values()
    ]
    monkeypatch.setattr(round0201_nodes, "synthesize", lambda *_args: {})
    result = round0201_nodes._checked_synthesize(*_fixture())
    assert result["boundary_rerank_validation"][
        "float64_zero_boundary_ties_all_cells"
    ] is True


def test_six_cell_fp32_diagnosis_drift_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    round0201_nodes._SEARCH_RECEIPTS[:] = [
        {"zero_boundary_gaps_float32_diagnostic": 0}
        for _ in round0201_nodes.EXPECTED_FP32_BOUNDARY_TIES
    ]
    monkeypatch.setattr(round0201_nodes, "synthesize", lambda *_args: {})
    with pytest.raises(Round0201Error, match="diagnosis changed"):
        round0201_nodes._checked_synthesize(*_fixture())
