from __future__ import annotations

import numpy as np
import pytest

from basemap.round0198_pile_loss_localization import (
    ANCHORS,
    CAPABILITY,
    Round0198Error,
    synthesize,
)
from experiments import round0198_nodes
from experiments.prepare_round0198_queue import REVIEW_CAPABILITIES, _review_lineage


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


def test_corrected_synthesis_has_r0198_provenance() -> None:
    result = synthesize(*_fixture())
    assert result["schema"] == "round0198-pile-boundary-loss-localization-v1"
    assert result["round_id"] == "0198"
    assert result["capabilities"] == [CAPABILITY]
    assert result["supersedes_round"] == "0194"


def test_boundary_ties_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        round0198_nodes,
        "ORIGINAL_EXACT_LOW_FRACTION",
        lambda coordinates, anchors: (
            np.zeros((len(anchors), 567), dtype=np.int64),
            {"zero_boundary_gaps": 1},
        ),
    )
    with pytest.raises(Round0198Error, match="boundary ties"):
        round0198_nodes._strict_exact_low_fraction(
            np.zeros((10, 2), dtype=np.float32), np.asarray([0])
        )


def test_exact_released_capability_lineages_resolve() -> None:
    assert REVIEW_CAPABILITIES == {
        "0187": "jina-document-english-composition-controlled-nested-ladder-v1",
        "0188": "jina-document-english-composition-controlled-half-full-seed43-replay-v1",
        "0189": "jina-document-english-composition-controlled-half-full-seed44-replay-v1",
    }
    for round_id in REVIEW_CAPABILITIES:
        assert len(_review_lineage(round_id)) == 5
