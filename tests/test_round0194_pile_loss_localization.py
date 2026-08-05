from __future__ import annotations

import numpy as np
import pytest

from basemap.round0194_pile_loss_localization import (
    ANCHORS,
    K_FRAC,
    Round0194Error,
    per_anchor_ffr,
    synthesize,
)


def test_per_anchor_ffr_preserves_truth_membership() -> None:
    high = np.tile(np.arange(10, dtype=np.int64), (3, 1))
    low = np.tile(np.arange(K_FRAC, dtype=np.int64), (3, 1))
    low[1] += 10
    low[2] += 5
    np.testing.assert_allclose(per_anchor_ffr(high, low), [1.0, 0.0, 0.5])


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


def test_synthesis_is_descriptive_and_complete() -> None:
    result = synthesize(*_fixture())
    assert result["scope"]["quality_gate"] is False
    assert result["scope"]["causal_claim"] is False
    assert set(result["per_seed"]) == {"42", "43", "44"}
    assert result["across_seed"]["anchors_negative_in_at_least_two_seeds"] > 0
    assert result["descriptive_pattern"] in {
        "diffuse", "cluster-concentrated", "mixed-or-unresolved"
    }


def test_missing_seed_fails_closed() -> None:
    scores, labels, predictors = _fixture()
    del scores[44]
    with pytest.raises(Round0194Error, match="seed"):
        synthesize(scores, labels, predictors)
