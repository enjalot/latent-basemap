"""Contract tests for the R0159 provisional seed-margin analysis."""
from __future__ import annotations

import numpy as np
import pytest

from basemap.round0159_seed_margin_proposal import (
    MEASURES,
    SEEDS,
    Round0159Error,
    build_margin_proposal,
)


def _family(offset: float) -> dict[int, dict[str, float]]:
    return {
        seed: {
            metric: offset + index + (seed - 42) * 0.1
            for index, metric in enumerate(MEASURES, start=1)
        }
        for seed in SEEDS
    }


def test_proposal_is_descriptive_and_not_adopted() -> None:
    proposal = build_margin_proposal(_family(0.0), _family(-0.05))
    assert proposal["adopted"] is False
    assert proposal["margin_or_floor_changed"] is False
    assert proposal["owner_decision_required_for_adoption"] is True
    for metric in MEASURES:
        raw = proposal["raw_control_family"][metric]
        assert raw["sample_standard_deviation"] > 0
        assert raw["provisional_mean_minus_2sd"] == pytest.approx(
            raw["mean"] - 2 * raw["sample_standard_deviation"]
        )


def test_paired_deltas_preserve_seed_alignment() -> None:
    proposal = build_margin_proposal(_family(1.0), _family(0.75))
    for metric in MEASURES:
        paired = proposal["paired_drop_only_minus_raw"][metric]
        assert paired["mean"] == pytest.approx(-0.25)
        assert paired["sample_standard_deviation"] == pytest.approx(0.0, abs=1e-14)
        assert np.isfinite(paired["provisional_mean_minus_2sd"])


def test_incomplete_seed_matrix_is_rejected() -> None:
    raw = _family(0.0)
    del raw[45]
    with pytest.raises(Round0159Error, match="seed matrix changed"):
        build_margin_proposal(raw, _family(0.0))

