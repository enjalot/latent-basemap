"""R0201 float64 boundary-rerank correction for Track D."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap import round0198_pile_loss_localization as prior


ROUND_ID = "0201"
CAPABILITY = prior.CAPABILITY
SEEDS = prior.SEEDS
ANCHORS = prior.ANCHORS
K_HIT = prior.K_HIT
K_FRAC = prior.K_FRAC
CLUSTER_KS = prior.CLUSTER_KS
Round0201Error = prior.Round0198Error
per_anchor_ffr = prior.per_anchor_ffr


def synthesize(
    scores: Mapping[int, Mapping[str, np.ndarray]],
    labels: Mapping[int, np.ndarray],
    predictors: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Reuse Track D's frozen estimand with corrected numerical provenance."""
    result = prior.synthesize(scores, labels, predictors)
    result["schema"] = "round0201-pile-boundary-loss-localization-v1"
    result["round_id"] = ROUND_ID
    result["capabilities"] = [CAPABILITY]
    result["supersedes_round"] = "0198"
    result["correction_scope"] = (
        "float64 squared-L2 candidate rerank removes one fp32-only k567/k568 "
        "boundary tie; population, cells, anchors, metric, and selector unchanged"
    )
    return result


__all__ = [
    "ANCHORS",
    "CAPABILITY",
    "CLUSTER_KS",
    "K_FRAC",
    "K_HIT",
    "ROUND_ID",
    "SEEDS",
    "Round0201Error",
    "per_anchor_ffr",
    "synthesize",
]
