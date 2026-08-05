"""Additive R0198 correction of the R0194 Pile localization contract."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap import round0194_pile_loss_localization as base


ROUND_ID = "0198"
CAPABILITY = base.CAPABILITY
SEEDS = base.SEEDS
ANCHORS = base.ANCHORS
K_HIT = base.K_HIT
K_FRAC = base.K_FRAC
CLUSTER_KS = base.CLUSTER_KS
Round0198Error = base.Round0194Error
per_anchor_ffr = base.per_anchor_ffr


def synthesize(
    scores: Mapping[int, Mapping[str, np.ndarray]],
    labels: Mapping[int, np.ndarray],
    predictors: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Reuse the registered analysis with corrected round provenance."""
    result = base.synthesize(scores, labels, predictors)
    result["schema"] = "round0198-pile-boundary-loss-localization-v1"
    result["round_id"] = ROUND_ID
    result["capabilities"] = [CAPABILITY]
    result["supersedes_round"] = "0194"
    result["correction_scope"] = (
        "exact accepted capability names, verified review releases, and "
        "fail-closed low-dimensional boundary-tie semantics"
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
    "Round0198Error",
    "per_anchor_ffr",
    "synthesize",
]
