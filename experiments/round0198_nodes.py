"""Execute the corrected R0198 Track-D localization."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.round0198_pile_loss_localization import (
    CAPABILITY,
    ROUND_ID,
    Round0198Error,
    synthesize,
)
from experiments import round0194_nodes as base


ORIGINAL_EXACT_LOW_FRACTION = base._exact_low_fraction


def _strict_exact_low_fraction(
    coordinates: np.ndarray, anchor_ids: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    neighbors, receipt = ORIGINAL_EXACT_LOW_FRACTION(coordinates, anchor_ids)
    if int(receipt.get("zero_boundary_gaps", -1)) != 0:
        raise Round0198Error(
            "R0198 refuses ambiguous fixed-fraction boundary ties"
        )
    return neighbors, {
        **receipt,
        "boundary_tie_policy": "fail-closed; exactly zero tied k567/k568 boundaries",
        "deterministic_membership_proved": True,
    }


def _configure() -> None:
    base.ROUND_ID = ROUND_ID
    base.CAPABILITY = CAPABILITY
    base.Round0194Error = Round0198Error
    base.synthesize = synthesize
    base._exact_low_fraction = _strict_exact_low_fraction


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    _configure()
    base.run_job(active, job)


__all__ = ["run_job"]
