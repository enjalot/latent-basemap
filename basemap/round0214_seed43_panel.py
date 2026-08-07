"""Frozen contract for scoring the seed-43 diverse cell — measurement, not a gate.

R0211 scored the seed-42 cell and its registered matched-2M retention gate
failed, because that gate divides by a model trained only on English 2M: it is
prompt-matched and width-matched but **not composition-matched**, so it mixes
composition change with scale. Accepted Review 0211 recorded that reading as
reasonable context and required that any *new* decisive gate be registered in a
future round.

R0214 does not register one. It exists to answer one narrow question cheaply:
**how far apart are the two diverse cells?** Every quality metric here is
therefore descriptive, and the only decisive checks are execution checks — the
transform is finite and non-collapsed, every group and language cell is present,
the OOD reserve is the repaired pack, and the scored model is the sealed seed-43
artifact trained on the same graph at the same dose as seed 42.

Two cells cannot establish a gate, and they cannot estimate a standard
deviation either: with n = 2 the sample SD is a single absolute difference
divided by sqrt(2), which carries no confidence. This round reports the paired
absolute difference per metric and says exactly that.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from basemap.round0166_prompted_8m import METRICS, NATIVE_ABSOLUTE_METRICS
from basemap.round0169_prompted_diverse import (
    LANGUAGE_TO_POOLED_ENGLISH_RATIO,
    POLISH_TO_IN_MIX_MEDIAN_RATIO,
    RETENTION_RATIO,
    ROWS,
    prompted_diverse_decision,
)
from basemap.round0211_prompted_diverse_panel import Round0211Error


ROUND_ID = "0214"
SEED = 43
PAIRED_SEED = 42
CAPABILITY = "jina-prompted-diverse-u12-seed43-panel-readout-v1"
EVALUATION_SCHEMA = "round0214-prompted-diverse-u12-seed43-panel-v1"
PAIRED_EVALUATION_SCHEMA = "round0211-prompted-diverse-u12-evaluation-v1"
CELLS_IN_FAMILY_AFTER_THIS_ROUND = 2
CELLS_REQUIRED_FOR_GATE = 3
NO_GATE_REASON = (
    "two cells cannot register a gate and cannot estimate a standard deviation; "
    "mean - 2 sigma needs at least three diverse seeds"
)


class Round0214Error(Round0211Error):
    """The registered seed-43 panel-readout contract changed."""


#: Set by the node before the panel runs, so the descriptive decision can pair
#: this cell against the accepted seed-42 readout.
PAIRED_REFERENCE: dict[str, Any] = {}


def paired_spread(
    *, this_cell: Mapping[str, float], paired_cell: Mapping[str, float]
) -> dict[str, Any]:
    """Report the two-cell difference per metric, without inventing a sigma."""
    cells: dict[str, Any] = {}
    for metric in METRICS:
        if metric not in this_cell or metric not in paired_cell:
            raise Round0214Error(f"paired spread is missing {metric}")
        here = float(this_cell[metric])
        there = float(paired_cell[metric])
        denominator = abs(there)
        cells[metric] = {
            f"seed{SEED}": here,
            f"seed{PAIRED_SEED}": there,
            "absolute_difference": abs(here - there),
            "relative_difference": (
                abs(here - there) / denominator if denominator else None
            ),
        }
    return {
        "cells": cells,
        "n_cells": CELLS_IN_FAMILY_AFTER_THIS_ROUND,
        "sigma_estimated": False,
        "sigma_reason": NO_GATE_REASON,
        "largest_relative_difference": max(
            (c["relative_difference"] for c in cells.values() if c["relative_difference"] is not None),
            default=None,
        ),
    }


def descriptive_panel_decision(
    *,
    native: Mapping[str, Any],
    matched_2m: Mapping[str, Any],
    baseline_2m_seed42: Mapping[str, Any],
    prompted_floors: Mapping[str, Any],
    group_ffr: Mapping[str, Any],
    prompted_ood: Mapping[str, Any],
    raw_r0132_ood: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute every cell R0211 computed, and make none of them decisive."""
    base = prompted_diverse_decision(
        native=native,
        matched_2m=matched_2m,
        baseline_2m_seed42=baseline_2m_seed42,
        prompted_floors=prompted_floors,
        group_ffr=group_ffr,
        prompted_ood=prompted_ood,
        raw_r0132_ood=raw_r0132_ood,
    )
    descriptive = {
        "native_absolute_cells": {
            metric: {
                **base["native_absolute_gates"][metric],
                "reference_floor_origin": "R0161 prompted FineWeb English 2M family",
            }
            for metric in NATIVE_ABSOLUTE_METRICS
        },
        "matched_2m_retention_cells": base["matched_2m_retention_gates"],
        "language_relative_ffr": base["language_relative_ffr"],
        "polish_ood_cell": base["polish_ood_gate"],
        "raw_r0132_ood_retention_cells": base["raw_r0132_ood_retention_gates"],
    }
    for cell in descriptive["native_absolute_cells"].values():
        cell["role"] = "descriptive"
        cell.pop("passed", None)
    decision: dict[str, Any] = {
        # No quality metric can fail this round; execution gates decide.
        "passed": True,
        "role": "measurement readout, no decisive quality gate registered",
        "decisive_quality_gate_registered": False,
        "why_no_gate": (
            "R0211's matched-2M retention gate is not composition-matched, and "
            "accepted Review 0211 requires any replacement to be registered in "
            "its own round; this round measures rather than judges"
        ),
        "descriptive_cells": descriptive,
        "reference_ratios_for_context": {
            "matched_2m_retention": RETENTION_RATIO,
            "language_to_pooled_english": LANGUAGE_TO_POOLED_ENGLISH_RATIO,
            "polish_to_in_mix_median": POLISH_TO_IN_MIX_MEDIAN_RATIO,
        },
        "atlas_quality_claim_available": False,
        "production_claim_available": False,
        "seed_family": {
            "this_seed": SEED,
            "paired_seed": PAIRED_SEED,
            "cells_after_this_round": CELLS_IN_FAMILY_AFTER_THIS_ROUND,
            "cells_required_for_gate": CELLS_REQUIRED_FOR_GATE,
            "gate_registerable_here": False,
            "reason": NO_GATE_REASON,
        },
    }
    paired = PAIRED_REFERENCE.get("native_decision_metrics")
    if paired:
        decision["paired_native_spread"] = paired_spread(
            this_cell={metric: float(native[metric]) for metric in METRICS},
            paired_cell=paired,
        )
    decision["outcome"] = "prompted-diverse-u12-seed43-panel-measured"
    return decision


__all__ = [
    "CAPABILITY",
    "CELLS_IN_FAMILY_AFTER_THIS_ROUND",
    "CELLS_REQUIRED_FOR_GATE",
    "EVALUATION_SCHEMA",
    "NO_GATE_REASON",
    "PAIRED_EVALUATION_SCHEMA",
    "PAIRED_REFERENCE",
    "PAIRED_SEED",
    "ROUND_ID",
    "ROWS",
    "Round0214Error",
    "SEED",
    "descriptive_panel_decision",
    "paired_spread",
]
