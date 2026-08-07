"""Frozen contract for the prompted-diverse U12 evaluation panel.

The accepted R0207 design memo registers this panel's decision structure, and
its instruction is unusually explicit: score the rung by **scale-relative
retention** against the nearest composition/prompt/width-matched rung, and
"do not reuse FineWeb-only absolute floors".  It also records
``commensurate_diverse_gate_family_required: true`` and requires that seed
variation be calibrated on this prompted-diverse population "before any
atlas-quality or production claim".

R0211 therefore splits R0169's single verdict in two:

* **Decisive** — every gate that is a *ratio* against a commensurate reference:
  the matched-2M retention gates against the accepted prompted 2M seed-42
  baseline, the per-language FFR floor relative to this map's own pooled
  English FFR, the Polish-to-in-mix OOD ratio, and the R0132 OOD retention
  cells.  These compare the rung to something measured on the same substrate
  and prompt, so they survive the memo's instruction.
* **Descriptive** — the native absolute cells against the R0161 prompted
  *English 2M* floors.  R0169 made these decisive.  They are transcribed here
  with their observed values and their reference floors, and they are excluded
  from the verdict, because a FineWeb-only floor is not commensurate with a
  20-language population.

Because no diverse seed family exists yet, a passing decisive verdict is
explicitly *not* an atlas-quality or production claim; the receipt stamps that
limitation rather than leaving it to prose.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from basemap.round0166_prompted_8m import NATIVE_ABSOLUTE_METRICS
from basemap.round0169_prompted_diverse import (
    LANGUAGE_TO_POOLED_ENGLISH_RATIO,
    POLISH_TO_IN_MIX_MEDIAN_RATIO,
    RETENTION_RATIO,
    ROWS,
    Round0169Error,
    prompted_diverse_decision,
)


ROUND_ID = "0211"
CAPABILITY = "jina-prompted-diverse-u12-evaluation-panel-v1"
EVALUATION_SCHEMA = "round0211-prompted-diverse-u12-evaluation-v1"
MODEL_CAPABILITY = "jina-prompted-diverse-u12-map-seed42-low-dose-v1"
TRAIN_SCHEMA = "round0210-prompted-diverse-u12-low-dose-train-receipt-v1"
PACK_CAPABILITY = "jina-prompted-u12-ood-probe-pack-v2"
PACK_SCHEMA = "round0208-prompted-u12-ood-probe-pack-v2"
PACK_CORPUS_ROWS = 49_494
PACK_QUERY_ROWS = 500
PACK_LANGUAGES = 20
PACK_ROWS = PACK_LANGUAGES * (PACK_CORPUS_ROWS + PACK_QUERY_ROWS)
ATLAS_QUALITY_BLOCKER = (
    "no commensurate prompted-diverse seed family exists; mean-2sigma gate "
    "calibration needs at least three diverse seeds"
)


class Round0211Error(Round0169Error):
    """The registered prompted-diverse evaluation contract changed."""


def diverse_panel_decision(
    *,
    native: Mapping[str, Any],
    matched_2m: Mapping[str, Any],
    baseline_2m_seed42: Mapping[str, Any],
    prompted_floors: Mapping[str, Any],
    group_ffr: Mapping[str, Any],
    prompted_ood: Mapping[str, Any],
    raw_r0132_ood: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the R0207 memo's scale-relative decision structure."""
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
        metric: {
            **base["native_absolute_gates"][metric],
            "reference_floor_origin": "R0161 prompted FineWeb English 2M family",
            "role": "descriptive",
            "commensurate_with_diverse_population": False,
        }
        for metric in NATIVE_ABSOLUTE_METRICS
    }
    for cell in descriptive.values():
        cell["would_have_passed_under_r0169"] = bool(cell.pop("passed"))
    decisive_stacks = {
        "matched_2m_retention_gates": base["matched_2m_retention_gates"],
        "language_relative_ffr_cells": base["language_relative_ffr"]["cells"],
        "raw_r0132_ood_retention_gates": base["raw_r0132_ood_retention_gates"],
    }
    passed = all(
        cell["passed"] for stack in decisive_stacks.values() for cell in stack.values()
    ) and bool(base["polish_ood_gate"]["passed"])
    return {
        "passed": passed,
        "outcome": (
            "prompted-diverse-u12-low-dose-rung-retention-qualified"
            if passed
            else "prompted-diverse-u12-low-dose-rung-retention-not-qualified"
        ),
        "primary_registered_readout": "scale-relative retention",
        "scale_relative_retention_gates": base["matched_2m_retention_gates"],
        "retention_reference": {
            "kind": "nearest composition/prompt/width-matched rung",
            "reference": "accepted prompted 2M seed-42 matched panel",
            "minimum_ratio": RETENTION_RATIO,
        },
        "language_relative_ffr": base["language_relative_ffr"],
        "language_relative_ratio": LANGUAGE_TO_POOLED_ENGLISH_RATIO,
        "polish_ood_gate": base["polish_ood_gate"],
        "polish_to_in_mix_ratio": POLISH_TO_IN_MIX_MEDIAN_RATIO,
        "raw_r0132_ood_retention_gates": base["raw_r0132_ood_retention_gates"],
        "native_absolute_cells": {
            "role": "descriptive",
            "reason": (
                "the R0207 memo forbids reusing FineWeb-only absolute floors on "
                "the diverse population"
            ),
            "cells": descriptive,
        },
        "diagnostic_only": [
            *base["diagnostic_only"],
            "density-v2 (transcribed)",
            "native absolute cells against R0161 English 2M floors",
        ],
        "atlas_quality_claim_available": False,
        "atlas_quality_blocker": ATLAS_QUALITY_BLOCKER,
        "production_claim_available": False,
        "r0169_style_verdict_for_reference": {
            "passed": bool(base["passed"]),
            "note": (
                "includes the non-commensurate native absolute gates; recorded "
                "for continuity with R0169's preregistration, not decisive here"
            ),
        },
    }


__all__ = [
    "ATLAS_QUALITY_BLOCKER",
    "CAPABILITY",
    "EVALUATION_SCHEMA",
    "MODEL_CAPABILITY",
    "PACK_CAPABILITY",
    "PACK_CORPUS_ROWS",
    "PACK_LANGUAGES",
    "PACK_QUERY_ROWS",
    "PACK_ROWS",
    "PACK_SCHEMA",
    "ROUND_ID",
    "ROWS",
    "Round0211Error",
    "TRAIN_SCHEMA",
    "diverse_panel_decision",
]
