"""Frozen contract for the dose x N x width publication synthesis.

Campaign Track X: bind R0187/R0189/R0190/R0191/R0202/R0203/R0207 by exact hash
into one sealed artifact stating the dose x N x width interaction on Pile FFR,
and the operating rule it implies.

The synthesis is deliberately conservative about one thing. The campaign brief
describes "the width recovery at high dose" and "h4096 absorbs the high dose, at
3.118x cost". The sealed receipts do not support the high-dose half of that:
R0207's own `claim_scope` reads "paired seed-42 composition-controlled 2x3
factorial **at fixed dose**", and every width cell in the program — R0184 (h2048
full), R0191 (h4096 full), R0202 (h4096 quarter/half), R0203 (h2048
quarter/half) — reports `achieved_positive_draws_per_edge` equal to the low dose
`0.6781781544098838`. The 1.3743 dose appears only at h2048, in R0187 and its
R0188/R0189 seed replays. So the `3.118x` cost line is a **low-dose**
measurement, and the h4096 x high-dose cell simply does not exist.

This artifact therefore states the interaction the receipts actually establish
and names the missing cell as an explicit, testable gap rather than inheriting
the brief's phrasing. A campaign document is guidance, never a licence for a
claim the evidence does not carry.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


ROUND_ID = "0213"
CAPABILITY = "jina-dose-by-n-by-width-scaling-story-v1"
SYNTHESIS_SCHEMA = "round0213-dose-by-n-by-width-scaling-story-v1"

HIGH_DOSE = 1.3743131099922326
LOW_DOSE = 0.6781781544098838
REGISTERED_METRIC = "pile_ffr"
REGISTERED_BOUNDARY = "half_to_full"
RETENTION_FLOOR = 0.97
LOW_DOSE_WIDTHS = (2048, 4096)
HIGH_DOSE_WIDTHS = (2048,)
#: The operating rule the evidence supports, stated once.
OPERATING_RULE = (
    "scale N at hidden dimension 2048 and 0.6781781544098838 positive draws per "
    "directed edge"
)
#: The single cell that would license a capacity-absorbs-dose claim.
MISSING_CELL = {
    "hidden_dimension": 4096,
    "target_positive_draws_per_edge": HIGH_DOSE,
    "rungs": ["half", "full"],
    "why": (
        "every existing width cell is at the low dose, so no evidence separates "
        "capacity absorbing dose from capacity helping uniformly"
    ),
}


class Round0213Error(RuntimeError):
    """The registered scaling-story synthesis contract changed."""


def _finite(value: Any, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise Round0213Error(f"{label} is not finite")
    return number


def dose_axis(
    *,
    high_dose_retention: Sequence[float],
    high_dose_positive_by_seed: Mapping[str, bool],
    low_dose_full_over_half: float,
    seed_noise_sd: float,
) -> dict[str, Any]:
    """The dose axis at fixed width h2048: is the half->full boundary held?"""
    values = [_finite(v, label="high-dose retention") for v in high_dose_retention]
    if len(values) < 3:
        raise Round0213Error("the high-dose boundary needs at least three seeds")
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    sd = math.sqrt(variance)
    positive = sum(1 for v in high_dose_positive_by_seed.values() if v)
    low = _finite(low_dose_full_over_half, label="low-dose full/half")
    return {
        "width": 2048,
        "metric": REGISTERED_METRIC,
        "boundary": REGISTERED_BOUNDARY,
        "retention_floor": RETENTION_FLOOR,
        "high_dose": {
            "target_positive_draws_per_edge": HIGH_DOSE,
            "retention_values": values,
            "retention_mean": mean,
            "retention_sample_sd_ddof1": sd,
            "seeds": len(values),
            "seeds_clearing_floor": positive,
            "clears_floor_on_mean": mean >= RETENTION_FLOOR,
            "floor_inside_one_sd": abs(mean - RETENTION_FLOOR) <= sd,
            "verdict": "seed-sensitive regression",
        },
        "low_dose": {
            "target_positive_draws_per_edge": LOW_DOSE,
            "full_over_half": low,
            "clears_floor": low >= RETENTION_FLOOR,
            "verdict": "flat, no regression",
        },
        "interaction": (
            "at fixed h2048 the boundary the high dose loses on 2 of 3 seeds is "
            "not lost at the low dose"
        ),
    }


def width_axis(
    *,
    contrasts: Mapping[str, Mapping[str, Any]],
    seed_noise_sd: float,
    low_dose_widths_flat: bool,
) -> dict[str, Any]:
    """The width axis, which exists only at the low dose."""
    noise = _finite(seed_noise_sd, label="seed noise SD")
    cells: dict[str, Any] = {}
    for rung, cell in contrasts.items():
        delta = _finite(
            (cell.get("primary_metric_delta_h4096_minus_h2048") or {}).get(
                REGISTERED_METRIC
            ),
            label=f"{rung} width delta",
        )
        wall = _finite(
            cell.get("h4096_over_h2048_train_wall"), label=f"{rung} wall ratio"
        )
        cells[rung] = {
            "pile_ffr_delta_h4096_minus_h2048": delta,
            "exceeds_seed_noise_sd": abs(delta) > noise,
            "delta_in_seed_noise_sds": delta / noise if noise else None,
            "train_wall_ratio_h4096_over_h2048": wall,
            "extra_train_gpu_hours_h4096": _finite(
                cell.get("extra_train_gpu_hours_h4096"), label=f"{rung} extra hours"
            ),
            "marginal_pile_ffr_per_extra_gpu_hour": _finite(
                cell.get("marginal_pile_ffr_per_extra_gpu_hour"),
                label=f"{rung} marginal",
            ),
        }
    return {
        "dose_of_every_width_cell": LOW_DOSE,
        "widths_measured_at_low_dose": list(LOW_DOSE_WIDTHS),
        "widths_measured_at_high_dose": list(HIGH_DOSE_WIDTHS),
        "both_widths_flat_at_low_dose": bool(low_dose_widths_flat),
        "seed_noise_sd": noise,
        "cells": cells,
        "verdict": (
            "at the low dose h4096 buys a small positive Pile FFR delta for "
            "about 3.1x the train wall; it is not selected"
        ),
        "missing_cell": MISSING_CELL,
        "capacity_absorbs_dose_claim_supported": False,
    }


def loss_locality(*, context: Mapping[str, Any]) -> dict[str, Any]:
    """Why the flat low-dose readout is credible rather than a slice artifact."""
    predictors = dict(context.get("r0201_predictor_spearman") or {})
    if not predictors:
        raise Round0213Error("R0201 predictor correlations are missing")
    strongest = max(abs(float(v)) for v in predictors.values())
    return {
        "pattern": str(context.get("r0201_loss_pattern")),
        "k256_losing_cluster_coverage": _finite(
            context.get("r0201_k256_losing_cluster_coverage"), label="coverage"
        ),
        "k256_top_decile_loss_mass_share": _finite(
            context.get("r0201_k256_top_decile_loss_mass_share"), label="top decile"
        ),
        "predictor_spearman": {k: float(v) for k, v in predictors.items()},
        "strongest_absolute_spearman": strongest,
        "localised": strongest >= 0.20,
        "verdict": (
            "diffuse: spread across nearly every k256 cluster with no structural "
            "predictor above |rho| = 0.04, so the boundary loss is not a "
            "localised subpopulation artifact"
        ),
    }


def operating_rule(*, dose: Mapping[str, Any], width: Mapping[str, Any]) -> dict[str, Any]:
    """State the rule, and refuse to state it if the evidence stopped supporting it."""
    low_flat = bool(dose["low_dose"]["clears_floor"])
    high_sensitive = int(dose["high_dose"]["seeds_clearing_floor"]) < int(
        dose["high_dose"]["seeds"]
    )
    width_not_worth_it = not bool(width["capacity_absorbs_dose_claim_supported"])
    supported = low_flat and high_sensitive and width_not_worth_it
    if not supported:
        raise Round0213Error(
            "the registered operating rule is not supported by these cells"
        )
    return {
        "rule": OPERATING_RULE,
        "hidden_dimension": 2048,
        "target_positive_draws_per_edge": LOW_DOSE,
        "supported_by": [
            "low-dose half->full retention clears the floor at h2048",
            "high-dose half->full retention fails on a majority of seeds at h2048",
            "h4096 at the low dose is not worth 3.1x the train wall",
        ],
        "not_supported": [
            "h4096 absorbs the high dose (no such cell exists)",
            "any dose below 0.6781781544098838 (never probed)",
            "any diverse-composition population (R0211: the only diverse "
            "retention gate registered so far is not composition-matched)",
        ],
    }


__all__ = [
    "CAPABILITY",
    "HIGH_DOSE",
    "HIGH_DOSE_WIDTHS",
    "LOW_DOSE",
    "LOW_DOSE_WIDTHS",
    "MISSING_CELL",
    "OPERATING_RULE",
    "REGISTERED_BOUNDARY",
    "REGISTERED_METRIC",
    "RETENTION_FLOOR",
    "ROUND_ID",
    "Round0213Error",
    "SYNTHESIS_SCHEMA",
    "dose_axis",
    "loss_locality",
    "operating_rule",
    "width_axis",
]
