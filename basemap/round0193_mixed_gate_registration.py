"""Binding mixed-quarter quality-gate registration for R0193."""
from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from typing import Any

from basemap.round0192_quarter_seed_family import GATE_METRICS, ROWS


ROUND_ID = "0193"
CAPABILITY = "jina-mixed-english-2m-quality-gates-v1"
FORMULA = "family mean - 2 * sample standard deviation (ddof=1)"
SEEDS = (42, 43, 44)


class Round0193Error(RuntimeError):
    """The R0193 gate registration changed after preregistration."""


def register_mixed_gates(family: Mapping[str, Any]) -> dict[str, Any]:
    if (
        family.get("outcome") != "mixed-quarter-three-seed-family-complete"
        or family.get("seeds") != list(SEEDS)
        or int(family.get("rows", -1)) != ROWS
        or family.get("gate_registration_deferred_to_reviewed_cpu_round") is not True
    ):
        raise Round0193Error("mixed-quarter family premise changed")
    cells = family.get("gate_metric_cells") or {}
    if set(cells) != {str(seed) for seed in SEEDS}:
        raise Round0193Error("mixed-quarter gate family is incomplete")
    gates: dict[str, Any] = {}
    for metric in GATE_METRICS:
        values = [float(cells[str(seed)][metric]) for seed in SEEDS]
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise Round0193Error(f"mixed gate metric {metric} is invalid")
        mean = statistics.fmean(values)
        sample_sd = statistics.stdev(values)
        floor = mean - 2.0 * sample_sd
        if not math.isfinite(floor):
            raise Round0193Error(f"mixed gate floor {metric} is invalid")
        gates[metric] = {
            "direction": "higher-is-better",
            "seed_order": list(SEEDS),
            "values": values,
            "mean": mean,
            "sample_sd_ddof1": sample_sd,
            "multiplier": 2.0,
            "floor": floor,
        }
    return {
        "schema": "round0193-mixed-english-quality-gates-v1",
        "round_id": ROUND_ID,
        "capability": CAPABILITY,
        "registered": True,
        "population": (
            "R0187 composition-controlled mixed-English quarter universe"
        ),
        "rows": ROWS,
        "seed_family": list(SEEDS),
        "n": len(SEEDS),
        "formula": FORMULA,
        "sample_standard_deviation_ddof": 1,
        "gates": gates,
        "r0161_prompted_fineweb_floors_unchanged": True,
        "raw_universe_floors_unchanged": True,
        "training_performed": False,
    }


__all__ = [
    "CAPABILITY",
    "FORMULA",
    "ROUND_ID",
    "SEEDS",
    "Round0193Error",
    "register_mixed_gates",
]
