"""Binding prompted-universe gate registration for Round 0161."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.round0160_prompted_seed_family import METRICS, SEEDS


ROUND_ID = "0161"
CAPABILITY = "jina-prompted-universe-quality-gates-v1"
FORMULA = "family mean - 2 * sample standard deviation (ddof=1)"


class Round0161Error(RuntimeError):
    """Raised when prompted gate registration changes after preregistration."""


def register_prompted_gates(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    expected = {f"seed{seed}" for seed in SEEDS}
    if set(cells) != expected:
        raise Round0161Error("prompted gate family is incomplete")
    gates: dict[str, Any] = {}
    for metric in METRICS:
        values = np.asarray(
            [float(cells[f"seed{seed}"]["decision_metrics"][metric]) for seed in SEEDS],
            dtype=np.float64,
        )
        if values.shape != (4,) or not np.isfinite(values).all():
            raise Round0161Error(f"prompted gate metric {metric} is invalid")
        mean = float(values.mean())
        sample_sd = float(values.std(ddof=1))
        gates[metric] = {
            "direction": "higher-is-better",
            "seed_order": list(SEEDS),
            "values": values.tolist(),
            "mean": mean,
            "sample_sd_ddof1": sample_sd,
            "multiplier": 2.0,
            "floor": mean - 2.0 * sample_sd,
        }
    return {
        "schema": "round0161-prompted-universe-quality-gates-v1",
        "round_id": ROUND_ID,
        "capability": CAPABILITY,
        "registered": True,
        "population": "R0113/R0115 native Document:-prompted 1,993,761-row universe",
        "seed_family": list(SEEDS),
        "n": len(SEEDS),
        "formula": FORMULA,
        "sample_standard_deviation_ddof": 1,
        "all_metrics_registered_before_r0160_results": True,
        "gates": gates,
        "raw_universe_floors_retired_and_unchanged": True,
        "raw_floor_changed": False,
        "training_performed": False,
    }
