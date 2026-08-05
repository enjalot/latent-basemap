"""Pure synthesis contract for the R0190 three-seed boundary decision."""
from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from typing import Any


ROUND_ID = "0190"
CAPABILITY = "jina-composition-boundary-three-seed-synthesis-v1"
SCHEMA = "round0190-three-seed-boundary-synthesis-v1"
SEEDS = (42, 43, 44)
RUNGS = ("half", "full")
RETENTION_FLOOR = 0.97
GATE_METRICS = (
    "density_v2",
    "ffr",
    "purity_fidelity_k256",
    "purity_fidelity_k1024",
    "projection_ffr",
    "heldout_recall_at_10",
)


class Round0190Error(RuntimeError):
    """The frozen R0190 synthesis inputs or arithmetic changed."""


def _finite_positive(value: Any, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise Round0190Error(f"{label} must be finite and positive")
    return number


def _summary(values: list[float]) -> dict[str, Any]:
    if len(values) < 2:
        raise Round0190Error("sample summary requires at least two values")
    return {
        "values": values,
        "n": len(values),
        "mean": statistics.mean(values),
        "sample_sd_ddof1": statistics.stdev(values),
    }


def synthesize(
    *,
    cells: Mapping[str, Mapping[str, Mapping[str, float]]],
    quarter_seed42: Mapping[str, float],
    fineweb_seed42: Mapping[str, float],
    gates: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Bind the reviewed cells and release the deterministic 2-of-3 verdict."""
    if set(cells) != {f"seed{seed}" for seed in SEEDS}:
        raise Round0190Error("three-seed cell set changed")
    normalized: dict[str, dict[str, dict[str, float]]] = {}
    for seed in SEEDS:
        seed_key = f"seed{seed}"
        if set(cells[seed_key]) != set(RUNGS):
            raise Round0190Error(f"{seed_key} rung set changed")
        normalized[seed_key] = {}
        for rung in RUNGS:
            metrics = cells[seed_key][rung]
            required = {"pile_ffr", *GATE_METRICS}
            if set(metrics) != required:
                raise Round0190Error(f"{seed_key}/{rung} metric set changed")
            normalized[seed_key][rung] = {
                metric: _finite_positive(
                    metrics[metric], label=f"{seed_key}/{rung}/{metric}"
                )
                for metric in required
            }
    if set(quarter_seed42) != {"pile_ffr", *GATE_METRICS}:
        raise Round0190Error("quarter seed-42 metric set changed")
    if set(fineweb_seed42) != set(GATE_METRICS):
        raise Round0190Error("FineWeb seed-42 metric set changed")
    if set(gates) != set(GATE_METRICS):
        raise Round0190Error("R0161 gate metric set changed")

    gate_floors = {
        metric: _finite_positive(gates[metric]["floor"], label=f"gate/{metric}")
        for metric in GATE_METRICS
    }
    retention = {
        seed_key: normalized[seed_key]["full"]["pile_ffr"]
        / normalized[seed_key]["half"]["pile_ffr"]
        for seed_key in normalized
    }
    positive = {seed: value < RETENTION_FLOOR for seed, value in retention.items()}
    positive_count = sum(positive.values())
    if positive_count >= 2:
        outcome = "confirmed-2-of-3-seed-sensitive"
        capacity_activated = True
    else:
        outcome = "not-confirmed-across-three-seeds"
        capacity_activated = False

    absolute_cells: dict[str, Any] = {}
    available = {
        "seed42_quarter": {
            metric: _finite_positive(
                quarter_seed42[metric], label=f"seed42/quarter/{metric}"
            )
            for metric in GATE_METRICS
        },
        **{
            f"{seed_key}_{rung}": {
                metric: normalized[seed_key][rung][metric]
                for metric in GATE_METRICS
            }
            for seed_key in normalized
            for rung in RUNGS
        },
    }
    for cell, metrics in available.items():
        absolute_cells[cell] = {
            metric: {
                "observed": value,
                "fineweb_only_r0161_floor": gate_floors[metric],
                "descriptive_pass": value >= gate_floors[metric],
            }
            for metric, value in metrics.items()
        }

    composition_shift = {}
    for metric in GATE_METRICS:
        mixed = _finite_positive(
            quarter_seed42[metric], label=f"mixed quarter/{metric}"
        )
        fineweb = _finite_positive(fineweb_seed42[metric], label=f"FineWeb/{metric}")
        composition_shift[metric] = {
            "mixed_quarter_seed42": mixed,
            "fineweb_2m_seed42": fineweb,
            "difference_mixed_minus_fineweb": mixed - fineweb,
            "ratio_mixed_over_fineweb": mixed / fineweb,
        }

    half_values = [normalized[f"seed{seed}"]["half"]["pile_ffr"] for seed in SEEDS]
    full_values = [normalized[f"seed{seed}"]["full"]["pile_ffr"] for seed in SEEDS]
    return {
        "outcome": outcome,
        "registered_metric": "pile_ffr",
        "registered_boundary": "half_to_full",
        "retention_floor": RETENTION_FLOOR,
        "cells": normalized,
        "retention": retention,
        "retention_summary": _summary([retention[f"seed{seed}"] for seed in SEEDS]),
        "positive_by_seed": positive,
        "positive_seed_count": positive_count,
        "capacity_sibling_activated": capacity_activated,
        "capacity_scope": (
            "exactly one full-rung h4096 seed-42 1M-update sibling versus "
            "the accepted R0184 h2048 midpoint; no width ladder"
        ),
        "pile_ffr_rung_summaries": {
            "half": _summary(half_values),
            "full": _summary(full_values),
        },
        "width_null_noise_scale": {
            "metric": "pile_ffr",
            "source": "three-seed full-rung sample SD (ddof=1)",
            "value": statistics.stdev(full_values),
        },
        "absolute_against_noncommensurate_fineweb_gates": {
            "role": (
                "descriptive-only: R0161 was calibrated on a FineWeb-only "
                "population and is not an acceptance gate for mixed cells"
            ),
            "cells": absolute_cells,
        },
        "quarter_vs_fineweb_2m_composition_shift": {
            "role": (
                "descriptive same-seed near-2M contrast; population and held-out "
                "query universes differ, so no causal gate is released"
            ),
            "metrics": composition_shift,
        },
    }


__all__ = [
    "CAPABILITY",
    "GATE_METRICS",
    "ROUND_ID",
    "SCHEMA",
    "Round0190Error",
    "synthesize",
]
