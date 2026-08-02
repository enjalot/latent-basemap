"""Provisional seed-variance margin proposal for Round 0159."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.round0140_subsystem_bisection import METRICS


ROUND_ID = "0159"
CAPABILITY = "jina-2m-seed-variance-margin-proposal-v1"
SEEDS = (42, 43, 44, 45)
MEASURES = (*METRICS, "density_v2")


class Round0159Error(RuntimeError):
    """Raised when the reviewed seed matrix is incomplete or malformed."""


def _matrix(
    values: Mapping[int, Mapping[str, float]], *, label: str
) -> np.ndarray:
    if set(values) != set(SEEDS):
        raise Round0159Error(f"{label} seed matrix changed")
    matrix = np.asarray(
        [[float(values[seed][metric]) for metric in MEASURES] for seed in SEEDS],
        dtype=np.float64,
    )
    if matrix.shape != (len(SEEDS), len(MEASURES)) or not np.isfinite(matrix).all():
        raise Round0159Error(f"{label} metric matrix changed")
    return matrix


def _summaries(matrix: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, metric in enumerate(MEASURES):
        values = matrix[:, index]
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        proposed = mean - 2.0 * sd
        result[metric] = {
            "values_by_seed": {
                str(seed): float(value) for seed, value in zip(SEEDS, values, strict=True)
            },
            "mean": mean,
            "sample_standard_deviation": sd,
            "standard_error": float(sd / np.sqrt(len(values))),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "range": float(np.ptp(values)),
            "coefficient_of_variation": float(sd / mean) if mean else None,
            "provisional_mean_minus_2sd": proposed,
            "provisional_absolute_tolerance_2sd": 2.0 * sd,
            "provisional_retention_ratio": proposed / mean if mean else None,
        }
    return result


def build_margin_proposal(
    raw: Mapping[int, Mapping[str, float]],
    drop_only: Mapping[int, Mapping[str, float]],
) -> dict[str, Any]:
    """Describe seed variance without changing any registered threshold."""
    raw_matrix = _matrix(raw, label="raw")
    drop_matrix = _matrix(drop_only, label="drop-only")
    raw_summary = _summaries(raw_matrix)
    drop_summary = _summaries(drop_matrix)
    paired = drop_matrix - raw_matrix
    paired_summary = _summaries(paired)
    proposal_tests: dict[str, Any] = {}
    for index, metric in enumerate(MEASURES):
        floor = float(raw_summary[metric]["provisional_mean_minus_2sd"])
        observed = drop_matrix[:, index]
        passes = observed >= floor
        proposal_tests[metric] = {
            "provisional_control_family_floor": floor,
            "drop_only_pass_by_seed": {
                str(seed): bool(value)
                for seed, value in zip(SEEDS, passes, strict=True)
            },
            "drop_only_pass_count": int(passes.sum()),
            "drop_only_all_four_pass": bool(passes.all()),
            "decision_use": False,
        }
    return {
        "schema": "round0159-seed-variance-margin-proposal-v1",
        "round_id": ROUND_ID,
        "capability": CAPABILITY,
        "seeds": list(SEEDS),
        "metrics": list(MEASURES),
        "raw_control_family": raw_summary,
        "drop_only_family": drop_summary,
        "paired_drop_only_minus_raw": paired_summary,
        "provisional_control_family_tests": proposal_tests,
        "proposal_rule": "raw control mean minus two sample standard deviations",
        "statistical_warning": (
            "n=4 seeds is an empirical calibration cell, not a population "
            "tolerance interval or confidence-bound guarantee"
        ),
        "owner_decision_required_for_adoption": True,
        "adopted": False,
        "margin_or_floor_changed": False,
        "training_performed": False,
    }

