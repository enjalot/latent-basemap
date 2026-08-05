"""Frozen CPU diagnosis contract for R0196 GrEASE batch-stable inference."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Callable

import numpy as np


ROUND_ID = "0196"
CAPABILITY = "jina-grease-batch-stability-diagnosis-v1"
PATCH_CAPABILITY = "jina-grease-batch-stable-inference-patch-v1"
NEGATIVE_CAPABILITY = "jina-grease-batch-stability-negative-v1"
RELOAD_TOLERANCE = 1.0e-4
INFERENCE_CHUNK_ROWS = 256


class Round0196Error(RuntimeError):
    """The bounded GrEASE inference diagnosis changed or is invalid."""


def fixed_chunks(
    values: Any,
    function: Callable[[Any], np.ndarray],
    *,
    chunk_rows: int = INFERENCE_CHUNK_ROWS,
) -> np.ndarray:
    """Apply a pure inference function with one frozen batch geometry."""
    if getattr(values, "ndim", None) != 2 or len(values) <= 0 or chunk_rows <= 0:
        raise Round0196Error("fixed-chunk inference geometry changed")
    output = [
        np.asarray(function(values[start : start + chunk_rows]), dtype=np.float32)
        for start in range(0, len(values), chunk_rows)
    ]
    if any(len(cell) <= 0 for cell in output):
        raise Round0196Error("fixed-chunk inference produced an empty cell")
    return np.concatenate(output, axis=0)


def diagnose_execution(value: Mapping[str, Any]) -> dict[str, Any]:
    if (
        value.get("schema") != "round0196-grease-batch-stable-cpu-execution-v1"
        or value.get("device") != "cpu"
        or int(value.get("probe_rows", -1)) != INFERENCE_CHUNK_ROWS
        or int(value.get("query_rows", -1)) < 2 * INFERENCE_CHUNK_ROWS
        or value.get("source_checkpoint_round") != "0181"
    ):
        raise Round0196Error("R0196 CPU execution contract changed")
    candidates = value.get("candidates") or {}
    if set(candidates) != {"baseline", "fixed_grease", "fixed_grease_and_pumap"}:
        raise Round0196Error("R0196 candidate set changed")
    for name, cell in candidates.items():
        for key in ("grease_batch_max_abs_error", "numap_batch_max_abs_error"):
            number = float(cell[key])
            if not math.isfinite(number) or number < 0:
                raise Round0196Error(f"R0196 {name}/{key} is invalid")
    grease_pass = (
        float(candidates["fixed_grease"]["numap_batch_max_abs_error"])
        <= RELOAD_TOLERANCE
    )
    both_pass = (
        float(candidates["fixed_grease_and_pumap"]["numap_batch_max_abs_error"])
        <= RELOAD_TOLERANCE
    )
    selected = value.get("selected_patch")
    expected = (
        "fixed-256-row-grease-network"
        if grease_pass
        else (
            "fixed-256-row-grease-and-pumap-networks" if both_pass else None
        )
    )
    if selected != expected:
        raise Round0196Error("R0196 did not select the minimal passing patch")
    baseline_reproduced = (
        float(candidates["baseline"]["numap_batch_max_abs_error"])
        > RELOAD_TOLERANCE
    )
    passed = expected is not None and baseline_reproduced
    return {
        "outcome": (
            "grease-batch-stable-inference-patch-qualified"
            if passed
            else "grease-batch-stable-inference-not-qualified"
        ),
        "passed": passed,
        "baseline_failure_reproduced": baseline_reproduced,
        "selected_patch": selected,
        "reload_tolerance": RELOAD_TOLERANCE,
        "f2_gpu_baseline_activated": passed,
        "f3_negative_closure_activated": not passed,
        "additional_debug_or_f4_authorized": False,
        "capabilities_releasable": [
            PATCH_CAPABILITY if passed else NEGATIVE_CAPABILITY
        ],
    }


__all__ = [
    "CAPABILITY",
    "INFERENCE_CHUNK_ROWS",
    "NEGATIVE_CAPABILITY",
    "PATCH_CAPABILITY",
    "RELOAD_TOLERANCE",
    "ROUND_ID",
    "Round0196Error",
    "diagnose_execution",
    "fixed_chunks",
]
