"""Frozen validation and synthesis for the conditional R0197 GrEASE baseline."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .round0108_evaluation import seal
from .round0175_aumap_baseline import HELD_HASHES, N_QUERIES, ROWS, SCALES
from .round0196_grease_batch_stable import (
    PATCH_CAPABILITY,
    RELOAD_TOLERANCE,
)


ROUND_ID = "0197"
CAPABILITY = "jina-grease-batch-stable-oos-baseline-v1"
SELECTED_PATCHES = {
    "fixed-256-row-grease-network",
    "fixed-256-row-grease-and-pumap-networks",
}


class Round0197Error(RuntimeError):
    """The conditional R0197 baseline contract changed or is invalid."""


def _finite_unit(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise Round0197Error(f"{label} is not numeric") from error
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise Round0197Error(f"{label} is outside [0, 1]")
    return result


def validate_execution(
    execution: Mapping[str, Any], *, scale: str, selected_patch: str
) -> None:
    if scale not in SCALES or selected_patch not in SELECTED_PATCHES:
        raise Round0197Error("R0197 scale or patch is not registered")
    base = execution.get("base_execution")
    patch = execution.get("inference_patch")
    if (
        execution.get("schema")
        != "round0197-grease-batch-stable-reference-execution-v1"
        or execution.get("mode") != "real"
        or execution.get("scale") != scale
        or execution.get("train_rows") != ROWS[scale]
        or execution.get("query_rows") != N_QUERIES
        or execution.get("dimension") != 768
        or execution.get("cuda_available") is not True
        or not isinstance(base, Mapping)
        or not isinstance(patch, Mapping)
        or patch.get("qualification_round") != "0200"
        or patch.get("implementation_origin_round") != "0196"
        or patch.get("source_capability") != PATCH_CAPABILITY
        or patch.get("selected_patch") != selected_patch
        or patch.get("chunk_rows") != 256
    ):
        raise Round0197Error("R0197 reference identity changed")
    checkpoint = base.get("checkpoint") or {}
    accounting = base.get("train_accounting") or {}
    config = base.get("config") or {}
    versions = base.get("package_versions") or {}
    if (
        base.get("schema") != "round0181-numap-fixed-normalization-execution-v1"
        or base.get("mode") != "real"
        or base.get("train_rows") != ROWS[scale]
        or base.get("query_rows") != N_QUERIES
        or base.get("dimension") != 768
        or versions.get("numap") != "0.2.3"
        or versions.get("grease-embeddings") != "0.1.5"
        or config.get("random_state") != 42
        or config.get("epochs") != 10
        or config.get("batch_size") != 64
        or config.get("grease_batch_size") != 1024
        or config.get("use_grease") is not True
        or config.get("use_residual_connections") is not True
        or not str(accounting.get("selected_pipeline") or "").startswith(
            "numap==0.2.3 GrEASE spectral extension"
        )
        or int(accounting.get("grease_optimizer_updates", 0)) <= 0
        or int(accounting.get("pumap_optimizer_updates", -1))
        != int(accounting.get("pumap_expected_updates", -2))
        or int(accounting.get("pumap_expected_updates", 0))
        != math.ceil(ROWS[scale] / 64) * 10
    ):
        raise Round0197Error("R0197 training configuration or accounting changed")
    for key in ("reload_full_max_abs_error", "reload_batch_max_abs_error"):
        try:
            error = float(checkpoint[key])
        except (KeyError, TypeError, ValueError) as cause:
            raise Round0197Error(f"R0197 checkpoint lacks {key}") from cause
        if not math.isfinite(error) or error > RELOAD_TOLERANCE:
            raise Round0197Error(f"R0197 checkpoint failed {key}")


def build_synthesis(
    *,
    cells: Mapping[str, Mapping[str, Any]],
    prior_table: Mapping[str, Any],
    selected_patch: str,
) -> dict[str, Any]:
    if set(cells) != set(SCALES) or selected_patch not in SELECTED_PATCHES:
        raise Round0197Error("R0197 synthesis inputs changed")
    if (
        prior_table.get("schema")
        != "round0183-heldout-projection-method-table-v1"
        or prior_table.get("round_id") != "0183"
    ):
        raise Round0197Error("R0197 prior method table changed")
    rows: dict[str, Any] = {}
    for scale in SCALES:
        cell = cells[scale]
        execution = cell.get("execution")
        metrics = cell.get("heldout_projection")
        if not isinstance(execution, Mapping) or not isinstance(metrics, Mapping):
            raise Round0197Error(f"R0197 {scale} cell is incomplete")
        validate_execution(execution, scale=scale, selected_patch=selected_patch)
        if (
            cell.get("schema") != "round0197-grease-batch-stable-cell-v1"
            or cell.get("round_id") != ROUND_ID
            or cell.get("scale") != scale
            or cell.get("rows") != ROWS[scale]
            or cell.get("n_queries") != N_QUERIES
            or cell.get("held_hash") != HELD_HASHES[scale]
            or cell.get("selected_patch") != selected_patch
            or cell.get("guards_passed") is not True
        ):
            raise Round0197Error(f"R0197 {scale} cell identity changed")
        rows[scale] = {
            "rows": ROWS[scale],
            "held_hash": HELD_HASHES[scale],
            "ffr": _finite_unit(metrics.get("ffr"), label=f"{scale} FFR"),
            "recall_at_10": _finite_unit(
                metrics.get("recall_at_10"), label=f"{scale} recall@10"
            ),
            "cell_identity_sha256": cell.get("identity_sha256"),
            "reference_seconds": float(cell["performance"]["reference_seconds"]),
        }
    return seal({
        "schema": "round0197-grease-batch-stable-oos-synthesis-v1",
        "round_id": ROUND_ID,
        "capability": CAPABILITY,
        "outcome": "grease-batch-stable-oos-baseline-measured",
        "selected_patch": selected_patch,
        "source_patch_capability": PATCH_CAPABILITY,
        "rows": rows,
        "extends_method_table_round": "0183",
        "comparison_scope": (
            "same deterministic held-out source IDs and canonical projection "
            "formulas as R0175/R0183; independently fitted teachers make all "
            "method contrasts descriptive, not a method-winner selector"
        ),
        "selector": "execution validity only; no quality floor or winner branch",
        "training_performed": True,
        "diagnostic_only": True,
        "numap_default_toy_fit_repaired": False,
        "numap_revived": False,
        "additional_retry_or_f4_authorized": False,
        "production_or_publishing": False,
    })


__all__ = [
    "CAPABILITY",
    "ROUND_ID",
    "Round0197Error",
    "SELECTED_PATCHES",
    "build_synthesis",
    "validate_execution",
]
