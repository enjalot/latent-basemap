"""Frozen design helpers for the R0179 NUMAP 200k OOS baseline."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


ROUND_ID = "0179"
CAPABILITY = "jina-numap-grease-oos-baseline-v1"
ROWS = 200_000
N_QUERIES = 20_000
DIMENSION = 768
HELD_HASH = "0e81ac067567"
K_HIT = 10
LOW_FRACTION = 0.001
NUMAP_VERSION = "0.2.3"
GREASE_VERSION = "0.1.5"
RELOAD_TOLERANCE = 1.0e-4


class Round0179Error(RuntimeError):
    """The registered R0179 NUMAP baseline contract was violated."""


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate_execution(execution: Mapping[str, Any]) -> None:
    versions = execution.get("package_versions")
    config = execution.get("config")
    accounting = execution.get("train_accounting")
    checkpoint = execution.get("checkpoint")
    train_coordinates = execution.get("train_coordinates")
    query_coordinates = execution.get("query_coordinates")
    if (
        execution.get("schema") != "round0179-numap-reference-execution-v1"
        or execution.get("mode") != "real"
        or execution.get("cuda_available") is not True
        or execution.get("train_rows") != ROWS
        or execution.get("query_rows") != N_QUERIES
        or execution.get("dimension") != DIMENSION
        or not isinstance(versions, Mapping)
        or versions.get("numap") != NUMAP_VERSION
        or versions.get("grease-embeddings") != GREASE_VERSION
        or not isinstance(config, Mapping)
        or not isinstance(accounting, Mapping)
        or not isinstance(checkpoint, Mapping)
        or not isinstance(train_coordinates, Mapping)
        or not isinstance(query_coordinates, Mapping)
    ):
        raise Round0179Error("NUMAP execution identity changed")
    required_config = {
        "n_neighbors": 10,
        "min_dist": 0.1,
        "metric": "cosine",
        "n_components": 2,
        "se_dim": 5,
        "se_neighbors": 10,
        "random_state": 42,
        "lr": 1.0e-3,
        "epochs": 10,
        "batch_size": 64,
        "use_se": True,
        "use_residual_connections": True,
        "use_grease": True,
        "grease_batch_size": 1024,
        "grease_lr": 1.0e-3,
        "learn_from_se": True,
        "negative_sample_rate": 5,
        "use_concat": False,
        "use_alpha": False,
        "alpha": 0.0,
        "init_method": "identity",
        "grease_hiddens": [128, 256, 256],
        "use_true_eigenvectors": True,
    }
    for key, expected in required_config.items():
        if config.get(key) != expected:
            raise Round0179Error(f"NUMAP config changed at {key}")
    if (
        accounting.get("selected_pipeline")
        != "numap==0.2.3 official-example GrEASE spectral extension + residual PUMAP encoder"
        or accounting.get("grease_architecture_actual") != [128, 256, 256, 6]
        or int(accounting.get("grease_optimizer_updates", 0)) <= 0
        or int(accounting.get("pumap_optimizer_updates", -1))
        != int(accounting.get("pumap_expected_updates", -2))
        or int(accounting.get("pumap_expected_updates", 0)) != 31_250
        or not _finite_number(checkpoint.get("reload_max_abs_error"))
        or float(checkpoint["reload_max_abs_error"]) > RELOAD_TOLERANCE
        or train_coordinates.get("shape") != [ROWS, 2]
        or query_coordinates.get("shape") != [N_QUERIES, 2]
    ):
        raise Round0179Error("NUMAP execution accounting or reload guard failed")
    for label, summary in (
        ("train", train_coordinates),
        ("query", query_coordinates),
    ):
        standard_deviation = summary.get("axis_standard_deviation")
        if (
            not isinstance(standard_deviation, list)
            or len(standard_deviation) != 2
            or any(not _finite_number(item) or float(item) <= 1.0e-6 for item in standard_deviation)
        ):
            raise Round0179Error(f"NUMAP {label} coordinates collapsed")


def build_synthesis(
    *, cell: Mapping[str, Any], aumap_context: Mapping[str, Any]
) -> dict[str, Any]:
    execution = cell.get("execution")
    metrics = cell.get("heldout_projection")
    if not isinstance(execution, Mapping):
        raise Round0179Error("R0179 execution receipt is absent")
    validate_execution(execution)
    if (
        cell.get("schema") != "round0179-numap-cell-v1"
        or cell.get("round_id") != ROUND_ID
        or cell.get("rows") != ROWS
        or cell.get("held_hash") != HELD_HASH
        or cell.get("n_queries") != N_QUERIES
        or cell.get("guards_passed") is not True
        or not isinstance(metrics, Mapping)
        or not _finite_number(metrics.get("ffr"))
        or not _finite_number(metrics.get("recall_at_10"))
    ):
        raise Round0179Error("R0179 NUMAP cell is incomplete")
    aumap_scales = aumap_context.get("scales")
    aumap_200k = (
        aumap_scales.get("200k") if isinstance(aumap_scales, Mapping) else None
    )
    aumap_metrics = (
        aumap_200k.get("aumap_inverse_distance")
        if isinstance(aumap_200k, Mapping)
        else None
    )
    if (
        aumap_context.get("round_id") != "0175"
        or aumap_context.get("outcome") != "aumap-oos-baseline-measured"
        or not isinstance(aumap_metrics, Mapping)
        or not _finite_number(aumap_metrics.get("ffr"))
        or not _finite_number(aumap_metrics.get("recall_at_10"))
    ):
        raise Round0179Error("accepted R0175 aUMAP context changed")
    comparison = {
        "numap": {
            "ffr": float(metrics["ffr"]),
            "recall_at_10": float(metrics["recall_at_10"]),
        },
        "aumap_inverse_distance": {
            "ffr": float(aumap_metrics["ffr"]),
            "recall_at_10": float(aumap_metrics["recall_at_10"]),
        },
        "numap_minus_aumap": {
            "ffr": float(metrics["ffr"]) - float(aumap_metrics["ffr"]),
            "recall_at_10": float(metrics["recall_at_10"])
            - float(aumap_metrics["recall_at_10"]),
        },
        "comparability": (
            "same frozen 200k high-dimensional rows, same 20k held source IDs, "
            "same exact high-neighbor truth, and same projection metric formulas; "
            "different transductive map teachers, so diagnostic rather than a "
            "pre-registered method-winner test"
        ),
    }
    return {
        "schema": "round0179-numap-synthesis-v1",
        "round_id": ROUND_ID,
        "outcome": "numap-grease-oos-baseline-measured",
        "cell": dict(cell),
        "comparison_to_reviewed_r0175": comparison,
        "selector": "execution-validity only; no quality floor or winner branch",
        "reference_scope": (
            "unmodified numap==0.2.3 and grease-embeddings==0.1.5 package "
            "bytes using the package repository's GrEASE/residual example path"
        ),
        "known_default_path_defect": (
            "R0175 found that the package default learn_from_se path feeds "
            "concatenated spectral+X rows to an encoder sized only for spectral "
            "coordinates; R0179 changes no package bytes and instead uses the "
            "documented example's residual flag plus GrEASE extension"
        ),
        "production_or_publishing": False,
        "capabilities": [CAPABILITY],
    }


__all__ = [
    "CAPABILITY",
    "DIMENSION",
    "GREASE_VERSION",
    "HELD_HASH",
    "K_HIT",
    "LOW_FRACTION",
    "NUMAP_VERSION",
    "N_QUERIES",
    "RELOAD_TOLERANCE",
    "ROUND_ID",
    "ROWS",
    "Round0179Error",
    "build_synthesis",
    "validate_execution",
]
