"""Frozen contract for the bounded fixed-normalization NUMAP/GrEASE baseline."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


ROUND_ID = "0181"
CAPABILITY = "jina-numap-grease-fixed-normalization-oos-baseline-v1"
ROWS = 200_000
N_QUERIES = 20_000
DIMENSION = 768
HELD_HASH = "0e81ac067567"
K_HIT = 10
LOW_FRACTION = 0.001
NUMAP_VERSION = "0.2.3"
GREASE_VERSION = "0.1.5"
RELOAD_TOLERANCE = 1.0e-4
NORMALIZATION_POLICY = (
    "featurewise train mean/std (torch correction=1) stored in the fitted "
    "GrEASE object and reused for every train/query/reload transform"
)


class Round0181Error(RuntimeError):
    """The registered fixed-normalization baseline contract was violated."""


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate_execution(execution: Mapping[str, Any]) -> None:
    versions = execution.get("package_versions")
    config = execution.get("config")
    accounting = execution.get("train_accounting")
    checkpoint = execution.get("checkpoint")
    normalization = execution.get("normalization")
    train_coordinates = execution.get("train_coordinates")
    query_coordinates = execution.get("query_coordinates")
    if (
        execution.get("schema")
        != "round0181-numap-fixed-normalization-execution-v1"
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
        or not isinstance(normalization, Mapping)
        or not isinstance(train_coordinates, Mapping)
        or not isinstance(query_coordinates, Mapping)
    ):
        raise Round0181Error("R0181 execution identity changed")
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
            raise Round0181Error(f"R0181 NUMAP config changed at {key}")
    if (
        accounting.get("selected_pipeline")
        != (
            "numap==0.2.3 GrEASE spectral extension + residual PUMAP with "
            "stored train-time feature normalization"
        )
        or accounting.get("grease_architecture_actual") != [128, 256, 256, 6]
        or int(accounting.get("grease_optimizer_updates", 0)) <= 0
        or int(accounting.get("pumap_optimizer_updates", -1))
        != int(accounting.get("pumap_expected_updates", -2))
        or int(accounting.get("pumap_expected_updates", 0)) != 31_250
        or normalization.get("policy") != NORMALIZATION_POLICY
        or normalization.get("statistics_stored_in_checkpoint") is not True
        or normalization.get("training_rows") != ROWS
        or normalization.get("features") != DIMENSION
        or normalization.get("torch_std_correction") != 1
        or normalization.get("batch_composition_probe_rows") != 256
        or not _finite(checkpoint.get("reload_full_max_abs_error"))
        or not _finite(checkpoint.get("reload_batch_max_abs_error"))
        or float(checkpoint["reload_full_max_abs_error"]) > RELOAD_TOLERANCE
        or float(checkpoint["reload_batch_max_abs_error"]) > RELOAD_TOLERANCE
        or train_coordinates.get("shape") != [ROWS, 2]
        or query_coordinates.get("shape") != [N_QUERIES, 2]
    ):
        raise Round0181Error("R0181 accounting or fixed-normalization guard failed")
    for label, summary in (("train", train_coordinates), ("query", query_coordinates)):
        deviation = summary.get("axis_standard_deviation")
        if (
            not isinstance(deviation, list)
            or len(deviation) != 2
            or any(not _finite(value) or float(value) <= 1.0e-6 for value in deviation)
        ):
            raise Round0181Error(f"R0181 {label} coordinates collapsed")


def build_synthesis(
    *, cell: Mapping[str, Any], aumap_context: Mapping[str, Any]
) -> dict[str, Any]:
    execution = cell.get("execution")
    metrics = cell.get("heldout_projection")
    if not isinstance(execution, Mapping):
        raise Round0181Error("R0181 execution receipt is absent")
    validate_execution(execution)
    if (
        cell.get("schema") != "round0181-numap-fixed-normalization-cell-v1"
        or cell.get("round_id") != ROUND_ID
        or cell.get("rows") != ROWS
        or cell.get("held_hash") != HELD_HASH
        or cell.get("n_queries") != N_QUERIES
        or cell.get("guards_passed") is not True
        or not isinstance(metrics, Mapping)
        or not _finite(metrics.get("ffr"))
        or not _finite(metrics.get("recall_at_10"))
    ):
        raise Round0181Error("R0181 fixed-normalization cell is incomplete")
    scales = aumap_context.get("scales")
    scale = scales.get("200k") if isinstance(scales, Mapping) else None
    aumap = (
        scale.get("aumap_inverse_distance") if isinstance(scale, Mapping) else None
    )
    if (
        aumap_context.get("round_id") != "0175"
        or aumap_context.get("outcome") != "aumap-oos-baseline-measured"
        or not isinstance(aumap, Mapping)
        or not _finite(aumap.get("ffr"))
        or not _finite(aumap.get("recall_at_10"))
    ):
        raise Round0181Error("accepted R0175 aUMAP context changed")
    comparison = {
        "numap_fixed_normalization": {
            "ffr": float(metrics["ffr"]),
            "recall_at_10": float(metrics["recall_at_10"]),
        },
        "aumap_inverse_distance": {
            "ffr": float(aumap["ffr"]),
            "recall_at_10": float(aumap["recall_at_10"]),
        },
        "numap_minus_aumap": {
            "ffr": float(metrics["ffr"]) - float(aumap["ffr"]),
            "recall_at_10": float(metrics["recall_at_10"])
            - float(aumap["recall_at_10"]),
        },
        "comparability": (
            "same frozen 200k rows, 20k held IDs, high-neighbor truth, and "
            "projection formulas; method comparison remains diagnostic"
        ),
    }
    return {
        "schema": "round0181-numap-fixed-normalization-synthesis-v1",
        "round_id": ROUND_ID,
        "outcome": "numap-grease-fixed-normalization-baseline-measured",
        "cell": dict(cell),
        "comparison_to_reviewed_r0175": comparison,
        "selector": "execution validity only; no quality floor or winner branch",
        "treatment_scope": (
            "R0179 package/config path with only GrEASE feature normalization "
            "changed from transform-batch statistics to stored training statistics"
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
    "NORMALIZATION_POLICY",
    "NUMAP_VERSION",
    "N_QUERIES",
    "RELOAD_TOLERANCE",
    "ROUND_ID",
    "ROWS",
    "Round0181Error",
    "build_synthesis",
    "validate_execution",
]
