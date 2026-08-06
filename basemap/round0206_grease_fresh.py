"""Frozen validation and decision logic for the final fresh GrEASE attempt."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .round0108_evaluation import seal
from .round0175_aumap_baseline import HELD_HASHES, N_QUERIES, ROWS, SCALES
from .round0181_fixed_normalization import NORMALIZATION_POLICY


ROUND_ID = "0206"
POSITIVE_CAPABILITY = "jina-grease-fresh-train-oos-baseline-v1"
NEGATIVE_CAPABILITY = "jina-grease-fresh-train-terminal-negative-v1"
REFERENCE_SCHEMA = "round0206-grease-fresh-reference-execution-v1"
CELL_SCHEMA = "round0206-grease-fresh-cell-v1"
SYNTHESIS_SCHEMA = "round0206-grease-fresh-synthesis-v1"
BATCH_TOLERANCE = 1.0e-4
INFERENCE_CHUNK_ROWS = 256
REAL_CONFIG = {
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
    "num_workers": 0,
    "num_gpus": 1,
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


class Round0206Error(RuntimeError):
    """The final fresh-train GrEASE contract changed or is invalid."""


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate_reference(
    execution: Mapping[str, Any], *, scale: str, smoke: bool = False
) -> bool:
    expected_rows = 256 if smoke else ROWS[scale]
    expected_queries = 64 if smoke else N_QUERIES
    versions = execution.get("package_versions") or {}
    config = execution.get("config") or {}
    accounting = execution.get("train_accounting") or {}
    stability = execution.get("batch_stability") or {}
    normalization = execution.get("normalization") or {}
    expected_config = dict(REAL_CONFIG)
    if smoke:
        expected_config.update({
            "epochs": 1,
            "num_gpus": 0,
            "grease_batch_size": 64,
            "grease_hiddens": [32],
            "se_dim": 2,
        })
    if (
        execution.get("schema") != REFERENCE_SCHEMA
        or execution.get("mode") != ("smoke" if smoke else "real")
        or execution.get("scale") != scale
        or execution.get("train_rows") != expected_rows
        or execution.get("query_rows") != expected_queries
        or execution.get("dimension") != (16 if smoke else 768)
        or execution.get("cuda_available") is not (not smoke)
        or versions.get("numap") != "0.2.3"
        or versions.get("grease-embeddings") != "0.1.5"
        or execution.get("checkpoint_restore_performed") is not False
        or execution.get("dill_or_pickle_object_written") is not False
        or accounting.get("selected_pipeline")
        != (
            "numap==0.2.3 GrEASE spectral extension + residual PUMAP with "
            "stored train-time normalization; same-process fresh-model inference"
        )
        or int(accounting.get("grease_optimizer_updates", 0)) <= 0
        or accounting.get("grease_architecture_actual")
        != ([32, 3] if smoke else [128, 256, 256, 6])
        or int(accounting.get("grease_batches_per_full_epoch", -1))
        != math.ceil(0.9 * expected_rows / expected_config["grease_batch_size"])
        or not _finite(accounting.get("grease_completed_epoch_equivalents"))
        or not (
            0.0 < float(accounting["grease_completed_epoch_equivalents"]) <= 200.0
        )
        or int(accounting.get("pumap_optimizer_updates", -1))
        != int(accounting.get("pumap_expected_updates", -2))
        or int(accounting.get("pumap_expected_updates", -1))
        != math.ceil(expected_rows / expected_config["batch_size"])
        * expected_config["epochs"]
        or normalization.get("policy") != NORMALIZATION_POLICY
        or normalization.get("statistics_stored_in_fitted_object") is not True
        or normalization.get("statistics_stored_in_checkpoint") is not False
        or normalization.get("training_rows") != expected_rows
        or normalization.get("features") != (16 if smoke else 768)
        or normalization.get("torch_std_correction") != 1
        or stability.get("full_query_rows") != expected_queries
        or stability.get("chunk_rows") != (16 if smoke else INFERENCE_CHUNK_ROWS)
    ):
        raise Round0206Error(f"R0206 {scale} reference identity changed")
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise Round0206Error(f"R0206 {scale} config changed at {key}")
    errors = []
    for key in ("grease_max_abs_error", "numap_max_abs_error"):
        value = stability.get(key)
        if not _finite(value) or float(value) < 0.0:
            raise Round0206Error(f"R0206 {scale} lacks finite {key}")
        errors.append(float(value))
    passed = max(errors) <= BATCH_TOLERANCE
    if stability.get("passed") is not passed:
        raise Round0206Error(f"R0206 {scale} batch selector changed")
    if passed:
        for key in ("train_coordinates", "query_coordinates"):
            summary = execution.get(key) or {}
            expected_shape = (
                [expected_rows, 2] if key == "train_coordinates"
                else [expected_queries, 2]
            )
            spread = summary.get("axis_standard_deviation") or []
            if (
                summary.get("shape") != expected_shape
                or len(spread) != 2
                or any(not _finite(item) or float(item) <= 1.0e-6 for item in spread)
            ):
                raise Round0206Error(f"R0206 {scale} {key} collapsed")
    return passed


def build_synthesis(
    *, cells: Mapping[str, Mapping[str, Any]], prior_table: Mapping[str, Any]
) -> dict[str, Any]:
    if set(cells) != set(SCALES):
        raise Round0206Error("R0206 synthesis cells changed")
    if (
        prior_table.get("schema")
        != "round0183-heldout-projection-method-table-v1"
        or prior_table.get("round_id") != "0183"
    ):
        raise Round0206Error("R0183 method table changed")
    rows: dict[str, Any] = {}
    first_failure: str | None = None
    for index, scale in enumerate(SCALES):
        cell = cells[scale]
        if (
            cell.get("schema") != CELL_SCHEMA
            or cell.get("round_id") != ROUND_ID
            or cell.get("scale") != scale
            or cell.get("rows") != ROWS[scale]
            or cell.get("held_hash") != HELD_HASHES[scale]
        ):
            raise Round0206Error(f"R0206 {scale} cell identity changed")
        status = cell.get("status")
        if first_failure is not None:
            if (
                status != "skipped-prior-batch-instability"
                or cell.get("prior_failure_scale") != first_failure
                or cell.get("training_performed") is not False
            ):
                raise Round0206Error(f"R0206 {scale} skip branch changed")
            rows[scale] = {
                "rows": ROWS[scale],
                "status": status,
                "prior_failure_scale": first_failure,
            }
            continue
        execution = cell.get("execution") or {}
        passed = validate_reference(execution, scale=scale)
        if cell.get("batch_stability_passed") is not passed:
            raise Round0206Error(f"R0206 {scale} cell selector changed")
        if not passed:
            if status != "batch-instability-measured" or cell.get(
                "heldout_projection"
            ) is not None:
                raise Round0206Error(f"R0206 {scale} negative branch changed")
            first_failure = scale
            rows[scale] = {
                "rows": ROWS[scale],
                "status": status,
                "batch_stability": dict(execution["batch_stability"]),
                "cell_identity_sha256": cell.get("identity_sha256"),
            }
            continue
        metrics = cell.get("heldout_projection") or {}
        if status != "stable-baseline-measured" or not all(
            _finite(metrics.get(key)) for key in ("ffr", "recall_at_10")
        ):
            raise Round0206Error(f"R0206 {scale} stable metrics are absent")
        rows[scale] = {
            "rows": ROWS[scale],
            "status": status,
            "ffr": float(metrics["ffr"]),
            "recall_at_10": float(metrics["recall_at_10"]),
            "batch_stability": dict(execution["batch_stability"]),
            "cell_identity_sha256": cell.get("identity_sha256"),
        }
    passed_all = first_failure is None
    capability = POSITIVE_CAPABILITY if passed_all else NEGATIVE_CAPABILITY
    return seal({
        "schema": SYNTHESIS_SCHEMA,
        "round_id": ROUND_ID,
        "outcome": (
            "grease-fresh-three-scale-baseline-measured"
            if passed_all
            else f"grease-fresh-batch-instability-at-{first_failure}"
        ),
        "capability": capability,
        "positive_baseline_released": passed_all,
        "terminal_negative_released": not passed_all,
        "first_batch_instability_scale": first_failure,
        "rows": rows,
        "extends_method_table_round": "0183",
        "comparison_scope": (
            "same held-out row IDs and canonical projection formulas; independently "
            "fitted teachers make method contrasts descriptive only"
        ),
        "selector": (
            "fresh same-process full-vs-fixed-chunk GrEASE and final NUMAP max "
            f"absolute error <= {BATCH_TOLERANCE}; later scales skip after first miss"
        ),
        "thread_closed_per_campaign": True,
        "checkpoint_restore_performed": False,
        "numap_default_path_repaired": False,
        "additional_attempt_authorized": False,
        "method_winner_claim": False,
        "production_or_publishing": False,
        "training_performed": True,
    })


__all__ = [
    "BATCH_TOLERANCE",
    "CELL_SCHEMA",
    "INFERENCE_CHUNK_ROWS",
    "NEGATIVE_CAPABILITY",
    "POSITIVE_CAPABILITY",
    "REAL_CONFIG",
    "REFERENCE_SCHEMA",
    "ROUND_ID",
    "Round0206Error",
    "SYNTHESIS_SCHEMA",
    "build_synthesis",
    "validate_reference",
]
