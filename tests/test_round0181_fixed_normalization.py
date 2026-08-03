"""Contract tests for the bounded fixed-normalization baseline."""
from __future__ import annotations

import pytest

from basemap.round0181_fixed_normalization import (
    CAPABILITY,
    HELD_HASH,
    NORMALIZATION_POLICY,
    N_QUERIES,
    ROWS,
    Round0181Error,
    build_synthesis,
    validate_execution,
)


def _execution() -> dict:
    return {
        "schema": "round0181-numap-fixed-normalization-execution-v1",
        "mode": "real",
        "package_versions": {
            "numap": "0.2.3",
            "grease-embeddings": "0.1.5",
        },
        "config": {
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
        },
        "cuda_available": True,
        "train_rows": ROWS,
        "query_rows": N_QUERIES,
        "dimension": 768,
        "normalization": {
            "policy": NORMALIZATION_POLICY,
            "statistics_stored_in_checkpoint": True,
            "training_rows": ROWS,
            "features": 768,
            "torch_std_correction": 1,
            "batch_composition_probe_rows": 256,
        },
        "train_accounting": {
            "selected_pipeline": (
                "numap==0.2.3 GrEASE spectral extension + residual PUMAP with "
                "stored train-time feature normalization"
            ),
            "grease_architecture_actual": [128, 256, 256, 6],
            "grease_optimizer_updates": 100,
            "pumap_optimizer_updates": 31_250,
            "pumap_expected_updates": 31_250,
        },
        "checkpoint": {
            "reload_full_max_abs_error": 1.0e-7,
            "reload_batch_max_abs_error": 2.0e-7,
        },
        "train_coordinates": {
            "shape": [ROWS, 2],
            "axis_standard_deviation": [1.0, 2.0],
        },
        "query_coordinates": {
            "shape": [N_QUERIES, 2],
            "axis_standard_deviation": [1.0, 2.0],
        },
    }


def _cell() -> dict:
    return {
        "schema": "round0181-numap-fixed-normalization-cell-v1",
        "round_id": "0181",
        "rows": ROWS,
        "n_queries": N_QUERIES,
        "held_hash": HELD_HASH,
        "guards_passed": True,
        "execution": _execution(),
        "heldout_projection": {"ffr": 0.4, "recall_at_10": 0.03},
    }


def _aumap() -> dict:
    return {
        "round_id": "0175",
        "outcome": "aumap-oos-baseline-measured",
        "scales": {
            "200k": {
                "aumap_inverse_distance": {
                    "ffr": 0.323410,
                    "recall_at_10": 0.040625,
                }
            }
        },
    }


def test_execution_requires_full_and_batch_reload_invariance() -> None:
    execution = _execution()
    validate_execution(execution)
    execution["checkpoint"]["reload_batch_max_abs_error"] = 0.1
    with pytest.raises(Round0181Error):
        validate_execution(execution)


def test_execution_requires_stored_training_statistics() -> None:
    execution = _execution()
    execution["normalization"]["statistics_stored_in_checkpoint"] = False
    with pytest.raises(Round0181Error):
        validate_execution(execution)


def test_synthesis_preserves_diagnostic_comparison() -> None:
    synthesis = build_synthesis(cell=_cell(), aumap_context=_aumap())
    assert synthesis["capabilities"] == [CAPABILITY]
    comparison = synthesis["comparison_to_reviewed_r0175"]
    assert comparison["numap_minus_aumap"]["ffr"] == pytest.approx(0.07659)
    assert comparison["numap_minus_aumap"]["recall_at_10"] == pytest.approx(
        -0.010625
    )
    assert synthesis["production_or_publishing"] is False
