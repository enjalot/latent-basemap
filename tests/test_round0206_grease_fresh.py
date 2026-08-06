"""Contract tests for the final fresh-train GrEASE attempt."""
from __future__ import annotations

import json
import math

import pytest

from basemap.round0108_evaluation import seal, validate_seal
from basemap.round0175_aumap_baseline import HELD_HASHES, N_QUERIES, ROWS, SCALES
from basemap.round0181_fixed_normalization import NORMALIZATION_POLICY
from basemap.round0206_grease_fresh import (
    CELL_SCHEMA,
    NEGATIVE_CAPABILITY,
    POSITIVE_CAPABILITY,
    REAL_CONFIG,
    REFERENCE_SCHEMA,
    Round0206Error,
    build_synthesis,
    validate_reference,
)
from experiments.round0206_nodes import run_scale


def _execution(scale: str, *, stable: bool = True) -> dict:
    rows = ROWS[scale]
    config = dict(REAL_CONFIG)
    config["grease_hiddens"] = list(REAL_CONFIG["grease_hiddens"])
    return {
        "schema": REFERENCE_SCHEMA,
        "mode": "real",
        "scale": scale,
        "package_versions": {
            "numap": "0.2.3",
            "grease-embeddings": "0.1.5",
        },
        "config": config,
        "cuda_available": True,
        "train_rows": rows,
        "query_rows": N_QUERIES,
        "dimension": 768,
        "normalization": {
            "policy": NORMALIZATION_POLICY,
            "statistics_stored_in_checkpoint": False,
            "statistics_stored_in_fitted_object": True,
            "training_rows": rows,
            "features": 768,
            "torch_std_correction": 1,
        },
        "batch_stability": {
            "full_query_rows": N_QUERIES,
            "chunk_rows": 256,
            "grease_max_abs_error": 0.0 if stable else 2.0e-4,
            "numap_max_abs_error": 0.0,
            "passed": stable,
        },
        "train_coordinates": (
            {"shape": [rows, 2], "axis_standard_deviation": [1.0, 2.0]}
            if stable else None
        ),
        "query_coordinates": (
            {
                "shape": [N_QUERIES, 2],
                "axis_standard_deviation": [1.0, 2.0],
            }
            if stable else None
        ),
        "checkpoint_restore_performed": False,
        "dill_or_pickle_object_written": False,
        "train_accounting": {
            "selected_pipeline": (
                "numap==0.2.3 GrEASE spectral extension + residual PUMAP with "
                "stored train-time normalization; same-process fresh-model inference"
            ),
            "grease_optimizer_updates": 100,
            "grease_architecture_actual": [128, 256, 256, 6],
            "grease_batches_per_full_epoch": math.ceil(0.9 * rows / 1024),
            "grease_completed_epoch_equivalents": 1.0,
            "pumap_optimizer_updates": ((rows + 63) // 64) * 10,
            "pumap_expected_updates": ((rows + 63) // 64) * 10,
        },
    }


def _cell(scale: str, *, stable: bool = True, prior_failure: str | None = None) -> dict:
    if prior_failure is not None:
        return seal({
            "schema": CELL_SCHEMA,
            "round_id": "0206",
            "scale": scale,
            "rows": ROWS[scale],
            "held_hash": HELD_HASHES[scale],
            "status": "skipped-prior-batch-instability",
            "prior_failure_scale": prior_failure,
            "training_performed": False,
        })
    return seal({
        "schema": CELL_SCHEMA,
        "round_id": "0206",
        "scale": scale,
        "rows": ROWS[scale],
        "held_hash": HELD_HASHES[scale],
        "status": (
            "stable-baseline-measured" if stable else "batch-instability-measured"
        ),
        "batch_stability_passed": stable,
        "execution": _execution(scale, stable=stable),
        "heldout_projection": (
            {"ffr": 0.4, "recall_at_10": 0.03} if stable else None
        ),
        "training_performed": True,
    })


def _prior_table() -> dict:
    return {
        "schema": "round0183-heldout-projection-method-table-v1",
        "round_id": "0183",
    }


def test_reference_requires_exact_fresh_pipeline_and_config() -> None:
    execution = _execution("200k")
    assert validate_reference(execution, scale="200k") is True
    execution["checkpoint_restore_performed"] = True
    with pytest.raises(Round0206Error):
        validate_reference(execution, scale="200k")


def test_reference_returns_registered_batch_failure() -> None:
    execution = _execution("200k", stable=False)
    assert validate_reference(execution, scale="200k") is False
    execution["batch_stability"]["passed"] = True
    with pytest.raises(Round0206Error):
        validate_reference(execution, scale="200k")


def test_positive_synthesis_requires_all_three_stable_cells() -> None:
    synthesis = build_synthesis(
        cells={scale: _cell(scale) for scale in SCALES},
        prior_table=_prior_table(),
    )
    assert synthesis["capability"] == POSITIVE_CAPABILITY
    assert synthesis["positive_baseline_released"] is True
    assert synthesis["thread_closed_per_campaign"] is True
    validate_seal(synthesis, label="test R0206 positive synthesis")


def test_negative_synthesis_requires_later_cells_to_skip() -> None:
    cells = {
        "200k": _cell("200k", stable=False),
        "500k": _cell("500k", prior_failure="200k"),
        "2m": _cell("2m", prior_failure="200k"),
    }
    synthesis = build_synthesis(cells=cells, prior_table=_prior_table())
    assert synthesis["capability"] == NEGATIVE_CAPABILITY
    assert synthesis["first_batch_instability_scale"] == "200k"
    assert synthesis["terminal_negative_released"] is True

    cells["500k"] = _cell("500k")
    with pytest.raises(Round0206Error):
        build_synthesis(cells=cells, prior_table=_prior_table())


def test_scale_handler_seals_skip_without_training(tmp_path) -> None:
    prior = tmp_path / "prior"
    prior.mkdir()
    (prior / "cell.json").write_text(
        json.dumps(_cell("200k", stable=False)), encoding="utf-8"
    )
    output = tmp_path / "next"
    active = {"manifest": {"round_id": "0206", "release_sha": "a" * 40}}
    job = {
        "action": "scale",
        "scale": "500k",
        "prior_output": str(prior),
        "outputs": [str(output)],
    }
    run_scale(active, job)
    value = json.loads((output / "cell.json").read_text(encoding="utf-8"))
    validate_seal(value, label="test R0206 skip")
    assert value["status"] == "skipped-prior-batch-instability"
    assert value["prior_failure_scale"] == "200k"
    assert value["training_performed"] is False
