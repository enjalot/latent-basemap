"""Contract tests for conditional R0197."""
from __future__ import annotations

from copy import deepcopy

import pytest

from basemap.round0108_evaluation import seal
from basemap.round0175_aumap_baseline import HELD_HASHES, ROWS, SCALES
from basemap.round0197_grease_baseline import (
    Round0197Error,
    build_synthesis,
    validate_execution,
)
from experiments.prepare_round0197_queue import GPU_HOURS_MAXIMUM, P90_SECONDS


PATCH = "fixed-256-row-grease-network"


def _execution(scale: str, patch: str = PATCH) -> dict:
    rows = ROWS[scale]
    return {
        "schema": "round0197-grease-batch-stable-reference-execution-v1",
        "mode": "real",
        "scale": scale,
        "train_rows": rows,
        "query_rows": 20_000,
        "dimension": 768,
        "cuda_available": True,
        "inference_patch": {
            "source_capability": "jina-grease-batch-stable-inference-patch-v1",
            "selected_patch": patch,
            "chunk_rows": 256,
        },
        "base_execution": {
            "schema": "round0181-numap-fixed-normalization-execution-v1",
            "mode": "real",
            "train_rows": rows,
            "query_rows": 20_000,
            "dimension": 768,
            "package_versions": {
                "numap": "0.2.3",
                "grease-embeddings": "0.1.5",
            },
            "config": {
                "random_state": 42,
                "epochs": 10,
                "batch_size": 64,
                "grease_batch_size": 1024,
                "use_grease": True,
                "use_residual_connections": True,
            },
            "checkpoint": {
                "reload_full_max_abs_error": 0.0,
                "reload_batch_max_abs_error": 1e-6,
            },
            "train_accounting": {
                "selected_pipeline": "numap==0.2.3 GrEASE spectral extension + residual PUMAP with stored train-time feature normalization",
                "grease_optimizer_updates": 10,
                "pumap_optimizer_updates": ((rows + 63) // 64) * 10,
                "pumap_expected_updates": ((rows + 63) // 64) * 10,
            },
        },
    }


def _cell(scale: str) -> dict:
    return seal({
        "schema": "round0197-grease-batch-stable-cell-v1",
        "round_id": "0197",
        "scale": scale,
        "rows": ROWS[scale],
        "n_queries": 20_000,
        "held_hash": HELD_HASHES[scale],
        "selected_patch": PATCH,
        "execution": _execution(scale),
        "heldout_projection": {"ffr": 0.5, "recall_at_10": 0.01},
        "performance": {"reference_seconds": 10.0},
        "guards_passed": True,
    })


def test_all_registered_scales_validate() -> None:
    for scale in SCALES:
        validate_execution(_execution(scale), scale=scale, selected_patch=PATCH)


def test_reload_drift_and_patch_mismatch_fail_closed() -> None:
    value = _execution("200k")
    value["base_execution"]["checkpoint"]["reload_batch_max_abs_error"] = 2e-4
    with pytest.raises(Round0197Error, match="reload_batch"):
        validate_execution(value, scale="200k", selected_patch=PATCH)
    value = _execution("200k")
    value["inference_patch"]["selected_patch"] = (
        "fixed-256-row-grease-and-pumap-networks"
    )
    with pytest.raises(Round0197Error, match="identity"):
        validate_execution(value, scale="200k", selected_patch=PATCH)


def test_accounting_is_scale_specific() -> None:
    value = _execution("2m")
    value["base_execution"]["train_accounting"]["pumap_expected_updates"] -= 1
    with pytest.raises(Round0197Error, match="accounting"):
        validate_execution(value, scale="2m", selected_patch=PATCH)


def test_synthesis_extends_table_without_winner_claim() -> None:
    table = {
        "schema": "round0183-heldout-projection-method-table-v1",
        "round_id": "0183",
    }
    result = build_synthesis(
        cells={scale: _cell(scale) for scale in SCALES},
        prior_table=table,
        selected_patch=PATCH,
    )
    assert result["outcome"] == "grease-batch-stable-oos-baseline-measured"
    assert set(result["rows"]) == set(SCALES)
    assert result["diagnostic_only"] is True
    assert result["numap_revived"] is False
    assert result["additional_retry_or_f4_authorized"] is False


def test_missing_scale_or_bad_metric_fails_closed() -> None:
    cells = {scale: _cell(scale) for scale in SCALES}
    cells.pop("500k")
    with pytest.raises(Round0197Error, match="inputs"):
        build_synthesis(
            cells=cells,
            prior_table={
                "schema": "round0183-heldout-projection-method-table-v1",
                "round_id": "0183",
            },
            selected_patch=PATCH,
        )
    cells = {scale: _cell(scale) for scale in SCALES}
    cells["200k"] = deepcopy(cells["200k"])
    cells["200k"]["heldout_projection"]["ffr"] = 2.0
    with pytest.raises(Round0197Error, match="outside"):
        build_synthesis(
            cells=cells,
            prior_table={
                "schema": "round0183-heldout-projection-method-table-v1",
                "round_id": "0183",
            },
            selected_patch=PATCH,
        )


def test_registered_gpu_p90_fits_campaign_cap() -> None:
    assert GPU_HOURS_MAXIMUM == 0.5
    assert sum(P90_SECONDS.values()) <= GPU_HOURS_MAXIMUM * 3600
