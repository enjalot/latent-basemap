"""Fresh-process handlers for the Round 0037 three-cell Jina MRL screen."""
from __future__ import annotations

import os
from typing import Any

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_write_new_json, create_fresh_directory
from basemap.round0037_program import (
    CELL_LABELS,
    DIMENSIONS,
    build_registered_decision,
    parse_cell,
    validate_job_cell,
    x_only_row_ceiling,
)
from experiments import round0027_nodes as inherited


def _configure_inherited_handlers() -> None:
    """Bind R0027's reviewed machinery to R0037's narrower contract."""
    inherited.CELL_LABELS = CELL_LABELS
    inherited.validate_job_cell = validate_job_cell
    inherited.parse_cell = parse_cell
    inherited.CANARY_CELL_LABEL = "d768_s42"


def run_sampler_canary(active: dict[str, Any], job: dict[str, Any]) -> None:
    _configure_inherited_handlers()
    try:
        inherited.run_sampler_canary(active, job)
    except Exception as error:
        output = job["outputs"][0]
        try:
            if not os.path.exists(output):
                create_fresh_directory(
                    output, label="Round 0037 failed canary output")
            verdict_path = os.path.join(output, "verdict.json")
            if not os.path.exists(verdict_path):
                admission_path = os.path.join(output, "admission.json")
                body = {
                    "schema": "round0037-mrl-sampler-canary-verdict-v1",
                    "round_id": "0037",
                    "passed": False,
                    "cell": str(job.get("cell") or ""),
                    "production_config_sha256": str(
                        job.get("production_config_sha256") or ""),
                    "minimum_updates_per_s": float(
                        job.get("minimum_updates_per_s", 90.0)),
                    "failure_stage": "inherited_canary_execution",
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                    "complete_performance_evidence": False,
                    "admission": (
                        expected_input_signature(admission_path)
                        if os.path.isfile(admission_path)
                        else None
                    ),
                }
                atomic_write_new_json(
                    verdict_path, inherited._seal(body), immutable=True)
        except Exception as persistence_error:
            raise RuntimeError(
                "Round 0037 canary failed and its sealed failure verdict "
                f"could not be persisted: {persistence_error}"
            ) from error
        raise


def run_shared_reference(active: dict[str, Any], job: dict[str, Any]) -> None:
    _configure_inherited_handlers()
    inherited.run_shared_reference(active, job)


def run_train(active: dict[str, Any], job: dict[str, Any]) -> None:
    _configure_inherited_handlers()
    inherited.run_train(active, job)


def run_transform(active: dict[str, Any], job: dict[str, Any]) -> None:
    _configure_inherited_handlers()
    inherited.run_transform(active, job)


def run_score(active: dict[str, Any], job: dict[str, Any]) -> None:
    _configure_inherited_handlers()
    inherited.run_score(active, job)


def run_decision(active: dict[str, Any], job: dict[str, Any]) -> None:
    _configure_inherited_handlers()
    output = create_fresh_directory(
        job["outputs"][0], label="Round 0037 decision output")
    cells: dict[str, dict[str, Any]] = {}
    for label in CELL_LABELS:
        paths = job["cell_outputs"][label]
        panel_path = os.path.join(paths["panel"], "panel.json")
        train_path = os.path.join(paths["train"], "train-receipt.json")
        panel = inherited._read_sealed(panel_path)
        train = inherited._read_sealed(train_path)
        dimension, seed = parse_cell(label)
        if (
            panel.get("cell") != label
            or train.get("cell") != label
            or panel.get("round_id") != "0037"
            or train.get("round_id") != "0037"
            or panel.get("numerical_guards_passed") is not True
        ):
            raise RuntimeError(f"Round 0037 decision input {label} changed")
        cells[label] = {
            "dimension": dimension,
            "seed": seed,
            "train": expected_input_signature(train_path),
            "panel": expected_input_signature(panel_path),
            "ffr": float(panel["panel"]["ffr"]),
            "purity_k256": float(panel["panel"]["purity"]["k256"]),
            "purity_k1024": float(panel["panel"]["purity"]["k1024"]),
            "density": float(panel["panel"]["density"]),
            "oos_proj_ffr": float(panel["projection"]["proj_ffr"]),
            "updates_per_s": float(train["updates_per_s"]),
            "peak_reserved_bytes": int(train["memory"]["peak_reserved_bytes"]),
            "pipeline": train["actual_pipeline"],
        }
    decision = build_registered_decision(cells)
    render = inherited._fixed_axis_render(
        os.path.join(output, "fixed-axis-three-cell.png"),
        job["cell_outputs"],
        round_id="0037",
    )
    ceilings = {
        str(dimension): {
            "x_only_rows_at_31GB_fp16": x_only_row_ceiling(dimension),
            "bytes_per_input_row": dimension * 2,
            "note": (
                "X-residency arithmetic only; graph/CDF/model/headroom reduce "
                "the end-to-end pipeline ceiling"
            ),
        }
        for dimension in DIMENSIONS
    }
    body = {
        "schema": "round0037-mrl-seed42-screen-v1",
        "round_id": "0037",
        "release_sha": active["manifest"]["release_sha"],
        "cells": cells,
        "registered_rule": {
            "screen_dimensions": [256, 384],
            "oos_proj_ffr_ratio_min": 0.85,
            "transductive_ffr_delta_min": -0.05,
            "all_numerical_and_execution_guards_required": True,
            "screen_cannot_adopt_a_dimension": True,
        },
        "qualification": decision["qualification"],
        "shortlisted_dimensions": decision["shortlisted_dimensions"],
        "decision": decision["decision"],
        "adopted_input_dimension": None,
        "seed43_completion_required_for_adoption": bool(
            decision["shortlisted_dimensions"]),
        "x_residency_ceiling_arithmetic": ceilings,
        "fixed_axis_render": render,
        "literal_prefix_proof": inherited._read_sealed(os.path.join(
            job["shared_reference_output"], "literal-prefix-proof.json")),
    }
    atomic_write_new_json(
        os.path.join(output, "mrl-seed42-screen-v1.json"),
        inherited._seal(body),
        immutable=True,
    )
