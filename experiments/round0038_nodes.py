"""Fresh-process handlers for the Round 0038 seed-43 Jina completion."""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_write_new_json, create_fresh_directory
from basemap.round0038_program import (
    CELL_LABELS,
    DIMENSIONS,
    build_registered_decision,
    parse_cell,
    validate_job_cell,
    x_only_row_ceiling,
)
from experiments import round0027_nodes as inherited


def _configure_inherited_handlers() -> None:
    """Bind the reviewed R0027 machinery to R0038's two seed-43 cells."""
    inherited.CELL_LABELS = CELL_LABELS
    inherited.DIMENSIONS = DIMENSIONS
    inherited.validate_job_cell = validate_job_cell
    inherited.parse_cell = parse_cell
    inherited.CANARY_CELL_LABEL = "d768_s43"


def run_sampler_canary(active: dict[str, Any], job: dict[str, Any]) -> None:
    _configure_inherited_handlers()
    try:
        inherited.run_sampler_canary(active, job)
    except Exception as error:
        output = job["outputs"][0]
        try:
            if not os.path.exists(output):
                create_fresh_directory(
                    output, label="Round 0038 failed canary output")
            verdict_path = os.path.join(output, "verdict.json")
            if not os.path.exists(verdict_path):
                admission_path = os.path.join(output, "admission.json")
                body = {
                    "schema": "round0038-mrl-sampler-canary-verdict-v1",
                    "round_id": "0038",
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
                "Round 0038 canary failed and its sealed failure verdict "
                f"could not be persisted: {persistence_error}"
            ) from error
        raise


def run_train(active: dict[str, Any], job: dict[str, Any]) -> None:
    _configure_inherited_handlers()
    inherited.run_train(active, job)


def run_transform(active: dict[str, Any], job: dict[str, Any]) -> None:
    _configure_inherited_handlers()
    inherited.run_transform(active, job)


def run_score(active: dict[str, Any], job: dict[str, Any]) -> None:
    _configure_inherited_handlers()
    inherited.run_score(active, job)


def _decision_cell(
    label: str,
    *,
    train_path: str,
    panel_path: str,
    expected_round_id: str,
) -> dict[str, Any]:
    panel = inherited._read_sealed(panel_path)
    train = inherited._read_sealed(train_path)
    dimension_text, seed_text = label.split("_")
    dimension = int(dimension_text[1:])
    seed = int(seed_text[1:])
    if (
        panel.get("cell") != label
        or train.get("cell") != label
        or panel.get("round_id") != expected_round_id
        or train.get("round_id") != expected_round_id
        or panel.get("numerical_guards_passed") is not True
    ):
        raise RuntimeError(f"Round 0038 decision input {label} changed")
    return {
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


def _outlier_tail_receipt(
    cell_outputs: dict[str, dict[str, str]],
    render: dict[str, Any],
) -> dict[str, Any]:
    """Make the quantile-clipped render's omitted mass explicit."""
    xlim = render["xlim"]
    ylim = render["ylim"]
    receipt: dict[str, Any] = {}
    for label in CELL_LABELS:
        transform_path = os.path.join(
            cell_outputs[label]["transform"], "transform-receipt.json")
        transform = inherited._read_sealed(transform_path)
        coordinate_path = inherited._verified_signature_path(
            transform.get("coordinates"), label=f"{label} tail coordinate")
        coords = np.load(coordinate_path, mmap_mode="r", allow_pickle=False)
        if coords.shape != (inherited.ROWS, 2) or not np.isfinite(coords).all():
            raise RuntimeError(f"Round 0038 {label} tail input is invalid")
        below_x = int(np.count_nonzero(coords[:, 0] < xlim[0]))
        above_x = int(np.count_nonzero(coords[:, 0] > xlim[1]))
        below_y = int(np.count_nonzero(coords[:, 1] < ylim[0]))
        above_y = int(np.count_nonzero(coords[:, 1] > ylim[1]))
        outside = int(np.count_nonzero(
            (coords[:, 0] < xlim[0])
            | (coords[:, 0] > xlim[1])
            | (coords[:, 1] < ylim[0])
            | (coords[:, 1] > ylim[1])
        ))
        quantiles = np.quantile(
            coords, [0.0, 0.0001, 0.001, 0.999, 0.9999, 1.0], axis=0)
        receipt[label] = {
            "coordinates": expected_input_signature(coordinate_path),
            "rows": int(len(coords)),
            "axis_quantiles": {
                name: [float(value) for value in row]
                for name, row in zip(
                    ("min", "q0.01pct", "q0.1pct",
                     "q99.9pct", "q99.99pct", "max"),
                    quantiles,
                )
            },
            "outside_render": {
                "x_below": below_x,
                "x_above": above_x,
                "y_below": below_y,
                "y_above": above_y,
                "rows_union": outside,
                "fraction_union": outside / len(coords),
            },
        }
    return receipt


def run_decision(active: dict[str, Any], job: dict[str, Any]) -> None:
    _configure_inherited_handlers()
    output = create_fresh_directory(
        job["outputs"][0], label="Round 0038 decision output")
    prior = inherited._read_sealed(job["prior_screen_path"])
    if (
        prior.get("round_id") != "0037"
        or prior.get("shortlisted_dimensions") != [384]
        or prior.get("decision") != "seed43-completion-required"
    ):
        raise RuntimeError("Round 0038 predecessor screen no longer shortlists 384d")

    cells: dict[str, dict[str, Any]] = {}
    for label, paths in job["seed42_cell_outputs"].items():
        cells[label] = _decision_cell(
            label,
            train_path=paths["train_receipt"],
            panel_path=paths["panel"],
            expected_round_id="0037",
        )
    for label in CELL_LABELS:
        paths = job["cell_outputs"][label]
        cells[label] = _decision_cell(
            label,
            train_path=os.path.join(paths["train"], "train-receipt.json"),
            panel_path=os.path.join(paths["panel"], "panel.json"),
            expected_round_id="0038",
        )

    decision = build_registered_decision(cells)
    render = inherited._fixed_axis_render(
        os.path.join(output, "fixed-axis-seed43-cells.png"),
        job["cell_outputs"],
        round_id="0038",
    )
    outlier_tail = _outlier_tail_receipt(job["cell_outputs"], render)
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
        "schema": "round0038-mrl-seed43-completion-v1",
        "round_id": "0038",
        "release_sha": active["manifest"]["release_sha"],
        "predecessor_screen": expected_input_signature(
            job["prior_screen_path"]),
        "cells": cells,
        "registered_rule": {
            "candidate_dimension": 384,
            "seeds": [42, 43],
            "mean_oos_proj_ffr_ratio_min": 0.90,
            "mean_transductive_ffr_floor": (
                "768d seed mean minus 768d max-minus-min seed spread"
            ),
            "all_numerical_and_execution_guards_required": True,
        },
        **decision,
        "x_residency_ceiling_arithmetic": ceilings,
        "fixed_axis_render": render,
        "outlier_tail_receipt": outlier_tail,
        "literal_prefix_proof": inherited._read_sealed(os.path.join(
            job["shared_reference_output"], "literal-prefix-proof.json")),
    }
    atomic_write_new_json(
        os.path.join(output, "mrl-seed43-completion-v1.json"),
        inherited._seal(body),
        immutable=True,
    )
