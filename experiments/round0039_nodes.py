"""Standalone canary and decision handlers for Round 0039."""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from basemap.output_safety import atomic_write_new_json, create_fresh_directory
from basemap.round0039_program import (
    ARMS,
    CAP_EXCLUDED_ROWS,
    CAP_RETAINED_ROWS,
    CAP_SHA256,
    GRAPH_EFFECTIVE_EDGES,
    GRAPH_PATH,
    GRAPH_RESIDENT_EDGES,
    GRAPH_SHA256,
    R0023_SEED_SPREAD,
    UPDATES_BY_ARM,
    train_config_for_arm,
)
from basemap.round0030_program import (
    train_config_for_arm as control_train_config,
)
from experiments import run_round0014_node as inherited
from experiments.round0030_compare import _quality, _read_sealed


def _configure(arm: str) -> None:
    inherited.configure_round0039(job={"arm": arm})


def run_sampler_canary(active: dict[str, Any], job: dict[str, Any]) -> None:
    """Exercise the one shared uniform sampler without allocating a model."""
    _configure(ARMS[0])
    output = create_fresh_directory(
        job["outputs"][0], label="Round 0039 shared sampler canary output")
    import torch

    X = inherited.RoundMaterializedArray()
    pumap = inherited._new_exact_model()
    dataset = loader = batch = None
    try:
        dataset, loader, edges = pumap._prepare_edge_list_training(
            X, GRAPH_PATH, len(X), low_memory=True, random_state=42)
        batch = next(iter(loader))
        torch.cuda.synchronize("cuda")
        src, dst, labels = batch
        pipeline = dict(pumap._pipeline_info)
        expected = inherited.TRAIN_CONFIG["execution"][
            "expected_pipeline_stamp"]
        expected = {
            **expected,
            "multiplicity_policy": "exact_duplicate_cap_one",
            "multiplicity_cap_artifact_sha256": CAP_SHA256,
            "multiplicity_excluded_source_rows": CAP_EXCLUDED_ROWS,
            "multiplicity_retained_rows": CAP_RETAINED_ROWS,
            "multiplicity_positive_edges_resident": GRAPH_RESIDENT_EDGES,
            "multiplicity_positive_edges_effective": GRAPH_EFFECTIVE_EDGES,
            "multiplicity_negative_sampling": "uniform_retained_rows_nonself",
            "multiplicity_positive_destinations": "original_graph_rows",
        }
        mismatches = {
            key: {"expected": value, "observed": pipeline.get(key)}
            for key, value in expected.items()
            if pipeline.get(key) != value
        }
        retained = getattr(loader, "_retained_node_rows_t", None)
        positive_labels = int(torch.count_nonzero(labels == 1.0).item())
        negative_labels = int(torch.count_nonzero(labels == 0.0).item())
        graph_sha = pumap._pipeline_verified_hashes.get("graph_sha256")
        free_bytes, total_bytes = torch.cuda.mem_get_info("cuda")
        headroom_gib = free_bytes / (1024 ** 3)
        if (
            mismatches
            or pumap.model is not None
            or hasattr(pumap, "_train_stats")
            or int(edges) != GRAPH_EFFECTIVE_EDGES
            or int(loader.source_n_pos) != GRAPH_RESIDENT_EDGES
            or int(loader.n_pos) != GRAPH_EFFECTIVE_EDGES
            or int(loader.excluded_positive_edges)
            != GRAPH_RESIDENT_EDGES - GRAPH_EFFECTIVE_EDGES
            or retained is None
            or len(retained) != CAP_RETAINED_ROWS
            or graph_sha != GRAPH_SHA256
            or tuple(src.shape) != (8192, 384)
            or tuple(dst.shape) != (8192, 384)
            or positive_labels != 409
            or negative_labels != 7783
            or not bool(torch.isfinite(src).all())
            or not bool(torch.isfinite(dst).all())
            or not bool(torch.isfinite(labels).all())
            or headroom_gib < 1.5
        ):
            raise RuntimeError(
                "R0039 shared sampler canary failed: "
                f"mismatches={mismatches}, edges={edges}, graph={graph_sha}, "
                f"headroom_gib={headroom_gib}"
            )
        scalar = inherited._fixture_scalar_equivalence(output)
        semantic = inherited._fixture_semantic_render(output)
        arms = {}
        for arm in ARMS:
            config, config_sha = train_config_for_arm(arm)
            if config["execution"]["expected_pipeline_stamp"] != (
                    inherited.TRAIN_CONFIG["execution"][
                        "expected_pipeline_stamp"]):
                raise RuntimeError("R0039 budget arms changed sampler semantics")
            arms[arm] = {
                "passed": True,
                "production_config_sha256": config_sha,
                "successful_positive_lr_updates": UPDATES_BY_ARM[arm],
                "pipeline": pipeline,
                "effective_edges": int(edges),
                "resident_edges": int(loader.source_n_pos),
            }
        body = {
            "schema": "round0039-shared-uniform-sampler-canary-evidence-v1",
            "round_id": "0039",
            "training_performed": False,
            "optimizer_updates": 0,
            "arms": arms,
            "single_sampler_shared_by_budget_arms": True,
            "verified_hashes": pumap._pipeline_verified_hashes,
            "batch": {
                "rows": 8192,
                "features": 384,
                "positive_labels": positive_labels,
                "negative_labels": negative_labels,
                "finite": True,
            },
            "post_setup_memory": {
                "free_bytes": int(free_bytes),
                "total_bytes": int(total_bytes),
                "headroom_gib": headroom_gib,
                "minimum_headroom_gib": 1.5,
            },
            "scorer_scalar_equivalence": scalar,
            "semantic_render_alignment": semantic,
        }
        evidence_path = os.path.join(output, "evidence.json")
        atomic_write_new_json(
            evidence_path, inherited._seal(body), immutable=True)
        verdict = {
            "schema": "round0039-shared-uniform-sampler-canary-verdict-v1",
            "round_id": "0039",
            "passed": True,
            "training_performed": False,
            "optimizer_updates": 0,
            "arms": arms,
            "evidence": expected_input_signature(evidence_path),
        }
        atomic_write_new_json(
            os.path.join(output, "verdict.json"),
            inherited._seal(verdict),
            immutable=True,
        )
    finally:
        if loader is not None and hasattr(loader, "close"):
            loader.close()
        del batch, loader, dataset, pumap, X
        torch.cuda.empty_cache()


def _verified_cell(
    label: str,
    *,
    updates: int,
    train_path: str,
    panel_path: str,
    config: dict[str, Any],
    config_sha: str,
) -> dict[str, Any]:
    train = _read_sealed(train_path)
    panel = _read_sealed(panel_path)
    stats = train["train_stats"]
    expected_stamp = config["execution"]["expected_pipeline_stamp"]
    mismatches = {
        key: {"expected": value, "observed": stats.get(f"pipeline_{key}")}
        for key, value in expected_stamp.items()
        if stats.get(f"pipeline_{key}") != value
    }
    exact = {
        "positive_lr_optimizer_steps": updates,
        "optimizer_steps_succeeded": updates,
        "scheduler_steps": updates,
        "lr_horizon": updates,
        "budget_satisfied": True,
        "n_pos_edges": GRAPH_EFFECTIVE_EDGES,
        "pipeline_multiplicity_cap_artifact_sha256": CAP_SHA256,
        "pipeline_multiplicity_positive_edges_resident": GRAPH_RESIDENT_EDGES,
        "pipeline_multiplicity_positive_edges_effective": GRAPH_EFFECTIVE_EDGES,
    }
    mismatches.update({
        key: {"expected": value, "observed": stats.get(key)}
        for key, value in exact.items()
        if stats.get(key) != value
    })
    model = train.get("model")
    if (
        train.get("production_config_sha256") != config_sha
        or panel.get("production_config_sha256") != config_sha
        or stats.get("verified_hashes", {}).get("graph_sha256")
        != GRAPH_SHA256
        or not isinstance(model, dict)
        or expected_input_signature(model.get("canonical_path", "")) != model
        or mismatches
    ):
        raise RuntimeError(
            f"R0039 {label} execution receipt mismatch: {mismatches}")
    quality = _quality(panel)
    if any(not np.isfinite(value) for value in quality.values()):
        raise RuntimeError(f"R0039 {label} quality receipt is non-finite")
    return {
        "updates": updates,
        "train": expected_input_signature(train_path),
        "model": model,
        "panel": expected_input_signature(panel_path),
        "quality": quality,
        "quality_selector_passed": bool(panel.get("selector_passed")),
        "updates_per_s": float(stats["updates_per_s"]),
        "pipeline": {
            key: stats.get(f"pipeline_{key}") for key in expected_stamp
        },
    }


def build_budget_response(
    *,
    release_sha: str,
    cell_paths: dict[str, dict[str, str]],
    control_paths: dict[str, str],
    scale_panel_path: str,
) -> dict[str, Any]:
    cells: dict[str, dict[str, Any]] = {}
    control_config, control_sha = control_train_config("uniform")
    cells["u500k_control"] = _verified_cell(
        "u500k_control",
        updates=500_000,
        train_path=control_paths["train"],
        panel_path=control_paths["panel"],
        config=control_config,
        config_sha=control_sha,
    )
    for arm in ARMS:
        config, config_sha = train_config_for_arm(arm)
        cells[arm] = _verified_cell(
            arm,
            updates=UPDATES_BY_ARM[arm],
            train_path=os.path.join(
                cell_paths[arm]["train"], "train-receipt.json"),
            panel_path=os.path.join(
                cell_paths[arm]["panel"], "panel.json"),
            config=config,
            config_sha=config_sha,
        )

    scale_panel = _read_sealed(scale_panel_path)
    scale_density = float(scale_panel["panel"]["density"])
    if (
        scale_panel.get("round_id") != "0036"
        or not np.isfinite(scale_density)
        or scale_density != 0.0933
    ):
        raise RuntimeError("R0039 bound R0036 density diagnostic changed")

    control = cells["u500k_control"]["quality"]
    deltas = {
        arm: {
            metric: cells[arm]["quality"][metric] - control[metric]
            for metric in R0023_SEED_SPREAD
        }
        for arm in ARMS
    }
    density_band = R0023_SEED_SPREAD["density"]
    low_delta = deltas["u250k"]["density"]
    high_delta = deltas["u1000k"]["density"]
    budget_sensitive = bool(
        abs(low_delta) > density_band or abs(high_delta) > density_band)
    high_budget_degradation = high_delta < -density_band
    low_budget_improvement = low_delta > density_band
    if high_budget_degradation and low_budget_improvement:
        classification = "monotone-density-degradation-with-update-budget"
    elif high_budget_degradation:
        classification = "higher-budget-density-degradation"
    elif not budget_sensitive:
        classification = "density-flat-within-seed-spread"
    else:
        classification = "mixed-or-low-budget-sensitive"

    return {
        "schema": "round0039-30m-budget-response-v1",
        "round_id": "0039",
        "release_sha": release_sha,
        "shared_graph": expected_input_signature(GRAPH_PATH),
        "cells": cells,
        "r0036_scale_diagnostic": {
            "panel": expected_input_signature(scale_panel_path),
            "rows": 150_000_000,
            "successful_positive_lr_updates": 2_454_507,
            "density": scale_density,
            "cross_scale_context_only": True,
        },
        "deltas_vs_500k_control": deltas,
        "registered_rule": {
            "primary_metric": "density",
            "materiality_band": density_band,
            "materiality_source": "R0023 30M seed42/43/44 max-minus-min",
            "budget_sensitive_if_either_endpoint_abs_delta_strictly_exceeds_band": True,
            "diagnostic_only_no_budget_adoption": True,
        },
        "checks": {
            "budget_sensitive": budget_sensitive,
            "u1000k_density_degradation_exceeds_band": high_budget_degradation,
            "u250k_density_improvement_exceeds_band": low_budget_improvement,
            "all_cells_pass_numerical_and_standard_quality_selector": all(
                cell["quality_selector_passed"] for cell in cells.values()),
        },
        "classification": classification,
        "interpretation_boundary": (
            "This isolates update horizon on the 30M R0030 graph. It does not "
            "identify the cause of the R0036 150M density collapse by itself "
            "because the 150M data, graph, sampler, and residency path differ."
        ),
    }


def run_budget_response(active: dict[str, Any], job: dict[str, Any]) -> None:
    output = create_fresh_directory(
        job["outputs"][0], label="Round 0039 budget response output")
    body = build_budget_response(
        release_sha=active["manifest"]["release_sha"],
        cell_paths=job["cell_paths"],
        control_paths=job["control_paths"],
        scale_panel_path=job["scale_panel_path"],
    )
    receipt = {
        **body,
        "identity_sha256": sha256_bytes(canonical_json(body)),
    }
    atomic_write_new_json(
        os.path.join(output, "budget-response-v1.json"),
        receipt,
        immutable=True,
    )
