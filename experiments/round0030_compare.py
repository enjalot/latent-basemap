"""Deterministic R0030 uniform-versus-fuzzy comparison and decision receipt."""
from __future__ import annotations

import json
import os
from typing import Any

from basemap.artifact_identity import canonical_json, expected_input_signature, sha256_bytes
from basemap.output_safety import atomic_write_new_json, create_fresh_directory
from basemap.round0030_program import (
    ARMS,
    CAP_SHA256,
    GRAPH_EFFECTIVE_EDGES,
    GRAPH_PATH,
    GRAPH_RESIDENT_EDGES,
    GRAPH_SHA256,
    OOD_RETENTION_NONINFERIORITY_MARGIN,
    PRIMARY_METRIC,
    R0023_LAYOUT_SHA256,
    R0023_SEED_SPREAD,
    R0023_SEED_VALUES,
    R0023_ZERO_RADIUS_ROWS,
    train_config_for_arm,
)


def _read_sealed(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    body = {key: item for key, item in value.items() if key != "identity_sha256"}
    if value.get("identity_sha256") != sha256_bytes(canonical_json(body)):
        raise RuntimeError(f"sealed receipt changed: {path}")
    return value


def _quality(panel: dict[str, Any]) -> dict[str, float]:
    return {
        "ffr": float(panel["panel"]["ffr"]),
        "density": float(panel["panel"]["density"]),
        "purity_k256": float(panel["panel"]["purity"]["k256"]),
        "purity_k1024": float(panel["panel"]["purity"]["k1024"]),
        "recall_at_10": float(panel["recall@10"]),
        "recall_at_50": float(panel["recall@50"]),
        "projection_ffr": float(panel["projection"]["proj_ffr"]),
    }


def _ood(panel: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "status": item.get("status"),
            "retention": (
                float(item["retention"])
                if isinstance(item.get("retention"), (int, float))
                else None
            ),
            "verdict": item.get("verdict"),
        }
        for name, item in panel["probes"].items()
    }


def build_comparison(
    *,
    release_sha: str,
    arm_outputs: dict[str, dict[str, str]],
) -> dict[str, Any]:
    shared_graph = expected_input_signature(GRAPH_PATH)
    if shared_graph["sha256"] != GRAPH_SHA256:
        raise RuntimeError("R0030 shared graph bytes changed before comparison")
    cells: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        paths = arm_outputs[arm]
        train_path = os.path.join(paths["train"], "train-receipt.json")
        panel_path = os.path.join(paths["panel"], "panel.json")
        ood_path = os.path.join(paths["ood_panel"], "universality-panel-v1.json")
        train = _read_sealed(train_path)
        panel = _read_sealed(panel_path)
        ood_panel = _read_sealed(ood_path)
        config, config_sha = train_config_for_arm(arm)
        stats = train["train_stats"]
        expected_stamp = config["execution"]["expected_pipeline_stamp"]
        stamp_mismatches = {
            key: {"expected": expected, "observed": stats.get(f"pipeline_{key}")}
            for key, expected in expected_stamp.items()
            if stats.get(f"pipeline_{key}") != expected
        }
        exact = {
            "production_config_sha256": config_sha,
            "n_pos_edges": GRAPH_EFFECTIVE_EDGES,
            "positive_lr_optimizer_steps": 500_000,
            "optimizer_steps_succeeded": 500_000,
            "budget_satisfied": True,
            "pipeline_multiplicity_cap_artifact_sha256": CAP_SHA256,
            "pipeline_multiplicity_positive_edges_resident": GRAPH_RESIDENT_EDGES,
            "pipeline_multiplicity_positive_edges_effective": GRAPH_EFFECTIVE_EDGES,
        }
        observed = {
            "production_config_sha256": train.get("production_config_sha256"),
            **{key: stats.get(key) for key in exact if key != "production_config_sha256"},
        }
        exact_mismatches = {
            key: {"expected": expected, "observed": observed.get(key)}
            for key, expected in exact.items()
            if observed.get(key) != expected
        }
        verified_graph = stats.get("verified_hashes", {}).get("graph_sha256")
        model = train.get("model")
        model_ok = bool(
            isinstance(model, dict)
            and expected_input_signature(model.get("canonical_path", "")) == model
        )
        panel_graph = panel.get("registered_inputs", {}).get("graph")
        ood_map = ood_panel.get("map")
        ood_model = ood_map.get("model") if isinstance(ood_map, dict) else None
        if (
            stamp_mismatches
            or exact_mismatches
            or verified_graph != GRAPH_SHA256
            or not model_ok
            or panel_graph != shared_graph
            or ood_model != model
        ):
            raise RuntimeError(
                f"R0030 {arm} execution receipt mismatch: "
                f"stamp={stamp_mismatches}, exact={exact_mismatches}, "
                f"graph={verified_graph}, model_ok={model_ok}, "
                f"panel_graph_ok={panel_graph == shared_graph}, "
                f"ood_model_ok={ood_model == model}"
            )
        if panel.get("production_config_sha256") != config_sha:
            raise RuntimeError(f"R0030 {arm} panel/config binding changed")
        cells[arm] = {
            "train": expected_input_signature(train_path),
            "model": train["model"],
            "panel": expected_input_signature(panel_path),
            "ood_panel": expected_input_signature(ood_path),
            "quality": _quality(panel),
            "quality_selector_passed": bool(panel.get("selector_passed")),
            "ood": _ood(ood_panel),
            "pipeline": {key: stats.get(f"pipeline_{key}") for key in expected_stamp},
        }

    uniform, fuzzy = cells["uniform"], cells["fuzzy"]
    quality_deltas = {
        name: fuzzy["quality"][name] - uniform["quality"][name]
        for name in R0023_SEED_SPREAD
    }
    quality_nonregression = {
        name: quality_deltas[name] >= -spread
        for name, spread in R0023_SEED_SPREAD.items()
        if name != PRIMARY_METRIC
    }
    ood_deltas: dict[str, float | None] = {}
    ood_nonregression: dict[str, bool] = {}
    verdict_rank = {"failure": 0, "amber": 1, "pass": 2}
    for name in sorted(set(uniform["ood"]) & set(fuzzy["ood"])):
        left, right = uniform["ood"][name], fuzzy["ood"][name]
        if left["status"] != right["status"]:
            ood_deltas[name] = None
            ood_nonregression[name] = False
            continue
        if left["status"] != "included":
            ood_deltas[name] = None
            ood_nonregression[name] = True
            continue
        delta = right["retention"] - left["retention"]
        ood_deltas[name] = delta
        ood_nonregression[name] = bool(
            delta >= -OOD_RETENTION_NONINFERIORITY_MARGIN
            and verdict_rank.get(right["verdict"], -1)
            >= verdict_rank.get(left["verdict"], -1)
        )

    primary_delta = quality_deltas[PRIMARY_METRIC]
    primary_spread = R0023_SEED_SPREAD[PRIMARY_METRIC]
    adopt = bool(
        fuzzy["quality_selector_passed"]
        and primary_delta > primary_spread
        and all(quality_nonregression.values())
        and all(ood_nonregression.values())
    )
    reject_reasons: list[str] = []
    if not fuzzy["quality_selector_passed"]:
        reject_reasons.append("fuzzy-quality-floor-failure")
    if primary_delta < -primary_spread:
        reject_reasons.append("ffr-regression-exceeds-r0023-seed-spread")
    reject_reasons.extend(
        f"{name}-regression-exceeds-r0023-seed-spread"
        for name, passed in quality_nonregression.items()
        if not passed
    )
    reject_reasons.extend(
        f"ood-{name}-regression-exceeds-registered-margin-or-verdict-worsens"
        for name, passed in ood_nonregression.items()
        if not passed
    )
    decision = "adopt" if adopt else "reject" if reject_reasons else "inconclusive/replicate"
    return {
        "schema": "round0030-fuzzy-sampling-comparison-v1",
        "round_id": "0030",
        "release_sha": release_sha,
        "shared_graph": shared_graph,
        "shared_graph_sha256_expected": GRAPH_SHA256,
        "shared_resident_edges": GRAPH_RESIDENT_EDGES,
        "shared_effective_retained_source_edges": GRAPH_EFFECTIVE_EDGES,
        "duplicate_cap_sha256": CAP_SHA256,
        "cells": cells,
        "r0023_seed_calibration": {
            "panel_values": {key: list(value) for key, value in R0023_SEED_VALUES.items()},
            "max_minus_min_spread": R0023_SEED_SPREAD,
            "layout_artifact_sha256": R0023_LAYOUT_SHA256,
            "zero_radius_infinite_drift_rows": R0023_ZERO_RADIUS_ROWS,
            "layout_disparity_is_diagnostic_only": True,
        },
        "registered_rule": {
            "primary_metric": PRIMARY_METRIC,
            "adopt_if_primary_improvement_strictly_exceeds_seed_spread": primary_spread,
            "metric_specific_nonregression_bands": R0023_SEED_SPREAD,
            "ood_retention_noninferiority_margin": OOD_RETENTION_NONINFERIORITY_MARGIN,
            "ood_margin_is_preregistered_practical_margin_not_observed_ood_seed_spread": True,
            "fuzzy_quality_floors_required_for_adoption": True,
        },
        "deltas_fuzzy_minus_uniform": {
            "quality": quality_deltas,
            "ood_retention": ood_deltas,
        },
        "checks": {
            "fuzzy_quality_selector_passed": fuzzy["quality_selector_passed"],
            "primary_improvement_exceeds_seed_spread": primary_delta > primary_spread,
            "quality_nonregression": quality_nonregression,
            "ood_nonregression": ood_nonregression,
        },
        "decision": decision,
        "reject_reasons": reject_reasons,
        "paired_replication_required": decision == "inconclusive/replicate",
    }


def run_comparison(
    *, output_root: str, release_sha: str, arm_outputs: dict[str, dict[str, str]]
) -> dict[str, Any]:
    output_root = create_fresh_directory(output_root, label="Round 0030 comparison output")
    body = build_comparison(release_sha=release_sha, arm_outputs=arm_outputs)
    receipt = {**body, "identity_sha256": sha256_bytes(canonical_json(body))}
    path = os.path.join(output_root, "fuzzy-sampling-decision-v1.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}
