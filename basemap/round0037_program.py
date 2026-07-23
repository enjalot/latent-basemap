"""Round 0037: bounded seed-42 Jina MRL input-prefix screen."""
from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np

from .artifact_identity import canonical_json, sha256_bytes
from . import round0027_program as source


ROUND_ID = "0037"
DIMENSIONS = source.DIMENSIONS
CELL_LABELS = ("d768_s42", "d384_s42", "d256_s42")
CANARY_MINIMUM_UPDATES_PER_S = 90.0
TRAIN_MINIMUM_UPDATES_PER_S = 75.0
TRAIN_WARNING_UPDATES_PER_S = 90.0

# The successor deliberately reuses the reviewed R0027 immutable science tuple.
ROWS = source.ROWS
SOURCE_DIMENSION = source.SOURCE_DIMENSION
TRAIN_PATH = source.TRAIN_PATH
TRAIN_SHA256 = source.TRAIN_SHA256
TRAIN_BYTES = source.TRAIN_BYTES
SOURCE_4M_PATH = source.SOURCE_4M_PATH
SOURCE_4M_SHA256 = source.SOURCE_4M_SHA256
SOURCE_4M_BYTES = source.SOURCE_4M_BYTES
PREFIX_PAYLOAD_SHA256 = source.PREFIX_PAYLOAD_SHA256
GRAPH_PATH = source.GRAPH_PATH
GRAPH_SHA256 = source.GRAPH_SHA256
GRAPH_BYTES = source.GRAPH_BYTES
GRAPH_EDGES = source.GRAPH_EDGES
GRAPH_K = source.GRAPH_K
GRAPH_ENDPOINT_PROBE = source.GRAPH_ENDPOINT_PROBE
CENTROIDS = source.CENTROIDS
QUERY_ROWS = source.QUERY_ROWS
SUCCESSFUL_UPDATES = source.SUCCESSFUL_UPDATES
CANARY_UPDATES = source.CANARY_UPDATES
PANEL_ANCHORS = source.PANEL_ANCHORS
PANEL_FRACTION = source.PANEL_FRACTION
PANEL_SEED = source.PANEL_SEED

query_row_ids = source.query_row_ids
input_array = source.input_array
cosine_truth_array = source.cosine_truth_array
preprocessing_for_dimension = source.preprocessing_for_dimension
x_only_row_ceiling = source.x_only_row_ceiling


def parse_cell(label: str) -> tuple[int, int]:
    if label not in CELL_LABELS:
        raise ValueError(f"unknown Round 0037 cell: {label!r}")
    dimension_text, seed_text = label.split("_")
    return int(dimension_text[1:]), int(seed_text[1:])


def graph_manifest_for_dimension(dimension: int) -> dict[str, Any]:
    manifest = copy.deepcopy(source.graph_manifest_for_dimension(dimension))
    manifest["verified_by"] = "round0037-queue-local-adapter-v1"
    truth = manifest["graph_construction_truth"]
    truth.pop("shared_across_all_six_cells", None)
    truth["shared_across_registered_cells"] = list(CELL_LABELS)
    return manifest


def train_config_for_cell(
    label: str,
    *,
    graph_manifest_path: str,
    graph_manifest_sha256: str,
) -> tuple[dict[str, Any], str]:
    parse_cell(label)
    config, _ = source.train_config_for_cell(
        label,
        graph_manifest_path=graph_manifest_path,
        graph_manifest_sha256=graph_manifest_sha256,
    )
    config = copy.deepcopy(config)
    dimension, _ = parse_cell(label)
    config["schema"] = f"round0037-{label}-production-config-v1"
    config["phrase"] = (
        f"2M jina MRL projector seed-42 screen input {dimension}d; "
        "shared full-768d fuzzy graph truth"
    )
    execution = config["execution"]
    execution["minimum_train_upd_s"] = TRAIN_MINIMUM_UPDATES_PER_S
    execution["warning_train_upd_s"] = TRAIN_WARNING_UPDATES_PER_S
    return config, sha256_bytes(canonical_json(config))


def validate_job_cell(job: dict[str, Any]) -> dict[str, Any]:
    label = str(job.get("cell") or "")
    manifest_path = str(job.get("graph_manifest_path") or "")
    manifest_sha = str(job.get("graph_manifest_sha256") or "")
    expected, digest = train_config_for_cell(
        label,
        graph_manifest_path=manifest_path,
        graph_manifest_sha256=manifest_sha,
    )
    if (
        job.get("production_config") != json.loads(canonical_json(expected))
        or job.get("production_config_sha256") != digest
    ):
        raise ValueError(f"Round 0037 {label} production config changed")
    dimension, seed = parse_cell(label)
    return {
        "label": label,
        "dimension": dimension,
        "seed": seed,
        "train_config": expected,
        "train_config_sha256": digest,
    }


def build_registered_decision(
    cells: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate the registered one-seed screen without claiming adoption."""
    if set(cells) != set(CELL_LABELS):
        raise ValueError("Round 0037 decision requires exactly three cells")
    metrics = (
        "ffr",
        "purity_k256",
        "purity_k1024",
        "density",
        "oos_proj_ffr",
        "updates_per_s",
        "peak_reserved_bytes",
    )
    for label, cell in cells.items():
        dimension, seed = parse_cell(label)
        if (
            cell.get("dimension") != dimension
            or cell.get("seed") != seed
            or any(
                not isinstance(cell.get(metric), (int, float))
                or not np.isfinite(cell[metric])
                for metric in metrics
            )
        ):
            raise ValueError(f"Round 0037 decision cell is malformed: {label}")

    control = cells["d768_s42"]
    qualifications: dict[int, dict[str, Any]] = {}
    shortlisted: list[int] = []
    for dimension in (256, 384):
        candidate = cells[f"d{dimension}_s42"]
        ratio = (
            float(candidate["oos_proj_ffr"]) / float(control["oos_proj_ffr"])
            if float(control["oos_proj_ffr"]) > 0
            else 0.0
        )
        delta = float(candidate["ffr"]) - float(control["ffr"])
        checks = {
            "oos_at_least_85pct_control": ratio >= 0.85,
            "transductive_ffr_delta_at_least_minus_0_05": delta >= -0.05,
        }
        passed = all(checks.values())
        qualifications[dimension] = {
            "shortlisted": passed,
            "checks": checks,
            "oos_ratio_to_768": ratio,
            "transductive_ffr_delta_to_768": delta,
        }
        if passed:
            shortlisted.append(dimension)

    return {
        "control_dimension": 768,
        "control_seed": 42,
        "qualification": {
            str(key): value for key, value in qualifications.items()
        },
        "shortlisted_dimensions": shortlisted,
        "decision": (
            "seed43-completion-required"
            if shortlisted
            else "negative-screen-stop"
        ),
        "adopted_input_dimension": None,
    }
