"""Round 0038: seed-43 completion of the shortlisted Jina MRL screen."""
from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np

from .artifact_identity import canonical_json, sha256_bytes
from . import round0027_program as source


ROUND_ID = "0038"
DIMENSIONS = (768, 384)
SEEDS = (42, 43)
CELL_LABELS = ("d768_s43", "d384_s43")
DECISION_CELL_LABELS = tuple(
    f"d{dimension}_s{seed}"
    for dimension in DIMENSIONS
    for seed in SEEDS
)
CANARY_MINIMUM_UPDATES_PER_S = 90.0
TRAIN_MINIMUM_UPDATES_PER_S = 75.0
TRAIN_WARNING_UPDATES_PER_S = 90.0

# Keep the reviewed R0027/R0037 science tuple exactly fixed.  This successor
# changes only the registered seed and omits the rejected 256-dimensional arm.
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
        raise ValueError(f"unknown Round 0038 cell: {label!r}")
    dimension_text, seed_text = label.split("_")
    return int(dimension_text[1:]), int(seed_text[1:])


def graph_manifest_for_dimension(dimension: int) -> dict[str, Any]:
    if dimension not in DIMENSIONS:
        raise ValueError(f"unsupported Round 0038 dimension: {dimension}")
    manifest = copy.deepcopy(source.graph_manifest_for_dimension(dimension))
    manifest["verified_by"] = "round0038-queue-local-adapter-v1"
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
    dimension, seed = parse_cell(label)
    config["schema"] = f"round0038-{label}-production-config-v1"
    config["phrase"] = (
        f"2M jina MRL projector seed-{seed} completion input {dimension}d; "
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
        raise ValueError(f"Round 0038 {label} production config changed")
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
    """Evaluate the original two-seed R0027 rule for the 384d shortlist."""
    if set(cells) != set(DECISION_CELL_LABELS):
        raise ValueError(
            "Round 0038 decision requires the 768d and 384d cells at both seeds"
        )
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
        dimension_text, seed_text = label.split("_")
        dimension = int(dimension_text[1:])
        seed = int(seed_text[1:])
        if (
            dimension not in DIMENSIONS
            or seed not in SEEDS
            or cell.get("dimension") != dimension
            or cell.get("seed") != seed
            or any(
                not isinstance(cell.get(metric), (int, float))
                or not np.isfinite(cell[metric])
                for metric in metrics
            )
        ):
            raise ValueError(f"Round 0038 decision cell is malformed: {label}")

    means: dict[int, dict[str, float]] = {}
    for dimension in DIMENSIONS:
        members = [cells[f"d{dimension}_s{seed}"] for seed in SEEDS]
        means[dimension] = {
            metric: float(np.mean([item[metric] for item in members]))
            for metric in metrics
        }
    control_values = [cells[f"d768_s{seed}"]["ffr"] for seed in SEEDS]
    control_spread = float(max(control_values) - min(control_values))
    oos_ratio = (
        means[384]["oos_proj_ffr"] / means[768]["oos_proj_ffr"]
        if means[768]["oos_proj_ffr"] > 0
        else 0.0
    )
    transductive_floor = means[768]["ffr"] - control_spread
    checks = {
        "oos_at_least_90pct_control": oos_ratio >= 0.90,
        "transductive_within_control_seed_spread": (
            means[384]["ffr"] >= transductive_floor
        ),
    }
    qualified = all(checks.values())
    return {
        "seed_means": {str(key): value for key, value in means.items()},
        "control_768_ffr_seed_spread_max_minus_min": control_spread,
        "qualification": {
            "384": {
                "qualified": qualified,
                "checks": checks,
                "oos_ratio_to_768": oos_ratio,
                "transductive_ffr_floor": transductive_floor,
            }
        },
        "decision": "adopt-384d" if qualified else "reject-384d",
        "adopted_input_dimension": 384 if qualified else None,
    }
