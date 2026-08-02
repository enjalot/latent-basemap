"""Assemble the reviewed raw/drop-only four-seed calibration matrix."""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_write_new_json, create_fresh_directory
from basemap.round0108_evaluation import seal, validate_seal
from basemap.round0140_subsystem_bisection import CURRENT_GRAPH_CURRENT_HOST, METRICS
from basemap.round0149_drop_only import TREATMENT
from basemap.round0159_seed_margin_proposal import (
    CAPABILITY,
    MEASURES,
    ROUND_ID,
    SEEDS,
    Round0159Error,
    build_margin_proposal,
)


def _read(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    actual = expected_input_signature(str(expected.get("canonical_path") or ""))
    if actual != dict(expected):
        raise Round0159Error(f"{label} bytes changed")
    with open(actual["canonical_path"], encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0159Error(f"{label} is not a JSON object")
    validate_seal(value, label=label)
    return value


def _functional(cell: Mapping[str, Any], *, seed: int, label: str) -> dict[str, float]:
    if cell.get("seed") != seed:
        raise Round0159Error(f"{label} seed changed")
    metrics = cell.get("decision_metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != set(METRICS):
        raise Round0159Error(f"{label} functional metrics changed")
    return {key: float(metrics[key]) for key in METRICS}


def _new_seed_cells(
    evidence: Mapping[str, Any], *, round_id: str, label: str
) -> dict[int, dict[str, float]]:
    if (
        evidence.get("round_id") != round_id
        or evidence.get("margin_or_floor_proposed") is not False
        or evidence.get("floor_changed") is not False
    ):
        raise Round0159Error(f"{label} evidence semantics changed")
    output: dict[int, dict[str, float]] = {}
    for seed in (44, 45):
        cell = evidence.get("cells", {}).get(f"seed{seed}")
        if not isinstance(cell, Mapping) or cell.get("seed") != seed:
            raise Round0159Error(f"{label} seed {seed} cell changed")
        functional = cell.get("functional_metrics")
        density = cell.get("density_v2")
        if (
            not isinstance(functional, Mapping)
            or set(functional) != set(METRICS)
            or not isinstance(density, Mapping)
            or "correlation" not in density
        ):
            raise Round0159Error(f"{label} seed {seed} measures changed")
        output[seed] = {
            **{key: float(functional[key]) for key in METRICS},
            "density_v2": float(density["correlation"]),
        }
    return output


def run_proposal(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0159Error("R0159 handler received another queue")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise Round0159Error("R0159 must run with CUDA hidden")
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0159 seed margin proposal"
    )
    r0140 = _read(job["r0140_panel"], label="R0140 raw seed-42 panel")
    r0149 = _read(job["r0149_panel"], label="R0149 drop seed-42 panel")
    r0150 = _read(job["r0150_panel"], label="R0150 seed-43 paired panel")
    density = _read(job["r0153_density"], label="R0153 density forensics")
    raw_new = _read(job["r0154_evidence"], label="R0154 raw seed evidence")
    drop_new = _read(job["r0158_evidence"], label="R0158 drop seed evidence")

    raw = {
        42: _functional(
            r0140["cells"][CURRENT_GRAPH_CURRENT_HOST], seed=42, label="raw42"
        ),
        43: _functional(
            r0150["cells"][CURRENT_GRAPH_CURRENT_HOST], seed=43, label="raw43"
        ),
        **_new_seed_cells(raw_new, round_id="0154", label="raw"),
    }
    drop = {
        42: _functional(r0149["cells"][TREATMENT], seed=42, label="drop42"),
        43: _functional(r0150["cells"][TREATMENT], seed=43, label="drop43"),
        **_new_seed_cells(drop_new, round_id="0158", label="drop-only"),
    }
    density_cells = density.get("cells")
    if not isinstance(density_cells, Mapping):
        raise Round0159Error("R0153 density cell matrix changed")
    density_bindings = {
        ("raw", 42): "r0140_current_graph_current_host",
        ("raw", 43): "r0150_raw_seed43",
        ("drop", 42): "r0149_drop_only",
        ("drop", 43): "r0150_drop_only_seed43",
    }
    for (family, seed), key in density_bindings.items():
        cell = density_cells.get(key)
        if not isinstance(cell, Mapping):
            raise Round0159Error(f"R0153 density binding {key} changed")
        target = raw if family == "raw" else drop
        target[seed]["density_v2"] = float(cell["density_v2"]["correlation"])
    if any(set(values) != set(MEASURES) for values in (*raw.values(), *drop.values())):
        raise Round0159Error("R0159 assembled measure matrix changed")

    proposal = build_margin_proposal(raw, drop)
    receipt = seal({
        **proposal,
        "release_sha": active["manifest"]["release_sha"],
        "lineage": {
            key: dict(job[key])
            for key in (
                "r0140_panel",
                "r0149_panel",
                "r0150_panel",
                "r0153_density",
                "r0154_evidence",
                "r0158_evidence",
            )
        },
        "review_bindings": [dict(item) for item in job["review_bindings"]],
        "raw_values": {str(seed): values for seed, values in raw.items()},
        "drop_only_values": {str(seed): values for seed, values in drop.items()},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    })
    atomic_write_new_json(
        os.path.join(output, "seed-margin-proposal.json"), receipt, immutable=True
    )


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if str(job.get("action") or "") != "build_seed_margin_proposal":
        raise Round0159Error("unknown R0159 action")
    run_proposal(active, job)

