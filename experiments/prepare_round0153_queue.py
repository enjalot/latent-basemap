#!/usr/bin/env python3
"""Prepare, but never launch, the R0153 CPU density-forensics queue."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0153_density_forensics import CAPABILITY, ROUND_ID
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe


ROUND_ROOT = "/data/latent-basemap/runs/round-0153"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-cpu-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0153-*.md")
R0108_CALIBRATION = (
    "/data/latent-basemap/runs/round-0108/queue-attempt-3/artifacts/"
    "jina-density-calibration/jina-density-calibration.json"
)
R0119_DENSITY_PANEL = (
    "/data/latent-basemap/runs/round-0119/queue/artifacts/"
    "density-localization-panel/density-localization-panel.json"
)
REVIEW_FILES = {
    "0108": os.path.join(LAB_ROOT, "review-0108-2026-07-30.md"),
    "0119": os.path.join(LAB_ROOT, "review-0119-2026-07-31.md"),
    # The first R0140 attempt is preserved, but only the -01 retry review
    # releases the accepted subsystem-bisection capability.
    "0140": os.path.join(LAB_ROOT, "review-0140-2026-08-01-01.md"),
    "0147": os.path.join(LAB_ROOT, "review-0147-2026-08-01.md"),
    "0149": os.path.join(LAB_ROOT, "review-0149-2026-08-02.md"),
    "0150": os.path.join(LAB_ROOT, "review-0150-2026-08-02.md"),
}

CELL_SPECS = (
    (
        "r0140_current_graph_current_host",
        "0140",
        "current_graph_current_host",
        "historical-row-r0037-current-graph-current-host-seed42",
        "/data/latent-basemap/runs/round-0140/queue-attempt-2/artifacts/functional-panel/functional-bisection.json",
    ),
    (
        "r0140_historical_graph_current_host",
        "0140",
        "historical_graph_current_host",
        "historical-row-r0037-historical-graph-current-host-seed42",
        "/data/latent-basemap/runs/round-0140/queue-attempt-2/artifacts/functional-panel/functional-bisection.json",
    ),
    (
        "r0140_historical_graph_device_reproduction",
        "0140",
        "historical_graph_device_reproduction",
        "historical-row-r0037-historical-graph-device-seed42",
        "/data/latent-basemap/runs/round-0140/queue-attempt-2/artifacts/functional-panel/functional-bisection.json",
    ),
    (
        "r0147_size_preserving",
        "0147",
        "eligible_historical_current_graph_current_host",
        "size-preserving-eligible-historical-row-policy-seed42",
        "/data/latent-basemap/runs/round-0147/queue/artifacts/functional-row-policy-panel/functional-panel.json",
    ),
    (
        "r0149_drop_only",
        "0149",
        "drop_only_historical_current_graph_current_host",
        "drop-only-eligible-historical-row-policy-seed42",
        "/data/latent-basemap/runs/round-0149/queue/artifacts/functional-drop-only-panel/functional-panel.json",
    ),
    (
        "r0150_raw_seed43",
        "0150",
        "current_graph_current_host",
        "historical-row-r0037-current-graph-current-host-seed43",
        "/data/latent-basemap/runs/round-0150/queue-attempt-2/artifacts/raw-seed43-functional-panel/functional-bisection.json",
    ),
    (
        "r0150_drop_only_seed43",
        "0150",
        "drop_only_historical_current_graph_current_host",
        "drop-only-eligible-historical-row-policy-seed43",
        "/data/latent-basemap/runs/round-0150/queue-attempt-2/artifacts/paired-seed43-functional-panel/functional-panel.json",
    ),
)


def _status(path: str) -> str | None:
    with open(path, encoding="utf-8") as handle:
        text = handle.read(8192)
    match = re.search(r"(?m)^status:\s*[\"']?([^\s\"']+)", text)
    return match.group(1) if match else None


def _unique(pattern: str, *, status: str, label: str) -> str:
    paths = [path for path in sorted(glob.glob(pattern)) if _status(path) == status]
    if len(paths) != 1:
        raise RuntimeError(f"R0153 requires one {label}; found {len(paths)}")
    return paths[0]


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _review(round_id: str) -> dict[str, Any]:
    path = REVIEW_FILES[round_id]
    if _status(path) != "accepted":
        raise RuntimeError(f"R0153 requires accepted R{round_id} review")
    return expected_input_signature(path)


def _cells() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cells: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for key, source_round, panel_cell, role, panel_path in CELL_SPECS:
        panel_signature = expected_input_signature(panel_path)
        panel = _read_json(panel_path)
        cell = panel.get("cells", {}).get(panel_cell)
        if (
            panel.get("round_id") != source_round
            or not isinstance(cell, dict)
            or not isinstance(cell.get("coordinates"), dict)
        ):
            raise RuntimeError(f"R0153 source cell changed: {key}")
        coordinate_signature = expected_input_signature(
            cell["coordinates"]["canonical_path"]
        )
        if coordinate_signature != cell["coordinates"]:
            raise RuntimeError(f"R0153 source coordinate binding changed: {key}")
        cells.append({
            "key": key,
            "source_round": source_round,
            "panel_cell": panel_cell,
            "role": role,
            "panel": panel_signature,
            "coordinates": coordinate_signature,
        })
        inputs.extend((panel_signature, coordinate_signature))
    return cells, _dedupe(inputs)


def prepare_round0153(
    *, release_sha: str, queue_root: str = os.path.join(ROUND_ROOT, "queue")
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0153 release SHA must be one full commit")
    round_file = _unique(
        ROUND_FILE_GLOB, status="issued", label="issued R0153 round"
    )
    cells, cell_inputs = _cells()
    calibration = _read_json(R0108_CALIBRATION)
    calibration_signature = expected_input_signature(R0108_CALIBRATION)
    r0119_signature = expected_input_signature(R0119_DENSITY_PANEL)
    reviews = [_review(value) for value in ("0108", "0119", "0140", "0147", "0149", "0150")]
    inherited = [
        dict(calibration[key])
        for key in ("census_receipt", "census", "representative_reference", "arrays")
    ]
    controls = {
        seed: dict(calibration["cells"][seed]["coordinates"])
        for seed in ("seed42", "seed43")
    }
    expected_inputs = _dedupe([
        expected_input_signature(round_file),
        *reviews,
        calibration_signature,
        r0119_signature,
        *inherited,
        *controls.values(),
        *cell_inputs,
    ])
    queue_root = create_fresh_directory(queue_root, label="R0153 CPU queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    output = os.path.join(artifacts, CAPABILITY)
    job = {
        "id": "score_track_a_density_forensics",
        "action": "track_a_density_forensics",
        "handler_module": "experiments.round0153_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(artifacts, "density-forensics.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": 300.0,
        "cpu_workers": 4,
        "r0108_calibration": calibration_signature,
        "r0119_density_panel": r0119_signature,
        "historical_control_coordinates": controls,
        "cells": cells,
        "node_policy": {
            "gpu_required": False,
            "training_performed": False,
            "cpu_heavy": True,
        },
    }
    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=round_file,
        queue_root=queue_root,
        gpu_hours_cap=0.05,
        execution_authority="autonomous-cpu",
        gpu=False,
    )
    queue.update({
        "schema": "round0153-track-a-density-forensics-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "cpu-research-parallel",
        "required_reviews": ["0108", "0119", "0140", "0147", "0149", "0150"],
        "ordering_dependencies": [],
        "capability_dependencies": [
            "jina-density-v2-calibration-v1",
            "jina-density-localization-v1",
            "jina-2m-subsystem-bisection-v1",
            "jina-2m-historical-row-policy-duplicate-control-v1",
            "jina-2m-historical-drop-only-decomposition-v1",
            "jina-2m-drop-only-seed-replication-v1",
        ],
        "capabilities_produced": [CAPABILITY],
        "training_performed": False,
        "jobs": [job],
        "scientific_contract": {
            "question": "does frozen density-v2 restore on the R0140 historical-row cells?",
            "cells": [spec["key"] for spec in cells],
            "metric": "R0108 density-v2 on the R0040 representative universe and exact anchors",
            "registered_floor": 0.17589389755990817,
            "classification": {
                "density-restores-with-row-universe": "all three R0140 historical-row cells clear and both R0119 current-2M references fail",
                "density-does-not-restore": "all three R0140 historical-row cells fail",
                "density-mixed-owner-decision-required": "any other pattern",
            },
            "cpu_scorer_must_reproduce_both_r0108_controls": True,
            "legacy_panel_density_is_not_density_v2": True,
            "floor_changed": False,
            "no_training": True,
        },
    })
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=os.path.join(ROUND_ROOT, "queue"))
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0153(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
