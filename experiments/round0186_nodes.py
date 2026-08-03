"""Build the R0186 prompted U12 graph and derive its exact dose horizon."""
from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_write_new_json, create_fresh_directory
from basemap import round0113_prompt_contrast as prompt_contract
from basemap.round0186_prompted_u12_graph import (
    BASELINE_GRAPH_EDGES,
    BASELINE_SUCCESSFUL_UPDATES,
    CAPABILITY,
    DOSE_PLAN_SCHEMA,
    EVALUATION_ALLOWANCE_SECONDS,
    GRAPH_MEAN_RECALL_FLOOR,
    GRAPH_NPROBE,
    GRAPH_P10_RECALL_FLOOR,
    GRAPH_SCHEMA,
    REFERENCE_THROUGHPUT_ROUND,
    REFERENCE_UPDATES_PER_SECOND,
    ROUND_ID,
    ROWS,
    Round0186Error,
    positive_draws_per_edge,
    successful_updates_for_edges,
)
from experiments import round0169_nodes as q3


def _configure() -> None:
    q3.ROUND_ID = ROUND_ID
    q3.CAPABILITY = CAPABILITY
    q3.GRAPH_SCHEMA = GRAPH_SCHEMA
    q3.Round0169Error = Round0186Error


def run_build_graph(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    _configure()
    q3.run_build_graph(active, job)


def run_derive_dose(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0186Error("R0186 dose planner received another queue")
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0186 exact dose plan"
    )
    started = time.monotonic()
    manifest_path = str(job["graph_manifest"])
    manifest_signature = expected_input_signature(manifest_path)
    manifest = prompt_contract.read_sealed(
        manifest_path, label="R0186 prompted U12 graph manifest"
    )
    graph_signature = expected_input_signature(
        str((manifest.get("graph") or {}).get("canonical_path") or "")
    )
    search = manifest.get("search_qualification") or {}
    selected = (search.get("cells") or {}).get(str(GRAPH_NPROBE), {})
    graph_edges = int(manifest.get("directed_edge_count", -1))
    if (
        manifest.get("schema") != GRAPH_SCHEMA
        or manifest.get("round_id") != ROUND_ID
        or int(manifest.get("retained_rows", -1)) != ROWS
        or graph_signature != manifest.get("graph")
        or selected.get("passed") is not True
        or float(selected.get("mean_recall_at_49", -1.0))
        < GRAPH_MEAN_RECALL_FLOOR
        or float(selected.get("p10_recall_at_49", -1.0))
        < GRAPH_P10_RECALL_FLOOR
        or graph_edges <= 0
    ):
        raise Round0186Error("R0186 graph is not qualified for dose planning")
    updates = successful_updates_for_edges(graph_edges)
    dose = positive_draws_per_edge(
        successful_updates=updates, graph_edges=graph_edges
    )
    lower_dose = positive_draws_per_edge(
        successful_updates=updates - 1, graph_edges=graph_edges
    )
    target_dose = positive_draws_per_edge(
        successful_updates=BASELINE_SUCCESSFUL_UPDATES,
        graph_edges=BASELINE_GRAPH_EDGES,
    )
    expected_train_seconds = updates / REFERENCE_UPDATES_PER_SECOND
    receipt = prompt_contract.seal({
        "schema": DOSE_PLAN_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capabilities": [CAPABILITY],
        "graph_manifest": manifest_signature,
        "graph": graph_signature,
        "retained_rows": ROWS,
        "directed_graph_edges": graph_edges,
        "dose_rule": (
            "ceil(500000 * directed_graph_edges / 148801612) successful updates"
        ),
        "baseline_round": "0115",
        "baseline_graph_edges": BASELINE_GRAPH_EDGES,
        "baseline_successful_updates": BASELINE_SUCCESSFUL_UPDATES,
        "successful_positive_lr_updates": updates,
        "target_positive_draws_per_edge": target_dose,
        "achieved_positive_draws_per_edge": dose,
        "previous_update_positive_draws_per_edge": lower_dose,
        "first_whole_update_at_or_above_target": (
            lower_dose < target_dose <= dose
        ),
        "runtime_projection": {
            "reference_round": REFERENCE_THROUGHPUT_ROUND,
            "reference_updates_per_second": REFERENCE_UPDATES_PER_SECOND,
            "expected_train_seconds": expected_train_seconds,
            "evaluation_allowance_seconds": EVALUATION_ALLOWANCE_SECONDS,
            "expected_train_plus_evaluation_gpu_hours": (
                expected_train_seconds + EVALUATION_ALLOWANCE_SECONDS
            ) / 3600.0,
            "fits_single_eight_gpu_hour_queue_at_reference_rate": (
                expected_train_seconds + EVALUATION_ALLOWANCE_SECONDS <= 8 * 3600
            ),
            "role": (
                "planning diagnostic only; the training round must bind its own "
                "measured throughput floor, p90, and queue segmentation"
            ),
        },
        "training_performed": False,
        "wall_seconds": time.monotonic() - started,
    })
    if receipt["first_whole_update_at_or_above_target"] is not True:
        raise Round0186Error("R0186 dose ceiling arithmetic failed")
    atomic_write_new_json(os.path.join(output, "dose-plan.json"), receipt, immutable=True)


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    action = str(job.get("action") or "")
    if action == "build_graph_and_reference":
        return run_build_graph(active, job)
    if action == "derive_dose_plan":
        return run_derive_dose(active, job)
    raise Round0186Error(f"R0186 does not authorize action {action!r}")


__all__ = ["run_job"]
