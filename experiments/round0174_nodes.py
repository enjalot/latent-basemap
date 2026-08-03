"""Execute the R0174 fixed-row fuzzy-k15 forensic."""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_write_new_json, create_fresh_directory
from basemap.round0108_evaluation import seal, validate_seal
from basemap.round0140_subsystem_bisection import metric_view
from basemap.round0174_k15_forensic import (
    CAPABILITY,
    CELL,
    GRAPH_K,
    ROUND_ID,
    Round0174Error,
    build_decision,
    host_train_config,
)
from experiments import round0140_nodes as base


K50_CONTROL = "current_graph_current_host"


def _configure() -> None:
    """Bind the reviewed R0140 machinery to the one registered R0174 cell."""
    base.ROUND_ID = ROUND_ID
    base.CAPABILITY = CAPABILITY
    base.Round0140Error = Round0174Error
    base.GRAPH_K = GRAPH_K
    base.CURRENT_GRAPH_CURRENT_HOST = CELL
    base.HISTORICAL_GRAPH_CURRENT_HOST = "unused_historical_graph_host_cell"
    base.HISTORICAL_GRAPH_DEVICE_REPRO = "unused_historical_graph_device_cell"
    base.NEW_CELLS = (CELL,)
    base.ARTIFACT_SCHEMA_PREFIX = "round0174"
    base.host_train_config = host_train_config
    base.metric_view = metric_view


def _read_panel(path: str) -> dict[str, Any]:
    signature = expected_input_signature(path)
    with open(signature["canonical_path"], encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0174Error("R0174 functional panel is not an object")
    validate_seal(value, label="R0174 functional panel")
    if value.get("round_id") != ROUND_ID:
        raise Round0174Error("R0174 functional panel identity changed")
    return value


def run_decision(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0174 k15 forensic decision"
    )
    panel_path = os.path.join(
        str(job["panel_output"]), "functional-bisection.json"
    )
    panel = _read_panel(panel_path)
    cells = panel.get("cells")
    if not isinstance(cells, Mapping) or CELL not in cells:
        raise Round0174Error("R0174 treatment cell is absent")
    control_signature = expected_input_signature(str(job["r0140_panel"]))
    with open(control_signature["canonical_path"], encoding="utf-8") as handle:
        control_panel = json.load(handle)
    if not isinstance(control_panel, dict):
        raise Round0174Error("R0140 k50 control panel is not an object")
    validate_seal(control_panel, label="accepted R0140 k50 control panel")
    control_cells = control_panel.get("cells")
    if (
        control_panel.get("round_id") != "0140"
        or not isinstance(control_cells, Mapping)
        or K50_CONTROL not in control_cells
    ):
        raise Round0174Error("R0140 k50 control cell changed")
    control = control_cells[K50_CONTROL]
    decision = build_decision(
        treatment=cells[CELL],
        k50_control=control,
    )
    treatment_density = float(cells[CELL]["panel"]["density"])
    control_density = float(control["panel"]["density"])
    receipt = seal({
        **decision,
        "release_sha": active["manifest"]["release_sha"],
        "capability": CAPABILITY,
        "panel": expected_input_signature(panel_path),
        "r0140_k50_control_panel": control_signature,
        "paired_density_diagnostic": {
            "k15": treatment_density,
            "r0140_k50": control_density,
            "delta_k15_minus_k50": treatment_density - control_density,
            "selector_input": False,
        },
        "graph_degree_treatment": {
            "control_k": 50,
            "treatment_k": GRAPH_K,
            "same_rows_trainer_seed_and_update_horizon": True,
        },
        "capabilities": [CAPABILITY],
    })
    atomic_write_new_json(
        os.path.join(output, "decision.json"), receipt, immutable=True
    )


def run_job(
    active: dict[str, Any], job: dict[str, Any] | None = None
) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID or job is None:
        raise Round0174Error("R0174 handler requires its exact round/job")
    _configure()
    action = str(job.get("action") or "")
    if action == "build_current_graph":
        return base.run_build_current_graph(active, job)
    if action == "train_host":
        return base.run_host_train(active, job)
    if action == "functional_panel":
        return base.run_panel(active, job)
    if action == "decide":
        return run_decision(active, job)
    raise Round0174Error(f"unknown R0174 action: {action!r}")
