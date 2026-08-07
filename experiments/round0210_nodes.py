"""Execute the R0210 low-dose h2048 prompted-diverse U12 train stage.

The train kernel is the accepted R0169 diverse implementation, which reads its
horizon from a module-level ``SUCCESSFUL_UPDATES``.  R0210 reads the sealed
R0209 graph receipt first, derives the registered low-dose horizon from its
actual directed-edge count, and binds that number into both the kernel and the
train config so the two can never disagree.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import expected_input_signature
from basemap.round0209_prompted_diverse_graph import (
    CAPABILITY as GRAPH_CAPABILITY,
    GRAPH_SCHEMA,
    plausible_directed_edges,
)
from basemap.round0210_prompted_diverse_low_dose import (
    CAPABILITY,
    PRODUCTION_CONFIG_SCHEMA,
    ROUND_ID,
    ROWS,
    Round0210Error,
    TRAIN_SCHEMA,
    low_dose_train_config,
    successful_updates_for_edges,
)
from basemap import round0113_prompt_contrast as prompt_contract
from experiments import round0169_nodes as diverse


def _sealed_graph_edges(job: Mapping[str, Any]) -> int:
    """Read the active directed-edge count from the sealed R0209 receipt."""
    manifest_path = str(job["graph_manifest"])
    signature = expected_input_signature(manifest_path)
    declared = job.get("graph_manifest_signature")
    if declared is not None and dict(declared) != signature:
        raise Round0210Error("R0210 sealed R0209 graph manifest bytes changed")
    manifest = prompt_contract.read_sealed(
        manifest_path, label="sealed R0209 prompted-diverse graph"
    )
    if (
        manifest.get("schema") != GRAPH_SCHEMA
        or manifest.get("round_id") != "0209"
        or int(manifest.get("retained_rows", -1)) != ROWS
        or manifest.get("training_performed") is not False
    ):
        raise Round0210Error("R0210 sealed R0209 graph contract changed")
    qualification = (manifest.get("search_qualification") or {})
    selected = str(qualification.get("selected_nprobe") or "")
    cell = (qualification.get("cells") or {}).get(selected) or {}
    if cell.get("passed") is not True:
        raise Round0210Error("R0210 requires a qualified R0209 graph")
    edges = int(manifest.get("directed_edge_count", 0))
    if edges <= 0:
        raise Round0210Error("R0210 sealed R0209 graph has no directed edges")
    if not plausible_directed_edges(edges):
        raise Round0210Error(
            "R0210 sealed R0209 directed-edge count is outside the registered "
            "plausibility band; the dose horizon would follow input drift"
        )
    return edges


def _configure(updates: int) -> None:
    bindings = {
        "ROUND_ID": ROUND_ID,
        "CAPABILITY": CAPABILITY,
        "TRAIN_SCHEMA": TRAIN_SCHEMA,
        "PRODUCTION_CONFIG_SCHEMA": PRODUCTION_CONFIG_SCHEMA,
        "SUCCESSFUL_UPDATES": updates,
        "diverse_train_config": low_dose_train_config,
        "Round0169Error": Round0210Error,
    }
    for name, value in bindings.items():
        setattr(diverse, name, value)


def run_train(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0210Error("R0210 train handler received another queue")
    edges = _sealed_graph_edges(job)
    updates = successful_updates_for_edges(edges)
    declared = job.get("registered_dose_bound")
    if declared is not None and updates > int(declared):
        raise Round0210Error(
            "R0210 derived update horizon exceeds the registered round bound"
        )
    _configure(updates)
    diverse.run_train(active, job)


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if str(job.get("action") or "") != "train_prompted_diverse_u12_low_dose":
        raise Round0210Error("R0210 authorizes only the low-dose diverse train")
    run_train(active, job)


__all__ = ["run_job", "run_train"]
