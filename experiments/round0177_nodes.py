"""Execute the duplicate-aware R0177 prompted-universality panel."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from basemap import round0167_prompted_universality as contract_base
from basemap.round0177_prompted_universality import (
    CAPABILITY,
    PROMPTED_MAP_ORDER,
    ROUND_ID,
    Round0177Error,
    exact_training_overlap_report,
)
from experiments import round0167_nodes as base
from experiments import round0176_nodes as audit_base


def _configure() -> None:
    contract_bindings = {
        "ROUND_ID": ROUND_ID,
        "CAPABILITY": CAPABILITY,
        "PROMPTED_MAP_ORDER": PROMPTED_MAP_ORDER,
        "Round0167Error": Round0177Error,
    }
    for name, value in contract_bindings.items():
        setattr(contract_base, name, value)
    node_bindings = {
        **contract_bindings,
        "CANARY_SCHEMA": "round0177-prompt-model-canary-v1",
        "PROBE_SCHEMA": "round0177-prompted-probe-embeddings-v1",
        "CONTROL_SCHEMA": "round0177-prompted-fineweb-control-v1",
        "MAP_PANEL_SCHEMA": (
            "round0177-prompted-universality-map-panel-v1"
        ),
    }
    for name, value in node_bindings.items():
        setattr(base, name, value)
    base.ALLOW_CROSS_SPLIT_FAMILIES = True
    base.DUPLICATE_SENSITIVITY = True
    audit_base.ROUND_ID = ROUND_ID
    audit_base.CAPABILITY = CAPABILITY
    audit_base.PROMPTED_MAP_ORDER = PROMPTED_MAP_ORDER
    audit_base.Round0176Error = Round0177Error
    audit_base.exact_training_overlap_report = (
        exact_training_overlap_report
    )


def run_job(
    active: Mapping[str, Any], job: Mapping[str, Any] | None = None
) -> None:
    _configure()
    if job is not None and job.get("action") == "audit_training_disjoint":
        return audit_base.run_training_disjoint_audit(active, job)
    base.run_job(
        dict(active), dict(job) if job is not None else None
    )
