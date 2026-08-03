"""Execute the corrected R0172 prompted-universality panel."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from basemap import round0167_prompted_universality as contract_base
from basemap.round0172_prompted_universality import (
    CAPABILITY,
    PROMPTED_MAP_ORDER,
    ROUND_ID,
    Round0172Error,
)
from experiments import round0167_nodes as base


def _configure() -> None:
    contract_bindings = {
        "ROUND_ID": ROUND_ID,
        "CAPABILITY": CAPABILITY,
        "PROMPTED_MAP_ORDER": PROMPTED_MAP_ORDER,
        "Round0167Error": Round0172Error,
    }
    for name, value in contract_bindings.items():
        setattr(contract_base, name, value)
    node_bindings = {
        **contract_bindings,
        "CANARY_SCHEMA": "round0172-prompt-model-canary-v1",
        "PROBE_SCHEMA": "round0172-prompted-probe-embeddings-v1",
        "CONTROL_SCHEMA": "round0172-prompted-fineweb-control-v1",
        "MAP_PANEL_SCHEMA": "round0172-prompted-universality-map-panel-v1",
    }
    for name, value in node_bindings.items():
        setattr(base, name, value)


def run_job(active: Mapping[str, Any], job: Mapping[str, Any] | None = None) -> None:
    _configure()
    base.run_job(dict(active), dict(job) if job is not None else None)
