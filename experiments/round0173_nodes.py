"""Execute the R0173 prompted OOD probe-pack staging queue."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from basemap.round0173_prompted_ood_pack import (
    CANARY_SCHEMA,
    CAPABILITY,
    LANGUAGE_PROBE_SCHEMA,
    OOD_AUDIT_SCHEMA,
    ROUND_ID,
    Round0173Error,
)
from experiments import round0169_nodes as base


def _configure() -> None:
    bindings = {
        "ROUND_ID": ROUND_ID,
        "Round0169Error": Round0173Error,
        "CANARY_SCHEMA": CANARY_SCHEMA,
        "LANGUAGE_PROBE_SCHEMA": LANGUAGE_PROBE_SCHEMA,
        "LANGUAGE_RECEIPT_ROUND_ID": ROUND_ID,
        "OOD_AUDIT_SCHEMA": OOD_AUDIT_SCHEMA,
        "OOD_PACK_CAPABILITY": CAPABILITY,
    }
    for name, value in bindings.items():
        setattr(base, name, value)


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    _configure()
    base.run_job(active, job)
