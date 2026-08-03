"""Bind the R0179 baseline machinery to R0181 fixed normalization."""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from basemap.round0181_fixed_normalization import (
    CAPABILITY,
    ROUND_ID,
    Round0181Error,
    build_synthesis,
    validate_execution,
)
from experiments import round0179_nodes as base


REFERENCE_SCRIPT = os.path.join(
    os.path.dirname(__file__), "round0181_numap_reference.py"
)


def _configure() -> None:
    bindings = {
        "ROUND_ID": ROUND_ID,
        "CAPABILITY": CAPABILITY,
        "Round0179Error": Round0181Error,
        "REFERENCE_SCRIPT": REFERENCE_SCRIPT,
        "CELL_SCHEMA": "round0181-numap-fixed-normalization-cell-v1",
        "validate_execution": validate_execution,
        "build_synthesis": build_synthesis,
    }
    for name, value in bindings.items():
        setattr(base, name, value)


def run_job(active: Mapping[str, Any], job: Mapping[str, Any] | None = None) -> None:
    _configure()
    base.run_job(dict(active), None if job is None else dict(job))


__all__ = ["run_job"]
