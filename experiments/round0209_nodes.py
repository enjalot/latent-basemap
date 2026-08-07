"""Execute the R0209 prompted-diverse U12 graph stage.

The graph kernel is the accepted R0169 diverse implementation; this module only
rebinds the round identity and the graph receipt schema so the artifact is owned
by R0209.  No threshold, seed, shard law, merge rule, or guard is changed.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from basemap.round0209_prompted_diverse_graph import (
    CAPABILITY,
    GRAPH_SCHEMA,
    ROUND_ID,
    Round0209Error,
)
from experiments import round0169_nodes as diverse


def _configure() -> None:
    bindings = {
        "ROUND_ID": ROUND_ID,
        "CAPABILITY": CAPABILITY,
        "GRAPH_SCHEMA": GRAPH_SCHEMA,
        "Round0169Error": Round0209Error,
    }
    for name, value in bindings.items():
        setattr(diverse, name, value)


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if str(job.get("action") or "") != "build_graph_and_reference":
        raise Round0209Error("R0209 authorizes only the diverse graph stage")
    _configure()
    diverse.run_build_graph(active, job)


__all__ = ["run_job"]
