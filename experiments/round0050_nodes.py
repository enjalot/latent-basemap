"""Round 0050 wrapper for the reviewed R0049 balanced-60M graph builder."""
from __future__ import annotations

from typing import Any

from basemap.round0049_program import Round0049Error
from experiments.round0049_nodes import run_build_graph


ROUND_ID = "0050"


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0049Error("R0050 handler received another queue")
    selected = job if job is not None else active.get("job") or {}
    if selected.get("action") != "build_graph":
        raise Round0049Error("R0050 accepts only the graph-build action")
    return run_build_graph(active, selected)

