"""Execute the R0212 seed-43 replay of the prompted-diverse U12 rung.

Identical to R0210's node except the round identity, the receipt schemas, and the
seed that reaches the train config. The horizon is still derived from the sealed
R0209 receipt, so both seed cells train against the same edge count and the same
dose by construction.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from basemap.round0209_prompted_diverse_graph import GRAPH_SCHEMA
from basemap.round0210_prompted_diverse_low_dose import successful_updates_for_edges
from basemap.round0212_prompted_diverse_seed43 import (
    CAPABILITY,
    PRODUCTION_CONFIG_SCHEMA,
    ROUND_ID,
    Round0212Error,
    SEED,
    TRAIN_SCHEMA,
    seed43_train_config,
)
from experiments import round0166_nodes as q2
from experiments import round0169_nodes as diverse
from experiments.round0210_nodes import _sealed_graph_edges


def _configure(updates: int) -> None:
    bindings = {
        "ROUND_ID": ROUND_ID,
        "CAPABILITY": CAPABILITY,
        "TRAIN_SCHEMA": TRAIN_SCHEMA,
        "PRODUCTION_CONFIG_SCHEMA": PRODUCTION_CONFIG_SCHEMA,
        "SUCCESSFUL_UPDATES": updates,
        "SEED": SEED,
        "GRAPH_SCHEMA": GRAPH_SCHEMA,
        "diverse_train_config": seed43_train_config,
        "Round0169Error": Round0212Error,
    }
    for name, value in bindings.items():
        setattr(diverse, name, value)
    diverse._configure_q2_kernel()
    for name, value in {
        "GRAPH_SOURCE_ROUND_ID": "0209",
        "GRAPH_BUILT_IN_ROUND": False,
    }.items():
        setattr(q2, name, value)
    if int(q2.SEED) != SEED:
        raise Round0212Error("R0212 seed did not reach the train kernel")


def run_train(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0212Error("R0212 train handler received another queue")
    edges = _sealed_graph_edges(job)
    updates = successful_updates_for_edges(edges)
    declared = job.get("registered_dose_bound")
    if declared is not None and updates > int(declared):
        raise Round0212Error(
            "R0212 derived update horizon exceeds the registered round bound"
        )
    _configure(updates)
    q2.run_train(active, job)


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if str(job.get("action") or "") != "train_prompted_diverse_u12_seed43":
        raise Round0212Error("R0212 authorizes only the seed-43 diverse train")
    run_train(active, job)


__all__ = ["run_job", "run_train"]
