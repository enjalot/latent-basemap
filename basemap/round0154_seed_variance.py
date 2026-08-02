"""Raw historical-row seed-variance evidence for Round 0154."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import canonical_json, sha256_bytes
from basemap.round0140_subsystem_bisection import (
    CURRENT_GRAPH_CURRENT_HOST,
    METRICS,
    host_train_config,
    metric_view,
)


ROUND_ID = "0154"
CAPABILITY = "jina-2m-raw-seed44-45-calibration-v1"
RAW = CURRENT_GRAPH_CURRENT_HOST
SEEDS = (44, 45)


class Round0154Error(RuntimeError):
    """Raised when the registered R0154 seed replay changes."""


def raw_seed_train_config(
    *,
    seed: int,
    cell: str = RAW,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    graph_edges: int,
) -> tuple[dict[str, Any], str]:
    if seed not in SEEDS or cell != RAW:
        raise Round0154Error("raw replay seed or cell changed")
    parent, _digest = host_train_config(
        cell=cell,
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        graph_edges=graph_edges,
    )
    value = copy.deepcopy(parent)
    paired = value.get("paired_invariant")
    optimizer = value.get("optimizer")
    causal = value.get("causal_matrix")
    if (
        not isinstance(paired, dict)
        or not isinstance(optimizer, dict)
        or not isinstance(causal, dict)
        or paired.get("seed") != 42
        or optimizer.get("seed") != 42
    ):
        raise Round0154Error("R0140 parent seed contract changed")
    paired["seed"] = seed
    optimizer["seed"] = seed
    causal["replication_seed"] = seed
    causal["graph_reused_byte_exact"] = True
    causal["only_varying_factor_from_r0140_raw_seed42"] = "model-optimizer-seed"
    value["schema"] = f"round0154-raw-historical-seed{seed}-train-v1"
    return value, sha256_bytes(canonical_json(value))


def build_seed_evidence(
    panels: Mapping[int, Mapping[str, Any]],
    density_cells: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(panels) != set(SEEDS) or set(density_cells) != {
        f"seed{seed}" for seed in SEEDS
    }:
        raise Round0154Error("seed evidence matrix is incomplete")
    cells: dict[str, Any] = {}
    for seed in SEEDS:
        panel = panels[seed]
        cell = panel.get("cells", {}).get(RAW)
        density = density_cells[f"seed{seed}"]
        if (
            panel.get("round_id") != ROUND_ID
            or not isinstance(cell, Mapping)
            or cell.get("seed") != seed
            or not isinstance(density, Mapping)
            or density.get("seed") != seed
        ):
            raise Round0154Error(f"seed {seed} evidence identity changed")
        metrics = metric_view(cell)
        if set(metrics) != set(METRICS):
            raise Round0154Error(f"seed {seed} functional metrics changed")
        cells[f"seed{seed}"] = {
            "seed": seed,
            "functional_metrics": {
                key: float(metrics[key]) for key in METRICS
            },
            "density_v2": dict(density["density_v2"]),
            "clears_registered_density_floor": bool(
                density["clears_registered_floor"]
            ),
            "legacy_panel_density_not_density_v2": float(
                cell["panel"]["density"]
            ),
        }
    return {
        "schema": "round0154-raw-seed-calibration-evidence-v1",
        "round_id": ROUND_ID,
        "capability": CAPABILITY,
        "cell": RAW,
        "seeds": list(SEEDS),
        "cells": cells,
        "seed45_minus_seed44": {
            metric: cells["seed45"]["functional_metrics"][metric]
            - cells["seed44"]["functional_metrics"][metric]
            for metric in METRICS
        } | {
            "density_v2": (
                float(cells["seed45"]["density_v2"]["correlation"])
                - float(cells["seed44"]["density_v2"]["correlation"])
            )
        },
        "margin_or_floor_proposed": False,
        "floor_changed": False,
        "training_performed": True,
    }
