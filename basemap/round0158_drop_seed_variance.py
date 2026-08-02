"""Drop-only historical-row seed-variance evidence for Round 0158."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import canonical_json, sha256_bytes
from basemap.round0140_subsystem_bisection import METRICS, metric_view
from basemap.round0149_drop_only import (
    TREATMENT,
    treatment_train_config as parent_train_config,
)


ROUND_ID = "0158"
CAPABILITY = "jina-2m-drop-only-seed44-45-calibration-v1"
SEEDS = (44, 45)


class Round0158Error(RuntimeError):
    """Raised when the registered drop-only seed replay changes."""


def drop_seed_train_config(
    *,
    seed: int,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    graph_edges: int,
    source_sha256: str,
    selection_sha256: str,
) -> tuple[dict[str, Any], str]:
    if seed not in SEEDS:
        raise Round0158Error("drop-only replay seed changed")
    parent, _digest = parent_train_config(
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        graph_edges=graph_edges,
        source_sha256=source_sha256,
        selection_sha256=selection_sha256,
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
        raise Round0158Error("R0149 parent seed contract changed")
    paired["seed"] = seed
    optimizer["seed"] = seed
    causal["replication_seed"] = seed
    causal["graph_reused_byte_exact"] = True
    causal["only_varying_factor_from_r0149_drop_seed42"] = "model-optimizer-seed"
    value["schema"] = f"round0158-drop-only-historical-seed{seed}-train-v1"
    return value, sha256_bytes(canonical_json(value))


def build_seed_evidence(
    panels: Mapping[int, Mapping[str, Any]],
    density_cells: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(panels) != set(SEEDS) or set(density_cells) != {
        f"seed{seed}" for seed in SEEDS
    }:
        raise Round0158Error("drop-only seed evidence matrix is incomplete")
    cells: dict[str, Any] = {}
    for seed in SEEDS:
        panel = panels[seed]
        cell = panel.get("cells", {}).get(TREATMENT)
        density = density_cells[f"seed{seed}"]
        if (
            panel.get("round_id") != ROUND_ID
            or not isinstance(cell, Mapping)
            or cell.get("seed") != seed
            or not isinstance(density, Mapping)
            or density.get("seed") != seed
        ):
            raise Round0158Error(f"drop-only seed {seed} evidence changed")
        metrics = metric_view(cell)
        if set(metrics) != set(METRICS):
            raise Round0158Error(f"drop-only seed {seed} functional metrics changed")
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
        "schema": "round0158-drop-only-seed-calibration-evidence-v1",
        "round_id": ROUND_ID,
        "capability": CAPABILITY,
        "cell": TREATMENT,
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

