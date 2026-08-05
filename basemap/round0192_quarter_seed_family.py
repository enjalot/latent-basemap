"""Frozen contract for the R0192 mixed-quarter seed family completion."""
from __future__ import annotations

import copy
import math
import statistics
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import canonical_json, sha256_bytes
from basemap.round0187_composition_nested_ladder import (
    PRIMARY_METRICS,
    RUNG_ROWS,
    successful_updates_for_edges,
    train_config as seed42_train_config,
)


ROUND_ID = "0192"
RUNG = "quarter"
SEEDS = (43, 44)
ROWS = RUNG_ROWS[RUNG]
CAPABILITY = "jina-document-english-mixed-quarter-three-seed-family-v1"
EVALUATION_SCHEMA = "round0192-mixed-quarter-common-core-evaluation-v1"
SYNTHESIS_SCHEMA = "round0192-mixed-quarter-three-seed-family-v1"
TRAIN_SCHEMA_PREFIX = "round0192-mixed-quarter-train-receipt"
GATE_METRICS = (
    "density_v2",
    "ffr",
    "purity_fidelity_k256",
    "purity_fidelity_k1024",
    "projection_ffr",
    "heldout_recall_at_10",
)


class Round0192Error(RuntimeError):
    """The R0192 seed-only treatment or family contract changed."""


def train_schema(seed: int) -> str:
    if seed not in SEEDS:
        raise Round0192Error("unknown R0192 seed")
    return f"{TRAIN_SCHEMA_PREFIX}-seed{seed}-v1"


def train_config(
    *,
    seed: int,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    graph_edges: int,
    retained_rows: int,
) -> tuple[dict[str, Any], str]:
    """Clone the accepted R0187 quarter treatment and change only seed."""
    if seed not in SEEDS or retained_rows != ROWS:
        raise Round0192Error("R0192 seed or quarter cardinality changed")
    config, _ = seed42_train_config(
        rung=RUNG,
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        graph_edges=graph_edges,
        retained_rows=retained_rows,
    )
    config = copy.deepcopy(config)
    config["schema"] = f"round0192-quarter-seed{seed}-train-config-v1"
    config["paired_invariant"]["seed"] = seed
    config["paired_invariant"]["only_model_treatment_relative_to_r0187"] = (
        f"seed 42 -> {seed}"
    )
    optimizer = config["optimizer"]
    optimizer["seed"] = seed
    optimizer["positive_rng_seed"] = seed
    optimizer["negative_rng_seed"] = 11_300_000 + seed
    stamp = config["execution"]["expected_pipeline_stamp"]
    stamp["positive_rng_seed"] = seed
    stamp["negative_rng_seed"] = 11_300_000 + seed
    config["execution"]["scale_change"] = (
        f"none relative to R0187 quarter; seed {seed} is the sole model "
        "treatment and population, graph, dose, and recipe remain frozen"
    )
    return config, sha256_bytes(canonical_json(config))


def gate_metric_view(evaluation: Mapping[str, Any]) -> dict[str, float]:
    primary = evaluation.get("primary_metrics") or {}
    diagnostic = evaluation.get("diagnostic_metrics") or {}
    pile_ood = evaluation.get("pile_ood") or {}
    values = {
        "density_v2": diagnostic.get("mixed_density"),
        "ffr": primary.get("mixed_ffr"),
        "purity_fidelity_k256": primary.get("mixed_purity_fidelity_k256"),
        "purity_fidelity_k1024": primary.get("mixed_purity_fidelity_k1024"),
        "projection_ffr": diagnostic.get("mixed_projection_ffr"),
        "heldout_recall_at_10": primary.get("pile_ood_recall_at_10"),
    }
    output = {key: float(value) for key, value in values.items()}
    if set(output) != set(GATE_METRICS) or any(
        not math.isfinite(value) or value <= 0 for value in output.values()
    ):
        raise Round0192Error("quarter gate metric vector changed")
    if not math.isclose(
        output["projection_ffr"], float(pile_ood.get("ffr", float("nan"))),
        rel_tol=0,
        abs_tol=1e-15,
    ):
        raise Round0192Error("quarter projection FFR binding changed")
    return output


def seed_family(
    *, evaluations: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    """Bind seed 42/43/44 cells without registering gates in R0192."""
    if set(evaluations) != {42, 43, 44}:
        raise Round0192Error("quarter seed family changed")
    primary_cells: dict[str, dict[str, float]] = {}
    gate_cells: dict[str, dict[str, float]] = {}
    for seed in (42, 43, 44):
        evaluation = evaluations[seed]
        primary = {
            key: float(value)
            for key, value in (evaluation.get("primary_metrics") or {}).items()
        }
        if set(primary) != set(PRIMARY_METRICS) or any(
            not math.isfinite(value) or value <= 0 for value in primary.values()
        ):
            raise Round0192Error(f"seed {seed} primary metric vector changed")
        primary_cells[str(seed)] = primary
        gate_cells[str(seed)] = gate_metric_view(evaluation)
    summaries = {}
    for metric in GATE_METRICS:
        values = [gate_cells[str(seed)][metric] for seed in (42, 43, 44)]
        summaries[metric] = {
            "values_seed42_seed43_seed44": values,
            "mean": statistics.fmean(values),
            "sample_sd_ddof1": statistics.stdev(values),
        }
    return {
        "outcome": "mixed-quarter-three-seed-family-complete",
        "seeds": [42, 43, 44],
        "rows": ROWS,
        "primary_metric_cells": primary_cells,
        "gate_metric_cells": gate_cells,
        "descriptive_summaries": summaries,
        "gate_registration_deferred_to_reviewed_cpu_round": True,
    }


__all__ = [
    "CAPABILITY",
    "EVALUATION_SCHEMA",
    "GATE_METRICS",
    "ROUND_ID",
    "ROWS",
    "RUNG",
    "SEEDS",
    "SYNTHESIS_SCHEMA",
    "Round0192Error",
    "gate_metric_view",
    "seed_family",
    "successful_updates_for_edges",
    "train_config",
    "train_schema",
]
