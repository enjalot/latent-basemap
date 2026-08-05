"""Frozen contract for the R0189 seed-44 half-to-full replay."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import canonical_json, sha256_bytes
from basemap.round0187_composition_nested_ladder import (
    PRIMARY_METRICS,
    RETENTION_RATIO,
)
from basemap import round0188_composition_boundary_seed43 as r0188


ROUND_ID = "0189"
SEED = 44
RUNGS = ("half", "full")
CAPABILITY = (
    "jina-document-english-composition-controlled-half-full-seed44-replay-v1"
)
EVALUATION_SCHEMA = "round0189-composition-boundary-evaluation-v1"
SYNTHESIS_SCHEMA = "round0189-composition-boundary-seed44-decision-v1"
TRAIN_SCHEMA_PREFIX = "round0189-composition-boundary-seed44-train-receipt"


class Round0189Error(RuntimeError):
    """The preregistered R0189 contract changed or failed authentication."""


def successful_updates_for_edges(edge_count: int) -> int:
    try:
        return r0188.successful_updates_for_edges(edge_count)
    except r0188.Round0188Error as error:
        raise Round0189Error(str(error)) from error


def train_schema(rung: str) -> str:
    if rung not in RUNGS:
        raise Round0189Error("unknown boundary rung")
    return f"{TRAIN_SCHEMA_PREFIX}-{rung}-v1"


def train_config(
    *,
    rung: str,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    graph_edges: int,
    retained_rows: int,
) -> tuple[dict[str, Any], str]:
    """Clone the authenticated R0188 treatment and change only the model seed."""
    try:
        config, _ = r0188.train_config(
            rung=rung,
            graph_signature=graph_signature,
            graph_manifest_signature=graph_manifest_signature,
            graph_edges=graph_edges,
            retained_rows=retained_rows,
        )
    except r0188.Round0188Error as error:
        raise Round0189Error(str(error)) from error
    config = copy.deepcopy(config)
    config["schema"] = f"round0189-{rung}-seed44-train-config-v1"
    invariant = config["paired_invariant"]
    invariant["seed"] = SEED
    invariant["only_model_treatment_relative_to_r0187"] = "seed 42 -> 44"
    optimizer = config["optimizer"]
    optimizer["seed"] = SEED
    optimizer["positive_rng_seed"] = SEED
    optimizer["negative_rng_seed"] = 11_300_044
    stamp = config["execution"]["expected_pipeline_stamp"]
    stamp["positive_rng_seed"] = SEED
    stamp["negative_rng_seed"] = 11_300_044
    config["execution"]["scale_change"] = (
        "none relative to the matching R0187 rung; seed 44 is the sole model "
        "treatment and graph/population/dose/recipe remain frozen"
    )
    return config, sha256_bytes(canonical_json(config))


def train_checks_close(value: Any) -> bool:
    return r0188.train_checks_close(value)


def boundary_decision(
    *,
    seed42: Mapping[str, Mapping[str, float]],
    seed44: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Report the preregistered seed-44 Pile-FFR boundary replay."""
    normalized: dict[str, dict[str, dict[str, float]]] = {}
    for seed, cells in (("seed42", seed42), ("seed44", seed44)):
        if set(cells) != set(RUNGS):
            raise Round0189Error(f"{seed} boundary rung set changed")
        normalized[seed] = {
            rung: {key: float(value) for key, value in cells[rung].items()}
            for rung in RUNGS
        }
        if any(
            set(metrics) != set(PRIMARY_METRICS)
            for metrics in normalized[seed].values()
        ):
            raise Round0189Error(f"{seed} primary metric set changed")
    values = [
        value
        for cells in normalized.values()
        for metrics in cells.values()
        for value in metrics.values()
    ]
    if not np.isfinite(values).all() or any(value <= 0 for value in values):
        raise Round0189Error("boundary metrics must be finite and positive")

    retention = {
        seed: {
            metric: cells["full"][metric] / cells["half"][metric]
            for metric in PRIMARY_METRICS
        }
        for seed, cells in normalized.items()
    }
    seed42_trigger = retention["seed42"]["pile_ffr"] < RETENTION_RATIO
    seed44_trigger = retention["seed44"]["pile_ffr"] < RETENTION_RATIO
    if not seed42_trigger:
        raise Round0189Error("accepted R0187 Pile-FFR trigger changed")
    outcome = (
        "composition-controlled-size-regression-seed44-positive"
        if seed44_trigger
        else "composition-controlled-size-regression-seed44-negative"
    )
    return {
        "outcome": outcome,
        "registered_metric": "pile_ffr",
        "registered_boundary": "half_to_full",
        "retention_floor": RETENTION_RATIO,
        "cells": normalized,
        "retention": retention,
        "seed42_trigger_reproduced_from_r0187": seed42_trigger,
        "seed44_confirms_registered_regression": seed44_trigger,
        "other_seed44_metric_misses_diagnostic": [
            metric
            for metric in PRIMARY_METRICS
            if metric != "pile_ffr" and retention["seed44"][metric] < RETENTION_RATIO
        ],
        "follow_up": (
            "combine with the independently reviewed R0188 seed-43 replay before "
            "making the aggregate multi-seed capacity decision"
        ),
    }


__all__ = [
    "CAPABILITY",
    "EVALUATION_SCHEMA",
    "ROUND_ID",
    "SEED",
    "SYNTHESIS_SCHEMA",
    "Round0189Error",
    "boundary_decision",
    "successful_updates_for_edges",
    "train_checks_close",
    "train_config",
    "train_schema",
]
