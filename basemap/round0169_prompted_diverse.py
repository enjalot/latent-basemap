"""Frozen scientific contracts for the conditional prompted-diverse Q3 rung."""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import canonical_json, sha256_bytes
from basemap.round0105_search import GROUPS
from basemap.round0108_evaluation import IN_MIX_LANGUAGES
from basemap import round0113_prompt_contrast as r0113
from basemap.round0166_prompted_8m import (
    METRICS,
    NATIVE_ABSOLUTE_METRICS,
    ScalePromptTrainingInput,
)


ROUND_ID = "0169"
CAPABILITY = "jina-document-diverse-u12-prompted-map-seed42-v1"
ROWS = 12_474_331
DIMENSION = 768
SEED = 42
SUCCESSFUL_UPDATES = 500_000
GRAPH_K = 50
GRAPH_NLIST = 8_192
GRAPH_NPROBE = 64
GRAPH_NPROBE_GRID = (16, 32, 64, 128, 256)
GRAPH_TRAIN_ROWS = 262_144
GRAPH_TRAIN_SEED = 113
GRAPH_QUALITY_ROWS = 4_096
GRAPH_QUALITY_SEED = 114
GRAPH_MEAN_RECALL_FLOOR = 0.90
GRAPH_P10_RECALL_FLOOR = 0.80
GRAPH_VECTOR_STORAGE = "gpu-ivfflat-fp32-four-shard-exact-merge"
GRAPH_EXECUTION = "four-row-disjoint-shards-shared-quantizer-global-topk"
RETENTION_RATIO = 0.97
LANGUAGE_TO_POOLED_ENGLISH_RATIO = 0.40
POLISH_TO_IN_MIX_MEDIAN_RATIO = 0.50
HOST_RSS_LIMIT_GIB = 90.0
MULTIPLICITY_POLICY = (
    "exact-r0132-u12-population-unchanged; exact duplicate families are metadata-only"
)


class Round0169Error(RuntimeError):
    """The conditional prompted-diverse scale contract changed."""


class DiversePromptTrainingInput(ScalePromptTrainingInput):
    """Q2's sampler adapter with Q3's exact, non-deduplicated U12 policy."""

    @staticmethod
    def _patch_runtime(runtime: Mapping[str, Any]) -> dict[str, Any]:
        value = dict(runtime)
        value["multiplicity_policy"] = MULTIPLICITY_POLICY
        return value


def diverse_train_config(
    *,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    graph_edges: int,
    retained_rows: int,
    seed: int = SEED,
) -> tuple[dict[str, Any], str]:
    """Clone the accepted prompted recipe with only Q3 population bindings."""
    if retained_rows != ROWS or graph_edges <= 0:
        raise Round0169Error("R0169 train config population is invalid")
    config, _ = r0113.train_config(
        "document",
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        graph_edges=graph_edges,
        retained_rows=r0113.RETAINED_ROWS,
        seed=seed,
    )
    config = copy.deepcopy(config)
    config["schema"] = "round0169-prompted-diverse-u12-train-config-v1"
    config["paired_invariant"] = {
        "rows": retained_rows,
        "dimension": DIMENSION,
        "seed": seed,
        "successful_positive_lr_updates": SUCCESSFUL_UPDATES,
        "dose_rule": "same fixed 500,000 successful-update dose as R0115/R0171",
        "graph_policy": "same fuzzy-k50 law and seeds as R0115/R0171",
        "graph_vector_storage": GRAPH_VECTOR_STORAGE,
        "sampler": r0113.SAMPLER_CLASS,
    }
    config["input"].update({
        "rows": retained_rows,
        "representation": "prompted-document-host-fp16-npy",
        "multiplicity_policy": MULTIPLICITY_POLICY,
    })
    expected = config["execution"]["expected_pipeline_stamp"]
    expected["negative_sampling"] = f"uniform-{retained_rows}-compact-representatives-nonself"
    expected["compact_retained_rows"] = retained_rows
    expected["multiplicity_policy"] = MULTIPLICITY_POLICY
    config["optimizer"]["successful_positive_lr_updates"] = SUCCESSFUL_UPDATES
    config["execution"]["scale_change"] = (
        "Q2-to-Q3 changes the exact population to reviewed diverse R0132 U12; "
        "recipe, k50 graph law, seed, and fixed dose remain unchanged"
    )
    config["execution"]["graph_vector_storage"] = GRAPH_VECTOR_STORAGE
    config["execution"]["graph_execution"] = GRAPH_EXECUTION
    return config, sha256_bytes(canonical_json(config))


def _at_least(observed: float, threshold: float) -> bool:
    """Inclusive decimal boundary without binary-float equality failures."""
    return bool(
        observed >= threshold
        or math.isclose(observed, threshold, rel_tol=1e-12, abs_tol=1e-12)
    )


def _finite_table(value: Mapping[str, Any], expected: tuple[str, ...], *, label: str) -> dict[str, float]:
    if set(value) != set(expected):
        raise Round0169Error(f"{label} metric set changed")
    output = {key: float(value[key]) for key in expected}
    if not all(math.isfinite(item) for item in output.values()):
        raise Round0169Error(f"{label} contains nonfinite metrics")
    return output


def prompted_diverse_decision(
    *,
    native: Mapping[str, Any],
    matched_2m: Mapping[str, Any],
    baseline_2m_seed42: Mapping[str, Any],
    prompted_floors: Mapping[str, Any],
    group_ffr: Mapping[str, Any],
    prompted_ood: Mapping[str, Any],
    raw_r0132_ood: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply every Q2-derived, language-relative, and Polish OOD gate.

    OOD projection FFR and individual-language held-out recalls are reported by
    the execution but intentionally absent here: prior rounds registered them
    as diagnostic.  The two R0132 OOD summaries are the frozen retention cells.
    """
    native_values = _finite_table(native, METRICS, label="native")
    matched_values = _finite_table(matched_2m, METRICS, label="matched 2M")
    baseline_values = _finite_table(
        baseline_2m_seed42, METRICS, label="baseline 2M seed42"
    )
    floor_values = _finite_table(prompted_floors, METRICS, label="prompted floors")
    if set(group_ffr) != set(GROUPS):
        raise Round0169Error("Q3 group FFR cells are incomplete")
    groups = {key: float(group_ffr[key]) for key in GROUPS}
    if not all(math.isfinite(value) and value >= 0 for value in groups.values()):
        raise Round0169Error("Q3 group FFR contains an invalid value")
    ood_names = (
        "polish_recall_at_50_of_high10",
        "in_mix_median_recall_at_50_of_high10",
    )
    prompted_ood_values = _finite_table(prompted_ood, ood_names, label="prompted OOD")
    raw_ood_values = _finite_table(raw_r0132_ood, ood_names, label="raw R0132 OOD")

    native_gates = {
        metric: {
            "observed": native_values[metric],
            "floor": floor_values[metric],
            "passed": _at_least(native_values[metric], floor_values[metric]),
        }
        for metric in NATIVE_ABSOLUTE_METRICS
    }
    matched_gates = {}
    for metric in METRICS:
        baseline = baseline_values[metric]
        if baseline <= 0:
            raise Round0169Error("Q3 baseline metric must be positive")
        ratio = matched_values[metric] / baseline
        matched_gates[metric] = {
            "observed": matched_values[metric],
            "baseline_seed42": baseline,
            "ratio": ratio,
            "minimum_ratio": RETENTION_RATIO,
            "passed": _at_least(ratio, RETENTION_RATIO),
        }

    pooled_english = sum(groups[name] for name in GROUPS[:3]) / 3.0
    language_floor = LANGUAGE_TO_POOLED_ENGLISH_RATIO * pooled_english
    language_gates = {
        language: {
            "observed": groups[language],
            "floor": language_floor,
            "passed": _at_least(groups[language], language_floor),
        }
        for language in IN_MIX_LANGUAGES
    }

    median = prompted_ood_values["in_mix_median_recall_at_50_of_high10"]
    polish = prompted_ood_values["polish_recall_at_50_of_high10"]
    if median <= 0:
        raise Round0169Error("Q3 prompted in-mix OOD median must be positive")
    polish_ratio = polish / median
    polish_gate = {
        "observed_polish": polish,
        "prompted_in_mix_median": median,
        "ratio": polish_ratio,
        "minimum_ratio": POLISH_TO_IN_MIX_MEDIAN_RATIO,
        "passed": _at_least(polish_ratio, POLISH_TO_IN_MIX_MEDIAN_RATIO),
    }
    ood_retention_gates = {}
    for name in ood_names:
        control = raw_ood_values[name]
        if control <= 0:
            raise Round0169Error("Q3 raw R0132 OOD control must be positive")
        ratio = prompted_ood_values[name] / control
        ood_retention_gates[name] = {
            "observed": prompted_ood_values[name],
            "raw_r0132_control": control,
            "ratio": ratio,
            "minimum_ratio": RETENTION_RATIO,
            "passed": _at_least(ratio, RETENTION_RATIO),
        }

    stacks = (
        native_gates,
        matched_gates,
        language_gates,
        ood_retention_gates,
    )
    passed = all(cell["passed"] for stack in stacks for cell in stack.values()) and polish_gate["passed"]
    return {
        "passed": passed,
        "outcome": (
            "prompted-diverse-u12-rung-qualified"
            if passed
            else "prompted-diverse-u12-rung-not-qualified"
        ),
        "native_absolute_gates": native_gates,
        "matched_2m_retention_gates": matched_gates,
        "language_relative_ffr": {
            "pooled_english_ffr": pooled_english,
            "ratio": LANGUAGE_TO_POOLED_ENGLISH_RATIO,
            "floor": language_floor,
            "cells": language_gates,
        },
        "polish_ood_gate": polish_gate,
        "raw_r0132_ood_retention_gates": ood_retention_gates,
        "diagnostic_only": [
            "native projection cells whose universe changes",
            "OOD projection FFR",
            "individual-language held-out recall cells",
        ],
    }
