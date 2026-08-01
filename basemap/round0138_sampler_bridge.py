"""Frozen config and decision arithmetic for the R0138 device-sampler bridge."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from .artifact_identity import canonical_json, sha256_bytes
from .round0104_training import (
    DIMENSION,
    ROWS,
    SEED,
    SUCCESSFUL_UPDATES,
    preprocessing_stamp,
    train_config as host_train_config,
)
from .round0134_functional_showdown import (
    COMPARISON_TOLERANCE,
    METRIC_ORDER,
    decision_metrics,
)


ROUND_ID = "0138"
PANEL_SCHEMA = "round0138-device-sampler-functional-panel-v1"
DECISION_SCHEMA = "round0138-device-sampler-bridge-decision-v1"
CAPABILITY = "jina-current-2m-device-sampler-bridge-v1"

HISTORICAL = "historical_r0037_seed42"
CONTROL = "current_r0104_fp16_seed42"
TREATMENT = "current_device_sampler_seed42"
CELL_ORDER = (HISTORICAL, CONTROL, TREATMENT)

PIPELINE = "device"
SAMPLER_CLASS = "DeviceEdgeSampler"
TRAIN_MINIMUM_UPDATES_PER_S = 100.0
TRAIN_WARNING_UPDATES_PER_S = 175.0


class Round0138Error(RuntimeError):
    """The registered R0138 treatment or selector was violated."""


def train_config(
    *,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    graph_edges: int,
) -> tuple[dict[str, Any], str]:
    """Change only R0104's endpoint sampler/runtime to the legacy device path."""
    base, _digest = host_train_config(
        "fp16_control",
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        graph_edges=graph_edges,
    )
    config = copy.deepcopy(base)
    config["schema"] = "round0138-device-sampler-bridge-config-v1"
    config["arm"] = "device_sampler_treatment"
    config["causal_change"] = "host-to-device-sampler-runtime-only"
    config["paired_invariant"]["sampler"] = SAMPLER_CLASS
    config["paired_invariant"]["rows"] = ROWS
    config["paired_invariant"]["dimension"] = DIMENSION
    config["paired_invariant"]["seed"] = SEED
    config["paired_invariant"]["successful_positive_lr_updates"] = (
        SUCCESSFUL_UPDATES
    )
    execution = config["execution"]
    execution.update({
        "required_pipeline": PIPELINE,
        "gpu_resident_data": True,
        "gpu_resident_vram_budget_gb": 31.0,
        "minimum_train_upd_s": TRAIN_MINIMUM_UPDATES_PER_S,
        "warning_train_upd_s": TRAIN_WARNING_UPDATES_PER_S,
        "expected_pipeline_stamp": {
            "pipeline": PIPELINE,
            "sampler_class": SAMPLER_CLASS,
            "positive_sampling": "weighted_with_replacement",
            "x_residency": "device_fp16",
            "weighted_requested": True,
            "weighted_effective": True,
            "uniform_with_replacement": False,
            "positive_with_replacement": True,
            "multiplicity_policy": "row_multiplicity_uncapped",
            **preprocessing_stamp("fp16_control"),
        },
    })
    return config, sha256_bytes(canonical_json(config))


def _contrast(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    baseline = decision_metrics(left)
    candidate = decision_metrics(right)
    metrics: dict[str, Any] = {}
    for metric in METRIC_ORDER:
        delta = candidate[metric] - baseline[metric]
        metrics[metric] = {
            "baseline": baseline[metric],
            "candidate": candidate[metric],
            "delta_candidate_minus_baseline": delta,
            "candidate_at_least_baseline": delta >= -COMPARISON_TOLERANCE,
        }
    return {
        "metrics": metrics,
        "candidate_at_least_baseline_on_all_metrics": all(
            row["candidate_at_least_baseline"] for row in metrics.values()
        ),
    }


def build_decision(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if set(cells) != set(CELL_ORDER) or len(cells) != len(CELL_ORDER):
        raise Round0138Error("sampler-bridge functional cells are missing or unexpected")
    # Canonical JSON sorts object keys.  Authenticate the exact cell set, then
    # restore the preregistered order before applying the frozen selector.
    cells = {key: cells[key] for key in CELL_ORDER}
    versus_control = _contrast(cells[CONTROL], cells[TREATMENT])
    versus_historical = _contrast(cells[HISTORICAL], cells[TREATMENT])
    restores_historical = versus_historical[
        "candidate_at_least_baseline_on_all_metrics"
    ]
    preserves_control = versus_control[
        "candidate_at_least_baseline_on_all_metrics"
    ]
    sufficient = restores_historical and preserves_control
    if sufficient:
        outcome = "device-sampler-sufficient-to-restore-function"
    elif not preserves_control:
        outcome = "device-sampler-regresses-current-control"
    else:
        outcome = "device-sampler-insufficient-to-restore-function"
    return {
        "schema": DECISION_SCHEMA,
        "round_id": ROUND_ID,
        "selector": {
            "metrics": list(METRIC_ORDER),
            "point_comparison": "candidate >= baseline",
            "absolute_tolerance": COMPARISON_TOLERANCE,
            "restoration_requires_all_metrics_vs_historical_and_control": True,
            "no_density_floor_or_posthoc_margin": True,
        },
        "treatment_vs_current_control": versus_control,
        "treatment_vs_historical_target": versus_historical,
        "outcome": outcome,
        "device_sampler_sufficient": sufficient,
        "restores_historical_on_all_metrics": restores_historical,
        "preserves_current_control_on_all_metrics": preserves_control,
        "density_recalibration_authorized": False,
        "training_performed": True,
        "registered_density_floor_changed": False,
        "map_registry_state_changed": False,
    }
