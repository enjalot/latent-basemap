"""Evaluate the seed-43 diverse-Jina replicate under R0108's frozen protocol."""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_write_new_json, create_fresh_directory
from basemap.round0108_evaluation import (
    DECISION_SCHEMA as R0108_DECISION_SCHEMA,
    EMBEDDING_PROMPT,
    MAP_KEY as SEED42_MAP_KEY,
    OOD_SCHEMA as R0108_OOD_SCHEMA,
    CORE_SCHEMA as R0108_CORE_SCHEMA,
    POLISH,
    Round0108Error,
    load_reviewed_model,
    read_sealed,
    seal,
)
from experiments import round0108_nodes as seed42_nodes
from experiments.round0109_nodes import (
    PRODUCTION_CONFIG_SCHEMA,
    SEED,
    TRAIN_RECEIPT_SCHEMA,
)


ROUND_ID = "0110"
MAP_KEY = "r0109-diverse-jina-25m-seed43"
MAP_LABEL = MAP_KEY
CORE_SCHEMA = "round0110-diverse-jina-core-geometry-v1"
OOD_SCHEMA = "round0110-diverse-jina-ood-evaluation-v1"
DECISION_SCHEMA = "round0110-diverse-jina-two-seed-decision-v1"


def _seed43_model(
    *,
    train_output: str,
    graph_manifest_path: str,
    graph_manifest_sha256: str,
) -> dict[str, Any]:
    return load_reviewed_model(
        train_output=train_output,
        graph_manifest_path=graph_manifest_path,
        graph_manifest_sha256=graph_manifest_sha256,
        expected_train_round_id="0109",
        expected_train_receipt_schema=TRAIN_RECEIPT_SCHEMA,
        expected_production_config_schema=PRODUCTION_CONFIG_SCHEMA,
        expected_seed=SEED,
    )


def _configure_seed43_contract() -> None:
    """Bind R0108's scorer implementation to the registered seed-43 identity.

    Runner nodes execute in separate processes, so these module-local bindings
    cannot leak into a different job. The frozen selectors, scorers, thresholds,
    calibration schema, and prompt semantics remain those of R0108.
    """
    seed42_nodes.ROUND_ID = ROUND_ID
    seed42_nodes.MAP_KEY = MAP_KEY
    seed42_nodes.MAP_LABEL = MAP_LABEL
    seed42_nodes.CORE_SCHEMA = CORE_SCHEMA
    seed42_nodes.OOD_SCHEMA = OOD_SCHEMA
    seed42_nodes.load_reviewed_model = _seed43_model


def run_transform(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    _configure_seed43_contract()
    return seed42_nodes.run_transform(active, job)


def run_core(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    _configure_seed43_contract()
    return seed42_nodes.run_core_score(active, job)


def run_ood(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    _configure_seed43_contract()
    return seed42_nodes.run_ood(active, job)


def _metric(receipt: Mapping[str, Any], *keys: str) -> float:
    value: Any = receipt
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise Round0108Error(
                f"R0110 comparison metric is missing: {'/'.join(keys)}"
            )
        value = value[key]
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise Round0108Error(
            f"R0110 comparison metric is nonnumeric: {'/'.join(keys)}"
        ) from exc


def run_decision(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    """Release two-seed quality only when both seeds pass the same gates."""
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0110 two-seed decision"
    )
    seed42_decision_path = str(job["seed42_decision"])
    seed42_core_path = str(job["seed42_core"])
    seed42_ood_path = str(job["seed42_ood"])
    seed43_core_path = os.path.join(
        str(job["core_output"]), "core-geometry.json"
    )
    seed43_ood_path = os.path.join(
        str(job["ood_output"]), "ood-evaluation.json"
    )
    seed42_decision = read_sealed(
        seed42_decision_path,
        label="R0108 seed-42 atlas decision",
        schema=R0108_DECISION_SCHEMA,
    )
    seed42_core = read_sealed(
        seed42_core_path,
        label="R0108 seed-42 core geometry",
        schema=R0108_CORE_SCHEMA,
    )
    seed42_ood = read_sealed(
        seed42_ood_path,
        label="R0108 seed-42 OOD evaluation",
        schema=R0108_OOD_SCHEMA,
    )
    seed43_core = read_sealed(
        seed43_core_path,
        label="R0110 seed-43 core geometry",
        schema=CORE_SCHEMA,
    )
    seed43_ood = read_sealed(
        seed43_ood_path,
        label="R0110 seed-43 OOD evaluation",
        schema=OOD_SCHEMA,
    )

    if (
        seed42_decision.get("round_id") != "0108"
        or seed42_decision.get("map_key") != SEED42_MAP_KEY
        or seed42_core.get("map_key") != SEED42_MAP_KEY
        or seed42_ood.get("map_key") != SEED42_MAP_KEY
        or seed43_core.get("round_id") != ROUND_ID
        or seed43_core.get("map_key") != MAP_KEY
        or seed43_ood.get("round_id") != ROUND_ID
        or seed43_ood.get("map_key") != MAP_KEY
    ):
        raise Round0108Error("R0110 seed comparison identity changed")

    seed42_core_passed = bool(
        (seed42_core.get("decision") or {}).get("passed")
    )
    seed42_ood_passed = bool(
        (seed42_ood.get("headline_decision") or {}).get("passed")
    )
    seed42_quality_passed = bool(
        seed42_decision.get("atlas_quality_capability_released")
    )
    if seed42_quality_passed != (
        seed42_core_passed and seed42_ood_passed
    ):
        raise Round0108Error("R0108 seed-42 decision does not close")

    seed43_core_passed = bool(
        (seed43_core.get("decision") or {}).get("passed")
    )
    seed43_ood_passed = bool(
        (seed43_ood.get("headline_decision") or {}).get("passed")
    )
    prompt_identity_closes = all(
        receipt.get("embedding_prompt") == EMBEDDING_PROMPT
        and receipt.get("prompt_applied") is False
        for receipt in (seed42_ood, seed43_ood)
    )
    two_seed_passed = (
        seed42_quality_passed
        and seed43_core_passed
        and seed43_ood_passed
        and prompt_identity_closes
    )

    comparisons = {
        "core_global_ffr": {
            "seed42": _metric(
                seed42_core, "metrics", "global", "ffr"
            ),
            "seed43": _metric(
                seed43_core, "metrics", "global", "ffr"
            ),
        },
        "core_global_recall_at_10": {
            "seed42": _metric(
                seed42_core, "metrics", "global", "recall_at_10"
            ),
            "seed43": _metric(
                seed43_core, "metrics", "global", "recall_at_10"
            ),
        },
        "core_global_recall_at_50_of_high10": {
            "seed42": _metric(
                seed42_core,
                "metrics",
                "global",
                "recall_at_50_of_high10",
            ),
            "seed43": _metric(
                seed43_core,
                "metrics",
                "global",
                "recall_at_50_of_high10",
            ),
        },
        "density_v2": {
            "seed42": _metric(
                seed42_core, "metrics", "density_v2", "correlation"
            ),
            "seed43": _metric(
                seed43_core, "metrics", "density_v2", "correlation"
            ),
        },
        "polish_recall_at_10": {
            "seed42": _metric(
                seed42_ood,
                "language_cells",
                POLISH,
                "probe",
                "recall_at_10",
            ),
            "seed43": _metric(
                seed43_ood,
                "language_cells",
                POLISH,
                "probe",
                "recall_at_10",
            ),
        },
        "polish_recall_at_50_of_high10": {
            "seed42": _metric(
                seed42_ood,
                "language_cells",
                POLISH,
                "probe",
                "recall_at_50_of_high10",
            ),
            "seed43": _metric(
                seed43_ood,
                "language_cells",
                POLISH,
                "probe",
                "recall_at_50_of_high10",
            ),
        },
        "polish_to_in_mix_median_ratio": {
            "seed42": _metric(
                seed42_ood,
                "headline_decision",
                "polish_to_in_mix_median_ratio",
            ),
            "seed43": _metric(
                seed43_ood,
                "headline_decision",
                "polish_to_in_mix_median_ratio",
            ),
        },
    }
    comparisons = {
        name: {
            **values,
            "seed43_minus_seed42": values["seed43"] - values["seed42"],
            "role": "diagnostic-only",
        }
        for name, values in comparisons.items()
    }
    receipt = seal({
        "schema": DECISION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "seed42": {
            "training_round": "0107",
            "seed": 42,
            "decision": expected_input_signature(seed42_decision_path),
            "core": expected_input_signature(seed42_core_path),
            "ood": expected_input_signature(seed42_ood_path),
        },
        "seed43": {
            "training_round": "0109",
            "seed": SEED,
            "core": expected_input_signature(seed43_core_path),
            "ood": expected_input_signature(seed43_ood_path),
        },
        "checks": {
            "seed42_fixed_core_gate_passed": seed42_core_passed,
            "seed42_fixed_polish_ood_gate_passed": seed42_ood_passed,
            "seed42_atlas_quality_passed": seed42_quality_passed,
            "seed43_fixed_core_gate_passed": seed43_core_passed,
            "seed43_fixed_polish_ood_gate_passed": seed43_ood_passed,
            "raw_prompt_identity_closes": prompt_identity_closes,
            "cross_seed_deltas_excluded_from_decision": True,
            "projection_ffr_excluded_from_decision": True,
        },
        "comparison_metrics": comparisons,
        "two_seed_quality_capability_released": two_seed_passed,
        "outcome": (
            "two-seed-quality-accepted"
            if two_seed_passed
            else "two-seed-quality-not-released"
        ),
        "embedding_prompt": EMBEDDING_PROMPT,
        "prompt_applied": False,
        "production_document_prompt_transfer_resolved": False,
        "production_readiness_claimed": False,
        "training_performed": False,
    })
    path = os.path.join(output, "two-seed-decision.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID or job is None:
        raise Round0108Error("R0110 handler requires its exact round/job")
    handlers = {
        "transform_seed43": run_transform,
        "score_seed43_core": run_core,
        "score_seed43_ood": run_ood,
        "decide_seed_stability": run_decision,
    }
    try:
        handler = handlers[str(job.get("action"))]
    except KeyError as exc:
        raise Round0108Error(
            f"unknown R0110 action {job.get('action')!r}"
        ) from exc
    return handler(active, job)
