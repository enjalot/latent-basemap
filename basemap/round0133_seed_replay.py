"""Pure contracts for the seed-43 replay of R0132's matched scale bridge.

R0133 changes only the RNG identity of the U12 training treatment.  The
population, graph, successful-update horizon, model/training policy, panel
math, selectors, and thresholds remain the reviewed R0132 values.  The
accepted R0109 model supplies the seed-43 25M arm; R0110 coordinates are not
released evidence for this consumer and are forbidden.
"""
from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from typing import Any

from .artifact_identity import canonical_json, sha256_bytes
from .round0132_scale_bridge import (
    DECISION_SCHEMA as R0132_DECISION_SCHEMA,
    OUTCOME_DENSITY_REGRESSION,
    OUTCOME_INCONCLUSIVE,
    OUTCOME_INVALID,
    OUTCOME_QUALITY_REGRESSION,
    OUTCOME_SUPPORTED,
    PRODUCTION_CONFIG_SCHEMA as R0132_PRODUCTION_CONFIG_SCHEMA,
    SCALE_POLICY_CAPABILITY,
    TRAIN_CONFIG_SCHEMA as R0132_TRAIN_CONFIG_SCHEMA,
    TRAIN_RECEIPT_SCHEMA as R0132_TRAIN_RECEIPT_SCHEMA,
    Round0132Error,
    scale_policy_decision,
    validate_train_execution,
)


ROUND_ID = "0133"
SEED = 43

TRAIN_CONFIG_SCHEMA = "round0133-half-seed43-train-config-v1"
PRODUCTION_CONFIG_SCHEMA = "round0133-half-seed43-production-config-v1"
TRAIN_RECEIPT_SCHEMA = "round0133-half-seed43-train-receipt-v1"
TRANSFORM_RECEIPT_SCHEMA = "round0133-seed43-u12-transform-pair-v1"
NATIVE_SCHEMA = "round0133-matched-native-scale-panel-v1"
OOD_SCHEMA = "round0133-matched-ood-scale-panel-v1"
DECISION_SCHEMA = "round0133-scale-policy-two-seed-decision-v1"

TWO_SEED_CAPABILITY = (
    "jina-diverse-12p5m-25m-scale-policy-geometry-two-seed-v1"
)
CONCORDANT = "two-seed-scale-policy-result-concordant"
DISCORDANT = "two-seed-scale-policy-result-discordant"

R0110_COORDINATE_ROOT = (
    "/data/latent-basemap/runs/round-0110/queue/artifacts/coordinates-seed43"
)

SEED_OUTCOMES = {
    OUTCOME_SUPPORTED,
    OUTCOME_DENSITY_REGRESSION,
    OUTCOME_QUALITY_REGRESSION,
    OUTCOME_INCONCLUSIVE,
    OUTCOME_INVALID,
}


class Round0133Error(RuntimeError):
    """The preregistered R0133 replay contract was violated."""


def _config(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    config = value.get("config")
    if not isinstance(config, Mapping):
        raise Round0133Error(f"{label} production config is missing")
    return copy.deepcopy(dict(config))


def normalized_training_policy(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize only the two registered seed-replay identity fields."""
    value = copy.deepcopy(dict(config))
    optimizer = value.get("optimizer")
    if not isinstance(optimizer, dict):
        raise Round0133Error("training optimizer identity is missing")
    value["schema"] = R0132_TRAIN_CONFIG_SCHEMA
    optimizer["seed"] = 42
    return value


def validate_seed43_train_execution(
    *,
    train: Mapping[str, Any],
    config_receipt: Mapping[str, Any],
    graph: Mapping[str, Any],
    accepted_r0132_config_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate R0133 by reducing it to the reviewed R0132 law.

    The original R0133 hash and schema close first.  We then normalize only
    the round-specific schema and seed, require byte-semantic equality to the
    accepted R0132 config, and pass the normalized receipt through R0132's
    unchanged accounting/pipeline validator.
    """
    config = _config(config_receipt, label="R0133")
    accepted = _config(accepted_r0132_config_receipt, label="accepted R0132")
    optimizer = config.get("optimizer") or {}
    accepted_optimizer = accepted.get("optimizer") or {}
    original_digest = sha256_bytes(canonical_json(config))
    if (
        train.get("schema") != TRAIN_RECEIPT_SCHEMA
        or train.get("round_id") != ROUND_ID
        or config_receipt.get("schema") != PRODUCTION_CONFIG_SCHEMA
        or config_receipt.get("round_id") != ROUND_ID
        or config.get("schema") != TRAIN_CONFIG_SCHEMA
        or optimizer.get("seed") != SEED
        or accepted_r0132_config_receipt.get("schema")
        != R0132_PRODUCTION_CONFIG_SCHEMA
        or accepted_r0132_config_receipt.get("round_id") != "0132"
        or accepted.get("schema") != R0132_TRAIN_CONFIG_SCHEMA
        or accepted_optimizer.get("seed") != 42
        or config_receipt.get("config_sha256") != original_digest
        or train.get("production_config_sha256") != original_digest
    ):
        raise Round0133Error("R0133 seed-43 train identity is incomplete")

    normalized = normalized_training_policy(config)
    if normalized != accepted:
        raise Round0133Error(
            "R0133 training policy differs from accepted R0132 beyond RNG identity"
        )

    normalized_digest = sha256_bytes(canonical_json(normalized))
    normalized_train = copy.deepcopy(dict(train))
    normalized_train.update({
        "schema": R0132_TRAIN_RECEIPT_SCHEMA,
        "round_id": "0132",
        "production_config_sha256": normalized_digest,
    })
    normalized_receipt = {
        **copy.deepcopy(dict(config_receipt)),
        "schema": R0132_PRODUCTION_CONFIG_SCHEMA,
        "round_id": "0132",
        "config": normalized,
        "config_sha256": normalized_digest,
    }
    try:
        authenticated = validate_train_execution(
            train=normalized_train,
            config_receipt=normalized_receipt,
            graph=graph,
        )
    except Round0132Error as exc:
        raise Round0133Error("R0133 execution differs from R0132's train law") from exc
    return {
        **authenticated,
        "registered_seed": SEED,
        "accepted_r0132_seed": 42,
        "only_registered_rng_identity_changed": True,
        "r0133_config_sha256": original_digest,
        "normalized_r0132_config_sha256": normalized_digest,
    }


def seed43_scale_policy_decision(
    *,
    validity_checks: Mapping[str, bool],
    density: Mapping[str, Any],
    quality: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply R0132's selector unchanged while labelling the new seed honestly."""
    selected = scale_policy_decision(
        validity_checks=validity_checks,
        density=density,
        quality=quality,
    )
    return {
        "outcome": selected["outcome"],
        "validity_checks": selected["validity_checks"],
        "density_selector": selected["density_selector"],
        "quality_and_ood_noninferiority": selected[
            "quality_and_ood_noninferiority"
        ],
        "seed": SEED,
        "selector_source": "unchanged R0132 seed-level selector",
        "estimand": selected["estimand"],
        "stale_absolute_jina_floor_role": selected[
            "stale_absolute_jina_floor_role"
        ],
        "native_global_ffr_role": selected["native_global_ffr_role"],
        "ood_projection_ffr_role": selected["ood_projection_ffr_role"],
        "trec_covid_role": selected["trec_covid_role"],
        "dadabase_role": selected["dadabase_role"],
        "one_seed_limitation": (
            "seed-43 matched contrast only; no seed-variance robustness claim"
        ),
    }


def failed_quality_gates(decision: Mapping[str, Any]) -> list[str]:
    quality = decision.get("quality_and_ood_noninferiority") or {}
    checks = quality.get("checks") or {}
    if not isinstance(checks, Mapping) or any(
        not isinstance(key, str) or value not in (True, False)
        for key, value in checks.items()
    ):
        raise Round0133Error("seed-level quality checks are malformed")
    return sorted(key for key, value in checks.items() if value is False)


def validate_accepted_seed42_decision(
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    outcome = decision.get("outcome")
    if (
        decision.get("schema") != R0132_DECISION_SCHEMA
        or decision.get("round_id") != "0132"
        or outcome not in SEED_OUTCOMES - {OUTCOME_INVALID}
        or decision.get("capabilities_produced") != [SCALE_POLICY_CAPABILITY]
    ):
        raise Round0133Error("accepted R0132 seed-42 decision is ineligible")
    return {
        "seed": 42,
        "outcome": outcome,
        "failed_quality_gates": failed_quality_gates(decision),
    }


def combine_seed_decisions(
    *,
    accepted_seed42: Mapping[str, Any],
    seed43: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine labels only; never pool per-seed anchors or bootstrap draws."""
    seed42 = validate_accepted_seed42_decision(accepted_seed42)
    outcome43 = seed43.get("outcome")
    if outcome43 not in SEED_OUTCOMES:
        raise Round0133Error("seed-43 outcome is not registered")
    failed43 = failed_quality_gates(seed43)
    same_outcome = seed42["outcome"] == outcome43
    same_failed_set = seed42["failed_quality_gates"] == failed43
    concordant = same_outcome and (
        outcome43 != OUTCOME_QUALITY_REGRESSION or same_failed_set
    )
    valid = outcome43 != OUTCOME_INVALID
    return {
        "seed42": seed42,
        "seed43": {
            "seed": SEED,
            "outcome": outcome43,
            "failed_quality_gates": failed43,
        },
        "concordance": CONCORDANT if concordant else DISCORDANT,
        "same_seed_level_outcome": same_outcome,
        "same_failed_quality_gate_set": same_failed_set,
        "bootstrap_combination": "none",
        "anchors_combined": False,
        "point_estimates_averaged": False,
        "population_seed_variance_estimated": False,
        "interpretation": (
            "replicated evidence for this exact scale-policy bundle only; "
            "not population-level seed robustness"
        ),
        "capabilities_produced": [TWO_SEED_CAPABILITY] if valid else [],
    }


def assert_no_r0110_coordinate_inputs(value: Any) -> None:
    """Reject the unreleased R0110 coordinate stream anywhere in a payload."""
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        resolved = (
            os.path.realpath(value).replace("\\", "/")
            if os.path.isabs(value)
            else normalized
        )
        if any(
            candidate == R0110_COORDINATE_ROOT
            or candidate.startswith(R0110_COORDINATE_ROOT + "/")
            for candidate in (normalized, resolved)
        ):
            raise Round0133Error("R0110 coordinate inputs are forbidden")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_no_r0110_coordinate_inputs(key)
            assert_no_r0110_coordinate_inputs(item)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            assert_no_r0110_coordinate_inputs(item)
