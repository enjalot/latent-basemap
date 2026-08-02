"""Frozen contract for the 12.5M diverse-Jina prefix/drop rescue rung.

R0152 consumes R0151's outcome-blind population census.  It changes the
population, induced graph, and coverage-aligned training horizon together, so
the result is a package-level scale-transfer test rather than a unique causal
claim about duplicate removal or cardinality.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .artifact_identity import canonical_json, sha256_bytes
from .round0107_training import BATCH_SIZE, POSITIVE_ROWS_PER_UPDATE
from .round0140_subsystem_bisection import METRICS, RESTORATION_FLOORS, metric_view


ROUND_ID = "0152"
CAPABILITY = "jina-diverse-12p5m-prefix-drop-rescue-v1"
PARENT_CAPABILITY = "jina-diverse-12p5m-prefix-drop-only-census-v1"
RETAINED_ROWS = 12_485_206
RAW_PREFIX_ROWS = 12_500_000
FULL_RETAINED_ROWS = 24_948_663
GRAPH_K = 15
N_NEIGHBORS = GRAPH_K + 1
SEED = 42

SUBSET_SCHEMA = "round0152-prefix-drop-subset-v1"
INDEX_SCHEMA = "round0152-prefix-drop-search-index-v1"
QUALIFICATION_SCHEMA = "round0152-prefix-drop-search-qualification-v1"
GRAPH_SHARD_SCHEMA = "round0152-prefix-drop-graph-shard-v1"
GRAPH_PART_SCHEMA = "round0152-prefix-drop-graph-part-v1"
GRAPH_SCHEMA = "round0152-prefix-drop-fuzzy-graph-v1"
TRAIN_CONFIG_SCHEMA = "round0152-prefix-drop-train-config-v1"
PRODUCTION_CONFIG_SCHEMA = "round0152-prefix-drop-production-config-v1"
TRAIN_RECEIPT_SCHEMA = "round0152-prefix-drop-train-receipt-v1"
NATIVE_SCHEMA = "round0152-prefix-drop-matched-native-panel-v1"
OOD_SCHEMA = "round0152-prefix-drop-matched-ood-panel-v1"
FUNCTIONAL_SCHEMA = "round0152-prefix-drop-functional-density-panel-v1"
DECISION_SCHEMA = "round0152-prefix-drop-rescue-decision-v1"

PIPELINE = "host_weighted_jina_diverse_12p5m_prefix_drop"
PIPELINE_SCHEMA = "round0152-host-weighted-jina-diverse-prefix-drop-pipeline-v1"
SAMPLER_CLASS = "DiverseWeightedJinaSampler"
POSITIVE_DESTINATION_POLICY = (
    "R0152-global-prefix-drop-retained-fuzzy-tconorm-graph"
)
UPDATE_RULE = "ceil(actual-R0152-directed-fuzzy-edges/409)"

DENSITY_FLOOR = 0.17589389755990817
OOD_RETENTION = 0.97
OUTCOME_PASS = "prefix-drop-only-12p5m-rescue-passes"
OUTCOME_FAIL = "prefix-drop-only-12p5m-rescue-fails"
OUTCOME_INVALID = "invalid-execution"


class Round0152Error(RuntimeError):
    """The preregistered R0152 contract was violated."""


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise Round0152Error(f"{label} is not numeric")
    try:
        observed = float(value)
    except (TypeError, ValueError) as exc:
        raise Round0152Error(f"{label} is not numeric") from exc
    if not math.isfinite(observed):
        raise Round0152Error(f"{label} is nonfinite")
    return observed


def _inclusive_at_least(observed: float, floor: float) -> bool:
    return bool(
        observed >= floor
        or math.isclose(observed, floor, rel_tol=1e-12, abs_tol=1e-12)
    )


def coverage_aligned_updates(directed_edges: int) -> int:
    edges = int(directed_edges)
    if edges <= 0:
        raise Round0152Error("R0152 graph must contain positive edges")
    return (edges + POSITIVE_ROWS_PER_UPDATE - 1) // POSITIVE_ROWS_PER_UPDATE


def quality_selector(
    *,
    functional_cell: Mapping[str, Any],
    density_v2: float,
    candidate_ood: Mapping[str, Any],
    accepted_25m_ood: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen absolute functional/density and matched OOD gates."""
    functional = metric_view(functional_cell)
    functional_checks = {
        name: _inclusive_at_least(functional[name], RESTORATION_FLOORS[name])
        for name in METRICS
    }
    density = _finite(density_v2, label="density-v2")
    density_check = _inclusive_at_least(density, DENSITY_FLOOR)
    ood_names = (
        "fineweb_recall_at_50_of_high10",
        "polish_recall_at_50_of_high10",
        "in_mix_median_recall_at_50_of_high10",
    )
    ood_metrics: dict[str, Any] = {}
    ood_checks: dict[str, bool] = {}
    for name in ood_names:
        candidate = _finite(candidate_ood.get(name), label=f"candidate {name}")
        control = _finite(accepted_25m_ood.get(name), label=f"25M {name}")
        threshold = OOD_RETENTION * control
        ood_metrics[name] = {
            "candidate_12p5m": candidate,
            "accepted_25m": control,
            "floor": threshold,
            "retention": candidate / control if control else None,
        }
        ood_checks[f"{name}_retains_0p97_of_accepted_25m"] = (
            _inclusive_at_least(candidate, threshold)
        )
    checks = {
        **{f"functional_{name}": value for name, value in functional_checks.items()},
        "fixed_density_v2_floor": density_check,
        **ood_checks,
    }
    return {
        "functional": {
            name: {
                "observed": functional[name],
                "floor": RESTORATION_FLOORS[name],
                "passed": functional_checks[name],
            }
            for name in METRICS
        },
        "density_v2": {
            "observed": density,
            "floor": DENSITY_FLOOR,
            "passed": density_check,
        },
        "ood": ood_metrics,
        "checks": checks,
        "passed": all(checks.values()),
    }


def build_decision(
    *, validity_checks: Mapping[str, bool], quality: Mapping[str, Any]
) -> dict[str, Any]:
    if not validity_checks or any(value is not True for value in validity_checks.values()):
        outcome = OUTCOME_INVALID
    elif quality.get("passed") is True:
        outcome = OUTCOME_PASS
    else:
        outcome = OUTCOME_FAIL
    return {
        "schema": DECISION_SCHEMA,
        "round_id": ROUND_ID,
        "capability": CAPABILITY,
        "outcome": outcome,
        "validity_checks": dict(validity_checks),
        "quality_selector": dict(quality),
        "atlas_rescue_candidate_released": outcome == OUTCOME_PASS,
        "full_25m_prefix_drop_test_released": outcome == OUTCOME_PASS,
        "registry_promotion_released": False,
        "production_ready": False,
        "unique_duplicate_or_cardinality_cause_claimed": False,
        "pure_scale_effect_claimed": False,
        "one_seed_limitation": "seed-42 package-level scale transfer only",
    }


def validate_train_execution(
    *, train: Mapping[str, Any], config_receipt: Mapping[str, Any], graph: Mapping[str, Any]
) -> dict[str, Any]:
    """Authenticate the exact graph-derived horizon and runtime pipeline."""
    config = config_receipt.get("config")
    if not isinstance(config, Mapping):
        raise Round0152Error("R0152 production config is missing")
    edges = int(graph.get("directed_edge_count", -1))
    updates = coverage_aligned_updates(edges)
    expected_draws = updates * POSITIVE_ROWS_PER_UPDATE
    expected_rows = updates * BATCH_SIZE
    optimizer = config.get("optimizer") or {}
    execution = config.get("execution") or {}
    config_graph = config.get("graph") or {}
    config_input = config.get("input") or {}
    expected_stamp = execution.get("expected_pipeline_stamp") or {}
    accounting = train.get("train_accounting") or {}
    runtime = train.get("exact_execution_receipt") or {}
    derivation = train.get("update_derivation") or {}
    train_checks = train.get("train_checks") or {}
    profiler = train.get("performance_profile") or {}

    static_stamp = {
        "schema": PIPELINE_SCHEMA,
        "pipeline": PIPELINE,
        "sampler_class": SAMPLER_CLASS,
        "positive_sampling": (
            "fuzzy_weight_proportional_with_replacement_via_exact_"
            "uniform_envelope_rejection"
        ),
        "positive_destination_policy": POSITIVE_DESTINATION_POLICY,
        "negative_sampling": (
            f"uniform-{RETAINED_ROWS:,}-compact-retained-rows-nonself"
        ),
        "graph_degree": "variable-symmetric-fuzzy-k15-topology",
        "host_prefetch": "single-producer-two-pinned-slot",
        "endpoint_forward": "fused-source-destination",
        "valid_canonical_edge_count": edges,
        "compact_retained_rows": RETAINED_ROWS,
        "weighted_requested": True,
        "weighted_effective": True,
        "uniform_with_replacement": False,
        "positive_with_replacement": True,
        "weight_sampler": "uniform-envelope-rejection-max-weight-one",
        "weight_uniform_dtype": "<f8",
        "source_representation": "int8-treatment",
    }
    exact_accounting = {
        "lr_horizon": updates,
        "positive_lr_optimizer_steps": updates,
        "scheduler_steps": updates,
        "attempted_batches": updates,
        "finite_loss_batches": updates,
        "optimizer_steps_attempted": updates,
        "optimizer_steps_succeeded": updates,
        "amp_overflow_skips": 0,
        "nonfinite_loss_skips": 0,
        "nonfinite_gradient_skips": 0,
        "stop_reason": "lr_horizon",
        "budget_satisfied": True,
        "n_pos_edges": edges,
    }
    dynamic_keys = (
        "endpoint_gather_calls",
        "source_rows_gathered",
        "destination_rows_gathered",
        "host_prefetch_batches_filled",
        "host_prefetch_producer_batches",
        "host_prefetch_consumer_batches",
        "host_prefetch_source_rows_filled",
        "host_prefetch_destination_rows_filled",
        "weight_proposals",
        "weight_acceptances",
        "weight_emitted_draws",
        "weight_buffered_draws",
        "weight_acceptance_rate",
        "weight_rejection_iterations",
    )
    checks = {
        "receipt_schema_and_round": (
            train.get("schema") == TRAIN_RECEIPT_SCHEMA
            and train.get("round_id") == ROUND_ID
        ),
        "config_schema_and_round": (
            config_receipt.get("schema") == PRODUCTION_CONFIG_SCHEMA
            and config_receipt.get("round_id") == ROUND_ID
            and config.get("schema") == TRAIN_CONFIG_SCHEMA
        ),
        "config_hash_closes": (
            config_receipt.get("config_sha256")
            == sha256_bytes(canonical_json(config))
            == train.get("production_config_sha256")
        ),
        "graph_identity_closes": (
            graph.get("schema") == GRAPH_SCHEMA
            and graph.get("round_id") == ROUND_ID
            and int(graph.get("retained_rows", -1)) == RETAINED_ROWS
            and int(graph.get("k_real", -1)) == GRAPH_K
            and int(graph.get("n_neighbors_including_self", -1)) == N_NEIGHBORS
            and int(config_graph.get("directed_edges", -1)) == edges
            and config_input.get("rows") == RETAINED_ROWS
        ),
        "coverage_horizon_closes": (
            optimizer.get("successful_positive_lr_updates") == updates
            and optimizer.get("positive_rows_per_update") == POSITIVE_ROWS_PER_UPDATE
            and optimizer.get("update_rule") == UPDATE_RULE
            and derivation.get("directed_fuzzy_edges") == edges
            and derivation.get("positive_rows_per_update") == POSITIVE_ROWS_PER_UPDATE
            and derivation.get("successful_updates") == updates
            and derivation.get("expected_positive_draws") == expected_draws
            and train.get("optimizer_updates") == updates
        ),
        "registered_pipeline_requested": (
            execution.get("required_pipeline") == PIPELINE
            and expected_stamp == static_stamp
        ),
        "actual_pipeline_matches_config": (
            bool(expected_stamp)
            and all(runtime.get(key) == value for key, value in expected_stamp.items())
        ),
        "exact_optimizer_accounting": all(
            accounting.get(key) == value for key, value in exact_accounting.items()
        ),
        "endpoint_accounting_closes": (
            runtime.get("source_rows_gathered") == expected_rows
            and runtime.get("destination_rows_gathered") == expected_rows
            and runtime.get("host_prefetch_consumer_batches") == updates
            and runtime.get("host_prefetch_producer_batches") in {updates, updates + 1}
        ),
        "weighted_draw_accounting_closes": (
            runtime.get("weight_emitted_draws") == expected_draws
            and runtime.get("weight_acceptances")
            == int(runtime.get("weight_emitted_draws", -1))
            + int(runtime.get("weight_buffered_draws", -1))
            and int(runtime.get("weight_proposals", -1))
            >= int(runtime.get("weight_acceptances", 0))
            and 0 < float(runtime.get("weight_acceptance_rate", 0.0)) <= 1
        ),
        "runtime_and_flattened_accounting_agree": all(
            key in runtime
            and accounting.get(f"pipeline_{key}") == runtime.get(key)
            for key in dynamic_keys
        ),
        "train_checks_exact_and_positive": (
            set(train_checks)
            == {
                "exact_update_closure",
                "zero_numerical_skips",
                "no_pipeline_stamp_drift",
                "endpoint_rows_match_updates",
                "weighted_rejection_accounting_closes",
            }
            and all(value is True for value in train_checks.values())
        ),
        "performance_admission_closes": (
            profiler.get("aborted") is False
            and _finite(train.get("steady_updates_per_s"), label="train rate")
            >= _finite(execution.get("minimum_train_upd_s"), label="rate floor")
        ),
        "training_receipt_closes": (
            train.get("training_performed") is True
            and train.get("evaluation_performed") is False
            and train.get("map_decision_made") is False
        ),
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise Round0152Error(f"R0152 train authentication failed: {failed}")
    return {
        "checks": checks,
        "directed_fuzzy_edges": edges,
        "successful_updates": updates,
        "expected_positive_draws": expected_draws,
        "expected_endpoint_rows": expected_rows,
        "actual_pipeline_stamp": dict(runtime),
    }
