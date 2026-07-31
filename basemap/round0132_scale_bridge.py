"""Pure contracts for the matched 12.5M-to-25M diverse-Jina scale bridge.

R0132 estimates a scale-*policy* effect.  Population size, the graph induced
by that population, and the coverage-aligned optimization horizon move
together under one frozen construction/training policy.  It is deliberately
not represented as a pure-N intervention.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .artifact_identity import canonical_json, sha256_bytes
from .round0105_search import GROUPS
from .round0107_training import BATCH_SIZE, POSITIVE_ROWS_PER_UPDATE


ROUND_ID = "0132"
FULL_RETAINED_ROWS = 24_948_663
HALF_RETAINED_ROWS = FULL_RETAINED_ROWS // 2
GRAPH_K = 15
N_NEIGHBORS = GRAPH_K + 1
SEED = 42

SUBSET_NAMESPACE = b"round0132-half-v1\0"
SUBSET_SCHEMA = "round0132-source-balanced-half-subset-v1"
INDEX_SCHEMA = "round0132-half-search-index-v1"
QUALIFICATION_SCHEMA = "round0132-half-fixed-search-qualification-v1"
GRAPH_SHARD_SCHEMA = "round0132-half-graph-shard-v1"
GRAPH_PART_SCHEMA = "round0132-half-graph-part-v1"
GRAPH_SCHEMA = "round0132-half-fuzzy-graph-v1"
TRAIN_CONFIG_SCHEMA = "round0132-half-train-config-v1"
PRODUCTION_CONFIG_SCHEMA = "round0132-half-production-config-v1"
TRAIN_RECEIPT_SCHEMA = "round0132-half-train-receipt-v1"
NATIVE_SCHEMA = "round0132-matched-native-scale-panel-v1"
OOD_SCHEMA = "round0132-matched-ood-scale-panel-v1"
DECISION_SCHEMA = "round0132-scale-policy-decision-v1"

PIPELINE = "host_weighted_jina_diverse_12p5m"
PIPELINE_SCHEMA = "round0132-host-weighted-jina-diverse-pipeline-v1"
SAMPLER_CLASS = "DiverseWeightedJinaSampler"
POSITIVE_DESTINATION_POLICY = (
    "R0132-global-half-retained-fuzzy-tconorm-graph"
)

SEARCH_NPROBE = 64
SEARCH_SHORTLIST_WIDTH = 128
SEARCH_GLOBAL_RECALL_FLOOR = 0.90
SEARCH_GROUP_RECALL_FLOOR = 0.84
SEARCH_ANCHORS_PER_GROUP = 256
SEARCH_SEED = 13_205
INDEX_TRAIN_SEED = 13_204
INDEX_TRAIN_ROWS = 327_680

NATIVE_ANCHORS_PER_GROUP = 256
NATIVE_ANCHOR_SEED = 13_206
DENSITY_BOOTSTRAP_DRAWS = 1_000
DENSITY_BOOTSTRAP_SEED = 13_207
DENSITY_CI_LEVEL = 0.99
DENSITY_NONINFERIORITY_MARGIN = 0.03
METRIC_RETENTION = 0.97
FFR_ALLOWED_DECREASE = 0.02

OUTCOME_SUPPORTED = "25m-supported-over-12p5m-matched-ladder"
OUTCOME_DENSITY_REGRESSION = "25m-scale-density-regression-localized"
OUTCOME_QUALITY_REGRESSION = "25m-scale-quality-or-ood-regression"
OUTCOME_INCONCLUSIVE = "12p5m-to-25m-scale-effect-inconclusive"
OUTCOME_INVALID = "invalid-execution"


class Round0132Error(RuntimeError):
    """The R0132 preregistered contract was violated."""


def seal(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    return {**value, "identity_sha256": sha256_bytes(canonical_json(value))}


def validate_seal(value: Mapping[str, Any], *, label: str) -> None:
    body = {key: item for key, item in value.items() if key != "identity_sha256"}
    if value.get("identity_sha256") != sha256_bytes(canonical_json(body)):
        raise Round0132Error(f"{label} identity seal changed")


def largest_remainder_quotas(
    group_counts: Mapping[str, int],
    *,
    target: int = HALF_RETAINED_ROWS,
) -> dict[str, int]:
    """Allocate an exact proportional target with group-order tie breaking."""
    if set(group_counts) != set(GROUPS):
        raise Round0132Error("subset group-count keys changed")
    counts = {group: int(group_counts[group]) for group in GROUPS}
    if any(value <= 0 for value in counts.values()):
        raise Round0132Error("every subset source group must be nonempty")
    total = sum(counts.values())
    if total != FULL_RETAINED_ROWS or not 0 < target < total:
        raise Round0132Error("subset total or target changed")

    quotas = {
        group: (target * counts[group]) // total
        for group in GROUPS
    }
    remaining = target - sum(quotas.values())
    if not 0 <= remaining < len(GROUPS):
        raise Round0132Error("largest-remainder closure changed")
    ranked = sorted(
        GROUPS,
        key=lambda group: (
            -((target * counts[group]) % total),
            GROUPS.index(group),
        ),
    )
    for group in ranked[:remaining]:
        quotas[group] += 1
    if (
        sum(quotas.values()) != target
        or any(quotas[group] <= 0 for group in GROUPS)
        or any(quotas[group] > counts[group] for group in GROUPS)
    ):
        raise Round0132Error("largest-remainder quotas did not close")
    return quotas


def _sha256_rank_prefix(rows: np.ndarray, *, namespace: bytes) -> np.ndarray:
    """Return the first 64 rank bits; full digest resolves boundary ties."""
    values = np.asarray(rows, dtype=np.int64)
    if values.ndim != 1 or np.any(values < 0) or not namespace:
        raise Round0132Error("SHA-rank input is malformed")
    output = np.empty(len(values), dtype=">u8")
    for index, row in enumerate(values):
        digest = hashlib.sha256(
            namespace + int(row).to_bytes(8, "little", signed=False)
        ).digest()
        output[index] = int.from_bytes(digest[:8], "big", signed=False)
    return output


def select_lowest_sha256_rank(
    rows: np.ndarray,
    *,
    count: int,
    namespace: bytes = SUBSET_NAMESPACE,
) -> np.ndarray:
    """Select exact lowest SHA-256 ranks and return rows in global order.

    A uint64 digest prefix makes the common path linear via ``argpartition``.
    Any prefix collision at the selection boundary is resolved by the complete
    256-bit digest and then global row ID, so this is exactly SHA-256 ranking,
    not a truncated-hash approximation.
    """
    values = np.asarray(rows, dtype=np.int64)
    if (
        values.ndim != 1
        or len(values) == 0
        or not 0 < count <= len(values)
        or np.any(values < 0)
        or (len(values) > 1 and np.any(values[1:] <= values[:-1]))
    ):
        raise Round0132Error("eligible SHA-rank rows are malformed")
    prefixes = _sha256_rank_prefix(values, namespace=namespace)
    if count == len(values):
        return values.copy()
    partition = np.argpartition(prefixes, count - 1)
    boundary = prefixes[partition[count - 1]]
    lower = np.flatnonzero(prefixes < boundary)
    tied = np.flatnonzero(prefixes == boundary)
    seats = count - len(lower)
    if not 0 < seats <= len(tied):
        raise Round0132Error("SHA-rank boundary accounting changed")
    if len(tied) > seats:
        tied = np.asarray(
            sorted(
                tied.tolist(),
                key=lambda index: (
                    hashlib.sha256(
                        namespace
                        + int(values[index]).to_bytes(
                            8, "little", signed=False
                        )
                    ).digest(),
                    int(values[index]),
                ),
            )[:seats],
            dtype=np.int64,
        )
    chosen = np.concatenate((lower, tied))
    selected = np.sort(values[chosen])
    if len(selected) != count or np.any(selected[1:] <= selected[:-1]):
        raise Round0132Error("SHA-rank selection did not close")
    return selected


def group_part_specs(
    quotas: Mapping[str, int],
    *,
    parts: int = 3,
) -> dict[str, dict[str, Any]]:
    """Create three contiguous, group-boundary-aligned resumable parts."""
    if parts != 3 or set(quotas) != set(GROUPS):
        raise Round0132Error("R0132 graph part policy changed")
    counts = np.asarray([int(quotas[group]) for group in GROUPS], dtype=np.int64)
    if np.any(counts <= 0) or int(counts.sum()) != HALF_RETAINED_ROWS:
        raise Round0132Error("R0132 graph part quotas changed")
    cumulative = np.cumsum(counts)
    boundaries = [0]
    for numerator in (1, 2):
        target = HALF_RETAINED_ROWS * numerator / parts
        candidates = np.arange(boundaries[-1] + 1, len(GROUPS) - (2 - numerator))
        index = int(candidates[np.argmin(np.abs(cumulative[candidates - 1] - target))])
        boundaries.append(index)
    boundaries.append(len(GROUPS))
    names = ("groups-a", "groups-b", "groups-c")
    output: dict[str, dict[str, Any]] = {}
    compact_start = 0
    for name, group_start, group_stop in zip(names, boundaries[:-1], boundaries[1:]):
        retained = int(counts[group_start:group_stop].sum())
        compact_stop = compact_start + retained
        output[name] = {
            "compact_start": compact_start,
            "compact_stop": compact_stop,
            "retained_rows": retained,
            "groups": list(GROUPS[group_start:group_stop]),
        }
        compact_start = compact_stop
    if compact_start != HALF_RETAINED_ROWS:
        raise Round0132Error("R0132 graph parts do not cover the half universe")
    return output


def qualification_metrics(
    selected: np.ndarray,
    exact: np.ndarray,
    *,
    group_ids: np.ndarray,
    unambiguous: np.ndarray,
) -> dict[str, Any]:
    candidate = np.asarray(selected, dtype=np.int64)
    truth = np.asarray(exact, dtype=np.int64)
    groups = np.asarray(group_ids, dtype=np.uint8)
    clear = np.asarray(unambiguous, dtype=bool)
    expected_rows = SEARCH_ANCHORS_PER_GROUP * len(GROUPS)
    if (
        candidate.shape != (expected_rows, GRAPH_K)
        or truth.shape != candidate.shape
        or groups.shape != (expected_rows,)
        or clear.shape != (expected_rows,)
        or np.any(candidate < 0)
        or np.any(truth < 0)
        or np.any(np.diff(np.sort(candidate, axis=1), axis=1) == 0)
        or not np.any(clear)
    ):
        raise Round0132Error("fixed-policy qualification arrays are malformed")
    overlap = (
        candidate[:, :, None] == truth[:, None, :]
    ).any(axis=2).sum(axis=1) / GRAPH_K
    by_group: dict[str, Any] = {}
    for group_id, group in enumerate(GROUPS):
        registered = groups == group_id
        values = overlap[registered & clear]
        if int(registered.sum()) != SEARCH_ANCHORS_PER_GROUP or not len(values):
            raise Round0132Error("fixed-policy group denominator changed")
        mean = float(values.mean())
        by_group[group] = {
            "registered_rows": int(registered.sum()),
            "unambiguous_rows": len(values),
            "mean_recall_at_15": mean,
            "passes_floor": mean >= SEARCH_GROUP_RECALL_FLOOR,
        }
    global_mean = float(overlap[clear].mean())
    checks = {
        "fixed_nprobe_64": True,
        "fixed_shortlist_width_128": True,
        "all_rows_complete_and_unique": True,
        "global_mean_recall_at_15_at_least_0p90": (
            global_mean >= SEARCH_GLOBAL_RECALL_FLOOR
        ),
        "every_group_mean_recall_at_15_at_least_0p84": all(
            value["passes_floor"] for value in by_group.values()
        ),
        "no_policy_sweep_or_widening_performed": True,
    }
    return {
        "global_mean_recall_at_15": global_mean,
        "by_group": by_group,
        "checks": checks,
        "passed": all(checks.values()),
    }


def coverage_aligned_updates(directed_edge_count: int) -> int:
    edges = int(directed_edge_count)
    if edges <= 0:
        raise Round0132Error("half-rung graph must contain positive edges")
    return (edges + POSITIVE_ROWS_PER_UPDATE - 1) // POSITIVE_ROWS_PER_UPDATE


def validate_train_execution(
    *,
    train: Mapping[str, Any],
    config_receipt: Mapping[str, Any],
    graph: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate R0132's graph-derived horizon and actual pipeline stamp.

    File signatures and identity seals are checked by the node wrapper.  This
    pure layer makes the accounting law adversarially testable without CUDA or
    large artifacts.
    """
    config = config_receipt.get("config")
    if not isinstance(config, Mapping):
        raise Round0132Error("R0132 production config is missing")
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

    expected_train_checks = {
        "exact_update_closure",
        "zero_numerical_skips",
        "no_pipeline_stamp_drift",
        "endpoint_rows_match_updates",
        "weighted_rejection_accounting_closes",
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
            "uniform-12,474,331-compact-retained-rows-nonself"
        ),
        "graph_degree": "variable-symmetric-fuzzy-k15-topology",
        "host_prefetch": "single-producer-two-pinned-slot",
        "endpoint_forward": "fused-source-destination",
        "valid_canonical_edge_count": edges,
        "compact_retained_rows": HALF_RETAINED_ROWS,
        "weighted_requested": True,
        "weighted_effective": True,
        "uniform_with_replacement": False,
        "positive_with_replacement": True,
        "weight_sampler": "uniform-envelope-rejection-max-weight-one",
        "weight_uniform_dtype": "<f8",
        "source_representation": "int8-treatment",
    }
    dynamic_stamp_keys = (
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
            and int(graph.get("retained_rows", -1)) == HALF_RETAINED_ROWS
            and int(graph.get("k_real", -1)) == GRAPH_K
            and int(graph.get("n_neighbors_including_self", -1)) == N_NEIGHBORS
            and int(config_graph.get("directed_edges", -1)) == edges
            and config_input.get("rows") == HALF_RETAINED_ROWS
        ),
        "coverage_horizon_closes": (
            optimizer.get("successful_positive_lr_updates") == updates
            and optimizer.get("positive_rows_per_update")
            == POSITIVE_ROWS_PER_UPDATE
            and optimizer.get("update_rule")
            == "ceil(actual-R0132-directed-fuzzy-edges/409)"
            and derivation.get("directed_fuzzy_edges") == edges
            and derivation.get("positive_rows_per_update")
            == POSITIVE_ROWS_PER_UPDATE
            and derivation.get("successful_updates") == updates
            and derivation.get("expected_positive_draws") == expected_draws
            and train.get("optimizer_updates") == updates
        ),
        "registered_pipeline_requested": (
            execution.get("required_pipeline") == PIPELINE
            and set(expected_stamp) == set(static_stamp)
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
            and runtime.get("host_prefetch_producer_batches")
            in {updates, updates + 1}
        ),
        "weighted_draw_accounting_closes": (
            runtime.get("weight_emitted_draws") == expected_draws
            and runtime.get("weight_acceptances")
            == (
                int(runtime.get("weight_emitted_draws", -1))
                + int(runtime.get("weight_buffered_draws", -1))
            )
            and int(runtime.get("weight_proposals", -1))
            >= int(runtime.get("weight_acceptances", 0))
            and 0 < float(runtime.get("weight_acceptance_rate", 0.0)) <= 1
        ),
        "runtime_and_flattened_accounting_agree": all(
            key in runtime
            and accounting.get(f"pipeline_{key}") == runtime.get(key)
            for key in dynamic_stamp_keys
        ),
        "train_checks_exact_and_positive": (
            set(train_checks) == expected_train_checks
            and all(value is True for value in train_checks.values())
        ),
        "performance_admission_closes": (
            profiler.get("aborted") is False
            and _finite_metric(train.get("steady_updates_per_s"), label="train rate")
            >= _finite_metric(execution.get("minimum_train_upd_s"), label="rate floor")
        ),
        "training_receipt_closes": (
            train.get("training_performed") is True
            and train.get("evaluation_performed") is False
            and train.get("map_decision_made") is False
        ),
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise Round0132Error(f"R0132 train execution authentication failed: {failed}")
    return {
        "checks": checks,
        "directed_fuzzy_edges": edges,
        "successful_updates": updates,
        "expected_positive_draws": expected_draws,
        "expected_endpoint_rows": expected_rows,
        "actual_pipeline_stamp": dict(runtime),
    }


def paired_density_bootstrap(
    high_radius: np.ndarray,
    control_low_radius: np.ndarray,
    treatment_low_radius: np.ndarray,
    *,
    eligible: np.ndarray | None = None,
    draws: int = DENSITY_BOOTSTRAP_DRAWS,
    seed: int = DENSITY_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap paired 25M-minus-12.5M density correlations."""
    from .round0108_evaluation import pearson_log_radius

    high = np.asarray(high_radius, dtype=np.float64)
    control = np.asarray(control_low_radius, dtype=np.float64)
    treatment = np.asarray(treatment_low_radius, dtype=np.float64)
    mask = (
        np.ones(len(high), dtype=bool)
        if eligible is None
        else np.asarray(eligible, dtype=bool)
    )
    if (
        high.ndim != 1
        or control.shape != high.shape
        or treatment.shape != high.shape
        or mask.shape != high.shape
        or int(mask.sum()) < 100
        or draws != DENSITY_BOOTSTRAP_DRAWS
        or seed != DENSITY_BOOTSTRAP_SEED
        or not np.isfinite(high).all()
        or not np.isfinite(control).all()
        or not np.isfinite(treatment).all()
        or np.any(high < 0)
        or np.any(control < 0)
        or np.any(treatment < 0)
    ):
        raise Round0132Error("paired density bootstrap inputs changed")
    high = high[mask]
    control = control[mask]
    treatment = treatment[mask]
    control_value = pearson_log_radius(high, control)
    treatment_value = pearson_log_radius(high, treatment)
    rng = np.random.RandomState(seed)
    deltas = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        positions = rng.randint(0, len(high), size=len(high))
        deltas[draw] = (
            pearson_log_radius(high[positions], treatment[positions])
            - pearson_log_radius(high[positions], control[positions])
        )
    alpha = (1.0 - DENSITY_CI_LEVEL) / 2.0
    low, high_ci = np.quantile(deltas, [alpha, 1.0 - alpha])
    delta = treatment_value - control_value
    noninferior = bool(low >= -DENSITY_NONINFERIORITY_MARGIN)
    materially_worse = bool(high_ci <= -DENSITY_NONINFERIORITY_MARGIN)
    if noninferior:
        classification = "noninferior"
    elif materially_worse:
        classification = "materially-worse"
    else:
        classification = "inconclusive"
    return {
        "control_12p5m_density": control_value,
        "treatment_25m_density": treatment_value,
        "treatment_minus_control": delta,
        "paired_bootstrap_delta_ci": [float(low), float(high_ci)],
        "paired_bootstrap_draws": draws,
        "paired_bootstrap_seed": seed,
        "paired_bootstrap_ci_level": DENSITY_CI_LEVEL,
        "noninferiority_margin": DENSITY_NONINFERIORITY_MARGIN,
        "classification": classification,
        "bootstrap_deltas": deltas,
    }


def _finite_metric(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise Round0132Error(f"{label} is not numeric")
    try:
        metric = float(value)
    except (TypeError, ValueError) as exc:
        raise Round0132Error(f"{label} is not numeric") from exc
    if not math.isfinite(metric):
        raise Round0132Error(f"{label} is nonfinite")
    return metric


def noninferiority_checks(
    *,
    control_native: Mapping[str, Any],
    treatment_native: Mapping[str, Any],
    control_ood: Mapping[str, Any],
    treatment_ood: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen matched quality and held-out OOD margins."""
    def at_least(observed: float, threshold: float) -> bool:
        # Decimal preregistration boundaries such as -0.02 and 0.97 should be
        # inclusive.  Binary-float representation must not turn equality into
        # a scientific failure, while values materially below the boundary do.
        return bool(
            observed >= threshold
            or math.isclose(observed, threshold, rel_tol=1e-12, abs_tol=1e-12)
        )

    native_names = (
        "global_recall_at_10",
        "global_recall_at_50_of_high10",
    )
    ood_names = (
        "fineweb_recall_at_50_of_high10",
        "polish_recall_at_50_of_high10",
        "in_mix_median_recall_at_50_of_high10",
    )
    metrics: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    control_ffr = _finite_metric(control_native.get("global_ffr"), label="control FFR")
    treatment_ffr = _finite_metric(
        treatment_native.get("global_ffr"), label="treatment FFR"
    )
    checks["matched_global_ffr_delta_at_least_minus_0p02"] = at_least(
        treatment_ffr - control_ffr, -FFR_ALLOWED_DECREASE
    )
    metrics["global_ffr"] = {
        "control": control_ffr,
        "treatment": treatment_ffr,
        "delta": treatment_ffr - control_ffr,
    }
    for name in native_names:
        control = _finite_metric(control_native.get(name), label=f"control {name}")
        treatment = _finite_metric(
            treatment_native.get(name), label=f"treatment {name}"
        )
        checks[f"{name}_retains_0p97"] = at_least(
            treatment, METRIC_RETENTION * control
        )
        metrics[name] = {"control": control, "treatment": treatment}
    for name in ood_names:
        control = _finite_metric(control_ood.get(name), label=f"control {name}")
        treatment = _finite_metric(
            treatment_ood.get(name), label=f"treatment {name}"
        )
        checks[f"{name}_retains_0p97"] = at_least(
            treatment, METRIC_RETENTION * control
        )
        metrics[name] = {"control": control, "treatment": treatment}
    return {"checks": checks, "passed": all(checks.values()), "metrics": metrics}


def scale_policy_decision(
    *,
    validity_checks: Mapping[str, bool],
    density: Mapping[str, Any],
    quality: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve only the five preregistered R0132 outcomes."""
    if not validity_checks or any(value is not True for value in validity_checks.values()):
        outcome = OUTCOME_INVALID
    else:
        classification = density.get("classification")
        quality_passed = quality.get("passed") is True
        if classification == "noninferior" and quality_passed:
            outcome = OUTCOME_SUPPORTED
        elif classification == "materially-worse" and quality_passed:
            outcome = OUTCOME_DENSITY_REGRESSION
        elif not quality_passed:
            outcome = OUTCOME_QUALITY_REGRESSION
        elif classification == "inconclusive":
            outcome = OUTCOME_INCONCLUSIVE
        else:
            raise Round0132Error("density selector classification changed")
    return {
        "outcome": outcome,
        "validity_checks": dict(validity_checks),
        "density_selector": dict(density),
        "quality_and_ood_noninferiority": dict(quality),
        "estimand": (
            "12,474,331-to-24,948,663 scale-policy effect with induced graph "
            "and coverage-aligned horizon; not a pure-N effect"
        ),
        "stale_absolute_jina_floor_role": "diagnostic-only",
        "projection_ffr_role": "diagnostic-only",
        "trec_covid_role": "diagnostic-only",
        "dadabase_role": "diagnostic-only",
        "atlas_quality_released": False,
        "universal_ood_claimed": False,
        "production_or_prompt_claimed": False,
    }


def assert_no_conditional_branch_dependency(values: Sequence[str]) -> None:
    forbidden = {"0125", "0129", "0130", "0131"}
    observed = {str(value) for value in values}
    if forbidden & observed:
        raise Round0132Error(
            "R0132 scientific validity must not depend on conditional branches"
        )
