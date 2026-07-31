#!/usr/bin/env python3
"""Prepare, but never launch, the conditional R0130 k49 rescue queue."""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from collections.abc import Mapping
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0105_search import ELIGIBILITY_PATH
from basemap.round0106_graph import PARTS
from basemap.round0104_training import validate_substrate_manifest
from basemap.round0130_k49_rescue import (
    ATLAS_QUALITY_CAPABILITY,
    DEGREE_RESCUE_CAPABILITY,
    DENSITY_BOOTSTRAP_DRAWS,
    DENSITY_BOOTSTRAP_SEED,
    DENSITY_MATERIAL_DELTA,
    FIXED_SUCCESSFUL_UPDATES,
    GRAPH_K,
    HEADLINE_KPI_RETENTION,
    ROUND_ID,
)
from basemap.round0124_degree_bridge import (
    BOOTSTRAP_CI_LEVEL,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    DECISION_SCHEMA as R0124_DECISION_SCHEMA,
    GRAPH_DEGREE as R0124_GRAPH_DEGREE,
    GRAPH_SEARCH_NEIGHBORS as R0124_GRAPH_SEARCH_NEIGHBORS,
    MATERIAL_DENSITY_DEGRADATION,
    OUTCOME_INCONCLUSIVE as R0124_INCONCLUSIVE_OUTCOME,
    read_sealed as read_r0124_sealed,
)
from basemap.round0129_degree_replicate import (
    CAPABILITY as R0129_CAPABILITY,
    DECISION_SCHEMA as R0129_DECISION_SCHEMA,
    NATIVE_DENSITY_SCHEMA as R0129_DENSITY_SCHEMA,
    PRODUCTION_CONFIG_SCHEMA as R0129_PRODUCTION_CONFIG_SCHEMA,
    SEED43_INITIAL_STATE_SHA256,
    SEED43_PARAMETER_COUNT,
    TRAIN_RECEIPT_SCHEMA as R0129_TRAIN_RECEIPT_SCHEMA,
    TRAINING_SEED as R0129_TRAINING_SEED,
)
from basemap.round0113_prompt_contrast import (
    RETAINED_ROWS as R0129_RETAINED_ROWS,
    SUCCESSFUL_UPDATES as R0129_SUCCESSFUL_UPDATES,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)
from experiments.prepare_round0106_queue import (
    DECISION,
    INDEX,
    INDEX_RECEIPT,
    QUALIFICATION,
)
from experiments.prepare_round0108_queue import (
    LABELS_PATH,
    R0040_CENSUS_RECEIPT,
    R0040_REFERENCE,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0130"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0130-*.md")
R0105_TRUTH = os.path.join(
    os.path.dirname(QUALIFICATION), "exact-truth.npz"
)
R0107_TRAIN_OUTPUT = (
    "/data/latent-basemap/runs/round-0107/queue/artifacts/"
    "train-diverse-jina-25m"
)
R0107_PRODUCTION_CONFIG = os.path.join(
    R0107_TRAIN_OUTPUT, "production-config.json"
)
R0107_TRAIN_RECEIPT = os.path.join(R0107_TRAIN_OUTPUT, "train-receipt.json")
R0108_ROOT = "/data/latent-basemap/runs/round-0108/queue-attempt-3"
R0108_QUEUE = os.path.join(R0108_ROOT, "queue.json")
R0108_TERMINAL = os.path.join(R0108_ROOT, "runner-terminal.json")
R0108_SELECTION = os.path.join(R0108_ROOT, "inputs", "registered-selections.npz")
R0108_CALIBRATION = os.path.join(
    R0108_ROOT, "artifacts", "jina-density-calibration"
)
R0108_CALIBRATION_RECEIPT = os.path.join(
    R0108_CALIBRATION, "jina-density-calibration.json"
)
R0108_CORE = os.path.join(R0108_ROOT, "artifacts", "core-geometry")
R0108_OOD = os.path.join(R0108_ROOT, "artifacts", "ood")
R0124_ROOT = "/data/latent-basemap/runs/round-0124/queue-attempt-2"
R0124_QUEUE = os.path.join(R0124_ROOT, "queue.json")
R0124_TERMINAL = os.path.join(R0124_ROOT, "runner-terminal.json")
R0124_DECISION = os.path.join(
    R0124_ROOT,
    "artifacts",
    "degree-bridge-decision",
    "decision.json",
)
R0129_ROOT = "/data/latent-basemap/runs/round-0129/queue"
R0129_QUEUE = os.path.join(R0129_ROOT, "queue.json")
R0129_TERMINAL = os.path.join(R0129_ROOT, "runner-terminal.json")
R0129_DECISION = os.path.join(
    R0129_ROOT,
    "artifacts",
    "degree-replicate-decision",
    "decision.json",
)

GPU_HOURS_CAP = 8.0
P90_QUALIFICATION_SECONDS = 900.0
P90_GRAPH_PART_SECONDS = 1_200.0
P90_TRAIN_SECONDS = 15_000.0
P90_TRANSFORM_SECONDS = 900.0
P90_CORE_SECONDS = 900.0
P90_OOD_SECONDS = 1_800.0
P90_MATCHED_SECONDS = 600.0
P90_GPU_TOTAL_SECONDS = (
    P90_QUALIFICATION_SECONDS
    + len(PARTS) * P90_GRAPH_PART_SECONDS
    + P90_TRAIN_SECONDS
    + P90_TRANSFORM_SECONDS
    + P90_CORE_SECONDS
    + P90_OOD_SECONDS
    + P90_MATCHED_SECONDS
)

REVIEW_DEFAULTS = {
    "0105": (
        "review-0105-2026-07-29.md",
        "084722e2641667333a673a8d9473da11d0aee1f97bca59e2ed646499e4169b96",
        "capability:jina-diverse-25m-full768-search-qualified-v1",
    ),
    "0106": (
        "review-0106-2026-07-29.md",
        "f00a8391cc47f038993b40337cbe71e07536d305015597ea2e39eed9ca116e1f",
        "capability:jina-diverse-25m-full768-fuzzy-graph-v1",
    ),
    "0107": (
        "review-0107-2026-07-30.md",
        "efac370df53f11cd50a3aad4fe8e18c9683bc84faa34ba783f5a342fc00a17ba",
        "capability:jina-diverse-25m-full768-trained-map-seed42-v1",
    ),
    "0108": (
        "review-0108-2026-07-30.md",
        "5ad9fbcf9307552862cff32ae7a86b771cb88f395c71b19f8f9c5b486dc476ee",
        "capability:jina-diverse-25m-map-registry-v1",
    ),
}


def _frontmatter(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if not text.startswith("---\n"):
        raise RuntimeError(f"missing frontmatter: {path}")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise RuntimeError(f"unterminated frontmatter: {path}")
    result: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip().strip("\"'")
    return result


def _require_issued_round() -> str:
    candidates = [
        path
        for path in sorted(glob.glob(ROUND_FILE_GLOB))
        if _frontmatter(path).get("status") == "issued"
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"R0130 requires exactly one issued round; found {len(candidates)}"
        )
    return candidates[0]


def _require_review(
    path: str,
    *,
    expected_sha256: str,
    round_id: str,
    required_text: str,
) -> dict[str, Any]:
    signature = expected_input_signature(path)
    frontmatter = _frontmatter(path)
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if (
        signature["sha256"] != expected_sha256
        or frontmatter.get("round_id") != round_id
        or frontmatter.get("status") != "accepted"
        or required_text not in text
    ):
        raise RuntimeError(f"Review {round_id} is not the required acceptance")
    return signature


def _document_text(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _require_accepted_execution(
    review_path: str,
    *,
    expected_sha256: str,
    round_id: str,
    capability: str,
    queue_path: str,
    terminal_path: str,
    decision_path: str,
    queue_schema: str,
    decision_schema: str,
    decision_job_id: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Authenticate review -> result -> queue -> terminal -> sealed decision."""
    review_signature = expected_input_signature(review_path)
    review_frontmatter = _frontmatter(review_path)
    try:
        review_releases = json.loads(review_frontmatter.get("releases") or "[]")
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Review {round_id} releases are malformed") from exc
    if (
        review_signature["sha256"] != expected_sha256
        or review_frontmatter.get("round_id") != round_id
        or review_frontmatter.get("status") != "accepted"
        or not isinstance(review_releases, list)
        or not all(isinstance(value, str) for value in review_releases)
        or f"capability:{capability}" not in review_releases
    ):
        raise RuntimeError(
            f"Review {round_id} is not accepted at the expected bytes"
        )
    result_name = review_frontmatter.get("result") or ""
    result_sha256 = review_frontmatter.get("result_sha256") or ""
    if (
        not result_name
        or os.path.basename(result_name) != result_name
        or not re.fullmatch(
            rf"result-{round_id}-[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}\.md",
            result_name,
        )
        or not re.fullmatch(r"[0-9a-f]{64}", result_sha256)
    ):
        raise RuntimeError(f"Review {round_id} does not bind one result")
    result_path = os.path.join(os.path.dirname(review_path), result_name)
    result_signature = expected_input_signature(result_path)
    result_frontmatter = _frontmatter(result_path)
    review_release = review_frontmatter.get("verified_release_commit")
    result_release = result_frontmatter.get("release_commit")
    result_queue = (result_frontmatter.get("queue_manifest") or "").removeprefix(
        "gsv:"
    )
    if (
        result_signature["sha256"] != result_sha256
        or result_frontmatter.get("round_id") != round_id
        or result_frontmatter.get("status") != "complete"
        or not re.fullmatch(r"[0-9a-f]{40}", result_release or "")
        or review_release != result_release
        or os.path.realpath(result_queue) != os.path.realpath(queue_path)
    ):
        raise RuntimeError(f"Accepted Review {round_id} result binding changed")

    queue_signature = expected_input_signature(queue_path)
    terminal_signature = expected_input_signature(terminal_path)
    decision_signature = expected_input_signature(decision_path)
    with open(queue_path, encoding="utf-8") as handle:
        queue = json.load(handle)
    with open(terminal_path, encoding="utf-8") as handle:
        terminal = json.load(handle)
    decision = read_r0124_sealed(decision_path, label=f"R{round_id} decision")
    jobs = queue.get("jobs") or []
    required_jobs = [str(job.get("id") or "") for job in jobs]
    decision_jobs = [
        job for job in jobs if job.get("id") == decision_job_id
    ]
    release = str(queue.get("release_sha") or "")
    terminal_nodes = terminal.get("nodes") or []
    terminal_required = terminal.get("required_jobs")
    terminal_completed = terminal.get("completed_jobs")
    terminal_node_ids = [str(node.get("node") or "") for node in terminal_nodes]
    result_text = _document_text(result_path)
    if (
        queue.get("schema") != queue_schema
        or queue.get("round_id") != round_id
        or capability not in (queue.get("capabilities_produced") or [])
        or release != result_release
        or not required_jobs
        or any(not value for value in required_jobs)
        or len(set(required_jobs)) != len(required_jobs)
        or len(decision_jobs) != 1
        or decision_jobs[0].get("outputs")
        != [os.path.dirname(os.path.realpath(decision_path))]
        or terminal.get("schema") != "slim-runner-terminal-v3"
        or terminal.get("round_id") != round_id
        or terminal.get("verdict") != "succeeded"
        or terminal_required != required_jobs
        or not isinstance(terminal_completed, list)
        or sorted(terminal_completed) != sorted(required_jobs)
        or len(terminal_completed) != len(required_jobs)
        or sorted(terminal_node_ids) != sorted(required_jobs)
        or len(terminal_node_ids) != len(required_jobs)
        or terminal.get("queue_manifest_sha256") != queue_signature["sha256"]
        or terminal.get("queue_manifest_sha256_at_finish")
        != queue_signature["sha256"]
        or terminal.get("queue_manifest_unchanged") is not True
        or terminal.get("release_checkout_unchanged") is not True
        or terminal.get("gpu_wall_accounting_complete") is not True
        or terminal.get("boundary_problems") != []
        or any(node.get("validation_problems") != [] for node in terminal_nodes)
        or (terminal.get("release_checkout") or {}).get("head") != release
        or (terminal.get("release_checkout_at_finish") or {}).get("head")
        != release
        or decision.get("schema") != decision_schema
        or decision.get("round_id") != round_id
        or decision.get("release_sha") != release
        or decision.get("capabilities_produced") != [capability]
        or result_frontmatter.get("queue_manifest_sha256")
        != queue_signature["sha256"]
        or decision_signature["sha256"] not in result_text
        or queue_signature["sha256"] not in result_text
        or terminal_signature["sha256"] not in result_text
    ):
        raise RuntimeError(f"R{round_id} execution/result linkage changed")
    signatures = {
        "review": review_signature,
        "result": result_signature,
        "queue": queue_signature,
        "terminal": terminal_signature,
        "decision": decision_signature,
    }
    return signatures, queue, decision


def _finite_selector(decision: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    selector = decision.get("registered_selector")
    if not isinstance(selector, Mapping):
        raise RuntimeError(f"{label} registered selector is missing")
    interval = selector.get("paired_bootstrap_delta_ci")
    numeric = (
        selector.get("control_density"),
        selector.get("treatment_density"),
        selector.get("treatment_minus_control"),
    )
    if (
        not isinstance(interval, list)
        or len(interval) != 2
        or any(isinstance(value, bool) for value in (*numeric, *interval))
        or not all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in (*numeric, *interval)
        )
        or float(interval[0]) > float(interval[1])
        or not math.isclose(
            float(selector["treatment_density"])
            - float(selector["control_density"]),
            float(selector["treatment_minus_control"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError(f"{label} selector arithmetic is malformed")
    return dict(selector)


def _linked_sealed(
    signature: Any,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(signature, Mapping):
        raise RuntimeError(f"{label} signature is missing")
    path = str(signature.get("canonical_path") or "")
    if expected_input_signature(path) != dict(signature):
        raise RuntimeError(f"{label} signature changed")
    return read_r0124_sealed(path, label=label)


def _linked_json(signature: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(signature, Mapping):
        raise RuntimeError(f"{label} signature is missing")
    path = str(signature.get("canonical_path") or "")
    if expected_input_signature(path) != dict(signature):
        raise RuntimeError(f"{label} signature changed")
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return value


def _require_inconclusive_r0124_review(
    review_path: str,
    *,
    expected_sha256: str,
    queue_path: str = R0124_QUEUE,
    terminal_path: str = R0124_TERMINAL,
    decision_path: str = R0124_DECISION,
) -> dict[str, Any]:
    """Require the exact seed-42 inconclusive-but-negative-direction result."""
    signatures, queue, decision = _require_accepted_execution(
        review_path,
        expected_sha256=expected_sha256,
        round_id="0124",
        capability="jina-fineweb-2m-native-k15-degree-bridge-v1",
        queue_path=queue_path,
        terminal_path=terminal_path,
        decision_path=decision_path,
        queue_schema="round0124-fineweb-2m-degree-bridge-retry-queue-v1",
        decision_schema=R0124_DECISION_SCHEMA,
        decision_job_id="decide_degree_bridge",
    )
    if decision.get("retry_provenance") != queue.get("retry_provenance"):
        raise RuntimeError("R0124 retry provenance changed")
    selector = _finite_selector(decision, label="R0124")
    interval = selector["paired_bootstrap_delta_ci"]
    if (
        selector.get("outcome") != R0124_INCONCLUSIVE_OUTCOME
        or float(selector["treatment_minus_control"])
        > -MATERIAL_DENSITY_DEGRADATION
        or float(interval[1]) >= 0.0
        or float(interval[0]) > -MATERIAL_DENSITY_DEGRADATION
        or float(interval[1]) <= -MATERIAL_DENSITY_DEGRADATION
    ):
        raise RuntimeError("R0124 is not the registered negative-direction inconclusive")
    density = _linked_sealed(decision.get("density_score"), label="R0124 density")
    if (
        density.get("round_id") != "0124"
        or density.get("release_sha") != decision.get("release_sha")
        or density.get("registered_selector") != selector
        or density.get("changed_factor") != "fuzzy graph neighbor degree only"
    ):
        raise RuntimeError("R0124 density linkage or intervention changed")
    return {
        "signatures": signatures,
        "selector": selector,
        "native_reference": density.get("native_reference"),
        "scientific_contract": queue.get("scientific_contract"),
        "density_score": expected_input_signature(
            str((decision.get("density_score") or {})["canonical_path"])
        ),
    }


def _require_positive_r0129_review(
    review_path: str,
    *,
    expected_sha256: str,
    expected_r0124_decision: Mapping[str, Any],
    queue_path: str = R0129_QUEUE,
    terminal_path: str = R0129_TERMINAL,
    decision_path: str = R0129_DECISION,
) -> dict[str, Any]:
    """Require the exact clean seed-43 positive result and execution closure."""
    signatures, queue, decision = _require_accepted_execution(
        review_path,
        expected_sha256=expected_sha256,
        round_id="0129",
        capability=R0129_CAPABILITY,
        queue_path=queue_path,
        terminal_path=terminal_path,
        decision_path=decision_path,
        queue_schema="round0129-seed43-native-degree-replicate-queue-v1",
        decision_schema=R0129_DECISION_SCHEMA,
        decision_job_id="decide_degree_replicate",
    )
    selector = _finite_selector(decision, label="R0129")
    interval = selector["paired_bootstrap_delta_ci"]
    if (
        selector.get("outcome") != "k15-materially-degrades-native-density"
        or float(interval[1]) > -MATERIAL_DENSITY_DEGRADATION
        or round(float(selector["control_density"]), 4) != 0.2116
        or decision.get("training_seed") != R0129_TRAINING_SEED
        or decision.get("optimizer_updates") != R0129_SUCCESSFUL_UPDATES
        or decision.get("diagnostics_can_rescue_or_fail_selector") is not False
        or decision.get("r0124_inconclusive_trigger")
        != dict(expected_r0124_decision)
        or (queue.get("conditional_trigger") or {}).get("decision")
        != dict(expected_r0124_decision)
    ):
        raise RuntimeError("R0129 is not the exact positive seed-43 replicate")
    density = _linked_sealed(decision.get("density_score"), label="R0129 density")
    train = _linked_sealed(decision.get("train_receipt"), label="R0129 train")
    production = _linked_json(
        train.get("production_config"), label="R0129 production config"
    )
    isolation = decision.get("config_equivalence")
    initial_state = decision.get("initial_model_state")
    config = production.get("config")
    accounting = train.get("train_accounting")
    checks = train.get("train_checks")
    if (
        density.get("round_id") != "0129"
        or density.get("schema") != R0129_DENSITY_SCHEMA
        or density.get("release_sha") != decision.get("release_sha")
        or density.get("registered_selector") != selector
        or density.get("changed_factor") != "fuzzy graph neighbor degree only"
        or density.get("non_graph_config_equal") is not True
        or density.get("sampling_mechanism_equal_conditioned_on_graph") is not True
        or density.get("positive_edge_distribution_equal") is not False
        or not isinstance(isolation, Mapping)
        or train.get("schema") != R0129_TRAIN_RECEIPT_SCHEMA
        or train.get("round_id") != "0129"
        or train.get("release_sha") != decision.get("release_sha")
        or train.get("config_equivalence") != isolation
        or production.get("schema") != R0129_PRODUCTION_CONFIG_SCHEMA
        or production.get("round_id") != "0129"
        or production.get("config_equivalence") != isolation
        or isolation.get("non_graph_config_equal") is not True
        or isolation.get("non_graph_execution_equal") is not True
        or isolation.get("exact_equal_sections")
        != ["arm", "input", "model", "optimizer"]
        or isolation.get("graph_sampler_policy_fields_equal") is not True
        or isolation.get("sampling_mechanism_equal_conditioned_on_graph") is not True
        or isolation.get("positive_edge_distribution_equal") is not False
        or isolation.get("negative_sampling_distribution_equal") is not True
        or isolation.get("identical_realized_edge_draws_claimed") is not False
        or isolation.get("identical_realized_negative_pairs_claimed") is not False
        or isolation.get("training_seed") != R0129_TRAINING_SEED
        or isolation.get("successful_updates") != R0129_SUCCESSFUL_UPDATES
        or not isinstance(initial_state, Mapping)
        or train.get("initial_model_state") != initial_state
        or initial_state.get("observed_sha256") != SEED43_INITIAL_STATE_SHA256
        or initial_state.get("parameter_count") != SEED43_PARAMETER_COUNT
        or initial_state.get(
            "captured_before_optimizer_construction_and_update_zero"
        )
        is not True
        or initial_state.get("actual_historical_r0117_bytes_claimed") is not False
        or initial_state.get("historical_evidence_kind")
        != "deterministic-reconstruction-not-original-reviewed-receipt"
        or not isinstance(config, Mapping)
        or (config.get("input") or {}).get("rows") != R0129_RETAINED_ROWS
        or (config.get("optimizer") or {}).get("seed") != R0129_TRAINING_SEED
        or (config.get("optimizer") or {}).get("successful_positive_lr_updates")
        != R0129_SUCCESSFUL_UPDATES
        or not isinstance(accounting, Mapping)
        or accounting.get("optimizer_steps_succeeded") != R0129_SUCCESSFUL_UPDATES
        or accounting.get("positive_lr_optimizer_steps") != R0129_SUCCESSFUL_UPDATES
        or accounting.get("amp_overflow_skips") != 0
        or accounting.get("nonfinite_loss_skips") != 0
        or accounting.get("nonfinite_gradient_skips") != 0
        or not isinstance(checks, Mapping)
        or not checks
        or any(value is not True for value in checks.values())
    ):
        raise RuntimeError("R0129 state/config/population closure changed")
    return {
        "signatures": signatures,
        "selector": selector,
        "native_reference": density.get("native_reference"),
        "scientific_contract": queue.get("scientific_contract"),
        "train_receipt": expected_input_signature(
            str((decision.get("train_receipt") or {})["canonical_path"])
        ),
        "production_config": expected_input_signature(
            str((train.get("production_config") or {})["canonical_path"])
        ),
        "density_score": expected_input_signature(
            str((decision.get("density_score") or {})["canonical_path"])
        ),
    }


def _assert_two_seed_contract_equal(
    r0124: Mapping[str, Any],
    r0129: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove the two seeds use one intervention, panel, and -0.03 selector."""
    left = r0124.get("selector")
    right = r0129.get("selector")
    left_reference = r0124.get("native_reference")
    right_reference = r0129.get("native_reference")
    left_contract = r0124.get("scientific_contract")
    right_contract = r0129.get("scientific_contract")
    equal_selector_fields = (
        "paired_bootstrap_ci_level",
        "paired_bootstrap_draws",
        "paired_bootstrap_seed",
        "material_degradation_threshold",
        "selector_metrics",
        "core_and_ood_diagnostics_can_rescue_or_fail",
        "legacy_density_floor_used",
        "single_cause_beyond_graph_degree_claimed",
    )
    if (
        not all(isinstance(value, Mapping) for value in (
            left, right, left_reference, right_reference, left_contract, right_contract
        ))
        or any(left.get(key) != right.get(key) for key in equal_selector_fields)
        or float(left["material_degradation_threshold"])
        != -MATERIAL_DENSITY_DEGRADATION
        or left.get("paired_bootstrap_ci_level") != BOOTSTRAP_CI_LEVEL
        or left.get("paired_bootstrap_draws") != BOOTSTRAP_DRAWS
        or left.get("paired_bootstrap_seed") != BOOTSTRAP_SEED
        or any(
            left_reference.get(key) != right_reference.get(key)
            for key in (
                "high_d_reference",
                "anchor_count",
                "anchor_seed",
                "k_density",
                "low_d_search",
            )
        )
        or left_reference.get("anchor_count") != 4_000
        or left_reference.get("anchor_seed") != 123
        or left_reference.get("k_density") != R0124_GRAPH_DEGREE
        or left_contract.get("changed_factor")
        != "fuzzy graph neighbor degree only"
        or right_contract.get("changed_factor")
        != "fuzzy graph neighbor degree only"
        or left_contract.get("population_rows") != R0129_RETAINED_ROWS
        or right_contract.get("population_rows") != R0129_RETAINED_ROWS
        or left_contract.get("graph_nonself_neighbors") != R0124_GRAPH_DEGREE
        or right_contract.get("graph_nonself_neighbors") != R0124_GRAPH_DEGREE
        or left_contract.get("graph_search_neighbors_including_self")
        != R0124_GRAPH_SEARCH_NEIGHBORS
        or right_contract.get("graph_search_neighbors_including_self")
        != R0124_GRAPH_SEARCH_NEIGHBORS
        or left_contract.get("successful_updates") != R0129_SUCCESSFUL_UPDATES
        or right_contract.get("successful_updates") != R0129_SUCCESSFUL_UPDATES
        or left_contract.get("paired_bootstrap")
        != right_contract.get("paired_bootstrap")
    ):
        raise RuntimeError("R0124/R0129 intervention, panel, or margin differs")
    return {
        "schema": "round0130-two-seed-degree-evidence-v1",
        "seed42_outcome": left["outcome"],
        "seed43_outcome": right["outcome"],
        "intervention_equal": True,
        "native_panel_equal": True,
        "material_margin_equal": True,
        "seeds_pooled": False,
        "degree_hypothesis_proven_claimed": False,
    }


def _require_clean_r0108() -> tuple[dict[str, Any], dict[str, Any]]:
    queue_signature = expected_input_signature(R0108_QUEUE)
    terminal_signature = expected_input_signature(R0108_TERMINAL)
    with open(R0108_QUEUE, encoding="utf-8") as handle:
        queue = json.load(handle)
    with open(R0108_TERMINAL, encoding="utf-8") as handle:
        terminal = json.load(handle)
    if (
        queue.get("round_id") != "0108"
        or terminal.get("round_id") != "0108"
        or terminal.get("verdict") != "succeeded"
        or terminal.get("completed_jobs") != terminal.get("required_jobs")
        or terminal.get("queue_manifest_unchanged") is not True
        or terminal.get("release_checkout_unchanged") is not True
    ):
        raise RuntimeError("R0108 is not a clean accepted execution")
    return queue_signature, terminal_signature


def _search_fields() -> dict[str, Any]:
    return {
        "release_sha": None,
        "index": INDEX,
        "index_sha256": expected_input_signature(INDEX)["sha256"],
        "index_receipt": INDEX_RECEIPT,
        "index_receipt_sha256": expected_input_signature(INDEX_RECEIPT)[
            "sha256"
        ],
        "qualification": QUALIFICATION,
        "qualification_sha256": expected_input_signature(QUALIFICATION)[
            "sha256"
        ],
        "decision": DECISION,
        "decision_sha256": expected_input_signature(DECISION)["sha256"],
    }


def _job(
    *,
    node_id: str,
    action: str,
    deps: list[str],
    output: str,
    expected_inputs: list[dict[str, Any]],
    p90_wall_s: float,
    gpu: bool,
    training: bool = False,
    **values: Any,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "action": action,
        "handler_module": "experiments.round0130_nodes",
        "handler_callable": "run_job",
        "deps": deps,
        "outputs": [output],
        "done_marker": os.path.join(
            os.path.dirname(output), f"{node_id}.done.json"
        ),
        "expected_inputs": _dedupe(expected_inputs),
        "p90_wall_s": p90_wall_s,
        "node_policy": {
            "gpu_required": gpu,
            "training_performed": training,
        },
        **values,
    }


def prepare_round0130(
    *,
    release_sha: str,
    r0124_review_path: str,
    r0124_review_sha256: str,
    r0129_review_path: str,
    r0129_review_sha256: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0130 release SHA must be one full commit")
    round_file = _require_issued_round()
    reviews = {
        round_id: _require_review(
            os.path.join(LAB_ROOT, name),
            expected_sha256=sha,
            round_id=round_id,
            required_text=capability,
        )
        for round_id, (name, sha, capability) in REVIEW_DEFAULTS.items()
    }
    r0124_evidence = _require_inconclusive_r0124_review(
        r0124_review_path,
        expected_sha256=r0124_review_sha256,
    )
    r0129_evidence = _require_positive_r0129_review(
        r0129_review_path,
        expected_sha256=r0129_review_sha256,
        expected_r0124_decision=r0124_evidence["signatures"]["decision"],
    )
    two_seed_proof = _assert_two_seed_contract_equal(
        r0124_evidence,
        r0129_evidence,
    )
    reviews["0124"] = r0124_evidence["signatures"]["review"]
    reviews["0129"] = r0129_evidence["signatures"]["review"]
    r0108_queue, r0108_terminal = _require_clean_r0108()
    queue_root = create_fresh_directory(
        queue_root, label="R0130 direct k49 degree-rescue queue"
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    common = _dedupe([
        expected_input_signature(round_file),
        *reviews.values(),
        *(
            value for key, value in r0124_evidence["signatures"].items()
            if key != "review"
        ),
        r0124_evidence["density_score"],
        *(
            value for key, value in r0129_evidence["signatures"].items()
            if key != "review"
        ),
        r0129_evidence["density_score"],
        r0129_evidence["train_receipt"],
        r0129_evidence["production_config"],
        r0108_queue,
        r0108_terminal,
    ])
    substrate = validate_substrate_manifest(verify_payloads=False)
    substrate_inputs = _dedupe([
        substrate["signature"],
        substrate["payloads"]["int8"],
        substrate["payloads"]["scales"],
        substrate["payloads"]["labels"],
    ])
    search = _search_fields()
    search["release_sha"] = release_sha
    search_inputs = _dedupe([
        *common,
        expected_input_signature(INDEX),
        expected_input_signature(INDEX_RECEIPT),
        expected_input_signature(QUALIFICATION),
        expected_input_signature(DECISION),
        expected_input_signature(R0105_TRUTH),
        expected_input_signature(ELIGIBILITY_PATH),
        *substrate_inputs,
    ])

    quality_output = os.path.join(artifacts, "k49-search-qualification")
    part_outputs = {
        part: os.path.join(artifacts, f"k49-graph-part-{part}")
        for part in PARTS
    }
    graph_output = os.path.join(artifacts, "k49-fuzzy-graph")
    graph_manifest = os.path.join(graph_output, "graph-manifest.json")
    train_output = os.path.join(artifacts, "train-k49-seed42")
    transform_output = os.path.join(artifacts, "coordinates")
    core_output = os.path.join(artifacts, "core-geometry")
    ood_output = os.path.join(artifacts, "ood")
    matched_output = os.path.join(artifacts, "matched-r0040-density")
    decision_output = os.path.join(artifacts, "decision")

    quality_job = _job(
        node_id="qualify_k49_selected_policy",
        action="qualify_k49_selected_policy",
        deps=[],
        output=quality_output,
        expected_inputs=search_inputs,
        p90_wall_s=P90_QUALIFICATION_SECONDS,
        gpu=True,
        r0105_truth=R0105_TRUTH,
        r0105_truth_sha256=expected_input_signature(R0105_TRUTH)["sha256"],
        **search,
    )
    graph_jobs = []
    for part in PARTS:
        graph_jobs.append(_job(
            node_id=f"build_k49_graph_part_{part}",
            action="build_graph_part",
            deps=["qualify_k49_selected_policy"],
            output=part_outputs[part],
            expected_inputs=search_inputs,
            p90_wall_s=P90_GRAPH_PART_SECONDS,
            gpu=True,
            part=part,
            quality_output=quality_output,
            **search,
        ))
    graph_job_ids = [str(job["id"]) for job in graph_jobs]
    assemble_job = _job(
        node_id="assemble_k49_graph",
        action="assemble_graph",
        deps=graph_job_ids,
        output=graph_output,
        expected_inputs=common,
        p90_wall_s=1_800.0,
        gpu=False,
        quality_output=quality_output,
        part_outputs=part_outputs,
    )
    r0107_inputs = [
        expected_input_signature(R0107_PRODUCTION_CONFIG),
        expected_input_signature(R0107_TRAIN_RECEIPT),
    ]
    train_job = _job(
        node_id="train_k49_treatment",
        action="train_k49_treatment",
        deps=["assemble_k49_graph"],
        output=train_output,
        expected_inputs=[*common, *r0107_inputs, *substrate_inputs],
        p90_wall_s=P90_TRAIN_SECONDS,
        gpu=True,
        training=True,
        release_sha=release_sha,
        graph_release_sha=release_sha,
        graph_manifest=graph_manifest,
        quality_output=quality_output,
        r0107_production_config=R0107_PRODUCTION_CONFIG,
        r0107_production_config_sha256=r0107_inputs[0]["sha256"],
        r0107_train_receipt=R0107_TRAIN_RECEIPT,
        r0107_train_receipt_sha256=r0107_inputs[1]["sha256"],
    )

    with open(R0108_QUEUE, encoding="utf-8") as handle:
        r0108_manifest = json.load(handle)
    source_jobs = {
        str(value["action"]): value for value in r0108_manifest["jobs"]
    }
    ood_source = source_jobs["score_ood"]
    eval_common = _dedupe([
        *common,
        expected_input_signature(R0108_SELECTION),
        expected_input_signature(LABELS_PATH),
        expected_input_signature(ELIGIBILITY_PATH),
        expected_input_signature(R0108_CALIBRATION_RECEIPT),
        *substrate_inputs,
    ])
    transform_job = _job(
        node_id="transform_retained_map",
        action="transform_retained_map",
        deps=["train_k49_treatment"],
        output=transform_output,
        expected_inputs=eval_common,
        p90_wall_s=P90_TRANSFORM_SECONDS,
        gpu=True,
        train_output=train_output,
        graph_manifest=graph_manifest,
        quality_output=quality_output,
    )
    core_job = _job(
        node_id="score_core_geometry",
        action="score_core_geometry",
        deps=["transform_retained_map"],
        output=core_output,
        expected_inputs=eval_common,
        p90_wall_s=P90_CORE_SECONDS,
        gpu=True,
        calibration_output=R0108_CALIBRATION,
        transform_output=transform_output,
        selection=R0108_SELECTION,
        train_output=train_output,
        graph_manifest=graph_manifest,
        part_outputs=part_outputs,
        eligibility=ELIGIBILITY_PATH,
        labels=LABELS_PATH,
        quality_output=quality_output,
    )
    ood_job = _job(
        node_id="score_ood",
        action="score_ood",
        deps=["transform_retained_map"],
        output=ood_output,
        expected_inputs=_dedupe([
            *eval_common,
            *ood_source["language_sources"].values(),
            *ood_source["diagnostic_sources"].values(),
        ]),
        p90_wall_s=P90_OOD_SECONDS,
        gpu=True,
        transform_output=transform_output,
        selection=R0108_SELECTION,
        train_output=train_output,
        graph_manifest=graph_manifest,
        language_sources=ood_source["language_sources"],
        language_training_stops=ood_source["language_training_stops"],
        diagnostic_sources=ood_source["diagnostic_sources"],
        embedding_prompt=ood_source["embedding_prompt"],
        quality_output=quality_output,
    )
    matched_job = _job(
        node_id="score_matched_r0040_density",
        action="score_matched_r0040_density",
        deps=["train_k49_treatment"],
        output=matched_output,
        expected_inputs=_dedupe([
            *common,
            expected_input_signature(R0040_CENSUS_RECEIPT),
            expected_input_signature(R0040_REFERENCE),
            expected_input_signature(R0108_CALIBRATION_RECEIPT),
        ]),
        p90_wall_s=P90_MATCHED_SECONDS,
        gpu=True,
        calibration_output=R0108_CALIBRATION,
        census_receipt=R0040_CENSUS_RECEIPT,
        census_receipt_sha256=expected_input_signature(
            R0040_CENSUS_RECEIPT
        )["sha256"],
        representative_reference=R0040_REFERENCE,
        representative_reference_sha256=expected_input_signature(
            R0040_REFERENCE
        )["sha256"],
        train_output=train_output,
        graph_manifest=graph_manifest,
        quality_output=quality_output,
    )
    control_core = os.path.join(R0108_CORE, "core-geometry.json")
    control_ood = os.path.join(R0108_OOD, "ood-evaluation.json")
    decision_job = _job(
        node_id="decide_k49_rescue",
        action="decide_k49_rescue",
        deps=[
            "score_core_geometry",
            "score_ood",
            "score_matched_r0040_density",
        ],
        output=decision_output,
        expected_inputs=_dedupe([
            *common,
            expected_input_signature(control_core),
            expected_input_signature(control_ood),
        ]),
        p90_wall_s=60.0,
        gpu=False,
        control_core=control_core,
        control_core_sha256=expected_input_signature(control_core)["sha256"],
        control_ood=control_ood,
        control_ood_sha256=expected_input_signature(control_ood)["sha256"],
        core_output=core_output,
        ood_output=ood_output,
        matched_density_output=matched_output,
        train_output=train_output,
    )
    jobs = [
        quality_job,
        *graph_jobs,
        assemble_job,
        train_job,
        transform_job,
        core_job,
        ood_job,
        matched_job,
        decision_job,
    ]
    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=round_file,
        queue_root=queue_root,
        gpu_hours_cap=GPU_HOURS_CAP,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue["schema"] = "round0130-direct-k49-degree-rescue-queue-v1"
    queue["repo_root"] = RELEASE_ROOT
    queue["queue_class"] = "gpu-research"
    queue["required_reviews"] = [
        "0105", "0106", "0107", "0108", "0124", "0129"
    ]
    queue["capability_dependencies"] = [
        "jina-diverse-25m-full768-search-qualified-v1",
        "jina-diverse-25m-full768-fuzzy-graph-v1",
        "jina-diverse-25m-full768-trained-map-seed42-v1",
        "jina-diverse-25m-map-registry-v1",
        "jina-fineweb-2m-native-k15-degree-bridge-v1",
        R0129_CAPABILITY,
    ]
    queue["capabilities_produced"] = [
        DEGREE_RESCUE_CAPABILITY,
        ATLAS_QUALITY_CAPABILITY,
    ]
    queue["training_performed"] = True
    queue["conditional_trigger"] = {
        "schema": "round0130-two-seed-degree-trigger-v1",
        "r0124_seed42": dict(r0124_evidence["signatures"]),
        "r0129_seed43": dict(r0129_evidence["signatures"]),
        "structured_proof": two_seed_proof,
        "r0124_required_outcome": R0124_INCONCLUSIVE_OUTCOME,
        "r0124_delta_at_most": -MATERIAL_DENSITY_DEGRADATION,
        "r0124_ci_upper_below": 0.0,
        "r0129_required_outcome": (
            "k15-materially-degrades-native-density"
        ),
        "review_prose_can_release": False,
    }
    queue["scientific_contract"] = {
        "treatment": "direct nonself graph degree 15->49 only",
        "population_representation_seed_model_optimizer_sampler_runtime": (
            "exact accepted R0107"
        ),
        "successful_updates": FIXED_SUCCESSFUL_UPDATES,
        "dose_rule": "fixed R0107 dose; never edge-derived for R0130",
        "k49_fixed_policy_qualification": {
            "nprobe": 64,
            "shortlist_width": 128,
            "global_mean_recall_floor": 0.90,
            "every_group_mean_recall_floor": 0.84,
            "boundary_ties": "exclude exact rank49/rank50 ties at atol 1e-7",
            "policy_sweep": False,
        },
        "native_density": {
            "high_and_low_radius_k": 15,
            "paired_delta_materiality": DENSITY_MATERIAL_DELTA,
            "bootstrap_draws": DENSITY_BOOTSTRAP_DRAWS,
            "bootstrap_seed": DENSITY_BOOTSTRAP_SEED,
            "frozen_floor_clearance_reported_separately": True,
        },
        "headline_noninferiority_retention": HEADLINE_KPI_RETENTION,
        "matched_r0040_density": "diagnostic-only",
        "projection_ffr": "diagnostic-only",
        "two_seed_trigger": two_seed_proof,
        "two_seed_results_pooled": False,
        "degree_hypothesis_proven_claimed": False,
        "second_control_training_performed": False,
        "registry_or_render_publication": (
            "not in R0130; separate CPU follow-up if atlas quality releases"
        ),
    }
    queue["jobs"] = jobs
    queue["p90_gpu_seconds"] = {
        "qualify_k49_selected_policy": P90_QUALIFICATION_SECONDS,
        **{
            str(job["id"]): P90_GRAPH_PART_SECONDS for job in graph_jobs
        },
        "train_k49_treatment": P90_TRAIN_SECONDS,
        "transform_retained_map": P90_TRANSFORM_SECONDS,
        "score_core_geometry": P90_CORE_SECONDS,
        "score_ood": P90_OOD_SECONDS,
        "score_matched_r0040_density": P90_MATCHED_SECONDS,
        "total": P90_GPU_TOTAL_SECONDS,
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--r0124-review", required=True)
    parser.add_argument("--r0124-review-sha256", required=True)
    parser.add_argument("--r0129-review", required=True)
    parser.add_argument("--r0129-review-sha256", required=True)
    parser.add_argument(
        "--queue-root", default=os.path.join(ROUND_ROOT, "queue")
    )
    args = parser.parse_args(argv)
    path = prepare_round0130(
        release_sha=args.release_sha,
        r0124_review_path=args.r0124_review,
        r0124_review_sha256=args.r0124_review_sha256,
        r0129_review_path=args.r0129_review,
        r0129_review_sha256=args.r0129_review_sha256,
        queue_root=args.queue_root,
    )
    print(json.dumps({"queue_manifest": path}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
