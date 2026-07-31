"""Frozen contract for the conditional R0129 seed-43 k15 replicate."""
from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from typing import Any

from .artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from .round0113_prompt_contrast import NEGATIVE_RNG_SEED_OFFSET
from .round0124_degree_bridge import (
    ARM,
    ATTEMPT1_EVIDENCE,
    ATTEMPT1_RELEASE_SHA,
    GRAPH_DEGREE,
    GRAPH_SEARCH_NEIGHBORS,
    RETAINED_ROWS,
    SUCCESSFUL_UPDATES,
    load_graph,
    read_sealed,
    train_config as r0124_train_config,
)


ROUND_ID = "0129"
TRAINING_SEED = 43
GRAPH_PROVENANCE_SCHEMA = "round0129-r0124-attempt1-k15-graph-v1"
TRAIN_CONFIG_SCHEMA = "round0129-seed43-k15-train-config-v1"
PRODUCTION_CONFIG_SCHEMA = "round0129-production-config-v1"
TRAIN_RECEIPT_SCHEMA = "round0129-seed43-k15-train-receipt-v1"
DIAGNOSTIC_SCHEMA = "round0129-seed43-k15-diagnostics-v1"
NATIVE_DENSITY_SCHEMA = "round0129-seed43-native-density-score-v1"
DECISION_SCHEMA = "round0129-seed43-degree-replicate-decision-v1"
CAPABILITY = "jina-fineweb-2m-native-k15-degree-bridge-seed43-v1"


class Round0129Error(RuntimeError):
    """The conditional seed-43 graph-degree replicate changed."""


def graph_provenance() -> dict[str, Any]:
    """Bind the exact successful R0124 attempt-1 graph node and bytes."""
    keep = {
        key: dict(ATTEMPT1_EVIDENCE[key])
        for key in (
            "queue_manifest",
            "runner_terminal",
            "graph_done_marker",
            "graph_manifest",
            "graph",
            "topology_probe",
        )
    }
    value = {
        "schema": GRAPH_PROVENANCE_SCHEMA,
        "source_round_id": "0124",
        "source_attempt": 1,
        "source_release_sha": ATTEMPT1_RELEASE_SHA,
        "source_terminal_verdict": "failed-after-successful-graph-node",
        "graph_rebuilt": False,
        "evidence": keep,
    }
    return verify_graph_provenance(value)


def _read_json(path: str, *, label: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError) as exc:
        raise Round0129Error(f"{label} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise Round0129Error(f"{label} is not a JSON object")
    return value


def verify_graph_provenance(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Round0129Error("R0129 graph provenance is missing")
    evidence = value.get("evidence")
    expected = {
        key: ATTEMPT1_EVIDENCE[key]
        for key in (
            "queue_manifest",
            "runner_terminal",
            "graph_done_marker",
            "graph_manifest",
            "graph",
            "topology_probe",
        )
    }
    if (
        value.get("schema") != GRAPH_PROVENANCE_SCHEMA
        or value.get("source_round_id") != "0124"
        or value.get("source_attempt") != 1
        or value.get("source_release_sha") != ATTEMPT1_RELEASE_SHA
        or value.get("source_terminal_verdict")
        != "failed-after-successful-graph-node"
        or value.get("graph_rebuilt") is not False
        or not isinstance(evidence, Mapping)
        or dict(evidence) != expected
    ):
        raise Round0129Error("R0129 graph provenance contract changed")
    for label, signature in expected.items():
        if expected_input_signature(signature["canonical_path"]) != signature:
            raise Round0129Error(f"R0129 source graph {label} bytes changed")
    queue = _read_json(
        expected["queue_manifest"]["canonical_path"],
        label="R0124 attempt-1 queue",
    )
    terminal = _read_json(
        expected["runner_terminal"]["canonical_path"],
        label="R0124 attempt-1 terminal",
    )
    done = _read_json(
        expected["graph_done_marker"]["canonical_path"],
        label="R0124 attempt-1 graph done marker",
    )
    manifest = read_sealed(
        expected["graph_manifest"]["canonical_path"],
        label="R0124 attempt-1 k15 graph manifest",
    )
    queue_sha = expected["queue_manifest"]["sha256"]
    if (
        queue.get("schema")
        != "round0124-fineweb-2m-degree-bridge-queue-v1"
        or queue.get("round_id") != "0124"
        or queue.get("release_sha") != ATTEMPT1_RELEASE_SHA
        or terminal.get("schema") != "slim-runner-terminal-v3"
        or terminal.get("round_id") != "0124"
        or terminal.get("verdict") != "failed"
        or terminal.get("queue_manifest_sha256") != queue_sha
        or terminal.get("queue_manifest_unchanged") is not True
        or terminal.get("release_checkout_unchanged") is not True
        or terminal.get("boundary_problems") != []
        or "build_k15_graph" not in (terminal.get("completed_jobs") or [])
        or done.get("schema") != "slim-runner-done-v2"
        or done.get("node") != "build_k15_graph"
        or done.get("returncode") != 0
        or done.get("queue_manifest_sha256") != queue_sha
        or done.get("release_sha") != ATTEMPT1_RELEASE_SHA
        or manifest.get("round_id") != "0124"
        or manifest.get("release_sha") != ATTEMPT1_RELEASE_SHA
        or manifest.get("graph") != expected["graph"]
        or manifest.get("topology_probe") != expected["topology_probe"]
    ):
        raise Round0129Error("R0124 source graph execution linkage changed")
    return dict(value)


def load_k15_graph(provenance: Mapping[str, Any]) -> dict[str, Any]:
    verified = verify_graph_provenance(provenance)
    evidence = verified["evidence"]
    return load_graph(
        evidence["graph_manifest"]["canonical_path"],
        expected_manifest_signature=evidence["graph_manifest"],
        expected_graph_signature=evidence["graph"],
        expected_topology_probe_signature=evidence["topology_probe"],
        expected_release_sha=ATTEMPT1_RELEASE_SHA,
    )


def train_config(
    *,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    graph_edges: int,
    retained_rows: int,
) -> tuple[dict[str, Any], str]:
    """Derive the seed-43 treatment from the reviewed R0124 recipe."""
    base, _base_sha = r0124_train_config(
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        graph_edges=graph_edges,
        retained_rows=retained_rows,
    )
    config = copy.deepcopy(base)
    config["schema"] = TRAIN_CONFIG_SCHEMA
    config["causal_invariant"] = {
        "control": "exact accepted R0117 raw seed-43 k49 map",
        "changed_factor": "fuzzy graph neighbor degree only",
        "population_rows": RETAINED_ROWS,
        "dimension": 768,
        "seed": TRAINING_SEED,
        "successful_positive_lr_updates": SUCCESSFUL_UPDATES,
        "control_topology": "R0115/R0117 variable-symmetric fuzzy k50",
        "treatment_topology": "R0124 exact immutable k15 graph bytes",
        "config_and_sampling_law_equivalent": True,
        "identical_realized_edge_draws_claimed": False,
    }
    optimizer = config["optimizer"]
    optimizer["seed"] = TRAINING_SEED
    optimizer["positive_rng_seed"] = TRAINING_SEED
    optimizer["negative_rng_seed"] = (
        TRAINING_SEED + NEGATIVE_RNG_SEED_OFFSET
    )
    pipeline = config["execution"]["expected_pipeline_stamp"]
    pipeline["positive_rng_seed"] = TRAINING_SEED
    pipeline["negative_rng_seed"] = (
        TRAINING_SEED + NEGATIVE_RNG_SEED_OFFSET
    )
    return config, sha256_bytes(canonical_json(config))


def config_equivalence(
    *,
    treatment: Mapping[str, Any],
    control: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove graph-only config/sampling-law equivalence without draw claims."""
    if treatment.get("schema") != TRAIN_CONFIG_SCHEMA:
        raise Round0129Error("R0129 treatment train config schema changed")
    if control.get("schema") != "round0113-prompt-arm-train-config-v1":
        raise Round0129Error("R0117 control train config schema changed")
    exact_sections = ("arm", "input", "model", "optimizer")
    if any(treatment.get(key) != control.get(key) for key in exact_sections):
        raise Round0129Error("R0129 non-graph train config differs from R0117")

    treatment_execution = copy.deepcopy(dict(treatment["execution"]))
    control_execution = copy.deepcopy(dict(control["execution"]))
    treatment_execution.pop("training_loop_plan", None)
    treatment_pipeline = treatment_execution.pop("expected_pipeline_stamp")
    control_pipeline = control_execution.pop("expected_pipeline_stamp")
    if treatment_execution != control_execution:
        raise Round0129Error("R0129 non-graph execution config differs")
    graph_pipeline_fields = {
        "positive_destination_policy",
        "graph_degree",
        "graph_search_neighbors_including_self",
        "graph_nonself_degree",
        "valid_canonical_edge_count",
        "feature_residency",
        "device_conversion",
        "graph",
    }
    treatment_sampling = {
        key: value
        for key, value in treatment_pipeline.items()
        if key not in graph_pipeline_fields
    }
    control_sampling = {
        key: value
        for key, value in control_pipeline.items()
        if key not in graph_pipeline_fields
    }
    if treatment_sampling != control_sampling:
        raise Round0129Error("R0129 sampling law differs beyond graph degree")
    treatment_graph = treatment["graph"]
    control_graph = control["graph"]
    if any(
        treatment_graph.get(key) != control_graph.get(key)
        for key in ("nprobe", "sampling", "positive_target_mode")
    ):
        raise Round0129Error("R0129 graph sampling policy changed")
    if (
        treatment_graph.get("k") != GRAPH_DEGREE
        or treatment_graph.get("n_neighbors_including_self")
        != GRAPH_SEARCH_NEIGHBORS
        or control_graph.get("k") != 50
    ):
        raise Round0129Error("R0129 graph-degree contrast changed")
    return {
        "schema": "round0129-config-equivalence-v1",
        "exact_equal_sections": list(exact_sections),
        "non_graph_execution_equal": True,
        "sampling_law_equal_after_graph_fields": True,
        "graph_policy_equal": True,
        "only_registered_config_difference": "graph topology/bytes/degree",
        "training_seed": TRAINING_SEED,
        "successful_updates": SUCCESSFUL_UPDATES,
        "identical_realized_edge_draws_claimed": False,
        "reason_realized_draws_are_not_paired": (
            "different weighted graph populations transform the same RNG seed "
            "through different categorical laws"
        ),
    }
