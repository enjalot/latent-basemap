"""Registered paired seed-43 source-exposure replication for Round 0048."""
from __future__ import annotations

import copy
from typing import Any, Mapping

from .artifact_identity import canonical_json, sha256_bytes
from .round0042_program import (
    CENTROIDS_K1024_PATH,
    CENTROIDS_K256_PATH,
    ELIGIBILITY_SHA256,
    QUERIES_PATH,
    QUERY_PROVENANCE_PATH,
    REFERENCE_RECEIPT,
    REFERENCE_RECEIPT_SHA256,
    ROW_COUNT,
    SELECTOR_PATH,
    SELECTOR_SHA256,
    SUCCESSFUL_UPDATES,
    train_config_from_graph as source_config_from_graph,
)
from .round0046_program import (
    train_config_from_graph as edge_config_from_graph,
)


ROUND_ID = "0048"
SEED = 43
ARMS = ("source_uniform", "edge_uniform")


def train_configs_from_graph(
    graph_manifest: Mapping[str, Any],
    *,
    graph_manifest_path: str,
    graph_manifest_sha256: str,
) -> dict[str, tuple[dict[str, Any], str]]:
    """Build a paired seed-43 control/treatment from the registered cells."""
    source, _ = source_config_from_graph(
        graph_manifest,
        graph_manifest_path=graph_manifest_path,
        graph_manifest_sha256=graph_manifest_sha256,
    )
    edge, _ = edge_config_from_graph(
        graph_manifest,
        graph_manifest_path=graph_manifest_path,
        graph_manifest_sha256=graph_manifest_sha256,
    )
    configs = {
        "source_uniform": copy.deepcopy(source),
        "edge_uniform": copy.deepcopy(edge),
    }
    common_pair = {
        "same": [
            "30M accepted fp16 feature rows",
            "R0041 canonical source-major k15 targets and degrees",
            "R0020 retained source and negative universe",
            "canonical destination policy",
            "uniform retained nonself negative sampling",
            "seed43",
            "h2048 residual bottleneck",
            "500k successful-update horizon",
            "bf16 autocast",
            "optimizer and schedule",
            "R0040 representative-only evaluation",
        ],
        "only_intended_change": "positive-source exposure law",
        "source_uniform": (
            "uniform retained positive source then uniform valid "
            "canonical destination"
        ),
        "edge_uniform": (
            "uniform valid canonical edge; degree-proportional source "
            "exposure"
        ),
    }
    thresholds = {
        "numerical_guards_required": True,
        "material_density_recovery_delta_min": 0.10,
        "density_equivalence_abs_delta_max": 0.03,
        "representative_ffr_delta_min": -0.02,
        "representative_projection_ffr_delta_min": -0.03,
        "representative_purity_delta_min": -0.05,
    }
    for arm, config in configs.items():
        config["schema"] = f"round0048-{arm}-production-config-v1"
        config["phrase"] = (
            f"30M MiniLM seed43 canonical {arm.replace('_', '-')} "
            "paired source-exposure replication"
        )
        config["optimizer"]["seed"] = SEED
        execution = config["execution"]
        if arm == "source_uniform":
            execution["expected_pipeline_stamp"]["schema"] = (
                "round0042-device-fp16-canonical-pipeline-v1"
            )
        execution.pop("matched_R0021_isolation", None)
        execution.pop("matched_R0042_source_exposure_isolation", None)
        execution["matched_R0048_pair"] = {
            **common_pair,
            "arm": arm,
        }
        config["decision_thresholds"] = dict(thresholds)

    return {
        arm: (
            config,
            sha256_bytes(canonical_json(config)),
        )
        for arm, config in configs.items()
    }
