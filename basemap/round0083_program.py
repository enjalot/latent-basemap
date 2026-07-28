"""Registered 30M graph-recall treatments for Round 0083."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from .artifact_identity import canonical_json, sha256_bytes
from .round0034_pipeline import GRAPH_SCHEMA
from .round0053_program import (
    EXPECTED_RETAINED_ROWS,
    ROW_COUNT,
)
from .round0055_program import (
    SUCCESSFUL_UPDATES,
    train_config_from_capabilities as _baseline_config,
)


ROUND_ID = "0083"
NPROBES = (16, 32)
TARGET_RECALL_BANDS = {
    16: (0.82, 0.88),
    32: (0.87, 0.91),
}
PANEL_SCHEMA = "round0083-registered-panel-v1"
CONFIG_SCHEMA = "round0083-production-config-v1"
TRAIN_RECEIPT_SCHEMAS = {
    nprobe: f"round0083-nprobe{nprobe}-train-receipt-v1"
    for nprobe in NPROBES
}


class Round0083ProgramError(RuntimeError):
    """A graph-recall treatment differs from the registered contract."""


def train_config_from_graph(
    graph_manifest: Mapping[str, Any],
    *,
    graph_manifest_path: str,
    graph_manifest_sha256: str,
    substrate_manifest: Mapping[str, Any],
    substrate_manifest_path: str,
    substrate_manifest_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Build the R0061 recipe while changing only the graph treatment."""
    candidate = graph_manifest.get("candidate_generator") or {}
    nprobe = int(candidate.get("nprobe", -1))
    summary = graph_manifest.get("summary") or {}
    if (
        graph_manifest.get("schema") != GRAPH_SCHEMA
        or graph_manifest.get("round_id") != ROUND_ID
        or nprobe not in NPROBES
        or int(graph_manifest.get("row_count", -1)) != ROW_COUNT
        or int(summary.get("retained_positive_source_count", -1))
        != EXPECTED_RETAINED_ROWS
        or candidate.get("search_width") != 128
        or candidate.get("index_search_width") != 129
        or candidate.get("selected_neighbors") != 15
        or candidate.get("exact_rerank") is not True
    ):
        raise Round0083ProgramError("R0083 graph treatment identity changed")

    # The reviewed R0055 builder admits the same fixed-degree graph geometry
    # from R0060. Normalize only that historical producer label, then restore
    # the actual treatment identity below. This preserves every recipe field.
    normalized = copy.deepcopy(dict(graph_manifest))
    normalized["round_id"] = "0060"
    config, _ = _baseline_config(
        normalized,
        graph_manifest_path=graph_manifest_path,
        graph_manifest_sha256=graph_manifest_sha256,
        substrate_manifest=substrate_manifest,
        substrate_manifest_path=substrate_manifest_path,
        substrate_manifest_sha256=substrate_manifest_sha256,
    )
    config["schema"] = CONFIG_SCHEMA
    config["phrase"] = (
        "matched balanced 30M MiniLM int8 seed42 graph-recall treatment "
        f"nprobe={nprobe}"
    )
    config["graph"]["schema"] = GRAPH_SCHEMA
    config["execution"]["graph_recall_treatment"] = {
        "round_id": ROUND_ID,
        "nprobe": nprobe,
        "candidate_recall_at_15_unambiguous": float(
            (graph_manifest.get("quality") or {})[
                "mean_recall_at_15_unambiguous"
            ]
        ),
        "baseline_nprobe": 64,
        "baseline_candidate_recall_at_15_unambiguous": 0.9224609375000001,
        "only_intended_difference_from_r0061": (
            "canonical graph neighbor identities induced by fixed nprobe"
        ),
    }
    matched = config["execution"]["matched_r0052_scale_control"]
    matched["same"] = [
        *matched["same"],
        "seed42 and exact 500003-update R0061 training/evaluation recipe",
    ]
    matched["treatment_difference"] = (
        f"graph candidate nprobe {nprobe} versus reviewed R0060 nprobe 64; "
        "all sources retain fixed degree 15"
    )
    config["decision_thresholds"]["graph_recall_sensitivity_only"] = True
    if (
        config["optimizer"]["seed"] != 42
        or config["optimizer"]["successful_positive_lr_updates"]
        != SUCCESSFUL_UPDATES
    ):
        raise Round0083ProgramError("R0061 optimizer contract changed")
    return config, sha256_bytes(canonical_json(config))
