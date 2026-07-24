"""Round 0039: 30M uniform update-budget response diagnostic."""
from __future__ import annotations

import copy
from typing import Any

from .artifact_identity import canonical_json, sha256_bytes
from .round0014_program import NodeSpec
from . import round0030_program as source


ROUND_ID = "0039"
Round0039MaterializedArray = source.Round0030MaterializedArray
ARMS = ("u250k", "u1000k")
UPDATES_BY_ARM = {"u250k": 250_000, "u1000k": 1_000_000}

GRAPH_PATH = source.GRAPH_PATH
GRAPH_SHA256 = source.GRAPH_SHA256
GRAPH_MANIFEST_SHA256 = source.GRAPH_MANIFEST_SHA256
GRAPH_BUILD_RECEIPT_SHA256 = source.GRAPH_BUILD_RECEIPT_SHA256
GRAPH_BUILD_CONTRACT_SHA256 = source.GRAPH_BUILD_CONTRACT_SHA256
GRAPH_RESIDENT_EDGES = source.GRAPH_RESIDENT_EDGES
GRAPH_EXCLUDED_SOURCE_EDGES = source.GRAPH_EXCLUDED_SOURCE_EDGES
GRAPH_EFFECTIVE_EDGES = source.GRAPH_EFFECTIVE_EDGES
CAP_SHA256 = source.CAP_SHA256
CAP_RETAINED_ROWS = source.CAP_RETAINED_ROWS
CAP_EXCLUDED_ROWS = source.CAP_EXCLUDED_ROWS
R0023_SEED_SPREAD = source.R0023_SEED_SPREAD


def train_config_for_arm(arm: str) -> tuple[dict[str, Any], str]:
    if arm not in ARMS:
        raise ValueError(f"unknown Round 0039 arm: {arm!r}")
    updates = UPDATES_BY_ARM[arm]
    config, _ = source.train_config_for_arm("uniform")
    config = copy.deepcopy(config)
    config["schema"] = f"round0039-{arm}-production-config-v1"
    config["phrase"] = (
        f"30M MiniLM uniform retained-source budget diagnostic at {updates} "
        "successful positive-LR updates"
    )
    config["optimizer"]["successful_positive_lr_updates"] = updates
    execution = config["execution"]
    execution.pop("round0030_sampling_arm", None)
    execution.pop("matched_arm_invariants", None)
    execution["round0039_budget_arm"] = arm
    execution["budget_ladder_invariants"] = {
        "same_except": [
            "schema",
            "phrase",
            "optimizer.successful_positive_lr_updates",
            "execution.round0039_budget_arm",
        ],
        "control": {
            "round_id": "0030",
            "arm": "uniform",
            "successful_positive_lr_updates": 500_000,
        },
    }
    return config, sha256_bytes(canonical_json(config))


def arm_from_job(job: dict[str, Any]) -> dict[str, Any]:
    arm = str(job.get("arm") or "")
    config, digest = train_config_for_arm(arm)
    return {
        "arm": arm,
        "updates": UPDATES_BY_ARM[arm],
        "train_config": config,
        "train_config_sha256": digest,
    }


NODES = (
    NodeSpec("sampler_canary", None, 120.0, 300.0, False, "sampler-canary"),
    NodeSpec("u250k_train_30m", "sampler_canary", 2200.0, 3000.0, True,
             "u250k/train"),
    NodeSpec("u1000k_train_30m", "u250k_train_30m", 8800.0, 10_000.0, True,
             "u1000k/train"),
    NodeSpec("u250k_transform_30m", "u250k_train_30m", 45.0, 300.0, False,
             "u250k/coordinates"),
    NodeSpec("u1000k_transform_30m", "u1000k_train_30m", 45.0, 300.0, False,
             "u1000k/coordinates"),
    NodeSpec("u250k_registered_panel", "u250k_transform_30m",
             2300.0, 2700.0, False, "u250k/panel"),
    NodeSpec("u1000k_registered_panel", "u1000k_transform_30m",
             2300.0, 2700.0, False, "u1000k/panel"),
    NodeSpec("u250k_semantic_renders", "u250k_registered_panel",
             5.0, 180.0, False, "u250k/semantic-renders"),
    NodeSpec("u1000k_semantic_renders", "u1000k_registered_panel",
             5.0, 180.0, False, "u1000k/semantic-renders"),
    NodeSpec("budget_response", "u1000k_registered_panel",
             5.0, 120.0, False, "budget-response"),
)
