"""One paired seed-sensitivity contrast for the reviewed balanced-90M map."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from .artifact_identity import canonical_json, sha256_bytes
from .round0071_substrate import ELIGIBILITY_SUMMARY, ROW_COUNT, TIER
from .round0075_training import SUCCESSFUL_UPDATES


ROUND_ID = "0084"
SEED = 43
BASELINE_SEED = 42
CONFIG_SCHEMA = "round0084-production-config-v1"
TRAIN_RECEIPT_SCHEMA = "round0075-train-receipt-v1"
MODEL_LABEL = "r0084-90m-seed43"
MATCHED_KEY = "r0084-seed43-90m-on-30m"
FULL_KEY = "r0084-seed43-90m-on-90m"
PANEL_SCHEMA = "round0084-registered-panel-v1"


class Round0084ProgramError(RuntimeError):
    """The registered one-seed contrast differs from the R0075 baseline."""


def seed43_config_from_seed42(
    baseline_config: Mapping[str, Any],
    *,
    graph_manifest: Mapping[str, Any],
    graph_manifest_path: str,
    graph_manifest_sha256: str,
    substrate_manifest: Mapping[str, Any],
    substrate_manifest_path: str,
    substrate_manifest_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Change only the optimizer seed and explanatory receipt metadata."""
    base = copy.deepcopy(dict(baseline_config))
    row_universe = base.get("row_universe") or {}
    graph = base.get("graph") or {}
    optimizer = base.get("optimizer") or {}
    execution = base.get("execution") or {}
    if (
        base.get("schema") != "round0075-production-config-v1"
        or row_universe.get("rows") != ROW_COUNT
        or row_universe.get("substrate_manifest") != {
            "canonical_path": substrate_manifest_path,
            "sha256": substrate_manifest_sha256,
        }
        or graph.get("path") != graph_manifest_path
        or graph.get("sha256") != graph_manifest_sha256
        or graph.get("schema") != graph_manifest.get("schema")
        or optimizer.get("seed") != BASELINE_SEED
        or optimizer.get("successful_positive_lr_updates")
        != SUCCESSFUL_UPDATES
        or execution.get("required_pipeline") != "host_int8_canonical"
        or execution.get("full_run_retry_count") != 0
        or substrate_manifest.get("round_id") != "0071"
        or substrate_manifest.get("tier") != TIER
        or substrate_manifest.get("row_count") != ROW_COUNT
        or graph_manifest.get("round_id") != "0073"
        or graph_manifest.get("tier") != TIER
        or int(
            (graph_manifest.get("summary") or {}).get(
                "retained_positive_source_count", -1
            )
        )
        != ELIGIBILITY_SUMMARY["retained_row_count"]
    ):
        raise Round0084ProgramError(
            "reviewed R0075 seed-42 production contract changed"
        )

    candidate = copy.deepcopy(base)
    candidate["schema"] = CONFIG_SCHEMA
    candidate["phrase"] = (
        "balanced 90M MiniLM seed43 paired sensitivity contrast against "
        "the reviewed seed42 map"
    )
    candidate["optimizer"]["seed"] = SEED
    transition = candidate["execution"]["scale_transition"]
    transition["same"] = [
        value for value in transition["same"] if value != "seed42"
    ]
    transition["same"].append(
        "all R0075 recipe fields except the registered random seed"
    )
    transition["treatment"] = "optimizer/sampler/model seed 43 versus seed 42"
    candidate["execution"]["seed_sensitivity_treatment"] = {
        "baseline_round": "0075",
        "baseline_seed": BASELINE_SEED,
        "treatment_seed": SEED,
        "only_intended_training_difference": "random seed",
        "one_contrast_is_not_a_variance_estimate": True,
    }
    candidate["decision_thresholds"]["one_seed_contrast_only"] = True
    candidate["decision_thresholds"][
        "does_not_establish_seed_noise_band"
    ] = True
    return candidate, sha256_bytes(canonical_json(candidate))
