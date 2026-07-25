"""Registered canonical-30M attraction/repulsion calibration for R0051."""
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
)
from .round0046_program import train_config_from_graph


ROUND_ID = "0051"
SEED = 42
ARMS = ("negative_0p50", "negative_0p25")
NEGATIVE_MULTIPLIERS = {
    "negative_0p50": 0.50,
    "negative_0p25": 0.25,
}
BASELINE_MULTIPLIER = 1.0
BASELINE_TRAIN_RECEIPT = (
    "/data/latent-basemap/runs/round-0046/queue/artifacts/train/"
    "train-receipt.json"
)
BASELINE_COORDINATES = (
    "/data/latent-basemap/runs/round-0046/queue/artifacts/coordinates"
)
BASELINE_COMPARISON = (
    "/data/latent-basemap/runs/round-0046/queue/artifacts/matched-panel/"
    "source-exposure-comparison-v1.json"
)


def train_configs_from_graph(
    graph_manifest: Mapping[str, Any],
    *,
    graph_manifest_path: str,
    graph_manifest_sha256: str,
) -> dict[str, tuple[dict[str, Any], str]]:
    """Derive two class-weight treatments from the exact R0046 baseline."""
    baseline, _ = train_config_from_graph(
        graph_manifest,
        graph_manifest_path=graph_manifest_path,
        graph_manifest_sha256=graph_manifest_sha256,
    )
    configs: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        multiplier = NEGATIVE_MULTIPLIERS[arm]
        config = copy.deepcopy(baseline)
        config["schema"] = f"round0051-{arm}-production-config-v1"
        config["phrase"] = (
            "30M MiniLM seed42 canonical edge-uniform normalized BCE "
            f"negative multiplier {multiplier:.2f}"
        )
        optimizer = config["optimizer"]
        optimizer["positive_bce_multiplier"] = 1.0
        optimizer["negative_bce_multiplier"] = multiplier
        optimizer["bce_reduction"] = (
            "sum(element_bce * class_multiplier) / "
            "sum(class_multiplier)"
        )
        execution = config["execution"]
        execution.pop("matched_R0042_source_exposure_isolation", None)
        execution["matched_R0051_repulsion_calibration"] = {
            "baseline_round": "0046",
            "baseline_negative_bce_multiplier": BASELINE_MULTIPLIER,
            "treatment_negative_bce_multiplier": multiplier,
            "positive_bce_multiplier": 1.0,
            "same": [
                "30M accepted fp16 feature rows",
                "R0041 canonical graph and edge-uniform source exposure",
                "R0020 retained negative universe",
                "seed42",
                "h2048 residual bottleneck",
                "batch8192 with positive ratio 0.05",
                "500k successful updates",
                "bf16 autocast",
                "optimizer, schedule, transform, and panel",
            ],
            "only_intended_change": (
                "negative BCE contribution relative to positive BCE"
            ),
        }
        execution["expected_loss_stamp"] = {
            "loss_class": "NormalizedClassWeightedBCELoss",
            "positive_multiplier": 1.0,
            "negative_multiplier": multiplier,
            "reduction": "weighted-sum-over-weight-sum",
            "positive_threshold": 0.5,
        }
        config["decision_thresholds"] = {
            "density_improvement_min": 0.05,
            "representative_ffr_delta_min": -0.02,
            "representative_projection_ffr_delta_min": -0.03,
            "representative_purity_delta_min": -0.05,
            "prefer_smaller_change_unless_extra_density_min": 0.05,
        }
        configs[arm] = config
    return {
        arm: (
            config,
            sha256_bytes(canonical_json(config)),
        )
        for arm, config in configs.items()
    }

