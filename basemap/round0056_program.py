"""Seed-43 confirmation of the selected R0051 repulsion treatment."""
from __future__ import annotations

import copy
from typing import Any, Mapping

from .artifact_identity import canonical_json, sha256_bytes
from .round0048_program import train_configs_from_graph as r0048_configs
from .round0051_program import NEGATIVE_MULTIPLIERS


ROUND_ID = "0056"
SEED = 43
SUCCESSFUL_UPDATES = 500_000
R0051_COMPARISON = (
    "/data/latent-basemap/runs/round-0051/queue/artifacts/"
    "matched-panel/negative-bce-calibration-v1.json"
)
BASELINE_TRAIN_RECEIPT = (
    "/data/latent-basemap/runs/round-0048/queue/artifacts/"
    "edge_uniform/train/train-receipt.json"
)
BASELINE_COORDINATES = (
    "/data/latent-basemap/runs/round-0048/queue/artifacts/"
    "edge_uniform/coordinates"
)
SELECTION_TO_ARM = {
    "negative-0p50-candidate": "negative_0p50",
    "negative-0p25-candidate": "negative_0p25",
}


class Round0056ProgramError(RuntimeError):
    """R0051 did not release a treatment that R0056 can replicate."""


def selected_arm(comparison: Mapping[str, Any]) -> str:
    """Return the exact R0051 candidate arm or fail closed."""
    if (
        comparison.get("schema")
        != "round0051-negative-bce-calibration-v1"
        or comparison.get("round_id") != "0051"
        or comparison.get("selection") not in SELECTION_TO_ARM
        or comparison.get("interpretation", {}).get(
            "external_ood_adoption_gate_run"
        )
        is not False
    ):
        raise Round0056ProgramError(
            "R0051 did not select a seed-confirmation candidate"
        )
    return SELECTION_TO_ARM[str(comparison["selection"])]


def train_config_from_graph(
    graph_manifest: Mapping[str, Any],
    *,
    graph_manifest_path: str,
    graph_manifest_sha256: str,
    arm: str,
) -> tuple[dict[str, Any], str]:
    """Apply one reviewed R0051 loss treatment to the R0048 seed-43 edge arm."""
    if arm not in NEGATIVE_MULTIPLIERS:
        raise Round0056ProgramError(f"unsupported R0056 arm: {arm!r}")
    baseline = r0048_configs(
        graph_manifest,
        graph_manifest_path=graph_manifest_path,
        graph_manifest_sha256=graph_manifest_sha256,
    )["edge_uniform"][0]
    config = copy.deepcopy(baseline)
    multiplier = NEGATIVE_MULTIPLIERS[arm]
    config["schema"] = f"round0056-{arm}-production-config-v1"
    config["phrase"] = (
        "30M MiniLM seed43 canonical edge-uniform normalized BCE "
        f"negative multiplier {multiplier:.2f} confirmation"
    )
    optimizer = config["optimizer"]
    optimizer["positive_bce_multiplier"] = 1.0
    optimizer["negative_bce_multiplier"] = multiplier
    optimizer["bce_reduction"] = (
        "sum(element_bce * class_multiplier) / sum(class_multiplier)"
    )
    execution = config["execution"]
    execution.pop("matched_R0048_pair", None)
    execution["matched_R0056_seed_confirmation"] = {
        "selection_round": "0051",
        "baseline_round": "0048",
        "baseline_arm": "edge_uniform",
        "selected_arm": arm,
        "baseline_negative_bce_multiplier": 1.0,
        "treatment_negative_bce_multiplier": multiplier,
        "positive_bce_multiplier": 1.0,
        "same": [
            "30M accepted fp16 feature rows",
            "R0041 canonical graph and edge-uniform source exposure",
            "R0020 retained negative universe",
            "seed43",
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
    }
    return config, sha256_bytes(canonical_json(config))
