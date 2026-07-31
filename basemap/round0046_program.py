"""Registered 30M positive-source exposure isolation for Round 0046."""
from __future__ import annotations

import copy
from typing import Any, Mapping

from .artifact_identity import canonical_json, sha256_bytes
from .round0042_program import (
    CENTROIDS_K1024_PATH,
    CENTROIDS_K256_PATH,
    DIMENSION,
    ELIGIBILITY_PATH,
    ELIGIBILITY_SHA256,
    K,
    QUERIES_PATH,
    QUERY_PROVENANCE_PATH,
    REFERENCE_RECEIPT,
    REFERENCE_RECEIPT_SHA256,
    ROW_COUNT,
    SEED,
    SELECTOR_PATH,
    SELECTOR_SHA256,
    SUCCESSFUL_UPDATES,
    train_config_from_graph as r0042_train_config_from_graph,
)


ROUND_ID = "0046"
R0042_TRAIN_RECEIPT = (
    "/data/latent-basemap/runs/round-0042/queue/artifacts/train/"
    "train-receipt.json"
)
R0042_COORDINATES = (
    "/data/latent-basemap/runs/round-0042/queue/artifacts/coordinates"
)
R0042_PANEL = (
    "/data/latent-basemap/runs/round-0042/queue/artifacts/matched-panel/"
    "r0042_canonical-panel.json"
)


def train_config_from_graph(
    graph_manifest: Mapping[str, Any],
    *,
    graph_manifest_path: str,
    graph_manifest_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Change only the positive-source exposure law from accepted R0042."""
    control, _control_sha256 = r0042_train_config_from_graph(
        graph_manifest,
        graph_manifest_path=graph_manifest_path,
        graph_manifest_sha256=graph_manifest_sha256,
    )
    config = copy.deepcopy(control)
    config["schema"] = "round0046-production-config-v1"
    config["phrase"] = (
        "30M MiniLM seed42 canonical edge-uniform source-exposure isolation"
    )
    config["graph"]["sampling"] = (
        "uniform-valid-canonical-edge-with-replacement"
    )
    execution = config["execution"]
    execution["required_pipeline"] = (
        "device_fp16_canonical_edge_uniform"
    )
    execution["expected_pipeline_stamp"].update({
        "schema": "round0046-device-fp16-canonical-edge-uniform-v1",
        "pipeline": "device_fp16_canonical_edge_uniform",
        "sampler_class": "DeviceEdgeUniformCanonicalSampler",
        "positive_sampling": (
            "uniform-valid-canonical-edge-with-replacement"
        ),
        "positive_source_sampling": (
            "degree-proportional-over-positive-sources"
        ),
        "flat_edge_rank_mapping": (
            "int64-degree-prefix-searchsorted-right"
        ),
    })
    execution.pop("matched_R0021_isolation", None)
    execution["matched_R0042_source_exposure_isolation"] = {
        "same": [
            "30M accepted fp16 feature rows",
            "R0041 canonical source-major k15 targets and degrees",
            "R0020 retained source and negative universe",
            "canonical destination policy",
            "uniform nonself negative sampling",
            "seed42",
            "h2048 residual bottleneck",
            "500k successful-update horizon",
            "bf16 autocast",
            "optimizer and schedule",
            "R0040 representative-only evaluation",
        ],
        "control": (
            "R0042 uniform positive source then uniform valid destination"
        ),
        "treatment": (
            "uniform valid canonical edge; source exposure proportional "
            "to post-canonicalization degree"
        ),
        "only_intended_change": "positive-source exposure law",
    }
    config["decision_thresholds"] = {
        "numerical_guards_required": True,
        "material_density_recovery_delta_min": 0.10,
        "density_equivalence_abs_delta_max": 0.03,
        "representative_ffr_delta_min": -0.02,
        "representative_projection_ffr_delta_min": -0.03,
        "representative_purity_delta_min": -0.05,
        "classification": {
            "source_exposure_primary_contributor": (
                "density delta >= 0.10 and all quality non-inferiority "
                "guards pass"
            ),
            "source_exposure_not_sufficient": (
                "absolute density delta <= 0.03"
            ),
            "mixed_or_seed_sensitive": "all other finite outcomes",
        },
    }
    return config, sha256_bytes(canonical_json(config))
