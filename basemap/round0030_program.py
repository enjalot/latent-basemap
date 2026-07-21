"""Round 0030: matched uniform-versus-fuzzy sampling on one 30M graph."""
from __future__ import annotations

import copy
from typing import Any

from .artifact_identity import canonical_json, sha256_bytes
from .round0014_program import NodeSpec
from .round0021_program import (
    Round0021MaterializedArray,
    TRAIN_CONFIG as _ROUND0021_TRAIN_CONFIG,
)


ROUND_ID = "0030"
Round0030MaterializedArray = Round0021MaterializedArray
ARMS = ("uniform", "fuzzy")

GRAPH_PATH = (
    "/data/latent-basemap/runs/round-0029/queue/artifacts/weighted-graph-v2/"
    "edges_30m_k15_fuzzy-v2.npz"
)
GRAPH_SHA256 = "fe0dbc9419575b29fd0b205c450c1b3da555b9bc25d2464e3cafb0b376542da2"
GRAPH_MANIFEST_SHA256 = (
    "e0c35246de3d9ce619de4d0ce3ac12eb9b1213d43e2cc668305f8e8a5df71d9c"
)
GRAPH_BUILD_RECEIPT_SHA256 = (
    "701de92ffeeb619e30d4ffaf240f0986f19629592674d76cf65f2bf0c7279ff8"
)
GRAPH_BUILD_CONTRACT_SHA256 = (
    "09c479e2d99ea555f513620d388b60527b851e911a89b5cab29b0b249ef0930f"
)
GRAPH_RESIDENT_EDGES = 738_221_242
GRAPH_EXCLUDED_SOURCE_EDGES = 5_617_916
GRAPH_EFFECTIVE_EDGES = 732_603_326

CAP_SHA256 = "9511ceca802da603bfbfe9164f8c6ffd7006df82df17b9499d4ed33288fde7cb"
CAP_RETAINED_ROWS = 29_781_758
CAP_EXCLUDED_ROWS = 218_242

R0023_LAYOUT_SHA256 = (
    "83d149c226b467f349f52576360c5c112ec113b389edbf4380b95ee8c338a656"
)
R0023_ZERO_RADIUS_ROWS = 46

# Max minus min across the exact accepted R0023 seed-42/43/44 panels.  These
# are metric-specific quality-noise bands.  Layout disparity is carried as a
# diagnostic separately and is deliberately not substituted for these bands.
R0023_SEED_VALUES: dict[str, tuple[float, float, float]] = {
    "ffr": (0.4664, 0.4545, 0.4442),
    "density": (0.7767, 0.7849, 0.7961),
    "purity_k256": (1.1207, 1.1230, 1.1150),
    "purity_k1024": (0.9420, 0.9118, 0.8858),
    "recall_at_10": (0.00374, 0.00376, 0.00360),
    "recall_at_50": (0.00460, 0.00460, 0.00439),
    "projection_ffr": (0.4219, 0.4155, 0.4104),
}
R0023_SEED_SPREAD = {
    name: round(max(values) - min(values), 8)
    for name, values in R0023_SEED_VALUES.items()
}
PRIMARY_METRIC = "ffr"
OOD_RETENTION_NONINFERIORITY_MARGIN = R0023_SEED_SPREAD[PRIMARY_METRIC]


def train_config_for_arm(arm: str) -> tuple[dict[str, Any], str]:
    if arm not in ARMS:
        raise ValueError(f"unknown Round 0030 arm: {arm!r}")
    weighted = arm == "fuzzy"
    config = copy.deepcopy(_ROUND0021_TRAIN_CONFIG)
    config["schema"] = f"round0030-{arm}-production-config-v1"
    config["phrase"] = (
        f"30M MiniLM matched {arm} sampling on the accepted fuzzy endpoint graph"
    )
    config["graph"] = {
        "path": GRAPH_PATH,
        "sha256": GRAPH_SHA256,
        "manifest_sha256": GRAPH_MANIFEST_SHA256,
        "build_receipt_sha256": GRAPH_BUILD_RECEIPT_SHA256,
        "build_contract_sha256": GRAPH_BUILD_CONTRACT_SHA256,
        "k": 15,
        "directed_edges": GRAPH_RESIDENT_EDGES,
        "directed_edges_effective": GRAPH_EFFECTIVE_EDGES,
        "sampling": (
            "fuzzy-weight-proportional-over-retained-source-edges-with-replacement"
            if weighted
            else "uniform-over-retained-source-edges-with-replacement"
        ),
        "with_replacement": True,
        "weights_consumed": weighted,
        "shared_endpoint_universe": True,
    }
    config["optimizer"]["weighted_edge_sampling"] = weighted
    multiplicity = config["execution"]["duplicate_multiplicity"]
    multiplicity["effective_positive_edges"] = GRAPH_EFFECTIVE_EDGES
    multiplicity["resident_positive_edges"] = GRAPH_RESIDENT_EDGES
    multiplicity["excluded_source_edges"] = GRAPH_EXCLUDED_SOURCE_EDGES
    multiplicity["cap_metadata_effective_positive_edges"] = (
        CAP_RETAINED_ROWS * multiplicity["fixed_edges_per_source"]
    )
    multiplicity["positive_source_sampling"] = (
        "fuzzy_weight_proportional_over_retained_source_edges_with_replacement"
        if weighted
        else "uniform_over_retained_source_edges_with_replacement"
    )
    # The global top-family table is useful context but its rejected R0021
    # every-family-halves rule is not a quality floor for this sampling A/B.
    multiplicity["diagnostic"] = "global-top50-family-mahalanobis-diagnostic-only"
    config["execution"].update(
        {
            "required_pipeline": "hybrid",
            "round0030_sampling_arm": arm,
            "minimum_train_upd_s": 45.0,
            "warning_train_upd_s": 55.0,
            "expected_pipeline_stamp": {
                "pipeline": "hybrid",
                "sampler_class": "HostStreamEdgeSampler",
                "positive_sampling": (
                    "weighted_with_replacement" if weighted else "uniform"
                ),
                "uniform_with_replacement": not weighted,
                "positive_with_replacement": True,
                "weighted_requested": weighted,
                "weighted_effective": weighted,
                "x_residency": "device_fp16",
                "multiplicity_positive_source_sampling": multiplicity[
                    "positive_source_sampling"
                ],
                "multiplicity_graph_degree": "variable_or_weighted_edge_universe",
            },
            "matched_arm_invariants": {
                "same_except": [
                    "schema",
                    "phrase",
                    "graph.sampling",
                    "graph.weights_consumed",
                    "optimizer.weighted_edge_sampling",
                    "execution.round0030_sampling_arm",
                    "execution.expected_pipeline_stamp.positive_sampling",
                    "execution.expected_pipeline_stamp.uniform_with_replacement",
                    "execution.expected_pipeline_stamp.weighted_requested",
                    "execution.expected_pipeline_stamp.weighted_effective",
                    (
                        "execution.expected_pipeline_stamp."
                        "multiplicity_positive_source_sampling"
                    ),
                    "execution.duplicate_multiplicity.positive_source_sampling",
                ]
            },
        }
    )
    return config, sha256_bytes(canonical_json(config))


def arm_from_job(job: dict[str, Any]) -> dict[str, Any]:
    arm = str(job.get("arm") or "uniform")
    config, digest = train_config_for_arm(arm)
    return {"arm": arm, "train_config": config, "train_config_sha256": digest}


# Used only by the legacy node entry point's argparse choices.  Slim queues use
# job handlers directly, but keeping this complete makes the adapter inspectable.
NODES = (
    NodeSpec("sampler_canary", None, 120.0, 300.0, False, "canary"),
    NodeSpec("uniform_train_30m", "sampler_canary", 4600.0, 6000.0, True, "uniform/train"),
    NodeSpec("fuzzy_train_30m", "uniform_train_30m", 4600.0, 6000.0, True, "fuzzy/train"),
    NodeSpec("uniform_transform_30m", "uniform_train_30m", 45.0, 300.0, False, "uniform/coordinates"),
    NodeSpec("fuzzy_transform_30m", "fuzzy_train_30m", 45.0, 300.0, False, "fuzzy/coordinates"),
    NodeSpec("uniform_registered_panel", "uniform_transform_30m", 2300.0, 2700.0, False, "uniform/panel"),
    NodeSpec("fuzzy_registered_panel", "fuzzy_transform_30m", 2300.0, 2700.0, False, "fuzzy/panel"),
    NodeSpec("uniform_semantic_renders", "uniform_registered_panel", 5.0, 180.0, False, "uniform/semantic-renders"),
    NodeSpec("fuzzy_semantic_renders", "fuzzy_registered_panel", 5.0, 180.0, False, "fuzzy/semantic-renders"),
    NodeSpec("uniform_ood_canary", "uniform_transform_30m", 30.0, 180.0, False, "uniform/ood-canary"),
    NodeSpec("uniform_ood_panel", "uniform_ood_canary", 20.0, 300.0, False, "uniform/ood-panel"),
    NodeSpec("fuzzy_ood_canary", "fuzzy_transform_30m", 30.0, 180.0, False, "fuzzy/ood-canary"),
    NodeSpec("fuzzy_ood_panel", "fuzzy_ood_canary", 20.0, 300.0, False, "fuzzy/ood-panel"),
    NodeSpec("comparison", "fuzzy_ood_panel", 5.0, 120.0, False, "comparison"),
)
