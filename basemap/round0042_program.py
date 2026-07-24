"""Registered 30M canonical-destination training cell for Round 0042."""
from __future__ import annotations

import copy
from typing import Any, Mapping

from .artifact_identity import canonical_json, sha256_bytes
from .round0014_program import (
    ACCEPTED_CAPABILITY_SHA256,
    CENTROIDS_K1024_PATH,
    CENTROIDS_K256_PATH,
    QUERIES_PATH,
    QUERY_PROVENANCE_PATH,
)
from .round0021_program import TRAIN_CONFIG as R0021_TRAIN_CONFIG


ROUND_ID = "0042"
ROW_COUNT = 30_000_000
DIMENSION = 384
K = 15
SEED = 42
SUCCESSFUL_UPDATES = 500_000

ELIGIBILITY_PATH = (
    "/data/latent-basemap/runs/round-0020/queue/artifacts/"
    "duplicate-census/global-duplicate-census-v1.npz"
)
ELIGIBILITY_SHA256 = (
    "834089fcbd9a722cec4f05be6382ed8430d27280e7e23ca0855785e3f48ea5e2"
)
SELECTOR_PATH = (
    "/data/latent-basemap/runs/round-0040/queue/artifacts/"
    "minilm-reference/representative-selector.npz"
)
SELECTOR_SHA256 = (
    "4f3b8a13649589d4b7ce6e4fb4828cefec606d84fc880d99ac7dd119ad787bde"
)
REFERENCE_RECEIPT = (
    "/data/latent-basemap/runs/round-0040/queue/artifacts/"
    "minilm-reference/receipt.json"
)
REFERENCE_RECEIPT_SHA256 = (
    "8fe3d0c41283ccde7d2779350ac7acdba34d7f50fa8a64d8703a632df7ce239a"
)
R0021_TRAIN_RECEIPT = (
    "/data/latent-basemap/runs/round-0021/queue/artifacts/train/"
    "train-receipt.json"
)
R0021_COORDINATES = (
    "/data/latent-basemap/runs/round-0021/queue/artifacts/coordinates"
)
R0021_PANEL = (
    "/data/latent-basemap/runs/round-0021/queue/artifacts/panel/panel.json"
)


class Round0042ProgramError(RuntimeError):
    """The registered canonical-destination cell changed."""


def train_config_from_graph(
    graph_manifest: Mapping[str, Any],
    *,
    graph_manifest_path: str,
    graph_manifest_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Derive the one training config from the immutable R0041 graph."""
    summary = graph_manifest.get("summary") or {}
    if (
        graph_manifest.get("schema")
        != "minilm-canonical-source-major-k15-v1"
        or graph_manifest.get("round_id") != "0041"
        or int(graph_manifest.get("row_count", -1)) != ROW_COUNT
        or int(graph_manifest.get("input_k", -1)) != K
        or graph_manifest.get("inputs", {}).get(
            "eligibility", {}
        ).get("sha256") != ELIGIBILITY_SHA256
        or int(summary.get("eligibility_excluded_source_count", -1))
        != 218_242
        or int(summary.get("eligibility_retained_row_count", -1))
        != 29_781_758
        or int(summary.get("retained_positive_source_count", -1))
        != 29_781_619
        or int(summary.get("zero_degree_retained_source_count", -1)) != 139
        or int(summary.get("valid_canonical_edge_count", -1))
        != 444_198_115
    ):
        raise Round0042ProgramError("R0041 graph geometry changed")

    config = copy.deepcopy(R0021_TRAIN_CONFIG)
    config["schema"] = "round0042-production-config-v1"
    config["phrase"] = (
        "30M MiniLM seed42 canonical-destination isolation against R0021"
    )
    config["graph"] = {
        "path": graph_manifest_path,
        "sha256": graph_manifest_sha256,
        "schema": graph_manifest["schema"],
        "k": K,
        "input_directed_edges": int(summary["input_edge_count"]),
        "valid_canonical_edges": int(
            summary["valid_canonical_edge_count"]
        ),
        "positive_source_rows": int(
            summary["retained_positive_source_count"]
        ),
        "sampling": (
            "uniform-positive-source-then-uniform-valid-canonical-"
            "destination-with-replacement"
        ),
        "weights_consumed": False,
    }
    config["execution"].pop("duplicate_multiplicity", None)
    config["execution"].update({
        "required_pipeline": "device_fp16_canonical",
        "residency": "device_fp16",
        "minimum_train_upd_s": 80.0,
        "warning_train_upd_s": 95.0,
        "performance_windows": 200,
        "performance_subfloor_patience": 2,
        "performance_abort_latency_at_floor_seconds_max": 63.0,
        "expected_pipeline_stamp": {
            "pipeline": "device_fp16_canonical",
            "sampler_class": "DeviceCanonicalSampler",
            "x_residency": "device_fp16",
            "positive_sampling": (
                "uniform-retained-positive-source-then-uniform-valid-"
                "canonical-destination-with-replacement"
            ),
            "positive_source_count": 29_781_619,
            "valid_canonical_edge_count": 444_198_115,
            "graph_degree": (
                "variable-1-through-15;zero-degree-sources-excluded"
            ),
            "positive_destination_policy": (
                "R0020-duplicate-to-representative;"
                "zero-self-repeated-dropped"
            ),
            "negative_sampling": "uniform-R0020-retained-rows-nonself",
            "uniform_with_replacement": True,
            "positive_with_replacement": True,
            "weighted_requested": False,
            "weighted_effective": False,
        },
        "duplicate_control": {
            "scientific_unit": "one-exact-fp16-vector",
            "source_copy_rows_excluded": 218_242,
            "destination_copies_mapped_to_representative": int(
                summary["duplicate_destinations_mapped"]
            ),
            "zero_degree_retained_sources_excluded": 139,
            "zero_degree_fraction_of_R0021_sources": (
                139 / 29_781_758
            ),
        },
        "matched_R0021_isolation": {
            "same": [
                "30M accepted feature rows",
                "source-major IVF-PQ k15 topology",
                "uniform positive-source law",
                "seed42",
                "h2048 residual bottleneck",
                "500k successful-update horizon",
                "bf16 autocast",
            ],
            "treatment": (
                "canonical representative destinations with self/repeat drop"
            ),
            "mechanical_source_universe_delta": (
                "139 retained rows become zero-degree after canonicalization "
                "and cannot be positive sources (4.67e-6 of R0021 sources)"
            ),
        },
        "accepted_input_pack_capability_sha256": (
            ACCEPTED_CAPABILITY_SHA256
        ),
    })
    config["optimizer"]["seed"] = SEED
    config["optimizer"]["successful_positive_lr_updates"] = (
        SUCCESSFUL_UPDATES
    )
    config["decision_thresholds"] = {
        "numerical_guards_required": True,
        "representative_ffr_delta_min": -0.01,
        "representative_projection_ffr_delta_min": -0.02,
        "representative_purity_delta_min": -0.05,
        "interpretation": (
            "non-inferiority guards protect the canonical scientific-unit "
            "policy; all signed deltas remain reportable"
        ),
    }
    return config, sha256_bytes(canonical_json(config))


REGISTERED_EVALUATION_INPUTS = {
    "selector": {"path": SELECTOR_PATH, "sha256": SELECTOR_SHA256},
    "reference_receipt": {
        "path": REFERENCE_RECEIPT,
        "sha256": REFERENCE_RECEIPT_SHA256,
    },
    "queries": {"path": QUERIES_PATH},
    "query_provenance": {"path": QUERY_PROVENANCE_PATH},
    "centroids_k256": {"path": CENTROIDS_K256_PATH},
    "centroids_k1024": {"path": CENTROIDS_K1024_PATH},
}
