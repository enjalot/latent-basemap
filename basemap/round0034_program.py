"""Dynamic, content-bound training contract for the 150M R0034 rung."""
from __future__ import annotations

import copy
from typing import Any, Mapping

from .artifact_identity import canonical_json, sha256_bytes
from .round0019_program import TRAIN_CONFIG as _R0019_CONFIG
from .round0034_pipeline import (
    DEFAULT_DIMENSION,
    DEFAULT_K,
    DEFAULT_ROWS,
    GRAPH_SCHEMA,
    coverage_aligned_successful_updates,
)


ROUND_ID = "0034"
INT8_PATH = (
    "/data/latent-basemap/runs/round-0025/queue/artifacts/int8-shards/"
    "minilm-int8-150m/embeddings.i8"
)
SCALES_PATH = (
    "/data/latent-basemap/runs/round-0025/queue/artifacts/int8-shards/"
    "minilm-int8-150m/scales.f16"
)
INT8_SHA256 = "2171e4bf3c21e7156435b4b4021ca62b2ef8a57d9404b2764e6e968d210b7090"
SCALES_SHA256 = "d282d4f5a5abbe17e981d957fce1cd9e227cbd67aa3262803542d496dbbecb49"


def train_config_from_capabilities(
    canonical_graph_manifest: Mapping[str, Any],
    *,
    canonical_graph_manifest_path: str,
    canonical_graph_manifest_sha256: str,
    eligibility_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Derive, rather than guess, the exact R0034 training horizon."""
    summary = canonical_graph_manifest.get("summary") or {}
    sources = int(summary.get("retained_positive_source_count", 0))
    valid_edges = int(summary.get("valid_canonical_edge_count", 0))
    if (
        canonical_graph_manifest.get("schema") != GRAPH_SCHEMA
        or int(canonical_graph_manifest.get("row_count", -1)) != DEFAULT_ROWS
        or int(canonical_graph_manifest.get("input_k", -1)) != DEFAULT_K
        or canonical_graph_manifest.get("inputs", {}).get("eligibility", {}).get(
            "sha256"
        ) != eligibility_sha256
        or sources <= 0
        or valid_edges < sources
    ):
        raise ValueError("R0034 canonical graph capability is incomplete or mismatched")
    successful_updates = coverage_aligned_successful_updates(sources)

    config = copy.deepcopy(_R0019_CONFIG)
    config["schema"] = "round0034-production-config-v1"
    config["phrase"] = "one coverage-aligned 150M MiniLM host-int8 rung"
    config["row_universe"] = {
        "corpus_order": ["fineweb", "redpajama", "pile"],
        "rows": DEFAULT_ROWS,
        "input_dimension": DEFAULT_DIMENSION,
        "embedding_dtype": "int8",
        "row_scale_dtype": "<f2",
        "int8_path": INT8_PATH,
        "int8_sha256": INT8_SHA256,
        "scale_path": SCALES_PATH,
        "scale_sha256": SCALES_SHA256,
        "eligibility_sha256": eligibility_sha256,
    }
    config["graph"] = {
        "path": canonical_graph_manifest_path,
        "sha256": canonical_graph_manifest_sha256,
        "k": DEFAULT_K,
        "directed_edges": valid_edges,
        "retained_positive_sources": sources,
        "sampling": (
            "uniform-retained-positive-source-then-uniform-valid-canonical-target"
        ),
        "with_replacement": True,
        "weights_consumed": False,
        "degree": "variable-after-canonical-destination-filtering",
    }
    config["optimizer"]["successful_positive_lr_updates"] = successful_updates
    config["optimizer"]["use_amp"] = "bf16"
    config["optimizer"]["weighted_edge_sampling"] = False
    config["execution"] = {
        "device_count": 1,
        "required_pipeline": "host_int8_canonical",
        "residency": "host-ram-int8-plus-fp16-scale",
        "source_policy": "uniform-retained-positive-source",
        "destination_policy": (
            "uniform-valid-canonical-target;duplicate-to-representative;"
            "zero-self-repeat-dropped"
        ),
        "negative_policy": "uniform-R0033-retained-rows-nonself",
        "canary_optimizer_updates": 0,
        "canary_operation": (
            "two-endpoint-gather-dequant-forward-bce-backward-clip-no-optimizer"
        ),
        "minimum_post_setup_headroom_gib": 1.5,
        "minimum_canary_train_step_equivalents_per_second": 90.0,
        "full_run_retry_count": 0,
        "coverage_alignment": {
            "reference_round": "0019",
            "reference_retained_positive_sources": 29_989_838,
            "reference_successful_updates": 500_000,
            "retained_positive_sources": sources,
            "formula": "ceil(500000 * retained_positive_sources / 29989838)",
            "successful_updates": successful_updates,
        },
        "expected_pipeline_stamp": {
            "pipeline": "host_int8_canonical",
            "sampler_class": "HostInt8CanonicalSampler",
            "x_residency": "host_int8_materialized",
            "positive_sampling": (
                "uniform-retained-positive-source-then-uniform-valid-canonical-"
                "destination-with-replacement"
            ),
            "negative_sampling": "uniform-R0033-retained-rows-nonself",
        },
    }
    # Do not inherit R0019's 30M index/query paths into a 150M receipt.  The
    # core queue stops after training; downstream transform/panel contracts
    # must independently bind the 150M scorer and excluded-copy policy.
    config["transform"] = {
        "status": "downstream-not-in-core-queue",
        "input": "host-int8-plus-exact-fp16-row-scale",
        "model_weight_dtype": "float32",
        "output_dtype": "<f4",
        "output_dimension": 2,
        "excluded_duplicate_copy_policy": "project-through-the-same-network",
    }
    config["scorer"] = {
        "status": "downstream-not-in-core-queue",
        "required": [
            "150M same-domain registered panel",
            "corrected Dadabase/TREC-COVID OOD card",
            "fixed-sample render",
        ],
    }
    digest = sha256_bytes(canonical_json(config))
    return config, digest
