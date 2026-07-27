"""Coverage-aligned training contract for the deliberate balanced-90M rung."""
from __future__ import annotations

import copy
import math
from typing import Any, Mapping

from .artifact_identity import canonical_json, sha256_bytes
from .round0021_program import TRAIN_CONFIG as _R0021_CONFIG
from .round0034_pipeline import GRAPH_SCHEMA
from .round0049_program import DIMENSION, K
from .round0071_substrate import (
    ELIGIBILITY_SUMMARY,
    INTERVALS,
    ROW_COUNT,
    ROWS_PER_CORPUS,
    SUBSTRATE_SCHEMA,
    TIER,
)


ROUND_ID = "0075"
SEED = 42
REFERENCE_ROUND = "0046"
REFERENCE_POSITIVE_SOURCES = 29_781_619
REFERENCE_SUCCESSFUL_UPDATES = 500_000
SUCCESSFUL_UPDATES = math.ceil(
    REFERENCE_SUCCESSFUL_UPDATES
    * ELIGIBILITY_SUMMARY["retained_row_count"]
    / REFERENCE_POSITIVE_SOURCES
)
PIPELINE_SCHEMA = "round0075-host-int8-balanced-90m-pipeline-v1"
SAMPLER_CLASS = "HostInt8Balanced90mCanonicalSampler"


class Round0075Error(RuntimeError):
    """The balanced-90M training capability contract changed."""


def train_config_from_capabilities(
    *,
    graph_manifest: Mapping[str, Any],
    graph_manifest_path: str,
    graph_manifest_sha256: str,
    substrate_manifest: Mapping[str, Any],
    substrate_manifest_path: str,
    substrate_manifest_sha256: str,
    scale_geometry_signature: Mapping[str, Any],
    anchor_leverage_signature: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Derive the exact production configuration from reviewed capabilities."""
    excluded = int(ELIGIBILITY_SUMMARY["excluded_row_count"])
    retained = int(ELIGIBILITY_SUMMARY["retained_row_count"])
    valid_edges = retained * K
    outputs = substrate_manifest.get("outputs") or {}
    eligibility = outputs.get("eligibility") or {}
    summary = graph_manifest.get("summary") or {}
    graph_inputs = graph_manifest.get("inputs") or {}
    quality = graph_manifest.get("quality") or {}
    if (
        substrate_manifest.get("schema") != SUBSTRATE_SCHEMA
        or substrate_manifest.get("round_id") != "0071"
        or substrate_manifest.get("tier") != TIER
        or substrate_manifest.get("row_count") != ROW_COUNT
        or substrate_manifest.get("dimension") != DIMENSION
        or substrate_manifest.get("global_150m_intervals")
        != [list(value) for value in INTERVALS]
        or graph_manifest.get("schema") != GRAPH_SCHEMA
        or graph_manifest.get("round_id") != "0073"
        or graph_manifest.get("tier") != TIER
        or graph_manifest.get("row_count") != ROW_COUNT
        or graph_manifest.get("input_k") != K
        or graph_inputs.get("eligibility") != eligibility
        or graph_inputs.get("substrate", {}).get("sha256")
        != substrate_manifest_sha256
        or int(summary.get("eligibility_excluded_source_count", -1))
        != excluded
        or int(summary.get("eligibility_retained_row_count", -1))
        != retained
        or int(summary.get("retained_positive_source_count", -1))
        != retained
        or int(summary.get("zero_degree_retained_source_count", -1)) != 0
        or int(summary.get("valid_canonical_edge_count", -1))
        != valid_edges
        or summary.get("degree_histogram")
        != {"0": excluded, str(K): retained}
        or float(quality.get("mean_recall_at_15_unambiguous", -1.0)) < 0.90
    ):
        raise Round0075Error("balanced-90M graph/substrate geometry changed")
    int8 = outputs.get("int8") or {}
    scales = outputs.get("scales") or {}
    if (
        int(int8.get("bytes", -1)) != ROW_COUNT * DIMENSION
        or int(scales.get("bytes", -1)) != ROW_COUNT * 2
        or not int8.get("sha256")
        or not scales.get("sha256")
        or not eligibility.get("sha256")
        or not scale_geometry_signature.get("sha256")
        or not anchor_leverage_signature.get("sha256")
    ):
        raise Round0075Error("balanced-90M reviewed capabilities are incomplete")

    config = copy.deepcopy(_R0021_CONFIG)
    config["schema"] = "round0075-production-config-v1"
    config["phrase"] = (
        "balanced 90M MiniLM seed42 native-k15 coverage-aligned rung"
    )
    config["row_universe"] = {
        "corpus_order": ["fineweb", "redpajama", "pile"],
        "rows_per_corpus": ROWS_PER_CORPUS,
        "rows": ROW_COUNT,
        "input_dimension": DIMENSION,
        "embedding_dtype": "int8",
        "row_scale_dtype": "<f2",
        "source_intervals_in_150m_namespace": [
            list(value) for value in INTERVALS
        ],
        "substrate_manifest": {
            "canonical_path": substrate_manifest_path,
            "sha256": substrate_manifest_sha256,
        },
        "int8_path": int8["canonical_path"],
        "int8_sha256": int8["sha256"],
        "scale_path": scales["canonical_path"],
        "scale_sha256": scales["sha256"],
        "eligibility_path": eligibility["canonical_path"],
        "eligibility_sha256": eligibility["sha256"],
        "scientific_unit": (
            "one exact nonzero int8-plus-fp16-scale vector in the "
            "balanced 90M subset"
        ),
    }
    config["graph"] = {
        "path": graph_manifest_path,
        "sha256": graph_manifest_sha256,
        "schema": graph_manifest["schema"],
        "k": K,
        "valid_canonical_edges": valid_edges,
        "positive_source_rows": retained,
        "degree": "fixed-15-for-every-retained-source",
        "sampling": (
            "uniform-retained-positive-source-then-uniform-native-"
            "representative-destination-with-replacement"
        ),
        "source_edge_uniform_equivalence": {
            "holds": True,
            "reason": (
                "every retained source has exactly 15 valid edges, so "
                "uniform-source/slot and uniform-edge induce the same pair law"
            ),
        },
        "weights_consumed": False,
        "qualification": graph_inputs["gpu_qualification"],
        "mean_recall_at_15_unambiguous": quality[
            "mean_recall_at_15_unambiguous"
        ],
    }
    config["optimizer"]["seed"] = SEED
    config["optimizer"]["use_amp"] = "bf16"
    config["optimizer"]["successful_positive_lr_updates"] = SUCCESSFUL_UPDATES
    config["execution"] = {
        "device_count": 1,
        "required_pipeline": "host_int8_canonical",
        "residency": "host-ram-int8-plus-fp16-scale",
        "minimum_train_upd_s": 80.0,
        "warning_train_upd_s": 100.0,
        "performance_windows": 200,
        "performance_subfloor_patience": 2,
        "performance_abort_latency_at_floor_seconds_max": 63.0,
        "full_run_retry_count": 0,
        "coverage_alignment": {
            "reference_round": REFERENCE_ROUND,
            "reference_retained_positive_sources": (
                REFERENCE_POSITIVE_SOURCES
            ),
            "reference_successful_updates": REFERENCE_SUCCESSFUL_UPDATES,
            "retained_positive_sources": retained,
            "formula": (
                f"ceil(500000 * {retained} / "
                f"{REFERENCE_POSITIVE_SOURCES})"
            ),
            "successful_updates": SUCCESSFUL_UPDATES,
        },
        "expected_pipeline_stamp": {
            "schema": PIPELINE_SCHEMA,
            "pipeline": "host_int8_canonical",
            "sampler_class": SAMPLER_CLASS,
            "x_residency": "host_int8_materialized",
            "positive_sampling": (
                "uniform-retained-positive-source-then-uniform-native-"
                "representative-destination-with-replacement"
            ),
            "positive_source_count": retained,
            "valid_canonical_edge_count": valid_edges,
            "graph_degree": (
                "fixed-15-for-every-retained-source;"
                "excluded-sources-degree-zero"
            ),
            "positive_destination_policy": (
                "native-balanced-90m-representative-only-k15;self-removed"
            ),
            "negative_sampling": (
                "uniform-balanced-90m-retained-rows-nonself"
            ),
            "uniform_with_replacement": True,
            "positive_with_replacement": True,
            "weighted_requested": False,
            "weighted_effective": False,
            "source_edge_uniform_equivalent": True,
        },
        "duplicate_control": {
            "eligibility": eligibility,
            "source_copy_or_zero_rows_excluded": excluded,
            "retained_rows": retained,
            "subset_families_recomputed_after_restriction": True,
        },
        "scale_transition": {
            "decision_sources": {
                "r0069_scale_geometry": dict(scale_geometry_signature),
                "r0074_anchor_leverage": dict(anchor_leverage_signature),
            },
            "selected_tier": TIER,
            "selection": (
                "deliberate 60M-to-120M midpoint after 45M noninferiority "
                "and duplicate-anchor explanation of the legacy density floor"
            ),
            "same": [
                "three-corpus balance",
                "one exact nonzero vector per scientific row",
                "native representative-only directed k15",
                "seed42",
                "h2048 residual bottleneck",
                "batch8192 with positive ratio 0.05",
                "bf16 autocast",
                "coverage-aligned successful-update horizon",
            ],
            "treatment": "balanced 60M evidence to balanced 90M midpoint",
            "source_exposure_confound": (
                "absent because retained graph degree is exactly 15"
            ),
            "density_floor_tuned": False,
        },
    }
    config["transform"] = {
        "status": "registered-downstream-successor",
        "input": "balanced-90m-int8-plus-exact-fp16-row-scale",
        "model_weight_dtype": "float32",
        "output_dtype": "<f4",
        "output_dimension": 2,
        "excluded_duplicate_copy_policy": (
            "project through the same network for product addressing"
        ),
    }
    config["scorer"] = {
        "status": "registered-downstream-successor",
        "required": [
            "matched retained-30M scale comparison",
            "full balanced-90M representative geometry panel",
            "held-out query projection",
            "fixed-sample render",
            "OOD projection panels",
        ],
        "density_semantics": (
            "representative anchors and representative candidate universe; "
            "legacy all-row absolute floor is invalid and remains untuned"
        ),
    }
    config["decision_thresholds"] = {
        "training_wall_only": True,
        "exact_successful_updates_required": SUCCESSFUL_UPDATES,
        "numerical_skip_counters_must_be_zero": True,
        "geometry_claim_requires_downstream_evaluation": True,
    }
    return config, sha256_bytes(canonical_json(config))
