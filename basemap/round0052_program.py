"""Registered balanced-60M MiniLM training contract for Round 0052."""
from __future__ import annotations

import copy
import math
from typing import Any, Mapping

from .artifact_identity import canonical_json, sha256_bytes
from .round0021_program import TRAIN_CONFIG as _R0021_CONFIG
from .round0034_pipeline import GRAPH_SCHEMA
from .round0049_program import (
    CORPUS_INTERVALS,
    DIMENSION,
    K,
    ROW_COUNT,
)


ROUND_ID = "0052"
SEED = 42
REFERENCE_ROUND = "0046"
REFERENCE_POSITIVE_SOURCES = 29_781_619
REFERENCE_SUCCESSFUL_UPDATES = 500_000
EXPECTED_EXCLUDED_ROWS = 600_712
EXPECTED_RETAINED_ROWS = 59_399_288
EXPECTED_VALID_EDGES = 890_989_320
SUCCESSFUL_UPDATES = math.ceil(
    REFERENCE_SUCCESSFUL_UPDATES
    * EXPECTED_RETAINED_ROWS
    / REFERENCE_POSITIVE_SOURCES
)
PIPELINE_SCHEMA = "round0052-host-int8-balanced-canonical-pipeline-v1"
SAMPLER_CLASS = "HostInt8BalancedCanonicalSampler"


class Round0052ProgramError(RuntimeError):
    """The reviewed balanced-60M training capabilities changed."""


def train_config_from_capabilities(
    graph_manifest: Mapping[str, Any],
    *,
    graph_manifest_path: str,
    graph_manifest_sha256: str,
    substrate_manifest: Mapping[str, Any],
    substrate_manifest_path: str,
    substrate_manifest_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Derive the exact coverage-aligned 60M production configuration."""
    summary = graph_manifest.get("summary") or {}
    substrate_outputs = substrate_manifest.get("outputs") or {}
    eligibility = substrate_outputs.get("eligibility") or {}
    degree_histogram = summary.get("degree_histogram")
    if (
        graph_manifest.get("schema") != GRAPH_SCHEMA
        or graph_manifest.get("round_id") not in {"0050", "0062"}
        or int(graph_manifest.get("row_count", -1)) != ROW_COUNT
        or int(graph_manifest.get("input_k", -1)) != K
        or graph_manifest.get("inputs", {}).get("eligibility")
        != eligibility
        or int(summary.get("eligibility_excluded_source_count", -1))
        != EXPECTED_EXCLUDED_ROWS
        or int(summary.get("eligibility_retained_row_count", -1))
        != EXPECTED_RETAINED_ROWS
        or int(summary.get("retained_positive_source_count", -1))
        != EXPECTED_RETAINED_ROWS
        or int(summary.get("zero_degree_retained_source_count", -1)) != 0
        or int(summary.get("valid_canonical_edge_count", -1))
        != EXPECTED_VALID_EDGES
        or degree_histogram != {
            "0": EXPECTED_EXCLUDED_ROWS,
            str(K): EXPECTED_RETAINED_ROWS,
        }
        or EXPECTED_VALID_EDGES != EXPECTED_RETAINED_ROWS * K
        or substrate_manifest.get("schema")
        != "round0049-balanced-60m-substrate-v1"
        or int(substrate_manifest.get("row_count", -1)) != ROW_COUNT
        or int(substrate_manifest.get("dimension", -1)) != DIMENSION
        or substrate_manifest.get("global_150m_intervals")
        != [list(value) for value in CORPUS_INTERVALS]
    ):
        raise Round0052ProgramError(
            "reviewed balanced-60M graph/substrate geometry changed"
        )
    int8 = substrate_outputs.get("int8") or {}
    scales = substrate_outputs.get("scales") or {}
    if (
        int(int8.get("bytes", -1)) != ROW_COUNT * DIMENSION
        or int(scales.get("bytes", -1)) != ROW_COUNT * 2
        or not int8.get("sha256")
        or not scales.get("sha256")
        or not eligibility.get("sha256")
    ):
        raise Round0052ProgramError(
            "balanced-60M feature capabilities are incomplete"
        )

    config = copy.deepcopy(_R0021_CONFIG)
    config["schema"] = "round0052-production-config-v1"
    config["phrase"] = (
        "balanced 60M MiniLM seed42 native-k15 coverage-aligned rung"
    )
    config["row_universe"] = {
        "corpus_order": ["fineweb", "redpajama", "pile"],
        "rows_per_corpus": 20_000_000,
        "rows": ROW_COUNT,
        "input_dimension": DIMENSION,
        "embedding_dtype": "int8",
        "row_scale_dtype": "<f2",
        "source_intervals_in_150m_namespace": [
            list(value) for value in CORPUS_INTERVALS
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
            "balanced 60M subset"
        ),
    }
    config["graph"] = {
        "path": graph_manifest_path,
        "sha256": graph_manifest_sha256,
        "schema": graph_manifest["schema"],
        "k": K,
        "valid_canonical_edges": EXPECTED_VALID_EDGES,
        "positive_source_rows": EXPECTED_RETAINED_ROWS,
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
    }
    config["optimizer"]["seed"] = SEED
    config["optimizer"]["use_amp"] = "bf16"
    config["optimizer"]["successful_positive_lr_updates"] = (
        SUCCESSFUL_UPDATES
    )
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
            "reference_successful_updates": (
                REFERENCE_SUCCESSFUL_UPDATES
            ),
            "retained_positive_sources": EXPECTED_RETAINED_ROWS,
            "formula": (
                "ceil(500000 * 59399288 / 29781619)"
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
            "positive_source_count": EXPECTED_RETAINED_ROWS,
            "valid_canonical_edge_count": EXPECTED_VALID_EDGES,
            "graph_degree": (
                "fixed-15-for-every-retained-source;"
                "excluded-sources-degree-zero"
            ),
            "positive_destination_policy": (
                "native-balanced-60m-representative-only-k15;self-removed"
            ),
            "negative_sampling": (
                "uniform-balanced-60m-retained-rows-nonself"
            ),
            "uniform_with_replacement": True,
            "positive_with_replacement": True,
            "weighted_requested": False,
            "weighted_effective": False,
            "source_edge_uniform_equivalent": True,
        },
        "duplicate_control": {
            "eligibility": eligibility,
            "source_copy_or_zero_rows_excluded": EXPECTED_EXCLUDED_ROWS,
            "retained_rows": EXPECTED_RETAINED_ROWS,
            "subset_families_recomputed_after_restriction": True,
        },
        "scale_transition": {
            "reference_round": REFERENCE_ROUND,
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
            "treatment": (
                "30M fp16/device recipe to balanced 60M "
                "int8-plus-scale/host recipe and native geometry"
            ),
            "known_representation_and_residency_confound": True,
            "pure_training_scale_law_requires": (
                "a matched balanced-30M int8-plus-scale/native-graph control"
            ),
            "source_exposure_confound": (
                "absent because retained graph degree is exactly 15"
            ),
        },
    }
    config["transform"] = {
        "status": "registered-downstream-successor",
        "input": "balanced-60m-int8-plus-exact-fp16-row-scale",
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
            "matched 30M semantic-row intersection",
            "held-out query projection",
            "fixed-sample render",
        ],
    }
    config["decision_thresholds"] = {
        "training_wall_only": True,
        "exact_successful_updates_required": SUCCESSFUL_UPDATES,
        "numerical_skip_counters_must_be_zero": True,
        "geometry_claim_requires_downstream_evaluation": True,
    }
    return config, sha256_bytes(canonical_json(config))
