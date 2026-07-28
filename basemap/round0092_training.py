"""Coverage-aligned training contract for the balanced-150M rung."""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

from .artifact_identity import canonical_json, sha256_bytes
from .round0021_program import TRAIN_CONFIG as _R0021_CONFIG
from .round0034_pipeline import GRAPH_SCHEMA
from .round0049_program import DIMENSION, K
from .round0086_program import (
    EXCLUDED_ROWS,
    RETAINED_ROWS,
    ROW_COUNT,
    SUBSTRATE_SCHEMA,
)


ROUND_ID = "0092"
SEED = 42
TIER = "150m"
REFERENCE_ROUND = "0046"
REFERENCE_POSITIVE_SOURCES = 29_781_619
REFERENCE_SUCCESSFUL_UPDATES = 500_000
SUCCESSFUL_UPDATES = math.ceil(
    REFERENCE_SUCCESSFUL_UPDATES
    * RETAINED_ROWS
    / REFERENCE_POSITIVE_SOURCES
)
PERFORMANCE_WARMUP_UPDATES = 200
PERFORMANCE_WINDOW_UPDATES_MAX = 2_500
PERFORMANCE_WINDOWS = math.ceil(
    (SUCCESSFUL_UPDATES - PERFORMANCE_WARMUP_UPDATES)
    / PERFORMANCE_WINDOW_UPDATES_MAX
)
MINIMUM_UPDATES_PER_SECOND = 100.0
WARNING_UPDATES_PER_SECOND = 110.0
PIPELINE_SCHEMA = "round0092-host-int8-balanced-150m-pipeline-v1"
SAMPLER_CLASS = "HostInt8Balanced150mCanonicalSampler"


class Round0092Error(RuntimeError):
    """The balanced-150M training capability contract changed."""


def train_config_from_capabilities(
    *,
    graph_manifest: Mapping[str, Any],
    graph_manifest_path: str,
    graph_manifest_sha256: str,
    substrate_manifest: Mapping[str, Any],
    substrate_manifest_path: str,
    substrate_manifest_sha256: str,
    scale_geometry_signature: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Derive one exact production configuration from reviewed capabilities."""
    outputs = substrate_manifest.get("outputs") or {}
    eligibility = outputs.get("eligibility") or {}
    graph_inputs = graph_manifest.get("inputs") or {}
    summary = graph_manifest.get("summary") or {}
    quality = graph_manifest.get("quality") or {}
    if (
        substrate_manifest.get("schema") != SUBSTRATE_SCHEMA
        or substrate_manifest.get("round_id") != "0086"
        or substrate_manifest.get("tier") != TIER
        or substrate_manifest.get("row_count") != ROW_COUNT
        or substrate_manifest.get("dimension") != DIMENSION
        or substrate_manifest.get("global_150m_intervals")
        != [[0, ROW_COUNT]]
        or graph_manifest.get("schema") != GRAPH_SCHEMA
        or graph_manifest.get("round_id") != "0091"
        or graph_manifest.get("row_count") != ROW_COUNT
        or graph_manifest.get("input_k") != K
        or graph_inputs.get("eligibility") != eligibility
        or graph_inputs.get("substrate", {}).get("sha256")
        != substrate_manifest_sha256
        or int(summary.get("eligibility_excluded_source_count", -1))
        != EXCLUDED_ROWS
        or int(summary.get("eligibility_retained_row_count", -1))
        != RETAINED_ROWS
        or int(summary.get("retained_positive_source_count", -1))
        != RETAINED_ROWS
        or int(summary.get("zero_degree_retained_source_count", -1)) != 0
        or int(summary.get("valid_canonical_edge_count", -1))
        != RETAINED_ROWS * K
        or summary.get("degree_histogram")
        != {"0": EXCLUDED_ROWS, str(K): RETAINED_ROWS}
        or float(quality.get("mean_recall_at_15_unambiguous", -1.0)) < 0.90
        or float(quality.get("floor", -1.0)) != 0.90
        or int(quality.get("qualification_sample_rows", -1)) != 4_096
        or int(quality.get("qualification_sample_seed", -1)) != 86
    ):
        raise Round0092Error("balanced-150M graph/substrate geometry changed")
    int8 = outputs.get("int8") or {}
    scales = outputs.get("scales") or {}
    if (
        int(int8.get("bytes", -1)) != ROW_COUNT * DIMENSION
        or int(scales.get("bytes", -1)) != ROW_COUNT * 2
        or not int8.get("sha256")
        or not scales.get("sha256")
        or not eligibility.get("sha256")
        or not scale_geometry_signature.get("sha256")
    ):
        raise Round0092Error("balanced-150M reviewed inputs are incomplete")

    config = copy.deepcopy(_R0021_CONFIG)
    config["schema"] = "round0092-production-config-v1"
    config["phrase"] = (
        "balanced 150M MiniLM seed42 native-k15 coverage-aligned rung"
    )
    config["row_universe"] = {
        "corpus_order": ["fineweb", "redpajama", "pile"],
        "rows_per_corpus": 50_000_000,
        "rows": ROW_COUNT,
        "input_dimension": DIMENSION,
        "embedding_dtype": "int8",
        "row_scale_dtype": "<f2",
        "source_intervals_in_150m_namespace": [[0, ROW_COUNT]],
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
            "one exact nonzero int8-plus-fp16-scale vector in the complete "
            "balanced 150M representative universe"
        ),
    }
    config["graph"] = {
        "path": graph_manifest_path,
        "sha256": graph_manifest_sha256,
        "schema": graph_manifest["schema"],
        "k": K,
        "valid_canonical_edges": RETAINED_ROWS * K,
        "positive_source_rows": RETAINED_ROWS,
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
        "qualification": {
            "parts": graph_inputs.get("parts"),
            "quality": dict(quality),
        },
        "mean_recall_at_15_unambiguous": quality[
            "mean_recall_at_15_unambiguous"
        ],
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
        "minimum_train_upd_s": MINIMUM_UPDATES_PER_SECOND,
        "warning_train_upd_s": WARNING_UPDATES_PER_SECOND,
        "performance_warmup_updates": PERFORMANCE_WARMUP_UPDATES,
        "performance_window_updates_max": PERFORMANCE_WINDOW_UPDATES_MAX,
        "performance_windows": PERFORMANCE_WINDOWS,
        "performance_subfloor_patience": 2,
        "performance_abort_latency_at_floor_seconds_max": 50.0,
        "full_run_retry_count": 0,
        "coverage_alignment": {
            "reference_round": REFERENCE_ROUND,
            "reference_retained_positive_sources": (
                REFERENCE_POSITIVE_SOURCES
            ),
            "reference_successful_updates": REFERENCE_SUCCESSFUL_UPDATES,
            "retained_positive_sources": RETAINED_ROWS,
            "formula": (
                f"ceil(500000 * {RETAINED_ROWS} / "
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
            "positive_source_count": RETAINED_ROWS,
            "valid_canonical_edge_count": RETAINED_ROWS * K,
            "graph_degree": (
                "fixed-15-for-every-retained-source;"
                "excluded-sources-degree-zero"
            ),
            "positive_destination_policy": (
                "native-balanced-150m-representative-only-k15;self-removed"
            ),
            "negative_sampling": (
                "uniform-balanced-150m-retained-rows-nonself"
            ),
            "uniform_with_replacement": True,
            "positive_with_replacement": True,
            "weighted_requested": False,
            "weighted_effective": False,
            "source_edge_uniform_equivalent": True,
        },
        "duplicate_control": {
            "eligibility": eligibility,
            "source_copy_or_zero_rows_excluded": EXCLUDED_ROWS,
            "retained_rows": RETAINED_ROWS,
            "full_150m_family_census_reused": True,
        },
        "scale_transition": {
            "decision_sources": {
                "r0080_scale_geometry": dict(scale_geometry_signature),
            },
            "selected_tier": TIER,
            "selection": (
                "next deliberate rung after reviewed 120M noninferiority "
                "and full-map non-density quality"
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
            "treatment": "balanced 120M evidence to balanced 150M rung",
            "source_exposure_confound": (
                "absent because retained graph degree is exactly 15"
            ),
            "density_floor_tuned": False,
        },
    }
    config["transform"] = {
        "status": "registered-downstream-successor",
        "input": "balanced-150m-int8-plus-exact-fp16-row-scale",
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
            "matched retained-120M scale comparison",
            "full balanced-150M representative geometry panel",
            "held-out query projection",
            "fixed-sample render",
            "OOD projection panels",
        ],
        "density_semantics": (
            "use the separately reviewed density-v2 calibration; never "
            "reinterpret the legacy all-row absolute floor"
        ),
    }
    config["decision_thresholds"] = {
        "training_wall_only": True,
        "exact_successful_updates_required": SUCCESSFUL_UPDATES,
        "numerical_skip_counters_must_be_zero": True,
        "geometry_claim_requires_downstream_evaluation": True,
    }
    return config, sha256_bytes(canonical_json(config))
