"""Coverage-aligned training contract for the R0064-selected next rung."""
from __future__ import annotations

import copy
import math
from typing import Any, Mapping

from .artifact_identity import canonical_json, sha256_bytes
from .round0021_program import TRAIN_CONFIG as _R0021_CONFIG
from .round0034_pipeline import GRAPH_SCHEMA
from .round0049_program import DIMENSION, K
from .round0065_substrates import SUBSETS, SUBSTRATE_SCHEMA


ROUND_ID = "0068"
SEED = 42
REFERENCE_ROUND = "0046"
REFERENCE_POSITIVE_SOURCES = 29_781_619
REFERENCE_SUCCESSFUL_UPDATES = 500_000
PIPELINE_SCHEMA = "round0068-host-int8-selected-canonical-pipeline-v1"
SAMPLER_CLASS = "HostInt8SelectedCanonicalSampler"


class Round0068Error(RuntimeError):
    """The selected next-rung training capabilities changed."""


def successful_updates_for_tier(tier: str) -> int:
    try:
        retained = int(
            SUBSETS[tier]["eligibility_summary"]["retained_row_count"]
        )
    except KeyError as exc:
        raise Round0068Error(f"unknown training tier {tier!r}") from exc
    return math.ceil(
        REFERENCE_SUCCESSFUL_UPDATES
        * retained
        / REFERENCE_POSITIVE_SOURCES
    )


def train_config_from_capabilities(
    *,
    tier: str,
    graph_manifest: Mapping[str, Any],
    graph_manifest_path: str,
    graph_manifest_sha256: str,
    substrate_manifest: Mapping[str, Any],
    substrate_manifest_path: str,
    substrate_manifest_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Derive the exact selected-tier production training configuration."""
    try:
        spec = SUBSETS[tier]
    except KeyError as exc:
        raise Round0068Error(f"unknown training tier {tier!r}") from exc
    row_count = int(spec["row_count"])
    excluded = int(spec["eligibility_summary"]["excluded_row_count"])
    retained = int(spec["eligibility_summary"]["retained_row_count"])
    valid_edges = retained * K
    updates = successful_updates_for_tier(tier)
    outputs = substrate_manifest.get("outputs") or {}
    eligibility = outputs.get("eligibility") or {}
    summary = graph_manifest.get("summary") or {}
    if (
        substrate_manifest.get("schema") != SUBSTRATE_SCHEMA
        or substrate_manifest.get("round_id") != "0065"
        or substrate_manifest.get("tier") != tier
        or int(substrate_manifest.get("row_count", -1)) != row_count
        or int(substrate_manifest.get("dimension", -1)) != DIMENSION
        or substrate_manifest.get("global_150m_intervals")
        != [list(value) for value in spec["intervals"]]
        or graph_manifest.get("schema") != GRAPH_SCHEMA
        or graph_manifest.get("round_id") != "0067"
        or graph_manifest.get("tier") != tier
        or int(graph_manifest.get("row_count", -1)) != row_count
        or int(graph_manifest.get("input_k", -1)) != K
        or graph_manifest.get("inputs", {}).get("eligibility")
        != eligibility
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
    ):
        raise Round0068Error(
            "selected graph/substrate geometry changed"
        )
    int8 = outputs.get("int8") or {}
    scales = outputs.get("scales") or {}
    if (
        int(int8.get("bytes", -1)) != row_count * DIMENSION
        or int(scales.get("bytes", -1)) != row_count * 2
        or not int8.get("sha256")
        or not scales.get("sha256")
        or not eligibility.get("sha256")
    ):
        raise Round0068Error("selected feature capabilities are incomplete")

    config = copy.deepcopy(_R0021_CONFIG)
    config["schema"] = "round0068-production-config-v1"
    config["phrase"] = (
        f"balanced {tier} MiniLM seed42 native-k15 coverage-aligned rung"
    )
    config["row_universe"] = {
        "corpus_order": ["fineweb", "redpajama", "pile"],
        "rows_per_corpus": spec["first_rows_per_corpus"],
        "rows": row_count,
        "input_dimension": DIMENSION,
        "embedding_dtype": "int8",
        "row_scale_dtype": "<f2",
        "source_intervals_in_150m_namespace": [
            list(value) for value in spec["intervals"]
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
            f"balanced {tier} subset"
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
    }
    config["optimizer"]["seed"] = SEED
    config["optimizer"]["use_amp"] = "bf16"
    config["optimizer"]["successful_positive_lr_updates"] = updates
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
            "retained_positive_sources": retained,
            "formula": (
                f"ceil(500000 * {retained} / "
                f"{REFERENCE_POSITIVE_SOURCES})"
            ),
            "successful_updates": updates,
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
                f"native-balanced-{tier}-representative-only-k15;"
                "self-removed"
            ),
            "negative_sampling": (
                f"uniform-balanced-{tier}-retained-rows-nonself"
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
            "decision_source": "reviewed R0064 matched scale evaluation",
            "selected_tier": tier,
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
                f"balanced 60M evaluation to selected balanced {tier} rung"
            ),
            "source_exposure_confound": (
                "absent because retained graph degree is exactly 15"
            ),
        },
    }
    config["transform"] = {
        "status": "registered-downstream-successor",
        "input": f"balanced-{tier}-int8-plus-exact-fp16-row-scale",
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
            "selected-tier absolute geometry panel",
            "held-out query projection",
            "fixed-sample render",
            "OOD projection panels",
        ],
    }
    config["decision_thresholds"] = {
        "training_wall_only": True,
        "exact_successful_updates_required": updates,
        "numerical_skip_counters_must_be_zero": True,
        "geometry_claim_requires_downstream_evaluation": True,
    }
    return config, sha256_bytes(canonical_json(config))
