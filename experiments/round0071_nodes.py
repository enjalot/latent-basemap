"""CPU-only builder for the balanced-90M scale substrate."""
from __future__ import annotations

import os
import resource
import time
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_write_new_json, create_fresh_directory
from basemap.round0049_program import (
    DIMENSION,
    SOURCE_ELIGIBILITY_PATH,
    SOURCE_ELIGIBILITY_SHA256,
    SOURCE_INT8_PATH,
    SOURCE_INT8_SHA256,
    SOURCE_SCALES_PATH,
    SOURCE_SCALES_SHA256,
    write_subset_eligibility,
)
from basemap.round0071_substrate import (
    ELIGIBILITY_SUMMARY,
    INTERVALS,
    ROUND_ID,
    ROW_COUNT,
    ROWS_PER_CORPUS,
    SUBSTRATE_SCHEMA,
    TIER,
    Round0071Error,
    seal,
)
from experiments.round0049_nodes import _copy_intervals


def run_build_substrate(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    output = create_fresh_directory(
        str(job["outputs"][0]),
        label="Round 0071 balanced-90M substrate",
    )
    started = time.monotonic()
    source_signatures = {
        "int8": expected_input_signature(SOURCE_INT8_PATH),
        "scales": expected_input_signature(SOURCE_SCALES_PATH),
        "eligibility": expected_input_signature(SOURCE_ELIGIBILITY_PATH),
    }
    if {
        key: value["sha256"]
        for key, value in source_signatures.items()
    } != {
        "int8": SOURCE_INT8_SHA256,
        "scales": SOURCE_SCALES_SHA256,
        "eligibility": SOURCE_ELIGIBILITY_SHA256,
    }:
        raise Round0071Error("registered 150M substrate inputs changed")

    int8_path = os.path.join(output, "embeddings.i8")
    scales_path = os.path.join(output, "scales.f16")
    eligibility_path = os.path.join(
        output,
        "minilm-balanced-90m-row-eligibility-v1.npz",
    )
    int8_copy = _copy_intervals(
        SOURCE_INT8_PATH,
        int8_path,
        row_bytes=DIMENSION,
        intervals=INTERVALS,
    )
    scales_copy = _copy_intervals(
        SOURCE_SCALES_PATH,
        scales_path,
        row_bytes=2,
        intervals=INTERVALS,
    )
    if (
        int8_copy["bytes"] != ROW_COUNT * DIMENSION
        or scales_copy["bytes"] != ROW_COUNT * 2
    ):
        raise Round0071Error("balanced-90M compact copy has wrong size")
    eligibility = write_subset_eligibility(
        eligibility_path,
        intervals=INTERVALS,
        round_id=ROUND_ID,
        universe="minilm-int8-balanced-90m",
        source_input_key="r0033_eligibility",
    )
    summary = eligibility["metadata"]["summary"]
    if any(
        int(summary.get(key, -1)) != value
        for key, value in ELIGIBILITY_SUMMARY.items()
    ):
        raise Round0071Error("balanced-90M eligibility census changed")
    body = {
        "schema": SUBSTRATE_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "tier": TIER,
        "row_count": ROW_COUNT,
        "dimension": DIMENSION,
        "corpus_order": ["fineweb", "redpajama", "pile"],
        "first_rows_per_corpus": ROWS_PER_CORPUS,
        "global_150m_intervals": [list(value) for value in INTERVALS],
        "compact_row_policy": (
            "concatenate the first 30M rows from each 50M corpus interval"
        ),
        "quantization": (
            "byte-identical subset of R0025 per-row symmetric int8 plus "
            "exact fp16 scale"
        ),
        "exact_family_policy": (
            "recompute representative membership after subset restriction"
        ),
        "inputs": source_signatures,
        "outputs": {
            "int8": int8_copy["signature"],
            "scales": scales_copy["signature"],
            "eligibility": eligibility["signature"],
        },
        "eligibility_summary": summary,
        "eligibility_metadata": eligibility["metadata"],
        "timing": {
            "int8_copy_seconds": int8_copy["wall_seconds"],
            "scales_copy_seconds": scales_copy["wall_seconds"],
            "total_seconds": time.monotonic() - started,
        },
        "peak_rss_gib": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024 ** 2)
        ),
        "training_performed": False,
        "optimizer_updates": 0,
        "scale_decision_made": False,
    }
    manifest = seal(body)
    path = os.path.join(output, "balanced-90m-substrate-v1.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return {**manifest, "manifest": expected_input_signature(path)}


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0071Error("R0071 handler received another queue")
    if job is None or job.get("action") != "build_substrate":
        raise Round0071Error("R0071 requires its exact substrate job")
    return run_build_substrate(active, job)
