"""CPU-only builders for the decision-ready R0065 scale substrates."""
from __future__ import annotations

import os
import resource
import time
from typing import Any, Mapping

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0049_program import (
    DIMENSION,
    SOURCE_ELIGIBILITY_PATH,
    SOURCE_ELIGIBILITY_SHA256,
    SOURCE_INT8_PATH,
    SOURCE_INT8_SHA256,
    SOURCE_SCALES_PATH,
    SOURCE_SCALES_SHA256,
    _seal,
    write_subset_eligibility,
)
from basemap.round0065_substrates import (
    ROUND_ID,
    SUBSTRATE_SCHEMA,
    Round0065Error,
    subset_spec,
)
from experiments.round0049_nodes import _copy_intervals


def run_build_substrate(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    tier = str(job["tier"])
    spec = subset_spec(tier)
    intervals = tuple(spec["intervals"])
    row_count = int(spec["row_count"])
    output = create_fresh_directory(
        str(job["outputs"][0]),
        label=f"Round 0065 balanced-{tier} substrate",
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
        raise Round0065Error("registered 150M substrate inputs changed")

    int8_path = os.path.join(output, "embeddings.i8")
    scales_path = os.path.join(output, "scales.f16")
    eligibility_path = os.path.join(
        output,
        f"minilm-balanced-{tier}-row-eligibility-v1.npz",
    )
    int8_copy = _copy_intervals(
        SOURCE_INT8_PATH,
        int8_path,
        row_bytes=DIMENSION,
        intervals=intervals,
    )
    scales_copy = _copy_intervals(
        SOURCE_SCALES_PATH,
        scales_path,
        row_bytes=2,
        intervals=intervals,
    )
    if (
        int8_copy["bytes"] != row_count * DIMENSION
        or scales_copy["bytes"] != row_count * 2
    ):
        raise Round0065Error(
            f"balanced-{tier} compact copy has wrong size"
        )
    eligibility = write_subset_eligibility(
        eligibility_path,
        intervals=intervals,
        round_id=ROUND_ID,
        universe=f"minilm-int8-balanced-{tier}",
        source_input_key="r0033_eligibility",
    )
    summary = eligibility["metadata"]["summary"]
    if any(
        int(summary.get(key, -1)) != value
        for key, value in spec["eligibility_summary"].items()
    ):
        raise Round0065Error(
            f"balanced-{tier} eligibility census changed"
        )
    body = {
        "schema": SUBSTRATE_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "tier": tier,
        "row_count": row_count,
        "dimension": DIMENSION,
        "corpus_order": ["fineweb", "redpajama", "pile"],
        "first_rows_per_corpus": spec["first_rows_per_corpus"],
        "global_150m_intervals": [list(value) for value in intervals],
        "compact_row_policy": (
            "concatenate the registered prefix from each 50M corpus interval"
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
    manifest = _seal(body)
    path = os.path.join(
        output,
        f"balanced-{tier}-substrate-v1.json",
    )
    atomic_write_new_json(path, manifest, immutable=True)
    return {
        **manifest,
        "manifest": expected_input_signature(path),
    }


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0065Error("R0065 handler received another queue")
    selected = job if job is not None else active.get("job") or {}
    if selected.get("action") != "build_substrate":
        raise Round0065Error(
            f"unknown R0065 action: {selected.get('action')!r}"
        )
    return run_build_substrate(active, selected)
