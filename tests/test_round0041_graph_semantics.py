from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from basemap.artifact_identity import (
    canonical_json,
    ordered_array_sha256,
    sha256_bytes,
)
from basemap.round0034_pipeline import validate_eligibility_view
from basemap.round0041_program import ELIGIBILITY_SCHEMA, read_training_semantics


def test_fp16_eligibility_schema_is_explicitly_admitted() -> None:
    excluded = np.asarray([2, 5], dtype=np.int64)
    view = {
        "metadata": {
            "schema": ELIGIBILITY_SCHEMA,
            "row_count": 8,
            "summary": {
                "excluded_row_count": 2,
                "retained_row_count": 6,
            },
        },
        "signature": {"sha256": "a" * 64},
        "zero_rows": np.empty(0, dtype=np.int64),
        "excluded_rows": excluded,
        "duplicate_excluded_rows": excluded,
        "duplicate_representative_rows": np.asarray([0, 1], dtype=np.int64),
    }
    result = validate_eligibility_view(
        view, row_count=8, expected_schema=ELIGIBILITY_SCHEMA
    )
    assert result["retained_row_count"] == 6


def test_training_semantics_reads_actual_pipeline_stamp(tmp_path: Path) -> None:
    receipt = {
        "train_stats": {
            "budget_satisfied": True,
            "n_pos_edges": 12,
            "positive_lr_optimizer_steps": 7,
            "pipeline_pipeline": "hybrid",
            "pipeline_sampler_class": "HostStreamEdgeSampler",
            "pipeline_positive_sampling": "uniform",
            "pipeline_multiplicity_positive_source_sampling": "edge-uniform",
            "pipeline_multiplicity_positive_destinations": "original",
            "pipeline_multiplicity_graph_degree": "variable",
        }
    }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    result = read_training_semantics(str(path))
    assert result["pipeline"] == "hybrid"
    assert result["effective_positive_edges"] == 12
    assert result["successful_updates"] == 7


def test_census_style_array_hashes_are_order_sensitive() -> None:
    rows = np.asarray([1, 3, 7], dtype=np.int64)
    assert ordered_array_sha256(rows) != ordered_array_sha256(rows[::-1])
    body = {"schema": ELIGIBILITY_SCHEMA, "row_count": 10}
    sealed = {**body, "identity_sha256": sha256_bytes(canonical_json(body))}
    assert sealed["identity_sha256"] == sha256_bytes(canonical_json(body))
