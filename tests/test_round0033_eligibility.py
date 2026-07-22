from __future__ import annotations

import json

import numpy as np
import pytest

from basemap.artifact_identity import expected_input_signature
from basemap.int8_eligibility import (
    build_int8_eligibility_census,
    find_exact_families,
    fingerprint_encoded_rows,
    load_int8_eligibility,
)
from experiments.census_round0033 import run_job


def test_encoded_census_excludes_all_zero_and_caps_exact_row_scale_families():
    encoded = np.asarray(
        [
            [0, 0, 0, 0],
            [1, -2, 3, -4],
            [1, -2, 3, -4],
            [1, -2, 3, -4],
            [5, 6, 7, 8],
            [0, 0, 0, 0],
            [5, 6, 7, 8],
        ],
        dtype=np.int8,
    )
    scales = np.asarray([1, 12, 12, 13, 9, 2, 9], dtype="<u2")
    records, zero = fingerprint_encoded_rows(encoded, scales)
    found = find_exact_families(records, encoded, scales, zero)
    arrays = found["arrays"]
    summary = found["summary"]
    assert arrays["zero_rows"].tolist() == [0, 5]
    assert arrays["representative_rows"].tolist() == [1, 4]
    assert arrays["family_counts"].tolist() == [2, 2]
    assert arrays["member_rows"].tolist() == [1, 2, 4, 6]
    assert arrays["duplicate_excluded_rows"].tolist() == [2, 6]
    assert arrays["excluded_rows"].tolist() == [0, 2, 5, 6]
    assert summary["retained_row_count"] == 3
    # Row 3 has identical int8 bytes to rows 1/2 but a different fp16 scale.
    assert 3 not in arrays["member_rows"]


def test_fingerprint_hash_is_candidate_only_and_exact_verification_splits_collision():
    encoded = np.asarray([[1, 2], [3, 4], [1, 2]], dtype=np.int8)
    scales = np.asarray([7, 8, 7], dtype="<u2")
    records, zero = fingerprint_encoded_rows(encoded, scales)
    # Force all records into one artificial fingerprint bucket.
    records["h0"] = 1
    records["h1"] = 2
    found = find_exact_families(records, encoded, scales, zero)
    assert found["arrays"]["member_rows"].tolist() == [0, 2]
    assert found["summary"]["fingerprint_collision_splits"] == 1


def test_round0033_artifact_round_trip_and_tamper_rejection(tmp_path):
    encoded = np.asarray(
        [[0, 0], [1, 2], [1, 2], [3, 4], [3, 4], [5, 6]], dtype=np.int8
    )
    scales = np.asarray([1, 2, 2, 3, 3, 4], dtype="<f2")
    int8_path = tmp_path / "embeddings.i8"
    scales_path = tmp_path / "scales.f16"
    int8_path.write_bytes(encoded.tobytes())
    scales_path.write_bytes(scales.tobytes())
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"identity_sha256": "test", "universes": {}}))
    output = tmp_path / "out"
    receipt = build_int8_eligibility_census(
        manifest_path=str(manifest_path),
        int8_path=str(int8_path),
        scales_path=str(scales_path),
        output_root=str(output),
        rows=6,
        dimension=2,
        enforce_registered_identity=False,
    )
    cap = output / "minilm-150m-row-eligibility-v1.npz"
    signature = expected_input_signature(cap)
    loaded = load_int8_eligibility(str(cap), expected_sha256=signature["sha256"], row_count=6)
    assert loaded["excluded_rows"].tolist() == [0, 2, 4]
    assert receipt["summary"]["retained_row_count"] == 3
    with pytest.raises(ValueError, match="SHA-256"):
        load_int8_eligibility(str(cap), expected_sha256="0" * 64, row_count=6)


def test_round0033_has_standalone_slim_runner_entrypoint():
    assert run_job.__module__ == "experiments.census_round0033"
    assert run_job.__name__ == "run_job"
