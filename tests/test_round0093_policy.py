from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from basemap.artifact_identity import canonical_json, sha256_bytes
from basemap.round0086_program import (
    POLICY_GRID as R0086_POLICY_GRID,
    QUALIFICATION_SCHEMA as R0086_QUALIFICATION_SCHEMA,
)
from basemap.round0093_policy import (
    FALLBACK_POLICY_GRID,
    LOWER_POLICY_GRID,
    MEAN_RECALL_FLOOR,
    POLICY_GRID,
    R0083_SCHEMA,
    R0084_SCHEMA,
    STABILITY_MARGINS,
    Round0093Error,
    select_cell,
    validate_r0083_sensitivity,
    validate_r0084_stability,
    validate_r0086_qualification,
)
from experiments import (
    prepare_round0093_queue,
    round0081_nodes,
    round0093_nodes,
)


def _write_sealed(path: Path, body: dict) -> str:
    value = {
        **body,
        "identity_sha256": sha256_bytes(canonical_json(body)),
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    from basemap.artifact_identity import expected_input_signature

    return expected_input_signature(str(path))["sha256"]


def _r0083_body() -> dict:
    checks = {
        "coords_finite": True,
        "coords_not_collapsed": True,
        "eligible_embeddings_nonzero": True,
        "embeddings_finite": True,
    }
    noninferiority = {
        metric: {"passed": True}
        for metric in STABILITY_MARGINS
    }
    return {
        "schema": R0083_SCHEMA,
        "round_id": "0083",
        "training_performed": True,
        "cells": {
            "16": {
                "nprobe": 16,
                "passed": True,
                "candidate_recall_at_15_unambiguous": (
                    0.8434244791666667
                ),
                "noninferiority_vs_r0061": noninferiority,
                "full_30m_non_density_checks": checks,
            },
        },
        "decision": {
            "verdict": "insensitive-through-lowest-tested-recall",
            "changes_frozen_floor_in_this_round": False,
            "lowest_passing_measured_recall": 0.8434244791666667,
        },
    }


def _r0084_body(*, ffr_delta: float = 0.01) -> dict:
    contrasts = {
        metric: {
            "absolute_delta": (
                ffr_delta if metric == "ffr" else margin / 2
            ),
        }
        for metric, margin in STABILITY_MARGINS.items()
    }
    return {
        "schema": R0084_SCHEMA,
        "round_id": "0084",
        "training_performed": True,
        "paired_metric_contrasts": {"matched": contrasts},
        "full_90m_non_density_checks": {
            "seed42": {"finite": True, "not_collapsed": True},
            "seed43": {"finite": True, "not_collapsed": True},
        },
        "interpretation": {
            "one_paired_seed_contrast": True,
            "estimates_variance": False,
            "establishes_error_bar": False,
            "changes_ladder_decision": False,
        },
    }


def _r0086_body() -> dict:
    nprobe, width = R0086_POLICY_GRID[0]
    key = f"nprobe-{nprobe}-width-{width}"
    selected = {
        "nprobe": nprobe,
        "shortlist_width": width,
        "passes_mean_floor": True,
        "mean_recall_at_15_unambiguous": 0.91,
        "benchmark": {"median_wall_seconds_per_query": 0.001},
    }
    return {
        "schema": R0086_QUALIFICATION_SCHEMA,
        "round_id": "0086",
        "validity_passed": True,
        "training_performed": False,
        "quality": {"floor": 0.90},
        "cells": {key: selected},
        "selected": selected,
        "checks": {"passing_policy_selected": True},
    }


def test_lower_recall_grid_is_bounded_and_selects_measured_fastest() -> None:
    assert MEAN_RECALL_FLOOR == 0.84
    assert LOWER_POLICY_GRID == (
        (32, 128),
        (64, 128),
        (96, 128),
        (32, 256),
        (64, 256),
        (96, 256),
        (32, 384),
        (64, 384),
        (96, 384),
    )
    assert FALLBACK_POLICY_GRID == tuple(R0086_POLICY_GRID)
    assert POLICY_GRID == LOWER_POLICY_GRID + FALLBACK_POLICY_GRID
    cells = {}
    for index, (nprobe, width) in enumerate(POLICY_GRID):
        cells[f"nprobe-{nprobe}-width-{width}"] = {
            "nprobe": nprobe,
            "shortlist_width": width,
            "passes_mean_floor": index in {1, 3},
            "benchmark": {
                "median_wall_seconds_per_query": 0.002 - index * 0.0001,
            },
        }
    selected = select_cell({"cells": cells})
    assert selected is cells["nprobe-32-width-256"]


def test_r0083_direct_treatment_must_support_preregistered_floor(
    tmp_path: Path,
) -> None:
    path = tmp_path / "r0083.json"
    digest = _write_sealed(path, _r0083_body())
    evidence = validate_r0083_sensitivity(
        str(path),
        expected_sha256=digest,
    )
    assert (
        evidence["supporting_cell"][
            "candidate_recall_at_15_unambiguous"
        ]
        > MEAN_RECALL_FLOOR
    )
    failed = _r0083_body()
    failed["cells"]["16"]["noninferiority_vs_r0061"]["ffr"][
        "passed"
    ] = False
    digest = _write_sealed(path, failed)
    with pytest.raises(Round0093Error, match="does not support"):
        validate_r0083_sensitivity(
            str(path),
            expected_sha256=digest,
        )


def test_r0084_is_only_a_conservative_descriptive_screen(
    tmp_path: Path,
) -> None:
    path = tmp_path / "r0084.json"
    digest = _write_sealed(path, _r0084_body())
    evidence = validate_r0084_stability(
        str(path),
        expected_sha256=digest,
    )
    assert evidence["matched_absolute_deltas"]["ffr"] == 0.01
    assert "not a variance estimate" in evidence["interpretation"]
    digest = _write_sealed(path, _r0084_body(ffr_delta=0.020001))
    with pytest.raises(Round0093Error, match="stability screen"):
        validate_r0084_stability(
            str(path),
            expected_sha256=digest,
        )


def test_r0086_old_floor_policy_is_an_authenticated_fallback(
    tmp_path: Path,
) -> None:
    path = tmp_path / "r0086.json"
    digest = _write_sealed(path, _r0086_body())
    evidence = validate_r0086_qualification(
        str(path),
        expected_sha256=digest,
    )
    assert evidence["receipt"]["quality"]["floor"] == 0.90


def test_generic_qualification_records_the_active_floor() -> None:
    source = inspect.getsource(round0081_nodes.run_qualification)
    assert "MEAN_RECALL_FLOOR:.2f" in source
    node_source = inspect.getsource(round0093_nodes.run_qualification)
    assert "validate_r0083_sensitivity" in node_source
    assert "validate_r0084_stability" in node_source
    assert "validate_r0086_qualification" in node_source
    assert '"one_contrast_is_not_variance_or_error_bar": True' in node_source


def test_queue_is_one_bounded_no_training_qualification() -> None:
    source = inspect.getsource(prepare_round0093_queue.prepare_round0093)
    assert source.count('"action": "qualify_lower_recall_150m_policy"') == 1
    assert "gpu_hours_cap=0.5" in source
    assert "p90_seconds = 1_800.0" in source
    assert 'manifest["required_reviews"] = ["0083", "0084", "0086"]' in source
    assert '"no_training": True' in source
    assert '"full_150m_map_evaluation_still_required": True' in source


def test_preparer_materializes_exact_one_node_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from basemap.artifact_identity import expected_input_signature

    round_file = tmp_path / "round-0093-2026-07-28.md"
    round_file.write_text("---\nstatus: issued\n---\n", encoding="utf-8")
    substrate_path = tmp_path / "substrate.json"
    filtered_path = tmp_path / "filtered.ivfpq"
    filter_receipt_path = tmp_path / "filter-receipt.json"
    runtime_path = tmp_path / "runtime.json"
    for path, payload in (
        (substrate_path, b"substrate"),
        (filtered_path, b"index"),
        (filter_receipt_path, b"receipt"),
        (runtime_path, b"runtime"),
    ):
        path.write_bytes(payload)
    substrate = expected_input_signature(str(substrate_path))
    filtered = expected_input_signature(str(filtered_path))
    filter_receipt = expected_input_signature(str(filter_receipt_path))
    runtime = expected_input_signature(str(runtime_path))
    r0083_signature = {
        "canonical_path": "/evidence/r0083.json",
        "bytes": 123,
        "sha256": "3" * 64,
    }
    r0084_signature = {
        "canonical_path": "/evidence/r0084.json",
        "bytes": 124,
        "sha256": "4" * 64,
    }
    r0086_signature = {
        "canonical_path": "/evidence/r0086.json",
        "bytes": 126,
        "sha256": "6" * 64,
    }
    monkeypatch.setattr(
        prepare_round0093_queue,
        "ROUND_FILE_GLOB",
        str(tmp_path / "round-0093-*.md"),
    )
    monkeypatch.setattr(
        prepare_round0093_queue,
        "validate_r0083_sensitivity",
        lambda *_args, **_kwargs: {
            "signature": r0083_signature,
            "supporting_cell": {
                "candidate_recall_at_15_unambiguous": 0.8434244791666667,
            },
        },
    )
    monkeypatch.setattr(
        prepare_round0093_queue,
        "validate_r0084_stability",
        lambda *_args, **_kwargs: {
            "signature": r0084_signature,
            "matched_absolute_deltas": {
                metric: margin / 2
                for metric, margin in STABILITY_MARGINS.items()
            },
            "margins": dict(STABILITY_MARGINS),
        },
    )
    monkeypatch.setattr(
        prepare_round0093_queue,
        "validate_r0086_qualification",
        lambda *_args, **_kwargs: {
            "signature": r0086_signature,
            "receipt": {
                "substrate": substrate,
                "filtered_index": filtered,
            },
        },
    )
    monkeypatch.setattr(
        prepare_round0093_queue,
        "_require_review",
        lambda path, *, expected_sha256, required_text: {
            "canonical_path": str(Path(path).resolve()),
            "bytes": 100,
            "sha256": expected_sha256,
        },
    )

    def fake_fresh_directory(path: str, *, label: str) -> str:
        assert label
        target = Path(path)
        target.mkdir(parents=True)
        return str(target)

    def fake_ensure_directory(path: str) -> str:
        target = Path(path)
        target.mkdir(parents=True)
        return str(target)

    def fake_atomic_json(
        path: str,
        payload: dict,
        *,
        immutable: bool,
    ) -> None:
        assert immutable is True
        Path(path).write_text(
            json.dumps(payload, sort_keys=True),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        prepare_round0093_queue,
        "create_fresh_directory",
        fake_fresh_directory,
    )
    monkeypatch.setattr(
        prepare_round0093_queue,
        "ensure_data_directory",
        fake_ensure_directory,
    )
    monkeypatch.setattr(
        prepare_round0093_queue,
        "atomic_write_new_json",
        fake_atomic_json,
    )
    queue = prepare_round0093_queue.prepare_round0093(
        release_sha="7" * 40,
        r0083_review_path="/reviews/r0083.md",
        r0083_review_sha256="8" * 64,
        r0083_sensitivity_path=r0083_signature["canonical_path"],
        r0083_sensitivity_sha256=r0083_signature["sha256"],
        r0084_review_path="/reviews/r0084.md",
        r0084_review_sha256="9" * 64,
        r0084_seed_contrast_path=r0084_signature["canonical_path"],
        r0084_seed_contrast_sha256=r0084_signature["sha256"],
        r0086_review_path="/reviews/r0086.md",
        r0086_review_sha256="a" * 64,
        r0086_qualification_path=r0086_signature["canonical_path"],
        r0086_qualification_sha256=r0086_signature["sha256"],
        substrate_manifest_path=substrate["canonical_path"],
        substrate_manifest_sha256=substrate["sha256"],
        filtered_index_path=filtered["canonical_path"],
        filtered_index_sha256=filtered["sha256"],
        filter_receipt_path=filter_receipt["canonical_path"],
        filter_receipt_sha256=filter_receipt["sha256"],
        runtime_spec_path=runtime["canonical_path"],
        runtime_spec_sha256=runtime["sha256"],
        queue_root=str(tmp_path / "queue"),
    )
    manifest = json.loads(Path(queue).read_text(encoding="utf-8"))
    assert manifest["release_sha"] == "7" * 40
    assert manifest["gpu_hours_cap"] == 0.5
    assert manifest["p90_gpu_seconds"]["total"] == 1_800.0
    assert manifest["required_reviews"] == ["0083", "0084", "0086"]
    assert len(manifest["jobs"]) == 1
    job = manifest["jobs"][0]
    assert job["action"] == "qualify_lower_recall_150m_policy"
    assert job["p90_wall_s"] == 1_800.0
    assert job["node_policy"] == {
        "gpu_required": True,
        "training_performed": False,
    }
    assert {item["canonical_path"] for item in job["expected_inputs"]} >= {
        substrate["canonical_path"],
        filtered["canonical_path"],
        filter_receipt["canonical_path"],
        runtime["canonical_path"],
    }
