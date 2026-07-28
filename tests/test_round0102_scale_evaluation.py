import inspect
import json
from pathlib import Path

import numpy as np

from basemap.round0064_evaluation import MODEL_SPECS, seal
from experiments import prepare_round0102_queue
from experiments.round0102_nodes import (
    DENSITY_V2_FLOOR,
    MATCHED_NONINFERIORITY_MARGINS,
    _map_anchor_family_sizes,
    _noninferiority,
    _pearson_log_radius,
    run_comparison,
    run_registry,
)


def _checks(*, density_passed: bool = False) -> dict[str, bool]:
    return {
        "ffr_at_least_0_40": True,
        "density_at_least_0_60": density_passed,
        "purity_k256_at_least_0_50": True,
        "purity_k1024_at_least_0_50": True,
        "heldout_projection_beats_untrained_floor": True,
        "recall_at_50_exceeds_recall_at_10": True,
        "coords_finite": True,
        "coords_not_collapsed": True,
        "embeddings_finite": True,
        "eligible_embeddings_nonzero": True,
    }


def _panel(
    *,
    schema: str,
    key: str,
    rows: int,
    substrate: str,
    delta: float = 0.0,
    projection_ffr: float | None = None,
) -> dict:
    matched_120m = rows == 118_067_492
    return seal({
        "schema": schema,
        "round_id": "0080" if key.startswith("r0079") else "0102",
        "map_key": key,
        "eligibility": {
            "sha256": "f" * 64 if matched_120m else "a" * 64,
        },
        "scientific_universe": {
            "rows": rows,
            "substrate": substrate,
            "row_namespace": f"compact ascending {substrate} retained rows",
            "excluded_rows_in_scoring": False,
        },
        "panel": {
            "n": rows,
            "anchor_hash": (
                "shared-120m-anchors"
                if matched_120m
                else "full-150m-anchors"
            ),
            "ffr": 0.48 + delta,
            "density": 0.10 + delta,
            "purity": {
                "k256": 1.05 + delta,
                "k1024": 0.88 + delta,
            },
            "recall@k": 0.001,
            "provenance": {
                "hiD_reference_key": (
                    "shared-120m-reference"
                    if matched_120m
                    else "full-150m-reference"
                ),
            },
        },
        "recall_at_10": 0.001,
        "recall_at_50": 0.003,
        "projection": {
            "proj_ffr": (
                0.44 + delta
                if projection_ffr is None
                else projection_ffr
            ),
        },
        "decision_checks": _checks(),
        "absolute_selector_passed": False,
    })


def _density(*, matched_pass: bool = True, full_pass: bool = True) -> dict:
    return seal({
        "schema": "round0102-density-v2-evaluation-v1",
        "round_id": "0102",
        "registered_floor": DENSITY_V2_FLOOR,
        "cells": {
            "matched_120m": {
                "density_v2": DENSITY_V2_FLOOR + 0.001,
                "passed_registered_floor": matched_pass,
            },
            "full_150m": {
                "density_v2": DENSITY_V2_FLOOR + 0.001,
                "passed_registered_floor": full_pass,
            },
        },
    })


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _comparison_fixture(
    tmp_path: Path,
    *,
    projection_ffr: float | None = None,
    density_passed: bool = True,
) -> dict:
    control = tmp_path / "control/panel.json"
    matched = tmp_path / "matched/panel.json"
    full = tmp_path / "full/panel.json"
    density = tmp_path / "density/density-v2-evaluation.json"
    _write_json(control, _panel(
        schema="round0080-registered-panel-v1",
        key="r0079-120m-on-120m",
        rows=118_067_492,
        substrate="balanced-120m",
    ))
    _write_json(matched, _panel(
        schema="round0102-registered-panel-v1",
        key="r0101-150m-on-120m",
        rows=118_067_492,
        substrate="balanced-120m",
        delta=0.002,
        projection_ffr=projection_ffr,
    ))
    _write_json(full, _panel(
        schema="round0102-registered-panel-v1",
        key="r0101-150m-on-150m",
        rows=147_221_757,
        substrate="balanced-150m",
        delta=0.002,
    ))
    _write_json(
        density,
        _density(
            matched_pass=density_passed,
            full_pass=density_passed,
        ),
    )
    return run_comparison({}, {
        "outputs": [str(tmp_path / "output")],
        "control_panel": str(control),
        "matched_panel": str(matched),
        "full_panel": str(full),
        "density_v2": str(density),
    })


def test_r0101_model_identity_is_registered_exactly() -> None:
    assert MODEL_SPECS["r0101-150m"] == {
        "round_id": "0101",
        "receipt_schema": "round0101-train-receipt-v1",
        "config_schema": "round0101-production-config-v1",
        "rows": 150_000_000,
        "retained_rows": 147_221_757,
        "updates": 2_471_689,
        "sampler_class": "HostInt8Balanced150mCanonicalSampler",
    }


def test_noninferiority_has_only_preregistered_decision_metrics() -> None:
    assert MATCHED_NONINFERIORITY_MARGINS == {
        "ffr": 0.02,
        "purity_k256": 0.05,
        "purity_k1024": 0.05,
    }
    control = {
        "ffr": 0.48,
        "density": 0.10,
        "purity_k256": 1.05,
        "purity_k1024": 0.88,
        "projection_ffr": 0.44,
        "recall_at_10": 0.001,
        "recall_at_50": 0.003,
    }
    treatment = dict(control)
    for metric, margin in MATCHED_NONINFERIORITY_MARGINS.items():
        treatment[metric] = control[metric] - margin
    comparison = _noninferiority(treatment, control)
    assert set(comparison) == set(MATCHED_NONINFERIORITY_MARGINS)
    assert all(item["passed"] for item in comparison.values())
    treatment["ffr"] -= 0.0000004
    assert _noninferiority(treatment, control)["ffr"]["passed"] is False


def test_projection_ffr_is_diagnostic_before_observing_150m(
    tmp_path: Path,
) -> None:
    receipt = _comparison_fixture(tmp_path, projection_ffr=-10.0)
    assert receipt["decision"]["150m_supported_as_deliberate_ladder_rung"]
    diagnostic = receipt["same_row_120m_comparison"][
        "projection_ffr_diagnostic"
    ]
    assert diagnostic["delta"] < -10
    assert diagnostic["decision_gating"] is False
    assert "0.0209" in diagnostic["reason"]


def test_comparison_gates_on_fixed_density_v2_not_legacy_density(
    tmp_path: Path,
) -> None:
    passed = _comparison_fixture(tmp_path / "passed")
    assert passed["same_row_120m_comparison"]["passed"]
    assert passed["full_150m_non_density_checks_passed"]
    assert passed["density_v2"]["passed"]
    assert passed["decision"]["150m_supported_as_deliberate_ladder_rung"]
    assert (
        passed["density_semantics"]["legacy_absolute_floor_reported"]
        is False
    )
    assert (
        passed["density_semantics"][
            "legacy_absolute_floor_used_for_decision"
        ]
        is False
    )

    failed = _comparison_fixture(
        tmp_path / "failed",
        density_passed=False,
    )
    assert failed["density_v2"]["passed"] is False
    assert (
        failed["decision"]["150m_supported_as_deliberate_ladder_rung"]
        is False
    )


def test_density_v2_math_and_duplicate_family_filter() -> None:
    high = np.linspace(0.1, 1.0, 1000)
    assert _pearson_log_radius(high, high * 3.0) > 0.999999
    eligibility = {
        "representative_rows": np.array([3, 10, 20]),
        "family_counts": np.array([2, 17, 4]),
    }
    assert _map_anchor_family_sizes(
        np.array([0, 3, 10, 20, 21]),
        eligibility,
    ).tolist() == [1, 2, 17, 4, 1]


def test_round0102_queue_is_bounded_no_training_and_preregistered() -> None:
    source = inspect.getsource(prepare_round0102_queue.prepare_round0102)
    assert "gpu_hours_cap=4.5" in source
    assert source.count('action="transform"') == 2
    assert source.count('action="panel"') == 2
    assert source.count('action="high_d_reference"') == 1
    assert source.count('action="density_v2"') == 1
    assert source.count('action="ood"') == 1
    assert 'action="train"' not in source
    for review in ("0025", "0033", "0080", "0084", "0085", "0101"):
        assert f'"{review}"' in source
    assert '"150m_noninferiority_control": "120m"' in source
    assert '"decision_gating": False' in source
    assert '"threshold_recalibrated": False' in source


def test_r0080_inputs_bind_exact_reviewed_hashes(monkeypatch) -> None:
    assert prepare_round0102_queue.R0080_ARTIFACTS == (
        "/data/latent-basemap/runs/round-0080/queue/artifacts"
    )
    assert prepare_round0102_queue.R0080_REVIEWED_SHA256[
        "panel-r0079-120m/panel.json"
    ] == "bb16e22530fd04f488b67ef9632eaede4da6148f494bb8120ec4146d322971c3"
    relative_path = "panel-r0079-120m/panel.json"
    monkeypatch.setattr(
        prepare_round0102_queue,
        "expected_input_signature",
        lambda _path: {
            "canonical_path": "/wrong/attempt/panel.json",
            "sha256": "0" * 64,
        },
    )
    try:
        prepare_round0102_queue._reviewed_r0080_signature(relative_path)
    except RuntimeError as exc:
        assert "reviewed R0080 artifact changed" in str(exc)
    else:
        raise AssertionError("unreviewed R0080 artifact was accepted")


def test_round0102_discovers_the_one_issued_dated_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    old = tmp_path / "round-0102-2026-07-27.md"
    current = tmp_path / "round-0102-2026-07-28.md"
    old.write_text("---\nstatus: draft\n---\n", encoding="utf-8")
    current.write_text("---\nstatus: issued\n---\n", encoding="utf-8")
    monkeypatch.setattr(
        prepare_round0102_queue,
        "ROUND_FILE_GLOB",
        str(tmp_path / "round-0102-*.md"),
    )
    assert prepare_round0102_queue._require_issued_round() == str(current)
    old.write_text("---\nstatus: issued\n---\n", encoding="utf-8")
    try:
        prepare_round0102_queue._require_issued_round()
    except RuntimeError as exc:
        assert "exactly one issued round document" in str(exc)
    else:
        raise AssertionError("multiple issued R0102 contracts were accepted")


def test_registry_binds_immutable_snapshot_without_gating_on_mutable_view(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from experiments import map_registry

    label = "r0101-balanced-150m-seed42"
    entries = [{
        "map_id": "base",
        "round_id": "0102",
        "kind": "round-map",
        "map_label": label,
    }]
    for probe in ("dadabase", "trec-covid", "code", "science", "latin"):
        entries.append({
            "map_id": f"projection-{probe}",
            "round_id": "0102",
            "kind": "projection-map",
            "base_map": label,
            "projection": {"probe": probe},
        })
    registry = {
        "schema": "basemap-map-registry-v1",
        "generated_utc": "2026-07-28T00:00:00+00:00",
        "counts": {"maps": len(entries)},
        "maps": entries,
    }
    monkeypatch.setattr(map_registry, "REGISTRY_PATH", tmp_path / "maps.json")
    monkeypatch.setattr(
        map_registry,
        "HISTORY_DIR",
        tmp_path / "registry-history",
    )
    monkeypatch.setattr(map_registry, "scan", lambda: registry)
    monkeypatch.setattr(map_registry, "publish", lambda _registry: None)

    receipt = run_registry({}, {
        "outputs": [str(tmp_path / "round-snapshot")],
    })

    assert receipt["immutable_registry_snapshot"]["sha256"]
    assert receipt["mutable_registry_after_publish"]["sha256"]
    assert (
        receipt["immutable_registry_snapshot"]["sha256"]
        != receipt["mutable_registry_after_publish"]["sha256"]
    )
    assert receipt["content_addressed_history_snapshot_if_new"]["sha256"]
    assert receipt["mutable_view_equality_is_nongating"] is True


def test_registry_treats_density_v2_maps_as_representative_only() -> None:
    from experiments import map_registry

    source = inspect.getsource(map_registry.scan_scale_evaluation_round)
    assert '"density-v2-fixed-floor-plus-legacy-diagnostic"' in source
    assert '"density_at_least_0_60"' in source
