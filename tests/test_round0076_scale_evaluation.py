import inspect
import json
from pathlib import Path

import pytest

from basemap.round0064_evaluation import (
    MODEL_SPECS,
    Round0064Error,
    expected_retained_rows_for_scale,
    seal,
)
from experiments import map_registry, prepare_round0076_queue
from experiments.round0076_nodes import (
    MATCHED_NONINFERIORITY_MARGINS,
    _noninferiority,
    run_comparison,
    run_renders,
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
    round_id: str,
    schema: str,
    key: str,
    rows: int = 29_781_754,
    delta: float = 0.0,
) -> dict:
    checks = _checks()
    return seal({
        "schema": schema,
        "round_id": round_id,
        "map_key": key,
        "eligibility": {"sha256": "e" * 64},
        "scientific_universe": {
            "rows": rows,
            "substrate": "balanced-30m" if rows < 90_000_000 else "balanced-90m",
            "row_namespace": (
                "compact ascending balanced-30m retained rows"
                if rows < 90_000_000
                else "compact ascending balanced-90m retained rows"
            ),
            "excluded_rows_in_scoring": False,
        },
        "panel": {
            "n": rows,
            "anchor_hash": "shared-30m-anchors",
            "ffr": 0.48 + delta,
            "density": 0.10 + delta,
            "purity": {
                "k256": 1.05 + delta,
                "k1024": 0.88 + delta,
            },
            "recall@k": 0.001,
            "provenance": {
                "hiD_reference_key": (
                    "shared-30m-reference"
                    if rows < 90_000_000
                    else "full-90m-reference"
                ),
            },
        },
        "recall_at_10": 0.001,
        "recall_at_50": 0.003,
        "projection": {"proj_ffr": 0.44 + delta},
        "decision_checks": checks,
        "absolute_selector_passed": all(checks.values()),
    })


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_r0075_model_identity_is_registered_exactly() -> None:
    assert MODEL_SPECS["r0075-90m"] == {
        "round_id": "0075",
        "receipt_schema": "round0075-train-receipt-v1",
        "config_schema": "round0075-production-config-v1",
        "rows": 90_000_000,
        "retained_rows": 88_945_313,
        "updates": 1_493_293,
        "sampler_class": "HostInt8Balanced90mCanonicalSampler",
    }


def test_90m_substrate_retained_rows_come_from_registered_model() -> None:
    assert expected_retained_rows_for_scale(90_000_000) == 88_945_313
    with pytest.raises(Round0064Error, match="not registered exactly"):
        expected_retained_rows_for_scale(75_000_000)


def test_noninferiority_uses_registered_relative_margins() -> None:
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
    comparison = _noninferiority(
        treatment,
        control,
        control_label="control",
    )
    assert all(item["passed"] for item in comparison.values())
    treatment["density"] -= 0.000001
    comparison = _noninferiority(
        treatment,
        control,
        control_label="control",
    )
    assert comparison["density"]["passed"] is False


def test_comparison_uses_two_scale_controls_and_not_absolute_density(
    tmp_path: Path,
) -> None:
    specs = {
        "control_30m": (
            "0064",
            "round0064-registered-panel-v1",
            "r0061-30m-on-30m",
            0.0,
            29_781_754,
        ),
        "rung_45m": (
            "0069",
            "round0069-registered-panel-v1",
            "r0068-45m-on-30m",
            0.002,
            29_781_754,
        ),
        "rung_60m": (
            "0064",
            "round0064-registered-panel-v1",
            "r0063-60m-on-30m",
            0.004,
            29_781_754,
        ),
        "treatment_90m_matched": (
            "0076",
            "round0076-registered-panel-v1",
            "r0075-90m-on-30m",
            0.003,
            29_781_754,
        ),
        "treatment_90m_full": (
            "0076",
            "round0076-registered-panel-v1",
            "r0075-90m-on-90m",
            0.003,
            88_945_313,
        ),
    }
    panels = {}
    for name, (round_id, schema, key, delta, rows) in specs.items():
        path = tmp_path / "inputs" / name / "panel.json"
        _write_json(path, _panel(
            round_id=round_id,
            schema=schema,
            key=key,
            rows=rows,
            delta=delta,
        ))
        panels[name] = {"path": str(path), "key": key, "schema": schema}

    anchor_path = tmp_path / "inputs/anchor.json"
    _write_json(anchor_path, seal({
        "schema": "round0074-duplicate-anchor-leverage-v1",
        "round_id": "0074",
        "legacy_density_exactly_replayed": True,
        "interpretation": {
            "classification": "duplicate-heavy-anchor-leverage-supported",
            "calibrates_density_threshold": False,
        },
    }))
    from basemap.artifact_identity import expected_input_signature

    receipt = run_comparison({}, {
        "outputs": [str(tmp_path / "output")],
        "panels": panels,
        "anchor_leverage": str(anchor_path),
        "anchor_leverage_sha256": expected_input_signature(
            str(anchor_path)
        )["sha256"],
    })
    assert receipt["decision"]["90m_supported_as_deliberate_ladder_rung"]
    assert receipt["decision"]["prepare_120m_search_and_graph_if_true"]
    assert receipt["decision"]["train_120m_without_separate_round"] is False
    assert (
        receipt["density_semantics"][
            "legacy_absolute_floor_used_for_decision"
        ]
        is False
    )
    assert (
        receipt["density_semantics"]["legacy_absolute_floor_reported"]
        is False
    )
    assert set(
        receipt["same_row_30m_comparison"][
            "90m_vs_30m_noninferiority"
        ]
    ) == set(MATCHED_NONINFERIORITY_MARGINS)
    assert set(
        receipt["same_row_30m_comparison"][
            "90m_vs_60m_noninferiority"
        ]
    ) == set(MATCHED_NONINFERIORITY_MARGINS)


def _registry_entry(
    tmp_path: Path,
    *,
    density_semantics: str | None,
    queue_name: str = "queue",
) -> dict:
    round_dir = tmp_path / "round-0076"
    queue_dir = round_dir / queue_name
    artifacts = queue_dir / "artifacts"
    coordinates = artifacts / "coordinates-r0075-90m"
    panel_path = artifacts / "panel-r0075-90m/panel.json"
    render_path = artifacts / "semantic-renders/r0075-90m-on-90m.png"
    chunk = coordinates / "chunk-00000"
    chunk.mkdir(parents=True)
    render_path.parent.mkdir(parents=True, exist_ok=True)
    render_path.write_bytes(b"png")
    (chunk / "coordinates.npy").write_bytes(b"coords")
    _write_json(queue_dir / "queue.json", {"release_sha": "a" * 40})
    _write_json(coordinates / "actual-transform.json", {
        "schema": "round0036-transform-capability-v1",
        "map_key": "r0075-90m-on-90m",
        "model": {"sha256": "b" * 64},
        "row_accounting": {
            "all_rows": 90_000_000,
            "retained_representatives": 88_945_313,
        },
    })
    _write_json(panel_path, {
        "schema": "round0076-registered-panel-v1",
        "map_key": "r0075-90m-on-90m",
        "absolute_selector_passed": False,
        "decision_checks": _checks(),
        "panel": {
            "ffr": 0.5,
            "density": 0.1,
            "purity": {"k256": 1.0, "k1024": 0.9},
            "formula_version": "panel_v2.2-2026-07-15",
        },
        "projection": {"proj_ffr": 0.45},
    })
    definition = {
        "key": "r0075-90m-on-90m",
        "label": "r0075-balanced-90m-seed42",
        "coordinates": "coordinates-r0075-90m",
        "panel": "panel-r0075-90m/panel.json",
        "render": "semantic-renders/r0075-90m-on-90m.png",
        "training_round": "0075",
        "panel_schema": "round0076-registered-panel-v1",
    }
    if density_semantics is not None:
        definition["density_semantics"] = density_semantics
    _write_json(
        artifacts / "semantic-renders/scale-map-definitions.json",
        {
            "schema": "scale-map-definitions-v1",
            "round_id": "0076",
            "maps": [definition],
        },
    )
    entries = map_registry.scan_scale_evaluation_round(
        round_dir,
        {},
        queue_dir=queue_dir,
    )
    assert len(entries) == 1
    return entries[0]


def test_registry_applies_reviewed_representative_density_semantics(
    tmp_path: Path,
) -> None:
    entry = _registry_entry(
        tmp_path,
        density_semantics="representative-relative-v1",
    )
    assert entry["capability_candidate"] is True
    assert (
        entry["scientific_status"]
        == "representative-non-density-selector-pass"
    )
    assert entry["panel"]["decision_checks_all_pass"] is True
    assert entry["panel"]["raw_decision_checks_all_pass"] is False
    assert entry["panel"]["legacy_absolute_selector_passed"] is False


def test_legacy_registry_semantics_preserve_registered_absolute_decision(
    tmp_path: Path,
) -> None:
    entry = _registry_entry(tmp_path, density_semantics=None)
    assert entry["capability_candidate"] is False
    assert entry["scientific_status"] == (
        "same-domain-selector-failed-diagnostic"
    )


def test_registry_scan_uses_latest_preserved_queue_attempt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    expected = _registry_entry(
        tmp_path,
        density_semantics="representative-relative-v1",
        queue_name="queue-attempt-2",
    )
    stale = tmp_path / "round-0076/queue"
    (stale / "artifacts").mkdir(parents=True)
    _write_json(stale / "queue.json", {"release_sha": "0" * 40})
    alias = tmp_path / "round-0076-attempt-2"
    alias.mkdir()
    (alias / "queue").symlink_to(
        "../round-0076/queue-attempt-2",
        target_is_directory=True,
    )
    monkeypatch.setattr(map_registry, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(
        map_registry,
        "LEDGER_DIR",
        tmp_path / "missing-ledger",
    )
    monkeypatch.setattr(
        map_registry,
        "CHECKPOINT_DIR",
        tmp_path / "missing-checkpoints",
    )

    registry = map_registry.scan()
    entries = [
        item
        for item in registry["maps"]
        if item.get("round_id") == "0076"
        and item.get("kind") == "round-map"
    ]
    assert len(entries) == 1
    assert entries[0]["map_id"] == expected["map_id"]
    assert entries[0]["release_sha"] == "a" * 40


def test_round0076_queue_is_bounded_no_training_and_reuses_geometry() -> None:
    source = inspect.getsource(prepare_round0076_queue.prepare_round0076)
    assert "gpu_hours_cap=2.0" in source
    assert source.count('action="transform"') == 2
    assert source.count('action="panel"') == 2
    assert source.count('action="high_d_reference"') == 1
    assert source.count('action="ood"') == 1
    assert 'action="train"' not in source
    assert '"required_reviews"] = ["0069", "0071", "0074", "0075"]' in source
    assert '"90m_noninferiority_controls": ["30m", "60m"]' in source
    assert '"legacy_absolute_floor_is_decision_gating": False' in source
    assert '"train_120m_without_separate_round": False' in source


def test_render_path_avoids_rehashing_large_int8_substrates() -> None:
    source = inspect.getsource(run_renders)
    assert "load_substrate" not in source
    assert "load_int8_eligibility" not in source
    assert "_selector(" in source
