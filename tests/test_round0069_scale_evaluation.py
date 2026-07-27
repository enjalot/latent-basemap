import inspect
import json
from pathlib import Path

import pytest

from basemap.round0064_evaluation import seal
from experiments import map_registry, prepare_round0069_queue
from experiments.round0069_nodes import run_comparison


def _panel(
    *,
    round_id: str,
    schema: str,
    key: str,
    rows: int = 29_781_754,
    density: float = 0.10,
    delta: float = 0.0,
) -> dict:
    checks = {
        "ffr_at_least_0_40": True,
        "density_at_least_0_60": density >= 0.60,
        "purity_k256_at_least_0_50": True,
        "purity_k1024_at_least_0_50": True,
        "heldout_projection_beats_untrained_floor": True,
        "recall_at_50_exceeds_recall_at_10": True,
        "coords_finite": True,
        "coords_not_collapsed": True,
        "embeddings_finite": True,
        "eligible_embeddings_nonzero": True,
    }
    body = {
        "schema": schema,
        "round_id": round_id,
        "map_key": key,
        "eligibility": {"sha256": "e" * 64},
        "scientific_universe": {
            "rows": rows,
            "substrate": (
                "balanced-30m" if rows == 29_781_754
                else "balanced-45m"
            ),
        },
        "panel": {
            "n": rows,
            "anchor_hash": "shared-anchors",
            "ffr": 0.48 + delta,
            "density": density,
            "purity": {
                "k256": 1.05 + delta,
                "k1024": 0.88 + delta,
            },
            "recall@k": 0.001,
            "provenance": {"hiD_reference_key": "shared-reference"},
        },
        "recall_at_10": 0.001,
        "recall_at_50": 0.003,
        "projection": {"proj_ffr": 0.44 + delta},
        "decision_checks": checks,
        "absolute_selector_passed": all(checks.values()),
    }
    return seal(body)


def test_comparison_keeps_legacy_density_diagnostic_and_blocks_120m(
    tmp_path: Path,
) -> None:
    panel_specs = {
        "control_30m": (
            "r0061-30m-on-30m",
            "round0064-registered-panel-v1",
            "0064",
            0.0991,
            0.0,
            29_781_754,
        ),
        "treatment_45m_matched": (
            "r0068-45m-on-30m",
            "round0069-registered-panel-v1",
            "0069",
            0.09,
            0.005,
            29_781_754,
        ),
        "upper_60m_matched": (
            "r0063-60m-on-30m",
            "round0064-registered-panel-v1",
            "0064",
            0.0819,
            0.01,
            29_781_754,
        ),
        "treatment_45m_full": (
            "r0068-45m-on-45m",
            "round0069-registered-panel-v1",
            "0069",
            0.08,
            0.005,
            44_598_360,
        ),
    }
    job_panels = {}
    matched_density = {}
    for name, (key, schema, round_id, density, delta, rows) in (
        panel_specs.items()
    ):
        root = tmp_path / name
        root.mkdir()
        path = root / "panel.json"
        path.write_text(json.dumps(_panel(
            round_id=round_id,
            schema=schema,
            key=key,
            rows=rows,
            density=density,
            delta=delta,
        )))
        job_panels[name] = {
            "path": str(path),
            "key": key,
            "schema": schema,
        }
        if name != "treatment_45m_full":
            matched_density[key] = {
                "density": {"global_correlation": density}
            }

    density_root = tmp_path / "density"
    density_root.mkdir()
    density_body = {
        "schema": "round0069-matched-density-diagnostic-v1",
        "round_id": "0069",
        "maps": matched_density,
    }
    (density_root / "density-diagnostic.json").write_text(
        json.dumps(seal(density_body))
    )
    receipt = run_comparison(
        {},
        {
            "outputs": [str(tmp_path / "comparison")],
            "panels": job_panels,
            "density_diagnostic": str(density_root),
        },
    )
    assert (
        receipt["decision"]["45m_supported_as_deliberate_ladder_rung"]
        is True
    )
    assert receipt["decision"]["balanced_density_gate_calibrated"] is False
    assert receipt["decision"]["advance_directly_to_120m"] is False
    assert (
        receipt["decision"]["45m_legacy_absolute_selector_passed"]
        is False
    )


def test_round0069_queue_is_bounded_reuses_control_and_never_trains() -> None:
    source = inspect.getsource(prepare_round0069_queue.prepare_round0069)
    assert "gpu_hours_cap=1.5" in source
    assert source.count('action="transform"') == 2
    assert source.count('action="panel"') == 2
    assert source.count('action="high_d_reference"') == 1
    assert source.count('action="density_diagnostic"') == 1
    assert 'action="train"' not in source
    assert "R0064_REFERENCE_30" in source
    assert "R0064_SCALE_COMPARISON" in source
    assert 'scale_comparison["sha256"]' in source
    assert '"training_performed"] = False' in source


def test_registry_discovers_declared_scale_map(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = tmp_path / "runs"
    ledger = tmp_path / "ledger"
    checkpoints = tmp_path / "checkpoints"
    round_dir = runs / "round-0069"
    artifacts = round_dir / "queue/artifacts"
    coordinates = artifacts / "coordinates-r0068-45m"
    panel_root = artifacts / "panel-r0068-45m"
    renders = artifacts / "semantic-renders"
    chunk = coordinates / "chunk-00000"
    for path in (ledger, checkpoints, chunk, panel_root, renders):
        path.mkdir(parents=True, exist_ok=True)
    (round_dir / "queue/queue.json").write_text(json.dumps({
        "release_sha": "a" * 40,
    }))
    (ledger / "round-0069-2026-07-27.md").write_text(
        "---\nround_id: \"0069\"\nstatus: issued\n---\n"
    )
    (chunk / "coordinates.npy").write_bytes(b"coords")
    (coordinates / "actual-transform.json").write_text(json.dumps({
        "schema": "round0036-transform-capability-v1",
        "map_key": "r0068-45m-on-45m",
        "model": {"sha256": "b" * 64},
        "row_accounting": {
            "all_rows": 45_000_000,
            "retained_representatives": 44_598_360,
        },
    }))
    (panel_root / "panel.json").write_text(json.dumps({
        "schema": "round0069-registered-panel-v1",
        "map_key": "r0068-45m-on-45m",
        "map": {"label": "r0068-balanced-45m-seed42"},
        "absolute_selector_passed": False,
        "panel": {
            "ffr": 0.5,
            "density": 0.1,
            "purity": {"k256": 1.0, "k1024": 0.9},
            "formula_version": "panel_v2.2-2026-07-15",
        },
        "projection": {"proj_ffr": 0.45},
    }))
    (renders / "r0068-45m-on-45m.png").write_bytes(b"png")
    (renders / "scale-map-definitions.json").write_text(json.dumps({
        "schema": "scale-map-definitions-v1",
        "round_id": "0069",
        "maps": [{
            "key": "r0068-45m-on-45m",
            "label": "r0068-balanced-45m-seed42",
            "coordinates": "coordinates-r0068-45m",
            "panel": "panel-r0068-45m/panel.json",
            "render": "semantic-renders/r0068-45m-on-45m.png",
            "training_round": "0068",
            "panel_schema": "round0069-registered-panel-v1",
        }],
    }))
    monkeypatch.setattr(map_registry, "RUNS_DIR", runs)
    monkeypatch.setattr(map_registry, "LEDGER_DIR", ledger)
    monkeypatch.setattr(map_registry, "CHECKPOINT_DIR", checkpoints)
    registry = map_registry.scan()
    entries = [
        item for item in registry["maps"]
        if item.get("round_id") == "0069"
        and item.get("kind") == "round-map"
    ]
    assert len(entries) == 1
    assert entries[0]["map_label"] == "r0068-balanced-45m-seed42"
    assert entries[0]["capability_candidate"] is False
