from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments import map_registry
from experiments.projection_gallery import build_projection_explorers


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    runs = tmp_path / "runs"
    labs = tmp_path / "labs"
    checkpoints = tmp_path / "checkpoints"
    artifacts = runs / "round-0042/queue/artifacts"
    panel_dir = artifacts / "panel"
    coords_dir = artifacts / "base/coordinates"
    semantic_dir = artifacts / "base/semantic-renders"
    panel_dir.mkdir(parents=True)
    (coords_dir / "chunk-00000").mkdir(parents=True)
    semantic_dir.mkdir(parents=True)
    labs.mkdir()
    checkpoints.mkdir()

    base = np.arange(24, dtype=np.float32).reshape(12, 2)
    np.save(coords_dir / "chunk-00000/coordinates.npy", base)
    (coords_dir / "actual-transform.json").write_text("{}\n")
    np.save(semantic_dir / "sample-semantic-ids.npy", np.arange(12, dtype=np.int64))

    np.savez(
        panel_dir / "custom-coordinates.npz",
        probe_corpus_coords=np.arange(16, dtype=np.float32).reshape(8, 2),
        probe_query_coords=np.arange(4, dtype=np.float32).reshape(2, 2),
        control_corpus_coords=np.zeros((8, 2), dtype=np.float32),
        control_query_coords=np.zeros((2, 2), dtype=np.float32),
        probe_corpus_ids=np.array([f"c{i}" for i in range(8)]),
        probe_query_ids=np.array(["q0", "q1"]),
        control_corpus_rows=np.arange(8, dtype=np.int64),
        control_query_rows=np.arange(2, dtype=np.int64),
    )
    panel = {
        "schema": "universality-panel-v1",
        "round_id": "0042",
        "map": {
            "label": "fixture-map",
            "model": {"canonical_path": "/tmp/model.pt", "sha256": "a" * 64},
            "coordinate_receipt": {
                "canonical_path": str(coords_dir / "actual-transform.json"),
                "sha256": "b" * 64,
            },
        },
        "probes": {
            "custom": {
                "status": "included",
                "verdict": "amber",
                "retention": 0.6,
                "probe": {"corpus_rows": 8, "query_rows": 2, "ffr": 0.3},
                "matched_control": {"ffr": 0.5},
            },
            "excluded": {"status": "excluded"},
        },
    }
    (panel_dir / "universality-panel-v1.json").write_text(json.dumps(panel))
    queue = {"release_sha": "c" * 40}
    (runs / "round-0042/queue/queue.json").write_text(json.dumps(queue))
    (labs / "round-0042-2026-07-22.md").write_text(
        '---\nround_id: "0042"\nstatus: issued\n---\n'
    )
    (labs / "review-0042-2026-07-22.md").write_text(
        '---\nround_id: "0042"\nstatus: accepted\n---\n'
    )
    return runs, labs, checkpoints


def test_scan_discovers_included_projection_with_base_context(
    tmp_path: Path, monkeypatch
) -> None:
    runs, labs, checkpoints = _fixture(tmp_path)
    monkeypatch.setattr(map_registry, "RUNS_DIR", runs)
    monkeypatch.setattr(map_registry, "LEDGER_DIR", labs)
    monkeypatch.setattr(map_registry, "CHECKPOINT_DIR", checkpoints)

    registry = map_registry.scan()
    assert registry["schema"] == "basemap-map-registry-v2"
    assert registry["counts"]["projection_maps"] == 1
    entry = next(item for item in registry["maps"] if item["kind"] == "projection-map")
    assert entry["map_id"] == "round-0042-fixture-map-custom-projection"
    assert entry["evidence_status"] == "review:accepted"
    assert entry["projection"]["ffr"] == 0.3
    assert entry["projection"]["control_ffr"] == 0.5
    assert entry["projection"]["retention"] == 0.6
    assert entry["base_coordinates"]["dir"].endswith("/base/coordinates")
    assert entry["base_sample_ids"]["path"].endswith("sample-semantic-ids.npy")


def test_projection_explorer_publishes_sampled_binary_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    runs, labs, checkpoints = _fixture(tmp_path)
    monkeypatch.setattr(map_registry, "RUNS_DIR", runs)
    monkeypatch.setattr(map_registry, "LEDGER_DIR", labs)
    monkeypatch.setattr(map_registry, "CHECKPOINT_DIR", checkpoints)
    registry = map_registry.scan()
    site = tmp_path / "site"

    built = build_projection_explorers(registry, site)
    assert len(built) == 1
    root = site / "projections/round-0042-fixture-map-custom-projection"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["sample"]["base_source_rows"] == 12
    assert manifest["sample"]["probe_corpus_source_rows"] == 8
    assert manifest["sample"]["probe_query_source_rows"] == 2
    assert (root / manifest["files"]["base"]["path"]).stat().st_size == 12 * 2 * 4
    assert (root / manifest["files"]["corpus"]["path"]).stat().st_size == 8 * 2 * 4
    assert "training-map context" in (root / "index.html").read_text()
    assert (site / "round-0042/index.html").is_file()


def test_registry_index_links_projection_explorer(tmp_path: Path, monkeypatch) -> None:
    runs, labs, checkpoints = _fixture(tmp_path)
    monkeypatch.setattr(map_registry, "RUNS_DIR", runs)
    monkeypatch.setattr(map_registry, "LEDGER_DIR", labs)
    monkeypatch.setattr(map_registry, "CHECKPOINT_DIR", checkpoints)
    site = tmp_path / "site"
    monkeypatch.setattr(map_registry, "SITE_DIR", site)
    registry = map_registry.scan()

    map_registry.publish(registry)
    index = (site / "index.html").read_text()
    assert "Projection maps" in index
    assert "round-0042-fixture-map-custom-projection" in index
    assert (site / "projections/round-0042-fixture-map-custom-projection/index.html").is_file()
