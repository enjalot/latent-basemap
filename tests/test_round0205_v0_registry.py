from __future__ import annotations

import json
from pathlib import Path

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_copy_new
from basemap import round0113_prompt_contrast as prompt_contract
from basemap.round0205_v0_registry import (
    CANDIDATE_ID,
    CAPABILITY,
    EXPECTED_COORDINATES_SHA256,
    MAP_DEFINITION_SCHEMA,
    ROUND_ID,
    canonical_metrics,
    named_ood_failures,
)
from experiments import map_registry
from experiments import round0205_nodes as nodes


BUNDLE = Path(
    "/data/latent-basemap/runs/round-0204/queue/artifacts/"
    "basemap-jina-v5-nano-en-2m-v0-release-bundle-v1/release-bundle.json"
)
SOURCE_COORDINATES = Path(
    "/data/latent-basemap/runs/round-0115/queue-attempt-2/artifacts/"
    "document/evaluation/coordinates.npy"
)
TRAIN_RECEIPT = Path(
    "/data/latent-basemap/runs/round-0115/queue-attempt-2/artifacts/"
    "document/train/train-receipt.json"
)
SCORE = Path(
    "/data/latent-basemap/runs/round-0115/queue-attempt-2/artifacts/"
    "document/evaluation/score.json"
)


def _bundle() -> dict:
    return prompt_contract.read_sealed(str(BUNDLE), label="test bundle")


def test_reviewed_bundle_exposes_exact_registry_metrics_and_failures() -> None:
    metrics = canonical_metrics(_bundle())
    assert metrics == {
        "density": 0.2327,
        "ffr": 0.6383,
        "purity_k256": 0.9831874938550783,
        "purity_k1024": 0.9357,
        "projection_ffr": 0.5755,
        "heldout_recall_at_10": 0.01135,
    }
    assert named_ood_failures(_bundle()) == [
        "code", "culture", "danish", "government", "latin", "science",
        "trec-covid",
    ]


def test_registry_scanner_discovers_only_exact_scoped_v0(tmp_path: Path) -> None:
    round_dir = tmp_path / f"round-{ROUND_ID}"
    output = round_dir / "queue" / "artifacts" / CAPABILITY
    coordinates = output / "coordinates" / "chunk-00000" / "coordinates.npy"
    coordinates.parent.mkdir(parents=True)
    atomic_copy_new(SOURCE_COORDINATES, coordinates)
    bundle = _bundle()
    metrics = canonical_metrics(bundle)
    definition = prompt_contract.seal({
        "schema": MAP_DEFINITION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": "f" * 40,
        "map_id": CANDIDATE_ID,
        "candidate_id": CANDIDATE_ID,
        "training_round": "0115",
        "evaluation_round": "0115",
        "release_bundle": expected_input_signature(BUNDLE),
        "coordinates": expected_input_signature(coordinates),
        "source_coordinates": expected_input_signature(SOURCE_COORDINATES),
        "train_receipt": expected_input_signature(TRAIN_RECEIPT),
        "score": {"canonical_path": "/tmp/score.json"},
        "model": {"sha256": "1" * 64},
        "population": {
            "rows": 1_993_761,
            "input_dimension": 768,
            "output_dimension": 2,
            "embedding_convention": "Document: ",
        },
        "architecture": {
            "name": "residual_bottleneck",
            "hidden_dimension": 2048,
            "low_dim_kernel": "legacy_lp",
        },
        "actual_pipeline": {
            "pipeline": "host_weighted_jina_prompt_contrast",
            "sampler_class": "PromptWeightedJinaSampler",
        },
        "metrics": {
            **metrics,
            "formula_version": "panel_v2.2-2026-07-15",
            "all_six_seed42_gates_pass": True,
            "all_four_seeds_pass_all_six_gates": True,
        },
        "limitations": {
            "canonical_seed42_named_ood_failures": named_ood_failures(bundle),
            "canonical_seed42_named_ood_failure_count": 7,
            "universal_ood_quality_claim": False,
        },
        "release_scope": {
            "local_registry_ready": True,
            "local_v0_release_registered": True,
            "production_readiness_claimed": False,
            "huggingface_upload_performed": False,
        },
    })
    (output / "map-definition.json").write_text(json.dumps(definition))
    entries = map_registry.scan_v0_release_map(round_dir, {ROUND_ID: {}})
    assert len(entries) == 1
    entry = entries[0]
    assert entry["map_id"] == CANDIDATE_ID
    assert entry["local_v0_release_registered"] is True
    assert entry["production_ready"] is False
    assert entry["universal_ood_ready"] is False
    assert entry["coordinates"]["receipt_sha256"] == EXPECTED_COORDINATES_SHA256
    assert entry["panel"]["decision_checks_all_pass"] is True

    definition["release_scope"]["huggingface_upload_performed"] = True
    (output / "map-definition.json").write_text(json.dumps(definition))
    assert map_registry.scan_v0_release_map(round_dir, {}) == []


def test_real_node_mints_snapshot_and_local_page(
    tmp_path: Path, monkeypatch
) -> None:
    runs = tmp_path / "runs"
    queue = runs / f"round-{ROUND_ID}" / "queue"
    artifacts = queue / "artifacts"
    artifacts.mkdir(parents=True)
    output = artifacts / CAPABILITY
    registry_path = tmp_path / "maps.json"
    history = tmp_path / "registry-history"
    site = tmp_path / "site"
    labs = tmp_path / "labs"
    labs.mkdir()
    monkeypatch.setattr(map_registry, "RUNS_DIR", runs)
    monkeypatch.setattr(map_registry, "REGISTRY_PATH", registry_path)
    monkeypatch.setattr(map_registry, "HISTORY_DIR", history)
    monkeypatch.setattr(map_registry, "SITE_DIR", site)
    monkeypatch.setattr(map_registry, "LEDGER_DIR", labs)
    monkeypatch.setattr(
        nodes,
        "ensure_data_directory",
        lambda path: str(Path(path).mkdir(parents=True, exist_ok=True) or path),
    )
    nodes.run_job(
        {"manifest": {"round_id": ROUND_ID, "release_sha": "f" * 40}},
        {
            "action": "register_v0_locally",
            "r0204_bundle": str(BUNDLE),
            "r0115_score": str(SCORE),
            "r0115_train_receipt": str(TRAIN_RECEIPT),
            "outputs": [str(output)],
        },
    )
    publication = prompt_contract.read_sealed(
        str(output / "registry-publication.json"), label="test publication"
    )
    assert publication["map_id"] == CANDIDATE_ID
    assert publication["checks"]["immutable_snapshot_minted"] is True
    assert publication["checks"]["huggingface_upload_performed"] is False
    assert registry_path.is_file()
    assert len(list(history.glob("maps-*.json"))) == 1
    assert CANDIDATE_ID in (site / f"round-{ROUND_ID}" / "index.html").read_text()
