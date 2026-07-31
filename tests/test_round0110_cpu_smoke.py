"""Bounded CPU smoke for R0110 transform -> density receipt -> decision."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from basemap.artifact_identity import expected_input_signature
from basemap.round0108_evaluation import (
    CORE_SCHEMA as R0108_CORE_SCHEMA,
    DECISION_SCHEMA as R0108_DECISION_SCHEMA,
    OOD_SCHEMA as R0108_OOD_SCHEMA,
    seal,
    validate_seal,
)
from experiments import round0110_nodes


def _write_sealed(path: Path, body: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(seal(body), sort_keys=True), encoding="utf-8")


def _core_body(
    *,
    schema: str,
    round_id: str,
    map_key: str,
    density_passed: bool,
) -> dict[str, Any]:
    checks = {
        "coordinates_finite_and_noncollapsed": True,
        "density_v2_clears_registered_jina_floor": density_passed,
        "every_language_ffr_at_least_0_40_of_pooled_english": True,
        "global_ffr_at_least_0_40": True,
        "global_recall50_strictly_exceeds_recall10": True,
    }
    return {
        "schema": schema,
        "round_id": round_id,
        "map_key": map_key,
        "metrics": {
            "global": {
                "ffr": 0.6,
                "recall_at_10": 0.1,
                "recall_at_50_of_high10": 0.2,
            },
            "density_v2": {"correlation": 0.16},
        },
        "decision": {"checks": checks, "passed": all(checks.values())},
    }


def _ood_body(*, schema: str, round_id: str, map_key: str) -> dict[str, Any]:
    return {
        "schema": schema,
        "round_id": round_id,
        "map_key": map_key,
        "embedding_prompt": "raw",
        "prompt_applied": False,
        "language_cells": {
            "pol_Latn": {
                "probe": {
                    "recall_at_10": 0.1,
                    "recall_at_50_of_high10": 0.2,
                }
            }
        },
        "headline_decision": {
            "passed": True,
            "polish_to_in_mix_median_ratio": 1.0,
        },
    }


def test_matched_density_and_revised_decision_cpu_smoke(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    rows = 256
    dimensions = 8
    anchors = 128
    monkeypatch.setattr(round0110_nodes, "MATCHED_SOURCE_ROWS", rows)
    monkeypatch.setattr(
        round0110_nodes, "MATCHED_SOURCE_DIMENSION", dimensions
    )
    monkeypatch.setattr(round0110_nodes, "MATCHED_ANCHORS", anchors)

    rng = np.random.default_rng(110)
    source_path = tmp_path / "source.npy"
    np.save(
        source_path,
        rng.normal(size=(rows, dimensions)).astype(np.float16),
    )
    source_signature = expected_input_signature(str(source_path))
    census_path = tmp_path / "census.json"
    census_path.write_text("{}", encoding="utf-8")
    census_signature = expected_input_signature(str(census_path))
    census_bundle = {
        "receipt": {"source": source_signature},
        "signature": {
            "kind": "file",
            "canonical_path": str(tmp_path / "census.npz"),
            "bytes": 1,
            "sha256": "a" * 64,
        },
        "arrays": {
            "excluded_rows": np.empty(0, dtype=np.int64),
            "representative_rows": np.empty(0, dtype=np.int64),
            "family_counts": np.empty(0, dtype=np.int64),
        },
    }
    monkeypatch.setattr(
        round0110_nodes, "load_jina_census", lambda _path: census_bundle
    )

    anchor_rows = np.arange(anchors, dtype=np.int64)
    high_radius = np.linspace(0.5, 2.0, anchors, dtype=np.float64)
    reference_path = tmp_path / "reference.npz"
    np.savez(
        reference_path,
        anchor_ids=anchor_rows,
        r_hd=high_radius,
        key=np.asarray("smoke-reference"),
    )
    reference_signature = expected_input_signature(str(reference_path))
    calibration_root = tmp_path / "calibration"
    calibration_root.mkdir()
    calibration_arrays = calibration_root / "jina-density-calibration-arrays.npz"
    np.savez(
        calibration_arrays,
        seed42__high_radius=high_radius,
        seed43__high_radius=high_radius,
    )
    calibration_path = calibration_root / "jina-density-calibration.json"
    _write_sealed(
        calibration_path,
        {
            "schema": "round0108-jina-density-v2-calibration-v1",
            "round_id": "0108",
            "representative_reference_key": "smoke-reference",
            "anchors": {
                "compact_rows_sha256": round0110_nodes.ordered_array_sha256(
                    anchor_rows
                ),
                "global_rows_sha256": round0110_nodes.ordered_array_sha256(
                    anchor_rows
                ),
            },
            "arrays": expected_input_signature(str(calibration_arrays)),
            "floor_calibration": {"registered_floor": 0.17589389755990817},
        },
    )

    class SmokeModel:
        def transform(self, values, batch_size: int) -> np.ndarray:
            blocks = [
                np.asarray(values[start : start + batch_size], dtype=np.float32)
                for start in range(0, len(values), batch_size)
            ]
            matrix = np.concatenate(blocks)
            return matrix[:, :2]

    def bundle(seed: int) -> dict[str, Any]:
        model_signature = {
            "kind": "file",
            "canonical_path": f"/smoke/model-{seed}.pt",
            "bytes": 1,
            "sha256": f"{seed:064x}"[-64:],
        }
        return {
            "model": SmokeModel(),
            "train": {"model": model_signature},
            "train_signature": {
                **model_signature,
                "canonical_path": f"/smoke/train-{seed}.json",
            },
            "config_signature": {
                **model_signature,
                "canonical_path": f"/smoke/config-{seed}.json",
            },
        }

    monkeypatch.setattr(
        round0110_nodes,
        "load_reviewed_model",
        lambda **_kwargs: bundle(42),
    )
    monkeypatch.setattr(
        round0110_nodes,
        "_seed43_model",
        lambda **_kwargs: bundle(43),
    )

    import basemap.panel_v2 as panel_v2

    def smoke_self_knn(
        _corpus,
        selected,
        k,
        _config,
        **_kwargs,
    ):
        assert len(selected) == anchors
        distances = np.repeat(high_radius[:, None], k, axis=1)
        neighbors = np.tile(np.arange(k, dtype=np.int64), (anchors, 1))
        return neighbors, distances, {"backend": "cpu-smoke"}

    monkeypatch.setattr(panel_v2, "_self_knn", smoke_self_knn)
    matched_root = tmp_path / "matched"
    matched = round0110_nodes.run_matched_density(
        {"manifest": {"release_sha": "b" * 40}},
        {
            "outputs": [str(matched_root)],
            "calibration_output": str(calibration_root),
            "census_receipt": str(census_path),
            "census_receipt_sha256": census_signature["sha256"],
            "representative_reference": str(reference_path),
            "representative_reference_sha256": reference_signature["sha256"],
            "graph_manifest": "/smoke/graph.json",
            "graph_manifest_sha256": "c" * 64,
            "seed42_train_output": "/smoke/seed42",
            "seed43_train_output": "/smoke/seed43",
        },
    )
    assert matched["checks"]["both_seeds_clear_matched_floor"] is True
    with (matched_root / "matched-density.json").open(
        encoding="utf-8"
    ) as handle:
        validate_seal(json.load(handle), label="R0110 smoke matched receipt")

    seed42_core_path = tmp_path / "seed42-core.json"
    seed42_ood_path = tmp_path / "seed42-ood.json"
    seed42_decision_path = tmp_path / "seed42-decision.json"
    seed43_core_root = tmp_path / "seed43-core"
    seed43_ood_root = tmp_path / "seed43-ood"
    _write_sealed(
        seed42_core_path,
        _core_body(
            schema=R0108_CORE_SCHEMA,
            round_id="0108",
            map_key="r0107-diverse-jina-25m-seed42",
            density_passed=False,
        ),
    )
    _write_sealed(
        seed42_ood_path,
        _ood_body(
            schema=R0108_OOD_SCHEMA,
            round_id="0108",
            map_key="r0107-diverse-jina-25m-seed42",
        ),
    )
    _write_sealed(
        seed42_decision_path,
        {
            "schema": R0108_DECISION_SCHEMA,
            "round_id": "0108",
            "map_key": "r0107-diverse-jina-25m-seed42",
            "atlas_quality_capability_released": False,
        },
    )
    _write_sealed(
        seed43_core_root / "core-geometry.json",
        _core_body(
            schema=round0110_nodes.CORE_SCHEMA,
            round_id="0110",
            map_key=round0110_nodes.MAP_KEY,
            density_passed=False,
        ),
    )
    _write_sealed(
        seed43_ood_root / "ood-evaluation.json",
        _ood_body(
            schema=round0110_nodes.OOD_SCHEMA,
            round_id="0110",
            map_key=round0110_nodes.MAP_KEY,
        ),
    )
    decision_root = tmp_path / "decision"
    decision = round0110_nodes.run_decision(
        {"manifest": {"release_sha": "b" * 40}},
        {
            "outputs": [str(decision_root)],
            "seed42_decision": str(seed42_decision_path),
            "seed42_core": str(seed42_core_path),
            "seed42_ood": str(seed42_ood_path),
            "core_output": str(seed43_core_root),
            "ood_output": str(seed43_ood_root),
            "matched_density_output": str(matched_root),
        },
    )
    assert decision["two_seed_quality_capability_released"] is False
    assert (
        decision[
            "matched_fineweb_qualified_atlas_capability_released"
        ]
        is True
    )
    assert decision[
        "full_diverse_universe_density_under_original_floor"
    ]["status"] == "failed"
    assert (
        decision[
            "full_diverse_universe_density_under_original_floor"
        ]["overridden_by_matched_fineweb_cell"]
        is False
    )
    with (decision_root / "two-seed-decision.json").open(
        encoding="utf-8"
    ) as handle:
        validate_seal(json.load(handle), label="R0110 smoke decision receipt")
