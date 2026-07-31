"""CPU smoke of R0119's transform -> density -> decision path."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from basemap.artifact_identity import (
    expected_input_signature,
    ordered_array_sha256,
)
from basemap.round0108_evaluation import seal, validate_seal
from experiments import round0119_nodes
from experiments.round0085_nodes import density_v2_calibration


def _write_json(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return expected_input_signature(str(path))


def test_density_localization_end_to_end_cpu_smoke(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    rows = 256
    dimensions = 8
    anchors_count = 128
    monkeypatch.setattr(round0119_nodes, "SOURCE_ROWS", rows)
    monkeypatch.setattr(round0119_nodes, "SOURCE_DIMENSION", dimensions)
    monkeypatch.setattr(round0119_nodes, "REPRESENTATIVE_ROWS", rows)
    monkeypatch.setattr(round0119_nodes, "ANCHORS", anchors_count)

    rng = np.random.default_rng(119)
    source_path = tmp_path / "source.npy"
    np.save(
        source_path,
        rng.normal(size=(rows, dimensions)).astype(np.float16),
    )
    source_signature = expected_input_signature(str(source_path))
    census_path = tmp_path / "census.json"
    census_signature = _write_json(
        census_path, {"source": source_signature}
    )
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
        round0119_nodes, "load_jina_census", lambda _path: census_bundle
    )
    reference_path = tmp_path / "representative-reference.npz"
    np.savez(reference_path, placeholder=np.asarray([1]))
    reference_signature = expected_input_signature(str(reference_path))

    anchors = np.arange(anchors_count, dtype=np.int64)
    global_rows = anchors.copy()
    high_radius = np.linspace(
        0.5, 2.0, anchors_count, dtype=np.float64
    )
    family_sizes = np.ones(anchors_count, dtype=np.int64)
    matched_arrays_path = tmp_path / "matched-arrays.npz"
    np.savez(
        matched_arrays_path,
        anchor_compact_rows=anchors,
        anchor_global_rows=global_rows,
        high_radius=high_radius,
        family_sizes=family_sizes,
    )
    matched_arrays_signature = expected_input_signature(
        str(matched_arrays_path)
    )
    frozen_low_radius = np.repeat(
        high_radius[:, None], round0119_nodes.K_DENSITY, axis=1
    ).mean(1)
    _, frozen_bootstrap, frozen_null = density_v2_calibration(
        high_radius,
        frozen_low_radius,
        bootstrap_draws=1_000,
        bootstrap_seed=10_801,
        null_draws=1_000,
        null_seed=10_802,
    )
    calibration_arrays_path = tmp_path / "calibration-arrays.npz"
    np.savez(
        calibration_arrays_path,
        seed42__high_radius=high_radius,
        seed42__low_radius=frozen_low_radius,
        seed42__bootstrap=frozen_bootstrap,
        seed42__permuted_null=frozen_null,
        seed43__high_radius=high_radius,
        seed43__low_radius=frozen_low_radius,
        seed43__bootstrap=frozen_bootstrap,
        seed43__permuted_null=frozen_null,
    )
    calibration_arrays_signature = expected_input_signature(
        str(calibration_arrays_path)
    )
    floor = 0.17589389755990817
    calibration_path = tmp_path / "calibration.json"
    calibration_signature = _write_json(
        calibration_path,
        seal({
            "schema": round0119_nodes.CALIBRATION_SCHEMA,
            "round_id": "0108",
            "floor_calibration": {"registered_floor": floor},
            "arrays": calibration_arrays_signature,
        }),
    )
    matched_path = tmp_path / "matched.json"
    matched_signature = _write_json(
        matched_path,
        seal({
            "schema": round0119_nodes.MATCHED_SCHEMA,
            "round_id": "0110",
            "calibration": calibration_signature,
            "registered_floor": floor,
            "floor_changed_or_tuned": False,
            "census_receipt": census_signature,
            "source": source_signature,
            "representative_reference": reference_signature,
            "universe": {
                "rows": rows,
                "anchors": anchors_count,
                "family_size_cutoff_exclusive": (
                    round0119_nodes.FAMILY_SIZE_CUTOFF
                ),
                "anchors_after_filter": anchors_count,
                "anchor_compact_rows_sha256": ordered_array_sha256(
                    anchors
                ),
                "anchor_global_rows_sha256": ordered_array_sha256(
                    global_rows
                ),
                "high_radius_sha256": ordered_array_sha256(high_radius),
            },
            "arrays": matched_arrays_signature,
        }),
    )

    class FakeModel:
        def __init__(self, marker: int):
            self.marker = marker

        def transform(self, values, batch_size: int) -> np.ndarray:
            assert len(values) == rows
            result = np.zeros((rows, 2), dtype=np.float32)
            result[:, 0] = self.marker
            result[:, 1] = np.arange(rows, dtype=np.float32)
            return result

    markers = {
        key: index for index, key in enumerate(round0119_nodes.CELL_ORDER)
    }

    def fake_authenticate(spec, *, device="cuda"):
        marker = markers[spec["key"]]
        signature = {
            "kind": "file",
            "canonical_path": f"/smoke/{spec['key']}",
            "bytes": 1,
            "sha256": f"{marker + 1:064x}",
        }
        return {
            "model": FakeModel(marker),
            "train": {
                **signature,
                "canonical_path": (
                    signature["canonical_path"] + ".train"
                ),
            },
            "production_config": {
                **signature,
                "canonical_path": signature["canonical_path"] + ".config",
            },
            "model_signature": signature,
            "seed": spec["seed"],
            "group": spec["group"],
            "training_population": "population",
            "training_graph": "graph",
            "training_dose": "dose",
        }

    monkeypatch.setattr(
        round0119_nodes, "_authenticate_model", fake_authenticate
    )
    import basemap.panel_v2 as panel_v2

    def fake_self_knn(corpus, selected, k, config, **kwargs):
        marker = int(corpus[0, 0])
        radius = high_radius if marker < 4 else high_radius[::-1]
        distances = np.repeat(radius[:, None], k, axis=1)
        neighbors = np.tile(
            np.arange(k, dtype=np.int64), (anchors_count, 1)
        )
        return neighbors, distances, {"backend": "cpu-smoke-exact"}

    monkeypatch.setattr(panel_v2, "_self_knn", fake_self_knn)
    specs = [
        {
            "key": key,
            "group": (
                "historical_2m"
                if key.startswith("historical")
                else (
                    "current_2m"
                    if key.startswith("current_2m")
                    else "current_25m"
                )
            ),
            "seed": 42 if key.endswith("42") else 43,
        }
        for key in round0119_nodes.CELL_ORDER
    ]
    score_root = tmp_path / "score"
    score = round0119_nodes.run_score(
        {"manifest": {"release_sha": "b" * 40}},
        {
            "outputs": [str(score_root)],
            "r0110_matched_receipt": matched_signature,
            "r0108_calibration": calibration_signature,
            "model_bundles": specs,
        },
    )
    assert score["training_performed"] is False
    assert score["cells"]["historical_2m_seed42"][
        "historical_control_reproduction"
    ]["reproduces_frozen_control"] is True
    with (score_root / "density-localization-panel.json").open(
        encoding="utf-8"
    ) as handle:
        validate_seal(json.load(handle), label="R0119 smoke score")

    decision_root = tmp_path / "decision"
    decision = round0119_nodes.run_decision(
        {"manifest": {"release_sha": "b" * 40}},
        {
            "outputs": [str(decision_root)],
            "score_output": str(score_root),
        },
    )
    assert decision["outcome"] == "bundled-2m-to-25m-transition-localized"
    assert decision["bundled_2m_to_25m_transition_localized"] is True
    assert decision["single_cause_localized"] is False
    assert decision["matched_cell_rescues_native_quality"] is False
    with (
        decision_root / "density-localization-decision.json"
    ).open(encoding="utf-8") as handle:
        validate_seal(json.load(handle), label="R0119 smoke decision")


def test_current_2m_failure_rejects_scale_specific_explanation(
    tmp_path: Path,
) -> None:
    score_root = tmp_path / "score"
    score_root.mkdir()
    cells: dict[str, Any] = {}
    for key in round0119_nodes.CELL_ORDER:
        historical = key.startswith("historical")
        cells[key] = {
            "clears_unchanged_registered_floor": (
                key != "current_2m_seed43"
            ),
            "historical_control_reproduction": (
                {"reproduces_frozen_control": True}
                if historical
                else None
            ),
        }
    _write_json(
        score_root / "density-localization-panel.json",
        seal({
            "schema": round0119_nodes.SCORE_SCHEMA,
            "round_id": "0119",
            "training_performed": False,
            "cells": cells,
        }),
    )
    decision = round0119_nodes.run_decision(
        {"manifest": {"release_sha": "c" * 40}},
        {
            "outputs": [str(tmp_path / "decision")],
            "score_output": str(score_root),
        },
    )
    assert decision["outcome"] == "scale-specific-explanation-rejected"
    assert decision["scale_specific_explanation_rejected"] is True
    assert decision["bundled_2m_to_25m_transition_localized"] is False
