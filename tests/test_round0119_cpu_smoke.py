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
    excluded = np.asarray([5, 17, 99, 230], dtype=np.int64)
    representatives_count = rows - len(excluded)
    monkeypatch.setattr(round0119_nodes, "SOURCE_ROWS", rows)
    monkeypatch.setattr(round0119_nodes, "SOURCE_DIMENSION", dimensions)
    monkeypatch.setattr(
        round0119_nodes, "REPRESENTATIVE_ROWS", representatives_count
    )
    monkeypatch.setattr(round0119_nodes, "ANCHORS", anchors_count)

    rng = np.random.default_rng(119)
    source_path = tmp_path / "source.npy"
    np.save(
        source_path,
        rng.normal(size=(rows, dimensions)).astype(np.float16),
    )
    source_signature = expected_input_signature(str(source_path))
    census_artifact_path = tmp_path / "census.npz"
    np.savez(census_artifact_path, excluded_rows=excluded)
    census_artifact_signature = expected_input_signature(
        str(census_artifact_path)
    )
    census_path = tmp_path / "census.json"
    census_signature = _write_json(
        census_path,
        {
            "source": source_signature,
            "census": census_artifact_signature,
        },
    )
    census_bundle = {
        "receipt": {
            "source": source_signature,
            "census": census_artifact_signature,
        },
        "signature": census_artifact_signature,
        "arrays": {
            "excluded_rows": excluded,
            "representative_rows": np.empty(0, dtype=np.int64),
            "family_counts": np.empty(0, dtype=np.int64),
        },
    }
    monkeypatch.setattr(
        round0119_nodes, "load_jina_census", lambda _path: census_bundle
    )
    anchors = np.arange(anchors_count, dtype=np.int64)
    retained_global = np.setdiff1d(
        np.arange(rows, dtype=np.int64), excluded, assume_unique=True
    )
    global_rows = retained_global[anchors]
    high_radius = np.linspace(
        0.5, 2.0, anchors_count, dtype=np.float64
    )
    family_sizes = np.ones(anchors_count, dtype=np.int64)
    reference_key = "f" * 64
    reference_path = tmp_path / "representative-reference.npz"
    np.savez(
        reference_path,
        schema=np.asarray(round0119_nodes.REFERENCE_SCHEMA),
        key=np.asarray(reference_key),
        anchor_ids=anchors,
        r_hd=high_radius,
    )
    reference_signature = expected_input_signature(str(reference_path))
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
            "scorer": (
                "R0085 density_v2: Pearson(log exact high-D mean-k15 "
                "radius, log exact low-D mean-k15 radius)"
            ),
            "threshold_tuned_after_treatment": False,
            "floor_calibration": {
                "registered_floor": floor,
                "gating_floor_registered": True,
            },
            "arrays": calibration_arrays_signature,
            "census": census_artifact_signature,
            "census_receipt": census_signature,
            "representative_reference": reference_signature,
            "representative_reference_key": reference_key,
            "anchors": {
                "before_family_filter": anchors_count,
                "after_family_lt_16_filter": anchors_count,
                "compact_rows_sha256": ordered_array_sha256(anchors),
                "global_rows_sha256": ordered_array_sha256(global_rows),
                "family_sizes_sha256": ordered_array_sha256(family_sizes),
            },
        }),
    )

    class FakeModel:
        def __init__(self, marker: int):
            self.marker = marker

        def transform(self, values, batch_size: int) -> np.ndarray:
            assert batch_size == round0119_nodes.TRANSFORM_BATCH_ROWS
            expected = rows if self.marker < 2 else representatives_count
            assert len(values) == expected
            result = np.zeros((len(values), 2), dtype=np.float32)
            result[:, 0] = self.marker
            result[:, 1] = np.arange(len(values), dtype=np.float32)
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
            "training_representation": "representation",
            "training_dequantization": "dequantization",
            "authenticated_training_semantics": {"verified": True},
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
            "r0108_calibration": calibration_signature,
            "model_bundles": specs,
        },
    )
    assert score["training_performed"] is False
    assert score["cells"]["historical_2m_seed42"][
        "historical_control_reproduction"
    ]["reproduces_frozen_control"] is True
    assert score["scorer"]["k"] == 15
    assert score["scorer"]["transform_batch_rows"] == 8_192
    assert score["cells"]["historical_2m_seed42"][
        "transform_input_rows"
    ] == rows
    assert score["cells"]["historical_2m_seed42"][
        "transform_selection_after_transform"
    ] is True
    assert score["cells"]["current_2m_seed42"][
        "transform_input_rows"
    ] == representatives_count
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


def test_current_2m_failure_rejects_only_25m_uniqueness(
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
    assert decision["outcome"] == "failure-not-unique-to-25m-tuple"
    assert decision["failure_unique_to_25m_tuple_rejected"] is True
    assert decision["scale_contribution_excluded"] is False
    assert decision["bundled_2m_to_25m_transition_localized"] is False
