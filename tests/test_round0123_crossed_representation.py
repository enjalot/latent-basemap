from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from basemap.artifact_identity import expected_input_signature
from basemap.round0108_evaluation import seal, validate_seal
from experiments import prepare_round0123_queue, round0123_nodes


def _signature(path: str, marker: int) -> dict[str, Any]:
    return {
        "kind": "file",
        "canonical_path": path,
        "bytes": 1,
        "sha256": f"{marker:064x}",
    }


def _write_json(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return expected_input_signature(str(path))


def test_indexed_rows_array_preserves_exact_order_and_indexing() -> None:
    source = np.arange(36, dtype=np.float16).reshape(12, 3)
    mapping = np.asarray([0, 2, 5, 8, 11], dtype=np.int64)
    compact = round0123_nodes.IndexedRowsArray(source, mapping)
    assert compact.shape == (5, 3)
    assert compact.dtype == np.dtype("float16")
    np.testing.assert_array_equal(compact[:], source[mapping])
    np.testing.assert_array_equal(compact[1:4], source[[2, 5, 8]])
    np.testing.assert_array_equal(compact[[4, 0, 2]], source[[11, 0, 5]])
    np.testing.assert_array_equal(compact[-1], source[11])


def _strong_alignment_radii(
    rows: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.RandomState(17)
    legacy_high = np.exp(rng.normal(0.0, 0.35, size=rows))
    fresh_high = np.exp(rng.normal(0.0, 0.35, size=rows))
    return legacy_high, fresh_high, {
        "legacy_map__legacy_input": legacy_high.copy(),
        "legacy_map__fresh_input": 1.0 / fresh_high,
        "fresh_map__legacy_input": 1.0 / legacy_high,
        "fresh_map__fresh_input": fresh_high.copy(),
    }


def test_paired_bootstrap_uses_matching_reference_space_and_shared_anchors() -> None:
    legacy_high, fresh_high, low = _strong_alignment_radii(400)
    summary, samples = round0123_nodes._paired_alignment_contrasts(
        legacy_high=legacy_high,
        fresh_high=fresh_high,
        low_radii=low,
        draws=300,
        seed=91,
    )
    assert summary["cell_correlations"] == {
        "legacy_map__legacy_input": 1.0,
        "legacy_map__fresh_input": -1.0,
        "fresh_map__legacy_input": -1.0,
        "fresh_map__fresh_input": 1.0,
    }
    assert all(
        value["direction"] == "positive"
        for value in summary["contrasts"].values()
    )
    assert round0123_nodes._classify_alignment(
        summary["contrasts"]
    ) == "symmetric-representation-alignment-penalty"

    repeated, repeated_samples = (
        round0123_nodes._paired_alignment_contrasts(
            legacy_high=legacy_high,
            fresh_high=fresh_high,
            low_radii=low,
            draws=300,
            seed=91,
        )
    )
    assert repeated == summary
    for key in samples:
        np.testing.assert_array_equal(samples[key], repeated_samples[key])
        # The constructed exact inverse/matched cells make each paired draw
        # share the same anchor resample and close to a fixed contrast.
        np.testing.assert_allclose(samples[key], 2.0 if key != (
            "crossed_interaction"
        ) else 4.0, atol=1e-12)


@pytest.mark.parametrize(
    ("legacy", "fresh", "outcome"),
    [
        (
            "positive",
            "positive",
            "symmetric-representation-alignment-penalty",
        ),
        (
            "positive",
            "indeterminate",
            "legacy-map-positive-alignment-only",
        ),
        (
            "negative",
            "positive",
            "fresh-map-positive-alignment-only",
        ),
        (
            "indeterminate",
            "negative",
            "no-reliable-positive-map-input-alignment",
        ),
    ],
)
def test_alignment_selector_reports_direction_without_absolute_floor(
    legacy: str,
    fresh: str,
    outcome: str,
) -> None:
    contrasts = {
        "legacy_map_alignment_advantage": {"direction": legacy},
        "fresh_map_alignment_advantage": {"direction": fresh},
        "crossed_interaction": {"direction": "indeterminate"},
    }
    assert round0123_nodes._classify_alignment(contrasts) == outcome


class _FakeModel:
    def __init__(self, map_marker: float):
        self.map_marker = map_marker
        self.inputs_seen: list[tuple[int, float, int]] = []

    def transform(self, values: Any, batch_size: int) -> np.ndarray:
        input_marker = float(np.asarray(values[0])[0])
        self.inputs_seen.append((len(values), input_marker, batch_size))
        coordinates = np.empty((len(values), 2), dtype=np.float32)
        coordinates[:, 0] = self.map_marker
        coordinates[:, 1] = input_marker
        return coordinates


def test_cpu_smoke_scores_exact_two_by_two_and_decides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    rows = 256
    anchors_count = 128
    dimension = 4
    monkeypatch.setattr(round0123_nodes, "SOURCE_ROWS", rows)
    monkeypatch.setattr(round0123_nodes, "COMPACT_ROWS", rows)
    monkeypatch.setattr(round0123_nodes, "DIMENSION", dimension)
    monkeypatch.setattr(round0123_nodes, "ANCHORS", anchors_count)

    anchors = np.arange(anchors_count, dtype=np.int64)
    legacy = np.ones((rows, dimension), dtype=np.float16)
    fresh = np.full((rows, dimension), 2.0, dtype=np.float16)
    legacy_high, fresh_high, low = _strong_alignment_radii(anchors_count)
    fake_r0122 = {
        key: _signature(f"/smoke/r0122-{key}", index + 1)
        for index, key in enumerate(
            ("review", "result", "queue", "terminal", "score", "decision")
        )
    }
    fake_r0122.update({
        "required_outcome": round0123_nodes.R0122_REQUIRED_OUTCOME,
        "accepted": True,
    })
    monkeypatch.setattr(
        round0123_nodes,
        "_load_registered_inputs",
        lambda _job: {
            "r0122": fake_r0122,
            "assembly": _signature("/smoke/assembly", 20),
            "mapping": _signature("/smoke/mapping", 21),
            "legacy": legacy,
            "legacy_lineage": {
                "shared_evidence": _signature("/smoke/shared", 22),
                "source_prefix_payload_sha256": "a" * 64,
                "source_segments": [],
                "compact_mapping": _signature("/smoke/mapping", 21),
                "compact_ids": {
                    "rows": rows,
                    "ordered_global_rows_sha256": "b" * 64,
                },
            },
            "fresh": fresh,
            "fresh_signature": _signature("/smoke/fresh", 23),
            "anchors": anchors,
            "fresh_high_radius": fresh_high,
            "fresh_reference": {
                "artifact": _signature("/smoke/reference", 24),
                "key": "c" * 64,
                "content_sha256": "d" * 64,
                "high_radius_sha256": "e" * 64,
            },
        },
    )
    models = {
        "legacy_map": _FakeModel(10.0),
        "fresh_map": _FakeModel(20.0),
    }

    def model_bundle(key: str, marker: int) -> dict[str, Any]:
        return {
            "model": models[key],
            "map_key": key,
            "source_round": "0104" if key == "legacy_map" else "0115",
            "training_bundle": f"{key} registered training",
            "train_receipt": _signature(f"/smoke/{key}-train", marker),
            "production_config": _signature(
                f"/smoke/{key}-config", marker + 1
            ),
            "model_signature": _signature(
                f"/smoke/{key}-model", marker + 2
            ),
        }

    monkeypatch.setattr(
        round0123_nodes,
        "_authenticate_models",
        lambda _job: {
            "legacy_map": model_bundle("legacy_map", 30),
            "fresh_map": model_bundle("fresh_map", 40),
        },
    )
    radii = iter([
        legacy_high,
        low["legacy_map__legacy_input"],
        low["legacy_map__fresh_input"],
        low["fresh_map__legacy_input"],
        low["fresh_map__fresh_input"],
    ])

    def fake_exact(
        _corpus: Any,
        _anchors: np.ndarray,
        *,
        high_dimensional: bool,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        value = next(radii)
        return value, (
            {"boundary_min_gap": 0.1, "overselect": 8}
            if high_dimensional
            else {}
        )

    monkeypatch.setattr(
        round0123_nodes, "_exact_density_radii", fake_exact
    )
    release = "f" * 40
    score_root = tmp_path / "score"
    score = round0123_nodes.run_score(
        {"manifest": {"release_sha": release}},
        {"outputs": [str(score_root)]},
    )
    assert tuple(score["cells"]) == round0123_nodes.CELL_ORDER
    assert score["scorer"]["historical_absolute_floor_applied"] is False
    assert score["training_performed"] is False
    assert score["native_quality_claimed"] is False
    assert score["single_factor_cause_claimed"] is False
    for model in models.values():
        assert model.inputs_seen == [
            (rows, 1.0, round0123_nodes.TRANSFORM_BATCH_ROWS),
            (rows, 2.0, round0123_nodes.TRANSFORM_BATCH_ROWS),
        ]
    with (
        score_root / "crossed-representation-panel.json"
    ).open(encoding="utf-8") as handle:
        validate_seal(json.load(handle), label="R0123 CPU smoke panel")

    decision = round0123_nodes.run_decision(
        {"manifest": {"release_sha": release}},
        {
            "outputs": [str(tmp_path / "decision")],
            "score_output": str(score_root),
        },
    )
    assert decision["outcome"] == (
        "symmetric-representation-alignment-penalty"
    )
    assert decision["selector"]["historical_absolute_floor_applied"] is False
    assert decision["native_quality_claimed"] is False
    assert decision["single_factor_cause_localized"] is False
    assert decision["production_transfer_claimed"] is False
    assert decision["training_performed"] is False


def test_r0122_gate_refuses_every_nonregistered_outcome(
    tmp_path: Path,
) -> None:
    evidence = {}
    for marker, key in enumerate(
        ("review", "result", "queue", "terminal"), start=1
    ):
        path = tmp_path / f"{key}.txt"
        path.write_text(key, encoding="utf-8")
        evidence[key] = expected_input_signature(str(path))
    score_path = tmp_path / "score.json"
    score = seal({
        "schema": round0123_nodes.R0122_SCORE_SCHEMA,
        "round_id": "0122",
        "release_sha": round0123_nodes.R0122_RELEASE_SHA,
        "training_performed": False,
    })
    evidence["score"] = _write_json(score_path, score)
    decision_path = tmp_path / "decision.json"
    decision = seal({
        "schema": round0123_nodes.R0122_DECISION_SCHEMA,
        "round_id": "0122",
        "release_sha": round0123_nodes.R0122_RELEASE_SHA,
        "score": evidence["score"],
        "outcome": "failure-already-present-pre-r0115",
        "evaluation_path_material": False,
        "boundary_localized": True,
        "single_factor_cause_localized": False,
        "native_training_geometry_declared_bad": False,
        "production_transfer_claimed": False,
        "training_performed": False,
    })
    evidence["decision"] = _write_json(decision_path, decision)
    with pytest.raises(
        round0123_nodes.Round0123Error,
        match="conditional branch",
    ):
        round0123_nodes._r0122_gate({"r0122_evidence": evidence})


def test_preparer_fails_before_discovery_or_write_outside_run_venv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        prepare_round0123_queue.sys,
        "executable",
        "/some/dev/.venv/bin/python",
    )
    monkeypatch.setattr(
        prepare_round0123_queue.sys,
        "prefix",
        "/some/dev/.venv",
    )
    discovered = False

    def should_not_discover() -> str:
        nonlocal discovered
        discovered = True
        raise AssertionError("round discovery must not run")

    monkeypatch.setattr(
        prepare_round0123_queue, "_issued_round", should_not_discover
    )
    queue_root = tmp_path / "queue"
    with pytest.raises(RuntimeError, match="dedicated run environment"):
        prepare_round0123_queue.prepare_round0123(
            release_sha="a" * 40,
            r0122_review=("/missing/review.md", "b" * 64),
            queue_root=str(queue_root),
        )
    assert discovered is False
    assert not queue_root.exists()


def test_registered_queue_contract_is_narrow_and_no_training() -> None:
    assert round0123_nodes.CELL_ORDER == (
        "legacy_map__legacy_input",
        "legacy_map__fresh_input",
        "fresh_map__legacy_input",
        "fresh_map__fresh_input",
    )
    assert round0123_nodes.K_DENSITY == 15
    assert round0123_nodes.BOOTSTRAP_DRAWS == 1_000
    assert round0123_nodes.BOOTSTRAP_SEED == 12_301
    assert prepare_round0123_queue.ROUND_ID == "0123"
