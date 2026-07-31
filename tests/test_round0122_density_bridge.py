from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from basemap.round0108_evaluation import seal, validate_seal
from experiments import prepare_round0122_queue, round0122_nodes


def _write_json(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return expected_input_signature(str(path))


def _signature(path: str, marker: int) -> dict[str, Any]:
    return {
        "kind": "file",
        "canonical_path": path,
        "bytes": 1,
        "sha256": f"{marker:064x}",
    }


def _bundle(key: str, marker: int, *, arm: str = "raw") -> dict[str, Any]:
    return {
        "model": _FakeModel(marker),
        "model_signature": _signature(f"/smoke/{key}.model", marker + 1),
        "train_receipt": _signature(f"/smoke/{key}.train", marker + 11),
        "production_config": _signature(
            f"/smoke/{key}.config", marker + 21
        ),
        "key": key,
        "arm": arm,
        "seed": 42 if "42" in key else 43,
        "training_population": "registered population",
        "training_representation": arm,
        "training_graph": "registered graph",
        "training_sampler": "registered sampler",
        "training_updates": 500_000,
    }


class _FakeModel:
    def __init__(self, marker: int):
        self.marker = marker
        self.rows_seen: list[int] = []

    def transform(self, values: np.ndarray, batch_size: int) -> np.ndarray:
        assert batch_size == round0122_nodes.TRANSFORM_BATCH_ROWS
        self.rows_seen.append(len(values))
        result = np.empty((len(values), 2), dtype=np.float32)
        result[:, 0] = self.marker
        result[:, 1] = np.arange(len(values), dtype=np.float32)
        return result


def test_density_bridge_cpu_smoke_transforms_full_source_before_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    rows = 256
    representatives = 250
    anchors_count = 128
    retained = np.delete(
        np.arange(rows, dtype=np.int64), [3, 9, 22, 45, 70, 188]
    )
    anchors = np.arange(anchors_count, dtype=np.int64)
    global_rows = retained[anchors]
    high_radius = np.linspace(1.0, 2.0, anchors_count)
    source = np.arange(rows * 4, dtype=np.float16).reshape(rows, 4)
    lineage = {
        "r0108_calibration": _signature("/smoke/calibration", 91),
        "registered_floor": 0.2,
    }
    monkeypatch.setattr(round0122_nodes, "SOURCE_ROWS", rows)
    monkeypatch.setattr(round0122_nodes, "SOURCE_DIMENSION", 4)
    monkeypatch.setattr(round0122_nodes, "REPRESENTATIVE_ROWS", representatives)
    monkeypatch.setattr(round0122_nodes, "ANCHORS", anchors_count)
    monkeypatch.setattr(round0122_nodes, "REGISTERED_FLOOR", 0.2)
    monkeypatch.setattr(
        round0122_nodes,
        "_load_universe",
        lambda _job: (
            source,
            None,
            retained,
            anchors,
            global_rows,
            high_radius,
            lineage,
            {},
        ),
    )

    direct_cells = {
        "historical_2m_seed42": {
            "clears_unchanged_registered_floor": True,
        },
        "historical_2m_seed43": {
            "clears_unchanged_registered_floor": True,
        },
        "current_2m_seed42": {
            "clears_unchanged_registered_floor": False,
        },
        "current_2m_seed43": {
            "clears_unchanged_registered_floor": False,
        },
    }
    monkeypatch.setattr(
        round0122_nodes,
        "_r0119_evidence",
        lambda *_args, **_kwargs: (
            {
                "panel": _signature("/smoke/r0119-panel", 92),
                "decision": _signature("/smoke/r0119-decision", 93),
                "historical_and_direct_cells": direct_cells,
            },
            direct_cells,
        ),
    )
    models = {
        key: _FakeModel(index)
        for index, key in enumerate(round0122_nodes.NEW_CELL_ORDER)
    }

    def fake_r0104(spec: dict[str, Any]) -> dict[str, Any]:
        key = spec["key"]
        result = _bundle(key, models[key].marker, arm=spec["arm"])
        result["model"] = models[key]
        return result

    def fake_replay(spec: dict[str, Any]) -> dict[str, Any]:
        key = (
            "r0115_raw_seed42_full_transform"
            if spec["key"] == "current_2m_seed42"
            else "r0117_raw_seed43_full_transform"
        )
        result = _bundle(key, models[key].marker)
        result["model"] = models[key]
        return {
            "model": result["model"],
            "train": result["train_receipt"],
            "production_config": result["production_config"],
            "model_signature": result["model_signature"],
            "seed": result["seed"],
            "training_population": result["training_population"],
            "training_representation": result["training_representation"],
            "training_graph": result["training_graph"],
            "authenticated_training_semantics": {
                "sampler_class": result["training_sampler"],
                "successful_updates": result["training_updates"],
            },
        }

    monkeypatch.setattr(
        round0122_nodes, "_authenticate_r0104_model", fake_r0104
    )
    monkeypatch.setattr(
        round0122_nodes, "_authenticate_r0119_model", fake_replay
    )
    import basemap.panel_v2 as panel_v2

    def fake_self_knn(
        corpus: np.ndarray,
        selected: np.ndarray,
        k: int,
        _config: dict[str, Any],
        **_kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        marker = int(corpus[0, 0])
        radius = high_radius if marker == 0 else high_radius[::-1]
        distances = np.repeat(radius[:, None], k, axis=1)
        neighbors = np.tile(
            np.arange(k, dtype=np.int64), (anchors_count, 1)
        )
        return neighbors, distances, {"backend": "cpu-smoke-exact"}

    monkeypatch.setattr(panel_v2, "_self_knn", fake_self_knn)
    r0104_specs = [
        {
            "key": key,
            "arm": (
                "fp16_control"
                if "fp16" in key
                else "int8_treatment"
            ),
        }
        for key in round0122_nodes.NEW_CELL_ORDER[:2]
    ]
    replay_specs = []
    for index, key in enumerate(
        ("current_2m_seed42", "current_2m_seed43")
    ):
        signatures = {
            "train_receipt": _signature(f"/smoke/{key}.train", 31 + index),
            "production_config": _signature(
                f"/smoke/{key}.config", 41 + index
            ),
            "model": _signature(f"/smoke/{key}.model", 51 + index),
        }
        direct_cells[key].update(signatures)
        replay_specs.append({"key": key, **signatures})

    release = "a" * 40
    score_root = tmp_path / "score"
    score = round0122_nodes.run_score(
        {"manifest": {"release_sha": release}},
        {
            "outputs": [str(score_root)],
            "r0104_model_bundles": r0104_specs,
            "r0119_replay_model_bundles": replay_specs,
        },
    )
    assert all(model.rows_seen == [rows] for model in models.values())
    assert all(
        cell["transform_input_rows"] == rows
        and cell["transform_selection_after_transform"] is True
        and cell["transform_selected_rows"] == representatives
        for cell in score["new_cells"].values()
    )
    assert score["training_performed"] is False
    assert score["single_factor_cause_claimed"] is False
    with (score_root / "density-bridge-panel.json").open(
        encoding="utf-8"
    ) as handle:
        validate_seal(json.load(handle), label="R0122 smoke panel")

    decision = round0122_nodes.run_decision(
        {"manifest": {"release_sha": release}},
        {
            "outputs": [str(tmp_path / "decision")],
            "score_output": str(score_root),
        },
    )
    assert decision["outcome"] == (
        "failure-enters-after-r0104-within-r0115-bundle"
    )
    assert decision["storage_sensitive_diagnostic"][
        "fp16_int8_floor_classification_disagrees"
    ] is True
    assert decision["boundary_conclusion_tied_to"] == (
        "r0104_fp16_seed42_full_transform"
    )
    assert decision["single_factor_cause_localized"] is False
    assert decision["native_training_geometry_declared_bad"] is False


def _decision_score(
    *,
    fp16_pass: bool,
    int8_pass: bool,
    seed42_direct: bool = False,
    seed42_replay: bool = False,
    seed43_direct: bool = False,
    seed43_replay: bool = False,
) -> dict[str, Any]:
    values = {
        "r0104_fp16_seed42_full_transform": fp16_pass,
        "r0104_int8_seed42_full_transform": int8_pass,
        "r0115_raw_seed42_full_transform": seed42_replay,
        "r0117_raw_seed43_full_transform": seed43_replay,
    }
    return seal({
        "schema": round0122_nodes.SCORE_SCHEMA,
        "round_id": "0122",
        "release_sha": "b" * 40,
        "training_performed": False,
        "new_cells": {
            key: {
                "clears_unchanged_registered_floor": value,
                "transform_input_rows": round0122_nodes.SOURCE_ROWS,
                "transform_selection_after_transform": True,
            }
            for key, value in values.items()
        },
        "r0119_reused_evidence": {
            "historical_and_direct_cells": {
                "historical_2m_seed42": {
                    "clears_unchanged_registered_floor": True,
                },
                "historical_2m_seed43": {
                    "clears_unchanged_registered_floor": True,
                },
                "current_2m_seed42": {
                    "clears_unchanged_registered_floor": seed42_direct,
                },
                "current_2m_seed43": {
                    "clears_unchanged_registered_floor": seed43_direct,
                },
            }
        },
    })


@pytest.mark.parametrize(
    ("score", "outcome", "localized"),
    [
        (
            _decision_score(
                fp16_pass=True,
                int8_pass=True,
                seed42_replay=True,
            ),
            "evaluation-path-material",
            False,
        ),
        (
            _decision_score(fp16_pass=True, int8_pass=True),
            "failure-enters-after-r0104-within-r0115-bundle",
            True,
        ),
        (
            _decision_score(fp16_pass=False, int8_pass=False),
            "failure-already-present-pre-r0115",
            True,
        ),
    ],
)
def test_registered_decision_branch_order(
    tmp_path: Path,
    score: dict[str, Any],
    outcome: str,
    localized: bool,
) -> None:
    case = len(list(tmp_path.parent.glob("test_registered*")))
    score_root = tmp_path / f"score-{case}"
    _write_json(score_root / "density-bridge-panel.json", score)
    decision = round0122_nodes.run_decision(
        {"manifest": {"release_sha": "b" * 40}},
        {
            "outputs": [str(tmp_path / f"decision-{case}")],
            "score_output": str(score_root),
        },
    )
    assert decision["outcome"] == outcome
    assert decision["boundary_localized"] is localized
    assert decision["single_factor_cause_localized"] is False


def test_storage_diagnostic_never_replaces_fp16_boundary(
    tmp_path: Path,
) -> None:
    score_root = tmp_path / "score"
    _write_json(
        score_root / "density-bridge-panel.json",
        _decision_score(fp16_pass=False, int8_pass=True),
    )
    decision = round0122_nodes.run_decision(
        {"manifest": {"release_sha": "b" * 40}},
        {
            "outputs": [str(tmp_path / "decision")],
            "score_output": str(score_root),
        },
    )
    assert decision["outcome"] == "failure-already-present-pre-r0115"
    assert decision["storage_sensitive_diagnostic"][
        "fp16_int8_floor_classification_disagrees"
    ] is True
    assert decision["storage_sensitive_diagnostic"][
        "diagnostic_only"
    ] is True


def test_authenticate_r0104_binds_model_and_actual_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    key = "r0104_fp16_seed42_full_transform"
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"r0104-model")
    model_signature = expected_input_signature(str(model_path))
    monkeypatch.setitem(
        round0122_nodes.R0104_MODEL_SHA256,
        key,
        model_signature["sha256"],
    )
    config = {
        "schema": "round0104-self-contained-paired-train-config-v2",
        "arm": "fp16_control",
        "model": {
            "architecture": "residual_bottleneck",
            "input_dimension": 768,
            "hidden_dimension": 2048,
            "hidden_layers": 3,
            "output_dimension": 2,
            "use_batchnorm": False,
            "use_dropout": False,
            "low_dim_kernel": "legacy_lp",
            "a": 1.0,
            "b": 1.0,
        },
        "optimizer": {
            "seed": 42,
            "batch_size": 8_192,
            "successful_positive_lr_updates": 500_000,
        },
        "graph": {
            "k": 50,
            "sampling": "fuzzy-weight-proportional-with-replacement",
            "positive_target_mode": "binary",
        },
        "input_preprocessing": {
            "source_rows": [0, 2_000_000],
            "source_dimension": 768,
        },
        "paired_invariant": {
            "rows": 2_000_000,
            "seed": 42,
            "successful_positive_lr_updates": 500_000,
        },
    }
    config_sha = sha256_bytes(canonical_json(config))
    config_signature = _write_json(
        tmp_path / "production-config.json",
        {
            "schema": "round0104-production-config-v2",
            "config_sha256": config_sha,
            "config": config,
        },
    )
    pipeline = {
        "pipeline": "host_weighted_jina_paired",
        "sampler_class": "PairedHostWeightedJinaSampler",
        "positive_sampling": "weighted_with_replacement",
        "positive_with_replacement": True,
        "weighted_effective": True,
        "source_representation": "fp16-control",
        "feature_residency": "host-mmap-fp16-source-shards",
        "device_conversion": "device-fp32-from-exact-fp16",
    }
    accounting = {
        "optimizer_steps_attempted": 500_000,
        "optimizer_steps_succeeded": 500_000,
        "pipeline_pipeline": pipeline["pipeline"],
        "pipeline_sampler_class": pipeline["sampler_class"],
        "pipeline_positive_sampling": pipeline["positive_sampling"],
        "pipeline_source_representation": pipeline["source_representation"],
        "pipeline_feature_residency": pipeline["feature_residency"],
        "pipeline_device_conversion": pipeline["device_conversion"],
        "amp_overflow_skips": 0,
        "nonfinite_loss_skips": 0,
        "nonfinite_gradient_skips": 0,
    }
    train_signature = _write_json(
        tmp_path / "train-receipt.json",
        seal({
            "schema": "round0104-paired-train-receipt-v2",
            "round_id": "0104",
            "arm": "fp16_control",
            "model": model_signature,
            "production_config_sha256": config_sha,
            "exact_execution_receipt": pipeline,
            "train_accounting": accounting,
            "train_checks": {
                "endpoint_rows_match_updates": True,
                "exact_update_closure": True,
                "no_pipeline_stamp_drift": True,
                "zero_numerical_skips": True,
            },
        }),
    )

    class FakeLoadedModel:
        architecture = "residual_bottleneck"
        input_dim = 768
        hidden_dim = 2048
        n_layers = 3
        n_components = 2
        use_batchnorm = False
        use_dropout = False
        low_dim_kernel = "legacy_lp"
        a = 1.0
        b = 1.0

    from basemap.pumap import parametric_umap

    monkeypatch.setattr(
        parametric_umap.ParametricUMAP,
        "load",
        lambda _path, device: FakeLoadedModel(),
    )
    spec = {
        "key": key,
        "arm": "fp16_control",
        "model": model_signature,
        "production_config": config_signature,
        "train_receipt": train_signature,
    }
    bundle = round0122_nodes._authenticate_r0104_model(spec)
    assert bundle["model_signature"] == model_signature
    changed = json.loads(
        Path(train_signature["canonical_path"]).read_text(encoding="utf-8")
    )
    changed["train_accounting"]["pipeline_sampler_class"] = "WrongSampler"
    changed = seal(
        {
            key_: value
            for key_, value in changed.items()
            if key_ != "identity_sha256"
        }
    )
    Path(train_signature["canonical_path"]).write_text(
        json.dumps(changed, sort_keys=True), encoding="utf-8"
    )
    drifted = {
        **spec,
        "train_receipt": expected_input_signature(
            train_signature["canonical_path"]
        ),
    }
    with pytest.raises(
        round0122_nodes.Round0122Error,
        match="execution semantics changed",
    ):
        round0122_nodes._authenticate_r0104_model(drifted)


def test_preparer_requires_exact_run_venv_before_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        prepare_round0122_queue.sys,
        "executable",
        "/some/dev/.venv/bin/python",
    )
    monkeypatch.setattr(
        prepare_round0122_queue.sys,
        "prefix",
        "/some/dev/.venv",
    )
    with pytest.raises(RuntimeError, match="dedicated run environment"):
        prepare_round0122_queue._require_dedicated_run_environment()
    monkeypatch.setattr(
        prepare_round0122_queue.sys,
        "executable",
        prepare_round0122_queue.RUN_PYTHON,
    )
    monkeypatch.setattr(
        prepare_round0122_queue.sys,
        "prefix",
        prepare_round0122_queue.RUN_ENVIRONMENT_PREFIX,
    )
    prepare_round0122_queue._require_dedicated_run_environment()


def test_registered_contract_is_narrow_and_no_training() -> None:
    assert round0122_nodes.NEW_CELL_ORDER == (
        "r0104_fp16_seed42_full_transform",
        "r0104_int8_seed42_full_transform",
        "r0115_raw_seed42_full_transform",
        "r0117_raw_seed43_full_transform",
    )
    assert round0122_nodes.REGISTERED_FLOOR == 0.17589389755990817
    assert round0122_nodes.TRANSFORM_BATCH_ROWS == 8_192
    assert prepare_round0122_queue.ROUND_ID == "0122"
