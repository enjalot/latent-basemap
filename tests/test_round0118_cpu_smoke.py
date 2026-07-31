"""CPU train-artifact -> reload -> panel -> seal -> decision smoke for R0118."""
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
from basemap.output_safety import atomic_write_new_json
from basemap.panel_v2 import PanelV2Config, score_panel
from basemap.round0106_graph import GRAPH_SCHEMA
from basemap.round0108_evaluation import seal, validate_seal
from experiments import round0110_nodes, round0118_nodes


def _write_sealed(path: Path, body: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(seal(body), sort_keys=True), encoding="utf-8")


def _prior_decision() -> dict[str, Any]:
    metrics = {
        "core_global_ffr": (0.60, 0.62),
        "core_global_recall_at_10": (0.10, 0.11),
        "core_global_recall_at_50_of_high10": (0.20, 0.23),
        "density_v2": (0.16, 0.17),
        "polish_recall_at_10": (0.09, 0.10),
        "polish_recall_at_50_of_high10": (0.19, 0.21),
        "polish_to_in_mix_median_ratio": (0.52, 0.54),
    }
    return {
        "schema": round0110_nodes.DECISION_SCHEMA,
        "round_id": "0110",
        "checks": {
            "both_seeds_clear_unchanged_floor_on_matched_fineweb": True,
            "broader_diverse_density_claim_unresolved": True,
            "cross_seed_deltas_excluded_from_decision": True,
            "original_frozen_two_seed_rule_unchanged": True,
            "projection_ffr_excluded_from_decision": True,
            "raw_prompt_identity_closes": True,
            "seed42_atlas_quality_passed": False,
            "seed42_fixed_core_gate_passed": False,
            "seed42_fixed_polish_ood_gate_passed": True,
            "seed42_native_non_density_core_passed": True,
            "seed43_fixed_core_gate_passed": False,
            "seed43_fixed_polish_ood_gate_passed": True,
            "seed43_native_non_density_core_passed": True,
        },
        "comparison_metrics": {
            name: {
                "seed42": values[0],
                "seed43": values[1],
                "role": "diagnostic-only",
            }
            for name, values in metrics.items()
        },
        "two_seed_quality_capability_released": False,
        "matched_fineweb_qualified_atlas_capability_released": True,
        "full_diverse_universe_density_under_original_floor": {
            "seed42_clears_floor": False,
            "seed43_clears_floor": False,
            "both_seeds_clear_floor": False,
            "status": "failed",
            "overridden_by_matched_fineweb_cell": False,
        },
        "broader_diverse_density_preservation_claimed": False,
        "production_document_prompt_transfer_resolved": False,
        "production_readiness_claimed": False,
    }


def test_round0118_train_artifact_to_decision_cpu_smoke(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Catch schema/reload/publication failures before a multi-hour train."""
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    train_output = tmp_path / "train"
    train_output.mkdir()
    graph_path = tmp_path / "graph-manifest.json"
    mapping_path = tmp_path / "compact-to-global.i64.npy"
    np.save(mapping_path, np.arange(240, dtype=np.int64))
    mapping_signature = expected_input_signature(mapping_path)
    _write_sealed(
        graph_path,
        {
            "schema": GRAPH_SCHEMA,
            "compact_mapping": mapping_signature,
        },
    )
    graph_signature = expected_input_signature(graph_path)

    model_path = train_output / "model.pt"
    model_path.write_bytes(b"round0118-cpu-smoke-model")
    model_signature = expected_input_signature(model_path)
    model_config = {
        "architecture": "residual_bottleneck",
        "input_dimension": 8,
        "hidden_dimension": 16,
        "hidden_layers": 2,
        "output_dimension": 2,
        "use_batchnorm": False,
        "use_dropout": False,
        "low_dim_kernel": "legacy_lp",
        "a": 1.0,
        "b": 1.0,
    }
    config = {
        "model": model_config,
        "optimizer": {"seed": 44},
    }
    config_sha = sha256_bytes(canonical_json(config))
    (train_output / "production-config.json").write_text(
        json.dumps({
            "schema": round0118_nodes.PRODUCTION_CONFIG_SCHEMA,
            "round_id": "0111",
            "config": config,
            "config_sha256": config_sha,
        }, sort_keys=True),
        encoding="utf-8",
    )
    _write_sealed(
        train_output / "train-receipt.json",
        {
            "schema": round0118_nodes.TRAIN_RECEIPT_SCHEMA,
            "round_id": "0111",
            "graph_manifest": graph_signature,
            "production_config_sha256": config_sha,
            "train_checks": {
                "exact_update_closure": True,
                "zero_numerical_skips": True,
                "no_pipeline_stamp_drift": True,
                "endpoint_rows_match_updates": True,
                "weighted_rejection_accounting_closes": True,
            },
            "model": model_signature,
        },
    )

    class SmokeModel:
        architecture = "residual_bottleneck"
        input_dim = 8
        hidden_dim = 16
        n_layers = 2
        n_components = 2
        use_batchnorm = False
        use_dropout = False
        low_dim_kernel = "legacy_lp"
        a = 1.0
        b = 1.0

        def transform(self, values, batch_size: int = 64) -> np.ndarray:
            blocks = [
                np.asarray(
                    values[start : start + batch_size], dtype=np.float32
                )
                for start in range(0, len(values), batch_size)
            ]
            matrix = np.concatenate(blocks)
            return np.stack(
                (
                    matrix[:, 0] - 0.25 * matrix[:, 2],
                    matrix[:, 1] + 0.20 * matrix[:, 3],
                ),
                axis=1,
            ).astype(np.float32)

    from basemap.pumap.parametric_umap import ParametricUMAP

    monkeypatch.setattr(
        ParametricUMAP,
        "load",
        classmethod(lambda cls, path, device=None: SmokeModel()),
    )
    bundle = round0118_nodes._seed44_model(
        train_output=str(train_output),
        graph_manifest_path=str(graph_path),
        graph_manifest_sha256=graph_signature["sha256"],
    )
    assert bundle["config"]["optimizer"]["seed"] == 44
    assert bundle["train"]["model"] == model_signature

    rng = np.random.default_rng(118)
    features = rng.normal(size=(240, 8)).astype(np.float32)
    coordinates = bundle["model"].transform(features)
    panel = score_panel(
        features,
        coordinates,
        config=PanelV2Config(
            frac=0.1,
            k_hit=3,
            k_density=3,
            n_anchors=24,
            corpus_chunk=64,
            overselect=4,
            block_elems=100_000,
            rerank_byte_cap=8_000_000,
            peak_byte_cap=16_000_000,
        ),
        provenance={"round": "0118", "mode": "cpu-smoke"},
    )
    assert panel["guards"]["coords_finite"] is True
    assert panel["guards"]["coords_collapsed"] is False

    prior_path = tmp_path / "r0110-decision.json"
    _write_sealed(prior_path, _prior_decision())
    transform_root = tmp_path / "coordinates"
    transform_root.mkdir()
    _write_sealed(
        transform_root / "actual-transform.json",
        {
            "schema": "round0036-transform-capability-v1",
            "round_id": "0118",
            "map_key": round0118_nodes.MAP_KEY,
            "model": model_signature,
        },
    )
    core_root = tmp_path / "core"
    core_root.mkdir()
    core_arrays = core_root / "core-panel-arrays.npz"
    np.savez(
        core_arrays,
        compact_anchor_rows=np.arange(24, dtype=np.int64),
    )
    checks = {
        "coordinates_finite_and_noncollapsed": True,
        "density_v2_clears_registered_jina_floor": False,
        "every_language_ffr_at_least_0_40_of_pooled_english": True,
        "global_ffr_at_least_0_40": True,
        "global_recall50_strictly_exceeds_recall10": True,
    }
    _write_sealed(
        core_root / "core-geometry.json",
        {
            "schema": round0118_nodes.CORE_SCHEMA,
            "round_id": "0118",
            "map_key": round0118_nodes.MAP_KEY,
            "metrics": {
                "global": {
                    "ffr": 0.61,
                    "recall_at_10": 0.12,
                    "recall_at_50_of_high10": 0.24,
                },
                "density_v2": {"correlation": 0.16},
            },
            "decision": {"checks": checks, "passed": False},
            "arrays": expected_input_signature(core_arrays),
        },
    )
    ood_root = tmp_path / "ood"
    _write_sealed(
        ood_root / "ood-evaluation.json",
        {
            "schema": round0118_nodes.OOD_SCHEMA,
            "round_id": "0118",
            "map_key": round0118_nodes.MAP_KEY,
            "embedding_prompt": "raw",
            "prompt_applied": False,
            "language_cells": {
                "pol_Latn": {
                    "probe": {
                        "recall_at_10": 0.11,
                        "recall_at_50_of_high10": 0.22,
                    }
                }
            },
            "headline_decision": {
                "passed": True,
                "polish_to_in_mix_median_ratio": 0.55,
            },
        },
    )
    matched_root = tmp_path / "matched"
    _write_sealed(
        matched_root / "matched-density.json",
        {
            "schema": round0118_nodes.MATCHED_DENSITY_SCHEMA,
            "round_id": "0118",
            "floor_changed_or_tuned": False,
            "full_diverse_universe_density_resolved": False,
            "full_diverse_universe_density_claimed": False,
            "checks": {
                "all_three_seeds_clear_matched_floor": True,
            },
            "calibration_portability_capability_released": True,
        },
    )

    def fake_registry_refresh(**kwargs) -> dict[str, Any]:
        assert kwargs["round_id"] == "0118"
        assert kwargs["map_key"] == round0118_nodes.MAP_KEY
        receipt = seal({
            "schema": kwargs["publication_schema"],
            "round_id": kwargs["round_id"],
            "cpu_smoke": True,
        })
        atomic_write_new_json(
            kwargs["receipt_path"], receipt, immutable=True
        )
        return receipt

    monkeypatch.setattr(
        round0118_nodes,
        "_refresh_registry_best_effort",
        fake_registry_refresh,
    )
    decision_root = tmp_path / "decision"
    render_root = tmp_path / "renders"
    result = round0118_nodes.run_decision(
        {"manifest": {"release_sha": "a" * 40}},
        {
            "outputs": [str(decision_root)],
            "r0110_decision": str(prior_path),
            "transform_output": str(transform_root),
            "core_output": str(core_root),
            "ood_output": str(ood_root),
            "matched_density_output": str(matched_root),
            "render_output": str(render_root),
        },
    )
    assert result["seed44_atlas_quality_capability_released"] is False
    assert result["three_seed_quality_capability_released"] is False
    assert (
        result["matched_fineweb_qualified_atlas_capability_released"]
        is True
    )
    assert result["production_readiness_claimed"] is False
    for path, label in (
        (
            decision_root / "three-seed-decision.json",
            "R0118 smoke decision",
        ),
        (render_root / "map-definition.json", "R0118 smoke definition"),
        (
            decision_root / "registry-publication.json",
            "R0118 smoke registry publication",
        ),
    ):
        with path.open(encoding="utf-8") as handle:
            validate_seal(json.load(handle), label=label)
