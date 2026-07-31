from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from basemap.artifact_identity import expected_input_signature
from basemap.round0108_evaluation import Round0108Error
from experiments import (
    map_registry,
    prepare_round0118_queue,
    round0108_nodes,
    round0118_nodes,
)
from experiments.prepare_round0118_queue import (
    CORE_P90_S,
    GPU_P90_S,
    MATCHED_DENSITY_P90_S,
    OOD_P90_S,
    TRANSFORM_P90_S,
)


def _core(*, density: bool = True) -> dict:
    checks = {
        "coordinates_finite_and_noncollapsed": True,
        "density_v2_clears_registered_jina_floor": density,
        "every_language_ffr_at_least_0_40_of_pooled_english": True,
        "global_ffr_at_least_0_40": True,
        "global_recall50_strictly_exceeds_recall10": True,
    }
    return {
        "schema": round0118_nodes.CORE_SCHEMA,
        "round_id": "0118",
        "map_key": round0118_nodes.MAP_KEY,
        "metrics": {
            "global": {
                "ffr": 0.61,
                "recall_at_10": 0.12,
                "recall_at_50_of_high10": 0.24,
            },
            "density_v2": {"correlation": 0.18 if density else 0.16},
        },
        "decision": {"checks": checks, "passed": all(checks.values())},
    }


def _ood() -> dict:
    return {
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
    }


def _prior(*, native_passed: bool = False) -> dict:
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
        "schema": round0118_nodes.prior_nodes.DECISION_SCHEMA,
        "round_id": "0110",
        "checks": {
            "both_seeds_clear_unchanged_floor_on_matched_fineweb": True,
            "broader_diverse_density_claim_unresolved": (
                not native_passed
            ),
            "cross_seed_deltas_excluded_from_decision": True,
            "original_frozen_two_seed_rule_unchanged": True,
            "projection_ffr_excluded_from_decision": True,
            "raw_prompt_identity_closes": True,
            "seed42_atlas_quality_passed": native_passed,
            "seed42_fixed_core_gate_passed": native_passed,
            "seed42_fixed_polish_ood_gate_passed": True,
            "seed42_native_non_density_core_passed": True,
            "seed43_fixed_core_gate_passed": native_passed,
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
        "two_seed_quality_capability_released": native_passed,
        "matched_fineweb_qualified_atlas_capability_released": True,
        "full_diverse_universe_density_under_original_floor": {
            "seed42_clears_floor": native_passed,
            "seed43_clears_floor": native_passed,
            "both_seeds_clear_floor": native_passed,
            "status": "passed" if native_passed else "failed",
            "overridden_by_matched_fineweb_cell": False,
        },
        "broader_diverse_density_preservation_claimed": native_passed,
        "production_document_prompt_transfer_resolved": False,
        "production_readiness_claimed": False,
    }


def _matched(*, passed: bool = True) -> dict:
    return {
        "schema": round0118_nodes.MATCHED_DENSITY_SCHEMA,
        "round_id": "0118",
        "floor_changed_or_tuned": False,
        "full_diverse_universe_density_resolved": False,
        "full_diverse_universe_density_claimed": False,
        "checks": {
            "all_three_seeds_clear_matched_floor": passed,
        },
        "calibration_portability_capability_released": passed,
    }


def test_seed44_loader_binds_exact_r0111_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict = {}

    def fake(**kwargs):
        observed.update(kwargs)
        return {"model": "seed44"}

    monkeypatch.setattr(round0118_nodes, "load_reviewed_model", fake)
    assert round0118_nodes._seed44_model(
        train_output="/train",
        graph_manifest_path="/graph",
        graph_manifest_sha256="a" * 64,
    ) == {"model": "seed44"}
    assert observed == {
        "train_output": "/train",
        "graph_manifest_path": "/graph",
        "graph_manifest_sha256": "a" * 64,
        "expected_train_round_id": "0111",
        "expected_train_receipt_schema": (
            "round0111-diverse-jina-train-receipt-v1"
        ),
        "expected_production_config_schema": (
            "round0111-production-config-v1"
        ),
        "expected_seed": 44,
    }


def test_seed44_wrapper_restores_frozen_module_globals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = {
        name: getattr(round0108_nodes, name)
        for name in round0118_nodes._FROZEN_BINDINGS
    }

    def fake_run(active, job):
        assert round0108_nodes.ROUND_ID == "0118"
        assert round0108_nodes.MAP_KEY == round0118_nodes.MAP_KEY
        assert (
            round0108_nodes.load_reviewed_model
            is round0118_nodes._seed44_model
        )
        return {"ok": True}

    monkeypatch.setattr(round0108_nodes, "run_transform", fake_run)
    assert round0118_nodes.run_transform({}, {}) == {"ok": True}
    assert {
        name: getattr(round0108_nodes, name)
        for name in round0118_nodes._FROZEN_BINDINGS
    } == before


def test_matched_scope_cannot_rescue_failed_prior_native_density() -> None:
    decision = round0118_nodes.three_seed_decision(
        prior=_prior(native_passed=False),
        seed44_core=_core(density=True),
        seed44_ood=_ood(),
        matched=_matched(passed=True),
    )
    assert decision["seed44_atlas_quality_capability_released"] is True
    assert decision["three_seed_quality_capability_released"] is False
    assert (
        decision[
            "matched_fineweb_qualified_atlas_capability_released"
        ]
        is True
    )
    assert decision[
        "full_diverse_universe_density_under_original_floor"
    ]["status"] == "failed"
    assert decision[
        "full_diverse_universe_density_under_original_floor"
    ]["overridden_by_matched_fineweb_cell"] is False
    assert decision["production_readiness_claimed"] is False
    assert all(
        cell["role"] == "diagnostic-only"
        for cell in decision["three_seed_diagnostics"].values()
    )


def test_seed44_native_density_failure_cannot_be_rescued() -> None:
    decision = round0118_nodes.three_seed_decision(
        prior=_prior(native_passed=True),
        seed44_core=_core(density=False),
        seed44_ood=_ood(),
        matched=_matched(passed=True),
    )
    assert decision["seed44_atlas_quality_capability_released"] is False
    assert decision["three_seed_quality_capability_released"] is False
    assert decision[
        "full_diverse_universe_density_under_original_floor"
    ]["seed44_clears_floor"] is False
    assert decision[
        "full_diverse_universe_density_under_original_floor"
    ]["overridden_by_matched_fineweb_cell"] is False


def test_three_seed_decision_rejects_production_overclaim() -> None:
    prior = _prior()
    prior["production_readiness_claimed"] = True
    with pytest.raises(Round0108Error, match="semantics changed"):
        round0118_nodes.three_seed_decision(
            prior=prior,
            seed44_core=_core(),
            seed44_ood=_ood(),
            matched=_matched(),
        )


def test_round0118_timing_is_receipt_calibrated_and_bounded() -> None:
    assert TRANSFORM_P90_S > 2 * 37.43145924899727
    assert CORE_P90_S > 2 * 90.80119433626533
    assert OOD_P90_S > 2 * 111.14488627016544
    assert MATCHED_DENSITY_P90_S > 2 * 13.835361725185066
    assert GPU_P90_S == (
        TRANSFORM_P90_S
        + CORE_P90_S
        + OOD_P90_S
        + MATCHED_DENSITY_P90_S
    )
    assert GPU_P90_S <= 15 * 60


def test_prepare_queue_materializes_frozen_seed44_job_graph(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def create_directory(path: str, **_kwargs) -> str:
        Path(path).mkdir(parents=True, exist_ok=False)
        return path

    def ensure_directory(path: str, **_kwargs) -> str:
        Path(path).mkdir(parents=True, exist_ok=True)
        return path

    def write_json(path: str, value: dict, **_kwargs) -> None:
        Path(path).write_text(
            json.dumps(value, sort_keys=True), encoding="utf-8"
        )

    monkeypatch.setattr(
        prepare_round0118_queue,
        "create_fresh_directory",
        create_directory,
    )
    monkeypatch.setattr(
        prepare_round0118_queue,
        "ensure_data_directory",
        ensure_directory,
    )
    monkeypatch.setattr(
        prepare_round0118_queue,
        "atomic_write_new_json",
        write_json,
    )
    labs = tmp_path / "labs"
    labs.mkdir()
    round_file = labs / "round-0118-2026-07-31.md"
    round_file.write_text(
        "---\nround_id: \"0118\"\nstatus: issued\n---\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        prepare_round0118_queue,
        "ROUND_FILE_GLOB",
        str(labs / "round-0118-*.md"),
    )

    def review(round_id: str) -> tuple[Path, str]:
        path = labs / f"review-{round_id}-2026-07-31.md"
        path.write_text(
            f"---\nround_id: \"{round_id}\"\nstatus: accepted\n---\n",
            encoding="utf-8",
        )
        return path, expected_input_signature(path)["sha256"]

    r0110_review, r0110_sha = review("0110")
    r0111_review, r0111_sha = review("0111")

    def terminal(root: Path, round_id: str, jobs: list[str]) -> None:
        root.mkdir(parents=True, exist_ok=True)
        (root / "runner-terminal.json").write_text(json.dumps({
            "schema": "slim-runner-terminal-v3",
            "round_id": round_id,
            "verdict": "succeeded",
            "completed_jobs": jobs,
            "required_jobs": jobs,
            "release_checkout_unchanged": True,
            "queue_manifest_unchanged": True,
        }), encoding="utf-8")

    r0108 = tmp_path / "r0108"
    terminal(r0108, "0108", ["source"])
    (r0108 / "inputs").mkdir()
    np.savez(
        r0108 / "inputs" / "registered-selections.npz",
        smoke=np.arange(2),
    )
    calibration = r0108 / "artifacts" / "jina-density-calibration"
    calibration.mkdir(parents=True)
    (calibration / "jina-density-calibration.json").write_text(
        "{}", encoding="utf-8"
    )
    np.savez(
        calibration / "jina-density-calibration-arrays.npz",
        smoke=np.arange(2),
    )
    source = tmp_path / "source.npy"
    np.save(source, np.zeros((2, 2), dtype=np.float16))
    census = tmp_path / "census.json"
    census.write_text(json.dumps({
        "source": expected_input_signature(source),
    }), encoding="utf-8")
    reference = tmp_path / "reference.npz"
    np.savez(reference, smoke=np.arange(2))
    graph = tmp_path / "graph.json"
    graph.write_text("{}", encoding="utf-8")
    common_source = {"expected_inputs": []}
    source_jobs = [
        {
            **common_source,
            "id": "calibration",
            "action": "calibrate_jina_density",
            "census_receipt": str(census),
            "census_receipt_sha256": expected_input_signature(census)[
                "sha256"
            ],
            "representative_reference": str(reference),
            "representative_reference_sha256": expected_input_signature(
                reference
            )["sha256"],
        },
        {
            **common_source,
            "id": "transform",
            "action": "transform_retained_map",
            "graph_manifest": str(graph),
            "graph_manifest_sha256": expected_input_signature(graph)[
                "sha256"
            ],
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
            },
        },
        {
            **common_source,
            "id": "core",
            "action": "score_core_geometry",
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
            },
        },
        {
            **common_source,
            "id": "ood",
            "action": "score_ood",
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
            },
        },
    ]
    (r0108 / "queue.json").write_text(json.dumps({
        "schema": "round0108-diverse-jina-evaluation-queue-v1",
        "round_id": "0108",
        "jobs": source_jobs,
    }), encoding="utf-8")

    r0110 = tmp_path / "r0110"
    terminal(r0110, "0110", ["evaluation"])
    (r0110 / "queue.json").write_text(json.dumps({
        "schema": "round0110-diverse-jina-seed43-evaluation-queue-v2",
        "round_id": "0110",
        "jobs": [],
    }), encoding="utf-8")
    r0110_matched = (
        r0110 / "artifacts" / "matched-calibration-density"
    )
    r0110_matched.mkdir(parents=True)
    (r0110_matched / "matched-density.json").write_text(
        "{}", encoding="utf-8"
    )
    np.savez(
        r0110_matched / "matched-density-arrays.npz",
        smoke=np.arange(2),
    )
    r0110_decision = r0110 / "artifacts" / "two-seed-decision"
    r0110_decision.mkdir(parents=True)
    (r0110_decision / "two-seed-decision.json").write_text(
        "{}", encoding="utf-8"
    )

    r0111 = tmp_path / "r0111"
    terminal(r0111, "0111", ["train"])
    (r0111 / "queue.json").write_text(json.dumps({
        "schema": "round0111-diverse-jina-seed44-training-queue-v1",
        "round_id": "0111",
        "jobs": [],
    }), encoding="utf-8")
    train = r0111 / "artifacts" / "train-diverse-jina-25m-seed44"
    train.mkdir(parents=True)
    for name in ("train-receipt.json", "production-config.json", "model.pt"):
        (train / name).write_bytes(name.encode())

    output = tmp_path / "r0118" / "queue"
    path = prepare_round0118_queue.prepare_round0118(
        release_sha="a" * 40,
        r0110_review_path=str(r0110_review),
        r0110_review_sha256=r0110_sha,
        r0111_review_path=str(r0111_review),
        r0111_review_sha256=r0111_sha,
        r0108_queue_path=str(r0108 / "queue.json"),
        r0110_queue_path=str(r0110 / "queue.json"),
        r0111_queue_path=str(r0111 / "queue.json"),
        r0111_train_output=str(train),
        queue_root=str(output),
    )
    queue = json.loads(Path(path).read_text(encoding="utf-8"))
    assert queue["p90_gpu_seconds"]["total"] == GPU_P90_S
    assert [job["action"] for job in queue["jobs"]] == [
        "transform_seed44",
        "score_seed44_core",
        "score_seed44_ood",
        "score_seed44_matched_fineweb_density",
        "decide_three_seed_stability_and_publish_registry",
    ]
    assert queue["jobs"][-1]["node_policy"]["gpu_required"] is False
    assert queue["scientific_contract"][
        "production_readiness_claimed"
    ] is False


def test_map_registry_discovers_explicit_round0118_seed44_atlas(
    tmp_path: Path,
) -> None:
    round_dir = tmp_path / "round-0118"
    queue = round_dir / "queue"
    artifacts = queue / "artifacts"
    transform = artifacts / "coordinates-seed44"
    core = artifacts / "core-geometry-seed44"
    decision = artifacts / "three-seed-decision"
    definitions = artifacts / "semantic-renders"
    for path in (transform, core, decision, definitions):
        path.mkdir(parents=True, exist_ok=True)
    (queue / "queue.json").write_text(
        json.dumps({"release_sha": "a" * 40}), encoding="utf-8"
    )
    (transform / "actual-transform.json").write_text(json.dumps({
        "round_id": "0118",
        "map_key": round0118_nodes.MAP_KEY,
        "row_accounting": {
            "all_rows": 24_948_663,
            "retained_representatives": 24_948_663,
        },
        "model": {"sha256": "b" * 64},
    }), encoding="utf-8")
    (core / "core-geometry.json").write_text(json.dumps({
        "schema": round0118_nodes.CORE_SCHEMA,
        "metrics": {
            "global": {"ffr": 0.6},
            "density_v2": {"correlation": 0.17},
        },
        "geometry_diagnostics": {"finite": True},
    }), encoding="utf-8")
    (decision / "three-seed-decision.json").write_text(json.dumps({
        "schema": round0118_nodes.DECISION_SCHEMA,
        "map_key": round0118_nodes.MAP_KEY,
        "seed44_atlas_quality_capability_released": True,
        "production_readiness_claimed": False,
    }), encoding="utf-8")
    (definitions / "map-definition.json").write_text(json.dumps({
        "schema": round0118_nodes.MAP_DEFINITION_SCHEMA,
        "round_id": "0118",
        "map_key": round0118_nodes.MAP_KEY,
        "map_label": round0118_nodes.MAP_LABEL,
        "training_round": "0111",
        "embedding_prompt": "raw",
        "prompt_applied": False,
        "production_document_prompt_transfer_resolved": False,
        "production_ready": False,
    }), encoding="utf-8")
    maps = map_registry.scan_round0118_atlas(
        round_dir, {}, queue_dir=queue
    )
    assert len(maps) == 1
    assert maps[0]["map_id"] == (
        "round-0118-r0111-diverse-jina-25m-seed44"
    )
    assert maps[0]["training_round"] == "0111"
    assert maps[0]["production_ready"] is False
