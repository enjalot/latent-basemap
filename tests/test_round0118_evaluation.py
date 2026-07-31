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


def test_review_admission_requires_the_registered_release(
    tmp_path: Path,
) -> None:
    path = tmp_path / "review-0111-2026-07-31.md"
    path.write_text(
        "---\nround_id: \"0111\"\nstatus: accepted\n"
        "releases: [\"unrelated-negative-evidence\"]\n---\n",
        encoding="utf-8",
    )
    signature = expected_input_signature(path)
    with pytest.raises(RuntimeError, match="release-complete"):
        prepare_round0118_queue._require_accepted_review(
            str(path),
            expected_sha256=signature["sha256"],
            round_id="0111",
            required_release=(
                prepare_round0118_queue.REVIEW_RELEASES["0111"]
            ),
        )

    path.write_text(
        "---\nround_id: \"0111\"\nstatus: accepted\n"
        "releases: ["
        f"\"{prepare_round0118_queue.REVIEW_RELEASES['0111']}\""
        "]\n---\n",
        encoding="utf-8",
    )
    signature = expected_input_signature(path)
    assert prepare_round0118_queue._require_accepted_review(
        str(path),
        expected_sha256=signature["sha256"],
        round_id="0111",
        required_release=prepare_round0118_queue.REVIEW_RELEASES["0111"],
    ) == signature


def _r0111_fixture(
    root: Path,
    *,
    documents: Path,
) -> dict[str, object]:
    release = "d" * 40
    queue_root = root / "r0111"
    train = (
        queue_root
        / "artifacts"
        / "train-diverse-jina-25m-seed44"
    )
    train.mkdir(parents=True)
    for name in ("train-receipt.json", "production-config.json", "model.pt"):
        (train / name).write_bytes(name.encode())
    queue_path = queue_root / "queue.json"
    queue_path.write_text(json.dumps({
        "schema": "round0111-diverse-jina-seed44-training-queue-v1",
        "round_id": "0111",
        "release_sha": release,
        "repo_root": prepare_round0118_queue.RELEASE_ROOT,
        "training_performed": True,
        "capabilities_produced": [
            "jina-diverse-25m-full768-trained-map-seed44-v1"
        ],
        "jobs": [{
            "id": "train",
            "action": "train_diverse_jina_seed44",
            "outputs": [str(train)],
            "node_policy": {
                "gpu_required": True,
                "training_performed": True,
            },
        }],
    }), encoding="utf-8")
    queue_sha256 = expected_input_signature(queue_path)["sha256"]
    checkout = {
        "repo_root": prepare_round0118_queue.RELEASE_ROOT,
        "head": release,
        "detached": True,
        "dirty": False,
    }
    terminal_path = queue_root / "runner-terminal.json"
    terminal_path.write_text(json.dumps({
        "schema": "slim-runner-terminal-v3",
        "round_id": "0111",
        "verdict": "succeeded",
        "stop_reason": None,
        "completed_jobs": ["train"],
        "required_jobs": ["train"],
        "gpu_wall_s": 10.0,
        "prior_attempt_gpu_wall_s": 0.0,
        "invocation_gpu_wall_s": 10.0,
        "gpu_wall_accounting_complete": True,
        "release_checkout": checkout,
        "release_checkout_at_finish": checkout,
        "release_checkout_unchanged": True,
        "queue_manifest_sha256": queue_sha256,
        "queue_manifest_sha256_at_finish": queue_sha256,
        "queue_manifest_unchanged": True,
        "boundary_problems": [],
        "nodes": [{
            "node": "train",
            "returncode": 0,
            "gpu_required": True,
            "validation_problems": [],
        }],
    }), encoding="utf-8")
    result_path = documents / "result-0111-2026-07-31.md"
    result_path.write_text(
        "\n".join([
            "---",
            'round_id: "0111"',
            "status: complete",
            f'release_commit: "{release}"',
            f'queue_manifest: "gsv:{queue_path}"',
            f'queue_manifest_sha256: "{queue_sha256}"',
            "---",
            "result",
            "",
        ]),
        encoding="utf-8",
    )
    result_sha256 = expected_input_signature(result_path)["sha256"]
    review_path = documents / "review-0111-2026-07-31.md"
    review_path.write_text(
        "\n".join([
            "---",
            'round_id: "0111"',
            "status: accepted",
            f"result: {result_path.name}",
            f'result_sha256: "{result_sha256}"',
            f'verified_release_commit: "{release}"',
            (
                'releases: ["capability:jina-diverse-25m-full768-'
                'trained-map-seed44-v1"]'
            ),
            "---",
            "review",
            "",
        ]),
        encoding="utf-8",
    )
    return {
        "queue_root": queue_root,
        "queue_path": queue_path,
        "terminal_path": terminal_path,
        "train": train,
        "result_path": result_path,
        "review_path": review_path,
        "review_sha256": expected_input_signature(review_path)["sha256"],
        "release": release,
    }


def test_r0111_admission_closes_review_result_queue_terminal_and_output(
    tmp_path: Path,
) -> None:
    fixture = _r0111_fixture(tmp_path, documents=tmp_path)
    review = prepare_round0118_queue._require_accepted_r0111_review(
        str(fixture["review_path"]),
        expected_sha256=str(fixture["review_sha256"]),
    )
    execution = prepare_round0118_queue._require_exact_r0111_execution(
        str(fixture["queue_path"]),
        review=review,
        train_output=str(fixture["train"]),
    )
    assert execution["queue_signature"]["sha256"] == review["queue_sha256"]
    assert execution["train_output"] == str(
        Path(fixture["train"]).resolve()
    )

    with pytest.raises(RuntimeError, match="queue/release/train output"):
        prepare_round0118_queue._require_exact_r0111_execution(
            str(fixture["queue_path"]),
            review=review,
            train_output=str(tmp_path / "unreviewed-train"),
        )

    result_path = Path(fixture["result_path"])
    result_path.write_text(
        result_path.read_text(encoding="utf-8") + "changed\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="does not close"):
        prepare_round0118_queue._require_accepted_r0111_review(
            str(fixture["review_path"]),
            expected_sha256=str(fixture["review_sha256"]),
        )

    terminal_path = Path(fixture["terminal_path"])
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["nodes"][0]["validation_problems"] = ["synthetic drift"]
    terminal_path.write_text(json.dumps(terminal), encoding="utf-8")
    with pytest.raises(RuntimeError, match="exact clean reviewed success"):
        prepare_round0118_queue._require_exact_r0111_execution(
            str(fixture["queue_path"]),
            review=review,
            train_output=str(fixture["train"]),
        )


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

    def review(round_id: str, release: str) -> tuple[Path, str]:
        path = labs / f"review-{round_id}-2026-07-31.md"
        path.write_text(
            f"---\nround_id: \"{round_id}\"\nstatus: accepted\n"
            f"releases: [\"{release}\"]\n---\n",
            encoding="utf-8",
        )
        return path, expected_input_signature(path)["sha256"]

    r0108_review, r0108_sha = review(
        "0108", prepare_round0118_queue.REVIEW_RELEASES["0108"]
    )
    r0110_review, r0110_sha = review(
        "0110", prepare_round0118_queue.REVIEW_RELEASES["0110"]
    )

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

    r0111 = _r0111_fixture(tmp_path, documents=labs)
    train = Path(r0111["train"])
    r0111_review = Path(r0111["review_path"])
    r0111_sha = str(r0111["review_sha256"])

    output = tmp_path / "r0118" / "queue"
    path = prepare_round0118_queue.prepare_round0118(
        release_sha="a" * 40,
        r0108_review_path=str(r0108_review),
        r0108_review_sha256=r0108_sha,
        r0110_review_path=str(r0110_review),
        r0110_review_sha256=r0110_sha,
        r0111_review_path=str(r0111_review),
        r0111_review_sha256=r0111_sha,
        r0108_queue_path=str(r0108 / "queue.json"),
        r0110_queue_path=str(r0110 / "queue.json"),
        r0111_queue_path=str(r0111["queue_path"]),
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
    assert queue["jobs"][-1]["outputs"] == [
        str(output / "artifacts" / "three-seed-decision"),
        str(output / "artifacts" / "semantic-renders"),
    ]
    assert queue["required_reviews"] == ["0108", "0110", "0111"]
    assert queue["scientific_contract"][
        "production_readiness_claimed"
    ] is False
    expected_paths = {
        item["canonical_path"]
        for job in queue["jobs"]
        for item in job["expected_inputs"]
    }
    assert str(Path(r0111["result_path"]).resolve()) in expected_paths

    terminal_path = Path(r0111["terminal_path"])
    r0111_terminal = json.loads(
        terminal_path.read_text(encoding="utf-8")
    )
    r0111_terminal["gpu_wall_accounting_complete"] = False
    terminal_path.write_text(json.dumps(r0111_terminal), encoding="utf-8")
    rejected_output = tmp_path / "r0118-rejected" / "queue"
    with pytest.raises(RuntimeError, match="exact clean reviewed success"):
        prepare_round0118_queue.prepare_round0118(
            release_sha="a" * 40,
            r0108_review_path=str(r0108_review),
            r0108_review_sha256=r0108_sha,
            r0110_review_path=str(r0110_review),
            r0110_review_sha256=r0110_sha,
            r0111_review_path=str(r0111_review),
            r0111_review_sha256=r0111_sha,
            r0108_queue_path=str(r0108 / "queue.json"),
            r0110_queue_path=str(r0110 / "queue.json"),
            r0111_queue_path=str(r0111["queue_path"]),
            r0111_train_output=str(train),
            queue_root=str(rejected_output),
        )
    assert not rejected_output.exists()


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


def test_round0118_registry_scan_and_publication_close_base_and_probes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    round_dir = tmp_path / "round-0118"
    queue = round_dir / "queue"
    artifacts = queue / "artifacts"
    transform = artifacts / "coordinates-seed44"
    core = artifacts / "core-geometry-seed44"
    ood = artifacts / "ood-seed44"
    decision = artifacts / "three-seed-decision"
    definitions = artifacts / "semantic-renders"
    for path in (transform, core, ood, decision, definitions):
        path.mkdir(parents=True, exist_ok=True)
    (transform / "chunk-00000").mkdir()
    np.save(
        transform / "chunk-00000" / "coordinates.npy",
        np.zeros((2, 2), dtype=np.float32),
    )
    queue_path = queue / "queue.json"
    queue_path.write_text(
        json.dumps({"release_sha": "a" * 40}), encoding="utf-8"
    )
    transform_path = transform / "actual-transform.json"
    transform_path.write_text(json.dumps({
        "round_id": "0118",
        "map_key": round0118_nodes.MAP_KEY,
        "row_accounting": {
            "all_rows": 24_948_663,
            "retained_representatives": 24_948_663,
        },
        "model": {"sha256": "b" * 64},
    }), encoding="utf-8")
    core_path = core / "core-geometry.json"
    core_path.write_text(json.dumps({
        "schema": round0118_nodes.CORE_SCHEMA,
        "metrics": {
            "global": {"ffr": 0.6},
            "density_v2": {"correlation": 0.17},
        },
        "geometry_diagnostics": {"finite": True},
    }), encoding="utf-8")
    decision_path = decision / "three-seed-decision.json"
    decision_path.write_text(json.dumps({
        "schema": round0118_nodes.DECISION_SCHEMA,
        "map_key": round0118_nodes.MAP_KEY,
        "seed44_atlas_quality_capability_released": True,
        "production_readiness_claimed": False,
    }), encoding="utf-8")
    sample_path = definitions / "sample-semantic-ids.npy"
    np.save(sample_path, np.asarray([3, 7], dtype=np.int64))
    definition_path = definitions / "map-definition.json"
    definition_path.write_text(json.dumps({
        "schema": round0118_nodes.MAP_DEFINITION_SCHEMA,
        "round_id": "0118",
        "map_key": round0118_nodes.MAP_KEY,
        "map_label": round0118_nodes.MAP_LABEL,
        "training_round": "0111",
        "embedding_prompt": "raw",
        "prompt_applied": False,
        "production_document_prompt_transfer_resolved": False,
        "production_ready": False,
        "coordinates": expected_input_signature(transform_path),
        "core_panel": expected_input_signature(core_path),
        "decision": expected_input_signature(decision_path),
        "sample_ids": expected_input_signature(sample_path),
    }), encoding="utf-8")

    probes = {}
    probe_names = {
        "dadabase",
        "fineweb-heldout",
        "pol_Latn",
        "trec-covid",
    }
    for index, probe in enumerate(sorted(probe_names)):
        coordinate_path = ood / f"{probe}-coordinates.npz"
        np.savez(
            coordinate_path,
            probe_corpus_coords=np.zeros((2, 2), dtype=np.float32),
            probe_query_coords=np.full(
                (1, 2), index + 1, dtype=np.float32
            ),
        )
        probes[probe] = {
            "status": "included",
            "probe": {
                "corpus_rows": 2,
                "query_rows": 1,
                "ffr": 0.5,
            },
            "coordinates": expected_input_signature(coordinate_path),
            "inputs": {},
            "truth": {},
            "selection": {},
            "duplicate_control": {},
            "verdict": "diagnostic-only",
        }
    panel_path = ood / "universality-panel-v1.json"
    panel_path.write_text(json.dumps({
        "schema": "universality-panel-v1",
        "round_id": "0118",
        "map": {
            "label": round0118_nodes.MAP_LABEL,
            "model": {"sha256": "b" * 64},
            "coordinate_receipt": expected_input_signature(transform_path),
        },
        "probes": probes,
    }), encoding="utf-8")

    base_maps = map_registry.scan_round0118_atlas(
        round_dir, {}, queue_dir=queue
    )
    projection_maps = map_registry.scan_projection_maps(
        round_dir, {}, queue_dir=queue
    )
    assert len(base_maps) == 1
    assert {
        item["projection"]["probe"] for item in projection_maps
    } == probe_names
    assert all(
        item["base_sample_ids"]
        == {"path": f"gsv:{sample_path}"}
        for item in projection_maps
    )
    registry = {
        "schema": map_registry.SCHEMA,
        "maps": [*base_maps, *projection_maps],
    }
    registry_path = tmp_path / "maps.json"
    history_path = tmp_path / "maps-history.json"
    site = tmp_path / "site"

    def write_registry(value: dict) -> Path:
        registry_path.write_text(json.dumps(value), encoding="utf-8")
        history_path.write_text(json.dumps(value), encoding="utf-8")
        return history_path

    def publish(value: dict) -> None:
        atlas = site / "round-0118"
        atlas.mkdir(parents=True)
        (atlas / "index.html").write_text("atlas", encoding="utf-8")
        for item in value["maps"]:
            if item.get("kind") != "projection-map":
                continue
            root = site / "projections" / item["map_id"]
            root.mkdir(parents=True)
            (root / "index.html").write_text(
                "projection", encoding="utf-8"
            )
            (root / "manifest.json").write_text(
                "{}", encoding="utf-8"
            )

    monkeypatch.setattr(map_registry, "REGISTRY_PATH", registry_path)
    monkeypatch.setattr(map_registry, "SITE_DIR", site)
    monkeypatch.setattr(map_registry, "scan", lambda: registry)
    monkeypatch.setattr(map_registry, "write_registry", write_registry)
    monkeypatch.setattr(map_registry, "publish", publish)
    publication_path = decision / "registry-publication.json"
    publication = round0118_nodes._refresh_registry_best_effort(
        receipt_path=str(publication_path),
        map_definition_path=str(definition_path),
        decision_path=str(decision_path),
        round_id="0118",
        map_key=round0118_nodes.MAP_KEY,
        publication_schema=round0118_nodes.REGISTRY_PUBLICATION_SCHEMA,
    )
    assert publication_path.is_file()
    assert publication["schema"] == (
        round0118_nodes.REGISTRY_PUBLICATION_SCHEMA
    )
    assert publication["expected_map_ids"] == [
        f"round-0118-{round0118_nodes.MAP_KEY}"
    ]
    assert set(publication["expected_projection_probes"]) == probe_names
    refresh = publication["mutable_view_refresh"]
    assert refresh["status"] == "published"
    assert refresh["requires_followup"] is False
    assert refresh["stages"]["inventory_validation"]["status"] == (
        "completed"
    )
    assert refresh["stages"]["site_artifacts"]["status"] == "completed"
