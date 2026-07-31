from __future__ import annotations

import json
from pathlib import Path

import pytest

from basemap.artifact_identity import expected_input_signature
from basemap.round0113_prompt_contrast import seal
from basemap.round0124_degree_bridge import (
    BOOTSTRAP_CI_LEVEL,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    DECISION_SCHEMA as R0124_DECISION_SCHEMA,
    OUTCOME_INCONCLUSIVE,
    OUTCOME_MATERIAL,
)
from basemap.round0129_degree_replicate import (
    CAPABILITY as R0129_CAPABILITY,
    DECISION_SCHEMA as R0129_DECISION_SCHEMA,
    SEED43_INITIAL_STATE_SHA256,
    SEED43_PARAMETER_COUNT,
)
from basemap.round0130_k49_rescue import (
    ATLAS_QUALITY_CAPABILITY,
    DECISION_SCHEMA,
    DEGREE_RESCUE_CAPABILITY,
    FIXED_SUCCESSFUL_UPDATES,
    GRAPH_SCHEMA,
    ROUND_ID,
)
from experiments.prepare_round0130_queue import (
    GPU_HOURS_CAP,
    P90_GPU_TOTAL_SECONDS,
    R0129_CORRECTED_QUEUE_SHA256,
    R0129_CORRECTED_RELEASE_SHA,
    R0129_CORRECTED_ROUND_SHA256,
    R0129_ROOT,
    _assert_two_seed_contract_equal,
    _require_inconclusive_r0124_review,
    _require_positive_r0129_review,
)
from experiments.round0130_nodes import GRAPH_CONTRACT


def _selector(
    *,
    outcome: str,
    delta: float,
    low: float,
    high: float,
) -> dict:
    control = 0.2116
    return {
        "outcome": outcome,
        "control_density": control,
        "treatment_density": control + delta,
        "treatment_minus_control": delta,
        "paired_bootstrap_delta_ci": [low, high],
        "paired_bootstrap_ci_level": BOOTSTRAP_CI_LEVEL,
        "paired_bootstrap_draws": BOOTSTRAP_DRAWS,
        "paired_bootstrap_seed": BOOTSTRAP_SEED,
        "material_degradation_threshold": -0.03,
        "selector_metrics": ["native-density-correlation-delta"],
        "core_and_ood_diagnostics_can_rescue_or_fail": False,
        "legacy_density_floor_used": False,
        "single_cause_beyond_graph_degree_claimed": False,
    }


def _native_reference(*, anchor_seed: int = 123) -> dict:
    return {
        "high_d_reference": {
            "kind": "file",
            "canonical_path": "/frozen/high-d-reference.npz",
            "bytes": 1234,
            "sha256": "9" * 64,
        },
        "anchor_count": 4_000,
        "anchor_seed": anchor_seed,
        "k_density": 15,
        "low_d_search": "panel-v2 exact global chunked top-k; mean k15 radius",
    }


def _scientific_contract() -> dict:
    return {
        "changed_factor": "fuzzy graph neighbor degree only",
        "population_rows": 1_993_761,
        "graph_nonself_neighbors": 15,
        "graph_search_neighbors_including_self": 16,
        "successful_updates": 500_000,
        "paired_bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "ci_level": BOOTSTRAP_CI_LEVEL,
            "material_density_degradation": 0.03,
        },
    }


def _write_json(path: Path, value: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return expected_input_signature(str(path))


def _write_sealed(path: Path, value: dict) -> dict:
    return _write_json(path, seal(value))


def _finish_documents(
    root: Path,
    *,
    round_id: str,
    release: str,
    capability: str,
    queue_path: Path,
    queue_signature: dict,
    terminal_signature: dict,
    decision_signature: dict,
    review_prose: str,
) -> tuple[Path, str]:
    result_path = root / f"result-{round_id}-2026-07-31.md"
    result_path.write_text(
        "\n".join(
            [
                "---",
                f'round_id: "{round_id}"',
                "status: complete",
                f'release_commit: "{release}"',
                f'queue_manifest: "gsv:{queue_path}"',
                f'queue_manifest_sha256: "{queue_signature["sha256"]}"',
                "---",
                f"# Result {round_id}",
                f'Queue `{queue_signature["sha256"]}`.',
                f'Terminal `{terminal_signature["sha256"]}`.',
                f'Decision `{decision_signature["sha256"]}`.',
                "",
            ]
        ),
        encoding="utf-8",
    )
    result_signature = expected_input_signature(str(result_path))
    review_path = root / f"review-{round_id}-2026-07-31.md"
    review_path.write_text(
        "\n".join(
            [
                "---",
                f'round_id: "{round_id}"',
                "status: accepted",
                f"result: {result_path.name}",
                f'result_sha256: "{result_signature["sha256"]}"',
                f'verified_release_commit: "{release}"',
                f'releases: ["capability:{capability}"]',
                "---",
                f"# Review {round_id}",
                review_prose,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return review_path, expected_input_signature(str(review_path))["sha256"]


def _r0124_fixture(
    root: Path,
    *,
    outcome: str = OUTCOME_INCONCLUSIVE,
    delta: float = -0.031,
    low: float = -0.052,
    high: float = -0.009,
    review_prose: str = "accepted",
) -> dict:
    release = "a" * 40
    selector = _selector(outcome=outcome, delta=delta, low=low, high=high)
    queue_root = root / "queue"
    decision_path = queue_root / "artifacts/degree-bridge-decision/decision.json"
    density_path = queue_root / "artifacts/native-density/native-density-score.json"
    density_signature = _write_sealed(
        density_path,
        {
            "schema": "round0124-native-density-score-v1",
            "round_id": "0124",
            "release_sha": release,
            "registered_selector": selector,
            "native_reference": _native_reference(),
            "changed_factor": "fuzzy graph neighbor degree only",
        },
    )
    retry = {"schema": "test-r0124-retry-v1"}
    decision_signature = _write_sealed(
        decision_path,
        {
            "schema": R0124_DECISION_SCHEMA,
            "round_id": "0124",
            "release_sha": release,
            "retry_provenance": retry,
            "density_score": density_signature,
            "registered_selector": selector,
            "capabilities_produced": [
                "jina-fineweb-2m-native-k15-degree-bridge-v1"
            ],
        },
    )
    queue_path = queue_root / "queue.json"
    queue = {
        "schema": "round0124-fineweb-2m-degree-bridge-retry-queue-v1",
        "round_id": "0124",
        "release_sha": release,
        "retry_provenance": retry,
        "capabilities_produced": [
            "jina-fineweb-2m-native-k15-degree-bridge-v1"
        ],
        "scientific_contract": _scientific_contract(),
        "jobs": [
            {
                "id": "decide_degree_bridge",
                "outputs": [str(decision_path.parent.resolve())],
            }
        ],
    }
    queue_signature = _write_json(queue_path, queue)
    terminal_path = queue_root / "runner-terminal.json"
    terminal_signature = _write_json(
        terminal_path,
        {
            "schema": "slim-runner-terminal-v3",
            "round_id": "0124",
            "verdict": "succeeded",
            "required_jobs": ["decide_degree_bridge"],
            "completed_jobs": ["decide_degree_bridge"],
            "queue_manifest_sha256": queue_signature["sha256"],
            "queue_manifest_sha256_at_finish": queue_signature["sha256"],
            "queue_manifest_unchanged": True,
            "release_checkout_unchanged": True,
            "gpu_wall_accounting_complete": True,
            "boundary_problems": [],
            "nodes": [
                {"node": "decide_degree_bridge", "validation_problems": []}
            ],
            "release_checkout": {"head": release},
            "release_checkout_at_finish": {"head": release},
        },
    )
    review, review_sha = _finish_documents(
        root,
        round_id="0124",
        release=release,
        capability="jina-fineweb-2m-native-k15-degree-bridge-v1",
        queue_path=queue_path,
        queue_signature=queue_signature,
        terminal_signature=terminal_signature,
        decision_signature=decision_signature,
        review_prose=review_prose,
    )
    return {
        "review": str(review),
        "review_sha": review_sha,
        "queue": str(queue_path),
        "terminal": str(terminal_path),
        "decision": str(decision_path),
        "decision_signature": decision_signature,
    }


def _r0129_fixture(
    root: Path,
    *,
    r0124_decision: dict,
    outcome: str = OUTCOME_MATERIAL,
    anchor_seed: int = 123,
    state_sha: str = SEED43_INITIAL_STATE_SHA256,
    review_prose: str = "accepted",
) -> dict:
    release = "b" * 40
    round_sha256 = "c" * 64
    selector = _selector(
        outcome=outcome,
        delta=-0.041,
        low=-0.055,
        high=-0.031 if outcome == OUTCOME_MATERIAL else -0.009,
    )
    queue_root = root / "queue"
    production_path = queue_root / "artifacts/k15-seed43-train/production-config.json"
    isolation = {
        "exact_equal_sections": ["arm", "input", "model", "optimizer"],
        "non_graph_config_equal": True,
        "non_graph_execution_equal": True,
        "sampling_mechanism_equal_conditioned_on_graph": True,
        "positive_edge_distribution_equal": False,
        "negative_sampling_distribution_equal": True,
        "identical_realized_edge_draws_claimed": False,
        "identical_realized_negative_pairs_claimed": False,
        "graph_sampler_policy_fields_equal": True,
        "training_seed": 43,
        "successful_updates": 500_000,
    }
    initial_state = {
        "observed_sha256": state_sha,
        "parameter_count": SEED43_PARAMETER_COUNT,
        "captured_before_optimizer_construction_and_update_zero": True,
        "actual_historical_r0117_bytes_claimed": False,
        "historical_evidence_kind": (
            "deterministic-reconstruction-not-original-reviewed-receipt"
        ),
    }
    production_signature = _write_json(
        production_path,
        {
            "schema": "round0129-production-config-v1",
            "round_id": "0129",
            "config": {
                "input": {"rows": 1_993_761},
                "optimizer": {
                    "seed": 43,
                    "successful_positive_lr_updates": 500_000,
                },
            },
            "config_equivalence": isolation,
        },
    )
    train_path = queue_root / "artifacts/k15-seed43-train/train-receipt.json"
    train_signature = _write_sealed(
        train_path,
        {
            "schema": "round0129-seed43-k15-train-receipt-v1",
            "round_id": "0129",
            "release_sha": release,
            "production_config": production_signature,
            "config_equivalence": isolation,
            "initial_model_state": initial_state,
            "train_accounting": {
                "optimizer_steps_succeeded": 500_000,
                "positive_lr_optimizer_steps": 500_000,
                "amp_overflow_skips": 0,
                "nonfinite_loss_skips": 0,
                "nonfinite_gradient_skips": 0,
            },
            "train_checks": {"exact_update_closure": True},
        },
    )
    density_path = queue_root / "artifacts/native-density-contrast/native-density-score.json"
    density_signature = _write_sealed(
        density_path,
        {
            "schema": "round0129-seed43-native-density-score-v1",
            "round_id": "0129",
            "release_sha": release,
            "registered_selector": selector,
            "native_reference": _native_reference(anchor_seed=anchor_seed),
            "changed_factor": "fuzzy graph neighbor degree only",
            "non_graph_config_equal": True,
            "sampling_mechanism_equal_conditioned_on_graph": True,
            "positive_edge_distribution_equal": False,
        },
    )
    decision_path = queue_root / "artifacts/degree-replicate-decision/decision.json"
    decision_signature = _write_sealed(
        decision_path,
        {
            "schema": R0129_DECISION_SCHEMA,
            "round_id": "0129",
            "release_sha": release,
            "training_seed": 43,
            "r0124_inconclusive_trigger": r0124_decision,
            "density_score": density_signature,
            "train_receipt": train_signature,
            "registered_selector": selector,
            "config_equivalence": isolation,
            "initial_model_state": initial_state,
            "diagnostics_can_rescue_or_fail_selector": False,
            "capabilities_produced": [R0129_CAPABILITY],
            "optimizer_updates": 500_000,
        },
    )
    queue_path = queue_root / "queue.json"
    queue_signature = _write_json(
        queue_path,
        {
            "schema": "round0129-seed43-native-degree-replicate-queue-v1",
            "round_id": "0129",
            "release_sha": release,
            "round_sha256": round_sha256,
            "capabilities_produced": [R0129_CAPABILITY],
            "conditional_trigger": {"decision": r0124_decision},
            "scientific_contract": _scientific_contract(),
            "jobs": [
                {
                    "id": "decide_degree_replicate",
                    "outputs": [str(decision_path.parent.resolve())],
                }
            ],
        },
    )
    terminal_path = queue_root / "runner-terminal.json"
    terminal_signature = _write_json(
        terminal_path,
        {
            "schema": "slim-runner-terminal-v3",
            "round_id": "0129",
            "verdict": "succeeded",
            "required_jobs": ["decide_degree_replicate"],
            "completed_jobs": ["decide_degree_replicate"],
            "queue_manifest_sha256": queue_signature["sha256"],
            "queue_manifest_sha256_at_finish": queue_signature["sha256"],
            "queue_manifest_unchanged": True,
            "release_checkout_unchanged": True,
            "gpu_wall_accounting_complete": True,
            "boundary_problems": [],
            "nodes": [
                {"node": "decide_degree_replicate", "validation_problems": []}
            ],
            "release_checkout": {"head": release},
            "release_checkout_at_finish": {"head": release},
        },
    )
    review, review_sha = _finish_documents(
        root,
        round_id="0129",
        release=release,
        capability=R0129_CAPABILITY,
        queue_path=queue_path,
        queue_signature=queue_signature,
        terminal_signature=terminal_signature,
        decision_signature=decision_signature,
        review_prose=review_prose,
    )
    return {
        "review": str(review),
        "review_sha": review_sha,
        "queue": str(queue_path),
        "terminal": str(terminal_path),
        "decision": str(decision_path),
        "release": release,
        "queue_sha": queue_signature["sha256"],
        "round_sha": round_sha256,
    }


def _authenticate_r0124(fixture: dict) -> dict:
    return _require_inconclusive_r0124_review(
        fixture["review"],
        expected_sha256=fixture["review_sha"],
        queue_path=fixture["queue"],
        terminal_path=fixture["terminal"],
        decision_path=fixture["decision"],
    )


def _authenticate_r0129(fixture: dict, r0124: dict) -> dict:
    return _require_positive_r0129_review(
        fixture["review"],
        expected_sha256=fixture["review_sha"],
        expected_r0124_decision=r0124["signatures"]["decision"],
        queue_path=fixture["queue"],
        terminal_path=fixture["terminal"],
        decision_path=fixture["decision"],
        expected_release_sha=fixture["release"],
        expected_queue_sha256=fixture["queue_sha"],
        expected_round_sha256=fixture["round_sha"],
    )


def test_two_seed_trigger_authenticates_numeric_and_execution_closure(
    tmp_path: Path,
) -> None:
    r0124_fixture = _r0124_fixture(tmp_path / "r0124")
    r0124 = _authenticate_r0124(r0124_fixture)
    r0129_fixture = _r0129_fixture(
        tmp_path / "r0129",
        r0124_decision=r0124["signatures"]["decision"],
    )
    r0129 = _authenticate_r0129(r0129_fixture, r0124)
    proof = _assert_two_seed_contract_equal(r0124, r0129)
    assert proof == {
        "schema": "round0130-two-seed-degree-evidence-v1",
        "seed42_outcome": OUTCOME_INCONCLUSIVE,
        "seed43_outcome": OUTCOME_MATERIAL,
        "intervention_equal": True,
        "native_panel_equal": True,
        "material_margin_equal": True,
        "seeds_pooled": False,
        "degree_hypothesis_proven_claimed": False,
    }


def test_review_prose_cannot_release_wrong_r0124_outcome(tmp_path: Path) -> None:
    fixture = _r0124_fixture(
        tmp_path,
        outcome=OUTCOME_MATERIAL,
        high=-0.031,
        review_prose=(
            "The desired k15-native-density-effect-inconclusive branch is "
            "mentioned here, but did not occur."
        ),
    )
    with pytest.raises(RuntimeError, match="negative-direction inconclusive"):
        _authenticate_r0124(fixture)


def test_review_prose_cannot_release_wrong_r0129_outcome(tmp_path: Path) -> None:
    r0124_fixture = _r0124_fixture(tmp_path / "r0124")
    r0124 = _authenticate_r0124(r0124_fixture)
    fixture = _r0129_fixture(
        tmp_path / "r0129",
        r0124_decision=r0124["signatures"]["decision"],
        outcome=OUTCOME_INCONCLUSIVE,
        review_prose=(
            "k15-materially-degrades-native-density is discussed as the "
            "branch that did not occur."
        ),
    )
    with pytest.raises(RuntimeError, match="exact positive seed-43"):
        _authenticate_r0129(fixture, r0124)


def test_r0124_inconclusive_requires_registered_negative_direction(
    tmp_path: Path,
) -> None:
    fixture = _r0124_fixture(tmp_path, delta=-0.02)
    with pytest.raises(RuntimeError, match="negative-direction inconclusive"):
        _authenticate_r0124(fixture)


def test_state_or_cross_seed_panel_drift_fails_closed(tmp_path: Path) -> None:
    r0124_fixture = _r0124_fixture(tmp_path / "r0124")
    r0124 = _authenticate_r0124(r0124_fixture)
    bad_state = _r0129_fixture(
        tmp_path / "bad-state",
        r0124_decision=r0124["signatures"]["decision"],
        state_sha="0" * 64,
    )
    with pytest.raises(RuntimeError, match="state/config/population"):
        _authenticate_r0129(bad_state, r0124)

    bad_panel = _r0129_fixture(
        tmp_path / "bad-panel",
        r0124_decision=r0124["signatures"]["decision"],
        anchor_seed=124,
    )
    r0129 = _authenticate_r0129(bad_panel, r0124)
    with pytest.raises(RuntimeError, match="intervention, panel, or margin"):
        _assert_two_seed_contract_equal(r0124, r0129)


def test_r0129_trigger_binds_corrected_execution_lineage(tmp_path: Path) -> None:
    assert R0129_ROOT.endswith("/round-0129/queue-correction-1")
    assert R0129_CORRECTED_RELEASE_SHA == (
        "c25c6abaeff74bb3e5ebcc9d85ef5abad6f7fcc9"
    )
    assert R0129_CORRECTED_QUEUE_SHA256 == (
        "0d2fb9b9d8f96df8bb02403a23875a7a7840a41a4cef215b30cb14128e8fe136"
    )
    assert R0129_CORRECTED_ROUND_SHA256 == (
        "893b170769fd9c6d48476fdd89b22cdc192300ec66db4e2f45a85d383ce1e896"
    )

    r0124_fixture = _r0124_fixture(tmp_path / "r0124")
    r0124 = _authenticate_r0124(r0124_fixture)
    fixture = _r0129_fixture(
        tmp_path / "r0129",
        r0124_decision=r0124["signatures"]["decision"],
    )
    common = {
        "review_path": fixture["review"],
        "expected_sha256": fixture["review_sha"],
        "expected_r0124_decision": r0124["signatures"]["decision"],
        "queue_path": fixture["queue"],
        "terminal_path": fixture["terminal"],
        "decision_path": fixture["decision"],
    }
    with pytest.raises(RuntimeError, match="corrected execution lineage"):
        _require_positive_r0129_review(
            **common,
            expected_release_sha="d" * 40,
            expected_queue_sha256=fixture["queue_sha"],
            expected_round_sha256=fixture["round_sha"],
        )
    with pytest.raises(RuntimeError, match="corrected execution lineage"):
        _require_positive_r0129_review(
            **common,
            expected_release_sha=fixture["release"],
            expected_queue_sha256="e" * 64,
            expected_round_sha256=fixture["round_sha"],
        )
    with pytest.raises(RuntimeError, match="corrected execution lineage"):
        _require_positive_r0129_review(
            **common,
            expected_release_sha=fixture["release"],
            expected_queue_sha256=fixture["queue_sha"],
            expected_round_sha256="f" * 64,
        )


def test_round0130_uses_distinct_contract_and_preserves_budget() -> None:
    assert ROUND_ID == "0130"
    assert GRAPH_CONTRACT.round_id == "0130"
    assert GRAPH_SCHEMA.startswith("round0130-")
    assert DECISION_SCHEMA.startswith("round0130-")
    assert "two-seed" in DEGREE_RESCUE_CAPABILITY
    assert "two-seed" in ATLAS_QUALITY_CAPABILITY
    assert FIXED_SUCCESSFUL_UPDATES == 1_459_722
    assert P90_GPU_TOTAL_SECONDS == 23_700.0
    assert GPU_HOURS_CAP == 8.0
