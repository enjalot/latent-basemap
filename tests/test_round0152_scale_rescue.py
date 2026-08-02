"""Focused pure tests for the conditional R0152 rescue rung."""
from __future__ import annotations

import copy

import pytest

from basemap.artifact_identity import canonical_json, sha256_bytes
from basemap.round0107_training import (
    BATCH_SIZE,
    POSITIVE_ROWS_PER_UPDATE,
    Round0107Error,
    train_config,
)
from basemap.round0140_subsystem_bisection import RESTORATION_FLOORS
from basemap.round0152_scale_rescue import (
    DECISION_SCHEMA,
    GRAPH_SCHEMA,
    OUTCOME_FAIL,
    OUTCOME_INVALID,
    OUTCOME_PASS,
    PIPELINE,
    PIPELINE_SCHEMA,
    POSITIVE_DESTINATION_POLICY,
    PRODUCTION_CONFIG_SCHEMA,
    RETAINED_ROWS,
    ROUND_ID,
    TRAIN_CONFIG_SCHEMA,
    TRAIN_RECEIPT_SCHEMA,
    UPDATE_RULE,
    build_decision,
    coverage_aligned_updates,
    quality_selector,
    validate_train_execution,
)
from experiments import round0132_nodes
from experiments.round0152_nodes import _configured_inherited_contract


def _functional_cell(delta: float = 0.0) -> dict:
    return {
        "panel": {
            "ffr": RESTORATION_FLOORS["ffr"] + delta,
            "purity": {
                "k256": RESTORATION_FLOORS["purity_fidelity_k256"] + delta,
                "k1024": RESTORATION_FLOORS["purity_fidelity_k1024"] + delta,
            },
        },
        "projection": {
            "ffr": RESTORATION_FLOORS["projection_ffr"] + delta,
            "recall_at_10": RESTORATION_FLOORS["ood_recall_at_10"] + delta,
        },
    }


def _ood(value: float) -> dict[str, float]:
    return {
        "fineweb_recall_at_50_of_high10": value,
        "polish_recall_at_50_of_high10": value,
        "in_mix_median_recall_at_50_of_high10": value,
    }


def test_quality_selector_is_inclusive_and_requires_every_registered_axis():
    passed = quality_selector(
        functional_cell=_functional_cell(),
        density_v2=0.17589389755990817,
        candidate_ood=_ood(0.194),
        accepted_25m_ood=_ood(0.2),
    )
    assert passed["passed"] is True
    assert all(passed["checks"].values())

    failed = quality_selector(
        functional_cell=_functional_cell(),
        density_v2=0.17589389755990816,
        candidate_ood=_ood(0.194),
        accepted_25m_ood=_ood(0.2),
    )
    assert failed["passed"] is True  # within the registered comparison tolerance

    failed = quality_selector(
        functional_cell=_functional_cell(delta=-1e-4),
        density_v2=0.18,
        candidate_ood=_ood(0.2),
        accepted_25m_ood=_ood(0.2),
    )
    assert failed["passed"] is False
    assert any(not value for value in failed["checks"].values())


def test_decision_releases_only_a_valid_all_axis_pass():
    quality = quality_selector(
        functional_cell=_functional_cell(),
        density_v2=0.18,
        candidate_ood=_ood(0.2),
        accepted_25m_ood=_ood(0.2),
    )
    passed = build_decision(validity_checks={"all": True}, quality=quality)
    assert passed["schema"] == DECISION_SCHEMA
    assert passed["outcome"] == OUTCOME_PASS
    assert passed["atlas_rescue_candidate_released"] is True
    assert passed["registry_promotion_released"] is False

    quality["passed"] = False
    failed = build_decision(validity_checks={"all": True}, quality=quality)
    assert failed["outcome"] == OUTCOME_FAIL
    invalid = build_decision(validity_checks={"all": False}, quality=quality)
    assert invalid["outcome"] == OUTCOME_INVALID


def _fake_graph(edges: int = 4_090) -> tuple[dict, dict]:
    signature = {
        "kind": "file",
        "canonical_path": "/immutable/graph-manifest.json",
        "bytes": 100,
        "sha256": "a" * 64,
    }
    output = {
        key: {
            "kind": "file",
            "canonical_path": f"/immutable/{key}.npy",
            "bytes": 100,
            "sha256": character * 64,
        }
        for key, character in zip(("sources", "targets", "weights"), "bcd")
    }
    mapping = {
        "kind": "file",
        "canonical_path": "/immutable/mapping.npy",
        "bytes": 100,
        "sha256": "e" * 64,
    }
    return ({
        "schema": GRAPH_SCHEMA,
        "round_id": ROUND_ID,
        "retained_rows": RETAINED_ROWS,
        "k_real": 15,
        "n_neighbors_including_self": 16,
        "directed_edge_count": edges,
        "outputs": output,
        "compact_mapping": mapping,
    }, signature)


def _valid_train_bundle() -> tuple[dict, dict, dict]:
    graph, graph_signature = _fake_graph()
    config, digest = train_config(
        graph_manifest=graph,
        graph_signature=graph_signature,
        schema=TRAIN_CONFIG_SCHEMA,
        compact_retained_rows=RETAINED_ROWS,
        pipeline=PIPELINE,
        pipeline_schema=PIPELINE_SCHEMA,
        positive_destination_policy=POSITIVE_DESTINATION_POLICY,
        update_rule=UPDATE_RULE,
    )
    assert digest == sha256_bytes(canonical_json(config))
    updates = coverage_aligned_updates(graph["directed_edge_count"])
    expected_rows = updates * BATCH_SIZE
    expected_draws = updates * POSITIVE_ROWS_PER_UPDATE
    runtime = {
        **config["execution"]["expected_pipeline_stamp"],
        "endpoint_gather_calls": updates,
        "source_rows_gathered": expected_rows,
        "destination_rows_gathered": expected_rows,
        "host_prefetch_batches_filled": updates + 1,
        "host_prefetch_producer_batches": updates + 1,
        "host_prefetch_consumer_batches": updates,
        "host_prefetch_source_rows_filled": expected_rows,
        "host_prefetch_destination_rows_filled": expected_rows,
        "weight_proposals": expected_draws + 10,
        "weight_acceptances": expected_draws + 1,
        "weight_emitted_draws": expected_draws,
        "weight_buffered_draws": 1,
        "weight_acceptance_rate": (expected_draws + 1) / (expected_draws + 10),
        "weight_rejection_iterations": 1,
    }
    accounting = {
        "lr_horizon": updates,
        "positive_lr_optimizer_steps": updates,
        "scheduler_steps": updates,
        "attempted_batches": updates,
        "finite_loss_batches": updates,
        "optimizer_steps_attempted": updates,
        "optimizer_steps_succeeded": updates,
        "amp_overflow_skips": 0,
        "nonfinite_loss_skips": 0,
        "nonfinite_gradient_skips": 0,
        "stop_reason": "lr_horizon",
        "budget_satisfied": True,
        "n_pos_edges": graph["directed_edge_count"],
        **{f"pipeline_{key}": value for key, value in runtime.items()},
    }
    train = {
        "schema": TRAIN_RECEIPT_SCHEMA,
        "round_id": ROUND_ID,
        "production_config_sha256": digest,
        "train_accounting": accounting,
        "exact_execution_receipt": runtime,
        "update_derivation": {
            "directed_fuzzy_edges": graph["directed_edge_count"],
            "positive_rows_per_update": POSITIVE_ROWS_PER_UPDATE,
            "successful_updates": updates,
            "expected_positive_draws": expected_draws,
        },
        "optimizer_updates": updates,
        "train_checks": {
            "exact_update_closure": True,
            "zero_numerical_skips": True,
            "no_pipeline_stamp_drift": True,
            "endpoint_rows_match_updates": True,
            "weighted_rejection_accounting_closes": True,
        },
        "performance_profile": {"aborted": False},
        "steady_updates_per_s": 100.0,
        "training_performed": True,
        "evaluation_performed": False,
        "map_decision_made": False,
    }
    config_receipt = {
        "schema": PRODUCTION_CONFIG_SCHEMA,
        "round_id": ROUND_ID,
        "config": config,
        "config_sha256": digest,
    }
    return train, config_receipt, graph


def test_exact_training_variant_and_accounting_close():
    train, config, graph = _valid_train_bundle()
    observed = validate_train_execution(
        train=train, config_receipt=config, graph=graph
    )
    assert observed["successful_updates"] == 10
    assert all(observed["checks"].values())

    changed = copy.deepcopy(train)
    changed["exact_execution_receipt"]["compact_retained_rows"] -= 1
    with pytest.raises(Exception, match="authentication failed"):
        validate_train_execution(train=changed, config_receipt=config, graph=graph)


def test_shared_trainer_accepts_only_the_exact_r0152_variant():
    graph, signature = _fake_graph()
    config, _ = train_config(
        graph_manifest=graph,
        graph_signature=signature,
        schema=TRAIN_CONFIG_SCHEMA,
        compact_retained_rows=RETAINED_ROWS,
        pipeline=PIPELINE,
        pipeline_schema=PIPELINE_SCHEMA,
        positive_destination_policy=POSITIVE_DESTINATION_POLICY,
        update_rule=UPDATE_RULE,
    )
    assert config["input"]["rows"] == RETAINED_ROWS
    assert config["execution"]["required_pipeline"] == PIPELINE
    with pytest.raises(Round0107Error, match="not registered"):
        train_config(
            graph_manifest=graph,
            graph_signature=signature,
            schema=TRAIN_CONFIG_SCHEMA,
            compact_retained_rows=RETAINED_ROWS + 1,
            pipeline=PIPELINE,
            pipeline_schema=PIPELINE_SCHEMA,
            positive_destination_policy=POSITIVE_DESTINATION_POLICY,
            update_rule=UPDATE_RULE,
        )


def test_inherited_r0132_configuration_is_bounded_and_restored():
    original_round = round0132_nodes.ROUND_ID
    original_rows = round0132_nodes.HALF_RETAINED_ROWS
    with _configured_inherited_contract():
        assert round0132_nodes.ROUND_ID == ROUND_ID
        assert round0132_nodes.HALF_RETAINED_ROWS == RETAINED_ROWS
        assert round0132_nodes.GRAPH_CONTRACT.round_id == ROUND_ID
    assert round0132_nodes.ROUND_ID == original_round
    assert round0132_nodes.HALF_RETAINED_ROWS == original_rows
