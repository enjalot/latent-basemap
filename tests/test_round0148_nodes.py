from __future__ import annotations

import copy
import json
from pathlib import Path

from basemap.artifact_identity import canonical_json, sha256_bytes
from basemap.round0107_training import (
    BATCH_SIZE,
    POSITIVE_ROWS_PER_UPDATE,
    train_config,
    Round0107Error,
)
from basemap.round0132_scale_bridge import (
    GRAPH_K,
    HALF_RETAINED_ROWS,
    PIPELINE,
    PIPELINE_SCHEMA,
    SAMPLER_CLASS,
)
from experiments.round0148_nodes import (
    GRAPH_SCHEMA,
    POSITIVE_DESTINATION_POLICY,
    PRODUCTION_CONFIG_SCHEMA,
    ROUND_ID,
    TRAIN_CONFIG_SCHEMA,
    TRAIN_RECEIPT_SCHEMA,
    Round0148NodeError,
    validate_train_execution,
)
import pytest
from experiments import prepare_round0148_queue as prepare


def _valid_bundle() -> tuple[dict, dict, dict]:
    edges = 818
    updates = 2
    graph = {
        "schema": GRAPH_SCHEMA,
        "round_id": ROUND_ID,
        "retained_rows": HALF_RETAINED_ROWS,
        "k_real": GRAPH_K,
        "n_neighbors_including_self": GRAPH_K + 1,
        "directed_edge_count": edges,
        "compact_mapping": {"sha256": "a" * 64},
        "outputs": {
            "sources": {"sha256": "b" * 64},
            "targets": {"sha256": "c" * 64},
            "weights": {"sha256": "d" * 64},
        },
    }
    signature = {
        "kind": "file",
        "canonical_path": "/data/r0148-graph.json",
        "bytes": 1,
        "sha256": "e" * 64,
    }
    config, digest = train_config(
        graph_manifest=graph,
        graph_signature=signature,
        schema=TRAIN_CONFIG_SCHEMA,
        compact_retained_rows=HALF_RETAINED_ROWS,
        pipeline=PIPELINE,
        pipeline_schema=PIPELINE_SCHEMA,
        sampler_class=SAMPLER_CLASS,
        update_rule="ceil(actual-R0132-directed-fuzzy-edges/409)",
        positive_destination_policy=POSITIVE_DESTINATION_POLICY,
    )
    runtime = {
        **config["execution"]["expected_pipeline_stamp"],
        "endpoint_gather_calls": updates,
        "source_rows_gathered": updates * BATCH_SIZE,
        "destination_rows_gathered": updates * BATCH_SIZE,
        "host_prefetch_batches_filled": updates,
        "host_prefetch_producer_batches": updates,
        "host_prefetch_consumer_batches": updates,
        "host_prefetch_source_rows_filled": updates * BATCH_SIZE,
        "host_prefetch_destination_rows_filled": updates * BATCH_SIZE,
        "weight_proposals": updates * POSITIVE_ROWS_PER_UPDATE + 7,
        "weight_acceptances": updates * POSITIVE_ROWS_PER_UPDATE,
        "weight_emitted_draws": updates * POSITIVE_ROWS_PER_UPDATE,
        "weight_buffered_draws": 0,
        "weight_acceptance_rate": 0.99,
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
        "n_pos_edges": edges,
        **{
            f"pipeline_{key}": value
            for key, value in runtime.items()
            if key not in config["execution"]["expected_pipeline_stamp"]
        },
    }
    train = {
        "schema": TRAIN_RECEIPT_SCHEMA,
        "round_id": ROUND_ID,
        "production_config_sha256": digest,
        "optimizer_updates": updates,
        "update_derivation": {
            "directed_fuzzy_edges": edges,
            "positive_rows_per_update": POSITIVE_ROWS_PER_UPDATE,
            "successful_updates": updates,
            "expected_positive_draws": updates * POSITIVE_ROWS_PER_UPDATE,
        },
        "train_accounting": accounting,
        "exact_execution_receipt": runtime,
        "performance_profile": {"aborted": False},
        "steady_updates_per_s": 150.0,
        "train_checks": {
            "exact_update_closure": True,
            "zero_numerical_skips": True,
            "no_pipeline_stamp_drift": True,
            "endpoint_rows_match_updates": True,
            "weighted_rejection_accounting_closes": True,
        },
        "training_performed": True,
        "evaluation_performed": False,
        "map_decision_made": False,
    }
    receipt = {
        "schema": PRODUCTION_CONFIG_SCHEMA,
        "round_id": ROUND_ID,
        "config": config,
        "config_sha256": sha256_bytes(canonical_json(config)),
    }
    return train, receipt, graph


def test_r0148_train_reduces_to_reviewed_r0132_law() -> None:
    train, config, graph = _valid_bundle()
    authenticated = validate_train_execution(
        train=train, config_receipt=config, graph=graph
    )
    assert authenticated["successful_updates"] == 2
    assert all(authenticated["checks"].values())
    assert authenticated["normalized_to_reviewed_r0132_train_law"] is True
    assert authenticated["r0148_positive_destination_policy"] == (
        POSITIVE_DESTINATION_POLICY
    )


@pytest.mark.parametrize("target", ["config", "train", "runtime"])
def test_r0148_train_rejects_original_identity_or_runtime_drift(target: str) -> None:
    train, config, graph = _valid_bundle()
    if target == "config":
        config["config"]["optimizer"]["seed"] = 43
    elif target == "train":
        train["production_config_sha256"] = "0" * 64
    else:
        train["exact_execution_receipt"]["positive_destination_policy"] = (
            "wrong-population"
        )
    with pytest.raises(Round0148NodeError):
        validate_train_execution(train=train, config_receipt=config, graph=graph)


def test_r0148_validation_does_not_mutate_original_receipts() -> None:
    train, config, graph = _valid_bundle()
    originals = copy.deepcopy((train, config, graph))
    validate_train_execution(train=train, config_receipt=config, graph=graph)
    assert (train, config, graph) == originals


def test_shared_trainer_rejects_every_unregistered_population_stamp() -> None:
    _train, config_receipt, graph = _valid_bundle()
    with pytest.raises(Round0107Error, match="not registered"):
        train_config(
            graph_manifest=graph,
            graph_signature={
                "kind": "file",
                "canonical_path": "/data/r0148-graph.json",
                "bytes": 1,
                "sha256": "e" * 64,
            },
            schema=TRAIN_CONFIG_SCHEMA,
            compact_retained_rows=HALF_RETAINED_ROWS,
            pipeline=PIPELINE,
            pipeline_schema=PIPELINE_SCHEMA,
            sampler_class=SAMPLER_CLASS,
            update_rule="ceil(actual-R0132-directed-fuzzy-edges/409)",
            positive_destination_policy="unregistered-population",
        )
    assert config_receipt["config"]["execution"][
        "expected_pipeline_stamp"
    ]["positive_destination_policy"] == POSITIVE_DESTINATION_POLICY


def test_r0148_budget_is_calibrated_from_r0132() -> None:
    assert (
        prepare.GPU_HOURS_MINIMUM,
        prepare.GPU_HOURS_EXPECTED,
        prepare.GPU_HOURS_P90,
        prepare.GPU_HOURS_MAXIMUM,
    ) == (2.0, 2.5, 3.2, 4.5)


def test_positive_r0147_review_is_required(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "review-0147-test.md"
    path.write_text(
        "---\nround_id: \"0147\"\nstatus: accepted\n---\n"
        "capability:jina-2m-historical-row-policy-duplicate-control-v1\n"
        "eligible-historical-row-policy-restores\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(prepare, "R0147_REVIEW_GLOB", str(tmp_path / "*.md"))
    assert prepare._require_positive_r0147_review()["canonical_path"] == str(path)
    path.write_text(path.read_text().replace("restores", "does-not-restore"))
    with pytest.raises(RuntimeError, match="accepted positive"):
        prepare._require_positive_r0147_review()


def test_dependency_terminal_must_bind_unchanged_queue(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(json.dumps({"round_id": "0147"}), encoding="utf-8")
    queue_sha = prepare.expected_input_signature(str(queue_path))["sha256"]
    terminal_path = tmp_path / "terminal.json"
    terminal = {
        "round_id": "0147",
        "verdict": "succeeded",
        "completed_jobs": ["one"],
        "required_jobs": ["one"],
        "queue_manifest_sha256": queue_sha,
        "queue_manifest_sha256_at_finish": queue_sha,
        "queue_manifest_unchanged": True,
        "release_checkout_unchanged": True,
    }
    terminal_path.write_text(json.dumps(terminal), encoding="utf-8")
    prepare._require_clean_execution(
        str(queue_path), str(terminal_path), round_id="0147"
    )
    terminal["queue_manifest_unchanged"] = False
    terminal_path.write_text(json.dumps(terminal), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not clean"):
        prepare._require_clean_execution(
            str(queue_path), str(terminal_path), round_id="0147"
        )


def test_queue_materializes_complete_conditional_job_graph(
    monkeypatch, tmp_path: Path
) -> None:
    release = "a" * 40
    round_file = tmp_path / "round-0148-test.md"
    round_file.write_text(
        f"---\nround_id: \"0148\"\nstatus: issued\nbase_commit: \"{release}\"\n---\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(prepare, "_require_issued_round", lambda: str(round_file))
    round_signature = prepare.expected_input_signature(str(round_file))
    monkeypatch.setattr(
        prepare, "_require_positive_r0147_review", lambda: round_signature
    )
    monkeypatch.setattr(prepare, "_require_r0132_review", lambda: round_signature)

    def clean(queue_path: str, _terminal_path: str, *, round_id: str):
        with open(queue_path, encoding="utf-8") as handle:
            queue = json.load(handle)
        assert queue["round_id"] == round_id
        signature = prepare.expected_input_signature(queue_path)
        return queue, signature, signature

    monkeypatch.setattr(prepare, "_require_clean_execution", clean)

    def fresh(path: str, **_kwargs) -> str:
        Path(path).mkdir(parents=True, exist_ok=False)
        return path

    def ensure(path: str, **_kwargs) -> str:
        Path(path).mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(prepare, "create_fresh_directory", fresh)
    monkeypatch.setattr(prepare, "ensure_data_directory", ensure)
    monkeypatch.setattr(
        prepare,
        "atomic_write_new_json",
        lambda path, value, **_kwargs: Path(path).write_text(
            json.dumps(value), encoding="utf-8"
        ),
    )
    queue_path = prepare.prepare_round0148(
        release_sha=release, queue_root=str(tmp_path / "queue")
    )
    with open(queue_path, encoding="utf-8") as handle:
        queue = json.load(handle)
    actions = [job["action"] for job in queue["jobs"]]
    assert actions == [
        "select_english_anchor_subset",
        "build_english_anchor_search_index",
        "qualify_english_anchor_search",
        "build_english_anchor_graph_part",
        "build_english_anchor_graph_part",
        "build_english_anchor_graph_part",
        "assemble_english_anchor_graph",
        "train_english_anchor_map",
        "transform_english_anchor_map",
        "score_english_anchor_function_density",
        "score_english_anchor_ood",
        "decide_english_anchor_rescue",
    ]
    by_action = {job["action"]: job for job in queue["jobs"]}
    assert by_action["qualify_english_anchor_search"]["index"].endswith(
        "/english-anchor-12p5m.ivfpq"
    )
    decision = by_action["decide_english_anchor_rescue"]
    assert decision["control_graph_manifest"] == prepare.CONTROL_GRAPH
    assert "r0108_calibration" in decision
    assert queue["release_sha"] == release
    assert queue["scientific_contract"]["density_floor_recalibration"] is False
