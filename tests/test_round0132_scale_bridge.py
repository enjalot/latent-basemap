"""Contracts, adversarial selectors, and bounded CPU smoke for R0132."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from basemap.round0105_search import GROUPS, RETAINED_ROWS
from basemap.round0036_pipeline import (
    COORDINATE_SCHEMA,
    TRANSFORM_SCHEMA,
    seal as coordinate_seal,
)
from basemap.panel_v2 import PanelV2Config, score_panel
from basemap.round0107_training import (
    BATCH_SIZE,
    DiverseWeightedJinaSampler,
    POSITIVE_ROWS_PER_UPDATE,
    train_config,
)
from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from basemap.round0132_scale_bridge import (
    DENSITY_BOOTSTRAP_DRAWS,
    DENSITY_BOOTSTRAP_SEED,
    DENSITY_COMPARISON_ATOL,
    DENSITY_NONINFERIORITY_MARGIN,
    FULL_RETAINED_ROWS,
    GRAPH_K,
    HALF_RETAINED_ROWS,
    OUTCOME_DENSITY_REGRESSION,
    OUTCOME_INCONCLUSIVE,
    OUTCOME_INVALID,
    OUTCOME_QUALITY_REGRESSION,
    OUTCOME_SUPPORTED,
    PIPELINE,
    PIPELINE_SCHEMA,
    SAMPLER_CLASS,
    SEARCH_ANCHORS_PER_GROUP,
    SUBSET_NAMESPACE,
    Round0132Error,
    assert_no_conditional_branch_dependency,
    coverage_aligned_updates,
    density_ci_classification,
    group_part_specs,
    largest_remainder_quotas,
    noninferiority_checks,
    paired_density_bootstrap,
    qualification_metrics,
    recall50_at_least_recall10,
    scale_policy_decision,
    seal,
    select_lowest_sha256_rank,
    validate_seal,
    validate_train_execution,
)
from experiments import round0106_nodes, round0107_nodes, round0132_nodes
from experiments.prepare_round0132_queue import (
    GPU_HOURS_EXPECTED,
    GPU_HOURS_MAXIMUM,
    GPU_HOURS_MINIMUM,
    GPU_HOURS_P90,
    P90_GPU_TOTAL_SECONDS,
    REVIEW_DEFAULTS,
    _accepted_control_signatures,
    _accepted_transform_signatures,
    _require_clean_r0108,
    _require_issued_round,
    _require_round_release,
)
from experiments.round0132_nodes import (
    _authenticate_native_selector,
    _native_ffr_truth_hits,
)


def _balanced_counts() -> dict[str, int]:
    quotient, remainder = divmod(FULL_RETAINED_ROWS, len(GROUPS))
    return {
        group: quotient + (1 if index < remainder else 0)
        for index, group in enumerate(GROUPS)
    }


def test_largest_remainder_closes_exact_half_and_preserves_every_group():
    counts = _balanced_counts()
    quotas = largest_remainder_quotas(counts)
    assert sum(counts.values()) == FULL_RETAINED_ROWS
    assert sum(quotas.values()) == HALF_RETAINED_ROWS
    assert list(quotas) == list(GROUPS)
    assert all(0 < quotas[group] <= counts[group] for group in GROUPS)
    assert max(quotas.values()) - min(quotas.values()) <= 1


def test_largest_remainder_rejects_missing_skew_or_wrong_total():
    counts = _balanced_counts()
    missing = dict(counts)
    missing.pop(GROUPS[-1])
    with pytest.raises(Round0132Error, match="keys"):
        largest_remainder_quotas(missing)
    counts[GROUPS[0]] -= 1
    with pytest.raises(Round0132Error, match="total"):
        largest_remainder_quotas(counts)


def test_sha_rank_is_exact_deterministic_and_not_a_prefix():
    rows = np.arange(100, 300, dtype=np.int64)
    observed = select_lowest_sha256_rank(rows, count=83)
    expected = np.asarray(sorted(
        rows.tolist(),
        key=lambda row: (
            hashlib.sha256(
                SUBSET_NAMESPACE + int(row).to_bytes(8, "little")
            ).digest(),
            row,
        ),
    )[:83], dtype=np.int64)
    expected.sort()
    np.testing.assert_array_equal(observed, expected)
    np.testing.assert_array_equal(
        observed, select_lowest_sha256_rank(rows, count=83)
    )
    assert not np.array_equal(observed, rows[:83])


def test_sha_rank_rejects_duplicates_unsorted_and_invalid_count():
    with pytest.raises(Round0132Error, match="malformed"):
        select_lowest_sha256_rank(np.asarray([2, 1]), count=1)
    with pytest.raises(Round0132Error, match="malformed"):
        select_lowest_sha256_rank(np.asarray([1, 1]), count=1)
    with pytest.raises(Round0132Error, match="malformed"):
        select_lowest_sha256_rank(np.arange(4), count=5)


def test_group_aligned_parts_are_contiguous_complete_and_nonempty():
    quotas = largest_remainder_quotas(_balanced_counts())
    parts = group_part_specs(quotas)
    assert list(parts) == ["groups-a", "groups-b", "groups-c"]
    cursor = 0
    all_groups: list[str] = []
    for part in parts.values():
        assert part["compact_start"] == cursor
        assert part["compact_stop"] > cursor
        assert part["retained_rows"] == part["compact_stop"] - cursor
        cursor = part["compact_stop"]
        all_groups.extend(part["groups"])
    assert cursor == HALF_RETAINED_ROWS
    assert all_groups == list(GROUPS)


def test_fixed_search_selector_passes_only_complete_dual_floor_cell():
    rows = SEARCH_ANCHORS_PER_GROUP * len(GROUPS)
    exact = np.arange(rows * GRAPH_K, dtype=np.int64).reshape(rows, GRAPH_K)
    selected = exact.copy()
    group_ids = np.repeat(
        np.arange(len(GROUPS), dtype=np.uint8), SEARCH_ANCHORS_PER_GROUP
    )
    metrics = qualification_metrics(
        selected,
        exact,
        group_ids=group_ids,
        unambiguous=np.ones(rows, dtype=bool),
    )
    assert metrics["passed"] is True
    assert metrics["global_mean_recall_at_15"] == 1.0
    assert metrics["checks"]["no_policy_sweep_or_widening_performed"] is True

    degraded = selected.copy()
    first_group = group_ids == 0
    degraded[first_group] += rows * GRAPH_K
    failed = qualification_metrics(
        degraded,
        exact,
        group_ids=group_ids,
        unambiguous=np.ones(rows, dtype=bool),
    )
    assert failed["passed"] is False
    assert not failed["by_group"][GROUPS[0]]["passes_floor"]


def test_coverage_horizon_is_computed_from_actual_edges():
    assert coverage_aligned_updates(409) == 1
    assert coverage_aligned_updates(410) == 2
    assert coverage_aligned_updates(298_500_000) == 729_829
    with pytest.raises(Round0132Error):
        coverage_aligned_updates(0)


def test_density_bootstrap_uses_paired_draws_and_registered_selector():
    rng = np.random.RandomState(132)
    high = np.exp(np.linspace(-2.0, 2.0, 256))
    control = high * np.exp(rng.normal(0.0, 0.03, len(high)))
    treatment = control.copy()
    result = paired_density_bootstrap(high, control, treatment)
    assert result["classification"] == "noninferior"
    assert result["treatment_minus_control"] == pytest.approx(0.0)
    assert result["paired_bootstrap_draws"] == DENSITY_BOOTSTRAP_DRAWS
    assert result["paired_bootstrap_seed"] == DENSITY_BOOTSTRAP_SEED
    assert len(result["bootstrap_deltas"]) == DENSITY_BOOTSTRAP_DRAWS
    assert result["comparison_atol"] == DENSITY_COMPARISON_ATOL


def test_density_margin_comparisons_are_inclusive_with_registered_tolerance():
    boundary = -DENSITY_NONINFERIORITY_MARGIN
    assert density_ci_classification(
        boundary - DENSITY_COMPARISON_ATOL, boundary + 0.01
    ) == "noninferior"
    assert density_ci_classification(
        boundary - 0.01, boundary + DENSITY_COMPARISON_ATOL
    ) == "materially-worse"
    assert density_ci_classification(
        boundary - 0.01, boundary + 0.01
    ) == "inconclusive"


def _quality(*, passed: bool = True) -> dict:
    checks = {"native": passed, "ood": passed}
    return {"passed": passed, "checks": checks, "metrics": {}}


@pytest.mark.parametrize(
    ("validity", "density", "quality", "expected"),
    [
        ({"all": True}, {"classification": "noninferior"}, _quality(), OUTCOME_SUPPORTED),
        (
            {"all": True},
            {"classification": "materially-worse"},
            _quality(),
            OUTCOME_DENSITY_REGRESSION,
        ),
        (
            {"all": True},
            {"classification": "noninferior"},
            _quality(passed=False),
            OUTCOME_QUALITY_REGRESSION,
        ),
        (
            {"all": True},
            {"classification": "inconclusive"},
            _quality(),
            OUTCOME_INCONCLUSIVE,
        ),
        (
            {"all": False},
            {"classification": "noninferior"},
            _quality(),
            OUTCOME_INVALID,
        ),
    ],
)
def test_decision_branch_order_is_frozen(validity, density, quality, expected):
    result = scale_policy_decision(
        validity_checks=validity, density=density, quality=quality
    )
    assert result["outcome"] == expected
    assert result["stale_absolute_jina_floor_role"] == "diagnostic-only"
    assert result["native_global_ffr_role"] == "registered-noninferiority-gate"
    assert result["ood_projection_ffr_role"] == "diagnostic-only"
    assert result["trec_covid_role"] == "diagnostic-only"
    assert result["dadabase_role"] == "diagnostic-only"
    assert "seed-42" in result["one_seed_limitation"]
    assert result["capabilities_produced"] == (
        []
        if expected == OUTCOME_INVALID
        else ["jina-diverse-12p5m-25m-scale-policy-geometry-v1"]
    )
    assert "not a pure-N effect" in result["estimand"]


def test_noninferiority_uses_exact_native_and_ood_margins():
    result = noninferiority_checks(
        control_native={
            "global_ffr": 0.50,
            "global_recall_at_10": 0.20,
            "global_recall_at_50_of_high10": 0.40,
        },
        treatment_native={
            "global_ffr": 0.48,
            "global_recall_at_10": 0.194,
            "global_recall_at_50_of_high10": 0.388,
        },
        control_ood={
            "fineweb_recall_at_50_of_high10": 0.50,
            "polish_recall_at_50_of_high10": 0.20,
            "in_mix_median_recall_at_50_of_high10": 0.25,
        },
        treatment_ood={
            "fineweb_recall_at_50_of_high10": 0.485,
            "polish_recall_at_50_of_high10": 0.194,
            "in_mix_median_recall_at_50_of_high10": 0.2425,
        },
    )
    assert result["passed"] is True
    treatment = {
        "fineweb_recall_at_50_of_high10": 0.4849,
        "polish_recall_at_50_of_high10": 0.194,
        "in_mix_median_recall_at_50_of_high10": 0.2425,
    }
    failed = noninferiority_checks(
        control_native={
            "global_ffr": 0.50,
            "global_recall_at_10": 0.20,
            "global_recall_at_50_of_high10": 0.40,
        },
        treatment_native={
            "global_ffr": 0.48,
            "global_recall_at_10": 0.194,
            "global_recall_at_50_of_high10": 0.388,
        },
        control_ood={
            "fineweb_recall_at_50_of_high10": 0.50,
            "polish_recall_at_50_of_high10": 0.20,
            "in_mix_median_recall_at_50_of_high10": 0.25,
        },
        treatment_ood=treatment,
    )
    assert failed["passed"] is False


def test_ood_recall_order_is_inclusive_and_has_no_absolute_polish_floor():
    cells = [
        {"recall_at_10": 0.90, "recall_at_50_of_high10": 0.95},
        # Equality is a valid wider-neighborhood invariant, including for a
        # very low held-out Polish cell relative to an in-mix population.
        {"recall_at_10": 0.001, "recall_at_50_of_high10": 0.001},
    ]
    assert recall50_at_least_recall10(cells) is True
    assert recall50_at_least_recall10([
        {"recall_at_10": 0.002, "recall_at_50_of_high10": 0.001}
    ]) is False


def test_r0132_has_no_conditional_branch_dependency():
    assert set(REVIEW_DEFAULTS) == {
        "0087", "0103", "0105", "0106", "0107", "0108", "0118", "0119"
    }
    assert_no_conditional_branch_dependency(tuple(REVIEW_DEFAULTS))
    for forbidden in ("0125", "0129", "0130", "0131"):
        with pytest.raises(Round0132Error, match="must not depend"):
            assert_no_conditional_branch_dependency([forbidden])


def test_registered_gpu_budget_is_exact():
    assert (GPU_HOURS_MINIMUM, GPU_HOURS_EXPECTED, GPU_HOURS_P90, GPU_HOURS_MAXIMUM) == (
        1.8, 2.3, 3.1, 4.5
    )
    assert P90_GPU_TOTAL_SECONDS == 3.1 * 3_600


def test_draft_or_multiple_round_files_cannot_materialize(monkeypatch, tmp_path):
    import experiments.prepare_round0132_queue as prepare

    draft = tmp_path / "round-0132-2026-07-31.md"
    draft.write_text("---\nstatus: draft\n---\n", encoding="utf-8")
    monkeypatch.setattr(prepare, "ROUND_FILE_GLOB", str(tmp_path / "round-0132-*.md"))
    with pytest.raises(RuntimeError, match="exactly one issued"):
        _require_issued_round()
    draft.write_text("---\nstatus: issued\n---\n", encoding="utf-8")
    second = tmp_path / "round-0132-2026-08-01.md"
    second.write_text("---\nstatus: issued\n---\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="found 2"):
        _require_issued_round()


def test_issued_round_release_must_match_materialized_commit(tmp_path):
    path = tmp_path / "round.md"
    path.write_text(
        f"---\nbase_commit: {'a' * 40}\nstatus: issued\n---\n",
        encoding="utf-8",
    )
    _require_round_release(str(path), "a" * 40)
    with pytest.raises(RuntimeError, match="base_commit"):
        _require_round_release(str(path), "b" * 40)


@pytest.mark.parametrize("field", [
    "queue_manifest_sha256",
    "queue_manifest_sha256_at_finish",
])
def test_r0108_terminal_binds_queue_bytes_at_start_and_finish(
    monkeypatch, tmp_path, field
):
    import experiments.prepare_round0132_queue as prepare

    queue = tmp_path / "queue.json"
    queue.write_text(json.dumps({"round_id": "0108"}), encoding="utf-8")
    queue_sha = expected_input_signature(str(queue))["sha256"]
    terminal = tmp_path / "runner-terminal.json"
    body = {
        "round_id": "0108",
        "verdict": "succeeded",
        "completed_jobs": 2,
        "required_jobs": 2,
        "queue_manifest_sha256": queue_sha,
        "queue_manifest_sha256_at_finish": queue_sha,
        "queue_manifest_unchanged": True,
        "release_checkout_unchanged": True,
    }
    terminal.write_text(json.dumps(body), encoding="utf-8")
    monkeypatch.setattr(prepare, "R0108_QUEUE", str(queue))
    monkeypatch.setattr(prepare, "R0108_TERMINAL", str(terminal))
    _require_clean_r0108()
    body[field] = "f" * 64
    terminal.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean accepted"):
        _require_clean_r0108()


def _accepted_control_fixture(tmp_path: Path) -> dict[str, Path]:
    direct = {
        name: tmp_path / name
        for name in (
            "graph.json",
            "mapping.npy",
            "selection.npz",
            "calibration.json",
        )
    }
    for path in direct.values():
        path.write_bytes(path.name.encode("utf-8"))
    train = tmp_path / "train"
    train.mkdir()
    for name in ("train-receipt.json", "production-config.json", "model.pt"):
        (train / name).write_bytes(name.encode("utf-8"))

    transform = tmp_path / "coordinates"
    transform.mkdir()
    members = []
    for index in range(5):
        chunk = transform / f"chunk-{index:05d}"
        chunk.mkdir()
        path = chunk / "coordinates.npy"
        np.save(path, np.asarray([[index, index + 1]], dtype=np.float32))
        signature = expected_input_signature(str(path))
        members.append({
            "chunk_index": index,
            "global_row_start": index,
            "global_row_stop": index + 1,
            "bytes": signature["bytes"],
            "sha256": signature["sha256"],
        })
    receipt = coordinate_seal({
        "schema": TRANSFORM_SCHEMA,
        "row_accounting": {"all_rows": 5},
        "coordinate_stream": {
            "schema": COORDINATE_SCHEMA,
            "row_count": 5,
            "dimension": 2,
            "dtype": "<f4",
            "ordered_chunks": members,
        },
    })
    (transform / "actual-transform.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    return {**direct, "train": train, "transform": transform}


def test_accepted_controls_bind_model_and_all_five_coordinate_members(tmp_path):
    fixture = _accepted_control_fixture(tmp_path)
    signatures = _accepted_control_signatures(
        full_graph_manifest=str(fixture["graph.json"]),
        full_mapping=str(fixture["mapping.npy"]),
        full_train_output=str(fixture["train"]),
        selection_path=str(fixture["selection.npz"]),
        calibration_path=str(fixture["calibration.json"]),
        transform_root=str(fixture["transform"]),
        expected_transform_rows=5,
        expected_transform_members=5,
    )
    paths = {Path(item["canonical_path"]) for item in signatures}
    assert fixture["train"] / "model.pt" in paths
    assert {
        fixture["transform"] / f"chunk-{index:05d}" / "coordinates.npy"
        for index in range(5)
    } <= paths
    assert fixture["transform"] / "actual-transform.json" in paths


@pytest.mark.parametrize("failure", ["missing", "wrong-bytes"])
def test_accepted_transform_members_fail_closed(tmp_path, failure):
    fixture = _accepted_control_fixture(tmp_path)
    member = fixture["transform"] / "chunk-00003" / "coordinates.npy"
    if failure == "missing":
        member.unlink()
    else:
        np.save(member, np.asarray([[99, 100]], dtype=np.float32))
    with pytest.raises(RuntimeError, match="failed authentication"):
        _accepted_transform_signatures(
            str(fixture["transform"]),
            expected_rows=5,
            expected_members=5,
        )


def test_accepted_controls_fail_closed_when_model_is_missing(tmp_path):
    fixture = _accepted_control_fixture(tmp_path)
    (fixture["train"] / "model.pt").unlink()
    with pytest.raises(RuntimeError, match="missing or invalid"):
        _accepted_control_signatures(
            full_graph_manifest=str(fixture["graph.json"]),
            full_mapping=str(fixture["mapping.npy"]),
            full_train_output=str(fixture["train"]),
            selection_path=str(fixture["selection.npz"]),
            calibration_path=str(fixture["calibration.json"]),
            transform_root=str(fixture["transform"]),
            expected_transform_rows=5,
            expected_transform_members=5,
        )


class _StampDataset:
    shape = (RETAINED_ROWS, 768)
    device = "cpu"

    def __len__(self):
        return RETAINED_ROWS

    def execution_stamp(self):
        return {"source_representation": "int8-treatment"}


def test_r0106_r0107_r0108_legacy_defaults_are_unchanged():
    graph = {
        "directed_edge_count": 1_000,
        "compact_mapping": {"sha256": "a" * 64},
        "outputs": {
            "sources": {"sha256": "b" * 64},
            "targets": {"sha256": "c" * 64},
            "weights": {"sha256": "d" * 64},
        },
    }
    signature = {
        "kind": "file",
        "canonical_path": "/data/synthetic-graph.json",
        "bytes": 1,
        "sha256": "e" * 64,
    }
    config, digest = train_config(graph_manifest=graph, graph_signature=signature)
    assert digest == "5619d2c0b7f4da6486f507b6ee03d305c9a5fce8e5df44e0f46e84165d2d3437"
    assert config["execution"]["expected_pipeline_stamp"]["negative_sampling"] == (
        "uniform-24,948,663-compact-retained-rows-nonself"
    )
    assert inspect.signature(round0106_nodes._write_shard).parameters[
        "universe_rows"
    ].default == RETAINED_ROWS
    assert inspect.signature(round0106_nodes._partition_forward_edges).parameters[
        "universe_rows"
    ].default == RETAINED_ROWS

    sampler = DiverseWeightedJinaSampler(
        _StampDataset(),
        sources=np.asarray([0, 1], dtype=np.int32),
        targets=np.asarray([1, 0], dtype=np.int32),
        weights=np.asarray([0.5, 0.5], dtype=np.float32),
        n_nodes=RETAINED_ROWS,
        batch_size=10,
        pos_ratio=0.5,
        random_state=42,
        graph_signatures={},
    )
    stamp = sampler.execution_stamp()
    assert stamp["pipeline"] == "host_weighted_jina_diverse_25m"
    assert stamp["schema"] == "round0107-host-weighted-jina-diverse-pipeline-v1"
    assert stamp["negative_sampling"] == (
        "uniform-24,948,663-compact-retained-rows-nonself"
    )


def test_dynamic_r0132_config_changes_only_registered_universe_pipeline_and_horizon():
    graph = {
        "directed_edge_count": 298_500_000,
        "compact_mapping": {"sha256": "a" * 64},
        "outputs": {
            "sources": {"sha256": "b" * 64},
            "targets": {"sha256": "c" * 64},
            "weights": {"sha256": "d" * 64},
        },
    }
    signature = {
        "kind": "file",
        "canonical_path": "/data/r0132-graph.json",
        "bytes": 1,
        "sha256": "e" * 64,
    }
    config, _digest = train_config(
        graph_manifest=graph,
        graph_signature=signature,
        schema="round0132-half-train-config-v1",
        compact_retained_rows=HALF_RETAINED_ROWS,
        pipeline=PIPELINE,
        pipeline_schema=PIPELINE_SCHEMA,
        sampler_class=SAMPLER_CLASS,
        update_rule="ceil(actual-R0132-directed-fuzzy-edges/409)",
        positive_destination_policy=(
            "R0132-global-half-retained-fuzzy-tconorm-graph"
        ),
    )
    assert config["input"]["rows"] == HALF_RETAINED_ROWS
    assert config["optimizer"]["successful_positive_lr_updates"] == 729_829
    assert config["execution"]["required_pipeline"] == PIPELINE
    assert config["execution"]["expected_pipeline_stamp"]["negative_sampling"] == (
        "uniform-12,474,331-compact-retained-rows-nonself"
    )


def _valid_train_execution_bundle():
    edges = 818
    updates = 2
    graph = {
        "schema": "round0132-half-fuzzy-graph-v1",
        "round_id": "0132",
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
        "canonical_path": "/data/r0132-graph.json",
        "bytes": 1,
        "sha256": "e" * 64,
    }
    config, digest = train_config(
        graph_manifest=graph,
        graph_signature=signature,
        schema="round0132-half-train-config-v1",
        compact_retained_rows=HALF_RETAINED_ROWS,
        pipeline=PIPELINE,
        pipeline_schema=PIPELINE_SCHEMA,
        sampler_class=SAMPLER_CLASS,
        update_rule="ceil(actual-R0132-directed-fuzzy-edges/409)",
        positive_destination_policy=(
            "R0132-global-half-retained-fuzzy-tconorm-graph"
        ),
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
        **{f"pipeline_{key}": value for key, value in runtime.items()
           if key not in config["execution"]["expected_pipeline_stamp"]},
    }
    train = {
        "schema": "round0132-half-train-receipt-v1",
        "round_id": "0132",
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
    return train, {
        "schema": "round0132-half-production-config-v1",
        "round_id": "0132",
        "config": config,
        "config_sha256": sha256_bytes(canonical_json(config)),
    }, graph


def test_train_execution_receipt_authenticates_actual_pipeline_and_accounting():
    train, config, graph = _valid_train_execution_bundle()
    authenticated = validate_train_execution(
        train=train, config_receipt=config, graph=graph
    )
    assert authenticated["successful_updates"] == 2
    assert all(authenticated["checks"].values())


@pytest.mark.parametrize(
    ("target", "key", "value"),
    [
        ("runtime", "pipeline", "uniform-fallback"),
        ("runtime", "weight_emitted_draws", 817),
        ("accounting", "optimizer_steps_succeeded", 1),
        ("train", "optimizer_updates", 1),
    ],
)
def test_train_execution_receipt_rejects_hand_minted_drift(target, key, value):
    train, config, graph = _valid_train_execution_bundle()
    if target == "runtime":
        train["exact_execution_receipt"][key] = value
    elif target == "accounting":
        train["train_accounting"][key] = value
    else:
        train[key] = value
    with pytest.raises(Round0132Error, match="authentication failed"):
        validate_train_execution(train=train, config_receipt=config, graph=graph)


def _native_selector_fixture(tmp_path: Path):
    rows = SEARCH_ANCHORS_PER_GROUP * len(GROUPS)
    base = np.arange(rows, dtype=np.int64)[:, None] * 64
    high = base + np.arange(GRAPH_K, dtype=np.int64)
    low = base + np.arange(50, dtype=np.int64)
    high_radius = np.exp(np.linspace(-2.0, 2.0, rows))
    control_radius = high_radius * (1.0 + 0.01 * np.sin(np.arange(rows)))
    treatment_radius = control_radius.copy()
    density = paired_density_bootstrap(
        high_radius, control_radius, treatment_radius
    )
    deltas = density.pop("bootstrap_deltas")
    path = tmp_path / "native-arrays.npz"
    np.savez(
        path,
        high_neighbors_top15=high,
        high_radius=high_radius,
        control_low_radius=control_radius,
        treatment_low_radius=treatment_radius,
        control_low_neighbors_top50=low,
        treatment_low_neighbors_top50=low,
        native_fraction_k=np.asarray(12_475, dtype=np.int64),
        control_ffr_truth_hits=np.ones((rows, 10), dtype=bool),
        treatment_ffr_truth_hits=np.ones((rows, 10), dtype=bool),
        family_sizes=np.ones(rows, dtype=np.int64),
        density_bootstrap_deltas=deltas,
    )
    return {
        "arrays": expected_input_signature(str(path)),
        "density_selector": density,
        "control_12p5m": {
            "global_ffr": 1.0,
            "global_recall_at_10": 1.0,
            "global_recall_at_50_of_high10": 1.0,
        },
        "treatment_25m_on_u12": {
            "global_ffr": 1.0,
            "global_recall_at_10": 1.0,
            "global_recall_at_50_of_high10": 1.0,
        },
    }


def test_terminal_density_selector_recomputes_from_bound_arrays(tmp_path: Path):
    native = _native_selector_fixture(tmp_path)
    authenticated = _authenticate_native_selector(native)
    assert authenticated["density_selector_recomputed"] is True
    assert authenticated["density_classification"] == "noninferior"
    assert authenticated[
        "native_global_ffr_recomputed_from_per_anchor_evidence"
    ] is True


def test_terminal_density_selector_rejects_minted_classification(tmp_path: Path):
    native = _native_selector_fixture(tmp_path)
    native["density_selector"]["classification"] = "materially-worse"
    with pytest.raises(Round0132Error, match="does not recompute"):
        _authenticate_native_selector(native)


def test_terminal_native_selector_rejects_minted_global_ffr_scalar(tmp_path: Path):
    native = _native_selector_fixture(tmp_path)
    native["treatment_25m_on_u12"]["global_ffr"] = 0.999
    with pytest.raises(Round0132Error, match="recall/FFR does not recompute"):
        _authenticate_native_selector(native)


def test_terminal_native_selector_rejects_noninteger_fraction_width(tmp_path: Path):
    native = _native_selector_fixture(tmp_path)
    path = Path(native["arrays"]["canonical_path"])
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    arrays["native_fraction_k"] = np.asarray(12_475.5, dtype=np.float64)
    np.savez(path, **arrays)
    native["arrays"] = expected_input_signature(str(path))
    with pytest.raises(Round0132Error, match="selector arrays changed"):
        _authenticate_native_selector(native)


def test_native_ffr_membership_evidence_exactly_recomputes_per_truth_hit():
    high10 = np.asarray([[2, 4, 6, 8, 10, 12, 14, 16, 18, 20]])
    low = np.arange(60, dtype=np.int64)[None, :]
    low[0, [2, 4, 6]] = np.asarray([2, 4, 999])
    hits = _native_ffr_truth_hits(high10, low, fraction_k=50)
    expected = np.isin(high10[0], low[0, :50])
    np.testing.assert_array_equal(hits[0], expected)
    assert float(hits.mean()) == pytest.approx(float(expected.mean()))


def test_cpu_train_seal_reload_transform_panel_smoke(monkeypatch, tmp_path: Path):
    """Two-minute-class CUDA-hidden check of the complete artifact path."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    import torch

    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(42)
    values = torch.randn(256, 8, generator=generator)
    target = values[:, :2] * 0.5
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.SiLU(), torch.nn.Linear(16, 2)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(values), target)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)

    model_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), model_path)
    receipt = seal({
        "schema": "round0132-cpu-smoke-train-v1",
        "train_accounting": {
            "attempted_batches": 1,
            "finite_loss_batches": 1,
            "optimizer_steps_succeeded": 1,
            "nonfinite_loss_skips": 0,
            "nonfinite_gradient_skips": 0,
        },
        "model_bytes": model_path.stat().st_size,
    })
    receipt_path = tmp_path / "train-receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    loaded_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    validate_seal(loaded_receipt, label="R0132 CPU smoke train")

    reloaded = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.SiLU(), torch.nn.Linear(16, 2)
    )
    reloaded.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    with torch.inference_mode():
        coordinates = reloaded(values).numpy().astype(np.float32)
    assert coordinates.shape == (256, 2)
    assert np.isfinite(coordinates).all()
    high_radius = np.linalg.norm(values.numpy()[:, :2], axis=1) + 1e-3
    low_radius = np.linalg.norm(coordinates, axis=1) + 1e-3
    panel = paired_density_bootstrap(high_radius, low_radius, low_radius)
    panel.pop("bootstrap_deltas")
    sealed_panel = seal({
        "schema": "round0132-cpu-smoke-panel-v1",
        "density": panel,
        "coordinates_finite": True,
        "path": "train->seal->reload->transform->panel",
    })
    validate_seal(sealed_panel, label="R0132 CPU smoke panel")
    assert panel["classification"] == "noninferior"


def test_actual_r0132_train_contract_seals_reloads_and_scores_on_cpu(
    monkeypatch, tmp_path: Path
):
    """Exercise the production post-fit handoff without allocating CUDA."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    import torch

    torch.set_num_threads(1)
    release = "a" * 40
    updates = 201
    edges = updates * POSITIVE_ROWS_PER_UPDATE
    graph_path = tmp_path / "graph-manifest.json"
    graph_manifest = seal({
        "schema": "round0132-half-fuzzy-graph-v1",
        "round_id": "0132",
        "release_sha": release,
        "retained_rows": HALF_RETAINED_ROWS,
        "dimension": 768,
        "k_real": GRAPH_K,
        "n_neighbors_including_self": GRAPH_K + 1,
        "directed_edge_count": edges,
        "compact_mapping": {"sha256": "1" * 64},
        "outputs": {
            "sources": {"sha256": "2" * 64},
            "targets": {"sha256": "3" * 64},
            "weights": {"sha256": "4" * 64},
        },
        "reciprocity_validation": {"every_reverse_present_once": True},
    })
    graph_path.write_text(json.dumps(graph_manifest), encoding="utf-8")
    graph_signature = expected_input_signature(str(graph_path))
    config, config_digest = train_config(
        graph_manifest=graph_manifest,
        graph_signature=graph_signature,
        schema="round0132-half-train-config-v1",
        compact_retained_rows=HALF_RETAINED_ROWS,
        pipeline=PIPELINE,
        pipeline_schema=PIPELINE_SCHEMA,
        sampler_class=SAMPLER_CLASS,
        update_rule="ceil(actual-R0132-directed-fuzzy-edges/409)",
        positive_destination_policy=(
            "R0132-global-half-retained-fuzzy-tconorm-graph"
        ),
    )
    expected_rows = updates * BATCH_SIZE
    emitted = updates * POSITIVE_ROWS_PER_UPDATE
    accepted = emitted + 3
    proposals = accepted + 101
    runtime = {
        **config["execution"]["expected_pipeline_stamp"],
        "endpoint_gather_calls": updates,
        "source_rows_gathered": expected_rows,
        "destination_rows_gathered": expected_rows,
        "host_prefetch_batches_filled": updates,
        "host_prefetch_producer_batches": updates,
        "host_prefetch_consumer_batches": updates,
        "host_prefetch_source_rows_filled": expected_rows,
        "host_prefetch_destination_rows_filled": expected_rows,
        "weight_proposals": proposals,
        "weight_acceptances": accepted,
        "weight_emitted_draws": emitted,
        "weight_buffered_draws": accepted - emitted,
        "weight_acceptance_rate": accepted / proposals,
        "weight_rejection_iterations": updates,
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
    fake_graph = {
        "manifest": graph_manifest,
        "signature": graph_signature,
        "successful_updates": updates,
        "arrays": {
            "mapping": np.arange(8, dtype=np.int64),
            "sources": np.arange(8, dtype=np.int32),
            "targets": np.roll(np.arange(8, dtype=np.int32), -1),
            "weights": np.ones(8, dtype=np.float32),
        },
    }

    class SmokeDataset:
        def __init__(self, *args, **kwargs):
            self.features = np.random.default_rng(132).normal(
                size=(240, 8)
            ).astype(np.float32)
            self.substrate = {
                "signature": {
                    "kind": "file",
                    "canonical_path": "/synthetic/r0132-substrate.json",
                    "bytes": 1,
                    "sha256": "5" * 64,
                }
            }

    class SmokeWrapper:
        def __init__(self, dataset, *args, **kwargs):
            self.dataset = dataset

        def runtime_stamp(self):
            return dict(runtime)

    class SmokeProfiler:
        def finalize(self, **kwargs):
            assert kwargs["bench_seconds"] > 0
            return {"aborted": False, "cpu_smoke": True}

    class SmokeModel:
        def __init__(self):
            self.layer = torch.nn.Linear(8, 2)
            self._canary_profiler = SmokeProfiler()
            self._bench_seconds = 0.01
            self._setup_seconds = 0.001

        def fit(self, wrapper, **kwargs):
            features = torch.from_numpy(wrapper.dataset.features)
            target = features[:, :2] - 0.25 * features[:, 2:4]
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.05)
            for _ in range(4):
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(self.layer(features), target)
                loss.backward()
                optimizer.step()
            self._train_stats = dict(accounting)

        def save(self, path):
            torch.save(
                {
                    "weight": self.layer.weight.detach(),
                    "bias": self.layer.bias.detach(),
                },
                path,
            )

    monkeypatch.setattr(
        round0107_nodes, "_graph", lambda active, job, **kwargs: fake_graph
    )
    monkeypatch.setattr(
        round0107_nodes,
        "train_config",
        lambda **kwargs: (config, config_digest),
    )
    monkeypatch.setattr(
        round0107_nodes, "CompactHostInt8MaterializedArray", SmokeDataset
    )
    monkeypatch.setattr(round0107_nodes, "Round0107TrainingInput", SmokeWrapper)
    monkeypatch.setattr(
        round0107_nodes, "_new_model", lambda config, **kwargs: SmokeModel()
    )
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda device: (1_000_000_000, 2_000_000_000)
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 1_024)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 2_048)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    output = tmp_path / "train-output"
    result = round0132_nodes.run_train(
        {"manifest": {"release_sha": release}},
        {
            "graph_manifest": str(graph_path),
            "outputs": [str(output)],
            "release_sha": release,
            "graph_release_sha": release,
        },
    )
    assert result["optimizer_updates"] == updates
    with (output / "train-receipt.json").open(encoding="utf-8") as handle:
        train_receipt = json.load(handle)
    with (output / "production-config.json").open(encoding="utf-8") as handle:
        production_receipt = json.load(handle)
    validate_seal(train_receipt, label="R0132 actual CPU smoke train")
    authenticated = validate_train_execution(
        train=train_receipt,
        config_receipt=production_receipt,
        graph=graph_manifest,
    )
    assert all(authenticated["checks"].values())

    checkpoint = torch.load(
        output / "model.pt", map_location="cpu", weights_only=True
    )
    features = SmokeDataset().features
    coordinates = torch.nn.functional.linear(
        torch.from_numpy(features), checkpoint["weight"], checkpoint["bias"]
    ).detach().numpy()
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
        provenance={"round": "0132", "mode": "cpu-smoke"},
    )
    assert panel["guards"]["coords_finite"] is True
    assert panel["guards"]["coords_collapsed"] is False
