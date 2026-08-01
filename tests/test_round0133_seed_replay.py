"""Focused contracts and CUDA-hidden train-to-panel smoke for R0133."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from basemap.panel_v2 import PanelV2Config, score_panel
from basemap.round0107_training import (
    BATCH_SIZE,
    POSITIVE_ROWS_PER_UPDATE,
    train_config,
)
from basemap.round0132_scale_bridge import (
    GRAPH_K,
    HALF_RETAINED_ROWS,
    OUTCOME_INCONCLUSIVE,
    OUTCOME_QUALITY_REGRESSION,
    OUTCOME_SUPPORTED,
    PIPELINE,
    PIPELINE_SCHEMA,
    POSITIVE_DESTINATION_POLICY,
    SAMPLER_CLASS,
    SCALE_POLICY_CAPABILITY,
    seal,
    validate_seal,
)
from basemap.round0133_seed_replay import (
    CONCORDANT,
    DISCORDANT,
    PRODUCTION_CONFIG_SCHEMA,
    ROUND_ID,
    SEED,
    TRAIN_CONFIG_SCHEMA,
    TRAIN_RECEIPT_SCHEMA,
    TWO_SEED_CAPABILITY,
    Round0133Error,
    assert_no_r0110_coordinate_inputs,
    combine_seed_decisions,
    validate_seed43_train_execution,
)
from experiments import round0107_nodes, round0133_nodes
from experiments.prepare_round0133_queue import (
    GPU_HOURS_EXPECTED,
    GPU_HOURS_MAXIMUM,
    GPU_HOURS_MINIMUM,
    GPU_HOURS_P90,
    P90_GPU_TOTAL_SECONDS,
    _discover_accepted_r0132_review,
    _require_issued_round,
    _require_review_result,
    _require_round_release,
)


def _signature(path: str = "/data/synthetic-r0132-graph.json") -> dict:
    return {
        "kind": "file",
        "canonical_path": path,
        "bytes": 1,
        "sha256": "e" * 64,
    }


def _graph(edges: int = 818) -> dict:
    return {
        "schema": "round0132-half-fuzzy-graph-v1",
        "round_id": "0132",
        "retained_rows": HALF_RETAINED_ROWS,
        "dimension": 768,
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


def _configs(graph: dict | None = None, signature: dict | None = None):
    graph = graph or _graph()
    signature = signature or _signature()
    common = {
        "graph_manifest": graph,
        "graph_signature": signature,
        "compact_retained_rows": HALF_RETAINED_ROWS,
        "pipeline": PIPELINE,
        "pipeline_schema": PIPELINE_SCHEMA,
        "sampler_class": SAMPLER_CLASS,
        "update_rule": "ceil(actual-R0132-directed-fuzzy-edges/409)",
        "positive_destination_policy": POSITIVE_DESTINATION_POLICY,
    }
    seed43, digest43 = train_config(
        **common, seed=SEED, schema=TRAIN_CONFIG_SCHEMA
    )
    seed42, digest42 = train_config(
        **common, seed=42, schema="round0132-half-train-config-v1"
    )
    return seed43, digest43, seed42, digest42


def _train_bundle():
    graph = _graph()
    seed43, digest43, seed42, digest42 = _configs(graph)
    updates = 2
    emitted = updates * POSITIVE_ROWS_PER_UPDATE
    runtime = {
        **seed43["execution"]["expected_pipeline_stamp"],
        "endpoint_gather_calls": updates,
        "source_rows_gathered": updates * BATCH_SIZE,
        "destination_rows_gathered": updates * BATCH_SIZE,
        "host_prefetch_batches_filled": updates,
        "host_prefetch_producer_batches": updates,
        "host_prefetch_consumer_batches": updates,
        "host_prefetch_source_rows_filled": updates * BATCH_SIZE,
        "host_prefetch_destination_rows_filled": updates * BATCH_SIZE,
        "weight_proposals": emitted + 11,
        "weight_acceptances": emitted + 3,
        "weight_emitted_draws": emitted,
        "weight_buffered_draws": 3,
        "weight_acceptance_rate": (emitted + 3) / (emitted + 11),
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
        "n_pos_edges": 818,
        **{
            f"pipeline_{key}": value
            for key, value in runtime.items()
            if key not in seed43["execution"]["expected_pipeline_stamp"]
        },
    }
    train = {
        "schema": TRAIN_RECEIPT_SCHEMA,
        "round_id": ROUND_ID,
        "production_config_sha256": digest43,
        "optimizer_updates": updates,
        "update_derivation": {
            "directed_fuzzy_edges": 818,
            "positive_rows_per_update": POSITIVE_ROWS_PER_UPDATE,
            "successful_updates": updates,
            "expected_positive_draws": emitted,
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
    config43 = {
        "schema": PRODUCTION_CONFIG_SCHEMA,
        "round_id": ROUND_ID,
        "config": seed43,
        "config_sha256": digest43,
    }
    config42 = {
        "schema": "round0132-half-production-config-v1",
        "round_id": "0132",
        "config": seed42,
        "config_sha256": digest42,
    }
    return train, config43, graph, config42


def test_seed43_train_reduces_exactly_to_accepted_r0132_policy():
    train, config43, graph, config42 = _train_bundle()
    authenticated = validate_seed43_train_execution(
        train=train,
        config_receipt=config43,
        graph=graph,
        accepted_r0132_config_receipt=config42,
    )
    assert authenticated["registered_seed"] == 43
    assert authenticated["accepted_r0132_seed"] == 42
    assert authenticated["only_registered_rng_identity_changed"] is True
    assert all(authenticated["checks"].values())


@pytest.mark.parametrize(
    ("target", "key", "value"),
    [
        ("optimizer", "seed", 44),
        ("optimizer", "learning_rate", 0.002),
        ("execution", "required_pipeline", "uniform-fallback"),
    ],
)
def test_seed43_train_rejects_unregistered_identity_or_policy_drift(
    target, key, value
):
    train, config43, graph, config42 = _train_bundle()
    config43["config"][target][key] = value
    config43["config_sha256"] = sha256_bytes(canonical_json(config43["config"]))
    train["production_config_sha256"] = config43["config_sha256"]
    with pytest.raises(Round0133Error):
        validate_seed43_train_execution(
            train=train,
            config_receipt=config43,
            graph=graph,
            accepted_r0132_config_receipt=config42,
        )


def _seed42_decision(outcome: str, failed: tuple[str, ...] = ()) -> dict:
    checks = {"native": "native" not in failed, "ood": "ood" not in failed}
    return {
        "schema": "round0132-scale-policy-decision-v1",
        "round_id": "0132",
        "outcome": outcome,
        "quality_and_ood_noninferiority": {"checks": checks},
        "capabilities_produced": [SCALE_POLICY_CAPABILITY],
    }


def _seed43_decision(outcome: str, failed: tuple[str, ...] = ()) -> dict:
    checks = {"native": "native" not in failed, "ood": "ood" not in failed}
    return {
        "outcome": outcome,
        "quality_and_ood_noninferiority": {"checks": checks},
    }


def test_two_seed_combiner_never_pools_draws_or_claims_seed_variance():
    combined = combine_seed_decisions(
        accepted_seed42=_seed42_decision(OUTCOME_SUPPORTED),
        seed43=_seed43_decision(OUTCOME_SUPPORTED),
    )
    assert combined["concordance"] == CONCORDANT
    assert combined["bootstrap_combination"] == "none"
    assert combined["anchors_combined"] is False
    assert combined["point_estimates_averaged"] is False
    assert combined["population_seed_variance_estimated"] is False
    assert combined["capabilities_produced"] == [TWO_SEED_CAPABILITY]


def test_quality_regression_requires_identical_failed_gate_set_for_concordance():
    seed42 = _seed42_decision(OUTCOME_QUALITY_REGRESSION, ("native",))
    mismatched = combine_seed_decisions(
        accepted_seed42=seed42,
        seed43=_seed43_decision(OUTCOME_QUALITY_REGRESSION, ("ood",)),
    )
    assert mismatched["same_seed_level_outcome"] is True
    assert mismatched["same_failed_quality_gate_set"] is False
    assert mismatched["concordance"] == DISCORDANT
    matched = combine_seed_decisions(
        accepted_seed42=seed42,
        seed43=_seed43_decision(OUTCOME_QUALITY_REGRESSION, ("native",)),
    )
    assert matched["concordance"] == CONCORDANT


def test_different_seed_outcomes_are_reported_discordant_not_averaged():
    combined = combine_seed_decisions(
        accepted_seed42=_seed42_decision(OUTCOME_INCONCLUSIVE),
        seed43=_seed43_decision(OUTCOME_SUPPORTED),
    )
    assert combined["concordance"] == DISCORDANT
    assert combined["capabilities_produced"] == [TWO_SEED_CAPABILITY]


def test_r0110_coordinate_stream_is_explicitly_forbidden():
    with pytest.raises(Round0133Error, match="forbidden"):
        assert_no_r0110_coordinate_inputs({
            "expected_inputs": [{
                "canonical_path": (
                    "/data/latent-basemap/runs/round-0110/queue/artifacts/"
                    "coordinates-seed43/chunk-00000/coordinates.npy"
                )
            }]
        })
    assert_no_r0110_coordinate_inputs({
        "canonical_path": (
            "/data/latent-basemap/runs/round-0109/queue/artifacts/"
            "train-diverse-jina-25m-seed43/model.pt"
        )
    })


def test_dispatch_rejects_r0110_before_any_handler_runs(tmp_path):
    with pytest.raises(Round0133Error, match="forbidden"):
        round0133_nodes.run_job(
            {"manifest": {"round_id": ROUND_ID}},
            {
                "action": "train_u12_seed43",
                "outputs": [str(tmp_path / "output")],
                "leak": (
                    "/data/latent-basemap/runs/round-0110/queue/artifacts/"
                    "coordinates-seed43"
                ),
            },
        )


def test_ood_probe_reuses_accepted_truth_without_a_truth_recompute():
    rng = np.random.default_rng(133)
    corpus = rng.normal(size=(64, 8)).astype(np.float32)
    queries = rng.normal(size=(5, 8)).astype(np.float32)
    normalized_corpus = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
    normalized_queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    truth = np.argsort(
        -(normalized_queries @ normalized_corpus.T), axis=1
    )[:, :10].astype(np.int64)

    class Projection:
        def transform(self, values, *, batch_size):
            assert batch_size > 0
            array = np.asarray(values, dtype=np.float32)
            return np.column_stack((array[:, 0], array[:, 1] + array[:, 2]))

    duplicate = round0133_nodes.exact_split_duplicate_diagnostics(
        corpus, queries
    )
    report = round0133_nodes._matched_probe_with_reviewed_truth(
        name="synthetic",
        corpus=corpus,
        queries=queries,
        control_model=Projection(),
        treatment_model=Projection(),
        duplicate_policy="require-disjoint",
        accepted_probe={
            "name": "synthetic",
            "corpus_rows": len(corpus),
            "query_rows": len(queries),
            "duplicate_control": duplicate,
            "truth_guard": {"backend": "accepted-r0132"},
        },
        accepted_truth=truth,
    )
    assert report["truth_reused_byte_for_byte_from_accepted_r0132"] is True
    assert np.array_equal(report["arrays"]["exact_high_d_top10"], truth)


def test_terminal_authenticates_both_transform_and_panel_model_lineages():
    def signature(name):
        return {
            "kind": "file",
            "canonical_path": f"/synthetic/{name}",
            "bytes": 1,
            "sha256": name[0] * 64,
        }

    authenticated_train = {
        "model": signature("a-model"),
        "train_receipt": signature("b-train"),
        "production_config": signature("c-config"),
        "graph_manifest": signature("d-graph"),
    }
    accepted_r0109 = {
        "model": signature("e-model"),
        "train_receipt": signature("f-train"),
        "production_config": signature("9-config"),
        "graph_manifest": signature("8-graph"),
    }
    control_signature = signature("7-control-transform")
    treatment_signature = signature("6-treatment-transform")
    shared_mapping = signature("5-mapping")
    shared_substrate = signature("4-substrate")
    control_transform = {
        "map_key": "r0133-diverse-jina-u12-seed43",
        "model": authenticated_train["model"],
        "train_receipt": authenticated_train["train_receipt"],
        "production_config": authenticated_train["production_config"],
        "model_training_graph": authenticated_train["graph_manifest"],
        "u12_scientific_universe_graph": authenticated_train["graph_manifest"],
        "compact_mapping": shared_mapping,
        "substrate": shared_substrate,
    }
    treatment_transform = {
        "map_key": "r0109-diverse-jina-25m-seed43-on-r0132-u12",
        "model": accepted_r0109["model"],
        "train_receipt": accepted_r0109["train_receipt"],
        "production_config": accepted_r0109["production_config"],
        "model_training_graph": accepted_r0109["graph_manifest"],
        "u12_scientific_universe_graph": authenticated_train["graph_manifest"],
        "compact_mapping": shared_mapping,
        "substrate": shared_substrate,
    }
    native_lineage = {
        "control_12p5m_transform": control_signature,
        "treatment_25m_transform": treatment_signature,
    }
    ood_lineage = {
        "control_12p5m_train_receipt": authenticated_train["train_receipt"],
        "control_12p5m_production_config": authenticated_train[
            "production_config"
        ],
        "control_12p5m_graph": authenticated_train["graph_manifest"],
        "treatment_25m_train_receipt": accepted_r0109["train_receipt"],
        "treatment_25m_production_config": accepted_r0109[
            "production_config"
        ],
        "treatment_25m_graph": accepted_r0109["graph_manifest"],
    }
    kwargs = {
        "native_lineage": native_lineage,
        "ood_lineage": ood_lineage,
        "control_transform": control_transform,
        "control_transform_signature": control_signature,
        "treatment_transform": treatment_transform,
        "treatment_transform_signature": treatment_signature,
        "authenticated_train": authenticated_train,
        "accepted_r0109": accepted_r0109,
    }
    round0133_nodes._validate_decision_model_lineage(**kwargs)
    treatment_transform["model"] = authenticated_train["model"]
    with pytest.raises(Round0133Error, match="lineage disagrees"):
        round0133_nodes._validate_decision_model_lineage(**kwargs)


def test_conditional_draft_cannot_materialize(monkeypatch, tmp_path):
    import experiments.prepare_round0133_queue as prepare

    draft = tmp_path / "round-0133-2026-07-31.md"
    draft.write_text("---\nstatus: draft\n---\n", encoding="utf-8")
    monkeypatch.setattr(prepare, "ROUND_FILE_GLOB", str(tmp_path / "round-0133-*.md"))
    with pytest.raises(RuntimeError, match="exactly one issued"):
        _require_issued_round()


def test_issued_round_must_bind_exact_r0133_release(tmp_path):
    path = tmp_path / "round.md"
    path.write_text(
        f"---\nbase_commit: {'a' * 40}\nstatus: issued\n---\n",
        encoding="utf-8",
    )
    _require_round_release(str(path), "a" * 40)
    with pytest.raises(RuntimeError, match="base_commit"):
        _require_round_release(str(path), "b" * 40)


def test_r0132_review_discovery_requires_capability_acceptance(monkeypatch, tmp_path):
    import experiments.prepare_round0133_queue as prepare

    review = tmp_path / "review-0132-2026-08-01.md"
    review.write_text(
        "---\nround_id: \"0132\"\nstatus: accepted\nreleases: []\n---\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(prepare, "R0132_REVIEW_GLOB", str(tmp_path / "review-0132-*.md"))
    with pytest.raises(RuntimeError, match="found 0"):
        _discover_accepted_r0132_review()
    review.write_text(
        "---\nround_id: \"0132\"\nstatus: accepted\n"
        f"releases: [\"capability:{SCALE_POLICY_CAPABILITY}\"]\n---\n",
        encoding="utf-8",
    )
    assert _discover_accepted_r0132_review() == str(review)


def test_review_must_close_exact_result_and_capability(tmp_path):
    result = tmp_path / "result-0132-2026-08-01.md"
    result.write_text(
        "---\nround_id: \"0132\"\nstatus: complete\n"
        f"release_commit: \"{'a' * 40}\"\n"
        f"capabilities_produced: [\"{SCALE_POLICY_CAPABILITY}\"]\n---\n",
        encoding="utf-8",
    )
    result_sha = expected_input_signature(str(result))["sha256"]
    review = tmp_path / "review-0132-2026-08-01.md"
    review.write_text(
        "---\nround_id: \"0132\"\nstatus: accepted\n"
        "result: result-0132-2026-08-01.md\n"
        f"result_sha256: \"{result_sha}\"\n"
        f"verified_release_commit: \"{'a' * 40}\"\n"
        f"releases: [\"capability:{SCALE_POLICY_CAPABILITY}\"]\n---\n",
        encoding="utf-8",
    )
    evidence = _require_review_result(
        str(review),
        expected_review_sha256=None,
        round_id="0132",
        capability=SCALE_POLICY_CAPABILITY,
    )
    assert evidence["result"]["sha256"] == result_sha
    result.write_text(result.read_text().replace("complete", "failed"), encoding="utf-8")
    with pytest.raises(RuntimeError, match="exact result"):
        _require_review_result(
            str(review),
            expected_review_sha256=None,
            round_id="0132",
            capability=SCALE_POLICY_CAPABILITY,
        )


def test_registered_provisional_budget_matches_draft():
    assert (
        GPU_HOURS_MINIMUM,
        GPU_HOURS_EXPECTED,
        GPU_HOURS_P90,
        GPU_HOURS_MAXIMUM,
    ) == (1.7, 1.85, 2.20, 3.5)
    assert P90_GPU_TOTAL_SECONDS == 7_920.0


def test_cuda_hidden_actual_train_seal_reload_transform_panel_smoke(
    monkeypatch, tmp_path: Path
):
    """Exercise R0133's production post-fit handoff without allocating CUDA."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    import torch

    assert not __import__("os").environ["CUDA_VISIBLE_DEVICES"]
    torch.set_num_threads(1)
    release = "a" * 40
    graph_release = "b" * 40
    updates = 201
    edges = updates * POSITIVE_ROWS_PER_UPDATE
    graph_path = tmp_path / "graph-manifest.json"
    graph_manifest = seal({
        "schema": "round0132-half-fuzzy-graph-v1",
        "round_id": "0132",
        "release_sha": graph_release,
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
    common = {
        "graph_manifest": graph_manifest,
        "graph_signature": graph_signature,
        "compact_retained_rows": HALF_RETAINED_ROWS,
        "pipeline": PIPELINE,
        "pipeline_schema": PIPELINE_SCHEMA,
        "sampler_class": SAMPLER_CLASS,
        "update_rule": "ceil(actual-R0132-directed-fuzzy-edges/409)",
        "positive_destination_policy": POSITIVE_DESTINATION_POLICY,
    }
    config43, digest43 = train_config(
        **common, seed=SEED, schema=TRAIN_CONFIG_SCHEMA
    )
    config42, digest42 = train_config(
        **common, seed=42, schema="round0132-half-train-config-v1"
    )
    expected_rows = updates * BATCH_SIZE
    emitted = updates * POSITIVE_ROWS_PER_UPDATE
    accepted = emitted + 3
    proposals = accepted + 101
    runtime = {
        **config43["execution"]["expected_pipeline_stamp"],
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
            if key not in config43["execution"]["expected_pipeline_stamp"]
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
            self.features = np.random.default_rng(133).normal(size=(240, 8)).astype(
                np.float32
            )
            self.substrate = {
                "signature": {
                    "kind": "file",
                    "canonical_path": "/synthetic/r0133-substrate.json",
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

    monkeypatch.setattr(round0107_nodes, "_graph", lambda active, job, **kwargs: fake_graph)
    monkeypatch.setattr(
        round0107_nodes, "train_config", lambda **kwargs: (config43, digest43)
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
    result = round0133_nodes.run_train(
        {"manifest": {"release_sha": release}},
        {
            "graph_manifest": str(graph_path),
            "graph_manifest_sha256": graph_signature["sha256"],
            "outputs": [str(output)],
            "release_sha": release,
            "graph_release_sha": graph_release,
        },
    )
    assert result["optimizer_updates"] == updates
    train_receipt = json.loads((output / "train-receipt.json").read_text())
    production = json.loads((output / "production-config.json").read_text())
    validate_seal(train_receipt, label="R0133 actual CPU smoke train")
    authenticated = validate_seed43_train_execution(
        train=train_receipt,
        config_receipt=production,
        graph=graph_manifest,
        accepted_r0132_config_receipt={
            "schema": "round0132-half-production-config-v1",
            "round_id": "0132",
            "config": config42,
            "config_sha256": digest42,
        },
    )
    assert all(authenticated["checks"].values())

    checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
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
        provenance={"round": ROUND_ID, "mode": "cuda-hidden-cpu-smoke"},
    )
    assert panel["guards"]["coords_finite"] is True
    assert panel["guards"]["coords_collapsed"] is False
    sealed_panel = seal({
        "schema": "round0133-cpu-smoke-panel-v1",
        "train_receipt": expected_input_signature(str(output / "train-receipt.json")),
        "coordinates_finite": True,
        "panel_guards": panel["guards"],
        "path": "train->account->seal->reload->transform->panel",
    })
    validate_seal(sealed_panel, label="R0133 CPU smoke panel")
