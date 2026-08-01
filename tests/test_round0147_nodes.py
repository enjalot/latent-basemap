from __future__ import annotations

from basemap.round0140_subsystem_bisection import (
    SUCCESSFUL_UPDATES,
    TRAIN_MINIMUM_UPDATES_PER_S,
)
from basemap.round0147_row_policy import (
    TREATMENT,
    treatment_train_config,
)
from experiments.round0147_nodes import training_accounting_mismatches


def _signature(path: str, digest: str) -> dict:
    return {
        "canonical_path": path,
        "kind": "file",
        "bytes": 123,
        "sha256": digest * 64,
    }


def _closure() -> tuple[dict, dict, dict, int, int]:
    graph_edges = 123_456
    config, _digest = treatment_train_config(
        graph_signature=_signature("/tmp/graph.npz", "1"),
        graph_manifest_signature=_signature("/tmp/manifest.json", "2"),
        graph_edges=graph_edges,
        source_sha256="3" * 64,
        selection_sha256="4" * 64,
    )
    batch_size = config["optimizer"]["batch_size"]
    rows = SUCCESSFUL_UPDATES * batch_size
    runtime = {
        **config["execution"]["expected_pipeline_stamp"],
        "source_rows_gathered": rows,
        "destination_rows_gathered": rows,
        "host_prefetch_producer_batches": SUCCESSFUL_UPDATES + 1,
        "host_prefetch_consumer_batches": SUCCESSFUL_UPDATES,
    }
    accounting = {
        "lr_horizon": SUCCESSFUL_UPDATES,
        "positive_lr_optimizer_steps": SUCCESSFUL_UPDATES,
        "scheduler_steps": SUCCESSFUL_UPDATES,
        "attempted_batches": SUCCESSFUL_UPDATES,
        "finite_loss_batches": SUCCESSFUL_UPDATES,
        "optimizer_steps_attempted": SUCCESSFUL_UPDATES,
        "optimizer_steps_succeeded": SUCCESSFUL_UPDATES,
        "amp_overflow_skips": 0,
        "nonfinite_loss_skips": 0,
        "nonfinite_gradient_skips": 0,
        "stop_reason": "lr_horizon",
        "budget_satisfied": True,
        "n_pos_edges": graph_edges,
    }
    return config, runtime, accounting, graph_edges, batch_size


def test_treatment_config_stamps_complete_row_policy_package() -> None:
    config, digest = treatment_train_config(
        graph_signature=_signature("/tmp/graph.npz", "1"),
        graph_manifest_signature=_signature("/tmp/manifest.json", "2"),
        graph_edges=123_456,
        source_sha256="3" * 64,
        selection_sha256="4" * 64,
    )
    expected = config["execution"]["expected_pipeline_stamp"]
    assert config["arm"] == TREATMENT
    assert config["causal_matrix"]["row_policy_includes_induced_graph_change"] is True
    assert expected["pipeline"] == "host_weighted_jina_paired"
    assert expected["positive_sampling"] == "weighted_with_replacement"
    assert expected["source_representation"] == "fp16-control"
    assert expected["source_sha256"] == "3" * 64
    assert expected["selection_sha256"] == "4" * 64
    assert len(digest) == 64


def test_postfit_accounting_accepts_exact_production_closure() -> None:
    config, runtime, accounting, graph_edges, batch_size = _closure()
    assert training_accounting_mismatches(
        accounting=accounting,
        runtime=runtime,
        expected_pipeline=config["execution"]["expected_pipeline_stamp"],
        graph_edges=graph_edges,
        batch_size=batch_size,
        profiler={"aborted": False},
        rate=TRAIN_MINIMUM_UPDATES_PER_S + 1.0,
    ) == {}


def test_postfit_accounting_rejects_pipeline_endpoint_and_performance_drift() -> None:
    config, runtime, accounting, graph_edges, batch_size = _closure()
    runtime["pipeline"] = "uniform-fallback"
    runtime["source_rows_gathered"] -= 1
    accounting["optimizer_steps_succeeded"] -= 1
    mismatches = training_accounting_mismatches(
        accounting=accounting,
        runtime=runtime,
        expected_pipeline=config["execution"]["expected_pipeline_stamp"],
        graph_edges=graph_edges,
        batch_size=batch_size,
        profiler={"aborted": True},
        rate=TRAIN_MINIMUM_UPDATES_PER_S - 1.0,
    )
    assert {
        "pipeline",
        "optimizer_steps_succeeded",
        "endpoint_accounting",
        "performance",
    } <= set(mismatches)
