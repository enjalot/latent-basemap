from __future__ import annotations

import inspect

from basemap.round0140_subsystem_bisection import (
    CURRENT_GRAPH_CURRENT_HOST,
    HISTORICAL_GRAPH_CURRENT_HOST,
    HISTORICAL_GRAPH_DEVICE_REPRO,
    HISTORICAL_FLOORS,
    NEW_CELLS,
    RESTORATION_FLOORS,
    build_decision,
    historical_preprocessing_stamp,
    host_train_config,
)
from experiments import round0027_nodes, round0140_nodes


def _cell(*, value: float) -> dict:
    return {
        "panel": {
            "ffr": value,
            "purity": {"k256": value, "k1024": value},
            "density": 0.2,
        },
        "projection": {"ffr": value, "recall_at_10": value},
    }


def _cells(current: float, graph: float, device: float) -> dict:
    return {
        CURRENT_GRAPH_CURRENT_HOST: _cell(value=current),
        HISTORICAL_GRAPH_CURRENT_HOST: _cell(value=graph),
        HISTORICAL_GRAPH_DEVICE_REPRO: _cell(value=device),
    }


def test_restoration_floors_use_historical_margins_and_context_maxima():
    assert RESTORATION_FLOORS["ffr"] == HISTORICAL_FLOORS["ffr"]
    assert RESTORATION_FLOORS["ood_recall_at_10"] == 0.00946
    assert set(RESTORATION_FLOORS) == {
        "ffr",
        "purity_fidelity_k256",
        "purity_fidelity_k1024",
        "projection_ffr",
        "ood_recall_at_10",
    }


def test_selector_identifies_fixed_historical_rows_when_both_host_cells_restore():
    decision = build_decision(_cells(0.99, 0.99, 0.99))
    assert decision["outcome"] == "historical-row-universe-restores-with-current-trainer"
    assert decision["next_action"] == "recover-and-test-row-policy-on-current-population"


def test_selector_identifies_historical_graph_when_only_graph_swap_restores():
    decision = build_decision(_cells(0.01, 0.99, 0.99))
    assert decision["outcome"] == "historical-graph-subsystem-restores"
    assert decision["historical_graph_current_host_restores"] is True


def test_selector_identifies_trainer_side_when_host_cells_fail_but_repro_restores():
    decision = build_decision(_cells(0.01, 0.01, 0.99))
    assert decision["outcome"] == "current-host-trainer-subsystem-does-not-restore"
    assert decision["historical_reproduction_restores"] is True


def test_selector_preserves_asymmetric_current_only_restore_as_interaction():
    decision = build_decision(_cells(0.99, 0.01, 0.99))
    assert decision["outcome"] == "subsystem-interaction-unresolved"
    assert decision["next_action"] == (
        "issue-one-cell-current-graph-historical-device-interaction"
    )


def test_selector_refuses_causal_claim_when_historical_recipe_does_not_reproduce():
    decision = build_decision(_cells(0.01, 0.01, 0.01))
    assert decision["outcome"] == "historical-recipe-not-reproduced-current-release"
    assert decision["next_action"] == "audit-reproduction-before-causal-claim"


def test_host_configs_bind_same_rows_and_change_only_graph_subsystem():
    graph = {"canonical_path": "/tmp/graph.npz", "kind": "file", "bytes": 1, "sha256": "a" * 64}
    manifest = {"canonical_path": "/tmp/manifest.json", "kind": "file", "bytes": 1, "sha256": "b" * 64}
    current, current_sha = host_train_config(
        cell=CURRENT_GRAPH_CURRENT_HOST,
        graph_signature=graph,
        graph_manifest_signature=manifest,
        graph_edges=100,
    )
    historical, historical_sha = host_train_config(
        cell=HISTORICAL_GRAPH_CURRENT_HOST,
        graph_signature=graph,
        graph_manifest_signature=manifest,
        graph_edges=100,
    )
    assert current_sha != historical_sha
    assert current["input_preprocessing"] == historical["input_preprocessing"]
    assert current["optimizer"] == historical["optimizer"]
    assert current["model"] == historical["model"]
    assert current["causal_matrix"]["graph_subsystem"] == "current-r0104-style"
    assert historical["causal_matrix"]["graph_subsystem"] == "historical-r0037-byte-exact"


def test_preprocessing_stamp_names_exact_historical_row_universe():
    stamp = historical_preprocessing_stamp()
    assert stamp["row_universe"] == "R0037-jina-en-2M-nested-exact-order"
    assert stamp["operation"] == "exact-r0037-fp16-to-device-fp32"
    assert len(stamp["identity_sha256"]) == 64


def test_node_source_contains_execution_and_postfit_guards():
    source = inspect.getsource(round0140_nodes)
    for required in (
        "source_rows_gathered",
        "optimizer_steps_succeeded",
        "no_pipeline_stamp_drift",
        "same_ordered_training_rows_across_new_cells",
        "cross_round_training_row_equivalence_claimed",
        "density_role",
        "validate_seal",
        "train_release_shas",
    ):
        assert required in source
    assert set(NEW_CELLS) == {
        CURRENT_GRAPH_CURRENT_HOST,
        HISTORICAL_GRAPH_CURRENT_HOST,
        HISTORICAL_GRAPH_DEVICE_REPRO,
    }


def test_short_device_canary_disables_only_full_budget_requirement(monkeypatch):
    captured = {}

    class FakeParametricUMAP:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.require_full_budget = kwargs["require_full_budget"]

    import basemap.pumap.parametric_umap as pumap

    monkeypatch.setattr(pumap, "ParametricUMAP", FakeParametricUMAP)
    config = {
        "model": {
            "output_dimension": 2,
            "hidden_dimension": 2048,
            "hidden_layers": 3,
            "a": 1.0,
            "b": 1.0,
            "low_dim_kernel": "legacy_lp",
            "use_batchnorm": False,
            "use_dropout": False,
            "architecture": "residual_bottleneck",
        },
        "optimizer": {
            "correlation_weight": 0.0,
            "learning_rate": 0.001,
            "batch_size": 8192,
            "clip_grad_norm": 1.0,
            "positive_ratio": 0.05,
            "warmup_successful_updates": 200,
            "successful_positive_lr_updates": 500_000,
            "use_amp": "bf16",
            "positive_target_mode": "binary",
            "reject_neighbors": False,
            "weighted_edge_sampling": True,
        },
        "execution": {
            "required_pipeline": "device",
            "gpu_resident_data": "auto",
            "gpu_resident_vram_budget_gb": 31.0,
        },
        "graph": {
            "manifest_path": "/tmp/manifest.json",
            "manifest_sha256": "a" * 64,
        },
    }
    canary = round0027_nodes._new_model(
        config, require_full_budget=False
    )
    assert canary.require_full_budget is False
    assert captured["total_steps_estimate"] == 500_000
    production = round0027_nodes._new_model(config)
    assert production.require_full_budget is True
    canary_source = inspect.getsource(round0027_nodes.run_sampler_canary)
    assert '"budget_satisfied": False' in canary_source
