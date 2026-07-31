from __future__ import annotations

import random

import numpy as np
import pytest

from basemap.round0107_training import train_config
from basemap.round0128_k49_rescue import (
    FIXED_SUCCESSFUL_UPDATES,
    GRAPH_K,
    HEADLINE_KPI_RETENTION,
    R0107_SEED42_INITIAL_STATE_SHA256,
    Round0128Error,
    assert_treatment_isolation,
    k49_train_config,
    noninferiority_checks,
    paired_density_materiality,
)
from experiments.prepare_round0128_queue import (
    GPU_HOURS_CAP,
    P90_GPU_TOTAL_SECONDS,
    P90_GRAPH_PART_SECONDS,
    _require_issued_round,
)
from experiments.round0105_nodes import _exact_rerank
from experiments.round0106_nodes import (
    R0106_GRAPH_CONTRACT,
    _part_contract,
    _validate_directed_memberships,
)
from experiments.round0128_nodes import (
    EVALUATION_CONTRACT,
    GRAPH_CONTRACT,
    _k49_policy_metrics,
    _legacy_initial_reconstruction,
    _model_class,
)
from experiments.round0108_nodes import (
    R0108_EVALUATION_CONTRACT,
    _ordered_core_panel_arrays,
)


def _signature(letter: str, name: str) -> dict:
    return {
        "kind": "file",
        "canonical_path": f"/x/{name}",
        "bytes": 4,
        "sha256": letter * 64,
    }


def _graph(*, edges: int, prefix: str) -> tuple[dict, dict]:
    mapping = _signature("a", "map")
    graph = {
        "directed_edge_count": edges,
        "compact_mapping": mapping,
        "outputs": {
            "sources": _signature(prefix, f"{prefix}-sources"),
            "targets": _signature(chr(ord(prefix) + 1), f"{prefix}-targets"),
            "weights": _signature(chr(ord(prefix) + 2), f"{prefix}-weights"),
        },
    }
    return graph, _signature(chr(ord(prefix) + 3), f"{prefix}-graph")


def test_legacy_r0107_default_config_is_byte_semantically_frozen() -> None:
    graph = {
        "directed_edge_count": 1_000,
        "compact_mapping": _signature("a", "map"),
        "outputs": {
            "sources": _signature("b", "sources"),
            "targets": _signature("c", "targets"),
            "weights": _signature("d", "weights"),
        },
    }
    signature = {
        "kind": "file",
        "canonical_path": "/x/graph.json",
        "bytes": 99,
        "sha256": "e" * 64,
    }
    config, digest = train_config(
        graph_manifest=graph, graph_signature=signature
    )
    assert digest == (
        "dd5b5f31c598307ee909f53e821791858e5c213ec96ea73d56fe399e8df37275"
    )
    assert config["schema"] == "round0107-diverse-jina-train-config-v1"
    assert config["graph"]["n_neighbors_including_self"] == 16
    assert config["optimizer"]["successful_positive_lr_updates"] == 3
    assert config["optimizer"]["update_rule"] == "ceil(directed_fuzzy_edges/409)"
    assert config["execution"]["expected_pipeline_stamp"]["graph_degree"] == (
        "variable-symmetric-fuzzy-k15-topology"
    )


def test_legacy_r0106_contract_and_part_hash_are_frozen() -> None:
    assert R0106_GRAPH_CONTRACT.round_id == "0106"
    assert R0106_GRAPH_CONTRACT.k == 15
    assert R0106_GRAPH_CONTRACT.n_neighbors == 16
    search = {
        "index": {"x": 1},
        "index_receipt": {"x": 2},
        "qualification": {"x": 3},
        "decision": {"x": 4},
        "selected": {"nprobe": 64, "shortlist_width": 128},
    }
    assert _part_contract(
        part="english",
        release_sha="a" * 40,
        search=search,
        substrate_signature={"sha256": "b" * 64},
    ) == "9a84ff1672d128ec17fe6ddd2babde739e0841c86e84947ac231e88e0e47dde3"
    assert GRAPH_CONTRACT.round_id == "0128"
    assert GRAPH_CONTRACT.k == GRAPH_K
    assert GRAPH_CONTRACT.n_neighbors == 50


def test_legacy_r0108_npz_member_order_is_frozen() -> None:
    common = {
        key: np.asarray([index])
        for index, key in enumerate((
            "global_anchor_rows",
            "compact_anchor_rows",
            "group_ids",
            "graph_fuzzy_weights",
            "low_neighbors_top50",
            "high_radius",
            "low_radius",
            "anchor_family_sizes",
            "density_bootstrap",
            "density_permuted_null",
            "anchor_coordinates",
            "observed_map_mixing",
            "centroid_distances",
        ))
    }
    high = np.asarray([[1]], dtype=np.int64)
    graph = np.asarray([[2]], dtype=np.int64)
    legacy = _ordered_core_panel_arrays(
        R0108_EVALUATION_CONTRACT,
        common,
        high_neighbors=high,
        graph_neighbors=graph,
    )
    assert list(legacy) == [
        "global_anchor_rows",
        "compact_anchor_rows",
        "group_ids",
        "high_neighbors_top15",
        "graph_neighbors_top15",
        "graph_fuzzy_weights",
        "low_neighbors_top50",
        "high_radius",
        "low_radius",
        "anchor_family_sizes",
        "density_bootstrap",
        "density_permuted_null",
        "anchor_coordinates",
        "observed_map_mixing",
        "centroid_distances",
    ]
    treatment = _ordered_core_panel_arrays(
        EVALUATION_CONTRACT,
        common,
        high_neighbors=high,
        graph_neighbors=graph,
    )
    assert "high_neighbors_topk" in treatment
    assert "graph_neighbors_topk" in treatment


def test_k49_membership_contract_counts_exact_degree() -> None:
    rows = np.asarray([10, 11], dtype=np.int64)
    targets = np.vstack((
        np.arange(100, 100 + GRAPH_K),
        np.arange(200, 200 + GRAPH_K),
    )).astype(np.int32)
    sources = np.repeat(rows.astype(np.int32), GRAPH_K)
    closure = _validate_directed_memberships(
        rows=rows,
        all_targets=targets,
        sources=sources,
        targets=targets.reshape(-1),
        weights=np.ones(2 * GRAPH_K, dtype=np.float32),
        k=GRAPH_K,
    )
    assert closure["knn_edges"] == 2 * GRAPH_K
    assert closure["zero_memberships_eliminated"] == 0


def test_exact_rerank_k49_extends_same_shortlist_top15() -> None:
    rng = np.random.default_rng(128)
    queries = rng.normal(size=(2, 768)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    encoded = rng.integers(-127, 128, size=(120, 768), dtype=np.int16).astype(
        np.int8
    )
    scales = np.ones(120, dtype=np.float16)
    shortlist = np.vstack((np.arange(0, 64), np.arange(56, 120))).astype(
        np.int64
    )
    top15, stamp15 = _exact_rerank(
        queries=queries,
        shortlist=shortlist,
        encoded=encoded,
        scales=scales,
        k=15,
    )
    top49, stamp49 = _exact_rerank(
        queries=queries,
        shortlist=shortlist,
        encoded=encoded,
        scales=scales,
        k=49,
    )
    np.testing.assert_array_equal(top49[:, :15], top15)
    assert stamp15["selected_neighbors"] == 15
    assert stamp49["selected_neighbors"] == 49


def test_k49_fixed_policy_metrics_apply_registered_group_floors() -> None:
    exact = np.tile(np.arange(GRAPH_K), (5_632, 1)).astype(np.int64)
    selected = exact.copy()
    groups = np.repeat(np.arange(22), 256).astype(np.uint8)
    metrics = _k49_policy_metrics(
        selected,
        exact,
        group_ids=groups,
        unambiguous=np.ones(5_632, dtype=bool),
    )
    assert metrics["passed"] is True
    assert metrics["mean_recall_at_49_unambiguous"] == 1.0
    assert all(
        value["unambiguous_rows"] == 256
        for value in metrics["by_group"].values()
    )


def test_treatment_isolation_allows_only_graph_identity_and_degree() -> None:
    control_graph, control_signature = _graph(edges=597_026_276, prefix="b")
    treatment_graph, treatment_signature = _graph(edges=1_400_000_000, prefix="f")
    control, _ = train_config(
        graph_manifest=control_graph, graph_signature=control_signature
    )
    treatment, _ = k49_train_config(
        graph_manifest=treatment_graph, graph_signature=treatment_signature
    )
    proof = assert_treatment_isolation(control, treatment)
    assert proof["fixed_successful_updates"] == FIXED_SUCCESSFUL_UPDATES
    assert treatment["optimizer"]["successful_positive_lr_updates"] == (
        FIXED_SUCCESSFUL_UPDATES
    )
    treatment["optimizer"]["learning_rate"] = 0.002
    with pytest.raises(Round0128Error, match="changes more"):
        assert_treatment_isolation(control, treatment)


def test_fixed_r0107_dose_is_not_derived_from_k49_edges() -> None:
    left_graph, left_signature = _graph(edges=1_000_000_000, prefix="f")
    right_graph, right_signature = _graph(edges=2_000_000_000, prefix="j")
    left, _ = k49_train_config(
        graph_manifest=left_graph, graph_signature=left_signature
    )
    right, _ = k49_train_config(
        graph_manifest=right_graph, graph_signature=right_signature
    )
    assert left["optimizer"]["successful_positive_lr_updates"] == (
        FIXED_SUCCESSFUL_UPDATES
    )
    assert right["optimizer"]["successful_positive_lr_updates"] == (
        FIXED_SUCCESSFUL_UPDATES
    )
    assert left["graph"]["directed_edges"] != right["graph"]["directed_edges"]


def test_paired_density_is_k15_radius_paired_and_deterministic() -> None:
    rng = np.random.default_rng(1)
    high = np.exp(rng.normal(size=1_000))
    control_low = np.exp(0.2 * np.log(high) + rng.normal(size=1_000))
    treatment_low = np.exp(0.9 * np.log(high) + 0.2 * rng.normal(size=1_000))
    left, left_draws = paired_density_materiality(
        control_high_radius=high,
        control_low_radius=control_low,
        treatment_high_radius=high.copy(),
        treatment_low_radius=treatment_low,
    )
    right, right_draws = paired_density_materiality(
        control_high_radius=high,
        control_low_radius=control_low,
        treatment_high_radius=high.copy(),
        treatment_low_radius=treatment_low,
    )
    assert left == right
    np.testing.assert_array_equal(left_draws, right_draws)
    assert left["outcome"] == "k49-materially-improves-native-density"
    changed = high.copy()
    changed[0] += 1e-12
    with pytest.raises(Round0128Error, match="arrays changed"):
        paired_density_materiality(
            control_high_radius=high,
            control_low_radius=control_low,
            treatment_high_radius=changed,
            treatment_low_radius=treatment_low,
        )


def _core(ffr: float, r10: float, r50: float) -> dict:
    return {
        "metrics": {
            "global": {
                "ffr": ffr,
                "recall_at_10": r10,
                "recall_at_50_of_high10": r50,
            }
        },
        "decision": {
            "checks": {
                "coordinates_finite_and_noncollapsed": True,
                "every_language_ffr_at_least_0_40_of_pooled_english": True,
                "global_ffr_at_least_0_40": True,
                "global_recall50_strictly_exceeds_recall10": True,
            }
        },
    }


def _ood(polish: float, inmix: float, ratio: float) -> dict:
    return {
        "language_cells": {
            "pol_Latn": {
                "probe": {"recall_at_50_of_high10": polish}
            }
        },
        "headline_decision": {
            "in_mix_median_recall_at_50_of_high10": inmix,
            "polish_to_in_mix_median_ratio": ratio,
            "passed": True,
        },
    }


def test_ninety_seven_percent_headlines_gate_but_ratio_is_diagnostic() -> None:
    control_core = _core(0.5, 0.2, 0.4)
    treatment_core = _core(
        HEADLINE_KPI_RETENTION * 0.5,
        HEADLINE_KPI_RETENTION * 0.2,
        HEADLINE_KPI_RETENTION * 0.4,
    )
    result = noninferiority_checks(
        control_core=control_core,
        treatment_core=treatment_core,
        control_ood=_ood(0.3, 0.4, 0.75),
        treatment_ood=_ood(0.291, 0.388, 0.01),
    )
    assert result["passed"] is True
    assert "polish_to_in_mix_median_ratio" not in result["checks"]
    failed = noninferiority_checks(
        control_core=control_core,
        treatment_core=treatment_core,
        control_ood=_ood(0.3, 0.4, 0.75),
        treatment_ood=_ood(0.2909, 0.388, 0.75),
    )
    assert failed["checks"]["polish_recall50_at_least_0p97_control"] is False


def test_initial_state_matches_deterministic_r0107_reconstruction() -> None:
    import torch

    reconstruction = _legacy_initial_reconstruction()
    assert reconstruction["observed_sha256"] == R0107_SEED42_INITIAL_STATE_SHA256
    assert reconstruction["historical_evidence_kind"] == (
        "deterministic-reconstruction-not-original-reviewed-receipt"
    )
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model = _model_class(reconstruction)(
        n_components=2,
        hidden_dim=2048,
        n_layers=3,
        architecture="residual_bottleneck",
        device="cpu",
    )
    model._init_model(768)
    receipt = model._initial_model_state_receipt
    assert receipt["same_release_byte_equal"] is True
    assert receipt["observed_sha256"] == R0107_SEED42_INITIAL_STATE_SHA256


def test_draft_round_cannot_materialize_queue(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "round-0128-2026-07-31.md"
    path.write_text('---\nround_id: "0128"\nstatus: draft\n---\n')
    monkeypatch.setattr(
        "experiments.prepare_round0128_queue.ROUND_FILE_GLOB", str(path)
    )
    with pytest.raises(RuntimeError, match="found 0"):
        _require_issued_round()


def test_p90_budget_has_graph_and_terminal_headroom() -> None:
    assert 3 * P90_GRAPH_PART_SECONDS == 3_600.0
    assert P90_GPU_TOTAL_SECONDS == 23_700.0
    assert P90_GPU_TOTAL_SECONDS / 3_600 == pytest.approx(6.5833333333)
    assert GPU_HOURS_CAP == 8.0
    assert P90_GPU_TOTAL_SECONDS < GPU_HOURS_CAP * 3_600
