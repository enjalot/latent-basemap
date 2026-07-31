"""Contract and bounded CPU smoke tests for R0121."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

from basemap.artifact_identity import (
    expected_input_signature,
    ordered_array_sha256,
)
from basemap.panel_v2 import PanelV2Config, score_panel
from basemap.round0113_prompt_contrast import (
    _DYNAMIC_PIPELINE_COUNTERS,
    read_sealed,
    train_config as r0115_train_config,
)
from basemap.round0121_degree_bridge import (
    DIAGNOSTIC_SCHEMA,
    DENSITY_SCHEMA,
    GRAPH_DEGREE,
    GRAPH_SEARCH_NEIGHBORS,
    OUTCOME_NOT_SUFFICIENT,
    OUTCOME_SUFFICIENT,
    REGISTERED_DENSITY_FLOOR,
    classify_degree_bridge,
    train_config,
)
from experiments import round0121_nodes


def _signature(path: str = "/tmp/r0121-graph") -> dict[str, Any]:
    return {
        "kind": "file",
        "canonical_path": path,
        "bytes": 123,
        "sha256": "a" * 64,
    }


def test_only_registered_training_change_is_graph_degree() -> None:
    graph = _signature()
    manifest = _signature("/tmp/r0121-manifest")
    control, _ = r0115_train_config(
        "raw",
        graph_signature=graph,
        graph_manifest_signature=manifest,
        graph_edges=100,
        retained_rows=1_993_761,
    )
    treatment, _ = train_config(
        graph_signature=graph,
        graph_manifest_signature=manifest,
        graph_edges=100,
        retained_rows=1_993_761,
    )
    assert treatment["input"] == control["input"]
    assert treatment["model"] == control["model"]
    assert treatment["optimizer"] == control["optimizer"]
    assert treatment["graph"]["k"] == GRAPH_DEGREE
    assert (
        treatment["graph"]["n_neighbors_including_self"]
        == GRAPH_SEARCH_NEIGHBORS
    )
    for key in (
        "device_count",
        "required_pipeline",
        "gpu_resident_data",
        "gpu_resident_vram_budget_gb",
        "minimum_train_upd_s",
        "warning_train_upd_s",
        "performance_subfloor_patience",
        "performance_windows",
    ):
        assert treatment["execution"][key] == control["execution"][key]
    stamp = treatment["execution"]["expected_pipeline_stamp"]
    assert stamp["sampler_class"] == "PromptWeightedJinaSampler"
    assert stamp["positive_sampling"] == control["execution"][
        "expected_pipeline_stamp"
    ]["positive_sampling"]
    assert stamp["graph_degree"] == "variable-symmetric-fuzzy-k15-topology"


def test_explicit_self_knn_enforces_15_distinct_nonself() -> None:
    rows = np.asarray([4, 20], dtype=np.int64)
    ids = np.vstack(
        (
            np.asarray([9, 4, *range(10, 24)], dtype=np.int64),
            np.asarray([20, *range(30, 45)], dtype=np.int64),
        )
    )
    similarities = np.linspace(1.0, 0.5, ids.size).reshape(ids.shape)
    canonical, distances, nonself = round0121_nodes._explicit_self_knn(
        ids, similarities, rows
    )
    assert canonical.shape == (2, 16)
    assert np.array_equal(canonical[:, 0], rows)
    assert nonself.shape == (2, 15)
    assert not np.any(nonself == rows[:, None])
    assert np.all(distances[:, 0] == 0)

    malformed = ids.copy()
    malformed[0, 1] = 8
    with pytest.raises(
        round0121_nodes.Round0121Error, match="exactly one self"
    ):
        round0121_nodes._explicit_self_knn(
            malformed, similarities, rows
        )


@pytest.mark.parametrize(
    ("treatment", "outcome", "sufficient"),
    [
        (
            REGISTERED_DENSITY_FLOOR - 0.001,
            OUTCOME_SUFFICIENT,
            True,
        ),
        (
            REGISTERED_DENSITY_FLOOR,
            OUTCOME_NOT_SUFFICIENT,
            False,
        ),
    ],
)
def test_degree_selector_is_density_only(
    treatment: float,
    outcome: str,
    sufficient: bool,
) -> None:
    result = classify_degree_bridge(
        localization_outcome="bundled-2m-to-25m-transition-localized",
        control_density=REGISTERED_DENSITY_FLOOR + 0.02,
        treatment_density=treatment,
        registered_floor=REGISTERED_DENSITY_FLOOR,
    )
    assert result["outcome"] == outcome
    assert result["k15_alone_sufficient"] is sufficient
    assert result["core_and_ood_diagnostics_can_rescue_or_fail"] is False


def test_decision_diagnostics_cannot_rescue_density(
    tmp_path: Path,
) -> None:
    density_root = tmp_path / "density"
    diagnostic_root = tmp_path / "diagnostic"
    density_root.mkdir()
    diagnostic_root.mkdir()
    from basemap.round0113_prompt_contrast import seal

    density = seal(
        {
            "schema": DENSITY_SCHEMA,
            "round_id": "0121",
            "r0119_outcome": "bundled-2m-to-25m-transition-localized",
            "scorer": {"registered_floor": REGISTERED_DENSITY_FLOOR},
            "control_reuse": {
                "cell": {
                    "density_v2": {
                        "correlation": REGISTERED_DENSITY_FLOOR + 0.01
                    }
                }
            },
            "treatment": {
                "density_v2": {
                    "correlation": REGISTERED_DENSITY_FLOOR - 0.01
                }
            },
        }
    )
    diagnostic = seal(
        {
            "schema": DIAGNOSTIC_SCHEMA,
            "round_id": "0121",
            "arm": "raw",
            "execution_gates": {
                "finite_noncollapsed_coordinates": True,
                "transductive_recall50_gt_recall10": True,
                "matched_projection_recall50_gt_recall10": True,
                "exact_update_closure": True,
                "zero_numerical_skips": True,
                "no_pipeline_stamp_drift": True,
            },
            # Deliberately excellent diagnostics; they cannot alter density.
            "metrics": {"density": 1.0, "ffr": 1.0},
            "ood": {"pol_Latn": {"matched": 1.0}},
        }
    )
    (density_root / "density-score.json").write_text(
        json.dumps(density), encoding="utf-8"
    )
    (diagnostic_root / "score.json").write_text(
        json.dumps(diagnostic), encoding="utf-8"
    )
    result = round0121_nodes.run_decision(
        {"manifest": {"round_id": "0121", "release_sha": "b" * 40}},
        {
            "outputs": [str(tmp_path / "decision")],
            "density_output": str(density_root),
            "diagnostic_output": str(diagnostic_root),
        },
    )
    assert result["registered_selector"]["outcome"] == OUTCOME_SUFFICIENT
    assert result["diagnostics_can_rescue_or_fail_selector"] is False
    read_sealed(
        str(tmp_path / "decision" / "decision.json"),
        label="R0121 smoke decision",
    )


def test_decision_rejects_missing_execution_gates(tmp_path: Path) -> None:
    from basemap.round0113_prompt_contrast import seal

    density_root = tmp_path / "density"
    diagnostic_root = tmp_path / "diagnostic"
    density_root.mkdir()
    diagnostic_root.mkdir()
    (density_root / "density-score.json").write_text(
        json.dumps(
            seal(
                {
                    "schema": DENSITY_SCHEMA,
                    "round_id": "0121",
                    "r0119_outcome": (
                        "bundled-2m-to-25m-transition-localized"
                    ),
                    "scorer": {
                        "registered_floor": REGISTERED_DENSITY_FLOOR
                    },
                    "control_reuse": {
                        "cell": {
                            "density_v2": {
                                "correlation": (
                                    REGISTERED_DENSITY_FLOOR + 0.01
                                )
                            }
                        }
                    },
                    "treatment": {
                        "density_v2": {
                            "correlation": REGISTERED_DENSITY_FLOOR
                        }
                    },
                }
            )
        ),
        encoding="utf-8",
    )
    (diagnostic_root / "score.json").write_text(
        json.dumps(
            seal(
                {
                    "schema": DIAGNOSTIC_SCHEMA,
                    "round_id": "0121",
                    "arm": "raw",
                    "execution_gates": {},
                }
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        round0121_nodes.Round0121Error,
        match="decision evidence changed",
    ):
        round0121_nodes.run_decision(
            {
                "manifest": {
                    "round_id": "0121",
                    "release_sha": "b" * 40,
                }
            },
            {
                "outputs": [str(tmp_path / "decision")],
                "density_output": str(density_root),
                "diagnostic_output": str(diagnostic_root),
            },
        )


def test_train_seal_reload_panel_cpu_smoke(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Catch post-training receipt/panel crashes without allocating CUDA."""
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    import torch

    updates = 201
    batch_size = 16
    positive_rows = 4
    expected_rows = updates * batch_size
    graph_signature = _signature(str(tmp_path / "graph.npz"))
    manifest_signature = _signature(str(tmp_path / "graph-manifest.json"))
    assembly_signature = _signature(str(tmp_path / "assembly.json"))
    (tmp_path / "graph-manifest.json").write_text("{}", encoding="utf-8")
    graph = {
        "signature": graph_signature,
        "manifest_signature": manifest_signature,
        "manifest": {},
        "sources": np.arange(64, dtype=np.int32),
        "targets": np.roll(np.arange(64, dtype=np.int32), -1),
        "weights": np.ones(64, dtype=np.float32),
        "n_nodes": 240,
    }
    expected_stamp = {
        "schema": "round0121-cpu-smoke-v1",
        "pipeline": "host_weighted_jina_prompt_contrast",
        "sampler_class": "PromptWeightedJinaSampler",
    }
    config = {
        "model": {
            "architecture": "residual_bottleneck",
            "input_dimension": 8,
            "hidden_dimension": 16,
            "hidden_layers": 1,
            "output_dimension": 2,
            "use_batchnorm": False,
            "use_dropout": False,
            "low_dim_kernel": "legacy_lp",
            "a": 1.0,
            "b": 1.0,
        },
        "execution": {
            "expected_pipeline_stamp": expected_stamp,
            "minimum_train_upd_s": 1.0,
            "warning_train_upd_s": 1.0,
        },
    }
    accepted = updates * positive_rows + 3
    runtime = {
        **expected_stamp,
        "positive_sampling": (
            "fuzzy_weight_proportional_with_replacement_via_exact_"
            "uniform_envelope_rejection"
        ),
        "multiplicity_policy": (
            "shared-source-raw-document-union-representative-only"
        ),
        "feature_residency": "host-contiguous-compact-fp16-memmap",
        "source_representation": "raw-fp16",
        "device_conversion": "device-fp32-from-exact-fp16",
        "endpoint_gather_calls": updates,
        "source_rows_gathered": expected_rows,
        "destination_rows_gathered": expected_rows,
        "host_prefetch_batches_filled": updates,
        "host_prefetch_producer_batches": updates,
        "host_prefetch_consumer_batches": updates,
        "host_prefetch_source_rows_filled": expected_rows,
        "host_prefetch_destination_rows_filled": expected_rows,
        "weight_proposals": accepted + 1_000,
        "weight_acceptances": accepted,
        "weight_emitted_draws": updates * positive_rows,
        "weight_buffered_draws": 3,
        "weight_acceptance_rate": accepted / (accepted + 1_000),
        "weight_rejection_iterations": updates,
    }
    accounting: dict[str, Any] = {
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
        "n_pos_edges": len(graph["sources"]),
    }
    accounting.update(
        {f"pipeline_{key}": runtime[key] for key in _DYNAMIC_PIPELINE_COUNTERS}
    )

    class SmokeDataset:
        def __init__(self, *args, **kwargs) -> None:
            self.features = np.random.default_rng(121).normal(
                size=(240, 8)
            ).astype(np.float32)

    class SmokeWrapper:
        def __init__(self, dataset, graph) -> None:
            self.dataset = dataset

        def runtime_stamp(self) -> dict[str, Any]:
            return dict(runtime)

    class SmokeProfiler:
        def finalize(self, **kwargs) -> dict[str, Any]:
            return {"aborted": False, "cpu_smoke": True}

    class SmokeModel:
        def __init__(self) -> None:
            self.layer = torch.nn.Linear(8, 2)
            self._canary_profiler = SmokeProfiler()
            self._bench_seconds = 0.01
            self._setup_seconds = 0.001

        def fit(self, wrapper, **kwargs) -> None:
            X = torch.from_numpy(wrapper.dataset.features)
            target = X[:, :2] - 0.2 * X[:, 2:4]
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.05)
            for _ in range(4):
                optimizer.zero_grad()
                loss = torch.nn.functional.mse_loss(self.layer(X), target)
                loss.backward()
                optimizer.step()
            self._train_stats = dict(accounting)

        def save(self, path: str) -> None:
            torch.save(
                {
                    "weight": self.layer.weight.detach(),
                    "bias": self.layer.bias.detach(),
                },
                path,
            )

    assembly = {"outputs": {"raw": _signature("/smoke/raw")}, "mapping": {}}
    monkeypatch.setattr(
        round0121_nodes.prompt_nodes,
        "_load_assembly",
        lambda job: (assembly, assembly_signature),
    )
    monkeypatch.setattr(
        round0121_nodes, "load_graph", lambda *args, **kwargs: graph
    )
    monkeypatch.setattr(
        round0121_nodes,
        "train_config",
        lambda **kwargs: (config, "c" * 64),
    )
    monkeypatch.setattr(
        round0121_nodes.prompt_nodes,
        "_open_compact",
        lambda assembly, arm: np.empty((240, 8), dtype=np.float16),
    )
    monkeypatch.setattr(round0121_nodes, "HostFp16EndpointArray", SmokeDataset)
    monkeypatch.setattr(
        round0121_nodes, "DegreeBridgeTrainingInput", SmokeWrapper
    )
    monkeypatch.setattr(
        round0121_nodes, "_new_model", lambda config: SmokeModel()
    )
    monkeypatch.setattr(round0121_nodes, "SUCCESSFUL_UPDATES", updates)
    monkeypatch.setattr(round0121_nodes, "BATCH_SIZE", batch_size)
    monkeypatch.setattr(
        round0121_nodes, "PERFORMANCE_WARMUP_UPDATES", 1
    )
    monkeypatch.setattr(
        round0121_nodes.prompt_nodes,
        "_weighted_rejection_accounting_mismatch",
        lambda runtime, producer_delta: None,
    )
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda device: (1_000_000_000, 2_000_000_000),
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 1_024)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 2_048)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    output = tmp_path / "train"
    result = round0121_nodes.run_train(
        {"manifest": {"round_id": "0121", "release_sha": "d" * 40}},
        {
            "outputs": [str(output)],
            "graph_manifest": str(tmp_path / "graph-manifest.json"),
        },
    )
    assert result["optimizer_updates"] == updates
    receipt = read_sealed(
        str(output / "train-receipt.json"), label="R0121 CPU smoke train"
    )
    assert receipt["train_checks"]["weighted_rejection_accounting_closes"]

    checkpoint = torch.load(
        output / "model.pt", map_location="cpu", weights_only=True
    )
    features = SmokeDataset().features
    coordinates = torch.nn.functional.linear(
        torch.from_numpy(features),
        checkpoint["weight"],
        checkpoint["bias"],
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
        provenance={"round": "0121", "mode": "cpu-smoke"},
    )
    assert panel["guards"]["coords_finite"] is True
    assert panel["guards"]["coords_collapsed"] is False

    # Exercise the actual post-fit authentication and R0115 diagnostic
    # adapter. Only model compute and corpus size are reduced.
    from basemap.panel_v2 import build_hiD_reference
    from basemap.pumap.parametric_umap import ParametricUMAP

    class LoadedSmokeModel:
        architecture = "residual_bottleneck"
        input_dim = 8
        hidden_dim = 16
        n_layers = 1
        n_components = 2
        use_batchnorm = False
        use_dropout = False
        low_dim_kernel = "legacy_lp"
        a = 1.0
        b = 1.0

        def __init__(self, checkpoint: Mapping[str, Any]) -> None:
            self.weight = checkpoint["weight"].numpy()
            self.bias = checkpoint["bias"].numpy()

        def transform(self, values, batch_size: int) -> np.ndarray:
            data = np.asarray(values, dtype=np.float32)
            return data @ self.weight.T + self.bias

    monkeypatch.setattr(
        ParametricUMAP,
        "load",
        staticmethod(
            lambda path, device: LoadedSmokeModel(
                torch.load(path, map_location="cpu", weights_only=True)
            )
        ),
    )
    auth_job = {
        "graph_manifest": str(tmp_path / "graph-manifest.json"),
        "train_output": str(output),
    }
    loaded, authenticated, _assembly, _graph = (
        round0121_nodes._authenticate_treatment_model(
            {"manifest": {"round_id": "0121", "release_sha": "d" * 40}},
            auth_job,
        )
    )
    assert authenticated["optimizer_updates"] == updates
    assert loaded.input_dim == 8

    prompt = round0121_nodes.prompt_nodes
    monkeypatch.setattr(prompt, "RETAINED_ROWS", 240)
    monkeypatch.setattr(prompt, "DIMENSION", 8)
    monkeypatch.setattr(prompt, "QUERY_CANDIDATES", 30)
    monkeypatch.setattr(prompt, "QUERY_ROWS", 20)
    monkeypatch.setattr(prompt, "POLISH_QUERY_ROWS", 10)
    panel_config = PanelV2Config(
        frac=0.1,
        k_clust=(),
        k_hit=10,
        k_density=3,
        n_anchors=24,
        anchor_seed=123,
        corpus_chunk=64,
        overselect=4,
        block_elems=100_000,
        rerank_byte_cap=8_000_000,
        peak_byte_cap=16_000_000,
    )
    monkeypatch.setattr(prompt, "panel_config", lambda: panel_config)
    data_identity = {
        "kind": "ordered_array",
        "shape": [240, 8],
        "dtype": np.dtype("<f4").str,
        "sha256": ordered_array_sha256(features),
    }
    monkeypatch.setattr(
        prompt, "_data_identity", lambda assembly, arm: data_identity
    )
    convention = {
        "row_order": (
            "R0113 shared source/raw/document union-representative compact "
            "order"
        ),
        "distance": "cosine via fp32-L2-normalized squared L2",
        "self_exclusion": True,
        "anchor_namespace": "R0113 compact IDs",
        "embedding_prompt": "raw",
    }
    normalized = features / np.linalg.norm(features, axis=1, keepdims=True)
    reference = build_hiD_reference(
        normalized,
        np.sort(
            np.random.RandomState(123)
            .choice(240, 24, replace=False)
            .astype(np.int64)
        ),
        panel_config,
        centroids_by_k=None,
        data_identity=data_identity,
        convention=convention,
    )
    graph["manifest"] = {
        "high_d_reference": _signature("/smoke/high-d-reference.npz"),
        "high_d_reference_key": reference["key"],
    }
    import basemap.panel_v2 as panel_v2

    monkeypatch.setattr(
        panel_v2,
        "load_hiD_reference",
        lambda path, expected_key: reference,
    )
    query_rng = np.random.default_rng(1210)
    query_signatures: dict[str, Any] = {}
    polish_signatures: dict[str, Any] = {}
    for arm in ("raw", "document"):
        query_path = tmp_path / f"{arm}-queries.npy"
        polish_path = tmp_path / f"{arm}-polish.npy"
        np.save(
            query_path,
            query_rng.normal(size=(30, 8)).astype(np.float16),
        )
        np.save(
            polish_path,
            query_rng.normal(size=(10, 8)).astype(np.float16),
        )
        query_signatures[arm] = expected_input_signature(str(query_path))
        polish_signatures[arm] = expected_input_signature(str(polish_path))
    polish_rows_path = tmp_path / "polish-rows.npy"
    global_rows_path = tmp_path / "query-global-rows.npy"
    np.save(polish_rows_path, np.arange(10, dtype=np.int64))
    np.save(global_rows_path, np.arange(20, dtype=np.int64))
    query_receipt = {
        "outputs": query_signatures,
        "ood": {
            "pol_Latn": {
                "outputs": polish_signatures,
                "query_rows": expected_input_signature(str(polish_rows_path)),
                "source_text": {"name": "cpu-smoke-polish"},
            }
        },
    }
    query_receipt_signature = _signature("/smoke/query-receipt.json")
    selection = {
        "identity_sha256": "e" * 64,
        "global_rows": expected_input_signature(str(global_rows_path)),
    }
    selection_root = tmp_path / "selection"
    selection_root.mkdir()
    (selection_root / "query-selection.json").write_text(
        json.dumps(selection), encoding="utf-8"
    )
    monkeypatch.setattr(
        prompt,
        "_load_query_reserve",
        lambda job: (query_receipt, query_receipt_signature),
    )
    monkeypatch.setattr(
        prompt,
        "_load_query_selection",
        lambda job: (selection, np.arange(20, dtype=np.int64)),
    )
    monkeypatch.setattr(
        prompt,
        "_open_compact",
        lambda assembly, arm: features,
    )
    diagnostic_root = tmp_path / "diagnostics"
    diagnostic = round0121_nodes.run_diagnostics(
        {"manifest": {"round_id": "0121", "release_sha": "d" * 40}},
        {
            **auth_job,
                "outputs": [str(diagnostic_root)],
                "arm": "raw",
                "query_output": "/smoke/query",
                "query_selection_output": str(selection_root),
        },
    )
    assert diagnostic["schema"] == DIAGNOSTIC_SCHEMA
    assert all(diagnostic["execution_gates"].values())
    read_sealed(
        str(diagnostic_root / "score.json"),
        label="R0121 CPU smoke diagnostics",
    )

    # Exercise the actual R0119 matched-density scoring adapter too. The
    # predecessor receipt is synthetic; the frozen scorer implementation is
    # not.
    monkeypatch.setattr(
        round0121_nodes.localization_nodes, "REPRESENTATIVE_ROWS", 240
    )
    control_density = REGISTERED_DENSITY_FLOOR + 0.01
    predecessor_panel = {
        "universe": {"representative_rows": 240},
        "scorer": {
            "registered_floor": REGISTERED_DENSITY_FLOOR,
            "k": 15,
        },
        "cells": {
            "current_2m_seed42": {
                "density_v2": {"correlation": control_density},
                "clears_unchanged_registered_floor": True,
            }
        },
    }
    monkeypatch.setattr(
        round0121_nodes,
        "_r0119_evidence",
        lambda job: (
            predecessor_panel,
            _signature("/smoke/r0119-panel.json"),
            {"outcome": "bundled-2m-to-25m-transition-localized"},
            _signature("/smoke/r0119-decision.json"),
        ),
    )
    anchors = np.arange(128, dtype=np.int64)
    high_radius = np.linspace(0.5, 2.0, 128, dtype=np.float64)
    monkeypatch.setattr(
        round0121_nodes.localization_nodes,
        "_load_universe",
        lambda job: (
            features,
            features,
            np.arange(240, dtype=np.int64),
            anchors,
            anchors,
            high_radius,
            {"registered_floor": REGISTERED_DENSITY_FLOOR},
            {},
        ),
    )
    density_root = tmp_path / "density-adapter"
    density = round0121_nodes.run_density(
        {"manifest": {"round_id": "0121", "release_sha": "d" * 40}},
        {
            **auth_job,
            "outputs": [str(density_root)],
        },
    )
    assert density["schema"] == DENSITY_SCHEMA
    assert density["control_reuse"]["score_recomputed_in_r0121"] is False
    assert np.isfinite(density["treatment"]["density_v2"]["correlation"])
    read_sealed(
        str(density_root / "density-score.json"),
        label="R0121 CPU smoke density",
    )
