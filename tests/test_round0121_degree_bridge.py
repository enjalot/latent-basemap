"""Contract and bounded CPU smoke tests for R0121."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from basemap.artifact_identity import expected_input_signature
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
                "finite": True,
                "accounting": True,
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
