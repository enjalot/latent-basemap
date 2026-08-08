"""Bounded CPU smoke for R0223's train -> seal -> publish -> reload -> panel path.

R0221's smoke shape, with the graph swapped. Only the GPU kernel, the endpoint
array and the sampler are stubbed; the ceil-derived dose, the R0217-template
config construction with the graph moved, the treatment-invariant digest
equality, the post-fit accounting, the checkpoint publish, the full-population
reload and the receipt seal are the real code.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

import basemap.pumap.parametric_umap as pumap
from basemap import round0113_prompt_contrast as prompt_contract
from basemap.round0221_minilm_2m_seed_extension import (
    SEALED_GRAPH_MANIFEST_SIGNATURE,
    SEALED_GRAPH_SIGNATURE,
    SEALED_SUBSTRATE_SIGNATURE,
)
from basemap.round0223_cuvs_graph_map import (
    BATCH_SIZE,
    CUVS_GRAPH_CAPABILITY,
    DIMENSION,
    POSITIVE_ROWS_PER_UPDATE,
    ROUND_ID,
    ROWS,
    Round0223Error,
    SEEDS,
    map_capability,
    successful_updates_for_edges,
    train_config,
)
from experiments import round0223_nodes


CUVS_EDGES = 48_100_000
CUVS_GRAPH_SIGNATURE: dict[str, Any] = {
    "kind": "file",
    "canonical_path": "/data/latent-basemap/runs/round-0223/queue/x/edges-k15-fuzzy.npz",
    "bytes": 123,
    "sha256": "a" * 64,
}
CUVS_MANIFEST_SIGNATURE: dict[str, Any] = {
    "kind": "file",
    "canonical_path": "/data/latent-basemap/runs/round-0223/queue/x/cuvs-graph.json",
    "bytes": 456,
    "sha256": "b" * 64,
}


def _smoke(monkeypatch: pytest.MonkeyPatch, tmp_path, *, seed: int) -> dict[str, Any]:
    import torch

    updates = successful_updates_for_edges(CUVS_EDGES)
    substrate_signature = dict(SEALED_SUBSTRATE_SIGNATURE)
    config, config_sha, invariant = train_config(
        seed=seed,
        graph_signature=dict(CUVS_GRAPH_SIGNATURE),
        graph_manifest_signature=dict(CUVS_MANIFEST_SIGNATURE),
        substrate_signature=substrate_signature,
        r0216_graph_signature=dict(SEALED_GRAPH_SIGNATURE),
        r0216_graph_manifest_signature=dict(SEALED_GRAPH_MANIFEST_SIGNATURE),
        graph_edges=CUVS_EDGES,
        rows=ROWS,
    )

    producer_batches = updates + 1
    expected_rows = updates * BATCH_SIZE
    emitted = producer_batches * POSITIVE_ROWS_PER_UPDATE
    accepted = emitted + 7
    proposals = accepted + 1_000
    runtime = {
        **config["execution"]["expected_pipeline_stamp"],
        "endpoint_gather_calls": updates,
        "source_rows_gathered": expected_rows,
        "destination_rows_gathered": expected_rows,
        "host_prefetch_batches_filled": producer_batches,
        "host_prefetch_producer_batches": producer_batches,
        "host_prefetch_consumer_batches": updates,
        "host_prefetch_source_rows_filled": producer_batches * BATCH_SIZE,
        "host_prefetch_destination_rows_filled": producer_batches * BATCH_SIZE,
        "weight_proposals": proposals,
        "weight_acceptances": accepted,
        "weight_emitted_draws": emitted,
        "weight_buffered_draws": accepted - emitted,
        "weight_acceptance_rate": accepted / proposals,
        "weight_rejection_iterations": producer_batches,
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
        "n_pos_edges": CUVS_EDGES,
    }
    accounting.update({
        f"pipeline_{key}": runtime[key]
        for key in prompt_contract._DYNAMIC_PIPELINE_COUNTERS
    })
    graph = {
        "manifest": {
            "substrate": substrate_signature,
            "builder": {
                "setting_id": "nnd-gd32-igd48-it20",
                "intermediate_graph_degree": 48,
            },
            "recall_against_r0220_exact_truth": {
                "cross_check": {"reproduces_r0220": True},
                "tie_aware": {"mean": 0.9941635666666666},
            },
        },
        "manifest_signature": dict(CUVS_MANIFEST_SIGNATURE),
        "signature": dict(CUVS_GRAPH_SIGNATURE),
        "sources": np.arange(8, dtype=np.int32),
        "targets": np.roll(np.arange(8, dtype=np.int32), -1),
        "weights": np.ones(8, dtype=np.float32),
        "n_nodes": ROWS,
        "directed_edges": CUVS_EDGES,
    }
    source = np.random.default_rng(223).normal(size=(64, DIMENSION)).astype(np.float32)
    full_coordinates = np.tile(
        np.asarray([[2.0, -3.0], [-1.0, 5.0]], dtype=np.float32), (ROWS // 2, 1)
    )

    class SmokeDataset:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.shape = (ROWS, DIMENSION)

    class SmokeWrapper:
        def __init__(self, dataset: Any, *args: Any, **kwargs: Any) -> None:
            self.dataset = dataset

        def runtime_stamp(self) -> dict[str, Any]:
            return dict(runtime)

    class SmokeProfiler:
        def finalize(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["bench_seconds"] > 0
            return {"aborted": False, "smoke": True}

    class SmokeModel:
        def __init__(self) -> None:
            self.layer = torch.nn.Linear(DIMENSION, 2)
            self._canary_profiler = SmokeProfiler()
            self._bench_seconds = 1.0
            self._setup_seconds = 0.001

        def fit(self, wrapper: Any, **kwargs: Any) -> None:
            assert kwargs["random_state"] == seed
            assert (
                kwargs["precomputed_edges_path"]
                == CUVS_GRAPH_SIGNATURE["canonical_path"]
            )
            self._train_stats = dict(accounting)

        def save(self, path: str) -> None:
            torch.save({"seed": seed}, path)

    class SmokeReloaded:
        def transform(self, X: Any, batch_size: int = 8192) -> np.ndarray:
            return full_coordinates

    class SmokeParametricUMAP:
        @classmethod
        def load(cls, path: str, device: str | None = None) -> SmokeReloaded:
            torch.load(path, map_location="cpu", weights_only=True)
            return SmokeReloaded()

    import basemap.round0217_minilm_2m_pipeline as pipeline_module

    monkeypatch.setattr(round0223_nodes, "_sealed_cuvs_graph", lambda job: graph)
    monkeypatch.setattr(
        round0223_nodes, "_open_substrate", lambda graph: (source, substrate_signature)
    )
    monkeypatch.setattr(
        pipeline_module, "MiniLMHostFp32EndpointArray", SmokeDataset
    )
    monkeypatch.setattr(pipeline_module, "MiniLMMixedTrainingInput", SmokeWrapper)
    monkeypatch.setattr(
        round0223_nodes.prompt_nodes, "_new_model", lambda config: SmokeModel()
    )
    monkeypatch.setattr(pumap, "ParametricUMAP", SmokeParametricUMAP)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda value: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda device: (1_000_000_000, 2_000_000_000)
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device=None: 1_024)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device=None: 2_048)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    output = tmp_path / f"train-output-seed{seed}"
    job = {
        "action": round0223_nodes.TRAIN_ACTION,
        "training_seed": seed,
        "capability": map_capability(seed),
        "cuvs_graph_manifest_signature": dict(CUVS_MANIFEST_SIGNATURE),
        "r0216_graph_signature": dict(SEALED_GRAPH_SIGNATURE),
        "r0216_graph_manifest_signature": dict(SEALED_GRAPH_MANIFEST_SIGNATURE),
        "treatment_invariant_sha256": invariant,
        "registered_dose_bound": 120_000,
        "outputs": [str(output)],
    }
    round0223_nodes.run_job(
        {"manifest": {"round_id": ROUND_ID, "release_sha": "f" * 40}}, job
    )
    with (output / "train-receipt.json").open(encoding="utf-8") as handle:
        receipt = json.load(handle)
    prompt_contract.validate_seal(receipt, label="R0223 CPU smoke receipt")
    with (output / "production-config.json").open(encoding="utf-8") as handle:
        published_config = json.load(handle)
    assert published_config["config_sha256"] == config_sha
    assert published_config["treatment_invariant_sha256"] == invariant
    assert published_config["treatment_config_round_id"] == "0217"
    return receipt


def test_round0223_train_seal_publish_reload_cpu_smoke(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    receipt = _smoke(monkeypatch, tmp_path, seed=42)
    assert receipt["round_id"] == ROUND_ID
    assert receipt["training_seed"] == 42
    assert receipt["treatment_config_round_id"] == "0217"
    assert receipt["capability"] == map_capability(42)
    assert receipt["graph_capability"] == CUVS_GRAPH_CAPABILITY
    assert receipt["directed_edges"] == CUVS_EDGES
    assert receipt["optimizer_updates"] == successful_updates_for_edges(CUVS_EDGES)
    assert receipt["gate_registerable_here"] is False
    assert receipt["map_decision_made"] is False
    assert receipt["train_accounting"]["pipeline_runtime"] == receipt[
        "exact_execution_receipt"
    ]
    checks = receipt["train_checks"]
    assert checks["treatment_identical_to_r0217_except_seed_and_graph"] is True
    assert checks["all_2m_coordinates_finite"] is True
    published = receipt["published_map_check"]
    assert published["transform_rows"] == ROWS
    assert published["transform_rows_finite"] == ROWS
    assert published["collapsed"] is False


def test_round0223_every_seed_runs_the_same_treatment(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    invariants = set()
    for seed in SEEDS:
        receipt = _smoke(monkeypatch, tmp_path, seed=seed)
        assert receipt["training_seed"] == seed
        invariants.add(receipt["treatment_invariant_sha256"])
    assert len(invariants) == 1


def test_round0223_rejects_foreign_queues_actions_and_seeds() -> None:
    with pytest.raises(Round0223Error):
        round0223_nodes.run_job(
            {"manifest": {"round_id": ROUND_ID}}, {"action": "something_else"}
        )
    with pytest.raises(Round0223Error):
        round0223_nodes.run_job(
            {"manifest": {"round_id": "0217"}},
            {"action": round0223_nodes.TRAIN_ACTION},
        )
    for seed in (45, 48, 99):
        with pytest.raises(Round0223Error):
            round0223_nodes._seed(
                {"training_seed": seed, "capability": f"x-{seed}"}
            )


def test_round0223_rejects_a_drifted_treatment_invariant(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import torch

    graph = {
        "manifest": {"substrate": dict(SEALED_SUBSTRATE_SIGNATURE)},
        "manifest_signature": dict(CUVS_MANIFEST_SIGNATURE),
        "signature": dict(CUVS_GRAPH_SIGNATURE),
        "n_nodes": ROWS,
        "directed_edges": CUVS_EDGES,
    }
    monkeypatch.setattr(round0223_nodes, "_sealed_cuvs_graph", lambda job: graph)
    monkeypatch.setattr(
        round0223_nodes,
        "_open_substrate",
        lambda graph: (
            np.zeros((4, DIMENSION), dtype=np.float32),
            dict(SEALED_SUBSTRATE_SIGNATURE),
        ),
    )
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda value: None)
    with pytest.raises(Round0223Error):
        round0223_nodes.run_train(
            {"manifest": {"round_id": ROUND_ID, "release_sha": "f" * 40}},
            {
                "action": round0223_nodes.TRAIN_ACTION,
                "training_seed": 43,
                "capability": map_capability(43),
                "cuvs_graph_manifest_signature": dict(CUVS_MANIFEST_SIGNATURE),
                "r0216_graph_signature": dict(SEALED_GRAPH_SIGNATURE),
                "r0216_graph_manifest_signature": dict(SEALED_GRAPH_MANIFEST_SIGNATURE),
                "treatment_invariant_sha256": "0" * 64,
                "outputs": [str(tmp_path / "never-created")],
            },
        )
