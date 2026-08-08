"""Bounded CPU smoke for the R0217 train -> seal -> publish -> reload path.

Reaches the real post-fit accounting, the real dose assertion, the real config
seal, the real checkpoint publish, the real reload-and-collapse check and the
real receipt seal. Only the GPU kernel, the endpoint array and the sampler are
stubbed. Its job is to catch late NameErrors, accounting-shape drift and
serialization failures in milliseconds instead of after four GPU nodes.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

import basemap.pumap.parametric_umap as pumap
from basemap import round0113_prompt_contrast as prompt_contract
from basemap.round0217_minilm_2m_seed_family import (
    BATCH_SIZE,
    DIMENSION,
    POSITIVE_ROWS_PER_UPDATE,
    ROUND_ID,
    ROWS,
    Round0217Error,
    SEALED_DIRECTED_EDGES,
    SEEDS,
    capability_for_seed,
    seed_invariant_sha256,
    successful_updates_for_edges,
    train_config,
)
from experiments import round0217_nodes


PROBE_ROWS = 96


def _signature(path: str, digest: str) -> dict[str, Any]:
    return {"kind": "file", "canonical_path": path, "bytes": 1_024, "sha256": digest * 64}


def _smoke(monkeypatch: pytest.MonkeyPatch, tmp_path, *, seed: int) -> dict[str, Any]:
    import torch

    updates = successful_updates_for_edges(SEALED_DIRECTED_EDGES)
    graph_signature = _signature(str(tmp_path / "edges-k15-fuzzy.npz"), "a")
    manifest_signature = _signature(str(tmp_path / "substrate-graph.json"), "b")
    substrate_signature = _signature(str(tmp_path / "substrate.f32.npy"), "c")
    config, config_sha = train_config(
        seed=seed,
        graph_signature=graph_signature,
        graph_manifest_signature=manifest_signature,
        substrate_signature=substrate_signature,
        graph_edges=SEALED_DIRECTED_EDGES,
        rows=ROWS,
    )
    invariant = seed_invariant_sha256(config)

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
        "n_pos_edges": SEALED_DIRECTED_EDGES,
    }
    accounting.update({
        f"pipeline_{key}": runtime[key]
        for key in prompt_contract._DYNAMIC_PIPELINE_COUNTERS
    })
    graph = {
        "manifest": {"substrate": substrate_signature},
        "manifest_signature": manifest_signature,
        "signature": graph_signature,
        "sources": np.arange(8, dtype=np.int32),
        "targets": np.roll(np.arange(8, dtype=np.int32), -1),
        "weights": np.ones(8, dtype=np.float32),
        "n_nodes": ROWS,
        "directed_edges": SEALED_DIRECTED_EDGES,
    }
    source = np.random.default_rng(217).normal(
        size=(PROBE_ROWS, DIMENSION)
    ).astype(np.float32)

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
            assert kwargs["precomputed_edges_path"] == graph_signature["canonical_path"]
            self._train_stats = dict(accounting)

        def save(self, path: str) -> None:
            torch.save(
                {"weight": self.layer.weight.detach(), "bias": self.layer.bias.detach()},
                path,
            )

    class SmokeReloaded:
        def __init__(self, checkpoint: dict[str, Any]) -> None:
            self.checkpoint = checkpoint

        def transform(self, X: Any, batch_size: int = 4096) -> np.ndarray:
            values = torch.from_numpy(np.asarray(X, dtype=np.float32))
            return torch.nn.functional.linear(
                values, self.checkpoint["weight"], self.checkpoint["bias"]
            ).detach().numpy()

    class SmokeParametricUMAP:
        @classmethod
        def load(cls, path: str, device: str | None = None) -> SmokeReloaded:
            return SmokeReloaded(
                torch.load(path, map_location="cpu", weights_only=True)
            )

    monkeypatch.setattr(round0217_nodes, "_sealed_graph", lambda job: graph)
    monkeypatch.setattr(
        round0217_nodes, "_open_substrate", lambda graph: (source, substrate_signature)
    )
    monkeypatch.setattr(round0217_nodes, "MiniLMHostFp32EndpointArray", SmokeDataset)
    monkeypatch.setattr(round0217_nodes, "MiniLMMixedTrainingInput", SmokeWrapper)
    monkeypatch.setattr(
        round0217_nodes.prompt_nodes, "_new_model", lambda config: SmokeModel()
    )
    monkeypatch.setattr(
        round0217_nodes, "_probe_rows", lambda: np.arange(PROBE_ROWS, dtype=np.int64)
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
        "action": round0217_nodes.ACTION,
        "training_seed": seed,
        "capability": capability_for_seed(seed),
        "graph_manifest_signature": manifest_signature,
        "family_seed_invariant_sha256": invariant,
        "registered_dose_bound": 120_000,
        "outputs": [str(output)],
    }
    round0217_nodes.run_job(
        {"manifest": {"round_id": ROUND_ID, "release_sha": "f" * 40}}, job
    )
    with (output / "train-receipt.json").open(encoding="utf-8") as handle:
        receipt = json.load(handle)
    prompt_contract.validate_seal(receipt, label="R0217 CPU smoke receipt")
    with (output / "production-config.json").open(encoding="utf-8") as handle:
        published_config = json.load(handle)
    assert published_config["config_sha256"] == config_sha
    assert published_config["seed_invariant_sha256"] == invariant
    return receipt


def test_round0217_train_seal_publish_reload_cpu_smoke(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    updates = successful_updates_for_edges(SEALED_DIRECTED_EDGES)
    receipt = _smoke(monkeypatch, tmp_path, seed=42)
    assert receipt["training_seed"] == 42
    assert receipt["capability"] == capability_for_seed(42)
    assert receipt["optimizer_updates"] == updates
    assert receipt["directed_edges"] == SEALED_DIRECTED_EDGES
    assert receipt["dose_registration"]["successful_updates"] == updates
    assert receipt["gate_registerable_here"] is False
    assert receipt["train_accounting"]["pipeline_runtime"] == receipt[
        "exact_execution_receipt"
    ]
    assert receipt["train_checks"]["weighted_rejection_accounting_closes"] is True
    assert receipt["train_checks"][
        "published_checkpoint_reloads_finite_and_uncollapsed"
    ] is True
    published = receipt["published_map_check"]
    assert published["coordinates_finite"] is True
    assert published["collapsed"] is False
    assert published["probe_rows"] == PROBE_ROWS


def test_round0217_every_seed_runs_the_same_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    invariants = set()
    for seed in SEEDS:
        receipt = _smoke(monkeypatch, tmp_path, seed=seed)
        assert receipt["training_seed"] == seed
        invariants.add(receipt["seed_invariant_sha256"])
    assert len(invariants) == 1


def test_round0217_rejects_a_foreign_action_and_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    with pytest.raises(Round0217Error):
        round0217_nodes.run_job(
            {"manifest": {"round_id": ROUND_ID}}, {"action": "something_else"}
        )
    with pytest.raises(Round0217Error):
        round0217_nodes.run_train(
            {"manifest": {"round_id": "0216"}},
            {"action": round0217_nodes.ACTION, "training_seed": 42},
        )
    with pytest.raises(Round0217Error):
        round0217_nodes.run_train(
            {"manifest": {"round_id": ROUND_ID}},
            {
                "action": round0217_nodes.ACTION,
                "training_seed": 99,
                "capability": "minilm-mixed-2m-map-seed99-low-dose-v1",
            },
        )


def test_round0217_rejects_a_drifted_family_invariant(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    with pytest.raises(Round0217Error):
        _smoke_with_bad_invariant(monkeypatch, tmp_path)


def test_round0217_real_endpoint_array_and_sampler_run_on_cpu() -> None:
    """Drive the un-stubbed residency + sampler path once, on CPU.

    The smoke above replaces the endpoint array and the sampler, so nothing else
    would exercise the fp32 pinned-slot gather, the separate negative RNG stream,
    or the stamp the train node compares against `expected_pipeline_stamp`.
    """
    from basemap.round0217_minilm_2m_pipeline import (
        MiniLMHostFp32EndpointArray,
        MiniLMMixedWeightedSampler,
    )

    rows, seed = 512, 43
    rng = np.random.default_rng(0)
    source = rng.normal(size=(rows, DIMENSION)).astype(np.float32)
    source /= np.linalg.norm(source, axis=1, keepdims=True)
    edges = 4_096
    sources = rng.integers(0, rows, size=edges).astype(np.int32)
    targets = ((sources.astype(np.int64) + 1) % rows).astype(np.int32)
    weights = rng.uniform(0.05, 1.0, size=edges).astype(np.float32)

    dataset = MiniLMHostFp32EndpointArray(
        source,
        source_signature=_signature("/data/substrate.f32.npy", "c"),
        buffer_rows=64,
        device="cpu",
    )
    sampler = MiniLMMixedWeightedSampler(
        dataset,
        sources=sources,
        targets=targets,
        weights=weights,
        n_nodes=rows,
        batch_size=64,
        pos_ratio=0.05,
        random_state=seed,
        graph_signatures={"graph": {}, "manifest": {}},
    )
    try:
        iterator = iter(sampler)
        for _ in range(4):
            left, right, labels = next(iterator)
            assert left.shape == right.shape == (64, DIMENSION)
            assert labels.shape == (64,)
            assert bool(left.isfinite().all()) and bool(right.isfinite().all())
    finally:
        sampler.close()

    stamp = sampler.execution_stamp()
    reference = train_config(
        seed=seed,
        graph_signature=_signature("/data/edges.npz", "a"),
        graph_manifest_signature=_signature("/data/manifest.json", "b"),
        substrate_signature=_signature("/data/substrate.f32.npy", "c"),
        graph_edges=SEALED_DIRECTED_EDGES,
        rows=ROWS,
    )[0]["execution"]["expected_pipeline_stamp"]
    cardinality = {
        "valid_canonical_edge_count",
        "compact_retained_rows",
        "negative_sampling",
    }
    for key, value in reference.items():
        if key in cardinality:
            continue
        assert stamp[key] == value, key
    assert stamp["negative_rng_seed"] == reference["negative_rng_seed"]
    assert stamp["valid_canonical_edge_count"] == edges
    assert stamp["compact_retained_rows"] == rows
    assert 0 < stamp["weight_acceptance_rate"] <= 1
    assert stamp["source_rows_gathered"] == 4 * 64


def _smoke_with_bad_invariant(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import torch

    graph_signature = _signature(str(tmp_path / "edges.npz"), "a")
    manifest_signature = _signature(str(tmp_path / "manifest.json"), "b")
    substrate_signature = _signature(str(tmp_path / "substrate.npy"), "c")
    graph = {
        "manifest": {"substrate": substrate_signature},
        "manifest_signature": manifest_signature,
        "signature": graph_signature,
        "n_nodes": ROWS,
        "directed_edges": SEALED_DIRECTED_EDGES,
    }
    monkeypatch.setattr(round0217_nodes, "_sealed_graph", lambda job: graph)
    monkeypatch.setattr(
        round0217_nodes,
        "_open_substrate",
        lambda graph: (np.zeros((4, DIMENSION), dtype=np.float32), substrate_signature),
    )
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda value: None)
    round0217_nodes.run_train(
        {"manifest": {"round_id": ROUND_ID, "release_sha": "f" * 40}},
        {
            "action": round0217_nodes.ACTION,
            "training_seed": 43,
            "capability": capability_for_seed(43),
            "graph_manifest_signature": manifest_signature,
            "family_seed_invariant_sha256": "0" * 64,
            "outputs": [str(tmp_path / "never-created")],
        },
    )
