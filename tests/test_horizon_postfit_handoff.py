"""Regression smoke for an exact-horizon train's expensive post-fit handoff."""
from __future__ import annotations

import json

import numpy as np

from basemap.artifact_identity import canonical_json, sha256_bytes
from basemap.panel_v2 import PanelV2Config, score_panel
from basemap.pumap.parametric_umap import ParametricUMAP


def _seal(body: dict) -> dict:
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}


def test_equal_horizon_cap_reaches_train_seal_reload_panel(tmp_path) -> None:
    """Exercise the real fit and every post-fit operation used by round wrappers.

    Production model rounds set ``_max_train_steps`` equal to the cosine
    horizon, then consume ``_bench_seconds`` immediately after ``fit``.  A
    unit test of the benchmark-only stop cannot protect that equal-boundary
    handoff, which is where the late R0129 failure would otherwise occur.
    """
    rows = 128
    dimension = 8
    updates = 12
    warmup = 3
    rng = np.random.default_rng(12_900)
    features = rng.normal(size=(rows, dimension)).astype(np.float32)
    sources = rng.integers(0, rows, size=2_048, dtype=np.int32)
    targets = rng.integers(0, rows, size=2_048, dtype=np.int32)
    weights = rng.uniform(0.2, 1.0, size=2_048).astype(np.float32)
    graph_path = tmp_path / "tiny-graph.npz"
    np.savez(
        graph_path,
        sources=sources,
        targets=targets,
        weights=weights,
        n_nodes=np.asarray(rows),
        k=np.asarray(3),
    )

    model = ParametricUMAP(
        n_components=2,
        hidden_dim=16,
        n_layers=1,
        n_neighbors=3,
        a=1.0,
        b=1.0,
        correlation_weight=0.0,
        learning_rate=0.01,
        n_epochs=2,
        batch_size=32,
        device="cpu",
        pos_ratio=0.25,
        lr_schedule="cosine",
        warmup_steps=1,
        total_steps_estimate=updates,
        require_full_budget=True,
        require_graph_manifest=False,
        required_input_pipeline="legacy",
        use_amp=False,
        positive_target_mode="binary",
        weighted_edge_sampling=False,
        gpu_resident_data=False,
    )
    model._max_train_steps = updates
    model._bench_warmup = warmup
    model._perf_profile = True
    model._perf_floor = 0.0
    model._perf_n_windows = 3
    model.fit(
        features,
        low_memory=True,
        verbose=False,
        n_processes=1,
        random_state=42,
        resample_negatives=False,
        precomputed_edges_path=str(graph_path),
        use_wandb=False,
    )

    accounting = model._train_stats
    assert accounting["stop_reason"] == "lr_horizon"
    assert accounting["positive_lr_optimizer_steps"] == updates
    assert accounting["budget_satisfied"] is True
    assert model._bench_seconds is not None and model._bench_seconds > 0
    profile = model._canary_profiler.finalize(
        bench_seconds=model._bench_seconds,
        setup_seconds=model._setup_seconds,
    )
    rate = (updates - warmup) / model._bench_seconds
    assert profile["aborted"] is False
    assert np.isfinite(rate) and rate > 0

    model_path = tmp_path / "model.pt"
    model.save(str(model_path))
    receipt = _seal({
        "schema": "exact-horizon-postfit-smoke-v1",
        "train_accounting": accounting,
        "bench_seconds": model._bench_seconds,
        "steady_updates_per_s": rate,
        "model_sha256": sha256_bytes(model_path.read_bytes()),
    })
    receipt_path = tmp_path / "train-receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    loaded_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    identity = loaded_receipt.pop("identity_sha256")
    assert identity == sha256_bytes(canonical_json(loaded_receipt))

    loaded = ParametricUMAP.load(str(model_path), device="cpu")
    coordinates = np.asarray(loaded.transform(features, batch_size=32))
    panel = score_panel(
        features,
        coordinates,
        config=PanelV2Config(
            frac=0.25,
            k_hit=3,
            k_density=3,
            n_anchors=24,
            corpus_chunk=64,
            overselect=4,
            block_elems=100_000,
            rerank_byte_cap=8_000_000,
            peak_byte_cap=16_000_000,
        ),
        provenance={"round": "future", "mode": "exact-horizon-cpu-smoke"},
    )
    assert panel["guards"]["coords_finite"] is True
    assert panel["guards"]["coords_collapsed"] is False
