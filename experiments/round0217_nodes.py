"""Execute one cell of the R0217 MiniLM 2M four-seed family.

One handler, four nodes. Every node reads the same sealed R0216 receipt, derives
the same horizon from the same sealed edge count, and builds the same config —
so the only thing that can differ between the four cells is the seed, and the
node proves that by recomputing the seed-invariant config digest and refusing to
train unless it equals the value the prepare script stamped into the queue.

The horizon is never hardcoded. It comes from `directed_edge_count` in R0216's
sealed `substrate-graph.json` through the registered R0184/R0202 low-dose rule,
and both the exact `ceil` and the achieved draws-per-edge are asserted before a
single optimizer step runs.
"""
from __future__ import annotations

import gc
import os
import random
import resource
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_build_new_file,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0217_minilm_2m_pipeline import (
    MiniLMHostFp32EndpointArray,
    MiniLMMixedTrainingInput,
)
from basemap.round0217_minilm_2m_seed_family import (
    BATCH_SIZE,
    CAPABILITY_TEMPLATE,
    DIMENSION,
    GATE_REGISTERABLE_HERE,
    GRAPH_CAPABILITY,
    GRAPH_K,
    GRAPH_SCHEMA,
    GRAPH_SOURCE_ROUND_ID,
    HOST_RSS_LIMIT_GIB,
    POSITIVE_ROWS_PER_UPDATE,
    PRODUCTION_CONFIG_SCHEMA,
    RELOAD_PROBE_ROWS,
    RELOAD_PROBE_SEED,
    ROUND_ID,
    ROWS,
    Round0217Error,
    SEALED_DIRECTED_EDGES,
    SEEDS,
    TRAIN_SCHEMA,
    WARMUP_SUCCESSFUL_UPDATES,
    capability_for_seed,
    performance_windows,
    seed_invariant_sha256,
    successful_updates_for_edges,
    train_config,
    validate_dose,
    validate_published_map,
)
from basemap import round0113_prompt_contrast as prompt_contract
from experiments import round0113_nodes as prompt_nodes


ACTION = "train_minilm_mixed_2m_seed"


def _seed(job: Mapping[str, Any]) -> int:
    seed = job.get("training_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS:
        raise Round0217Error(f"R0217 job seed {seed!r} is not a registered cell")
    if str(job.get("capability") or "") != capability_for_seed(seed):
        raise Round0217Error("R0217 job capability does not match its seed")
    return int(seed)


def _sealed_graph(job: Mapping[str, Any]) -> dict[str, Any]:
    """Read the sealed R0216 receipt and load its exact k15 fuzzy graph."""
    manifest_signature = dict(job["graph_manifest_signature"])
    manifest_path = prompt_contract.verify_signature(
        manifest_signature, label="R0217 sealed R0216 substrate+graph receipt"
    )
    manifest = prompt_contract.read_sealed(
        manifest_path, label="R0217 sealed R0216 substrate+graph receipt"
    )
    checks = manifest.get("graph_checks") or {}
    degrees = manifest.get("degrees") or {}
    if (
        manifest.get("schema") != GRAPH_SCHEMA
        or manifest.get("round_id") != GRAPH_SOURCE_ROUND_ID
        or manifest.get("capability") != GRAPH_CAPABILITY
        or int(manifest.get("rows", -1)) != ROWS
        or int(manifest.get("dimension", -1)) != DIMENSION
        or int(manifest.get("k", -1)) != GRAPH_K
        or manifest.get("training_performed") is not False
    ):
        raise Round0217Error("R0217 sealed R0216 substrate+graph contract changed")
    if (
        int(checks.get("zero_degree_rows", -1)) != 0
        or int(degrees.get("zero_degree_rows", -1)) != 0
        or float(checks.get("mean_recall_at_k", 0.0))
        < float(checks.get("mean_recall_floor", 1.0))
        or float(checks.get("p10_recall_at_k", 0.0))
        < float(checks.get("p10_recall_floor", 1.0))
    ):
        raise Round0217Error(
            "R0217 requires the sealed R0216 graph to have passed its exactness "
            "and zero-degree checks"
        )
    edges = int(manifest.get("directed_edge_count", 0))
    if edges <= 0:
        edges = int(checks.get("directed_edges", 0))
    if edges != SEALED_DIRECTED_EDGES:
        raise Round0217Error(
            f"R0217 sealed graph reports {edges} directed edges, registered "
            f"{SEALED_DIRECTED_EDGES}"
        )
    graph_signature = dict(manifest["graph"])
    graph_path = prompt_contract.verify_signature(
        graph_signature, label="R0217 sealed R0216 fuzzy graph"
    )
    from basemap.pumap.parametric_umap.datasets.edge_list_dataset import (
        load_edge_arrays,
    )

    sources, targets, weights, n_nodes = load_edge_arrays(graph_path, load_weights=True)
    if (
        weights is None
        or int(n_nodes) != ROWS
        or len(sources) != edges
        or targets.shape != sources.shape
        or weights.shape != sources.shape
        or sources.dtype != np.int32
        or targets.dtype != np.int32
        or weights.dtype != np.float32
    ):
        raise Round0217Error("R0217 sealed R0216 graph arrays changed")
    return {
        "manifest": manifest,
        "manifest_signature": manifest_signature,
        "signature": graph_signature,
        "sources": sources,
        "targets": targets,
        "weights": weights,
        "n_nodes": int(n_nodes),
        "directed_edges": edges,
    }


def _open_substrate(graph: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Serve the 3.07 GB sealed substrate lazily; never materialize it."""
    signature = dict(graph["manifest"]["substrate"])
    path = prompt_contract.verify_signature(
        signature, label="R0217 sealed R0216 substrate"
    )
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != (ROWS, DIMENSION) or array.dtype != np.float32:
        raise Round0217Error("R0217 sealed R0216 substrate geometry changed")
    return array, signature


def _probe_rows() -> np.ndarray:
    return np.sort(
        np.random.RandomState(RELOAD_PROBE_SEED).choice(
            ROWS, RELOAD_PROBE_ROWS, replace=False
        )
    ).astype(np.int64)


def _weighted_rejection_accounting_mismatch(
    runtime: Mapping[str, Any], *, producer_delta: int, updates: int
) -> dict[str, Any] | None:
    expected_emitted = (updates + producer_delta) * POSITIVE_ROWS_PER_UPDATE
    if (
        int(runtime["weight_emitted_draws"]) != expected_emitted
        or int(runtime["weight_acceptances"])
        != int(runtime["weight_emitted_draws"]) + int(runtime["weight_buffered_draws"])
        or int(runtime["weight_proposals"]) < int(runtime["weight_acceptances"])
        or not 0 < float(runtime["weight_acceptance_rate"]) <= 1
    ):
        return {
            "expected_emitted_positive_draws": expected_emitted,
            "expected_consumed_positive_draws": updates * POSITIVE_ROWS_PER_UPDATE,
            "producer_delta": producer_delta,
            "runtime": runtime,
        }
    return None


def run_train(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch

    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0217Error("R0217 train handler received another queue")
    seed = _seed(job)
    graph = _sealed_graph(job)
    edges = graph["directed_edges"]
    updates = successful_updates_for_edges(edges)
    dose = validate_dose(updates=updates, edge_count=edges)
    declared_bound = job.get("registered_dose_bound")
    if declared_bound is not None and updates > int(declared_bound):
        raise Round0217Error(
            "R0217 derived update horizon exceeds the registered round bound"
        )
    source, substrate_signature = _open_substrate(graph)
    config, config_sha = train_config(
        seed=seed,
        graph_signature=graph["signature"],
        graph_manifest_signature=graph["manifest_signature"],
        substrate_signature=substrate_signature,
        graph_edges=edges,
        rows=ROWS,
    )
    declared_invariant = str(job.get("family_seed_invariant_sha256") or "")
    observed_invariant = seed_invariant_sha256(config)
    if not declared_invariant or observed_invariant != declared_invariant:
        raise Round0217Error(
            "R0217 cell config differs from the registered family outside the "
            f"seed: {observed_invariant} != {declared_invariant}"
        )
    if int(config["optimizer"]["successful_positive_lr_updates"]) != updates:
        raise Round0217Error("R0217 horizon did not reach the train config")

    output = create_fresh_directory(str(job["outputs"][0]), label="R0217 train output")
    config_path = os.path.join(output, "production-config.json")
    atomic_write_new_json(
        config_path,
        {
            "schema": PRODUCTION_CONFIG_SCHEMA,
            "round_id": ROUND_ID,
            "seed": seed,
            "capability": capability_for_seed(seed),
            "seed_invariant_sha256": observed_invariant,
            "config": config,
            "config_sha256": config_sha,
        },
        immutable=True,
    )

    dataset = MiniLMHostFp32EndpointArray(
        source,
        source_signature=substrate_signature,
        buffer_rows=BATCH_SIZE,
    )
    wrapper = MiniLMMixedTrainingInput(dataset, graph, seed=seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats("cuda")
    model = prompt_nodes._new_model(config)
    model._max_train_steps = updates
    model._bench_warmup = WARMUP_SUCCESSFUL_UPDATES
    model._perf_profile = True
    model._perf_floor = config["execution"]["minimum_train_upd_s"]
    model._perf_warn_rate = config["execution"]["warning_train_upd_s"]
    model._perf_subfloor_patience = 2
    model._perf_n_windows = performance_windows(updates)
    model._abort_on_first_nonfinite = True
    model._admission_artifact_path = os.path.join(output, "admission.json")
    started = time.monotonic()
    model.fit(
        wrapper,
        low_memory=True,
        verbose=False,
        n_processes=6,
        random_state=seed,
        resample_negatives=False,
        precomputed_edges_path=graph["signature"]["canonical_path"],
        use_wandb=False,
    )
    wall = time.monotonic() - started

    accounting = dict(model._train_stats)
    runtime = wrapper.runtime_stamp()
    expected_stamp = config["execution"]["expected_pipeline_stamp"]
    mismatches = {
        key: {"expected": value, "observed": runtime.get(key)}
        for key, value in expected_stamp.items()
        if runtime.get(key) != value
    }
    exact = {
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
    }
    mismatches.update({
        key: {"expected": value, "observed": accounting.get(key)}
        for key, value in exact.items()
        if accounting.get(key) != value
    })
    expected_rows = updates * BATCH_SIZE
    producer_delta = (
        int(runtime["host_prefetch_producer_batches"])
        - int(runtime["host_prefetch_consumer_batches"])
    )
    if (
        int(runtime["source_rows_gathered"]) != expected_rows
        or int(runtime["destination_rows_gathered"]) != expected_rows
        or int(runtime["host_prefetch_consumer_batches"]) != updates
        or producer_delta not in {0, 1}
    ):
        mismatches["endpoint_accounting"] = {
            "expected_rows": expected_rows,
            "runtime": runtime,
        }
    weighted = _weighted_rejection_accounting_mismatch(
        runtime, producer_delta=producer_delta, updates=updates
    )
    if weighted is not None:
        mismatches["weighted_rejection_accounting"] = weighted
    if mismatches:
        raise Round0217Error(f"R0217 train accounting failed: {mismatches}")
    prompt_contract.synchronize_runtime_counters(accounting, runtime)
    accounting["pipeline_runtime"] = dict(runtime)

    profiler = model._canary_profiler.finalize(
        bench_seconds=model._bench_seconds,
        setup_seconds=getattr(model, "_setup_seconds", None),
    )
    rate = (
        (updates - WARMUP_SUCCESSFUL_UPDATES) / model._bench_seconds
        if model._bench_seconds
        else 0.0
    )
    if profiler.get("aborted") is not False or rate < config["execution"][
        "minimum_train_upd_s"
    ]:
        raise Round0217Error("R0217 train performance admission failed")

    model_path = os.path.join(output, "model.pt")
    atomic_build_new_file(model_path, model.save, immutable=True)
    free_bytes, total_bytes = torch.cuda.mem_get_info("cuda")
    memory = {
        "device_total_bytes": int(total_bytes),
        "post_train_free_bytes": int(free_bytes),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda")),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda")),
    }
    del model, wrapper, dataset
    torch.cuda.empty_cache()
    gc.collect()

    # The published checkpoint must reload from disk and project a real map.
    from basemap.pumap.parametric_umap import ParametricUMAP

    reloaded = ParametricUMAP.load(model_path, device="cuda")
    probe = _probe_rows()
    coordinates = np.asarray(
        reloaded.transform(np.asarray(source[probe], dtype=np.float32), batch_size=4096),
        dtype=np.float64,
    )
    published = validate_published_map(coordinates)
    published["model"] = expected_input_signature(model_path)
    del reloaded
    torch.cuda.empty_cache()
    gc.collect()

    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    if peak_rss_gib > HOST_RSS_LIMIT_GIB:
        raise Round0217Error(
            f"R0217 train peak RSS {peak_rss_gib:.2f} GiB exceeds "
            f"{HOST_RSS_LIMIT_GIB:.0f} GiB"
        )
    memory["peak_host_rss_gib"] = peak_rss_gib

    receipt = prompt_contract.seal({
        "schema": TRAIN_SCHEMA,
        "round_id": ROUND_ID,
        "capability": capability_for_seed(seed),
        "capabilities": [capability_for_seed(seed)],
        "training_seed": seed,
        "release_sha": active["manifest"]["release_sha"],
        "production_config": expected_input_signature(config_path),
        "production_config_sha256": config_sha,
        "seed_invariant_sha256": observed_invariant,
        "model": published["model"],
        "substrate": substrate_signature,
        "graph_manifest": graph["manifest_signature"],
        "graph": graph["signature"],
        "graph_capability": GRAPH_CAPABILITY,
        "rows": ROWS,
        "dimension": DIMENSION,
        "directed_edges": edges,
        "dose_registration": dose,
        "train_accounting": accounting,
        "exact_execution_receipt": runtime,
        "requested_positive_draws_per_edge": float(
            config["execution"]["target_positive_draws_per_edge"]
        ),
        "consumed_positive_draws": int(updates * POSITIVE_ROWS_PER_UPDATE),
        "consumed_positive_draws_per_edge": float(
            updates * POSITIVE_ROWS_PER_UPDATE / edges
        ),
        "performance_profile": profiler,
        "steady_updates_per_s": rate,
        "train_wall_s": wall,
        "published_map_check": published,
        "train_checks": {
            "exact_update_closure": True,
            "zero_numerical_skips": True,
            "no_pipeline_stamp_drift": True,
            "endpoint_rows_match_updates": True,
            "weighted_rejection_accounting_closes": True,
            "dose_derived_from_sealed_edge_count": True,
            "family_config_identical_except_seed": True,
            "published_checkpoint_reloads_finite_and_uncollapsed": True,
        },
        "memory": memory,
        "training_performed": True,
        "optimizer_updates": updates,
        "map_decision_made": False,
        "gate_registerable_here": GATE_REGISTERABLE_HERE,
    })
    atomic_write_new_json(
        os.path.join(output, "train-receipt.json"), receipt, immutable=True
    )
    del source, graph
    gc.collect()


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if str(job.get("action") or "") != ACTION:
        raise Round0217Error("R0217 authorizes only the MiniLM 2M seed-family train")
    run_train(active, job)


__all__ = ["ACTION", "CAPABILITY_TEMPLATE", "run_job", "run_train"]
