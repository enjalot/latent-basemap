"""Execute the conditional R0129 seed-43 native-degree replicate."""
from __future__ import annotations

import gc
import json
import os
import random
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_build_new_file,
    atomic_save_new_npz,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.panel_v2 import _self_knn, load_hiD_reference, sample_anchors
from basemap.round0113_prompt_contrast import (
    BATCH_SIZE,
    DIMENSION,
    PERFORMANCE_WARMUP_UPDATES,
    PERFORMANCE_WINDOWS,
    RETAINED_ROWS,
    SUCCESSFUL_UPDATES,
    TRAIN_MINIMUM_UPDATES_PER_S,
    HostFp16EndpointArray,
    read_sealed,
    seal,
    synchronize_runtime_counters,
    train_config as r0117_train_config,
    verify_signature,
)
from basemap.round0124_degree_bridge import (
    ARM,
    DECISION_SCHEMA as R0124_DECISION_SCHEMA,
    DegreeBridgeTrainingInput,
    GRAPH_DEGREE,
    NATIVE_ANCHOR_SEED,
    NATIVE_DENSITY_ANCHORS,
    OUTCOME_INCONCLUSIVE,
    OUTCOME_MATERIAL,
    OUTCOME_NOT_MATERIAL,
    classify_degree_bridge,
    paired_density_bootstrap,
)
from basemap.round0129_degree_replicate import (
    CAPABILITY,
    DECISION_SCHEMA,
    DIAGNOSTIC_SCHEMA,
    NATIVE_DENSITY_SCHEMA,
    PRODUCTION_CONFIG_SCHEMA,
    ROUND_ID,
    TRAIN_RECEIPT_SCHEMA,
    TRAINING_SEED,
    Round0129Error,
    config_equivalence,
    load_k15_graph,
    train_config,
    verify_graph_provenance,
)
from experiments import round0113_nodes as prompt_nodes
from experiments import round0124_nodes as bridge_nodes


def _execution_round_id(active: Mapping[str, Any]) -> str:
    round_id = str((active.get("manifest") or {}).get("round_id", ""))
    if round_id != ROUND_ID:
        raise Round0129Error("R0129 handler received another queue")
    return round_id


def _graph(job: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    provenance = verify_graph_provenance(job.get("graph_provenance"))
    expected_path = provenance["evidence"]["graph_manifest"]["canonical_path"]
    if str(job.get("graph_manifest") or "") != expected_path:
        raise Round0129Error("R0129 graph path changed")
    return load_k15_graph(provenance), provenance


def _r0117_control_config(
    job: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    train_path = str(job["r0117_control_train_receipt"])
    train = read_sealed(train_path, label="R0117 raw seed-43 train")
    config_path = verify_signature(
        train.get("production_config"), label="R0117 production config"
    )
    with open(config_path, encoding="utf-8") as handle:
        receipt = json.load(handle)
    config = receipt.get("config")
    if not isinstance(config, Mapping):
        raise Round0129Error("R0117 production config is missing")
    return dict(config), expected_input_signature(config_path)


def _new_model(config: Mapping[str, Any]):
    model = bridge_nodes._new_model(config)
    return model


def run_train(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any]:
    import torch

    _execution_round_id(active)
    if int(job.get("training_seed", -1)) != TRAINING_SEED:
        raise Round0129Error("R0129 training seed changed")
    assembly, assembly_signature = prompt_nodes._load_assembly(job)
    graph, provenance = _graph(job)
    config, config_sha = train_config(
        graph_signature=graph["signature"],
        graph_manifest_signature=graph["manifest_signature"],
        graph_edges=len(graph["sources"]),
        retained_rows=graph["n_nodes"],
    )
    control_config, control_config_signature = _r0117_control_config(job)
    equivalence = config_equivalence(
        treatment=config, control=control_config
    )
    source = prompt_nodes._open_compact(assembly, ARM)
    dataset = HostFp16EndpointArray(
        source,
        arm=ARM,
        source_signature=assembly["outputs"][ARM],
        mapping_signature=assembly["mapping"],
        buffer_rows=BATCH_SIZE,
    )
    wrapper = DegreeBridgeTrainingInput(dataset, graph)
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0129 k15 seed-43 train"
    )
    config_path = os.path.join(output, "production-config.json")
    atomic_write_new_json(
        config_path,
        {
            "schema": PRODUCTION_CONFIG_SCHEMA,
            "round_id": ROUND_ID,
            "arm": ARM,
            "training_seed": TRAINING_SEED,
            "config": config,
            "config_sha256": config_sha,
            "r0117_control_config": control_config_signature,
            "config_equivalence": equivalence,
        },
        immutable=True,
    )
    random.seed(TRAINING_SEED)
    np.random.seed(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)
    torch.cuda.manual_seed_all(TRAINING_SEED)
    torch.cuda.reset_peak_memory_stats("cuda")
    model = _new_model(config)
    model._max_train_steps = SUCCESSFUL_UPDATES
    model._bench_warmup = PERFORMANCE_WARMUP_UPDATES
    model._perf_profile = True
    model._perf_floor = config["execution"]["minimum_train_upd_s"]
    model._perf_warn_rate = config["execution"]["warning_train_upd_s"]
    model._perf_subfloor_patience = 2
    model._perf_n_windows = PERFORMANCE_WINDOWS
    model._abort_on_first_nonfinite = True
    model._admission_artifact_path = os.path.join(output, "admission.json")
    started = time.monotonic()
    model.fit(
        wrapper,
        low_memory=True,
        verbose=False,
        n_processes=6,
        random_state=TRAINING_SEED,
        resample_negatives=False,
        precomputed_edges_path=graph["signature"]["canonical_path"],
        use_wandb=False,
    )
    wall = time.monotonic() - started
    accounting = dict(model._train_stats)
    runtime = wrapper.runtime_stamp()
    synchronize_runtime_counters(accounting, runtime)
    expected_stamp = config["execution"]["expected_pipeline_stamp"]
    bridge_nodes._verify_train_accounting(
        accounting=accounting,
        runtime=runtime,
        expected_stamp=expected_stamp,
        expected_edges=len(graph["sources"]),
        label="R0129 seed-43 treatment",
    )
    profiler = model._canary_profiler.finalize(
        bench_seconds=model._bench_seconds,
        setup_seconds=getattr(model, "_setup_seconds", None),
    )
    rate = (
        (SUCCESSFUL_UPDATES - PERFORMANCE_WARMUP_UPDATES)
        / model._bench_seconds
        if model._bench_seconds
        else 0.0
    )
    if profiler.get("aborted") is not False or rate < TRAIN_MINIMUM_UPDATES_PER_S:
        raise Round0129Error("R0129 treatment performance admission failed")
    model_path = os.path.join(output, "model.pt")
    atomic_build_new_file(model_path, model.save, immutable=True)
    free_bytes, total_bytes = torch.cuda.mem_get_info("cuda")
    body = {
        "schema": TRAIN_RECEIPT_SCHEMA,
        "round_id": ROUND_ID,
        "arm": ARM,
        "training_seed": TRAINING_SEED,
        "release_sha": active["manifest"]["release_sha"],
        "production_config": expected_input_signature(config_path),
        "production_config_sha256": config_sha,
        "model": expected_input_signature(model_path),
        "assembly": assembly_signature,
        "graph_manifest": graph["manifest_signature"],
        "graph": graph["signature"],
        "graph_provenance": provenance,
        "r0117_control_config": control_config_signature,
        "config_equivalence": equivalence,
        "train_accounting": accounting,
        "exact_execution_receipt": runtime,
        "performance_profile": profiler,
        "steady_updates_per_s": rate,
        "train_wall_s": wall,
        "train_checks": {
            "exact_update_closure": True,
            "zero_numerical_skips": True,
            "no_pipeline_stamp_drift": True,
            "endpoint_rows_match_updates": True,
            "weighted_rejection_accounting_closes": True,
            "graph_only_config_equivalence": True,
        },
        "memory": {
            "device_total_bytes": int(total_bytes),
            "post_train_free_bytes": int(free_bytes),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda")),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda")),
        },
        "causal_change": "graph-degree-k49-to-k15-only-at-seed43",
        "identical_realized_edge_draws_claimed": False,
        "training_performed": True,
        "optimizer_updates": SUCCESSFUL_UPDATES,
        "map_decision_made": False,
    }
    receipt = seal(body)
    receipt_path = os.path.join(output, "train-receipt.json")
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    del model, wrapper, dataset, source, graph
    torch.cuda.empty_cache()
    gc.collect()
    return {**receipt, "receipt": expected_input_signature(receipt_path)}


def _authenticate_model(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    _execution_round_id(active)
    assembly, assembly_signature = prompt_nodes._load_assembly(job)
    graph, provenance = _graph(job)
    config, config_sha = train_config(
        graph_signature=graph["signature"],
        graph_manifest_signature=graph["manifest_signature"],
        graph_edges=len(graph["sources"]),
        retained_rows=graph["n_nodes"],
    )
    control_config, control_config_signature = _r0117_control_config(job)
    equivalence = config_equivalence(treatment=config, control=control_config)
    train_path = os.path.join(str(job["train_output"]), "train-receipt.json")
    train = read_sealed(train_path, label="R0129 treatment train")
    config_path = verify_signature(
        train.get("production_config"), label="R0129 production config"
    )
    with open(config_path, encoding="utf-8") as handle:
        production = json.load(handle)
    runtime = train.get("exact_execution_receipt")
    accounting = train.get("train_accounting")
    checks = train.get("train_checks")
    if (
        train.get("schema") != TRAIN_RECEIPT_SCHEMA
        or train.get("round_id") != ROUND_ID
        or train.get("arm") != ARM
        or train.get("training_seed") != TRAINING_SEED
        or train.get("release_sha") != active["manifest"]["release_sha"]
        or train.get("production_config_sha256") != config_sha
        or train.get("assembly") != assembly_signature
        or train.get("graph_manifest") != graph["manifest_signature"]
        or train.get("graph") != graph["signature"]
        or train.get("graph_provenance") != provenance
        or train.get("r0117_control_config") != control_config_signature
        or train.get("config_equivalence") != equivalence
        or production.get("schema") != PRODUCTION_CONFIG_SCHEMA
        or production.get("round_id") != ROUND_ID
        or production.get("config") != config
        or production.get("config_sha256") != config_sha
        or production.get("config_equivalence") != equivalence
        or train.get("optimizer_updates") != SUCCESSFUL_UPDATES
        or not isinstance(runtime, Mapping)
        or not isinstance(accounting, Mapping)
        or not isinstance(checks, Mapping)
        or any(value is not True for value in checks.values())
    ):
        raise Round0129Error("R0129 treatment train/config binding changed")
    bridge_nodes._verify_train_accounting(
        accounting=accounting,
        runtime=runtime,
        expected_stamp=config["execution"]["expected_pipeline_stamp"],
        expected_edges=len(graph["sources"]),
        label="R0129 treatment reload",
    )
    model_path = verify_signature(train["model"], label="R0129 treatment model")
    from basemap.pumap.parametric_umap import ParametricUMAP

    model = ParametricUMAP.load(model_path, device="cuda")
    expected = config["model"]
    observed = {
        "architecture": model.architecture,
        "input_dimension": model.input_dim,
        "hidden_dimension": model.hidden_dim,
        "hidden_layers": model.n_layers,
        "output_dimension": model.n_components,
        "use_batchnorm": model.use_batchnorm,
        "use_dropout": model.use_dropout,
        "low_dim_kernel": model.low_dim_kernel,
        "a": model.a,
        "b": model.b,
    }
    if observed != expected:
        raise Round0129Error("R0129 treatment architecture changed")
    return model, train, assembly, graph


def run_diagnostics(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any]:
    _execution_round_id(active)
    originals = {
        "_authenticate_model": prompt_nodes._authenticate_model,
        "_execution_round_id": prompt_nodes._execution_round_id,
        "_schema": prompt_nodes._schema,
    }
    prompt_nodes._authenticate_model = _authenticate_model
    prompt_nodes._execution_round_id = _execution_round_id
    prompt_nodes._schema = (
        lambda stem: DIAGNOSTIC_SCHEMA
        if stem == "prompt-arm-score"
        else f"round0129-{stem}-v1"
    )
    try:
        result = prompt_nodes.run_evaluate(dict(active), dict(job))
    finally:
        for name, value in originals.items():
            setattr(prompt_nodes, name, value)
    if (
        result.get("schema") != DIAGNOSTIC_SCHEMA
        or result.get("training_seed") != TRAINING_SEED
    ):
        raise Round0129Error("R0129 diagnostic receipt changed")
    return result


def _r0117_native_evidence(
    job: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    assembly, assembly_signature = prompt_nodes._load_assembly(job)
    graph, graph_signature = bridge_nodes._control_graph(
        job, assembly_signature=assembly_signature
    )
    train_path = str(job["r0117_control_train_receipt"])
    score_path = str(job["r0117_control_score"])
    train_signature = expected_input_signature(train_path)
    score_signature = expected_input_signature(score_path)
    train = read_sealed(train_path, label="R0117 raw seed-43 train")
    score = read_sealed(score_path, label="R0117 raw seed-43 score")
    control_config, control_sha = r0117_train_config(
        ARM,
        graph_signature=graph["graph"],
        graph_manifest_signature=graph_signature,
        graph_edges=int(graph["directed_edge_count"]),
        retained_rows=int(graph["retained_rows"]),
        seed=TRAINING_SEED,
    )
    production_path = verify_signature(
        train.get("production_config"), label="R0117 control config"
    )
    with open(production_path, encoding="utf-8") as handle:
        production = json.load(handle)
    accounting = train.get("train_accounting")
    runtime = train.get("exact_execution_receipt")
    checks = train.get("train_checks")
    coordinates = (score.get("coordinates") or {}).get("training")
    panel = score.get("panel")
    if (
        train.get("schema") != "round0113-train-receipt-v1"
        or train.get("round_id") != "0117"
        or train.get("release_sha") != job.get("r0117_release_sha")
        or train.get("arm") != ARM
        or train.get("training_seed") != TRAINING_SEED
        or train.get("assembly") != assembly_signature
        or train.get("graph_manifest") != graph_signature
        or train.get("graph") != graph["graph"]
        or train.get("production_config_sha256") != control_sha
        or production.get("config") != control_config
        or train.get("optimizer_updates") != SUCCESSFUL_UPDATES
        or not isinstance(accounting, Mapping)
        or not isinstance(runtime, Mapping)
        or not isinstance(checks, Mapping)
        or any(checks.get(key) is not True for key in checks)
        or score.get("schema") != "round0113-prompt-arm-score-v1"
        or score.get("round_id") != "0117"
        or score.get("release_sha") != job.get("r0117_release_sha")
        or score.get("arm") != ARM
        or score.get("training_seed") != TRAINING_SEED
        or score.get("graph_manifest") != graph_signature
        or score.get("train_receipt") != train_signature
        or score.get("high_d_reference") != graph["high_d_reference"]
        or not isinstance(coordinates, Mapping)
        or not isinstance(panel, Mapping)
        or panel.get("n") != RETAINED_ROWS
        or panel.get("n_anchors") != NATIVE_DENSITY_ANCHORS
        or panel.get("anchor_seed") != NATIVE_ANCHOR_SEED
        or panel.get("k_density") != GRAPH_DEGREE
        or panel.get("density") != 0.2116
    ):
        raise Round0129Error("R0117 native control evidence changed")
    bridge_nodes._verify_train_accounting(
        accounting=accounting,
        runtime=runtime,
        expected_stamp=control_config["execution"]["expected_pipeline_stamp"],
        expected_edges=int(graph["directed_edge_count"]),
        label="R0117 raw seed-43 control",
    )
    verify_signature(train["model"], label="R0117 raw model")
    verify_signature(coordinates, label="R0117 raw coordinates")
    verify_signature(graph["high_d_reference"], label="R0117 high-D reference")
    return graph, graph_signature, score, score_signature


def run_native_density(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any]:
    _execution_round_id(active)
    provenance = verify_graph_provenance(job.get("graph_provenance"))
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0129 native density contrast"
    )
    control_graph, control_graph_signature, control_score, control_score_signature = (
        _r0117_native_evidence(job)
    )
    diagnostic_path = os.path.join(str(job["diagnostic_output"]), "score.json")
    diagnostics = read_sealed(diagnostic_path, label="R0129 diagnostics")
    treatment_coordinates = (
        (diagnostics.get("coordinates") or {}).get("training") or {}
    )
    treatment_train_signature = expected_input_signature(
        os.path.join(str(job["train_output"]), "train-receipt.json")
    )
    treatment_graph_signature = provenance["evidence"]["graph_manifest"]
    if (
        diagnostics.get("schema") != DIAGNOSTIC_SCHEMA
        or diagnostics.get("round_id") != ROUND_ID
        or diagnostics.get("release_sha") != active["manifest"]["release_sha"]
        or diagnostics.get("arm") != ARM
        or diagnostics.get("training_seed") != TRAINING_SEED
        or diagnostics.get("train_receipt") != treatment_train_signature
        or diagnostics.get("graph_manifest") != treatment_graph_signature
        or diagnostics.get("high_d_reference")
        != control_graph["high_d_reference"]
        or not isinstance(treatment_coordinates, Mapping)
    ):
        raise Round0129Error("R0129 treatment diagnostics changed")
    control_coordinates = control_score["coordinates"]["training"]
    control_path = verify_signature(
        control_coordinates, label="R0117 raw seed-43 coordinates"
    )
    treatment_path = verify_signature(
        treatment_coordinates, label="R0129 k15 seed-43 coordinates"
    )
    reference_path = verify_signature(
        control_graph["high_d_reference"], label="R0117 high-D reference"
    )
    reference = load_hiD_reference(
        reference_path,
        expected_key=str(control_graph["high_d_reference_key"]),
    )
    config = prompt_nodes.panel_config()
    anchors = np.asarray(reference["anchor_ids"], dtype=np.int64)
    high_radius = np.asarray(reference["r_hd"], dtype=np.float64)
    if (
        config.n_anchors != NATIVE_DENSITY_ANCHORS
        or config.anchor_seed != NATIVE_ANCHOR_SEED
        or config.k_density != GRAPH_DEGREE
        or anchors.shape != (NATIVE_DENSITY_ANCHORS,)
        or high_radius.shape != anchors.shape
        or not np.array_equal(anchors, sample_anchors(RETAINED_ROWS, config))
    ):
        raise Round0129Error("R0129 native-density anchor contract changed")
    control_array = np.load(control_path, mmap_mode="r", allow_pickle=False)
    treatment_array = np.load(treatment_path, mmap_mode="r", allow_pickle=False)
    if (
        control_array.shape != (RETAINED_ROWS, 2)
        or treatment_array.shape != (RETAINED_ROWS, 2)
        or not np.isfinite(control_array).all()
        or not np.isfinite(treatment_array).all()
    ):
        raise Round0129Error("R0129 coordinate geometry changed")
    _cn, control_distances, _cg = _self_knn(
        control_array, anchors, GRAPH_DEGREE, config, hi_dim=False, want_dist=True
    )
    _tn, treatment_distances, _tg = _self_knn(
        treatment_array,
        anchors,
        GRAPH_DEGREE,
        config,
        hi_dim=False,
        want_dist=True,
    )
    control_low_radius = np.asarray(control_distances).mean(1)
    treatment_low_radius = np.asarray(treatment_distances).mean(1)
    bootstrap = paired_density_bootstrap(
        high_radius=high_radius,
        control_low_radius=control_low_radius,
        treatment_low_radius=treatment_low_radius,
    )
    deltas = np.asarray(bootstrap.pop("bootstrap_deltas"), dtype=np.float64)
    interval = bootstrap["paired_bootstrap_delta_ci"]
    selector = classify_degree_bridge(
        control_density=float(bootstrap["control_density"]),
        treatment_density=float(bootstrap["treatment_density"]),
        delta_ci_low=float(interval[0]),
        delta_ci_high=float(interval[1]),
    )
    if (
        round(float(bootstrap["control_density"]), 4)
        != control_score["metrics"]["density"]
        or round(float(bootstrap["treatment_density"]), 4)
        != diagnostics["metrics"]["density"]
    ):
        raise Round0129Error("R0129 density does not reproduce panel scores")
    arrays_path = os.path.join(output, "native-density-arrays.npz")
    atomic_save_new_npz(
        arrays_path,
        immutable=True,
        anchor_compact_rows=anchors,
        high_radius=high_radius,
        control_low_radius=control_low_radius,
        treatment_low_radius=treatment_low_radius,
        paired_bootstrap_deltas=deltas,
    )
    body = {
        "schema": NATIVE_DENSITY_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "training_seed": TRAINING_SEED,
        "graph_provenance": provenance,
        "control": {
            "role": "exact accepted R0117 raw seed-43 k49 native re-score",
            "score": control_score_signature,
            "graph_manifest": control_graph_signature,
            "coordinates": control_coordinates,
            "density": bootstrap["control_density"],
            "recorded_panel_density": control_score["metrics"]["density"],
        },
        "treatment": {
            "role": "R0129 raw seed-43 k15 native score",
            "diagnostics": expected_input_signature(diagnostic_path),
            "train_receipt": treatment_train_signature,
            "graph_manifest": treatment_graph_signature,
            "coordinates": treatment_coordinates,
            "density": bootstrap["treatment_density"],
            "recorded_panel_density": diagnostics["metrics"]["density"],
        },
        "native_reference": {
            "high_d_reference": control_graph["high_d_reference"],
            "anchor_count": len(anchors),
            "anchor_seed": config.anchor_seed,
            "k_density": config.k_density,
            "low_d_search": "panel-v2 exact global chunked top-k; mean k15 radius",
        },
        "registered_selector": selector,
        "bootstrap_diagnostics": bootstrap,
        "arrays": expected_input_signature(arrays_path),
        "changed_factor": "fuzzy graph neighbor degree only",
        "config_and_sampling_law_equivalent": True,
        "identical_realized_edge_draws_claimed": False,
        "core_and_ood_diagnostics_registered_role": "diagnostic-only",
        "legacy_density_floor_used": False,
        "training_performed_in_this_node": False,
    }
    score = seal(body)
    score_path = os.path.join(output, "native-density-score.json")
    atomic_write_new_json(score_path, score, immutable=True)
    del control_array, treatment_array
    gc.collect()
    return {**score, "receipt": expected_input_signature(score_path)}


def run_decision(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any]:
    _execution_round_id(active)
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0129 degree replicate decision"
    )
    density_path = os.path.join(
        str(job["density_output"]), "native-density-score.json"
    )
    diagnostic_path = os.path.join(str(job["diagnostic_output"]), "score.json")
    train_path = os.path.join(str(job["train_output"]), "train-receipt.json")
    density = read_sealed(density_path, label="R0129 density score")
    diagnostics = read_sealed(diagnostic_path, label="R0129 diagnostics")
    train = read_sealed(train_path, label="R0129 train receipt")
    trigger_signature = dict(job["r0124_inconclusive_decision"])
    trigger_path = verify_signature(
        trigger_signature, label="R0124 inconclusive decision"
    )
    trigger = read_sealed(trigger_path, label="R0124 inconclusive decision")
    expected_gates = {
        "finite_noncollapsed_coordinates",
        "transductive_recall50_gt_recall10",
        "matched_projection_recall50_gt_recall10",
        "exact_update_closure",
        "zero_numerical_skips",
        "no_pipeline_stamp_drift",
    }
    gates = diagnostics.get("execution_gates")
    if (
        trigger.get("schema") != R0124_DECISION_SCHEMA
        or (trigger.get("registered_selector") or {}).get("outcome")
        != OUTCOME_INCONCLUSIVE
        or density.get("schema") != NATIVE_DENSITY_SCHEMA
        or density.get("round_id") != ROUND_ID
        or density.get("release_sha") != active["manifest"]["release_sha"]
        or diagnostics.get("schema") != DIAGNOSTIC_SCHEMA
        or diagnostics.get("round_id") != ROUND_ID
        or diagnostics.get("release_sha") != active["manifest"]["release_sha"]
        or train.get("schema") != TRAIN_RECEIPT_SCHEMA
        or train.get("round_id") != ROUND_ID
        or train.get("config_equivalence", {}).get(
            "identical_realized_edge_draws_claimed"
        )
        is not False
        or density.get("treatment", {}).get("diagnostics")
        != expected_input_signature(diagnostic_path)
        or not isinstance(gates, Mapping)
        or set(gates) != expected_gates
        or not all(gates.values())
    ):
        raise Round0129Error("R0129 decision evidence changed")
    observed = density.get("registered_selector") or {}
    interval = observed.get("paired_bootstrap_delta_ci") or []
    if len(interval) != 2:
        raise Round0129Error("R0129 paired density interval is missing")
    selector = classify_degree_bridge(
        control_density=float(density["control"]["density"]),
        treatment_density=float(density["treatment"]["density"]),
        delta_ci_low=float(interval[0]),
        delta_ci_high=float(interval[1]),
    )
    if selector != observed:
        raise Round0129Error("R0129 registered selector changed")
    interpretation = {
        OUTCOME_MATERIAL: "seed43-supports-material-k15-degradation",
        OUTCOME_NOT_MATERIAL: "seed43-does-not-support-material-k15-degradation",
        OUTCOME_INCONCLUSIVE: "two-seed-degree-effect-remains-unresolved",
    }[selector["outcome"]]
    body = {
        "schema": DECISION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "training_seed": TRAINING_SEED,
        "r0124_inconclusive_trigger": trigger_signature,
        "density_score": expected_input_signature(density_path),
        "diagnostics": expected_input_signature(diagnostic_path),
        "train_receipt": expected_input_signature(train_path),
        "registered_selector": selector,
        "cross_seed_interpretation": interpretation,
        "causal_claim": (
            "within the exact accepted raw 2M seed-43 recipe, the registered "
            "difference is graph degree/topology; configuration and sampling "
            "law are equivalent, while realized weighted edge draws are not "
            "claimed identical"
        ),
        "config_equivalence": train["config_equivalence"],
        "diagnostic_metrics": diagnostics["metrics"],
        "polish_ood": diagnostics["ood"]["pol_Latn"],
        "diagnostics_can_rescue_or_fail_selector": False,
        "capabilities_produced": [CAPABILITY],
        "legacy_density_floor_used": False,
        "scale_contribution_excluded": True,
        "training_performed": True,
        "optimizer_updates": SUCCESSFUL_UPDATES,
        "production_ready": False,
    }
    receipt = seal(body)
    path = os.path.join(output, "decision.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_job(
    active: Mapping[str, Any], job: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    selected = job if job is not None else active.get("job") or {}
    action = selected.get("action")
    if action == "train_k15_seed43":
        return run_train(active, selected)
    if action == "evaluate_core_ood":
        return run_diagnostics(active, selected)
    if action == "score_native_density":
        return run_native_density(active, selected)
    if action == "decide_degree_replicate":
        return run_decision(active, selected)
    raise Round0129Error(f"unknown R0129 action: {action!r}")
