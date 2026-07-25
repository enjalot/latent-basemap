"""Fresh-process handlers for the selected seed-43 repulsion confirmation."""
from __future__ import annotations

import os
import random
import time
from typing import Any, Mapping

import numpy as np

from basemap.artifact_identity import (
    expected_input_signature,
    ordered_array_sha256,
)
from basemap.output_safety import (
    atomic_build_new_file,
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0034_pipeline import load_canonical_graph
from basemap.round0040_program import RepresentativeArrayView, panel_config
from basemap.round0048_program import (
    CENTROIDS_K1024_PATH,
    CENTROIDS_K256_PATH,
    ELIGIBILITY_SHA256,
    QUERIES_PATH,
    QUERY_PROVENANCE_PATH,
    REFERENCE_RECEIPT,
    REFERENCE_RECEIPT_SHA256,
    ROW_COUNT,
    SELECTOR_PATH,
    SELECTOR_SHA256,
    train_configs_from_graph as r0048_configs,
)
from basemap.round0051_program import NEGATIVE_MULTIPLIERS
from basemap.round0056_program import (
    BASELINE_COORDINATES,
    BASELINE_TRAIN_RECEIPT,
    R0051_COMPARISON,
    ROUND_ID,
    SEED,
    SUCCESSFUL_UPDATES,
    selected_arm,
    train_config_from_graph,
)
from experiments.round0046_nodes import (
    _coordinate_stream,
    _exact_model,
    _panel_scalars,
    _projection,
    _read_sealed,
    _seal,
)
from experiments.round0048_nodes import _load_training_input
from experiments.round0051_nodes import (
    NormalizedClassWeightedBCELoss,
    _optimizer_without_loss,
)


def _load_context(
    job: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, Any], str]:
    graph = load_canonical_graph(
        str(job["canonical_graph_manifest"]),
        expected_sha256=str(job["canonical_graph_manifest_sha256"]),
        expected_eligibility_sha256=ELIGIBILITY_SHA256,
        row_count=ROW_COUNT,
    )
    comparison_signature = expected_input_signature(R0051_COMPARISON)
    if (
        comparison_signature["sha256"]
        != str(job["r0051_comparison_sha256"])
    ):
        raise RuntimeError("R0051 comparison bytes changed")
    comparison = _read_sealed(
        R0051_COMPARISON,
        label="R0051 negative-BCE selection",
    )
    arm = selected_arm(comparison)
    if job.get("selected_arm") != arm:
        raise RuntimeError("R0056 queue selected a different arm")
    config, config_sha256 = train_config_from_graph(
        graph["manifest"],
        graph_manifest_path=graph["signature"]["canonical_path"],
        graph_manifest_sha256=graph["signature"]["sha256"],
        arm=arm,
    )
    if job.get("train_config_sha256") != config_sha256:
        raise RuntimeError("R0056 production config identity changed")
    return graph, comparison, arm, config, config_sha256


def run_train(active: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    graph, comparison, arm, config, config_sha256 = _load_context(job)
    output = create_fresh_directory(
        job["outputs"][0],
        label="Round 0056 seed-43 train output",
    )
    atomic_write_new_json(
        os.path.join(output, "production-config.json"),
        {
            "schema": "round0056-production-config-receipt-v1",
            "selected_arm": arm,
            "config": config,
            "config_sha256": config_sha256,
        },
        immutable=True,
    )

    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats("cuda")
    wrapper = _load_training_input(graph, arm="edge_uniform")
    instance = _exact_model(config)
    loss = NormalizedClassWeightedBCELoss(
        positive_multiplier=1.0,
        negative_multiplier=NEGATIVE_MULTIPLIERS[arm],
    )
    instance.loss_fn = loss
    if loss.runtime_stamp() != config["execution"]["expected_loss_stamp"]:
        raise RuntimeError("R0056 actual loss stamp changed")
    instance._max_train_steps = SUCCESSFUL_UPDATES
    instance._bench_warmup = 200
    instance._perf_profile = True
    instance._perf_floor = float(
        config["execution"]["minimum_train_upd_s"]
    )
    instance._perf_warn_rate = float(
        config["execution"]["warning_train_upd_s"]
    )
    instance._perf_subfloor_patience = int(
        config["execution"]["performance_subfloor_patience"]
    )
    instance._perf_n_windows = int(
        config["execution"]["performance_windows"]
    )
    instance._abort_on_first_nonfinite = True
    instance._admission_artifact_path = os.path.join(
        output,
        "admission.json",
    )
    started = time.monotonic()
    instance.fit(
        wrapper,
        low_memory=False,
        verbose=False,
        n_processes=6,
        random_state=SEED,
        resample_negatives=False,
        precomputed_edges_path=graph["signature"]["canonical_path"],
        use_wandb=False,
    )
    wall_seconds = time.monotonic() - started
    accounting = dict(instance._train_stats)
    expected_stamp = config["execution"]["expected_pipeline_stamp"]
    expected = {
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
        "n_pos_edges": 444_198_115,
        **{
            f"pipeline_{key}": value
            for key, value in expected_stamp.items()
        },
    }
    mismatches = {
        key: {"expected": value, "observed": accounting.get(key)}
        for key, value in expected.items()
        if accounting.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"R0056 exact train accounting failed: {mismatches}")
    runtime = wrapper.runtime_stamp()
    runtime_mismatches = {
        key: {"expected": value, "observed": runtime.get(key)}
        for key, value in expected_stamp.items()
        if runtime.get(key) != value
    }
    if (
        runtime_mismatches
        or accounting.get("pipeline_runtime") != runtime
    ):
        raise RuntimeError(
            f"R0056 runtime stamp differs: {runtime_mismatches}"
        )
    profiler = instance._canary_profiler.finalize(
        bench_seconds=instance._bench_seconds,
        setup_seconds=getattr(instance, "_setup_seconds", None),
    )
    if (
        profiler.get("n_windows")
        != config["execution"]["performance_windows"]
        or len(profiler.get("rate_windows") or [])
        != config["execution"]["performance_windows"]
    ):
        raise RuntimeError("R0056 profiler did not close every window")

    from experiments.run_round0014_node import _publish_model

    model_path = os.path.join(output, "model.pt")
    _publish_model(instance, model_path)
    body = {
        "schema": "round0056-train-receipt-v1",
        "round_id": ROUND_ID,
        "selected_arm": arm,
        "release_sha": active["manifest"]["release_sha"],
        "production_config": config,
        "production_config_sha256": config_sha256,
        "model": expected_input_signature(model_path),
        "graph": graph["signature"],
        "eligibility": graph["manifest"]["inputs"]["eligibility"],
        "train_accounting": accounting,
        "exact_execution_receipt": runtime,
        "loss_runtime_stamp": loss.runtime_stamp(),
        "performance_profile": profiler,
        "r0051_comparison": expected_input_signature(R0051_COMPARISON),
        "r0051_selection": comparison["selection"],
        "train_wall_seconds": wall_seconds,
        "seed": SEED,
        "retry_count": 0,
    }
    receipt = _seal(body)
    path = os.path.join(output, "train-receipt.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_transform(
    active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    _graph, _comparison, arm, config, config_sha256 = _load_context(job)
    train_path = os.path.join(job["train_output"], "train-receipt.json")
    train = _read_sealed(train_path, label="R0056 train receipt")
    if (
        train.get("schema") != "round0056-train-receipt-v1"
        or train.get("selected_arm") != arm
        or train.get("production_config_sha256") != config_sha256
        or train.get("loss_runtime_stamp")
        != config["execution"]["expected_loss_stamp"]
        or train.get("train_accounting", {}).get("budget_satisfied")
        is not True
    ):
        raise RuntimeError("R0056 transform lacks a valid train")
    output = create_fresh_directory(
        job["outputs"][0],
        label="Round 0056 coordinate output",
    )
    from basemap.round0014_transform import (
        production_transform,
        stream_production_coordinates,
    )

    result = stream_production_coordinates(
        model_path=train["model"]["canonical_path"],
        template_path=job["transform_spec_template"],
        release_root=active["manifest"]["repo_root"],
        release_sha=active["manifest"]["release_sha"],
        output_root=output,
        production_config=config,
        production_config_sha256=config_sha256,
    )
    queries = np.load(QUERIES_PATH, mmap_mode="r", allow_pickle=False)
    query_coordinates = production_transform(
        np.asarray(queries, dtype="<f4")
    )
    query_path = os.path.join(output, "heldout-query-coordinates.npy")
    atomic_save_new_npy(query_path, query_coordinates, immutable=True)
    body = {
        "schema": "round0056-transform-capability-v1",
        "round_id": ROUND_ID,
        "selected_arm": arm,
        **result,
        "train_receipt": expected_input_signature(train_path),
        "heldout_queries": expected_input_signature(QUERIES_PATH),
        "heldout_query_provenance": expected_input_signature(
            QUERY_PROVENANCE_PATH
        ),
        "heldout_query_coordinates": expected_input_signature(query_path),
    }
    receipt = _seal(body)
    path = os.path.join(output, "actual-transform.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def _render(
    *,
    output: str,
    selector: Any,
    coordinates: Mapping[str, RepresentativeArrayView],
) -> dict[str, Any]:
    rng = np.random.RandomState(20260726)
    compact = np.sort(
        rng.choice(
            len(next(iter(coordinates.values()))),
            50_000,
            replace=False,
        )
    ).astype(np.int64)
    global_rows = selector.compact_to_global(compact)
    ids_path = os.path.join(output, "sample-global-row-ids.npy")
    atomic_save_new_npy(ids_path, global_rows, immutable=True)
    points = {
        label: np.asarray(value[compact], dtype=np.float32)
        for label, value in coordinates.items()
    }
    if any(
        not np.isfinite(value).all()
        or np.any(np.std(value, axis=0) <= 1e-8)
        for value in points.values()
    ):
        raise RuntimeError("R0056 render coordinates invalid")
    image_path = os.path.join(
        output,
        "seed43-negative-bce-confirmation.png",
    )

    def draw(path: str) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(16, 8))
        for axis, (label, value) in zip(axes, points.items()):
            axis.scatter(
                value[:, 0],
                value[:, 1],
                s=0.15,
                alpha=0.35,
                linewidths=0,
                rasterized=True,
            )
            axis.set_aspect("equal", adjustable="box")
            axis.set_title(label)
            axis.set_xticks([])
            axis.set_yticks([])
        figure.tight_layout()
        figure.savefig(path, format="png", dpi=180, bbox_inches="tight")
        plt.close(figure)

    atomic_build_new_file(image_path, draw, immutable=True)
    return {
        "sample_seed": 20260726,
        "sample_size": len(compact),
        "sample_global_row_ids": expected_input_signature(ids_path),
        "sample_global_row_ids_sha256": ordered_array_sha256(global_rows),
        "same_semantic_rows_all_maps": True,
        "image": expected_input_signature(image_path),
    }


def run_matched_panel(
    active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    from basemap.panel_v2 import (
        load_hiD_reference,
        load_query_truth,
        score_panel,
    )
    from experiments.round0040_nodes import (
        _load_minilm_selector,
        _minilm_base,
    )

    graph, comparison, arm, config, _config_sha256 = _load_context(job)
    output = create_fresh_directory(
        job["outputs"][0],
        label="Round 0056 matched panel output",
    )
    started = time.monotonic()
    reference_signature = expected_input_signature(REFERENCE_RECEIPT)
    if reference_signature["sha256"] != REFERENCE_RECEIPT_SHA256:
        raise RuntimeError("R0056 reference receipt changed")
    reference_receipt = _read_sealed(
        REFERENCE_RECEIPT,
        label="R0040 MiniLM reference",
    )
    reference = load_hiD_reference(
        reference_receipt["reference"]["canonical_path"],
        expected_key=reference_receipt["reference_key"],
    )
    truth = load_query_truth(
        reference_receipt["query_truth"]["canonical_path"],
        expected_key=reference_receipt["query_truth_key"],
        expected_candidate_compute_backend="cuda",
    )
    selector_signature = expected_input_signature(SELECTOR_PATH)
    if selector_signature["sha256"] != SELECTOR_SHA256:
        raise RuntimeError("R0056 selector changed")
    selector, _selector_artifact = _load_minilm_selector(
        selector_signature
    )
    X = RepresentativeArrayView(_minilm_base(), selector)
    panel_cfg = panel_config()
    centroids = {
        256: np.load(
            CENTROIDS_K256_PATH,
            mmap_mode="r",
            allow_pickle=False,
        ),
        1024: np.load(
            CENTROIDS_K1024_PATH,
            mmap_mode="r",
            allow_pickle=False,
        ),
    }
    coordinate_specs = {
        "baseline_edge_1p00": (
            BASELINE_COORDINATES,
            "round0048-transform-capability-v1",
        ),
        arm: (
            str(job["transform_output"]),
            "round0056-transform-capability-v1",
        ),
    }
    cells: dict[str, Any] = {}
    coordinate_views: dict[str, RepresentativeArrayView] = {}
    records: dict[str, Any] = {}
    for label, (root, schema) in coordinate_specs.items():
        full, record = _coordinate_stream(root, expected_schema=schema)
        if (
            label == arm
            and record.get("selected_arm") != arm
        ) or (
            label == "baseline_edge_1p00"
            and record.get("arm") != "edge_uniform"
        ):
            raise RuntimeError("R0056 coordinate arm mismatch")
        records[label] = record
        coordinates = RepresentativeArrayView(full, selector)
        coordinate_views[label] = coordinates
        panel = score_panel(
            X,
            coordinates,
            config=panel_cfg,
            centroids_by_k=centroids,
            hiD_reference=reference,
            reference_identity=reference_receipt["identity"],
            provenance={
                "round_id": ROUND_ID,
                "release_sha": active["manifest"]["release_sha"],
                "map_label": label,
                "coordinate_capability": expected_input_signature(
                    os.path.join(root, "actual-transform.json")
                ),
                "scientific_universe": "exact-fp16-representatives",
            },
        )
        query_coordinates = np.load(
            os.path.join(root, "heldout-query-coordinates.npy"),
            mmap_mode="r",
            allow_pickle=False,
        )
        projection = _projection(
            query_coordinates=query_coordinates,
            coordinates=coordinates,
            truth=truth,
            config=panel_cfg,
        )
        cell_body = {
            "schema": "round0056-representative-cell-v1",
            "round_id": ROUND_ID,
            "cell": label,
            "negative_bce_multiplier": (
                1.0
                if label == "baseline_edge_1p00"
                else NEGATIVE_MULTIPLIERS[arm]
            ),
            "scientific_rows": len(coordinates),
            "full_rows": ROW_COUNT,
            "panel": panel,
            "projection": projection,
            "coordinate_capability": expected_input_signature(
                os.path.join(root, "actual-transform.json")
            ),
            "coordinate_model": record["actual_transform"][
                "model_signature"
            ],
        }
        cell_path = os.path.join(output, f"{label}-panel.json")
        atomic_write_new_json(
            cell_path,
            _seal(cell_body),
            immutable=True,
        )
        cells[label] = {
            "receipt": expected_input_signature(cell_path),
            "scalars": _panel_scalars(panel),
            "projection": projection,
        }

    baseline_train = _read_sealed(
        BASELINE_TRAIN_RECEIPT,
        label="R0048 edge baseline train",
    )
    treatment_train_path = os.path.join(
        job["train_output"],
        "train-receipt.json",
    )
    treatment_train = _read_sealed(
        treatment_train_path,
        label="R0056 treatment train",
    )
    baseline_config = r0048_configs(
        graph["manifest"],
        graph_manifest_path=graph["signature"]["canonical_path"],
        graph_manifest_sha256=graph["signature"]["sha256"],
    )["edge_uniform"][0]
    baseline_runtime = baseline_train["exact_execution_receipt"]
    runtime = treatment_train["exact_execution_receipt"]
    same_runtime_fields = (
        "pipeline",
        "sampler_class",
        "x_residency",
        "positive_sampling",
        "positive_source_sampling",
        "positive_destination_policy",
        "negative_sampling",
        "positive_source_count",
        "valid_canonical_edge_count",
        "graph_degree",
        "graph_manifest",
        "eligibility",
        "weighted_requested",
        "weighted_effective",
    )
    matched = {
        "r0051_selected_arm": selected_arm(comparison) == arm,
        "baseline_is_r0048_seed43_edge_uniform": (
            baseline_train.get("schema") == "round0048-train-receipt-v1"
            and baseline_train.get("arm") == "edge_uniform"
            and baseline_train.get("seed") == SEED
        ),
        "row_universe": config["row_universe"] == baseline_config["row_universe"],
        "model": config["model"] == baseline_config["model"],
        "optimizer_except_loss": (
            _optimizer_without_loss(config)
            == _optimizer_without_loss(baseline_config)
        ),
        "graph": treatment_train["graph"] == baseline_train["graph"],
        "features": (
            treatment_train["train_accounting"]["verified_hashes"]["features"]
            == baseline_train["train_accounting"]["verified_hashes"][
                "features"
            ]
        ),
        "seed": (
            treatment_train["seed"] == baseline_train["seed"] == SEED
        ),
        "successful_updates": (
            treatment_train["train_accounting"][
                "positive_lr_optimizer_steps"
            ]
            == baseline_train["train_accounting"][
                "positive_lr_optimizer_steps"
            ]
            == SUCCESSFUL_UPDATES
        ),
        "runtime": all(
            runtime.get(key) == baseline_runtime.get(key)
            for key in same_runtime_fields
        ),
        "loss_stamp": (
            treatment_train["loss_runtime_stamp"]
            == config["execution"]["expected_loss_stamp"]
        ),
    }
    baseline = cells["baseline_edge_1p00"]
    treatment = cells[arm]
    scalar_delta = {
        key: treatment["scalars"][key] - baseline["scalars"][key]
        for key in treatment["scalars"]
    }
    projection_delta = {
        key: treatment["projection"][key] - baseline["projection"][key]
        for key in ("proj_ffr", "proj_recall_at_10")
    }
    thresholds = config["decision_thresholds"]
    guards = {
        "density_improvement": (
            scalar_delta["density"]
            >= thresholds["density_improvement_min"]
        ),
        "ffr": (
            scalar_delta["ffr"]
            >= thresholds["representative_ffr_delta_min"]
        ),
        "projection_ffr": (
            projection_delta["proj_ffr"]
            >= thresholds["representative_projection_ffr_delta_min"]
        ),
        "purity_k256": (
            scalar_delta["purity_k256"]
            >= thresholds["representative_purity_delta_min"]
        ),
        "purity_k1024": (
            scalar_delta["purity_k1024"]
            >= thresholds["representative_purity_delta_min"]
        ),
    }
    numerical = {}
    for label, cell in cells.items():
        sealed = _read_sealed(
            cell["receipt"]["canonical_path"],
            label=f"R0056 {label} panel",
        )
        numerical[label] = (
            sealed["panel"]["guards"].get("coords_finite") is True
            and sealed["panel"]["guards"].get("coords_collapsed") is False
        )
    valid = (
        all(matched.values())
        and all(numerical.values())
        and treatment_train["performance_profile"]["n_windows"] == 200
    )
    classification = (
        "invalid-isolation"
        if not valid
        else (
            "seed43-replicated-candidate"
            if all(guards.values())
            else "seed-sensitive-not-replicated"
        )
    )
    render = _render(
        output=output,
        selector=selector,
        coordinates=coordinate_views,
    )
    body = {
        "schema": "round0056-negative-bce-seed43-confirmation-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "seed": SEED,
        "selected_arm": arm,
        "negative_bce_multiplier": NEGATIVE_MULTIPLIERS[arm],
        "r0051_comparison": expected_input_signature(R0051_COMPARISON),
        "baseline_train": expected_input_signature(
            BASELINE_TRAIN_RECEIPT
        ),
        "treatment_train": expected_input_signature(
            treatment_train_path
        ),
        "cells": cells,
        "treatment_minus_baseline": {
            "scalars": scalar_delta,
            "projection": projection_delta,
        },
        "quality_guards": guards,
        "matched_contract": matched,
        "numerical_guards": numerical,
        "classification": classification,
        "interpretation": {
            "isolated_treatment": (
                "review-selected normalized negative BCE contribution"
            ),
            "external_ood_adoption_gate_run": False,
            "candidate_only_until_external_ood": True,
        },
        "render": render,
        "performance": {
            "panel_wall_seconds": time.monotonic() - started,
        },
    }
    receipt = _seal(body)
    path = os.path.join(
        output,
        "negative-bce-seed43-confirmation-v1.json",
    )
    atomic_write_new_json(path, receipt, immutable=True)
    if not valid:
        raise RuntimeError("R0056 matched isolation is invalid")
    return {**receipt, "receipt": expected_input_signature(path)}


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise RuntimeError("R0056 handler received another queue")
    selected = job if job is not None else active.get("job") or {}
    handler = {
        "train": run_train,
        "transform": run_transform,
        "matched_panel": run_matched_panel,
    }.get(selected.get("action"))
    if handler is None:
        raise RuntimeError(
            f"unknown R0056 action: {selected.get('action')!r}"
        )
    return handler(active, selected)
