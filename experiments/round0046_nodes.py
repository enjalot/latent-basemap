"""Fresh-process handlers for the R0046 source-exposure isolation."""
from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Mapping

import numpy as np

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    ordered_array_sha256,
    sha256_bytes,
)
from basemap.output_safety import (
    atomic_build_new_file,
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0014_program import (
    Round0014MaterializedArray,
    validate_materialized_pack,
)
from basemap.round0034_pipeline import load_canonical_graph
from basemap.round0040_program import (
    RepresentativeArrayView,
    panel_config,
)
from basemap.round0041_program import load_fp16_eligibility
from basemap.round0042_pipeline import (
    Round0042PipelineError,
    Round0042TrainingInput,
)
from basemap.round0046_program import (
    CENTROIDS_K1024_PATH,
    CENTROIDS_K256_PATH,
    ELIGIBILITY_SHA256,
    QUERIES_PATH,
    QUERY_PROVENANCE_PATH,
    R0042_COORDINATES,
    R0042_PANEL,
    R0042_TRAIN_RECEIPT,
    REFERENCE_RECEIPT,
    REFERENCE_RECEIPT_SHA256,
    ROUND_ID,
    ROW_COUNT,
    SEED,
    SELECTOR_PATH,
    SELECTOR_SHA256,
    SUCCESSFUL_UPDATES,
    train_config_from_graph,
)
from experiments.round0042_nodes import (
    _coordinate_stream,
    _exact_model,
    _panel_scalars,
    _projection,
    _read_sealed,
)


def _seal(body: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(body)
    return {
        **payload,
        "identity_sha256": sha256_bytes(canonical_json(payload)),
    }


def _load_graph_config(
    job: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    graph = load_canonical_graph(
        str(job["canonical_graph_manifest"]),
        expected_sha256=str(job["canonical_graph_manifest_sha256"]),
        expected_eligibility_sha256=ELIGIBILITY_SHA256,
        row_count=ROW_COUNT,
    )
    config, config_sha256 = train_config_from_graph(
        graph["manifest"],
        graph_manifest_path=graph["signature"]["canonical_path"],
        graph_manifest_sha256=graph["signature"]["sha256"],
    )
    if job.get("train_config_sha256") != config_sha256:
        raise Round0042PipelineError("R0046 queue/config identity changed")
    return graph, config, config_sha256


def _load_training_input(
    graph: Mapping[str, Any],
) -> Round0042TrainingInput:
    from basemap.pumap.parametric_umap.datasets.edge_list_dataset import (
        DeviceArrayDataset,
    )

    features = Round0014MaterializedArray()
    feature_signature = validate_materialized_pack(features)
    eligibility = load_fp16_eligibility()
    if (
        eligibility["signature"]["sha256"] != ELIGIBILITY_SHA256
        or not np.array_equal(
            eligibility["excluded_rows"],
            np.unique(eligibility["excluded_rows"]),
        )
    ):
        raise Round0042PipelineError("R0046 eligibility changed")
    dataset = DeviceArrayDataset(features, "cuda")
    return Round0042TrainingInput(
        dataset,
        graph=graph,
        excluded_rows=eligibility["excluded_rows"],
        feature_signature=feature_signature,
        positive_source_law="uniform-edge",
    )


def run_train(
    active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    graph, config, config_sha256 = _load_graph_config(job)
    output = create_fresh_directory(
        job["outputs"][0],
        label="Round 0046 train output",
    )
    atomic_write_new_json(
        os.path.join(output, "production-config.json"),
        {
            "schema": "round0046-production-config-receipt-v1",
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
    wrapper = _load_training_input(graph)
    instance = _exact_model(config)
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
        raise Round0042PipelineError(
            f"R0046 exact train accounting failed: {mismatches}"
        )
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
        raise Round0042PipelineError(
            "R0046 runtime stamp differs from admission: "
            f"{runtime_mismatches}"
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
        raise Round0042PipelineError(
            "R0046 profiler did not close every scheduled window"
        )

    from experiments.run_round0014_node import _publish_model

    model_path = os.path.join(output, "model.pt")
    _publish_model(instance, model_path)
    body = {
        "schema": "round0046-train-receipt-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "production_config": config,
        "production_config_sha256": config_sha256,
        "model": expected_input_signature(model_path),
        "graph": graph["signature"],
        "eligibility": graph["manifest"]["inputs"]["eligibility"],
        "train_accounting": accounting,
        "exact_execution_receipt": runtime,
        "performance_profile": profiler,
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
    graph, config, config_sha256 = _load_graph_config(job)
    del graph
    train_path = os.path.join(job["train_output"], "train-receipt.json")
    train = _read_sealed(train_path, label="Round 0046 train receipt")
    if (
        train.get("schema") != "round0046-train-receipt-v1"
        or train.get("production_config_sha256") != config_sha256
        or train.get("train_accounting", {}).get("budget_satisfied")
        is not True
    ):
        raise Round0042PipelineError("R0046 transform lacks a valid train")
    output = create_fresh_directory(
        job["outputs"][0],
        label="Round 0046 coordinate output",
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
        "schema": "round0046-transform-capability-v1",
        "round_id": ROUND_ID,
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


def _render_comparison(
    *,
    output: str,
    selector: Any,
    coordinates: Mapping[str, RepresentativeArrayView],
) -> dict[str, Any]:
    labels = ("r0042_source_uniform", "r0046_edge_uniform")
    rng = np.random.RandomState(20260725)
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
        label: np.asarray(coordinates[label][compact], dtype=np.float32)
        for label in labels
    }
    if any(
        not np.isfinite(value).all()
        or np.any(np.std(value, axis=0) <= 1e-8)
        for value in points.values()
    ):
        raise Round0042PipelineError(
            "R0046 comparison render coordinates invalid"
        )
    image_path = os.path.join(
        output,
        "r0042-source-vs-r0046-edge.png",
    )

    def draw(path: str) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(16, 8))
        for axis, label in zip(axes, labels):
            value = points[label]
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
        "sample_seed": 20260725,
        "sample_size": len(compact),
        "sample_global_row_ids": expected_input_signature(ids_path),
        "sample_global_row_ids_sha256": ordered_array_sha256(global_rows),
        "same_semantic_rows_both_maps": True,
        "image": expected_input_signature(image_path),
        "diagnostics": {
            label: {
                "axis_std": value.std(axis=0).astype(float).tolist(),
                "axis_span": np.ptp(value, axis=0).astype(float).tolist(),
            }
            for label, value in points.items()
        },
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

    output = create_fresh_directory(
        job["outputs"][0],
        label="Round 0046 matched panel output",
    )
    started = time.monotonic()
    reference_signature = expected_input_signature(REFERENCE_RECEIPT)
    if reference_signature["sha256"] != REFERENCE_RECEIPT_SHA256:
        raise Round0042PipelineError("R0046 reference receipt changed")
    reference_receipt = _read_sealed(
        REFERENCE_RECEIPT,
        label="Round 0040 MiniLM reference",
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
        raise Round0042PipelineError("R0046 selector changed")
    selector, _selector_artifact = _load_minilm_selector(
        selector_signature
    )
    X = RepresentativeArrayView(_minilm_base(), selector)
    config = panel_config()
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
        "r0042_source_uniform": (
            R0042_COORDINATES,
            "round0042-transform-capability-v1",
        ),
        "r0046_edge_uniform": (
            job["transform_output"],
            "round0046-transform-capability-v1",
        ),
    }
    cells: dict[str, Any] = {}
    coordinate_views: dict[str, RepresentativeArrayView] = {}
    for label, (root, schema) in coordinate_specs.items():
        full, record = _coordinate_stream(root, expected_schema=schema)
        coordinates = RepresentativeArrayView(full, selector)
        coordinate_views[label] = coordinates
        panel = score_panel(
            X,
            coordinates,
            config=config,
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
            config=config,
        )
        cell_body = {
            "schema": "round0046-representative-cell-v1",
            "round_id": ROUND_ID,
            "cell": label,
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

    control = cells["r0042_source_uniform"]
    treatment = cells["r0046_edge_uniform"]
    deltas = {
        key: treatment["scalars"][key] - control["scalars"][key]
        for key in treatment["scalars"]
    }
    projection_deltas = {
        key: treatment["projection"][key] - control["projection"][key]
        for key in ("proj_ffr", "proj_recall_at_10")
    }
    control_train = _read_sealed(
        R0042_TRAIN_RECEIPT,
        label="R0042 train receipt",
    )
    treatment_train_path = os.path.join(
        job["train_output"],
        "train-receipt.json",
    )
    treatment_train = _read_sealed(
        treatment_train_path,
        label="R0046 train receipt",
    )
    control_runtime = control_train["exact_execution_receipt"]
    treatment_runtime = treatment_train["exact_execution_receipt"]
    control_config = control_train["production_config"]
    treatment_config = treatment_train["production_config"]
    same_runtime_fields = (
        "x_residency",
        "positive_destination_policy",
        "negative_sampling",
        "positive_source_count",
        "valid_canonical_edge_count",
        "graph_degree",
        "graph_manifest",
        "eligibility",
    )
    matched = {
        "row_universe": (
            control_config["row_universe"]
            == treatment_config["row_universe"]
        ),
        "model": (
            control_config["model"] == treatment_config["model"]
        ),
        "optimizer": (
            control_config["optimizer"] == treatment_config["optimizer"]
        ),
        "graph": (
            control_train["graph"] == treatment_train["graph"]
        ),
        "features": (
            control_train["train_accounting"]["verified_hashes"]["features"]
            == treatment_train["train_accounting"][
                "verified_hashes"
            ]["features"]
        ),
        "seed": (
            control_train["seed"] == treatment_train["seed"] == SEED
        ),
        "successful_updates": (
            control_train["train_accounting"][
                "positive_lr_optimizer_steps"
            ]
            == treatment_train["train_accounting"][
                "positive_lr_optimizer_steps"
            ]
            == SUCCESSFUL_UPDATES
        ),
        "all_non_source_runtime_fields": all(
            control_runtime.get(key) == treatment_runtime.get(key)
            for key in same_runtime_fields
        ),
        "control_source_uniform": (
            control_runtime["positive_sampling"].startswith(
                "uniform-retained-positive-source"
            )
        ),
        "treatment_edge_uniform": (
            treatment_runtime["positive_sampling"]
            == "uniform-valid-canonical-edge-with-replacement"
            and treatment_runtime["positive_source_sampling"]
            == "degree-proportional-over-positive-sources"
        ),
    }
    numerical_guards = {
        label: (
            _read_sealed(
                cell["receipt"]["canonical_path"],
                label=f"{label} panel",
            )["panel"]["guards"].get("coords_finite")
            is True
            and _read_sealed(
                cell["receipt"]["canonical_path"],
                label=f"{label} panel",
            )["panel"]["guards"].get("coords_collapsed")
            is False
        )
        for label, cell in cells.items()
    }
    thresholds = treatment_config["decision_thresholds"]
    quality_noninferiority = {
        "representative_ffr": (
            deltas["ffr"]
            >= thresholds["representative_ffr_delta_min"]
        ),
        "representative_projection_ffr": (
            projection_deltas["proj_ffr"]
            >= thresholds["representative_projection_ffr_delta_min"]
        ),
        "representative_purity_k256": (
            deltas["purity_k256"]
            >= thresholds["representative_purity_delta_min"]
        ),
        "representative_purity_k1024": (
            deltas["purity_k1024"]
            >= thresholds["representative_purity_delta_min"]
        ),
    }
    isolation_valid = (
        all(matched.values())
        and all(numerical_guards.values())
    )
    if not isolation_valid:
        classification = "invalid-isolation"
    elif (
        deltas["density"]
        >= thresholds["material_density_recovery_delta_min"]
        and all(quality_noninferiority.values())
    ):
        classification = "source-exposure-primary-contributor"
    elif (
        abs(deltas["density"])
        <= thresholds["density_equivalence_abs_delta_max"]
    ):
        classification = "source-exposure-not-sufficient"
    else:
        classification = "mixed-or-seed-sensitive"
    checks = {
        "matched_contract": all(matched.values()),
        "both_numerical_guards": all(numerical_guards.values()),
        "all_quality_noninferiority": all(
            quality_noninferiority.values()
        ),
        "profiler_has_200_windows": (
            treatment_train["performance_profile"]["n_windows"] == 200
        ),
    }
    render = _render_comparison(
        output=output,
        selector=selector,
        coordinates=coordinate_views,
    )
    body = {
        "schema": "round0046-source-exposure-comparison-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "scientific_universe": {
            "rows": len(X),
            "row_namespace": (
                "compact ascending exact-fp16 representative global rows"
            ),
            "selector": selector.identity(),
        },
        "reference_receipt": reference_signature,
        "cells": cells,
        "edge_uniform_minus_source_uniform": deltas,
        "projection_deltas": projection_deltas,
        "matched_contract": matched,
        "numerical_guards": numerical_guards,
        "quality_noninferiority": quality_noninferiority,
        "decision_checks": checks,
        "classification": classification,
        "interpretation": {
            "isolated_treatment": (
                "positive-source exposure law only: uniform source versus "
                "degree-proportional uniform valid canonical edge"
            ),
            "density_primary_geometry_gate": True,
            "scale_claimed": False,
            "topology_change_claimed": False,
        },
        "prior_R0042_panel": expected_input_signature(R0042_PANEL),
        "render": render,
        "wall_seconds": time.monotonic() - started,
    }
    receipt = _seal(body)
    path = os.path.join(
        output,
        "source-exposure-comparison-v1.json",
    )
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise RuntimeError("R0046 handler received another queue")
    selected = job if job is not None else active.get("job") or {}
    if len(selected.get("outputs") or []) != 1:
        raise RuntimeError("R0046 job output contract changed")
    handlers = {
        "train": run_train,
        "transform": run_transform,
        "matched_panel": run_matched_panel,
    }
    handler = handlers.get(selected.get("action"))
    if handler is None:
        raise RuntimeError(
            f"unknown R0046 action: {selected.get('action')!r}"
        )
    return handler(active, selected)
