"""Fresh-process handlers for the R0051 normalized-BCE calibration."""
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
from basemap.round0042_pipeline import (
    Round0042PipelineError,
)
from basemap.round0051_program import (
    ARMS,
    BASELINE_COMPARISON,
    BASELINE_COORDINATES,
    BASELINE_MULTIPLIER,
    BASELINE_TRAIN_RECEIPT,
    CENTROIDS_K1024_PATH,
    CENTROIDS_K256_PATH,
    ELIGIBILITY_SHA256,
    NEGATIVE_MULTIPLIERS,
    QUERIES_PATH,
    QUERY_PROVENANCE_PATH,
    REFERENCE_RECEIPT,
    REFERENCE_RECEIPT_SHA256,
    ROUND_ID,
    ROW_COUNT,
    SEED,
    SELECTOR_PATH,
    SELECTOR_SHA256,
    SUCCESSFUL_UPDATES,
    train_configs_from_graph,
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


class NormalizedClassWeightedBCELoss:
    """BCE with an explicit class ratio and invariant mean weight."""

    def __init__(
        self,
        *,
        positive_multiplier: float,
        negative_multiplier: float,
    ):
        if positive_multiplier <= 0 or negative_multiplier <= 0:
            raise ValueError("BCE class multipliers must be positive")
        self.positive_multiplier = float(positive_multiplier)
        self.negative_multiplier = float(negative_multiplier)

    def __call__(self, values: Any, targets: Any) -> Any:
        import torch
        import torch.nn.functional as functional

        element = functional.binary_cross_entropy(
            values,
            targets,
            reduction="none",
        )
        positive = (targets > 0.5).to(dtype=element.dtype)
        weights = (
            self.negative_multiplier
            + positive
            * (
                self.positive_multiplier
                - self.negative_multiplier
            )
        )
        denominator = weights.sum()
        return (element * weights).sum() / denominator

    def runtime_stamp(self) -> dict[str, Any]:
        return {
            "loss_class": type(self).__name__,
            "positive_multiplier": self.positive_multiplier,
            "negative_multiplier": self.negative_multiplier,
            "reduction": "weighted-sum-over-weight-sum",
            "positive_threshold": 0.5,
        }


def _arm(job: Mapping[str, Any]) -> str:
    arm = str(job.get("arm"))
    if arm not in ARMS:
        raise Round0042PipelineError(f"unknown R0051 arm: {arm!r}")
    return arm


def _load_graph_configs(
    job: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, tuple[dict[str, Any], str]],
]:
    graph = load_canonical_graph(
        str(job["canonical_graph_manifest"]),
        expected_sha256=str(job["canonical_graph_manifest_sha256"]),
        expected_eligibility_sha256=ELIGIBILITY_SHA256,
        row_count=ROW_COUNT,
    )
    configs = train_configs_from_graph(
        graph["manifest"],
        graph_manifest_path=graph["signature"]["canonical_path"],
        graph_manifest_sha256=graph["signature"]["sha256"],
    )
    expected = job.get("train_config_sha256_by_arm")
    observed = {arm: value[1] for arm, value in configs.items()}
    if expected != observed:
        raise Round0042PipelineError("R0051 queue/config identities changed")
    return graph, configs


def run_train(
    active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    arm = _arm(job)
    graph, configs = _load_graph_configs(job)
    config, config_sha256 = configs[arm]
    output = create_fresh_directory(
        job["outputs"][0],
        label=f"Round 0051 {arm} train output",
    )
    atomic_write_new_json(
        os.path.join(output, "production-config.json"),
        {
            "schema": f"round0051-{arm}-production-config-receipt-v1",
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
        raise Round0042PipelineError("R0051 actual loss stamp changed")
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
            f"R0051 {arm} exact train accounting failed: {mismatches}"
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
            f"R0051 {arm} runtime stamp differs: {runtime_mismatches}"
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
            f"R0051 {arm} profiler did not close every window"
        )
    from experiments.run_round0014_node import _publish_model

    model_path = os.path.join(output, "model.pt")
    _publish_model(instance, model_path)
    body = {
        "schema": "round0051-train-receipt-v1",
        "round_id": ROUND_ID,
        "arm": arm,
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
    arm = _arm(job)
    _graph, configs = _load_graph_configs(job)
    config, config_sha256 = configs[arm]
    train_path = os.path.join(job["train_output"], "train-receipt.json")
    train = _read_sealed(
        train_path,
        label=f"Round 0051 {arm} train receipt",
    )
    if (
        train.get("schema") != "round0051-train-receipt-v1"
        or train.get("arm") != arm
        or train.get("production_config_sha256") != config_sha256
        or train.get("loss_runtime_stamp")
        != config["execution"]["expected_loss_stamp"]
        or train.get("train_accounting", {}).get("budget_satisfied")
        is not True
    ):
        raise Round0042PipelineError(
            f"R0051 {arm} transform lacks a valid train"
        )
    output = create_fresh_directory(
        job["outputs"][0],
        label=f"Round 0051 {arm} coordinate output",
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
        "schema": "round0051-transform-capability-v1",
        "round_id": ROUND_ID,
        "arm": arm,
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


def _render_cells(
    *,
    output: str,
    selector: Any,
    coordinates: Mapping[str, RepresentativeArrayView],
) -> dict[str, Any]:
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
        label: np.asarray(value[compact], dtype=np.float32)
        for label, value in coordinates.items()
    }
    if any(
        not np.isfinite(value).all()
        or np.any(np.std(value, axis=0) <= 1e-8)
        for value in points.values()
    ):
        raise Round0042PipelineError("R0051 render coordinates invalid")
    image_path = os.path.join(
        output,
        "seed42-negative-bce-calibration.png",
    )

    def draw(path: str) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 3, figsize=(24, 8))
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
        "sample_seed": 20260725,
        "sample_size": len(compact),
        "sample_global_row_ids": expected_input_signature(ids_path),
        "sample_global_row_ids_sha256": ordered_array_sha256(global_rows),
        "same_semantic_rows_all_maps": True,
        "image": expected_input_signature(image_path),
        "diagnostics": {
            label: {
                "axis_std": value.std(axis=0).astype(float).tolist(),
                "axis_span": np.ptp(value, axis=0).astype(float).tolist(),
            }
            for label, value in points.items()
        },
    }


def _optimizer_without_loss(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config["optimizer"].items()
        if key not in {
            "positive_bce_multiplier",
            "negative_bce_multiplier",
            "bce_reduction",
        }
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

    _graph, configs = _load_graph_configs(job)
    output = create_fresh_directory(
        job["outputs"][0],
        label="Round 0051 matched panel output",
    )
    started = time.monotonic()
    reference_signature = expected_input_signature(REFERENCE_RECEIPT)
    if reference_signature["sha256"] != REFERENCE_RECEIPT_SHA256:
        raise Round0042PipelineError("R0051 reference receipt changed")
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
        raise Round0042PipelineError("R0051 selector changed")
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
        "baseline_1p00": (
            BASELINE_COORDINATES,
            "round0046-transform-capability-v1",
        ),
        **{
            arm: (
                str(job[f"{arm}_transform_output"]),
                "round0051-transform-capability-v1",
            )
            for arm in ARMS
        },
    }
    cells: dict[str, Any] = {}
    coordinate_views: dict[str, RepresentativeArrayView] = {}
    for label, (root, schema) in coordinate_specs.items():
        full, record = _coordinate_stream(
            root,
            expected_schema=schema,
        )
        if label != "baseline_1p00" and record.get("arm") != label:
            raise Round0042PipelineError(
                f"R0051 coordinate arm mismatch: {label}"
            )
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
            "schema": "round0051-representative-cell-v1",
            "round_id": ROUND_ID,
            "cell": label,
            "negative_bce_multiplier": (
                BASELINE_MULTIPLIER
                if label == "baseline_1p00"
                else NEGATIVE_MULTIPLIERS[label]
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
        label="R0046 baseline train receipt",
    )
    baseline_comparison = _read_sealed(
        BASELINE_COMPARISON,
        label="R0046 source-exposure comparison",
    )
    trains = {
        arm: _read_sealed(
            os.path.join(
                job[f"{arm}_train_output"],
                "train-receipt.json",
            ),
            label=f"R0051 {arm} train receipt",
        )
        for arm in ARMS
    }
    baseline_runtime = baseline_train["exact_execution_receipt"]
    baseline_config = baseline_train["production_config"]
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
    matched: dict[str, bool] = {
        "r0046_baseline_classification_available": (
            baseline_comparison.get("classification") is not None
        ),
    }
    for arm in ARMS:
        config = configs[arm][0]
        train = trains[arm]
        runtime = train["exact_execution_receipt"]
        matched.update({
            f"{arm}_row_universe": (
                config["row_universe"]
                == baseline_config["row_universe"]
            ),
            f"{arm}_model": (
                config["model"] == baseline_config["model"]
            ),
            f"{arm}_optimizer_except_loss": (
                _optimizer_without_loss(config)
                == _optimizer_without_loss(baseline_config)
            ),
            f"{arm}_graph": (
                train["graph"] == baseline_train["graph"]
            ),
            f"{arm}_features": (
                train["train_accounting"]["verified_hashes"]["features"]
                == baseline_train["train_accounting"][
                    "verified_hashes"
                ]["features"]
            ),
            f"{arm}_seed": train["seed"] == baseline_train["seed"] == SEED,
            f"{arm}_successful_updates": (
                train["train_accounting"][
                    "positive_lr_optimizer_steps"
                ]
                == baseline_train["train_accounting"][
                    "positive_lr_optimizer_steps"
                ]
                == SUCCESSFUL_UPDATES
            ),
            f"{arm}_runtime": all(
                runtime.get(key) == baseline_runtime.get(key)
                for key in same_runtime_fields
            ),
            f"{arm}_loss_stamp": (
                train["loss_runtime_stamp"]
                == config["execution"]["expected_loss_stamp"]
            ),
        })
    baseline = cells["baseline_1p00"]
    deltas: dict[str, Any] = {}
    guards: dict[str, Any] = {}
    eligible: list[str] = []
    for arm in ARMS:
        scalar_delta = {
            key: (
                cells[arm]["scalars"][key]
                - baseline["scalars"][key]
            )
            for key in cells[arm]["scalars"]
        }
        projection_delta = {
            key: (
                cells[arm]["projection"][key]
                - baseline["projection"][key]
            )
            for key in ("proj_ffr", "proj_recall_at_10")
        }
        thresholds = configs[arm][0]["decision_thresholds"]
        arm_guards = {
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
        deltas[arm] = {
            "scalars": scalar_delta,
            "projection": projection_delta,
        }
        guards[arm] = arm_guards
        if all(arm_guards.values()):
            eligible.append(arm)
    numerical = {}
    for label, cell in cells.items():
        sealed = _read_sealed(
            cell["receipt"]["canonical_path"],
            label=f"R0051 {label} panel",
        )
        numerical[label] = (
            sealed["panel"]["guards"].get("coords_finite") is True
            and sealed["panel"]["guards"].get("coords_collapsed")
            is False
        )
    valid = (
        all(matched.values())
        and all(numerical.values())
        and all(
            trains[arm]["performance_profile"]["n_windows"] == 200
            for arm in ARMS
        )
    )
    if not valid:
        selection = "invalid-isolation"
    elif not eligible:
        selection = "retain-baseline-1p00"
    elif eligible == ["negative_0p25"]:
        selection = "negative-0p25-candidate"
    elif eligible == ["negative_0p50"]:
        selection = "negative-0p50-candidate"
    else:
        extra = (
            cells["negative_0p25"]["scalars"]["density"]
            - cells["negative_0p50"]["scalars"]["density"]
        )
        selection = (
            "negative-0p25-candidate"
            if extra >= configs["negative_0p25"][0][
                "decision_thresholds"
            ]["prefer_smaller_change_unless_extra_density_min"]
            else "negative-0p50-candidate"
        )
    render = _render_cells(
        output=output,
        selector=selector,
        coordinates=coordinate_views,
    )
    body = {
        "schema": "round0051-negative-bce-calibration-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "seed": SEED,
        "scientific_universe": {
            "rows": len(X),
            "row_namespace": (
                "compact ascending exact-fp16 representative global rows"
            ),
            "selector": selector.identity(),
        },
        "reference_receipt": reference_signature,
        "baseline_train": expected_input_signature(
            BASELINE_TRAIN_RECEIPT
        ),
        "baseline_comparison": expected_input_signature(
            BASELINE_COMPARISON
        ),
        "cells": cells,
        "treatment_minus_baseline": deltas,
        "quality_guards": guards,
        "matched_contract": matched,
        "numerical_guards": numerical,
        "eligible_treatments": eligible,
        "selection": selection,
        "interpretation": {
            "isolated_treatment": (
                "normalized negative BCE contribution at fixed sampled "
                "positive/negative composition"
            ),
            "external_ood_adoption_gate_run": False,
            "scale_claimed": False,
            "selected_treatment_is_candidate_only": (
                selection.endswith("-candidate")
            ),
        },
        "render": render,
        "wall_seconds": time.monotonic() - started,
    }
    receipt = _seal(body)
    path = os.path.join(
        output,
        "negative-bce-calibration-v1.json",
    )
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise RuntimeError("R0051 handler received another queue")
    selected = job if job is not None else active.get("job") or {}
    if len(selected.get("outputs") or []) != 1:
        raise RuntimeError("R0051 job output contract changed")
    handler = {
        "train": run_train,
        "transform": run_transform,
        "matched_panel": run_matched_panel,
    }.get(selected.get("action"))
    if handler is None:
        raise RuntimeError(
            f"unknown R0051 action: {selected.get('action')!r}"
        )
    return handler(active, selected)
