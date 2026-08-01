"""No-training functional showdown for the historical/current Jina recipes."""
from __future__ import annotations

import gc
import json
import os
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    ordered_array_sha256,
    sha256_bytes,
)
from basemap.output_safety import (
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0108_evaluation import seal, validate_seal
from basemap.round0134_functional_showdown import (
    CELL_ORDER,
    CURRENT_R0104_SEED42,
    CURRENT_RAW_SEED42,
    CURRENT_RAW_SEED43,
    DECISION_SCHEMA,
    HISTORICAL_SEED42,
    HISTORICAL_SEED43,
    PANEL_SCHEMA,
    ROUND_ID,
    Round0134Error,
    build_decision,
)
from experiments.round0027_nodes import _panel_config
from experiments.round0119_nodes import (
    SOURCE_DIMENSION,
    SOURCE_ROWS,
    _architecture,
    _authenticate_model,
    _read_json_signature,
)


TRANSFORM_BATCH_ROWS = 8_192
QUERY_ROWS = 20_000
RENDER_ROWS = 100_000
RENDER_SEED = 13_400


def _signature(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    path = str(expected.get("canonical_path") or "")
    actual = expected_input_signature(path)
    if actual != dict(expected):
        raise Round0134Error(f"{label} bytes changed")
    return actual


def _authenticate_r0104_fp16(
    spec: Mapping[str, Any], *, device: str = "cuda"
) -> dict[str, Any]:
    if (
        spec.get("key") != CURRENT_R0104_SEED42
        or spec.get("arm") != "fp16_control"
    ):
        raise Round0134Error("R0104 fp16 cell identity changed")
    train, train_signature = _read_json_signature(
        spec["train_receipt"], label="R0104 fp16 train receipt", sealed=True
    )
    config_receipt, config_signature = _read_json_signature(
        spec["production_config"],
        label="R0104 fp16 production config",
        sealed=False,
    )
    model_signature = _signature(spec["model"], label="R0104 fp16 model")
    config = config_receipt.get("config")
    if not isinstance(config, Mapping):
        raise Round0134Error("R0104 fp16 production config is missing")
    optimizer = config.get("optimizer")
    graph = config.get("graph")
    pipeline = train.get("exact_execution_receipt")
    accounting = train.get("train_accounting")
    checks = train.get("train_checks")
    if not all(
        isinstance(value, Mapping)
        for value in (optimizer, graph, pipeline, accounting, checks)
    ):
        raise Round0134Error("R0104 fp16 execution evidence is incomplete")
    config_sha256 = sha256_bytes(canonical_json(config))
    if (
        train.get("schema") != "round0104-paired-train-receipt-v2"
        or train.get("round_id") != "0104"
        or train.get("arm") != "fp16_control"
        or train.get("model") != model_signature
        or config_receipt.get("schema") != "round0104-production-config-v2"
        or config_receipt.get("config_sha256") != config_sha256
        or train.get("production_config_sha256") != config_sha256
        or optimizer.get("seed") != 42
        or optimizer.get("successful_positive_lr_updates") != 500_000
        or graph.get("k") != 50
        or graph.get("sampling")
        != "fuzzy-weight-proportional-with-replacement"
        or pipeline.get("pipeline") != "host_weighted_jina_paired"
        or pipeline.get("sampler_class") != "PairedHostWeightedJinaSampler"
        or pipeline.get("source_representation") != "fp16-control"
        or pipeline.get("feature_residency")
        != "host-mmap-fp16-source-shards"
        or pipeline.get("device_conversion")
        != "device-fp32-from-exact-fp16"
        or pipeline.get("positive_sampling") != "weighted_with_replacement"
        or pipeline.get("positive_with_replacement") is not True
        or pipeline.get("weighted_effective") is not True
        or accounting.get("optimizer_steps_attempted") != 500_000
        or accounting.get("optimizer_steps_succeeded") != 500_000
        or any(
            accounting.get(key) != 0
            for key in (
                "amp_overflow_skips",
                "nonfinite_loss_skips",
                "nonfinite_gradient_skips",
            )
        )
        or any(
            checks.get(key) is not True
            for key in (
                "endpoint_rows_match_updates",
                "exact_update_closure",
                "no_pipeline_stamp_drift",
                "zero_numerical_skips",
            )
        )
    ):
        raise Round0134Error("R0104 fp16 execution semantics changed")

    from basemap.pumap.parametric_umap import ParametricUMAP

    model = ParametricUMAP.load(model_signature["canonical_path"], device=device)
    expected_architecture = _architecture(config)
    observed_architecture = {
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
    if observed_architecture != expected_architecture:
        raise Round0134Error("R0104 fp16 checkpoint architecture changed")
    return {
        "model": model,
        "model_signature": model_signature,
        "train": train_signature,
        "production_config": config_signature,
        "seed": 42,
        "training_semantics": {
            "pipeline": pipeline.get("pipeline"),
            "sampler_class": pipeline.get("sampler_class"),
            "positive_sampling": pipeline.get("positive_sampling"),
            "feature_residency": pipeline.get("feature_residency"),
            "source_representation": pipeline.get("source_representation"),
            "device_conversion": pipeline.get("device_conversion"),
            "successful_updates": accounting.get("optimizer_steps_succeeded"),
        },
    }


def _load_frozen_query_truth(
    path: str,
    *,
    expected_key: str,
    expected_policy: Mapping[str, Any],
    expected_payload_sha256: str,
) -> dict[str, Any]:
    """Authenticate the reviewed R0037 truth without rebuilding it.

    ``panel_v2.load_query_truth`` deliberately requires the archived builder's
    ``cross_knn`` source hash to equal the *current* source hash.  That is the
    right guard for a truth archive being reused as current implementation
    evidence, but R0134's contract is different: it reuses the already reviewed
    R0037 neighbor bytes verbatim while scoring all maps with the current
    evaluator.  Later performance-only edits to ``cross_knn`` therefore cannot
    be allowed to make those frozen bytes unreadable.

    This narrow compatibility loader retains every content and semantic check:
    exact archive fields, complete key identity, the accepted R0037 policy,
    payload hash, shape, dtype, bounds, and per-row uniqueness.  It omits only
    the invalid comparison between a historical builder hash and current source
    text.  The expected policy and payload are themselves sealed in R0037's
    accepted shared-reference receipt.
    """
    from basemap.panel_v2 import QUERY_TRUTH_SCHEMA, _validate_truth_neighbors

    with np.load(path, allow_pickle=False) as archive:
        required = {
            "key",
            "k",
            "query_rows",
            "corpus_cardinality",
            "payload_sha256",
            "neighbors",
            "meta",
        }
        if set(archive.files) != required:
            raise Round0134Error("R0037 query truth archive fields changed")
        meta = json.loads(str(archive["meta"]))
        if not isinstance(meta, dict) or set(meta) != {
            "schema",
            "key_parts",
            "build_wall_s",
        }:
            raise Round0134Error("R0037 query truth metadata changed")
        key_parts = meta.get("key_parts")
        if (
            meta.get("schema") != QUERY_TRUTH_SCHEMA
            or not isinstance(key_parts, dict)
            or key_parts.get("schema") != QUERY_TRUTH_SCHEMA
            or key_parts.get("policy") != dict(expected_policy)
        ):
            raise Round0134Error("R0037 query truth policy changed")
        key = str(archive["key"])
        if (
            key != expected_key
            or sha256_bytes(canonical_json(key_parts)) != expected_key
        ):
            raise Round0134Error("R0037 query truth complete identity changed")
        k = int(archive["k"])
        query_rows = int(archive["query_rows"])
        corpus_cardinality = int(archive["corpus_cardinality"])
        neighbors = np.asarray(archive["neighbors"])
        try:
            _validate_truth_neighbors(
                neighbors,
                k=k,
                query_rows=query_rows,
                corpus_cardinality=corpus_cardinality,
            )
        except ValueError as exc:
            raise Round0134Error("R0037 query truth neighbors changed") from exc
        payload_sha256 = str(archive["payload_sha256"])
        if (
            payload_sha256 != expected_payload_sha256
            or ordered_array_sha256(neighbors) != expected_payload_sha256
        ):
            raise Round0134Error("R0037 query truth payload changed")
        return {
            "schema": QUERY_TRUTH_SCHEMA,
            "key": key,
            "key_parts": key_parts,
            "k": k,
            "query_rows": query_rows,
            "corpus_cardinality": corpus_cardinality,
            "neighbors": neighbors.copy(),
            "payload_sha256": payload_sha256,
            "build_wall_s": meta.get("build_wall_s"),
            "historical_builder_policy_authenticated": True,
            "current_builder_source_hash_required": False,
        }


def _load_reference(job: Mapping[str, Any]):
    from basemap.panel_v2 import load_hiD_reference

    shared, shared_signature = _read_json_signature(
        job["shared_reference_receipt"],
        label="R0037 shared functional reference receipt",
        sealed=True,
    )
    for key in ("high_d_reference", "query_truth", "query_embeddings"):
        if _signature(shared[key], label=f"R0037 {key}") != job[key]:
            raise Round0134Error(f"R0037 {key} queue binding changed")
    reference = load_hiD_reference(
        shared["high_d_reference"]["canonical_path"],
        expected_key=shared["high_d_reference_key"],
    )
    truth = _load_frozen_query_truth(
        shared["query_truth"]["canonical_path"],
        expected_key=shared["query_truth_key"],
        expected_policy=shared["query_truth_exactness"],
        expected_payload_sha256=shared["query_truth_payload_sha256"],
    )
    if (
        truth["neighbors"].shape != (QUERY_ROWS, 10)
        or truth["payload_sha256"] != shared["query_truth_payload_sha256"]
    ):
        raise Round0134Error("R0037 held-out truth changed")
    centroids = {
        int(key): np.load(
            _signature(value, label=f"R0037 centroid {key}")["canonical_path"],
            mmap_mode="r",
            allow_pickle=False,
        )
        for key, value in job["centroids"].items()
    }
    return shared, shared_signature, reference, truth, centroids


def _load_shared_evaluation_inputs(
    job: Mapping[str, Any],
) -> tuple[dict[str, Any], Any, np.ndarray]:
    """Load the exact R0037 scoring views, not just their storage arrays."""
    source_signature = _signature(job["source"], label="R0037 2M source")
    source_storage = np.load(
        source_signature["canonical_path"], mmap_mode="r", allow_pickle=False
    )
    if (
        source_storage.shape != (SOURCE_ROWS, SOURCE_DIMENSION)
        or source_storage.dtype != np.dtype("<f2")
        or not source_storage.flags.c_contiguous
    ):
        raise Round0134Error("R0037 2M source shape/dtype changed")

    # R0037's frozen high-D reference and historical transforms used the
    # full-768 PrefixL2NormalizedArray view.  At 768 dimensions it does not
    # renormalize, but it does expose fp32 scoring slices and the exact source
    # path identity required by the shared-reference key.  Passing the raw fp16
    # memmap would be numerically close but would not be the registered view.
    from basemap.round0027_program import input_array

    source = input_array(
        SOURCE_DIMENSION, path=source_signature["canonical_path"]
    )
    queries = np.load(
        job["query_embeddings"]["canonical_path"],
        mmap_mode="r",
        allow_pickle=False,
    )
    # R0037 intentionally materialized held-out queries as fp32 scoring rows;
    # this is sealed in the 61,440,128-byte query artifact and its truth key.
    if (
        queries.shape != (QUERY_ROWS, SOURCE_DIMENSION)
        or queries.dtype != np.dtype("<f4")
        or not queries.flags.c_contiguous
    ):
        raise Round0134Error("R0037 held-out query shape/dtype changed")
    return source_signature, source, queries


def _projection_metrics(
    *, coordinates: np.ndarray, query_coordinates: np.ndarray, truth: Mapping[str, Any]
) -> dict[str, Any]:
    from basemap.panel_v2 import (
        cross_knn,
        ffr_from_neighbors,
        recall_at_k_from_neighbors,
    )

    cfg = _panel_config()
    k_fraction = max(cfg.k_hit, int(np.ceil(cfg.frac * len(coordinates))))
    low = cross_knn(
        query_coordinates, coordinates, k_fraction, cfg, hi_dim=False
    )
    high = np.asarray(truth["neighbors"], dtype=np.int64)[:, : cfg.k_hit]
    return {
        "ffr": round(ffr_from_neighbors(high, low, cfg.k_hit), 4),
        "recall_at_10": round(
            recall_at_k_from_neighbors(high, low, cfg.k_hit), 5
        ),
        "queries": QUERY_ROWS,
        "k_fraction": k_fraction,
        "truth_payload_sha256": truth["payload_sha256"],
        "semantic_role": (
            "held-out R0037 2M-to-4M prefix projection; recall_at_10 is the "
            "registered OOD recall metric"
        ),
    }


def _render(
    *, output: str, coordinate_paths: Mapping[str, str], labels: np.ndarray
) -> dict[str, Any]:
    render_root = create_fresh_directory(
        os.path.join(output, "renders"), label="R0134 side-by-side renders"
    )
    rng = np.random.RandomState(RENDER_SEED)
    sample = np.sort(rng.choice(SOURCE_ROWS, RENDER_ROWS, replace=False))
    sample_path = os.path.join(render_root, "sample-row-ids.npy")
    atomic_save_new_npy(sample_path, sample, immutable=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(CELL_ORDER), figsize=(25, 5), dpi=140)
    titles = {
        HISTORICAL_SEED42: "historical R0037 s42",
        HISTORICAL_SEED43: "historical R0038 s43",
        CURRENT_R0104_SEED42: "current R0104 fp16 s42",
        CURRENT_RAW_SEED42: "current R0115 raw s42",
        CURRENT_RAW_SEED43: "current R0117 raw s43",
    }
    limits: dict[str, Any] = {}
    color = np.asarray(labels[sample] % 20, dtype=np.int16)
    for axis, key in zip(axes, CELL_ORDER, strict=True):
        coordinates = np.load(coordinate_paths[key], mmap_mode="r", allow_pickle=False)
        points = np.asarray(coordinates[sample], dtype=np.float32)
        low = np.quantile(points, 0.001, axis=0)
        high = np.quantile(points, 0.999, axis=0)
        pad = np.maximum((high - low) * 0.03, 1.0e-6)
        axis.scatter(
            points[:, 0], points[:, 1], c=color, cmap="tab20", s=0.18,
            alpha=0.35, linewidths=0, rasterized=True,
        )
        axis.set_xlim(float(low[0] - pad[0]), float(high[0] + pad[0]))
        axis.set_ylim(float(low[1] - pad[1]), float(high[1] + pad[1]))
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(titles[key], fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
        limits[key] = {
            "quantile_low": low.tolist(),
            "quantile_high": high.tolist(),
            "axis_padding_fraction": 0.03,
        }
    figure.suptitle(
        "R0134: identical 2M rows and 100k sample; per-cell 0.1–99.9% axes",
        fontsize=11,
    )
    figure.tight_layout()
    png_path = os.path.join(render_root, "historical-vs-current-2m.png")
    figure.savefig(png_path, bbox_inches="tight")
    plt.close(figure)
    os.chmod(png_path, 0o444)
    receipt = seal(
        {
            "schema": "round0134-side-by-side-render-v1",
            "round_id": ROUND_ID,
            "sample": expected_input_signature(sample_path),
            "sample_seed": RENDER_SEED,
            "sample_rows": RENDER_ROWS,
            "same_ordered_rows_in_every_cell": True,
            "color": "frozen R0037 k256 label modulo 20; diagnostic only",
            "axes": "per-cell robust quantiles; diagnostic only",
            "limits": limits,
            "render": expected_input_signature(png_path),
        }
    )
    manifest_path = os.path.join(render_root, "render-manifest.json")
    atomic_write_new_json(manifest_path, receipt, immutable=True)
    return {**receipt, "manifest": expected_input_signature(manifest_path)}


def run_panel(active: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0134 functional showdown panel"
    )
    started = time.monotonic()
    source_signature, source, queries = _load_shared_evaluation_inputs(job)
    (
        shared,
        shared_signature,
        reference,
        truth,
        centroids,
    ) = _load_reference(job)
    from basemap.panel_v2 import reset_process_cuda_peak, score_panel

    reset_process_cuda_peak()
    specs = {spec["key"]: spec for spec in job["model_bundles"]}
    if tuple(specs) != CELL_ORDER:
        raise Round0134Error("R0134 model bundles are missing or reordered")
    coordinate_root = create_fresh_directory(
        os.path.join(output, "coordinates"), label="R0134 current coordinates"
    )
    cells: dict[str, Any] = {}
    coordinate_paths: dict[str, str] = {}
    for key in CELL_ORDER:
        spec = specs[key]
        if key in (HISTORICAL_SEED42, HISTORICAL_SEED43):
            bundle = _authenticate_model(spec)
            # Historical coordinates are already reviewed artifacts.  Loading
            # the model above authenticates the exact training bundle; no
            # transform is repeated for this cell.
            del bundle["model"]
            frozen = job["frozen_coordinates"][key]
            coordinate_signature = _signature(
                frozen["coordinates"], label=f"{key} frozen coordinates"
            )
            query_signature = _signature(
                frozen["query_coordinates"], label=f"{key} frozen query coordinates"
            )
            bundle_receipt = {
                "model": _signature(spec["model"], label=f"{key} model"),
                "train": _signature(spec["train_receipt"], label=f"{key} train"),
                "production_config": _signature(
                    spec["production_config"], label=f"{key} production config"
                ),
                "seed": spec["seed"],
                "training_semantics": spec["semantic_contract"],
            }
        else:
            bundle = (
                _authenticate_r0104_fp16(spec)
                if key == CURRENT_R0104_SEED42
                else _authenticate_model(spec)
            )
            model = bundle["model"]
            coordinates = np.asarray(
                model.transform(source, batch_size=TRANSFORM_BATCH_ROWS),
                dtype=np.float32,
            )
            query_coordinates = np.asarray(
                model.transform(queries, batch_size=TRANSFORM_BATCH_ROWS),
                dtype=np.float32,
            )
            if (
                coordinates.shape != (SOURCE_ROWS, 2)
                or query_coordinates.shape != (QUERY_ROWS, 2)
                or not np.isfinite(coordinates).all()
                or not np.isfinite(query_coordinates).all()
            ):
                raise Round0134Error(f"{key} transform is malformed")
            cell_root = create_fresh_directory(
                os.path.join(coordinate_root, key), label=f"R0134 {key} coordinates"
            )
            coordinate_path = os.path.join(cell_root, "coordinates.npy")
            query_path = os.path.join(cell_root, "query-coordinates.npy")
            atomic_save_new_npy(coordinate_path, coordinates, immutable=True)
            atomic_save_new_npy(query_path, query_coordinates, immutable=True)
            coordinate_signature = expected_input_signature(coordinate_path)
            query_signature = expected_input_signature(query_path)
            bundle_receipt = {
                "model": bundle["model_signature"],
                "train": bundle["train"],
                "production_config": bundle["production_config"],
                "seed": bundle["seed"],
                "training_semantics": (
                    bundle.get("training_semantics")
                    or bundle.get("authenticated_training_semantics")
                ),
            }
            del model, bundle["model"]
        coordinates = np.load(
            coordinate_signature["canonical_path"], mmap_mode="r", allow_pickle=False
        )
        query_coordinates = np.load(
            query_signature["canonical_path"], mmap_mode="r", allow_pickle=False
        )
        panel = score_panel(
            source,
            coordinates,
            config=_panel_config(),
            centroids_by_k=centroids,
            hiD_reference=reference,
            scale_admission=None,
            provenance={
                "round_id": ROUND_ID,
                "cell": key,
                "release_sha": active["manifest"]["release_sha"],
                "source": source_signature,
                "coordinates": coordinate_signature,
                "shared_reference_receipt": shared_signature,
            },
        )
        projection = _projection_metrics(
            coordinates=coordinates,
            query_coordinates=query_coordinates,
            truth=truth,
        )
        if (
            panel.get("guards", {}).get("coords_finite") is not True
            or panel.get("guards", {}).get("coords_collapsed") is not False
            or panel.get("purity", {}).get("k256") is None
            or panel.get("purity", {}).get("k1024") is None
        ):
            raise Round0134Error(f"{key} functional panel guards failed")
        cells[key] = {
            "seed": bundle_receipt["seed"],
            "role": "historical" if key.startswith("historical_") else "current",
            "training": bundle_receipt,
            "coordinates": coordinate_signature,
            "query_coordinates": query_signature,
            "coordinates_ordered_sha256": ordered_array_sha256(coordinates),
            "query_coordinates_ordered_sha256": ordered_array_sha256(query_coordinates),
            "panel": panel,
            "projection": projection,
        }
        coordinate_paths[key] = coordinate_signature["canonical_path"]
        del coordinates, query_coordinates, bundle
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    render = _render(
        output=output,
        coordinate_paths=coordinate_paths,
        labels=np.asarray(reference["labels"][256], dtype=np.int32),
    )
    receipt = seal(
        {
            "schema": PANEL_SCHEMA,
            "round_id": ROUND_ID,
            "release_sha": active["manifest"]["release_sha"],
            "training_performed": False,
            "source": source_signature,
            "source_rows": SOURCE_ROWS,
            "same_ordered_source_for_every_cell": True,
            "shared_reference_receipt": shared_signature,
            "high_d_reference": job["high_d_reference"],
            "query_truth": job["query_truth"],
            "query_embeddings": job["query_embeddings"],
            "functional_metrics": [
                "ffr",
                "purity k256",
                "purity k1024",
                "projection FFR",
                "held-out OOD recall@10",
            ],
            "density_role": "not recomputed and not a selector input",
            "cells": cells,
            "render": render,
            "wall_seconds": time.monotonic() - started,
            "map_registry_state_changed": False,
        }
    )
    path = os.path.join(output, "functional-showdown.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_decision(active: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0134 functional showdown decision"
    )
    panel_path = os.path.join(str(job["panel_output"]), "functional-showdown.json")
    with open(panel_path, encoding="utf-8") as handle:
        panel = json.load(handle)
    validate_seal(panel, label="R0134 functional showdown panel")
    recovery = job.get("recovery_kind") is not None
    if recovery:
        panel_signature = _signature(
            job["panel_receipt"], label="R0134 immutable recovery panel"
        )
        if (
            panel_signature["canonical_path"] != panel_path
            or panel.get("release_sha") != job.get("panel_release_sha")
            or job.get("recovery_kind")
            != "cpu-decision-from-immutable-attempt-3-panel"
        ):
            raise Round0134Error("R0134 recovery panel lineage changed")
    elif panel.get("release_sha") != active["manifest"]["release_sha"]:
        raise Round0134Error("R0134 panel/decision release changed")
    if panel.get("schema") != PANEL_SCHEMA or panel.get("round_id") != ROUND_ID:
        raise Round0134Error("R0134 panel identity changed")
    decision = build_decision(panel["cells"])
    receipt = seal(
        {
            **decision,
            "release_sha": active["manifest"]["release_sha"],
            "panel": expected_input_signature(panel_path),
            "panel_release_sha": panel.get("release_sha"),
            "decision_recovery": recovery,
            "capability": "jina-density-functional-showdown-v1",
            "next_branch": (
                "density-v3-current-recipe-calibration-and-frozen-25m-replays"
                if decision["density_v3_calibration_authorized"]
                else "single-factor-fuzzy-graph-and-sampler-bridges"
            ),
        }
    )
    path = os.path.join(output, "decision.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_job(
    active: dict[str, Any], job: dict[str, Any] | None = None
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID or job is None:
        raise Round0134Error("R0134 handler requires its exact round/job")
    action = str(job.get("action") or "")
    if action == "functional_showdown_panel":
        return run_panel(active, job)
    if action == "functional_showdown_decision":
        return run_decision(active, job)
    raise Round0134Error(f"unknown R0134 action: {action!r}")
