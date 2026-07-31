"""R0122 no-training provenance/representation density bridge."""
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
    atomic_save_new_npz,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0108_evaluation import (
    K_DENSITY,
    TRANSFORM_BATCH_ROWS,
    seal,
    validate_seal,
)
from experiments.round0085_nodes import density_v2_calibration
from experiments.round0108_nodes import _panel_config
from experiments.round0119_nodes import (
    ANCHORS,
    CALIBRATION_SCHEMA,
    CELL_ORDER as R0119_CELL_ORDER,
    DECISION_SCHEMA as R0119_DECISION_SCHEMA,
    REFERENCE_SCHEMA,
    REPRESENTATIVE_ROWS,
    SCORE_SCHEMA as R0119_SCORE_SCHEMA,
    SOURCE_DIMENSION,
    SOURCE_ROWS,
    _architecture,
    _authenticate_model as _authenticate_r0119_model,
    _load_universe,
    _read_json_signature,
)


ROUND_ID = "0122"
R0119_RELEASE_SHA = "2097703a6209c67ccee8a728d931f52a09543009"
R0119_PANEL_SHA256 = (
    "3bfc8c535a374e16577758d983ae24e4a990078ff28f6b15410a97a4a6fb9dd7"
)
R0119_DECISION_SHA256 = (
    "26b8550c415ef0e1eac3af6a05abc5e32bdea657a1a23d1ab27a8909d5fcead1"
)
R0104_MODEL_SHA256 = {
    "r0104_fp16_seed42_full_transform": (
        "36a7fb86784b6a891f7c73b83d008aead320a7729eea913efc117e4bcd5b3e08"
    ),
    "r0104_int8_seed42_full_transform": (
        "e485cd961853c55fb9838a7c1fa1f8dd078c0338e1fa3dc6e3c6d058900af9d4"
    ),
}
NEW_CELL_ORDER = (
    "r0104_fp16_seed42_full_transform",
    "r0104_int8_seed42_full_transform",
    "r0115_raw_seed42_full_transform",
    "r0117_raw_seed43_full_transform",
)
R0119_REUSED_CELL_ORDER = (
    "historical_2m_seed42",
    "historical_2m_seed43",
    "current_2m_seed42",
    "current_2m_seed43",
)
SCORE_SCHEMA = "round0122-jina-density-provenance-bridge-panel-v1"
DECISION_SCHEMA = "round0122-jina-density-provenance-bridge-decision-v1"
REGISTERED_FLOOR = 0.17589389755990817


class Round0122Error(RuntimeError):
    """Raised when R0122's registered evidence contract changes."""


def _authenticate_r0104_model(
    spec: Mapping[str, Any],
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    key = str(spec.get("key") or "")
    arm = str(spec.get("arm") or "")
    if key not in R0104_MODEL_SHA256 or arm not in {
        "fp16_control",
        "int8_treatment",
    }:
        raise Round0122Error("R0104 model cell identity changed")
    train, train_signature = _read_json_signature(
        spec["train_receipt"],
        label=f"{key} train receipt",
        sealed=True,
    )
    config_receipt, config_signature = _read_json_signature(
        spec["production_config"],
        label=f"{key} production config",
        sealed=False,
    )
    model_signature = expected_input_signature(
        str(spec["model"]["canonical_path"])
    )
    if (
        model_signature != dict(spec["model"])
        or model_signature["sha256"] != R0104_MODEL_SHA256[key]
    ):
        raise Round0122Error(f"{key} model bytes changed")

    config = config_receipt.get("config")
    if not isinstance(config, Mapping):
        raise Round0122Error(f"{key} production config is missing")
    config_sha256 = sha256_bytes(canonical_json(config))
    optimizer = config.get("optimizer")
    graph = config.get("graph")
    preprocessing = config.get("input_preprocessing")
    paired = config.get("paired_invariant")
    pipeline = train.get("exact_execution_receipt")
    accounting = train.get("train_accounting")
    checks = train.get("train_checks")
    if not all(
        isinstance(value, Mapping)
        for value in (
            optimizer,
            graph,
            preprocessing,
            paired,
            pipeline,
            accounting,
            checks,
        )
    ):
        raise Round0122Error(f"{key} execution evidence is incomplete")

    expected_representation = (
        "fp16-control" if arm == "fp16_control" else "int8-treatment"
    )
    expected_residency = (
        "host-mmap-fp16-source-shards"
        if arm == "fp16_control"
        else "host-ram-int8-plus-fp16-scale"
    )
    expected_conversion_key = (
        "device_conversion" if arm == "fp16_control" else "dequantization"
    )
    expected_conversion = (
        "device-fp32-from-exact-fp16"
        if arm == "fp16_control"
        else "device-fp32-int8-times-exact-row-fp16-scale"
    )
    if (
        train.get("schema") != "round0104-paired-train-receipt-v2"
        or train.get("round_id") != "0104"
        or train.get("arm") != arm
        or train.get("model") != model_signature
        or config_receipt.get("schema") != "round0104-production-config-v2"
        or config_receipt.get("config_sha256") != config_sha256
        or train.get("production_config_sha256") != config_sha256
        or config.get("schema")
        != "round0104-self-contained-paired-train-config-v2"
        or config.get("arm") != arm
        or optimizer.get("seed") != 42
        or optimizer.get("successful_positive_lr_updates") != 500_000
        or optimizer.get("batch_size") != 8_192
        or graph.get("k") != 50
        or graph.get("sampling")
        != "fuzzy-weight-proportional-with-replacement"
        or graph.get("positive_target_mode") != "binary"
        or preprocessing.get("source_rows") != [0, SOURCE_ROWS]
        or preprocessing.get("source_dimension") != SOURCE_DIMENSION
        or paired.get("rows") != SOURCE_ROWS
        or paired.get("seed") != 42
        or paired.get("successful_positive_lr_updates") != 500_000
        or pipeline.get("pipeline") != "host_weighted_jina_paired"
        or pipeline.get("sampler_class")
        != "PairedHostWeightedJinaSampler"
        or pipeline.get("positive_sampling") != "weighted_with_replacement"
        or pipeline.get("positive_with_replacement") is not True
        or pipeline.get("weighted_effective") is not True
        or pipeline.get("source_representation") != expected_representation
        or pipeline.get("feature_residency") != expected_residency
        or pipeline.get(expected_conversion_key) != expected_conversion
        or accounting.get("optimizer_steps_attempted") != 500_000
        or accounting.get("optimizer_steps_succeeded") != 500_000
        or accounting.get("pipeline_pipeline")
        != "host_weighted_jina_paired"
        or accounting.get("pipeline_sampler_class")
        != "PairedHostWeightedJinaSampler"
        or accounting.get("pipeline_positive_sampling")
        != "weighted_with_replacement"
        or accounting.get("pipeline_source_representation")
        != expected_representation
        or accounting.get("pipeline_feature_residency")
        != expected_residency
        or accounting.get(f"pipeline_{expected_conversion_key}")
        != expected_conversion
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
        raise Round0122Error(f"{key} execution semantics changed")

    expected_architecture = _architecture(config)
    from basemap.pumap.parametric_umap import ParametricUMAP

    model = ParametricUMAP.load(
        model_signature["canonical_path"], device=device
    )
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
        raise Round0122Error(f"{key} checkpoint architecture changed")
    return {
        "model": model,
        "model_signature": model_signature,
        "train_receipt": train_signature,
        "production_config": config_signature,
        "key": key,
        "arm": arm,
        "seed": 42,
        "training_population": (
            "R0104 attempt-3 first 2,000,000 R0103 FineWeb rows"
        ),
        "training_representation": expected_representation,
        "training_graph": "R0104 attempt-3 shared fuzzy k50 graph",
        "training_sampler": "PairedHostWeightedJinaSampler",
        "training_updates": 500_000,
    }


def _score_full_transform(
    *,
    key: str,
    bundle: Mapping[str, Any],
    source: np.ndarray,
    retained_global_rows: np.ndarray,
    anchors: np.ndarray,
    high_radius: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Transform every source row, then select the exact R0040 representatives."""
    from basemap.panel_v2 import _self_knn

    transformed = np.asarray(
        bundle["model"].transform(
            source,
            batch_size=TRANSFORM_BATCH_ROWS,
        ),
        dtype=np.float32,
    )
    if (
        transformed.shape != (SOURCE_ROWS, 2)
        or not np.isfinite(transformed).all()
    ):
        raise Round0122Error(f"{key} full transform is malformed")
    coordinates = np.asarray(
        transformed[retained_global_rows], dtype=np.float32
    )
    del transformed
    if (
        coordinates.shape != (REPRESENTATIVE_ROWS, 2)
        or not np.isfinite(coordinates).all()
    ):
        raise Round0122Error(f"{key} representative selection is malformed")

    config = _panel_config(anchors=len(anchors))
    _, distances, guard = _self_knn(
        coordinates,
        anchors,
        K_DENSITY,
        config,
        hi_dim=False,
        want_dist=True,
        exact=True,
    )
    low_radius = np.asarray(distances.mean(1), dtype=np.float64)
    summary, bootstrap, null = density_v2_calibration(
        high_radius,
        low_radius,
        bootstrap_draws=1_000,
        bootstrap_seed=10_801,
        null_draws=1_000,
        null_seed=10_802,
    )
    cell = {
        "key": key,
        "seed": bundle["seed"],
        "arm": bundle.get("arm", "raw"),
        "training_population": bundle["training_population"],
        "training_representation": bundle["training_representation"],
        "training_graph": bundle["training_graph"],
        "training_sampler": bundle["training_sampler"],
        "training_updates": bundle["training_updates"],
        "train_receipt": bundle["train_receipt"],
        "production_config": bundle["production_config"],
        "model": bundle["model_signature"],
        "evaluation_source": (
            "R0040 FineWeb source; this need not equal the model's training rows"
        ),
        "transform_batch_rows": TRANSFORM_BATCH_ROWS,
        "transform_input_rows": SOURCE_ROWS,
        "transform_selection_after_transform": True,
        "transform_selected_rows": REPRESENTATIVE_ROWS,
        "coordinates": {
            "rows": len(coordinates),
            "dtype": coordinates.dtype.str,
            "ordered_sha256": ordered_array_sha256(coordinates),
            "axis_standard_deviation": coordinates.std(0).tolist(),
            "finite": True,
        },
        "density_v2": summary,
        "low_dim_exact_search_guard": guard,
        "clears_unchanged_registered_floor": (
            float(summary["correlation"]) >= REGISTERED_FLOOR
        ),
    }
    arrays = {
        f"{key}__low_radius": low_radius,
        f"{key}__bootstrap": bootstrap,
        f"{key}__permuted_null": null,
    }
    return cell, arrays


def _r0119_evidence(
    job: Mapping[str, Any],
    *,
    lineage: Mapping[str, Any],
    anchors: np.ndarray,
    global_rows: np.ndarray,
    high_radius: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    panel, panel_signature = _read_json_signature(
        job["r0119_panel"],
        label="R0119 density localization panel",
        sealed=True,
    )
    decision, decision_signature = _read_json_signature(
        job["r0119_decision"],
        label="R0119 density localization decision",
        sealed=True,
    )
    cells = panel.get("cells")
    expected_scorer = {
        "metric": "density-v2 radius correlation",
        "k": K_DENSITY,
        "transform_batch_rows": TRANSFORM_BATCH_ROWS,
        "historical_full_source_transform_before_selection": True,
        "low_dim_search": "exact",
        "bootstrap_draws": 1_000,
        "bootstrap_seed": 10_801,
        "null_draws": 1_000,
        "null_seed": 10_802,
        "registered_floor": REGISTERED_FLOOR,
        "floor_changed_or_tuned": False,
    }
    universe = panel.get("universe")
    if (
        panel_signature["sha256"] != R0119_PANEL_SHA256
        or decision_signature["sha256"] != R0119_DECISION_SHA256
        or panel.get("schema") != R0119_SCORE_SCHEMA
        or panel.get("round_id") != "0119"
        or panel.get("release_sha") != R0119_RELEASE_SHA
        or panel.get("lineage") != dict(lineage)
        or panel.get("scorer") != expected_scorer
        or not isinstance(universe, Mapping)
        or universe.get("source_rows") != SOURCE_ROWS
        or universe.get("representative_rows") != REPRESENTATIVE_ROWS
        or universe.get("anchors") != ANCHORS
        or universe.get("anchor_compact_rows_sha256")
        != ordered_array_sha256(anchors)
        or universe.get("anchor_global_rows_sha256")
        != ordered_array_sha256(global_rows)
        or universe.get("high_radius_sha256")
        != ordered_array_sha256(high_radius)
        or not isinstance(cells, Mapping)
        or set(cells) != set(R0119_CELL_ORDER)
        or decision.get("schema") != R0119_DECISION_SCHEMA
        or decision.get("round_id") != "0119"
        or decision.get("release_sha") != R0119_RELEASE_SHA
        or decision.get("score") != panel_signature
        or decision.get("outcome") != "failure-not-unique-to-25m-tuple"
        or decision.get("training_performed") is not False
    ):
        raise Round0122Error("R0119 accepted density evidence changed")
    reused = {
        key: cells[key]
        for key in R0119_REUSED_CELL_ORDER
    }
    return (
        {
            "panel": panel_signature,
            "decision": decision_signature,
            "historical_and_direct_cells": reused,
        },
        cells,
    )


def _replay_bundle(
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    bundle = _authenticate_r0119_model(spec)
    return {
        **bundle,
        "train_receipt": bundle["train"],
        "training_sampler": bundle[
            "authenticated_training_semantics"
        ]["sampler_class"],
        "training_updates": bundle[
            "authenticated_training_semantics"
        ]["successful_updates"],
    }


def run_score(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0122 density bridge panel"
    )
    started = time.monotonic()
    (
        source,
        representatives,
        retained_global_rows,
        anchors,
        global_rows,
        high_radius,
        lineage,
        _reference,
    ) = _load_universe(job)
    r0119, r0119_cells = _r0119_evidence(
        job,
        lineage=lineage,
        anchors=anchors,
        global_rows=global_rows,
        high_radius=high_radius,
    )
    r0104_specs = job.get("r0104_model_bundles")
    replay_specs = job.get("r0119_replay_model_bundles")
    if (
        not isinstance(r0104_specs, list)
        or [item.get("key") for item in r0104_specs]
        != list(NEW_CELL_ORDER[:2])
        or not isinstance(replay_specs, list)
        or [item.get("key") for item in replay_specs]
        != ["current_2m_seed42", "current_2m_seed43"]
    ):
        raise Round0122Error("registered R0122 model-cell order changed")
    for spec, expected_r0119_key in zip(
        replay_specs,
        ("current_2m_seed42", "current_2m_seed43"),
        strict=True,
    ):
        r0119_cell = r0119_cells[expected_r0119_key]
        if any(
            dict(spec[field]) != r0119_cell[field]
            for field in ("train_receipt", "production_config", "model")
        ):
            raise Round0122Error(
                f"{expected_r0119_key} replay bundle changed from R0119"
            )

    cells: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {
        "anchor_compact_rows": anchors,
        "anchor_global_rows": global_rows,
        "high_radius": high_radius,
    }
    bundles = [
        *[
            _authenticate_r0104_model(spec)
            for spec in r0104_specs
        ],
        *[
            _replay_bundle(spec)
            for spec in replay_specs
        ],
    ]
    keyed_bundles = dict(zip(NEW_CELL_ORDER, bundles, strict=True))
    for key in NEW_CELL_ORDER:
        bundle = keyed_bundles[key]
        cell, cell_arrays = _score_full_transform(
            key=key,
            bundle=bundle,
            source=source,
            retained_global_rows=retained_global_rows,
            anchors=anchors,
            high_radius=high_radius,
        )
        cells[key] = cell
        arrays.update(cell_arrays)
        del bundle["model"]
        gc.collect()

    arrays_path = os.path.join(output, "density-bridge-arrays.npz")
    atomic_save_new_npz(arrays_path, immutable=True, **arrays)
    receipt = seal({
        "schema": SCORE_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "lineage": lineage,
        "r0119_reused_evidence": r0119,
        "universe": {
            "name": (
                "exact R0119/R0040 FineWeb representative universe, "
                "anchors, high-D radii, family filter, and floor"
            ),
            "source_rows": SOURCE_ROWS,
            "representative_rows": REPRESENTATIVE_ROWS,
            "anchors": len(anchors),
            "anchor_compact_rows_sha256": ordered_array_sha256(anchors),
            "anchor_global_rows_sha256": ordered_array_sha256(global_rows),
            "high_radius_sha256": ordered_array_sha256(high_radius),
        },
        "scorer": {
            "metric": "density-v2 radius correlation",
            "k": K_DENSITY,
            "low_dim_search": "exact",
            "bootstrap_draws": 1_000,
            "bootstrap_seed": 10_801,
            "null_draws": 1_000,
            "null_seed": 10_802,
            "registered_floor": REGISTERED_FLOOR,
            "floor_changed_or_tuned": False,
        },
        "new_cells": cells,
        "native_r0115_context": {
            "reported_density": 0.2304,
            "numerically_clears_r0119_floor": 0.2304 >= REGISTERED_FLOOR,
            "same_matched_r0040_universe_and_scorer": False,
            "interpretation": (
                "the accepted R0115 native panel passes numerically; "
                "R0122 is calibration/representation-transfer localization, "
                "not proof of bad native training geometry"
            ),
        },
        "arrays": expected_input_signature(arrays_path),
        "training_performed": False,
        "single_factor_cause_claimed": False,
        "wall_seconds": time.monotonic() - started,
    })
    path = os.path.join(output, "density-bridge-panel.json")
    atomic_write_new_json(path, receipt, immutable=True)
    del source, representatives, retained_global_rows
    gc.collect()
    return {**receipt, "receipt": expected_input_signature(path)}


def run_decision(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0122 density bridge decision"
    )
    score_path = os.path.join(
        str(job["score_output"]), "density-bridge-panel.json"
    )
    with open(score_path, encoding="utf-8") as handle:
        score = json.load(handle)
    validate_seal(score, label="R0122 density bridge panel")
    new_cells = score.get("new_cells")
    reused = (
        score.get("r0119_reused_evidence") or {}
    ).get("historical_and_direct_cells")
    if (
        score.get("schema") != SCORE_SCHEMA
        or score.get("round_id") != ROUND_ID
        or score.get("release_sha") != active["manifest"]["release_sha"]
        or not isinstance(new_cells, Mapping)
        or tuple(new_cells) != NEW_CELL_ORDER
        or not isinstance(reused, Mapping)
        or set(reused) != set(R0119_REUSED_CELL_ORDER)
        or score.get("training_performed") is not False
    ):
        raise Round0122Error("R0122 density bridge panel changed")

    replay_pairs = {
        "seed42": (
            "current_2m_seed42",
            "r0115_raw_seed42_full_transform",
        ),
        "seed43": (
            "current_2m_seed43",
            "r0117_raw_seed43_full_transform",
        ),
    }
    replay_checks: dict[str, Any] = {}
    for seed, (direct_key, replay_key) in replay_pairs.items():
        direct_pass = bool(
            reused[direct_key]["clears_unchanged_registered_floor"]
        )
        replay_pass = bool(
            new_cells[replay_key]["clears_unchanged_registered_floor"]
        )
        replay_checks[seed] = {
            "r0119_direct_key": direct_key,
            "full_transform_replay_key": replay_key,
            "direct_clears_floor": direct_pass,
            "full_transform_clears_floor": replay_pass,
            "classification_changed": direct_pass != replay_pass,
        }

    evaluation_path_material = any(
        item["classification_changed"]
        for item in replay_checks.values()
    )
    fp16_pass = bool(
        new_cells["r0104_fp16_seed42_full_transform"][
            "clears_unchanged_registered_floor"
        ]
    )
    int8_pass = bool(
        new_cells["r0104_int8_seed42_full_transform"][
            "clears_unchanged_registered_floor"
        ]
    )
    storage_sensitive = fp16_pass != int8_pass
    if evaluation_path_material:
        outcome = "evaluation-path-material"
        boundary_conclusion = (
            "stop: the R0115/R0117 direct representative transform path "
            "changes at least one registered floor classification relative "
            "to full-2M-transform-then-select"
        )
        boundary_localized = False
    elif fp16_pass:
        outcome = "failure-enters-after-r0104-within-r0115-bundle"
        boundary_conclusion = (
            "the fp16 control clears before R0115, while both R0115/R0117 "
            "full-transform replays fail; the boundary is the bundled "
            "fresh-native8192/representative-selection/graph/sampler "
            "transition, not a single factor"
        )
        boundary_localized = True
    else:
        outcome = "failure-already-present-pre-r0115"
        boundary_conclusion = (
            "the R0104 fp16 control already fails on the exact matched "
            "R0040 scorer; R0115 is not the first observed failure boundary"
        )
        boundary_localized = True

    receipt = seal({
        "schema": DECISION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "score": expected_input_signature(score_path),
        "outcome": outcome,
        "checks": {
            "r0119_historical_and_direct_values_reused": True,
            "exact_r0119_r0040_universe_floor_and_scorer": True,
            "all_new_cells_transform_full_2000000_before_selection": all(
                cell.get("transform_input_rows") == SOURCE_ROWS
                and cell.get("transform_selection_after_transform") is True
                for cell in new_cells.values()
            ),
            "r0115_replay": replay_checks,
            "r0104_fp16_clears_floor": fp16_pass,
            "r0104_int8_clears_floor": int8_pass,
            "training_performed": False,
        },
        "evaluation_path_material": evaluation_path_material,
        "boundary_localized": boundary_localized,
        "boundary_conclusion_tied_to": (
            "r0104_fp16_seed42_full_transform"
        ),
        "boundary_conclusion": boundary_conclusion,
        "storage_sensitive_diagnostic": {
            "fp16_int8_floor_classification_disagrees": storage_sensitive,
            "diagnostic_only": True,
            "does_not_change_fp16_boundary_conclusion": True,
        },
        "localized_bundle": (
            [
                "fresh native-8192 embedding provenance",
                "representative selection",
                "fuzzy graph construction",
                "host sampler/execution path",
            ]
            if outcome
            == "failure-enters-after-r0104-within-r0115-bundle"
            else []
        ),
        "single_factor_cause_localized": False,
        "native_r0115_density": 0.2304,
        "native_r0115_density_numerically_clears_floor": True,
        "native_training_geometry_declared_bad": False,
        "role": (
            "calibration/representation-transfer localization only; the "
            "matched R0040 score does not override R0115's passing native "
            "density and cannot by itself prove bad native training geometry"
        ),
        "production_transfer_claimed": False,
        "map_registry_state_changed": False,
        "training_performed": False,
    })
    path = os.path.join(output, "density-bridge-decision.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if job is None:
        job = active["job"]
    if (active.get("manifest") or {}).get("round_id") != ROUND_ID:
        raise Round0122Error("R0122 handler received another round")
    action = str(job.get("action"))
    if action == "score_density_provenance_bridge":
        return run_score(active, job)
    if action == "decide_density_provenance_bridge":
        return run_decision(active, job)
    raise Round0122Error(f"unknown R0122 action: {action}")
