"""No-training localization of the matched FineWeb density failure."""
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
from basemap.round0040_program import (
    RepresentativeArrayView,
    RepresentativeRowSelector,
    load_jina_census,
)
from basemap.round0108_evaluation import (
    FAMILY_SIZE_CUTOFF,
    K_DENSITY,
    Round0108Error,
    TRANSFORM_BATCH_ROWS,
    map_family_sizes,
    seal,
    validate_seal,
)
from experiments.round0085_nodes import density_v2_calibration
from experiments.round0108_nodes import _panel_config


ROUND_ID = "0119"
SCORE_SCHEMA = "round0119-jina-density-localization-panel-v1"
DECISION_SCHEMA = "round0119-jina-density-localization-decision-v1"
CALIBRATION_SCHEMA = "round0108-jina-density-v2-calibration-v1"
REFERENCE_SCHEMA = "hiD_reference.v3"
SOURCE_ROWS = 2_000_000
SOURCE_DIMENSION = 768
REPRESENTATIVE_ROWS = 1_996_279
ANCHORS = 10_000

CELL_ORDER = (
    "historical_2m_seed42",
    "historical_2m_seed43",
    "current_2m_seed42",
    "current_2m_seed43",
    "current_25m_seed42",
    "current_25m_seed43",
)
GROUPS = {
    "historical_2m": (
        "historical_2m_seed42",
        "historical_2m_seed43",
    ),
    "current_2m": ("current_2m_seed42", "current_2m_seed43"),
    "current_25m": ("current_25m_seed42", "current_25m_seed43"),
}


class Round0119Error(RuntimeError):
    """Raised when the registered R0119 evidence contract changes."""


def _exact_signature(
    expected: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    path = str(expected.get("canonical_path") or "")
    actual = expected_input_signature(path)
    if actual != dict(expected):
        raise Round0119Error(f"{label} bytes changed")
    return actual


def _read_json_signature(
    expected: Mapping[str, Any],
    *,
    label: str,
    sealed: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    signature = _exact_signature(expected, label=label)
    with open(signature["canonical_path"], encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0119Error(f"{label} is not a JSON object")
    if sealed:
        try:
            validate_seal(value, label=label)
        except Round0108Error as error:
            raise Round0119Error(str(error)) from error
    return value, signature


def _architecture(config: Mapping[str, Any]) -> dict[str, Any]:
    model = config.get("model")
    if not isinstance(model, Mapping):
        raise Round0119Error("production model configuration is missing")
    expected = {
        "architecture": "residual_bottleneck",
        "input_dimension": SOURCE_DIMENSION,
        "hidden_dimension": 2048,
        "hidden_layers": 3,
        "output_dimension": 2,
        "use_batchnorm": False,
        "use_dropout": False,
        "low_dim_kernel": "legacy_lp",
        "a": 1.0,
        "b": 1.0,
    }
    observed = {key: model.get(key) for key in expected}
    if observed != expected:
        raise Round0119Error("production model architecture changed")
    return observed


def _authenticate_training_semantics(
    train: Mapping[str, Any],
    config: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    expected = spec.get("semantic_contract")
    optimizer = config.get("optimizer")
    graph = config.get("graph")
    accounting = train.get("train_accounting")
    if not all(
        isinstance(value, Mapping)
        for value in (expected, optimizer, graph, accounting)
    ):
        raise Round0119Error(
            f"{spec['key']} training semantic contract is missing"
        )
    expected = dict(expected)
    updates = expected["successful_updates"]
    if (
        optimizer.get("successful_positive_lr_updates") != updates
        or optimizer.get("seed") != spec["seed"]
        or accounting.get("optimizer_steps_attempted") != updates
        or accounting.get("optimizer_steps_succeeded") != updates
        or graph.get("sampling")
        != "fuzzy-weight-proportional-with-replacement"
        or graph.get("positive_target_mode") != "binary"
    ):
        raise Round0119Error(
            f"{spec['key']} population/graph/update semantics changed"
        )

    group = spec["group"]
    if group == "historical_2m":
        population = config.get("row_universe")
        pipeline = train.get("actual_pipeline")
        observed = {
            "population_rows": (population or {}).get("rows"),
            "graph_neighbors": graph.get("k"),
            "successful_updates": accounting.get(
                "optimizer_steps_succeeded"
            ),
            "pipeline": (pipeline or {}).get("pipeline"),
            "sampler_class": (pipeline or {}).get("sampler_class"),
            "positive_sampling": (pipeline or {}).get(
                "positive_sampling"
            ),
            "multiplicity_policy": (pipeline or {}).get(
                "multiplicity_policy"
            ),
            "feature_residency": (pipeline or {}).get("x_residency"),
            "source_representation": (population or {}).get("source_dtype"),
            "dequantization": (population or {})
            .get("input_preprocessing", {})
            .get("operation"),
        }
        accounting_observed = {
            "pipeline": accounting.get("pipeline_pipeline"),
            "sampler_class": accounting.get("pipeline_sampler_class"),
            "positive_sampling": accounting.get(
                "pipeline_positive_sampling"
            ),
            "multiplicity_policy": accounting.get(
                "pipeline_multiplicity_policy"
            ),
            "feature_residency": accounting.get("pipeline_x_residency"),
        }
    elif group == "current_2m":
        population = config.get("input")
        pipeline = train.get("exact_execution_receipt")
        observed = {
            "population_rows": (population or {}).get("rows"),
            "graph_neighbors": graph.get("k"),
            "successful_updates": train.get("optimizer_updates"),
            "pipeline": (pipeline or {}).get("pipeline"),
            "sampler_class": (pipeline or {}).get("sampler_class"),
            "positive_sampling": (pipeline or {}).get(
                "positive_sampling"
            ),
            "multiplicity_policy": (pipeline or {}).get(
                "multiplicity_policy"
            ),
            "feature_residency": (pipeline or {}).get(
                "feature_residency"
            ),
            "source_representation": (pipeline or {}).get(
                "source_representation"
            ),
            "dequantization": (pipeline or {}).get("device_conversion"),
        }
        accounting_observed = {
            "pipeline": accounting.get("pipeline_pipeline"),
            "sampler_class": accounting.get("pipeline_sampler_class"),
            "positive_sampling": accounting.get(
                "pipeline_positive_sampling"
            ),
            "multiplicity_policy": accounting.get(
                "pipeline_multiplicity_policy"
            ),
            "feature_residency": accounting.get(
                "pipeline_feature_residency"
            ),
            "source_representation": accounting.get(
                "pipeline_source_representation"
            ),
            "dequantization": accounting.get("pipeline_device_conversion"),
            "graph_neighbors": (
                50
                if accounting.get("pipeline_graph_degree")
                == "variable-symmetric-fuzzy-k50-topology"
                else None
            ),
            "population_rows": accounting.get(
                "pipeline_compact_retained_rows"
            ),
        }
        if (population or {}).get("representation") != "fresh-local-raw-fp16":
            raise Round0119Error(
                f"{spec['key']} configured source representation changed"
            )
    elif group == "current_25m":
        population = config.get("input")
        pipeline = train.get("exact_execution_receipt")
        observed = {
            "population_rows": (population or {}).get("rows"),
            "graph_neighbors": (
                graph.get("n_neighbors_including_self") - 1
                if isinstance(
                    graph.get("n_neighbors_including_self"), int
                )
                else None
            ),
            "graph_neighbors_including_self": graph.get(
                "n_neighbors_including_self"
            ),
            "successful_updates": train.get("optimizer_updates"),
            "pipeline": (pipeline or {}).get("pipeline"),
            "sampler_class": (pipeline or {}).get("sampler_class"),
            "positive_sampling": (pipeline or {}).get(
                "positive_sampling"
            ),
            "multiplicity_policy": None,
            "feature_residency": (pipeline or {}).get(
                "feature_residency"
            ),
            "source_representation": (pipeline or {}).get(
                "source_representation"
            ),
            "dequantization": (pipeline or {}).get("dequantization"),
        }
        accounting_observed = {
            "pipeline": accounting.get("pipeline_pipeline"),
            "sampler_class": accounting.get("pipeline_sampler_class"),
            "positive_sampling": accounting.get(
                "pipeline_positive_sampling"
            ),
            "multiplicity_policy": None,
            "feature_residency": accounting.get(
                "pipeline_feature_residency"
            ),
            "source_representation": accounting.get(
                "pipeline_source_representation"
            ),
            "dequantization": accounting.get("pipeline_dequantization"),
            "graph_neighbors": (
                15
                if accounting.get("pipeline_graph_degree")
                == "variable-symmetric-fuzzy-k15-topology"
                else None
            ),
            "graph_neighbors_including_self": 16,
            "population_rows": accounting.get(
                "pipeline_compact_retained_rows"
            ),
        }
        if (
            (population or {}).get("representation")
            != "signed-int8-plus-exact-fp16-row-scale"
        ):
            raise Round0119Error(
                f"{spec['key']} configured source representation changed"
            )
    else:
        raise Round0119Error(f"{spec['key']} has an unknown model group")

    if (
        observed != expected
        or any(
            accounting_observed.get(key) != expected[key]
            for key in accounting_observed
        )
        or (pipeline or {}).get("positive_with_replacement") is not True
        or (pipeline or {}).get("weighted_effective") is not True
    ):
        raise Round0119Error(
            f"{spec['key']} actual pipeline semantics changed"
        )
    checks = train.get("train_checks")
    if checks is not None and (
        not isinstance(checks, Mapping)
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
        raise Round0119Error(
            f"{spec['key']} training accounting does not close"
        )
    return expected


def _authenticate_model(
    spec: Mapping[str, Any],
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    """Authenticate all three bundle files before loading one frozen model."""
    train, train_signature = _read_json_signature(
        spec["train_receipt"],
        label=f"{spec['key']} train receipt",
        sealed=True,
    )
    config_receipt, config_signature = _read_json_signature(
        spec["production_config"],
        label=f"{spec['key']} production config",
        sealed=False,
    )
    model_signature = _exact_signature(
        spec["model"], label=f"{spec['key']} model"
    )
    config = config_receipt.get("config")
    if not isinstance(config, dict):
        raise Round0119Error(f"{spec['key']} config body is missing")
    config_sha256 = sha256_bytes(canonical_json(config))
    recorded_config_sha256 = config_receipt.get("config_sha256")
    canonical_config_closes = recorded_config_sha256 == config_sha256
    legacy_key_roundtrip = bool(
        spec.get("legacy_integer_key_json_roundtrip")
    )
    optimizer = config.get("optimizer")
    expected_config_round = spec.get("config_receipt_round_id")
    if (
        train.get("schema") != spec["train_schema"]
        or train.get("round_id") != spec["round_id"]
        or config_receipt.get("schema") != spec["config_receipt_schema"]
        or config_receipt.get("round_id") != expected_config_round
        or config.get("schema") != spec["config_schema"]
        or (
            not canonical_config_closes
            and not legacy_key_roundtrip
        )
        or not isinstance(optimizer, Mapping)
        or optimizer.get("seed") != spec["seed"]
        or train.get("model") != model_signature
        or train.get("production_config_sha256") != recorded_config_sha256
    ):
        raise Round0119Error(f"{spec['key']} train/config identity changed")
    internal_config = train.get("production_config")
    if internal_config is not None and internal_config != config_signature:
        raise Round0119Error(
            f"{spec['key']} production-config receipt binding changed"
        )
    train_seed = train.get("training_seed", train.get("seed"))
    if train_seed is not None and train_seed != spec["seed"]:
        raise Round0119Error(f"{spec['key']} training seed changed")
    if spec.get("arm") is not None and (
        train.get("arm") != spec["arm"]
        or config.get("arm") != spec["arm"]
    ):
        raise Round0119Error(f"{spec['key']} prompt arm changed")
    expected_architecture = _architecture(config)
    authenticated_semantics = _authenticate_training_semantics(
        train, config, spec
    )

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
        raise Round0119Error(f"{spec['key']} checkpoint architecture changed")
    return {
        "model": model,
        "train": train_signature,
        "production_config": config_signature,
        "model_signature": model_signature,
        "seed": spec["seed"],
        "group": spec["group"],
        "training_population": spec["training_population"],
        "training_graph": spec["training_graph"],
        "training_dose": spec["training_dose"],
        "training_representation": spec["training_representation"],
        "training_dequantization": spec["training_dequantization"],
        "authenticated_training_semantics": authenticated_semantics,
        "canonical_config_rehash_closes": canonical_config_closes,
        "legacy_integer_key_json_roundtrip": legacy_key_roundtrip,
    }


def _load_universe(
    job: Mapping[str, Any],
) -> tuple[
    np.ndarray,
    RepresentativeArrayView,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
    dict[str, np.ndarray],
]:
    calibration, calibration_signature = _read_json_signature(
        job["r0108_calibration"],
        label="R0108 density calibration",
        sealed=True,
    )
    if (
        calibration.get("schema") != CALIBRATION_SCHEMA
        or calibration.get("round_id") != "0108"
        or calibration.get("threshold_tuned_after_treatment") is not False
        or "mean-k15 radius" not in str(calibration.get("scorer"))
    ):
        raise Round0119Error("frozen R0108 density contract changed")
    floor_calibration = calibration.get("floor_calibration")
    if (
        not isinstance(floor_calibration, Mapping)
        or floor_calibration.get("registered_floor")
        != 0.17589389755990817
        or floor_calibration.get("gating_floor_registered") is not True
    ):
        raise Round0119Error("R0108 registered floor changed")

    census, census_signature = _read_json_signature(
        calibration["census_receipt"],
        label="R0040 census receipt",
        sealed=False,
    )
    census_artifact_signature = _exact_signature(
        calibration["census"], label="R0040 exact-family census"
    )
    if census.get("census") != census_artifact_signature:
        raise Round0119Error("R0040 census receipt/artifact binding changed")
    representative_reference_signature = _exact_signature(
        calibration["representative_reference"],
        label="R0040 representative high-D reference",
    )
    census_bundle = load_jina_census(census_signature["canonical_path"])
    if census_bundle["signature"] != census_artifact_signature:
        raise Round0119Error("R0040 loaded census identity changed")
    source_signature = _exact_signature(
        census["source"], label="R0040 FineWeb source"
    )
    if census_bundle["receipt"].get("source") != source_signature:
        raise Round0119Error("R0040 census/source binding changed")
    source = np.load(
        source_signature["canonical_path"], mmap_mode="r", allow_pickle=False
    )
    if (
        source.shape != (SOURCE_ROWS, SOURCE_DIMENSION)
        or source.dtype != np.dtype("<f2")
    ):
        raise Round0119Error("R0040 FineWeb source shape/dtype changed")
    selector = RepresentativeRowSelector(
        census_bundle["arrays"]["excluded_rows"],
        row_count=SOURCE_ROWS,
        source=census_bundle["signature"],
        policy="R0040 exact nonzero fp16 family; minimum row representative",
    )
    representatives = RepresentativeArrayView(source, selector)

    with np.load(
        representative_reference_signature["canonical_path"],
        allow_pickle=False,
    ) as archive:
        if set(("schema", "key", "anchor_ids", "r_hd")) - set(
            archive.files
        ):
            raise Round0119Error("R0040 high-D reference arrays are missing")
        reference_schema = str(np.asarray(archive["schema"]).item())
        reference_key = str(np.asarray(archive["key"]).item())
        anchors = np.asarray(archive["anchor_ids"], dtype=np.int64)
        high_radius = np.asarray(archive["r_hd"], dtype=np.float64)
    anchor_contract = calibration.get("anchors")
    if (
        reference_schema != REFERENCE_SCHEMA
        or reference_key
        != calibration.get("representative_reference_key")
        or anchors.shape != (ANCHORS,)
        or high_radius.shape != anchors.shape
        or not isinstance(anchor_contract, Mapping)
        or anchor_contract.get("before_family_filter") != ANCHORS
        or anchor_contract.get("after_family_lt_16_filter") != ANCHORS
        or ordered_array_sha256(anchors)
        != anchor_contract.get("compact_rows_sha256")
    ):
        raise Round0119Error("R0040 anchor/high-D reference changed")

    global_rows = selector.compact_to_global(anchors)
    family_sizes = map_family_sizes(
        global_rows,
        census_bundle["arrays"]["representative_rows"],
        census_bundle["arrays"]["family_counts"],
    )
    retained_global_rows = selector.compact_to_global(
        np.arange(len(representatives), dtype=np.int64)
    )
    if (
        len(representatives) != REPRESENTATIVE_ROWS
        or len(retained_global_rows) != REPRESENTATIVE_ROWS
        or ordered_array_sha256(global_rows)
        != anchor_contract.get("global_rows_sha256")
        or ordered_array_sha256(family_sizes)
        != anchor_contract.get("family_sizes_sha256")
        or not np.all(family_sizes < FAMILY_SIZE_CUTOFF)
    ):
        raise Round0119Error("R0040 representative/family filter changed")

    calibration_arrays_signature = _exact_signature(
        calibration["arrays"], label="R0108 calibration arrays"
    )
    reference: dict[str, np.ndarray] = {}
    with np.load(
        calibration_arrays_signature["canonical_path"], allow_pickle=False
    ) as archive:
        for seed in ("seed42", "seed43"):
            for suffix in (
                "high_radius",
                "low_radius",
                "bootstrap",
                "permuted_null",
            ):
                reference[f"{seed}__{suffix}"] = np.asarray(
                    archive[f"{seed}__{suffix}"]
                )
            if not np.array_equal(
                reference[f"{seed}__high_radius"], high_radius
            ):
                raise Round0119Error(
                    "R0108 historical high-D radii changed"
                )
    lineage = {
        "r0108_calibration": calibration_signature,
        "r0108_calibration_arrays": calibration_arrays_signature,
        "census_receipt": census_signature,
        "census": census_artifact_signature,
        "representative_reference": representative_reference_signature,
        "representative_reference_key": reference_key,
        "source": source_signature,
        "registered_floor": float(
            floor_calibration["registered_floor"]
        ),
    }
    return (
        source,
        representatives,
        retained_global_rows,
        anchors,
        global_rows,
        high_radius,
        lineage,
        reference,
    )


def _score_cell(
    *,
    key: str,
    bundle: Mapping[str, Any],
    source: np.ndarray,
    representatives: RepresentativeArrayView,
    retained_global_rows: np.ndarray,
    anchors: np.ndarray,
    high_radius: np.ndarray,
    reference: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    from basemap.panel_v2 import _self_knn

    historical = bundle["group"] == "historical_2m"
    transform_input = source if historical else representatives
    transformed = np.asarray(
        bundle["model"].transform(
            transform_input, batch_size=TRANSFORM_BATCH_ROWS
        ),
        dtype=np.float32,
    )
    expected_transform_rows = SOURCE_ROWS if historical else REPRESENTATIVE_ROWS
    if (
        transformed.shape != (expected_transform_rows, 2)
        or not np.isfinite(transformed).all()
    ):
        raise Round0119Error(f"{key} transformed coordinates are malformed")
    if historical:
        # R0037/R0038 transformed the full 2M source in 8192-row batches.
        # Selecting representatives only after that GEMM path is necessary to
        # reproduce the frozen R0108 controls exactly.
        coordinates = np.asarray(
            transformed[retained_global_rows], dtype=np.float32
        )
        del transformed
    else:
        coordinates = transformed
    if (
        coordinates.shape != (REPRESENTATIVE_ROWS, 2)
        or not np.isfinite(coordinates).all()
    ):
        raise Round0119Error(f"{key} coordinates are malformed")
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
    historical_seed = (
        f"seed{bundle['seed']}"
        if bundle["group"] == "historical_2m"
        else None
    )
    reproduction = None
    if historical_seed is not None:
        reference_low = reference[f"{historical_seed}__low_radius"]
        reference_bootstrap = reference[f"{historical_seed}__bootstrap"]
        reference_null = reference[f"{historical_seed}__permuted_null"]
        reproduction = {
            "absolute_tolerance": 1.0e-6,
            "relative_tolerance": 1.0e-6,
            "low_radius_within_tolerance": np.allclose(
                low_radius,
                reference_low,
                rtol=1.0e-6,
                atol=1.0e-6,
            ),
            "bootstrap_within_tolerance": np.allclose(
                bootstrap,
                reference_bootstrap,
                rtol=1.0e-6,
                atol=1.0e-6,
            ),
            "permuted_null_within_tolerance": np.allclose(
                null,
                reference_null,
                rtol=1.0e-6,
                atol=1.0e-6,
            ),
            "low_radius_maximum_absolute_delta": float(
                np.max(np.abs(low_radius - reference_low))
            ),
        }
        reproduction["reproduces_frozen_control"] = all(
            reproduction[key]
            for key in (
                "low_radius_within_tolerance",
                "bootstrap_within_tolerance",
                "permuted_null_within_tolerance",
            )
        )
    receipt = {
        "key": key,
        "group": bundle["group"],
        "seed": bundle["seed"],
        "training_population": bundle["training_population"],
        "training_graph": bundle["training_graph"],
        "training_dose": bundle["training_dose"],
        "training_representation": bundle["training_representation"],
        "training_dequantization": bundle["training_dequantization"],
        "authenticated_training_semantics": bundle[
            "authenticated_training_semantics"
        ],
        "train_receipt": bundle["train"],
        "production_config": bundle["production_config"],
        "model": bundle["model_signature"],
        "transform_batch_rows": TRANSFORM_BATCH_ROWS,
        "transform_input_rows": expected_transform_rows,
        "transform_selection_after_transform": historical,
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
        "historical_control_reproduction": reproduction,
    }
    arrays = {
        f"{key}__low_radius": low_radius,
        f"{key}__bootstrap": bootstrap,
        f"{key}__permuted_null": null,
    }
    return receipt, arrays


def run_score(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0119 density localization panel"
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
        reference,
    ) = _load_universe(job)
    specs = job.get("model_bundles")
    if (
        not isinstance(specs, list)
        or [spec.get("key") for spec in specs] != list(CELL_ORDER)
    ):
        raise Round0119Error("registered model-cell order changed")

    cells: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {
        "anchor_compact_rows": anchors,
        "anchor_global_rows": global_rows,
        "high_radius": high_radius,
    }
    for spec in specs:
        key = str(spec["key"])
        bundle = _authenticate_model(spec)
        cell, cell_arrays = _score_cell(
            key=key,
            bundle=bundle,
            source=source,
            representatives=representatives,
            retained_global_rows=retained_global_rows,
            anchors=anchors,
            high_radius=high_radius,
            reference=reference,
        )
        cell["clears_unchanged_registered_floor"] = (
            float(cell["density_v2"]["correlation"])
            >= lineage["registered_floor"]
        )
        cells[key] = cell
        arrays.update(cell_arrays)
        del bundle["model"]
        gc.collect()

    arrays_path = os.path.join(output, "density-localization-arrays.npz")
    atomic_save_new_npz(arrays_path, immutable=True, **arrays)
    receipt = seal({
        "schema": SCORE_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "lineage": lineage,
        "universe": {
            "name": (
                "R0040 exact FineWeb representative universe reconstructed "
                "from accepted R0108 calibration lineage"
            ),
            "source_rows": SOURCE_ROWS,
            "representative_rows": len(representatives),
            "anchors": len(anchors),
            "family_size_cutoff_exclusive": FAMILY_SIZE_CUTOFF,
            "anchor_compact_rows_sha256": ordered_array_sha256(anchors),
            "anchor_global_rows_sha256": ordered_array_sha256(global_rows),
            "high_radius_sha256": ordered_array_sha256(high_radius),
        },
        "scorer": {
            "metric": "density-v2 radius correlation",
            "k": K_DENSITY,
            "transform_batch_rows": TRANSFORM_BATCH_ROWS,
            "historical_full_source_transform_before_selection": True,
            "low_dim_search": "exact",
            "bootstrap_draws": 1_000,
            "bootstrap_seed": 10_801,
            "null_draws": 1_000,
            "null_seed": 10_802,
            "registered_floor": lineage["registered_floor"],
            "floor_changed_or_tuned": False,
        },
        "cells": cells,
        "arrays": expected_input_signature(arrays_path),
        "training_performed": False,
        "native_quality_recomputed": False,
        "production_or_prompt_transfer_evaluated": False,
        "wall_seconds": time.monotonic() - started,
    })
    receipt_path = os.path.join(output, "density-localization-panel.json")
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    del source, representatives, retained_global_rows
    gc.collect()
    return {**receipt, "receipt": expected_input_signature(receipt_path)}


def _pair_passes(cells: Mapping[str, Any], group: str) -> bool:
    return all(
        bool(cells[key]["clears_unchanged_registered_floor"])
        for key in GROUPS[group]
    )


def run_decision(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0119 density localization decision"
    )
    score_path = os.path.join(
        str(job["score_output"]), "density-localization-panel.json"
    )
    with open(score_path, encoding="utf-8") as handle:
        score = json.load(handle)
    validate_seal(score, label="R0119 density localization panel")
    cells = score.get("cells")
    if (
        score.get("schema") != SCORE_SCHEMA
        or score.get("round_id") != ROUND_ID
        or not isinstance(cells, Mapping)
        or set(cells) != set(CELL_ORDER)
        or score.get("training_performed") is not False
    ):
        raise Round0119Error("R0119 localization panel identity changed")

    historical_reproduced = all(
        bool(
            (
                cells[key].get("historical_control_reproduction") or {}
            ).get("reproduces_frozen_control")
        )
        for key in GROUPS["historical_2m"]
    )
    historical_clear = _pair_passes(cells, "historical_2m")
    current_2m_clear = _pair_passes(cells, "current_2m")
    current_25m_clear = _pair_passes(cells, "current_25m")
    if not historical_reproduced or not historical_clear:
        outcome = "historical-controls-not-reproduced"
        bundled_transition_localized = False
        failure_unique_to_25m_tuple_rejected = False
    elif not current_2m_clear:
        outcome = "failure-not-unique-to-25m-tuple"
        bundled_transition_localized = False
        failure_unique_to_25m_tuple_rejected = True
    elif not current_25m_clear:
        outcome = "bundled-2m-to-25m-transition-localized"
        bundled_transition_localized = True
        failure_unique_to_25m_tuple_rejected = False
    else:
        outcome = "matched-density-failure-not-reproduced"
        bundled_transition_localized = False
        failure_unique_to_25m_tuple_rejected = False

    receipt = seal({
        "schema": DECISION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "score": expected_input_signature(score_path),
        "checks": {
            "historical_controls_reproduce_within_frozen_tolerance": (
                historical_reproduced
            ),
            "historical_control_pair_clears_unchanged_floor": (
                historical_clear
            ),
            "current_2m_pair_clears_unchanged_floor": current_2m_clear,
            "current_25m_pair_clears_unchanged_floor": current_25m_clear,
            "unchanged_universe_anchors_radii_family_filter_and_floor": True,
            "all_six_train_model_config_bundles_bound": True,
            "training_performed": False,
        },
        "outcome": outcome,
        "bundled_2m_to_25m_transition_localized": (
            bundled_transition_localized
        ),
        "localized_bundle": (
            [
                "training population",
                "fuzzy graph",
                "positive-update dose",
                "source representation",
                "dequantization path",
                "associated scale-dependent execution tuple",
            ]
            if bundled_transition_localized
            else []
        ),
        "single_cause_localized": False,
        "failure_unique_to_25m_tuple_rejected": (
            failure_unique_to_25m_tuple_rejected
        ),
        "scale_contribution_excluded": False,
        "matched_cell_rescues_native_quality": False,
        "native_diverse_universe_quality_overridden": False,
        "production_transfer_claimed": False,
        "prompt_transfer_claimed": False,
        "map_registry_state_changed": False,
        "role": (
            "localizes only whether the matched-density loss enters between "
            "the current 2M and 25M bundled training tuples; it cannot "
            "separate population, graph, dose, representation, "
            "dequantization, or scale-dependent execution, and a current-2M "
            "failure cannot exclude an additional scale contribution"
        ),
        "training_performed": False,
    })
    path = os.path.join(output, "density-localization-decision.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if job is None:
        job = active["job"]
    if (active.get("manifest") or {}).get("round_id") != ROUND_ID:
        raise Round0119Error("R0119 handler received another round")
    action = str(job.get("action"))
    if action == "score_density_localization":
        return run_score(active, job)
    if action == "decide_density_localization":
        return run_decision(active, job)
    raise Round0119Error(f"unknown R0119 action: {action}")
