"""Fresh-process nodes for the matched 30M density-factorial diagnostic."""
from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Mapping
from typing import Any

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
from basemap.round0014_program import Round0014MaterializedArray
from basemap.round0036_pipeline import (
    COORDINATE_SCHEMA,
    TRANSFORM_SCHEMA,
    CoordinateStream,
)
from basemap.round0040_program import RepresentativeArrayView
from basemap.round0064_evaluation import (
    load_substrate,
    load_train_model,
    seal,
    validate_seal,
    validate_train_bundle,
)
from experiments.run_round0036_node import _project_encoded_block


ROUND_ID = "0070"
ROWS = 30_000_000
DIMENSION = 384


class Round0070Error(RuntimeError):
    """The registered density-factorial contract was violated."""


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0070Error(f"{path} is not a JSON object")
    return value


def _panel_config():
    from basemap.panel_v2 import PanelV2Config
    from basemap.round0036_pipeline import panel_config_identity

    return PanelV2Config(**{
        key: tuple(value) if key == "k_clust" else value
        for key, value in panel_config_identity().items()
        if key != "formula_version"
    })


def _balanced_bundle(job: Mapping[str, Any]) -> dict[str, Any]:
    return validate_train_bundle(
        label="r0061-balanced-30m",
        model_path=str(job["balanced_model_path"]),
        model_sha256=str(job["balanced_model_sha256"]),
        train_receipt_path=str(job["balanced_receipt_path"]),
        train_receipt_sha256=str(job["balanced_receipt_sha256"]),
    )


def _balanced_source(job: Mapping[str, Any]):
    return load_substrate(
        int8_path=str(job["int8_path"]),
        int8_sha256=str(job["int8_sha256"]),
        scales_path=str(job["scales_path"]),
        scales_sha256=str(job["scales_sha256"]),
        eligibility_path=str(job["eligibility_path"]),
        eligibility_sha256=str(job["eligibility_sha256"]),
        row_count=ROWS,
    )


def run_modern_transform(
    _active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    output = create_fresh_directory(
        job["outputs"][0],
        label=f"R0070 {job['map_key']} coordinates",
    )
    started = time.monotonic()
    bundle = _balanced_bundle(job)
    model = load_train_model(bundle, device="cuda")
    source = Round0014MaterializedArray()
    input_identity = {
        "kind": "accepted-r0013-fp16-30m-pack",
        "row_count": len(source),
        "dtype": source.dtype.str,
        "pack_seal": source.round0014_pack_seal,
    }
    if len(source) != ROWS or source.shape != (ROWS, DIMENSION):
        raise Round0070Error("R0070 transform source geometry changed")

    chunk_rows = int(job.get("coordinate_chunk_rows", 5_000_000))
    batch_rows = int(job.get("model_batch_rows", 65_536))
    if chunk_rows != 5_000_000 or batch_rows != 65_536:
        raise Round0070Error("R0070 transform batching changed")
    members: list[dict[str, Any]] = []
    for index, start in enumerate(range(0, ROWS, chunk_rows)):
        stop = min(start + chunk_rows, ROWS)
        root = create_fresh_directory(
            os.path.join(output, f"chunk-{index:05d}"),
            label="R0070 coordinate chunk",
        )
        path = os.path.join(root, "coordinates.npy")
        coordinates = _project_encoded_block(
            model,
            source,
            start,
            stop,
            batch_rows=batch_rows,
        )
        if (
            coordinates.shape != (stop - start, 2)
            or coordinates.dtype.str != "<f4"
            or not np.isfinite(coordinates).all()
        ):
            raise Round0070Error("R0070 cross transform emitted invalid coordinates")
        atomic_save_new_npy(path, coordinates, immutable=True)
        signature = expected_input_signature(path)
        members.append({
            "chunk_index": index,
            "global_row_start": start,
            "global_row_stop": stop,
            "bytes": signature["bytes"],
            "sha256": signature["sha256"],
        })
        del coordinates

    body = {
        "schema": TRANSFORM_SCHEMA,
        "round_id": ROUND_ID,
        "map_key": job["map_key"],
        "model_kind": "balanced-r0061",
        "model": bundle["model"],
        "train_receipt": bundle["train_receipt"],
        "production_config_sha256": bundle["production_config_sha256"],
        "scientific_universe": "original-fp16-all-rows",
        "input": input_identity,
        "eligibility": None,
        "inference": {
            "batch_rows": batch_rows,
            "short_tail_policy": "zero-pad-to-fixed-batch-then-discard-padding",
            "all_real_rows_projected": True,
        },
        "row_accounting": {
            "all_rows": ROWS,
            "scientific_rows": ROWS,
        },
        "coordinate_stream": {
            "schema": COORDINATE_SCHEMA,
            "row_count": ROWS,
            "dimension": 2,
            "dtype": "<f4",
            "row_order": "FineWeb[0:10M], RedPajama[0:10M], Pile[0:10M]",
            "ordered_chunks": members,
        },
        "training_performed": False,
        "wall_seconds": time.monotonic() - started,
    }
    receipt = seal(body)
    path = os.path.join(output, "actual-transform.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_original_reference(
    _active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    from basemap.panel_v2 import _self_knn

    output = create_fresh_directory(
        job["outputs"][0],
        label="R0070 original-universe high-D radii",
    )
    started = time.monotonic()
    anchors = np.load(
        str(job["common_anchor_rows_path"]),
        mmap_mode="r",
        allow_pickle=False,
    )
    anchors = np.asarray(anchors, dtype=np.int64)
    if (
        anchors.shape != (10_000,)
        or ordered_array_sha256(anchors) != job["common_anchor_rows_sha256"]
        or np.any(anchors < 0)
        or np.any(anchors >= ROWS)
    ):
        raise Round0070Error("common global anchors changed")
    source = Round0014MaterializedArray()
    _, selector, retained, eligibility = _balanced_source(job)
    compact_anchors = selector.global_to_compact(anchors)
    if (
        np.any(compact_anchors < 0)
        or not np.array_equal(selector.compact_to_global(compact_anchors), anchors)
    ):
        raise Round0070Error("a common anchor is absent from representatives")
    representative_source = RepresentativeArrayView(source, selector)
    _, original_distances, original_guard = _self_knn(
        source,
        anchors,
        15,
        _panel_config(),
        hi_dim=True,
        want_dist=True,
        exact=True,
    )
    _, representative_distances, representative_guard = _self_knn(
        representative_source,
        compact_anchors,
        15,
        _panel_config(),
        hi_dim=True,
        want_dist=True,
        exact=True,
    )
    original_radii = np.asarray(original_distances.mean(1), dtype="<f8")
    representative_radii = np.asarray(
        representative_distances.mean(1),
        dtype="<f8",
    )
    if (
        original_radii.shape != anchors.shape
        or representative_radii.shape != anchors.shape
        or not np.isfinite(original_radii).all()
        or not np.isfinite(representative_radii).all()
        or np.any(original_radii < 0)
        or np.any(representative_radii < 0)
    ):
        raise Round0070Error("fp16 high-D radii are invalid")
    archive_path = os.path.join(output, "fp16-high-d-radii.npz")

    def write_archive(path: str) -> None:
        with open(path, "wb") as handle:
            np.savez(
                handle,
                anchor_global_rows=anchors,
                anchor_representative_compact_rows=compact_anchors,
                high_d_radius_original=original_radii,
                high_d_radius_representative=representative_radii,
            )

    atomic_build_new_file(archive_path, write_archive, immutable=True)
    body = {
        "schema": "round0070-fp16-high-d-reference-v1",
        "round_id": ROUND_ID,
        "scientific_universes": {
            "original_fp16_all_rows": ROWS,
            "fp16_representatives": len(retained),
        },
        "eligibility": eligibility["signature"],
        "anchors": expected_input_signature(
            str(job["common_anchor_rows_path"])
        ),
        "anchor_global_rows_sha256": ordered_array_sha256(anchors),
        "anchor_representative_compact_rows_sha256": ordered_array_sha256(
            compact_anchors
        ),
        "k_density": 15,
        "exactness": "exact-fp32-rerank",
        "guards": {
            "original": original_guard,
            "representative": representative_guard,
        },
        "radii": expected_input_signature(archive_path),
        "training_performed": False,
        "wall_seconds": time.monotonic() - started,
    }
    receipt = seal(body)
    path = os.path.join(output, "reference-receipt.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def _density_summary(
    high_radius: np.ndarray,
    low_radius: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    high = np.log(np.asarray(high_radius, dtype=np.float64) + 1e-12)
    low = np.log(np.asarray(low_radius, dtype=np.float64) + 1e-12)
    if (
        high.shape != low.shape
        or high.shape != labels.shape
        or len(high) != 10_000
        or not np.isfinite(high).all()
        or not np.isfinite(low).all()
    ):
        raise Round0070Error("density inputs are malformed")

    def one(mask: np.ndarray) -> dict[str, Any]:
        correlation = float(np.corrcoef(high[mask], low[mask])[0, 1])
        if not math.isfinite(correlation):
            raise Round0070Error("density correlation is nonfinite")
        return {
            "anchors": int(mask.sum()),
            "correlation": round(correlation, 4),
            "mean_log_high_d_radius": round(float(high[mask].mean()), 6),
            "mean_log_low_d_radius": round(float(low[mask].mean()), 6),
        }

    groups = {
        name: one(labels == name)
        for name in ("fineweb", "redpajama", "pile")
    }
    return {
        **one(np.ones(len(high), dtype=bool)),
        "by_corpus": groups,
    }


def classify_factorial(cells: Mapping[str, float]) -> dict[str, Any]:
    """Apply the registered, deliberately coarse 2x2 interpretation rule."""
    a = float(cells["legacy_original"])
    b = float(cells["legacy_representative"])
    c = float(cells["modern_original"])
    d = float(cells["modern_representative"])
    model_original = c - a
    model_balanced = d - b
    universe_legacy = b - a
    universe_modern = d - c
    interaction = universe_modern - universe_legacy
    tolerance = 0.05
    material = 0.20
    interaction_band = 0.10

    def same_material_direction(left: float, right: float) -> bool:
        return (
            abs(left) >= material
            and abs(right) >= material
            and math.copysign(1.0, left) == math.copysign(1.0, right)
        )

    model_material = same_material_direction(model_original, model_balanced)
    universe_material = same_material_direction(
        universe_legacy,
        universe_modern,
    )
    if abs(interaction) >= interaction_band:
        classification = "model-by-universe-interaction"
    elif universe_material and max(
        abs(model_original), abs(model_balanced)
    ) <= tolerance:
        classification = "data-universe-dominant"
    elif model_material and max(
        abs(universe_legacy), abs(universe_modern)
    ) <= tolerance:
        classification = "model-training-dominant"
    elif model_material and universe_material:
        classification = "additive-model-and-universe"
    else:
        classification = "inconclusive-under-registered-bands"
    return {
        "classification": classification,
        "contrasts": {
            "model_effect_in_original": round(model_original, 4),
            "model_effect_in_balanced": round(model_balanced, 4),
            "universe_effect_for_legacy": round(universe_legacy, 4),
            "universe_effect_for_modern": round(universe_modern, 4),
            "difference_of_universe_effects": round(interaction, 4),
        },
        "registered_bands": {
            "near_equivalence": tolerance,
            "material_main_effect": material,
            "interaction": interaction_band,
        },
        "model_effect_material_and_same_direction": model_material,
        "universe_effect_material_and_same_direction": universe_material,
    }


def run_density_factorial(
    _active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    from basemap.panel_v2 import _self_knn, load_hiD_reference
    from experiments import run_round0014_node as legacy

    output = create_fresh_directory(
        job["outputs"][0],
        label="R0070 matched density factorial",
    )
    started = time.monotonic()
    encoded, selector, _, eligibility = _balanced_source(job)
    del encoded
    int8_reference = load_hiD_reference(str(job["int8_reference_path"]))
    compact_anchors = np.asarray(
        int8_reference["anchor_ids"],
        dtype=np.int64,
    )
    global_anchors = np.load(
        str(job["common_anchor_rows_path"]),
        mmap_mode="r",
        allow_pickle=False,
    )
    global_anchors = np.asarray(global_anchors, dtype=np.int64)
    if (
        ordered_array_sha256(global_anchors)
        != job["common_anchor_rows_sha256"]
        or not np.array_equal(
            selector.compact_to_global(compact_anchors),
            global_anchors,
        )
    ):
        raise Round0070Error("balanced and original anchor identity differs")
    int8_high = np.asarray(int8_reference["r_hd"], dtype=np.float64)
    fp16_archive = np.load(
        str(job["fp16_reference_archive"]),
        allow_pickle=False,
    )
    if (
        not np.array_equal(fp16_archive["anchor_global_rows"], global_anchors)
        or not np.array_equal(
            fp16_archive["anchor_representative_compact_rows"],
            compact_anchors,
        )
        or fp16_archive["high_d_radius_original"].shape != int8_high.shape
        or fp16_archive["high_d_radius_representative"].shape
        != int8_high.shape
    ):
        raise Round0070Error("fp16 high-D reference anchors changed")
    original_high = np.asarray(
        fp16_archive["high_d_radius_original"],
        dtype=np.float64,
    )
    representative_high = np.asarray(
        fp16_archive["high_d_radius_representative"],
        dtype=np.float64,
    )
    labels = np.asarray(
        ["fineweb", "redpajama", "pile"],
        dtype="<U10",
    )[global_anchors // 10_000_000]

    legacy.configure_round0019()
    legacy_original = legacy.StreamedCoordinateArray(
        str(job["legacy_original_coordinates"])
    )
    legacy_model = legacy_original.record["actual_transform"][
        "model_signature"
    ]
    if legacy_model["sha256"] != job["legacy_model_sha256"]:
        raise Round0070Error("R0019 coordinates bind another model")
    modern_original = CoordinateStream(
        str(job["modern_original_coordinates"])
    )
    modern_int8_full = CoordinateStream(
        str(job["modern_int8_coordinates"])
    )
    if (
        modern_original.receipt["model"]["sha256"]
        != job["balanced_model_sha256"]
        or modern_int8_full.receipt["model"]["sha256"]
        != job["balanced_model_sha256"]
    ):
        raise Round0070Error("R0061 coordinate stream binds another model")
    streams = {
        "legacy_original": (
            legacy_original,
            global_anchors,
            original_high,
            "legacy-r0019",
            "original-fp16-all-rows",
        ),
        "legacy_representative": (
            RepresentativeArrayView(legacy_original, selector),
            compact_anchors,
            representative_high,
            "legacy-r0019",
            "fp16-exact-representatives",
        ),
        "modern_original": (
            modern_original,
            global_anchors,
            original_high,
            "balanced-r0061",
            "original-fp16-all-rows",
        ),
        "modern_representative": (
            RepresentativeArrayView(modern_original, selector),
            compact_anchors,
            representative_high,
            "balanced-r0061",
            "fp16-exact-representatives",
        ),
        "modern_int8_bridge": (
            RepresentativeArrayView(modern_int8_full, selector),
            compact_anchors,
            int8_high,
            "balanced-r0061",
            "int8-exact-representatives",
        ),
    }
    config = _panel_config()
    cells: dict[str, Any] = {}
    low_radii: dict[str, np.ndarray] = {}
    for name, (coordinates, anchors, high_radius, model_kind, universe) in (
        streams.items()
    ):
        _, distances, guard = _self_knn(
            coordinates,
            anchors,
            15,
            config,
            hi_dim=False,
            want_dist=True,
            exact=True,
        )
        low_radius = np.asarray(distances.mean(1), dtype=np.float64)
        summary = _density_summary(high_radius, low_radius, labels)
        cells[name] = {
            "model_kind": model_kind,
            "scientific_universe": universe,
            "density": summary,
            "low_dim_guard": guard,
        }
        low_radii[name] = low_radius

    modern_panel = _read_json(str(job["modern_int8_panel_path"]))
    validate_seal(modern_panel, label="R0064 modern balanced panel")
    if modern_panel.get("map_key") != "r0061-30m-on-30m":
        raise Round0070Error("R0064 modern balanced panel identity changed")
    registered_modern = float(modern_panel["panel"]["density"])
    if (
        cells["modern_int8_bridge"]["density"]["correlation"]
        != registered_modern
    ):
        raise Round0070Error(
            "modern int8 density does not replay the reviewed R0064 panel"
        )
    legacy_panel = _read_json(str(job["legacy_original_panel_path"]))
    registered_legacy = float(legacy_panel["panel"]["density"])
    scalars = {
        name: float(cell["density"]["correlation"])
        for name, cell in cells.items()
        if name != "modern_int8_bridge"
    }
    interpretation = classify_factorial(scalars)
    archive_path = os.path.join(output, "matched-density-factorial-radii.npz")

    def write_archive(path: str) -> None:
        with open(path, "wb") as handle:
            np.savez(
                handle,
                anchor_global_rows=global_anchors,
                anchor_representative_compact_rows=compact_anchors,
                corpus_labels=labels,
                high_d_radius_fp16_original=original_high,
                high_d_radius_fp16_representative=representative_high,
                high_d_radius_int8_representative=int8_high,
                **{
                    f"low_d_radius_{name}": values
                    for name, values in low_radii.items()
                },
            )

    atomic_build_new_file(archive_path, write_archive, immutable=True)
    body = {
        "schema": "round0070-density-factorial-v1",
        "round_id": ROUND_ID,
        "design": (
            "two reviewed models crossed with original-all-row and exact-"
            "representative universes; identical 10,000 global anchors"
        ),
        "anchor_global_rows_sha256": ordered_array_sha256(global_anchors),
        "anchor_representative_compact_rows_sha256": ordered_array_sha256(
            compact_anchors
        ),
        "balanced_eligibility": eligibility["signature"],
        "cells": cells,
        "registered_native_panel_context": {
            "legacy_original_different_anchor_draw": registered_legacy,
            "modern_int8_exact_replay": registered_modern,
        },
        "representation_bridge": {
            "modern_fp16_representative_density": cells[
                "modern_representative"
            ]["density"]["correlation"],
            "modern_int8_representative_density": cells[
                "modern_int8_bridge"
            ]["density"]["correlation"],
            "int8_minus_fp16": round(
                cells["modern_int8_bridge"]["density"]["correlation"]
                - cells["modern_representative"]["density"]["correlation"],
                4,
            ),
            "primary_factorial_uses_one_numeric_representation": True,
        },
        "interpretation": interpretation,
        "radii_archive": expected_input_signature(archive_path),
        "limits": {
            "density_threshold_calibrated": False,
            "training_graph_sampler_components_separated": False,
            "causal_claim": (
                "the crossed cells isolate aggregate model/training versus "
                "candidate-universe effects at fixed fp16 input precision; "
                "the bridge separately reports int8 representation drift. "
                "The round does not separate graph from sampler or define a "
                "production density floor"
            ),
        },
        "training_performed": False,
        "wall_seconds": time.monotonic() - started,
    }
    receipt = seal(body)
    path = os.path.join(output, "density-factorial.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise RuntimeError("R0070 handler received another queue")
    if job is None:
        raise RuntimeError("R0070 handler requires the exact job")
    handlers = {
        "modern_transform": run_modern_transform,
        "original_reference": run_original_reference,
        "density_factorial": run_density_factorial,
    }
    try:
        handler = handlers[str(job["action"])]
    except KeyError as exc:
        raise RuntimeError(
            f"unknown R0070 action {job.get('action')!r}"
        ) from exc
    return handler(active, job)
