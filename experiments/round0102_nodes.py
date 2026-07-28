"""Evaluate the deliberate balanced-150M rung on matched and full universes."""
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
from basemap.int8_eligibility import load_int8_eligibility
from basemap.output_safety import (
    atomic_build_new_file,
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0036_pipeline import CoordinateStream
from basemap.round0040_program import (
    RepresentativeArrayView,
    RepresentativeRowSelector,
)
from basemap.round0064_evaluation import seal, validate_seal
from experiments import round0064_nodes as shared
from experiments import round0069_nodes as prior


ROUND_ID = "0102"
PANEL_SCHEMA = "round0102-registered-panel-v1"
CONTROL_KEY = "r0079-120m-on-120m"
MATCHED_KEY = "r0101-150m-on-120m"
FULL_KEY = "r0101-150m-on-150m"
MAP_LABELS = {
    CONTROL_KEY: "r0079-balanced-120m-seed42",
    MATCHED_KEY: "r0101-balanced-150m-seed42-on-matched-120m",
    FULL_KEY: "r0101-balanced-150m-seed42",
}
MATCHED_NONINFERIORITY_MARGINS = {
    "ffr": 0.02,
    "purity_k256": 0.05,
    "purity_k1024": 0.05,
}
DENSITY_V2_FLOOR = 0.041703756293199175
DENSITY_ANCHOR_COUNT = 10_000
DENSITY_K = 15
DENSITY_FAMILY_SIZE_CUTOFF = 16
DENSITY_LOG_EPSILON = 1e-12


class Round0102Error(RuntimeError):
    """The balanced-150M scale evaluation contract was violated."""


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0102Error(f"{path} is not a JSON object")
    return value


def _configure_shared() -> None:
    shared.ROUND_ID = ROUND_ID
    shared.MAP_LABELS = MAP_LABELS


def _metrics(panel: Mapping[str, Any]) -> dict[str, float]:
    scientific = panel["panel"]
    purity = scientific["purity"]
    return {
        "ffr": float(scientific["ffr"]),
        "density": float(scientific["density"]),
        "purity_k256": float(purity["k256"]),
        "purity_k1024": float(purity["k1024"]),
        "projection_ffr": float(panel["projection"]["proj_ffr"]),
        "recall_at_10": float(panel["recall_at_10"]),
        "recall_at_50": float(panel["recall_at_50"]),
    }


def _load_panel(
    path: str,
    *,
    schema: str,
    key: str,
) -> dict[str, Any]:
    panel = _read_json(path)
    validate_seal(panel, label=f"R0102 {key} panel")
    if panel.get("schema") != schema or panel.get("map_key") != key:
        raise Round0102Error(f"panel identity changed for {key}")
    return panel


def _noninferiority(
    treatment: Mapping[str, float],
    control: Mapping[str, float],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric, margin in MATCHED_NONINFERIORITY_MARGINS.items():
        delta = treatment[metric] - control[metric]
        boundary = -margin
        result[metric] = {
            "control": "r0079-balanced-120m",
            "control_value": control[metric],
            "treatment_150m": treatment[metric],
            "delta": delta,
            "maximum_allowed_decrease": margin,
            "passed": (
                delta >= boundary
                or math.isclose(
                    delta,
                    boundary,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ),
        }
    return result


def run_comparison(
    _active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    output = create_fresh_directory(
        job["outputs"][0],
        label="R0102 deliberate 120M/150M comparison",
    )
    control = _load_panel(
        str(job["control_panel"]),
        schema="round0080-registered-panel-v1",
        key=CONTROL_KEY,
    )
    matched = _load_panel(
        str(job["matched_panel"]),
        schema=PANEL_SCHEMA,
        key=MATCHED_KEY,
    )
    full = _load_panel(
        str(job["full_panel"]),
        schema=PANEL_SCHEMA,
        key=FULL_KEY,
    )
    if (
        matched.get("eligibility") != control.get("eligibility")
        or matched.get("scientific_universe")
        != control.get("scientific_universe")
        or matched["panel"].get("n") != control["panel"].get("n")
        or matched["panel"].get("anchor_hash")
        != control["panel"].get("anchor_hash")
        or matched["panel"].get("provenance", {}).get("hiD_reference_key")
        != control["panel"].get("provenance", {}).get("hiD_reference_key")
        or control.get("scientific_universe", {}).get(
            "excluded_rows_in_scoring"
        )
        is not False
    ):
        raise Round0102Error(
            "120M/150M matched panels do not share one representative universe"
        )

    comparison = _noninferiority(_metrics(matched), _metrics(control))
    matched_pass = all(item["passed"] for item in comparison.values())
    checks = dict(full.get("decision_checks") or {})
    non_density = {
        key: value
        for key, value in checks.items()
        if key != "density_at_least_0_60"
    }
    expected_non_density = {
        "ffr_at_least_0_40",
        "purity_k256_at_least_0_50",
        "purity_k1024_at_least_0_50",
        "heldout_projection_beats_untrained_floor",
        "recall_at_50_exceeds_recall_at_10",
        "coords_finite",
        "coords_not_collapsed",
        "embeddings_finite",
        "eligible_embeddings_nonzero",
    }
    if (
        set(non_density) != expected_non_density
        or full.get("scientific_universe", {}).get(
            "excluded_rows_in_scoring"
        )
        is not False
    ):
        raise Round0102Error("full-150M non-density selector changed")
    full_pass = all(value is True for value in non_density.values())
    density = _read_json(str(job["density_v2"]))
    validate_seal(density, label="R0102 density_v2 evaluation")
    density_cells = density.get("cells") or {}
    if (
        density.get("schema") != "round0102-density-v2-evaluation-v1"
        or float(density.get("registered_floor", -1.0))
        != DENSITY_V2_FLOOR
        or set(density_cells) != {"matched_120m", "full_150m"}
        or any(
            value.get("passed_registered_floor") is not True
            for value in density_cells.values()
        )
    ):
        density_pass = False
    else:
        density_pass = True
    supported = matched_pass and full_pass and density_pass
    body = {
        "schema": "round0102-scale-geometry-comparison-v1",
        "round_id": ROUND_ID,
        "panels": {
            "control_120m": expected_input_signature(
                str(job["control_panel"])
            ),
            "treatment_150m_matched": expected_input_signature(
                str(job["matched_panel"])
            ),
            "treatment_150m_full": expected_input_signature(
                str(job["full_panel"])
            ),
        },
        "same_row_120m_comparison": {
            "universe": (
                "exact R0065 balanced-120M retained representatives, one "
                "high-D reference and one anchor set for both models"
            ),
            "metrics_by_training_rung": {
                "120m": _metrics(control),
                "150m": _metrics(matched),
            },
            "150m_vs_120m_noninferiority": comparison,
            "projection_ffr_diagnostic": {
                "control": _metrics(control)["projection_ffr"],
                "treatment": _metrics(matched)["projection_ffr"],
                "delta": (
                    _metrics(matched)["projection_ffr"]
                    - _metrics(control)["projection_ffr"]
                ),
                "decision_gating": False,
                "reason": (
                    "R0084 observed one-seed movement of 0.0209, larger "
                    "than the retired 0.02 margin; no replacement margin "
                    "was calibrated"
                ),
            },
            "passed": matched_pass,
        },
        "full_150m_metrics": _metrics(full),
        "full_150m_checks": checks,
        "full_150m_non_density_checks": non_density,
        "full_150m_non_density_checks_passed": full_pass,
        "density_v2": {
            "receipt": expected_input_signature(str(job["density_v2"])),
            "registered_floor": DENSITY_V2_FLOOR,
            "cells": density_cells,
            "passed": density_pass,
        },
        "density_semantics": {
            "anchors": "representative-only",
            "candidate_universe": "representative-only",
            "selector": "fixed-density-v2-floor",
            "legacy_absolute_floor_reported": checks.get(
                "density_at_least_0_60"
            ),
            "legacy_absolute_floor_used_for_decision": False,
            "density_v2_threshold_calibrated_in_round": False,
        },
        "decision": {
            "150m_supported_as_deliberate_ladder_rung": supported,
            "reason": (
                "The 150M rung is supported only if it is non-inferior to "
                "the nearer 120M rung on identical 120M representative "
                "rows, both treatment density_v2 cells clear the fixed "
                "R0085 floor, and every full-150M non-density check passes."
            ),
        },
        "ood_is_reported_separately_and_non_gating": True,
    }
    receipt = seal(body)
    path = os.path.join(output, "scale-comparison.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def _map_anchor_family_sizes(
    anchors: np.ndarray,
    eligibility: Mapping[str, Any],
) -> np.ndarray:
    """Return the original exact-family size for retained anchor rows."""
    anchors = np.asarray(anchors, dtype=np.int64)
    representatives = np.asarray(
        eligibility["representative_rows"],
        dtype=np.int64,
    )
    counts = np.asarray(eligibility["family_counts"], dtype=np.int64)
    family_count = np.ones(anchors.shape, dtype=np.int64)
    positions = np.searchsorted(representatives, anchors)
    family = positions < len(representatives)
    valid = np.flatnonzero(family)
    family[valid] &= representatives[positions[valid]] == anchors[valid]
    family_count[family] = counts[positions[family]]
    return family_count


def _pearson_log_radius(
    high_radius: np.ndarray,
    low_radius: np.ndarray,
) -> float:
    high = np.log(
        np.asarray(high_radius, dtype=np.float64) + DENSITY_LOG_EPSILON
    )
    low = np.log(
        np.asarray(low_radius, dtype=np.float64) + DENSITY_LOG_EPSILON
    )
    if (
        high.ndim != 1
        or low.shape != high.shape
        or len(high) < 100
        or not np.isfinite(high).all()
        or not np.isfinite(low).all()
    ):
        raise Round0102Error("density_v2 radii are malformed")
    high = high - high.mean()
    low = low - low.mean()
    denominator = math.sqrt(
        float(np.dot(high, high)) * float(np.dot(low, low))
    )
    if not denominator > 0.0 or not math.isfinite(denominator):
        raise Round0102Error("density_v2 radius variance collapsed")
    value = float(np.dot(high, low) / denominator)
    if not math.isfinite(value):
        raise Round0102Error("density_v2 correlation is nonfinite")
    return value


def _density_universe(spec: Mapping[str, Any]) -> dict[str, Any]:
    from basemap.panel_v2 import _self_knn
    from basemap.round0036_pipeline import panel_config_identity
    from basemap.panel_v2 import PanelV2Config

    row_count = int(spec["row_count"])
    expected_retained = {
        120_000_000: 118_067_492,
        150_000_000: 147_221_757,
    }.get(row_count)
    if expected_retained is None:
        raise Round0102Error("density_v2 universe is not registered")
    eligibility = load_int8_eligibility(
        str(spec["eligibility_path"]),
        expected_sha256=str(spec["eligibility_sha256"]),
        row_count=row_count,
    )
    selector = RepresentativeRowSelector(
        eligibility["excluded_rows"],
        row_count=row_count,
        source=eligibility["signature"],
        policy=(
            "exact within-subset zero/duplicate exclusion; first ordered "
            "family member is the retained representative"
        ),
    )
    if selector.retained_count != expected_retained:
        raise Round0102Error("density_v2 representative count changed")

    reference_path = str(spec["reference_path"])
    reference_receipt_path = str(spec["reference_receipt_path"])
    anchor_rows_path = str(spec["anchor_rows_path"])
    reference_signature = expected_input_signature(reference_path)
    receipt_signature = expected_input_signature(reference_receipt_path)
    anchor_signature = expected_input_signature(anchor_rows_path)
    receipt = _read_json(reference_receipt_path)
    validate_seal(receipt, label=f"R0102 {spec['key']} reference")
    if (
        receipt.get("reference") != reference_signature
        or receipt.get("anchor_substrate_rows") != anchor_signature
        or receipt.get("eligibility") != eligibility["signature"]
        or (receipt.get("selector") or {}).get("representative_count")
        != selector.retained_count
    ):
        raise Round0102Error("density_v2 reference binding changed")
    with np.load(reference_path, allow_pickle=False) as archive:
        anchors = np.asarray(archive["anchor_ids"], dtype=np.int64)
        high_radius = np.asarray(archive["r_hd"], dtype=np.float64)
    anchor_rows = np.load(
        anchor_rows_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    if (
        anchors.shape != (DENSITY_ANCHOR_COUNT,)
        or high_radius.shape != (DENSITY_ANCHOR_COUNT,)
        or anchor_rows.shape != (DENSITY_ANCHOR_COUNT,)
        or not np.array_equal(
            np.asarray(anchor_rows, dtype=np.int64),
            selector.compact_to_global(anchors),
        )
        or np.any(high_radius < 0)
        or not np.isfinite(high_radius).all()
    ):
        raise Round0102Error("density_v2 anchor geometry changed")
    family_counts = _map_anchor_family_sizes(
        np.asarray(anchor_rows, dtype=np.int64),
        eligibility,
    )
    eligible = family_counts < DENSITY_FAMILY_SIZE_CUTOFF
    if int(eligible.sum()) < 9_000:
        raise Round0102Error("too many density_v2 anchors fail family filter")

    coordinates = CoordinateStream(str(spec["coordinates_path"]))
    if (
        len(coordinates) != row_count
        or coordinates.receipt.get("map_key") != spec["map_key"]
        or coordinates.receipt.get("model", {}).get("sha256")
        != spec["model_sha256"]
        or coordinates.receipt.get("eligibility")
        != eligibility["signature"]
    ):
        raise Round0102Error("density_v2 coordinate identity changed")
    config = PanelV2Config(**{
        key: tuple(value) if key == "k_clust" else value
        for key, value in panel_config_identity().items()
        if key != "formula_version"
    })
    representative_coordinates = RepresentativeArrayView(
        coordinates,
        selector,
    )
    _, distances, search_guard = _self_knn(
        representative_coordinates,
        anchors[eligible],
        DENSITY_K,
        config,
        hi_dim=False,
        want_dist=True,
        exact=True,
    )
    low_radius = np.asarray(distances.mean(1), dtype=np.float64)
    selected_high = high_radius[eligible]
    correlation = _pearson_log_radius(selected_high, low_radius)
    return {
        "key": str(spec["key"]),
        "map_key": str(spec["map_key"]),
        "correlation": correlation,
        "high_radius": selected_high,
        "low_radius": low_radius,
        "search_guard": search_guard,
        "identity": {
            "row_count": row_count,
            "retained_rows": selector.retained_count,
            "anchors_before_filter": DENSITY_ANCHOR_COUNT,
            "anchors_after_family_lt_16_filter": int(eligible.sum()),
            "maximum_anchor_family_size": int(family_counts.max()),
            "eligibility": eligibility["signature"],
            "reference": reference_signature,
            "reference_receipt": receipt_signature,
            "anchor_substrate_rows": anchor_signature,
            "coordinate_receipt": expected_input_signature(
                os.path.join(
                    str(spec["coordinates_path"]),
                    "actual-transform.json",
                )
            ),
        },
    }


def run_density_v2(
    _active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    output = create_fresh_directory(
        job["outputs"][0],
        label="R0102 fixed-floor density_v2 evaluation",
    )
    floor = float(job["registered_floor"])
    if floor != DENSITY_V2_FLOOR:
        raise Round0102Error("density_v2 floor changed")
    calibration = _read_json(str(job["calibration"]))
    validate_seal(calibration, label="R0102 density_v2 calibration")
    if (
        calibration.get("schema") != "round0085-density-v2-calibration-v1"
        or float(
            calibration.get("floor_calibration", {}).get(
                "registered_floor", -1.0
            )
        )
        != floor
    ):
        raise Round0102Error("density_v2 calibration binding changed")
    started = time.monotonic()
    values = [
        _density_universe(spec)
        for spec in job["universes"]
    ]
    if {value["key"] for value in values} != {
        "matched_120m", "full_150m",
    }:
        raise Round0102Error("density_v2 cells are incomplete")
    arrays_path = os.path.join(output, "density-v2-radii.npz")

    def write_arrays(path: str) -> None:
        arrays: dict[str, np.ndarray] = {}
        for value in values:
            arrays[f"{value['key']}__high_radius"] = value["high_radius"]
            arrays[f"{value['key']}__low_radius"] = value["low_radius"]
        with open(path, "wb") as handle:
            np.savez(handle, **arrays)

    atomic_build_new_file(arrays_path, write_arrays, immutable=True)
    body = {
        "schema": "round0102-density-v2-evaluation-v1",
        "round_id": ROUND_ID,
        "metric": (
            "Pearson correlation of log exact high-/low-D mean-k15 radii "
            "on reference anchors with original exact-family size <16"
        ),
        "registered_floor": floor,
        "calibration": expected_input_signature(str(job["calibration"])),
        "threshold_recalibrated": False,
        "cells": {
            value["key"]: {
                "map_key": value["map_key"],
                "density_v2": value["correlation"],
                "passed_registered_floor": value["correlation"] >= floor,
                "identity": value["identity"],
                "low_dim_exact_search_guard": value["search_guard"],
            }
            for value in values
        },
        "arrays": expected_input_signature(arrays_path),
        "wall_seconds": time.monotonic() - started,
        "training_performed": False,
    }
    receipt = seal(body)
    path = os.path.join(output, "density-v2-evaluation.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def _selector(
    *,
    eligibility_path: str,
    eligibility_sha256: str,
    row_count: int,
) -> tuple[RepresentativeRowSelector, dict[str, Any]]:
    eligibility = load_int8_eligibility(
        eligibility_path,
        expected_sha256=eligibility_sha256,
        row_count=row_count,
    )
    selector = RepresentativeRowSelector(
        eligibility["excluded_rows"],
        row_count=row_count,
        source=eligibility["signature"],
        policy="exact-nonzero-family-representative-after-subset-restriction",
    )
    return selector, eligibility


def _draw(
    output: str,
    *,
    coordinates: CoordinateStream,
    rows: np.ndarray,
    label: str,
) -> dict[str, Any]:
    points = coordinates[rows]
    if (
        not np.isfinite(points).all()
        or np.any(np.std(points, axis=0) <= 1e-8)
    ):
        raise Round0102Error(f"{label} render coordinates collapsed")
    atomic_build_new_file(
        output,
        lambda path: prior._draw_points(path, points, label),
        immutable=True,
    )
    return {
        "image": expected_input_signature(output),
        "axis_std": points.std(axis=0).astype(float).tolist(),
        "axis_span": np.ptp(points, axis=0).astype(float).tolist(),
    }


def run_renders(
    _active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    output = create_fresh_directory(
        job["outputs"][0],
        label="R0102 fixed scale renders",
    )
    selector120, eligibility120 = _selector(
        eligibility_path=str(job["matched_eligibility_path"]),
        eligibility_sha256=str(job["matched_eligibility_sha256"]),
        row_count=120_000_000,
    )
    sample120_path = str(job["matched_sample_rows"])
    sample120 = np.load(sample120_path, mmap_mode="r", allow_pickle=False)
    if (
        sample120.shape != (50_000,)
        or sample120.dtype.str != "<i8"
        or np.any(sample120 < 0)
        or np.any(sample120 >= selector120.row_count)
        or not np.all(selector120.is_retained(sample120))
    ):
        raise Round0102Error("R0080 matched-120M render sample changed")
    renders: dict[str, Any] = {}
    for definition in job["matched_maps"]:
        key = str(definition["map_key"])
        renders[key] = {
            **_draw(
                os.path.join(output, f"{key}.png"),
                coordinates=CoordinateStream(
                    str(definition["transform_output"])
                ),
                rows=sample120,
                label=MAP_LABELS[key],
            ),
            "sample_rows": expected_input_signature(sample120_path),
            "sample_rows_sha256": ordered_array_sha256(sample120),
            "sample_universe": "balanced-120m retained representatives",
        }

    selector150, eligibility150 = _selector(
        eligibility_path=str(job["full_eligibility_path"]),
        eligibility_sha256=str(job["full_eligibility_sha256"]),
        row_count=150_000_000,
    )
    rng = np.random.RandomState(20260728)
    compact = np.sort(
        rng.choice(selector150.retained_count, 50_000, replace=False)
    ).astype(np.int64)
    sample150 = selector150.compact_to_global(compact)
    sample150_path = os.path.join(output, "full-150m-sample-rows.npy")
    atomic_save_new_npy(sample150_path, sample150, immutable=True)
    renders[FULL_KEY] = {
        **_draw(
            os.path.join(output, f"{FULL_KEY}.png"),
            coordinates=CoordinateStream(str(job["full_transform"])),
            rows=sample150,
            label=MAP_LABELS[FULL_KEY],
        ),
        "sample_rows": expected_input_signature(sample150_path),
        "sample_rows_sha256": ordered_array_sha256(sample150),
        "sample_universe": "balanced-150m retained representatives",
        "eligibility": eligibility150["signature"],
    }
    definitions = seal({
        "schema": "scale-map-definitions-v1",
        "round_id": ROUND_ID,
        "maps": [{
            "key": FULL_KEY,
            "label": MAP_LABELS[FULL_KEY],
            "coordinates": "coordinates-r0101-150m",
            "panel": "panel-r0101-150m/panel.json",
            "render": f"semantic-renders/{FULL_KEY}.png",
            "training_round": "0101",
            "panel_schema": PANEL_SCHEMA,
            "density_semantics": (
                "density-v2-fixed-floor-plus-legacy-diagnostic"
            ),
        }],
    })
    definitions_path = os.path.join(output, "scale-map-definitions.json")
    atomic_write_new_json(definitions_path, definitions, immutable=True)
    body = {
        "schema": "round0102-scale-render-v1",
        "round_id": ROUND_ID,
        "matched_eligibility": eligibility120["signature"],
        "full_eligibility": eligibility150["signature"],
        "matched_sample_rows": expected_input_signature(sample120_path),
        "matched_sample_rows_sha256": ordered_array_sha256(sample120),
        "identical_semantic_rows_across_matched_maps": True,
        "full_sample_seed": 20260728,
        "full_sample_size": 50_000,
        "map_definitions": expected_input_signature(definitions_path),
        "renders": renders,
    }
    receipt = seal(body)
    path = os.path.join(output, "render-manifest.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_registry(
    _active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    output = create_fresh_directory(
        job["outputs"][0],
        label="R0102 immutable map registry snapshot",
    )
    from experiments import map_registry

    registry = map_registry.scan()
    entries = [
        item for item in registry["maps"]
        if item.get("round_id") == ROUND_ID
    ]
    maps = [item for item in entries if item.get("kind") == "round-map"]
    projections = [
        item for item in entries if item.get("kind") == "projection-map"
    ]
    required_label = MAP_LABELS[FULL_KEY]
    probes = {
        item.get("projection", {}).get("probe")
        for item in projections
        if item.get("base_map") == required_label
    }
    expected_probes = {"dadabase", "trec-covid", "code", "science", "latin"}
    if (
        len(maps) != 1
        or maps[0].get("map_label") != required_label
        or probes != expected_probes
    ):
        raise Round0102Error(
            "registry did not discover the 150M map and five projections"
        )
    snapshot_path = os.path.join(output, "registry-snapshot.json")
    payload = json.dumps(registry, indent=1).encode("utf-8")

    def write_snapshot(path: str) -> None:
        with open(path, "wb") as handle:
            handle.write(payload)

    atomic_build_new_file(snapshot_path, write_snapshot, immutable=True)
    history_path = map_registry.write_registry(registry)
    map_registry.publish(registry)
    snapshot = expected_input_signature(snapshot_path)
    current = expected_input_signature(str(map_registry.REGISTRY_PATH))
    history = (
        expected_input_signature(str(history_path))
        if history_path is not None
        else None
    )
    body = {
        "schema": "round0102-map-registry-publication-v1",
        "round_id": ROUND_ID,
        "immutable_registry_snapshot": snapshot,
        "mutable_registry_after_publish": current,
        "mutable_registry_content_sha256": map_registry._content_sha(
            registry
        ),
        "content_addressed_history_snapshot_if_new": history,
        "mutable_view_equality_is_nongating": True,
        "map_ids": sorted(item["map_id"] for item in entries),
        "base_map": required_label,
        "projection_probes": sorted(expected_probes),
        "local_site_url": map_registry.SITE_URL,
    }
    receipt = seal(body)
    path = os.path.join(output, "registry-publication.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise RuntimeError("R0102 handler received another queue")
    if job is None:
        raise RuntimeError("R0102 handler requires the exact job")
    _configure_shared()
    handlers = {
        "transform": shared.run_transform,
        "high_d_reference": shared.run_high_d_reference,
        "panel": shared.run_panel,
        "ood": shared.run_ood,
        "density_v2": run_density_v2,
        "comparison": run_comparison,
        "renders": run_renders,
        "registry": run_registry,
    }
    try:
        handler = handlers[str(job["action"])]
    except KeyError as exc:
        raise RuntimeError(
            f"unknown R0102 action {job.get('action')!r}"
        ) from exc
    return handler(active, job)
