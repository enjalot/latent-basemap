"""Evaluate the deliberate balanced-120M rung on matched and full universes."""
from __future__ import annotations

import json
import os
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
from basemap.round0040_program import RepresentativeRowSelector
from basemap.round0064_evaluation import seal, validate_seal
from experiments import round0064_nodes as shared
from experiments import round0069_nodes as prior
from experiments.round0076_nodes import MATCHED_NONINFERIORITY_MARGINS


ROUND_ID = "0080"
PANEL_SCHEMA = "round0080-registered-panel-v1"
CONTROL_KEY = "r0075-90m-on-90m"
MATCHED_KEY = "r0079-120m-on-90m"
FULL_KEY = "r0079-120m-on-120m"
MAP_LABELS = {
    CONTROL_KEY: "r0075-balanced-90m-seed42",
    MATCHED_KEY: "r0079-balanced-120m-seed42-on-matched-90m",
    FULL_KEY: "r0079-balanced-120m-seed42",
}


class Round0080Error(RuntimeError):
    """The balanced-120M scale evaluation contract was violated."""


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0080Error(f"{path} is not a JSON object")
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
    validate_seal(panel, label=f"R0080 {key} panel")
    if panel.get("schema") != schema or panel.get("map_key") != key:
        raise Round0080Error(f"panel identity changed for {key}")
    return panel


def _noninferiority(
    treatment: Mapping[str, float],
    control: Mapping[str, float],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric, margin in MATCHED_NONINFERIORITY_MARGINS.items():
        delta = round(treatment[metric] - control[metric], 6)
        result[metric] = {
            "control": "r0075-balanced-90m",
            "control_value": control[metric],
            "treatment_120m": treatment[metric],
            "delta": delta,
            "maximum_allowed_decrease": margin,
            "passed": delta >= -margin,
        }
    return result


def run_comparison(
    _active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    output = create_fresh_directory(
        job["outputs"][0],
        label="R0080 deliberate 90M/120M comparison",
    )
    control = _load_panel(
        str(job["control_panel"]),
        schema="round0076-registered-panel-v1",
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
        raise Round0080Error(
            "90M/120M matched panels do not share one representative universe"
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
        raise Round0080Error("full-120M non-density selector changed")
    full_pass = all(value is True for value in non_density.values())
    supported = matched_pass and full_pass
    body = {
        "schema": "round0080-scale-geometry-comparison-v1",
        "round_id": ROUND_ID,
        "panels": {
            "control_90m": expected_input_signature(
                str(job["control_panel"])
            ),
            "treatment_120m_matched": expected_input_signature(
                str(job["matched_panel"])
            ),
            "treatment_120m_full": expected_input_signature(
                str(job["full_panel"])
            ),
        },
        "same_row_90m_comparison": {
            "universe": (
                "exact R0071 balanced-90M retained representatives, one "
                "high-D reference and one anchor set for both models"
            ),
            "metrics_by_training_rung": {
                "90m": _metrics(control),
                "120m": _metrics(matched),
            },
            "120m_vs_90m_noninferiority": comparison,
            "passed": matched_pass,
        },
        "full_120m_metrics": _metrics(full),
        "full_120m_checks": checks,
        "full_120m_non_density_checks": non_density,
        "full_120m_non_density_checks_passed": full_pass,
        "density_semantics": {
            "anchors": "representative-only",
            "candidate_universe": "representative-only",
            "selector": "relative-noninferiority-only",
            "legacy_absolute_floor_reported": checks.get(
                "density_at_least_0_60"
            ),
            "legacy_absolute_floor_used_for_decision": False,
            "threshold_calibrated": False,
        },
        "decision": {
            "120m_supported_as_deliberate_ladder_rung": supported,
            "reason": (
                "The 120M rung is supported only if it is non-inferior to "
                "the nearer 90M rung on identical 90M representative rows "
                "and every full-120M non-density integrity/quality check "
                "passes."
            ),
        },
        "ood_is_reported_separately_and_non_gating": True,
    }
    receipt = seal(body)
    path = os.path.join(output, "scale-comparison.json")
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
        raise Round0080Error(f"{label} render coordinates collapsed")
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
        label="R0080 fixed scale renders",
    )
    selector90, eligibility90 = _selector(
        eligibility_path=str(job["matched_eligibility_path"]),
        eligibility_sha256=str(job["matched_eligibility_sha256"]),
        row_count=90_000_000,
    )
    sample90_path = str(job["matched_sample_rows"])
    sample90 = np.load(sample90_path, mmap_mode="r", allow_pickle=False)
    if (
        sample90.shape != (50_000,)
        or sample90.dtype.str != "<i8"
        or np.any(sample90 < 0)
        or np.any(sample90 >= selector90.row_count)
        or not np.all(selector90.is_retained(sample90))
    ):
        raise Round0080Error("R0076 matched-90M render sample changed")
    renders: dict[str, Any] = {}
    for definition in job["matched_maps"]:
        key = str(definition["map_key"])
        renders[key] = {
            **_draw(
                os.path.join(output, f"{key}.png"),
                coordinates=CoordinateStream(
                    str(definition["transform_output"])
                ),
                rows=sample90,
                label=MAP_LABELS[key],
            ),
            "sample_rows": expected_input_signature(sample90_path),
            "sample_rows_sha256": ordered_array_sha256(sample90),
            "sample_universe": "balanced-90m retained representatives",
        }

    selector120, eligibility120 = _selector(
        eligibility_path=str(job["full_eligibility_path"]),
        eligibility_sha256=str(job["full_eligibility_sha256"]),
        row_count=120_000_000,
    )
    rng = np.random.RandomState(20260727)
    compact = np.sort(
        rng.choice(selector120.retained_count, 50_000, replace=False)
    ).astype(np.int64)
    sample120 = selector120.compact_to_global(compact)
    sample120_path = os.path.join(output, "full-120m-sample-rows.npy")
    atomic_save_new_npy(sample120_path, sample120, immutable=True)
    renders[FULL_KEY] = {
        **_draw(
            os.path.join(output, f"{FULL_KEY}.png"),
            coordinates=CoordinateStream(str(job["full_transform"])),
            rows=sample120,
            label=MAP_LABELS[FULL_KEY],
        ),
        "sample_rows": expected_input_signature(sample120_path),
        "sample_rows_sha256": ordered_array_sha256(sample120),
        "sample_universe": "balanced-120m retained representatives",
        "eligibility": eligibility120["signature"],
    }
    definitions = seal({
        "schema": "scale-map-definitions-v1",
        "round_id": ROUND_ID,
        "maps": [{
            "key": FULL_KEY,
            "label": MAP_LABELS[FULL_KEY],
            "coordinates": "coordinates-r0079-120m",
            "panel": "panel-r0079-120m/panel.json",
            "render": f"semantic-renders/{FULL_KEY}.png",
            "training_round": "0079",
            "panel_schema": PANEL_SCHEMA,
            "density_semantics": "representative-relative-v1",
        }],
    })
    definitions_path = os.path.join(output, "scale-map-definitions.json")
    atomic_write_new_json(definitions_path, definitions, immutable=True)
    body = {
        "schema": "round0080-scale-render-v1",
        "round_id": ROUND_ID,
        "matched_eligibility": eligibility90["signature"],
        "full_eligibility": eligibility120["signature"],
        "matched_sample_rows": expected_input_signature(sample90_path),
        "matched_sample_rows_sha256": ordered_array_sha256(sample90),
        "identical_semantic_rows_across_matched_maps": True,
        "full_sample_seed": 20260727,
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
        label="R0080 immutable map registry snapshot",
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
        raise Round0080Error(
            "registry did not discover the 120M map and five projections"
        )
    snapshot_path = os.path.join(output, "registry-snapshot.json")
    payload = json.dumps(registry, indent=1).encode("utf-8")

    def write_snapshot(path: str) -> None:
        with open(path, "wb") as handle:
            handle.write(payload)

    atomic_build_new_file(snapshot_path, write_snapshot, immutable=True)
    map_registry.REGISTRY_PATH.write_bytes(payload)
    map_registry.publish(registry)
    snapshot = expected_input_signature(snapshot_path)
    current = expected_input_signature(str(map_registry.REGISTRY_PATH))
    if (
        snapshot["sha256"] != current["sha256"]
        or snapshot["bytes"] != current["bytes"]
    ):
        raise Round0080Error("published registry differs from snapshot")
    body = {
        "schema": "round0080-map-registry-publication-v1",
        "round_id": ROUND_ID,
        "immutable_registry_snapshot": snapshot,
        "mutable_registry_after_publish": current,
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
        raise RuntimeError("R0080 handler received another queue")
    if job is None:
        raise RuntimeError("R0080 handler requires the exact job")
    _configure_shared()
    handlers = {
        "transform": shared.run_transform,
        "high_d_reference": shared.run_high_d_reference,
        "panel": shared.run_panel,
        "ood": shared.run_ood,
        "comparison": run_comparison,
        "renders": run_renders,
        "registry": run_registry,
    }
    try:
        handler = handlers[str(job["action"])]
    except KeyError as exc:
        raise RuntimeError(
            f"unknown R0080 action {job.get('action')!r}"
        ) from exc
    return handler(active, job)
