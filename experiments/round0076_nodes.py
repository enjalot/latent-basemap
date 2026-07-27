"""Evaluate the deliberate balanced-90M rung on matched and full universes."""
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
from basemap.round0064_evaluation import (
    Round0064Error,
    seal,
    validate_seal,
)
from experiments import round0064_nodes as shared
from experiments import round0069_nodes as prior


ROUND_ID = "0076"
PANEL_SCHEMA = "round0076-registered-panel-v1"
FULL_KEY = "r0075-90m-on-90m"
MATCHED_KEY = "r0075-90m-on-30m"
MAP_LABELS = {
    "r0061-30m-on-30m": "r0061-balanced-30m-seed42",
    "r0068-45m-on-30m": "r0068-balanced-45m-seed42-on-matched-30m",
    "r0063-60m-on-30m": "r0063-balanced-60m-seed42-on-matched-30m",
    MATCHED_KEY: "r0075-balanced-90m-seed42-on-matched-30m",
    FULL_KEY: "r0075-balanced-90m-seed42",
}
MATCHED_NONINFERIORITY_MARGINS = dict(
    prior.MATCHED_NONINFERIORITY_MARGINS
)


class Round0076Error(Round0064Error):
    """The balanced-90M scale evaluation contract was violated."""


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0076Error(f"{path} is not a JSON object")
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


def _load_panel(spec: Mapping[str, Any]) -> dict[str, Any]:
    panel = _read_json(str(spec["path"]))
    validate_seal(panel, label=f"R0076 {spec['key']} panel")
    if (
        panel.get("schema") != spec["schema"]
        or panel.get("map_key") != spec["key"]
    ):
        raise Round0076Error(f"panel identity changed for {spec['key']}")
    return panel


def _noninferiority(
    treatment: Mapping[str, float],
    control: Mapping[str, float],
    *,
    control_label: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric, margin in MATCHED_NONINFERIORITY_MARGINS.items():
        delta = treatment[metric] - control[metric]
        reported_delta = round(delta, 6)
        result[metric] = {
            "control": control_label,
            "control_value": control[metric],
            "treatment_90m": treatment[metric],
            "delta": reported_delta,
            "maximum_allowed_decrease": margin,
            # The registered panel metrics and this receipt are interpreted at
            # six decimals. Use that same value for the decision so binary
            # floating-point noise cannot reject an exact boundary equality.
            "passed": reported_delta >= -margin,
        }
    return result


def run_comparison(
    _active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    output = create_fresh_directory(
        job["outputs"][0],
        label="R0076 deliberate 30M/45M/60M/90M comparison",
    )
    panels = {
        name: _load_panel(spec)
        for name, spec in job["panels"].items()
    }
    matched_names = (
        "control_30m",
        "rung_45m",
        "rung_60m",
        "treatment_90m_matched",
    )
    matched = [panels[name] for name in matched_names]
    control = matched[0]
    scientific = [panel["panel"] for panel in matched]
    if (
        any(
            panel.get("eligibility") != control.get("eligibility")
            for panel in matched[1:]
        )
        or any(
            panel.get("scientific_universe")
            != control.get("scientific_universe")
            for panel in matched[1:]
        )
        or len({panel.get("n") for panel in scientific}) != 1
        or len({panel.get("anchor_hash") for panel in scientific}) != 1
        or len({
            panel.get("provenance", {}).get("hiD_reference_key")
            for panel in scientific
        }) != 1
        or any(
            panel.get("scientific_universe", {}).get(
                "excluded_rows_in_scoring"
            ) is not False
            for panel in matched
        )
    ):
        raise Round0076Error(
            "matched scale panels do not share one representative universe"
        )

    anchor_path = str(job["anchor_leverage"])
    anchor = _read_json(anchor_path)
    validate_seal(anchor, label="R0076 R0074 anchor-leverage evidence")
    anchor_signature = expected_input_signature(anchor_path)
    interpretation = anchor.get("interpretation") or {}
    if (
        anchor_signature["sha256"] != job["anchor_leverage_sha256"]
        or anchor.get("schema") != "round0074-duplicate-anchor-leverage-v1"
        or anchor.get("legacy_density_exactly_replayed") is not True
        or interpretation.get("classification")
        != "duplicate-heavy-anchor-leverage-supported"
        or interpretation.get("calibrates_density_threshold") is not False
    ):
        raise Round0076Error("R0074 density interpretation changed")

    metrics = {
        name: _metrics(panels[name])
        for name in matched_names
    }
    treatment = metrics["treatment_90m_matched"]
    versus_30m = _noninferiority(
        treatment,
        metrics["control_30m"],
        control_label="r0061-balanced-30m",
    )
    versus_60m = _noninferiority(
        treatment,
        metrics["rung_60m"],
        control_label="r0063-balanced-60m",
    )
    matched_pass = all(
        item["passed"]
        for comparison in (versus_30m, versus_60m)
        for item in comparison.values()
    )

    full = panels["treatment_90m_full"]
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
        ) is not False
    ):
        raise Round0076Error("full-90M non-density selector changed")
    full_pass = all(value is True for value in non_density.values())
    rung_supported = matched_pass and full_pass

    body = {
        "schema": "round0076-scale-geometry-comparison-v1",
        "round_id": ROUND_ID,
        "panels": {
            name: expected_input_signature(str(spec["path"]))
            for name, spec in job["panels"].items()
        },
        "same_row_30m_comparison": {
            "universe": (
                "exact R0053 balanced-30M retained representatives, one "
                "high-D reference and one anchor set for all four models"
            ),
            "metrics_by_training_rung": {
                "30m": metrics["control_30m"],
                "45m": metrics["rung_45m"],
                "60m": metrics["rung_60m"],
                "90m": treatment,
            },
            "90m_vs_30m_noninferiority": versus_30m,
            "90m_vs_60m_noninferiority": versus_60m,
            "passed": matched_pass,
        },
        "full_90m_metrics": _metrics(full),
        "full_90m_checks": checks,
        "full_90m_non_density_checks": non_density,
        "full_90m_non_density_checks_passed": full_pass,
        "density_semantics": {
            "anchors": "representative-only",
            "candidate_universe": "representative-only",
            "selector": "relative-noninferiority-only",
            "legacy_absolute_floor_reported": checks.get(
                "density_at_least_0_60"
            ),
            "legacy_absolute_floor_used_for_decision": False,
            "threshold_calibrated": False,
            "anchor_leverage_evidence": anchor_signature,
        },
        "decision": {
            "90m_supported_as_deliberate_ladder_rung": rung_supported,
            "prepare_120m_search_and_graph_if_true": rung_supported,
            "train_120m_without_separate_round": False,
            "reason": (
                "The 90M rung is supported only if its model is non-inferior "
                "to both the 30M control and the nearer 60M rung on identical "
                "30M rows, and every full-90M non-density integrity/quality "
                "check passes. R0074 invalidates the duplicate-dominated "
                "legacy absolute density floor; this round does not tune one."
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


def run_renders(
    _active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    output = create_fresh_directory(
        job["outputs"][0],
        label="R0076 fixed scale renders",
    )
    selector30, eligibility30 = _selector(
        eligibility_path=str(job["eligibility_path"]),
        eligibility_sha256=str(job["eligibility_sha256"]),
        row_count=int(job["row_count"]),
    )
    sample30_path = str(job["matched_sample_rows"])
    sample30 = np.load(sample30_path, mmap_mode="r", allow_pickle=False)
    if (
        sample30.shape != (50_000,)
        or sample30.dtype.str != "<i8"
        or np.any(sample30 < 0)
        or np.any(sample30 >= selector30.row_count)
        or not np.all(selector30.is_retained(sample30))
    ):
        raise Round0076Error("R0064 matched render sample changed")

    renders: dict[str, Any] = {}
    for definition in job["matched_maps"]:
        key = str(definition["map_key"])
        coordinates = CoordinateStream(str(definition["transform_output"]))
        points = coordinates[sample30]
        if (
            not np.isfinite(points).all()
            or np.any(np.std(points, axis=0) <= 1e-8)
        ):
            raise Round0076Error(f"{key} render coordinates collapsed")
        image_path = os.path.join(output, f"{key}.png")
        atomic_build_new_file(
            image_path,
            lambda path, values=points, label=MAP_LABELS[key]:
                prior._draw_points(path, values, label),
            immutable=True,
        )
        renders[key] = {
            "image": expected_input_signature(image_path),
            "axis_std": points.std(axis=0).astype(float).tolist(),
            "axis_span": np.ptp(points, axis=0).astype(float).tolist(),
            "sample_rows": expected_input_signature(sample30_path),
            "sample_rows_sha256": ordered_array_sha256(sample30),
            "sample_universe": "balanced-30m retained representatives",
        }

    full_rows = int(job["full_row_count"])
    selector90, eligibility90 = _selector(
        eligibility_path=str(job["full_eligibility_path"]),
        eligibility_sha256=str(job["full_eligibility_sha256"]),
        row_count=full_rows,
    )
    rng = np.random.RandomState(20260727)
    compact = np.sort(
        rng.choice(selector90.retained_count, 50_000, replace=False)
    ).astype(np.int64)
    sample90 = selector90.compact_to_global(compact)
    sample90_path = os.path.join(output, "full-90m-sample-rows.npy")
    atomic_save_new_npy(sample90_path, sample90, immutable=True)
    full_coordinates = CoordinateStream(str(job["full_transform"]))
    full_points = full_coordinates[sample90]
    if (
        not np.isfinite(full_points).all()
        or np.any(np.std(full_points, axis=0) <= 1e-8)
    ):
        raise Round0076Error("full 90M render coordinates collapsed")
    full_image_path = os.path.join(output, f"{FULL_KEY}.png")
    atomic_build_new_file(
        full_image_path,
        lambda path: prior._draw_points(
            path,
            full_points,
            MAP_LABELS[FULL_KEY],
        ),
        immutable=True,
    )
    renders[FULL_KEY] = {
        "image": expected_input_signature(full_image_path),
        "axis_std": full_points.std(axis=0).astype(float).tolist(),
        "axis_span": np.ptp(full_points, axis=0).astype(float).tolist(),
        "sample_rows": expected_input_signature(sample90_path),
        "sample_rows_sha256": ordered_array_sha256(sample90),
        "sample_universe": "balanced-90m retained representatives",
        "eligibility": eligibility90["signature"],
    }

    definitions_body = {
        "schema": "scale-map-definitions-v1",
        "round_id": ROUND_ID,
        "maps": [{
            "key": FULL_KEY,
            "label": MAP_LABELS[FULL_KEY],
            "coordinates": "coordinates-r0075-90m",
            "panel": "panel-r0075-90m/panel.json",
            "render": f"semantic-renders/{FULL_KEY}.png",
            "training_round": "0075",
            "panel_schema": PANEL_SCHEMA,
            "density_semantics": "representative-relative-v1",
        }],
    }
    definitions = seal(definitions_body)
    definitions_path = os.path.join(output, "scale-map-definitions.json")
    atomic_write_new_json(
        definitions_path,
        definitions,
        immutable=True,
    )
    body = {
        "schema": "round0076-scale-render-v1",
        "round_id": ROUND_ID,
        "matched_eligibility": eligibility30["signature"],
        "full_eligibility": eligibility90["signature"],
        "matched_sample_rows": expected_input_signature(sample30_path),
        "matched_sample_rows_sha256": ordered_array_sha256(sample30),
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
        label="R0076 immutable map registry snapshot",
    )
    from experiments import map_registry

    registry = map_registry.scan()
    entries = [
        item
        for item in registry["maps"]
        if item.get("round_id") == ROUND_ID
    ]
    map_entries = [
        item for item in entries if item.get("kind") == "round-map"
    ]
    projection_entries = [
        item for item in entries if item.get("kind") == "projection-map"
    ]
    required_label = MAP_LABELS[FULL_KEY]
    required_probes = {
        "dadabase",
        "trec-covid",
        "code",
        "science",
        "latin",
    }
    observed_probes = {
        item.get("projection", {}).get("probe")
        for item in projection_entries
        if item.get("base_map") == required_label
    }
    if (
        len(map_entries) != 1
        or map_entries[0].get("map_label") != required_label
        or observed_probes != required_probes
    ):
        raise Round0076Error(
            "registry did not discover the 90M map and five projections"
        )

    snapshot_path = os.path.join(output, "registry-snapshot.json")
    payload = json.dumps(registry, indent=1).encode("utf-8")

    def write_snapshot(path: str) -> None:
        with open(path, "wb") as handle:
            handle.write(payload)

    atomic_build_new_file(
        snapshot_path,
        write_snapshot,
        immutable=True,
    )
    map_registry.REGISTRY_PATH.write_bytes(payload)
    map_registry.publish(registry)
    snapshot = expected_input_signature(snapshot_path)
    current = expected_input_signature(str(map_registry.REGISTRY_PATH))
    if (
        snapshot["sha256"] != current["sha256"]
        or snapshot["bytes"] != current["bytes"]
    ):
        raise Round0076Error("published registry differs from snapshot")
    body = {
        "schema": "round0076-map-registry-publication-v1",
        "round_id": ROUND_ID,
        "immutable_registry_snapshot": snapshot,
        "mutable_registry_after_publish": current,
        "map_ids": sorted(item["map_id"] for item in entries),
        "base_map": required_label,
        "projection_probes": sorted(required_probes),
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
        raise RuntimeError("R0076 handler received another queue")
    if job is None:
        raise RuntimeError("R0076 handler requires the exact job")
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
            f"unknown R0076 action {job.get('action')!r}"
        ) from exc
    return handler(active, job)
