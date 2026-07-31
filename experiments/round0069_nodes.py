"""Fresh-process nodes for the balanced 45M scale-geometry evaluation."""
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
from basemap.round0036_pipeline import CoordinateStream
from basemap.round0040_program import RepresentativeArrayView
from basemap.round0064_evaluation import (
    Round0064Error,
    load_substrate,
    seal,
    validate_seal,
)
from experiments import round0064_nodes as shared


ROUND_ID = "0069"
PANEL_SCHEMA = "round0069-registered-panel-v1"
MAP_LABELS = {
    "r0061-30m-on-30m": "r0061-balanced-30m-seed42",
    "r0068-45m-on-30m": "r0068-balanced-45m-seed42-on-matched-30m",
    "r0068-45m-on-45m": "r0068-balanced-45m-seed42",
    "r0063-60m-on-30m": "r0063-balanced-60m-seed42-on-matched-30m",
}
MATCHED_NONINFERIORITY_MARGINS = {
    "ffr": 0.02,
    "density": 0.05,
    "purity_k256": 0.05,
    "purity_k1024": 0.05,
    "projection_ffr": 0.02,
}


class Round0069Error(Round0064Error):
    """The registered 45M evaluation contract was violated."""


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0069Error(f"{path} is not a JSON object")
    return value


def _configure_shared() -> None:
    # Every queue node runs in a fresh process. Reusing the accepted R0064
    # transform/panel implementation this way keeps one implementation of the
    # expensive geometry math while giving R0069 its own receipt identity.
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


def _density_groups(
    high_radius: np.ndarray,
    low_radius: np.ndarray,
    corpus_labels: np.ndarray,
) -> dict[str, Any]:
    eps = 1e-12
    high_log = np.log(np.asarray(high_radius, dtype=np.float64) + eps)
    low_log = np.log(np.asarray(low_radius, dtype=np.float64) + eps)
    groups: dict[str, Any] = {}
    for label in ("fineweb", "redpajama", "pile"):
        mask = corpus_labels == label
        if int(mask.sum()) < 2:
            raise Round0069Error(f"density group {label} is undersampled")
        groups[label] = {
            "anchors": int(mask.sum()),
            "correlation": round(
                float(np.corrcoef(high_log[mask], low_log[mask])[0, 1]),
                4,
            ),
            "mean_log_high_d_radius": round(
                float(high_log[mask].mean()),
                6,
            ),
            "mean_log_low_d_radius": round(
                float(low_log[mask].mean()),
                6,
            ),
        }
    return {
        "global_correlation": round(
            float(np.corrcoef(high_log, low_log)[0, 1]),
            4,
        ),
        "anchors": int(len(high_log)),
        "by_corpus": groups,
    }


def run_density_diagnostic(
    _active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    """Recompute matched density for all three rungs and expose corpus strata."""
    from basemap.panel_v2 import _self_knn, load_hiD_reference

    output = create_fresh_directory(
        job["outputs"][0],
        label="R0069 matched density diagnostic",
    )
    started = time.monotonic()
    encoded, selector, _, eligibility = shared._substrate(job)
    del encoded
    reference_path = os.path.join(
        str(job["reference_output"]),
        "reference.npz",
    )
    reference = load_hiD_reference(reference_path)
    anchors = np.asarray(reference["anchor_ids"], dtype=np.int64)
    high_radius = np.asarray(reference["r_hd"], dtype=np.float64)
    substrate_rows = selector.compact_to_global(anchors)
    rows_per_corpus = int(job["rows_per_corpus"])
    names = np.asarray(["fineweb", "redpajama", "pile"], dtype="<U10")
    corpus_index = substrate_rows // rows_per_corpus
    if np.any(corpus_index < 0) or np.any(corpus_index >= len(names)):
        raise Round0069Error("matched density corpus assignment changed")
    corpus_labels = names[corpus_index]
    config = shared._panel_config()

    maps: dict[str, Any] = {}
    radii: dict[str, np.ndarray] = {}
    for definition in job["matched_maps"]:
        key = str(definition["map_key"])
        panel_path = str(definition["panel_path"])
        panel = _read_json(panel_path)
        validate_seal(panel, label=f"R0069 density panel {key}")
        if panel.get("map_key") != key:
            raise Round0069Error(f"density panel key changed for {key}")
        full_coordinates = CoordinateStream(
            str(definition["transform_output"])
        )
        coordinates = RepresentativeArrayView(full_coordinates, selector)
        _, distances, _ = _self_knn(
            coordinates,
            anchors,
            config.k_density,
            config,
            hi_dim=False,
            want_dist=True,
        )
        low_radius = distances.mean(1)
        summary = _density_groups(
            high_radius,
            low_radius,
            corpus_labels,
        )
        registered = float(panel["panel"]["density"])
        if summary["global_correlation"] != registered:
            raise Round0069Error(
                f"{key} density replay {summary['global_correlation']} "
                f"does not match registered {registered}"
            )
        maps[key] = {
            "map_label": MAP_LABELS[key],
            "panel": expected_input_signature(panel_path),
            "density": summary,
        }
        radii[key] = low_radius

    archive_path = os.path.join(output, "matched-density-radii.npz")

    def write_archive(path: str) -> None:
        with open(path, "wb") as handle:
            np.savez(
                handle,
                anchor_compact_rows=anchors,
                anchor_substrate_rows=substrate_rows,
                high_d_radius=high_radius,
                corpus_labels=corpus_labels,
                **{
                    f"low_d_radius_{key.replace('-', '_')}": value
                    for key, value in radii.items()
                },
            )

    atomic_build_new_file(archive_path, write_archive, immutable=True)
    body = {
        "schema": "round0069-matched-density-diagnostic-v1",
        "round_id": ROUND_ID,
        "reference": expected_input_signature(reference_path),
        "eligibility": eligibility["signature"],
        "anchor_compact_rows_sha256": ordered_array_sha256(anchors),
        "anchor_substrate_rows_sha256": ordered_array_sha256(
            substrate_rows
        ),
        "maps": maps,
        "radii_archive": expected_input_signature(archive_path),
        "interpretation_contract": {
            "legacy_absolute_floor": 0.60,
            "legacy_floor_is_recalibrated_here": False,
            "purpose": (
                "separate exact matched-rung density trend from corpus-stratum "
                "composition; no post-hoc replacement threshold"
            ),
        },
        "wall_seconds": time.monotonic() - started,
    }
    receipt = seal(body)
    path = os.path.join(output, "density-diagnostic.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_comparison(
    _active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    output = create_fresh_directory(
        job["outputs"][0],
        label="R0069 30M/45M/60M scale comparison",
    )
    panel_defs = {
        name: (str(spec["path"]), str(spec["key"]), str(spec["schema"]))
        for name, spec in job["panels"].items()
    }
    panels: dict[str, dict[str, Any]] = {}
    for name, (path, key, schema) in panel_defs.items():
        panel = _read_json(path)
        validate_seal(panel, label=f"R0069 {name} panel")
        if (
            panel.get("schema") != schema
            or panel.get("map_key") != key
        ):
            raise Round0069Error(f"{name} panel identity changed")
        panels[name] = panel

    control = panels["control_30m"]
    treatment = panels["treatment_45m_matched"]
    upper = panels["upper_60m_matched"]
    matched = (control, treatment, upper)
    scientific = [panel["panel"] for panel in matched]
    if (
        any(panel.get("eligibility") != control.get("eligibility")
            for panel in matched[1:])
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
    ):
        raise Round0069Error(
            "30M/45M/60M matched panels do not share one exact universe"
        )

    baseline_metrics = _metrics(control)
    treatment_metrics = _metrics(treatment)
    upper_metrics = _metrics(upper)
    noninferiority: dict[str, Any] = {}
    endpoint_trend: dict[str, Any] = {}
    for metric, margin in MATCHED_NONINFERIORITY_MARGINS.items():
        delta = treatment_metrics[metric] - baseline_metrics[metric]
        noninferiority[metric] = {
            "control_30m": baseline_metrics[metric],
            "treatment_45m": treatment_metrics[metric],
            "delta": round(delta, 6),
            "maximum_allowed_decrease": margin,
            "passed": delta >= -margin,
        }
        lower_bound = min(
            baseline_metrics[metric],
            upper_metrics[metric],
        )
        upper_bound = max(
            baseline_metrics[metric],
            upper_metrics[metric],
        )
        endpoint_trend[metric] = {
            "model_30m": baseline_metrics[metric],
            "model_45m": treatment_metrics[metric],
            "model_60m": upper_metrics[metric],
            "45m_between_30m_and_60m_inclusive": (
                lower_bound <= treatment_metrics[metric] <= upper_bound
            ),
            "45m_minus_60m": round(
                treatment_metrics[metric] - upper_metrics[metric],
                6,
            ),
        }

    full = panels["treatment_45m_full"]
    checks = dict(full.get("decision_checks") or {})
    required_non_density = {
        key: value
        for key, value in checks.items()
        if key != "density_at_least_0_60"
    }
    expected_check_names = {
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
    if set(required_non_density) != expected_check_names:
        raise Round0069Error("45M non-density selector fields changed")
    matched_pass = all(
        value["passed"] for value in noninferiority.values()
    )
    full_non_density_pass = all(required_non_density.values())
    density_path = os.path.join(
        str(job["density_diagnostic"]),
        "density-diagnostic.json",
    )
    density = _read_json(density_path)
    validate_seal(density, label="R0069 matched density diagnostic")
    for key, panel in zip(
        (
            "r0061-30m-on-30m",
            "r0068-45m-on-30m",
            "r0063-60m-on-30m",
        ),
        matched,
        strict=True,
    ):
        replayed = density["maps"][key]["density"]["global_correlation"]
        if replayed != panel["panel"]["density"]:
            raise Round0069Error("density diagnostic/panel mismatch")

    candidate_rung = matched_pass and full_non_density_pass
    body = {
        "schema": "round0069-scale-geometry-comparison-v1",
        "round_id": ROUND_ID,
        "panels": {
            name: expected_input_signature(path)
            for name, (path, _, _) in panel_defs.items()
        },
        "same_row_30m_comparison": {
            "universe": (
                "exact R0053 balanced-30M retained representatives, one "
                "high-D reference and one anchor set for all three models"
            ),
            "45m_vs_30m_noninferiority": noninferiority,
            "45m_vs_30m_passed": matched_pass,
            "30m_45m_60m_endpoint_trend": endpoint_trend,
        },
        "full_45m_metrics": _metrics(full),
        "full_45m_checks": checks,
        "full_45m_non_density_checks_passed": full_non_density_pass,
        "matched_density_diagnostic": expected_input_signature(density_path),
        "decision": {
            "45m_supported_as_deliberate_ladder_rung": candidate_rung,
            "45m_legacy_absolute_selector_passed": bool(
                full.get("absolute_selector_passed")
            ),
            "balanced_density_gate_calibrated": False,
            "advance_directly_to_120m": False,
            "reason": (
                "45M rung support requires same-row noninferiority and every "
                "registered non-density quality/integrity check. The legacy "
                "0.60 density floor is still reported but was calibrated on a "
                "different data regime; this round neither replaces nor tunes "
                "it, so no direct 120M advance is authorized."
            ),
            "next_action": (
                "isolate balanced-universe versus training/graph causes of the "
                "density regime before another larger training rung"
                if candidate_rung
                else "bisect the first failed 30M-to-45M matched interval"
            ),
        },
        "ood_is_reported_separately_and_non_gating": True,
    }
    receipt = seal(body)
    path = os.path.join(output, "scale-comparison.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def _draw_points(path: str, points: np.ndarray, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 10))
    axis.scatter(
        points[:, 0],
        points[:, 1],
        s=0.15,
        alpha=0.35,
        linewidths=0,
        rasterized=True,
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(path, format="png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_renders(
    _active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    output = create_fresh_directory(
        job["outputs"][0],
        label="R0069 fixed scale renders",
    )
    _, selector30, _, eligibility30 = load_substrate(
        int8_path=str(job["int8_path"]),
        int8_sha256=str(job["int8_sha256"]),
        scales_path=str(job["scales_path"]),
        scales_sha256=str(job["scales_sha256"]),
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
        or np.any(sample30 >= 30_000_000)
    ):
        raise Round0069Error("R0064 matched render sample changed")
    # Every sampled row was selected from the retained compact universe in
    # R0064. Re-check that none became excluded before reusing it.
    excluded = np.asarray(selector30.excluded_rows)
    if np.intersect1d(sample30, excluded, assume_unique=False).size:
        raise Round0069Error("matched render sample contains excluded rows")

    renders: dict[str, Any] = {}
    for definition in job["matched_maps"]:
        key = str(definition["map_key"])
        coordinates = CoordinateStream(
            str(definition["transform_output"])
        )
        points = coordinates[sample30]
        if (
            not np.isfinite(points).all()
            or np.any(np.std(points, axis=0) <= 1e-8)
        ):
            raise Round0069Error(f"{key} render coordinates collapsed")
        image_path = os.path.join(output, f"{key}.png")
        atomic_build_new_file(
            image_path,
            lambda path, values=points, label=MAP_LABELS[key]: _draw_points(
                path,
                values,
                label,
            ),
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

    _, selector45, _, eligibility45 = load_substrate(
        int8_path=str(job["full_int8_path"]),
        int8_sha256=str(job["full_int8_sha256"]),
        scales_path=str(job["full_scales_path"]),
        scales_sha256=str(job["full_scales_sha256"]),
        eligibility_path=str(job["full_eligibility_path"]),
        eligibility_sha256=str(job["full_eligibility_sha256"]),
        row_count=45_000_000,
    )
    rng = np.random.RandomState(20260727)
    compact45 = np.sort(
        rng.choice(selector45.retained_count, 50_000, replace=False)
    ).astype(np.int64)
    sample45 = selector45.compact_to_global(compact45)
    sample45_path = os.path.join(output, "full-45m-sample-rows.npy")
    atomic_save_new_npy(sample45_path, sample45, immutable=True)
    full_coordinates = CoordinateStream(str(job["full_transform"]))
    full_points = full_coordinates[sample45]
    if (
        not np.isfinite(full_points).all()
        or np.any(np.std(full_points, axis=0) <= 1e-8)
    ):
        raise Round0069Error("full 45M render coordinates collapsed")
    full_key = "r0068-45m-on-45m"
    full_image_path = os.path.join(output, f"{full_key}.png")
    atomic_build_new_file(
        full_image_path,
        lambda path: _draw_points(path, full_points, MAP_LABELS[full_key]),
        immutable=True,
    )
    renders[full_key] = {
        "image": expected_input_signature(full_image_path),
        "axis_std": full_points.std(axis=0).astype(float).tolist(),
        "axis_span": np.ptp(full_points, axis=0).astype(float).tolist(),
        "sample_rows": expected_input_signature(sample45_path),
        "sample_rows_sha256": ordered_array_sha256(sample45),
        "sample_universe": "balanced-45m retained representatives",
        "eligibility": eligibility45["signature"],
    }

    definitions_body = {
        "schema": "scale-map-definitions-v1",
        "round_id": ROUND_ID,
        "maps": [{
            "key": full_key,
            "label": MAP_LABELS[full_key],
            "coordinates": "coordinates-r0068-45m",
            "panel": "panel-r0068-45m/panel.json",
            "render": f"semantic-renders/{full_key}.png",
            "training_round": "0068",
            "panel_schema": PANEL_SCHEMA,
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
        "schema": "round0069-scale-render-v1",
        "round_id": ROUND_ID,
        "matched_eligibility": eligibility30["signature"],
        "full_eligibility": eligibility45["signature"],
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
        label="R0069 immutable map registry snapshot",
    )
    from experiments import map_registry

    registry = map_registry.scan()
    entries = [
        item for item in registry["maps"]
        if item.get("round_id") == ROUND_ID
    ]
    map_entries = [
        item for item in entries if item.get("kind") == "round-map"
    ]
    projection_entries = [
        item for item in entries if item.get("kind") == "projection-map"
    ]
    required_label = MAP_LABELS["r0068-45m-on-45m"]
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
        raise Round0069Error(
            "registry did not discover the 45M map and five projections"
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
    # The mutable discovery file/site are convenience views. The queue-local
    # snapshot above is the reviewable capability and never changes.
    map_registry.REGISTRY_PATH.write_bytes(payload)
    map_registry.publish(registry)
    snapshot = expected_input_signature(snapshot_path)
    current = expected_input_signature(str(map_registry.REGISTRY_PATH))
    if (
        snapshot["sha256"] != current["sha256"]
        or snapshot["bytes"] != current["bytes"]
    ):
        raise Round0069Error("published registry differs from snapshot")
    body = {
        "schema": "round0069-map-registry-publication-v1",
        "round_id": ROUND_ID,
        "immutable_registry_snapshot": snapshot,
        "mutable_registry_after_publish": current,
        "map_ids": sorted(item["map_id"] for item in entries),
        "base_map": required_label,
        "projection_probes": sorted(required_probes),
        "local_site_url": map_registry.SITE_URL,
        "evidence_status_at_queue_time": (
            "round:issued; review state is intentionally refreshed by later "
            "registry scans without changing the immutable snapshot"
        ),
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
        raise RuntimeError("R0069 handler received another queue")
    if job is None:
        raise RuntimeError("R0069 handler requires the exact job")
    _configure_shared()
    handlers = {
        "transform": shared.run_transform,
        "high_d_reference": shared.run_high_d_reference,
        "panel": shared.run_panel,
        "ood": shared.run_ood,
        "density_diagnostic": run_density_diagnostic,
        "comparison": run_comparison,
        "renders": run_renders,
        "registry": run_registry,
    }
    try:
        handler = handlers[str(job["action"])]
    except KeyError as exc:
        raise RuntimeError(
            f"unknown R0069 action {job.get('action')!r}"
        ) from exc
    return handler(active, job)
