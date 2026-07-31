"""Evaluate the seed-43 diverse-Jina replicate under R0108's frozen protocol."""
from __future__ import annotations

import gc
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
    CALIBRATION_SCHEMA as R0108_CALIBRATION_SCHEMA,
    DECISION_SCHEMA as R0108_DECISION_SCHEMA,
    EMBEDDING_PROMPT,
    FAMILY_SIZE_CUTOFF,
    K_DENSITY,
    MAP_KEY as SEED42_MAP_KEY,
    OOD_SCHEMA as R0108_OOD_SCHEMA,
    CORE_SCHEMA as R0108_CORE_SCHEMA,
    POLISH,
    Round0108Error,
    load_reviewed_model,
    map_family_sizes,
    read_sealed,
    seal,
    verify_signature,
)
from experiments import round0108_nodes as seed42_nodes
from experiments.round0085_nodes import density_v2_calibration
from experiments.round0109_nodes import (
    PRODUCTION_CONFIG_SCHEMA,
    SEED,
    TRAIN_RECEIPT_SCHEMA,
)


ROUND_ID = "0110"
MAP_KEY = "r0109-diverse-jina-25m-seed43"
MAP_LABEL = MAP_KEY
CORE_SCHEMA = "round0110-diverse-jina-core-geometry-v1"
OOD_SCHEMA = "round0110-diverse-jina-ood-evaluation-v1"
MATCHED_DENSITY_SCHEMA = "round0110-matched-fineweb-density-v1"
DECISION_SCHEMA = "round0110-diverse-jina-two-seed-decision-v2"
MATCHED_SOURCE_ROWS = 2_000_000
MATCHED_SOURCE_DIMENSION = 768
MATCHED_ANCHORS = 10_000


def _seed43_model(
    *,
    train_output: str,
    graph_manifest_path: str,
    graph_manifest_sha256: str,
) -> dict[str, Any]:
    return load_reviewed_model(
        train_output=train_output,
        graph_manifest_path=graph_manifest_path,
        graph_manifest_sha256=graph_manifest_sha256,
        expected_train_round_id="0109",
        expected_train_receipt_schema=TRAIN_RECEIPT_SCHEMA,
        expected_production_config_schema=PRODUCTION_CONFIG_SCHEMA,
        expected_seed=SEED,
    )


def _seed43_job(job: Mapping[str, Any]) -> dict[str, Any]:
    """Bind R0108's scorer through an explicit, process-local contract."""
    selected = dict(job)
    selected["evaluation_node_contract"] = {
        "round_id": ROUND_ID,
        "map_key": MAP_KEY,
        "map_label": MAP_LABEL,
        "graph_round_id": "0106",
        "graph_k": 15,
        "core_schema": CORE_SCHEMA,
        "ood_schema": OOD_SCHEMA,
        "train_round_id": "0109",
        "train_receipt_schema": TRAIN_RECEIPT_SCHEMA,
        "production_config_schema": PRODUCTION_CONFIG_SCHEMA,
        "seed": SEED,
        "graph_schema": "round0106-jina-diverse-25m-fuzzy-graph-v1",
    }
    return selected


def run_transform(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    return seed42_nodes.run_transform(active, _seed43_job(job))


def run_core(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    return seed42_nodes.run_core_score(active, _seed43_job(job))


def run_ood(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    return seed42_nodes.run_ood(active, _seed43_job(job))


def _exact_signature(
    path: str,
    expected_sha256: str,
    *,
    label: str,
) -> dict[str, Any]:
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise Round0108Error(f"{label} bytes changed")
    return signature


def _matched_density_cell(
    *,
    key: str,
    seed: int,
    bundle: Mapping[str, Any],
    representatives: RepresentativeArrayView,
    anchors: np.ndarray,
    high_radius: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Project and score one atlas model on the exact R0040 universe."""
    from basemap.panel_v2 import _self_knn

    coordinates = np.asarray(
        bundle["model"].transform(
            representatives,
            batch_size=seed42_nodes.TRANSFORM_BATCH_ROWS,
        ),
        dtype=np.float32,
    )
    if (
        coordinates.shape != (len(representatives), 2)
        or not np.isfinite(coordinates).all()
    ):
        raise Round0108Error(
            f"R0110 {key} matched-universe coordinates are malformed"
        )
    config = seed42_nodes._panel_config(anchors=len(anchors))
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
    receipt = {
        "key": key,
        "seed": seed,
        "model": bundle["train"]["model"],
        "train_receipt": bundle["train_signature"],
        "production_config": bundle["config_signature"],
        "coordinates": {
            "rows": len(coordinates),
            "dtype": coordinates.dtype.str,
            "ordered_sha256": ordered_array_sha256(coordinates),
            "axis_standard_deviation": coordinates.std(0).tolist(),
            "finite": True,
        },
        "density_v2": summary,
        "low_dim_exact_search_guard": guard,
    }
    arrays = {
        f"{key}__low_radius": low_radius,
        f"{key}__bootstrap": bootstrap,
        f"{key}__permuted_null": null,
    }
    return receipt, arrays


def run_matched_density(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    """Test floor portability on the calibration universe, not the atlas."""
    output = create_fresh_directory(
        str(job["outputs"][0]),
        label="R0110 matched FineWeb calibration density",
    )
    started = time.monotonic()
    calibration_path = os.path.join(
        str(job["calibration_output"]),
        "jina-density-calibration.json",
    )
    calibration = read_sealed(
        calibration_path,
        label="R0108 Jina density calibration",
        schema=R0108_CALIBRATION_SCHEMA,
    )
    floor = (calibration.get("floor_calibration") or {}).get(
        "registered_floor"
    )
    if floor is None or not np.isfinite(float(floor)):
        raise Round0108Error("R0108 Jina density floor is not registered")
    floor = float(floor)

    census_path = str(job["census_receipt"])
    census_signature = _exact_signature(
        census_path,
        str(job["census_receipt_sha256"]),
        label="R0040 Jina census receipt",
    )
    census = load_jina_census(census_path)
    source_signature = census["receipt"].get("source")
    source_path = verify_signature(
        source_signature,
        label="R0040 Jina calibration source",
    )
    source = np.load(source_path, mmap_mode="r", allow_pickle=False)
    if (
        source.shape != (MATCHED_SOURCE_ROWS, MATCHED_SOURCE_DIMENSION)
        or source.dtype != np.dtype("<f2")
    ):
        raise Round0108Error("R0040 Jina calibration source changed")
    selector = RepresentativeRowSelector(
        census["arrays"]["excluded_rows"],
        row_count=MATCHED_SOURCE_ROWS,
        source=census["signature"],
        policy="R0040 exact nonzero fp16 family; minimum row representative",
    )
    representatives = RepresentativeArrayView(source, selector)

    reference_path = str(job["representative_reference"])
    reference_signature = _exact_signature(
        reference_path,
        str(job["representative_reference_sha256"]),
        label="R0040 representative high-D reference",
    )
    with np.load(reference_path, allow_pickle=False) as archive:
        anchors = np.asarray(archive["anchor_ids"], dtype=np.int64)
        high_radius = np.asarray(archive["r_hd"], dtype=np.float64)
        reference_key = str(archive["key"].item())
    global_rows = selector.compact_to_global(anchors)
    family_sizes = map_family_sizes(
        global_rows,
        census["arrays"]["representative_rows"],
        census["arrays"]["family_counts"],
    )
    eligible = family_sizes < FAMILY_SIZE_CUTOFF
    if (
        anchors.shape != (MATCHED_ANCHORS,)
        or high_radius.shape != anchors.shape
        or not np.all(eligible)
        or reference_key
        != calibration.get("representative_reference_key")
        or ordered_array_sha256(anchors)
        != (calibration.get("anchors") or {}).get("compact_rows_sha256")
        or ordered_array_sha256(global_rows)
        != (calibration.get("anchors") or {}).get("global_rows_sha256")
    ):
        raise Round0108Error(
            "R0110 matched calibration anchor universe changed"
        )

    calibration_arrays_path = verify_signature(
        calibration.get("arrays"),
        label="R0108 Jina calibration arrays",
    )
    with np.load(calibration_arrays_path, allow_pickle=False) as archive:
        for key in ("seed42", "seed43"):
            if not np.array_equal(
                np.asarray(archive[f"{key}__high_radius"]),
                high_radius,
            ):
                raise Round0108Error(
                    "R0108 calibration high-D radii changed"
                )

    graph_manifest_path = str(job["graph_manifest"])
    graph_manifest_sha256 = str(job["graph_manifest_sha256"])
    bundles = {
        "seed42": load_reviewed_model(
            train_output=str(job["seed42_train_output"]),
            graph_manifest_path=graph_manifest_path,
            graph_manifest_sha256=graph_manifest_sha256,
        ),
        "seed43": _seed43_model(
            train_output=str(job["seed43_train_output"]),
            graph_manifest_path=graph_manifest_path,
            graph_manifest_sha256=graph_manifest_sha256,
        ),
    }
    cells: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {
        "anchor_compact_rows": anchors,
        "anchor_global_rows": global_rows,
        "high_radius": high_radius,
        "family_sizes": family_sizes,
    }
    for key, seed in (("seed42", 42), ("seed43", 43)):
        cell, cell_arrays = _matched_density_cell(
            key=key,
            seed=seed,
            bundle=bundles[key],
            representatives=representatives,
            anchors=anchors,
            high_radius=high_radius,
        )
        cell["clears_unchanged_registered_floor"] = (
            float(cell["density_v2"]["correlation"]) >= floor
        )
        cells[key] = cell
        arrays.update(cell_arrays)

    arrays_path = os.path.join(output, "matched-density-arrays.npz")
    atomic_save_new_npz(arrays_path, immutable=True, **arrays)
    both_pass = all(
        bool(cell["clears_unchanged_registered_floor"])
        for cell in cells.values()
    )
    receipt = seal({
        "schema": MATCHED_DENSITY_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "design_timing": {
            "registered_after_reviewed_seed42_native_density_failure": True,
            "registered_before_seed43_evaluation": True,
        },
        "calibration": expected_input_signature(calibration_path),
        "registered_floor": floor,
        "floor_changed_or_tuned": False,
        "census_receipt": census_signature,
        "source": dict(source_signature),
        "source_preprocessing": (
            "exact stored fp16 rows cast per transform batch to fp32; "
            "no L2 renormalization"
        ),
        "representative_reference": reference_signature,
        "reference_key": reference_key,
        "universe": {
            "name": "R0040 exact-family representative FineWeb calibration",
            "rows": len(representatives),
            "anchors": len(anchors),
            "anchor_compact_rows_sha256": ordered_array_sha256(anchors),
            "anchor_global_rows_sha256": ordered_array_sha256(global_rows),
            "high_radius_sha256": ordered_array_sha256(high_radius),
            "family_size_cutoff_exclusive": FAMILY_SIZE_CUTOFF,
            "anchors_after_filter": int(eligible.sum()),
        },
        "cells": cells,
        "checks": {
            "same_universe_as_r0108_floor_calibration": True,
            "same_anchors_and_high_d_radii_as_r0108_calibration": True,
            "unchanged_registered_floor": True,
            "seed42_clears_matched_floor": cells["seed42"][
                "clears_unchanged_registered_floor"
            ],
            "seed43_clears_matched_floor": cells["seed43"][
                "clears_unchanged_registered_floor"
            ],
            "both_seeds_clear_matched_floor": both_pass,
        },
        "calibration_portability_capability_released": both_pass,
        "full_diverse_universe_density_resolved": False,
        "full_diverse_universe_density_claimed": False,
        "role": (
            "tests whether the R0108 FineWeb-calibrated floor is portable "
            "when universe and anchors are held exact; it does not measure "
            "density preservation on the full diverse training universe"
        ),
        "arrays": expected_input_signature(arrays_path),
        "training_performed": False,
        "wall_seconds": time.monotonic() - started,
    })
    receipt_path = os.path.join(output, "matched-density.json")
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    for bundle in bundles.values():
        del bundle["model"]
    del source, representatives
    gc.collect()
    return {**receipt, "receipt": expected_input_signature(receipt_path)}


def _metric(receipt: Mapping[str, Any], *keys: str) -> float:
    value: Any = receipt
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise Round0108Error(
                f"R0110 comparison metric is missing: {'/'.join(keys)}"
            )
        value = value[key]
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise Round0108Error(
            f"R0110 comparison metric is nonnumeric: {'/'.join(keys)}"
        ) from exc


def _native_non_density_core_passed(
    receipt: Mapping[str, Any],
) -> bool:
    checks = (receipt.get("decision") or {}).get("checks")
    required = {
        "coordinates_finite_and_noncollapsed",
        "density_v2_clears_registered_jina_floor",
        "every_language_ffr_at_least_0_40_of_pooled_english",
        "global_ffr_at_least_0_40",
        "global_recall50_strictly_exceeds_recall10",
    }
    if not isinstance(checks, Mapping) or set(checks) != required:
        raise Round0108Error("R0110 core decision checks changed")
    return all(
        bool(value)
        for key, value in checks.items()
        if key != "density_v2_clears_registered_jina_floor"
    )


def run_decision(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    """Release two-seed quality only when both seeds pass the same gates."""
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0110 two-seed decision"
    )
    seed42_decision_path = str(job["seed42_decision"])
    seed42_core_path = str(job["seed42_core"])
    seed42_ood_path = str(job["seed42_ood"])
    seed43_core_path = os.path.join(
        str(job["core_output"]), "core-geometry.json"
    )
    seed43_ood_path = os.path.join(
        str(job["ood_output"]), "ood-evaluation.json"
    )
    matched_density_path = os.path.join(
        str(job["matched_density_output"]), "matched-density.json"
    )
    seed42_decision = read_sealed(
        seed42_decision_path,
        label="R0108 seed-42 atlas decision",
        schema=R0108_DECISION_SCHEMA,
    )
    seed42_core = read_sealed(
        seed42_core_path,
        label="R0108 seed-42 core geometry",
        schema=R0108_CORE_SCHEMA,
    )
    seed42_ood = read_sealed(
        seed42_ood_path,
        label="R0108 seed-42 OOD evaluation",
        schema=R0108_OOD_SCHEMA,
    )
    seed43_core = read_sealed(
        seed43_core_path,
        label="R0110 seed-43 core geometry",
        schema=CORE_SCHEMA,
    )
    seed43_ood = read_sealed(
        seed43_ood_path,
        label="R0110 seed-43 OOD evaluation",
        schema=OOD_SCHEMA,
    )
    matched_density = read_sealed(
        matched_density_path,
        label="R0110 matched FineWeb density",
        schema=MATCHED_DENSITY_SCHEMA,
    )

    if (
        seed42_decision.get("round_id") != "0108"
        or seed42_decision.get("map_key") != SEED42_MAP_KEY
        or seed42_core.get("map_key") != SEED42_MAP_KEY
        or seed42_ood.get("map_key") != SEED42_MAP_KEY
        or seed43_core.get("round_id") != ROUND_ID
        or seed43_core.get("map_key") != MAP_KEY
        or seed43_ood.get("round_id") != ROUND_ID
        or seed43_ood.get("map_key") != MAP_KEY
        or matched_density.get("round_id") != ROUND_ID
    ):
        raise Round0108Error("R0110 seed comparison identity changed")

    seed42_core_passed = bool(
        (seed42_core.get("decision") or {}).get("passed")
    )
    seed42_ood_passed = bool(
        (seed42_ood.get("headline_decision") or {}).get("passed")
    )
    seed42_quality_passed = bool(
        seed42_decision.get("atlas_quality_capability_released")
    )
    if seed42_quality_passed != (
        seed42_core_passed and seed42_ood_passed
    ):
        raise Round0108Error("R0108 seed-42 decision does not close")

    seed43_core_passed = bool(
        (seed43_core.get("decision") or {}).get("passed")
    )
    seed43_ood_passed = bool(
        (seed43_ood.get("headline_decision") or {}).get("passed")
    )
    seed42_non_density_passed = _native_non_density_core_passed(seed42_core)
    seed43_non_density_passed = _native_non_density_core_passed(seed43_core)
    seed42_native_density_passed = bool(
        seed42_core["decision"]["checks"][
            "density_v2_clears_registered_jina_floor"
        ]
    )
    seed43_native_density_passed = bool(
        seed43_core["decision"]["checks"][
            "density_v2_clears_registered_jina_floor"
        ]
    )
    matched_checks = matched_density.get("checks")
    if (
        not isinstance(matched_checks, Mapping)
        or matched_density.get(
            "calibration_portability_capability_released"
        )
        is not bool(matched_checks.get("both_seeds_clear_matched_floor"))
        or matched_density.get("floor_changed_or_tuned") is not False
        or matched_density.get("full_diverse_universe_density_resolved")
        is not False
    ):
        raise Round0108Error("R0110 matched density decision does not close")
    matched_both_passed = bool(
        matched_checks["both_seeds_clear_matched_floor"]
    )
    prompt_identity_closes = all(
        receipt.get("embedding_prompt") == EMBEDDING_PROMPT
        and receipt.get("prompt_applied") is False
        for receipt in (seed42_ood, seed43_ood)
    )
    two_seed_passed = (
        seed42_quality_passed
        and seed43_core_passed
        and seed43_ood_passed
        and prompt_identity_closes
    )
    matched_fineweb_qualified_atlas = (
        seed42_non_density_passed
        and seed43_non_density_passed
        and seed42_ood_passed
        and seed43_ood_passed
        and matched_both_passed
        and prompt_identity_closes
    )

    comparisons = {
        "core_global_ffr": {
            "seed42": _metric(
                seed42_core, "metrics", "global", "ffr"
            ),
            "seed43": _metric(
                seed43_core, "metrics", "global", "ffr"
            ),
        },
        "core_global_recall_at_10": {
            "seed42": _metric(
                seed42_core, "metrics", "global", "recall_at_10"
            ),
            "seed43": _metric(
                seed43_core, "metrics", "global", "recall_at_10"
            ),
        },
        "core_global_recall_at_50_of_high10": {
            "seed42": _metric(
                seed42_core,
                "metrics",
                "global",
                "recall_at_50_of_high10",
            ),
            "seed43": _metric(
                seed43_core,
                "metrics",
                "global",
                "recall_at_50_of_high10",
            ),
        },
        "density_v2": {
            "seed42": _metric(
                seed42_core, "metrics", "density_v2", "correlation"
            ),
            "seed43": _metric(
                seed43_core, "metrics", "density_v2", "correlation"
            ),
        },
        "polish_recall_at_10": {
            "seed42": _metric(
                seed42_ood,
                "language_cells",
                POLISH,
                "probe",
                "recall_at_10",
            ),
            "seed43": _metric(
                seed43_ood,
                "language_cells",
                POLISH,
                "probe",
                "recall_at_10",
            ),
        },
        "polish_recall_at_50_of_high10": {
            "seed42": _metric(
                seed42_ood,
                "language_cells",
                POLISH,
                "probe",
                "recall_at_50_of_high10",
            ),
            "seed43": _metric(
                seed43_ood,
                "language_cells",
                POLISH,
                "probe",
                "recall_at_50_of_high10",
            ),
        },
        "polish_to_in_mix_median_ratio": {
            "seed42": _metric(
                seed42_ood,
                "headline_decision",
                "polish_to_in_mix_median_ratio",
            ),
            "seed43": _metric(
                seed43_ood,
                "headline_decision",
                "polish_to_in_mix_median_ratio",
            ),
        },
    }
    comparisons = {
        name: {
            **values,
            "seed43_minus_seed42": values["seed43"] - values["seed42"],
            "role": "diagnostic-only",
        }
        for name, values in comparisons.items()
    }
    receipt = seal({
        "schema": DECISION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "seed42": {
            "training_round": "0107",
            "seed": 42,
            "decision": expected_input_signature(seed42_decision_path),
            "core": expected_input_signature(seed42_core_path),
            "ood": expected_input_signature(seed42_ood_path),
        },
        "seed43": {
            "training_round": "0109",
            "seed": SEED,
            "core": expected_input_signature(seed43_core_path),
            "ood": expected_input_signature(seed43_ood_path),
        },
        "matched_fineweb_density": expected_input_signature(
            matched_density_path
        ),
        "checks": {
            "seed42_fixed_core_gate_passed": seed42_core_passed,
            "seed42_fixed_polish_ood_gate_passed": seed42_ood_passed,
            "seed42_atlas_quality_passed": seed42_quality_passed,
            "seed43_fixed_core_gate_passed": seed43_core_passed,
            "seed43_fixed_polish_ood_gate_passed": seed43_ood_passed,
            "seed42_native_non_density_core_passed": (
                seed42_non_density_passed
            ),
            "seed43_native_non_density_core_passed": (
                seed43_non_density_passed
            ),
            "both_seeds_clear_unchanged_floor_on_matched_fineweb": (
                matched_both_passed
            ),
            "raw_prompt_identity_closes": prompt_identity_closes,
            "cross_seed_deltas_excluded_from_decision": True,
            "projection_ffr_excluded_from_decision": True,
            "original_frozen_two_seed_rule_unchanged": True,
            "broader_diverse_density_claim_unresolved": (
                not two_seed_passed
            ),
        },
        "comparison_metrics": comparisons,
        "two_seed_quality_capability_released": two_seed_passed,
        "calibration_portability_capability_released": (
            matched_both_passed
        ),
        "matched_fineweb_qualified_atlas_capability_released": (
            matched_fineweb_qualified_atlas
        ),
        "matched_fineweb_qualification_scope": (
            "both seeds pass native diverse-universe non-density and Polish "
            "OOD gates plus the unchanged Jina density floor on the exact "
            "R0040 FineWeb calibration universe"
        ),
        "full_diverse_universe_density_under_original_floor": {
            "seed42_clears_floor": seed42_native_density_passed,
            "seed43_clears_floor": seed43_native_density_passed,
            "both_seeds_clear_floor": (
                seed42_native_density_passed
                and seed43_native_density_passed
            ),
            "status": (
                "passed"
                if (
                    seed42_native_density_passed
                    and seed43_native_density_passed
                )
                else "failed"
            ),
            "overridden_by_matched_fineweb_cell": False,
        },
        "broader_diverse_density_preservation_claimed": two_seed_passed,
        "outcome": (
            "two-seed-quality-accepted"
            if two_seed_passed
            else (
                "matched-fineweb-qualified-atlas"
                if matched_fineweb_qualified_atlas
                else "two-seed-quality-not-released"
            )
        ),
        "embedding_prompt": EMBEDDING_PROMPT,
        "prompt_applied": False,
        "production_document_prompt_transfer_resolved": False,
        "production_readiness_claimed": False,
        "training_performed": False,
    })
    path = os.path.join(output, "two-seed-decision.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID or job is None:
        raise Round0108Error("R0110 handler requires its exact round/job")
    handlers = {
        "transform_seed43": run_transform,
        "score_seed43_core": run_core,
        "score_seed43_ood": run_ood,
        "score_matched_calibration_density": run_matched_density,
        "decide_seed_stability": run_decision,
    }
    try:
        handler = handlers[str(job.get("action"))]
    except KeyError as exc:
        raise Round0108Error(
            f"unknown R0110 action {job.get('action')!r}"
        ) from exc
    return handler(active, job)
