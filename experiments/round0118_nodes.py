"""Evaluate the R0111 seed-44 map under the frozen diverse-Jina protocol.

R0118 is an evaluation-only third-seed replicate.  Native diverse-universe
geometry and held-out Polish OOD use the exact R0108 selectors, scorers, and
thresholds.  The matched-FineWeb density cell extends R0110 on the exact same
representative universe, anchors, high-dimensional radii, and registered
floor; it is a separately scoped qualification and can never repair a native
density failure.
"""
from __future__ import annotations

import gc
import os
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import numpy as np

from basemap.artifact_identity import (
    expected_input_signature,
    ordered_array_sha256,
)
from basemap.output_safety import (
    atomic_save_new_npy,
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
    EMBEDDING_PROMPT,
    FAMILY_SIZE_CUTOFF,
    POLISH,
    Round0108Error,
    load_reviewed_model,
    map_family_sizes,
    read_sealed,
    seal,
    verify_signature,
)
from experiments import round0108_nodes as frozen_nodes
from experiments import round0110_nodes as prior_nodes
from experiments.round0108_nodes import _refresh_registry_best_effort
from experiments.round0111_nodes import (
    PRODUCTION_CONFIG_SCHEMA,
    SEED,
    TRAIN_RECEIPT_SCHEMA,
)


ROUND_ID = "0118"
MAP_KEY = "r0111-diverse-jina-25m-seed44"
MAP_LABEL = MAP_KEY
CORE_SCHEMA = "round0118-diverse-jina-core-geometry-v1"
OOD_SCHEMA = "round0118-diverse-jina-ood-evaluation-v1"
MATCHED_DENSITY_SCHEMA = "round0118-matched-fineweb-density-v1"
DECISION_SCHEMA = "round0118-diverse-jina-three-seed-decision-v1"
MAP_DEFINITION_SCHEMA = "round0118-map-definition-v1"
REGISTRY_PUBLICATION_SCHEMA = "round0118-map-registry-publication-v1"

MATCHED_SOURCE_ROWS = prior_nodes.MATCHED_SOURCE_ROWS
MATCHED_SOURCE_DIMENSION = prior_nodes.MATCHED_SOURCE_DIMENSION
MATCHED_ANCHORS = prior_nodes.MATCHED_ANCHORS

_FROZEN_BINDINGS = (
    "ROUND_ID",
    "MAP_KEY",
    "MAP_LABEL",
    "CORE_SCHEMA",
    "OOD_SCHEMA",
    "load_reviewed_model",
)


def _seed44_model(
    *,
    train_output: str,
    graph_manifest_path: str,
    graph_manifest_sha256: str,
) -> dict[str, Any]:
    """Authenticate and load the exact accepted R0111 seed-44 bundle."""
    return load_reviewed_model(
        train_output=train_output,
        graph_manifest_path=graph_manifest_path,
        graph_manifest_sha256=graph_manifest_sha256,
        expected_train_round_id="0111",
        expected_train_receipt_schema=TRAIN_RECEIPT_SCHEMA,
        expected_production_config_schema=PRODUCTION_CONFIG_SCHEMA,
        expected_seed=SEED,
    )


@contextmanager
def _seed44_contract() -> Iterator[None]:
    """Temporarily bind the frozen R0108 scorer to the seed-44 identity.

    Production nodes execute one action per process, but restoring the module
    globals also makes in-process smokes and future composition deterministic.
    No selector, metric, threshold, or prompt semantic is replaced.
    """
    previous = {
        name: getattr(frozen_nodes, name) for name in _FROZEN_BINDINGS
    }
    replacements = {
        "ROUND_ID": ROUND_ID,
        "MAP_KEY": MAP_KEY,
        "MAP_LABEL": MAP_LABEL,
        "CORE_SCHEMA": CORE_SCHEMA,
        "OOD_SCHEMA": OOD_SCHEMA,
        "load_reviewed_model": _seed44_model,
    }
    try:
        for name, value in replacements.items():
            setattr(frozen_nodes, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(frozen_nodes, name, value)


def run_transform(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    with _seed44_contract():
        return frozen_nodes.run_transform(active, job)


def run_core(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    with _seed44_contract():
        return frozen_nodes.run_core_score(active, job)


def run_ood(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    with _seed44_contract():
        return frozen_nodes.run_ood(active, job)


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


def run_matched_density(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    """Extend R0110's exact matched-FineWeb density cell to seed 44."""
    output = create_fresh_directory(
        str(job["outputs"][0]),
        label="R0118 seed-44 matched FineWeb density",
    )
    started = time.monotonic()
    prior_path = str(job["r0110_matched_density"])
    prior = read_sealed(
        prior_path,
        label="R0110 matched FineWeb density",
        schema=prior_nodes.MATCHED_DENSITY_SCHEMA,
    )
    calibration_path = os.path.join(
        str(job["calibration_output"]),
        "jina-density-calibration.json",
    )
    calibration = read_sealed(
        calibration_path,
        label="R0108 Jina density calibration",
        schema=prior_nodes.R0108_CALIBRATION_SCHEMA,
    )
    floor = (calibration.get("floor_calibration") or {}).get(
        "registered_floor"
    )
    if floor is None or not np.isfinite(float(floor)):
        raise Round0108Error("R0108 Jina density floor is not registered")
    floor = float(floor)
    if (
        prior.get("calibration") != expected_input_signature(calibration_path)
        or float(prior.get("registered_floor", np.nan)) != floor
        or prior.get("floor_changed_or_tuned") is not False
        or prior.get("full_diverse_universe_density_resolved") is not False
        or prior.get("full_diverse_universe_density_claimed") is not False
    ):
        raise Round0108Error("R0110 matched-density scope or floor changed")

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

    prior_arrays_path = verify_signature(
        prior.get("arrays"),
        label="R0110 matched FineWeb arrays",
    )
    with np.load(prior_arrays_path, allow_pickle=False) as archive:
        prior_anchors = np.asarray(
            archive["anchor_compact_rows"], dtype=np.int64
        )
        prior_global_rows = np.asarray(
            archive["anchor_global_rows"], dtype=np.int64
        )
        prior_high_radius = np.asarray(
            archive["high_radius"], dtype=np.float64
        )
    prior_universe = prior.get("universe") or {}
    if (
        anchors.shape != (MATCHED_ANCHORS,)
        or high_radius.shape != anchors.shape
        or not np.all(eligible)
        or reference_key != calibration.get("representative_reference_key")
        or not np.array_equal(anchors, prior_anchors)
        or not np.array_equal(global_rows, prior_global_rows)
        or not np.array_equal(high_radius, prior_high_radius)
        or prior_universe.get("rows") != len(representatives)
        or prior_universe.get("anchors") != len(anchors)
        or prior_universe.get("anchor_compact_rows_sha256")
        != ordered_array_sha256(anchors)
        or prior_universe.get("anchor_global_rows_sha256")
        != ordered_array_sha256(global_rows)
        or prior_universe.get("high_radius_sha256")
        != ordered_array_sha256(high_radius)
        or prior.get("source") != dict(source_signature)
        or prior.get("representative_reference") != reference_signature
        or prior.get("census_receipt") != census_signature
    ):
        raise Round0108Error(
            "R0118 matched calibration universe differs from R0110"
        )

    bundle = _seed44_model(
        train_output=str(job["seed44_train_output"]),
        graph_manifest_path=str(job["graph_manifest"]),
        graph_manifest_sha256=str(job["graph_manifest_sha256"]),
    )
    cell, arrays = prior_nodes._matched_density_cell(
        key="seed44",
        seed=SEED,
        bundle=bundle,
        representatives=representatives,
        anchors=anchors,
        high_radius=high_radius,
    )
    seed44_passed = (
        float(cell["density_v2"]["correlation"]) >= floor
    )
    cell["clears_unchanged_registered_floor"] = seed44_passed
    prior_checks = prior.get("checks")
    if (
        not isinstance(prior_checks, Mapping)
        or prior_checks.get("same_universe_as_r0108_floor_calibration")
        is not True
        or prior_checks.get(
            "same_anchors_and_high_d_radii_as_r0108_calibration"
        )
        is not True
        or prior_checks.get("unchanged_registered_floor") is not True
        or prior.get("calibration_portability_capability_released")
        is not bool(
            prior_checks.get("both_seeds_clear_matched_floor")
        )
        or prior.get("source_preprocessing")
        != (
            "exact stored fp16 rows cast per transform batch to fp32; "
            "no L2 renormalization"
        )
    ):
        raise Round0108Error("R0110 matched-density checks are missing")
    prior_two_passed = bool(
        prior_checks.get("both_seeds_clear_matched_floor")
    )
    three_passed = prior_two_passed and seed44_passed

    arrays.update({
        "anchor_compact_rows": anchors,
        "anchor_global_rows": global_rows,
        "high_radius": high_radius,
        "family_sizes": family_sizes,
    })
    arrays_path = os.path.join(output, "matched-density-arrays.npz")
    atomic_save_new_npz(arrays_path, immutable=True, **arrays)
    receipt = seal({
        "schema": MATCHED_DENSITY_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "design_timing": {
            "extends_preregistered_r0110_matched_universe": True,
            "registered_before_seed44_evaluation": True,
        },
        "r0110_matched_density": expected_input_signature(prior_path),
        "calibration": expected_input_signature(calibration_path),
        "registered_floor": floor,
        "floor_changed_or_tuned": False,
        "census_receipt": census_signature,
        "source": dict(source_signature),
        "source_preprocessing": prior.get("source_preprocessing"),
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
        "cells": {"seed44": cell},
        "checks": {
            "exact_r0110_universe_reused": True,
            "exact_r0110_anchors_and_high_d_radii_reused": True,
            "unchanged_registered_floor": True,
            "r0110_seed42_and_seed43_clear_matched_floor": (
                prior_two_passed
            ),
            "seed44_clears_matched_floor": seed44_passed,
            "all_three_seeds_clear_matched_floor": three_passed,
        },
        "calibration_portability_capability_released": three_passed,
        "full_diverse_universe_density_resolved": False,
        "full_diverse_universe_density_claimed": False,
        "role": (
            "extends the R0110 exact matched-FineWeb calibration test to "
            "seed 44; it cannot measure or repair density preservation on "
            "the native diverse training universe"
        ),
        "arrays": expected_input_signature(arrays_path),
        "training_performed": False,
        "wall_seconds": time.monotonic() - started,
    })
    receipt_path = os.path.join(output, "matched-density.json")
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    del bundle["model"], source, representatives
    gc.collect()
    return {**receipt, "receipt": expected_input_signature(receipt_path)}


def _metric(receipt: Mapping[str, Any], *keys: str) -> float:
    return prior_nodes._metric(receipt, *keys)


def _three_seed_diagnostics(
    prior: Mapping[str, Any],
    seed44_core: Mapping[str, Any],
    seed44_ood: Mapping[str, Any],
) -> dict[str, Any]:
    seed44_paths = {
        "core_global_ffr": ("core", "metrics", "global", "ffr"),
        "core_global_recall_at_10": (
            "core", "metrics", "global", "recall_at_10"
        ),
        "core_global_recall_at_50_of_high10": (
            "core", "metrics", "global", "recall_at_50_of_high10"
        ),
        "density_v2": (
            "core", "metrics", "density_v2", "correlation"
        ),
        "polish_recall_at_10": (
            "ood", "language_cells", POLISH, "probe", "recall_at_10"
        ),
        "polish_recall_at_50_of_high10": (
            "ood",
            "language_cells",
            POLISH,
            "probe",
            "recall_at_50_of_high10",
        ),
        "polish_to_in_mix_median_ratio": (
            "ood",
            "headline_decision",
            "polish_to_in_mix_median_ratio",
        ),
    }
    prior_metrics = prior.get("comparison_metrics")
    if not isinstance(prior_metrics, Mapping):
        raise Round0108Error("R0110 comparison metrics are missing")
    result: dict[str, Any] = {}
    receipts = {"core": seed44_core, "ood": seed44_ood}
    for name, path in seed44_paths.items():
        old = prior_metrics.get(name)
        if (
            not isinstance(old, Mapping)
            or old.get("role") != "diagnostic-only"
        ):
            raise Round0108Error(
                f"R0110 diagnostic metric changed: {name}"
            )
        values = {
            "seed42": float(old["seed42"]),
            "seed43": float(old["seed43"]),
            "seed44": _metric(receipts[path[0]], *path[1:]),
        }
        result[name] = {
            **values,
            "minimum": min(values.values()),
            "maximum": max(values.values()),
            "range": max(values.values()) - min(values.values()),
            "role": "diagnostic-only",
        }
    return result


def three_seed_decision(
    *,
    prior: Mapping[str, Any],
    seed44_core: Mapping[str, Any],
    seed44_ood: Mapping[str, Any],
    matched: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen native rule and separately scoped matched rule."""
    if (
        prior.get("schema") != prior_nodes.DECISION_SCHEMA
        or prior.get("round_id") != "0110"
        or prior.get("production_readiness_claimed") is not False
        or prior.get("production_document_prompt_transfer_resolved")
        is not False
        or prior.get("broader_diverse_density_preservation_claimed")
        is not bool(prior.get("two_seed_quality_capability_released"))
    ):
        raise Round0108Error("R0110 two-seed decision semantics changed")
    prior_checks = prior.get("checks")
    full_prior = prior.get(
        "full_diverse_universe_density_under_original_floor"
    )
    prior_two_quality = bool(
        prior.get("two_seed_quality_capability_released")
    )
    expected_prior_two_quality = all(
        bool(prior_checks.get(name))
        for name in (
            "seed42_atlas_quality_passed",
            "seed43_fixed_core_gate_passed",
            "seed43_fixed_polish_ood_gate_passed",
            "raw_prompt_identity_closes",
        )
    ) if isinstance(prior_checks, Mapping) else False
    expected_prior_matched = all(
        bool(prior_checks.get(name))
        for name in (
            "seed42_native_non_density_core_passed",
            "seed43_native_non_density_core_passed",
            "seed42_fixed_polish_ood_gate_passed",
            "seed43_fixed_polish_ood_gate_passed",
            "both_seeds_clear_unchanged_floor_on_matched_fineweb",
            "raw_prompt_identity_closes",
        )
    ) if isinstance(prior_checks, Mapping) else False
    if (
        not isinstance(prior_checks, Mapping)
        or prior_checks.get("original_frozen_two_seed_rule_unchanged")
        is not True
        or prior_checks.get("cross_seed_deltas_excluded_from_decision")
        is not True
        or prior_checks.get("projection_ffr_excluded_from_decision")
        is not True
        or prior_checks.get("broader_diverse_density_claim_unresolved")
        is not (not prior_two_quality)
        or prior_two_quality is not expected_prior_two_quality
        or prior.get(
            "matched_fineweb_qualified_atlas_capability_released"
        )
        is not expected_prior_matched
        or not isinstance(full_prior, Mapping)
        or full_prior.get("overridden_by_matched_fineweb_cell") is not False
    ):
        raise Round0108Error("R0110 native-density non-rescue rule changed")
    if (
        seed44_core.get("schema") != CORE_SCHEMA
        or seed44_core.get("round_id") != ROUND_ID
        or seed44_core.get("map_key") != MAP_KEY
        or seed44_ood.get("schema") != OOD_SCHEMA
        or seed44_ood.get("round_id") != ROUND_ID
        or seed44_ood.get("map_key") != MAP_KEY
        or matched.get("schema") != MATCHED_DENSITY_SCHEMA
        or matched.get("round_id") != ROUND_ID
    ):
        raise Round0108Error("R0118 seed-44 evaluation identity changed")

    seed44_core_passed = bool(
        (seed44_core.get("decision") or {}).get("passed")
    )
    seed44_ood_passed = bool(
        (seed44_ood.get("headline_decision") or {}).get("passed")
    )
    seed44_non_density_passed = (
        prior_nodes._native_non_density_core_passed(seed44_core)
    )
    seed44_native_density_passed = bool(
        seed44_core["decision"]["checks"][
            "density_v2_clears_registered_jina_floor"
        ]
    )
    prompt_identity_closes = (
        seed44_ood.get("embedding_prompt") == EMBEDDING_PROMPT
        and seed44_ood.get("prompt_applied") is False
    )
    matched_checks = matched.get("checks")
    if (
        not isinstance(matched_checks, Mapping)
        or matched.get("floor_changed_or_tuned") is not False
        or matched.get("full_diverse_universe_density_resolved") is not False
        or matched.get("full_diverse_universe_density_claimed") is not False
        or matched.get("calibration_portability_capability_released")
        is not bool(
            matched_checks.get("all_three_seeds_clear_matched_floor")
        )
    ):
        raise Round0108Error("R0118 matched-density decision does not close")

    seed44_quality = (
        seed44_core_passed and seed44_ood_passed and prompt_identity_closes
    )
    three_seed_quality = (
        prior_two_quality and seed44_quality
    )
    three_seed_matched = (
        bool(
            prior.get(
                "matched_fineweb_qualified_atlas_capability_released"
            )
        )
        and seed44_non_density_passed
        and seed44_ood_passed
        and bool(
            matched_checks.get("all_three_seeds_clear_matched_floor")
        )
        and prompt_identity_closes
    )
    seed42_density = bool(full_prior.get("seed42_clears_floor"))
    seed43_density = bool(full_prior.get("seed43_clears_floor"))
    all_native_density = (
        seed42_density and seed43_density and seed44_native_density_passed
    )
    diagnostics = _three_seed_diagnostics(
        prior, seed44_core, seed44_ood
    )
    return {
        "checks": {
            "r0110_original_frozen_two_seed_rule_preserved": True,
            "r0110_two_seed_native_quality_passed": bool(
                prior_two_quality
            ),
            "seed44_fixed_native_core_gate_passed": seed44_core_passed,
            "seed44_fixed_polish_ood_gate_passed": seed44_ood_passed,
            "seed44_native_non_density_core_passed": (
                seed44_non_density_passed
            ),
            "all_three_clear_unchanged_floor_on_matched_fineweb": bool(
                matched_checks.get(
                    "all_three_seeds_clear_matched_floor"
                )
            ),
            "raw_prompt_identity_closes": prompt_identity_closes,
            "cross_seed_diagnostics_excluded_from_decision": True,
            "projection_ffr_excluded_from_decision": True,
            "matched_fineweb_cannot_override_native_density": True,
        },
        "three_seed_diagnostics": diagnostics,
        "seed44_atlas_quality_capability_released": seed44_quality,
        "three_seed_quality_capability_released": three_seed_quality,
        "calibration_portability_capability_released": bool(
            matched_checks.get("all_three_seeds_clear_matched_floor")
        ),
        "matched_fineweb_qualified_atlas_capability_released": (
            three_seed_matched
        ),
        "matched_fineweb_qualification_scope": (
            "all three seeds pass native diverse-universe non-density and "
            "Polish OOD gates plus the unchanged Jina density floor on the "
            "exact R0040 FineWeb calibration universe"
        ),
        "full_diverse_universe_density_under_original_floor": {
            "seed42_clears_floor": seed42_density,
            "seed43_clears_floor": seed43_density,
            "seed44_clears_floor": seed44_native_density_passed,
            "all_three_seeds_clear_floor": all_native_density,
            "status": "passed" if all_native_density else "failed",
            "overridden_by_matched_fineweb_cell": False,
        },
        "broader_diverse_density_preservation_claimed": (
            three_seed_quality
        ),
        "outcome": (
            "three-seed-native-quality-accepted"
            if three_seed_quality
            else (
                "three-seed-matched-fineweb-qualified-atlas"
                if three_seed_matched
                else "three-seed-quality-not-released"
            )
        ),
        "embedding_prompt": EMBEDDING_PROMPT,
        "prompt_applied": False,
        "production_document_prompt_transfer_resolved": False,
        "production_readiness_claimed": False,
        "training_performed": False,
    }


def run_decision(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0118 three-seed decision"
    )
    prior_path = str(job["r0110_decision"])
    core_path = os.path.join(
        str(job["core_output"]), "core-geometry.json"
    )
    ood_path = os.path.join(
        str(job["ood_output"]), "ood-evaluation.json"
    )
    matched_path = os.path.join(
        str(job["matched_density_output"]), "matched-density.json"
    )
    prior = read_sealed(
        prior_path,
        label="R0110 two-seed decision",
        schema=prior_nodes.DECISION_SCHEMA,
    )
    core = read_sealed(
        core_path, label="R0118 seed-44 core geometry", schema=CORE_SCHEMA
    )
    ood = read_sealed(
        ood_path, label="R0118 seed-44 OOD", schema=OOD_SCHEMA
    )
    matched = read_sealed(
        matched_path,
        label="R0118 seed-44 matched FineWeb density",
        schema=MATCHED_DENSITY_SCHEMA,
    )
    body = three_seed_decision(
        prior=prior,
        seed44_core=core,
        seed44_ood=ood,
        matched=matched,
    )
    receipt = seal({
        "schema": DECISION_SCHEMA,
        "round_id": ROUND_ID,
        "map_key": MAP_KEY,
        "release_sha": active["manifest"]["release_sha"],
        "r0110_two_seed_decision": expected_input_signature(prior_path),
        "seed44": {
            "training_round": "0111",
            "seed": SEED,
            "core": expected_input_signature(core_path),
            "ood": expected_input_signature(ood_path),
        },
        "matched_fineweb_density": expected_input_signature(matched_path),
        **body,
    })
    receipt_path = os.path.join(output, "three-seed-decision.json")
    atomic_write_new_json(receipt_path, receipt, immutable=True)

    core_arrays = verify_signature(
        core.get("arrays"), label="R0118 core arrays"
    )
    with np.load(core_arrays, allow_pickle=False) as archive:
        sample_ids = np.asarray(
            archive["compact_anchor_rows"], dtype=np.int64
        )
    render_root = create_fresh_directory(
        str(job["render_output"]), label="R0118 registry support"
    )
    sample_path = os.path.join(render_root, "sample-semantic-ids.npy")
    atomic_save_new_npy(sample_path, sample_ids, immutable=True)
    transform_path = os.path.join(
        str(job["transform_output"]), "actual-transform.json"
    )
    definitions = seal({
        "schema": MAP_DEFINITION_SCHEMA,
        "round_id": ROUND_ID,
        "map_key": MAP_KEY,
        "map_label": MAP_LABEL,
        "training_round": "0111",
        "embedding_prompt": EMBEDDING_PROMPT,
        "prompt_applied": False,
        "production_document_prompt_transfer_resolved": False,
        "production_ready": False,
        "coordinates": expected_input_signature(transform_path),
        "core_panel": expected_input_signature(core_path),
        "decision": expected_input_signature(receipt_path),
        "sample_ids": expected_input_signature(sample_path),
    })
    definitions_path = os.path.join(render_root, "map-definition.json")
    atomic_write_new_json(definitions_path, definitions, immutable=True)

    registry_path = os.path.join(output, "registry-publication.json")
    _refresh_registry_best_effort(
        receipt_path=registry_path,
        map_definition_path=definitions_path,
        decision_path=receipt_path,
        round_id=ROUND_ID,
        map_key=MAP_KEY,
        publication_schema=REGISTRY_PUBLICATION_SCHEMA,
    )
    return {
        **receipt,
        "receipt": expected_input_signature(receipt_path),
        "registry": expected_input_signature(registry_path),
    }


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID or job is None:
        raise Round0108Error("R0118 handler requires its exact round/job")
    handlers = {
        "transform_seed44": run_transform,
        "score_seed44_core": run_core,
        "score_seed44_ood": run_ood,
        "score_seed44_matched_fineweb_density": run_matched_density,
        "decide_three_seed_stability_and_publish_registry": run_decision,
    }
    try:
        handler = handlers[str(job.get("action"))]
    except KeyError as exc:
        raise Round0108Error(
            f"unknown R0118 action {job.get('action')!r}"
        ) from exc
    return handler(active, job)
