"""Fresh-process node for the R0019 duplicate-anchor leverage diagnostic."""
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
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0036_pipeline import CoordinateStream
from basemap.round0064_evaluation import seal, validate_seal


ROUND_ID = "0074"
ROWS = 30_000_000
ANCHORS = 10_000
K_DENSITY = 15
LOG_EPSILON = 1e-12
EXPECTED_FAMILY_ANCHORS = 124
EXPECTED_EXTREME_ANCHORS = 20
EXPECTED_UNIQUE_CANONICAL_ANCHORS = 9_983
EXPECTED_MAX_FAMILY_SIZE = 30_088


class Round0074Error(RuntimeError):
    """The registered duplicate-anchor diagnostic contract was violated."""


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0074Error(f"{path} is not a JSON object")
    return value


def _panel_config():
    from basemap.panel_v2 import PanelV2Config
    from basemap.round0036_pipeline import panel_config_identity

    return PanelV2Config(**{
        key: tuple(value) if key == "k_clust" else value
        for key, value in panel_config_identity().items()
        if key != "formula_version"
    })


def map_anchor_families(
    anchors: np.ndarray,
    eligibility: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Map every row anchor to its exact-family representative and size."""
    anchors = np.asarray(anchors, dtype=np.int64)
    duplicate_rows = np.asarray(
        eligibility["duplicate_excluded_rows"],
        dtype=np.int64,
    )
    duplicate_representatives = np.asarray(
        eligibility["duplicate_representative_rows"],
        dtype=np.int64,
    )
    representatives = np.asarray(
        eligibility["representative_rows"],
        dtype=np.int64,
    )
    counts = np.asarray(eligibility["family_counts"], dtype=np.int64)

    canonical = anchors.copy()
    positions = np.searchsorted(duplicate_rows, anchors)
    duplicate = positions < len(duplicate_rows)
    valid = np.flatnonzero(duplicate)
    duplicate[valid] &= duplicate_rows[positions[valid]] == anchors[valid]
    canonical[duplicate] = duplicate_representatives[positions[duplicate]]

    family_count = np.ones(anchors.shape, dtype=np.int64)
    positions = np.searchsorted(representatives, canonical)
    family = positions < len(representatives)
    valid = np.flatnonzero(family)
    family[valid] &= representatives[positions[valid]] == canonical[valid]
    family_count[family] = counts[positions[family]]
    return canonical, family_count


def _correlation(high_log: np.ndarray, low_log: np.ndarray) -> float | None:
    if (
        len(high_log) < 2
        or float(np.var(high_log)) == 0.0
        or float(np.var(low_log)) == 0.0
    ):
        return None
    value = float(np.corrcoef(high_log, low_log)[0, 1])
    return round(value, 4) if math.isfinite(value) else None


def _one_group(
    high_log: np.ndarray,
    low_log: np.ndarray,
    mask: np.ndarray,
    cross_product: np.ndarray,
    high_ss: np.ndarray,
    low_ss: np.ndarray,
) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    covariance_total = float(cross_product.sum())
    high_total = float(high_ss.sum())
    low_total = float(low_ss.sum())

    def fraction(numerator: float, denominator: float) -> float | None:
        if denominator == 0.0:
            return None
        return round(numerator / denominator, 6)

    return {
        "anchors": int(selected.sum()),
        "correlation": _correlation(
            high_log[selected],
            low_log[selected],
        ),
        "mean_log_high_d_radius": (
            round(float(high_log[selected].mean()), 6)
            if selected.any()
            else None
        ),
        "mean_log_low_d_radius": (
            round(float(low_log[selected].mean()), 6)
            if selected.any()
            else None
        ),
        "zero_high_d_radius": int(
            np.count_nonzero(high_log[selected] == math.log(LOG_EPSILON))
        ),
        "zero_low_d_radius": int(
            np.count_nonzero(low_log[selected] == math.log(LOG_EPSILON))
        ),
        "full_sample_covariance_numerator_fraction": fraction(
            float(cross_product[selected].sum()),
            covariance_total,
        ),
        "full_sample_high_variance_fraction": fraction(
            float(high_ss[selected].sum()),
            high_total,
        ),
        "full_sample_low_variance_fraction": fraction(
            float(low_ss[selected].sum()),
            low_total,
        ),
    }


def density_leverage_summary(
    high_radius: np.ndarray,
    low_radius: np.ndarray,
    family_count: np.ndarray,
    canonical_rows: np.ndarray,
    corpus_labels: np.ndarray,
) -> dict[str, Any]:
    """Report Pearson density and exact-family leverage without tuning a floor."""
    high_radius = np.asarray(high_radius, dtype=np.float64)
    low_radius = np.asarray(low_radius, dtype=np.float64)
    family_count = np.asarray(family_count, dtype=np.int64)
    canonical_rows = np.asarray(canonical_rows, dtype=np.int64)
    corpus_labels = np.asarray(corpus_labels)
    if (
        high_radius.shape != (ANCHORS,)
        or low_radius.shape != (ANCHORS,)
        or family_count.shape != (ANCHORS,)
        or canonical_rows.shape != (ANCHORS,)
        or corpus_labels.shape != (ANCHORS,)
        or np.any(high_radius < 0)
        or np.any(low_radius < 0)
        or not np.isfinite(high_radius).all()
        or not np.isfinite(low_radius).all()
        or np.any(family_count < 1)
    ):
        raise Round0074Error("density leverage inputs are malformed")

    high_log = np.log(high_radius + LOG_EPSILON)
    low_log = np.log(low_radius + LOG_EPSILON)
    high_centered = high_log - high_log.mean()
    low_centered = low_log - low_log.mean()
    cross_product = high_centered * low_centered
    high_ss = high_centered * high_centered
    low_ss = low_centered * low_centered

    first_unique = np.zeros(ANCHORS, dtype=bool)
    first_unique[np.unique(canonical_rows, return_index=True)[1]] = True
    masks = {
        "all_anchors": np.ones(ANCHORS, dtype=bool),
        "singleton_family_only": family_count == 1,
        "exclude_family_ge_16": family_count < 16,
        "one_anchor_per_canonical_family": first_unique,
        "family_gt_1": family_count > 1,
        "family_ge_16": family_count >= 16,
    }
    family_size_strata = {
        "size_1": family_count == 1,
        "size_2_3": (family_count >= 2) & (family_count <= 3),
        "size_4_15": (family_count >= 4) & (family_count <= 15),
        "size_16_255": (family_count >= 16) & (family_count <= 255),
        "size_256_9999": (family_count >= 256) & (family_count <= 9_999),
        "size_ge_10000": family_count >= 10_000,
    }

    def summarize(mask: np.ndarray) -> dict[str, Any]:
        return _one_group(
            high_log,
            low_log,
            mask,
            cross_product,
            high_ss,
            low_ss,
        )

    groups = {name: summarize(mask) for name, mask in masks.items()}
    all_correlation = groups["all_anchors"]["correlation"]
    without_extreme = groups["exclude_family_ge_16"]["correlation"]
    singleton = groups["singleton_family_only"]["correlation"]
    return {
        "all": groups["all_anchors"],
        "anchor_population_sensitivity": {
            "exclude_family_ge_16": groups["exclude_family_ge_16"],
            "singleton_family_only": groups["singleton_family_only"],
            "one_anchor_per_canonical_family": groups[
                "one_anchor_per_canonical_family"
            ],
            "all_minus_exclude_family_ge_16": (
                round(float(all_correlation) - float(without_extreme), 4)
                if all_correlation is not None and without_extreme is not None
                else None
            ),
            "all_minus_singleton_family_only": (
                round(float(all_correlation) - float(singleton), 4)
                if all_correlation is not None and singleton is not None
                else None
            ),
        },
        "duplicate_group_attribution": {
            "family_gt_1": groups["family_gt_1"],
            "family_ge_16": groups["family_ge_16"],
        },
        "by_family_size": {
            name: summarize(mask)
            for name, mask in family_size_strata.items()
        },
        "by_corpus": {
            name: summarize(corpus_labels == name)
            for name in ("fineweb", "redpajama", "pile")
        },
    }


def classify_anchor_leverage(
    *,
    replay_exact: bool,
    legacy: Mapping[str, Any],
    modern: Mapping[str, Any],
    representative_anchor_cells: Mapping[str, float],
) -> dict[str, Any]:
    """Apply the preregistered leverage rule to both fixed-model comparisons."""
    material_delta = 0.20
    dominant_fraction = 0.50

    def evidence(cell: Mapping[str, Any], representative: float) -> dict[str, Any]:
        full = float(cell["all"]["correlation"])
        without = float(
            cell["anchor_population_sensitivity"][
                "exclude_family_ge_16"
            ]["correlation"]
        )
        covariance_fraction = float(
            cell["duplicate_group_attribution"]["family_ge_16"][
                "full_sample_covariance_numerator_fraction"
            ]
        )
        return {
            "full_minus_exclude_family_ge_16": round(full - without, 4),
            "full_minus_r0070_representative_anchor_cell": round(
                full - float(representative),
                4,
            ),
            "family_ge_16_covariance_numerator_fraction": covariance_fraction,
            "material_within_sample_drop": full - without >= material_delta,
            "material_cross_anchor_population_drop": (
                full - float(representative) >= material_delta
            ),
            "extreme_family_covariance_dominant": (
                covariance_fraction >= dominant_fraction
            ),
        }

    legacy_evidence = evidence(
        legacy,
        float(representative_anchor_cells["legacy_original"]),
    )
    modern_evidence = evidence(
        modern,
        float(representative_anchor_cells["modern_original"]),
    )
    supported = (
        replay_exact
        and all(
            item[key]
            for item in (legacy_evidence, modern_evidence)
            for key in (
                "material_within_sample_drop",
                "material_cross_anchor_population_drop",
                "extreme_family_covariance_dominant",
            )
        )
    )
    if supported:
        classification = "duplicate-heavy-anchor-leverage-supported"
    elif replay_exact:
        classification = "duplicate-heavy-anchor-leverage-inconclusive"
    else:
        classification = "legacy-density-replay-failed"
    return {
        "classification": classification,
        "legacy": legacy_evidence,
        "modern": modern_evidence,
        "registered_bands": {
            "material_correlation_delta": material_delta,
            "dominant_covariance_numerator_fraction": dominant_fraction,
        },
        "calibrates_density_threshold": False,
        "authorizes_larger_training_rung": False,
    }


def run_anchor_leverage(
    _active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    from basemap.panel_v2 import _self_knn
    from experiments import run_round0014_node as legacy

    output = create_fresh_directory(
        job["outputs"][0],
        label="R0074 duplicate-anchor leverage",
    )
    started = time.monotonic()
    reference_signature = expected_input_signature(
        str(job["legacy_reference_path"])
    )
    if reference_signature["sha256"] != job["legacy_reference_sha256"]:
        raise Round0074Error("R0019 reference bytes changed")
    with np.load(
        str(job["legacy_reference_path"]),
        allow_pickle=False,
    ) as reference:
        anchors = np.asarray(reference["anchor_ids"], dtype=np.int64)
        high_radius = np.asarray(reference["r_hd"], dtype=np.float64)
    if (
        anchors.shape != (ANCHORS,)
        or high_radius.shape != (ANCHORS,)
        or ordered_array_sha256(anchors) != job["legacy_anchor_rows_sha256"]
        or ordered_array_sha256(high_radius) != job["legacy_high_radius_sha256"]
        or np.any(anchors < 0)
        or np.any(anchors >= ROWS)
    ):
        raise Round0074Error("R0019 anchor/reference identity changed")

    eligibility = load_int8_eligibility(
        str(job["eligibility_path"]),
        expected_sha256=str(job["eligibility_sha256"]),
        row_count=ROWS,
    )
    canonical_rows, family_count = map_anchor_families(
        anchors,
        eligibility,
    )
    if (
        int(np.count_nonzero(family_count > 1)) != EXPECTED_FAMILY_ANCHORS
        or int(np.count_nonzero(family_count >= 16))
        != EXPECTED_EXTREME_ANCHORS
        or len(np.unique(canonical_rows))
        != EXPECTED_UNIQUE_CANONICAL_ANCHORS
        or int(family_count.max()) != EXPECTED_MAX_FAMILY_SIZE
        or int(np.count_nonzero(high_radius == 0))
        != EXPECTED_EXTREME_ANCHORS
        or not np.all(high_radius[family_count >= 16] == 0)
    ):
        raise Round0074Error("registered R0019 anchor-family census changed")

    legacy.configure_round0019()
    legacy_coordinates = legacy.StreamedCoordinateArray(
        str(job["legacy_coordinates"])
    )
    modern_coordinates = CoordinateStream(str(job["modern_coordinates"]))
    if (
        legacy_coordinates.record["actual_transform"]["model_signature"][
            "sha256"
        ] != job["legacy_model_sha256"]
        or modern_coordinates.receipt["model"]["sha256"]
        != job["modern_model_sha256"]
    ):
        raise Round0074Error("coordinate model identity changed")

    labels = np.asarray(
        ["fineweb", "redpajama", "pile"],
        dtype="<U10",
    )[anchors // 10_000_000]
    config = _panel_config()
    cells: dict[str, Any] = {}
    low_radii: dict[str, np.ndarray] = {}
    for name, coordinates in (
        ("legacy_r0019", legacy_coordinates),
        ("modern_r0061", modern_coordinates),
    ):
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
        cells[name] = {
            "candidate_universe": "original-fp16-all-30m-rows",
            "anchor_population": "r0019-uniform-all-row-seed123",
            "density": density_leverage_summary(
                high_radius,
                low_radius,
                family_count,
                canonical_rows,
                labels,
            ),
            "low_dim_guard": guard,
        }
        low_radii[name] = low_radius

    legacy_panel = _read_json(str(job["legacy_panel_path"]))
    registered_legacy = float(legacy_panel["panel"]["density"])
    replayed_legacy = float(
        cells["legacy_r0019"]["density"]["all"]["correlation"]
    )
    replay_exact = replayed_legacy == registered_legacy
    if not replay_exact:
        raise Round0074Error(
            "R0019 density did not exactly replay before interpretation"
        )

    r0070 = _read_json(str(job["r0070_factorial_path"]))
    validate_seal(r0070, label="R0070 density factorial")
    representative_anchor_cells = {
        name: float(r0070["cells"][name]["density"]["correlation"])
        for name in ("legacy_original", "modern_original")
    }
    if representative_anchor_cells != {
        "legacy_original": 0.0748,
        "modern_original": 0.1003,
    }:
        raise Round0074Error("reviewed R0070 anchor-population bridge changed")

    interpretation = classify_anchor_leverage(
        replay_exact=replay_exact,
        legacy=cells["legacy_r0019"]["density"],
        modern=cells["modern_r0061"]["density"],
        representative_anchor_cells=representative_anchor_cells,
    )
    archive_path = os.path.join(output, "anchor-leverage-radii.npz")

    def write_archive(path: str) -> None:
        with open(path, "wb") as handle:
            np.savez(
                handle,
                anchor_global_rows=anchors,
                canonical_family_rows=canonical_rows,
                family_counts=family_count,
                corpus_labels=labels,
                high_d_radius_original=high_radius,
                low_d_radius_legacy_r0019=low_radii["legacy_r0019"],
                low_d_radius_modern_r0061=low_radii["modern_r0061"],
            )

    atomic_build_new_file(archive_path, write_archive, immutable=True)
    body = {
        "schema": "round0074-duplicate-anchor-leverage-v1",
        "round_id": ROUND_ID,
        "design": (
            "hold original-fp16 30M candidate universe and exact R0019 "
            "high-D reference fixed; replay the R0019 and R0061 models on the "
            "same all-row anchor population, then attribute Pearson covariance "
            "to exact-family multiplicity"
        ),
        "anchor_identity": {
            "rows_sha256": ordered_array_sha256(anchors),
            "high_d_radius_sha256": ordered_array_sha256(high_radius),
            "population": "r0019-uniform-all-row-seed123",
            "anchors": ANCHORS,
            "anchors_in_exact_families": int(
                np.count_nonzero(family_count > 1)
            ),
            "anchors_in_families_ge_16": int(
                np.count_nonzero(family_count >= 16)
            ),
            "unique_canonical_family_rows": int(
                len(np.unique(canonical_rows))
            ),
            "maximum_family_size": int(family_count.max()),
            "zero_high_d_radii": int(np.count_nonzero(high_radius == 0)),
        },
        "eligibility": eligibility["signature"],
        "legacy_registered_density": registered_legacy,
        "legacy_density_exactly_replayed": replay_exact,
        "r0070_representative_anchor_original_universe_cells": (
            representative_anchor_cells
        ),
        "cells": cells,
        "interpretation": interpretation,
        "radii": expected_input_signature(archive_path),
        "scientific_contract": {
            "training_performed": False,
            "candidate_universe_changed": False,
            "threshold_tuned": False,
            "graph_or_sampler_claim": False,
            "authorizes_larger_training_rung": False,
        },
        "training_performed": False,
        "wall_seconds": time.monotonic() - started,
    }
    receipt = seal(body)
    path = os.path.join(output, "duplicate-anchor-leverage.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise RuntimeError("R0074 handler requires the exact round")
    if job is None or job.get("action") != "anchor_leverage":
        raise RuntimeError(
            f"unknown R0074 action {(job or {}).get('action')!r}"
        )
    return run_anchor_leverage(active, job)
