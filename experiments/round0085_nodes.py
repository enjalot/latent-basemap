"""Duplicate-controlled density calibration over the reviewed scale ladder."""
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
from basemap.round0040_program import (
    RepresentativeArrayView,
    RepresentativeRowSelector,
)
from basemap.round0064_evaluation import (
    expected_retained_rows_for_scale,
    seal,
    validate_seal,
)
from experiments.round0074_nodes import map_anchor_families


ROUND_ID = "0085"
ANCHOR_COUNT = 10_000
K_DENSITY = 15
FAMILY_SIZE_CUTOFF = 16
LOG_EPSILON = 1e-12
BOOTSTRAP_DRAWS = 1_000
BOOTSTRAP_SEED = 85_001
NULL_DRAWS = 1_000
NULL_SEED = 85_002
MATCHED_CELL_KEYS = (
    "r0061_30m_on_30m",
    "r0068_45m_on_30m",
    "r0063_60m_on_30m",
    "r0075_90m_on_30m",
)
R0074_FILTERED_EXPECTED = {
    "legacy_r0019": 0.0985,
    "modern_r0061": 0.1125,
}


class Round0085Error(RuntimeError):
    """The registered density-v2 calibration contract was violated."""


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0085Error(f"{path} is not a JSON object")
    return value


def _panel_config():
    from basemap.panel_v2 import PanelV2Config
    from basemap.round0036_pipeline import panel_config_identity

    return PanelV2Config(**{
        key: tuple(value) if key == "k_clust" else value
        for key, value in panel_config_identity().items()
        if key != "formula_version"
    })


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if (
        left.ndim != 1
        or right.shape != left.shape
        or len(left) < 3
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise Round0085Error("Pearson inputs are malformed")
    left = left - left.mean()
    right = right - right.mean()
    denominator = math.sqrt(
        float(np.dot(left, left)) * float(np.dot(right, right))
    )
    if not denominator > 0.0 or not math.isfinite(denominator):
        raise Round0085Error("Pearson input variance collapsed")
    value = float(np.dot(left, right) / denominator)
    if not math.isfinite(value):
        raise Round0085Error("Pearson correlation is nonfinite")
    return value


def density_v2_calibration(
    high_radius: np.ndarray,
    low_radius: np.ndarray,
    *,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    null_draws: int = NULL_DRAWS,
    null_seed: int = NULL_SEED,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Score one fixed anchor population and quantify sampling/null variation."""
    high_radius = np.asarray(high_radius, dtype=np.float64)
    low_radius = np.asarray(low_radius, dtype=np.float64)
    if (
        high_radius.ndim != 1
        or low_radius.shape != high_radius.shape
        or len(high_radius) < 100
        or np.any(high_radius < 0)
        or np.any(low_radius < 0)
        or not np.isfinite(high_radius).all()
        or not np.isfinite(low_radius).all()
        or bootstrap_draws < 10
        or null_draws < 10
    ):
        raise Round0085Error("density-v2 radius inputs are malformed")
    high_log = np.log(high_radius + LOG_EPSILON)
    low_log = np.log(low_radius + LOG_EPSILON)
    point = _pearson(high_log, low_log)

    bootstrap_rng = np.random.RandomState(bootstrap_seed)
    bootstrap = np.empty(bootstrap_draws, dtype=np.float64)
    for draw in range(bootstrap_draws):
        sample = bootstrap_rng.randint(0, len(high_log), size=len(high_log))
        bootstrap[draw] = _pearson(high_log[sample], low_log[sample])

    null_rng = np.random.RandomState(null_seed)
    null = np.empty(null_draws, dtype=np.float64)
    for draw in range(null_draws):
        null[draw] = _pearson(high_log, low_log[null_rng.permutation(len(low_log))])

    summary = {
        "correlation": point,
        "anchors": int(len(high_radius)),
        "bootstrap": {
            "draws": int(bootstrap_draws),
            "seed": int(bootstrap_seed),
            "standard_deviation": float(bootstrap.std(ddof=1)),
            "central_99_percent": [
                float(np.quantile(bootstrap, 0.005)),
                float(np.quantile(bootstrap, 0.995)),
            ],
        },
        "permuted_radius_null": {
            "draws": int(null_draws),
            "seed": int(null_seed),
            "mean": float(null.mean()),
            "standard_deviation": float(null.std(ddof=1)),
            "absolute_99_9_percentile": float(
                np.quantile(np.abs(null), 0.999)
            ),
        },
    }
    return summary, bootstrap, null


def registered_floor(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the preregistered matched-universe floor and null guard."""
    if set(MATCHED_CELL_KEYS) - set(cells):
        raise Round0085Error("matched density-v2 cells are incomplete")
    point_values = [
        float(cells[key]["density_v2"]["correlation"])
        for key in MATCHED_CELL_KEYS
    ]
    bootstrap_sd = [
        float(
            cells[key]["density_v2"]["bootstrap"]["standard_deviation"]
        )
        for key in MATCHED_CELL_KEYS
    ]
    null_bounds = [
        float(
            cells[key]["density_v2"]["permuted_radius_null"][
                "absolute_99_9_percentile"
            ]
        )
        for key in MATCHED_CELL_KEYS
    ]
    minimum_point = min(point_values)
    maximum_sd = max(bootstrap_sd)
    proposed = minimum_point - 3.0 * maximum_sd
    maximum_null = max(null_bounds)
    finite = all(
        math.isfinite(value)
        for value in (*point_values, *bootstrap_sd, *null_bounds, proposed)
    )
    valid = finite and proposed > 0.0 and proposed > maximum_null
    return {
        "rule": (
            "minimum matched-universe ladder density_v2 minus three times "
            "the maximum matched-cell bootstrap standard deviation"
        ),
        "minimum_matched_density_v2": minimum_point,
        "maximum_matched_bootstrap_standard_deviation": maximum_sd,
        "proposed_floor": proposed,
        "maximum_matched_absolute_null_99_9_percentile": maximum_null,
        "positive": finite and proposed > 0.0,
        "separated_from_permuted_radius_null": finite and proposed > maximum_null,
        "gating_floor_registered": valid,
        "registered_floor": proposed if valid else None,
        "status": (
            "registered"
            if valid
            else "diagnostic-only; positive/null-separation guard failed"
        ),
    }


def replay_r0074(
    *,
    receipt_path: str,
    receipt_sha256: str,
    radii_path: str,
    radii_sha256: str,
) -> dict[str, Any]:
    """Prove scorer continuity against both reviewed R0074 filtered cells."""
    receipt_signature = expected_input_signature(receipt_path)
    radii_signature = expected_input_signature(radii_path)
    if (
        receipt_signature["sha256"] != receipt_sha256
        or radii_signature["sha256"] != radii_sha256
    ):
        raise Round0085Error("R0074 replay artifact bytes changed")
    receipt = _read_json(receipt_path)
    validate_seal(receipt, label="R0074 duplicate-anchor leverage")
    if (
        receipt.get("schema") != "round0074-duplicate-anchor-leverage-v1"
        or receipt.get("radii") != radii_signature
    ):
        raise Round0085Error("R0074 replay receipt identity changed")
    with np.load(radii_path, allow_pickle=False) as archive:
        required = {
            "family_counts",
            "high_d_radius_original",
            "low_d_radius_legacy_r0019",
            "low_d_radius_modern_r0061",
        }
        if not required.issubset(archive.files):
            raise Round0085Error("R0074 radii archive members changed")
        family_counts = np.asarray(archive["family_counts"], dtype=np.int64)
        high_radius = np.asarray(
            archive["high_d_radius_original"], dtype=np.float64
        )
        low = {
            "legacy_r0019": np.asarray(
                archive["low_d_radius_legacy_r0019"], dtype=np.float64
            ),
            "modern_r0061": np.asarray(
                archive["low_d_radius_modern_r0061"], dtype=np.float64
            ),
        }
    eligible = family_counts < FAMILY_SIZE_CUTOFF
    if (
        family_counts.shape != (ANCHOR_COUNT,)
        or high_radius.shape != (ANCHOR_COUNT,)
        or int(eligible.sum()) != ANCHOR_COUNT - 20
    ):
        raise Round0085Error("R0074 filtered anchor population changed")
    replayed: dict[str, float] = {}
    for key, low_radius in low.items():
        value = _pearson(
            np.log(high_radius[eligible] + LOG_EPSILON),
            np.log(low_radius[eligible] + LOG_EPSILON),
        )
        replayed[key] = round(value, 4)
        registered = float(
            receipt["cells"][key]["density"][
                "anchor_population_sensitivity"
            ]["exclude_family_ge_16"]["correlation"]
        )
        if (
            replayed[key] != R0074_FILTERED_EXPECTED[key]
            or replayed[key] != registered
        ):
            raise Round0085Error(
                f"R0074 {key} filtered density did not exactly replay"
            )
    return {
        "receipt": receipt_signature,
        "radii": radii_signature,
        "family_size_cutoff_exclusive": FAMILY_SIZE_CUTOFF,
        "anchors_before_filter": ANCHOR_COUNT,
        "anchors_after_filter": int(eligible.sum()),
        "replayed_correlations": replayed,
        "exact": True,
    }


def _load_universe(spec: Mapping[str, Any]) -> dict[str, Any]:
    row_count = int(spec["row_count"])
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
    if selector.retained_count != expected_retained_rows_for_scale(row_count):
        raise Round0085Error("registered representative count changed")

    reference_signature = expected_input_signature(str(spec["reference_path"]))
    receipt_signature = expected_input_signature(
        str(spec["reference_receipt_path"])
    )
    anchors_signature = expected_input_signature(str(spec["anchor_rows_path"]))
    if (
        reference_signature["sha256"] != spec["reference_sha256"]
        or receipt_signature["sha256"] != spec["reference_receipt_sha256"]
        or anchors_signature["sha256"] != spec["anchor_rows_sha256"]
    ):
        raise Round0085Error("high-D reference bytes changed")
    receipt = _read_json(str(spec["reference_receipt_path"]))
    validate_seal(receipt, label=f"{spec['label']} high-D reference")
    if (
        receipt.get("reference") != reference_signature
        or receipt.get("anchor_substrate_rows") != anchors_signature
        or receipt.get("eligibility") != eligibility["signature"]
        or (receipt.get("selector") or {}).get("representative_count")
        != selector.retained_count
    ):
        raise Round0085Error("high-D reference receipt binding changed")
    with np.load(str(spec["reference_path"]), allow_pickle=False) as archive:
        anchors = np.asarray(archive["anchor_ids"], dtype=np.int64)
        high_radius = np.asarray(archive["r_hd"], dtype=np.float64)
    global_rows = np.load(
        str(spec["anchor_rows_path"]),
        mmap_mode="r",
        allow_pickle=False,
    )
    if (
        anchors.shape != (ANCHOR_COUNT,)
        or high_radius.shape != (ANCHOR_COUNT,)
        or global_rows.shape != (ANCHOR_COUNT,)
        or np.any(anchors < 0)
        or np.any(anchors >= selector.retained_count)
        or not np.array_equal(
            np.asarray(global_rows, dtype=np.int64),
            selector.compact_to_global(anchors),
        )
        or receipt.get("anchor_compact_rows_sha256")
        != ordered_array_sha256(anchors)
        or receipt.get("anchor_substrate_rows_sha256")
        != ordered_array_sha256(np.asarray(global_rows, dtype=np.int64))
        or np.any(high_radius < 0)
        or not np.isfinite(high_radius).all()
    ):
        raise Round0085Error("high-D reference anchor geometry changed")
    canonical, family_counts = map_anchor_families(
        np.asarray(global_rows, dtype=np.int64),
        eligibility,
    )
    if not np.array_equal(canonical, np.asarray(global_rows, dtype=np.int64)):
        raise Round0085Error("reference anchors are not retained representatives")
    eligible = family_counts < FAMILY_SIZE_CUTOFF
    if int(eligible.sum()) < 9_000:
        raise Round0085Error("too many reference anchors fail duplicate control")
    return {
        "label": str(spec["label"]),
        "selector": selector,
        "anchors": anchors,
        "high_radius": high_radius,
        "family_counts": family_counts,
        "eligible": eligible,
        "identity": {
            "row_count": row_count,
            "retained_rows": selector.retained_count,
            "eligibility": eligibility["signature"],
            "reference": reference_signature,
            "reference_receipt": receipt_signature,
            "anchor_substrate_rows": anchors_signature,
            "anchors_sha256": ordered_array_sha256(anchors),
            "anchors_before_filter": ANCHOR_COUNT,
            "anchors_after_family_lt_16_filter": int(eligible.sum()),
            "maximum_anchor_family_size": int(family_counts.max()),
        },
    }


def run_density_v2(
    _active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    from basemap.panel_v2 import _self_knn

    output = create_fresh_directory(
        job["outputs"][0],
        label="R0085 density-v2 calibration",
    )
    started = time.monotonic()
    replay = replay_r0074(
        receipt_path=str(job["r0074_receipt_path"]),
        receipt_sha256=str(job["r0074_receipt_sha256"]),
        radii_path=str(job["r0074_radii_path"]),
        radii_sha256=str(job["r0074_radii_sha256"]),
    )
    universes = {
        str(spec["label"]): _load_universe(spec)
        for spec in job["universes"]
    }
    config = _panel_config()
    cells: dict[str, Any] = {}
    archives: dict[str, np.ndarray] = {}
    for spec in job["cells"]:
        key = str(spec["key"])
        universe = universes[str(spec["universe"])]
        full_coordinates = CoordinateStream(
            str(spec["coordinates_path"]),
            expected_receipt_sha256=str(spec["coordinate_receipt_sha256"]),
        )
        receipt = full_coordinates.receipt
        if (
            len(full_coordinates) != universe["selector"].row_count
            or receipt.get("map_key") != spec["map_key"]
            or (receipt.get("model") or {}).get("sha256")
            != spec["model_sha256"]
        ):
            raise Round0085Error(f"{key} coordinate/model identity changed")
        coordinates = RepresentativeArrayView(
            full_coordinates,
            universe["selector"],
        )
        anchors = universe["anchors"]
        eligible = universe["eligible"]
        _, distances, guard = _self_knn(
            coordinates,
            anchors[eligible],
            K_DENSITY,
            config,
            hi_dim=False,
            want_dist=True,
            exact=True,
        )
        low_radius = np.asarray(distances.mean(1), dtype=np.float64)
        high_radius = universe["high_radius"][eligible]
        summary, bootstrap, null = density_v2_calibration(
            high_radius,
            low_radius,
        )
        cells[key] = {
            "map_key": spec["map_key"],
            "model_sha256": spec["model_sha256"],
            "coordinate_receipt": expected_input_signature(
                os.path.join(
                    str(spec["coordinates_path"]),
                    "actual-transform.json",
                )
            ),
            "universe": str(spec["universe"]),
            "candidate_population": (
                "all retained representatives in ascending compact order"
            ),
            "anchor_population": (
                "reviewed high-D reference anchors whose original exact "
                "family size is <16"
            ),
            "k": K_DENSITY,
            "log_epsilon": LOG_EPSILON,
            "density_v2": summary,
            "low_dim_exact_search_guard": guard,
        }
        archives[f"{key}__high_radius"] = high_radius
        archives[f"{key}__low_radius"] = low_radius
        archives[f"{key}__bootstrap"] = bootstrap
        archives[f"{key}__permuted_null"] = null

    if set(cells) != {str(spec["key"]) for spec in job["cells"]}:
        raise Round0085Error("density-v2 cell accounting changed")
    floor = registered_floor(cells)
    archive_path = os.path.join(output, "density-v2-calibration-arrays.npz")

    def write_archive(path: str) -> None:
        with open(path, "wb") as handle:
            np.savez(handle, **archives)

    atomic_build_new_file(archive_path, write_archive, immutable=True)
    body = {
        "schema": "round0085-density-v2-calibration-v1",
        "round_id": ROUND_ID,
        "design": (
            "reuse each reviewed representative-universe high-D reference; "
            "exclude anchors whose original exact-family size is >=16; "
            "compute exact low-D mean-k15 radii; quantify fixed-anchor "
            "sampling uncertainty by deterministic bootstrap and reject a "
            "floor that does not clear a deterministic permuted-radius null"
        ),
        "r0074_scorer_continuity": replay,
        "universes": {
            key: value["identity"] for key, value in universes.items()
        },
        "cells": cells,
        "floor_calibration": floor,
        "arrays": expected_input_signature(archive_path),
        "scientific_contract": {
            "training_performed": False,
            "coordinate_transform_recomputed": False,
            "high_d_reference_recomputed": False,
            "legacy_density_floor_reactivated": False,
            "density_v2_gating_authorized": bool(
                floor["gating_floor_registered"]
            ),
            "threshold_tuned_after_observing_cells": False,
        },
        "training_performed": False,
        "wall_seconds": time.monotonic() - started,
    }
    receipt = seal(body)
    receipt_path = os.path.join(output, "density-v2-calibration.json")
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(receipt_path)}


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise RuntimeError("R0085 handler requires the exact round")
    if job is None or job.get("action") != "density_v2":
        raise RuntimeError(
            f"unknown R0085 action {(job or {}).get('action')!r}"
        )
    return run_density_v2(active, job)
