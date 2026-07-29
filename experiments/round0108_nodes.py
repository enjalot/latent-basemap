"""Execute the preregistered diverse-Jina atlas evaluation."""
from __future__ import annotations

import gc
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
    atomic_save_new_npz,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0036_pipeline import (
    COORDINATE_SCHEMA,
    TRANSFORM_SCHEMA,
    CoordinateStream,
    seal as coordinate_seal,
)
from basemap.round0040_program import (
    RepresentativeArrayView,
    RepresentativeRowSelector,
    load_jina_census,
)
from basemap.round0105_search import GROUPS, K, RETAINED_ROWS
from basemap.round0106_graph import PARTS, SHARD_ROWS
from basemap.round0108_evaluation import (
    ANCHORS_PER_GROUP,
    CALIBRATION_BOOTSTRAP_DRAWS,
    CALIBRATION_BOOTSTRAP_SEED,
    CALIBRATION_NULL_DRAWS,
    CALIBRATION_NULL_SEED,
    CALIBRATION_SCHEMA,
    CORE_SCHEMA,
    CROSS_ATLAS_CONTROL_SEED,
    DECISION_SCHEMA,
    DIMENSION,
    FAMILY_SIZE_CUTOFF,
    FRACTION,
    HELDOUT_CORPUS_ROWS,
    HELDOUT_QUERY_ROWS,
    IN_MIX_LANGUAGES,
    K_DENSITY,
    K_HIT,
    K_LOW_MAX,
    MAP_KEY,
    MAP_LABEL,
    OOD_SCHEMA,
    PANEL_ANCHORS,
    PANEL_SEED,
    POLISH,
    ROUND_ID,
    TRANSFORM_BATCH_ROWS,
    TRANSFORM_CHUNK_ROWS,
    CompactInt8DequantizedArray,
    Round0108Error,
    core_geometry_decision,
    exact_cosine_topk,
    exact_split_duplicate_diagnostics,
    headline_ood_decision,
    identity_for_rows,
    jina_density_floor,
    load_reviewed_model,
    map_family_sizes,
    normalize_rows,
    pearson_log_radius,
    projection_metrics,
    recall_from_neighbors,
    read_sealed,
    seal,
    validate_seal,
    verify_signature,
)
from experiments.round0085_nodes import density_v2_calibration


def _panel_config(*, anchors: int):
    from basemap.panel_v2 import PanelV2Config

    return PanelV2Config(
        frac=FRACTION,
        k_clust=(),
        k_density=K_DENSITY,
        k_hit=K_HIT,
        n_anchors=anchors,
        anchor_seed=PANEL_SEED,
        corpus_chunk=500_000,
        overselect=8,
        block_elems=500_000_000,
        rerank_byte_cap=2_000_000_000,
        rerank_scratch=3.0,
        peak_byte_cap=26_000_000_000,
    )


def _signature(path: str, expected_sha256: str, *, label: str) -> dict[str, Any]:
    value = expected_input_signature(path)
    if value["sha256"] != expected_sha256:
        raise Round0108Error(f"{label} bytes changed")
    return value


def run_calibration(
    _active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    """Calibrate the Jina density floor before touching the R0107 map."""
    from basemap.panel_v2 import _self_knn

    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0108 Jina density calibration"
    )
    started = time.monotonic()
    census = load_jina_census(str(job["census_receipt"]))
    census_signature = _signature(
        str(job["census_receipt"]),
        str(job["census_receipt_sha256"]),
        label="R0040 Jina census receipt",
    )
    arrays = census["arrays"]
    selector = RepresentativeRowSelector(
        arrays["excluded_rows"],
        row_count=2_000_000,
        source=census["signature"],
        policy=(
            "R0040 exact nonzero fp16 family; minimum row representative"
        ),
    )
    reference_path = str(job["representative_reference"])
    reference_signature = _signature(
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
        arrays["representative_rows"],
        arrays["family_counts"],
    )
    eligible = family_sizes < FAMILY_SIZE_CUTOFF
    if (
        anchors.shape != (10_000,)
        or high_radius.shape != anchors.shape
        or int(eligible.sum()) < 9_000
        or np.any(anchors < 0)
        or np.any(anchors >= selector.retained_count)
    ):
        raise Round0108Error("R0040 Jina calibration anchors changed")

    config = _panel_config(anchors=int(eligible.sum()))
    cells: dict[str, Any] = {}
    archive_values: dict[str, np.ndarray] = {}
    for cell in job["cells"]:
        key = str(cell["key"])
        coordinate_path = str(cell["coordinates"])
        coordinate_signature = _signature(
            coordinate_path,
            str(cell["coordinates_sha256"]),
            label=f"{key} accepted coordinates",
        )
        coordinates = np.load(
            coordinate_path, mmap_mode="r", allow_pickle=False
        )
        if coordinates.shape != (2_000_000, 2) or coordinates.dtype != np.float32:
            raise Round0108Error(f"{key} coordinate geometry changed")
        representatives = RepresentativeArrayView(coordinates, selector)
        _, distances, guard = _self_knn(
            representatives,
            anchors[eligible],
            K_DENSITY,
            config,
            hi_dim=False,
            want_dist=True,
            exact=True,
        )
        low_radius = np.asarray(distances.mean(1), dtype=np.float64)
        summary, bootstrap, null = density_v2_calibration(
            high_radius[eligible],
            low_radius,
            bootstrap_draws=CALIBRATION_BOOTSTRAP_DRAWS,
            bootstrap_seed=CALIBRATION_BOOTSTRAP_SEED,
            null_draws=CALIBRATION_NULL_DRAWS,
            null_seed=CALIBRATION_NULL_SEED,
        )
        cells[key] = {
            "map": str(cell["map"]),
            "seed": int(cell["seed"]),
            "coordinates": coordinate_signature,
            "candidate_population": (
                "R0040 retained exact-family representatives"
            ),
            "anchor_population": (
                "R0040 representative reference anchors with original "
                "exact-family size <16"
            ),
            "density_v2": summary,
            "low_dim_exact_search_guard": guard,
        }
        archive_values[f"{key}__high_radius"] = high_radius[eligible]
        archive_values[f"{key}__low_radius"] = low_radius
        archive_values[f"{key}__bootstrap"] = bootstrap
        archive_values[f"{key}__permuted_null"] = null
    floor = jina_density_floor(cells)
    arrays_path = os.path.join(output, "jina-density-calibration-arrays.npz")
    atomic_save_new_npz(arrays_path, immutable=True, **archive_values)
    receipt = seal({
        "schema": CALIBRATION_SCHEMA,
        "round_id": ROUND_ID,
        "ordering": "completed-before-any-R0107-treatment-score",
        "scorer": (
            "R0085 density_v2: Pearson(log exact high-D mean-k15 radius, "
            "log exact low-D mean-k15 radius)"
        ),
        "census_receipt": census_signature,
        "census": census["signature"],
        "representative_reference": reference_signature,
        "representative_reference_key": reference_key,
        "anchors": {
            "before_family_filter": len(anchors),
            "after_family_lt_16_filter": int(eligible.sum()),
            "compact_rows_sha256": ordered_array_sha256(anchors),
            "global_rows_sha256": ordered_array_sha256(global_rows),
            "family_sizes_sha256": ordered_array_sha256(family_sizes),
        },
        "cells": cells,
        "floor_calibration": floor,
        "arrays": expected_input_signature(arrays_path),
        "threshold_tuned_after_treatment": False,
        "training_performed": False,
        "wall_seconds": time.monotonic() - started,
    })
    receipt_path = os.path.join(output, "jina-density-calibration.json")
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(receipt_path)}


class _SliceView:
    def __init__(self, source: Any, start: int, stop: int):
        self.source = source
        self.start = int(start)
        self.stop = int(stop)
        self.shape = (stop - start, source.shape[1])
        self.dtype = source.dtype

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key: Any) -> np.ndarray:
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return self.source[
                self.start + start : self.start + stop : step
            ]
        rows = np.asarray(key, dtype=np.int64)
        return self.source[rows + self.start]


def run_transform(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    """Project all retained rows without materialising the 25M input."""
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0108 retained coordinate stream"
    )
    started = time.monotonic()
    bundle = load_reviewed_model(
        train_output=str(job["train_output"]),
        graph_manifest_path=str(job["graph_manifest"]),
        graph_manifest_sha256=str(job["graph_manifest_sha256"]),
    )
    source = CompactInt8DequantizedArray(bundle["mapping"])
    members: list[dict[str, Any]] = []
    for index, start in enumerate(
        range(0, RETAINED_ROWS, TRANSFORM_CHUNK_ROWS)
    ):
        stop = min(start + TRANSFORM_CHUNK_ROWS, RETAINED_ROWS)
        root = create_fresh_directory(
            os.path.join(output, f"chunk-{index:05d}"),
            label="R0108 coordinate chunk",
        )
        coordinates = np.asarray(
            bundle["model"].transform(
                _SliceView(source, start, stop),
                batch_size=TRANSFORM_BATCH_ROWS,
            ),
            dtype=np.float32,
        )
        if (
            coordinates.shape != (stop - start, 2)
            or not np.isfinite(coordinates).all()
        ):
            raise Round0108Error("R0108 transform emitted invalid coordinates")
        path = os.path.join(root, "coordinates.npy")
        atomic_save_new_npy(path, coordinates, immutable=True)
        signature = expected_input_signature(path)
        members.append({
            "chunk_index": index,
            "global_row_start": start,
            "global_row_stop": stop,
            "bytes": signature["bytes"],
            "sha256": signature["sha256"],
        })
        print(
            f"R0108 transform {stop:,}/{RETAINED_ROWS:,} retained rows",
            flush=True,
        )
        del coordinates
    body = {
        "schema": TRANSFORM_SCHEMA,
        "round_id": ROUND_ID,
        "map_key": MAP_KEY,
        "model": bundle["train"]["model"],
        "train_receipt": bundle["train_signature"],
        "production_config": bundle["config_signature"],
        "production_config_sha256": bundle["train"][
            "production_config_sha256"
        ],
        "graph_manifest": bundle["graph_signature"],
        "compact_mapping": bundle["graph"]["compact_mapping"],
        "substrate": source.substrate["signature"],
        "scientific_universe": "R0106 retained compact representatives",
        "input_preprocessing": (
            "signed-int8 times exact fp16 row scale to device fp32; "
            "no L2 renormalization before model"
        ),
        "row_accounting": {
            "all_rows": RETAINED_ROWS,
            "retained_representatives": RETAINED_ROWS,
            "original_rows": 25_000_000,
            "excluded_exact_duplicate_or_invalid_rows": (
                25_000_000 - RETAINED_ROWS
            ),
        },
        "coordinate_stream": {
            "schema": COORDINATE_SCHEMA,
            "row_count": RETAINED_ROWS,
            "dimension": 2,
            "dtype": "<f4",
            "row_order": "R0106 compact retained order",
            "ordered_chunks": members,
        },
        "inference": {
            "batch_rows": TRANSFORM_BATCH_ROWS,
            "chunk_rows": TRANSFORM_CHUNK_ROWS,
            "all_real_rows_projected": True,
        },
        "release_sha": active["manifest"]["release_sha"],
        "training_performed": False,
        "wall_seconds": time.monotonic() - started,
    }
    receipt = coordinate_seal(body)
    receipt_path = os.path.join(output, "actual-transform.json")
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    del bundle["model"], source
    gc.collect()
    return {**receipt, "receipt": expected_input_signature(receipt_path)}


def _load_graph_truth(
    compact_anchors: np.ndarray,
    part_outputs: Mapping[str, str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Gather each anchor's exact R0106 top-15 neighbors from sealed shards."""
    from experiments.round0106_nodes import _validate_part_receipt

    anchors = np.asarray(compact_anchors, dtype=np.int64)
    targets = np.empty((len(anchors), K), dtype=np.int64)
    weights = np.empty((len(anchors), K), dtype=np.float32)
    touched: dict[str, dict[str, Any]] = {}
    for part, spec in PARTS.items():
        mask = (
            (anchors >= int(spec["compact_start"]))
            & (anchors < int(spec["compact_stop"]))
        )
        if not np.any(mask):
            continue
        receipt, receipt_signature = _validate_part_receipt(
            str(part_outputs[part]), expected_sha256=None, part=part
        )
        touched[part] = {
            "receipt": receipt_signature,
            "shards": [],
        }
        positions = np.flatnonzero(mask)
        by_shard: dict[int, list[int]] = {}
        for position in positions.tolist():
            shard = (
                int(anchors[position]) - int(spec["compact_start"])
            ) // SHARD_ROWS
            by_shard.setdefault(shard, []).append(position)
        members = {
            int(member["shard"]): member for member in receipt["shards"]
        }
        for shard, selected_positions in sorted(by_shard.items()):
            member = members[shard]
            artifact_path = verify_signature(
                member["artifact"],
                label=f"R0106 {part} shard {shard}",
            )
            with np.load(artifact_path, allow_pickle=False) as archive:
                source = np.asarray(archive["sources"], dtype=np.int64).reshape(
                    -1, K
                )
                target = np.asarray(archive["targets"], dtype=np.int64).reshape(
                    -1, K
                )
                weight = np.asarray(archive["weights"], dtype=np.float32).reshape(
                    -1, K
                )
                rows = anchors[selected_positions] - int(
                    member["compact_start"]
                )
                if not np.all(
                    source[rows]
                    == anchors[selected_positions, None]
                ):
                    raise Round0108Error("R0106 source ordering changed")
                targets[selected_positions] = target[rows]
                weights[selected_positions] = weight[rows]
            touched[part]["shards"].append(member["artifact"])
    if (
        np.any(targets < 0)
        or np.any(targets >= RETAINED_ROWS)
        or np.any(targets == anchors[:, None])
        or np.any(np.diff(np.sort(targets, axis=1), axis=1) == 0)
    ):
        raise Round0108Error("R0106 anchor truth is malformed")
    return targets, weights, touched


def _family_arrays(path: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return (
            np.asarray(archive["representative_rows"], dtype=np.int64),
            np.asarray(archive["family_counts"], dtype=np.int64),
        )


def _id_identity(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.integer):
        return identity_for_rows(array.astype(np.int64, copy=False))
    text = np.ascontiguousarray(array.astype("U"))
    return {
        "rows": len(text),
        "dtype": text.dtype.str,
        "ordered_sha256": ordered_array_sha256(text),
    }


def run_core_score(
    _active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    """Score retained map geometry on fixed stratified anchors."""
    from basemap.panel_v2 import _self_knn

    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0108 core geometry score"
    )
    started = time.monotonic()
    calibration_path = os.path.join(
        str(job["calibration_output"]), "jina-density-calibration.json"
    )
    calibration = read_sealed(
        calibration_path,
        label="R0108 Jina calibration",
        schema=CALIBRATION_SCHEMA,
    )
    transform_path = os.path.join(
        str(job["transform_output"]), "actual-transform.json"
    )
    coordinates = CoordinateStream(str(job["transform_output"]))
    if (
        coordinates.receipt.get("round_id") != ROUND_ID
        or coordinates.receipt.get("map_key") != MAP_KEY
        or len(coordinates) != RETAINED_ROWS
    ):
        raise Round0108Error("R0108 coordinate stream identity changed")
    with np.load(
        str(job["selection"]), allow_pickle=False
    ) as selection:
        global_anchors = np.asarray(
            selection["core_global_rows"], dtype=np.int64
        )
        compact_anchors = np.asarray(
            selection["core_compact_rows"], dtype=np.int64
        )
        group_ids = np.asarray(
            selection["core_group_ids"], dtype=np.uint8
        )
    if (
        global_anchors.shape != (PANEL_ANCHORS,)
        or compact_anchors.shape != global_anchors.shape
        or group_ids.shape != global_anchors.shape
        or any(np.count_nonzero(group_ids == index) != ANCHORS_PER_GROUP
               for index in range(len(GROUPS)))
    ):
        raise Round0108Error("R0108 core anchor selection changed")

    graph15, graph_weights, graph_truth_inputs = _load_graph_truth(
        compact_anchors, job["part_outputs"]
    )
    config = _panel_config(anchors=PANEL_ANCHORS)
    fraction_k = max(K_LOW_MAX, int(math.ceil(FRACTION * RETAINED_ROWS)))
    low_neighbors, low_distances, low_guard = _self_knn(
        coordinates,
        compact_anchors,
        fraction_k,
        config,
        hi_dim=False,
        want_dist=True,
        exact=True,
    )
    low_neighbors = np.asarray(low_neighbors, dtype=np.int64)
    low_distances = np.asarray(low_distances, dtype=np.float32)
    if (
        low_neighbors.shape != (PANEL_ANCHORS, fraction_k)
        or low_distances.shape != low_neighbors.shape
    ):
        raise Round0108Error("R0108 low-D neighbor geometry changed")

    bundle = load_reviewed_model(
        train_output=str(job["train_output"]),
        graph_manifest_path=str(job["graph_manifest"]),
        graph_manifest_sha256=str(job["graph_manifest_sha256"]),
    )
    if not np.array_equal(
        np.asarray(bundle["mapping"][compact_anchors], dtype=np.int64),
        global_anchors,
    ):
        raise Round0108Error("R0108 compact/global anchor mapping changed")
    source = CompactInt8DequantizedArray(bundle["mapping"])
    high15, high_distances, high_guard = _self_knn(
        source,
        compact_anchors,
        K,
        config,
        hi_dim=True,
        want_dist=True,
        exact=True,
    )
    high15 = np.asarray(high15, dtype=np.int64)
    high_distances = np.asarray(high_distances, dtype=np.float64)
    if (
        high15.shape != (PANEL_ANCHORS, K)
        or high_distances.shape != high15.shape
    ):
        raise Round0108Error("R0108 exact high-D truth geometry changed")
    high_radius = high_distances[:, :K_DENSITY].mean(1)
    graph_recall_at_15 = recall_from_neighbors(
        high15, graph15, truth_k=K
    )
    low_radius = low_distances[:, :K_DENSITY].mean(1)
    representatives, family_counts = _family_arrays(str(job["eligibility"]))
    anchor_family_sizes = map_family_sizes(
        global_anchors, representatives, family_counts
    )
    density_mask = anchor_family_sizes < FAMILY_SIZE_CUTOFF
    density_value = pearson_log_radius(
        high_radius[density_mask], low_radius[density_mask]
    )
    density_summary, density_bootstrap, density_null = density_v2_calibration(
        high_radius[density_mask],
        low_radius[density_mask],
        bootstrap_draws=1_000,
        bootstrap_seed=10_803,
        null_draws=1_000,
        null_seed=10_804,
    )

    high10 = high15[:, :K_HIT]
    raw_global_metrics = projection_metrics(
        high10, low_neighbors, fraction_k=fraction_k
    )
    all_metrics = {
        "ffr": raw_global_metrics["ffr_diagnostic"],
        "recall_at_10": raw_global_metrics["recall_at_10"],
        "recall_at_50_of_high10": raw_global_metrics[
            "recall_at_50_of_high10"
        ],
    }
    group_metrics: dict[str, Any] = {}
    group_ffr: dict[str, float] = {}
    for index, group in enumerate(GROUPS):
        mask = group_ids == index
        raw_metrics = projection_metrics(
            high10[mask], low_neighbors[mask], fraction_k=fraction_k
        )
        metrics = {
            "ffr": raw_metrics["ffr_diagnostic"],
            "recall_at_10": raw_metrics["recall_at_10"],
            "recall_at_50_of_high10": raw_metrics[
                "recall_at_50_of_high10"
            ],
        }
        group_metrics[group] = {
            "anchors": int(mask.sum()),
            **metrics,
        }
        group_ffr[group] = float(metrics["ffr"])

    with np.load(str(job["labels"]), allow_pickle=False) as labels:
        dataset_ids = np.asarray(labels["dataset_id"], dtype=np.uint8)
        mapping = np.asarray(bundle["mapping"])
        low_groups = np.asarray(
            dataset_ids[mapping[low_neighbors[:, :K_DENSITY]]],
            dtype=np.uint8,
        )
    observed_mixing = np.zeros((len(GROUPS), len(GROUPS)), dtype=np.int64)
    for source_group in range(len(GROUPS)):
        values = low_groups[group_ids == source_group].reshape(-1)
        observed_mixing[source_group] = np.bincount(
            values, minlength=len(GROUPS)
        )

    anchor_coordinates = np.asarray(
        coordinates[compact_anchors], dtype=np.float32
    )
    extent_min = np.asarray(coordinates.min(axis=0), dtype=np.float64)
    extent_max = np.asarray(coordinates.max(axis=0), dtype=np.float64)
    span = extent_max - extent_min
    finite_noncollapsed = bool(
        np.isfinite(extent_min).all()
        and np.isfinite(extent_max).all()
        and np.all(span > 1e-6)
        and np.all(anchor_coordinates.std(axis=0) > 1e-8)
    )
    scaled = (
        (anchor_coordinates - extent_min)
        / np.maximum(span, 1e-12)
    )
    bins = np.clip((scaled * 128).astype(np.int64), 0, 127)
    occupied = len(np.unique(bins[:, 0] * 128 + bins[:, 1]))
    centroids = {
        group: anchor_coordinates[group_ids == index].mean(0).tolist()
        for index, group in enumerate(GROUPS)
    }
    centroid_array = np.asarray(
        [centroids[group] for group in GROUPS], dtype=np.float64
    )
    centroid_distances = np.linalg.norm(
        centroid_array[:, None, :] - centroid_array[None, :, :], axis=2
    )
    registered_floor = (
        calibration.get("floor_calibration") or {}
    ).get("registered_floor")
    decision = core_geometry_decision(
        density_value=density_value,
        density_floor=(
            float(registered_floor)
            if registered_floor is not None else None
        ),
        global_ffr=float(all_metrics["ffr"]),
        group_ffr=group_ffr,
        recall_at_10=float(all_metrics["recall_at_10"]),
        recall_at_50=float(all_metrics["recall_at_50_of_high10"]),
        finite_noncollapsed=finite_noncollapsed,
    )
    arrays_path = os.path.join(output, "core-panel-arrays.npz")
    atomic_save_new_npz(
        arrays_path,
        immutable=True,
        global_anchor_rows=global_anchors,
        compact_anchor_rows=compact_anchors,
        group_ids=group_ids,
        high_neighbors_top15=high15,
        graph_neighbors_top15=graph15,
        graph_fuzzy_weights=graph_weights,
        low_neighbors_top50=low_neighbors[:, :K_LOW_MAX],
        high_radius=high_radius,
        low_radius=low_radius,
        anchor_family_sizes=anchor_family_sizes,
        density_bootstrap=density_bootstrap,
        density_permuted_null=density_null,
        anchor_coordinates=anchor_coordinates,
        observed_map_mixing=observed_mixing,
        centroid_distances=centroid_distances,
    )
    receipt = seal({
        "schema": CORE_SCHEMA,
        "round_id": ROUND_ID,
        "map_key": MAP_KEY,
        "model": bundle["train"]["model"],
        "train_receipt": bundle["train_signature"],
        "transform_receipt": expected_input_signature(transform_path),
        "graph_manifest": bundle["graph_signature"],
        "graph_truth_inputs": graph_truth_inputs,
        "exact_high_d_truth": {
            **high_guard,
            "distance": "fp32 Euclidean, matching R0040/R0085 full-768 scorer",
            "top15_sha256": ordered_array_sha256(high15),
        },
        "selection": expected_input_signature(str(job["selection"])),
        "anchors": {
            "total": PANEL_ANCHORS,
            "per_group": ANCHORS_PER_GROUP,
            "seed": PANEL_SEED,
            "global_rows_sha256": ordered_array_sha256(global_anchors),
            "compact_rows_sha256": ordered_array_sha256(compact_anchors),
            "group_ids_sha256": ordered_array_sha256(group_ids),
        },
        "metrics": {
            "global": all_metrics,
            "by_group": group_metrics,
            "density_v2": {
                **density_summary,
                "correlation": density_value,
                "family_size_cutoff_exclusive": FAMILY_SIZE_CUTOFF,
                "anchors_after_filter": int(density_mask.sum()),
                "registered_jina_floor": registered_floor,
            },
        },
        "geometry_diagnostics": {
            "coordinate_extent": {
                "minimum": extent_min.tolist(),
                "maximum": extent_max.tolist(),
                "span": span.tolist(),
            },
            "anchor_axis_std": anchor_coordinates.std(0).tolist(),
            "occupancy_128x128_on_fixed_anchors": {
                "occupied_cells": occupied,
                "total_cells": 128 * 128,
                "fraction": occupied / (128 * 128),
            },
            "low_d_k15_radius": {
                "minimum": float(low_radius.min()),
                "median": float(np.median(low_radius)),
                "p90": float(np.percentile(low_radius, 90)),
                "maximum": float(low_radius.max()),
            },
            "group_centroids": centroids,
            "group_centroid_distance_matrix": centroid_distances.tolist(),
            "observed_anchor_map_mixing_k15": observed_mixing.tolist(),
            "graph_mixing_and_hubs": bundle["graph"]["diagnostics"],
            "r0106_graph_top15_recall_against_exact_high_d_top15": (
                graph_recall_at_15
            ),
            "low_dim_exact_search_guard": low_guard,
        },
        "decision": decision,
        "projection_ffr_role": "diagnostic-only",
        "arrays": expected_input_signature(arrays_path),
        "training_performed": False,
        "wall_seconds": time.monotonic() - started,
    })
    receipt_path = os.path.join(output, "core-geometry.json")
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    del bundle["model"], source, low_neighbors, low_distances
    gc.collect()
    return {**receipt, "receipt": expected_input_signature(receipt_path)}


def _probe_score(
    *,
    name: str,
    corpus: np.ndarray,
    queries: np.ndarray,
    corpus_ids: np.ndarray,
    query_ids: np.ndarray,
    model: Any,
    output: str,
    inputs: Mapping[str, Any],
    save_coordinates: bool,
    duplicate_policy: str,
) -> dict[str, Any]:
    started = time.monotonic()
    corpus_values = np.asarray(corpus)
    query_values = np.asarray(queries)
    duplicate_control = exact_split_duplicate_diagnostics(
        corpus_values, query_values
    )
    if duplicate_policy not in {
        "require-corpus-query-exact-family-disjoint",
        "diagnostic-only",
    }:
        raise Round0108Error("unknown probe duplicate policy")
    if (
        duplicate_policy
        == "require-corpus-query-exact-family-disjoint"
        and not duplicate_control["corpus_query_exact_family_disjoint"]
    ):
        raise Round0108Error(
            f"{name} exact embedding family crosses corpus/query split"
        )
    truth, truth_guard = exact_cosine_topk(
        query_values, corpus_values, k=K_HIT
    )
    corpus_coordinates = np.asarray(
        model.transform(corpus_values, batch_size=TRANSFORM_BATCH_ROWS),
        dtype=np.float32,
    )
    query_coordinates = np.asarray(
        model.transform(query_values, batch_size=TRANSFORM_BATCH_ROWS),
        dtype=np.float32,
    )
    from basemap.panel_v2 import cross_knn

    fraction_k = max(K_LOW_MAX, int(math.ceil(FRACTION * len(corpus_values))))
    low = cross_knn(
        query_coordinates,
        corpus_coordinates,
        fraction_k,
        _panel_config(anchors=len(query_values)),
        hi_dim=False,
        exact=True,
    )
    metrics = projection_metrics(truth, low, fraction_k=fraction_k)
    covariance = np.cov(corpus_coordinates, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    artifact = None
    if save_coordinates:
        path = os.path.join(output, f"{name}-coordinates.npz")
        atomic_save_new_npz(
            path,
            immutable=True,
            probe_corpus_coords=corpus_coordinates,
            probe_query_coords=query_coordinates,
            probe_corpus_ids=np.asarray(corpus_ids),
            probe_query_ids=np.asarray(query_ids),
            exact_high_d_top10=truth,
            low_d_top50=np.asarray(low[:, :K_LOW_MAX], dtype=np.int64),
        )
        artifact = expected_input_signature(path)
    return {
        "name": name,
        "status": "included",
        "probe": {
            "corpus_rows": len(corpus_values),
            "query_rows": len(query_values),
            "ffr": metrics["ffr_diagnostic"],
            **metrics,
            "fraction_k": fraction_k,
            "dispersion": {
                "axis_std": corpus_coordinates.std(0).tolist(),
                "axis_span": np.ptp(corpus_coordinates, axis=0).tolist(),
                "covariance_eigenvalues": eigenvalues.tolist(),
                "eigenvalue_ratio": (
                    float(eigenvalues[-1] / eigenvalues[0])
                    if eigenvalues[0] > 0 else None
                ),
            },
        },
        "truth": {
            **truth_guard,
            "ordered_top10_sha256": ordered_array_sha256(truth),
        },
        "selection": {
            "corpus": _id_identity(np.asarray(corpus_ids)),
            "queries": _id_identity(np.asarray(query_ids)),
            "disjoint": len(np.intersect1d(corpus_ids, query_ids)) == 0,
        },
        "coordinates": artifact,
        "inputs": dict(inputs),
        "duplicate_control": {
            **duplicate_control,
            "policy": duplicate_policy,
            "passed": (
                duplicate_control["corpus_query_exact_family_disjoint"]
                if duplicate_policy
                == "require-corpus-query-exact-family-disjoint"
                else None
            ),
        },
        "projection_ffr_role": "diagnostic-only",
        "wall_seconds": time.monotonic() - started,
    }


def _load_selected(
    source_path: str,
    corpus_rows: np.ndarray,
    query_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.load(source_path, mmap_mode="r", allow_pickle=False)
    if source.ndim != 2 or source.shape[1] != DIMENSION:
        raise Round0108Error(f"probe source geometry changed: {source_path}")
    return (
        np.asarray(source[corpus_rows]),
        np.asarray(source[query_rows]),
    )


def run_ood(
    _active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate nineteen in-mix languages, held-out Polish, and map cards."""
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0108 OOD evaluation"
    )
    started = time.monotonic()
    bundle = load_reviewed_model(
        train_output=str(job["train_output"]),
        graph_manifest_path=str(job["graph_manifest"]),
        graph_manifest_sha256=str(job["graph_manifest_sha256"]),
    )
    model = bundle["model"]
    training_source = CompactInt8DequantizedArray(bundle["mapping"])
    selection_signature = expected_input_signature(str(job["selection"]))
    in_mix_control_vectors: list[np.ndarray] = []
    in_mix_control_rows: list[np.ndarray] = []
    in_mix_control_languages: list[np.ndarray] = []
    polish_queries: np.ndarray | None = None
    polish_query_rows: np.ndarray | None = None
    with np.load(str(job["selection"]), allow_pickle=False) as selected:
        language_metrics: dict[str, Any] = {}
        language_artifacts: dict[str, Any] = {}
        for index, language in enumerate((*IN_MIX_LANGUAGES, POLISH)):
            corpus_rows = np.asarray(
                selected[f"{language}__corpus"], dtype=np.int64
            )
            query_rows = np.asarray(
                selected[f"{language}__queries"], dtype=np.int64
            )
            source_spec = job["language_sources"][language]
            source_path = str(source_spec["canonical_path"])
            if expected_input_signature(source_path) != dict(source_spec):
                raise Round0108Error(f"{language} source bytes changed")
            corpus, queries = _load_selected(
                source_path, corpus_rows, query_rows
            )
            report = _probe_score(
                name=language,
                corpus=corpus,
                queries=queries,
                corpus_ids=corpus_rows,
                query_ids=query_rows,
                model=model,
                output=output,
                inputs={
                    "source": source_spec,
                    "selection": selection_signature,
                    "prompt_semantics": "R0087 production embedding convention",
                    "dimension": DIMENSION,
                    "training_membership": (
                        "absent"
                        if language == POLISH
                        else "rows beyond R0087 selected prefix"
                    ),
                },
                save_coordinates=language == POLISH,
                duplicate_policy=(
                    "require-corpus-query-exact-family-disjoint"
                ),
            )
            language_metrics[language] = report["probe"]
            language_artifacts[language] = report
            if language == POLISH:
                polish_queries = np.asarray(queries, dtype=np.float32)
                polish_query_rows = query_rows.copy()
            else:
                # Pool exactly 500 in-mix queries, balanced to within one row
                # across the nineteen sources, for a diagnostic cross-atlas
                # comparator.  This does not participate in the frozen gate.
                count = (
                    HELDOUT_QUERY_ROWS // len(IN_MIX_LANGUAGES)
                    + (
                        1
                        if index
                        < HELDOUT_QUERY_ROWS % len(IN_MIX_LANGUAGES)
                        else 0
                    )
                )
                rng = np.random.RandomState(CROSS_ATLAS_CONTROL_SEED + index)
                chosen = np.sort(
                    rng.choice(len(queries), size=count, replace=False)
                )
                in_mix_control_vectors.append(
                    np.asarray(queries[chosen], dtype=np.float32)
                )
                in_mix_control_rows.append(query_rows[chosen])
                in_mix_control_languages.append(
                    np.full(count, language, dtype="U16")
                )
            print(
                f"R0108 OOD {index + 1}/20 {language}: "
                f"r50={report['probe']['recall_at_50_of_high10']:.4f}",
                flush=True,
            )
            del corpus, queries

        headline = headline_ood_decision(language_metrics)

        if polish_queries is None or polish_query_rows is None:
            raise Round0108Error("held-out Polish query cell is missing")
        control_queries = np.concatenate(in_mix_control_vectors, axis=0)
        control_rows = np.concatenate(in_mix_control_rows)
        control_languages = np.concatenate(in_mix_control_languages)
        if len(control_queries) != HELDOUT_QUERY_ROWS:
            raise Round0108Error("in-mix cross-atlas control does not close")
        cross_queries = np.concatenate(
            (control_queries, polish_queries), axis=0
        )
        exact_training_top10, exact_training_guard = exact_cosine_topk(
            cross_queries,
            training_source,
            k=K_HIT,
            candidate_block_rows=200_000,
        )
        cross_coordinates = np.asarray(
            model.transform(
                cross_queries, batch_size=TRANSFORM_BATCH_ROWS
            ),
            dtype=np.float32,
        )
        training_coordinates = CoordinateStream(str(job["transform_output"]))
        from basemap.panel_v2 import cross_knn

        fraction_k = max(
            K_LOW_MAX, int(math.ceil(FRACTION * RETAINED_ROWS))
        )
        low_training_neighbors = cross_knn(
            cross_coordinates,
            training_coordinates,
            fraction_k,
            _panel_config(anchors=len(cross_queries)),
            hi_dim=False,
            exact=True,
        )
        split = len(control_queries)
        control_alignment = projection_metrics(
            exact_training_top10[:split],
            low_training_neighbors[:split],
            fraction_k=fraction_k,
        )
        polish_alignment = projection_metrics(
            exact_training_top10[split:],
            low_training_neighbors[split:],
            fraction_k=fraction_k,
        )
        alignment_path = os.path.join(
            output, "polish-to-training-atlas-alignment.npz"
        )
        atomic_save_new_npz(
            alignment_path,
            immutable=True,
            in_mix_query_coordinates=cross_coordinates[:split],
            in_mix_query_rows=control_rows,
            in_mix_query_languages=control_languages,
            in_mix_exact_training_top10=exact_training_top10[:split],
            in_mix_low_training_top50=low_training_neighbors[
                :split, :K_LOW_MAX
            ],
            polish_query_coordinates=cross_coordinates[split:],
            polish_query_rows=polish_query_rows,
            polish_exact_training_top10=exact_training_top10[split:],
            polish_low_training_top50=low_training_neighbors[
                split:, :K_LOW_MAX
            ],
        )
        cross_atlas_alignment = {
            "role": "diagnostic-only; excluded from the frozen atlas gate",
            "question": (
                "do held-out Polish queries attach near the same retained "
                "training rows in high-D and in the projected atlas?"
            ),
            "training_universe": {
                "rows": RETAINED_ROWS,
                "compact_mapping": bundle["graph"]["compact_mapping"],
                "representation": (
                    "R0103 signed-int8 times fp16 scale, normalized only "
                    "for exact cosine truth"
                ),
            },
            "query_cells": {
                "in_mix_balanced_500": {
                    **control_alignment,
                    "queries": len(control_queries),
                    "language_count": len(IN_MIX_LANGUAGES),
                    "rows_sha256": ordered_array_sha256(control_rows),
                    "languages_sha256": ordered_array_sha256(
                        control_languages
                    ),
                },
                "pol_Latn_500": {
                    **polish_alignment,
                    "queries": len(polish_queries),
                    "rows_sha256": ordered_array_sha256(polish_query_rows),
                },
            },
            "polish_to_in_mix_recall50_ratio": (
                polish_alignment["recall_at_50_of_high10"]
                / control_alignment["recall_at_50_of_high10"]
                if control_alignment["recall_at_50_of_high10"] > 0
                else None
            ),
            "exact_high_d_search": exact_training_guard,
            "low_d_search": {
                "backend": "panel-v2 exact global chunked top-k",
                "fraction_k": fraction_k,
                "corpus_rows": RETAINED_ROWS,
            },
            "arrays": expected_input_signature(alignment_path),
        }
        del (
            control_queries,
            cross_queries,
            cross_coordinates,
            exact_training_top10,
            low_training_neighbors,
            training_coordinates,
        )

        diagnostic_reports: dict[str, Any] = {}
        dad_corpus_rows = np.asarray(
            selected["dadabase__corpus"], dtype=np.int64
        )
        dad_query_rows = np.asarray(
            selected["dadabase__queries"], dtype=np.int64
        )
        dad_source = job["diagnostic_sources"]["dadabase"]
        dad_corpus, dad_queries = _load_selected(
            dad_source["canonical_path"], dad_corpus_rows, dad_query_rows
        )
        diagnostic_reports["dadabase"] = _probe_score(
            name="dadabase",
            corpus=dad_corpus,
            queries=dad_queries,
            corpus_ids=dad_corpus_rows,
            query_ids=dad_query_rows,
            model=model,
            output=output,
            inputs={
                "embeddings": dad_source,
                "texts": job["diagnostic_sources"]["dadabase_texts"],
                "selection": selection_signature,
                "prompt_semantics": "Jina-v5-nano Dadabase embedding artifact",
                "dimension": DIMENSION,
            },
            save_coordinates=True,
            duplicate_policy="diagnostic-only",
        )
        del dad_corpus, dad_queries

        fineweb_corpus_rows = np.asarray(
            selected["fineweb__corpus"], dtype=np.int64
        )
        fineweb_query_rows = np.asarray(
            selected["fineweb__queries"], dtype=np.int64
        )
        fineweb_source = job["diagnostic_sources"]["fineweb"]
        fw_corpus, fw_queries = _load_selected(
            fineweb_source["canonical_path"],
            fineweb_corpus_rows,
            fineweb_query_rows,
        )
        diagnostic_reports["fineweb-heldout"] = _probe_score(
            name="fineweb-heldout",
            corpus=fw_corpus,
            queries=fw_queries,
            corpus_ids=fineweb_corpus_rows,
            query_ids=fineweb_query_rows,
            model=model,
            output=output,
            inputs={
                "embeddings": fineweb_source,
                "selection": selection_signature,
                "prompt_semantics": "R0087 FineWeb production convention",
                "training_membership": "dedicated FineWeb held-out artifact",
                "dimension": DIMENSION,
            },
            save_coordinates=True,
            duplicate_policy="diagnostic-only",
        )
        del fw_corpus, fw_queries

    trec_corpus_spec = job["diagnostic_sources"]["trec_corpus"]
    trec_query_spec = job["diagnostic_sources"]["trec_queries"]
    trec_corpus = np.load(
        trec_corpus_spec["canonical_path"], mmap_mode="r", allow_pickle=False
    )
    trec_queries = np.load(
        trec_query_spec["canonical_path"], mmap_mode="r", allow_pickle=False
    )
    with open(
        job["diagnostic_sources"]["trec_corpus_ids"]["canonical_path"],
        encoding="utf-8",
    ) as handle:
        trec_corpus_ids = np.asarray(json.load(handle))
    with open(
        job["diagnostic_sources"]["trec_query_ids"]["canonical_path"],
        encoding="utf-8",
    ) as handle:
        trec_query_ids = np.asarray(json.load(handle))
    diagnostic_reports["trec-covid"] = _probe_score(
        name="trec-covid",
        corpus=trec_corpus,
        queries=trec_queries,
        corpus_ids=trec_corpus_ids,
        query_ids=trec_query_ids,
        model=model,
        output=output,
        inputs={
            "corpus_embeddings": trec_corpus_spec,
            "query_embeddings": trec_query_spec,
            "corpus_ids": job["diagnostic_sources"]["trec_corpus_ids"],
            "query_ids": job["diagnostic_sources"]["trec_query_ids"],
            "prompt_semantics": "Jina-v5-nano BEIR corpus/query conventions",
            "dimension": DIMENSION,
        },
        save_coordinates=True,
        duplicate_policy="diagnostic-only",
    )

    transform_path = os.path.join(
        str(job["transform_output"]), "actual-transform.json"
    )
    panel = seal({
        "schema": "universality-panel-v1",
        "round_id": ROUND_ID,
        "map": {
            "label": MAP_LABEL,
            "model": bundle["train"]["model"],
            "coordinate_receipt": expected_input_signature(transform_path),
        },
        "probes": {
            name: {
                "status": report["status"],
                "probe": report["probe"],
                "coordinates": report["coordinates"],
                "inputs": report["inputs"],
                "truth": report["truth"],
                "selection": report["selection"],
                "duplicate_control": report["duplicate_control"],
                "verdict": "diagnostic-only",
            }
            for name, report in diagnostic_reports.items()
        },
        "headline_ood_probe": POLISH,
        "projection_ffr_role": "diagnostic-only",
        "training_performed": False,
    })
    panel_path = os.path.join(output, "universality-panel-v1.json")
    atomic_write_new_json(panel_path, panel, immutable=True)
    receipt = seal({
        "schema": OOD_SCHEMA,
        "round_id": ROUND_ID,
        "map_key": MAP_KEY,
        "model": bundle["train"]["model"],
        "train_receipt": bundle["train_signature"],
        "selection": selection_signature,
        "language_cells": language_artifacts,
        "headline_decision": headline,
        "cross_atlas_alignment": cross_atlas_alignment,
        "diagnostic_map_cards": diagnostic_reports,
        "universality_panel": expected_input_signature(panel_path),
        "projection_ffr_role": "diagnostic-only",
        "universal_ood_claim_made": False,
        "training_performed": False,
        "wall_seconds": time.monotonic() - started,
    })
    receipt_path = os.path.join(output, "ood-evaluation.json")
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    del model, training_source
    gc.collect()
    return {**receipt, "receipt": expected_input_signature(receipt_path)}


def _registry_error(stage: str, exc: Exception) -> dict[str, Any]:
    return {
        "status": "failed",
        "stage": stage,
        "error_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
        "error_message": str(exc)[:2_000],
    }


def _refresh_registry_best_effort(
    *,
    receipt_path: str,
    map_definition_path: str,
    decision_path: str,
) -> dict[str, Any]:
    """Refresh mutable registry views without invalidating immutable evidence."""
    from experiments import map_registry

    stages: dict[str, Any] = {}
    registry: dict[str, Any] | None = None
    map_ids: list[str] = []
    expected_map_id = f"round-{ROUND_ID}-{MAP_KEY}"
    expected_projection_probes = {
        "dadabase", "fineweb-heldout", "trec-covid"
    }
    observed_projection_probes: set[str] = set()
    try:
        registry = map_registry.scan()
        round_entries = [
            item
            for item in registry["maps"]
            if item.get("round_id") == ROUND_ID
        ]
        map_ids = sorted(
            str(item["map_id"])
            for item in round_entries
        )
        observed_projection_probes = {
            str((item.get("projection") or {}).get("probe"))
            for item in round_entries
            if item.get("kind") == "projection-map"
        }
        stages["scan"] = {
            "status": "completed",
            "round_map_ids": map_ids,
            "projection_probes": sorted(observed_projection_probes),
        }
    except Exception as exc:
        stages["scan"] = _registry_error("scan", exc)

    if registry is None:
        stages["write_mutable_registry"] = {
            "status": "skipped",
            "reason": "registry scan failed",
        }
        stages["publish_site"] = {
            "status": "skipped",
            "reason": "registry scan failed",
        }
        stages["site_artifacts"] = {
            "status": "skipped",
            "reason": "registry scan failed",
        }
    else:
        try:
            history = map_registry.write_registry(registry)
            stages["write_mutable_registry"] = {
                "status": "completed",
                "mutable_view_observed": expected_input_signature(
                    str(map_registry.REGISTRY_PATH)
                ),
                "immutable_history_snapshot": (
                    expected_input_signature(str(history))
                    if history is not None else None
                ),
                "scientific_input_binding": False,
            }
        except Exception as exc:
            stages["write_mutable_registry"] = _registry_error(
                "write_mutable_registry", exc
            )
        try:
            map_registry.publish(registry)
            stages["publish_site"] = {
                "status": "completed",
                "site_url": map_registry.SITE_URL,
            }
        except Exception as exc:
            stages["publish_site"] = _registry_error("publish_site", exc)

        if stages["publish_site"]["status"] != "completed":
            stages["site_artifacts"] = {
                "status": "skipped",
                "reason": "site publisher failed",
            }
        else:
            try:
                expected_projection_entries = {
                    str((item.get("projection") or {}).get("probe")): item
                    for item in registry["maps"]
                    if (
                        item.get("round_id") == ROUND_ID
                        and item.get("kind") == "projection-map"
                        and str((item.get("projection") or {}).get("probe"))
                        in expected_projection_probes
                    )
                }
                missing = []
                base_page = (
                    map_registry.SITE_DIR
                    / f"round-{ROUND_ID}"
                    / "index.html"
                )
                if not base_page.is_file():
                    missing.append(str(base_page))
                for probe in sorted(expected_projection_probes):
                    entry = expected_projection_entries.get(probe)
                    if entry is None:
                        missing.append(f"registry projection probe:{probe}")
                        continue
                    root = (
                        map_registry.SITE_DIR
                        / "projections"
                        / str(entry["map_id"])
                    )
                    for name in ("index.html", "manifest.json"):
                        if not (root / name).is_file():
                            missing.append(str(root / name))
                stages["site_artifacts"] = {
                    "status": "completed" if not missing else "failed",
                    "expected_projection_probes": sorted(
                        expected_projection_probes
                    ),
                    "missing": missing,
                }
            except Exception as exc:
                stages["site_artifacts"] = _registry_error(
                    "validate_site_artifacts", exc
                )

    expected_map_discovered = expected_map_id in map_ids
    expected_projections_discovered = (
        expected_projection_probes <= observed_projection_probes
    )
    stages["inventory_validation"] = {
        "status": (
            "completed"
            if expected_map_discovered and expected_projections_discovered
            else "failed"
        ),
        "expected_map_id": expected_map_id,
        "expected_map_discovered": expected_map_discovered,
        "expected_projection_probes": sorted(expected_projection_probes),
        "observed_projection_probes": sorted(
            observed_projection_probes
        ),
        "expected_projections_discovered": (
            expected_projections_discovered
        ),
    }
    view_published = (
        expected_map_discovered
        and all(
            stages[name]["status"] == "completed"
            for name in (
                "scan",
                "write_mutable_registry",
                "publish_site",
                "inventory_validation",
                "site_artifacts",
            )
        )
    )
    receipt = seal({
        "schema": "round0108-map-registry-publication-v2",
        "round_id": ROUND_ID,
        "immutable_artifacts": {
            "map_definition": expected_input_signature(map_definition_path),
            "atlas_decision": expected_input_signature(decision_path),
        },
        "expected_map_ids": [expected_map_id],
        "expected_projection_probes": sorted(expected_projection_probes),
        "observed_round_map_ids": map_ids,
        "mutable_view_refresh": {
            "status": (
                "published"
                if view_published
                else "deferred-best-effort-view-refresh"
            ),
            "stages": stages,
            "requires_followup": not view_published,
            "scientific_decision_affected": False,
        },
    })
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    return receipt


def run_decision(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0108 atlas decision"
    )
    calibration_path = os.path.join(
        str(job["calibration_output"]), "jina-density-calibration.json"
    )
    core_path = os.path.join(
        str(job["core_output"]), "core-geometry.json"
    )
    ood_path = os.path.join(
        str(job["ood_output"]), "ood-evaluation.json"
    )
    calibration = read_sealed(
        calibration_path,
        label="R0108 calibration",
        schema=CALIBRATION_SCHEMA,
    )
    core = read_sealed(
        core_path, label="R0108 core geometry", schema=CORE_SCHEMA
    )
    ood = read_sealed(
        ood_path, label="R0108 OOD evaluation", schema=OOD_SCHEMA
    )
    core_passed = bool((core.get("decision") or {}).get("passed"))
    ood_passed = bool((ood.get("headline_decision") or {}).get("passed"))
    accepted = core_passed and ood_passed
    receipt = seal({
        "schema": DECISION_SCHEMA,
        "round_id": ROUND_ID,
        "map_key": MAP_KEY,
        "release_sha": active["manifest"]["release_sha"],
        "calibration": expected_input_signature(calibration_path),
        "core_geometry": expected_input_signature(core_path),
        "ood_evaluation": expected_input_signature(ood_path),
        "checks": {
            "jina_density_floor_registered": bool(
                (calibration.get("floor_calibration") or {}).get(
                    "gating_floor_registered"
                )
            ),
            "core_geometry_passed": core_passed,
            "headline_polish_ood_passed": ood_passed,
            "projection_ffr_excluded_from_decision": True,
        },
        "atlas_quality_capability_released": accepted,
        "map_registry_artifacts_complete": all(
            (ood.get("diagnostic_map_cards") or {}).get(name, {}).get(
                "coordinates"
            )
            for name in ("dadabase", "trec-covid")
        ),
        "outcome": "accepted" if accepted else "failed-with-diagnostics",
        "universal_ood_claim_made": False,
        "training_performed": False,
    })
    receipt_path = os.path.join(output, "atlas-decision.json")
    atomic_write_new_json(receipt_path, receipt, immutable=True)

    core_arrays = verify_signature(
        core.get("arrays"), label="R0108 core arrays"
    )
    with np.load(core_arrays, allow_pickle=False) as archive:
        sample_ids = np.asarray(
            archive["compact_anchor_rows"], dtype=np.int64
        )
    render_root = create_fresh_directory(
        str(job["render_output"]), label="R0108 registry support"
    )
    sample_path = os.path.join(render_root, "sample-semantic-ids.npy")
    atomic_save_new_npy(sample_path, sample_ids, immutable=True)
    definitions = seal({
        "schema": "round0108-map-definition-v1",
        "round_id": ROUND_ID,
        "map_key": MAP_KEY,
        "map_label": MAP_LABEL,
        "training_round": "0107",
        "coordinates": expected_input_signature(
            os.path.join(
                str(job["transform_output"]), "actual-transform.json"
            )
        ),
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
        raise Round0108Error("R0108 handler requires its exact round/job")
    handlers = {
        "calibrate_jina_density": run_calibration,
        "transform_retained_map": run_transform,
        "score_core_geometry": run_core_score,
        "score_ood": run_ood,
        "decide_and_publish_registry": run_decision,
    }
    try:
        handler = handlers[str(job["action"])]
    except KeyError as exc:
        raise Round0108Error(
            f"unknown R0108 action {job.get('action')!r}"
        ) from exc
    return handler(active, job)
