"""R0123 exact crossed-representation density-alignment panel.

This no-training panel crosses two frozen maps with two embedding
representations over the exact same R0113 compact row and anchor namespaces:

    R0104 fp16 map x {R0104 source, R0113 fresh raw}
    R0115 raw map  x {R0104 source, R0113 fresh raw}

Each cell is scored against the high-dimensional radii of its *input*
representation.  The decision uses paired bootstrap differences between a
map's matched and crossed inputs.  It deliberately does not apply an absolute
density floor across different high-dimensional spaces.
"""
from __future__ import annotations

import dataclasses
import gc
import json
import math
import os
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    ordered_array_sha256,
    sha256_bytes,
)
from basemap.output_safety import (
    atomic_save_new_npz,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.panel_v2 import (
    PanelV2Config,
    load_hiD_reference,
)
from basemap.round0104_training import InventoryFp16Array, L2NormalizedArray
from basemap.round0108_evaluation import (
    TRANSFORM_BATCH_ROWS,
    seal,
    validate_seal,
)
from experiments.round0119_nodes import (
    _authenticate_model as _authenticate_r0119_model,
    _read_json_signature,
)
from experiments.round0122_nodes import (
    DECISION_SCHEMA as R0122_DECISION_SCHEMA,
    R0115_NATIVE_HIGH_D_REFERENCE_KEY,
    SCORE_SCHEMA as R0122_SCORE_SCHEMA,
    _authenticate_r0104_model,
)


ROUND_ID = "0123"
R0122_RELEASE_SHA = "79c228e0b5d22027bf76a188c1f1daf895bb2aec"
R0122_REQUIRED_OUTCOME = "failure-enters-after-r0104-within-r0115-bundle"
R0104_RELEASE_SHA = "2b1b51746d4aeb01e9dd88b19aa6dc80ccbb8329"
R0115_RELEASE_SHA = "3b6ed28e1801e13228c78e05cf992a30e398a678"

SOURCE_ROWS = 2_000_000
COMPACT_ROWS = 1_993_761
DIMENSION = 768
ANCHORS = 4_000
K_DENSITY = 15
BOOTSTRAP_DRAWS = 1_000
BOOTSTRAP_SEED = 12_301

ASSEMBLY_SHA256 = (
    "432b3caf8e29944eddea553e5d390003d94eb37c9c4d6835c0b0e0cba2b62486"
)
MAPPING_SHA256 = (
    "64c82d495777fa73e075706c495c3feaf1346b5abbd2579f1cf750603714f371"
)
FRESH_INPUT_SHA256 = (
    "e8e7a718ecda27617b2bf9c33cb12b6f5506244b61cdb48a72e1ae3128940e37"
)
FRESH_HIGH_D_REFERENCE_SHA256 = (
    "f64be840213652410460731ef9a08d9537367f11e87f8a8656e85d1c988cba8b"
)
LEGACY_MODEL_SHA256 = (
    "36a7fb86784b6a891f7c73b83d008aead320a7729eea913efc117e4bcd5b3e08"
)
FRESH_MODEL_SHA256 = (
    "cae817cee31e17ae2bb06be732df18dd1a8ecfbd5c3c56c651b6300a88cb47ac"
)
R0104_SOURCE_PAYLOAD_SHA256 = (
    "f4a0050e81a3755de84ba73405ba6823fa387f09a15d3ad299083fa60093f069"
)
R0115_REVIEW_SHA256 = (
    "cbc6ad74773624a0fd8ea966f5a1e9cd37be120b554a0ca56c28011720d3bb02"
)
R0115_RESULT_SHA256 = (
    "7e2a28c703dc3b793ce38fb47badc4471ca66b94ba6b7836242cd68e7dc2dba6"
)

MAP_ORDER = ("legacy_map", "fresh_map")
INPUT_ORDER = ("legacy_input", "fresh_input")
CELL_ORDER = (
    "legacy_map__legacy_input",
    "legacy_map__fresh_input",
    "fresh_map__legacy_input",
    "fresh_map__fresh_input",
)
PANEL_SCHEMA = "round0123-crossed-representation-density-panel-v1"
DECISION_SCHEMA = "round0123-crossed-representation-alignment-decision-v1"


class Round0123Error(RuntimeError):
    """Raised when R0123's exact registered contract changes."""


class IndexedRowsArray:
    """Lazy ordered row selection without materializing the compact matrix."""

    def __init__(self, source: Any, row_ids: np.ndarray):
        rows = np.asarray(row_ids)
        if (
            rows.ndim != 1
            or rows.dtype != np.dtype("int64")
            or not len(rows)
            or np.any(rows[1:] <= rows[:-1])
            or int(rows[0]) < 0
            or int(rows[-1]) >= len(source)
            or len(source.shape) != 2
        ):
            raise Round0123Error("compact indexed-row view is malformed")
        self.source = source
        self.row_ids = rows
        self.shape = (len(rows), int(source.shape[1]))
        self.dtype = np.dtype(source.dtype)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key: Any) -> np.ndarray:
        scalar = isinstance(key, (int, np.integer))
        selected = self.row_ids[key]
        values = self.source[selected]
        result = np.asarray(values)
        if scalar and result.shape != (self.shape[1],):
            raise Round0123Error("indexed scalar row shape changed")
        return result


def _panel_config() -> PanelV2Config:
    return PanelV2Config(
        frac=0.001,
        k_clust=(),
        k_density=K_DENSITY,
        k_hit=10,
        n_anchors=ANCHORS,
        anchor_seed=123,
        corpus_chunk=500_000,
        overselect=8,
        block_elems=500_000_000,
        rerank_byte_cap=2_000_000_000,
        rerank_scratch=3.0,
        peak_byte_cap=26_000_000_000,
    )


def _exact_signature(
    expected: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    path = str(expected.get("canonical_path") or "")
    actual = expected_input_signature(path)
    if actual != dict(expected):
        raise Round0123Error(f"{label} bytes changed")
    return actual


def _r0122_gate(job: Mapping[str, Any]) -> dict[str, Any]:
    evidence = job.get("r0122_evidence")
    if not isinstance(evidence, Mapping):
        raise Round0123Error("accepted R0122 evidence is missing")
    signatures = {
        key: _exact_signature(evidence[key], label=f"R0122 {key}")
        for key in (
            "review",
            "result",
            "queue",
            "terminal",
            "score",
            "decision",
        )
    }
    with open(
        signatures["score"]["canonical_path"], encoding="utf-8"
    ) as handle:
        score = json.load(handle)
    with open(
        signatures["decision"]["canonical_path"], encoding="utf-8"
    ) as handle:
        decision = json.load(handle)
    validate_seal(score, label="R0122 density provenance bridge panel")
    validate_seal(decision, label="R0122 density provenance bridge decision")
    if (
        score.get("schema") != R0122_SCORE_SCHEMA
        or score.get("round_id") != "0122"
        or score.get("release_sha") != R0122_RELEASE_SHA
        or score.get("training_performed") is not False
        or decision.get("schema") != R0122_DECISION_SCHEMA
        or decision.get("round_id") != "0122"
        or decision.get("release_sha") != R0122_RELEASE_SHA
        or decision.get("score") != signatures["score"]
        or decision.get("outcome") != R0122_REQUIRED_OUTCOME
        or decision.get("evaluation_path_material") is not False
        or decision.get("boundary_localized") is not True
        or decision.get("single_factor_cause_localized") is not False
        or decision.get("native_training_geometry_declared_bad") is not False
        or decision.get("production_transfer_claimed") is not False
        or decision.get("training_performed") is not False
    ):
        raise Round0123Error(
            "R0122 did not release the registered conditional branch"
        )
    return {
        **signatures,
        "required_outcome": R0122_REQUIRED_OUTCOME,
        "accepted": True,
    }


def _load_r0104_source(
    job: Mapping[str, Any],
) -> tuple[IndexedRowsArray, dict[str, Any]]:
    shared, shared_signature = _read_json_signature(
        job["r0104_shared_evidence"],
        label="R0104 shared evidence",
        sealed=True,
    )
    source_proof = shared.get("source_prefix_proof")
    if (
        shared.get("schema") != "round0104-paired-shared-evidence-v2"
        or shared.get("round_id") != "0104"
        or shared.get("release_sha") != R0104_RELEASE_SHA
        or not isinstance(source_proof, Mapping)
        or source_proof.get("schema")
        != "round0104-r0103-first2m-source-proof-v2"
        or source_proof.get("rows") != SOURCE_ROWS
        or source_proof.get("dimension") != DIMENSION
        or source_proof.get("dtype") != "<f2"
        or source_proof.get("payload_sha256")
        != R0104_SOURCE_PAYLOAD_SHA256
        or source_proof.get("cross_round_row_equivalence_claimed") is not False
    ):
        raise Round0123Error("R0104 exact source-prefix evidence changed")
    segments = source_proof.get("segments")
    expected_segments = job.get("r0104_source_segments")
    if (
        not isinstance(segments, list)
        or not isinstance(expected_segments, list)
        or segments != expected_segments
    ):
        raise Round0123Error("R0104 source segment order changed")
    for index, segment in enumerate(segments):
        _exact_signature(
            segment["shard"], label=f"R0104 source segment {index}"
        )

    source = InventoryFp16Array(0, SOURCE_ROWS)
    if source.segments != segments:
        raise Round0123Error("R0104 runtime source layout changed")
    mapping_signature = _exact_signature(
        job["compact_mapping"], label="R0113 compact-to-global mapping"
    )
    if mapping_signature["sha256"] != MAPPING_SHA256:
        raise Round0123Error("R0113 compact mapping identity changed")
    mapping = np.load(
        mapping_signature["canonical_path"], mmap_mode="r", allow_pickle=False
    )
    if (
        mapping.shape != (COMPACT_ROWS,)
        or mapping.dtype != np.dtype("int64")
        or np.any(mapping[1:] <= mapping[:-1])
        or int(mapping[0]) < 0
        or int(mapping[-1]) >= SOURCE_ROWS
    ):
        raise Round0123Error("R0113 compact mapping payload changed")
    return IndexedRowsArray(source, mapping), {
        "shared_evidence": shared_signature,
        "source_prefix_payload_sha256": R0104_SOURCE_PAYLOAD_SHA256,
        "source_segments": [dict(item["shard"]) for item in segments],
        "compact_mapping": mapping_signature,
        "compact_ids": {
            "rows": COMPACT_ROWS,
            "ordered_global_rows_sha256": ordered_array_sha256(mapping),
        },
    }


def _load_registered_inputs(job: Mapping[str, Any]) -> dict[str, Any]:
    r0122 = _r0122_gate(job)
    assembly, assembly_signature = _read_json_signature(
        job["assembly_manifest"],
        label="R0113 compact assembly",
        sealed=True,
    )
    mapping_signature = _exact_signature(
        job["compact_mapping"], label="R0113 compact mapping"
    )
    fresh_signature = _exact_signature(
        job["fresh_input"], label="R0113 fresh raw compact input"
    )
    if (
        assembly_signature["sha256"] != ASSEMBLY_SHA256
        or mapping_signature["sha256"] != MAPPING_SHA256
        or fresh_signature["sha256"] != FRESH_INPUT_SHA256
        or assembly.get("schema")
        != "round0113-compact-prompt-arrays-v1"
        or assembly.get("round_id") != "0113"
        or assembly.get("source_rows") != SOURCE_ROWS
        or assembly.get("retained_rows") != COMPACT_ROWS
        or assembly.get("dimension") != DIMENSION
        or assembly.get("dtype") != "<f2"
        or assembly.get("mapping") != mapping_signature
        or (assembly.get("outputs") or {}).get("raw") != fresh_signature
        or assembly.get("paired_row_population_identical") is not True
        or assembly.get("training_performed") is not False
    ):
        raise Round0123Error("R0113 compact assembly contract changed")

    legacy, legacy_lineage = _load_r0104_source(job)
    fresh = np.memmap(
        fresh_signature["canonical_path"],
        dtype="<f2",
        mode="r",
        shape=(COMPACT_ROWS, DIMENSION),
    )
    reference_signature = _exact_signature(
        job["fresh_high_d_reference"],
        label="R0115 fresh native high-D reference",
    )
    if reference_signature["sha256"] != FRESH_HIGH_D_REFERENCE_SHA256:
        raise Round0123Error("R0115 fresh high-D reference bytes changed")
    reference = load_hiD_reference(reference_signature["canonical_path"])
    anchors = np.asarray(reference["anchor_ids"], dtype=np.int64)
    fresh_radius = np.asarray(reference["r_hd"], dtype=np.float64)
    key_parts = reference.get("key_parts") or {}
    data = key_parts.get("data") or {}
    shards = data.get("shards") or []
    expected_config = dataclasses.asdict(_panel_config())
    if (
        reference.get("key") != R0115_NATIVE_HIGH_D_REFERENCE_KEY
        or anchors.shape != (ANCHORS,)
        or fresh_radius.shape != anchors.shape
        or np.any(anchors < 0)
        or np.any(anchors >= COMPACT_ROWS)
        or len(np.unique(anchors)) != ANCHORS
        or canonical_json(key_parts.get("config"))
        != canonical_json(expected_config)
        or key_parts.get("k_frac_effective") != 1_994
        or data.get("kind") != "ordered_shards"
        or data.get("shape") != [COMPACT_ROWS, DIMENSION]
        or data.get("dtype") != "<f2"
        or len(shards) != 1
        or shards[0].get("sha256") != fresh_signature["sha256"]
        or shards[0].get("bytes") != fresh_signature["bytes"]
        or (key_parts.get("convention") or {}).get("anchor_namespace")
        != "R0113 compact IDs"
        or (key_parts.get("convention") or {}).get("distance")
        != "cosine via fp32-L2-normalized squared L2"
    ):
        raise Round0123Error("R0115 fresh reference-space contract changed")
    return {
        "r0122": r0122,
        "assembly": assembly_signature,
        "mapping": mapping_signature,
        "legacy": legacy,
        "legacy_lineage": legacy_lineage,
        "fresh": fresh,
        "fresh_signature": fresh_signature,
        "anchors": anchors,
        "fresh_high_radius": fresh_radius,
        "fresh_reference": {
            "artifact": reference_signature,
            "key": reference["key"],
            "content_sha256": reference["content_sha256"],
            "high_radius_sha256": ordered_array_sha256(fresh_radius),
        },
    }


def _authenticate_models(
    job: Mapping[str, Any],
    *,
    device: str = "cuda",
) -> dict[str, dict[str, Any]]:
    legacy_spec = job.get("legacy_model_bundle")
    fresh_spec = job.get("fresh_model_bundle")
    if (
        not isinstance(legacy_spec, Mapping)
        or legacy_spec.get("key")
        != "r0104_fp16_seed42_full_transform"
        or legacy_spec.get("arm") != "fp16_control"
        or (legacy_spec.get("model") or {}).get("sha256")
        != LEGACY_MODEL_SHA256
        or not isinstance(fresh_spec, Mapping)
        or fresh_spec.get("key") != "current_2m_seed42"
        or fresh_spec.get("group") != "current_2m"
        or fresh_spec.get("arm") != "raw"
        or fresh_spec.get("seed") != 42
        or (fresh_spec.get("model") or {}).get("sha256")
        != FRESH_MODEL_SHA256
        or (fresh_spec.get("accepted_review") or {}).get("sha256")
        != R0115_REVIEW_SHA256
        or (fresh_spec.get("accepted_result") or {}).get("sha256")
        != R0115_RESULT_SHA256
    ):
        raise Round0123Error("registered crossed-map bundle changed")

    legacy = _authenticate_r0104_model(legacy_spec, device=device)
    fresh = _authenticate_r0119_model(fresh_spec, device=device)
    legacy_train, _ = _read_json_signature(
        legacy_spec["train_receipt"],
        label="R0104 fp16 train receipt",
        sealed=True,
    )
    fresh_train, _ = _read_json_signature(
        fresh_spec["train_receipt"],
        label="R0115 raw train receipt",
        sealed=True,
    )
    if (
        legacy_train.get("shared_evidence")
        != dict(job["r0104_shared_evidence"])
        or fresh_train.get("assembly") != dict(job["assembly_manifest"])
        or fresh.get("seed") != 42
        or fresh.get("group") != "current_2m"
        or fresh.get("training_representation")
        != "raw compact fp16 host memmap"
    ):
        raise Round0123Error("crossed-map execution lineage changed")
    return {
        "legacy_map": {
            **legacy,
            "map_key": "legacy_map",
            "source_round": "0104",
            "training_bundle": (
                "R0104 fp16 seed42, all first-2M rows, fuzzy-k50 paired "
                "host-weighted training"
            ),
        },
        "fresh_map": {
            **fresh,
            "map_key": "fresh_map",
            "source_round": "0115",
            "train_receipt": fresh["train"],
            "training_sampler": fresh[
                "authenticated_training_semantics"
            ]["sampler_class"],
            "training_updates": fresh[
                "authenticated_training_semantics"
            ]["successful_updates"],
            "training_bundle": (
                "R0115 raw seed42, R0113 compact rows, fuzzy-k50 prompt "
                "contrast host-weighted training"
            ),
        },
    }


def _exact_density_radii(
    corpus: Any,
    anchors: np.ndarray,
    *,
    high_dimensional: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    from basemap.panel_v2 import _self_knn

    _, distances, guard = _self_knn(
        corpus,
        anchors,
        K_DENSITY,
        _panel_config(),
        hi_dim=high_dimensional,
        want_dist=True,
        exact=True,
    )
    radii = np.asarray(distances.mean(1), dtype=np.float64)
    if (
        radii.shape != (len(anchors),)
        or np.any(radii < 0)
        or not np.isfinite(radii).all()
    ):
        raise Round0123Error("exact density radii are malformed")
    return radii, guard


def _pearson_log_radius(
    high_radius: np.ndarray,
    low_radius: np.ndarray,
) -> float:
    high = np.asarray(high_radius, dtype=np.float64)
    low = np.asarray(low_radius, dtype=np.float64)
    if (
        high.ndim != 1
        or low.shape != high.shape
        or len(high) < 100
        or np.any(high < 0)
        or np.any(low < 0)
        or not np.isfinite(high).all()
        or not np.isfinite(low).all()
    ):
        raise Round0123Error("density correlation radii are malformed")
    high = np.log(high + 1e-12)
    low = np.log(low + 1e-12)
    high -= high.mean()
    low -= low.mean()
    denominator = math.sqrt(
        float(np.dot(high, high)) * float(np.dot(low, low))
    )
    if not denominator > 0 or not math.isfinite(denominator):
        raise Round0123Error("density correlation variance collapsed")
    value = float(np.dot(high, low) / denominator)
    if not math.isfinite(value):
        raise Round0123Error("density correlation is nonfinite")
    return value


def run_score(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    output = create_fresh_directory(
        str(job["outputs"][0]),
        label="R0123 crossed-representation density panel",
    )
    started = time.monotonic()
    inputs = _load_registered_inputs(job)
    models = _authenticate_models(job)
    anchors = inputs["anchors"]
    legacy_high, legacy_guard = _exact_density_radii(
        L2NormalizedArray(inputs["legacy"]),
        anchors,
        high_dimensional=True,
    )
    high_radii = {
        "legacy_input": legacy_high,
        "fresh_input": inputs["fresh_high_radius"],
    }
    sources = {
        "legacy_input": inputs["legacy"],
        "fresh_input": inputs["fresh"],
    }
    cells: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {
        "anchor_compact_ids": anchors,
        "legacy_input__high_radius": legacy_high,
        "fresh_input__high_radius": inputs["fresh_high_radius"],
    }

    for map_key in MAP_ORDER:
        bundle = models[map_key]
        for input_key in INPUT_ORDER:
            cell_key = f"{map_key}__{input_key}"
            coordinates = np.asarray(
                bundle["model"].transform(
                    sources[input_key],
                    batch_size=TRANSFORM_BATCH_ROWS,
                ),
                dtype=np.float32,
            )
            if (
                coordinates.shape != (COMPACT_ROWS, 2)
                or not np.isfinite(coordinates).all()
            ):
                raise Round0123Error(
                    f"{cell_key} transformed coordinates are malformed"
                )
            low_radius, low_guard = _exact_density_radii(
                coordinates,
                anchors,
                high_dimensional=False,
            )
            correlation = _pearson_log_radius(
                high_radii[input_key], low_radius
            )
            cells[cell_key] = {
                "key": cell_key,
                "map": map_key,
                "input_representation": input_key,
                "matched_representation": (
                    (map_key, input_key)
                    in {
                        ("legacy_map", "legacy_input"),
                        ("fresh_map", "fresh_input"),
                    }
                ),
                "map_source_round": bundle["source_round"],
                "training_bundle": bundle["training_bundle"],
                "train_receipt": bundle["train_receipt"],
                "production_config": bundle["production_config"],
                "model": bundle["model_signature"],
                "compact_rows": COMPACT_ROWS,
                "anchor_compact_ids_sha256": ordered_array_sha256(anchors),
                "transform_batch_rows": TRANSFORM_BATCH_ROWS,
                "coordinates": {
                    "rows": len(coordinates),
                    "dimensions": 2,
                    "dtype": coordinates.dtype.str,
                    "ordered_sha256": ordered_array_sha256(coordinates),
                    "axis_standard_deviation": (
                        coordinates.std(axis=0).tolist()
                    ),
                    "finite": True,
                },
                "high_radius_sha256": ordered_array_sha256(
                    high_radii[input_key]
                ),
                "low_radius_sha256": ordered_array_sha256(low_radius),
                "density_correlation": correlation,
                "low_dim_exact_search_guard": low_guard,
                "historical_absolute_floor_applied": False,
            }
            arrays[f"{cell_key}__low_radius"] = low_radius
            del coordinates, low_radius
            gc.collect()

    arrays_path = os.path.join(
        output, "crossed-representation-radii.npz"
    )
    atomic_save_new_npz(arrays_path, immutable=True, **arrays)
    receipt = seal({
        "schema": PANEL_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "conditional_predecessor": inputs["r0122"],
        "common_population": {
            "name": "exact R0113 prompt-family-union compact FineWeb rows",
            "source_rows": SOURCE_ROWS,
            "compact_rows": COMPACT_ROWS,
            "dimension": DIMENSION,
            "assembly": inputs["assembly"],
            "mapping": inputs["mapping"],
            "ordered_global_rows_sha256": inputs["legacy_lineage"][
                "compact_ids"
            ]["ordered_global_rows_sha256"],
            "anchor_count": len(anchors),
            "anchor_compact_ids_sha256": ordered_array_sha256(anchors),
        },
        "input_representations": {
            "legacy_input": {
                "name": (
                    "exact R0104 first-2M fp16 source rows selected by "
                    "the R0113 compact-to-global mapping"
                ),
                **inputs["legacy_lineage"],
            },
            "fresh_input": {
                "name": "R0113 fresh local raw compact fp16",
                "artifact": inputs["fresh_signature"],
            },
        },
        "high_dimensional_references": {
            "legacy_input": {
                "construction": (
                    "one exact panel-v2.2 cosine k15 radius pass over "
                    "L2-normalized legacy compact rows"
                ),
                "panel_config": dataclasses.asdict(_panel_config()),
                "anchor_compact_ids_sha256": ordered_array_sha256(anchors),
                "high_radius_sha256": ordered_array_sha256(legacy_high),
                "exact_search_guard": legacy_guard,
            },
            "fresh_input": {
                "construction": (
                    "reuse accepted R0115 native high-D reference radii"
                ),
                **inputs["fresh_reference"],
            },
        },
        "cell_order": list(CELL_ORDER),
        "cells": cells,
        "arrays": expected_input_signature(arrays_path),
        "scorer": {
            "metric": "density-v2 Pearson correlation of log mean-k15 radii",
            "low_dimensional_search": "exact",
            "high_dimensional_reference_matches_each_input": True,
            "identical_compact_ids_and_anchors_across_all_cells": True,
            "historical_absolute_floor_applied": False,
        },
        "training_performed": False,
        "native_quality_claimed": False,
        "single_factor_cause_claimed": False,
        "production_transfer_claimed": False,
        "wall_seconds": time.monotonic() - started,
    })
    path = os.path.join(output, "crossed-representation-panel.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def _paired_alignment_contrasts(
    *,
    legacy_high: np.ndarray,
    fresh_high: np.ndarray,
    low_radii: Mapping[str, np.ndarray],
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if set(low_radii) != set(CELL_ORDER) or draws < 100:
        raise Round0123Error("paired alignment bootstrap contract changed")
    high = {
        "legacy_input": np.asarray(legacy_high, dtype=np.float64),
        "fresh_input": np.asarray(fresh_high, dtype=np.float64),
    }
    low = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in low_radii.items()
    }
    rows = len(high["legacy_input"])
    if (
        high["fresh_input"].shape != (rows,)
        or rows < 100
        or any(value.shape != (rows,) for value in low.values())
    ):
        raise Round0123Error("paired alignment arrays are not co-indexed")

    def correlations(index: np.ndarray) -> dict[str, float]:
        return {
            key: _pearson_log_radius(
                high[
                    "legacy_input"
                    if key.endswith("__legacy_input")
                    else "fresh_input"
                ][index],
                low[key][index],
            )
            for key in CELL_ORDER
        }

    all_rows = np.arange(rows, dtype=np.int64)
    point_cells = correlations(all_rows)

    def differences(values: Mapping[str, float]) -> dict[str, float]:
        legacy = (
            values["legacy_map__legacy_input"]
            - values["legacy_map__fresh_input"]
        )
        fresh = (
            values["fresh_map__fresh_input"]
            - values["fresh_map__legacy_input"]
        )
        return {
            "legacy_map_alignment_advantage": legacy,
            "fresh_map_alignment_advantage": fresh,
            "crossed_interaction": legacy + fresh,
        }

    point = differences(point_cells)
    samples = {
        key: np.empty(draws, dtype=np.float64) for key in point
    }
    rng = np.random.RandomState(seed)
    for draw in range(draws):
        selected = rng.randint(0, rows, size=rows)
        values = differences(correlations(selected))
        for key in samples:
            samples[key][draw] = values[key]

    contrasts: dict[str, Any] = {}
    for key, values in samples.items():
        interval = [
            float(np.quantile(values, 0.005)),
            float(np.quantile(values, 0.995)),
        ]
        direction = (
            "positive"
            if interval[0] > 0.0
            else "negative"
            if interval[1] < 0.0
            else "indeterminate"
        )
        contrasts[key] = {
            "point_difference": float(point[key]),
            "bootstrap_draws": draws,
            "bootstrap_seed": seed,
            "paired_anchor_resampling": True,
            "central_99_percent": interval,
            "direction": direction,
        }
    return {
        "cell_correlations": point_cells,
        "contrasts": contrasts,
    }, samples


def _classify_alignment(contrasts: Mapping[str, Any]) -> str:
    legacy = (
        contrasts.get("legacy_map_alignment_advantage") or {}
    ).get("direction")
    fresh = (
        contrasts.get("fresh_map_alignment_advantage") or {}
    ).get("direction")
    if legacy == "positive" and fresh == "positive":
        return "symmetric-representation-alignment-penalty"
    if legacy == "positive":
        return "legacy-map-positive-alignment-only"
    if fresh == "positive":
        return "fresh-map-positive-alignment-only"
    return "no-reliable-positive-map-input-alignment"


def run_decision(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    output = create_fresh_directory(
        str(job["outputs"][0]),
        label="R0123 crossed-representation alignment decision",
    )
    panel_path = os.path.join(
        str(job["score_output"]), "crossed-representation-panel.json"
    )
    with open(panel_path, encoding="utf-8") as handle:
        panel = json.load(handle)
    validate_seal(panel, label="R0123 crossed-representation density panel")
    arrays_signature = _exact_signature(
        panel["arrays"], label="R0123 crossed-representation radii"
    )
    cells = panel.get("cells")
    if (
        panel.get("schema") != PANEL_SCHEMA
        or panel.get("round_id") != ROUND_ID
        or panel.get("release_sha") != active["manifest"]["release_sha"]
        or panel.get("cell_order") != list(CELL_ORDER)
        or not isinstance(cells, Mapping)
        or set(cells) != set(CELL_ORDER)
        or (panel.get("scorer") or {}).get(
            "historical_absolute_floor_applied"
        )
        is not False
        or panel.get("training_performed") is not False
    ):
        raise Round0123Error("R0123 crossed panel contract changed")

    expected_fields = {
        "anchor_compact_ids",
        "legacy_input__high_radius",
        "fresh_input__high_radius",
        *{f"{key}__low_radius" for key in CELL_ORDER},
    }
    with np.load(
        arrays_signature["canonical_path"], allow_pickle=False
    ) as archive:
        if set(archive.files) != expected_fields:
            raise Round0123Error("R0123 crossed radius arrays changed")
        anchors = np.asarray(archive["anchor_compact_ids"], dtype=np.int64)
        legacy_high = np.asarray(
            archive["legacy_input__high_radius"], dtype=np.float64
        )
        fresh_high = np.asarray(
            archive["fresh_input__high_radius"], dtype=np.float64
        )
        low = {
            key: np.asarray(
                archive[f"{key}__low_radius"], dtype=np.float64
            )
            for key in CELL_ORDER
        }
    common = panel.get("common_population") or {}
    if (
        anchors.shape != (ANCHORS,)
        or ordered_array_sha256(anchors)
        != common.get("anchor_compact_ids_sha256")
        or legacy_high.shape != anchors.shape
        or fresh_high.shape != anchors.shape
        or any(value.shape != anchors.shape for value in low.values())
    ):
        raise Round0123Error("R0123 common anchor alignment changed")
    for key in CELL_ORDER:
        high = (
            legacy_high
            if key.endswith("__legacy_input")
            else fresh_high
        )
        if (
            _pearson_log_radius(high, low[key])
            != cells[key].get("density_correlation")
            or ordered_array_sha256(low[key])
            != cells[key].get("low_radius_sha256")
            or cells[key].get("historical_absolute_floor_applied") is not False
        ):
            raise Round0123Error(f"{key} score/radius binding changed")

    summary, samples = _paired_alignment_contrasts(
        legacy_high=legacy_high,
        fresh_high=fresh_high,
        low_radii=low,
    )
    sample_path = os.path.join(output, "paired-alignment-bootstrap.npz")
    atomic_save_new_npz(sample_path, immutable=True, **samples)
    outcome = _classify_alignment(summary["contrasts"])
    receipt = seal({
        "schema": DECISION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "panel": expected_input_signature(panel_path),
        "paired_bootstrap": expected_input_signature(sample_path),
        "outcome": outcome,
        "cell_correlations": summary["cell_correlations"],
        "alignment_contrasts": summary["contrasts"],
        "selector": {
            "rule": (
                "classify each map's matched-minus-crossed density "
                "correlation as positive only when its paired central-99% "
                "bootstrap interval is strictly above zero"
            ),
            "common_exact_compact_ids": True,
            "common_exact_anchor_ids": True,
            "each_cell_uses_its_input_representation_high_d_reference": True,
            "historical_absolute_floor_applied": False,
        },
        "interpretation": (
            "This classifies map-input representation alignment on the "
            "shared R0113 FineWeb compact population. It does not rank native "
            "map quality across distinct high-D spaces and does not isolate "
            "which training-bundle factor created any alignment."
        ),
        "native_quality_claimed": False,
        "single_factor_cause_localized": False,
        "production_transfer_claimed": False,
        "map_registry_state_changed": False,
        "training_performed": False,
    })
    path = os.path.join(
        output, "crossed-representation-alignment-decision.json"
    )
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if job is None:
        job = active["job"]
    if (active.get("manifest") or {}).get("round_id") != ROUND_ID:
        raise Round0123Error("R0123 handler received another round")
    action = str(job.get("action"))
    if action == "score_crossed_representation_density":
        return run_score(active, job)
    if action == "decide_crossed_representation_alignment":
        return run_decision(active, job)
    raise Round0123Error(f"unknown R0123 action: {action}")
