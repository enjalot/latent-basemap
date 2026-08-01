"""Single CPU-heavy analysis node for R0146."""
from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from basemap.artifact_identity import expected_input_signature, ordered_array_sha256
from basemap.output_safety import (
    atomic_save_new_npz,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0142_jina_universality import (
    MAP_ORDER,
    PROBE_ORDER,
    validate_seal as validate_r0142_seal,
)
from basemap.round0146_projection_predictors import (
    BLAS_THREADS,
    CAPABILITY,
    DIMENSION,
    PREDICTOR_ORDER,
    QUERY_ID_OFFSET,
    ROUND_ID,
    TRAINING_SUPPORT_ROWS,
    Round0146Error,
    correlation_table,
    geometry_predictors,
    seal,
    stable_seed,
    support_distance_predictor,
    systematic_positions,
)


def _signature(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    value = expected_input_signature(str(expected.get("canonical_path") or ""))
    if value != dict(expected):
        raise Round0146Error(f"{label} bytes changed")
    return value


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0146Error(f"JSON object required: {path}")
    return value


def _read_r0142_panel(signature: Mapping[str, Any], *, map_key: str) -> dict[str, Any]:
    bound = _signature(signature, label=f"R0142 {map_key} panel")
    panel = _read_json(bound["canonical_path"])
    validate_r0142_seal(panel, label=f"R0142 {map_key} panel")
    if (
        panel.get("schema") != "round0142-jina-universality-map-panel-v1"
        or panel.get("round_id") != "0142"
        or panel.get("map_key") != map_key
        or panel.get("probe_order") != list(PROBE_ORDER)
        or panel.get("training_performed") is not False
    ):
        raise Round0146Error(f"R0142 {map_key} panel semantics changed")
    return panel


def _load_support(
    map_key: str, spec: Mapping[str, Any]
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    mapping_signature = _signature(spec["mapping"], label=f"{map_key} mapping")
    int8_signature = _signature(spec["int8"], label="R0103 int8 payload")
    scales_signature = _signature(spec["scales"], label="R0103 scale payload")
    mapping = np.load(
        mapping_signature["canonical_path"], mmap_mode="r", allow_pickle=False
    )
    expected_mapping_rows = int(spec["mapping_rows"])
    if (
        mapping.shape != (expected_mapping_rows,)
        or mapping.dtype != np.int64
        or int(mapping[0]) < 0
        or int(mapping[-1]) >= 25_000_000
        or np.any(mapping[1:] <= mapping[:-1])
    ):
        raise Round0146Error(f"{map_key} train mapping changed")
    positions = systematic_positions(
        len(mapping),
        TRAINING_SUPPORT_ROWS,
        seed=stable_seed(map_key, "training-support"),
    )
    global_rows = np.asarray(mapping[positions], dtype=np.int64)
    encoded = np.memmap(
        int8_signature["canonical_path"],
        dtype=np.int8,
        mode="r",
        shape=(25_000_000, DIMENSION),
    )
    scales = np.memmap(
        scales_signature["canonical_path"],
        dtype="<f2",
        mode="r",
        shape=(25_000_000,),
    )
    values = np.asarray(encoded[global_rows], dtype=np.float32)
    values *= np.asarray(scales[global_rows], dtype=np.float32)[:, None]
    if values.shape != (TRAINING_SUPPORT_ROWS, DIMENSION):
        raise Round0146Error(f"{map_key} support materialization failed")
    return values, {
        "policy": "one seeded systematic row per equal-width compact-train-order stratum",
        "seed": stable_seed(map_key, "training-support"),
        "mapping": mapping_signature,
        "mapping_rows": expected_mapping_rows,
        "sample_rows": TRAINING_SUPPORT_ROWS,
        "compact_positions_sha256": ordered_array_sha256(positions),
        "global_rows_sha256": ordered_array_sha256(global_rows),
        "int8": int8_signature,
        "scales": scales_signature,
    }, positions, global_rows


def _coordinate_ids(
    cell: Mapping[str, Any], *, label: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    probe_report = dict(cell.get("probe") or {})
    coordinate_signature = _signature(
        probe_report.get("coordinates") or {}, label=f"{label} coordinates"
    )
    with np.load(coordinate_signature["canonical_path"], allow_pickle=False) as archive:
        corpus_ids = np.asarray(archive["probe_corpus_ids"], dtype=np.int64)
        query_ids = np.asarray(archive["probe_query_ids"], dtype=np.int64)
        corpus_coordinates = np.asarray(archive["probe_corpus_coords"])
        query_coordinates = np.asarray(archive["probe_query_coords"])
    selection = dict(probe_report.get("selection") or {})
    if (
        corpus_ids.ndim != 1
        or query_ids.ndim != 1
        or corpus_coordinates.shape != (len(corpus_ids), 2)
        or query_coordinates.shape != (len(query_ids), 2)
        or not np.isfinite(corpus_coordinates).all()
        or not np.isfinite(query_coordinates).all()
        or np.any(corpus_ids < 0)
        or np.any(query_ids < QUERY_ID_OFFSET)
        or (selection.get("corpus") or {}).get("ordered_sha256")
        != ordered_array_sha256(corpus_ids)
        or (selection.get("queries") or {}).get("ordered_sha256")
        != ordered_array_sha256(query_ids)
    ):
        raise Round0146Error(f"{label} coordinate-row identity changed")
    return corpus_ids, query_ids - QUERY_ID_OFFSET, coordinate_signature


def _load_probe_values(
    panels: Mapping[str, Mapping[str, Any]], probe: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    first_cell = panels[MAP_ORDER[0]]["probes"][probe]
    corpus_ids, query_rows, first_coordinate = _coordinate_ids(
        first_cell, label=f"{MAP_ORDER[0]} {probe}"
    )
    second_cell = panels[MAP_ORDER[1]]["probes"][probe]
    second_corpus_ids, second_query_rows, second_coordinate = _coordinate_ids(
        second_cell, label=f"{MAP_ORDER[1]} {probe}"
    )
    if not (
        np.array_equal(corpus_ids, second_corpus_ids)
        and np.array_equal(query_rows, second_query_rows)
    ):
        raise Round0146Error(f"{probe} row selection differs between maps")

    first_inputs = dict(first_cell["probe"].get("inputs") or {})
    second_inputs = dict(second_cell["probe"].get("inputs") or {})
    if first_inputs != second_inputs:
        raise Round0146Error(f"{probe} high-dimensional inputs differ between maps")
    if "embeddings" in first_inputs:
        source_signature = _signature(
            first_inputs["embeddings"], label=f"{probe} embeddings"
        )
        source = np.load(
            source_signature["canonical_path"], mmap_mode="r", allow_pickle=False
        )
        if source.ndim != 2 or source.shape[1] != DIMENSION:
            raise Round0146Error(f"{probe} source geometry changed")
        if int(corpus_ids.max()) >= len(source) or int(query_rows.max()) >= len(source):
            raise Round0146Error(f"{probe} selected row is out of bounds")
        corpus = np.asarray(source[corpus_ids])
        queries = np.asarray(source[query_rows])
        source_inputs = {"embeddings": source_signature}
    else:
        corpus_signature = _signature(
            first_inputs.get("corpus_embeddings") or {},
            label=f"{probe} corpus embeddings",
        )
        query_signature = _signature(
            first_inputs.get("query_embeddings") or {},
            label=f"{probe} query embeddings",
        )
        corpus_source = np.load(
            corpus_signature["canonical_path"], mmap_mode="r", allow_pickle=False
        )
        query_source = np.load(
            query_signature["canonical_path"], mmap_mode="r", allow_pickle=False
        )
        if (
            corpus_source.ndim != 2
            or query_source.ndim != 2
            or corpus_source.shape[1] != DIMENSION
            or query_source.shape[1] != DIMENSION
            or int(corpus_ids.max()) >= len(corpus_source)
            or int(query_rows.max()) >= len(query_source)
        ):
            raise Round0146Error(f"{probe} separate source geometry changed")
        corpus = np.asarray(corpus_source[corpus_ids])
        queries = np.asarray(query_source[query_rows])
        source_inputs = {
            "corpus_embeddings": corpus_signature,
            "query_embeddings": query_signature,
        }
    return corpus, queries, corpus_ids, query_rows, {
        **source_inputs,
        "coordinates": {
            MAP_ORDER[0]: first_coordinate,
            MAP_ORDER[1]: second_coordinate,
        },
        "corpus_row_ids_sha256": ordered_array_sha256(corpus_ids),
        "query_source_row_ids_sha256": ordered_array_sha256(query_rows),
    }


def run_predictors(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0146Error("R0146 handler received another queue")
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0146 projection-loss predictor analysis"
    )
    started = time.monotonic()
    threads = int(job.get("cpu_threads", BLAS_THREADS))
    if threads < 1 or threads > (os.cpu_count() or 1):
        raise Round0146Error("R0146 CPU thread registration is invalid")

    table_signature = _signature(job["retention_table"], label="R0142 retention table")
    table = _read_json(table_signature["canonical_path"])
    validate_r0142_seal(table, label="R0142 retention table")
    if (
        table.get("schema") != "jina-diverse-universality-panel-v1"
        or table.get("round_id") != "0142"
        or table.get("probe_order") != list(PROBE_ORDER)
    ):
        raise Round0146Error("R0142 retention-table semantics changed")
    panels = {
        map_key: _read_r0142_panel(job["panels"][map_key], map_key=map_key)
        for map_key in MAP_ORDER
    }
    if any(
        table["maps"].get(map_key) != dict(job["panels"][map_key])
        for map_key in MAP_ORDER
    ):
        raise Round0146Error("R0142 table-to-panel binding changed")

    support_values: dict[str, np.ndarray] = {}
    support_receipts: dict[str, dict[str, Any]] = {}
    support_arrays: dict[str, np.ndarray] = {}
    for map_key in MAP_ORDER:
        values, receipt, positions, global_rows = _load_support(
            map_key, job["training_support"][map_key]
        )
        support_values[map_key] = values
        support_receipts[map_key] = receipt
        support_arrays[f"{map_key}__compact_positions"] = positions
        support_arrays[f"{map_key}__global_rows"] = global_rows

    cells: list[dict[str, Any]] = []
    probes: dict[str, Any] = {}
    with threadpool_limits(limits=threads):
        for index, probe in enumerate(PROBE_ORDER):
            corpus, queries, corpus_ids, query_rows, inputs = _load_probe_values(
                panels, probe
            )
            geometry = geometry_predictors(
                corpus, source_row_ids=corpus_ids, label=probe
            )
            probe_maps: dict[str, Any] = {}
            for map_key in MAP_ORDER:
                support = support_distance_predictor(
                    queries,
                    support_values[map_key],
                    label=f"{map_key} {probe}",
                )
                metrics = panels[map_key]["probes"][probe]["metrics"]
                cell = {
                    "map": map_key,
                    "probe": probe,
                    "ffr_retention": float(metrics["ffr_retention"]),
                    "recall10_retention": (
                        float(metrics["recall10_retention"])
                        if metrics.get("recall10_retention") is not None
                        else None
                    ),
                    "twonn_intrinsic_dimension": float(
                        geometry["twonn"]["intrinsic_dimension"]
                    ),
                    "hubness_k10_skew": float(geometry["hubness"]["skew"]),
                    "anisotropy_eigen_ratio": float(
                        geometry["anisotropy"]["eigen_ratio"]
                    ),
                    "support_cosine_distance_p50": float(support["p50"]),
                    "support_cosine_distance_p90": float(support["p90"]),
                }
                cells.append(cell)
                probe_maps[map_key] = {
                    "outcomes": {
                        "ffr_retention": cell["ffr_retention"],
                        "recall10_retention": cell["recall10_retention"],
                    },
                    "distance_to_training_support": support,
                }
            probes[probe] = {
                "inputs": inputs,
                "corpus_rows": int(len(corpus)),
                "query_rows": int(len(queries)),
                "query_source_row_ids_sha256": ordered_array_sha256(query_rows),
                "geometry": geometry,
                "maps": probe_maps,
            }
            print(
                f"R0146 probe {index + 1}/{len(PROBE_ORDER)} {probe}: "
                f"TwoNN={geometry['twonn']['intrinsic_dimension']:.3f} "
                f"hub-skew={geometry['hubness']['skew']:.3f}",
                flush=True,
            )
            del corpus, queries

        correlations = correlation_table(cells)

    support_path = os.path.join(output, "training-support-rows.npz")
    atomic_save_new_npz(support_path, immutable=True, **support_arrays)
    report = seal({
        "schema": CAPABILITY,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capability": CAPABILITY,
        "r0142_retention_table": table_signature,
        "r0142_panels": {
            map_key: dict(job["panels"][map_key]) for map_key in MAP_ORDER
        },
        "predictor_order": list(PREDICTOR_ORDER),
        "primary_outcome": "ffr_retention",
        "secondary_outcome": "recall10_retention",
        "cells": cells,
        "correlations": correlations,
        "probes": probes,
        "training_support": support_receipts,
        "training_support_rows": expected_input_signature(support_path),
        "cpu_threads": threads,
        "interpretation": (
            "exploratory rank correlations across eleven heterogeneous probes; "
            "pooled rows are descriptive because probe geometry repeats across maps"
        ),
        "no_causal_predictor_claim": True,
        "no_universal_map_claim": True,
        "no_quality_gate_change": True,
        "training_performed": False,
        "gpu_used": False,
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(
        os.path.join(output, "projection-loss-predictors.json"),
        report,
        immutable=True,
    )


def run_job(active: dict[str, Any], job: dict[str, Any] | None = None) -> None:
    if job is None or job.get("action") != "projection_loss_predictors":
        raise Round0146Error("R0146 requires its exact predictor job")
    run_predictors(active, job)

