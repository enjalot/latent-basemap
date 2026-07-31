"""Execute the seed-43 replay of R0132's matched scale-policy bridge."""
from __future__ import annotations

import gc
import json
import math
import os
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
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
    validate_seal as validate_coordinate_seal,
)
from basemap.round0105_search import GROUPS, ROW_COUNT
from basemap.round0108_evaluation import (
    FAMILY_SIZE_CUTOFF,
    FRACTION,
    HELDOUT_CORPUS_ROWS,
    HELDOUT_QUERY_ROWS,
    IN_MIX_LANGUAGES,
    K_HIT,
    K_LOW_MAX,
    POLISH,
    TRANSFORM_BATCH_ROWS,
    TRANSFORM_CHUNK_ROWS,
    CompactInt8DequantizedArray,
    exact_split_duplicate_diagnostics,
    load_reviewed_model,
    projection_metrics,
)
from basemap.round0132_scale_bridge import (
    DENSITY_BOOTSTRAP_DRAWS,
    GRAPH_K,
    GRAPH_SCHEMA,
    HALF_RETAINED_ROWS,
    NATIVE_ANCHORS_PER_GROUP,
    NATIVE_SCHEMA as R0132_NATIVE_SCHEMA,
    N_NEIGHBORS,
    OOD_SCHEMA as R0132_OOD_SCHEMA,
    PIPELINE,
    PIPELINE_SCHEMA,
    POSITIVE_DESTINATION_POLICY,
    SAMPLER_CLASS,
    Round0132Error as R0132ContractError,
    noninferiority_checks,
    paired_density_bootstrap,
    recall50_at_least_recall10,
    seal,
    validate_seal,
)
from basemap.round0133_seed_replay import (
    DECISION_SCHEMA,
    NATIVE_SCHEMA,
    OOD_SCHEMA,
    PRODUCTION_CONFIG_SCHEMA,
    ROUND_ID,
    SEED,
    TRAIN_CONFIG_SCHEMA,
    TRAIN_RECEIPT_SCHEMA,
    TRANSFORM_RECEIPT_SCHEMA,
    Round0133Error,
    assert_no_r0110_coordinate_inputs,
    combine_seed_decisions,
    seed43_scale_policy_decision,
    validate_accepted_seed42_decision,
    validate_seed43_train_execution,
)
from experiments.round0107_nodes import run_train_contract
from experiments.round0108_nodes import _panel_config
from experiments.round0132_nodes import (
    FULL_GRAPH_SCHEMA,
    _authenticate_native_selector,
    _authenticate_ood_metrics,
    _load_json,
    _load_subset,
    _native_metrics,
    _selected_source,
    _signature,
    _SliceView,
)


R0109_TRAIN_RECEIPT_SCHEMA = "round0109-diverse-jina-train-receipt-v1"
R0109_PRODUCTION_CONFIG_SCHEMA = "round0109-production-config-v1"
U12_COORDINATE_DIR = "u12-seed43"
FULL_COORDINATE_DIR = "full25m-seed43-on-u12"


def _load_seed43_u12_bundle(job: Mapping[str, Any]) -> dict[str, Any]:
    graph = _signature(
        str(job["graph_manifest"]),
        str(job["graph_manifest_sha256"]),
        label="reviewed R0132 U12 graph",
    )
    return load_reviewed_model(
        train_output=str(job["train_output"]),
        graph_manifest_path=str(job["graph_manifest"]),
        graph_manifest_sha256=graph["sha256"],
        expected_train_round_id=ROUND_ID,
        expected_train_receipt_schema=TRAIN_RECEIPT_SCHEMA,
        expected_production_config_schema=PRODUCTION_CONFIG_SCHEMA,
        expected_seed=SEED,
        expected_graph_schema=GRAPH_SCHEMA,
    )


def _load_seed43_full_bundle(job: Mapping[str, Any]) -> dict[str, Any]:
    graph = _signature(
        str(job["full_graph_manifest"]),
        str(job["full_graph_manifest_sha256"]),
        label="reviewed R0106 full graph",
    )
    for key, label in (
        ("full_model", "accepted R0109 model"),
        ("full_production_config", "accepted R0109 production config"),
        ("full_train_receipt", "accepted R0109 train receipt"),
    ):
        _signature(
            str(job[key]), str(job[f"{key}_sha256"]), label=label
        )
    train_root = os.path.realpath(str(job["full_train_output"]))
    expected_members = {
        "full_model": os.path.join(train_root, "model.pt"),
        "full_production_config": os.path.join(
            train_root, "production-config.json"
        ),
        "full_train_receipt": os.path.join(train_root, "train-receipt.json"),
    }
    if any(
        os.path.realpath(str(job[key])) != expected_path
        for key, expected_path in expected_members.items()
    ):
        raise Round0133Error("accepted R0109 bundle paths disagree")
    return load_reviewed_model(
        train_output=str(job["full_train_output"]),
        graph_manifest_path=str(job["full_graph_manifest"]),
        graph_manifest_sha256=graph["sha256"],
        expected_train_round_id="0109",
        expected_train_receipt_schema=R0109_TRAIN_RECEIPT_SCHEMA,
        expected_production_config_schema=R0109_PRODUCTION_CONFIG_SCHEMA,
        expected_seed=SEED,
        expected_graph_schema=FULL_GRAPH_SCHEMA,
    )


def run_train(active: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    selected = dict(job)
    graph_path = str(job["graph_manifest"])
    graph_signature = _signature(
        graph_path,
        str(job["graph_manifest_sha256"]),
        label="reviewed R0132 U12 graph",
    )
    graph = _load_json(graph_path)
    validate_seal(graph, label="reviewed R0132 U12 graph")
    if (
        graph.get("schema") != GRAPH_SCHEMA
        or graph.get("round_id") != "0132"
        or graph.get("release_sha") != str(job["graph_release_sha"])
    ):
        raise Round0133Error("R0132 U12 graph lineage changed")
    selected["graph_manifest_sha256"] = graph_signature["sha256"]
    return run_train_contract(
        active,
        selected,
        round_id=ROUND_ID,
        seed=SEED,
        train_config_schema=TRAIN_CONFIG_SCHEMA,
        production_config_schema=PRODUCTION_CONFIG_SCHEMA,
        train_receipt_schema=TRAIN_RECEIPT_SCHEMA,
        output_label="R0133 U12 seed-43 train output",
        graph_load_kwargs={
            "expected_graph_schema": GRAPH_SCHEMA,
            "expected_graph_round_id": "0132",
            "expected_k_real": GRAPH_K,
            "expected_retained_rows": HALF_RETAINED_ROWS,
        },
        train_config_kwargs={
            "n_neighbors_including_self": N_NEIGHBORS,
            "compact_retained_rows": HALF_RETAINED_ROWS,
            "pipeline": PIPELINE,
            "pipeline_schema": PIPELINE_SCHEMA,
            "sampler_class": SAMPLER_CLASS,
            "positive_destination_policy": POSITIVE_DESTINATION_POLICY,
            "graph_degree": "variable-symmetric-fuzzy-k15-topology",
            "update_rule": "ceil(actual-R0132-directed-fuzzy-edges/409)",
        },
        training_input_kwargs={
            "positive_destination_policy": POSITIVE_DESTINATION_POLICY,
            "graph_degree": "variable-symmetric-fuzzy-k15-topology",
        },
    )


def _write_coordinate_stream(
    *,
    root: str,
    label: str,
    map_key: str,
    model_bundle: Mapping[str, Any],
    source: Any,
    release_sha: str,
    u12_graph_signature: Mapping[str, Any],
) -> dict[str, Any]:
    output = create_fresh_directory(root, label=label)
    started = time.monotonic()
    members: list[dict[str, Any]] = []
    for index, start in enumerate(range(0, HALF_RETAINED_ROWS, TRANSFORM_CHUNK_ROWS)):
        stop = min(start + TRANSFORM_CHUNK_ROWS, HALF_RETAINED_ROWS)
        chunk = create_fresh_directory(
            os.path.join(output, f"chunk-{index:05d}"),
            label=f"{label} coordinate chunk",
        )
        coordinates = np.asarray(
            model_bundle["model"].transform(
                _SliceView(source, start, stop), batch_size=TRANSFORM_BATCH_ROWS
            ),
            dtype=np.float32,
        )
        if coordinates.shape != (stop - start, 2) or not np.isfinite(
            coordinates
        ).all():
            raise Round0133Error(f"{label} emitted malformed coordinates")
        path = os.path.join(chunk, "coordinates.npy")
        atomic_save_new_npy(path, coordinates, immutable=True)
        signature = expected_input_signature(path)
        members.append({
            "chunk_index": index,
            "global_row_start": start,
            "global_row_stop": stop,
            "bytes": signature["bytes"],
            "sha256": signature["sha256"],
        })
        del coordinates
    receipt = coordinate_seal({
        "schema": TRANSFORM_SCHEMA,
        "round0133_schema": TRANSFORM_RECEIPT_SCHEMA,
        "round_id": ROUND_ID,
        "map_key": map_key,
        "model": model_bundle["train"]["model"],
        "train_receipt": model_bundle["train_signature"],
        "production_config": model_bundle["config_signature"],
        "model_training_graph": model_bundle["graph_signature"],
        "u12_scientific_universe_graph": dict(u12_graph_signature),
        "compact_mapping": dict(model_bundle["u12_mapping_signature"]),
        "substrate": source.substrate["signature"],
        "scientific_universe": "reviewed R0132 deterministic U12 rows",
        "input_preprocessing": (
            "signed-int8 times exact fp16 row scale to device fp32; no L2 "
            "renormalization before model"
        ),
        "row_accounting": {
            "all_rows": HALF_RETAINED_ROWS,
            "retained_representatives": HALF_RETAINED_ROWS,
            "original_rows": ROW_COUNT,
            "not_selected_or_excluded_rows": ROW_COUNT - HALF_RETAINED_ROWS,
        },
        "coordinate_stream": {
            "schema": COORDINATE_SCHEMA,
            "row_count": HALF_RETAINED_ROWS,
            "dimension": 2,
            "dtype": "<f4",
            "row_order": "reviewed R0132 U12 compact order",
            "ordered_chunks": members,
        },
        "inference": {
            "batch_rows": TRANSFORM_BATCH_ROWS,
            "chunk_rows": TRANSFORM_CHUNK_ROWS,
            "all_u12_rows_projected": True,
        },
        "release_sha": release_sha,
        "training_performed": False,
        "wall_seconds": time.monotonic() - started,
    })
    path = os.path.join(output, "actual-transform.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_transform(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any]:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0133 paired seed-43 U12 transforms"
    )
    graph_signature = _signature(
        str(job["graph_manifest"]),
        str(job["graph_manifest_sha256"]),
        label="reviewed R0132 U12 graph",
    )
    _authenticate_train(active, job)
    u12 = _load_seed43_u12_bundle(job)
    full = _load_seed43_full_bundle(job)
    source = CompactInt8DequantizedArray(u12["mapping"])
    if len(source) != HALF_RETAINED_ROWS:
        raise Round0133Error("R0133 transform source is not reviewed U12")
    mapping_signature = u12["graph"].get("compact_mapping")
    if not isinstance(mapping_signature, Mapping):
        raise Round0133Error("R0132 U12 mapping signature is missing")
    u12["u12_mapping_signature"] = dict(mapping_signature)
    full["u12_mapping_signature"] = dict(mapping_signature)
    u12_receipt = _write_coordinate_stream(
        root=os.path.join(output, U12_COORDINATE_DIR),
        label="R0133 U12 seed-43 transform",
        map_key="r0133-diverse-jina-u12-seed43",
        model_bundle=u12,
        source=source,
        release_sha=str(active["manifest"]["release_sha"]),
        u12_graph_signature=graph_signature,
    )
    full_receipt = _write_coordinate_stream(
        root=os.path.join(output, FULL_COORDINATE_DIR),
        label="R0133 accepted R0109 seed-43 transform on U12",
        map_key="r0109-diverse-jina-25m-seed43-on-r0132-u12",
        model_bundle=full,
        source=source,
        release_sha=str(active["manifest"]["release_sha"]),
        u12_graph_signature=graph_signature,
    )
    del u12["model"], full["model"], source
    gc.collect()
    return {
        "schema": TRANSFORM_RECEIPT_SCHEMA,
        "round_id": ROUND_ID,
        "u12": u12_receipt["receipt"],
        "full25m_on_u12": full_receipt["receipt"],
        "same_ordered_u12_source": True,
        "r0110_coordinates_used": False,
    }


def _accepted_r0132_native(job: Mapping[str, Any]) -> dict[str, Any]:
    path = str(job["r0132_native_receipt"])
    _signature(
        path,
        str(job["r0132_native_receipt_sha256"]),
        label="accepted R0132 native receipt",
    )
    native = _load_json(path)
    validate_seal(native, label="accepted R0132 native receipt")
    if native.get("schema") != R0132_NATIVE_SCHEMA or native.get("round_id") != "0132":
        raise Round0133Error("accepted R0132 native receipt changed")
    _authenticate_native_selector(native)
    return native


def run_score_native(
    _active: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any]:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0133 matched seed-43 U12 native panel"
    )
    started = time.monotonic()
    subset = _load_subset(str(job["subset_output"]))
    accepted = _accepted_r0132_native(job)
    arrays_spec = accepted["arrays"]
    _signature(
        str(arrays_spec["canonical_path"]),
        str(arrays_spec["sha256"]),
        label="accepted R0132 native arrays",
    )
    with np.load(str(arrays_spec["canonical_path"]), allow_pickle=False) as arrays:
        global_anchors = np.asarray(arrays["global_anchor_rows"], dtype=np.int64)
        compact_anchors = np.asarray(arrays["compact_anchor_rows"], dtype=np.int64)
        group_ids = np.asarray(arrays["group_ids"], dtype=np.uint8)
        high_neighbors = np.asarray(arrays["high_neighbors_top15"], dtype=np.int64)
        high_radius = np.asarray(arrays["high_radius"], dtype=np.float64)
        family_sizes = np.asarray(arrays["family_sizes"], dtype=np.int64)
    expected_rows = NATIVE_ANCHORS_PER_GROUP * len(GROUPS)
    if (
        global_anchors.shape != (expected_rows,)
        or compact_anchors.shape != global_anchors.shape
        or group_ids.shape != global_anchors.shape
        or high_neighbors.shape != (expected_rows, GRAPH_K)
        or high_radius.shape != global_anchors.shape
        or family_sizes.shape != global_anchors.shape
        or not np.array_equal(subset["mapping"][compact_anchors], global_anchors)
        or not np.array_equal(subset["group_ids"][compact_anchors], group_ids)
        or accepted.get("subset_manifest") != subset["manifest_signature"]
    ):
        raise Round0133Error("accepted R0132 native universe changed")

    transform_root = str(job["transform_output"])
    control_coordinates = CoordinateStream(
        os.path.join(transform_root, U12_COORDINATE_DIR)
    )
    treatment_coordinates = CoordinateStream(
        os.path.join(transform_root, FULL_COORDINATE_DIR)
    )
    if len(control_coordinates) != HALF_RETAINED_ROWS or len(
        treatment_coordinates
    ) != HALF_RETAINED_ROWS:
        raise Round0133Error("R0133 native coordinates do not cover U12")
    high10 = high_neighbors[:, :K_HIT]
    config = _panel_config(anchors=expected_rows)
    (
        control_metrics,
        control_low_radius,
        control_low50,
        control_ffr_hits,
        control_guard,
    ) = _native_metrics(
        coordinates=control_coordinates,
        anchors=compact_anchors,
        high10=high10,
        config=config,
    )
    (
        treatment_metrics,
        treatment_low_radius,
        treatment_low50,
        treatment_ffr_hits,
        treatment_guard,
    ) = _native_metrics(
        coordinates=treatment_coordinates,
        anchors=compact_anchors,
        high10=high10,
        config=config,
    )
    density = paired_density_bootstrap(
        high_radius,
        control_low_radius,
        treatment_low_radius,
        eligible=family_sizes < FAMILY_SIZE_CUTOFF,
    )
    deltas = np.asarray(density.pop("bootstrap_deltas"), dtype=np.float64)
    control_min = np.asarray(control_coordinates.min(axis=0), dtype=np.float64)
    control_max = np.asarray(control_coordinates.max(axis=0), dtype=np.float64)
    treatment_min = np.asarray(treatment_coordinates.min(axis=0), dtype=np.float64)
    treatment_max = np.asarray(treatment_coordinates.max(axis=0), dtype=np.float64)
    control_finite = bool(
        np.isfinite(control_min).all()
        and np.isfinite(control_max).all()
        and np.all(control_max - control_min > 1e-6)
    )
    treatment_finite = bool(
        np.isfinite(treatment_min).all()
        and np.isfinite(treatment_max).all()
        and np.all(treatment_max - treatment_min > 1e-6)
    )
    arrays_path = os.path.join(output, "matched-native-arrays.npz")
    atomic_save_new_npz(
        arrays_path,
        immutable=True,
        global_anchor_rows=global_anchors,
        compact_anchor_rows=compact_anchors,
        group_ids=group_ids,
        high_neighbors_top15=high_neighbors,
        high_radius=high_radius,
        control_low_radius=control_low_radius,
        treatment_low_radius=treatment_low_radius,
        control_low_neighbors_top50=control_low50,
        treatment_low_neighbors_top50=treatment_low50,
        native_fraction_k=np.asarray(
            max(K_LOW_MAX, int(math.ceil(FRACTION * HALF_RETAINED_ROWS))),
            dtype=np.int64,
        ),
        control_ffr_truth_hits=control_ffr_hits,
        treatment_ffr_truth_hits=treatment_ffr_hits,
        family_sizes=family_sizes,
        density_bootstrap_deltas=deltas,
    )
    control_transform = _signature(
        os.path.join(transform_root, U12_COORDINATE_DIR, "actual-transform.json"),
        label="R0133 U12 seed-43 transform",
    )
    treatment_transform = _signature(
        os.path.join(transform_root, FULL_COORDINATE_DIR, "actual-transform.json"),
        label="R0133 full seed-43 transform on U12",
    )
    receipt = seal({
        "schema": NATIVE_SCHEMA,
        "round_id": ROUND_ID,
        "seed": SEED,
        "subset_manifest": subset["manifest_signature"],
        "accepted_r0132_native": _signature(
            str(job["r0132_native_receipt"]),
            str(job["r0132_native_receipt_sha256"]),
            label="accepted R0132 native receipt",
        ),
        "model_lineage": {
            "control_12p5m_transform": control_transform,
            "treatment_25m_transform": treatment_transform,
        },
        "control_12p5m": {**control_metrics, "finite_noncollapsed": control_finite},
        "treatment_25m_on_u12": {
            **treatment_metrics,
            "finite_noncollapsed": treatment_finite,
        },
        "density_selector": density,
        "stale_absolute_jina_floor": accepted["stale_absolute_jina_floor"],
        "native_global_ffr_role": "registered-noninferiority-gate",
        "ood_projection_ffr_role": "diagnostic-only",
        "truth": {
            "reused_byte_for_byte_from_accepted_r0132": True,
            "accepted_native_arrays": dict(arrays_spec),
            "anchor_seed": (accepted.get("truth") or {}).get("anchor_seed"),
            "anchors_per_group": NATIVE_ANCHORS_PER_GROUP,
        },
        "low_d_guards": {"control": control_guard, "treatment": treatment_guard},
        "arrays": expected_input_signature(arrays_path),
        "checks": {
            "same_reviewed_u12_rows": True,
            "same_reviewed_high_d_truth_anchors_and_radii": True,
            "same_reviewed_duplicate_family_policy": True,
            "control_finite_noncollapsed": control_finite,
            "treatment_finite_noncollapsed": treatment_finite,
            "control_recall50_at_least_recall10": (
                control_metrics["global_recall_at_50_of_high10"]
                >= control_metrics["global_recall_at_10"]
            ),
            "treatment_recall50_at_least_recall10": (
                treatment_metrics["global_recall_at_50_of_high10"]
                >= treatment_metrics["global_recall_at_10"]
            ),
            "paired_density_bootstrap_exactly_1000": (
                len(deltas) == DENSITY_BOOTSTRAP_DRAWS
            ),
            "stale_floor_diagnostic_only": True,
        },
        "training_performed": False,
        "wall_seconds": time.monotonic() - started,
    })
    path = os.path.join(output, "matched-native.json")
    atomic_write_new_json(path, receipt, immutable=True)
    del control_coordinates, treatment_coordinates
    gc.collect()
    return {**receipt, "receipt": expected_input_signature(path)}


def _accepted_r0132_ood(job: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    path = str(job["r0132_ood_receipt"])
    _signature(
        path,
        str(job["r0132_ood_receipt_sha256"]),
        label="accepted R0132 OOD receipt",
    )
    receipt = _load_json(path)
    validate_seal(receipt, label="accepted R0132 OOD receipt")
    if receipt.get("schema") != R0132_OOD_SCHEMA or receipt.get("round_id") != "0132":
        raise Round0133Error("accepted R0132 OOD receipt changed")
    _authenticate_ood_metrics(receipt)
    arrays_spec = receipt.get("arrays") or {}
    _signature(
        str(arrays_spec.get("canonical_path") or ""),
        str(arrays_spec.get("sha256") or ""),
        label="accepted R0132 OOD arrays",
    )
    with np.load(str(arrays_spec["canonical_path"]), allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    return receipt, arrays


def _authenticate_r0133_ood_metrics(ood: Mapping[str, Any]) -> dict[str, Any]:
    checks = ood.get("checks") or {}
    if checks.get("truth_matches_accepted_r0132_for_every_probe") is not True:
        raise Round0133Error("R0133 OOD truth did not match accepted R0132")
    # R0132's authenticator owns the frozen arrays, recall arithmetic, and
    # four scientific validity checks.  Remove only R0133's additional lineage
    # check before invoking that unchanged math.
    normalized = dict(ood)
    normalized["checks"] = {
        key: checks.get(key)
        for key in (
            "same_queries_corpora_and_truth_for_both_models",
            "polish_absent_from_registered_training_inventory",
            "every_cell_recall50_at_least_recall10",
            "all_probe_coordinates_finite_noncollapsed",
        )
    }
    try:
        authenticated = _authenticate_ood_metrics(normalized)
    except R0132ContractError as exc:
        raise Round0133Error("R0133 OOD metrics failed R0132 authentication") from exc
    return {
        **authenticated,
        "truth_matches_accepted_r0132_for_every_probe": True,
    }


def _record_probe(
    *,
    report: dict[str, Any],
    prefix: str,
    accepted_arrays: Mapping[str, np.ndarray],
    output_arrays: dict[str, np.ndarray],
) -> bool:
    values = report.pop("arrays")
    truth = np.asarray(values["exact_high_d_top10"], dtype=np.int64)
    expected = np.asarray(
        accepted_arrays[f"{prefix}__exact_high_d_top10"], dtype=np.int64
    )
    if not np.array_equal(truth, expected):
        raise Round0133Error(f"{report['name']} high-D truth differs from R0132")
    for key, value in values.items():
        output_arrays[f"{prefix}__{key}"] = value
    return True


def _matched_probe_with_reviewed_truth(
    *,
    name: str,
    corpus: np.ndarray,
    queries: np.ndarray,
    control_model: Any,
    treatment_model: Any,
    duplicate_policy: str,
    accepted_probe: Mapping[str, Any],
    accepted_truth: np.ndarray,
) -> dict[str, Any]:
    """Score new coordinates against the exact accepted R0132 truth.

    R0132 already computed and sealed the exact high-dimensional neighbors for
    these byte-bound selectors and sources.  Reusing them avoids another CUDA
    exact-cosine pass and removes an otherwise pointless top-k tie-order
    comparison while leaving all low-dimensional panel math unchanged.
    """
    from basemap.panel_v2 import cross_knn

    duplicate = exact_split_duplicate_diagnostics(corpus, queries)
    if (
        duplicate_policy == "require-disjoint"
        and not duplicate["corpus_query_exact_family_disjoint"]
    ):
        raise Round0133Error(f"{name} exact family crosses corpus/query split")
    truth = np.asarray(accepted_truth, dtype=np.int64)
    if (
        truth.shape != (len(queries), K_HIT)
        or np.any(truth < 0)
        or np.any(truth >= len(corpus))
        or np.any(np.diff(np.sort(truth, axis=1), axis=1) == 0)
        or accepted_probe.get("name") != name
        or accepted_probe.get("corpus_rows") != len(corpus)
        or accepted_probe.get("query_rows") != len(queries)
        or accepted_probe.get("duplicate_control") != duplicate
        or not isinstance(accepted_probe.get("truth_guard"), Mapping)
    ):
        raise Round0133Error(f"{name} accepted R0132 truth lineage changed")

    config = _panel_config(anchors=len(queries))
    fraction_k = max(K_LOW_MAX, int(math.ceil(FRACTION * len(corpus))))
    cells: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {"exact_high_d_top10": truth}
    for label, model in (
        ("control_12p5m", control_model),
        ("treatment_25m", treatment_model),
    ):
        corpus_coordinates = np.asarray(
            model.transform(corpus, batch_size=TRANSFORM_BATCH_ROWS),
            dtype=np.float32,
        )
        query_coordinates = np.asarray(
            model.transform(queries, batch_size=TRANSFORM_BATCH_ROWS),
            dtype=np.float32,
        )
        low = cross_knn(
            query_coordinates,
            corpus_coordinates,
            fraction_k,
            config,
            hi_dim=False,
            exact=True,
        )
        metrics = projection_metrics(truth, low, fraction_k=fraction_k)
        cells[label] = {
            "recall_at_10": float(metrics["recall_at_10"]),
            "recall_at_50_of_high10": float(
                metrics["recall_at_50_of_high10"]
            ),
            "projection_ffr": float(metrics["ffr_diagnostic"]),
            "projection_ffr_role": "diagnostic-only",
            "finite_noncollapsed": bool(
                np.isfinite(corpus_coordinates).all()
                and np.isfinite(query_coordinates).all()
                and np.all(corpus_coordinates.std(axis=0) > 1e-8)
            ),
        }
        arrays[f"{label}_query_coordinates"] = query_coordinates
        arrays[f"{label}_low_neighbors_top50"] = np.asarray(
            low[:, :K_LOW_MAX], dtype=np.int64
        )
    return {
        "name": name,
        "corpus_rows": len(corpus),
        "query_rows": len(queries),
        "truth_computed_once_for_both_models": True,
        "truth_reused_byte_for_byte_from_accepted_r0132": True,
        "truth_guard": dict(accepted_probe["truth_guard"]),
        "duplicate_control": duplicate,
        "cells": cells,
        "arrays": arrays,
    }


def run_score_ood(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any]:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0133 matched seed-43 held-out OOD panel"
    )
    started = time.monotonic()
    _authenticate_train(active, job)
    control = _load_seed43_u12_bundle(job)
    treatment = _load_seed43_full_bundle(job)
    accepted, accepted_arrays = _accepted_r0132_ood(job)
    selection_signature = _signature(
        str(job["selection"]),
        str(job["selection_sha256"]),
        label="reviewed R0132/R0108 selectors",
    )
    if accepted.get("selection") != selection_signature:
        raise Round0133Error("accepted R0132 OOD selection lineage changed")
    probe_receipts: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    truth_matches: dict[str, bool] = {}
    with np.load(str(job["selection"]), allow_pickle=False) as selected:
        for language in (*IN_MIX_LANGUAGES, POLISH):
            corpus_rows = np.asarray(selected[f"{language}__corpus"], dtype=np.int64)
            query_rows = np.asarray(selected[f"{language}__queries"], dtype=np.int64)
            if corpus_rows.shape != (HELDOUT_CORPUS_ROWS,) or query_rows.shape != (
                HELDOUT_QUERY_ROWS,
            ):
                raise Round0133Error(f"{language} reviewed selector changed")
            corpus, queries = _selected_source(
                job["language_sources"][language],
                corpus_rows,
                query_rows,
                label=f"reviewed R0132 {language}",
            )
            report = _matched_probe_with_reviewed_truth(
                name=language,
                corpus=corpus,
                queries=queries,
                control_model=control["model"],
                treatment_model=treatment["model"],
                duplicate_policy="require-disjoint",
                accepted_probe=accepted["probes"][language],
                accepted_truth=accepted_arrays[
                    f"{language}__exact_high_d_top10"
                ],
            )
            truth_matches[language] = _record_probe(
                report=report,
                prefix=language,
                accepted_arrays=accepted_arrays,
                output_arrays=arrays,
            )
            probe_receipts[language] = report
            del corpus, queries

        fineweb_rows = np.asarray(selected["fineweb__corpus"], dtype=np.int64)
        fineweb_queries = np.asarray(selected["fineweb__queries"], dtype=np.int64)
        corpus, queries = _selected_source(
            job["diagnostic_sources"]["fineweb"],
            fineweb_rows,
            fineweb_queries,
            label="reviewed R0132 held-out FineWeb",
        )
        fineweb = _matched_probe_with_reviewed_truth(
            name="fineweb-heldout",
            corpus=corpus,
            queries=queries,
            control_model=control["model"],
            treatment_model=treatment["model"],
            duplicate_policy="diagnostic",
            accepted_probe=accepted["probes"]["fineweb-heldout"],
            accepted_truth=accepted_arrays["fineweb__exact_high_d_top10"],
        )
        truth_matches["fineweb-heldout"] = _record_probe(
            report=fineweb,
            prefix="fineweb",
            accepted_arrays=accepted_arrays,
            output_arrays=arrays,
        )
        probe_receipts["fineweb-heldout"] = fineweb

        dad_rows = np.asarray(selected["dadabase__corpus"], dtype=np.int64)
        dad_queries = np.asarray(selected["dadabase__queries"], dtype=np.int64)
        dad_corpus, dad_query = _selected_source(
            job["diagnostic_sources"]["dadabase"],
            dad_rows,
            dad_queries,
            label="reviewed R0132 Dadabase",
        )
        dadabase = _matched_probe_with_reviewed_truth(
            name="dadabase",
            corpus=dad_corpus,
            queries=dad_query,
            control_model=control["model"],
            treatment_model=treatment["model"],
            duplicate_policy="diagnostic",
            accepted_probe=accepted["probes"]["dadabase"],
            accepted_truth=accepted_arrays["dadabase__exact_high_d_top10"],
        )
        truth_matches["dadabase"] = _record_probe(
            report=dadabase,
            prefix="dadabase",
            accepted_arrays=accepted_arrays,
            output_arrays=arrays,
        )
        probe_receipts["dadabase"] = dadabase

    trec_corpus = job["diagnostic_sources"]["trec_corpus"]
    trec_queries = job["diagnostic_sources"]["trec_queries"]
    _signature(
        trec_corpus["canonical_path"], trec_corpus["sha256"], label="R0132 TREC corpus"
    )
    _signature(
        trec_queries["canonical_path"], trec_queries["sha256"], label="R0132 TREC queries"
    )
    trec = _matched_probe_with_reviewed_truth(
        name="trec-covid",
        corpus=np.load(trec_corpus["canonical_path"], mmap_mode="r", allow_pickle=False),
        queries=np.load(trec_queries["canonical_path"], mmap_mode="r", allow_pickle=False),
        control_model=control["model"],
        treatment_model=treatment["model"],
        duplicate_policy="diagnostic",
        accepted_probe=accepted["probes"]["trec-covid"],
        accepted_truth=accepted_arrays["trec__exact_high_d_top10"],
    )
    truth_matches["trec-covid"] = _record_probe(
        report=trec,
        prefix="trec",
        accepted_arrays=accepted_arrays,
        output_arrays=arrays,
    )
    probe_receipts["trec-covid"] = trec

    in_mix_control = np.asarray([
        probe_receipts[language]["cells"]["control_12p5m"][
            "recall_at_50_of_high10"
        ]
        for language in IN_MIX_LANGUAGES
    ])
    in_mix_treatment = np.asarray([
        probe_receipts[language]["cells"]["treatment_25m"][
            "recall_at_50_of_high10"
        ]
        for language in IN_MIX_LANGUAGES
    ])
    polish_control = probe_receipts[POLISH]["cells"]["control_12p5m"]
    polish_treatment = probe_receipts[POLISH]["cells"]["treatment_25m"]
    every_probe_cell = [
        cell for probe in probe_receipts.values() for cell in probe["cells"].values()
    ]
    control_summary = {
        "fineweb_recall_at_50_of_high10": probe_receipts["fineweb-heldout"]["cells"]["control_12p5m"]["recall_at_50_of_high10"],
        "polish_recall_at_50_of_high10": polish_control["recall_at_50_of_high10"],
        "in_mix_median_recall_at_50_of_high10": float(np.median(in_mix_control)),
    }
    treatment_summary = {
        "fineweb_recall_at_50_of_high10": probe_receipts["fineweb-heldout"]["cells"]["treatment_25m"]["recall_at_50_of_high10"],
        "polish_recall_at_50_of_high10": polish_treatment["recall_at_50_of_high10"],
        "in_mix_median_recall_at_50_of_high10": float(np.median(in_mix_treatment)),
    }
    arrays_path = os.path.join(output, "matched-ood-arrays.npz")
    atomic_save_new_npz(arrays_path, immutable=True, **arrays)
    receipt = seal({
        "schema": OOD_SCHEMA,
        "round_id": ROUND_ID,
        "seed": SEED,
        "selection": selection_signature,
        "accepted_r0132_ood": _signature(
            str(job["r0132_ood_receipt"]),
            str(job["r0132_ood_receipt_sha256"]),
            label="accepted R0132 OOD receipt",
        ),
        "model_lineage": {
            "control_12p5m_train_receipt": control["train_signature"],
            "control_12p5m_production_config": control["config_signature"],
            "control_12p5m_graph": control["graph_signature"],
            "treatment_25m_train_receipt": treatment["train_signature"],
            "treatment_25m_production_config": treatment["config_signature"],
            "treatment_25m_graph": treatment["graph_signature"],
        },
        "control_12p5m": control_summary,
        "treatment_25m": treatment_summary,
        "probes": probe_receipts,
        "checks": {
            "same_queries_corpora_and_truth_for_both_models": True,
            "truth_matches_accepted_r0132_for_every_probe": all(
                truth_matches.values()
            ),
            "polish_absent_from_registered_training_inventory": True,
            "every_cell_recall50_at_least_recall10": recall50_at_least_recall10(
                every_probe_cell
            ),
            "all_probe_coordinates_finite_noncollapsed": all(
                cell["finite_noncollapsed"] for cell in every_probe_cell
            ),
        },
        "roles": accepted["roles"],
        "arrays": expected_input_signature(arrays_path),
        "training_performed": False,
        "universal_ood_claimed": False,
        "wall_seconds": time.monotonic() - started,
    })
    path = os.path.join(output, "matched-ood.json")
    atomic_write_new_json(path, receipt, immutable=True)
    del control["model"], treatment["model"]
    gc.collect()
    return {**receipt, "receipt": expected_input_signature(path)}


def _authenticate_train(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any]:
    train_path = os.path.join(str(job["train_output"]), "train-receipt.json")
    config_path = os.path.join(str(job["train_output"]), "production-config.json")
    graph_path = str(job["graph_manifest"])
    accepted_config_path = str(job["r0132_production_config"])
    train = _load_json(train_path)
    config = _load_json(config_path)
    graph = _load_json(graph_path)
    accepted_config = _load_json(accepted_config_path)
    validate_seal(train, label="R0133 train receipt")
    validate_seal(graph, label="reviewed R0132 graph")
    graph_signature = _signature(
        graph_path,
        str(job["graph_manifest_sha256"]),
        label="reviewed R0132 graph",
    )
    if (
        train.get("release_sha") != active["manifest"]["release_sha"]
        or graph.get("release_sha") != str(job["graph_release_sha"])
        or train.get("graph_manifest") != graph_signature
        or (config.get("config") or {}).get("graph", {}).get("manifest")
        != graph_signature
    ):
        raise Round0133Error("R0133 train release/graph lineage changed")
    authenticated = validate_seed43_train_execution(
        train=train,
        config_receipt=config,
        graph=graph,
        accepted_r0132_config_receipt=accepted_config,
    )
    model = train.get("model") or {}
    model_signature = _signature(
        str(model.get("canonical_path") or ""),
        str(model.get("sha256") or ""),
        label="R0133 trained model",
    )
    return {
        **authenticated,
        "train_receipt": _signature(train_path, label="R0133 train receipt"),
        "production_config": _signature(config_path, label="R0133 production config"),
        "model": model_signature,
        "graph_manifest": graph_signature,
        "accepted_r0132_production_config": _signature(
            accepted_config_path,
            str(job["r0132_production_config_sha256"]),
            label="accepted R0132 production config",
        ),
    }


def _load_transform_receipt(
    path: str, *, label: str, release_sha: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    signature = _signature(path, label=label)
    receipt = _load_json(path)
    validate_coordinate_seal(receipt, label=label)
    stream = receipt.get("coordinate_stream") or {}
    if (
        receipt.get("schema") != TRANSFORM_SCHEMA
        or receipt.get("round0133_schema") != TRANSFORM_RECEIPT_SCHEMA
        or receipt.get("round_id") != ROUND_ID
        or receipt.get("release_sha") != release_sha
        or stream.get("schema") != COORDINATE_SCHEMA
        or stream.get("row_count") != HALF_RETAINED_ROWS
        or stream.get("dimension") != 2
    ):
        raise Round0133Error(f"{label} contract changed")
    return receipt, signature


def _accepted_r0109_lineage_signatures(
    job: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    train_root = os.path.realpath(str(job["full_train_output"]))
    paths = {
        "model": str(job["full_model"]),
        "production_config": str(job["full_production_config"]),
        "train_receipt": str(job["full_train_receipt"]),
    }
    expected_paths = {
        "model": os.path.join(train_root, "model.pt"),
        "production_config": os.path.join(train_root, "production-config.json"),
        "train_receipt": os.path.join(train_root, "train-receipt.json"),
    }
    if any(
        os.path.realpath(paths[key]) != expected_paths[key]
        for key in expected_paths
    ):
        raise Round0133Error("accepted R0109 bundle paths disagree")
    return {
        "model": _signature(
            paths["model"],
            str(job["full_model_sha256"]),
            label="accepted R0109 model",
        ),
        "production_config": _signature(
            paths["production_config"],
            str(job["full_production_config_sha256"]),
            label="accepted R0109 production config",
        ),
        "train_receipt": _signature(
            paths["train_receipt"],
            str(job["full_train_receipt_sha256"]),
            label="accepted R0109 train receipt",
        ),
        "graph_manifest": _signature(
            str(job["full_graph_manifest"]),
            str(job["full_graph_manifest_sha256"]),
            label="accepted R0109 graph",
        ),
    }


def _validate_decision_model_lineage(
    *,
    native_lineage: Mapping[str, Any],
    ood_lineage: Mapping[str, Any],
    control_transform: Mapping[str, Any],
    control_transform_signature: Mapping[str, Any],
    treatment_transform: Mapping[str, Any],
    treatment_transform_signature: Mapping[str, Any],
    authenticated_train: Mapping[str, Any],
    accepted_r0109: Mapping[str, Any],
) -> None:
    control_expected = {
        "model": authenticated_train["model"],
        "train_receipt": authenticated_train["train_receipt"],
        "production_config": authenticated_train["production_config"],
        "model_training_graph": authenticated_train["graph_manifest"],
        "u12_scientific_universe_graph": authenticated_train["graph_manifest"],
    }
    treatment_expected = {
        "model": accepted_r0109["model"],
        "train_receipt": accepted_r0109["train_receipt"],
        "production_config": accepted_r0109["production_config"],
        "model_training_graph": accepted_r0109["graph_manifest"],
        "u12_scientific_universe_graph": authenticated_train["graph_manifest"],
    }
    if (
        any(control_transform.get(key) != value for key, value in control_expected.items())
        or any(
            treatment_transform.get(key) != value
            for key, value in treatment_expected.items()
        )
        or control_transform.get("map_key") != "r0133-diverse-jina-u12-seed43"
        or treatment_transform.get("map_key")
        != "r0109-diverse-jina-25m-seed43-on-r0132-u12"
        or control_transform.get("compact_mapping")
        != treatment_transform.get("compact_mapping")
        or control_transform.get("substrate") != treatment_transform.get("substrate")
        or native_lineage.get("control_12p5m_transform")
        != control_transform_signature
        or native_lineage.get("treatment_25m_transform")
        != treatment_transform_signature
        or ood_lineage.get("control_12p5m_train_receipt")
        != authenticated_train["train_receipt"]
        or ood_lineage.get("control_12p5m_production_config")
        != authenticated_train["production_config"]
        or ood_lineage.get("control_12p5m_graph")
        != authenticated_train["graph_manifest"]
        or ood_lineage.get("treatment_25m_train_receipt")
        != accepted_r0109["train_receipt"]
        or ood_lineage.get("treatment_25m_production_config")
        != accepted_r0109["production_config"]
        or ood_lineage.get("treatment_25m_graph")
        != accepted_r0109["graph_manifest"]
    ):
        raise Round0133Error("R0133 panel/transform/model lineage disagrees")


def run_decision(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any]:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0133 two-seed scale-policy decision"
    )
    native_path = os.path.join(str(job["native_output"]), "matched-native.json")
    ood_path = os.path.join(str(job["ood_output"]), "matched-ood.json")
    native = _load_json(native_path)
    ood = _load_json(ood_path)
    validate_seal(native, label="R0133 native panel")
    validate_seal(ood, label="R0133 OOD panel")
    if (
        native.get("schema") != NATIVE_SCHEMA
        or native.get("round_id") != ROUND_ID
        or ood.get("schema") != OOD_SCHEMA
        or ood.get("round_id") != ROUND_ID
    ):
        raise Round0133Error("R0133 decision panel schema changed")
    authenticated_train = _authenticate_train(active, job)
    authenticated_native = _authenticate_native_selector(native)
    authenticated_ood = _authenticate_r0133_ood_metrics(ood)
    native_lineage = native.get("model_lineage") or {}
    ood_lineage = ood.get("model_lineage") or {}
    control_transform, control_transform_signature = _load_transform_receipt(
        os.path.join(
            str(job["transform_output"]),
            U12_COORDINATE_DIR,
            "actual-transform.json",
        ),
        label="R0133 U12 transform",
        release_sha=str(active["manifest"]["release_sha"]),
    )
    treatment_transform, treatment_transform_signature = _load_transform_receipt(
        os.path.join(
            str(job["transform_output"]),
            FULL_COORDINATE_DIR,
            "actual-transform.json",
        ),
        label="R0133 full transform on U12",
        release_sha=str(active["manifest"]["release_sha"]),
    )
    _validate_decision_model_lineage(
        native_lineage=native_lineage,
        ood_lineage=ood_lineage,
        control_transform=control_transform,
        control_transform_signature=control_transform_signature,
        treatment_transform=treatment_transform,
        treatment_transform_signature=treatment_transform_signature,
        authenticated_train=authenticated_train,
        accepted_r0109=_accepted_r0109_lineage_signatures(job),
    )
    quality = noninferiority_checks(
        control_native=native["control_12p5m"],
        treatment_native=native["treatment_25m_on_u12"],
        control_ood=ood["control_12p5m"],
        treatment_ood=ood["treatment_25m"],
    )
    validity = {
        "native_panel_checks_pass": all(native["checks"].values()),
        "ood_panel_checks_pass": all(ood["checks"].values()),
        "train_accounting_authenticated": all(
            authenticated_train["checks"].values()
        ),
        "only_registered_rng_identity_changed": (
            authenticated_train["only_registered_rng_identity_changed"] is True
        ),
        "same_reviewed_u12_and_ood_selectors": (
            authenticated_native["density_selector_recomputed"] is True
            and authenticated_ood["gating_summaries_recomputed"] is True
        ),
    }
    seed43 = seed43_scale_policy_decision(
        validity_checks=validity,
        density=native["density_selector"],
        quality=quality,
    )
    seed42_path = str(job["r0132_decision"])
    seed42_signature = _signature(
        seed42_path,
        str(job["r0132_decision_sha256"]),
        label="accepted R0132 seed-42 decision",
    )
    seed42 = _load_json(seed42_path)
    validate_seal(seed42, label="accepted R0132 seed-42 decision")
    validate_accepted_seed42_decision(seed42)
    combined = combine_seed_decisions(accepted_seed42=seed42, seed43=seed43)
    receipt = seal({
        "schema": DECISION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "accepted_seed42_decision": seed42_signature,
        "native_panel": _signature(native_path, label="R0133 native panel"),
        "ood_panel": _signature(ood_path, label="R0133 OOD panel"),
        "authenticated_train_execution": authenticated_train,
        "authenticated_native_selector": authenticated_native,
        "authenticated_ood_metrics": authenticated_ood,
        "seed42_outcome_carried_unchanged": seed42["outcome"],
        "seed43_decision": seed43,
        **combined,
        "training_performed": True,
        "r0110_coordinates_used": False,
        "map_registry_state_changed": False,
        "preferred_production_rung_claimed": False,
        "population_seed_robustness_claimed": False,
        "realized_graph_conditioned_draw_pairing_claimed": False,
        "production_ready": False,
    })
    path = os.path.join(output, "decision.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_job(
    active: dict[str, Any], job: dict[str, Any] | None = None
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID or job is None:
        raise Round0133Error("R0133 handler requires its exact round/job")
    assert_no_r0110_coordinate_inputs(job)
    handlers = {
        "train_u12_seed43": run_train,
        "transform_seed43_models_on_u12": run_transform,
        "score_matched_native_seed43": run_score_native,
        "score_matched_ood_seed43": run_score_ood,
        "decide_two_seed_scale_policy": run_decision,
    }
    action = str(job.get("action") or "")
    try:
        handler = handlers[action]
    except KeyError as exc:
        raise Round0133Error(f"unknown R0133 action: {action!r}") from exc
    return handler(active, job)
