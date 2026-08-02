"""Execute the conditional R0152 prefix/drop-only 12.5M rescue rung.

The graph, transform, matched-native, and matched-OOD mechanics are the
reviewed R0132 implementation under an exact R0152 contract.  Each queue node
runs in a fresh process; the explicit configuration below closes the inherited
implementation over R0152's different population cardinality and schemas.
"""
from __future__ import annotations

import gc
import json
import os
import time
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any

import numpy as np

from basemap import round0132_scale_bridge as r0132_contract
from basemap.artifact_identity import expected_input_signature, ordered_array_sha256
from basemap.output_safety import (
    atomic_save_new_npy,
    atomic_save_new_npz,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0105_search import GROUPS, ROW_COUNT
from basemap.round0107_training import SAMPLER_CLASS
from basemap.round0108_evaluation import TRANSFORM_BATCH_ROWS, seal, validate_seal
from basemap.round0151_scale_census import (
    EXPECTED_GROUP_IDS_ORDERED_SHA256,
    EXPECTED_MAPPING_ORDERED_SHA256,
)
from basemap.round0152_scale_rescue import (
    CAPABILITY,
    DECISION_SCHEMA,
    FUNCTIONAL_SCHEMA,
    GRAPH_K,
    GRAPH_PART_SCHEMA,
    GRAPH_SCHEMA,
    GRAPH_SHARD_SCHEMA,
    INDEX_SCHEMA,
    NATIVE_SCHEMA,
    N_NEIGHBORS,
    OOD_SCHEMA,
    PARENT_CAPABILITY,
    PIPELINE,
    PIPELINE_SCHEMA,
    POSITIVE_DESTINATION_POLICY,
    PRODUCTION_CONFIG_SCHEMA,
    QUALIFICATION_SCHEMA,
    RETAINED_ROWS,
    ROUND_ID,
    SEED,
    SUBSET_SCHEMA,
    TRAIN_CONFIG_SCHEMA,
    TRAIN_RECEIPT_SCHEMA,
    UPDATE_RULE,
    Round0152Error,
    build_decision,
    quality_selector,
    validate_train_execution,
)
from experiments import round0132_nodes as inherited
from experiments.round0107_nodes import run_train_contract
from experiments.round0106_nodes import GraphNodeContract


GRAPH_PART_NAMES = ("groups-a", "groups-b", "groups-c")


@contextmanager
def _configured_inherited_contract():
    """Bind R0132's reviewed mechanics to the exact R0152 contract."""
    values = {
        "ROUND_ID": ROUND_ID,
        "HALF_RETAINED_ROWS": RETAINED_ROWS,
        "SUBSET_SCHEMA": SUBSET_SCHEMA,
        "INDEX_SCHEMA": INDEX_SCHEMA,
        "QUALIFICATION_SCHEMA": QUALIFICATION_SCHEMA,
        "GRAPH_SHARD_SCHEMA": GRAPH_SHARD_SCHEMA,
        "GRAPH_PART_SCHEMA": GRAPH_PART_SCHEMA,
        "GRAPH_SCHEMA": GRAPH_SCHEMA,
        "TRAIN_CONFIG_SCHEMA": TRAIN_CONFIG_SCHEMA,
        "PRODUCTION_CONFIG_SCHEMA": PRODUCTION_CONFIG_SCHEMA,
        "TRAIN_RECEIPT_SCHEMA": TRAIN_RECEIPT_SCHEMA,
        "NATIVE_SCHEMA": NATIVE_SCHEMA,
        "OOD_SCHEMA": OOD_SCHEMA,
        "DECISION_SCHEMA": DECISION_SCHEMA,
        "PIPELINE": PIPELINE,
        "PIPELINE_SCHEMA": PIPELINE_SCHEMA,
        "SAMPLER_CLASS": SAMPLER_CLASS,
        "POSITIVE_DESTINATION_POLICY": POSITIVE_DESTINATION_POLICY,
    }
    old_contract = {name: getattr(r0132_contract, name) for name in values}
    old_nodes = {name: getattr(inherited, name) for name in values}
    old_graph_contract = inherited.GRAPH_CONTRACT
    old_validator = inherited.validate_train_execution
    try:
        for name, value in values.items():
            setattr(r0132_contract, name, value)
            setattr(inherited, name, value)
        inherited.GRAPH_CONTRACT = GraphNodeContract(
            round_id=ROUND_ID,
            k=GRAPH_K,
            n_neighbors=N_NEIGHBORS,
            shard_schema=GRAPH_SHARD_SCHEMA,
            part_schema=GRAPH_PART_SCHEMA,
            graph_schema=GRAPH_SCHEMA,
        )
        inherited.validate_train_execution = validate_train_execution
        yield
    finally:
        for name, value in old_contract.items():
            setattr(r0132_contract, name, value)
        for name, value in old_nodes.items():
            setattr(inherited, name, value)
        inherited.GRAPH_CONTRACT = old_graph_contract
        inherited.validate_train_execution = old_validator


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0152Error(f"JSON object required: {path}")
    return value


def _signature(path: str, expected: Mapping[str, Any] | None = None) -> dict[str, Any]:
    observed = expected_input_signature(path)
    if expected is not None and observed != dict(expected):
        raise Round0152Error(f"input bytes changed: {path}")
    return observed


def run_materialize_subset(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0152 authenticated prefix/drop subset"
    )
    started = time.monotonic()
    census_path = str(job["census"]["canonical_path"])
    _signature(census_path, job["census"])
    census = _read_json(census_path)
    validate_seal(census, label="accepted R0151 census")
    if (
        census.get("round_id") != "0151"
        or census.get("capability") != PARENT_CAPABILITY
        or census.get("retained_rows") != RETAINED_ROWS
        or census.get("replacement_rows") != 0
        or census.get("mapping") != dict(job["mapping"])
        or census.get("group_ids") != dict(job["group_ids"])
        or census.get("mapping_ordered_sha256")
        != EXPECTED_MAPPING_ORDERED_SHA256
        or census.get("group_ids_ordered_sha256")
        != EXPECTED_GROUP_IDS_ORDERED_SHA256
        or not all((census.get("checks") or {}).values())
    ):
        raise Round0152Error("R0151 census contract changed")
    mapping_path = str(job["mapping"]["canonical_path"])
    group_ids_path = str(job["group_ids"]["canonical_path"])
    _signature(mapping_path, job["mapping"])
    _signature(group_ids_path, job["group_ids"])
    mapping = np.asarray(
        np.load(mapping_path, mmap_mode="r", allow_pickle=False), dtype=np.int64
    )
    group_ids = np.asarray(
        np.load(group_ids_path, mmap_mode="r", allow_pickle=False), dtype=np.uint8
    )
    if (
        mapping.shape != (RETAINED_ROWS,)
        or group_ids.shape != mapping.shape
        or np.any(mapping[1:] <= mapping[:-1])
        or int(mapping[0]) < 0
        or int(mapping[-1]) >= ROW_COUNT
        or set(np.unique(group_ids).tolist()) != set(range(len(GROUPS)))
        or ordered_array_sha256(mapping) != EXPECTED_MAPPING_ORDERED_SHA256
        or ordered_array_sha256(group_ids) != EXPECTED_GROUP_IDS_ORDERED_SHA256
    ):
        raise Round0152Error("R0151 population arrays changed")
    quotas = {
        group: int(np.count_nonzero(group_ids == group_id))
        for group_id, group in enumerate(GROUPS)
    }
    registered = census.get("groups") or {}
    if (
        set(registered) != set(GROUPS)
        or any(quotas[group] != int(registered[group]["retained_rows"]) for group in GROUPS)
        or sum(quotas.values()) != RETAINED_ROWS
    ):
        raise Round0152Error("R0151 group census changed")
    keep = np.zeros(ROW_COUNT, dtype=bool)
    keep[mapping] = True
    excluded = np.flatnonzero(~keep).astype(np.int64, copy=False)
    if len(excluded) != ROW_COUNT - RETAINED_ROWS:
        raise Round0152Error("R0152 complement did not close")

    paths = {
        "mapping": os.path.join(output, "compact-to-global.i64.npy"),
        "group_ids": os.path.join(output, "compact-group-ids.u8.npy"),
        "excluded": os.path.join(output, "excluded-from-prefix-drop.i64.npy"),
    }
    atomic_save_new_npy(paths["mapping"], mapping, immutable=True)
    atomic_save_new_npy(paths["group_ids"], group_ids, immutable=True)
    atomic_save_new_npy(paths["excluded"], excluded, immutable=True)
    signatures = {name: _signature(path) for name, path in paths.items()}
    manifest = seal({
        "schema": SUBSET_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "r0151_census": dict(job["census"]),
        "selected_rows": RETAINED_ROWS,
        "full_raw_rows": ROW_COUNT,
        "selector": census["selector"],
        "quotas": quotas,
        "mapping": signatures["mapping"],
        "group_ids": signatures["group_ids"],
        "excluded": signatures["excluded"],
        "checks": {
            "accepted_r0151_population_reused_exactly": True,
            "mapping_strictly_increasing": True,
            "mapping_and_excluded_partition_25m": True,
            "every_group_present": True,
            "zero_replacements": True,
        },
        "training_performed": False,
        "optimizer_updates": 0,
        "map_outcomes_observed": False,
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(
        os.path.join(output, "subset-manifest.json"), manifest, immutable=True
    )


def run_train(active: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    selected = dict(job)
    graph_signature = inherited._signature(
        str(job["graph_manifest"]), label="R0152 assembled graph"
    )
    graph = inherited._load_json(str(job["graph_manifest"]))
    inherited.validate_seal(graph, label="R0152 assembled graph")
    if (
        graph.get("schema") != GRAPH_SCHEMA
        or graph.get("round_id") != ROUND_ID
        or graph.get("release_sha") != active["manifest"]["release_sha"]
    ):
        raise Round0152Error("R0152 assembled graph lineage changed")
    selected["graph_manifest_sha256"] = graph_signature["sha256"]
    selected["graph_release_sha"] = graph["release_sha"]
    return run_train_contract(
        active,
        selected,
        round_id=ROUND_ID,
        seed=SEED,
        train_config_schema=TRAIN_CONFIG_SCHEMA,
        production_config_schema=PRODUCTION_CONFIG_SCHEMA,
        train_receipt_schema=TRAIN_RECEIPT_SCHEMA,
        output_label="R0152 12.5M prefix/drop coverage-aligned train output",
        graph_load_kwargs={
            "expected_graph_schema": GRAPH_SCHEMA,
            "expected_graph_round_id": ROUND_ID,
            "expected_k_real": GRAPH_K,
            "expected_retained_rows": RETAINED_ROWS,
        },
        train_config_kwargs={
            "n_neighbors_including_self": N_NEIGHBORS,
            "compact_retained_rows": RETAINED_ROWS,
            "pipeline": PIPELINE,
            "pipeline_schema": PIPELINE_SCHEMA,
            "sampler_class": SAMPLER_CLASS,
            "positive_destination_policy": POSITIVE_DESTINATION_POLICY,
            "graph_degree": "variable-symmetric-fuzzy-k15-topology",
            "update_rule": UPDATE_RULE,
        },
        training_input_kwargs={
            "positive_destination_policy": POSITIVE_DESTINATION_POLICY,
            "graph_degree": "variable-symmetric-fuzzy-k15-topology",
        },
    )


def run_functional_density(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> None:
    from basemap.panel_v2 import score_panel
    from experiments import round0119_nodes as density_nodes
    from experiments import round0134_nodes as functional_nodes
    from experiments.round0027_nodes import _panel_config

    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0152 fixed functional and density panel"
    )
    started = time.monotonic()
    graph_signature = inherited._signature(
        str(job["graph_manifest"]), label="R0152 graph manifest"
    )
    bundle = inherited._load_model_bundle(
        train_output=str(job["train_output"]),
        graph_manifest=str(job["graph_manifest"]),
        graph_sha256=graph_signature["sha256"],
        half=True,
    )
    authenticated_train = inherited._authenticate_half_train(active, job)
    source_signature, source, queries = functional_nodes._load_shared_evaluation_inputs(job)
    shared, shared_signature, reference, truth, centroids = functional_nodes._load_reference(job)
    model = bundle["model"]
    coordinates = np.asarray(
        model.transform(source, batch_size=TRANSFORM_BATCH_ROWS), dtype=np.float32
    )
    query_coordinates = np.asarray(
        model.transform(queries, batch_size=TRANSFORM_BATCH_ROWS), dtype=np.float32
    )
    if (
        coordinates.shape != (2_000_000, 2)
        or query_coordinates.shape != (20_000, 2)
        or not np.isfinite(coordinates).all()
        or not np.isfinite(query_coordinates).all()
    ):
        raise Round0152Error("R0152 functional coordinates are malformed")
    coordinates_path = os.path.join(output, "functional-coordinates.npy")
    query_path = os.path.join(output, "functional-query-coordinates.npy")
    atomic_save_new_npy(coordinates_path, coordinates, immutable=True)
    atomic_save_new_npy(query_path, query_coordinates, immutable=True)
    panel = score_panel(
        source,
        coordinates,
        config=_panel_config(),
        centroids_by_k=centroids,
        hiD_reference=reference,
        scale_admission=None,
        provenance={
            "round_id": ROUND_ID,
            "cell": "prefix_drop_12p5m_seed42",
            "release_sha": active["manifest"]["release_sha"],
            "source": source_signature,
            "coordinates": _signature(coordinates_path),
            "shared_reference_receipt": shared_signature,
        },
    )
    projection = functional_nodes._projection_metrics(
        coordinates=coordinates,
        query_coordinates=query_coordinates,
        truth=truth,
    )
    if (
        panel.get("guards", {}).get("coords_finite") is not True
        or panel.get("guards", {}).get("coords_collapsed") is not False
        or panel.get("purity", {}).get("k256") is None
        or panel.get("purity", {}).get("k1024") is None
    ):
        raise Round0152Error("R0152 functional panel guards failed")

    (
        _density_source,
        representatives,
        retained_global_rows,
        anchors,
        global_rows,
        high_radius,
        density_lineage,
        density_reference,
    ) = density_nodes._load_universe(job)
    density_bundle = {
        "model": model,
        "group": "current_25m",
        "seed": SEED,
        "training_population": "R0151 12,485,206-row prefix/drop-only census",
        "training_graph": "R0152 induced k15 fuzzy graph",
        "training_dose": "coverage-aligned ceil(directed edges / 409)",
        "training_representation": "signed-int8-plus-exact-fp16-row-scale",
        "training_dequantization": "host-fp16-scale-to-device-fp32",
        "authenticated_training_semantics": authenticated_train["actual_pipeline_stamp"],
        "train": authenticated_train["train_receipt"],
        "production_config": authenticated_train["production_config"],
        "model_signature": dict(bundle["train"]["model"]),
    }
    density_cell, density_arrays = density_nodes._score_cell(
        key="prefix_drop_12p5m_seed42",
        bundle=density_bundle,
        source=_density_source,
        representatives=representatives,
        retained_global_rows=retained_global_rows,
        anchors=anchors,
        high_radius=high_radius,
        reference=density_reference,
    )
    density_value = float(density_cell["density_v2"]["correlation"])
    density_cell["clears_unchanged_registered_floor"] = (
        density_value >= density_lineage["registered_floor"]
    )
    density_arrays_path = os.path.join(output, "density-v2-arrays.npz")
    atomic_save_new_npz(
        density_arrays_path,
        immutable=True,
        anchor_compact_rows=anchors,
        anchor_global_rows=global_rows,
        high_radius=high_radius,
        **density_arrays,
    )
    receipt = seal({
        "schema": FUNCTIONAL_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "model_lineage": {
            "train_receipt": authenticated_train["train_receipt"],
            "production_config": authenticated_train["production_config"],
            "model": dict(bundle["train"]["model"]),
            "graph": authenticated_train["graph_manifest"],
        },
        "functional_universe": {
            "source": source_signature,
            "shared_reference": shared_signature,
            "high_d_reference": job["high_d_reference"],
            "query_truth": job["query_truth"],
            "query_embeddings": job["query_embeddings"],
        },
        "functional_cell": {
            "panel": panel,
            "projection": projection,
            "coordinates": _signature(coordinates_path),
            "query_coordinates": _signature(query_path),
        },
        "density_universe": density_lineage,
        "density_cell": density_cell,
        "density_arrays": _signature(density_arrays_path),
        "checks": {
            "fixed_r0037_functional_universe": True,
            "fixed_r0108_density_universe_and_floor": True,
            "functional_coordinates_finite_noncollapsed": True,
            "same_authenticated_model_for_both_panels": True,
        },
        "training_performed": False,
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(
        os.path.join(output, "functional-density.json"), receipt, immutable=True
    )
    del model, bundle["model"], coordinates, query_coordinates
    gc.collect()


def _authenticate_functional_density(value: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the density evidence and authenticate functional arrays."""
    from experiments.round0085_nodes import density_v2_calibration

    cell = value.get("functional_cell") or {}
    coordinates_spec = cell.get("coordinates") or {}
    queries_spec = cell.get("query_coordinates") or {}
    arrays_spec = value.get("density_arrays") or {}
    _signature(str(coordinates_spec.get("canonical_path") or ""), coordinates_spec)
    _signature(str(queries_spec.get("canonical_path") or ""), queries_spec)
    _signature(str(arrays_spec.get("canonical_path") or ""), arrays_spec)
    coordinates = np.load(
        str(coordinates_spec["canonical_path"]), mmap_mode="r", allow_pickle=False
    )
    queries = np.load(
        str(queries_spec["canonical_path"]), mmap_mode="r", allow_pickle=False
    )
    if (
        coordinates.shape != (2_000_000, 2)
        or queries.shape != (20_000, 2)
        or coordinates.dtype != np.float32
        or queries.dtype != np.float32
        or not np.isfinite(coordinates).all()
        or not np.isfinite(queries).all()
    ):
        raise Round0152Error("R0152 functional coordinate evidence changed")
    key = "prefix_drop_12p5m_seed42"
    with np.load(str(arrays_spec["canonical_path"]), allow_pickle=False) as arrays:
        high = np.asarray(arrays["high_radius"], dtype=np.float64)
        low = np.asarray(arrays[f"{key}__low_radius"], dtype=np.float64)
        stored_bootstrap = np.asarray(
            arrays[f"{key}__bootstrap"], dtype=np.float64
        )
        stored_null = np.asarray(
            arrays[f"{key}__permuted_null"], dtype=np.float64
        )
    if high.shape != (10_000,) or low.shape != high.shape:
        raise Round0152Error("R0152 fixed density arrays changed geometry")
    summary, bootstrap, null = density_v2_calibration(
        high,
        low,
        bootstrap_draws=1_000,
        bootstrap_seed=10_801,
        null_draws=1_000,
        null_seed=10_802,
    )
    density_cell = value.get("density_cell") or {}
    density_lineage = value.get("density_universe") or {}
    if (
        density_cell.get("density_v2") != summary
        or not np.array_equal(stored_bootstrap, bootstrap)
        or not np.array_equal(stored_null, null)
        or density_lineage.get("registered_floor")
        != 0.17589389755990817
        or density_cell.get("clears_unchanged_registered_floor")
        != (float(summary["correlation"]) >= 0.17589389755990817)
        or not all((value.get("checks") or {}).values())
    ):
        raise Round0152Error("R0152 fixed density evidence does not recompute")
    return {
        "coordinates": dict(coordinates_spec),
        "query_coordinates": dict(queries_spec),
        "density_arrays": dict(arrays_spec),
        "density_v2_recomputed": summary,
        "bootstrap_and_null_recomputed": True,
    }


def run_decision(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0152 rescue decision"
    )
    native_path = os.path.join(str(job["native_output"]), "matched-native.json")
    ood_path = os.path.join(str(job["ood_output"]), "matched-ood.json")
    functional_path = os.path.join(
        str(job["functional_output"]), "functional-density.json"
    )
    native = _read_json(native_path)
    ood = _read_json(ood_path)
    functional = _read_json(functional_path)
    for value, label in (
        (native, "R0152 native panel"),
        (ood, "R0152 OOD panel"),
        (functional, "R0152 functional/density panel"),
    ):
        validate_seal(value, label=label)
    if (
        native.get("schema") != NATIVE_SCHEMA
        or native.get("round_id") != ROUND_ID
        or ood.get("schema") != OOD_SCHEMA
        or ood.get("round_id") != ROUND_ID
        or functional.get("schema") != FUNCTIONAL_SCHEMA
        or functional.get("round_id") != ROUND_ID
    ):
        raise Round0152Error("R0152 decision input identity changed")
    authenticated_train = inherited._authenticate_half_train(active, job)
    authenticated_native = inherited._authenticate_native_selector(native)
    authenticated_ood = inherited._authenticate_ood_metrics(ood)
    authenticated_functional = _authenticate_functional_density(functional)
    model_lineage = functional.get("model_lineage") or {}
    if (
        model_lineage.get("train_receipt") != authenticated_train["train_receipt"]
        or model_lineage.get("graph") != authenticated_train["graph_manifest"]
        or (native.get("model_lineage") or {}).get("control_12p5m_train_receipt")
        != authenticated_train["train_receipt"]
        or (ood.get("model_lineage") or {}).get("control_12p5m_train_receipt")
        != authenticated_train["train_receipt"]
    ):
        raise Round0152Error("R0152 train/panel lineage disagrees")
    quality = quality_selector(
        functional_cell=functional["functional_cell"],
        density_v2=functional["density_cell"]["density_v2"]["correlation"],
        candidate_ood=ood["control_12p5m"],
        accepted_25m_ood=ood["treatment_25m"],
    )
    validity = {
        "train_accounting_authenticated": all(authenticated_train["checks"].values()),
        "native_panel_checks_pass": all((native.get("checks") or {}).values()),
        "ood_panel_checks_pass": all((ood.get("checks") or {}).values()),
        "functional_density_checks_pass": all(
            (functional.get("checks") or {}).values()
        ),
        "native_selector_recomputed": (
            authenticated_native.get("density_selector_recomputed") is True
        ),
        "ood_selector_recomputed": (
            authenticated_ood.get("gating_summaries_recomputed") is True
        ),
        "fixed_density_bootstrap_and_null_recomputed": (
            authenticated_functional.get("bootstrap_and_null_recomputed") is True
        ),
    }
    decision = build_decision(validity_checks=validity, quality=quality)
    receipt = seal({
        **decision,
        "release_sha": active["manifest"]["release_sha"],
        "native_panel": _signature(native_path),
        "ood_panel": _signature(ood_path),
        "functional_density_panel": _signature(functional_path),
        "authenticated_train_execution": authenticated_train,
        "authenticated_native_selector": authenticated_native,
        "authenticated_ood_metrics": authenticated_ood,
        "authenticated_functional_density": authenticated_functional,
        "training_performed": True,
        "map_registry_state_changed": False,
    })
    atomic_write_new_json(os.path.join(output, "decision.json"), receipt, immutable=True)


def run_job(active: dict[str, Any], job: dict[str, Any] | None = None) -> Any:
    if active.get("manifest", {}).get("round_id") != ROUND_ID or job is None:
        raise Round0152Error("R0152 handler requires its exact queue job")
    handlers = {
        "materialize_prefix_drop_subset": run_materialize_subset,
        "build_search_index": inherited.run_build_index,
        "qualify_fixed_search": inherited.run_qualify_search,
        "build_graph_part": inherited.run_graph_part,
        "assemble_graph": inherited.run_assemble_graph,
        "train_map": run_train,
        "transform_map": inherited.run_transform,
        "score_matched_native": inherited.run_score_native,
        "score_matched_ood": inherited.run_score_ood,
        "score_functional_density": run_functional_density,
        "decide_rescue": run_decision,
    }
    action = str(job.get("action") or "")
    try:
        handler = handlers[action]
    except KeyError as exc:
        raise Round0152Error(f"unknown R0152 action: {action!r}") from exc
    with _configured_inherited_contract():
        return handler(active, job)
