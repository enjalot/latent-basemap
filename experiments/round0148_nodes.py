"""Conditional 12.5M English-anchor rescue nodes for R0148.

The expensive index/graph/train/transform machinery deliberately reuses the
reviewed R0132 implementation.  This wrapper changes only round-specific
artifact identities and the preregistered population selector.  R0148 is not
launchable until an accepted R0147 review selects its positive branch and the
remaining panel/decision nodes are present in the issued release.
"""
from __future__ import annotations

import copy
import gc
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
    atomic_save_new_npy,
    atomic_save_new_npz,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0104_training import validate_substrate_manifest
from basemap.round0105_search import ELIGIBILITY_PATH, GROUPS, ROW_COUNT, group_ranges
from basemap.round0132_scale_bridge import (
    GRAPH_K,
    HALF_RETAINED_ROWS,
    N_NEIGHBORS,
    PIPELINE,
    PIPELINE_SCHEMA,
    SAMPLER_CLASS,
    TRAIN_CONFIG_SCHEMA as R0132_TRAIN_CONFIG_SCHEMA,
    TRAIN_RECEIPT_SCHEMA as R0132_TRAIN_RECEIPT_SCHEMA,
    PRODUCTION_CONFIG_SCHEMA as R0132_PRODUCTION_CONFIG_SCHEMA,
    GRAPH_SCHEMA as R0132_GRAPH_SCHEMA,
    ROUND_ID as R0132_ROUND_ID,
    POSITIVE_DESTINATION_POLICY as R0132_POSITIVE_DESTINATION_POLICY,
    validate_train_execution as validate_r0132_train_execution,
)
from basemap.round0148_english_anchor import (
    DENSITY_V2_FLOOR,
    OOD_METRICS,
    ROUND_ID,
    SUBSET_SCHEMA,
    build_subset_plan,
    english_anchor_decision,
    english_anchor_quotas,
    ranking_namespace,
)
from experiments import round0132_nodes as base
from experiments import round0119_nodes as r0119
from experiments import round0147_nodes as r0147
from experiments.round0105_nodes import _retained_batch
from experiments.round0106_nodes import GraphNodeContract
from experiments.round0085_nodes import density_v2_calibration
from basemap.round0108_evaluation import (
    HELDOUT_CORPUS_ROWS,
    HELDOUT_QUERY_ROWS,
    IN_MIX_LANGUAGES,
    K_HIT,
    K_LOW_MAX,
    POLISH,
    TRANSFORM_BATCH_ROWS,
    load_reviewed_model,
    recall_from_neighbors,
)


INDEX_SCHEMA = "round0148-english-anchor-search-index-v1"
QUALIFICATION_SCHEMA = "round0148-english-anchor-search-qualification-v1"
GRAPH_SHARD_SCHEMA = "round0148-english-anchor-graph-shard-v1"
GRAPH_PART_SCHEMA = "round0148-english-anchor-graph-part-v1"
GRAPH_SCHEMA = "round0148-english-anchor-fuzzy-graph-v1"
TRAIN_CONFIG_SCHEMA = "round0148-english-anchor-train-config-v1"
PRODUCTION_CONFIG_SCHEMA = "round0148-english-anchor-production-config-v1"
TRAIN_RECEIPT_SCHEMA = "round0148-english-anchor-train-receipt-v1"
NATIVE_SCHEMA = "round0148-english-anchor-density-v2-panel-v1"
OOD_SCHEMA = "round0148-english-anchor-matched-ood-panel-v1"
DECISION_SCHEMA = "round0148-english-anchor-rescue-decision-v1"
POSITIVE_DESTINATION_POLICY = "R0148-english-anchor-fuzzy-tconorm-graph"
CONTROL = "control_r0132_u12"
CANDIDATE = "candidate_english_anchor"


class Round0148NodeError(RuntimeError):
    """The conditional R0148 execution contract is malformed."""


def _normalize_train_identity(
    *,
    train: Mapping[str, Any],
    config_receipt: Mapping[str, Any],
    graph: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Reduce R0148's renamed receipts to the reviewed R0132 train law."""
    normalized_train = copy.deepcopy(dict(train))
    normalized_config_receipt = copy.deepcopy(dict(config_receipt))
    normalized_graph = copy.deepcopy(dict(graph))
    config = normalized_config_receipt.get("config")
    if not isinstance(config, dict):
        raise Round0148NodeError("R0148 production config is missing")

    normalized_train["schema"] = R0132_TRAIN_RECEIPT_SCHEMA
    normalized_train["round_id"] = R0132_ROUND_ID
    normalized_config_receipt["schema"] = R0132_PRODUCTION_CONFIG_SCHEMA
    normalized_config_receipt["round_id"] = R0132_ROUND_ID
    config["schema"] = R0132_TRAIN_CONFIG_SCHEMA
    normalized_graph["schema"] = R0132_GRAPH_SCHEMA
    normalized_graph["round_id"] = R0132_ROUND_ID

    execution = config.get("execution")
    if not isinstance(execution, dict):
        raise Round0148NodeError("R0148 execution config is missing")
    execution["required_pipeline"] = PIPELINE
    expected = execution.get("expected_pipeline_stamp")
    runtime = normalized_train.get("exact_execution_receipt")
    if not isinstance(expected, dict) or not isinstance(runtime, dict):
        raise Round0148NodeError("R0148 pipeline stamps are missing")
    expected["positive_destination_policy"] = R0132_POSITIVE_DESTINATION_POLICY
    runtime["positive_destination_policy"] = R0132_POSITIVE_DESTINATION_POLICY

    config_digest = sha256_bytes(canonical_json(config))
    normalized_config_receipt["config_sha256"] = config_digest
    normalized_train["production_config_sha256"] = config_digest
    return normalized_train, normalized_config_receipt, normalized_graph


def validate_train_execution(
    *,
    train: Mapping[str, Any],
    config_receipt: Mapping[str, Any],
    graph: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate R0148 as the exact R0132 train law on a new population."""
    original_config = config_receipt.get("config")
    if not isinstance(original_config, Mapping):
        raise Round0148NodeError("R0148 production config is missing")
    original_digest = sha256_bytes(canonical_json(original_config))
    if (
        train.get("schema") != TRAIN_RECEIPT_SCHEMA
        or train.get("round_id") != ROUND_ID
        or config_receipt.get("schema") != PRODUCTION_CONFIG_SCHEMA
        or config_receipt.get("round_id") != ROUND_ID
        or (config_receipt.get("config") or {}).get("schema")
        != TRAIN_CONFIG_SCHEMA
        or graph.get("schema") != GRAPH_SCHEMA
        or graph.get("round_id") != ROUND_ID
        or config_receipt.get("config_sha256") != original_digest
        or train.get("production_config_sha256") != original_digest
    ):
        raise Round0148NodeError("R0148 train identity is incomplete")
    expected = ((config_receipt.get("config") or {}).get("execution") or {}).get(
        "expected_pipeline_stamp"
    ) or {}
    runtime = train.get("exact_execution_receipt") or {}
    if (
        expected.get("positive_destination_policy")
        != POSITIVE_DESTINATION_POLICY
        or runtime.get("positive_destination_policy")
        != POSITIVE_DESTINATION_POLICY
    ):
        raise Round0148NodeError("R0148 population graph stamp changed")
    normalized = _normalize_train_identity(
        train=train, config_receipt=config_receipt, graph=graph
    )
    authenticated = validate_r0132_train_execution(
        train=normalized[0], config_receipt=normalized[1], graph=normalized[2]
    )
    return {
        **authenticated,
        "normalized_to_reviewed_r0132_train_law": True,
        "r0148_positive_destination_policy": POSITIVE_DESTINATION_POLICY,
    }


def _configure_base() -> None:
    """Configure the reviewed generic machinery inside this node process."""
    base.ROUND_ID = ROUND_ID
    base.SUBSET_SCHEMA = SUBSET_SCHEMA
    base.INDEX_SCHEMA = INDEX_SCHEMA
    base.QUALIFICATION_SCHEMA = QUALIFICATION_SCHEMA
    base.GRAPH_SHARD_SCHEMA = GRAPH_SHARD_SCHEMA
    base.GRAPH_PART_SCHEMA = GRAPH_PART_SCHEMA
    base.GRAPH_SCHEMA = GRAPH_SCHEMA
    base.TRAIN_CONFIG_SCHEMA = TRAIN_CONFIG_SCHEMA
    base.PRODUCTION_CONFIG_SCHEMA = PRODUCTION_CONFIG_SCHEMA
    base.TRAIN_RECEIPT_SCHEMA = TRAIN_RECEIPT_SCHEMA
    base.NATIVE_SCHEMA = NATIVE_SCHEMA
    base.OOD_SCHEMA = OOD_SCHEMA
    base.DECISION_SCHEMA = DECISION_SCHEMA
    base.PIPELINE = PIPELINE
    base.PIPELINE_SCHEMA = PIPELINE_SCHEMA
    base.SAMPLER_CLASS = SAMPLER_CLASS
    base.POSITIVE_DESTINATION_POLICY = POSITIVE_DESTINATION_POLICY
    base.GRAPH_CONTRACT = GraphNodeContract(
        round_id=ROUND_ID,
        k=GRAPH_K,
        n_neighbors=N_NEIGHBORS,
        shard_schema=GRAPH_SHARD_SCHEMA,
        part_schema=GRAPH_PART_SCHEMA,
        graph_schema=GRAPH_SCHEMA,
    )
    base.TRANSFORM_MAP_KEY = "r0148-diverse-jina-12p5m-english-anchor-seed42"
    base.TRANSFORM_SCIENTIFIC_UNIVERSE = (
        "R0148 deterministic 12,474,331-row English-anchor subset"
    )
    base.TRANSFORM_ROW_ORDER = "R0148 English-anchor compact global-row order"
    base.TRAIN_OUTPUT_LABEL = "R0148 12.5M English-anchor train output"
    base.INDEX_FILENAME = "english-anchor-12p5m.ivfpq"
    base.CANDIDATE_UNIVERSE_LABEL = "exact R0148 English-anchor subset"
    base.validate_train_execution = validate_train_execution


def run_select_subset(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any]:
    """Materialize the exact precomputed nested English-anchor population."""
    _configure_base()
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0148 deterministic English-anchor subset"
    )
    started = time.monotonic()
    substrate = validate_substrate_manifest(verify_payloads=False)
    eligibility = base._signature(
        ELIGIBILITY_PATH,
        str(job["eligibility_sha256"]),
        label="R0087 eligibility",
    )
    with np.load(ELIGIBILITY_PATH, allow_pickle=False) as archive:
        original_excluded = np.asarray(archive["excluded_rows"], dtype=np.int64)
    ranges = group_ranges(substrate["manifest"])
    counts: dict[str, int] = {}
    for group in GROUPS:
        start, stop = ranges[group]
        left = int(np.searchsorted(original_excluded, start, side="left"))
        right = int(np.searchsorted(original_excluded, stop, side="left"))
        counts[group] = stop - start - (right - left)
    quotas = english_anchor_quotas(counts)
    population_plan = build_subset_plan(counts)

    selected_groups: list[np.ndarray] = []
    selected_group_ids: list[np.ndarray] = []
    group_receipts: dict[str, Any] = {}
    for group_id, group in enumerate(GROUPS):
        start, stop = ranges[group]
        eligible = _retained_batch(original_excluded, start=start, stop=stop)
        chosen = base.select_lowest_sha256_rank(
            eligible,
            count=quotas[group],
            namespace=ranking_namespace(group),
        )
        selected_groups.append(chosen)
        selected_group_ids.append(np.full(len(chosen), group_id, dtype=np.uint8))
        group_receipts[group] = {
            "global_start": start,
            "global_stop": stop,
            "retained_rows": len(eligible),
            "selected_rows": len(chosen),
            "ordered_rows_sha256": ordered_array_sha256(chosen),
        }

    mapping = np.concatenate(selected_groups).astype(np.int64, copy=False)
    group_ids = np.concatenate(selected_group_ids)
    keep = np.zeros(ROW_COUNT, dtype=bool)
    keep[mapping] = True
    excluded = np.flatnonzero(~keep).astype(np.int64, copy=False)
    if (
        len(mapping) != HALF_RETAINED_ROWS
        or len(np.unique(mapping)) != HALF_RETAINED_ROWS
        or np.any(mapping[1:] <= mapping[:-1])
        or len(excluded) != ROW_COUNT - HALF_RETAINED_ROWS
    ):
        raise Round0148NodeError("R0148 deterministic subset did not close")

    paths = base._subset_paths(output)
    atomic_save_new_npy(paths["mapping"], mapping, immutable=True)
    atomic_save_new_npy(paths["group_ids"], group_ids, immutable=True)
    atomic_save_new_npy(paths["excluded"], excluded, immutable=True)
    signatures = {
        key: expected_input_signature(paths[key])
        for key in ("mapping", "group_ids", "excluded")
    }
    manifest = base.seal({
        "schema": SUBSET_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "full_retained_rows": sum(counts.values()),
        "selected_rows": len(mapping),
        "population_plan": population_plan,
        "selector": population_plan["selector"],
        "group_counts": counts,
        "quotas": quotas,
        "groups": group_receipts,
        "eligibility": eligibility,
        "substrate": substrate["signature"],
        "mapping": signatures["mapping"],
        "group_ids": signatures["group_ids"],
        "excluded": signatures["excluded"],
        "parts": base.group_part_specs(quotas),
        "checks": {
            "exact_target": True,
            "every_group_present": True,
            "all_eligible_english_retained": True,
            "languages_nested_inside_accepted_r0132_u12": True,
            "duplicate_control_inherited": True,
            "mapping_strictly_increasing": True,
            "mapping_and_excluded_partition_25m": True,
        },
        "training_performed": False,
        "optimizer_updates": 0,
        "map_outcomes_observed": False,
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(paths["manifest"], manifest, immutable=True)
    return {**manifest, "receipt": expected_input_signature(paths["manifest"])}


def _load_candidate_model(
    job: Mapping[str, Any], *, release_sha: str
) -> dict[str, Any]:
    graph = base._signature(
        str(job["graph_manifest"]), label="R0148 graph manifest"
    )
    bundle = base._load_model_bundle(
        train_output=str(job["train_output"]),
        graph_manifest=str(job["graph_manifest"]),
        graph_sha256=graph["sha256"],
        half=True,
    )
    if bundle["train"].get("release_sha") != release_sha:
        raise Round0148NodeError("R0148 candidate release changed")
    return bundle


def _load_r0132_control(job: Mapping[str, Any]) -> dict[str, Any]:
    graph = base._signature(
        str(job["control_graph_manifest"]),
        str(job["control_graph_manifest_sha256"]),
        label="accepted R0132 graph",
    )
    return load_reviewed_model(
        train_output=str(job["control_train_output"]),
        graph_manifest_path=str(job["control_graph_manifest"]),
        graph_manifest_sha256=graph["sha256"],
        expected_train_round_id="0132",
        expected_train_receipt_schema=R0132_TRAIN_RECEIPT_SCHEMA,
        expected_production_config_schema=R0132_PRODUCTION_CONFIG_SCHEMA,
        expected_seed=42,
        expected_graph_schema=R0132_GRAPH_SCHEMA,
    )


def _score_functional_cell(
    *,
    key: str,
    model: Any,
    source: np.ndarray,
    queries: np.ndarray,
    source_signature: Mapping[str, Any],
    shared_signature: Mapping[str, Any],
    reference: Mapping[str, Any],
    truth: Mapping[str, Any],
    centroids: Mapping[int, Any],
    output: str,
    model_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    from basemap.panel_v2 import score_panel
    from experiments.round0027_nodes import _panel_config

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
        raise Round0148NodeError(f"R0148 {key} functional transform is malformed")
    root = create_fresh_directory(
        f"{output}/{key}", label=f"R0148 {key} functional coordinates"
    )
    coordinate_path = f"{root}/coordinates.npy"
    query_path = f"{root}/query-coordinates.npy"
    atomic_save_new_npy(coordinate_path, coordinates, immutable=True)
    atomic_save_new_npy(query_path, query_coordinates, immutable=True)
    coordinate_signature = expected_input_signature(coordinate_path)
    query_signature = expected_input_signature(query_path)
    panel = score_panel(
        source,
        coordinates,
        config=_panel_config(),
        centroids_by_k=centroids,
        hiD_reference=reference,
        scale_admission=None,
        provenance={
            "round_id": ROUND_ID,
            "cell": key,
            "source": dict(source_signature),
            "coordinates": coordinate_signature,
            "shared_reference_receipt": dict(shared_signature),
        },
    )
    projection = r0147._projection_metrics(
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
        raise Round0148NodeError(f"R0148 {key} functional guards failed")
    return {
        "model_lineage": dict(model_lineage),
        "coordinates": coordinate_signature,
        "query_coordinates": query_signature,
        "coordinates_ordered_sha256": ordered_array_sha256(coordinates),
        "query_coordinates_ordered_sha256": ordered_array_sha256(
            query_coordinates
        ),
        "panel": panel,
        "projection": projection,
    }


def _score_fixed_density(
    *,
    key: str,
    bundle: Mapping[str, Any],
    universe: tuple[Any, ...],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    (
        source,
        representatives,
        retained_global_rows,
        anchors,
        _global_rows,
        high_radius,
        _lineage,
        reference,
    ) = universe
    return r0119._score_cell(
        key=key,
        bundle={
            "model": bundle["model"],
            "group": "historical_2m",
            "seed": 42,
            "training_population": key,
            "training_graph": key,
            "training_dose": "coverage-aligned-R0132-law",
            "training_representation": "int8-plus-fp16-scale",
            "training_dequantization": "device-fp32",
            "authenticated_training_semantics": {
                "source": "authenticated by R0148/R0132 model loader"
            },
            "train": bundle["train_signature"],
            "production_config": bundle["config_signature"],
            "model_signature": bundle["train"]["model"],
        },
        source=source,
        representatives=representatives,
        retained_global_rows=retained_global_rows,
        anchors=anchors,
        high_radius=high_radius,
        reference=reference,
    )


def _render_functional_pair(
    *,
    output: str,
    cells: Mapping[str, Mapping[str, Any]],
    labels: np.ndarray,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = create_fresh_directory(
        f"{output}/renders", label="R0148 functional comparison render"
    )
    sample = np.sort(
        np.random.RandomState(14_800).choice(2_000_000, 100_000, replace=False)
    )
    sample_path = f"{root}/sample-row-ids.npy"
    atomic_save_new_npy(sample_path, sample, immutable=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=140)
    colors = np.asarray(labels[sample] % 20, dtype=np.int16)
    titles = {
        CONTROL: "R0132 proportional U12 control",
        CANDIDATE: "R0148 English-anchor U12 candidate",
    }
    limits: dict[str, Any] = {}
    for axis, key in zip(axes, (CONTROL, CANDIDATE), strict=True):
        coordinates = np.load(
            cells[key]["coordinates"]["canonical_path"],
            mmap_mode="r",
            allow_pickle=False,
        )
        points = np.asarray(coordinates[sample], dtype=np.float32)
        low = np.quantile(points, 0.001, axis=0)
        high = np.quantile(points, 0.999, axis=0)
        pad = np.maximum((high - low) * 0.03, 1.0e-6)
        axis.scatter(
            points[:, 0],
            points[:, 1],
            c=colors,
            cmap="tab20",
            s=0.18,
            alpha=0.35,
            linewidths=0,
            rasterized=True,
        )
        axis.set_xlim(float(low[0] - pad[0]), float(high[0] + pad[0]))
        axis.set_ylim(float(low[1] - pad[1]), float(high[1] + pad[1]))
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(titles[key], fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
        limits[key] = {
            "quantile_low": low.tolist(),
            "quantile_high": high.tolist(),
        }
    figure.tight_layout()
    render_path = f"{root}/english-anchor-functional-comparison.png"
    figure.savefig(render_path, bbox_inches="tight")
    plt.close(figure)
    os.chmod(render_path, 0o444)
    receipt = base.seal({
        "schema": "round0148-english-anchor-functional-render-v1",
        "round_id": ROUND_ID,
        "sample": expected_input_signature(sample_path),
        "sample_seed": 14_800,
        "sample_rows": 100_000,
        "color": "frozen R0037 k256 label modulo 20",
        "axes": "per-cell 0.1%-99.9% robust diagnostic axes",
        "limits": limits,
        "render": expected_input_signature(render_path),
    })
    manifest_path = f"{root}/render-manifest.json"
    atomic_write_new_json(manifest_path, receipt, immutable=True)
    return {**receipt, "manifest": expected_input_signature(manifest_path)}


def run_functional_density_panel(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any]:
    """Score candidate and accepted U12 control on frozen function/density."""
    _configure_base()
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0148 functional and density-v2 panel"
    )
    started = time.monotonic()
    subset = base._load_subset(str(job["subset_output"]))
    source_signature, source, queries = r0147._load_shared_evaluation_inputs(job)
    _shared, shared_signature, reference, truth, centroids = r0147._load_reference(
        job
    )
    candidate = _load_candidate_model(
        job, release_sha=active["manifest"]["release_sha"]
    )
    control = _load_r0132_control(job)
    cells = {
        CONTROL: _score_functional_cell(
            key=CONTROL,
            model=control["model"],
            source=source,
            queries=queries,
            source_signature=source_signature,
            shared_signature=shared_signature,
            reference=reference,
            truth=truth,
            centroids=centroids,
            output=output,
            model_lineage={
                "train": control["train_signature"],
                "config": control["config_signature"],
                "graph": control["graph_signature"],
            },
        ),
        CANDIDATE: _score_functional_cell(
            key=CANDIDATE,
            model=candidate["model"],
            source=source,
            queries=queries,
            source_signature=source_signature,
            shared_signature=shared_signature,
            reference=reference,
            truth=truth,
            centroids=centroids,
            output=output,
            model_lineage={
                "train": candidate["train_signature"],
                "config": candidate["config_signature"],
                "graph": candidate["graph_signature"],
            },
        ),
    }
    universe = r0119._load_universe(job)
    if universe[6]["source"] != source_signature:
        raise Round0148NodeError("functional and density source lineage differs")
    density_cells: dict[str, Any] = {}
    density_arrays: dict[str, np.ndarray] = {}
    for key, bundle in ((CONTROL, control), (CANDIDATE, candidate)):
        density, arrays = _score_fixed_density(
            key=key, bundle=bundle, universe=universe
        )
        density_cells[key] = density
        for name, value in arrays.items():
            density_arrays[name] = value
    arrays_path = f"{output}/density-v2-arrays.npz"
    atomic_save_new_npz(arrays_path, immutable=True, **density_arrays)
    render = _render_functional_pair(
        output=output,
        cells=cells,
        labels=np.asarray(reference["labels"][256], dtype=np.int32),
    )
    candidate_density = float(
        density_cells[CANDIDATE]["density_v2"]["correlation"]
    )
    receipt = base.seal({
        "schema": NATIVE_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "functional_universe": "exact R0037/R0140 fixed functional universe",
        "source": source_signature,
        "subset_manifest": subset["manifest_signature"],
        "population_plan": subset["manifest"]["population_plan"],
        "shared_reference_receipt": shared_signature,
        "cells": cells,
        "render": render,
        "density_v2": {
            "universe_lineage": universe[6],
            "cells": density_cells,
            "arrays": expected_input_signature(arrays_path),
            "registered_floor": DENSITY_V2_FLOOR,
            "candidate_clears_floor": candidate_density >= DENSITY_V2_FLOOR,
            "floor_recalibrated": False,
        },
        "checks": {
            "same_functional_source_queries_truth_and_references": True,
            "same_fixed_density_source_reference_and_anchors": True,
            "both_functional_cells_finite_noncollapsed": all(
                cell["panel"]["guards"]["coords_finite"] is True
                and cell["panel"]["guards"]["coords_collapsed"] is False
                for cell in cells.values()
            ),
            "density_floor_unchanged": DENSITY_V2_FLOOR
            == universe[6]["registered_floor"],
        },
        "training_performed": False,
        "map_registry_state_changed": False,
        "wall_seconds": time.monotonic() - started,
    })
    path = f"{output}/functional-density-panel.json"
    atomic_write_new_json(path, receipt, immutable=True)
    del candidate["model"], control["model"]
    gc.collect()
    return {**receipt, "receipt": expected_input_signature(path)}


def run_matched_ood_panel(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare the candidate with accepted R0132 on identical OOD probes."""
    _configure_base()
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0148 matched held-out OOD panel"
    )
    started = time.monotonic()
    candidate = _load_candidate_model(
        job, release_sha=active["manifest"]["release_sha"]
    )
    control = _load_r0132_control(job)
    selection_signature = base._signature(
        str(job["selection"]),
        str(job["selection_sha256"]),
        label="accepted R0108 selectors",
    )
    probes: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    labels = (CONTROL, CANDIDATE)

    def record(prefix: str, report: dict[str, Any]) -> None:
        for key, value in report.pop("arrays").items():
            arrays[f"{prefix}__{key}"] = value
        probes[str(report["name"])] = report

    with np.load(str(job["selection"]), allow_pickle=False) as selected:
        for language in (*IN_MIX_LANGUAGES, POLISH):
            corpus_rows = np.asarray(
                selected[f"{language}__corpus"], dtype=np.int64
            )
            query_rows = np.asarray(
                selected[f"{language}__queries"], dtype=np.int64
            )
            if (
                corpus_rows.shape != (HELDOUT_CORPUS_ROWS,)
                or query_rows.shape != (HELDOUT_QUERY_ROWS,)
            ):
                raise Round0148NodeError(f"{language} selector changed")
            corpus, queries = base._selected_source(
                job["language_sources"][language],
                corpus_rows,
                query_rows,
                label=f"accepted R0108 {language}",
            )
            record(
                language,
                base._matched_probe(
                    name=language,
                    corpus=corpus,
                    queries=queries,
                    control_model=control["model"],
                    treatment_model=candidate["model"],
                    duplicate_policy="require-disjoint",
                    cell_labels=labels,
                ),
            )
            del corpus, queries

        for name, prefix in (("fineweb", "fineweb"), ("dadabase", "dadabase")):
            corpus_rows = np.asarray(selected[f"{name}__corpus"], dtype=np.int64)
            query_rows = np.asarray(selected[f"{name}__queries"], dtype=np.int64)
            corpus, queries = base._selected_source(
                job["diagnostic_sources"][name],
                corpus_rows,
                query_rows,
                label=f"accepted R0108 {name}",
            )
            report_name = "fineweb-heldout" if name == "fineweb" else name
            record(
                prefix,
                base._matched_probe(
                    name=report_name,
                    corpus=corpus,
                    queries=queries,
                    control_model=control["model"],
                    treatment_model=candidate["model"],
                    duplicate_policy="diagnostic",
                    cell_labels=labels,
                ),
            )
            del corpus, queries

    trec_corpus = job["diagnostic_sources"]["trec_corpus"]
    trec_queries = job["diagnostic_sources"]["trec_queries"]
    base._signature(
        trec_corpus["canonical_path"], trec_corpus["sha256"], label="TREC corpus"
    )
    base._signature(
        trec_queries["canonical_path"],
        trec_queries["sha256"],
        label="TREC queries",
    )
    record(
        "trec",
        base._matched_probe(
            name="trec-covid",
            corpus=np.load(
                trec_corpus["canonical_path"], mmap_mode="r", allow_pickle=False
            ),
            queries=np.load(
                trec_queries["canonical_path"], mmap_mode="r", allow_pickle=False
            ),
            control_model=control["model"],
            treatment_model=candidate["model"],
            duplicate_policy="diagnostic",
            cell_labels=labels,
        ),
    )

    summaries: dict[str, dict[str, float]] = {}
    for label in labels:
        in_mix = np.asarray([
            probes[language]["cells"][label]["recall_at_50_of_high10"]
            for language in IN_MIX_LANGUAGES
        ])
        summaries[label] = {
            "fineweb_recall_at_50_of_high10": probes["fineweb-heldout"][
                "cells"
            ][label]["recall_at_50_of_high10"],
            "polish_recall_at_50_of_high10": probes[POLISH]["cells"][label][
                "recall_at_50_of_high10"
            ],
            "in_mix_median_recall_at_50_of_high10": float(np.median(in_mix)),
        }
    every_cell = [
        cell for probe in probes.values() for cell in probe["cells"].values()
    ]
    arrays_path = f"{output}/matched-ood-arrays.npz"
    atomic_save_new_npz(arrays_path, immutable=True, **arrays)
    receipt = base.seal({
        "schema": OOD_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "selection": selection_signature,
        "model_lineage": {
            CONTROL: {
                "train": control["train_signature"],
                "config": control["config_signature"],
                "graph": control["graph_signature"],
            },
            CANDIDATE: {
                "train": candidate["train_signature"],
                "config": candidate["config_signature"],
                "graph": candidate["graph_signature"],
            },
        },
        "summaries": summaries,
        "probes": probes,
        "checks": {
            "same_queries_corpora_and_truth_for_both_models": True,
            "polish_absent_from_both_training_inventories": True,
            "every_cell_recall50_at_least_recall10": all(
                cell["recall_at_50_of_high10"] >= cell["recall_at_10"]
                for cell in every_cell
            ),
            "all_probe_coordinates_finite_noncollapsed": all(
                cell["finite_noncollapsed"] for cell in every_cell
            ),
        },
        "roles": {
            "fineweb_polish_and_in_mix_recall50": "registered 0.97 retention gates",
            "projection_ffr": "diagnostic-only",
            "trec-covid": "diagnostic-only",
            "dadabase": "diagnostic-only",
        },
        "arrays": expected_input_signature(arrays_path),
        "training_performed": False,
        "universal_ood_claimed": False,
        "wall_seconds": time.monotonic() - started,
    })
    path = f"{output}/matched-ood.json"
    atomic_write_new_json(path, receipt, immutable=True)
    del candidate["model"], control["model"]
    gc.collect()
    return {**receipt, "receipt": expected_input_signature(path)}


def _authenticate_ood_panel(ood: Mapping[str, Any]) -> dict[str, Any]:
    arrays_spec = ood.get("arrays") or {}
    arrays_path = str(arrays_spec.get("canonical_path") or "")
    if base._signature(arrays_path, label="R0148 OOD arrays") != arrays_spec:
        raise Round0148NodeError("R0148 OOD arrays changed")
    probes = ood.get("probes") or {}
    expected = (
        *IN_MIX_LANGUAGES,
        POLISH,
        "fineweb-heldout",
        "dadabase",
        "trec-covid",
    )
    if set(probes) != set(expected):
        raise Round0148NodeError("R0148 OOD probe set changed")
    prefixes = {
        **{language: language for language in (*IN_MIX_LANGUAGES, POLISH)},
        "fineweb-heldout": "fineweb",
        "dadabase": "dadabase",
        "trec-covid": "trec",
    }
    with np.load(arrays_path, allow_pickle=False) as arrays:
        for probe in expected:
            prefix = prefixes[probe]
            truth = np.asarray(
                arrays[f"{prefix}__exact_high_d_top10"], dtype=np.int64
            )
            if truth.ndim != 2 or truth.shape[1] != K_HIT:
                raise Round0148NodeError(f"R0148 {probe} truth changed")
            for label in (CONTROL, CANDIDATE):
                low = np.asarray(
                    arrays[f"{prefix}__{label}_low_neighbors_top50"],
                    dtype=np.int64,
                )
                cell = probes[probe]["cells"][label]
                expected10 = recall_from_neighbors(truth, low[:, :K_HIT])
                expected50 = recall_from_neighbors(truth, low[:, :K_LOW_MAX])
                if (
                    low.shape != (len(truth), K_LOW_MAX)
                    or not math.isclose(
                        float(cell["recall_at_10"]),
                        expected10,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                    or not math.isclose(
                        float(cell["recall_at_50_of_high10"]),
                        expected50,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                ):
                    raise Round0148NodeError(
                        f"R0148 {probe} {label} recall does not recompute"
                    )
    summaries = ood.get("summaries") or {}
    recomputed: dict[str, dict[str, float]] = {}
    for label in (CONTROL, CANDIDATE):
        in_mix = np.asarray([
            probes[language]["cells"][label]["recall_at_50_of_high10"]
            for language in IN_MIX_LANGUAGES
        ])
        recomputed[label] = {
            "fineweb_recall_at_50_of_high10": probes["fineweb-heldout"][
                "cells"
            ][label]["recall_at_50_of_high10"],
            "polish_recall_at_50_of_high10": probes[POLISH]["cells"][label][
                "recall_at_50_of_high10"
            ],
            "in_mix_median_recall_at_50_of_high10": float(np.median(in_mix)),
        }
    if summaries != recomputed:
        raise Round0148NodeError("R0148 OOD summaries do not recompute")
    return {
        "arrays": dict(arrays_spec),
        "probe_count": len(expected),
        "recall10_and_recall50_recomputed": True,
        "summaries_recomputed": True,
    }


def _authenticate_density_panel(
    functional: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any]:
    density = functional.get("density_v2") or {}
    arrays_spec = density.get("arrays") or {}
    arrays_path = str(arrays_spec.get("canonical_path") or "")
    if (
        base._signature(arrays_path, label="R0148 density arrays") != arrays_spec
        or density.get("registered_floor") != DENSITY_V2_FLOOR
        or density.get("floor_recalibrated") is not False
    ):
        raise Round0148NodeError("R0148 density evidence changed")
    universe = r0119._load_universe(job)
    high_radius = np.asarray(universe[5], dtype=np.float64)
    with np.load(arrays_path, allow_pickle=False) as arrays:
        for key in (CONTROL, CANDIDATE):
            low = np.asarray(arrays[f"{key}__low_radius"], dtype=np.float64)
            stored_bootstrap = np.asarray(
                arrays[f"{key}__bootstrap"], dtype=np.float64
            )
            stored_null = np.asarray(
                arrays[f"{key}__permuted_null"], dtype=np.float64
            )
            summary, bootstrap, null = density_v2_calibration(
                high_radius,
                low,
                bootstrap_draws=1_000,
                bootstrap_seed=10_801,
                null_draws=1_000,
                null_seed=10_802,
            )
            if (
                functional["density_v2"]["cells"][key]["density_v2"]
                != summary
                or not np.array_equal(stored_bootstrap, bootstrap)
                or not np.array_equal(stored_null, null)
            ):
                raise Round0148NodeError(
                    f"R0148 {key} density-v2 does not recompute"
                )
    candidate_value = float(
        functional["density_v2"]["cells"][CANDIDATE]["density_v2"][
            "correlation"
        ]
    )
    if density.get("candidate_clears_floor") != (
        candidate_value >= DENSITY_V2_FLOOR
    ):
        raise Round0148NodeError("R0148 density floor comparison changed")
    return {
        "arrays": dict(arrays_spec),
        "fixed_universe_reloaded": True,
        "both_cells_recomputed": True,
        "registered_floor": DENSITY_V2_FLOOR,
    }


def run_decision(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any]:
    """Authenticate all evidence and apply the immutable rescue selector."""
    _configure_base()
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0148 English-anchor rescue decision"
    )
    functional_path = (
        f"{job['functional_output']}/functional-density-panel.json"
    )
    ood_path = f"{job['ood_output']}/matched-ood.json"
    functional = base._load_json(functional_path)
    ood = base._load_json(ood_path)
    base.validate_seal(functional, label="R0148 functional/density panel")
    base.validate_seal(ood, label="R0148 OOD panel")
    if (
        functional.get("schema") != NATIVE_SCHEMA
        or functional.get("round_id") != ROUND_ID
        or ood.get("schema") != OOD_SCHEMA
        or ood.get("round_id") != ROUND_ID
        or functional.get("release_sha") != active["manifest"]["release_sha"]
        or ood.get("release_sha") != active["manifest"]["release_sha"]
    ):
        raise Round0148NodeError("R0148 decision input identity changed")
    authenticated_train = base._authenticate_half_train(active, job)
    expected_candidate = {
        "train": authenticated_train["train_receipt"],
        "config": authenticated_train["production_config"],
        "graph": authenticated_train["graph_manifest"],
    }
    expected_control = {
        "train": expected_input_signature(
            f"{job['control_train_output']}/train-receipt.json"
        ),
        "config": expected_input_signature(
            f"{job['control_train_output']}/production-config.json"
        ),
        "graph": base._signature(
            str(job["control_graph_manifest"]),
            str(job["control_graph_manifest_sha256"]),
            label="accepted R0132 graph",
        ),
    }
    if (
        functional["cells"][CANDIDATE]["model_lineage"] != expected_candidate
        or ood["model_lineage"][CANDIDATE] != expected_candidate
        or functional["cells"][CONTROL]["model_lineage"] != expected_control
        or ood["model_lineage"][CONTROL] != expected_control
    ):
        raise Round0148NodeError("R0148 panel/train lineage differs")
    density = functional.get("density_v2") or {}
    authenticated_density = _authenticate_density_panel(functional, job)
    authenticated_ood = _authenticate_ood_panel(ood)
    validity = {
        "train_accounting_authenticated": all(
            authenticated_train["checks"].values()
        ),
        "functional_and_density_checks_pass": all(
            functional["checks"].values()
        ),
        "ood_panel_checks_pass": all(ood["checks"].values()),
        "candidate_model_lineage_shared_by_all_panels": True,
        "fixed_density_floor_not_recalibrated": True,
    }
    selected = english_anchor_decision(
        validity_checks=validity,
        candidate_functional_cell=functional["cells"][CANDIDATE],
        density_v2=float(
            density["cells"][CANDIDATE]["density_v2"]["correlation"]
        ),
        control_ood=ood["summaries"][CONTROL],
        candidate_ood=ood["summaries"][CANDIDATE],
    )
    receipt = base.seal({
        **selected,
        "release_sha": active["manifest"]["release_sha"],
        "functional_density_panel": base._signature(
            functional_path, label="R0148 functional/density panel"
        ),
        "ood_panel": base._signature(ood_path, label="R0148 OOD panel"),
        "authenticated_train_execution": authenticated_train,
        "authenticated_density_metrics": authenticated_density,
        "authenticated_ood_metrics": authenticated_ood,
        "authenticated_r0132_control": expected_control,
        "population_plan": functional.get("population_plan"),
        "training_performed": True,
    })
    path = f"{output}/decision.json"
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_job(
    active: dict[str, Any], job: dict[str, Any] | None = None
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID or job is None:
        raise Round0148NodeError("R0148 handler requires its exact round/job")
    _configure_base()
    action = str(job.get("action") or "")
    if action == "select_english_anchor_subset":
        return run_select_subset(active, job)
    handlers = {
        "build_english_anchor_search_index": base.run_build_index,
        "qualify_english_anchor_search": base.run_qualify_search,
        "build_english_anchor_graph_part": base.run_graph_part,
        "assemble_english_anchor_graph": base.run_assemble_graph,
        "train_english_anchor_map": base.run_train,
        "transform_english_anchor_map": base.run_transform,
        "score_english_anchor_function_density": run_functional_density_panel,
        "score_english_anchor_ood": run_matched_ood_panel,
        "decide_english_anchor_rescue": run_decision,
    }
    try:
        handler = handlers[action]
    except KeyError as exc:
        raise Round0148NodeError(
            f"R0148 action is absent or not yet implemented: {action!r}"
        ) from exc
    return handler(active, job)
