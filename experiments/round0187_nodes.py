"""Execute the R0187 composition-controlled nested ladder."""
from __future__ import annotations

import gc
import json
import os
import resource
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
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0104_training import L2NormalizedArray
from basemap.round0187_composition_nested_ladder import (
    CAPABILITY,
    CORPORA,
    DIMENSION,
    EVALUATION_SCHEMA,
    FULL_COUNTS,
    GRAPH_SCHEMA_PREFIX,
    HASH_NAMESPACE,
    NestedScalePromptTrainingInput,
    POPULATION_CAPABILITY,
    POPULATION_SCHEMA,
    POSITIVE_ROWS_PER_UPDATE,
    ROUND_ID,
    RUNG_COUNTS,
    RUNG_ROWS,
    SEED,
    SYNTHESIS_SCHEMA,
    TARGET_POSITIVE_DRAWS_PER_EDGE,
    TRAIN_SCHEMA_PREFIX,
    Round0187Error,
    ladder_decision,
    primary_metric_view,
    select_nested_positions,
    successful_updates_for_edges,
    train_checks_close,
    train_config,
)
from basemap import round0113_prompt_contrast as prompt_contract
from experiments import round0113_nodes as prompt_nodes
from experiments import round0166_nodes as q2


GRAPH_INDEX_DESCRIPTION = (
    "complete row-disjoint GPU IndexIVFFlat/IP fp32 shard search with one "
    "shared coarse quantizer and deterministic global top-k"
)
PILE_QUERY_SCHEMA = "round0171-prompted-8m-heldout-query-v1"
R0180_TRAIN_SCHEMA = "round0180-prompted-8m-dose-matched-train-receipt-v1"
ALLOWED_ACTIONS = {
    "stage_nested_populations",
    "build_nested_graph",
    "train_nested_rung",
    "evaluate_nested_rung",
    "evaluate_full_endpoint",
    "synthesize_nested_ladder",
}


def _signature(path: str, *, label: str) -> dict[str, Any]:
    try:
        return expected_input_signature(path)
    except Exception as error:
        raise Round0187Error(f"{label} is unavailable or changed") from error


def _read_sealed(path: str, *, label: str) -> dict[str, Any]:
    try:
        return prompt_contract.read_sealed(path, label=label)
    except Exception as error:
        raise Round0187Error(f"{label} seal is invalid") from error


def _write_subset_matrix(
    path: str,
    source: np.memmap,
    positions: np.ndarray,
    *,
    block_rows: int = 100_000,
) -> None:
    def writer(temp_path: str) -> None:
        target = np.memmap(
            temp_path,
            mode="w+",
            dtype="<f2",
            shape=(len(positions), DIMENSION),
        )
        for start in range(0, len(positions), block_rows):
            stop = min(start + block_rows, len(positions))
            target[start:stop] = source[positions[start:stop]]
        target.flush()
        del target

    atomic_build_new_file(path, writer, immutable=True)


def _corpus_ranges(counts: Mapping[str, int]) -> dict[str, list[int]]:
    cursor = 0
    output: dict[str, list[int]] = {}
    for corpus, _start, _stop in CORPORA:
        count = int(counts[corpus])
        output[corpus] = [cursor, cursor + count]
        cursor += count
    return output


def run_stage_populations(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0187Error("population handler received another queue")
    source_signature = dict(job["source_population_receipt"])
    source_population = _read_sealed(
        prompt_contract.verify_signature(
            source_signature, label="accepted R0165 population receipt"
        ),
        label="accepted R0165 population",
    )
    if (
        source_population.get("schema")
        != "round0165-prompted-english-frozen-prefix-population-v1"
        or source_population.get("round_id") != "0165"
        or source_population.get("outcome")
        != "prompted-8m-frozen-prefix-population-qualified"
        or int(source_population.get("retained_rows", -1)) != RUNG_ROWS["full"]
        or int(source_population.get("dimension", -1)) != DIMENSION
        or source_population.get("dtype") != "<f2"
        or (source_population.get("proofs") or {}).get("multiplicity_is_metadata")
        is not True
    ):
        raise Round0187Error("accepted R0165 population contract changed")
    mapping_path = prompt_contract.verify_signature(
        source_population["mapping"], label="R0165 canonical mapping"
    )
    source_path = prompt_contract.verify_signature(
        source_population["document_compact"], label="R0165 prompted matrix"
    )
    mapping = np.load(mapping_path, mmap_mode="r", allow_pickle=False)
    source = np.memmap(
        source_path,
        mode="r",
        dtype="<f2",
        shape=(RUNG_ROWS["full"], DIMENSION),
    )
    selections = select_nested_positions(mapping)
    roots = {str(value) for value in job["outputs"]}
    expected_roots = {
        str(job["population_roots"]["quarter"]),
        str(job["population_roots"]["half"]),
    }
    if roots != expected_roots:
        raise Round0187Error("population output roots changed")

    receipts: dict[str, Any] = {}
    started = time.monotonic()
    for rung in ("quarter", "half"):
        positions = selections[rung]
        root = create_fresh_directory(
            str(job["population_roots"][rung]),
            label=f"R0187 {rung} population",
        )
        positions_path = os.path.join(root, "source-compact-positions.i64.npy")
        mapping_output = os.path.join(root, "compact-to-canonical.i64.npy")
        matrix_output = os.path.join(root, "document-compact.f16")
        atomic_save_new_npy(positions_path, positions, immutable=True)
        selected_mapping = np.asarray(mapping[positions], dtype=np.int64)
        atomic_save_new_npy(mapping_output, selected_mapping, immutable=True)
        _write_subset_matrix(matrix_output, source, positions)
        counts = {
            corpus: int(
                np.count_nonzero(
                    (selected_mapping >= canonical_start)
                    & (selected_mapping < canonical_stop)
                )
            )
            for corpus, canonical_start, canonical_stop in CORPORA
        }
        if counts != RUNG_COUNTS[rung]:
            raise Round0187Error(f"{rung} corpus count drift")
        receipt_path = os.path.join(root, "population.json")
        receipt = prompt_contract.seal({
            "schema": POPULATION_SCHEMA,
            "round_id": ROUND_ID,
            "rung": rung,
            "outcome": "composition-nested-population-qualified",
            "capabilities": [POPULATION_CAPABILITY],
            "source_population": source_signature,
            "source_mapping": dict(source_population["mapping"]),
            "source_document_compact": dict(source_population["document_compact"]),
            "selection": {
                "rank": "ascending SHA-256 digest, canonical row ascending tie-break",
                "hash_namespace": HASH_NAMESPACE,
                "hash_input": "corpus-name NUL canonical-row-as-big-endian-u64",
                "within_corpus": True,
                "emit_order": "accepted R0165 compact/canonical order",
                "positions": _signature(
                    positions_path, label=f"R0187 {rung} source positions"
                ),
                "positions_ordered_sha256": ordered_array_sha256(positions),
                "mapping_ordered_sha256": ordered_array_sha256(selected_mapping),
            },
            "retained_rows": len(positions),
            "dimension": DIMENSION,
            "dtype": "<f2",
            "corpus_counts": counts,
            "corpus_compact_ranges": _corpus_ranges(counts),
            "mapping": _signature(
                mapping_output, label=f"R0187 {rung} canonical mapping"
            ),
            "document_compact": _signature(
                matrix_output, label=f"R0187 {rung} prompted matrix"
            ),
            "proofs": {
                "mapping_is_r0165_subset": True,
                "canonical_order_restored": True,
                "composition_counts_exact": True,
                "quarter_subset_of_half": True,
                "multiplicity_is_metadata": True,
                "reembedding_performed": False,
            },
            "graph_built": False,
            "training_performed": False,
        })
        atomic_write_new_json(receipt_path, receipt, immutable=True)
        receipts[rung] = _signature(
            receipt_path, label=f"R0187 {rung} population receipt"
        )

    quarter_positions = selections["quarter"]
    half_positions = selections["half"]
    if not np.array_equal(
        quarter_positions,
        np.intersect1d(quarter_positions, half_positions, assume_unique=True),
    ):
        raise Round0187Error("published quarter/half nesting failed")
    summary = prompt_contract.seal({
        "schema": "round0187-composition-nested-population-summary-v1",
        "round_id": ROUND_ID,
        "source_population": source_signature,
        "population_receipts": receipts,
        "quarter_subset_of_half": True,
        "half_subset_of_full": True,
        "full_endpoint_reused_byte_exact_from_round": "0180",
        "wall_s": time.monotonic() - started,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024**2),
        "gpu_used": False,
    })
    atomic_write_new_json(
        str(job["population_summary"]), summary, immutable=True
    )
    del source, mapping, selections
    gc.collect()


def _population_reader(
    job: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    rung = str(job.get("rung") or "")
    if rung not in {"quarter", "half"}:
        raise Round0187Error("nested population rung changed")
    path = str(job["population_receipt_path"])
    signature = _signature(path, label=f"R0187 {rung} population receipt")
    population = _read_sealed(path, label=f"R0187 {rung} population")
    if (
        population.get("schema") != POPULATION_SCHEMA
        or population.get("round_id") != ROUND_ID
        or population.get("rung") != rung
        or population.get("outcome") != "composition-nested-population-qualified"
        or population.get("capabilities") != [POPULATION_CAPABILITY]
        or int(population.get("retained_rows", -1)) != RUNG_ROWS[rung]
        or population.get("corpus_counts") != RUNG_COUNTS[rung]
        or population.get("corpus_compact_ranges")
        != _corpus_ranges(RUNG_COUNTS[rung])
        or int(population.get("dimension", -1)) != DIMENSION
        or population.get("dtype") != "<f2"
        or (population.get("proofs") or {}).get("quarter_subset_of_half") is not True
        or (population.get("proofs") or {}).get("multiplicity_is_metadata")
        is not True
    ):
        raise Round0187Error(f"R0187 {rung} population contract changed")
    for key in ("mapping", "document_compact"):
        prompt_contract.verify_signature(
            population[key], label=f"R0187 {rung} {key}"
        )
    prompt_contract.verify_signature(
        population["selection"]["positions"],
        label=f"R0187 {rung} selection positions",
    )
    return population, signature


def _build_corpus_references(
    *,
    output: str,
    X: np.ndarray,
    population: Mapping[str, Any],
    population_signature: Mapping[str, Any],
    config: Any,
) -> dict[str, Any]:
    from basemap.panel_v2 import (
        _matrix_identity,
        build_hiD_reference,
        sample_anchors,
        save_hiD_reference,
    )
    from experiments.score_complete_panel import frozen_centroids

    if population.get("rung") != "quarter":
        return {}
    root = create_fresh_directory(
        os.path.join(output, "common-corpus-references"),
        label="R0187 common corpus references",
    )
    results: dict[str, Any] = {}
    for corpus, _canonical_start, _canonical_stop in CORPORA:
        start, stop = population["corpus_compact_ranges"][corpus]
        values = np.ascontiguousarray(X[int(start) : int(stop)])
        corpus_root = create_fresh_directory(
            os.path.join(root, corpus), label=f"R0187 {corpus} reference"
        )
        centroid_root = create_fresh_directory(
            os.path.join(corpus_root, "centroids"),
            label=f"R0187 {corpus} centroids",
        )
        centroids = frozen_centroids(
            values, (256, 1024), centroid_root, seed=0, iters=25
        )
        centroid_signatures = {
            str(k): _signature(
                os.path.join(centroid_root, f"centroids_k{k}.npy"),
                label=f"R0187 {corpus} k{k} centroids",
            )
            for k in (256, 1024)
        }
        identity = {
            # panel-v2 accepts only its exact ordered_array/ordered_shards
            # identity schemas.  Bind the actual normalized slice bytes here;
            # the population receipt and range remain explicit convention
            # fields and are checked again by the evaluation node.
            "data_identity": _matrix_identity(values),
            "convention": {
                "row_order": "R0187 quarter canonical order within corpus",
                "distance": "cosine via fp32-L2-normalized squared L2",
                "self_exclusion": True,
                "anchor_namespace": f"R0187 quarter {corpus} local compact IDs",
                "embedding_prompt": "document",
                "population_receipt_sha256": str(population_signature["sha256"]),
                "compact_range": [int(start), int(stop)],
                "corpus": corpus,
            },
        }
        reference = build_hiD_reference(
            values,
            sample_anchors(len(values), config),
            config,
            centroids_by_k=centroids,
            **identity,
        )
        reference_path = os.path.join(corpus_root, "high-d-reference.npz")
        save_hiD_reference(reference, reference_path)
        results[corpus] = {
            "rows": len(values),
            "compact_range": [int(start), int(stop)],
            "centroids": centroid_signatures,
            "high_d_reference": _signature(
                reference_path, label=f"R0187 {corpus} high-D reference"
            ),
            "high_d_reference_key": reference["key"],
            "high_d_reference_content_sha256": reference["content_sha256"],
            "reference_identity": identity,
        }
        del values, centroids, reference
        gc.collect()
    return results


def _graph_schema(rung: str) -> str:
    return f"{GRAPH_SCHEMA_PREFIX}-{rung}-v1"


def _train_schema(rung: str) -> str:
    return f"{TRAIN_SCHEMA_PREFIX}-{rung}-v1"


def _configure_q2(rung: str, job: Mapping[str, Any], *, action: str) -> None:
    if rung not in {"quarter", "half"}:
        raise Round0187Error("nested execution rung changed")
    updates = 1
    if action == "train":
        graph = _read_sealed(
            str(job["graph_manifest"]), label=f"R0187 {rung} graph manifest"
        )
        updates = successful_updates_for_edges(int(graph["directed_edge_count"]))
    bindings = {
        "ROUND_ID": ROUND_ID,
        "CAPABILITY": CAPABILITY,
        "SUCCESSFUL_UPDATES": updates,
        "HOST_RSS_LIMIT_GIB": 56.0 if action == "graph" else 28.0,
        "Round0166Error": Round0187Error,
        "GRAPH_SCHEMA": _graph_schema(rung),
        "TRAIN_SCHEMA": _train_schema(rung),
        "PRODUCTION_CONFIG_SCHEMA": f"round0187-{rung}-production-config-v1",
        "GRAPH_INDEX_DESCRIPTION": GRAPH_INDEX_DESCRIPTION,
        "GRAPH_REFERENCE_ROW_ORDER": f"R0187 {rung} canonical nested order",
        "GRAPH_REFERENCE_ANCHOR_NAMESPACE": f"R0187 {rung} compact IDs",
        "GRAPH_SHARD_ROWS": 4_000_000,
        "GRAPH_SOURCE_ROUND_ID": ROUND_ID,
        "GRAPH_BUILT_IN_ROUND": True,
        "POPULATION_READER": _population_reader,
        "MIN_SCALE_ROWS_EXCLUSIVE": 0,
        "ScalePromptTrainingInput": NestedScalePromptTrainingInput,
        "GRAPH_EXTRA_REFERENCE_BUILDER": (
            _build_corpus_references if action == "graph" and rung == "quarter" else None
        ),
        "scale_train_config": (
            lambda **kwargs: train_config(rung=rung, **kwargs)
        ),
    }
    for name, value in bindings.items():
        setattr(q2, name, value)


def run_build_graph(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    rung = str(job.get("rung") or "")
    _configure_q2(rung, job, action="graph")
    q2.run_build_graph(active, job)


def run_train(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    rung = str(job.get("rung") or "")
    _configure_q2(rung, job, action="train")
    q2.run_train(active, job)


def _load_centroids(signatures: Mapping[str, Any], *, label: str) -> dict[int, np.ndarray]:
    return q2._centroids(signatures, label=label)


def _load_current_model(
    job: Mapping[str, Any], rung: str
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    population, population_signature = _population_reader(job)
    graph_path = str(job["graph_manifest"])
    graph_signature = _signature(graph_path, label=f"R0187 {rung} graph manifest")
    graph = _read_sealed(graph_path, label=f"R0187 {rung} graph manifest")
    train_path = os.path.join(str(job["train_output"]), "train-receipt.json")
    train_signature = _signature(train_path, label=f"R0187 {rung} train receipt")
    train = _read_sealed(train_path, label=f"R0187 {rung} train receipt")
    config, config_sha = train_config(
        rung=rung,
        graph_signature=graph["graph"],
        graph_manifest_signature=graph_signature,
        graph_edges=int(graph["directed_edge_count"]),
        retained_rows=int(population["retained_rows"]),
    )
    updates = successful_updates_for_edges(int(graph["directed_edge_count"]))
    expected_draws = updates * POSITIVE_ROWS_PER_UPDATE
    if (
        graph.get("schema") != _graph_schema(rung)
        or graph.get("round_id") != ROUND_ID
        or graph.get("population") != population_signature
        or train.get("schema") != _train_schema(rung)
        or train.get("round_id") != ROUND_ID
        or train.get("population") != population_signature
        or train.get("graph_manifest") != graph_signature
        or int(train.get("training_seed", -1)) != SEED
        or int(train.get("optimizer_updates", -1)) != updates
        or train.get("production_config_sha256") != config_sha
        or not train_checks_close(train.get("train_checks"))
        or int(train.get("consumed_positive_draws", -1)) != expected_draws
        or not np.isclose(
            float(train.get("requested_positive_draws_per_edge", float("nan"))),
            TARGET_POSITIVE_DRAWS_PER_EDGE,
            rtol=0,
            atol=1e-15,
        )
        or not np.isclose(
            float(train.get("consumed_positive_draws_per_edge", float("nan"))),
            expected_draws / int(graph["directed_edge_count"]),
            rtol=0,
            atol=1e-15,
        )
    ):
        raise Round0187Error(f"R0187 {rung} model receipt changed")
    model_path = prompt_contract.verify_signature(
        train["model"], label=f"R0187 {rung} model"
    )
    from basemap.pumap.parametric_umap import ParametricUMAP

    model = ParametricUMAP.load(model_path, device="cuda")
    if model.hidden_dim != 2048 or model.input_dim != DIMENSION or model.n_components != 2:
        raise Round0187Error(f"R0187 {rung} architecture changed")
    return model, train, train_signature


def _load_full_model(job: Mapping[str, Any]) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    train_path = str(job["full_train_receipt"])
    train_signature = _signature(train_path, label="accepted R0180 train receipt")
    train = _read_sealed(train_path, label="accepted R0180 train receipt")
    if (
        train.get("schema") != R0180_TRAIN_SCHEMA
        or train.get("round_id") != "0180"
        or int(train.get("training_seed", -1)) != SEED
        or int(train.get("optimizer_updates", -1)) != 2_026_478
        or not train_checks_close(train.get("train_checks"))
    ):
        raise Round0187Error("accepted R0180 train receipt changed")
    model_path = prompt_contract.verify_signature(train["model"], label="R0180 model")
    from basemap.pumap.parametric_umap import ParametricUMAP

    model = ParametricUMAP.load(model_path, device="cuda")
    if model.hidden_dim != 2048 or model.input_dim != DIMENSION or model.n_components != 2:
        raise Round0187Error("R0180 full architecture changed")
    return model, train, train_signature


def _load_pile_queries(job: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    path = str(job["pile_query_receipt"])
    signature = _signature(path, label="accepted R0171 Pile query receipt")
    receipt = _read_sealed(path, label="accepted R0171 Pile query receipt")
    if (
        receipt.get("schema") != PILE_QUERY_SCHEMA
        or receipt.get("round_id") != "0171"
        or receipt.get("candidate_canonical_range") != [8_000_000, 8_004_096]
        or int(receipt.get("selected_rows", -1)) != 2_000
        or receipt.get("selected_before_training") is not True
        or (receipt.get("training_copy_audit") or {}).get(
            "selected_exact_training_identity_disjoint"
        )
        is not True
    ):
        raise Round0187Error("accepted R0171 Pile query reserve changed")
    values = np.load(
        prompt_contract.verify_signature(receipt["queries"], label="R0171 Pile queries"),
        mmap_mode="r",
        allow_pickle=False,
    )
    if values.shape != (2_000, DIMENSION) or values.dtype != np.float16:
        raise Round0187Error("Pile query geometry changed")
    return values, receipt, signature


def _evaluate_model(
    active: Mapping[str, Any], job: Mapping[str, Any], *, rung: str, full: bool
) -> None:
    import torch
    from basemap.panel_v2 import (
        build_query_truth,
        load_hiD_reference,
        load_query_truth,
        save_query_truth,
        score_panel,
    )

    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0187Error("evaluation handler received another queue")
    if rung not in {"quarter", "half", "full"}:
        raise Round0187Error("evaluation rung changed")
    common_population, common_population_signature = _population_reader({
        **dict(job),
        "rung": "quarter",
        "population_receipt_path": job["common_population_receipt_path"],
    })
    common_graph_path = str(job["common_graph_manifest"])
    common_graph_signature = _signature(
        common_graph_path, label="R0187 quarter common graph manifest"
    )
    common_graph = _read_sealed(
        common_graph_path, label="R0187 quarter common graph manifest"
    )
    if (
        common_graph.get("schema") != _graph_schema("quarter")
        or common_graph.get("round_id") != ROUND_ID
        or common_graph.get("population") != common_population_signature
        or set(common_graph.get("comparison_references") or {}) != set(FULL_COUNTS)
    ):
        raise Round0187Error("common evaluation graph/reference contract changed")
    for key in ("graph", "high_d_reference"):
        prompt_contract.verify_signature(
            common_graph[key], label=f"common graph {key}"
        )
    for signature in (common_graph.get("centroids") or {}).values():
        prompt_contract.verify_signature(signature, label="common mixed centroid")

    model, train, train_signature = (
        _load_full_model(job) if full else _load_current_model(job, rung)
    )
    output = create_fresh_directory(
        str(job["outputs"][0]), label=f"R0187 {rung} common-core evaluation"
    )
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats("cuda")
    raw_path = prompt_contract.verify_signature(
        common_population["document_compact"], label="R0187 common prompted matrix"
    )
    raw = np.memmap(
        raw_path,
        mode="r",
        dtype="<f2",
        shape=(RUNG_ROWS["quarter"], DIMENSION),
    )
    source = L2NormalizedArray(raw)
    coordinates = np.asarray(model.transform(raw, batch_size=8192), dtype=np.float32)
    if coordinates.shape != (RUNG_ROWS["quarter"], 2) or not np.isfinite(
        coordinates
    ).all():
        raise Round0187Error(f"R0187 {rung} common coordinates are invalid")
    coordinates_path = os.path.join(output, "common-quarter-coordinates.npy")
    atomic_save_new_npy(coordinates_path, coordinates, immutable=True)
    cfg = prompt_contract.panel_config()
    mixed_reference = load_hiD_reference(
        prompt_contract.verify_signature(
            common_graph["high_d_reference"], label="common mixed high-D reference"
        ),
        expected_key=str(common_graph["high_d_reference_key"]),
    )
    mixed_centroids = _load_centroids(
        common_graph["centroids"], label="R0187 common mixed"
    )
    mixed_panel = score_panel(
        source,
        coordinates,
        config=cfg,
        centroids_by_k=mixed_centroids,
        hiD_reference=mixed_reference,
        reference_identity=common_graph["reference_identity"],
        scale_admission=None,
        provenance={
            "round_id": ROUND_ID,
            "rung": rung,
            "universe": "R0187-quarter-common-mixed-core",
            "population": common_population_signature,
            "train_receipt": train_signature,
            "coordinates": _signature(
                coordinates_path, label=f"R0187 {rung} common coordinates"
            ),
        },
    )
    corpus_panels: dict[str, Any] = {}
    for corpus, cell in common_graph["comparison_references"].items():
        start, stop = (int(value) for value in cell["compact_range"])
        corpus_raw = raw[start:stop]
        corpus_source = L2NormalizedArray(corpus_raw)
        reference = load_hiD_reference(
            prompt_contract.verify_signature(
                cell["high_d_reference"], label=f"R0187 {corpus} high-D reference"
            ),
            expected_key=str(cell["high_d_reference_key"]),
        )
        centroids = _load_centroids(
            cell["centroids"], label=f"R0187 common {corpus}"
        )
        corpus_panels[corpus] = score_panel(
            corpus_source,
            coordinates[start:stop],
            config=cfg,
            centroids_by_k=centroids,
            hiD_reference=reference,
            reference_identity=cell["reference_identity"],
            scale_admission=None,
            provenance={
                "round_id": ROUND_ID,
                "rung": rung,
                "universe": f"R0187-quarter-common-{corpus}-core",
                "population": common_population_signature,
                "compact_range": [start, stop],
                "train_receipt": train_signature,
            },
        )
        del corpus_raw, corpus_source, reference, centroids
        gc.collect()

    query_values, query_receipt, query_signature = _load_pile_queries(job)
    query_coordinates = np.asarray(
        model.transform(query_values, batch_size=8192), dtype=np.float32
    )
    query_coordinates_path = os.path.join(output, "pile-ood-query-coordinates.npy")
    atomic_save_new_npy(
        query_coordinates_path, query_coordinates, immutable=True
    )
    query_identity = {
        "schema": "round0187-pile-ood-query-identity-v1",
        "receipt": query_signature,
        "ordered_rows_sha256": query_receipt["ordered_canonical_rows_sha256"],
        "ordered_fp16_sha256": query_receipt["ordered_prompted_fp16_sha256"],
        "corpus": "Pile heldout tail beyond canonical first-8M training view",
    }
    if rung == "quarter":
        truth = build_query_truth(
            L2NormalizedArray(query_values),
            source,
            cfg=cfg,
            corpus_identity=common_graph["reference_identity"]["data_identity"],
            query_identity=query_identity,
            k=cfg.k_hit,
        )
        truth_path = os.path.join(output, "pile-ood-truth-k10.npz")
        save_query_truth(truth, truth_path)
        truth_signature = _signature(truth_path, label="R0187 Pile OOD truth")
    else:
        truth_path = os.path.join(
            str(job["quarter_evaluation_output"]), "pile-ood-truth-k10.npz"
        )
        truth_signature = _signature(
            truth_path, label="R0187 shared Pile OOD truth"
        )
        truth = load_query_truth(
            prompt_contract.verify_signature(
                truth_signature, label="R0187 shared Pile OOD truth"
            )
        )
    pile_ood = q2._projection_metrics(
        high10=np.asarray(truth["neighbors"], dtype=np.int64),
        query_coordinates=query_coordinates,
        coordinates=coordinates,
        cfg=cfg,
        truth_signature=truth_signature,
        truth_row_range=(0, len(query_values)),
    )
    metrics = primary_metric_view(
        mixed_panel=mixed_panel,
        corpus_panels=corpus_panels,
        pile_ood=pile_ood,
    )
    execution_checks = {
        "common_mixed_panel_finite_noncollapsed": q2._panel_execution_ok(mixed_panel),
        "all_corpus_panels_finite_noncollapsed": all(
            q2._panel_execution_ok(panel) for panel in corpus_panels.values()
        ),
        "pile_query_selected_before_training": query_receipt.get(
            "selected_before_training"
        )
        is True,
        "pile_query_exact_identity_disjoint_from_full_and_nested_training": (
            (query_receipt.get("training_copy_audit") or {}).get(
                "selected_exact_training_identity_disjoint"
            )
            is True
            and query_receipt.get("candidate_canonical_range") == [8_000_000, 8_004_096]
        ),
        "model_train_receipt_closes": train_checks_close(train.get("train_checks")),
    }
    if not all(execution_checks.values()):
        raise Round0187Error(f"R0187 {rung} evaluation checks failed")
    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
    if peak_rss_gib > 28.0:
        raise Round0187Error(f"R0187 {rung} evaluation exceeded 28 GiB RSS")
    receipt = prompt_contract.seal({
        "schema": EVALUATION_SCHEMA,
        "round_id": ROUND_ID,
        "rung": rung,
        "release_sha": active["manifest"]["release_sha"],
        "common_population": common_population_signature,
        "common_graph_manifest": common_graph_signature,
        "train_receipt": train_signature,
        "train_source_round": "0180" if full else ROUND_ID,
        "coordinates": _signature(
            coordinates_path, label=f"R0187 {rung} common coordinates"
        ),
        "mixed_panel": mixed_panel,
        "corpus_panels": corpus_panels,
        "pile_ood": pile_ood,
        "pile_query_receipt": query_signature,
        "pile_query_coordinates": _signature(
            query_coordinates_path, label=f"R0187 {rung} Pile query coordinates"
        ),
        "pile_query_truth": truth_signature,
        "primary_metrics": metrics,
        "diagnostic_metrics": {
            "mixed_density": float(mixed_panel["density"]),
            "mixed_projection_ffr": float(pile_ood["ffr"]),
            "corpus_density": {
                corpus: float(panel["density"])
                for corpus, panel in corpus_panels.items()
            },
        },
        "execution_checks": execution_checks,
        "evaluation_only": True,
        "training_performed_in_evaluation_node": False,
        "performance": {
            "wall_s": time.monotonic() - started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda")),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda")),
            "peak_host_rss_gib": peak_rss_gib,
        },
    })
    atomic_write_new_json(
        os.path.join(output, "common-core-evaluation.json"),
        receipt,
        immutable=True,
    )
    del (
        model,
        raw,
        source,
        coordinates,
        mixed_reference,
        mixed_centroids,
        query_coordinates,
        truth,
    )
    torch.cuda.empty_cache()
    gc.collect()


def run_evaluate(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    _evaluate_model(
        active,
        job,
        rung=str(job.get("rung") or ""),
        full=False,
    )


def run_evaluate_full(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    _evaluate_model(active, job, rung="full", full=True)


def run_synthesize(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0187Error("synthesis handler received another queue")
    cells: dict[str, dict[str, float]] = {}
    signatures: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, Any] = {}
    for rung in ("quarter", "half", "full"):
        path = os.path.join(
            str(job["evaluation_outputs"][rung]), "common-core-evaluation.json"
        )
        signature = _signature(path, label=f"R0187 {rung} evaluation")
        receipt = _read_sealed(path, label=f"R0187 {rung} evaluation")
        if (
            receipt.get("schema") != EVALUATION_SCHEMA
            or receipt.get("round_id") != ROUND_ID
            or receipt.get("rung") != rung
            or not all((receipt.get("execution_checks") or {}).values())
        ):
            raise Round0187Error(f"R0187 {rung} evaluation contract changed")
        cells[rung] = {
            key: float(value)
            for key, value in (receipt.get("primary_metrics") or {}).items()
        }
        signatures[rung] = signature
        diagnostics[rung] = receipt["diagnostic_metrics"]
    decision = ladder_decision(cells)
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0187 ladder synthesis"
    )
    receipt = prompt_contract.seal({
        "schema": SYNTHESIS_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capabilities": [CAPABILITY],
        "decision": decision,
        "evaluations": signatures,
        "diagnostic_metrics": diagnostics,
        "scientific_scope": {
            "composition_control": dict(RUNG_COUNTS),
            "nesting": "quarter subset half subset accepted R0165 full",
            "full_endpoint": "byte-exact accepted R0180 model",
            "seed": SEED,
            "hidden_dimension": 2048,
            "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
            "density_role": "diagnostic-only",
            "pile_ood_role": "genuine disjoint post-8M projection query reserve",
            "fineweb_redpajama_ood_limitation": (
                "no genuinely held-out prompted reserves exist in the accepted "
                "embedded first-view; per-corpus common cores are in-support"
            ),
            "optimizer_estimand": (
                "dose-matched complete cosine schedules; N changes the schedule "
                "horizon, so this is the deployable matched-dose scaling estimand"
            ),
        },
        "training_performed_in_synthesis_node": False,
    })
    atomic_write_new_json(
        os.path.join(output, "ladder-decision.json"), receipt, immutable=True
    )


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    action = str(job.get("action") or "")
    if action not in ALLOWED_ACTIONS:
        raise Round0187Error(f"R0187 does not authorize action {action!r}")
    actions = {
        "stage_nested_populations": run_stage_populations,
        "build_nested_graph": run_build_graph,
        "train_nested_rung": run_train,
        "evaluate_nested_rung": run_evaluate,
        "evaluate_full_endpoint": run_evaluate_full,
        "synthesize_nested_ladder": run_synthesize,
    }
    actions[action](active, job)


__all__ = ["run_job"]
