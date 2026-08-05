"""Execute the R0188 seed-43 half-to-full composition-boundary replay."""
from __future__ import annotations

import gc
import os
import resource
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_save_new_npy, atomic_write_new_json, create_fresh_directory
from basemap.round0104_training import L2NormalizedArray
from basemap.round0187_composition_nested_ladder import (
    DIMENSION,
    FULL_COUNTS,
    PRIMARY_METRICS,
    RUNG_ROWS,
)
from basemap.round0188_composition_boundary_seed43 import (
    CAPABILITY,
    EVALUATION_SCHEMA,
    ROUND_ID,
    RUNGS,
    SEED,
    SYNTHESIS_SCHEMA,
    Round0188Error,
    boundary_decision,
    successful_updates_for_edges,
    train_checks_close,
    train_config,
    train_schema,
)
from basemap import round0113_prompt_contrast as prompt_contract
from experiments import round0166_nodes as q2
from experiments import round0180_nodes as r0180_nodes
from experiments import round0187_nodes as r0187_nodes


GRAPH_META = {
    "half": {
        "schema": "round0187-composition-nested-fuzzy-graph-half-v1",
        "source_round": "0187",
        "index": r0187_nodes.GRAPH_INDEX_DESCRIPTION,
        "row_order": "R0187 half canonical nested order",
        "anchor_namespace": "R0187 half compact IDs",
    },
    "full": {
        "schema": "round0171-prompted-8m-fuzzy-graph-v1",
        "source_round": "0171",
        "index": r0180_nodes.GRAPH_INDEX_DESCRIPTION,
        "row_order": "R0165 frozen-prefix prompted compact order",
        "anchor_namespace": "R0165 compact IDs",
    },
}
COMMON_GRAPH_SCHEMA = "round0187-composition-nested-fuzzy-graph-quarter-v1"
COMMON_EVALUATION_SCHEMA = (
    "round0187-composition-nested-common-core-evaluation-v1"
)
ALLOWED_ACTIONS = {
    "train_seed43_boundary_rung",
    "evaluate_seed43_boundary_rung",
    "synthesize_seed43_boundary",
}


def _signature(path: str, *, label: str) -> dict[str, Any]:
    try:
        return expected_input_signature(path)
    except Exception as error:
        raise Round0188Error(f"{label} is unavailable or changed") from error


def _read_sealed(path: str, *, label: str) -> dict[str, Any]:
    try:
        return prompt_contract.read_sealed(path, label=label)
    except Exception as error:
        raise Round0188Error(f"{label} seal is invalid") from error


def _half_population_reader(
    job: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if str(job.get("rung") or "") != "half":
        raise Round0188Error("half population reader received another rung")
    return r0187_nodes._population_reader(job)


def _configure_q2(rung: str, job: Mapping[str, Any]) -> None:
    if rung not in RUNGS:
        raise Round0188Error("training rung changed")
    graph = _read_sealed(
        str(job["graph_manifest"]), label=f"accepted {rung} graph manifest"
    )
    updates = successful_updates_for_edges(int(graph.get("directed_edge_count", -1)))
    meta = GRAPH_META[rung]
    bindings = {
        "ROUND_ID": ROUND_ID,
        "CAPABILITY": CAPABILITY,
        "SEED": SEED,
        "SUCCESSFUL_UPDATES": updates,
        "HOST_RSS_LIMIT_GIB": 28.0,
        "Round0166Error": Round0188Error,
        "GRAPH_SCHEMA": meta["schema"],
        "TRAIN_SCHEMA": train_schema(rung),
        "PRODUCTION_CONFIG_SCHEMA": f"round0188-{rung}-seed43-production-config-v1",
        "GRAPH_INDEX_DESCRIPTION": meta["index"],
        "GRAPH_REFERENCE_ROW_ORDER": meta["row_order"],
        "GRAPH_REFERENCE_ANCHOR_NAMESPACE": meta["anchor_namespace"],
        "GRAPH_SOURCE_ROUND_ID": meta["source_round"],
        "GRAPH_BUILT_IN_ROUND": False,
        "POPULATION_READER": _half_population_reader if rung == "half" else None,
        "MIN_SCALE_ROWS_EXCLUSIVE": 0,
        "ScalePromptTrainingInput": r0187_nodes.NestedScalePromptTrainingInput,
        "scale_train_config": (
            lambda **kwargs: train_config(rung=rung, **kwargs)
        ),
    }
    for name, value in bindings.items():
        setattr(q2, name, value)


def run_train(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0188Error("train handler received another queue")
    rung = str(job.get("rung") or "")
    _configure_q2(rung, job)
    q2.run_train(active, job)


def _training_population(
    job: Mapping[str, Any], rung: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    _configure_q2(rung, job)
    return q2._read_population(job)


def _load_seed43_model(
    job: Mapping[str, Any], rung: str
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    population, population_signature = _training_population(job, rung)
    graph_path = str(job["graph_manifest"])
    graph_signature = _signature(graph_path, label=f"accepted {rung} graph manifest")
    graph = _read_sealed(graph_path, label=f"accepted {rung} graph manifest")
    meta = GRAPH_META[rung]
    fixed = ((graph.get("search_qualification") or {}).get("cells") or {}).get(
        "64", {}
    )
    if (
        graph.get("schema") != meta["schema"]
        or graph.get("round_id") != meta["source_round"]
        or graph.get("population") != population_signature
        or int(graph.get("retained_rows", -1)) != RUNG_ROWS[rung]
        or int(graph.get("dimension", -1)) != DIMENSION
        or int(graph.get("k", -1)) != 50
        or int(graph.get("directed_edge_count", -1)) <= 0
        or int((graph.get("search_qualification") or {}).get("selected_nprobe", -1))
        != 64
        or fixed.get("passed") is not True
        or (graph.get("search_qualification") or {}).get("index") != meta["index"]
    ):
        raise Round0188Error(f"accepted {rung} graph contract changed")
    train_path = os.path.join(str(job["train_output"]), "train-receipt.json")
    train_signature = _signature(train_path, label=f"R0188 {rung} train receipt")
    train = _read_sealed(train_path, label=f"R0188 {rung} train receipt")
    config, config_sha = train_config(
        rung=rung,
        graph_signature=graph["graph"],
        graph_manifest_signature=graph_signature,
        graph_edges=int(graph["directed_edge_count"]),
        retained_rows=RUNG_ROWS[rung],
    )
    updates = successful_updates_for_edges(int(graph["directed_edge_count"]))
    expected_draws = updates * prompt_contract.POSITIVE_ROWS_PER_UPDATE
    if (
        train.get("schema") != train_schema(rung)
        or train.get("round_id") != ROUND_ID
        or int(train.get("training_seed", -1)) != SEED
        or train.get("population") != population_signature
        or train.get("graph_manifest") != graph_signature
        or train.get("production_config_sha256") != config_sha
        or int(train.get("optimizer_updates", -1)) != updates
        or int(train.get("consumed_positive_draws", -1)) != expected_draws
        or not train_checks_close(train.get("train_checks"))
    ):
        raise Round0188Error(f"R0188 {rung} train receipt changed")
    model_path = prompt_contract.verify_signature(
        train["model"], label=f"R0188 {rung} model"
    )
    from basemap.pumap.parametric_umap import ParametricUMAP

    model = ParametricUMAP.load(model_path, device="cuda")
    expected = config["model"]
    observed = {
        "architecture": model.architecture,
        "input_dimension": model.input_dim,
        "hidden_dimension": model.hidden_dim,
        "hidden_layers": model.n_layers,
        "output_dimension": model.n_components,
        "use_batchnorm": model.use_batchnorm,
        "use_dropout": model.use_dropout,
        "low_dim_kernel": model.low_dim_kernel,
        "a": model.a,
        "b": model.b,
    }
    if observed != expected:
        raise Round0188Error(f"R0188 {rung} model architecture changed")
    return model, train, train_signature


def _common_inputs(
    job: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    population, population_signature = r0187_nodes._population_reader({
        **dict(job),
        "rung": "quarter",
        "population_receipt_path": job["common_population_receipt_path"],
    })
    graph_path = str(job["common_graph_manifest"])
    graph_signature = _signature(graph_path, label="R0187 common graph manifest")
    graph = _read_sealed(graph_path, label="R0187 common graph manifest")
    if (
        graph.get("schema") != COMMON_GRAPH_SCHEMA
        or graph.get("round_id") != "0187"
        or graph.get("population") != population_signature
        or set(graph.get("comparison_references") or {}) != set(FULL_COUNTS)
    ):
        raise Round0188Error("R0187 common graph/reference contract changed")
    for key in ("graph", "high_d_reference"):
        prompt_contract.verify_signature(graph[key], label=f"common graph {key}")
    for signature in (graph.get("centroids") or {}).values():
        prompt_contract.verify_signature(signature, label="common mixed centroid")
    return population, population_signature, graph, graph_signature


def run_evaluate(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch
    from basemap.panel_v2 import load_hiD_reference, load_query_truth, score_panel

    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0188Error("evaluation handler received another queue")
    rung = str(job.get("rung") or "")
    if rung not in RUNGS:
        raise Round0188Error("evaluation rung changed")
    common_population, common_population_signature, common_graph, common_graph_signature = (
        _common_inputs(job)
    )
    model, train, train_signature = _load_seed43_model(job, rung)
    output = create_fresh_directory(
        str(job["outputs"][0]), label=f"R0188 {rung} common-core evaluation"
    )
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats("cuda")
    raw_path = prompt_contract.verify_signature(
        common_population["document_compact"], label="R0187 common prompted matrix"
    )
    raw = np.memmap(
        raw_path, mode="r", dtype="<f2", shape=(RUNG_ROWS["quarter"], DIMENSION)
    )
    source = L2NormalizedArray(raw)
    coordinates = np.asarray(model.transform(raw, batch_size=8192), dtype=np.float32)
    if coordinates.shape != (RUNG_ROWS["quarter"], 2) or not np.isfinite(
        coordinates
    ).all():
        raise Round0188Error(f"R0188 {rung} common coordinates are invalid")
    coordinates_path = os.path.join(output, "common-quarter-coordinates.npy")
    atomic_save_new_npy(coordinates_path, coordinates, immutable=True)
    cfg = prompt_contract.panel_config()
    mixed_reference = load_hiD_reference(
        prompt_contract.verify_signature(
            common_graph["high_d_reference"], label="common mixed high-D reference"
        ),
        expected_key=str(common_graph["high_d_reference_key"]),
    )
    mixed_centroids = r0187_nodes._load_centroids(
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
        centroids = r0187_nodes._load_centroids(
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

    query_values, query_receipt, query_signature = r0187_nodes._load_pile_queries(job)
    query_coordinates = np.asarray(
        model.transform(query_values, batch_size=8192), dtype=np.float32
    )
    query_coordinates_path = os.path.join(output, "pile-ood-query-coordinates.npy")
    atomic_save_new_npy(query_coordinates_path, query_coordinates, immutable=True)
    prior_evaluation_path = str(job["r0187_quarter_evaluation"])
    prior_evaluation = _read_sealed(
        prior_evaluation_path, label="accepted R0187 quarter evaluation"
    )
    truth_signature = _signature(
        str(job["shared_truth_path"]), label="accepted R0187 Pile OOD truth"
    )
    if (
        prior_evaluation.get("schema") != COMMON_EVALUATION_SCHEMA
        or prior_evaluation.get("round_id") != "0187"
        or prior_evaluation.get("rung") != "quarter"
        or prior_evaluation.get("pile_query_truth") != truth_signature
    ):
        raise Round0188Error("accepted R0187 shared query truth binding changed")
    truth = load_query_truth(
        prompt_contract.verify_signature(
            truth_signature, label="accepted R0187 shared Pile OOD truth"
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
    metrics = r0187_nodes.primary_metric_view(
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
        "pile_query_exact_identity_disjoint_from_training": (
            (query_receipt.get("training_copy_audit") or {}).get(
                "selected_exact_training_identity_disjoint"
            )
            is True
            and query_receipt.get("candidate_canonical_range")
            == [8_000_000, 8_004_096]
        ),
        "model_train_receipt_closes": train_checks_close(train.get("train_checks")),
        "shared_truth_is_byte_exact_r0187_quarter_truth": True,
    }
    if not all(execution_checks.values()):
        raise Round0188Error(f"R0188 {rung} evaluation checks failed")
    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
    if peak_rss_gib > 28.0:
        raise Round0188Error(f"R0188 {rung} evaluation exceeded 28 GiB RSS")
    receipt = prompt_contract.seal({
        "schema": EVALUATION_SCHEMA,
        "round_id": ROUND_ID,
        "rung": rung,
        "seed": SEED,
        "release_sha": active["manifest"]["release_sha"],
        "common_population": common_population_signature,
        "common_graph_manifest": common_graph_signature,
        "train_receipt": train_signature,
        "coordinates": _signature(
            coordinates_path, label=f"R0188 {rung} common coordinates"
        ),
        "mixed_panel": mixed_panel,
        "corpus_panels": corpus_panels,
        "pile_ood": pile_ood,
        "pile_query_receipt": query_signature,
        "pile_query_coordinates": _signature(
            query_coordinates_path, label=f"R0188 {rung} Pile query coordinates"
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
        os.path.join(output, "common-core-evaluation.json"), receipt, immutable=True
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


def run_synthesize(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0188Error("synthesis handler received another queue")
    prior_path = str(job["r0187_ladder_decision"])
    prior_signature = _signature(prior_path, label="accepted R0187 ladder decision")
    prior = _read_sealed(prior_path, label="accepted R0187 ladder decision")
    prior_decision = prior.get("decision") or {}
    if (
        prior.get("schema") != "round0187-composition-nested-ladder-decision-v1"
        or prior.get("round_id") != "0187"
        or prior_decision.get("outcome") != "composition-controlled-size-regression"
        or (prior_decision.get("concordant_material_regression") or {}).get(
            "pile_ffr"
        )
        is not True
    ):
        raise Round0188Error("accepted R0187 decision branch changed")
    seed43: dict[str, dict[str, float]] = {}
    evaluation_signatures: dict[str, dict[str, Any]] = {}
    for rung in RUNGS:
        path = os.path.join(
            str(job["evaluation_outputs"][rung]), "common-core-evaluation.json"
        )
        signature = _signature(path, label=f"R0188 {rung} evaluation")
        receipt = _read_sealed(path, label=f"R0188 {rung} evaluation")
        if (
            receipt.get("schema") != EVALUATION_SCHEMA
            or receipt.get("round_id") != ROUND_ID
            or receipt.get("rung") != rung
            or int(receipt.get("seed", -1)) != SEED
            or not all((receipt.get("execution_checks") or {}).values())
        ):
            raise Round0188Error(f"R0188 {rung} evaluation contract changed")
        seed43[rung] = {
            key: float(value)
            for key, value in (receipt.get("primary_metrics") or {}).items()
        }
        evaluation_signatures[rung] = signature
    seed42 = {
        rung: {
            key: float(value)
            for key, value in prior_decision["cells"][rung].items()
        }
        for rung in RUNGS
    }
    decision = boundary_decision(seed42=seed42, seed43=seed43)
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0188 seed-43 boundary synthesis"
    )
    receipt = prompt_contract.seal({
        "schema": SYNTHESIS_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capabilities": [CAPABILITY],
        "decision": decision,
        "r0187_ladder_decision": prior_signature,
        "evaluations": evaluation_signatures,
        "scientific_scope": {
            "boundary": "half_to_full",
            "registered_metric": "pile_ffr",
            "seed42_source": "accepted R0187",
            "seed43": SEED,
            "population_graph_and_dose_reused": True,
            "hidden_dimension": 2048,
            "primary_metric_vector": list(PRIMARY_METRICS),
            "other_seed43_metric_misses_role": "diagnostic-only",
        },
        "training_performed_in_synthesis_node": False,
    })
    atomic_write_new_json(
        os.path.join(output, "boundary-decision.json"), receipt, immutable=True
    )


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    action = str(job.get("action") or "")
    if action not in ALLOWED_ACTIONS:
        raise Round0188Error(f"R0188 does not authorize action {action!r}")
    actions = {
        "train_seed43_boundary_rung": run_train,
        "evaluate_seed43_boundary_rung": run_evaluate,
        "synthesize_seed43_boundary": run_synthesize,
    }
    actions[action](active, job)


__all__ = ["run_job"]
