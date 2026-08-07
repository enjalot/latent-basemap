"""Execute the R0211 prompted-diverse U12 evaluation panel.

This is R0169's evaluation, adapted at exactly three points and nowhere else:

1. the OOD reserve is the sealed R0208 pack v2, and every language's probe is
   restricted to that pack's retained ordinals before scoring;
2. the model, graph, and train receipt come from separate sealed rounds
   (R0209 graph, R0210 low-dose train) rather than from this queue;
3. the verdict uses the R0207 memo's scale-relative decision structure, which
   demotes the R0161 English-2M absolute floors to descriptive cells.

Every scoring helper — the native panel, the matched-2M panel, the training
alignment, the probe scorer — is imported from the accepted implementations
rather than reimplemented.
"""
from __future__ import annotations

import gc
import os
import resource
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0105_search import GROUPS
from basemap.round0108_evaluation import IN_MIX_LANGUAGES, POLISH
from basemap.round0160_prompted_seed_family import metric_view
from basemap.round0169_prompted_diverse import HOST_RSS_LIMIT_GIB, ROWS
from basemap.round0209_prompted_diverse_graph import GRAPH_SCHEMA
from basemap.round0210_prompted_diverse_low_dose import (
    CAPABILITY as MODEL_CAPABILITY,
    TRAIN_SCHEMA,
    low_dose_train_config,
    successful_updates_for_edges,
)
from basemap.round0211_prompted_diverse_panel import (
    CAPABILITY,
    EVALUATION_SCHEMA,
    PACK_CAPABILITY,
    PACK_CORPUS_ROWS,
    PACK_QUERY_ROWS,
    PACK_ROWS,
    PACK_SCHEMA,
    ROUND_ID,
    Round0211Error,
    diverse_panel_decision,
)
from basemap import round0113_prompt_contrast as prompt_contract
from experiments import round0108_nodes as ood_nodes
from experiments import round0166_nodes as q2
from experiments import round0169_nodes as diverse


LANGUAGES = (*IN_MIX_LANGUAGES, POLISH)
PROMPT_PREFIX = "Document: "


def _signature(value: Any, *, label: str) -> dict[str, Any]:
    return dict(
        expected_input_signature(
            prompt_contract.verify_signature(value, label=label)
        )
    )


def _sealed_pack(job: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind the R0208 pack v2 and prove it is the repaired, disjoint reserve."""
    signature = _signature(job["ood_pack"], label="accepted R0208 OOD pack v2")
    pack = prompt_contract.read_sealed(
        signature["canonical_path"], label="accepted R0208 OOD pack v2"
    )
    audit = pack.get("audit") or {}
    shape = pack.get("shape") or {}
    policy = pack.get("repair_policy") or {}
    if (
        pack.get("schema") != PACK_SCHEMA
        or pack.get("round_id") != "0208"
        or pack.get("capability") != PACK_CAPABILITY
        or pack.get("capabilities") != [PACK_CAPABILITY]
        or pack.get("embedding_performed") is not False
        or pack.get("training_performed") is not False
        or audit.get("passed_after_repair") is not True
        or int(audit.get("source_row_identity_overlaps", -1)) != 0
        or int(shape.get("pack_rows", -1)) != PACK_ROWS
        or int(shape.get("corpus_rows_per_language", -1)) != PACK_CORPUS_ROWS
        or int(shape.get("query_rows_per_language", -1)) != PACK_QUERY_ROWS
        or tuple(shape.get("languages") or ()) != LANGUAGES
        or shape.get("held_out_language") != POLISH
        or policy.get("query_ids_unchanged_from_r0173") is not True
        or int(policy.get("rows_embedded", -1)) != 0
        or int(policy.get("rows_reselected", -1)) != 0
    ):
        raise Round0211Error("R0211 accepted R0208 OOD pack v2 changed")
    return pack, signature


def _retained_probe(
    pack: Mapping[str, Any], language: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load one language probe restricted to the pack's retained ordinals."""
    entry = (pack.get("languages") or {}).get(language) or {}
    source = entry.get("source_arrays") or {}
    retained = entry.get("retained") or {}
    if set(source) != {
        "corpus_embeddings",
        "query_embeddings",
        "corpus_source_rows",
        "query_source_rows",
    } or set(retained) != {"corpus", "queries"}:
        raise Round0211Error(f"R0211 {language} pack v2 binding changed")
    signatures = {
        key: _signature(value, label=f"R0208 {language} {key}")
        for key, value in source.items()
    }
    arrays = {
        key: np.load(value["canonical_path"], mmap_mode="r", allow_pickle=False)
        for key, value in signatures.items()
    }
    keep: dict[str, np.ndarray] = {}
    for split, rows in (("corpus", PACK_CORPUS_ROWS), ("queries", PACK_QUERY_ROWS)):
        cell = retained[split]
        ordinals = np.load(
            _signature(
                cell["ordinals"], label=f"R0208 {language} {split} retained ordinals"
            )["canonical_path"],
            allow_pickle=False,
        )
        if (
            ordinals.shape != (rows,)
            or ordinals.dtype != np.int64
            or int(cell.get("rows", -1)) != rows
            or not np.all(np.diff(ordinals) > 0)
            or int(ordinals[0]) < 0
        ):
            raise Round0211Error(f"R0211 {language} {split} retained ordinals changed")
        keep[split] = ordinals
    corpus = np.ascontiguousarray(
        np.asarray(arrays["corpus_embeddings"])[keep["corpus"]]
    )
    queries = np.ascontiguousarray(
        np.asarray(arrays["query_embeddings"])[keep["queries"]]
    )
    corpus_rows = np.ascontiguousarray(
        np.asarray(arrays["corpus_source_rows"])[keep["corpus"]], dtype=np.int64
    )
    query_rows = np.ascontiguousarray(
        np.asarray(arrays["query_source_rows"])[keep["queries"]], dtype=np.int64
    )
    if corpus.shape[0] != PACK_CORPUS_ROWS or queries.shape[0] != PACK_QUERY_ROWS:
        raise Round0211Error(f"R0211 {language} retained probe shape changed")
    return corpus, queries, corpus_rows, query_rows, {
        **signatures,
        "retained_corpus_ordinals": _signature(
            retained["corpus"]["ordinals"], label=f"R0208 {language} corpus ordinals"
        ),
        "retained_query_ordinals": _signature(
            retained["queries"]["ordinals"], label=f"R0208 {language} query ordinals"
        ),
    }


def _configure(updates: int) -> None:
    """Bind the accepted kernels to R0211's cross-round evidence."""
    diverse._configure_q2_kernel()
    bindings = {
        "ROUND_ID": "0210",
        "CAPABILITY": MODEL_CAPABILITY,
        "GRAPH_SCHEMA": GRAPH_SCHEMA,
        "GRAPH_SOURCE_ROUND_ID": "0209",
        "GRAPH_BUILT_IN_ROUND": False,
        "TRAIN_SCHEMA": TRAIN_SCHEMA,
        "SUCCESSFUL_UPDATES": updates,
        "scale_train_config": low_dose_train_config,
    }
    for name, value in bindings.items():
        setattr(q2, name, value)


def run_evaluate(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch
    from basemap.panel_v2 import load_hiD_reference, score_panel

    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0211Error("R0211 evaluation received another queue")
    graph_manifest = prompt_contract.read_sealed(
        str(job["graph_manifest"]), label="sealed R0209 graph"
    )
    updates = successful_updates_for_edges(int(graph_manifest["directed_edge_count"]))
    _configure(updates)

    population, population_signature = diverse._read_population(job)
    family, gates, floors = q2._read_family_and_gates(job)
    model, train, train_signature, graph = q2._authenticate_model(
        job, population, population_signature
    )
    if not diverse._graph_execution_ok(graph):
        raise Round0211Error("R0211 graph execution contract changed")
    pack, pack_signature = _sealed_pack(job)

    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0211 prompted diverse evaluation"
    )
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats("cuda")

    source_raw = diverse._open_source(population)
    source = diverse.PromptedDiverseScaleArray(
        source_raw,
        population=population,
        population_signature=population_signature,
    )
    coordinates = np.asarray(
        model.transform(source_raw, batch_size=8192), dtype=np.float32
    )
    if coordinates.shape != (ROWS, 2) or not np.isfinite(coordinates).all():
        raise Round0211Error("R0211 native transform output is invalid")
    coordinate_path = os.path.join(output, "native-u12-coordinates.npy")
    atomic_save_new_npy(coordinate_path, coordinates, immutable=True)

    group_signature = _signature(job["group_ids"], label="accepted R0132 group IDs")
    group_ids = np.load(
        group_signature["canonical_path"], mmap_mode="r", allow_pickle=False
    )
    if (
        group_ids.shape != (ROWS,)
        or group_ids.dtype != np.uint8
        or set(np.unique(group_ids).tolist()) != set(range(len(GROUPS)))
    ):
        raise Round0211Error("accepted R0132 group IDs changed")
    native_centroids = q2._centroids(graph["centroids"], label="R0211 native")
    native_reference = load_hiD_reference(
        prompt_contract.verify_signature(
            graph["high_d_reference"], label="R0211 native high-D reference"
        ),
        expected_key=str(graph["high_d_reference_key"]),
    )
    anchor_ids = np.asarray(native_reference["anchor_ids"], dtype=np.int64)
    anchor_groups = np.asarray(
        [GROUPS[int(value)] for value in group_ids[anchor_ids]], dtype="U80"
    )
    native_panel = score_panel(
        source,
        coordinates,
        config=prompt_contract.panel_config(),
        centroids_by_k=native_centroids,
        hiD_reference=native_reference,
        reference_identity=graph["reference_identity"],
        ffr_group_labels=anchor_groups,
        scale_admission=None,
        provenance={
            "round_id": ROUND_ID,
            "universe": "exact-r0132-u12-prompted",
            "population": population_signature,
            "group_ids": group_signature,
            "train_receipt": train_signature,
            "coordinates": expected_input_signature(coordinate_path),
        },
    )
    group_cells = native_panel.get("ffr_by_group") or {}
    if set(group_cells) != set(GROUPS) or any(
        int(group_cells[name].get("anchors", 0)) <= 0 for name in GROUPS
    ):
        raise Round0211Error("R0211 native group FFR cells are incomplete")
    group_ffr = {name: float(group_cells[name]["ffr"]) for name in GROUPS}

    native_alignment = diverse._native_training_alignment(
        model=model,
        source=source_raw,
        coordinates=coordinates,
        language_outputs=job["language_outputs"],
        output=output,
    )
    native_metrics = metric_view(
        panel=native_panel,
        native_score={
            "projections": {"matched": native_alignment["in_mix_balanced_500"]}
        },
    )
    matched = diverse._matched_2m_panel(
        model=model,
        family=family,
        train_signature=train_signature,
        output=output,
    )

    ood_reports: dict[str, Any] = {}
    for language in LANGUAGES:
        corpus, queries, corpus_rows, query_rows, signatures = _retained_probe(
            pack, language
        )
        ood_reports[language] = ood_nodes._probe_score(
            name=f"prompted-{language}",
            corpus=corpus,
            queries=queries,
            corpus_ids=corpus_rows,
            query_ids=1_000_000_000 + query_rows,
            model=model,
            output=output,
            inputs={
                **signatures,
                "prompt_applied": True,
                "prompt_prefix": PROMPT_PREFIX,
                "training_disjoint_pack": pack_signature,
            },
            save_coordinates=True,
            duplicate_policy="require-corpus-query-exact-family-disjoint",
        )
        del corpus, queries
        gc.collect()

    in_mix_recall50 = [
        float(ood_reports[language]["probe"]["recall_at_50_of_high10"])
        for language in IN_MIX_LANGUAGES
    ]
    prompted_ood = {
        "polish_recall_at_50_of_high10": float(
            ood_reports[POLISH]["probe"]["recall_at_50_of_high10"]
        ),
        "in_mix_median_recall_at_50_of_high10": float(np.median(in_mix_recall50)),
    }
    raw_signature = _signature(job["raw_r0132_ood"], label="accepted R0132 OOD")
    raw = prompt_contract.read_sealed(
        raw_signature["canonical_path"], label="accepted R0132 OOD"
    )
    raw_control = raw.get("control_12p5m") or {}
    if (
        raw.get("schema") != "round0132-matched-ood-scale-panel-v1"
        or raw.get("round_id") != "0132"
        or set(raw_control)
        != {
            "fineweb_recall_at_50_of_high10",
            "in_mix_median_recall_at_50_of_high10",
            "polish_recall_at_50_of_high10",
        }
    ):
        raise Round0211Error("accepted R0132 OOD control changed")
    raw_ood = {
        name: float(raw_control[name])
        for name in (
            "polish_recall_at_50_of_high10",
            "in_mix_median_recall_at_50_of_high10",
        )
    }

    decision = diverse_panel_decision(
        native=native_metrics,
        matched_2m=matched["decision_metrics"],
        baseline_2m_seed42=matched["baseline_seed42_metrics"],
        prompted_floors=floors,
        group_ffr=group_ffr,
        prompted_ood=prompted_ood,
        raw_r0132_ood=raw_ood,
    )
    execution_gates = {
        "train_receipt_closes": all(
            bool(value) for value in (train.get("train_checks") or {}).values()
        ),
        "graph_fixed_nprobe_qualified": (
            ((graph.get("search_qualification") or {}).get("cells") or {})
            .get(str(graph["search_qualification"]["selected_nprobe"]), {})
            .get("passed")
            is True
        ),
        "graph_uses_registered_sharded_fp32_execution": diverse._graph_execution_ok(
            graph
        ),
        "ood_reserve_repaired_and_exactly_disjoint": (
            (pack.get("audit") or {}).get("passed_after_repair") is True
        ),
        "native_panel_finite_noncollapsed": q2._panel_execution_ok(native_panel),
        "matched_panel_finite_noncollapsed": q2._panel_execution_ok(matched["panel"]),
        "all_twenty_ood_cells_complete": set(ood_reports) == set(LANGUAGES),
        "low_dose_horizon_matches_sealed_graph": (
            int(train["train_accounting"]["optimizer_steps_succeeded"]) == updates
        ),
    }
    passed = bool(decision["passed"] and all(execution_gates.values()))
    decision = {
        **decision,
        "metric_gates_passed": bool(decision["passed"]),
        "execution_gates": execution_gates,
        "passed": passed,
        "outcome": (
            decision["outcome"]
            if passed
            else "prompted-diverse-u12-low-dose-rung-retention-not-qualified"
        ),
    }
    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    if peak_rss_gib > HOST_RSS_LIMIT_GIB:
        raise Round0211Error(
            f"R0211 evaluation peak RSS {peak_rss_gib:.2f} GiB exceeds "
            f"{HOST_RSS_LIMIT_GIB:.0f} GiB"
        )
    receipt = prompt_contract.seal({
        "schema": EVALUATION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capabilities": [CAPABILITY] if passed else [],
        "decision": decision,
        "population": population_signature,
        "group_ids": group_signature,
        "ood_probe_pack_v2": pack_signature,
        "graph_manifest": expected_input_signature(str(job["graph_manifest"])),
        "train_receipt": train_signature,
        "low_dose": {
            "sealed_directed_edges": int(graph_manifest["directed_edge_count"]),
            "successful_positive_lr_updates": updates,
        },
        "prompted_gate_registration": dict(job["gate_registration"]),
        "prompted_seed_family": dict(job["family_evidence"]),
        "native_u12": {
            "coordinates": expected_input_signature(coordinate_path),
            "panel": native_panel,
            "group_ffr": group_ffr,
            "training_alignment": native_alignment,
            "decision_metrics": native_metrics,
            "projection_metrics_role": "diagnostic-only",
            "density_v2_role": "diagnostic-only, transcribed",
        },
        "matched_2m": matched,
        "prompted_ood": {
            "summary": prompted_ood,
            "language_cells": ood_reports,
            "raw_r0132_control": raw_ood,
            "raw_r0132_evidence": raw_signature,
            "projection_ffr_role": "diagnostic-only",
            "reserve": "repaired, sealed R0208 pack v2 retained ordinals only",
        },
        "prompted_english_2m_reference_floors": floors,
        "training_performed_in_round": False,
        "graph_built_in_round": False,
        "atlas_quality_claim_available": False,
        "performance": {
            "evaluation_wall_s": time.monotonic() - started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda")),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda")),
            "peak_host_rss_gib": peak_rss_gib,
        },
    })
    atomic_write_new_json(
        os.path.join(output, "scale-evaluation.json"), receipt, immutable=True
    )
    del model, source_raw, source, coordinates, native_centroids, native_reference
    torch.cuda.empty_cache()
    gc.collect()


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if str(job.get("action") or "") != "evaluate_prompted_diverse_u12_low_dose":
        raise Round0211Error("R0211 authorizes only the diverse evaluation panel")
    run_evaluate(active, job)


__all__ = ["run_evaluate", "run_job"]
