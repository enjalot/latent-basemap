#!/usr/bin/env python3
"""Prepare the preregistered diverse-Jina atlas evaluation queue."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_save_new_npz,
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0104_training import validate_substrate_manifest
from basemap.round0105_search import (
    ELIGIBILITY_PATH,
    GROUPS,
    group_ranges,
    sample_stratified_rows,
)
from basemap.round0106_graph import PARTS, global_to_compact
from basemap.round0108_evaluation import (
    ANCHORS_PER_GROUP,
    CALIBRATION_BOOTSTRAP_DRAWS,
    CALIBRATION_BOOTSTRAP_SEED,
    CALIBRATION_NULL_DRAWS,
    CALIBRATION_NULL_SEED,
    EMBEDDING_PROMPT,
    HELDOUT_REPLACEMENT_RESERVE_ROWS,
    HELDOUT_REPLACEMENT_SEED_OFFSET,
    HELDOUT_SEED,
    IN_MIX_LANGUAGES,
    MAP_KEY,
    POLISH,
    ROUND_ID,
    exact_reference_copy_mask,
    fixed_probe_split,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0108"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0108-*.md")

R0040_CENSUS_RECEIPT = (
    "/data/latent-basemap/runs/round-0040/queue/artifacts/"
    "jina-census/receipt.json"
)
R0040_REFERENCE = (
    "/data/latent-basemap/runs/round-0040/queue/artifacts/"
    "jina-rescore/representative-high-d-reference.npz"
)
R0037_COORDINATES = (
    "/data/latent-basemap/runs/round-0037/queue/artifacts/"
    "d768_s42/transform/coordinates.npy"
)
R0038_COORDINATES = (
    "/data/latent-basemap/runs/round-0038/queue/artifacts/"
    "d768_s43/transform/coordinates.npy"
)
INVENTORY_PATH = (
    "/data/latent-basemap/runs/round-0087/queue/artifacts/"
    "jina-diverse-25m-inventory/jina-diverse-25m-inventory-v1.json"
)
LABELS_PATH = (
    "/data/latent-basemap/runs/round-0103/queue/artifacts/"
    "jina-diverse-25m-full768-int8-substrate/labels.npz"
)
GRAPH_MANIFEST = (
    "/data/latent-basemap/runs/round-0106/queue-attempt-3/artifacts/"
    "canonical-fuzzy-graph/graph-manifest.json"
)
TRAIN_OUTPUT = (
    "/data/latent-basemap/runs/round-0107/queue/artifacts/"
    "train-diverse-jina-25m"
)
PART_OUTPUTS = {
    part: (
        "/data/latent-basemap/runs/round-0106/queue-attempt-3/artifacts/"
        f"graph-part-{part}"
    )
    for part in PARTS
}

DIAGNOSTIC_PATHS = {
    "dadabase": "/data/embeddings/dadabase/jina-v5-nano.npy",
    "dadabase_texts": "/data/embeddings/dadabase/jokes.parquet",
    "trec_corpus": (
        "/data/embeddings/beir/trec-covid-pooled-jina-v5-nano/"
        "corpus_vectors.npy"
    ),
    "trec_queries": (
        "/data/embeddings/beir/trec-covid-pooled-jina-v5-nano/"
        "query_vectors.npy"
    ),
    "trec_corpus_ids": (
        "/data/embeddings/beir/trec-covid-pooled-jina-v5-nano/"
        "corpus_ids.json"
    ),
    "trec_query_ids": (
        "/data/embeddings/beir/trec-covid-pooled-jina-v5-nano/"
        "query_ids.json"
    ),
    "fineweb": (
        "/data/embeddings/fineweb-edu-sample-10BT-chunked-500-"
        "jina-v5-nano-heldout/train/data-00000.npy"
    ),
}
POLISH_PATH = (
    "/data/embeddings/fineweb2-pol_Latn-chunked-500-jina-v5-nano/"
    "train/000_00000.npy"
)

REVIEW_DEFAULTS = {
    "0037": (
        "review-0037-2026-07-23.md",
        "8192d5478c63c1e961283c398370619144bfa97828aabcecbbd56ed7fbdb39a1",
        "capability:jina-mrl-seed42-screen-v1",
    ),
    "0038": (
        "review-0038-2026-07-24.md",
        "fdafdb50286526e6a8a491f4a281c0a95967dc8e4238d8fd270d52b28798cc78",
        "capability:jina-mrl-two-seed-decision-v1",
    ),
    "0040": (
        "review-0040-2026-07-24.md",
        "bdd78c58550f36239e6cc0858fb0da21bf1b8d6f2e1990b4dffa578d2f9a25ba",
        "capability:duplicate-controlled-panel-v1",
    ),
    "0085": (
        "review-0085-2026-07-28.md",
        "d91ea09d4909297d3048e8c8075b65f6f010fe8a5451bcf3684f2564f6bc1143",
        "capability:minilm-density-v2-calibration-v1",
    ),
    "0087": (
        "review-0087-2026-07-28.md",
        "61ab9268899c2edc47519bdbe4efeea65a54f0c9fda52bd89e7cad0dafd9d483",
        "capability:jina-diverse-25m-inventory-v1",
    ),
    "0103": (
        "review-0103-2026-07-29.md",
        "c6c4f780c20cc34c7707132581ffaaf8daa8cc7ea9eb1cee3f76e128b6c37a51",
        "capability:jina-diverse-25m-full768-int8-substrate-v1",
    ),
}


def _frontmatter_status(path: str) -> str | None:
    with open(path, encoding="utf-8") as handle:
        text = handle.read(4_096)
    match = re.search(r"(?m)^status:\s*[\"']?([^\s\"']+)", text)
    return match.group(1) if match else None


def _require_issued_round() -> str:
    candidates = [
        path
        for path in sorted(glob.glob(ROUND_FILE_GLOB))
        if _frontmatter_status(path) == "issued"
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"R0108 requires exactly one issued round; found {len(candidates)}"
        )
    return candidates[0]


def _require_review(
    path: str,
    *,
    expected_sha256: str,
    capability: str,
) -> dict[str, Any]:
    if _frontmatter_status(path) != "accepted":
        raise RuntimeError(f"{path} is not an accepted review")
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise RuntimeError(f"{path} bytes changed")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if capability not in text:
        raise RuntimeError(f"{path} lacks {capability}")
    return signature


def _load_inventory() -> tuple[dict[str, Any], dict[str, Any]]:
    signature = expected_input_signature(INVENTORY_PATH)
    with open(INVENTORY_PATH, encoding="utf-8") as handle:
        inventory = json.load(handle)
    if (
        inventory.get("schema") != "jina-diverse-25m-inventory-v1"
        or int((inventory.get("selection") or {}).get("selected_rows", -1))
        != 25_000_000
    ):
        raise RuntimeError("R0087 inventory content changed")
    return inventory, signature


def _language_sources(
    inventory: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    ranges = (inventory.get("selection") or {}).get("ranges") or []
    sources: dict[str, dict[str, Any]] = {}
    selected_stops: dict[str, int] = {}
    for item in ranges:
        language = item.get("language")
        if language not in IN_MIX_LANGUAGES:
            continue
        signature = dict(item["shard"])
        signature.pop("rows", None)
        signature.setdefault("kind", "file")
        sources[str(language)] = signature
        selected_stops[str(language)] = int(item["dataset_row_stop"])
    if set(sources) != set(IN_MIX_LANGUAGES):
        raise RuntimeError("R0087 language sources are incomplete")
    polish = expected_input_signature(POLISH_PATH)
    sources[POLISH] = polish
    selected_stops[POLISH] = 0
    return sources, selected_stops


def _selection_artifact(
    path: str,
    *,
    inventory: dict[str, Any],
    substrate: dict[str, Any],
    language_sources: dict[str, dict[str, Any]],
    selected_stops: dict[str, int],
) -> dict[str, Any]:
    with np.load(ELIGIBILITY_PATH, allow_pickle=False) as archive:
        excluded = np.asarray(archive["excluded_rows"], dtype=np.int64)
    ranges = group_ranges(substrate["manifest"])
    global_rows, group_ids = sample_stratified_rows(
        excluded,
        ranges,
        rows_per_group=ANCHORS_PER_GROUP,
        seed=108,
    )
    compact_rows = global_to_compact(global_rows, excluded)
    arrays: dict[str, np.ndarray] = {
        "core_global_rows": global_rows,
        "core_compact_rows": compact_rows,
        "core_group_ids": group_ids,
    }
    for index, language in enumerate((*IN_MIX_LANGUAGES, POLISH)):
        source = np.load(
            language_sources[language]["canonical_path"],
            mmap_mode="r",
            allow_pickle=False,
        )
        start = selected_stops[language]
        corpus, queries = fixed_probe_split(
            row_start=start,
            row_stop=len(source),
            seed=HELDOUT_SEED + index,
        )
        original_queries = queries.copy()
        replacement_mask = np.zeros(len(queries), dtype=bool)
        if start:
            query_values = np.asarray(source[queries])
            training_copies, _training_audit = exact_reference_copy_mask(
                source[:start], query_values
            )
            corpus_copies, _corpus_audit = exact_reference_copy_mask(
                np.asarray(source[corpus]), query_values
            )
            replacement_mask = training_copies | corpus_copies
            needed = int(replacement_mask.sum())
            if needed:
                span = len(source) - start
                reserve_count = min(HELDOUT_REPLACEMENT_RESERVE_ROWS, span)
                reserve_rng = np.random.RandomState(
                    HELDOUT_REPLACEMENT_SEED_OFFSET + HELDOUT_SEED + index
                )
                reserve = (
                    reserve_rng.choice(span, size=reserve_count, replace=False)
                    .astype(np.int64)
                    + start
                )
                selected_rows = np.concatenate((corpus, queries))
                reserve = reserve[~np.isin(reserve, selected_rows)]
                reserve_values = np.asarray(source[reserve])
                reserve_training_copies, _ = exact_reference_copy_mask(
                    source[:start], reserve_values
                )
                clean_reference = np.concatenate(
                    (
                        np.asarray(source[corpus]),
                        query_values[~replacement_mask],
                    ),
                    axis=0,
                )
                reserve_panel_copies, _ = exact_reference_copy_mask(
                    clean_reference, reserve_values
                )
                selected_replacements: list[int] = []
                selected_bytes: set[bytes] = set()
                for reserve_position in np.flatnonzero(
                    ~(reserve_training_copies | reserve_panel_copies)
                ).tolist():
                    value_bytes = np.asarray(
                        reserve_values[reserve_position]
                    ).tobytes(order="C")
                    if value_bytes in selected_bytes:
                        continue
                    selected_bytes.add(value_bytes)
                    selected_replacements.append(
                        int(reserve[reserve_position])
                    )
                    if len(selected_replacements) == needed:
                        break
                if len(selected_replacements) != needed:
                    raise RuntimeError(
                        f"{language} duplicate-free query reserve exhausted"
                    )
                queries[replacement_mask] = np.asarray(
                    selected_replacements, dtype=np.int64
                )
            final_values = np.asarray(source[queries])
            final_training_copies, _ = exact_reference_copy_mask(
                source[:start], final_values
            )
            final_corpus_copies, _ = exact_reference_copy_mask(
                np.asarray(source[corpus]), final_values
            )
            if (
                np.any(final_training_copies)
                or np.any(final_corpus_copies)
                or len(np.unique(queries)) != len(queries)
            ):
                raise RuntimeError(
                    f"{language} held-out query family hygiene did not close"
                )
        arrays[f"{language}__corpus"] = corpus
        arrays[f"{language}__queries"] = queries
        arrays[f"{language}__original_queries"] = original_queries
        arrays[f"{language}__query_replacement_mask"] = replacement_mask
    dadabase = np.load(
        DIAGNOSTIC_PATHS["dadabase"], mmap_mode="r", allow_pickle=False
    )
    dad_corpus, dad_queries = fixed_probe_split(
        row_start=0,
        row_stop=len(dadabase),
        seed=HELDOUT_SEED + 100,
    )
    arrays["dadabase__corpus"] = dad_corpus
    arrays["dadabase__queries"] = dad_queries
    fineweb = np.load(
        DIAGNOSTIC_PATHS["fineweb"], mmap_mode="r", allow_pickle=False
    )
    fineweb_corpus, fineweb_queries = fixed_probe_split(
        row_start=0,
        row_stop=len(fineweb),
        seed=HELDOUT_SEED + 101,
    )
    arrays["fineweb__corpus"] = fineweb_corpus
    arrays["fineweb__queries"] = fineweb_queries
    atomic_save_new_npz(path, immutable=True, **arrays)
    return expected_input_signature(path)


def prepare_round0108(
    *,
    release_sha: str,
    r0106_review_path: str,
    r0106_review_sha256: str,
    r0107_review_path: str,
    r0107_review_sha256: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0108 release SHA must be one full commit")
    round_file = _require_issued_round()
    reviews = {
        round_id: _require_review(
            os.path.join(LAB_ROOT, name),
            expected_sha256=sha,
            capability=capability,
        )
        for round_id, (name, sha, capability) in REVIEW_DEFAULTS.items()
    }
    reviews["0106"] = _require_review(
        r0106_review_path,
        expected_sha256=r0106_review_sha256,
        capability="capability:jina-diverse-25m-full768-fuzzy-graph-v1",
    )
    reviews["0107"] = _require_review(
        r0107_review_path,
        expected_sha256=r0107_review_sha256,
        capability="capability:jina-diverse-25m-full768-trained-map-seed42-v1",
    )

    substrate = validate_substrate_manifest(verify_payloads=False)
    inventory, inventory_signature = _load_inventory()
    language_sources, selected_stops = _language_sources(inventory)
    diagnostics = {
        key: expected_input_signature(path)
        for key, path in DIAGNOSTIC_PATHS.items()
    }
    graph_signature = expected_input_signature(GRAPH_MANIFEST)
    with open(GRAPH_MANIFEST, encoding="utf-8") as handle:
        graph = json.load(handle)
    if graph.get("schema") != "round0106-jina-diverse-25m-fuzzy-graph-v1":
        raise RuntimeError("R0106 graph manifest changed")
    train_receipt = expected_input_signature(
        os.path.join(TRAIN_OUTPUT, "train-receipt.json")
    )
    model = expected_input_signature(os.path.join(TRAIN_OUTPUT, "model.pt"))
    config = expected_input_signature(
        os.path.join(TRAIN_OUTPUT, "production-config.json")
    )
    part_receipts = {
        part: expected_input_signature(
            os.path.join(root, "part-receipt.json")
        )
        for part, root in PART_OUTPUTS.items()
    }

    queue_root = create_fresh_directory(
        queue_root, label="R0108 diverse-Jina evaluation queue"
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    inputs = ensure_data_directory(os.path.join(queue_root, "inputs"))
    selection_path = os.path.join(inputs, "registered-selections.npz")
    selection_signature = _selection_artifact(
        selection_path,
        inventory=inventory,
        substrate=substrate,
        language_sources=language_sources,
        selected_stops=selected_stops,
    )

    common = _dedupe([
        expected_input_signature(round_file),
        *reviews.values(),
        inventory_signature,
        selection_signature,
    ])
    calibration_output = os.path.join(artifacts, "jina-density-calibration")
    transform_output = os.path.join(artifacts, "coordinates")
    core_output = os.path.join(artifacts, "core-geometry")
    ood_output = os.path.join(artifacts, "ood")
    decision_output = os.path.join(artifacts, "decision")
    render_output = os.path.join(artifacts, "semantic-renders")

    calibration_inputs = _dedupe([
        *common,
        expected_input_signature(R0040_CENSUS_RECEIPT),
        expected_input_signature(R0040_REFERENCE),
        expected_input_signature(R0037_COORDINATES),
        expected_input_signature(R0038_COORDINATES),
    ])
    transform_inputs = _dedupe([
        *common,
        train_receipt,
        model,
        config,
        graph_signature,
        graph["compact_mapping"],
        substrate["signature"],
        substrate["payloads"]["int8"],
        substrate["payloads"]["scales"],
    ])
    core_inputs = _dedupe([
        *common,
        graph_signature,
        graph["compact_mapping"],
        expected_input_signature(ELIGIBILITY_PATH),
        expected_input_signature(LABELS_PATH),
        *part_receipts.values(),
        substrate["signature"],
        substrate["payloads"]["int8"],
        substrate["payloads"]["scales"],
        train_receipt,
        model,
        config,
    ])
    ood_inputs = _dedupe([
        *common,
        graph_signature,
        graph["compact_mapping"],
        substrate["signature"],
        substrate["payloads"]["int8"],
        substrate["payloads"]["scales"],
        train_receipt,
        model,
        config,
        *language_sources.values(),
        *diagnostics.values(),
    ])
    jobs = [
        {
            "id": "calibrate_jina_density",
            "action": "calibrate_jina_density",
            "handler_module": "experiments.round0108_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [calibration_output],
            "done_marker": os.path.join(
                artifacts, "calibrate_jina_density.done.json"
            ),
            "expected_inputs": calibration_inputs,
            "p90_wall_s": 1_200.0,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
            },
            "census_receipt": R0040_CENSUS_RECEIPT,
            "census_receipt_sha256": expected_input_signature(
                R0040_CENSUS_RECEIPT
            )["sha256"],
            "representative_reference": R0040_REFERENCE,
            "representative_reference_sha256": expected_input_signature(
                R0040_REFERENCE
            )["sha256"],
            "cells": [
                {
                    "key": "seed42",
                    "map": "R0037 full-768 seed42",
                    "seed": 42,
                    "coordinates": R0037_COORDINATES,
                    "coordinates_sha256": expected_input_signature(
                        R0037_COORDINATES
                    )["sha256"],
                },
                {
                    "key": "seed43",
                    "map": "R0038 full-768 seed43",
                    "seed": 43,
                    "coordinates": R0038_COORDINATES,
                    "coordinates_sha256": expected_input_signature(
                        R0038_COORDINATES
                    )["sha256"],
                },
            ],
        },
        {
            "id": "transform_retained_map",
            "action": "transform_retained_map",
            "handler_module": "experiments.round0108_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [transform_output],
            "done_marker": os.path.join(
                artifacts, "transform_retained_map.done.json"
            ),
            "expected_inputs": transform_inputs,
            "p90_wall_s": 3_600.0,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
            },
            "train_output": TRAIN_OUTPUT,
            "graph_manifest": GRAPH_MANIFEST,
            "graph_manifest_sha256": graph_signature["sha256"],
        },
        {
            "id": "score_core_geometry",
            "action": "score_core_geometry",
            "handler_module": "experiments.round0108_nodes",
            "handler_callable": "run_job",
            "deps": [
                "calibrate_jina_density",
                "transform_retained_map",
            ],
            "outputs": [core_output],
            "done_marker": os.path.join(
                artifacts, "score_core_geometry.done.json"
            ),
            "expected_inputs": core_inputs,
            "p90_wall_s": 3_600.0,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
            },
            "calibration_output": calibration_output,
            "transform_output": transform_output,
            "selection": selection_path,
            "train_output": TRAIN_OUTPUT,
            "graph_manifest": GRAPH_MANIFEST,
            "graph_manifest_sha256": graph_signature["sha256"],
            "part_outputs": PART_OUTPUTS,
            "eligibility": ELIGIBILITY_PATH,
            "labels": LABELS_PATH,
        },
        {
            "id": "score_ood",
            "action": "score_ood",
            "handler_module": "experiments.round0108_nodes",
            "handler_callable": "run_job",
            "deps": ["transform_retained_map"],
            "outputs": [ood_output],
            "done_marker": os.path.join(artifacts, "score_ood.done.json"),
            "expected_inputs": ood_inputs,
            "p90_wall_s": 7_200.0,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
            },
            "transform_output": transform_output,
            "selection": selection_path,
            "train_output": TRAIN_OUTPUT,
            "graph_manifest": GRAPH_MANIFEST,
            "graph_manifest_sha256": graph_signature["sha256"],
            "language_sources": language_sources,
            "language_training_stops": selected_stops,
            "diagnostic_sources": diagnostics,
            "embedding_prompt": EMBEDDING_PROMPT,
        },
        {
            "id": "decide_and_publish_registry",
            "action": "decide_and_publish_registry",
            "handler_module": "experiments.round0108_nodes",
            "handler_callable": "run_job",
            "deps": ["score_core_geometry", "score_ood"],
            "outputs": [decision_output],
            "done_marker": os.path.join(
                artifacts, "decide_and_publish_registry.done.json"
            ),
            "expected_inputs": common,
            "p90_wall_s": 600.0,
            "node_policy": {
                "gpu_required": False,
                "training_performed": False,
            },
            "calibration_output": calibration_output,
            "transform_output": transform_output,
            "core_output": core_output,
            "ood_output": ood_output,
            "render_output": render_output,
        },
    ]
    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=round_file,
        queue_root=queue_root,
        gpu_hours_cap=4.5,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue["schema"] = "round0108-diverse-jina-evaluation-queue-v1"
    queue["repo_root"] = RELEASE_ROOT
    queue["queue_class"] = "gpu-research"
    queue["required_reviews"] = [
        "0037", "0038", "0040", "0085", "0087", "0103", "0106", "0107"
    ]
    queue["capability_dependencies"] = [
        "jina-mrl-seed42-screen-v1",
        "jina-mrl-two-seed-decision-v1",
        "duplicate-controlled-panel-v1",
        "minilm-density-v2-calibration-v1",
        "jina-diverse-25m-inventory-v1",
        "jina-diverse-25m-full768-int8-substrate-v1",
        "jina-diverse-25m-full768-fuzzy-graph-v1",
        "jina-diverse-25m-full768-trained-map-seed42-v1",
    ]
    queue["capabilities_produced"] = [
        "jina-diverse-25m-atlas-quality-v1",
        "jina-diverse-25m-map-registry-v1",
    ]
    queue["training_performed"] = False
    queue["scientific_contract"] = {
        "map_key": MAP_KEY,
        "embedding_prompt": EMBEDDING_PROMPT,
        "prompt_applied": False,
        "production_document_prompt_transfer_resolved": False,
        "production_or_publishing_claim": False,
        "calibration": {
            "cells": ["R0037 full-768 seed42", "R0038 full-768 seed43"],
            "representative_census": expected_input_signature(
                R0040_CENSUS_RECEIPT
            ),
            "bootstrap": {
                "draws": CALIBRATION_BOOTSTRAP_DRAWS,
                "seed": CALIBRATION_BOOTSTRAP_SEED,
            },
            "permuted_radius_null": {
                "draws": CALIBRATION_NULL_DRAWS,
                "seed": CALIBRATION_NULL_SEED,
            },
            "floor_rule": "min(two cells) - 3 * max(two bootstrap SDs)",
        },
        "core_anchor_rows_per_group": ANCHORS_PER_GROUP,
        "headline_ood": {
            "probe": POLISH,
            "in_mix_controls": list(IN_MIX_LANGUAGES),
            "corpus_rows": 49_500,
            "query_rows": 500,
            "query_exact_training_family_copies": 0,
            "duplicate_replacement_policy": (
                "fixed selector followed by deterministic reserve replacement "
                "of exact training-prefix or corpus-family copies"
            ),
            "polish_recall50_minimum_relative_to_in_mix_median": 0.50,
            "recall50_must_exceed_recall10": True,
        },
        "cross_atlas_ood_diagnostic": {
            "query_cells": [
                "balanced 500-query in-mix language control",
                "500-query held-out pol_Latn",
            ],
            "candidate_universe": (
                "all 24,948,663 retained R0106 training representatives"
            ),
            "high_d_truth": "exact fp32 cosine top10",
            "low_d_membership": "exact global top-0.1% and top50",
            "role": "diagnostic-only",
        },
        "projection_ffr": "diagnostic-only",
        "diagnostic_map_cards": [
            "Dadabase Jina-v5-nano",
            "TREC-COVID Jina-v5-nano",
            "held-out FineWeb Jina-v5-nano",
        ],
        "thresholds_tunable_after_treatment": False,
        "map_decision": True,
    }
    queue["jobs"] = jobs
    queue["p90_gpu_seconds"] = {
        "calibrate_jina_density": 1_200.0,
        "transform_retained_map": 3_600.0,
        "score_core_geometry": 3_600.0,
        "score_ood": 7_200.0,
        "total": 15_600.0,
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--r0106-review", required=True)
    parser.add_argument("--r0106-review-sha256", required=True)
    parser.add_argument("--r0107-review", required=True)
    parser.add_argument("--r0107-review-sha256", required=True)
    parser.add_argument(
        "--queue-root", default=os.path.join(ROUND_ROOT, "queue")
    )
    args = parser.parse_args(argv)
    path = prepare_round0108(
        release_sha=args.release_sha,
        r0106_review_path=args.r0106_review,
        r0106_review_sha256=args.r0106_review_sha256,
        r0107_review_path=args.r0107_review,
        r0107_review_sha256=args.r0107_review_sha256,
        queue_root=args.queue_root,
    )
    print(json.dumps({"queue_manifest": path}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
