#!/usr/bin/env python3
"""Prepare, but never launch, the R0166 prompted-English 8M scale rung."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0160_prompted_seed_family import CAPABILITY as FAMILY_CAPABILITY
from basemap.round0161_prompted_gate_registration import CAPABILITY as GATE_CAPABILITY
from basemap.round0165_frozen_prefix_population import (
    CAPABILITY as POPULATION_CAPABILITY,
    HOST_CAPABILITY as POPULATION_HOST_CAPABILITY,
)
from basemap.round0166_prompted_8m import CAPABILITY, ROUND_ID
from basemap.round0166_prompted_8m import (
    MULTIPLICITY_POLICY,
    scale_train_config,
)
from basemap.round0113_prompt_contrast import seal
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter
from experiments.prepare_round0163_queue import LAYOUT_PATH, _dedupe_signatures
from experiments.round0163_nodes import _read_sealed
from experiments.round0166_nodes import _full_text_layouts, _query_payload_inputs


ROUND_ROOT = "/data/latent-basemap/runs/round-0166"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0166-2026-08-03.md")
POPULATION_PATH = (
    "/data/latent-basemap/runs/round-0165/queue-correction-1/artifacts/"
    "prompted-english-8m-frozen-prefix/frozen-prefix-population.json"
)
FAMILY_PATH = (
    "/data/latent-basemap/runs/round-0160/queue/artifacts/"
    "jina-fineweb-2m-prompted-seed42-45-family-v1/prompted-seed-family.json"
)
GATES_PATH = (
    "/data/latent-basemap/runs/round-0161/queue/artifacts/"
    "jina-prompted-universe-quality-gates-v1/prompted-quality-gates.json"
)
HANDLER_MODULE = "experiments.round0166_nodes"
QUEUE_SCHEMA = "round0166-prompted-english-8m-scale-queue-v1"
QUEUE_LABEL = "R0166 GPU queue"
GRAPH_VECTOR_STORAGE = "gpu-ivfflat-fp32"
GPU_HOURS_CAP = 5.5
SELECT_P90_WALL_S = 900.0
GRAPH_P90_WALL_S = 7_200.0
TRAIN_P90_WALL_S = 6_000.0
EVALUATION_P90_WALL_S = 3_600.0


def _one_document(prefix: str, round_id: str, *, status: str) -> dict[str, Any]:
    paths = [
        path
        for path in sorted(glob.glob(os.path.join(LAB_ROOT, f"{prefix}-{round_id}-*.md")))
        if _frontmatter(path).get("status") == status
    ]
    if len(paths) != 1:
        raise RuntimeError(
            f"R0166 requires one {status} {prefix} for R{round_id}; found {len(paths)}"
        )
    return expected_input_signature(paths[0])


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    base_commit = str(frontmatter.get("base_commit") or "")
    descendant = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "merge-base", "--is-ancestor", base_commit, release_sha],
        check=False,
        timeout=10,
    ).returncode == 0
    if frontmatter.get("status") != "issued" or not descendant:
        raise RuntimeError("R0166 round is not issued for this release")
    return expected_input_signature(ROUND_FILE)


def _as_signatures(values: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return _dedupe_signatures([dict(value) for value in values])


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError("R0166 release checkout differs from requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0166_cpu_smoke.py",
        "tests/test_round0166_prompted_8m.py",
    ]
    environment = os.environ.copy()
    environment.update({"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"})
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=RELEASE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    receipt = seal({
        "schema": "round0166-release-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "release_sha": release_sha,
        "command": command,
        "cwd": RELEASE_ROOT,
        "cuda_visible_devices": "",
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "wall_seconds": time.monotonic() - started,
        "path_exercised": "R0166 train -> accounting -> seal -> checkpoint reload -> transform -> panel",
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0166 CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return receipt


def _config_smoke() -> dict[str, Any]:
    config, digest = scale_train_config(
        graph_signature={"canonical_path": "/future/graph.npz", "sha256": "a" * 64},
        graph_manifest_signature={
            "canonical_path": "/future/graph-manifest.json",
            "sha256": "b" * 64,
        },
        graph_edges=123,
        retained_rows=7_952_419,
    )
    stamp = config["execution"]["expected_pipeline_stamp"]
    if (
        config["paired_invariant"]["rows"] != 7_952_419
        or config["paired_invariant"]["successful_positive_lr_updates"] != 500_000
        or config["graph"]["k"] != 50
        or config["optimizer"]["seed"] != 42
        or stamp["compact_retained_rows"] != 7_952_419
        or stamp["multiplicity_policy"] != MULTIPLICITY_POLICY
    ):
        raise RuntimeError("R0166 config smoke changed")
    return seal({
        "schema": "round0166-config-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "config_sha256": digest,
        "rows": 7_952_419,
        "successful_updates": 500_000,
        "graph_k": 50,
        "training_seed": 42,
        "expected_pipeline_stamp": stamp,
    })


def prepare_round0166(
    *, release_sha: str, queue_root: str = os.path.join(ROUND_ROOT, "queue")
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0166 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    reviews = {
        round_id: _one_document("review", round_id, status="accepted")
        for round_id in ("0160", "0161", "0165")
    }
    results = {
        round_id: _one_document("result", round_id, status="complete")
        for round_id in ("0160", "0161", "0165")
    }
    population_signature = expected_input_signature(POPULATION_PATH)
    population = _read_sealed(population_signature, label="accepted R0165 population")
    if (
        population.get("outcome") != "prompted-8m-frozen-prefix-population-qualified"
        or population.get("capabilities")
        != [POPULATION_CAPABILITY, POPULATION_HOST_CAPABILITY]
        or int(population.get("retained_rows", -1)) != 7_952_419
    ):
        raise RuntimeError("R0166 accepted population changed")

    family_signature = expected_input_signature(FAMILY_PATH)
    family = _read_sealed(family_signature, label="accepted R0160 prompted family")
    gate_signature = expected_input_signature(GATES_PATH)
    gates = _read_sealed(gate_signature, label="accepted R0161 prompted gates")
    if (
        family.get("capability") != FAMILY_CAPABILITY
        or gates.get("capability") != GATE_CAPABILITY
        or gates.get("registered") is not True
        or gates.get("family_evidence") != family_signature
    ):
        raise RuntimeError("R0166 accepted family/gate lineage changed")

    layout_signature = expected_input_signature(LAYOUT_PATH)
    layout = _read_sealed(layout_signature, label="accepted R0162 canonical layout")
    manifests = layout.get("source_manifests") or {}
    r0116_signature = dict(manifests["0116"])
    r0120_signature = dict(manifests["0120"])
    r0116 = _read_sealed(r0116_signature, label="accepted R0116 manifest")
    r0120 = _read_sealed(r0120_signature, label="accepted R0120 manifest")
    text_layout = _full_text_layouts(r0116, r0120)
    query_payloads = _query_payload_inputs(layout, text_layout)

    seed42 = family["cells"]["seed42"]
    accepted_score_signature = dict(seed42["native_score"])
    accepted_score = _read_sealed(
        accepted_score_signature, label="accepted prompted seed-42 score"
    )
    accepted_query_signature = dict(accepted_score["query_reserve"])
    accepted_query = _read_sealed(
        accepted_query_signature, label="accepted R0113 query reserve"
    )
    accepted_selection_signature = dict(accepted_score["query_selection"])
    accepted_selection = _read_sealed(
        accepted_selection_signature, label="accepted seed-42 query selection"
    )
    accepted_inputs = _as_signatures([
        family_signature,
        gate_signature,
        family["lineage"]["assembly"],
        family["lineage"]["document_compact"],
        family["shared_prompted_reference"],
        *family["centroids"].values(),
        accepted_score_signature,
        accepted_score["train_receipt"],
        accepted_score["combined_query_truth"],
        accepted_query_signature,
        accepted_query["outputs"]["document"],
        accepted_selection_signature,
        accepted_selection["positions"],
    ])
    population_inputs = _as_signatures([
        population_signature,
        population["mapping"],
        population["document_compact"],
        population["source_text_hash_index"],
    ])
    common = _dedupe([
        round_signature,
        *reviews.values(),
        *results.values(),
        *population_inputs,
        *accepted_inputs,
    ])

    queue_root = create_fresh_directory(queue_root, label=QUEUE_LABEL)
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    release_smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(
        release_smoke_path, _release_cpu_smoke(release_sha), immutable=True
    )
    config_smoke_path = os.path.join(preflight, "config-smoke.json")
    atomic_write_new_json(config_smoke_path, _config_smoke(), immutable=True)
    common = _dedupe([
        *common,
        expected_input_signature(release_smoke_path),
        expected_input_signature(config_smoke_path),
    ])
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    query_output = os.path.join(artifacts, "heldout-query-reserve")
    graph_output = os.path.join(artifacts, "fuzzy-k50-graph-and-reference")
    train_output = os.path.join(artifacts, "seed42-train")
    evaluation_output = os.path.join(artifacts, CAPABILITY)
    graph_manifest = os.path.join(graph_output, "graph-manifest.json")
    jobs = [
        {
            "id": "select_heldout_queries",
            "action": "select_heldout_queries",
            "handler_module": HANDLER_MODULE,
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [query_output],
            "done_marker": os.path.join(artifacts, "heldout-queries.done.json"),
            "expected_inputs": _dedupe([
                *common,
                layout_signature,
                r0116_signature,
                r0120_signature,
                *query_payloads,
            ]),
            "p90_wall_s": SELECT_P90_WALL_S,
            "population_receipt": population_signature,
            "canonical_layout": layout_signature,
            "r0116_manifest": r0116_signature,
            "r0120_manifest": r0120_signature,
            "payload_inputs": query_payloads,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
                "cpu_heavy": False,
            },
        },
        {
            "id": "build_graph_and_reference",
            "action": "build_graph_and_reference",
            "handler_module": HANDLER_MODULE,
            "handler_callable": "run_job",
            "deps": ["select_heldout_queries"],
            "outputs": [graph_output],
            "done_marker": os.path.join(artifacts, "graph-reference.done.json"),
            "expected_inputs": common,
            "p90_wall_s": GRAPH_P90_WALL_S,
            "population_receipt": population_signature,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
                "cpu_heavy": True,
            },
        },
        {
            "id": "train_prompted_8m",
            "action": "train_prompted_8m",
            "handler_module": HANDLER_MODULE,
            "handler_callable": "run_job",
            "deps": ["build_graph_and_reference"],
            "outputs": [train_output],
            "done_marker": os.path.join(artifacts, "train.done.json"),
            "expected_inputs": common,
            "p90_wall_s": TRAIN_P90_WALL_S,
            "population_receipt": population_signature,
            "graph_manifest": graph_manifest,
            "node_policy": {
                "gpu_required": True,
                "training_performed": True,
                "cpu_heavy": False,
            },
        },
        {
            "id": "evaluate_prompted_8m",
            "action": "evaluate_prompted_8m",
            "handler_module": HANDLER_MODULE,
            "handler_callable": "run_job",
            "deps": ["select_heldout_queries", "build_graph_and_reference", "train_prompted_8m"],
            "outputs": [evaluation_output],
            "done_marker": os.path.join(artifacts, "evaluation.done.json"),
            "expected_inputs": common,
            "p90_wall_s": EVALUATION_P90_WALL_S,
            "population_receipt": population_signature,
            "query_output": query_output,
            "graph_manifest": graph_manifest,
            "train_output": train_output,
            "family_evidence": family_signature,
            "gate_registration": gate_signature,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
                "cpu_heavy": False,
            },
        },
    ]
    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=GPU_HOURS_CAP,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue.update({
        "schema": QUEUE_SCHEMA,
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-research",
        "required_reviews": ["0160", "0161", "0165"],
        "capability_dependencies": [
            FAMILY_CAPABILITY,
            GATE_CAPABILITY,
            POPULATION_CAPABILITY,
            POPULATION_HOST_CAPABILITY,
        ],
        "capabilities_produced": [CAPABILITY],
        "training_performed": True,
        "jobs": jobs,
        "p90_gpu_seconds": {
            **{job["id"]: float(job["p90_wall_s"]) for job in jobs},
            "total": sum(float(job["p90_wall_s"]) for job in jobs),
        },
        "scientific_contract": {
            "question": "does the calibrated prompted recipe retain quality when English N scales from 2M to 8M?",
            "only_treatment": "population size and its induced graph",
            "source_rows": 8_000_000,
            "representative_rows": 7_952_419,
            "embedding_convention": "Document: ",
            "graph": {
                "k": 50,
                "nlist": 8_192,
                "nprobe": 64,
                "same_builder_parameters_and_seeds_as_r0115": True,
                "vector_storage": GRAPH_VECTOR_STORAGE,
            },
            "training": {
                "seed": 42,
                "successful_positive_lr_updates": 500_000,
                "same_recipe_and_fixed_dose_as_r0115": True,
                "multiplicity_is_metadata": True,
            },
            "heldout_queries_selected_before_training": True,
            "native_absolute_gate_metrics": [
                "density_v2",
                "ffr",
                "purity_fidelity_k256",
                "purity_fidelity_k1024",
            ],
            "native_projection_metrics": "diagnostic-only because N and query reserve change",
            "matched_2m_retention_metrics": [
                "density_v2",
                "ffr",
                "purity_fidelity_k256",
                "purity_fidelity_k1024",
                "projection_ffr",
                "heldout_recall_at_10",
            ],
            "matched_2m_minimum_ratio_to_seed42": 0.97,
            "host_rss_hard_abort_gib": 90.0,
            "negative_outcome_releases_no_map_capability": True,
            "release_cpu_smoke": expected_input_signature(release_smoke_path),
            "config_cpu_smoke": expected_input_signature(config_smoke_path),
        },
    })
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=os.path.join(ROUND_ROOT, "queue"))
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0166(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
