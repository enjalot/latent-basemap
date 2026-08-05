#!/usr/bin/env python3
"""Prepare, but never launch, the R0188 seed-43 boundary replay queue."""
from __future__ import annotations

import argparse
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
from basemap.output_safety import atomic_write_new_json, create_fresh_directory, ensure_data_directory
from basemap.round0187_composition_nested_ladder import (
    PRIMARY_METRICS,
    RETENTION_RATIO,
    RUNG_ROWS,
)
from basemap.round0188_composition_boundary_seed43 import (
    CAPABILITY,
    ROUND_ID,
    SEED,
    successful_updates_for_edges,
    train_config,
)
from basemap import round0113_prompt_contrast as prompt_contract
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter


ROUND_ROOT = "/data/latent-basemap/runs/round-0188"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0188-2026-08-05.md")
HANDLER_MODULE = "experiments.round0188_nodes"
QUEUE_SCHEMA = "round0188-composition-boundary-seed43-queue-v1"
GPU_HOURS_CAP = 8.0

R0165_POPULATION = (
    "/data/latent-basemap/runs/round-0165/queue-correction-1/artifacts/"
    "prompted-english-8m-frozen-prefix/frozen-prefix-population.json"
)
R0171_GRAPH = (
    "/data/latent-basemap/runs/round-0171/queue/artifacts/"
    "fuzzy-k50-graph-and-reference/graph-manifest.json"
)
PILE_QUERY_RECEIPT = (
    "/data/latent-basemap/runs/round-0171/queue/artifacts/"
    "heldout-query-reserve/query-reserve.json"
)
R0187_QUEUE = "/data/latent-basemap/runs/round-0187/queue-correction-1/queue.json"
R0187_TERMINAL = (
    "/data/latent-basemap/runs/round-0187/queue-correction-1/runner-terminal.json"
)
R0187_HALF_POPULATION = (
    "/data/latent-basemap/runs/round-0187/queue/artifacts/"
    "nested-populations/half/population.json"
)
R0187_QUARTER_POPULATION = (
    "/data/latent-basemap/runs/round-0187/queue/artifacts/"
    "nested-populations/quarter/population.json"
)
R0187_HALF_GRAPH = (
    "/data/latent-basemap/runs/round-0187/queue-correction-1/artifacts/"
    "half-graph/graph-manifest.json"
)
R0187_COMMON_GRAPH = (
    "/data/latent-basemap/runs/round-0187/queue-correction-1/artifacts/"
    "quarter-graph/graph-manifest.json"
)
R0187_QUARTER_EVALUATION = (
    "/data/latent-basemap/runs/round-0187/queue-correction-1/artifacts/"
    "quarter-common-core-evaluation/common-core-evaluation.json"
)
R0187_SHARED_TRUTH = (
    "/data/latent-basemap/runs/round-0187/queue-correction-1/artifacts/"
    "quarter-common-core-evaluation/pile-ood-truth-k10.npz"
)
R0187_DECISION = (
    "/data/latent-basemap/runs/round-0187/queue-correction-1/artifacts/"
    "nested-ladder-synthesis/ladder-decision.json"
)

P90 = {
    "train_half_seed43": 9_500.0,
    "evaluate_half_seed43": 75.0,
    "train_full_seed43": 19_000.0,
    "evaluate_full_seed43": 75.0,
    "synthesize_seed43_boundary": 30.0,
}


def _read_json(path: str, *, label: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return value


def _document(prefix: str, round_id: str, *, status: str) -> dict[str, Any]:
    candidates = []
    for name in sorted(os.listdir(LAB_ROOT)):
        if not re.fullmatch(
            rf"{prefix}-{round_id}-[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}(?:-[0-9]{{2}})?\.md",
            name,
        ):
            continue
        path = os.path.join(LAB_ROOT, name)
        if _frontmatter(path).get("status") == status:
            candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(
            f"R0188 requires one {status} {prefix} for R{round_id}; "
            f"found {len(candidates)}"
        )
    return expected_input_signature(candidates[0])


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    base_commit = str(frontmatter.get("base_commit") or "")
    descendant = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "merge-base", "--is-ancestor", base_commit, release_sha],
        check=False,
        timeout=10,
    ).returncode == 0
    if frontmatter.get("status") != "issued" or not descendant:
        raise RuntimeError("R0188 round is not issued for this descendant release")
    return expected_input_signature(ROUND_FILE)


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError("R0188 release checkout differs from requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0188_composition_boundary_seed43.py",
        "tests/test_round0166_cpu_smoke.py",
    ]
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    })
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=RELEASE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    receipt = prompt_contract.seal({
        "schema": "round0188-release-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "release_sha": release_sha,
        "command": command,
        "cwd": RELEASE_ROOT,
        "cuda_visible_devices": "",
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "wall_seconds": time.monotonic() - started,
        "path_exercised": (
            "R0188 seed/config dispatch -> shared tiny fit -> accounting -> seal -> "
            "checkpoint reload -> transform -> downstream tiny panel"
        ),
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0188 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return receipt


def _config_smoke() -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for rung, graph_path in (("half", R0187_HALF_GRAPH), ("full", R0171_GRAPH)):
        graph_signature = expected_input_signature(graph_path)
        graph = prompt_contract.read_sealed(
            graph_path, label=f"accepted {rung} graph manifest"
        )
        config, digest = train_config(
            rung=rung,
            graph_signature=graph["graph"],
            graph_manifest_signature=graph_signature,
            graph_edges=int(graph["directed_edge_count"]),
            retained_rows=RUNG_ROWS[rung],
        )
        expected_updates = successful_updates_for_edges(
            int(graph["directed_edge_count"])
        )
        expected_stamp = config["execution"]["expected_pipeline_stamp"]
        if (
            config["optimizer"]["seed"] != SEED
            or config["optimizer"]["positive_rng_seed"] != SEED
            or config["optimizer"]["negative_rng_seed"] != 11_300_043
            or expected_stamp["positive_rng_seed"] != SEED
            or expected_stamp["negative_rng_seed"] != 11_300_043
            or config["optimizer"]["successful_positive_lr_updates"]
            != expected_updates
        ):
            raise RuntimeError(f"R0188 {rung} config smoke changed")
        cells[rung] = {
            "graph_manifest": graph_signature,
            "graph_edges": int(graph["directed_edge_count"]),
            "successful_updates": expected_updates,
            "config_sha256": digest,
        }
    return prompt_contract.seal({
        "schema": "round0188-config-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "seed": SEED,
        "cells": cells,
    })


def _add_signature(value: Any, output: list[dict[str, Any]]) -> None:
    if isinstance(value, Mapping) and {
        "canonical_path",
        "bytes",
        "sha256",
    }.issubset(value):
        output.append(dict(value))


def _accepted_lineage() -> list[dict[str, Any]]:
    documents = []
    for round_id in ("0165", "0171", "0187"):
        documents.append(_document("review", round_id, status="accepted"))
        documents.append(_document("result", round_id, status="complete"))

    queue_signature = expected_input_signature(R0187_QUEUE)
    terminal_signature = expected_input_signature(R0187_TERMINAL)
    terminal = _read_json(R0187_TERMINAL, label="R0187 correction terminal")
    decision_signature = expected_input_signature(R0187_DECISION)
    decision = prompt_contract.read_sealed(
        R0187_DECISION, label="accepted R0187 ladder decision"
    )
    if (
        terminal.get("verdict") != "succeeded"
        or terminal.get("round_id") != "0187"
        or terminal.get("queue_manifest_sha256") != queue_signature["sha256"]
        or terminal.get("queue_manifest_sha256_at_finish")
        != queue_signature["sha256"]
        or terminal.get("queue_manifest_unchanged") is not True
        or terminal.get("release_checkout_unchanged") is not True
        or terminal.get("boundary_problems") != []
        or decision.get("round_id") != "0187"
        or (decision.get("decision") or {}).get("outcome")
        != "composition-controlled-size-regression"
        or ((decision.get("decision") or {}).get(
            "concordant_material_regression"
        ) or {}).get("pile_ffr")
        is not True
    ):
        raise RuntimeError("accepted R0187 replay premise changed")

    signatures: list[dict[str, Any]] = [
        *documents,
        queue_signature,
        terminal_signature,
        decision_signature,
        expected_input_signature(R0187_QUARTER_EVALUATION),
        expected_input_signature(R0187_SHARED_TRUTH),
    ]
    for population_path in (
        R0165_POPULATION,
        R0187_HALF_POPULATION,
        R0187_QUARTER_POPULATION,
    ):
        signatures.append(expected_input_signature(population_path))
        population = prompt_contract.read_sealed(
            population_path, label="accepted replay population"
        )
        for key in ("mapping", "document_compact", "source_text_hash_index"):
            _add_signature(population.get(key), signatures)
        selection = population.get("selection") or {}
        _add_signature(selection.get("positions"), signatures)

    for graph_path in (R0171_GRAPH, R0187_HALF_GRAPH, R0187_COMMON_GRAPH):
        signatures.append(expected_input_signature(graph_path))
        graph = prompt_contract.read_sealed(
            graph_path, label="accepted replay graph"
        )
        for key in ("graph", "high_d_reference", "topology_probe"):
            _add_signature(graph.get(key), signatures)
        for value in (graph.get("centroids") or {}).values():
            _add_signature(value, signatures)
        for cell in (graph.get("comparison_references") or {}).values():
            _add_signature(cell.get("high_d_reference"), signatures)
            for value in (cell.get("centroids") or {}).values():
                _add_signature(value, signatures)

    signatures.append(expected_input_signature(PILE_QUERY_RECEIPT))
    query = prompt_contract.read_sealed(
        PILE_QUERY_RECEIPT, label="accepted Pile query reserve"
    )
    for key in ("queries", "canonical_rows", "source_text_hashes"):
        _add_signature(query.get(key), signatures)
    return _dedupe(signatures)


def prepare_round0188(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0188 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    lineage = _accepted_lineage()
    full_population_signature = expected_input_signature(R0165_POPULATION)

    queue_root = create_fresh_directory(
        queue_root, label="R0188 seed-43 boundary queue"
    )
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    release_smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(
        release_smoke_path, _release_cpu_smoke(release_sha), immutable=True
    )
    config_smoke_path = os.path.join(preflight, "config-smoke.json")
    atomic_write_new_json(config_smoke_path, _config_smoke(), immutable=True)
    common = _dedupe([
        round_signature,
        *lineage,
        expected_input_signature(release_smoke_path),
        expected_input_signature(config_smoke_path),
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    train_outputs = {
        rung: os.path.join(artifacts, f"{rung}-seed43-train") for rung in ("half", "full")
    }
    evaluation_outputs = {
        rung: os.path.join(artifacts, f"{rung}-seed43-common-core-evaluation")
        for rung in ("half", "full")
    }
    graph_paths = {"half": R0187_HALF_GRAPH, "full": R0171_GRAPH}
    jobs: list[dict[str, Any]] = []
    prior: list[str] = []
    for rung in ("half", "full"):
        train_id = f"train_{rung}_seed43"
        train_job: dict[str, Any] = {
            "id": train_id,
            "action": "train_seed43_boundary_rung",
            "handler_module": HANDLER_MODULE,
            "handler_callable": "run_job",
            "deps": list(prior),
            "outputs": [train_outputs[rung]],
            "done_marker": os.path.join(artifacts, f"{rung}-seed43-train.done.json"),
            "expected_inputs": common,
            "p90_wall_s": P90[train_id],
            "rung": rung,
            "graph_manifest": graph_paths[rung],
            "node_policy": {
                "gpu_required": True,
                "training_performed": True,
                "cpu_heavy": False,
            },
        }
        if rung == "half":
            train_job["population_receipt_path"] = R0187_HALF_POPULATION
        else:
            train_job["population_receipt"] = full_population_signature
        jobs.append(train_job)

        evaluate_id = f"evaluate_{rung}_seed43"
        evaluate_job = {
            **train_job,
            "id": evaluate_id,
            "action": "evaluate_seed43_boundary_rung",
            "deps": [train_id],
            "outputs": [evaluation_outputs[rung]],
            "done_marker": os.path.join(
                artifacts, f"{rung}-seed43-evaluation.done.json"
            ),
            "p90_wall_s": P90[evaluate_id],
            "train_output": train_outputs[rung],
            "common_population_receipt_path": R0187_QUARTER_POPULATION,
            "common_graph_manifest": R0187_COMMON_GRAPH,
            "pile_query_receipt": PILE_QUERY_RECEIPT,
            "r0187_quarter_evaluation": R0187_QUARTER_EVALUATION,
            "shared_truth_path": R0187_SHARED_TRUTH,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
                "cpu_heavy": False,
            },
        }
        jobs.append(evaluate_job)
        prior = [evaluate_id]

    synthesis_output = os.path.join(artifacts, "seed43-boundary-synthesis")
    jobs.append({
        "id": "synthesize_seed43_boundary",
        "action": "synthesize_seed43_boundary",
        "handler_module": HANDLER_MODULE,
        "handler_callable": "run_job",
        "deps": ["evaluate_half_seed43", "evaluate_full_seed43"],
        "outputs": [synthesis_output],
        "done_marker": os.path.join(artifacts, "seed43-boundary-synthesis.done.json"),
        "expected_inputs": common,
        "p90_wall_s": P90["synthesize_seed43_boundary"],
        "r0187_ladder_decision": R0187_DECISION,
        "evaluation_outputs": evaluation_outputs,
        "node_policy": {
            "gpu_required": False,
            "training_performed": False,
            "cpu_heavy": False,
        },
    })

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
        "required_reviews": ["0165", "0171", "0187"],
        "capability_dependencies": [
            "jina-document-english-first8m-frozen-prefix-population-v1",
            "jina-document-english-composition-controlled-nested-ladder-v1",
        ],
        "capabilities_produced": [CAPABILITY],
        "training_performed": True,
        "jobs": jobs,
        "p90_gpu_seconds": {
            **{
                job["id"]: float(job["p90_wall_s"])
                for job in jobs
                if job["node_policy"]["gpu_required"]
            },
            "total": sum(
                float(job["p90_wall_s"])
                for job in jobs
                if job["node_policy"]["gpu_required"]
            ),
        },
        "scientific_contract": {
            "question": (
                "does seed 43 confirm R0187's half-to-full Pile-FFR regression "
                "under identical nested populations, graphs, recipe, and dose?"
            ),
            "only_model_treatment_relative_to_r0187": "seed 42 -> seed 43",
            "rungs": {rung: RUNG_ROWS[rung] for rung in ("half", "full")},
            "graphs_reused_byte_exact": {
                rung: expected_input_signature(path)
                for rung, path in graph_paths.items()
            },
            "training": {
                "seed": SEED,
                "hidden_dimension": 2048,
                "successful_updates": {
                    rung: successful_updates_for_edges(
                        int(prompt_contract.read_sealed(
                            graph_paths[rung], label=f"accepted {rung} graph"
                        )["directed_edge_count"])
                    )
                    for rung in ("half", "full")
                },
                "same_positive_draws_per_edge_as_r0187": True,
                "same_sampler_precision_residency_optimizer": True,
            },
            "evaluation": {
                "common_core_rows": RUNG_ROWS["quarter"],
                "primary_metrics": list(PRIMARY_METRICS),
                "shared_references_from_round": "0187",
                "pile_query_truth_reused_byte_exact": expected_input_signature(
                    R0187_SHARED_TRUTH
                ),
            },
            "decision": {
                "registered_metric": "pile_ffr",
                "registered_boundary": "half_to_full",
                "minimum_retention": RETENTION_RATIO,
                "seed43_below_floor": (
                    "confirm two-seed composition-controlled size regression and "
                    "activate one targeted h4096 sibling"
                ),
                "seed43_at_or_above_floor": (
                    "do not confirm; block capacity work and record seed sensitivity"
                ),
                "other_metric_misses": "diagnostic-only in this bounded replay",
            },
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
    parser.add_argument("--queue-root", default=QUEUE_ROOT)
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0188(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
