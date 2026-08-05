#!/usr/bin/env python3
"""Prepare, but never launch, the R0192 mixed-quarter seed family queue."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0192_quarter_seed_family import (
    CAPABILITY,
    ROUND_ID,
    ROWS,
    RUNG,
    SEEDS,
    successful_updates_for_edges,
    train_config,
)
from basemap import round0113_prompt_contrast as prompt_contract
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter
from experiments.prepare_round0188_queue import (
    PILE_QUERY_RECEIPT,
    R0187_COMMON_GRAPH,
    R0187_QUARTER_EVALUATION,
    R0187_QUARTER_POPULATION,
    R0187_SHARED_TRUTH,
    _accepted_lineage,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0192"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0192-2026-08-05.md")
HANDLER_MODULE = "experiments.round0192_nodes"
QUEUE_SCHEMA = "round0192-mixed-quarter-seed-family-queue-v1"
GPU_HOURS_CAP = 3.0
R0187_QUARTER_TRAIN = (
    "/data/latent-basemap/runs/round-0187/queue-correction-1/artifacts/"
    "quarter-seed42-train/train-receipt.json"
)
P90_TRAIN_S = 4_800.0
P90_EVALUATION_S = 75.0


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or frontmatter.get("base_commit") != release_sha
    ):
        raise RuntimeError("R0192 round is not issued for this exact release")
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
        raise RuntimeError("R0192 release checkout differs from requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0192_quarter_seed_family.py",
        "tests/test_round0187_composition_nested_ladder.py",
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
        "schema": "round0192-release-cpu-smoke-v1",
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
            "R0192 seed43/44 quarter config -> shared tiny fit -> exact "
            "accounting -> seal -> checkpoint reload -> transform -> tiny panel"
        ),
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0192 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return receipt


def _config_smoke() -> dict[str, Any]:
    graph_signature = expected_input_signature(R0187_COMMON_GRAPH)
    graph = prompt_contract.read_sealed(
        R0187_COMMON_GRAPH, label="accepted R0187 quarter graph"
    )
    updates = successful_updates_for_edges(int(graph["directed_edge_count"]))
    cells = {}
    for seed in SEEDS:
        config, digest = train_config(
            seed=seed,
            graph_signature=graph["graph"],
            graph_manifest_signature=graph_signature,
            graph_edges=int(graph["directed_edge_count"]),
            retained_rows=ROWS,
        )
        stamp = config["execution"]["expected_pipeline_stamp"]
        if (
            updates != 501_014
            or config["optimizer"]["successful_positive_lr_updates"] != updates
            or config["optimizer"]["seed"] != seed
            or config["optimizer"]["positive_rng_seed"] != seed
            or config["optimizer"]["negative_rng_seed"] != 11_300_000 + seed
            or stamp["positive_rng_seed"] != seed
            or stamp["negative_rng_seed"] != 11_300_000 + seed
            or config["model"]["hidden_dimension"] != 2048
        ):
            raise RuntimeError(f"R0192 seed {seed} config smoke changed")
        cells[str(seed)] = {"config_sha256": digest, "successful_updates": updates}
    return prompt_contract.seal({
        "schema": "round0192-config-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "rung": RUNG,
        "rows": ROWS,
        "graph_manifest": graph_signature,
        "cells": cells,
    })


def prepare_round0192(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0192 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    lineage = _dedupe([
        *_accepted_lineage(),
        expected_input_signature(R0187_QUARTER_TRAIN),
    ])
    queue_root = create_fresh_directory(
        queue_root, label="R0192 mixed-quarter seed queue"
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
        str(seed): os.path.join(artifacts, f"quarter-seed{seed}-train")
        for seed in SEEDS
    }
    evaluation_outputs = {
        str(seed): os.path.join(
            artifacts, f"quarter-seed{seed}-common-core-evaluation"
        )
        for seed in SEEDS
    }
    jobs: list[dict[str, Any]] = []
    prior: list[str] = []
    for seed in SEEDS:
        train_id = f"train_quarter_seed{seed}"
        jobs.append({
            "id": train_id,
            "action": "train_quarter_seed",
            "handler_module": HANDLER_MODULE,
            "handler_callable": "run_job",
            "deps": list(prior),
            "outputs": [train_outputs[str(seed)]],
            "done_marker": os.path.join(artifacts, f"quarter-seed{seed}-train.done.json"),
            "expected_inputs": common,
            "p90_wall_s": P90_TRAIN_S,
            "seed": seed,
            "rung": RUNG,
            "population_receipt_path": R0187_QUARTER_POPULATION,
            "graph_manifest": R0187_COMMON_GRAPH,
            "node_policy": {
                "gpu_required": True,
                "training_performed": True,
                "cpu_heavy": False,
            },
        })
        evaluate_id = f"evaluate_quarter_seed{seed}"
        jobs.append({
            "id": evaluate_id,
            "action": "evaluate_quarter_seed",
            "handler_module": HANDLER_MODULE,
            "handler_callable": "run_job",
            "deps": [train_id],
            "outputs": [evaluation_outputs[str(seed)]],
            "done_marker": os.path.join(
                artifacts, f"quarter-seed{seed}-evaluation.done.json"
            ),
            "expected_inputs": common,
            "p90_wall_s": P90_EVALUATION_S,
            "seed": seed,
            "rung": RUNG,
            "population_receipt_path": R0187_QUARTER_POPULATION,
            "graph_manifest": R0187_COMMON_GRAPH,
            "train_output": train_outputs[str(seed)],
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
        })
        prior = [evaluate_id]

    jobs.append({
        "id": "synthesize_quarter_seed_family",
        "action": "synthesize_quarter_seed_family",
        "handler_module": HANDLER_MODULE,
        "handler_callable": "run_job",
        "deps": [f"evaluate_quarter_seed{seed}" for seed in SEEDS],
        "outputs": [os.path.join(artifacts, CAPABILITY)],
        "done_marker": os.path.join(artifacts, "quarter-seed-family.done.json"),
        "expected_inputs": common,
        "p90_wall_s": 30.0,
        "r0187_quarter_evaluation": R0187_QUARTER_EVALUATION,
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
                "what is the seed-42/43/44 distribution of commensurate mixed "
                "quarter-map quality metrics?"
            ),
            "rung": RUNG,
            "rows": ROWS,
            "new_training_seeds": list(SEEDS),
            "only_treatment_relative_to_r0187_quarter": "model/RNG seed",
            "population_graph_and_evaluation_core_reused_byte_exact": True,
            "training": {
                "hidden_dimension": 2048,
                "successful_updates": 501_014,
                "same_draws_per_edge_sampler_precision_residency_optimizer": True,
            },
            "evaluation": {
                "full_mixed_and_per_corpus_panels": True,
                "disjoint_pile_reserve": True,
                "density_v2_and_projection_ffr_transcribed": True,
            },
            "output": (
                "three-seed cells and descriptive mean/sample-SD only; mixed "
                "gate registration is deferred to a reviewed CPU round"
            ),
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
        "queue_manifest": prepare_round0192(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
