#!/usr/bin/env python3
"""Prepare, but never launch, the R0160 prompted seed-44/45 queue."""
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
from basemap.round0108_evaluation import seal
from basemap.round0113_prompt_contrast import (
    GRAPH_K,
    GRAPH_NPROBE,
    RETAINED_ROWS,
    SUCCESSFUL_UPDATES,
    read_sealed,
    train_config,
)
from basemap.round0160_prompted_seed_family import (
    CAPABILITY,
    NEW_SEEDS,
    ROUND_ID,
    SEEDS,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0117_queue import _accepted_r0115_inputs
from experiments.prepare_round0138_queue import _frontmatter


ROUND_ROOT = "/data/latent-basemap/runs/round-0160"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0160-2026-08-03.md")
ASSEMBLY_ROOT = "/data/latent-basemap/runs/round-0113/queue/artifacts/compact-arrays"
ASSEMBLY_MANIFEST = os.path.join(ASSEMBLY_ROOT, "assembly-manifest.json")
QUERY_ROOT = "/data/latent-basemap/runs/round-0113/queue/artifacts/query-reserve"
R0115_ROOT = "/data/latent-basemap/runs/round-0115/queue-attempt-2"
GRAPH_ROOT = os.path.join(R0115_ROOT, "artifacts/document/graph")
GRAPH_MANIFEST = os.path.join(GRAPH_ROOT, "graph-manifest.json")
QUERY_SELECTION_ROOT = os.path.join(R0115_ROOT, "artifacts/query-selection")
REVIEWS = (
    os.path.join(LAB_ROOT, "review-0115-2026-07-30.md"),
    os.path.join(LAB_ROOT, "review-0117-2026-07-31.md"),
    os.path.join(LAB_ROOT, "review-0157-2026-08-02.md"),
)
ACCEPTED_CELLS = {
    42: {
        "score_path": os.path.join(R0115_ROOT, "artifacts/document/evaluation/score.json"),
        "coordinates_path": os.path.join(
            R0115_ROOT, "artifacts/document/evaluation/coordinates.npy"
        ),
    },
    43: {
        "score_path": (
            "/data/latent-basemap/runs/round-0117/queue/artifacts/"
            "document/evaluation/score.json"
        ),
        "coordinates_path": (
            "/data/latent-basemap/runs/round-0117/queue/artifacts/"
            "document/evaluation/coordinates.npy"
        ),
    },
}

GPU_HOURS_MINIMUM = 2.50
GPU_HOURS_EXPECTED = 3.00
GPU_HOURS_P90 = 3.55
GPU_HOURS_MAXIMUM = 4.50


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    if frontmatter.get("status") != "issued" or frontmatter.get("base_commit") != release_sha:
        raise RuntimeError("R0160 round is not issued for this release")
    return expected_input_signature(ROUND_FILE)


def _accepted_reviews() -> list[dict[str, Any]]:
    signatures = []
    for path, round_id in zip(REVIEWS, ("0115", "0117", "0157"), strict=True):
        frontmatter = _frontmatter(path)
        if frontmatter.get("status") != "accepted" or frontmatter.get("round_id") != round_id:
            raise RuntimeError(f"R0160 required Review {round_id} is not accepted")
        signatures.append(expected_input_signature(path))
    return signatures


def _pytest_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError("R0160 release checkout differs from requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0160_prompted_seed_family.py",
        "tests/test_round0161_prompted_gate_registration.py",
        "tests/test_round0117_seed43_prompt_contrast.py",
        "tests/test_round0117_cpu_smoke.py",
        "tests/test_panel_v2.py",
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
        timeout=120,
        check=False,
    )
    receipt = seal({
        "schema": "round0160-release-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "release_sha": release_sha,
        "command": command,
        "cwd": RELEASE_ROOT,
        "cuda_visible_devices": "",
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "wall_seconds": time.monotonic() - started,
        "path_exercised": "train-config -> seal -> checkpoint reload -> transform -> panel",
    })
    if completed.returncode != 0:
        raise RuntimeError(f"R0160 CPU smoke failed:\n{completed.stdout}\n{completed.stderr}")
    return receipt


def _seed_config_smoke(graph: dict[str, Any]) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    configs: dict[int, dict[str, Any]] = {}
    for seed in (42, *NEW_SEEDS):
        config, digest = train_config(
            "document",
            graph_signature=graph["graph"],
            graph_manifest_signature=expected_input_signature(GRAPH_MANIFEST),
            graph_edges=int(graph["directed_edge_count"]),
            retained_rows=RETAINED_ROWS,
            seed=seed,
        )
        configs[seed] = config
        reports[f"seed{seed}"] = {
            "config_sha256": digest,
            "seed": seed,
            "expected_pipeline_stamp": config["execution"]["expected_pipeline_stamp"],
        }
    allowed = {
        "paired_invariant.seed",
        "optimizer.seed",
        "optimizer.positive_rng_seed",
        "optimizer.negative_rng_seed",
        "execution.expected_pipeline_stamp.positive_rng_seed",
        "execution.expected_pipeline_stamp.negative_rng_seed",
    }

    def changed(left: Any, right: Any, prefix: str = "") -> set[str]:
        if isinstance(left, dict) and isinstance(right, dict):
            if set(left) != set(right):
                raise RuntimeError("R0160 config field set changed")
            output: set[str] = set()
            for key in left:
                output.update(changed(left[key], right[key], f"{prefix}.{key}" if prefix else key))
            return output
        return {prefix} if left != right else set()

    for seed in NEW_SEEDS:
        observed = changed(configs[42], configs[seed])
        if observed != allowed:
            raise RuntimeError(f"R0160 seed-{seed} config changes {observed}, expected {allowed}")
    return seal({
        "schema": "round0160-seed-config-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "only_changed_fields": sorted(allowed),
        "cells": reports,
    })


def _authenticate_inputs() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reused, graph_outputs, selection_output, _decision, assembly, query = (
        _accepted_r0115_inputs()
    )
    if graph_outputs["document"] != GRAPH_ROOT or selection_output != QUERY_SELECTION_ROOT:
        raise RuntimeError("R0160 accepted R0115 path binding changed")
    graph = read_sealed(GRAPH_MANIFEST, label="accepted R0115 document graph")
    if (
        graph.get("round_id") != "0115"
        or graph.get("arm") != "document"
        or int(graph.get("retained_rows", -1)) != RETAINED_ROWS
        or int(graph.get("k", -1)) != GRAPH_K
        or int(graph.get("search_qualification", {}).get("selected_nprobe", -1)) != GRAPH_NPROBE
    ):
        raise RuntimeError("R0160 accepted document graph changed")
    cells: dict[str, Any] = {}
    cell_inputs: list[dict[str, Any]] = []
    for seed, paths in ACCEPTED_CELLS.items():
        score_signature = expected_input_signature(paths["score_path"])
        coordinates_signature = expected_input_signature(paths["coordinates_path"])
        score = read_sealed(paths["score_path"], label=f"accepted seed-{seed} prompted score")
        observed_seed = int(score.get("training_seed", 42 if seed == 42 else -1))
        if (
            score.get("arm") != "document"
            or observed_seed != seed
            or score.get("coordinates", {}).get("training") != coordinates_signature
            or score.get("graph_manifest") != expected_input_signature(GRAPH_MANIFEST)
        ):
            raise RuntimeError(f"R0160 accepted seed-{seed} score changed")
        cells[f"seed{seed}"] = {
            "score": score_signature,
            "coordinates": coordinates_signature,
        }
        cell_inputs.extend((score_signature, coordinates_signature, score["train_receipt"]))
    inputs = _dedupe([
        *reused,
        *cell_inputs,
        graph["graph"],
        graph["high_d_reference"],
    ])
    receipt = seal({
        "schema": "round0160-pretrain-input-authentication-v1",
        "round_id": ROUND_ID,
        "assembly_identity": assembly["identity_sha256"],
        "query_reserve_identity": query["identity_sha256"],
        "graph_manifest": expected_input_signature(GRAPH_MANIFEST),
        "graph": graph["graph"],
        "accepted_high_d_reference": graph["high_d_reference"],
        "accepted_cells": cells,
        "all_evaluation_inputs_authenticated_before_training": True,
    })
    return inputs, receipt


def prepare_round0160(
    *, release_sha: str, queue_root: str = os.path.join(ROUND_ROOT, "queue")
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0160 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    reviews = _accepted_reviews()
    reused_inputs, authentication = _authenticate_inputs()
    graph = read_sealed(GRAPH_MANIFEST, label="accepted R0115 document graph")
    assembly = read_sealed(ASSEMBLY_MANIFEST, label="R0113 compact assembly")

    queue_root = create_fresh_directory(queue_root, label="R0160 prompted seed queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    authentication_path = os.path.join(preflight, "input-authentication.json")
    atomic_write_new_json(authentication_path, authentication, immutable=True)
    seed_smoke_path = os.path.join(preflight, "seed-config-smoke.json")
    atomic_write_new_json(seed_smoke_path, _seed_config_smoke(graph), immutable=True)
    pytest_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(pytest_path, _pytest_smoke(release_sha), immutable=True)

    expected_inputs = _dedupe([
        round_signature,
        *reviews,
        *reused_inputs,
        expected_input_signature(authentication_path),
        expected_input_signature(seed_smoke_path),
        expected_input_signature(pytest_path),
    ])
    train_outputs = {seed: os.path.join(artifacts, f"seed{seed}", "train") for seed in NEW_SEEDS}
    score_outputs = {
        seed: os.path.join(artifacts, f"seed{seed}", "evaluation") for seed in NEW_SEEDS
    }
    jobs: list[dict[str, Any]] = []
    prior: str | None = None
    for seed in NEW_SEEDS:
        train_id = f"train_document_seed{seed}"
        evaluate_id = f"evaluate_document_seed{seed}"
        jobs.append({
            "id": train_id,
            "action": "train_arm",
            "handler_module": "experiments.round0113_nodes",
            "handler_callable": "run_job",
            "deps": [] if prior is None else [prior],
            "outputs": [train_outputs[seed]],
            "done_marker": os.path.join(artifacts, f"train-document-seed{seed}.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": 5_000.0,
            "arm": "document",
            "training_seed": seed,
            "graph_execution_round_id": "0115",
            "assembly_output": ASSEMBLY_ROOT,
            "graph_manifest": GRAPH_MANIFEST,
            "node_policy": {"gpu_required": True, "training_performed": True},
        })
        jobs.append({
            "id": evaluate_id,
            "action": "evaluate_arm",
            "handler_module": "experiments.round0113_nodes",
            "handler_callable": "run_job",
            "deps": [train_id],
            "outputs": [score_outputs[seed]],
            "done_marker": os.path.join(artifacts, f"evaluate-document-seed{seed}.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": 300.0,
            "arm": "document",
            "training_seed": seed,
            "graph_execution_round_id": "0115",
            "assembly_output": ASSEMBLY_ROOT,
            "query_output": QUERY_ROOT,
            "query_selection_output": QUERY_SELECTION_ROOT,
            "graph_manifest": GRAPH_MANIFEST,
            "train_output": train_outputs[seed],
            "node_policy": {"gpu_required": True, "training_performed": False},
        })
        prior = evaluate_id

    family_output = os.path.join(artifacts, CAPABILITY)
    cells = [
        {"seed": seed, **ACCEPTED_CELLS[seed]} for seed in (42, 43)
    ] + [
        {
            "seed": seed,
            "score_path": os.path.join(score_outputs[seed], "score.json"),
            "coordinates_path": os.path.join(score_outputs[seed], "coordinates.npy"),
        }
        for seed in NEW_SEEDS
    ]
    jobs.append({
        "id": "score_prompted_seed_family",
        "action": "score_prompted_seed_family",
        "handler_module": "experiments.round0160_nodes",
        "handler_callable": "run_job",
        "deps": [f"evaluate_document_seed{seed}" for seed in NEW_SEEDS],
        "outputs": [family_output],
        "done_marker": os.path.join(artifacts, "prompted-seed-family.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": 2_200.0,
        "assembly_manifest": ASSEMBLY_MANIFEST,
        "document_compact": assembly["outputs"]["document"]["canonical_path"],
        "accepted_high_d_reference": graph["high_d_reference"]["canonical_path"],
        "accepted_reviews": reviews,
        "cells": cells,
        "node_policy": {"gpu_required": True, "training_performed": False},
    })

    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=GPU_HOURS_MAXIMUM,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue.update({
        "schema": "round0160-prompted-seed-family-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-research",
        "required_reviews": ["0115", "0117", "0157"],
        "capability_dependencies": [
            "jina-fineweb-2m-prompt-map-contrast-v1",
            "jina-fineweb-2m-prompt-map-seed43-contrast-v1",
            "jina-fineweb-2m-native-prompted-density-v2-v1",
        ],
        "capabilities_produced": [CAPABILITY],
        "training_performed": True,
        "jobs": jobs,
        "p90_gpu_seconds": {
            **{job["id"]: float(job["p90_wall_s"]) for job in jobs if job["node_policy"]["gpu_required"]},
            "total": sum(float(job["p90_wall_s"]) for job in jobs if job["node_policy"]["gpu_required"]),
        },
        "scientific_contract": {
            "question": "what is the native prompted-map quality family across seeds 42-45?",
            "rows": RETAINED_ROWS,
            "dimension": 768,
            "embedding_convention": "Document: ",
            "new_seeds": list(NEW_SEEDS),
            "accepted_context_seeds": [42, 43],
            "graph": graph["graph"],
            "graph_reused_byte_exact": True,
            "graph_builds": 0,
            "successful_updates_per_new_seed": SUCCESSFUL_UPDATES,
            "only_training_factor_changed": "model/optimizer seed",
            "full_shared_reference_panel": [
                "density_v2",
                "ffr",
                "purity_fidelity_k256",
                "purity_fidelity_k1024",
                "projection_ffr",
                "heldout_recall_at_10",
            ],
            "purity_reference": "native prompted full-population k-means, seed 0, 25 Lloyd iterations",
            "gate_formula_deferred_to_preregistered_round": "0161",
            "raw_floor_changed": False,
            "input_authentication": expected_input_signature(authentication_path),
            "seed_config_smoke": expected_input_signature(seed_smoke_path),
            "train_seal_panel_cpu_smoke": expected_input_signature(pytest_path),
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
    print(json.dumps({"queue_manifest": prepare_round0160(release_sha=args.release_sha, queue_root=args.queue_root)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
