#!/usr/bin/env python3
"""Prepare, but never launch, the R0202 h4096 nested matched-dose queue."""
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
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0187_composition_nested_ladder import PRIMARY_METRICS, RUNG_ROWS
from basemap.round0202_h4096_nested_dose_ladder import (
    CAPABILITY,
    HIDDEN_DIMENSION,
    ROUND_ID,
    RUNGS,
    TARGET_POSITIVE_DRAWS_PER_EDGE,
    successful_updates_for_edges,
    train_config,
)
from basemap import round0113_prompt_contrast as prompt_contract
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter
from experiments.prepare_round0184_queue import _accepted_terminal_review
from experiments.prepare_round0188_queue import (
    PILE_QUERY_RECEIPT,
    R0187_COMMON_GRAPH,
    R0187_HALF_GRAPH,
    R0187_HALF_POPULATION,
    R0187_QUARTER_EVALUATION,
    R0187_QUARTER_POPULATION,
    R0187_SHARED_TRUTH,
    _accepted_lineage,
)
from experiments.prepare_round0191_queue import _embedded_signatures, _terminal_evidence


ROUND_ROOT = "/data/latent-basemap/runs/round-0202"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0202-2026-08-06.md")
HANDLER_MODULE = "experiments.round0202_nodes"
QUEUE_SCHEMA = "round0202-h4096-composition-nested-dose-ladder-queue-v1"
GPU_HOURS_CAP = 7.5

R0187_QUARTER_GRAPH = R0187_COMMON_GRAPH
R0191_QUEUE = "/data/latent-basemap/runs/round-0191/queue/queue.json"
R0191_TERMINAL = "/data/latent-basemap/runs/round-0191/queue/runner-terminal.json"
R0191_FULL_TRAIN = (
    "/data/latent-basemap/runs/round-0191/queue/artifacts/"
    "full-h4096-train/train-receipt.json"
)
R0191_FULL_EVALUATION = (
    "/data/latent-basemap/runs/round-0191/queue/artifacts/"
    "full-h4096-common-core-evaluation/common-core-evaluation.json"
)

P90 = {
    "train_quarter_h4096": 7_500.0,
    "evaluate_quarter_h4096": 90.0,
    "train_half_h4096": 15_000.0,
    "evaluate_half_h4096": 90.0,
    "synthesize_h4096_ladder": 30.0,
}


def _read_json(path: str, *, label: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return value


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or frontmatter.get("base_commit") != release_sha
    ):
        raise RuntimeError("R0202 round is not issued for this exact release")
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
        raise RuntimeError("R0202 release checkout differs from requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0202_h4096_nested_dose_ladder.py",
        "tests/test_round0187_composition_nested_ladder.py",
        "tests/test_round0191_full_width_contrast.py",
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
        timeout=180,
        check=False,
    )
    receipt = prompt_contract.seal({
        "schema": "round0202-release-cpu-smoke-v1",
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
            "R0202 dynamic rung/config dispatch -> shared tiny fit -> exact "
            "accounting -> seal -> state-dict reload -> transform -> tiny panel"
        ),
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0202 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return receipt


def _config_smoke() -> dict[str, Any]:
    graph_paths = {
        "quarter": R0187_QUARTER_GRAPH,
        "half": R0187_HALF_GRAPH,
    }
    cells: dict[str, Any] = {}
    expected_updates = {"quarter": 247_234, "half": 498_383}
    for rung in RUNGS:
        path = graph_paths[rung]
        signature = expected_input_signature(path)
        graph = prompt_contract.read_sealed(path, label=f"accepted R0187 {rung} graph")
        updates = successful_updates_for_edges(int(graph["directed_edge_count"]))
        config, digest = train_config(
            rung=rung,
            graph_signature=graph["graph"],
            graph_manifest_signature=signature,
            graph_edges=int(graph["directed_edge_count"]),
            retained_rows=RUNG_ROWS[rung],
        )
        if (
            updates != expected_updates[rung]
            or config["model"]["hidden_dimension"] != HIDDEN_DIMENSION
            or config["optimizer"]["seed"] != 42
            or config["optimizer"]["successful_positive_lr_updates"] != updates
            or config["execution"]["target_positive_draws_per_edge"]
            != TARGET_POSITIVE_DRAWS_PER_EDGE
        ):
            raise RuntimeError(f"R0202 {rung} config smoke changed")
        cells[rung] = {
            "population_rows": RUNG_ROWS[rung],
            "graph_manifest": signature,
            "directed_edges": int(graph["directed_edge_count"]),
            "successful_updates": updates,
            "achieved_positive_draws_per_edge": (
                updates * prompt_contract.POSITIVE_ROWS_PER_UPDATE
                / int(graph["directed_edge_count"])
            ),
            "config_sha256": digest,
        }
    return prompt_contract.seal({
        "schema": "round0202-config-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
        "cells": cells,
    })


def _accepted_inputs() -> list[dict[str, Any]]:
    signatures = [
        *_accepted_lineage(),
        *_accepted_terminal_review("0191"),
        *_terminal_evidence(
            round_id="0191", queue_path=R0191_QUEUE, terminal_path=R0191_TERMINAL
        ),
        expected_input_signature(R0191_FULL_TRAIN),
        expected_input_signature(R0191_FULL_EVALUATION),
    ]
    full_train = prompt_contract.read_sealed(
        R0191_FULL_TRAIN, label="accepted R0191 full train"
    )
    full_evaluation = prompt_contract.read_sealed(
        R0191_FULL_EVALUATION, label="accepted R0191 full evaluation"
    )
    if (
        full_train.get("schema") != "round0191-full-h4096-width-train-receipt-v1"
        or full_train.get("round_id") != "0191"
        or int(full_train.get("optimizer_updates", -1)) != 1_000_000
        or full_evaluation.get("schema")
        != "round0191-full-h4096-common-core-evaluation-v1"
        or full_evaluation.get("round_id") != "0191"
        or full_evaluation.get("rung") != "full"
        or not all((full_evaluation.get("execution_checks") or {}).values())
    ):
        raise RuntimeError("accepted R0191 full endpoint changed")
    _embedded_signatures(full_train, signatures)
    _embedded_signatures(full_evaluation, signatures)
    return _dedupe(signatures)


def prepare_round0202(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0202 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    lineage = _accepted_inputs()
    graph_paths = {
        "quarter": R0187_QUARTER_GRAPH,
        "half": R0187_HALF_GRAPH,
    }
    population_paths = {
        "quarter": R0187_QUARTER_POPULATION,
        "half": R0187_HALF_POPULATION,
    }

    queue_root = create_fresh_directory(queue_root, label="R0202 h4096 ladder queue")
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
        rung: os.path.join(artifacts, f"{rung}-h4096-train") for rung in RUNGS
    }
    evaluation_outputs = {
        rung: os.path.join(artifacts, f"{rung}-h4096-common-core-evaluation")
        for rung in RUNGS
    }
    jobs: list[dict[str, Any]] = []
    prior: list[str] = []
    for rung in RUNGS:
        train_id = f"train_{rung}_h4096"
        jobs.append({
            "id": train_id,
            "action": "train_h4096_nested_rung",
            "handler_module": HANDLER_MODULE,
            "handler_callable": "run_job",
            "deps": list(prior),
            "outputs": [train_outputs[rung]],
            "done_marker": os.path.join(artifacts, f"{rung}-h4096-train.done.json"),
            "expected_inputs": common,
            "p90_wall_s": P90[train_id],
            "rung": rung,
            "population_receipt_path": population_paths[rung],
            "graph_manifest": graph_paths[rung],
            "node_policy": {
                "gpu_required": True,
                "training_performed": True,
                "cpu_heavy": False,
            },
        })
        evaluate_id = f"evaluate_{rung}_h4096"
        jobs.append({
            "id": evaluate_id,
            "action": "evaluate_h4096_nested_rung",
            "handler_module": HANDLER_MODULE,
            "handler_callable": "run_job",
            "deps": [train_id],
            "outputs": [evaluation_outputs[rung]],
            "done_marker": os.path.join(
                artifacts, f"{rung}-h4096-evaluation.done.json"
            ),
            "expected_inputs": common,
            "p90_wall_s": P90[evaluate_id],
            "rung": rung,
            "population_receipt_path": population_paths[rung],
            "graph_manifest": graph_paths[rung],
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
        })
        prior = [evaluate_id]

    jobs.append({
        "id": "synthesize_h4096_ladder",
        "action": "synthesize_h4096_nested_ladder",
        "handler_module": HANDLER_MODULE,
        "handler_callable": "run_job",
        "deps": [f"evaluate_{rung}_h4096" for rung in RUNGS],
        "outputs": [os.path.join(artifacts, "h4096-nested-ladder-synthesis")],
        "done_marker": os.path.join(artifacts, "h4096-ladder-synthesis.done.json"),
        "expected_inputs": common,
        "p90_wall_s": P90["synthesize_h4096_ladder"],
        "evaluation_outputs": evaluation_outputs,
        "train_outputs": train_outputs,
        "graph_manifests": graph_paths,
        "r0191_full_train": R0191_FULL_TRAIN,
        "r0191_full_evaluation": R0191_FULL_EVALUATION,
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
        "required_reviews": ["0165", "0171", "0187", "0191"],
        "capability_dependencies": [
            "jina-document-english-composition-controlled-nested-ladder-v1",
            "jina-document-english-8m-h4096-width-contrast-v1",
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
                "does h4096 flatten the composition-controlled N regression at "
                "the exact accepted R0191 full-rung dose?"
            ),
            "rungs": {rung: RUNG_ROWS[rung] for rung in (*RUNGS, "full")},
            "full_endpoint": expected_input_signature(R0191_FULL_TRAIN),
            "graphs_reused_byte_exact": {
                rung: expected_input_signature(path)
                for rung, path in graph_paths.items()
            },
            "training": {
                "seed": 42,
                "hidden_dimension": HIDDEN_DIMENSION,
                "successful_updates": {"quarter": 247_234, "half": 498_383},
                "target_positive_draws_per_directed_edge": (
                    TARGET_POSITIVE_DRAWS_PER_EDGE
                ),
                "dose_source": "accepted R0191 1M / 603,086,368 edges",
                "horizon_rounding": "exact rational ceiling",
                "population_graph_sampler_optimizer_precision_frozen": True,
            },
            "evaluation": {
                "common_core_rows": RUNG_ROWS["quarter"],
                "primary_metrics": list(PRIMARY_METRICS),
                "shared_references_from_round": "0187",
                "pile_query_truth_reused_byte_exact": expected_input_signature(
                    R0187_SHARED_TRUTH
                ),
                "density_v2_transcribed": True,
            },
            "output": {
                "per_metric_step_and_compound_retentions": True,
                "registered_metric": "pile_ffr",
                "training_economics_by_rung": True,
                "cross_width_decision": "deferred to Track A3 after h2048 A2",
            },
            "release_cpu_smoke": expected_input_signature(release_smoke_path),
            "config_cpu_smoke": expected_input_signature(config_smoke_path),
        },
    })
    if queue["p90_gpu_seconds"]["total"] > GPU_HOURS_CAP * 3600:
        raise RuntimeError("R0202 P90 exceeds its queue cap")
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=QUEUE_ROOT)
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0202(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
