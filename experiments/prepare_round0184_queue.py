#!/usr/bin/env python3
"""Prepare, but never launch, the conditional R0184 diagnostic dose queue."""
from __future__ import annotations

import argparse
import glob
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
from basemap import round0113_prompt_contrast as prompt_contract
from basemap.round0160_prompted_seed_family import CAPABILITY as FAMILY_CAPABILITY
from basemap.round0161_prompted_gate_registration import CAPABILITY as GATE_CAPABILITY
from basemap.round0165_frozen_prefix_population import (
    CAPABILITY as POPULATION_CAPABILITY,
    HOST_CAPABILITY as POPULATION_HOST_CAPABILITY,
)
from basemap.round0180_dose_matched_8m import CAPABILITY as R0180_CAPABILITY
from basemap.round0184_prompted_8m_dose_midpoint import (
    ACHIEVED_POSITIVE_DRAWS_PER_EDGE,
    CAPABILITY,
    HOST_RSS_LIMIT_GIB,
    RETAINED_ROWS,
    ROUND_ID,
    SUCCESSFUL_UPDATES,
    TARGET_GRAPH_EDGES,
    scale_train_config,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import (
    _accepted_review,
    _frontmatter,
)
from experiments.prepare_round0166_queue import FAMILY_PATH, GATES_PATH, POPULATION_PATH
from experiments.prepare_round0180_queue import (
    R0171_GRAPH,
    R0171_QUERY,
    R0171_QUERY_OUTPUT,
    _r0171_lineage,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0184"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0184-2026-08-03.md")
HANDLER_MODULE = "experiments.round0184_nodes"
QUEUE_SCHEMA = "round0184-prompted-8m-dose-midpoint-queue-v1"
GPU_HOURS_CAP = 4.0
TRAIN_P90_WALL_S = 10_800.0
EVALUATION_P90_WALL_S = 900.0


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or frontmatter.get("base_commit") != release_sha
    ):
        raise RuntimeError("R0184 issued round binding changed")
    return expected_input_signature(ROUND_FILE)


def _accepted_terminal_review(round_id: str) -> list[dict[str, Any]]:
    matches: list[list[dict[str, Any]]] = []
    for review_path in sorted(
        glob.glob(os.path.join(LAB_ROOT, f"review-{round_id}-*.md"))
    ):
        review = _frontmatter(review_path)
        if review.get("round_id") != round_id or review.get("status") != "accepted":
            continue
        result_path = os.path.join(LAB_ROOT, review.get("result") or "")
        round_path = os.path.join(LAB_ROOT, review.get("round") or "")
        if not os.path.isfile(result_path) or not os.path.isfile(round_path):
            raise RuntimeError(f"Review {round_id} points to missing evidence")
        result_frontmatter = _frontmatter(result_path)
        if result_frontmatter.get("status") not in {"complete", "failed", "blocked"}:
            continue
        result = expected_input_signature(result_path)
        issued = expected_input_signature(round_path)
        if (
            result["sha256"] != review.get("result_sha256")
            or issued["sha256"] != review.get("round_sha256")
            or result_frontmatter.get("release_commit")
            != review.get("verified_release_commit")
        ):
            raise RuntimeError(f"Review {round_id} binding changed")
        matches.append([issued, result, expected_input_signature(review_path)])
    if len(matches) != 1:
        raise RuntimeError(
            f"R0184 requires one accepted terminal Review {round_id}; found {len(matches)}"
        )
    return matches[0]


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError("R0184 release checkout differs from requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0184_prompted_8m_dose_midpoint.py",
        "tests/test_round0180_dose_matched_8m.py",
        "tests/test_round0171_prompted_8m.py",
        "tests/test_round0166_cpu_smoke.py",
        "tests/test_round0166_prompted_8m.py",
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
        "schema": "round0184-release-cpu-smoke-v1",
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
            "R0184 dose config and diagnostic policy -> shared train -> accounting "
            "-> seal -> checkpoint reload -> transform -> tiny panel"
        ),
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0184 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return receipt


def _config_smoke() -> dict[str, Any]:
    graph = expected_input_signature(R0171_GRAPH)
    manifest = prompt_contract.read_sealed(
        R0171_GRAPH, label="accepted R0171 graph manifest"
    )
    config, digest = scale_train_config(
        graph_signature=dict(manifest["graph"]),
        graph_manifest_signature=graph,
        graph_edges=int(manifest["directed_edge_count"]),
        retained_rows=int(manifest["retained_rows"]),
    )
    if (
        config["optimizer"]["successful_positive_lr_updates"] != SUCCESSFUL_UPDATES
        or config["optimizer"]["seed"] != 42
        or config["graph"]["k"] != 50
        or config["execution"]["expected_pipeline_stamp"]["compact_retained_rows"]
        != RETAINED_ROWS
    ):
        raise RuntimeError("R0184 config smoke changed")
    return prompt_contract.seal({
        "schema": "round0184-config-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "config_sha256": digest,
        "successful_updates": SUCCESSFUL_UPDATES,
        "target_graph_edges": TARGET_GRAPH_EDGES,
        "achieved_positive_draws_per_edge": ACHIEVED_POSITIVE_DRAWS_PER_EDGE,
        "metric_gates_required_for_capability": False,
    })


def prepare_round0184(
    *, release_sha: str, queue_root: str = QUEUE_ROOT
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0184 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    r0180_evidence = _accepted_review("0180", R0180_CAPABILITY)
    r0181_evidence = _accepted_terminal_review("0181")
    lineage = _r0171_lineage()
    population_signature = expected_input_signature(POPULATION_PATH)
    family_signature = expected_input_signature(FAMILY_PATH)
    gate_signature = expected_input_signature(GATES_PATH)

    queue_root = create_fresh_directory(queue_root, label="R0184 dose-midpoint queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    release_smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(
        release_smoke_path, _release_cpu_smoke(release_sha), immutable=True
    )
    config_smoke_path = os.path.join(preflight, "config-smoke.json")
    atomic_write_new_json(config_smoke_path, _config_smoke(), immutable=True)
    common = _dedupe([
        round_signature,
        *r0180_evidence,
        *r0181_evidence,
        *lineage,
        population_signature,
        family_signature,
        gate_signature,
        expected_input_signature(release_smoke_path),
        expected_input_signature(config_smoke_path),
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    train_output = os.path.join(artifacts, "seed42-1m-update-train")
    evaluation_output = os.path.join(artifacts, CAPABILITY)
    jobs = [
        {
            "id": "train_prompted_8m_dose_midpoint",
            "action": "train_prompted_8m",
            "handler_module": HANDLER_MODULE,
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [train_output],
            "done_marker": os.path.join(artifacts, "train.done.json"),
            "expected_inputs": common,
            "p90_wall_s": TRAIN_P90_WALL_S,
            "population_receipt": population_signature,
            "graph_manifest": R0171_GRAPH,
            "node_policy": {
                "gpu_required": True,
                "training_performed": True,
                "cpu_heavy": False,
            },
        },
        {
            "id": "evaluate_prompted_8m_dose_midpoint",
            "action": "evaluate_prompted_8m",
            "handler_module": HANDLER_MODULE,
            "handler_callable": "run_job",
            "deps": ["train_prompted_8m_dose_midpoint"],
            "outputs": [evaluation_output],
            "done_marker": os.path.join(artifacts, "evaluation.done.json"),
            "expected_inputs": common,
            "p90_wall_s": EVALUATION_P90_WALL_S,
            "population_receipt": population_signature,
            "query_output": R0171_QUERY_OUTPUT,
            "graph_manifest": R0171_GRAPH,
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
        "required_reviews": ["0160", "0161", "0165", "0171", "0180", "0181"],
        "capability_dependencies": [
            FAMILY_CAPABILITY,
            GATE_CAPABILITY,
            POPULATION_CAPABILITY,
            POPULATION_HOST_CAPABILITY,
            R0180_CAPABILITY,
        ],
        "capabilities_produced": [CAPABILITY],
        "training_performed": True,
        "jobs": jobs,
        "p90_gpu_seconds": {
            **{job["id"]: float(job["p90_wall_s"]) for job in jobs},
            "total": sum(float(job["p90_wall_s"]) for job in jobs),
        },
        "scientific_contract": {
            "question": "what is the middle cell of the 8M prompted dose-response curve?",
            "only_treatment_relative_to_r0171": (
                "successful positive-LR horizon 500000 -> 1000000"
            ),
            "only_treatment_relative_to_r0180": (
                "successful positive-LR horizon 2026478 -> 1000000"
            ),
            "population_rows": RETAINED_ROWS,
            "embedding_convention": "Document: ",
            "graph": {
                "source_round": "0171",
                "manifest": expected_input_signature(R0171_GRAPH),
                "directed_edges": TARGET_GRAPH_EDGES,
                "reused_byte_exact": True,
                "built_in_round": False,
            },
            "training": {
                "seed": 42,
                "successful_positive_lr_updates": SUCCESSFUL_UPDATES,
                "positive_draws_per_edge": ACHIEVED_POSITIVE_DRAWS_PER_EDGE,
                "multiplicity_is_metadata": True,
            },
            "heldout_query_reserve": {
                "source_round": "0171",
                "receipt": expected_input_signature(R0171_QUERY),
                "selected_before_training": True,
                "reused_byte_exact": True,
            },
            "metric_policy": (
                "report frozen R0160/R0161 quality cells verbatim; execution validity "
                "alone releases the diagnostic readout"
            ),
            "metric_gates_required_for_capability": False,
            "memory_basis": {
                "accepted_round": "0171",
                "train_peak_rss_gib": 19.822986602783203,
                "train_peak_reserved_vram_gib": 1.088,
                "evaluation_peak_rss_gib": 16.121620178222656,
                "evaluation_peak_reserved_vram_gib": 6.8262,
                "host_rss_hard_abort_gib": HOST_RSS_LIMIT_GIB,
                "scaling_argument": (
                    "identical N/graph/model/batches; horizon lies between the two "
                    "same-shape measured cells and changes wall time, not tensors"
                ),
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
        "queue_manifest": prepare_round0184(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
