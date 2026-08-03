#!/usr/bin/env python3
"""Prepare, but never launch, the R0180 dose-matched prompted 8M queue."""
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
from basemap import round0113_prompt_contrast as prompt_contract
from basemap.round0160_prompted_seed_family import CAPABILITY as FAMILY_CAPABILITY
from basemap.round0161_prompted_gate_registration import CAPABILITY as GATE_CAPABILITY
from basemap.round0165_frozen_prefix_population import (
    CAPABILITY as POPULATION_CAPABILITY,
    HOST_CAPABILITY as POPULATION_HOST_CAPABILITY,
)
from basemap.round0180_dose_matched_8m import (
    ACHIEVED_POSITIVE_DRAWS_PER_EDGE,
    BASELINE_GRAPH_EDGES,
    BASELINE_SUCCESSFUL_UPDATES,
    CAPABILITY,
    HOST_RSS_LIMIT_GIB,
    RETAINED_ROWS,
    ROUND_ID,
    SUCCESSFUL_UPDATES,
    TARGET_GRAPH_EDGES,
    TARGET_POSITIVE_DRAWS_PER_EDGE,
    scale_train_config,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter
from experiments.prepare_round0166_queue import (
    FAMILY_PATH,
    GATES_PATH,
    POPULATION_PATH,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0180"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0180-2026-08-03.md")
R0171_QUEUE = "/data/latent-basemap/runs/round-0171/queue/queue.json"
R0171_TERMINAL = "/data/latent-basemap/runs/round-0171/queue/runner-terminal.json"
R0171_GRAPH_ROOT = (
    "/data/latent-basemap/runs/round-0171/queue/artifacts/"
    "fuzzy-k50-graph-and-reference"
)
R0171_GRAPH = os.path.join(R0171_GRAPH_ROOT, "graph-manifest.json")
R0171_QUERY_OUTPUT = (
    "/data/latent-basemap/runs/round-0171/queue/artifacts/heldout-query-reserve"
)
R0171_QUERY = os.path.join(R0171_QUERY_OUTPUT, "query-reserve.json")
HANDLER_MODULE = "experiments.round0180_nodes"
QUEUE_SCHEMA = "round0180-prompted-8m-dose-matched-queue-v1"
GPU_HOURS_CAP = 7.5
TRAIN_P90_WALL_S = 21_000.0
EVALUATION_P90_WALL_S = 900.0


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
            f"R0180 requires one {status} {prefix} for R{round_id}; "
            f"found {len(candidates)}"
        )
    return expected_input_signature(candidates[0])


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    if (
        frontmatter.get("status") != "issued"
        or frontmatter.get("base_commit") != release_sha
    ):
        raise RuntimeError("R0180 round is not issued for the exact release")
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
        raise RuntimeError("R0180 release checkout differs from requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
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
        "schema": "round0180-release-cpu-smoke-v1",
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
            "R0180 dose config and dispatch -> shared train -> accounting -> seal "
            "-> checkpoint reload -> transform -> tiny panel"
        ),
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0180 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
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
        config["optimizer"]["successful_positive_lr_updates"]
        != SUCCESSFUL_UPDATES
        or config["optimizer"]["seed"] != 42
        or config["graph"]["k"] != 50
        or config["execution"]["expected_pipeline_stamp"][
            "compact_retained_rows"
        ]
        != RETAINED_ROWS
    ):
        raise RuntimeError("R0180 config smoke changed")
    return prompt_contract.seal({
        "schema": "round0180-config-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "config_sha256": digest,
        "successful_updates": SUCCESSFUL_UPDATES,
        "baseline_graph_edges": BASELINE_GRAPH_EDGES,
        "target_graph_edges": TARGET_GRAPH_EDGES,
        "baseline_successful_updates": BASELINE_SUCCESSFUL_UPDATES,
        "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
        "achieved_positive_draws_per_edge": ACHIEVED_POSITIVE_DRAWS_PER_EDGE,
    })


def _r0171_lineage() -> list[dict[str, Any]]:
    review = _document("review", "0171", status="accepted")
    result = _document("result", "0171", status="complete")
    queue_signature = expected_input_signature(R0171_QUEUE)
    terminal_signature = expected_input_signature(R0171_TERMINAL)
    queue = _read_json(R0171_QUEUE, label="R0171 queue")
    terminal = _read_json(R0171_TERMINAL, label="R0171 terminal")
    graph_signature = expected_input_signature(R0171_GRAPH)
    graph = prompt_contract.read_sealed(
        R0171_GRAPH, label="accepted R0171 graph manifest"
    )
    query_signature = expected_input_signature(R0171_QUERY)
    query = prompt_contract.read_sealed(
        R0171_QUERY, label="accepted R0171 held-out query receipt"
    )
    required = [str(job.get("id") or "") for job in queue.get("jobs") or []]
    if (
        queue.get("round_id") != "0171"
        or terminal.get("round_id") != "0171"
        or terminal.get("verdict") != "succeeded"
        or terminal.get("required_jobs") != required
        or sorted(terminal.get("completed_jobs") or []) != sorted(required)
        or terminal.get("queue_manifest_sha256") != queue_signature["sha256"]
        or terminal.get("queue_manifest_sha256_at_finish")
        != queue_signature["sha256"]
        or terminal.get("queue_manifest_unchanged") is not True
        or terminal.get("release_checkout_unchanged") is not True
        or terminal.get("gpu_wall_accounting_complete") is not True
        or terminal.get("boundary_problems") != []
        or graph.get("schema") != "round0171-prompted-8m-fuzzy-graph-v1"
        or graph.get("round_id") != "0171"
        or int(graph.get("directed_edge_count", -1)) != TARGET_GRAPH_EDGES
        or int(graph.get("retained_rows", -1)) != RETAINED_ROWS
        or ((graph.get("search_qualification") or {}).get("cells") or {})
        .get("64", {})
        .get("passed")
        is not True
        or query.get("schema") != "round0171-prompted-8m-heldout-query-v1"
        or query.get("selected_before_training") is not True
        or (query.get("training_copy_audit") or {}).get(
            "selected_exact_training_identity_disjoint"
        )
        is not True
    ):
        raise RuntimeError("R0171 reusable graph/query lineage changed")

    inherited = []
    for job in queue.get("jobs") or []:
        if job.get("id") in {"train_prompted_8m", "evaluate_prompted_8m"}:
            inherited.extend(dict(value) for value in job.get("expected_inputs") or [])
    direct = [
        review,
        result,
        queue_signature,
        terminal_signature,
        graph_signature,
        dict(graph["graph"]),
        dict(graph["high_d_reference"]),
        *(dict(value) for value in (graph.get("centroids") or {}).values()),
        query_signature,
        dict(query["queries"]),
        dict(query["canonical_rows"]),
        dict(query["source_text_hashes"]),
    ]
    return _dedupe([*inherited, *direct])


def prepare_round0180(
    *, release_sha: str, queue_root: str = QUEUE_ROOT
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0180 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    lineage = _r0171_lineage()
    population_signature = expected_input_signature(POPULATION_PATH)
    family_signature = expected_input_signature(FAMILY_PATH)
    gate_signature = expected_input_signature(GATES_PATH)

    queue_root = create_fresh_directory(queue_root, label="R0180 dose-matched queue")
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
        population_signature,
        family_signature,
        gate_signature,
        expected_input_signature(release_smoke_path),
        expected_input_signature(config_smoke_path),
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    train_output = os.path.join(artifacts, "seed42-dose-matched-train")
    evaluation_output = os.path.join(artifacts, CAPABILITY)
    jobs = [
        {
            "id": "train_prompted_8m_dose_matched",
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
            "id": "evaluate_prompted_8m_dose_matched",
            "action": "evaluate_prompted_8m",
            "handler_module": HANDLER_MODULE,
            "handler_callable": "run_job",
            "deps": ["train_prompted_8m_dose_matched"],
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
        "required_reviews": ["0160", "0161", "0165", "0171"],
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
            "question": (
                "does the prompted 8M rung retain the R0161/R0160 quality stack "
                "when positive-draws-per-edge exposure matches the 2M seed-42 map?"
            ),
            "only_treatment_relative_to_r0171": (
                "successful positive-LR horizon 500000 -> 2026478"
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
                "baseline_graph_edges": BASELINE_GRAPH_EDGES,
                "baseline_successful_updates": BASELINE_SUCCESSFUL_UPDATES,
                "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
                "achieved_positive_draws_per_edge": ACHIEVED_POSITIVE_DRAWS_PER_EDGE,
                "multiplicity_is_metadata": True,
            },
            "heldout_query_reserve": {
                "source_round": "0171",
                "receipt": expected_input_signature(R0171_QUERY),
                "selected_before_training": True,
                "reused_byte_exact": True,
            },
            "native_absolute_gate_metrics": [
                "density_v2",
                "ffr",
                "purity_fidelity_k256",
                "purity_fidelity_k1024",
            ],
            "matched_2m_retention_metrics": [
                "density_v2",
                "ffr",
                "purity_fidelity_k256",
                "purity_fidelity_k1024",
                "projection_ffr",
                "heldout_recall_at_10",
            ],
            "matched_2m_minimum_ratio_to_seed42": 0.97,
            "memory_basis": {
                "accepted_round": "0171",
                "train_peak_rss_gib": 19.822986602783203,
                "train_peak_reserved_vram_gib": 1.088,
                "evaluation_peak_rss_gib": 16.121620178222656,
                "evaluation_peak_reserved_vram_gib": 6.8262,
                "r0180_host_rss_hard_abort_gib": HOST_RSS_LIMIT_GIB,
                "scaling_argument": (
                    "identical N/graph/model/batches; longer horizon changes wall "
                    "time, not resident tensor geometry"
                ),
            },
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
    parser.add_argument("--queue-root", default=QUEUE_ROOT)
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0180(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
