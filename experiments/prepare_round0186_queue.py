#!/usr/bin/env python3
"""Prepare, but never launch, the conditional R0186 U12 graph queue."""
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
from basemap import round0113_prompt_contrast as prompt_contract
from basemap.round0168_prompted_diverse_staging import CAPABILITY as STAGING_CAPABILITY
from basemap.round0180_dose_matched_8m import CAPABILITY as R0180_CAPABILITY
from basemap.round0185_prompted_ood_disjoint_pack import CAPABILITY as R0185_CAPABILITY
from basemap.round0186_prompted_u12_graph import (
    BASELINE_GRAPH_EDGES,
    BASELINE_SUCCESSFUL_UPDATES,
    CAPABILITY,
    GRAPH_K,
    GRAPH_MEAN_RECALL_FLOOR,
    GRAPH_NLIST,
    GRAPH_NPROBE,
    GRAPH_P10_RECALL_FLOOR,
    GRAPH_VECTOR_STORAGE,
    HOST_RSS_LIMIT_GIB,
    REFERENCE_UPDATES_PER_SECOND,
    ROUND_ID,
    ROWS,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _accepted_review, _frontmatter
from experiments.prepare_round0185_queue import _accepted_r0168_review


ROUND_ROOT = "/data/latent-basemap/runs/round-0186"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0186-2026-08-03.md")
STAGING_MANIFEST = (
    "/data/latent-basemap/runs/round-0168/queue/artifacts/"
    "prompted-diverse-u12/prompted-u12-manifest.json"
)
R0185_PACK = (
    "/data/latent-basemap/runs/round-0185/queue/artifacts/"
    f"{R0185_CAPABILITY}/pack.json"
)
GPU_HOURS_CAP = 2.0
GRAPH_P90_WALL_S = 5_400.0


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or frontmatter.get("base_commit") != release_sha
    ):
        raise RuntimeError("R0186 issued round binding changed")
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
        raise RuntimeError("R0186 release checkout differs from requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0186_prompted_u12_graph.py",
        "tests/test_round0169_prompted_diverse.py",
        "tests/test_round0171_prompted_8m.py",
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
        "schema": "round0186-release-cpu-smoke-v1",
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
            "R0186 U12 graph dispatch, four-shard kernel binding, graph qualification, "
            "and exact integer dose-horizon synthesis"
        ),
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0186 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return receipt


def _staging_inputs() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    signature = expected_input_signature(STAGING_MANIFEST)
    staging = prompt_contract.read_sealed(
        STAGING_MANIFEST, label="accepted R0168 prompted staging"
    )
    if (
        staging.get("round_id") != "0168"
        or staging.get("capability") != STAGING_CAPABILITY
        or int(staging.get("rows", -1)) != ROWS
        or not isinstance(staging.get("host_fp16"), Mapping)
        or not isinstance((staging.get("population") or {}).get("mapping"), Mapping)
        or not isinstance((staging.get("duplicate_control") or {}).get("arrays"), Mapping)
    ):
        raise RuntimeError("R0168 staging contract changed")
    inputs = [signature]
    for expected in (
        staging["host_fp16"],
        staging["population"]["mapping"],
        staging["duplicate_control"]["arrays"],
    ):
        observed = expected_input_signature(expected["canonical_path"])
        if observed != expected:
            raise RuntimeError("R0168 staging payload changed")
        inputs.append(observed)
    return signature, inputs


def _r0185_pack() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    signature = expected_input_signature(R0185_PACK)
    pack = prompt_contract.read_sealed(R0185_PACK, label="accepted R0185 probe pack")
    if (
        pack.get("round_id") != "0185"
        or pack.get("passed") is not True
        or pack.get("capabilities") != [R0185_CAPABILITY]
        or int(pack.get("exact_retained_training_family_overlap_count", -1)) != 0
        or pack.get("queries_unchanged") is not True
    ):
        raise RuntimeError("R0185 disjoint probe pack changed")
    payloads = [signature]
    for cell in (pack.get("language_outputs") or {}).values():
        for key in ("corpus_retained_positions", "query_retained_positions"):
            expected = cell.get(key) if isinstance(cell, Mapping) else None
            if not isinstance(expected, Mapping):
                raise RuntimeError("R0185 retained-position binding is missing")
            observed = expected_input_signature(expected["canonical_path"])
            if observed != expected:
                raise RuntimeError("R0185 retained-position bytes changed")
            payloads.append(observed)
    return signature, _dedupe(payloads)


def prepare_round0186(
    *, release_sha: str, queue_root: str = QUEUE_ROOT
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0186 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    r0168_evidence = _accepted_r0168_review()
    r0180_evidence = _accepted_review("0180", R0180_CAPABILITY)
    r0185_evidence = _accepted_review("0185", R0185_CAPABILITY)
    staging_signature, staging_inputs = _staging_inputs()
    pack_signature, pack_inputs = _r0185_pack()

    queue_root = create_fresh_directory(queue_root, label="R0186 GPU queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(smoke_path, _release_cpu_smoke(release_sha), immutable=True)
    smoke_signature = expected_input_signature(smoke_path)
    common = _dedupe([
        round_signature,
        *r0168_evidence,
        *r0180_evidence,
        *r0185_evidence,
        *staging_inputs,
        *pack_inputs,
        smoke_signature,
    ])
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    graph_output = os.path.join(artifacts, "fuzzy-k50-graph-and-reference")
    graph_manifest = os.path.join(graph_output, "graph-manifest.json")
    dose_output = os.path.join(artifacts, CAPABILITY)
    jobs = [
        {
            "id": "build_prompted_u12_graph_and_reference",
            "action": "build_graph_and_reference",
            "handler_module": "experiments.round0186_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [graph_output],
            "done_marker": os.path.join(artifacts, "graph-reference.done.json"),
            "expected_inputs": common,
            "p90_wall_s": GRAPH_P90_WALL_S,
            "staging_manifest": staging_signature,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
                "cpu_heavy": True,
            },
        },
        {
            "id": "derive_exact_dose_plan",
            "action": "derive_dose_plan",
            "handler_module": "experiments.round0186_nodes",
            "handler_callable": "run_job",
            "deps": ["build_prompted_u12_graph_and_reference"],
            "outputs": [dose_output],
            "done_marker": os.path.join(artifacts, "dose-plan.done.json"),
            "expected_inputs": common,
            "p90_wall_s": 60.0,
            "graph_manifest": graph_manifest,
            "node_policy": {
                "gpu_required": False,
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
        "schema": "round0186-prompted-u12-graph-dose-plan-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-research",
        "required_reviews": ["0168", "0180", "0185"],
        "capability_dependencies": [
            STAGING_CAPABILITY,
            R0180_CAPABILITY,
            R0185_CAPABILITY,
        ],
        "capabilities_produced": [CAPABILITY],
        "training_performed": False,
        "jobs": jobs,
        "p90_gpu_seconds": {
            "build_prompted_u12_graph_and_reference": GRAPH_P90_WALL_S,
            "total": GRAPH_P90_WALL_S,
        },
        "scientific_contract": {
            "operation": (
                "build and qualify the frozen R0169 fuzzy-k50 four-shard fp32 graph; "
                "derive, but do not execute, the exact matched-dose train horizon"
            ),
            "population_rows": ROWS,
            "population": "exact accepted R0132 U12 compact order via R0168",
            "embedding_convention": "Document: ",
            "probe_pack_branch_evidence": pack_signature,
            "graph": {
                "k": GRAPH_K,
                "nlist": GRAPH_NLIST,
                "nprobe": GRAPH_NPROBE,
                "mean_recall_floor": GRAPH_MEAN_RECALL_FLOOR,
                "p10_recall_floor": GRAPH_P10_RECALL_FLOOR,
                "vector_storage": GRAPH_VECTOR_STORAGE,
                "four_row_disjoint_shards": True,
            },
            "dose_formula": {
                "baseline_round": "0115",
                "baseline_graph_edges": BASELINE_GRAPH_EDGES,
                "baseline_successful_updates": BASELINE_SUCCESSFUL_UPDATES,
                "formula": "ceil(500000 * observed_directed_edges / 148801612)",
                "reference_updates_per_second": REFERENCE_UPDATES_PER_SECOND,
            },
            "map_training_performed": False,
            "next_round_policy": (
                "use the sealed observed edge count and runtime projection to register "
                "a <=8 GPU-hour train/evaluation queue or explicit continuation split"
            ),
            "memory_basis": {
                "accepted_round": "0171",
                "r0171_rows": 7_952_419,
                "r0171_shards": 2,
                "r0171_graph_wall_s": 932.5969,
                "r0171_peak_host_rss_gib": 49.29814910888672,
                "r0186_rows": ROWS,
                "r0186_shards": 4,
                "host_rss_abort_gib": HOST_RSS_LIMIT_GIB,
                "scaling_argument": (
                    "same fp32 IVF8192/k50 exact-merge law; 1.57x rows and four rather "
                    "than two search shards justify 5400s p90 and inherited 90GiB abort"
                ),
            },
            "release_cpu_smoke": smoke_signature,
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
        "queue_manifest": prepare_round0186(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
