#!/usr/bin/env python3
"""Prepare, but never launch, the CPU-only balanced-90M substrate queue."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0049_program import (
    SOURCE_ELIGIBILITY_PATH,
    SOURCE_INT8_PATH,
    SOURCE_SCALES_PATH,
)
from basemap.round0071_substrate import (
    ELIGIBILITY_SUMMARY,
    INTERVALS,
    ROUND_ID,
    ROW_COUNT,
    ROWS_PER_CORPUS,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
    _file_inputs,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0071"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0071-2026-07-27.md")


def _require_issued_round() -> None:
    with open(ROUND_FILE, encoding="utf-8") as handle:
        text = handle.read(4096)
    match = re.search(r"(?m)^status:\s*[\"']?([^\s\"']+)", text)
    if not match or match.group(1) != "issued":
        raise RuntimeError("R0071 remains draft; refuse queue materialization")


def prepare_round0071(
    *,
    release_sha: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    _require_issued_round()
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0071 release SHA must be one full commit")
    queue_root = create_fresh_directory(
        queue_root,
        label="Round 0071 CPU queue",
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    inputs = _dedupe([
        *_file_inputs([
            ROUND_FILE,
            SOURCE_INT8_PATH,
            SOURCE_SCALES_PATH,
            SOURCE_ELIGIBILITY_PATH,
            os.path.join(LAB_ROOT, "review-0033-2026-07-22.md"),
        ]),
    ])
    output = os.path.join(
        artifacts,
        "balanced-90m-int8-substrate",
    )
    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=0.0,
        execution_authority="autonomous-cpu",
        gpu=False,
    )
    manifest["schema"] = "round0071-balanced-90m-substrate-queue-v1"
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "cpu-research-preparation"
    manifest["required_reviews"] = ["0033"]
    manifest["capability_dependencies"] = [
        "minilm-int8-shards-v1",
        "minilm-150m-row-eligibility-v1",
    ]
    manifest["capabilities_produced"] = [
        "minilm-balanced-90m-int8-input-v1",
    ]
    manifest["training_performed"] = False
    manifest["scientific_contract"] = {
        "purpose": (
            "stage the registered 60M-to-120M midpoint without selecting or "
            "launching that rung"
        ),
        "row_count": ROW_COUNT,
        "first_rows_per_corpus": ROWS_PER_CORPUS,
        "global_150m_intervals": [list(value) for value in INTERVALS],
        "expected_eligibility": ELIGIBILITY_SUMMARY,
        "exact_families_recomputed_after_subset_restriction": True,
        "no_candidate_search": True,
        "no_graph": True,
        "no_training": True,
        "no_scale_decision": True,
    }
    manifest["jobs"] = [{
        "id": "build_balanced_90m_int8_substrate",
        "action": "build_substrate",
        "handler_module": "experiments.round0071_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(
            artifacts,
            "build_balanced_90m_int8_substrate.done.json",
        ),
        "expected_inputs": inputs,
        "p90_wall_s": 300.0,
        "node_policy": {
            "gpu_required": False,
            "training_performed": False,
        },
    }]
    manifest["p90_cpu_seconds"] = {
        "build_balanced_90m_int8_substrate": 300.0,
        "total": 300.0,
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    args = parser.parse_args(argv)
    path = prepare_round0071(release_sha=args.release_sha)
    print(json.dumps({"queue_manifest": path}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
