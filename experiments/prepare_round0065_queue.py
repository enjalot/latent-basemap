#!/usr/bin/env python3
"""Prepare, but never launch, the two CPU-only R0065 substrates."""
from __future__ import annotations

import argparse
import json
import os
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
from basemap.round0065_substrates import ROUND_ID, SUBSETS
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
    _file_inputs,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0065"
RELEASE_ROOT = (
    "/home/enjalot/code/latent-basemap-worktrees/round-0065"
)
ROUND_FILE = os.path.join(
    LAB_ROOT,
    "round-0065-2026-07-26.md",
)


def _require_issued_round() -> None:
    with open(ROUND_FILE, encoding="utf-8") as handle:
        lines = handle.readlines()
    statuses: list[str] = []
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, separator, value = line.partition(":")
        if separator and key.strip() == "status":
            statuses.append(value.strip().strip("\"'"))
    if statuses != ["issued"]:
        raise RuntimeError(
            f"R0065 requires one issued status; observed {statuses}"
        )


def prepare_round0065(
    *,
    release_sha: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    _require_issued_round()
    if len(release_sha) != 40:
        raise ValueError("R0065 release SHA must be one full commit")
    queue_root = create_fresh_directory(
        queue_root,
        label="Round 0065 CPU queue",
    )
    artifacts = ensure_data_directory(
        os.path.join(queue_root, "artifacts")
    )
    inputs = _dedupe([
        *_file_inputs([
            ROUND_FILE,
            SOURCE_INT8_PATH,
            SOURCE_SCALES_PATH,
            SOURCE_ELIGIBILITY_PATH,
            os.path.join(LAB_ROOT, "review-0033-2026-07-22.md"),
        ]),
    ])
    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=0.0,
        execution_authority="autonomous-cpu",
        gpu=False,
    )
    manifest["schema"] = "round0065-decision-ready-substrates-queue-v1"
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "cpu-research-preparation"
    manifest["required_reviews"] = ["0033"]
    manifest["capability_dependencies"] = [
        "minilm-int8-shards-v1",
        "minilm-150m-row-eligibility-v1",
    ]
    manifest["capabilities_produced"] = [
        "minilm-balanced-45m-int8-input-v1",
        "minilm-balanced-120m-int8-input-v1",
    ]
    manifest["training_performed"] = False
    manifest["scientific_contract"] = {
        "purpose": (
            "remove the mechanical implementation wall after the R0064 "
            "scale decision without choosing that decision in advance"
        ),
        "subsets": {
            tier: {
                "row_count": spec["row_count"],
                "first_rows_per_corpus": spec["first_rows_per_corpus"],
                "global_150m_intervals": [
                    list(value) for value in spec["intervals"]
                ],
                "expected_eligibility": spec["eligibility_summary"],
            }
            for tier, spec in SUBSETS.items()
        },
        "exact_families_recomputed_after_subset_restriction": True,
        "no_candidate_search": True,
        "no_graph": True,
        "no_training": True,
        "no_scale_decision": True,
    }
    common = {
        "action": "build_substrate",
        "handler_module": "experiments.round0065_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "expected_inputs": inputs,
        "node_policy": {
            "gpu_required": False,
            "training_performed": False,
        },
    }
    manifest["jobs"] = []
    for tier, p90_wall_s in (("45m", 1_200.0), ("120m", 2_400.0)):
        output = os.path.join(
            artifacts,
            f"balanced-{tier}-int8-substrate",
        )
        manifest["jobs"].append({
            **common,
            "id": f"build_balanced_{tier}_int8_substrate",
            "tier": tier,
            "outputs": [output],
            "done_marker": os.path.join(
                artifacts,
                f"build_balanced_{tier}_int8_substrate.done.json",
            ),
            "p90_wall_s": p90_wall_s,
        })
    manifest["p90_cpu_seconds"] = {
        "build_balanced_45m_int8_substrate": 1_200.0,
        "build_balanced_120m_int8_substrate": 2_400.0,
        "total": 3_600.0,
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument(
        "--queue-root",
        default=os.path.join(ROUND_ROOT, "queue"),
    )
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0065(
            release_sha=args.release_sha,
            queue_root=args.queue_root,
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
