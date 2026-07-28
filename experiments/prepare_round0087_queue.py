#!/usr/bin/env python3
"""Prepare, but never launch, the CPU/I/O-heavy diverse jina inventory."""
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
from basemap.round0087_inventory import (
    CATALOG_PATH,
    ENGLISH_BUDGETS,
    HELDOUT,
    MULTILINGUAL_BASE,
    MULTILINGUAL_REMAINDER,
    POLISH,
    ROUND_ID,
    TARGET_ROWS,
    discover_inventory_files,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
    _file_inputs,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0087"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0087-2026-07-27.md")


def _frontmatter_status(path: str) -> str | None:
    with open(path, encoding="utf-8") as handle:
        text = handle.read(4096)
    match = re.search(r"(?m)^status:\s*[\"']?([^\s\"']+)", text)
    return match.group(1) if match else None


def prepare_round0087(
    *,
    release_sha: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    if _frontmatter_status(ROUND_FILE) != "issued":
        raise RuntimeError("R0087 remains draft; refuse queue materialization")
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0087 release SHA must be one full commit")
    discovered = discover_inventory_files()
    if not discovered:
        raise RuntimeError("R0087 found no jina-v5-nano inventory files")
    inputs = _dedupe(_file_inputs([
        ROUND_FILE,
        str(CATALOG_PATH),
        *discovered,
    ]))
    queue_root = create_fresh_directory(
        queue_root,
        label="Round 0087 CPU inventory queue",
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    output = os.path.join(artifacts, "jina-diverse-25m-inventory")
    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=0.0,
        execution_authority="autonomous-cpu",
        gpu=False,
    )
    manifest["schema"] = "round0087-diverse-jina-inventory-queue-v1"
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "cpu-io-heavy"
    manifest["required_reviews"] = []
    manifest["capability_dependencies"] = []
    manifest["capabilities_produced"] = [
        "jina-diverse-25m-inventory-v1",
    ]
    manifest["training_performed"] = False
    manifest["scientific_contract"] = {
        "target_rows": TARGET_ROWS,
        "english_budgets": ENGLISH_BUDGETS,
        "multilingual_budget_rule": {
            "languages": 19,
            "sorted_first_languages_receiving_plus_one": (
                MULTILINGUAL_REMAINDER
            ),
            "base_rows": MULTILINGUAL_BASE,
        },
        "heldout_language": POLISH,
        "heldout_english": HELDOUT,
        "selection_order": (
            "registered dataset order, lexicographic shard path, "
            "ascending row"
        ),
        "duplicate_definition": (
            "exact raw fp16 768d row bytes, hash accelerated and "
            "byte-verified across all selected datasets"
        ),
        "capability_requires_exact_25m_selection": True,
        "no_embedding": True,
        "no_graph": True,
        "no_training": True,
        "must_not_overlap_active_gpu_queue": True,
    }
    manifest["inventory_file_count_at_preparation"] = len(discovered)
    manifest["jobs"] = [{
        "id": "inventory_jina_diverse_25m",
        "action": "inventory",
        "handler_module": "experiments.round0087_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(
            artifacts, "inventory_jina_diverse_25m.done.json"
        ),
        "expected_inputs": inputs,
        "p90_wall_s": 14_400.0,
        "node_policy": {
            "gpu_required": False,
            "training_performed": False,
        },
    }]
    manifest["p90_gpu_seconds"] = {"total": 0.0}
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
        "queue_manifest": prepare_round0087(
            release_sha=args.release_sha,
            queue_root=args.queue_root,
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
