#!/usr/bin/env python3
"""Prepare the seed-44 replicate of the 25M diverse-Jina training queue."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0104_training import validate_substrate_manifest
from basemap.round0107_training import (
    BATCH_SIZE,
    DIMENSION,
    PERFORMANCE_WARMUP_UPDATES,
    PIPELINE,
    POSITIVE_RATIO,
    POSITIVE_ROWS_PER_UPDATE,
    RETAINED_ROWS,
    TRAIN_MINIMUM_UPDATES_PER_S,
    TRAIN_WARNING_UPDATES_PER_S,
    load_graph_manifest,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)
from experiments.prepare_round0107_queue import _require_review


ROUND_ID = "0111"
SEED = 44
ROUND_ROOT = "/data/latent-basemap/runs/round-0111"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0111-*.md")
GRAPH_MANIFEST = (
    "/data/latent-basemap/runs/round-0106/queue-attempt-3/artifacts/"
    "canonical-fuzzy-graph/graph-manifest.json"
)
R0106_REVIEW = os.path.join(LAB_ROOT, "review-0106-2026-07-29.md")
R0106_REVIEW_SHA256 = (
    "f00a8391cc47f038993b40337cbe71e07536d305015597ea2e39eed9ca116e1f"
)


def _frontmatter_status(path: str) -> str | None:
    with open(path, encoding="utf-8") as handle:
        text = handle.read(4_096)
    match = re.search(r"(?m)^status:\s*[\"']?([^\s\"']+)", text)
    return match.group(1) if match else None


def _require_issued_round() -> str:
    candidates = [
        path
        for path in sorted(glob.glob(ROUND_FILE_GLOB))
        if _frontmatter_status(path) == "issued"
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"R0111 requires exactly one issued round; found {len(candidates)}"
        )
    return candidates[0]


def _require_successful_r0109_terminal(
    path: str,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise RuntimeError("R0109 terminal receipt bytes changed")
    with open(path, encoding="utf-8") as handle:
        terminal = json.load(handle)
    if (
        terminal.get("schema") != "slim-runner-terminal-v3"
        or terminal.get("round_id") != "0109"
        or terminal.get("verdict") != "succeeded"
        or terminal.get("completed_jobs") != terminal.get("required_jobs")
        or terminal.get("release_checkout_unchanged") is not True
        or terminal.get("queue_manifest_unchanged") is not True
    ):
        raise RuntimeError("R0109 did not reach a clean terminal training run")
    return signature


def prepare_round0111(
    *,
    release_sha: str,
    r0107_review_path: str,
    r0107_review_sha256: str,
    r0109_terminal_path: str,
    r0109_terminal_sha256: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0111 release SHA must be one full commit")
    round_file = _require_issued_round()
    reviews = {
        "0106": _require_review(
            R0106_REVIEW,
            expected_sha256=R0106_REVIEW_SHA256,
            capability="capability:jina-diverse-25m-full768-fuzzy-graph-v1",
        ),
        "0107": _require_review(
            r0107_review_path,
            expected_sha256=r0107_review_sha256,
            capability=(
                "capability:"
                "jina-diverse-25m-full768-trained-map-seed42-v1"
            ),
        ),
    }
    r0109_terminal = _require_successful_r0109_terminal(
        r0109_terminal_path,
        expected_sha256=r0109_terminal_sha256,
    )
    graph_signature = expected_input_signature(GRAPH_MANIFEST)
    graph = load_graph_manifest(
        GRAPH_MANIFEST, expected_sha256=graph_signature["sha256"]
    )
    substrate = validate_substrate_manifest(verify_payloads=False)
    manifest = graph["manifest"]
    updates = int(graph["successful_updates"])

    queue_root = create_fresh_directory(
        queue_root, label="R0111 seed-44 diverse-Jina train queue"
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    inputs = _dedupe([
        expected_input_signature(round_file),
        *reviews.values(),
        r0109_terminal,
        substrate["signature"],
        substrate["payloads"]["int8"],
        substrate["payloads"]["scales"],
        graph_signature,
        manifest["outputs"]["sources"],
        manifest["outputs"]["targets"],
        manifest["outputs"]["weights"],
        manifest["compact_mapping"],
    ])
    output = os.path.join(artifacts, "train-diverse-jina-25m-seed44")
    node_id = "train_diverse_jina_25m_seed44"
    jobs = [{
        "id": node_id,
        "action": "train_diverse_jina_seed44",
        "handler_module": "experiments.round0111_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(artifacts, f"{node_id}.done.json"),
        "expected_inputs": inputs,
        "p90_wall_s": 18_000.0,
        "node_policy": {
            "gpu_required": True,
            "training_performed": True,
        },
        "release_sha": release_sha,
        "graph_manifest": GRAPH_MANIFEST,
        "graph_manifest_sha256": graph_signature["sha256"],
        "graph_release_sha": manifest["release_sha"],
    }]
    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=round_file,
        queue_root=queue_root,
        gpu_hours_cap=5.5,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue["schema"] = "round0111-diverse-jina-seed44-training-queue-v1"
    queue["repo_root"] = RELEASE_ROOT
    queue["queue_class"] = "gpu-research"
    queue["required_reviews"] = ["0106", "0107"]
    queue["capability_dependencies"] = [
        "jina-diverse-25m-full768-fuzzy-graph-v1",
        "jina-diverse-25m-full768-trained-map-seed42-v1",
    ]
    queue["capabilities_produced"] = [
        "jina-diverse-25m-full768-trained-map-seed44-v1"
    ]
    queue["training_performed"] = True
    queue["scientific_contract"] = {
        "role": (
            "third independent seed replicate; no seed-42/43 outcome tuning"
        ),
        "rows": RETAINED_ROWS,
        "dimension": DIMENSION,
        "seed": SEED,
        "r0109_terminal_ordering_receipt": r0109_terminal,
        "graph_manifest": graph_signature,
        "directed_fuzzy_edges": int(manifest["directed_edge_count"]),
        "batch_size": BATCH_SIZE,
        "positive_ratio": POSITIVE_RATIO,
        "positive_rows_per_update": POSITIVE_ROWS_PER_UPDATE,
        "successful_updates": updates,
        "update_rule": "ceil(directed_fuzzy_edges/409)",
        "pipeline": PIPELINE,
        "weighted_sampling": (
            "exact-proportional-with-replacement-via-uniform-envelope-rejection"
        ),
        "minimum_train_updates_per_s": TRAIN_MINIMUM_UPDATES_PER_S,
        "warning_train_updates_per_s": TRAIN_WARNING_UPDATES_PER_S,
        "warmup_updates": PERFORMANCE_WARMUP_UPDATES,
        "evaluation": False,
        "map_decision": False,
        "threshold_tuning": False,
    }
    queue["jobs"] = jobs
    queue["p90_gpu_seconds"] = {node_id: 18_000.0, "total": 18_000.0}
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--r0107-review", required=True)
    parser.add_argument("--r0107-review-sha256", required=True)
    parser.add_argument("--r0109-terminal", required=True)
    parser.add_argument("--r0109-terminal-sha256", required=True)
    parser.add_argument(
        "--queue-root", default=os.path.join(ROUND_ROOT, "queue")
    )
    args = parser.parse_args(argv)
    path = prepare_round0111(
        release_sha=args.release_sha,
        r0107_review_path=args.r0107_review,
        r0107_review_sha256=args.r0107_review_sha256,
        r0109_terminal_path=args.r0109_terminal,
        r0109_terminal_sha256=args.r0109_terminal_sha256,
        queue_root=args.queue_root,
    )
    print(json.dumps({"queue_manifest": path}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
