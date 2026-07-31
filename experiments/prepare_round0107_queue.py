#!/usr/bin/env python3
"""Prepare the 25M diverse-Jina atlas training queue."""
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
    ROUND_ID,
    SEED,
    TRAIN_MINIMUM_UPDATES_PER_S,
    TRAIN_WARNING_UPDATES_PER_S,
    load_graph_manifest,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0107"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0107-*.md")
GRAPH_MANIFEST = (
    "/data/latent-basemap/runs/round-0106/queue-attempt-3/artifacts/"
    "canonical-fuzzy-graph/graph-manifest.json"
)

REVIEW_DEFAULTS = {
    "0037": (
        os.path.join(LAB_ROOT, "review-0037-2026-07-23.md"),
        "8192d5478c63c1e961283c398370619144bfa97828aabcecbbd56ed7fbdb39a1",
        "capability:jina-mrl-seed42-screen-v1",
    ),
    "0038": (
        os.path.join(LAB_ROOT, "review-0038-2026-07-24.md"),
        "fdafdb50286526e6a8a491f4a281c0a95967dc8e4238d8fd270d52b28798cc78",
        "capability:jina-mrl-two-seed-decision-v1",
    ),
    "0103": (
        os.path.join(LAB_ROOT, "review-0103-2026-07-29.md"),
        "c6c4f780c20cc34c7707132581ffaaf8daa8cc7ea9eb1cee3f76e128b6c37a51",
        "capability:jina-diverse-25m-full768-int8-substrate-v1",
    ),
}


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
            f"R0107 requires exactly one issued round; found {len(candidates)}"
        )
    return candidates[0]


def _require_review(
    path: str,
    *,
    expected_sha256: str,
    capability: str,
) -> dict[str, Any]:
    if _frontmatter_status(path) != "accepted":
        raise RuntimeError(f"{path} is not an accepted review")
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise RuntimeError(f"{path} bytes changed")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if capability not in text:
        raise RuntimeError(f"{path} lacks {capability}")
    return signature


def prepare_round0107(
    *,
    release_sha: str,
    r0104_review_path: str,
    r0104_review_sha256: str,
    r0106_review_path: str,
    r0106_review_sha256: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0107 release SHA must be one full commit")
    round_file = _require_issued_round()
    reviews = {
        round_id: _require_review(
            path, expected_sha256=sha, capability=capability
        )
        for round_id, (path, sha, capability) in REVIEW_DEFAULTS.items()
    }
    reviews["0104"] = _require_review(
        r0104_review_path,
        expected_sha256=r0104_review_sha256,
        capability="capability:jina-full768-host-int8-training-validation-v1",
    )
    reviews["0106"] = _require_review(
        r0106_review_path,
        expected_sha256=r0106_review_sha256,
        capability="capability:jina-diverse-25m-full768-fuzzy-graph-v1",
    )
    graph_signature = expected_input_signature(GRAPH_MANIFEST)
    graph = load_graph_manifest(
        GRAPH_MANIFEST, expected_sha256=graph_signature["sha256"]
    )
    substrate = validate_substrate_manifest(verify_payloads=False)
    manifest = graph["manifest"]
    updates = int(graph["successful_updates"])
    queue_root = create_fresh_directory(
        queue_root, label="R0107 diverse-Jina train queue"
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    inputs = _dedupe([
        expected_input_signature(round_file),
        *reviews.values(),
        substrate["signature"],
        substrate["payloads"]["int8"],
        substrate["payloads"]["scales"],
        graph_signature,
        manifest["outputs"]["sources"],
        manifest["outputs"]["targets"],
        manifest["outputs"]["weights"],
        manifest["compact_mapping"],
    ])
    output = os.path.join(artifacts, "train-diverse-jina-25m")
    node_id = "train_diverse_jina_25m"
    jobs = [{
        "id": node_id,
        "action": "train_diverse_jina",
        "handler_module": "experiments.round0107_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(artifacts, f"{node_id}.done.json"),
        "expected_inputs": inputs,
        "p90_wall_s": 28_800.0,
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
        gpu_hours_cap=8.0,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue["schema"] = "round0107-diverse-jina-training-queue-v1"
    queue["repo_root"] = RELEASE_ROOT
    queue["queue_class"] = "gpu-research"
    queue["required_reviews"] = ["0037", "0038", "0103", "0104", "0106"]
    queue["capability_dependencies"] = [
        "jina-mrl-seed42-screen-v1",
        "jina-mrl-two-seed-decision-v1",
        "jina-diverse-25m-full768-int8-substrate-v1",
        "jina-full768-host-int8-training-validation-v1",
        "jina-diverse-25m-full768-fuzzy-graph-v1",
    ]
    queue["capabilities_produced"] = [
        "jina-diverse-25m-full768-trained-map-seed42-v1"
    ]
    queue["training_performed"] = True
    queue["scientific_contract"] = {
        "rows": RETAINED_ROWS,
        "dimension": DIMENSION,
        "seed": SEED,
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
    }
    queue["jobs"] = jobs
    queue["p90_gpu_seconds"] = {node_id: 28_800.0, "total": 28_800.0}
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--r0104-review", required=True)
    parser.add_argument("--r0104-review-sha256", required=True)
    parser.add_argument("--r0106-review", required=True)
    parser.add_argument("--r0106-review-sha256", required=True)
    parser.add_argument(
        "--queue-root", default=os.path.join(ROUND_ROOT, "queue")
    )
    args = parser.parse_args(argv)
    path = prepare_round0107(
        release_sha=args.release_sha,
        r0104_review_path=args.r0104_review,
        r0104_review_sha256=args.r0104_review_sha256,
        r0106_review_path=args.r0106_review,
        r0106_review_sha256=args.r0106_review_sha256,
        queue_root=args.queue_root,
    )
    print(json.dumps({"queue_manifest": path}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
