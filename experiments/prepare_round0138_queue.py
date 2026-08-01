#!/usr/bin/env python3
"""Materialize, but never launch, conditional R0138 sampler bridge."""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import re
import sys
from collections.abc import Mapping
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.graph_validation import data_fingerprint
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0104_training import (
    ROWS,
    preprocessing_stamp,
    source_prefix_proof,
)
from basemap.round0108_evaluation import validate_seal
from basemap.round0138_sampler_bridge import CAPABILITY, ROUND_ID, train_config
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.round0138_nodes import DeviceInventoryFp16Array


ROUND_ROOT = "/data/latent-basemap/runs/round-0138"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0138-*.md")
R0037_SHARED = (
    "/data/latent-basemap/runs/round-0037/queue/artifacts/"
    "shared-reference/receipt.json"
)
R0104_SHARED = (
    "/data/latent-basemap/runs/round-0104/queue-attempt-3/artifacts/"
    "shared/receipt.json"
)
R0134_PANEL = (
    "/data/latent-basemap/runs/round-0134/queue/artifacts/"
    "functional-showdown/functional-showdown.json"
)
R0134_DECISION = (
    "/data/latent-basemap/runs/round-0134/queue/artifacts/decision/decision.json"
)
R0137_DECISION = (
    "/data/latent-basemap/runs/round-0137/queue/artifacts/decision/decision.json"
)

REVIEW_CAPABILITIES = {
    "0037": "jina-mrl-seed42-screen-v1",
    "0103": "jina-diverse-25m-full768-int8-substrate-v1",
    "0104": "jina-full768-host-int8-training-validation-v1",
    "0122": "jina-density-provenance-representation-bridge-v1",
    "0134": "jina-density-functional-showdown-v1",
    "0137": "jina-current-2m-high-recall-graph-bridge-v1",
}

GPU_HOURS_MINIMUM = 1.20
GPU_HOURS_EXPECTED = 1.55
GPU_HOURS_P90 = 2.00
GPU_HOURS_MAXIMUM = 2.50


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _frontmatter(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if not text.startswith("---\n"):
        raise RuntimeError(f"frontmatter missing: {path}")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise RuntimeError(f"frontmatter unterminated: {path}")
    output: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            output[key.strip()] = value.strip().strip("\"'")
    return output


def _frontmatter_list(frontmatter: Mapping[str, str], key: str) -> list[str]:
    value = json.loads(frontmatter.get(key) or "[]")
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise RuntimeError(f"frontmatter {key} is malformed")
    return value


def _issued_round(release_sha: str) -> tuple[str, dict[str, Any]]:
    candidates = [
        path for path in sorted(glob.glob(ROUND_FILE_GLOB))
        if _frontmatter(path).get("status") == "issued"
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"R0138 requires exactly one issued round; found {len(candidates)}")
    if _frontmatter(candidates[0]).get("base_commit") != release_sha:
        raise RuntimeError("R0138 issued base_commit differs from release")
    return candidates[0], expected_input_signature(candidates[0])


def _accepted_review(round_id: str, capability: str) -> list[dict[str, Any]]:
    accepted = []
    for review_path in sorted(glob.glob(os.path.join(LAB_ROOT, f"review-{round_id}-*.md"))):
        frontmatter = _frontmatter(review_path)
        if (
            frontmatter.get("round_id") != round_id
            or frontmatter.get("status") != "accepted"
            or f"capability:{capability}" not in _frontmatter_list(frontmatter, "releases")
        ):
            continue
        result_path = os.path.join(LAB_ROOT, frontmatter.get("result") or "")
        round_path = os.path.join(LAB_ROOT, frontmatter.get("round") or "")
        result = expected_input_signature(result_path)
        issued = expected_input_signature(round_path)
        if (
            result["sha256"] != frontmatter.get("result_sha256")
            or issued["sha256"] != frontmatter.get("round_sha256")
            or _frontmatter(result_path).get("release_commit")
            != frontmatter.get("verified_release_commit")
        ):
            raise RuntimeError(f"Review {round_id} binding changed")
        accepted.append([issued, result, expected_input_signature(review_path)])
    if len(accepted) != 1:
        raise RuntimeError(f"R0138 requires one accepted Review {round_id}; found {len(accepted)}")
    return accepted[0]


def _embedded_signatures(value: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        if set(("canonical_path", "kind", "bytes", "sha256")) <= set(value):
            actual = expected_input_signature(str(value["canonical_path"]))
            if actual != dict(value):
                raise RuntimeError(f"embedded input changed: {value['canonical_path']}")
            output.append(actual)
        else:
            for child in value.values():
                output.extend(_embedded_signatures(child))
    elif isinstance(value, list):
        for child in value:
            output.extend(_embedded_signatures(child))
    return output


def _require_negative_branch() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    r0134_panel = expected_input_signature(R0134_PANEL)
    r0134_decision = expected_input_signature(R0134_DECISION)
    decision_134 = _read_json(R0134_DECISION)
    validate_seal(decision_134, label="R0134 functional decision")
    if (
        decision_134.get("outcome") != "historical-recipe-functionally-better"
        or decision_134.get("fuzzy_graph_or_sampler_bridges_authorized") is not True
        or decision_134.get("panel") != r0134_panel
    ):
        raise RuntimeError("R0134 does not activate the graph/sampler branch")
    r0137_decision = expected_input_signature(R0137_DECISION)
    decision_137 = _read_json(R0137_DECISION)
    validate_seal(decision_137, label="R0137 graph-bridge decision")
    if (
        decision_137.get("outcome") not in {
            "high-recall-graph-regresses-current-control",
            "high-recall-graph-insufficient-to-restore-function",
        }
        or decision_137.get("sampler_bridge_authorized") is not True
    ):
        raise RuntimeError("R0137 does not activate the sampler bridge")
    return r0134_panel, r0134_decision, r0137_decision


def _write_device_manifest(
    *, root: str, graph: Mapping[str, Any], parent_signature: Mapping[str, Any]
) -> dict[str, Any]:
    parent = _read_json(str(parent_signature["canonical_path"]))
    if (
        parent.get("schema") != "graph_manifest.v2"
        or parent.get("graph_sha256") != graph["sha256"]
        or parent.get("n_nodes") != ROWS
    ):
        raise RuntimeError("R0104 parent graph manifest changed")
    source = DeviceInventoryFp16Array()
    records = []
    for ordinal, item in enumerate(source.segments):
        shard = item["shard"]
        records.append({
            "ordinal": ordinal,
            "canonical_path": shard["canonical_path"],
            "bytes": int(shard["bytes"]),
            "sha256": shard["sha256"],
        })
    ids, fingerprint = data_fingerprint(source)
    manifest = copy.deepcopy(parent)
    manifest.update({
        "data_len": ROWS,
        "data_fingerprint": fingerprint,
        "data_fingerprint_n": len(ids),
        "data_shard_records": records,
        "input_preprocessing": preprocessing_stamp("fp16_control"),
        "parent_manifest": dict(parent_signature),
        "verified_by": "round0138-device-sampler-manifest-adapter-v1",
        "adapter_changes_graph_bytes": False,
    })
    path = os.path.join(root, "graph-manifest-device-adapter.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return expected_input_signature(path)


def prepare_round0138(
    *, release_sha: str, queue_root: str = os.path.join(ROUND_ROOT, "queue")
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0138 release SHA must be one full commit")
    round_path, round_signature = _issued_round(release_sha)
    reviews = _dedupe([
        signature
        for round_id, capability in REVIEW_CAPABILITIES.items()
        for signature in _accepted_review(round_id, capability)
    ])
    r0134_panel, r0134_decision, r0137_decision = _require_negative_branch()
    proof = source_prefix_proof()

    r0104_shared_signature = expected_input_signature(R0104_SHARED)
    r0104_shared = _read_json(R0104_SHARED)
    validate_seal(r0104_shared, label="R0104 shared receipt")
    graph = expected_input_signature(r0104_shared["graph"]["canonical_path"])
    parent_manifest = expected_input_signature(
        r0104_shared["graph_manifest"]["canonical_path"]
    )
    if graph != r0104_shared["graph"] or parent_manifest != r0104_shared["graph_manifest"]:
        raise RuntimeError("R0104 graph evidence changed")

    shared_signature = expected_input_signature(R0037_SHARED)
    shared = _read_json(R0037_SHARED)
    validate_seal(shared, label="R0037 functional reference")
    source = expected_input_signature(shared["train"]["canonical_path"])
    high_d_reference = expected_input_signature(shared["high_d_reference"]["canonical_path"])
    query_truth = expected_input_signature(shared["query_truth"]["canonical_path"])
    query_embeddings = expected_input_signature(shared["query_embeddings"]["canonical_path"])
    centroids = {
        key: expected_input_signature(value["canonical_path"])
        for key, value in shared["centroids"].items()
    }

    queue_root = create_fresh_directory(queue_root, label="R0138 sampler bridge queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    device_manifest = _write_device_manifest(
        root=preflight, graph=graph, parent_signature=parent_manifest
    )
    config, config_sha = train_config(
        graph_signature=graph,
        graph_manifest_signature=device_manifest,
        graph_edges=int(r0104_shared["graph_edges"]),
    )
    common = _dedupe([
        round_signature,
        *reviews,
        r0134_panel,
        r0134_decision,
        r0137_decision,
        r0104_shared_signature,
        graph,
        parent_manifest,
        device_manifest,
        shared_signature,
        source,
        high_d_reference,
        query_truth,
        query_embeddings,
        *centroids.values(),
        *_embedded_signatures(proof),
    ])
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    train_output = os.path.join(artifacts, "device-sampler-train")
    panel_output = os.path.join(artifacts, "functional-panel")
    decision_output = os.path.join(artifacts, "decision")
    shared_job = {
        "graph": graph,
        "graph_edges": int(r0104_shared["graph_edges"]),
        "parent_graph_manifest": parent_manifest,
        "device_graph_manifest": device_manifest,
        "source_prefix_proof": proof,
        "production_config": config,
        "production_config_sha256": config_sha,
    }
    jobs = [
        {
            "id": "train_device_sampler",
            "action": "train_device_sampler",
            "handler_module": "experiments.round0138_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [train_output],
            "done_marker": os.path.join(artifacts, "train_device_sampler.done.json"),
            "expected_inputs": common,
            "p90_wall_s": 5_400.0,
            "node_policy": {"gpu_required": True, "training_performed": True},
            **shared_job,
        },
        {
            "id": "score_device_sampler",
            "action": "score_device_sampler",
            "handler_module": "experiments.round0138_nodes",
            "handler_callable": "run_job",
            "deps": ["train_device_sampler"],
            "outputs": [panel_output],
            "done_marker": os.path.join(artifacts, "score_device_sampler.done.json"),
            "expected_inputs": common,
            "p90_wall_s": 600.0,
            "node_policy": {"gpu_required": True, "training_performed": False},
            **shared_job,
            "train_output": train_output,
            "r0134_panel": r0134_panel,
            "source": source,
            "shared_reference_receipt": shared_signature,
            "high_d_reference": high_d_reference,
            "query_truth": query_truth,
            "query_embeddings": query_embeddings,
            "centroids": centroids,
        },
        {
            "id": "decide_device_sampler",
            "action": "decide_device_sampler",
            "handler_module": "experiments.round0138_nodes",
            "handler_callable": "run_job",
            "deps": ["score_device_sampler"],
            "outputs": [decision_output],
            "done_marker": os.path.join(artifacts, "decide_device_sampler.done.json"),
            "expected_inputs": common,
            "p90_wall_s": 30.0,
            "node_policy": {"gpu_required": False, "training_performed": False},
            "panel_output": panel_output,
        },
    ]
    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=round_path,
        queue_root=queue_root,
        gpu_hours_cap=GPU_HOURS_MAXIMUM,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue.update({
        "schema": "round0138-device-sampler-bridge-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-research",
        "required_reviews": list(REVIEW_CAPABILITIES),
        "capability_dependencies": list(REVIEW_CAPABILITIES.values()),
        "capabilities_produced": [CAPABILITY],
        "training_performed": True,
        "gpu_hours": {
            "minimum": GPU_HOURS_MINIMUM,
            "expected": GPU_HOURS_EXPECTED,
            "p90": GPU_HOURS_P90,
            "maximum": GPU_HOURS_MAXIMUM,
        },
        "jobs": jobs,
    })
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=os.path.join(ROUND_ROOT, "queue"))
    args = parser.parse_args()
    print(prepare_round0138(release_sha=args.release_sha, queue_root=args.queue_root))


if __name__ == "__main__":
    main()
