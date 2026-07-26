#!/usr/bin/env python3
"""Prepare, but never launch, the balanced-60M GPU IVF-PQ qualification."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0049_program import INDEX_PATH, validate_substrate_manifest
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
    _file_inputs,
)
from experiments.round0059_nodes import (
    FAISS_WHEEL,
    MAX_PROJECTED_SEARCH_HOURS,
    MIN_ENGINE_OVERLAP,
    MIN_SEARCH_SPEEDUP,
    R0058_SCHEMA,
    _load_sealed_json,
    _selected_nprobe,
)


ROUND_ID = "0059"
ROUND_ROOT = "/data/latent-basemap/runs/round-0059"
RELEASE_ROOT = (
    "/home/enjalot/code/latent-basemap-worktrees/round-0059"
)
ROUND_FILE = os.path.join(
    LAB_ROOT,
    "round-0059-2026-07-26.md",
)
SUBSTRATE_MANIFEST = (
    "/data/latent-basemap/runs/round-0049/queue/artifacts/"
    "balanced-60m-substrate/balanced-60m-substrate-v1.json"
)
NPROBE_RECEIPT = (
    "/data/latent-basemap/runs/round-0058/queue/artifacts/"
    "nprobe-sweep-60m/balanced-60m-nprobe-sweep-v1.json"
)
RUNTIME_SPEC = os.path.join(
    os.path.dirname(__file__),
    "round0059_runtime.json",
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
            f"R0059 requires one issued status; observed {statuses}"
        )


def prepare_round0059(
    *,
    release_sha: str,
    substrate_manifest_sha256: str,
    nprobe_receipt_sha256: str,
    runtime_spec_sha256: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    _require_issued_round()
    substrate = validate_substrate_manifest(
        SUBSTRATE_MANIFEST,
        expected_sha256=substrate_manifest_sha256,
    )
    sweep, _sweep_signature = _load_sealed_json(
        NPROBE_RECEIPT,
        expected_sha256=nprobe_receipt_sha256,
        schema=R0058_SCHEMA,
    )
    selected_nprobe = _selected_nprobe(sweep)
    runtime_signature = expected_input_signature(RUNTIME_SPEC)
    if runtime_signature["sha256"] != runtime_spec_sha256:
        raise RuntimeError("R0059 runtime specification changed")
    outputs = substrate["manifest"]["outputs"]
    queue_root = create_fresh_directory(
        queue_root,
        label="Round 0059 queue",
    )
    artifacts = ensure_data_directory(
        os.path.join(queue_root, "artifacts")
    )
    output = os.path.join(artifacts, "gpu-ivfpq-qualification")
    inputs = _dedupe([
        *_file_inputs([
            ROUND_FILE,
            SUBSTRATE_MANIFEST,
            outputs["int8"]["canonical_path"],
            outputs["scales"]["canonical_path"],
            outputs["eligibility"]["canonical_path"],
            INDEX_PATH,
            NPROBE_RECEIPT,
            RUNTIME_SPEC,
            FAISS_WHEEL,
            os.path.join(LAB_ROOT, "review-0049-2026-07-26.md"),
            os.path.join(LAB_ROOT, "review-0058-2026-07-26.md"),
        ]),
    ])
    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=1.0,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["schema"] = "round0059-gpu-ivfpq-qualification-queue-v1"
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "gpu-research"
    manifest["required_reviews"] = ["0049", "0058"]
    manifest["capability_dependencies"] = [
        "minilm-balanced-60m-input-v1",
        "minilm-balanced-60m-candidate-quality-v1",
        "minilm-balanced-60m-nprobe-calibration-v1",
    ]
    manifest["capabilities_produced"] = [
        "minilm-balanced-60m-gpu-ivfpq-search-qualified-v1",
    ]
    manifest["training_performed"] = False
    manifest["scientific_contract"] = {
        "purpose": (
            "qualify the local RTX 5090 as a semantically equivalent, "
            "quality-certified accelerator for the dominant R0050 search"
        ),
        "candidate_universe": (
            "same balanced 60M intervals with within-subset zero and "
            "duplicate copies physically removed"
        ),
        "nprobe": selected_nprobe,
        "search_width": 128,
        "selected_neighbors": 15,
        "exact_rerank": True,
        "mean_recall_at_15_unambiguous_floor": 0.90,
        "minimum_cpu_gpu_selected_neighbor_overlap": MIN_ENGINE_OVERLAP,
        "minimum_search_speedup": MIN_SEARCH_SPEEDUP,
        "maximum_projected_full_graph_hours": (
            MAX_PROJECTED_SEARCH_HOURS
        ),
        "qualification_shard_rows": 100_000,
        "no_training": True,
        "does_not_modify_r0050": True,
    }
    manifest["jobs"] = [{
        "id": "qualify_balanced_60m_gpu_ivfpq",
        "action": "qualify_gpu_ivfpq",
        "handler_module": "experiments.round0059_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(
            artifacts,
            "qualify_balanced_60m_gpu_ivfpq.done.json",
        ),
        "expected_inputs": inputs,
        "p90_wall_s": 2_400.0,
        "substrate_manifest": SUBSTRATE_MANIFEST,
        "substrate_manifest_sha256": substrate_manifest_sha256,
        "nprobe_receipt": NPROBE_RECEIPT,
        "nprobe_receipt_sha256": nprobe_receipt_sha256,
        "selected_nprobe": selected_nprobe,
        "runtime_spec": RUNTIME_SPEC,
        "runtime_spec_sha256": runtime_spec_sha256,
        "node_policy": {
            "gpu_required": True,
            "training_performed": False,
        },
    }]
    manifest["p90_gpu_seconds"] = {
        "qualify_balanced_60m_gpu_ivfpq": 2_400.0,
        "total": 2_400.0,
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--substrate-manifest-sha256", required=True)
    parser.add_argument("--nprobe-receipt-sha256", required=True)
    parser.add_argument("--runtime-spec-sha256", required=True)
    parser.add_argument(
        "--queue-root",
        default=os.path.join(ROUND_ROOT, "queue"),
    )
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0059(
            release_sha=args.release_sha,
            substrate_manifest_sha256=args.substrate_manifest_sha256,
            nprobe_receipt_sha256=args.nprobe_receipt_sha256,
            runtime_spec_sha256=args.runtime_spec_sha256,
            queue_root=args.queue_root,
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
